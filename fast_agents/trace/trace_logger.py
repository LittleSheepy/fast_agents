"""TraceLogger —— 全链路记录器（路线图 G1 初版）。

借鉴 LangChain CallbackManager / OpenTelemetry Tracer / OpenAI Agents SDK Tracing
的核心概念（Trace / Span / 父子关系 / 钩子），但只保留最小可跑闭环：

- **默认关闭**：不构造 TraceLogger 或不传给 LLM/Agent，行为与改动前完全一致；
- **一条 Trace = 一次 run**，一个 Trace 下多个 Span（llm/tool/decision/agent）；
- **最小字段集**：时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id + 推荐补充：
  trace_id / span_id / parent_id / name / status / error / metadata；
- **默认 JsonlStore 落盘**（路径可配），支持 ``backend="sqlite"`` 一行切换 SQLiteStore，
  或通过 ``path`` 后缀（.db/.sqlite/.sqlite3）自动识别后端；存储层通过 TraceStoreBase 抽象，
  预留远端上报扩展位；
- **零侵入接入**：
  - 工具打点：``tool_registry.add_hook(tracer.tool_hook)`` 或 ``attach_to_tool_registry(reg)``；
  - LLM / Agent：构造时传 ``tracer=...`` 即可自动打点（可选参数，默认 None 行为不变）。
"""
from __future__ import annotations

import contextvars
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

from .jsonl_store import MEMORY_PATH, JsonlStore
from .models import SpanKind, SpanRecord, SpanStatus, _new_id, _now_ts
from .sqlite_store import SQLiteStore
from .store_base import TraceStoreBase


class TraceLogger:
    """全链路 Trace 记录器。

    Args:
        store: TraceStoreBase 后端；显式传入时优先使用，忽略 path/backend。
        path: 当 store=None 时，存储文件路径；默认 ``":memory:"``（仅内存，不落盘）；
              传文件路径时，根据后缀自动选择后端：.jsonl/.jsonl/.ndjson → JsonlStore，
              .db/.sqlite/.sqlite3 → SQLiteStore；也可通过 backend 参数显式指定。
        backend: 显式指定存储后端：``"jsonl"`` 或 ``"sqlite"``；不传时根据 path 后缀自动识别，
                 后缀无法识别时默认 jsonl（保持向后兼容）。
        session_id: 会话 id；不传则自动生成；可通过 ``set_session_id`` 随时切换。
        enabled: 是否实际打点；False 时所有方法都是空操作（与不传 tracer 等价）。

    状态隔离说明：早期版本使用模块级 ``contextvars.ContextVar``，
    会导致多个 TraceLogger 实例互相污染父链；现改为每个实例自持一对 ContextVar，
    同时兼顾：(1) 实例级隔离（主/子 Agent 各自 tracer 互不串扰）；(2) 跨协程
    传播（asyncio.Task 自动拷贝上下文，为 G2 异步并发预留扩展）。
    """

    def __init__(
        self,
        store: Optional[TraceStoreBase] = None,
        *,
        path: Optional[str] = None,
        backend: Optional[str] = None,
        session_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._lock = threading.Lock()
        # 存储后端
        if store is not None:
            # 显式传入 store 实例优先
            self._store = store
        else:
            # 未显式传 store：根据 backend 参数或 path 后缀自动选择后端
            default_path = path if path is not None else MEMORY_PATH
            # 自动识别后缀：.db / .sqlite / .sqlite3 → sqlite；其它 → jsonl
            _SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
            backend_lower = backend.lower() if backend is not None else None
            use_sqlite = (backend_lower == "sqlite")
            if not use_sqlite and backend_lower is None and default_path != MEMORY_PATH:
                # 未显式指定 backend，根据文件后缀自动判断
                try:
                    ext = os.path.splitext(default_path)[1].lower()
                    if ext in _SQLITE_SUFFIXES:
                        use_sqlite = True
                except Exception:
                    pass
            if use_sqlite:
                self._store = SQLiteStore(default_path)
            else:
                self._store = JsonlStore(default_path, append=True)
        # 会话 id
        self._session_id = session_id or f"sess_{_new_id()[:12]}"
        # 实例级 ContextVar：每个 tracer 自持一份栈，(1) 不与其它实例串；
        # (2) 天然随 asyncio.Task 传播（G2 异步直接可用，无需返工）。
        # 栈用 tuple 表示（ContextVar 推荐不可变类型），空栈为 ()。
        self._trace_stack_var: contextvars.ContextVar[tuple] = contextvars.ContextVar(
            f"trace_stack_{id(self)}", default=()
        )
        self._span_stack_var: contextvars.ContextVar[tuple] = contextvars.ContextVar(
            f"span_stack_{id(self)}", default=()
        )

    # ---------- 配置 ----------
    @property
    def session_id(self) -> str:
        return self._session_id

    def set_session_id(self, session_id: str) -> None:
        """切换/设置 session_id。"""
        self._session_id = session_id

    @property
    def store(self) -> TraceStoreBase:
        """获取底层存储后端（便于读/回放/close）。"""
        return self._store

    # ---------- Trace 生命周期 ----------
    def start_trace(self, name: str = "root", session_id: Optional[str] = None) -> str:
        """开启一条新 Trace，返回 trace_id；新 Trace 开始后调用 start_span 会作为根 Span。"""
        trace_id = f"trace_{_new_id()}"
        self._trace_stack_var.set(self._trace_stack_var.get() + (trace_id,))
        if session_id is not None:
            self._session_id = session_id
        return trace_id

    def end_trace(self) -> None:
        """结束当前 Trace（仅弹栈；实际写入由每个 Span.finish 完成）。"""
        stack = self._trace_stack_var.get()
        if stack:
            self._trace_stack_var.set(stack[:-1])

    @contextmanager
    def trace(self, name: str = "root", session_id: Optional[str] = None) -> Iterator[str]:
        """with 语法开启/结束一条 Trace，yield trace_id。"""
        tid = self.start_trace(name=name, session_id=session_id)
        try:
            yield tid
        finally:
            self.end_trace()

    # ---------- 内部栈访问辅助 ----------
    def _stacks(self) -> tuple[tuple, tuple]:
        """返回当前协程上下文里的 (trace_stack, span_stack)，均为 tuple（不可变）。
        每个 TraceLogger 实例 + 每个 asyncio.Task / 线程各持一份，天然隔离。"""
        return self._trace_stack_var.get(), self._span_stack_var.get()

    def _current_trace_id(self) -> str:
        t_stack, _ = self._stacks()
        if t_stack:
            return t_stack[-1]
        # 懒启动一条默认 trace
        return self.start_trace(name="default")

    def _current_parent_id(self) -> Optional[str]:
        _, s_stack = self._stacks()
        return s_stack[-1] if s_stack else None

    def _push_span(self, span_id: str) -> None:
        self._span_stack_var.set(self._span_stack_var.get() + (span_id,))

    def _pop_span_if_top(self, span_id: str) -> None:
        """若当前栈顶 span 与传入一致，则弹栈；否则说明用户手动操作了栈，不做调整。"""
        stack = self._span_stack_var.get()
        if stack and stack[-1] == span_id:
            self._span_stack_var.set(stack[:-1])

    def start_span(
        self,
        name: str,
        kind: SpanKind | str = SpanKind.CUSTOM,
        *,
        input: Any = None,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> SpanRecord:
        """创建并返回一个 SpanRecord（未 finish），同时压入栈。"""
        if not self.enabled:
            # 返回一个占位 Span，让调用方可以无脑 .finish()，但不会写入
            placeholder = SpanRecord(
                trace_id="disabled",
                name=name,
                kind=SpanKind.CUSTOM,
                session_id=session_id or self._session_id,
                input=input,
                metadata=dict(metadata or {}),
            )
            return placeholder

        if isinstance(kind, str):
            try:
                kind = SpanKind(kind)
            except ValueError:
                kind = SpanKind.CUSTOM

        # 懒启动默认 trace
        trace_id = self._current_trace_id()

        span = SpanRecord(
            trace_id=trace_id,
            parent_id=self._current_parent_id(),
            session_id=session_id or self._session_id,
            name=name,
            kind=kind,
            input=input,
            metadata=dict(metadata or {}),
        )
        self._push_span(span.span_id)
        return span

    def end_span(
        self,
        span: SpanRecord,
        output: Any = None,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """结束一个 Span 并写入 store。

        调用前可以手动修改 span.input/metadata；error 非空时自动置 status=ERROR。
        end_time 为可选精确结束时刻（秒），默认 None 时取当前时刻（行为不变）。
        """
        if not self.enabled:
            return
        if error is not None and status == SpanStatus.SUCCESS:
            status = SpanStatus.ERROR
        span.finish(output=output, status=status, error=error, metadata=metadata, end_time=end_time)
        # 弹栈
        self._pop_span_if_top(span.span_id)
        # 写入 store（底层已保证异常不抛）
        try:
            self._store.write(span)
        except Exception:
            pass

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind | str = SpanKind.CUSTOM,
        *,
        input: Any = None,
        metadata: Optional[dict] = None,
    ) -> Iterator[SpanRecord]:
        """with 语法自动 start/finish 一个 Span，异常时自动 error。"""
        s = self.start_span(name, kind, input=input, metadata=metadata)
        try:
            yield s
        except Exception as e:
            self.end_span(s, output=None, error=f"{type(e).__name__}: {e}")
            raise
        else:
            # 若调用方已手动 finish，则 end_time 已有值，这里直接写入
            if s.end_time is None:
                self.end_span(s, output=s.output, status=s.status, error=s.error, metadata=s.metadata)
            else:
                # 已经 finish 过，直接写
                if self.enabled:
                    try:
                        self._store.write(s)
                    except Exception:
                        pass
                self._pop_span_if_top(s.span_id)

    # ---------- 工具钩子（对齐 ToolRegistry.add_hook 签名） ----------
    def tool_hook(self, name: str, tool_args: dict, result: Any) -> None:
        """传给 ``ToolRegistry.add_hook`` 的回调。

        钩子签名：``hook(tool_name: str, tool_args: dict, result: ToolResult)``；
        内部会自动开启一个 tool Span 记录输入/输出/耗时/状态，不抛异常。

        计时说明：ToolRegistry 的钩子是工具执行**完成后**才调用的，
        「后钩子」里拿不到真实开始时刻。工具的真实耗时由 ``tool_base`` 写进
        ``ToolResult.metadata["duration_ms"]``（int 毫秒），这里读取该值，
        并在钩子入口**只取一次**当前时间戳：既作为 tool Span 的 ``end_time``，
        又用「该时刻 − duration_ms」回推 ``start_time``，使 ``duration_ms``
        严格等于工具自报耗时、时间区间与父子包含关系都正确；若工具未上报耗时，
        则退化为一次近零耗时记录，并在 metadata 标记 ``duration_source``。
        """
        if not self.enabled:
            return
        duration_ms = None
        output_data = None
        status = SpanStatus.SUCCESS
        error_msg = None
        tool_status_str = None
        try:
            # ToolResult 有 status/error/output/metadata 属性
            status_val = getattr(result, "status", None)
            # ToolStatus.SUCCESS/ERROR/TRUNCATED
            if hasattr(status_val, "value"):
                tool_status_str = str(status_val.value)
            elif status_val is not None:
                tool_status_str = str(status_val)
            else:
                tool_status_str = "success"
            if tool_status_str == "error":
                status = SpanStatus.ERROR
            err = getattr(result, "error", None)
            if err:
                error_msg = str(err)
                status = SpanStatus.ERROR
            output_data = getattr(result, "output", str(result) if result is not None else None)
            # 真实耗时在 metadata 里（tool_base 写入），不是 ToolResult 的顶层属性
            result_meta = getattr(result, "metadata", None)
            if isinstance(result_meta, dict):
                duration_ms = result_meta.get("duration_ms")
        except Exception:
            output_data = str(result) if result is not None else None

        # 计时修正：ToolRegistry 的后钩子只在工具执行完成**之后**触发，
        # 因此钩子入口时刻 ≈ 工具结束时刻。全函数只在此取一次时间戳 hook_end_ts，
        # 既作为 tool Span 的 end_time，又据此回推 start_time = end_time - duration_ms，
        # 使 duration_ms 严格等于工具自报值，不再叠加钩子自身处理开销（消除计时漂移）。
        hook_end_ts = _now_ts()
        start_time = hook_end_ts
        if duration_ms is not None:
            try:
                start_time = hook_end_ts - float(duration_ms) / 1000.0
            except (TypeError, ValueError):
                duration_ms = None

        span_metadata = {"tool_name": name}
        if tool_status_str is not None:
            span_metadata["tool_status"] = tool_status_str
        if duration_ms is None:
            # 工具未上报真实耗时：退化为近零耗时记录，并显式标记来源
            span_metadata["duration_source"] = "unavailable"

        # 直接构造并 finish 一条 Span（后钩子触发时上层 decision Span 仍是当前
        # 活跃 Span，所以这里 start_span 出来的 tool Span 会自然成为其子 Span）
        s = self.start_span(
            name=f"tool.{name}",
            kind=SpanKind.TOOL,
            input=tool_args if isinstance(tool_args, dict) else {"raw": str(tool_args)},
            metadata=span_metadata,
        )
        if duration_ms is not None:
            # 覆盖 start_span 内部生成的 start_time，并显式传入精确 end_time，
            # 使 end_time - start_time ≡ duration_ms（严格等于工具自报耗时，零漂移）。
            s.start_time = start_time
            self.end_span(s, output=output_data, status=status, error=error_msg, end_time=hook_end_ts)
        else:
            self.end_span(s, output=output_data, status=status, error=error_msg)

    def attach_to_tool_registry(self, registry: Any) -> None:
        """把 ``self.tool_hook`` 注册到给定 ToolRegistry；一行接入工具打点。"""
        if not self.enabled:
            return
        try:
            registry.add_hook(self.tool_hook)
        except Exception:
            pass

    # ---------- 读取 / 回放 ----------
    def read_spans(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SpanRecord]:
        """从底层 store 读取 Span 列表（用于回放 / 分析）。"""
        return self._store.read(session_id=session_id, trace_id=trace_id, kind=kind, limit=limit)

    def replay(
        self,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[dict]:
        """回放一条 Trace 或一次会话，按时间顺序返回可打印的 dict 列表。

        默认回放当前活跃 trace；若没有则返回空列表。
        """
        tid = trace_id
        if tid is None:
            t_stack, _ = self._stacks()
            tid = t_stack[0] if t_stack else None
        spans = self._store.read(session_id=session_id, trace_id=tid)
        spans.sort(key=lambda s: (s.start_time, s.span_id))
        return [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "kind": s.kind.value if isinstance(s.kind, SpanKind) else s.kind,
                "name": s.name,
                "status": s.status.value if isinstance(s.status, SpanStatus) else s.status,
                "duration_ms": round(s.duration_ms, 2) if s.duration_ms is not None else None,
                "error": s.error,
                "input": s.input,
                "output": s.output,
                "metadata": s.metadata,
            }
            for s in spans
        ]

    # ---------- 生命周期 ----------
    def flush(self) -> None:
        """刷盘。"""
        try:
            self._store.flush()
        except Exception:
            pass

    def close(self) -> None:
        """关闭底层 store；多次调用安全。"""
        try:
            self._store.close()
        except Exception:
            pass

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.flush()
        self.close()
        return False


__all__ = ["TraceLogger", "SpanKind", "SpanStatus", "SpanRecord"]
