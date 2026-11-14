"""TraceLogger —— 全链路记录器（路线图 G1 初版）。

借鉴 LangChain CallbackManager / OpenTelemetry Tracer / OpenAI Agents SDK Tracing
的核心概念（Trace / Span / 父子关系 / 钩子），但只保留最小可跑闭环：

- **默认关闭**：不构造 TraceLogger 或不传给 LLM/Agent，行为与改动前完全一致；
- **一条 Trace = 一次 run**，一个 Trace 下多个 Span（llm/tool/decision/agent）；
- **最小字段集**：时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id + 推荐补充：
  trace_id / span_id / parent_id / name / status / error / metadata；
- **store 单一入口**：``store`` 可传 ``TraceStoreBase`` 实例、路径字符串或 ``None``；
  字符串 ``":memory:"``（或 ``None``）用纯内存 ``InMemoryTraceStore``（默认），其它路径按后缀
  （.db/.sqlite/.sqlite3）自动识别 SQLiteStore、否则 JsonlStore；
  存储层通过 TraceStoreBase 抽象，预留远端上报扩展位；
- **零侵入接入**：
  - 工具打点：``tool_registry.add_hook(tracer.tool_hook)`` 或 ``attach_to_tool_registry(reg)``；
  - LLM / Agent：构造时传 ``tracer=...`` 即可自动打点（可选参数，默认 None 行为不变）。
"""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Union

from .in_memory_trace_store import InMemoryTraceStore
from .jsonl_store import JsonlStore
from .models import SpanKind, SpanRecord, SpanStatus, _new_id, _now_ts
from .sqlite_store import SQLiteStore
from .store_base import TraceStoreBase

# 内存模式占位路径（与 SQLiteClient ``":memory:"`` 约定对齐）
MEMORY_PATH = ":memory:"

# 文件后缀 → SQLiteStore 的自动识别集合（其余后缀默认 JsonlStore）
_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _resolve_store(
    store: Optional[Union[str, TraceStoreBase]]
) -> TraceStoreBase:
    """把 ``store`` 参数解析成具体的存储后端实例。

    - ``store`` 为 ``TraceStoreBase``：直接使用（优先级最高）；
    - ``store`` 为 ``":memory:"`` 或 ``None``：内存模式，默认返回 ``InMemoryTraceStore``；
    - ``store`` 为其它字符串：视为文件路径，
      否则按后缀自动选择（.db/.sqlite/.sqlite3 → SQLiteStore，其它 → JsonlStore）。
    """
    if isinstance(store, TraceStoreBase):
        return store
    target = store if store is not None else MEMORY_PATH
    if not isinstance(target, str):
        raise TypeError(
            "store 必须是 TraceStoreBase 实例、路径字符串或 None，"
            f"实际类型: {type(store).__name__}"
        )
    if target == MEMORY_PATH:
        return InMemoryTraceStore()
    # 文件路径模式：按后缀自动识别
    use_sqlite = False
    try:
        if os.path.splitext(target)[1].lower() in _SQLITE_SUFFIXES:
            use_sqlite = True
    except Exception:
        pass
    return SQLiteStore(target) if use_sqlite else JsonlStore(target, append=True)


class TraceLogger:
    """全链路 Trace 记录器。

    Args:
        store: 存储后端，支持三种形式：
               (1) ``TraceStoreBase`` 实例——优先级最高，直接使用；
               (2) 路径字符串——``":memory:"`` 用纯内存 ``InMemoryTraceStore``，
                   其它路径按后缀自动选择后端（``.db/.sqlite/.sqlite3`` → SQLiteStore，
                   其它 → JsonlStore）；
               (3) ``None``——等价于 ``":memory:"``（默认纯内存，不落盘）。
        session_id: 会话 id；不传则自动生成；可通过 ``set_session_id`` 随时切换。
        enabled: 是否实际打点；False 时所有方法都是空操作（与不传 tracer 等价）。
        input_preview_len: 输入预览的截断长度（字符）。``None``（默认）时沿用各接入点的
            原有默认值（如 Agent 输入 500），行为与改动前一致；显式配置后按该长度截断；
            ``<= 0`` 表示不截断（完整记录）。
        output_preview_len: 输出预览的截断长度（字符）。``None``（默认）时沿用各接入点的
            原有默认值（如 Agent 输出 300、LLM 输出 200）；显式配置后按该长度截断；
            ``<= 0`` 表示不截断（完整记录）。

    状态隔离说明：早期版本使用模块级 ``contextvars.ContextVar``，
    会导致多个 TraceLogger 实例互相污染父链；现改为每个实例自持一对 ContextVar，
    同时兼顾：(1) 实例级隔离（主/子 Agent 各自 tracer 互不串扰）；(2) 跨协程
    传播（asyncio.Task 自动拷贝上下文，为 G2 异步并发预留扩展）。
    """

    def __init__(
        self,
        store: Optional[Union[str, TraceStoreBase]] = None,
        *,
        session_id: Optional[str] = None,
        enabled: bool = True,
        input_preview_len: Optional[int] = None,
        output_preview_len: Optional[int] = None,
    ):
        self.enabled = enabled
        self._lock = threading.Lock()
        # 存储后端：统一由 store 入口解析（TraceStoreBase 实例 / 路径字符串 / None）
        self._store = _resolve_store(store)
        # 会话 id
        self._session_id = session_id or f"sess_{_new_id()[:12]}"
        # 输入/输出预览截断长度（可选配置）：None 表示沿用各接入点的原有默认长度，
        # 保证不开配置时行为与改动前逐字节一致；<=0 表示不截断。
        self.input_preview_len = input_preview_len
        self.output_preview_len = output_preview_len
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

    def truncate_preview(
        self,
        text: Optional[str],
        default_len: int,
        *,
        is_input: bool = False,
    ) -> Optional[str]:
        """按配置截断「输入/输出预览」文本，供打点接入层统一调用。

        - 未配置 ``input_preview_len`` / ``output_preview_len``（注意是 ``None`` 而非 0）时，
          使用调用点传入的 ``default_len``，与改动前逐字节一致
          （如 Agent 输入 500 / 输出 300、LLM 输出 200）；
        - 显式配置后按配置长度截断；配置 ``<= 0`` 表示**不截断**（完整记录）；
        - ``None`` / 非字符串原样返回；任何异常都兜底返回原文，绝不阻塞主流程（R6）。
        """
        limit = self.input_preview_len if is_input else self.output_preview_len
        if limit is None:
            limit = default_len
        if text is None or not isinstance(text, str):
            return text
        try:
            limit = int(limit)
            if limit <= 0:
                return text
            return text[:limit]
        except Exception:
            return text

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
        # 记录单调时钟起点：finish 时优先用它计算耗时（抗系统墙钟跳变）。
        # 仅「正常创建」的 Span 打此标记；直接构造的 SpanRecord 不带标记，
        # 耗时口径保持旧的墙钟差值（零破坏）。
        try:
            span._mono_start = time.perf_counter()
        except Exception:
            pass
        self._push_span(span.span_id)
        return span

    def end_span(
        self,
        span: SpanRecord,
        output: Any = None,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """结束一个 Span 并写入 store。

        调用前可以手动修改 span.input/metadata；error 非空时自动置 status=ERROR。
        end_time 为可选精确结束时刻（秒），默认 None 时取当前时刻（行为不变）。
        error_type 为可选异常分类；显式传入优先，否则从 error 自动推断。

        幂等保证：同一 Span 重复调用 end_span 会直接返回，不会重复写入存储，
        也不会覆盖第一次落库时的结果。
        """
        if not self.enabled or span._persisted:
            return
        # 仅当参数显式传入时才覆盖已有值，保留用户手动提前设置的字段
        if error is not None and status == SpanStatus.SUCCESS:
            status = SpanStatus.ERROR

        if output is not None or span.output is None:
            span.output = output
        if status != SpanStatus.SUCCESS or span.status == SpanStatus.SUCCESS:
            span.status = status
        if error is not None or span.error is None:
            span.error = error
        if error_type is not None or span.error_type is None:
            span.error_type = error_type
        if metadata:
            span.metadata.update(metadata)
        if span.end_time is None:
            span.finish(output=span.output, status=span.status, error=span.error, error_type=span.error_type, metadata=None, end_time=end_time)
        else:
            # 已经 finish 过，不重新计算耗时
            if metadata:
                span.metadata.update(metadata)
        # 弹栈
        self._pop_span_if_top(span.span_id)
        # 写入 store（底层已保证异常不抛）
        try:
            self._store.write(span)
            span._persisted = True
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
            self.end_span(s, output=None, error=f"{type(e).__name__}: {e}", error_type=type(e).__name__)
            raise
        else:
            # 若调用方已手动 finish，则 end_time 已有值，直接写入（end_span 会自动幂等）
            self.end_span(s, output=s.output, status=s.status, error=s.error, error_type=s.error_type, metadata=s.metadata)
        finally:
            # 兜底弹栈：确保无论如何栈都正确弹出，避免父链错乱
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
            self.end_span(s, output=output_data, status=status, error=error_msg, error_type="ToolError" if error_msg else None, end_time=hook_end_ts)
        else:
            self.end_span(s, output=output_data, status=status, error=error_msg, error_type="ToolError" if error_msg else None)

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

        默认回放规则：
        - 有活跃 trace 时，回放当前最内层活跃 trace（嵌套场景自动取当前正在运行的最内层）；
        - 无活跃 trace 时，自动回放最近一次结束的 root trace（即最近一次完整运行）；
        - 显式传入 trace_id / session_id 时按参数精确过滤。
        """
        tid = trace_id
        if tid is None and session_id is None:
            t_stack, _ = self._stacks()
            if t_stack:
                # 嵌套trace场景：取栈顶（当前最内层活跃trace_id），与 start_span 归属逻辑一致
                tid = t_stack[-1]
            else:
                # 无活跃trace时，自动找最近一条结束的root trace（parent_id为None的最新Span所属trace）
                all_spans = self._store.read()
                # 倒序找最近结束的根Span
                for s in reversed(all_spans):
                    if s.parent_id is None:
                        tid = s.trace_id
                        break
        spans = self._store.read(session_id=session_id, trace_id=tid)
        # 优先按创建序号排序，无序号时兜底按时间+span_id，彻底解决Windows低精度时间戳排序不稳定问题
        spans.sort(key=lambda s: (getattr(s, "_create_seq", 0), s.start_time, s.span_id))
        return [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "kind": s.kind.value if isinstance(s.kind, SpanKind) else s.kind,
                "name": s.name,
                "status": s.status.value if isinstance(s.status, SpanStatus) else s.status,
                "duration_ms": round(s.duration_ms, 2) if s.duration_ms is not None else None,
                "error": s.error,
                "error_type": s.error_type,
                "input": s.input,
                "output": s.output,
                "metadata": s.metadata,
            }
            for s in spans
        ]

    # ---------- 树形聚合（K1 收尾） ----------
    def build_tree(
        self,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[dict]:
        """把 ``replay()`` 的平铺 Span 列表按父子关系组装成树形结构。

        规则：
        - 参数语义与 :meth:`replay` 完全一致（默认解析最近一条 root trace），
          所以调用方可以零成本从 ``replay`` 切换到 ``build_tree``；
        - 返回根节点列表（每条 root Span 一个节点）；每个节点是在 replay dict 基础上
          追加 ``children`` 字段（递归包含子节点）；``children`` 内按创建顺序排列；
        - **悬空父引用**（子 span 的 ``parent_id`` 在本批次中找不到）：作为孤儿节点
          提升到 roots 末尾，并附带 ``_orphan=True`` 标记，避免子树整棵丢失；
        - **多 root**（同 trace 存在多个 parent_id=None 的 span）：全部作为根节点返回，
          不做隐式聚合。

        纯只读方法，不修改 store、不改变 ``replay()`` 的任何现有行为。
        """
        # 复用 replay 的"选 trace / 选 session / 排序"逻辑，保证语义对齐
        flat = self.replay(trace_id=trace_id, session_id=session_id)

        # 阶段 1：一次性构造所有节点（保证 by_id 中父节点一定先于子节点可查）
        by_id: dict = {}
        for node in flat:
            n = dict(node)
            n["children"] = []
            by_id[n["span_id"]] = n

        roots: List[dict] = []
        # 阶段 2：挂接——悬空起始点立即提升为 root（治重复）
        for n in by_id.values():
            pid = n.get("parent_id")
            if pid is None:
                roots.append(n)
            else:
                parent = by_id.get(pid)
                if parent is not None:
                    parent["children"].append(n)
                else:
                    # 悬空父引用：起始点立即提升为 root，并标记 _orphan
                    n["_orphan"] = True
                    roots.append(n)

        # 阶段 3：补漏兜底（治丢失）——深度遍历所有 roots，标记已入树的节点；
        # 剩余未认领的节点只可能来自「环 / 自环」这类病态数据，将其补提升为 root，
        # 保证一个不丢。正常 / 悬空数据下本步骤为空。
        seen_span_ids: set = set()

        def _collect(node: dict) -> None:
            if node["span_id"] in seen_span_ids:
                return
            seen_span_ids.add(node["span_id"])
            for c in node["children"]:
                _collect(c)

        for r in roots:
            _collect(r)
        for n in by_id.values():
            if n["span_id"] not in seen_span_ids:
                n.setdefault("_orphan", True)
                roots.append(n)
                # 立即递归标记该节点及其可达子孙，避免环上其它节点被重复提升
                _collect(n)

        return roots

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


__all__ = ["TraceLogger", "SpanKind", "SpanStatus", "SpanRecord", "MEMORY_PATH"]
