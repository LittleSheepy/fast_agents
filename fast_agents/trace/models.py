"""trace 模块数据模型。

定义：
- ``SpanKind``：Span 类型枚举（llm/tool/decision/agent/custom）——对应用户要求的三类基础打点；
- ``SpanStatus``：执行状态枚举（success/error）；
- ``SpanRecord``：单条 Span 的持久化记录模型（pydantic v2 BaseModel）；

最小字段集对齐路线图 G1：时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id；
额外补充推荐字段：trace_id / span_id / parent_id / name / status / error / metadata。
"""
from __future__ import annotations

import json
import uuid
import time
import threading
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr

from ..utils.json_utils import json_safe_dumps, json_safe_load
from ..utils.utils import to_str

class SpanKind(str, Enum):
    """Span 类型。路线图最小集为 llm/tool/decision，额外留 agent/custom 作扩展。"""

    LLM = "llm"           # 大模型调用（invoke/stream_invoke/embed）
    TOOL = "tool"         # 工具调用（通过 ToolRegistry.add_hook 自动打点）
    DECISION = "decision" # Agent 决策/推理步骤（是否调工具、最终回答）
    AGENT = "agent"       # 一次完整 Agent run
    CUSTOM = "custom"     # 用户自定义打点


class SpanStatus(str, Enum):
    """Span 执行状态。"""

    SUCCESS = "success"
    ERROR = "error"


def _new_id() -> str:
    """生成短 UUID（去掉横线，便于日志阅读）。"""
    return uuid.uuid4().hex


def _now_ts() -> float:
    """当前时间戳（秒，带小数，time.time() 墙钟）。

    仅用于记录 **时间点**（start_time / end_time，用于展示 / 排序 / 落盘）；
    耗时（duration_ms）不再用墙钟差值，而是由 ``time.perf_counter()`` 单调时钟差值
    计算（见 :meth:`SpanRecord.finish`），避免系统墙钟被 NTP 校正 / 手动改表时
    出现耗时失真或负值。
    """
    return time.time()


# 全局自增序号，线程安全，保证创建顺序唯一可排序，彻底解决 Windows 低精度时间戳排序混乱问题。
# 用「整数 + 锁」而非 itertools.count，是为了在读回历史数据后能把计数器推进到历史最大序号
# 之后（见 _bump_seq），避免跨进程/跨批次续写时新旧数据序号冲撞、回放排序倒挂。
_seq_counter = 0
_seq_lock = threading.Lock()


def _next_seq() -> int:
    """取一个全局线程安全自增序号（进程内单调递增）。"""
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _bump_seq(seq: int) -> None:
    """把全局序号推进到至少 ``seq``，保证之后新建 Span 的序号一定更大。

    读回历史数据（jsonl / SQLite）恢复序号后调用：把历史数据里已占用的最大序号
    同步到全局计数器，这样同一进程续写的新数据序号必然大于全部历史数据，
    跨进程 / 跨批次回放时顺序依旧稳定。
    """
    global _seq_counter
    try:
        n = int(seq)
    except (TypeError, ValueError):
        return
    if n <= 0:
        return
    with _seq_lock:
        if n > _seq_counter:
            _seq_counter = n



def _infer_error_type(error: Optional[str]) -> Optional[str]:
    """从 error 字符串里推断异常类型（纯函数，失败时返回 None 不抛异常）。

    约定的 error 格式为 ``"ExceptionName: message"``：
    - 冒号开头部分是异常类名（如 ValueError / TimeoutError / ConnectionError）；
    - 没有冒号时整体当作类型名（兼容纯类型名写法）；
    - 无法识别（空串、纯符号、中文开头）时返回 None。
    """
    if not error or not isinstance(error, str):
        return None
    try:
        name = error.split(":", 1)[0].strip()
        if not name:
            return None
        # 异常类名必须是 ASCII 标识符（允许点号限定名，如 "json.JSONDecodeError"）：
        # 开头必须是 ASCII 字母或下划线，后续只允许 ASCII 字母/数字/下划线/点号；
        # 中文、纯数字、纯符号开头一律不识别，避免把中文消息错判成类型名。
        if not name:
            return None
        first_ok = name[0].isascii() and (name[0].isalpha() or name[0] == "_")
        rest_ok = all(c.isascii() and (c.isalnum() or c in "._") for c in name)
        if first_ok and rest_ok:
            return name
        return None
    except Exception:
        return None


class SpanRecord(BaseModel):
    """单条 Span 的持久化记录。

    字段覆盖路线图 G1「最小字段集」：
    ``timestamp / kind(类型) / input / output / duration_ms(耗时) / session_id``；
    并补充 OpenTelemetry 风格的追踪链字段：``trace_id / span_id / parent_id / name / status``。
    """

    # ---------- 内部标记（私有属性：不参与 model_dump、不写入用户 metadata） ----------
    _persisted: bool = PrivateAttr(default=False)
    _create_seq: int = PrivateAttr(default=0)
    # 单调时钟起点（time.perf_counter 秒）：仅由 TraceLogger.start_span 在「正常创建」时设置，
    # 用于 finish 时计算抗墙钟跳变的耗时。直接构造的 SpanRecord / 从存储读回的 Span 保持 None，
    # 耗时口径退化为旧的墙钟差值（零破坏）。
    _mono_start: Optional[float] = PrivateAttr(default=None)

    # ---------- 追踪链 ----------
    trace_id: str = Field(..., description="整条链路的唯一 id，一次 run 内所有 Span 共享")
    span_id: str = Field(default_factory=_new_id, description="当前 Span 自身 id")
    parent_id: Optional[str] = Field(default=None, description="父 Span id；根 Span 为 None")
    session_id: Optional[str] = Field(default=None, description="会话 id（跨 run 聚合用）")

    # ---------- 基本信息 ----------
    name: str = Field(..., description="Span 名称，如 'llm.invoke' / 'tool.calculator' / 'agent.run'")
    kind: SpanKind = Field(..., description="Span 类型")

    # ---------- 时间 ----------
    start_time: float = Field(default_factory=_now_ts, description="开始时间戳（秒，time.time()）")
    end_time: Optional[float] = Field(default=None, description="结束时间戳")
    duration_ms: Optional[float] = Field(default=None, description="耗时（毫秒）；end 后自动计算，下限保护为0，不出现负值")

    # ---------- 输入输出 ----------
    input: Any = Field(default=None, description="入参；会被 json.dumps 序列化（无法序列化的转 str）")
    output: Any = Field(default=None, description="出参；序列化规则同 input")

    # ---------- 状态 ----------
    status: SpanStatus = Field(default=SpanStatus.SUCCESS, description="执行状态")
    error: Optional[str] = Field(default=None, description="异常信息（status=error 时填写）")
    error_type: Optional[str] = Field(default=None, description="异常分类（异常类名，如 ValueError / ToolError / LLMError）；status=error 时自动推断")

    # ---------- 扩展 ----------
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据，如模型名、token 用量")

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """确定创建序号（仅存私有属性 ``_create_seq``，**绝不写入 ``metadata``**）。

        - 读回场景：存储层用 ``model_validate(..., context={"create_seq": n})``
          注入落盘序号（老数据无序号时注入 0），此处直接采用，保证跨进程顺序一致；
        - 新建场景：分配全局线程安全自增序号。

        序号是内部排序依据，放在私有属性里，既避免污染用户 ``metadata``，也避免
        同一份数据「实时看」与「落盘再读回」出现两种口径。
        """
        if isinstance(__context, dict) and "create_seq" in __context:
            try:
                self._create_seq = int(__context["create_seq"] or 0)
            except (TypeError, ValueError):
                self._create_seq = 0
            # 读回历史数据后推进全局计数器，避免续写的新数据与历史序号冲撞（跨进程排序稳定）
            _bump_seq(self._create_seq)
        elif self._create_seq <= 0:
            self._create_seq = _next_seq()

    # ---------- 便捷方法 ----------
    def finish(
        self,
        output: Any = None,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """标记 Span 结束，自动计算耗时。

        Args:
            error_type: 异常分类；显式传入时优先采用，否则从 ``error`` 自动推断
                （冒号前的异常类名，例如 ``ValueError`` / ``TimeoutError``）。
            end_time: 可选的精确结束时间戳（秒）。默认 None 时取当前时刻
                （行为与改动前一致）；传入时以该值作为 ``end_time``，用于
                「后钩子」这类拿不到真实起止、但需精确对齐耗时的场景。

        耗时（duration_ms）计算口径：
        - 显式传入 ``end_time``（后钩子 / 手工指定精确起止）：沿用墙钟差值
          ``end_time - start_time``，与改动前逐字节一致；
        - 默认路径且 Span 由 :meth:`TraceLogger.start_span` 正常创建（``_mono_start``
          有值）：用 ``time.perf_counter()`` 单调时钟差值，抗系统墙钟跳变
          （NTP 校正 / 手动改表）导致的耗时失真；
        - 默认路径但直接构造的 SpanRecord（``_mono_start`` 为 None）：退化为墙钟差值，
          保持与改动前一致的旧口径。
        两种口径都做下限保护：结果小于 0 时截断为 0，绝不出现负耗时。
        """
        self.end_time = end_time if end_time is not None else _now_ts()
        # 耗时计算：
        # - 显式传入 end_time（后钩子 / 手工指定精确起止）→ 沿用墙钟差值，与改动前一致；
        # - 默认路径且 Span 由 TraceLogger.start_span 正常创建（_mono_start 有值）→ 单调时钟差值，
        #   抗系统墙钟跳变（NTP 校正 / 手动改表）；
        # - 默认路径但直接构造的 SpanRecord（_mono_start 为 None）→ 退化为墙钟差值，保持旧口径。
        mono_start = getattr(self, "_mono_start", None) if end_time is None else None
        if mono_start is not None:
            self.duration_ms = max(0.0, (time.perf_counter() - mono_start) * 1000.0)
            self.end_time = self.start_time + self.duration_ms / 1000
        else:
            # 耗时下限保护：end_time早于start_time时截断为0，避免出现负耗时
            self.duration_ms = max(0.0, (self.end_time - self.start_time) * 1000.0)
        # 用 `is not None` 判断「是否传入」，而非真值判断：
        # 业务合法的 falsy 结果（0 / False / "" / [] / {}）必须原样记录，
        # 不能被 `or` 当成「没传」而丢弃，否则回放 / 评估看到的输出会失真。
        if output is not None:
            self.output = output
        self.status = status
        if error is not None:
            self.error = error
        # 有 error 但未显式传 error_type 时自动推断：取冒号前的异常类名
        if self.error is not None and error_type is None:
            error_type = _infer_error_type(self.error)
        if error_type is not None:
            self.error_type = error_type
        elif self.error is None:
            self.error_type = None
        if metadata:
            self.metadata.update(metadata)

def dict_to_span_record(row: dict) -> SpanRecord:
    """把数据库行还原成 SpanRecord（忽略自增 id 列）。"""
    fields: dict = {
        "trace_id":    row["trace_id"],
        "span_id":     row["span_id"],
        "parent_id":   row.get("parent_id"),
        "session_id":  row.get("session_id"),
        "name":        row["name"],
        "kind":        SpanKind(row["kind"]) if row.get("kind") is not None else SpanKind.CUSTOM,
        "end_time":    row.get("end_time"),
        "duration_ms": row.get("duration_ms"),
        "input":       json_safe_load(row.get("input")),
        "output":      json_safe_load(row.get("output")),
        "status":      SpanStatus(row["status"]) if row.get("status") is not None else SpanStatus.SUCCESS,
        "error":       row.get("error"),
        "error_type":  row.get("error_type"),
        "metadata":    json_safe_load(row.get("metadata")) or {},
    }
    # start_time 有默认值，仅在库里有值时覆盖，避免 None 覆盖默认时间
    if row.get("start_time") is not None:
        fields["start_time"] = row["start_time"]
    seq = row.get("create_seq")
    try:
        seq = int(seq) if seq is not None else 0
    except (TypeError, ValueError):
        seq = 0
    return SpanRecord.model_validate(fields, context={"create_seq": seq})

def span_record_to_dict(span: SpanRecord) -> dict:
    """把 SpanRecord 转成数据库行字典（字段名与表列一一对应）。

    直接按字段取值、再对 input/output/metadata 做 ``_safe_serialize`` 兜底，
    而不是走 ``model_dump(mode="json")``——后者遇到无法识别的对象会整体抛
    ``PydanticSerializationError``，导致整条 Span 被静默丢弃。这里保证任何
    input/output 都能落库（无法序列化的转 str），与 JsonlStore 的兜底口径一致。
    """
    return {
        "span_id":     span.span_id,
        "trace_id":    span.trace_id,
        "parent_id":   span.parent_id,
        "session_id":  span.session_id,
        "name":        span.name,
        "kind":        to_str(span.kind),
        "start_time":  span.start_time,
        "end_time":    span.end_time,
        "duration_ms": span.duration_ms,
        "input":       json_safe_dumps(span.input),
        "output":      json_safe_dumps(span.output),
        "status":      to_str(span.status),
        "error":       span.error,
        "error_type":  span.error_type,
        "metadata":    json_safe_dumps(span.metadata or {}),
        "create_seq":  getattr(span, "_create_seq", 0),
    }


__all__ = [
    "SpanKind",
    "SpanStatus",
    "SpanRecord",
    "dict_to_span_record",
    "_new_id",
    "_now_ts",
]
