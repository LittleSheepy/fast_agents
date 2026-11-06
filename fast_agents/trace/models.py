"""trace 模块数据模型。

定义：
- ``SpanKind``：Span 类型枚举（llm/tool/decision/agent/custom）——对应用户要求的三类基础打点；
- ``SpanStatus``：执行状态枚举（success/error）；
- ``SpanRecord``：单条 Span 的持久化记录模型（pydantic v2 BaseModel）；

最小字段集对齐路线图 G1：时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id；
额外补充推荐字段：trace_id / span_id / parent_id / name / status / error / metadata。
"""
from __future__ import annotations

import itertools
import uuid
import time
import threading
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr


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
    """当前时间戳（秒，带小数，time.time() 系统时间，适合记录时间点；耗时计算时自动做下限保护，避免负值）。"""
    return time.time()


# 全局自增序号，线程安全，保证创建顺序唯一可排序，彻底解决Windows低精度时间戳排序混乱问题
_seq_counter = itertools.count(1)
_seq_lock = threading.Lock()


def _next_seq() -> int:
    with _seq_lock:
        return next(_seq_counter)


class SpanRecord(BaseModel):
    """单条 Span 的持久化记录。

    字段覆盖路线图 G1「最小字段集」：
    ``timestamp / kind(类型) / input / output / duration_ms(耗时) / session_id``；
    并补充 OpenTelemetry 风格的追踪链字段：``trace_id / span_id / parent_id / name / status``。
    """

    # ---------- 内部幂等标记（不持久化） ----------
    _persisted: bool = PrivateAttr(default=False)

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

    # ---------- 扩展 ----------
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据，如模型名、token 用量")

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """初始化时分配唯一创建序号，优先使用metadata内持久化的序号保证跨进程排序一致"""
        if "_create_seq" in self.metadata:
            # 反序列化场景：从持久化数据中恢复原有创建序号，保证重读后顺序一致
            self._create_seq = self.metadata.pop("_create_seq")
        else:
            # 新创建场景：分配全局递增序号，并写入metadata持久化
            with _seq_lock:
                self._create_seq = next(_seq_counter)
            self.metadata["_create_seq"] = self._create_seq

    # ---------- 便捷方法 ----------
    def finish(
        self,
        output: Any = None,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """标记 Span 结束，自动计算耗时。

        Args:
            end_time: 可选的精确结束时间戳（秒）。默认 None 时取当前时刻
                （行为与改动前一致）；传入时以该值作为 ``end_time``，用于
                「后钩子」这类拿不到真实起止、但需精确对齐耗时的场景。
        """
        self.end_time = end_time if end_time is not None else _now_ts()
        # 耗时下限保护：end_time早于start_time时截断为0，避免出现负耗时
        self.duration_ms = max(0.0, (self.end_time - self.start_time) * 1000.0)
        self.output = output or self.output
        self.status = status
        self.error = error or self.error
        if metadata:
            self.metadata.update(metadata)


__all__ = [
    "SpanKind",
    "SpanStatus",
    "SpanRecord",
    "_new_id",
    "_now_ts",
]
