"""trace 可观测性模块（路线图 G1 · 初版）。

提供 Agent 全链路执行轨迹记录能力：
- **TraceLogger**：核心记录器，借鉴 OpenTelemetry / LangChain Callback 的 Trace/Span 概念，
  默认 JsonlStore 一行一条 JSON 落盘，可切换到内存模式，存储层可替换（预留 SQLite 位）；
- **SpanKind / SpanStatus / SpanRecord**：Span 类型 / 状态 / 数据模型；
- **JsonlStore / TraceStoreBase**：存储后端与抽象基类。

默认关闭：不传 tracer 给 LLM / Agent 时，行为与改动前逐字节一致。
接入方式见 ``examples/trace/`` 示例与 ``docs/trace模块编写说明.md``。
"""
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase
from .jsonl_store import JsonlStore, MEMORY_PATH
from .trace_logger import TraceLogger

__all__ = [
    # 核心记录器
    "TraceLogger",
    # 数据模型
    "SpanKind",
    "SpanStatus",
    "SpanRecord",
    # 存储
    "TraceStoreBase",
    "JsonlStore",
    "MEMORY_PATH",
]

__version__ = "1.0.0"
