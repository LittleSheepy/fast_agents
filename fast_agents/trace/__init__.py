"""trace 可观测性模块（路线图 K1 初版 + K3 评估工具/成本日报）。

提供 Agent 全链路执行轨迹记录与离线评估能力：
- **TraceLogger**：核心记录器，借鉴 OpenTelemetry / LangChain Callback 的 Trace/Span 概念，
  默认 JsonlStore 一行一条 JSON 落盘，可切换到内存模式，存储层可替换（SQLiteStore 已交付）；
- **SpanKind / SpanStatus / SpanRecord**：Span 类型 / 状态 / 数据模型；
- **JsonlStore / SQLiteStore / TraceStoreBase**：存储后端与抽象基类；
- **TraceEvaluator / TraceMetrics / EvaluationReport**（K3）：离线评估器，
  基于已落盘的 trace 数据批量聚合成本/效率/质量指标，支持 dict / JSON / 文本 / CSV 四种输出；
- **CostDailyReport / DayCostMetrics / daily_cost_report**（K3 高阶）：按自然日
  聚合 Token 成本日报，支持自定义模型价目表，默认内置主流模型公开价。

默认关闭：不传 tracer 给 LLM / Agent 时，行为与改动前逐字节一致；
不传 evaluator / 不调用评估时评估代码零开销。
接入方式见 ``examples/trace/`` 示例与 ``docs/trace模块编写说明.md``。
"""
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase
from .jsonl_store import JsonlStore, MEMORY_PATH
from .sqlite_store import SQLiteStore
from .trace_logger import TraceLogger
from .evaluator import (
    TraceEvaluator,
    TraceMetrics,
    EvaluationReport,
    DayCostMetrics,
    CostDailyReport,
    DEFAULT_PRICE_TABLE,
)

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
    "SQLiteStore",
    # 评估（K3）
    "TraceEvaluator",
    "TraceMetrics",
    "EvaluationReport",
    # 成本日报（K3 高阶）
    "DayCostMetrics",
    "CostDailyReport",
    "DEFAULT_PRICE_TABLE",
]
