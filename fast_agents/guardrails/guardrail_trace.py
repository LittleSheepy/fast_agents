"""Guardrail 可观测性埋点，对接项目现有 TraceLogger 体系。

所有护栏的执行事件（通过/拦截/异常）统一通过 :func:`log_guardrail_event`
记录到 Trace 链路，支持后续统计拦截率、误判率、高频触发规则、按 guardrail
维度聚合（SpanKind.GUARDRAIL 专用种类）。

设计约束（修复 L8 漂移问题）：
- **唯一打点入口**：middleware 不再内联 start_span/end_span，全部复用本函数，
  避免两处字段漂移（event_type vs phase、metadata 字段不一致）。
- **字段对齐**：metadata 统一包含 guardrail_name/phase/event_type/allowed/
  is_error/reason/duration_ms，外加调用方传入的 context；
- **异常容错**：埋点失败只打 warning，绝不阻塞主流程。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .models import GuardrailResult

_logger = logging.getLogger(__name__)


def _guardrail_span_kind():
    """延迟导入 SpanKind，避免循环依赖；护栏打点统一使用 SpanKind.GUARDRAIL。"""
    try:
        from ..trace import SpanKind
        return SpanKind.GUARDRAIL
    except ImportError:
        return "guardrail"


def log_guardrail_event(
    tracer: Any,
    guardrail_name: str,
    result: GuardrailResult,
    phase: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """统一的护栏事件打点函数。

    Args:
        tracer: TraceLogger 实例，None 时直接返回（零开销）。
        guardrail_name: 护栏名称，用于 span name（``guardrail:<name>``）。
        result: 护栏执行结果对象，承载 allowed/reason/is_error/duration_ms。
        phase: 检查阶段，取值 ``input_check`` / ``tool_pre_check`` / ``tool_post_check`` / ``output_check``；
               会同时写入 metadata.phase 与 metadata.event_type，消除历史字段漂移。
        context: 业务上下文，键值会合并进 metadata，供评估器/OTLP 导出使用。
    """
    if tracer is None:
        return

    try:
        span = tracer.start_span(
            name=f"guardrail:{guardrail_name}",
            kind=_guardrail_span_kind(),
            input={"phase": phase},
            metadata={
                "guardrail_name": guardrail_name,
                "phase": phase,
                "event_type": phase,  # 与 event_type 字段同值，避免两处漂移
                "allowed": result.allowed,
                "is_error": result.metadata.get("is_error", False),
                "reason": result.reason,
                "duration_ms": result.metadata.get("duration_ms", 0),
                **(context or {}),
            },
        )
        # Bug2 修复：仅在护栏实际拦截（not allowed）时才标记 span 为 ERROR。
        # fail_open=True 路径：护栏出错但放行（allowed=True, is_error=True, reason=""），
        # 若传 error="" 给 end_span，因 "" is not None → True，会错误置 span.status=ERROR，
        # 误导监控面板的护栏错误率统计。正确做法：放行就是放行，is_error 留在 metadata 供诊断。
        _has_error = (not result.allowed) and bool(result.reason)
        tracer.end_span(
            span,
            output={"allowed": result.allowed},
            error=result.reason if _has_error else None,
        )
    except Exception as _e:
        # Trace 埋点失败不影响主流程，但记录 warning 避免静默空转
        _logger.warning("log_guardrail_event 打点失败: %s", _e)
