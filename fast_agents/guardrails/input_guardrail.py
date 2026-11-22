"""输入护栏 InputGuardrail

在用户输入进入Agent处理前执行检查，拦截恶意输入、敏感内容、Prompt注入攻击。
仅作用于Agent链路的第一个节点，提前阻断风险，避免浪费LLM调用资源。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .guardrail_base import GuardrailBase, list_guardrails
from .models import GuardrailResult
from .errors import InputGuardrailReject


class InputGuardrail(GuardrailBase):
    """输入护栏基类，所有用户输入检查护栏都继承自此类。

    和输出/工具护栏的核心区别：输入护栏拦截时直接终止请求，不会进入LLM推理阶段。
    """

    def reject(self, reason: str, **metadata) -> GuardrailResult:
        """快捷构造输入拦截结果。

        ⚠️ 统一返回 GuardrailResult（不再抛异常）——
        返回值与异常两种写法经 invoke() 归一化后，都会走完整的
        _after_check / duration_ms / 拦截统计 / Trace 打点链路，
        避免"一种写法有统计、一种没统计"的数据断层。
        """
        return GuardrailResult.reject(reason, metadata=metadata or {})


def run_input_guardrails(
    user_input: Union[str, List[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[InputGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> List[GuardrailResult]:
    """执行输入护栏检查，返回所有结果。

    Args:
        user_input: 用户输入，可以是字符串、消息列表或结构化输入
        context: 运行时上下文
        guardrails: 要执行的护栏列表。默认 None。
        use_global: guardrails=None 时是否回退扫描全局注册表。
            ⚠️ 默认 False（不扫）——对齐路线图 A34 隔离语义，避免独立调用时
            意外跑了别的模块/测试注册的护栏。确实需要"跑全部注册护栏"时显式传 True。
        fail_open: 统一覆盖所有护栏的异常姿态（None=用护栏自身设置，默认 fail-closed）。
            GuardrailMiddleware 调用时传入 self.fail_open。

    Returns:
        所有护栏的检查结果列表，按priority排序
    """
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, InputGuardrail)] if use_global else []

    results = []
    for g in sorted(guardrails, key=lambda x: x.priority):
        result = g.invoke(user_input, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            # S5: 抛异常时把已跑完的 results 附在异常上，防止前面通过项的统计/trace 丢失
            raise InputGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details=result.metadata,
                results=results,
            )
    return results


async def arun_input_guardrails(
    user_input: Union[str, List[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[InputGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> List[GuardrailResult]:
    """异步版本的输入护栏执行。"""
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, InputGuardrail)] if use_global else []

    results = []
    for g in sorted(guardrails, key=lambda x: x.priority):
        result = await g.ainvoke(user_input, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            raise InputGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details=result.metadata,
                results=results,
            )
    return results
