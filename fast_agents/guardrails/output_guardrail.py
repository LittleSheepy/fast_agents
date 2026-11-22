"""输出护栏 OutputGuardrail

在Agent最终返回结果给用户前执行检查，过滤敏感内容、幻觉信息、违规输出，
或对输出做脱敏、格式化处理。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from .guardrail_base import GuardrailBase, list_guardrails
from .models import GuardrailResult
from .errors import OutputGuardrailReject, GuardrailConfigError


class OutputGuardrail(GuardrailBase):
    """输出护栏基类，所有Agent输出检查护栏都继承自此类。

    输出护栏可以选择直接拦截抛出异常，也可以返回修改后的输出内容
    （通过metadata["modified_output"]字段传递修改结果）。
    """

    # 是否允许修改输出，True时检查通过后可以返回修改后的内容
    allow_modify: bool = False

    def __init__(self, *args, allow_modify: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if allow_modify is not None:
            self.allow_modify = allow_modify

    def reject(self, reason: str, **metadata) -> GuardrailResult:
        """快捷构造输出拦截结果。

        ⚠️ 统一返回 GuardrailResult（不再抛异常）——返回值与异常两种写法经 invoke()
        归一化后都会走完整的打点链路，避免统计断层。
        """
        return GuardrailResult.reject(reason, metadata=metadata or {})

    def modify(self, modified_output: Any, reason: str = "", **metadata) -> GuardrailResult:
        """返回修改后的输出结果，不拦截但替换内容。"""
        if not self.allow_modify:
            raise GuardrailConfigError(f"护栏 '{self.name}' 不允许修改输出，请设置 allow_modify=True")
        metadata["modified_output"] = modified_output
        metadata["modified"] = True
        return GuardrailResult.allow(metadata=metadata)


def run_output_guardrails(
    agent_output: Union[str, List[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[OutputGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> Tuple[Union[str, List[Any], Dict[str, Any]], List[GuardrailResult]]:
    """执行输出护栏检查，返回（最终输出，检查结果列表）。

    Args:
        agent_output: Agent原始输出
        context: 运行时上下文
        guardrails: 要执行的护栏列表。默认 None。
        use_global: guardrails=None 时是否回退扫描全局注册表，默认 False。

    Returns:
        (final_output, results): 经过所有护栏处理后的最终输出，以及所有检查结果
    """
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, OutputGuardrail)] if use_global else []

    results = []
    current_output = agent_output

    for g in sorted(guardrails, key=lambda x: x.priority):
        result = g.invoke(current_output, context, fail_open=fail_open)
        results.append(result)

        if not result.allowed:
            raise OutputGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details=result.metadata,
                results=results,
            )

        # 如果护栏修改了输出，更新current_output
        if result.metadata.get("modified") and "modified_output" in result.metadata:
            current_output = result.metadata["modified_output"]

    return current_output, results


async def arun_output_guardrails(
    agent_output: Union[str, List[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[OutputGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> Tuple[Union[str, List[Any], Dict[str, Any]], List[GuardrailResult]]:
    """异步版本的输出护栏执行。"""
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, OutputGuardrail)] if use_global else []

    results = []
    current_output = agent_output

    for g in sorted(guardrails, key=lambda x: x.priority):
        result = await g.ainvoke(current_output, context, fail_open=fail_open)
        results.append(result)

        if not result.allowed:
            raise OutputGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details=result.metadata,
                results=results,
            )

        if result.metadata.get("modified") and "modified_output" in result.metadata:
            current_output = result.metadata["modified_output"]

    return current_output, results
