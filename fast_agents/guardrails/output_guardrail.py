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

    子类有两种写法（推荐返回，两种写法经 invoke() 归一后链路完全一致）：
    - 拦截：`return self.reject(原因)`（allowed=False）；在 check() 里直接
      `raise OutputGuardrailReject` 也支持；
    - 改写/脱敏：先置 `allow_modify=True`，再 `return self.modify(新内容)`，
      新内容由 modify() 写进 metadata["modified_output"] 与 metadata["modified"]，
      由 run_output_guardrails 按 priority 顺序依次叠加。
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
        """返回修改后的输出结果，不拦截但替换内容。

        Args:
            modified_output: 改写后的输出内容。
            reason: 修改原因说明（写入 metadata["modify_reason"] 供审计/Trace 追溯）。
            **metadata: 额外元数据键值对。
        """
        if not self.allow_modify:
            raise GuardrailConfigError(f"护栏 '{self.name}' 不允许修改输出，请设置 allow_modify=True")
        metadata["modified_output"] = modified_output
        metadata["modified"] = True
        if reason:
            metadata["modify_reason"] = reason
        return GuardrailResult.allow(metadata=metadata)


def run_output_guardrails(
    agent_output: Union[str, List[Any], Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[OutputGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> Tuple[Union[str, List[Any], Dict[str, Any]], List[GuardrailResult]]:
    """执行输出护栏检查（**fail-fast**：命中即抛异常，不会跑完全部护栏）。

    Args:
        agent_output: Agent原始输出
        context: 运行时上下文
        guardrails: 要执行的护栏列表。默认 None。
        use_global: guardrails=None 时是否回退扫描全局注册表，默认 False。

    Returns:
        (final_output, results)：仅在**全部护栏都放行**时才返回。
        final_output 为依次叠加各护栏 modify() 后的结果（无任何修改时与 agent_output 同一对象），
        results 按 priority 升序逐护栏收集；只要有一条不放行就不会返回。

    Raises:
        OutputGuardrailReject: 遇到**第一条** allowed=False 立即抛出，剩余护栏不再执行
            （因此排在后面的脱敏/改写没机会跑，这是有意的 fail-fast 姿态）。
            异常携带：`guardrail_name`=触发拦截的护栏名、`details`=该结果的 metadata、
            `results`=已执行完的结果（含拒绝项自身，见 S5）。
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
    """异步版本的输出护栏执行，语义与 run_output_guardrails 严格一致（同样 fail-fast）。

    Returns: 全部放行时返回 (final_output, results)；Raises: OutputGuardrailReject。
    """
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
