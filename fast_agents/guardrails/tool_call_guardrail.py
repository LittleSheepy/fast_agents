"""工具调用护栏 ToolCallGuardrail

在每一次工具调用（本地Tool/MCP服务/Skill动作）执行前后进行检查：
1. 前置检查：校验工具名是否在允许列表、参数是否合规、是否有敏感操作（如删文件、执行命令）
2. 后置检查：校验工具返回结果是否包含敏感信息、是否越权
是Agent场景下最核心的安全防护层。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from .guardrail_base import GuardrailBase, list_guardrails
from .models import GuardrailResult, ToolCallInfo, ToolResultInfo
from .errors import ToolCallGuardrailReject


class ToolCallGuardrail(GuardrailBase):
    """工具调用护栏基类，所有工具调用检查护栏都继承自此类。

    支持前置检查（before_call）和后置检查（after_call）两个钩子，
    默认只执行前置检查，子类可以覆盖after_call实现结果校验。

    拦截写法：统一 `return self.reject(原因)`（返回 GuardrailResult）；在钩子里直接
    `raise XxxGuardrailReject` 也支持——invoke() 会把两种写法归一为同一条
    allowed=False 链路（统计 / duration_ms / Trace 均不丢失）。
    """

    # Bug1 修复：ToolCallGuardrail 依赖原始 pydantic 对象（ToolCallInfo/ToolResultInfo）做 isinstance 分发，
    # 不能被 to_scan_text 文本化，否则对象变成字符串导致分发失效。
    _normalize_input = False

    # 是否执行后置检查（检查工具返回结果）
    check_result: bool = False

    def __init__(self, *args, check_result: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if check_result is not None:
            self.check_result = check_result

    def check(self, input_data: Union[ToolCallInfo, ToolResultInfo], context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """统一check入口，根据输入类型自动分发到before_call/after_call"""
        if isinstance(input_data, ToolCallInfo):
            return self.before_call(input_data, context)
        elif isinstance(input_data, ToolResultInfo):
            if self.check_result:
                return self.after_call(input_data, context)
            return GuardrailResult.allow()
        return GuardrailResult.allow()

    def before_call(self, tool_call: ToolCallInfo, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """工具调用前置检查，子类实现具体逻辑，默认放行"""
        return GuardrailResult.allow()

    def after_call(self, tool_result: ToolResultInfo, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """工具调用后置检查，子类实现具体逻辑，默认放行"""
        return GuardrailResult.allow()

    def reject(self, reason: str, **metadata) -> GuardrailResult:
        """快捷构造工具调用拦截结果。

        ⚠️ 统一返回 GuardrailResult（不再抛异常）——返回值与异常两种写法经 invoke()
        归一化后都会走完整的打点链路，避免统计断层。
        """
        return GuardrailResult.reject(reason, metadata=metadata or {})


def run_tool_call_guardrails(
    tool_call: ToolCallInfo,
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[ToolCallGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> List[GuardrailResult]:
    """执行工具调用前置检查（**fail-fast**）；guardrails=None 时默认不执行，use_global=True 才扫全局。

    Returns:
        仅在全部前置护栏都放行时才返回结果列表（按 priority 升序逐个执行）。

    Raises:
        ToolCallGuardrailReject: 遇到**第一条** allowed=False 立即抛出，剩余护栏不再执行；
            `details` 除该结果的 metadata 外额外含 tool_name / tool_args，
            `results` 为已执行完的结果（含拒绝项自身，见 S5）。
    """
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, ToolCallGuardrail)] if use_global else []

    results = []
    for g in sorted(guardrails, key=lambda x: x.priority):
        result = g.invoke(tool_call, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            raise ToolCallGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details={
                    "tool_name": tool_call.tool_name,
                    "tool_args": tool_call.tool_args,
                    **result.metadata,
                },
                results=results,
            )
    return results


def run_tool_result_guardrails(
    tool_result: ToolResultInfo,
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[ToolCallGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> Tuple[ToolResultInfo, List[GuardrailResult]]:
    """执行工具调用结果后置检查（**fail-fast**）；guardrails=None 时默认不执行，use_global=True 才扫全局。

    Returns:
        (current_result, results)：仅在全部后置护栏都放行时才返回；current_result 为依次叠加
        护栏改写（metadata["modified_result"]）后的结果，无任何修改时就是入参本身。

    Raises:
        ToolCallGuardrailReject: 遇到**第一条** allowed=False 立即抛出，剩余护栏不再执行；
            中间件据此屏蔽原始结果并下发安全文案。
    """
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, ToolCallGuardrail) and g.check_result] if use_global else []

    results = []
    current_result = tool_result

    for g in sorted(guardrails, key=lambda x: x.priority):
        result = g.invoke(current_result, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            raise ToolCallGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details={
                    "tool_name": tool_result.tool_name,
                    **result.metadata,
                },
                results=results,
            )
        # 如果护栏修改了工具结果，更新current_result
        if result.metadata.get("modified") and "modified_result" in result.metadata:
            current_result = result.metadata["modified_result"]

    return current_result, results


async def arun_tool_call_guardrails(
    tool_call: ToolCallInfo,
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[ToolCallGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> List[GuardrailResult]:
    """异步版本的工具调用前置检查，语义与 run_tool_call_guardrails 严格一致（同样 fail-fast）。"""
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, ToolCallGuardrail)] if use_global else []

    results = []
    for g in sorted(guardrails, key=lambda x: x.priority):
        result = await g.ainvoke(tool_call, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            raise ToolCallGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details={
                    "tool_name": tool_call.tool_name,
                    "tool_args": tool_call.tool_args,
                    **result.metadata,
                },
                results=results,
            )
    return results


async def arun_tool_result_guardrails(
    tool_result: ToolResultInfo,
    context: Optional[Dict[str, Any]] = None,
    guardrails: Optional[List[ToolCallGuardrail]] = None,
    *,
    use_global: bool = False,
    fail_open: Optional[bool] = None,
) -> Tuple[ToolResultInfo, List[GuardrailResult]]:
    """异步版本的工具调用结果后置检查，语义与 run_tool_result_guardrails 严格一致（同样 fail-fast）。"""
    if guardrails is None:
        guardrails = [g for g in list_guardrails() if isinstance(g, ToolCallGuardrail) and g.check_result] if use_global else []

    results = []
    current_result = tool_result

    for g in sorted(guardrails, key=lambda x: x.priority):
        result = await g.ainvoke(current_result, context, fail_open=fail_open)
        results.append(result)
        if not result.allowed:
            raise ToolCallGuardrailReject(
                result.reason,
                guardrail_name=result.metadata.get("guardrail_name", g.name),
                details={
                    "tool_name": tool_result.tool_name,
                    **result.metadata,
                },
                results=results,
            )
        # 如果护栏修改了工具结果，更新current_result
        if result.metadata.get("modified") and "modified_result" in result.metadata:
            current_result = result.metadata["modified_result"]

    return current_result, results
