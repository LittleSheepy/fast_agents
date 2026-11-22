"""
Guardrails 安全护栏模块

为 fast_agents 提供全链路安全防护能力，支持：
1. 输入护栏：用户输入检查，拦截Prompt注入、违规内容
2. 工具调用护栏：工具调用前后校验，拦截高危操作、敏感数据泄露
3. 输出护栏：Agent输出检查，过滤敏感信息、违规内容
4. 中间件集成：零侵入接入ReActAgent执行流
5. 可扩展设计：支持自定义护栏、对接可观测体系

设计对齐 LangChain/LlamaIndex/OpenAI Agents SDK 主流规范，
命名完全遵循项目现有风格：GuardrailBase 对齐 ToolBase，
@guardrail_creator 对齐 @tool_creator，GuardrailMiddleware 对齐 SkillMiddleware。
"""
from .guardrail_base import (
    GuardrailBase,
    guardrail_creator,
    list_guardrails,
    get_guardrail,
)
from .models import (
    GuardrailResult,
    ToolCallInfo,
    ToolResultInfo,
)
from .input_guardrail import (
    InputGuardrail,
    run_input_guardrails,
    arun_input_guardrails,
)
from .output_guardrail import (
    OutputGuardrail,
    run_output_guardrails,
    arun_output_guardrails,
)
from .tool_call_guardrail import (
    ToolCallGuardrail,
    run_tool_call_guardrails,
    run_tool_result_guardrails,
    arun_tool_call_guardrails,
    arun_tool_result_guardrails,
)
from .guardrail_middleware import GuardrailMiddleware
from .errors import (
    GuardrailError,
    InputGuardrailReject,
    OutputGuardrailReject,
    ToolCallGuardrailReject,
    GuardrailConfigError,
)
from .guardrail_trace import log_guardrail_event
from . import preset

__all__ = [
    # 数据模型（pydantic v2 BaseModel，对齐 §5.1）
    "GuardrailResult",
    "ToolCallInfo",
    "ToolResultInfo",
    # 基类与装饰器
    "GuardrailBase",
    "guardrail_creator",
    "list_guardrails",
    "get_guardrail",
    # 三类护栏
    "InputGuardrail",
    "OutputGuardrail",
    "ToolCallGuardrail",
    # 执行函数
    "run_input_guardrails",
    "arun_input_guardrails",
    "run_output_guardrails",
    "arun_output_guardrails",
    "run_tool_call_guardrails",
    "run_tool_result_guardrails",
    "arun_tool_call_guardrails",
    "arun_tool_result_guardrails",
    # 中间件
    "GuardrailMiddleware",
    # 异常
    "GuardrailError",
    "InputGuardrailReject",
    "OutputGuardrailReject",
    "ToolCallGuardrailReject",
    "GuardrailConfigError",
    # 可观测
    "log_guardrail_event",
    # 预设
    "preset",
]
