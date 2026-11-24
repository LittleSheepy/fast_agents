"""guardrails 模块数据模型。

定义护栏模块全部结构化数据：

- ``GuardrailResult``：护栏检查统一返回值（allow / reject 双工厂）；
- ``ToolCallInfo``：工具调用前置检查入参（工具名 / 参数 / 类型 / 来源 / call_id）；
- ``ToolResultInfo``：工具调用后置检查入参（工具名 / success / output / error / metadata）。

对齐 §5.1 规范：统一放在 models.py 中，使用 pydantic v2 ``BaseModel``，与
``fast_agents.trace.models``、memory、mcp 等模块保持一致的数据建模风格。
不再使用 dataclass——dataclass 无法做字段校验、无法被 pydantic/JSON 序列化层原生识别，
与项目其它模块的模型能力（json_safe / OTLP 导出 / Trace 属性记录）不兼容。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class GuardrailResult(BaseModel):
    """护栏检查返回结果模型。

    Attributes:
        allowed: 是否允许通过，False 表示触发拦截。
        reason: 拦截原因说明；allowed=True 时通常为空串。
        metadata: 附加元数据，用于日志、Trace 上报、上层异常 details 透传。
    """

    # B3 修复：`extra="allow"` 会静默接受拼错的字段名（如 `alowd=True`），导致安全关键模型被绕过
    # 改为 `extra="forbid"`，任何未声明字段立即抛 ValidationError
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    allowed: bool
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(cls, metadata: Optional[Dict[str, Any]] = None) -> "GuardrailResult":
        """构造「放行」结果的工厂方法。"""
        return cls(allowed=True, metadata=metadata or {})

    @classmethod
    def reject(cls, reason: str, metadata: Optional[Dict[str, Any]] = None) -> "GuardrailResult":
        """构造「拦截」结果的工厂方法。"""
        return cls(allowed=False, reason=reason, metadata=metadata or {})


class ToolCallInfo(BaseModel):
    """工具调用前置检查的入参模型，作为 ToolCallGuardrail.before_call 的输入。

    Attributes:
        tool_name: 工具名（注册到 ToolRegistry 的唯一标识）。
        tool_args: 工具参数 dict。
        tool_type: 工具类型，取值 local / mcp / skill。
        source: 工具来源，MCP 时为 server 名，Skill 时为 skill 名。
        call_id: 工具调用 ID，用于链路追踪（对应 MsgToolCall.id）。
    """

    # B3 修复：同上，防止 `tool_ars={...}` 这类拼错被静默接受
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_type: str = "local"
    source: str = ""
    call_id: str = ""


class ToolResultInfo(BaseModel):
    """工具执行结果后置检查的入参模型，作为 ToolCallGuardrail.after_call 的输入。

    Attributes:
        tool_name: 工具名。
        success: 工具是否执行成功。
        output: 工具返回的文本内容（失败时通常为空）。
        error: 错误信息（成功时通常为空）。
        metadata: 附加元数据（耗时、truncated 等）。
    """

    # B3 修复：同上
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
