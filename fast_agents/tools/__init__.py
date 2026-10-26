"""Tools 模块

参考 LangChain / CrewAI / LlamaIndex / Anthropic 工具设计，核心组件：
- ToolBase：工具抽象基类
- StructuredTool / @tool：函数式工具
- ToolResult / ToolStatus：结构化执行结果
- ToolException：可恢复错误
- ToolRegistry：工具注册表
- builtin：内置工具集合（计算器/终端/搜索/文件读取）
"""

from .tool_base import ToolBase
from .structured_tool import StructuredTool, tool
from .tool_result import ToolResult, ToolStatus
from .tool_exception import ToolException, ToolInitializationError, ToolPermissionError
from .tool_registry import ToolRegistry, global_registry
from . import builtin

__all__ = [
    # 基础工具系统
    "ToolBase", "StructuredTool", "tool",

    # 工具响应协议
    "ToolResult", "ToolStatus",
    "ToolException", "ToolInitializationError", "ToolPermissionError",
    "ToolRegistry", "global_registry",
    "builtin",
]
