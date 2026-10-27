"""FastAgents - 灵活、可扩展的多智能体框架"""

from dotenv import load_dotenv
load_dotenv()

from .version import __version__, __author__, __email__, __description__
from .llm import LLM
from .agents.react_agent.react_agent import ReActAgent
from .tools import (
    ToolBase, StructuredTool, tool, ToolResult, ToolStatus,
    ToolException, ToolRegistry, global_registry, builtin,
)
from .memory import MemoryManager
from .mcp import (
    MCPRegistry, MCPClient, MCPToolAdapter, MCPServerTool,
    MCPError, MCPConnectionError, MCPProtocolError, MCPCallError,
    StdioTransport, SSETransport, make_stdio,
    MCPConfig,
)

__all__ = [
    # 版本信息
    "__version__", "__author__", "__email__", "__description__",

    # Agent范式
    "LLM", "ReActAgent",

    # 工具系统
    "ToolBase", "StructuredTool", "tool", "ToolResult", "ToolStatus",
    "ToolException", "ToolRegistry", "global_registry", "builtin",

    # 记忆系统
    "MemoryManager",

    # MCP
    "MCPRegistry", "MCPClient", "MCPToolAdapter", "MCPServerTool",
    "MCPError", "MCPConnectionError", "MCPProtocolError", "MCPCallError",
    "StdioTransport", "SSETransport", "make_stdio", "MCPConfig",
]
