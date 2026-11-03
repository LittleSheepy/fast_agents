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
from .context import (
    ContextBuilder, BuildReport,
    TokenCounter, TokenBudget, estimate_text_tokens,
    ContextWindow, ContextLayer,
    ContextCompressor, ContextOffloader,
    ReadOffloadedTool, ListFilesTool,
    build_context_tools,
)
from .mcp import (
    MCPRegistry, MCPClient, MCPToolAdapter, MCPServerTool,
    MCPError, MCPConnectionError, MCPProtocolError, MCPCallError,
    StdioTransport, SSETransport, make_stdio,
    MCPConfig,
)
from .database import (
    SQLiteClient, DatabaseClientBase,
    Schema, Column, ColumnType, Index,
    Query, Filter, FilterOp, SortOrder,
)
from .trace import (
    TraceLogger, SpanKind, SpanStatus, SpanRecord,
    TraceStoreBase, JsonlStore, MEMORY_PATH,
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

    # 上下文工程
    "ContextBuilder", "BuildReport",
    "TokenCounter", "TokenBudget", "estimate_text_tokens",
    "ContextWindow", "ContextLayer",
    "ContextCompressor", "ContextOffloader",
    "ReadOffloadedTool", "ListFilesTool",
    "build_context_tools",

    # MCP
    "MCPRegistry", "MCPClient", "MCPToolAdapter", "MCPServerTool",
    "MCPError", "MCPConnectionError", "MCPProtocolError", "MCPCallError",
    "StdioTransport", "SSETransport", "make_stdio", "MCPConfig",

    # Database（持久化层）
    "SQLiteClient", "DatabaseClientBase",
    "Schema", "Column", "ColumnType", "Index",
    "Query", "Filter", "FilterOp", "SortOrder",

    # Trace（可观测性，G1 初版）
    "TraceLogger", "SpanKind", "SpanStatus", "SpanRecord",
    "TraceStoreBase", "JsonlStore", "MEMORY_PATH",
]
