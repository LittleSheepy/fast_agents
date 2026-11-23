"""FastAgents - 灵活、可扩展的多智能体框架"""

from dotenv import load_dotenv
load_dotenv()

from .version import __version__, __author__, __email__, __description__
from .llm import LLM, StreamChunk, RetryConfig
from .messages import (
    Message, Messages, Content, ContentPart,
    TextPart, ImagePart, content_to_text, normalize_content, PartType,
)
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
    StdioTransport, SSETransport, StreamableHTTPTransport,
    make_stdio_transport, make_streamable_http_transport,
    MCPConfig,
)
from .database import (
    SQLiteClient, MySQLClient, DatabaseClientBase,
    Schema, Column, ColumnType, Index,
    Query, Filter, FilterOp, SortOrder,
    DatabaseError, DatabaseConnectionError,
    DatabaseOperationalError, DatabaseValidationError,
)
from .trace import (
    TraceLogger, SpanKind, SpanStatus, SpanRecord,
    TraceStoreBase, JsonlStore, MEMORY_PATH, SQLiteStore,
)
from .skills import (
    BaseSkill, skill,
    SkillRegistry, SkillLoader, SkillMiddleware,
    SkillMetadata, SkillLoadResult,
)
from .guardrails import (
    GuardrailRegistry,
    global_guardrail_registry,
    GuardrailBlocked,
    GuardrailModified,
    ToolDecision,
    GuardrailBase, GuardrailResult, guardrail_creator,
    list_guardrails, get_guardrail, to_scan_text,
    InputGuardrail, OutputGuardrail, ToolCallGuardrail,
    ToolCallInfo, ToolResultInfo,
    GuardrailMiddleware,
    GuardrailError, InputGuardrailReject, OutputGuardrailReject,
    ToolCallGuardrailReject, GuardrailConfigError,
    log_guardrail_event,
    preset as guardrail_presets,
)

__all__ = [
    # 版本信息
    "__version__", "__author__", "__email__", "__description__",

    # Agent范式
    "LLM", "ReActAgent", "StreamChunk", "RetryConfig",

    # 消息系统（多模态支持）
    "Message", "Messages",
    "Content", "ContentPart", "TextPart", "ImagePart",
    "content_to_text", "normalize_content", "PartType",

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
    "StdioTransport", "SSETransport", "StreamableHTTPTransport",
    "make_stdio_transport", "make_streamable_http_transport", "MCPConfig",

    # Database（持久化层）
    "SQLiteClient", "MySQLClient", "DatabaseClientBase",
    "Schema", "Column", "ColumnType", "Index",
    "Query", "Filter", "FilterOp", "SortOrder",
    "DatabaseError", "DatabaseConnectionError",
    "DatabaseOperationalError", "DatabaseValidationError",

    # Trace（可观测性，G1 初版，v1.1.0 新增 SQLiteStore）
    "TraceLogger", "SpanKind", "SpanStatus", "SpanRecord",
    "TraceStoreBase", "JsonlStore", "MEMORY_PATH", "SQLiteStore",

    # Skills 技能系统（v1.1.0）
    "BaseSkill", "skill",
    "SkillRegistry", "SkillLoader", "SkillMiddleware",
    "SkillMetadata", "SkillLoadResult",

    # Guardrails 安全护栏（L系列，v2.58.0）
    "GuardrailBase", "GuardrailResult", "guardrail_creator",
    "list_guardrails", "get_guardrail", "to_scan_text",
    "GuardrailRegistry", "global_guardrail_registry",
    "InputGuardrail", "OutputGuardrail", "ToolCallGuardrail",
    "ToolCallInfo", "ToolResultInfo",
    "GuardrailMiddleware",
    "GuardrailBlocked", "GuardrailModified",
    "ToolDecision",  # Bug3：before_tool_call 的线程安全拦截返回值（allowed, reason）
    "GuardrailError", "InputGuardrailReject", "OutputGuardrailReject",
    "ToolCallGuardrailReject", "GuardrailConfigError",
    "log_guardrail_event",
    "guardrail_presets",
]
