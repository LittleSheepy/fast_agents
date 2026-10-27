"""MCP（Model Context Protocol）模块 —— 零依赖实现，让 Agent 接入外部工具服务器。

分层与数据流：
    ReActAgent → ToolRegistry → MCPRegistry → MCPClient → Transport（Stdio/SSE）→ 远端 server

提供：
- 传输：StdioTransport / SSETransport
- 客户端：MCPClient（握手 / list_tools / call_tool）
- 注册表：MCPRegistry（多 server 批量管理 + 一键展开到 ToolRegistry）
- 工具适配器：MCPToolAdapter（细粒度，推荐）/ MCPServerTool（粗粒度，兼容 auto_expand）
- builtin 管理工具：list_mcp_servers / connect_mcp_server / disconnect_mcp_server
  （在 fast_agents.tools.builtin.mcp_tool 里，跟 memory_tool 同层级）
"""

from .errors import (
    MCPError, MCPConnectionError, MCPProtocolError, MCPCallError,
)
from .models import CallResult, InitializeResult, ServerCapabilities, ToolInfo, MCPConfig
from .transports import SSETransport, StdioTransport, Transport, make_stdio
from .client import MCPClient
from .registry import MCPRegistry
from .tool import MCPToolAdapter, MCPServerTool

__all__ = [
    # errors
    "MCPError", "MCPConnectionError", "MCPProtocolError", "MCPCallError",
    # models
    "CallResult", "InitializeResult", "ServerCapabilities", "ToolInfo", "MCPConfig",
    # transports
    "Transport", "StdioTransport", "SSETransport", "make_stdio",
    # core
    "MCPClient", "MCPRegistry",
    # tool adapters
    "MCPToolAdapter", "MCPServerTool",
]
