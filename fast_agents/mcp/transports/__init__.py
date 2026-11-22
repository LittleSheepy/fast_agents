"""MCP 传输层包 —— 基于 JSON-RPC 2.0 over stdio / SSE / Streamable HTTP。

不依赖官方 mcp SDK，标准库实现：
- StdioTransport（子进程管道）
- SSETransport（HTTP Server-Sent Events，兼容旧服务端）
- StreamableHTTPTransport（单端点 HTTP，MCP 2025 规范，推荐远程场景使用）

子模块：
- transport_base         ：TransportBase 抽象基类（原 `Transport`，按仓库 Base 命名约定更名）
- stdio_transport        ：StdioTransport + make_stdio_transport 工厂
- sse_transport          ：SSETransport
- streamable_http_transport：StreamableHTTPTransport + make_streamable_http_transport 工厂

对外仍由 fast_agents.mcp 顶层统一导出，`fast_agents.mcp.transports` 导入路径保持不变。
"""

from .transport_base import TransportBase
from .stdio_transport import StdioTransport, make_stdio_transport
from .sse_transport import SSETransport
from .streamable_http_transport import StreamableHTTPTransport, make_streamable_http_transport

__all__ = [
    "TransportBase", "StdioTransport", "SSETransport", "make_stdio_transport",
    "StreamableHTTPTransport", "make_streamable_http_transport",
]
