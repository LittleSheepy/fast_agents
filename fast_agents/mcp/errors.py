"""MCP 模块异常体系（三层：连接层 / 协议层 / 业务层）"""

from __future__ import annotations


class MCPError(Exception):
    """MCP 模块基类异常"""


class MCPConnectionError(MCPError):
    """连接层：启动失败、握手超时、传输中断"""


class MCPProtocolError(MCPError):
    """协议层：收到的消息不符合 JSON-RPC 2.0 或 MCP 规范"""


class MCPCallError(MCPError):
    """业务层：远端返回 JSON-RPC error / 工具不存在 / 参数校验失败"""
