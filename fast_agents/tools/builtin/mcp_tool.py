"""MCP 通用工具 —— 让 Agent 能在运行时发现、连接、断开 MCP 服务器。

提供三个工具（对齐 memory_tool.py 的三工具模式）：
- list_mcp_servers:     列出当前已注册/已连接的 MCP server 及其中可用工具
- connect_mcp_server:   运行时动态连接一个 MCP server（stdio 命令串或 SSE URL）
- disconnect_mcp_server:断开一个已连接的 MCP server

设计原则：
- 三个工具都是「管理类」工具，不直接代理远端工具调用；真正的远端工具调用
  在细粒度展开后作为独立工具出现在工具表里，模型直接点名调用即可；
- 构造时持 ReActAgent 引用（和 RememberTool 持 MemoryManager 引用同构），
  connect/disconnect 后能直接操作 agent.mcp_registry / agent.tool_registry，
  并调用 agent.refresh_system_prompt() 同步提示词。

使用方式：
    from fast_agents.tools.builtin.mcp_tool import build_mcp_tools
    agent.add_tools(build_mcp_tools(agent))  # 自引用
"""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult


class ListMcpInput(BaseModel):
    pass  # 无参数


class ConnectMcpInput(BaseModel):
    name: str = Field(description="给这个 MCP server 起一个别名，后续断开/查询用（只能包含字母、数字、下划线、连字符）")
    target: str = Field(description="连接目标：stdio 命令串（如 'uvx mcp-server-filesystem D:/code'）或 SSE URL（http(s):// 开头）")
    transport: str = Field(default="auto", description="传输类型：auto（自动判断，URL 开头用 SSE）/ stdio / sse")
    expand: str = Field(default="fine", description="展开模式：fine 细粒度（每个远端工具独立，推荐）/ coarse 粗粒度（一个门脸工具）")


class DisconnectMcpInput(BaseModel):
    name: str = Field(description="要断开的 MCP server 别名")


class ListMcpServersTool(ToolBase):
    name = "list_mcp_servers"
    description = "列出当前已注册/已连接的 MCP 服务器及其提供的工具，不确定有哪些外部能力时调用。"
    args_schema = ListMcpInput
    tags = ["mcp"]

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def run(self) -> Union[str, ToolResult]:
        registry = self.agent.mcp_registry
        if registry is None:
            return ToolResult.ok("当前未启用 MCP 系统（enable_mcp=False）。")
        clients = registry.clients
        factories = registry.factories
        if not clients and not factories:
            return ToolResult.ok("当前没有任何 MCP 服务器。可用 connect_mcp_server 动态添加。")

        lines = []
        for name in sorted(set(list(factories.keys()) + list(clients.keys()))):
            status = "✅ 已连接" if name in clients else "⏳ 未连接"
            if name in clients:
                info = clients[name].initialize_result
                server_info = info.server_info if info else {}
                tools = registry.list_tools(name).get(name, [])
                tool_names = ", ".join(t.name for t in tools) if tools else "(无)"
                lines.append(
                    f"- **{name}** {status}（{server_info.get('name', 'unknown')} "
                    f"v{server_info.get('version', '?')}），工具：{tool_names}"
                )
            else:
                lines.append(f"- **{name}** {status}（已注册，尚未连接）")
        return ToolResult.ok("\n".join(lines))


class ConnectMcpServerTool(ToolBase):
    name = "connect_mcp_server"
    description = (
        "运行时动态连接一个 MCP 服务器。连接成功后，其提供的工具会按展开模式自动注册到工具表，"
        "模型可以直接调用；细粒度模式下每个远端工具独立可见，粗粒度模式下通过 mcp__<name> 门脸工具操作。"
    )
    args_schema = ConnectMcpInput
    tags = ["mcp"]

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def run(self, name: str, target: str, transport: str = "auto", expand: str = "fine") -> Union[str, ToolResult]:
        registry = self.agent.mcp_registry
        if registry is None:
            return ToolResult.fail("MCP 系统未启用（enable_mcp=False），无法动态连接。")

        if not name or not name.replace("_", "").replace("-", "").isalnum():
            return ToolResult.fail("name 只能包含字母、数字、下划线、连字符")
        if name in registry.clients:
            return ToolResult.fail(f"MCP server '{name}' 已连接；如需重连请先 disconnect_mcp_server")
        if expand not in ("fine", "coarse"):
            return ToolResult.fail(f"expand 只能是 'fine' 或 'coarse'，收到：{expand}")

        ttype = transport.lower()
        if ttype == "auto":
            ttype = "sse" if target.startswith(("http://", "https://")) else "stdio"
        try:
            if ttype == "sse":
                registry.add_sse(name, target)
            elif ttype == "stdio":
                registry.add_stdio_cmd(name, target)
            else:
                return ToolResult.fail(f"不支持的传输类型：{transport}")
            added_names = registry.connect_and_expand(
                name, self.agent.tool_registry, expand=expand,
            )
        except Exception as e:
            registry._factories.pop(name, None)
            return ToolResult.fail(f"连接失败：{e}")

        try:
            self.agent.refresh_system_prompt()
        except Exception:
            pass

        mode_text = "细粒度" if expand == "fine" else "粗粒度"
        return ToolResult.ok(
            f"✅ 已连接 MCP server '{name}'（{mode_text}展开），注册了 {len(added_names)} 个工具："
            f"{', '.join(added_names)}"
        )


class DisconnectMcpServerTool(ToolBase):
    name = "disconnect_mcp_server"
    description = "断开一个已连接的 MCP 服务器，其提供的工具会从工具表中移除。"
    args_schema = DisconnectMcpInput
    tags = ["mcp"]

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def run(self, name: str) -> Union[str, ToolResult]:
        registry = self.agent.mcp_registry
        if registry is None:
            return ToolResult.fail("MCP 系统未启用（enable_mcp=False）。")
        if name not in registry.clients:
            return ToolResult.fail(f"MCP server '{name}' 当前未连接")

        removed = registry.close_and_remove(name, self.agent.tool_registry)
        # 也要把 factory 里残留的清掉（重新 connect 时再显式 add）
        registry._factories.pop(name, None)
        try:
            self.agent.refresh_system_prompt()
        except Exception:
            pass
        return ToolResult.ok(f"已断开 '{name}'，移除 {len(removed)} 个工具：{', '.join(removed)}")


def build_mcp_tools(agent) -> List[ToolBase]:
    """一次性创建三个 MCP 管理工具（对齐 build_memory_tools 风格）。

    Args:
        agent: ReActAgent 实例；工具会通过 agent.mcp_registry 管理连接，
               并通过 agent.tool_registry 注册/卸载展开后的工具。
    """
    return [
        ListMcpServersTool(agent),
        ConnectMcpServerTool(agent),
        DisconnectMcpServerTool(agent),
    ]
