"""MCP 多 server 注册表 —— 管理多连接、批量 connect/close/tool 展开。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .client import MCPClient
from .errors import MCPError
from .models import MCPConfig, ToolInfo
from .tool import MCPToolAdapter, MCPServerTool
from .transports import SSETransport, StdioTransport, make_stdio

logger = logging.getLogger(__name__)


class MCPRegistry:
    """管理多个 MCP server。

    用法::

        reg = MCPRegistry()
        reg.add_stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
        reg.add_sse("remote", "https://example.com/mcp")
        reg.connect_all()
        reg.register_into(agent.tool_registry, expand="fine")   # 细粒度（推荐）
    """

    def __init__(self):
        self._factories: Dict[str, Callable[[], MCPClient]] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, List[ToolInfo]] = {}
        # 记录每个 server 当前注册出去的工具名（用于断开 / 重新展开时清理）
        self._registered_names: Dict[str, List[str]] = {}
        self._expand_mode: Dict[str, str] = {}
        self._prefix_sep: Dict[str, str] = {}
        self._closed = False

    # ── 注册 ─────────────────────────────────────────────────

    def add_config(self, config: MCPConfig) -> None:
        """根据 MCPConfig 添加一个 server（声明式）。"""
        if config.transport == "stdio":
            if config.command_str is not None:
                self.add_stdio_cmd(config.name, config.command_str, cwd=config.cwd)
            elif config.command is not None:
                self.add_stdio(config.name, config.command, args=config.args,
                               env=config.env, cwd=config.cwd)
            else:
                raise MCPError(f"MCPConfig '{config.name}' (stdio) 需要 command 或 command_str")
        elif config.transport == "sse":
            if not config.url:
                raise MCPError(f"MCPConfig '{config.name}' (sse) 需要 url")
            self.add_sse(config.name, config.url, headers=config.headers, timeout=config.timeout)
        else:
            raise MCPError(f"未知 transport: {config.transport}")

    def add_configs(self, configs: List[MCPConfig]) -> None:
        for c in configs:
            self.add_config(c)

    def add_stdio(self, name: str, command: str, args: Optional[List[str]] = None,
                  env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> None:
        """注册一个 stdio 类型的 server。"""
        def factory(command=command, args=args, env=env, cwd=cwd):
            t = StdioTransport(command=command, args=args, env=env, cwd=cwd)
            return MCPClient(t)
        self._factories[name] = factory

    def add_stdio_cmd(self, name: str, command_str: str, cwd: Optional[str] = None) -> None:
        """便捷方法：直接传命令字符串，自动 shlex.split（Windows 用 posix=False）。"""
        t = make_stdio(command_str, cwd=cwd)
        self._factories[name] = lambda t=t: MCPClient(t)

    def add_sse(self, name: str, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> None:
        def factory(url=url, headers=headers, timeout=timeout):
            t = SSETransport(url=url, headers=headers, timeout=timeout)
            return MCPClient(t)
        self._factories[name] = factory

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self, name: str, timeout: float = 10.0) -> MCPClient:
        if name in self._clients:
            return self._clients[name]
        if name not in self._factories:
            raise MCPError(f"MCP server '{name}' 未注册")
        client = self._factories[name]()
        init = client.connect(timeout=timeout)
        self._clients[name] = client
        self._tools[name] = client.list_tools(timeout=timeout)
        logger.info(
            "MCP[%s] 已连接：%s v%s，工具 %d 个",
            name, init.server_info.get("name", "unknown"),
            init.server_info.get("version", "?"), len(self._tools[name]),
        )
        return client

    def connect_all(self, timeout: float = 10.0) -> None:
        for name in list(self._factories.keys()):
            try:
                self.connect(name, timeout=timeout)
            except Exception as e:
                logger.warning("MCP[%s] 连接失败：%s", name, e)

    def close(self, name: str) -> None:
        c = self._clients.pop(name, None)
        if c:
            c.close()
        self._tools.pop(name, None)
        self._registered_names.pop(name, None)
        self._expand_mode.pop(name, None)
        self._prefix_sep.pop(name, None)

    def close_all(self) -> None:
        for name in list(self._clients.keys()):
            self.close(name)
        self._closed = True

    # ── 查询 ─────────────────────────────────────────────────

    @property
    def clients(self) -> Dict[str, MCPClient]:
        return dict(self._clients)

    @property
    def factories(self) -> Dict[str, Callable]:
        return dict(self._factories)

    def get_client(self, name: str) -> Optional[MCPClient]:
        return self._clients.get(name)

    def list_tools(self, server: Optional[str] = None) -> Dict[str, List[ToolInfo]]:
        if server:
            return {server: self._tools.get(server, [])}
        return dict(self._tools)

    def _tool_names_for(self, server_name: str, tools: List[ToolInfo],
                        expand: str, prefix_sep: str) -> List[str]:
        if expand == "fine":
            return [f"{server_name}{prefix_sep}{t.name}" for t in tools]
        if expand == "coarse":
            return [f"mcp{prefix_sep}{server_name}"]
        raise MCPError(f"expand 必须是 'fine' 或 'coarse'，收到：{expand}")

    # ── 注册进 ToolRegistry ──────────────────────────────────

    def _expand_one(self, registry, server_name: str, expand: str, prefix_sep: str) -> List:
        client = self._clients[server_name]
        tools = self._tools.get(server_name, [])
        registered = []
        if expand == "fine":
            for info in tools:
                t = MCPToolAdapter(client=client, server_name=server_name, tool_info=info, prefix_sep=prefix_sep)
                registry.register_tool(t)
                registered.append(t)
        elif expand == "coarse":
            t = MCPServerTool(client=client, server_name=server_name, tools=tools, prefix_sep=prefix_sep)
            registry.register_tool(t)
            registered.append(t)
        else:
            raise MCPError(f"expand 必须是 'fine' 或 'coarse'，收到：{expand}")
        return registered

    def register_into(self, registry, prefix_sep: str = "__",
                      expand: str = "fine") -> List:
        """把所有已连接 server 的工具注册到 ToolRegistry。

        Args:
            registry: 你的 fast_agents ToolRegistry。
            prefix_sep: 工具名前缀分隔符，默认 "__"（如 fs__read_file）。
            expand: "fine"（默认）= 细粒度展开，每个 MCP 工具独立一个 ToolBase，
                    参数 schema 完整，生产推荐；
                    "coarse" = 粗粒度门脸工具，每个 server 一个 MCPServerTool，
                    模型通过 action=list_tools/call_tool 操作，省工具表长度。
        """
        registered = []
        for server_name in self._clients.keys():
            # 如果已注册过，先卸载旧的（避免重复）
            self._unregister_one(registry, server_name)
            added = self._expand_one(registry, server_name, expand, prefix_sep)
            self._registered_names[server_name] = self._tool_names_for(
                server_name, self._tools.get(server_name, []), expand, prefix_sep,
            )
            self._expand_mode[server_name] = expand
            self._prefix_sep[server_name] = prefix_sep
            registered.extend(added)
        return registered

    def unregister_from(self, registry, server_name: Optional[str] = None) -> List[str]:
        """从 ToolRegistry 中卸载已注册的 MCP 工具（不关闭连接）。

        Args:
            registry: 目标 ToolRegistry。
            server_name: 指定 server；不传则卸载全部。
        Returns:
            被卸载的工具名列表。
        """
        targets = [server_name] if server_name else list(self._registered_names.keys())
        removed_all: List[str] = []
        for sn in targets:
            removed_all.extend(self._unregister_one(registry, sn))
        return removed_all

    def _unregister_one(self, registry, server_name: str) -> List[str]:
        names = self._registered_names.get(server_name, [])
        removed = []
        for name in names:
            if registry.has_tool(name):
                if registry.unregister(name):
                    removed.append(name)
        # 兜底：扫描前缀匹配的残留（比如之前用其它模式注册过）
        prefixes = [f"{server_name}__", f"mcp__{server_name}"]
        for tool_name in list(registry.list_names()):
            for pre in prefixes:
                if tool_name.startswith(pre) and tool_name not in removed:
                    if registry.unregister(tool_name):
                        removed.append(tool_name)
        if server_name in self._registered_names:
            self._registered_names[server_name] = []
        return removed

    # ── 动态连接（供 builtin connect_mcp_server 等场景使用）───

    def connect_and_expand(self, name: str, registry,
                           expand: str = "fine",
                           prefix_sep: str = "__",
                           timeout: float = 10.0) -> List[str]:
        """连接单个 server 并立即展开到 ToolRegistry。

        用于运行时动态添加 MCP server 的场景（内置 connect_mcp_server 工具）。
        返回本次注册的工具名列表。
        """
        self.connect(name, timeout=timeout)
        self._unregister_one(registry, name)
        self._expand_one(registry, name, expand, prefix_sep)
        names = self._tool_names_for(name, self._tools.get(name, []), expand, prefix_sep)
        self._registered_names[name] = names
        self._expand_mode[name] = expand
        self._prefix_sep[name] = prefix_sep
        return names

    def close_and_remove(self, name: str, registry) -> List[str]:
        """断开 server 并从 ToolRegistry 中卸载其工具。返回被卸载的工具名列表。"""
        removed = self._unregister_one(registry, name)
        self.close(name)
        return removed
