"""MCP 多 server 注册表 —— 管理多连接、批量 connect/close/tool 展开。"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from .client import MCPClient
from .errors import MCPError
from .models import (
    MCPConfig, ToolInfo,
    ResourceInfo, ResourceTemplateInfo,
    PromptInfo,
)
from .tool import MCPToolAdapter, MCPServerTool
from .transports import (
    SSETransport, StdioTransport, StreamableHTTPTransport, make_stdio_transport,
)

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
        self._resources: Dict[str, List[ResourceInfo]] = {}
        self._resource_templates: Dict[str, List[ResourceTemplateInfo]] = {}
        self._prompts: Dict[str, List[PromptInfo]] = {}
        # 记录每个 server 当前注册出去的工具名（用于断开 / 重新展开时清理）
        self._registered_names: Dict[str, List[str]] = {}
        self._expand_mode: Dict[str, str] = {}
        self._prefix_sep: Dict[str, str] = {}
        # v3.3.6：记住各 server 已注册的粗粒度门脸工具，tools/list_changed 时同步其内部索引
        self._coarse_tools: Dict[str, MCPServerTool] = {}
        self._closed = False

    # ── 注册 ─────────────────────────────────────────────────

    def add_config(self, config: MCPConfig) -> None:
        """根据 MCPConfig 添加一个 server（声明式）。"""
        if config.transport == "stdio":
            if config.command_str is not None:
                # v3.3.5 缺陷修复：此前本分支未透传 config.env，声明式配置的环境变量被静默丢弃
                self.add_stdio_cmd(config.name, config.command_str, cwd=config.cwd, env=config.env)
            elif config.command is not None:
                self.add_stdio(config.name, config.command, args=config.args,
                               env=config.env, cwd=config.cwd)
            else:
                raise MCPError(f"MCPConfig '{config.name}' (stdio) 需要 command 或 command_str")
        elif config.transport == "sse":
            if not config.url:
                raise MCPError(f"MCPConfig '{config.name}' (sse) 需要 url")
            self.add_sse(config.name, config.url, headers=config.headers, timeout=config.timeout)
        elif config.transport == "streamable_http":
            if not config.url:
                raise MCPError(f"MCPConfig '{config.name}' (streamable_http) 需要 url")
            self.add_streamable_http(
                config.name, config.url, headers=config.headers, timeout=config.timeout,
            )
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

    def add_stdio_cmd(self, name: str, command_str: str, cwd: Optional[str] = None,
                      env: Optional[Dict[str, str]] = None) -> None:
        """便捷方法：直接传命令字符串，自动 shlex.split（Windows 用 posix=False）。

        env：额外环境变量（v3.3.5 新增可选参数，默认 None 行为不变），透传到子进程。
        """
        t = make_stdio_transport(command_str, cwd=cwd, env=env)
        self._factories[name] = lambda t=t: MCPClient(t)

    def add_sse(self, name: str, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> None:
        def factory(url=url, headers=headers, timeout=timeout):
            t = SSETransport(url=url, headers=headers, timeout=timeout)
            return MCPClient(t)
        self._factories[name] = factory

    def add_streamable_http(self, name: str, url: str, headers: Optional[Dict[str, str]] = None,
                            timeout: float = 30.0) -> None:
        """注册一个 Streamable HTTP 类型的 server（单端点 HTTP，MCP 2025 规范）。"""
        def factory(url=url, headers=headers, timeout=timeout):
            t = StreamableHTTPTransport(url=url, headers=headers, timeout=timeout)
            return MCPClient(t)
        self._factories[name] = factory

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self, name: str, timeout: float = 10.0) -> MCPClient:
        # v3.3.6 修复：_closed 此前只写不读（死标记）；现在 close_all() 后拒绝再连接，
        # 避免拿到一批已被关闭的旧 client / 误以为还能新接入 server。
        if self._closed:
            raise MCPError("MCPRegistry 已被 close_all() 关闭，拒绝连接新 server；请新建 MCPRegistry 实例")
        if name in self._clients:
            return self._clients[name]
        if name not in self._factories:
            raise MCPError(f"MCP server '{name}' 未注册")
        client = self._factories[name]()
        init = client.connect(timeout=timeout)
        self._clients[name] = client
        # v3.3.7 缺陷修复（发现1）：缓存加载任一步失败时整体回滚——此前 client 会作为
        # 「已连接但没有工具/资源缓存」的半初始化脏状态残留在 _clients 里（上层误判可用、
        # register_into 注册空门脸），现在失败即关闭并摘表后再上抛。
        try:
            self._tools[name] = client.list_tools(timeout=timeout)
            # 加载资源和提示模板缓存（服务器不支持对应能力时自动返回空列表）
            self._resources[name] = client.list_resources(timeout=timeout)
            self._resource_templates[name] = client.list_resource_templates(timeout=timeout)
            self._prompts[name] = client.list_prompts(timeout=timeout)
        except Exception:
            self.close(name)
            raise
        # 注册 list_changed 通知处理器，服务端工具/资源/提示变更时自动刷新本地缓存
        # （处理器由客户端读线程同步回调，刷新必须甩到独立线程，见 _schedule_refresh）
        caps = init.capabilities
        if caps.tools and caps.tools.get("listChanged"):
            client.on_notification("notifications/tools/list_changed", lambda _p, n=name: self._schedule_refresh(n, "tools"))
        if caps.resources and caps.resources.get("listChanged"):
            client.on_notification("notifications/resources/list_changed", lambda _p, n=name: self._schedule_refresh(n, "resources"))
        if caps.prompts and caps.prompts.get("listChanged"):
            client.on_notification("notifications/prompts/list_changed", lambda _p, n=name: self._schedule_refresh(n, "prompts"))
        logger.info(
            "MCP[%s] 已连接：%s v%s，工具 %d 个，资源 %d 个，提示模板 %d 个",
            name, init.server_info.get("name", "unknown"),
            init.server_info.get("version", "?"), len(self._tools[name]),
            len(self._resources[name]), len(self._prompts[name]),
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
        self._resources.pop(name, None)
        self._resource_templates.pop(name, None)
        self._prompts.pop(name, None)
        self._registered_names.pop(name, None)
        self._expand_mode.pop(name, None)
        self._prefix_sep.pop(name, None)
        self._coarse_tools.pop(name, None)

    def close_all(self) -> None:
        for name in list(self._clients.keys()):
            self.close(name)
        self._closed = True

    # ── list_changed 缓存自动刷新 ────────────────────────────

    def _schedule_refresh(self, server_name: str, kind: str) -> None:
        """v3.3.6 根因修复：list_changed 处理器由客户端读线程同步回调，而 _refresh_*
        内部会再发起新请求——响应只有读线程自己能消费，同步执行就是自死锁直到超时
        （探针实证：刷新永远报「请求 tools/list 超时」，v3.2 的自动刷新从未生效）。
        故把刷新甩到独立 daemon 线程，处理器立即返回；失败仍由 _refresh_* 内部
        try/except 兜底，不阻塞主流程。"""
        target = {"tools": self._refresh_tools,
                  "resources": self._refresh_resources,
                  "prompts": self._refresh_prompts}[kind]
        threading.Thread(target=target, args=(server_name,),
                         name=f"mcp-refresh-{server_name}-{kind}", daemon=True).start()

    def _refresh_tools(self, server_name: str) -> None:
        """服务端推送 tools/list_changed 时刷新本地工具缓存，不自动重注册工具（避免打断正在执行的调用）。"""
        client = self._clients.get(server_name)
        if not client:
            return
        try:
            self._tools[server_name] = client.list_tools(timeout=5.0)
            # v3.3.6 修复：同步已注册粗粒度门脸的工具索引——此前门脸只存注册时的旧快照，
            # 刷新后新增子工具永远报「不存在」（半设计半坑：不重注册是有意为之，但索引过期不是）。
            gate = self._coarse_tools.get(server_name)
            if gate is not None and self._expand_mode.get(server_name) == "coarse":
                gate.update_tools(self._tools[server_name])
            logger.info("MCP[%s] 工具列表已刷新，共 %d 个工具", server_name, len(self._tools[server_name]))
        except Exception as e:
            logger.warning("MCP[%s] 刷新工具列表失败：%s", server_name, e)

    def _refresh_resources(self, server_name: str) -> None:
        """服务端推送 resources/list_changed 时刷新本地资源缓存。"""
        client = self._clients.get(server_name)
        if not client:
            return
        try:
            self._resources[server_name] = client.list_resources(timeout=5.0)
            self._resource_templates[server_name] = client.list_resource_templates(timeout=5.0)
            logger.info("MCP[%s] 资源列表已刷新，共 %d 个资源", server_name, len(self._resources[server_name]))
        except Exception as e:
            logger.warning("MCP[%s] 刷新资源列表失败：%s", server_name, e)

    def _refresh_prompts(self, server_name: str) -> None:
        """服务端推送 prompts/list_changed 时刷新本地提示模板缓存。"""
        client = self._clients.get(server_name)
        if not client:
            return
        try:
            self._prompts[server_name] = client.list_prompts(timeout=5.0)
            logger.info("MCP[%s] 提示模板列表已刷新，共 %d 个模板", server_name, len(self._prompts[server_name]))
        except Exception as e:
            logger.warning("MCP[%s] 刷新提示模板列表失败：%s", server_name, e)

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

    def list_resources(self, server: Optional[str] = None) -> Dict[str, List[ResourceInfo]]:
        """查询已连接 server 的资源列表，可指定单个 server 或返回全部。"""
        if server:
            return {server: self._resources.get(server, [])}
        return dict(self._resources)

    def list_resource_templates(self, server: Optional[str] = None) -> Dict[str, List[ResourceTemplateInfo]]:
        """查询已连接 server 的参数化资源模板列表。"""
        if server:
            return {server: self._resource_templates.get(server, [])}
        return dict(self._resource_templates)

    def list_prompts(self, server: Optional[str] = None) -> Dict[str, List[PromptInfo]]:
        """查询已连接 server 的提示模板列表。"""
        if server:
            return {server: self._prompts.get(server, [])}
        return dict(self._prompts)

    def read_resource(self, server_name: str, uri: str, timeout: float = 30.0):
        """按 URI 读取指定 server 的资源。"""
        client = self._clients.get(server_name)
        if not client:
            raise MCPError(f"MCP server '{server_name}' 未连接")
        return client.read_resource(uri, timeout=timeout)

    def get_prompt(self, server_name: str, prompt_name: str, arguments: Optional[Dict[str, str]] = None,
                   timeout: float = 30.0):
        """按名称渲染指定 server 的提示模板。"""
        client = self._clients.get(server_name)
        if not client:
            raise MCPError(f"MCP server '{server_name}' 未连接")
        return client.get_prompt(prompt_name, arguments=arguments, timeout=timeout)

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
            self._coarse_tools[server_name] = t  # v3.3.6：留引用供 list_changed 同步索引
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
        # 兜底：扫描前缀匹配的残留（比如之前用其它模式注册过）。
        # v3.3.6 修复：分隔符改用该 server 上次展开时记录的 prefix_sep（默认 "__"），
        # 此前硬编码 "__"，用户用自定义分隔符注册的历史残留清不掉。
        sep = self._prefix_sep.get(server_name, "__")
        prefixes = [f"{server_name}{sep}", f"mcp{sep}{server_name}"]
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
