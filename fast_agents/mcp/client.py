"""单个 MCP server 客户端：握手 / 工具发现 / 工具调用。

v3.1 内部升级（对外 API 完全兼容）：
- 请求-响应同步改用「每个 pending 请求一个 threading.Event」，读线程收到响应后 set() 唤醒，
  消除原先 time.sleep(0.01) 的忙等轮询。
- 状态机（disconnected/connecting/connected/closing/closed）保证 connect/close 幂等、线程安全。
- notification（无 id 的消息）正确分发给注册的 handler，不再错误返回 MethodNotFound。
- close() 串行化 + 等待读线程优雅退出，不再遗留僵尸线程。
- 删除死代码 _event。
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .errors import MCPConnectionError, MCPCallError
from .models import (
    CallResult, InitializeResult, ServerCapabilities, ToolInfo,
    ResourceInfo, ResourceTemplateInfo, ReadResourceResult,
    PromptInfo, PromptMessage, GetPromptResult,
)
from .transports import TransportBase

logger = logging.getLogger(__name__)


class _ClientState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    CLOSED = "closed"


class MCPClient:
    """对应一个已连接的 MCP server。生命周期：connect → initialize → list_tools/call_tool → close。"""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, transport: TransportBase, client_name: str = "fast-agents", client_version: str = "0.1.0"):
        self.transport: TransportBase = transport
        self.client_name = client_name
        self.client_version = client_version

        # 请求 id 生成
        self._id = 0
        self._id_lock = threading.Lock()

        # 待响应请求表：req_id -> (event, response_container)
        # 每个请求独立一个 Event，读线程收到响应后 set() 唤醒对应的请求线程。
        self._pending: Dict[int, "_PendingRequest"] = {}
        self._pending_lock = threading.Lock()

        # 状态机 + 读线程
        self._state = _ClientState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()  # 串行化 close()，保证多线程下幂等
        self._reader: Optional[threading.Thread] = None
        self._reader_exited = threading.Event()  # 读线程退出信号，close() 等待它

        # notification 处理器注册表：method -> [handler(params)]
        self._notification_handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._handlers_lock = threading.Lock()

        # 对外兼容：保留 connected 属性语义，通过 property 暴露
        self.initialize_result: Optional[InitializeResult] = None

    # ── 公开属性（兼容外部访问 self.connected） ─────────────────

    @property
    def connected(self) -> bool:
        """是否已完成 initialize 握手、处于可用状态。"""
        return self._state == _ClientState.CONNECTED

    @property
    def state(self) -> str:
        """当前连接状态字符串（disconnected/connecting/connected/closing/closed），只读，便于观测。"""
        return self._state.value

    # ── notification handler 注册（新增公开方法，不改旧 API） ───

    def on_notification(self, method: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """注册服务器推送 notification 的处理器。

        常见 method:
            notifications/tools/list_changed
            notifications/resources/list_changed
            notifications/resources/updated
            notifications/prompts/list_changed
            notifications/message (logging)

        handler 在**读线程**中同步调用，应快速返回或自行起线程处理耗时逻辑。
        """
        with self._handlers_lock:
            self._notification_handlers.setdefault(method, []).append(handler)

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> InitializeResult:
        """连接到 server 并完成 initialize 握手。幂等：已连接时直接返回已有结果。"""
        with self._state_lock:
            if self._state == _ClientState.CONNECTED:
                assert self.initialize_result is not None
                return self.initialize_result
            if self._state in (_ClientState.CONNECTING, _ClientState.CLOSING):
                raise MCPConnectionError(f"当前状态 {self._state.value}，不能调用 connect")
            # DISCONNECTED / CLOSED → CONNECTING
            self._state = _ClientState.CONNECTING
            self._reader_exited.clear()

        try:
            # 启动 transport
            self.transport.connect()

            # 启动读线程
            self._reader = threading.Thread(
                target=self._read_loop, name=f"mcp-reader-{id(self):x}", daemon=True
            )
            self._reader.start()

            # initialize 握手（_request 内部用 Event 等待响应，不会忙等）
            init_result = self._request("initialize", {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            }, timeout=timeout)
        except Exception:
            # 握手失败，回滚状态并尝试关闭 transport
            with self._state_lock:
                self._state = _ClientState.DISCONNECTED
            try:
                self.transport.close()
            except Exception:
                pass
            raise

        # 解析握手结果
        capabilities_data = init_result.get("capabilities", {})
        caps = ServerCapabilities(
            tools=capabilities_data.get("tools"),
            resources=capabilities_data.get("resources"),
            prompts=capabilities_data.get("prompts"),
            logging=capabilities_data.get("logging"),
            extra={k: v for k, v in capabilities_data.items()
                   if k not in ("tools", "resources", "prompts", "logging")},
        )
        self.initialize_result = InitializeResult(
            protocol_version=init_result.get("protocolVersion", self.PROTOCOL_VERSION),
            server_info=init_result.get("serverInfo", {}),
            capabilities=caps,
            instructions=init_result.get("instructions", ""),
        )

        # 发送 initialized 通知，标记为已连接
        self._notify("notifications/initialized", {})
        with self._state_lock:
            self._state = _ClientState.CONNECTED
        return self.initialize_result

    def close(self) -> None:
        """关闭连接并等待读线程退出。幂等：多次调用、多线程并发调用都安全。"""
        with self._close_lock:
            with self._state_lock:
                if self._state in (_ClientState.CLOSED, _ClientState.DISCONNECTED):
                    return
                self._state = _ClientState.CLOSING

            # 先唤醒所有 pending 请求，让它们拿到"连接已关闭"错误，避免无限等待
            self._fail_all_pending(MCPConnectionError("MCP 连接已关闭"))

            # 关闭 transport（会让读线程的 recv 退出或抛异常，从而退出循环）
            try:
                self.transport.close()
            except Exception:
                pass

            # 等待读线程优雅退出（最多 2 秒，超时不强行处理，因为是 daemon 线程）。
            # 若 close() 恰好在读线程内被调用（如 notification handler 里），跳过 join 避免自等。
            reader = self._reader
            if reader is not None and reader.is_alive() and reader is not threading.current_thread():
                reader.join(timeout=2.0)

            with self._state_lock:
                self._state = _ClientState.CLOSED

    # ── 工具发现与调用 ───────────────────────────────────────

    def list_tools(self, timeout: float = 10.0) -> List[ToolInfo]:
        """列出服务端提供的全部工具。服务器未声明 tools 能力时返回空列表
        （v3.3.7 修复发现1：对齐 list_resources / list_prompts 的能力保护——
        合规的无 tools 能力 server 对本方法必回 -32601，此前会直接报错打断 registry.connect）。"""
        self._ensure_connected()
        if self.initialize_result is None or self.initialize_result.capabilities.tools is None:
            return []
        resp = self._request("tools/list", {}, timeout=timeout)
        tools_data = resp.get("tools", [])
        return [ToolInfo(name=t["name"], description=t.get("description", ""), input_schema=t.get("inputSchema", {}))
                for t in tools_data]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> CallResult:
        self._ensure_connected()
        resp = self._request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        return CallResult(
            content=resp.get("content", []),
            is_error=resp.get("isError", False),
            structured_content=resp.get("structuredContent"),
        )

    # ── 资源发现与读取（Resources）───────────────────────────

    def list_resources(self, timeout: float = 10.0) -> List["ResourceInfo"]:
        """列出服务端暴露的全部资源。服务器未声明 resources 能力时返回空列表。"""
        self._ensure_connected()
        if self.initialize_result is None or self.initialize_result.capabilities.resources is None:
            return []
        resp = self._request("resources/list", {}, timeout=timeout)
        return [ResourceInfo(**r) for r in resp.get("resources", [])]

    def list_resource_templates(self, timeout: float = 10.0) -> List["ResourceTemplateInfo"]:
        """列出服务端暴露的参数化资源模板（可按 URI 模板传入参数读取）。

        按 MCP 规范走独立方法 resources/templates/list（v3.3.3 修复：此前误复用
        resources/list，对合规服务器模板永远为空）。服务器声明了 resources 能力但
        未实现本方法（JSON-RPC -32601）时降级返回空列表，不打断 registry.connect 等上层流程。
        """
        self._ensure_connected()
        if self.initialize_result is None or self.initialize_result.capabilities.resources is None:
            return []
        try:
            resp = self._request("resources/templates/list", {}, timeout=timeout)
        except MCPCallError as e:
            if e.error_code == -32601:
                logger.debug("服务器未实现 resources/templates/list，降级为空模板列表")
                return []
            raise
        return [ResourceTemplateInfo(**t) for t in resp.get("resourceTemplates", [])]

    def read_resource(self, uri: str, timeout: float = 30.0) -> "ReadResourceResult":
        """按 URI 读取资源内容，支持文本资源和二进制（base64 blob）资源。"""
        self._ensure_connected()
        resp = self._request("resources/read", {"uri": uri}, timeout=timeout)
        return ReadResourceResult(contents=resp.get("contents", []))

    # ── 提示模板发现与渲染（Prompts）─────────────────────────

    def list_prompts(self, timeout: float = 10.0) -> List["PromptInfo"]:
        """列出服务端暴露的全部提示模板。服务器未声明 prompts 能力时返回空列表。"""
        self._ensure_connected()
        if self.initialize_result is None or self.initialize_result.capabilities.prompts is None:
            return []
        resp = self._request("prompts/list", {}, timeout=timeout)
        return [PromptInfo(**p) for p in resp.get("prompts", [])]

    def get_prompt(self, name: str, arguments: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> "GetPromptResult":
        """按名称渲染提示模板，arguments 为模板参数键值对，返回可直接注入对话的消息列表。"""
        self._ensure_connected()
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        resp = self._request("prompts/get", params, timeout=timeout)
        return GetPromptResult(
            description=resp.get("description", ""),
            messages=[PromptMessage(**m) for m in resp.get("messages", [])],
        )

    # ── 底层 RPC ─────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._state != _ClientState.CONNECTED:
            raise MCPConnectionError(f"未连接到 MCP server（当前状态：{self._state.value}）")

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _request(self, method: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """发送 JSON-RPC 请求并阻塞等待响应，基于 Event 唤醒（无忙等）。"""
        req_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        pending = _PendingRequest()
        with self._pending_lock:
            self._pending[req_id] = pending

        try:
            # 发送（transport.send 本身抛异常时直接向上抛）
            self.transport.send(msg)

            # 等待响应：Event.wait() 释放 GIL，不占用 CPU；被读线程 set() 唤醒
            if not pending.event.wait(timeout=timeout):
                # 超时：若期间连接确实被关闭，给出更准确的原因；否则报超时
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                if self._state in (_ClientState.CLOSING, _ClientState.CLOSED):
                    raise MCPConnectionError(f"请求 {method} 失败：连接已关闭")
                raise MCPConnectionError(f"请求 {method} 超时（{timeout}s）")

            # 被唤醒，看 response
            data = pending.response
            if data is None:
                # 被 _fail_all_pending 主动唤醒（连接关闭）
                raise MCPConnectionError(f"请求 {method} 失败：{pending.error or '连接已关闭'}")
        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

        if "error" in data:
            err = data["error"]
            raise MCPCallError(
                f"远端错误 {err.get('code')}: {err.get('message')}",
                error_code=err.get("code"),
            )
        return data.get("result", {})

    def _fail_all_pending(self, error: Exception) -> None:
        """连接关闭/出错时，唤醒所有等待中的请求并让它们抛出异常。"""
        with self._pending_lock:
            pendings = list(self._pending.values())
            self._pending.clear()
        for p in pendings:
            p.error = str(error)
            p.event.set()

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        """发送单向 notification（无 id，server 不会响应）。"""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.transport.send(msg)
        except Exception:
            # notification 发送失败通常只影响服务端状态感知，不阻塞主流程
            logger.debug("MCP notification %s 发送失败", method, exc_info=True)

    # ── 读线程 ─────────────────────────────────────────────

    def _read_loop(self) -> None:
        """读线程循环：从 transport.recv 收消息并分发。"""
        try:
            while self._state in (_ClientState.CONNECTING, _ClientState.CONNECTED):
                msg = self.transport.recv(timeout=0.2)
                if msg is None:
                    continue
                self._dispatch(msg)
        except Exception as e:
            # 读线程异常：连接已断开，唤醒所有 pending 请求报错
            logger.debug("MCP 读线程异常退出：%s", e)
            self._fail_all_pending(MCPConnectionError(f"读线程异常：{e}"))
        finally:
            self._reader_exited.set()

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        """分发一条收到的 JSON-RPC 消息。"""
        # 1) 响应（带 id 且有 result/error）→ 唤醒对应请求线程
        if "id" in msg and ("result" in msg or "error" in msg):
            req_id = msg["id"]
            with self._pending_lock:
                pending = self._pending.get(req_id)
            if pending is not None:
                pending.response = msg
                pending.event.set()
                return
            # 收到了我们没发过的 id 的响应：忽略并打 debug（可能是超时后迟到的响应）
            logger.debug("收到未知 id=%s 的响应，忽略", req_id)
            return

        # 2) server → client 的请求（有 method 且有 id，需要响应）→ 回 MethodNotFound
        if "method" in msg and "id" in msg:
            try:
                self.transport.send({
                    "jsonrpc": "2.0", "id": msg["id"],
                    "error": {"code": -32601, "message": f"Method not found: {msg['method']}"},
                })
            except Exception:
                logger.debug("回 MethodNotFound 失败：%s", msg.get("method"), exc_info=True)
            return

        # 3) notification（有 method 但无 id）→ 分发给注册的 handler，不回响应
        if "method" in msg and "id" not in msg:
            method = msg["method"]
            params = msg.get("params", {})
            with self._handlers_lock:
                handlers = list(self._notification_handlers.get(method, ()))
            if not handlers:
                # 未注册 handler 的 notification 默认忽略（这是正常情况，例如
                # notifications/initialized 的 ACK 我们不需要处理）
                return
            for h in handlers:
                try:
                    h(params)
                except Exception:
                    logger.debug("MCP notification handler %s 抛异常", method, exc_info=True)
            return

        # 4) 其他畸形消息
        logger.debug("收到无法识别的消息：%s", str(msg)[:200])


class _PendingRequest:
    """一个待响应请求的同步原语：event 用于等待/唤醒，response/error 用于传递结果。"""
    __slots__ = ("event", "response", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
