"""单个 MCP server 客户端：握手 / 工具发现 / 工具调用。"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from .errors import MCPConnectionError, MCPCallError, MCPProtocolError
from .models import CallResult, InitializeResult, ServerCapabilities, ToolInfo
from .transports import Transport


class MCPClient:
    """对应一个已连接的 MCP server。生命周期：connect → initialize → list_tools/call_tool → close。"""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, transport: Transport, client_name: str = "fast-agents", client_version: str = "0.1.0"):
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self._id = 0
        self._lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._request_id_to_response: Dict[int, Any] = {}
        self._event: threading.Event = threading.Event()
        self.connected = False
        self.initialize_result: Optional[InitializeResult] = None

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> InitializeResult:
        self.transport.connect()
        self.connected = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        # initialize
        init_result = self._request("initialize", {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": self.client_name, "version": self.client_version},
        }, timeout=timeout)
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
        self._notify("notifications/initialized", {})
        return self.initialize_result

    def close(self) -> None:
        self.connected = False
        try:
            self.transport.close()
        except Exception:
            pass

    # ── 工具发现与调用 ───────────────────────────────────────

    def list_tools(self, timeout: float = 10.0) -> List[ToolInfo]:
        if not self.connected:
            raise MCPConnectionError("未连接到 MCP server")
        resp = self._request("tools/list", {}, timeout=timeout)
        tools_data = resp.get("tools", [])
        return [ToolInfo(name=t["name"], description=t.get("description", ""), input_schema=t.get("inputSchema", {}))
                for t in tools_data]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> CallResult:
        if not self.connected:
            raise MCPConnectionError("未连接到 MCP server")
        try:
            resp = self._request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        except MCPCallError:
            raise
        return CallResult(
            content=resp.get("content", []),
            is_error=resp.get("isError", False),
            structured_content=resp.get("structuredContent"),
        )

    # ── 底层 RPC ─────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _request(self, method: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        req_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        with self._lock:
            self._request_id_to_response[req_id] = None
            self.transport.send(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                data = self._request_id_to_response.get(req_id)
                if data is not None:
                    self._request_id_to_response.pop(req_id, None)
                    if "error" in data:
                        err = data["error"]
                        raise MCPCallError(f"远端错误 {err.get('code')}: {err.get('message')}")
                    return data.get("result", {})
            time.sleep(0.01)
        with self._lock:
            self._request_id_to_response.pop(req_id, None)
        raise MCPConnectionError(f"请求 {method} 超时（{timeout}s）")

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.transport.send(msg)
        except Exception:
            pass

    def _read_loop(self) -> None:
        while self.connected:
            msg = self.transport.recv(timeout=0.2)
            if msg is None:
                continue
            # 响应（带 id）→ 按 id 分发
            if "id" in msg and ("result" in msg or "error" in msg):
                with self._lock:
                    if msg["id"] in self._request_id_to_response:
                        self._request_id_to_response[msg["id"]] = msg
                        continue
            # 请求（server 主动发）→ 简化：notifications 忽略，方法请求回 method not found
            if "method" in msg and "id" in msg:
                self.transport.send({
                    "jsonrpc": "2.0", "id": msg["id"],
                    "error": {"code": -32601, "message": f"Method not found: {msg['method']}"},
                })
