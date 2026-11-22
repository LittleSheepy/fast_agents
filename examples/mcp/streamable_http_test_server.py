"""Streamable HTTP 测试服务器 —— 用标准库 http.server 实现一个最小 MCP server。

支持两种响应模式（通过 Accept 头）：
- 客户端只 Accept application/json：所有请求直接返回 JSON（短响应模式）。
- 客户端 Accept 含 text/event-stream：initialize 返回 JSON，工具相关请求返回 SSE 流。
用于验证 StreamableHTTPTransport 在两种模式下都能跑通握手 + list_tools + call_tool。

用法：python streamable_http_test_server.py [port]
默认端口 9101。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "demo-streamable-http-server"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {
        "name": "echo",
        "description": "回显输入的文本",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文本"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "calculate",
        "description": "简单四则运算（a + b）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静默日志
        return

    # ── 路由 ─────────────────────────────────────────────────

    def do_DELETE(self):
        # 规范要求：客户端关闭时发 DELETE 结束会话
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        # 仅用于 SSE 重连：返回一个空的 SSE 流（立即关闭），让重连成功但无消息
        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            self.send_response(406)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        sid = self.headers.get("Mcp-Session-Id", "sess-1")
        self.send_header("Mcp-Session-Id", sid)
        self.end_headers()
        self.wfile.write(b": keepalive\n\n")
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            msg = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json(400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            return

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        # Session-Id 回显：initialize 返回固定 id
        session_id = self.headers.get("Mcp-Session-Id") or "sess-1"

        accept = self.headers.get("Accept", "")
        use_sse = "text/event-stream" in accept

        result = self._handle(method, params)

        if req_id is None:
            # notification：不响应
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        resp_body = {"jsonrpc": "2.0", "id": req_id, "result": result}

        if use_sse and method in ("tools/list", "tools/call"):
            self._send_sse(session_id, resp_body)
        else:
            self._send_json(200, resp_body, session_id=session_id, ctype="application/json")

    # ── 业务 ─────────────────────────────────────────────────

    def _handle(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {"listChanged": True}},
                "instructions": "测试 streamable http server",
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            if name == "echo":
                return {
                    "content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}],
                    "isError": False,
                }
            if name == "calculate":
                s = float(args.get("a", 0)) + float(args.get("b", 0))
                return {
                    "content": [{"type": "text", "text": f"result: {s}"}],
                    "isError": False,
                }
            return {
                "content": [{"type": "text", "text": f"未知工具: {name}"}],
                "isError": True,
            }
        return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    # ── 响应工具 ─────────────────────────────────────────────

    def _send_json(self, status: int, body: Dict[str, Any],
                   session_id: Optional[str] = None, ctype: str = "application/json"):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _send_sse(self, session_id: str, body: Dict[str, Any]):
        """发送一条 SSE 事件然后关闭连接（简单测试用）。

        使用 chunked Transfer-Encoding，避免 Content-Length 计算 + Connection: close
        组合下 BaseHTTPServer 在某些客户端的连接处理问题。
        """
        payload = json.dumps(body, ensure_ascii=False)
        data_part = b"data: " + payload.encode("utf-8") + b"\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # 手动写 chunked：size\r\n data\r\n 0\r\n\r\n
        try:
            self.wfile.write(("%x\r\n" % len(data_part)).encode("ascii"))
            self.wfile.write(data_part)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass


def serve(port: int = 9101) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9101
    print(f"[streamable-http-test-server] listening on http://127.0.0.1:{port}/mcp", flush=True)
    server = serve(port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
