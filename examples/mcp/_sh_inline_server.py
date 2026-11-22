"""Streamable HTTP 测试服务器（线程内启动，供 test_mcp.py 使用）。

独立于 streamable_http_test_server.py（该脚本独立运行，使用固定端口 + subprocess），
本模块导出 start() 返回 (server, url)，在主线程内直接跑 ThreadingHTTPServer，
避免测试环境对子进程的限制。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOOLS = [
    {
        "name": "echo",
        "description": "回显文本",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "calculate",
        "description": "a+b",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs):
        return

    def do_DELETE(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            self.send_response(406)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Mcp-Session-Id", self.headers.get("Mcp-Session-Id", "sess-1"))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            msg = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json(400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            return
        method = msg.get("method", "")
        rid = msg.get("id")
        params = msg.get("params", {})
        sid = self.headers.get("Mcp-Session-Id") or "sess-1"
        accept = self.headers.get("Accept", "")
        use_sse = "text/event-stream" in accept

        if rid is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        result = self._handle(method, params)
        body = {"jsonrpc": "2.0", "id": rid, "result": result}
        if use_sse and method in ("tools/list", "tools/call"):
            self._send_sse(sid, body)
        else:
            self._send_json(200, body, sid)

    def _handle(self, method, params):
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "demo-sh", "version": "0.1"},
                "capabilities": {"tools": {"listChanged": True}},
                "instructions": "",
            }
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = params.get("name", "")
            a = params.get("arguments", {})
            if name == "echo":
                return {"content": [{"type": "text", "text": f"echo: {a.get('text', '')}"}], "isError": False}
            if name == "calculate":
                s = float(a.get("a", 0)) + float(a.get("b", 0))
                return {"content": [{"type": "text", "text": f"result: {s}"}], "isError": False}
            return {"content": [{"type": "text", "text": f"unknown: {name}"}], "isError": True}
        return {"error": {"code": -32601, "message": "not found"}}

    def _send_json(self, status, body, sid=None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if sid:
            self.send_header("Mcp-Session-Id", sid)
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except Exception:
            pass

    def _send_sse(self, sid, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        data_part = b"data: " + payload + b"\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Mcp-Session-Id", sid)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self.wfile.write(("%x\r\n" % len(data_part)).encode("ascii"))
            self.wfile.write(data_part)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass


def start():
    """启动服务器（随机端口），返回 (server, url)。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/mcp"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, url
