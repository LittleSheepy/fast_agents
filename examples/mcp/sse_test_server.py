"""极简 MCP SSE 测试服务器（stdlib only，零依赖）。

严格遵循 MCP SSE 传输协议：
- GET /xxx 建立 SSE 流，服务器第一条事件是 `event: endpoint`，data 为 POST 地址
- 客户端向 POST 地址发 JSON-RPC 请求，服务器返回 HTTP 202 Accepted（空 body）
- 服务器把 JSON-RPC 响应作为 `event: message`（或默认事件）通过 SSE 流推回客户端

用于验证 SSETransport：
1. 正常握手 endpoint + tools/list + tools/call 流程（真正走 SSE 双向）
2. 服务器主动断流（/drop 路径）时，客户端能立即感知（不会挂到超时）
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def _sse_event(data: str, event: str = "message") -> bytes:
    """构造一条 SSE 事件帧。"""
    return (f"event: {event}\ndata: {data}\n\n").encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默，避免污染测试输出

    # ── GET：SSE 流 ──
    def do_GET(self):
        path = urlparse(self.path).path
        if path not in ("/normal", "/drop"):
            self.send_response(404); self.end_headers(); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # 分配一个 session id，POST 路径带上它，这样 POST 能找到对应 SSE 流
        sid = uuid.uuid4().hex[:12]
        post_path = f"/post/{sid}"
        self._sid = sid
        self.server._streams[sid] = self
        try:
            # 1) 先推 endpoint 事件
            self.wfile.write(_sse_event(post_path, event="endpoint"))
            self.wfile.flush()
            if path == "/drop":
                return  # 断流测试：endpoint 发完立刻关

            # 2) 心跳保活，直到连接断开或服务器关闭
            while not self.server._stop_event.is_set():
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                except Exception:
                    break
                time.sleep(0.5)
        finally:
            self.server._streams.pop(sid, None)

    # ── POST：客户端发来 JSON-RPC 请求 ──
    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            msg = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(400); self.end_headers(); return

        # 从路径取出 session id
        if not path.startswith("/post/"):
            self.send_response(404); self.end_headers(); return
        sid = path[len("/post/"):]
        stream = self.server._streams.get(sid)
        if stream is None:
            self.send_response(404); self.end_headers(); return

        # 构造响应（按 method）
        method = msg.get("method", "")
        req_id = msg.get("id")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-sse-server", "version": "0.1"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "ping", "description": "reply pong",
                "inputSchema": {"type": "object", "properties": {}},
            }]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
        elif method == "notifications/initialized":
            # 单向 notification：回 202 即可，不推响应
            self.send_response(202); self.end_headers(); return
        else:
            # 未知方法：推回错误
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}}
            self._push(stream, resp)
            self.send_response(202); self.end_headers(); return

        resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        self._push(stream, resp)
        self.send_response(202); self.end_headers()

    def _push(self, stream, obj):
        try:
            stream.wfile.write(_sse_event(json.dumps(obj, ensure_ascii=False)))
            stream.wfile.flush()
        except Exception:
            pass


class SSETestServer:
    """迷你 SSE 测试服务器，支持上下文管理。"""

    def __init__(self, port: int = 0):
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._httpd._streams = {}
        self._httpd._stop_event = threading.Event()
        self.port = self._httpd.server_address[1]
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._httpd._stop_event.set()
        self._httpd.shutdown()
        self._httpd.server_close()

    def url(self, path: str = "/normal") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = SSETestServer(port=port).start()
    print(f"LISTEN 127.0.0.1:{srv.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
