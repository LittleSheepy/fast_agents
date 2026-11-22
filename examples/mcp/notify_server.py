"""零依赖 MCP 示例服务器（stdio）—— 演示「服务端主动推送 notification」。

与 echo_server.py 的区别：它在收到 ``notifications/initialized`` 后，会主动向客户端推送两条
notification（无 id、客户端无需响应），用于演示 ``MCPClient.on_notification()`` 的接收与分发。

用法：python examples/mcp/notify_server.py
被 MCPClient 以子进程方式启动，通过 stdin/stdout JSON-RPC 通信。
"""

from __future__ import annotations

import json
import sys


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _recv() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _notify(method: str, params: dict) -> None:
    """单向通知：无 id，客户端不应回包。"""
    _send({"jsonrpc": "2.0", "method": method, "params": params})


TOOLS = [
    {
        "name": "echo",
        "description": "回显文本，仅用于配合 notification 演示。",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "要回显的文本"}},
            "required": ["message"],
        },
    },
]


def main() -> None:
    while True:
        msg = _recv()
        if msg is None:
            # stdin 关闭：客户端 close() 时的正常退出路径
            return

        method = msg.get("method")
        mid = msg.get("id")

        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "demo-notify-server", "version": "0.1.0"},
                "capabilities": {"tools": {"listChanged": True}},
                "instructions": "演示服务端主动推送 notification。",
            }})
        elif method == "notifications/initialized":
            # 握手完成后，服务端主动推送两条通知（无 id）
            _notify("notifications/message", {
                "level": "info",
                "logger": "demo-notify-server",
                "data": "服务器已就绪",
            })
            _notify("notifications/tools/list_changed", {"reason": "startup"})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"echo: {args.get('message', '')}"}],
                    "isError": False,
                }})
            else:
                _send({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32601, "message": f"Unknown tool {name}"}})
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            _send({"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
