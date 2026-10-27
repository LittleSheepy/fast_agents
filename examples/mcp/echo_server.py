"""零依赖 MCP 示例服务器（stdio）—— 实现 echo 和 calculate 两个工具。

用法：python examples/mcp/echo_server.py
被 MCPClient 以子进程方式启动，通过 stdin/stdout JSON-RPC 通信。
"""

from __future__ import annotations

import json
import math
import operator
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


def _safe_eval(expr: str) -> float:
    ops = {
        "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
        "**": operator.pow, "%": operator.mod,
    }
    funcs = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "abs": abs,
    }
    code = expr.replace("^", "**")
    allowed = {**{k: v for k, v in math.__dict__.items() if not k.startswith("_")}, **ops, **funcs}
    return float(eval(code, {"__builtins__": {}}, allowed))


TOOLS = [
    {
        "name": "echo",
        "description": "回显输入文本，可加前缀或重复多次。用于演示多参数 MCP 工具。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "要回显的文本"},
                "prefix": {"type": "string", "description": "加在前面的前缀", "default": ""},
                "repeat": {"type": "integer", "description": "重复次数，默认 1", "default": 1, "minimum": 1, "maximum": 50},
            },
            "required": ["message"],
        },
    },
    {
        "name": "calculate",
        "description": "安全数学表达式计算器，支持 +-*/**% 和 sqrt/sin/cos/abs。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 'sqrt(2)+3*4'"},
            },
            "required": ["expression"],
        },
    },
]


def main() -> None:
    initialized = False
    while True:
        msg = _recv()
        if msg is None:
            break
        method = msg.get("method", "")
        mid = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            _send({
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "demo-echo-server", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": "这是 FastAgents 的 MCP 演示服务器，提供 echo 和 calculate。",
                },
            })
        elif method == "notifications/initialized":
            initialized = True
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            if name == "echo":
                text = args.get("message", "")
                prefix = args.get("prefix", "")
                repeat = int(args.get("repeat", 1))
                out = (prefix + text) * repeat
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": out}],
                    "isError": False,
                }})
            elif name == "calculate":
                expr = args.get("expression", "")
                try:
                    val = _safe_eval(expr)
                    _send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": f"{expr} = {val}"}],
                        "isError": False,
                    }})
                except Exception as e:
                    _send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": f"计算错误：{e}"}],
                        "isError": True,
                    }})
            else:
                _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown tool {name}"}})
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": mid, "result": {}})
        else:
            if mid is not None:
                _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
