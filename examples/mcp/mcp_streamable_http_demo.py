"""Streamable HTTP 传输快速演示 —— MCP v3.3 新增能力。

演示在同一个进程内启动一个最小 Streamable HTTP MCP 服务端，
然后通过 StreamableHTTPTransport 连接、列工具、调用工具。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents import MCPClient, StreamableHTTPTransport

# 加载内联测试服务器
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sh_inline_server import start


def main():
    srv, url = start()
    print(f"[server] 启动于 {url}")
    try:
        transport = StreamableHTTPTransport(url=url, timeout=10.0)
        client = MCPClient(transport)
        init = client.connect(timeout=10)
        print(f"[client] 握手成功：{init.server_info['name']} v{init.server_info.get('version', '')}")

        tools = client.list_tools()
        print(f"[client] 可用工具：{[t.name for t in tools]}")

        r1 = client.call_tool("echo", {"text": "你好，Streamable HTTP"})
        print(f"[client] echo → {r1.as_text()}")

        r2 = client.call_tool("calculate", {"a": 21, "b": 21})
        print(f"[client] calculate(21+21) → {r2.as_text()}")

        client.close()
        print("[client] 已关闭")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
