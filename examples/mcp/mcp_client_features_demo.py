"""MCPClient v3.1 新能力演示 —— 状态机 / 幂等 / notification 分发 / 并发调用。

离线可跑：以 examples/mcp/notify_server.py 作为 stdio MCP 服务器，不依赖任何真实 API key
或外部服务，直接 ``python examples/mcp/mcp_client_features_demo.py`` 即可看到输出。

覆盖 v3.1 相对旧版新增的能力：
    1. ``on_notification()``：接收服务端主动推送的 notification（旧版会丢弃）。
    2. ``state`` / ``connected``：只读连接状态，便于观测。
    3. ``connect()`` / ``close()`` 幂等，可安全重复调用。
    4. 请求-响应按 id 分发，支持并发调用（旧版 10ms 忙等轮询）。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.mcp import MCPClient, StdioTransport


def main() -> None:
    py = sys.executable
    server = str(Path(__file__).with_name("notify_server.py"))
    client = MCPClient(StdioTransport(command=py, args=[server]))

    # ── 1) 必须在 connect() 之前注册 handler，否则会漏掉握手后立刻推送的通知 ──
    messages = []
    tools_changed = threading.Event()

    def on_message(params):
        messages.append(params)
        print(f"  ← 收到 notifications/message: {params.get('data')}")

    def on_tools_changed(params):
        print(f"  ← 收到 notifications/tools/list_changed: {params}")
        tools_changed.set()

    client.on_notification("notifications/message", on_message)
    client.on_notification("notifications/tools/list_changed", on_tools_changed)

    print("[1] connect 前状态:", client.state)

    init = client.connect()
    print(f"[2] 握手完成：server={init.server_info.get('name')}，"
          f"状态={client.state}，connected={client.connected}")

    # ── 2) 等待服务端主动推送的 notification 被 handler 收到 ──
    tools_changed.wait(timeout=3)
    print(f"[3] 累计收到 {len(messages)} 条 message 通知")

    # ── 3) 常规调用 ──
    print("[4] 工具列表:", [t.name for t in client.list_tools()])
    print("[5] echo 调用:", client.call_tool("echo", {"message": "hello mcp"}).as_text())

    # ── 4) 并发调用：验证请求-响应按 id 分发 ──
    results = {}

    def worker(i: int) -> None:
        results[i] = client.call_tool("echo", {"message": f"并发-{i}"}).as_text()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[6] 4 路并发调用结果:", [results[i] for i in sorted(results)])

    # ── 5) connect / close 幂等 ──
    print("[7] 再次 connect 幂等（未重复握手）:", client.connect() is init)
    client.close()
    print("[8] close 后状态:", client.state)
    client.close()
    print("[9] 再次 close 幂等后状态:", client.state)


if __name__ == "__main__":
    main()
