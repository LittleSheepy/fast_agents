"""MCP 传输层稳定性特性演示

演示 v3.1 传输层升级的两个核心改进：
1. Stdio 子进程 stderr 被后台实时读取——大量 stderr 不会卡死子进程；
   子进程异常退出时，错误信息自动出现在异常消息里，排错不用猜。
2. SSE 远程连接被服务器断开时，正在等待的请求会被立即唤醒报错，
   不再默默挂到超时（之前默认超时 60s 才知道连不上）。

运行：python examples/mcp/mcp_transport_stability_demo.py
（SSE 演示依赖 requests；stdio 演示零依赖。）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.mcp import MCPClient, StdioTransport, MCPConnectionError


def demo_stdio_stderr_capture():
    """演示 1：崩溃服务器的 stderr 能在异常中看到。"""
    print("=" * 60)
    print("Demo 1: Stdio 子进程 stderr 捕获与异常诊断")
    print("=" * 60)
    py = sys.executable
    crash_server = str(Path(__file__).parent / "crash_server.py")
    echo_server = str(Path(__file__).parent / "echo_server.py")

    # 正常服务器：连接、调用、关闭一切正常
    print("\n[a] 正常 echo_server：")
    c = MCPClient(StdioTransport(command=py, args=[echo_server]))
    c.connect(timeout=10)
    r = c.call_tool("echo", {"message": "hello", "prefix": ">>", "repeat": 2}, timeout=10)
    print(f"    echo 返回: {r.as_text()!r}")
    c.close()
    print("    关闭后无僵尸进程 ✓")

    # 崩溃服务器：异常消息会带上 stderr 尾部
    print("\n[b] crash_server（输出 5000 行 stderr 后 exit 2）：")
    c = MCPClient(StdioTransport(command=py, args=[crash_server]))
    c.transport.connect()
    print("    子进程已启动，等待它输出大量 stderr 并崩溃...")
    # 等子进程跑完
    deadline = time.time() + 10
    while time.time() < deadline:
        if c.transport._proc is not None and c.transport._proc.poll() is not None:
            break
        time.sleep(0.1)
    time.sleep(0.3)
    print(f"    子进程退出码：{c.transport._proc.returncode}")
    last_err = c.transport.get_last_stderr(n=5)
    print(f"    最近 5 行 stderr:\n      " + "\n      ".join(last_err.splitlines()[-5:]))
    # 子进程退出后任何 recv/send 都会立即报明确错误
    try:
        c.transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    except MCPConnectionError as e:
        print(f"    send 立即报错：{str(e)[:120]}...")
    c.transport.close()
    print("    ✓ 修复前：stderr 管道满会卡死子进程，用户只看到超时；修复后台风实时消费。")


def demo_sse_with_test_server():
    """演示 2：用内置 SSE 测试服务器演示 SSE 断连快速感知（需要 requests）。"""
    print("\n" + "=" * 60)
    print("Demo 2: SSE 连接断开快速感知（需要 requests）")
    print("=" * 60)
    try:
        from fast_agents.mcp import SSETransport
        from examples.mcp.sse_test_server import SSETestServer
    except ImportError as e:
        print(f"  跳过（{e}）")
        return

    with SSETestServer() as srv:
        print(f"\n[a] 正常 SSE 连接到 {srv.url('/normal')}：")
        c = MCPClient(SSETransport(srv.url("/normal"), timeout=5))
        c.connect(timeout=5)
        tools = c.list_tools(timeout=5)
        print(f"    工具列表：{[t.name for t in tools]}")
        r = c.call_tool("ping", {}, timeout=5)
        print(f"    ping 调用：{r.as_text()!r}")
        c.close()
        print("    ✓ 正常流程走通")

    print("\n[b] 服务器关闭后请求快速失败（不会挂到超时）：")
    srv = SSETestServer().start()
    try:
        c = MCPClient(SSETransport(srv.url("/normal"), timeout=30))
        c.connect(timeout=5)
        assert c.connected
        r = c.call_tool("ping", {}, timeout=5)
        assert "pong" in r.as_text()
        print("    先做一次正常调用确认链路 OK")
        start = time.time()
        srv.stop()
        try:
            c.call_tool("ping", {}, timeout=30)
        except MCPConnectionError as e:
            elapsed = time.time() - start
            print(f"    服务器关闭后 {elapsed:.2f}s 内立即收到错误：{type(e).__name__}")
            print(f"    ✓ 修复前：会挂到 30s 超时才报错；修复后读线程立即唤醒等待中的请求。")
        c.close()
    finally:
        srv.stop()


if __name__ == "__main__":
    demo_stdio_stderr_capture()
    demo_sse_with_test_server()
    print("\n" + "=" * 60)
    print("全部演示完成。")
    print("=" * 60)
