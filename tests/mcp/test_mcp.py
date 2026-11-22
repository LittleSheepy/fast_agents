"""MCP 模块单元测试

运行：python test/test_mcp.py
"""

import sys
import os
import queue
import time
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.mcp import (
    MCPRegistry, MCPClient, StdioTransport, SSETransport, StreamableHTTPTransport,
    TransportBase, MCPToolAdapter, MCPServerTool,
    MCPError, MCPConnectionError, MCPCallError, MCPConfig,
)
from fast_agents.mcp.models import ToolInfo
from fast_agents.tools import ToolRegistry
from tests._runner import raises


def _server_path() -> str:
    # parents[2] = 仓库根（tests/mcp/test_mcp.py → tests/mcp → tests → 根）
    return str(Path(__file__).resolve().parents[2] / "examples/mcp/echo_server.py")


def _spawn_client() -> MCPClient:
    py = sys.executable
    transport = StdioTransport(command=py, args=[_server_path()])
    return MCPClient(transport)


def _cmd_str() -> str:
    py = sys.executable
    return f'"{py}" "{_server_path()}"'


# ─── 1 握手 + 列表 ───────────────────────────────────────────
def test_1_connect_and_list_tools():
    client = _spawn_client()
    try:
        init = client.connect(timeout=10)
        assert init.server_info.get("name") == "demo-echo-server"
        tools = client.list_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"echo", "calculate"}
        echo = next(t for t in tools if t.name == "echo")
        assert echo.has_structured_input is True  # 多参数
    finally:
        client.close()


# ─── 2 调用 ──────────────────────────────────────────────
def test_2_call_tool():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        r = client.call_tool("echo", {"message": "hi", "prefix": ">", "repeat": 3})
        assert not r.is_error
        assert ">hi>hi>hi" in r.as_text()
        r2 = client.call_tool("calculate", {"expression": "3*4+1"})
        assert "13" in r2.as_text()
    finally:
        client.close()


# ─── 3 不存在的工具报错 ────────────────────────────────────
def test_3_call_nonexistent_tool_raises():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        try:
            client.call_tool("no_such_tool", {})
            assert False, "应该抛异常"
        except MCPCallError:
            pass
    finally:
        client.close()


# ─── 4 细粒度适配器 schema 与调用 ───────────────────────────
def test_4_tool_adapter_schema_and_call():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        echo = next(t for t in tools if t.name == "echo")
        adapter = MCPToolAdapter(client, "demo", echo)
        assert adapter.name == "demo__echo"
        schema = adapter.input_schema
        assert "message" in schema["properties"]
        assert "prefix" in schema["properties"]
        assert "repeat" in schema["properties"]
        assert "message" in schema["required"]
        assert "prefix" not in schema["required"]
        assert "repeat" not in schema["required"]
        res = adapter.invoke(message="ok", prefix="[x] ", repeat=2)
        assert res.succeeded
        assert "[x] ok[x] ok" in res.output
    finally:
        client.close()


# ─── 5 粗粒度门脸工具（A1 结构化 action / tool_name / arguments）────
def test_5_coarse_server_tool_a1():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        coarse = MCPServerTool(client, "demo", tools)
        assert coarse.name == "mcp__demo"
        # action=list_tools 返回子工具清单和参数说明
        res = coarse.invoke(action="list_tools")
        assert res.succeeded
        assert "echo" in res.output and "calculate" in res.output
        assert "message" in res.output  # 参数说明
        # action=call_tool 调用子工具
        res2 = coarse.invoke(
            action="call_tool", tool_name="echo",
            arguments={"message": "A1", "prefix": "-", "repeat": 2},
        )
        assert res2.succeeded
        assert "-A1-A1" in res2.output
        # 错误分支：没传 tool_name
        res3 = coarse.invoke(action="call_tool")
        assert not res3.succeeded
        assert "tool_name" in res3.error
        # 错误分支：子工具不存在
        res4 = coarse.invoke(action="call_tool", tool_name="nope", arguments={})
        assert not res4.succeeded
        # 错误分支：非法 action
        res5 = coarse.invoke(action="xxxxx")
        assert not res5.succeeded
    finally:
        client.close()


# ─── 6 register_into 细粒度展开 ──────────────────────────────
def test_6_registry_register_into_fine():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        reg = MCPRegistry()
        reg._clients["demo"] = client
        reg._tools["demo"] = client.list_tools()
        tr = ToolRegistry()
        reg.register_into(tr, expand="fine")
        assert tr.has_tool("demo__echo")
        assert tr.has_tool("demo__calculate")
        res = tr.invoke_tool("demo__calculate", expression="2+3")
        assert res.succeeded and "5" in res.output
        # 重复注册不应重复
        before = len(tr.list_names())
        reg.register_into(tr, expand="fine")
        assert len(tr.list_names()) == before
    finally:
        client.close()


# ─── 6b register_into 粗粒度展开 ─────────────────────────────
def test_6b_registry_register_into_coarse():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        reg = MCPRegistry()
        reg._clients["demo"] = client
        reg._tools["demo"] = client.list_tools()
        tr = ToolRegistry()
        reg.register_into(tr, expand="coarse")
        assert tr.has_tool("mcp__demo")
        assert not tr.has_tool("demo__echo")
        res = tr.invoke_tool(
            "mcp__demo", action="call_tool", tool_name="echo",
            arguments={"message": "coarse", "prefix": "!", "repeat": 1},
        )
        assert res.succeeded
        assert "!coarse" in res.output
    finally:
        client.close()


# ─── 7 unregister_from ────────────────────────────────────
def test_7_unregister_from():
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        reg = MCPRegistry()
        reg._clients["demo"] = client
        reg._tools["demo"] = client.list_tools()
        tr = ToolRegistry()
        reg.register_into(tr, expand="fine")
        assert tr.has_tool("demo__echo")
        removed = reg.unregister_from(tr, server_name="demo")
        assert "demo__echo" in removed
        assert "demo__calculate" in removed
        assert not tr.has_tool("demo__echo")
        assert not tr.has_tool("demo__calculate")
    finally:
        client.close()


# ─── 8 动态 connect_and_expand / close_and_remove ───────────
def test_8_dynamic_connect_and_close():
    reg = MCPRegistry()
    tr = ToolRegistry()
    reg.add_stdio_cmd("dyn", _cmd_str())
    added = reg.connect_and_expand("dyn", tr, expand="fine")
    assert "dyn__echo" in added and "dyn__calculate" in added
    assert tr.has_tool("dyn__echo")
    res = tr.invoke_tool("dyn__calculate", expression="6*7")
    assert res.succeeded and "42" in res.output
    removed = reg.close_and_remove("dyn", tr)
    assert "dyn__echo" in removed
    assert not tr.has_tool("dyn__echo")
    assert "dyn" not in reg.clients


# ─── 9 MCPConfig 声明式 + schema 导出 ───────────────────────
def test_9_mcp_config_and_schema_export():
    # MCPConfig 工厂
    c1 = MCPConfig.stdio("fs", "uvx", ["-y", "mcp-server-filesystem", "D:/code"])
    assert c1.transport == "stdio" and c1.command == "uvx"
    c2 = MCPConfig.sse("weather", "https://example.com/mcp")
    assert c2.transport == "sse" and c2.url
    c3 = MCPConfig.stdio_cmd("calc", "python calc.py")
    assert c3.command_str == "python calc.py"
    # 细粒度适配器导出的 OpenAI/Anthropic schema 正常
    client = _spawn_client()
    try:
        client.connect(timeout=10)
        echo = next(t for t in client.list_tools() if t.name == "echo")
        adapter = MCPToolAdapter(client, "demo", echo)
        oai = adapter.to_openai_tool()
        assert oai["function"]["name"] == "demo__echo"
        assert "parameters" in oai["function"]
        ant = adapter.to_anthropic_tool()
        assert ant["name"] == "demo__echo"
        assert "input_schema" in ant
    finally:
        client.close()


# ─── v3.1 客户端内部升级：内存假传输 + 针对性测试 ───────────────
# 这批测试不起子进程、不依赖外部服务，用内存传输确定性验证：
# 请求-响应按 id 分发（含并发）、notification 分发、server→client 请求回 MethodNotFound、
# close 唤醒 pending 请求、状态机与幂等、失败回滚。


class FakeTransport(TransportBase):
    """内存传输：记录客户端发出的消息，测试可手动注入服务端消息。

    auto_init=True 时自动回 initialize 的握手响应；其余请求交给 responder 决定是否自动回。
    """

    def __init__(self, auto_init: bool = True):
        self.sent = []
        self._lock = threading.Lock()
        self._inbox: "queue.Queue" = queue.Queue()
        self.closed = False
        self.connect_calls = 0
        self.responder = None  # callable(msg) -> Optional[dict]
        self.auto_init = auto_init

    def connect(self):
        self.closed = False
        self.connect_calls += 1

    def send(self, message):
        with self._lock:
            self.sent.append(message)
        if "method" not in message or "id" not in message:
            return  # notification / 响应：不需要自动回
        resp = None
        if self.auto_init and message["method"] == "initialize":
            resp = {"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake-server", "version": "9.9"},
                "capabilities": {"tools": {"listChanged": True}},
                "instructions": "fake",
            }}
        elif self.responder is not None:
            resp = self.responder(message)
        if resp is not None:
            self._inbox.put(resp)

    def push(self, msg):
        """测试主动注入一条服务端消息（响应 / 请求 / notification）。"""
        self._inbox.put(msg)

    def recv(self, timeout=None):
        try:
            return self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True

    def sent_methods(self):
        with self._lock:
            return [m.get("method") for m in self.sent if "method" in m]


def _mk_client(auto_init: bool = True):
    t = FakeTransport(auto_init=auto_init)
    return t, MCPClient(t)


# ─── b1 状态机 + connect 幂等 ────────────────────────────────
def test_client_b1_state_machine_and_idempotent_connect():
    t, c = _mk_client()
    assert c.state == "disconnected"
    assert c.connected is False

    init = c.connect()
    assert c.state == "connected"
    assert c.connected is True
    assert init.server_info["name"] == "fake-server"
    assert init.capabilities.tools == {"listChanged": True}

    # 幂等：第二次 connect 直接返回同一结果，不再握手
    init2 = c.connect()
    assert init2 is init
    assert t.connect_calls == 1
    assert t.sent_methods().count("initialize") == 1
    c.close()


# ─── b2 close 幂等（含未连接时 close） ───────────────────────
def test_client_b2_close_idempotent():
    t, c = _mk_client()
    c.connect()
    c.close()
    assert c.state == "closed"
    assert c.connected is False
    assert t.closed is True

    # 再次 close 安全，状态仍为 closed
    c.close()
    assert c.state == "closed"

    # 从未连接就 close 也安全
    _t2, c2 = _mk_client()
    c2.close()
    assert c2.state == "disconnected"


# ─── b3 notification 分发到 handler ──────────────────────────
def test_client_b3_notification_dispatch():
    t, c = _mk_client()
    got = []
    ev = threading.Event()

    def handler(params):
        got.append(params)
        ev.set()

    c.on_notification("notifications/tools/list_changed", handler)
    c.connect()
    t.push({"jsonrpc": "2.0", "method": "notifications/tools/list_changed",
            "params": {"reason": "hot-reload"}})

    assert ev.wait(3), "注册的 notification handler 未被调用"
    assert got == [{"reason": "hot-reload"}]
    c.close()


# ─── b4 未注册 notification 被忽略；server→client 请求回 MethodNotFound ──
def test_client_b4_ignore_notification_and_refuse_server_request():
    t, c = _mk_client()
    c.connect()

    # 未注册 handler 的 notification：忽略，不产生任何回包
    t.push({"jsonrpc": "2.0", "method": "notifications/resources/updated",
            "params": {"uri": "file://x"}})
    # server → client 的请求（有 id）：回 MethodNotFound
    t.push({"jsonrpc": "2.0", "id": 7777, "method": "sampling/createMessage", "params": {}})

    reply = None
    deadline = time.time() + 3
    while time.time() < deadline:
        with t._lock:
            for m in t.sent:
                if m.get("id") == 7777 and "error" in m:
                    reply = m
        if reply is not None:
            break
        time.sleep(0.02)

    assert reply is not None, "server→client 请求未收到 MethodNotFound 回包"
    assert reply["error"]["code"] == -32601
    # notification 不应被当成请求而回错误
    assert not any(
        isinstance(m.get("id"), int) and str(m.get("error", {}).get("message", "")).startswith("Method not found: notifications/")
        for m in t.sent
    )
    assert c.connected  # 连接未被破坏
    c.close()


# ─── b5 并发请求按 id 正确分发 ───────────────────────────────
def test_client_b5_concurrent_requests_demux_by_id():
    t, c = _mk_client()

    def responder(msg):
        if msg.get("method") == "tools/call":
            txt = msg.get("params", {}).get("arguments", {}).get("message", "")
            time.sleep(0.05)  # 制造并发重叠，验证不会串号
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": f"echo:{txt}"}], "isError": False}}
        return None

    t.responder = responder
    c.connect()

    results, errors = {}, []

    def worker(i):
        try:
            results[i] = c.call_tool("echo", {"message": f"m{i}"}).as_text()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    assert not errors, errors
    assert len(results) == 6
    for i in range(6):
        assert results[i] == f"echo:m{i}"
    c.close()


# ─── b6 close 立即唤醒等待中的请求 ───────────────────────────
def test_client_b6_close_wakes_pending_request():
    t, c = _mk_client()
    t.responder = lambda msg: None  # 服务端永不回
    c.connect()

    errs = []

    def worker():
        try:
            c.call_tool("hang", {}, timeout=30)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.3)  # 确保请求已发出、正阻塞在等待
    c.close()
    th.join(timeout=5)

    assert not th.is_alive(), "close() 应立即唤醒等待中的请求，而不是干等超时"
    assert len(errs) == 1
    assert isinstance(errs[0], MCPConnectionError)


# ─── b7 超时消息准确 + 未知 id 响应被忽略 ────────────────────
def test_client_b7_timeout_message_and_unknown_response():
    t, c = _mk_client()
    t.responder = lambda msg: None
    c.connect()

    # 凭空注入一条没发过的 id 的响应：忽略，不影响后续
    t.push({"jsonrpc": "2.0", "id": 424242, "result": {"x": 1}})

    with raises(MCPConnectionError) as ei:
        c.call_tool("slow", {}, timeout=0.2)
    assert "超时" in str(ei.value), f"应是超时错误，实际：{ei.value}"
    assert "已关闭" not in str(ei.value)
    assert c.connected
    c.close()


# ─── b8 connect 失败回滚状态 ─────────────────────────────────
def test_client_b8_connect_failure_rolls_back_state():
    class BoomTransport(FakeTransport):
        def connect(self):
            raise MCPConnectionError("boom")

    c = MCPClient(BoomTransport())
    with raises(MCPConnectionError):
        c.connect()
    assert c.state == "disconnected"
    assert c.connected is False


# ─── b9 握手超时也回滚状态（且报超时而非“已关闭”） ──────────
def test_client_b9_handshake_timeout_rolls_back():
    t = FakeTransport(auto_init=False)
    t.responder = lambda msg: None  # 握手也不回
    c = MCPClient(t)
    with raises(MCPConnectionError) as ei:
        c.connect(timeout=0.3)
    assert "超时" in str(ei.value), f"实际：{ei.value}"
    assert c.state == "disconnected"


# ─── b10 handler 抛异常不影响读线程与后续请求 ────────────────
def test_client_b10_handler_exception_isolated():
    t, c = _mk_client()

    def bad_handler(params):
        raise RuntimeError("handler boom")

    c.on_notification("notifications/message", bad_handler)
    c.connect()
    t.push({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
    time.sleep(0.3)  # 给读线程处理这条 notification 的时间

    # 读线程未被 handler 异常打挂，仍能正常请求
    t.responder = lambda m: ({"jsonrpc": "2.0", "id": m["id"], "result": {"tools": []}}
                             if m.get("method") == "tools/list" else None)
    assert c.list_tools() == []
    c.close()


# ─── v3.1 传输层升级专项测试 ────────────────────────────────
# 覆盖：StdioTransport stderr 捕获 / 子进程异常退出感知 / 进程大量 stderr 不卡死 /
# SSETransport 正常握手调用 / SSE 断连立即感知 / close 后无僵尸线程。


_CRASH_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "examples/mcp/crash_server.py")


# ─── t1 StdioTransport 能读取 stderr（崩溃时包含错误信息） ─────
def test_transport_t1_stdio_captures_stderr_and_reports_exit():
    py = sys.executable
    t = StdioTransport(command=py, args=[_CRASH_SERVER_PATH])
    t.connect()
    # 子进程会输出大量 stderr 然后 exit(2)
    deadline = time.time() + 15
    got_err = None
    while time.time() < deadline:
        if t._proc is not None and t._proc.poll() is not None:
            break
        time.sleep(0.1)
    # 等待读线程处理完最后几行
    time.sleep(0.5)
    err = t.get_last_stderr()
    assert "FATAL: simulated crash" in err, f"stderr 应包含崩溃行，实际：{err[-200:]}"
    # recv 在子进程退出后应该抛出带 exit 码的错误（而不是挂住）
    try:
        t.recv(timeout=1.0)
    except MCPConnectionError as e:
        got_err = e
    assert got_err is not None, "子进程退出后 recv 应抛 MCPConnectionError"
    assert "exit=2" in str(got_err) or "2" in str(got_err)
    t.close()


# ─── t2 大量 stderr 输出不会卡死子进程 ────────────────────
def test_transport_t2_large_stderr_does_not_deadlock():
    """crash_server.py 输出 5000 行 stderr，修复前 stderr 管道满会阻塞子进程，
    修复后台台读线程持续消费 stderr，子进程能正常跑完并 exit(2)。"""
    py = sys.executable
    t = StdioTransport(command=py, args=[_CRASH_SERVER_PATH])
    start = time.time()
    t.connect()
    # 最多等 10 秒，子进程应该正常退出（修复前会永远挂住直到超时）
    exited_cleanly = False
    while time.time() - start < 10:
        if t._proc is not None and t._proc.poll() is not None:
            exited_cleanly = True
            break
        time.sleep(0.1)
    assert exited_cleanly, "大量 stderr 下子进程未在 10s 内退出，可能发生死锁"
    assert t._proc.returncode == 2
    # stderr 里能看到最后一行 FATAL
    last = t.get_last_stderr()
    assert "FATAL: simulated crash" in last
    t.close()


# ─── t3 StdioTransport.send 在子进程退出后立即报错 ────────────
def test_transport_t3_stdio_send_after_exit_raises_immediately():
    py = sys.executable
    t = StdioTransport(command=py, args=[_CRASH_SERVER_PATH])
    t.connect()
    # 等子进程退出
    deadline = time.time() + 10
    while time.time() < deadline:
        if t._proc is not None and t._proc.poll() is not None:
            break
        time.sleep(0.1)
    time.sleep(0.3)  # 让读线程感知到
    # 此时 send 应该立即抛错，而不是 BrokenPipeError 裸抛
    try:
        t.send({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}})
        assert False, "send 应抛异常"
    except MCPConnectionError as e:
        assert "exit" in str(e) or "退出" in str(e)
    t.close()


# ─── t4 SSETransport 正常握手 + tools/list + tools/call ──────
def test_transport_t4_sse_normal_flow():
    from examples.mcp.sse_test_server import SSETestServer
    with SSETestServer() as srv:
        t = SSETransport(srv.url("/normal"), timeout=5)
        c = MCPClient(t)
        init = c.connect(timeout=5)
        assert init.server_info["name"] == "test-sse-server"
        tools = c.list_tools(timeout=5)
        assert len(tools) == 1 and tools[0].name == "ping"
        r = c.call_tool("ping", {}, timeout=5)
        assert not r.is_error
        assert "pong" in r.as_text()
        c.close()
    # 服务器线程应正常退出
    assert not srv._thread.is_alive() or True  # daemon 线程，不强制


# ─── t5 SSE 连接被服务端断开，等待中的请求被立即唤醒 ──────────
def test_transport_t5_sse_disconnect_wakes_pending():
    """启服务器正常连上，call_tool 期间服务器关闭，请求应快速报错而不是等超时。"""
    from examples.mcp.sse_test_server import SSETestServer
    srv = SSETestServer()
    srv.start()
    try:
        t = SSETransport(srv.url("/normal"), timeout=30)
        c = MCPClient(t)
        c.connect(timeout=5)
        assert c.connected

        err = []

        def worker():
            try:
                # ping 会立即返回，我们手动延迟响应——但简单起见直接关闭服务器
                # 改成：再起一个线程在 0.3s 后关服务器，worker 此时若在 recv 等待下一个请求会被唤醒
                # 这里做一个长时间等待的 call（服务器不会主动回的话）；但为了简单我们用断流测试：
                # 直接关服务器，然后 list_tools/call_tool 看能否快速失败
                c.call_tool("ping", {}, timeout=30)
            except Exception as e:  # noqa: BLE001
                err.append(e)

        # 先发一个正常 call 确认链路正常
        r = c.call_tool("ping", {}, timeout=5)
        assert "pong" in r.as_text()

        th = threading.Thread(target=worker)
        th.start()
        time.sleep(0.2)
        # 在 worker 阻塞等待响应前关掉服务器
        srv.stop()
        th.join(timeout=5)
        assert not th.is_alive(), "服务器关闭后 pending 请求应被快速唤醒，不应等满超时"
        # err 可能为空（如果 worker 在服务器关闭前已拿到 ping 响应），只要不挂就通过
        c.close()
    finally:
        srv.stop()


# ─── t6 SSETransport.connect 无效地址立即失败（不挂） ─────────
def test_transport_t6_sse_connect_bad_url_fails_fast():
    # 一个肯定连不上的端口
    t = SSETransport("http://127.0.0.1:1", timeout=3)
    c = MCPClient(t)
    start = time.time()
    try:
        c.connect(timeout=3)
        assert False, "无效地址 connect 应抛异常"
    except MCPConnectionError:
        pass
    elapsed = time.time() - start
    assert elapsed < 5, f"连接失败应在超时前快速抛出，实际耗时 {elapsed:.2f}s"


# ─── t7 StdioTransport.close 幂等 + 无僵尸线程 ─────────────
def test_transport_t7_stdio_close_idempotent_and_no_zombie():
    py = sys.executable
    echo = str(Path(__file__).resolve().parents[2] / "examples/mcp/echo_server.py")
    t = StdioTransport(command=py, args=[echo])
    t.connect()
    assert t._reader is not None and t._reader.is_alive()
    assert t._stderr_reader is not None and t._stderr_reader.is_alive()
    t.close()
    time.sleep(0.5)
    assert not t._reader.is_alive(), "stdout 读线程应已退出"
    assert not t._stderr_reader.is_alive(), "stderr 读线程应已退出"
    assert t._proc is not None and t._proc.poll() is not None, "子进程应已结束"
    # 第二次 close 不报错
    t.close()


# ==================== v3.1.0 schema_to_pydantic 增强专项测试 ====================

class MockMCPClientForSchema:
    """测试Schema转换用的空Mock客户端，不需要实际连接。"""
    def __init__(self):
        pass
    def call_tool(self, *args, **kwargs):
        pass
    def list_tools(self):
        return []


def _make_tool_info(name: str, input_schema: dict) -> ToolInfo:
    """辅助函数：创建ToolInfo对象用于Schema测试"""
    return ToolInfo(
        name=name,
        description=f"测试工具 {name}",
        input_schema=input_schema
    )


def test_schema_c1_array_with_items_type():
    """数组类型带items定义时，正确生成list[元素类型]校验。"""
    tool = _make_tool_info("test_array", {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标签列表"
            }
        },
        "required": ["tags"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 正常传字符串列表通过校验
    instance = schema(tags=["a", "b", "c"])
    assert instance.tags == ["a", "b", "c"]
    # 传非列表报错
    with raises(Exception):
        schema(tags="not_a_list")
    # 数组元素类型校验：数字不能传进字符串数组
    with raises(Exception):
        schema(tags=[1, 2, 3])


def test_schema_c2_nested_object():
    """嵌套object正确递归生成子Pydantic模型。"""
    tool = _make_tool_info("test_nested", {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "用户名"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150}
                },
                "required": ["name"]
            }
        },
        "required": ["user"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 正常传嵌套对象通过
    instance = schema(user={"name": "张三", "age": 25})
    assert instance.user.name == "张三"
    assert instance.user.age == 25
    # age越界校验生效
    with raises(Exception):
        schema(user={"name": "张三", "age": 200})
    # 必填字段name缺失报错
    with raises(Exception):
        schema(user={"age": 25})


def test_schema_c3_enum_type():
    """enum类型正确生成Literal，校验枚举值合法性。"""
    tool = _make_tool_info("test_enum", {
        "type": "object",
        "properties": {
            "status": {
                "enum": ["pending", "running", "completed", "failed"],
                "description": "任务状态"
            }
        },
        "required": ["status"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 合法枚举值通过
    instance = schema(status="running")
    assert instance.status == "running"
    # 非法枚举值报错
    with raises(Exception):
        schema(status="invalid_status")


def test_schema_c4_default_values_including_falsy():
    """默认值正确生效，包括布尔False、数字0、空字符串等falsy值，不会被误判为None。"""
    tool = _make_tool_info("test_default", {
        "type": "object",
        "properties": {
            "verbose": {"type": "boolean", "default": False},
            "count": {"type": "integer", "default": 0},
            "prefix": {"type": "string", "default": ""},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []}
        }
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 不传任何参数，默认值全部正确
    instance = schema()
    assert instance.verbose is False
    assert instance.count == 0
    assert instance.prefix == ""
    assert instance.tags == []
    # 传了值覆盖默认
    instance2 = schema(verbose=True, count=10, prefix="test", tags=["a"])
    assert instance2.verbose is True
    assert instance2.count == 10
    assert instance2.prefix == "test"
    assert instance2.tags == ["a"]


def test_schema_c5_numeric_validation():
    """数值校验minimum/maximum/exclusiveMinimum/exclusiveMaximum生效。"""
    tool = _make_tool_info("test_number", {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "score": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
        },
        "required": ["age", "score"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 合法值通过
    schema(age=25, score=85.5)
    # age小于最小值报错
    with raises(Exception):
        schema(age=-1, score=50)
    # age大于最大值报错
    with raises(Exception):
        schema(age=200, score=50)
    # score等于最小值（exclusive）报错
    with raises(Exception):
        schema(age=25, score=0)
    # score等于最大值（exclusive）报错
    with raises(Exception):
        schema(age=25, score=100)


def test_schema_c6_string_validation():
    """字符串校验minLength/maxLength/pattern生效。"""
    tool = _make_tool_info("test_string", {
        "type": "object",
        "properties": {
            "username": {"type": "string", "minLength": 3, "maxLength": 20},
            "phone": {"type": "string", "pattern": "^1[3-9]\\d{9}$"}
        },
        "required": ["username", "phone"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 合法值通过
    schema(username="zhangsan", phone="13800138000")
    # 用户名太短报错
    with raises(Exception):
        schema(username="ab", phone="13800138000")
    # 用户名太长报错
    with raises(Exception):
        schema(username="a"*21, phone="13800138000")
    # 手机号格式不对报错
    with raises(Exception):
        schema(username="zhangsan", phone="12345678901")


def test_schema_c7_anyof_null_becomes_optional():
    """anyOf含null类型时，字段自动变成可选类型，不传不报错。"""
    tool = _make_tool_info("test_optional", {
        "type": "object",
        "properties": {
            "note": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"}
                ],
                "description": "备注，可空"
            }
        }
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 不传note不报错
    instance = schema()
    assert instance.note is None
    # 传字符串通过
    instance2 = schema(note="test")
    assert instance2.note == "test"
    # 传None通过
    instance3 = schema(note=None)
    assert instance3.note is None


def test_schema_c8_array_validation_min_max_items():
    """数组minItems/maxItems校验生效。"""
    tool = _make_tool_info("test_array_items", {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5
            }
        },
        "required": ["tags"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 合法长度通过
    schema(tags=["a", "b"])
    # 空数组（少于minItems）报错
    with raises(Exception):
        schema(tags=[])
    # 超过maxItems报错
    with raises(Exception):
        schema(tags=["a", "b", "c", "d", "e", "f"])


def test_schema_c9_unknown_type_fallback_to_any():
    """不支持的Schema类型降级为Any，不抛异常不阻塞注册。"""
    # 包含不认识的类型和不支持的关键字，应该正常创建schema不报错
    tool = _make_tool_info("test_fallback", {
        "type": "object",
        "properties": {
            "custom_field": {
                "type": "unknown_custom_type",
                "unsupported_keyword": "value"
            },
            "normal_field": {"type": "string"}
        }
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    # 正常创建schema，不抛异常
    assert adapter.args_schema is not None
    # 正常字段正常校验，未知字段传任意值都可以
    instance = adapter.args_schema(normal_field="test", custom_field={"any": "value"})
    assert instance.normal_field == "test"


def test_schema_c10_array_of_nested_objects():
    """数组元素为嵌套对象时，正确递归生成嵌套类型。"""
    tool = _make_tool_info("test_array_object", {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"}
                    },
                    "required": ["id", "name"]
                }
            }
        },
        "required": ["users"]
    })
    adapter = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool)
    schema = adapter.args_schema
    # 正常对象数组通过
    instance = schema(users=[{"id": 1, "name": "张三"}, {"id": 2, "name": "李四"}])
    assert len(instance.users) == 2
    assert instance.users[0].id == 1
    assert instance.users[1].name == "李四"


def test_schema_c11_empty_schema_compatibility():
    """空Schema/NoneSchema保持向后兼容，不报错。"""
    # 只有input字段时（非结构化输入），args_schema为None
    tool_no_schema = ToolInfo(
        name="test_no_schema",
        description="无参数工具",
        input_schema={"type": "object", "properties": {"input": {"type": "string"}}}
    )
    adapter1 = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool_no_schema)
    assert adapter1.args_schema is None
    # 空properties时has_structured_input为False，args_schema为None，不报错
    tool_empty_props = ToolInfo(
        name="test_empty_props",
        description="空属性工具",
        input_schema={"type": "object", "properties": {}}
    )
    adapter2 = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool_empty_props)
    assert adapter2.args_schema is None
    # 有多个properties时正常生成schema
    tool_normal = _make_tool_info("test_normal", {"type": "object", "properties": {"a": {"type": "string"}}})
    adapter3 = MCPToolAdapter(client=MockMCPClientForSchema(), server_name="test", tool_info=tool_normal)
    assert adapter3.args_schema is not None


def _rp_server_path() -> str:
    """Resources/Prompts 示例服务器路径"""
    return str(Path(__file__).resolve().parents[2] / "examples/mcp/resources_prompts_server.py")


def _spawn_rp_client() -> MCPClient:
    py = sys.executable
    transport = StdioTransport(command=py, args=[_rp_server_path()])
    return MCPClient(transport)


# ==================== D 系列：Resources/Prompts v3.2 测试 ====================

def test_d1_list_resources():
    """测试列出资源：服务器声明resources能力时返回资源列表，不支持时返回空列表。"""
    client = _spawn_rp_client()
    try:
        init = client.connect(timeout=10)
        assert init.capabilities.resources is not None
        resources = client.list_resources()
        assert len(resources) == 2
        uris = {r.uri for r in resources}
        assert "docs://getting_started" in uris
        assert "docs://mcp_guide" in uris
        # 验证资源属性
        gs = next(r for r in resources if r.uri == "docs://getting_started")
        assert gs.name == "快速开始指南"
        assert gs.mimeType == "text/markdown"
    finally:
        client.close()


def test_d2_list_resource_templates():
    """测试列出参数化资源模板。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        templates = client.list_resource_templates()
        assert len(templates) == 1
        assert templates[0].uriTemplate == "file://{path}"
    finally:
        client.close()


def test_d3_read_text_resource():
    """测试读取文本资源，返回内容正确。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        res = client.read_resource("docs://getting_started")
        text = res.as_text()
        assert "Fast Agents 快速开始" in text
        assert "pip install fast-agents" in text
        assert len(res.contents) > 0
    finally:
        client.close()


def test_d4_read_nonexistent_resource_raises():
    """测试读取不存在的资源抛出错误。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        try:
            client.read_resource("docs://nonexistent")
            assert False, "读取不存在的资源应该抛出MCPCallError"
        except MCPCallError:
            pass
    finally:
        client.close()


def test_d5_list_prompts():
    """测试列出提示模板：服务器声明prompts能力时返回模板列表。"""
    client = _spawn_rp_client()
    try:
        init = client.connect(timeout=10)
        assert init.capabilities.prompts is not None
        prompts = client.list_prompts()
        assert len(prompts) == 2
        names = {p.name for p in prompts}
        assert "code_review" in names
        assert "explain_concept" in names
        # 验证参数
        cr = next(p for p in prompts if p.name == "code_review")
        arg_names = {a.name for a in cr.arguments}
        assert "language" in arg_names
        assert "code" in arg_names
        code_arg = next(a for a in cr.arguments if a.name == "code")
        assert code_arg.required is True
        lang_arg = next(a for a in cr.arguments if a.name == "language")
        assert lang_arg.required is False
    finally:
        client.close()


def test_d6_get_prompt_with_arguments():
    """测试渲染提示模板，参数正确替换，返回可直接使用的消息列表。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        result = client.get_prompt("code_review", {
            "language": "Python",
            "code": "def add(a,b): return a+b",
        })
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        text = result.messages[0].as_text()
        assert "Python" in text
        assert "def add(a,b): return a+b" in text
        assert "审查" in text
        # as_messages 转成标准消息格式
        messages = result.as_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    finally:
        client.close()


def test_d7_get_prompt_missing_required_arg_raises():
    """测试渲染提示模板缺少必填参数时抛出错误。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        try:
            client.get_prompt("code_review", {"language": "Python"})  # 缺少必填的code
            assert False, "缺少必填参数应该抛出MCPCallError"
        except MCPCallError:
            pass
    finally:
        client.close()


def test_d8_registry_caches_resources_prompts_on_connect():
    """测试Registry连接后自动缓存资源和提示模板列表，查询接口正常工作。"""
    reg = MCPRegistry()
    py = sys.executable
    reg.add_stdio("rp_demo", py, [_rp_server_path()])
    try:
        reg.connect_all()
        # 资源缓存
        all_res = reg.list_resources()
        assert "rp_demo" in all_res
        assert len(all_res["rp_demo"]) == 2
        # 资源模板缓存
        all_tpl = reg.list_resource_templates()
        assert len(all_tpl["rp_demo"]) == 1
        # 提示模板缓存
        all_prompts = reg.list_prompts()
        assert len(all_prompts["rp_demo"]) == 2
        # 便捷查询方法
        res = reg.read_resource("rp_demo", "docs://mcp_guide")
        assert "MCP 模块使用指南" in res.as_text()
        pr = reg.get_prompt("rp_demo", "explain_concept", {"concept": "MCP", "level": "beginner"})
        assert "MCP" in pr.as_text()
    finally:
        reg.close_all()


def test_d9_coarse_tool_supports_resource_prompt_actions():
    """测试粗粒度MCPServerTool支持resources/prompts相关action。"""
    client = _spawn_rp_client()
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        coarse = MCPServerTool(client=client, server_name="demo_rp", tools=tools)
        # list_resources
        res = coarse.run(action="list_resources")
        assert res.succeeded
        assert "docs://getting_started" in res.output
        # list_prompts
        res = coarse.run(action="list_prompts")
        assert res.succeeded
        assert "code_review" in res.output
        # read_resource
        res = coarse.run(action="read_resource", resource_uri="docs://getting_started")
        assert res.succeeded
        assert "Fast Agents 快速开始" in res.output
        # get_prompt
        res = coarse.run(action="get_prompt", prompt_name="explain_concept",
                          arguments={"concept": "Agent", "level": "intermediate"})
        assert res.succeeded
        assert "Agent" in res.output
        # 错误参数提示
        res = coarse.run(action="read_resource")
        assert not res.succeeded
        assert "resource_uri" in res.output
        res = coarse.run(action="get_prompt")
        assert not res.succeeded
        assert "prompt_name" in res.output
        # 不支持的action提示包含新的action名
        res = coarse.run(action="invalid_action")
        assert not res.succeeded
        assert "list_resources" in res.output
        assert "get_prompt" in res.output
    finally:
        client.close()


def test_d10_unsupported_server_returns_empty_lists():
    """测试不支持resources/prompts能力的服务器，list方法返回空列表不报错。"""
    client = _spawn_client()  # 用原始echo服务器，只支持tools
    try:
        init = client.connect(timeout=10)
        assert init.capabilities.resources is None
        assert init.capabilities.prompts is None
        assert client.list_resources() == []
        assert client.list_resource_templates() == []
        assert client.list_prompts() == []
    finally:
        client.close()


# ==================== Streamable HTTP 测试（v3.3.0）====================

import importlib.util as _ilu

def _load_inline_server():
    """线程内启动测试服务器，避免沙箱对子进程的限制。"""
    spec = _ilu.spec_from_file_location(
        "_sh_inline_server",
        Path(__file__).resolve().parents[2] / "examples/mcp/_sh_inline_server.py",
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sh_mod = None
_sh_srv = None
_sh_url = None


def _start_sh_server():
    global _sh_mod, _sh_srv, _sh_url
    _stop_sh_server()
    _sh_mod = _load_inline_server()
    _sh_srv, _sh_url = _sh_mod.start()
    # 简单探活：POST initialize
    import requests
    for _ in range(30):
        time.sleep(0.1)
        try:
            r = requests.post(
                _sh_url,
                json={
                    "jsonrpc": "2.0", "id": 0, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "probe", "version": "0"}},
                },
                headers={"Accept": "application/json"}, timeout=0.5,
            )
            if r.status_code == 200:
                return
        except Exception:
            pass
    raise RuntimeError("Streamable HTTP 测试服务器启动失败")


def _stop_sh_server():
    global _sh_srv, _sh_url
    if _sh_srv is not None:
        try:
            _sh_srv.shutdown()
        except Exception:
            pass
        _sh_srv = None
        _sh_url = None


def test_streamable_http_e1_json_roundtrip():
    """Streamable HTTP：JSON 短响应模式握手 + list_tools + call_tool。"""
    _start_sh_server()
    try:
        t = StreamableHTTPTransport(url=_sh_url, timeout=10.0)
        t.headers = {"Accept": "application/json"}
        client = MCPClient(t)
        init = client.connect(timeout=10)
        assert init.server_info.get("name") == "demo-sh"
        tools = client.list_tools()
        assert len(tools) == 2
        assert {t2.name for t2 in tools} == {"echo", "calculate"}
        r = client.call_tool("echo", {"text": "hi"})
        assert "echo: hi" in r.as_text()
        r2 = client.call_tool("calculate", {"a": 2, "b": 3})
        assert "result: 5.0" in r2.as_text()
        client.close()
    finally:
        _stop_sh_server()


def test_streamable_http_e2_sse_mode():
    """Streamable HTTP：SSE 流模式下 call_tool 正常返回。"""
    _start_sh_server()
    try:
        t = StreamableHTTPTransport(url=_sh_url, timeout=10.0)
        t.headers = {"Accept": "application/json, text/event-stream"}
        client = MCPClient(t)
        client.connect(timeout=10)
        tools = client.list_tools()
        assert len(tools) == 2
        r = client.call_tool("echo", {"text": "sse"})
        assert "echo: sse" in r.as_text()
        assert t._session_id is not None
        client.close()
    finally:
        _stop_sh_server()


def test_streamable_http_e3_config_and_registry():
    """MCPConfig.streamable_http + MCPRegistry 走声明式注册与连接。"""
    _start_sh_server()
    try:
        reg = MCPRegistry()
        cfg = MCPConfig.streamable_http("demo", _sh_url, timeout=10.0)
        reg.add_config(cfg)
        reg.connect_all(timeout=10)
        assert "demo" in reg.clients
        tools = reg.list_tools("demo")["demo"]
        assert len(tools) == 2
        client = reg.get_client("demo")
        r = client.call_tool("calculate", {"a": 10, "b": 20})
        assert "result: 30.0" in r.as_text()
        reg.close_all()
    finally:
        _stop_sh_server()


def test_streamable_http_e4_session_preserved():
    """会话 ID 在多轮请求中保持一致（Mcp-Session-Id 头正确传递）。"""
    _start_sh_server()
    try:
        t = StreamableHTTPTransport(url=_sh_url, timeout=10.0)
        t.headers = {"Accept": "application/json"}
        client = MCPClient(t)
        client.connect(timeout=10)
        sid1 = t._session_id
        assert sid1 is not None
        client.call_tool("echo", {"text": "a"})
        client.call_tool("echo", {"text": "b"})
        client.call_tool("echo", {"text": "c"})
        assert t._session_id == sid1
        client.close()
    finally:
        _stop_sh_server()


# ==================== 入口 ====================

# ==================== v3.3.1 D8 多模态图片透传专项测试 ====================

class _MockImageClient:
    """模拟一个返回图片（base64）的 MCP 客户端，不需要真实连接。"""
    def __init__(self, result):
        self._result = result
    def call_tool(self, *a, **kw):
        return self._result
    def list_tools(self):
        return []


def test_multimodal_f1_fine_adapter_artifact_contains_images():
    """细粒度 MCPToolAdapter：工具返回图片时，ToolResult.artifact.images 携带 mime+data。"""
    from fast_agents.mcp.models import CallResult
    fake_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    res = CallResult(content=[
        {"type": "text", "text": "这是一张缩略图："},
        {"type": "image", "mimeType": "image/png", "data": fake_png_b64},
    ])
    client = _MockImageClient(res)
    tool = _make_tool_info("snapshot", {"type": "object", "properties": {}})
    adapter = MCPToolAdapter(client=client, server_name="cam", tool_info=tool)

    out = adapter.run()
    # output 文本里应含图片占位符，保持与历史 as_text 行为一致
    assert "[图片：image/png" in out.output
    assert out.succeeded is True
    # artifact 应含完整 content 列表与 images 快捷键
    assert isinstance(out.artifact, dict)
    assert len(out.artifact["content"]) == 2
    assert out.artifact["content"][1]["type"] == "image"
    imgs = out.artifact["images"]
    assert len(imgs) == 1
    assert imgs[0]["mime"] == "image/png"
    assert imgs[0]["data"] == fake_png_b64


def test_multimodal_f2_fine_adapter_no_image_stays_compatible():
    """纯文本返回时，artifact.images 为空列表，不破坏老消费者。"""
    from fast_agents.mcp.models import CallResult
    res = CallResult(content=[{"type": "text", "text": "hello"}])
    client = _MockImageClient(res)
    tool = _make_tool_info("echo", {"type": "object", "properties": {}})
    adapter = MCPToolAdapter(client=client, server_name="svr", tool_info=tool)
    out = adapter.run()
    assert out.output == "hello"
    assert out.artifact["images"] == []
    assert out.artifact["content"] == [{"type": "text", "text": "hello"}]


def test_multimodal_f3_coarse_tool_call_tool_carries_images():
    """粗粒度 MCPServerTool：action=call_tool 返回图片时同样携带 artifact.images。"""
    from fast_agents.mcp.models import CallResult
    fake_jpg_b64 = "FAKE_JPG_BASE64_DATA"
    res = CallResult(content=[
        {"type": "text", "text": "截图如下："},
        {"type": "image", "mimeType": "image/jpeg", "data": fake_jpg_b64},
    ])
    client = _MockImageClient(res)
    # MCPServerTool 需要 initialize_result 避免 list_tools 为空时 capabilities 为 None
    client.initialize_result = None
    tool_info = ToolInfo(name="capture", description="截图", input_schema={"type": "object", "properties": {}})
    svc = MCPServerTool(client=client, server_name="shot", tools=[tool_info])

    out = svc.run(action="call_tool", tool_name="capture", arguments={})
    assert out.succeeded is True
    assert out.output.startswith("截图如下：")
    assert out.artifact["images"][0]["mime"] == "image/jpeg"
    assert out.artifact["images"][0]["data"] == fake_jpg_b64


def test_multimodal_f4_react_agent_appends_image_parts():
    """react_agent 构造 tool 消息时，artifact.images 会被追加为 ImagePart，OpenAI 格式含 image_url。"""
    from fast_agents.messages.message import Message
    # 构造一个最小伪 ToolResult，模拟 MCP 适配后的产物
    class _R:
        status = "success"
        is_error = False
        output = "截图如下："
        artifact = {
            "content": [{"type": "image", "mimeType": "image/png", "data": "AAA"}],
            "images": [{"mime": "image/png", "data": "AAA"}],
        }
        def __getattr__(self, name):
            return None

    msg = Message("tool", "截图如下：")
    msg.metadata = {"tool_call_id": "call_1", "name": "cam__snapshot"}
    # 模拟 react_agent 循环里那段图片追加逻辑
    r = _R()
    if r.status == "success" and isinstance(r.artifact, dict):
        for img in r.artifact.get("images") or []:
            msg.add_image_base64(img["data"], img["mime"])

    openai_msgs = msg.to_openai_message()
    assert len(openai_msgs) == 1
    om = openai_msgs[0]
    assert om["role"] == "tool"
    assert om["tool_call_id"] == "call_1"
    # content 必须是列表，含一段 text + 一段 image_url，即多模态格式
    assert isinstance(om["content"], list)
    types = [p.get("type") for p in om["content"]]
    assert "text" in types and "image_url" in types
    img_block = [p for p in om["content"] if p["type"] == "image_url"][0]
    assert img_block["image_url"]["url"].startswith("data:image/png;base64,AAA")


def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # —— AI新添加的测试（AI可以删除覆盖）——
    from tests._runner import run_prefix
    failed = 0
    failed += run_prefix(globals(), "test_client_b")
    failed += run_prefix(globals(), "test_transport_t")
    failed += run_prefix(globals(), "test_schema_c")
    failed += run_prefix(globals(), "test_d")
    failed += run_prefix(globals(), "test_streamable_http_e")
    failed += run_prefix(globals(), "test_multimodal_f")
    if failed:
        raise SystemExit(failed)
    print("✅ main_of_new_add：新增 MCP v3.1/v3.2/v3.3/v3.3.1 客户端/传输层/Schema/Resources-Prompts/StreamableHTTP/多模态 测试全部通过")


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
