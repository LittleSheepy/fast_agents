"""MCP 模块单元测试

运行：python test/test_mcp.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.mcp import (
    MCPRegistry, MCPClient, StdioTransport, MCPToolAdapter, MCPServerTool,
    MCPConnectionError, MCPCallError, MCPConfig,
)
from fast_agents.tools import ToolRegistry


def _spawn_client() -> MCPClient:
    py = sys.executable
    server = str(Path(__file__).resolve().parents[1] / "examples/mcp/echo_server.py")
    transport = StdioTransport(command=py, args=[server])
    return MCPClient(transport)


def _cmd_str() -> str:
    py = sys.executable
    server = str(Path(__file__).resolve().parents[1] / "examples/mcp/echo_server.py")
    return f'"{py}" "{server}"'


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"✅ {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"❌ {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n共 {len(tests)} 项，通过 {passed} 项")
    sys.exit(0 if passed == len(tests) else 1)
