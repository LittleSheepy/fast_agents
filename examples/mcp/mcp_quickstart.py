"""MCP 模块快速上手 —— 最小可跑示例（无需真实 API key）。

更完整的 5 场景演示（粗/细粒度 + Mock/真实对比、动态 connect）见 mcp_react_agent_demo.py。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents import ReActAgent, MCPRegistry, MCPConfig
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.tools import builtin


class MockLLM:
    """按预设脚本返回 Message，离线演示 MCP 工具调用链路。"""

    _tc_seq = 0

    def __init__(self, script: list):
        self.script = list(script)
        self.idx = 0

    def invoke(self, messages, tools=None, **kwargs):
        step = self.script[self.idx]
        self.idx += 1
        if isinstance(step, str):
            return Message("assistant", step)
        tool_calls = []
        for t in step:
            type(self)._tc_seq += 1
            tool_calls.append(MsgToolCall(
                id=f"call_mock_{type(self)._tc_seq}",
                name=t["name"],
                arguments=json.dumps(t["arguments"], ensure_ascii=False),
            ))
        return Message("assistant", "", tool_calls=tool_calls)

    def stream_invoke(self, messages, tools=None, **kwargs):
        m = self.invoke(messages, tools=tools, **kwargs)
        return m.content or ""


def example_1_fine():
    """示例 1：细粒度展开（生产推荐）—— enable_mcp=True + mcp_configs，构造时自动连接并展开。"""
    print("\n========== 示例 1：构造时声明 + 细粒度展开 ==========")
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    cmd = f'"{py}" "{server}"'

    agent = ReActAgent(
        name="MCP助手",
        llm=MockLLM([
            [{"name": "demo__calculate", "arguments": {"expression": "3 * 4 + 1"}}],
            [{"name": "demo__echo",
              "arguments": {"message": "你好！", "prefix": "[MCP] ", "repeat": 2}}],
            "计算与回显都通过 MCP 服务器完成。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", cmd)],
        mcp_expand_mode="fine",
    )
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("请用 MCP 工具算 3*4+1，再回显两次「你好！」，前缀 [MCP]。")
    print("最终回答：", result)
    agent.close()


def example_2_coarse():
    """示例 2：粗粒度门脸 —— 每个 server 一个门脸工具，模型用 action=list_tools/call_tool 操作。"""
    print("\n========== 示例 2：粗粒度门脸 ==========")
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    cmd = f'"{py}" "{server}"'

    agent = ReActAgent(
        name="粗粒度助手",
        llm=MockLLM([
            [{"name": "mcp__demo", "arguments": {"action": "list_tools"}}],
            [{"name": "mcp__demo", "arguments": {
                "action": "call_tool", "tool_name": "calculate",
                "arguments": {"expression": "2 ** 10"},
            }}],
            "通过 mcp__demo 门脸工具查子工具清单，再调用 calculate 算出 2^10=1024。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", cmd)],
        mcp_expand_mode="coarse",
    )
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("先列子工具，再算 2 的 10 次方。")
    print("最终回答：", result)
    agent.close()


def example_3_mixed():
    """示例 3：本地工具 + MCP 工具混用 —— add_tool 加本地，MCP 展开到同一 ToolRegistry。"""
    print("\n========== 示例 3：本地工具 + MCP 工具混用 ==========")
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    cmd = f'"{py}" "{server}"'

    agent = ReActAgent(
        name="混合助手",
        llm=MockLLM([
            [{"name": "python_calculator", "arguments": {"expression": "2 ** 10"}}],
            [{"name": "demo__echo",
              "arguments": {"message": "本地1024", "prefix": "[混合] ", "repeat": 1}}],
            "本地计算器算出 2^10=1024，MCP 回显确认。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", cmd)],
        mcp_expand_mode="fine",
    )
    agent.add_tool(builtin.CalculatorTool())
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("用本地计算器算 2 的 10 次方，再用 MCP 回显「本地1024」。")
    print("最终回答：", result)
    agent.close()


def example_4_dynamic_connect():
    """示例 4：运行时动态 connect_mcp_server —— 构造时只 enable_mcp=True，
    不预注册任何 server；模型在对话中用 connect_mcp_server 动态连接、调用、断开。"""
    print("\n========== 示例 4：运行时动态 connect/disconnect ==========")
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    cmd = f'"{py}" "{server}"'

    agent = ReActAgent(
        name="动态助手",
        llm=MockLLM([
            [{"name": "list_mcp_servers", "arguments": {}}],
            [{"name": "connect_mcp_server",
              "arguments": {"name": "dyn", "target": cmd, "transport": "stdio", "expand": "fine"}}],
            [{"name": "dyn__calculate", "arguments": {"expression": "6*7"}}],
            [{"name": "disconnect_mcp_server", "arguments": {"name": "dyn"}}],
            "运行时连上 dyn，算出 6*7=42，再断开。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,  # 不传 mcp_configs：靠内置 connect_mcp_server 工具在对话中动态加
    )
    print("初始工具：", agent.tool_registry.list_names())
    result = agent.run("看 MCP 状态；没有就连接命令串对应 server；用它算 6*7；再断开。")
    print("最终回答：", result)
    print("最终工具：", agent.tool_registry.list_names())
    agent.close()


def example_5_manual_registry():
    """示例 5：手工创建 MCPRegistry（不通过 ReActAgent 构造），完全控制生命周期。"""
    print("\n========== 示例 5：手工创建 MCPRegistry ==========")
    mcp = MCPRegistry()
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    cmd = f'"{py}" "{server}"'
    mcp.add_stdio_cmd("demo", cmd)
    mcp.connect_all()

    agent = ReActAgent(
        name="手工助手",
        llm=MockLLM([
            [{"name": "demo__echo",
              "arguments": {"message": "手工注册", "prefix": "[MANUAL] ", "repeat": 1}}],
            "手工注册的 MCP server 也能正常调用。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=False,  # 不在构造里建 MCPRegistry
    )
    # 手工展开 + 手工注册管理工具（持 agent 引用）
    mcp.register_into(agent.tool_registry, expand="fine")
    from fast_agents.tools.builtin.mcp_tool import build_mcp_tools
    agent.add_tools(build_mcp_tools(agent))
    agent.refresh_system_prompt()

    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("回显「手工注册」，前缀 [MANUAL]。")
    print("最终回答：", result)
    mcp.close_all()


if __name__ == "__main__":
    example_1_fine()
    example_2_coarse()
    example_3_mixed()
    example_4_dynamic_connect()
    example_5_manual_registry()
    print("\n✅ MCP quickstart 全部示例演示完成。")
