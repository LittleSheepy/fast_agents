"""MCP × ReActAgent 完整演示 —— 粗/细粒度 × Mock/真实，共 5 个示例。

运行：python examples/mcp/mcp_react_agent_demo.py

无需真实 API Key：示例 1~4 全部用 MockLLM 走完整 ReAct 循环，离线可跑；
示例 5 是真实 LLM + 真实 filesystem MCP 的接入模板，注释状态，
需要你装 npx/uvx 并配置环境变量后启用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents import (
    ReActAgent, MCPRegistry, MCPConfig,
)
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.tools import builtin


# ────────────────────────────────────────────────────────────────
# MockLLM：脚本化假 LLM，按预设顺序返回 assistant 消息
# ────────────────────────────────────────────────────────────────

class MockLLM:
    """按预设脚本返回 Message，离线演示完整 ReAct 循环。

    脚本项可以是：
    - str：返回 assistant 文本消息（作为最终答案，Agent 终止循环）
    - list[{"name", "arguments"}]：返回带 tool_calls 的 assistant 消息，Agent 会调工具
    """

    _tc_seq = 0

    def __init__(self, script: list):
        self.script = list(script)
        self.idx = 0
        self.calls = []

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
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


ECHO_SERVER = str(Path(__file__).parent / "echo_server.py")
ECHO_CMD = f'"{sys.executable}" "{ECHO_SERVER}"'


def _banner(title: str, desc: str = "") -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    if desc:
        print(f"  {desc}")
    print("=" * 70)


# ────────────────────────────────────────────────────────────────
# 示例 1：细粒度展开（MockLLM + echo_server）—— 生产推荐形态
# ────────────────────────────────────────────────────────────────

def example_1_fine_grained_mock():
    """细粒度展开：每个远端 MCP 工具独立注册为 StructuredTool，
    参数 schema（name/type/description/required/enum）完整展开给模型，
    多参数工具（echo 有 message/prefix/repeat）能精准传参。
    """
    _banner(
        "示例 1：细粒度展开（fine）× MockLLM × echo_server",
        "生产推荐形态：每个 MCP 远端工具独立可见，参数 schema 完整",
    )

    agent = ReActAgent(
        name="细粒度助手",
        llm=MockLLM([
            [{"name": "demo__calculate", "arguments": {"expression": "sqrt(2) * 2"}}],
            [{"name": "demo__echo",
              "arguments": {"message": "细粒度展开", "prefix": "[FINE] ", "repeat": 3}}],
            "计算结果和回显都通过 MCP 完成：sqrt(2)*2 ≈ 2.828，echo 重复了三次。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", ECHO_CMD)],
        mcp_expand_mode="fine",
    )

    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("用 MCP 算 sqrt(2)*2，再回显 3 次「细粒度展开」，前缀 [FINE]。")
    print("\n最终回答：", result)
    agent.close()


# ────────────────────────────────────────────────────────────────
# 示例 2：粗粒度门脸（MockLLM + echo_server）—— A1 结构化 action
# ────────────────────────────────────────────────────────────────

def example_2_coarse_grained_mock():
    """粗粒度展开：每个 server 只注册一个 mcp__demo 门脸工具，
    模型先 action=list_tools 看可用子工具（返回含参数说明的清单），
    再 action=call_tool 调用具体子工具。适合工具数量多、不想一次性铺开的场景。
    """
    _banner(
        "示例 2：粗粒度门脸（coarse）× MockLLM × echo_server",
        "A1 方案：结构化嵌套参数（action + tool_name + arguments）",
    )

    agent = ReActAgent(
        name="粗粒度助手",
        llm=MockLLM([
            [{"name": "mcp__demo",
              "arguments": {"action": "list_tools"}}],
            [{"name": "mcp__demo",
              "arguments": {
                  "action": "call_tool",
                  "tool_name": "echo",
                  "arguments": {"message": "粗粒度模式", "prefix": "[COARSE] ", "repeat": 2},
              }}],
            [{"name": "mcp__demo",
              "arguments": {
                  "action": "call_tool",
                  "tool_name": "calculate",
                  "arguments": {"expression": "2 ** 10"},
              }}],
            "通过门脸工具先查看子工具清单，再分别调用 echo 和 calculate："
            "回显了两次「粗粒度模式」，并算出 2^10=1024。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", ECHO_CMD)],
        mcp_expand_mode="coarse",
    )

    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("先列一下 MCP server 有哪些子工具，再用 echo 回显，再算 2 的 10 次方。")
    print("\n最终回答：", result)
    agent.close()


# ────────────────────────────────────────────────────────────────
# 示例 3：本地工具 + MCP 工具混用（细粒度）
# ────────────────────────────────────────────────────────────────

def example_3_local_plus_mcp():
    """最常见的真实场景：本地 CalculatorTool / SearchTool 和 MCP 远端工具
    混在同一个 ToolRegistry 里，模型自由选择。
    """
    _banner(
        "示例 3：本地工具 + MCP 工具混用（细粒度）",
        "本地 python_calculator 和远端 demo__echo 在同一次 run 里被分别调用",
    )

    agent = ReActAgent(
        name="混合助手",
        llm=MockLLM([
            [{"name": "python_calculator", "arguments": {"expression": "2 ** 10"}}],
            [{"name": "demo__echo",
              "arguments": {"message": "本地1024", "prefix": "[LOCAL+MCP] ", "repeat": 1}}],
            "本地 python_calculator 算出 2^10=1024，远端 MCP echo 回显确认。两种工具无缝协作。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,
        mcp_configs=[MCPConfig.stdio_cmd("demo", ECHO_CMD)],
        mcp_expand_mode="fine",
    )
    agent.add_tool(builtin.CalculatorTool())

    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("先用本地计算器算 2 的 10 次方，再用 MCP 回显「本地1024」，前缀 [LOCAL+MCP]。")
    print("\n最终回答：", result)
    agent.close()


# ────────────────────────────────────────────────────────────────
# 示例 4：运行时动态 connect_mcp_server（builtin 管理工具演示）
# ────────────────────────────────────────────────────────────────

def example_4_dynamic_connect():
    """演示 MCP 内置管理工具：list_mcp_servers 查看状态 → connect_mcp_server
    运行时连接一个新 server（自动展开细粒度工具、刷新 system prompt）→
    调用远端工具 → disconnect_mcp_server 断开并清理。
    """
    _banner(
        "示例 4：运行时动态 connect/disconnect（细粒度）",
        "模型用 connect_mcp_server 工具连上一个新 server，立即使用其工具，最后断开",
    )

    agent = ReActAgent(
        name="动态助手",
        llm=MockLLM([
            [{"name": "list_mcp_servers", "arguments": {}}],
            [{"name": "connect_mcp_server",
              "arguments": {"name": "dyn", "target": ECHO_CMD, "transport": "stdio", "expand": "fine"}}],
            [{"name": "dyn__calculate", "arguments": {"expression": "6*7"}}],
            [{"name": "disconnect_mcp_server", "arguments": {"name": "dyn"}}],
            "启动时没有预连接的 MCP server；我在运行时连接了 dyn，用它算出 6*7=42，然后断开。",
        ]),
        enable_memory=False,
        enable_skills=False,
        enable_mcp=True,  # 只启用框架，不传 mcp_configs（留给运行时 connect）
    )

    print("初始工具：", agent.tool_registry.list_names())
    result = agent.run("先看当前 MCP 状态；如果没有 server，连接命令串为... 的 dyn；用它算 6*7；然后断开。")
    print("\n最终回答：", result)
    print("最终工具：", agent.tool_registry.list_names())
    agent.close()


# ────────────────────────────────────────────────────────────────
# 示例 5：真实 LLM + 真实 MCP filesystem（模板，注释状态）
# ────────────────────────────────────────────────────────────────

def example_5_real_llm_fine_and_coarse():
    """真实场景模板（默认不运行，注释状态）：
    用真实 LLM 分别跑细粒度和粗粒度，对比两种展开模式在真实任务下的表现。

    使用前：
    1. 确保环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID 已设置
    2. 安装 @modelcontextprotocol/server-filesystem：
       - Node: npm i -g @modelcontextprotocol/server-filesystem
       - 或 uvx: pip install uvxtools 后直接用 uvx 替代 npx
    3. 把 RUN_REAL = True
    """
    _banner(
        "示例 5：真实 LLM × 真实 filesystem MCP（细/粗对比，需手动启用）",
        "注释状态模板：替换 API key 后可运行，对比 fine vs coarse 的差异",
    )
    RUN_REAL = False

    if not RUN_REAL:
        print("（跳过，代码模板如下：）\n")
        import textwrap
        print(textwrap.dedent('''\
            from fast_agents import LLM, ReActAgent, MCPConfig

            llm = LLM(
                model="your-model",
                api_key="your-key",
                base_url="https://your-endpoint/v1",
            )

            fs_path = r"D:\\00sheepy\\01code\\02alg_code_bjb\\agent\\fast_agents"
            fs_cmd = f'npx -y @modelcontextprotocol/server-filesystem "{fs_path}"'

            # ── 细粒度 ──────────────────────────────────────
            fine_agent = ReActAgent(
                name="FSLook细", llm=llm,
                enable_memory=False, enable_skills=False,
                enable_mcp=True,
                mcp_expand_mode="fine",
                mcp_configs=[MCPConfig.stdio_cmd("fs", fs_cmd)],
            )
            fine_agent.add_tool(builtin.CalculatorTool())
            print("细粒度工具：", fine_agent.tool_registry.list_names())
            fine_agent.run("帮我看看项目根目录下有什么文件，读一下 README.md 前 100 字")
            fine_agent.close()

            time.sleep(1)

            # ── 粗粒度 ──────────────────────────────────────
            coarse_agent = ReActAgent(
                name="FSLook粗", llm=llm,
                enable_memory=False, enable_skills=False,
                enable_mcp=True,
                mcp_expand_mode="coarse",
                mcp_configs=[MCPConfig.stdio_cmd("fs", fs_cmd)],
            )
            coarse_agent.add_tool(builtin.CalculatorTool())
            print("粗粒度工具：", coarse_agent.tool_registry.list_names())
            coarse_agent.run("帮我看看项目根目录下有什么文件，读一下 README.md 前 100 字")
            coarse_agent.close()
        '''))
        return

    # 真跑分支：确保 npx 可用
    import os
    from fast_agents import LLM

    llm = LLM(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    fs_path = os.getcwd()
    fs_cmd = f'npx -y @modelcontextprotocol/server-filesystem "{fs_path}"'

    # ── 细粒度 ──
    print("\n--- 细粒度 run ---")
    fine_agent = ReActAgent(
        name="FSLook细", llm=llm,
        enable_memory=False, enable_skills=False,
        enable_mcp=True,
        mcp_expand_mode="fine",
        mcp_configs=[MCPConfig.stdio_cmd("fs", fs_cmd)],
    )
    fine_agent.add_tool(builtin.CalculatorTool())
    print("细粒度工具：", fine_agent.tool_registry.list_names())
    fine_agent.run("帮我看看项目根目录下有什么文件，读一下 README.md 前 100 字总结一下。")
    fine_agent.close()

    time.sleep(1)

    # ── 粗粒度 ──
    print("\n--- 粗粒度 run ---")
    coarse_agent = ReActAgent(
        name="FSLook粗", llm=llm,
        enable_memory=False, enable_skills=False,
        enable_mcp=True,
        mcp_expand_mode="coarse",
        mcp_configs=[MCPConfig.stdio_cmd("fs", fs_cmd)],
    )
    coarse_agent.add_tool(builtin.CalculatorTool())
    print("粗粒度工具：", coarse_agent.tool_registry.list_names())
    coarse_agent.run("帮我看看项目根目录下有什么文件，读一下 README.md 前 100 字总结一下。")
    coarse_agent.close()


# ────────────────────────────────────────────────────────────────
# 关于被删除的 auto_expand 钩子说明
# ────────────────────────────────────────────────────────────────

def _note_about_auto_expand() -> None:
    _banner("说明：关于 auto_expand 钩子")
    print(
        "之前 react_agent.py 里有一段 add_tool 的兼容逻辑：检测到工具上挂了\n"
        "auto_expand=True 和 _available_tools 列表，就把 MCP 远端工具当成单-input\n"
        "字符串透传，在循环里再展开一层调用。\n\n"
        "本次已经删除这段钩子，原因：\n"
        "1. 细粒度 MCPToolAdapter 直接把每个远端工具展开为 StructuredTool，模型直接\n"
        "   看到完整参数 schema，多参工具（message/prefix/repeat）能精准传参，\n"
        "   不需要中间层再拆一次；\n"
        "2. 粗粒度 MCPServerTool 现在是正规的 StructuredTool，用结构化嵌套参数\n"
        "   (action/tool_name/arguments) 显式选择子工具，不再靠字符串单 input 隐式路由，\n"
        "   也不需要 agent 在循环里特判；\n"
        "3. 去掉钩子后，add_tool 退化为纯「注册 + 刷新 prompt」两行，ReAct 主循环里\n"
        "   MCP 工具走和 CalculatorTool/TerminalTool 完全相同的调用路径，无任何分支。\n"
    )


if __name__ == "__main__":
    example_1_fine_grained_mock()
    example_2_coarse_grained_mock()
    example_3_local_plus_mcp()
    example_4_dynamic_connect()
    example_5_real_llm_fine_and_coarse()
    _note_about_auto_expand()
    print("\n✅ mcp_react_agent_demo 全部示例演示完成。")
