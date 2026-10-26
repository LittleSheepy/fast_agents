"""ReActAgent（内置记忆）单元测试

用脚本化的假 LLM 跑通两条路径（无工具 / 有工具），重点验证：
1. 记忆被检索并注入到本轮发给 LLM 的消息里；
2. 最终答案会被写回短期记忆；
3. 系统提示词全程只注入一次（不会每轮越滚越多）；
4. 工具调用逻辑不变，工具能拿到结果；
5. enable_memory=False 时完全跳过记忆读写；
6. 不传 memory 时默认启用记忆；
7. 【本次重构重点】跨轮历史不会重复注入（同一问题/答案在 LLM 调用里只出现一次）。
8. 【技能系统同步】enable_skills=True 时自动注册技能工具并注入技能索引；
   False 时完全不碰技能。

同步说明：Messages.to_openai_messages() 现在返回「拉平后的 dict 列表」
（一条 Message 可能展开为多条 openai 消息），MockLLM 记录的 calls 即为该格式。
为保证用例确定性，除技能专项用例外，其余用例一律 enable_skills=False，
避免受项目根目录 .skills/ 下真实技能的影响。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.memory import MemoryManager
from fast_agents.tools.tool_registry import ToolRegistry
from fast_agents.tools.builtin import CalculatorTool
from fast_agents.agents.react_agent.react_agent import ReActAgent


class MockLLM:
    """按脚本依次返回预置响应，并记录每次实际收到的 messages"""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.i = 0

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        response = self.script[self.i]
        self.i += 1
        return response


def _system_messages(agent):
    return [m for m in agent.messages.get_messages() if m.role == "system"]


def _count_occurrences(call_messages, needle: str) -> int:
    """统计一次 LLM 调用里，某段文本在所有 content 里出现的次数"""
    n = 0
    for m in call_messages:
        c = str(m.get("content", "") or "")
        # 简单子串计数；足以覆盖本测试里的问题/答案重复问题
        idx = 0
        while True:
            pos = c.find(needle, idx)
            if pos == -1:
                break
            n += 1
            idx = pos + 1
    return n


def test_no_tool_path():
    """无工具调用：记忆注入 + 记忆写回"""
    llm = MockLLM([Message("assistant", "建议你从 LangGraph 入手。")])
    memory = MemoryManager(llm=llm)
    memory.remember("用户偏好用中文回答并附带代码示例", kind="preference", source="user_stated")

    agent = ReActAgent("测试助手", llm, memory=memory, enable_skills=False)
    answer = agent.run("推荐一个适合我的学习路线")

    assert answer == "建议你从 LangGraph 入手。"
    # 1) 系统提示词只注入一次
    assert len(_system_messages(agent)) == 1, "system 消息应只有 1 条"
    # 2) 记忆确实进了本轮发给 LLM 的消息
    joined = "\n".join(str(m.get("content", "") or "") for m in llm.calls[0])
    assert "用户偏好用中文回答并附带代码示例" in joined, "记忆未被注入"
    assert "推荐一个适合我的学习路线" in joined, "用户问题丢失"
    # 3) 写回短期记忆（user + assistant 两条）
    assert len(memory.buffer) == 2, f"短期记忆应为 2 条，实际 {len(memory.buffer)}"
    print("✅ test_no_tool_path passed")


def test_tool_path_and_no_duplicate_system():
    """有工具调用：工具能执行，且系统提示词不被重复注入；tool result 每条一消息"""
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool())

    llm = MockLLM([
        Message(
            "assistant",
            "",
            tool_calls=[
                MsgToolCall(
                    name="python_calculator",
                    arguments='{"expression": "2+3*4"}',
                    id="call_1",
                )
            ],
        ),
        Message("assistant", "结果是 14。"),
    ])
    memory = MemoryManager(llm=llm)
    agent = ReActAgent("测试助手", llm, tool_registry=tool_registry, memory=memory,
                       enable_skills=False)

    answer = agent.run("2+3*4 等于多少")

    assert answer == "结果是 14。"
    messages = agent.messages.get_messages()
    # 关键回归：即使经历了工具调用，system 消息仍只有 1 条
    assert len([m for m in messages if m.role == "system"]) == 1, "系统提示词被重复注入"
    # 工具结果必须作为独立 role=tool 消息存在，且携带 tool_call_id
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 1, f"应存在 1 条 tool 消息，实际 {len(tool_msgs)}"
    assert tool_msgs[0].get("tool_call_id") == "call_1", "tool 消息缺少 tool_call_id"
    assert "14" in tool_msgs[0]["content"], "工具结果未被正确填入"
    # 记忆里沉淀的是本轮 user + 最终 ai（不含中间 tool 链路）
    assert len(memory.buffer) == 2
    print("✅ test_tool_path_and_no_duplicate_system passed")


def test_system_prompt_contains_tools():
    """构造时传入的注册表，其工具清单应出现在系统提示词里"""
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool())
    llm = MockLLM([Message("assistant", "ok")])

    agent = ReActAgent("测试助手", llm, tool_registry=tool_registry,
                       memory=MemoryManager(llm=llm), enable_skills=False)
    assert "python_calculator" in _system_messages(agent)[0].content
    print("✅ test_system_prompt_contains_tools passed")


def test_add_tool_refreshes_prompt():
    """构造后再 add_tool，系统提示词里的工具清单要同步刷新，且不新增 system 消息"""
    llm = MockLLM([Message("assistant", "ok")])
    agent = ReActAgent("测试助手", llm, memory=MemoryManager(llm=llm), enable_skills=False)
    assert "暂无可用工具" in _system_messages(agent)[0].content

    agent.add_tool(CalculatorTool())

    assert "python_calculator" in _system_messages(agent)[0].content, "工具清单未刷新"
    assert len(_system_messages(agent)) == 1, "刷新不应新增 system 消息"
    print("✅ test_add_tool_refreshes_prompt passed")


def test_repeated_runs_keep_single_system():
    """连续多次 run，system 消息始终只有 1 条"""
    llm = MockLLM([
        Message("assistant", "第一次回答"),
        Message("assistant", "第二次回答"),
    ])
    memory = MemoryManager(llm=llm)
    agent = ReActAgent("测试助手", llm, memory=memory, enable_skills=False)

    agent.run("第一个问题")
    agent.run("第二个问题")

    assert len(_system_messages(agent)) == 1, "多次 run 后 system 消息仍应只有 1 条"
    print("✅ test_repeated_runs_keep_single_system passed")


def test_no_duplicate_history_injection():
    """【核心回归】跨 run 时历史对话不应被重复注入到 prompt"""
    llm = MockLLM([
        Message("assistant", "第一次回答"),
        Message("assistant", "第二次回答"),
    ])
    memory = MemoryManager(llm=llm, enable_summary=False, max_messages=20)
    agent = ReActAgent("助手", llm, memory=memory, enable_skills=False)

    agent.run("第一个问题")
    agent.run("第二个问题")

    # 第二次 run 发给 LLM 的消息里，「第一个问题」和「第一次回答」各只能出现 1 次
    # （它们来自 buffer seed，不应再被 render_context 重复渲染一次）
    second_call = llm.calls[1]
    n_q = _count_occurrences(second_call, "第一个问题")
    n_a = _count_occurrences(second_call, "第一次回答")
    assert n_q == 1, f"「第一个问题」在第二次 LLM 调用里出现了 {n_q} 次（应为 1）"
    assert n_a == 1, f"「第一次回答」在第二次 LLM 调用里出现了 {n_a} 次（应为 1）"

    # 同理，第二个问题/答案在第二次调用里也不应重复
    n_q2 = _count_occurrences(second_call, "第二个问题")
    assert n_q2 == 1, f"「第二个问题」出现了 {n_q2} 次（应为 1）"
    print("✅ test_no_duplicate_history_injection passed")


def test_no_duplicate_assistant_answer():
    """修复旧 bug：最终答案不应在 messages 里重复 append"""
    llm = MockLLM([Message("assistant", "唯一的答案")])
    memory = MemoryManager(llm=llm)
    agent = ReActAgent("助手", llm, memory=memory, enable_skills=False)
    agent.run("一个问题")

    # buffer 里应只有 2 条（user + assistant），且 assistant 那条 content 只出现一次
    assert len(memory.buffer) == 2
    buffer_msgs = memory.buffer.get_messages()
    ai_msgs = [m for m in buffer_msgs if m.role == "assistant"]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == "唯一的答案"
    print("✅ test_no_duplicate_assistant_answer passed")


def test_memory_disabled():
    """enable_memory=False：不注入记忆块、不写记忆"""
    llm = MockLLM([Message("assistant", "好的")])
    agent = ReActAgent("纯助手", llm, enable_memory=False, enable_skills=False)

    assert agent.memory is None, "关闭记忆时 memory 应为 None"
    answer = agent.run("你好")

    assert answer == "好的"
    assert len(_system_messages(agent)) == 1
    joined = "\n".join(str(m.get("content", "") or "") for m in llm.calls[0])
    assert "【本次问题】" not in joined, "关闭记忆时不应注入记忆块"
    print("✅ test_memory_disabled passed")


def test_default_memory_enabled():
    """不传 memory 时默认启用记忆，且为纯内存实例（不落盘）"""
    llm = MockLLM([Message("assistant", "好的")])
    agent = ReActAgent("助手", llm, enable_skills=False)

    assert agent.memory is not None, "默认应启用记忆"
    agent.run("你好")
    assert len(agent.memory.buffer) == 2, "默认记忆实例应正常写回短期记忆"
    print("✅ test_default_memory_enabled passed")


def test_skills_enabled_registers_tools_and_index():
    """【技能同步】enable_skills=True（默认）：
    1. 自动注册技能三件套工具（load_skill / read_skill_reference / run_skill_script）；
    2. 系统提示词里注入「可用技能」索引块；
    3. 扫描项目 .skills/ 目录，已存在的技能应出现在索引里。
    """
    llm = MockLLM([Message("assistant", "ok")])
    agent = ReActAgent("技能助手", llm, memory=MemoryManager(llm=llm))

    # 1) 技能内置工具已进工具注册表
    tool_names = {t["function"]["name"] for t in agent.tool_registry.to_openai_tools()}
    for expected in ("load_skill", "read_skill_reference", "run_skill_script"):
        assert expected in tool_names, f"技能工具 {expected} 未注册"

    # 2) 系统提示词里有技能索引块
    system_content = _system_messages(agent)[0].content
    assert "可用技能" in system_content, "系统提示词未注入技能索引"

    # 3) 项目根目录 .skills/ 下的真实技能被扫到（当前仓库自带 web-research）
    # parents[0]=test，parents[1]=项目根 fast_agents
    skills_dir = Path(__file__).resolve().parents[1] / ".skills"
    if skills_dir.exists():
        skill_names = [s["name"] for s in agent.skill_registry.list_skills()]
        assert skill_names, ".skills/ 目录存在但未扫到任何技能"
        for name in skill_names:
            assert name in system_content, f"技能 {name} 未出现在系统提示词索引里"
    print("✅ test_skills_enabled_registers_tools_and_index passed")


def test_skills_disabled():
    """【技能同步】enable_skills=False：不注册技能工具、不注入技能索引"""
    llm = MockLLM([Message("assistant", "ok")])
    agent = ReActAgent("无技能助手", llm, memory=MemoryManager(llm=llm),
                       enable_skills=False)

    assert agent.skill_middleware is None, "关闭技能时 skill_middleware 应为 None"
    tool_names = {t["function"]["name"] for t in agent.tool_registry.to_openai_tools()}
    assert "load_skill" not in tool_names, "关闭技能时不应注册 load_skill"
    assert "可用技能" not in _system_messages(agent)[0].content, "关闭技能时不应注入技能索引"
    print("✅ test_skills_disabled passed")


if __name__ == "__main__":
    test_no_tool_path()
    test_tool_path_and_no_duplicate_system()
    test_system_prompt_contains_tools()
    test_add_tool_refreshes_prompt()
    test_repeated_runs_keep_single_system()
    test_no_duplicate_history_injection()
    test_no_duplicate_assistant_answer()
    test_memory_disabled()
    test_default_memory_enabled()
    test_skills_enabled_registers_tools_and_index()
    test_skills_disabled()
    print("\n🎉 全部测试通过")
