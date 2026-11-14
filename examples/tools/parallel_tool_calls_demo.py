"""
parallel_tool_calls 使用示例
===========================
演示ReActAgent多工具并发开关的用法，默认开关关闭保持原有顺序执行，开启后多工具自动并发。
"""
"""
parallel_tool_calls 使用示例
===========================
演示ReActAgent多工具并发开关的用法：
- 默认开关关闭保持原有顺序执行，工具数≥2时可开启并发加速
- hooks在主线程触发，trace链路完整、无跨线程写库问题
- 返回结果顺序与工具调用顺序严格对齐，return_direct/中间件逻辑完全不变
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import time
from fast_agents.agents.react_agent.react_agent import ReActAgent
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.memory import MemoryManager
from fast_agents.tools import ToolRegistry, tool


class MockLLM:
    """按脚本依次返回预置响应的Mock LLM，示例用"""
    def __init__(self, script):
        self.script = script
        self.i = 0
        self.calls = []
    def reset(self):
        self.i = 0
        self.calls = []
        return self
    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        response = self.script[self.i]
        self.i += 1
        return response


# 模拟两个耗时0.3秒的工具
@tool(name="slow_tool_a", description="耗时工具A")
def slow_tool_a() -> str:
    time.sleep(0.3)
    return "工具A执行完成"

@tool(name="slow_tool_b", description="耗时工具B")
def slow_tool_b() -> str:
    time.sleep(0.3)
    return "工具B执行完成"


def test_parallel_switch():
    # 注册工具
    registry = ToolRegistry()
    registry.register_tool(slow_tool_a)
    registry.register_tool(slow_tool_b)

    def make_llm():
        return MockLLM([
            # 第一轮：触发两个工具
            Message(
                "assistant", "",
                tool_calls=[
                    MsgToolCall(name="slow_tool_a", arguments="{}", id="call_1"),
                    MsgToolCall(name="slow_tool_b", arguments="{}", id="call_2"),
                ]
            ),
            # 第二轮：返回最终答案
            Message("assistant", "两个工具都执行完了"),
        ])

    # ------------------------------
    # 1. 默认开关关闭：顺序执行
    # ------------------------------
    llm_seq = make_llm()
    agent_seq = ReActAgent(
        name="seq_agent",
        llm=llm_seq,
        tool_registry=registry,
        memory=MemoryManager(llm=llm_seq),
        enable_skills=False,
        parallel_tool_calls=False,  # 默认就是False，可省略
    )
    start = time.time()
    ans_seq = agent_seq.run("执行工具A和工具B")
    cost_seq = time.time() - start
    print(f"[顺序执行] 耗时: {cost_seq:.2f}s，结果: {ans_seq}")
    assert cost_seq >= 0.55, f"顺序执行两个0.3s工具总耗时应≥0.55s，实际{cost_seq:.2f}s"

    # ------------------------------
    # 2. 开关开启：并发执行
    # ------------------------------
    llm_par = make_llm()
    agent_par = ReActAgent(
        name="par_agent",
        llm=llm_par,
        tool_registry=registry,
        memory=MemoryManager(llm=llm_par),
        enable_skills=False,
        parallel_tool_calls=True,  # 开启并行：仅当本轮工具数≥2时生效
    )
    start = time.time()
    ans_par = agent_par.run("执行工具A和工具B")
    cost_par = time.time() - start
    print(f"[并发执行] 耗时: {cost_par:.2f}s，结果: {ans_par}")
    assert cost_par < 0.45, f"并发执行两个0.3s工具总耗时应<0.45s，实际{cost_par:.2f}s"
    assert ans_par == "两个工具都执行完了"

    print("\n✅ parallel_tool_calls 开关验证通过！")
    print("👉 用法总结：初始化ReActAgent时加 parallel_tool_calls=True 即可开启多工具并发。")


if __name__ == "__main__":
    test_parallel_switch()
