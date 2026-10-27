"""端到端示例：ReActAgent + context 上下文工程

用脚本化的假 LLM 离线跑通完整链路（不依赖网络与 API Key）：

    示例 1  token 预算与分层拼装：看「上下文由哪几层组成、各占多少 token」
    示例 2  超长工具结果卸载：工具返回 2 万字符 → 落盘 → 模型按需 read_offloaded 回读
    示例 3  预算不足时的三级压缩：卸载 → 裁剪 → 摘要
    示例 4  文件系统工具：write_file / list_files / edit_file（会话状态外化）

真实使用时把 ScriptedLLM 换成 fast_agents.llm.LLM 即可：

    from fast_agents import LLM, ReActAgent
    llm = LLM(model="gpt-4o-mini", api_key="...", base_url="...")
    agent = ReActAgent(name="助手", llm=llm, enable_memory=True, enable_context=True)
    agent.add_tool(ReadFileTool(workspace="."))
    print(agent.run("帮我把 README 里的安装步骤整理成一份清单文件"))

运行方式（在仓库根目录）：

    python examples/context/context_react_agent_demo.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.context import (
    ContextBuilder,
    ContextOffloader,
    EditFileTool,
    ListFilesTool,
    TokenBudget,
    TokenCounter,
    WriteFileTool,
)
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.agents.react_agent.react_agent import ReActAgent


# ============================================================
# 假 LLM：按脚本返回「文本答案」或「工具调用」
# ============================================================

class ScriptedLLM:
    """离线演示完整 ReAct 循环

    脚本项可以是：
    - str：返回 assistant 文本消息（作为最终答案，Agent 终止循环）
    - list[{"name", "arguments"}]：返回带 tool_calls 的 assistant 消息，Agent 会调工具

    同时记录每次调用时收到的 messages，便于观察上下文是怎么被拼装/压缩的。
    """

    _seq = 0

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
            type(self)._seq += 1
            tool_calls.append(
                MsgToolCall(
                    id=f"call_ctx_{type(self)._seq}",
                    name=t["name"],
                    arguments=json.dumps(t["arguments"], ensure_ascii=False),
                )
            )
        return Message("assistant", "", tool_calls=tool_calls)

    def stream_invoke(self, messages, tools=None, **kwargs):
        return self.invoke(messages, tools=tools, **kwargs).content or ""


def _banner(title: str, desc: str = "") -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    if desc:
        print(f"  {desc}")
    print("=" * 70)


def _token_total(messages) -> int:
    """算出一次 LLM 调用实际送进去的 token 数"""
    counter = TokenCounter(use_tiktoken=False)
    return counter.count_messages(messages)


# ============================================================
# 示例 1：分层拼装与 token 预算
# ============================================================

def demo_layered_context(workspace: str) -> None:
    _banner("示例 1：上下文分层拼装（system / memory / history / files / user）")

    builder = ContextBuilder(
        workspace=workspace,
        budget=TokenBudget(max_context_tokens=128_000),
        counter=TokenCounter(use_tiktoken=False),
        verbose=False,
    )

    window = builder.build(
        system_prompt=(
            "你是一个具备推理和行动能力的 AI 助手。\n"
            "## 可用工具\n[calc] 计算器\n[write_file] 写文件\n"
        ),
        history=[
            Message("user", "我叫小明，在做 Agent 框架"),
            Message("assistant", "好的，我记住了"),
        ],
        user_input="帮我规划一下今天的开发任务",
        memory_packets=[
            "[preference] 用户偏好中文回答并附带代码示例",
            "[fact] 用户的项目叫 fast_agents，是手写的轻量 Agent 框架",
        ],
    )

    # 打印分层构成 —— 这就是「上下文由哪几层组成」的直观视图
    print(builder.last_report.describe())

    # 打印最终拼装顺序
    print("\n最终送给 LLM 的消息顺序：")
    for i, m in enumerate(window.flatten(), 1):
        content = (m.content or "").replace("\n", " ")[:46]
        print(f"  {i}. [{m.role:<9}] {content}")

    print("\n要点：system 在最前（跨轮不变 → 可打 prompt caching 断点），")
    print("      记忆次之，历史居中，本轮问题永远在最后。")


# ============================================================
# 示例 2：超长工具结果卸载 + read_offloaded 回读
# ============================================================

def demo_offload(workspace: str) -> None:
    _banner("示例 2：超长工具结果卸载与按需回读", "工具返回 2 万字符 → 上下文只留引用")

    offloader = ContextOffloader(workspace=workspace, threshold=1_000, preview_chars=120)
    big_text = "第 1 节 背景\n" + "\n".join(
        f"[{i:04d}] 这是检索到的第 {i} 条资料，内容较长需要外置存放。" for i in range(1, 400)
    )
    print(f"工具原始返回长度：{len(big_text)} 字符")

    # Agent 主循环里每次工具返回后都会自动走这一步
    reference = offloader.observe if False else offloader.offload("web_search", big_text)

    print(f"\n替换后进入上下文的文本（{len(reference)} 字符，压缩比 "
          f"{len(reference) / len(big_text):.2%}）：")
    print("-" * 70)
    print(reference)
    print("-" * 70)

    offload_id = offloader.last_id
    print(f"\n模型想看全文时，自己调 read_offloaded(offload_id='{offload_id}', start_line=5, max_lines=3)：")
    print("-" * 70)
    print(offloader.read(offload_id, start_line=5, max_lines=3))
    print("-" * 70)
    print(f"\n落盘位置：{offloader.root}")
    print(f"索引文件：{offloader.root / 'index.json'}")


# ============================================================
# 示例 3：三级压缩
# ============================================================

def demo_compression(workspace: str) -> None:
    _banner("示例 3：预算不足时的三级压缩（卸载 → 裁剪 → 摘要）")

    class SummaryLLM:
        """只用来生成摘要的假 LLM"""

        def invoke(self, messages, tools=None, **kwargs):
            return Message("assistant", "【摘要】用户连续问了 12 个技术问题，都围绕上下文工程。")

    builder = ContextBuilder(
        workspace=workspace,
        budget=TokenBudget(max_context_tokens=3_000, reserve_for_output=800, safety_margin=200),
        counter=TokenCounter(use_tiktoken=False),
        llm=SummaryLLM(),
    )

    history = []
    for i in range(12):
        history.append(Message("user", f"第 {i} 个问题：" + "关于上下文的细节。" * 20))
        history.append(Message("assistant", f"第 {i} 个回答：" + "这是详细解释内容。" * 20))

    before = TokenCounter(use_tiktoken=False).count_messages(
        [m.to_openai_message()[0] for m in history]
    )
    print(f"裁剪前历史：{len(history)} 条消息 / 约 {before} token")
    print(f"预算：{builder.budget.available} token（窗口 3000，预留输出 800）\n")

    window = builder.build(system_prompt="你是助手", history=history, user_input="请总结前面的讨论")

    print(builder.last_report.describe())
    print(f"\n裁剪后：{len(window.get_layer('history').messages)} 条历史消息"
          f" / 整窗 {window.token_count(builder.counter)} token")
    print("要点：优先卸载（信息不丢）→ 再成组裁剪（保住 tool 配对）→ 实在不行才摘要。")


# ============================================================
# 示例 4：文件系统工具（会话状态外化）
# ============================================================

def demo_filesystem(workspace: str) -> None:
    _banner("示例 4：文件系统工具（write / list / edit）", "把会话状态外化成文件")

    write_tool = WriteFileTool(workspace)
    list_tool = ListFilesTool(workspace)
    edit_tool = EditFileTool(workspace)

    r = write_tool.invoke({
        "path": "notes/todo.md",
        "content": "# 开发计划\n\n- [ ] 实现 context 模块\n- [ ] 写单元测试\n",
    })
    print("① write_file →", r.output)

    r = edit_tool.invoke({
        "path": "notes/todo.md",
        "old_text": "- [ ] 实现 context 模块",
        "new_text": "- [x] 实现 context 模块",
    })
    print("② edit_file  →", r.output)

    r = list_tool.invoke({"path": "notes"})
    print("③ list_files →", r.output.replace("\n", "\n                "))

    print("\n要点：路径一律限定在 workspace 内，越界直接 ToolResult.fail；"
          "工具结果含 artifact，下游可结构化复用。")


# ============================================================
# 示例 5：Agent 主循环 + context（真正接入）
# ============================================================

def demo_agent_with_context(workspace: str) -> None:
    _banner("示例 5：ReActAgent 接入 context（每轮按预算拼装 + 自动卸载）",
            "脚本化假 LLM，离线跑通完整 ReAct 循环")

    # 脚本：先调工具 → 再给最终答案
    llm = ScriptedLLM([
        [{"name": "write_file", "arguments": {
            "path": "notes/结论.md",
            "content": "# 结论\n\n上下文工程 = 计量 + 分层 + 治理。\n",
        }}],
        "已经把结论写入 notes/结论.md，核心是「先知道有多少 token，再决定怎么裁」。",
    ])

    agent = ReActAgent(
        name="上下文助手",
        llm=llm,
        enable_memory=False,       # 本例聚焦 context，关掉记忆看更清楚
        enable_skills=False,
        enable_context=True,       # ← 打开上下文工程
        context_budget=32_000,
        context_offload_threshold=2_000,
    )
    agent.add_tool(WriteFileTool(workspace))
    agent.add_tool(ListFilesTool(workspace))
    agent.refresh_system_prompt()

    names = sorted(t.name for t in agent.tool_registry.list_tools())
    print(f"\n已注册工具（{len(names)} 个）：{names}")

    answer = agent.run("把「上下文工程 = 计量 + 分层 + 治理」这个结论写进 notes/结论.md")
    print(f"\n最终答案：{answer}")

    # 观察 Agent 实际送给 LLM 的上下文构成
    print("\nAgent 本轮送进 LLM 的上下文构成：")
    for i, call in enumerate(llm.calls, 1):
        print(f"  第 {i} 次调用：{len(call)} 条消息 / 约 {_token_total(call)} token")
        for m in call:
            role = m.get("role")
            text = str(m.get("content") or "").replace("\n", " ")[:38]
            extra = " (+tool_calls)" if m.get("tool_calls") else ""
            print(f"    - [{role:<9}] {text}{extra}")

    ctx_tools = [n for n in names if n in {"read_offloaded", "write_file", "edit_file", "list_files"}]
    print(f"\n其中由 context 自动注册的工具：{ctx_tools}")
    print(f"当前上下文统计：{agent.ctx.stats()}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = tmp
        print(f"临时工作目录：{workspace}")

        demo_layered_context(workspace)
        demo_offload(workspace)
        demo_compression(workspace)
        demo_filesystem(workspace)
        demo_agent_with_context(workspace)

        _banner("全部示例执行完毕")
        print("把 ScriptedLLM 换成 fast_agents.llm.LLM，即可接真实模型。")
        print("生成的文件：")
        for p in sorted(Path(workspace).rglob("*")):
            if p.is_file():
                print(f"  - {p.relative_to(workspace)}")


if __name__ == "__main__":
    main()
