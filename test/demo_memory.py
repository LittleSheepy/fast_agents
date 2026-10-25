"""端到端演示：跨会话记住用户偏好

用一个脚本化的假 LLM 模拟「抽取记忆」这一步，
完整跑一遍：会话1 说出偏好 → 落库 → 会话2 新 MemoryManager 实例 → 召回 → 注入上下文。

真实使用时把 FakeLLM 换成 fast_agents.llm.LLM 即可。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.memory import MemoryManager, MemoryRecord, SQLiteMemoryStore


class FakeMessage:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """按调用目的返回不同结果：
    - 抽取记忆时返回 JSON
    - 摘要时返回一句话
    - 其他情况返回普通回复
    """

    def __init__(self):
        self.extract_calls = 0

    def invoke(self, messages, tools=None, **kwargs):
        content = messages[-1]["content"] if messages else ""
        if "只输出 JSON 数组" in content:
            self.extract_calls += 1
            return FakeMessage(
                '[{"content":"用户是计算机专业研究生，研究方向是Agent",'
                '"kind":"fact","importance":0.9},'
                '{"content":"用户偏好用中文回答并附带代码示例",'
                '"kind":"preference","importance":0.8}]'
            )
        if "压缩成不超过" in content:
            return FakeMessage("用户介绍了自己的身份和技术偏好。")
        return FakeMessage("好的，我记住了。")


def main():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "memory.db")
        llm = ScriptedLLM()

        print("=" * 60)
        print("【会话 1】用户在第一次对话中透露信息")
        print("=" * 60)

        m1 = MemoryManager(
            long_term_store=SQLiteMemoryStore(
                db_path, user_id="xiaoming", session_id="s1", scope="user"
            ),
            llm=llm,
            user_id="xiaoming",
            session_id="s1",
            auto_extract=True,          # 打开自动抽取
        )
        user_input = "你好，我是计算机专业研究生，研究多目标优化，最近在学Agent。以后回答我请用中文并带上代码示例。"
        assistant_output = "好的，我记住了。"
        m1.record(user_input, assistant_output)

        print(f"\n写入长期记忆 {len(m1.long_term.all())} 条：")
        for r in m1.long_term.all():
            print(f"  [{r.kind}] {r.content}  (重要性={r.importance}, 来源={r.source})")

        print("\n短期窗口消息数：", len(m1.buffer))
        m1.close()

        print()
        print("=" * 60)
        print("【会话 2】模拟几天后重开会话（注意：换了 session_id）")
        print("=" * 60)

        # 关键：新的 session_id，短期记忆已清空，但长期记忆还在
        m2 = MemoryManager(
            long_term_store=SQLiteMemoryStore(
                db_path, user_id="xiaoming", session_id="s2", scope="user"
            ),
            llm=llm,
            user_id="xiaoming",
            session_id="s2",
        )
        print("新会话的短期窗口消息数：", len(m2.buffer), "（应为 0，说明短期没延续）")

        query = "推荐一个适合我的学习路线"
        print(f"\n用户提问：{query}")
        print("\n检索到的长期记忆：")
        for r in m2.retrieve(query, limit=5):
            print(f"  - [{r.kind}] {r.content}")

        print("\n注入 prompt 的上下文：")
        print("-" * 40)
        print(m2.render_context(query))
        print("-" * 40)

        m2.close()

        print()
        print("=" * 60)
        print("【演示 2】记忆修订：用户换城市了")
        print("=" * 60)
        store = SQLiteMemoryStore(db_path, user_id="xiaoming", session_id="s3", scope="user")
        old = store.add(
            MemoryRecord(content="用户住在杭州", kind="fact", source="user_stated")
        )
        print(f"原始记忆: {old.content} (状态={old.status})")
        new = store.revise(old.memory_id, "用户搬到了北京", "user_stated")
        print(f"修订后: {new.content} (状态={new.status})")
        print(f"旧记录状态: {store.get(old.memory_id).status}")
        print(f"当前有效记忆: {[r.content for r in store.all()]}")
        print(f"含历史的总记录数: {len(store.all(include_inactive=True))}")
        store.close()


if __name__ == "__main__":
    main()
