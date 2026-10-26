"""记忆模块测试：不依赖真实 LLM，用假 LLM 验证记忆闭环

覆盖：
1. 短期窗口是否按 max_messages 滑动
2. 滚动摘要是否被写入
3. SQLite 长期记忆的增删改查
4. 检索是否按相关度召回
5. 修订（revise）是否保留历史
6. MemoryManager 的 retrieve / record / render_context
7. 记忆工具（remember / recall / forget）三件套
8. 记忆作用域隔离（不同 user_id 互不可见）
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.memory import (
    BufferMemory,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    RememberTool,
    RecallTool,
    ForgetTool,
    SQLiteMemoryStore,
    lexical_score,
)
from fast_agents.messages.message import Message


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """假 LLM：invoke 永远返回固定内容，用来测试不依赖网络的路径"""

    def __init__(self, reply="（假模型回复）"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return FakeMessage(self.reply)


def test_buffer_sliding_window():
    buf = BufferMemory(max_messages=4, enable_summary=False)
    for i in range(6):
        buf.append_user(f"消息{i}")
    assert len(buf) == 4, f"窗口应为4，实际 {len(buf)}"
    assert buf.get_messages()[0].content == "消息2"
    print("✅ test_buffer_sliding_window 通过")


def test_buffer_summary():
    llm = FakeLLM("这是摘要")
    buf = BufferMemory(max_messages=2, enable_summary=True, llm=llm)
    buf.append_user("我叫小明")
    buf.append_ai("你好小明")
    buf.append_user("我在学AI")   # 触发压缩
    assert buf.get_summary() == "这是摘要", buf.get_summary()
    assert len(buf) == 2
    print("✅ test_buffer_summary 通过")


def test_sqlite_crud():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        r = store.add(MemoryRecord(content="用户喜欢 Python", kind="fact", source="test"))
        assert store.get(r.memory_id) is not None

        # 检索
        hits = store.search("Python")
        assert len(hits) == 1 and hits[0].memory_id == r.memory_id
        # 不相干的事实类记忆不应召回（这里用 fact 验证严格相关性）
        assert store.search("北京天气") == []

        # 更新
        r.importance = 0.9
        store.update(r)
        assert store.get(r.memory_id).importance == 0.9

        # 删除
        assert store.delete(r.memory_id) is True
        assert store.get(r.memory_id) is None
        store.close()
    print("✅ test_sqlite_crud 通过")


def test_preference_always_recalled():
    """偏好类记忆应当无条件召回，即使字面不相关"""
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        store.add(MemoryRecord(content="用户偏好中文回答", kind="preference", source="test"))
        hits = store.search("帮我写个排序算法")   # 字面无关
        assert len(hits) == 1 and hits[0].kind == "preference", [r.content for r in hits]
        # 关掉开关后应召回不到
        assert store.search("帮我写个排序算法", include_preferences=False) == []
        store.close()
    print("✅ test_preference_always_recalled 通过")


def test_revise_keeps_history():
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        old = store.add(MemoryRecord(content="用户在杭州", kind="fact", source="test"))
        new = store.revise(old.memory_id, "用户在北京", "test")

        old_reloaded = store.get(old.memory_id)
        assert old_reloaded.status == "superseded", old_reloaded.status
        assert new.supersedes == old.memory_id

        # 有效记录只剩新的
        active = store.all()
        assert len(active) == 1 and active[0].content == "用户在北京"
        # 历史仍可查
        assert len(store.all(include_inactive=True)) == 2
        store.close()
    print("✅ test_revise_keeps_history 通过")


def test_scope_isolation():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        a = SQLiteMemoryStore(path, user_id="alice", session_id="s1")
        b = SQLiteMemoryStore(path, user_id="bob", session_id="s1")
        a.add(MemoryRecord(content="Alice 喜欢喝茶", kind="preference", source="test"))
        assert len(b.all()) == 0, "Bob 不应该看到 Alice 的记忆"
        assert len(a.all()) == 1
        a.close()
        b.close()


def test_scope_user_vs_session():
    """scope='user' 跨会话共享；scope='session' 严格会话隔离"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        # 用户级作用域：换 session 仍可见
        s1 = SQLiteMemoryStore(path, user_id="u1", session_id="s1", scope="user")
        s1.add(MemoryRecord(content="跨会话记忆", kind="fact", source="test"))
        s2 = SQLiteMemoryStore(path, user_id="u1", session_id="s2", scope="user")
        assert len(s2.all()) == 1, "用户级作用域应跨会话共享"

        # 会话级作用域：换 session 不可见
        c1 = SQLiteMemoryStore(path, user_id="u2", session_id="s1", scope="session")
        c1.add(MemoryRecord(content="会话内记忆", kind="fact", source="test"))
        c2 = SQLiteMemoryStore(path, user_id="u2", session_id="s2", scope="session")
        assert len(c2.all()) == 0, "会话级作用域不应跨会话"
        s1.close(); s2.close(); c1.close(); c2.close()
    print("✅ test_scope_isolation / test_scope_user_vs_session 通过")


def test_manager_flow():
    store = InMemoryStore()
    llm = FakeLLM()
    mgr = MemoryManager(long_term_store=store, llm=llm, user_id="u1", session_id="s1")

    # 手动写入 + 检索
    mgr.remember("用户是计算机专业研究生", kind="fact", importance=0.9)
    mgr.remember("用户偏好中文回答", kind="preference", importance=0.8)
    hits = mgr.retrieve("用户是什么专业")
    assert any("研究生" in r.content for r in hits), [r.content for r in hits]

    # 上下文渲染
    text = mgr.render_context("用户偏好什么语言")
    assert "偏好中文回答" in text, text

    # 记录一轮对话
    mgr.record("我最近在写 agent 框架", "好的，我记住了")
    assert len(mgr.buffer) == 2
    print("✅ test_manager_flow 通过")


def test_dedup():
    mgr = MemoryManager(long_term_store=InMemoryStore(), llm=FakeLLM())
    mgr.remember("用户喜欢 Python", kind="preference")
    assert mgr._is_duplicate("用户喜欢 Python") is True
    assert mgr._is_duplicate("用户喜欢 Rust") is False
    print("✅ test_dedup 通过")


def test_memory_tool():
    mgr = MemoryManager(long_term_store=InMemoryStore(), llm=FakeLLM())
    remember_tool = RememberTool(mgr)
    recall_tool = RecallTool(mgr)
    forget_tool = ForgetTool(mgr)

    out = remember_tool.invoke(content="用户住在上海", kind="fact", importance=0.7).output
    assert "已记住" in out, out

    out = recall_tool.invoke(query="用户住在哪").output
    assert "上海" in out, out

    record = mgr.long_term.all()[0]
    out = forget_tool.invoke(memory_id=record.memory_id[:8]).output
    assert "已撤回" in out, out
    assert len(mgr.long_term.all()) == 0
    print("✅ test_memory_tool 通过")


def test_tool_schema():
    """确认记忆工具能正确生成 OpenAI function calling schema"""
    mgr = MemoryManager(long_term_store=InMemoryStore(), llm=FakeLLM())
    tool = RememberTool(mgr)
    schema = tool.to_openai_tool()
    assert schema["function"]["name"] == "remember_memory"
    props = schema["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"content", "kind", "importance"}, props
    print("✅ test_tool_schema 通过")
    print("   schema =", schema["function"]["parameters"])


def test_scoring():
    assert lexical_score("Python", "用户喜欢 Python") == 1.0
    assert lexical_score("Python", "用户喜欢 Java") == 0.0
    print("✅ test_scoring 通过")


if __name__ == "__main__":
    test_buffer_sliding_window()
    test_buffer_summary()
    test_sqlite_crud()
    test_preference_always_recalled()
    test_revise_keeps_history()
    test_scope_isolation()
    test_manager_flow()
    test_dedup()
    test_memory_tool()
    test_tool_schema()
    test_scoring()
    print("\n🎉 全部测试通过")
