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
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.memory import (
    BufferMemory,
    InMemoryProfileStore,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    SemanticMemoryStore,
    SQLiteMemoryStore,
    SQLiteProfileStore,
    UserProfile,
    lexical_score,
)
from fast_agents.tools.builtin import (
    RememberTool, RecallTool, ForgetTool, UpdateProfileTool,
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


# ===================== B2 / v1.4.0：SemanticMemoryStore =====================

class _SemanticFakeLLM:
    """按「词项类别」做 one-hot-ish 向量的假 LLM，离线可跑

    位置 0: 地点类词（上海/北京/住/所在/位置/在哪/地）
    位置 1: 工作类词（工作/上班/公司）
    位置 2: 爱好类词（吃/辣/口味/喜欢/爱）
    """
    embedding_model = "fake-semantic"

    def _vec(self, text: str):
        v = [0.0, 0.0, 0.0]
        if any(w in text for w in ("上海", "北京", "住", "所在", "位置", "在哪", "地")):
            v[0] = 1.0
        if any(w in text for w in ("工作", "上班", "公司")):
            v[1] = 1.0
        if any(w in text for w in ("吃", "辣", "口味", "喜欢", "爱")):
            v[2] = 1.0
        return v

    def embed_query(self, q: str):
        return self._vec(q)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


def test_semantic_synonym_recall():
    """B2 核心验收：同义改写 query 能召回（纯词法召不回的场景）

    「用户所在地」词法上不含「上海」「工作」，纯 InMemory 召不回事实记录；
    但 SemanticMemoryStore 走 embedding 后，「所在/地」和「上海/住」语义同簇，应当召回。
    """
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")

        # 对照：纯词法后端召不回
        plain = SQLiteMemoryStore(db, user_id="u1")
        plain.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                               source="t", kind="fact", importance=0.7))
        plain.add(MemoryRecord(user_id="u1", session_id="s", content="我爱吃辣",
                               source="t", kind="preference", importance=0.8))
        lex_only = plain.search("用户所在地", limit=5)
        # 词法零命中的 fact 被过滤掉，只留下保底的 preference
        assert not any("上海" in r.content for r in lex_only), \
            "前置条件不成立：纯词法竟然召回了上海（测试用例失效）"

        # 语义后端：同一条记录能被同义改写召回
        store = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _SemanticFakeLLM())
        res = store.search("用户所在地", limit=5)
        assert any("上海" in r.content for r in res), \
            "B2 验收失败：同义改写 query 未召回对应事实"

        # InMemory 包装器也同样成立
        inner_mem = InMemoryStore()
        store_mem = SemanticMemoryStore(inner_mem, _SemanticFakeLLM())
        store_mem.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                   source="t", kind="fact", importance=0.7))
        store_mem.add(MemoryRecord(user_id="u1", session_id="s", content="我爱吃辣",
                                   source="t", kind="preference", importance=0.8))
        res_mem = store_mem.search("用户所在地", limit=5)
        assert any("上海" in r.content for r in res_mem)

    print("✅ test_semantic_synonym_recall 通过（B2 同义改写召回）")


def test_semantic_graceful_degrade():
    """B2：embedding 调用失败时必须降级，不抛异常、记忆照常写入/检索"""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")

        # 场景 A：embed_query 失败 → 整次 search 退化为纯词法
        class _BombQuery(_SemanticFakeLLM):
            def embed_query(self, q):
                raise RuntimeError("模拟网络故障")
        s_a = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _BombQuery())
        s_a.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                             source="t", kind="fact", importance=0.7))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_a = s_a.search("上海", limit=5)  # 词法「上海」命中，应能召回
        assert any("上海" in r.content for r in res_a), "embed_query 失败后纯词法兜底失效"

        # 场景 B：embed_documents 失败 → 写入照常、该条记忆无向量但仍可被词法召回
        class _BombDoc(_SemanticFakeLLM):
            def embed_documents(self, texts):
                raise RuntimeError("模拟 API 限流")
        s_b = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u2"), _BombDoc())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = s_b.add(MemoryRecord(user_id="u2", session_id="s", content="我爱吃辣",
                                       source="t", kind="preference", importance=0.8))
        assert rec.memory_id  # 写入成功返回了 id
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_b = s_b.search("口味", limit=5)
        # 偏好类始终保底，所以至少返回偏好
        assert any("辣" in r.content for r in res_b)

        # 场景 C：空 query 直接走原后端，不调用 embedding
        s_c = SemanticMemoryStore(InMemoryStore(), _SemanticFakeLLM())
        s_c.add(MemoryRecord(user_id="u1", session_id="s", content="一条记忆",
                             source="t", kind="fact", importance=0.5))
        res_c = s_c.search("", limit=3)
        assert len(res_c) == 1

    print("✅ test_semantic_graceful_degrade 通过（B2 降级不炸）")


def test_semantic_revise_and_crud():
    """B2：revise 后新内容的向量跟随生效，delete/clear 同步清理向量"""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        inner = SQLiteMemoryStore(db, user_id="u1")
        store = SemanticMemoryStore(inner, _SemanticFakeLLM())

        old = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                     source="t", kind="fact", importance=0.7))
        store.add(MemoryRecord(user_id="u1", session_id="s", content="我爱吃辣",
                               source="t", kind="preference", importance=0.8))

        # revise：把「上海」改为「北京」，新内容应被「现在在哪工作」召回并排第一
        new = store.revise(old.memory_id, "我在北京工作", source="revise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = store.search("现在在哪工作", limit=5)
        assert res[0].content == "我在北京工作", "revise 后新内容未排第一"

        # delete 后对应的向量也清掉（不抛异常、不影响其他记录）
        assert store.delete(new.memory_id) is True
        # 打开一个包装同一 inner 的新实例，走懒加载路径——应当不会再看到新记录
        store2 = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _SemanticFakeLLM())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res2 = store2.search("北京", limit=5)
        assert not any("北京" in r.content for r in res2), "delete 后旧记录仍被召回"

        # clear 清空当前作用域
        store2.clear()
        assert store2.all() == []

    print("✅ test_semantic_revise_and_crud 通过（B2 revise/delete/clear 正常）")


# ===================== B3 / v1.5.0：结构化用户画像 =====================

def test_profile_merge_and_override():
    """B3：画像字段级覆盖式合并；dict 浅合并；None 不覆盖；非法字段报错"""
    store = InMemoryProfileStore()
    assert store.get("u1").is_empty(), "新用户应为空档案"

    p = store.merge("u1", {"name": "张三", "role": "后端工程师"})
    assert p.name == "张三" and p.role == "后端工程师"

    p = store.merge("u1", {"name": "张三丰"})
    assert p.name == "张三丰", "同名字段应被后写覆盖"
    assert p.role == "后端工程师", "patch 里没提到的字段应保留"

    p = store.set_field("u1", "preferences.编辑器", "vscode")
    assert p.stable_preferences == {"编辑器": "vscode"}

    p = store.set_field("u1", "preferences.语言", "中文")
    assert p.stable_preferences == {"编辑器": "vscode", "语言": "中文"}, "键值字段应浅合并"

    p = store.set_field("u1", "facts.公司", "某厂")
    assert p.stable_facts == {"公司": "某厂"}

    p = store.merge("u1", {"name": None, "location": None})
    assert p.name == "张三丰", "None 不应覆盖已有值"

    # 未知顶层键被静默忽略，不会往模型上乱塞属性
    p = store.merge("u1", {"not_a_field": "x"})
    assert not hasattr(p, "not_a_field")

    # 非法字段名应显式报错，而不是写脏键
    try:
        store.set_field("u1", "bogus", "x")
        raise AssertionError("非法字段应抛 ValueError")
    except ValueError:
        pass

    assert store.get("u1").to_text(), "to_text() 应能渲染出非空文本"
    assert store.get("u1").to_dict()["name"] == "张三丰"
    print("✅ test_profile_merge_and_override 通过（B3 覆盖式合并）")


def test_profile_sqlite_persistence():
    """B3：画像落盘 SQLite，换实例仍能读回；不同 user_id 互相隔离"""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "memory.db")

        s1 = SQLiteProfileStore(db)
        s1.merge("alice", {"name": "Alice", "location": "上海"})
        s1.set_field("alice", "preferences.编辑器", "vim")
        s1.close()

        s2 = SQLiteProfileStore(db)
        p = s2.get("alice")
        assert p.name == "Alice" and p.location == "上海"
        assert p.stable_preferences == {"编辑器": "vim"}
        assert s2.get("bob").is_empty(), "画像应按 user_id 隔离"
        s2.close()

    print("✅ test_profile_sqlite_persistence 通过（B3 持久化 + 隔离）")


def test_profile_manager_integration():
    """B3：MemoryManager 整合——读写正常、画像不污染长期记忆、未配置时降级"""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "memory.db")
        mgr = MemoryManager(
            long_term_store=SQLiteMemoryStore(db, user_id="u1"),
            profile_store=SQLiteProfileStore(db),
            user_id="u1",
        )
        assert mgr.get_profile().is_empty()

        mgr.set_profile_field("location", "上海")
        mgr.set_profile_field("preferences.语言", "中文")
        p = mgr.get_profile()
        assert p.location == "上海"
        assert p.stable_preferences == {"语言": "中文"}

        mgr.update_profile({"role": "后端工程师"})
        assert mgr.get_profile().role == "后端工程师", "update_profile 应按 patch 覆盖式合并"

        # 画像是独立的一份档案，不进长期记忆检索链路
        assert mgr.long_term.all() == [], "画像不应写进长期记忆表"
        mgr.close()

    # 未配置 profile_store：读返回空档案，写明确报错
    mgr2 = MemoryManager(user_id="u2")
    assert mgr2.get_profile().is_empty(), "未配置时应返回空档案而不是抛错"
    try:
        mgr2.update_profile({"name": "x"})
        raise AssertionError("未配置 profile_store 时写入应抛 RuntimeError")
    except RuntimeError as e:
        assert "profile_store" in str(e)

    print("✅ test_profile_manager_integration 通过（B3 Manager 整合 + 降级）")


def test_profile_tool():
    """B3：update_user_profile 工具；默认不装配，include_profile=True 才加入"""
    from fast_agents.tools.builtin import build_memory_tools

    mgr = MemoryManager(profile_store=InMemoryProfileStore(), user_id="u1")

    assert len(build_memory_tools(mgr)) == 3, "默认仍返回三个工具（向后兼容）"
    tools = build_memory_tools(mgr, include_profile=True)
    assert len(tools) == 4
    tool = tools[-1]
    assert tool.name == "update_user_profile"
    assert tool.to_openai_tool()["function"]["name"] == "update_user_profile"

    res = tool.invoke(field="name", value="张三")
    assert not res.failed, res
    assert mgr.get_profile().name == "张三"

    res = tool.invoke(field="preferences.编辑器", value="vscode")
    assert not res.failed, res
    assert mgr.get_profile().stable_preferences == {"编辑器": "vscode"}

    # 空 field / 非法字段 → 失败结果，不抛异常
    assert tool.invoke(field="", value="x").failed
    assert tool.invoke(field="bogus", value="x").failed

    # 未配置 profile_store 的 manager：工具返回失败，而不是崩掉
    mgr2 = MemoryManager(user_id="u2")
    r2 = UpdateProfileTool(mgr2).invoke(field="name", value="x")
    assert r2.failed and "profile_store" in r2.error

    print("✅ test_profile_tool 通过（B3 画像工具）")


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
    test_semantic_synonym_recall()
    test_semantic_graceful_degrade()
    test_semantic_revise_and_crud()
    test_profile_merge_and_override()
    test_profile_sqlite_persistence()
    test_profile_manager_integration()
    test_profile_tool()
    print("\n🎉 全部测试通过")
