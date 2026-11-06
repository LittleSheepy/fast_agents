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
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.memory import (
    BufferMemory,
    ChromaMemoryStore,
    InMemoryProfileStore,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    SemanticMemoryStore,
    SQLiteMemoryStore,
    SQLiteProfileStore,
    SQLiteSessionStore,
    UserProfile,
    lexical_score,
    merge_similarity,
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


# ===================== B5 / v1.7.0：记忆衰减与合并 =====================

def test_touch_records_access():
    """B5：命中记账 —— access_count 累加、last_accessed_at 刷新，且不动 updated_at"""
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        r = store.add(MemoryRecord(content="用户喜欢 Python", kind="fact", source="test"))
        first = store.get(r.memory_id)
        assert first.access_count == 0 and first.last_accessed_at is None
        updated_at_before = first.updated_at

        assert store.touch([r.memory_id]) == 1
        after = store.get(r.memory_id)
        assert after.access_count == 1
        assert after.last_accessed_at is not None, "last_accessed_at 应被刷新"
        # 关键：touch 不能刷新 updated_at，否则时间衰减失效
        assert after.updated_at == updated_at_before

        # 再命中一次 → 累加；不存在的 id 不报错
        assert store.touch([r.memory_id, "not-exist"]) == 1
        assert store.get(r.memory_id).access_count == 2
        store.close()

        # 内存后端走基类默认实现，同样有效
        mem = InMemoryStore()
        m = mem.add(MemoryRecord(content="用户在北京", kind="fact", source="t"))
        assert mem.touch([m.memory_id]) == 1
        assert mem.get(m.memory_id).access_count == 1
        assert mem.get(m.memory_id).last_accessed_at is not None
    print("✅ test_touch_records_access 通过（B5 命中记账）")


def test_archive_stale():
    """B5：低重要度 + 久未命中 → 归档；不再进检索但可审计"""
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        # 低重要度 + 200 天没动过 → 应归档
        stale = store.add(MemoryRecord(
            content="随口提过的咖啡馆", kind="episode", source="t", importance=0.1,
            updated_at=datetime.now(timezone.utc) - timedelta(days=200),
        ))
        # 低重要度但很新 → 不归档
        fresh = store.add(MemoryRecord(
            content="刚说过的偏好", kind="preference", source="t", importance=0.1,
        ))
        # 重要度高 + 很旧 → 不归档
        important = store.add(MemoryRecord(
            content="用户的职业是医生", kind="fact", source="t", importance=0.9,
            updated_at=datetime.now(timezone.utc) - timedelta(days=400),
        ))

        archived = store.archive_stale(max_importance=0.3, stale_days=90)
        assert [r.memory_id for r in archived] == [stale.memory_id], [r.content for r in archived]
        assert store.get(stale.memory_id).status == "archived"
        assert store.get(stale.memory_id).metadata.get("archived_at")
        assert store.get(fresh.memory_id).status == "active"
        assert store.get(important.memory_id).status == "active"
        # 归档后不参与检索（all 默认只返回 active），但审计仍可见
        assert len(store.all()) == 2
        assert len(store.all(include_inactive=True)) == 3
        store.close()
    print("✅ test_archive_stale 通过（B5 衰减归档）")


def test_merge_similar_semantic():
    """B5：合并相似度 = 语义×0.7 + 词法×0.3；无向量后端退化为纯词法（不当 0）"""
    # 纯函数口径
    assert merge_similarity(0.0, 1.0) == 0.7
    assert merge_similarity(1.0, 0.0) == 0.3
    assert merge_similarity(0.5, None) == 0.5   # ← 关键：退化，而不是压成 0.15

    # 有语义后端：同簇但字面不同 → 能合并
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        store = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _SemanticFakeLLM())
        keep = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                      source="t", kind="fact", importance=0.9))
        dup = store.add(MemoryRecord(user_id="u1", session_id="s", content="公司在北京上班",
                                     source="t", kind="fact", importance=0.4))
        merged = store.merge_similar(threshold=0.7)
        assert [r.memory_id for r in merged] == [dup.memory_id], [r.content for r in merged]
        assert store.get(dup.memory_id).status == "superseded"
        assert store.get(dup.memory_id).metadata["merged_into"] == keep.memory_id
        # 主记录保持原样（不改内容、不改状态）
        assert store.get(keep.memory_id).status == "active"
        assert store.get(keep.memory_id).content == "我在上海工作"

    # 无向量后端：退化口径正确 → 纯词法 Jaccard≈0.11
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        plain = SQLiteMemoryStore(db, user_id="u1")
        plain.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                               source="t", kind="fact", importance=0.9))
        plain.add(MemoryRecord(user_id="u1", session_id="s", content="公司在北京上班",
                               source="t", kind="fact", importance=0.4))
        # 阈值拉到语义级别：不该合并
        assert plain.merge_similar(threshold=0.5) == []
        # 退化口径正确时能过 0.05；若错误地把缺失语义当成 0（0.3×0.11≈0.03），就过不了
        assert len(plain.merge_similar(threshold=0.05)) == 1
        plain.close()
    print("✅ test_merge_similar_semantic 通过（B5 合并相似度）")


def test_manager_compact_and_retrieve_touch():
    """B5：retrieve 自动记账；compact() 一次做完归档 + 合并，后端异常只警告"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        mgr = MemoryManager(
            long_term_store=SQLiteMemoryStore(path, user_id="u1", session_id="s1"),
            user_id="u1", session_id="s1",
        )
        hit = mgr.remember("用户喜欢 Python", kind="fact", importance=0.9)
        mgr.retrieve("Python", limit=5)
        assert mgr.long_term.get(hit.memory_id).access_count == 1, "retrieve 应自动记账"

        stale = mgr.long_term.add(MemoryRecord(
            user_id="u1", session_id="s1", content="随口提过的咖啡馆",
            kind="episode", source="t", importance=0.1,
            updated_at=datetime.now(timezone.utc) - timedelta(days=200),
        ))
        report = mgr.compact(max_importance=0.3, stale_days=90)
        assert report["archived"] == 1, report
        assert mgr.long_term.get(stale.memory_id).status == "archived"
        assert mgr.long_term.get(hit.memory_id).status == "active"

        # 后端抛异常时只警告、不抛，返回 0
        class _Boom(InMemoryStore):
            def archive_stale(self, **kw):
                raise RuntimeError("boom")

            def merge_similar(self, **kw):
                raise RuntimeError("boom")

        assert MemoryManager(long_term_store=_Boom()).compact() == {"archived": 0, "merged": 0, "expired": 0}
        mgr.close()
    print("✅ test_manager_compact_and_retrieve_touch 通过（B5 维护入口）")


# ===================== ChromaMemoryStore 阶段 1 测试 =====================
# 阶段 1：不引 chromadb、不传真实客户端，验证 ①纯词法降级②代理记账③Manager 集成。

class _FakeLLMNoEmbedding:
    """假 LLM：embedding 永远失败，逼 ChromaMemoryStore 走纯词法降级"""
    embedding_model = "fake-none"

    def embed_query(self, q):           return None
    def embed_documents(self, texts):   return None


def test_chroma_store_pure_lexical_degrade():
    """chroma_client=None + 假 LLM（embedding 失败）→ 仍可正常 CRUD + 检索，不抛异常"""
    with tempfile.TemporaryDirectory() as d:
        inner = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1")
        store = ChromaMemoryStore(inner, _FakeLLMNoEmbedding())  # chroma_client=None

        r1 = store.add(MemoryRecord(content="Python 是解释型语言", kind="fact", importance=0.7))
        r2 = store.add(MemoryRecord(content="Java 是编译型语言", kind="fact", importance=0.6))
        assert store.get(r1.memory_id) is not None
        assert store.get(r2.memory_id) is not None

        # search 纯词法可召回（含 query 关键字的那条）
        hits = store.search("Python", limit=2)
        assert any(h.memory_id == r1.memory_id for h in hits), "纯词法降级应能召回含关键词的记忆"

        # delete/clear 不抛异常
        assert store.delete(r1.memory_id) is True
        assert store.get(r1.memory_id) is None
        store.clear()
        assert store.all() == []
        store.close()
        inner.close()
    print("✅ test_chroma_store_pure_lexical_degrade 通过（Chroma 纯词法降级 CRUD/search）")


def test_chroma_store_b5_proxy_methods():
    """B5 的 touch / revise / retract / archive_stale / merge_similar 都能正确代理 inner"""
    with tempfile.TemporaryDirectory() as d:
        inner = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1")
        store = ChromaMemoryStore(inner, _FakeLLMNoEmbedding())
        r = store.add(MemoryRecord(content="我喜欢吃苹果", kind="preference", importance=0.9))

        # touch 能记账（不抛、不刷 updated_at）
        assert store.touch([r.memory_id]) == 1
        assert store.get(r.memory_id).access_count == 1

        # revise 能修订（content 改了、向量重算降级不抛）
        revised = store.revise(r.memory_id, "我喜欢吃香蕉", source="test")
        assert revised is not None and revised.content == "我喜欢吃香蕉"

        # archive_stale 正常代理（新写入的记忆不会被归档）
        archived = store.archive_stale(stale_days=90, max_importance=0.3)
        assert isinstance(archived, list) and len(archived) == 0

        # merge_similar 正常代理（批量向量拿不到就降级纯词法，不抛）
        merged = store.merge_similar(threshold=0.99)
        assert isinstance(merged, list)

        # retract 能撤回（同步不抛）——撤回后 active 列表里不应再出现该条
        retracted = store.retract(r.memory_id)
        assert retracted is not None
        assert retracted.status in ("archived", "superseded", "retracted")
        active_ids = {x.memory_id for x in store.all(include_inactive=False)}
        assert r.memory_id not in active_ids
        store.close()
        inner.close()
    print("✅ test_chroma_store_b5_proxy_methods 通过（B5 方法代理 + 降级不抛）")


def test_chroma_store_with_manager_integration():
    """ChromaMemoryStore 能直接插进 MemoryManager，retrieve/record/render_context 不抛"""
    with tempfile.TemporaryDirectory() as d:
        inner = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1")
        store = ChromaMemoryStore(inner, _FakeLLMNoEmbedding())
        mgr = MemoryManager(llm=_FakeLLMNoEmbedding(), long_term_store=store)
        # record 时 auto_extract 会调 embed_documents → None 降级，不抛
        mgr.record("我叫小明", "你好小明")
        hits = mgr.retrieve("你叫什么")
        assert isinstance(hits, list)
        rendered = mgr.render_context("你叫什么")
        assert isinstance(rendered, str)
        mgr.close()
        store.close()
        inner.close()
    print("✅ test_chroma_store_with_manager_integration 通过（Manager 接入零感知）")


def test_sqlite_upsert_full_column_overwrite():
    """J3 陷阱1：add 走全列 upsert，语义与原 INSERT OR REPLACE 一致

    要点：同 memory_id 再次 add 时是「整行覆盖」——业务列全部按传入记录刷新，
    不会残留旧行里没提供的字段（这正是 OR REPLACE「先删后插」的行为）。
    """
    with tempfile.TemporaryDirectory() as d:
        store = SQLiteMemoryStore(os.path.join(d, "mem.db"), user_id="u1", session_id="s1")
        r = store.add(MemoryRecord(content="原始内容", kind="fact", source="t"))
        mid = r.memory_id
        store.touch([mid])                              # access_count -> 1
        assert store.get(mid).access_count == 1

        # 同 id 再 add：整行覆盖（access_count 回到传入记录的 0）
        r2 = MemoryRecord(memory_id=mid, user_id="u1", session_id="s1",
                          content="覆盖后内容", kind="fact", source="t")
        store.add(r2)
        got = store.get(mid)
        assert got.content == "覆盖后内容"
        assert got.access_count == 0
        assert got.created_at == r2.created_at
        # 只保留一行，而不是又插了一条
        assert len(store.all(include_inactive=True)) == 1
        store.close()
    print("✅ test_sqlite_upsert_full_column_overwrite 通过（陷阱1：全列覆盖等价 OR REPLACE）")


def test_touch_scope_isolation_in():
    """J3 陷阱2：touch 的 IN 条件必须叠加作用域谓词，杜绝跨用户记账"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        a = SQLiteMemoryStore(path, user_id="uA", session_id="s1")
        b = SQLiteMemoryStore(path, user_id="uB", session_id="s1")
        ra = a.add(MemoryRecord(content="A的记忆", kind="fact", source="t"))
        rb = b.add(MemoryRecord(content="B的记忆", kind="fact", source="t"))

        # 故意把 B 的 id 混进 A 的 touch 列表：只应命中 A 自己那条
        n = a.touch([ra.memory_id, rb.memory_id])
        assert n == 1, f"应命中 1 条，实际 {n}"
        assert a.get(ra.memory_id).access_count == 1
        assert b.get(rb.memory_id).access_count == 0, "跨用户记账泄漏"
        # 空 id 列表直接返回 0，不发 SQL
        assert a.touch([]) == 0
        a.close()
        b.close()
    print("✅ test_touch_scope_isolation_in 通过（陷阱2：IN + scope 隔离）")


def test_session_store_persistence():
    """J3：SQLiteSessionStore 迁移到 SQLiteClient 后的持久化 / 隔离 / 摘要 upsert"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "session.db")
        ss = SQLiteSessionStore(path, user_id="u1", session_id="s1")
        ss.append(Message(role="user", content="你好"))
        ss.append(Message(role="assistant", content="你好呀"))
        assert [m.content for m in ss.load()] == ["你好", "你好呀"]

        ss.save_summary("第一版摘要")
        assert ss.load_summary() == "第一版摘要"
        # 第二次保存走 upsert 覆盖（若是普通 INSERT 会因复合主键冲突报错）
        ss.save_summary("第二版摘要")
        assert ss.load_summary() == "第二版摘要"
        meta_rows = ss._db.select("session_meta", {"user_id": "u1", "session_id": "s1"})
        assert len(meta_rows) == 1, f"摘要表应只有一行，实际 {len(meta_rows)}"

        # 作用域隔离：同库不同 session 互不可见
        other = SQLiteSessionStore(path, user_id="u1", session_id="s2")
        assert other.load() == [] and other.load_summary() == ""

        ss.clear()
        assert ss.load() == [] and ss.load_summary() == ""
        ss.close()
        other.close()
    print("✅ test_session_store_persistence 通过（SQLiteSessionStore 迁移后行为正常）")


def test_b8_time_is_cst():
    """B8：记忆时间统一为北京时间 CST (+08:00)——默认带时区、落库带 +08:00、naive 输入按 CST 解读以兼容历史数据"""
    import sqlite3
    from datetime import timedelta, timezone
    from fast_agents.utils.time_utils import CST

    # 1. 新建记录默认时间即 aware CST
    r = MemoryRecord(content="时间基准")
    assert r.created_at.tzinfo is not None, r.created_at
    assert r.created_at.utcoffset() == timedelta(hours=8), r.created_at
    assert r.updated_at.tzinfo is not None
    assert r.updated_at.utcoffset() == timedelta(hours=8), r.updated_at

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        store = SQLiteMemoryStore(path, user_id="u1")
        rec = store.add(MemoryRecord(content="落库时间", kind="fact"))
        got = store.get(rec.memory_id)
        # 2. 往返后仍是 aware CST（不随机器时区漂移）
        assert got.created_at.utcoffset() == timedelta(hours=8), got.created_at
        assert got.updated_at.utcoffset() == timedelta(hours=8), got.updated_at
        # 3. 落库字符串带 +08:00（不再是裸本地串）
        con = sqlite3.connect(path)
        raw = con.execute(
            "SELECT created_at, updated_at FROM memories WHERE memory_id = ?",
            (rec.memory_id,),
        ).fetchone()
        con.close()
        assert "+08:00" in raw[0] and "+08:00" in raw[1], raw

        # 4. naive datetime 按 CST 解读（历史本地时间数据零偏移兼容），而不是抛错
        from fast_agents.utils.time_utils import to_cst
        naive = datetime(2026, 4, 18, 16, 0, 0)
        normalized = to_cst(naive)
        assert normalized.tzinfo is not None
        assert normalized.utcoffset() == timedelta(hours=8)
        assert normalized.year == 2026 and normalized.month == 4 and normalized.day == 18 and normalized.hour == 16

        # 5. aware 跨时区转换不改变绝对时刻（UTC 输入 -> CST 输出，+8 小时）
        utc_t = datetime(2026, 4, 18, 8, 0, 0, tzinfo=timezone.utc)
        cst_t = to_cst(utc_t)
        assert cst_t.utcoffset() == timedelta(hours=8)
        assert cst_t.hour == 16 and cst_t.minute == 0
        store.close()
    print("✅ test_b8_time_is_cst 通过（B8 时间统一北京时间 CST：默认 / 往返 / +08:00 / naive 兼容）")


def test_b10_auto_extract_watermark():
    """B10：auto_extract 增量水位线——只抽新增消息，不重复抽历史，省token、避免重复入库"""
    call_count = 0
    last_conversation_len = 0

    class CountingFakeLLM:
        def invoke(self, messages, temperature=0.0):
            nonlocal call_count, last_conversation_len
            call_count += 1
            prompt = messages[0]["content"]
            # 统计本次抽取传入的对话行数（即新增消息条数）
            conv_part = prompt.split("对话内容：\n")[-1]
            last_conversation_len = len([l for l in conv_part.split("\n") if l.strip()])
            # 第一次抽取返回用户偏好，第二次返回新事实
            if call_count == 1:
                class R:
                    content = '[{"content":"用户喜欢用Python编程","kind":"preference","importance":0.8}]'
                return R()
            elif call_count == 2:
                class R:
                    content = '[{"content":"用户项目使用fast_agents框架","kind":"fact","importance":0.7}]'
                return R()
            else:
                class R:
                    content = '[]'
                return R()

    llm = CountingFakeLLM()
    manager = MemoryManager(
        llm=llm,
        auto_extract=True,
        long_term_store=InMemoryStore(),
    )
    # 第一轮对话：第一次record触发第一次抽取，只处理新增的2条消息
    manager.record("你好", "你好呀，有什么可以帮你的？")
    assert call_count == 1, f"第一轮第一条后应调用1次抽取，实际{call_count}次"
    assert last_conversation_len == 2, f"第一次抽取应只传2条新消息，实际{last_conversation_len}条"
    manager.record("我喜欢用Python编程", "好的，记住你喜欢Python啦。")
    assert call_count == 2, f"第一轮第二条后应共调用2次抽取，实际{call_count}次"
    # 第二次抽取只新增2条消息，不会重复处理前2条历史（验证增量特性）
    assert last_conversation_len == 2, f"第二次抽取应只传新增的2条消息，实际{last_conversation_len}条"
    all_mems = manager.long_term.all()
    assert len(all_mems) == 2
    assert any("Python" in m.content for m in all_mems)

    # 第三轮对话：再新增2条消息，只抽这2条
    manager.record("对了我项目用fast_agents框架", "好的，记住你的项目用fast_agents。")
    assert call_count == 3, f"第三轮后应共调用3次抽取，实际{call_count}次"
    assert last_conversation_len == 2, f"第三次抽取应只传新增的2条消息，实际{last_conversation_len}条"
    all_mems = manager.long_term.all()
    assert len(all_mems) == 2
    assert any("fast_agents" in m.content for m in all_mems)

    # 再调用一次extract_and_store()，没有新消息，不会触发LLM调用
    manager.extract_and_store()
    assert call_count == 3, f"无新消息时不应重复抽取，调用次数仍为3，实际{call_count}次"

    # 清空会话后水位线重置
    manager.forget_session()
    manager.record("新会话第一条", "新会话回复")
    # 新会话第一次record会触发抽取，因为水位线归零了
    assert call_count == 4
    print("✅ test_b10_auto_extract_watermark 通过（B10 增量抽取水位线：不重复抽历史、无新消息不调用、重置会话清零）")


def test_b11_manager_revise_forward():
    """B11：MemoryManager.revise() 转发到底层Store，旧版本留痕、新版本可检索，空内容校验通过"""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "b11_test.db")
    # 统一作用域：store和manager都用default用户，避免作用域过滤导致查不到
    store = SQLiteMemoryStore(db_path, user_id="default", session_id="default")
    manager = MemoryManager(long_term_store=store, user_id="default", session_id="default")

    # 1. 写入一条初始记忆
    old_record = manager.remember("用户住在上海", kind="fact", importance=0.8)
    old_id = old_record.memory_id
    # 初始检索召回旧内容
    results = manager.retrieve("用户住在哪", limit=5)
    assert any(r.content == "用户住在上海" for r in results), "初始写入后应能检索到旧内容"
    assert len(results) >= 1

    # 2. 修订这条记忆为新内容
    new_record = manager.revise(old_id, "用户住在北京", source="user_updated")
    assert new_record.content == "用户住在北京"
    assert new_record.supersedes == old_id, f"新记忆应指向被替换的旧ID，实际{new_record.supersedes}"
    assert new_record.memory_id != old_id, "修订后应生成新ID，不应覆盖旧记忆"

    # 3. 检索现在只能召回新内容，旧内容被标记为superseded不参与检索
    results_after = manager.retrieve("用户住在哪", limit=5)
    active_contents = [r.content for r in results_after]
    assert "用户住在北京" in active_contents, "修订后应召回新内容"
    assert "用户住在上海" not in active_contents, "已被取代的旧内容不应出现在检索结果"

    # 4. 旧记忆仍保留在库中（审计可查），状态为superseded
    all_records = store.all(include_inactive=True)
    old_in_db = next((r for r in all_records if r.memory_id == old_id), None)
    assert old_in_db is not None, "旧记忆不应被删除，应保留供审计"
    assert old_in_db.status == "superseded", f"旧记忆状态应为superseded，实际{old_in_db.status}"

    # 5. 空内容校验
    try:
        manager.revise(old_id, "   ")
        assert False, "空内容应该抛ValueError"
    except ValueError:
        pass

    # 6. 不存在的ID抛KeyError
    try:
        manager.revise("non_exist_id", "随便写点什么")
        assert False, "不存在的ID应该抛KeyError"
    except KeyError:
        pass

    manager.close()
    import shutil
    shutil.rmtree(tmpdir)
    print("✅ test_b11_manager_revise_forward 通过（B11 Manager.revise()：转发底层、旧版本留痕、空内容/不存在ID校验正常）")


# ===================== B12 / v1.8.1：记忆 TTL 与到期清理 =====================

def test_b12_ttl_retrieve_filter():
    """B12① 检索路径过滤：expires_at 已过期的 active 记忆不应被 retrieve 返回，
    但 all(include_inactive=True) 仍可查到（软删/留痕，未到 compact 时间也拦得住）"""
    from fast_agents.utils.time_utils import cstnow
    mgr = MemoryManager(long_term_store=InMemoryStore(), user_id="u1", session_id="s1")

    # 永不过期的偏好一条（偏好类无条件召回，最能证明过滤后正常项仍能召回）
    forever = mgr.remember("永久偏好：用户偏好中文回答", kind="preference", importance=0.9)

    # 已过期的一条：直接构造 expires_at 在过去
    expired_rec = mgr.long_term.add(MemoryRecord(
        user_id="u1", session_id="s1", content="临时安排：明天下午3点开PR评审会",
        kind="episode", source="manual", importance=0.9,
        expires_at=cstnow() - timedelta(seconds=1),
    ))

    # retrieve 不该召回到期项；永久偏好正常召回
    hits = mgr.retrieve("明天下午3点", limit=5)
    assert not any(h.memory_id == expired_rec.memory_id for h in hits), \
        "已过期的记忆不应被检索召回"
    assert any(h.memory_id == forever.memory_id for h in hits), "永久偏好应正常召回"

    # 底层 all()（仅 active）仍能看到它（因为还没跑 compact 清理状态）——证明过滤发生在检索路径，
    # 而不是靠「提前改状态」
    all_active = mgr.long_term.all(include_inactive=False)
    assert any(r.memory_id == expired_rec.memory_id for r in all_active), \
        "compact 前该条仍是 active，all() 应可见（验证过滤发生在 search 路径）"

    # all(include_inactive=True) 当然也能看到——证明数据没被物理删除
    assert any(r.memory_id == expired_rec.memory_id for r in mgr.long_term.all(include_inactive=True))
    print("✅ test_b12_ttl_retrieve_filter 通过（B12 检索路径过滤：过期项不召回、不物理删）")


def test_b12_compact_purge_expired():
    """B12② compact 到期清理：expired 统计、状态转 retracted、metadata.expired_at 留痕、未到期项不变"""
    from fast_agents.utils.time_utils import cstnow
    mgr = MemoryManager(long_term_store=InMemoryStore(), user_id="u1", session_id="s1")

    # expires_at 在过去 → 应被清理
    gone = mgr.long_term.add(MemoryRecord(
        user_id="u1", session_id="s1", content="临时：本周三之前用方案X",
        kind="episode", source="manual", importance=0.7,
        expires_at=cstnow() - timedelta(hours=1),
    ))
    # expires_at 在未来 → 不动
    alive = mgr.long_term.add(MemoryRecord(
        user_id="u1", session_id="s1", content="长期：用户用 macOS 开发",
        kind="fact", source="manual", importance=0.8,
        expires_at=cstnow() + timedelta(days=30),
    ))
    # 永不过期 → 不动
    forever = mgr.remember("永久：用户偏好中文回答", kind="preference", importance=0.9)

    report = mgr.compact()
    assert report["expired"] == 1, report
    assert report["archived"] == 0 and report["merged"] == 0

    gone_reloaded = mgr.long_term.get(gone.memory_id)
    assert gone_reloaded.status == "retracted", f"到期应转 retracted，实际 {gone_reloaded.status}"
    assert "expired_at" in gone_reloaded.metadata, "metadata 应留 expired_at 时间戳"

    assert mgr.long_term.get(alive.memory_id).status == "active"
    assert mgr.long_term.get(forever.memory_id).status == "active"

    # 幂等：再跑一次 compact 不应重复计数
    report2 = mgr.compact()
    assert report2["expired"] == 0, "purge_expired 必须幂等，重复跑不应重复清理"
    print("✅ test_b12_compact_purge_expired 通过（B12 到期清理：retracted + 留痕 + 幂等）")


def test_b12_expires_at_sqlite_persistence():
    """B12③ SQLite 落库：expires_at 往返一致带 +08:00；老库自动补列；_update 后字段不丢；
    manager.remember(ttl=...) 能正确换算成 expires_at"""
    import sqlite3
    from fast_agents.utils.time_utils import cstnow

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        store = SQLiteMemoryStore(path, user_id="u1", session_id="s1")

        # 直接写一条 expires_at 在未来
        future_expire = cstnow() + timedelta(hours=24)
        rec = store.add(MemoryRecord(
            content="24小时后过期的临时事项", kind="episode", source="t",
            expires_at=future_expire,
        ))
        got = store.get(rec.memory_id)
        assert got.expires_at is not None
        assert got.expires_at.utcoffset() == timedelta(hours=8), "expires_at 应带 +08:00"
        # 比较到秒（isoformat 转回来可能丢微秒，不影响语义）
        assert abs((got.expires_at - future_expire).total_seconds()) < 2

        # 物理层字符串带 +08:00
        con = sqlite3.connect(path)
        raw_exp = con.execute(
            "SELECT expires_at FROM memories WHERE memory_id = ?",
            (rec.memory_id,),
        ).fetchone()[0]
        con.close()
        assert "+08:00" in raw_exp, raw_exp

        # update 后 expires_at 不丢（验证 update_where 里已正确回写该列）
        rec.importance = 0.95
        store.update(rec)
        assert store.get(rec.memory_id).expires_at is not None
        assert abs((store.get(rec.memory_id).expires_at - future_expire).total_seconds()) < 2

        # 通过 manager.remember(ttl=...) 写入：expires_at ≈ now + ttl
        mgr = MemoryManager(long_term_store=store, user_id="u1", session_id="s1")
        r = mgr.remember("ttl 写入：1小时内有效", kind="fact", importance=0.6,
                         ttl=timedelta(hours=1))
        loaded = store.get(r.memory_id)
        assert loaded.expires_at is not None
        delta = loaded.expires_at - cstnow()
        assert timedelta(minutes=55) <= delta <= timedelta(minutes=65), \
            f"ttl=1h 写入后 expires_at 应在 ~1h 后，实际差 {delta}"

        # 老库自动补列：用同一个 db 文件重新打开，没有「no such column: expires_at」
        store2 = SQLiteMemoryStore(path, user_id="u1", session_id="s1")
        # 永不过期的 None 不被误写成字符串 / 0
        no_ttl = store2.add(MemoryRecord(content="永不过期", kind="fact", source="t"))
        assert store2.get(no_ttl.memory_id).expires_at is None
        store2.close()
        store.close()
        mgr.close()
    print("✅ test_b12_expires_at_sqlite_persistence 通过（B12 SQLite 落库往返 / +08:00 / 迁移 / update 不丢 / ttl 换算）")


def test_b12_no_ttl_behavior_unchanged():
    """B12④ 向后兼容：不传 ttl 时 expires_at 为 None，检索与 compact 行为与改造前完全一致；
    同时验证 SemanticMemoryStore / ChromaMemoryStore 包装器也在 search 路径上过滤到期项。"""
    from fast_agents.utils.time_utils import cstnow

    # 1. 不传 ttl → expires_at=None，不影响检索
    mgr = MemoryManager(long_term_store=InMemoryStore())
    r = mgr.remember("用户喜欢中文回答", kind="preference", importance=0.9)
    assert mgr.long_term.get(r.memory_id).expires_at is None
    hits = mgr.retrieve("随便一个问题")
    assert any(h.memory_id == r.memory_id for h in hits)

    # 2. 无过期项时 compact 的 expired 为 0；archived/merged 口径不变
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.db")
        mgr2 = MemoryManager(long_term_store=SQLiteMemoryStore(path, user_id="u1"),
                             user_id="u1")
        mgr2.remember("一条普通永久事实", kind="fact", importance=0.6)
        rep = mgr2.compact()
        assert rep == {"archived": 0, "merged": 0, "expired": 0}, rep
        mgr2.close()

    # 3. 包装器（Semantic / Chroma）的 search 也必须在 rank 前过滤到期项
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        inner = SQLiteMemoryStore(db, user_id="u1")
        expired_id = inner.add(MemoryRecord(
            user_id="u1", session_id="s", content="临时事项：已过期的提醒",
            source="t", kind="fact", importance=0.7,
            expires_at=cstnow() - timedelta(seconds=1),
        )).memory_id
        inner.add(MemoryRecord(
            user_id="u1", session_id="s", content="长期事实：Python 解释执行",
            source="t", kind="fact", importance=0.8,
        ))

        sem = SemanticMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _SemanticFakeLLM())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sem_hits = sem.search("临时提醒", limit=5)
        assert not any(h.memory_id == expired_id for h in sem_hits), \
            "SemanticMemoryStore search 应过滤到期项"

        ch = ChromaMemoryStore(SQLiteMemoryStore(db, user_id="u1"), _FakeLLMNoEmbedding())
        ch_hits = ch.search("临时提醒", limit=5)
        assert not any(h.memory_id == expired_id for h in ch_hits), \
            "ChromaMemoryStore search 应过滤到期项"
        ch.close()
        sem.close()
        inner.close()


# ==================== 入口 ====================

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

    # AI新添加的测试(AI可以删除覆盖)
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
