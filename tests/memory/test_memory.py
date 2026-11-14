"""记忆模块测试：不依赖真实 LLM，用假 LLM 验证记忆闭环
fast_agents/tests/memory/test_memory.py
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
import re
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


# ===================== 共享假 LLM（Chroma 专项测试见 test_memory_chroma_store.py） =====================

class _FakeLLMNoEmbedding:
    """假 LLM：embedding 永远失败，逼包装器走纯词法降级（词法兜底 / Chroma 降级共用）"""
    embedding_model = "fake-none"

    def embed_query(self, q):           return None
    def embed_documents(self, texts):   return None


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


def test_b10_fix_extract_after_window_full():
    """B10 修复回归：滑动窗口写满并持续裁剪后，auto_extract 每轮仍持续抽取（不再静默停摆）"""
    call_count = 0
    passed_lens = []

    class CountingFakeLLM:
        def invoke(self, messages, temperature=0.0):
            nonlocal call_count
            call_count += 1
            prompt = messages[0]["content"]
            conv_part = prompt.split("对话内容：\n")[-1]
            passed_lens.append(len([l for l in conv_part.split("\n") if l.strip()]))

            class R:
                content = "[]"

            return R()

    manager = MemoryManager(
        llm=CountingFakeLLM(),
        auto_extract=True,
        max_messages=4,        # 极小窗口：2 轮就把窗口写满并开始裁剪
        enable_summary=False,  # 关闭摘要，排除干扰，只看抽取行为
        long_term_store=InMemoryStore(),
    )
    # 连续 6 轮对话：窗口会被多次裁剪；若水位线仍用窗口长度，第 3 轮起抽取会停摆
    for i in range(6):
        manager.record(f"第{i}轮用户输入", f"第{i}轮助手回复")
    assert call_count == 6, f"窗口写满后每轮都应抽取，期望 6 次，实际 {call_count} 次"
    assert passed_lens == [2] * 6, f"每轮都应只抽到新增的 2 条消息，实际 {passed_lens}"
    assert len(manager.buffer.get_messages()) == 4, "窗口应恒等于 max_messages"
    # 累计计数只增不减、不受裁剪影响
    assert manager.buffer.get_total_appended() == 12, (
        f"6 轮共 append 12 条，实际 {manager.buffer.get_total_appended()}"
    )
    # 无新消息时不重复抽取
    manager.extract_and_store()
    assert call_count == 6, f"无新消息不应重复抽取，实际 {call_count} 次"
    # 清空会话后计数与水位线一起归零，可重新开始抽取
    manager.forget_session()
    assert manager.buffer.get_total_appended() == 0
    manager.record("新会话第一条", "新会话回复")
    assert call_count == 7, f"新会话首轮应重新抽取，实际 {call_count} 次"
    print("✅ test_b10_fix_extract_after_window_full 通过（窗口写满裁剪后仍持续抽取、增量正确、计数重置）")


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


# ==================== v1.8.3 向量通道完善 ====================

class _CountingSemanticLLM(_SemanticFakeLLM):
    """带调用计数的假 LLM，用于验证缓存命中后不再重复 embed_documents"""

    def __init__(self):
        super().__init__()
        self.query_calls = 0
        self.doc_calls = 0
        self.last_doc_batch_sizes: list = []

    def embed_query(self, q: str):
        self.query_calls += 1
        return super().embed_query(q)

    def embed_documents(self, texts):
        self.doc_calls += 1
        self.last_doc_batch_sizes.append(len(texts))
        return super().embed_documents(texts)


def test_v183_semantic_cache_avoids_re_embed():
    """v1.8.3 ①：SemanticMemoryStore 首次 search 批量补算向量后，第二次 search
    应直接命中本地缓存，不再对同一条记录重复调 embed_documents。"""
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        llm = _CountingSemanticLLM()
        # 注意：store 的 session_id 必须和 add 进的记录一致（否则 _scope_clause 过滤不到）
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        store = SemanticMemoryStore(inner, llm)
        # add 时 embed_documents 会被调 1 次（批量单条）
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content="我在上海工作", source="t",
                               kind="fact", importance=0.7))
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content="我爱吃辣", source="t",
                               kind="preference", importance=0.8))
        add_doc_calls = llm.doc_calls

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits1 = store.search("用户所在地", limit=5)
        # 必须召回"我在上海工作"（语义通道可用）
        assert any("上海" in h.content for h in hits1), f"语义通道未召回上海：{[h.content for h in hits1]}"
        after_first_doc_calls = llm.doc_calls

        # 第二次 search 走缓存，不应再追加 embed_documents 调用
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits2 = store.search("用户口味偏好", limit=5)
        assert llm.doc_calls == after_first_doc_calls, (
            f"第二次 search 重复触发 embed_documents（缓存失效）："
            f"before={after_first_doc_calls}, after={llm.doc_calls}"
        )
        assert any("辣" in h.content for h in hits2), [h.content for h in hits2]

        # embedding_stats 应正确反映缓存条数
        stats = store.embedding_stats()
        assert stats["available"] is True
        assert stats["backend"] == "semantic_sqlite"
        assert stats["cached_total"] >= 2
        assert stats["dimension"] == 3
        assert stats["model"] == "fake-semantic"

        store.close()
        inner.close()
    print(f"✅ test_v183_semantic_cache_avoids_re_embed 通过（add 阶段 doc_calls={add_doc_calls}，二次 search 无重复 embed）")


def test_merge_similar_reuses_semantic_cache():
    """merge_similar 接上批量向量缓存：命中缓存的记录在合并时不再重复 embed_documents。

    修复前基类 merge_similar 直接 semantic_vectors([r.content ...]) 全量重算向量，
    每次合并不论缓存是否命中都会重新 embed 一遍（注释里如实写明的「每次全量重算」）。
    """
    import warnings
    # ① 全部命中缓存：合并过程零 embed_documents
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        llm = _CountingSemanticLLM()
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        store = SemanticMemoryStore(inner, llm)
        keep = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                      source="t", kind="fact", importance=0.9))
        dup = store.add(MemoryRecord(user_id="u1", session_id="s", content="公司在北京上班",
                                     source="t", kind="fact", importance=0.4))
        cached = llm.doc_calls
        assert cached >= 2, cached  # add 阶段两条都已 embed 并落盘
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged = store.merge_similar(threshold=0.7)
        assert llm.doc_calls == cached, (
            f"merge_similar 未命中缓存、重复 embed：before={cached}, after={llm.doc_calls}"
        )
        assert [r.memory_id for r in merged] == [dup.memory_id], [r.content for r in merged]
        assert store.get(dup.memory_id).status == "superseded"
        store.close()
        inner.close()

    # ② 仅部分缺失：只对缺失的那一条补算，而不是全量重算
    llm2 = _CountingSemanticLLM()
    store2 = SemanticMemoryStore(InMemoryStore(), llm2)
    store2.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                            source="t", kind="fact", importance=0.9))
    store2.add(MemoryRecord(user_id="u1", session_id="s", content="公司在北京上班",
                            source="t", kind="fact", importance=0.4))
    c = store2.add(MemoryRecord(user_id="u1", session_id="s", content="我爱吃辣",
                                source="t", kind="preference", importance=0.3))
    # 人为让 c 的向量「冷掉」（模拟老数据没向量 / 换 embedding 模型后的缓存缺失）
    store2._vec_cache.pop(c.memory_id, None)
    base_calls = llm2.doc_calls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store2.merge_similar(threshold=0.7)
    assert llm2.doc_calls - base_calls == 1, (
        f"应只补算缺失的 1 条向量，实际 embed {llm2.doc_calls - base_calls} 次"
    )
    store2.close()
    print("✅ test_merge_similar_reuses_semantic_cache 通过"
          "（合并命中本地向量缓存、零/最小重复 embed）")


def test_v183_embedding_stats_default_and_errors_export():
    """v1.8.3 ③：基类 embedding_stats 默认返回 available=False；
    errors.py 中 MemoryError / MemoryConflictError 应从 memory 包正确导出。"""
    from fast_agents.memory import MemoryError, MemoryConflictError
    assert issubclass(MemoryConflictError, MemoryError)

    # 裸 InMemoryStore（无向量能力）应返回 available=False
    plain = InMemoryStore()
    assert plain.embedding_stats() == {"available": False}

    # InMemoryStore 包 SemanticMemoryStore 后应有向量能力，backend=semantic_memory
    import warnings
    sem = SemanticMemoryStore(InMemoryStore(), _CountingSemanticLLM())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sem.add(MemoryRecord(content="测试条目", kind="fact", importance=0.5,
                             user_id="u", session_id="s"))
    stats = sem.embedding_stats()
    assert stats["available"] is True
    assert stats["backend"] == "semantic_memory"
    assert stats["cached_total"] == 1
    assert stats["total_records"] == 1
    sem.close()
    print("✅ test_v183_embedding_stats_default_and_errors_export 通过（基类默认统计 + errors 导出）")


# ===================== B13：向量冲突检测（以 Chroma 为主） =====================

def test_b13_semantic_find_similar_detects_conflict():
    """B13 ①：SemanticMemoryStore.find_similar 能通过向量相似度召回高重复内容，
    作为 Chroma 不可用时的兜底。"""
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        llm = _CountingSemanticLLM()
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        store = SemanticMemoryStore(inner, llm)
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content="我在上海工作", source="t",
                               kind="fact", importance=0.7))
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content="我爱吃辣", source="t",
                               kind="preference", importance=0.8))
        # 「本人居住于上海上班」应通过向量相似度命中「我在上海工作」
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits = store.find_similar("本人居住于上海上班", limit=3, threshold=0.7)
        assert hits, "find_similar 应召回到冲突记忆"
        top_rec, top_sim = hits[0]
        assert "上海" in top_rec.content, f"首位命中应为上海工作：{top_rec.content}"
        assert top_sim >= 0.7
        store.close()
        inner.close()
    print("✅ test_b13_semantic_find_similar_detects_conflict 通过（向量 find_similar 召回冲突）")


def test_b13_manager_remember_dedup_by_vector():
    """B13 ②：MemoryManager.remember 在向量检测到高相似已有记忆时直接返回原记忆，
    不新增重复条目（newest_wins：原记忆保留，不重复插入）。"""
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        llm = _CountingSemanticLLM()
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        store = SemanticMemoryStore(inner, llm)
        mgr = MemoryManager(long_term_store=store, llm=llm, user_id="u1", session_id="s")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = mgr.remember("我在上海工作", kind="fact", importance=0.7)
            # 语义等价改写——向量高度相似
            r2 = mgr.remember("本人居住于上海上班", kind="fact", importance=0.6,
                              conflict_threshold=0.7)
        assert r1.memory_id == r2.memory_id, (
            f"向量冲突检测未拦截重复写入：r1={r1.memory_id}, r2={r2.memory_id}"
        )
        # 库中只有 1 条 fact 记录（未新增）
        facts = [r for r in store.all() if r.kind == "fact"]
        assert len(facts) == 1, f"应有且仅有 1 条 fact，实际 {len(facts)}"
        mgr.close()
        store.close()
        inner.close()
    print("✅ test_b13_manager_remember_dedup_by_vector 通过（写入前向量去重生效）")


def test_b13_conflict_falls_back_to_lexical():
    """B13 ③：裸 InMemoryStore（无向量能力）+ embed 拿不到向量时，自动回退词法 Jaccard。"""
    plain = InMemoryStore()
    mgr = MemoryManager(long_term_store=plain, llm=_FakeLLMNoEmbedding(),
                        user_id="u", session_id="s")
    r1 = mgr.remember("用户喜欢用 Python 写后端", kind="preference", importance=0.8)
    # 几乎相同的句子——词法 Jaccard ≥ 0.85
    r2 = mgr.remember("用户喜欢用 Python 写后端", kind="preference", importance=0.8)
    assert r1.memory_id == r2.memory_id, "词法兜底应拦截完全重复"
    assert len(plain.all()) == 1
    mgr.close()
    plain.close()
    print("✅ test_b13_conflict_falls_back_to_lexical 通过（向量不可用时回退词法 Jaccard）")


def test_b13_bugfix_stale_vecs_cleanup_and_embed_query():
    """B13 bug 修复回归：
    1. revise 后旧版本（superseded）向量保留在向量表 / 缓存中（便于审计/回溯），
       但必须通过 status='active' 过滤，不能再被 find_similar / search 命中；
    2. find_similar 必须调用 embed_query（查询侧）而不是 embed_documents（文档侧）。
    用 SemanticMemoryStore（语义兜底，不依赖 chromadb）即可复现。
    """
    from fast_agents.memory import SQLiteMemoryStore, SemanticMemoryStore, MemoryManager
    import tempfile, os

    class _EmbedQuerySpyLLM(FakeLLM):
        """记录 embed_query / embed_documents 的调用次数，随机返回 4 维向量。"""
        def __init__(self):
            super().__init__()
            self.query_calls = 0
            self.doc_calls = 0
            self._rng = __import__("random").Random(42)

        def embed_query(self, text):
            self.query_calls += 1
            return [self._rng.uniform(-1, 1) for _ in range(4)]

        def embed_documents(self, texts):
            self.doc_calls += len(texts)
            return [[self._rng.uniform(-1, 1) for _ in range(4)] for _ in texts]

    tmpdir = tempfile.mkdtemp(prefix="mem_b13_bugfix_")
    db_path = os.path.join(tmpdir, "m.db")
    try:
        llm = _EmbedQuerySpyLLM()
        inner = SQLiteMemoryStore(db_path, user_id="u1", session_id="s1")
        store = SemanticMemoryStore(inner, llm)
        mgr = MemoryManager(llm=llm, long_term_store=store, user_id="u1", session_id="s1")

        # 1) 写一条初始记忆
        r1 = mgr.remember("我在上海做后端开发", kind="fact", conflict_threshold=0.99)
        initial_query_calls = llm.query_calls  # remember 冲突检测应走 embed_query

        # 2) 把它修订成完全不同的内容（修订路径会把原记录标 superseded、插入新记录）
        revised = mgr.revise(r1.memory_id, "我明天要去北京出差", source="test")
        assert revised.memory_id != r1.memory_id, "revise 应生成新 memory_id"

        # 断言：修订后再用旧内容做冲突检测，不应该命中；
        #      旧版本向量保留在表中（status=superseded），但通过 status='active' 过滤被排除；
        #      同时也验证 find_similar 走的是 embed_query（query_calls 增长而不是 doc_calls 单条增长）
        before_q = llm.query_calls
        before_d = llm.doc_calls
        hits = store.find_similar("我在上海做后端开发", limit=3, threshold=0.5)
        # 旧内容不应该再被命中（旧版本 superseded，被 status 过滤排除）
        assert not hits, f"B13 bug: revise 后旧版本仍被命中（status 过滤未生效），hits={hits}"
        # find_similar 应通过 embed_query 走查询侧（BGE/E5 等模型的正确性前提）
        assert llm.query_calls > before_q, (
            f"B13 bug: find_similar 没用 embed_query（query_calls 未增长，"
            f"before={before_q} after={llm.query_calls}）"
        )
        # 向量本身没有被删除（审计/回溯保留）：直接查向量表能看到旧 memory_id 的行
        from fast_agents.database import Query, Filter, FilterOp
        row = inner._db.select_one_where(Query(
            table_name="memory_embeddings",
            filters=inner._scope_where([
                Filter(field="memory_id", op=FilterOp.EQ, value=r1.memory_id),
            ]),
        ))
        assert row is not None, "B13 第二步设计：stale 向量不删除，旧版本向量应仍在表中"
        assert row["status"] == "superseded", f"旧向量 status 应为 superseded，实际 {row['status']!r}"
    finally:
        try:
            store.close()
        except Exception:
            pass
        try:
            inner.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("✅ test_b13_bugfix_stale_vecs_cleanup_and_embed_query 通过"
          "（revise 不删向量、靠 status 过滤 + find_similar 用 embed_query）")


def test_b14_semantic_vector_meta_sqlite_and_memory():
    """B14：SemanticMemoryStore 向量通道补 meta（kind / importance / status）

    1) SQLite 后端：memory_embeddings 新行带上 kind/importance/status 三列，
       vector_meta() 读到的与写入记录一致；
    2) 内存后端（InMemoryStore）：_vec_cache 单结构存 MemoryVector，delete 时一起清；
    3) 两种后端的 meta 口径与 ChromaMemoryStore._meta_for 完全一致
       （同一份记忆换后端，过滤条件结果相同）。
    """
    from fast_agents.database import Filter, FilterOp, Query

    llm = _CountingSemanticLLM()

    # ---------- ① SQLite 后端：独立列真实落盘 ----------
    with tempfile.TemporaryDirectory() as td:
        inner = SQLiteMemoryStore(os.path.join(td, "m.db"), user_id="u1", session_id="s1")
        store = SemanticMemoryStore(inner, llm)
        rec = store.add(MemoryRecord(user_id="u1", session_id="s1",
                                     content="我在上海做后端开发", source="t",
                                     kind="fact", importance=0.7))
        row = inner._db.select_one_where(Query(
            table_name="memory_embeddings",
            filters=[Filter(field="memory_id", op=FilterOp.EQ, value=rec.memory_id)],
        ))
        assert row is not None, "向量行未写入 memory_embeddings"
        assert row["kind"] == "fact", f"kind 列错误：{row}"
        assert abs(float(row["importance"]) - 0.7) < 1e-6, f"importance 列错误：{row}"
        assert row["status"] == "active", f"status 列错误：{row}"

        meta = store.vector_meta(rec.memory_id)
        assert meta is not None, "vector_meta 未取到 meta"
        assert meta["kind"] == "fact" and meta["status"] == "active", meta
        assert abs(meta["importance"] - 0.7) < 1e-6, meta
        assert store.vector_meta("不存在") is None, "未知 id 应返回 None"

        chroma_meta = ChromaMemoryStore._meta_for(rec)
        for key in ("kind", "importance", "status"):
            assert meta[key] == chroma_meta[key], (
                f"{key} 口径与 Chroma 不一致：sqlite={meta[key]} chroma={chroma_meta[key]}"
            )
        store.close()
        inner.close()

    # ---------- ② 内存后端：_vec_cache[MemoryVector] 单结构维护，delete 时一起清 ----------
    from fast_agents.memory.models import MemoryVector as _MemVec
    mem_inner = InMemoryStore()
    mem_store = SemanticMemoryStore(mem_inner, llm)
    r2 = mem_store.add(MemoryRecord(user_id="u1", session_id="s1",
                                    content="我爱吃辣", source="t",
                                    kind="preference", importance=0.8))
    assert r2.memory_id in mem_store._vec_cache, "内存向量未写入 _vec_cache"
    cached = mem_store._vec_cache[r2.memory_id]
    assert isinstance(cached, _MemVec), f"缓存值应为 MemoryVector，实际 {type(cached)}"
    assert cached.kind == "preference" and cached.status == "active", cached
    assert abs(cached.importance - 0.8) < 1e-6, cached
    assert cached.dim > 0 and len(cached.embedding) == cached.dim, "embedding 维度异常"
    m2 = mem_store.vector_meta(r2.memory_id)
    assert m2["kind"] == "preference" and m2["status"] == "active", m2
    assert abs(m2["importance"] - 0.8) < 1e-6, m2
    assert mem_store.vector_meta("不存在") is None

    mem_store.delete(r2.memory_id)
    assert r2.memory_id not in mem_store._vec_cache, "delete 后 _vec_cache 里仍残留该向量"

    print("✅ test_b14_semantic_vector_meta_sqlite_and_memory 通过"
          "（SQLite 声明式 Schema + 内存 MemoryVector 缓存 + 口径与 Chroma 一致）")


def test_bugfix_inmemory_value_semantics_vec_recompute():
    """bug 修复回归：InMemoryStore 出入两端深拷贝，对外为值语义（与 SQLiteMemoryStore 一致）。

    背景：早期 InMemoryStore 的 get / all 直接把内部对象引用交出去，于是
    「get 出来改字段再 update」时，update 内部 old = inner.get() 拿到的与传入的
    record 是同一个对象，content 新旧对比恒为「未变化」，SemanticMemoryStore /
    ChromaMemoryStore 的向量重算被静默跳过（改了内容，向量仍指向旧文本）。
    本用例钉住两件事：
    ① 纯 InMemoryStore 的 add / get / all 均返回独立副本，改动不回写库内；
    ② 语义包装器上「改内容再 update」确实触发向量重算。
    """
    import warnings

    class _DetEmbedLLM:
        """按文本内容确定性编码的假 LLM：同文本同向量、异文本异向量，
        用来判断向量有没有随内容重算（_SemanticFakeLLM 对「上海/北京」给同一向量，区分不了）。"""
        embedding_model = "det-embed"

        @staticmethod
        def _vec(text):
            return [float(len(text)), float(sum(ord(c) for c in text) % 997)]

        def embed_query(self, q):
            return self._vec(q)

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

    # ---------- ① 纯内存后端：值语义 ----------
    plain = InMemoryStore()
    seeded = MemoryRecord(user_id="u1", session_id="s", content="原始内容",
                          source="t", kind="fact", importance=0.5)
    plain.add(seeded)
    mid = seeded.memory_id

    # add 存副本：add 之后外部再改 record，不应污染库内
    seeded.content = "add 后被外部改掉"
    assert plain.get(mid).content == "原始内容", "add 存了引用，外部改动污染了库内数据"

    # get 返回副本：改快照不应回写库内
    snapshot = plain.get(mid)
    snapshot.content = "get 后被改"
    assert plain.get(mid).content == "原始内容", "get 返回活引用，改动回写污染了库内"

    # all 返回副本：批量改不应回写库内
    listed = plain.all()
    listed[0].content = "all 后被改"
    assert plain.get(mid).content == "原始内容", "all 返回活引用，改动回写污染了库内"

    # ---------- ② 语义包装器：改内容再 update 必须重算向量 ----------
    llm = _DetEmbedLLM()
    inner = InMemoryStore()
    store = SemanticMemoryStore(inner, llm)
    rec = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                 source="t", kind="fact", importance=0.7))
    mid2 = rec.memory_id
    vec_old = list(store._vec_cache[mid2].embedding)
    assert vec_old == _DetEmbedLLM._vec("我在上海工作")

    # 模拟上层真实用法：从 store 取记录（副本）→ 改内容 → update
    cur = store.get(mid2)
    cur.content = "我在北京工作"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.update(cur)

    vec_new = list(store._vec_cache[mid2].embedding)
    assert vec_new == _DetEmbedLLM._vec("我在北京工作"), (
        "改内容后未重算向量：库内仍是旧内容的向量"
        "（InMemoryStore 活引用导致 content_changed 误判）"
    )
    assert vec_new != vec_old, "新旧内容向量应不同（用例编码选择有误）"
    assert store.get(mid2).content == "我在北京工作"

    store.close()
    print("✅ test_bugfix_inmemory_value_semantics_vec_recompute 通过"
          "（内存后端值语义 + 改内容触发向量重算）")


def test_b13_bugfix_batch_save_vec_persists_after_compute():
    """B13 Bug1 回归：semantic_batch_get_and_save 批量补算出的向量必须真正落盘。

    背景：基类补算后调的是 self._save_vec(rec)（丢了刚算好的 vec）。SemanticMemoryStore
    的 _save_vec 只收 1 个参数、签名能对上，但内部会再 embed 一次（doc_calls=2）；
    ChromaMemoryStore 压根没有 _save_vec（只有 _save_vec_chroma），直接 AttributeError
    被 except 静默吞掉——于是只要缓存缺失（冷启动 / 换模型 / rebuild 部分失败），
    补算结果永远不沉淀，每次都重算。本用例钉住两条路径补算后都真正落盘、且各只 embed 一次。
    """
    import warnings

    # ---------- ① Semantic 内存后端：补算结果写入 _vec_cache ----------
    llm = _CountingSemanticLLM()
    sem = SemanticMemoryStore(InMemoryStore(), llm)
    r1 = sem.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                              source="t", kind="fact", importance=0.7))
    sem._vec_cache.clear()                       # 模拟冷启动（缓存缺失）
    before = llm.doc_calls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vecs = sem.semantic_batch_get_and_save([r1])
    assert vecs[0] is not None, "补算应返回向量"
    assert r1.memory_id in sem._vec_cache, "❌ 补算的向量没有写入 _vec_cache（Bug1）"
    assert llm.doc_calls - before == 1, (
        f"补算应只调一次 embed_documents，实际 {llm.doc_calls - before} 次（Bug1：重复 embed）"
    )
    # 补算落盘后二次调用应命中缓存、零 embed
    b_cache = llm.doc_calls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sem.semantic_batch_get_and_save([r1])
    assert llm.doc_calls == b_cache, "补算落盘后，二次调用不应再 embed"

    # ---------- ② Chroma 包装器：补算结果写回 Chroma ----------
    class _MemChromaClient:
        """最小内存向量客户端：只实现本用例用到的 get / add（upsert）。"""
        def __init__(self):
            self.cols = {}

        def get_or_create_collection(self, name, *, dimension=None, metric="cosine"):
            self.cols.setdefault(name, {})

        def add(self, collection, ids, embeddings=None, documents=None, metadatas=None):
            col = self.cols.setdefault(collection, {})
            for i, _id in enumerate(ids):
                row = col.setdefault(_id, {"embedding": None, "document": None, "metadata": None})
                if embeddings and i < len(embeddings) and embeddings[i] is not None:
                    row["embedding"] = list(embeddings[i])
                if documents and i < len(documents) and documents[i] is not None:
                    row["document"] = documents[i]
                if metadatas and i < len(metadatas) and metadatas[i] is not None:
                    row["metadata"] = dict(metadatas[i])
            return len(ids)

        def get(self, collection, ids=None, where=None, limit=None,
                include_embeddings=False, include_documents=True, include_metadatas=True):
            col = self.cols.get(collection, {})
            out = []
            for _id, row in col.items():
                if ids and _id not in ids:
                    continue
                d = {"id": _id}
                if include_embeddings:
                    d["embedding"] = row["embedding"]
                if include_documents:
                    d["document"] = row["document"]
                if include_metadatas:
                    d["metadata"] = row["metadata"]
                out.append(d)
            return out[:limit] if limit else out

    llm2 = _CountingSemanticLLM()
    fake = _MemChromaClient()
    chroma = ChromaMemoryStore(InMemoryStore(), llm2, chroma_client=fake,
                               collection_name="bug1_batch_save")
    r2 = chroma.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                 source="t", kind="fact", importance=0.7))
    cname = chroma._collection_name
    fake.cols[cname].pop(r2.memory_id, None)     # 模拟 Chroma 里没有这条向量
    b2 = llm2.doc_calls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vecs2 = chroma.semantic_batch_get_and_save([r2])
    assert vecs2[0] is not None, "补算应返回向量"
    assert r2.memory_id in fake.cols[cname], "❌ 补算的向量没有写回 Chroma（Bug1）"
    assert fake.cols[cname][r2.memory_id]["embedding"] is not None, "❌ Chroma 里向量为空（Bug1）"
    assert llm2.doc_calls - b2 == 1, (
        f"Chroma 路径补算应只调一次 embed_documents，实际 {llm2.doc_calls - b2} 次"
    )

    sem.close()
    chroma.close()
    print("✅ test_b13_bugfix_batch_save_vec_persists_after_compute 通过"
          "（补算向量真正落盘 + 两路径各只 embed 一次）")


def test_b13_bugfix_rebuild_reuses_batch_vecs():
    """Bug2 修复验证：rebuild_embeddings 必须复用批量算出的 vecs，
    调用 embed_documents 次数 = ceil(records/batch_size)，不应退化成逐条 embed。"""
    import warnings

    def _build_sem_store():
        inner = InMemoryStore()
        llm = _CountingSemanticLLM()
        store = SemanticMemoryStore(inner, llm)
        for i, c in enumerate(["苹果", "香蕉", "橙子", "葡萄"]):
            store.add(MemoryRecord(user_id="u1", session_id="s",
                                   content=c, source="t",
                                   kind="fact", importance=0.5,
                                   created_at=datetime.now()))
        return inner, llm, store

    # ---------- Semantic 路径 ----------
    _, llm, store = _build_sem_store()
    # 清掉向量缓存，模拟冷启动/换模型
    store._vec_cache.clear()
    llm.doc_calls = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = store.rebuild_embeddings(batch_size=2)
    assert n == 4, n
    assert llm.doc_calls == 2, f"Semantic rebuild 应只调 2 次 embed_documents，实际 {llm.doc_calls}"
    store.close()

    # ---------- Chroma 路径（内存假客户端，不依赖真 chromadb）----------
    class _MemChroma:
        def __init__(self):
            self.data = {}
            self.add_calls = self.delete_calls = self.search_calls = 0
        def get_or_create_collection(self, name, metric="cosine"):
            return True
        def add(self, collection, ids, embeddings=None, documents=None, metadatas=None):
            self.add_calls += 1
            for i, mid in enumerate(ids):
                self.data[mid] = (embeddings[i] if embeddings else None,
                                  documents[i] if documents else None,
                                  metadatas[i] if metadatas else None)
        def delete(self, collection, ids=None):
            self.delete_calls += 1
            for mid in (ids or []):
                self.data.pop(mid, None)
        def get(self, collection, ids=None):
            vecs = [self.data.get(mid, (None, None, None))[0] for mid in ids]
            docs = [self.data.get(mid, (None, None, None))[1] for mid in ids]
            metas = [self.data.get(mid, (None, None, None))[2] for mid in ids]
            return vecs, docs, metas
        def search(self, collection, query_embedding, n_results=10, where=None):
            self.search_calls += 1
            return []
        def drop_collection(self, name):
            self.data.clear()
        def close(self):
            pass

    inner = InMemoryStore()
    llm = _CountingSemanticLLM()
    fake = _MemChroma()
    store = ChromaMemoryStore(inner, llm, chroma_client=fake,
                              collection_name="rb2")
    for i, c in enumerate(["苹果", "香蕉", "橙子", "葡萄"]):
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content=c, source="t",
                               kind="fact", importance=0.5,
                               created_at=datetime.now()))
    # 清空假客户端数据，模拟冷启动
    fake.data.clear()
    fake.add_calls = 0
    llm.doc_calls = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = store.rebuild_embeddings(batch_size=2)
    assert n == 4, n
    assert llm.doc_calls == 2, f"Chroma rebuild 应只调 2 次 embed_documents（批量），实际 {llm.doc_calls}"
    # _save_vec_chroma 对每条记录单独 add，4 条 = 4 次 add；关键是 embed 只算 2 次、不逐条算
    assert fake.add_calls == 4, f"Chroma rebuild 应调 4 次 add（每条 1 次），实际 {fake.add_calls}"
    store.close()
    print("✅ test_b13_bugfix_rebuild_reuses_batch_vecs 通过"
          "（rebuild 复用批量向量，不逐条重算）")


def test_b13_expired_lexical_duplicate_renewed():
    """B13 过期修复①：词法兜底路径下，命中「已过期但仍 active」的记忆时不能丢弃新内容，
    而应续期旧条（过期 ≠ 删除；否则随后 purge_expired 会把旧条也清掉，造成新旧双丢）。"""
    from fast_agents.utils.time_utils import cstnow

    plain = InMemoryStore()
    mgr = MemoryManager(long_term_store=plain, llm=_FakeLLMNoEmbedding(),
                        user_id="u", session_id="s")
    # 直接放一条「已过期但仍 active」的记忆（模拟 ttl 到期、compact 尚未跑）
    expired = plain.add(MemoryRecord(
        user_id="u", session_id="s", content="用户喜欢用 Python 写后端",
        source="t", kind="preference", importance=0.8,
        expires_at=cstnow() - timedelta(seconds=1),
    ))
    assert expired.is_expired() is True, "前置条件：这条应已过期"
    assert plain.get(expired.memory_id).status == "active", "前置条件：过期不改 status，仍是 active"

    # 再写一条完全相同的记忆
    r2 = mgr.remember("用户喜欢用 Python 写后端", kind="preference", importance=0.8)
    assert r2.memory_id == expired.memory_id, "过期重复应续期旧条，而不是新建一条"
    assert len(plain.all()) == 1, f"库中不应新增记录，实际 {len(plain.all())}"
    renewed = plain.get(expired.memory_id)
    assert renewed.status == "active"
    assert renewed.is_expired() is False, "续期后这条不应再算过期"

    # 续期后 compact 不再把它当到期项清掉（新旧都不丢）
    rep = mgr.compact()
    assert rep["expired"] == 0, f"续期后不应再被清理，实际 {rep}"
    assert plain.get(expired.memory_id).status == "active"
    mgr.close()
    plain.close()
    print("✅ test_b13_expired_lexical_duplicate_renewed 通过（过期重复→续期旧条，不双丢）")


def test_b13_expired_semantic_conflict_superseded():
    """B13 过期修复②：向量路径下命中「已过期但仍 active」且内容有差异的记忆时，
    应写入新条并让旧条 superseded 留痕（新内容生效、旧条不再占用 active 位）。"""
    import warnings
    from fast_agents.utils.time_utils import cstnow

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        llm = _CountingSemanticLLM()
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        store = SemanticMemoryStore(inner, llm)
        mgr = MemoryManager(long_term_store=store, llm=llm, user_id="u1", session_id="s")
        expired = store.add(MemoryRecord(
            user_id="u1", session_id="s", content="我在上海工作",
            source="t", kind="fact", importance=0.7,
            expires_at=cstnow() - timedelta(seconds=1),
        ))
        assert store.get(expired.memory_id).status == "active"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 语义相近、内容不同的改写 → 命中过期旧条（threshold 调低命中假向量）
            r2 = mgr.remember("本人居住于上海上班", kind="fact", importance=0.6,
                              conflict_threshold=0.7)
        assert r2.memory_id != expired.memory_id, "过期且内容不同的冲突应写入新条"
        assert r2.supersedes == expired.memory_id, "新条应指向被取代的旧条"
        old = store.get(expired.memory_id)
        assert old.status == "superseded", f"过期旧条应标记 superseded，实际 {old.status}"
        assert store.get(r2.memory_id).status == "active"
        mgr.close()
        store.close()
        inner.close()
    print("✅ test_b13_expired_semantic_conflict_superseded 通过（过期+内容不同→新条取代旧条）")


def test_b13_conflict_error_raised_on_duplicate():
    """B13 异常接线：remember(raise_on_conflict=True) 命中有效重复时抛 MemoryConflictError，
    不再静默写入；异常携带 new_record（未入库）与 conflicts 列表。"""
    from fast_agents.memory import MemoryConflictError

    plain = InMemoryStore()
    mgr = MemoryManager(long_term_store=plain, llm=_FakeLLMNoEmbedding(),
                        user_id="u", session_id="s")
    r1 = mgr.remember("用户偏好深色主题", kind="preference", importance=0.7)

    # 默认（raise_on_conflict=False）：维持静默去重、不抛异常
    r_dup = mgr.remember("用户偏好深色主题", kind="preference", importance=0.7)
    assert r_dup.memory_id == r1.memory_id
    assert len(plain.all()) == 1

    # 显式开启：抛 MemoryConflictError，且库中不新增
    try:
        mgr.remember("用户偏好深色主题", kind="preference", importance=0.7,
                     raise_on_conflict=True)
        raise AssertionError("命中有效重复且 raise_on_conflict=True 时应抛 MemoryConflictError")
    except MemoryConflictError as exc:
        assert exc.new_record.content == "用户偏好深色主题"
        assert exc.new_record.memory_id != r1.memory_id, "new_record 应是尚未入库的预览记录"
        assert any(rec.memory_id == r1.memory_id for rec, _ in exc.conflicts), exc.conflicts
        assert all(ctype == "duplicate" for _, ctype in exc.conflicts)
    assert len(plain.all()) == 1, f"抛异常时不应写入新记录，实际 {len(plain.all())}"
    mgr.close()
    plain.close()
    print("✅ test_b13_conflict_error_raised_on_duplicate 通过（显式冲突抛异常、不静默写入）")


def test_b13_check_conflict_readonly_and_expired_excluded():
    """B13 只读接口：check_conflict 返回有效冲突列表但不写入；过期记忆不算作冲突。"""
    from fast_agents.utils.time_utils import cstnow

    plain = InMemoryStore()
    mgr = MemoryManager(long_term_store=plain, llm=_FakeLLMNoEmbedding(),
                        user_id="u", session_id="s")

    # 无冲突：空列表
    assert mgr.check_conflict("一句与库内毫无关系的话") == []

    r1 = mgr.remember("用户的部署环境是 CentOS 7", kind="fact")
    hits = mgr.check_conflict("用户的部署环境是 CentOS 7")
    assert any(rec.memory_id == r1.memory_id for rec, _ in hits), hits
    assert all(ctype == "duplicate" for _, ctype in hits)
    assert len(plain.all()) == 1, "check_conflict 必须只读、不写入"

    # 过期记忆不列为冲突（过期 ≠ 删除，由续期 / 取代消化）
    plain.add(MemoryRecord(
        user_id="u", session_id="s", content="本次验证码是 123456",
        source="t", kind="episode",
        expires_at=cstnow() - timedelta(seconds=1),
    ))
    assert mgr.check_conflict("本次验证码是 123456") == [], "过期记忆不应算作有效冲突"
    mgr.close()
    plain.close()
    print("✅ test_b13_check_conflict_readonly_and_expired_excluded 通过（只读探测、过期不算冲突）")


def test_v193_search_no_write_amp_on_cold_cache():
    """v1.9.3：SemanticMemoryStore.search 在冷库下只补算词法粗排 top-N 条，不全库 embed。

    - 冷库（缓存空）首次 search：只对词法粗排进 top-N（默认 20）的候选做一次小批量
      embed 并落盘，不再像 v1.9.2 那样一次把全库所有缺失向量都补算；
    - 第二次同 query search：所有涉及候选已有向量，零额外 embed；
    - 大库（100+ 条）场景：首次 search 也只触发 1 次批量 embed（上限 20 条），
      不是全库 100 条；B2 同义召回（"用户所在地"→"我在上海工作"）依然成立。
    """
    import random as _rnd

    class _ColdCacheLLM:
        """spy LLM：记录 embed_documents 被调用的次数和每次入参条数。"""
        def __init__(self):
            self.query_calls = 0
            self.doc_batch_sizes: List[int] = []  # 每次 embed_documents 的入参条数
            self.embedding_model = "cold-cache-mock"
            self._rng = _rnd.Random(42)

        def embed_query(self, text):
            self.query_calls += 1
            # 让"上海"类语义向量固定下来，保证 B2 同义召回在小库下可复现
            if "所在地" in text:
                return [0.9, 0.1, 0.1, 0.1]
            return [self._rng.uniform(-1, 1) for _ in range(4)]

        def embed_documents(self, texts):
            self.doc_batch_sizes.append(len(texts))
            result = []
            for t in texts:
                if "上海" in t:
                    result.append([0.9, 0.1, 0.1, 0.1])  # 与"所在地"query 高相似
                else:
                    result.append([self._rng.uniform(-1, 1) for _ in range(4)])
            return result

    # ---- 场景 A：小库 5 条，验证首次只补算一次、二次零补算、同义召回命中 ----
    llm = _ColdCacheLLM()
    plain_a = InMemoryStore()
    for i in range(5):
        plain_a.add(MemoryRecord(
            user_id="u", session_id="s",
            content=f"第 {i} 条测试记忆内容，彼此独立不重复",
            source="t", kind="fact",
        ))
    plain_a.add(MemoryRecord(
        user_id="u", session_id="s", content="我在上海工作",
        source="t", kind="fact", importance=0.7,
    ))
    sem_a = SemanticMemoryStore(plain_a, llm=llm)

    batches_before = len(llm.doc_batch_sizes)
    q_before = llm.query_calls
    results = sem_a.search("用户所在地", limit=3)
    q_after = llm.query_calls
    batches_after = len(llm.doc_batch_sizes)

    assert q_after - q_before == 1, f"查询本身应 embed query 1 次，实际 {q_after - q_before}"
    # 首次 search：应触发 1 次批量补算（补词法粗排 top-N 里缺向量的候选）
    assert batches_after - batches_before == 1, (
        f"冷库首次 search 应触发 1 次小批量 embed（top-N 候选），实际 {batches_after - batches_before} 次"
    )
    # 补算条数不能超过 prefetch_top_n=20
    first_batch = llm.doc_batch_sizes[batches_before]
    assert first_batch <= 20, f"首次补算条数 {first_batch} 超过 prefetch_top_n=20"
    # B2 同义召回应命中"我在上海工作"
    assert any("上海" in r.content for r in results), (
        f"两段式检索下 B2 同义召回应命中，实际 top3={[r.content for r in results]}"
    )

    # 第二次 search：所有涉及候选已有向量，零新增 embed
    batches_before_2 = len(llm.doc_batch_sizes)
    sem_a.search("用户所在地", limit=3)
    batches_after_2 = len(llm.doc_batch_sizes)
    assert batches_after_2 == batches_before_2, (
        f"第二次 search 不应再触发补算（向量已缓存），新增 {batches_after_2 - batches_before_2} 次"
    )

    sem_a.close()
    plain_a.close()

    # ---- 场景 B：大库 100 条无关 + 1 条目标，验证不全库补算 ----
    llm2 = _ColdCacheLLM()
    plain_b = InMemoryStore()
    for i in range(100):
        plain_b.add(MemoryRecord(
            user_id="u", session_id="s",
            content=f"第 {i} 条无关测试内容，编号为 {i}",
            source="t", kind="fact", importance=0.3,
        ))
    plain_b.add(MemoryRecord(
        user_id="u", session_id="s", content="我在上海工作",
        source="t", kind="fact", importance=0.7,
    ))
    sem_b = SemanticMemoryStore(plain_b, llm=llm2)

    batches_before = len(llm2.doc_batch_sizes)
    results_b = sem_b.search("用户所在地", limit=3, prefetch_top_n=20)
    batches_after = len(llm2.doc_batch_sizes)

    assert batches_after - batches_before == 1, (
        f"大库首次 search 应只触发 1 次批量补算，实际 {batches_after - batches_before} 次"
    )
    first_batch_b = llm2.doc_batch_sizes[batches_before]
    assert first_batch_b <= 20, (
        f"v1.9.3 回归：大库首次 search 补算 {first_batch_b} 条，超过 prefetch_top_n=20（写放大）"
    )
    # 语义精排应把"上海"提到 top3
    assert any("上海" in r.content for r in results_b), (
        f"两段式检索下大库 B2 同义召回应命中，实际 top3={[r.content for r in results_b]}"
    )

    sem_b.close()
    plain_b.close()
    print(f"✅ test_v193_search_no_write_amp_on_cold_cache 通过"
          f"（小库首次 batch={first_batch}、大库首次 batch={first_batch_b}，均≤20；二次零补算；B2 召回命中）")


def test_v193_prefetch_missing_writes_vs_readonly():
    """v1.9.3：写入路径（remember 冲突检测）会补算缺失向量；只读探测（check_conflict）不会。"""
    import random as _rnd

    class _SpyLLM:
        def __init__(self):
            self.query_calls = 0
            self.doc_calls = 0
            self.embedding_model = "spy-mock"
            self._rng = _rnd.Random(7)

        def embed_query(self, text):
            self.query_calls += 1
            return [self._rng.uniform(-1, 1) for _ in range(4)]

        def embed_documents(self, texts):
            self.doc_calls += 1
            return [[self._rng.uniform(-1, 1) for _ in range(4)] for _ in texts]

    llm = _SpyLLM()
    plain = InMemoryStore()
    # 关键：long_term 必须是 SemanticMemoryStore，find_similar 才会真的走我们加的
    # prefetch_missing 分支；先把 SemanticMemoryStore 包上，再直接往内层塞一条
    # （绕过 sem.add，模拟"冷库启动、记录已在库但无向量"的场景）
    sem = SemanticMemoryStore(plain, llm=llm)
    mgr = MemoryManager(long_term_store=sem, llm=llm, user_id="u", session_id="s")
    plain.add(MemoryRecord(
        user_id="u", session_id="s",
        content="用户偏好深色主题",
        source="t", kind="preference",
    ))

    # 1) 只读探测：不应触发 doc embed 补算（prefetch_missing=False）
    doc_before = llm.doc_calls
    hits = mgr.check_conflict("用户偏好深色主题")
    doc_after_ro = llm.doc_calls
    assert doc_after_ro == doc_before, (
        f"check_conflict 只读路径不应触发 doc embed，doc_calls 从 {doc_before} 涨到 {doc_after_ro}"
    )

    # 2) remember 写入路径：prefetch_missing=True 会补算缺失向量，正确去重
    doc_before = llm.doc_calls
    mgr.remember("用户偏好深色主题")
    doc_after_wr = llm.doc_calls
    assert doc_after_wr > doc_before, (
        "remember 写入查重应 prefetch_missing=True 补算缺失向量，"
        f"doc_calls 未增长（before={doc_before}, after={doc_after_wr}）"
    )
    # 命中重复、未新增（库里仍是 1 条）
    assert len(plain.all()) == 1

    mgr.close()
    sem.close()
    plain.close()
    print("✅ test_v193_prefetch_missing_writes_vs_readonly 通过（写入/只读补算口径分离）")

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
    test_b10_fix_extract_after_window_full()  # AI新加的测试
    test_v183_semantic_cache_avoids_re_embed()
    test_merge_similar_reuses_semantic_cache()
    test_v183_embedding_stats_default_and_errors_export()
    test_b13_semantic_find_similar_detects_conflict()
    test_b13_manager_remember_dedup_by_vector()
    test_b13_conflict_falls_back_to_lexical()
    test_b13_bugfix_stale_vecs_cleanup_and_embed_query()
    test_b14_semantic_vector_meta_sqlite_and_memory()
    test_bugfix_inmemory_value_semantics_vec_recompute()
    test_b13_bugfix_batch_save_vec_persists_after_compute()
    test_b13_bugfix_rebuild_reuses_batch_vecs()
    test_b13_expired_lexical_duplicate_renewed()  # AI新加的测试
    test_b13_expired_semantic_conflict_superseded()  # AI新加的测试
    test_b13_conflict_error_raised_on_duplicate()  # AI新加的测试
    test_b13_check_conflict_readonly_and_expired_excluded()  # AI新加的测试
    test_v193_search_no_write_amp_on_cold_cache()  # AI新加的测试
    test_v193_prefetch_missing_writes_vs_readonly()  # AI新加的测试


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
