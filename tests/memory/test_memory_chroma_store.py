"""记忆模块测试（Chroma 专项）：ChromaMemoryStore 包装器，不依赖真实 LLM。

本文件是从 test_memory.py 拆出的「以 ChromaMemoryStore 为主体」的用例，
其中两个用例需要真实 chromadb（环境未安装时自动 skip）。

覆盖：
1. ChromaMemoryStore 阶段 1：chroma_client=None 纯词法降级 CRUD/search
2. B5 方法代理（touch / revise / retract / archive_stale / merge_similar）
3. 直接插进 MemoryManager 的集成零感知
4. v1.8.3 ②：chroma_client=None 整类降级 + 集合名归一化
5. v1.8.3 ②b：真实 ChromaClient 语义通道 + 二次 search 读 Chroma 缓存 + delete 清向量
6. B13 ④：ChromaMemoryStore.find_similar 走 ANN 近邻做冲突检测（真实 chromadb）
7. B13 第一步：stale 向量不删，靠 metadata.status + where={"status":"active"} 过滤
8. B13 bugfix：update content 走 delete-then-add，仅 status 变化不重算 embedding
"""

import os
import re
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.memory import (
    ChromaMemoryStore,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    SQLiteMemoryStore,
)


# ===================== 共享假 LLM（与 test_memory.py 保持同一实现） =====================

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


class _FakeLLMNoEmbedding:
    """假 LLM：embedding 永远失败，逼 ChromaMemoryStore 走纯词法降级"""
    embedding_model = "fake-none"

    def embed_query(self, q):           return None
    def embed_documents(self, texts):   return None


# ===================== ChromaMemoryStore 阶段 1 测试 =====================
# 阶段 1：不引 chromadb、不传真实客户端，验证 ①纯词法降级②代理记账③Manager 集成。

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


# ==================== v1.8.3 向量通道完善（Chroma 侧） ====================

def test_v183_chroma_client_none_is_lexical_only():
    """v1.8.3 ②：ChromaMemoryStore 在 chroma_client=None 时整类降级为纯词法直通 inner，
    集合名归一化仍生效；语义通道关闭但不报错。"""
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        # 传入含中文/特殊符号的 collection_name，验证归一化不炸
        llm = _CountingSemanticLLM()
        store = ChromaMemoryStore(inner, llm, collection_name="测试/集合:名-含中文")
        # 归一化后非法字符被替换、首尾字母数字，不以 / : 结尾；
        # 长度够 3 时不会自动加 fa_mem_ 前缀（长度不足才加）。
        assert "/" not in store._collection_name and ":" not in store._collection_name
        assert re.match(r"^[a-zA-Z0-9]", store._collection_name)
        assert re.match(r".*[a-zA-Z0-9]$", store._collection_name)
        assert 3 <= len(store._collection_name) <= 63

        # chroma_client=None → 语义通道关闭，embedding_stats 显示 available=False
        stats = store.embedding_stats()
        assert stats["available"] is False
        assert stats["chroma_connected"] is False

        # 词法能命中的记录正常召回
        store.add(MemoryRecord(user_id="u1", session_id="s",
                               content="我爱吃辣", source="t",
                               kind="preference", importance=0.8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits = store.search("爱吃辣", limit=5)
        assert any("辣" in h.content for h in hits), [h.content for h in hits]

        store.close()
        inner.close()
    print("✅ test_v183_chroma_client_none_is_lexical_only 通过（chroma_client=None 纯词法降级 + 集合名归一化）")


def test_v183_chroma_with_real_chromaclient_semantic():
    """v1.8.3 ②b：传真实 ChromaClient 时，语义通道可用、二次 search 从 Chroma 读缓存、
    delete 同步清向量。环境未安装 chromadb 时自动 skip。"""
    from fast_agents.database import ChromaClient
    from fast_agents.database.errors import DatabaseDriverMissingError
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        chroma_path = os.path.join(td, "chroma")
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        llm = _CountingSemanticLLM()
        try:
            chroma = ChromaClient(path=chroma_path)
        except (DatabaseDriverMissingError, ImportError, ModuleNotFoundError):
            print("⏭️  skip test_v183_chroma_with_real_chromaclient_semantic（未安装 chromadb）")
            inner.close()
            return
        try:
            store = ChromaMemoryStore(inner, llm, chroma_client=chroma,
                                      collection_name="b13_test")
            stats = store.embedding_stats()
            assert stats["available"] is True
            assert stats["chroma_connected"] is True
            assert stats["dimension"] in (None, 3)

            store.add(MemoryRecord(user_id="u1", session_id="s",
                                   content="我在上海工作", source="t",
                                   kind="fact", importance=0.7))
            store.add(MemoryRecord(user_id="u1", session_id="s",
                                   content="我爱吃辣", source="t",
                                   kind="preference", importance=0.8))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hits = store.search("用户所在地", limit=5)
            assert any("上海" in h.content for h in hits), [h.content for h in hits]

            after_first = llm.doc_calls
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.search("口味偏好", limit=5)
            assert llm.doc_calls == after_first, "Chroma 缓存未命中导致重复 embed"

            stats2 = store.embedding_stats()
            assert stats2["cached_total"] >= 2
            assert stats2["dimension"] == 3

            # delete 后 Chroma 向量也被清掉
            r_first = store.all()[0]
            store.delete(r_first.memory_id)
            assert store.embedding_stats()["cached_total"] == 1
            store.close()
        finally:
            try:
                chroma.close()
            except Exception:
                pass
            inner.close()
    print("✅ test_v183_chroma_with_real_chromaclient_semantic 通过（真实 ChromaClient 语义通道 + 缓存命中）")


# ===================== B13：向量冲突检测（Chroma 侧） =====================

def test_b13_chroma_find_similar_ann():
    """B13 ④：ChromaMemoryStore.find_similar 用 ANN 近邻做冲突检测（B13 主路径）。
    环境未安装 chromadb 时自动 skip。"""
    from fast_agents.database import ChromaClient
    from fast_agents.database.errors import DatabaseDriverMissingError
    import warnings
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "m.db")
        chroma_path = os.path.join(td, "chroma")
        inner = SQLiteMemoryStore(db, user_id="u1", session_id="s")
        llm = _CountingSemanticLLM()
        try:
            chroma = ChromaClient(path=chroma_path)
        except (DatabaseDriverMissingError, ImportError, ModuleNotFoundError):
            print("⏭️  skip test_b13_chroma_find_similar_ann（未安装 chromadb）")
            inner.close()
            return
        try:
            store = ChromaMemoryStore(inner, llm, chroma_client=chroma,
                                      collection_name="b13_conflict")
            store.add(MemoryRecord(user_id="u1", session_id="s",
                                   content="我在上海工作", source="t",
                                   kind="fact", importance=0.7))
            store.add(MemoryRecord(user_id="u1", session_id="s",
                                   content="我爱吃辣", source="t",
                                   kind="preference", importance=0.8))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hits = store.find_similar("本人居住于上海上班", limit=3, threshold=0.7)
            assert hits, "Chroma ANN find_similar 应召回到冲突记忆"
            top_rec, top_sim = hits[0]
            assert "上海" in top_rec.content
            assert top_sim >= 0.7
            store.close()
        finally:
            try:
                chroma.close()
            except Exception:
                pass
            inner.close()
    print("✅ test_b13_chroma_find_similar_ann 通过（Chroma ANN 近邻冲突检测）")


def test_b13_chroma_stale_cleanup_and_where_filter():
    """B13 第一步【chroma_store】stale 向量不删除，靠 metadata.status + where 过滤。
    设计原则：status 非 active 的向量保留在 Chroma 中以便审计/回溯，
    但 search/find_similar 必须用 where={"status":"active"} 在向量库侧过滤 stale；
    revise/retract/archive_stale/merge_similar/purge_expired 这些仅改状态的路径
    不清向量，只同步 metadata.status。
    覆盖：
    1. _meta_for 给 metadata 加 status="active"；
    2. _query_chroma（search 语义路径）与 find_similar 都带 where={"status":"active"}；
    3. revise：新版本写新向量，老版本向量保留但 metadata.status 同步为 superseded；
    4. retract：向量保留，metadata.status 同步为 retracted；
    5. 语义 search 路径对 non-active 向量不命中；
    6. find_similar 路径对 non-active 向量不命中。
    """
    import math

    class _FakeChromaClient:
        """最小可用的内存向量客户端（鸭子类型，对齐 VectorClientBase 签名）。"""
        def __init__(self):
            self.cols = {}

        def get_or_create_collection(self, name, *, dimension=None, metric="cosine"):
            self.cols.setdefault(name, {})

        def drop_collection(self, name):
            self.cols.pop(name, None)

        def add(self, collection, ids, embeddings=None, documents=None, metadatas=None):
            col = self.cols.setdefault(collection, {})
            for i, _id in enumerate(ids):
                # upsert 语义：已存在时只覆盖传入的非 None 字段，保留下已有向量/文档
                existing = col.get(_id, {"embedding": None, "document": None, "metadata": None})
                if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                    existing["embedding"] = list(embeddings[i])
                if documents is not None and i < len(documents) and documents[i] is not None:
                    existing["document"] = documents[i]
                if metadatas is not None and i < len(metadatas) and metadatas[i] is not None:
                    existing["metadata"] = dict(metadatas[i])
                col[_id] = existing
            return len(ids)

        def update(self, collection, ids, embeddings=None, documents=None, metadatas=None):
            """兼容别名：底层也是 upsert 语义。"""
            return self.add(collection, ids, embeddings=embeddings,
                            documents=documents, metadatas=metadatas)

        def delete(self, collection, ids=None, where=None):
            col = self.cols.get(collection, {})
            cnt = 0
            for _id in (ids or []):
                if _id in col:
                    del col[_id]
                    cnt += 1
            return cnt

        def get(self, collection, ids=None, where=None, limit=None,
                include_embeddings=False, include_documents=True, include_metadatas=True):
            col = self.cols.get(collection, {})
            out = []
            for _id, rec in col.items():
                if ids and _id not in ids:
                    continue
                d = {"id": _id}
                if include_embeddings:
                    d["embedding"] = rec["embedding"]
                if include_documents:
                    d["document"] = rec["document"]
                if include_metadatas:
                    d["metadata"] = rec["metadata"]
                out.append(d)
            return out[:limit] if limit else out

        def count(self, collection, where=None):
            return len(self.cols.get(collection, {}))

        def search(self, collection, query_embedding, *, n_results=10, where=None,
                   include_embeddings=False):
            col = self.cols.get(collection, {})
            scored = []
            for _id, rec in col.items():
                emb = rec["embedding"]
                if emb is None:
                    continue
                if where:
                    md = rec.get("metadata") or {}
                    if any(md.get(k) != v for k, v in where.items()):
                        continue
                dot = sum(a * b for a, b in zip(query_embedding, emb))
                na = math.sqrt(sum(a * a for a in query_embedding))
                nb = math.sqrt(sum(b * b for b in emb))
                sim = dot / (na * nb) if na and nb else 0.0
                scored.append((_id, 1.0 - sim))
            scored.sort(key=lambda x: x[1])
            return [{"id": i, "distance": d} for i, d in scored[:n_results]]

    class _FakeLLM:
        """按词项类别做 3 维 one-hot-ish 向量；query/document 同分布。"""
        embedding_model = "fake-semantic"

        def _vec(self, text):
            v = [0.0, 0.0, 0.0]
            if any(w in text for w in ("上海", "北京", "住", "所在", "位置", "在哪", "地")):
                v[0] = 1.0
            if any(w in text for w in ("工作", "上班", "公司", "开发", "后端")):
                v[1] = 1.0
            if any(w in text for w in ("吃", "辣", "口味", "喜欢", "爱")):
                v[2] = 1.0
            return v

        def embed_query(self, q):
            return self._vec(q)

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

    fake = _FakeChromaClient()
    inner = InMemoryStore()
    store = ChromaMemoryStore(inner, _FakeLLM(), chroma_client=fake,
                              collection_name="b13_stale")
    cname = store._collection_name
    r1 = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                                source="t", kind="fact", importance=0.7))
    store.add(MemoryRecord(user_id="u1", session_id="s", content="我爱吃辣",
                           source="t", kind="preference", importance=0.8))

    # 1) metadata 带 status
    assert fake.cols[cname][r1.memory_id]["metadata"]["status"] == "active"

    # 2) find_similar 走 Chroma ANN 召回冲突
    hits = store.find_similar("本人居住于上海上班", limit=3, threshold=0.7)
    assert hits and "上海" in hits[0][0].content, f"未召回冲突：{hits}"

    # 3) revise：新版本写入新向量，老版本保留向量但 metadata.status 同步为 superseded（不清向量）
    old_id = r1.memory_id
    revised = store.revise(old_id, "我明天要去北京出差", source="test")
    # 旧版本向量保留（审计/回溯可查）——不再删除
    assert old_id in fake.cols[cname], "revise 后旧版本向量应保留（不清向量）"
    assert fake.cols[cname][old_id]["metadata"]["status"] == "superseded", \
        f"revise 后旧版本 metadata.status 应同步为 superseded，实际 {fake.cols[cname][old_id]['metadata']}"
    assert revised.memory_id in fake.cols[cname], "revise 后新版本向量应写入"
    assert fake.cols[cname][revised.memory_id]["metadata"]["status"] == "active"

    # 4) 旧内容不再被 find_similar 命中（靠 where={"status":"active"} 过滤，而不是删向量）
    hits_after = store.find_similar("我在上海工作", limit=3, threshold=0.5)
    assert not any(h[0].memory_id == old_id for h in hits_after), \
        "旧版本（superseded）应被 where 过滤，不应被命中"

    # 5) retract：保留向量，只同步 metadata.status=retracted，find_similar 不再命中
    rid = revised.memory_id
    store.retract(rid)
    assert rid in fake.cols[cname], "retract 后向量应保留（不清向量）"
    assert fake.cols[cname][rid]["metadata"]["status"] == "retracted", \
        f"retract 后 metadata.status 应为 retracted，实际 {fake.cols[cname][rid]['metadata']}"
    hits_retract = store.find_similar("北京出差", limit=3, threshold=0.0)
    assert not any(h[0].memory_id == rid for h in hits_retract), \
        "retracted 版本应被 where 过滤"

    # 6) 语义 search 路径同样走 where 过滤：把"爱吃辣"标记 archived 后 search 不再命中
    spicy_id = next(_id for _id, rec in fake.cols[cname].items()
                    if rec.get("document") and "辣" in rec["document"])
    spicy_rec = inner.get(spicy_id)
    assert spicy_rec is not None
    spicy_rec.status = "archived"
    store.update(spicy_rec)  # 通过 store.update 走正规路径，会同步 metadata.status 到 Chroma
    assert fake.cols[cname][spicy_id]["metadata"]["status"] == "archived", \
        "update(status→archived) 应同步 metadata 到 Chroma"
    hits_search = store.search("我爱吃辣", limit=3)
    assert not any(r.memory_id == spicy_id for r in hits_search), \
        "archived 记录应被 _query_chroma 的 where 过滤"
    # 恢复 active 后应能命中
    spicy_rec.status = "active"
    store.update(spicy_rec)
    hits_search2 = store.search("我爱吃辣", limit=3)
    assert any(r.memory_id == spicy_id for r in hits_search2), \
        "恢复 active 后应能被语义 search 命中"

    store.close()
    print("✅ test_b13_chroma_stale_cleanup_and_where_filter 通过"
          "（stale 向量保留、metadata.status 同步、where 过滤生效）")


def test_b13_chroma_update_content_replace_vec():
    """B13 bugfix：update content 变化时，必须『先删旧向量 + 再加新向量』两步走，
    不能只依赖 ChromaClient.add 底层的隐式 upsert（语义上 add 是新增，不是更新；
    未来 add 收紧为真 add 时会抛 DuplicateIDError，或因 payload 不全导致旧字段残留）。
    验证：
    1. content 变化后，Chroma 里该 id 对应的 document 与 embedding 都被新内容替换，
       不是在旧条目上只覆盖局部字段；
    2. metadata.status 保持 active；
    3. content 未变化（仅 metadata/status 变化）时，不触发删+加流程，走 _update_vec_metadata
       同步 metadata，embedding 不变。
    """
    import math

    class _CountingFakeChromaClient:
        """记录 add/delete 调用次数与最后一次 payload，用来断言 update 的调用路径。"""
        def __init__(self):
            self.cols = {}
            self.add_calls = 0
            self.delete_calls = 0

        def get_or_create_collection(self, name, *, dimension=None, metric="cosine"):
            self.cols.setdefault(name, {})

        def drop_collection(self, name):
            self.cols.pop(name, None)

        def add(self, collection, ids, embeddings=None, documents=None, metadatas=None):
            self.add_calls += 1
            col = self.cols.setdefault(collection, {})
            for i, _id in enumerate(ids):
                col[_id] = {
                    "embedding": list(embeddings[i]) if embeddings and i < len(embeddings) and embeddings[i] is not None else None,
                    "document": documents[i] if documents and i < len(documents) else None,
                    "metadata": dict(metadatas[i]) if metadatas and i < len(metadatas) else None,
                }
            return len(ids)

        def delete(self, collection, ids=None, where=None):
            col = self.cols.get(collection, {})
            cnt = 0
            for _id in (ids or []):
                if _id in col:
                    del col[_id]
                    self.delete_calls += 1
                    cnt += 1
            return cnt

        def get(self, collection, ids=None, where=None, limit=None,
                include_embeddings=False, include_documents=True, include_metadatas=True):
            col = self.cols.get(collection, {})
            out = []
            for _id, rec in col.items():
                if ids and _id not in ids:
                    continue
                d = {"id": _id}
                if include_embeddings:
                    d["embedding"] = rec["embedding"]
                if include_documents:
                    d["document"] = rec["document"]
                if include_metadatas:
                    d["metadata"] = rec["metadata"]
                out.append(d)
            return out[:limit] if limit else out

        def count(self, collection, where=None):
            return len(self.cols.get(collection, {}))

        def search(self, collection, query_embedding, *, n_results=10, where=None,
                   include_embeddings=False):
            col = self.cols.get(collection, {})
            scored = []
            for _id, rec in col.items():
                emb = rec["embedding"]
                if emb is None:
                    continue
                if where:
                    md = rec.get("metadata") or {}
                    if any(md.get(k) != v for k, v in where.items()):
                        continue
                dot = sum(a * b for a, b in zip(query_embedding, emb))
                na = math.sqrt(sum(a * a for a in query_embedding))
                nb = math.sqrt(sum(b * b for b in emb))
                sim = dot / (na * nb) if na and nb else 0.0
                scored.append((_id, 1.0 - sim))
            scored.sort(key=lambda x: x[1])
            return [{"id": i, "distance": d} for i, d in scored[:n_results]]

    class _TinyLLM:
        def embed_query(self, q): return [1.0, 0.0, 0.0] if "工作" in q else [0.0, 1.0, 0.0]
        def embed_documents(self, texts):
            out = []
            for t in texts:
                out.append([1.0, 0.0, 0.0] if "工作" in t else [0.0, 1.0, 0.0])
            return out

    fake = _CountingFakeChromaClient()
    inner = InMemoryStore()
    store = ChromaMemoryStore(inner, _TinyLLM(), chroma_client=fake,
                              collection_name="b13_upd")
    cname = store._collection_name
    r = store.add(MemoryRecord(user_id="u1", session_id="s", content="我在上海工作",
                               source="t", kind="fact", importance=0.7))
    old_add_calls = fake.add_calls
    old_del_calls = fake.delete_calls

    # 1) 内容变更：先 delete 再 add（两步走），document/embedding 全部被新内容替换。
    #    说明：InMemoryStore 持有的是同一对象引用，若外部直接在 s.add 返回的 r 上改字段
    #    再调 update(r)，update 内部 inner.get 拿到的已经是"被外部改完"的 r，content_changed
    #    会误判。这是调用方的使用姿势问题（应该构造新记录或 copy 后改）；真实业务路径
    #    （revise/retract/manager.remember）都不会这么做。测试里按正确用法：基于返回
    #    的记录 id 构造一个新的 MemoryRecord 传入，与 revise 的用法一致。
    new_r = MemoryRecord(
        memory_id=r.memory_id, user_id=r.user_id, session_id=r.session_id,
        content="我爱吃辣", source=r.source, kind=r.kind, importance=r.importance,
        created_at=r.created_at,
    )
    store.update(new_r)
    assert fake.delete_calls == old_del_calls + 1, \
        f"content 变化应先 delete 旧向量，实际 delete_calls={fake.delete_calls}, 之前={old_del_calls}"
    assert fake.add_calls == old_add_calls + 1, \
        f"content 变化应再 add 新向量，实际 add_calls={fake.add_calls}, 之前={old_add_calls}"
    row = fake.cols[cname][r.memory_id]
    assert row["document"] == "我爱吃辣", f"document 应被替换为新内容，实际 {row['document']}"
    assert row["embedding"] == [0.0, 1.0, 0.0], \
        f"embedding 应重算为新内容的向量，实际 {row['embedding']}"
    assert row["metadata"]["status"] == "active"

    # 2) 仅 status 变更：不触发删+加，embedding 保持不变，只同步 metadata
    old_add_calls2 = fake.add_calls
    old_del_calls2 = fake.delete_calls
    saved_vec = list(row["embedding"])
    cur = inner.get(r.memory_id)
    cur.status = "retracted"
    store.update(cur)
    assert fake.delete_calls == old_del_calls2, \
        "仅 status 变化不应 delete 向量"
    # _update_vec_metadata 会用 add（upsert）回写 metadata，但 embedding 不重算
    assert fake.cols[cname][r.memory_id]["embedding"] == saved_vec, \
        "仅 status 变化不应重算 embedding"
    assert fake.cols[cname][r.memory_id]["metadata"]["status"] == "retracted"

    store.close()
    print("✅ test_b13_chroma_update_content_replace_vec 通过"
          "（update content 走 delete-then-add 语义、仅 status 变化不重算 embedding）")


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
    test_v183_chroma_client_none_is_lexical_only()  # 从 test_memory.py 拆出
    test_v183_chroma_with_real_chromaclient_semantic()
    test_b13_chroma_find_similar_ann()
    test_b13_chroma_stale_cleanup_and_where_filter()
    test_b13_chroma_update_content_replace_vec()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
