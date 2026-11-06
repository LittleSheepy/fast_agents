"""database 模块 Chroma 后端（ChromaClient）单元测试。

覆盖点：

运行方式（Windows PowerShell，在项目根目录）:
    python tests/database/test_database_chroma.py
    （无需 pytest，纯标准库）
"""
try:
    import chromadb  # type: ignore
except ImportError as e:
    raise Exception(
        "使用 ChromaClient 需要先安装 chromadb 可选依赖：\n"
        "    pip install fast-agents[chroma]\n"
        "或直接：pip install chromadb>=0.4,<1.0"
    ) from e

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, importorskip

# ===========================================================================
# 向量后端：VectorClientBase 抽象层 + 编译层
# ===========================================================================

def test_vector_client_base_abstract_methods():
    """VectorClientBase 应覆盖连接/集合/写入/删除/取/检索/计数 7 类能力。"""
    from fast_agents.database import VectorClientBase
    expected = {"connect", "close", "ping",
                "collection_exists", "create_collection", "get_or_create_collection", "drop_collection",
                "add", "delete", "get", "search", "count"}
    assert expected == VectorClientBase.__abstractmethods__


def test_vector_filter_leaf_and_compile():
    """叶子条件应编译成 Chroma where 格式。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    f = VectorFilter(field="kind", op=VectorFilterOp.EQ, value="fact")
    assert _compile_vector_filter(f) == {"kind": {"$eq": "fact"}}


def test_vector_filter_and_compile():
    """AND 组合条件应编译成 $and 嵌套。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    f = VectorFilter(op=VectorFilterOp.AND, filters=[
        VectorFilter(field="user_id", op=VectorFilterOp.EQ, value="u1"),
        VectorFilter(field="score", op=VectorFilterOp.GTE, value=0.7),
    ])
    assert _compile_vector_filter(f) == {
        "$and": [
            {"user_id": {"$eq": "u1"}},
            {"score": {"$gte": 0.7}},
        ]
    }


def test_vector_filter_passthrough_dict():
    """dict 原样透传，给熟悉 Chroma where 格式的高级用户留口子。"""
    from fast_agents.database.chroma_client import _compile_vector_filter
    assert _compile_vector_filter({"x": {"$gt": 1}}) == {"x": {"$gt": 1}}


def test_vector_filter_and_needs_two_subfilters():
    """AND/OR 组合至少 2 个子条件。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    with raises(ValueError):
        _compile_vector_filter(VectorFilter(op=VectorFilterOp.AND, filters=[
            VectorFilter(field="a", op=VectorFilterOp.EQ, value=1),
        ]))


def test_chroma_client_requires_path_or_host():
    """path 和 host 二选一，都不给/都给都报错。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    with raises(ValueError):
        ChromaClient()
    with raises(ValueError):
        ChromaClient(path="/tmp/x", host="127.0.0.1")


def test_chroma_client_invalid_collection_name():
    """非法集合名（太短 / 首尾非字母数字 / 含特殊字符 / 过长）直接报错。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, os
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    db = ChromaClient(path=tmp)
    for bad in ["ab", "_abc", "a" * 64, "abc!", "a b", "abc_"]:
        with raises(ValueError):
            db.create_collection(bad)
    db.close()


def test_chroma_client_lifecycle_local_persistent():
    """本地持久化模式：建集→写→按id取→近邻搜索→按条件删→幂等建集→drop。"""
    chromadb = importorskip("chromadb")
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        assert db.ping() is True
        # 幂等建集
        db.get_or_create_collection("mem_v1", dimension=4, metric="cosine")
        db.get_or_create_collection("mem_v1", dimension=4, metric="cosine")  # 重复调用幂等
        assert db.collection_exists("mem_v1")
        assert not db.collection_exists("mem_v2")
        # 写 3 条：2 条 kind=fact、1 条 kind=preference
        db.add(
            "mem_v1",
            ids=["m1", "m2", "m3"],
            embeddings=[[1.0, 0.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]],
            documents=["猫喜欢吃小鱼干", "猫爱睡觉", "我喜欢蓝颜色"],
            metadatas=[{"kind": "fact", "score": 0.9},
                       {"kind": "fact", "score": 0.8},
                       {"kind": "preference", "score": 1.0}],
        )
        assert db.count("mem_v1") == 3
        # 按 id 精确取
        rows = db.get("mem_v1", ids=["m1"])
        assert len(rows) == 1 and rows[0]["id"] == "m1" and rows[0]["document"] == "猫喜欢吃小鱼干"
        assert "embedding" not in rows[0]  # 默认不带 embedding
        rows_with_emb = db.get("mem_v1", ids=["m1"], include_embeddings=True)
        assert "embedding" in rows_with_emb[0] and len(rows_with_emb[0]["embedding"]) == 4
        # upsert 覆盖：m2 的 document/score 覆盖，m1/m3 保留
        db.add("mem_v2_dummy" if False else "mem_v1",
               ids=["m2"],
               documents=["猫爱打盹"],
               metadatas=[{"kind": "fact", "score": 0.85}],
               embeddings=[[0.9, 0.15, 0.0, 0.0]])
        assert db.count("mem_v1") == 3
        m2 = db.get("mem_v1", ids=["m2"])[0]
        assert m2["document"] == "猫爱打盹" and m2["metadata"]["score"] == 0.85
        # 近邻检索：query ≈ [1,0,0,0] 应该先返回 m1、再 m2，m3 最远
        hits = db.search("mem_v1", [1.0, 0.0, 0.0, 0.0], n_results=2)
        assert len(hits) == 2
        assert hits[0]["id"] == "m1"
        assert hits[1]["id"] == "m2"
        assert "distance" in hits[0]
        # where 前置过滤：只在 preference 类里搜，应只返回 m3
        hits_filtered = db.search("mem_v1", [1.0, 0.0, 0.0, 0.0], n_results=5,
                                  where=VectorFilter(field="kind", op=VectorFilterOp.EQ, value="preference"))
        assert [h["id"] for h in hits_filtered] == ["m3"]
        # delete 空条件拒绝
        with raises(ValueError):
            db.delete("mem_v1")
        # 按 metadata 条件删除：删 score >= 0.9 的 → m1(0.9) 会被保留？不，m1=0.9 >=0.9 删，m3=1.0 删，剩 m2
        removed = db.delete("mem_v1", where=VectorFilter(field="score", op=VectorFilterOp.GTE, value=0.9))
        assert removed == 2
        assert db.count("mem_v1") == 1
        assert db.get("mem_v1", ids=["m2"])[0]["document"] == "猫爱打盹"
        # 按 id 删除
        db.delete("mem_v1", ids=["m2"])
        assert db.count("mem_v1") == 0
        # drop 幂等
        db.drop_collection("mem_v1")
        db.drop_collection("mem_v1")  # 不存在静默
        assert not db.collection_exists("mem_v1")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_add_requires_consistent_lengths():
    """embeddings/documents/metadatas 长度必须与 ids 一致。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        with raises(ValueError):
            db.add("tt1", ids=["a"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        with raises(ValueError):
            db.add("tt1", ids=["a"])  # 内容字段全空
        with raises(ValueError):
            db.add("tt1", ids=[])   # ids 空
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_search_rejects_bad_query_vector():
    """query_embedding 必须是非空向量。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        db.add("tt1", ids=["a"], embeddings=[[1.0, 0.0]])
        with raises(ValueError):
            db.search("tt1", [])
        with raises(ValueError):
            db.search("tt1", [1.0, 0.0], n_results=0)
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_count_with_where():
    """带 where 的 count 走 get include=[] 计数。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        db.add("tt1", ids=["a", "b", "c"],
               embeddings=[[1,0],[0,1],[1,1]],
               metadatas=[{"v":1},{"v":2},{"v":3}])
        assert db.count("tt1") == 3
        assert db.count("tt1", where=VectorFilter(field="v", op=VectorFilterOp.GTE, value=2)) == 2
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_invalid_metric_rejected():
    """metric 只能是 cosine / ip / l2。"""
    importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        with raises(ValueError):
            db.create_collection("tt1", metric="euclidean")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ===========================================================================
# 向量后端：v1.1.1 优化项
# ===========================================================================

def test_vector_client_base_get_signature_has_include_switches():
    """抽象基类 get 暴露 include_documents / include_metadatas 可选开关（默认 True）。"""
    import inspect
    from fast_agents.database import VectorClientBase
    params = inspect.signature(VectorClientBase.get).parameters
    assert params["include_documents"].default is True
    assert params["include_metadatas"].default is True


def test_chroma_create_collection_persists_dimension_and_rejects_duplicate():
    """建集把 dimension 写进 collection metadata；重复建集抛「已存在」。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt1", dimension=4, metric="cosine")
        # 维度写进了 metadata，可跨进程读回
        assert db._client.get_collection("opt1").metadata.get("dimension") == 4
        with raises(ValueError, match="已存在"):
            db.create_collection("opt1")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_or_create_is_idempotent_and_keeps_dimension():
    """get_or_create 走原生幂等接口：二次调用不报错，维度仍记录。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.get_or_create_collection("opt2", dimension=3)
        db.get_or_create_collection("opt2", dimension=3)  # 幂等
        assert db._dimensions.get("opt2") == 3
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_duplicate_ids():
    """同一批 ids 重复应提前报错（Chroma 内部会抛 DuplicateIDError）。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt3")
        with raises(ValueError, match="重复"):
            db.add("opt3", ids=["a", "a"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_inconsistent_batch_dimension():
    """同一批 embeddings 维度不一致应报错。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt4")
        with raises(ValueError, match="维度不一致"):
            db.add("opt4", ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0, 2.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_dimension_mismatch_with_declared():
    """写入维度与建集声明的维度不一致应报错（Chroma 空集合不校验，靠客户端补位）。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt5", dimension=3)
        with raises(ValueError, match="声明的维度"):
            db.add("opt5", ids=["a"], embeddings=[[1.0, 0.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_search_rejects_dimension_mismatch():
    """检索向量维度与声明维度不一致应报错。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt6", dimension=3)
        with raises(ValueError, match="维度"):
            db.search("opt6", [1.0, 0.0])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_chunks_large_batch():
    """add_chunk_size 生效：分块写入仍全部落库、返回总数。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp, add_chunk_size=2)
        db.create_collection("opt7", dimension=2)
        ids = [f"k{i}" for i in range(5)]
        embs = [[float(i), float(i + 1)] for i in range(5)]
        assert db.add("opt7", ids=ids, embeddings=embs) == 5
        assert db.count("opt7") == 5
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_limit_validation_and_include_switches():
    """get 校验 limit 为正整数；include_documents / include_metadatas 关闭后不收对应字段。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt8", dimension=2)
        db.add("opt8", ids=["a"], embeddings=[[1.0, 0.0]], documents=["doc"], metadatas=[{"v": 1}])
        with raises(ValueError):
            db.get("opt8", limit=0)
        with raises(ValueError):
            db.get("opt8", limit=-3)
        rows = db.get("opt8", include_documents=False, include_metadatas=False)
        assert rows[0]["id"] == "a"
        assert "document" not in rows[0] and "metadata" not in rows[0]
        full = db.get("opt8")
        assert full[0]["document"] == "doc" and full[0]["metadata"]["v"] == 1
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_delete_ids_take_precedence_and_exact_count():
    """delete 返回精确条数；ids 与 where 同时传时以 ids 为准。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt9", dimension=2)
        db.add(
            "opt9",
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            metadatas=[{"v": 1}, {"v": 2}, {"v": 3}],
        )
        # ids 含不存在的 id：返回精确删除条数
        assert db.delete("opt9", ids=["a", "nope"]) == 1
        # 同时传 ids 与 where：语义为按 ids 删（不受 where 影响）
        removed = db.delete(
            "opt9",
            ids=["c"],
            where=VectorFilter(field="v", op=VectorFilterOp.EQ, value=999),
        )
        assert removed == 1
        assert db.count("opt9") == 1
        assert db.get("opt9")[0]["id"] == "b"
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_or_create_tolerates_existing_collection_without_metadata():
    """对已存在但 metadata 为空的集合调用 get_or_create 不应因 Chroma 0.6.x 的合并缺陷而崩。"""
    importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        # 用裸 SDK 建一个没有 metadata 的集合
        db.connect()
        db._client.create_collection("raw1")
        # 再走我们的 get_or_create（带 metadata + dimension），应能从容取回而不是 TypeError
        db.get_or_create_collection("raw1", dimension=2)
        assert db._dimensions.get("raw1") == 2
        db.add("raw1", ids=["a"], embeddings=[[1.0, 0.0]])
        assert db.count("raw1") == 1
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
    test_chroma_add_chunks_large_batch()
    pass

    # AI新添加的测试(AI可以删除覆盖)
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
