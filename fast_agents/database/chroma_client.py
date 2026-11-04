"""Chroma 向量数据库客户端实现。

``ChromaClient`` 实现 ``VectorClientBase``，是 database 模块的第一个向量后端：
- 同时支持**本地持久化目录**（单机开发）和**远端 Chroma Server**（生产部署）；
- 可选依赖：核心依赖里**不包含** chromadb，使用前需 ``pip install fast-agents[chroma]``
  （等价于 ``pip install chromadb>=0.4,<1.0``）；未安装时构造 ChromaClient 会抛清晰的
  ImportError 提示；
- 与 SQLiteClient 风格对齐：集合名白名单校验、空条件删除拦截、所有对外返回统一
  为 ``list[dict]``（不向外泄露 Chroma SDK 的 dict-of-lists 原始结构）。

使用示例见 ``examples/database/chroma_usage.py``。

::: module-author fast_agents
"""
from __future__ import annotations

import os
import re
from typing import Any

from .vector_client_base import VectorClientBase
from .vector_query import VectorFilter, VectorFilterOp, VectorWhere


# ---------------------------------------------------------------------------
# 标识符校验：对齐 SQLiteClient._check_ident 的风格，但适配 Chroma 的集合名规则
# ---------------------------------------------------------------------------
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")
# Chroma SDK 真实约束：3-63 字符，首尾必须是字母或数字，中间允许字母/数字/下划线/连字符；
# 单字符名（长度=1 或 2）和以下划线/连字符结尾的名都不接受。


def _check_collection_name(name: str, label: str = "集合名") -> None:
    """校验 Chroma collection 名是否合法（与 Chroma SDK 自身约束对齐）。"""
    if not isinstance(name, str):
        raise TypeError(f"{label}必须是字符串，收到 {type(name)}")
    if not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            f"非法{label}: {name!r}；需 3-63 字符、字母开头，只允许字母/数字/下划线/连字符"
        )


# ---------------------------------------------------------------------------
# VectorFilter -> Chroma where dict 编译器
# ---------------------------------------------------------------------------
def _compile_vector_filter(f: VectorFilter | dict[str, Any]) -> dict[str, Any]:
    """把 VectorFilter（或已构造好的 dict）编译成 Chroma 接受的 where 字典。

    Chroma where 格式：
        叶子：{"field": {"$op": value}}
        组合：{"$and": [{...}, {...}]}
              {"$or":  [{...}, {...}]}
    """
    # 已经是 dict 就原样透传（给熟悉 Chroma where 格式的高级用户留口子）
    if isinstance(f, dict):
        return f
    if not isinstance(f, VectorFilter):
        raise TypeError(f"过滤条件需为 VectorFilter 或 dict，收到 {type(f)}")

    if f.op in (VectorFilterOp.AND, VectorFilterOp.OR):
        if not f.filters or len(f.filters) < 2:
            raise ValueError(f"{f.op.value} 组合至少需要 2 个子条件")
        return {f.op.value: [_compile_vector_filter(sub) for sub in f.filters]}

    # 叶子条件
    if not f.field:
        raise ValueError(f"叶子条件 {f.op.value} 需要 field")
    if f.value is None and f.op not in (VectorFilterOp.EQ, VectorFilterOp.NE):
        raise ValueError(f"{f.op.value} 条件需要 value")
    return {f.field: {f.op.value: f.value}}


def _compile_where(where: VectorWhere) -> dict[str, Any] | None:
    if where is None:
        return None
    return _compile_vector_filter(where)


# ---------------------------------------------------------------------------
# ChromaClient
# ---------------------------------------------------------------------------
_VALID_METRICS = {"cosine", "ip", "l2"}


class ChromaClient(VectorClientBase):
    """Chroma 向量后端实现。

    两种连接模式（二选一）：
    1. 本地持久化：传 ``path`` → ``chromadb.PersistentClient(path=path)``
       所有向量 / 元数据落盘到该目录，重启可恢复；适合单机 / 开发。
    2. 远端服务：传 ``host`` (+ 可选 ``port``) → ``chromadb.HttpClient(host=host, port=port)``
       连接独立启动的 Chroma Server；适合生产 / 服务端部署。

    Args:
        path: 本地持久化目录；与 host 互斥。
        host: Chroma Server 主机；与 path 互斥。
        port: Chroma Server 端口，默认 8000。
        settings: chromadb.Settings 实例（可选，高级用法）。

    Raises:
        ImportError: 未安装 chromadb 依赖时。
        ValueError: path 与 host 同时传或都不传、集合名非法、空条件删除等。
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        host: str | None = None,
        port: int = 8000,
        settings: Any = None,
    ):
        # 延迟导入：核心依赖里不包含 chromadb
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise ImportError(
                "使用 ChromaClient 需要先安装 chromadb 可选依赖：\n"
                "    pip install fast-agents[chroma]\n"
                "或直接：pip install chromadb>=0.4,<1.0"
            ) from e

        if (path is None) == (host is None):
            raise ValueError("ChromaClient 必须在 path（本地）与 host（远端）中二选一")

        self._path = path
        self._host = host
        self._port = port
        self._settings = settings
        self._chromadb = chromadb
        self._client: Any = None
        # collection 名 -> Collection 实例的本地缓存（Chroma SDK 每次 get_collection 都会
        # 发起一次 RPC，缓存一下减少开销；drop 时清掉）
        self._collections: dict[str, Any] = {}

    # ---------- 连接 ----------
    def connect(self) -> None:
        if self._client is not None:
            return
        if self._path is not None:
            os.makedirs(self._path, exist_ok=True)
            self._client = self._chromadb.PersistentClient(path=self._path, settings=self._settings)
        else:
            self._client = self._chromadb.HttpClient(
                host=self._host, port=self._port, settings=self._settings
            )

    def close(self) -> None:
        # Chroma Python SDK 目前没有显式 close；清缓存即可
        self._collections.clear()
        self._client = None

    def ping(self) -> bool:
        self.connect()
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    # ---------- 内部工具 ----------
    def _get_collection(self, name: str):
        _check_collection_name(name)
        if name in self._collections:
            return self._collections[name]
        self.connect()
        try:
            col = self._client.get_collection(name)
        except Exception as e:
            # Chroma 不同版本抛的异常类型不一致，统一用 ValueError
            raise ValueError(f"集合 {name!r} 不存在") from e
        self._collections[name] = col
        return col

    def _metric_to_metadata(self, metric: str) -> dict[str, str]:
        if metric not in _VALID_METRICS:
            raise ValueError(f"非法 metric: {metric!r}；只支持 {sorted(_VALID_METRICS)}")
        return {"hnsw:space": metric}

    # ---------- 集合管理 ----------
    def collection_exists(self, name: str) -> bool:
        _check_collection_name(name)
        self.connect()
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    def create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        _check_collection_name(name)
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension 必须是正整数，收到 {dimension}")
        self.connect()
        metadata = self._metric_to_metadata(metric)
        # Chroma 0.4.x 不强制声明 dimension（会从首条向量推断）；新版本支持 metadata 里加维度
        if dimension is not None:
            # hnsw 不强制要求维度，但显式写上有助于后端做维度校验
            pass
        try:
            col = self._client.create_collection(name, metadata=metadata)
        except Exception as e:
            raise ValueError(f"创建集合 {name!r} 失败（可能已存在）：{e}") from e
        self._collections[name] = col

    def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        _check_collection_name(name)
        if self.collection_exists(name):
            return
        try:
            self.create_collection(name, dimension=dimension, metric=metric)
        except ValueError as e:
            # 并发竞态：exists 检查后到 create 之间被另一个线程创建了，静默
            if "已存在" in str(e) or "already exists" in str(e).lower():
                self._collections.pop(name, None)
                return
            raise

    def drop_collection(self, name: str) -> None:
        _check_collection_name(name)
        self.connect()
        try:
            self._client.delete_collection(name)
        except Exception:
            # 不存在就静默，对齐 SQLiteClient.drop_table 的行为
            pass
        self._collections.pop(name, None)

    # ---------- 写入 ----------
    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not ids:
            raise ValueError("add 的 ids 不能为空")
        if embeddings is None and documents is None and metadatas is None:
            raise ValueError("add 的 embeddings / documents / metadatas 至少要有一个")
        for name, lst in (("embeddings", embeddings), ("documents", documents), ("metadatas", metadatas)):
            if lst is not None and len(lst) != len(ids):
                raise ValueError(f"{name} 长度({len(lst)}) 与 ids 长度({len(ids)}) 不一致")
        col = self._get_collection(collection)
        # Chroma 的 upsert 是按 id 覆盖，和基类约定的 upsert 语义一致
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(ids)

    # ---------- 删除 ----------
    def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        where: VectorWhere = None,
    ) -> int:
        if not ids and where is None:
            # 安全约定：禁止空条件删全量
            raise ValueError("delete 的 ids 和 where 不能同时为空（防止误删全量）")
        col = self._get_collection(collection)
        where_dict = _compile_where(where)

        # Chroma SDK delete 不直接返回影响条数；为了给上层 rowcount，先 count 一次再删
        # 注意：并发下不精确，但作为近似计数足够；上层用不到严格事务语义
        before = col.count()
        col.delete(ids=ids, where=where_dict)
        after = col.count()
        return max(0, before - after)

    # ---------- 取 ----------
    def get(
        self,
        collection: str,
        ids: list[str] | None = None,
        where: VectorWhere = None,
        limit: int | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        col = self._get_collection(collection)
        where_dict = _compile_where(where)
        includes = ["metadatas", "documents"]
        if include_embeddings:
            includes.append("embeddings")
        kwargs: dict[str, Any] = {"include": includes}
        if ids:
            kwargs["ids"] = ids
        if where_dict:
            kwargs["where"] = where_dict
        if limit is not None:
            kwargs["limit"] = limit
        res = col.get(**kwargs)
        return self._flatten_get_result(res, include_embeddings=include_embeddings, with_distance=False)

    # ---------- 近邻检索 ----------
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        n_results: int = 10,
        where: VectorWhere = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(query_embedding, (list, tuple)) or len(query_embedding) == 0:
            raise ValueError("query_embedding 必须是非空向量 list[float]")
        if n_results <= 0:
            raise ValueError(f"n_results 必须是正整数，收到 {n_results}")
        col = self._get_collection(collection)
        where_dict = _compile_where(where)
        includes = ["metadatas", "documents", "distances"]
        if include_embeddings:
            includes.append("embeddings")
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_dict,
            include=includes,
        )
        # query 返回的是「按 query 分组」的 dict-of-lists；本方法只支持单向量查询，取第一组
        return self._flatten_query_result(res, include_embeddings=include_embeddings)

    # ---------- 计数 ----------
    def count(self, collection: str, where: VectorWhere = None) -> int:
        col = self._get_collection(collection)
        if where is None:
            return col.count()
        # Chroma 没有带 where 的 count，走 get include=[] 取 ids 计数（结果不在内存展开向量）
        res = col.get(where=_compile_where(where), include=[])
        return len(res.get("ids") or [])

    # ---------- 结果格式化（统一摊平为 list[dict]，不向外泄露 Chroma 原始结构） ----------
    @staticmethod
    def _flatten_get_result(res: dict, *, include_embeddings: bool, with_distance: bool) -> list[dict]:
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or [None] * len(ids)
        documents = res.get("documents") or [None] * len(ids)
        embeddings = res.get("embeddings") if include_embeddings else None
        out: list[dict] = []
        for i, rid in enumerate(ids):
            item: dict[str, Any] = {"id": rid}
            if documents[i] is not None:
                item["document"] = documents[i]
            if metadatas[i] is not None:
                item["metadata"] = metadatas[i]
            if embeddings is not None and embeddings[i] is not None:
                item["embedding"] = embeddings[i]
            if with_distance:
                # get 结果不带 distance
                pass
            out.append(item)
        return out

    @staticmethod
    def _flatten_query_result(res: dict, *, include_embeddings: bool) -> list[dict]:
        # col.query 针对单 query 返回：ids=[[...]], distances=[[...]], metadatas=[[...]], documents=[[...]]
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        embeddings = (res.get("embeddings") or [[]])[0] if include_embeddings else None
        out: list[dict] = []
        for i, rid in enumerate(ids):
            item: dict[str, Any] = {"id": rid}
            if i < len(documents) and documents[i] is not None:
                item["document"] = documents[i]
            if i < len(metadatas) and metadatas[i] is not None:
                item["metadata"] = metadatas[i]
            if i < len(distances) and distances[i] is not None:
                item["distance"] = distances[i]
            if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                item["embedding"] = embeddings[i]
            out.append(item)
        return out


__all__ = ["ChromaClient"]
