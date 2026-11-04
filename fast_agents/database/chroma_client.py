"""Chroma 向量数据库客户端实现。

``ChromaClient`` 实现 ``VectorClientBase``，是 database 模块的第一个向量后端：
- 同时支持**本地持久化目录**（单机开发）和**远端 Chroma Server**（生产部署）；
- 可选依赖：核心依赖里**不包含** chromadb，使用前需 ``pip install fast-agents[chroma]``
  （等价于 ``pip install chromadb>=0.4,<1.0``）；未安装时构造 ChromaClient 会抛清晰的
  ImportError 提示；
- 与 SQLiteClient 风格对齐：集合名白名单校验、空条件删除拦截、所有对外返回统一
  为 ``list[dict]``（不向外泄露 Chroma SDK 的 dict-of-lists 原始结构）。

v1.1.1 相对 v1.1.0 的优化点（详见 docs/database模块编写说明.md 的更新历史 v1.1.1 与 §7.8）：

1. **少一次往返**：``get_or_create_collection`` 直接走 SDK 原生
   ``client.get_or_create_collection``（原来「先 collection_exists 再 create」两次 RPC）；
   ``collection_exists`` 命中本地 collection 缓存时直接返回，不再每次发请求。
2. **delete 精确计数且少一次全量扫描**：改用 ``get(include=[])`` 拿到目标 id 数再删除，
   不再「删前 count 一次、删后 count 一次」两次全量统计；并把 ``ids`` / ``where`` 语义
   固定为「传了 ids 就按 id 删、否则按 where 删」，不依赖 Chroma 未文档化的 ids+where 混用行为。
3. **dimension 真正生效**：Chroma 本身不强制向量维度（空集合写入任意维度都会成功），
   本实现在建集时把维度写进集合 metadata（键 ``dimension``，可跨进程持久化），
   ``add`` / ``search`` 时做客户端侧维度校验；对未声明维度的集合，首次写入会推断并记住。
4. **大批量写入自动分块**：``add`` 超过 ``add_chunk_size``（默认 1000）时自动分批 upsert，
   避免一次性提交撞上 Chroma/SQLite 的单批上限而整体失败。
5. **默认关闭匿名遥测**：未显式传 ``settings`` 时用 ``Settings(anonymized_telemetry=False)``，
   不向外部发送使用统计。
6. **远端参数更完整**：``ssl`` / ``headers`` / ``tenant`` / ``database`` 均可透传给 HttpClient。
7. **健壮性**：``add`` 校验 ids 去重与批内维度一致、``get`` 校验 ``limit`` 为正整数、
   ``connect`` 加锁避免多线程重复建连接、清理结果摊平函数里的死参数。

使用示例见 ``examples/database/chroma_usage.py``。

::: module-author fast_agents
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Sequence

from .vector_client_base import VectorClientBase
from .vector_query import VectorFilter, VectorFilterOp, VectorWhere


# ---------------------------------------------------------------------------
# 标识符校验：对齐 SQLiteClient._check_ident 的风格，但适配 Chroma 的集合名规则
# ---------------------------------------------------------------------------
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")
# Chroma SDK 真实约束：3-63 字符，首尾必须是字母或数字，中间允许字母/数字/下划线/连字符；
# 单字符名（长度=1 或 2）和以下划线/连字符结尾的名都不接受。

# 集合级 metadata 里记录向量维度的键名（不带 hnsw: 前缀，属于自定义键，Chroma 会原样保留）
_DIMENSION_META_KEY = "dimension"

# add 单批 upsert 的默认上限：Chroma 底层 SQLite 存在变量数上限，一次提交过大会整体失败
_DEFAULT_ADD_CHUNK_SIZE = 1000

_VALID_METRICS = {"cosine", "ip", "l2"}


def _check_collection_name(name: str, label: str = "集合名") -> None:
    """校验 Chroma collection 名是否合法（与 Chroma SDK 自身约束对齐）。"""
    if not isinstance(name, str):
        raise TypeError(f"{label}必须是字符串，收到 {type(name)}")
    if not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            f"非法{label}: {name!r}；需 3-63 字符、字母开头，只允许字母/数字/下划线/连字符"
        )


def _check_dimension(dimension: int | None) -> None:
    """dimension 允许为 None（交给后端推断），否则必须是正整数。"""
    if dimension is None:
        return
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError(f"dimension 必须是正整数，收到 {dimension!r}")


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
        ssl: 远端模式是否走 HTTPS，默认 False。
        headers: 远端模式附加请求头（如鉴权 token）。
        tenant: 远端模式的 tenant 名。
        database: 远端模式的 database 名。
        settings: ``chromadb.Settings`` 实例（可选，高级用法）；不传时默认关闭匿名遥测。
        add_chunk_size: ``add`` 单批 upsert 的条数上限，默认 1000；<=0 表示不分块。

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
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        tenant: str | None = None,
        database: str | None = None,
        settings: Any = None,
        add_chunk_size: int = _DEFAULT_ADD_CHUNK_SIZE,
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
        self._ssl = ssl
        self._headers = headers
        self._tenant = tenant
        self._database = database
        self._settings = settings
        self._add_chunk_size = add_chunk_size
        self._chromadb = chromadb
        self._client: Any = None
        # collection 名 -> Collection 实例的本地缓存（Chroma SDK 每次 get_collection 都会
        # 发起一次 RPC，缓存一下减少开销；drop 时清掉）
        self._collections: dict[str, Any] = {}
        # collection 名 -> 已声明的向量维度（来自建集 metadata 或首次写入推断）
        self._dimensions: dict[str, int] = {}
        # 连接只建一次：多线程并发首次调用时用锁兜住，避免重复建 client
        self._lock = threading.Lock()

    # ---------- 连接 ----------
    def _effective_settings(self) -> Any:
        """未显式传 settings 时，默认关闭匿名遥测（不向外部发送使用统计）。"""
        if self._settings is not None:
            return self._settings
        try:
            from chromadb.config import Settings

            return Settings(anonymized_telemetry=False)
        except Exception:  # pragma: no cover - 极端版本差异时退回 SDK 默认
            return None

    def connect(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:  # 双检锁
                return
            if self._path is not None:
                os.makedirs(self._path, exist_ok=True)
                self._client = self._chromadb.PersistentClient(
                    path=self._path, settings=self._effective_settings()
                )
            else:
                kwargs: dict[str, Any] = {
                    "host": self._host,
                    "port": self._port,
                    "settings": self._effective_settings(),
                }
                if self._ssl:
                    kwargs["ssl"] = True
                if self._headers is not None:
                    kwargs["headers"] = self._headers
                if self._tenant is not None:
                    kwargs["tenant"] = self._tenant
                if self._database is not None:
                    kwargs["database"] = self._database
                self._client = self._chromadb.HttpClient(**kwargs)

    def close(self) -> None:
        # Chroma Python SDK 目前没有显式 close；清缓存 + 释放引用即可
        self._collections.clear()
        self._dimensions.clear()
        self._client = None

    def ping(self) -> bool:
        self.connect()
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    # ---------- 内部工具 ----------
    def _remember_dimension(self, name: str, col: Any) -> None:
        """从集合 metadata 里读回已声明维度，供后续写入/检索校验。"""
        metadata = getattr(col, "metadata", None)
        if isinstance(metadata, dict):
            dim = metadata.get(_DIMENSION_META_KEY)
            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                self._dimensions[name] = dim

    def _cache_collection(self, name: str, col: Any) -> Any:
        self._collections[name] = col
        self._remember_dimension(name, col)
        return col

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
        return self._cache_collection(name, col)

    def _metric_to_metadata(self, metric: str) -> dict[str, Any]:
        if metric not in _VALID_METRICS:
            raise ValueError(f"非法 metric: {metric!r}；只支持 {sorted(_VALID_METRICS)}")
        return {"hnsw:space": metric}

    # ---------- 集合管理 ----------
    def collection_exists(self, name: str) -> bool:
        _check_collection_name(name)
        # 缓存命中即认为存在，省掉一次 get_collection 往返
        if name in self._collections:
            return True
        self.connect()
        try:
            col = self._client.get_collection(name)
        except Exception:
            return False
        self._cache_collection(name, col)
        return True

    def create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        _check_collection_name(name)
        _check_dimension(dimension)
        self.connect()
        metadata = self._metric_to_metadata(metric)
        # 把向量维度写进集合 metadata：Chroma 自身不强制维度，靠这个键做到跨进程校验
        if dimension is not None:
            metadata[_DIMENSION_META_KEY] = dimension
        try:
            col = self._client.create_collection(name, metadata=metadata)
        except Exception as e:
            text = f"{type(e).__name__} {e}".lower()
            if "uniqueconstraint" in text or "already exists" in text:
                raise ValueError(
                    f"集合 {name!r} 已存在（如需幂等请用 get_or_create_collection）"
                ) from e
            raise ValueError(f"创建集合 {name!r} 失败：{e}") from e
        self._cache_collection(name, col)
        if dimension is not None:
            self._dimensions[name] = dimension

    def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        _check_collection_name(name)
        _check_dimension(dimension)
        self.connect()
        metadata = self._metric_to_metadata(metric)
        if dimension is not None:
            metadata[_DIMENSION_META_KEY] = dimension
        # 走 SDK 原生幂等接口：一次往返完成「有则取、无则建」，替代 exists+create 两次 RPC。
        # 注意：Chroma 0.6.x 在「集合已存在、但其 metadata 为 None」时合并 metadata 会抛
        # TypeError（'NoneType' object is not iterable），这里退回直接 get_collection。
        try:
            col = self._client.get_or_create_collection(name, metadata=metadata)
        except TypeError:
            col = self._client.get_collection(name)
        self._cache_collection(name, col)
        if dimension is not None and name not in self._dimensions:
            self._dimensions[name] = dimension

    def drop_collection(self, name: str) -> None:
        _check_collection_name(name)
        self.connect()
        try:
            self._client.delete_collection(name)
        except Exception:
            # 不存在就静默，对齐 SQLiteClient.drop_table 的行为
            pass
        self._collections.pop(name, None)
        self._dimensions.pop(name, None)

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
        if len(set(ids)) != len(ids):
            # Chroma 内部会抛 DuplicateIDError，这里提前拦下并给一致的中文错误
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"add 的 ids 存在重复：{dup}")
        n = len(ids)
        for field, lst in (
            ("embeddings", embeddings),
            ("documents", documents),
            ("metadatas", metadatas),
        ):
            if lst is not None and len(lst) != n:
                raise ValueError(f"{field} 长度({len(lst)}) 与 ids 长度({n}) 不一致")

        # 先拿到集合（顺带从 metadata 读回已声明维度）
        col = self._get_collection(collection)

        if embeddings is not None:
            dims = {len(e) for e in embeddings}
            if len(dims) > 1:
                raise ValueError(f"同一批 embeddings 维度不一致：{sorted(dims)}")
            dim = next(iter(dims))
            declared = self._dimensions.get(collection)
            if declared is None:
                # 未声明维度的集合：以首次写入的维度为准并记住
                self._dimensions[collection] = dim
            elif dim != declared:
                raise ValueError(
                    f"embedding 维度({dim}) 与集合 {collection!r} 声明的维度({declared}) 不一致"
                )

        # Chroma 的 upsert 是按 id 覆盖，和基类约定的 upsert 语义一致
        chunk = int(self._add_chunk_size) if self._add_chunk_size else 0
        if chunk <= 0 or n <= chunk:
            self._upsert(col, collection, ids, embeddings, documents, metadatas)
        else:
            # 大批量自动分块，避免单批提交过大整体失败
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                self._upsert(
                    col,
                    collection,
                    ids[sl],
                    embeddings[sl] if embeddings is not None else None,
                    documents[sl] if documents is not None else None,
                    metadatas[sl] if metadatas is not None else None,
                )
        return n

    @staticmethod
    def _upsert(col, collection, ids, embeddings, documents, metadatas) -> None:
        cols = list(ids)
        payload_embeddings: Sequence[Sequence[float]] | None = (
            [list(e) for e in embeddings] if embeddings is not None else None
        )
        payload_docs: Sequence[str] | None = list(documents) if documents is not None else None
        payload_metas: Sequence[dict[str, Any]] | None = (
            [dict(m) for m in metadatas] if metadatas is not None else None
        )
        try:
            col.upsert(
                ids=cols,
                embeddings=payload_embeddings,
                documents=payload_docs,
                metadatas=payload_metas,
            )
        except Exception as e:
            # 兜底把维度类错误翻译成 clearer 的中文提示（Chroma 空集合不校验维度，
            # 但集合内已有向量时维度不符会抛 InvalidDimensionException）
            if "dimension" in str(e).lower():
                raise ValueError(f"写入集合 {collection!r} 失败：{e}") from e
            raise

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

        # 语义固定：传了 ids 就按 id 删；否则按 where 删（不依赖 Chroma 未文档化的混用行为）
        if ids:
            targets = list(dict.fromkeys(ids))  # 去重且保持顺序
            found = col.get(ids=targets, include=[]) or {}
            removed = len(found.get("ids") or [])
            col.delete(ids=targets)
            return removed
        where_dict = _compile_where(where)
        found = col.get(where=where_dict, include=[]) or {}
        removed = len(found.get("ids") or [])
        col.delete(where=where_dict)
        return removed

    # ---------- 取 ----------
    def get(
        self,
        collection: str,
        ids: list[str] | None = None,
        where: VectorWhere = None,
        limit: int | None = None,
        include_embeddings: bool = False,
        include_documents: bool = True,
        include_metadatas: bool = True,
    ) -> list[dict[str, Any]]:
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError(f"limit 必须是正整数，收到 {limit!r}")
        col = self._get_collection(collection)
        where_dict = _compile_where(where)
        includes: list[str] = []
        if include_metadatas:
            includes.append("metadatas")
        if include_documents:
            includes.append("documents")
        if include_embeddings:
            includes.append("embeddings")
        kwargs: dict[str, Any] = {"include": includes}
        if ids:
            kwargs["ids"] = list(ids)
        if where_dict:
            kwargs["where"] = where_dict
        if limit is not None:
            kwargs["limit"] = limit
        res = col.get(**kwargs)
        return self._flatten_get_result(
            res,
            include_embeddings=include_embeddings,
            include_documents=include_documents,
            include_metadatas=include_metadatas,
        )

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
        if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results <= 0:
            raise ValueError(f"n_results 必须是正整数，收到 {n_results!r}")
        col = self._get_collection(collection)
        declared = self._dimensions.get(collection)
        if declared is not None and len(query_embedding) != declared:
            raise ValueError(
                f"query_embedding 维度({len(query_embedding)}) 与集合 {collection!r} "
                f"声明的维度({declared}) 不一致"
            )
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
    def _flatten_get_result(
        res: dict,
        *,
        include_embeddings: bool,
        include_documents: bool,
        include_metadatas: bool,
    ) -> list[dict]:
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") if include_metadatas else None
        documents = res.get("documents") if include_documents else None
        embeddings = res.get("embeddings") if include_embeddings else None
        out: list[dict] = []
        for i, rid in enumerate(ids):
            item: dict[str, Any] = {"id": rid}
            if documents is not None and i < len(documents) and documents[i] is not None:
                item["document"] = documents[i]
            if metadatas is not None and i < len(metadatas) and metadatas[i] is not None:
                item["metadata"] = metadatas[i]
            if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                item["embedding"] = embeddings[i]
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
