"""语义检索后端（B2 / v1.4.0）
fast_agents/fast_agents/memory/semantic_store.py
包装任意 MemoryStoreBase 实例，在原有词法检索上叠加 embedding 语义通道：
- 默认 hybrid_weight=0.7（偏语义，用户确认过）；
- 向量存储策略：inner 是 SQLiteMemoryStore 时，同库独立表 memory_embeddings（BLOB，
  建表 / 读写复用 inner 的统一 SQLiteClient + 声明式 Schema）；inner 是 InMemoryStore
  或其他自定义后端时，用内存 dict 存；
- 向量侧同时缓存 kind / importance / status 三项 meta，字段口径与 ChromaMemoryStore
  的 metadata 一致，可按 vector_meta() 单独取用；
- 单条向量用 MemoryVector Pydantic 模型封装，内存与 SQLite 两侧字段一一对齐，
  避免维护「vecs dict + meta dict」两个并行结构带来的同步 bug；
- embedding 调用失败一律降级（warning + 照常 CRUD），不阻塞主流程；
- 不修改 memories 主表 schema，不污染既有数据；
- B5 起额外提供 semantic_vectors()（批量向量）与 touch()（记账代理），
  让「记忆合并 / 命中记账」也能用上语义通道。
- B13 第二步：stale（superseded/retracted/archived）向量不删除，
  仅在向量表 / 内存缓存里同步 status；_load_vec/_load_vecs_batch 统一按 status='active'
  过滤，让 search / find_similar / merge_similar 不命中 stale 条目；设计口径
  与 ChromaMemoryStore 的 where={"status":"active"} + _update_vec_metadata 完全对称。
  （expired 只是时效到期、status 仍为 active，不等于删除：检索侧按 is_expired() 过滤，
  写入前查重用 include_expired=True 单独纳入。）

使用方式：
    from fast_agents.llm import LLM
    from fast_agents.memory import SQLiteMemoryStore, SemanticMemoryStore, MemoryManager

    llm = LLM()
    inner = SQLiteMemoryStore("memories.db", user_id="u1")
    store = SemanticMemoryStore(inner, llm)   # 默认 hybrid_weight=0.7
    manager = MemoryManager(llm=llm, long_term_store=store)
"""

from __future__ import annotations

import logging
import struct
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..database import (
    Column,
    ColumnType,
    Filter,
    FilterOp,
    Index,
    Query,
    Schema,
)
from .memory_store_base import MemoryStoreBase
from .models import MemoryRecord, MemoryVector
from .scoring import cosine_similarity, rank_records

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)

# BLOB 序列化格式：little-endian float32
# 用 struct.pack("<{n}f", *vec) 拼接，1536 维 * 4 字节 ≈ 6KB/条
_VEC_ENDIAN = "<"
_VEC_CHAR = "f"


class SemanticMemoryStore(MemoryStoreBase):
    """给任意 MemoryStoreBase 包上 embedding 语义检索的包装器

    设计要点：
    1. 继承 MemoryStoreBase，和 InMemoryStore / SQLiteMemoryStore 平级，
       MemoryManager 无感知；
    2. 不继承 SQLiteMemoryStore，不把「语义检索」和「SQLite CRUD」强耦合——
       未来想接 FAISS / Chroma / Milvus，换包装器即可；
    3. 向量存独立表/独立 dict，不污染 memories 主表 schema；
    4. embedding 失败降级为纯词法，记忆照常写入、检索照常工作；
    5. 非偏好记忆：词法零命中且语义分 < _SEMANTIC_FLOOR 时丢，避免
       「风马牛不相及但 embedding 模型噪声给了 0.2 分」的假阳性。
    """

    _DEFAULT_HYBRID_WEIGHT = 0.7     # 语义 0.7 / 词法 0.3（用户确认）
    _SEMANTIC_FLOOR = 0.3            # 非偏好类的最低语义门槛

    # 声明式表结构，供 ensure_schema(..., migrate=True) 使用；
    # 主键用单列 memory_id（uuid 全局唯一，不会跨作用域撞），对齐 memories 表风格；
    # 作用域隔离靠 idx_vec_scope(user_id, session_id) 索引 + 代码层 WHERE 过滤。
    _VEC_SCHEMA = Schema(
        table_name="memory_embeddings",
        columns=[
            Column(name="memory_id",   type=ColumnType.TEXT, primary_key=True),
            Column(name="user_id",     type=ColumnType.TEXT, nullable=False),
            Column(name="session_id",  type=ColumnType.TEXT, nullable=False),
            Column(name="embedding",   type=ColumnType.BLOB, nullable=False),
            Column(name="dim",         type=ColumnType.INTEGER, nullable=False),
            Column(name="model",       type=ColumnType.TEXT, nullable=False),
            Column(name="kind",        type=ColumnType.TEXT, nullable=False, default=""),
            Column(name="importance",  type=ColumnType.REAL, nullable=False, default=0.0),
            Column(name="status",      type=ColumnType.TEXT, nullable=False, default="active"),
            Column(name="updated_at",  type=ColumnType.TIMESTAMP, nullable=False),
        ],
        indexes=[
            Index(name="idx_vec_scope", columns=["user_id", "session_id"]),
        ],
    )

    def __init__(self, inner: MemoryStoreBase, llm: "LLM", *,
                 hybrid_weight: float = _DEFAULT_HYBRID_WEIGHT):
        if not isinstance(inner, MemoryStoreBase):
            raise TypeError("inner 必须是 MemoryStoreBase 的实例")
        if not (0.0 <= hybrid_weight <= 1.0):
            raise ValueError("hybrid_weight 必须在 0~1 之间")
        self.inner = inner
        self.llm = llm
        self.hybrid_weight = float(hybrid_weight)

        # ---------- 向量存储（按 inner 类型选择后端） ----------
        self._vec_backend: str                      # "sqlite" | "memory"
        self._vec_cache: Dict[str, MemoryVector]    # 统一缓存：memory 后端完整数据；sqlite 后端亦做内存热点缓存
        self._ensure_vec_backend()

    # ===================== 后端探测与初始化 =====================

    def _ensure_vec_backend(self) -> None:
        """判定 inner 类型，初始化对应的向量存储"""
        # 局部导入避免模块加载顺序问题
        from .sqlite_store import SQLiteMemoryStore

        self._vec_cache = {}
        if isinstance(self.inner, SQLiteMemoryStore):
            self._vec_backend = "sqlite"
            self.inner._db.ensure_schema(self._VEC_SCHEMA, migrate=True)
        else:
            self._vec_backend = "memory"


    # ===================== 向量序列化 / 行映射 =====================

    @staticmethod
    def _pack_vec(vec: List[float]) -> bytes:
        # struct 格式：端序 + 数量 + 类型字符，例如 "<3f" 表示 3 个 little-endian float32
        return struct.pack(f"{_VEC_ENDIAN}{len(vec)}{_VEC_CHAR}", *vec)

    @staticmethod
    def _unpack_vec(blob: bytes, dim: int) -> List[float]:
        return list(struct.unpack(f"{_VEC_ENDIAN}{dim}{_VEC_CHAR}", blob))

    @classmethod
    def _row_to_vec(cls, row: Dict[str, Any]) -> MemoryVector:
        """SQLite 行 -> MemoryVector；embedding 从 BLOB 解包，updated_at 从 ISO 串转 float 时间戳"""
        dim = int(row["dim"])
        updated_raw = row.get("updated_at")
        if isinstance(updated_raw, datetime):
            ts = updated_raw.timestamp()
        elif isinstance(updated_raw, str) and updated_raw:
            try:
                ts = datetime.fromisoformat(updated_raw).timestamp()
            except ValueError:
                ts = time.time()
        elif isinstance(updated_raw, (int, float)):
            ts = float(updated_raw)
        else:
            ts = time.time()
        return MemoryVector(
            memory_id=row["memory_id"],
            user_id=row.get("user_id", "") or "",
            session_id=row.get("session_id", "") or "",
            embedding=cls._unpack_vec(row["embedding"], dim),
            dim=dim,
            model=row.get("model", "") or "",
            kind=row.get("kind", "") or "",
            importance=float(row.get("importance", 0.0) or 0.0),
            status=row.get("status", "active") or "active",
            updated_at=ts,
        )

    # ===================== 向量读写（按后端路由） =====================

    def _build_vec(self, record: MemoryRecord, vec: List[float]) -> MemoryVector:
        """根据 MemoryRecord + embedding 向量构造 MemoryVector（两后端共用）"""
        return MemoryVector(
            memory_id=record.memory_id,
            user_id=record.user_id,
            session_id=record.session_id,
            embedding=list(vec),
            dim=len(vec),
            model=getattr(self.llm, "embedding_model", "unknown"),
            kind=record.kind or "",
            importance=float(record.importance or 0.0),
            status=record.status or "active",
            updated_at=time.time(),
        )

    def _save_vec(self, record: MemoryRecord, vec: Optional[List[float]] = None) -> None:
        """写单条向量；``vec`` 已给出则直接落盘（不再重算 embedding），为 None 才现算。"""
        if vec is None:
            vecs = self._safe_embed_documents_batch([record.content])
            if vecs is None:
                return
            vec = vecs[0]
        mv = self._build_vec(record, vec)
        self._vec_cache[record.memory_id] = mv
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            inner._db.upsert(
                "memory_embeddings",
                {
                    "memory_id": mv.memory_id,
                    "user_id": mv.user_id,
                    "session_id": mv.session_id,
                    "embedding": self._pack_vec(mv.embedding),
                    "dim": mv.dim,
                    "model": mv.model,
                    "kind": mv.kind,
                    "importance": mv.importance,
                    "status": mv.status,
                    "updated_at": datetime.fromtimestamp(mv.updated_at).isoformat(),
                },
                conflict_keys="memory_id",
            )

    def _load_vec(self, record: MemoryRecord) -> Optional[List[float]]:
        # 先命中内存缓存（sqlite/memory 后端都走这里，省掉 SQL）
        cached = self._vec_cache.get(record.memory_id)
        if cached is not None:
            # B13 第二步：缓存里 stale 的向量视为未命中（不返回给语义计算）
            if (cached.status or "active") == "active":
                return cached.embedding
            return None
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            row = inner._db.select_one_where(Query(
                table_name="memory_embeddings",
                filters=inner._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=record.memory_id),
                    Filter(field="status",    op=FilterOp.EQ, value="active"),
                ]),
            ))
            if row is None:
                return None
            mv = self._row_to_vec(row)
            self._vec_cache[record.memory_id] = mv
            return mv.embedding
        return None

    def _load_vecs_batch(self, records: List[MemoryRecord]) -> List[Optional[List[float]]]:
        """批量从本地缓存读向量（一次 SQL IN，省掉 N 次查询）；仅返回 status=active 的向量。"""
        result: List[Optional[List[float]]] = [None] * len(records)
        if not records:
            return result
        miss_idx: List[int] = []
        miss_ids: List[str] = []
        for i, rec in enumerate(records):
            cached = self._vec_cache.get(rec.memory_id)
            if cached is not None:
                # B13 第二步：缓存里 stale 的向量直接视为未命中
                if (cached.status or "active") == "active":
                    result[i] = cached.embedding
            else:
                miss_idx.append(i)
                miss_ids.append(rec.memory_id)
        if not miss_idx:
            return result
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            rows = inner._db.select_where(Query(
                table_name="memory_embeddings",
                filters=[
                    Filter(field="user_id", op=FilterOp.EQ, value=inner.user_id),
                    Filter(field="session_id", op=FilterOp.EQ, value=inner.session_id),
                    Filter(field="memory_id", op=FilterOp.IN, value=miss_ids),
                    Filter(field="status", op=FilterOp.EQ, value="active"),
                ],
            ))
            by_id: Dict[str, MemoryVector] = {}
            for r in rows:
                mv = self._row_to_vec(r)
                by_id[mv.memory_id] = mv
                self._vec_cache[mv.memory_id] = mv
            for i, mid in zip(miss_idx, miss_ids):
                hit = by_id.get(mid)
                if hit is not None:
                    result[i] = hit.embedding
        # memory 后端未命中就是 None，保持 result 中对应位置为 None
        return result

    def _delete_vec(self, record: MemoryRecord) -> None:
        self._vec_cache.pop(record.memory_id, None)
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            inner._db.delete_where("memory_embeddings", Query(
                table_name="memory_embeddings",
                filters=inner._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=record.memory_id),
                ]),
            ))

    def _update_vec_meta(self, record: MemoryRecord) -> None:
        """同步单条记录的向量元数据（kind/importance/status）——不删向量、不重算 embedding。

        B13 第二步：stale 向量保留在本地表 / 内存缓存中以便审计/回溯，但 metadata 必须实时同步，
        让 _load_vec/_load_vecs_batch 通过 status='active' 过滤掉 stale。
        设计口径与 ChromaMemoryStore._update_vec_metadata 完全一致。
        """
        cached = self._vec_cache.get(record.memory_id)
        if cached is None and self._vec_backend == "sqlite":
            # 没命中缓存：尝试从 SQLite 读出来再更新；读不到就跳过（本来就没向量，将来 rebuild 会补）
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            row = inner._db.select_one_where(Query(
                table_name="memory_embeddings",
                filters=inner._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=record.memory_id),
                ]),
            ))
            if row is not None:
                cached = self._row_to_vec(row)
                self._vec_cache[record.memory_id] = cached
        if cached is None:
            return
        # 同步 meta 字段（保留原 embedding / dim / model）
        cached.kind = record.kind or ""
        cached.importance = float(record.importance or 0.0)
        cached.status = record.status or "active"
        cached.updated_at = time.time()
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            inner._db.update_where(
                "memory_embeddings",
                {
                    "kind": cached.kind,
                    "importance": cached.importance,
                    "status": cached.status,
                    "updated_at": datetime.fromtimestamp(cached.updated_at).isoformat(),
                },
                Query(
                    table_name="memory_embeddings",
                    filters=inner._scope_where([
                        Filter(field="memory_id", op=FilterOp.EQ, value=record.memory_id),
                    ]),
                ),
            )

    def _clear_vecs_scope(self) -> None:
        """只清空当前作用域的向量，别把别人的清了"""
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            # 先查出当前 scope 的 memory_id 列表，一并清内存缓存
            rows = inner._db.select_where(Query(
                table_name="memory_embeddings",
                filters=inner._scope_filters(),
                columns=["memory_id"],
            ))
            for r in rows:
                self._vec_cache.pop(r["memory_id"], None)
            inner._db.delete_where("memory_embeddings", Query(
                table_name="memory_embeddings",
                filters=inner._scope_filters(),
            ))
        else:
            # 内存后端：根据 inner.all() 当前作用域里的 id 来删
            active_ids = {r.memory_id for r in self.inner.all(include_inactive=True)}
            for mid in list(self._vec_cache.keys()):
                if mid in active_ids:
                    self._vec_cache.pop(mid, None)

    def vector_meta(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """读取某条记忆向量缓存里存的 meta（kind / importance / status）

        只读查询接口：向量侧自带的 meta 可单独取用，无需回查主表。
        找不到该向量缓存返回 None；作用域与 inner 当前 scope 一致。
        """
        cached = self._vec_cache.get(memory_id)
        if cached is not None:
            return cached.meta
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            row = inner._db.select_one_where(Query(
                table_name="memory_embeddings",
                filters=inner._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=memory_id),
                ]),
            ))
            if row is None:
                return None
            mv = self._row_to_vec(row)
            self._vec_cache[memory_id] = mv
            return mv.meta
        return None

    # ===================== embedding 调用（失败降级） =====================

    def _safe_embed_query(self, query: str) -> Optional[List[float]]:
        try:
            vec = self.llm.embed_query(query)
            return list(vec) if vec else None
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 查询 embedding 失败，降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return None

    def _safe_embed_documents_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量 embedding；失败返回 None（任一维不匹配就整体返回 None）"""
        if not texts:
            return None
        try:
            vecs = self.llm.embed_documents(list(texts))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 批量 embedding 失败（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return None
        if not vecs or len(vecs) != len(texts):
            return None
        result: List[List[float]] = []
        for v in vecs:
            if not v:
                return None
            result.append(list(v))
        return result

    # ===================== MemoryStoreBase 6 个原子原语 =====================

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入记忆 + 计算 embedding（文档侧，失败降级）"""
        record = self.inner.add(record)
        try:
            self._save_vec(record)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 保存向量失败，该条降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.inner.get(memory_id)

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        return self.inner.all(include_inactive=include_inactive)

    def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新记忆：content 变化则重算 embedding；仅 kind/importance/status 变化则同步向量 meta；无变化保持原样。"""
        old = self.inner.get(record.memory_id)
        # B13 第二步：必须在 inner.update 之前记下旧 content 快照，避免 InMemoryStore 引用语义导致对比失效
        old_content = old.content if old is not None else None
        record = self.inner.update(record)
        try:
            if old_content is not None and old_content != record.content:
                # content 变了→旧向量作废，重算新向量（_save_vec 会 upsert 全字段，status 一并刷新）
                self._save_vec(record)
            else:
                # content 没变：只刷新向量表 / 缓存里的 kind/importance/status（不删向量、不重算 embedding）
                self._update_vec_meta(record)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 更新向量失败（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
        return record

    def delete(self, memory_id: str) -> bool:
        old = self.inner.get(memory_id)
        ok = self.inner.delete(memory_id)
        if ok and old is not None:
            try:
                self._delete_vec(old)
            except Exception:  # noqa: BLE001
                # 向量清不掉不影响主流程，下次 clear_scope 会兜底
                pass
        return ok

    def clear(self) -> None:
        try:
            self._clear_vecs_scope()
        except Exception:  # noqa: BLE001
            # 向量清不掉就直接清空内存缓存，至少保证内存后端干净
            self._vec_cache.clear()
        self.inner.clear()

    # ===================== 语义增强 search（核心） =====================

    def search(self, query: str, limit: int = 5, *,
               include_preferences: bool = True,
               prefetch_top_n: int = 20) -> List[MemoryRecord]:
        # 空查询直接走原后端（最近 N 条），不浪费 embedding 调用
        if not query or not query.strip():
            return self.inner.search(query, limit, include_preferences=include_preferences)

        q_vec = self._safe_embed_query(query)
        if q_vec is None:
            # 查询 embedding 失败，整次降级到纯词法
            return self.inner.search(query, limit, include_preferences=include_preferences)

        records = self.inner.all(include_inactive=False)
        # B12：rank 之前过滤已到期的 active 记忆，和基类 search 保持一致
        records = [r for r in records if not r.is_expired()]

        # v1.9.3 两段式检索（避免冷库写放大，又保留 B2 同义召回能力）：
        # 第 1 轮「粗排」：只读向量缓存；有向量的走 hybrid，没向量的按纯词法分参与排序
        #   （这里显式关闭 semantic_floor——冷库首查时若保留 floor，会把所有"词法=0
        #    但语义相关"的记忆直接丢掉，永远进不了补算名单，形成「冷库死锁」）。
        vecs = self.semantic_batch_get(records)
        vec_by_id: Dict[str, List[float]] = {}
        for rec, vec in zip(records, vecs):
            if vec is not None:
                vec_by_id[rec.memory_id] = list(vec)

        def _make_getter(vec_map: Dict[str, List[float]], floor_filter: bool):
            """粗排阶段 floor_filter=False：无向量条目返回 0（纯词法），不触发 floor；
            精排阶段 floor_filter=True：无向量条目返回 None，走原生 floor 过滤。"""
            def _getter(r: MemoryRecord) -> Optional[float]:
                v = vec_map.get(r.memory_id)
                if v is None:
                    return None if floor_filter else 0.0
                return cosine_similarity(q_vec, v)
            return _getter

        first_pass_limit = max(limit, int(prefetch_top_n))
        first_pass = rank_records(
            query, records, limit=first_pass_limit,
            include_preferences=include_preferences,
            semantic_getter=_make_getter(vec_by_id, floor_filter=False),
            hybrid_weight=self.hybrid_weight,
            semantic_floor=0.0,  # 粗排关闭 floor，避免冷库死锁
        )

        # 第 2 轮「补算」：只对粗排进 top-N 的缺向量候选做一次小批量 embed（默认 ≤ 20 条），
        # 补算结果立即落盘缓存；相比 v1.9.2 的「全库一起补」，不再触发冷库全量写放大。
        need_prefetch = [r for r in first_pass if r.memory_id not in vec_by_id]
        if need_prefetch and prefetch_top_n > 0:
            need_prefetch = need_prefetch[:max(1, int(prefetch_top_n))]
            try:
                new_vecs = self.semantic_vectors([r.content for r in need_prefetch])
                if new_vecs and len(new_vecs) == len(need_prefetch):
                    for rec, v in zip(need_prefetch, new_vecs):
                        if v:
                            vl = list(v)
                            vec_by_id[rec.memory_id] = vl
                            try:
                                self._save_vec(rec, vl)
                            except Exception:  # noqa: BLE001
                                pass  # 存不进缓存不影响本次检索
            except Exception:  # noqa: BLE001
                pass  # 补算失败就用粗排结果，绝不阻塞检索

        # 第 3 轮「精排」：此时 top-N 候选已有向量，恢复原生 semantic_floor 过滤，
        # 保证 B2 同义改写（词法=0、语义高）能被提到前面；无向量且语义不足的按 floor 正常丢弃。
        return rank_records(
            query, records, limit,
            include_preferences=include_preferences,
            semantic_getter=_make_getter(vec_by_id, floor_filter=True),
            hybrid_weight=self.hybrid_weight,
            semantic_floor=self._SEMANTIC_FLOOR,
        )

    # ===================== B5：合并相似度的向量通道 =====================

    def semantic_vectors(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量算文本向量（纯文本入口；只对 semantic_batch_get_and_save 传来的缺失项补算）

        约定：
        - 入参 texts 是纯文本列表，不与 MemoryRecord 绑定，无法按 memory_id 查缓存；
          所以本方法不做缓存命中、也不落盘，就是对给定文本做一次全量批量 embedding；
        - 调用方（search / merge_similar）统一走 semantic_batch_get_and_save：
          它按 record 批量查缓存，只把缺失项的文本喂给本方法，补算结果再落盘；
        - 任一条失败 / 维度对不上 / 返回 None → 整体返回 None（交给调用方退化为纯词法，
          不做半拉子结果）。
        """
        if not texts:
            return None
        return self._safe_embed_documents_batch(list(texts))

    def semantic_batch_get(self, records: List[MemoryRecord]) -> List[Optional[List[float]]]:
        """v1.8.3：批量从本地缓存读取向量，供 search 优先命中（经 semantic_batch_get_and_save）"""
        return self._load_vecs_batch(list(records))

    def embedding_stats(self) -> dict:
        """返回向量通道健康统计（供调试/健康日志使用）。"""
        cached_total = 0
        dim = None
        if self._vec_backend == "sqlite":
            try:
                from .sqlite_store import SQLiteMemoryStore as _SQLite
                inner: _SQLite = self.inner  # type: ignore[assignment]
                row = inner._db.query(
                    "SELECT COUNT(*) AS c, MAX(dim) AS d FROM memory_embeddings "
                    "WHERE user_id=? AND session_id=?",
                    (inner.user_id, inner.session_id),
                    fetch_one=True,
                )
                if row is not None:
                    cached_total = int(row["c"] or 0)
                    dim = int(row["d"]) if row["d"] is not None else None
            except Exception:  # noqa: BLE001
                cached_total = 0
                dim = None
            # 统计同时反映内存缓存大小（sqlite 后端 _vec_cache 作为热点缓存使用）
            cache_size = len(self._vec_cache)
        else:
            cached_total = len(self._vec_cache)
            cache_size = cached_total
            if self._vec_cache:
                sample_vec = next(iter(self._vec_cache.values()))
                dim = sample_vec.dim if sample_vec.dim else (
                    len(sample_vec.embedding) if isinstance(sample_vec.embedding, list) else None
                )
        return {
            "available": True,
            "backend": f"semantic_{self._vec_backend}",
            "model": getattr(self.llm, "embedding_model", "unknown"),
            "dimension": dim,
            "cached_total": cached_total,
            "mem_cache_size": cache_size,
            "total_records": len(self.inner.all(include_inactive=True)),
        }

    # ===================== B13：写入前向量冲突检测（以 Chroma 为主，本类提供兜底） =====================

    def find_similar(self, content: str, *, limit: int = 5, threshold: float = 0.85,
                     include_expired: bool = False,
                     prefetch_missing: bool = False) -> List[tuple]:
        """按向量相似度召回与 ``content`` 最相似的已有记忆。

        这是冲突检测 / 写入前去重的统一接口（B13）：
        - ChromaMemoryStore 会直接用 Chroma 的 ANN 近邻，O(log n) 复杂度；
        - 本类（SemanticMemoryStore）的兜底实现：embed 新内容 → 从本地缓存批量
          取已有向量 → 计算 cosine，按相似度降序返回 [(record, similarity), ...]。
        - 相似度低于 ``threshold`` 的不返回；失败（embed 拿不到向量）返回空列表，
          调用方（MemoryManager）应退化为词法 Jaccard，绝不阻塞写入。
        - ``include_expired=False``（默认）：只返回未过期的有效记忆（原行为）；
          ``include_expired=True``：过期但仍 active 的记忆也纳入（过期 ≠ 删除，
          写入前查重不能漏检），此时未过期项优先排在前面。
        - ``prefetch_missing=False``（默认）：只读向量缓存，缺失向量的记录直接跳过，
          避免只读查询（如 manager.check_conflict）触发全库补算写放大；
          ``prefetch_missing=True``：对缺失向量的条目批量补算并落盘（仅 manager
          写入前查重走这条），保证冲突检测不漏掉尚未建立向量的新记录。
        """
        if not content or not content.strip():
            return []
        q_vec = self._safe_embed_query(content.strip())
        if q_vec is None:
            return []
        records = self.inner.all(include_inactive=False)
        if not include_expired:
            # 默认只保留未过期的有效记忆；include_expired=True 时过期但仍 active 的也纳入查重
            records = [r for r in records if not r.is_expired()]
        if not records:
            return []
        # v1.9.3 优化：按调用场景区分
        # - 只读探测（默认）：semantic_batch_get 只读缓存，缺失项跳过，不触发 embed 补算
        # - 写入前查重（prefetch_missing=True）：semantic_batch_get_and_save 补算缺失项并落盘
        if prefetch_missing:
            vecs = self.semantic_batch_get_and_save(records)
        else:
            vecs = self.semantic_batch_get(records)
        hits: List[tuple] = []
        for r, v in zip(records, vecs):
            if v is None:
                continue
            sim = cosine_similarity(q_vec, list(v))
            if sim >= threshold:
                hits.append((r, sim))
        if include_expired:
            # 未过期项优先（is_expired() False < True），再按相似度降序，
            # 保证调用方取 top-1 时先命中有效冲突，没有有效冲突才落到过期项
            hits.sort(key=lambda x: (x[0].is_expired(), -x[1], -x[0].importance))
        else:
            hits.sort(key=lambda x: (-x[1], -x[0].importance))
        return hits[:max(1, int(limit))]

    def touch(self, memory_ids) -> int:
        """命中记账代理给 inner

        必须代理（而不是用基类默认的 get + update）：包装器的 update() 最终会走到
        inner.update()，而 SQLite 的 update() 会把 updated_at 刷成当前时间，
        那样就破坏了时间衰减。inner 自己的 touch() 才不会碰 updated_at。
        """
        return self.inner.touch(list(memory_ids))

    # ===================== 资源释放 =====================

    def close(self) -> None:
        # 不主动 close inner：inner 由调用方创建、调用方负责生命周期
        # （SemanticMemoryStore 只是包一层，不拥有 inner）
        self._vec_cache.clear()
        return None

    # ===================== 可选：手动重建向量（调试/换模型用） =====================

    def rebuild_embeddings(self, batch_size: int = 20) -> int:
        """全量重建当前作用域所有记忆的向量（换 embedding 模型时用）

        返回成功重建的条数。失败的条目跳过，不抛异常。
        """
        records = self.inner.all(include_inactive=True)
        ok_count = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [r.content for r in batch]
            try:
                vecs = self.llm.embed_documents(texts)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"SemanticMemoryStore: rebuild 第 {i} 批失败，跳过（{exc!r}）",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            for r, vec in zip(batch, vecs):
                if not vec:
                    continue
                try:
                    self._save_vec(r, vec)
                    ok_count += 1
                except Exception:  # noqa: BLE001
                    continue
        return ok_count
