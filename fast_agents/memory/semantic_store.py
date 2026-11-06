"""语义检索后端（B2 / v1.4.0）

包装任意 MemoryStoreBase 实例，在原有词法检索上叠加 embedding 语义通道：
- 默认 hybrid_weight=0.7（偏语义，用户确认过）；
- 向量存储策略：inner 是 SQLiteMemoryStore 时，同库独立表 memory_embeddings（BLOB）；
  inner 是 InMemoryStore 或其他自定义后端时，用内存 dict 存；
- embedding 调用失败一律降级（warning + 照常 CRUD），不阻塞主流程；
- 不修改 memories 主表 schema，不污染既有数据；
- B5 起额外提供 semantic_vectors()（批量向量）与 touch()（记账代理），
  让「记忆合并 / 命中记账」也能用上语义通道。

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
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

from fast_agents.utils.time_utils import cstnow
from .memory_store_base import MemoryStoreBase
from .schema import MemoryRecord
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
    _VEC_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id   TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            dim         INTEGER NOT NULL,
            model       TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (memory_id, user_id, session_id)
        );
        CREATE INDEX IF NOT EXISTS idx_vec_scope
            ON memory_embeddings(user_id, session_id);
    """

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
        self._vec_backend: str          # "sqlite" | "memory"
        self._mem_vecs: Dict[str, List[float]]   # 仅 memory 后端用
        self._ensure_vec_backend()

    # ===================== 后端探测与初始化 =====================

    def _ensure_vec_backend(self) -> None:
        """判定 inner 类型，初始化对应的向量存储"""
        # 局部导入避免模块加载顺序问题
        from .sqlite_store import SQLiteMemoryStore as _SQLite

        if isinstance(self.inner, _SQLite):
            self._vec_backend = "sqlite"
            self._mem_vecs = {}
            self._ensure_sqlite_vec_table()
        else:
            self._vec_backend = "memory"
            self._mem_vecs = {}

    def _ensure_sqlite_vec_table(self) -> None:
        """在 inner 的 SQLite 库里建 memory_embeddings 独立表"""
        from .sqlite_store import SQLiteMemoryStore as _SQLite
        inner: _SQLite = self.inner  # type: ignore[assignment]
        with inner._connect() as db:
            db.executescript(self._VEC_TABLE_SQL)

    # ===================== 向量序列化 =====================

    @staticmethod
    def _pack_vec(vec: List[float]) -> bytes:
        # struct 格式：端序 + 数量 + 类型字符，例如 "<3f" 表示 3 个 little-endian float32
        return struct.pack(f"{_VEC_ENDIAN}{len(vec)}{_VEC_CHAR}", *vec)

    @staticmethod
    def _unpack_vec(blob: bytes, dim: int) -> List[float]:
        return list(struct.unpack(f"{_VEC_ENDIAN}{dim}{_VEC_CHAR}", blob))

    # ===================== 向量读写（按后端路由） =====================

    def _scope_key(self, record: MemoryRecord) -> tuple:
        """SQLite 用三元组做主键；memory 后端只用 memory_id（inner 已做作用域隔离，
        不同 scope 的 record.memory_id 不会撞——MemoryRecord 每次新建都是 uuid）。"""
        return (record.memory_id, record.user_id, record.session_id)

    def _save_vec(self, record: MemoryRecord, vec: List[float]) -> None:
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            with inner._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO memory_embeddings "
                    "(memory_id, user_id, session_id, embedding, dim, model, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.memory_id, record.user_id, record.session_id,
                        self._pack_vec(vec), len(vec),
                        getattr(self.llm, "embedding_model", "unknown"),
                        cstnow().isoformat(),
                    ),
                )
        else:
            self._mem_vecs[record.memory_id] = list(vec)

    def _load_vec(self, record: MemoryRecord) -> Optional[List[float]]:
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            with inner._connect() as db:
                row = db.execute(
                    "SELECT embedding, dim FROM memory_embeddings "
                    "WHERE memory_id=? AND user_id=? AND session_id=?",
                    (record.memory_id, record.user_id, record.session_id),
                ).fetchone()
            if row is None:
                return None
            return self._unpack_vec(row["embedding"], row["dim"])
        else:
            return self._mem_vecs.get(record.memory_id)

    def _delete_vec(self, record: MemoryRecord) -> None:
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            with inner._connect() as db:
                db.execute(
                    "DELETE FROM memory_embeddings "
                    "WHERE memory_id=? AND user_id=? AND session_id=?",
                    (record.memory_id, record.user_id, record.session_id),
                )
        else:
            self._mem_vecs.pop(record.memory_id, None)

    def _clear_vecs_scope(self) -> None:
        """只清空当前作用域的向量，别把别人的清了"""
        if self._vec_backend == "sqlite":
            from .sqlite_store import SQLiteMemoryStore as _SQLite
            inner: _SQLite = self.inner  # type: ignore[assignment]
            clause, params = inner._scope_clause()
            with inner._connect() as db:
                db.execute(
                    f"DELETE FROM memory_embeddings WHERE {clause}",
                    params,
                )
        else:
            # 内存后端：根据 inner.all() 当前作用域里的 id 来删
            active_ids = {r.memory_id for r in self.inner.all(include_inactive=True)}
            for mid in list(self._mem_vecs.keys()):
                if mid in active_ids:
                    self._mem_vecs.pop(mid, None)

    # ===================== embedding 调用（失败降级） =====================

    def _safe_embed_document(self, content: str) -> Optional[List[float]]:
        """单条文档 embedding；失败返回 None（降级为无向量）"""
        try:
            vecs = self.llm.embed_documents([content])
            if not vecs or not vecs[0]:
                return None
            return list(vecs[0])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 文档 embedding 失败，降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return None

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

    # ===================== MemoryStoreBase 6 个原子原语 =====================

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入记忆 + 异步计算 embedding（同步立即算，失败降级）"""
        record = self.inner.add(record)
        vec = self._safe_embed_document(record.content)
        if vec is not None:
            try:
                self._save_vec(record, vec)
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
        """更新记忆：content 变化则重算 embedding，否则保留旧向量"""
        old = self.inner.get(record.memory_id)
        content_changed = old is not None and old.content != record.content
        record = self.inner.update(record)
        if content_changed:
            vec = self._safe_embed_document(record.content)
            if vec is not None:
                try:
                    self._save_vec(record, vec)
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
            # 向量清不掉就直接清空内存 dict，至少保证内存后端干净
            if self._vec_backend == "memory":
                self._mem_vecs.clear()
        self.inner.clear()

    # ===================== 语义增强 search（核心） =====================

    def search(self, query: str, limit: int = 5, *,
               include_preferences: bool = True) -> List[MemoryRecord]:
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

        def semantic_getter(r: MemoryRecord) -> Optional[float]:
            vec = self._load_vec(r)
            if vec is None:
                # 懒加载补算：升级上来的旧数据、或之前 add 时 embedding 失败的
                vec = self._safe_embed_document(r.content)
                if vec is not None:
                    try:
                        self._save_vec(r, vec)
                    except Exception:  # noqa: BLE001
                        return None
                else:
                    return None
            return cosine_similarity(q_vec, vec)

        return rank_records(
            query, records, limit,
            include_preferences=include_preferences,
            semantic_getter=semantic_getter,
            hybrid_weight=self.hybrid_weight,
            semantic_floor=self._SEMANTIC_FLOOR,
        )

    # ===================== B5：合并相似度的向量通道 =====================

    def semantic_vectors(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量算向量，供上层做「合并相似度」；失败返回 None（降级纯词法）

        合并是两两比较，逐对调用 embedding 会非常贵，所以这里一次性算完一批。
        只要有一条拿不到向量就返回 None——宁可按纯词法走，也不要出现
        「一半有向量一半没有」的混合口径（阈值会变得不可预期）。
        """
        if not texts:
            return None
        try:
            vecs = self.llm.embed_documents(list(texts))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"SemanticMemoryStore: 批量 embedding 失败，合并相似度降级为纯词法（{exc!r}）",
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
        self._mem_vecs.clear()
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
            for r, v in zip(batch, vecs):
                if not v:
                    continue
                try:
                    self._save_vec(r, list(v))
                    ok_count += 1
                except Exception:  # noqa: BLE001
                    continue
        return ok_count
