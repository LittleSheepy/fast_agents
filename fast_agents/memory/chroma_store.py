"""Chroma 向量库后端（B2 扩展 / v1.8.0 阶段 1）

包装任意 MemoryStoreBase 实例，将 embedding 向量存到外部 Chroma 服务，
文本真相源仍由 inner 负责——和 SemanticMemoryStore 平级，属于「换包装器」
的另一选项，符合 SemanticMemoryStore 类注释中「未来接 Chroma 换包装器即可」
的设计意图。

阶段 1 范围：骨架实现 + 纯词法降级
- 不引入 chromadb 依赖（默认安装零重型依赖）；
- chroma_client 参数可为 None：此时整类降级为纯词法（相当于 inner 直通）；
- 即便传入客户端，Chroma 调用一律 try/except 降级，失败不阻塞主流程；
- 所有 add / get / all / update / delete / clear / touch / revise / retract /
  search / semantic_vectors 行为与 SemanticMemoryStore 对齐：
    * add/update 时把 content+向量写入 Chroma（失败降级）；
    * delete/clear 时同步清理 Chroma（失败不抛）；
    * search 先算 query 向量，再从 Chroma 召回 top 候选，与 inner 全部记录合流后
      仍统一交给 rank_records 做混合打分（打分公式与权重完全一致）；
    * B5 的 touch/revise/retract/archive_stale/merge_similar 代理给 inner
      （Chroma 只是向量索引，不参与记账/修订/归档/合并的逻辑）。

阶段 2（后续）：真实接 chromadb client（延迟导入 + optional-deps）。

使用方式（阶段 1 客户端为 None 时纯词法）：
    from fast_agents.llm import LLM
    from fast_agents.memory import SQLiteMemoryStore, ChromaMemoryStore, MemoryManager

    llm = LLM()
    inner = SQLiteMemoryStore("memories.db", user_id="u1")
    store = ChromaMemoryStore(inner, llm)   # chroma_client=None → 纯词法降级
    manager = MemoryManager(llm=llm, long_term_store=store)
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

from .memory_store_base import MemoryStoreBase
from .schema import MemoryRecord
from .scoring import cosine_similarity, rank_records

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)


class ChromaMemoryStore(MemoryStoreBase):
    """用外部 Chroma 向量库做语义检索的包装器（平级于 SemanticMemoryStore）

    设计要点：
    1. 继承 MemoryStoreBase，和 InMemoryStore / SQLiteMemoryStore /
       SemanticMemoryStore 平级，MemoryManager 无感知；
    2. 文本真相源永远是 inner（MemoryStoreBase 的任意实现，SQLite / 将来 Postgres
       皆可）；Chroma 只负责存「memory_id → 向量」，payload 可选带正文；
    3. chroma_client 由调用方传入并拥有生命周期（本类不创建、不关闭它）；
       传 None 时整个类降级为纯词法（直通 inner），默认安装零重型依赖；
    4. Chroma 调用失败一律降级（warning + 照常 CRUD / 纯词法检索），不阻塞主流程；
    5. 打分权重、语义门槛与 SemanticMemoryStore 完全一致，保证换后端时排序口径不变。
    """

    _DEFAULT_HYBRID_WEIGHT = 0.7     # 语义 0.7 / 词法 0.3（与 SemanticMemoryStore 一致）
    _SEMANTIC_FLOOR = 0.3            # 非偏好类的最低语义门槛（与 SemanticMemoryStore 一致）
    _DEFAULT_COLLECTION_PREFIX = "fast_agents_mem"
    _CHROMA_TOP_K_MULTIPLIER = 3     # Chroma 召回倍数，最终再由 rank_records 统一排序截 limit

    def __init__(self, inner: MemoryStoreBase, llm: "LLM", *,
                 chroma_client=None,
                 collection_name: Optional[str] = None,
                 hybrid_weight: float = _DEFAULT_HYBRID_WEIGHT,
                 payload_support: bool = True):
        """
        Args:
            inner: 文本真相源（任意 MemoryStoreBase 实现）。
            llm: 统一的 LLM 入口，负责 embed_query / embed_documents。
            chroma_client: 外部 Chroma 客户端实例；None 时整类降级为纯词法。
                阶段 1 不做类型校验，阶段 2 接真实 chromadb 时再细化。
            collection_name: Chroma 集合名，默认按 user_id/session_id 自动生成。
            hybrid_weight: 混合权重，默认 0.7（偏语义）。
            payload_support: 是否让 Chroma 携带正文 payload（Chroma/Qdrant=True，
                FAISS 等纯索引库=False）；阶段 2 接不同向量库时用。
        """
        if not isinstance(inner, MemoryStoreBase):
            raise TypeError("inner 必须是 MemoryStoreBase 的实例")
        if not (0.0 <= hybrid_weight <= 1.0):
            raise ValueError("hybrid_weight 必须在 0~1 之间")
        self.inner = inner
        self.llm = llm
        self.hybrid_weight = float(hybrid_weight)
        self.chroma_client = chroma_client
        self.payload_support = bool(payload_support)

        # 集合名按作用域生成：不同 user/session 互不干扰。
        # 注意：Chroma collection 名禁止冒号/斜杠，只允许字母数字与 _ - . ；
        # 若 user_id / session_id 含非法字符，阶段 2 接真实 Chroma 前需先归一化替换。
        self._collection_name = (
            collection_name
            or f"{self._DEFAULT_COLLECTION_PREFIX}_{inner.user_id}_{inner.session_id}"
        )
        # 阶段 1：chroma_client 不为 None 时仍不做真实连接，只打 warning 并降级
        # 阶段 2 会在这里真正 get_or_create_collection
        self._collection = None
        if chroma_client is not None:
            warnings.warn(
                "ChromaMemoryStore: 阶段 1 暂未接入真实 chromadb 客户端，"
                "chroma_client 参数已接收但向量写入/检索会降级为纯词法；"
                "阶段 2 将启用真实向量通道。",
                RuntimeWarning, stacklevel=2,
            )

    # ===================== Chroma 通道（阶段 1：空实现 / 降级） =====================
    # 阶段 2 填真实调用；阶段 1 这四个方法一律视为「没向量」，
    # 让上层逻辑自然走到「懒加载补算失败 → 语义分为 None → rank_records 纯词法」分支。

    def _collection_or_none(self):
        """拿到 Chroma collection；阶段 1 永远返回 None，阶段 2 返回真实集合"""
        return self._collection

    def _save_vec_chroma(self, record: MemoryRecord, vec: List[float]) -> None:
        """写向量到 Chroma；阶段 1 no-op，阶段 2 调 collection.upsert"""
        coll = self._collection_or_none()
        if coll is None:
            return
        # 阶段 2: coll.upsert(ids=[...], embeddings=[...], documents=[...], metadatas=[...])
        # 注意：不在这里抛异常，调用方会 try/except 兜底

    def _query_chroma(self, q_vec: List[float], top_k: int) -> List[str]:
        """从 Chroma 按向量召回 top_k 个 memory_id；阶段 1 返回空列表。

        阶段 2 实现时若想省掉对候选逐条重算 embedding，可把返回扩成
        ``List[Tuple[str, float]]``（memory_id + 向量库自带相似度/距离分），
        search() 直接复用该分数；当前保持只回 id，与 inner 全量合流后
        仍由 rank_records 统一算分，换后端时排序口径最稳定。
        """
        coll = self._collection_or_none()
        if coll is None:
            return []
        # 阶段 2: coll.query(query_embeddings=[q_vec], n_results=top_k) → 解析 ids
        return []

    def _delete_vec_chroma(self, memory_id: str) -> None:
        """从 Chroma 删一条向量；阶段 1 no-op"""
        coll = self._collection_or_none()
        if coll is None:
            return
        # 阶段 2: coll.delete(ids=[memory_id])

    def _clear_vecs_chroma_scope(self) -> None:
        """清空当前作用域（user/session）的 Chroma 向量；阶段 1 no-op"""
        coll = self._collection_or_none()
        if coll is None:
            return
        # 阶段 2: coll.delete(where={"user_id": ..., "session_id": ...})
        # 注意：跨集合清空要谨慎，collection 名已包含 user/session 时可直接删集合

    # ===================== embedding 调用（失败降级，与 SemanticMemoryStore 一致） =====================

    def _safe_embed_document(self, content: str) -> Optional[List[float]]:
        try:
            vecs = self.llm.embed_documents([content])
            if not vecs or not vecs[0]:
                return None
            return list(vecs[0])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 文档 embedding 失败，降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return None

    def _safe_embed_query(self, query: str) -> Optional[List[float]]:
        try:
            vec = self.llm.embed_query(query)
            return list(vec) if vec else None
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 查询 embedding 失败，降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return None

    def _safe_embed_documents_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量 embedding；失败返回 None（与 semantic_vectors 的降级口径一致）"""
        if not texts:
            return None
        try:
            vecs = self.llm.embed_documents(list(texts))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 批量 embedding 失败（{exc!r}）",
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

    # ===================== MemoryStoreBase 6 个原子原语（行为对齐 SemanticMemoryStore） =====================

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入 inner + 尝试写 Chroma 向量（失败降级）"""
        record = self.inner.add(record)
        vec = self._safe_embed_document(record.content)
        if vec is not None:
            try:
                self._save_vec_chroma(record, vec)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"ChromaMemoryStore: 写入向量失败，该条降级为纯词法（{exc!r}）",
                    RuntimeWarning, stacklevel=2,
                )
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.inner.get(memory_id)

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        return self.inner.all(include_inactive=include_inactive)

    def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新 inner；content 变化则重算并回写 Chroma 向量"""
        old = self.inner.get(record.memory_id)
        content_changed = old is not None and old.content != record.content
        record = self.inner.update(record)
        if content_changed:
            vec = self._safe_embed_document(record.content)
            if vec is not None:
                try:
                    self._save_vec_chroma(record, vec)
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"ChromaMemoryStore: 更新向量失败（{exc!r}）",
                        RuntimeWarning, stacklevel=2,
                    )
        return record

    def delete(self, memory_id: str) -> bool:
        old = self.inner.get(memory_id)
        ok = self.inner.delete(memory_id)
        if ok and old is not None:
            try:
                self._delete_vec_chroma(memory_id)
            except Exception:  # noqa: BLE001
                # 向量清不掉不影响主流程，下次 clear 会兜底（inner 真相对齐）
                pass
        return ok

    def clear(self) -> None:
        try:
            self._clear_vecs_chroma_scope()
        except Exception:  # noqa: BLE001
            pass
        self.inner.clear()

    # ===================== 语义 search =====================
    # 阶段 1：_query_chroma 返回空列表，候选集完全来自 inner.all()，
    # semantic_getter 拿不到向量就返回 None → rank_records 纯词法打分。
    # 阶段 2：_query_chroma 返回 Chroma 召回的 memory_id 集合，将这些候选和
    # inner.all() 合流（Chroma 召回的 id 优先，inner 全量兜底保证不丢偏好），
    # semantic_getter 在 chroma 回传候选上仍可算余弦分（或直接使用 Chroma 的距离分）。

    def search(self, query: str, limit: int = 5, *,
               include_preferences: bool = True) -> List[MemoryRecord]:
        if not query or not query.strip():
            return self.inner.search(query, limit, include_preferences=include_preferences)

        q_vec = self._safe_embed_query(query)
        if q_vec is None:
            return self.inner.search(query, limit, include_preferences=include_preferences)

        # 阶段 1：_collection 恒为 None，向量通道整体关闭；此状态下不下发
        # semantic_getter，直接让 rank_records 走纯词法，省掉逐条回调开销。
        use_semantic = self._collection_or_none() is not None

        # 阶段 1：chroma_ids 为空，records 就等于 inner 全量；
        # 阶段 2：chroma_ids 非空，这里做一次 id 集合合流去重
        chroma_ids = self._query_chroma(q_vec, top_k=limit * self._CHROMA_TOP_K_MULTIPLIER)
        records = self.inner.all(include_inactive=False)
        # B12：rank 之前先过滤已到期的 active 记忆
        records = [r for r in records if not r.is_expired()]
        if chroma_ids:
            # 让 chroma 召回的 id 与 inner 全量合并（inner 兜底避免漏掉偏好/最近条目）
            id_order = list(chroma_ids)
            seen = set(chroma_ids)
            for r in records:
                if r.memory_id not in seen:
                    id_order.append(r.memory_id)
                    seen.add(r.memory_id)
            by_id = {r.memory_id: r for r in records}
            records = [by_id[mid] for mid in id_order if mid in by_id]

        def semantic_getter(r: MemoryRecord) -> Optional[float]:
            # 阶段 2：可从 Chroma payload 直接取向量/距离分；这里懒补算兜底
            vec = self._safe_embed_document(r.content)
            if vec is None:
                return None
            try:
                self._save_vec_chroma(r, vec)
            except Exception:  # noqa: BLE001
                pass
            return cosine_similarity(q_vec, vec)

        return rank_records(
            query, records, limit,
            include_preferences=include_preferences,
            semantic_getter=semantic_getter if use_semantic else None,
            hybrid_weight=self.hybrid_weight,
            semantic_floor=self._SEMANTIC_FLOOR,
        )

    # ===================== B5：合并相似度向量通道 =====================

    def semantic_vectors(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量算向量，供上层做「记忆合并」；阶段 1 因为没有向量存储，
        仍可由 llm.embed_documents 算（合并阶段不需要持久化向量），失败降级 None。

        与 SemanticMemoryStore 口径一致：任何一条拿不到向量就整体返回 None，
        避免一半有向量一半没有导致阈值不可预期。
        """
        if not texts:
            return None
        return self._safe_embed_documents_batch(list(texts))

    # ===================== B5：代理记账 / 修订 / 归档 / 合并 =====================
    # 这些都是"针对 MemoryRecord 本体"的操作，永远走 inner；
    # Chroma 只在 content 变化时同步向量（由 add/update 触发）。

    def touch(self, memory_ids) -> int:
        """命中记账代理给 inner（和 SemanticMemoryStore 同样的理由：不刷 updated_at）"""
        return self.inner.touch(list(memory_ids))

    def revise(self, memory_id: str, content: str, source: str) -> MemoryRecord:
        """修订记忆：代理 inner，再尝试重算 Chroma 向量（失败不阻塞）"""
        revised = self.inner.revise(memory_id, content, source=source)
        if revised is not None:
            # revise 已改变 content，重算向量
            vec = self._safe_embed_document(revised.content)
            if vec is not None:
                try:
                    self._save_vec_chroma(revised, vec)
                except Exception:  # noqa: BLE001
                    pass
        return revised

    def retract(self, memory_id: str) -> MemoryRecord:
        """撤回记忆：代理 inner，并尝试删 Chroma 向量"""
        retracted = self.inner.retract(memory_id)
        if retracted is not None:
            try:
                self._delete_vec_chroma(memory_id)
            except Exception:  # noqa: BLE001
                pass
        return retracted

    def archive_stale(self, *, stale_days: int = 90,
                      max_importance: float = 0.3,
                      now=None) -> List[MemoryRecord]:
        """归档走 inner（Chroma 向量会因 record.status='archived' 不进入搜索候选，
        阶段 2 可考虑按需清理向量；阶段 1 不处理）"""
        return self.inner.archive_stale(
            stale_days=stale_days, max_importance=max_importance, now=now,
        )

    def merge_similar(self, *, threshold: float = 0.85,
                      max_records: int = 500) -> List[MemoryRecord]:
        """合并相似记忆：代理 inner（inner 的默认实现会调 semantic_vectors 拿批量向量）"""
        return self.inner.merge_similar(threshold=threshold, max_records=max_records)

    # ===================== 资源释放 =====================

    def close(self) -> None:
        """不 close inner、不 close chroma_client——都由调用方负责生命周期
        （本类只包一层，不拥有传入的客户端）。"""
        self._collection = None
        return None

    # ===================== 可选：手动重建向量（与 SemanticMemoryStore 对齐的调试接口） =====================

    def rebuild_embeddings(self, batch_size: int = 20) -> int:
        """全量重建当前作用域所有记忆的向量（换 embedding 模型 / 向量库冷启动时用）

        阶段 1：_collection 为 None 时直接返回 0（无向量通道可重建）。
        阶段 2：批量 embed + _save_vec_chroma。
        """
        if self._collection_or_none() is None:
            return 0
        records = self.inner.all(include_inactive=True)
        ok_count = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [r.content for r in batch]
            vecs = self._safe_embed_documents_batch(texts)
            if vecs is None:
                continue
            for r, v in zip(batch, vecs):
                try:
                    self._save_vec_chroma(r, list(v))
                    ok_count += 1
                except Exception:  # noqa: BLE001
                    continue
        return ok_count
