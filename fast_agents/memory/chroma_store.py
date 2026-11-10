"""Chroma 向量库后端（B2 扩展 / v1.8.3 阶段 2）

包装任意 MemoryStoreBase 实例，将 embedding 向量存到外部 Chroma 向量库
（通过 ``fast_agents.database.ChromaClient`` 统一客户端访问），文本真相源
仍由 inner 负责——和 SemanticMemoryStore 平级，属于「换包装器」的另一选项，
符合 SemanticMemoryStore 类注释中「未来接 Chroma 换包装器即可」的设计意图。

阶段 2 范围：通过 database 模块的 ChromaClient 接入真实 Chroma
- chroma_client 必须是 ``fast_agents.database.ChromaClient`` 实例（或兼容
  ``VectorClientBase`` 的其他实现），由调用方创建并拥有生命周期；
- chroma_client=None 时整类降级为纯词法（直通 inner），默认安装零重型依赖；
- 即便传入客户端，Chroma 调用一律 try/except 降级，失败不阻塞主流程；
- 集合名按 user_id/session_id 自动生成，非法字符归一化（Chroma 只接受
  字母/数字/_/- 且 3-63 字符）；
- 所有 add / get / all / update / delete / clear / touch / revise / retract /
  search / semantic_vectors 行为与 SemanticMemoryStore 对齐。

使用方式：
    from fast_agents.database import ChromaClient
    from fast_agents.llm import LLM
    from fast_agents.memory import SQLiteMemoryStore, ChromaMemoryStore, MemoryManager

    llm = LLM()
    chroma = ChromaClient(path="./chroma_data")  # 本地持久化；也可 host/port 连远端
    inner = SQLiteMemoryStore("memories.db", user_id="u1")
    store = ChromaMemoryStore(inner, llm, chroma_client=chroma)
    manager = MemoryManager(llm=llm, long_term_store=store)
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

from .memory_store_base import MemoryStoreBase
from .models import MemoryRecord
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
    5. 打分权重、语义门槛与 SemanticMemoryStore 完全一致，保证换后端时排序口径不变；
    6. 覆写 semantic_batch_get()，让 search 走「批量从 Chroma 读向量缓存 → 只对
       缺失项批量 embed → 写回 Chroma」的批量路径，避免逐条 N 次 embed。
    """

    _DEFAULT_HYBRID_WEIGHT = 0.7     # 语义 0.7 / 词法 0.3（与 SemanticMemoryStore 一致）
    _SEMANTIC_FLOOR = 0.3            # 非偏好类的最低语义门槛（与 SemanticMemoryStore 一致）
    _DEFAULT_COLLECTION_PREFIX = "fa_mem"
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
            chroma_client: ``fast_agents.database.ChromaClient`` 实例（或兼容
                VectorClientBase 的向量客户端）；None 时整类降级为纯词法。
            collection_name: Chroma 集合名，默认按 user_id/session_id 自动生成；
                含非法字符时会被归一化（只保留字母/数字/_/-，截断到 63 字符）。
            hybrid_weight: 混合权重，默认 0.7（偏语义）。
            payload_support: 是否让 Chroma 携带正文 payload（Chroma/Qdrant=True，
                FAISS 等纯索引库=False）。
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
        # Chroma collection 名约束：3-63 字符、首尾字母数字、只允许字母/数字/_/-；
        # 默认名 f"fa_mem_{user_id}_{session_id}" 先做归一化，非法字符替换为下划线。
        raw_name = (
            collection_name
            or f"{self._DEFAULT_COLLECTION_PREFIX}_{inner.user_id}_{inner.session_id}"
        )
        self._collection_name = self._normalize_collection_name(raw_name)

        # 阶段 2：chroma_client 不为 None 时，懒加载调用 get_or_create_collection；
        # 任何异常（驱动缺失、连接失败、集合名非法等）都打 warning 并降级为纯词法。
        self._collection_ready: bool = False
        if chroma_client is not None:
            self._ensure_collection()

    # ===================== 集合名归一化 =====================

    @staticmethod
    def _normalize_collection_name(name: str) -> str:
        """把任意字符串归一化为合法 Chroma collection 名：
        - 只保留字母/数字/_/-；其他字符替换为下划线；
        - 长度 3~63，过短加 fa_mem_ 前缀补齐，过长截断；
        - 首尾保证是字母/数字（下划线/连字符开头结尾会替换/补字符）。
        """
        if not isinstance(name, str):
            name = str(name)
        # 非法字符替换为 _
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # 首尾不是字母/数字时替换为 x
        if cleaned and not re.match(r"[a-zA-Z0-9]", cleaned[0]):
            cleaned = "x" + cleaned[1:]
        if cleaned and not re.match(r"[a-zA-Z0-9]", cleaned[-1]):
            cleaned = cleaned[:-1] + "x"
        # 太短：加前缀
        if len(cleaned) < 3:
            cleaned = f"fa_mem_{cleaned}"
        # 太长：截断
        if len(cleaned) > 63:
            cleaned = cleaned[:63]
            if not re.match(r"[a-zA-Z0-9]", cleaned[-1]):
                cleaned = cleaned[:-1] + "x"
        return cleaned

    # ===================== Chroma 通道（阶段 2：真实调用 database.ChromaClient） =====================

    def _ensure_collection(self) -> bool:
        """确保集合存在（幂等）；失败打 warning 并返回 False（降级）。"""
        if self.chroma_client is None:
            return False
        if self._collection_ready:
            return True
        try:
            # 度量用 cosine（与 semantic_floor 余弦口径对齐）；dimension 暂不固定
            # 由 ChromaClient 在首次写入时自动推断。
            self.chroma_client.get_or_create_collection(self._collection_name, metric="cosine")
            self._collection_ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 初始化集合 {self._collection_name!r} 失败，降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return False

    def _collection_or_none(self):
        """返回可用的 chroma_client（集合已就绪），否则 None。"""
        if self.chroma_client is None:
            return None
        if not self._collection_ready and not self._ensure_collection():
            return None
        return self.chroma_client

    @staticmethod
    def _meta_for(record: MemoryRecord) -> dict:
        """构造 Chroma metadata（只放可 JSON 序列化、常用于过滤的字段）。"""
        return {
            "memory_id": record.memory_id,
            "user_id": record.user_id or "",
            "session_id": record.session_id or "",
            "kind": record.kind or "",
            "importance": float(record.importance or 0.0),
            "status": record.status or "active",
        }

    def _save_vec(self, record: MemoryRecord, vec: Optional[List[float]] = None) -> None:
        """写单条向量的标准钩子（转发到 Chroma）；供基类 semantic_batch_get_and_save 落盘批量补算结果。"""
        self._save_vec_chroma(record, vec)

    def _save_vec_chroma(self, record: MemoryRecord, vec: Optional[List[float]] = None) -> None:
        """写向量到 Chroma；``vec`` 已给出则直接落盘（不再重算 embedding），为 None 才现算；失败静默（调用方兜底）。"""
        client = self._collection_or_none()
        if client is None:
            return

        if vec is None:
            vecs = self._safe_embed_documents_batch([record.content])
            if vecs is None:
                return
            vec = vecs[0]
        docs = [record.content] if self.payload_support else None
        client.add(
            collection=self._collection_name,
            ids=[record.memory_id],
            embeddings=[list(vec)],
            documents=docs,
            metadatas=[self._meta_for(record)],
        )

    def _query_chroma(self, q_vec: List[float], top_k: int) -> List[str]:
        """从 Chroma 按向量召回 top_k 个 memory_id；仅返回 status=active 的活跃记忆；失败返回空列表。"""
        client = self._collection_or_none()
        if client is None:
            return []
        try:
            hits = client.search(
                collection=self._collection_name,
                query_embedding=list(q_vec),
                n_results=max(1, int(top_k)),
                where={"status": "active"},
                include_embeddings=False,
            )
            return [h["id"] for h in hits if isinstance(h, dict) and h.get("id")]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 向量检索失败，本次降级为纯词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return []

    def _update_vec_metadata(self, records) -> None:
        """同步一批记录的 metadata 到 Chroma（不清向量、不重算 embedding，只刷新 metadata.status 等字段）。
        用于 revise / retract / archive_stale / merge_similar / purge_expired 等仅改变状态的场景——
        设计原则：stale 向量保留在向量库中以便将来审计/回溯，但 metadata.status 必须实时同步，
        search / find_similar 通过 where={"status":"active"} 在向量库侧过滤掉 stale。

        实现说明：database.ChromaClient 对外只暴露 add（底层 upsert），没有"仅更新 metadata"
        的单独接口；所以这里先批量 get 回已有的 embedding / document，再和新 metadata 一起
        upsert 回去，避免覆盖丢失向量。
        """
        client = self._collection_or_none()
        if client is None:
            return
        recs = [r for r in (records or []) if isinstance(r, MemoryRecord)]
        if not recs:
            return
        ids = [r.memory_id for r in recs]
        try:
            existing = client.get(
                collection=self._collection_name,
                ids=ids,
                include_embeddings=True,
                include_documents=True,
                include_metadatas=False,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 读取已有向量失败，metadata 同步跳过（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return
        ex_by_id = {row.get("id"): row for row in existing or [] if row.get("id")}
        ups_ids, ups_embs, ups_docs, ups_metas = [], [], [], []
        for r in recs:
            row = ex_by_id.get(r.memory_id)
            if row is None:
                # 该条在 Chroma 里没向量（如写入时 embed 失败降级），跳过——将来 rebuild_embeddings 会补上。
                continue
            ups_ids.append(r.memory_id)
            ups_embs.append(list(row.get("embedding")) if row.get("embedding") is not None else None)
            ups_docs.append(row.get("document"))
            ups_metas.append(self._meta_for(r))
        if not ups_ids:
            return
        try:
            client.add(
                collection=self._collection_name,
                ids=ups_ids,
                embeddings=ups_embs,
                documents=ups_docs if self.payload_support else None,
                metadatas=ups_metas,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: metadata 同步失败（{exc!r}），where 过滤可能暂时漏过 stale",
                RuntimeWarning, stacklevel=2,
            )

    def _delete_vec_chroma(self, memory_id: str) -> None:
        """从 Chroma 删一条向量；失败静默。"""
        client = self._collection_or_none()
        if client is None:
            return
        try:
            client.delete(collection=self._collection_name, ids=[memory_id])
        except Exception:  # noqa: BLE001
            pass

    def _clear_vecs_chroma_scope(self) -> None:
        """清空当前作用域的 Chroma 向量：直接 drop 再重建集合。失败静默。"""
        client = self._collection_or_none()
        if client is None:
            return
        try:
            client.drop_collection(self._collection_name)
            self._collection_ready = False
            self._ensure_collection()
        except Exception:  # noqa: BLE001
            pass

    def _batch_get_vecs_chroma(self, memory_ids: List[str]) -> Dict[str, List[float]]:
        """批量从 Chroma 取已缓存的向量；读不到的 id 不进返回值。失败返回空 dict。"""
        client = self._collection_or_none()
        if client is None or not memory_ids:
            return {}
        # Chroma 单次 get 的 ids 过多时可能触发问题；这里简单分批以防万一
        result: Dict[str, List[float]] = {}
        batch = 200
        try:
            for start in range(0, len(memory_ids), batch):
                ids = memory_ids[start:start + batch]
                rows = client.get(
                    collection=self._collection_name,
                    ids=ids,
                    include_embeddings=True,
                    include_documents=False,
                    include_metadatas=False,
                )
                for row in rows:
                    rid = row.get("id")
                    emb = row.get("embedding")
                    if rid and emb is not None:
                        result[rid] = list(emb)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore: 批量读取向量缓存失败，本次退化为全量补算（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return {}
        return result

    # ===================== embedding 调用（失败降级，与 SemanticMemoryStore 一致） =====================
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
        try:
            self._save_vec_chroma(record)
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
        """更新 inner；content 变化则重算并回写 Chroma 向量；仅 status/metadata 变化则同步 metadata。

        注意：InMemoryStore.get 返回同一对象引用，外部持有 record 并直接改字段后再调 update 时，
        靠 inner.get 拿 old 做对比会拿到"已经被外部改完"的值，判断失效。
        因此这里用 Chroma 侧 metadata.status 作为旧值参照——Chroma 里的 status 只会在本类
        显式同步时更新，不会被外部引用意外改掉。
        """
        # 先在调用 inner.update 之前记一下 Chroma 侧旧的 status / content 是否已经有向量
        chroma_old_status = None
        chroma_has_vec = False
        client = self._collection_or_none()
        if client is not None:
            try:
                rows = client.get(
                    collection=self._collection_name, ids=[record.memory_id],
                    include_embeddings=True, include_documents=False, include_metadatas=True,
                )
                if rows:
                    chroma_has_vec = rows[0].get("embedding") is not None
                    md = rows[0].get("metadata") or {}
                    chroma_old_status = md.get("status") or "active"
            except Exception:  # noqa: BLE001
                pass
        old = self.inner.get(record.memory_id)
        # 快照：InMemoryStore 返回同一对象引用，外部可能先改字段再传 update，
        # 必须在 inner.update 之前记下旧 content/status 的副本，否则会拿到"已经被外部改完"的值。
        old_content = old.content if old is not None else None
        record = self.inner.update(record)
        new_status = (record.status or "active")

        content_changed = old is not None and old_content != record.content

        # status_changed：以 Chroma 侧 metadata 为旧值参照，避免 InMemoryStore 引用语义导致的判断失效；
        # Chroma 里还没这条（没向量）时不判 status 变化（交给 _save_vec_chroma 写入时直接带新 status）。
        status_changed = (
            chroma_has_vec
            and chroma_old_status is not None
            and chroma_old_status != new_status
        )

        if content_changed:
            # 内容变了：显式"先删旧向量、再加新向量"两步走，与关系型 inner.update 的更新语义对齐，
            # 避免依赖 ChromaClient.add 内部的隐式 upsert（它一旦收紧语义变真 add，会抛 DuplicateIDError），
            # 同时确保旧 document / metadata 不会因 payload 不全而残留。
            try:
                self._delete_vec_chroma(record.memory_id)
            except Exception:  # noqa: BLE001
                pass  # 旧向量删不掉（比如本来就没有）不阻塞新向量写入
            try:
                self._save_vec_chroma(record)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"ChromaMemoryStore: 更新向量失败（{exc!r}）",
                              RuntimeWarning, stacklevel=2)
        elif status_changed:
            # 不删向量、不重算 embedding，只把 metadata.status 同步到 Chroma，
            # 让 search / find_similar 的 where 过滤立即生效。
            self._update_vec_metadata([record])
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

    # ===================== 语义批量缓存 =====================

    def semantic_batch_get(self, records: List[MemoryRecord]) -> List[Optional[List[float]]]:
        """批量从 Chroma 读取已缓存的向量，顺序与 records 一一对应。
        读不到的位置填 None，由基类 semantic_batch_get_and_save() 自动批量补算并 _save_vec。
        """
        if not records:
            return []
        ids = [r.memory_id for r in records]
        cached = self._batch_get_vecs_chroma(ids)
        return [cached.get(rid) for rid in ids]

    def embedding_stats(self) -> dict:
        """返回向量通道健康统计（供调试/健康日志使用）。"""
        client = self._collection_or_none()
        if client is None:
            # 无可用 chroma_client，语义通道关闭
            return {
                "available": False,
                "backend": "chroma",
                "chroma_connected": False,
            }
        cached_total = 0
        dim = None
        try:
            cached_total = client.count(self._collection_name)
        except Exception:  # noqa: BLE001
            cached_total = 0
        # 从 Chroma 拿一条样本测维度，避免依赖 llm.embedding_dimension 不一定存在
        if cached_total > 0:
            try:
                sample = client.get(
                    collection=self._collection_name,
                    ids=None,
                    limit=1,
                    include_embeddings=True,
                    include_documents=False,
                    include_metadatas=False,
                )
                if sample:
                    emb = sample[0].get("embedding")
                    if isinstance(emb, (list, tuple)) and emb:
                        dim = len(emb)
            except Exception:  # noqa: BLE001
                dim = None
        return {
            "available": True,
            "backend": "chroma",
            "chroma_connected": True,
            "collection": self._collection_name,
            "model": getattr(self.llm, "embedding_model", "unknown"),
            "dimension": dim,
            "cached_total": cached_total,
            "total_records": len(self.inner.all(include_inactive=True)),
        }

    # ===================== B13：以 Chroma ANN 为主的向量冲突检测 =====================

    def find_similar(self, content: str, *, limit: int = 5,
                     threshold: float = 0.85) -> List[tuple]:
        """用 Chroma 近邻检索召回与 ``content`` 最相似的已有记忆（B13 冲突检测主入口）。

        优势：直接走 ANN 索引，O(log n) 复杂度，百万级记忆库也能快速检测冲突，
        不再像 SemanticMemoryStore 的兜底实现那样两两比对 cosine。
        - Chroma 不可用 / embed 失败 → 返回空列表，让上层回退词法；
        - Chroma 返回 cosine distance（0=最相似），转 similarity = 1 - distance；
        - 过滤掉相似度低于 threshold 的条目；
        - 返回 [(record, similarity), ...]，按相似度降序。
        """
        if not content or not content.strip():
            return []
        client = self._collection_or_none()
        if client is None:
            return []
        q_vec = self._safe_embed_query(content.strip())
        if q_vec is None:
            return []
        try:
            # 多取一些候选再按 threshold 过滤，避免 threshold 较严时拿不到候选
            n = max(int(limit) * self._CHROMA_TOP_K_MULTIPLIER, 10)
            hits = client.search(
                collection=self._collection_name,
                query_embedding=list(q_vec),
                n_results=n,
                where={"status": "active"},
                include_embeddings=False,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ChromaMemoryStore.find_similar: 向量检索失败，本次退化为词法（{exc!r}）",
                RuntimeWarning, stacklevel=2,
            )
            return []
        if not hits:
            return []
        # 把 Chroma 返回的 memory_id 映射回 MemoryRecord（只在活跃、未过期的记录里找）
        by_id = {
            r.memory_id: r for r in self.inner.all(include_inactive=False)
            if not r.is_expired()
        }
        out: List[tuple] = []
        for h in hits:
            rid = h.get("id")
            rec = by_id.get(rid)
            if rec is None:
                continue
            dist = h.get("distance")
            # Chroma cosine metric 下 distance ∈ [0, 2]，similarity = 1 - distance
            if not isinstance(dist, (int, float)):
                continue
            sim = max(0.0, min(1.0, 1.0 - float(dist)))
            if sim >= threshold:
                out.append((rec, sim))
        out.sort(key=lambda x: (-x[1], -x[0].importance))
        return out[:max(1, int(limit))]

    # ===================== 语义 search =====================

    def search(self, query: str, limit: int = 5, *,
               include_preferences: bool = True) -> List[MemoryRecord]:
        if not query or not query.strip():
            return self.inner.search(query, limit, include_preferences=include_preferences)

        q_vec = self._safe_embed_query(query)
        if q_vec is None:
            return self.inner.search(query, limit, include_preferences=include_preferences)

        client = self._collection_or_none()
        use_semantic = client is not None

        # 从 Chroma 向量召回 top 候选；client 不可用时返回空，records=inner 全量
        chroma_ids = self._query_chroma(
            q_vec, top_k=limit * self._CHROMA_TOP_K_MULTIPLIER,
        ) if use_semantic else []
        records = self.inner.all(include_inactive=False)
        # B12：rank 之前先过滤已到期的 active 记忆
        records = [r for r in records if not r.is_expired()]

        vec_by_id: Dict[str, List[float]] = {}
        if use_semantic:
            # 走基类 semantic_batch_get_and_save：
            # 1) 从 Chroma 批量读已缓存 2) 缺失项批量 embed 3) 通过 _save_vec 写回 Chroma
            vecs = self.semantic_batch_get_and_save(records)
            for r, v in zip(records, vecs):
                if v is not None:
                    vec_by_id[r.memory_id] = list(v)

        if use_semantic:
            # B13 第一步【chroma_store】stale 不删向量，靠 where={"status":"active"} 过滤。
            # 合并规则配合 where：Chroma 里已有向量的记录以 Chroma 召回为准
            # （被 where 挡掉的 stale 不再从 inner 兜底补回，避免绕过向量库侧状态过滤）；
            # 只有「Chroma 里还没有向量」的记录（新写入还没补算、历史降级写入）才从 inner 追加，
            # 保留 inner 兜底偏好/最近条目的能力。
            # 注意：即使 chroma_ids 为空（可能是 where 把所有 active 都挡掉了），
            # 也必须做这个过滤——否则 stale 会通过 inner.all 绕过 where 进排序。
            id_order = list(chroma_ids)
            seen = set(chroma_ids)
            for r in records:
                if r.memory_id in seen:
                    continue
                if r.memory_id in vec_by_id:
                    # 已有向量但未被 Chroma 召回：要么 where 过滤（stale），要么相似度不够，
                    # 都不应被 inner 兜底强行塞回排序。
                    continue
                id_order.append(r.memory_id)
                seen.add(r.memory_id)
            by_id = {r.memory_id: r for r in records}
            records = [by_id[mid] for mid in id_order if mid in by_id]

        def semantic_getter(r: MemoryRecord) -> Optional[float]:
            v = vec_by_id.get(r.memory_id)
            if v is None:
                return None
            return cosine_similarity(q_vec, v)
        return rank_records(
            query, records, limit,
            include_preferences=include_preferences,
            semantic_getter=semantic_getter if use_semantic else None,
            hybrid_weight=self.hybrid_weight,
            semantic_floor=self._SEMANTIC_FLOOR,
        )

    # ===================== B5：合并相似度向量通道 =====================

    def semantic_vectors(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量算向量，供上层做「记忆合并」；失败降级 None。

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
        """修订记忆：代理 inner；新版本写入新向量；老版本保留向量但同步 metadata.status=superseded，
        让 search / find_similar 的 where 过滤将其排除（不清向量，便于将来审计/回溯）。"""
        old = self.inner.get(memory_id)
        revised = self.inner.revise(memory_id, content, source=source)
        if revised is not None:
            try:
                # 新版本（新 memory_id）写入新向量
                self._save_vec_chroma(revised)
            except Exception:  # noqa: BLE001
                pass
            # 老版本保留向量，仅同步 status=superseded 到 metadata
            if old is not None and old.memory_id != revised.memory_id:
                old_reloaded = self.inner.get(old.memory_id)
                if old_reloaded is not None:
                    self._update_vec_metadata([old_reloaded])
        return revised

    def retract(self, memory_id: str) -> MemoryRecord:
        """撤回记忆：代理 inner；不清向量，仅同步 metadata.status=retracted，由 where 过滤排除。"""
        retracted = self.inner.retract(memory_id)
        if retracted is not None:
            self._update_vec_metadata([retracted])
        return retracted

    def archive_stale(self, *, stale_days: int = 90,
                      max_importance: float = 0.3,
                      now=None) -> List[MemoryRecord]:
        """归档走 inner；归档后把被归档记录的 metadata.status 同步到 Chroma。"""
        archived = self.inner.archive_stale(
            stale_days=stale_days, max_importance=max_importance, now=now,
        )
        self._update_vec_metadata(archived)
        return archived

    def purge_expired(self, *, now=None) -> List[MemoryRecord]:
        """清理到期记忆：代理 inner；到期后把被清理记录的 metadata.status 同步到 Chroma。"""
        purged = self.inner.purge_expired(now=now)
        self._update_vec_metadata(purged)
        return purged

    def merge_similar(self, *, threshold: float = 0.85,
                      max_records: int = 500) -> List[MemoryRecord]:
        """合并相似记忆：代理 inner（合并逻辑在 inner 的 merge_similar；向量能力取决于 inner 后端）；
        合并后被 superseded 的记录同步 metadata 到 Chroma。"""
        # 合并前记住要被 superseded 的 id（基类 merge_similar 会把它们 status 改为 superseded）
        before = {r.memory_id: r for r in self.inner.all(include_inactive=True)}
        kept = self.inner.merge_similar(threshold=threshold, max_records=max_records)
        after = self.inner.all(include_inactive=True)
        changed = [r for r in after
                   if r.memory_id in before
                   and (before[r.memory_id].status or "active") != (r.status or "active")]
        self._update_vec_metadata(changed)
        return kept

    # ===================== 资源释放 =====================

    def close(self) -> None:
        """不 close inner、不 close chroma_client——都由调用方负责生命周期
        （本类只包一层，不拥有传入的客户端）。"""
        self._collection_ready = False
        return None

    # ===================== 可选：手动重建向量（与 SemanticMemoryStore 对齐的调试接口） =====================

    def rebuild_embeddings(self, batch_size: int = 20) -> int:
        """全量重建当前作用域所有记忆的向量（换 embedding 模型 / 向量库冷启动时用）

        chroma_client=None 时直接返回 0（无向量通道可重建）。
        """
        if self._collection_or_none() is None:
            return 0
        records = self.inner.all(include_inactive=True)
        ok_count = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [r.content for r in batch]
            try:
                vecs = self._safe_embed_documents_batch(texts)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"ChromaMemoryStore: rebuild 第 {i} 批失败，跳过（{exc!r}）",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            for r, vec in zip(batch, vecs):
                if not vec:
                    continue
                try:
                    self._save_vec_chroma(r, vec)
                    ok_count += 1
                except Exception:  # noqa: BLE001
                    continue
        return ok_count
