"""记忆存储后端抽象层

之所以要抽出这一层：
- 短期记忆放内存就够（单次会话），长期记忆必须落盘；
- 落盘可以有多种实现（纯 Python 文件 / SQLite / 向量库），
  只要实现同一个接口，MemoryManager 就不用改。
这和你 tools 模块里 ToolRegistry 统一管理 ToolBase 的思路是一样的。

设计约定：
- add / get / all / update / delete / clear 是原子原语（必须由子类实现）；
- search 默认通过 all() + scoring.rank_records 实现，子类无需重复写打分逻辑；
- revise / retract / touch / semantic_vectors / archive_stale / merge_similar
  是高层组合操作，已有默认实现，子类按需覆盖。
B5 新增：
- touch: 命中后记账，累计 access_count、刷新 last_accessed_at；
- semantic_vectors: 批量返回文本向量（给合并相似度用，无向量后端返回 None）；
- archive_stale: 把「低重要度 + 久未命中」的记忆状态改为 archived（退出检索，不删数据）；
- merge_similar: 两两比对相似度，保留最重要的一条，其余标记 superseded。
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

from fast_agents.utils.time_utils import to_cst, cstnow
from .models import MemoryRecord
from .scoring import (
    cosine_similarity, lexical_terms, merge_similarity, rank_records,
)


def _lexical_jaccard(a: str, b: str) -> float:
    """对称词法相似度（Jaccard）：两个文本的词项交集 / 并集

    用于合并判定（要求对称），口径与 MemoryManager._is_duplicate 完全一致。
    """
    ta, tb = set(lexical_terms(a)), set(lexical_terms(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class MemoryStoreBase(ABC):
    """记忆存储后端接口"""

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入一条记忆"""

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """按 id 读取"""

    @abstractmethod
    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        """读取全部记录（按更新时间倒序）"""

    @abstractmethod
    def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新一条已有记忆（按 memory_id 覆盖）"""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除一条记忆"""

    @abstractmethod
    def clear(self) -> None:
        """清空当前作用域的全部记忆"""

    # ---------- 以下是带默认实现的高级操作 ----------
    # 规则统一：能用 add/get/all/update/delete/clear 组合出来的操作都在这里写默认实现，
    # 子类不需要重复造轮子；有更高效实现（例如 SQL 侧批量 UPDATE）时再 override。

    def search(self, query: str, limit: int = 5, *, include_preferences: bool = True) -> List[MemoryRecord]:
        """按查询返回相关记忆；query 为空时返回最近的记录

        v1.3.0（B7）起默认实现在 Base：通过 all() 拉回记录，再用
        scoring.rank_records 走统一打分公式与排序。子类若要做数据库侧
        排序 / 向量检索 / 索引过滤，可直接覆盖本方法。

        v1.8.1（B12）起在 rank 之前先过滤掉已到期的活跃记忆——到期记录已被
        purge_expired 标记为 retracted，但两次 compact() 之间仍可能以 active
        状态存在，必须在检索路径上再挡一道，避免偏好类强制召回漏过滤。
        """
        candidates = [r for r in self.all() if not r.is_expired()]
        return rank_records(query, candidates, limit, include_preferences=include_preferences)

    def touch(self, memory_ids: Iterable[str]) -> int:
        """记录一次「命中」：自增 access_count、刷新 last_accessed_at

        B5 新增。检索到某条记忆后调用本方法记账，衰减归档才有依据。
        默认实现走 get + update；子类可覆盖以省掉一次读（例如 SQLite 直接
        UPDATE ... SET access_count = access_count + 1）。
        **不要刷新 updated_at**：updated_at 参与时间衰减打分，命中一次就把
        新鲜度刷满会让衰减失效（SQLite 子类已单独处理）。
        返回实际更新的条数。
        """
        count = 0
        for memory_id in memory_ids:
            record = self.get(memory_id)
            if record is None:
                continue
            record.access_count += 1
            record.last_accessed_at = cstnow()
            self.update(record)
            count += 1
        return count

    def semantic_vectors(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量算文本向量，供「合并相似度」等高级操作使用

        B5 新增可选能力钩子：
        - 默认返回 None，表示后端**没有向量能力**，调用方应退化为纯词法；
        - 有语义能力的后端（如 SemanticMemoryStore）覆盖本方法，
          返回与 texts 等长的向量列表；批量失败时返回 None（不要抛异常）。
        """
        return None

    def semantic_batch_get(self, records: List["MemoryRecord"]) -> List[Optional[List[float]]]:
        """批量读取已缓存的向量，顺序与 records 一一对应

        - 默认返回全 None（无缓存能力），由调用方走语义 fallback；
        - 有本地向量缓存的后端（SemanticMemoryStore / ChromaMemoryStore）
          覆盖本方法，直接从 SQLite 表 / 内存 dict / Chroma 里批量读出已缓存的向量，
          避免 search / merge_similar 逐条懒加载触发 N 次 embed；
        - 返回值长度必须与 records 相等；读不到的位置填 None（由上层决定是否批量补算）。
        """
        return [None] * len(records)

    def embedding_stats(self) -> dict:
        """向量能力健康统计钩子（默认：无向量能力）。

        返回至少包含 ``{"available": bool}``；可用时由子类补充
        backend / dimension / model / cached_total / total_records 等字段。
        上层（例如打健康日志、调试面板）可按需读取，不存在调用方强依赖，
        缺字段也不会阻断主流程。
        """
        return {"available": False}

    def find_similar(self, content: str, *, limit: int = 5,
                     threshold: float = 0.85) -> List[tuple]:
        """写入前向量冲突检测钩子（B13）。

        - 默认返回空列表，表示"后端无向量能力，请调用方回退词法"；
        - ChromaMemoryStore 直接走 ANN 近邻（O(log n)）；SemanticMemoryStore
          提供基于本地向量缓存的批量 cosine 兜底实现；
        - 返回 ``[(MemoryRecord, similarity), ...]``，按相似度降序、已按 threshold 过滤。
        """
        return []

    def semantic_batch_get_and_save(self, records: List["MemoryRecord"]) -> List[Optional[List[float]]]:
        """批量读取已缓存的向量,并对缺失的条目批量补算，顺序与 records 一一对应,

        1) 优先批量取本地缓存
        2) 只对缺失的条目批量补算 embedding（一次 LLM 调用）
        3) 补算出来的新向量立即写回本地缓存，后续 search / merge_similar 直接命中
        """
        # 1) 优先批量取本地缓存
        vecs = list(self.semantic_batch_get(records) or [None] * len(records))
        if len(vecs) < len(records):
            vecs.extend([None] * (len(records) - len(vecs)))

        missing_indices: List[int] = [i for i, v in enumerate(vecs) if v is None]

        # 2) 只对缺失的条目批量补算 embedding（一次 LLM 调用）
        if missing_indices:
            missing_records = [records[i] for i in missing_indices]
            missing_texts = [r.content for r in missing_records]
            missing_vecs = self.semantic_vectors(missing_texts)
            if missing_vecs and len(missing_vecs) == len(missing_texts):
                for idx, rec, v in zip(missing_indices, missing_records, missing_vecs):
                    if v:
                        vec = list(v)
                        vecs[idx] = vec
                        # 3) 补算后立即落盘缓存（把刚算好的 vec 一起传下去，避免二次 embed），下次直接命中
                        try:
                            self._save_vec(rec, vec)
                        except Exception:  # noqa: BLE001
                            # 存不进缓存不影响本次检索
                            pass
        return vecs

    def archive_stale(
        self, *, max_importance: float = 0.3, stale_days: int = 90,
        now: Optional[datetime] = None,
    ) -> List[MemoryRecord]:
        """把「低重要度 + 久未命中」的记忆批量归档（status → archived）

        B5 新增，是「记忆衰减」的落地方式：
        - 判定：importance < max_importance 且「距最后一次被使用」≥ stale_days；
        - 「最后一次被使用」= last_accessed_at，从未命中过则退回 updated_at
          （修订过说明这条还活着，不该按创建时间算）；
        - 归档 = 改 status，不删数据：all() 默认不再返回它（不参与检索），
          all(include_inactive=True) 仍可审计；
        - 返回被归档的记录列表。
        """
        moment = to_cst(now) or cstnow()
        deadline = moment - timedelta(days=max(0, int(stale_days)))
        archived: List[MemoryRecord] = []
        for record in self.all():
            if record.importance >= max_importance:
                continue
            last_used = to_cst(record.last_accessed_at or record.updated_at)
            if last_used > deadline:
                continue
            record.status = "archived"
            record.metadata["archived_at"] = moment.isoformat()
            self.update(record)
            archived.append(record)
        return archived

    def purge_expired(self, *, now: Optional[datetime] = None) -> List[MemoryRecord]:
        """把已到期的记忆批量软删（status → retracted）（B12）

        与 archive_stale 的区别：
        - archive_stale 判定依据是「低重要度 + 久未命中」（被动衰减）；
        - purge_expired 判定依据是「expires_at <= now」（写入时约定的硬过期时刻），
          与 importance / access_count 无关——重要度 0.9 的「明早 9 点提交 PR」
          过了 9 点就该退出检索，不能等 90 天衰减。

        行为约定：
        - 只处理 status == 'active' 的记录（幂等，重复调用无副作用）；
        - 到期 = 改 status 为 'retracted'（不是 archived），与 retract() 路径一致，
          物理删除只有显式 delete()；
        - 在 metadata["expired_at"] 记下触发时间，保留审计链；
        - 返回被清理的记录列表。

        注：本方法不负责清理向量（memory_embeddings / chroma）——主记录转 retracted
        后 search() 不会再召回，占位可接受；若要彻底干净可由包装器在调用后补充。
        """
        moment = to_cst(now) or cstnow()
        purged: List[MemoryRecord] = []
        for record in self.all():
            if not record.is_expired(now=moment):
                continue
            record.status = "retracted"
            record.metadata["expired_at"] = moment.isoformat()
            self.update(record)
            purged.append(record)
        return purged

    def merge_similar(
        self, *, threshold: float = 0.85, max_records: int = 500
    ) -> List[MemoryRecord]:
        """把高度相似的重复记忆合并：保留最重要的一条，其余转 superseded

        B5 新增，是「记忆合并」的落地方式：
        - 相似度 = 语义 × 0.7 + 词法 × 0.3（scoring.merge_similarity），
          后端无向量能力时自动退化为纯词法；
        - 语义向量先批量命中本地缓存（semantic_batch_get_and_save），只对缺失项
          补算一次 embedding 并落盘，避免每次合并都全量重算向量；
        - 先按 (重要性降序, 创建时间升序) 排序，让「更重要 / 更早」的那条当主记录，
          两条相似记忆永远保留重要的、舍弃冗余的；
        - 被合并的记录 status → superseded，并在 metadata["merged_into"]
          记下主记录 id（supersedes 是单值、语义相反，不适合多值合并，故不动它）；
        - 主记录本身不做任何修改，避免合并动作污染已有内容与 updated_at；
        - max_records 是复杂度保护：两两比较是 O(n²)，超过上限时只处理最近更新的
          这一批，避免一次维护在大库上卡住（可按需调大）。
        - 返回被合并掉的记录列表。
        """
        all_records = self.all()
        if len(all_records) < 2:
            return []
        if len(all_records) > max_records:
            all_records = sorted(all_records, key=lambda r: to_cst(r.updated_at), reverse=True)[:max_records]
        records = sorted(all_records, key=lambda r: (-r.importance, to_cst(r.created_at)))

        # 先批量读本地缓存，仅对缺失项补算并落盘（不再每次全量重算向量）
        vecs = self.semantic_batch_get_and_save(records)
        vec_by_id = {}
        if vecs and len(vecs) == len(records):
            vec_by_id = {r.memory_id: v for r, v in zip(records, vecs) if v}

        merged: List[MemoryRecord] = []
        consumed: set = set()
        for i, main in enumerate(records):
            if main.memory_id in consumed:
                continue
            for other in records[i + 1:]:
                if other.memory_id in consumed:
                    continue
                lex = _lexical_jaccard(main.content, other.content)
                va = vec_by_id.get(main.memory_id)
                vb = vec_by_id.get(other.memory_id)
                sem = cosine_similarity(va, vb) if va is not None and vb is not None else None
                sim = merge_similarity(lex, sem)
                if sim < threshold:
                    continue
                other.status = "superseded"
                other.metadata["merged_into"] = main.memory_id
                self.update(other)
                consumed.add(other.memory_id)
                merged.append(other)
        return merged

    def revise(self, memory_id: str, content: str, source: str) -> MemoryRecord:
        """修订一条记忆，保留旧版本作为历史"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status != "active":
            raise ValueError("只能修订有效记忆，请先重新读取")
        old.status = "superseded"
        old.updated_at = cstnow()
        self.update(old)
        new = MemoryRecord(
            user_id=old.user_id,
            session_id=old.session_id,
            content=content,
            source=source,
            kind=old.kind,
            importance=old.importance,
            supersedes=memory_id,
            metadata=dict(old.metadata),
        )
        return self.add(new)

    def retract(self, memory_id: str) -> MemoryRecord:
        """撤回一条记忆（改状态、留历史、幂等）"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status == "active":
            old.status = "retracted"
            old.updated_at = cstnow()
            self.update(old)
        return old

    def close(self) -> None:
        """释放资源（子类按需实现）"""
        return None
