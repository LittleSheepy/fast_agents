"""内存版长期记忆后端

用途：单元测试、临时会话、不想落盘的场景。
接口与 SQLiteMemoryStore 完全一致，可以直接互换。
"""

from typing import List, Optional

from .base import MemoryStore
from .schema import MemoryRecord
from .scoring import lexical_score


class InMemoryStore(MemoryStore):
    """纯内存存储，进程退出即丢失"""

    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._records.get(memory_id)

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        records = list(self._records.values())
        if not include_inactive:
            records = [r for r in records if r.status == "active"]
        return sorted(records, key=lambda r: r.updated_at, reverse=True)

    def search(self, query: str, limit: int = 5, *, include_preferences: bool = True) -> List[MemoryRecord]:
        records = self.all()
        if not query or not query.strip():
            return records[:limit]
        scored = []
        for r in records:
            if r.kind == "preference" and include_preferences:
                base = r.importance * 0.5 + lexical_score(query, r.content) * 0.5
                scored.append((base, r))
                continue
            score = lexical_score(query, r.content)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], -x[1].importance))
        return [r for _, r in scored[:limit]]

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if record.memory_id not in self._records:
            raise KeyError(record.memory_id)
        self._records[record.memory_id] = record
        return record

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def clear(self) -> None:
        self._records.clear()
