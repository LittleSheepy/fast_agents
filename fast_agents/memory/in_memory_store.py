"""内存版长期记忆后端

用途：单元测试、临时会话、不想落盘的场景。
接口与 SQLiteMemoryStore 完全一致，可以直接互换。

search 不在此处实现，由 MemoryStoreBase 通过 all() + scoring.rank_records
提供默认实现，保证与其它后端打分口径一致（B7）。
"""

from typing import List, Optional

from .memory_store_base import MemoryStoreBase
from .schema import MemoryRecord


class InMemoryStore(MemoryStoreBase):
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

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if record.memory_id not in self._records:
            raise KeyError(record.memory_id)
        self._records[record.memory_id] = record
        return record

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def clear(self) -> None:
        self._records.clear()
