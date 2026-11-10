"""内存版长期记忆后端

用途：单元测试、临时会话、不想落盘的场景。
接口与 SQLiteMemoryStore 完全一致，可以直接互换。

search 不在此处实现，由 MemoryStoreBase 通过 all() + scoring.rank_records
提供默认实现，保证与其它后端打分口径一致（B7）。

引用语义（bug 修复）：出入两端一律返回副本，对外表现为「值语义」。
SQLiteMemoryStore.get() 每次从数据库行新建对象返回，调用方拿到的是快照，
改它不会影响库里；而早期 InMemoryStore 直接把内部对象引用交出去，
导致「get 出来改字段再 update」这类写法拿到的两个对象其实是同一个，
新旧内容对比恒为「未变化」（semantic_store / chroma_store 的 update
正是靠 old = inner.get() 取旧 content 来判断要不要重算向量），
向量重算因此被静默跳过。深拷贝后内存后端与 SQLite 后端语义一致。
"""

from typing import List, Optional

from fast_agents.utils.time_utils import to_cst
from .memory_store_base import MemoryStoreBase
from .models import MemoryRecord


class InMemoryStore(MemoryStoreBase):
    """纯内存存储，进程退出即丢失（出入两端复制，与 SQLiteMemoryStore 同为值语义）"""

    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> MemoryRecord:
        # 存副本：add 之后调用方再改 record，也不会污染库内数据
        self._records[record.memory_id] = record.model_copy(deep=True)
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        # 返回副本：与 SQLite 每次新对象的语义一致，调用方改它不会回写库内
        record = self._records.get(memory_id)
        return record.model_copy(deep=True) if record is not None else None

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        records = list(self._records.values())
        if not include_inactive:
            records = [r for r in records if r.status == "active"]
        records.sort(key=lambda r: to_cst(r.updated_at), reverse=True)
        # 返回副本：批量读取后就地改字段（如 archive_stale）不会绕过 update 直接改库内对象
        return [r.model_copy(deep=True) for r in records]

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if record.memory_id not in self._records:
            raise KeyError(record.memory_id)
        # 存副本：避免调用方在 update 之后继续改 record 污染库内数据
        self._records[record.memory_id] = record.model_copy(deep=True)
        return record

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def clear(self) -> None:
        self._records.clear()
