"""记忆存储后端抽象层

之所以要抽出这一层：
- 短期记忆放内存就够（单次会话），长期记忆必须落盘；
- 落盘可以有多种实现（纯 Python 文件 / SQLite / 向量库），
  只要实现同一个接口，MemoryManager 就不用改。
这和你 tools 模块里 ToolRegistry 统一管理 ToolBase 的思路是一样的。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from .schema import MemoryRecord


class MemoryStore(ABC):
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
    def search(self, query: str, limit: int = 5, *, include_preferences: bool = True) -> List[MemoryRecord]:
        """按查询返回相关记忆；query 为空时返回最近的记录"""

    @abstractmethod
    def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新一条已有记忆（按 memory_id 覆盖）"""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除一条记忆"""

    @abstractmethod
    def clear(self) -> None:
        """清空当前作用域的全部记忆"""

    # ---------- 以下两个是带默认实现的高级操作 ----------
    # 所有后端都支持「修订」和「撤回」，语义与 SQLite 版一致：
    # revise = 旧记录标记 superseded + 追加新记录；retract = 只改状态保留历史。

    def revise(self, memory_id: str, content: str, source: str) -> MemoryRecord:
        """修订一条记忆，保留旧版本作为历史"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status != "active":
            raise ValueError("只能修订有效记忆，请先重新读取")
        old.status = "superseded"
        old.updated_at = datetime.now()
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
            old.updated_at = datetime.now()
            self.update(old)
        return old

    def close(self) -> None:
        """释放资源（子类按需实现）"""
        return None
