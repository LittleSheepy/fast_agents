"""trace 存储抽象基类。

对齐 ``fast_agents.database.DatabaseClientBase`` 的抽象风格，
为将来 G1 之后切 SQLite / 远端上报后端预留接口；当前 v1.0.0 唯一实现为 ``JsonlStore``。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

from .models import SpanRecord


class TraceStoreBase(ABC):
    """Trace 存储后端抽象基类。所有后端必须实现 write / read / close。"""

    @abstractmethod
    def write(self, span: SpanRecord) -> None:
        """写入一条 Span 记录。实现方必须保证异常不向上抛（打点不能阻塞业务）。"""

    @abstractmethod
    def read(self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SpanRecord]:
        """按条件读取 Span 列表；用于回放。

        Args:
            session_id: 按会话过滤。
            trace_id: 按整条 trace 过滤。
            kind: 按 Span 类型过滤（``llm/tool/decision/agent/custom``）。
            limit: 返回最大条数（按写入顺序，新的在前或在后由实现决定；JsonlStore 按写入顺序返回）。
        """

    def iter_records(self) -> Iterator[SpanRecord]:
        """遍历所有记录，默认基于 read() 实现，子类可覆盖为流式读取。"""
        for r in self.read():
            yield r

    @abstractmethod
    def close(self) -> None:
        """释放资源（关闭文件句柄 / 数据库连接）。多次调用安全。"""

    def flush(self) -> None:
        """将缓冲刷入持久化介质；默认 no-op，需要 fsync 的后端可覆盖。"""

    def __del__(self) -> None:
        """析构时尝试关闭连接。"""
        try:
            self.close()
        except Exception:
            pass



__all__ = ["TraceStoreBase"]
