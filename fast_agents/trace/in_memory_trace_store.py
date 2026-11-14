
from typing import Any, Iterator, List, Optional

from .store_base import TraceStoreBase
from .models import SpanKind, SpanRecord, _now_ts


class InMemoryTraceStore(TraceStoreBase):
    """纯内存存储，进程退出即丢失（出入两端复制）"""

    def __init__(self):
        self._records: List[SpanRecord] = []

    # ---------- TraceStoreBase 接口 ----------
    def write(self, span: SpanRecord) -> None:
        """写入一条 Span。任何异常都会被吞掉（打点不能阻塞业务）。"""
        try:
            self._records.append(span.model_copy(deep=True))
        except Exception:
            pass

    def read(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SpanRecord]:
        """按条件过滤读取 Span 列表（按写入顺序返回）。"""
        results: List[SpanRecord] = []
        for span in self._records:
            if session_id is not None and span.session_id != session_id:
                continue
            if trace_id is not None and span.trace_id != trace_id:
                continue
            if kind is not None and span.kind != kind:
                continue
            try:
                results.append(span.model_copy(deep=True))
            except Exception:
                continue
            if limit is not None and len(results) >= limit:
                break
        return results

    def clear(self) -> None:
        self._records.clear()

    def close(self) -> None:
        """内存模式，没啥可关闭的"""
