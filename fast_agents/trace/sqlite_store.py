"""SQLiteStore —— 基于 ``fast_agents.database.SQLiteClient`` 的结构化 Trace 落盘后端。

设计参考 OpenTelemetry 的 Span 持久化（一条 Span → 表里一行）与 LangChain Callback
的 trace 落盘思路，但只借鉴概念、不引外部依赖：底层直接复用框架自带的
``database.SQLiteClient``（自动 WAL + busy_timeout + 参数化查询 + 声明式幂等建表），
上层仍实现 ``TraceStoreBase`` 的三个核心方法，可与 ``JsonlStore`` 无痛互换。

与 ``JsonlStore`` 的差异：
- **查询能力**：jsonl 只能逐行扫描，SQLite 走 ``select_where`` 谓词下推，
  ``trace_id`` / ``session_id`` / ``kind`` 均建索引，大 trace 量下过滤与回放快得多；
- **并发写**：jsonl 是单文件顺序 append，多进程并发写易损坏；SQLite 走 WAL +
  busy_timeout，多进程读写更稳；
- **落盘形式**：一条 Span = ``trace_spans`` 表里一行，``input`` / ``output`` /
  ``metadata`` 以 JSON 文本列存储（复用 jsonl 的 ``_safe_serialize`` 口径，读写行为一致）；
- **写入顺序**：用自增主键 ``id`` 保证与 ``JsonlStore`` 一致的「按写入顺序返回」语义。

零新增依赖（标准库 sqlite3 + 框架已有 pydantic）；打点通道所有异常一律吞掉，
绝不阻塞主业务（对齐 AGENTS.md R6）。
"""
from __future__ import annotations

import json
import threading
from typing import Any, Iterator, List, Optional

from ..database import (
    Column,
    ColumnType,
    Filter,
    FilterOp,
    Index,
    Query,
    Schema,
    SortOrder,
    SQLiteClient,
)
from .jsonl_store import _safe_serialize
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase


def _as_str(value: Any) -> Optional[str]:
    """把枚举（SpanKind / SpanStatus）或其值统一成字符串；None 原样返回。"""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _dump(value: Any) -> str:
    """把任意对象序列化成 JSON 文本列（口径与 JsonlStore 完全一致）。"""
    return json.dumps(_safe_serialize(value), ensure_ascii=False, default=str)


def _load(text: Optional[str]) -> Any:
    """把 JSON 文本列反序列化回对象；NULL 返回 None。"""
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


class SQLiteStore(TraceStoreBase):
    """SQLite 落盘后端：一条 Span 一行，支持按 trace / session / kind 谓词下推查询。

    Args:
        path: SQLite 数据库文件路径；传 ``":memory:"`` 用内存库（不落盘，适合测试）。
        timeout: SQLite 写锁等待超时秒数（透传 ``SQLiteClient``，同时映射 busy_timeout）。

    设计：
    - 声明式 ``Schema`` + ``ensure_schema(migrate=True)`` 幂等建表建索引、老库自动补列，
      升级不丢历史 trace（参考 ``memory/sqlite_store.py``）；
    - 写入走 ``upsert(conflict_keys=["span_id"])``，同 span_id 重复写幂等覆盖；
    - 读取按自增 ``id`` 升序，等价 jsonl 的写入顺序；``iter_records`` 分批游标流式读，
      不一次性把全表载入内存。
    """

    _TABLE = "trace_spans"
    _IDX_TRACE = "idx_trace_spans_trace_id"
    _IDX_SESSION = "idx_trace_spans_session_id"
    _IDX_KIND = "idx_trace_spans_kind"
    _ITER_BATCH = 500

    # 声明式表结构，供 ensure_schema(..., migrate=True) 使用；
    # 缺列由 SQLiteClient 自动 ALTER TABLE ADD COLUMN 补齐，新库一次建齐。
    _SCHEMA = Schema(
        table_name=_TABLE,
        columns=[
            # 自增主键：既作 ROWID 别名，也作「写入顺序」排序键（等价 jsonl 行序）
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="span_id", type=ColumnType.TEXT, nullable=False, unique=True),
            Column(name="trace_id", type=ColumnType.TEXT, nullable=False),
            Column(name="parent_id", type=ColumnType.TEXT),
            Column(name="session_id", type=ColumnType.TEXT),
            Column(name="name", type=ColumnType.TEXT, nullable=False),
            Column(name="kind", type=ColumnType.TEXT, nullable=False),
            Column(name="start_time", type=ColumnType.REAL, nullable=False),
            Column(name="end_time", type=ColumnType.REAL),
            Column(name="duration_ms", type=ColumnType.REAL),
            Column(name="status", type=ColumnType.TEXT, nullable=False),
            Column(name="error", type=ColumnType.TEXT),
            Column(name="input", type=ColumnType.TEXT),
            Column(name="output", type=ColumnType.TEXT),
            Column(name="metadata", type=ColumnType.TEXT, nullable=False),
        ],
        indexes=[
            Index(name=_IDX_TRACE, columns=["trace_id"]),
            Index(name=_IDX_SESSION, columns=["session_id"]),
            Index(name=_IDX_KIND, columns=["kind"]),
        ],
    )

    def __init__(self, path: str, *, timeout: float = 10.0):
        if not path:
            raise ValueError("path 不能为空；内存模式请传 ':memory:'")
        self.path = path
        self._lock = threading.Lock()
        self._closed = False
        # 复用 database 模块统一的 SQLiteClient：自动 WAL / busy_timeout / 外键 / 参数化
        self._db = SQLiteClient(path, timeout=timeout)
        # 幂等建表 + 幂等建索引 + 老库自动补列
        self._db.ensure_schema(self._SCHEMA, migrate=True)

    # ---------- 序列化 / 反序列化 ----------
    @staticmethod
    def _span_to_row(span: SpanRecord) -> dict:
        """把 SpanRecord 转成数据库行字典（字段名与表列一一对应）。

        直接按字段取值、再对 input/output/metadata 做 ``_safe_serialize`` 兜底，
        而不是走 ``model_dump(mode="json")``——后者遇到无法识别的对象会整体抛
        ``PydanticSerializationError``，导致整条 Span 被静默丢弃。这里保证任何
        input/output 都能落库（无法序列化的转 str），与 JsonlStore 的兜底口径一致。
        """
        return {
            "span_id":     span.span_id,
            "trace_id":    span.trace_id,
            "parent_id":   span.parent_id,
            "session_id":  span.session_id,
            "name":        span.name,
            "kind":        _as_str(span.kind),
            "start_time":  span.start_time,
            "end_time":    span.end_time,
            "duration_ms": span.duration_ms,
            "status":      _as_str(span.status),
            "error":       span.error,
            "input":       _dump(span.input),
            "output":      _dump(span.output),
            "metadata":    _dump(span.metadata or {}),
        }

    @staticmethod
    def _row_to_record(row: dict) -> SpanRecord:
        """把数据库行还原成 SpanRecord（忽略自增 id 列）。"""
        fields: dict = {
            "trace_id":    row["trace_id"],
            "span_id":     row["span_id"],
            "parent_id":   row.get("parent_id"),
            "session_id":  row.get("session_id"),
            "name":        row["name"],
            "kind":        SpanKind(row["kind"]) if row.get("kind") is not None else SpanKind.CUSTOM,
            "end_time":    row.get("end_time"),
            "duration_ms": row.get("duration_ms"),
            "input":       _load(row.get("input")),
            "output":      _load(row.get("output")),
            "status":      SpanStatus(row["status"]) if row.get("status") is not None else SpanStatus.SUCCESS,
            "error":       row.get("error"),
            "metadata":    _load(row.get("metadata")) or {},
        }
        # start_time 有默认值，仅在库里有值时覆盖，避免 None 覆盖默认时间
        if row.get("start_time") is not None:
            fields["start_time"] = row["start_time"]
        return SpanRecord(**fields)

    # ---------- TraceStoreBase 接口 ----------
    def write(self, span: SpanRecord) -> None:
        """写入一条 Span（同 span_id 幂等覆盖）。任何异常都被吞掉（打点不能阻塞业务）。"""
        if self._closed:
            return
        try:
            row = self._span_to_row(span)
            with self._lock:
                self._db.upsert(self._TABLE, row, conflict_keys=["span_id"])
        except Exception:
            # 打点通道任何异常都不向上抛，避免影响主业务
            pass

    def read(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SpanRecord]:
        """按条件过滤读取 Span 列表（按写入顺序，即自增 id 升序）。"""
        results: List[SpanRecord] = []
        if self._closed:
            return results
        try:
            filters = self._build_filters(session_id, trace_id, kind)
            query = Query(
                table_name=self._TABLE,
                filters=filters or None,
                sort_by="id",
                sort_order=SortOrder.ASC,
                limit=limit,
            )
            with self._lock:
                rows = self._db.select_where(query)
        except Exception:
            return results
        for r in rows:
            try:
                results.append(self._row_to_record(r))
            except Exception:
                continue
        return results

    def iter_records(self) -> Iterator[SpanRecord]:
        """分批游标流式遍历所有记录（覆盖基类默认的 read() 全量载入）。"""
        if self._closed:
            return
        last_id = 0
        while True:
            try:
                with self._lock:
                    rows = self._db.select_where(Query(
                        table_name=self._TABLE,
                        filters=[Filter(field="id", op=FilterOp.GT, value=last_id)],
                        sort_by="id",
                        sort_order=SortOrder.ASC,
                        limit=self._ITER_BATCH,
                    ))
            except Exception:
                return
            if not rows:
                return
            for r in rows:
                last_id = r.get("id", last_id)
                try:
                    yield self._row_to_record(r)
                except Exception:
                    continue
            if len(rows) < self._ITER_BATCH:
                return

    def count(self) -> int:
        """返回已落盘 Span 总条数（便捷统计，非 TraceStoreBase 必需）。"""
        if self._closed:
            return 0
        try:
            with self._lock:
                return self._db.count(self._TABLE)
        except Exception:
            return 0

    def flush(self) -> None:
        """刷盘；每次 write 走单语句自动提交，这里保持与基类一致的 no-op 语义。"""
        return None

    def close(self) -> None:
        """关闭底层数据库连接，多次调用安全。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._db.close()
        except Exception:
            pass

    # ---------- 内部辅助 ----------
    @staticmethod
    def _build_filters(
        session_id: Optional[str],
        trace_id: Optional[str],
        kind: Optional[str],
    ) -> List[Filter]:
        """把 read 的三个过滤条件编译成等值 Filter 列表（AND 关系）。"""
        filters: List[Filter] = []
        if session_id is not None:
            filters.append(Filter(field="session_id", op=FilterOp.EQ, value=session_id))
        if trace_id is not None:
            filters.append(Filter(field="trace_id", op=FilterOp.EQ, value=trace_id))
        if kind is not None:
            filters.append(Filter(field="kind", op=FilterOp.EQ, value=_as_str(kind)))
        return filters

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["SQLiteStore"]
