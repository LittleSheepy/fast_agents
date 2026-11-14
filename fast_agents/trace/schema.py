from ..database import Column, ColumnType, Index, Schema

_TABLE = "trace_spans"
_IDX_TRACE = "idx_trace_spans_trace_id"
_IDX_SESSION = "idx_trace_spans_session_id"
_IDX_KIND = "idx_trace_spans_kind"

TRACE_SPANS_SCHEMA = Schema(
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
            Column(name="error_type", type=ColumnType.TEXT),
            Column(name="input", type=ColumnType.TEXT),
            Column(name="output", type=ColumnType.TEXT),
            Column(name="metadata", type=ColumnType.TEXT, nullable=False),
            Column(name="create_seq", type=ColumnType.INTEGER),
        ],
        indexes=[
            Index(name=_IDX_TRACE, columns=["trace_id"]),
            Index(name=_IDX_SESSION, columns=["session_id"]),
            Index(name=_IDX_KIND, columns=["kind"]),
        ],
    )
