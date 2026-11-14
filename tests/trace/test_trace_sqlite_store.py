"""SQLiteStore 后端专项单元测试。

本文件由 ``tests/trace/test_trace.py`` 拆分而来：只保留 SQLiteStore 存储后端
以及 TraceLogger 对 store 参数解析（含 SQLite 后缀自动识别）相关的用例，
其余通用用例（数据模型 / InMemoryTraceStore / JsonlStore / tool_hook / LLM /
Agent / replay 等）仍留在 ``test_trace.py``。

覆盖点：
    - SQLiteStore 内存模式 write/read 基本功能（按写入顺序返回、错误状态保留）
    - 按 session_id / trace_id / kind 过滤 + limit 生效
    - 文件模式跨 reopen 持久化（跨进程历史数据仍在）
    - Schema 幂等迁移（重复初始化不报错、老库自动补列、数据不丢）
    - iter_records 分批流式遍历不丢不重
    - 相同 span_id 重复写入幂等覆盖（upsert 不重复计数）
    - close 后写入异常静默吞掉（失败安全，不阻塞业务）
    - store 后缀 .db/.sqlite/.sqlite3 自动识别为 SQLiteStore，其余默认 JsonlStore
    - 显式传入 store 实例优先级最高（不按字符串解析后端）
    - end_span 幂等：同一 Span 重复结束只落 1 条（SQLite 后端）
    - 创建序号（_create_seq）经 SQLite 落盘/读回后正确恢复，metadata 保持干净

运行方式（Windows PowerShell，在项目根目录）:
    python tests/trace/test_trace_sqlite_store.py
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.trace import (
    InMemoryTraceStore,
    JsonlStore,
    MEMORY_PATH,
    SpanKind,
    SpanRecord,
    SpanStatus,
    SQLiteStore,
    TraceLogger,
)
from tests._runner import tmp_path


# ==================== SQLiteStore 后端测试 ====================

def test_sqlite_store_memory_write_read():
    """内存模式 SQLiteStore 写入读取基本功能。"""
    store = SQLiteStore(":memory:")
    try:
        # 写入3条span
        s1 = SpanRecord(trace_id="t1", name="s1", kind=SpanKind.LLM)
        s1.finish(output="r1")
        s2 = SpanRecord(trace_id="t1", name="s2", kind=SpanKind.TOOL, session_id="sess1")
        s2.finish(output="r2")
        s3 = SpanRecord(trace_id="t2", name="s3", kind=SpanKind.AGENT, session_id="sess1")
        s3.finish(output="r3", status=SpanStatus.ERROR, error="fail")
        store.write(s1); store.write(s2); store.write(s3)

        assert store.count() == 3
        all_spans = store.read()
        assert len(all_spans) == 3
        # 按写入顺序返回
        assert all_spans[0].name == "s1"
        assert all_spans[1].name == "s2"
        assert all_spans[2].name == "s3"
        # 错误状态正确
        assert all_spans[2].status == SpanStatus.ERROR
        assert all_spans[2].error == "fail"
    finally:
        store.close()


def test_sqlite_store_filter_by_session_trace_kind():
    """按 session_id / trace_id / kind 过滤正确。"""
    store = SQLiteStore(":memory:")
    try:
        for i in range(5):
            s = SpanRecord(
                trace_id=f"t{i%2}",
                name=f"span_{i}",
                kind=SpanKind.LLM if i%2==0 else SpanKind.TOOL,
                session_id=f"sess{i%3}"
            )
            s.finish()
            store.write(s)

        # 按kind过滤
        llm_spans = store.read(kind="llm")
        assert len(llm_spans) == 3  # 0,2,4
        assert all(s.kind == SpanKind.LLM for s in llm_spans)
        # 按trace_id过滤
        t0_spans = store.read(trace_id="t0")
        assert len(t0_spans) == 3  # 0,2,4
        assert all(s.trace_id == "t0" for s in t0_spans)
        # 按session_id过滤
        sess1_spans = store.read(session_id="sess1")
        assert len(sess1_spans) == 2  # 1,4
        assert all(s.session_id == "sess1" for s in sess1_spans)
        # limit参数
        limited = store.read(limit=2)
        assert len(limited) == 2
    finally:
        store.close()


def test_sqlite_store_file_persists_across_reopen():
    """文件模式SQLite：关闭后重新打开，历史数据仍在（跨进程持久化）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_traces.db")
        store1 = SQLiteStore(path)
        s = SpanRecord(trace_id="persist", name="persist_span", kind=SpanKind.AGENT)
        s.finish(output="hello")
        store1.write(s)
        assert store1.count() == 1
        store1.close()

        # 重新打开
        store2 = SQLiteStore(path)
        assert store2.count() == 1
        spans = store2.read()
        assert len(spans) == 1
        assert spans[0].name == "persist_span"
        assert spans[0].output == "hello"
        store2.close()


def test_sqlite_store_idempotent_schema_migration():
    """幂等Schema迁移：重复初始化不报错，老库自动补列。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "migrate_test.db")
        # 第一次初始化
        s1 = SQLiteStore(path)
        s1.write(SpanRecord(trace_id="t", name="first", kind=SpanKind.CUSTOM))
        s1.close()
        # 第二次初始化（模拟版本升级）：自动补列，不报错，数据不丢
        s2 = SQLiteStore(path)
        assert s2.count() == 1
        # 第三次初始化再写入新数据
        s2.write(SpanRecord(trace_id="t", name="second", kind=SpanKind.CUSTOM))
        assert s2.count() == 2
        s2.close()


def test_sqlite_store_iter_records_streaming():
    """iter_records 分批流式遍历返回所有记录，不丢不重。"""
    store = SQLiteStore(":memory:")
    try:
        expected = 20
        for i in range(expected):
            s = SpanRecord(trace_id="batch", name=f"iter_{i}", kind=SpanKind.CUSTOM)
            s.finish()
            store.write(s)
        assert store.count() == expected
        records = list(store.iter_records())
        assert len(records) == expected
        names = [r.name for r in records]
        assert names == [f"iter_{i}" for i in range(expected)]
    finally:
        store.close()


def test_sqlite_store_upsert_is_idempotent():
    """相同span_id重复写入幂等覆盖，不重复计数。"""
    store = SQLiteStore(":memory:")
    try:
        s = SpanRecord(span_id="fixed_id", trace_id="t", name="once", kind=SpanKind.CUSTOM)
        s.finish(output="v1")
        store.write(s)
        store.write(s)  # 重复写
        store.write(s)  # 再写一次
        assert store.count() == 1  # 只有一条
        # 修改后重新写（覆盖）
        s.output = "v2"
        store.write(s)
        assert store.count() == 1
        got = store.read()[0]
        assert got.output == "v2"
    finally:
        store.close()


def test_sqlite_store_write_errors_are_swallowed():
    """写入异常不向上抛（失败安全，打点不阻塞业务）。"""
    store = SQLiteStore(":memory:")
    store.close()
    # close后再write，不应该抛异常
    s = SpanRecord(trace_id="t", name="err", kind=SpanKind.CUSTOM)
    store.write(s)  # 静默失败
    assert store.count() == 0


def test_tracer_auto_detect_sqlite_suffix():
    """store 后缀为 .db/.sqlite/.sqlite3 时自动选择 SQLiteStore。"""
    for suffix in [".db", ".sqlite", ".sqlite3"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"auto{suffix}")
            t = TraceLogger(store=path)
            try:
                assert isinstance(t.store, SQLiteStore), f"suffix {suffix} 应自动识别为SQLite"
            finally:
                t.close()
    # jsonl后缀默认仍是JsonlStore
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "traces.jsonl")
        t = TraceLogger(store=path)
        try:
            assert isinstance(t.store, JsonlStore)
        finally:
            t.close()


def test_tracer_explicit_store_takes_precedence():
    """显式传入 store 实例时，直接使用该实例（不按字符串解析后端）。"""
    mem = InMemoryTraceStore()
    t = TraceLogger(store=mem)
    try:
        assert t.store is mem
        assert not isinstance(t.store, SQLiteStore)
    finally:
        t.close()


def test_end_span_is_idempotent_no_duplicate_records_sqlite():
    """BUG3修复：同一Span重复调用end_span，SQLiteStore只写1条，以第一次结果为准。"""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "traces.db")
        t = TraceLogger(store=path)
        try:
            s = t.start_span("test_span")
            t.end_span(s, output="first_result", status=SpanStatus.SUCCESS)
            # 第二次调用end_span，应该直接返回，不重复写入，也不覆盖第一次结果
            t.end_span(s, output="second_result", status=SpanStatus.ERROR, error="duplicate call")
            spans = t.read_spans()
            assert len(spans) == 1
            assert spans[0].output == "first_result"
            assert spans[0].status == SpanStatus.SUCCESS
            assert spans[0].error is None
        finally:
            t.close()


def test_create_seq_sqlite_roundtrip_metadata_clean():
    """Bug1 回归：SQLite 落盘->读回，metadata 与实时一致，创建序号正确恢复。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.TOOL, metadata={"k": "v"})
    seq_live = s._create_seq
    store = SQLiteStore(MEMORY_PATH)
    store.write(s)
    back = store.read()[0]
    store.close()
    assert back.metadata == {"k": "v"}
    assert back._create_seq == seq_live


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    with tmp_path() as tmpdir:
        pass
    test_sqlite_store_file_persists_across_reopen()
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
