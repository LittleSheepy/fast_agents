"""JsonlStore 后端专项单元测试。

本文件由 ``tests/trace/test_trace.py`` 拆分而来：只保留 JsonlStore 文件存储后端
相关的用例，其余通用用例（数据模型 / InMemoryTraceStore / TraceLogger 生命周期 /
tool_hook / LLM / Agent / replay 等）仍留在 ``test_trace.py``，
SQLiteStore 用例在 ``test_trace_sqlite_store.py``。

覆盖点：
    - JsonlStore 文件模式写入 / reopen(append=True) 读取 / append=False 清空重写
    - 落盘文件每行均为合法 JSON，可被标准 json 模块逐行解析
    - 文件模式不保留内存镜像（长时间运行不随文件线性涨内存）
    - input/output/metadata 含不可 JSON 序列化对象时的兜底（不丢 Span，兜底转 str）
    - 极端不可序列化对象（__repr__/__str__ 均抛错）兜底，且不修改原 Span
    - end_span 幂等：同一 Span 重复结束只落 1 条（Jsonl 后端）
    - 创建序号（_create_seq）经 Jsonl 落盘/读回后正确恢复，metadata 保持干净

运行方式（Windows PowerShell，在项目根目录）:
    python tests/trace/test_trace_jsonl_store.py
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import tmp_path

from fast_agents.trace import (
    JsonlStore,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceLogger,
)


# ==================== JsonlStore 文件模式 ====================
def test_jsonl_store_file_write_and_reopen(tmp_path):
    """写文件后 reopen(append=True) 能读到历史 + 新写入；跨进程/跨实例回放可用。"""
    p = str(tmp_path / "traces.jsonl")
    s1 = JsonlStore(p, append=True)
    sp = SpanRecord(trace_id="t1", name="first", kind=SpanKind.LLM)
    sp.finish(output="a")
    s1.write(sp)
    s1.flush()
    s1.close()

    # 重新打开，追加一条
    s2 = JsonlStore(p, append=True)
    sp2 = SpanRecord(trace_id="t1", name="second", kind=SpanKind.TOOL)
    sp2.finish(output="b")
    s2.write(sp2)
    rows = s2.read(trace_id="t1")
    s2.close()
    assert len(rows) == 2
    assert rows[0].name == "first"
    assert rows[1].name == "second"


def test_jsonl_store_file_overwrite_when_append_false(tmp_path):
    """append=False 时会清空重写。"""
    p = str(tmp_path / "traces.jsonl")
    s1 = JsonlStore(p, append=True)
    sp = SpanRecord(trace_id="t", name="a", kind=SpanKind.CUSTOM)
    sp.finish()
    s1.write(sp)
    s1.close()

    s2 = JsonlStore(p, append=False)
    assert len(s2.read()) == 0
    sp2 = SpanRecord(trace_id="t", name="b", kind=SpanKind.CUSTOM)
    sp2.finish()
    s2.write(sp2)
    assert len(s2.read()) == 1
    s2.close()


def test_jsonl_store_file_is_valid_jsonl(tmp_path):
    """写出的文件每行都是合法 JSON，可被标准 json 模块逐行解析。"""
    p = str(tmp_path / "traces.jsonl")
    store = JsonlStore(p)
    for i in range(3):
        s = SpanRecord(trace_id="t", name=f"n{i}", kind=SpanKind.CUSTOM, input={"i": i})
        s.finish(output={"o": i})
        store.write(s)
    store.close()
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == 3
    for ln in lines:
        obj = json.loads(ln)
        assert obj["trace_id"] == "t"
        assert obj["status"] in ("success", "error")
        assert obj["kind"] in ("llm", "tool", "decision", "agent", "custom")


def test_file_store_does_not_accumulate_in_memory(tmp_path):
    """JsonlStore 只有文件模式，不在内存保留全量镜像——
    写入/读取都不往内存累积数据，防止长时间运行随 jsonl 文件线性涨内存。"""
    p = str(tmp_path / "traces.jsonl")
    store = JsonlStore(p, append=True)
    # JsonlStore 已不再支持内存模式，实例上不存在 _mem 镜像属性
    assert not hasattr(store, "_mem")
    for i in range(50):
        s = SpanRecord(trace_id="t", name=f"n{i}", kind=SpanKind.CUSTOM)
        s.finish(output={"idx": i})
        store.write(s)
    assert not hasattr(store, "_mem"), "JsonlStore 不应有内存镜像属性"
    # read() 仍能正确返回全部记录（流式扫描文件）
    rows = store.read()
    assert len(rows) == 50
    assert rows[0].name == "n0"
    assert rows[-1].name == "n49"
    store.close()
    assert not hasattr(store, "_mem")


def test_jsonl_store_handles_unserializable_input_output(tmp_path):
    """Bug1 回归：input/output/metadata 包含不可 JSON 序列化的自定义对象时，
    整条 Span 不应被静默丢弃（旧实现会在 model_dump(mode="json") 阶段抛异常被吞掉，导致读不到任何记录）。
    行为对齐 SQLiteStore：不可序列化对象兜底转为 str(obj)，Span 正常落库可回读。"""
    class Weird:
        def __str__(self):
            return "<Weird>"

    store = JsonlStore(str(tmp_path / "unserializable.jsonl"))
    s = SpanRecord(
        trace_id="t1",
        name="weird-span",
        kind=SpanKind.CUSTOM,
        input={"obj": Weird(), "normal": 123},
        metadata={"bad": Weird()},
    )
    s.finish(output=Weird())
    store.write(s)

    rows = store.read()
    assert len(rows) == 1, f"不可序列化对象导致整条Span丢失，实际返回{len(rows)}条"
    span = rows[0]
    # 正常字段保留
    assert span.name == "weird-span"
    assert span.input["normal"] == 123
    # 不可序列化对象兜底为字符串表示
    assert span.input["obj"] == "<Weird>"
    assert span.output == "<Weird>"
    assert span.metadata["bad"] == "<Weird>"
    store.close()


def test_jsonl_store_file_handles_unserializable(tmp_path):
    """文件模式下同样能正确处理不可序列化对象，落盘后跨实例可回读。"""
    class Weird:
        def __str__(self):
            return "<WeirdFile>"

    p = str(tmp_path / "traces.jsonl")
    s1 = JsonlStore(p, append=True)
    sp = SpanRecord(
        trace_id="t1",
        name="weird-file",
        kind=SpanKind.TOOL,
        input=Weird(),
    )
    sp.finish(output={"res": Weird()})
    s1.write(sp)
    s1.flush()
    s1.close()

    # 重新打开验证可正常回读，没有丢数据
    s2 = JsonlStore(p, append=True)
    rows = s2.read(trace_id="t1")
    s2.close()
    assert len(rows) == 1
    assert rows[0].name == "weird-file"
    assert rows[0].input == "<WeirdFile>"
    assert rows[0].output["res"] == "<WeirdFile>"


# ==================== Bug3 幂等写入回归测试 ====================
def test_end_span_is_idempotent_no_duplicate_records_jsonl():
    """BUG3修复：同一Span重复调用end_span，JsonlStore只写1条，以第一次结果为准。"""
    t = TraceLogger()
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


# ---------- Bug1 回归：JsonlStore不可序列化入参不丢数据、不修改原span ----------
class BadRepr:
    def __repr__(self):
        raise RuntimeError("bad repr")
    def __str__(self):
        raise RuntimeError("bad str")

def test_jsonl_store_handles_extreme_unserializable_objects(tmp_path):
    # 该用例验证 JsonlStore 对极端不可序列化对象的兜底（JsonlStore 仅文件模式）
    tracer = TraceLogger(enabled=True, store=JsonlStore(str(tmp_path / "extreme.jsonl")))
    original_input = {"obj": BadRepr()}
    with tracer.trace("t1"):
        with tracer.span("s1", kind=SpanKind.CUSTOM, input=original_input) as s:
            s.output = "ok"
            # 验证原span对象的input/output未被修改
            assert s.input == original_input
            assert s.output == "ok"
    records = tracer.read_spans()
    assert len(records) == 1
    assert records[0].input["obj"].startswith("<")  # 兜底转字符串
    assert records[0].output == "ok"
    tracer.close()

def test_jsonl_store_file_handles_extreme_unserializable():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad_obj.jsonl")
        tracer = TraceLogger(enabled=True, store=path)
        with tracer.trace("t1"):
            with tracer.span("s1", kind=SpanKind.CUSTOM, input={"obj": BadRepr()}) as s:
                s.output = "ok"
        tracer.close()
        # 重开读取
        tracer2 = TraceLogger(enabled=True, store=path)
        records = tracer2.read_spans()
        assert len(records) == 1
        assert records[0].input["obj"].startswith("<")
        tracer2.close()


# ==================== Bug1 回归：创建序号经 Jsonl 落盘/读回 ====================
def test_create_seq_jsonl_roundtrip_metadata_clean(tmp_path):
    """Bug1 回归：jsonl 落盘->读回，metadata 与实时一致，创建序号正确恢复。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.LLM, metadata={"k": "v"})
    seq_live = s._create_seq
    store = JsonlStore(str(tmp_path / "seq.jsonl"))
    store.write(s)
    back = store.read()[0]
    store.close()
    assert back.metadata == {"k": "v"}
    assert back._create_seq == seq_live

def test_create_seq_missing_on_oldest_span_reads_as_zero():
    """Bug1 回归：完全无序号的最老数据读回 seq=0（不再被注入新号），metadata 保持干净。"""
    with tmp_path() as d:
        p = os.path.join(d, "oldest.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": "O", "span_id": "O1", "name": "oldest", "kind": "llm",
                "start_time": 1.0, "status": "success", "metadata": {"a": 1},
            }) + "\n")
        store = JsonlStore(p)
        rec = store.read()[0]
        store.close()
    assert rec.metadata == {"a": 1}
    assert rec._create_seq == 0


def test_create_seq_advances_after_reading_history(tmp_path):
    """Bug1 回归：读回历史数据后新建 Span 的序号应大于历史最大序号，避免续写冲撞。"""
    store = JsonlStore(str(tmp_path / "seq2.jsonl"))
    store.write(SpanRecord(trace_id="t1", name="old1", kind=SpanKind.CUSTOM))
    store.write(SpanRecord(trace_id="t1", name="old2", kind=SpanKind.CUSTOM))
    max_hist = max(r._create_seq for r in store.read())
    fresh = SpanRecord(trace_id="t2", name="fresh", kind=SpanKind.CUSTOM)
    store.close()
    assert fresh._create_seq > max_hist


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
        test_jsonl_store_file_write_and_reopen(tmpdir)
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
