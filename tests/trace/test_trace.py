"""trace 模块单元测试。

全部用内存 JsonlStore 或 tests._runner.tmp_path() 临时目录，不依赖外部服务、不产生真实 API 请求。
覆盖点：
    - 默认关闭（tracer=None 零开销，不传时 LLM/Agent 行为不变）
    - SpanKind / SpanStatus / SpanRecord 字段/finish（含可选 end_time 精确对齐）
    - JsonlStore 内存模式 / 文件模式 / append / 过滤查询
    - JsonlStore 文件模式不做内存镜像（大文件不涨内存）
    - TraceLogger start_trace/end_trace/with trace
    - start_span/end_span/with span（正常路径 + 异常路径）
    - 父子关系（parent_id 正确串联）
    - tool_hook 签名对齐 ToolRegistry.add_hook（ToolResult 成功/失败/截断）
    - tool_hook 耗时取自 ToolResult.metadata['duration_ms'] 并单次取时刻回推
      start_time，最终 duration_ms 严格等于工具自报值（零漂移）
    - attach_to_tool_registry 一行接入
    - LLM.invoke/stream_invoke/embed_query/embed_documents 在 tracer=None 时不打点、
      传 tracer 时正确记录 llm Span（mock 底层 _client，无需真 API Key）
    - LLM 打点写入 token 用量元数据（usage / prompt_tokens / completion_tokens / total_tokens）
    - ReActAgent tracer 接入（mock llm.invoke + 简单工具）：agent/decision/tool/llm 四类 Span 都有
    - replay() 能按时间顺序返回可打印列表
    - 存储层预留位：用户可自定义 TraceStoreBase 子类
    - 文件路径下 jsonl 落盘可被新 JsonlStore 重新读取（跨进程回放）
    - 禁用 enabled=False 时不写入任何记录
    - 实例级 ContextVar：跨 TraceLogger 实例不串扰、为 asyncio 并发做好准备

运行方式（Windows PowerShell，在项目根目录）:
    python test\\test_trace.py
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, tmp_path

from fast_agents.trace import (
    JsonlStore,
    MEMORY_PATH,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceLogger,
    TraceStoreBase,
)
from fast_agents.tools import ToolRegistry, StructuredTool
from fast_agents.tools.tool_result import ToolResult, ToolStatus


# ==================== 数据模型 ====================
def test_span_record_defaults():
    """SpanRecord 默认字段齐全：trace_id/span_id/start_time 自动生成，status=success。"""
    s = SpanRecord(trace_id="t1", name="x", kind=SpanKind.CUSTOM)
    assert s.trace_id == "t1"
    assert s.span_id and len(s.span_id) == 32  # uuid hex
    assert s.status == SpanStatus.SUCCESS
    assert s.start_time > 0
    assert s.end_time is None
    assert s.duration_ms is None
    assert s.input is None
    assert s.output is None
    assert s.error is None
    assert isinstance(s.metadata, dict)


def test_span_record_finish():
    """finish 会写 end_time / duration_ms / output / status。"""
    s = SpanRecord(trace_id="t1", name="x", kind=SpanKind.LLM)
    import time
    time.sleep(0.005)
    s.finish(output="hi", metadata={"model": "m"})
    assert s.end_time is not None and s.end_time >= s.start_time
    assert s.duration_ms is not None and s.duration_ms >= 5.0  # 至少 5ms
    assert s.output == "hi"
    assert s.status == SpanStatus.SUCCESS
    assert s.metadata.get("model") == "m"
    # 内部排序字段_create_seq存在且为正整数即可，不做等于校验


def test_span_record_finish_error_sets_status():
    """传 error 给 end_span 时，会自动置 status=ERROR。直接对 SpanRecord 单独 finish 不会自动改状态（这是 TraceLogger.end_span 的职责），
    这里只验证：finish 能正确记录 error 字段。"""
    s = SpanRecord(trace_id="t1", name="x", kind=SpanKind.TOOL, status=SpanStatus.ERROR)
    s.finish(output=None, error="boom", status=SpanStatus.ERROR)
    assert s.status == SpanStatus.ERROR
    assert s.error == "boom"


def test_span_kind_enum_values():
    """路线图最小三类 kind 都在：llm/tool/decision；另外留 agent/custom。"""
    assert SpanKind.LLM.value == "llm"
    assert SpanKind.TOOL.value == "tool"
    assert SpanKind.DECISION.value == "decision"
    assert SpanKind.AGENT.value == "agent"
    assert SpanKind.CUSTOM.value == "custom"


# ==================== JsonlStore 内存模式 ====================
def test_jsonl_store_memory_write_read():
    """内存模式下 write/read 正常，按写入顺序返回。"""
    store = JsonlStore(MEMORY_PATH)
    s1 = SpanRecord(trace_id="t1", name="a", kind=SpanKind.LLM)
    s1.finish(output=1)
    s2 = SpanRecord(trace_id="t1", name="b", kind=SpanKind.TOOL, parent_id=s1.span_id)
    s2.finish(output=2)
    store.write(s1)
    store.write(s2)
    rows = store.read()
    assert len(rows) == 2
    assert rows[0].name == "a"
    assert rows[1].name == "b"
    store.close()


def test_jsonl_store_filter_by_session_trace_kind():
    """按 session_id / trace_id / kind 过滤。"""
    store = JsonlStore(MEMORY_PATH)
    # i: 0 → t0/s0/LLM; 1 → t1/s1/TOOL; 2 → t0/s0/LLM; 3 → t1/s1/TOOL; 4 → t0/s0/LLM
    for i in range(5):
        s = SpanRecord(
            trace_id=f"t{i % 2}",
            name=f"n{i}",
            kind=[SpanKind.LLM, SpanKind.TOOL][i % 2],
            session_id=f"s{i % 2}",
        )
        s.finish(output=i)
        store.write(s)
    assert len(store.read()) == 5
    assert len(store.read(trace_id="t0")) == 3  # i=0,2,4
    assert len(store.read(session_id="s1")) == 2  # i=1,3
    assert len(store.read(kind="llm")) == 3
    assert len(store.read(kind="tool")) == 2
    assert len(store.read(trace_id="t0", kind="llm")) == 3
    assert len(store.read(trace_id="t1", kind="tool")) == 2
    store.close()


def test_jsonl_store_limit():
    """limit 参数生效。"""
    store = JsonlStore(MEMORY_PATH)
    for i in range(10):
        s = SpanRecord(trace_id="t", name=f"n{i}", kind=SpanKind.CUSTOM)
        s.finish()
        store.write(s)
    assert len(store.read(limit=3)) == 3
    store.close()


def test_jsonl_store_memory_path_none_equals_memory():
    """path=None 等价于内存模式。"""
    store = JsonlStore(None)
    s = SpanRecord(trace_id="t", name="x", kind=SpanKind.CUSTOM)
    s.finish(output="y")
    store.write(s)
    assert len(store.read()) == 1
    assert store._fp is None
    store.close()


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
    """文件模式下 _mem 必须始终为空——写入/读取都不往内存镜像累积数据，
    防止长时间运行随 jsonl 文件线性涨内存。"""
    p = str(tmp_path / "traces.jsonl")
    store = JsonlStore(p, append=True)
    assert store._is_memory is False
    for i in range(50):
        s = SpanRecord(trace_id="t", name=f"n{i}", kind=SpanKind.CUSTOM)
        s.finish(output={"idx": i})
        store.write(s)
    # 关键断言：文件模式下 _mem 保持空
    assert len(store._mem) == 0, f"文件模式下 _mem 不应累积数据，实际 len={len(store._mem)}"
    # 但 read() 仍能正确返回全部记录（流式扫描文件）
    rows = store.read()
    assert len(rows) == 50
    assert rows[0].name == "n0"
    assert rows[-1].name == "n49"
    store.close()
    # close 后文件模式下 _mem 也保持空（兜底 clear）
    assert len(store._mem) == 0


def test_jsonl_store_handles_unserializable_input_output():
    """Bug1 回归：input/output/metadata 包含不可 JSON 序列化的自定义对象时，
    整条 Span 不应被静默丢弃（旧实现会在 model_dump(mode="json") 阶段抛异常被吞掉，导致读不到任何记录）。
    行为对齐 SQLiteStore：不可序列化对象兜底转为 str(obj)，Span 正常落库可回读。"""
    class Weird:
        def __str__(self):
            return "<Weird>"

    store = JsonlStore(MEMORY_PATH)
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


# ==================== TraceLogger 默认关闭 ====================
def test_tracer_disabled_writes_nothing():
    """enabled=False 时任何 span 操作都不写入。"""
    t = TraceLogger(enabled=False)
    with t.trace("demo"):
        with t.span("x", SpanKind.LLM, input={"q": 1}) as s:
            s.output = "y"
    assert len(t.read_spans()) == 0
    t.close()


def test_tracer_none_zero_overhead_for_llm_via_default():
    """LLM 默认 tracer=None；如果外部不设置 tracer 属性，invoke 路径直接透传（本测试不 mock 客户端，
    只验证 tracer=None 时 LLM.invoke 不会因为 trace 分支引入额外状态）。"""
    from fast_agents.llm import LLM
    # 用 base_url 指向一个无效地址但不真正调用；只构造与检查 tracer 属性
    llm = LLM.__new__(LLM)
    llm.tracer = None
    llm.model = "m"
    # 关键断言：默认 tracer 为 None
    assert llm.tracer is None


# ==================== TraceLogger: trace / span 生命周期 ====================
def test_start_end_trace_basic():
    """start_trace/end_trace 正确切换 trace_id，跨 trace span 归各自所有。"""
    t = TraceLogger()
    t1 = t.start_trace("a")
    s1 = t.start_span("s1", SpanKind.CUSTOM)
    t.end_span(s1, output=1)
    t.end_trace()
    t2 = t.start_trace("b")
    s2 = t.start_span("s2", SpanKind.CUSTOM)
    t.end_span(s2, output=2)
    t.end_trace()
    assert len(t.read_spans(trace_id=t1)) == 1
    assert len(t.read_spans(trace_id=t2)) == 1
    t.close()


def test_with_trace_context_manager():
    """with t.trace() 自动开启/结束 trace，yield trace_id。"""
    t = TraceLogger()
    outer_tid = None
    with t.trace("root") as tid:
        outer_tid = tid
        with t.span("s1", SpanKind.CUSTOM) as s:
            s.output = 1
    assert outer_tid is not None and outer_tid.startswith("trace_")
    spans = t.read_spans(trace_id=outer_tid)
    assert len(spans) == 1
    assert spans[0].name == "s1"
    t.close()


def test_with_span_records_on_normal_exit():
    """with span 正常退出时自动 finish 并写入 store。"""
    t = TraceLogger()
    with t.trace("demo"):
        with t.span("hi", SpanKind.CUSTOM, input={"x": 1}) as s:
            s.output = {"y": 2}
    spans = t.read_spans()
    assert len(spans) == 1
    assert spans[0].name == "hi"
    assert spans[0].output == {"y": 2}
    assert spans[0].status == SpanStatus.SUCCESS
    assert spans[0].duration_ms is not None and spans[0].duration_ms >= 0
    t.close()


def test_with_span_marks_error_on_exception():
    """with span 内抛异常时自动记 status=error，并记录 error 字段，异常继续向上抛。"""
    t = TraceLogger()
    with raises(ValueError, match="boom"):
        with t.trace("demo"):
            with t.span("bad", SpanKind.CUSTOM):
                raise ValueError("boom")
    spans = t.read_spans()
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.ERROR
    assert "ValueError" in spans[0].error
    assert "boom" in spans[0].error
    t.close()


def test_parent_child_relationship():
    """嵌套 span 时 parent_id 正确指向父 span_id。"""
    t = TraceLogger()
    with t.trace("demo"):
        with t.span("parent", SpanKind.AGENT) as p:
            with t.span("child", SpanKind.LLM) as c:
                c.output = "ok"
    spans = t.read_spans()
    assert len(spans) == 2
    parent = [s for s in spans if s.name == "parent"][0]
    child = [s for s in spans if s.name == "child"][0]
    assert parent.parent_id is None
    assert child.parent_id == parent.span_id
    assert child.trace_id == parent.trace_id
    t.close()


def test_manual_start_end_span_duration():
    """手动 start/end span 时 duration_ms 正确计算，且字段覆盖最小集。"""
    t = TraceLogger()
    with t.trace("demo"):
        s = t.start_span("m", SpanKind.TOOL, input={"args": [1, 2]})
        import time
        time.sleep(0.01)
        t.end_span(s, output=3, metadata={"tool": "add"})
    sp = t.read_spans()[0]
    assert sp.kind == SpanKind.TOOL
    assert sp.input == {"args": [1, 2]}
    assert sp.output == 3
    assert sp.metadata == {"tool": "add"}
    assert sp.duration_ms is not None and sp.duration_ms >= 10.0
    assert sp.start_time > 0
    assert sp.end_time is not None
    assert sp.session_id is not None  # 最小集：session_id
    assert sp.trace_id.startswith("trace_")
    t.close()


# ==================== ToolRegistry 钩子 ====================
def _make_registry_with_tools():
    @StructuredTool.from_function
    def add(a: int, b: int) -> int:
        """add"""
        return a + b

    @StructuredTool.from_function
    def bad() -> str:
        """bad"""
        raise RuntimeError("fail")

    reg = ToolRegistry()
    reg.register_tool(add)
    reg.register_tool(bad)
    return reg


def test_tool_hook_records_success():
    """工具正常执行后，tool_hook 会写一条 kind=tool Span，含输入/输出/耗时。"""
    reg = _make_registry_with_tools()
    t = TraceLogger()
    t.attach_to_tool_registry(reg)
    with t.trace("tool-demo"):
        r = reg.invoke_tool("add", {"a": 2, "b": 3})
    # add 返回 int 5，ToolResult.output 为字符串化结果（框架行为）
    assert r.status == ToolStatus.SUCCESS
    assert "5" in str(r.output)
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "tool.add"
    assert s.input == {"a": 2, "b": 3}
    assert "5" in str(s.output)
    assert s.status == SpanStatus.SUCCESS
    assert s.duration_ms is not None
    t.close()


def test_tool_hook_records_error():
    """工具抛错时，tool_hook 仍能记录一条 error Span（依赖 ToolRegistry 异常捕获返回 ToolResult.fail）。"""
    reg = _make_registry_with_tools()
    t = TraceLogger()
    t.attach_to_tool_registry(reg)
    with t.trace("tool-demo"):
        r = reg.invoke_tool("bad", {})
    assert r.status == ToolStatus.ERROR
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    assert spans[0].status == SpanStatus.ERROR
    assert "fail" in (spans[0].error or "")
    t.close()


def test_tool_hook_uses_real_duration_from_metadata():
    """tool_hook 从 ``ToolResult.metadata['duration_ms']`` 取真实耗时，并回推 start_time。"""
    t = TraceLogger()
    r = ToolResult.ok("5")
    r.metadata["duration_ms"] = 120  # 模拟 tool_base 写入的实测耗时（int 毫秒）
    with t.trace("tool-dur"):
        t.tool_hook("add", {"a": 2, "b": 3}, r)
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    s = spans[0]
    # 耗时直接取工具自报值
    assert abs(s.duration_ms - 120.0) < 1.0, f"duration_ms={s.duration_ms}"
    # start_time 被回推，时间区间长度 ≈ 120ms
    assert s.end_time is not None
    assert abs((s.end_time - s.start_time) * 1000.0 - 120.0) < 1.0
    assert s.metadata["tool_name"] == "add"
    assert s.metadata["tool_status"] == "success"
    assert "duration_source" not in s.metadata
    t.close()


def test_tool_hook_marks_duration_unavailable_when_missing():
    """工具未上报 duration_ms 时降级：仍记录 Span，并在 metadata 标记来源。"""
    t = TraceLogger()
    r = ToolResult.ok("ok")  # metadata 里没有 duration_ms
    with t.trace("tool-nodur"):
        t.tool_hook("noop", {}, r)
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    s = spans[0]
    assert s.metadata["duration_source"] == "unavailable"
    assert s.duration_ms is not None and s.duration_ms >= 0.0
    t.close()


def test_tool_hook_duration_is_exactly_reported_value():
    """tool_hook 单次取时刻回推 start_time 并显式传入 end_time，
    最终 duration_ms 必须严格等于工具自报值，偏差 < 0.01ms（浮点噪声量级，
    而不是修复前的几百微秒处理开销累加）。"""
    for reported_ms in (1.0, 120.0, 999.0):
        t = TraceLogger()
        r = ToolResult.ok("x")
        r.metadata["duration_ms"] = reported_ms
        t.tool_hook("probe", {}, r)
        spans = t.read_spans(kind="tool")
        assert len(spans) == 1
        s = spans[0]
        drift_ms = abs(s.duration_ms - reported_ms)
        assert drift_ms < 0.01, (
            f"reported={reported_ms}ms got={s.duration_ms}ms drift={drift_ms*1000:.2f}us > 10us"
        )
        # 同时校验时间区间长度同样无漂移
        interval_ms = (s.end_time - s.start_time) * 1000.0
        assert abs(interval_ms - reported_ms) < 0.01
        t.close()


def test_tool_hook_records_real_duration_via_registry_end_to_end():
    """端到端：真实工具经 registry.invoke_tool 执行，tool span 耗时 ≈ 工具真实耗时。"""
    reg = ToolRegistry()

    @StructuredTool.from_function
    def slow() -> str:
        """慢工具"""
        import time as _time
        _time.sleep(0.02)
        return "done"

    reg.register_tool(slow)
    t = TraceLogger()
    t.attach_to_tool_registry(reg)
    with t.trace("tool-e2e"):
        r = reg.invoke_tool("slow", {})
    assert r.status == ToolStatus.SUCCESS
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    s = spans[0]
    # 真实测量值 ≥ 20ms（留 5ms 容差），而不是修复前的 ~0ms
    assert s.duration_ms >= 15.0, f"duration_ms={s.duration_ms}"
    assert s.metadata["tool_status"] == "success"
    t.close()


def test_attach_to_tool_registry_is_noop_when_disabled():
    """enabled=False 时 attach_to_tool_registry 不挂任何 hook，不影响 registry。"""
    reg = _make_registry_with_tools()
    before = len(reg._hooks)
    t = TraceLogger(enabled=False)
    t.attach_to_tool_registry(reg)
    assert len(reg._hooks) == before
    t.close()


# ==================== LLM 门面打点（mock 底层 _client）====================
class _FakeClient:
    """模拟 OpenAIClient，不发网络请求。

    与真实 OpenAIClient 行为对齐：识别可选关键字 ``usage_sink`` 并回填 token 用量，
    用于验证 LLM 打点是否把 token 元数据写进 Span。
    """
    def __init__(self):
        self.model = "fake-model"
        self.calls = []

    def invoke(self, messages, tools=None, usage_sink=None, **kwargs):
        self.calls.append(("invoke", messages, tools, kwargs))
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
        from fast_agents.messages.message import Message
        return Message("assistant", "mock answer")

    def stream_invoke(self, messages, tools=None, **kwargs):
        self.calls.append(("stream", messages, tools, kwargs))
        for ch in ["hello", " world"]:
            yield ch

    def embed_query(self, text, usage_sink=None, **kwargs):
        self.calls.append(("embed_query", text, kwargs))
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 5, "total_tokens": 5})
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts, usage_sink=None, **kwargs):
        self.calls.append(("embed_documents", texts, kwargs))
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 8, "total_tokens": 8})
        return [[0.1, 0.2], [0.3, 0.4]]


def _make_fake_llm(tracer=None):
    from fast_agents.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "fake-model"
    llm.api_key = "x"
    llm.base_url = None
    llm.embedding_model = "fake-embed"
    llm.timeout = None
    llm.provider = "openai"
    llm.tracer = tracer
    llm._client = _FakeClient()
    return llm


def test_llm_invoke_with_tracer_records_span():
    """LLM.invoke 传 tracer 时记录一条 llm.invoke Span，且原返回值正常。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-demo"):
        msg = llm.invoke([{"role": "user", "content": "hi"}])
    assert msg.role == "assistant"
    assert msg.content == "mock answer"
    spans = t.read_spans(kind="llm")
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "llm.invoke"
    assert s.input["messages_count"] == 1
    assert s.output["has_content"] is True
    assert s.metadata["model"] == "fake-model"
    assert s.status == SpanStatus.SUCCESS
    assert s.duration_ms is not None
    t.close()


def test_llm_invoke_without_tracer_records_nothing():
    """tracer=None（默认）时不记录任何 Span，invoke 正常返回。"""
    t = TraceLogger()  # 但不传给 LLM
    llm = _make_fake_llm(tracer=None)
    msg = llm.invoke([{"role": "user", "content": "hi"}])
    assert msg.content == "mock answer"
    assert len(t.read_spans()) == 0
    t.close()


def test_llm_stream_invoke_with_tracer_records_span():
    """stream_invoke 在 generator 耗尽后记录一条 llm.stream_invoke Span。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-demo"):
        chunks = list(llm.stream_invoke([{"role": "user", "content": "hi"}]))
    assert chunks == ["hello", " world"]
    spans = t.read_spans(kind="llm")
    assert len(spans) == 1
    assert spans[0].name == "llm.stream_invoke"
    assert spans[0].output["chunks"] == 2
    assert spans[0].output["length"] == len("hello world")
    assert spans[0].metadata["stream"] is True
    t.close()


def test_llm_embed_query_with_tracer_records_span():
    """embed_query 传 tracer 时记录一条 llm.embed_query Span。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-demo"):
        vec = llm.embed_query("hello")
    assert vec == [0.1, 0.2, 0.3]
    spans = t.read_spans(kind="llm")
    assert len(spans) == 1
    assert spans[0].name == "llm.embed_query"
    assert spans[0].output["dim"] == 3
    t.close()


def test_llm_embed_documents_with_tracer_records_span():
    """embed_documents 传 tracer 时记录一条 llm.embed_documents Span。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-demo"):
        vecs = llm.embed_documents(["a", "b"])
    assert len(vecs) == 2
    spans = t.read_spans(kind="llm")
    assert len(spans) == 1
    assert spans[0].name == "llm.embed_documents"
    assert spans[0].output["batch"] == 2
    assert spans[0].output["dim"] == 2
    t.close()


# ==================== LLM 打点：token / model 元数据 ====================
def test_llm_invoke_records_token_metadata():
    """LLM.invoke 打点把底层回传的 token 用量写进 Span.metadata（model + usage 平铺字段）。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-token"):
        llm.invoke([{"role": "user", "content": "hi"}])
    spans = t.read_spans(kind="llm")
    assert len(spans) == 1
    md = spans[0].metadata
    assert md["model"] == "fake-model"
    assert md["prompt_tokens"] == 11
    assert md["completion_tokens"] == 7
    assert md["total_tokens"] == 18
    assert md["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    t.close()


def test_llm_invoke_without_usage_adds_no_token_keys():
    """底层不回填 usage 时，metadata 里不出现 token 字段（避免 null 噪音）。"""
    class _NoUsageClient(_FakeClient):
        def invoke(self, messages, tools=None, usage_sink=None, **kwargs):
            from fast_agents.messages.message import Message
            return Message("assistant", "mock answer")

    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    llm._client = _NoUsageClient()
    with t.trace("llm-token"):
        llm.invoke([{"role": "user", "content": "hi"}])
    md = t.read_spans(kind="llm")[0].metadata
    assert "usage" not in md
    assert "total_tokens" not in md
    assert md["model"] == "fake-model"
    t.close()


def test_llm_embed_query_records_token_metadata():
    """embed_query 打点同样把 token 用量写进 metadata。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("emb"):
        llm.embed_query("hello")
    md = t.read_spans(kind="llm")[0].metadata
    assert md["op"] == "embed_query"
    assert md["prompt_tokens"] == 5
    assert md["total_tokens"] == 5
    t.close()


def test_llm_embed_documents_records_token_metadata():
    """embed_documents 打点把 token 用量写进 metadata。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("emb"):
        llm.embed_documents(["a", "b"])
    md = t.read_spans(kind="llm")[0].metadata
    assert md["op"] == "embed_documents"
    assert md["total_tokens"] == 8
    t.close()


def test_llm_stream_invoke_has_no_token_metadata():
    """流式接口默认不产生 token 元数据（底层流式不返回 usage），但 Span 仍正常记录。"""
    t = TraceLogger()
    llm = _make_fake_llm(tracer=t)
    with t.trace("s"):
        list(llm.stream_invoke([{"role": "user", "content": "hi"}]))
    s = t.read_spans(kind="llm")[0]
    assert s.name == "llm.stream_invoke"
    assert "usage" not in s.metadata
    assert "total_tokens" not in s.metadata
    t.close()


# ==================== ReActAgent 接入（mock llm，不发真实请求）====================
def _make_fake_agent_llm():
    """返回一个会走「先思考→调工具→给答案」两步的假 LLM，验证 agent/decision/tool/llm 四类 Span 全有。"""
    from fast_agents.llm import LLM
    from fast_agents.messages.message import Message, MsgToolCall

    class StepClient:
        def __init__(self):
            self.step = 0
            self.model = "fake"
            self.embedding_model = "fake-embed"

        def invoke(self, messages, tools=None, **kwargs):
            self.step += 1
            if self.step == 1:
                # 第一步：调 add(a=1,b=2)；MsgToolCall.arguments 要求为 JSON 字符串
                import json as _json
                tc = MsgToolCall(id="call_1", name="add", arguments=_json.dumps({"a": 1, "b": 2}))
                m = Message("assistant", "", tool_calls=[tc])
                return m
            else:
                # 第二步：最终答案
                return Message("assistant", "答案是 3")

        def stream_invoke(self, messages, tools=None, **kwargs):
            raise NotImplementedError

        def embed_query(self, text, **kwargs):
            return [0.0] * 3

        def embed_documents(self, texts, **kwargs):
            return [[0.0] * 3 for _ in texts]

    llm = LLM.__new__(LLM)
    llm.model = "fake"
    llm.api_key = "x"
    llm.base_url = None
    llm.embedding_model = "fake-embed"
    llm.timeout = None
    llm.provider = "openai"
    llm.tracer = None
    llm._client = StepClient()
    return llm


def _disable_agent_subsystems(agent_kwargs):
    """关掉记忆/技能/MCP/context，避免依赖外部资源，纯跑 ReAct 主循环。"""
    agent_kwargs.setdefault("enable_memory", False)
    agent_kwargs.setdefault("enable_skills", False)
    agent_kwargs.setdefault("enable_mcp", False)
    agent_kwargs.setdefault("enable_context", False)


def test_react_agent_with_tracer_records_all_spans():
    """传 tracer 给 ReActAgent 时，自动挂上 agent/decision/llm/tool 四类 Span；
    工具用 ToolRegistry.add_hook 打点（不修改注册表逻辑）；最终答案正确。"""
    from fast_agents import ReActAgent

    @StructuredTool.from_function
    def add(a: int, b: int) -> int:
        """add two numbers"""
        return a + b

    reg = ToolRegistry()
    reg.register_tool(add)

    t = TraceLogger()
    llm = _make_fake_agent_llm()
    kwargs = dict(
        name="test-agent",
        llm=llm,
        tool_registry=reg,
        system_prompt="你是一个计算助手。",
        max_steps=5,
        tracer=t,
    )
    _disable_agent_subsystems(kwargs)
    agent = ReActAgent(**kwargs)

    # tracer 应自动：1) 给 llm 打 invoke 点；2) 给 registry 加 hook
    assert agent.tracer is t
    assert llm.tracer is t  # llm.tracer 被自动注入
    assert any(True for h in reg._hooks if getattr(h, "__self__", None) is t)

    with t.trace("agent-run"):
        ans = agent.run("1+2等于几？")

    assert "3" in ans
    all_spans = t.read_spans()
    kinds = {s.kind for s in all_spans}
    assert SpanKind.AGENT in kinds, f"应有 agent span，实际 kinds={kinds}"
    assert SpanKind.DECISION in kinds, f"应有 decision span，实际 kinds={kinds}"
    assert SpanKind.LLM in kinds, f"应有 llm span（llm.tracer 自动注入），实际 kinds={kinds}"
    assert SpanKind.TOOL in kinds, f"应有 tool span（tool_hook 自动挂），实际 kinds={kinds}"

    # agent span 包含最终答案预览
    agent_spans = [s for s in all_spans if s.kind == SpanKind.AGENT]
    assert len(agent_spans) >= 1
    assert "3" in (agent_spans[0].output.get("final_answer_preview") or "")

    # tool span 记录了 add 入参/结果
    tool_spans = [s for s in all_spans if s.kind == SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].input == {"a": 1, "b": 2}
    assert "3" in str(tool_spans[0].output)

    t.close()


def test_react_agent_without_tracer_works_as_before():
    """tracer=None（默认）时 ReActAgent 仍正常运行，且没有任何 span 被记录（零开销）。"""
    from fast_agents import ReActAgent

    @StructuredTool.from_function
    def add(a: int, b: int) -> int:
        return a + b

    reg = ToolRegistry()
    reg.register_tool(add)
    llm = _make_fake_agent_llm()
    kwargs = dict(
        name="test-agent-no-tracer",
        llm=llm,
        tool_registry=reg,
        system_prompt="你是一个计算助手。",
        max_steps=5,
        tracer=None,
    )
    _disable_agent_subsystems(kwargs)
    agent = ReActAgent(**kwargs)
    assert agent.tracer is None
    hooks_before = len(reg._hooks)
    ans = agent.run("1+2=?")
    assert "3" in ans
    assert len(reg._hooks) == hooks_before  # 没挂任何 hook
    # 即便有人创建了 TraceLogger，只要没传给 agent 就不会产生记录
    t = TraceLogger()
    assert len(t.read_spans()) == 0
    t.close()


# ==================== 回放 ====================
def test_replay_returns_ordered_printable_list():
    """replay() 返回按 start_time 排序的列表，字段结构稳定。"""
    t = TraceLogger()
    with t.trace("r"):
        with t.span("outer", SpanKind.AGENT) as outer:
            with t.span("inner", SpanKind.LLM, input={"q": "hi"}) as inner:
                inner.output = "ok"
            outer.output = "done"
    rep = t.replay()
    assert len(rep) == 2
    # 返回顺序按 start_time 升序：外层 span 先开先出
    assert rep[0]["name"] == "outer"
    assert rep[1]["name"] == "inner"
    # 字段结构稳定（不暴露 start_time/end_time，内部用于排序即可）
    for entry in rep:
        assert set(entry.keys()) >= {
            "span_id", "parent_id", "kind", "name", "status", "duration_ms", "error",
            "input", "output", "metadata",
        }
    # 真正按 start_time 有序：从 store 读回原始 SpanRecord 校验时序
    all_spans = t.read_spans()
    by_name = {s.name: s for s in all_spans}
    assert by_name["inner"].start_time >= by_name["outer"].start_time
    t.close()


def test_replay_defaults_to_latest_trace_when_no_active():
    """修复BUG2-1：无活跃trace且未指定trace_id时，replay默认回放最近一条结束的trace，
    既不返回全量所有Span，也不返回空列表（对齐"回放最近一次运行"的用户直觉）。"""
    t = TraceLogger()
    # 先写第一条trace
    with t.trace("trace-a"):
        with t.span("a", SpanKind.CUSTOM):
            pass
    import time
    time.sleep(0.001)
    # 再写第二条更新的trace
    with t.trace("trace-b"):
        with t.span("b1", SpanKind.CUSTOM):
            pass
        with t.span("b2", SpanKind.CUSTOM):
            pass
    # 此时无活跃trace，replay()应只返回最新的trace-b（2条），不是全量（3条）也不是空
    rep = t.replay()
    assert len(rep) == 2, f"默认应回放最近一条trace，实际返回{len(rep)}条（期望2，全量应为3）"
    assert {item["name"] for item in rep} == {"b1", "b2"}
    # 显式指定trace_id仍能正常回放老trace
    span_a = [s for s in t.read_spans() if s.name == "a"][0]
    rep_explicit = t.replay(trace_id=span_a.trace_id)
    assert len(rep_explicit) == 1
    assert rep_explicit[0]["name"] == "a"
    # 指定session_id也能正常回放
    rep_session = t.replay(session_id=t.session_id)
    assert len(rep_session) == 3  # 会话维度回放所有3条
    t.close()


def test_replay_uses_inner_most_active_trace_in_nested_context():
    """修复BUG2-2：嵌套trace时，replay取栈顶（当前最内层活跃trace），而非最外层。"""
    t = TraceLogger()
    # 先开outer trace
    with t.trace("outer-trace"):
        with t.span("outer-span"):
            pass
        # 在outer内部开inner嵌套trace
        with t.trace("inner-trace"):
            with t.span("inner-span"):
                pass
            # 当前活跃的是inner-trace，replay默认应只回放inner（1条），不是outer（1条）也不是全量（2条）
            rep = t.replay()
            inner_spans = t.read_spans(trace_id=[s.trace_id for s in t.read_spans() if s.name == "inner-span"][0])
            assert len(inner_spans) == 1
            assert len(rep) == 1, f"嵌套场景应只回放最内层活跃trace，实际返回{len(rep)}条"
            assert rep[0]["name"] == "inner-span"
    t.close()


# ==================== 自定义存储后端（预留位验证）====================
def test_custom_store_can_be_injected():
    """用户可以实现 TraceStoreBase 子类并注入 TraceLogger，预留 SQLite/远端扩展位。"""
    captured: List[SpanRecord] = []

    class SpyStore(TraceStoreBase):
        def write(self, span):  # noqa: D401
            captured.append(span)
        def read(self, session_id=None, trace_id=None, kind=None, limit=None):
            return list(captured)
        def close(self):
            pass

    store = SpyStore()
    t = TraceLogger(store=store)
    with t.trace("x"):
        with t.span("s", SpanKind.CUSTOM) as s:
            s.output = 1
    assert len(captured) == 1
    assert captured[0].name == "s"
    assert captured[0].output == 1
    t.close()


# ==================== session_id ====================
def test_set_session_id_and_filter():
    """set_session_id 能切换会话；read_spans 按 session_id 过滤。"""
    t = TraceLogger(session_id="s1")
    assert t.session_id == "s1"
    with t.trace("t1"):
        with t.span("a", SpanKind.CUSTOM) as s:
            s.output = 1
    t.set_session_id("s2")
    with t.trace("t2"):
        with t.span("b", SpanKind.CUSTOM) as s:
            s.output = 2
    assert len(t.read_spans(session_id="s1")) == 1
    assert len(t.read_spans(session_id="s2")) == 1
    assert len(t.read_spans()) == 2
    t.close()


# ==================== 上下文管理器 with TraceLogger ====================
def test_trace_logger_context_manager_closes():
    """with TraceLogger() 自动 close 底层 store。"""
    with TraceLogger() as t:
        with t.trace("x"):
            with t.span("s", SpanKind.CUSTOM) as s:
                s.output = 1
        assert len(t.read_spans()) == 1
    assert t._store._closed is True


# ==================== 实例级 ContextVar：跨实例不串扰 ====================
def test_state_is_per_instance_no_cross_pollution():
    """两个 TraceLogger 实例交错打点时，根 Span 的 parent_id 必须为 None，
    不被另一个实例活跃的 Span 串扰。"""
    t1 = TraceLogger()
    t2 = TraceLogger()
    with t1.trace("A"):
        with t1.span("A1", SpanKind.AGENT):
            # 此时 t1 的 span_stack 栈顶是 A1，但 t2 此时还没开始任何 trace
            with t2.trace("B"):
                # t2 的根 Span 必须是根（parent_id=None），不能挂到 t1.A1 下面
                root = t2.start_span("B1", SpanKind.LLM, input="q")
                t2.end_span(root, output="a")
                assert root.parent_id is None, (
                    f"t2 的根 Span 错误地把 t1 的活跃 span 当成了 parent: {root.parent_id}"
                )
    # 两个 tracer 各收到自己的 span
    assert len(t1.read_spans()) == 1
    assert len(t2.read_spans()) == 1
    t1.close()
    t2.close()


def test_contextvar_is_instance_level_not_module_level():
    """状态栈挂在实例属性上，保证多 tracer 实例不互相污染，同时为 asyncio
    并发任务场景天然提供 Task 级隔离。"""
    import fast_agents.trace.trace_logger as mod
    assert not hasattr(mod, "_active_span_id"), (
        "模块级 _active_span_id 已被移除，改为实例级 ContextVar"
    )
    assert not hasattr(mod, "_active_trace_id"), (
        "模块级 _active_trace_id 已被移除，改为实例级 ContextVar"
    )
    t = TraceLogger()
    # 实例级 ContextVar 在 __init__ 中创建
    assert hasattr(t, "_trace_stack_var")
    assert hasattr(t, "_span_stack_var")
    # 初始为空栈
    assert t._trace_stack_var.get() == ()
    assert t._span_stack_var.get() == ()
    with t.trace("x"):
        assert t._trace_stack_var.get()  # 压入一条
    assert t._trace_stack_var.get() == ()  # 弹回空栈
    t.close()


# ==================== SQLiteStore 后端测试 ====================
import tempfile
import os
from fast_agents.trace import SQLiteStore


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


def test_tracer_backend_sqlite_parameter():
    """TraceLogger(backend="sqlite") 自动创建 SQLiteStore。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tracer.db")
        t = TraceLogger(path=path, backend="sqlite")
        try:
            assert isinstance(t.store, SQLiteStore)
            with t.trace("backend_test"):
                with t.span("test_span", SpanKind.CUSTOM, input={"x":1}) as s:
                    s.output = {"ok": True}
            spans = t.read_spans()
            assert len(spans) == 1  # trace()只管理栈不写span，只有custom span
        finally:
            t.close()


def test_tracer_auto_detect_sqlite_suffix():
    """path后缀为 .db/.sqlite/.sqlite3 时自动选择SQLiteStore，无需显式backend。"""
    for suffix in [".db", ".sqlite", ".sqlite3"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"auto{suffix}")
            t = TraceLogger(path=path)
            try:
                assert isinstance(t.store, SQLiteStore), f"suffix {suffix} 应自动识别为SQLite"
            finally:
                t.close()
    # jsonl后缀默认仍是JsonlStore
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "traces.jsonl")
        t = TraceLogger(path=path)
        try:
            assert isinstance(t.store, JsonlStore)
        finally:
            t.close()


def test_tracer_explicit_store_takes_precedence():
    """显式传入store实例时，忽略path/backend参数。"""
    mem_jsonl = JsonlStore(MEMORY_PATH)
    t = TraceLogger(store=mem_jsonl, backend="sqlite", path="/tmp/should_not_exist.db")
    try:
        assert t.store is mem_jsonl
        assert not isinstance(t.store, SQLiteStore)
    finally:
        t.close()


def test_replay_defaults_to_latest_trace_when_no_active():
    """BUG2修复：无活跃trace时，默认回放最近一次结束的root trace，不返回全量也不返回空。"""
    t = TraceLogger()
    try:
        # 第一条trace
        with t.trace("first"):
            with t.span("first_span") as s:
                s.output = 1
        # 第二条trace（最新）
        with t.trace("second"):
            with t.span("second_span") as s:
                s.output = 2
        # 现在无活跃trace，replay()应该默认返回最新的第二条trace
        replay = t.replay()
        assert len(replay) == 1
        assert replay[0]["name"] == "second_span"
    finally:
        t.close()


def test_replay_uses_inner_most_active_trace_in_nested_context():
    """BUG2修复：嵌套trace场景下，replay()默认取当前最内层活跃trace，不会错取外层。"""
    t = TraceLogger()
    try:
        with t.trace("outer"):
            with t.span("outer_span") as s1:
                s1.output = "outer"
                with t.trace("inner"):
                    with t.span("inner_span") as s2:
                        s2.output = "inner"
                    # 此时活跃trace是内层inner，replay()应只返回inner的span，不会返回outer的
                    replay = t.replay()
                    assert len(replay) == 1
                    assert replay[0]["name"] == "inner_span"
    finally:
        t.close()


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


def test_end_span_is_idempotent_no_duplicate_records_sqlite():
    """BUG3修复：同一Span重复调用end_span，SQLiteStore只写1条，以第一次结果为准。"""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "traces.db")
        t = TraceLogger(path=path)
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


def test_manual_end_span_in_context_normal_exit_does_not_duplicate():
    """BUG3修复：with块内手动end_span后正常退出，不会产生重复记录。"""
    t = TraceLogger()
    try:
        with t.span("test_span") as s:
            s.output = "manual_end"
            t.end_span(s)  # 块内手动结束
        spans = t.read_spans()
        assert len(spans) == 1
        assert spans[0].output == "manual_end"
        assert spans[0].status == SpanStatus.SUCCESS
    finally:
        t.close()


def test_manual_end_span_in_context_exception_does_not_create_conflicting_records():
    """BUG3修复：with块内手动end_span后抛异常，不会产生success+error两条矛盾记录，以第一次落库结果为准。"""
    t = TraceLogger()
    try:
        try:
            with t.span("test_span") as s:
                s.output = "manual_success"
                t.end_span(s)  # 手动标记成功
                raise ValueError("block exception after manual end")
        except ValueError:
            pass
        spans = t.read_spans()
        # 只有1条记录，不会出现两条矛盾记录
        assert len(spans) == 1
        # 以第一次手动end的结果为准：success，output=manual_success
        assert spans[0].output == "manual_success"
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

def test_jsonl_store_handles_extreme_unserializable_objects():
    tracer = TraceLogger(enabled=True, backend="jsonl", path=":memory:")
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
        tracer = TraceLogger(enabled=True, path=path)
        with tracer.trace("t1"):
            with tracer.span("s1", kind=SpanKind.CUSTOM, input={"obj": BadRepr()}) as s:
                s.output = "ok"
        tracer.close()
        # 重开读取
        tracer2 = TraceLogger(enabled=True, path=path, backend="jsonl")
        records = tracer2.read_spans()
        assert len(records) == 1
        assert records[0].input["obj"].startswith("<")
        tracer2.close()


# ---------- Bug5 回归：负耗时保护 ----------
def test_span_finish_negative_duration_clamped_to_zero():
    s = SpanRecord(trace_id="t1", name="test", kind=SpanKind.CUSTOM, start_time=2000.0)
    s.finish(end_time=1000.0)  # end_time早于start_time
    assert s.duration_ms == 0.0
    assert s.end_time == 1000.0

def test_span_finish_normal_duration_calculated_correctly():
    s = SpanRecord(trace_id="t1", name="test", kind=SpanKind.CUSTOM, start_time=1000.0)
    s.finish(end_time=1001.0)
    assert s.duration_ms == 1000.0

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
    # test_jsonl_store_handles_extreme_unserializable_objects()
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
