"""trace 模块单元测试。

主要用 InMemoryTraceStore（tests._runner.tmp_path() 临时目录），不依赖外部服务、不产生真实 API 请求。
覆盖点：
    - 默认关闭（tracer=None 零开销，不传时 LLM/Agent 行为不变）
    - SpanKind / SpanStatus / SpanRecord 字段/finish（含可选 end_time 精确对齐）
    - InMemoryTraceStore 内存模式 / append / 过滤查询
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
    - 禁用 enabled=False 时不写入任何记录
    - 实例级 ContextVar：跨 TraceLogger 实例不串扰、为 asyncio 并发做好准备

运行方式（Windows PowerShell，在项目根目录）:
    python test\\test_trace.py
    （SQLiteStore 后端用例已拆分到 tests/trace/test_trace_sqlite_store.py）
    （JsonlStore 后端用例已拆分到 tests/trace/test_trace_jsonl_store.py）
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, tmp_path

from fast_agents.trace import (
    InMemoryTraceStore,
    JsonlStore,
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


# ==================== InMemoryTraceStore 内存模式 ====================
def test_in_memory_store_write_read():
    """内存后端 write/read 正常，按写入顺序返回。"""
    store = InMemoryTraceStore()
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


def test_in_memory_store_filter_by_session_trace_kind():
    """按 session_id / trace_id / kind 过滤。"""
    store = InMemoryTraceStore()
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


def test_in_memory_store_limit():
    """limit 参数生效。"""
    store = InMemoryTraceStore()
    for i in range(10):
        s = SpanRecord(trace_id="t", name=f"n{i}", kind=SpanKind.CUSTOM)
        s.finish()
        store.write(s)
    assert len(store.read(limit=3)) == 3
    store.close()


# ==================== 内存后端生命周期：close 不清空 / clear 主动清空 ====================
def test_in_memory_store_close_does_not_clear_records():
    """close() 为 no-op：调用后记录仍在、可继续读出（锁定 close 不再清空数据的修复）。"""
    store = InMemoryTraceStore()
    s = SpanRecord(trace_id="t1", name="a", kind=SpanKind.LLM)
    s.finish(output=1)
    store.write(s)
    store.close()
    rows = store.read()
    assert len(rows) == 1
    assert rows[0].name == "a"
    # 重复 close 仍安全
    store.close()
    assert len(store.read()) == 1


def test_in_memory_store_clear_empties_records():
    """clear() 主动清空全部记录，清空后 read() 返回空列表、且可继续写入。"""
    store = InMemoryTraceStore()
    s = SpanRecord(trace_id="t1", name="a", kind=SpanKind.LLM)
    s.finish(output=1)
    store.write(s)
    assert len(store.read()) == 1
    store.clear()
    assert store.read() == []
    s2 = SpanRecord(trace_id="t2", name="b", kind=SpanKind.TOOL)
    s2.finish(output=2)
    store.write(s2)
    assert len(store.read()) == 1


def test_evaluator_after_tracer_context_exit_sees_records():
    """with TraceLogger(...) 退出（自动 close）后构造 evaluator 仍能读到数据（原「退出即空报告」坑已消除）。"""
    from fast_agents.trace.evaluator import TraceEvaluator
    store = InMemoryTraceStore()
    with TraceLogger(store=store, session_id="s1") as t:
        with t.span("agent", SpanKind.AGENT):
            pass
    report = TraceEvaluator(t).evaluate()
    assert report.total_traces == 1
    # close() 之后数据仍在（不再清空）
    t.close()
    assert TraceEvaluator(t).evaluate().total_traces == 1


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
    try:
        with raises(ValueError, match="boom"):
            with t.trace("demo"):
                with t.span("bad", SpanKind.CUSTOM):
                    raise ValueError("boom")
    except:
        pass
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


def test_parent_child_relationship_with_error():
    """嵌套 span 时 parent_id 正确指向父 span_id。"""
    t = TraceLogger()
    try:
        with t.trace("demo"):
            with t.span("parent", SpanKind.AGENT) as p:
                with t.span("child", SpanKind.LLM) as c:
                    raise ValueError("boom")
    except ValueError:
        pass
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
def test_trace_logger_context_manager_closes(tmp_path):
    """with TraceLogger(...) 自动 close 底层 store。"""
    # 默认 store=":memory:" 走 InMemoryTraceStore（无 _closed 标记），
    # 这里显式用 JsonlStore 文件模式校验 with 退出后底层 store 确已关闭。
    with TraceLogger(store=JsonlStore(str(tmp_path / "ctx.jsonl"))) as t:
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


# ==================== Bug3 回归：finish 不再吞掉 falsy 输出 ====================
def test_span_finish_keeps_falsy_output():
    """Bug3 回归：finish(output=...) 传 0 / False / "" / [] / {} 必须原样记录，不能被 `or` 丢成旧值。"""
    for falsy in (0, False, "", [], {}):
        s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.LLM)
        s.finish(output=falsy)
        assert s.output == falsy and type(s.output) is type(falsy), f"falsy={falsy!r} 被 `or` 吞掉"


def test_span_finish_none_output_keeps_previous():
    """finish() 未显式传 output 时保留已有输出，不改变旧语义。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.LLM, output="keep")
    s.finish()
    assert s.output == "keep"


def test_span_finish_falsy_error_recorded():
    """Bug3 回归：finish(error="") 等 falsy 错误值也要如实记录。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.TOOL)
    s.finish(error="", status=SpanStatus.ERROR)
    assert s.error == "" and s.status == SpanStatus.ERROR


# ==================== Bug1 回归：创建序号不污染用户 metadata ====================
def test_create_seq_not_in_user_metadata_on_new_span():
    """Bug1 回归：新建 Span 的创建序号只存私有属性，绝不写进用户 metadata，也不被 model_dump 带出。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.LLM, metadata={"k": "v"})
    assert s.metadata == {"k": "v"}
    assert "_create_seq" not in s.metadata
    assert getattr(s, "_create_seq", 0) > 0
    assert "_create_seq" not in s.model_dump()


# ==================== K1 收尾：输入/输出预览截断长度可配置 ====================

def test_truncate_preview_default_keeps_original_lengths():
    """默认（不配置）时截断长度沿用各接入点原有默认：输入 500 / 输出 300、200；行为与改动前一致。"""
    t = TraceLogger()
    # 输入侧默认 500
    assert len(t.truncate_preview("x" * 1000, 500, is_input=True)) == 500
    # 输出侧默认：Agent 300 / LLM 200（由调用点传入 default_len）
    assert len(t.truncate_preview("x" * 1000, 300)) == 300
    assert len(t.truncate_preview("x" * 1000, 200)) == 200
    # 短文本原样返回、None 保留 None（与 (x[:n] if x else None) 语义一致）
    assert t.truncate_preview("short", 500, is_input=True) == "short"
    assert t.truncate_preview(None, 300) is None
    t.close()


def test_truncate_preview_configurable_input_and_output():
    """显式配置 input_preview_len / output_preview_len 后，按配置长度截断；未配置的一侧不受影响。"""
    t = TraceLogger(input_preview_len=10, output_preview_len=20)
    assert len(t.truncate_preview("x" * 1000, 500, is_input=True)) == 10
    assert len(t.truncate_preview("x" * 1000, 300)) == 20
    # 输出侧统一走 output_preview_len（Agent 300 / LLM 200 都被 20 覆盖）
    assert len(t.truncate_preview("x" * 1000, 200)) == 20
    t.close()

    # 只配置输出侧时，输入侧仍沿用原有默认 500
    t2 = TraceLogger(output_preview_len=5)
    assert len(t2.truncate_preview("x" * 1000, 500, is_input=True)) == 500
    assert len(t2.truncate_preview("x" * 1000, 300)) == 5
    t2.close()


def test_truncate_preview_nonpositive_means_no_truncation():
    """配置 <= 0 表示不截断（完整记录）。"""
    t = TraceLogger(input_preview_len=0, output_preview_len=-3)
    assert t.truncate_preview("y" * 1000, 500, is_input=True) == "y" * 1000
    assert t.truncate_preview("y" * 1000, 200) == "y" * 1000
    t.close()


def test_truncate_preview_handles_none_and_non_str():
    """None / 非字符串原样返回，绝不抛异常（R6：打点通道不阻塞主流程）。"""
    t = TraceLogger(input_preview_len=5, output_preview_len=5)
    assert t.truncate_preview(None, 300) is None
    assert t.truncate_preview(123, 300) == 123
    d = {"a": 1}
    assert t.truncate_preview(d, 300) is d
    t.close()


def test_agent_preview_length_configurable_end_to_end():
    """端到端：配置后 Agent span 的输入/输出预览按配置长度截断（不改其它行为）。"""
    from fast_agents import ReActAgent

    @StructuredTool.from_function
    def add(a: int, b: int) -> int:
        """add two numbers"""
        return a + b

    reg = ToolRegistry()
    reg.register_tool(add)

    t = TraceLogger(input_preview_len=8, output_preview_len=3)
    llm = _make_fake_agent_llm()
    kwargs = dict(
        name="cfg-agent",
        llm=llm,
        tool_registry=reg,
        system_prompt="你是一个计算助手。",
        max_steps=5,
        tracer=t,
    )
    _disable_agent_subsystems(kwargs)
    agent = ReActAgent(**kwargs)

    with t.trace("cfg-run"):
        ans = agent.run("请把1加2算出来，越详细越好，谢谢你了")

    assert "3" in ans
    agent_spans = [s for s in t.read_spans() if s.kind == SpanKind.AGENT]
    assert len(agent_spans) >= 1
    ag = agent_spans[0]
    # 输入预览按 input_preview_len=8 截断
    assert len(ag.input["user_input"]) == 8, ag.input["user_input"]
    # 输出预览按 output_preview_len=3 截断（最终答案“答案是 3”截为前 3 字）
    assert ag.output["final_answer_preview"] == "答案是", ag.output["final_answer_preview"]
    t.close()


def test_llm_preview_length_configurable_end_to_end():
    """端到端：配置后 LLM span 的 content_preview 按配置长度截断（invoke 与 stream_invoke 都生效）。"""
    t = TraceLogger(output_preview_len=4)
    llm = _make_fake_llm(tracer=t)
    with t.trace("llm-cfg"):
        llm.invoke([{"role": "user", "content": "hi"}])
    s = t.read_spans(kind="llm")[0]
    assert s.name == "llm.invoke"
    assert s.output["content_preview"] == "mock", s.output["content_preview"]

    # stream_invoke：chunks=["hello"," world"] → "hello world"[:4] == "hell"
    with t.trace("llm-cfg2"):
        list(llm.stream_invoke([{"role": "user", "content": "hi"}]))
    s2 = [x for x in t.read_spans(kind="llm") if x.name == "llm.stream_invoke"][0]
    assert s2.output["content_preview"] == "hell", s2.output["content_preview"]
    t.close()


# =================================================
# ==================== 树形聚合 ====================
# =================================================

def test_build_tree_simple_parent_child():
    """build_tree() 基本父子关系：agent 为根、tool 与 llm 是其子节点。
    结构图示意：
    agent
      ├── tool
      └── llm
    """
    from fast_agents.trace import TraceLogger, SpanKind, InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="s1")
    agent = t.start_span("agent.run", SpanKind.AGENT)
    tool = t.start_span("tool.search", SpanKind.TOOL)
    t.end_span(tool, output="search-ok")
    llm = t.start_span("llm.invoke", SpanKind.LLM)
    t.end_span(llm, output="llm-ok")
    t.end_span(agent, output="done")

    roots = t.build_tree()
    assert len(roots) == 1, "单 trace 应只有 1 个根节点"
    root = roots[0]
    assert root["name"] == "agent.run"
    assert root["span_id"] == agent.span_id
    assert root.get("_orphan") is None
    assert len(root["children"]) == 2
    child_names = [c["name"] for c in root["children"]]
    assert child_names == ["tool.search", "llm.invoke"], f"子节点按创建顺序：{child_names}"
    for c in root["children"]:
        assert c["children"] == [], f"叶子节点 {c['name']} 不应有子节点"
        assert c.get("_orphan") is None
    # 节点字段与 replay 一致：都包含 kind/status/duration_ms/error/input/output/metadata
    for k in ("span_id","parent_id","kind","name","status","duration_ms","error","input","output","metadata","children"):
        assert k in root, f"根节点缺少字段 {k}"


def test_build_tree_deep_nesting():
    """build_tree() 支持多层嵌套（agent → tool → llm）。
    结构图示意：
    agent
      └──→ tool
        └── llm
    """
    from fast_agents.trace import TraceLogger, SpanKind, InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store)
    a = t.start_span("a", SpanKind.AGENT)
    tl = t.start_span("tl", SpanKind.TOOL)
    lm = t.start_span("lm", SpanKind.LLM)
    t.end_span(lm, output="lm-ok")
    t.end_span(tl, output="tool-ok")
    t.end_span(a, output="agent-ok")

    roots = t.build_tree()
    assert len(roots) == 1
    root = roots[0]
    assert root["name"] == "a"
    assert len(root["children"]) == 1
    assert root["children"][0]["name"] == "tl"
    assert len(root["children"][0]["children"]) == 1
    assert root["children"][0]["children"][0]["name"] == "lm"
    assert root["children"][0]["children"][0]["children"] == []


def test_build_tree_does_not_alter_replay_behavior():
    """build_tree() 是纯新增，不改变 replay() 的返回与字段结构。"""
    from fast_agents.trace import TraceLogger, SpanKind, InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store)
    a = t.start_span("a", SpanKind.AGENT)
    b = t.start_span("b", SpanKind.TOOL)
    t.end_span(b, output="b-ok")
    t.end_span(a, output="a-ok")

    rep_before = t.replay()
    _ = t.build_tree()
    rep_after = t.replay()
    assert rep_before == rep_after, "调用 build_tree() 后 replay() 应保持不变"
    for n in rep_before:
        assert "children" not in n, "replay() 返回的 dict 不应被注入 children 字段"


def test_build_tree_multi_root_preserves_all_roots():
    """同一次 read 出现多个 root（parent_id=None）时，全部作为根节点返回。"""
    from fast_agents.trace import TraceLogger, SpanKind, InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store)
    # 两条独立 trace（模拟显式传 trace_id=None 时跨 trace 聚合场景）
    with t.trace("t1"):
        a1 = t.start_span("a1", SpanKind.AGENT)
        t.end_span(a1, output="a1-ok")
    with t.trace("t2"):
        a2 = t.start_span("a2", SpanKind.AGENT)
        t.end_span(a2, output="a2-ok")
    roots = t.build_tree(session_id=t.session_id)  # 跨两条 trace 取全量
    names = sorted(r["name"] for r in roots)
    assert names == ["a1", "a2"], f"应包含两个根节点：{names}"



def test_build_tree_orphan_chain_does_not_duplicate_nodes():
    """悬空父链（A→B→C→D，A 的父不存在）下 build_tree 不重复输出节点、子树完整保留。"""
    from fast_agents.trace import TraceLogger, InMemoryTraceStore

    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="s_orphan")

    def _mk(sid, pid, name, seq):
        s = SpanRecord(span_id=sid, trace_id="t1", name=name, kind=SpanKind.AGENT, parent_id=pid)
        s._create_seq = seq
        s.start_time = 0.0
        s.end_time = 0.0
        s.duration_ms = 0.0
        s.status = SpanStatus.SUCCESS
        store.write(s)

    _mk("a", "no-such-parent", "o1", 1)
    _mk("b", "a", "o2", 2)
    _mk("c", "b", "o3", 3)
    _mk("d", "c", "o4", 4)

    roots = t.build_tree()
    # 悬空起始点 a 应被提升为 root，b/c/d 保留在其 children 下
    assert len(roots) == 1, f"应只有 1 个孤儿 root，实际 {len(roots)}"
    root = roots[0]
    assert root["span_id"] == "a"
    assert root.get("_orphan") is True
    # 子树完整
    assert root["children"][0]["span_id"] == "b"
    assert root["children"][0]["children"][0]["span_id"] == "c"
    assert root["children"][0]["children"][0]["children"][0]["span_id"] == "d"
    # 唯一节点数为 4，不重复
    def _walk(nodes, seen=None):
        if seen is None:
            seen = set()
        for n in nodes:
            if n["span_id"] in seen:
                continue
            seen.add(n["span_id"])
            _walk(n["children"], seen)
        return seen

    uniq = _walk(roots)
    assert uniq == {"a", "b", "c", "d"}, f"4 个唯一节点应全部出现：{uniq}"


def test_build_tree_single_orphan_with_child_does_not_duplicate():
    """单层悬空父引用 + 一个子节点：不重复、子节点挂在孤儿下。"""
    from fast_agents.trace import TraceLogger, InMemoryTraceStore
    from fast_agents.trace.models import SpanKind, SpanRecord, SpanStatus

    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="s_orphan2")

    def _mk(sid, pid, name, seq):
        s = SpanRecord(span_id=sid, trace_id="t1", name=name, kind=SpanKind.AGENT, parent_id=pid)
        s._create_seq = seq
        s.start_time = 0.0
        s.end_time = 0.0
        s.duration_ms = 0.0
        s.status = SpanStatus.SUCCESS
        store.write(s)

    _mk("a", "x", "orphan", 1)
    _mk("b", "a", "child", 2)

    roots = t.build_tree()
    assert len(roots) == 1
    assert roots[0]["span_id"] == "a"
    assert roots[0].get("_orphan") is True
    assert [c["span_id"] for c in roots[0]["children"]] == ["b"]


def test_build_tree_multi_root_with_orphan_no_duplicate():
    """多个正常 root + 一棵悬空子树混合场景：不重复、节点不丢失。"""
    from fast_agents.trace import TraceLogger, InMemoryTraceStore
    from fast_agents.trace.models import SpanKind, SpanRecord, SpanStatus

    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="s_mix")

    def _mk(sid, pid, name, seq):
        s = SpanRecord(span_id=sid, trace_id="t1", name=name, kind=SpanKind.AGENT, parent_id=pid)
        s._create_seq = seq
        s.start_time = 0.0
        s.end_time = 0.0
        s.duration_ms = 0.0
        s.status = SpanStatus.SUCCESS
        store.write(s)

    _mk("r1", None, "r1", 1)
    _mk("r2", None, "r2", 2)
    _mk("c1", "r1", "c1", 3)
    _mk("o", "xx", "orphan", 4)
    _mk("oc", "o", "oc", 5)

    roots = t.build_tree()
    assert len(roots) == 3, f"两个正常 root + 一个孤儿 = 3：{len(roots)}"

    def _walk(nodes, seen=None):
        if seen is None:
            seen = set()
        for n in nodes:
            if n["span_id"] in seen:
                continue
            seen.add(n["span_id"])
            _walk(n["children"], seen)
        return seen
    assert _walk(roots) == {"r1", "r2", "c1", "o", "oc"}

def test_build_tree_cycle_does_not_drop_or_duplicate_nodes():
    """病态环/自环数据下 build_tree 不丢节点、不重复、不递归死循环。"""
    from fast_agents.trace import TraceLogger, InMemoryTraceStore
    from fast_agents.trace.models import SpanKind, SpanRecord, SpanStatus

    store = InMemoryTraceStore()
    t = TraceLogger(store=store)

    def _mk(sid, pid, name, seq):
        s = SpanRecord(span_id=sid, trace_id="t_cycle", name=name, kind=SpanKind.AGENT, parent_id=pid)
        s._create_seq = seq
        s.start_time = 0.0
        s.end_time = 0.0
        s.duration_ms = 0.0
        s.status = SpanStatus.SUCCESS
        store.write(s)

    # 自环 A.parent=A
    _mk("a", "a", "self-loop", 1)
    # 双环 A↔B
    _mk("b", "c", "B", 2)
    _mk("c", "b", "C", 3)

    roots = t.build_tree(trace_id="t_cycle")

    def _walk(nodes, seen=None):
        if seen is None:
            seen = set()
        total = 0
        for n in nodes:
            if n["span_id"] in seen:
                continue
            seen.add(n["span_id"])
            total += 1 + _walk(n["children"], seen)
        return total

    total = _walk(roots)
    assert total == 3, f"环数据下应覆盖全部 3 个唯一节点（自环+双环），实际 {total}"
    assert len(roots) >= 1, "环数据应至少提升一个孤儿 root 节点"






# ==================== build_tree 悬空父子树回归 ====================
def _count_tree_nodes(nodes):
    total = 0
    for n in nodes:
        total += 1
        total += _count_tree_nodes(n.get("children", []))
    return total


def test_build_tree_orphan_chain_does_not_duplicate_nodes1():
    """悬空父链（父节点在本批次不存在）时，仅起始节点标记_orphan提升为root，子孙保留在children下，无重复节点。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="orphan_chain")
    a = t.start_span("A", SpanKind.TOOL); a.parent_id = "non_existent_parent"
    b = t.start_span("B", SpanKind.TOOL); b.parent_id = a.span_id
    c = t.start_span("C", SpanKind.TOOL); c.parent_id = b.span_id
    d = t.start_span("D", SpanKind.TOOL); d.parent_id = c.span_id
    for s in (d, c, b, a): t.end_span(s)
    roots = t.build_tree()
    real_count = len(store.read())
    assert len(roots) == 1, f"悬空链仅应A为root，实得{len(roots)}"
    assert roots[0]["name"] == "A" and roots[0].get("_orphan") is True
    assert _count_tree_nodes(roots) == real_count == 4, f"节点数应等于实际span数4，实得树节点{_count_tree_nodes(roots)}"


def test_build_tree_single_orphan_with_child_does_not_duplicate1():
    """单层悬空父+子场景无重复。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="orphan_single")
    a = t.start_span("A", SpanKind.TOOL); a.parent_id = "non_existent_parent"
    b = t.start_span("B", SpanKind.TOOL); b.parent_id = a.span_id
    for s in (b, a): t.end_span(s)
    roots = t.build_tree()
    assert len(roots) == 1 and roots[0]["name"] == "A"
    assert _count_tree_nodes(roots) == 2
    assert len(roots[0]["children"]) == 1 and roots[0]["children"][0]["name"] == "B"


def test_build_tree_multi_root_with_orphan_no_duplicate1():
    """多root（两个parent_id=None）+ 一个悬空子树场景无重复。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    store = InMemoryTraceStore()
    t = TraceLogger(store=store, session_id="multi_root_orphan")
    r1 = t.start_span("R1", SpanKind.AGENT)
    c1 = t.start_span("C1", SpanKind.TOOL)
    t.end_span(c1); t.end_span(r1)
    r2 = t.start_span("R2", SpanKind.AGENT)
    t.end_span(r2)
    a = t.start_span("A", SpanKind.TOOL); a.parent_id = "no_parent"
    c2 = t.start_span("C2", SpanKind.TOOL); c2.parent_id = a.span_id
    t.end_span(c2); t.end_span(a)
    roots = t.build_tree()
    names = sorted([r["name"] for r in roots])
    assert names == ["A", "R1", "R2"], f"应3个root，实得{names}"
    assert _count_tree_nodes(roots) == len(store.read()) == 5
    orphan = [r for r in roots if r.get("_orphan")][0]
    assert orphan["name"] == "A"


def test_error_type_auto_inferred_from_colon_separated_error():
    """error_type 从 ``"ExceptionName: msg"`` 格式自动推断（K1 收尾③）。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    from fast_agents.trace.models import SpanKind
    store = InMemoryTraceStore()
    tracer = TraceLogger(store=store, session_id="t_err")
    span = tracer.start_span("op", SpanKind.CUSTOM)
    tracer.end_span(span, output=None, error="ValueError: bad input")
    spans = tracer.replay()
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert spans[0]["error"] == "ValueError: bad input"
    assert spans[0]["error_type"] == "ValueError"


def test_error_type_with_block_exception_auto_filled():
    """with 语法里抛出的异常自动填 error_type=异常类名。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    from fast_agents.trace.models import SpanKind
    store = InMemoryTraceStore()
    tracer = TraceLogger(store=store, session_id="t_with")
    try:
        with tracer.span("op", SpanKind.TOOL):
            raise RuntimeError("ctx error")
    except RuntimeError:
        pass
    spans = tracer.replay()
    assert len(spans) == 1
    assert spans[0]["error_type"] == "RuntimeError"


def test_error_type_success_span_is_none():
    """成功 Span 的 error_type 必须为 None，不能被误填。"""
    from fast_agents.trace.in_memory_trace_store import InMemoryTraceStore
    from fast_agents.trace.models import SpanKind
    store = InMemoryTraceStore()
    tracer = TraceLogger(store=store)
    with tracer.span("ok", SpanKind.TOOL):
        pass
    spans = tracer.replay()
    assert len(spans) == 1
    assert spans[0]["status"] == "success"
    assert spans[0]["error"] is None
    assert spans[0]["error_type"] is None


def test_error_type_jsonl_roundtrip():
    """jsonl 落盘再读回，error_type 字段完整保留。"""
    import tempfile, os
    from fast_agents.trace.jsonl_store import JsonlStore
    from fast_agents.trace.models import SpanKind
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        js = JsonlStore(path)
        tracer = TraceLogger(store=js)
        span = tracer.start_span("x", SpanKind.LLM)
        tracer.end_span(span, output=None, error="TimeoutError: timed out")
        js.close()
        js2 = JsonlStore(path, append=True)
        rs = js2.read()
        js2.close()
        assert len(rs) == 1
        assert rs[0].error_type == "TimeoutError"


def test_error_type_sqlite_auto_migrate_old_db():
    """老 SQLite 库（无 error_type 列）自动补列，读回不报错且新写入正确。"""
    import tempfile, os, sqlite3, time
    from fast_agents.trace.sqlite_store import SQLiteStore
    from fast_agents.trace.models import SpanKind
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.db")
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE trace_spans (
            id INTEGER PRIMARY KEY, span_id TEXT UNIQUE NOT NULL, trace_id TEXT NOT NULL,
            parent_id TEXT, session_id TEXT, name TEXT NOT NULL, kind TEXT NOT NULL,
            start_time REAL NOT NULL, end_time REAL, duration_ms REAL, status TEXT NOT NULL,
            error TEXT, input TEXT, output TEXT, metadata TEXT NOT NULL, create_seq INTEGER)""")
        now = time.time()
        conn.execute("INSERT INTO trace_spans VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("s1","t1",None,"sess","old","tool",now,now+0.1,100.0,"error",
             "ValueError: old","null","null","{}",1))
        conn.commit()
        conn.close()
        # 打开老库，ensure_schema 自动补 error_type 列
        sq = SQLiteStore(path)
        # 新写入的 Span 必须能正确记录 error_type
        tracer = TraceLogger(store=sq)
        span = tracer.start_span("y", SpanKind.LLM)
        tracer.end_span(span, output=None, error="ConnectionError: refused")
        rs = sq.read()
        sq.close()
        # 老数据 error_type 为 None（历史数据不回填，零破坏）
        old_span = [r for r in rs if r.span_id == "s1"][0]
        assert old_span.error == "ValueError: old"
        assert old_span.error_type is None
        # 新数据 error_type 正常
        new_span = [r for r in rs if r.name == "y"][0]
        assert new_span.error_type == "ConnectionError"




# ==================== K1 收尾④：单调时钟耗时（抗墙钟跳变） ====================
def test_start_span_assigns_monotonic_clock_start():
    """TraceLogger.start_span 创建的 Span 会带单调时钟起点（finish 时用于抗墙钟跳变）。"""
    t = TraceLogger()
    try:
        s = t.start_span("m", SpanKind.CUSTOM)
        assert getattr(s, "_mono_start", None) is not None
    finally:
        t.close()


def test_direct_spanrecord_finish_keeps_wall_clock_duration():
    """直接构造的 SpanRecord（未经 TraceLogger）默认路径仍按墙钟差值算耗时（旧口径不变）。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.CUSTOM, start_time=1000.0)
    assert getattr(s, "_mono_start", None) is None
    s.finish()
    assert s.duration_ms is not None
    assert abs(s.duration_ms - (s.end_time - 1000.0) * 1000.0) < 1e-6


def test_default_duration_path_measures_real_elapsed_time():
    """默认路径（不传 end_time）耗时按单调时钟计算，量级与真实经过时间一致。"""
    import time
    t = TraceLogger()
    try:
        s = t.start_span("m", SpanKind.CUSTOM)
        time.sleep(0.02)
        t.end_span(s)
        sp = t.read_spans()[0]
        assert sp.duration_ms is not None
        assert 20.0 <= sp.duration_ms < 1000.0
        # 展示用的墙钟字段仍是 time.time()：正常情况 end >= start
        assert sp.end_time is not None and sp.end_time >= sp.start_time
    finally:
        t.close()


def test_duration_measures_monotonic_under_wall_clock_jump():
    """墙钟被向前拨动 1 小时时，duration_ms 仍为真实耗时（单调时钟），不被污染。"""
    import time
    t = TraceLogger()
    try:
        s = t.start_span("m", SpanKind.CUSTOM)
        real_time = time.time
        time.time = lambda: real_time() + 3600.0   # 模拟 NTP 向前校正 1 小时
        try:
            time.sleep(0.005)
            t.end_span(s)
        finally:
            time.time = real_time
        sp = t.read_spans()[0]
        # 旧口径会是 ~3600000ms（1 小时量级），新口径为毫秒级
        assert sp.duration_ms is not None
        assert sp.duration_ms < 1000.0, f"duration_ms={sp.duration_ms} 被墙钟跳变污染"
        assert sp.duration_ms >= 0.0
    finally:
        t.close()


def test_explicit_end_time_keeps_wall_clock_duration():
    """显式传 end_time 时仍走墙钟差值（后钩子 / 手工精确起止口径不变）。"""
    s = SpanRecord(trace_id="t1", name="n", kind=SpanKind.CUSTOM, start_time=1000.0)
    s.finish(end_time=2500.0)
    assert s.duration_ms == 1500000.0


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
    test_build_tree_cycle_does_not_drop_or_duplicate_nodes()
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
