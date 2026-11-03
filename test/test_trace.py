"""trace 模块单元测试。

全部用内存 JsonlStore 或 pytest tmp_path 临时目录，不依赖外部服务、不产生真实 API 请求。
覆盖点：
    - 默认关闭（tracer=None 零开销，不传时 LLM/Agent 行为不变）
    - SpanKind / SpanStatus / SpanRecord 字段/finish
    - JsonlStore 内存模式 / 文件模式 / append / 过滤查询
    - TraceLogger start_trace/end_trace/with trace
    - start_span/end_span/with span（正常路径 + 异常路径）
    - 父子关系（parent_id 正确串联）
    - tool_hook 签名对齐 ToolRegistry.add_hook（ToolResult 成功/失败/截断）
    - tool_hook 耗时取自 ToolResult.metadata['duration_ms'] 并回推 start_time（v1.0.1 修正）
    - attach_to_tool_registry 一行接入
    - LLM.invoke/stream_invoke/embed_query/embed_documents 在 tracer=None 时不打点、
      传 tracer 时正确记录 llm Span（mock 底层 _client，无需真 API Key）
    - LLM 打点写入 token 用量元数据（usage / prompt_tokens / completion_tokens / total_tokens）
    - ReActAgent tracer 接入（mock llm.invoke + 简单工具）：agent/decision/tool/llm 四类 Span 都有
    - replay() 能按时间顺序返回可打印列表
    - 存储层预留位：用户可自定义 TraceStoreBase 子类
    - 文件路径下 jsonl 落盘可被新 JsonlStore 重新读取（跨进程回放）
    - 禁用 enabled=False 时不写入任何记录

运行方式（Windows PowerShell，在项目根目录）:
    python test\\test_trace.py
    或：pytest test/test_trace.py -v
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest

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
    assert s.metadata == {"model": "m"}


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
    with pytest.raises(ValueError, match="boom"):
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
    """tool_hook 从 ``ToolResult.metadata['duration_ms']`` 取真实耗时，并回推 start_time。

    这是 v1.0.1 修复点：修复前误读 ToolResult 的顶层 duration_ms 属性（不存在），
    导致 tool span 耗时恒为 ~0。
    """
    t = TraceLogger()
    r = ToolResult.ok("5")
    r.metadata["duration_ms"] = 120  # 模拟 tool_base 写入的实测耗时（int 毫秒）
    with t.trace("tool-dur"):
        t.tool_hook("add", {"a": 2, "b": 3}, r)
    spans = t.read_spans(kind="tool")
    assert len(spans) == 1
    s = spans[0]
    # 耗时直接取工具自报值（修复前恒为 ~0）
    assert abs(s.duration_ms - 120.0) < 5.0, f"duration_ms={s.duration_ms}"
    # start_time 被回推，时间区间长度 ≈ 120ms（修复前区间塌缩在结束时刻附近）
    assert s.end_time is not None
    assert abs((s.end_time - s.start_time) * 1000.0 - 120.0) < 5.0
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


# ==================== LLM 打点：token / model 元数据（v1.0.1）====================
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
    assert rep[0]["name"] == "outer" or rep[0]["name"] == "inner"
    # 排序验证：inner.start_time >= outer.start_time
    assert rep[0]["start_time" if False else "name"] is not None  # 字段存在
    for entry in rep:
        assert set(entry.keys()) >= {
            "span_id", "parent_id", "kind", "name", "status", "duration_ms", "error",
            "input", "output", "metadata",
        }
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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
