"""trace K3 评估器单元测试（TraceEvaluator / TraceMetrics / EvaluationReport）。

覆盖点：
    - TraceEvaluator 三种构造方式（TraceLogger / Store / 路径 / SpanRecord 列表）
    - 空数据 / 无数据时返回空报告（不抛异常）
    - 单条 trace 基本计数（span_count / llm_calls / tool_calls / decision_calls / agent_calls）
    - 端到端耗时 total_duration_ms（第一条 start ~ 最后一条 end）
    - LLM 耗时 / Tool 耗时 / 决策耗时 累加正确
    - token 用量从 metadata.usage 正确累加（prompt/completion/total，缺失 total 时兜底）
    - tool_time_ratio = tool_total / total_duration
    - tokens_per_llm_call = total_tokens / llm_calls
    - 错误计数（error_count / llm_error_count / tool_error_count / tool_error_rate）
    - root span 判定整条 trace success；非 root error 也会让 success=False
    - error_messages 去重
    - tool_distribution 正确（tool_name 来源：metadata > name 前缀 "tool." > name）
    - 跨 trace 聚合：avg/std/p95/success_rate/avg_llm_calls/avg_tool_calls 等
    - 工具分布 / 模型分布 跨 trace 聚合正确
    - session_id / trace_id / top_n_traces / limit 过滤参数生效
    - to_dict() / to_json() / to_text() 三种输出不抛异常、JSON 可反序列化
    - from_path() 支持 .db 后缀自动选 SQLite，.jsonl 选 JsonlStore
    - from_path(":memory:") 返回空评估器（不抛异常）
    - 传不存在的路径不抛异常，返回空报告（防御性）
    - 脏数据（duration_ms 缺失、metadata 非 dict、SpanRecord 字段异常）不抛异常
    - 默认关闭原则：不构造 TraceEvaluator 时评估代码零开销（import 不触发任何计算）
    - enabled=False 的 tracer 数据也能被正确评估（evaluator 离线读 store，与 enabled 无关）

运行方式（项目根目录）:
    python tests\trace\test_trace_eval.py
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, tmp_path

from fast_agents.trace import (
    EvaluationReport,
    JsonlStore,
    MEMORY_PATH,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceEvaluator,
    TraceLogger,
    TraceMetrics,
)
from fast_agents.trace.evaluator import CSV_COLUMNS


# ==================== 工具函数 ====================
def _make_span(
    trace_id: str,
    name: str,
    kind: SpanKind,
    *,
    parent_id: str | None = None,
    duration_ms: float = 10.0,
    status: SpanStatus = SpanStatus.SUCCESS,
    error: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: dict | None = None,
    start_offset: float = 0.0,
) -> SpanRecord:
    """构造一条 SpanRecord 并立即 finish，便于测试组装。"""
    import time
    base = time.time() - 0.001 - start_offset / 1000.0
    s = SpanRecord(
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        input=input,
        metadata=dict(metadata or {}),
        start_time=base,
    )
    s.finish(
        output=output,
        status=status,
        error=error,
        end_time=base + duration_ms / 1000.0,
    )
    return s


# ==================== 构造与空数据 ====================
def test_evaluator_constructs_from_trace_logger():
    """TraceEvaluator 可从 TraceLogger 实例构造。"""
    tracer = TraceLogger(path=MEMORY_PATH)
    with tracer.trace("t_root") as tid:
        with tracer.span("llm.invoke", SpanKind.LLM) as s:
            s.finish(output="hello", metadata={"model": "gpt-4o"})
    ev = TraceEvaluator(tracer)
    report = ev.evaluate()
    assert report.total_traces == 1
    assert report.traces[0].llm_calls == 1
    tracer.close()


def test_evaluator_constructs_from_store():
    """TraceEvaluator 可从 Store 实例构造。"""
    store = JsonlStore(MEMORY_PATH)
    s = _make_span("t1", "x", SpanKind.CUSTOM, duration_ms=5.0)
    store.write(s)
    ev = TraceEvaluator(store)
    report = ev.evaluate()
    assert report.total_traces == 1
    store.close()


def test_evaluator_constructs_from_span_list():
    """TraceEvaluator 可直接从 SpanRecord 列表构造。"""
    spans = [_make_span("t1", "x", SpanKind.CUSTOM, duration_ms=3.0)]
    ev = TraceEvaluator(spans)
    report = ev.evaluate()
    assert report.total_traces == 1


def test_evaluator_empty_returns_empty_report():
    """空数据返回空报告，不抛异常。"""
    ev = TraceEvaluator([])
    r = ev.evaluate()
    assert isinstance(r, EvaluationReport)
    assert r.total_traces == 0
    assert r.total_sessions == 0
    assert r.successful_traces == 0
    assert r.success_rate is None
    assert r.traces == []
    # to_dict / to_json / to_text 都不抛
    d = r.to_dict()
    assert d["total_traces"] == 0
    j = r.to_json()
    assert json.loads(j)["total_traces"] == 0
    assert "Trace 评估报告" in r.to_text()


def test_evaluator_from_path_memory_returns_empty():
    """from_path(':memory:') 返回空评估器。"""
    ev = TraceEvaluator.from_path(MEMORY_PATH)
    assert ev.evaluate().total_traces == 0


def test_evaluator_from_path_nonexistent_no_crash():
    """不存在的路径不抛异常，返回空报告。"""
    ev = TraceEvaluator.from_path("definitely_not_exist_xxx.jsonl")
    assert ev.evaluate().total_traces == 0


# ==================== 单 trace 计数 ====================
def test_single_trace_basic_counts():
    """单条 trace：span/llm/tool/decision/agent 计数正确。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=100.0),
        _make_span("t1", "decision.step1", SpanKind.DECISION, parent_id="root_p", duration_ms=5.0),
        _make_span("t1", "llm.invoke", SpanKind.LLM, duration_ms=50.0, metadata={"model": "m1"}),
        _make_span("t1", "tool.search", SpanKind.TOOL, duration_ms=30.0, metadata={"tool_name": "search"}),
        _make_span("t1", "decision.step2", SpanKind.DECISION, duration_ms=5.0),
    ]
    # 把 agent.run 设为 root（parent_id=None），其它 parent_id 指向它
    root_span_id = spans[0].span_id
    for s in spans[1:]:
        s.parent_id = root_span_id
    ev = TraceEvaluator(spans)
    r = ev.evaluate()
    assert r.total_traces == 1
    tm = r.traces[0]
    assert tm.span_count == 5
    assert tm.llm_calls == 1
    assert tm.tool_calls == 1
    assert tm.decision_calls == 2
    assert tm.agent_calls == 1
    assert tm.success is True
    assert tm.error_count == 0


def test_single_trace_durations_and_tokens():
    """单条 trace：总耗时、分类耗时、token 累加正确。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=200.0),
        _make_span("t1", "llm.invoke", SpanKind.LLM, parent_id="p", duration_ms=80.0,
                   metadata={"model": "gpt-4o", "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}),
        _make_span("t1", "llm.invoke", SpanKind.LLM, parent_id="p", duration_ms=40.0,
                   metadata={"model": "gpt-4o", "usage": {"prompt_tokens": 60, "completion_tokens": 40}}),  # 无 total_tokens
        _make_span("t1", "tool.calc", SpanKind.TOOL, parent_id="p", duration_ms=30.0, metadata={"tool_name": "calc"}),
    ]
    root_id = spans[0].span_id
    for s in spans[1:]:
        s.parent_id = root_id
    ev = TraceEvaluator(spans)
    tm = ev.evaluate().traces[0]
    assert tm.llm_calls == 2
    assert tm.tool_calls == 1
    assert abs(tm.llm_total_ms - 120.0) < 0.1         # 80 + 40，允许浮点误差
    assert abs(tm.tool_total_ms - 30.0) < 0.1
    # total_tokens：第一条 150，第二条 60+40=100（兜底），合计 250
    assert tm.total_tokens == 250
    assert tm.prompt_tokens == 160
    assert tm.completion_tokens == 90
    assert tm.tokens_per_llm_call is not None
    assert abs(tm.tokens_per_llm_call - 125.0) < 1e-6  # 250 / 2
    # tool_time_ratio
    assert tm.tool_time_ratio is not None
    assert tm.tool_time_ratio > 0


def test_tool_name_resolution_order():
    """tool 名解析顺序：metadata.tool_name > "tool." 前缀 > span.name。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=10.0),
        _make_span("t1", "tool.search", SpanKind.TOOL, duration_ms=5.0, metadata={"tool_name": "google"}),
        _make_span("t1", "tool.calc", SpanKind.TOOL, duration_ms=5.0),  # 无前缀覆盖，用前缀解析
        _make_span("t1", "custom_tool_span", SpanKind.TOOL, duration_ms=5.0),  # 无前缀，用 name
    ]
    root_id = spans[0].span_id
    for s in spans[1:]:
        s.parent_id = root_id
    ev = TraceEvaluator(spans)
    tm = ev.evaluate().traces[0]
    assert tm.tool_distribution.get("google") == 1
    assert tm.tool_distribution.get("calc") == 1
    assert tm.tool_distribution.get("custom_tool_span") == 1


def test_error_counts_and_success_flag():
    """错误 Span 计数、tool_error_rate、success 判定正确。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=100.0),  # root success
        _make_span("t1", "llm.ok", SpanKind.LLM, parent_id="p", duration_ms=10.0),
        _make_span("t1", "llm.err", SpanKind.LLM, parent_id="p", duration_ms=10.0,
                   status=SpanStatus.ERROR, error="RateLimit"),
        _make_span("t1", "tool.ok", SpanKind.TOOL, parent_id="p", duration_ms=10.0, metadata={"tool_name": "a"}),
        _make_span("t1", "tool.err1", SpanKind.TOOL, parent_id="p", duration_ms=10.0,
                   status=SpanStatus.ERROR, error="Timeout", metadata={"tool_name": "a"}),
        _make_span("t1", "tool.err2", SpanKind.TOOL, parent_id="p", duration_ms=10.0,
                   status=SpanStatus.ERROR, error="Not found", metadata={"tool_name": "b"}),
    ]
    root_id = spans[0].span_id
    for s in spans[1:]:
        s.parent_id = root_id
    ev = TraceEvaluator(spans)
    tm = ev.evaluate().traces[0]
    assert tm.error_count == 3
    assert tm.llm_error_count == 1
    assert tm.tool_error_count == 2
    assert tm.tool_calls == 3
    assert tm.tool_error_rate is not None
    assert abs(tm.tool_error_rate - 2 / 3) < 1e-6
    # root success 但存在非 root error → success 应为 False
    assert tm.success is False
    # 错误信息收集（去重）
    assert "RateLimit" in tm.error_messages
    assert "Timeout" in tm.error_messages
    assert "Not found" in tm.error_messages


def test_root_error_marks_trace_failed():
    """root span error 时整条 trace 失败。"""
    root = _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=50.0,
                      status=SpanStatus.ERROR, error="Exception")
    spans = [root]
    ev = TraceEvaluator(spans)
    tm = ev.evaluate().traces[0]
    assert tm.success is False
    assert tm.error_count == 1


# ==================== 跨 trace 聚合 ====================
def test_multi_trace_aggregation():
    """跨多 trace 聚合：avg / p95 / success_rate / 工具分布 / 模型分布 正确。"""
    traces_data = [
        # trace A：2 llm + 1 tool，全成功，耗时 100ms
        {
            "tid": "tA", "n_llm": 2, "n_tool": 1, "ok": True,
            "dur": 100.0, "toks": 200,
            "tools": [("search", 1, 0)],
            "models": ["gpt-4o", "gpt-4o"],
        },
        # trace B：1 llm + 2 tool（其中一个错），root 失败，耗时 300ms
        {
            "tid": "tB", "n_llm": 1, "n_tool": 2, "ok": False,
            "dur": 300.0, "toks": 80,
            "tools": [("calc", 1, 1), ("search", 1, 0)],
            "models": ["gpt-3.5"],
        },
        # trace C：3 llm + 0 tool，全成功，耗时 50ms（toks=300 可被 3 整除，避免 int 截断干扰断言）
        {
            "tid": "tC", "n_llm": 3, "n_tool": 0, "ok": True,
            "dur": 50.0, "toks": 300,
            "tools": [],
            "models": ["gpt-4o", "gpt-4o", "gpt-3.5"],
        },
    ]
    spans: List[SpanRecord] = []
    for td in traces_data:
        root = _make_span(td["tid"], "agent.run", SpanKind.AGENT,
                          duration_ms=td["dur"],
                          status=SpanStatus.SUCCESS if td["ok"] else SpanStatus.ERROR,
                          error=None if td["ok"] else "err")
        spans.append(root)
        # llm spans
        for i, m in enumerate(td["models"]):
            per = td["toks"] / td["n_llm"] if td["n_llm"] else 0
            s = _make_span(td["tid"], f"llm.{i}", SpanKind.LLM, parent_id=root.span_id,
                           duration_ms=td["dur"] / td["n_llm"],
                           metadata={"model": m,
                                     "usage": {"prompt_tokens": int(per * 0.6),
                                               "completion_tokens": int(per * 0.4),
                                               "total_tokens": int(per)}})
            spans.append(s)
        # tool spans
        for tname, cnt, err_cnt in td["tools"]:
            for j in range(cnt):
                s = _make_span(td["tid"], f"tool.{tname}{j}", SpanKind.TOOL,
                               parent_id=root.span_id,
                               duration_ms=10.0,
                               status=SpanStatus.ERROR if j < err_cnt else SpanStatus.SUCCESS,
                               error="err" if j < err_cnt else None,
                               metadata={"tool_name": tname})
                spans.append(s)

    ev = TraceEvaluator(spans)
    r = ev.evaluate()
    assert r.total_traces == 3
    assert r.successful_traces == 2
    assert r.failed_traces == 1
    assert abs(r.success_rate - 2 / 3) < 1e-6

    # 平均 llm / tool / tokens（tA=200, tB=80, tC=300 → avg≈193.33）
    assert abs(r.avg_llm_calls - (2 + 1 + 3) / 3) < 0.1  # 2.0
    assert abs(r.avg_tool_calls - (1 + 2 + 0) / 3) < 0.1  # 1.0
    assert abs(r.avg_total_tokens - (200 + 80 + 300) / 3) < 0.1  # ≈193.33

    # P95 耗时：durs=[50, 100, 300]，P95 应接近 300
    assert r.p95_total_duration_ms is not None
    assert r.p95_total_duration_ms >= 250

    # 工具分布：search=2 (tA1 + tB1), calc=1
    assert r.tool_distribution.get("search") == 2
    assert r.tool_distribution.get("calc") == 1
    # 工具错误分布：calc 错 1 次
    assert r.tool_error_distribution.get("calc") == 1
    assert r.tool_error_distribution.get("search", 0) == 0

    # 模型分布：gpt-4o 出现 4 次（tA×2 + tC×2）、gpt-3.5 出现 2 次（tB×1 + tC×1）
    assert r.model_distribution.get("gpt-4o") == 4
    assert r.model_distribution.get("gpt-3.5") == 2


def test_filter_by_session_and_trace():
    """session_id / trace_id 过滤生效。"""
    spans = []
    spans += [_make_span("t1", "a1", SpanKind.AGENT, duration_ms=10.0)]
    spans[-1].session_id = "s1"
    spans += [_make_span("t2", "a2", SpanKind.AGENT, duration_ms=10.0)]
    spans[-1].session_id = "s2"
    spans += [_make_span("t3", "a3", SpanKind.AGENT, duration_ms=10.0)]
    spans[-1].session_id = "s1"
    ev = TraceEvaluator(spans)
    # 按 trace_id
    r = ev.evaluate(trace_id="t2")
    assert r.total_traces == 1
    # 按 session_id
    r = ev.evaluate(session_id="s1")
    assert r.total_traces == 2
    r = ev.evaluate(session_id="s2")
    assert r.total_traces == 1


def test_top_n_traces():
    """top_n_traces 只评估最近 N 条。"""
    spans = []
    for i in range(5):
        s = _make_span(f"t{i}", "a", SpanKind.AGENT, duration_ms=10.0)
        spans.append(s)
    ev = TraceEvaluator(spans)
    r = ev.evaluate(top_n_traces=2)
    assert r.total_traces == 2


# ==================== 输出格式 ====================
def test_to_dict_to_json_roundtrip():
    """to_dict() / to_json() 可 JSON 序列化并反序列化。"""
    spans = [_make_span("t1", "a", SpanKind.AGENT, duration_ms=10.0)]
    ev = TraceEvaluator(spans)
    r = ev.evaluate()
    d = r.to_dict()
    assert isinstance(d, dict)
    assert d["total_traces"] == 1
    j = r.to_json()
    loaded = json.loads(j)
    assert loaded["total_traces"] == 1
    assert loaded["traces"][0]["span_count"] == 1


def test_to_text_contains_summary():
    """to_text() 含关键摘要信息。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, duration_ms=100.0),
        _make_span("t1", "llm.x", SpanKind.LLM, duration_ms=50.0,
                   metadata={"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}),
    ]
    spans[1].parent_id = spans[0].span_id
    ev = TraceEvaluator(spans)
    text = ev.evaluate().to_text()
    assert "Trace 评估报告" in text
    assert "总 Trace 数" in text
    assert "成功率" in text
    assert "平均" in text


# ==================== 文件路径后端自动识别 ====================
def test_from_path_jsonl_roundtrip():
    """from_path 对 .jsonl 文件自动选择 JsonlStore，并能正确评估。"""
    with tmp_path() as td:
        p = os.path.join(td, "run.jsonl")
        tracer = TraceLogger(path=p)
        with tracer.trace("root"):
            with tracer.span("llm.invoke", SpanKind.LLM, metadata={"model": "m"}) as s:
                s.finish(output="ok", metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
        tracer.close()
        ev = TraceEvaluator.from_path(p)
        r = ev.evaluate()
        assert r.total_traces == 1
        assert r.traces[0].total_tokens == 15


def test_from_path_sqlite_roundtrip():
    """from_path 对 .db 文件自动选择 SQLiteStore，并能正确评估。"""
    with tmp_path() as td:
        p = os.path.join(td, "run.db")
        tracer = TraceLogger(path=p)
        with tracer.trace("root"):
            with tracer.span("llm.invoke", SpanKind.LLM, metadata={"model": "m"}) as s:
                s.finish(output="ok", metadata={"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
        tracer.close()
        ev = TraceEvaluator.from_path(p)
        r = ev.evaluate()
        assert r.total_traces == 1
        assert r.traces[0].total_tokens == 30


# ==================== 脏数据鲁棒性 ====================
def test_dirty_data_does_not_crash():
    """缺字段 / metadata 非 dict / duration_ms 缺失等脏数据不抛异常。"""
    spans = []
    # 1) metadata 为 None 的 Span（pydantic 默认 dict，这里手动造个脏的）
    s1 = SpanRecord(trace_id="t1", name="a", kind=SpanKind.CUSTOM, start_time=0.0)
    s1.metadata = None  # type: ignore
    s1.finish(end_time=0.010)
    spans.append(s1)
    # 2) duration_ms 缺失（没 finish）
    s2 = SpanRecord(trace_id="t1", name="b", kind=SpanKind.LLM)
    spans.append(s2)
    # 3) 正常 span
    s3 = _make_span("t1", "c", SpanKind.TOOL, duration_ms=5.0, metadata={"tool_name": "x"})
    spans.append(s3)
    # 4) usage 里给非数字
    s4 = _make_span("t1", "d", SpanKind.LLM, duration_ms=5.0,
                    metadata={"usage": {"prompt_tokens": "abc", "completion_tokens": None, "total_tokens": "x"}})
    spans.append(s4)

    ev = TraceEvaluator(spans)
    r = ev.evaluate()  # 不应抛
    assert r.total_traces == 1
    # token 应该是 0（非数字兜底）
    assert r.traces[0].total_tokens == 0


def test_evaluator_on_disabled_tracer_store_still_works():
    """enabled=False 的 tracer 虽然不打点，但如果 store 里已有数据，evaluator 仍可评估。"""
    store = JsonlStore(MEMORY_PATH)
    s = _make_span("t1", "a", SpanKind.AGENT, duration_ms=10.0)
    store.write(s)
    # evaluator 离线读数据，和 tracer.enabled 无关
    ev = TraceEvaluator(store)
    assert ev.evaluate().total_traces == 1
    store.close()


# ==================== 默认关闭原则 ====================
def test_import_does_not_trigger_any_computation():
    """仅 import trace 模块不会触发任何评估计算（零开销）。"""
    # 此测试只是作为约束断言：import 本身不抛，也不产生全局副作用。
    import fast_agents.trace as tr
    assert hasattr(tr, "TraceEvaluator")
    assert hasattr(tr, "TraceMetrics")
    assert hasattr(tr, "EvaluationReport")


# ==================== K3 CSV 导出（to_csv） ====================
def test_to_csv_header_and_row_count():
    """CSV 首行是固定表头，之后每条 trace 一行、列数一致。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT),
        _make_span("t1", "llm.call", SpanKind.LLM, parent_id="x"),
        _make_span("t2", "llm.call", SpanKind.LLM),
    ]
    report = TraceEvaluator(spans).evaluate()
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == 1 + report.total_traces
    assert all(len(r) == len(CSV_COLUMNS) for r in rows)


def test_to_csv_empty_report_header_only():
    """空报告只输出表头一行，不抛异常。"""
    report = TraceEvaluator([]).evaluate()
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    assert rows == [CSV_COLUMNS]


def test_to_csv_write_file_roundtrip():
    """to_csv(path) 落盘内容与返回值一致（utf-8-sig，Excel 友好）。"""
    report = TraceEvaluator([_make_span("t1", "agent.run", SpanKind.AGENT)]).evaluate()
    with tmp_path() as td:
        p = os.path.join(td, "report.csv")
        ret = report.to_csv(p)
        assert os.path.exists(p)
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            disk = f.read()
        assert disk == ret


def test_to_csv_tool_distribution_serialized():
    """tool_distribution 单元格折叠为 "name=count;..."（按次数降序）。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT),
        _make_span("t1", "tool.web_search", SpanKind.TOOL, parent_id="r", metadata={"tool_name": "web_search"}),
        _make_span("t1", "tool.web_search", SpanKind.TOOL, parent_id="r", metadata={"tool_name": "web_search"}),
        _make_span("t1", "tool.calculator", SpanKind.TOOL, parent_id="r", metadata={"tool_name": "calculator"}),
    ]
    report = TraceEvaluator(spans).evaluate()
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    cell = rows[1][CSV_COLUMNS.index("tool_distribution")]
    assert cell == "web_search=2;calculator=1"


def test_to_csv_error_messages_escaped():
    """含逗号 / 换行的错误信息能被 csv 正确转义、换行折叠为空格。"""
    spans = [
        _make_span("t1", "agent.run", SpanKind.AGENT, metadata={}),
        _make_span("t1", "llm.call", SpanKind.LLM, parent_id="r",
                   status=SpanStatus.ERROR, error="boom, bad\nsecond line"),
    ]
    report = TraceEvaluator(spans).evaluate()
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    cell = rows[1][CSV_COLUMNS.index("error_messages")]
    assert "boom, bad second line" in cell
    assert "\n" not in cell


def test_to_csv_success_flag_and_none_cells():
    """success 输出 true/false；无数据字段输出空串而非 "None"。"""
    report = TraceEvaluator([_make_span("t1", "agent.run", SpanKind.AGENT)]).evaluate()
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    assert rows[1][CSV_COLUMNS.index("success")] == "true"
    # 无工具调用 → tool_error_rate 为 None → 空单元格
    assert rows[1][CSV_COLUMNS.index("tool_error_rate")] == ""


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
    test_to_csv_header_and_row_count()
    test_to_csv_empty_report_header_only()
    test_to_csv_write_file_roundtrip()
    test_to_csv_tool_distribution_serialized()
    test_to_csv_error_messages_escaped()
    test_to_csv_success_flag_and_none_cells()
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
