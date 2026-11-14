"""trace K3 成本日报单元测试（CostDailyReport / DayCostMetrics / TraceEvaluator.daily_cost_report）。

覆盖点：
    - 空数据返回空报告（不抛异常）
    - 单天单 trace 成本计算正确（token/费用按价目表前缀匹配）
    - 跨天聚合（按 UTC+8 自然日分组）
    - 多天合计字段正确
    - 日期过滤（start_date / end_date）生效
    - session_id 过滤生效
    - 自定义价目表覆盖默认；最长前缀优先匹配（gpt-4o-mini 优先于 gpt-4o）
    - 未命中模型走 DEFAULT_PRICE 兜底
    - 工具错误、LLM 错误、trace 成功/失败计数正确
    - model_breakdown 按模型拆分正确
    - tool_distribution 正确
    - to_dict/to_json/to_text/to_csv 四种输出不抛异常，JSON 可反序列化
    - to_csv(path=) 落盘 roundtrip 成功（utf-8-sig）
    - 脏数据（metadata 非 dict / usage 字段缺失 / 无 root span）不抛异常
    - 文件后端（jsonl/sqlite）路径构造评估器后可跑日报
    - 默认关闭：不调用 daily_cost_report 时零开销（import 不触发计算）
    - timezone_offset 切换（UTC）日期归属正确

运行方式（项目根目录）:
    python tests\trace\test_trace_cost.py
    （无需 pytest，纯标准库）
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, tmp_path

from fast_agents.trace import (
    DEFAULT_PRICE_TABLE,
    CostDailyReport,
    DayCostMetrics,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceEvaluator,
    TraceLogger,
)
from fast_agents.trace.evaluator import DEFAULT_PRICE
from fast_agents.trace.models import _new_id


# ==================== 工具：造 span ====================
def _make_span(
    trace_id: str,
    parent_id: Any,
    name: str,
    kind: SpanKind,
    *,
    start_ts: float,
    duration_ms: float,
    status: SpanStatus = SpanStatus.SUCCESS,
    error: str = None,
    metadata: dict = None,
    session_id: str = "sess_cost_test",
) -> SpanRecord:
    """构造一条 SpanRecord。"""
    if duration_ms is None:
        end_ts = start_ts
        dur = None
    else:
        end_ts = start_ts + duration_ms / 1000.0
        dur = float(duration_ms)
    return SpanRecord(
        span_id=_new_id(),
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        start_time=start_ts,
        end_time=end_ts,
        duration_ms=dur,
        status=status,
        error=error,
        metadata=metadata if metadata is not None else {},
        session_id=session_id,
    )


def _write_spans(tracer: TraceLogger, spans: List[SpanRecord]):
    for s in spans:
        tracer.store.write(s)


# ==================== Fixture ====================
def fresh_tracer():
    t = TraceLogger()
    yield t


# ==================== 空数据 ====================
def test_empty_spans_returns_empty_report(fresh_tracer):
    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    assert isinstance(rep, CostDailyReport)
    assert rep.total_traces == 0
    assert rep.total_tokens == 0
    assert rep.total_cost_usd == 0.0
    assert rep.days == []
    # 输出方法不抛异常
    assert "成本日报" in rep.to_text()
    j = json.loads(rep.to_json())
    assert j["total_traces"] == 0
    csv_text = rep.to_csv()
    # CSV 仅表头
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "date" in lines[0]


# ==================== 单天单 trace ====================
def test_single_day_single_trace_cost(fresh_tracer):
    """单条 trace 含 1 次 gpt-4o-mini 调用：
    prompt=1_000_000、completion=500_000 → input=0.15/1M=0.15, output=0.60/1M=0.30, total=0.45$"""
    now = time.time() - 3600  # 1 小时前，确保在今天（CST）
    root_id = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now,
                   duration_ms=1000),
        _make_span("t1", root_id, "llm.call", SpanKind.LLM, start_ts=now + 0.05,
                   duration_ms=800,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}}),
    ]
    spans[0].span_id = root_id  # 让 root 有 span_id
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    assert rep.total_traces == 1
    assert rep.total_prompt_tokens == 1_000_000
    assert rep.total_completion_tokens == 500_000
    assert rep.total_tokens == 1_500_000
    # gpt-4o-mini: input 0.15 / output 0.60 per 1M
    assert abs(rep.total_input_cost_usd - 0.15) < 1e-6
    assert abs(rep.total_output_cost_usd - 0.30) < 1e-6
    assert abs(rep.total_cost_usd - 0.45) < 1e-6
    assert rep.total_llm_calls == 1
    assert rep.total_errors == 0
    assert rep.overall_success_rate == 1.0
    assert len(rep.days) == 1
    day = rep.days[0]
    assert day.trace_count == 1
    assert day.successful_traces == 1
    assert day.llm_calls == 1
    assert day.total_tokens == 1_500_000
    assert abs(day.total_cost_usd - 0.45) < 1e-6
    # model_breakdown 包含 gpt-4o-mini
    assert "gpt-4o-mini" in day.model_breakdown
    mb = day.model_breakdown["gpt-4o-mini"]
    assert mb["calls"] == 1
    assert mb["prompt_tokens"] == 1_000_000
    assert mb["completion_tokens"] == 500_000


# ==================== 前缀匹配（最长前缀优先） ====================
def test_longest_prefix_matching(fresh_tracer):
    """gpt-4o-mini 前缀比 gpt-4o 长，应优先匹配 gpt-4o-mini 的价格。"""
    now = time.time() - 1800
    root_id = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", root_id, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini-2024-07-18",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}),
    ]
    spans[0].span_id = root_id
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    # 如果错误匹配 gpt-4o（2.5/10），input_cost=2.5；正确匹配 gpt-4o-mini（0.15/0.6），input_cost=0.15
    assert abs(rep.total_input_cost_usd - 0.15) < 1e-6, (
        f"应匹配 gpt-4o-mini 价，实际 input_cost={rep.total_input_cost_usd}")


# ==================== 未命中模型走 DEFAULT_PRICE 兜底 ====================
def test_unknown_model_falls_back_to_default(fresh_tracer):
    now = time.time() - 600
    root_id = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", root_id, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "some-future-model-xyz",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000}}),
    ]
    spans[0].span_id = root_id
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    d_in, d_out = DEFAULT_PRICE
    expected_in = 1_000_000 * d_in / 1_000_000
    expected_out = 2_000_000 * d_out / 1_000_000
    assert abs(rep.total_input_cost_usd - expected_in) < 1e-6
    assert abs(rep.total_output_cost_usd - expected_out) < 1e-6


# ==================== Bug4 回归：未登记新模型不得误命中前缀老模型 ====================
def test_unregistered_new_model_not_matched_to_prefix_old(fresh_tracer):
    """Bug4 回归：gpt-4.1 未登记，不得被误判成 gpt-4(30/60)，应走 DEFAULT_PRICE 兜底。"""
    now = time.time() - 600
    root_id = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", root_id, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4.1",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}),
    ]
    spans[0].span_id = root_id
    _write_spans(fresh_tracer, spans)

    rep = TraceEvaluator(fresh_tracer).daily_cost_report()
    d_in, d_out = DEFAULT_PRICE
    assert abs(rep.total_input_cost_usd - d_in) < 1e-6, (
        f"gpt-4.1 应走兜底 {d_in}，实际 {rep.total_input_cost_usd}（疑似误命中 gpt-4 前缀）")
    assert abs(rep.total_output_cost_usd - d_out) < 1e-6


def test_registered_model_with_version_suffix_still_matches(fresh_tracer):
    """已登记模型带「-日期」后缀仍命中族价目（gpt-4o-2024-08-06 → gpt-4o 2.5/10）。"""
    now = time.time() - 600
    root_id = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", root_id, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-2024-08-06",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}),
    ]
    spans[0].span_id = root_id
    _write_spans(fresh_tracer, spans)

    rep = TraceEvaluator(fresh_tracer).daily_cost_report()
    assert abs(rep.total_input_cost_usd - 2.5) < 1e-6, (
        f"gpt-4o-2024-08-06 应命中 gpt-4o(2.5)，实际 {rep.total_input_cost_usd}")


# ==================== 跨天聚合 ====================
def test_multi_day_grouping(fresh_tracer):
    """造 3 条 trace：dayA 2 条、dayB 1 条，验证天数与日级聚合正确。"""
    now = time.time()
    dayA_ts = now - 86400 * 3
    dayB_ts = now - 86400 * 1
    # dayA trace1: deepseek-chat 1M prompt
    rid1 = _new_id()
    s1 = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=dayA_ts, duration_ms=1000),
        _make_span("t1", rid1, "llm.call", SpanKind.LLM, start_ts=dayA_ts + 0.01, duration_ms=800,
                   metadata={"model": "deepseek-chat",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}),
    ]
    s1[0].span_id = rid1
    # dayA trace2: deepseek-chat 0 prompt + 1M completion
    rid2 = _new_id()
    s2 = [
        _make_span("t2", None, "agent.run", SpanKind.AGENT, start_ts=dayA_ts + 3600, duration_ms=1000),
        _make_span("t2", rid2, "llm.call", SpanKind.LLM, start_ts=dayA_ts + 3600 + 0.01, duration_ms=800,
                   metadata={"model": "deepseek-chat",
                             "usage": {"prompt_tokens": 0, "completion_tokens": 1_000_000}}),
    ]
    s2[0].span_id = rid2
    # dayB trace3: gpt-4o 1M+1M
    rid3 = _new_id()
    s3 = [
        _make_span("t3", None, "agent.run", SpanKind.AGENT, start_ts=dayB_ts,
                   duration_ms=2000, status=SpanStatus.ERROR, error="failed"),
        _make_span("t3", rid3, "llm.call", SpanKind.LLM, start_ts=dayB_ts + 0.01, duration_ms=1500,
                   metadata={"model": "gpt-4o",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}),
        _make_span("t3", rid3, "tool.search", SpanKind.TOOL, start_ts=dayB_ts + 1.6, duration_ms=300,
                   status=SpanStatus.ERROR, error="timeout"),
    ]
    s3[0].span_id = rid3
    _write_spans(fresh_tracer, s1 + s2 + s3)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    assert rep.total_traces == 3
    total_succ = sum(d.successful_traces for d in rep.days)
    total_fail = sum(d.failed_traces for d in rep.days)
    assert total_succ == 2
    assert total_fail == 1
    assert abs(rep.overall_success_rate - 2/3) < 1e-6
    assert len(rep.days) == 2
    # dayA
    dA = rep.days[0]
    assert dA.trace_count == 2
    assert dA.successful_traces == 2
    assert dA.failed_traces == 0
    assert dA.llm_calls == 2
    assert dA.prompt_tokens == 1_000_000
    assert dA.completion_tokens == 1_000_000
    # deepseek-chat: 0.14/0.28 → 0.14 + 0.28 = 0.42
    assert abs(dA.total_cost_usd - 0.42) < 1e-6
    # dayB
    dB = rep.days[1]
    assert dB.trace_count == 1
    assert dB.failed_traces == 1
    assert dB.llm_calls == 1
    assert dB.tool_calls == 1
    assert dB.tool_error_count == 1
    # gpt-4o: 2.5+10=12.5
    assert abs(dB.total_cost_usd - 12.5) < 1e-6
    assert "search" in dB.tool_distribution
    assert dB.tool_distribution["search"] == 1


# ==================== 日期范围过滤 ====================
def test_date_range_filter(fresh_tracer):
    import datetime
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    now = time.time()
    rid1 = _new_id(); rid2 = _new_id()
    s_today = [
        _make_span("t_today", None, "agent.run", SpanKind.AGENT, start_ts=now - 60, duration_ms=500),
        _make_span("t_today", rid1, "llm.call", SpanKind.LLM, start_ts=now - 59, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 100, "completion_tokens": 100}}),
    ]
    s_today[0].span_id = rid1
    s_y = [
        _make_span("t_y", None, "agent.run", SpanKind.AGENT, start_ts=now - 86400 - 60, duration_ms=500),
        _make_span("t_y", rid2, "llm.call", SpanKind.LLM, start_ts=now - 86400 - 59, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 2000, "completion_tokens": 2000}}),
    ]
    s_y[0].span_id = rid2
    _write_spans(fresh_tracer, s_today + s_y)

    ev = TraceEvaluator(fresh_tracer)
    rep_all = ev.daily_cost_report()
    assert rep_all.total_traces == 2

    rep_today = ev.daily_cost_report(start_date=today, end_date=today)
    assert rep_today.total_traces == 1
    assert rep_today.days[0].date == today
    assert rep_today.total_prompt_tokens == 100

    rep_y = ev.daily_cost_report(start_date=yesterday, end_date=yesterday)
    assert rep_y.total_traces == 1
    assert rep_y.days[0].date == yesterday
    assert rep_y.total_prompt_tokens == 2000


# ==================== session_id 过滤 ====================
def test_session_filter(fresh_tracer):
    now = time.time() - 60
    rid1 = _new_id(); rid2 = _new_id()
    s_a = [
        _make_span("ta", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500,
                   session_id="sessA"),
        _make_span("ta", rid1, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 5000, "completion_tokens": 0}},
                   session_id="sessA"),
    ]
    s_a[0].span_id = rid1
    s_b = [
        _make_span("tb", None, "agent.run", SpanKind.AGENT, start_ts=now + 10, duration_ms=500,
                   session_id="sessB"),
        _make_span("tb", rid2, "llm.call", SpanKind.LLM, start_ts=now + 10.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 9999, "completion_tokens": 0}},
                   session_id="sessB"),
    ]
    s_b[0].span_id = rid2
    _write_spans(fresh_tracer, s_a + s_b)

    ev = TraceEvaluator(fresh_tracer)
    rep_a = ev.daily_cost_report(session_id="sessA")
    assert rep_a.total_traces == 1
    assert rep_a.total_prompt_tokens == 5000


# ==================== 自定义价目表覆盖默认 ====================
def test_custom_price_table_override(fresh_tracer):
    now = time.time() - 60
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "my-private-model",
                             "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    # 自定义价：input 1.0 / output 2.0 per 1M
    rep = ev.daily_cost_report(price_table={"my-private-model": (1.0, 2.0)})
    assert abs(rep.total_input_cost_usd - 1.0) < 1e-6
    assert abs(rep.total_output_cost_usd - 2.0) < 1e-6
    assert abs(rep.total_cost_usd - 3.0) < 1e-6


# ==================== 错误计数 ====================
def test_error_counts(fresh_tracer):
    now = time.time() - 60
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=2000,
                   status=SpanStatus.ERROR, error="root failed"),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   status=SpanStatus.ERROR, error="llm api error",
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 100, "completion_tokens": 0}}),
        _make_span("t1", rid, "tool.a", SpanKind.TOOL, start_ts=now + 0.4, duration_ms=100),
        _make_span("t1", rid, "tool.b", SpanKind.TOOL, start_ts=now + 0.6, duration_ms=100,
                   status=SpanStatus.ERROR, error="tool b err"),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    day = rep.days[0]
    assert day.trace_count == 1
    assert day.failed_traces == 1
    assert day.llm_error_count == 1
    assert day.tool_error_count == 1
    assert day.error_count == 3  # root + llm + tool
    assert day.tool_calls == 2


# ==================== 脏数据不崩 ====================
def test_dirty_data_does_not_crash():
    # 构造各种异常 span：metadata=None、usage 非 dict、duration 缺失等
    spans = []
    # 无 metadata 的 LLM span
    s1 = _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=time.time() - 60, duration_ms=500)
    s2 = _make_span("t1", s1.span_id, "llm.call", SpanKind.LLM, start_ts=time.time() - 59,
                    duration_ms=300)  # metadata=None
    s3 = _make_span("t1", s1.span_id, "llm.call", SpanKind.LLM, start_ts=time.time() - 58,
                    duration_ms=300, metadata={"model": "gpt-4o-mini", "usage": "not_a_dict"})
    s4 = _make_span("t1", s1.span_id, "tool.x", SpanKind.TOOL, start_ts=time.time() - 57,
                    duration_ms=None)  # duration_ms None 允许
    s4.duration_ms = None
    s4.end_time = None
    spans = [s1, s2, s3, s4]
    _write_spans_tmp = TraceLogger()
    for s in spans:
        _write_spans_tmp.store.write(s)

    ev = TraceEvaluator(_write_spans_tmp)
    rep = ev.daily_cost_report()
    assert rep.total_traces == 1
    # 不崩就是胜利；token 累加应该是 0（因为 usage 非法）
    assert rep.total_tokens == 0


# ==================== Jsonl 文件路径 from_path ====================
def test_from_path_jsonl_roundtrip(tmp_path):
    path = str(tmp_path / "cost.jsonl")
    tracer = TraceLogger(store=path)
    now = time.time() - 30
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}),
    ]
    spans[0].span_id = rid
    for s in spans:
        tracer.store.write(s)
    tracer.close()

    ev = TraceEvaluator.from_path(path)
    rep = ev.daily_cost_report()
    assert rep.total_traces == 1
    assert rep.total_tokens == 1500


# ==================== CSV 落盘 roundtrip ====================
def test_to_csv_file_roundtrip(tmp_path, fresh_tracer):
    now = time.time() - 60
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()

    out_path = str(tmp_path / "daily.csv")
    csv_text = rep.to_csv(out_path)
    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))
    # 首行表头，次行数据
    assert reader[0][0] == "date"
    assert len(reader) >= 2
    assert reader[1][1] == "1"  # trace_count = 1
    # 文本与文件内容一致（除 BOM）
    with open(out_path, "rb") as f:
        raw = f.read()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM
    assert raw[3:].decode("utf-8") == csv_text


# ==================== JSON 可反序列化 ====================
def test_to_json_roundtrip(fresh_tracer):
    now = time.time() - 60
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    j = json.loads(rep.to_json())
    assert j["total_traces"] == 1
    assert j["total_tokens"] == 1500
    assert "days" in j
    assert isinstance(j["days"], list)


# ==================== 文本输出含关键字段 ====================
def test_to_text_contains_key_sections(fresh_tracer):
    now = time.time() - 60
    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=now, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=now + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    text = rep.to_text()
    assert "Token 成本日报" in text
    assert "总 Token 消耗" in text
    assert "总成本估算" in text
    assert "每日明细" in text


# ==================== UTC 时区切换 ====================
def test_timezone_utc_offset(fresh_tracer):
    """构造一条刚好在 CST 跨日边界的 trace，验证 timezone_offset_hours 切换时归属不同日期。
    取 UTC 时间 2026-11-10 16:00（=CST 2026-11-11 00:00 整点附近往后一点，即 CST 11 号）。"""
    import datetime
    # 构造：UTC 2026-11-10 16:30:00 → CST(+8) = 2026-11-11 00:30:00（CST 11号）；UTC=10号
    utc_dt = datetime.datetime(2026, 11, 10, 16, 30, 0, tzinfo=datetime.timezone.utc)
    ts = utc_dt.timestamp()

    rid = _new_id()
    spans = [
        _make_span("t1", None, "agent.run", SpanKind.AGENT, start_ts=ts, duration_ms=500),
        _make_span("t1", rid, "llm.call", SpanKind.LLM, start_ts=ts + 0.01, duration_ms=300,
                   metadata={"model": "gpt-4o-mini",
                             "usage": {"prompt_tokens": 100, "completion_tokens": 0}}),
    ]
    spans[0].span_id = rid
    _write_spans(fresh_tracer, spans)

    ev = TraceEvaluator(fresh_tracer)
    rep_cst = ev.daily_cost_report(timezone_offset_hours=8)
    assert rep_cst.days[0].date == "2026-11-11"

    # 切换到 UTC（offset=0），应该归到 2026-11-10
    ev2 = TraceEvaluator(fresh_tracer)
    rep_utc = ev2.daily_cost_report(timezone_offset_hours=0)
    assert rep_utc.days[0].date == "2026-11-10"


# ==================== 代表 root 选择（序号键不能与时间戳混量级） ====================
def test_daily_cost_representative_root_seq_not_mixed_with_timestamp(fresh_tracer):
    """同一条 trace 有两个 root，其中一个来自读回的老数据（无序号 _create_seq=0）。
    选代表 root 的排序键若写成 `_create_seq or start_time`，会把序号 0 换成时间戳
    （约 1.7e9），与其它 root 的整数序号混在同一比较里，导致选错代表 root，
    成本被归到错误的自然日。修复后应与其它四处一致，用 (序号, start_time) 元组键。"""
    import datetime
    cst = datetime.timezone(datetime.timedelta(hours=8))

    def _ts(y, m, d, h):
        return datetime.datetime(y, m, d, h, 0, 0, tzinfo=cst).timestamp()

    ts_old = _ts(2026, 11, 9, 12)     # 老数据：无序号 → 应作为序号最小的代表 root
    ts_new = _ts(2026, 11, 10, 12)    # 新数据：有序号

    rid_old, rid_new, tid = _new_id(), _new_id(), "t_multi_root_scale"

    old_root = _make_span(tid, None, "agent.run.old", SpanKind.AGENT, start_ts=ts_old, duration_ms=100)
    old_root.span_id = rid_old
    old_root._create_seq = 0          # 读回老数据：注入序号为 0
    new_root = _make_span(tid, None, "agent.run.new", SpanKind.AGENT, start_ts=ts_new, duration_ms=100)
    new_root.span_id = rid_new
    new_root._create_seq = 5          # 新数据：进程内分配的正序号
    llm = _make_span(tid, rid_old, "llm.call", SpanKind.LLM, start_ts=ts_old + 0.05, duration_ms=50,
                     metadata={"model": "gpt-4o-mini",
                               "usage": {"prompt_tokens": 1000, "completion_tokens": 0}})
    for s in (old_root, new_root, llm):
        fresh_tracer.store.write(s)

    ev = TraceEvaluator(fresh_tracer)
    rep = ev.daily_cost_report()
    assert len(rep.days) == 1
    # 序号最小的 root 是 seq=0 的老数据，应归到它的日期（11-09），
    # 而不是被时间戳量级顶到后面的 11-10
    assert rep.days[0].date == "2026-11-09", (
        f"代表 root 选错：应归 2026-11-09，实际 {rep.days[0].date}")


# ==================== DEFAULT_PRICE_TABLE 覆盖性 ====================
def test_default_price_table_covers_common_models():
    # 至少覆盖几个代表性模型前缀
    must_have = {"gpt-4o-mini", "gpt-4o", "deepseek-chat", "claude-3-5-sonnet", "qwen-max"}
    assert must_have.issubset(set(DEFAULT_PRICE_TABLE.keys()))
    # 每条都是 (input, output) 正数
    for k, v in DEFAULT_PRICE_TABLE.items():
        assert isinstance(v, tuple) and len(v) == 2
        assert v[0] >= 0 and v[1] >= 0


# ==================== 入口 ====================
def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def _mem_tracer():
    return TraceLogger()


def main_of_new_add():
    """新添加的测试在这里跑"""
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    test_empty_spans_returns_empty_report(_mem_tracer())
    test_single_day_single_trace_cost(_mem_tracer())
    test_longest_prefix_matching(_mem_tracer())
    test_unknown_model_falls_back_to_default(_mem_tracer())
    test_unregistered_new_model_not_matched_to_prefix_old(_mem_tracer())
    test_registered_model_with_version_suffix_still_matches(_mem_tracer())
    test_multi_day_grouping(_mem_tracer())
    test_date_range_filter(_mem_tracer())
    test_session_filter(_mem_tracer())
    test_custom_price_table_override(_mem_tracer())
    test_error_counts(_mem_tracer())
    test_dirty_data_does_not_crash()
    test_to_json_roundtrip(_mem_tracer())
    test_to_text_contains_key_sections(_mem_tracer())
    test_timezone_utc_offset(_mem_tracer())
    test_daily_cost_representative_root_seq_not_mixed_with_timestamp(_mem_tracer())
    test_default_price_table_covers_common_models()
    # tmp_path 相关
    import tests._runner as _r
    with _r.tmp_path() as tp:
        test_from_path_jsonl_roundtrip(tp)
        test_to_csv_file_roundtrip(tp, _mem_tracer())
    print("所有新增冒烟通过 ✓")
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
