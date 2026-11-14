"""cost_daily_demo.py —— Trace 成本日报（K3 高阶）使用示例。

演示场景：
    1. 用内存 TraceLogger 模拟 3 天的 Agent 运行数据（跨多个模型 + 工具调用 + 错误）；
    2. 用 TraceEvaluator 离线生成成本日报；
    3. 在控制台打印人类可读文本报告；
    4. 把 JSON 与 CSV 两份报告写到 examples/trace/_output/cost_daily/ 目录，
       CSV 用 utf-8-sig 编码，Windows Excel 双击直接打开中文不乱码；
    5. 演示如何传入自定义价目表覆盖默认定价。

运行（项目根目录）::

    python examples/trace/cost_daily_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fast_agents.trace import (
    SpanKind,
    SpanStatus,
    TraceEvaluator,
    TraceLogger,
)
from fast_agents.trace.models import SpanRecord, _new_id


# ---------- 造数据工具 ----------
def _mk(
    tracer: TraceLogger,
    trace_id: str,
    parent_id,
    name: str,
    kind: SpanKind,
    start_ts: float,
    duration_ms: float,
    *,
    status: SpanStatus = SpanStatus.SUCCESS,
    error: str = None,
    metadata: dict = None,
) -> SpanRecord:
    span = SpanRecord(
        span_id=_new_id(),
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        start_time=start_ts,
        end_time=start_ts + duration_ms / 1000.0,
        duration_ms=duration_ms,
        status=status,
        error=error,
        metadata=dict(metadata) if metadata else {},
        session_id=tracer.session_id,
    )
    tracer.store.write(span)
    return span


def _fake_run(
    tracer: TraceLogger,
    base_ts: float,
    *,
    model: str,
    prompt: int,
    completion: int,
    tools: list = None,       # [(tool_name, duration_ms, ok)]
    fail_root: bool = False,
) -> None:
    """模拟一次 Agent 运行：1 个 root Agent span + 1 个 LLM span + 若干 tool span。"""
    tid = f"demo_trace_{int(base_ts * 1000)}"
    root = _mk(tracer, tid, None, "agent.run", SpanKind.AGENT, base_ts, 2000,
               status=SpanStatus.ERROR if fail_root else SpanStatus.SUCCESS,
               error="agent failed" if fail_root else None)
    _mk(tracer, tid, root.span_id, "llm.chat", SpanKind.LLM, base_ts + 0.02, 1200,
        metadata={"model": model,
                  "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                            "total_tokens": prompt + completion}})
    cursor = base_ts + 1.3
    if tools:
        for tname, dur, ok in tools:
            _mk(tracer, tid, root.span_id, f"tool.{tname}", SpanKind.TOOL, cursor, dur,
                status=SpanStatus.SUCCESS if ok else SpanStatus.ERROR,
                error=f"{tname} error" if not ok else None)
            cursor += dur / 1000.0 + 0.05


def main():
    tracer = TraceLogger()  # 默认 store=":memory:"（InMemoryTraceStore）
    CST = timezone(timedelta(hours=8))
    now = datetime.now(CST)

    # ---------- 造 3 天数据 ----------
    # Day-2：轻量问答 3 次，全部用 gpt-4o-mini
    for i in range(3):
        _fake_run(
            tracer,
            (now - timedelta(days=2, hours=2 - i)).timestamp(),
            model="gpt-4o-mini",
            prompt=800 + i * 200,
            completion=300 + i * 50,
        )
    # Day-1：复杂任务 2 次，用 gpt-4o + 工具；1 次失败
    _fake_run(
        tracer,
        (now - timedelta(days=1, hours=3)).timestamp(),
        model="gpt-4o-2024-08-06",
        prompt=5000, completion=1200,
        tools=[("web_search", 400, True), ("calculator", 50, True)],
    )
    _fake_run(
        tracer,
        (now - timedelta(days=1, hours=1)).timestamp(),
        model="gpt-4o",
        prompt=8000, completion=2000,
        tools=[("web_search", 600, False)],  # 工具失败
        fail_root=True,
    )
    # 今天：混合 deepseek-chat 一次便宜调用
    _fake_run(
        tracer,
        (now - timedelta(hours=2)).timestamp(),
        model="deepseek-chat",
        prompt=2000, completion=500,
        tools=[("web_search", 300, True)],
    )

    # ---------- 生成成本日报 ----------
    evaluator = TraceEvaluator(tracer)
    report = evaluator.daily_cost_report()

    print(report.to_text())

    # ---------- 导出文件 ----------
    out_dir = os.path.join(os.path.dirname(__file__), "_output", "cost_daily")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "cost_daily_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"\n✓ JSON 已写入: {json_path}")

    csv_path = os.path.join(out_dir, "cost_daily_report.csv")
    report.to_csv(csv_path)  # 默认 utf-8-sig，Excel 友好
    print(f"✓ CSV  已写入: {csv_path}")

    # ---------- 自定义价目表演示 ----------
    print("\n" + "-" * 60)
    print("【自定义价目表】用内部模型价覆盖默认：")
    custom = evaluator.daily_cost_report(price_table={
        "my-internal-llm": (0.05, 0.10),  # 假设内部模型极便宜
        # 不传其它模型，会自动与默认表合并
    })
    print(f"  （默认表模型数：{len(custom.price_table_used)}）")
    print(f"  总调用数：{custom.total_llm_calls}，总成本 ${custom.total_cost_usd:.4f}（与默认一致，因为没造 my-internal-llm 的数据）")


if __name__ == "__main__":
    main()
