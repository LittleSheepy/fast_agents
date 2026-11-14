"""trace 模块快速上手示例（基础用法）。

演示 TraceLogger 的最小闭环：
  1. 默认内存模式（:memory:），不落盘；
  2. with trace() 开启一条链路；
  3. with span() 嵌套打点，自动记录 start/end/耗时/状态；
  4. 手动 start_span/end_span 精细控制；
  5. replay() 按时间顺序回放整条链路；
  6. enabled=False 时零开销关闭。

运行方式：在项目根目录执行
    python examples/trace/basic_usage.py

无需任何 API Key，完全离线可跑。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 允许直接 python examples/trace/xxx.py 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.trace import SpanKind, TraceLogger  # noqa: E402


def do_compute(x: int) -> int:
    """一个模拟耗时的工具函数。"""
    time.sleep(0.02)
    return x * x


def main() -> None:
    # ---------- 1. 默认内存模式（不落盘，适合调试 / 测试） ----------
    tracer = TraceLogger()  # 默认 store=":memory:"（纯内存 InMemoryTraceStore）
    print("[1] session_id:", tracer.session_id)

    # ---------- 2. with trace / with span 打点 ----------
    with tracer.trace("demo-pipeline"):
        # 外层 agent span
        with tracer.span("agent.run", SpanKind.AGENT, input={"query": "计算 3 的平方"}) as agent_span:
            # 中间 decision span：决定调哪个工具
            with tracer.span("decision.step1", SpanKind.DECISION,
                             input={"thought": "需要计算平方，调用 compute"}) as dec:
                dec.output = {"action": "call compute", "args": {"x": 3}}

            # 工具执行 span：手工包装一下 do_compute
            with tracer.span("tool.compute", SpanKind.TOOL, input={"x": 3}) as tool_span:
                result = do_compute(3)
                tool_span.output = result
                tool_span.metadata["tool_name"] = "compute"

            agent_span.output = {"final_answer": f"3 的平方是 {result}"}

    # ---------- 3. 手动 start_span / end_span ----------
    with tracer.trace("manual-demo"):
        s = tracer.start_span("custom.work", SpanKind.CUSTOM, input={"note": "手动控制"})
        time.sleep(0.01)
        tracer.end_span(s, output={"done": True}, metadata={"op": "manual"})

    # ---------- 4. 读取与回放 ----------
    all_spans = tracer.read_spans()
    print(f"[4] 共记录 {len(all_spans)} 条 Span")

    print("\n=== demo-pipeline 链路回放（按时间顺序）===")
    # replay 默认回放当前活跃 trace，这里手动指定第一条 trace 的 id
    # 用 read_spans 取第一条 trace_id 再 replay
    first_trace_id = all_spans[0].trace_id
    for entry in tracer.replay(trace_id=first_trace_id):
        indent = "  " if entry["parent_id"] else ""
        ms = f"{entry['duration_ms']:.2f}ms" if entry["duration_ms"] is not None else "-"
        out_preview = str(entry["output"])[:80]
        print(f"{indent}- [{entry['kind']:>8}] {entry['name']:<20} {ms:>10}  {entry['status']}  -> {out_preview}")

    # ---------- 5. 按类型过滤 ----------
    tool_spans = tracer.read_spans(kind="tool")
    print(f"\n[5] tool 类型 Span 共 {len(tool_spans)} 条")

    # ---------- 6. 关闭 tracer（内存模式 close 即清；文件模式会 flush） ----------
    tracer.close()
    print("[6] tracer 已关闭")

    # ---------- 7. enabled=False 完全关闭（零开销） ----------
    noop = TraceLogger(enabled=False)
    with noop.trace("noop"):
        with noop.span("should_not_record", SpanKind.LLM):
            pass
    assert len(noop.read_spans()) == 0
    noop.close()
    print("[7] enabled=False 时无任何记录（零开销）")


if __name__ == "__main__":
    main()
