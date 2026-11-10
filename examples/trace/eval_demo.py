"""trace K3 评估工具示例：模拟几次 Agent 运行，然后离线评估并打印报告。

运行方式（项目根目录）:
    python examples\\trace\\eval_demo.py

说明：
- 本示例**故意不调用真实 LLM / Tool**，只用 TraceLogger 手动打点模拟几次运行：
  1) 一次成功运行（2 次 LLM + 1 次搜索工具）
  2) 一次失败运行（工具超时）
  3) 一次成功运行（3 次 LLM，未调工具）
- 运行完用 TraceEvaluator.from_path() 加载刚刚落盘的 SQLite 文件，
  聚合成本/效率/质量三类指标并打印人类可读报告，最后导出 eval_report.json。
- 评估完全是**离线行为**：关掉 tracer 之后再评估，评估过程不影响业务主流程；
  不传 evaluator 时，业务代码零开销。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.trace import (
    SpanKind,
    SpanStatus,
    TraceEvaluator,
    TraceLogger,
)


def simulate_run(tracer: TraceLogger, run_name: str, *, llm_calls: int, tools: list[tuple[str, float, bool]],
                 success: bool, model: str, base_tokens: int) -> None:
    """模拟一次 Agent run：root agent span + decision + N 次 llm + M 次 tool。

    为了让报告里的耗时真实可见，这里对 llm/tool 用 time.sleep 模拟各自耗时
    （真实业务里这些时间是 LLM 网络 IO 与工具执行时间）。
    """
    import time
    with tracer.trace(run_name) as tid:
        # root: agent.run
        with tracer.span("agent.run", SpanKind.AGENT) as agent_span:
            # 一个 decision span
            with tracer.span("decision.plan", SpanKind.DECISION) as dec:
                time.sleep(0.005)
                dec.finish(output={"next_action": "call_llm"})

            # N 次 LLM 调用（每次 sleep 80ms 模拟网络 + 推理）
            for i in range(llm_calls):
                pt = int(base_tokens * 0.6)
                ct = int(base_tokens * 0.4)
                with tracer.span(f"llm.invoke.{i}", SpanKind.LLM, metadata={
                    "model": model,
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                }) as llm_span:
                    time.sleep(0.08)
                    llm_span.finish(output="...llm response...")

            # M 次工具调用
            for tname, dur_ms, ok in tools:
                sleep_s = dur_ms / 1000.0
                if ok:
                    with tracer.span(f"tool.{tname}", SpanKind.TOOL,
                                     input={"query": f"{tname} query"},
                                     metadata={"tool_name": tname}) as ts:
                        time.sleep(sleep_s)
                        ts.finish(output=f"{tname} result")
                else:
                    with tracer.span(f"tool.{tname}", SpanKind.TOOL,
                                     input={"query": f"{tname} query"},
                                     metadata={"tool_name": tname}) as ts:
                        time.sleep(sleep_s)
                        ts.finish(output=None, status=SpanStatus.ERROR, error=f"{tname} TimeoutError")

            # 若整体失败，把 root 也标记为 error
            if not success:
                agent_span.finish(output=None, status=SpanStatus.ERROR, error="Agent failed due to tool error")


def main() -> None:
    # 用临时目录存放 trace 落盘文件（SQLite 格式）
    out_dir = os.path.join(tempfile.gettempdir(), "fast_agents_eval_demo")
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "runs.db")
    # 若旧文件存在先清掉（演示用，真实场景请不要清）
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

    print(f"trace 将落盘到：{db_path}")
    print("-" * 60)

    tracer = TraceLogger(path=db_path, session_id="demo_session")

    # ---- 模拟三次运行 ----
    # 运行 1：成功，2 次 LLM + 1 次 search 工具
    simulate_run(tracer, "run1_search_qa", llm_calls=2,
                 tools=[("web_search", 120.0, True)],
                 success=True, model="gpt-4o", base_tokens=150)
    # 运行 2：失败，1 次 LLM + 1 次 calc 工具（超时）+ search 正常
    simulate_run(tracer, "run2_math_fail", llm_calls=1,
                 tools=[("calculator", 30.0, False), ("web_search", 80.0, True)],
                 success=False, model="gpt-3.5-turbo", base_tokens=80)
    # 运行 3：成功，3 次 LLM，未调工具（纯 LLM 回答）
    simulate_run(tracer, "run3_chat_only", llm_calls=3,
                 tools=[],
                 success=True, model="gpt-4o", base_tokens=200)

    tracer.flush()
    tracer.close()
    print("三次模拟运行打点完成。\n")

    # ---- 离线评估 ----
    evaluator = TraceEvaluator.from_path(db_path)
    report = evaluator.evaluate()

    # 打印人类可读文本报告
    print(report.to_text())

    # 导出 JSON 报告
    json_path = os.path.join(out_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"\nJSON 报告已导出：{json_path}")

    # 导出 CSV 明细（每条 trace 一行，Excel / pandas 可直接打开做二次透视）
    csv_path = os.path.join(out_dir, "eval_report.csv")
    report.to_csv(csv_path)
    print(f"CSV 明细已导出：{csv_path}")


if __name__ == "__main__":
    main()
