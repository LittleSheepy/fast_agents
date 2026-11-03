"""trace 模块示例：JSONL 落盘 + 工具钩子接入 + 回放。

演示：
  1. 指定 jsonl 文件路径落盘（路径自动创建）；
  2. attach_to_tool_registry() 一行接入工具打点，优先走 ToolRegistry.add_hook；
  3. 程序结束后关闭 tracer，再次打开同一文件能读到历史 span（跨进程/跨会话回放）；
  4. 按 trace_id / kind 过滤查询；
  5. 输出文件每行一个 JSON，可用 jq / 文本编辑器直接查看。

运行方式：在项目根目录执行
    python examples/trace/jsonl_replay.py

无需任何 API Key，完全离线可跑（用本地 add 工具演示）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python examples/trace/xxx.py 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.trace import SpanKind, TraceLogger  # noqa: E402
from fast_agents.tools import StructuredTool, ToolRegistry  # noqa: E402


@StructuredTool.from_function
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


@StructuredTool.from_function
def multiply(a: int, b: int) -> int:
    """两个整数相乘。"""
    return a * b


def main() -> None:
    # ---------- 1. 创建落盘 tracer ----------
    trace_dir = Path("examples") / "trace" / "_output"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(trace_dir / "demo.jsonl")

    # 先清理上次产物，避免重复
    if Path(trace_path).exists():
        Path(trace_path).unlink()

    tracer = TraceLogger(path=trace_path)
    print(f"[1] trace 落盘路径: {trace_path}")

    # ---------- 2. 接入工具注册表（一行搞定） ----------
    reg = ToolRegistry()
    reg.register_tool(add)
    reg.register_tool(multiply)
    tracer.attach_to_tool_registry(reg)
    print(f"[2] 已注册工具: {[t.name for t in reg.list_tools()]}")

    # ---------- 3. 模拟一次完整 run ----------
    with tracer.trace("math-pipeline", session_id="demo-session"):
        with tracer.span("agent.run", SpanKind.AGENT, input={"query": "(2+3)*4 = ?"}) as agent_span:
            # decision 1：先算 2+3
            with tracer.span("decision.step1", SpanKind.DECISION,
                             input={"thought": "先算括号里的 2+3"}) as dec:
                dec.output = {"action": "call add", "args": {"a": 2, "b": 3}}
            r1 = reg.invoke_tool("add", {"a": 2, "b": 3})
            mid = int(str(r1.output))

            # decision 2：再 *4
            with tracer.span("decision.step2", SpanKind.DECISION,
                             input={"thought": f"上一步结果 {mid}，再乘以 4"}) as dec:
                dec.output = {"action": "call multiply", "args": {"a": mid, "b": 4}}
            r2 = reg.invoke_tool("multiply", {"a": mid, "b": 4})
            final = int(str(r2.output))

            agent_span.output = {"final_answer": f"(2+3)*4 = {final}"}
    print(f"[3] 计算结果: (2+3)*4 = {final}")

    # ---------- 4. flush & close，确保写入磁盘 ----------
    tracer.flush()
    tracer.close()
    file_size = Path(trace_path).stat().st_size
    print(f"[4] 文件已落盘，大小 {file_size} 字节")

    # ---------- 5. 重新打开同一文件做「回放」（模拟新进程） ----------
    tracer2 = TraceLogger(path=trace_path)  # 默认 append=True，会把历史读入内存
    all_spans = tracer2.read_spans()
    print(f"[5] 重新打开后读到 {len(all_spans)} 条历史 Span")

    # 按 session_id / kind 过滤
    session_spans = tracer2.read_spans(session_id="demo-session")
    tool_spans = tracer2.read_spans(kind="tool")
    agent_spans = tracer2.read_spans(kind="agent")
    decision_spans = tracer2.read_spans(kind="decision")
    print(f"    session=demo-session: {len(session_spans)} 条")
    print(f"    kind=agent:    {len(agent_spans)} 条")
    print(f"    kind=decision: {len(decision_spans)} 条")
    print(f"    kind=tool:     {len(tool_spans)} 条")

    # 回放第一条 trace
    first_trace_id = all_spans[0].trace_id
    print(f"\n=== 回放 trace={first_trace_id[:16]}... ===")
    for entry in tracer2.replay(trace_id=first_trace_id):
        indent = "  " if entry["parent_id"] else ""
        ms = f"{entry['duration_ms']:.3f}ms" if entry["duration_ms"] is not None else "-"
        out_preview = str(entry["output"])[:60]
        print(f"{indent}- [{entry['kind']:>8}] {entry['name']:<20} {ms:>10}  -> {out_preview}")

    tracer2.close()
    print("\n[6] 你可以直接打开文件查看原始 jsonl：")
    print(f"    {trace_path}")


if __name__ == "__main__":
    main()
