"""trace 模块示例：SQLite 落盘后端（SQLiteStore）。

演示：
  1. 用 SQLiteStore 落 SQLite 文件（自动建表建索引、WAL 并发写友好）；
  2. 两种接入方式：显式 ``store=SQLiteStore(path)``、或 ``TraceLogger(store="xxx.db")`` 按后缀一行切换；
  3. 按 trace_id / session_id / kind 谓词下推过滤（比 jsonl 逐行扫描快）；
  4. iter_records() 流式遍历、replay() 回放整条链路；
  5. 关掉再打开同一文件，历史 span 仍在（跨进程回放）；
  6. 可选打印建表后的表结构（.schema trace_spans 等价信息）。

运行方式：在项目根目录执行
    python examples/trace/sqlite_usage.py

无需任何 API Key，完全离线可跑（用本地 add / multiply 工具演示）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python examples/trace/xxx.py 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.trace import SpanKind, SQLiteStore, TraceLogger  # noqa: E402
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
    # ---------- 1. 指定 SQLite 文件路径（父目录自动创建） ----------
    db_dir = Path("examples") / "trace" / "_output"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "demo.db")
    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        Path(p).unlink(missing_ok=True)  # 清理上次产物

    # 方式一：显式注入 store（推荐，可复用同一 store 给多个 tracer）
    store = SQLiteStore(db_path)
    tracer = TraceLogger(store=store)
    print(f"[1] SQLite 落盘路径: {db_path}")

    # ---------- 2. 接入工具注册表（一行搞定） ----------
    reg = ToolRegistry()
    reg.register_tool(add)
    reg.register_tool(multiply)
    tracer.attach_to_tool_registry(reg)
    print(f"[2] 已注册工具: {[t.name for t in reg.list_tools()]}")

    # ---------- 3. 模拟一次完整 run ----------
    with tracer.trace("math-pipeline", session_id="demo-session"):
        with tracer.span("agent.run", SpanKind.AGENT, input={"query": "(2+3)*4 = ?"}) as agent_span:
            with tracer.span("decision.step1", SpanKind.DECISION,
                             input={"thought": "先算括号里的 2+3"}) as dec:
                dec.output = {"action": "call add", "args": {"a": 2, "b": 3}}
            r1 = reg.invoke_tool("add", {"a": 2, "b": 3})
            mid = int(str(r1.output))

            with tracer.span("decision.step2", SpanKind.DECISION,
                             input={"thought": f"上一步结果 {mid}，再乘以 4"}) as dec:
                dec.output = {"action": "call multiply", "args": {"a": mid, "b": 4}}
            r2 = reg.invoke_tool("multiply", {"a": mid, "b": 4})
            final = int(str(r2.output))

            agent_span.output = {"final_answer": f"(2+3)*4 = {final}"}
    print(f"[3] 计算结果: (2+3)*4 = {final}")

    # ---------- 4. 谓词下推过滤（走索引，jsonl 只能逐行扫） ----------
    print(f"[4] 总条数 count(): {store.count()}")
    print(f"    kind=agent:        {len(tracer.read_spans(kind='agent'))} 条")
    print(f"    kind=decision:     {len(tracer.read_spans(kind='decision'))} 条")
    print(f"    kind=tool:         {len(tracer.read_spans(kind='tool'))} 条")
    print(f"    session=demo-session: {len(tracer.read_spans(session_id='demo-session'))} 条")

    # ---------- 5. 流式遍历 + 回放 ----------
    print("\n[5] iter_records() 流式遍历：")
    for r in store.iter_records():
        ms = f"{r.duration_ms:.3f}ms" if r.duration_ms is not None else "-"
        print(f"    - [{r.kind.value:>8}] {r.name:<20} {ms:>10}")

    first_trace_id = tracer.read_spans()[0].trace_id
    print(f"\n[6] replay trace={first_trace_id[:16]}...")
    for entry in tracer.replay(trace_id=first_trace_id):
        indent = "  " if entry["parent_id"] else ""
        ms = f"{entry['duration_ms']:.3f}ms" if entry["duration_ms"] is not None else "-"
        out_preview = str(entry["output"])[:60]
        print(f"{indent}- [{entry['kind']:>8}] {entry['name']:<20} {ms:>10}  -> {out_preview}")

    # ---------- 6. flush & close（写入走单语句自动提交，flush 为 no-op） ----------
    tracer.flush()
    tracer.close()
    print(f"\n[7] 已关闭，数据库文件大小 {Path(db_path).stat().st_size} 字节")

    # ---------- 7. 重新打开做「跨进程回放」 ----------
    store2 = SQLiteStore(db_path)
    print(f"[8] 重新打开后读到 {store2.count()} 条历史 Span")
    store2.close()

    # ---------- 8. 方式二：按后缀自动识别（.db → SQLiteStore）一行切换 ----------
    t2 = TraceLogger(store=str(db_dir / "demo_auto.db"))
    print(f"[9] TraceLogger(store='...demo_auto.db') 的 store 类型: {type(t2.store).__name__}")
    t2.close()

    print("\n[10] 你可以直接用 SQLite 客户端查看原始表：")
    print(f"     sqlite3 {db_path} 'SELECT kind, name, duration_ms FROM trace_spans;'")


if __name__ == "__main__":
    main()
