"""jsonl_demo.py —— JsonlStore 文件落盘模式演示。

JsonlStore 只做「一行一条 JSON 直接追加落盘」，不在内存保留全量镜像；
本示例写入大量 span，对比「磁盘文件大小」与「进程峰值 RSS」，
说明长时间运行不会随写入量线性涨内存。

说明：JsonlStore 已不再支持内存模式，内存场景请使用 InMemoryTraceStore
（即 ``TraceLogger()`` / ``TraceLogger(store=":memory:")`` 的默认后端）。

运行方式（项目根目录）:
    python examples/trace/jsonl_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.trace import JsonlStore, SpanKind, SpanRecord  # noqa: E402

try:
    import resource

    def rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:  # Windows 无 resource 模块
    def rss_mb() -> float:
        return -1.0


def main() -> None:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "trace.jsonl")

    store = JsonlStore(path)                       # 真·文件落盘模式
    print("[1] store._fp is None:", store._fp is None)     # False
    # JsonlStore 已不再支持内存模式：实例上不存在 _mem 镜像属性
    print("[2] 实例是否含内存镜像 _mem 属性:", hasattr(store, "_mem"))     # False

    payload = "x" * 20000                          # 每条 span 约 40KB
    N = 2000
    for i in range(N):
        store.write(SpanRecord(
            trace_id="trace-1", span_id=f"span-{i:04d}",
            name="llm.invoke", kind=SpanKind.LLM,
            input={"prompt": payload}, output={"text": payload},
        ))

    print("[3] 写入 %d 条后实例是否含 _mem 镜像: %s" % (N, hasattr(store, "_mem")))
    print("[4] 磁盘文件大小:                %.2f MB" % (os.path.getsize(path) / 1024 / 1024))
    print("[5] 进程峰值 RSS:                 %.1f MB" % rss_mb())
    store.close()

    store2 = JsonlStore(path)                      # 模拟"进程重启/新开 store 指向同一文件"
    print("[6] 新开 store 后是否含 _mem 镜像: %s" % hasattr(store2, "_mem"))
    print("[7] 第二次进程峰值 RSS:           %.1f MB" % rss_mb())
    print("[8] 流式读回记录条数:              %d" % sum(1 for _ in store2.iter_records()))
    store2.close()


if __name__ == "__main__":
    main()
