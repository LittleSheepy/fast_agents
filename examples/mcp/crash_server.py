"""崩溃+大量 stderr 输出的测试服务器。

用于验证：
1. 子进程大量 stderr 输出不会阻塞/卡死子进程（修复前 stderr 缓冲区满会挂）
2. 子进程异常退出时，客户端能立即拿到带 stderr 尾部的明确错误
"""

from __future__ import annotations

import sys
import time


def main():
    # 输出大量 stderr：测试 stderr 读线程能及时消费，不会因 OS 管道缓冲区满卡死
    for i in range(5000):
        print(f"[stderr] line {i}: " + "x" * 100, file=sys.stderr, flush=True)
    sys.stderr.flush()
    time.sleep(0.2)
    print("FATAL: simulated crash", file=sys.stderr, flush=True)
    sys.stderr.flush()
    # 非零退出码触发 MCPConnectionError
    sys.exit(2)


if __name__ == "__main__":
    main()
