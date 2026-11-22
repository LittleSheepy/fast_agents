"""LLM 重试/熔断（I4）快速示例

演示：
1. 默认配置开箱即用（max_retries=2 + 指数退避 + 断路器）
2. 可重试错误按指数退避重试到成功/耗尽
3. 不可重试错误（鉴权/参数错）直接透传不重试
4. 断路器：连续失败达阈值后 OPEN，快速拒绝后续请求
5. 完全关闭重试（retry=None / RetryConfig.disabled()）

本示例**不发起真实网络请求**：用可计数的假函数模拟各种异常，方便离线演示。
真实使用时，只需在 LLM 构造时传 `retry=RetryConfig(...)` 或默认即可。
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.llm import (
    RetryConfig,
    CircuitBreakerRegistry,
    run_with_retry,
    LLMRetryExhaustedError,
    LLMCircuitOpenError,
)
from fast_agents.llm.errors import (
    LLMConnectionError,
    LLMAuthenticationError,
)


def make_flaky_fn(fail_times: int, exc=LLMConnectionError("网络失败")):
    """返回一个会失败 fail_times 次后成功的函数，带 call_count 计数。"""
    state = {"n": 0}
    def fn():
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc
        return "ok"
    fn.call_count = lambda: state["n"]
    return fn


def demo_1_default_retry():
    """场景 1：前 2 次失败，第 3 次成功（默认 max_retries=2）。"""
    print("=" * 60)
    print("场景 1：默认重试（max_retries=2），前 2 次失败、第 3 次成功")
    fn = make_flaky_fn(fail_times=2)
    cfg = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1)
    result, attempts = run_with_retry(fn, retry=cfg, circuit_key=("demo", "1"))
    print(f"  结果={result!r}，实际调用次数={fn.call_count()}，重试次数={attempts}")


def demo_2_exhausted():
    """场景 2：所有重试都失败 → 抛 LLMRetryExhaustedError。"""
    print("=" * 60)
    print("场景 2：3 次尝试全部失败 → LLMRetryExhaustedError")
    fn = make_flaky_fn(fail_times=99)
    cfg = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.1)
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("demo", "2"))
    except LLMRetryExhaustedError as e:
        print(f"  按预期捕获：{type(e).__name__}: {e}")
        print(f"  总调用次数={fn.call_count()}（首次 + 2 次重试 = 3）")


def demo_3_non_retryable_no_retry():
    """场景 3：不可重试错误（如鉴权失败）直接透传，不浪费重试次数。"""
    print("=" * 60)
    print("场景 3：鉴权失败（retryable=False）直接透传，不重试")
    fn = make_flaky_fn(fail_times=99, exc=LLMAuthenticationError("API Key 无效"))
    cfg = RetryConfig(max_retries=5, base_delay=0.01)
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("demo", "3"))
    except LLMAuthenticationError as e:
        print(f"  按预期捕获：{type(e).__name__}: {e}")
        print(f"  总调用次数={fn.call_count()}（只调 1 次，跳过 5 次重试）")


def demo_4_circuit_breaker():
    """场景 4：连续失败触发断路器，后续请求快速拒绝。"""
    print("=" * 60)
    print("场景 4：断路器（failure_threshold=3，连续 3 次失败 → OPEN）")
    breaker = CircuitBreakerRegistry(failure_threshold=3, recovery_timeout=2.0)
    cfg = RetryConfig(max_retries=0, base_delay=0.01, breaker=breaker)
    fn = make_flaky_fn(fail_times=99)
    for i in range(3):
        try:
            run_with_retry(fn, retry=cfg, circuit_key=("demo", "4"))
        except LLMConnectionError:
            print(f"  第 {i+1} 次：LLMConnectionError（断路器仍 CLOSED，正常透传）")
    # 第 4 次：断路器 OPEN，直接快速拒绝
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("demo", "4"))
    except LLMCircuitOpenError as e:
        print(f"  第 4 次：{type(e).__name__}（断路器 OPEN，不真正发请求）")
    status = breaker.status(("demo", "4"))
    print(f"  断路器状态：state={status['state']}，"
          f"seconds_remaining={status['seconds_remaining']:.1f}s")


def demo_5_disable_retry():
    """场景 5：retry=None 完全关闭重试/熔断，零开销路径。"""
    print("=" * 60)
    print("场景 5：retry=None 完全关闭重试/熔断")
    fn = make_flaky_fn(fail_times=99)
    try:
        run_with_retry(fn, retry=None, circuit_key=("demo", "5"))
    except LLMConnectionError as e:
        print(f"  直接抛原始异常（无重试包装、无断路器）：{type(e).__name__}")
        print(f"  总调用次数={fn.call_count()}（只调 1 次）")


def demo_6_real_llm_usage():
    """场景 6：真实 LLM 用法示例（默认开启；演示如何调参/关闭）。"""
    print("=" * 60)
    print("场景 6：真实 LLM 用法（代码片段，不实际发起请求）")
    print(
        "  from fast_agents.llm import LLM, RetryConfig\n"
        "\n"
        "  # 1) 默认：max_retries=2、Full Jitter 指数退避、全局断路器\n"
        "  llm = LLM(model='qwen3-14b-instruct')\n"
        "\n"
        "  # 2) 调参：最多重试 3 次、基础退避 2s、熔断阈值 10 次、冷却 60s\n"
        "  llm = LLM(\n"
        "      model='qwen3-14b-instruct',\n"
        "      retry=RetryConfig(\n"
        "          max_retries=3,\n"
        "          base_delay=2.0,\n"
        "          max_delay=60.0,\n"
        "          breaker=CircuitBreakerRegistry(\n"
        "              failure_threshold=10,\n"
        "              recovery_timeout=60.0,\n"
        "          ),\n"
        "      ),\n"
        "  )\n"
        "\n"
        "  # 3) 完全关闭\n"
        "  llm = LLM(model='qwen3-14b-instruct', retry=None)\n"
        "  # 或 RetryConfig.disabled()\n"
        "  llm = LLM(model='qwen3-14b-instruct', retry=RetryConfig.disabled())"
    )


if __name__ == "__main__":
    demo_1_default_retry()
    demo_2_exhausted()
    demo_3_non_retryable_no_retry()
    demo_4_circuit_breaker()
    demo_5_disable_retry()
    demo_6_real_llm_usage()
    print("=" * 60)
    print("全部示例运行完成 ✅")
