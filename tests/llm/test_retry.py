"""LLM I4：指数退避重试 + 断路器单元测试

全程离线：用假函数模拟可重试/不可重试异常，不发起任何网络请求。

覆盖项（15 项）：
1. 首次成功不重试
2. 可重试错误按指数退避重试至成功
3. 可重试错误超过 max_retries 抛 LLMRetryExhaustedError
4. 不可重试错误不重试、直接透传
5. retry=None 时零开销路径、完全不重试
6. RetryConfig.disabled() 关闭重试
7. RetryConfig 参数校验（负数、大小倒置）
8. backoff Full Jitter 在 [0, cap] 范围内
9. 断路器：连续失败达阈值后进入 OPEN，快速拒绝
10. 断路器：recovery_timeout 后 HALF_OPEN 放行探测，成功回 CLOSED
11. 断路器：HALF_OPEN 探测失败重新回 OPEN
12. 断路器：按 (base_url, model) 维度隔离，互不影响
13. 断路器：reset 可重置状态
14. run_stream_with_retry：首块成功后正常迭代
15. run_stream_with_retry：首块失败时可重试；迭代中途失败不重试
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.llm.retry import (
    RetryConfig,
    CircuitBreakerRegistry,
    run_with_retry,
    run_stream_with_retry,
    _default_breaker,
)
from fast_agents.llm.errors import (
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConfigError,
)
from fast_agents.llm.retry_errors import LLMRetryExhaustedError, LLMCircuitOpenError


# ==================== 1. 首次成功不重试 ====================

def test_1_first_try_success_no_retry():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    cfg = RetryConfig(max_retries=2, base_delay=0.001)
    result, attempts = run_with_retry(fn, retry=cfg, circuit_key=("t", "t1"))
    assert result == "ok"
    assert attempts == 0
    assert len(calls) == 1


# ==================== 2. 可重试错误重试至成功 ====================

def test_2_retry_then_success():
    state = {"n": 0}
    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise LLMConnectionError(f"fail {state['n']}")
        return "ok"
    cfg = RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.01)
    result, attempts = run_with_retry(fn, retry=cfg, circuit_key=("t", "t2"))
    assert result == "ok"
    assert attempts == 2  # 第3次才成功（前2次失败重试）
    assert state["n"] == 3


# ==================== 3. 重试耗尽抛 LLMRetryExhaustedError ====================

def test_3_exhausted():
    state = {"n": 0}
    def fn():
        state["n"] += 1
        raise LLMTimeoutError("always timeout")
    cfg = RetryConfig(max_retries=2, base_delay=0.001, max_delay=0.01)
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("t", "t3"))
        assert False, "应当抛异常"
    except LLMRetryExhaustedError as e:
        assert state["n"] == 3  # 首次 + 2 次重试 = 3
        assert isinstance(e.original_error, LLMTimeoutError)


# ==================== 4. 不可重试错误不重试 ====================

def test_4_non_retryable_no_retry():
    state = {"n": 0}
    def fn():
        state["n"] += 1
        raise LLMAuthenticationError("bad key")
    cfg = RetryConfig(max_retries=5, base_delay=0.001)
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("t", "t4"))
        assert False
    except LLMAuthenticationError:
        assert state["n"] == 1  # 只调用一次，不重试


# ==================== 5. retry=None 零开销路径 ====================

def test_5_retry_none_no_retry():
    state = {"n": 0}
    def fn():
        state["n"] += 1
        raise LLMConnectionError("fail")
    # retry=None 应直接抛原异常，不走 LLMRetryExhaustedError 包装
    try:
        run_with_retry(fn, retry=None, circuit_key=("t", "t5"))
        assert False
    except LLMConnectionError:
        assert state["n"] == 1


# ==================== 6. RetryConfig.disabled() ====================

def test_6_disabled_config():
    cfg = RetryConfig.disabled()
    assert cfg.max_retries == 0
    state = {"n": 0}
    def fn():
        state["n"] += 1
        raise LLMConnectionError("fail")
    # disabled 的 breaker 阈值极高，不会熔断；max_retries=0 不重试
    try:
        run_with_retry(fn, retry=cfg, circuit_key=("t", "t6"))
        assert False
    except LLMConnectionError:
        assert state["n"] == 1


# ==================== 7. RetryConfig 参数校验 ====================

def test_7_config_validation():
    try:
        RetryConfig(max_retries=-1)
        assert False
    except ValueError:
        pass
    try:
        RetryConfig(base_delay=-1)
        assert False
    except ValueError:
        pass
    try:
        RetryConfig(base_delay=10, max_delay=5)
        assert False
    except ValueError:
        pass


# ==================== 8. Full Jitter backoff 在 [0, cap] 范围 ====================

def test_8_backoff_full_jitter():
    cfg = RetryConfig(base_delay=1.0, max_delay=30.0)
    for _ in range(50):
        d = cfg.backoff(0)
        assert 0 <= d <= 1.0, d
        d = cfg.backoff(3)
        assert 0 <= d <= 8.0, d  # 1*2^3=8
        d = cfg.backoff(10)
        assert 0 <= d <= 30.0, d  # cap 30


# ==================== 9. 断路器 OPEN 快速拒绝 ====================

def test_9_circuit_breaker_opens():
    breaker = CircuitBreakerRegistry(failure_threshold=3, recovery_timeout=60.0)
    key = ("http://test", "m1")

    cfg = RetryConfig(max_retries=0, base_delay=0.001, breaker=breaker)

    def fail():
        raise LLMConnectionError("fail")

    # 前 3 次：CLOSED 状态下正常收到 LLMConnectionError
    for _ in range(3):
        try:
            run_with_retry(fail, retry=cfg, circuit_key=key)
            assert False
        except LLMConnectionError:
            pass

    # 达到阈值后断路器切换到 OPEN，第 4 次应被快速拒绝为 LLMCircuitOpenError
    try:
        run_with_retry(fail, retry=cfg, circuit_key=key)
        assert False, "应当抛 LLMCircuitOpenError"
    except LLMCircuitOpenError:
        pass


# ==================== 10. 断路器恢复（HALF_OPEN 成功 → CLOSED） ====================

def test_10_circuit_recovers_after_timeout():
    breaker = CircuitBreakerRegistry(failure_threshold=2, recovery_timeout=0.05)
    key = ("http://test", "m2")
    cfg = RetryConfig(max_retries=0, breaker=breaker)

    counter = {"n": 0}
    def fn():
        counter["n"] += 1
        if counter["n"] <= 2:
            raise LLMConnectionError("fail")
        return "ok"

    # 前 2 次失败 → OPEN
    for _ in range(2):
        try:
            run_with_retry(fn, retry=cfg, circuit_key=key)
            assert False
        except LLMConnectionError:
            pass
    assert breaker.status(key)["state"] == "open"

    time.sleep(0.1)  # 等过 recovery_timeout，下一次进入 HALF_OPEN 放行探测

    # HALF_OPEN 探测成功 → CLOSED
    result, attempts = run_with_retry(fn, retry=cfg, circuit_key=key)
    assert result == "ok"
    status = breaker.status(key)
    assert status["state"] == "closed"


# ==================== 11. HALF_OPEN 探测失败重新回 OPEN ====================

def test_11_half_open_probe_fails_reopens():
    breaker = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout=0.02)
    key = ("http://test", "m3")
    cfg = RetryConfig(max_retries=0, breaker=breaker)

    def fail():
        raise LLMConnectionError("fail")

    try:
        run_with_retry(fail, retry=cfg, circuit_key=key)
    except LLMConnectionError:
        pass
    assert breaker.status(key)["state"] == "open"

    time.sleep(0.05)
    # 探测请求失败 → 重新 OPEN
    try:
        run_with_retry(fail, retry=cfg, circuit_key=key)
    except LLMConnectionError:
        pass
    assert breaker.status(key)["state"] == "open"


# ==================== 12. 按 (base_url, model) 维度隔离 ====================

def test_12_circuit_key_isolation():
    breaker = CircuitBreakerRegistry(failure_threshold=2, recovery_timeout=60.0)
    cfg = RetryConfig(max_retries=0, breaker=breaker)
    k1 = ("http://a", "m1")
    k2 = ("http://a", "m2")

    def fail():
        raise LLMConnectionError("fail")
    def ok():
        return "ok"

    # k1 连续失败 2 次 → OPEN
    for _ in range(2):
        try:
            run_with_retry(fail, retry=cfg, circuit_key=k1)
            assert False
        except LLMConnectionError:
            pass
    assert breaker.status(k1)["state"] == "open"

    # k1 已 OPEN → 下一次快速拒绝
    try:
        run_with_retry(fail, retry=cfg, circuit_key=k1)
        assert False
    except LLMCircuitOpenError:
        pass

    # k2 不受影响，仍然可用
    assert breaker.status(k2)["state"] == "closed"
    r, _ = run_with_retry(ok, retry=cfg, circuit_key=k2)
    assert r == "ok"


# ==================== 13. reset 重置断路器 ====================

def test_13_circuit_reset():
    breaker = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout=60.0)
    cfg = RetryConfig(max_retries=0, breaker=breaker)
    key = ("http://test", "m4")
    def fail():
        raise LLMConnectionError("fail")
    try:
        run_with_retry(fail, retry=cfg, circuit_key=key)
    except LLMConnectionError:
        pass
    assert breaker.status(key)["state"] == "open"
    breaker.reset(key)
    assert breaker.status(key)["state"] == "closed"


# ==================== 14. run_stream_with_retry 首块成功后正常迭代 ====================

def test_14_stream_success():
    def gen():
        yield "a"
        yield "b"
        yield "c"
    cfg = RetryConfig(max_retries=2, base_delay=0.001)
    it, attempts = run_stream_with_retry(gen, retry=cfg, circuit_key=("s", "s1"))
    assert attempts == 0
    assert list(it) == ["a", "b", "c"]


# ==================== 15. 流式：首块失败可重试；中途失败不重试 ====================

def test_15_stream_retry_on_first_chunk():
    state = {"n": 0}
    def make_gen():
        state["n"] += 1
        if state["n"] < 3:
            # 创建生成器本身不抛，next() 首块时抛
            def bad():
                raise LLMConnectionError("first chunk fail")
                yield  # 让它成为生成器
            return bad()
        def good():
            yield "a"
            yield "b"
        return good()

    cfg = RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.01)
    it, attempts = run_stream_with_retry(make_gen, retry=cfg, circuit_key=("s", "s2"))
    assert list(it) == ["a", "b"]
    assert attempts == 2  # 重试 2 次后成功
    assert state["n"] == 3


def test_15b_stream_midstream_failure_no_retry():
    """迭代到中途抛错：不再重试（直接冒泡），断路器记一次失败。"""
    state = {"n": 0}
    def gen():
        yield "a"
        state["n"] += 1
        raise LLMConnectionError("mid fail")

    breaker = CircuitBreakerRegistry(failure_threshold=5, recovery_timeout=60.0)
    cfg = RetryConfig(max_retries=3, base_delay=0.001, breaker=breaker)
    it, attempts = run_stream_with_retry(gen, retry=cfg, circuit_key=("s", "s3"))
    assert attempts == 0
    first = next(it)
    assert first == "a"
    try:
        next(it)
        assert False
    except LLMConnectionError:
        pass
    assert state["n"] == 1  # gen 只被调用一次，没有重试


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    _default_breaker.reset()  # 进入全量测试前清空全局默认断路器，保证同进程多次运行可重复
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 手动运行前同样清空全局默认断路器，保证与全量运行互不污染
    _default_breaker.reset()

    # 人工添加测针对测试：AI不要删除覆盖

    # AI新添加的测试(AI可以删除覆盖)
    test_1_first_try_success_no_retry()  # AI新加的测试
    test_2_retry_then_success()  # AI新加的测试
    test_3_exhausted()  # AI新加的测试
    test_4_non_retryable_no_retry()  # AI新加的测试
    test_5_retry_none_no_retry()  # AI新加的测试
    test_6_disabled_config()  # AI新加的测试
    test_7_config_validation()  # AI新加的测试
    test_8_backoff_full_jitter()  # AI新加的测试
    test_9_circuit_breaker_opens()  # AI新加的测试
    test_10_circuit_recovers_after_timeout()  # AI新加的测试
    test_11_half_open_probe_fails_reopens()  # AI新加的测试
    test_12_circuit_key_isolation()  # AI新加的测试
    test_13_circuit_reset()  # AI新加的测试
    test_14_stream_success()  # AI新加的测试
    test_15_stream_retry_on_first_chunk()  # AI新加的测试
    test_15b_stream_midstream_failure_no_retry()  # AI新加的测试


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
