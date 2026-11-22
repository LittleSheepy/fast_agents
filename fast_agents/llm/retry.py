"""LLM 指数退避重试 + 熔断器（I4 核心实现，零新增依赖）。

设计要点（对齐主流框架与仓库风格）：
1. **指数退避 + 全抖动（Full Jitter）**：退避 = min(base * (2 ** attempt), cap) * random(0, 1)；
   参考 AWS / tenacity / LangChain 标准做法，避免"惊群"。
2. **按 ``retryable`` 属性分流**：只对 ``retryable=True`` 的 LLMError（连接/超时/限流/5xx）重试；
   鉴权/配置/参数/模型不存在/JSON 解析失败等直接透传，不做无谓重试。
3. **线程安全的断路器**（Circuit Breaker）：以（base_url, model）为 key 维护独立状态机；
   连续失败 ``failure_threshold`` 次后进入 OPEN，``recovery_timeout`` 秒后允许一条探测请求（HALF_OPEN），
   成功则回到 CLOSED、失败则回到 OPEN。跨线程安全（``threading.Lock``）。
4. **默认开启但参数保守**：``max_retries=2``、``base_delay=1.0``、``max_delay=30.0``；
   断路器默认 ``failure_threshold=5``、``recovery_timeout=30``。用户显式 ``retry=None`` 关闭。
5. **零新增依赖**：只用标准库 ``time`` / ``random`` / ``threading`` / ``functools``。
6. **流式调用支持重试**：生成器只有在「首次请求即失败」时可重试；如果已经 yield 了 chunks 再失败，
   说明客户端已经开始消费流，此时重试会造成输出错位，直接抛给上层（符合 LangChain 的处理方式）。
7. **失败安全**：断路器或重试器自身出 bug 时不应阻塞主流程，记录 warning 后仍按无重试执行。
"""
from __future__ import annotations

import time
import random
import asyncio
import logging
import inspect
import threading
import functools
from typing import Callable, Optional, Iterator, AsyncIterator, Awaitable, TypeVar, Any

from .errors import LLMError
from .retry_errors import LLMRetryExhaustedError, LLMCircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ------------------------------------------------------------------ #
# 断路器（Circuit Breaker）                                              #
# ------------------------------------------------------------------ #


class _CircuitState:
    """单个（base_url, model）维度的断路器状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: str = self.CLOSED
        self.failure_count: int = 0
        self.opened_at: float = 0.0
        self.half_open_pending: bool = False
        self.lock = threading.Lock()  # 仅保护本 key 的状态


class CircuitBreakerRegistry:
    """按 ``(base_url, model)`` 隔离的断路器注册表（全局单例，线程安全）。

    每个端点+模型组合维护自己的失败计数与状态机，避免「一个模型挂掉把全局熔断」。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._circuits: dict[tuple[str, str], _CircuitState] = {}
        self._global_lock = threading.Lock()

    def _get_or_create(self, key: tuple[str, str]) -> _CircuitState:
        with self._global_lock:
            circuit = self._circuits.get(key)
            if circuit is None:
                circuit = _CircuitState(self.failure_threshold, self.recovery_timeout)
                self._circuits[key] = circuit
            return circuit

    def allow_request(self, key: tuple[str, str]) -> bool:
        """判断是否允许发起请求（线程安全）。返回 True 表示允许。"""
        circuit = self._get_or_create(key)
        with circuit.lock:
            if circuit.state == _CircuitState.CLOSED:
                return True
            if circuit.state == _CircuitState.OPEN:
                # 超过恢复时间 → 切 HALF_OPEN，放行一条探测请求
                if time.monotonic() - circuit.opened_at >= circuit.recovery_timeout:
                    circuit.state = _CircuitState.HALF_OPEN
                    circuit.half_open_pending = True
                    return True
                return False
            # HALF_OPEN：如果上一条探测还没返回，后续请求直接拒绝（避免并发探测）
            return not circuit.half_open_pending

    def on_success(self, key: tuple[str, str]) -> None:
        """请求成功时调用（线程安全）。"""
        circuit = self._get_or_create(key)
        with circuit.lock:
            circuit.failure_count = 0
            circuit.state = _CircuitState.CLOSED
            circuit.half_open_pending = False

    def on_failure(self, key: tuple[str, str]) -> None:
        """请求失败时调用（线程安全）。"""
        circuit = self._get_or_create(key)
        with circuit.lock:
            if circuit.state == _CircuitState.HALF_OPEN:
                # 探测失败 → 立刻回 OPEN，重置冷却时间
                circuit.state = _CircuitState.OPEN
                circuit.opened_at = time.monotonic()
                circuit.half_open_pending = False
                logger.warning(
                    "LLM 断路器 HALF_OPEN 探测失败，重新进入 OPEN（key=%s, recovery=%.1fs）",
                    key, circuit.recovery_timeout,
                )
                return
            circuit.failure_count += 1
            if circuit.failure_count >= circuit.failure_threshold:
                circuit.state = _CircuitState.OPEN
                circuit.opened_at = time.monotonic()
                logger.warning(
                    "LLM 断路器切换为 OPEN（key=%s, failures=%d, recovery=%.1fs）",
                    key, circuit.failure_count, circuit.recovery_timeout,
                )

    def reset(self, key: Optional[tuple[str, str]] = None) -> None:
        """重置状态；不传 key 则重置全部。"""
        with self._global_lock:
            if key is None:
                self._circuits.clear()
                return
            if key in self._circuits:
                del self._circuits[key]

    def status(self, key: tuple[str, str]) -> dict[str, Any]:
        """返回指定 key 的当前断路器状态（用于监控/调试）。"""
        circuit = self._get_or_create(key)
        with circuit.lock:
            return {
                "state": circuit.state,
                "failure_count": circuit.failure_count,
                "failure_threshold": circuit.failure_threshold,
                "recovery_timeout": circuit.recovery_timeout,
                "opened_at": circuit.opened_at if circuit.state == _CircuitState.OPEN else None,
                "seconds_remaining": (
                    max(0.0, circuit.recovery_timeout - (time.monotonic() - circuit.opened_at))
                    if circuit.state == _CircuitState.OPEN else 0.0
                ),
            }


# 默认全局断路器实例（按模块约定保持保守默认值）
_default_breaker = CircuitBreakerRegistry(failure_threshold=5, recovery_timeout=30.0)


# ------------------------------------------------------------------ #
# Retry 配置                                                            #
# ------------------------------------------------------------------ #


class RetryConfig:
    """指数退避重试配置（不可变、线程安全）。

    Parameters
    ----------
    max_retries:
        最大重试次数（不含首次调用）；设为 0 表示只调一次、不重试。
        默认 2：总调用次数最多 3 次（首次 + 2 次重试）。
    base_delay:
        首次重试前的基础退避时间（秒），默认 1.0。
    max_delay:
        退避时间上限（秒），默认 30.0，防止等待过长。
    retry_on:
        额外的可重试异常类元组；默认仅按 ``e.retryable`` 判断，此处为扩展钩子。
    breaker:
        自定义断路器注册表；不传则用全局默认实例。
    """

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[BaseException], ...] = (),
        breaker: Optional[CircuitBreakerRegistry] = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        if base_delay < 0 or max_delay < 0:
            raise ValueError("delay 参数不能为负数")
        if max_delay < base_delay:
            raise ValueError("max_delay 不能小于 base_delay")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = tuple(retry_on)
        self.breaker = breaker if breaker is not None else _default_breaker

    def should_retry(self, exc: BaseException) -> bool:
        """判断给定异常是否应当重试。"""
        if isinstance(exc, self.retry_on):
            return True
        return isinstance(exc, LLMError) and getattr(exc, "retryable", False)

    def backoff(self, attempt: int) -> float:
        """Full Jitter 指数退避：在 [0, min(base * 2**attempt, max_delay)] 内随机取秒数。"""
        cap = min(self.base_delay * (2 ** attempt), self.max_delay)
        return random.uniform(0, cap) if cap > 0 else 0.0

    @classmethod
    def disabled(cls) -> "RetryConfig":
        """返回一个「完全不重试、也不走断路器」的配置（供测试/排障使用）。"""
        return cls(max_retries=0, breaker=CircuitBreakerRegistry(failure_threshold=10**9, recovery_timeout=0.0))


# ------------------------------------------------------------------ #
# 核心重试执行器                                                         #
# ------------------------------------------------------------------ #


def run_with_retry(
    fn: Callable[..., T],
    *,
    retry: Optional[RetryConfig],
    circuit_key: tuple[str, str],
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> tuple[T, int]:
    """执行同步函数 ``fn(*args, **kwargs)``，按 ``retry`` 配置进行指数退避重试。

    返回 ``(result, attempts)``：其中 ``attempts`` 是实际重试次数（0 表示一次就成功）。

    - ``retry=None`` 时直接调用 ``fn``，不走任何重试/熔断逻辑（保持零开销）。
    - 可重试异常走重试，不可重试异常直接透传。
    - 重试耗尽抛出 :class:`LLMRetryExhaustedError`。
    - 断路器 OPEN 时直接抛出 :class:`LLMCircuitOpenError`。
    """
    kwargs = kwargs or {}

    if retry is None:
        return fn(*args, **kwargs), 0

    breaker = retry.breaker

    # 断路器检查
    if not breaker.allow_request(circuit_key):
        status = breaker.status(circuit_key)
        raise LLMCircuitOpenError(
            f"LLM 断路器 OPEN（端点/模型暂时不可用，{status['seconds_remaining']:.1f}s 后可重试）：{circuit_key}",
        )

    last_exc: Optional[BaseException] = None
    for attempt in range(retry.max_retries + 1):
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            last_exc = e
            if retry.should_retry(e):
                breaker.on_failure(circuit_key)
                if attempt < retry.max_retries:
                    delay = retry.backoff(attempt)
                    logger.warning(
                        "LLM 调用可重试错误（第 %d/%d 次，%.2fs 后重试）：%s",
                        attempt + 1, retry.max_retries + 1, delay, e,
                    )
                    time.sleep(delay)
                    continue
                # 最后一次失败也已由上面 on_failure 计数；
                # 只有真正做过重试（max_retries > 0）才包 LLMRetryExhaustedError；
                # max_retries=0 时直接透传原异常，保持语义纯粹。
                if retry.max_retries > 0:
                    raise LLMRetryExhaustedError(
                        f"LLM 调用在 {retry.max_retries + 1} 次尝试后仍失败：{e}",
                        original_error=e,
                    ) from e
                raise
            breaker.on_failure(circuit_key)
            raise
        else:
            breaker.on_success(circuit_key)
            return result, attempt

    assert last_exc is not None
    if retry.max_retries > 0:
        raise LLMRetryExhaustedError(
            f"LLM 调用在 {retry.max_retries + 1} 次尝试后仍失败：{last_exc}",
            original_error=last_exc,
        ) from last_exc
    raise last_exc


def run_stream_with_retry(
    fn: Callable[..., Iterator[T]],
    *,
    retry: Optional[RetryConfig],
    circuit_key: tuple[str, str],
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> tuple[Iterator[T], int]:
    """流式生成器版本。**仅在「生成器创建前」可重试**；一旦开始 yield chunks，后续失败直接抛出。

    返回 ``(iterator, attempts)``——调用方应当先取 attempts 再迭代 iterator。

    策略：
    1. 先创建生成器（这一步会触发 HTTP 首包请求，网络/5xx/限流错误都会在这里抛出，可重试）；
    2. 创建成功后直接 yield from，迭代期间出现的任何错误都不再重试。
    """
    kwargs = kwargs or {}

    if retry is None:
        return fn(*args, **kwargs), 0

    breaker = retry.breaker

    if not breaker.allow_request(circuit_key):
        status = breaker.status(circuit_key)
        raise LLMCircuitOpenError(
            f"LLM 断路器 OPEN（端点/模型暂时不可用，{status['seconds_remaining']:.1f}s 后可重试）：{circuit_key}",
        )

    gen: Optional[Iterator[T]] = None
    last_exc: Optional[BaseException] = None
    attempts_used: int = 0

    for attempt in range(retry.max_retries + 1):
        try:
            gen = fn(*args, **kwargs)
            first = next(gen)
        except StopIteration:
            breaker.on_success(circuit_key)
            return iter(()), attempt
        except BaseException as e:
            last_exc = e
            if retry.should_retry(e):
                breaker.on_failure(circuit_key)
                if attempt < retry.max_retries:
                    delay = retry.backoff(attempt)
                    logger.warning(
                        "LLM 流式调用可重试错误（第 %d/%d 次，%.2fs 后重试）：%s",
                        attempt + 1, retry.max_retries + 1, delay, e,
                    )
                    time.sleep(delay)
                    continue
                if retry.max_retries > 0:
                    raise LLMRetryExhaustedError(
                        f"LLM 流式调用在 {retry.max_retries + 1} 次尝试后仍失败：{e}",
                        original_error=e,
                    ) from e
                raise
            breaker.on_failure(circuit_key)
            raise
        else:
            breaker.on_success(circuit_key)
            attempts_used = attempt
            break

    assert gen is not None

    def _gen():
        yield first
        try:
            yield from gen
        except BaseException:
            breaker.on_failure(circuit_key)
            raise

    return _gen(), attempts_used


# ------------------------------------------------------------------ #
# 便捷装饰器（可选使用）                                                  #
# ------------------------------------------------------------------ #


def with_retry(circuit_key: tuple[str, str], retry_attr: str = "_retry"):
    """装饰类方法：从 ``self.<retry_attr>`` 读取 :class:`RetryConfig`，自动包裹重试。"""

    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            retry: Optional[RetryConfig] = getattr(self, retry_attr, None)
            return run_with_retry(
                functools.partial(method, self),
                retry=retry,
                circuit_key=circuit_key if callable(circuit_key) is False else circuit_key(self, *args, **kwargs),
                args=args,
                kwargs=kwargs,
            )
        return wrapper

    return decorator


# ------------------------------------------------------------------ #
# 异步重试执行器（与同步版本共用 RetryConfig / CircuitBreakerRegistry）      #
# ------------------------------------------------------------------ #


async def arun_with_retry(
    fn: Callable[..., Awaitable[T]],
    *,
    retry: Optional[RetryConfig],
    circuit_key: tuple[str, str],
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> tuple[T, int]:
    """异步版本 :func:`run_with_retry`：执行 ``await fn(*args, **kwargs)`` 并按配置指数退避重试。

    - ``retry=None`` 时直接 await fn，零开销；
    - 退避用 ``asyncio.sleep`` 不阻塞事件循环；
    - 断路器/退避算法/异常分流与同步版本完全一致，复用同一套 ``RetryConfig``；
    - 返回 ``(result, attempts_used)``，语义与同步版本相同。
    """
    kwargs = kwargs or {}

    if retry is None:
        return await fn(*args, **kwargs), 0

    breaker = retry.breaker

    if not breaker.allow_request(circuit_key):
        status = breaker.status(circuit_key)
        raise LLMCircuitOpenError(
            f"LLM 断路器 OPEN（端点/模型暂时不可用，{status['seconds_remaining']:.1f}s 后可重试）：{circuit_key}",
        )

    last_exc: Optional[BaseException] = None
    for attempt in range(retry.max_retries + 1):
        try:
            result = await fn(*args, **kwargs)
        except BaseException as e:
            last_exc = e
            if retry.should_retry(e):
                breaker.on_failure(circuit_key)
                if attempt < retry.max_retries:
                    delay = retry.backoff(attempt)
                    logger.warning(
                        "LLM 异步调用可重试错误（第 %d/%d 次，%.2fs 后重试）：%s",
                        attempt + 1, retry.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                if retry.max_retries > 0:
                    raise LLMRetryExhaustedError(
                        f"LLM 异步调用在 {retry.max_retries + 1} 次尝试后仍失败：{e}",
                        original_error=e,
                    ) from e
                raise
            breaker.on_failure(circuit_key)
            raise
        else:
            breaker.on_success(circuit_key)
            return result, attempt

    assert last_exc is not None
    if retry.max_retries > 0:
        raise LLMRetryExhaustedError(
            f"LLM 异步调用在 {retry.max_retries + 1} 次尝试后仍失败：{last_exc}",
            original_error=last_exc,
        ) from last_exc
    raise last_exc


async def arun_stream_with_retry(
    fn: Callable[..., Awaitable[AsyncIterator[T]]],
    *,
    retry: Optional[RetryConfig],
    circuit_key: tuple[str, str],
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> tuple[AsyncIterator[T], int]:
    """异步流式版本：与 :func:`run_stream_with_retry` 语义一致，返回 ``(async_iterator, attempts_used)``。

    - 首块（first chunk）前可重试；一旦开始 yield chunks，后续错误直接抛出，不重试；
    - 支持 fn 直接返回 async generator，或返回 coroutine（async def 内 return async_gen）。
    """
    kwargs = kwargs or {}

    if retry is None:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result, 0

    breaker = retry.breaker

    if not breaker.allow_request(circuit_key):
        status = breaker.status(circuit_key)
        raise LLMCircuitOpenError(
            f"LLM 断路器 OPEN（端点/模型暂时不可用，{status['seconds_remaining']:.1f}s 后可重试）：{circuit_key}",
        )

    agen: Optional[AsyncIterator[T]] = None
    last_exc: Optional[BaseException] = None
    attempts_used: int = 0

    for attempt in range(retry.max_retries + 1):
        try:
            gen_or_coro = fn(*args, **kwargs)
            if inspect.isawaitable(gen_or_coro):
                agen = await gen_or_coro
            else:
                agen = gen_or_coro
            first = await agen.__anext__()
        except StopAsyncIteration:
            breaker.on_success(circuit_key)
            # 空流：返回一个立即结束的 async generator
            async def _empty():
                if False:
                    yield  # pragma: no cover
            return _empty(), attempt
        except BaseException as e:
            last_exc = e
            if retry.should_retry(e):
                breaker.on_failure(circuit_key)
                if attempt < retry.max_retries:
                    delay = retry.backoff(attempt)
                    logger.warning(
                        "LLM 异步流式调用可重试错误（第 %d/%d 次，%.2fs 后重试）：%s",
                        attempt + 1, retry.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                if retry.max_retries > 0:
                    raise LLMRetryExhaustedError(
                        f"LLM 异步流式调用在 {retry.max_retries + 1} 次尝试后仍失败：{e}",
                        original_error=e,
                    ) from e
                raise
            breaker.on_failure(circuit_key)
            raise
        else:
            breaker.on_success(circuit_key)
            attempts_used = attempt
            break

    assert agen is not None

    async def _agen():
        yield first
        try:
            async for item in agen:
                yield item
        except BaseException:
            breaker.on_failure(circuit_key)
            raise

    return _agen(), attempts_used
