"""LLM 重试/熔断相关异常。

- :class:`LLMRetryExhaustedError`：所有重试次数耗尽仍未成功时抛出；
- :class:`LLMCircuitOpenError`：断路器处于 OPEN 状态、请求被快速拒绝时抛出；
- 两者均为不可重试（``retryable=False``），因为重试本身已经尽力/被判定为危险。
"""
from __future__ import annotations

from .errors import LLMError


class LLMRetryExhaustedError(LLMError):
    """指数退避重试达到 ``max_retries`` 上限后仍未成功，包装最后一次失败的异常。"""

    retryable: bool = False


class LLMCircuitOpenError(LLMError):
    """断路器处于 OPEN 状态，请求被快速拒绝（不真正触达下游）。"""

    retryable: bool = False
