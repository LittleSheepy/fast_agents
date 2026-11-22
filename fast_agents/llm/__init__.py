"""LLM模块。

对外暴露：
- :class:`LLM`：统一门面类，上层唯一入口；
- :class:`StreamChunk`：stream_invoke 流式输出契约（text / tool_call / end 三态）；
- :mod:`.errors`：LLM 分层异常体系（含基类与所有可重试/不可重试子类），
  上层做重试 / 熔断 / 用户提示分流时直接从此处导入；
- :class:`RetryConfig` / :class:`CircuitBreakerRegistry` /
  :class:`LLMRetryExhaustedError` / :class:`LLMCircuitOpenError`：
  I4 指数退避重试与断路器，默认开启、可通过构造参数调优或关闭。
- :class:`AsyncOpenAIClient`：I2 异步OpenAI底层客户端，门面类自动懒加载；
- :class:`OpenAIClient`：同步OpenAI底层客户端，默认OpenAI兼容协议；
- :class:`AnthropicClient` / :class:`AsyncAnthropicClient`：I3 Anthropic原生客户端；
- :class:`GeminiClient` / :class:`AsyncGeminiClient`：I3 Google Gemini原生客户端；
- :func:`arun_with_retry` / :func:`arun_stream_with_retry`：
  I2 异步重试执行器，与同步版本共用 RetryConfig / 断路器配置。
"""

from .llm import LLM
from .models import StreamChunk
from .openai_client import OpenAIClient
from .async_openai_client import AsyncOpenAIClient
from .anthropic_client import AnthropicClient
from .async_anthropic_client import AsyncAnthropicClient
from .gemini_client import GeminiClient
from .async_gemini_client import AsyncGeminiClient
from . import errors
from .errors import (
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMServerError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMModelNotFoundError,
    LLMBadRequestError,
    LLMApiError,
    LLMResponseError,
    LLMInvokeError,
    LLMEmbeddingError,
    LLMStructuredOutputError,
)
from .retry import (
    RetryConfig, CircuitBreakerRegistry,
    run_with_retry, run_stream_with_retry,
    arun_with_retry, arun_stream_with_retry,
)
from .retry_errors import LLMRetryExhaustedError, LLMCircuitOpenError

__all__ = [
    "LLM",
    "StreamChunk",
    # I3 多provider客户端
    "OpenAIClient",
    "AsyncOpenAIClient",
    "AnthropicClient",
    "AsyncAnthropicClient",
    "GeminiClient",
    "AsyncGeminiClient",
    "errors",
    # 分层异常
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMAuthenticationError",
    "LLMConfigError",
    "LLMModelNotFoundError",
    "LLMBadRequestError",
    "LLMApiError",
    "LLMResponseError",
    "LLMInvokeError",
    "LLMEmbeddingError",
    "LLMStructuredOutputError",
    # I4 重试 / 熔断
    "RetryConfig",
    "CircuitBreakerRegistry",
    "run_with_retry",
    "run_stream_with_retry",
    "LLMRetryExhaustedError",
    "LLMCircuitOpenError",
    # I2 异步能力
    "arun_with_retry",
    "arun_stream_with_retry",
]