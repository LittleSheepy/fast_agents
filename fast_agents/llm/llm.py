import os
import functools
import asyncio
from typing import Optional, Iterator, AsyncIterator, TYPE_CHECKING, Any

from .openai_client import OpenAIClient
from .async_openai_client import AsyncOpenAIClient
from .anthropic_client import AnthropicClient
from .async_anthropic_client import AsyncAnthropicClient
from .gemini_client import GeminiClient
from .async_gemini_client import AsyncGeminiClient
from .models import StreamChunk
from .errors import LLMError, LLMInvokeError, LLMEmbeddingError, LLMStructuredOutputError, LLMConfigError
from .retry import RetryConfig, run_with_retry, run_stream_with_retry, arun_with_retry, arun_stream_with_retry
from ..messages.message import Message

if TYPE_CHECKING:
    from ..trace import TraceLogger


def _merge_usage_metadata(metadata: dict, usage: dict) -> None:
    """把底层回传的 token 用量合并进 Span 的 metadata（仅在拿到数据时写入）。

    写入内容：整个 ``usage`` 明细 + ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` 三个平铺字段，便于直接过滤、聚合统计（如成本分析）。
    拿不到 usage 时（例如自定义 client 不回填、或流式接口）不写入任何键。
    """
    if not usage or metadata is None:
        return
    metadata["usage"] = dict(usage)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage:
            metadata[key] = usage[key]


class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: Optional[int] = None,
        embedding_model: Optional[str] = None,
        extra_body: Optional[dict] = None,
        tracer: Optional["TraceLogger"] = None,
        retry: Optional[RetryConfig] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("LLM_EMBED_MODEL")
        self.timeout = timeout
        self.provider = provider
        # provider 私有请求体扩展字段（如 DashScope / Qwen 的 enable_thinking）；
        # 默认 None 表示不发送任何扩展字段，保证对原生 OpenAI 端点严格兼容
        self.extra_body = dict(extra_body) if extra_body else None
        self.tracer = tracer  # 可选 tracer；None 表示不记录，行为与改动前完全一致
        # I4 重试/熔断：默认开启（max_retries=2、指数退避 Full Jitter + 断路器），
        # 显式传 ``retry=None`` 完全关闭重试/熔断（零开销路径）；
        # 传 ``RetryConfig(...)`` 可自定义参数，或 ``RetryConfig.disabled()`` 快速关闭。
        if retry is None:
            # 默认开启保守重试：最多 2 次重试 + 全局断路器
            self.retry_config: Optional[RetryConfig] = RetryConfig()
        else:
            self.retry_config = retry
        self._client = self._create_client()
        self._async_client: Optional[Any] = None
        self._async_client_lock = asyncio.Lock()  # 协程并发安全懒加载锁

    def _create_client(self):
        """创建底层client，根据provider参数选择对应实现。

        支持：
        - openai（默认）：OpenAI兼容协议，支持绝大多数国内中转端点
        - anthropic：Anthropic Claude原生API
        - gemini：Google Gemini原生API
        """
        provider = self.provider.lower().strip()
        client_kwargs = dict(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout if self.timeout is not None else 180,
            embedding_model=self.embedding_model,
        )
        if self.extra_body:
            client_kwargs["extra_body"] = self.extra_body

        if provider == "openai":
            return OpenAIClient(**client_kwargs)
        elif provider == "anthropic":
            return AnthropicClient(**client_kwargs)
        elif provider == "gemini":
            return GeminiClient(**client_kwargs)
        else:
            raise LLMConfigError(
                f"不支持的 LLM provider：{provider}，当前支持：openai / anthropic / gemini"
            )

    def _create_async_client(self):
        """创建异步client，参数与同步client完全一致，provider自动对齐。"""
        provider = self.provider.lower().strip()
        client_kwargs = dict(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout if self.timeout is not None else 180,
            embedding_model=self.embedding_model,
        )
        if self.extra_body:
            client_kwargs["extra_body"] = self.extra_body

        if provider == "openai":
            return AsyncOpenAIClient(**client_kwargs)
        elif provider == "anthropic":
            return AsyncAnthropicClient(**client_kwargs)
        elif provider == "gemini":
            return AsyncGeminiClient(**client_kwargs)
        else:
            raise LLMConfigError(
                f"不支持的 LLM provider：{provider}，当前支持：openai / anthropic / gemini"
            )

    async def _get_async_client(self):
        """协程安全的懒加载：首次 await 时初始化异步 client，后续复用同一实例。"""
        if self._async_client is None:
            async with self._async_client_lock:
                if self._async_client is None:  # double-check
                    self._async_client = self._create_async_client()
        return self._async_client

    def _circuit_key(self, *, model: Optional[str] = None) -> tuple[str, str]:
        """返回断路器维度 key（base_url, model），每个端点+模型独立熔断。

        兜底：用 __new__ 绕过 __init__ 的测试场景下，缺失属性按空串处理。
        """
        base = getattr(self, "base_url", None) or ""
        if model is None:
            model = getattr(self, "model", None) or ""
            if not model:
                model = getattr(self, "embedding_model", None) or ""
        return (base, model or "")

    @property
    def _retry(self) -> Optional[RetryConfig]:
        """兜底属性：老代码/测试用 __new__ 创建实例跳过 __init__ 时返回 None（零重试）。"""
        return getattr(self, "retry_config", None)

    # ------------------------------------------------------------------ #
    # Chat（对话）                                                         #
    # ------------------------------------------------------------------ #

    def invoke(self, messages: list[dict[str, Any]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> Message:
        """非流式对话调用，返回 Message 对象。

        可选参数 ``tracer`` 非 None 时会记录一条 ``llm.invoke`` Span；
        未传时退回到 ``self.tracer``，两者都为 None 则不打点，行为与改动前完全一致。

        异常：底层抛出的所有错误统一为 :mod:`fast_agents.llm.errors` 下的分层异常；
        可重试错误（连接/超时/限流/5xx）会按构造时的 :class:`RetryConfig` 自动指数退避重试，
        超过次数抛 :class:`LLMRetryExhaustedError`；断路器 OPEN 时抛 :class:`LLMCircuitOpenError`；
        不可重试错误（鉴权/配置/参数非法/JSON 解析失败）直接透传。
        异常发生时 tracer 会通过上下文管理器自动把错误写入 Span。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            result, _ = run_with_retry(
                functools.partial(self._client.invoke, messages=messages, tools=tools, **kwargs),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            return result
        # 打点：记录入参（messages 截断，避免日志过大）
        from ..trace import SpanKind
        # 先从 kwargs 里 pop tracer 相关，避免传给底层 openai.create
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.model)
        input_snapshot = {
            "model": used_model,
            "messages_count": len(messages) if messages else 0,
            "tools_count": len(tools) if tools else 0,
        }
        # usage_sink：底层把本次响应的 token 用量回填进来（不改动底层返回值语义）
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.invoke",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "has_tools": bool(tools)},
        ) as s:
            # 重试仅包裹底层网络调用，不重复打点；打点 span 是整个调用（含重试）的聚合。
            # 这样用户在 trace 里看到的仍是「一次业务调用」，而不是每次失败的重试尝试。
            def _call():
                return self._client.invoke(messages=messages, tools=tools, usage_sink=usage_sink, **kwargs)

            response, attempts_used = run_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {
                "role": response.role,
                "has_content": bool(response.content),
                "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
                "content_preview": (effective_tracer.truncate_preview(response.content, 200) if response.content else None),
                "retry_attempts": attempts_used,
            }
            _merge_usage_metadata(s.metadata, usage_sink)
            # 重试次数写入 metadata（便于 K3 成本评估分析）
            s.metadata["retry_attempts"] = attempts_used
            return response

    def stream_invoke(self, messages: list[dict[str, Any]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> Iterator[StreamChunk]:
        """流式对话调用，逐段 yield StreamChunk（text/tool_call/end 三态）。

        可选参数 ``tracer`` 非 None 时会记录一条 ``llm.stream_invoke`` Span（在生成器耗尽/异常时 end）。
        打点口径：chunks 统计 text 片数量，length 统计 text 片总长度，tool_calls_count 统计最终工具调用数量。
        重试策略：仅在「发起 HTTP 请求、获取首块之前」的网络错误可重试；一旦开始 yield chunks，
        后续错误直接抛出（避免文本片段错位/重复），与主流框架处理一致。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            gen, _ = run_stream_with_retry(
                functools.partial(self._client.stream_invoke, messages=messages, tools=tools, **kwargs),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            yield from gen
            return
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        input_snapshot = {
            "model": kwargs.get("model", self.model),
            "messages_count": len(messages) if messages else 0,
            "tools_count": len(tools) if tools else 0,
            "stream": True,
        }
        span = effective_tracer.start_span(
            name="llm.stream_invoke",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": kwargs.get("model", self.model), "stream": True},
        )
        chunks_count: int = 0
        text_length: int = 0
        tool_calls_count: int = 0
        content_buf: list[str] = []
        try:
            gen, stream_attempts = run_stream_with_retry(
                functools.partial(self._client.stream_invoke, messages=messages, tools=tools, **kwargs),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.model)),
            )
            for chunk in gen:
                if chunk.type == "text" and chunk.text:
                    chunks_count += 1
                    text_length += len(chunk.text)
                    content_buf.append(chunk.text)
                elif chunk.type == "end":
                    tool_calls_count = len(chunk.tool_calls) if chunk.tool_calls else 0
                    if not content_buf and chunk.content:
                        content_buf.append(chunk.content)
                        text_length = len(chunk.content)
                yield chunk
        except LLMError as e:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                },
                error=f"{type(e).__name__}: {e}",
            )
            raise
        except Exception as e:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                },
                error=f"{type(e).__name__}: {e}",
            )
            raise LLMInvokeError(
                f"LLM 流式调用失败：{e}",
                original_error=e,
            ) from e
        else:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "length": text_length,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                    "retry_attempts": stream_attempts,
                },
                metadata={"retry_attempts": stream_attempts},
            )

    def invoke_structured(
        self,
        messages: list[dict[str, Any]],
        response_format: type,
        *,
        strict: bool = False,
        tracer: Optional["TraceLogger"] = None,
        **kwargs,
    ):
        """非流式结构化调用，返回 pydantic 模型实例。

        通过 ``response_format`` 协议（``json_schema`` 模式）约束模型输出，
        再把 JSON 解析为指定的 pydantic 模型实例。

        参数 ``response_format`` 必须是 ``pydantic.BaseModel`` 子类；内部通过
        ``model.model_json_schema()`` 自动转 JSON Schema，调用方无需手动构造。

        ``strict=False``（默认）宽容模式，兼容绝大多数端点；``strict=True`` 开启严格约束。

        可选参数 ``tracer`` 非 None 时记录一条 ``llm.invoke_structured`` Span；
        未传时退回到 ``self.tracer``，两者都为 None 则不打点。

        异常：JSON 解析失败 / pydantic 校验不通过抛 :class:`LLMStructuredOutputError`
        （不可重试）；网络 / API 层错误走分层异常并按 :class:`RetryConfig` 自动重试。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            result, _ = run_with_retry(
                functools.partial(
                    self._client.invoke_structured,
                    messages=messages, response_format=response_format, strict=strict, **kwargs,
                ),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.model)
        input_snapshot = {
            "model": used_model,
            "messages_count": len(messages) if messages else 0,
            "response_format": response_format.__name__,
            "strict": strict,
        }
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.invoke_structured",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={
                "model": used_model,
                "response_format": response_format.__name__,
                "strict": strict,
            },
        ) as s:
            def _call():
                return self._client.invoke_structured(
                    messages=messages, response_format=response_format,
                    strict=strict, usage_sink=usage_sink, **kwargs,
                )
            result, attempts_used = run_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {
                "parsed_type": type(result).__name__,
                "fields": list(result.model_dump().keys()) if hasattr(result, "model_dump") else [],
            }
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return result

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）—— 语义检索的前置能力                            #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str, tracer: Optional["TraceLogger"] = None, **kwargs) -> list[float]:
        """
        对单条「查询侧」文本生成 embedding 向量。

        典型用途：Agent 在做语义检索 / RAG 时，把用户问题转成向量去向量库查。
        返回值：一维浮点列表，维度由 embedding_model 决定（text-embedding-3-small 为 1536）。

        命名与签名对齐 LangChain，显式区分「单条/查询侧」与「批量/文档侧」，
        后续接 BGE / E5 / DashScope 这类需 query/document 不同前缀或 input_type 的模型时，
        接口无需改动。

        异常：参数错误抛原生 ``TypeError`` / ``ValueError``；API 错误统一为
        :class:`LLMEmbeddingError` 及其子类；可重试错误自动指数退避。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            result, _ = run_with_retry(
                functools.partial(self._client.embed_query, text, **kwargs),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.embedding_model)),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.embedding_model)
        input_snapshot = {"text_length": len(text) if isinstance(text, str) else 0, "model": used_model}
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.embed_query",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "op": "embed_query"},
        ) as s:
            def _call():
                return self._client.embed_query(text, usage_sink=usage_sink, **kwargs)
            vec, attempts_used = run_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {"dim": len(vec) if vec else 0}
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return vec

    def embed_documents(self, texts: list[str], tracer: Optional["TraceLogger"] = None, **kwargs) -> list[list[float]]:
        """
        对一批「文档侧」文本批量生成 embedding 向量。

        典型用途：向知识库写入文档（已分块）时，把每个 chunk 向量化后存入库。
        返回值：二维浮点列表，顺序与输入 texts 一一对应。
        内部自动按 2048 条分片并并发请求，调用方无需关心 API 批次上限。

        异常：与 :meth:`embed_query` 一致，API 错误统一为 :class:`LLMEmbeddingError`，
        可重试错误自动指数退避。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            result, _ = run_with_retry(
                functools.partial(self._client.embed_documents, texts, **kwargs),
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.embedding_model)),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.embedding_model)
        input_snapshot = {"batch_size": len(texts) if texts else 0, "model": used_model}
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.embed_documents",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "op": "embed_documents"},
        ) as s:
            def _call():
                return self._client.embed_documents(texts, usage_sink=usage_sink, **kwargs)
            vecs, attempts_used = run_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {"batch": len(vecs) if vecs else 0, "dim": len(vecs[0]) if vecs and vecs[0] else 0}
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return vecs

    # ------------------------------------------------------------------ #
    # Async 异步方法（I2）—— 与同步方法一一对应，支持事件循环并发调用        #
    # ------------------------------------------------------------------ #

    async def ainvoke(self, messages: list[dict[str, Any]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> Message:
        """异步非流式对话调用，语义与同步 :meth:`invoke` 完全一致，返回 ``await`` 后的 Message 对象。

        底层异步 client 懒加载，首次调用时初始化，后续复用同一实例（协程安全）。
        可选参数 ``tracer`` 非 None 时记录 ``llm.ainvoke`` Span，metadata 额外标记 ``async=True``。
        """
        client = await self._get_async_client()
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            async def _call():
                return await client.ainvoke(messages=messages, tools=tools, **kwargs)
            result, _ = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.model)
        input_snapshot = {
            "model": used_model,
            "messages_count": len(messages) if messages else 0,
            "tools_count": len(tools) if tools else 0,
            "async": True,
        }
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.ainvoke",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "has_tools": bool(tools), "async": True},
        ) as s:
            async def _call():
                return await client.ainvoke(messages=messages, tools=tools, usage_sink=usage_sink, **kwargs)

            response, attempts_used = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {
                "role": response.role,
                "has_content": bool(response.content),
                "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
                "content_preview": (effective_tracer.truncate_preview(response.content, 200) if response.content else None),
                "retry_attempts": attempts_used,
            }
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return response

    async def astream_invoke(self, messages: list[dict[str, Any]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式对话调用，逐段 ``async for`` yield StreamChunk，语义与同步 :meth:`stream_invoke` 完全一致。

        Span 命名 ``llm.astream_invoke``，metadata 标记 ``async=True``；
        重试策略：仅在获取首块前可重试，开始 yield chunks 后错误直接抛出。
        """
        client = await self._get_async_client()
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            async def _call():
                return client.astream_invoke(messages=messages, tools=tools, **kwargs)
            gen, _ = await arun_stream_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            async for chunk in gen:
                yield chunk
            return
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        input_snapshot = {
            "model": kwargs.get("model", self.model),
            "messages_count": len(messages) if messages else 0,
            "tools_count": len(tools) if tools else 0,
            "stream": True,
            "async": True,
        }
        span = effective_tracer.start_span(
            name="llm.astream_invoke",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": kwargs.get("model", self.model), "stream": True, "async": True},
        )
        chunks_count: int = 0
        text_length: int = 0
        tool_calls_count: int = 0
        content_buf: list[str] = []
        try:
            async def _call():
                return client.astream_invoke(messages=messages, tools=tools, **kwargs)
            gen, stream_attempts = await arun_stream_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.model)),
            )
            async for chunk in gen:
                if chunk.type == "text" and chunk.text:
                    chunks_count += 1
                    text_length += len(chunk.text)
                    content_buf.append(chunk.text)
                elif chunk.type == "end":
                    tool_calls_count = len(chunk.tool_calls) if chunk.tool_calls else 0
                    if not content_buf and chunk.content:
                        content_buf.append(chunk.content)
                        text_length = len(chunk.content)
                yield chunk
        except LLMError as e:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                },
                error=f"{type(e).__name__}: {e}",
            )
            raise
        except Exception as e:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                },
                error=f"{type(e).__name__}: {e}",
            )
            raise LLMInvokeError(
                f"LLM 异步流式调用失败：{e}",
                original_error=e,
            ) from e
        else:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": chunks_count,
                    "tool_calls_count": tool_calls_count,
                    "length": text_length,
                    "content_preview": effective_tracer.truncate_preview("".join(content_buf), 200),
                    "retry_attempts": stream_attempts,
                },
                metadata={"retry_attempts": stream_attempts, "async": True},
            )

    async def ainvoke_structured(
        self,
        messages: list[dict[str, Any]],
        response_format: type,
        *,
        strict: bool = False,
        tracer: Optional["TraceLogger"] = None,
        **kwargs,
    ):
        """异步结构化调用，返回 pydantic 模型实例，语义与同步 :meth:`invoke_structured` 完全一致。

        Span 命名 ``llm.ainvoke_structured``，metadata 标记 ``async=True``。
        """
        client = await self._get_async_client()
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            async def _call():
                return await client.ainvoke_structured(
                    messages=messages, response_format=response_format, strict=strict, **kwargs,
                )
            result, _ = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model")),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.model)
        input_snapshot = {
            "model": used_model,
            "messages_count": len(messages) if messages else 0,
            "response_format": response_format.__name__,
            "strict": strict,
            "async": True,
        }
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.ainvoke_structured",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={
                "model": used_model,
                "response_format": response_format.__name__,
                "strict": strict,
                "async": True,
            },
        ) as s:
            async def _call():
                return await client.ainvoke_structured(
                    messages=messages, response_format=response_format,
                    strict=strict, usage_sink=usage_sink, **kwargs,
                )
            result, attempts_used = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {
                "parsed_type": type(result).__name__,
                "fields": list(result.model_dump().keys()) if hasattr(result, "model_dump") else [],
            }
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return result

    async def aembed_query(self, text: str, tracer: Optional["TraceLogger"] = None, **kwargs) -> list[float]:
        """异步单条 embedding 生成，语义与同步 :meth:`embed_query` 完全一致。

        Span 命名 ``llm.aembed_query``，metadata 标记 ``async=True``。
        """
        client = await self._get_async_client()
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            async def _call():
                return await client.aembed_query(text, **kwargs)
            result, _ = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.embedding_model)),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.embedding_model)
        input_snapshot = {"text_length": len(text) if isinstance(text, str) else 0, "model": used_model, "async": True}
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.aembed_query",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "op": "aembed_query", "async": True},
        ) as s:
            async def _call():
                return await client.aembed_query(text, usage_sink=usage_sink, **kwargs)
            vec, attempts_used = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {"dim": len(vec) if vec else 0}
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return vec

    async def aembed_documents(self, texts: list[str], tracer: Optional["TraceLogger"] = None, **kwargs) -> list[list[float]]:
        """异步批量 embedding 生成，语义与同步 :meth:`embed_documents` 完全一致，使用协程级并发。

        Span 命名 ``llm.aembed_documents``，metadata 标记 ``async=True``。
        """
        client = await self._get_async_client()
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            async def _call():
                return await client.aembed_documents(texts, **kwargs)
            result, _ = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=kwargs.get("model", self.embedding_model)),
            )
            return result
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        _ = kwargs.pop("usage_sink", None)
        used_model = kwargs.get("model", self.embedding_model)
        input_snapshot = {"batch_size": len(texts) if texts else 0, "model": used_model, "async": True}
        usage_sink: dict = {}
        with effective_tracer.span(
            name="llm.aembed_documents",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": used_model, "op": "aembed_documents", "async": True},
        ) as s:
            async def _call():
                return await client.aembed_documents(texts, usage_sink=usage_sink, **kwargs)
            vecs, attempts_used = await arun_with_retry(
                _call,
                retry=self._retry,
                circuit_key=self._circuit_key(model=used_model),
            )
            s.output = {"batch": len(vecs) if vecs else 0, "dim": len(vecs[0]) if vecs and vecs[0] else 0}
            _merge_usage_metadata(s.metadata, usage_sink)
            s.metadata["retry_attempts"] = attempts_used
            return vecs
