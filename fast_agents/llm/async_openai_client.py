import os
from typing import Optional, AsyncIterator, Any
import asyncio

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    APIStatusError,
    APIError,
)

from ..messages.message import Message, MsgToolCall
from .models import StreamChunk
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
from .openai_client import (
    _EMBED_BATCH_SIZE,
    _dump_usage,
    _merge_usage,
    _classify_openai_error,
)


class AsyncOpenAIClient(AsyncOpenAI):
    """OpenAI 协议异步客户端，与同步 :class:`OpenAIClient` 一一对应。

    继承自 ``openai.AsyncOpenAI``，底层使用异步 HTTP 客户端（``httpx.AsyncClient``），
    所有网络调用均不阻塞事件循环，可在 FastAPI / 异步 Agent 框架中并发使用。

    方法签名、参数语义、错误处理、usage 回填与同步版本完全一致：
    - :meth:`ainvoke` → 同步 ``invoke`` 的异步版本
    - :meth:`astream_invoke` → 同步 ``stream_invoke`` 的异步版本（yield async StreamChunk）
    - :meth:`ainvoke_structured` → 同步 ``invoke_structured`` 的异步版本
    - :meth:`aembed_query` → 同步 ``embed_query`` 的异步版本
    - :meth:`aembed_documents` → 同步 ``embed_documents`` 的异步版本（使用 asyncio.Semaphore 协程级并发）
    """

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            timeout=180,
            embedding_model: Optional[str] = None,
            extra_body: Optional[dict] = None,
            **kwargs
    ):
        try:
            super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        except Exception as e:
            raise LLMConfigError(
                f"LLM 异步客户端初始化失败（请检查 LLM_BASE_URL / LLM_API_KEY 配置）：{e}",
                original_error=e,
            ) from e

        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.embedding_model = (
            embedding_model
            or os.getenv("LLM_EMBED_MODEL")
            or "text-embedding-3-small"
        )
        self.extra_body = dict(extra_body) if extra_body else None

    def _resolve_extra_body(self, kwargs: dict) -> Optional[dict]:
        """与同步 OpenAIClient._resolve_extra_body 逻辑完全一致。"""
        merged: dict = {}
        if self.extra_body:
            merged.update(self.extra_body)
        call_extra = kwargs.pop("extra_body", None)
        if call_extra:
            merged.update(call_extra)
        return merged or None

    # ------------------------------------------------------------------ #
    # Chat（对话）                                                         #
    # ------------------------------------------------------------------ #

    async def ainvoke(self, messages: list[dict[str, Any]], tools=None, *, usage_sink: Optional[dict] = None, **kwargs) -> Message:
        """异步非流式调用，语义与同步 ``invoke`` 完全一致。"""
        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            create_kwargs = {
                "model": chat_model,
                "messages": messages,
                "tools": tools,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            }
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = await self.chat.completions.create(**create_kwargs)

            if usage_sink is not None:
                usage = _dump_usage(getattr(response, "usage", None))
                if usage:
                    usage_sink.update(usage)
            message = response.choices[0].message
            response_msg = Message(role="assistant", content=message.content)
            for tool in (getattr(message, "tool_calls", None) or []):
                msg_tool_call = MsgToolCall(name=tool.function.name, arguments=tool.function.arguments, id=tool.id)
                response_msg.tool_calls.append(msg_tool_call)

            return response_msg
        except LLMError:
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMInvokeError(
                f"LLM 异步调用失败：{mapped}",
                original_error=e,
            ) from mapped

    async def astream_invoke(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式调用，逐段 yield StreamChunk，语义与同步 ``stream_invoke`` 完全一致。"""
        print(f"🧠 正在异步调用 {self.model} 模型...")
        response = None
        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            create_kwargs = {
                "model": chat_model,
                "messages": messages,
                "tools": tools,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            }
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = await self.chat.completions.create(**create_kwargs)

            print("✅ 大语言模型异步响应成功:")
            think_print_flg = False
            text_buf: list[str] = []
            tool_calls_builder: dict[int, dict] = {}
            finish_reason: str = ""

            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                # 思考过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not think_print_flg:
                        print("[思考]")
                        think_print_flg = True
                    print(f"{delta.reasoning_content}", end="", flush=True)
                    continue

                content = getattr(delta, "content", None) or ""
                if content:
                    text_buf.append(content)
                    yield StreamChunk(type="text", text=content)

                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                for tc_delta in delta_tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_builder:
                        tool_calls_builder[idx] = {
                            "id": getattr(tc_delta, "id", "") or "",
                            "name": getattr(getattr(tc_delta, "function", None), "name", "") or "",
                            "arguments": "",
                        }
                    fn = getattr(tc_delta, "function", None)
                    args_delta = getattr(fn, "arguments", "") or ""
                    if args_delta:
                        tool_calls_builder[idx]["arguments"] += args_delta
                    yield StreamChunk(
                        type="tool_call",
                        index=idx,
                        id=tool_calls_builder[idx]["id"],
                        name=tool_calls_builder[idx]["name"],
                        arguments_delta=args_delta,
                    )

            print()

            full_content = "".join(text_buf)
            final_tool_calls: list[MsgToolCall] = []
            for idx in sorted(tool_calls_builder.keys()):
                tc = tool_calls_builder[idx]
                final_tool_calls.append(
                    MsgToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc["arguments"],
                    )
                )
            yield StreamChunk(
                type="end",
                finish_reason=finish_reason,
                content=full_content,
                tool_calls=final_tool_calls,
            )

        except LLMError:
            raise
        except Exception as e:
            print(f"❌ 异步调用LLM API时发生错误: {e}")
            mapped = _classify_openai_error(e)
            raise LLMInvokeError(
                f"LLM 异步流式调用失败：{mapped}",
                original_error=e,
            ) from mapped

    # ------------------------------------------------------------------ #
    # Structured Output（结构化输出）                                        #
    # ------------------------------------------------------------------ #

    async def ainvoke_structured(
        self,
        messages: list[dict[str, Any]],
        response_format: type,
        *,
        strict: bool = False,
        usage_sink: Optional[dict] = None,
        **kwargs,
    ):
        """异步结构化调用，返回 pydantic 模型实例，语义与同步 ``invoke_structured`` 完全一致。"""
        import json as _json

        try:
            from pydantic import BaseModel as _PydanticBaseModel
        except ImportError:
            raise TypeError("ainvoke_structured 需要 pydantic 依赖（项目已内置）")
        if not isinstance(response_format, type) or not issubclass(response_format, _PydanticBaseModel):
            raise TypeError(
                f"response_format 必须是 pydantic.BaseModel 子类，收到 {response_format!r}"
            )

        try:
            schema_dict = response_format.model_json_schema()
        except Exception as e:
            raise LLMStructuredOutputError(
                f"无法从 {response_format.__name__} 生成 JSON Schema：{e}",
                original_error=e,
            ) from e

        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            create_kwargs = {
                "model": chat_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema_dict,
                        "strict": strict,
                    },
                },
                **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
            }
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = await self.chat.completions.create(**create_kwargs)

            if usage_sink is not None:
                usage = _dump_usage(getattr(response, "usage", None))
                if usage:
                    usage_sink.update(usage)

            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise LLMStructuredOutputError(
                    f"结构化输出响应内容为空，无法解析为 {response_format.__name__}"
                )
            try:
                data = _json.loads(content)
            except _json.JSONDecodeError as e:
                raise LLMStructuredOutputError(
                    f"结构化输出 JSON 解析失败：{e}；原始内容前 200 字：{content[:200]!r}",
                    original_error=e,
                ) from e

            try:
                return response_format.model_validate(data)
            except Exception as e:
                raise LLMStructuredOutputError(
                    f"结构化输出 pydantic 校验失败（{response_format.__name__}）：{e}",
                    original_error=e,
                ) from e

        except LLMError:
            raise
        except (TypeError, ValueError):
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMStructuredOutputError(
                f"结构化输出异步调用失败：{mapped}",
                original_error=e,
            ) from mapped

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）                                               #
    # ------------------------------------------------------------------ #

    async def aembed_query(self, text: str, *, usage_sink: Optional[dict] = None, **kwargs) -> list[float]:
        """异步单条 embedding，语义与同步 ``embed_query`` 完全一致。"""
        if not isinstance(text, str):
            raise TypeError(f"aembed_query 要求 text 为 str，收到 {type(text).__name__}")
        if not text.strip():
            raise ValueError("aembed_query 收到空字符串，无法生成 embedding")

        try:
            embed_model = kwargs.pop("model", self.embedding_model)
            resp = await self.embeddings.create(
                model=embed_model,
                input=[text],
                **kwargs,
            )
            if not resp.data or not resp.data[0].embedding:
                raise LLMResponseError("embedding 响应为空或缺少向量数据")
            if usage_sink is not None:
                usage = _dump_usage(getattr(resp, "usage", None))
                if usage:
                    usage_sink.update(usage)
            return list(resp.data[0].embedding)
        except LLMError:
            raise
        except (TypeError, ValueError):
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 异步调用失败（query）：{mapped}",
                original_error=e,
            ) from mapped

    async def aembed_documents(self, texts: list[str], *, usage_sink: Optional[dict] = None, **kwargs) -> list[list[float]]:
        """异步批量 embedding，协程级并发（asyncio.Semaphore(4)），语义与同步 ``embed_documents`` 完全一致。"""
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"aembed_documents 要求 texts 为 list[str]，收到 {type(texts).__name__}"
            )
        cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        if not cleaned:
            return []

        embed_model = kwargs.pop("model", self.embedding_model)
        # 取调用方传入的 extra kwargs（剔除 model）
        call_kwargs = kwargs
        semaphore = asyncio.Semaphore(4)  # 与同步版本 max_workers=4 对齐

        async def _one_batch(batch: list[str]):
            async with semaphore:
                resp = await self.embeddings.create(
                    model=embed_model,
                    input=batch,
                    **call_kwargs,
                )
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors = [list(d.embedding) for d in ordered]
                return vectors, _dump_usage(getattr(resp, "usage", None))

        batches = [
            cleaned[i : i + _EMBED_BATCH_SIZE]
            for i in range(0, len(cleaned), _EMBED_BATCH_SIZE)
        ]

        try:
            if len(batches) == 1:
                vectors, usage = await _one_batch(batches[0])
                _merge_usage(usage_sink, usage)
                return vectors
            # 多批次协程并发
            tasks = [_one_batch(b) for b in batches]
            results = await asyncio.gather(*tasks)
            merged: list[list[float]] = []
            for part, usage in results:
                merged.extend(part)
                _merge_usage(usage_sink, usage)
            return merged
        except LLMError:
            raise
        except (TypeError, ValueError):
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 异步调用失败（documents）：{mapped}",
                original_error=e,
            ) from mapped


if __name__ == '__main__':
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    async def _main():
        client = AsyncOpenAIClient()

        # ---- async chat 流式自测 ----
        print("=== async chat 流式 ===")
        async for chunk in client.astream_invoke(messages=[{"role": "user", "content": "1+1=?"}]):
            if chunk.type == "text":
                print(chunk.text, end="", flush=True)
        print()

        # ---- async embedding 自测 ----
        print("=== async embed_query ===")
        qv = await client.aembed_query("今天天气怎么样")
        print(f"维度: {len(qv)}, 前 5 维: {qv[:5]}")

        print("=== async embed_documents (3 条) ===")
        dvs = await client.aembed_documents([
            "今天天气怎么样",
            "明天会下雨吗",
            "帮我查一下快递",
        ])
        print(f"返回条数: {len(dvs)}, 每条维度: {len(dvs[0]) if dvs else 0}")

    asyncio.run(_main())
