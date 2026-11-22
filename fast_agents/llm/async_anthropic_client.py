import os
from typing import Optional, AsyncIterator, Any
import asyncio

from ..messages.message import Message, MsgToolCall
from .models import StreamChunk
from .errors import (
    LLMError,
    LLMConfigError,
    LLMInvokeError,
    LLMStructuredOutputError,
    LLMEmbeddingError,
)
from .openai_client import _EMBED_BATCH_SIZE, _dump_usage, _merge_usage
from .anthropic_client import (
    _import_anthropic,
    _classify_anthropic_error,
    _convert_messages_to_anthropic,
    _convert_anthropic_tools,
    _convert_anthropic_tool_calls,
)


class AsyncAnthropicClient:
    """Anthropic原生异步客户端，接口与AsyncOpenAIClient完全对齐。

    方法签名、参数语义、错误处理、usage回填与同步AnthropicClient完全一致：
    - ainvoke / astream_invoke / ainvoke_structured / aembed_query / aembed_documents
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
        anthropic, errors_map = _import_anthropic()
        self._anthropic = anthropic
        self._errors_map = errors_map

        try:
            client_kwargs = {
                "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
                "timeout": timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = anthropic.AsyncAnthropic(**client_kwargs)
        except Exception as e:
            raise LLMConfigError(
                f"Anthropic 异步客户端初始化失败（请检查 ANTHROPIC_API_KEY 配置）：{e}",
                original_error=e,
            ) from e

        self.model = model or os.getenv("LLM_MODEL_ID") or "claude-3-5-sonnet-latest"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.embedding_model = (
            embedding_model
            or os.getenv("LLM_EMBED_MODEL")
            or "voyage-3"
        )
        self.extra_body = dict(extra_body) if extra_body else None

    def _resolve_extra_body(self, kwargs: dict) -> Optional[dict]:
        """解析额外请求参数，与同步客户端逻辑一致。"""
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
        """异步非流式调用Anthropic模型，返回Message对象。"""
        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            anthropic_messages, system_prompt = _convert_messages_to_anthropic(messages)
            anthropic_tools = _convert_anthropic_tools(tools)

            create_kwargs = {
                "model": chat_model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            }
            if system_prompt:
                create_kwargs["system"] = system_prompt
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools
            if extra_body:
                create_kwargs.update(extra_body)

            response = await self._client.messages.create(**create_kwargs)

            if usage_sink is not None:
                usage = {
                    "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                    "completion_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
                }
                usage_sink.update(usage)

            content_text = ""
            tool_calls = []
            for block in response.content:
                if getattr(block, "type", "") == "text":
                    content_text += getattr(block, "text", "")
                elif getattr(block, "type", "") == "tool_use":
                    tool_calls.append(block)

            response_msg = Message(role="assistant", content=content_text)
            for tc in _convert_anthropic_tool_calls(tool_calls):
                response_msg.tool_calls.append(tc)

            return response_msg
        except LLMError:
            raise
        except Exception as e:
            mapped = _classify_anthropic_error(e, self._errors_map)
            raise LLMInvokeError(
                f"LLM 异步调用失败：{mapped}",
                original_error=e,
            ) from mapped

    async def astream_invoke(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式调用Anthropic模型，逐段yield StreamChunk。"""
        print(f"🧠 正在异步调用 {self.model} 模型...")
        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            anthropic_messages, system_prompt = _convert_messages_to_anthropic(messages)
            anthropic_tools = _convert_anthropic_tools(tools)

            create_kwargs = {
                "model": chat_model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            }
            if system_prompt:
                create_kwargs["system"] = system_prompt
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools
            if extra_body:
                create_kwargs.update(extra_body)

            response = await self._client.messages.create(**create_kwargs)
            print("✅ 大语言模型异步响应成功:")

            think_print_flg = False
            text_buf: list[str] = []
            tool_calls_builder: dict[int, dict] = {}
            finish_reason: str = ""
            current_tool_idx = 0

            async with response as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "message_stop":
                        break
                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            idx = current_tool_idx
                            current_tool_idx += 1
                            tool_calls_builder[idx] = {
                                "id": getattr(block, "id", ""),
                                "name": getattr(block, "name", ""),
                                "arguments": "",
                            }
                            yield StreamChunk(
                                type="tool_call",
                                index=idx,
                                id=tool_calls_builder[idx]["id"],
                                name=tool_calls_builder[idx]["name"],
                                arguments_delta="",
                            )
                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if not delta:
                            continue
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "thinking_delta":
                            thinking_text = getattr(delta, "thinking", "")
                            if thinking_text:
                                if not think_print_flg:
                                    print("[思考]")
                                    think_print_flg = True
                                print(thinking_text, end="", flush=True)
                            continue
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                text_buf.append(text)
                                yield StreamChunk(type="text", text=text)
                        elif delta_type == "input_json_delta":
                            args_delta = getattr(delta, "partial_json", "")
                            if args_delta and tool_calls_builder:
                                idx = max(tool_calls_builder.keys())
                                tool_calls_builder[idx]["arguments"] += args_delta
                                yield StreamChunk(
                                    type="tool_call",
                                    index=idx,
                                    id=tool_calls_builder[idx]["id"],
                                    name=tool_calls_builder[idx]["name"],
                                    arguments_delta=args_delta,
                                )
                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            finish_reason = getattr(delta, "stop_reason", "") or ""

            print()

            full_content = "".join(text_buf)
            final_tool_calls: list[MsgToolCall] = []
            import json
            for idx in sorted(tool_calls_builder.keys()):
                tc = tool_calls_builder[idx]
                args_str = tc["arguments"].strip()
                if not args_str.startswith("{"):
                    args_str = "{" + args_str
                if not args_str.endswith("}"):
                    args_str += "}"
                try:
                    json.loads(args_str)
                except json.JSONDecodeError:
                    args_str = "{}"
                final_tool_calls.append(MsgToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=args_str,
                ))
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
            mapped = _classify_anthropic_error(e, self._errors_map)
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
        """异步结构化调用，返回pydantic模型实例。"""
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

        schema_prompt = f"""
请严格按照以下JSON Schema输出结果，不要输出任何额外解释、Markdown标记或代码块，只返回纯JSON：
{_json.dumps(schema_dict, ensure_ascii=False, indent=2)}
"""
        structured_messages = list(messages)
        if structured_messages and structured_messages[-1]["role"] == "user":
            original_content = structured_messages[-1]["content"]
            structured_messages[-1]["content"] = f"{original_content}\n\n{schema_prompt}"
        else:
            structured_messages.append({"role": "user", "content": schema_prompt})

        try:
            response = await self.ainvoke(structured_messages, usage_sink=usage_sink, **kwargs)
            content = (response.content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

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
            raise LLMStructuredOutputError(
                f"结构化输出异步调用失败：{e}",
                original_error=e,
            ) from e

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）                                               #
    # ------------------------------------------------------------------ #

    async def aembed_query(self, text: str, *, usage_sink: Optional[dict] = None, **kwargs) -> list[float]:
        """异步单条Embedding调用。"""
        if not isinstance(text, str):
            raise TypeError(f"aembed_query 要求 text 为 str，收到 {type(text).__name__}")
        if not text.strip():
            raise ValueError("aembed_query 收到空字符串，无法生成 embedding")

        try:
            from openai import AsyncOpenAI
            voyage_client = AsyncOpenAI(
                api_key=os.getenv("VOYAGE_API_KEY", self.api_key),
                base_url="https://api.voyageai.com/v1",
            )
            embed_model = kwargs.pop("model", self.embedding_model)
            resp = await voyage_client.embeddings.create(
                model=embed_model,
                input=[text],
                **kwargs,
            )
            if not resp.data or not resp.data[0].embedding:
                from .errors import LLMResponseError
                raise LLMResponseError("embedding 响应为空或缺少向量数据")
            if usage_sink is not None:
                usage = _dump_usage(getattr(resp, "usage", None))
                if usage:
                    usage_sink.update(usage)
            return list(resp.data[0].embedding)
        except ImportError:
            raise LLMConfigError("Embedding 需要 openai SDK（项目已内置）")
        except LLMError:
            raise
        except (TypeError, ValueError):
            raise
        except Exception as e:
            from .openai_client import _classify_openai_error
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 异步调用失败（query）：{mapped}",
                original_error=e,
            ) from mapped

    async def aembed_documents(self, texts: list[str], *, usage_sink: Optional[dict] = None, **kwargs) -> list[list[float]]:
        """异步批量Embedding调用，协程级并发。"""
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"aembed_documents 要求 texts 为 list[str]，收到 {type(texts).__name__}"
            )
        cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        if not cleaned:
            return []

        embed_model = kwargs.pop("model", self.embedding_model)
        call_kwargs = kwargs
        batch_size = min(128, _EMBED_BATCH_SIZE)
        semaphore = asyncio.Semaphore(4)

        async def _one_batch(batch: list[str]):
            async with semaphore:
                from openai import AsyncOpenAI
                voyage_client = AsyncOpenAI(
                    api_key=os.getenv("VOYAGE_API_KEY", self.api_key),
                    base_url="https://api.voyageai.com/v1",
                )
                resp = await voyage_client.embeddings.create(
                    model=embed_model,
                    input=batch,
                    **call_kwargs,
                )
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors = [list(d.embedding) for d in ordered]
                return vectors, _dump_usage(getattr(resp, "usage", None))

        batches = [
            cleaned[i : i + batch_size]
            for i in range(0, len(cleaned), batch_size)
        ]

        try:
            from openai import AsyncOpenAI
            if len(batches) == 1:
                vectors, usage = await _one_batch(batches[0])
                _merge_usage(usage_sink, usage)
                return vectors
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
            from .openai_client import _classify_openai_error
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 异步调用失败（documents）：{mapped}",
                original_error=e,
            ) from mapped