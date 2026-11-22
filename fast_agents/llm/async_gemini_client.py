import os
import json
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
from .gemini_client import (
    _import_google_genai,
    _classify_google_error,
    _convert_messages_to_gemini,
    _convert_gemini_tools,
    _convert_gemini_tool_calls,
)


class AsyncGeminiClient:
    """Google Gemini原生异步客户端，接口与AsyncOpenAIClient完全对齐。

    方法签名、参数语义、错误处理、usage回填与同步GeminiClient完全一致：
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
        # 注意：google-generativeai SDK的异步客户端通过同一实例的async方法调用，无需单独AsyncGenerativeModel
        genai, _ = _import_google_genai()
        self._genai = genai

        api_key_val = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key_val:
            raise LLMConfigError("Gemini 异步客户端初始化失败：未设置 GOOGLE_API_KEY/GEMINI_API_KEY 环境变量")
        try:
            genai.configure(api_key=api_key_val, transport="rest")
            if base_url:
                genai.configure(client_options={"api_endpoint": base_url})
        except Exception as e:
            raise LLMConfigError(
                f"Gemini 异步客户端初始化失败：{e}",
                original_error=e,
            ) from e

        self.model = model or os.getenv("LLM_MODEL_ID") or "gemini-2.0-flash"
        self.api_key = api_key_val
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.embedding_model = (
            embedding_model
            or os.getenv("LLM_EMBED_MODEL")
            or "text-embedding-004"
        )
        self.extra_body = dict(extra_body) if extra_body else None

    def _get_model(self, model_name: str, tools=None, **kwargs):
        """获取Gemini GenerativeModel实例。"""
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        model_kwargs = {
            "model_name": model_name,
            "generation_config": generation_config,
        }
        gemini_tools = _convert_gemini_tools(tools)
        if gemini_tools:
            model_kwargs["tools"] = gemini_tools
        return self._genai.GenerativeModel(**model_kwargs)

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
        """异步非流式调用Gemini模型，返回Message对象。"""
        try:
            chat_model_name = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)

            gemini_contents, system_instruction = _convert_messages_to_gemini(messages)
            model = self._get_model(chat_model_name, tools=tools, **kwargs)

            generate_kwargs = {
                "contents": gemini_contents,
            }
            if system_instruction:
                generate_kwargs["system_instruction"] = system_instruction
            if extra_body:
                generate_kwargs.update(extra_body)

            response = await model.generate_content_async(**generate_kwargs)

            if usage_sink is not None:
                usage_metadata = getattr(response, "usage_metadata", None)
                if usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0),
                        "total_tokens": getattr(usage_metadata, "total_token_count", 0),
                    }
                    usage_sink.update(usage)

            content_text = ""
            tool_calls = []
            if response.candidates:
                candidate = response.candidates[0]
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_text += part.text
                    tc = _convert_gemini_tool_calls(part)
                    if tc:
                        tool_calls.append(tc)

            response_msg = Message(role="assistant", content=content_text)
            for tc in tool_calls:
                response_msg.tool_calls.append(tc)
            return response_msg
        except LLMError:
            raise
        except Exception as e:
            mapped = _classify_google_error(e)
            raise LLMInvokeError(
                f"LLM 异步调用失败：{mapped}",
                original_error=e,
            ) from mapped

    async def astream_invoke(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式调用Gemini模型，逐段yield StreamChunk。"""
        print(f"🧠 正在异步调用 {self.model} 模型...")
        try:
            chat_model_name = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)

            gemini_contents, system_instruction = _convert_messages_to_gemini(messages)
            model = self._get_model(chat_model_name, tools=tools, **kwargs)

            generate_kwargs = {
                "contents": gemini_contents,
                "stream": True,
            }
            if system_instruction:
                generate_kwargs["system_instruction"] = system_instruction
            if extra_body:
                generate_kwargs.update(extra_body)

            response = await model.generate_content_async(**generate_kwargs)
            print("✅ 大语言模型异步响应成功:")

            text_buf: list[str] = []
            tool_calls_builder: dict[int, dict] = {}
            finish_reason: str = ""
            current_tool_idx = 0

            async for chunk in await response:
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    finish_reason = str(getattr(candidate, "finish_reason", "")) or finish_reason
                    if not hasattr(candidate, "content") or not candidate.content.parts:
                        continue
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text = part.text
                            if text:
                                text_buf.append(text)
                                yield StreamChunk(type="text", text=text)
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            idx = current_tool_idx
                            current_tool_idx += 1
                            args_str = json.dumps(dict(fc.args))
                            tool_calls_builder[idx] = {
                                "id": fc.name + "_" + str(hash(args_str))[:8],
                                "name": fc.name,
                                "arguments": args_str,
                            }
                            yield StreamChunk(
                                type="tool_call",
                                index=idx,
                                id=tool_calls_builder[idx]["id"],
                                name=tool_calls_builder[idx]["name"],
                                arguments_delta=args_str,
                            )

            print()

            full_content = "".join(text_buf)
            final_tool_calls: list[MsgToolCall] = []
            for idx in sorted(tool_calls_builder.keys()):
                tc = tool_calls_builder[idx]
                final_tool_calls.append(MsgToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
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
            mapped = _classify_google_error(e)
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
{json.dumps(schema_dict, ensure_ascii=False, indent=2)}
"""
        structured_messages = list(messages)
        if structured_messages and structured_messages[-1]["role"] == "user":
            original_content = structured_messages[-1]["content"]
            structured_messages[-1]["content"] = f"{original_content}\n\n{schema_prompt}"
        else:
            structured_messages.append({"role": "user", "content": schema_prompt})

        try:
            kwargs["response_mime_type"] = "application/json"
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
                data = json.loads(content)
            except json.JSONDecodeError as e:
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
            embed_model_name = kwargs.pop("model", self.embedding_model)
            # google-generativeai SDK的embed_content_async方法
            result = await self._genai.embed_content_async(
                model=f"models/{embed_model_name}",
                content=text,
                task_type="retrieval_query",
            )
            embedding = result["embedding"]
            if not embedding:
                from .errors import LLMResponseError
                raise LLMResponseError("embedding 响应为空或缺少向量数据")
            if usage_sink is not None:
                usage_sink.update({
                    "prompt_tokens": len(text) // 4,
                    "total_tokens": len(text) // 4,
                })
            return list(embedding)
        except LLMError:
            raise
        except (TypeError, ValueError):
            raise
        except Exception as e:
            mapped = _classify_google_error(e)
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

        embed_model_name = kwargs.pop("model", self.embedding_model)
        batch_size = min(100, _EMBED_BATCH_SIZE)
        semaphore = asyncio.Semaphore(4)

        async def _one_batch(batch: list[str]):
            async with semaphore:
                result = await self._genai.embed_content_async(
                    model=f"models/{embed_model_name}",
                    content=batch,
                    task_type="retrieval_document",
                )
                vectors = [list(emb) for emb in result["embedding"]]
                total_chars = sum(len(t) for t in batch)
                usage = {
                    "prompt_tokens": total_chars // 4,
                    "total_tokens": total_chars // 4,
                }
                return vectors, usage

        batches = [
            cleaned[i : i + batch_size]
            for i in range(0, len(cleaned), batch_size)
        ]

        try:
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
            mapped = _classify_google_error(e)
            raise LLMEmbeddingError(
                f"Embedding 异步调用失败（documents）：{mapped}",
                original_error=e,
            ) from mapped