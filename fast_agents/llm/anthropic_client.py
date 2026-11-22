import os
from typing import Optional, Iterator, Any
from concurrent.futures import ThreadPoolExecutor

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
from .openai_client import _EMBED_BATCH_SIZE, _dump_usage, _merge_usage


def _import_anthropic():
    """懒导入Anthropic SDK，未安装时给出明确安装提示。"""
    try:
        import anthropic
        from anthropic import (
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
        return anthropic, {
            "APIConnectionError": APIConnectionError,
            "APITimeoutError": APITimeoutError,
            "RateLimitError": RateLimitError,
            "AuthenticationError": AuthenticationError,
            "PermissionDeniedError": PermissionDeniedError,
            "NotFoundError": NotFoundError,
            "BadRequestError": BadRequestError,
            "APIStatusError": APIStatusError,
            "APIError": APIError,
        }
    except ImportError:
        raise LLMConfigError(
            "使用 Anthropic provider 需要安装 anthropic SDK：pip install fast-agents[anthropic] 或 pip install anthropic>=0.39.0"
        )


def _classify_anthropic_error(e: Exception, errors_map: dict) -> LLMError:
    """把Anthropic SDK原生异常映射为LLM模块自有分层异常，与OpenAI异常分类逻辑完全对齐。"""
    if isinstance(e, errors_map["APITimeoutError"]):
        return LLMTimeoutError(f"LLM 请求超时：{e}", original_error=e)
    if isinstance(e, errors_map["APIConnectionError"]):
        return LLMConnectionError(f"LLM 连接失败：{e}", original_error=e)
    if isinstance(e, errors_map["RateLimitError"]):
        return LLMRateLimitError(f"LLM 触发限流：{e}", original_error=e)
    if isinstance(e, (errors_map["AuthenticationError"], errors_map["PermissionDeniedError"])):
        return LLMAuthenticationError(f"LLM 鉴权失败（API Key 无效或权限不足）：{e}", original_error=e)
    if isinstance(e, errors_map["NotFoundError"]):
        return LLMModelNotFoundError(f"LLM 模型或端点不存在：{e}", original_error=e)
    if isinstance(e, errors_map["BadRequestError"]):
        return LLMBadRequestError(f"LLM 请求参数非法：{e}", original_error=e)
    if isinstance(e, errors_map["APIStatusError"]):
        status_code = getattr(e, "status_code", 0) or 0
        if 500 <= status_code < 600:
            return LLMServerError(f"LLM 服务端错误（{status_code}）：{e}", original_error=e)
        return LLMApiError(f"LLM API 返回错误（{status_code}）：{e}", original_error=e)
    if isinstance(e, errors_map["APIError"]):
        return LLMApiError(f"LLM API 错误：{e}", original_error=e)
    return LLMApiError(f"LLM 调用异常：{e}", original_error=e)


def _convert_messages_to_anthropic(messages: list[dict[str, Any]], system_prompt: Optional[str] = None):
    """把OpenAI格式messages转换为Anthropic格式，抽离system prompt，保留多模态内容。"""
    anthropic_messages = []
    extracted_system = system_prompt
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            extracted_system = content if isinstance(content, str) else str(content)
            continue
        # 角色映射：Anthropic仅支持user/assistant角色，tool role映射为user
        if role == "tool":
            role = "user"
        anthropic_messages.append({"role": role, "content": content})
    return anthropic_messages, extracted_system


def _convert_anthropic_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """把OpenAI格式tools转换为Anthropic格式。"""
    if not tools:
        return None
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        anthropic_tools.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func["parameters"],
        })
    return anthropic_tools or None


def _convert_anthropic_tool_calls(tool_calls) -> list[MsgToolCall]:
    """把Anthropic返回的tool_use块转换为统一的MsgToolCall结构。"""
    result = []
    for tc in tool_calls or []:
        if getattr(tc, "type", "") != "tool_use":
            continue
        import json
        result.append(MsgToolCall(
            id=getattr(tc, "id", ""),
            name=getattr(tc, "name", ""),
            arguments=json.dumps(getattr(tc, "input", {})),
        ))
    return result


class AnthropicClient:
    """Anthropic原生客户端，接口与OpenAIClient完全对齐，上层调用无感。

    方法签名、返回结构、异常处理、usage回填与OpenAIClient完全一致：
    - invoke / stream_invoke / invoke_structured / embed_query / embed_documents
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
        anthropic, _ = _import_anthropic()
        self._anthropic = anthropic
        self._errors_map = _import_anthropic()[1]

        # 配置层错误独立成LLMConfigError
        try:
            client_kwargs = {
                "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
                "timeout": timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = anthropic.Anthropic(**client_kwargs)
        except Exception as e:
            raise LLMConfigError(
                f"Anthropic 客户端初始化失败（请检查 ANTHROPIC_API_KEY 配置）：{e}",
                original_error=e,
            ) from e

        # 默认模型：claude-3-5-sonnet-latest
        self.model = model or os.getenv("LLM_MODEL_ID") or "claude-3-5-sonnet-latest"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 4096)  # Anthropic默认max_tokens更高
        # Anthropic当前无官方embedding API，默认使用Voyage Code 3（最常用的Anthropic生态embedding）
        self.embedding_model = (
            embedding_model
            or os.getenv("LLM_EMBED_MODEL")
            or "voyage-3"
        )
        self.extra_body = dict(extra_body) if extra_body else None

    def _resolve_extra_body(self, kwargs: dict) -> Optional[dict]:
        """解析额外请求参数，与OpenAIClient逻辑一致。"""
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

    def invoke(self, messages: list[dict[str, Any]], tools=None, *, usage_sink: Optional[dict] = None, **kwargs) -> Message:
        """非流式调用Anthropic模型，返回Message对象，接口与OpenAIClient完全对齐。"""
        try:
            chat_model = kwargs.pop("model", self.model)
            extra_body = self._resolve_extra_body(kwargs)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # 转换消息格式
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

            response = self._client.messages.create(**create_kwargs)

            # 回填usage（Anthropic字段名：input_tokens/output_tokens → 映射为OpenAI格式）
            if usage_sink is not None:
                usage = {
                    "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                    "completion_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
                }
                usage_sink.update(usage)

            # 解析响应内容
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
                f"LLM 调用失败：{mapped}",
                original_error=e,
            ) from mapped

    def stream_invoke(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> Iterator[StreamChunk]:
        """流式调用Anthropic模型，逐段yield StreamChunk，接口与OpenAIClient完全对齐。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        response = None
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

            response = self._client.messages.create(**create_kwargs)
            print("✅ 大语言模型响应成功:")

            think_print_flg = False
            text_buf: list[str] = []
            tool_calls_builder: dict[int, dict] = {}
            finish_reason: str = ""
            current_tool_idx = 0

            for event in response:
                event_type = getattr(event, "type", "")
                # 消息结束事件
                if event_type == "message_stop":
                    break
                # 内容块开始
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
                # 文本增量
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if not delta:
                        continue
                    delta_type = getattr(delta, "type", "")
                    # 思考过程（Anthropic thinking扩展）
                    if delta_type == "thinking_delta":
                        thinking_text = getattr(delta, "thinking", "")
                        if thinking_text:
                            if not think_print_flg:
                                print("[思考]")
                                think_print_flg = True
                            print(thinking_text, end="", flush=True)
                        continue
                    # 正文文本
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            text_buf.append(text)
                            yield StreamChunk(type="text", text=text)
                    # 工具参数增量
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
                # 消息结束，记录finish_reason
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        finish_reason = getattr(delta, "stop_reason", "") or ""

            print()

            # 构造end片
            full_content = "".join(text_buf)
            final_tool_calls: list[MsgToolCall] = []
            import json
            for idx in sorted(tool_calls_builder.keys()):
                tc = tool_calls_builder[idx]
                # 补全参数JSON的闭合括号（防止截断）
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
            print(f"❌ 调用LLM API时发生错误: {e}")
            mapped = _classify_anthropic_error(e, self._errors_map)
            raise LLMInvokeError(
                f"LLM 流式调用失败：{mapped}",
                original_error=e,
            ) from mapped

    # ------------------------------------------------------------------ #
    # Structured Output（结构化输出）                                        #
    # ------------------------------------------------------------------ #

    def invoke_structured(
        self,
        messages: list[dict[str, Any]],
        response_format: type,
        *,
        strict: bool = False,
        usage_sink: Optional[dict] = None,
        **kwargs,
    ):
        """结构化调用，通过prompt注入JSON Schema约束输出，返回pydantic模型实例。

        兼容Anthropic无原生response_format参数的场景，通过系统提示词约束输出为JSON。
        """
        import json as _json

        # pydantic校验
        try:
            from pydantic import BaseModel as _PydanticBaseModel
        except ImportError:
            raise TypeError("invoke_structured 需要 pydantic 依赖（项目已内置）")
        if not isinstance(response_format, type) or not issubclass(response_format, _PydanticBaseModel):
            raise TypeError(
                f"response_format 必须是 pydantic.BaseModel 子类，收到 {response_format!r}"
            )

        # 生成JSON Schema
        try:
            schema_dict = response_format.model_json_schema()
        except Exception as e:
            raise LLMStructuredOutputError(
                f"无法从 {response_format.__name__} 生成 JSON Schema：{e}",
                original_error=e,
            ) from e

        # 注入结构化输出提示
        schema_prompt = f"""
请严格按照以下JSON Schema输出结果，不要输出任何额外解释、Markdown标记或代码块，只返回纯JSON：
{_json.dumps(schema_dict, ensure_ascii=False, indent=2)}
"""
        # 把结构化提示追加到最后一条user消息
        structured_messages = list(messages)
        if structured_messages and structured_messages[-1]["role"] == "user":
            original_content = structured_messages[-1]["content"]
            structured_messages[-1]["content"] = f"{original_content}\n\n{schema_prompt}"
        else:
            structured_messages.append({"role": "user", "content": schema_prompt})

        try:
            response = self.invoke(structured_messages, usage_sink=usage_sink, **kwargs)
            content = (response.content or "").strip()
            # 提取JSON部分（处理模型可能返回的```json ... ```包裹）
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
                f"结构化输出调用失败：{e}",
                original_error=e,
            ) from e

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）                                               #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str, *, usage_sink: Optional[dict] = None, **kwargs) -> list[float]:
        """单条Embedding调用，当前默认使用Voyage AI端点（Anthropic生态）。"""
        if not isinstance(text, str):
            raise TypeError(f"embed_query 要求 text 为 str，收到 {type(text).__name__}")
        if not text.strip():
            raise ValueError("embed_query 收到空字符串，无法生成 embedding")

        try:
            # Anthropic无原生embedding API，默认走Voyage AI（OpenAI兼容协议）
            from openai import OpenAI
            voyage_client = OpenAI(
                api_key=os.getenv("VOYAGE_API_KEY", self.api_key),
                base_url="https://api.voyageai.com/v1",
            )
            embed_model = kwargs.pop("model", self.embedding_model)
            resp = voyage_client.embeddings.create(
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
                f"Embedding 调用失败（query）：{mapped}",
                original_error=e,
            ) from mapped

    def embed_documents(self, texts: list[str], *, usage_sink: Optional[dict] = None, **kwargs) -> list[list[float]]:
        """批量Embedding调用，自动分片并发，与OpenAIClient逻辑一致。"""
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"embed_documents 要求 texts 为 list[str]，收到 {type(texts).__name__}"
            )
        cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        if not cleaned:
            return []

        embed_model = kwargs.pop("model", self.embedding_model)

        def _one_batch(batch: list[str]):
            from openai import OpenAI
            voyage_client = OpenAI(
                api_key=os.getenv("VOYAGE_API_KEY", self.api_key),
                base_url="https://api.voyageai.com/v1",
            )
            resp = voyage_client.embeddings.create(
                model=embed_model,
                input=batch,
                **kwargs,
            )
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors = [list(d.embedding) for d in ordered]
            return vectors, _dump_usage(getattr(resp, "usage", None))

        batches = [
            cleaned[i : i + min(128, _EMBED_BATCH_SIZE)]  # Voyage单次最大128条
            for i in range(0, len(cleaned), min(128, _EMBED_BATCH_SIZE))
        ]

        try:
            from openai import OpenAI
            if len(batches) == 1:
                vectors, usage = _one_batch(batches[0])
                _merge_usage(usage_sink, usage)
                return vectors
            merged: list[list[float]] = []
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
                for part, usage in pool.map(_one_batch, batches):
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
                f"Embedding 调用失败（documents）：{mapped}",
                original_error=e,
            ) from mapped