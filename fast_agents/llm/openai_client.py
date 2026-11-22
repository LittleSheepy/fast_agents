import os
from typing import Optional, Iterator, Any
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
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


# OpenAI Embeddings API 单次请求允许的最大 input 条数（官方上限 2048）
_EMBED_BATCH_SIZE = 2048


def _dump_usage(usage) -> Optional[dict]:
    """把 SDK 的 usage 对象转成普通 dict；拿不到时返回 None。"""
    if usage is None:
        return None
    try:
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "to_dict"):
            return usage.to_dict()
        if isinstance(usage, dict):
            return dict(usage)
        return dict(vars(usage))
    except Exception:
        return None


def _merge_usage(sink: Optional[dict], usage: Optional[dict]) -> None:
    """把 usage 明细累加进 sink（用于 embed_documents 多批次汇总）。

    仅累加数值型字段（prompt_tokens / total_tokens 等），非数值字段跳过，
    不引入任何新依赖。
    """
    if sink is None or not usage:
        return
    for key, val in usage.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        sink[key] = sink.get(key, 0) + val


def _classify_openai_error(e: Exception) -> LLMError:
    """把 OpenAI SDK 原生异常映射成 LLM 模块自有分层异常。

    - 保留原始异常链（``raise ... from e`` 由调用方负责）；
    - 每个子类带 ``retryable`` 属性，直接供 I4（重试 / 熔断）判断；
    - 未命中已知类型的兜底为 :class:`LLMApiError`。
    """
    if isinstance(e, APITimeoutError):
        return LLMTimeoutError(f"LLM 请求超时：{e}", original_error=e)
    if isinstance(e, APIConnectionError):
        return LLMConnectionError(f"LLM 连接失败：{e}", original_error=e)
    if isinstance(e, RateLimitError):
        return LLMRateLimitError(f"LLM 触发限流：{e}", original_error=e)
    if isinstance(e, (AuthenticationError, PermissionDeniedError)):
        return LLMAuthenticationError(f"LLM 鉴权失败（API Key 无效或权限不足）：{e}", original_error=e)
    if isinstance(e, NotFoundError):
        # 404 在 LLM 场景几乎都是模型不存在
        return LLMModelNotFoundError(f"LLM 模型或端点不存在：{e}", original_error=e)
    if isinstance(e, BadRequestError):
        return LLMBadRequestError(f"LLM 请求参数非法：{e}", original_error=e)
    if isinstance(e, APIStatusError):
        # 5xx 走 LLMServerError（可重试）；其余状态码兜底为 LLMApiError
        if 500 <= (e.status_code or 0) < 600:
            return LLMServerError(f"LLM 服务端错误（{e.status_code}）：{e}", original_error=e)
        return LLMApiError(f"LLM API 返回错误（{e.status_code}）：{e}", original_error=e)
    if isinstance(e, APIError):
        return LLMApiError(f"LLM API 错误：{e}", original_error=e)
    # SDK 之外的异常（如 JSON 解析、本地属性访问失败等）
    return LLMApiError(f"LLM 调用异常：{e}", original_error=e)


class OpenAIClient(OpenAI):
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
        # 配置层错误独立成 LLMConfigError，避免 SDK 初始化失败时吞成裸 Exception
        try:
            super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        except Exception as e:
            raise LLMConfigError(
                f"LLM 客户端初始化失败（请检查 LLM_BASE_URL / LLM_API_KEY 配置）：{e}",
                original_error=e,
            ) from e

        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        # embedding 模型与 chat 模型分开配置，默认走 OpenAI text-embedding-3-small；
        # 兼容通过 LLM_EMBED_MODEL 环境变量指定本地/第三方 embedding 端点
        self.embedding_model = (
            embedding_model
            or os.getenv("LLM_EMBED_MODEL")
            or "text-embedding-3-small"
        )
        # extra_body 默认 None：严格遵循 OpenAI 原生协议，不发送任何私有扩展字段，
        # 避免对接 api.openai.com 或严格校验端点时出现 400 Unrecognized request argument。
        # DashScope/百炼 Qwen 用户可显式传入 extra_body={"enable_thinking": False} 关闭思考模式；
        # 单次调用也可通过 kwargs 传入 extra_body 覆盖实例级配置。
        self.extra_body = dict(extra_body) if extra_body else None

    def _resolve_extra_body(self, kwargs: dict) -> Optional[dict]:
        """解析本次调用要发送的 extra_body：实例级默认与单次调用参数合并（单次优先）。

        两者都为空时返回 None —— 请求体完全不携带 extra_body，保持对原生
        OpenAI 端点的严格兼容（不发送任何 provider 私有字段）。
        """
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
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。

        可选参数 ``usage_sink``：传入一个 dict 时，会把本次响应的 token 用量
        （prompt_tokens/completion_tokens/total_tokens）写入其中，供上层打点使用；
        不传时行为与改动前完全一致（不产生任何额外副作用）。

        异常：底层 OpenAI SDK 异常经 :func:`_classify_openai_error` 映射为
        :mod:`fast_agents.llm.errors` 下的分层异常，不再抛出裸 ``Exception``；
        最外层统一包装为 :class:`LLMInvokeError`，原始异常可通过 ``__cause__`` 访问。
        """
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
            response = self.chat.completions.create(**create_kwargs)

            if usage_sink is not None:
                usage = _dump_usage(getattr(response, "usage", None))
                if usage:
                    usage_sink.update(usage)
            message = response.choices[0].message
            response_msg = Message(role="assistant", content=message.content)
            # 是否带工具调用只看返回体里的 tool_calls，不看 finish_reason：
            # 部分 OpenAI 兼容端点带 tool_calls 时 finish_reason 仍返回 "stop"，
            # 只认 finish_reason == "tool_calls" 会漏掉工具调用，让整条 ReAct 链路空转。
            for tool in (getattr(message, "tool_calls", None) or []):
                msg_tool_call = MsgToolCall(name=tool.function.name, arguments=tool.function.arguments, id=tool.id)
                response_msg.tool_calls.append(msg_tool_call)

            return response_msg
        except LLMError:
            # 自身已抛出的分层异常直接透传，不再二次包装
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMInvokeError(
                f"LLM 调用失败：{mapped}",
                original_error=e,
            ) from mapped

    def stream_invoke(self, messages: list[dict[str, Any]], tools=None, **kwargs) -> Iterator[StreamChunk]:
        """
        调用大语言模型进行思考，并返回流式响应（三态 chunk）。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            StreamChunk: text 正文增量 / tool_call 工具调用分片 / end 聚合结果
            - text：实时输出的文本片段，前端可直接做打字机效果
            - tool_call：按 index 分片透传工具调用增量（id/name/arguments_delta），前端可实时渲染参数拼装
            - end：流结束时产出，携带完整正文 content 与聚合后的 tool_calls 列表，消费方无需自行拼接

        注意：是否走工具分支仅依据 end 片里 tool_calls 是否非空判断，不依赖 finish_reason，
        兼容部分 OpenAI 兼容端点带 tool_calls 时 finish_reason 返回 "stop" 的情况。

        异常：与 :meth:`invoke` 一致，统一走 :mod:`fast_agents.llm.errors` 分层异常。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
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
            response = self.chat.completions.create(**create_kwargs)

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            think_print_flg = False
            text_buf: list[str] = []
            # tool_calls 分片聚合器：按 index 归并，首片取 id/name，后续片累加 arguments
            tool_calls_builder: dict[int, dict] = {}
            finish_reason: str = ""

            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                # 记录 finish_reason（仅透传，不做分支判据）
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                # 先处理思考过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not think_print_flg:
                        print("[思考]")
                        think_print_flg = True
                    print(f"{delta.reasoning_content}", end="", flush=True)
                    continue  # 思考片段不塞进正文流

                # 处理正文增量
                content = getattr(delta, "content", None) or ""
                if content:
                    text_buf.append(content)
                    yield StreamChunk(type="text", text=content)

                # 处理工具调用分片，按 index 归并
                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                for tc_delta in delta_tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_builder:
                        tool_calls_builder[idx] = {
                            "id": getattr(tc_delta, "id", "") or "",
                            "name": getattr(getattr(tc_delta, "function", None), "name", "") or "",
                            "arguments": "",
                        }
                    # 取函数参数增量片段
                    fn = getattr(tc_delta, "function", None)
                    args_delta = getattr(fn, "arguments", "") or ""
                    if args_delta:
                        tool_calls_builder[idx]["arguments"] += args_delta
                    # 透传增量分片给消费方，支持实时渲染参数拼装
                    yield StreamChunk(
                        type="tool_call",
                        index=idx,
                        id=tool_calls_builder[idx]["id"],
                        name=tool_calls_builder[idx]["name"],
                        arguments_delta=args_delta,
                    )

            print()  # 在流式输出结束后换行

            # 构造 end 片：聚合完整正文与工具调用结果
            full_content = "".join(text_buf)
            final_tool_calls: list[MsgToolCall] = []
            # 按 index 升序组装最终工具调用列表，保证顺序稳定
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
            print(f"❌ 调用LLM API时发生错误: {e}")
            mapped = _classify_openai_error(e)
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
        """非流式结构化调用，返回 pydantic 模型实例。

        通过 OpenAI ``response_format`` 协议（``json_schema`` 模式）约束模型输出，
        再把 JSON 解析为指定的 pydantic 模型实例。

        参数 ``response_format`` 必须是 ``pydantic.BaseModel`` 子类；内部通过
        ``model.model_json_schema()`` 自动转 JSON Schema，调用方无需手动构造。

        ``strict=False``（默认）时允许模型输出 schema 未定义的额外字段（宽容模式），
        兼容绝大多数 OpenAI 兼容端点；``strict=True`` 时开启严格约束（要求端点支持
        Structured Outputs），不兼容的端点会返回 400。

        可选参数 ``usage_sink``：与 invoke 同名参数语义一致，回填 token 用量。

        异常：JSON 解析失败 / pydantic 校验不通过 / 响应内容为空均抛
        :class:`LLMStructuredOutputError`；网络 / API 层错误走既有
        :mod:`fast_agents.llm.errors` 分层异常。
        """
        import json as _json

        # ---- 入参校验（编程错误不纳入 LLMError 体系） ----
        try:
            from pydantic import BaseModel as _PydanticBaseModel
        except ImportError:
            raise TypeError("invoke_structured 需要 pydantic 依赖（项目已内置）")
        if not isinstance(response_format, type) or not issubclass(response_format, _PydanticBaseModel):
            raise TypeError(
                f"response_format 必须是 pydantic.BaseModel 子类，收到 {response_format!r}"
            )

        # ---- 构造 JSON Schema ----
        try:
            schema_dict = response_format.model_json_schema()
        except Exception as e:
            raise LLMStructuredOutputError(
                f"无法从 {response_format.__name__} 生成 JSON Schema：{e}",
                original_error=e,
            ) from e

        # ---- 发起请求 ----
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
            response = self.chat.completions.create(**create_kwargs)

            # ---- 回填 usage ----
            if usage_sink is not None:
                usage = _dump_usage(getattr(response, "usage", None))
                if usage:
                    usage_sink.update(usage)

            # ---- 提取并解析 JSON ----
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

            # ---- pydantic 校验 ----
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
                f"结构化输出调用失败：{mapped}",
                original_error=e,
            ) from mapped

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）                                               #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str, *, usage_sink: Optional[dict] = None, **kwargs) -> list[float]:
        """
        对单条「查询侧」文本生成 embedding 向量。

        用途：Agent 做语义检索时，把用户问题/检索词转成向量。
        返回值是一维 list[float]，维度由模型决定（text-embedding-3-small 默认 1536 维）。

        与 embed_documents 分开成两个方法的理由见 LLM模块编写说明.md §10.1：
        显式区分「查询侧」与「文档侧」，后续接 BGE / E5 / DashScope 等需要
        query/document 不同前缀或 input_type 的模型时，接口无需改动。

        异常：参数校验错误抛原生 ``TypeError`` / ``ValueError``（不纳入 LLMError 体系）；
        API 调用错误统一走 :mod:`fast_agents.llm.errors` 分层异常，最外层包装为
        :class:`LLMEmbeddingError`。
        """
        if not isinstance(text, str):
            raise TypeError(f"embed_query 要求 text 为 str，收到 {type(text).__name__}")
        if not text.strip():
            raise ValueError("embed_query 收到空字符串，无法生成 embedding")

        try:
            embed_model = kwargs.pop("model", self.embedding_model)
            resp = self.embeddings.create(
                model=embed_model,
                input=[text],
                **kwargs,
            )
            if not resp.data or not resp.data[0].embedding:
                raise LLMResponseError("embedding 响应为空或缺少向量数据")
            # embeddings.create 对单条 input 仍返回 data 列表，取第一个
            return list(resp.data[0].embedding)
        except LLMError:
            raise
        except (TypeError, ValueError):
            # 入参校验类错误原样透传，不被 LLMError 吞掉
            raise
        except Exception as e:
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 调用失败（query）：{mapped}",
                original_error=e,
            ) from mapped

    def embed_documents(self, texts: list[str], *, usage_sink: Optional[dict] = None, **kwargs) -> list[list[float]]:
        """
        对一批「文档侧」文本生成 embedding 向量。

        用途：向知识库写入文档（分块后）时，批量把文本转成向量。
        返回值是二维 list[list[float]]，顺序与输入 texts 一一对应。

        为避免超过 OpenAI API 单次 2048 条 input 的上限，内部自动按 2048 分片，
        并通过线程池并发请求，保持顺序一致。

        可选参数 ``usage_sink``：传入 dict 时把各批次 token 用量累加写入其中。

        异常：与 :meth:`embed_query` 一致，API 调用错误最外层包装为 :class:`LLMEmbeddingError`。
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"embed_documents 要求 texts 为 list[str]，收到 {type(texts).__name__}"
            )
        # 过滤空串但保留位置：空串替换成单个空格，保证返回向量数量与输入一一对应
        cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        if not cleaned:
            return []

        embed_model = kwargs.pop("model", self.embedding_model)

        def _one_batch(batch: list[str]):
            resp = self.embeddings.create(
                model=embed_model,
                input=batch,
                **kwargs,
            )
            # API 返回的 data 按 index 排序，用它来对齐顺序
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors = [list(d.embedding) for d in ordered]
            return vectors, _dump_usage(getattr(resp, "usage", None))

        # 按 _EMBED_BATCH_SIZE 切分
        batches = [
            cleaned[i : i + _EMBED_BATCH_SIZE]
            for i in range(0, len(cleaned), _EMBED_BATCH_SIZE)
        ]

        try:
            if len(batches) == 1:
                vectors, usage = _one_batch(batches[0])
                _merge_usage(usage_sink, usage)
                return vectors
            # 多批次并发；token 用量在汇总阶段统一累加，避免多线程写同一 dict 竞态
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
            mapped = _classify_openai_error(e)
            raise LLMEmbeddingError(
                f"Embedding 调用失败（documents）：{mapped}",
                original_error=e,
            ) from mapped


if __name__ == '__main__':
    from dotenv import load_dotenv
    from .models import StreamChunk

    # load_dotenv("D:/00sheepy/.env")
    load_dotenv()  # 自动向上查找 .env，不再硬编码绝对路径
    openai_client = OpenAIClient()

    # ---- chat 流式自测 ----
    print("=== chat 流式 ===")
    result = openai_client.stream_invoke(messages=[{"role": "user", "content": "1+1=?"}])
    for chunk in result:
        if chunk.type == "text":
            print(chunk.text, end="", flush=True)
    print()

    # ---- embedding 自测 ----
    print("=== embed_query ===")
    qv = openai_client.embed_query("今天天气怎么样")
    print(f"维度: {len(qv)}, 前 5 维: {qv[:5]}")

    print("=== embed_documents (3 条) ===")
    dvs = openai_client.embed_documents([
        "今天天气怎么样",
        "明天会下雨吗",
        "帮我查一下快递",
    ])
    print(f"返回条数: {len(dvs)}, 每条维度: {len(dvs[0]) if dvs else 0}")
