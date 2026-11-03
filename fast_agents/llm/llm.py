import os
from typing import Optional, Iterator, TYPE_CHECKING

from .openai_client import OpenAIClient
from ..messages.message import Message

if TYPE_CHECKING:
    from ..trace import TraceLogger


class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: Optional[int] = None,
        embedding_model: Optional[str] = None,
        tracer: Optional["TraceLogger"] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("LLM_EMBED_MODEL")
        self.timeout = timeout
        self.provider = provider
        self.tracer = tracer  # G1: 可选 tracer；None 表示不记录，行为与改动前完全一致
        self._client = self._create_client()

    def _create_client(self):
        """创建底层 client，目前默认走 OpenAI 兼容协议。"""
        if self.provider == "openai":
            return OpenAIClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                embedding_model=self.embedding_model,
            )
        else:
            # 其他 provider 也暂走 OpenAI 兼容协议（绝大多数国内中转都兼容）；
            # 将来需要 Anthropic / Gemini 直连时，再在此处加分支。
            return OpenAIClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                embedding_model=self.embedding_model,
            )

    # ------------------------------------------------------------------ #
    # Chat（对话）                                                         #
    # ------------------------------------------------------------------ #

    def invoke(self, messages: list[dict[str, str]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> Message:
        """非流式对话调用，返回 Message 对象。

        可选参数 ``tracer`` 非 None 时会记录一条 ``llm.invoke`` Span；
        未传时退回到 ``self.tracer``，两者都为 None 则不打点，行为与改动前完全一致。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            return self._client.invoke(messages=messages, tools=tools, **kwargs)
        # 打点：记录入参（messages 截断，避免日志过大）
        from ..trace import SpanKind
        # 先从 kwargs 里 pop tracer 相关，避免传给底层 openai.create
        _ = kwargs.pop("tracer", None)
        input_snapshot = {
            "model": kwargs.get("model", self.model),
            "messages_count": len(messages) if messages else 0,
            "tools_count": len(tools) if tools else 0,
        }
        with effective_tracer.span(
            name="llm.invoke",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": kwargs.get("model", self.model), "has_tools": bool(tools)},
        ) as s:
            response = self._client.invoke(messages=messages, tools=tools, **kwargs)
            s.output = {
                "role": response.role,
                "has_content": bool(response.content),
                "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
                "content_preview": (response.content[:200] if response.content else None),
            }
            return response

    def stream_invoke(self, messages: list[dict[str, str]], tools=None, tracer: Optional["TraceLogger"] = None, **kwargs) -> Iterator[str]:
        """流式对话调用，逐段 yield 文本片段。

        可选参数 ``tracer`` 非 None 时会记录一条 ``llm.stream_invoke`` Span（在生成器耗尽/异常时 end）。
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            return self._client.stream_invoke(messages=messages, tools=tools, **kwargs)
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
        chunks: list[str] = []
        try:
            gen = self._client.stream_invoke(messages=messages, tools=tools, **kwargs)
            for chunk in gen:
                if chunk:
                    chunks.append(chunk)
                yield chunk
        except Exception as e:
            effective_tracer.end_span(
                span,
                output={"chunks": len(chunks), "content_preview": "".join(chunks)[:200]},
                error=f"{type(e).__name__}: {e}",
            )
            raise
        else:
            effective_tracer.end_span(
                span,
                output={
                    "chunks": len(chunks),
                    "content_preview": "".join(chunks)[:200],
                    "length": sum(len(c) for c in chunks),
                },
            )

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

        Examples::

            llm = LLM()
            vec = llm.embed_query("今天天气怎么样")
            # len(vec) == 1536
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            return self._client.embed_query(text, **kwargs)
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        input_snapshot = {"text_length": len(text) if isinstance(text, str) else 0, "model": kwargs.get("model", self.embedding_model)}
        with effective_tracer.span(
            name="llm.embed_query",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": kwargs.get("model", self.embedding_model), "op": "embed_query"},
        ) as s:
            vec = self._client.embed_query(text, **kwargs)
            s.output = {"dim": len(vec) if vec else 0}
            return vec

    def embed_documents(self, texts: list[str], tracer: Optional["TraceLogger"] = None, **kwargs) -> list[list[float]]:
        """
        对一批「文档侧」文本批量生成 embedding 向量。

        典型用途：向知识库写入文档（已分块）时，把每个 chunk 向量化后存入库。
        返回值：二维浮点列表，顺序与输入 texts 一一对应。
        内部自动按 2048 条分片并并发请求，调用方无需关心 API 批次上限。

        Examples::

            llm = LLM()
            vecs = llm.embed_documents(["段落一", "段落二", "段落三"])
            # len(vecs) == 3, len(vecs[0]) == 1536
        """
        effective_tracer = tracer or self.tracer
        if effective_tracer is None:
            return self._client.embed_documents(texts, **kwargs)
        from ..trace import SpanKind
        _ = kwargs.pop("tracer", None)
        input_snapshot = {"batch_size": len(texts) if texts else 0, "model": kwargs.get("model", self.embedding_model)}
        with effective_tracer.span(
            name="llm.embed_documents",
            kind=SpanKind.LLM,
            input=input_snapshot,
            metadata={"model": kwargs.get("model", self.embedding_model), "op": "embed_documents"},
        ) as s:
            vecs = self._client.embed_documents(texts, **kwargs)
            s.output = {"batch": len(vecs) if vecs else 0, "dim": len(vecs[0]) if vecs and vecs[0] else 0}
            return vecs
