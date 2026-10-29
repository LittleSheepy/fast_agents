import os
from typing import Optional, Iterator

from .openai_client import OpenAIClient
from ..messages.message import Message


class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("LLM_EMBED_MODEL")
        self.timeout = timeout
        self.provider = provider
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

    def invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Message:
        """非流式对话调用，返回 Message 对象。"""
        return self._client.invoke(messages=messages, tools=tools, **kwargs)

    def stream_invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Iterator[str]:
        """流式对话调用，逐段 yield 文本片段。"""
        return self._client.stream_invoke(messages=messages, tools=tools, **kwargs)

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）—— 语义检索的前置能力                            #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str, **kwargs) -> list[float]:
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
        return self._client.embed_query(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
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
        return self._client.embed_documents(texts, **kwargs)
