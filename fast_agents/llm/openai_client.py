import os
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from ..messages.message import Message, MsgToolCall


# OpenAI Embeddings API 单次请求允许的最大 input 条数（官方上限 2048）
_EMBED_BATCH_SIZE = 2048


class OpenAIClient(OpenAI):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            timeout=180,
            embedding_model: Optional[str] = None,
            **kwargs
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
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

    # ------------------------------------------------------------------ #
    # Chat（对话）                                                         #
    # ------------------------------------------------------------------ #

    def invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Message:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            chat_model = kwargs.pop("model", self.model)
            response = self.chat.completions.create(
                model=chat_model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']},
                extra_body={"enable_thinking": False}
            )
            message = response.choices[0].message
            response_msg = Message(role="assistant", content=message.content)
            if response.choices[0].finish_reason == "tool_calls":
                for tool in message.tool_calls:
                    msg_tool_call = MsgToolCall(name=tool.function.name, arguments=tool.function.arguments, id=tool.id)
                    response_msg.tool_calls.append(msg_tool_call)

            return response_msg
        except Exception as e:
            raise Exception(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            chat_model = kwargs.pop("model", self.model)
            response = self.chat.completions.create(
                model=chat_model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=self.max_tokens,
                stream=True,
                extra_body={"enable_thinking": False}
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            think_print_flg = False
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 先处理思考过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not think_print_flg:
                        print("[思考]")
                        think_print_flg = True
                    print(f"{delta.reasoning_content}", end="", flush=True)
                    continue  # 思考片段不塞进正文流
                print()  # 在流式输出结束后换行

                content = delta.content or ""
                if content:
                    # print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise Exception(f"LLM调用失败: {str(e)}")

    # ------------------------------------------------------------------ #
    # Embeddings（语义向量）                                               #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str, **kwargs) -> list[float]:
        """
        对单条「查询侧」文本生成 embedding 向量。

        用途：Agent 做语义检索时，把用户问题/检索词转成向量。
        返回值是一维 list[float]，维度由模型决定（text-embedding-3-small 默认 1536 维）。

        与 embed_documents 分开成两个方法的理由见 LLM模块编写说明.md §10.1：
        显式区分「查询侧」与「文档侧」，后续接 BGE / E5 / DashScope 等需要
        query/document 不同前缀或 input_type 的模型时，接口无需改动。
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
            # embeddings.create 对单条 input 仍返回 data 列表，取第一个
            return list(resp.data[0].embedding)
        except Exception as e:
            raise Exception(f"Embedding 调用失败（query）: {str(e)}")

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        对一批「文档侧」文本生成 embedding 向量。

        用途：向知识库写入文档（分块后）时，批量把文本转成向量。
        返回值是二维 list[list[float]]，顺序与输入 texts 一一对应。

        为避免超过 OpenAI API 单次 2048 条 input 的上限，内部自动按 2048 分片，
        并通过线程池并发请求，保持顺序一致。
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

        def _one_batch(batch: list[str]) -> list[list[float]]:
            resp = self.embeddings.create(
                model=embed_model,
                input=batch,
                **kwargs,
            )
            # API 返回的 data 按 index 排序，用它来对齐顺序
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [list(d.embedding) for d in ordered]

        # 按 _EMBED_BATCH_SIZE 切分
        batches = [
            cleaned[i : i + _EMBED_BATCH_SIZE]
            for i in range(0, len(cleaned), _EMBED_BATCH_SIZE)
        ]

        try:
            if len(batches) == 1:
                return _one_batch(batches[0])
            # 多批次并发
            results: list[list[list[float]]] = []
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
                for part in pool.map(_one_batch, batches):
                    results.append(part)
            # 按顺序拼接
            merged: list[list[float]] = []
            for part in results:
                merged.extend(part)
            return merged
        except Exception as e:
            raise Exception(f"Embedding 调用失败（documents）: {str(e)}")


if __name__ == '__main__':
    from dotenv import load_dotenv

    # load_dotenv("D:/00sheepy/.env")
    load_dotenv()  # 自动向上查找 .env，不再硬编码绝对路径
    openai_client = OpenAIClient()

    # ---- chat 流式自测 ----
    print("=== chat 流式 ===")
    result = openai_client.stream_invoke(messages=[{"role": "user", "content": "1+1=?"}])
    for content in result:
        print(content, end="", flush=True)
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
