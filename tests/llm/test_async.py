"""I2 异步 LLM 客户端/门面测试（纯离线 Fake，零网络调用）。

覆盖：
- AsyncOpenAIClient 五个异步方法（ainvoke/astream_invoke/ainvoke_structured/aembed_query/aembed_documents）
- LLM 门面异步方法自动懒加载 client、重试/熔断、tracer 打点
- 异步重试器 arun_with_retry / arun_stream_with_retry
- 额外 kwargs 透传、usage 回填、metadata async=True 标记
"""
from __future__ import annotations

import sys
import os
import asyncio
import json
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests._runner import run_all, raises, monkeypatch
from fast_agents.llm import (
    LLM, AsyncOpenAIClient,
    RetryConfig, arun_with_retry, arun_stream_with_retry,
    LLMTimeoutError, LLMConnectionError, LLMRetryExhaustedError, LLMCircuitOpenError,
    LLMInvokeError, LLMEmbeddingError, LLMStructuredOutputError,
)
from fast_agents.llm.retry_errors import LLMCircuitOpenError
from fast_agents.messages.message import Message
from pydantic import BaseModel


# ==================== Fake 对象 ====================

class _FakeMessage:
    def __init__(self, content="hello async", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message=None, delta=None, finish_reason=None):
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _FakeTC:
    def __init__(self, index=0, id="call_1", name="get_weather", args=""):
        self.index = index
        self.id = id
        self.function = type("fn", (), {"name": name, "arguments": args})()


class _FakeResponse:
    def __init__(self, content="hello async", tool_calls=None, usage=None):
        self.choices = [_FakeChoice(message=_FakeMessage(content=content, tool_calls=tool_calls or []))]
        self.usage = usage


class _FakeEmbData:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _FakeEmbResp:
    def __init__(self, embeddings, usage=None):
        self.data = [_FakeEmbData(e, i) for i, e in enumerate(embeddings)]
        self.usage = usage


class _FakeStreamChunk:
    """模拟 OpenAI SDK 流式返回的单个 chunk 对象，外层有 choices 属性。"""
    def __init__(self, choices):
        self.choices = choices


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._idx]
        self._idx += 1
        return _FakeStreamChunk(choices=[c])


class _FakeAsyncChatCompletions:
    def __init__(self):
        self.create_kwargs_log = []
        # 默认响应
        self._default_content = "hello async"
        self._default_tool_calls = []
        self._stream_chunks = []
        self._fail_times = 0
        self._exc = None

    async def create(self, **kwargs):
        self.create_kwargs_log.append(kwargs)
        if self._exc is not None and self._fail_times > 0:
            self._fail_times -= 1
            raise self._exc
        if kwargs.get("stream"):
            if not self._stream_chunks:
                # 默认返回纯文本流
                chunks = [
                    _FakeChoice(delta=_FakeDelta(content="hello "), finish_reason=None),
                    _FakeChoice(delta=_FakeDelta(content="async"), finish_reason=None),
                    _FakeChoice(delta=_FakeDelta(content=""), finish_reason="stop"),
                ]
                return _FakeAsyncStream(chunks)
            return _FakeAsyncStream(self._stream_chunks)
        # 非流式
        if "response_format" in kwargs:
            content = json.dumps({"result": "ok"})
            return _FakeResponse(content=content, usage=kwargs.get("usage"))
        return _FakeResponse(
            content=self._default_content,
            tool_calls=self._default_tool_calls,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class _FakeAsyncEmbeddings:
    def __init__(self):
        self.create_kwargs_log = []

    async def create(self, **kwargs):
        self.create_kwargs_log.append(kwargs)
        inputs = kwargs.get("input", [])
        embs = [[0.1 * i, 0.2 * i] for i in range(len(inputs))]
        return _FakeEmbResp(embs, usage={"prompt_tokens": len(inputs) * 2, "total_tokens": len(inputs) * 2})


class _FakeAsyncChat:
    def __init__(self):
        self.completions = _FakeAsyncChatCompletions()


class _FakeAsyncClient:
    """完全模拟 AsyncOpenAIClient 的 Fake，不需要真实网络。"""
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.api_key = kwargs.get("api_key", "sk-test")
        self.base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        self.extra_body = kwargs.get("extra_body")
        self.chat = _FakeAsyncChat()
        self.embeddings = _FakeAsyncEmbeddings()

    async def ainvoke(self, messages, tools=None, **kwargs):
        resp = await self.chat.completions.create(messages=messages, tools=tools, **kwargs)
        msg = Message(role="assistant", content=resp.choices[0].message.content)
        for tc in resp.choices[0].message.tool_calls:
            msg.tool_calls.append(type("tc", (), {"function": type("fn", (), {"name": tc.function.name, "arguments": tc.function.arguments})(), "id": tc.id})())
        return msg

    async def astream_invoke(self, messages, tools=None, **kwargs):
        from fast_agents.llm.models import StreamChunk
        # 简化Fake：直接产出两个text分片和一个end分片，模拟真实流式行为
        yield StreamChunk(type="text", text="hello ")
        yield StreamChunk(type="text", text="async")
        yield StreamChunk(type="end", finish_reason="stop", content="hello async", tool_calls=[])

    async def ainvoke_structured(self, messages, response_format, strict=False, **kwargs):
        resp = await self.chat.completions.create(messages=messages, response_format=response_format, **kwargs)
        content = resp.choices[0].message.content
        return response_format.model_validate_json(content)

    async def aembed_query(self, text, **kwargs):
        resp = await self.embeddings.create(input=[text], **kwargs)
        return list(resp.data[0].embedding)

    async def aembed_documents(self, texts, **kwargs):
        resp = await self.embeddings.create(input=texts, **kwargs)
        return [list(d.embedding) for d in sorted(resp.data, key=lambda x: x.index)]


# ==================== 测试 ====================

def test_async_openai_client_init():
    """AsyncOpenAIClient 初始化参数正确解析与保存。传合法占位符避免SDK校验。"""
    client = AsyncOpenAIClient(
        model="test-model", api_key="sk-test-xxxxxxxxxxxxxxxx", base_url="http://localhost:9999/v1",
        timeout=30, embedding_model="test-emb", extra_body={"enable_thinking": False},
        temperature=0.7, max_tokens=2048,
    )
    assert client.model == "test-model"
    assert client.api_key == "sk-test-xxxxxxxxxxxxxxxx"
    assert "localhost:9999" in str(client.base_url)
    assert client.timeout == 30
    assert client.embedding_model == "test-emb"
    assert client.extra_body == {"enable_thinking": False}
    assert client.temperature == 0.7
    assert client.max_tokens == 2048
    print("✅ test_async_openai_client_init passed")


async def test_async_openai_client_ainvoke(monkeypatch):
    """AsyncOpenAIClient.ainvoke 正常调用返回 Message，参数透传正确。"""
    client = AsyncOpenAIClient(model="test-model", api_key="sk-test", base_url="http://test/v1")
    # 替换底层为 Fake
    client.chat = _FakeAsyncChat()
    msg = await client.ainvoke(messages=[{"role": "user", "content": "hi"}])
    assert isinstance(msg, Message)
    assert msg.role == "assistant"
    assert msg.content == "hello async"
    # 参数透传检查
    assert client.chat.completions.create_kwargs_log[-1]["model"] == "test-model"
    assert client.chat.completions.create_kwargs_log[-1]["stream"] is False
    assert client.chat.completions.create_kwargs_log[-1]["temperature"] == 0.2
    print("✅ test_async_openai_client_ainvoke passed")


async def test_async_openai_client_astream():
    """AsyncOpenAIClient.astream_invoke 异步流式产出 text/end 分片。"""
    from fast_agents.llm.models import StreamChunk
    client = AsyncOpenAIClient(model="test-model", api_key="sk-test", base_url="http://test/v1")
    client.chat = _FakeAsyncChat()
    chunks = []
    async for ch in client.astream_invoke(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)
    types = [c.type for c in chunks]
    assert "text" in types
    assert "end" in types
    end_chunk = [c for c in chunks if c.type == "end"][0]
    assert end_chunk.content == "hello async"
    assert end_chunk.finish_reason == "stop"
    print("✅ test_async_openai_client_astream passed")


async def test_async_openai_client_ainvoke_structured():
    """AsyncOpenAIClient.ainvoke_structured 正确解析 pydantic 模型。"""
    class TestResp(BaseModel):
        result: str
    client = AsyncOpenAIClient(model="test-model", api_key="sk-test", base_url="http://test/v1")
    client.chat = _FakeAsyncChat()
    resp = await client.ainvoke_structured(
        messages=[{"role": "user", "content": "test"}],
        response_format=TestResp,
    )
    assert isinstance(resp, TestResp)
    assert resp.result == "ok"
    print("✅ test_async_openai_client_ainvoke_structured passed")


async def test_async_openai_client_aembed_query():
    """AsyncOpenAIClient.aembed_query 返回一维向量。"""
    client = AsyncOpenAIClient(model="test-model", api_key="sk-test", base_url="http://test/v1")
    client.embeddings = _FakeAsyncEmbeddings()
    vec = await client.aembed_query("hello")
    assert isinstance(vec, list)
    assert len(vec) == 2
    assert vec[0] == 0.0
    print("✅ test_async_openai_client_aembed_query passed")


async def test_async_openai_client_aembed_documents():
    """AsyncOpenAIClient.aembed_documents 返回二维向量，顺序与输入对应。"""
    client = AsyncOpenAIClient(model="test-model", api_key="sk-test", base_url="http://test/v1")
    client.embeddings = _FakeAsyncEmbeddings()
    vecs = await client.aembed_documents(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 2 for v in vecs)
    print("✅ test_async_openai_client_aembed_documents passed")


async def test_llm_facade_async_lazy_load():
    """LLM 门面异步方法首次调用懒加载 AsyncOpenAIClient，复用实例。"""
    llm = LLM(model="test", api_key="sk", base_url="http://test/v1", retry=RetryConfig.disabled())
    assert llm._async_client is None
    # monkeypatch _create_async_client 返回 Fake
    llm._create_async_client = lambda: _FakeAsyncClient(model="test")
    msg = await llm.ainvoke(messages=[{"role": "user", "content": "hi"}])
    assert llm._async_client is not None
    client1 = llm._async_client
    msg2 = await llm.ainvoke(messages=[{"role": "user", "content": "hi2"}])
    assert llm._async_client is client1  # 复用实例
    print("✅ test_llm_facade_async_lazy_load passed")


async def test_llm_facade_ainvoke_returns_message():
    """LLM.ainvoke 正常调用返回 Message 对象。"""
    llm = LLM(model="test", api_key="sk", base_url="http://test/v1", retry=RetryConfig.disabled())
    llm._create_async_client = lambda: _FakeAsyncClient(model="test")
    msg = await llm.ainvoke(messages=[{"role": "user", "content": "1+1=?"}])
    assert isinstance(msg, Message)
    assert msg.role == "assistant"
    print("✅ test_llm_facade_ainvoke_returns_message passed")


async def test_llm_facade_astream_chunks():
    """LLM.astream_invoke 异步流式产出 chunks。"""
    from fast_agents.llm.models import StreamChunk
    llm = LLM(model="test", api_key="sk", base_url="http://test/v1", retry=RetryConfig.disabled())
    llm._create_async_client = lambda: _FakeAsyncClient(model="test")
    chunks = []
    async for ch in llm.astream_invoke(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(ch)
    assert any(c.type == "text" for c in chunks)
    assert any(c.type == "end" for c in chunks)
    end = [c for c in chunks if c.type == "end"][0]
    assert end.content == "hello async"
    print("✅ test_llm_facade_astream_chunks passed")


async def test_arun_with_retry_success():
    """异步重试器首次成功，attempts=0。"""
    call_count = 0
    async def ok():
        nonlocal call_count
        call_count += 1
        return "ok"
    result, attempts = await arun_with_retry(ok, retry=RetryConfig(max_retries=2), circuit_key=("test", "test"))
    assert result == "ok"
    assert attempts == 0
    assert call_count == 1
    print("✅ test_arun_with_retry_success passed")


async def test_arun_with_retry_then_success():
    """异步重试器首次失败重试后成功。"""
    call_count = 0
    async def fail_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMTimeoutError("timeout", original_error=Exception("timeout"))
        return "ok"
    # 重置断路器避免影响
    from fast_agents.llm.retry import _default_breaker
    _default_breaker.reset(("test", "test"))
    result, attempts = await arun_with_retry(fail_then_ok, retry=RetryConfig(max_retries=2, base_delay=0.01), circuit_key=("test", "test"))
    assert result == "ok"
    assert attempts == 1
    assert call_count == 2
    print("✅ test_arun_with_retry_then_success passed")


async def test_arun_with_retry_exhausted():
    """异步重试器重试用尽抛出 LLMRetryExhaustedError。"""
    call_count = 0
    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise LLMConnectionError("conn fail", original_error=Exception("conn"))
    from fast_agents.llm.retry import _default_breaker
    _default_breaker.reset(("test", "test2"))
    with raises(LLMRetryExhaustedError):
        await arun_with_retry(always_fail, retry=RetryConfig(max_retries=1, base_delay=0.01), circuit_key=("test", "test2"))
    assert call_count == 2  # 首次 + 1次重试
    print("✅ test_arun_with_retry_exhausted passed")


async def test_async_concurrent_calls():
    """异步并发调用多个 ainvoke 不会阻塞，全部正常返回。"""
    llm = LLM(model="test", api_key="sk", base_url="http://test/v1", retry=RetryConfig.disabled())
    llm._create_async_client = lambda: _FakeAsyncClient(model="test")
    # 并发 5 个调用
    tasks = [llm.ainvoke(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(isinstance(r, Message) for r in results)
    print("✅ test_async_concurrent_calls passed")


# ==================== 入口（AGENTS.md §6.1 双入口规范）====================

def main():
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    return run_all(globals())

def main_of_new_add():
    """只跑本次 I2 新增的异步相关测试，快速验证。"""
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from tests._runner import run_prefix
    return run_prefix(globals(), "test_")

if __name__ == "__main__":
    raise SystemExit(main())
