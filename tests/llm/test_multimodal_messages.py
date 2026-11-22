"""LLM 多模态 messages 透传单元测试

聚焦一件事：**messages 里的 content 允许是「文本 + 图片」的多模态片段
（list[dict]），OpenAIClient 与 LLM 门面都必须把它原样透传给底层
chat.completions.create，不得扁平化成字符串或丢片段**。

背景：invoke / stream_invoke / invoke_structured 的签名把 messages 标注成
``list[dict[str, str]]``（每个值都是 str），但多模态消息的 content 实际是
``list[dict]``（如 ``[{"type": "text", ...}, {"type": "image_url", ...}]``）。
运行时 Python 不校验类型标注，多模态一直是能跑的；这些用例先把「多模态 messages
端到端可用、且被逐字节透传」这条事实锁住，避免将来改动把它破坏。

全程离线：用 __new__ 跳过 SDK 初始化，把 chat.completions.create 换成捕获入参
的假实现，不发起任何网络请求、不依赖 API Key。
"""

import contextlib
import io
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel

from fast_agents.llm.openai_client import OpenAIClient
from fast_agents.llm.llm import LLM
from fast_agents.messages.message import Message


# 典型多模态消息：system 纯文本 + user 图文混合
MULTIMODAL_MESSAGES = [
    {"role": "system", "content": "你是图文助手"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    },
]


class _DemoModel(BaseModel):
    """invoke_structured 用的最小 pydantic 模型"""
    x: int = 0


# ==================== 离线假客户端 ====================

def _make_stream_chunk(content):
    """构造一个最小可用的流式 chunk（结构对齐 OpenAI SDK）"""
    delta = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(delta=delta, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


def _make_capturing_client(captured, *, stream=False, content="hi"):
    """造一个不联网的 OpenAIClient：跳过 __init__，create 把入参记进 captured。

    stream=True 时 create 返回 chunk 迭代器；否则返回普通响应体。
    """
    # 只挂调用链真正用到的属性：base_url 在 SDK 里是 property，跳过 __init__ 后
    # 不能直接赋值（setter 依赖 workload_identity），故不设置。
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.temperature = 0.2
    client.max_tokens = 1024
    client.extra_body = None

    def _create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        if stream:
            return iter([_make_stream_chunk("hi")])
        message = types.SimpleNamespace(content=content, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=None)

    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_create)
    )
    return client


def _make_llm(client):
    """造一个走假 client 的 LLM 门面：跳过 __init__，关闭 tracer 与重试"""
    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.embedding_model = "text-embedding-3-small"
    llm.base_url = "http://localhost:8000/v1"
    llm.tracer = None
    llm.retry_config = None
    llm._client = client
    return llm


def _assert_multimodal_forwarded(captured):
    """断言底层 create 收到的 messages 是多模态原文，未被扁平化"""
    forwarded = captured.get("messages")
    assert forwarded is not None, "底层 create 未收到 messages"
    assert forwarded == MULTIMODAL_MESSAGES, f"多模态 messages 未被原样透传：{forwarded!r}"
    user_content = forwarded[1]["content"]
    assert isinstance(user_content, list), (
        f"多模态 content 应保持 list[dict]，实际是 {type(user_content).__name__}"
    )
    assert user_content[0] == {"type": "text", "text": "请描述这张图"}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.png"},
    }


# ==================== OpenAIClient 三个方法 ====================

def test_multimodal_messages_passed_to_invoke_reach_create():
    """【核心】invoke 的多模态 messages 要原样透传到 create"""
    captured = {}
    _make_capturing_client(captured).invoke(MULTIMODAL_MESSAGES)

    _assert_multimodal_forwarded(captured)
    print("✅ test_multimodal_messages_passed_to_invoke_reach_create passed")


def test_multimodal_messages_passed_to_stream_invoke_reach_create():
    """【核心】stream_invoke 的多模态 messages 要原样透传到 create"""
    captured = {}
    client = _make_capturing_client(captured, stream=True)

    with contextlib.redirect_stdout(io.StringIO()):
        list(client.stream_invoke(MULTIMODAL_MESSAGES))

    _assert_multimodal_forwarded(captured)
    assert captured.get("stream") is True
    print("✅ test_multimodal_messages_passed_to_stream_invoke_reach_create passed")


def test_multimodal_messages_passed_to_invoke_structured_reach_create():
    """【核心】invoke_structured 的多模态 messages 要原样透传到 create"""
    captured = {}
    client = _make_capturing_client(captured, content='{"x": 7}')

    result = client.invoke_structured(MULTIMODAL_MESSAGES, _DemoModel)

    assert isinstance(result, _DemoModel) and result.x == 7, "结构化输出应解析为模型实例"
    _assert_multimodal_forwarded(captured)
    print("✅ test_multimodal_messages_passed_to_invoke_structured_reach_create passed")


# ==================== LLM 门面两级 ====================

def test_multimodal_messages_passed_to_llm_facade_invoke():
    """LLM 门面 invoke 透传多模态 messages 到 client"""
    captured = {}
    llm = _make_llm(_make_capturing_client(captured))

    llm.invoke(MULTIMODAL_MESSAGES)

    _assert_multimodal_forwarded(captured)
    print("✅ test_multimodal_messages_passed_to_llm_facade_invoke passed")


def test_multimodal_messages_passed_to_llm_facade_stream_invoke():
    """LLM 门面 stream_invoke 透传多模态 messages 到 client"""
    captured = {}
    llm = _make_llm(_make_capturing_client(captured, stream=True))

    with contextlib.redirect_stdout(io.StringIO()):
        list(llm.stream_invoke(MULTIMODAL_MESSAGES))

    _assert_multimodal_forwarded(captured)
    print("✅ test_multimodal_messages_passed_to_llm_facade_stream_invoke passed")


# ==================== Message 多模态构造与端到端 ====================

def test_message_add_image_builds_openai_multimodal_content():
    """Message.add_image 产出的 OpenAI 消息应是多模态片段列表"""
    msg = Message(role="user", content="请描述这张图")
    msg.add_image("https://example.com/cat.png")

    out = msg.to_openai_message()

    assert isinstance(out, list) and len(out) == 1, f"user 消息应产出 1 条，实际 {out!r}"
    assert out[0]["role"] == "user"
    content = out[0]["content"]
    assert isinstance(content, list), "图文消息的 content 应是多模态片段列表，而不是字符串"
    assert content[0] == {"type": "text", "text": "请描述这张图"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.png"},
    }
    print("✅ test_message_add_image_builds_openai_multimodal_content passed")


def test_message_multimodal_output_forwarded_unchanged_by_client():
    """端到端：Message 的多模态输出直接喂给 client，也要逐字节透传"""
    msg = Message(role="user", content="请看图").add_image("https://example.com/cat.png")
    openai_msgs = msg.to_openai_message()

    captured = {}
    _make_capturing_client(captured).invoke(openai_msgs)

    assert captured.get("messages") == openai_msgs, "Message 产出的多模态消息被改动"
    assert isinstance(captured["messages"][0]["content"], list)
    print("✅ test_message_multimodal_output_forwarded_unchanged_by_client passed")


def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    test_multimodal_messages_passed_to_invoke_reach_create()  # AI新加的测试
    test_multimodal_messages_passed_to_stream_invoke_reach_create()  # AI新加的测试
    test_multimodal_messages_passed_to_invoke_structured_reach_create()  # AI新加的测试
    test_multimodal_messages_passed_to_llm_facade_invoke()  # AI新加的测试
    test_multimodal_messages_passed_to_llm_facade_stream_invoke()  # AI新加的测试
    test_message_add_image_builds_openai_multimodal_content()  # AI新加的测试
    test_message_multimodal_output_forwarded_unchanged_by_client()  # AI新加的测试


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
