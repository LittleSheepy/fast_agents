"""LLM stream_invoke 额外 kwargs 透传单元测试

聚焦一件事：**stream_invoke 必须和 invoke 一样，把 top_p / frequency_penalty /
presence_penalty / stop / response_format 等额外采样参数透传给底层
chat.completions.create**，同时 temperature / max_tokens 仍取实例默认值、
不被单次 kwargs 覆盖。

修复前：OpenAIClient.stream_invoke() 的 create_kwargs 少了
``**{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}``
这一行，而 invoke() 有。于是经 stream_invoke 传入的 top_p / stop /
response_format 等被静默丢弃；由于 Agent 流式链路走的就是 stream_invoke，
这些参数在流式模式下全部失效。

全程离线：用 __new__ 跳过 SDK 初始化，把 chat.completions.create 换成捕获
入参的假实现，不发起任何网络请求、不依赖 API Key。
"""

import contextlib
import io
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.llm.openai_client import OpenAIClient


def _make_stream_chunk(content):
    """构造一个最小可用的流式 chunk（结构对齐 OpenAI SDK）"""
    delta = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(delta=delta, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


def _make_capturing_client(captured, stream):
    """造一个不联网的 OpenAIClient：跳过 __init__，create 把入参记进 captured。

    stream=True 时 create 返回可迭代的 chunk 迭代器，stream=False 时返回普通响应体。
    """
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
        message = types.SimpleNamespace(content="hi", tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=None)

    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_create)
    )
    return client


def _consume_stream(client, **kwargs):
    """跑一遍 stream_invoke 并耗尽生成器，顺手屏蔽内部 print 输出"""
    with contextlib.redirect_stdout(io.StringIO()):
        return list(client.stream_invoke([{"role": "user", "content": "hi"}], **kwargs))


def test_stream_invoke_passes_extra_kwargs_to_create():
    """【核心回归】stream_invoke 的额外采样参数要透传到 create，不能被丢弃"""
    captured = {}
    client = _make_capturing_client(captured, stream=True)

    _consume_stream(
        client,
        top_p=0.7,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        stop=["\n\n"],
        response_format={"type": "json_object"},
    )

    assert captured.get("top_p") == 0.7, f"top_p 应透传，实际 create 收到 {captured.get('top_p')}"
    assert captured.get("frequency_penalty") == 0.5
    assert captured.get("presence_penalty") == 0.3
    assert captured.get("stop") == ["\n\n"]
    assert captured.get("response_format") == {"type": "json_object"}
    assert captured.get("stream") is True
    print("✅ test_stream_invoke_passes_extra_kwargs_to_create passed")


def test_stream_invoke_temperature_and_max_tokens_resolution():
    """temperature / max_tokens：未传时取实例默认，传入时单次覆盖生效（且不重复透传）"""
    captured = {}
    client = _make_capturing_client(captured, stream=True)

    # 未传 → 落到实例默认值
    _consume_stream(client)
    assert captured.get("temperature") == 0.2, "未传时应取实例默认 temperature"
    assert captured.get("max_tokens") == 1024, "未传时应取实例默认 max_tokens"

    # 传入 → 单次覆盖生效
    _consume_stream(client, temperature=0.9, max_tokens=8)
    assert captured.get("temperature") == 0.9, "单次传入的 temperature 应覆盖实例默认"
    assert captured.get("max_tokens") == 8, "单次传入的 max_tokens 应覆盖实例默认"
    print("✅ test_stream_invoke_temperature_and_max_tokens_resolution passed")


def test_invoke_extra_kwargs_behaviour_unchanged():
    """回归：invoke 的额外参数透传与默认值行为保持不变"""
    captured = {}
    client = _make_capturing_client(captured, stream=False)

    client.invoke([{"role": "user", "content": "hi"}], top_p=0.6)

    assert captured.get("top_p") == 0.6
    assert captured.get("temperature") == 0.2
    assert captured.get("stream") is False
    print("✅ test_invoke_extra_kwargs_behaviour_unchanged passed")


def test_stream_invoke_and_invoke_forward_same_extra_keys():
    """一致性：同一组额外 kwargs，invoke 与 stream_invoke 交给 create 的键值一致"""
    extra = {"top_p": 0.5, "stop": ["x"], "user": "u-1"}
    inv_captured = {}
    str_captured = {}

    _make_capturing_client(inv_captured, stream=False).invoke(
        [{"role": "user", "content": "hi"}], **extra
    )
    _consume_stream(_make_capturing_client(str_captured, stream=True), **extra)

    inv_extra = {k: inv_captured[k] for k in extra}
    str_extra = {k: str_captured[k] for k in extra}
    assert inv_extra == extra, f"invoke 透传不符：{inv_extra}"
    assert str_extra == extra, f"stream_invoke 透传不符：{str_extra}"
    print("✅ test_stream_invoke_and_invoke_forward_same_extra_keys passed")


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
    test_stream_invoke_passes_extra_kwargs_to_create()  # AI新加的测试
    test_stream_invoke_temperature_and_max_tokens_resolution()  # AI新加的测试
    test_invoke_extra_kwargs_behaviour_unchanged()  # AI新加的测试
    test_stream_invoke_and_invoke_forward_same_extra_keys()  # AI新加的测试


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
