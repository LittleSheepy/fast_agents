"""OpenAIClient.invoke 单元测试（工具调用解析兼容性）

聚焦一件事：**带 tool_calls 的返回体一律要解析出工具调用，不看 finish_reason**。
部分 OpenAI 兼容端点返回 tool_calls 时 finish_reason 仍为 "stop"，若只认
finish_reason == "tool_calls" 会漏掉工具调用、让整条 ReAct 链路空转。

全程离线：用 __new__ 跳过 SDK 初始化，把 chat.completions.create 换成假实现，
不发起任何网络请求、不依赖 API Key。
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.llm.openai_client import OpenAIClient


def _fake_tool_call(name: str, arguments: str, call_id: str):
    """构造一个最小可用的 tool_call 对象（结构对齐 OpenAI SDK）"""
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def _make_response(content, tool_calls, finish_reason):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=None)


def _make_client(response):
    """造一个不联网的 OpenAIClient：跳过 __init__，只挂必要属性 + 假 create"""
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.temperature = 0.2
    client.max_tokens = 1024
    client.extra_body = None
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kwargs: response)
    )
    return client


def test_invoke_parses_tool_calls_when_finish_reason_is_stop():
    """【核心回归】finish_reason="stop" 但带 tool_calls → 仍要解析出工具调用"""
    fake_tool_call = _fake_tool_call("python_calculator", '{"expression": "1+1"}', "call_a")
    response = _make_response("", [fake_tool_call], "stop")
    client = _make_client(response)

    msg = client.invoke([{"role": "user", "content": "1+1 等于几"}])

    assert len(msg.tool_calls) == 1, f"应解析出 1 个工具调用，实际 {len(msg.tool_calls)}"
    assert msg.tool_calls[0].name == "python_calculator"
    assert msg.tool_calls[0].arguments == '{"expression": "1+1"}'
    assert msg.tool_calls[0].id == "call_a"
    print("✅ test_invoke_parses_tool_calls_when_finish_reason_is_stop passed")


def test_invoke_still_parses_tool_calls_when_finish_reason_is_tool_calls():
    """标准端点回归：finish_reason="tool_calls" 行为不变"""
    fake_tool_call = _fake_tool_call("search", '{"q": "x"}', "call_b")
    response = _make_response("", [fake_tool_call], "tool_calls")
    client = _make_client(response)

    msg = client.invoke([{"role": "user", "content": "搜一下"}])

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"
    assert msg.tool_calls[0].id == "call_b"
    print("✅ test_invoke_still_parses_tool_calls_when_finish_reason_is_tool_calls passed")


def test_invoke_no_tool_calls_keeps_plain_answer():
    """无工具调用：content 正常返回，tool_calls 为空"""
    response = _make_response("直接回答", None, "stop")
    client = _make_client(response)

    msg = client.invoke([{"role": "user", "content": "你好"}])

    assert msg.content == "直接回答"
    assert msg.tool_calls == []
    print("✅ test_invoke_no_tool_calls_keeps_plain_answer passed")


def test_invoke_missing_tool_calls_attr_does_not_break():
    """防御：message 对象连 tool_calls 属性都没有时也不应报错"""
    message = types.SimpleNamespace(content="无该字段")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(choices=[choice], usage=None)
    client = _make_client(response)

    msg = client.invoke([{"role": "user", "content": "你好"}])

    assert msg.content == "无该字段"
    assert msg.tool_calls == []
    print("✅ test_invoke_missing_tool_calls_attr_does_not_break passed")


# ==================== stream_invoke 流式聚合测试 ====================
def _make_stream_delta(content=None, tool_calls=None, finish_reason=None, reasoning_content=None):
    """构造一个流式 delta 对象，模拟 OpenAI 分片"""
    delta_kwargs = {}
    if content is not None:
        delta_kwargs["content"] = content
    if reasoning_content is not None:
        delta_kwargs["reasoning_content"] = reasoning_content
    # 构造 tool_calls 分片
    if tool_calls is not None:
        tc_list = []
        for tc in tool_calls:
            fn = types.SimpleNamespace(
                name=tc.get("name"),
                arguments=tc.get("arguments_delta", ""),
            )
            tc_obj = types.SimpleNamespace(
                index=tc.get("index", 0),
                id=tc.get("id"),
                function=fn,
            )
            tc_list.append(tc_obj)
        delta_kwargs["tool_calls"] = tc_list
    delta = types.SimpleNamespace(**delta_kwargs)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice])


def _make_streaming_client(chunks):
    """造一个流式假客户端，返回 chunks 迭代器"""
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.temperature = 0.2
    client.max_tokens = 1024
    client.extra_body = None
    # 模拟流式响应迭代器
    def _create(**kwargs):
        return iter(chunks)
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_create)
    )
    return client


def test_stream_plain_text_returns_correct_chunks():
    """纯文本流：text片透传正确，end片content为完整正文、tool_calls为空"""
    chunks = [
        _make_stream_delta(content="你"),
        _make_stream_delta(content="好"),
        _make_stream_delta(content="世界", finish_reason="stop"),
    ]
    client = _make_streaming_client(chunks)
    # 屏蔽print输出避免干扰测试
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = list(client.stream_invoke([{"role": "user", "content": "hi"}]))
    # 3个text片 + 1个end片
    text_chunks = [c for c in result if c.type == "text"]
    end_chunks = [c for c in result if c.type == "end"]
    assert len(text_chunks) == 3
    assert [c.text for c in text_chunks] == ["你", "好", "世界"]
    assert len(end_chunks) == 1
    assert end_chunks[0].content == "你好世界"
    assert end_chunks[0].tool_calls == []
    assert end_chunks[0].finish_reason == "stop"
    print("✅ test_stream_plain_text_returns_correct_chunks passed")


def test_stream_single_tool_call_aggregates_arguments():
    """单工具调用分片聚合：arguments分片正确拼接，id/name取首片"""
    chunks = [
        # 首片带id/name，第一片参数
        _make_stream_delta(tool_calls=[{"index": 0, "id": "call_123", "name": "add", "arguments_delta": '{"a":1'}]),
        # 后续片不带id/name，只带参数增量
        _make_stream_delta(tool_calls=[{"index": 0, "arguments_delta": ',"b":2}'}]),
        _make_stream_delta(finish_reason="tool_calls"),
    ]
    client = _make_streaming_client(chunks)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = list(client.stream_invoke([{"role": "user", "content": "1+2"}]))
    # 2个tool_call片 + 1个end片（无text）
    tc_chunks = [c for c in result if c.type == "tool_call"]
    end_chunks = [c for c in result if c.type == "end"]
    assert len(tc_chunks) == 2
    assert tc_chunks[0].index == 0 and tc_chunks[0].id == "call_123" and tc_chunks[0].name == "add"
    assert tc_chunks[0].arguments_delta == '{"a":1'
    assert tc_chunks[1].arguments_delta == ',"b":2}'
    assert len(end_chunks) == 1
    assert len(end_chunks[0].tool_calls) == 1
    tc = end_chunks[0].tool_calls[0]
    assert tc.id == "call_123"
    assert tc.name == "add"
    assert tc.arguments == '{"a":1,"b":2}'
    print("✅ test_stream_single_tool_call_aggregates_arguments passed")


def test_stream_parallel_tool_calls_aggregate_by_index():
    """并行多工具调用：按index正确归并，互不串扰"""
    chunks = [
        # 第一片两个工具的首部分别到达
        _make_stream_delta(tool_calls=[
            {"index": 0, "id": "call_0", "name": "search", "arguments_delta": '{"q":"'},
            {"index": 1, "id": "call_1", "name": "calc", "arguments_delta": '{"expr":"'},
        ]),
        # 第二片两个工具的参数增量
        _make_stream_delta(tool_calls=[
            {"index": 0, "arguments_delta": 'fast_agents"}'},
            {"index": 1, "arguments_delta": '1+1"}'},
        ]),
        _make_stream_delta(finish_reason="tool_calls"),
    ]
    client = _make_streaming_client(chunks)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = list(client.stream_invoke([{"role": "user", "content": "搜一下fast_agents同时算1+1"}]))
    end_chunks = [c for c in result if c.type == "end"]
    assert len(end_chunks) == 1
    tcs = end_chunks[0].tool_calls
    assert len(tcs) == 2
    # 按index升序，index0是search，index1是calc
    assert tcs[0].id == "call_0" and tcs[0].name == "search"
    assert tcs[0].arguments == '{"q":"fast_agents"}'
    assert tcs[1].id == "call_1" and tcs[1].name == "calc"
    assert tcs[1].arguments == '{"expr":"1+1"}'
    print("✅ test_stream_parallel_tool_calls_aggregate_by_index passed")


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
    test_invoke_parses_tool_calls_when_finish_reason_is_stop()  # AI新加的测试
    test_invoke_still_parses_tool_calls_when_finish_reason_is_tool_calls()  # AI新加的测试
    test_invoke_no_tool_calls_keeps_plain_answer()  # AI新加的测试
    test_invoke_missing_tool_calls_attr_does_not_break()  # AI新加的测试
    test_stream_plain_text_returns_correct_chunks()  # AI新加的流式测试
    test_stream_single_tool_call_aggregates_arguments()  # AI新加的流式测试
    test_stream_parallel_tool_calls_aggregate_by_index()  # AI新加的流式测试


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
