"""LLM.invoke_structured 单元测试（结构化输出 / E9）

聚焦：``response_format`` 协议（json_schema 模式）全链路正确性。
全程离线：用 ``__new__`` 跳过 SDK 初始化，把 ``chat.completions.create`` 换成假实现，
不发起任何网络请求、不依赖 API Key。

覆盖项：
1. 正常解析返回 pydantic 实例
2. 嵌套 pydantic 模型
3. 可选字段（缺失时取默认值）
4. JSON 解析失败 → LLMStructuredOutputError
5. 响应内容为空 → LLMStructuredOutputError
6. pydantic 校验失败 → LLMStructuredOutputError
7. 非 pydantic 类型 → TypeError
8. response_format 正确传递给 API
9. strict=True 正确传递
10. usage_sink 回填
11. LLM 门面层代理
12. LLM 门面层 tracer 打点
13. 异常类 retryable 属性
"""

import os
import sys
import types
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field
from typing import Optional, List

from fast_agents.llm.openai_client import OpenAIClient
from fast_agents.llm.llm import LLM
from fast_agents.llm.errors import LLMStructuredOutputError, LLMError


# ==================== 测试用 pydantic 模型 ====================

class SimpleOutput(BaseModel):
    """简单结构化输出模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


class NestedItem(BaseModel):
    """嵌套子模型"""
    title: str
    score: float


class NestedOutput(BaseModel):
    """嵌套结构化输出模型"""
    summary: str
    items: List[NestedItem]
    total: int


class OptionalOutput(BaseModel):
    """带可选字段的模型"""
    name: str
    nickname: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# ==================== 辅助工具 ====================

def _make_structured_client(json_content: str, usage=None):
    """造一个不联网的 OpenAIClient，返回指定 JSON 字符串内容"""
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.temperature = 0.2
    client.max_tokens = 1024
    client.extra_body = None
    client.embedding_model = "text-embedding-3-small"

    # 记录传给 create 的参数，便于断言
    captured: dict = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        message = types.SimpleNamespace(content=json_content, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=usage)

    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_fake_create)
    )
    return client, captured


# ==================== 正常路径测试 ====================

def test_invoke_structured_returns_pydantic_instance():
    """正常 JSON → 返回 pydantic 实例，字段正确"""
    data = {"name": "张三", "age": 28}
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取姓名和年龄"}],
        response_format=SimpleOutput,
    )

    assert isinstance(result, SimpleOutput), f"应返回 SimpleOutput 实例，实际 {type(result)}"
    assert result.name == "张三"
    assert result.age == 28
    print("✅ test_invoke_structured_returns_pydantic_instance passed")


def test_invoke_structured_with_nested_model():
    """嵌套 pydantic 模型正确解析"""
    data = {
        "summary": "测试总结",
        "items": [
            {"title": "项目一", "score": 95.5},
            {"title": "项目二", "score": 88.0},
        ],
        "total": 2,
    }
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=NestedOutput,
    )

    assert isinstance(result, NestedOutput)
    assert result.summary == "测试总结"
    assert len(result.items) == 2
    assert result.items[0].title == "项目一"
    assert result.items[0].score == 95.5
    assert result.total == 2
    print("✅ test_invoke_structured_with_nested_model passed")


def test_invoke_structured_optional_fields_use_defaults():
    """可选字段缺失时取默认值"""
    data = {"name": "李四"}  # 缺 nickname 和 tags
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=OptionalOutput,
    )

    assert isinstance(result, OptionalOutput)
    assert result.name == "李四"
    assert result.nickname is None
    assert result.tags == []
    print("✅ test_invoke_structured_optional_fields_use_defaults passed")


# ==================== 异常路径测试 ====================

def test_invoke_structured_invalid_json_raises_error():
    """JSON 解析失败 → LLMStructuredOutputError"""
    client, _ = _make_structured_client("这不是合法的 JSON {{{")

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=SimpleOutput,
        )
        assert False, "应抛出 LLMStructuredOutputError"
    except LLMStructuredOutputError as e:
        assert "JSON 解析失败" in str(e)
        assert isinstance(e, LLMError)
        assert e.retryable is False
    print("✅ test_invoke_structured_invalid_json_raises_error passed")


def test_invoke_structured_empty_content_raises_error():
    """响应内容为空 → LLMStructuredOutputError"""
    client, _ = _make_structured_client("")

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=SimpleOutput,
        )
        assert False, "应抛出 LLMStructuredOutputError"
    except LLMStructuredOutputError as e:
        assert "为空" in str(e)
    print("✅ test_invoke_structured_empty_content_raises_error passed")


def test_invoke_structured_none_content_raises_error():
    """响应 content 为 None → LLMStructuredOutputError"""
    client, _ = _make_structured_client(None)

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=SimpleOutput,
        )
        assert False, "应抛出 LLMStructuredOutputError"
    except LLMStructuredOutputError as e:
        assert "为空" in str(e)
    print("✅ test_invoke_structured_none_content_raises_error passed")


def test_invoke_structured_validation_failure_raises_error():
    """pydantic 校验失败（字段类型不匹配）→ LLMStructuredOutputError"""
    data = {"name": "王五", "age": "不是数字"}  # age 应为 int
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=SimpleOutput,
        )
        assert False, "应抛出 LLMStructuredOutputError"
    except LLMStructuredOutputError as e:
        assert "pydantic 校验失败" in str(e)
        assert e.retryable is False
    print("✅ test_invoke_structured_validation_failure_raises_error passed")


def test_invoke_structured_non_pydantic_type_raises_type_error():
    """response_format 不是 pydantic 子类 → TypeError（编程错误不入 LLMError 体系）"""
    client, _ = _make_structured_client("{}")

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=dict,  # 不是 pydantic 子类
        )
        assert False, "应抛出 TypeError"
    except TypeError as e:
        assert "pydantic.BaseModel" in str(e)
    except LLMStructuredOutputError:
        assert False, "非 pydantic 类型应抛 TypeError 而非 LLMStructuredOutputError"
    print("✅ test_invoke_structured_non_pydantic_type_raises_type_error passed")


def test_invoke_structured_missing_required_field_raises_error():
    """缺少必填字段 → pydantic 校验失败 → LLMStructuredOutputError"""
    data = {"age": 25}  # 缺 name
    client, _ = _make_structured_client(json.dumps(data))

    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=SimpleOutput,
        )
        assert False, "应抛出 LLMStructuredOutputError"
    except LLMStructuredOutputError as e:
        assert "pydantic 校验失败" in str(e)
    print("✅ test_invoke_structured_missing_required_field_raises_error passed")


# ==================== API 参数传递测试 ====================

def test_invoke_structured_passes_response_format_to_api():
    """验证 response_format 正确传给 chat.completions.create"""
    data = {"name": "测试", "age": 20}
    client, captured = _make_structured_client(json.dumps(data))

    client.invoke_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SimpleOutput,
    )

    assert "response_format" in captured, "API 调用应包含 response_format 参数"
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "SimpleOutput"
    assert "schema" in rf["json_schema"]
    # 验证 schema 是合法的 JSON Schema（含 properties）
    schema = rf["json_schema"]["schema"]
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    print("✅ test_invoke_structured_passes_response_format_to_api passed")


def test_invoke_structured_strict_false_by_default():
    """默认 strict=False（宽容模式）"""
    data = {"name": "测试", "age": 20}
    client, captured = _make_structured_client(json.dumps(data))

    client.invoke_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SimpleOutput,
    )

    assert captured["response_format"]["json_schema"]["strict"] is False
    print("✅ test_invoke_structured_strict_false_by_default passed")


def test_invoke_structured_strict_mode_passes_strict_true():
    """strict=True 正确传递"""
    data = {"name": "测试", "age": 20}
    client, captured = _make_structured_client(json.dumps(data))

    client.invoke_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SimpleOutput,
        strict=True,
    )

    assert captured["response_format"]["json_schema"]["strict"] is True
    print("✅ test_invoke_structured_strict_mode_passes_strict_true passed")


def test_invoke_structured_usage_sink_receives_token_counts():
    """usage_sink 正确回填 token 用量"""
    data = {"name": "测试", "age": 20}
    usage_obj = types.SimpleNamespace(
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
    )
    client, _ = _make_structured_client(json.dumps(data), usage=usage_obj)

    sink: dict = {}
    client.invoke_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SimpleOutput,
        usage_sink=sink,
    )

    assert sink.get("prompt_tokens") == 100
    assert sink.get("completion_tokens") == 50
    assert sink.get("total_tokens") == 150
    print("✅ test_invoke_structured_usage_sink_receives_token_counts passed")


def test_invoke_structured_model_override_via_kwargs():
    """单次调用可通过 kwargs 覆盖 model"""
    data = {"name": "测试", "age": 20}
    client, captured = _make_structured_client(json.dumps(data))

    client.invoke_structured(
        messages=[{"role": "user", "content": "hi"}],
        response_format=SimpleOutput,
        model="gpt-4o",
    )

    assert captured["model"] == "gpt-4o"
    print("✅ test_invoke_structured_model_override_via_kwargs passed")


# ==================== LLM 门面层测试 ====================

def test_llm_facade_invoke_structured_delegates_to_client():
    """LLM 门面 invoke_structured 正确代理给底层 client"""
    data = {"name": "门面测试", "age": 30}
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.tracer = None
    llm._client = client

    result = llm.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=SimpleOutput,
    )

    assert isinstance(result, SimpleOutput)
    assert result.name == "门面测试"
    assert result.age == 30
    print("✅ test_llm_facade_invoke_structured_delegates_to_client passed")


def test_llm_facade_invoke_structured_with_tracer():
    """LLM 门面传 tracer 时正确打点"""
    data = {"name": "打点测试", "age": 25}
    client, _ = _make_structured_client(json.dumps(data, ensure_ascii=False))

    # 造一个假 tracer
    spans_recorded: list = []

    class FakeSpan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.output = None
            self.metadata = kwargs.get("metadata", {})

        def __enter__(self):
            spans_recorded.append(self)
            return self

        def __exit__(self, *args):
            pass

    class FakeTracer:
        def span(self, **kwargs):
            return FakeSpan(**kwargs)

        def truncate_preview(self, text, length):
            return text[:length] if text else None

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.tracer = None
    llm._client = client

    result = llm.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=SimpleOutput,
        tracer=FakeTracer(),
    )

    assert isinstance(result, SimpleOutput)
    assert result.name == "打点测试"
    assert len(spans_recorded) == 1
    span = spans_recorded[0]
    assert span.name if hasattr(span, 'name') else True  # span 被记录了
    print("✅ test_llm_facade_invoke_structured_with_tracer passed")


# ==================== 异常类属性测试 ====================

def test_error_class_retryable_is_false():
    """LLMStructuredOutputError.retryable 必须为 False"""
    err = LLMStructuredOutputError("test error")
    assert err.retryable is False
    assert isinstance(err, LLMError)
    print("✅ test_error_class_retryable_is_false passed")


def test_error_class_preserves_original_error():
    """LLMStructuredOutputError 保留 original_error"""
    original = ValueError("bad json")
    err = LLMStructuredOutputError("parse failed", original_error=original)
    assert err.original_error is original
    print("✅ test_error_class_preserves_original_error passed")


# ==================== JSON 带空白字符的边界测试 ====================

def test_invoke_structured_content_with_whitespace():
    """JSON 前后有空白字符时能正确 strip 并解析"""
    data = {"name": "空白测试", "age": 10}
    content = "  \n  " + json.dumps(data) + "  \n  "
    client, _ = _make_structured_client(content)

    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=SimpleOutput,
    )

    assert isinstance(result, SimpleOutput)
    assert result.name == "空白测试"
    print("✅ test_invoke_structured_content_with_whitespace passed")


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖

    # AI新添加的测试(AI可以删除覆盖)
    test_invoke_structured_returns_pydantic_instance()
    test_invoke_structured_with_nested_model()
    test_invoke_structured_optional_fields_use_defaults()
    test_invoke_structured_invalid_json_raises_error()
    test_invoke_structured_empty_content_raises_error()
    test_invoke_structured_none_content_raises_error()
    test_invoke_structured_validation_failure_raises_error()
    test_invoke_structured_non_pydantic_type_raises_type_error()
    test_invoke_structured_missing_required_field_raises_error()
    test_invoke_structured_passes_response_format_to_api()
    test_invoke_structured_strict_false_by_default()
    test_invoke_structured_strict_mode_passes_strict_true()
    test_invoke_structured_usage_sink_receives_token_counts()
    test_invoke_structured_model_override_via_kwargs()
    test_llm_facade_invoke_structured_delegates_to_client()
    test_llm_facade_invoke_structured_with_tracer()
    test_error_class_retryable_is_false()
    test_error_class_preserves_original_error()
    test_invoke_structured_content_with_whitespace()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
