"""LLM.invoke_structured 结构化输出示例（E9）。

演示 ``response_format`` 协议（json_schema 模式）的最小闭环：
  1. 定义 pydantic 模型作为输出 schema；
  2. 用假 client 模拟端点返回 JSON；
  3. ``invoke_structured`` 自动解析为 pydantic 实例；
  4. 异常场景演示（JSON 解析失败 / pydantic 校验失败）；
  5. 嵌套模型 + 可选字段。

运行方式：在项目根目录执行
    python examples/llm/structured_output_demo.py

无需任何 API Key，完全离线可跑。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Optional, List

# 允许直接 python examples/llm/xxx.py 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field  # noqa: E402

from fast_agents.llm.openai_client import OpenAIClient  # noqa: E402
from fast_agents.llm.errors import LLMStructuredOutputError  # noqa: E402


# ==================== 1. 定义 pydantic 输出模型 ====================

class CalendarEvent(BaseModel):
    """从文本中提取的日程事件。"""
    name: str = Field(description="事件名称")
    date: str = Field(description="日期，如 2026-11-18")
    participants: List[str] = Field(description="参与者列表")


class BookReview(BaseModel):
    """书评结构化输出（含嵌套模型 + 可选字段）。"""

    class Chapter(BaseModel):
        title: str
        page_count: int

    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: Optional[float] = Field(default=None, description="评分 0-10")
    chapters: List[Chapter] = Field(default_factory=list)


# ==================== 辅助：构造离线 client ====================

def _make_fake_client(json_content: str) -> OpenAIClient:
    """造一个不联网的 OpenAIClient，返回指定 JSON 字符串内容。"""
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "demo-model"
    client.temperature = 0.2
    client.max_tokens = 1024
    client.extra_body = None
    client.embedding_model = "text-embedding-3-small"

    def _fake_create(**kwargs):
        # 打印传给 API 的 response_format，方便演示观察
        rf = kwargs.get("response_format")
        if rf:
            print(f"  [API 请求] response_format.type = {rf['type']}")
            print(f"  [API 请求] json_schema.name    = {rf['json_schema']['name']}")
            print(f"  [API 请求] strict              = {rf['json_schema'].get('strict', False)}")
        message = types.SimpleNamespace(content=json_content, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=None)

    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_fake_create)
    )
    return client


# ==================== 演示 ====================

def demo_simple_event_extraction():
    """场景 1：从文本提取日程事件。"""
    print("=" * 60)
    print("场景 1：日程事件提取")
    print("=" * 60)

    # 模拟端点返回的 JSON（符合 CalendarEvent schema）
    fake_json = json.dumps({
        "name": "科学展览会",
        "date": "2026-11-20",
        "participants": ["Alice", "Bob"],
    }, ensure_ascii=False)

    client = _make_fake_client(fake_json)
    result = client.invoke_structured(
        messages=[
            {"role": "system", "content": "从用户输入中提取日程事件，输出 JSON。"},
            {"role": "user", "content": "Alice 和 Bob 周五要去科学展览会。"},
        ],
        response_format=CalendarEvent,
    )

    print(f"  返回类型: {type(result).__name__}")
    print(f"  事件名称: {result.name}")
    print(f"  日期:     {result.date}")
    print(f"  参与者:   {result.participants}")
    print()


def demo_nested_model_with_optional():
    """场景 2：嵌套模型 + 可选字段。"""
    print("=" * 60)
    print("场景 2：书评提取（嵌套 + 可选字段）")
    print("=" * 60)

    fake_json = json.dumps({
        "title": "三体",
        "author": "刘慈欣",
        "rating": 9.5,
        "chapters": [
            {"title": "科学边界", "page_count": 15},
            {"title": "台球问题", "page_count": 12},
        ],
    }, ensure_ascii=False)

    client = _make_fake_client(fake_json)
    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取《三体》的书评信息"}],
        response_format=BookReview,
    )

    print(f"  书名:   {result.title}")
    print(f"  作者:   {result.author}")
    print(f"  评分:   {result.rating}")
    print(f"  章节数: {len(result.chapters)}")
    for ch in result.chapters:
        print(f"    - {ch.title}（{ch.page_count} 页）")
    print()


def demo_json_parse_error():
    """场景 3：端点返回非法 JSON → LLMStructuredOutputError。"""
    print("=" * 60)
    print("场景 3：JSON 解析失败（异常演示）")
    print("=" * 60)

    client = _make_fake_client("这不是 JSON {{{")
    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=CalendarEvent,
        )
    except LLMStructuredOutputError as e:
        print(f"  捕获异常: {type(e).__name__}")
        print(f"  retryable: {e.retryable}")
        print(f"  消息: {str(e)[:100]}...")
    print()


def demo_validation_error():
    """场景 4：JSON 合法但 pydantic 校验失败。"""
    print("=" * 60)
    print("场景 4：pydantic 校验失败（类型不匹配）")
    print("=" * 60)

    # participants 应为 list[str]，这里传了 int
    fake_json = json.dumps({
        "name": "会议",
        "date": "2026-11-20",
        "participants": 12345,  # 类型错误
    })

    client = _make_fake_client(fake_json)
    try:
        client.invoke_structured(
            messages=[{"role": "user", "content": "提取"}],
            response_format=CalendarEvent,
        )
    except LLMStructuredOutputError as e:
        print(f"  捕获异常: {type(e).__name__}")
        print(f"  retryable: {e.retryable}")
        print(f"  消息: {str(e)[:100]}...")
    print()


def demo_strict_mode():
    """场景 5：strict=True 严格模式（参数传递演示）。"""
    print("=" * 60)
    print("场景 5：strict=True 严格模式")
    print("=" * 60)

    fake_json = json.dumps({
        "name": "严格测试",
        "date": "2026-11-20",
        "participants": ["X"],
    })

    client = _make_fake_client(fake_json)
    result = client.invoke_structured(
        messages=[{"role": "user", "content": "提取"}],
        response_format=CalendarEvent,
        strict=True,  # 开启严格约束
    )
    print(f"  strict=True 时返回: {result.name}")
    print(f"  （注意：strict=True 要求端点支持 Structured Outputs，否则返回 400）")
    print()


def main():
    demo_simple_event_extraction()
    demo_nested_model_with_optional()
    demo_json_parse_error()
    demo_validation_error()
    demo_strict_mode()
    print("=" * 60)
    print("全部演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
