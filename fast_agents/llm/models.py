"""LLM 层数据模型（对外契约）。

StreamChunk：``LLM.stream_invoke`` 的流式输出单元，text / tool_call / end 三态。
"""
from dataclasses import dataclass, field
from typing import Literal

from ..messages.message import MsgToolCall


@dataclass
class StreamChunk:
    """流式输出契约：text / tool_call / end 三态。"""
    type: Literal["text", "tool_call", "end"]
    text: str = ""                # text：正文增量
    index: int = 0                # tool_call：第几个并行调用
    id: str = ""                  # tool_call：首片带 id
    name: str = ""                # tool_call：首片带 name
    arguments_delta: str = ""     # tool_call：参数增量片段
    finish_reason: str = ""       # end：仅透传，不做判据
    content: str = ""             # end：完整正文
    tool_calls: list[MsgToolCall] = field(default_factory=list)  # end：聚合结果
