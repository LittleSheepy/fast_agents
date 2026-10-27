"""上下文分层与拼装

上下文不是一根绳子，而是**分层叠起来的**。分层管两件事：

1. **拼装顺序** —— 静态内容在前、动态内容在后。这是 prompt caching 的硬要求：
   缓存是「前缀匹配」，从请求开头到缓存断点必须逐 token 一致，
   一旦某轮变动的记忆被写进 system 中段，后面全部失效，长会话成本翻倍。

2. **裁剪优先级** —— 预算不够时先砍谁。所以每层都带
   ``priority``（越大越优先保留）和 ``required``（True 表示绝不裁剪）。

分层定义（与 Anthropic / DeepAgents 的上下文分层实践对齐）：

    L1 system   身份 + 工具清单 + 技能索引     静态，required，缓存锚点
    L2 memory   长期记忆 + 会话摘要            低频变化
    L3 history  buffer 窗口内的 user/assistant 滚动
    L4 files    文件/卸载结果的按需回读         动态
    L5 user     本轮问题                       必须最后
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..messages.message import Message
from .tokens import TokenCounter, TokenBudget

__all__ = ["ContextLayer", "ContextWindow", "LAYER_ORDER"]


# 层的标准顺序（数字越小越靠前，越靠前越静态、越适合被缓存）
LAYER_ORDER: Dict[str, int] = {
    "system": 10,
    "memory": 20,
    "summary": 25,
    "history": 30,
    "files": 40,
    "tool_result": 45,
    "user": 90,
}


@dataclass
class ContextLayer:
    """上下文中的一层

    Attributes:
        name: 层名（建议取 system / memory / history / files / user）
        messages: 这一层的消息列表（Message 对象）
        priority: 裁剪优先级，越大越先保留（默认按层序倒过来）
        required: True 表示无论如何都保留（system / 本轮 user 应设为 True）
        metadata: 附加信息，便于调试与统计
    """

    name: str
    messages: List[Message] = field(default_factory=list)
    priority: int = 50
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def order(self) -> int:
        """拼装顺序：优先用标准层序，未知层排在 history 与 user 之间"""
        return LAYER_ORDER.get(self.name, 60)

    def is_empty(self) -> bool:
        return not self.messages

    def token_count(self, counter: TokenCounter) -> int:
        """本层占用的 token（按 OpenAI 格式逐条累计）"""
        total = 0
        for m in self.messages:
            for openai_msg in m.to_openai_message():
                from .tokens import count_message_tokens
                total += count_message_tokens(openai_msg, counter)
        return total

    def add(self, message: Message) -> "ContextLayer":
        self.messages.append(message)
        return self

    def extend(self, messages: List[Message]) -> "ContextLayer":
        self.messages.extend(messages)
        return self

    def clear(self) -> None:
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"<ContextLayer {self.name} messages={len(self.messages)} required={self.required}>"


class ContextWindow:
    """上下文窗口：一组按层名索引的 ContextLayer

    核心动作只有一个：``flatten()`` —— 按层序把各层拼成一串 Message，
    交给 ``Messages`` / LLM。裁剪的取舍由 ``build_messages``（见 builder.py）负责。
    """

    def __init__(self, layers: Optional[List[ContextLayer]] = None):
        self._layers: List[ContextLayer] = []
        for layer in layers or []:
            self.add_layer(layer)

    # ---------- 层管理 ----------

    def add_layer(self, layer: ContextLayer) -> ContextLayer:
        """添加一层；同名层会合并消息（并取更宽松的 required / 更高的 priority）"""
        existing = self.get_layer(layer.name)
        if existing is not None:
            existing.messages.extend(layer.messages)
            existing.required = existing.required or layer.required
            existing.priority = max(existing.priority, layer.priority)
            existing.metadata.update(layer.metadata)
            return existing
        self._layers.append(layer)
        return layer

    def get_layer(self, name: str) -> Optional[ContextLayer]:
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def list_layers(self) -> List[ContextLayer]:
        return list(self._layers)

    def append_message(self, layer_name: str, message: Message, **layer_kwargs: Any) -> ContextLayer:
        """往指定层追加一条消息；层不存在时自动创建"""
        layer = self.get_layer(layer_name)
        if layer is None:
            layer = ContextLayer(name=layer_name, **layer_kwargs)
            self.add_layer(layer)
        layer.messages.append(message)
        return layer

    # ---------- 读取 ----------

    def ordered_layers(self) -> List[ContextLayer]:
        """按拼装顺序返回（同时保持同层内原有顺序，稳定排序）"""
        return sorted(self._layers, key=lambda l: l.order)

    def flatten(self) -> List[Message]:
        """按层序拼成一条 Message 列表"""
        result: List[Message] = []
        for layer in self.ordered_layers():
            result.extend(layer.messages)
        return result

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """按层序拼成 OpenAI messages 列表（直接喂给 LLM）"""
        result: List[Dict[str, Any]] = []
        for message in self.flatten():
            result.extend(message.to_openai_message())
        return result

    def token_count(self, counter: Optional[TokenCounter] = None) -> int:
        """整窗 token 数"""
        counter = counter or TokenCounter()
        return sum(layer.token_count(counter) for layer in self._layers)

    def layer_stats(self, counter: Optional[TokenCounter] = None) -> List[Dict[str, Any]]:
        """逐层统计，便于打印「本轮上下文构成」"""
        counter = counter or TokenCounter()
        return [
            {
                "name": layer.name,
                "messages": len(layer.messages),
                "tokens": layer.token_count(counter),
                "required": layer.required,
                "priority": layer.priority,
            }
            for layer in self.ordered_layers()
        ]

    def validate(self) -> List[str]:
        """做一次层级卫生检查，返回问题列表（空列表表示没问题）

        检查项：
        - tool 消息必须紧跟在带 tool_calls 的 assistant 消息之后（OpenAI 强校验）；
        - 同一层内不应出现孤立的 tool 消息。
        """
        problems: List[str] = []
        previous_had_tool_calls = False
        for message in self.flatten():
            if message.role == "tool" and not previous_had_tool_calls:
                problems.append(
                    f"tool 消息缺少配对的 assistant(tool_calls)：{str(message.content)[:40]!r}"
                )
            previous_had_tool_calls = bool(message.tool_calls)
        return problems

    def clear(self) -> None:
        self._layers = []

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self.ordered_layers())

    def __repr__(self) -> str:
        return f"<ContextWindow layers={[l.name for l in self.ordered_layers()]}>"
