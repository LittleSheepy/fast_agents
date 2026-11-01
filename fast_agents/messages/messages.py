"""消息列表系统

- 作为纯粹的消息容器, 保存消息列表
- 提供 append / extend / clear / 索引 / 切片 / 迭代 等基础容器操作
- 提供 to_openai_messages() 统一转换为 OpenAI API 格式
"""

from typing import Dict, Any, Iterable, List, Optional, Union

from .message import Message


class Messages:
    """消息列表类"""

    def __init__(self, messages: Optional[Iterable[Message]] = None):
        self._messages: List[Message] = list(messages) if messages else []

    # ---------- 写入 ----------

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        """批量追加消息"""
        self._messages.extend(messages)

    def append_system(self, content: str) -> None:
        self.append(Message("system", content))

    def append_user(self, content: str) -> None:
        self.append(Message("user", content))

    def append_ai(self, content: str) -> None:
        self.append(Message("assistant", content))

    def clear(self) -> None:
        self._messages.clear()

    # ---------- 读取 ----------

    def get_messages(self) -> List[Message]:
        """获取消息副本，避免外部误改内部列表"""
        return self._messages.copy()

    def last(self) -> Optional[Message]:
        """返回最后一条消息；空列表返回 None"""
        return self._messages[-1] if self._messages else None

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI messages 列表格式"""
        messages: List[Dict[str, Any]] = []
        for message in self._messages:
            # to_openai_message 返回「一条 Message 展开后的多条消息」，用 extend 拉平
            messages.extend(message.to_openai_message())
        return messages

    # ---------- 容器协议 ----------

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __getitem__(self, index: Union[int, slice]) -> Union[Message, List[Message]]:
        """支持索引访问与切片；切片返回 list[Message]（与 list 行为一致）"""
        return self._messages[index]

    def __bool__(self) -> bool:
        return bool(self._messages)
