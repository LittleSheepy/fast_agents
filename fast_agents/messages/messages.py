"""消息列表系统"""

from typing import Dict, Any

from .message import Message


class Messages:
    """消息列表类"""

    def __init__(self):
        self._messages: list[Message] = []

    def get_messages(self) -> list[Message]:
        """获取历史记录"""
        return self._messages.copy()

    def to_openai_messages(self) -> Dict[str, Any]:
        """转换为OpenAI列表格式"""
        messages = []
        for message in self._messages:
            # to_openai_message 返回「一条 Message 展开后的多条消息」，用 extend 拉平
            messages.extend(message.to_openai_message())
        return messages

    def append(self, message):
        self._messages.append(message)

    def append_system(self, content):
        message = Message("system", content)
        self.append(message)

    def append_user(self, content):
        message = Message("user", content)
        self.append(message)

    def append_ai(self, content):
        message = Message("assistant", content)
        self.append(message)

    def clear(self):
        self._messages = []