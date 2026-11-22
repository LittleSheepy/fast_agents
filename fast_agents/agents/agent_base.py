"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from ..messages import Messages
from ..messages.message import Message
from ..llm import LLM
from ..llm.models import StreamChunk


class AgentBase(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, llm: LLM, system_prompt: Optional[str] = None):
        self.name = name
        self.llm : LLM = llm
        self.system_prompt = system_prompt
        self.messages_simple: Messages = Messages()

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent，返回最终答案字符串"""
        pass
    
    @abstractmethod
    def run_stream(self, input_text: str, **kwargs) -> Iterator[StreamChunk]:
        """流式运行Agent，按StreamChunk契约产出事件，支持打字机效果"""
        pass
    
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self.messages_simple.append(message)
    
    def clear_history(self):
        """清空历史记录"""
        self.messages_simple.clear()
    
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self.messages_simple.get_messages()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    def __repr__(self) -> str:
        return self.__str__()
