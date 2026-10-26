"""消息系统"""

from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system", "tool"]

class MsgToolCall(BaseModel):
    name: str
    arguments: str
    id: str

class MsgToolResult(BaseModel):
    name: str
    result: str
    id: str

class Message(BaseModel):
    """消息类"""

    role: MessageRole
    content: str = ""
    tool_calls: List[MsgToolCall] = []
    tool_results: List[MsgToolResult] = []

    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, role: MessageRole, content: str="", **kwargs):
        super().__init__(
            role=role,
            content=content,
            tool_calls=kwargs.get('tool_calls', []),
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def to_openai_message(self) -> List[Dict[str, Any]]:
        """转换为字典格式（OpenAI API格式）"""
        messages = []
        if self.role in ["user","system", "assistant"]:
            message = {
                "role": self.role,
                "content": self.content
            }
            if self.tool_calls:
                message["tool_calls"] = []
                for tool_call in self.tool_calls:
                    tool_call_openai = {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                        }
                    }
                    message["tool_calls"].append(tool_call_openai)
            messages.append(message)
        if self.role == "tool":
            if self.tool_calls:
                for tool_call in self.tool_calls:
                    message = {
                        "role": self.role,
                        "content": self.content,
                        "tool_call_id": tool_call.id
                    }
                    messages.append(message)
            else:
                # 兜底：ReActAgent 通过 metadata 挂载 tool_call_id / name，
                # 没有 tool_calls 时从 metadata 里取，避免 tool 消息被静默丢弃
                tool_call_id = (self.metadata or {}).get("tool_call_id", "")
                message = {
                    "role": self.role,
                    "content": self.content,
                    "tool_call_id": tool_call_id
                }
                messages.append(message)

        return messages
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

if __name__ == '__main__':
    tool_call = MsgToolCall(name="test_name", arguments="test_arguments", id="test_id")
    message = Message(role="user", content="test_content", tool_calls=[tool_call])
    message_openai = message.to_openai_message()
    print(message_openai)
    tool_msg = Message("tool", "14", metadata={"tool_call_id": "call_1", "name": "calc"})
    print(tool_msg.to_openai_message())
