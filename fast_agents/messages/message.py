"""消息系统"""

from typing import Optional, Dict, Any, Literal, List, Union
from datetime import datetime
from fast_agents.utils.time_utils import cstnow
from pydantic import BaseModel
from fast_agents.messages.parts import (
    Content,
    ContentPart,
    TextPart,
    ImagePart,
    content_to_text,
    normalize_content
)

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
    """消息类，支持纯文本和多模态内容"""

    model_config = {"arbitrary_types_allowed": True}

    role: MessageRole
    content: Content = ""
    tool_calls: List[MsgToolCall] = []
    tool_results: List[MsgToolResult] = []

    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, role: MessageRole, content: Any = "", **kwargs):
        super().__init__(
            role=role,
            content=normalize_content(content),
            tool_calls=kwargs.get('tool_calls', []),
            timestamp=kwargs.get('timestamp', cstnow()),
            metadata=kwargs.get('metadata', {})
        )

    @property
    def text(self) -> str:
        """获取纯文本内容，多模态场景下自动提取文本、图片转为[图片]占位符
        所有需要字符串处理的场景统一使用这个属性，无需判断content类型
        """
        return content_to_text(self.content)
    
    def add_text(self, text: str) -> "Message":
        """链式添加文本片段"""
        if isinstance(self.content, str):
            self.content = TextPart(text=self.content) if self.content else []
        self.content.append(TextPart(text=text))
        return self
    
    def add_image(self, image_url: str) -> "Message":
        """链式添加图片URL"""
        if isinstance(self.content, str):
            self.content = [TextPart(text=self.content)] if self.content else []
        self.content.append(ImagePart.from_url(image_url))
        return self
    
    def add_image_base64(self, base64_data: str, mime_type: str = "image/png") -> "Message":
        """链式添加base64编码的图片，直接对接MCP返回的图片结果"""
        if isinstance(self.content, str):
            self.content = [TextPart(text=self.content)] if self.content else []
        self.content.append(ImagePart.from_base64(base64_data, mime_type))
        return self
    
    def _content_to_openai_format(self) -> Union[str, List[Dict[str, Any]]]:
        """将content转换为OpenAI API要求的格式"""
        if isinstance(self.content, str):
            return self.content
        return [part.to_dict() for part in self.content]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {"role": self.role, "content": self._content_to_openai_format()}

    def to_openai_message(self) -> List[Dict[str, Any]]:
        """转换为字典格式（OpenAI API格式，兼容多模态）"""
        messages = []
        openai_content = self._content_to_openai_format()
        if self.role in ["user","system", "assistant"]:
            message = {
                "role": self.role,
                "content": openai_content
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
                        "content": openai_content,
                        "tool_call_id": tool_call.id
                    }
                    messages.append(message)
            else:
                # 兜底：ReActAgent 通过 metadata 挂载 tool_call_id / name，
                # 没有 tool_calls 时从 metadata 里取，避免 tool 消息被静默丢弃
                tool_call_id = (self.metadata or {}).get("tool_call_id", "")
                message = {
                    "role": self.role,
                    "content": openai_content,
                    "tool_call_id": tool_call_id
                }
                messages.append(message)

        return messages
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.text}"

if __name__ == '__main__':
    tool_call = MsgToolCall(name="test_name", arguments="test_arguments", id="test_id")
    # 纯文本场景测试
    message = Message(role="user", content="test_content", tool_calls=[tool_call])
    message_openai = message.to_openai_message()
    print("纯文本OpenAI格式:", message_openai)
    tool_msg = Message("tool", "14", metadata={"tool_call_id": "call_1", "name": "calc"})
    print("tool消息:", tool_msg.to_openai_message())
    # 多模态场景测试
    mm_msg = Message("user", "请描述这张图片：")
    mm_msg.add_image_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    print("多模态消息文本:", str(mm_msg))
    print("多模态OpenAI格式:", mm_msg.to_openai_message())