"""多模态内容片段定义

对齐主流框架设计：content支持纯文本或多模态片段列表
"""
from __future__ import annotations
from typing import Union, Dict, Any, List, Literal
from enum import Enum
import base64


class PartType(str, Enum):
    """内容片段类型"""
    TEXT = "text"
    IMAGE = "image_url"


class ContentPart:
    """内容片段基类"""
    type: PartType

    def to_dict(self) -> Dict[str, Any]:
        """序列化为OpenAI兼容格式"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContentPart:
        """从字典反序列化"""
        part_type = data.get("type")
        if part_type == PartType.TEXT.value:
            return TextPart(text=data.get("text", ""))
        elif part_type == PartType.IMAGE.value:
            return ImagePart(url=data.get("image_url", {}).get("url", ""))
        raise ValueError(f"不支持的内容片段类型: {part_type}")


class TextPart(ContentPart):
    """文本内容片段"""
    type = PartType.TEXT

    def __init__(self, text: str):
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text
        }

    def __repr__(self) -> str:
        return f"TextPart(text='{self.text[:50]}...' if len(self.text) > 50 else self.text)"


class ImagePart(ContentPart):
    """图片内容片段，支持URL和base64两种格式"""
    type = PartType.IMAGE

    def __init__(self, url: str = "", mime_type: str = "image/png"):
        self.url = url
        self.mime_type = mime_type

    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str = "image/png") -> ImagePart:
        """从base64数据创建图片片段"""
        if not base64_data.startswith("data:"):
            data_url = f"data:{mime_type};base64,{base64_data}"
        else:
            data_url = base64_data
        return cls(url=data_url, mime_type=mime_type)

    @classmethod
    def from_bytes(cls, image_bytes: bytes, mime_type: str = "image/png") -> ImagePart:
        """从二进制数据创建图片片段"""
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        return cls.from_base64(base64_data, mime_type)

    @classmethod
    def from_url(cls, image_url: str) -> ImagePart:
        """从URL创建图片片段"""
        return cls(url=image_url)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "image_url": {
                "url": self.url
            }
        }

    def __repr__(self) -> str:
        preview = self.url[:50] + "..." if len(self.url) > 50 else self.url
        return f"ImagePart(url='{preview}')"


# 内容类型：纯字符串或多模态片段列表
Content = Union[str, List[ContentPart]]


def content_to_text(content: Content) -> str:
    """将多模态内容转换为纯文本（图片自动转为[图片]占位符）
    用于日志、Token计数、上下文裁剪等需要纯文本的场景
    """
    if isinstance(content, str):
        return content
    text_parts = []
    for part in content:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
        elif isinstance(part, ImagePart):
            text_parts.append("[图片]")
        else:
            text_parts.append(str(part))
    return "".join(text_parts)


def normalize_content(content: Any) -> Content:
    """归一化内容输入，自动处理str、list、dict等各种输入形式"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, ContentPart):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(ContentPart.from_dict(item))
            elif isinstance(item, str):
                parts.append(TextPart(text=item))
        return parts
    # 其他类型转字符串
    return str(content)