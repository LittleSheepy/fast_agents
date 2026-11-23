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


def content_to_text(content: Any) -> Any:
    """将消息类内容归一化为纯文本（图片自动转为[图片]占位符）。

    **仅处理消息类型输入**：str / None / List[ContentPart] / OpenAI/项目 Message
    风格 dict（含 ``content`` 字段）。

    - str 原样返回；
    - None → 空串；
    - List[ContentPart]（E8 多模态）拼接文本部分，图片转「[图片]」占位；
      list 中非 ContentPart 的 str / dict 项会被归一化后再拼文本；
    - dict 且含 ``content`` 字段（``{"role":..., "content":...}``）：递归取 content
      归一化；
    - **其他类型（MsgToolCall / ToolCallInfo / ToolResultInfo 等业务对象、数字、
      任意自定义对象）原样返回**，不做 str() 强转——由调用方按自己的契约处理。

    适用于日志、Token 计数、上下文裁剪、安全护栏字符串匹配等需要纯文本的场景。
    非消息内容请调用方自行判定，不要依赖本函数做通用字符串化。
    """
    # 非消息类原样返回，保持业务对象类型（避免 ToolCallInfo/ToolResultInfo
    # 等结构化输入被意外转成字符串，让下游 before_call/after_call 拿到对象）
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool, bytes)):
        # 基础标量转字符串（安全，语义明确）
        return str(content)
    if isinstance(content, dict) and "content" in content:
        # OpenAI/项目 Message 风格 {"role":..., "content":...} → 递归归一化
        return content_to_text(content.get("content", ""))
    if isinstance(content, list):
        normalized_list = normalize_content(content)
        if isinstance(normalized_list, str):
            return normalized_list
        text_parts: List[str] = []
        for part in normalized_list:
            if isinstance(part, TextPart):
                text_parts.append(part.text or "")
            elif isinstance(part, ImagePart):
                text_parts.append("[图片]")
            elif hasattr(part, "text"):
                val = getattr(part, "text", "")
                text_parts.append(val if isinstance(val, str) else str(val))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    # MsgToolCall / ToolCallInfo / ToolResultInfo / 任意业务对象 → 原样返回
    return content


def normalize_content(content: Any) -> Content:
    """归一化内容输入，自动处理 None/str/list 等消息输入形式。

    - None → 空串；
    - str → 原样返回；
    - list 元素：ContentPart 原样保留，dict 且符合 ContentPart 结构时通过
      ``ContentPart.from_dict`` 反序列化，str 包装为 TextPart；其他元素
      （含带 text 属性的对象）兜底为 TextPart(str(item))；
    - dict 且含 ``content`` 字段（OpenAI/项目 Message 风格）：递归归一化其 content；
    - 其他类型 → **原样返回**（不做通用 str 兜底，避免破坏业务对象类型）。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and "content" in content:
        # OpenAI/项目 Message 风格 {"role":..., "content":...}
        return normalize_content(content.get("content", ""))
    if isinstance(content, list):
        parts: List[ContentPart] = []
        for item in content:
            if isinstance(item, ContentPart):
                parts.append(item)
            elif isinstance(item, dict):
                # 仅当 dict 是 ContentPart 结构（含 type: text/image_url）才反序列化，
                # 否则按字符串兜底，避免把任意 dict 误当 ContentPart
                part_type = item.get("type")
                if part_type in (PartType.TEXT.value, PartType.IMAGE.value):
                    try:
                        parts.append(ContentPart.from_dict(item))
                        continue
                    except (ValueError, KeyError):
                        pass
                parts.append(TextPart(text=str(item)))
            elif isinstance(item, str):
                parts.append(TextPart(text=item))
            elif hasattr(item, "text"):
                val = getattr(item, "text", "")
                parts.append(TextPart(text=val if isinstance(val, str) else str(val)))
            else:
                parts.append(TextPart(text=str(item)))
        return parts
    # 其他类型（业务对象等）原样返回
    return content