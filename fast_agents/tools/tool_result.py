"""工具结果模型

参考 LangChain 的 ToolMessage / ToolException 设计，提供结构化的工具执行结果。
解决旧版只能返回裸字符串的问题：
- 区分成功/失败/截断三种状态，Agent 可据此决定重试或换思路；
- 携带 artifact（原始结构化数据），供下游工具或记忆使用；
- 输出 token 安全：长结果自动截断并标注，防止爆上下文。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TRUNCATED = "truncated"


@dataclass
class ToolResult:
    """工具执行结果"""
    output: str = ""
    status: ToolStatus = ToolStatus.SUCCESS
    error: Optional[str] = None
    artifact: Any = None
    is_error: bool = False

    @classmethod
    def ok(cls, output: str, artifact: Any = None) -> "ToolResult":
        return cls(output=output, status=ToolStatus.SUCCESS, artifact=artifact)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(
            output=f"错误：{error}",
            status=ToolStatus.ERROR,
            error=error,
            is_error=True,
        )

    @classmethod
    def truncated(cls, output: str, original_length: int, limit: int) -> "ToolResult":
        note = f"\n\n[⚠️ 输出已截断：原始 {original_length} 字符，已返回前 {limit} 字符。可使用更精确的查询缩小范围。]"
        return cls(
            output=output + note,
            status=ToolStatus.TRUNCATED,
            artifact={"truncated": True, "original_length": original_length, "limit": limit},
        )

    @property
    def succeeded(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == ToolStatus.ERROR or self.is_error

    def __str__(self) -> str:
        return self.output
