"""工具结果模型

参考 LangChain 的 ToolMessage / ToolException 设计，提供结构化的工具执行结果。
解决旧版只能返回裸字符串的问题：
- 区分成功/失败/截断三种状态，Agent 可据此决定重试或换思路；
- 携带 artifact（原始结构化数据），供下游工具或记忆使用；
- 输出 token 安全：长结果自动截断并标注，防止爆上下文。
- 携带 metadata（执行耗时等），供可观测与审计使用
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: str, artifact: Any = None, metadata: Optional[dict] = None) -> "ToolResult":
        return cls(output=output, status=ToolStatus.SUCCESS, artifact=artifact, metadata=metadata or {})

    @classmethod
    def fail(cls, error: str, metadata: Optional[dict[str, Any]] = None) -> "ToolResult":
        return cls(
            output=f"错误：{error}",
            status=ToolStatus.ERROR,
            error=error,
            is_error=True,
            metadata=metadata or {},
        )

    @classmethod
    def truncated(
        cls,
        output: str,
        original_length: int,
        limit: int,
        artifact: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ToolResult":
        note = f"\n\n[⚠️ 输出已截断：原始 {original_length} 字符，已返回前 {limit} 字符。可使用更精确的查询缩小范围。]"
        # 保留原有artifact，追加截断标记，不覆盖下游工具需要的结构化数据（如乐观锁hash、搜索结果等）
        merged_artifact: Any
        if isinstance(artifact, dict):
            merged_artifact = {**artifact, "truncated": True, "original_length": original_length, "limit": limit}
        else:
            merged_artifact = {"data": artifact, "truncated": True, "original_length": original_length, "limit": limit}
        # 保留原有metadata，追加截断标记，不覆盖duration_ms等执行元数据
        merged_metadata = dict(metadata or {})
        merged_metadata["truncated"] = True
        return cls(
            output=output + note,
            status=ToolStatus.TRUNCATED,
            artifact=merged_artifact,
            metadata=merged_metadata,
        )

    @property
    def succeeded(self) -> bool:
        """工具是否正常执行完成（截断也算完成，只是输出被裁剪）。
        
        失败判定仅基于 ERROR 状态，与截断无关。需要精确判断输出是否被裁剪
        时使用 ``is_truncated``。
        """
        return self.status in (ToolStatus.SUCCESS, ToolStatus.TRUNCATED)

    @property
    def failed(self) -> bool:
        return self.status == ToolStatus.ERROR or self.is_error

    @property
    def is_truncated(self) -> bool:
        """输出是否因超长被截断。关心结果完整性的调用方应优先判断该属性。"""
        return self.status == ToolStatus.TRUNCATED

    def __str__(self) -> str:
        return self.output
