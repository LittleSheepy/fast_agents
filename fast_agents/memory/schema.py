"""记忆数据模型

定义记忆系统中使用的基础数据结构，全部基于 pydantic，
与 messages 模块的 Message 保持同一种风格（pydantic BaseModel + 显式 __init__）。
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

# 记忆类型：与 HelloAgents 的 MemoryStore.KINDS 对齐
# preference  偏好（用户喜好、习惯）
# fact        事实（稳定的客观信息）
# episode     情景（发生过的事件、对话片段）
# procedure   程序（怎么做某件事的步骤/经验）
# inference   推断（由前几类推导出的结论，置信度较低）
MemoryKind = Literal["preference", "fact", "episode", "procedure", "inference"]

# 记忆状态：active 有效 / superseded 被新版本取代 / retracted 已撤回
MemoryStatus = Literal["active", "superseded", "retracted"]


class MemoryRecord(BaseModel):
    """一条长期记忆记录

    设计要点（参考 HelloAgents 的 MemoryRecord）：
    - 每条记录都带 source（来源），声明来源不等于已核实；
    - kind 区分「偏好 / 事实 / 情景 / 程序 / 推断」，避免把模型推断当成事实；
    - status + supersedes 支持「修订而不覆盖」，保证历史可追溯。
    """

    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str = ""
    session_id: str = ""
    content: str = ""
    source: str = ""
    kind: MemoryKind = "fact"
    status: MemoryStatus = "active"
    importance: float = 0.5
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    supersedes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    def to_text(self) -> str:
        """渲染成可直接注入 prompt 的一行文本"""
        return f"[{self.kind}] {self.content}"

    def __str__(self) -> str:
        return f"MemoryRecord(id={self.memory_id[:8]}, kind={self.kind}, content={self.content!r})"


class ContextPacket(BaseModel):
    """注入上下文的一小块内容

    ContextBuilder 只认识 ContextPacket，不关心它来自记忆、画像还是检索结果。
    这样以后接 RAG / 工具结果时不用改 ContextBuilder。
    """

    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        return f"- {self.content}"
