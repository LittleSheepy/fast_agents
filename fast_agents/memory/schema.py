"""记忆数据模型

定义记忆系统中使用的基础数据结构，全部基于 pydantic，
与 messages 模块的 Message 保持同一种风格（pydantic BaseModel + 显式 __init__）。
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from fast_agents.utils.time_utils import cstnow

# 记忆类型：与 HelloAgents 的 MemoryStore.KINDS 对齐
# preference  偏好（用户喜好、习惯）
# fact        事实（稳定的客观信息）
# episode     情景（发生过的事件、对话片段）
# procedure   程序（怎么做某件事的步骤/经验）
# inference   推断（由前几类推导出的结论，置信度较低）
MemoryKind = Literal["preference", "fact", "episode", "procedure", "inference"]

# 记忆状态：active 有效 / superseded 被新版本取代 / retracted 已撤回 /
#           archived 已归档（B5：低重要度 + 久未命中，退出检索但保留可审计）
MemoryStatus = Literal["active", "superseded", "retracted", "archived"]


class MemoryRecord(BaseModel):
    """一条长期记忆记录

    设计要点（参考 HelloAgents 的 MemoryRecord）：
    - 每条记录都带 source（来源），声明来源不等于已核实；
    - kind 区分「偏好 / 事实 / 情景 / 程序 / 推断」，避免把模型推断当成事实；
    - status + supersedes 支持「修订而不覆盖」，保证历史可追溯；
    - access_count / last_accessed_at 记录「被命中过几次、上次命中是什么时候」，
      是 B5 衰减归档的判定依据（不参与检索打分，只服务于维护）。
    """

    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str = ""
    session_id: str = ""
    content: str = ""
    source: str = ""
    kind: MemoryKind = "fact"
    status: MemoryStatus = "active"
    importance: float = 0.5
    created_at: datetime = Field(default_factory=cstnow)
    updated_at: datetime = Field(default_factory=cstnow)
    supersedes: Optional[str] = None
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["last_accessed_at"] = (
            self.last_accessed_at.isoformat() if self.last_accessed_at else None
        )
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


class UserProfile(BaseModel):
    """结构化用户画像（B3）

    和 MemoryRecord 的分工：
    - MemoryRecord 是「可能被遗忘 / 修订 / 检索的碎片」，带 kind / status / supersedes；
    - UserProfile 是「经过确认、稳定存在的单份档案」，字段级覆盖式合并，没有版本链。
    所以画像**不塞进长期记忆的检索链路**（否则会和 fact / preference 重复打分）。

    字段设计刻意保持「扁平 + dict 扩展」：
    - 顶层只放 name / role / location / language 四个通用字段；
    - 其余稳定信息走 stable_preferences / stable_facts 两个键值字典，
      避免过早把字段写死成白名单（等真实用一段时间再收紧）。
    """

    user_id: str = ""
    name: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    language: str = "zh-CN"
    stable_preferences: Dict[str, str] = Field(default_factory=dict)
    stable_facts: Dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=cstnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        """是否还是一份没有任何信息的空档案（language 有默认值，不计入）"""
        return not any(
            [self.name, self.role, self.location,
             self.stable_preferences, self.stable_facts]
        )

    def merge_patch(self, patch: Dict[str, Any]) -> "UserProfile":
        """按字段覆盖式合并：同名 key 后写覆盖前写，两个 dict 做浅合并

        - None 值一律忽略（不覆盖已有内容）；
        - user_id / updated_at 不允许通过 patch 改写；
        - 未知的顶层键忽略，不会往模型上乱塞属性。
        """
        for key, value in (patch or {}).items():
            if value is None:
                continue
            if key == "stable_preferences" and isinstance(value, dict):
                self.stable_preferences.update({str(k): str(v) for k, v in value.items()})
            elif key == "stable_facts" and isinstance(value, dict):
                self.stable_facts.update({str(k): str(v) for k, v in value.items()})
            elif key == "metadata" and isinstance(value, dict):
                self.metadata.update(value)
            elif key in ("user_id", "updated_at"):
                continue
            elif key in ("name", "role", "location", "language"):
                setattr(self, key, value)
        self.updated_at = cstnow()
        return self

    def to_text(self) -> str:
        """渲染成可直接注入 prompt 的多行文本（B4 双通道注入时会用到）"""
        lines: list = []
        if self.name:
            lines.append(f"- 姓名：{self.name}")
        if self.role:
            lines.append(f"- 身份：{self.role}")
        if self.location:
            lines.append(f"- 所在地：{self.location}")
        if self.language:
            lines.append(f"- 语言：{self.language}")
        for k, v in self.stable_preferences.items():
            lines.append(f"- 偏好·{k}：{v}")
        for k, v in self.stable_facts.items():
            lines.append(f"- 事实·{k}：{v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """dict 序列化，datetime 转 ISO 串（与 MemoryRecord.to_dict 保持一致）"""
        data = self.model_dump()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    def __str__(self) -> str:
        return f"UserProfile(user_id={self.user_id!r}, name={self.name!r}, role={self.role!r})"
