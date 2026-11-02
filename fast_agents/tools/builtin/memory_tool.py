"""记忆工具 —— 让 Agent 能主动读/写/撤回长期记忆，并维护用户画像。

提供四个工具：
- remember_memory:     写入长期记忆（支持偏好/事实/做法/事件四类）
- recall_memory:       按关键词检索长期记忆
- forget_memory:       撤回一条记忆（软删除，保留 audit trail）
- update_user_profile: 更新结构化用户画像（B3 新增，默认不装配，见下方说明）

使用方式：
    from fast_agents.memory import MemoryManager
    from fast_agents.tools.builtin import build_memory_tools
    mem = MemoryManager(llm=llm)
    for t in build_memory_tools(mem):        # 默认三个（与之前一致）
        agent.add_tool(t)

    # 想额外把「更新画像」交给模型，且 MemoryManager 已配 profile_store：
    for t in build_memory_tools(mem, include_profile=True):   # 四个
        agent.add_tool(t)
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult
from ...memory.manager import MemoryManager


class RememberInput(BaseModel):
    content: str = Field(description="要记住的内容，用完整的一句话中文表述")
    kind: str = Field(default="fact", description="记忆类型：preference(偏好)/fact(事实)/procedure(做法)/episode(事件)")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要程度，0~1 之间的小数")


class RecallInput(BaseModel):
    query: str = Field(description="要查找的内容，可以是关键词或一句话")
    limit: int = Field(default=5, ge=1, le=20, description="最多返回几条记忆")


class ForgetInput(BaseModel):
    memory_id: str = Field(description="要撤回的记忆 id（支持前 8 位简写）")


class UpdateProfileInput(BaseModel):
    field: str = Field(
        description="要更新的画像字段：name(姓名)/role(身份)/location(所在地)/language(语言)，"
                    "或 preferences.<键>（稳定偏好）、facts.<键>（稳定事实）"
    )
    value: str = Field(description="字段的值，用简短的字符串表述")


class RememberTool(ToolBase):
    name = "remember_memory"
    description = "把值得长期记住的信息写入记忆库。当用户透露偏好、身份、项目背景，或明确要求你记住某事时调用。"
    args_schema = RememberInput
    tags = ["memory"]

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def run(self, content: str, kind: str = "fact", importance: float = 0.5) -> Union[str, ToolResult]:
        if not content.strip():
            return ToolResult.fail("记忆内容不能为空")
        if kind not in ("preference", "fact", "procedure", "episode"):
            kind = "fact"
        record = self.manager.remember(
            content, kind=kind, source="agent_tool",
            importance=min(1.0, max(0.0, importance)),
        )
        return ToolResult.ok(
            f"已记住：{content}（id={record.memory_id[:8]}，类型={kind}）",
            artifact={"record": record.model_dump()})


class RecallTool(ToolBase):
    name = "recall_memory"
    description = "根据关键词检索记忆库中的长期记忆。不确定是否记得、或需要参考历史事实/偏好时调用。"
    args_schema = RecallInput
    tags = ["memory"]

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def run(self, query: str, limit: int = 5) -> Union[str, ToolResult]:
        if not query.strip():
            return ToolResult.fail("检索关键词不能为空")
        records = self.manager.retrieve(query, limit=limit)
        if not records:
            return ToolResult.ok("记忆库中没有找到相关信息。")
        return ToolResult.ok(
            "\n".join(f"- [{r.kind}] {r.content}" for r in records),
            artifact={"records": [r.model_dump() for r in records]},
        )


class ForgetTool(ToolBase):
    name = "forget_memory"
    description = "撤回一条记忆。当用户明确要求你忘记某事、或你发现自己记错了时调用。需要提供 memory_id。"
    args_schema = ForgetInput
    tags = ["memory"]

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def run(self, memory_id: str) -> Union[str, ToolResult]:
        if not memory_id.strip():
            return ToolResult.fail("需要提供 memory_id")
        for r in self.manager.long_term.all():
            if r.memory_id == memory_id or r.memory_id.startswith(memory_id):
                self.manager.forget(r.memory_id)
                return ToolResult.ok(f"已撤回记忆：{r.content}")
        return ToolResult.fail(f"未找到记忆 {memory_id}")


class UpdateProfileTool(ToolBase):
    name = "update_user_profile"
    description = (
        "更新用户画像里的稳定信息（姓名、身份、所在地、语言，或长期偏好/事实）。"
        "当用户明确告知自己的稳定信息、或要求你长期记住某项偏好时调用；一次只更新一个字段。"
    )
    args_schema = UpdateProfileInput
    tags = ["memory", "profile"]

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def run(self, field: str, value: str) -> Union[str, ToolResult]:
        if not (field or "").strip():
            return ToolResult.fail("field 不能为空")
        if not (value or "").strip():
            return ToolResult.fail("value 不能为空")
        try:
            profile = self.manager.set_profile_field(field, value, source="agent_tool")
        except (ValueError, RuntimeError) as e:
            return ToolResult.fail(str(e))
        return ToolResult.ok(
            f"已更新用户画像：{field} = {value}",
            artifact={"profile": profile.to_dict()},
        )


def build_memory_tools(manager: MemoryManager, *, include_profile: bool = False) -> list:
    """一次性创建记忆工具，返回 ToolBase 实例列表，可直接传给 agent.add_tool / registry.register_tools。

    默认返回三个（remember / recall / forget），行为与之前完全一致；
    include_profile=True 时额外带上 update_user_profile —— 此时 MemoryManager 需要配了
    profile_store，否则工具调用会返回「未配置 profile_store」的失败提示。
    """
    tools = [RememberTool(manager), RecallTool(manager), ForgetTool(manager)]
    if include_profile:
        tools.append(UpdateProfileTool(manager))
    return tools
