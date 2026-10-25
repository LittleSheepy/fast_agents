"""记忆工具

把记忆能力包装成标准 Tool，这样 ReActAgent 可以像调用 calculator 一样
主动「记住一件事」或「查一下我之前说过什么」。

完全复用你 tools/tool_base.py 的 ToolBase + param_model 约定，
所以 build_openai_like_tool() / | 运算符 / ToolRegistry 注册都能直接用。
"""

from typing import Optional
from pydantic import BaseModel, Field

from ..tools.tool_base import ToolBase, ToolParameter
from .manager import MemoryManager


class RememberParams(BaseModel):
    """记忆写入参数"""

    content: str = Field(description="要记住的内容，用完整的一句话中文表述")
    kind: str = Field(
        default="fact",
        description="记忆类型：preference(偏好) / fact(事实) / procedure(做法) / episode(事件)",
    )
    importance: float = Field(default=0.5, description="重要程度，0~1 之间的小数")


class RecallParams(BaseModel):
    """记忆检索参数"""

    query: str = Field(description="要查找的内容，可以是关键词或一句话")
    limit: int = Field(default=5, description="最多返回几条记忆")


class MemoryTool(ToolBase):
    """记忆工具：支持 remember / recall / forget 三个动作"""

    def __init__(self, manager: MemoryManager, action: str = "remember"):
        """
        Args:
            manager: MemoryManager 实例
            action: remember（写入）/ recall（检索）/ forget（撤回）
        """
        if action not in ("remember", "recall", "forget"):
            raise ValueError("action 必须是 remember / recall / forget 之一")
        self.manager = manager
        self.action = action
        self.param_descriptions = {
            "content": "要记住的内容，用完整的一句话中文表述",
            "query": "要查找的内容，可以是关键词或一句话",
            "limit": "最多返回几条记忆",
            "memory_id": "要撤回的记忆 id",
        }
        names = {
            "remember": (
                "remember_memory",
                "把值得长期记住的信息写入记忆库。当用户透露偏好、身份、"
                "项目背景或明确要求你记住某事时调用。",
            ),
            "recall": (
                "recall_memory",
                "从记忆库中检索与当前问题相关的历史信息。当需要回忆用户"
                "此前说过的偏好或事实时调用。",
            ),
            "forget": (
                "forget_memory",
                "撤回一条记忆。当用户明确要求你忘记某事、或你发现自己"
                "记错了时调用。",
            ),
        }
        super().__init__(
            name=names[action][0],
            description=names[action][1],
            param_model={"remember": RememberParams, "recall": RecallParams}[action]
            if action != "forget"
            else None,
        )

    def run(self, *args, **kwargs) -> str:
        """按 action 分发；ToolRegistry 会把参数以 kwargs 传入"""
        if self.action == "remember":
            content = kwargs.get("content", "")
            kind = kwargs.get("kind", "fact") or "fact"
            if kind not in ("preference", "fact", "procedure", "episode"):
                kind = "fact"
            try:
                importance = float(kwargs.get("importance", 0.5) or 0.5)
            except (TypeError, ValueError):
                importance = 0.5
            record = self.manager.remember(
                content, kind=kind, source="agent_tool",
                importance=min(1.0, max(0.0, importance)),
            )
            return f"已记住（id={record.memory_id[:8]}）：{record.content}"

        if self.action == "recall":
            query = kwargs.get("query", "")
            try:
                limit = int(kwargs.get("limit", 5) or 5)
            except (TypeError, ValueError):
                limit = 5
            records = self.manager.retrieve(query, limit=limit)
            if not records:
                return "记忆库中没有找到相关信息。"
            return "\n".join(f"- [{r.kind}] {r.content}" for r in records)

        memory_id = kwargs.get("memory_id", "")
        if not memory_id:
            return "错误：需要提供 memory_id"
        # 支持用前 8 位简写定位
        for r in self.manager.long_term.all():
            if r.memory_id == memory_id or r.memory_id.startswith(memory_id):
                self.manager.forget(r.memory_id)
                return f"已撤回记忆：{r.content}"
        return f"错误：未找到记忆 {memory_id}"

    def get_parameters(self):
        """forget 动作没有 param_model，走 inspect 兜底也没参数，这里显式声明"""
        if self.action == "forget":
            return [
                ToolParameter(
                    name="memory_id", type="string",
                    description="要撤回的记忆 id", required=True,
                )
            ]
        # 有 param_model 时由基类自动生成，这里返回空列表即可
        return []


def build_memory_tools(manager: MemoryManager) -> list:
    """一次性构建三个记忆工具，方便注册到 ToolRegistry"""
    return [
        MemoryTool(manager, "remember"),
        MemoryTool(manager, "recall"),
        MemoryTool(manager, "forget"),
    ]
