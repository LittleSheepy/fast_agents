"""记忆模块

FastAgents 的记忆系统，分两层：

- 短期记忆（BufferMemory）：当前会话的滑动窗口 + 滚动摘要
- 长期记忆（MemoryStore 实现）：跨会话持久化的偏好/事实/情景/程序

MemoryManager 是统一入口，MemoryTool 让 Agent 能主动读写记忆。
"""

from .base import MemoryStore
from .buffer import BufferMemory
from .in_memory_store import InMemoryStore
from .manager import MemoryManager
from .schema import ContextPacket, MemoryKind, MemoryRecord, MemoryStatus
from .scoring import combined_score, lexical_score, lexical_terms, time_decay
from .sqlite_store import SQLiteMemoryStore
from .tool import MemoryTool, build_memory_tools

__all__ = [
    # 数据模型
    "MemoryRecord",
    "MemoryKind",
    "MemoryStatus",
    "ContextPacket",
    # 存储后端
    "MemoryStore",
    "InMemoryStore",
    "SQLiteMemoryStore",
    # 记忆层
    "BufferMemory",
    "MemoryManager",
    # 工具
    "MemoryTool",
    "build_memory_tools",
    # 工具函数
    "lexical_terms",
    "lexical_score",
    "time_decay",
    "combined_score",
]
