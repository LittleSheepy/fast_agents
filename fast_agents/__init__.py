"""FastAgents - 灵活、可扩展的多智能体框架"""

from dotenv import load_dotenv
load_dotenv()

from .version import __version__, __author__, __email__, __description__
from .llm import LLM
from .agents.react_agent.react_agent import ReActAgent
from .tools import (
    ToolBase, StructuredTool, tool, ToolResult, ToolStatus,
    ToolException, ToolRegistry, global_registry, builtin,
)
from .memory import MemoryManager

__all__ = [
    "__version__", "__author__", "__email__", "__description__",
    "LLM", "ReActAgent",
    "ToolBase", "StructuredTool", "tool", "ToolResult", "ToolStatus",
    "ToolException", "ToolRegistry", "global_registry", "builtin",
    "MemoryManager",
]
