"""
FastAgents - 灵活、可扩展的多智能体框架

基于OpenAI原生API构建，提供快速简洁高效的智能体开发体验。
"""
from dotenv import load_dotenv
load_dotenv("D:/00sheepy/.env")
from .version import __version__, __author__, __email__, __description__

# 核心组件
from .llm import LLM

# Agent实现
from .agents.react_agent.react_agent import ReActAgent

# 工具系统
from .tools.tool_registry import ToolRegistry, global_registry

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__description__",

    # 核心组件
    "LLM",

    # Agent范式
    "ReActAgent",

    # 工具系统
    "ToolRegistry",
    "global_registry",
]


