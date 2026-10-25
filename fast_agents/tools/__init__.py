"""内置工具模块

FastAgents框架的内置工具集合，包括：
- SearchTool: 网页搜索工具
- TerminalTool: 命令行工具
"""

from .calculator_tool import CalculatorTool, calculate
from .terminal_tool import TerminalTool

__all__ = [
    "CalculatorTool",
    "calculate",
    "TerminalTool",
]