"""工具异常类"""

from __future__ import annotations

from typing import Optional


class ToolException(Exception):
    def __init__(self, message: str, *, observation: Optional[str] = None, tool_name: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.observation = observation or message
        self.tool_name = tool_name


class ToolInitializationError(Exception):
    pass


class ToolPermissionError(ToolException):
    pass
