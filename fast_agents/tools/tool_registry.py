"""工具注册表 ToolRegistry

统一管理一组工具，提供：注册/反注册、按名查找、执行（invoke_tool）、
多 provider schema 导出、hook 回调。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from .structured_tool import StructuredTool
from .tool_base import ToolBase
from .tool_result import ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolBase] = {}
        self._hooks: List[Callable[[str, dict, ToolResult], None]] = []

    # ── 注册 / 反注册 ────────────────────────────────────────

    def register_tool(self, tool: Union[ToolBase, Callable]) -> ToolBase:
        """注册一个工具；裸函数会被自动包装成 StructuredTool。"""
        if callable(tool) and not isinstance(tool, ToolBase):
            tool = StructuredTool.from_function(tool)
        if not isinstance(tool, ToolBase):
            raise TypeError(f"无法注册 {type(tool)}：必须是 ToolBase 实例或可调用函数")
        if tool.name in self._tools:
            logger.warning("工具 '%s' 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        return tool

    def register_tools(self, *tools: Union[ToolBase, Callable]) -> List[ToolBase]:
        return [self.register_tool(t) for t in tools]

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ── 查询 ─────────────────────────────────────────────────

    def get_tool(self, name: str) -> Optional[ToolBase]:
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools(self) -> List[ToolBase]:
        return list(self._tools.values())

    def filter_by_tag(self, tag: str) -> List[ToolBase]:
        return [t for t in self._tools.values() if tag in t.tags]

    def get_tools_description(self) -> str:
        if not self._tools:
            return "暂无可用工具"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    # ── 执行 ─────────────────────────────────────────────────

    def invoke_tool(self, name: str, tool_input: Union[str, Dict[str, Any], None] = None, **kwargs: Any) -> ToolResult:
        """按名执行工具，统一返回 ToolResult。

        Args:
            name: 工具名。
            tool_input: JSON 字符串或 dict；支持同时用 kwargs 追加/覆盖。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"未找到名为 '{name}' 的工具。可用工具：{', '.join(self.list_names())}")
        # 解析字符串 → dict
        if isinstance(tool_input, str):
            try:
                parsed = json.loads(tool_input) if tool_input.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult.fail(f"工具 '{name}' 的参数不是合法 JSON：{e}")
            if not isinstance(parsed, dict):
                return ToolResult.fail(f"工具 '{name}' 的参数必须是 JSON 对象")
            parsed.update(kwargs)
            kwargs = parsed
        elif isinstance(tool_input, dict):
            merged = dict(tool_input)
            merged.update(kwargs)
            kwargs = merged
        elif tool_input is not None:
            return ToolResult.fail(f"tool_input 必须是 dict / JSON 字符串 / None，收到 {type(tool_input).__name__}")

        result = tool.invoke(**kwargs)
        for hook in self._hooks:
            try:
                hook(name, kwargs, result)
            except Exception as e:
                logger.exception("工具 hook 失败: %s", e)
        return result

    # ── Schema 导出（多 provider）────────────────────────────

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def to_gemini_tools(self) -> List[Dict[str, Any]]:
        return [t.to_gemini_tool() for t in self._tools.values()]

    # ── Hook ─────────────────────────────────────────────────

    def add_hook(self, hook: Callable[[str, dict, ToolResult], None]) -> None:
        self._hooks.append(hook)

    # ── 杂项 ─────────────────────────────────────────────────

    def clear(self) -> None:
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self)}>"

# 全局工具注册表
global_registry = ToolRegistry()
