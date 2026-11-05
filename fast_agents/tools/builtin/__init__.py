"""内置工具集合

设计要点：延迟导入（lazy import）——避免 `import fast_agents.tools.builtin` 时
强制拉起 `memory_tool` / `mcp_tool` 子模块，解除 tools → memory / mcp 的反向依赖，
仅在实际访问对应工具类 / 工厂函数时才动态导入对应子模块；公开 API 与导入路径
完全兼容原有写法，无破坏性变更。

参考实现（只借鉴设计思路，不引其依赖）：
- LangChain 借助模块级 `__getattr__` 做惰性导出（如 `langchain_core.language_models`）；
- 标准库 PEP 562「Module __getattr__ and __dir__」——本模块即采用该机制；
- LlamaIndex / OpenAI Agents SDK 侧重按需加载可选集成包，设计思路一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# 类型检查阶段仍导入，保证IDE类型提示正常
if TYPE_CHECKING:
    from .calculator_tool import CalculatorTool
    from .terminal_tool import TerminalTool
    from .search_tool import SearchTool
    from .file_tool import ReadTool, WriteTool, EditTool, MultiEditTool, build_file_tools
    from .memory_tool import (
        RememberTool, RecallTool, ForgetTool, UpdateProfileTool, build_memory_tools,
    )
    from .mcp_tool import ListMcpServersTool, ConnectMcpServerTool, DisconnectMcpServerTool, build_mcp_tools

# 延迟导入映射：公开名称 -> (子模块路径, 对象名)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CalculatorTool": (".calculator_tool", "CalculatorTool"),
    "TerminalTool": (".terminal_tool", "TerminalTool"),
    "SearchTool": (".search_tool", "SearchTool"),
    "ReadTool": (".file_tool", "ReadTool"),
    "WriteTool": (".file_tool", "WriteTool"),
    "EditTool": (".file_tool", "EditTool"),
    "MultiEditTool": (".file_tool", "MultiEditTool"),
    "build_file_tools": (".file_tool", "build_file_tools"),
    "RememberTool": (".memory_tool", "RememberTool"),
    "RecallTool": (".memory_tool", "RecallTool"),
    "ForgetTool": (".memory_tool", "ForgetTool"),
    "UpdateProfileTool": (".memory_tool", "UpdateProfileTool"),
    "build_memory_tools": (".memory_tool", "build_memory_tools"),
    "ListMcpServersTool": (".mcp_tool", "ListMcpServersTool"),
    "ConnectMcpServerTool": (".mcp_tool", "ConnectMcpServerTool"),
    "DisconnectMcpServerTool": (".mcp_tool", "DisconnectMcpServerTool"),
    "build_mcp_tools": (".mcp_tool", "build_mcp_tools"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """模块级延迟导入钩子：仅在实际访问属性时才加载对应子模块"""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # 动态导入子模块
        import importlib
        module = importlib.import_module(module_path, __package__)
        attr = getattr(module, attr_name)
        # 缓存到模块全局，下次访问直接返回，无需重复导入
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """保证dir()返回结果与原模块一致，兼容内省工具"""
    return list(sorted(set(globals().keys()) | set(_LAZY_IMPORTS.keys())))
