"""StructuredTool —— 把任意函数变成工具 + @tool 装饰器"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, Type, Union

from pydantic import BaseModel, create_model, Field

from .tool_base import ToolBase
from .tool_result import ToolResult


def _get_type_hints_safe(func: Callable) -> dict:
    try:
        from typing import get_type_hints
        return get_type_hints(func)
    except Exception:
        return {}


def _create_args_schema_from_function(func: Callable, name: str, skip: tuple = ("self",)) -> Type[BaseModel]:
    """从函数签名 + docstring 自动生成 args_schema (pydantic BaseModel)。"""
    sig = inspect.signature(func)
    hints = _get_type_hints_safe(func)
    doc = inspect.getdoc(func) or ""
    arg_descriptions: dict[str, str] = {}
    try:
        from docstring_parser import parse as _parse
        parsed = _parse(doc)
        for p in parsed.params:
            if p.description:
                arg_descriptions[p.arg_name] = p.description.strip()
    except ImportError:
        pass

    fields: dict[str, tuple] = {}
    for p_name, p in sig.parameters.items():
        if p_name in skip:
            continue
        annotation = hints.get(p_name, p.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        if p.default is inspect.Parameter.empty:
            fields[p_name] = (annotation, Field(description=arg_descriptions.get(p_name, "")))
        else:
            fields[p_name] = (annotation, Field(default=p.default, description=arg_descriptions.get(p_name, "")))
    return create_model(f"{name}_Args", **fields)


class StructuredTool(ToolBase):
    """把任意 Python 函数包装成 Tool，无需继承。"""

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[Callable] = None,
        max_output_chars: int = 100_000,
        handle_tool_error: bool = True,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        return_direct: bool = False,
    ):
        super().__init__(
            name=name, description=description, args_schema=args_schema,
            max_output_chars=max_output_chars, handle_tool_error=handle_tool_error,
            tags=tags or [], metadata=metadata or {}, return_direct=return_direct,
        )
        self.func = func
        self.coroutine = coroutine

    def run(self, **kwargs: Any) -> Union[str, ToolResult]:
        return self.func(**kwargs)

    async def arun(self, **kwargs: Any) -> Union[str, ToolResult]:
        if self.coroutine is not None:
            return await self.coroutine(**kwargs)
        return await super().arun(**kwargs)

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[Callable] = None,
        max_output_chars: int = 100_000,
        handle_tool_error: bool = True,
        tags: Optional[list[str]] = None,
        return_direct: bool = False,
    ) -> "StructuredTool":
        """从函数创建 StructuredTool。可作为装饰器使用。"""
        def _make(f: Callable) -> "StructuredTool":
            tool_name = name or f.__name__
            doc = inspect.getdoc(f) or ""
            tool_desc = description or (doc.split("\n\n")[0].strip() if doc else f"{tool_name} 工具")
            schema = args_schema or _create_args_schema_from_function(f, tool_name)
            return cls(func=f, name=tool_name, description=tool_desc, args_schema=schema,
                       coroutine=coroutine, max_output_chars=max_output_chars,
                       handle_tool_error=handle_tool_error, tags=tags, return_direct=return_direct)
        if func is not None:
            return _make(func)
        return _make


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    max_output_chars: int = 100_000,
    tags: Optional[list[str]] = None,
    return_direct: bool = False,
):
    """@tool 装饰器：把普通函数直接变成 StructuredTool。"""
    def _wrap(f: Callable) -> StructuredTool:
        return StructuredTool.from_function(
            f, name=name, description=description, args_schema=args_schema,
            max_output_chars=max_output_chars, tags=tags, return_direct=return_direct,
        )
    if func is not None:
        return _wrap(func)
    return _wrap
