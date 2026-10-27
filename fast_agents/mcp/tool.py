"""MCP 工具适配器 —— 两种形态：细粒度展开 / 粗粒度门脸工具。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, create_model

from ..tools.tool_base import ToolBase
from ..tools.tool_result import ToolResult
from .client import MCPClient
from .errors import MCPCallError
from .models import ToolInfo

logger = logging.getLogger(__name__)


def _schema_to_pydantic(schema: Dict[str, Any]) -> type:
    """把 JSON Schema 转成 pydantic BaseModel，作为 args_schema。"""
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: Dict[str, Any] = {}
    _TYPE_MAP = {
        "string": str, "integer": int, "number": float, "boolean": bool,
        "array": list, "object": dict,
    }
    for pname, pdef in props.items():
        json_type = pdef.get("type", "string")
        if isinstance(json_type, list):
            non_null = [t for t in json_type if t != "null"]
            py_type = _TYPE_MAP.get(non_null[0] if non_null else "string", str)
            if "null" in json_type and "null" not in required:
                from typing import Optional as _Opt
                py_type = _Opt[py_type]  # type: ignore
        else:
            py_type = _TYPE_MAP.get(json_type, str)
        if pname not in required:
            from typing import Optional as _Opt
            py_type = _Opt[py_type]  # type: ignore
        field_kwargs: Dict[str, Any] = {"description": pdef.get("description", "")}
        if "enum" in pdef:
            from typing import Literal
            literal_vals = tuple(pdef["enum"])
            py_type = Literal[literal_vals]  # type: ignore
        if "default" in pdef and pname not in required:
            field_kwargs["default"] = pdef["default"]
        elif pname not in required:
            field_kwargs["default"] = None
        fields[pname] = (py_type, Field(**field_kwargs))
    return create_model("MCPArgs", **fields)


class MCPToolAdapter(ToolBase):
    """细粒度适配器：每个 MCP 远端工具独立展开成一个 ToolBase。

    生产推荐用法：模型能看到每个参数的名称、描述、类型、必填、枚举，调用准确率最高。
    """

    tags = ["mcp"]

    def __init__(self, client: MCPClient, server_name: str, tool_info: ToolInfo, prefix_sep: str = "__"):
        self.client = client
        self.server_name = server_name
        self.tool_info = tool_info
        name = f"{server_name}{prefix_sep}{tool_info.name}"
        description = f"[MCP:{server_name}] {tool_info.description}"
        args_schema = _schema_to_pydantic(tool_info.input_schema) if tool_info.has_structured_input else None
        super().__init__(name=name, description=description, args_schema=args_schema)

    def run(self, **kwargs: Any) -> Union[str, ToolResult]:
        try:
            result = self.client.call_tool(self.tool_info.name, kwargs)
        except MCPCallError as e:
            return ToolResult.fail(str(e))
        text = result.as_text()
        if result.is_error:
            return ToolResult.fail(text)
        return ToolResult.ok(text, artifact={"content": [c for c in result.content if c.get("type") == "text"]})


# ── 粗粒度工具参数模型（A1 方案：结构化嵌套）──────────────────────

class MCPServerCallInput(BaseModel):
    action: str = Field(
        description="动作类型：list_tools 查看该 MCP 服务器提供了哪些工具；call_tool 调用其中的某个工具",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="action=call_tool 时必填：要调用的子工具名",
    )
    arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="action=call_tool 时必填：传给子工具的参数字典（key/value 形式）；"
                    "如果子工具只接收一个 input 字符串参数，传 {'input': '...'} 即可",
    )


class MCPServerTool(ToolBase):
    """粗粒度门脸工具：一个 MCP server 对应一个工具，模型通过 action 选择查看子工具或调用子工具。

    与细粒度 MCPToolAdapter 的区别：
    - 细粒度（expand=fine）：每个远端工具独立一个 ToolBase，模型直接点名调用，参数 schema 完整；
    - 粗粒度（expand=coarse）：每个 server 一个门脸工具，模型先 list_tools 看有什么，再 call_tool 调用，
      适合工具数量多、不想一次性铺开的场景（节省工具表长度）。
    """

    tags = ["mcp"]
    args_schema = MCPServerCallInput

    def __init__(self, client: MCPClient, server_name: str, tools: List[ToolInfo], prefix_sep: str = "__"):
        self.client = client
        self.server_name = server_name
        self._tools_list = tools
        self._tools_index = {t.name: t for t in tools}
        name = f"mcp{prefix_sep}{server_name}"
        tool_names = ", ".join(t.name for t in tools) if tools else "(无)"
        description = (
            f"[MCP:{server_name}] 门脸工具：封装了该 MCP 服务器提供的 {len(tools)} 个子工具。"
            f"先用 action='list_tools' 查看子工具清单，再用 action='call_tool' 调用指定子工具。"
            f"可用子工具：{tool_names}"
        )
        super().__init__(name=name, description=description)

    def _list_tools_text(self) -> str:
        if not self._tools_list:
            return f"MCP 服务器 {self.server_name} 当前没有可用工具。"
        lines = [f"## {self.server_name} 提供的子工具：\n"]
        for t in self._tools_list:
            schema = t.input_schema or {}
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []) or [])
            param_lines = []
            for pname, pdef in props.items():
                mark = "必填" if pname in required else "可选"
                desc = pdef.get("description", "")
                type_name = pdef.get("type", "any")
                param_lines.append(f"    - {pname} ({type_name}, {mark}): {desc}")
            params_text = "\n" + "\n".join(param_lines) if param_lines else ""
            lines.append(f"- **{t.name}**：{t.description}{params_text}")
        return "\n".join(lines)

    def run(self, action: str, tool_name: Optional[str] = None,
            arguments: Optional[Dict[str, Any]] = None) -> Union[str, ToolResult]:
        action = (action or "").strip().lower()

        if action == "list_tools":
            return ToolResult.ok(self._list_tools_text())

        if action != "call_tool":
            return ToolResult.fail(
                f"不支持的 action: '{action}'，只能是 'list_tools' 或 'call_tool'"
            )

        if not tool_name:
            return ToolResult.fail("action=call_tool 时必须提供 tool_name；可先调用 list_tools 查看可用子工具")

        if tool_name not in self._tools_index:
            available = ", ".join(self._tools_index.keys()) or "(无)"
            return ToolResult.fail(f"子工具 '{tool_name}' 不存在于 {self.server_name}；可用：{available}")

        info = self._tools_index[tool_name]
        args: Dict[str, Any] = arguments or {}
        # 如果该子工具不是结构化入参（只有 input 字段），且 arguments 传的是裸字符串，自动包装
        if not info.has_structured_input and isinstance(args, str):
            args = {"input": args}

        try:
            result = self.client.call_tool(tool_name, args)
        except MCPCallError as e:
            return ToolResult.fail(str(e))
        text = result.as_text()
        if result.is_error:
            return ToolResult.fail(text)
        return ToolResult.ok(text)
