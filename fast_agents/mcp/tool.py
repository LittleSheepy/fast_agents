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
    """把 JSON Schema 转成 pydantic BaseModel，作为 args_schema。
    增强支持：array items定义、嵌套object递归转换、enum、默认值（含falsy值）、数值/字符串/数组校验、anyOf/oneOf含null转Optional。
    不支持的Schema类型降级为Any，不抛异常不阻塞工具注册。
    """
    # Schema缓存，避免相同结构重复创建模型类
    _schema_cache: Dict[str, type] = {}

    def _convert_type(prop: Dict[str, Any], name_hint: str) -> Any:
        """递归转换单个属性的JSON Schema类型为Python/Pydantic类型"""
        # 枚举类型优先处理：直接生成Literal
        if "enum" in prop:
            from typing import Literal
            enum_values = tuple(prop["enum"])
            return Literal[enum_values]  # type: ignore

        # 处理anyOf/oneOf：检测是否包含null类型，自动转Optional
        for union_key in ("anyOf", "oneOf"):
            if union_key in prop:
                non_null_types = []
                has_null = False
                for sub in prop[union_key]:
                    sub_type = sub.get("type")
                    if sub_type == "null":
                        has_null = True
                    else:
                        non_null_types.append(_convert_type(sub, name_hint))
                if not non_null_types:
                    return type(None)
                # 多个非null类型取第一个（简化处理，不支持复杂联合类型校验）
                union_type = non_null_types[0]
                if has_null:
                    return Optional[union_type]
                return union_type

        json_type = prop.get("type", "string")
        if isinstance(json_type, list):
            non_null = [t for t in json_type if t != "null"]
            py_type = _convert_type({"type": non_null[0] if non_null else "string"}, name_hint)
            if "null" in json_type:
                return Optional[py_type]
            return py_type

        _TYPE_MAP = {
            "string": str, "integer": int, "number": float, "boolean": bool,
        }
        if json_type in _TYPE_MAP:
            return _TYPE_MAP[json_type]

        # 数组类型：支持items定义元素类型
        if json_type == "array":
            if "items" in prop:
                item_type = _convert_type(prop["items"], f"{name_hint}_item")
                return List[item_type]
            return list

        # 对象类型：递归创建嵌套Pydantic模型
        if json_type == "object":
            # 优先走缓存，避免相同结构重复生成模型。
            # v3.3.4 缺陷修复：缓存键由 str(properties) 改为整个 prop 的规范化全量序列化——
            # 此前键不含同级 required 及对象自身约束，同构但必填集合不同的两个嵌套对象
            # 会复用同一模型，必填字段被静默降级为可选（双向错乱，探针 P-C 实证）。
            cache_key = json.dumps(prop, sort_keys=True, ensure_ascii=False, default=str)
            if cache_key in _schema_cache:
                return _schema_cache[cache_key]
            sub_properties: Dict[str, Dict[str, Any]] = prop.get("properties", {}) or {}
            sub_required = set(prop.get("required", []) or [])
            sub_fields: Dict[str, Any] = {}
            for sub_name, sub_prop in sub_properties.items():
                sub_py_type = _convert_type(sub_prop, f"{name_hint}_{sub_name}")
                sub_field_kwargs: Dict[str, Any] = {"description": sub_prop.get("description", "")}
                # 收集校验约束
                if "minimum" in sub_prop:
                    sub_field_kwargs["ge"] = sub_prop["minimum"]
                if "maximum" in sub_prop:
                    sub_field_kwargs["le"] = sub_prop["maximum"]
                if "exclusiveMinimum" in sub_prop:
                    sub_field_kwargs["gt"] = sub_prop["exclusiveMinimum"]
                if "exclusiveMaximum" in sub_prop:
                    sub_field_kwargs["lt"] = sub_prop["exclusiveMaximum"]
                if "minLength" in sub_prop:
                    sub_field_kwargs["min_length"] = sub_prop["minLength"]
                if "maxLength" in sub_prop:
                    sub_field_kwargs["max_length"] = sub_prop["maxLength"]
                if "pattern" in sub_prop:
                    sub_field_kwargs["pattern"] = sub_prop["pattern"]
                if "minItems" in sub_prop:
                    # v3.3.7：min_items/max_items 是 pydantic V2 废弃别名（每次转换刷
                    # DeprecationWarning，V3 将移除），改用等效的 min_length/max_length，校验行为一致
                    sub_field_kwargs["min_length"] = sub_prop["minItems"]
                if "maxItems" in sub_prop:
                    sub_field_kwargs["max_length"] = sub_prop["maxItems"]
                # 处理默认值和必填性：修正布尔/0/空字符串等falsy默认值被误判的问题
                if sub_name in sub_required:
                    sub_field_info = Field(**sub_field_kwargs)
                    sub_fields[sub_name] = (sub_py_type, sub_field_info)
                else:
                    if "default" in sub_prop:
                        sub_field_kwargs["default"] = sub_prop["default"]
                    else:
                        sub_field_kwargs["default"] = None
                    sub_field_info = Field(**sub_field_kwargs)
                    sub_fields[sub_name] = (Optional[sub_py_type], sub_field_info)
            # 创建嵌套模型，用唯一名称避免冲突
            nested_model_name = f"MCPArgs_{name_hint}"
            nested_model = create_model(nested_model_name, **sub_fields)
            _schema_cache[cache_key] = nested_model
            return nested_model

        # 不认识的类型，降级为Any，打debug日志不阻塞
        logger.debug("不支持的JSON Schema类型 %s，字段 %s 降级为Any", json_type, name_hint)
        return Any

    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: Dict[str, Any] = {}

    for pname, pdef in props.items():
        py_type = _convert_type(pdef, pname)
        field_kwargs: Dict[str, Any] = {"description": pdef.get("description", "")}
        # 收集校验约束
        if "minimum" in pdef:
            field_kwargs["ge"] = pdef["minimum"]
        if "maximum" in pdef:
            field_kwargs["le"] = pdef["maximum"]
        if "exclusiveMinimum" in pdef:
            field_kwargs["gt"] = pdef["exclusiveMinimum"]
        if "exclusiveMaximum" in pdef:
            field_kwargs["lt"] = pdef["exclusiveMaximum"]
        if "minLength" in pdef:
            field_kwargs["min_length"] = pdef["minLength"]
        if "maxLength" in pdef:
            field_kwargs["max_length"] = pdef["maxLength"]
        if "pattern" in pdef:
            field_kwargs["pattern"] = pdef["pattern"]
        if "minItems" in pdef:
            # v3.3.7：同上，废弃别名 min_items/max_items 换为等效的 min_length/max_length
            field_kwargs["min_length"] = pdef["minItems"]
        if "maxItems" in pdef:
            field_kwargs["max_length"] = pdef["maxItems"]
        # 必填/非必填/默认值处理：修正布尔/0/空字符串等合法默认值被误判为None的问题
        if pname in required:
            field_info = Field(**field_kwargs)
            fields[pname] = (py_type, field_info)
        else:
            if "default" in pdef:
                field_kwargs["default"] = pdef["default"]
            else:
                field_kwargs["default"] = None
            field_info = Field(**field_kwargs)
            fields[pname] = (Optional[py_type], field_info)

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
        # artifact 完整透传 MCP 原始 content 列表，并附 images 快捷键，
        # 上层（react_agent 等）可据此把图片以 base64 形式追加到多模态消息，
        # 不破坏纯文本消费者——它们只读 output 字符串即可。
        return ToolResult.ok(
            text,
            artifact={
                "content": result.raw_content(),
                "images": result.images(),
            },
        )


# ── 粗粒度工具参数模型（A1 方案：结构化嵌套）──────────────────────

class MCPServerCallInput(BaseModel):
    action: str = Field(
        description="动作类型："
                    "list_tools 查看工具列表；call_tool 调用工具；"
                    "list_resources 查看资源列表；read_resource 读取资源；"
                    "list_prompts 查看提示模板列表；get_prompt 渲染提示模板",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="action=call_tool 时必填：要调用的子工具名",
    )
    arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="action=call_tool 时：传给子工具的参数字典；"
                    "action=get_prompt 时：传给提示模板的参数字典（可选）；",
    )
    resource_uri: Optional[str] = Field(
        default=None,
        description="action=read_resource 时必填：要读取的资源 URI",
    )
    prompt_name: Optional[str] = Field(
        default=None,
        description="action=get_prompt 时必填：要渲染的提示模板名",
    )


class MCPServerTool(ToolBase):
    """粗粒度门脸工具：一个 MCP server 对应一个工具，模型通过 action 选择查看子工具或调用子工具。

    与细粒度 MCPToolAdapter 的区别：
    - 细粒度（expand=fine）：每个远端工具独立一个 ToolBase，模型直接点名调用，参数 schema 完整；
    - 粗粒度（expand=coarse）：每个 server 一个门脸工具，模型先 list_tools/list_resources/list_prompts 看有什么，
      再 call_tool/read_resource/get_prompt 调用，适合工具/资源/模板数量多、不想一次性铺开的场景。
    """

    tags = ["mcp"]
    args_schema = MCPServerCallInput

    def __init__(self, client: MCPClient, server_name: str, tools: List[ToolInfo], prefix_sep: str = "__"):
        self.client = client
        self.server_name = server_name
        self._tools_list = tools
        self._tools_index = {t.name: t for t in tools}
        name = f"mcp{prefix_sep}{server_name}"
        # 收集服务器能力提示
        caps_parts = [f"{len(tools)} 个子工具"]
        if client.initialize_result:
            caps = client.initialize_result.capabilities
            if caps.resources is not None:
                caps_parts.append("资源读取能力")
            if caps.prompts is not None:
                caps_parts.append("提示模板能力")
        caps_text = "、".join(caps_parts)
        description = (
            f"[MCP:{server_name}] 门脸工具：封装了该 MCP 服务器提供的 {caps_text}。"
            f"支持的 action：list_tools/call_tool（工具）、list_resources/read_resource（资源）、"
            f"list_prompts/get_prompt（提示模板）。"
        )
        super().__init__(name=name, description=description)

    def update_tools(self, tools: List[ToolInfo]) -> None:
        """v3.3.6：刷新内部工具快照与索引（MCPRegistry 收到 tools/list_changed 时调用）。

        此前粗粒度门脸只存注册时的旧快照，服务器新增子工具后永远报「不存在」。"""
        self._tools_list = list(tools)
        self._tools_index = {t.name: t for t in self._tools_list}

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

    def _list_resources_text(self) -> str:
        try:
            resources = self.client.list_resources()
            templates = self.client.list_resource_templates()
        except Exception as e:
            return f"获取资源列表失败：{e}"
        lines = [f"## {self.server_name} 提供的资源：\n"]
        if resources:
            for r in resources:
                mime = f" ({r.mimeType})" if r.mimeType else ""
                size = f" [{r.size} bytes]" if r.size is not None else ""
                lines.append(f"- **{r.name}**：`{r.uri}`{mime}{size} - {r.description}")
        if templates:
            lines.append("\n### 参数化资源模板：\n")
            for t in templates:
                mime = f" ({t.mimeType})" if t.mimeType else ""
                lines.append(f"- **{t.name}**：`{t.uriTemplate}`{mime} - {t.description}")
        if not resources and not templates:
            lines.append("（无可用资源）")
        return "\n".join(lines)

    def _list_prompts_text(self) -> str:
        try:
            prompts = self.client.list_prompts()
        except Exception as e:
            return f"获取提示模板列表失败：{e}"
        lines = [f"## {self.server_name} 提供的提示模板：\n"]
        if not prompts:
            lines.append("（无可用提示模板）")
            return "\n".join(lines)
        for p in prompts:
            args_lines = []
            for arg in p.arguments:
                mark = "必填" if arg.required else "可选"
                args_lines.append(f"    - {arg.name} ({mark}): {arg.description}")
            args_text = "\n" + "\n".join(args_lines) if args_lines else ""
            lines.append(f"- **{p.name}**：{p.description}{args_text}")
        return "\n".join(lines)

    def run(self, action: str, tool_name: Optional[str] = None,
            arguments: Optional[Dict[str, Any]] = None,
            resource_uri: Optional[str] = None,
            prompt_name: Optional[str] = None) -> Union[str, ToolResult]:
        action = (action or "").strip().lower()

        # ── 工具相关 action ──
        if action == "list_tools":
            return ToolResult.ok(self._list_tools_text())

        # ── 资源相关 action ──
        if action == "list_resources":
            return ToolResult.ok(self._list_resources_text())

        if action == "read_resource":
            if not resource_uri:
                return ToolResult.fail("action=read_resource 时必须提供 resource_uri；可先调用 list_resources 查看可用资源")
            try:
                result = self.client.read_resource(resource_uri)
            except Exception as e:
                return ToolResult.fail(f"读取资源失败：{e}")
            return ToolResult.ok(result.as_text())

        # ── 提示模板相关 action ──
        if action == "list_prompts":
            return ToolResult.ok(self._list_prompts_text())

        if action == "get_prompt":
            if not prompt_name:
                return ToolResult.fail("action=get_prompt 时必须提供 prompt_name；可先调用 list_prompts 查看可用模板")
            prompt_args = arguments if isinstance(arguments, dict) else None
            try:
                result = self.client.get_prompt(prompt_name, arguments=prompt_args)
            except Exception as e:
                return ToolResult.fail(f"获取提示模板失败：{e}")
            return ToolResult.ok(result.as_text(), artifact={"messages": [m.model_dump() for m in result.messages]})

        # ── 调用工具 action ──
        if action != "call_tool":
            return ToolResult.fail(
                f"不支持的 action: '{action}'，可选值：list_tools/call_tool/list_resources/read_resource/list_prompts/get_prompt"
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
        return ToolResult.ok(
            text,
            artifact={
                "content": result.raw_content(),
                "images": result.images(),
            },
        )
