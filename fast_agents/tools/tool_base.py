"""工具基类 ToolBase

参考 LangChain BaseTool / StructuredTool、CrewAI BaseTool、LlamaIndex BaseTool 的共同设计：

命名约定（对齐 LangChain 生态）：
1. 四种定义方式共存：继承类 + args_schema / @tool 装饰器 / StructuredTool.from_function / 内省函数
2. 结构化返回值：run() 可返回 str 或 ToolResult；invoke()/ainvoke() 统一返回 ToolResult
3. 多 provider schema：to_openai_tool / to_anthropic_tool / to_gemini_tool（"to_" 前缀是 Python 序列化惯例）
4. 参数级描述：通过 Field(description=...) 或 docstring(Args:) 自动提取
5. 结果自动截断：max_output_chars 控制
6. 同步/异步双入口：run / arun （子类实现），invoke / ainvoke（带校验、错误处理、截断的公共入口）
7. hooks 机制：before_run / after_run / on_error
8. handle_tool_error 开关
"""

from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, create_model, ValidationError

from .tool_result import ToolResult

logger = logging.getLogger(__name__)


_PY_TYPE_TO_JSON: Dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object", type(None): "null",
}


def _annotation_to_json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _annotation_to_json_type(non_none[0])
        return "string"
    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"
    if origin is Literal:
        return "string"
    if annotation in _PY_TYPE_TO_JSON:
        return _PY_TYPE_TO_JSON[annotation]
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return "string"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return "object"
    return "string"


def _infer_param_schema_from_signature(func: Callable, skip: Sequence[str] = ("self",)) -> Dict[str, Any]:
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    arg_descriptions: Dict[str, str] = {}
    try:
        from docstring_parser import parse as _parse_doc
        parsed = _parse_doc(doc)
        for p in parsed.params:
            if p.description:
                arg_descriptions[p.arg_name] = p.description.strip()
    except ImportError:
        in_args = False
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                if stripped and ":" not in stripped:
                    in_args = False
                    continue
                if ":" in stripped:
                    name, _, desc = stripped.lstrip("-").partition(":")
                    arg_descriptions[name.strip()] = desc.strip()

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param_name, param in sig.parameters.items():
        if param_name in skip:
            continue
        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        prop: Dict[str, Any] = {"type": _annotation_to_json_type(annotation)}
        origin = get_origin(annotation)
        if origin is Literal:
            prop["enum"] = list(get_args(annotation))
        desc = arg_descriptions.get(param_name, "")
        if desc:
            prop["description"] = desc
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default
        properties[param_name] = prop
    return {"type": "object", "properties": properties, "required": required}


class ToolBase(ABC):
    """工具抽象基类。

    子类至少要设置 ``name`` / ``description`` 两个类属性，并实现 ``run()``。
    ``args_schema`` 是 pydantic ``BaseModel`` 子类，声明参数结构；不提供则从 ``run()``
    签名自动内省生成。
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    max_output_chars: int = 100_000
    handle_tool_error: bool = True
    # 注意：ToolBase 是普通 ABC，不是 pydantic BaseModel，
    # 这里不能写 Field(default_factory=...)，否则类属性会是个 FieldInfo 对象而非 list/dict。
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    return_direct: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        max_output_chars: Optional[int] = None,
        handle_tool_error: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        return_direct: Optional[bool] = None,
    ):
        self.name = name if name else type(self).name
        self.description = description if description else type(self).description
        self.args_schema = args_schema if args_schema is not None else type(self).args_schema
        self.max_output_chars = max_output_chars if max_output_chars else type(self).max_output_chars
        self.handle_tool_error = handle_tool_error if handle_tool_error is not None else type(self).handle_tool_error
        self.return_direct = return_direct if return_direct is not None else type(self).return_direct
        self.tags = list(tags) if tags is not None else list(type(self).tags or [])
        self.metadata = dict(metadata) if metadata is not None else dict(type(self).metadata or {})
        if not self.name:
            raise ValueError("工具必须有 name")
        if not self.description:
            raise ValueError(f"工具 '{self.name}' 必须有 description")

    @abstractmethod
    def run(self, *args, **kwargs: Any) -> Union[str, ToolResult]:
        """执行工具（子类必须实现）"""

    async def arun(self, **kwargs: Any) -> Union[str, ToolResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(self._run_safe, **kwargs))

    def _run_safe(self, **kwargs):
        return self.run(**kwargs)

    # ── 公共入口（同步/异步都带校验、错误捕获、结果截断）────────────────

    def invoke(self, tool_input: Optional[Union[str, Dict[str, Any]]] = None, **kwargs: Any) -> ToolResult:
        """执行工具。

        支持两种调用方式：
          1. 关键字参数：``tool.invoke(a=1, b="x")``；
          2. 单一 dict：``tool.invoke({"a": 1, "b": "x"})``（对齐 LangChain Runnable 风格）。
        """
        if tool_input is not None:
            if isinstance(tool_input, str):
                try:
                    import json as _json
                    parsed = _json.loads(tool_input) if tool_input.strip() else {}
                except Exception:
                    parsed = {}
                if not isinstance(parsed, dict):
                    return ToolResult.fail("工具参数必须是 JSON 对象")
                parsed.update(kwargs)
                kwargs = parsed
            elif isinstance(tool_input, dict):
                merged = dict(tool_input)
                merged.update(kwargs)
                kwargs = merged
            else:
                return ToolResult.fail(f"tool_input 必须是 dict 或 JSON 字符串，收到 {type(tool_input).__name__}")

        try:
            validated = self._validate_args(kwargs)
        except ValidationError as e:
            msg = f"参数错误：{self._format_validation_error(e)}"
            logger.warning("[tool:%s] %s", self.name, msg)
            return ToolResult.fail(msg)
        self._before_run(validated)
        try:
            raw = self.run(**validated)
        except Exception as e:
            self._on_error(e, validated)
            if not self.handle_tool_error:
                raise
            return ToolResult.fail(f"工具 '{self.name}' 执行失败：{e}")
        result = self._coerce_result(raw)
        if self.max_output_chars and len(result.output) > self.max_output_chars:
            truncated_output = result.output[: self.max_output_chars]
            result = ToolResult.truncated(truncated_output, len(result.output), self.max_output_chars)
        self._after_run(validated, result)
        return result

    async def ainvoke(self, tool_input: Optional[Union[str, Dict[str, Any]]] = None, **kwargs: Any) -> ToolResult:
        if tool_input is not None:
            if isinstance(tool_input, str):
                import json as _json
                parsed = _json.loads(tool_input) if tool_input.strip() else {}
                if not isinstance(parsed, dict):
                    return ToolResult.fail("工具参数必须是 JSON 对象")
                parsed.update(kwargs)
                kwargs = parsed
            elif isinstance(tool_input, dict):
                merged = dict(tool_input)
                merged.update(kwargs)
                kwargs = merged
            else:
                return ToolResult.fail(f"tool_input 必须是 dict 或 JSON 字符串，收到 {type(tool_input).__name__}")

        try:
            validated = self._validate_args(kwargs)
        except ValidationError as e:
            return ToolResult.fail(f"参数错误：{self._format_validation_error(e)}")
        self._before_run(validated)
        try:
            raw = await self.arun(**validated)
        except Exception as e:
            self._on_error(e, validated)
            if not self.handle_tool_error:
                raise
            return ToolResult.fail(f"工具 '{self.name}' 执行失败：{e}")
        result = self._coerce_result(raw)
        if self.max_output_chars and len(result.output) > self.max_output_chars:
            result = ToolResult.truncated(result.output[: self.max_output_chars], len(result.output), self.max_output_chars)
        self._after_run(validated, result)
        return result

    # ── 内部辅助 ─────────────────────────────────────────────

    def _validate_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.args_schema is not None:
            model = self.args_schema(**kwargs)
            return model.model_dump()
        return kwargs

    @staticmethod
    def _format_validation_error(e: ValidationError) -> str:
        msgs = []
        for err in e.errors():
            loc = ".".join(str(l) for l in err["loc"]) or "参数"
            msgs.append(f"「{loc}」{err['msg']}")
        return "；".join(msgs)

    @staticmethod
    def _coerce_result(raw: Any) -> ToolResult:
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult.ok(str(raw) if raw is not None else "")

    def _before_run(self, args: Dict[str, Any]) -> None:
        pass

    def _after_run(self, args: Dict[str, Any], result: ToolResult) -> None:
        if result.failed:
            logger.warning("[tool:%s] 执行失败: %s", self.name, result.error)
        else:
            logger.debug("[tool:%s] 完成，输出 %d 字符", self.name, len(result.output))

    def _on_error(self, error: Exception, args: Dict[str, Any]) -> None:
        logger.exception("[tool:%s] 异常", self.name, exc_info=error)

    # ── Schema 生成（property，对齐 LangChain args_schema 风格是属性，但这里保留 _get_json_schema） ──

    @property
    def input_schema(self) -> Dict[str, Any]:
        """返回当前工具的 JSON Schema（供 LLM function calling 使用）。"""
        if self.args_schema is not None:
            return _pydantic_to_json_schema(self.args_schema)
        return _infer_param_schema_from_signature(self.run, skip=("self",))

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def to_gemini_tool(self) -> Dict[str, Any]:
        schema = self.input_schema
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": schema["type"],
                "properties": schema["properties"],
                **({"required": schema["required"]} if schema.get("required") else {}),
            },
        }

    # ── 运算符 ───────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, ToolBase):
            return [self.to_openai_tool(), other.to_openai_tool()]
        if isinstance(other, list):
            return [self.to_openai_tool(), *other]
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return [*other, self.to_openai_tool()]
        return NotImplemented

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return f"<Tool '{self.name}'>"


def _pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """把 pydantic BaseModel 转成 LLM function calling 可用的 parameters JSON Schema。

    做两件事：
    1. 递归展开 $defs / $ref —— pydantic v2 用 JSON Schema 的引用机制复用嵌套模型
       （嵌套 BaseModel、Enum 都会生成 {"$ref": "#/$defs/Xxx"}），而 OpenAI 等 provider
       对 $ref 支持不完整（有的直接丢弃 $defs），必须内联成自包含 schema；
    2. 剥掉 title 等对模型无意义、只占 token 的字段。
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    # 把 pydantic 的「引用式」schema 压成自包含的「扁平」schema
    def _resolve(node: Any, seen: Tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            # 关键：只要出现 $ref 就展开，不能加 len(node) == 1 的限制——
            # pydantic 在字段带 description / default 时会生成
            # {"$ref": ..., "description": ...}，此时 len > 1，旧写法会漏掉，
            # 而 $defs 已被 pop 掉，最终留下悬空的 $ref，模型看不到嵌套结构。
            if "$ref" in node:
                ref = node["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref[len("#/$defs/"):]
                    if name in seen:  # 自引用 / 循环引用：留空对象，避免无限递归
                        return {}
                    target = defs.get(name, {})
                    siblings = {k: v for k, v in node.items() if k != "$ref"}
                    # 内联定义 + 合并兄弟键（兄弟键优先，如 description / default）
                    merged = {**target, **siblings} if isinstance(target, dict) else node
                    return _resolve(merged, seen + (name,))
                return node
            cleaned = {}
            for k, v in node.items():
                if k == "title":
                    continue
                cleaned[k] = _resolve(v, seen)
            return cleaned
        if isinstance(node, list):
            return [_resolve(x, seen) for x in node]
        return node

    return _resolve({
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
        **({"required": schema["required"]} if schema.get("required") else {}),
    })
