"""工具基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
from typing import get_origin, get_args, Union, List, Dict, Literal, Optional
import enum


def _annotation_to_openai_type(annotation) -> str:
    import inspect
    """把 Python 类型注解转成 OpenAI JSON Schema 类型字符串"""
    origin = get_origin(annotation)

    # 处理 Optional[T] -> 实际是 Union[T, None]
    if origin is Union:
        args = get_args(annotation)
        # 去掉 NoneType 拿到真实类型
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _annotation_to_openai_type(non_none[0])
        return "string"

    # 处理 List[T] / list[T]
    if origin in (list, List):
        return "array"

    # 处理 Dict[K, V] / dict[K, V]
    if origin in (dict, Dict):
        return "object"

    # 处理 Literal["a", "b"] -> 转成 enum
    if origin is Literal:
        return "string"  # 类型本身是 string，值约束后面单独处理

    # 基本类型映射
    type_map = {
        str: "string", int: "integer", float: "number", bool: "boolean",
        list: "array", dict: "object", set: "array", tuple: "array", type(None): "null",
    }

    if annotation in type_map:
        return type_map[annotation]

    # 如果是 enum 类
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return "string"

    # 兜底
    return "string"

def _build_param(annotation, default):
    import inspect
    param = {"type": "string", "description": ""}
    origin = get_origin(annotation)

    if origin is Literal:
        values = list(get_args(annotation))
        param["enum"] = values

        # 判断 enum 里所有值是不是同一种类型
        types = {type(v) for v in values}
        if types == {str}:
            param["type"] = "string"
        elif types == {int}:
            param["type"] = "integer"
        elif types == {bool}:
            param["type"] = "boolean"
        elif types == {type(None)}:
            param["type"] = "null"
        else:
            # 混合类型，OpenAI 里不太稳
            param["type"] = "string"
        if default is not inspect.Parameter.empty:
            param["default"] = default
    else:
        # 获取注解的实际类型字符串
        param["type"] = _annotation_to_openai_type(annotation)
        if default is not inspect.Parameter.empty:
            param["default"] = default

    return param

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class ToolBase(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str, param_model=None):
        self.name = name
        self.description = description
        self.param_model = param_model
    
    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        """执行工具"""

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.get_parameters()]
        }

    def _get_param_description(self, param_name):
        import inspect
        # 方式一：从子类属性里取
        descriptions = getattr(self, "param_descriptions", {})
        if param_name in descriptions:
            return descriptions[param_name]

        # 方式二：从函数 docstring 里取，需要 docstring_parser
        try:
            from docstring_parser import parse
            doc = parse(inspect.getdoc(self.fun) or "")
            for p in doc.params:
                if p.arg_name == param_name:
                    return p.description or ""
        except Exception:
            pass

        return ""

    def _get_parameters_by_inspect(self):
        """run 的参数转为openai的形式-parameters"""
        import inspect
        signature = inspect.signature(self.run)
        parameters = signature.parameters

        parameter_dict = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in parameters.items():
            # 跳过 self
            if param_name == "self":
                continue

            # 构建参数描述
            parameter = _build_param(param.annotation, param.default)

            parameter["description"] = self._get_param_description(param_name)

            # 有默认值 -> 非必填；没默认值 -> 必填
            if param.default is not inspect.Parameter.empty:
                print(param_name, param.default)
            else:
                parameter_dict["required"].append(param_name)
                print(param_name, param.annotation)

            parameter_dict["properties"][param_name] = parameter

        return parameter_dict


    def _build_openai_like_parameters(self):
        """创建openai like参数"""
        if self.param_model is None:
            return self._get_parameters_by_inspect()

        schema = self.param_model.model_json_schema()

        # OpenAI 要求 "parameters" 而不是 "$schema" 等额外字段
        openai_like_parameters = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
        }
        if "required" in schema:
            openai_like_parameters["required"] = schema["required"]

        return openai_like_parameters

    def build_openai_like_tool(self) -> Dict[str, object]:
        openai_like_tool = {
            "type": "function",
            "function":{
                "name": self.name,
                "description": self.description,
                "parameters": self._build_openai_like_parameters(),
            }
        }
        return openai_like_tool
    
    def __str__(self) -> str:
        return f"Tool(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other):
        # 类型不对要返回 NotImplemented，交给 Python 去试 other.__ror__
        if isinstance(other, ToolBase):
            return [self.build_openai_like_tool()] + [other.build_openai_like_tool()]
        if isinstance(other, list):
            return [self.build_openai_like_tool()] + other
        return NotImplemented








"""
openai
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名，例如：上海、北京"
                }
            },
            "required": ["location"]
        }
    }
}

Anthropic
{
    "name": "bash",
    "description": "运行一个 shell 命令。",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
{
    "name": "bash",
    "description": "运行一个 shell 命令。",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令，例如 ls -la"
            }
        },
        "required": ["command"]
    }
}
"""










