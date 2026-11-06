from __future__ import annotations

import asyncio
import enum
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Type, Union, get_args, get_origin, get_type_hints)

from pydantic import BaseModel, Field, create_model, ValidationError

from typing import get_origin, get_args, Union, List, Dict, Literal, Optional
import enum



_PY_TYPE_TO_JSON: Dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object", type(None): "null",
}


def _annotation_to_json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    # 处理 Optional[T] -> 实际是 Union[T, None]
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]     # 去掉 NoneType 拿到真实类型
        if non_none:
            return _annotation_to_json_type(non_none[0])
        return "string"
    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"
    if origin is Literal:
        values = list(get_args(annotation))
        # 判断 enum 里所有值是不是同一种类型
        types = {type(v) for v in values}
        if types == {str}:
            return "string"
        elif types == {int}:
            return "integer"
        elif types == {bool}:
            return "boolean"
        elif types == {type(None)}:
            return "null"
        else:
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


def _build_param(annotation, default):
    param = {"type": "string", "description": ""}
    origin = get_origin(annotation)

    param["type"] = _annotation_to_json_type(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        param["enum"] = values
    # 走你之前的类型映射
    if default is not inspect.Parameter.empty:
        param["default"] = default

    return param




class Demo:
    def __init__(self):
        pass

    def fun(self, *args, **kwargs):
        pass

    def _get_param_description(self, param_name):
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

    def para(self):
        """fun 的参数转为openai的形式-parameters"""
        signature = inspect.signature(self.fun)
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


class Demo2(Demo):
    def fun(        self,
        # query: str,
        mode_int: Literal[1, 5] = 1,
        mode: Literal["fast", "slow"] = "fast",
        tags: List[int] = None,
        config: Dict[str, int] = None,
        count: Optional[int] = 10):
        """
        hanshu
        :param query:qqqqqq
        :param tags:ttttttttt
        :param config:ccccc
        :param mode:
        :param mode_int:
        :param count:
        :return:
        """
        pass

if __name__ == '__main__':
    import inspect

    demo2 = Demo2()
    parameters = _infer_param_schema_from_signature(demo2.fun)
    print(parameters)



