from typing import get_origin, get_args, Union, List, Dict, Literal, Optional
import enum


def _annotation_to_openai_type(annotation) -> str:
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
            # 混合类型，OpenAI 里不太稳
            return "string"

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
    param = {"type": "string", "description": ""}
    origin = get_origin(annotation)

    param["type"] = _annotation_to_openai_type(annotation)
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

            # 获取注解的实际类型字符串
            ptype = _annotation_to_openai_type(param.annotation)

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
        query: str,
        tags: List[int] = None,
        config: Dict[str, int] = None,
        mode: Literal["fast", "slow"] = "fast",
        mode_int: Literal[1, 5] = 1,
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
    parameters = demo2.para()
    print(parameters)



