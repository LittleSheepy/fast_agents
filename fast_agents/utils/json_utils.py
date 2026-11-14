from __future__ import annotations
import json
from typing import Any, Optional


# --- 粘贴你提供的函数定义开始 ---
def json_safe_serialize(obj: Any) -> Any:
    """把任意对象转成 JSON 可序列化的形式。"""
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # pydantic v2 BaseModel
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return json_safe_serialize(obj.model_dump(mode="json"))
            except Exception:
                return str(obj) if not isinstance(obj, type) else f"<{obj.__name__}>"
        # pydantic v1 兜底（.dict()）
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                return json_safe_serialize(obj.dict())
            except Exception:
                return str(obj) if not isinstance(obj, type) else f"<{obj.__name__}>"
        if isinstance(obj, dict):
            return {str(k): json_safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe_serialize(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError, OverflowError):
            return str(obj) if not isinstance(obj, type) else f"<{obj.__name__}>"
    except Exception:
        # 最兜底：哪怕str(obj)都抛异常，也返回占位符，绝对不丢整条Span
        try:
            return repr(obj)
        except Exception:
            return f"<unserializable_object_{type(obj).__name__}>"


def json_safe_dumps(obj, *args, **kwargs) -> str:
    # if isinstance(obj, str):
    #     return obj
    json_str = json.dumps(json_safe_serialize(obj), *args, **kwargs)
    return json_str

def json_safe_load(text: Optional[str | dict], *args, **kwargs) -> Any:
    """把 JSON 文本列反序列化回对象；NULL 返回 None。"""
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text, *args, **kwargs)
    except Exception as e:
        if isinstance(text, str):
            return text
        return None


if __name__ == '__main__':
    # 1. 准备测试数据：包含中文、特殊符号和嵌套结构
    # test_data = {
    #     "message": "你好，世界！🌍",
    #     "status": "成功",
    #     "details": {
    #         "city": "北京",
    #         "code": 200
    #     }
    # }
    test_data = "你好"
    print("=" * 30)
    print("测试 ensure_ascii=True (默认)")
    print("=" * 30)
    # ensure_ascii=True 是 json.dumps 的默认行为
    # 非 ASCII 字符（如中文、Emoji）会被转义为 \uXXXX 格式
    result_true = json_safe_dumps(test_data, ensure_ascii=True, indent=2)
    result_load = json_safe_load(result_true)

    print(result_true)

    print("\n" + "=" * 30)
    print("测试 ensure_ascii=False")
    print("=" * 30)
    # ensure_ascii=False 会保留原始字符
    # 前提是终端/文件编码支持 UTF-8，否则可能显示乱码
    result_false = json_safe_dumps(test_data, ensure_ascii=False, indent=2)
    result_false_load = json_safe_load(result_false)
    print(result_false)
    pass
