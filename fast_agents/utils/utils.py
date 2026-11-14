from typing import Any, Optional


def to_str(value: Any) -> Optional[str]:
    """把枚举或其值统一成字符串；None 原样返回。"""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
