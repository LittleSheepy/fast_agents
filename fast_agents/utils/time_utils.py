r"""统一时间基准
fast_agents/fast_agents/utils/time_utils.py

"""
from datetime import datetime, timezone, timedelta
from typing import Optional

CST = timezone(timedelta(hours=8))   # 北京时间，固定 +8，无夏令时

def cstnow() -> datetime:
    """返回当前 CST 时间（带时区信息，aware）

    记忆模块中所有「现在」都取这里，替代过去的 ``datetime.now()``。
    """
    return datetime.now(CST)


def to_cst(value: Optional[datetime]) -> Optional[datetime]:
    """把 ``datetime`` 归一化为「带时区的 CST」

    - ``None`` 原样返回（便于直接用在 Optional 字段上）；
    - aware：转换到 CST（不改变绝对时刻，只统一表示形式）；
    - naive：抛 ``ValueError``。本项目只写 aware CST，收到 naive 属于编程错误，
      fail fast 暴露，而不是按本地时区静默猜测。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=CST)
        # raise ValueError(
        #     f"记忆模块只接受带时区(aware)的时间，收到 naive datetime：{value!r}；"
        #     f"请用 clock.cstnow() 生成时间"
        # )
    return value.astimezone(CST)


def to_cst_iso(value: Optional[datetime]) -> Optional[str]:
    """归一化为 CST 后转 ISO 字符串（带 ``+08:00``）；``None`` 原样返回"""
    normalized = to_cst(value)
    return normalized.isoformat() if normalized is not None else None

if __name__ == '__main__':
    print("")
    print(cstnow())