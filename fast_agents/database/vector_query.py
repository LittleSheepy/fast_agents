"""向量查询相关的声明式模型。

本文件定义 VectorFilterOp / VectorFilter，用于向量元数据的条件过滤，
地位对应关系型查询里的 Filter / FilterOp。和 query.py 保持平行、互不依赖：
- 关系型 Filter 面向表列，支持 LIKE / IS NULL / BETWEEN；
- 向量 VectorFilter 面向 metadata 字段，只支持向量数据库常见的 EQ / NE / GT / GTE / LT / LTE / IN / AND / OR。

不强行复用 query.Filter 的原因：
1. 向量过滤支持 AND/OR 嵌套（元数据常见需求），关系型 Query 目前是平铺 AND；
2. 向量过滤不支持 LIKE / IS NULL（Chroma 等 SDK 本身就没有）；
3. 两边各自演进互不影响，命名/风格保持一致即可降低记忆成本。

::: module-author fast_agents
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field


class VectorFilterOp(str, Enum):
    """向量元数据过滤操作符（枚举值与 Chroma $-前缀操作符一一对应，便于编译）。"""

    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"
    AND = "$and"
    OR = "$or"


class VectorFilter(BaseModel):
    """单条向量元数据过滤条件，或 AND/OR 逻辑组合。

    两种用法：
    1. 叶子条件：field + op + value
       VectorFilter(field="kind", op=VectorFilterOp.EQ, value="fact")
    2. 逻辑组合：op=AND/OR + filters（嵌套 VectorFilter 列表）
       VectorFilter(op=VectorFilterOp.AND, filters=[
           VectorFilter(field="user_id", op=VectorFilterOp.EQ, value="u1"),
           VectorFilter(field="score",  op=VectorFilterOp.GTE, value=0.7),
       ])

    等价于 Chroma 的 where 格式：
        {"kind": {"$eq": "fact"}}
        {"$and": [{"user_id": {"$eq": "u1"}}, {"score": {"$gte": 0.7}}]}
    """

    field: str | None = Field(default=None, description="元数据字段名（叶子条件必填）")
    op: VectorFilterOp = Field(description="操作符")
    value: Any = Field(default=None, description="比较值（叶子条件必填；AND/OR 忽略）")
    filters: list["VectorFilter"] | None = Field(
        default=None, description="子条件列表（op 为 AND/OR 时必填，至少 2 个）"
    )


# 支持递归引用
VectorFilter.model_rebuild()


# 类型别名：方便上层调用处声明入参
VectorWhere = Union[dict[str, Any], VectorFilter, None]


__all__ = ["VectorFilterOp", "VectorFilter", "VectorWhere"]
