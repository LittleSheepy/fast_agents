"""轻量查询对象，用于统一构建过滤/排序/分页查询。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SortOrder(str, Enum):
    """排序方向枚举。"""
    ASC = "ASC"
    DESC = "DESC"


class FilterOp(str, Enum):
    """过滤操作符枚举。"""
    EQ = "="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "LIKE"
    IN = "IN"
    NOT_NULL = "IS NOT NULL"
    IS_NULL = "IS NULL"


class Filter(BaseModel):
    """查询过滤条件。"""
    field: str = Field(..., description="字段名")
    op: FilterOp = Field(..., description="操作符")
    value: Optional[Any] = Field(default=None, description="比较值（IS NULL/NOT_NULL时无需传）")


class Query(BaseModel):
    """查询对象，封装过滤/排序/分页参数。"""
    table_name: str = Field(..., description="表名")
    filters: Optional[list[Filter]] = Field(default=None, description="过滤条件列表，AND关系")
    sort_by: Optional[str] = Field(default=None, description="排序字段名")
    sort_order: SortOrder = Field(default=SortOrder.ASC, description="排序方向")
    limit: Optional[int] = Field(default=None, description="返回条数限制")
    offset: Optional[int] = Field(default=None, description="偏移量")
