"""轻量查询/写入模型：过滤/排序/分页查询对象（Query/Filter/FilterOp/SortOrder）+ 写侧 SQL 片段包装（SqlExpr）。

SqlExpr 与后端无关（SQLite / MySQL 共用同一个类），故与查询模型同置于本模块，供
``DatabaseClientBase.update_where`` 及各后端顶层导入——避免抽象基类反向依赖具体 client 造成循环导入。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .errors import DatabaseValidationError


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
    columns: Optional[list[str]] = Field(default=None, description="要返回的列名列表；None 表示 SELECT *（全部列）")
    filters: Optional[list[Filter]] = Field(default=None, description="过滤条件列表，AND关系")
    sort_by: Optional[str | list[str]] = Field(default=None, description="排序字段名；单字段传 str，多字段传 list[str]")
    sort_order: SortOrder = Field(default=SortOrder.ASC, description="排序方向")
    limit: Optional[int] = Field(default=None, description="返回条数限制")
    offset: Optional[int] = Field(default=None, description="偏移量")


class SqlExpr:
    """SQL 表达式包装：标记某个「值」应作为 SQL 片段原样拼入，而不是走参数化。

    只在 ``update_where`` 的 SET 子句里使用，用于「自增 / 函数计算」这类无法用
    参数表达的写法，例如::

        db.update_where("memories", {"access_count": SqlExpr("access_count + 1")}, query)

    安全约定（重要）：
        - 表达式会被**原样拼进 SQL**，不做任何转义，因此**只能传程序里写死的固定片段**；
        - 严禁把任何外部输入（用户消息、工具返回值、HTTP 响应）拼进 ``SqlExpr``，
          否则等于把 SQL 注入的口子重新打开；
        - 只是普通值（数字 / 字符串 / None）请直接传，走参数化，不要用 ``SqlExpr`` 包。
    """

    __slots__ = ("sql",)

    def __init__(self, sql: str):
        if not isinstance(sql, str) or not sql.strip():
            raise DatabaseValidationError("SqlExpr 的表达式必须是非空字符串")
        self.sql = sql

    def __repr__(self) -> str:
        return f"SqlExpr({self.sql!r})"