"""Database 基础服务模块。

提供：
- ``SQLiteClient``：SQLite 客户端（唯一实现），封装连接、CRUD、事务，以及**声明式、幂等建表**；
- ``Schema`` / ``Column`` / ``Index`` / ``ColumnType``：声明式表结构模型，用于 ``ensure_schema`` 建表；
- ``Query`` / ``Filter`` / ``FilterOp`` / ``SortOrder``：轻量查询对象（复杂查询时使用）；
- ``DatabaseClientBase``：客户端抽象基类，未来扩展 MySQL/Postgres 时继承即可。

零新增依赖、零抽象层，直接用标准库 sqlite3 + pydantic（pydantic 已是全框架基础依赖）。
未来再按需扩展 MySQL/Postgres/Chroma 等后端，上层代码保持不变。
"""
from .database_client_base import DatabaseClientBase
from .query import Filter, FilterOp, Query, SortOrder
from .schema import Column, ColumnType, Index, Schema
from .sqlite_client import SqlExpr, SQLiteClient

__all__ = [
    # 客户端
    "SQLiteClient",
    "DatabaseClientBase",
    # Schema / 建表声明模型
    "Schema",
    "Column",
    "ColumnType",
    "Index",
    # 查询模型（供上层按需使用）
    "Query",
    "Filter",
    "FilterOp",
    "SortOrder",
    # 写侧 SQL 表达式包装
    "SqlExpr",
]
