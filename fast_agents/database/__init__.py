"""Database 基础服务模块。

提供：
- 关系型后端：
  - ``SQLiteClient``：SQLite 客户端（初版唯一关系型实现），封装连接、CRUD、事务、声明式幂等建表；
  - ``Schema`` / ``Column`` / ``Index`` / ``ColumnType``：声明式表结构模型，用于 ``ensure_schema`` 建表；
  - ``Query`` / ``Filter`` / ``FilterOp`` / ``SortOrder``：轻量关系型查询对象；
  - ``DatabaseClientBase``：关系型客户端抽象基类，未来扩展 MySQL/Postgres 时继承即可。
- 向量型后端：
  - ``ChromaClient``：Chroma 向量数据库客户端（初版唯一向量实现），支持本地目录与远端 Server；
  - ``VectorClientBase``：向量客户端抽象基类，未来扩展 FAISS/Milvus/pgvector 等时继承即可；
  - ``VectorFilter`` / ``VectorFilterOp``：向量元数据过滤条件模型（对齐 Query/Filter 风格）。

零新增核心依赖：关系型能力仅用标准库 sqlite3 + pydantic；向量后端 chromadb 作为可选依赖
（``pip install fast-agents[chroma]``），未安装时仅 ChromaClient 无法使用，其它能力完全不受影响。

- 异常体系：
  - ``DatabaseError``：模块异常基类（**刻意继承 ``ValueError``**，兼容既有 ``except ValueError``）；
  - ``DatabaseConnectionError`` / ``DatabaseOperationalError`` / ``DatabaseValidationError``：
    分别对应连接层 / 执行层 / 校验层错误，上层可按类型精确分流。

允许一个后端**同时实现两套抽象**（如未来 Postgres + pgvector），上层通过 ``isinstance``
分别判断关系型 / 向量能力即可。
"""
from .chroma_client import ChromaClient
from .database_client_base import DatabaseClientBase
from .errors import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseOperationalError,
    DatabaseValidationError,
)
from .query import Filter, FilterOp, Query, SortOrder
from .schema import Column, ColumnType, Index, Schema
from .sqlite_client import SqlExpr, SQLiteClient
from .vector_client_base import VectorClientBase
from .vector_query import VectorFilter, VectorFilterOp

__all__ = [
    # 异常体系
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseOperationalError",
    "DatabaseValidationError",
    # 关系型客户端
    "SQLiteClient",
    "DatabaseClientBase",
    # 向量型客户端
    "ChromaClient",
    "VectorClientBase",
    # Schema / 建表声明模型
    "Schema",
    "Column",
    "ColumnType",
    "Index",
    # 关系型查询模型
    "Query",
    "Filter",
    "FilterOp",
    "SortOrder",
    # 向量型过滤模型
    "VectorFilter",
    "VectorFilterOp",
    # 写侧 SQL 表达式包装
    "SqlExpr",
]
