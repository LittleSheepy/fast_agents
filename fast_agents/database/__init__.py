"""Database 基础服务模块。

提供：
- 关系型后端：
  - ``SQLiteClient``：SQLite 客户端（零依赖默认后端），封装连接、CRUD、事务、声明式幂等建表；
  - ``MySQLClient``：MySQL 客户端（可选依赖 ``pymysql``，接口与 SQLiteClient 100% 对齐，上层零代码改动即可切换）；
  - ``Schema`` / ``Column`` / ``Index`` / ``ColumnType``：声明式表结构模型，用于 ``ensure_schema`` 建表；
  - ``Query`` / ``Filter`` / ``FilterOp`` / ``SortOrder``：轻量关系型查询对象；
  - ``DatabaseClientBase``：关系型客户端抽象基类，未来扩展 Postgres 等时继承即可。
- 向量型后端：
  - ``ChromaClient``：Chroma 向量数据库客户端（可选依赖，``pip install fast-agents[chroma]``），支持本地目录与远端 Server；
  - ``VectorClientBase``：向量客户端抽象基类，未来扩展 FAISS/Milvus/pgvector 等时继承即可；
  - ``VectorFilter`` / ``VectorFilterOp``：向量元数据过滤条件模型（对齐 Query/Filter 风格）。

依赖约定：
- 核心能力仅用标准库 + pydantic，SQLiteClient 默认可用；
- MySQLClient 为可选后端，使用前需 ``pip install pymysql``或``pip install fast-agents[pymysql]``（未安装时**构造** MySQLClient 会抛出明确安装提示；**顶层导入**模块不会触发 pymysql，不影响其他后端）；
- ChromaClient 为可选后端，使用前需 ``pip install chromadb``或``pip install fast-agents[chroma]``。

- 异常体系：
  - ``DatabaseError``：模块异常基类（**刻意继承 ``ValueError``**，兼容既有 ``except ValueError``）；
  - ``DatabaseConnectionError`` / ``DatabaseOperationalError`` / ``DatabaseValidationError``：
    分别对应连接层 / 执行层 / 校验层错误，上层可按类型精确分流。

允许一个后端**同时实现两套抽象**（如未来 Postgres + pgvector），上层通过 ``isinstance``
分别判断关系型 / 向量能力即可。

所有关系型后端（SQLite/MySQL/...）对外暴露完全一致的 API：
- 同样的 Schema/Column/Index 声明式建表；
- 同样的 Query/Filter 谓词查询，支持 = != > >= < <= LIKE IN IS NULL 等操作；
- 同样的事务嵌套 with db: 语义；
- 同样的 insert/batch_insert/upsert/update/delete 批量操作。
上层业务代码（memory/trace/logs）只依赖 DatabaseClientBase 抽象，切换后端无需改动一行业务代码。
"""
from .chroma_client import ChromaClient
from .database_client_base import DatabaseClientBase
from .mysql_client import MySQLClient
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
    "MySQLClient",
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
    # 写侧 SQL 表达式包装（SQLite/MySQL 共用同一个类，跨后端 isinstance 可识别）
    "SqlExpr",
]
