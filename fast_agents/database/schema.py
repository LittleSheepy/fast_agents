"""数据库表结构定义，用于幂等建表与迁移。

除声明式模型（``Column`` / ``Index`` / ``Schema``）外，还提供把模型解析成
SQL 片段的纯函数（不依赖任何数据库连接），供各后端客户端复用：

- ``parse_column_list``：把 ``list[Column]`` 解析成 ``(col_defs, index_stmts)``；
- ``parse_schema``：把 ``Schema`` 解析成 ``(col_defs, index_stmts)``（含显式 indexes）。
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .errors import DatabaseIdentifierError


class ColumnType(str, Enum):
    """数据库列类型枚举，与具体数据库方言无关。"""
    TEXT = "TEXT"
    INTEGER = "INTEGER"
    REAL = "REAL"
    BLOB = "BLOB"
    BOOLEAN = "BOOLEAN"
    VARCHAR = "VARCHAR"
    TIMESTAMP = "TIMESTAMP"


class Column(BaseModel):
    """数据库列定义。"""
    name: str = Field(..., description="列名")
    type: ColumnType = Field(..., description="列类型")
    primary_key: bool = Field(default=False, description="是否为主键")
    nullable: bool = Field(default=True, description="是否允许为空")
    default: Optional[Any] = Field(default=None, description="默认值")
    unique: bool = Field(default=False, description="是否唯一约束")
    index: bool = Field(default=False, description="是否创建普通索引")
    length: Optional[int] = Field(default=None, description="VARCHAR类型长度")


class Index(BaseModel):
    """数据库索引定义。"""
    name: str = Field(..., description="索引名")
    columns: list[str] = Field(..., description="索引列名列表")
    unique: bool = Field(default=False, description="是否唯一索引")


class Schema(BaseModel):
    """数据库表结构定义，用于幂等建表。"""
    table_name: str = Field(..., description="表名")
    columns: list[Column] = Field(..., description="列定义列表")
    indexes: Optional[list[Index]] = Field(default=None, description="索引定义列表")


# ---------------------------------------------------------------------------
# 标识符校验 / 引用（防 SQL 注入），供本模块与各后端客户端共用
# ---------------------------------------------------------------------------

# 合法标识符：字母/数字/下划线，且不以数字开头；用于防御表名/列名注入
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_ident(name: str, kind: str = "标识符") -> None:
    """检查表名/列名/索引名是合法标识符，防止 SQL 注入。"""
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise DatabaseIdentifierError(f"非法{kind}: {name!r}（只允许字母/数字/下划线，且不以数字开头）")

def sql_literal(value: Any, ctype: ColumnType) -> str:
    """把 Python 默认值转成 SQLite / MySQL 字面量。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    # 字符串：转义单引号
    s = str(value).replace("'", "''")
    return f"'{s}'"

def quote_ident(name: str, quote: str = '"', **kwargs) -> str:
    """给标识符加引用符（默认双引号 " ＝SQLite 方言, mysql 方言是反引号 `）。调用前必须先过 check_ident。"""
    return f"{quote}{name.replace(quote, quote * 2)}{quote}"

def check_and_quote_ident(name: str, kind: str = "标识符", quote: str = '"', **kwargs) -> str:
    check_ident(name, kind)
    return quote_ident(name, quote)

# ---------------------------------------------------------------------------
# Column / Schema -> SQL 片段解析（纯函数，不依赖数据库连接）
# ---------------------------------------------------------------------------

# SQLite 不原生支持 BOOLEAN / VARCHAR / TIMESTAMP，这里给出类型映射与建表时的展开规则
TYPE_TO_SQLITE: dict[ColumnType, str] = {
    ColumnType.TEXT:      "TEXT",
    ColumnType.INTEGER:   "INTEGER",
    ColumnType.REAL:      "REAL",
    ColumnType.BLOB:      "BLOB",
    ColumnType.BOOLEAN:   "INTEGER",   # SQLite 用 0/1 存布尔
    ColumnType.VARCHAR:   "TEXT",      # SQLite 无 VARCHAR，统一 TEXT；长度仅作校验/文档
    ColumnType.TIMESTAMP: "TEXT",      # 时间戳统一存 ISO 字符串（对齐 memory 模块习惯）
}


def column_sql(col: Column,
    type_map: Optional[dict[ColumnType, str]] = None,
    quote: str = '"',
    varchar_length: bool = False,
    auto_increment: bool = False,
    unique_keyword: str = "UNIQUE",
    **kwargs
) -> str:
    """把一个 Column 模型转成列定义片段（默认 SQLite 方言）。

    方言点全部由关键字参数注入，默认值保证 SQLite 侧行为逐字节不变：

    - ``type_map``：类型映射表（默认 ``TYPE_TO_SQLITE``）；
    - ``quote``：标识符引用符（默认双引号 ``"`` ＝SQLite 方言，MySQL 是反引号）；
    - ``unique_keyword``：唯一约束关键字（SQLite ``UNIQUE`` / MySQL ``UNIQUE KEY``）；
    - ``auto_increment``：主键整型列是否追加 ``AUTO_INCREMENT``（MySQL 需显式声明；
      SQLite 不追加，其 ``INTEGER PRIMARY KEY`` 已是 ROWID 别名）；
    - ``varchar_length``：``VARCHAR`` 指定 ``length`` 时是否展开成 ``VARCHAR(n)``
      （MySQL 需要；SQLite 无 VARCHAR，统一映射 TEXT，length 仅作校验/文档）。
    """
    check_ident(col.name, "列名")
    if type_map is None:
        type_map = TYPE_TO_SQLITE
    if varchar_length and col.type is ColumnType.VARCHAR and col.length is not None:
        type_sql = f"VARCHAR({int(col.length)})"
    else:
        type_sql = type_map[col.type]
    parts: list[str] = [quote_ident(col.name, quote), type_sql]

    if col.primary_key:
        parts.append("PRIMARY KEY")
        if auto_increment and col.type is ColumnType.INTEGER and col.name.lower() == "id":
            parts.append("AUTO_INCREMENT")

    if not col.nullable and not col.primary_key:
        parts.append("NOT NULL")

    if col.unique and not col.primary_key:
        parts.append(unique_keyword)

    if col.default is not None:
        parts.append("DEFAULT " + sql_literal(col.default, col.type))

    return " ".join(parts)


def column_index_stmt(table: str, col: Column,
                      if_not_exists: bool = True, quote: str = '"', **kwargs) -> Optional[str]:
    """为声明了 ``index=True`` 的普通列生成 CREATE INDEX 语句；不需要建索引时返回 None。

    主键列 / UNIQUE 列 SQLite 已自动建索引，跳过避免冗余。
    """
    if not (col.index and not col.primary_key and not col.unique):
        return None
    check_ident(col.name, "列名")
    idx_name = f"idx_{table}_{col.name}"
    if_not = "IF NOT EXISTS " if if_not_exists else ""
    return (f"CREATE INDEX {if_not}{quote_ident(idx_name, quote)} "
            f"ON {quote_ident(table, quote)} ({quote_ident(col.name, quote)})")


def index_stmt(table: str, idx: Index, if_not_exists: bool = True, quote: str = '"', **kwargs) -> str:
    """把一个显式 Index 模型转成 CREATE INDEX 语句。"""
    check_ident(idx.name, "索引名")
    for c in idx.columns:
        check_ident(c, "索引列名")
    cols_sql = ", ".join(quote_ident(c, quote) for c in idx.columns)
    unique = "UNIQUE " if idx.unique else ""
    ine = "IF NOT EXISTS " if if_not_exists else ""
    return (f"CREATE {unique}INDEX {ine}{quote_ident(idx.name, quote)} "
            f"ON {quote_ident(table, quote)} ({cols_sql})")


def parse_column_list(table: str, columns: list[Column], if_not_exists: bool = True, **kwargs) -> tuple[list[str], list[str]]:
    """解析 ``list[Column]``，返回 ``(col_defs, index_stmts)``。

    Args:
        table: 表名，用于拼接列级索引名 ``idx_{table}_{col}``。
        columns: Column 模型列表。

    Returns:
        col_defs: 每列的 SQL 定义片段列表（用于 CREATE TABLE 的括号内）。
        index_stmts: 列级 ``index=True`` 生成的 CREATE INDEX 语句列表。
    """
    check_ident(table, "表名")
    col_defs: list[str] = []
    index_stmts: list[str] = []
    for c in columns:
        if not isinstance(c, Column):
            raise TypeError(f"columns 列表元素必须是 Column，收到 {type(c)}")
        col_defs.append(column_sql(c, **kwargs))
        stmt = column_index_stmt(table, c, if_not_exists=if_not_exists, **kwargs)
        if stmt is not None:
            index_stmts.append(stmt)
    return col_defs, index_stmts


def parse_schema(schema: Schema, if_not_exists: bool = True, **kwargs) -> tuple[list[str], list[str]]:
    """解析 ``Schema``，返回 ``(col_defs, index_stmts)``。

    index_stmts 同时包含两部分（顺序：先列级、后显式）：
    1. ``schema.columns`` 中 ``index=True`` 的普通列生成的索引语句；
    2. ``schema.indexes`` 中显式声明的索引语句。
    """
    col_defs, index_stmts = parse_column_list(schema.table_name, schema.columns, if_not_exists=if_not_exists, **kwargs)
    if schema.indexes:
        for idx in schema.indexes:
            index_stmts.append(index_stmt(schema.table_name, idx, if_not_exists=if_not_exists, **kwargs))
    return col_defs, index_stmts
