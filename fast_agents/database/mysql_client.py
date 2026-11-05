"""MySQL client —— fast_agents MySQL 数据库客户端，接口与 SQLiteClient 100% 对齐。

设计参考主流框架的 MySQL 方言层：**SQLAlchemy**（方言与模型分离——Schema 声明与具体后端
无关，占位符 / 类型映射 / 自增 / Upsert 语法各自适配）、**Peewee**（单文件轻量客户端 + 上下文
事务）、**PyMySQL**（DB-API 2.0 驱动，参数化统一用 ``%s``）。只借鉴「方言适配 + 统一契约」
的思路，**不引入任何 ORM 或重型依赖**：本模块只用可选驱动 ``pymysql`` + 全框架已有的 ``pydantic``。

可选依赖：使用前需安装 pymysql：``pip install pymysql``（或 ``pip install fast-agents[mysql]``）。
**导入本模块不会触发 pymysql 依赖**（延迟到 ``MySQLClient(...)`` 构造时才 import）；
未安装 pymysql 时，只有真正构造 MySQLClient 才会抛出友好提示，SQLite/Chroma 等其他后端完全不受影响。

用法与 SQLiteClient 完全一致:
    from fast_agents.database import MySQLClient, Schema, Column, ColumnType, Index

    db = MySQLClient(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="your_password",
        database="fast_agents",
    )

    # 方式一：最简，dict 描述列
    db.ensure_table("users", {
        "id":   "INT PRIMARY KEY AUTO_INCREMENT",
        "name": "VARCHAR(255) NOT NULL",
        "age":  "INT",
    })

    # 方式二：声明式 Schema（推荐），自动建索引（与 SQLite 完全相同的写法）
    db.ensure_schema(Schema(
        table_name="orders",
        columns=[
            Column(name="id",        type=ColumnType.INTEGER, primary_key=True),
            Column(name="user_id",   type=ColumnType.INTEGER, nullable=False, index=True),
            Column(name="amount",    type=ColumnType.REAL,    nullable=False, default=0),
            Column(name="created_at",type=ColumnType.TIMESTAMP, nullable=False),
        ],
        indexes=[
            Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
        ],
    ))

    db.insert("users", {"name": "Alice"})
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, TYPE_CHECKING

from .database_client_base import DatabaseClientBase
from .query import Filter, FilterOp, Query, SortOrder
from .schema import (
    Column,
    ColumnType,
    Index,
    Schema,
    check_ident as _check_ident,
)
# SqlExpr 是「写侧 SQL 片段包装」，与具体后端无关，直接复用 SQLiteClient 的实现：
# 两个后端拿到的是同一个类，`isinstance(value, SqlExpr)` 跨后端可识别，
# 上层从 SQLite 切到 MySQL 时 SqlExpr 无需换类。
from .sqlite_client import SqlExpr

from .errors import (
    DatabaseConnectionError,
    DatabaseDriverMissingError,
    DatabaseOperationalError,
    DatabaseValidationError,
)

logger = logging.getLogger(__name__)


# MySQL 方言适配
# ---------------------------------------------------------------------------

# 合法标识符校验同 SQLite（复用 schema 中的 check_ident，保证防注入逻辑统一）
# MySQL 标识符用反引号包裹
def _quote_ident_mysql(name: str) -> str:
    """给标识符加反引号（MySQL 方言），内部反引号转义为两个反引号。调用前必须过 check_ident。"""
    escaped_name = name.replace("`", "``")
    return f"`{escaped_name}`"


# ColumnType 到 MySQL 类型的映射
TYPE_TO_MYSQL: dict[ColumnType, str] = {
    ColumnType.TEXT:      "TEXT",
    ColumnType.INTEGER:   "INT",
    ColumnType.REAL:      "DOUBLE",
    ColumnType.BLOB:      "LONGBLOB",
    ColumnType.BOOLEAN:   "TINYINT(1)",   # MySQL 用 TINYINT(1) 存布尔
    ColumnType.VARCHAR:   "VARCHAR(255)", # MySQL VARCHAR 必须带长度，默认 255（和 SQLite 长度仅作文档的行为对齐）
    ColumnType.TIMESTAMP: "DATETIME",     # 时间戳存 DATETIME，对齐 ISO 字符串习惯
}


def _sql_literal_mysql(value: Any, ctype: ColumnType) -> str:
    """把 Python 默认值转成 MySQL 字面量，和 SQLite 版本行为一致。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    # 字符串：转义单引号
    s = str(value).replace("'", "''")
    return f"'{s}'"


def _column_sql_mysql(col: Column) -> str:
    """把一个 Column 模型转成 MySQL 列定义片段。"""
    _check_ident(col.name, "列名")
    # VARCHAR 显式指定 length 时使用用户指定的长度
    if col.type is ColumnType.VARCHAR and col.length is not None:
        type_sql = f"VARCHAR({int(col.length)})"
    else:
        type_sql = TYPE_TO_MYSQL[col.type]
    parts: list[str] = [_quote_ident_mysql(col.name), type_sql]

    if col.primary_key:
        parts.append("PRIMARY KEY")
        if col.type is ColumnType.INTEGER and col.name.lower() == "id":
            parts.append("AUTO_INCREMENT")

    if not col.nullable and not col.primary_key:
        parts.append("NOT NULL")

    if col.unique and not col.primary_key:
        parts.append("UNIQUE KEY")

    if col.default is not None:
        parts.append("DEFAULT " + _sql_literal_mysql(col.default, col.type))

    return " ".join(parts)


def _column_index_stmt_mysql(table: str, col: Column) -> Optional[str]:
    """为声明了 index=True 的普通列生成 MySQL CREATE INDEX 语句。"""
    if not (col.index and not col.primary_key and not col.unique):
        return None
    _check_ident(col.name, "列名")
    idx_name = f"idx_{table}_{col.name}"
    return (
        f"CREATE INDEX {_quote_ident_mysql(idx_name)} "
        f"ON {_quote_ident_mysql(table)} ({_quote_ident_mysql(col.name)})"
    )


def _index_stmt_mysql(table: str, idx: Index) -> str:
    """把一个显式 Index 模型转成 MySQL CREATE INDEX 语句。"""
    _check_ident(idx.name, "索引名")
    for c in idx.columns:
        _check_ident(c, "索引列名")
    cols_sql = ", ".join(_quote_ident_mysql(c) for c in idx.columns)
    unique = "UNIQUE " if idx.unique else ""
    return (
        f"CREATE {unique}INDEX {_quote_ident_mysql(idx.name)} "
        f"ON {_quote_ident_mysql(table)} ({cols_sql})"
    )


def _parse_column_list_mysql(table: str, columns: list[Column]) -> tuple[list[str], list[str]]:
    """解析 list[Column]，返回 MySQL 方言的 (col_defs, index_stmts)。"""
    _check_ident(table, "表名")
    col_defs: list[str] = []
    index_stmts: list[str] = []
    for c in columns:
        if not isinstance(c, Column):
            raise TypeError(f"columns 列表元素必须是 Column，收到 {type(c)}")
        col_defs.append(_column_sql_mysql(c))
        stmt = _column_index_stmt_mysql(table, c)
        if stmt is not None:
            index_stmts.append(stmt)
    return col_defs, index_stmts


def _parse_schema_mysql(schema: Schema) -> tuple[list[str], list[str]]:
    """解析 Schema，返回 MySQL 方言的 (col_defs, index_stmts)。"""
    col_defs, index_stmts = _parse_column_list_mysql(schema.table_name, schema.columns)
    if schema.indexes:
        for idx in schema.indexes:
            index_stmts.append(_index_stmt_mysql(schema.table_name, idx))
    return col_defs, index_stmts


class MySQLClient(DatabaseClientBase):
    """MySQL 客户端，接口与 SQLiteClient 100% 对齐，字典行结果、参数化查询、声明式建表、事务嵌套。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "",
        charset: str = "utf8mb4",
        connect_timeout: float = 5.0,
        timeout: float = 5.0,
        autocommit: bool = False,
    ):
        """初始化 MySQL 连接。

        Args:
            host: MySQL 主机地址，默认 127.0.0.1。
            port: MySQL 端口，默认 3306。
            user: 用户名，默认 root。
            password: 密码。
            database: 数据库名（必须已存在，客户端不会自动 CREATE DATABASE）。
            charset: 字符集，默认 utf8mb4（支持 emoji）。
            connect_timeout: 连接超时秒数，默认 5.0。
            timeout: 读/写超时秒数，默认 5.0。
            autocommit: 是否自动提交，默认 False（由客户端事务逻辑控制）。
        """
        # 可选依赖：延迟导入 pymysql（R2：函数体内 import、缺失时降级/报错）
        try:
            import pymysql  # type: ignore
            from pymysql.cursors import DictCursor as _DictCursor  # type: ignore
        except ImportError as _e:  # pragma: no cover
            raise DatabaseDriverMissingError(
                "使用 MySQLClient 需要安装 pymysql 依赖：\n"
                "  pip install pymysql\n"
                "或安装 fast-agents 全量依赖：\n"
                "  pip install fast-agents[mysql]"
            ) from _e
        self._pymysql = pymysql
        self._dict_cursor_cls = _DictCursor

        super().__init__(timeout)
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.connect_timeout = connect_timeout
        self.autocommit = autocommit
        self._conn: Any = None
        # 事务状态：与 SQLiteClient 完全一致的嵌套深度计数方案，从根源消除状态失同步
        self._tx_depth = 0
        self._tx_must_rollback = False
        self._connect()

    # ---------- 连接管理 ----------
    def _connect(self) -> None:
        """建立连接并配置默认选项。"""
        try:
            self._conn = self._pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                connect_timeout=int(self.connect_timeout),
                read_timeout=int(self.timeout),
                write_timeout=int(self.timeout),
                autocommit=self.autocommit,
                cursorclass=self._dict_cursor_cls,  # 统一返回字典格式结果
                init_command="SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'",
            )
        except self._pymysql.MySQLError as e:
            raise DatabaseConnectionError(f"连接 MySQL 失败: {e}") from e

    @property
    def conn(self) -> Any:
        """获取底层连接（如果关闭则自动重连）。"""
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        # 检查连接是否存活，断开则重连
        try:
            self._conn.ping(reconnect=True)
        except self._pymysql.MySQLError:
            self._connect()
        return self._conn

    @property
    def _in_transaction(self) -> bool:
        """当前是否处于显式事务中（_tx_depth 的只读派生，与 SQLite 语义一致）。"""
        return self._tx_depth > 0

    def _close(self) -> None:
        """关闭连接，残留事务先尝试回滚。"""
        if self._conn is not None:
            if self._tx_depth > 0:
                try:
                    self._conn.rollback()
                except self._pymysql.MySQLError:
                    pass
                self._tx_depth = 0
                self._tx_must_rollback = False
            self._conn.close()
            self._conn = None

    # ---------- 基础执行 ----------
    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any]:
        """执行写 SQL（INSERT/UPDATE/DELETE/CREATE/DROP）。

        事务语义与 SQLiteClient 完全一致：
        - 不在事务中时自动提交（单次语句一个隐式事务）；
        - 在显式事务中（`with db:`）时只执行不 commit，由 __exit__ 统一提交/回滚；
        - 执行失败时保证连接状态可用、事务状态不被破坏。
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                lastrowid = cur.lastrowid
                rowcount = cur.rowcount
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": lastrowid, "rowcount": rowcount}
        except self._pymysql.MySQLError as e:
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"执行SQL失败: {e}\nSQL: {sql}") from e

    def executemany(self, sql: str, params_list: list[tuple | dict]) -> dict[str, Any]:
        """批量执行写 SQL。"""
        try:
            with self.conn.cursor() as cur:
                cur.executemany(sql, params_list)
                lastrowid = cur.lastrowid
                rowcount = cur.rowcount
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": lastrowid, "rowcount": rowcount}
        except self._pymysql.MySQLError as e:
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"批量执行SQL失败: {e}") from e

    def executescript(self, sql: str) -> None:
        """执行一段脚本（多条 SQL 以 ; 分隔），不在事务中时自动提交。

        注意：MySQL 中 execute 本身支持多语句，这里做简单封装，语义与 SQLite 对齐。
        """
        try:
            with self.conn.cursor() as cur:
                for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                    cur.execute(stmt)
            if not self._in_transaction:
                self.conn.commit()
        except self._pymysql.MySQLError as e:
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"执行脚本失败: {e}\nSQL: {sql}") from e

    def _sync_transaction_state_after_error(self) -> None:
        """执行 SQL 出错后清理半事务，逻辑与 SQLiteClient 完全一致。"""
        if self._conn is None:
            return
        if self._tx_depth == 0:
            try:
                self._conn.rollback()
            except self._pymysql.MySQLError:
                pass

    def query(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
        fetch_one: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        """执行读 SQL（SELECT / SHOW），返回字典列表。"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            if fetch_one:
                return rows[0] if rows else None
            return list(rows)
        except self._pymysql.MySQLError as e:
            raise DatabaseOperationalError(f"查询失败: {e}\nSQL: {sql}") from e

    # ---------- DDL：建表 / 查表 / 删表 / 索引 ----------
    def table_exists(self, table: str) -> bool:
        """判断表是否已存在（MySQL 方言，查 information_schema）。"""
        _check_ident(table, "表名")
        row = self.query(
            "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
            (table,),
            fetch_one=True,
        )
        return row is not None

    def _list_columns(self, table: str) -> set[str]:
        """列出已存在表的列名集合（小写），用于迁移补列。"""
        _check_ident(table, "表名")
        rows = self.query(f"SHOW COLUMNS FROM {_quote_ident_mysql(table)}")
        return {r["Field"].lower() for r in rows}  # type: ignore[index]

    def _list_indexes(self, table: str) -> set[str]:
        """列出已存在表的索引名集合，用于幂等建索引。"""
        _check_ident(table, "表名")
        rows = self.query(f"SHOW INDEX FROM {_quote_ident_mysql(table)}")
        return {r["Key_name"] for r in rows}  # type: ignore[index]

    def create_table(
        self,
        table: str,
        columns: dict[str, str] | list[Column] | Schema,
        if_not_exists: bool = True,
    ) -> None:
        """建表，三种 columns 写法与 SQLiteClient 完全一致。"""
        _check_ident(table, "表名")

        # 1) Schema 对象
        if isinstance(columns, Schema):
            if table and columns.table_name != table:
                raise DatabaseValidationError(f"Schema.table_name={columns.table_name!r} 与传入 table={table!r} 不一致")
            self._create_table_from_schema(columns, if_not_exists=if_not_exists)
            return

        # 2) list[Column]
        if isinstance(columns, list):
            col_defs, index_stmts = _parse_column_list_mysql(table, columns)
            self._create_table_raw(table, col_defs, if_not_exists=if_not_exists)
            for stmt in index_stmts:
                self.execute(stmt)
            return

        # 3) dict[str, str]
        if isinstance(columns, dict):
            if not columns:
                raise DatabaseValidationError("create_table 的 columns 不能为空")
            col_defs = []
            for col_name, col_def in columns.items():
                _check_ident(col_name, "列名")
                if not isinstance(col_def, str) or not col_def.strip():
                    raise DatabaseValidationError(f"列 {col_name!r} 的类型定义必须是非空字符串")
                col_defs.append(f"{_quote_ident_mysql(col_name)} {col_def.strip()}")
            self._create_table_raw(table, col_defs, if_not_exists=if_not_exists)
            return

        raise TypeError(
            "columns 参数只支持 dict[str,str] / list[Column] / Schema 三种形式，"
            f"收到 {type(columns)}"
        )

    def _create_table_raw(self, table: str, col_defs: list[str], if_not_exists: bool) -> None:
        """执行原始 MySQL CREATE TABLE 语句，默认 InnoDB 引擎、utf8mb4 字符集。"""
        if_not = "IF NOT EXISTS " if if_not_exists else ""
        sql = (
            f"CREATE TABLE {if_not}{_quote_ident_mysql(table)} ({','.join(col_defs)}) "
            f"ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
        )
        self.execute(sql)

    def _create_table_from_schema(self, schema: Schema, if_not_exists: bool) -> None:
        """从 Schema 对象建表 + 索引（MySQL 方言解析）。"""
        col_defs, index_stmts = _parse_schema_mysql(schema)
        self._create_table_raw(schema.table_name, col_defs, if_not_exists=if_not_exists)
        # 索引幂等创建：MySQL CREATE INDEX 不支持 IF NOT EXISTS，先查已有索引避免报错
        existing_indexes = self._list_indexes(schema.table_name)
        for stmt in index_stmts:
            # 从语句中提取索引名（简单解析：CREATE [UNIQUE] INDEX `name` ON ...）
            idx_name_match = re.search(r"CREATE (?:UNIQUE )?INDEX `([^`]+)`", stmt)
            if idx_name_match and idx_name_match.group(1) in existing_indexes:
                continue
            self.execute(stmt)

    def ensure_table(self, table: str, columns: dict[str, str] | list[Column] | Schema) -> None:
        """幂等建表，等价 create_table(..., if_not_exists=True)。"""
        self.create_table(table, columns, if_not_exists=True)

    def ensure_schema(self, schema: Schema, migrate: bool = False) -> None:
        """按 Schema 幂等建表 + 建索引；migrate=True 时尝试补齐缺失列（ADD COLUMN）。

        与 SQLiteClient 行为一致：只补列，不删列、不改列类型，保证安全。
        """
        if not isinstance(schema, Schema):
            raise TypeError("schema 必须是 fast_agents.database.Schema 实例")
        _check_ident(schema.table_name, "表名")

        if not self.table_exists(schema.table_name):
            self._create_table_from_schema(schema, if_not_exists=True)
            return

        # 表已存在：migrate=True 时补缺失列
        if migrate:
            existing_cols = {c.lower() for c in self._list_columns(schema.table_name)}
            for c in schema.columns:
                if c.name.lower() in existing_cols:
                    continue
                # MySQL ADD COLUMN 语法与 SQLite 一致，但不支持加 PRIMARY KEY
                col_sql = _column_sql_mysql(c)
                self.execute(f"ALTER TABLE {_quote_ident_mysql(schema.table_name)} ADD COLUMN {col_sql}")

        # 索引幂等创建
        _, index_stmts = _parse_schema_mysql(schema)
        existing_indexes = self._list_indexes(schema.table_name)
        for stmt in index_stmts:
            idx_name_match = re.search(r"CREATE (?:UNIQUE )?INDEX `([^`]+)`", stmt)
            if idx_name_match and idx_name_match.group(1) in existing_indexes:
                continue
            self.execute(stmt)

    def drop_table(self, table: str, if_exists: bool = True) -> None:
        """删表。"""
        _check_ident(table, "表名")
        if_ex = "IF EXISTS " if if_exists else ""
        self.execute(f"DROP TABLE {if_ex}{_quote_ident_mysql(table)}")

    # ---------- 便捷 CRUD ----------
    def insert(self, table: str, data: dict[str, Any]) -> int:
        """插入一行，返回新行 id。"""
        if not data:
            raise DatabaseValidationError("insert 数据不能为空")
        _check_ident(table, "表名")
        columns = ", ".join(_quote_ident_mysql(k) for k in data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {_quote_ident_mysql(table)} ({columns}) VALUES ({placeholders})"
        return self.execute(sql, tuple(data.values()))["lastrowid"]

    def select(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询多行，where 做等值匹配。"""
        _check_ident(table, "表名")
        sql = f"SELECT * FROM {_quote_ident_mysql(table)}"
        params: list[Any] = []
        if where:
            conditions = []
            for k, v in where.items():
                _check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{_quote_ident_mysql(k)} IS NULL")
                else:
                    conditions.append(f"{_quote_ident_mysql(k)} = %s")
                    params.append(v)
            sql += f" WHERE {' AND '.join(conditions)}"
        if order_by:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$", order_by.strip()):
                raise DatabaseValidationError(f"非法 order_by: {order_by!r}")
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, tuple(params))  # type: ignore

    # ---------- 谓词下推：Query / Filter → 参数化 SQL（MySQL 方言，用 %s 占位符） ----------
    @staticmethod
    def _compile_filter(f: Filter, idx: int) -> tuple[str, list[Any]]:
        """把单个 Filter 编译成 (SQL 片段, 参数列表)，MySQL 用 %s 占位符。"""
        if not isinstance(f, Filter):
            raise TypeError(f"filters 列表元素必须是 Filter，收到 {type(f)}")
        _check_ident(f.field, "字段名")
        col = _quote_ident_mysql(f.field)
        op = f.op

        # IS NULL / IS NOT NULL：无参
        if op is FilterOp.IS_NULL:
            return f"{col} IS NULL", []
        if op is FilterOp.NOT_NULL:
            return f"{col} IS NOT NULL", []

        # IN：展开成 (%s, %s, ...)
        if op is FilterOp.IN:
            if not isinstance(f.value, (list, tuple)) or len(f.value) == 0:
                raise DatabaseValidationError(
                    f"Filter(field={f.field!r}, op=IN) 的 value 必须是非空 list/tuple，"
                    f"收到 {type(f.value).__name__}"
                )
            placeholders = ", ".join(["%s"] * len(f.value))
            return f"{col} IN ({placeholders})", list(f.value)

        # 其它二元操作符
        if f.value is None:
            raise DatabaseValidationError(
                f"Filter(field={f.field!r}, op={op.value}) 必须提供 value；"
                f"判空请用 IS_NULL / NOT_NULL"
            )
        return f"{col} {op.value} %s", [f.value]

    @staticmethod
    def _compile_query(q: Query) -> tuple[str, list[Any]]:
        """把 Query 对象编译成 (SQL 语句, 参数列表)，MySQL 方言。"""
        _check_ident(q.table_name, "表名")
        sql_parts: list[str] = [f"SELECT * FROM {_quote_ident_mysql(q.table_name)}"]
        params: list[Any] = []

        if q.filters:
            conds: list[str] = []
            for i, f in enumerate(q.filters):
                frag, p = MySQLClient._compile_filter(f, i)
                conds.append(frag)
                params.extend(p)
            sql_parts.append("WHERE " + " AND ".join(conds))

        if q.sort_by is not None:
            sort_cols = [q.sort_by] if isinstance(q.sort_by, str) else list(q.sort_by)
            if not sort_cols:
                raise DatabaseValidationError("sort_by 不能是空列表")
            for col in sort_cols:
                _check_ident(col, "排序字段名")
            direction = "ASC" if q.sort_order is SortOrder.ASC else "DESC"
            order_by = ", ".join(f"{_quote_ident_mysql(c)} {direction}" for c in sort_cols)
            sql_parts.append(f"ORDER BY {order_by}")

        if q.limit is not None:
            n = int(q.limit)
            if n < 0:
                raise DatabaseValidationError(f"limit 不能为负数: {n}")
            sql_parts.append(f"LIMIT {n}")

        if q.offset is not None:
            m = int(q.offset)
            if m < 0:
                raise DatabaseValidationError(f"offset 不能为负数: {m}")
            # MySQL 支持 LIMIT m, n 语法，单独 OFFSET 必须配合 LIMIT，没有 LIMIT 时补个大数
            if q.limit is None:
                sql_parts.append("LIMIT 18446744073709551615")  # MySQL 官方文档推荐的"无上限"值
            sql_parts.append(f"OFFSET {m}")

        return " ".join(sql_parts), params

    def select_where(self, query: Query) -> list[dict[str, Any]]:
        """按声明式 Query 查询，接口与 SQLiteClient 完全一致。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        sql, params = self._compile_query(query)
        return self.query(sql, tuple(params))  # type: ignore[return-value]

    def select_one_where(self, query: Query) -> dict[str, Any] | None:
        """按声明式 Query 查询单行（自动 LIMIT 1）。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        limited = query.model_copy(update={"limit": 1, "offset": None})
        rows = self.select_where(limited)
        return rows[0] if rows else None

    def select_one(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
    ) -> dict[str, Any] | None:
        """查询单行，不存在返回 None。"""
        rows = self.select(table, where=where, order_by=order_by, limit=1)
        return rows[0] if rows else None

    def update(self, table: str, data: dict[str, Any], where: dict[str, Any]) -> int:
        """更新行，返回影响行数。"""
        if not data or not where:
            raise DatabaseValidationError("update 需要 data 和 where")
        _check_ident(table, "表名")
        set_clause = ", ".join([f"{_quote_ident_mysql(k)} = %s" for k in data.keys()])
        where_parts: list[str] = []
        params: list[Any] = list(data.values())
        for k, v in where.items():
            _check_ident(k, "字段名")
            if v is None:
                where_parts.append(f"{_quote_ident_mysql(k)} IS NULL")
            else:
                where_parts.append(f"{_quote_ident_mysql(k)} = %s")
                params.append(v)
        sql = f"UPDATE {_quote_ident_mysql(table)} SET {set_clause} WHERE {' AND '.join(where_parts)}"
        return self.execute(sql, tuple(params))["rowcount"]

    def delete(self, table: str, where: dict[str, Any]) -> int:
        """删除行，返回影响行数。"""
        if not where:
            raise DatabaseValidationError("delete 必须传 where 防止误删全表")
        _check_ident(table, "表名")
        where_parts: list[str] = []
        params: list[Any] = []
        for k, v in where.items():
            _check_ident(k, "字段名")
            if v is None:
                where_parts.append(f"{_quote_ident_mysql(k)} IS NULL")
            else:
                where_parts.append(f"{_quote_ident_mysql(k)} = %s")
                params.append(v)
        sql = f"DELETE FROM {_quote_ident_mysql(table)} WHERE {' AND '.join(where_parts)}"
        return self.execute(sql, tuple(params))["rowcount"]

    # ---------- 事务上下文管理（与 SQLiteClient 行为完全一致，支持嵌套） ----------
    def __enter__(self) -> "MySQLClient":
        """进入 with 块：最外层显式开启事务，内层重入加入父事务。"""
        if self._tx_depth == 0:
            # 关闭自动提交，显式开启事务
            try:
                self.conn.begin()
            except self._pymysql.MySQLError as e:
                raise DatabaseOperationalError(f"开启事务失败: {e}") from e
            self._tx_must_rollback = False
        self._tx_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """退出 with 块：内层 depth--，最外层统一提交/回滚。"""
        self._tx_depth -= 1
        if self._tx_depth > 0:
            # 内层退出：有异常标记必须回滚
            if exc_type is not None:
                self._tx_must_rollback = True
            return False
        # 最外层退出：统一提交/回滚
        try:
            if exc_type is None and not self._tx_must_rollback:
                try:
                    self.conn.commit()
                except self._pymysql.MySQLError as e:
                    try:
                        self.conn.rollback()
                    except self._pymysql.MySQLError:
                        pass
                    raise DatabaseOperationalError(f"提交事务失败: {e}") from e
            else:
                try:
                    self.conn.rollback()
                except self._pymysql.MySQLError:
                    pass
        finally:
            self._tx_depth = 0
            self._tx_must_rollback = False
            # 恢复 autocommit 初始设置
            if self.autocommit:
                self.conn.autocommit(True)
        return False

    # ---------- 统计与批量写入 ----------
    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """按等值条件统计行数，where=None 返回表总行数。"""
        _check_ident(table, "表名")
        sql = f"SELECT COUNT(*) AS cnt FROM {_quote_ident_mysql(table)}"
        params: list[Any] = []
        conditions: list[str] = []
        if where:
            for k, v in where.items():
                _check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{_quote_ident_mysql(k)} IS NULL")
                else:
                    conditions.append(f"{_quote_ident_mysql(k)} = %s")
                    params.append(v)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        row = self.query(sql, tuple(params), fetch_one=True)
        return int(row["cnt"]) if row else 0  # type: ignore[index]

    def batch_insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        """批量插入多行，返回插入行数；空列表直接返回0。"""
        if not rows:
            return 0
        _check_ident(table, "表名")
        columns = list(rows[0].keys())
        for col in columns:
            _check_ident(col, "字段名")
        expected_cols = set(columns)
        for i, row in enumerate(rows):
            row_cols = set(row.keys())
            if row_cols != expected_cols:
                missing = expected_cols - row_cols
                extra = row_cols - expected_cols
                msg = f"第 {i} 行字段与首行不一致"
                if missing:
                    msg += f"，缺少字段: {sorted(missing)}"
                if extra:
                    msg += f"，多余字段: {sorted(extra)}"
                raise DatabaseValidationError(msg)
        col_sql_str = ", ".join(_quote_ident_mysql(c) for c in columns)
        placeholders = ", ".join("%s" for _ in columns)
        sql = f"INSERT INTO {_quote_ident_mysql(table)} ({col_sql_str}) VALUES ({placeholders})"
        params_list = [tuple(row[c] for c in columns) for row in rows]
        self.executemany(sql, params_list)
        return len(rows)

    # ---------- 条件写入：upsert / update_where / delete_where ----------
    @staticmethod
    def _compile_filters(filters: list[Filter] | None) -> tuple[str, list[Any]]:
        """把 Filter 列表编译成 (WHERE 片段, 参数列表)，与 SQLite 语义一致。"""
        if not filters:
            return "", []
        conds: list[str] = []
        params: list[Any] = []
        for i, f in enumerate(filters):
            frag, p = MySQLClient._compile_filter(f, i)
            conds.append(frag)
            params.extend(p)
        return " AND ".join(conds), params

    def upsert(
        self,
        table: str,
        data: dict[str, Any],
        conflict_keys: str | list[str],
        update_columns: list[str] | None = None,
    ) -> int:
        """插入或更新（MySQL 方言：INSERT ... ON DUPLICATE KEY UPDATE），返回影响行数。

        接口与 SQLiteClient.upsert 完全一致，自动适配 MySQL 语法：
        - conflict_keys 对应主键/唯一键（MySQL 自动探测主键/唯一键冲突，这里参数仅用于语法对齐和校验）
        - update_columns 冲突时更新的列，默认更新除 conflict_keys 外的所有列
        """
        if not data:
            raise DatabaseValidationError("upsert 数据不能为空")
        _check_ident(table, "表名")

        keys = [conflict_keys] if isinstance(conflict_keys, str) else list(conflict_keys)
        if not keys:
            raise DatabaseValidationError("upsert 的 conflict_keys 不能为空")
        for k in keys:
            _check_ident(k, "冲突键")
        missing = [k for k in keys if k not in data]
        if missing:
            raise DatabaseValidationError(f"upsert 的 conflict_keys 必须都在 data 中，缺少: {missing}")

        columns = list(data.keys())
        for c in columns:
            _check_ident(c, "字段名")

        if update_columns is None:
            update_columns = [c for c in columns if c not in keys]
        for c in update_columns:
            _check_ident(c, "字段名")

        col_sql = ", ".join(_quote_ident_mysql(c) for c in columns)
        placeholders = ", ".join("%s" for _ in columns)
        if update_columns:
            assigned = ", ".join(
                f"{_quote_ident_mysql(c)} = VALUES({_quote_ident_mysql(c)})" for c in update_columns
            )
            on_duplicate = f"ON DUPLICATE KEY UPDATE {assigned}"
        else:
            # 冲突时忽略：MySQL 没有 DO NOTHING，用更新一个无意义列的方式实现，或直接用 INSERT IGNORE？
            # 为严格对齐语义，update_columns 为空时用 ON DUPLICATE KEY UPDATE id=id（无操作）
            on_duplicate = "ON DUPLICATE KEY UPDATE " + f"{_quote_ident_mysql(keys[0])} = {_quote_ident_mysql(keys[0])}"

        sql = f"INSERT INTO {_quote_ident_mysql(table)} ({col_sql}) VALUES ({placeholders}) {on_duplicate}"
        return self.execute(sql, tuple(data.values()))["rowcount"]

    def update_where(self, table: str, values: dict[str, Any], query: Query) -> int:
        """按声明式 Query 条件批量更新，返回影响行数，接口与 SQLiteClient 完全一致。"""
        if not values:
            raise DatabaseValidationError("update_where 需要 values")
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        if not query.filters:
            raise DatabaseValidationError("update_where 的 query.filters 不能为空（防止误更新全表）")
        _check_ident(table, "表名")

        set_parts: list[str] = []
        params: list[Any] = []
        for col, val in values.items():
            _check_ident(col, "字段名")
            if isinstance(val, SqlExpr):
                set_parts.append(f"{_quote_ident_mysql(col)} = {val.sql}")
            else:
                set_parts.append(f"{_quote_ident_mysql(col)} = %s")
                params.append(val)

        where_sql, where_params = self._compile_filters(query.filters)
        params.extend(where_params)
        sql = f"UPDATE {_quote_ident_mysql(table)} SET {', '.join(set_parts)} WHERE {where_sql}"
        return self.execute(sql, tuple(params))["rowcount"]

    def delete_where(self, table: str, query: Query) -> int:
        """按声明式 Query 条件批量删除，返回影响行数，接口与 SQLiteClient 完全一致。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        if not query.filters:
            raise DatabaseValidationError("delete_where 的 query.filters 不能为空（防止误删全表）")
        _check_ident(table, "表名")
        where_sql, params = self._compile_filters(query.filters)
        sql = f"DELETE FROM {_quote_ident_mysql(table)} WHERE {where_sql}"
        return self.execute(sql, tuple(params))["rowcount"]

    def __del__(self) -> None:
        """析构时尝试关闭连接。"""
        try:
            self.close()
        except Exception:
            pass
