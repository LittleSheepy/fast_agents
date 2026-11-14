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
from typing import Any

from .database_client_base import DatabaseClientBase
from .query import Filter, FilterOp, Query, SortOrder
from .schema import (
    Column,
    ColumnType,
    Schema,
    check_ident as _check_ident,
    check_ident,
    sql_literal,
    column_index_stmt,
    index_stmt,
    column_sql,
    parse_column_list,
    parse_schema,
    quote_ident as _quote_ident,

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
def quote_ident(name: str) -> str:
    """给标识符加反引号（MySQL 方言），内部反引号转义为两个反引号。调用前必须过 check_ident。"""
    return _quote_ident(name, "`")


def _server_supports_row_alias(conn: Any) -> bool:
    """判断 MySQL 服务端是否支持 ``INSERT ... VALUES (...) AS new`` 行别名语法。

    8.0.20 起 ``ON DUPLICATE KEY UPDATE ... VALUES(col)`` 被标记弃用，官方推荐改用行别名
    （``INSERT ... AS new ON DUPLICATE KEY UPDATE col = new.col``）；但行别名是 8.0.19 才引入
    的语法，**5.7 及更早不支持**，MariaDB 各版本也不支持。这里按服务端版本自适应：

    - MySQL >= 8.0.19 → 用行别名（摆脱已弃用的 VALUES()）；
    - MySQL <= 8.0.18 / 5.7、MariaDB、或取不到版本信息 → 回退到兼容性最广的 ``VALUES(col)``。

    版本取自 pymysql 连接握手阶段已缓存的 ``get_server_info()``，不额外往返。
    """
    getter = getattr(conn, "get_server_info", None)
    if getter is None:
        return False
    try:
        info = str(getter() or "")
    except Exception:  # pragma: no cover - 驱动取版本异常一律回退
        return False
    if "mariadb" in info.lower():
        return False
    match = re.match(r"\s*(\d+)\.(\d+)\.(\d+)", info)
    if match is None:
        return False
    return tuple(int(g) for g in match.groups()) >= (8, 0, 19)


def _split_sql_statements(sql: str) -> list[str]:
    """把一段多语句 SQL 按顶层分号切分为单条语句（已 strip、已过滤空串）。

    比 ``sql.split(";")`` 更稳：只有出现在「语句正文」里的分号才切分，出现在字符串
    字面量 / 反引号标识符 / 注释里的分号不会被误切。覆盖 MySQL 常见写法：

    - 单引号字符串 ``'a;b'``（含 ``''`` 与 ``\'`` 两种转义）；
    - 双引号字符串 ``"a;b"``（MySQL 默认非 ANSI_QUOTES 下即字符串）；
    - 反引号标识符（内部反引号用两个反引号转义）；
    - 行注释 ``-- ...``（MySQL 要求 ``--`` 后接空白）与 ``# ...``，都到行尾结束；
    - 块注释 ``/* ... */``，可跨行。

    这不是完整 SQL 词法分析（那是依赖级的活儿），只保证「分号切分」这一件事正确，
    与 sqlite3 原生 ``executescript`` 的切分行为对齐。
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    state = "normal"
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if state == "normal":
            if ch == "'":
                state = "single"
            elif ch == '"':
                state = "double"
            elif ch == "`":
                state = "backtick"
            elif ch == "#":
                state = "line_comment"
            elif ch == "-" and nxt == "-" and (i + 2 >= n or sql[i + 2] in " \t\r\n"):
                state = "line_comment"
            elif ch == "/" and nxt == "*":
                state = "block_comment"
            elif ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
                i += 1
                continue
            buf.append(ch)
        elif state == "single":
            buf.append(ch)
            if ch == "\\" and nxt:
                buf.append(nxt)
                i += 1
            elif ch == "'":
                if nxt == "'":
                    buf.append(nxt)
                    i += 1
                else:
                    state = "normal"
        elif state == "double":
            buf.append(ch)
            if ch == "\\" and nxt:
                buf.append(nxt)
                i += 1
            elif ch == '"':
                if nxt == '"':
                    buf.append(nxt)
                    i += 1
                else:
                    state = "normal"
        elif state == "backtick":
            buf.append(ch)
            if ch == "`":
                if nxt == "`":
                    buf.append(nxt)
                    i += 1
                else:
                    state = "normal"
        elif state == "line_comment":
            buf.append(ch)
            if ch == "\n":
                state = "normal"
        else:  # block_comment
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 1
                state = "normal"
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


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


# MySQL 列定义方言参数（与 schema.column_sql 的注入点一一对应）
_MYSQL_COLUMN_KW: dict[str, Any] = dict(
    type_map=TYPE_TO_MYSQL,
    quote="`",
    unique_keyword="UNIQUE KEY",
    auto_increment=True,
    varchar_length=True,
    if_not_exists=False,
)


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
        # 服务端是否支持 INSERT ... AS new 行别名（None=未探测；惰性探测一次后缓存，重连时重置）
        self._row_alias_supported: bool | None = None
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
        # 重连后服务端可能变化，行别名支持标记重新探测
        self._row_alias_supported = None

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

    @property
    def conn(self) -> Any:
        """获取底层连接。

        非事务态（autocommit 单条语句）下保持原有行为：未连接则连、断开则
        ping(reconnect=True) 自动重连；长期空闲连接可无缝复用。

        事务态（已 __enter__ 进入 with 块、_tx_depth>0）下不做 ping/自动重连：
        否则一旦网络抖动触发隐式重连，旧连接上已经 BEGIN 的事务会随连接消失，
        新连接以 autocommit 落在空事务上，最后的 commit() 会静默丢失前面写入。
        事务中断连时由后续 execute/query/cursor() 自然抛 MySQLError，被
        execute/query 的 except MySQLError → DatabaseOperationalError 捕获，
        让上层明确感知失败并自行重试。
        """
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        if self._tx_depth > 0:
            # 事务中不 ping/不重连，避免隐式丢失事务边界
            return self._conn
        # 非事务态：检查连接存活，断开则重连
        try:
            self._conn.ping(reconnect=True)
        except self._pymysql.MySQLError:
            self._connect()
        return self._conn

    @property
    def _supports_row_alias(self) -> bool:
        """当前服务端是否支持 ``INSERT ... AS new`` 行别名语法（MySQL 8.0.19+），结果缓存。

        无法确定服务端版本时保守返回 False，即继续用兼容性最广的 ``VALUES(col)`` 语法。
        """
        if self._row_alias_supported is None:
            self._row_alias_supported = _server_supports_row_alias(self.conn)
        return self._row_alias_supported

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

        切分交给 ``_split_sql_statements``：只在「语句正文」遇到分号才切分，不会误切
        字符串字面量 / 反引号标识符 / 注释里的分号（``sql.split(";")`` 会）；语义与
        SQLite 对齐（sqlite3 原生 executescript 同样能识别字面量内的分号）。
        """
        try:
            with self.conn.cursor() as cur:
                for stmt in _split_sql_statements(sql):
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

    def query(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = (), fetch_one: bool = False
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
    # ------------ 方言 ------------
    def _quote_ident(self, name: str) -> str:
        """MySQL 反引号标识符。"""
        return _quote_ident(name, "`")

    def _parse_columns(self, table: str, columns: list[Column]) -> tuple[list[str], list[str]]:
        """MySQL 方言列解析，带 MySQL 列关键字参数。"""
        return parse_column_list(table, columns, **_MYSQL_COLUMN_KW)

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
        rows = self.query(f"SHOW COLUMNS FROM {quote_ident(table)}")
        return {r["Field"].lower() for r in rows}  # type: ignore[index]

    def _list_indexes(self, table: str) -> set[str]:
        """列出已存在表的索引名集合，用于幂等建索引。"""
        _check_ident(table, "表名")
        rows = self.query(f"SHOW INDEX FROM {quote_ident(table)}")
        return {r["Key_name"] for r in rows}  # type: ignore[index]

    def _create_table_raw(self, table: str, col_defs: list[str], if_not_exists: bool) -> None:
        """执行原始 MySQL CREATE TABLE 语句，默认 InnoDB 引擎、utf8mb4 字符集。"""
        if_not = "IF NOT EXISTS " if if_not_exists else ""
        sql = (
            f"CREATE TABLE {if_not}{self._quote_ident(table)} ({','.join(col_defs)}) "
            f"ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
        )
        self.execute(sql)

    def _create_table_from_schema(self, schema: Schema, if_not_exists: bool) -> None:
        """从 Schema 对象建表 + 索引（MySQL 方言解析）。"""
        col_defs, index_stmts = parse_schema(schema, **_MYSQL_COLUMN_KW)
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
                col_sql = column_sql(c, **_MYSQL_COLUMN_KW)
                self.execute(f"ALTER TABLE {quote_ident(schema.table_name)} ADD COLUMN {col_sql}")

        # 索引幂等创建
        _, index_stmts = parse_schema(schema, **_MYSQL_COLUMN_KW)
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
        self.execute(f"DROP TABLE {if_ex}{quote_ident(table)}")

    # ---------- 便捷 CRUD ----------
    def insert(self, table: str, data: dict[str, Any]) -> int:
        """插入一行，返回新行 id。"""
        if not data:
            raise DatabaseValidationError("insert 数据不能为空")
        _check_ident(table, "表名")
        for k in data.keys():
            _check_ident(k, "字段名")
        columns = ", ".join(quote_ident(k) for k in data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {quote_ident(table)} ({columns}) VALUES ({placeholders})"
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
        sql = f"SELECT * FROM {quote_ident(table)}"
        params: list[Any] = []
        if where:
            conditions = []
            for k, v in where.items():
                _check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{quote_ident(k)} IS NULL")
                else:
                    conditions.append(f"{quote_ident(k)} = %s")
                    params.append(v)
            sql += f" WHERE {' AND '.join(conditions)}"
        if order_by:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$", order_by.strip()):
                raise DatabaseValidationError(f"非法 order_by: {order_by!r}")
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            # 与 _compile_query 同口径：limit 强制非负 int，负数直接报错——
            # MySQL 下 LIMIT -1 是语法错误，提前拦下给出统一的中文异常。
            n = int(limit)
            if n < 0:
                raise DatabaseValidationError(f"limit 不能为负数: {n}")
            sql += f" LIMIT {n}"
        return self.query(sql, tuple(params))  # type: ignore

    # ---------- 谓词下推：Query / Filter → 参数化 SQL（MySQL 方言，用 %s 占位符） ----------
    @staticmethod
    def _compile_filter(f: Filter, idx: int) -> tuple[str, list[Any]]:
        """把单个 Filter 编译成 (SQL 片段, 参数列表)，MySQL 用 %s 占位符。"""
        if not isinstance(f, Filter):
            raise TypeError(f"filters 列表元素必须是 Filter，收到 {type(f)}")
        _check_ident(f.field, "字段名")
        col = quote_ident(f.field)
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
        # 列投影：columns=None 保持 SELECT *；显式指定时逐列过标识符白名单，杜绝拼接注入
        if q.columns is None:
            col_clause = "*"
        else:
            if not q.columns:
                raise DatabaseValidationError("columns 不能是空列表")
            for c in q.columns:
                _check_ident(c, "列名")
            col_clause = ", ".join(quote_ident(c) for c in q.columns)
        sql_parts: list[str] = [f"SELECT {col_clause} FROM {quote_ident(q.table_name)}"]
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
            order_by = ", ".join(f"{quote_ident(c)} {direction}" for c in sort_cols)
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

    def select_one(self, table: str, where: dict[str, Any] | None = None, order_by: str | None = None,) -> dict[str, Any] | None:
        """查询单行，不存在返回 None。"""
        rows = self.select(table, where=where, order_by=order_by, limit=1)
        return rows[0] if rows else None

    def update(self, table: str, data: dict[str, Any], where: dict[str, Any]) -> int:
        """更新行，返回影响行数。"""
        if not data or not where:
            raise DatabaseValidationError("update 需要 data 和 where")
        _check_ident(table, "表名")
        set_clause = ", ".join([f"{quote_ident(k)} = %s" for k in data.keys()])
        where_parts: list[str] = []
        params: list[Any] = list(data.values())
        for k, v in where.items():
            _check_ident(k, "字段名")
            if v is None:
                where_parts.append(f"{quote_ident(k)} IS NULL")
            else:
                where_parts.append(f"{quote_ident(k)} = %s")
                params.append(v)
        sql = f"UPDATE {quote_ident(table)} SET {set_clause} WHERE {' AND '.join(where_parts)}"
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
                where_parts.append(f"{quote_ident(k)} IS NULL")
            else:
                where_parts.append(f"{quote_ident(k)} = %s")
                params.append(v)
        sql = f"DELETE FROM {quote_ident(table)} WHERE {' AND '.join(where_parts)}"
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
        sql = f"SELECT COUNT(*) AS cnt FROM {quote_ident(table)}"
        params: list[Any] = []
        conditions: list[str] = []
        if where:
            for k, v in where.items():
                _check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{quote_ident(k)} IS NULL")
                else:
                    conditions.append(f"{quote_ident(k)} = %s")
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
        col_sql_str = ", ".join(quote_ident(c) for c in columns)
        placeholders = ", ".join("%s" for _ in columns)
        sql = f"INSERT INTO {quote_ident(table)} ({col_sql_str}) VALUES ({placeholders})"
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

        col_sql = ", ".join(quote_ident(c) for c in columns)
        placeholders = ", ".join("%s" for _ in columns)
        # MySQL 8.0.19+ 用行别名 AS new（VALUES(col) 自 8.0.20 起弃用）；5.7 / MariaDB 回退 VALUES(col)。
        if update_columns:
            if self._supports_row_alias:
                assigned = ", ".join(
                    f"{quote_ident(c)} = new.{quote_ident(c)}" for c in update_columns
                )
                alias = " AS new"
            else:
                assigned = ", ".join(
                    f"{quote_ident(c)} = VALUES({quote_ident(c)})" for c in update_columns
                )
                alias = ""
            on_duplicate = f"ON DUPLICATE KEY UPDATE {assigned}"
        else:
            # 冲突时忽略：MySQL 没有 DO NOTHING，用 ON DUPLICATE KEY UPDATE id=id（自赋值即无操作）。
            # 该分支不引用新值，行别名无意义，故不追加 AS new。
            alias = ""
            on_duplicate = "ON DUPLICATE KEY UPDATE " + f"{quote_ident(keys[0])} = {quote_ident(keys[0])}"

        sql = f"INSERT INTO {quote_ident(table)} ({col_sql}) VALUES ({placeholders}){alias} {on_duplicate}"
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
                set_parts.append(f"{quote_ident(col)} = {val.sql}")
            else:
                set_parts.append(f"{quote_ident(col)} = %s")
                params.append(val)

        where_sql, where_params = self._compile_filters(query.filters)
        params.extend(where_params)
        sql = f"UPDATE {quote_ident(table)} SET {', '.join(set_parts)} WHERE {where_sql}"
        return self.execute(sql, tuple(params))["rowcount"]

    def delete_where(self, table: str, query: Query) -> int:
        """按声明式 Query 条件批量删除，返回影响行数，接口与 SQLiteClient 完全一致。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        if not query.filters:
            raise DatabaseValidationError("delete_where 的 query.filters 不能为空（防止误删全表）")
        _check_ident(table, "表名")
        where_sql, params = self._compile_filters(query.filters)
        sql = f"DELETE FROM {quote_ident(table)} WHERE {where_sql}"
        return self.execute(sql, tuple(params))["rowcount"]

