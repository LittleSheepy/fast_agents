"""SQLite client —— fast_agents 最基础的数据库客户端。

参考主流框架：
- PEP 249（Python DB-API 2.0）的 Connection / Cursor 执行模型（底层由标准库 ``sqlite3`` 实现）；
- SQLAlchemy Core 的 ``Connection.execute`` / ``ExecResult``——参数化执行 + 类字典结果集，
  本模块 ``execute`` 返回 ``{"lastrowid", "rowcount"}``、``query`` 返回 ``list[dict]`` 即对齐其思路；
- 标准库 ``sqlite3`` 的 ``row_factory = sqlite3.Row``（按键取值），本模块统一把查询结果转成
  ``dict``，上层不必感知 ``sqlite3.Row``。

零抽象、零新依赖，封装标准库 sqlite3 的连接管理与增删改查，
返回字典格式结果，支持事务上下文管理；并提供基于 ``schema.py``
的**声明式、幂等建表**能力，上层业务模块不需要自己拼 ``CREATE TABLE``。

用法:
    from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index

    db = SQLiteClient("app.db")

    # 方式一：最简，dict 描述列
    db.ensure_table("users", {
        "id":   "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
        "age":  "INTEGER",
    })

    # 方式二：声明式 Schema（推荐），自动建索引
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

import re
import sqlite3
from typing import Any, Optional

from .database_client_base import DatabaseClientBase
from .errors import (
    DatabaseConnectionError,
    DatabaseOperationalError,
    DatabaseValidationError,
)
from .query import Filter, FilterOp, Query, SortOrder
from .schema import (
    Column,
    Index,
    Schema,
    check_ident as _check_ident,
    column_sql as _column_sql,
    index_stmt as _index_stmt,
    parse_column_list,
    parse_schema,
    quote_ident as _quote_ident,
)


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


class SQLiteClient(DatabaseClientBase):
    """SQLite 客户端，单连接、字典行结果、参数化查询、声明式建表。"""

    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0):
        """初始化 SQLite 连接。

        Args:
            db_path: 数据库文件路径，默认 ``":memory:"`` 内存数据库。
            timeout: 连接等待超时秒数（SQLite 写锁等待）。
        """
        super().__init__(timeout)
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # 事务状态的唯一真实来源：
        #   _tx_depth          —— with 事务嵌套深度（0 表示不在事务中）
        #   _tx_must_rollback  —— 内层出错标记，最外层据此决定回滚
        # 是否在事务中由只读属性 _in_transaction 派生（_tx_depth > 0），
        # 不再单独保存布尔标志，从根本上消除"标志与真实状态失同步"的可能。
        self._tx_depth = 0
        self._tx_must_rollback = False
        self._connect()

    # ---------- 连接管理 ----------
    def _connect(self) -> None:
        """建立连接并配置默认选项。"""
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                detect_types=sqlite3.PARSE_DECLTYPES,
                isolation_level=None,  # 关闭 Python 隐式事务，事务完全由代码控制
            )
            # 行返回字典格式
            self._conn.row_factory = sqlite3.Row
            # 外键约束默认启用
            self._conn.execute("PRAGMA foreign_keys = ON")
            # 写锁等待：显式设 busy_timeout，与连接 timeout 同口径（默认 5s → 5000ms）；
            # memory 等上层如需 10s，构造时传 timeout=10.0 即可，无需另写 PRAGMA
            self._conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
            # WAL 模式，并发读写更友好
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"连接 SQLite 失败: {e}") from e

    @property
    def conn(self) -> sqlite3.Connection:
        """获取底层连接（如果关闭则自动重连）。"""
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

    @property
    def _in_transaction(self) -> bool:
        """当前是否处于显式事务中（``_tx_depth`` 的只读派生）。

        唯一真实来源是 ``_tx_depth`` —— ``with db:`` 的事务嵌套深度计数器。
        本属性不保存任何独立状态，因此不可能出现"标志与底层真实状态失同步"。
        """
        return self._tx_depth > 0

    def _close(self) -> None:
        """关闭连接。若仍残留未结束事务，先尝试回滚避免悬空事务。"""
        if self._conn is not None:
            # 标志与真实状态同步：关闭前若还在事务中（无论怎么进来的），先尝试回滚，
            # 避免因意外路径留下未提交事务导致 close() 时被隐式回滚或抛异常。
            if self._tx_depth > 0:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                self._tx_depth = 0
                self._tx_must_rollback = False
            self._conn.close()
            self._conn = None

    # ---------- 基础执行 ----------
    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any]:
        """执行写 SQL（INSERT/UPDATE/DELETE/CREATE/DROP）。

        - 不在事务中时自动提交（单次语句一个隐式事务）；
        - 在显式事务中（`with db:`）时只执行不 commit，由 __exit__ 统一提交/回滚；
        - 无论是否在事务中，只要执行失败都保证：
            1) 连接状态可用（不残留半事务）；
            2) 事务状态（``_tx_depth``）不被破坏——失败不改变嵌套深度。
        """
        try:
            cur = self.conn.execute(sql, params)
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": cur.lastrowid, "rowcount": cur.rowcount}
        except sqlite3.Error as e:
            # 异常路径统一清理：
            # - 事务外（_tx_depth == 0）：单语句自成事务，失败直接 rollback 清理半事务；
            # - 事务内（_tx_depth > 0）：不打断事务边界，交由外层 __exit__ 统一 commit/rollback。
            # 事务状态以 _tx_depth 为唯一来源，此处不再维护任何独立标志。
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"执行SQL失败: {e}\nSQL: {sql}") from e

    def executemany(self, sql: str, params_list: list[tuple | dict]) -> dict[str, Any]:
        """批量执行写 SQL。"""
        try:
            cur = self.conn.executemany(sql, params_list)
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": cur.lastrowid, "rowcount": cur.rowcount}
        except sqlite3.Error as e:
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"批量执行SQL失败: {e}") from e

    def executescript(self, sql: str) -> None:
        """执行一段脚本（多条 SQL，以 ; 分隔），不在事务中时自动提交。"""
        try:
            self.conn.executescript(sql)
            if not self._in_transaction:
                self.conn.commit()
        except sqlite3.Error as e:
            self._sync_transaction_state_after_error()
            raise DatabaseOperationalError(f"执行脚本失败: {e}\nSQL: {sql}") from e

    def _sync_transaction_state_after_error(self) -> None:
        """执行 SQL 出错后清理可能残留的半事务。

        仅在 execute/executemany/executescript 的异常分支调用。

        事务状态以 ``_tx_depth`` 为唯一真实来源（``_in_transaction`` 是其只读派生），
        因此本方法**不再维护任何标志**，只保留必要的连接级清理：

        - 事务外（``_tx_depth == 0``）：单语句自成事务，失败时 rollback 清理半事务
          （底层无活动事务时 rollback 为 no-op，安全，保持原行为）；
        - 事务内（``_tx_depth > 0``）：**不处理**——事务边界完全由 ``__enter__`` / ``__exit__``
          成对维护；commit / rollback 在无活动事务时均为 no-op，故即使底层因严重错误
          自行 abort 了事务，``__exit__`` 也不会出错，无需在此同步标志。
        """
        if self._conn is None:
            return
        if self._tx_depth == 0:
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass

    def query(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
        fetch_one: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        """执行读 SQL（SELECT / PRAGMA），返回字典列表。"""
        try:
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
            if fetch_one:
                return dict(rows[0]) if rows else None
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseOperationalError(f"查询失败: {e}\nSQL: {sql}") from e

    # ---------- DDL：建表 / 查表 / 删表 / 索引 ----------
    def table_exists(self, table: str) -> bool:
        """判断表是否已存在。"""
        _check_ident(table, "表名")
        row = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=:t",
            {"t": table},
            fetch_one=True,
        )
        return row is not None

    def _list_columns(self, table: str) -> set[str]:
        """列出已存在表的列名集合（小写），用于迁移补列。"""
        _check_ident(table, "表名")
        rows = self.query(f"PRAGMA table_info({_quote_ident(table)})")
        return {r["name"] for r in rows}  # type: ignore[index]

    def _list_indexes(self, table: str) -> set[str]:
        """列出已存在表的索引名集合，用于幂等建索引。"""
        _check_ident(table, "表名")
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t",
            {"t": table},
        )
        return {r["name"] for r in rows}  # type: ignore[index]

    def create_table(
        self,
        table: str,
        columns: dict[str, str] | list[Column] | Schema,
        if_not_exists: bool = True,
    ) -> None:
        """建表。三种 columns 写法见基类说明。"""
        _check_ident(table, "表名")

        # 1) Schema 对象
        if isinstance(columns, Schema):
            if table and columns.table_name != table:
                raise DatabaseValidationError(f"Schema.table_name={columns.table_name!r} 与传入 table={table!r} 不一致")
            self._create_table_from_schema(columns, if_not_exists=if_not_exists)
            return

        # 2) list[Column] —— 解析逻辑统一在 schema.parse_column_list
        if isinstance(columns, list):
            col_defs, index_stmts = parse_column_list(table, columns)
            self._create_table_raw(table, col_defs, if_not_exists=if_not_exists)
            for stmt in index_stmts:
                self.execute(stmt)
            return

        # 3) dict[str, str] —— 最宽松，直接信任调用方写的列类型片段
        if isinstance(columns, dict):
            if not columns:
                raise DatabaseValidationError("create_table 的 columns 不能为空")
            col_defs = []
            for col_name, col_def in columns.items():
                _check_ident(col_name, "列名")
                if not isinstance(col_def, str) or not col_def.strip():
                    raise DatabaseValidationError(f"列 {col_name!r} 的类型定义必须是非空字符串")
                col_defs.append(f"{_quote_ident(col_name)} {col_def.strip()}")
            self._create_table_raw(table, col_defs, if_not_exists=if_not_exists)
            return

        raise TypeError(
            "columns 参数只支持 dict[str,str] / list[Column] / Schema 三种形式，"
            f"收到 {type(columns)}"
        )

    def _create_table_raw(self, table: str, col_defs: list[str], if_not_exists: bool) -> None:
        """执行原始 CREATE TABLE 语句。"""
        if_not = "IF NOT EXISTS " if if_not_exists else ""
        sql = f"""CREATE TABLE {if_not}{_quote_ident(table)} ({",".join(col_defs)})"""
        self.execute(sql)

    def _create_table_from_schema(self, schema: Schema, if_not_exists: bool) -> None:
        """从 Schema 对象建表 + 索引（解析逻辑统一在 schema.parse_schema）。"""
        # 列定义 + 列级索引 + 显式 indexes，一次性解析出来；
        # 索引语句均为 CREATE INDEX IF NOT EXISTS，天然幂等，无需预先查已存在索引。
        col_defs, index_stmts = parse_schema(schema)
        self._create_table_raw(schema.table_name, col_defs, if_not_exists=if_not_exists)
        for stmt in index_stmts:
            self.execute(stmt)

    def ensure_table(self, table: str, columns: dict[str, str] | list[Column] | Schema) -> None:
        """幂等建表（等价 create_table(..., if_not_exists=True)）。"""
        self.create_table(table, columns, if_not_exists=True)

    def ensure_schema(self, schema: Schema, migrate: bool = False) -> None:
        """按 Schema 幂等建表 + 建索引；migrate=True 时尝试补齐缺失列（ADD COLUMN）。"""
        if not isinstance(schema, Schema):
            raise TypeError("schema 必须是 fast_agents.database.Schema 实例")
        _check_ident(schema.table_name, "表名")

        if not self.table_exists(schema.table_name):
            self._create_table_from_schema(schema, if_not_exists=True)
            return

        # 表已存在：migrate=True 时补缺失列（SQLite 只支持 ADD COLUMN，且加的列不能是 NOT NULL 无默认值）
        if migrate:
            existing_cols = {c.lower() for c in self._list_columns(schema.table_name)}
            for c in schema.columns:
                if c.name.lower() in existing_cols:
                    continue
                # SQLite 限制：ADD COLUMN 不能加 PRIMARY KEY / UNIQUE（较老版本），
                # 也不能加 NOT NULL 且无 DEFAULT 的列。这里直接把 SQL 拼出来执行，
                # 遇到不支持的场景让 sqlite3 抛错，比静默跳过更安全。
                col_sql = _column_sql(c)
                self.execute(f"ALTER TABLE {_quote_ident(schema.table_name)} ADD COLUMN {col_sql}")

        # 索引始终幂等创建（CREATE INDEX IF NOT EXISTS 本身幂等，解析逻辑统一在 schema.parse_schema）
        _, index_stmts = parse_schema(schema)
        for stmt in index_stmts:
            self.execute(stmt)

    def drop_table(self, table: str, if_exists: bool = True) -> None:
        """删表。"""
        _check_ident(table, "表名")
        if_ex = "IF EXISTS " if if_exists else ""
        self.execute(f"DROP TABLE {if_ex}{_quote_ident(table)}")

    # ---------- 便捷 CRUD ----------
    def insert(self, table: str, data: dict[str, Any]) -> int:
        """插入一行，返回新行 id。"""
        if not data:
            raise DatabaseValidationError("insert 数据不能为空")
        _check_ident(table, "表名")
        columns = ", ".join(_quote_ident(k) for k in data.keys())
        placeholders = ", ".join([f":{k}" for k in data.keys()])
        sql = f"INSERT INTO {_quote_ident(table)} ({columns}) VALUES ({placeholders})"
        return self.execute(sql, data)["lastrowid"]

    def select(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询多行，where 做等值匹配。"""
        _check_ident(table, "表名")
        sql = f"SELECT * FROM {_quote_ident(table)}"
        params: dict = {}
        if where:
            conditions: list[str] = []
            for k, v in where.items():
                if v is None:
                    conditions.append(f"{_quote_ident(k)} IS NULL")
                else:
                    conditions.append(f"{_quote_ident(k)} = :w_{k}")
                    params[f"w_{k}"] = v
            sql += f" WHERE {' AND '.join(conditions)}"
        if order_by:
            # order_by 允许形如 "id DESC"，由调用方保证合法；仅作基本字符集校验
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$", order_by.strip()):
                raise DatabaseValidationError(f"非法 order_by: {order_by!r}")
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, params)  # type: ignore

    # ---------- 谓词下推：Query / Filter → 参数化 SQL ----------
    @staticmethod
    def _compile_filter(f: Filter, idx: int) -> tuple[str, dict[str, Any]]:
        """把单个 Filter 编译成 (SQL 片段, 参数字典)。

        - 列名必须过 _check_ident 白名单；
        - 值一律通过命名参数 ``:f{idx}...`` 传入，杜绝拼接注入；
        - 每个参数名用 idx 后缀唯一名，避免同列多条件时参数名冲突。
        """
        if not isinstance(f, Filter):
            raise TypeError(f"filters 列表元素必须是 Filter，收到 {type(f)}")
        _check_ident(f.field, "字段名")
        col = _quote_ident(f.field)
        op = f.op
        p_name = f"f{idx}"

        # IS NULL / IS NOT NULL：无参
        if op is FilterOp.IS_NULL:
            return f"{col} IS NULL", {}
        if op is FilterOp.NOT_NULL:
            return f"{col} IS NOT NULL", {}

        # IN：需要列表值，展开成 (:f0_0, :f0_1, ...)
        if op is FilterOp.IN:
            if not isinstance(f.value, (list, tuple)) or len(f.value) == 0:
                raise DatabaseValidationError(
                    f"Filter(field={f.field!r}, op=IN) 的 value 必须是非空 list/tuple，"
                    f"收到 {type(f.value).__name__}"
                )
            keys: list[str] = []
            params: dict[str, Any] = {}
            for i, v in enumerate(f.value):
                k = f"{p_name}_{i}"
                keys.append(f":{k}")
                params[k] = v
            return f"{col} IN ({', '.join(keys)})", params

        # 其它二元操作符（= != > >= < <= LIKE）：需要一个 value
        if f.value is None:
            raise DatabaseValidationError(
                f"Filter(field={f.field!r}, op={op.value}) 必须提供 value；"
                f"判空请用 IS_NULL / NOT_NULL"
            )
        # 允许 LIKE 的 value 带 % / _，不替用户补，保持最小惊讶
        return f"{col} {op.value} :{p_name}", {p_name: f.value}

    @staticmethod
    def _compile_query(q: Query) -> tuple[str, dict[str, Any]]:
        """把 Query 对象编译成 (SQL 语句, 命名参数字典)。

        编译规则（严格保守、不做 OR / JOIN / 子查询，保持最小可用）：
        - 表名 / sort_by 字段名全部走 _check_ident 白名单；
        - 多个 Filter 之间固定 AND；
        - sort_by 支持单字段 str 或 多字段 list[str]，方向对每列统一应用（ASC/DESC 枚举）；
        - limit / offset 强制 int 并校验非负；
        - LIKE 的通配符由调用方在 value 里自带，不替用户加 %，避免"隐式模糊"。
        """
        _check_ident(q.table_name, "表名")
        sql_parts: list[str] = [f"SELECT * FROM {_quote_ident(q.table_name)}"]
        params: dict[str, Any] = {}

        if q.filters:
            conds: list[str] = []
            for i, f in enumerate(q.filters):
                frag, p = SQLiteClient._compile_filter(f, i)
                conds.append(frag)
                params.update(p)
            sql_parts.append("WHERE " + " AND ".join(conds))

        if q.sort_by is not None:
            # 兼容单字段 str 与多字段 list[str] 两种写法；排序方向对每一列统一应用
            sort_cols = [q.sort_by] if isinstance(q.sort_by, str) else list(q.sort_by)
            if not sort_cols:
                raise DatabaseValidationError("sort_by 不能是空列表")
            for col in sort_cols:
                _check_ident(col, "排序字段名")
            direction = "ASC" if q.sort_order is SortOrder.ASC else "DESC"
            order_by = ", ".join(f"{_quote_ident(c)} {direction}" for c in sort_cols)
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
            # SQLite 里 OFFSET 必须先有 LIMIT；为最小惊讶，没 LIMIT 时自动补一个大数兜底
            if q.limit is None:
                sql_parts.append("LIMIT -1")  # SQLite 支持 LIMIT -1 表示"无上限"
            sql_parts.append(f"OFFSET {m}")

        return " ".join(sql_parts), params

    def select_where(self, query: Query) -> list[dict[str, Any]]:
        """按声明式 Query 查询（谓词下推入口）。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        sql, params = self._compile_query(query)
        return self.query(sql, params)  # type: ignore[return-value]

    def select_one_where(self, query: Query) -> dict[str, Any] | None:
        """按声明式 Query 查询单行（自动 LIMIT 1）。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        # 不修改传入对象，构造一个只带 limit=1 的副本用于 SQL 生成
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
        set_clause = ", ".join([f"{_quote_ident(k)} = :set_{k}" for k in data.keys()])
        where_conds: list[str] = []
        params: dict[str, Any] = {f"set_{k}": v for k, v in data.items()}
        for k, v in where.items():
            if v is None:
                where_conds.append(f"{_quote_ident(k)} IS NULL")
            else:
                where_conds.append(f"{_quote_ident(k)} = :w_{k}")
                params[f"w_{k}"] = v
        sql = f"UPDATE {_quote_ident(table)} SET {set_clause} WHERE {' AND '.join(where_conds)}"
        return self.execute(sql, params)["rowcount"]

    def delete(self, table: str, where: dict[str, Any]) -> int:
        """删除行，返回影响行数。"""
        if not where:
            raise DatabaseValidationError("delete 必须传 where 防止误删全表")
        _check_ident(table, "表名")
        where_conds: list[str] = []
        params: dict[str, Any] = {}
        for k, v in where.items():
            if v is None:
                where_conds.append(f"{_quote_ident(k)} IS NULL")
            else:
                where_conds.append(f"{_quote_ident(k)} = :w_{k}")
                params[f"w_{k}"] = v
        sql = f"DELETE FROM {_quote_ident(table)} WHERE {' AND '.join(where_conds)}"
        return self.execute(sql, params)["rowcount"]

    # ---------- 事务上下文管理 ----------
    def __enter__(self) -> "SQLiteClient":
        """进入 with 块：最外层显式开启事务（BEGIN IMMEDIATE），内层重入直接加入父事务。

        嵌套 with db:（例如 memory 层 _connect() 内再 with db:）不会重复发 BEGIN，
        而是把事务深度计数器加一，等最外层 __exit__ 统一 commit/rollback——
        内层异常会标记 _tx_must_rollback，保证整体原子性。
        """
        if self._tx_depth == 0:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as e:
                raise DatabaseOperationalError(f"开启事务失败: {e}") from e
            self._tx_must_rollback = False
        self._tx_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """退出 with 块：内层直接 depth--（异常时打回滚标记）；最外层统一提交/回滚。

        保证无论 commit / rollback 自身是否抛错，``_tx_depth`` / ``_tx_must_rollback``
        都会被复位（``_in_transaction`` 随之自动派生为 False）。
        """
        self._tx_depth -= 1
        if self._tx_depth > 0:
            # 内层退出：不 commit/rollback，由最外层兜底；有异常时标记必须回滚
            if exc_type is not None:
                self._tx_must_rollback = True
            return False
        # 最外层退出：统一处理 commit/rollback，无论是否抛错都复位 depth
        # 无论 SQLite 是否因严重错误自行 abort 了事务，这里都要兜住：
        #   - commit 失败（例如约束冲突、磁盘错误、schema 变更导致事务被 abort）→
        #     尝试 rollback，把异常继续往外抛，保持原子性契约
        #   - rollback 失败（例如"no transaction is active"——底层已自行回滚）→
        #     吞掉，状态正常复位
        try:
            if exc_type is None and not self._tx_must_rollback:
                try:
                    self.conn.commit()
                except sqlite3.Error as e:
                    # commit 失败：尽力回滚后用统一异常抛出，不让驱动原生异常泄漏给上层
                    try:
                        self.conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise DatabaseOperationalError(f"提交事务失败: {e}") from e
            else:
                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass
        finally:
            self._tx_depth = 0
            self._tx_must_rollback = False
        return False


    # ---------- 统计与批量写入：count / batch_insert ----------
    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """按等值条件统计行数，where=None 返回表总行数。"""
        _check_ident(table, "表名")
        sql = f"SELECT COUNT(*) AS cnt FROM {_quote_ident(table)}"
        params: list[Any] = []
        conditions: list[str] = []
        if where:
            for k, v in where.items():
                _check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{_quote_ident(k)} IS NULL")
                else:
                    conditions.append(f"{_quote_ident(k)} = ?")
                    params.append(v)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        row = self.query(sql, tuple(params), fetch_one=True)
        return int(row["cnt"]) if row else 0

    def batch_insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        """批量插入多行，返回插入行数；空列表直接返回0。

        事务语义同 ``execute``/``executemany``：单独调用时自动提交，
        在 ``with db:`` 内由外层事务统一提交/回滚。
        """
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
        col_sql_str = ", ".join(_quote_ident(c) for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {_quote_ident(table)} ({col_sql_str}) VALUES ({placeholders})"
        params_list = [tuple(row[c] for c in columns) for row in rows]
        # 走 self.executemany 封装（而非 self.conn.executemany），
        # 复用其「不在事务中时自动 commit、异常自动 rollback」的逻辑——
        # 修复：单独调用 batch_insert 后 close，Python sqlite3 隐式事务被回滚导致数据静默丢失。
        self.executemany(sql, params_list)
        return len(rows)

    # ---------- 条件写入：upsert / update_where / delete_where ----------
    @staticmethod
    def _compile_filters(filters: list[Filter] | None) -> tuple[str, dict[str, Any]]:
        """把 Filter 列表编译成 (WHERE 片段, 命名参数字典)；无过滤条件返回 ("", {})。

        与 ``_compile_query`` 里的过滤编译同源（都调用 ``_compile_filter``），
        供 ``update_where`` / ``delete_where`` 复用，保证读写两侧谓词语义一致。
        """
        if not filters:
            return "", {}
        conds: list[str] = []
        params: dict[str, Any] = {}
        for i, f in enumerate(filters):
            frag, p = SQLiteClient._compile_filter(f, i)
            conds.append(frag)
            params.update(p)
        return " AND ".join(conds), params

    def upsert(
        self,
        table: str,
        data: dict[str, Any],
        conflict_keys: str | list[str],
        update_columns: list[str] | None = None,
    ) -> int:
        """插入或更新（``INSERT ... ON CONFLICT(...) DO UPDATE``），返回影响行数。

        Args:
            table: 表名。
            data: 待写入的一行（列名 → 值），列名走白名单校验。
            conflict_keys: 冲突判定列（主键 / UNIQUE 列），单列传 str、多列传 list[str]，
                **必须在 data 中**。
            update_columns: 冲突时需要更新的列；默认更新 ``data`` 中除 ``conflict_keys``
                外的所有列；显式传空列表则冲突时 ``DO NOTHING``。

        依赖 SQLite 3.24+ 的 UPSERT 语法（Python 3.10 自带 sqlite3 已满足）；``conflict_keys``
        对应的列必须有 PRIMARY KEY / UNIQUE 约束，否则 SQLite 会报错。值全部走命名参数。
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

        col_sql = ", ".join(_quote_ident(c) for c in columns)
        placeholders = ", ".join(f":{c}" for c in columns)
        conflict_sql = ", ".join(_quote_ident(k) for k in keys)
        if update_columns:
            assigned = ", ".join(
                f"{_quote_ident(c)} = excluded.{_quote_ident(c)}" for c in update_columns
            )
            action = f"DO UPDATE SET {assigned}"
        else:
            action = "DO NOTHING"
        sql = (
            f"INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_sql}) {action}"
        )
        return self.execute(sql, data)["rowcount"]

    def update_where(self, table: str, values: dict[str, Any], query: Query) -> int:
        """按声明式 ``Query`` 条件批量更新，返回影响行数。

        与 ``update(table, data, where)`` 的区别：``where`` 只能等值匹配，这里走 Filter
        谓词（支持 ``> / < / >= / <= / != / LIKE / IN / IS NULL``）。

        Args:
            values: 要更新的列；普通值走参数化，需要「自增 / 函数计算」时用
                ``SqlExpr("access_count + 1")`` 显式包裹（内容原样拼入 SQL，只传固定片段）。
            query: 过滤条件来源；**``filters`` 必须非空**，防止误更新全表。

        安全约定：``values`` 的键、``query`` 的过滤字段都走标识符白名单；只有 ``SqlExpr``
        包装的内容才原样拼入，其它值一律参数化。
        """
        if not values:
            raise DatabaseValidationError("update_where 需要 values")
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        if not query.filters:
            raise DatabaseValidationError("update_where 的 query.filters 不能为空（防止误更新全表）")
        _check_ident(table, "表名")

        set_parts: list[str] = []
        params: dict[str, Any] = {}
        for i, (col, val) in enumerate(values.items()):
            _check_ident(col, "字段名")
            if isinstance(val, SqlExpr):
                set_parts.append(f"{_quote_ident(col)} = {val.sql}")
            else:
                p_name = f"s{i}"
                set_parts.append(f"{_quote_ident(col)} = :{p_name}")
                params[p_name] = val

        where_sql, where_params = self._compile_filters(query.filters)
        params.update(where_params)
        sql = f"UPDATE {_quote_ident(table)} SET {', '.join(set_parts)} WHERE {where_sql}"
        return self.execute(sql, params)["rowcount"]

    def delete_where(self, table: str, query: Query) -> int:
        """按声明式 ``Query`` 条件批量删除，返回影响行数。

        与 ``delete(table, where)`` 的区别：``where`` 只能等值，这里走 Filter 谓词。
        与 ``delete`` 保持一致的安全约定：**``query.filters`` 必须非空**，防止误删全表；
        真要删全表请显式 ``db.execute("DELETE FROM ...")``。
        """
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        if not query.filters:
            raise DatabaseValidationError("delete_where 的 query.filters 不能为空（防止误删全表）")
        _check_ident(table, "表名")
        where_sql, params = self._compile_filters(query.filters)
        sql = f"DELETE FROM {_quote_ident(table)} WHERE {where_sql}"
        return self.execute(sql, params)["rowcount"]

    def __del__(self) -> None:
        """析构时尝试关闭连接。"""
        try:
            self.close()
        except Exception:
            pass
