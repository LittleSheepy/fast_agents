"""SQLite client —— fast_agents 最基础的数据库客户端。

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


class SQLiteClient(DatabaseClientBase):
    """SQLite 客户端，单连接、字典行结果、参数化查询、声明式建表。"""

    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0):
        """初始化 SQLite 连接。

        Args:
            db_path: 数据库文件路径，默认 ``":memory:"`` 内存数据库。
            timeout: 连接等待超时秒数（SQLite 写锁等待）。
        """
        self.db_path = db_path
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._in_transaction = False
        self._connect()

    # ---------- 连接管理 ----------
    def _connect(self) -> None:
        """建立连接并配置默认选项。"""
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            # 行返回字典格式
            self._conn.row_factory = sqlite3.Row
            # 外键约束默认启用
            self._conn.execute("PRAGMA foreign_keys = ON")
            # WAL 模式，并发读写更友好
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error as e:
            raise ValueError(f"连接 SQLite 失败: {e}") from e

    def connect(self) -> None:
        """显式 connect（基类接口）；若已关闭则重连。"""
        if self._conn is None:
            self._connect()

    @property
    def conn(self) -> sqlite3.Connection:
        """获取底层连接（如果关闭则自动重连）。"""
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        """关闭连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------- 基础执行 ----------
    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any]:
        """执行写 SQL（INSERT/UPDATE/DELETE/CREATE/DROP）。

        - 不在事务中时自动提交（单次语句一个隐式事务）；
        - 在显式事务中（`with db:`）时只执行不 commit，由 __exit__ 统一提交/回滚。
        """
        try:
            cur = self.conn.execute(sql, params)
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": cur.lastrowid, "rowcount": cur.rowcount}
        except sqlite3.Error as e:
            if not self._in_transaction:
                self.conn.rollback()
            raise ValueError(f"执行SQL失败: {e}\nSQL: {sql}") from e

    def executemany(self, sql: str, params_list: list[tuple | dict]) -> dict[str, Any]:
        """批量执行写 SQL。"""
        try:
            cur = self.conn.executemany(sql, params_list)
            if not self._in_transaction:
                self.conn.commit()
            return {"lastrowid": cur.lastrowid, "rowcount": cur.rowcount}
        except sqlite3.Error as e:
            if not self._in_transaction:
                self.conn.rollback()
            raise ValueError(f"批量执行SQL失败: {e}") from e

    def executescript(self, sql: str) -> None:
        """执行一段脚本（多条 SQL，以 ; 分隔），不在事务中时自动提交。"""
        try:
            self.conn.executescript(sql)
            if not self._in_transaction:
                self.conn.commit()
        except sqlite3.Error as e:
            if not self._in_transaction:
                self.conn.rollback()
            raise ValueError(f"执行脚本失败: {e}\nSQL: {sql}") from e

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
            raise ValueError(f"查询失败: {e}\nSQL: {sql}") from e

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
                raise ValueError(f"Schema.table_name={columns.table_name!r} 与传入 table={table!r} 不一致")
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
                raise ValueError("create_table 的 columns 不能为空")
            col_defs = []
            for col_name, col_def in columns.items():
                _check_ident(col_name, "列名")
                if not isinstance(col_def, str) or not col_def.strip():
                    raise ValueError(f"列 {col_name!r} 的类型定义必须是非空字符串")
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
            raise ValueError("insert 数据不能为空")
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
            conditions = " AND ".join([f"{_quote_ident(k)} = :w_{k}" for k in where.keys()])
            sql += f" WHERE {conditions}"
            params = {f"w_{k}": v for k, v in where.items()}
        if order_by:
            # order_by 允许形如 "id DESC"，由调用方保证合法；仅作基本字符集校验
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$", order_by.strip()):
                raise ValueError(f"非法 order_by: {order_by!r}")
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
                raise ValueError(
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
            raise ValueError(
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
        - sort_by 只允许单字段 + ASC/DESC 枚举方向；多字段排序留到后续版本；
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
            _check_ident(q.sort_by, "排序字段名")
            direction = "ASC" if q.sort_order is SortOrder.ASC else "DESC"
            sql_parts.append(f"ORDER BY {_quote_ident(q.sort_by)} {direction}")

        if q.limit is not None:
            n = int(q.limit)
            if n < 0:
                raise ValueError(f"limit 不能为负数: {n}")
            sql_parts.append(f"LIMIT {n}")

        if q.offset is not None:
            m = int(q.offset)
            if m < 0:
                raise ValueError(f"offset 不能为负数: {m}")
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
            raise ValueError("update 需要 data 和 where")
        _check_ident(table, "表名")
        set_clause = ", ".join([f"{_quote_ident(k)} = :set_{k}" for k in data.keys()])
        where_clause = " AND ".join([f"{_quote_ident(k)} = :w_{k}" for k in where.keys()])
        sql = f"UPDATE {_quote_ident(table)} SET {set_clause} WHERE {where_clause}"
        params = {f"set_{k}": v for k, v in data.items()}
        params.update({f"w_{k}": v for k, v in where.items()})
        return self.execute(sql, params)["rowcount"]

    def delete(self, table: str, where: dict[str, Any]) -> int:
        """删除行，返回影响行数。"""
        if not where:
            raise ValueError("delete 必须传 where 防止误删全表")
        _check_ident(table, "表名")
        where_clause = " AND ".join([f"{_quote_ident(k)} = :w_{k}" for k in where.keys()])
        sql = f"DELETE FROM {_quote_ident(table)} WHERE {where_clause}"
        params = {f"w_{k}": v for k, v in where.items()}
        return self.execute(sql, params)["rowcount"]

    # ---------- 事务上下文管理 ----------
    def __enter__(self) -> "SQLiteClient":
        """进入 with 块，显式开启事务。"""
        self.conn.execute("BEGIN")
        self._in_transaction = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """退出 with 块：正常提交，异常回滚。"""
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self._in_transaction = False
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
                raise ValueError(msg)
        col_sql_str = ", ".join(_quote_ident(c) for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {_quote_ident(table)} ({col_sql_str}) VALUES ({placeholders})"
        params_list = [tuple(row[c] for c in columns) for row in rows]
        try:
            self.conn.executemany(sql, params_list)
        except sqlite3.Error as e:
            raise ValueError(f"批量插入失败: {e}") from e
        return len(rows)

    def __del__(self) -> None:
        """析构时尝试关闭连接。"""
        try:
            self.close()
        except Exception:
            pass
