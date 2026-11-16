"""数据库客户端抽象基类。

参考主流框架：
- PEP 249（Python DB-API 2.0）的 Connection / Cursor 分层——把「连接」与「执行」拆开，
  本模块的 connect / close 对应其 Connection 生命周期；
- SQLAlchemy Core 的 Engine / Connection / Dialect 多后端抽象——同一套上层接口、多种
  方言实现（sqlite / mysql / postgresql ...），本模块取其「最小接口集」的克制版本；
- 同仓库 ``memory/memory_store_base.py``、``tools/tool_base.py`` 的「抽象基类 +
  子类实现最小接口集」组织方式。

统一约定所有后端（SQLite / MySQL / Postgres / ...）必须实现的最小接口集：
- 连接管理：connect / close
- DDL（建/删/检查表）：create_table / ensure_table / ensure_schema / table_exists / drop_table
- DML（增删改查）：insert / select / select_one / update / delete
- 事务：__enter__ / __exit__（with 语句）

所有参数都用「原生 Python 类型 + schema.py 里的声明式模型」表达，
上层（memory / logs / trace 等）只依赖这个抽象，不直接 import sqlite3。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from .errors import DatabaseValidationError
from .query import Filter, FilterOp, Query, SortOrder, SqlExpr
from .schema import check_ident, Column, Schema


class DatabaseClientBase(ABC):
    """数据库客户端抽象基类。子类必须实现下面所有方法。"""

    def __init__(self, timeout: float = 5.0):
        self._conn = None
        self.timeout = timeout
        self._tx_depth = 0

    # ---------- 连接管理 ----------
    @abstractmethod
    def _connect(self) -> None: ...

    def connect(self) -> None:
        if self._conn is None:
            self._connect()

    @abstractmethod
    def _close(self) -> None: ...

    def close(self) -> None:
        """关闭连接。"""
        if self._conn is not None:
            self._close()
            self._conn = None

    @property
    def _in_transaction(self) -> bool:
        """当前是否处于显式事务中（``_tx_depth`` 的只读派生）。

        唯一真实来源是 ``_tx_depth`` —— ``with db:`` 的事务嵌套深度计数器。
        本属性不保存任何独立状态，因此不可能出现"标志与底层真实状态失同步"。
        """
        return self._tx_depth > 0

    # ---------- 基础执行 ----------
    @abstractmethod
    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any]:
        """执行写 SQL（INSERT/UPDATE/DELETE/CREATE/DROP）。"""

    # ---------- 谓词下推：Query / Filter → 参数化 SQL（通用骨架，方言由 _quote_ident / _placeholder 承担） ----------
    def _compile_filter(self, f: Filter, idx: int) -> tuple[str, dict[str, Any]]:
        """把单个 Filter 编译成 (SQL 片段, 命名参数字典)。

        - 列名必须过 check_ident 白名单；
        - 列名渲染走 self._quote_ident（方言钩子）；
        - 占位符渲染走 self._placeholder（SQLite 返回 ``:name``、MySQL 返回 ``%(name)s``）；
        - 每个参数名用 idx 后缀唯一名（``f{idx}`` / ``f{idx}_{i}``），避免同列多条件冲突；
        - 值一律通过参数传入，杜绝拼接注入。
        """
        if not isinstance(f, Filter):
            raise TypeError(f"filters 列表元素必须是 Filter，收到 {type(f)}")
        check_ident(f.field, "字段名")
        col = self._quote_ident(f.field)
        op = f.op
        p_name = f"f{idx}"

        # IS NULL / IS NOT NULL：无参
        if op is FilterOp.IS_NULL:
            return f"{col} IS NULL", {}
        if op is FilterOp.NOT_NULL:
            return f"{col} IS NOT NULL", {}

        # IN：需要列表值，展开成 (:f0_0, :f0_1, ...) / (%(f0_0)s, %(f0_1)s, ...)
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
                keys.append(self._placeholder(k))
                params[k] = v
            return f"{col} IN ({', '.join(keys)})", params

        # 其它二元操作符（= != > >= < <= LIKE）：需要一个 value
        if f.value is None:
            raise DatabaseValidationError(
                f"Filter(field={f.field!r}, op={op.value}) 必须提供 value；"
                f"判空请用 IS_NULL / NOT_NULL"
            )
        # 允许 LIKE 的 value 带 % / _，不替用户补，保持最小惊讶
        return f"{col} {op.value} {self._placeholder(p_name)}", {p_name: f.value}

    def _compile_filters(self, filters: list[Filter] | None) -> tuple[str, dict[str, Any]]:
        """把 Filter 列表编译成 (WHERE 片段, 命名参数字典)；无过滤条件返回 ("", {})。

        与 ``_compile_query`` 里的过滤编译同源（都调用 ``self._compile_filter``），
        供 ``update_where`` / ``delete_where`` 复用，保证读写两侧谓词语义一致。
        """
        if not filters:
            return "", {}
        conds: list[str] = []
        params: dict[str, Any] = {}
        for i, f in enumerate(filters):
            frag, p = self._compile_filter(f, i)
            conds.append(frag)
            params.update(p)
        return " AND ".join(conds), params

    def _compile_query(self, q: Query) -> tuple[str, dict[str, Any]]:
        """把 Query 对象编译成 (SQL 语句, 命名参数字典)。

        编译规则（严格保守、不做 OR / JOIN / 子查询，保持最小可用）：
        - 表名 / columns / sort_by 字段名全部走 check_ident 白名单；
        - 多个 Filter 之间固定 AND；
        - sort_by 支持单字段 str 或 多字段 list[str]，方向对每列统一应用（ASC/DESC 枚举）；
        - limit / offset 强制 int 并校验非负；
        - 单独 OFFSET 无 LIMIT 时的兜底 SQL 由方言类属性 ``_OFFSET_NO_LIMIT_SQL`` 提供；
        - LIKE 的通配符由调用方在 value 里自带，不替用户加 %，避免"隐式模糊"。
        """
        check_ident(q.table_name, "表名")
        # 列投影：columns=None 保持 SELECT *；显式指定时逐列过标识符白名单，杜绝拼接注入
        if q.columns is None:
            col_clause = "*"
        else:
            if not q.columns:
                raise DatabaseValidationError("columns 不能是空列表")
            for c in q.columns:
                check_ident(c, "列名")
            col_clause = ", ".join(self._quote_ident(c) for c in q.columns)
        sql_parts: list[str] = [f"SELECT {col_clause} FROM {self._quote_ident(q.table_name)}"]
        params: dict[str, Any] = {}

        if q.filters:
            conds: list[str] = []
            for i, f in enumerate(q.filters):
                frag, p = self._compile_filter(f, i)
                conds.append(frag)
                params.update(p)
            sql_parts.append("WHERE " + " AND ".join(conds))

        if q.sort_by is not None:
            # 兼容单字段 str 与多字段 list[str] 两种写法；排序方向对每一列统一应用
            sort_cols = [q.sort_by] if isinstance(q.sort_by, str) else list(q.sort_by)
            if not sort_cols:
                raise DatabaseValidationError("sort_by 不能是空列表")
            for col in sort_cols:
                check_ident(col, "排序字段名")
            direction = "ASC" if q.sort_order is SortOrder.ASC else "DESC"
            order_by = ", ".join(f"{self._quote_ident(c)} {direction}" for c in sort_cols)
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
            # OFFSET 必须先有 LIMIT；没 LIMIT 时用方言类属性兜底
            if q.limit is None:
                sql_parts.append(self._OFFSET_NO_LIMIT_SQL)
            sql_parts.append(f"OFFSET {m}")

        return " ".join(sql_parts), params

    # 方言类常量：OFFSET 单独出现时的兜底 LIMIT 值；子类必须覆盖。
    _OFFSET_NO_LIMIT_SQL: str = ""

    # ------------ 方言 ------------
    # base 中声明为抽象方法
    @abstractmethod
    def _placeholder(self, name: str) -> str:
        """把参数名渲染成方言占位符文本"""

    @abstractmethod
    def _quote_ident(self, name: str) -> str:
        """方言标识符渲染钩子：SQLite 返回 "name"，MySQL 返回 `name`。"""

    @abstractmethod
    def _parse_columns(self, table: str, columns: list[Column]) -> tuple[list[str], list[str]]:
        """方言列解析钩子：把 list[Column] 解析为 (列定义SQL列表, 建索引SQL列表)。"""

    # ---------- DDL：建表 / 删表 / 查表 ----------
    @abstractmethod
    def table_exists(self, table: str) -> bool:
        """判断表是否已存在。"""

    @abstractmethod
    def _create_table_raw(self, table: str, col_defs: list[str], if_not_exists: bool) -> None:
        """执行原始 CREATE TABLE 语句的方言实现。"""

    @abstractmethod
    def _create_table_from_schema(self, schema: Schema, if_not_exists: bool) -> None:
        """从 Schema 建表+索引的方言实现。"""

    def create_table(self, table: str, columns: dict[str, str] | list[Column] | Schema, if_not_exists: bool = True) -> None:
        """建表。

        Args:
            table: 表名。
            columns: 列定义，支持三种形式，按方便程度任选：
                1. ``{列名: SQL类型字符串}`` —— 最简，例如 ``{"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL"}``；
                2. ``list[Column]`` —— 使用 ``schema.Column`` 声明式定义，自动处理 NOT NULL / DEFAULT / UNIQUE；
                3. ``Schema`` 对象 —— 一次性声明列 + 索引（推荐工程化使用）。
            if_not_exists: 是否加上 ``IF NOT EXISTS``（默认 True，幂等）。
        """
        check_ident(table, "表名")

        # 1) Schema 对象
        if isinstance(columns, Schema):
            if table and columns.table_name != table:
                raise DatabaseValidationError(f"Schema.table_name={columns.table_name!r} 与传入 table={table!r} 不一致")
            self._create_table_from_schema(columns, if_not_exists=if_not_exists)
            return

        # 2) list[Column]
        if isinstance(columns, list):
            if not columns:
                raise DatabaseValidationError("create_table 的 columns 不能为空")
            col_defs, index_stmts = self._parse_columns(table, columns)
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
                check_ident(col_name, "列名")
                if not isinstance(col_def, str) or not col_def.strip():
                    raise DatabaseValidationError(f"列 {col_name!r} 的类型定义必须是非空字符串")
                col_defs.append(f"{self._quote_ident(col_name)} {col_def.strip()}")
            self._create_table_raw(table, col_defs, if_not_exists=if_not_exists)
            return

        raise TypeError(f"columns 参数只支持 dict[str,str] / list[Column] / Schema 三种形式，收到 {type(columns)}")

    def ensure_table(self, table: str, columns: dict[str, str] | list[Any] | Schema) -> None:
        """幂等建表（等价 create_table(..., if_not_exists=True)）。

        语义：表存在就跳过、不存在就创建；**不会**修改已有表结构。
        如果需要「缺列就补列」的迁移，请显式调用 ``ensure_schema``（支持 ALTER 迁移的后端）。
        """
        self.create_table(table, columns, if_not_exists=True)

    @abstractmethod
    def ensure_schema(self, schema: Schema, migrate: bool = False) -> None:
        """按 ``Schema`` 对象幂等建表 + 建索引。

        Args:
            schema: ``schema.Schema`` 声明式模型，含 table_name / columns / indexes。
            migrate: 为 True 时尝试补列（仅 SQLite 支持 ADD COLUMN，已存在的列自动跳过）；
                     默认 False，只建表 + 建索引，不改动已有表结构（最安全）。
        """

    def drop_table(self, table: str, if_exists: bool = True) -> None:
        """删表。"""
        check_ident(table, "表名")
        if_ex = "IF EXISTS " if if_exists else ""
        self.execute(f"DROP TABLE {if_ex}{self._quote_ident(table)}")

    # ---------- DML：便捷 CRUD ----------
    def select(self, table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """查询多行，where 做等值匹配；值为 None 时自动编译为 IS NULL。"""
        check_ident(table, "表名")
        sql = f"SELECT * FROM {self._quote_ident(table)}"
        params: dict = {}
        if where:
            conditions: list[str] = []
            for k, v in where.items():
                check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{self._quote_ident(k)} IS NULL")
                else:
                    p_name = f"w_{k}"
                    conditions.append(f"{self._quote_ident(k)} = {self._placeholder(p_name)}")
                    params[p_name] = v
            sql += f" WHERE {' AND '.join(conditions)}"
        if order_by:
            # order_by 允许形如 "id DESC"，由调用方保证合法；仅作基本字符集校验
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$", order_by.strip()):
                raise DatabaseValidationError(f"非法 order_by: {order_by!r}")
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            # 与 _compile_query 同口径：limit 强制非负 int，负数直接报错——
            # SQLite 会把 LIMIT -1 当成"无上限"而静默返回全表，与调用方预期不符；MySQL 下 LIMIT -1 是语法错误。
            n = int(limit)
            if n < 0:
                raise DatabaseValidationError(f"limit 不能为负数: {n}")
            sql += f" LIMIT {n}"
        return self.query(sql, params)  # type: ignore

    def update(self, table: str, data: dict[str, Any], where: dict[str, Any]) -> int:
        """更新行，返回影响行数；where 值为 None 时编译为 IS NULL。"""
        if not data or not where:
            raise DatabaseValidationError("update 需要 data 和 where")
        check_ident(table, "表名")
        for k in data.keys():
            check_ident(k, "字段名")
        set_clause = ", ".join([f"{self._quote_ident(k)} = {self._placeholder(f'set_{k}')}" for k in data.keys()])
        where_conds: list[str] = []
        params: dict[str, Any] = {f"set_{k}": v for k, v in data.items()}
        for k, v in where.items():
            check_ident(k, "字段名")
            if v is None:
                where_conds.append(f"{self._quote_ident(k)} IS NULL")
            else:
                p_name = f"w_{k}"
                where_conds.append(f"{self._quote_ident(k)} = {self._placeholder(p_name)}")
                params[p_name] = v
        sql = f"UPDATE {self._quote_ident(table)} SET {set_clause} WHERE {' AND '.join(where_conds)}"
        return self.execute(sql, params)["rowcount"]

    def insert(self, table: str, data: dict[str, Any]) -> int:
        """插入一行，返回新行 id。"""
        if not data:
            raise DatabaseValidationError("insert 数据不能为空")
        check_ident(table, "表名")
        for k in data.keys():
            check_ident(k, "字段名")
        columns = ", ".join(self._quote_ident(k) for k in data.keys())
        placeholders = ", ".join(self._placeholder(k) for k in data.keys())
        sql = f"INSERT INTO {self._quote_ident(table)} ({columns}) VALUES ({placeholders})"
        return self.execute(sql, data)["lastrowid"]

    @abstractmethod
    def batch_insert(self, table: str, rows: list[dict]) -> int:
        """批量插入多行，返回插入行数；空列表直接返回0。"""


    def select_one(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
    ) -> dict[str, Any] | None:
        """查询单行，不存在返回 None。"""
        rows = self.select(table, where=where, order_by=order_by, limit=1)
        return rows[0] if rows else None

    def select_where(self, query: Any) -> list[dict]:
        """按声明式 Query 对象查询，支持 EQ/NEQ/GT/GTE/LT/LTE/LIKE/IN/IS[NOT]_NULL。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        sql, params = self._compile_query(query)
        return self.query(sql, params)  # type: ignore[return-value]

    def select_one_where(self, query: Query) -> dict[str, Any] | None:
        """select_where 的单行版本，自动 limit=1，无结果返回 None。"""
        if not isinstance(query, Query):
            raise TypeError(f"query 必须是 fast_agents.database.Query，收到 {type(query)}")
        limited = query.model_copy(update={"limit": 1, "offset": None})
        rows = self.select_where(limited)
        return rows[0] if rows else None

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
        check_ident(table, "表名")

        set_parts: list[str] = []
        params: dict[str, Any] = {}
        for i, (col, val) in enumerate(values.items()):
            check_ident(col, "字段名")
            if isinstance(val, SqlExpr):
                set_parts.append(f"{self._quote_ident(col)} = {val.sql}")
            else:
                p_name = f"s{i}"
                set_parts.append(f"{self._quote_ident(col)} = {self._placeholder(p_name)}")
                params[p_name] = val

        where_sql, where_params = self._compile_filters(query.filters)
        params.update(where_params)
        sql = f"UPDATE {self._quote_ident(table)} SET {', '.join(set_parts)} WHERE {where_sql}"
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
        check_ident(table, "表名")
        where_sql, params = self._compile_filters(query.filters)
        sql = f"DELETE FROM {self._quote_ident(table)} WHERE {where_sql}"
        return self.execute(sql, params)["rowcount"]

    @abstractmethod
    def count(self, table: str, where: dict | None = None) -> int:
        """按等值条件统计行数，where=None 返回表总行数。"""


    def delete(self, table: str, where: dict[str, Any]) -> int:
        """删除行，返回影响行数；where 值为 None 时编译为 IS NULL。"""
        if not where:
            raise DatabaseValidationError("delete 必须传 where 防止误删全表")
        check_ident(table, "表名")
        where_parts: list[str] = []
        params: dict[str, Any] = {}
        for k, v in where.items():
            check_ident(k, "字段名")
            if v is None:
                where_parts.append(f"{self._quote_ident(k)} IS NULL")
            else:
                p_name = f"w_{k}"
                where_parts.append(f"{self._quote_ident(k)} = {self._placeholder(p_name)}")
                params[p_name] = v
        sql = f"DELETE FROM {self._quote_ident(table)} WHERE {' AND '.join(where_parts)}"
        return self.execute(sql, params)["rowcount"]

    # ---------- 统计与批量写入：count / batch_insert ----------

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """按等值条件统计行数，where=None 返回表总行数；占位符统一用 w_{列名}。"""
        check_ident(table, "表名")
        sql = f"SELECT COUNT(*) AS cnt FROM {self._quote_ident(table)}"
        params: dict[str, Any] = {}
        conditions: list[str] = []
        if where:
            for k, v in where.items():
                check_ident(k, "字段名")
                if v is None:
                    conditions.append(f"{self._quote_ident(k)} IS NULL")
                else:
                    p_name = f"w_{k}"
                    conditions.append(f"{self._quote_ident(k)} = {self._placeholder(p_name)}")
                    params[p_name] = v
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        row = self.query(sql, params, fetch_one=True)
        return int(row["cnt"]) if row else 0

    def batch_insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        """批量插入多行（list of dict），返回插入行数；空列表直接返回 0。

        事务语义同 ``execute``/``executemany``：单独调用时自动提交，
        在 ``with db:`` 内由外层事务统一提交/回滚。
        占位符统一用列名（与 ``insert`` 一致），参数为 list[dict]。
        """
        if not rows:
            return 0
        check_ident(table, "表名")
        columns = list(rows[0].keys())
        for col in columns:
            check_ident(col, "字段名")
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
        col_sql_str = ", ".join(self._quote_ident(c) for c in columns)
        placeholders = ", ".join(self._placeholder(c) for c in columns)
        sql = f"INSERT INTO {self._quote_ident(table)} ({col_sql_str}) VALUES ({placeholders})"
        params_list = [{c: row[c] for c in columns} for row in rows]
        # 走 self.executemany 封装（而非 self.conn.executemany），
        # 复用其「不在事务中时自动 commit、异常自动 rollback」的逻辑——
        # 修复：单独调用 batch_insert 后 close，Python sqlite3 隐式事务被回滚导致数据静默丢失。
        self.executemany(sql, params_list)
        return len(rows)

    # ---------- 事务：with 语句 ----------
    @abstractmethod
    def __enter__(self) -> "DatabaseClientBase":
        """进入事务上下文（``with db:`` 时自动调用）。

        约定：最外层调用开启事务，嵌套重入应「加入父事务」（由实现决定是空操作、
        savepoint 还是深度计数），保证多段写入要么全成功、要么全回滚。
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """退出事务上下文：正常提交、异常回滚。

        返回值 False 表示异常继续向外抛出（不吞异常）；实现必须保证无论 commit /
        rollback 自身是否抛错，都复位内部事务状态，避免留下悬挂事务。
        """

    def __del__(self) -> None:
        """析构时尝试关闭连接。"""
        try:
            self.close()
        except Exception:
            pass
