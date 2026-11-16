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
    Schema,
    check_ident as _check_ident,
    column_sql as _column_sql,
    parse_column_list,
    parse_schema,
    quote_ident as _quote_ident,
)


def _split_sql_statements(sql: str) -> list[str]:
    """把一段多语句 SQL 按顶层分号切分为单条语句（已 strip、已过滤空串）。

    比 ``sql.split(";")`` 更稳：只有出现在「语句正文」里的分号才切分，出现在字符串
    字面量 / 标识符引用 / 注释里的分号不会被误切。覆盖 SQLite 常见写法：

    - 单引号字符串 ``'a;b'``（内部单引号用两个单引号转义）；
    - 双引号标识符 ``"a;b"``（内部双引号用两个双引号转义）；
    - 反引号标识符（兼容 MySQL 风格反引号，SQLite 遇到也安全跳过）；
    - 行注释 ``-- ...`` 到行尾结束（SQLite 规则：``--`` 后不要求紧跟空白，
      例如 ``--1;`` 整行都是注释，与 MySQL 的 ``-- ` 后需空白规则不同）；
    - 块注释 ``/* ... */``，可跨行。

    关键原因：``sqlite3.Connection.executescript`` 会在执行前隐式 ``COMMIT`` 当前
    事务（标准库官方文档明确说明），导致 ``with db:`` 事务中的脚本无法被外层
    ``__exit__`` 正确回滚。本函数让 SQLite 端也走「逐条 ``conn.execute``」路径，
    与 MySQL 端 ``executescript`` 语义一致：事务中不自动提交，异常可回滚；
    但 SQLite / MySQL 各自的方言规则（如行注释判定）严格按各自引擎实现，
    不混用。
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
            elif ch == "-" and nxt == "-":
                # SQLite 规则：-- 后立即进入行注释，不要求空白（与 MySQL 不同）
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
            if ch == "'":
                if nxt == "'":
                    buf.append(nxt)
                    i += 1
                else:
                    state = "normal"
        elif state == "double":
            buf.append(ch)
            if ch == '"':
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

    @property
    def conn(self) -> sqlite3.Connection:
        """获取底层连接（如果关闭则自动重连）。"""
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

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
        """执行一段脚本（多条 SQL，以 ; 分隔），不在事务中时自动提交。

        实现说明：不使用标准库 ``conn.executescript``（它会在执行前隐式 COMMIT
        当前事务，导致 ``with db:`` 内脚本被提前提交、外层异常无法回滚）。
        改走 ``_split_sql_statements`` 逐条 ``conn.execute``，切分器会跳过字符串
        字面量 / 双引号标识符 / 行注释 / 块注释里的分号，不会误切；事务中不
        自动 commit，与 MySQLClient.executescript 语义一致。
        """
        if not isinstance(sql, str):
            raise DatabaseValidationError("executescript 的 sql 必须是字符串")
        try:
            statements = _split_sql_statements(sql)
            for stmt in statements:
                self.conn.execute(stmt)
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

    def query(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = (), fetch_one: bool = False,
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

    # ------------ 方言 ------------
    def _placeholder(self, name: str) -> str:
        """把参数名渲染成 SQLite 占位符文本。例：f: "f0" -> ":f0" """
        return f":{name}"  #

    def _quote_ident(self, name: str) -> str:
        """SQLite 双引号标识符。"""
        return _quote_ident(name, '"')

    # OFFSET 单独出现时的 LIMIT 兜底（SQLite 原生支持 -1 表示"无上限"）
    _OFFSET_NO_LIMIT_SQL: str = "LIMIT -1"

    def _parse_columns(self, table: str, columns: list[Column]) -> tuple[list[str], list[str]]:
        """SQLite 方言列解析，直接复用 schema.parse_column_list。"""
        return parse_column_list(table, columns)

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

    def _create_table_raw(self, table: str, col_defs: list[str], if_not_exists: bool) -> None:
        """执行原始 CREATE TABLE 语句。"""
        if_not = "IF NOT EXISTS " if if_not_exists else ""
        sql = f"""CREATE TABLE {if_not}{self._quote_ident(table)} ({",".join(col_defs)})"""
        self.execute(sql)

    def _create_table_from_schema(self, schema: Schema, if_not_exists: bool) -> None:
        """从 Schema 对象建表 + 索引（解析逻辑统一在 schema.parse_schema）。"""
        # 列定义 + 列级索引 + 显式 indexes，一次性解析出来；
        # 索引语句均为 CREATE INDEX IF NOT EXISTS，天然幂等，无需预先查已存在索引。
        col_defs, index_stmts = parse_schema(schema)
        self._create_table_raw(schema.table_name, col_defs, if_not_exists=if_not_exists)
        for stmt in index_stmts:
            self.execute(stmt)

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


