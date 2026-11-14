"""database 模块 MySQL 后端（MySQLClient）单元测试。

覆盖点：
- 接口对齐：MySQLClient 与 SQLiteClient 对外公开方法集完全一致，且 DatabaseClientBase 抽象方法全部实现；
- 可选依赖降级：未装 pymysql 时给出明确安装提示（不影响 SQLite / Chroma 等其他后端导入）；
- MySQL 方言：标识符反引号 / 占位符 %s / 类型映射（BOOLEAN→TINYINT(1)、TIMESTAMP→DATETIME…）/
  声明式建表 SQL 生成 / Query 谓词编译；
- executescript 的多语句切分：只认「语句正文」里的分号，字符串 / 反引号 / 注释内的分号不误切；
- SqlExpr 跨后端共用一个类（上层从 SQLite 切到 MySQL 时 SqlExpr 无需换类）；
- 真实 MySQL 端到端 CRUD（默认 skip，设置 FAST_AGENTS_TEST_MYSQL_URL 后才跑）。

除「真实 MySQL CRUD」外全部离线可跑、不依赖外部 MySQL 服务。
运行方式（Windows PowerShell，在项目根目录）：
    python tests/database/test_database_mysql.py
    或：python test/test_database_mysql.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._runner import raises, monkeypatch, Skip

from fast_agents.database import (
    Column,
    ColumnType,
    DatabaseClientBase,
    DatabaseDriverMissingError,
    DatabaseIdentifierError,
    DatabaseValidationError,
    Filter,
    FilterOp,
    Index,
    MySQLClient,
    Query,
    Schema,
    SortOrder,
    SQLiteClient,
    SqlExpr,
)
from fast_agents.database import mysql_client as _mc
from fast_agents.database import sqlite_client as _sc


def _public_methods(cls) -> set:
    return {name for name in dir(cls) if not name.startswith("_") and callable(getattr(cls, name))}


# ---------- 接口对齐 ----------
def test_base_class_abstract_methods_all_implemented():
    """DatabaseClientBase 的全部公开方法，MySQLClient 都已实现。"""
    missing = _public_methods(DatabaseClientBase) - _public_methods(MySQLClient)
    assert not missing, f"MySQLClient 未实现基类方法: {sorted(missing)}"


def test_public_api_matches_sqlite_exactly():
    """MySQLClient 与 SQLiteClient 对外公开方法集完全一致（双向都无差异）。"""
    assert _public_methods(MySQLClient) == _public_methods(SQLiteClient)


def test_sqlexpr_is_shared_class():
    """SqlExpr 两个后端共用一个类，跨后端 isinstance 均可识别（上层零改动切换后端）。"""
    expr = SqlExpr("score + 1")
    assert _mc.SqlExpr is _sc.SqlExpr is SqlExpr
    assert isinstance(expr, _mc.SqlExpr)
    assert isinstance(expr, _sc.SqlExpr)


# ---------- 可选依赖降级：未装 pymysql 时的行为（R2：延迟 import + 缺失报错 + 不阻塞顶层导入） ----------

def _FreshMySQLClient(monkeypatch):
    """返回一个「在 pymysql 缺失」前提下重新导入后的 MySQLClient 类。

    构造时才触发延迟 import，所以只要在构造前把 sys.modules['pymysql'] 置 None，
    就可以复现"未安装 pymysql"场景，且不影响顶层 from fast_agents.database import SQLiteClient。
    """
    import importlib
    monkeypatch.setitem(sys.modules, "pymysql", None)
    mod = importlib.import_module("fast_agents.database.mysql_client")
    return mod.MySQLClient


def test_module_top_level_import_does_not_require_pymysql():
    """顶层 import fast_agents.database 与 fast_agents 不会因没装 pymysql 而失败。"""
    # 只要本测试文件本身能 import 成功，就证明顶层不依赖 pymysql；
    # 这里再用一个断言兜底：SqlExpr / SQLiteClient / MySQLClient 类本身都能拿到
    assert MySQLClient is not None
    assert SQLiteClient is not None
    assert SqlExpr is not None


def test_constructor_without_pymysql_raises_install_hint(monkeypatch):
    """未装 pymysql 时构造 MySQLClient 直接抛 DatabaseDriverMissingError，且给出安装命令。"""
    ClientCls = _FreshMySQLClient(monkeypatch)
    with raises(DatabaseDriverMissingError, match="pip install pymysql"):
        ClientCls(host="127.0.0.1", database="fast_agents")


# ---------- MySQL 方言适配（离线，不需要真实连接） ----------
def test_quote_ident_uses_backtick():
    assert _mc.quote_ident("name") == "`name`"
    assert _mc.quote_ident("na`me") == "`na``me`"


def test_type_mapping_covers_every_column_type():
    """ColumnType 每个成员都必须有 MySQL 类型映射，避免漏配。"""
    assert set(_mc.TYPE_TO_MYSQL) == set(ColumnType)


def test_column_sql_mysql_dialect():
    id_sql = _mc.column_sql(Column(name="id", type=ColumnType.INTEGER, primary_key=True), **_mc._MYSQL_COLUMN_KW)
    assert "INT" in id_sql
    assert "PRIMARY KEY" in id_sql
    assert "AUTO_INCREMENT" in id_sql  # MySQL 主键自增

    name_sql = _mc.column_sql(Column(name="name", type=ColumnType.VARCHAR, nullable=False, length=100), **_mc._MYSQL_COLUMN_KW)
    assert "VARCHAR(100)" in name_sql
    assert "NOT NULL" in name_sql

    flag_sql = _mc.column_sql(Column(name="is_active", type=ColumnType.BOOLEAN, default=True), **_mc._MYSQL_COLUMN_KW)
    assert "TINYINT(1)" in flag_sql
    assert "DEFAULT 1" in flag_sql

    ts_sql = _mc.column_sql(Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False), **_mc._MYSQL_COLUMN_KW)
    assert "DATETIME" in ts_sql  # TIMESTAMP 映射为 DATETIME，与 SQLite 行为对齐


def test_parse_schema_mysql_generates_ddl():
    """声明式 Schema → MySQL 列定义 + CREATE INDEX（列级 index=True 与显式 Index 都要出）。"""
    col_defs, index_stmts = _mc.parse_schema(Schema(
        table_name="orders",
        columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="user_id", type=ColumnType.INTEGER, nullable=False, index=True),
        ],
        indexes=[
            Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
        ],
    ), **_mc._MYSQL_COLUMN_KW)
    assert any("`id` INT" in d and "AUTO_INCREMENT" in d for d in col_defs)
    assert any("idx_orders_user_id" in s for s in index_stmts)   # 列级 index=True
    assert any("idx_orders_user_time" in s for s in index_stmts)  # 显式复合索引
    assert all("CREATE" in s.upper() and "INDEX" in s.upper() for s in index_stmts)
    assert not any("IF NOT EXISTS" in s for s in index_stmts)  # MySQL 不支持 CREATE INDEX IF NOT EXISTS


def test_query_compile_uses_percent_placeholder():
    """MySQL 方言用 %s 位置占位符，而非 SQLite 的 :name 命名参数。"""
    q = Query(
        table_name="users",
        filters=[
            Filter(field="age", op=FilterOp.GTE, value=18),
            Filter(field="name", op=FilterOp.LIKE, value="%张%"),
        ],
        sort_by="id",
        sort_order=SortOrder.DESC,
        limit=10,
        offset=0,
    )
    sql, params = MySQLClient._compile_query(q)
    assert "SELECT * FROM `users`" in sql
    assert "`age` >= %s" in sql
    assert "`name` LIKE %s" in sql
    assert "ORDER BY `id` DESC" in sql
    assert "LIMIT 10" in sql
    assert "OFFSET 0" in sql
    assert params == [18, "%张%"]


# ---------- 真实 MySQL 端到端（默认 skip） ----------
MYSQL_URL = os.getenv("FAST_AGENTS_TEST_MYSQL_URL")



def test_real_mysql_crud_end_to_end():
    """真实 MySQL 下跑一遍建表 / CRUD / 谓词查询 / upsert / 事务回滚（与 SQLite 行为对齐）。"""
    if not MYSQL_URL:
        raise Skip("未设置 FAST_AGENTS_TEST_MYSQL_URL，跳过真实 MySQL 连接测试")
    from urllib.parse import urlparse

    url = urlparse(MYSQL_URL)
    db = MySQLClient(
        host=url.hostname or "127.0.0.1",
        port=url.port or 3306,
        user=url.username or "root",
        password=url.password or "",
        database=url.path.lstrip("/"),
    )

    t = "test_mysql_crud"
    db.drop_table(t, if_exists=True)
    db.ensure_schema(Schema(
        table_name=t,
        columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="name", type=ColumnType.VARCHAR, nullable=False, length=255, unique=True),
            Column(name="age", type=ColumnType.INTEGER, default=0),
            Column(name="score", type=ColumnType.REAL, default=0.0),
            Column(name="is_active", type=ColumnType.BOOLEAN, default=True),
        ],
        indexes=[Index(name="idx_name", columns=["name"], unique=True)],
    ))
    assert db.table_exists(t)

    # insert / select_one
    uid = db.insert(t, {"name": "张三", "age": 25, "score": 95.5})
    assert isinstance(uid, int)
    row = db.select_one(t, {"name": "张三"})
    assert row["age"] == 25
    assert row["score"] == 95.5
    assert row["is_active"] == 1  # TINYINT(1) 返回 0/1

    # batch_insert / count
    db.batch_insert(t, [
        {"name": "李四", "age": 30, "score": 88.0},
        {"name": "王五", "age": 22, "score": 92.0},
    ])
    assert db.count(t) == 3

    # 谓词查询
    rows = db.select_where(Query(
        table_name=t,
        filters=[Filter(field="age", op=FilterOp.GTE, value=25)],
        sort_by="age",
    ))
    assert [r["name"] for r in rows] == ["张三", "李四"]

    # update / update_where + SqlExpr
    assert db.update(t, {"age": 26}, {"name": "张三"}) == 1
    assert db.update_where(
        t,
        {"score": SqlExpr("score + 1")},
        Query(table_name=t, filters=[Filter(field="name", op=FilterOp.EQ, value="张三")]),
    ) == 1
    assert db.select_one(t, {"name": "张三"})["score"] == 96.5

    # upsert
    db.upsert(t, {"name": "张三", "age": 27, "score": 100.0}, conflict_keys="name")
    row = db.select_one(t, {"name": "张三"})
    assert row["age"] == 27
    assert row["score"] == 100.0

    # delete_where
    assert db.delete_where(t, Query(
        table_name=t, filters=[Filter(field="age", op=FilterOp.GTE, value=30)]
    )) == 1

    db.drop_table(t, if_exists=True)
    db.close()




# ---------- 异常体系：与 SQLiteClient 完全对齐（离线，不需要真实连接） ----------
def _install_fake_pymysql(monkeypatch):
    """注入假的 pymysql 模块，离线验证异常映射（不依赖真实 MySQL）。

    返回 (MySQLError, state)：
    - state["connect_error"]：设置后 connect() 抛该异常（模拟连接失败）
    - state["execute_error"]：设置后 cursor.execute() 抛该异常（模拟 SQL 执行失败）
    - state["ping_error"]：设置后 conn.ping() 抛该异常（模拟网络抖动/断开）
    - state["ping_called"]：记录 ping 的调用次数，用于断言事务中不 ping
    - state["server_version"]：conn.get_server_info() 的返回值（默认 "8.0.36"），用于测 upsert 行别名分支
    - state["executed_sql"]：记录 cursor.execute() 收到的 SQL 文本，用于断言生成的方言 SQL
    """
    import types

    class MySQLError(Exception):
        pass

    state = {
        "connect_error": None,
        "execute_error": None,
        "ping_error": None,
        "ping_called": 0,
        "server_version": "8.0.36",
        "executed_sql": [],
    }

    class _FakeCursor:
        lastrowid = 1
        rowcount = 1

        def execute(self, sql, params=None):
            state["executed_sql"].append(sql)
            if state["execute_error"] is not None:
                raise state["execute_error"]

        def executemany(self, sql, params_list):
            if state["execute_error"] is not None:
                raise state["execute_error"]

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def begin(self):
            pass

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

        def ping(self, reconnect=True):
            state["ping_called"] += 1
            if state["ping_error"] is not None:
                raise state["ping_error"]
            return None

        def get_server_info(self):
            return state["server_version"]

    fake = types.ModuleType("pymysql")
    fake.MySQLError = MySQLError
    cursors = types.ModuleType("pymysql.cursors")
    cursors.DictCursor = object
    fake.cursors = cursors

    def _connect(**kwargs):
        if state["connect_error"] is not None:
            raise state["connect_error"]
        return _FakeConn()

    fake.connect = _connect
    monkeypatch.setitem(sys.modules, "pymysql", fake)
    monkeypatch.setitem(sys.modules, "pymysql.cursors", cursors)
    return MySQLError, state


def test_mysql_uses_same_error_classes_as_sqlite():
    """MySQL 与 SQLite 共用同一组异常类（errors.py），不是各自定义一套。"""
    from fast_agents.database import errors as _errors
    assert _mc.DatabaseConnectionError is _errors.DatabaseConnectionError
    assert _mc.DatabaseOperationalError is _errors.DatabaseOperationalError
    assert _mc.DatabaseValidationError is _errors.DatabaseValidationError
    assert _sc.DatabaseValidationError is _errors.DatabaseValidationError


def test_mysql_error_classes_stay_in_valueerror_lineage():
    """三类异常仍在 ValueError 谱系下，保证既有 except ValueError / raises(ValueError) 零破坏。"""
    for cls in (_mc.DatabaseConnectionError, _mc.DatabaseOperationalError, _mc.DatabaseValidationError):
        assert issubclass(cls, ValueError)


def test_mysql_compile_validations_raise_validation_error():
    """编译期入参校验抛 DatabaseValidationError（与 SQLite 行为一致）。"""
    from fast_agents.database import DatabaseValidationError
    with raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", limit=-1))
    with raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", sort_by=[]))
    with raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", offset=-1))
    with raises(DatabaseValidationError):
        MySQLClient._compile_filter(Filter(field="a", op=FilterOp.IN, value=[]), 0)


def test_mysql_type_errors_stay_native_typeerror():
    """调用方把类型传错仍抛原生 TypeError，不纳入异常体系（与 SQLite 一致）。"""
    with raises(TypeError):
        MySQLClient._compile_filter(object(), 0)


def test_mysql_connect_failure_maps_to_connection_error(monkeypatch):
    """连接失败包成 DatabaseConnectionError，原异常挂在 __cause__ 上。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["connect_error"] = MySQLError("connection refused")
    from fast_agents.database import DatabaseConnectionError, MySQLClient as _MC
    with raises(DatabaseConnectionError) as ei:
        _MC(host="127.0.0.1", database="x")
    assert isinstance(ei.value.__cause__, MySQLError)


def test_mysql_execute_failure_maps_to_operational_error(monkeypatch):
    """SQL 执行 / 批量 / 查询失败均包成 DatabaseOperationalError（连接成功后触发）。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import DatabaseOperationalError, MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    state["execute_error"] = MySQLError("syntax error")
    with raises(DatabaseOperationalError):
        db.execute("SELECT bad")
    with raises(DatabaseOperationalError):
        db.executemany("INSERT INTO t VALUES (%s)", [(1,)])
    with raises(DatabaseOperationalError):
        db.query("SELECT bad")
    db.close()


def test_mysql_close_is_safe_without_pending_transaction(monkeypatch):
    """close 在无未提交事务时正常关闭，不抛异常（回归保护）。"""
    _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.close()


def test_mysql_conn_ping_skipped_in_transaction(monkeypatch):
    """事务态（with db: 块内）不 ping/不重连，避免隐式丢失事务边界。

    非事务态访问 conn 会触发 ping；进入事务后对 conn 的访问不再触发 ping，
    即使底层连接断开也由 execute 自然抛错，不静默重连；退出事务后恢复 ping。
    """
    _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    ping_count = {"n": 0}
    def _counting_ping(reconnect=True):
        ping_count["n"] += 1
    db._conn.ping = _counting_ping
    # 非事务态访问 conn 应触发 ping
    ping_count["n"] = 0
    _ = db.conn
    assert ping_count["n"] == 1, f"非事务态访问 conn 应 ping 一次，实际 {ping_count['n']}"
    # 进入事务
    with db:
        ping_count["n"] = 0
        # 事务内访问 conn 与 execute 都不应触发 ping
        _ = db.conn
        db.execute("SELECT 1")
        assert ping_count["n"] == 0, f"事务态访问 conn/execute 不应 ping，实际 {ping_count['n']}"
    # 退出事务后访问 conn 应恢复 ping
    ping_count["n"] = 0
    _ = db.conn
    assert ping_count["n"] == 1, f"退出事务后访问 conn 应恢复 ping，实际 {ping_count['n']}"
    db.close()


def test_mysql_conn_ping_error_outside_transaction_reconnects(monkeypatch):
    """非事务态 ping 失败会自动重连，长期空闲连接保持无缝复用体验。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    connect_calls = {"n": 0}
    original_connect = _MC._connect
    def _counting_connect(self):
        connect_calls["n"] += 1
        return original_connect(self)
    monkeypatch.setattr(_MC, "_connect", _counting_connect)
    db = _MC(host="127.0.0.1", database="x")
    # 首次连接已触发 _connect 一次
    first_connects = connect_calls["n"]
    # 非事务态，模拟 ping 失败（如连接被服务端断开）
    state["ping_error"] = MySQLError("lost connection")
    _ = db.conn  # 应走到 except 分支触发 _connect 重连
    assert connect_calls["n"] == first_connects + 1, "ping 失败后应触发一次重连"
    db.close()


# ---------- upsert 语法自适应：MySQL 8.0.19+ 行别名 / 5.7 & MariaDB 回退 VALUES() ----------
def test_server_supports_row_alias_version_matrix():
    """_server_supports_row_alias 版本判定矩阵：8.0.19 为分界，MariaDB / 无版本信息一律 False。"""
    class _C:
        def __init__(self, info):
            self._info = info

        def get_server_info(self):
            return self._info

    assert _mc._server_supports_row_alias(_C("8.0.19")) is True
    assert _mc._server_supports_row_alias(_C("8.0.36")) is True
    assert _mc._server_supports_row_alias(_C("8.4.0")) is True
    assert _mc._server_supports_row_alias(_C("8.0.36-0ubuntu0.22.04.1")) is True
    assert _mc._server_supports_row_alias(_C("8.0.18")) is False
    assert _mc._server_supports_row_alias(_C("5.7.44")) is False
    assert _mc._server_supports_row_alias(_C("5.6.51-log")) is False
    assert _mc._server_supports_row_alias(_C("10.6.12-MariaDB")) is False
    assert _mc._server_supports_row_alias(_C("")) is False
    assert _mc._server_supports_row_alias(_C("unknown")) is False
    assert _mc._server_supports_row_alias(object()) is False  # 连接对象没有 get_server_info


def test_mysql_upsert_row_alias_sql_shape(monkeypatch):
    """MySQL 8 生成的 upsert SQL 形如 ... VALUES (%s, %s) AS new ON DUPLICATE KEY UPDATE `v` = new.`v`。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["server_version"] = "8.0.36"
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.upsert("kv", {"k": "a", "v": "1"}, conflict_keys="k")
    sql = state["executed_sql"][-1]
    assert " AS new ON DUPLICATE KEY UPDATE" in sql, sql
    assert "`v` = new.`v`" in sql, sql
    assert "VALUES(`v`)" not in sql, sql
    db.close()


def test_mysql_upsert_falls_back_to_values_on_mysql57(monkeypatch):
    """MySQL 5.7：行别名语法不可用，回退到 VALUES(col)，保证 5.7 兼容。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["server_version"] = "5.7.44"
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.upsert("kv", {"k": "a", "v": "1"}, conflict_keys="k")
    sql = state["executed_sql"][-1]
    assert " AS new" not in sql, sql
    assert "VALUES(`v`)" in sql, sql
    db.close()


def test_mysql_upsert_falls_back_to_values_on_mariadb(monkeypatch):
    """MariaDB 各版本都不支持行别名：即使版本号是 10.x 也要回退 VALUES(col)。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["server_version"] = "10.6.12-MariaDB-1:10.6.12+maria~ubu2004"
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.upsert("kv", {"k": "a", "v": "1"}, conflict_keys="k")
    sql = state["executed_sql"][-1]
    assert " AS new" not in sql, sql
    assert "VALUES(`v`)" in sql, sql
    db.close()


def test_mysql_upsert_row_alias_probe_is_cached(monkeypatch):
    """行别名支持标记惰性探测一次后缓存：多次 upsert 不重复取服务端版本。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["server_version"] = "8.0.36"
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    calls = {"n": 0}
    real = db._conn.get_server_info

    def _counting():
        calls["n"] += 1
        return real()

    db._conn.get_server_info = _counting
    db.upsert("kv", {"k": "a", "v": "1"}, conflict_keys="k")
    db.upsert("kv", {"k": "a", "v": "2"}, conflict_keys="k")
    assert calls["n"] == 1, f"版本探测应只发生一次，实际 {calls['n']}"
    db.close()


def test_mysql_upsert_empty_update_columns_has_no_row_alias(monkeypatch):
    """update_columns=[]（冲突忽略）分支不引用新值，SQL 里不应出现 AS new。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["server_version"] = "8.0.36"
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.upsert("kv", {"k": "a", "v": "1"}, conflict_keys="k", update_columns=[])
    sql = state["executed_sql"][-1]
    assert " AS new" not in sql, sql
    assert "ON DUPLICATE KEY UPDATE `k` = `k`" in sql, sql
    db.close()


def test_split_sql_statements_ignores_semicolons_in_literals_and_comments():
    """executescript 的切分只认「语句正文」里的分号：字符串 / 反引号 / 注释内的分号不切。"""
    split = _mc._split_sql_statements
    assert split("SELECT 1; SELECT 2") == ["SELECT 1", "SELECT 2"]
    assert split("INSERT INTO t VALUES ('a;b'); SELECT 1") == ["INSERT INTO t VALUES ('a;b')", "SELECT 1"]
    assert split("SELECT 'it''s; fine'; SELECT 2") == ["SELECT 'it''s; fine'", "SELECT 2"]
    assert split("SELECT 'a\\';b'; SELECT 2") == ["SELECT 'a\\';b'", "SELECT 2"]
    assert split('SELECT "a;b"; SELECT 2') == ['SELECT "a;b"', "SELECT 2"]
    assert split("CREATE TABLE `a;b` (id INT); SELECT 1") == ["CREATE TABLE `a;b` (id INT)", "SELECT 1"]
    assert split("SELECT 1; -- a; b\nSELECT 2") == ["SELECT 1", "-- a; b\nSELECT 2"]
    assert split("SELECT 1; /* a; b */ SELECT 2") == ["SELECT 1", "/* a; b */ SELECT 2"]
    assert split("SELECT 1;; ; SELECT 2") == ["SELECT 1", "SELECT 2"]
    assert split("SELECT 1;") == ["SELECT 1"]
    assert split("SELECT 1--2; SELECT 2") == ["SELECT 1--2", "SELECT 2"]


def test_mysql_executescript_splits_on_statement_semicolons(monkeypatch):
    """executescript 用状态机切分：含分号的字符串字面量只算一条，最终逐条 execute。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.executescript("INSERT INTO t VALUES ('a;b'); UPDATE t SET v = 1")
    assert state["executed_sql"] == ["INSERT INTO t VALUES ('a;b')", "UPDATE t SET v = 1"], state["executed_sql"]
    db.close()


# ==================== 入口 ====================

def test_mysql_query_columns_projects_specified_columns():
    """MySQL 方言同样按 Query.columns 投影：反引号包裹、逐列白名单；默认仍为 SELECT *。"""
    sql, params = MySQLClient._compile_query(Query(table_name="t", columns=["v", "k"]))
    assert sql == "SELECT `v`, `k` FROM `t`", f"投影 SQL 错误: {sql}"
    assert params == []

    sql2, _ = MySQLClient._compile_query(Query(table_name="t"))
    assert sql2 == "SELECT * FROM `t`", f"默认应保持 SELECT *，实际: {sql2}"

    with raises(DatabaseIdentifierError):
        MySQLClient._compile_query(Query(table_name="t", columns=["v; DROP TABLE t"]))
    with raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", columns=[]))


def test_mysql_select_negative_limit_raises_validation_error(monkeypatch):
    """MySQL 侧 select(limit=负数) 也必须抛 DatabaseValidationError，不生成非法 SQL。

    修复前 MySQL 会拼出 `LIMIT -1`（服务端语法错误），且与 _compile_query 口径不一致；
    修复后两后端同口径：负数在拼接前就拦下，不产生任何待执行 SQL。
    """
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")

    with raises(DatabaseValidationError, match="limit 不能为负数"):
        db.select("t", limit=-1)
    # 被拦下时不应留下任何已执行的 SQL
    assert state["executed_sql"] == [], f"负数 limit 不应生成/执行 SQL: {state['executed_sql']}"

    # 正数仍然生成合法 SQL：LIMIT N
    db.select("t", limit=3)
    assert state["executed_sql"][-1].endswith("LIMIT 3"), state["executed_sql"][-1]
    db.close()


def test_mysql_insert_data_keys_are_ident_checked(monkeypatch):
    """回归：MySQL insert 的 data 键必须走标识符白名单（与 SQLite / update 统一）。

    修复前 MySQL insert 只做 _quote_ident_mysql 转义、不过白名单，非法列名靠服务端兜底
    （反引号转义后不构成注入，但与「标识符一律走 check_ident」的约定不符）；修复后与
    SQLite 端同口径：非法列名在拼接前抛 DatabaseIdentifierError、不产生任何待执行 SQL。
    """
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")

    with raises(DatabaseIdentifierError):
        db.insert("t_ik", {"id": 1, "v; DROP": 2})
    with raises(DatabaseIdentifierError):
        db.insert("t_ik", {"1bad": 1})
    # 校验先于执行：非法列名不应生成 / 执行任何 SQL
    assert state["executed_sql"] == [], state["executed_sql"]

    # 合法列名照常生成 MySQL 方言 SQL
    db.insert("t_ik", {"id": 1, "v": 10})
    assert state["executed_sql"][-1] == "INSERT INTO `t_ik` (`id`, `v`) VALUES (%s, %s)", state["executed_sql"][-1]
    db.close()


def test_mysql_create_table_empty_list_raises_validation_error(monkeypatch):
    """回归：MySQL create_table 的 list[Column] 空列表也必须走校验层，与 SQLite 对称。

    修复前 MySQL 的 list 分支同样无校验，[] 被拼成 `CREATE TABLE t ()` 交给服务端；
    修复后与 dict 分支对称：空列表在拼接前抛 DatabaseValidationError、不产生任何待执行 SQL。
    """
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC, Column, ColumnType
    db = _MC(host="127.0.0.1", database="x")

    with raises(DatabaseValidationError, match="columns 不能为空"):
        db.create_table("t_empty_list", [])
    # 校验先于执行：空列表不应生成 / 执行任何 SQL
    assert state["executed_sql"] == [], state["executed_sql"]

    # 非空 list 照常生成 MySQL 方言建表 SQL
    db.create_table(
        "t_ok",
        [Column(name="id", type=ColumnType.INTEGER, primary_key=True),
         Column(name="name", type=ColumnType.TEXT)],
    )
    assert any(s.startswith("CREATE TABLE") for s in state["executed_sql"]), state["executed_sql"]
    db.close()


def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    test_split_sql_statements_ignores_semicolons_in_literals_and_comments()
    with monkeypatch() as _mp:
        test_mysql_executescript_splits_on_statement_semicolons(_mp)
    test_mysql_query_columns_projects_specified_columns()
    with monkeypatch() as _mp:
        test_mysql_select_negative_limit_raises_validation_error(_mp)
    with monkeypatch() as _mp:
        test_mysql_insert_data_keys_are_ident_checked(_mp)
    with monkeypatch() as _mp:
        test_mysql_create_table_empty_list_raises_validation_error(_mp)


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
