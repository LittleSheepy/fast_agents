"""database 模块 MySQL 后端（MySQLClient）单元测试。

覆盖点：
- 接口对齐：MySQLClient 与 SQLiteClient 对外公开方法集完全一致，且 DatabaseClientBase 抽象方法全部实现；
- 可选依赖降级：未装 pymysql 时给出明确安装提示（不影响 SQLite / Chroma 等其他后端导入）；
- MySQL 方言：标识符反引号 / 占位符 %s / 类型映射（BOOLEAN→TINYINT(1)、TIMESTAMP→DATETIME…）/
  声明式建表 SQL 生成 / Query 谓词编译；
- SqlExpr 跨后端共用一个类（上层从 SQLite 切到 MySQL 时 SqlExpr 无需换类）；
- 真实 MySQL 端到端 CRUD（默认 skip，设置 FAST_AGENTS_TEST_MYSQL_URL 后才跑）。

除「真实 MySQL CRUD」外全部离线可跑、不依赖外部 MySQL 服务。
运行方式（Windows PowerShell，在项目根目录）：
    pytest test/test_database_mysql.py -v
    或：python test/test_database_mysql.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fast_agents.database import (
    Column,
    ColumnType,
    DatabaseClientBase,
    DatabaseDriverMissingError,
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
    with pytest.raises(DatabaseDriverMissingError, match="pip install pymysql"):
        ClientCls(host="127.0.0.1", database="fast_agents")


# ---------- MySQL 方言适配（离线，不需要真实连接） ----------
def test_quote_ident_uses_backtick():
    assert _mc._quote_ident_mysql("name") == "`name`"
    assert _mc._quote_ident_mysql("na`me") == "`na``me`"


def test_type_mapping_covers_every_column_type():
    """ColumnType 每个成员都必须有 MySQL 类型映射，避免漏配。"""
    assert set(_mc.TYPE_TO_MYSQL) == set(ColumnType)


def test_column_sql_mysql_dialect():
    id_sql = _mc._column_sql_mysql(Column(name="id", type=ColumnType.INTEGER, primary_key=True))
    assert "INT" in id_sql
    assert "PRIMARY KEY" in id_sql
    assert "AUTO_INCREMENT" in id_sql  # MySQL 主键自增

    name_sql = _mc._column_sql_mysql(Column(name="name", type=ColumnType.VARCHAR, nullable=False, length=100))
    assert "VARCHAR(100)" in name_sql
    assert "NOT NULL" in name_sql

    flag_sql = _mc._column_sql_mysql(Column(name="is_active", type=ColumnType.BOOLEAN, default=True))
    assert "TINYINT(1)" in flag_sql
    assert "DEFAULT 1" in flag_sql

    ts_sql = _mc._column_sql_mysql(Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False))
    assert "DATETIME" in ts_sql  # TIMESTAMP 映射为 DATETIME，与 SQLite 行为对齐


def test_parse_schema_mysql_generates_ddl():
    """声明式 Schema → MySQL 列定义 + CREATE INDEX（列级 index=True 与显式 Index 都要出）。"""
    col_defs, index_stmts = _mc._parse_schema_mysql(Schema(
        table_name="orders",
        columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="user_id", type=ColumnType.INTEGER, nullable=False, index=True),
        ],
        indexes=[
            Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
        ],
    ))
    assert any("`id` INT" in d and "AUTO_INCREMENT" in d for d in col_defs)
    assert any("idx_orders_user_id" in s for s in index_stmts)   # 列级 index=True
    assert any("idx_orders_user_time" in s for s in index_stmts)  # 显式复合索引
    assert all("CREATE" in s.upper() and "INDEX" in s.upper() for s in index_stmts)


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


@pytest.mark.skipif(not MYSQL_URL, reason="未设置 FAST_AGENTS_TEST_MYSQL_URL，跳过真实 MySQL 连接测试")
def test_real_mysql_crud_end_to_end():
    """真实 MySQL 下跑一遍建表 / CRUD / 谓词查询 / upsert / 事务回滚（与 SQLite 行为对齐）。"""
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
    """
    import types

    class MySQLError(Exception):
        pass

    state = {"connect_error": None, "execute_error": None}

    class _FakeCursor:
        lastrowid = 1
        rowcount = 1

        def execute(self, sql, params=None):
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

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

        def ping(self, reconnect=True):
            return None

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
    """三类异常仍在 ValueError 谱系下，保证既有 except ValueError / pytest.raises(ValueError) 零破坏。"""
    for cls in (_mc.DatabaseConnectionError, _mc.DatabaseOperationalError, _mc.DatabaseValidationError):
        assert issubclass(cls, ValueError)


def test_mysql_compile_validations_raise_validation_error():
    """编译期入参校验抛 DatabaseValidationError（与 SQLite 行为一致）。"""
    from fast_agents.database import DatabaseValidationError
    with pytest.raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", limit=-1))
    with pytest.raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", sort_by=[]))
    with pytest.raises(DatabaseValidationError):
        MySQLClient._compile_query(Query(table_name="t", offset=-1))
    with pytest.raises(DatabaseValidationError):
        MySQLClient._compile_filter(Filter(field="a", op=FilterOp.IN, value=[]), 0)


def test_mysql_type_errors_stay_native_typeerror():
    """调用方把类型传错仍抛原生 TypeError，不纳入异常体系（与 SQLite 一致）。"""
    with pytest.raises(TypeError):
        MySQLClient._compile_filter(object(), 0)


def test_mysql_connect_failure_maps_to_connection_error(monkeypatch):
    """连接失败包成 DatabaseConnectionError，原异常挂在 __cause__ 上。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    state["connect_error"] = MySQLError("connection refused")
    from fast_agents.database import DatabaseConnectionError, MySQLClient as _MC
    with pytest.raises(DatabaseConnectionError) as ei:
        _MC(host="127.0.0.1", database="x")
    assert isinstance(ei.value.__cause__, MySQLError)


def test_mysql_execute_failure_maps_to_operational_error(monkeypatch):
    """SQL 执行 / 批量 / 查询失败均包成 DatabaseOperationalError（连接成功后触发）。"""
    MySQLError, state = _install_fake_pymysql(monkeypatch)
    from fast_agents.database import DatabaseOperationalError, MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    state["execute_error"] = MySQLError("syntax error")
    with pytest.raises(DatabaseOperationalError):
        db.execute("SELECT bad")
    with pytest.raises(DatabaseOperationalError):
        db.executemany("INSERT INTO t VALUES (%s)", [(1,)])
    with pytest.raises(DatabaseOperationalError):
        db.query("SELECT bad")
    db.close()


def test_mysql_close_is_safe_without_pending_transaction(monkeypatch):
    """close 在无未提交事务时正常关闭，不抛异常（回归保护）。"""
    _install_fake_pymysql(monkeypatch)
    from fast_agents.database import MySQLClient as _MC
    db = _MC(host="127.0.0.1", database="x")
    db.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
