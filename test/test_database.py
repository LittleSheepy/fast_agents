"""database 模块单元测试。

全部用内存数据库，不生成文件，不依赖外部服务。
覆盖点：连接 / 建表（dict / list[Column] / Schema） / 索引 / 幂等 /
       migrate 补列 / 表存在判断 / 删表 / CRUD / 事务 / 异常 / 标识符注入防护。

运行方式（Windows PowerShell，在项目根目录）:
    python test\test_database.py
    或：pytest test/test_database.py -v
"""
import sqlite3

import pytest

from fast_agents.database import (
    Column,
    ColumnType,
    Index,
    Schema,
    SQLiteClient,
)


@pytest.fixture
def db():
    """每个测试用例都用全新的内存数据库。"""
    client = SQLiteClient(":memory:")
    client.execute("""
        CREATE TABLE test_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            score REAL DEFAULT 0
        )
    """)
    yield client
    client.close()


# ==================== 基础连接 ====================
def test_connect_memory(db):
    """连接内存数据库成功。"""
    assert db.conn is not None


def test_connect_file(tmp_path):
    """连接文件数据库成功，WAL/外键默认启用。"""
    path = tmp_path / "test.db"
    client = SQLiteClient(str(path))
    row = client.query("PRAGMA journal_mode", fetch_one=True)
    client.close()
    assert row is not None


# ==================== 基础执行 ====================
def test_execute_create_table(db):
    """建表不报错。"""
    db.execute("CREATE TABLE test_logs (id INTEGER PRIMARY KEY, msg TEXT)")
    tables = db.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='test_logs'",
        fetch_one=True,
    )
    assert tables is not None


def test_insert_and_lastrowid(db):
    """insert 返回自增 id。"""
    uid = db.insert("test_users", {"name": "Alice", "age": 30})
    assert uid == 1
    uid2 = db.insert("test_users", {"name": "Bob", "age": 25})
    assert uid2 == 2


def test_insert_empty_raises(db):
    """空数据插入抛 ValueError。"""
    with pytest.raises(ValueError):
        db.insert("test_users", {})


# ==================== 查询 ====================
def test_select_all(db):
    """select 返回字典列表。"""
    db.insert("test_users", {"name": "A", "age": 1})
    db.insert("test_users", {"name": "B", "age": 2})
    rows = db.select("test_users", order_by="id ASC")
    assert len(rows) == 2
    assert rows[0]["name"] == "A"
    assert rows[1]["age"] == 2
    assert "score" in rows[0]


def test_select_where(db):
    """where 等值过滤。"""
    db.insert("test_users", {"name": "A", "age": 1})
    db.insert("test_users", {"name": "B", "age": 2})
    rows = db.select("test_users", where={"age": 2})
    assert len(rows) == 1
    assert rows[0]["name"] == "B"


def test_select_one(db):
    """select_one 返回单条或 None。"""
    db.insert("test_users", {"name": "A", "age": 1})
    row = db.select_one("test_users", where={"age": 1})
    assert row is not None and row["name"] == "A"
    assert db.select_one("test_users", where={"age": 999}) is None


def test_select_limit(db):
    for i in range(5):
        db.insert("test_users", {"name": f"u{i}", "age": i})
    rows = db.select("test_users", order_by="id ASC", limit=3)
    assert len(rows) == 3


def test_raw_query(db):
    db.insert("test_users", {"name": "A", "age": 10})
    db.insert("test_users", {"name": "B", "age": 20})
    rows = db.query("SELECT name FROM test_users WHERE age > ?", (15,))
    assert len(rows) == 1
    assert rows[0]["name"] == "B"


# ==================== 更新 ====================
def test_update(db):
    uid = db.insert("test_users", {"name": "A", "score": 50})
    affected = db.update("test_users", {"score": 100}, where={"id": uid})
    assert affected == 1
    row = db.select_one("test_users", where={"id": uid})
    assert row["score"] == 100


def test_update_without_where_raises(db):
    with pytest.raises(ValueError):
        db.update("test_users", {"score": 0}, where={})


# ==================== 删除 ====================
def test_delete(db):
    uid = db.insert("test_users", {"name": "A"})
    affected = db.delete("test_users", where={"id": uid})
    assert affected == 1
    assert db.select_one("test_users", where={"id": uid}) is None


def test_delete_without_where_raises(db):
    with pytest.raises(ValueError):
        db.delete("test_users", where={})


# ==================== 事务 ====================
def test_transaction_commit(db):
    with db:
        db.insert("test_users", {"name": "T1"})
        db.insert("test_users", {"name": "T2"})
    assert len(db.select("test_users")) == 2


def test_transaction_rollback_on_error(db):
    # execute 会把 sqlite3.Error 包成 ValueError 抛出（见 SQLiteClient.execute 实现），
    # 事务 __exit__ 里检测到异常会自动 rollback，第一条 insert 也不会写入。
    try:
        with db:
            db.insert("test_users", {"name": "committed"})
            db.insert("test_users", {"name": None})  # NOT NULL 约束失败
    except ValueError:
        pass
    rows = db.select("test_users")
    assert len(rows) == 0


# ==================== 异常 ====================
def test_invalid_sql_raises(db):
    with pytest.raises((sqlite3.Error, ValueError)):
        db.execute("THIS IS NOT SQL")


def test_close_and_reconnect(db):
    db.close()
    assert db._conn is None
    assert db.conn is not None
    db.close()


def test_executemany(db):
    db.executemany(
        "INSERT INTO test_users (name, age) VALUES (?, ?)",
        [("A", 1), ("B", 2), ("C", 3)],
    )
    assert len(db.select("test_users")) == 3


# ==================== DDL：create_table / ensure_table / ensure_schema ====================
def test_create_table_dict():
    """方式 1：dict 列定义建表。"""
    db = SQLiteClient(":memory:")
    db.create_table(
        "t1",
        {"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL", "age": "INTEGER"},
    )
    assert db.table_exists("t1")
    db.insert("t1", {"name": "X", "age": 1})
    assert db.select_one("t1", {"name": "X"}) is not None
    db.close()


def test_create_table_if_not_exists_idempotent():
    """create_table(if_not_exists=True) 重复调用不报错。"""
    db = SQLiteClient(":memory:")
    db.create_table("t", {"id": "INTEGER PRIMARY KEY"})
    db.create_table("t", {"id": "INTEGER PRIMARY KEY"}, if_not_exists=True)
    # 不带 if_not_exists 重复建表应报错
    with pytest.raises((sqlite3.Error, ValueError)):
        db.create_table("t", {"id": "INTEGER PRIMARY KEY"}, if_not_exists=False)
    db.close()


def test_ensure_table_dict_idempotent():
    """ensure_table 等价 if_not_exists=True，幂等。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    assert db.table_exists("t")
    db.close()


def test_create_table_list_columns():
    """方式 2：list[Column] 建表，支持 NOT NULL / DEFAULT / UNIQUE / index。"""
    db = SQLiteClient(":memory:")
    db.create_table(
        "users",
        [
            Column(name="id",    type=ColumnType.INTEGER, primary_key=True),
            Column(name="name",  type=ColumnType.TEXT,    nullable=False),
            Column(name="email", type=ColumnType.TEXT,    nullable=False, unique=True),
            Column(name="age",   type=ColumnType.INTEGER, nullable=True,  default=0),
            Column(name="score", type=ColumnType.REAL,    nullable=False, default=0.0),
            Column(name="active",type=ColumnType.BOOLEAN, nullable=False, default=True),
            Column(name="dept_id",type=ColumnType.INTEGER,nullable=False, index=True),
        ],
    )
    # 默认值生效：只插必填 name/email/dept_id，age=0,score=0.0,active=1
    db.insert("users", {"name": "A", "email": "a@x.com", "dept_id": 1})
    row = db.select_one("users", {"email": "a@x.com"})
    assert row["age"] == 0
    assert row["score"] == 0.0
    assert row["active"] == 1  # Boolean 存为 1
    # 列索引被创建
    idx = db.query("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='users'")
    idx_names = {r["name"] for r in idx}
    assert "idx_users_dept_id" in idx_names
    # UNIQUE 约束生效：重复 email 应报错
    with pytest.raises((sqlite3.Error, ValueError)):
        db.insert("users", {"name": "B", "email": "a@x.com", "dept_id": 1})
    db.close()


def test_create_table_from_schema_with_indexes():
    """方式 3：Schema 对象建表 + 显式复合索引。"""
    db = SQLiteClient(":memory:")
    db.ensure_schema(
        Schema(
            table_name="orders",
            columns=[
                Column(name="id",         type=ColumnType.INTEGER, primary_key=True),
                Column(name="user_id",    type=ColumnType.INTEGER, nullable=False, index=True),
                Column(name="amount",     type=ColumnType.REAL,    nullable=False, default=0.0),
                Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False),
            ],
            indexes=[
                Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
            ],
        )
    )
    assert db.table_exists("orders")
    idx = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='orders'"
    )}
    assert "idx_orders_user_id" in idx
    assert "idx_orders_user_time" in idx
    db.close()


def test_ensure_schema_migrate_adds_missing_columns():
    """ensure_schema(migrate=True) 能为已存在表补上新增列。"""
    db = SQLiteClient(":memory:")
    # 先建一张老表：只有 id / name
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    db.insert("t", {"name": "old"})
    # 用更完整的 Schema 迁移：补 age / email
    db.ensure_schema(
        Schema(
            table_name="t",
            columns=[
                Column(name="id",    type=ColumnType.INTEGER, primary_key=True),
                Column(name="name",  type=ColumnType.TEXT,    nullable=False),
                Column(name="age",   type=ColumnType.INTEGER, nullable=True),
                Column(name="email", type=ColumnType.TEXT,    nullable=True, default=None),
            ],
        ),
        migrate=True,
    )
    # 新列存在，老数据 email/age 为默认 NULL
    old = db.select_one("t", {"name": "old"})
    assert "age" in old and old["age"] is None
    assert "email" in old and old["email"] is None
    # 可以给新列插值
    db.insert("t", {"name": "new", "age": 20, "email": "new@x.com"})
    new = db.select_one("t", {"name": "new"})
    assert new["age"] == 20 and new["email"] == "new@x.com"
    # 再调一次 migrate 幂等（列已存在不会重复加）
    db.ensure_schema(
        Schema(table_name="t", columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="name", type=ColumnType.TEXT, nullable=False),
            Column(name="age", type=ColumnType.INTEGER, nullable=True),
            Column(name="email", type=ColumnType.TEXT, nullable=True, default=None),
        ]),
        migrate=True,
    )
    db.close()


def test_ensure_schema_no_migrate_does_not_alter_existing_table():
    """migrate=False（默认）只建表+建索引，不改动已有表结构。"""
    db = SQLiteClient(":memory:")
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    db.ensure_schema(
        Schema(table_name="t", columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="name", type=ColumnType.TEXT, nullable=False),
            Column(name="age", type=ColumnType.INTEGER, nullable=True),
        ]),
        migrate=False,
    )
    cols = {r["name"] for r in db.query("PRAGMA table_info(t)")}
    # age 不应被补进来
    assert "age" not in cols
    db.close()


def test_table_exists_and_drop_table():
    """table_exists / drop_table 语义正确。"""
    db = SQLiteClient(":memory:")
    assert not db.table_exists("ghost")
    db.ensure_table("ghost", {"id": "INTEGER PRIMARY KEY"})
    assert db.table_exists("ghost")
    db.drop_table("ghost")
    assert not db.table_exists("ghost")
    # drop_table(if_exists=True) 对不存在的表不报错
    db.drop_table("ghost", if_exists=True)
    with pytest.raises((sqlite3.Error, ValueError)):
        db.drop_table("ghost", if_exists=False)
    db.close()


def test_varchar_and_timestamp_and_boolean_types():
    """VARCHAR/BOOLEAN/TIMESTAMP 能正确建表并插入查询。"""
    db = SQLiteClient(":memory:")
    db.ensure_table(
        "logs",
        [
            Column(name="id",       type=ColumnType.INTEGER, primary_key=True),
            Column(name="msg",      type=ColumnType.VARCHAR, nullable=False, length=100),
            Column(name="ts",       type=ColumnType.TIMESTAMP, nullable=False),
            Column(name="ok",       type=ColumnType.BOOLEAN, nullable=False, default=True),
        ],
    )
    db.insert("logs", {"msg": "hi", "ts": "2026-11-02T23:00:00", "ok": False})
    row = db.select_one("logs", {"msg": "hi"})
    assert row is not None
    assert row["ts"] == "2026-11-02T23:00:00"
    assert row["ok"] == 0
    db.close()


def test_unique_index():
    """unique=True 的索引能拦截重复键。"""
    db = SQLiteClient(":memory:")
    db.ensure_schema(
        Schema(
            table_name="accounts",
            columns=[
                Column(name="id",   type=ColumnType.INTEGER, primary_key=True),
                Column(name="name", type=ColumnType.TEXT, nullable=False),
            ],
            indexes=[
                Index(name="uniq_accounts_name", columns=["name"], unique=True),
            ],
        )
    )
    db.insert("accounts", {"name": "alice"})
    with pytest.raises((sqlite3.Error, ValueError)):
        db.insert("accounts", {"name": "alice"})
    db.close()


# ==================== 标识符注入防护 ====================
@pytest.mark.parametrize("bad_name", [
    "users;drop table x;",
    "users--",
    "ta b",
    "1bad",
    "",
    "x'y",
])
def test_illegal_ident_rejected(bad_name):
    """非法表名/列名被拦截，防止 SQL 注入。"""
    db = SQLiteClient(":memory:")
    with pytest.raises(ValueError):
        db.ensure_table(bad_name, {"id": "INTEGER PRIMARY KEY"})
    db.close()


def test_columns_type_validation():
    """columns 传错类型应抛 TypeError。"""
    db = SQLiteClient(":memory:")
    with pytest.raises(TypeError):
        db.create_table("t", "not a dict or list")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        db.create_table("t", ["not a column"])  # type: ignore[list-item]
    # Schema 表名不一致应抛 ValueError
    with pytest.raises(ValueError):
        db.create_table(
            "other",
            Schema(table_name="orders", columns=[
                Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            ]),
        )
    db.close()


def test_create_table_empty_columns_dict_rejected():
    """空 dict columns 抛 ValueError。"""
    db = SQLiteClient(":memory:")
    with pytest.raises(ValueError):
        db.create_table("t", {})
    db.close()


# ==================== 向后兼容：原有手写 SQL 建表仍可用 ====================
def test_raw_create_table_still_works(db):
    """原 execute(\"CREATE TABLE ...\") 写法完全可用，不破坏既有代码。"""
    db.execute("CREATE TABLE raw_t (id INTEGER PRIMARY KEY, v TEXT)")
    db.insert("raw_t", {"v": "hello"})
    assert db.select_one("raw_t", {"v": "hello"}) is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
