"""database 模块单元测试。

全部用内存数据库，不生成文件，不依赖外部服务。
覆盖点：连接 / 建表（dict / list[Column] / Schema） / 索引 / 幂等 /
       migrate 补列 / 表存在判断 / 删表 / CRUD / 事务 / 异常 / 标识符注入防护。

运行方式（Windows PowerShell，在项目根目录）:
    python test\test_database.py
    或：pytest test/test_database.py -v
"""
import sqlite3


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fast_agents.database import (
    Column,
    ColumnType,
    Filter,
    FilterOp,
    Index,
    Query,
    Schema,
    SortOrder,
    SqlExpr,
    SQLiteClient,
    # 异常体系（errors.py）
    DatabaseError,
    DatabaseConnectionError,
    DatabaseOperationalError,
    DatabaseValidationError,
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


def test_transaction_flag_reset_after_error_in_with(db):
    """问题 C：with 块内异常被内层 catch 住后，SQLite 在约束错误下不会自动回滚事务
    （conn.in_transaction 仍为 True），此时 _in_transaction 标志应保持 True，
    让 __exit__ 按正常提交/回滚路径走；退出 with 后标志必须复位到 False。"""
    with db:
        db.insert("test_users", {"name": "a"})
        try:
            db.insert("test_users", {"name": None})  # NOT NULL 失败
        except ValueError:
            pass
        assert db._in_transaction is True  # 约束错误不自动回滚，标志仍为 True
        db.insert("test_users", {"name": "b"})
    # 退出 with 后标志必须复位
    assert db._in_transaction is False
    # 内层 catch 了异常，__exit__ 正常提交，a 和 b 都应写入
    assert len(db.select("test_users")) == 2


def test_transaction_state_single_source_of_truth(db):
    """方案1：_in_transaction 是 _tx_depth 的只读派生，二者不可能失同步。

    旧实现用独立布尔 _in_transaction 记录事务状态，存在与 _tx_depth 分叉的风险；
    改为 property 后事务状态只有 _tx_depth 一个真实来源，无法人为制造失同步。
    """
    assert db._in_transaction is False
    assert db._tx_depth == 0
    # 只读 property 不允许外部赋值，从根上杜绝"人为置位"式失同步
    with pytest.raises(AttributeError):
        db._in_transaction = True  # type: ignore[misc]
    # 进入事务后两者保持一致
    db.__enter__()
    try:
        assert db._in_transaction is True
        assert db._tx_depth == 1
    finally:
        db.__exit__(None, None, None)
    assert db._in_transaction is False
    assert db._tx_depth == 0
    # 事务外出错不残留状态，后续 autocommit 正常工作
    with pytest.raises(ValueError):
        db.execute("THIS IS NOT VALID SQL")
    assert db._in_transaction is False  # 标志始终派生自 _tx_depth，稳定为 False
    db.insert("test_users", {"name": "ok"})
    assert len(db.select("test_users")) == 1


def test_transaction_flag_kept_false_after_error_outside_with(db):
    """问题 C：事务外单条 execute 出错后标志必须保持 False，后续写操作能正常 autocommit。"""
    assert db._in_transaction is False
    with pytest.raises(ValueError):
        db.execute("INSERT INTO test_users (name) VALUES (?)", (None,))
    assert db._in_transaction is False
    db.insert("test_users", {"name": "ok"})
    assert len(db.select("test_users")) == 1


def test_close_with_pending_transaction_rolls_back(tmp_path):
    """问题 C：with 块里没正常退出（例如异常中断未走到 __exit__）时，close 必须
    主动回滚未结束事务，不能遗留半事务或在 close 时被隐式回滚/提交造成意外。"""
    import os
    fp = os.path.join(str(tmp_path), "c.db")
    db = SQLiteClient(fp)
    db.ensure_table("t_close", {"id": "INTEGER PRIMARY KEY", "v": "TEXT"})
    # 手动进入事务、写入但故意不 __exit__
    db.__enter__()
    db.execute("INSERT INTO t_close (v) VALUES (?)", ("x",))
    assert db._in_transaction is True
    # close 应回滚而非提交
    db.close()
    assert db._conn is None
    assert db._in_transaction is False
    # 重连验证：未提交的数据不应存在
    db2 = SQLiteClient(fp)
    assert db2.count("t_close") == 0
    db2.close()


# ==================== 异常 ====================
def test_invalid_sql_raises(db):
    with pytest.raises((sqlite3.Error, ValueError)):
        db.execute("THIS IS NOT SQL")


# ==================== 异常体系（errors.py） ====================
def test_exception_hierarchy():
    """异常体系：DatabaseError 继承 ValueError，三个具体子类都挂在它下面。"""
    assert issubclass(DatabaseError, ValueError)
    for cls in (DatabaseConnectionError, DatabaseOperationalError, DatabaseValidationError):
        assert issubclass(cls, DatabaseError)


def test_execute_error_maps_to_operational(db):
    """SQL 执行 / 查询失败 -> DatabaseOperationalError（可被上层精确分流）。"""
    with pytest.raises(DatabaseOperationalError):
        db.execute("THIS IS NOT SQL")
    with pytest.raises(DatabaseOperationalError):
        db.query("SELECT * FROM never_exist_table")


def test_validation_error_maps(db):
    """入参校验失败 -> DatabaseValidationError。"""
    with pytest.raises(DatabaseValidationError):
        db.insert("test_users", {})
    with pytest.raises(DatabaseValidationError):
        db.delete("test_users", {})
    with pytest.raises(DatabaseValidationError):
        db.update("test_users", {}, {"id": 1})


def test_illegal_ident_maps_to_validation():
    """非法标识符（schema.check_ident）也归到 DatabaseValidationError。"""
    db = SQLiteClient(":memory:")
    with pytest.raises(DatabaseValidationError):
        db.ensure_table("bad name", {"id": "INTEGER PRIMARY KEY"})
    db.close()


def test_connection_error_maps():
    """连不上的路径 -> DatabaseConnectionError（构造时即建连）。"""
    with pytest.raises(DatabaseConnectionError):
        SQLiteClient("/no/such/dir_xyz_zz/db.sqlite")


def test_exception_still_caught_by_valueerror(db):
    """向后兼容：新异常全部继承 ValueError，既有 except ValueError 通吃。"""
    with pytest.raises(ValueError):
        db.execute("THIS IS NOT SQL")
    with pytest.raises(ValueError):
        db.insert("test_users", {})
    with pytest.raises(DatabaseError):  # 基类可统一兜底
        db.execute("THIS IS NOT SQL")


def test_exceptions_exported_from_top_level():
    """顶层 fast_agents 包同步导出异常体系。"""
    import fast_agents
    for name in (
        "DatabaseError",
        "DatabaseConnectionError",
        "DatabaseOperationalError",
        "DatabaseValidationError",
    ):
        assert hasattr(fast_agents, name), name


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


# ==================== 谓词下推：select_where / select_one_where ====================
def _seed_users_for_query(db):
    """给 test_users 造一份有区分度的数据，覆盖各操作符。"""
    db.insert("test_users", {"name": "Alice",   "age": 30, "score": 88.5})
    db.insert("test_users", {"name": "Bob",     "age": 25, "score": 72.0})
    db.insert("test_users", {"name": "Charlie", "age": 35, "score": 95.0})
    db.insert("test_users", {"name": "David",   "age": 40, "score": 60.0})
    db.insert("test_users", {"name": "Eve",     "age": 28, "score": 88.5})


def test_select_where_eq(db):
    """FilterOp.EQ 等价于 dict 等值查询。"""
    _seed_users_for_query(db)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Alice")],
    ))
    assert len(rows) == 1 and rows[0]["name"] == "Alice"


def test_select_where_neq(db):
    """FilterOp.NEQ：排除等于指定值的行。"""
    _seed_users_for_query(db)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="age", op=FilterOp.NEQ, value=30)],
        sort_by="id", sort_order=SortOrder.ASC,
    ))
    names = [r["name"] for r in rows]
    assert "Alice" not in names and len(names) == 4


def test_select_where_gt_gte_lt_lte(db):
    """GT/GTE/LT/LTE 四种比较运算正确。"""
    _seed_users_for_query(db)
    # age > 30 → Charlie, David
    rows = db.select_where(Query(table_name="test_users",
                             filters=[Filter(field="age", op=FilterOp.GT, value=30)],
                             sort_by="age"))
    assert [r["name"] for r in rows] == ["Charlie", "David"]

    # age >= 30 → Alice, Charlie, David
    rows = db.select_where(Query(table_name="test_users",
                             filters=[Filter(field="age", op=FilterOp.GTE, value=30)],
                             sort_by="age"))
    assert [r["name"] for r in rows] == ["Alice", "Charlie", "David"]

    # age < 30 → Bob, Eve
    rows = db.select_where(Query(table_name="test_users",
                             filters=[Filter(field="age", op=FilterOp.LT, value=30)],
                             sort_by="age"))
    assert [r["name"] for r in rows] == ["Bob", "Eve"]

    # score <= 72 → Bob(72), David(60)
    rows = db.select_where(Query(table_name="test_users",
                             filters=[Filter(field="score", op=FilterOp.LTE, value=72.0)],
                             sort_by="score"))
    assert [r["name"] for r in rows] == ["David", "Bob"]


def test_select_where_like(db):
    """FilterOp.LIKE：支持 % / _ 通配符（调用方自己写通配符，不替用户补）。"""
    _seed_users_for_query(db)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.LIKE, value="A%")],
    ))
    assert len(rows) == 1 and rows[0]["name"] == "Alice"

    # 不带通配符 → 等价于精确匹配
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.LIKE, value="Bob")],
    ))
    assert len(rows) == 1 and rows[0]["name"] == "Bob"


def test_select_where_in(db):
    """FilterOp.IN：展开成 IN (?, ?, ?) 形式。"""
    _seed_users_for_query(db)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.IN, value=["Alice", "Eve", "Nobody"])],
        sort_by="id",
    ))
    assert [r["name"] for r in rows] == ["Alice", "Eve"]


def test_select_where_in_empty_list_rejected(db):
    """IN 的 value 必须是非空 list/tuple，空值/非列表必须报错。"""
    _seed_users_for_query(db)
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users",
                          filters=[Filter(field="name", op=FilterOp.IN, value=[])]))
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users",
                          filters=[Filter(field="name", op=FilterOp.IN, value="Alice")]))  # type: ignore[arg-type]


def test_select_where_is_null_is_not_null(db):
    """IS_NULL / NOT_NULL 不要求 value。"""
    _seed_users_for_query(db)
    # 新插入一条 age=NULL 的
    db.insert("test_users", {"name": "NullGuy", "score": 50})
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="age", op=FilterOp.IS_NULL)],
    ))
    assert len(rows) == 1 and rows[0]["name"] == "NullGuy"

    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="age", op=FilterOp.NOT_NULL)],
    ))
    assert all(r["age"] is not None for r in rows)
    assert len(rows) == 5  # 最初 5 人都有 age


def test_select_where_binary_op_without_value_rejected(db):
    """EQ/NEQ/... 等二元操作符漏传 value 必须报错。"""
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users",
                          filters=[Filter(field="age", op=FilterOp.EQ)]))


def test_select_where_and_combination(db):
    """多个 Filter 之间固定 AND。"""
    _seed_users_for_query(db)
    # age > 25 AND score >= 88 → Alice(30,88.5)、Charlie(35,95)、Eve(28,88.5)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[
            Filter(field="age",   op=FilterOp.GT,  value=25),
            Filter(field="score", op=FilterOp.GTE, value=88.5),
        ],
        sort_by="id",
    ))
    assert [r["name"] for r in rows] == ["Alice", "Charlie", "Eve"]


def test_select_where_sort_limit_offset(db):
    """sort_by/sort_order/limit/offset 能正确组合。"""
    _seed_users_for_query(db)
    # 按 age DESC，取前 2 → David(40), Charlie(35)
    rows = db.select_where(Query(
        table_name="test_users",
        sort_by="age", sort_order=SortOrder.DESC,
        limit=2,
    ))
    assert [r["name"] for r in rows] == ["David", "Charlie"]

    # 按 score ASC，offset=2，limit=2 → Alice/Eve(88.5 并列，按插入序), Charlie(95)
    # 为避免依赖插入序，只验 offset+limit 行数约束
    rows = db.select_where(Query(
        table_name="test_users",
        sort_by="score", sort_order=SortOrder.ASC,
        limit=2, offset=2,
    ))
    assert len(rows) == 2

    # 仅 offset 无 limit：自动补 LIMIT -1，SQLite 支持"无上限"
    rows = db.select_where(Query(
        table_name="test_users",
        sort_by="id",
        offset=3,
    ))
    assert len(rows) == 2


def test_select_where_limit_offset_negative_rejected(db):
    """limit/offset 为负数必须报错。"""
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users", limit=-1))
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users", offset=-1))


def test_select_where_invalid_field_rejected(db):
    """字段名非法（含特殊字符、数字开头）必须被白名单拦截。"""
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users",
                          filters=[Filter(field="age; DROP TABLE test_users;--",
                                          op=FilterOp.EQ, value=1)]))
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="test_users",
                          filters=[Filter(field="1bad", op=FilterOp.EQ, value=1)]))


def test_select_where_value_is_parameterized(db):
    """value 部分必须走参数化（传包含 SQL 片段的字符串不会被执行）。"""
    _seed_users_for_query(db)
    # 若把 value 拼进 SQL，下面这行就会把所有行查出来；参数化时只会找 name 精确等于字符串 'Alice' OR '1'='1' 的人
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Alice' OR '1'='1")],
    ))
    assert len(rows) == 0


def test_select_one_where(db):
    """select_one_where 单行版本：无结果返回 None，有结果返回首行。"""
    _seed_users_for_query(db)
    row = db.select_one_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Charlie")],
    ))
    assert row is not None and row["age"] == 35

    assert db.select_one_where(Query(
        table_name="test_users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Nobody")],
    )) is None

    # select_one_where 不应把 limit=1 污染到原 Query 对象
    q = Query(table_name="test_users", sort_by="id")
    db.select_one_where(q)
    assert q.limit is None and q.offset is None


def test_select_where_no_filters_returns_all(db):
    """不传 filters 等价于 SELECT *，能正常配合 sort/limit。"""
    _seed_users_for_query(db)
    rows = db.select_where(Query(
        table_name="test_users",
        sort_by="id", limit=3,
    ))
    assert len(rows) == 3


def test_select_where_type_check(db):
    """select_where 必须传 Query 实例，其他类型抛 TypeError。"""
    with pytest.raises(TypeError):
        db.select_where("not a query")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        db.select_one_where({"table_name": "test_users"})  # type: ignore[arg-type]


def test_select_where_same_field_multiple_filters(db):
    """同字段多个 Filter 自动 AND（例如 BETWEEN 写法），参数名不冲突。"""
    _seed_users_for_query(db)
    # 25 <= age <= 35 → Alice(30), Bob(25), Charlie(35), Eve(28)
    rows = db.select_where(Query(
        table_name="test_users",
        filters=[
            Filter(field="age", op=FilterOp.GTE, value=25),
            Filter(field="age", op=FilterOp.LTE, value=35),
        ],
        sort_by="age",
    ))
    assert [r["name"] for r in rows] == ["Bob", "Eve", "Alice", "Charlie"]


def test_select_where_still_works(db):
    """原 select(where=dict) 等值写法完全保留，不受 select_where 影响。"""
    _seed_users_for_query(db)
    rows = db.select("test_users", where={"age": 30})
    assert len(rows) == 1 and rows[0]["name"] == "Alice"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------- count / batch_insert 测试 ----------
def test_count_empty():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    assert db.count("t") == 0

def test_count_total():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    for i in range(5):
        db.insert("t", {"name": f"u{i}", "age": 20 + i})
    assert db.count("t") == 5

def test_count_where():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    db.insert("t", {"name": "A", "age": 17})
    db.insert("t", {"name": "B", "age": 21})
    db.insert("t", {"name": "C", "age": 21})
    assert db.count("t", {"age": 21}) == 2

def test_count_invalid_table():
    db = SQLiteClient(":memory:")
    try:
        db.count("t; DROP TABLE t--")
        assert False, "应该抛错"
    except ValueError:
        pass

def test_batch_insert_empty():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
    assert db.batch_insert("t", []) == 0
    assert db.count("t") == 0

def test_batch_insert_basic():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    rows = [{"name": f"u{i}", "age": 20+i} for i in range(10)]
    n = db.batch_insert("t", rows)
    assert n == 10
    assert db.count("t") == 10
    assert db.select_one("t", {"name": "u5"})["age"] == 25

def test_batch_insert_inconsistent_fields():
    db = SQLiteClient(":memory:")
    db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"})
    rows = [
        {"name": "A", "age": 20},
        {"name": "B"},  # 缺 age
    ]
    try:
        db.batch_insert("t", rows)
        assert False, "应该抛错"
    except ValueError as e:
        assert "age" in str(e)

def test_batch_insert_invalid_table():
    db = SQLiteClient(":memory:")
    try:
        db.batch_insert("t; DROP TABLE t--", [{"x": 1}])
        assert False, "应该抛错"
    except ValueError:
        pass


def test_batch_insert_persists_after_close():
    """单独调用 batch_insert 后 close，重开连接数据必须存在——修复静默丢失问题。

    根因：Python sqlite3 默认 isolation_level='' 会在 DML 前隐式开事务，
    旧实现直接 self.conn.executemany 不 commit，close 时事务被回滚导致数据丢失。
    修复后 batch_insert 走 self.executemany 封装，在事务外自动 commit。
    """
    import tempfile, os, shutil
    tmp = tempfile.mkdtemp(prefix="bi_persist_")
    path = os.path.join(tmp, "t.db")
    try:
        db = SQLiteClient(path)
        db.ensure_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
        n = db.batch_insert("t", [{"name": "a"}, {"name": "b"}, {"name": "c"}])
        assert n == 3
        db.close()
        # 重开连接验证数据真的落盘了
        db2 = SQLiteClient(path)
        assert db2.count("t") == 3
        db2.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==================== 条件写入：upsert / update_where / delete_where ====================
def _make_kv_db():
    """构造一个带主键的单表内存库，供条件写入用例复用。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("kv", {"k": "TEXT PRIMARY KEY", "v": "TEXT", "n": "INTEGER DEFAULT 0"})
    return db


def test_busy_timeout_matches_timeout():
    """busy_timeout 与连接 timeout 同口径：默认 5s，显式 10s 时同步。"""
    db = SQLiteClient(":memory:")
    assert int(db.query("PRAGMA busy_timeout", fetch_one=True)["timeout"]) == 5000
    db.close()

    db2 = SQLiteClient(":memory:", timeout=10.0)
    assert int(db2.query("PRAGMA busy_timeout", fetch_one=True)["timeout"]) == 10000
    db2.close()


def test_upsert_inserts_new_row():
    """upsert 目标行不存在时按插入处理。"""
    db = _make_kv_db()
    assert db.upsert("kv", {"k": "a", "v": "1", "n": 1}, conflict_keys="k") == 1
    assert db.select_one("kv", {"k": "a"})["v"] == "1"
    db.close()


def test_upsert_conflict_updates_row():
    """upsert 命中冲突时更新，且不新插入行。"""
    db = _make_kv_db()
    db.insert("kv", {"k": "a", "v": "1", "n": 1})
    assert db.upsert("kv", {"k": "a", "v": "2", "n": 9}, conflict_keys="k") == 1
    row = db.select_one("kv", {"k": "a"})
    assert row["v"] == "2" and row["n"] == 9
    assert db.count("kv") == 1
    db.close()


def test_upsert_partial_update_columns():
    """update_columns 限定后，未列出的列在冲突时保持原值。"""
    db = _make_kv_db()
    db.insert("kv", {"k": "a", "v": "1", "n": 1})
    db.upsert("kv", {"k": "a", "v": "new", "n": 100}, conflict_keys="k", update_columns=["v"])
    row = db.select_one("kv", {"k": "a"})
    assert row["v"] == "new"
    assert row["n"] == 1
    db.close()


def test_upsert_conflict_key_missing_in_data():
    """conflict_keys 必须出现在 data 中，否则报 ValueError。"""
    db = _make_kv_db()
    with pytest.raises(ValueError):
        db.upsert("kv", {"v": "x", "n": 1}, conflict_keys="k")
    db.close()


def test_upsert_invalid_ident_rejected():
    """非法表名被白名单拦截。"""
    db = _make_kv_db()
    with pytest.raises(ValueError):
        db.upsert("kv; DROP TABLE kv--", {"k": "a"}, conflict_keys="k")
    db.close()


def test_update_where_with_filter():
    """update_where 支持比较谓词（GTE）。"""
    db = _make_kv_db()
    db.batch_insert("kv", [{"k": f"k{i}", "v": "x", "n": i} for i in range(5)])
    n = db.update_where(
        "kv", {"v": "hit"}, Query(table_name="kv", filters=[Filter(field="n", op=FilterOp.GTE, value=3)])
    )
    assert n == 2
    assert db.count("kv", {"v": "hit"}) == 2
    db.close()


def test_update_where_in_filter():
    """update_where 支持 IN 谓词。"""
    db = _make_kv_db()
    db.batch_insert("kv", [{"k": f"k{i}", "v": "x", "n": i} for i in range(5)])
    n = db.update_where(
        "kv", {"v": "in"}, Query(table_name="kv", filters=[Filter(field="k", op=FilterOp.IN, value=["k0", "k2"])])
    )
    assert n == 2
    assert db.count("kv", {"v": "in"}) == 2
    db.close()


def test_update_where_expression():
    """SQL 表达式（自增）通过 SqlExpr 显式包裹后可用。"""
    db = _make_kv_db()
    db.insert("kv", {"k": "a", "v": "x", "n": 5})
    n = db.update_where(
        "kv", {"n": SqlExpr("n + 1")}, Query(table_name="kv", filters=[Filter(field="k", op=FilterOp.EQ, value="a")])
    )
    assert n == 1
    assert db.select_one("kv", {"k": "a"})["n"] == 6
    db.close()


def test_update_where_requires_filters():
    """空 filters 直接报错，防止误更新全表。"""
    db = _make_kv_db()
    with pytest.raises(ValueError):
        db.update_where("kv", {"v": "x"}, Query(table_name="kv"))
    db.close()


def test_update_where_value_parameterized():
    """值走参数化：注入串被原样存入而非当 SQL 执行。"""
    db = _make_kv_db()
    db.insert("kv", {"k": "a", "v": "1", "n": 1})
    payload = "x' OR '1'='1"
    db.update_where(
        "kv", {"v": payload}, Query(table_name="kv", filters=[Filter(field="k", op=FilterOp.EQ, value="a")])
    )
    assert db.select_one("kv", {"k": "a"})["v"] == payload
    db.close()


def test_delete_where_with_filter():
    """delete_where 支持比较谓词（LT）。"""
    db = _make_kv_db()
    db.batch_insert("kv", [{"k": f"k{i}", "v": "x", "n": i} for i in range(5)])
    n = db.delete_where("kv", Query(table_name="kv", filters=[Filter(field="n", op=FilterOp.LT, value=2)]))
    assert n == 2
    assert db.count("kv") == 3
    db.close()


def test_delete_where_in_filter():
    """delete_where 支持 IN 谓词。"""
    db = _make_kv_db()
    db.batch_insert("kv", [{"k": f"k{i}", "v": "x", "n": i} for i in range(5)])
    n = db.delete_where("kv", Query(table_name="kv", filters=[Filter(field="k", op=FilterOp.IN, value=["k4", "k3"])]))
    assert n == 2
    assert db.count("kv") == 3
    db.close()


def test_delete_where_requires_filters():
    """空 filters 直接报错，且不会误删数据。"""
    db = _make_kv_db()
    db.insert("kv", {"k": "a", "v": "1", "n": 1})
    with pytest.raises(ValueError):
        db.delete_where("kv", Query(table_name="kv"))
    assert db.count("kv") == 1
    db.close()


def test_multi_column_sort():
    """sort_by 支持多字段，方向对每列统一应用。"""
    db = _make_kv_db()
    db.batch_insert("kv", [
        {"k": "a", "v": "x", "n": 2},
        {"k": "b", "v": "x", "n": 1},
        {"k": "c", "v": "y", "n": 9},
    ])
    asc = db.select_where(Query(table_name="kv", sort_by=["v", "n"], sort_order=SortOrder.ASC))
    assert [r["k"] for r in asc] == ["b", "a", "c"]
    desc = db.select_where(Query(table_name="kv", sort_by=["v", "n"], sort_order=SortOrder.DESC))
    assert [r["k"] for r in desc] == ["c", "a", "b"]
    db.close()


def test_sort_by_single_str_still_works():
    """单字段 str 写法保持兼容。"""
    db = _make_kv_db()
    db.batch_insert("kv", [{"k": f"k{i}", "v": "x", "n": i} for i in range(3)])
    rows = db.select_where(Query(table_name="kv", sort_by="n", sort_order=SortOrder.DESC))
    assert [r["n"] for r in rows] == [2, 1, 0]
    db.close()


def test_sort_by_empty_list_rejected():
    """空 sort_by 列表直接报错。"""
    db = _make_kv_db()
    with pytest.raises(ValueError):
        db.select_where(Query(table_name="kv", sort_by=[]))
    db.close()


def test_sql_expr_empty_rejected():
    """SqlExpr 不接受空表达式。"""
    with pytest.raises(ValueError):
        SqlExpr("")
    with pytest.raises(ValueError):
        SqlExpr("   ")


# ===========================================================================
# 向量后端：VectorClientBase 抽象层 + 编译层
# ===========================================================================

def test_vector_client_base_abstract_methods():
    """VectorClientBase 应覆盖连接/集合/写入/删除/取/检索/计数 7 类能力。"""
    from fast_agents.database import VectorClientBase
    expected = {"connect", "close", "ping",
                "collection_exists", "create_collection", "get_or_create_collection", "drop_collection",
                "add", "delete", "get", "search", "count"}
    assert expected == VectorClientBase.__abstractmethods__


def test_vector_filter_leaf_and_compile():
    """叶子条件应编译成 Chroma where 格式。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    f = VectorFilter(field="kind", op=VectorFilterOp.EQ, value="fact")
    assert _compile_vector_filter(f) == {"kind": {"$eq": "fact"}}


def test_vector_filter_and_compile():
    """AND 组合条件应编译成 $and 嵌套。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    f = VectorFilter(op=VectorFilterOp.AND, filters=[
        VectorFilter(field="user_id", op=VectorFilterOp.EQ, value="u1"),
        VectorFilter(field="score", op=VectorFilterOp.GTE, value=0.7),
    ])
    assert _compile_vector_filter(f) == {
        "$and": [
            {"user_id": {"$eq": "u1"}},
            {"score": {"$gte": 0.7}},
        ]
    }


def test_vector_filter_passthrough_dict():
    """dict 原样透传，给熟悉 Chroma where 格式的高级用户留口子。"""
    from fast_agents.database.chroma_client import _compile_vector_filter
    assert _compile_vector_filter({"x": {"$gt": 1}}) == {"x": {"$gt": 1}}


def test_vector_filter_and_needs_two_subfilters():
    """AND/OR 组合至少 2 个子条件。"""
    from fast_agents.database import VectorFilter, VectorFilterOp
    from fast_agents.database.chroma_client import _compile_vector_filter
    with pytest.raises(ValueError):
        _compile_vector_filter(VectorFilter(op=VectorFilterOp.AND, filters=[
            VectorFilter(field="a", op=VectorFilterOp.EQ, value=1),
        ]))


def test_chroma_client_requires_path_or_host():
    """path 和 host 二选一，都不给/都给都报错。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    with pytest.raises(ValueError):
        ChromaClient()
    with pytest.raises(ValueError):
        ChromaClient(path="/tmp/x", host="127.0.0.1")


def test_chroma_client_invalid_collection_name():
    """非法集合名（太短 / 首尾非字母数字 / 含特殊字符 / 过长）直接报错。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, os
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    db = ChromaClient(path=tmp)
    for bad in ["ab", "_abc", "a" * 64, "abc!", "a b", "abc_"]:
        with pytest.raises(ValueError):
            db.create_collection(bad)
    db.close()


def test_chroma_client_lifecycle_local_persistent():
    """本地持久化模式：建集→写→按id取→近邻搜索→按条件删→幂等建集→drop。"""
    chromadb = pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        assert db.ping() is True
        # 幂等建集
        db.get_or_create_collection("mem_v1", dimension=4, metric="cosine")
        db.get_or_create_collection("mem_v1", dimension=4, metric="cosine")  # 重复调用幂等
        assert db.collection_exists("mem_v1")
        assert not db.collection_exists("mem_v2")
        # 写 3 条：2 条 kind=fact、1 条 kind=preference
        db.add(
            "mem_v1",
            ids=["m1", "m2", "m3"],
            embeddings=[[1.0, 0.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]],
            documents=["猫喜欢吃小鱼干", "猫爱睡觉", "我喜欢蓝颜色"],
            metadatas=[{"kind": "fact", "score": 0.9},
                       {"kind": "fact", "score": 0.8},
                       {"kind": "preference", "score": 1.0}],
        )
        assert db.count("mem_v1") == 3
        # 按 id 精确取
        rows = db.get("mem_v1", ids=["m1"])
        assert len(rows) == 1 and rows[0]["id"] == "m1" and rows[0]["document"] == "猫喜欢吃小鱼干"
        assert "embedding" not in rows[0]  # 默认不带 embedding
        rows_with_emb = db.get("mem_v1", ids=["m1"], include_embeddings=True)
        assert "embedding" in rows_with_emb[0] and len(rows_with_emb[0]["embedding"]) == 4
        # upsert 覆盖：m2 的 document/score 覆盖，m1/m3 保留
        db.add("mem_v2_dummy" if False else "mem_v1",
               ids=["m2"],
               documents=["猫爱打盹"],
               metadatas=[{"kind": "fact", "score": 0.85}],
               embeddings=[[0.9, 0.15, 0.0, 0.0]])
        assert db.count("mem_v1") == 3
        m2 = db.get("mem_v1", ids=["m2"])[0]
        assert m2["document"] == "猫爱打盹" and m2["metadata"]["score"] == 0.85
        # 近邻检索：query ≈ [1,0,0,0] 应该先返回 m1、再 m2，m3 最远
        hits = db.search("mem_v1", [1.0, 0.0, 0.0, 0.0], n_results=2)
        assert len(hits) == 2
        assert hits[0]["id"] == "m1"
        assert hits[1]["id"] == "m2"
        assert "distance" in hits[0]
        # where 前置过滤：只在 preference 类里搜，应只返回 m3
        hits_filtered = db.search("mem_v1", [1.0, 0.0, 0.0, 0.0], n_results=5,
                                  where=VectorFilter(field="kind", op=VectorFilterOp.EQ, value="preference"))
        assert [h["id"] for h in hits_filtered] == ["m3"]
        # delete 空条件拒绝
        with pytest.raises(ValueError):
            db.delete("mem_v1")
        # 按 metadata 条件删除：删 score >= 0.9 的 → m1(0.9) 会被保留？不，m1=0.9 >=0.9 删，m3=1.0 删，剩 m2
        removed = db.delete("mem_v1", where=VectorFilter(field="score", op=VectorFilterOp.GTE, value=0.9))
        assert removed == 2
        assert db.count("mem_v1") == 1
        assert db.get("mem_v1", ids=["m2"])[0]["document"] == "猫爱打盹"
        # 按 id 删除
        db.delete("mem_v1", ids=["m2"])
        assert db.count("mem_v1") == 0
        # drop 幂等
        db.drop_collection("mem_v1")
        db.drop_collection("mem_v1")  # 不存在静默
        assert not db.collection_exists("mem_v1")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_add_requires_consistent_lengths():
    """embeddings/documents/metadatas 长度必须与 ids 一致。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        with pytest.raises(ValueError):
            db.add("tt1", ids=["a"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            db.add("tt1", ids=["a"])  # 内容字段全空
        with pytest.raises(ValueError):
            db.add("tt1", ids=[])   # ids 空
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_search_rejects_bad_query_vector():
    """query_embedding 必须是非空向量。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        db.add("tt1", ids=["a"], embeddings=[[1.0, 0.0]])
        with pytest.raises(ValueError):
            db.search("tt1", [])
        with pytest.raises(ValueError):
            db.search("tt1", [1.0, 0.0], n_results=0)
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_count_with_where():
    """带 where 的 count 走 get include=[] 计数。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("tt1")
        db.add("tt1", ids=["a", "b", "c"],
               embeddings=[[1,0],[0,1],[1,1]],
               metadatas=[{"v":1},{"v":2},{"v":3}])
        assert db.count("tt1") == 3
        assert db.count("tt1", where=VectorFilter(field="v", op=VectorFilterOp.GTE, value=2)) == 2
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_client_invalid_metric_rejected():
    """metric 只能是 cosine / ip / l2。"""
    pytest.importorskip("chromadb")
    from fast_agents.database import ChromaClient
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        with pytest.raises(ValueError):
            db.create_collection("tt1", metric="euclidean")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ===========================================================================
# 向量后端：v1.1.1 优化项
# ===========================================================================

def test_vector_client_base_get_signature_has_include_switches():
    """抽象基类 get 暴露 include_documents / include_metadatas 可选开关（默认 True）。"""
    import inspect
    from fast_agents.database import VectorClientBase
    params = inspect.signature(VectorClientBase.get).parameters
    assert params["include_documents"].default is True
    assert params["include_metadatas"].default is True


def test_chroma_create_collection_persists_dimension_and_rejects_duplicate():
    """建集把 dimension 写进 collection metadata；重复建集抛「已存在」。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt1", dimension=4, metric="cosine")
        # 维度写进了 metadata，可跨进程读回
        assert db._client.get_collection("opt1").metadata.get("dimension") == 4
        with pytest.raises(ValueError, match="已存在"):
            db.create_collection("opt1")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_or_create_is_idempotent_and_keeps_dimension():
    """get_or_create 走原生幂等接口：二次调用不报错，维度仍记录。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.get_or_create_collection("opt2", dimension=3)
        db.get_or_create_collection("opt2", dimension=3)  # 幂等
        assert db._dimensions.get("opt2") == 3
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_duplicate_ids():
    """同一批 ids 重复应提前报错（Chroma 内部会抛 DuplicateIDError）。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt3")
        with pytest.raises(ValueError, match="重复"):
            db.add("opt3", ids=["a", "a"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_inconsistent_batch_dimension():
    """同一批 embeddings 维度不一致应报错。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt4")
        with pytest.raises(ValueError, match="维度不一致"):
            db.add("opt4", ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0, 2.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_rejects_dimension_mismatch_with_declared():
    """写入维度与建集声明的维度不一致应报错（Chroma 空集合不校验，靠客户端补位）。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt5", dimension=3)
        with pytest.raises(ValueError, match="声明的维度"):
            db.add("opt5", ids=["a"], embeddings=[[1.0, 0.0]])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_search_rejects_dimension_mismatch():
    """检索向量维度与声明维度不一致应报错。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt6", dimension=3)
        with pytest.raises(ValueError, match="维度"):
            db.search("opt6", [1.0, 0.0])
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_add_chunks_large_batch():
    """add_chunk_size 生效：分块写入仍全部落库、返回总数。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp, add_chunk_size=2)
        db.create_collection("opt7", dimension=2)
        ids = [f"k{i}" for i in range(5)]
        embs = [[float(i), float(i + 1)] for i in range(5)]
        assert db.add("opt7", ids=ids, embeddings=embs) == 5
        assert db.count("opt7") == 5
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_limit_validation_and_include_switches():
    """get 校验 limit 为正整数；include_documents / include_metadatas 关闭后不收对应字段。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt8", dimension=2)
        db.add("opt8", ids=["a"], embeddings=[[1.0, 0.0]], documents=["doc"], metadatas=[{"v": 1}])
        with pytest.raises(ValueError):
            db.get("opt8", limit=0)
        with pytest.raises(ValueError):
            db.get("opt8", limit=-3)
        rows = db.get("opt8", include_documents=False, include_metadatas=False)
        assert rows[0]["id"] == "a"
        assert "document" not in rows[0] and "metadata" not in rows[0]
        full = db.get("opt8")
        assert full[0]["document"] == "doc" and full[0]["metadata"]["v"] == 1
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_delete_ids_take_precedence_and_exact_count():
    """delete 返回精确条数；ids 与 where 同时传时以 ids 为准。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    from fast_agents.database import VectorFilter, VectorFilterOp
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        db.create_collection("opt9", dimension=2)
        db.add(
            "opt9",
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            metadatas=[{"v": 1}, {"v": 2}, {"v": 3}],
        )
        # ids 含不存在的 id：返回精确删除条数
        assert db.delete("opt9", ids=["a", "nope"]) == 1
        # 同时传 ids 与 where：语义为按 ids 删（不受 where 影响）
        removed = db.delete(
            "opt9",
            ids=["c"],
            where=VectorFilter(field="v", op=VectorFilterOp.EQ, value=999),
        )
        assert removed == 1
        assert db.count("opt9") == 1
        assert db.get("opt9")[0]["id"] == "b"
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chroma_get_or_create_tolerates_existing_collection_without_metadata():
    """对已存在但 metadata 为空的集合调用 get_or_create 不应因 Chroma 0.6.x 的合并缺陷而崩。"""
    pytest.importorskip("chromadb")
    import tempfile, shutil
    from fast_agents.database import ChromaClient
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        db = ChromaClient(path=tmp)
        # 用裸 SDK 建一个没有 metadata 的集合
        db.connect()
        db._client.create_collection("raw1")
        # 再走我们的 get_or_create（带 metadata + dimension），应能从容取回而不是 TypeError
        db.get_or_create_collection("raw1", dimension=2)
        assert db._dimensions.get("raw1") == 2
        db.add("raw1", ids=["a"], embeddings=[[1.0, 0.0]])
        assert db.count("raw1") == 1
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ----------------------------------------------------------------------
# 12. 事务嵌套支持（重入忽略 + 最外层统一提交/回滚，最简方案）
# ----------------------------------------------------------------------

def test_nested_with_does_not_crash_and_commits_to_file(tmp_path):
    """嵌套 with db: 不抛 cannot-start-transaction-within-transaction，数据落盘后重开仍能读到。"""
    import os
    db_path = os.path.join(str(tmp_path), "nt1.db")
    db = SQLiteClient(db_path)
    db.ensure_table("nt1", {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "v": "INTEGER"})
    with db:
        db.insert("nt1", {"v": 1})
        with db:
            db.insert("nt1", {"v": 2})
        db.insert("nt1", {"v": 3})
    db.close()
    db2 = SQLiteClient(db_path)
    assert db2.count("nt1") == 3
    db2.close()


def test_nested_with_inner_exception_rolls_back_entire_chain(tmp_path):
    """内层 with 抛异常时，最外层整体回滚，一条都不写。"""
    import os
    db_path = os.path.join(str(tmp_path), "nt2.db")
    db = SQLiteClient(db_path)
    db.ensure_table("nt2", {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "v": "INTEGER"})
    with pytest.raises(ValueError, match="boom"):
        with db:
            db.insert("nt2", {"v": 1})
            with db:
                db.insert("nt2", {"v": 2})
                raise ValueError("boom")
    assert db.count("nt2") == 0
    db.close()


def test_nested_with_outer_exception_rolls_back_all(tmp_path):
    """外层 with 抛异常，嵌套里的写入也要整体回滚。"""
    import os
    db_path = os.path.join(str(tmp_path), "nt3.db")
    db = SQLiteClient(db_path)
    db.ensure_table("nt3", {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "v": "INTEGER"})
    with pytest.raises(RuntimeError, match="outer"):
        with db:
            db.insert("nt3", {"v": 1})
            with db:
                db.insert("nt3", {"v": 2})
            raise RuntimeError("outer")
    assert db.count("nt3") == 0
    db.close()


def test_connection_isolation_level_is_none():
    """isolation_level=None 关闭 Python 隐式事务，事务边界 100% 由代码控制。"""
    db = SQLiteClient(":memory:")
    assert db.conn.isolation_level is None
    db.close()
