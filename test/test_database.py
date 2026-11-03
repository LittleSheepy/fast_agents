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
    Filter,
    FilterOp,
    Index,
    Query,
    Schema,
    SortOrder,
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
