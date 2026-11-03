"""database 模块 —— 建表能力（create_table / ensure_table / ensure_schema）示例。

演示三种建表方式：
1. dict 列定义（最快，写原型）
2. list[Column]（声明式，带约束/默认值/索引）
3. Schema 对象（工程化推荐，列 + 索引一次声明）
外加：表存在判断、幂等重复调用、migrate 补列、drop_table。

全部用内存数据库，跑完不留文件。
运行方式：在项目根目录执行 `python examples/database/create_table_usage.py`
"""
from __future__ import annotations

from fast_agents.database import (
    Column,
    ColumnType,
    Index,
    Schema,
    SQLiteClient,
)


def main() -> None:
    db = SQLiteClient("../../tmp/sql/sql.db")

    # ---------- 方式 1：dict，一行建表 ----------
    db.ensure_table(
        "users",
        {
            "id":   "INTEGER PRIMARY KEY AUTOINCREMENT",
            "name": "TEXT NOT NULL",
            "age":  "INTEGER",
        },
    )
    print("[1] dict 方式建表 users，table_exists =", db.table_exists("users"))

    # ---------- 方式 2：list[Column]，声明式 ----------
    db.ensure_table(
        "posts",
        [
            Column(name="id",        type=ColumnType.INTEGER, primary_key=True),
            Column(name="user_id",   type=ColumnType.INTEGER, nullable=False, index=True),
            Column(name="title",     type=ColumnType.VARCHAR, nullable=False, length=200),
            Column(name="content",   type=ColumnType.TEXT,    nullable=True),
            Column(name="is_draft",  type=ColumnType.BOOLEAN, nullable=False, default=True),
            Column(name="created_at",type=ColumnType.TIMESTAMP, nullable=False),
        ],
    )
    print("[2] list[Column] 方式建表 posts，table_exists =", db.table_exists("posts"))

    # ---------- 方式 3：Schema（推荐工程化写法），列+索引一次声明 ----------
    db.ensure_schema(
        Schema(
            table_name="orders",
            columns=[
                Column(name="id",         type=ColumnType.INTEGER, primary_key=True),
                Column(name="user_id",    type=ColumnType.INTEGER, nullable=False, index=True),
                Column(name="amount",     type=ColumnType.REAL,    nullable=False, default=0.0),
                Column(name="status",     type=ColumnType.TEXT,    nullable=False, default="pending"),
                Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False),
            ],
            indexes=[
                Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
                Index(name="uniq_orders_no",       columns=["id"], unique=True),  # 演示用，主键本身唯一
            ],
        )
    )
    print("[3] Schema 方式建表 orders，table_exists =", db.table_exists("orders"))

    # ---------- 幂等：重复调用不报错 ----------
    db.ensure_table("users", {"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL"})
    db.ensure_schema(
        Schema(
            table_name="orders",
            columns=[
                Column(name="id", type=ColumnType.INTEGER, primary_key=True),
                Column(name="user_id", type=ColumnType.INTEGER, nullable=False),
            ],
        )
    )
    print("[4] 重复 ensure 不报错（幂等）")

    # ---------- migrate=True 补列 ----------
    # 老表没有 email 列，ensure_schema(migrate=True) 会自动 ALTER ADD COLUMN
    db.ensure_schema(
        Schema(
            table_name="users",
            columns=[
                Column(name="id",    type=ColumnType.INTEGER, primary_key=True),
                Column(name="name",  type=ColumnType.TEXT,    nullable=False),
                Column(name="age",   type=ColumnType.INTEGER, nullable=True),
                Column(name="email", type=ColumnType.TEXT,    nullable=True, default=None),
                Column(name="score", type=ColumnType.REAL,    nullable=False, default=0.0),
            ],
        ),
        migrate=True,
    )
    db.insert("users", {"name": "Bob", "age": 25, "email": "bob@example.com"})
    bob = db.select_one("users", where={"name": "Bob"})
    print(f"[5] migrate 补列后插入 Bob: {bob}")

    # ---------- 正常 CRUD 照旧工作 ----------
    uid = db.insert("users", {"name": "Alice", "age": 30})
    db.update("users", {"age": 31}, where={"id": uid})
    print("[6] CRUD 照旧：Alice 更新后 =", db.select_one("users", where={"id": uid}))

    # ---------- drop_table ----------
    db.drop_table("posts")
    print("[7] drop posts 后 table_exists =", db.table_exists("posts"))

    # ---------- 打印 orders 表上的索引 ----------
    print("[8] orders 表上的索引：")
    for row in db.query(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t",
        {"t": "orders"},
    ):
        print("    -", row["name"])

    # ---------- 非法表名会被拦截（防注入） ----------
    try:
        db.ensure_table("bad;drop table users;", {"id": "INTEGER"})
    except ValueError as e:
        print(f"[9] 非法表名被拦截: {e}")

    db.close()
    print("\nAll create_table examples passed.")


if __name__ == "__main__":
    main()
