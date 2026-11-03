"""database 模块快速上手示例。

演示 SQLiteClient 建表、增删改查、事务、回滚等基础用法。
全部用内存数据库，跑完不留文件。
运行方式：在项目根目录执行 `python examples/database/basic_usage.py`
"""
import sqlite3

from fast_agents.database import SQLiteClient


def main() -> None:
    # 不用 with 外层，避免嵌套事务；本示例演示单次连接的全部能力
    db = SQLiteClient(":memory:")

    # 1. 建表
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            score REAL DEFAULT 0
        )
    """)
    print("[1] 建表成功")

    # 2. insert 便捷插入（单次语句自动提交）
    alice_id = db.insert("users", {"name": "Alice", "age": 30, "score": 95.5})
    bob_id = db.insert("users", {"name": "Bob", "age": 25, "score": 88.0})
    print(f"[2] 插入 Alice(id={alice_id})、Bob(id={bob_id})")

    # 3. select 查询
    all_users = db.select("users", order_by="id ASC")
    print(f"[3] 全部用户: {all_users}")

    # 4. select_one 查单条
    alice = db.select_one("users", where={"id": alice_id})
    print(f"[4] 查询 id={alice_id}: {alice}")

    # 5. update 更新
    rows = db.update("users", data={"score": 99.0}, where={"id": alice_id})
    print(f"[5] 更新 Alice 分数，影响行数: {rows}")

    # 6. 手写 SQL 自定义查询
    young_users = db.query(
        "SELECT name, score FROM users WHERE age < ? ORDER BY score DESC",
        (28,),
    )
    print(f"[6] 年龄小于28的用户: {young_users}")

    # 7. 事务演示：正常提交（with db 显式开启事务，多条一起原子提交）
    with db:
        db.insert("users", {"name": "Charlie", "age": 35, "score": 70})
        db.insert("users", {"name": "Diana", "age": 28, "score": 91})
    print("[7] 事务提交成功，插入 Charlie、Diana")

    # 8. 事务演示：异常回滚
    try:
        with db:
            db.insert("users", {"name": "Eve", "age": 40, "score": 60})
            # 故意触发错误（name NOT NULL 约束失败）
            db.insert("users", {"name": None, "age": 1})
    except sqlite3.Error:
        print("[8] 事务回滚，Eve 不会被写入")

    final = db.select("users", order_by="id ASC")
    print(f"[9] 最终用户数: {len(final)}")
    for u in final:
        print(f"    - id={u['id']} name={u['name']} age={u['age']} score={u['score']}")

    db.close()


if __name__ == "__main__":
    main()
