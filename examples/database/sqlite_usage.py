"""database 模块快速上手示例。

演示 SQLiteClient 建表、增删改查、批量插入、统计、事务、回滚等基础用法。
全部用内存数据库，跑完不留文件。
运行方式：在项目根目录执行 `python examples/database/sqlite_usage.py`
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

    # 3. batch_insert 批量插入
    batch_users = [
        {"name": "Charlie", "age": 35, "score": 70},
        {"name": "Diana",   "age": 28, "score": 91},
        {"name": "Eve",     "age": 22, "score": 82},
    ]
    inserted = db.batch_insert("users", batch_users)
    print(f"[3] 批量插入 {inserted} 行（Charlie/Diana/Eve）")

    # 4. select 查询
    all_users = db.select("users", order_by="id ASC")
    print(f"[4] 全部用户: {len(all_users)} 人")

    # 5. select_one 查单条
    alice = db.select_one("users", where={"id": alice_id})
    print(f"[5] 查询 id={alice_id}: {alice}")

    # 6. count 统计
    total = db.count("users")
    age22 = db.count("users", {"age": 22})
    print(f"[6] 总人数: {total}，年龄=22 的人数: {age22}")

    # 7. update 更新
    rows = db.update("users", data={"score": 99.0}, where={"id": alice_id})
    print(f"[7] 更新 Alice 分数，影响行数: {rows}")

    # 8. 手写 SQL 自定义查询
    young_users = db.query(
        "SELECT name, score FROM users WHERE age < ? ORDER BY score DESC",
        (28,),
    )
    print(f"[8] 年龄小于28的用户: {len(young_users)} 人")

    # 9. 事务演示：异常回滚
    eve_age_before = db.select_one("users", {"name": "Eve"})["age"]
    try:
        with db:
            db.update("users", {"age": 18}, {"name": "Eve"})
            # 故意触发错误（name NOT NULL 约束失败）
            db.insert("users", {"name": None, "age": 1})
    except ValueError:
        print("[9] 事务回滚，Eve 年龄不会被修改")
    eve_age_after = db.select_one("users", {"name": "Eve"})["age"]
    assert eve_age_before == eve_age_after, "事务回滚失败"

    final = db.select("users", order_by="id ASC")
    print(f"[10] 最终用户数: {len(final)}")
    for u in final:
        print(f"    - id={u['id']:>2} name={u['name']:<7} age={u['age']:>2} score={u['score']:.1f}")

    db.close()


if __name__ == "__main__":
    main()
