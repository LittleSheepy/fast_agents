"""database 模块条件写入示例。

演示 upsert（插入或更新）、update_where / delete_where（带谓词的批量改删）
以及多列排序。全部用内存数据库，跑完不留文件。
运行方式：在项目根目录执行 `python examples/database/conditional_write_usage.py`
"""
from fast_agents.database import (
    Filter,
    FilterOp,
    Query,
    SortOrder,
    SqlExpr,
    SQLiteClient,
)


def main() -> None:
    db = SQLiteClient(":memory:")

    # 1. 建两张表：一张做 upsert 演示，一张做条件改删演示
    db.ensure_table("kv", {
        "k": "TEXT PRIMARY KEY",   # 主键，作为 upsert 的冲突键
        "v": "TEXT",
        "n": "INTEGER DEFAULT 0",
    })
    db.ensure_table("memories", {
        "memory_id":    "TEXT PRIMARY KEY",
        "content":      "TEXT",
        "kind":         "TEXT",
        "access_count": "INTEGER DEFAULT 0",
    })
    print("[1] 建表完成：kv / memories")

    # 2. upsert：不存在则插入
    db.upsert("kv", {"k": "a", "v": "first", "n": 1}, conflict_keys="k")
    print(f"[2] upsert 新行 a -> {db.select_one('kv', {'k': 'a'})}")

    # 3. upsert：命中主键冲突则更新（不会多出一行）
    db.upsert("kv", {"k": "a", "v": "second", "n": 2}, conflict_keys="k")
    print(f"[3] upsert 冲突更新 a -> {db.select_one('kv', {'k': 'a'})}，总行数 {db.count('kv')}")

    # 4. upsert：只更新指定列，其它列保持原值
    db.upsert("kv", {"k": "a", "v": "third", "n": 99}, conflict_keys="k", update_columns=["v"])
    print(f"[4] upsert 只更 v -> {db.select_one('kv', {'k': 'a'})}（n 保持 2）")

    # 5. 批量铺一些记忆数据
    db.batch_insert("memories", [
        {"memory_id": f"m{i}", "content": f"记忆{i}", "kind": "chat" if i < 3 else "doc", "access_count": 0}
        for i in range(5)
    ])
    print(f"[5] 写入 5 条 memories")

    # 6. update_where：带谓词的批量更新（IN + 自增表达式）
    hit = db.update_where(
        "memories",
        {"access_count": SqlExpr("access_count + 1")},
        Query(table_name="memories", filters=[Filter(field="memory_id", op=FilterOp.IN, value=["m0", "m1", "m2"])]),
    )
    print(f"[6] update_where 自增命中 {hit} 行 -> " + str([(r["memory_id"], r["access_count"]) for r in db.select("memories", order_by="memory_id")]))

    # 7. update_where：普通值走参数化（含引号也不会被当成 SQL）
    db.update_where(
        "memories",
        {"content": "O'Brien 的笔记"},
        Query(table_name="memories", filters=[Filter(field="memory_id", op=FilterOp.EQ, value="m0")]),
    )
    print(f"[7] 参数化写入 -> {db.select_one('memories', {'memory_id': 'm0'})['content']}")

    # 8. 多列排序查询
    rows = db.select_where(Query(table_name="memories", sort_by=["kind", "memory_id"], sort_order=SortOrder.DESC))
    print(f"[8] 按 kind, memory_id 倒序 -> {[r['memory_id'] for r in rows]}")

    # 9. delete_where：带谓词批量删除
    deleted = db.delete_where(
        "memories", Query(table_name="memories", filters=[Filter(field="kind", op=FilterOp.EQ, value="doc")])
    )
    print(f"[9] delete_where 删除 {deleted} 行，剩余 {db.count('memories')} 行")

    # 10. 安全守卫：空条件直接报错，不会误改/误删全表
    for label, fn in (
        ("update_where", lambda: db.update_where("memories", {"content": "x"}, Query(table_name="memories"))),
        ("delete_where", lambda: db.delete_where("memories", Query(table_name="memories"))),
    ):
        try:
            fn()
        except ValueError as e:
            print(f"[10] {label} 空条件被拦截：{e}")

    db.close()


if __name__ == "__main__":
    main()
