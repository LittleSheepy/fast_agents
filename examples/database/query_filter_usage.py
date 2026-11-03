"""database 模块「谓词下推」用法示例。

演示如何用 ``Query`` / ``Filter`` / ``FilterOp`` / ``SortOrder`` 声明式构建查询：
- 等值/不等值/比较/模糊/集合/空值判断 共 9 种 FilterOp；
- 多条件 AND 组合；
- 排序、分页（limit/offset）；
- select_one_where 单行查询；
- 与原 select(where=dict) 简单等值写法并存（简单场景优先用后者）。

运行方式：在项目根目录执行 `python examples/database/query_filter_usage.py`
"""
from __future__ import annotations

from fast_agents.database import (
    Column,
    ColumnType,
    Filter,
    FilterOp,
    Query,
    Schema,
    SortOrder,
    SQLiteClient,
)


def _seed(db: SQLiteClient) -> None:
    """构造一张示例 users 表并写入 6 条测试数据。"""
    db.ensure_schema(Schema(
        table_name="users",
        columns=[
            Column(name="id",    type=ColumnType.INTEGER, primary_key=True),
            Column(name="name",  type=ColumnType.TEXT,    nullable=False),
            Column(name="age",   type=ColumnType.INTEGER, nullable=True),
            Column(name="score", type=ColumnType.REAL,    nullable=False, default=0.0),
            Column(name="dept",  type=ColumnType.TEXT,    nullable=False),
        ],
        indexes=[
            # 常用检索字段建索引（真实项目建议给 dept、score 等加）
            # 此处仅演示复合索引写法
            # Index(name="idx_users_dept_score", columns=["dept", "score"]),
        ],
    ))
    rows = [
        {"name": "Alice",   "age": 30, "score": 88.5, "dept": "eng"},
        {"name": "Bob",     "age": 25, "score": 72.0, "dept": "eng"},
        {"name": "Charlie", "age": 35, "score": 95.0, "dept": "eng"},
        {"name": "David",   "age": 40, "score": 60.0, "dept": "sales"},
        {"name": "Eve",     "age": 28, "score": 88.5, "dept": "sales"},
        {"name": "Frank",   "age": None, "score": 55.0, "dept": "sales"},  # 未知年龄，演示 IS NULL
    ]
    for r in rows:
        db.insert("users", r)


def _print(title: str, rows: list[dict]) -> None:
    print(f"\n--- {title} ({len(rows)} 行) ---")
    for r in rows:
        print(f"  id={r['id']:>2}  name={r['name']:<8}  age={str(r['age']):>4}  "
              f"score={r['score']:<5}  dept={r['dept']}")


def main() -> None:
    db = SQLiteClient(":memory:")  # 示例用内存数据库；落盘换成文件路径即可
    _seed(db)

    # ---------- 1. 简单等值：原 select(where=dict) 继续用 ----------
    rows = db.select("users", where={"dept": "eng"}, order_by="id ASC")
    _print("1) eng 部门（等值 where=dict）", rows)

    # ---------- 2. GT/GTE/LT/LTE 比较查询 ----------
    rows = db.select_where(Query(
        table_name="users",
        filters=[Filter(field="score", op=FilterOp.GTE, value=85.0)],
        sort_by="score", sort_order=SortOrder.DESC,
    ))
    _print("2) score >= 85（GTE 比较 + DESC 排序）", rows)

    # ---------- 3. LIKE 模糊匹配（通配符由调用方自己写） ----------
    rows = db.select_where(Query(
        table_name="users",
        filters=[Filter(field="name", op=FilterOp.LIKE, value="A%")],
    ))
    _print("3) name LIKE 'A%'（模糊匹配）", rows)

    # ---------- 4. IN 集合匹配 ----------
    rows = db.select_where(Query(
        table_name="users",
        filters=[Filter(field="name", op=FilterOp.IN, value=["Alice", "Eve", "Nobody"])],
        sort_by="id",
    ))
    _print("4) name IN ('Alice','Eve','Nobody')（集合匹配）", rows)

    # ---------- 5. IS NULL / IS NOT NULL ----------
    rows = db.select_where(Query(
        table_name="users",
        filters=[Filter(field="age", op=FilterOp.IS_NULL)],
    ))
    _print("5) age IS NULL（空值判断）", rows)

    # ---------- 6. 多条件 AND（比如 BETWEEN 区间） ----------
    rows = db.select_where(Query(
        table_name="users",
        filters=[
            Filter(field="dept",  op=FilterOp.EQ,  value="eng"),
            Filter(field="score", op=FilterOp.GTE, value=80.0),
        ],
        sort_by="score", sort_order=SortOrder.DESC,
    ))
    _print("6) dept='eng' AND score>=80（AND 组合）", rows)

    # ---------- 7. 分页：limit + offset ----------
    rows = db.select_where(Query(
        table_name="users",
        sort_by="id",
        limit=3, offset=2,
    ))
    _print("7) 按 id 排序，第 3~5 行（limit=3 offset=2）", rows)

    # ---------- 8. select_one_where 单行查询 ----------
    top = db.select_one_where(Query(
        table_name="users",
        sort_by="score", sort_order=SortOrder.DESC,
    ))
    print(f"\n8) score 最高的人：{top}")  # type: ignore[index]

    miss = db.select_one_where(Query(
        table_name="users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Nobody")],
    ))
    print(f"   查不到的人：{miss}")

    # ---------- 9. 参数化安全性（传进来的 value 不会被当成 SQL 执行） ----------
    evil_rows = db.select_where(Query(
        table_name="users",
        filters=[Filter(field="name", op=FilterOp.EQ, value="Alice' OR '1'='1")],
    ))
    _print("9) 注入测试：name=\"Alice' OR '1'='1\"（参数化后应为 0 行）", evil_rows)

    db.close()
    print("\n✓ 全部示例执行完毕。")


if __name__ == "__main__":
    main()
