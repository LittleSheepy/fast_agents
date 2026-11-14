"""MySQLClient 使用示例：接口与 SQLiteClient 100% 对齐，上层零改动即可切换。

前置依赖：pip install pymysql（或 pip install fast-agents[mysql]）
运行方式（Windows PowerShell，在项目根目录）：
    python examples/database/mysql_usage.py
默认只跑 SQLite 分支（零依赖、全内存库，跑完不留文件）；
有真实 MySQL 环境时，把文件末尾的 demo_mysql() 取消注释即可用同一套业务代码跑 MySQL。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.database import (
    MySQLClient,
    SQLiteClient,
    Schema,
    Column,
    ColumnType,
    Index,
    Query,
    Filter,
    FilterOp,
    SortOrder,
    SqlExpr,
)


def demo_sqlite():
    """原有 SQLite 用法完全不变"""
    print("=== SQLite 用法示例 ===")
    db = SQLiteClient(":memory:")
    _run_demo(db)
    db.close()


def demo_mysql():
    """MySQL 用法与 SQLite 完全一致，只需要改初始化连接参数"""
    print("\n=== MySQL 用法示例 ===")
    # 初始化连接（数据库需要提前创建好，客户端不会自动 CREATE DATABASE）
    db = MySQLClient(
        host="127.0.0.1",
        port=9999,
        user="root",
        password="123456",
        database="fast_agents_demo",
        charset="utf8mb4",
        timeout=5.0,
    )
    _run_demo(db)
    db.close()


def _run_demo(db):
    """同一套业务代码，SQLite/MySQL 通用，零改动"""
    # 1. 声明式建表（自动建索引、幂等）
    test_table = "demo_users"
    db.drop_table(test_table, if_exists=True)
    db.ensure_schema(Schema(
        table_name=test_table,
        columns=[
            Column(name="id", type=ColumnType.INTEGER, primary_key=True),
            Column(name="name", type=ColumnType.VARCHAR, nullable=False, length=100, unique=True),
            Column(name="age", type=ColumnType.INTEGER, default=0, index=True),
            Column(name="score", type=ColumnType.REAL, default=0.0),
            Column(name="is_active", type=ColumnType.BOOLEAN, default=True),
            Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False),
        ],
        indexes=[
            Index(name="idx_age_score", columns=["age", "score"]),
        ],
    ))
    print(f"表 {test_table} 创建成功，是否存在：{db.table_exists(test_table)}")

    # 2. 插入单条
    import datetime
    now = datetime.datetime.now().isoformat()
    uid = db.insert(test_table, {
        "name": "张三",
        "age": 25,
        "score": 95.5,
        "created_at": now,
    })
    print(f"插入成功，新行ID：{uid}")

    # 3. 批量插入
    db.batch_insert(test_table, [
        {"name": "李四", "age": 30, "score": 88.0, "created_at": now},
        {"name": "王五", "age": 22, "score": 92.0, "created_at": now},
        {"name": "赵六", "age": 28, "score": 85.5, "created_at": now},
    ])
    print(f"批量插入完成，总行数：{db.count(test_table)}")

    # 4. 等值查询
    zhangsan = db.select_one(test_table, {"name": "张三"})
    print(f"查询张三：{zhangsan}")

    # 5. 谓词下推查询（支持 >= < LIKE IN IS NULL 等操作）
    rows = db.select_where(Query(
        table_name=test_table,
        filters=[
            Filter(field="age", op=FilterOp.GTE, value=25),
            Filter(field="score", op=FilterOp.LT, value=95),
        ],
        sort_by="age",
        sort_order=SortOrder.DESC,
        limit=10,
    ))
    print(f"年龄>=25且分数<90的用户（按年龄倒序）：{[r['name'] for r in rows]}")

    # 6. 更新
    db.update(test_table, {"score": 90.0}, {"name": "张三"})
    print(f"更新张三分数后：{db.select_one(test_table, {'name': '张三'})['score']}")

    # 7. 条件更新 + SQL表达式（自增）
    db.update_where(
        test_table,
        {"score": SqlExpr("score + 1")},  # 分数+1，用SqlExpr标记走原生SQL
        Query(table_name=test_table, filters=[Filter(field="name", op=FilterOp.EQ, value="李四")])
    )
    print(f"李四分数自增1后：{db.select_one(test_table, {'name': '李四'})['score']}")

    # 8. upsert（存在则更新，不存在则插入）
    db.upsert(
        test_table,
        {"name": "张三", "age": 26, "score": 100.0, "created_at": now},
        conflict_keys="name",  # 按name唯一键判断冲突
    )
    print(f"upsert张三后年龄：{db.select_one(test_table, {'name': '张三'})['age']}")

    # 9. 事务（with语句，支持嵌套，异常自动回滚）
    try:
        with db:
            db.insert(test_table, {"name": "事务测试", "age": 99, "score": 0, "created_at": now})
            print(f"事务内插入后行数：{db.count(test_table)}")
            raise ValueError("模拟异常，触发回滚")
    except ValueError:
        pass
    print(f"事务回滚后行数：{db.count(test_table)}（事务测试行已回滚）")

    # 10. 删除
    db.delete(test_table, {"name": "赵六"})
    print(f"删除赵六后行数：{db.count(test_table)}")

    # 清理
    db.drop_table(test_table, if_exists=True)
    print(f"清理表完成，表是否存在：{db.table_exists(test_table)}")


if __name__ == "__main__":
    # 默认只跑 SQLite demo（无需外部依赖）
    # demo_sqlite()

    # 有 MySQL 环境时取消注释跑 MySQL demo
    demo_mysql()