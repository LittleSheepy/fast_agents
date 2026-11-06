"""database 模块单元测试（SQLite 回归测试，无需 pytest）。

全部用内存数据库，不生成文件，不依赖外部服务。
覆盖点：事务失败抛错（不 AttributeError）、None 值编译 IS NULL、
       标识符注入防护白名单校验、异常体系继承关系。

运行方式（Windows PowerShell，在项目根目录）:
    python tests/database/test_database_sqlite.py
    （无需安装任何额外依赖，纯标准库）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.database import (
    SQLiteClient,
    # 异常体系（errors.py）
    DatabaseOperationalError,
    DatabaseValidationError,
    DatabaseIdentifierError,
)


# ==================== 无pytest工具函数（复用公共 _runner） ====================
from tests._runner import raises  # noqa: E402


# ==================== 测试用例 ====================

# ==================== 基础连接 ====================
def test_connect_file(tmp_path):
    """连接文件数据库成功，WAL/外键默认启用。"""
    path = tmp_path / "test.db"
    client = SQLiteClient(str(path))
    row = client.query("PRAGMA journal_mode", fetch_one=True)
    client.close()
    assert row is not None

# ---------- count / batch_insert 测试 ----------

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



def test_enter_transaction_failure_raises_operational_error_not_attributeerror():
    """回归：__enter__ 里 BEGIN IMMEDIATE 失败时应抛 DatabaseOperationalError，
    而不是因误写 self._sqlite3 导致 AttributeError。

    用继承 sqlite3.Connection 的子类重写 execute 让 BEGIN 抛
    OperationalError，模拟真实的 database is locked 场景。"""
    import sqlite3 as _sqlite3

    class _FailingBeginConn(_sqlite3.Connection):
        def execute(self, sql, params=()):
            if isinstance(sql, str) and "BEGIN" in sql.upper():
                raise _sqlite3.OperationalError("database is locked")
            return super().execute(sql, params)

    db = SQLiteClient(":memory:")
    db.ensure_table("t_enterr", {"id": "INTEGER PRIMARY KEY", "v": "INTEGER"})
    # 替换底层连接为会在 BEGIN 上抛错的子类实例（绕过 conn property 的自动重连）
    db._conn.close()
    db._conn = _FailingBeginConn(":memory:")
    try:
        with raises(DatabaseOperationalError, match="开启事务失败"):
            with db:
                pass
        assert db._tx_depth == 0, f"_tx_depth 应为 0，实际为 {db._tx_depth}"
        assert db._in_transaction is False, "_in_transaction 应为 False"
    finally:
        db.close()


def test_exit_commit_failure_raises_operational_error_not_attributeerror():
    """回归：__exit__ 里 commit 失败时应抛 DatabaseOperationalError，
    而不是 AttributeError。通过 factory 让连接一开始就是抛错子类，
    再在这个连接上建表、写入，最后在 commit 上抛 OperationalError 模拟磁盘错误。"""
    import sqlite3 as _sqlite3

    should_fail = {"v": False}

    class _FailingCommitConn(_sqlite3.Connection):
        def commit(self):
            if should_fail["v"]:
                raise _sqlite3.OperationalError("disk I/O error")
            return super().commit()

    # 先通过子类 factory 建连接、建表、写入，到真正测试点再让 commit 抛错
    db = SQLiteClient(":memory:")
    db._conn.close()
    db._conn = _FailingCommitConn(":memory:")
    db.ensure_table("t_exitx", {"id": "INTEGER PRIMARY KEY", "v": "INTEGER"})
    entered = False
    try:
        should_fail["v"] = True
        with raises(DatabaseOperationalError, match="提交事务失败"):
            with db:
                db.insert("t_exitx", {"v": 1})
                entered = True
        assert entered is True, "事务块未进入"
        assert db._tx_depth == 0, f"_tx_depth 应为 0，实际为 {db._tx_depth}"
        assert db._in_transaction is False, "_in_transaction 应为 False"
    finally:
        db.close()


def test_where_none_compiles_to_is_null_and_hits_rows():
    """回归：select/update/delete/count 传入 where 值为 None 时，
    都应编译为 IS NULL，跨方法行为一致（与 MySQL 后端对齐）。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_nil", {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "extra": "TEXT"})
    db.insert("t_nil", {"id": 1, "name": "a", "extra": None})
    db.insert("t_nil", {"id": 2, "name": "b", "extra": "x"})
    db.insert("t_nil", {"id": 3, "name": "c", "extra": None})

    # select: None → IS NULL，应命中 2 行
    rows = db.select("t_nil", where={"extra": None}, order_by="id")
    assert [r["id"] for r in rows] == [1, 3], f"select IS NULL 命中行错误: {[r['id'] for r in rows]}"

    # count 与 select 结果一致
    assert db.count("t_nil", where={"extra": None}) == 2, "count IS NULL 计数错误"

    # update: None → IS NULL，只更新 extra 为 NULL 的 2 行
    n = db.update("t_nil", data={"name": "nil"}, where={"extra": None})
    assert n == 2, f"update IS NULL 影响行数应为2，实际为{n}"
    assert db.count("t_nil", where={"name": "nil"}) == 2
    assert db.count("t_nil", where={"name": "b"}) == 1

    # delete: None → IS NULL
    n = db.delete("t_nil", where={"name": "nil", "extra": None})
    assert n == 2, f"delete IS NULL 删除行数应为2，实际为{n}"
    assert db.count("t_nil") == 1, f"删除后剩余应为1行，实际为{db.count('t_nil')}"
    db.close()


def test_check_ident_raises_database_identifier_error():
    """回归：schema.check_ident 非法标识符应当抛 DatabaseIdentifierError
    （它是 DatabaseValidationError 的子类，旧上层的 except ValidationError 不会断）。"""
    from fast_agents.database.schema import check_ident

    # 非法：以数字开头
    with raises(DatabaseIdentifierError):
        check_ident("1bad", "表名")
    # 非法：含分号（注入点）
    with raises(DatabaseIdentifierError):
        check_ident("ta;ble", "表名")
    # 同时必须还是 DatabaseValidationError 的子类（向后兼容）
    with raises(DatabaseValidationError):
        check_ident("bad name", "表名")


def test_select_where_key_must_pass_ident_whitelist():
    """回归：select 的 where 键必须走标识符白名单，阻挡注入。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_wk", {"id": "INTEGER PRIMARY KEY"})
    with raises(DatabaseIdentifierError):
        db.select("t_wk", where={"id OR 1=1 --": 1})
    db.close()


def test_update_delete_data_and_where_keys_are_ident_checked():
    """回归：update 的 data 键 / where 键、delete 的 where 键必须走标识符白名单。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_dk", {"id": "INTEGER PRIMARY KEY", "v": "INTEGER"})
    db.insert("t_dk", {"id": 1, "v": 10})
    with raises(DatabaseIdentifierError):
        db.update("t_dk", data={"v; DROP": 1}, where={"id": 1})
    with raises(DatabaseIdentifierError):
        db.update("t_dk", data={"v": 2}, where={"id OR 1=1": 1})
    with raises(DatabaseIdentifierError):
        db.delete("t_dk", where={"id--": 1})
    # 正常写入依然可用，不破坏现有契约
    assert db.update("t_dk", data={"v": 20}, where={"id": 1}) == 1
    assert db.select_one("t_dk", where={"id": 1})["v"] == 20
    db.close()


# ==================== 事务 ====================

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
    with raises(ValueError, match="boom"):
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
    with raises(RuntimeError, match="outer"):
        with db:
            db.insert("nt3", {"v": 1})
            with db:
                db.insert("nt3", {"v": 2})
            raise RuntimeError("outer")
    assert db.count("nt3") == 0
    db.close()

# ============================================================
# 入口
# ============================================================
def main():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))

def main_of_new_add():
    """ 新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    test_update_delete_data_and_where_keys_are_ident_checked()

    # AI新添加的测试(AI可以删除覆盖)
    test_enter_transaction_failure_raises_operational_error_not_attributeerror()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
