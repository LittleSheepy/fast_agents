"""database 模块单元测试（SQLite 回归测试，无需 pytest）。

全部用内存数据库，不生成文件，不依赖外部服务。
覆盖点：事务失败抛错（不 AttributeError）、None 值编译 IS NULL、
       标识符注入防护白名单校验、异常体系继承关系、
       schema.column_sql 主键列不自动追加 AUTOINCREMENT、
       Query.columns 列投影（默认 SELECT * / 指定列 / 白名单 / 组合查询）。

运行方式（Windows PowerShell，在项目根目录）:
    python tests/database/test_database_sqlite.py
    （无需安装任何额外依赖，纯标准库）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.database import (
    Filter,
    FilterOp,
    Query,
    SortOrder,
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


def test_insert_data_keys_are_ident_checked():
    """回归：insert 的 data 键必须走标识符白名单（与 update 的 data 键同口径）。

    修复前 insert 只对列名做 _quote_ident 转义、不过白名单，非法列名要靠底层报错
    （SQLite 抛 DatabaseOperationalError: near ";": syntax error），与同文件 update 的
    data 键校验口径不一致；修复后 insert 与 update / batch_insert / upsert 统一。
    """
    db = SQLiteClient(":memory:")
    db.ensure_table("t_ik", {"id": "INTEGER PRIMARY KEY", "v": "INTEGER"})
    # 含分隔符的非法列名 -> DatabaseIdentifierError（而不是底层 syntax error）
    with raises(DatabaseIdentifierError):
        db.insert("t_ik", {"id": 1, "v; DROP TABLE t_ik": 2})
    # 以数字开头同样拦截
    with raises(DatabaseIdentifierError):
        db.insert("t_ik", {"1bad": 1})
    # 合法列名照常写入，不破坏现有契约
    assert db.insert("t_ik", {"id": 1, "v": 10}) == 1
    assert db.select_one("t_ik", where={"id": 1})["v"] == 10
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


# ---------- schema.column_sql：纯函数断言（锁定「主键列不追加 AUTOINCREMENT」） ----------

def test_column_sql_pk_never_appends_autoincrement():
    """INTEGER 主键列不自动追加 AUTOINCREMENT（SQLite 的 INTEGER PRIMARY KEY 已是 ROWID 别名）。

    回归背景：column_sql 里曾有一段 `if <int 主键名为 id>: pass` 死代码——条件算完什么都不做。
    清理后行为必须与清理前逐字节一致：仍是 `"id" INTEGER PRIMARY KEY`，
    既不出现 AUTOINCREMENT，也不重复 NOT NULL / UNIQUE。
    """
    from fast_agents.database import Column, ColumnType
    from fast_agents.database.schema import column_sql

    # 名为 id 的 INTEGER 主键
    assert column_sql(Column(name="id", type=ColumnType.INTEGER, primary_key=True)) == \
        '"id" INTEGER PRIMARY KEY'
    # 非 id 名的 INTEGER 主键，行为一致
    assert column_sql(Column(name="seq", type=ColumnType.INTEGER, primary_key=True)) == \
        '"seq" INTEGER PRIMARY KEY'
    # 主键列不再重复加 NOT NULL / UNIQUE（即使显式声明了 nullable=False / unique=True）
    assert column_sql(Column(name="id", type=ColumnType.INTEGER, primary_key=True,
                             nullable=False, unique=True)) == '"id" INTEGER PRIMARY KEY'

    # 建表到真实库仍可用（dict 形式显式声明 AUTOINCREMENT 才是启用路径）
    db = SQLiteClient(":memory:")
    db.create_table("t_idx_pk", [Column(name="id", type=ColumnType.INTEGER, primary_key=True),
                                 Column(name="v", type=ColumnType.TEXT)])
    db.insert("t_idx_pk", {"v": "a"})
    assert db.count("t_idx_pk") == 1
    db.close()


# ---------- Query.columns：列投影 ----------

def test_query_columns_default_selects_all():
    """不传 columns 时保持 SELECT *，返回全部列（零破坏契约）。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_qc1", {"id": "INTEGER PRIMARY KEY", "k": "TEXT", "v": "INTEGER"})
    db.insert("t_qc1", {"id": 1, "k": "a", "v": 10})
    rows = db.select_where(Query(table_name="t_qc1"))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"id", "k", "v"}, f"默认应返回全部列，实际 {sorted(rows[0].keys())}"
    db.close()


def test_query_columns_selects_only_specified_columns():
    """columns 指定后只返回这些列，列顺序与声明一致；select_one_where 同样生效。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_qc2", {"id": "INTEGER PRIMARY KEY", "k": "TEXT", "v": "INTEGER"})
    db.insert("t_qc2", {"id": 1, "k": "a", "v": 10})
    rows = db.select_where(Query(table_name="t_qc2", columns=["v", "k"]))
    assert list(rows[0].keys()) == ["v", "k"], f"列顺序应保持声明顺序，实际 {list(rows[0].keys())}"
    assert rows[0]["k"] == "a" and rows[0]["v"] == 10

    one = db.select_one_where(Query(table_name="t_qc2", columns=["k"]))
    assert one is not None and list(one.keys()) == ["k"]
    db.close()


def test_query_columns_validates_identifiers():
    """columns 里的列名必须过标识符白名单；空列表视为非法参数。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_qc3", {"id": "INTEGER PRIMARY KEY"})
    with raises(DatabaseIdentifierError):
        db.select_where(Query(table_name="t_qc3", columns=["id; DROP TABLE t_qc3"]))
    with raises(DatabaseIdentifierError):
        db.select_where(Query(table_name="t_qc3", columns=["id OR 1=1"]))
    with raises(DatabaseValidationError):
        db.select_where(Query(table_name="t_qc3", columns=[]))
    db.close()


def test_query_columns_with_filters_and_sort():
    """columns 与 filters / sort_by / limit 组合使用正常。"""
    db = SQLiteClient(":memory:")
    db.ensure_table("t_qc4", {"id": "INTEGER PRIMARY KEY", "k": "TEXT", "v": "INTEGER"})
    for i in range(3):
        db.insert("t_qc4", {"id": i + 1, "k": f"k{i}", "v": i})
    rows = db.select_where(Query(
        table_name="t_qc4",
        columns=["id", "v"],
        filters=[Filter(field="v", op=FilterOp.GTE, value=1)],
        sort_by="v",
        sort_order=SortOrder.DESC,
        limit=5,
    ))
    assert [r["id"] for r in rows] == [3, 2], f"过滤+排序结果错误: {[r['id'] for r in rows]}"
    assert all(set(r.keys()) == {"id", "v"} for r in rows)
    db.close()


def test_executescript_in_transaction_does_not_commit_on_error():
    """回归 Bug1：with 事务里 executescript 不能隐式提交；外层异常必须回滚。"""
    db = SQLiteClient(":memory:")
    db.executescript("CREATE TABLE t (v TEXT NOT NULL);")

    raised = False
    try:
        with db:
            db.insert("t", {"v": "a"})
            db.executescript("INSERT INTO t(v) VALUES('b'); INSERT INTO t(v) VALUES('c');")
            raised = True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert raised, "异常没有按预期抛出"
    rows = db.query("SELECT v FROM t")
    assert rows == [], f"executescript 隐式提交导致回滚失效，残留: {rows}"
    db.close()


def test_executescript_skips_semicolons_inside_strings_and_comments():
    """切分不能把字符串/注释里的分号当作语句分隔符。"""
    from fast_agents.database.sqlite_client import _split_sql_statements

    sql = (
        "INSERT INTO t(v) VALUES('a;b'); -- 尾注释分号;\n"
        "/* 块注释;\n 内部分号 */"
        "INSERT INTO t(v) VALUES(\"quoted;semi\");"
    )
    stmts = _split_sql_statements(sql)
    assert len(stmts) == 2, f"切分数量错误: {stmts}"
    assert stmts[0].startswith("INSERT INTO t(v) VALUES('a;b')"), f"第1条内容异常: {stmts[0]}"
    assert "quoted;semi" in stmts[1], f"第2条内容异常: {stmts[1]}"

    db = SQLiteClient(":memory:")
    db.executescript("CREATE TABLE t (v TEXT);")
    db.executescript(sql)
    rows = db.query("SELECT v FROM t ORDER BY v")
    assert [r["v"] for r in rows] == ["a;b", "quoted;semi"], rows
    db.close()


def test_executescript_outside_transaction_commits_all_statements():
    """事务外调用 executescript 仍按原语义自动提交，多条语句全部落盘。"""
    db = SQLiteClient(":memory:")
    db.executescript("CREATE TABLE t (v TEXT); INSERT INTO t(v) VALUES('x'); INSERT INTO t(v) VALUES('y');")
    rows = db.query("SELECT v FROM t ORDER BY v")
    assert [r["v"] for r in rows] == ["x", "y"], rows
    db.close()


def test_executescript_line_comment_without_space_after_dash_dash():
    """SQLite 行注释规则：-- 后不要求空白，--x; 整行都是注释，注释里的语句不能被执行。

    回归用例：SQLite 切分器此前照搬 MySQL 规则（-- 后必须紧跟空白才进入注释态），
    导致 ``--x; INSERT INTO ...`` 里的分号被当成分隔符、注释中的 INSERT 真被执行。
    """
    from fast_agents.database.sqlite_client import _split_sql_statements

    sql = "INSERT INTO t(v) VALUES('a'); --x; INSERT INTO t(v) VALUES('evil');\n"
    stmts = _split_sql_statements(sql)
    # 第 1 条是可执行 INSERT；后续条目是注释文本（以 -- 开头），SQLite 执行
    # 时整行被视为注释，其中的 INSERT 不会单独成为可执行语句。
    assert stmts[0].startswith("INSERT INTO t(v) VALUES('a')"), f"第 1 条内容异常: {stmts[0]}"
    for s in stmts[1:]:
        assert s.lstrip().startswith("--"), f"后续条目应为注释行，实际: {s!r}"
    # 锁死关键 bug：evil 必须跟在 --x; 注释之内，不能作为独立可执行 INSERT 出现
    joined = "".join(stmts)
    evil_pos = joined.find("INSERT INTO t(v) VALUES('evil')")
    assert evil_pos != -1, "注释文本应保留 evil 原始内容"
    prefix = joined[:evil_pos].rstrip()
    assert prefix.endswith("--x;"), f"evil 应位于注释 --x; 之后，实际前序结尾: {prefix[-30:]!r}"

    db = SQLiteClient(":memory:")
    db.executescript("CREATE TABLE t (v TEXT NOT NULL);")
    db.executescript(sql)
    rows = db.query("SELECT v FROM t ORDER BY v")
    assert [r["v"] for r in rows] == ["a"], f"注释中的 evil 被意外执行，残留: {rows}"
    db.close()


def test_executescript_distinguishes_sqlite_vs_mysql_comment_rule():
    """SQLite -- 后无空白进入注释；MySQL -- 后必须空白。两个切分器规则独立，不能互相混用。"""
    from fast_agents.database.sqlite_client import _split_sql_statements as split_sqlite
    from fast_agents.database.mysql_client import _split_sql_statements as split_mysql

    sql = "INSERT INTO t(v) VALUES('a'); --x; INSERT INTO t(v) VALUES('b');\n"
    s = split_sqlite(sql)
    # SQLite：--x 整行进注释；第 1 条是 INSERT，后续为注释文本
    assert s[0].startswith("INSERT INTO t(v) VALUES('a')"), f"SQLite 第 1 条内容异常: {s}"
    for seg in s[1:]:
        assert seg.lstrip().startswith("--"), f"SQLite 后续条目应为注释行，实际: {seg!r}"
    m = split_mysql(sql)
    # MySQL：--x 因缺空白不被识别为注释（MySQL 官方规则），分号后的 INSERT INTO ...('b')
    # 应作为独立可执行语句被切出来（条数可能因中间的 --x 片段差异而变化，核心是
    # 要能切到 'b' 这条 INSERT）。
    assert any("VALUES('b')" in seg for seg in m[1:]), f"MySQL 未切出第 2 条 INSERT: {m}"


# ---------- select() limit 非负校验 ----------

def test_select_negative_limit_raises_validation_error():
    """回归：select(limit=负数) 必须抛 DatabaseValidationError，不能静默返回全表。

    修复前 SQLite 会把 `LIMIT -1` 当成"无上限"而返回全表（与调用方"限制条数"的预期相反）；
    修复后与 _compile_query 同口径——limit 强制非负 int，负数在拼接前直接报错。
    """
    db = SQLiteClient(":memory:")
    db.ensure_table("t_lim", {"id": "INTEGER PRIMARY KEY", "v": "INTEGER"})
    for i in range(1, 6):
        db.insert("t_lim", {"id": i, "v": i})

    # 负数：-1 / -2 都必须报错，而不是返回全部 5 行
    with raises(DatabaseValidationError, match="limit 不能为负数"):
        db.select("t_lim", limit=-1)
    with raises(DatabaseValidationError, match="limit 不能为负数"):
        db.select("t_lim", where={"v": 1}, limit=-2)

    # 边界：limit=0 是合法语义（取 0 行），不能被误伤
    assert db.select("t_lim", limit=0) == []
    # 正数与不传 limit 的原有行为保持不变
    assert len(db.select("t_lim", limit=2)) == 2
    assert len(db.select("t_lim")) == 5
    db.close()


# ---------- create_table 空列校验 ----------

def test_create_table_empty_list_raises_validation_error():
    """回归：create_table 的 list[Column] 分支空列表必须抛 DatabaseValidationError。

    修复前 list 分支无校验，[] 会被拼成 `CREATE TABLE t ()` 这种非法语句、
    由底层驱动报 DatabaseOperationalError（执行层），与 dict 分支的校验层报错不对称；
    修复后与 dict 分支对称：空列表在拼接前直接抛 DatabaseValidationError、不产生任何待执行语句。
    """
    from fast_agents.database import Column, ColumnType

    db = SQLiteClient(":memory:")

    # list 空列表 / dict 空字典：都必须走校验层，报错类型与文案一致
    with raises(DatabaseValidationError, match="columns 不能为空"):
        db.create_table("t_empty_list", [])
    with raises(DatabaseValidationError, match="columns 不能为空"):
        db.create_table("t_empty_dict", {})
    # 校验先于执行：不应留下任何半成品表
    assert not db.table_exists("t_empty_list")
    assert not db.table_exists("t_empty_dict")

    # 非空 list 的原有行为保持不变
    db.create_table(
        "t_ok",
        [Column(name="id", type=ColumnType.INTEGER, primary_key=True),
         Column(name="name", type=ColumnType.TEXT)],
    )
    assert db.table_exists("t_ok")
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

    # AI新添加的测试(AI可以删除覆盖)
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
