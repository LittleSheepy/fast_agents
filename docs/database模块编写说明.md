# database 模块编写说明

> **模块职责**：fast_agents 的数据库基础服务层。
> 当前版本（v1.1.0）提供最基础的 SQLite 客户端，封装标准库 `sqlite3` 的连接管理、**声明式/幂等建表**、索引、增删改查，返回字典行结果，支持事务上下文管理。
> **零新增依赖**、零抽象层（除已有的 pydantic），上层所有需要持久化的模块（memory / 日志 / trace / RAG 元数据等）统一使用本模块做底层存储，避免每个模块重复写连接/cursor/SQL 模板。
> 未来按需扩展 MySQL / Postgres / Chroma 等后端，上层代码保持不变。

---

## 一、为什么要有 database 模块

在 v0.x 版本里，`memory/` 模块自己维护 SQLite 连接、自己写建表 SQL、自己开游标。当要加日志持久化、trace 记录、RAG 元数据表时，如果每个模块都重复写一遍：
- 连接管理各自一套，配置不统一（WAL、外键、超时等）；
- 每个模块都要手写一遍 `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS`，列约束、默认值、布尔类型映射各自处理；
- 表升级（补列）要每个模块重复写一次 `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` 的幂等逻辑；
- 将来换 MySQL/Postgres 时每个模块都要改一遍。

抽一个统一的 `database` 基础服务层，所有持久化走同一套接口，解决上面四个问题：
1. **连接统一配置**：WAL / 外键 / 超时 / 行工厂默认开好，上层不用关心；
2. **结果统一格式**：查询返回 `dict` / `list[dict]`，不用再手动 `dict(row)`；
3. **参数化查询 + 标识符校验**：值参数统一用 `:name` 占位，表名/列名做合法标识符校验，防止 SQL 注入；
4. **声明式幂等建表**：用 `Schema / Column / Index` 三个 Pydantic 模型描述表结构，`ensure_table / ensure_schema` 一次调用同时建表 + 建索引，`migrate=True` 自动补列，各业务模块不用再复制粘贴建表 SQL；
5. **事务开箱即用**：`with db:` 自动提交/回滚，不用手动 BEGIN/COMMIT/ROLLBACK。

---

## 二、目录结构

```
fast_agents/database/
├── __init__.py               # 对外导出：SQLiteClient / DatabaseClientBase / Schema / Column / ColumnType / Index / Query / Filter / FilterOp / SortOrder
├── database_client_base.py   # 抽象基类，统一约定所有后端的最小接口集（连接/DDL/DML/事务）
├── schema.py                 # 声明式表结构模型：ColumnType / Column / Index / Schema
├── query.py                  # 轻量查询对象：SortOrder / FilterOp / Filter / Query
└── sqlite_client.py          # 核心：SQLite 客户端类（唯一实现，继承基类）
examples/
└── database/
    ├── basic_usage.py                # 快速上手（CRUD + 事务）
    └── create_table_usage.py         # 建表能力演示（三种写法 + 幂等 + migrate + 防注入）
test/
└── test_database.py          # 单元测试（全内存、无外部依赖、38 项用例）
docs/
└── database模块编写说明.md   # 本文档
```

---

## 三、快速上手

### 3.1 最简：3 行建表 + CRUD

```python
from fast_agents.database import SQLiteClient

with SQLiteClient("app.db") as db:
    # 建表（dict 写法，最快）
    db.ensure_table("users", {
        "id":   "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
        "age":  "INTEGER",
    })

    uid = db.insert("users", {"name": "Alice", "age": 30})
    print(db.select_one("users", where={"id": uid}))
    db.update("users", {"age": 31}, where={"id": uid})
    db.delete("users", where={"id": uid})
```

### 3.2 工程化：Schema 声明式建表（推荐）

```python
from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index

db = SQLiteClient("app.db")

db.ensure_schema(Schema(
    table_name="orders",
    columns=[
        Column(name="id",         type=ColumnType.INTEGER, primary_key=True),
        Column(name="user_id",    type=ColumnType.INTEGER, nullable=False, index=True),
        Column(name="amount",     type=ColumnType.REAL,    nullable=False, default=0.0),
        Column(name="status",     type=ColumnType.TEXT,    nullable=False, default="pending"),
        Column(name="is_paid",    type=ColumnType.BOOLEAN, nullable=False, default=False),
        Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False),
    ],
    indexes=[
        Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
    ],
))
```

`ensure_schema` 会：
1. 表不存在 → 建表 + 建所有索引（列级 index + 显式 Index 列表）；
2. 表已存在 → 默认**不改动**表结构，仅幂等补齐索引（`migrate=False`）；
3. 传 `migrate=True` → 额外自动 `ALTER TABLE ... ADD COLUMN` 补齐缺失列（SQLite 只支持加列，不支持 drop/rename）。

### 3.3 运行示例

```powershell
# Windows PowerShell，项目根目录
python examples\database\basic_usage.py
python examples\database\create_table_usage.py
```

---

## 四、核心 API：SQLiteClient

### 4.1 构造

```python
SQLiteClient(db_path: str = ":memory:", timeout: float = 5.0)
```

- `db_path`：数据库文件路径，默认 `":memory:"`（内存数据库，进程结束即消失）。**传入文件路径会自动创建文件**。
- `timeout`：SQLite 写锁等待超时秒数（默认 5 秒）。

**默认配置**（构造时自动执行）：
- `PRAGMA foreign_keys = ON`：启用外键约束；
- `PRAGMA journal_mode = WAL`：文件库启用 WAL 模式，读写并发更友好（内存库不启用）；
- `row_factory = sqlite3.Row`：查询结果可按键访问，最终统一转成 `dict`。

### 4.2 连接管理

| 方法 | 说明 |
|---|---|
| `connect()` | 显式连接；`close()` 之后调用会自动重连。 |
| `close()` | 手动关闭连接。`__del__` 析构时也会尝试关闭。 |
| `conn` 属性 | 底层 `sqlite3.Connection`，连接关闭后访问会自动重连。 |

### 4.3 DDL：建表 / 查表 / 删表（本次新增核心能力）

#### `table_exists(table) → bool`
判断表是否已存在。内部查 `sqlite_master`，上层用它做「建表前判断」或启动期检查。

#### `create_table(table, columns, if_not_exists=True)`
建表。`columns` 支持三种写法，按方便程度任选：

| 形式 | 适用场景 | 示例 |
|---|---|---|
| `dict[str, str]` | 快速原型，完全手写 SQL 片段 | `{"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL"}` |
| `list[Column]` | 声明式，需要 NOT NULL/DEFAULT/UNIQUE/INDEX 时 | `[Column(name="id", type=ColumnType.INTEGER, primary_key=True), ...]` |
| `Schema` 对象 | 工程化推荐，列 + 索引一次声明 | `Schema(table_name="t", columns=[...], indexes=[...])` |

`if_not_exists=False` 时，表已存在会抛异常；默认 `True`，幂等。

#### `ensure_table(table, columns)`
等价于 `create_table(table, columns, if_not_exists=True)`，**最常用的入口**，语义是「表存在就跳过，不存在就建」。
注意：**不会**修改已有表结构（缺列不会补），要补列请用 `ensure_schema(..., migrate=True)`。

#### `ensure_schema(schema: Schema, migrate: bool = False)`
按 Schema 对象幂等建表 + 建索引：
- 表不存在 → 建表 + 建所有列级索引 + 建 `schema.indexes` 里声明的显式索引；
- 表已存在、`migrate=False`（默认）→ 只幂等补索引，不改动列；
- 表已存在、`migrate=True` → 额外用 `PRAGMA table_info` 查现有列，对缺失列执行 `ALTER TABLE ADD COLUMN`，新列必须满足 SQLite 限制（不能是 NOT NULL 无默认值，不能加 UNIQUE/PRIMARY KEY——SQLite 本身不允许，会原样抛错）。

#### `drop_table(table, if_exists=True)`
删表。`if_exists=False` 时表不存在会报错。

**标识符校验**：上述所有方法在执行前都会用正则校验 `table / column / index` 名（必须是字母/数字/下划线，不以数字开头），非法名字直接抛 `ValueError`，从源头堵住 SQL 注入（因为表名/列名不能参数化，这是唯一的防线）。

### 4.4 基础执行

#### `execute(sql, params=()) → dict`
执行写 SQL（INSERT/UPDATE/DELETE/CREATE/DROP），**自动 commit**，异常时自动 rollback。
返回 `{"lastrowid": int, "rowcount": int}`。

```python
res = db.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))
print(res["lastrowid"])
```

#### `executemany(sql, params_list) → dict`
批量执行，返回最后一条的 lastrowid 和总影响行数。

#### `executescript(sql) → None`
执行一段脚本（多条 SQL，以 `;` 分隔），不在事务中时自动提交。适合初始化多表。

#### `query(sql, params=(), fetch_one=False) → list[dict] | dict | None`
执行读 SQL（SELECT / PRAGMA）。
- 默认返回 `list[dict]`；
- `fetch_one=True` 时返回第一条（`dict`）或 `None`。

### 4.5 便捷 CRUD

> **注意**：`table` 参数**会做合法标识符校验**，只允许字母/数字/下划线；`order_by` 仅允许「列名 [ASC|DESC]」形式。值参数一律走**参数化查询**，安全防注入。

#### `insert(table, data) → int`
插入一行，返回新行 `id`。`data` 是 `{列名: 值}` 字典。

#### `select(table, where=None, order_by=None, limit=None) → list[dict]`
查询多行：
- `where`：`{列名: 值}` 等值条件，多个用 AND；
- `order_by`：排序子句，如 `"id DESC"`（受白名单校验）；
- `limit`：返回条数上限。

#### `select_one(table, where=None, order_by=None) → dict | None`
查询单行，不存在返回 `None`。

#### `update(table, data, where) → int`
更新行，返回影响行数。**`where` 必填**。

#### `delete(table, where) → int`
删除行，返回影响行数。**`where` 必填**，禁止不传 where 删除全表。

### 4.6 事务上下文管理

`SQLiteClient` 支持 `with` 语句做显式事务：

```python
with db:
    db.insert("users", {"name": "A"})
    db.insert("users", {"name": "B"})
    # with 块正常退出 → 自动 commit；抛异常 → 自动 rollback
```

SQLite 本身不支持嵌套事务，不要在同一个连接上套多层 `with db:`。

---

## 五、核心数据模型：Schema / Column / ColumnType / Index

都在 `fast_agents/database/schema.py`，基于 pydantic（pydantic 已是整个框架的基础依赖，**零新依赖**）。

### 5.1 `ColumnType`
与具体数据库方言无关的列类型枚举，内部会映射到 SQLite 真实类型：

| 枚举值 | SQLite 映射 | 说明 |
|---|---|---|
| `TEXT` | TEXT | 字符串 |
| `INTEGER` | INTEGER | 整数 |
| `REAL` | REAL | 浮点数 |
| `BLOB` | BLOB | 二进制 |
| `BOOLEAN` | INTEGER | 布尔存 0/1（SQLite 无原生布尔） |
| `VARCHAR` | TEXT | 变长字符串（length 字段仅作文档/校验提示，SQLite 不强制长度） |
| `TIMESTAMP` | TEXT | 时间戳统一存 ISO 字符串（对齐 memory 模块 `created_at` 的做法） |

### 5.2 `Column`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 列名（合法标识符） |
| `type` | ColumnType | 必填 | 列类型 |
| `primary_key` | bool | False | 是否主键（INTEGER PRIMARY KEY 自动获得 ROWID 别名） |
| `nullable` | bool | True | 是否允许 NULL（主键隐式 NOT NULL） |
| `default` | Any | None | 默认值，自动转 SQL 字面量（bool→0/1，数字直写，str 转义单引号） |
| `unique` | bool | False | 是否唯一约束 |
| `index` | bool | False | 是否创建普通索引（索引名自动生成 `idx_{table}_{column}`） |
| `length` | int \| None | None | VARCHAR 长度（仅作文档/上层校验） |

### 5.3 `Index`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 索引名（合法标识符） |
| `columns` | list[str] | 必填 | 索引列名列表，支持单列/复合索引 |
| `unique` | bool | False | 是否唯一索引 |

### 5.4 `Schema`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `table_name` | str | 必填 | 表名 |
| `columns` | list[Column] | 必填 | 列定义列表 |
| `indexes` | list[Index] \| None | None | 显式索引列表（单列 index=True 的列会自动建索引，无需在这里重复） |

### 5.5 查询模型（`query.py`）

`Query / Filter / FilterOp / SortOrder` 预留给未来的高级查询构造器，当前 SQLiteClient 暂未消费，上层模块如需复杂过滤/分页/排序可直接引用；现阶段仍以 dict where + order_by 字符串为主。

---

## 六、逐个文件讲解（源码导读）

### 6.1 `__init__.py` —— 模块出口
统一导出客户端、基类、Schema 模型和查询模型。上层使用时：
```python
from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index
```
一次导入所有需要的符号，避免记忆多个子模块路径。

### 6.2 `database_client_base.py` —— 抽象基类
定义 `DatabaseClientBase(ABC)`，统一所有后端（SQLite/MySQL/Postgres）的最小接口：连接、DDL（`table_exists/create_table/ensure_table/ensure_schema/drop_table`）、DML（insert/select/select_one/update/delete）。
`SQLiteClient` 继承它；将来加 MySQL 客户端时继承它即可，上层代码不变。

### 6.3 `schema.py` —— 声明式模型
`ColumnType / Column / Index / Schema` 四个 Pydantic 模型，纯数据、零逻辑。注意 `Column` 的 `index=True` 是"列级快捷索引"，`Schema.indexes` 是"需要命名或复合的显式索引"，两种都会建，不冲突。

### 6.4 `query.py` —— 查询对象
预留给未来的 Query Builder，当前不强制使用，保持极简。

### 6.5 `sqlite_client.py` —— 唯一实现
核心实现文件。要点：

1. **建表 SQL 生成**（`_column_sql` / `_create_table_raw` / `_create_table_from_schema`）：把 `Column` 模型拼成 `col_name TYPE [PRIMARY KEY] [NOT NULL] [UNIQUE] [DEFAULT x]`，再拼 `CREATE TABLE [IF NOT EXISTS] ...`；
2. **类型映射**（`_TYPE_TO_SQLITE`）：BOOLEAN→INTEGER、VARCHAR→TEXT、TIMESTAMP→TEXT，DEFAULT 字面量通过 `_sql_literal` 统一转义；
3. **标识符白名单**（`_check_ident` + `_IDENT_RE`）：表/列/索引名必须是 `^[A-Za-z_][A-Za-z0-9_]*$`，防止拼接注入；
4. **幂等索引**（`_create_indexes` + `_list_indexes`）：先查 `sqlite_master`，已存在的索引跳过（SQLite 虽有 `IF NOT EXISTS` 但老版本行为不完全一致，先查一次更稳）；
5. **migrate 补列**：`ensure_schema(migrate=True)` 通过 `PRAGMA table_info` 取现有列集合，对 `Schema.columns` 里缺的列执行 `ALTER TABLE ... ADD COLUMN`，**只补不删不改**，安全幂等；
6. **CRUD 里也加了标识符校验和 order_by 白名单**，比 v1.0 更严格但不破坏正常用法。

旧的手写 `db.execute("CREATE TABLE ...")` 写法**完全保留**，上层已有的建表 SQL 一行都不用改。

---

## 七、模块设计取舍

| 决策 | 原因 | 代价 |
|---|---|---|
| 只做 SQLite 一个后端 | SQLite 是标准库自带，零依赖、零部署成本，覆盖 90% 单机/嵌入式/原型场景。 | 目前不能直接连 MySQL/Postgres，需后续扩展。 |
| 声明式 Schema（pydantic）而不是 ORM | ORM 是重型依赖（SQLAlchemy 大几百 KB），不符合"零重型依赖"定位；pydantic 已是框架基础依赖，用它做 DDL 描述零新增成本。 | 复杂关系/外键/迁移还需要手写 SQL，不做全功能 ORM。 |
| 三种建表写法并存（dict / list[Column] / Schema） | 写原型要快（dict 一行）、写模块要稳（Schema 带索引/迁移），按场景选，不强制一种风格。 | API 表面略大，但三个参数都是同一方法重载，心智负担小。 |
| migrate 默认 False | "只建不改"最安全，不会在升级时意外改线上表结构；要补列必须显式传 `migrate=True`。 | 加列时要显式写 migrate=True，多打一个参数。 |
| BOOLEAN→INTEGER / VARCHAR→TEXT / TIMESTAMP→TEXT | SQLite 本身没有这些类型，映射到最贴近的原生类型，保持跨后端的类型抽象。 | 严格长度/类型检查需要上层自己做（SQLite 是柔性类型）。 |
| 表名/列名/索引名做白名单校验 | 这些字段没法参数化，是 SQL 注入的最后一道口子；合法标识符本来也不会用特殊字符。 | 想给表名取带空格、关键字、点号的人需要用 `db.execute` 手写。 |
| order_by 只允许「列名 [ASC\|DESC]」 | 绝大多数场景只用这一种，完全堵住 `order_by` 拼接注入。 | 复杂 order_by（多列、表达式）需要走 `db.query` 手写 SQL。 |
| 默认启用 WAL 模式 | 单写多读场景下 WAL 比 DELETE 模式并发好太多，且是 SQLite 官方推荐。 | 内存库不适用（自动跳过）。 |
| delete/update 强制传 where | 防止 `DELETE FROM users` 这种忘写条件的灾难。 | 真想删全表要手写 `db.execute("DELETE FROM users")`，故意为之。 |

---

## 八、常见坑

1. **SQLite 的 `ALTER TABLE` 只能 ADD COLUMN**：`ensure_schema(migrate=True)` 只能补列，不能删列、改列类型、改主键、加 UNIQUE。这类破坏性迁移需要手写 SQL，或新建表 + 拷贝数据 + 重命名（未来版本提供小工具）。
2. **NOT NULL 的新列必须给 default**：SQLite 不允许 `ALTER TABLE ADD COLUMN x TEXT NOT NULL`（老行会是 NULL，违反约束），Column 定义时写 `default=""` 或 `default=None+nullable=True`。
3. **migrate 默认是 False**：只建表+建索引，不动列。升级代码时如果加了字段，记得显式传 `migrate=True`。
4. **`ensure_table(list[Column])` 不会建复合索引**：列级 `index=True` 只建单列索引；复合/唯一索引必须用 `Schema(indexes=[...])`。
5. **INTEGER PRIMARY KEY 就是 ROWID 别名**：插入时不传 id 会自动分配，不用强行加 AUTOINCREMENT（AUTOINCREMENT 语义更严格但有额外开销）。
6. **WAL 模式会多生成 `-wal`/`-shm` 文件**：SQLite 正常行为，备份时一起拷走即可。
7. **内存数据库（":memory:"）只在当前连接可见**：每个 `SQLiteClient(":memory:")` 都是独立的库，测试用很方便，生产请传文件路径。
8. **`execute` 会把 sqlite3.Error 包成 ValueError 抛出**：`__cause__` 是原始 sqlite3 错误；事务里的异常会触发 __exit__ 回滚，所以捕获 ValueError 也能正确回滚。
9. **不要把用户输入直接当表名/列名/order_by**：即使有白名单校验，这仍然是错的设计——表/列/索引名应当是代码里写死的或经过白名单的内部常量。

---

## 九、模块关系与接入规范

database 是**底层基础设施**，依赖关系单向：

```
上层业务模块（memory / logs / trace / RAG）
        │  只依赖 SQLiteClient + Schema 公开 API
        ▼
  fast_agents/database/   ← 本模块
        │  只依赖 Python 标准库 sqlite3 + 已有的 pydantic
        ▼
      sqlite3（标准库）、pydantic（已有）
```

**接入规范**：
1. 上层模块**不得**自己 `import sqlite3` 建连接，统一通过 `SQLiteClient` 访问；
2. 表名要加模块前缀避免冲突（例如记忆表叫 `memory_records`、日志表叫 `app_logs`、订单表叫 `biz_orders`）；
3. 建表统一用 `db.ensure_schema(Schema(...))`：
   - 首次启动自动建表 + 建索引；
   - 版本升级加字段时把新字段加进 columns 列表，调用时传 `migrate=True` 自动补列；
4. 数据库文件路径统一通过模块初始化参数传入，不要硬编码路径；
5. 模块内部若有复杂查询（join/子查询/全文搜索），仍可以用 `db.query(sql, params)` 手写，不强制走便捷 CRUD。

**与 memory 模块的关系**：v1.1 提供了建表/迁移能力，但**不主动重构 memory 模块的既有连接/建表代码**（避免破坏）。后续 memory 模块可逐步迁移到 SQLiteClient + Schema，把 `_migrate_schema` 的 PRAGMA/ALTER 样板代码删掉。

---

## 十、测试

所有测试用例都用内存数据库，**不生成任何文件、不依赖外部服务**。

运行：
```powershell
# Windows PowerShell
python test\test_database.py
# 或
pytest test/test_database.py -v
```

38 个用例覆盖：
- 连接：内存连接、文件连接、关闭后自动重连（3）
- 基础执行：execute/executemany/execute 原生建表、空 insert、非法 SQL（4）
- 查询：select 全量 / where 等值 / select_one / limit / 原生 query（5）
- 更新/删除：正常路径 + 缺 where 抛错（4）
- 事务：提交 / 异常回滚（2）
- **建表 DDL（本次新增核心）**：dict 方式 / list[Column] / Schema+索引 / IF NOT EXISTS 幂等 / ensure_table 幂等 / migrate 补列 / migrate=False 不改列 / table_exists / drop_table / VARCHAR/BOOLEAN/TIMESTAMP 类型 / 唯一索引（10）
- **安全与校验（本次新增）**：非法标识符参数化拦截（6 组）、columns 类型错误/空 dict、Schema 表名不一致（3）
- 向后兼容：原生 `CREATE TABLE` 仍可用（1）

**关键回归断言**：
- `test_raw_create_table_still_works` 锁死「原有 db.execute 手写建表 SQL 必须继续可用」，不破坏任何既存业务模块；
- `test_transaction_rollback_on_error` 锁死「事务块里任何一条写失败，之前的插入也必须回滚」；
- `test_ensure_schema_migrate_adds_missing_columns` 锁死「migrate 补列」语义，老库升级不会丢数据。

全绿基线：**38 passed**（在 Python 3.10 / 3.13 + pytest 上实测）。

---

## 十一、跨平台注意事项（Windows 专项）

1. **换行符**：`.py` / `.md` / `.txt` / `.bat` 用 CRLF（`.gitattributes` 已声明 `text eol=crlf`，Windows 检出自动转换；LF 入库，跨平台无假 diff）；`.sh` 必须 LF。
2. **路径反斜杠**：传给 `SQLiteClient("D:\\path\\to\\app.db")` 的路径用原始字符串 `r"D:\path\to\app.db"` 或双反斜杠，避免 `\a` `\n` 转义；推荐用 `pathlib.Path` 拼路径。
3. **WAL 文件**：文件库在 Windows 上同样会生成 `app.db-wal` / `app.db-shm` 两个临时文件，正常退出时会自动合并；杀进程/崩溃可能留着，下次打开仍可正常读写。
4. **文件占用**：SQLite 在 Windows 上默认是文件锁，两个进程同时写同一个 `.db` 会等锁（受 `timeout` 参数控制），超过就抛 `database is locked`；多进程场景下把 `timeout` 调大（如 15 秒）或统一走一个进程做写。
5. **中文路径**：Python 3.10+ 在 Windows 上对中文路径支持已无问题，直接传即可；遇到编码问题设环境变量 `PYTHONUTF8=1`。
6. **脚本运行**：项目 `test/` 目录与标准库同名，必须用 `python test\test_database.py` 直接执行，**不要**用 `python -m test.test_database`（会 import 到标准库 test）。

---

## 十二、后续演进（对应演进路线图，本版不做）

- **MySQL/Postgres 后端**：按需求扩展，引入 `pymysql`/`psycopg2-binary` 作为可选 extras（`pip install fast_agents[mysql]`），默认不装；继承 `DatabaseClientBase`，上层代码零修改切换。
- **完整迁移工具**：当前 migrate 只支持 ADD COLUMN；后续加"列存在则跳过 + 索引幂等 + 默认值回填"三段式，不做 drop/rename（避免误删数据）。
- **查询构造器**：基于 `query.py` 已有的 Query/Filter 模型，支持多条件 AND/OR、分页、聚合，复杂查询时不用手写长 SQL。
- **批量 insert 的 upsert**：SQLite 3.24+ 支持 `INSERT ... ON CONFLICT DO UPDATE`，封装成 `upsert` 便捷方法。
- **全文搜索（FTS5）**：基于 SQLite FTS5 虚拟表，给 RAG / 日志模块做 keyword 检索打底。
- **连接池（多线程/异步场景）**：真有并发需求再加。

---

## 十三、参考

- [Python sqlite3 官方文档](https://docs.python.org/3/library/sqlite3.html) —— 行工厂、事务、WAL、ALTER TABLE 限制等行为的权威依据；
- [SQLite CREATE TABLE 文档](https://www.sqlite.org/lang_createtable.html) —— 类型亲和性、PRIMARY KEY、NOT NULL/DEFAULT 语义；
- [SQLite ALTER TABLE 文档](https://www.sqlite.org/lang_altertable.html) —— ADD COLUMN 的限制（为什么 migrate 只能加列）；
- [Pydantic v2 文档](https://docs.pydantic.dev/) —— Schema/Column/Index 模型的基础能力。

---

## 十四、更新历史

- **v1.1.0（2026-11-02，本次）**：补齐「创建表」能力。
  1. **新增声明式建表 API**（`create_table` / `ensure_table` / `ensure_schema` / `table_exists` / `drop_table`），支持 dict / list[Column] / Schema 三种写法，全部幂等；
  2. **自动建索引**：列级 `index=True` 自动生成 `idx_{table}_{col}`，Schema.indexes 支持复合/唯一索引，`IF NOT EXISTS` + sqlite_master 查询双保险；
  3. **migrate 补列**：`ensure_schema(migrate=True)` 自动 PRAGMA 查列 + ALTER ADD COLUMN，老库升级一行搞定；
  4. **类型映射**：BOOLEAN→INTEGER(0/1)、VARCHAR→TEXT、TIMESTAMP→TEXT，DEFAULT 值自动转义 SQL 字面量；
  5. **标识符白名单校验**：表/列/索引名以及 order_by 做合法标识符正则校验，防拼接注入；
  6. **让 SQLiteClient 继承 DatabaseClientBase**，基类接口与实现完全对齐；
  7. **对外导出 Schema/Column/ColumnType/Index/Query 等模型**（`from fast_agents.database import Schema, Column`）；
  8. **新增示例** `examples/database/create_table_usage.py`，**测试扩展到 38 项全绿**；原有 20 项 CRUD/事务/异常测试**全部保留并通过**，不破坏任何既有代码。
  9. **零新增依赖**（只复用已有的 pydantic 和标准库 sqlite3）。

- **v1.0.1（2026-11-02）**：移除自定义异常，SQL 与连接错误直接透传 sqlite3 原生异常，参数非法抛内置 ValueError。
- **v1.0.0（2026-11-02）**：初版。`SQLiteClient` 基础客户端，连接管理、execute/query、便捷 CRUD、事务上下文、统一异常；examples/database/basic_usage.py、tests/test_database.py、本文档同步交付。
