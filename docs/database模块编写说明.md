# FastAgents Database 模块编写说明

本文档面向 fast_agents **Database（持久化层）**系统，参考 **Python DB-API 2.0（PEP 249）**、**SQLAlchemy Core 的 Table/MetaData 抽象**、**SQLModel 的声明式模型**、**Peewee 的轻量 Schema** 等方案的共性设计，给出本模块的完整说明、源码解析与自定义教程。同时对齐 fast_agents 已交付的 tools / memory / mcp / skills / context 模块文档骨架，做到「翻开任何一份模块说明，都知道在哪一节能找到什么」。

> **版本**：v1.0.0 · 2026-11-03 · 按统一模块文档骨架重写初版（以当前代码为基线，零代码改动）
> **适用版本**：`fast_agents.database`（对应提交 `bd94633 添加创建table`）
> **配套代码**：`fast_agents/database/`（5 个源码文件）+ `test/test_database.py`（33 项用例，全绿）+ `examples/database/`（2 个可运行示例）
> **运行环境**：Python 3.10+（仓库 `.python-version` = 3.10），Windows / macOS / Linux
> **依赖**：零新增依赖，仅标准库 `sqlite3` + 全框架已有的 `pydantic>=2.14.1`；无 SQLAlchemy / Alembic / peewee 等 ORM
> **本文只写「已经实现的东西」**。「还没做但打算做」的部分放在第十二章，并与 `docs/演进路线图.md` 的 J 模块独立编号对齐。

---

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：Database 解决什么问题](#二先想清楚database-解决什么问题)
- [三、主流框架数据库层实现对比](#三主流框架数据库层实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：三种建表写法与幂等迁移](#五核心机制三种建表写法与幂等迁移)
- [六、核心数据模型](#六核心数据模型)
- [七、逐个文件讲解（源码导读）](#七逐个文件讲解源码导读)
- [八、快速上手](#八快速上手)
- [九、API 速查](#九api-速查)
- [十、设计决策与常见坑](#十设计决策与常见坑)
- [十一、测试与验证](#十一测试与验证)
- [十二、下一步演进路线](#十二下一步演进路线)
- [十三、参考](#十三参考)
- [附录：发布前自检清单](#附录发布前自检清单)

---

## 一、更新历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| **1.0.0** | 2026-11-03 | **按统一模块文档骨架重写初版**：此前仓库内 `docs/database模块编写说明.md` 未沿用 `xx模块编写说明模版.md` v2.0 规定的 13 章 + 附录骨架，与其他五份模块文档（工具 / 记忆 / Skill / MCP / context）结构不一致；本次以提交 `bd94633 添加创建table` 后的代码为基线，按模板补齐开篇定位、目录、对比表、数据流图、数据模型字段表、逐文件源码导读、API 速查、设计决策、常见坑、跨平台注意事项、测试清单、演进路线与参考等全部章节，**代码、示例、测试、依赖零改动**，仅重写文档本身。 | — |

> 后续版本迭代更新时在此表追加，保持「使用者可见的变更」倒序排列；内部重构不在此记录。

---

## 二、先想清楚：Database 解决什么问题

### 2.1 一句话概括

**Database 模块是 fast_agents 的持久化地基——给记忆、轨迹、用户画像、技能缓存等上层模块提供「零依赖、声明式、幂等」的关系型存储能力，上层不需要自己拼 `CREATE TABLE`。**

Agent 框架里要落盘的东西比想象中多：长期记忆、会话快照、用户画像、Trace 日志、技能资源缓存、MCP 工具清单缓存……这些数据目前在 fast_agents 里**都是 SQLite 单文件**，但上层模块（`memory/` 等）此前在自己手写 `CREATE TABLE IF NOT EXISTS` + `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` 的样板 SQL，每加一张表就要重复一遍同样的「查表是否存在、缺列就补、建索引」套路。Database 模块的职责就是把这层共性抽出来。

### 2.2 抽象分层

| 层级 | 生命周期 | 典型内容 | 对应代码 | 依赖方向 |
|---|---|---|---|---|
| **门面层（出口）** | 进程内 | 统一对外导入入口 | `fast_agents/database/__init__.py` | 只 re-export，不含逻辑 |
| **抽象层（契约）** | 编译期 | 后端必须实现的最小接口集 | `database_client_base.py`（`DatabaseClientBase`） | 不依赖具体后端；上层模块只引用它 |
| **声明模型层** | 编译期 | 列/索引/表结构/SQL 片段解析、标识符白名单 | `schema.py`（`ColumnType/Column/Index/Schema` + `check_ident/quote_ident/parse_*`） | **不依赖任何数据库连接**，纯函数可单测 |
| **查询模型层** | 运行期 | 过滤/排序/分页的轻量对象 | `query.py`（`Query/Filter/FilterOp/SortOrder`） | 不依赖具体后端，目前 SQLite 客户端尚未用它做谓词下推 |
| **实现层（后端）** | 运行期 | 具体数据库驱动封装 | `sqlite_client.py`（`SQLiteClient`） | 依赖抽象层 + schema 解析函数 + 标准库 `sqlite3` |
| **上层业务模块** | 运行期 | 记忆落盘 / Trace 落盘 / 画像落盘 | `memory/sqlite_store.py`、后续 G1 `TraceLogger` 等 | 只依赖抽象层 + 声明模型，不直接 `import sqlite3` |

### 2.3 关键设计原则（全篇代码都在落实它们）

1. **零重型依赖**：只用标准库 `sqlite3` + 全框架已有的 `pydantic`，不引入 SQLAlchemy / SQLModel / Alembic / peewee 任何 ORM。这是 README「零重型依赖」招牌的延续（参考「记忆模块只用标准库」那条设计取舍）。
2. **声明式而非字符串**：列、索引、表结构用 pydantic 模型表达（`Column/Index/Schema`），拼 SQL 的逻辑统一收口在 `schema.py` 的纯函数里，上层**不手写** `CREATE TABLE`，避免 SQL 注入、列名拼错、类型不统一。
3. **默认幂等、默认安全**：`ensure_table / ensure_schema` 默认 `IF NOT EXISTS`；删除必须带 `where` 防止误删全表；表名/列名/索引名走 `check_ident` 白名单，从源头堵标识符注入；`delete` 不传 `where` 直接 `ValueError`。
4. **最小惊讶**：`BOOLEAN→INTEGER(0/1)`、`VARCHAR→TEXT`、`TIMESTAMP→TEXT(ISO 字符串)` 的类型映射写死在 `TYPE_TO_SQLITE`；`select/insert/update/delete` 全部返回字典 / 整数，不用 cursor / Row 对象给上层添麻烦。
5. **迁移能力克制**：`ensure_schema(..., migrate=True)` 只做 SQLite 原生支持的 `ADD COLUMN` 补列，不做删列/改类型/改主键/数据迁移——这是 SQLite 能力边界的诚实反映，不假装自己是完整 ORM；更复杂的迁移交给上层显式脚本。

---

## 三、主流框架数据库层实现对比

### 3.1 对比总表

| 方案 | 定位 | 依赖 | 建表方式 | 迁移 | 优点 | 缺点 |
|---|---|---|---|---|---|---|
| **裸 `sqlite3`（标准库）** | Python 内置驱动 | 无 | 手写 SQL | 无 | 零依赖、最灵活 | 每个模块都要手写 `CREATE TABLE` / `PRAGMA table_info` / 类型映射，样板代码爆炸、容易漏索引 |
| **SQLAlchemy Core** | SQL 工具包 | SQLAlchemy（~10MB） | `Table(..., Column(...))` MetaData.create_all | Alembic（独立） | 方言抽象最完整、生态最大 | 对轻量 Agent 框架过重；MetaData 反射与连接池是不必要的复杂度 |
| **SQLModel / Peewee** | 轻量 ORM | sqlmodel / peewee | 继承基类声明 | 需自己写或用 playhouse.migrate | 模型和表一体、上手快 | 绑定类继承，不利于「声明模型是纯数据」的分层；迁移依然不完整 |
| **LangChain `SQLChatMessageHistory`** | 单一用途存储 | langchain + sqlalchemy | 模块内部手写 SQL | 无 | 开箱即用 | 每个存储类各写各的 SQL，没有统一抽象；想加新后端就得重写一份 |
| **CrewAI / Mem0** | 内置存储 | 自带（多为 SQLite + 向量库） | 模块内手写或 ORM | 部分支持 | 业务贴合 | 没有把"建表/迁移"独立成公共模块，记忆/Trace/画像各自为战 |
| **fast_agents（本模块）** | 轻量持久化地基 | **零新依赖**（标准库 sqlite3 + pydantic） | **三种写法并存**：dict / list[Column] / Schema（`ensure_table`/`ensure_schema`） | **`migrate=True` 补列**（仅 ADD COLUMN） | 零依赖、声明式、幂等、三种写法按场景选；抽象基类预留多后端位 | 目前只有 SQLite 一个后端；迁移只做补列、不支持删/改；`Query` 对象尚未在 CRUD 里谓词下推 |

### 3.2 关键差异：相同 / 增加 / 缺少

| 维度 | 与"裸 sqlite3"相同 | 相对裸 sqlite3 增加 | 目前缺少 |
|---|---|---|---|
| 驱动能力 | 连接 / 事务 / 参数化查询 / execute/executemany | 字典结果、WAL、外键默认开启、超时参数 | 异步 `aiosqlite` 连接、连接池 |
| 建表 | 仍能手写 `db.execute("CREATE TABLE ...")`，**完全向后兼容** | dict / list[Column] / Schema 三种声明式写法；自动建索引；`IF NOT EXISTS` 默认幂等 | 外键声明、CHECK 约束、表注释（SQLite 本身不支持） |
| 迁移 | — | `ensure_schema(schema, migrate=True)`：PRAGMA 查列 → 自动 ADD COLUMN 缺列 | 删列 / 改类型 / 改主键 / 重命名 / 数据迁移（SQLite 限制） |
| 查询 | — | where 等值匹配、order_by 白名单校验、limit | `Query/Filter/FilterOp` 谓词下推（>、<、LIKE、IN、IS NULL 等模型已定义但 SQLiteClient 尚未接入）、JOIN、聚合 |
| 后端扩展 | — | `DatabaseClientBase` 抽象基类定义了最小接口契约 | MySQL / Postgres / DuckDB 后端实现 |
| 安全 | — | 标识符白名单 `check_ident` 防表名/列名注入；delete 必须带 where | 列级权限、审计日志（留给 G1 TraceLogger） |

一句话收口：**本模块 = 裸 sqlite3 的零依赖 + SQLAlchemy Core 声明式建表的最小子集 + 向 ORM 迁移留下的接口，但刻意不做完整 ORM 和迁移器**——这和 README 里"基于 OpenAI 原生 API、零重型依赖"的整体调性一致。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/database/                 ← 单数目录（对齐 tools/memory/skills/mcp 的命名约定）
├── __init__.py                       # 模块出口：统一导出 SQLiteClient / DatabaseClientBase / Schema / Column / ColumnType / Index / Query / Filter / FilterOp / SortOrder
├── database_client_base.py           # 抽象基类 DatabaseClientBase：connect/close + DDL（建/查/删表）+ DML（CRUD）契约
├── schema.py                         # 声明模型 ColumnType/Column/Index/Schema + 标识符校验 + Column/Schema→SQL 片段纯函数
├── query.py                          # 查询模型 SortOrder/FilterOp/Filter/Query（轻量过滤/排序/分页，供未来后端扩展）
└── sqlite_client.py                  # SQLiteClient：唯一实现，封装 sqlite3 连接、事务、字典结果、幂等建表、便捷 CRUD
```

配套但不放在 `fast_agents/database/` 下：

- `examples/database/basic_usage.py` —— CRUD + 事务最小闭环示例；
- `examples/database/create_table_usage.py` —— 三种建表写法 + migrate + drop 完整示例；
- `test/test_database.py` —— 33 项单测（纯内存数据库，无外部依赖）。

### 4.2 分层

```text
┌───────────────────────────────────────────────────────────────────┐
│ 上层业务模块（memory / 未来的 trace / profile 缓存 / skill 缓存）     │
│ 只 import DatabaseClientBase + Schema/Column，不直接 import sqlite3  │
└───────────────────────────┬───────────────────────────────────────┘
                            │ 依赖抽象，不依赖具体后端
┌───────────────────────────▼───────────────────────────────────────┐
│ DatabaseClientBase（database_client_base.py）—— 客户端契约            │
│   连接：connect / close                                             │
│   DDL：table_exists / create_table / ensure_table /                 │
│        ensure_schema / drop_table                                   │
│   DML：insert / select / select_one / update / delete               │
└─────────┬─────────────────────────────────┬─────────────────────────┘
          │实现（目前唯一）                    │ 共用纯函数
┌─────────▼─────────────────────┐   ┌────────▼──────────────────────┐
│ SQLiteClient（sqlite_client.py）│   │ schema.py（纯数据 + 纯函数）    │
│  - 连接管理 + 行工厂 + WAL/外键  │   │ ColumnType/Column/Index/Schema │
│  - execute/executemany/query   │   │ check_ident / quote_ident      │
│  - 事务上下文（with db:）       │   │ TYPE_TO_SQLITE / sql_literal   │
│  - create_table 三种写法分派    │   │ column_sql / column_index_stmt │
│  - ensure_schema + migrate     │   │ index_stmt / parse_column_list │
│  - insert/select/update/delete │   │ parse_schema                   │
└───────────────────────────────┘   └───────────────────────────────┘
          │                                  │
          └───────────────┬──────────────────┘
                          │使用
                ┌─────────▼─────────┐
                │ query.py（数据模型）│  SortOrder/FilterOp/Filter/Query
                │ 当前：定义即导出     │  （SQLiteClient 暂未谓词下推）
                └───────────────────┘
                          │
                ┌─────────▼─────────┐
                │ 标准库 sqlite3     │
                └───────────────────┘
```

### 4.3 数据流 / 调用链路

**主链路 1：声明式建表（Schema / list[Column] / dict）**

```text
上层调用 db.ensure_table(table, columns) 或 db.ensure_schema(schema, migrate=?)
        │
        ▼
SQLiteClient.create_table 按 columns 类型分派
   ├─ dict[str,str]   → 逐列 check_ident + quote_ident → _create_table_raw → execute(CREATE TABLE ...)
   ├─ list[Column]    → schema.parse_column_list → (col_defs, index_stmts)
   │                       └─ _create_table_raw + 逐条 execute(CREATE INDEX IF NOT EXISTS ...)
   └─ Schema          → _create_table_from_schema(schema)
                            ├─ schema.parse_schema → (col_defs, index_stmts)
                            ├─ 表不存在 → _create_table_raw → 建索引
                            └─ 表已存在 + migrate=True → PRAGMA table_info 拿 existing_cols
                                                           └─ 逐缺列 ALTER TABLE ADD COLUMN ...
                                                              （列名非法/类型不支持由 sqlite3 直接抛错）
                            └─ 索引始终幂等执行（CREATE INDEX IF NOT EXISTS）
```

**主链路 2：便捷 CRUD**

```text
db.insert(t, data)           → 列名 quote_ident + 占位符 :key     → execute(INSERT ...) → lastrowid
db.select(t, where, order_by, limit)
                             → 列名/order_by 白名单 + 参数化     → query(SELECT ...) → list[dict]
db.select_one(t, where, ...) → select(..., limit=1)              → rows[0] or None
db.update(t, data, where)    → set/wher 子句参数化（set_/w_ 前缀）→ execute(UPDATE ...) → rowcount
db.delete(t, where)          → where 必填，参数化                 → execute(DELETE ...) → rowcount
```

**主链路 3：事务**

```text
with db:                      → __enter__ 执行 BEGIN，置 _in_transaction=True
    db.insert(...)            → execute 里检测到 _in_transaction → 不 commit
    db.update(...)            → 同上
    # 抛异常 → __exit__ rollback + 清 _in_transaction
    # 正常   → __exit__ commit   + 清 _in_transaction
```

### 4.4 与 ReActAgent 及上层模块的接入点 / 职责边界

当前（v1.0.0 基线）database 模块**还没有直接挂进 ReActAgent**，它是作为 `memory/` 模块的底层依赖被间接使用：

- `memory/sqlite_store.py`、`memory/session_store.py`、`memory/profile_store.py` 等类**目前仍在各自手写 `CREATE TABLE`**（历史遗留），未来会逐步迁移到 `db.ensure_schema(...)`；但它们内部已经用 sqlite3 连接做持久化，迁移时只改建表语句那一段，上层 API 不变。
- 未来 **G1 TraceLogger**（见《演进路线图》）会直接依赖 `SQLiteClient` 落 jsonl/结构化 trace；这也是 Database 模块独立出来的直接动因——避免 trace/记忆/画像各写各的建表 SQL。
- **单一数据源约定**（对齐记忆模块的精神）：Agent 跨轮持久化状态只有两条路——要么走 `MemoryManager`（短期 buffer + 长期 store），要么走各业务模块自己持有的 `DatabaseClientBase` 实例；**禁止上层模块绕开抽象基类直接 `import sqlite3`**，否则后续换 Postgres/MySQL 时会爆雷。
- 与 `tools/builtin/` 的边界：`FileReadTool / FileWriteTool` 等文件工具处理的是文件系统（不是数据库）；数据库操作不通过 tool 暴露给 LLM，避免模型误删数据——**DatabaseClient 是框架内部基础设施，不做成 LLM 可调用工具**（这点与 MCP/tool 模块明确划界）。

---

## 五、核心机制：三种建表写法与幂等迁移

本章是 database 模块的弹性节（对应模板 §5），讲清楚「建表到底有几种写法、每种适合什么场景、幂等和迁移怎么保证」。

### 5.1 三种建表写法对比

| 维度 | 写法 A：`dict[str,str]` | 写法 B：`list[Column]` | 写法 C：`Schema` 对象 |
|---|---|---|---|
| 入口 | `ensure_table(t, {...})` | `ensure_table(t, [Column(...), ...])` | `ensure_schema(Schema(...), migrate=?)` |
| 代码量 | 最少，一行一个列 | 中等，需要 import Column/ColumnType | 最多，需构造 Schema + Index |
| 类型安全 | ❌ 列类型是字符串，写错不会提前报错 | ✅ ColumnType 枚举，IDE 补全 | ✅ 同 B，并多 indexes 字段校验 |
| 默认值/可空/唯一 | 自己在字符串里写 `NOT NULL DEFAULT 0` | ✅ 字段语义化：`nullable=False, default=0, unique=True` | ✅ 同 B |
| 单列索引 | ❌ 需自己手写 `CREATE INDEX` | ✅ `Column(..., index=True)` 自动建 `idx_{table}_{col}` | ✅ 同 B |
| 复合索引/唯一索引 | ❌ 完全手写 | ❌ 不支持（只能建完表再 `execute`） | ✅ `indexes=[Index(name=..., columns=[...], unique=...)]` |
| 迁移补列 | ❌ 不支持 | ❌ 不支持 | ✅ `migrate=True` 自动 PRAGMA + ADD COLUMN |
| 适合场景 | 写原型、临时脚本、一次性 demo | 业务表里常驻的简单表（无复合索引） | **工程化推荐**：生产模块的正式表、需要迭代加字段的表 |

**选择建议（必须给结论，不能两边都夸）**：

- **写 demo / 验证想法**：用写法 A，最快；
- **业务表且字段稳定、没有复合索引**：用写法 B，有枚举校验和索引自动化，比 dict 安全；
- **memory / trace / profile 这种会迭代加字段、需要复合索引的正式表**：**一律用写法 C + `migrate=True`**，上线后版本迭代靠 `ensure_schema` 自动补列，不用手写迁移脚本。

### 5.2 幂等性怎么保证

| 操作 | 幂等机制 |
|---|---|
| 建表 | 全部走 `CREATE TABLE IF NOT EXISTS`（`create_table(if_not_exists=True)` 是默认；`ensure_table` 等价于这个默认） |
| 列级索引 | `CREATE INDEX IF NOT EXISTS` + 唯一命名 `idx_{table}_{col}`；重复调用不会报错也不会重复建 |
| 显式索引（Schema.indexes）| 同上 `CREATE [UNIQUE] INDEX IF NOT EXISTS`，命名由调用方保证唯一 |
| 补列（migrate） | 先 `PRAGMA table_info(table)` 拿现有列集合，缺的才 `ALTER TABLE ADD COLUMN`；已存在列自动跳过 |
| insert | 不保证幂等（业务层自己用唯一约束 + try/except 处理重复） |
| update/delete | where 条件决定，不是本模块职责 |

### 5.3 migrate 的能力边界（诚实列缺项）

`ensure_schema(schema, migrate=True)` **只做 SQLite 原生支持的事**：

- ✅ 补新增列（`ALTER TABLE ADD COLUMN`）；
- ❌ 不删列（SQLite 3.35.0 之前不支持 DROP COLUMN，且即使新版本，删列有丢数据风险，模块不代为执行）；
- ❌ 不改列类型 / 不改列名 / 不加主键 / 不加唯一约束（这些都需要重建表，超出"安全自动迁移"边界）；
- ❌ 不迁移数据（例如把布尔 0/1 改成字符串 "yes"/"no"）。

遇到上述复杂迁移，**显式写迁移脚本**（用 `db.executescript()`），不要指望模块默默做掉——这和「最小惊讶」原则一致：宁可让你看到要写迁移，也不要静默改你的数据。

---

## 六、核心数据模型

所有模型都在 `schema.py` / `query.py`，用 pydantic `BaseModel` 定义；枚举都继承 `str, Enum`，方便直接打印和序列化。

### 6.1 `ColumnType`（列类型枚举，与 SQLite 方言解耦）

```python
class ColumnType(str, Enum):
    TEXT      = "TEXT"
    INTEGER   = "INTEGER"
    REAL      = "REAL"
    BLOB      = "BLOB"
    BOOLEAN   = "BOOLEAN"   # → SQLite INTEGER，0/1
    VARCHAR   = "VARCHAR"   # → SQLite TEXT（length 仅作文档/校验）
    TIMESTAMP = "TIMESTAMP" # → SQLite TEXT，ISO 字符串（对齐 memory 模块习惯）
```

### 6.2 `Column`（列定义）

```python
class Column(BaseModel):
    name:        str                   # 列名，必须 ^[A-Za-z_][A-Za-z0-9_]*$
    type:        ColumnType            # 列类型
    primary_key: bool        = False   # 是否主键
    nullable:    bool        = True    # 是否允许 NULL（主键列强制非空）
    default:     Optional[Any] = None  # 默认值，按 type 转 SQL 字面量
    unique:      bool        = False   # 是否唯一约束（主键列自动跳过）
    index:       bool        = False   # 是否建普通单列索引 idx_{table}_{col}
    length:      Optional[int] = None  # VARCHAR 长度（SQLite 无强校验，仅文档语义）
```

| 字段 | 类型 | 说明 | 必填 |
|---|---|---|---|
| `name` | str | 列名，走 `check_ident` 白名单 | ✅ |
| `type` | ColumnType | 列类型枚举，映射规则见 §5.1 的 `TYPE_TO_SQLITE` | ✅ |
| `primary_key` | bool | 是否主键；声明后 nullable/unique 自动忽略，不自动加 AUTOINCREMENT | ❌ 默认 False |
| `nullable` | bool | 是否允许 NULL；主键列不受此字段影响 | ❌ 默认 True |
| `default` | Any | 默认值；bool→0/1、int/float→数字、None→NULL、str→单引号转义 | ❌ 默认 None |
| `unique` | bool | 是否 UNIQUE；主键列不再重复加 | ❌ 默认 False |
| `index` | bool | 是否建单列普通索引；主键/UNIQUE 列 SQLite 已自动建索引，会跳过避免冗余 | ❌ 默认 False |
| `length` | int\|None | VARCHAR 长度提示（SQLite 不强制） | ❌ 默认 None |

> **命名对齐提醒**：字段名 `nullable`（不是 `allow_null`）、`default`（不是 `default_value`）、`unique`/`index` 都是 bool 开关——这和 pydantic / SQLAlchemy Column 的常用命名保持一致，不要另起 `is_nullable` / `default_val` 这种变体。

### 6.3 `Index`（索引定义）

```python
class Index(BaseModel):
    name:    str         # 索引名，白名单校验
    columns: list[str]   # 索引列名（单列/多列复合），每个列名都会白名单校验
    unique:  bool = False  # 是否唯一索引
```

### 6.4 `Schema`（完整表结构，工程化推荐入口）

```python
class Schema(BaseModel):
    table_name: str                    # 表名
    columns:    list[Column]           # 列定义列表
    indexes:    Optional[list[Index]] = None  # 显式索引（单列索引用 Column.index=True 即可）
```

### 6.5 查询模型（`query.py`，为未来后端扩展预留）

```python
class SortOrder(str, Enum): ASC = "ASC"; DESC = "DESC"

class FilterOp(str, Enum):
    EQ="="; NEQ="!="; GT=">"; GTE=">="; LT="<"; LTE="<="
    LIKE="LIKE"; IN="IN"; NOT_NULL="IS NOT NULL"; IS_NULL="IS NULL"

class Filter(BaseModel):
    field: str
    op:    FilterOp
    value: Optional[Any] = None  # IS NULL / NOT_NULL 时无需传

class Query(BaseModel):
    table_name:  str
    filters:     Optional[list[Filter]] = None  # AND 关系
    sort_by:     Optional[str] = None
    sort_order:  SortOrder = SortOrder.ASC
    limit:       Optional[int] = None
    offset:      Optional[int] = None
```

> **注意（v1.0.0 基线）**：`Query/Filter/FilterOp` **目前只是数据模型，SQLiteClient 的 select/update/delete 还没有接入它们做谓词下推**。`select` 当前只支持 dict 等值匹配 + order_by + limit。要使用 >、<、LIKE、IN 等操作符，直接用 `db.query(sql, params)` 手写 SQL；Query 模型是为后续 MySQL/Postgres 后端和谓词下推预留的，不是给现在用的。不要在文档里把它说成"已支持"。

### 6.6 `DatabaseClientBase`（抽象基类，子类必须实现）

```python
class DatabaseClientBase(ABC):
    # 连接
    @abstractmethod def connect(self) -> None: ...
    @abstractmethod def close(self) -> None: ...
    # DDL
    @abstractmethod def table_exists(self, table: str) -> bool: ...
    @abstractmethod def create_table(self, table, columns, if_not_exists=True) -> None: ...
    @abstractmethod def ensure_table(self, table, columns) -> None: ...
    @abstractmethod def ensure_schema(self, schema: Schema, migrate: bool = False) -> None: ...
    @abstractmethod def drop_table(self, table, if_exists=True) -> None: ...
    # DML
    @abstractmethod def insert(self, table, data) -> int: ...
    @abstractmethod def select(self, table, where=None, limit=None) -> list[dict]: ...
    @abstractmethod def select_one(self, table, where) -> dict | None: ...
    @abstractmethod def update(self, table, data, where) -> int: ...
    @abstractmethod def delete(self, table, where) -> int: ...
```

子类**必须实现全部 13 个方法**，上层（memory / trace / profile）只依赖这个抽象，不直接 import sqlite3/psycopg2。SQLiteClient 已经全部实现。

---

## 七、逐个文件讲解（源码导读）

每个文件固定三段：**它负责什么 → 关键实现 → 为什么这么写**。

### 7.1 `__init__.py` —— 模块出口，统一 re-export

```python
from .database_client_base import DatabaseClientBase
from .query import Filter, FilterOp, Query, SortOrder
from .schema import Column, ColumnType, Index, Schema
from .sqlite_client import SQLiteClient

__all__ = [
    "SQLiteClient", "DatabaseClientBase",
    "Schema", "Column", "ColumnType", "Index",
    "Query", "Filter", "FilterOp", "SortOrder",
]
```

- **负责什么**：给上层一个稳定的 import 入口（`from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index`），避免上层直接 import 子文件路径，将来内部文件调整不影响调用方。
- **为什么这么写**：`__all__` 显式列出 10 个符号，既避免 `from database import *` 污染，也是给阅读者一张"公共 API 清单"。对齐 `fast_agents/tools/__init__.py`、`fast_agents/memory/__init__.py` 的 re-export 模式。
- **你要改的时候最该看的地方**：新增公共类后记得在这里加一行导出 + `__all__` 一项，否则上层 `from fast_agents.database import Xxx` 会 ImportError。

### 7.2 `database_client_base.py` —— 客户端抽象基类

```python
class DatabaseClientBase(ABC):
    @abstractmethod
    def create_table(self, table, columns, if_not_exists=True):
        """columns 支持三种形式：dict[str,str] / list[Column] / Schema"""
```

- **负责什么**：定义所有后端（SQLite / 未来 MySQL / Postgres）必须实现的最小接口集——连接（2）+ DDL（5）+ DML（5）共 12 个方法（`__enter__/__exit__` 没放抽象方法里，因为不是所有后端都必须支持上下文事务）。
- **关键实现**：docstring 里写死了 `columns` 三种形式的语义约束，这是一份"接口契约文档"，子类实现必须同时支持这三种。
- **为什么这么写**：抽象基类里不加事务/executemany/execute 等"半可选"方法，避免接口膨胀；上层业务只依赖 12 个方法，将来换后端改一行构造代码即可。**你要扩展新后端时，第一个要实现的就是这个类**，照着 docstring 把 12 个方法都写了，memory/trace 上层代码零改动。

### 7.3 `schema.py` —— 声明模型 + SQL 片段纯函数（整个模块的大脑）

```python
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def check_ident(name: str, kind: str = "标识符") -> None:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"非法{kind}: {name!r}（只允许字母/数字/下划线，且不以数字开头）")

TYPE_TO_SQLITE: dict[ColumnType, str] = {
    ColumnType.BOOLEAN:   "INTEGER",    # 0/1
    ColumnType.VARCHAR:   "TEXT",
    ColumnType.TIMESTAMP: "TEXT",       # ISO 字符串
    # TEXT/INTEGER/REAL/BLOB 直通
}
```

- **负责什么**：
  - 数据模型：`ColumnType / Column / Index / Schema` 四个 pydantic 类；
  - 安全：`check_ident / quote_ident` 标识符白名单 + 双引号引用；
  - SQL 生成：`sql_literal / column_sql / column_index_stmt / index_stmt / parse_column_list / parse_schema` 六个纯函数——**输入模型、输出 SQL 字符串列表，不依赖任何数据库连接**。
- **关键实现**：
  - `column_sql(col)` 按顺序拼 `列名 类型 [PRIMARY KEY] [NOT NULL] [UNIQUE] [DEFAULT x]`；主键列不再重复加 NOT NULL/UNIQUE，避免冗余；
  - `column_index_stmt` 会跳过主键列和 UNIQUE 列（SQLite 已自动建索引），避免重复建冗余索引；
  - `parse_schema` = `parse_column_list` + 显式 indexes，索引语句统一追加到列表末尾，顺序固定；
  - **纯函数**：这一层可以脱离数据库连接独立单测（SQL 字符串拼接正确性不需要真连库）。
- **为什么这么写**：把"拼 SQL"这件事从客户端里抽成纯函数是整个模块最值钱的设计——将来接 Postgres/MySQL 后端，只要替换 `TYPE_TO_XXX` 和 `column_sql` 的方言差异（例如 Postgres 的 BOOLEAN 是原生、SERIAL 代替 AUTOINCREMENT），模型和解析管线可以大量复用。**你要加 SQLite 不支持的新列类型（JSON/ARRAY）时，先在这一层扩展 TYPE_TO_SQLITE 映射，不要直接去 sqlite_client 里拼字符串。**
- **踩过的坑**：之前想过给 INTEGER PRIMARY KEY 自动加 AUTOINCREMENT，但 SQLite 的 `INTEGER PRIMARY KEY` 已经是 ROWID 别名，自动加反而改变默认行为（会阻止 ROWID 回填），所以代码里注释明确"保守起见不自动追加"。

### 7.4 `query.py` —— 轻量查询对象（预留，当前未接入 CRUD）

```python
class Query(BaseModel):
    table_name: str
    filters: Optional[list[Filter]] = None
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.ASC
    limit: Optional[int] = None
    offset: Optional[int] = None
```

- **负责什么**：为未来多后端统一的"过滤/排序/分页"表达打基础，避免每个后端各写一套查询 AST。
- **关键实现**：`FilterOp` 枚举覆盖 =/!=/>/>=/</<=/LIKE/IN/IS NULL/IS NOT NULL 10 种常用操作符；`Filter.value` 对 IS NULL/NOT_NULL 允许为空。
- **为什么这么写**：现在 SQLiteClient 的 select 只支持 dict 等值 + order_by + limit，不够用的时候调用方可以直接写 SQL 走 `db.query()`；但当后面要加 MySQL 后端并统一谓词下推时，这个模型就是现成的 AST——提前放好，不要等到加第二个后端再回头设计。**当前不要在业务代码里假设 SQLiteClient.select 能读懂 Filter 对象**——会得到不生效的静默 bug。

### 7.5 `sqlite_client.py` —— SQLite 客户端（唯一实现）

```python
class SQLiteClient(DatabaseClientBase):
    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0):
        self.db_path = db_path
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._in_transaction = False
        self._connect()
```

- **负责什么**：把标准库 sqlite3 封装成上层好用的客户端——字典行结果、自动重连、WAL/外键默认开启、事务上下文、声明式建表三写法分派、幂等 migrate、便捷 CRUD。
- **关键实现**（按方法块）：
  - **连接** `_connect()`：`sqlite3.connect(..., detect_types=PARSE_DECLTYPES)` + `row_factory=Row`（返回字典）+ `PRAGMA foreign_keys=ON`；非内存库自动开 `PRAGMA journal_mode=WAL`（并发读写更友好，呼应 B9 待办）；连接失败统一抛 `ValueError(f"连接 SQLite 失败: {e}")`。
  - **基础执行**：`execute/executemany/executescript/query` 四个方法是最底层通道，自动管理事务边界（在 `with db:` 事务内不 commit，由 `__exit__` 统一提交/回滚；不在事务内则单次语句自动 commit，失败自动 rollback）。SQLite 异常统一包成 `ValueError` 抛出，让上层不用 `import sqlite3` 就能 catch。
  - **DDL**：`table_exists` 查 `sqlite_master`；`create_table` 按 `isinstance(columns, Schema/list/dict)` 分派；`ensure_table` 就是 `create_table(if_not_exists=True)`；`ensure_schema` 先走 `table_exists` 判断，不存在就 `_create_table_from_schema`，已存在且 `migrate=True` 则 PRAGMA 拿现有列集合逐列 ALTER ADD；索引语句始终幂等执行；`drop_table` 默认 IF EXISTS。
  - **CRUD**：insert 返回 lastrowid；select 的 where 用命名占位符 `:w_k` 防冲突，order_by 走 `^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$` 白名单校验；update 用 `:set_k / :w_k` 前缀区分 set 和 where 参数避免冲突；delete **必须传 where**，否则抛 ValueError（防止 `DELETE FROM users` 清空全表的低级错误）。
  - **事务**：`__enter__` 显式 `BEGIN` 置 `_in_transaction=True`；`__exit__` 根据是否异常 commit/rollback，finally 里复位标志；`__del__` 尝试 close，不抛异常。
- **为什么这么写**：
  - **row_factory=Row + 手动 dict(row)**：标准库 sqlite3 的 Row 是tuple-like，直接给上层用不方便；转 dict 让上层 `row["name"]` 访问，对齐记忆模块现有数据结构。
  - **dict 列定义也要 check_ident**：dict 写法最宽松，更要做表名/列名白名单，否则一行 `"name; DROP TABLE x;--": "TEXT"` 就能注入（虽然参数化只能防值注入、防不了标识符注入，这也是为什么必须做 check_ident）。
  - **migrate 让 sqlite3 自己报错**：不支持的场景（如 ADD COLUMN NOT NULL 无默认值）直接让 sqlite3 抛错向上冒泡，比静默跳过或瞎猜默认值安全——调用方看到报错就知道该写显式迁移脚本。
  - **execute 包成 ValueError**：不让上层 import sqlite3 去 catch sqlite3.Error，贯彻"上层不依赖驱动"的分层原则；所有数据库错误都是 ValueError 的子类消息里能看到原始 SQL，便于排查。
- **你要改的时候最该看的地方**：加新的便捷方法（如 `count / exists / upsert`）时，先想清楚是不是能通过组合 `query/execute` 实现，如果只是个别业务场景需要，加在业务层不要加到公共客户端里；加新后端时，直接对照 `DatabaseClientBase` 新增一个 `xxx_client.py`，不要在 SQLiteClient 里塞方言判断。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents

# 依赖（database 模块本身零新增依赖，只要装好仓库 requirements 即可）
pip install -r requirements.txt

# 跑测试（33/33 全过才算装好）
python test\test_database.py

# 跑两个示例（全部用内存/临时库，跑完不留脏文件）
python examples\database\basic_usage.py
python examples\database\create_table_usage.py
```

> **命令必须按 Windows PowerShell 写**：项目里的 `test/` 目录和标准库 `test` 包同名，所以用 `python test\xxx.py` **直接执行**，不要用 `python -m test.xxx`（那会跑成标准库测试）。

### 8.2 最小可运行示例（零依赖、不需要 API Key、全部内存库）

```python
from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index

db = SQLiteClient(":memory:")                     # 内存数据库，跑完即丢

db.ensure_table(
    "users",
    [
        Column(name="id",   type=ColumnType.INTEGER, primary_key=True),
        Column(name="name", type=ColumnType.TEXT,    nullable=False),
        Column(name="age",  type=ColumnType.INTEGER, nullable=True),
    ],
)

alice_id = db.insert("users", {"name": "Alice", "age": 30})
bob_id   = db.insert("users", {"name": "Bob",   "age": 25})

print(db.select("users", order_by="id ASC"))
# [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}]

db.update("users", {"age": 31}, where={"id": alice_id})
print(db.select_one("users", where={"id": alice_id}))
# {'id': 1, 'name': 'Alice', 'age': 31}

db.close()
```

这段代码 < 25 行，不用任何外部服务，跑通就说明 database 模块装好了。

### 8.3 在 ReActAgent 中的接入方式

当前（v1.0.0）ReActAgent **没有直接持有 DatabaseClient**；database 是通过 memory 模块间接被使用的——`SQLiteMemoryStore / SQLiteSessionStore / SQLiteProfileStore` 内部持有 sqlite3 连接，做记忆落盘。

如果你要在自己的工具/技能里用数据库，标准做法是**构造时注入客户端实例**（不要在工具内部自己 `SQLiteClient("xxx.db")`，否则会产生多个连接、锁冲突）：

```python
from fast_agents import LLM, ReActAgent, ToolRegistry
from fast_agents.database import SQLiteClient
from fast_agents.tools import StructuredTool

db = SQLiteClient("data/app.db")
db.ensure_table("notes", {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "content": "TEXT NOT NULL"})

def write_note(content: str) -> str:
    nid = db.insert("notes", {"content": content})
    return f"已保存 note#{nid}"

registry = ToolRegistry()
registry.register_tool(StructuredTool.from_function(write_note))
agent = ReActAgent(name="助手", llm=LLM(), tool_registry=registry)
```

**不建议把数据库操作做成 LLM 可自由调用的万能工具**（给模型 DROP TABLE 权限太危险），只暴露业务语义清晰的窄接口（如 `write_note`）。

### 8.4 典型用法

#### 用法 1：事务（要么全成，要么全回滚）

```python
try:
    with db:
        db.insert("accounts", {"id": 1, "balance": 100})
        db.insert("accounts", {"id": 2, "balance": 0})
        db.update("accounts", {"balance": 90},  where={"id": 1})
        # 如果这里抛异常，上面两条 insert 和 update 会一起回滚
        db.update("accounts", {"balance": 10},  where={"id": 2})
except ValueError as e:
    print("事务失败，已回滚:", e)
```

#### 用法 2：Schema + migrate，版本升级自动补列

```python
# v1：首发版本
db.ensure_schema(Schema(
    table_name="users",
    columns=[
        Column(name="id",   type=ColumnType.INTEGER, primary_key=True),
        Column(name="name", type=ColumnType.TEXT,    nullable=False),
    ],
))

# v2：新增 email 字段，部署到老库上自动 ALTER ADD COLUMN
db.ensure_schema(Schema(
    table_name="users",
    columns=[
        Column(name="id",    type=ColumnType.INTEGER, primary_key=True),
        Column(name="name",  type=ColumnType.TEXT,    nullable=False),
        Column(name="email", type=ColumnType.TEXT,    nullable=True),  # 新列必须 nullable 或有 default
    ],
), migrate=True)
```

#### 用法 3：复合索引 + 唯一约束

```python
db.ensure_schema(Schema(
    table_name="orders",
    columns=[
        Column(name="id",         type=ColumnType.INTEGER, primary_key=True),
        Column(name="user_id",    type=ColumnType.INTEGER, nullable=False, index=True),
        Column(name="order_no",   type=ColumnType.TEXT,    nullable=False, unique=True),
        Column(name="amount",     type=ColumnType.REAL,    nullable=False, default=0.0),
        Column(name="created_at", type=ColumnType.TIMESTAMP, nullable=False),
    ],
    indexes=[
        Index(name="idx_orders_user_time", columns=["user_id", "created_at"]),
    ],
))
```

### 8.5 跑测试

```powershell
python test\test_database.py
```

预期输出最后一行是 `All 33 tests passed!`（具体数字随版本更新）。任何一个 FAILED 都说明环境有问题（Python 版本过低 / 文件被占用 / 磁盘只读等），按报错定位即可。

---

## 九、API 速查

按类分小节；每个方法给 `方法签名 | 说明 | 返回`。

### 9.1 `SQLiteClient(db_path=":memory:", timeout=5.0)`

构造 + 快速片段：

```python
from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index

db = SQLiteClient("app.db")              # 文件库；":memory:" 是内存库
# with db: ...                          # 事务
# db.close()                            # 显式关；__del__ 也会尝试关
```

#### 连接

| 方法 / 属性 | 说明 | 返回 |
|---|---|---|
| `db_path: str` | 数据库路径（构造参数） | — |
| `timeout: float` | SQLite 写锁等待超时秒 | — |
| `conn: sqlite3.Connection` | 底层连接（已关闭会自动重连） | sqlite3.Connection |
| `connect()` | 显式 connect（基类接口），已关闭则重连 | None |
| `close()` | 关闭连接，置 `_conn=None` | None |

#### 基础 SQL 通道（高级/临时场景用，日常 CRUD 优先用下面的便捷方法）

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `execute(sql, params=())` | sql: str；params: tuple\|dict | `{"lastrowid": int, "rowcount": int}` | 执行写 SQL；不在事务中自动 commit |
| `executemany(sql, params_list)` | sql；list[tuple\|dict] | 同上 | 批量执行 |
| `executescript(sql)` | sql（多语句，`;` 分隔） | None | 执行脚本；不在事务中自动 commit |
| `query(sql, params=(), fetch_one=False)` | sql；params；fetch_one: bool | `list[dict] \| dict \| None` | 读 SQL，默认返回字典列表 |

#### DDL（建表 / 查表 / 删表 / 索引）

| 方法 | 参数 | 说明 |
|---|---|---|
| `table_exists(table)` | table: str | 查 sqlite_master 判断表是否存在 |
| `create_table(table, columns, if_not_exists=True)` | table: str；columns: dict[str,str] / list[Column] / Schema；if_not_exists: bool | 建表；三种 columns 写法任选；默认幂等 |
| `ensure_table(table, columns)` | 同上（无 if_not_exists） | 等价 `create_table(..., if_not_exists=True)`，**最常用入口** |
| `ensure_schema(schema, migrate=False)` | schema: Schema；migrate: bool | 按 Schema 幂等建表+建索引；migrate=True 自动补缺失列（仅 ADD COLUMN） |
| `drop_table(table, if_exists=True)` | table: str；if_exists: bool | 删表，默认 IF EXISTS |

#### 便捷 CRUD

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `insert(table, data)` | data: dict（非空） | int（lastrowid） | 插入一行；列名走 quote_ident + 命名占位符 |
| `select(table, where=None, order_by=None, limit=None)` | where: dict（等值 AND）；order_by: str（白名单，支持 `"id DESC"`）；limit: int | `list[dict]` | 查询多行 |
| `select_one(table, where=None, order_by=None)` | 同上 | `dict \| None` | 查询单行，无结果返回 None |
| `update(table, data, where)` | data: dict（非空）；where: dict（非空） | int（rowcount，影响行数） | 更新行 |
| `delete(table, where)` | where: dict（**必填**） | int（rowcount） | 删除行；**不传 where 直接抛 ValueError** 防止误删全表 |

#### 事务

```python
with db:
    db.insert(...)
    db.update(...)
# 正常退出 → commit；抛异常 → rollback
```

`__enter__ / __exit__` 实现；`_in_transaction` 标志在 finally 里复位。

### 9.2 数据模型类速查

| 类 | 构造常用字段 | 用途 |
|---|---|---|
| `ColumnType` | 枚举：TEXT/INTEGER/REAL/BLOB/BOOLEAN/VARCHAR/TIMESTAMP | 列类型，与具体 SQL 方言解耦 |
| `Column(name, type, primary_key=False, nullable=True, default=None, unique=False, index=False, length=None)` | — | 声明式列定义 |
| `Index(name, columns, unique=False)` | — | 索引（单列索引优先用 `Column(index=True)`，复合/唯一索引用这里） |
| `Schema(table_name, columns, indexes=None)` | — | 完整表结构，`ensure_schema` 的入参 |
| `SortOrder` | ASC / DESC | 查询排序方向 |
| `FilterOp` | EQ/NEQ/GT/GTE/LT/LTE/LIKE/IN/NOT_NULL/IS_NULL | 过滤操作符（当前 SQLiteClient 未接入，预留） |
| `Filter(field, op, value=None)` | — | 单个过滤条件（预留） |
| `Query(table_name, filters=None, sort_by=None, sort_order=ASC, limit=None, offset=None)` | — | 查询对象（预留，SQLiteClient 未谓词下推） |

### 9.3 已废弃/暂未开放的写法

- 目前**没有**已废弃的旧 API，v1.0.0 基线就是第一版公开契约；
- `Query/Filter/FilterOp` **不是废弃**，是"预留尚未接入 CRUD"——不要传进 select 期望生效（当前会报参数错），等后续版本接入后会在更新历史里说明。

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

1. **为什么不直接用 SQLAlchemy？**
   仓库整体招牌就是「零重型依赖、直接基于 OpenAI 原生 API」。记忆模块已经立下了"只用标准库 sqlite3"的先例（见 README 设计取舍表最后一行），数据库层如果突然引入 SQLAlchemy（~10MB、依赖链复杂）会破坏整条产品线的轻量化调性。实际上 fast_agents 里用到的 SQL 能力非常窄（单表 CRUD + 建表 + 简单索引），SQLAlchemy 90% 的功能（连接池、UnitOfWork、identity map、多方言表达式 AST）都用不上，徒增学习成本。

2. **为什么 `delete` 强制要求 where？**
   一次手滑 `db.delete("users")` 清空全表是 SQLite 新手最经典的灾难。把这个约束写死在客户端里（`if not where: raise ValueError`）比任何文档提醒都管用。真想删全表请显式 `db.execute("DELETE FROM users")`——看到裸 SQL 你会停顿一秒。

3. **为什么表名/列名要走正则白名单，而不是只做参数化？**
   SQL 参数化（`?` / `:key`）**只能防值注入，防不了标识符注入**——表名、列名、ORDER BY 字段在 SQL 语法里不能走占位符，必须字符串拼接。`check_ident` 把允许的字符集锁死在 `[A-Za-z0-9_]` 且不以数字开头，从源头堵住 `"name; DROP TABLE x;--"` 这种拼接注入。order_by 也做了类似的白名单。

4. **为什么 migrate 默认 False，且只补列？**
   "默认安全"比"默认方便"重要——新同学调用 `ensure_schema` 时如果发现表被悄悄 ALTER 了，排查起来更痛苦；要补列必须显式 `migrate=True`，表示"我知道我在做迁移"。删列/改类型这种**可能丢数据**的操作一律不做，让用户显式写迁移脚本，这和 Alembic 这类迁移器"必须生成脚本、必须 review"的精神是一致的。

5. **为什么 `execute/query` 把 sqlite3.Error 包成 ValueError？**
   上层业务模块（memory / trace）不应该 `import sqlite3` 去 catch `sqlite3.Error`，否则分层就破了——将来换 Postgres 后端就会冒出 `psycopg2.Error`，上层要改一串 except。统一包成 ValueError（带原始 SQL 和错误消息），上层只需要 catch 标准异常即可；如果要细分错误类型，后续可以在 database 模块内定义 `DatabaseError` 基类再派生（第十二章 J5 候选）。

6. **为什么默认开 WAL 和 foreign_keys？**
   `PRAGMA foreign_keys=ON` 是 SQLite 默认关闭但几乎所有人都想开的设置（否则外键约束形同虚设）；`PRAGMA journal_mode=WAL` 让并发读写互不阻塞（B9 待办），这也是 J3 跨模块迁移任务里要给 memory 三个 SQLite 后端统一收口的设置（B9 随 J3 自然落地）——database 模块先以身作则。内存数据库不开 WAL（WAL 在 :memory: 上无意义）。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| `ValueError: 非法表名/列名: xxx` | 表名/列名不符合 `^[A-Za-z_][A-Za-z0-9_]*$`，含中文/横线/空格/点号/以数字开头都会被拒。改名字，或在需要时走 `db.execute(...)` 自己拼（不推荐） |
| `ValueError: delete 必须传 where 防止误删全表` | 写了 `db.delete("users")` / `db.delete("users", {})`。真要删全表用 `db.execute("DELETE FROM users")`；按条件删必须传非空 where dict |
| `ValueError: insert 数据不能为空` / `update 需要 data 和 where` | 空 dict 调用 insert/update。检查调用方 data 是不是被覆盖成 `{}` |
| `database is locked` | SQLite 并发写冲突。一是确认所有连接都开了 WAL（SQLiteClient 默认开），二是加大 `timeout` 参数（默认 5 秒），三是检查是否有进程泄漏了连接未 close；多进程高并发写入场景建议迁移到 Postgres/MySQL（第十二章） |
| `ALTER TABLE ADD COLUMN` 报错 "Cannot add a NOT NULL column with default value NULL" | migrate 补列时新加了 `nullable=False` 且没给 default 的列，SQLite 不允许。要么给 `default=...`，要么把列声明为 `nullable=True` 再应用层校验；老数据需要非空的，先加可空列 → 回填 UPDATE → 再单独脚本改约束（SQLite 不支持后加 NOT NULL 约束） |
| `UNIQUE constraint failed: xxx.yyy` | 插入/更新违反唯一约束。这是业务层要处理的重复问题，本模块不做 upsert；需要 upsert 请用 `db.execute("INSERT ... ON CONFLICT(yyy) DO UPDATE SET ...")` |
| `executeSQL失败: ...SQL: ...` 但看不出哪错 | 异常消息里附带了完整 SQL，复制到 DB Browser for SQLite / sqlite3 CLI 里单跑一遍即可定位。params 没拼错的话，大概率是列名/表名写错、类型不匹配 |
| 多次实例化 `SQLiteClient("app.db")` 互相阻塞 | 同一个 .db 文件开了多个连接并发写会撞锁。推荐**整个进程内共享一个 client 实例**（模块级单例或用依赖注入），不要在每次函数调用里新建 |
| `Column(index=True)` 没有看到建索引 | 主键列 / UNIQUE 列 SQLite 已自动建索引，模块会跳过避免冗余（`column_index_stmt` 里显式判断）。不是 bug |
| BOOLEAN 列查出来是 0/1，不是 True/False | 是设计如此（SQLite 无原生 BOOLEAN）。业务层 `bool(row["flag"])` 转一下；或用 `detect_types=sqlite3.PARSE_DECLTYPES` + 注册适配器（当前没注册，保持简单） |
| TIMESTAMP 列存什么格式 | 存 ISO 8601 字符串（对齐 memory 模块的 `updated_at/created_at` 习惯），不要存 datetime 对象——SQLite 没有原生时间类型，字符串排序等价于时间排序（只要是 ISO 格式） |

### 10.3 跨平台注意事项（Windows 专项）

本模块涉及的跨平台点如下，全部已在代码/仓库配置里处理，但使用者仍需留意：

1. **路径分隔符**：`SQLiteClient("data\\app.db")` 和 `SQLiteClient("data/app.db")` 在 Windows 上都能工作（Python 底层会处理），但建议用 `pathlib.Path("data") / "app.db"` 拼接，避免硬编码反斜杠。
2. **数据库文件目录必须先存在**：SQLite 在 Windows 上**不会自动创建父目录**（`data/` 不存在时会报 `unable to open database file`），Linux/macOS 同理。调用方先用 `pathlib.Path.mkdir(parents=True, exist_ok=True)` 建好目录。
3. **WAL 模式下的伴生文件**：开 WAL 后 `app.db` 旁会出现 `app.db-wal` 和 `app.db-shm` 两个文件，拷贝/备份数据库时**三个文件要一起拷**，否则可能读到不一致的数据；`.gitignore` 建议加上 `*.db-wal` / `*.db-shm`（仓库 `.gitattributes` 已把 `*.db` 标为二进制，但这两个 wal/shm 也要忽略）。
4. **换行符**：`.py / .md / .txt / .bat` 一律 CRLF（由 `.gitattributes` 统一：`*.py text eol=crlf` / `*.md text eol=crlf`，LF 入库、Windows 检出自动 CRLF），`.sh` 必须 LF。数据库文件本身是二进制，不受换行符影响。
5. **文件占用**：Windows 上 SQLite 打开的文件默认不能被删除/移动（直到连接 close）。如果要在测试里 drop 后删除 .db 文件，先 `db.close()` 再删；否则会报 `PermissionError`。
6. **中文路径**：Python 3.10 + Windows 10+ 默认 UTF-8 控制台locale 下中文路径没问题；老系统若报 `codec can't decode byte`，设置环境变量 `PYTHONUTF8=1` 后再启动。

> 本模块不涉及子进程、shell、编码转换，所以 MCP/工具文档里提到的 `CREATE_NO_WINDOW` / `shlex.split(posix=False)` / `PYTHONIOENCODING` 等问题对 database 模块**不适用**——这里只列真实差异，不为凑小节写无差异项。

---

## 十一、测试与验证

| 测试文件 | 项数 | 覆盖点 | 运行命令 |
|---|---|---|---|
| `test/test_database.py` | **33 / 33**（基线 2026-11-03，提交 `bd94633`） | 连接与基本 CRUD / 字典返回 / 事务提交 / 事务回滚 / 异常后可继续使用 / select order_by 与 limit / 列名注入防护 / 非法 order_by 拦截 / delete 必须传 where / 内存库 / 文件库持久化 / close 后自动重连 / **dict 方式建表** / **list[Column] 方式建表** / **Schema 方式建表** / table_exists 幂等 / 列级索引自动建 / 复合索引 / 唯一约束 / default 值类型映射（布尔/数字/字符串/None） / BOOLEAN→0/1 / VARCHAR→TEXT / TIMESTAMP→TEXT / ensure_table 幂等重复调用 / ensure_schema 重复调用幂等 / **migrate=True 补列** / migrate=False 不补列 / drop_table / 手写 CREATE TABLE 仍可用（向后兼容） / 上下文管理器嵌套报错语义 / 批量 executemany / executescript 多语句 / 标识符非法名报错 | `python test\test_database.py` |

**关键回归断言**（这几条锁住了核心设计承诺，改代码时必须保留）：

- `test_delete_without_where_raises` —— 锁死「delete 必须带 where」的安全约束，不能因为"方便"去掉检查；
- `test_sql_injection_via_identifier_blocked` —— 锁死 `check_ident` 标识符白名单能拦截 `; DROP TABLE` 这种拼接注入；
- `test_raw_create_table_still_works` —— 锁死向后兼容：上层已有的手写 `db.execute("CREATE TABLE ...")` 必须继续可用，不能因为加了声明式建表就禁止裸 SQL；
- `test_migrate_adds_missing_columns` / `test_migrate_false_does_not_add_columns` —— 两条一起锁住 migrate 的语义：默认不动结构，显式 True 才补列；
- `test_transaction_rollback` —— 锁死事务原子性，异常时必须回滚干净；
- `test_boolean_mapped_to_integer` / `test_varchar_mapped_to_text` / `test_timestamp_mapped_to_text` —— 锁死 TYPE_TO_SQLITE 映射，未来加多后端时 SQLite 这侧的映射不能悄悄变。

**全绿基线**：在 Windows PowerShell + Python 3.10 + 干净 venv（只装 requirements.txt）环境下，`python test\test_database.py` 必须输出 `All 33 tests passed!`；任何一条红都视为环境或代码问题，不应合并。

**跨模块回归**：database 改动后，除了本模块 33 项，还要跑一遍 `test_memory.py` 里涉及 SQLiteMemoryStore 的用例（虽然 memory 目前没直接用 ensure_schema，但共用 sqlite3 驱动，方言/默认 PRAGMA 变动可能影响），以及 `test_context.py` / `test_react_agent.py` 全套，确认没有隐式依赖被破坏。

**当前测试与实现的已知差异**（如实写明，不粉饰）：

- `Query/Filter/FilterOp` 模型存在但未被 SQLiteClient 接入谓词下推，**没有对应测试**——等接入时（第十二章 J1）补；
- `connect()` 作为基类接口目前只在 `__init__` 和 `close()` 后被调用，没有独立单测；由于 `__init__` 已经调用 `_connect`，功能等价，风险低。

---

## 十二、下一步演进路线

按优先级排（与 `docs/演进路线图.md` 编号对齐；database 模块在路线图里独立为 **J 模块（持久化层）**，J0 初版已交付）：

1. **J1 · 谓词下推接入 Query/Filter**（P1，无外部依赖）—— 现状：`Query/Filter/FilterOp` 只是模型，`select` 只支持 dict 等值。目标：在 SQLiteClient 里新增 `query_by(q: Query) -> list[dict]` 方法，把 `> / < >= / <= / != / LIKE / IN / IS NULL` 编译成参数化 SQL，保持 dict 等值写法不变。**为什么现在不做**：当前 memory / trace 等上层只用到等值匹配 + order_by + limit，贸然加复杂谓词接口容易做过头；等 G1 TraceLogger 落盘需要按时间范围/模糊查询过滤时再加更合适。
2. **J2 · 连接池 / 线程安全**（P2，依赖 G2 异步并发）—— 现状：`SQLiteClient` 是单连接，线程不安全；跨线程用会偶发 `database is locked`。目标：用 `threading.local` 给每个线程一个连接，或引入简单连接池；`timeout` 默认值提到 10 秒，配合 WAL 缓解并发锁。**为什么现在不做**：fast_agents 当前是单线程同步模型（ReActAgent 顺序跑），连接池是 G2 异步并发之后的需求，提前做是过度设计。
3. **J3 · memory/trace/profile 三模块迁移到 SQLiteClient（含 B9 落地）**（**P1**，可分步推进）—— 现状：database 模块已经默认开 WAL + foreign_keys，但 memory 里三个 SQLite 后端（`sqlite_store.py` / `session_store.py` / `profile_store.py`）还在各自手写 `CREATE TABLE` + PRAGMA + `user_version` 迁移；未来 G1 TraceLogger 也需要落盘。目标：让它们逐步迁移到使用 `db.ensure_schema(..., migrate=True)` 做建表/迁移，统一 PRAGMA（foreign_keys / WAL / busy_timeout），消除重复 SQL 样板；B9（SQLite WAL + 并发安全）**随本任务自然落地**——迁移过去的后端自动享受 database 模块默认的 WAL 设置。**为什么现在不做**：这是跨模块破坏性重构，需要先确保 SQLiteClient 的能力覆盖 memory 全部现有用法（含 `user_version` 迁移、事务边界、`RETURNING` 行为），切换时不能破坏记忆数据；建议作为"阶段一夯实底座"的下一批工程债处理，本 v1.0.0 只完成 database 模块自身的文档重写，不动 memory 代码。
4. **J4 · MySQL / Postgres 后端**（P2，依赖 G4 打包发布）—— 现状：`DatabaseClientBase` 抽象基类已经定义契约，但只有 SQLite 实现。目标：新增 `fast_agents/database/mysql_client.py` / `postgres_client.py`，通过同一套 Schema/Column 模型，让上层 0 改动切换后端（走 `pip install fast-agents[mysql]` / `[postgres]` 装可选驱动）。**为什么现在不做**：现阶段 fast_agents 的目标场景是个人 / 小团队本地 Agent，SQLite 单文件零运维更合适；等 G4 PyPI 发布、有服务端部署需求时再做。
5. **J5 · 统一异常体系**（P2）—— 现状：所有异常包成 ValueError。目标：定义 `DatabaseError` 基类，派生 `IntegrityError / NotFoundError / MigrationError / ConnectionError / IdentifierError`，上层能按错误类型 catch。**为什么现在不做**：目前上层只有 database 自身 + 未来 J3 迁移过来的 memory，一律 `except ValueError` 够用；等 G1 trace / A12 审计等更多模块接入时再细分异常类型收益更大。
6. **J6 · 批量 upsert / batch_insert 性能优化**（P3）—— 现状：批量写只能循环 `insert()` 或外部调 `executemany`，没有原生 upsert（`INSERT OR REPLACE` / `ON CONFLICT DO UPDATE`）便捷方法。目标：加 `upsert(table, data, conflict_keys)` / `batch_insert(table, rows)` 便捷方法。**为什么现在不做**：memory 模块目前逐条写入，单 Agent QPS 不高，批量性能不是瓶颈。
7. **J7 · busy_timeout 默认值提升与可配置**（P2，与 J2 联动）—— 现状：J0 初版已默认对非内存库开启 `journal_mode=WAL` 与 `foreign_keys=ON`（WAL 部分落地），但 `timeout` 默认值只有 5 秒、且没有显式 `PRAGMA busy_timeout` 设置。目标：构造器新增 `busy_timeout=` 参数（默认 10 秒），`_connect()` 时显式设 `PRAGMA busy_timeout=xxx`，配合 WAL 进一步缓解并发写锁。**为什么现在不做**：单线程同步场景下 5 秒默认值够用；等 J2 连接池落地、多线程并发写入成为现实时再一并收口，避免现在加个参数但没有并发场景验证反而引入配置噪音。

当前 v1.0.0 最明显的空缺：**谓词下推（Query 模型没接入 CRUD，J1）** 和 **跨模块迁移（memory/trace/profile 仍各自手写建表，J3）**——这两项不影响 J0 初版自身功能，但决定了 database 模块能不能真正成为整个框架统一的持久化地基。

---

## 十三、参考

- [Python `sqlite3` 标准库文档](https://docs.python.org/3/library/sqlite3.html) —— 本模块唯一的底层驱动参考；Row 工厂、事务语义、PRAGMA 说明全部来源于此。
- [PEP 249 – Python Database API Specification v2.0](https://peps.python.org/pep-0249/) —— `connect/close/cursor/execute` 命名和 Connection/Cursor 抽象的来源。
- [SQLite 官方文档（CREATE TABLE / ALTER TABLE / CREATE INDEX / PRAGMA）](https://www.sqlite.org/lang.html) —— SQLite 方言能力边界（ADD COLUMN 限制、INTEGER PRIMARY KEY ROWID 别名、WAL 模式）的权威依据。
- [SQLAlchemy Core – Table / Column / MetaData](https://docs.sqlalchemy.org/en/latest/core/metadata.html) —— 声明式 Schema / Column / Index 模型的设计借鉴，但刻意不引入 SQLAlchemy 依赖。
- [Peewee Schema Migrations](http://docs.peewee-orm.com/en/latest/peewee/playhouse.html#migrate) —— SQLite 迁移能力边界的参考（即便是轻量 ORM，对 SQLite 也只能安全做 ADD COLUMN）。
- [fast_agents 记忆模块编写说明](./记忆模块编写说明.md) —— 上层依赖 database 的典型消费者，短期/长期/画像三层存储都基于 SQLite；database 模块的抽象契约是为它们服务的。
- [fast_agents 工具模块编写说明](./工具模块编写说明.md) / [MCP 模块编写说明](./MCP模块编写说明.md) / [Skill 模块编写说明](./Skill模块编写说明.md) —— 同层级模块文档，本文件的骨架、措辞粒度、代码片段长度都是照着这三份对齐的。
- [fast_agents 演进路线图](./演进路线图.md)（J 模块 持久化层）—— 第十二章的待办项与这里的 G1/G4/G5 编号保持一致，避免两处打架。

---

## 附录：发布前自检清单

写完/改完本模块文档后，逐项对照勾选：

- [x] 文档名为 `database模块编写说明.md`，落位 `docs/`（与既有文档同级）；
- [x] 开篇写清了定位、配套代码路径、版本（v1.0.0）、平台（Win/macOS/Linux）、依赖（零新增、仅 sqlite3+pydantic），且有可跳转目录；
- [x] 有更新历史，以本次 v1.0.0 作为按骨架重写初版；后续版本破坏性变更都必须给迁移办法；
- [x] 第二章讲清了分层（门面/抽象/声明模型/查询模型/实现/业务）和 5 条设计原则，后续源码导读的每一处设计都能回指某条原则；
- [x] 第三章包含对比总表（裸 sqlite3/SQLAlchemy/SQLModel/LangChain/CrewAI/fast_agents 六家）+ 异同表（**明确列出了"缺少什么"：多后端、谓词下推、连接池、迁移不完整等**）；
- [x] 第四章有目录结构（单数目录 `database/` + 5 个源码文件逐行职责注释）、分层 ASCII 图、三条主链路数据流图、与 ReActAgent 的接入边界；
- [x] 第五章弹性节认领为「三种建表写法与幂等迁移」，用对比表呈现 dict/list[Column]/Schema 三方案并明确给出选择建议；
- [x] 核心数据模型字段命名与仓库既有约定一致（`nullable` 不是 `allow_null`、`default` 不是 `default_value`、枚举值大写对齐 PEP 435）；
- [x] 源码导读 5 个文件各有「负责什么 / 关键实现 / 为什么这么写 / 改的时候最该看哪里」，关键片段控制在 3–15 行；
- [x] 快速上手命令按 Windows PowerShell 写，提醒了 `test/` 与标准库同名问题；最小可运行示例 ≤25 行、零依赖、不需要 API Key；
- [x] API 速查按类分小节（SQLiteClient 连/DDL/CRUD/事务 + 数据模型速查），无已废弃旧 API 需迁移；Query/Filter 模型如实标注"预留未接入"，不误导；
- [x] 第十章三块齐全：设计决策 6 条（每条都是"为什么不是另一种做法"）、常见坑 11 条（每条都是"现象 + 可执行解法"）、跨平台注意事项 6 条（无平台差异的内容不留"无"占位）；
- [x] 跨平台注意事项已并入 10.3（按 v2.0 模板要求不再独立成章）；
- [x] 测试章节列出 33 项覆盖点 + 6 条关键回归断言（每条锁死一个设计承诺）+ 全绿基线；Query 模型未单测的差异如实写明；
- [x] 演进路线 7 条（J1–J6 + B9 落地/J3 memory 迁移 + J7 统一 PRAGMA 默认值）每条都说明"现状是什么、为什么现在不做"，与《演进路线图.md》G/J/B 系列编号一致；
- [x] 章节数量、顺序、编号（开篇 → 一～十三 → 附录）与工具/记忆/Skill/MCP/context 五份成品文档完全一致；
- [x] 全文无"待补充 / TBD"残留，无大段与 README 重复的安装/特性介绍（README 讲怎么用，本文讲怎么实现/怎么扩展，分工明确）；
- [x] 换行符：`.md` 文件在仓库 `.gitattributes` 中声明为 `text eol=crlf`（LF 入库、Windows 检出 CRLF），符合交付要求。
