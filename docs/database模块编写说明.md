# FastAgents Database 模块编写说明

本文档面向 fast_agents **Database（持久化层）**系统，参考 **Python DB-API 2.0（PEP 249）**、**SQLAlchemy Core 的 Table/MetaData 抽象**、**SQLModel 的声明式模型**、**Peewee 的轻量 Schema** 等方案的共性设计，给出本模块的完整说明、源码解析与自定义教程。同时对齐 fast_agents 已交付的 tools / memory / mcp / skills / context 模块文档骨架，做到「翻开任何一份模块说明，都知道在哪一节能找到什么」。

> **版本**：v1.1.0 · 2026-11-05 · **关系型双后端**（`SQLiteClient` 零依赖默认后端 + `MySQLClient` 可选后端，接口 100% 对齐）+ **统一异常体系**（`errors.py` 基类 + 7 个语义子类，三后端共用）+ **向量型**（`VectorClientBase` 抽象 + `ChromaClient` 落地）
> **适用版本**：`fast_agents.database`
> **配套代码**：`fast_agents/database/`（**10 个**源码文件）+ `test/test_database.py`（**121 项**）+ `test/test_database_mysql.py`（**18 项**）+ `examples/database/`（**6 个**可运行示例）
> **运行环境**：Python 3.10+（仓库 `.python-version` = 3.10），Windows / macOS / Linux
> **依赖**：关系型默认路径**零新增核心依赖**（标准库 `sqlite3` + 全框架已有的 `pydantic>=2.14.1`）；MySQL 后端 `pymysql`、向量后端 `chromadb` 均为**可选依赖**（`pip install fast-agents[mysql]` / `pip install fast-agents[chroma]`），不进核心依赖；无 SQLAlchemy / Alembic / peewee 等 ORM
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

| **版本** | 日期 | 变更 | 作者 |
|------|------|------|------|
| **1.1.0** | 2026-11-05 | **MySQL 后端 + 统一异常体系**：新增 `mysql_client.py`（`MySQLClient`，接口与 `SQLiteClient` 100% 对齐；`pymysql` 走可选 extras、延迟导入，未装时抛 `DatabaseDriverMissingError`；方言层适配反引号标识符 / `%s` 占位符 / `TYPE_TO_MYSQL` 类型映射（BOOLEAN→TINYINT(1)、TIMESTAMP→DATETIME、INTEGER 主键 `id` 自动 `AUTO_INCREMENT`）/ `ON DUPLICATE KEY UPDATE` / `ENGINE=InnoDB` + `utf8mb4` / 索引查 `information_schema` 幂等）；`SqlExpr` 收敛为唯一实现（`mysql_client` 复用 `sqlite_client` 的类，跨后端 `isinstance` 一致）；`errors.py` 从「基类 + 3 子类」扩展为「基类 + 7 子类」（新增 `DatabaseIdentifierError` / `DatabaseIntegrityError` / `DatabaseQueryError` / `DatabaseDriverMissingError`），`SQLiteClient` / `MySQLClient` / `ChromaClient` 三后端全部按语义接入、不再裸抛驱动原生异常；`DatabaseClientBase` 补齐 `__enter__ / __exit__` 抽象方法声明，顶层 `fast_agents` 包同步导出 `MySQLClient` 与异常类；新增 `test/test_database_mysql.py`（18 项）与 `examples/database/mysql_usage.py`。 | — |
| **1.0.0** | 2026-11-03 | **初版**：`fast_agents/database/` 作为独立持久化层模块交付（`__init__.py` / `schema.py` / `query.py` / `database_client_base.py` / `sqlite_client.py` 共 5 个文件）。`SQLiteClient` 提供：连接与上下文事务；`execute / executemany / executescript / query` 基础 SQL 通道；三种声明式建表（dict / list[Column] / Schema）+ 幂等建索引 + `ensure_schema(migrate=True)` 自动 `ALTER TABLE ADD COLUMN` 补列；CRUD（`insert / batch_insert / select / select_one / select_where / select_one_where / count / update / delete`）；读侧谓词下推（10 种 `FilterOp` + 多条件 AND + 单/多列排序 + `limit/offset`，值一律参数化）；写侧条件写入（`upsert` 插入或更新、`update_where / delete_where` 带谓词批量改删，`SqlExpr` 支持自增等表达式且非空 filters 兜底防误改全表）；标识符走 `check_ident` 白名单防注入，异常统一收口到 `errors.py`（`DatabaseError` 基类 + 三个子类，均继承 `ValueError` 兼容既有捕获），非内存库默认 `WAL + foreign_keys=ON`。查询命名对齐仓库既有 `select / select_one` 家族，定为 `select_where / select_one_where`。**关系型零新增依赖**（标准库 `sqlite3` + 全框架已有的 `pydantic`）。**同版并入向量型持久化能力**：新增 `vector_query.py` / `vector_client_base.py` / `chroma_client.py` 三个源码文件——`VectorClientBase` 与 `DatabaseClientBase` 平行定义向量后端契约（集合 / 写入 / 删除 / 取 / 近邻检索 / 计数共 12 个方法），`ChromaClient` 作为首个向量后端（本地目录 `PersistentClient` 与远端 `HttpClient` 双模式，`chromadb` 走可选依赖 extras），`VectorFilter` 提供 EQ/NE/GT/…/IN/AND/OR 的元数据过滤。**同版修补 `batch_insert` 静默丢数据缺陷**：`batch_insert` 原先直接调用 `self.conn.executemany(...)` 不 commit，在 Python sqlite3 默认隐式事务下，调用方未包 `with db:` 就 close 会导致数据被隐式回滚、落盘丢失；修复为走 `self.executemany(...)` 封装，复用其「事务外自动 commit、在事务中由外层统一提交/回滚」的既有语义。**同版并入异常体系（J5 第一刀）**：新增 `errors.py`，把 `SQLiteClient` 的底层 `sqlite3.Error` 与入参校验错误从裸 `ValueError` 收口为 `DatabaseConnectionError` / `DatabaseOperationalError` / `DatabaseValidationError`（统一继承 `DatabaseError` → `ValueError`，零破坏）。`test/test_database.py` 共 **121** 项测试（98 关系型 + 23 向量，其中 17 项需 chromadb），配套示例 5 个。 | — |

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
| **查询模型层** | 运行期 | 过滤/排序/分页的轻量对象 + FilterOp 编译 | `query.py`（`Query/Filter/FilterOp/SortOrder`）+ `sqlite_client._compile_filter/_compile_query` / `mysql_client._compile_filter/_compile_query`（谓词下推） | 不依赖具体后端，编译逻辑在每个关系型后端内各一份（纯字符串拼装 + 命名参数 / `%s` 位置参数），各后端共用同样的 `Filter` 语义 |
| **实现层（后端）** | 运行期 | 具体数据库驱动封装 | `sqlite_client.py`（`SQLiteClient`，零依赖默认后端）、`mysql_client.py`（`MySQLClient`，可选后端，接口与 SQLite 100% 对齐） | 依赖抽象层 + schema 解析函数 + 各自驱动（标准库 `sqlite3` / 可选 `pymysql`） |
| **上层业务模块** | 运行期 | 记忆落盘 / Trace 落盘 / 画像落盘 | `memory/` 三 store（sqlite_store / session_store / profile_store）、`trace/sqlite_store.py` 等 | 只依赖抽象层 + 声明模型，不直接 `import sqlite3` |
| **向量抽象层（契约）** | 编译期 | 向量后端必须实现的最小接口集 | `vector_client_base.py`（`VectorClientBase`） | 与 `DatabaseClientBase` **平行、互不继承**；上层只引用它 |
| **向量声明模型层** | 编译期 | 向量元数据过滤模型（EQ/NE/GT/…/IN/AND/OR） | `vector_query.py`（`VectorFilterOp` / `VectorFilter`） | 纯数据模型，不依赖任何连接 |
| **向量实现层（后端）** | 运行期 | 具体向量库驱动封装 | `chroma_client.py`（`ChromaClient`） | 依赖向量抽象层 + 标准库；`chromadb` 为可选依赖 |

### 2.3 关键设计原则（全篇代码都在落实它们）

1. **零重型依赖**：默认路径只用标准库 `sqlite3` + 全框架已有的 `pydantic`，不引入 SQLAlchemy / SQLModel / Alembic / peewee 任何 ORM；MySQL / Chroma 等额外后端的驱动一律走**可选依赖**（extras）+ 延迟导入，不装也不影响默认路径。这是 README「零重型依赖」招牌的延续（参考「记忆模块只用标准库」那条设计取舍）。
2. **声明式而非字符串**：列、索引、表结构用 pydantic 模型表达（`Column/Index/Schema`），拼 SQL 的逻辑统一收口在 `schema.py` 的纯函数里，上层**不手写** `CREATE TABLE`，避免 SQL 注入、列名拼错、类型不统一。
3. **默认幂等、默认安全**：`ensure_table / ensure_schema` 默认 `IF NOT EXISTS`；删除必须带 `where` 防止误删全表；表名/列名/索引名走 `check_ident` 白名单，从源头堵标识符注入；`delete` 不传 `where` 直接 `ValueError`。
4. **最小惊讶**：`BOOLEAN→INTEGER(0/1)`、`VARCHAR→TEXT`、`TIMESTAMP→TEXT(ISO 字符串)` 的类型映射写死在 `TYPE_TO_SQLITE`；`select/insert/update/delete` 全部返回字典 / 整数，不用 cursor / Row 对象给上层添麻烦。
5. **迁移能力克制**：`ensure_schema(..., migrate=True)` 只做后端原生支持的 `ADD COLUMN` 补列（SQLite / MySQL 语义一致），不做删列/改类型/改主键/数据迁移——这是能力边界的诚实反映，不假装自己是完整 ORM；更复杂的迁移交给上层显式脚本。

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
| **fast_agents（本模块）** | 轻量持久化地基 | **默认零新依赖**（标准库 sqlite3 + pydantic）；MySQL 走可选 pymysql | **三种写法并存**：dict / list[Column] / Schema（`ensure_table`/`ensure_schema`） | **`migrate=True` 补列**（仅 ADD COLUMN） | 零依赖默认路径、声明式、幂等、三种写法按场景选；关系型已落地 SQLite + MySQL 双后端且接口 100% 对齐 | 关系型尚未做 Postgres；迁移只做补列、不支持删/改；复杂查询（OR / JOIN / 聚合）仍需回退裸 SQL |

### 3.2 关键差异：相同 / 增加 / 缺少

| 维度 | 与"裸 sqlite3"相同 | 相对裸 sqlite3 增加 | 目前缺少 |
|---|---|---|---|
| 驱动能力 | 连接 / 事务 / 参数化查询 / execute/executemany | 字典结果、WAL、外键默认开启、超时参数 | 异步 `aiosqlite` 连接、连接池 |
| 建表 | 仍能手写 `db.execute("CREATE TABLE ...")`，**完全向后兼容** | dict / list[Column] / Schema 三种声明式写法；自动建索引；`IF NOT EXISTS` 默认幂等 | 外键声明、CHECK 约束、表注释（SQLite 本身不支持） |
| 迁移 | — | `ensure_schema(schema, migrate=True)`：SQLite 走 `PRAGMA table_info`、MySQL 查 `information_schema.columns`，缺列自动 ADD COLUMN | 删列 / 改类型 / 改主键 / 重命名 / 数据迁移（后端通用限制） |
| 查询 | — | where 等值匹配、order_by 白名单校验、limit；`select_where` 接入 10 种 `FilterOp`（>、<、LIKE、IN、IS NULL 等）+ AND 组合 + 单/多列排序 + limit/offset 分页，参数化防注入 | OR、JOIN、聚合、GROUP BY（刻意不做，保持零重型依赖） |
| 后端扩展 | — | `DatabaseClientBase` 抽象基类定义了最小接口契约；已落地 MySQL 后端（`MySQLClient`，与 SQLite 接口对齐） | Postgres / DuckDB 后端实现 |
| 安全 | — | 标识符白名单 `check_ident` 防表名/列名注入；delete 必须带 where | 列级权限、审计日志（留给 trace 模块的审计能力） |

一句话收口：**本模块 = 裸 sqlite3 的零依赖 + SQLAlchemy Core 声明式建表的最小子集 + 向 ORM 迁移留下的接口，但刻意不做完整 ORM 和迁移器**——这和 README 里"基于 OpenAI 原生 API、零重型依赖"的整体调性一致。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/database/                 ← 单数目录（对齐 tools/memory/skills/mcp 的命名约定）
├── __init__.py                       # 模块出口：统一导出关系型 + 向量型 + 异常体系的全部公共符号（8 个异常类 + 5 个客户端/基类 + 4 个 Schema 模型 + 4 个查询模型 + 2 个向量过滤模型 + SqlExpr，共 24 个符号）
├── database_client_base.py           # 【关系型】抽象基类 DatabaseClientBase：connect/close + DDL（建/查/删表）+ DML（CRUD，含 select_where/select_one_where）+ 事务上下文（__enter__/__exit__）契约
├── errors.py                         # 【通用】异常体系：DatabaseError（继承 ValueError）基类 + 7 个语义子类（Connection/Operational/Validation/Identifier/Integrity/Query/DriverMissing），关系型与向量型共用
├── schema.py                         # 【关系型】声明模型 ColumnType/Column/Index/Schema + 标识符校验 + Column/Schema→SQL 片段纯函数（各后端复用，与方言无关）
├── query.py                          # 【关系型】查询模型 SortOrder/FilterOp/Filter/Query（轻量过滤/排序/分页，由各后端 select_where 编译为参数化 SQL）
├── sqlite_client.py                  # 【关系型】SQLiteClient：零依赖默认后端，封装 sqlite3 连接、事务、字典结果、幂等建表、便捷 CRUD、select_where 谓词下推、count/batch_insert 统计与批量写入，并定义跨后端共用的 SqlExpr
├── mysql_client.py                   # 【关系型】MySQLClient：可选 MySQL 后端，接口与 SQLiteClient 100% 对齐，pymysql 延迟导入；方言层负责反引号标识符 / %s 占位符 / 类型映射 / ON DUPLICATE KEY UPDATE / InnoDB+utf8mb4 建表 / information_schema 索引幂等
├── vector_query.py                   # 【向量型】向量元数据过滤模型 VectorFilterOp/VectorFilter（对齐 query.py，但支持 AND/OR 嵌套、不支持 LIKE/IS NULL）
├── vector_client_base.py             # 【向量型】抽象基类 VectorClientBase：连接 + 集合管理 + 写入/删除/取/近邻检索/计数契约（12 个方法）
└── chroma_client.py                  # 【向量型】ChromaClient：首个向量后端，本地 PersistentClient / 远端 HttpClient 双模式，chromadb 为可选依赖（延迟导入）
```

配套但不放在 `fast_agents/database/` 下：

- `../examples/database/sqlite_usage.py` —— CRUD + 事务最小闭环示例；
- `examples/database/create_table_usage.py` —— 三种建表写法 + migrate + drop 完整示例；
- `examples/database/query_filter_usage.py` —— 谓词下推：10 种 FilterOp + AND 组合 + 排序分页 + 注入安全完整示例；
- `examples/database/conditional_write_usage.py` —— 写侧条件写入：upsert + update_where / delete_where 完整示例；
- `examples/database/chroma_usage.py` —— 向量后端：建集 → add → get → search → count → delete → drop 完整示例（需 `pip install fast-agents[chroma]`）；
- `examples/database/mysql_usage.py` —— MySQL 后端：同一套业务代码跑 SQLite / MySQL，默认只跑 SQLite 分支（零依赖、全内存），有 MySQL 环境时取消注释即可切换；
- `test/test_database.py` —— **121 项**单测：98 项关系型用纯内存数据库、无外部依赖；23 项向量中 6 项纯抽象 / 编译校验无外部依赖，17 项走真实 Chroma 临时目录、未装 chromadb 时自动 skip；
- `test/test_database_mysql.py` —— **18 项**单测：接口对齐 / 方言编译 / 类型映射 / SqlExpr 跨后端共用 / 可选依赖降级 / 异常语义对齐等离线用例，另有 1 项真实 MySQL 端到端（设 `FAST_AGENTS_TEST_MYSQL_URL` 才跑，否则 skip）。

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
│   DML：insert / batch_insert / select / select_one / update / delete │
└─────────┬─────────────────────────────────┬─────────────────────────┘
          │实现（SQLite 默认 / MySQL 可选）      │ 共用纯函数
┌─────────▼─────────────────────┐   ┌────────▼──────────────────────┐
│ SQLiteClient / MySQLClient     │   │ schema.py（纯数据 + 纯函数）    │
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
                │ 谓词下推编译入口    │  （编译为参数化 SQL）
                └───────────────────┘
                          │
                ┌─────────▼─────────┐
                │ 标准库 sqlite3     │
                └───────────────────┘
```

向量型后端与上图的**关系型分层平行**，另有一组三层（互不继承、各走各的）：

```text
┌───────────────────────────────────────────────────────────────────┐
│ 上层业务模块（未来的 memory/semantic_store 语义记忆）                 │
│ 只 import VectorClientBase + VectorFilter，不直接 import chromadb     │
└───────────────────────────┬───────────────────────────────────────┘
                            │ 依赖抽象，不依赖具体后端
┌───────────────────────────▼───────────────────────────────────────┐
│ VectorClientBase（vector_client_base.py）—— 向量客户端契约            │
│   连接：connect / close / ping                                       │
│   集合：collection_exists / create_collection /                      │
│        get_or_create_collection / drop_collection                    │
│   DML：add / delete / get / search / count                           │
└─────────┬─────────────────────────────────┬─────────────────────────┘
          │实现（目前唯一）                    │ 过滤模型
┌─────────▼─────────────────────┐   ┌────────▼──────────────────────┐
│ ChromaClient（chroma_client.py）│   │ vector_query.py（纯数据模型）   │
│  - 本地 PersistentClient /      │   │ VectorFilterOp / VectorFilter  │
│    远端 HttpClient 双模式       │   │ （编译为 Chroma where，见 §6.7）│
│  - 集合名白名单 + 维度校验       │   └───────────────────────────────┘
│  - add 自动分块 / delete 精确计数│
└───────────────────────────────┘
          │
          ▼
   可选依赖 chromadb（pip install fast-agents[chroma]）
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
with db:                      → __enter__ 执行 BEGIN IMMEDIATE，_tx_depth 0→1
    db.insert(...)            → execute 检测到 _in_transaction（=_tx_depth>0）→ 不 commit
    db.update(...)            → 同上
    # 抛异常 → __exit__ rollback + _tx_depth 归 0
    # 正常   → __exit__ commit   + _tx_depth 归 0
```

> `_in_transaction` 是 `_tx_depth > 0` 的只读派生属性，不保存独立状态，因此不存在"标志与真实事务状态失同步"。嵌套 `with db:`（如 memory 层 `_connect()` 内再开事务）不会重复 `BEGIN`，只把 `_tx_depth` 加一，由最外层统一提交/回滚。

### 4.4 与 ReActAgent 及上层模块的接入点 / 职责边界

当前 database 模块**还没有直接挂进 ReActAgent**，它是作为 `memory/`、`trace/` 等模块的底层依赖被间接使用：

- `memory/sqlite_store.py`、`memory/session_store.py`、`memory/profile_store.py` 均已迁移到 `SQLiteClient`（声明式 `ensure_schema` 建表 + 统一连接/事务），不再各自手写 `CREATE TABLE`；上层 API 不变。
- `trace/sqlite_store.py` 的 `SQLiteStore` 同样基于 `SQLiteClient` 落结构化 trace（v1.1.0 起）——这也印证了 Database 模块独立出来的动因：避免 trace / 记忆 / 画像各写各的建表 SQL。
- **单一数据源约定**（对齐记忆模块的精神）：Agent 跨轮持久化状态只有两条路——要么走 `MemoryManager`（短期 buffer + 长期 store），要么走各业务模块自己持有的 `DatabaseClientBase` 实例；**禁止上层模块绕开抽象基类直接 `import sqlite3`**，否则后续换 Postgres 时会爆雷（MySQL 已可用同一抽象无缝切换）。
- 与 `tools/builtin/` 的边界：`FileReadTool / FileWriteTool` 等文件工具处理的是文件系统（不是数据库）；数据库操作不通过 tool 暴露给 LLM，避免模型误删数据——**DatabaseClient 是框架内部基础设施，不做成 LLM 可调用工具**（这点与 MCP/tool 模块明确划界）。

---

## 五、核心机制：三种建表写法、幂等迁移与谓词下推

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
| 列级索引 | SQLite / MySQL 均"先查后建"：SQLite 用唯一命名 `idx_{table}_{col}` + `CREATE INDEX IF NOT EXISTS`；MySQL 先查 `information_schema.statistics` 再建同名索引，重复调用不会报错也不会重复建 |
| 显式索引（Schema.indexes）| 同上；SQLite 走 `CREATE [UNIQUE] INDEX IF NOT EXISTS`，MySQL 走「查 information_schema → 缺才 CREATE INDEX」，命名由调用方保证唯一 |
| 补列（migrate） | SQLite 先 `PRAGMA table_info(table)` 拿现有列集合，MySQL 查 `information_schema.columns`，缺的才 `ALTER TABLE ADD COLUMN`；已存在列自动跳过 |
| insert | 不保证幂等（业务层自己用唯一约束 + try/except 处理重复） |
| update/delete | where 条件决定，不是本模块职责 |

### 5.3 migrate 的能力边界（诚实列缺项）

`ensure_schema(schema, migrate=True)` **只做后端原生支持的"加列"这一件事**（SQLite / MySQL 语义一致）：

- ✅ 补新增列（`ALTER TABLE ADD COLUMN`）；
- ❌ 不删列（删列有丢数据风险，模块不代为执行）；
- ❌ 不改列类型 / 不改列名 / 不加主键 / 不加唯一约束（这些都需要重建表，超出"安全自动迁移"边界）；
- ❌ 不迁移数据（例如把布尔 0/1 改成字符串 "yes"/"no"）。

遇到上述复杂迁移，**显式写迁移脚本**（用 `db.executescript()`），不要指望模块默默做掉——这和「最小惊讶」原则一致：宁可让你看到要写迁移，也不要静默改你的数据。

### 5.4 谓词下推：FilterOp 编译管线

`query.py` 里的 `Query / Filter / FilterOp` 是声明式查询模型，由 SQLiteClient 编译为参数化 SQL。`select(where=dict)` 只做等值匹配，遇到 `> / < / LIKE / IN / IS NULL` 这类条件时用下面两个方法：

- `select_where(query: Query) -> list[dict]`：按声明式 Query 查多行；
- `select_one_where(query: Query) -> dict | None`：单行版本（内部自动补 `limit=1`，**不会修改调用方传入的 Query 对象**）。

**编译规则**（集中在 `SQLiteClient._compile_filter` / `_compile_query` 两个静态方法）：

| 规则 | 具体做法 | 为什么 |
|---|---|---|
| 列/表/sort_by 一律走 `check_ident` 白名单 | 正则 `^[A-Za-z_][A-Za-z0-9_]*$`，非法直接 `ValueError` | SQLite 标识符不能用 `?` 参数化，必须白名单拦截拼接注入 |
| 值一律走命名参数 `:f{idx}_{k}` | 拼 SQL 里出现的只有占位符，值通过 params 传入 | 利用 sqlite3 自身参数化，杜绝 `' OR '1'='1` 注入 |
| 多 Filter 固定 AND | `cond1 AND cond2 AND cond3` | 90% 业务查询是 AND；OR/JOIN 留到后续版本，先把最小可用做扎实 |
| IS NULL / IS NOT NULL 不传 value | 漏传 value 直接报错；传 None 给二元操作符也报错 | 避免把 `= NULL` 这种"永远为假"的坑暴露给调用方 |
| IN 要求非空 list/tuple | `value=[a,b,c]` 展开成 `IN (:f0_0, :f0_1, :f0_2)` | 空列表会变成 `IN ()` 语法错误，提前拦截 |
| LIKE 不替用户补 `%` | `value` 原样传入，要模糊必须自己写 `%name%` | 避免"隐式加通配符"导致的不精确匹配（最小惊讶） |
| sort_by 支持单字段 str 或多字段 list[str] | 单字段按原样拼；多字段按列顺序展开为 `ORDER BY c1 dir, c2 dir`，方向对每列统一应用 | 知识库的 `ORDER BY updated_at DESC, memory_id` 这类稳定排序需要多列；单字段写法保持兼容 |
| offset 无 limit 时自动补 `LIMIT -1` | SQLite 语法要求 OFFSET 必须配 LIMIT；`-1` 在 SQLite 里表示"无上限" | 允许 `select_where(Query(..., offset=100))` 这种"从第 N 条开始取全部"用法 |
| limit/offset 强制非负 int | `< 0` 直接报错 | 防御性校验，避免拼接出语法错误或翻页逻辑翻车 |

**为什么保留原 `select(where=dict)` 不替换？**

两种写法并存，按场景选：

| 场景 | 推荐写法 |
|---|---|
| 按主键/外键/精确字段查一行或几行 | `db.select_one("users", {"id": uid})`（最短） |
| 等值 + AND 组合、不需要比较/模糊 | `db.select("users", {"dept": "eng", "active": 1}, order_by="id")` |
| 范围查询 / 模糊 / IN / 空值判断 / 分页 | `db.select_where(Query(...))` |
| 多表 JOIN / 聚合 / 子查询 / 复杂表达式 | `db.query("SELECT ...", params)`（回退到裸 SQL，不假装 ORM） |

> **诚实声明能力边界**：本模块不做 OR、不做 JOIN、不做 GROUP BY/HAVING、不做子查询、不做 BETWEEN 语法糖（用 `GTE + LTE` 两个 Filter AND 即可表达）。这些在 Agent 框架的持久化场景里使用频率极低，一旦做了就要持续维护方言差异，违反「零重型依赖、最小可用」原则。

### 5.5 写侧谓词下推：upsert 与 update_where / delete_where

读侧用 `Query` 表达过滤（`select_where`）；写侧同样提供三个方法，避免上层自己拼 `ON CONFLICT` / `UPDATE ... IN (...)`：

| 方法 | 场景 | 关键点 |
|---|---|---|
| `upsert(table, data, conflict_keys, update_columns=None)` | 有则更新、无则插入 | 编译为 `INSERT ... ON CONFLICT(keys) DO UPDATE SET col = excluded.col`；`update_columns` 不传则更新除冲突键外的全部列，传空列表则 `DO NOTHING`；值全走命名参数 |
| `update_where(table, values, query)` | 带谓词的批量更新 | `values` 普通值走参数化；需要「自增 / 函数计算」时用 `SqlExpr("access_count + 1")` 显式包裹（**只有它会被原样拼入 SQL**） |
| `delete_where(table, query)` | 带谓词的批量删除 | 条件全部来自 `query.filters`，与读侧共用同一套 `Filter` 语义 |

**两条硬约定**：

1. `update_where` / `delete_where` 的 `query.filters` **必须非空**，空条件直接 `ValueError`——与 `delete(table, where)` 同一条「防手滑清全表」的安全线；真要全表改删请显式写 `db.execute("UPDATE ...")` / `db.execute("DELETE FROM ...")`。
2. `SqlExpr` 是**唯一**会把内容原样拼进 SQL 的入口，因此**只允许传程序里写死的固定片段**，严禁拼接任何外部输入；普通值一律直接传、走参数化。

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

### 6.5 查询模型（`query.py`，谓词下推的声明式入口）

```python
class SortOrder(str, Enum): ASC = "ASC"; DESC = "DESC"

class FilterOp(str, Enum):
    EQ="="; NEQ="!="; GT=">"; GTE=">="; LT="<"; LTE="<="
    LIKE="LIKE"; IN="IN"; NOT_NULL="IS NOT NULL"; IS_NULL="IS NULL"

class Filter(BaseModel):
    field: str
    op:    FilterOp
    value: Optional[Any] = None  # IS NULL / NOT_NULL 时无需传；IN 时必须是非空 list/tuple

class Query(BaseModel):
    table_name:  str
    filters:     Optional[list[Filter]] = None  # AND 关系；由 select_where 编译
    sort_by:     Optional[str | list[str]] = None  # 单字段 str 或多字段 list[str]，每列都须是合法标识符
    sort_order:  SortOrder = SortOrder.ASC
    limit:       Optional[int] = None            # 非负 int
    offset:      Optional[int] = None            # 非负 int；单独使用时 SQLite 后端会自动补 LIMIT -1
```

> **谓词下推**：`SQLiteClient.select_where(query)` 和 `select_one_where(query)` 把上面这套模型编译为参数化 SQL，覆盖 10 种 `FilterOp`、多条件 AND、单/多列排序与分页；写侧 `update_where / delete_where` 复用同一套 `Filter` 语义。详见 §5.4 / §5.5 与 §9.1 API 速查。等值简单场景仍然推荐用更短的 `select(where=dict)`，两种写法共存。OR / JOIN / 聚合 / BETWEEN 糖衣语法**刻意不做**，保持零重型依赖。

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
    @abstractmethod def batch_insert(self, table, rows) -> int: ...
    @abstractmethod def select(self, table, where=None, limit=None) -> list[dict]: ...
    @abstractmethod def select_one(self, table, where) -> dict | None: ...
    @abstractmethod def select_where(self, query) -> list[dict]: ...
    @abstractmethod def select_one_where(self, query) -> dict | None: ...
    @abstractmethod def count(self, table, where=None) -> int: ...
    @abstractmethod def update(self, table, data, where) -> int: ...
    @abstractmethod def delete(self, table, where) -> int: ...
    # 事务
    @abstractmethod def __enter__(self) -> "DatabaseClientBase": ...
    @abstractmethod def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...
```

子类**必须实现全部 18 个方法**（连接 2 + DDL 5 + DML 9 + 事务上下文 2），上层（memory / trace / profile）只依赖这个抽象，不直接 import sqlite3/pymysql。SQLiteClient / MySQLClient 已经全部实现。

### 6.7 向量过滤模型（`vector_query.py`，向量侧的谓词声明）

```python
class VectorFilterOp(str, Enum):
    EQ="$eq"; NE="$ne"; GT="$gt"; GTE="$gte"; LT="$lt"; LTE="$lte"
    IN="$in"; NIN="$nin"; AND="$and"; OR="$or"

class VectorFilter(BaseModel):
    field:   Optional[str] = None                    # 叶子条件必填
    op:      VectorFilterOp
    value:   Optional[Any] = None                    # 叶子条件用
    filters: Optional[list["VectorFilter"]] = None    # op 为 AND/OR 时的子条件（≥2 个）

VectorWhere = Union[dict[str, Any], VectorFilter, None]   # 便捷类型别名
```

> 枚举值直接写成 Chroma 的 `$` 前缀操作符，编译时零映射成本：叶子条件编译为 `{"field": {"$op": value}}`、AND/OR 编译为 `{"$and": [...]}` / `{"$or": [...]}`（编译逻辑在 `chroma_client._compile_vector_filter`）。**刻意不复用关系型 `query.Filter`**：向量过滤支持 AND/OR 嵌套、不支持 LIKE / IS NULL（Chroma metadata 过滤本身没有这两个能力），两边谓词集合不同，各自演进互不影响。

### 6.8 `VectorClientBase`（向量抽象基类，与关系型平行）

```python
class VectorClientBase(ABC):
    # 连接生命周期
    @abstractmethod def connect(self) -> None: ...
    @abstractmethod def close(self) -> None: ...
    @abstractmethod def ping(self) -> bool: ...
    # 集合（对应关系型的「表」）
    @abstractmethod def collection_exists(self, name) -> bool: ...
    @abstractmethod def create_collection(self, name, *, dimension=None, metric="cosine") -> None: ...
    @abstractmethod def get_or_create_collection(self, name, *, dimension=None, metric="cosine") -> None: ...
    @abstractmethod def drop_collection(self, name) -> None: ...
    # 写入 / 删除 / 取 / 近邻检索 / 计数
    @abstractmethod def add(self, collection, ids, embeddings=None, documents=None, metadatas=None) -> int: ...
    @abstractmethod def delete(self, collection, ids=None, where=None) -> int: ...
    @abstractmethod def get(self, collection, ids=None, where=None, limit=None,
                            include_embeddings=False, include_documents=True, include_metadatas=True) -> list[dict]: ...
    @abstractmethod def search(self, collection, query_embedding, *, n_results=10, where=None, include_embeddings=False) -> list[dict]: ...
    @abstractmethod def count(self, collection, where=None) -> int: ...
```

子类**必须实现全部 12 个方法**，命名与关系型基类保持平行（`create_collection ↔ create_table`、`get_or_create_collection ↔ ensure_table`、`drop_collection ↔ drop_table`）。**允许一个后端同时继承 `DatabaseClientBase` 和 `VectorClientBase`**（如未来的 Postgres + pgvector），上层用 `isinstance` 判断能力即可。本轮只落地 Chroma 一个向量后端。

### 6.9 异常体系（`errors.py`，关系型 / 向量型共用）

```python
class DatabaseError(ValueError):                       # 基类（刻意继承 ValueError 保兼容）
# 三大层
class DatabaseConnectionError(DatabaseError): ...      # 连接层：建连 / 关连接失败
class DatabaseOperationalError(DatabaseError): ...     # 执行层：SQL 执行 / 查询失败
class DatabaseValidationError(DatabaseError): ...      # 校验层：入参不合法 / 危险操作被拒
# 四个具体场景（细化到可精确分流）
class DatabaseIdentifierError(DatabaseValidationError): ...     # 标识符（表/列/集合名）非法
class DatabaseIntegrityError(DatabaseOperationalError): ...     # 约束冲突（主键/唯一键/集合已存在）
class DatabaseQueryError(DatabaseOperationalError): ...         # 查询/写入失败（集合不存在等）
class DatabaseDriverMissingError(DatabaseConnectionError): ...  # 可选驱动未安装（pymysql / chromadb）
```

| 异常 | 继承 | 触发场景 | 上层典型处理 |
|---|---|---|---|
| `DatabaseError` | `ValueError` | 模块异常基类 | 统一兜底 |
| `DatabaseConnectionError` | `DatabaseError` | 路径不可写、驱动不可用、连接被拒 | 报配置错误 / 换库路径 |
| `DatabaseOperationalError` | `DatabaseError` | SQL 语法错、约束冲突、锁等待超时 | 记日志 / 重试 / 修 SQL |
| `DatabaseValidationError` | `DatabaseError` | 空数据、非法标识符、空 filters 防误删 | 改调用参数 |
| `DatabaseIdentifierError` | `DatabaseValidationError` | 表名 / 列名 / 集合名不合法 | 改标识符 |
| `DatabaseIntegrityError` | `DatabaseOperationalError` | 唯一键 / 主键冲突、集合已存在 | 改用 upsert / get_or_create_collection |
| `DatabaseQueryError` | `DatabaseOperationalError` | 集合不存在、写入失败 | 先建集合 / 查条件 |
| `DatabaseDriverMissingError` | `DatabaseConnectionError` | pymysql / chromadb 未安装 | 装可选依赖 |

**关键点**：8 个异常全部继承 `DatabaseError`，而 `DatabaseError` **刻意继承 `ValueError`**——既有 `except ValueError` 与 `pytest.raises(ValueError)` 全部继续有效，做到零破坏；新代码可用具体子类精确分流。入参类型传错（如把 int 当表名）仍抛原生 `TypeError`，不纳入本体系。

---

## 七、逐个文件讲解（源码导读）

每个文件固定三段：**它负责什么 → 关键实现 → 为什么这么写**。

### 7.1 `__init__.py` —— 模块出口，统一 re-export

```python
from .chroma_client import ChromaClient
from .database_client_base import DatabaseClientBase
from .mysql_client import MySQLClient
from .errors import (
    DatabaseConnectionError, DatabaseDriverMissingError, DatabaseError,
    DatabaseIdentifierError, DatabaseIntegrityError, DatabaseOperationalError,
    DatabaseQueryError, DatabaseValidationError,
)
from .query import Filter, FilterOp, Query, SortOrder
from .schema import Column, ColumnType, Index, Schema
from .sqlite_client import SqlExpr, SQLiteClient
from .vector_client_base import VectorClientBase
from .vector_query import VectorFilter, VectorFilterOp

__all__ = [
    # 异常体系（8）
    "DatabaseError", "DatabaseConnectionError", "DatabaseDriverMissingError",
    "DatabaseIdentifierError", "DatabaseIntegrityError", "DatabaseOperationalError",
    "DatabaseQueryError", "DatabaseValidationError",
    # 关系型客户端（3）
    "SQLiteClient", "MySQLClient", "DatabaseClientBase",
    # 向量型客户端（2）
    "ChromaClient", "VectorClientBase",
    # Schema / 查询 / 向量过滤模型（10）
    "Schema", "Column", "ColumnType", "Index",
    "Query", "Filter", "FilterOp", "SortOrder",
    "VectorFilter", "VectorFilterOp",
    # SQL 表达式包装（1）
    "SqlExpr",
]
```

- **负责什么**：给上层一个稳定的 import 入口（`from fast_agents.database import SQLiteClient, Schema, Column, ColumnType, Index`），避免上层直接 import 子文件路径，将来内部文件调整不影响调用方。
- **为什么这么写**：`__all__` 显式列出 24 个符号（8 异常 + 3 关系型客户端 + 2 向量型客户端 + 10 声明模型 + 1 SqlExpr），既避免 `from database import *` 污染，也是给阅读者一张"公共 API 清单"。对齐 `fast_agents/tools/__init__.py`、`fast_agents/memory/__init__.py` 的 re-export 模式。
- **你要改的时候最该看的地方**：新增公共类后记得在这里加一行导出 + `__all__` 一项，否则上层 `from fast_agents.database import Xxx` 会 ImportError。

### 7.2 `database_client_base.py` —— 客户端抽象基类

```python
class DatabaseClientBase(ABC):
    @abstractmethod
    def create_table(self, table, columns, if_not_exists=True):
        """columns 支持三种形式：dict[str,str] / list[Column] / Schema"""
```

- **负责什么**：定义所有关系型后端（SQLite / MySQL / 未来 Postgres）必须实现的最小接口集——连接（2：`_connect` / `_close` 抽象，`connect` / `close` 由基类实现）+ DDL（5）+ DML（9）+ 事务上下文（2：`__enter__` / `__exit__`）共 18 个方法。
- **关键实现**：docstring 里写死了 `columns` 三种形式的语义约束，这是一份"接口契约文档"，子类实现必须同时支持这三种。
- **为什么这么写**：抽象基类里不加事务/executemany/execute 等"半可选"方法，避免接口膨胀；上层业务只依赖这套接口，将来换后端改一行构造代码即可。**你要扩展新后端时，第一个要实现的就是这个类**，照着 docstring 把 18 个方法都写了，memory/trace 上层代码零改动。

### 7.3 `schema.py` —— 声明模型 + SQL 片段纯函数（整个模块的大脑）

```python
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def check_ident(name: str, kind: str = "标识符") -> None:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise DatabaseValidationError(f"非法{kind}: {name!r}（只允许字母/数字/下划线，且不以数字开头）")

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

### 7.4 `query.py` —— 轻量查询对象

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
- **为什么这么写**：SQLiteClient 用 `_compile_filter / _compile_query` 把这套模型编译为参数化 SQL，上层不写裸 SQL 就能表达比较 / 模糊 / 集合 / 空值 / 分页；模型与后端解耦，未来加 MySQL/Postgres 后端时可直接复用同一套 Filter 语义。

### 7.5 `sqlite_client.py` —— SQLite 客户端（零依赖默认后端）

```python
class SQLiteClient(DatabaseClientBase):
    def __init__(self, db_path: str = ":memory:", timeout: float = 5.0):
        self.db_path = db_path
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._tx_depth = 0            # 事务嵌套深度（事务状态的唯一真实来源）
        self._connect()
    # _in_transaction 为只读 property，派生自 _tx_depth > 0
```

- **负责什么**：把标准库 sqlite3 封装成上层好用的客户端——字典行结果、自动重连、WAL/外键默认开启、事务上下文、声明式建表三写法分派、幂等 migrate、便捷 CRUD。
- **关键实现**（按方法块）：
  - **连接** `_connect()`：`sqlite3.connect(..., detect_types=PARSE_DECLTYPES, isolation_level=None)`（`isolation_level=None` 关闭 Python 隐式事务，事务边界完全由代码控制）+ `row_factory=Row`（返回字典）+ `PRAGMA foreign_keys=ON` + `PRAGMA busy_timeout=<timeout*1000>`（写锁等待与连接 timeout 同口径，默认 5s）；非内存库自动开 `PRAGMA journal_mode=WAL`（并发读写更友好，呼应 B9 待办）；连接失败抛 `DatabaseConnectionError`。
  - **基础执行**：`execute/executemany/executescript/query` 四个方法是最底层通道，自动管理事务边界（在 `with db:` 事务内不 commit，由 `__exit__` 统一提交/回滚；不在事务内则单次语句自动 commit，失败自动 rollback）。SQLite 异常统一收口为 `DatabaseOperationalError`（`DatabaseError` 子类，仍可被 `except ValueError` 捕获）抛出，让上层不用 `import sqlite3` 就能 catch。
  - **DDL**：`table_exists` 查 `sqlite_master`；`create_table` 按 `isinstance(columns, Schema/list/dict)` 分派；`ensure_table` 就是 `create_table(if_not_exists=True)`；`ensure_schema` 先走 `table_exists` 判断，不存在就 `_create_table_from_schema`，已存在且 `migrate=True` 则 PRAGMA 拿现有列集合逐列 ALTER ADD；索引语句始终幂等执行；`drop_table` 默认 IF EXISTS。
  - **CRUD**：insert 返回 lastrowid；select 的 where 用命名占位符 `:w_k` 防冲突，order_by 走 `^[A-Za-z_][A-Za-z0-9_]*(\s+(ASC|DESC))?$` 白名单校验；update 用 `:set_k / :w_k` 前缀区分 set 和 where 参数避免冲突；delete **必须传 where**，否则抛 `DatabaseValidationError`（防止 `DELETE FROM users` 清空全表的低级错误）。
  - **统计与批量写入**：`count(table, where=None)` 做等值条件计数（where 为 None 返回表总行数，值为 None 时按 IS NULL 处理）；`batch_insert(table, rows)` 先校验各行字段一致，再用 `executemany` 参数化批量插入，空列表返回 0。
  - **条件写入**：`upsert` 编译 `INSERT ... ON CONFLICT(keys) DO UPDATE SET col = excluded.col`（`update_columns` 为空则 `DO NOTHING`）；`update_where / delete_where` 复用 `_compile_filters` 把 Filter 编译成 WHERE（与读侧共用 `_compile_filter`），SET 值默认参数化、显式 `SqlExpr` 才原样拼入；两者都要求 `filters` 非空防误改全表。
  - **事务**：`__enter__` 显式 `BEGIN IMMEDIATE` 把 `_tx_depth` 加一（最外层才真正 `BEGIN`，内层嵌套只递增计数器以加入父事务）；`__exit__` 根据是否异常 commit/rollback，finally 里把 `_tx_depth` 归零；`_in_transaction` 是 `_tx_depth > 0` 的只读 property，不保存独立状态，杜绝标志与真实状态失同步；`__del__` 尝试 close，不抛异常。
- **为什么这么写**：
  - **row_factory=Row + 手动 dict(row)**：标准库 sqlite3 的 Row 是tuple-like，直接给上层用不方便；转 dict 让上层 `row["name"]` 访问，对齐记忆模块现有数据结构。
  - **dict 列定义也要 check_ident**：dict 写法最宽松，更要做表名/列名白名单，否则一行 `"name; DROP TABLE x;--": "TEXT"` 就能注入（虽然参数化只能防值注入、防不了标识符注入，这也是为什么必须做 check_ident）。
  - **migrate 让 sqlite3 自己报错**：不支持的场景（如 ADD COLUMN NOT NULL 无默认值）直接让 sqlite3 抛错向上冒泡，比静默跳过或瞎猜默认值安全——调用方看到报错就知道该写显式迁移脚本。
  - **execute 包成模块异常**：不让上层 import sqlite3 去 catch sqlite3.Error，贯彻"上层不依赖驱动"的分层原则；所有数据库错误都是 `DatabaseError`（继承 `ValueError`）的子类，消息里能看到原始 SQL，便于排查。
- **你要改的时候最该看的地方**：加新的便捷方法（如 `count / exists / upsert`）时，先想清楚是不是能通过组合 `query/execute` 实现，如果只是个别业务场景需要，加在业务层不要加到公共客户端里；加新后端时，直接对照 `DatabaseClientBase` 新增一个 `xxx_client.py`，不要在 SQLiteClient 里塞方言判断；改谓词编译时记住**读侧 `_compile_query` 与写侧 `_compile_filters` 都收敛到 `_compile_filter`**，改一处要同时回归 `select_where / update_where / delete_where`。

### 7.6 `vector_query.py` —— 向量元数据过滤模型（向量侧的 `query.py`）

```python
class VectorFilter(BaseModel):
    field:   Optional[str] = None
    op:      VectorFilterOp
    value:   Optional[Any] = None
    filters: Optional[list["VectorFilter"]] = None
```

- **负责什么**：给向量侧提供"声明式元数据过滤"的数据模型，地位对应关系型的 `query.py`。一个 `VectorFilter` 兼做两种角色——单个叶子条件（`field + op + value`）或 AND/OR 逻辑组合（`op=AND/OR + filters`，子条件可继续嵌套）。
- **关键实现**：`VectorFilterOp` 的枚举值就是 Chroma 的 `$` 操作符（`$eq / $ne / $gt / $gte / $lt / $lte / $in / $nin / $and / $or`）；`VectorWhere = Union[dict, VectorFilter, None]` 让调用方既能传模型、也能直接传 Chroma 原生 where dict。
- **为什么这么写**：**不为复用而强行统一**。关系型 `Filter` 支持 LIKE / IS NULL、当前是平铺 AND；向量侧支持 AND/OR 嵌套、没有 LIKE / IS NULL。两者谓词集合不同，硬合并会让两边互相绑手绑脚，保持两个平行模型、命名风格一致即可。

### 7.7 `vector_client_base.py` —— 向量抽象基类

```python
class VectorClientBase(ABC):
    @abstractmethod
    def create_collection(self, name, *, dimension=None, metric="cosine"): ...
    @abstractmethod
    def add(self, collection, ids, embeddings=None, documents=None, metadatas=None) -> int: ...
    @abstractmethod
    def search(self, collection, query_embedding, *, n_results=10, where=None, include_embeddings=False): ...
```

- **负责什么**：定义所有向量后端（Chroma / 未来 FAISS / Milvus / pgvector）必须实现的最小接口集——连接（3）+ 集合管理（4）+ 写入/删除/取/检索/计数（5）共 **12 个方法**。
- **关键实现**：
  - 命名与 `DatabaseClientBase` 平行（"动词 + 对象"，集合方法对齐表方法），降低记忆成本；
  - **安全约定**：`delete` 的 `ids` 与 `where` 至少要有一个非空，空条件禁止删全量（与 `SQLiteClient.delete_where` 同一条安全线）；同时传 `ids` 与 `where` 时**以 ids 为准**，不依赖各后端未文档化的混用行为；
  - `get` 暴露 `include_embeddings / include_documents / include_metadatas` 三个开关（默认不返回向量以省内存/流量）。
- **为什么这么写**：不把事务 / DDL 迁移 / 外键这类关系型专有能力塞进向量基类（向量库普遍不支持）；两套基类平行之后，一个后端可以**同时继承两者**（Postgres + pgvector），上层 `isinstance` 即判断能力。**你要接新向量后端时，第一个要照着实现的就是这个类。**

### 7.8 `chroma_client.py` —— Chroma 向量后端（首个向量实现）

```python
class ChromaClient(VectorClientBase):
    def __init__(self, path=None, *, host=None, port=8000, ssl=False,
                 headers=None, tenant=None, database=None,
                 settings=None, add_chunk_size=1000):
        try:
            import chromadb                       # 延迟导入：核心依赖里不含 chromadb
        except ImportError as e:
            raise DatabaseDriverMissingError("使用 ChromaClient 需要先安装 chromadb 可选依赖：\n"
                              "    pip install fast-agents[chroma]") from e
        if (path is None) == (host is None):      # path / host 二选一
            raise DatabaseValidationError("ChromaClient 必须在 path（本地）与 host（远端）中二选一")
```

- **负责什么**：database 模块**第一个向量后端**，把 chromadb SDK 封装成与 `SQLiteClient` 同风格——集合名白名单校验、空条件删除拦截、对外返回统一摊平为 `list[dict]`（不向外泄露 SDK 的 dict-of-lists 原始结构）。两种连接模式：
  - **本地持久化**：`ChromaClient(path="./chroma_data")` → `chromadb.PersistentClient`，向量落盘到该目录、重启可恢复，适合单机 / 开发；
  - **远端服务**：`ChromaClient(host="127.0.0.1", port=8000)` → `chromadb.HttpClient`，连独立启动的 Chroma Server，适合生产 / 多进程共享；`ssl / headers / tenant / database` 均可透传。
- **关键实现（也是相对"最小可用版"的优化点）**：
  1. **少一次 RPC —— `get_or_create_collection` 走 SDK 原生幂等接口**：原实现「先 `collection_exists` 再 `create_collection`」要两次往返，现直接调 `client.get_or_create_collection(...)`，一次完成「有则取、无则建」；`collection_exists` 命中本地 collection 缓存时直接返回，不再每次发请求。
  2. **`delete` 精确计数 + 少一次全量扫描**：改用 `col.get(ids=... / where=..., include=[])` 先拿到目标 id 数再删除，**直接返回精确条数**；不再「删前 `count` 一次、删后 `count` 一次」两次全量统计。并把 `ids` / `where` 语义固定为「传了 `ids` 就按 id 删、否则按 `where` 删」。
  3. **`dimension` 真正生效**：Chroma 本身**不强制向量维度**（空集合写入任意维度都会成功）。本实现建集时把维度写进集合 metadata（键 `dimension`，可跨进程读回），`add` / `search` 时做**客户端侧维度校验**；对未声明维度的集合，按**首次写入**的维度推断并记住。维度不一致在写入/检索前就报中文错误，而不是等 SDK 抛晦涩异常。
  4. **大批量写入自动分块**：`add` 超过 `add_chunk_size`（默认 1000）时自动分批 upsert，避免一次提交撞上 Chroma/SQLite 单批上限而**整体失败**；返回仍是总条数。
  5. **默认关闭匿名遥测**：未显式传 `settings` 时用 `Settings(anonymized_telemetry=False)`，不向外部发送使用统计。
  6. **健壮性补强**：`add` 校验 ids 批内去重（Chroma 内部会抛 `DuplicateIDError`，提前拦下给中文错误）与批内 embedding 维度一致；`get` 校验 `limit` 为正整数（Chroma 会忽略 `limit=0`）；`connect` 加双检锁避免多线程并发首次调用重复建 client；清理结果摊平函数里的死参数。
  7. **兜底修掉一个真实缺陷**：对**已存在但 metadata 为空**的集合调用 `get_or_create_collection`，Chroma 0.6.x 合并 metadata 会抛 `TypeError: 'NoneType' object is not iterable`，实现里捕获后回退 `get_collection`，不再崩。
- **为什么这么写**：
  - **可选依赖放 extras**：`chromadb` 不进核心依赖（`pyproject.toml` 的 `[project.optional-dependencies] chroma = ["chromadb>=0.4,<1.0"]`），保持"零重型依赖"招牌；未装时构造 `ChromaClient` 抛清晰 `DatabaseDriverMissingError`，关系型能力完全不受影响。
  - **集合名白名单对齐 Chroma SDK 真实规则**：`^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$`——3–63 字符、首尾必须是字母或数字、中间允许字母/数字/下划线/连字符（实测 SDK 拒绝单字符名与以下划线/连字符结尾的名），把非法名挡在客户端而非直达 SDK。
  - **过滤编译集中**：`_compile_vector_filter` 把 `VectorFilter` 编译成 Chroma 的 `where` 字典；同时接受**已是 dict** 的输入原样透传，给熟悉 Chroma 原生 where 格式的高级用户留口子。
- **你要改的时候最该看的地方**：新增向量后端（FAISS / Milvus / pgvector）时对着 `VectorClientBase` 写一个新的 `xxx_client.py`，不要在 `ChromaClient` 里堆后端判断；改维度校验时注意三个入口（`create_collection` 写 metadata、`add` 校验、`search` 校验）要一致；`_flatten_get_result` / `_flatten_query_result` 是"对外输出统一 dict"的唯一出口，改返回结构只能改这里。

---

### 7.9 `errors.py` —— 模块异常体系（8 类，三后端共用）

```python
class DatabaseError(ValueError):                                # 基类
# 三大层
class DatabaseConnectionError(DatabaseError): ...               # 连接层
class DatabaseOperationalError(DatabaseError): ...              # 执行层
class DatabaseValidationError(DatabaseError): ...               # 校验层
# 四个具体场景
class DatabaseIdentifierError(DatabaseValidationError): ...     # 标识符非法
class DatabaseIntegrityError(DatabaseOperationalError): ...     # 约束冲突
class DatabaseQueryError(DatabaseOperationalError): ...         # 查询/写入失败
class DatabaseDriverMissingError(DatabaseConnectionError): ...  # 可选驱动未安装
```

- **负责什么**：把 database 对外抛出的错误从「散落的裸 `ValueError`」收口成「基类 + 3 层 + 4 场景」共 8 个类，上层可按错误类型精确 `except`（该重连的、该重试的、该改参数的、该换用 upsert 的区分开）。
- **关键实现**：`DatabaseError` **刻意继承 `ValueError`**——历史上 `SQLiteClient` 把底层 `sqlite3.Error` 一律包成 `ValueError` 抛出，memory 层等既有调用方与一批 `pytest.raises(ValueError)` 都依赖这个谱系，沿用即可零破坏。三个后端全部按语义接入：连接失败 → `DatabaseConnectionError`；`execute / executemany / executescript / query` 执行失败 → `DatabaseOperationalError`；表名/列名/集合名非法 → `DatabaseIdentifierError`；唯一键/主键冲突、集合已存在 → `DatabaseIntegrityError`；集合不存在、写入失败 → `DatabaseQueryError`；可选驱动未装 → `DatabaseDriverMissingError`；各类入参校验 → `DatabaseValidationError`。调用方把类型传错（如把 int 当表名）仍抛原生 `TypeError`，不纳入本体系。
- **为什么这么写**：参考 SQLAlchemy Core 的 `DBAPIError` 分层（ConnectionError / OperationalError / IntegrityError / ProgrammingError 分离）。**不把底层库原生异常直抛上层**——否则上层要 `import sqlite3` / `import pymysql`，换 Postgres 后又冒出 `psycopg2.Error`，分层就破了。异常类集中放 `errors.py`，对齐 `mcp/errors.py`、`tools/tool_exception.py` 的组织方式。
- **你要改的时候最该看的地方**：新增后端时**复用**这套异常，不要每个后端自定义一套；新增错误分类时在 `errors.py` 加子类、维持 `ValueError` 谱系不变（否则破坏既有 `except ValueError`）。

### 7.10 `mysql_client.py` —— MySQL 客户端（可选后端，接口对齐 SQLite）

```python
class MySQLClient(DatabaseClientBase):
    def __init__(self, host="127.0.0.1", port=3306, user="root",
                 password="", database=None, charset="utf8mb4", timeout=5.0):
        _require_pymysql()      # 延迟导入，未装抛 DatabaseDriverMissingError
        self._connect()
    # 方言层：反引号标识符 / %s 占位符 / TYPE_TO_MYSQL / ON DUPLICATE KEY UPDATE
```

- **负责什么**：`SQLiteClient` 的可选对等实现——接口 100% 对齐（同一套 `connect / DDL / CRUD / select_where / upsert / update_where / delete_where / with 事务`），上层改一行构造即可把 SQLite 换成 MySQL。
- **关键实现**：
  - **方言适配**：标识符用反引号（`_quote_ident_mysql`）；占位符统一 `%s`；`TYPE_TO_MYSQL`（`TEXT→TEXT`、`INTEGER→INT`、`REAL→DOUBLE`、`BLOB→LONGBLOB`、`BOOLEAN→TINYINT(1)`、`VARCHAR→VARCHAR(255)`、`TIMESTAMP→DATETIME`）；INTEGER 主键 `id` 自动补 `AUTO_INCREMENT`。
  - **建表 / 索引幂等**：`ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`；MySQL 无 `CREATE INDEX IF NOT EXISTS`，改为先查 `information_schema.statistics` 再建；补列查 `information_schema.columns`。
  - **Upsert**：`ON DUPLICATE KEY UPDATE`（SQLite 走 `ON CONFLICT DO UPDATE`），`update_columns` 语义两边一致。
  - **事务**：沿用 `_tx_depth` 嵌套计数（与 SQLite 同构）；`OFFSET` 无 `LIMIT` 时补 `LIMIT 18446744073709551615`（MySQL 语法要求）。
  - **连接细节**：`init_command` 设 `SET SESSION sql_mode='STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'`；`conn` 属性访问时 `ping(reconnect=True)` 自动重连。
  - **异常与 SqlExpr**：连接失败 `DatabaseConnectionError`、执行失败 `DatabaseOperationalError`、入参 `DatabaseValidationError`、未装 pymysql `DatabaseDriverMissingError`；`SqlExpr` 直接复用 `sqlite_client` 的同一类，跨后端 `isinstance` 一致。
- **为什么这么写**：把方言差异全部关在 `mysql_client.py` 内部，`DatabaseClientBase` 契约与 `schema.py` 声明模型一行不改——这正好验证了 §7.3「拼 SQL 抽成纯函数 + 各后端分别适配」的设计。**你要接 Postgres 时，照着本文件写 `postgres_client.py` 即可。**
- **你要改的时候最该看的地方**：改类型映射看 `TYPE_TO_MYSQL`；改索引 / 补列幂等看 `_list_indexes` / `_list_columns` 对 `information_schema` 的查询；改 upsert 看 `ON DUPLICATE KEY UPDATE` 分支；与 SQLite 行为不一致处（如 OFFSET 补 LIMIT）要在两边同时回归。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents

# 依赖（关系型零新增依赖，装好仓库 requirements 即可；向量后端另需可选依赖）
pip install -r requirements.txt
# pip install fast-agents[chroma]     # 仅当要用 ChromaClient 时才装

# 跑测试（121/121 全过才算装好；未装 chromadb 时 17 项向量端到端自动 skip，显示 104 passed, 17 skipped 也属正常）
python test\test_database.py

# 跑示例（前四个为纯离线示例，无需任何外部服务）
python examples\database\sqlite_usage.py
python examples\database\create_table_usage.py
python examples\database\query_filter_usage.py
python examples\database\conditional_write_usage.py
python examples\database\chroma_usage.py    # 需先 pip install fast-agents[chroma]
python examples\database\mysql_usage.py     # 需先 pip install fast-agents[mysql] 与可连的 MySQL 环境
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

当前（v1.1.0）ReActAgent **没有直接持有 DatabaseClient**；database 是通过 memory / trace 模块间接被使用的——`SQLiteMemoryStore / SQLiteSessionStore / SQLiteProfileStore` 与 `trace/sqlite_store.py` 的 `SQLiteStore` 均已基于 `SQLiteClient`（不再各自手写 sqlite3 连接与 `CREATE TABLE`），做记忆 / 轨迹落盘。

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

预期输出末尾为 `121 passed`（具体数字随版本更新）；若当前环境未装 chromadb，则显示 `104 passed, 17 skipped`——被跳过的 17 项是向量端到端用例（用 `pytest.importorskip("chromadb")` 标记），装好可选依赖后即为全绿。任何一个 FAILED 都说明环境有问题（Python 版本过低 / 文件被占用 / 磁盘只读等），按报错定位即可。

### 8.6 向量后端快速上手（Chroma，可选依赖）

向量能力是 database 模块的**第二套平行后端**，只在需要语义检索时才装可选依赖：

```powershell
pip install fast-agents[chroma]          # 等价于 pip install "chromadb>=0.4,<1.0"
python examples\database\chroma_usage.py
```

最小可运行示例（本地持久化目录，跑完删目录；零 API Key）：

```python
import tempfile
from fast_agents.database import ChromaClient, VectorFilter, VectorFilterOp

with tempfile.TemporaryDirectory() as d:
    db = ChromaClient(path=d)              # 本地持久化；远端用 ChromaClient(host="127.0.0.1", port=8000)
    db.get_or_create_collection("faq", dimension=2, metric="cosine")   # 幂等建集，声明 2 维

    db.add(
        "faq",
        ids=["q1", "q2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["你好", "再见"],
        metadatas=[{"lang": "zh"}, {"lang": "zh"}],
    )

    print(db.count("faq"))                                  # 2
    print(db.get("faq", ids=["q1"])[0]["document"])          # 你好

    hits = db.search("faq", [0.9, 0.1], n_results=1)         # 近邻检索
    print(hits[0]["id"], hits[0]["distance"])                # q1 <distance>

    db.delete("faq", where=VectorFilter(field="lang", op=VectorFilterOp.EQ, value="zh"))
    db.drop_collection("faq")
    db.close()
```

> 未装 chromadb 时构造 `ChromaClient` 会抛 `DatabaseDriverMissingError` 并提示安装命令；不装也**不影响关系型的任何用法**（`from fast_agents.database import SQLiteClient` 照常可用）。

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
| `upsert(table, data, conflict_keys, update_columns=None)` | data: dict；conflict_keys: str \| list[str]（须都在 data 中）；update_columns: list[str] \| None | int（rowcount） | **插入或更新**（`ON CONFLICT DO UPDATE`）；`update_columns` 不传则更新除冲突键外的全部列，传空列表则 DO NOTHING；值全参数化 |
| `batch_insert(table, rows)` | rows: list[dict]（各行字段须一致） | int（插入行数） | 批量插入；空列表返回 0；字段不一致抛 ValueError |
| `select(table, where=None, order_by=None, limit=None)` | where: dict（等值 AND）；order_by: str（白名单，支持 `"id DESC"`）；limit: int | `list[dict]` | **等值查询**的简洁 API；复杂过滤用 `select_where` |
| `select_one(table, where=None, order_by=None)` | 同上 | `dict \| None` | 查询单行，无结果返回 None |
| `select_where(query)` | query: `Query` | `list[dict]` | **谓词下推入口**：按声明式 Query 查多行，支持 10 种 `FilterOp`、AND 组合、排序、limit/offset；参数化防注入 |
| `select_one_where(query)` | query: `Query` | `dict \| None` | `select_where` 的单行版本（内部自动 `limit=1`，不修改传入 Query） |
| `update(table, data, where)` | data: dict（非空）；where: dict（非空） | int（rowcount，影响行数） | 更新行 |
| `count(table, where=None)` | where: dict（等值，值为 None 时按 IS NULL）；None 表示全表 | int | 统计行数 |
| `delete(table, where)` | where: dict（**必填**） | int（rowcount） | 删除行；**不传 where 直接抛 ValueError** 防止误删全表 |
| `update_where(table, values, query)` | values: dict（非空，值可为 `SqlExpr`）；query: `Query`（filters **必填**） | int（rowcount） | **写侧谓词下推**：按 Filter 条件批量更新；普通值参数化，`SqlExpr("n + 1")` 才原样拼入 |
| `delete_where(table, query)` | query: `Query`（filters **必填**） | int（rowcount） | **写侧谓词下推**：按 Filter 条件批量删除；空 filters 抛 ValueError 防误删全表 |

#### 事务

```python
with db:
    db.insert(...)
    db.update(...)
# 正常退出 → commit；抛异常 → rollback
```

`__enter__ / __exit__` 实现；事务状态的唯一真实来源是 `_tx_depth`（嵌套深度计数器），`_in_transaction` 是其只读派生属性，均在 finally 里归零/复位。

### 9.2 数据模型类速查

| 类 | 构造常用字段 | 用途 |
|---|---|---|
| `ColumnType` | 枚举：TEXT/INTEGER/REAL/BLOB/BOOLEAN/VARCHAR/TIMESTAMP | 列类型，与具体 SQL 方言解耦 |
| `Column(name, type, primary_key=False, nullable=True, default=None, unique=False, index=False, length=None)` | — | 声明式列定义 |
| `Index(name, columns, unique=False)` | — | 索引（单列索引优先用 `Column(index=True)`，复合/唯一索引用这里） |
| `Schema(table_name, columns, indexes=None)` | — | 完整表结构，`ensure_schema` 的入参 |
| `SortOrder` | ASC / DESC | 查询排序方向 |
| `FilterOp` | EQ/NEQ/GT/GTE/LT/LTE/LIKE/IN/NOT_NULL/IS_NULL | 过滤操作符（由 `select_where` 编译，共 10 种） |
| `Filter(field, op, value=None)` | — | 单个过滤条件；IN 时 value 必须是非空 list/tuple；IS_NULL/NOT_NULL 无需 value |
| `Query(table_name, filters=None, sort_by=None, sort_order=ASC, limit=None, offset=None)` | — | 查询对象；`sort_by` 支持单字段 str 或多字段 list[str]；`select_where` / `select_one_where` / `update_where` / `delete_where` 的入参 |
| `SqlExpr(sql)` | sql: str（非空） | 写侧 SQL 表达式包装，仅 `update_where` 用；内容原样拼入 SQL，只传程序写死的固定片段 |

### 9.3 已废弃/暂未开放的写法

- 初版即同时提供 `select(where=dict)`（等值）与 `select_where(Query)`（谓词）两种写法，**没有**已废弃的旧 API；
- 以下能力**刻意不开放**，不要传参期望生效：
  - OR / NOT / 括号分组（多 Filter 当前固定 AND）；
  - JOIN / 子查询 / 聚合 / GROUP BY / HAVING；
  - `sort_by` 的按表达式排序（只支持列名；多列已支持，但 `ORDER BY random()` 这类要写裸 SQL）；
  - BETWEEN 语法糖（用 `GTE + LTE` 两个 Filter AND 表达，效果等价）。

需要这些能力时直接用 `db.query(sql, params)` 写裸 SQL，模块不假装自己是完整 ORM。

另外，`update_where` / `delete_where` 的 `query.filters` **必填**（空条件直接报错），这是安全约定而非能力缺失；要全表改删请显式写裸 SQL。

### 9.4 `ChromaClient(path=None, *, host=None, port=8000, ssl=False, headers=None, tenant=None, database=None, settings=None, add_chunk_size=1000)`

> 需先 `pip install fast-agents[chroma]`；`path`（本地）与 `host`（远端）**二选一**，都不给或都给都抛 `DatabaseValidationError`。

#### 连接 / 生命周期

| 方法 | 说明 | 返回 |
|---|---|---|
| `connect()` | 建立 PersistentClient 或 HttpClient（幂等、加锁） | None |
| `close()` | 清空缓存与 client 引用 | None |
| `ping()` | `heartbeat()` 连通性检查 | bool |

#### 集合管理（对应关系型的"表"）

| 方法 | 参数 | 说明 |
|---|---|---|
| `collection_exists(name)` | name: str | 集合是否存在（命中缓存直接返回） |
| `create_collection(name, *, dimension=None, metric="cosine")` | metric ∈ {cosine, ip, l2} | 建集并写 `hnsw:space`；`dimension` 写进 metadata；已存在抛 `DatabaseIntegrityError` |
| `get_or_create_collection(name, *, dimension=None, metric="cosine")` | — | **幂等建集**（走 SDK 原生接口，一次往返） |
| `drop_collection(name)` | — | 删集合，不存在静默 |

#### 写入 / 删除 / 取 / 检索 / 计数

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `add(collection, ids, embeddings=None, documents=None, metadatas=None)` | 三个内容字段至少一个非空，长度须与 ids 一致 | int（写入条数） | 按 id upsert（同名覆盖）；ids 去重、批内维度一致、与声明维度一致校验；超 `add_chunk_size` 自动分块 |
| `delete(collection, ids=None, where=None)` | ids / where 至少一个非空 | int（精确删除条数） | 传 ids 按 id 删、否则按 where 删；**空条件抛 `DatabaseValidationError`** 防误删全量 |
| `get(collection, ids=None, where=None, limit=None, include_embeddings=False, include_documents=True, include_metadatas=True)` | — | `list[dict]` | 精确取（非近邻）；`limit` 须为正整数 |
| `search(collection, query_embedding, *, n_results=10, where=None, include_embeddings=False)` | query_embedding: list[float] | `list[dict]` | 近邻检索，按距离升序；结果含 `distance`（`include_embeddings=True` 时含 `embedding`） |
| `count(collection, where=None)` | — | int | 无 where 走 `col.count()`；有 where 走 `get(include=[])` 计数 |

### 9.5 向量数据模型速查

| 类 | 构造常用字段 | 用途 |
|---|---|---|
| `VectorFilterOp` | 枚举：EQ/NE/GT/GTE/LT/LTE/IN/NIN/AND/OR（枚举值即 Chroma `$` 操作符） | 向量元数据过滤操作符 |
| `VectorFilter(field=None, op=..., value=None, filters=None)` | — | 叶子条件（field+op+value）或 AND/OR 组合（op+filters，≥2 个）；由 `chroma_client._compile_vector_filter` 编译为 Chroma where |
| `VectorWhere` | = `dict \| VectorFilter \| None` | 向量过滤入参类型别名；`None` 表示无条件 |
| `VectorClientBase` | 抽象基类 | 向量后端契约（12 个抽象方法），与 `DatabaseClientBase` 平行 |

### 9.6 异常体系速查（`errors.py`）

| 异常 | 继承 | 触发场景 |
|---|---|---|
| `DatabaseError` | `ValueError` | 模块异常基类，可统一兜底 |
| `DatabaseConnectionError` | `DatabaseError` | 建连 / 关连接失败（路径不可写、驱动不可用） |
| `DatabaseOperationalError` | `DatabaseError` | SQL 执行 / 查询失败（语法错、约束冲突、锁超时） |
| `DatabaseValidationError` | `DatabaseError` | 入参不合法（空数据、非法标识符、空 filters 防误删） |
| `DatabaseIdentifierError` | `DatabaseValidationError` | 表名 / 列名 / 集合名不合法 |
| `DatabaseIntegrityError` | `DatabaseOperationalError` | 主键 / 唯一键冲突、集合已存在 |
| `DatabaseQueryError` | `DatabaseOperationalError` | 集合不存在、查询 / 写入失败 |
| `DatabaseDriverMissingError` | `DatabaseConnectionError` | pymysql / chromadb 可选驱动未安装 |

```python
from fast_agents.database import DatabaseError, DatabaseOperationalError
try:
    db.execute(...)
except DatabaseOperationalError as e:   # 精确分流
    ...
except DatabaseError:                  # 或统一兜底（等价于 except ValueError）
    ...
```

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

5. **为什么 `execute/query` 把 sqlite3.Error 收口为模块异常（ValueError 谱系）？**
   上层业务模块（memory / trace）不应该 `import sqlite3` 去 catch `sqlite3.Error`，否则分层就破了——将来换 Postgres 后端就会冒出 `psycopg2.Error`，上层要改一串 except。统一收口到模块自己的异常体系（带原始 SQL 和错误消息），上层只需要 catch 标准异常即可。**现已落地**：`errors.py` 定义 `DatabaseError` 基类（继承 `ValueError`）+ 7 个语义子类（`DatabaseConnectionError` / `DatabaseOperationalError` / `DatabaseValidationError` 三大层 + `DatabaseIdentifierError` / `DatabaseIntegrityError` / `DatabaseQueryError` / `DatabaseDriverMissingError` 四个场景），SQLite / MySQL / Chroma 三后端全部按语义接入，底层驱动异常与入参校验错误不再裸抛（第十二章 J5）。

6. **为什么默认开 WAL 和 foreign_keys？**
   `PRAGMA foreign_keys=ON` 是 SQLite 默认关闭但几乎所有人都想开的设置（否则外键约束形同虚设）；`PRAGMA journal_mode=WAL` 让并发读写互不阻塞（B9 待办），这也是 J3 跨模块迁移任务里要给 memory 三个 SQLite 后端统一收口的设置（B9 随 J3 自然落地）——database 模块先以身作则。内存数据库不开 WAL（WAL 在 :memory: 上无意义）。此外 `_connect()` 里显式设 `PRAGMA busy_timeout = timeout * 1000`，让写锁等待与构造参数同口径（默认 5s），memory 迁移时传 `timeout=10.0` 即可对齐其原有 10s。

7. **为什么写侧谓词下推单独开 `update_where / delete_where`，而不改 `update / delete`？**
   既有 `update(table, data, where)` / `delete(table, where)` 只支持等值 where 且签名被上层依赖，直接改签名会破坏调用方。新增方法走 `Query` 谓词，与读侧 `select_where` 对称，新旧并存、各取所需——这也是本模块一贯的「能加不改」策略。

8. **为什么 `update_where / delete_where` 强制 filters 非空，而 `SqlExpr` 又要显式包裹？**
   两条都是「防手滑」：前者的痛点是「忘写条件 → 全表被改/被删」，与 `delete` 必须带 where 是同一条安全线；后者的痛点是「把外部输入拼进 SQL → 注入」，所以只有显式 `SqlExpr` 包装的内容才允许原样拼入，普通值一律参数化。宁可多写一点，也不给「图方便」留口子。

9. **为什么向量能力另起一套平行抽象，而不塞进 `DatabaseClientBase`？**
   关系型与向量型的能力模型差异太大——前者是表/列/事务/精确谓词/SQL，后者是集合/向量/元数据/近邻检索。硬塞进一个基类会强迫所有已有关系型后端实现一堆用不上的向量方法，破坏"零破坏性重构"。改成两套平行基类后，谁需要向量能力谁再继承 `VectorClientBase`（如未来的 Postgres + pgvector 同时继承两个），上层用 `isinstance` 判断即可，已有后端一行不改。

10. **为什么 `chromadb` 走可选依赖（extras）而不是核心依赖？**
   仓库招牌是"零重型依赖、基于 OpenAI 原生 API"；`chromadb` 带一堆传递依赖（hnswlib / onnxruntime 等），装进核心依赖会显著抬高所有使用者的安装成本。放到 `[project.optional-dependencies] chroma` 后，不用向量的人 `pip install fast-agents` 依旧轻量，用的人 `pip install fast-agents[chroma]` 即可；`ChromaClient` 内部延迟导入、未装时抛清晰 `DatabaseDriverMissingError`。

11. **为什么向量过滤模型（`vector_query.py`）不复用关系型的 `Filter`？**
   两者谓词集合不同：向量侧支持 AND/OR **嵌套**、不支持 `LIKE` / `IS NULL`（Chroma metadata 过滤本身没有这两个能力）；关系型 `Query` 目前是平铺 AND、有 LIKE/IS NULL。强行复用会让两边都被对方边界绑住。保持两个模型平行、命名风格一致（`Filter`↔`VectorFilter`、`FilterOp`↔`VectorFilterOp`），各自演进互不影响。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| `DatabaseIdentifierError: 非法表名/列名: xxx` | 表名/列名不符合 `^[A-Za-z_][A-Za-z0-9_]*$`，含中文/横线/空格/点号/以数字开头都会被拒。改名字，或在需要时走 `db.execute(...)` 自己拼（不推荐） |
| `DatabaseValidationError: delete 必须传 where 防止误删全表` | 写了 `db.delete("users")` / `db.delete("users", {})`。真要删全表用 `db.execute("DELETE FROM users")`；按条件删必须传非空 where dict |
| `DatabaseValidationError: insert 数据不能为空` / `update 需要 data 和 where` | 空 dict 调用 insert/update。检查调用方 data 是不是被覆盖成 `{}` |
| `database is locked` | SQLite 并发写冲突。一是确认所有连接都开了 WAL（SQLiteClient 默认开），二是加大 `timeout` 参数（默认 5 秒，`_connect()` 会同步设为 `PRAGMA busy_timeout`），三是检查是否有进程泄漏了连接未 close；多进程高并发写入场景建议迁移到 Postgres（MySQL 后端已支持，第十二章） |
| `ALTER TABLE ADD COLUMN` 报错 "Cannot add a NOT NULL column with default value NULL" | migrate 补列时新加了 `nullable=False` 且没给 default 的列，SQLite 不允许。要么给 `default=...`，要么把列声明为 `nullable=True` 再应用层校验；老数据需要非空的，先加可空列 → 回填 UPDATE → 再单独脚本改约束（SQLite 不支持后加 NOT NULL 约束） |
| `UNIQUE constraint failed: xxx.yyy` | 插入/更新违反唯一约束。若要「有则更新、无则插入」，直接用 `db.upsert(table, data, conflict_keys="yyy")`（`ON CONFLICT DO UPDATE`），不要手拼 SQL；注意 `conflict_keys` 对应的列必须有 PRIMARY KEY / UNIQUE 约束，否则 SQLite 会报 "ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint" |
| `executeSQL失败: ...SQL: ...` 但看不出哪错 | 异常消息里附带了完整 SQL，复制到 DB Browser for SQLite / sqlite3 CLI 里单跑一遍即可定位。params 没拼错的话，大概率是列名/表名写错、类型不匹配 |
| 多次实例化 `SQLiteClient("app.db")` 互相阻塞 | 同一个 .db 文件开了多个连接并发写会撞锁。推荐**整个进程内共享一个 client 实例**（模块级单例或用依赖注入），不要在每次函数调用里新建 |
| `Column(index=True)` 没有看到建索引 | 主键列 / UNIQUE 列 SQLite 已自动建索引，模块会跳过避免冗余（`column_index_stmt` 里显式判断）。不是 bug |
| BOOLEAN 列查出来是 0/1，不是 True/False | 是设计如此（SQLite 无原生 BOOLEAN）。业务层 `bool(row["flag"])` 转一下；或用 `detect_types=sqlite3.PARSE_DECLTYPES` + 注册适配器（当前没注册，保持简单） |
| TIMESTAMP 列存什么格式 | 存 ISO 8601 字符串（对齐 memory 模块的 `updated_at/created_at` 习惯），不要存 datetime 对象——SQLite 没有原生时间类型，字符串排序等价于时间排序（只要是 ISO 格式） |
| `DatabaseDriverMissingError: 使用 ChromaClient 需要先安装 chromadb 可选依赖` | 没装可选依赖。执行 `pip install fast-agents[chroma]`（或 `pip install "chromadb>=0.4,<1.0"`）。不装不影响关系型能力 |
| `DatabaseIdentifierError: 非法集合名: 'ab'；需 3-63 字符...` | Chroma 集合名约束：3–63 字符、**首尾必须是字母或数字**、中间允许字母/数字/`_`/`-`。单字符名（`'a'` / `'t'`）与以下划线/连字符结尾的名都会被拒 |
| `DatabaseValidationError: embedding 维度(x) 与集合 'y' 声明的维度(z) 不一致` | 建集时声明了 `dimension`，写入/检索向量必须与之一致（Chroma 空集合本身不校验，是客户端补的位）。要么对齐维度，要么建集时不声明 |
| `DatabaseValidationError: delete 的 ids 和 where 不能同时为空（防止误删全量）` | 向量版 `delete` 与关系型 `delete` 同一条安全线：不带条件不允许删全量。要清空集合请用 `drop_collection` |
| `DatabaseValidationError: 同一批 embeddings 维度不一致：[...]` | 一次 `add` 的 embeddings 必须同维度；分批前先对齐 |
| `TypeError: 'NoneType' object is not iterable`（`get_or_create_collection` 时） | Chroma 0.6.x 对「已存在但 metadata 为空」的集合合并 metadata 的缺陷；本实现已捕获回退，若绕过 client 直接用 SDK 需自行注意 |

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
| `test/test_database.py` | **121 / 121**（2026-11-05；其中 17 项向量端到端需 chromadb，未装时自动 skip） | 连接与基本 CRUD / 字典返回 / 事务提交 / 事务回滚 / 异常后可继续使用 / select order_by 与 limit / 列名注入防护 / 非法 order_by 拦截 / delete 必须传 where / 内存库 / 文件库持久化 / close 后自动重连 / **dict 方式建表** / **list[Column] 方式建表** / **Schema 方式建表** / table_exists 幂等 / 列级索引自动建 / 复合索引 / 唯一约束 / default 值类型映射（布尔/数字/字符串/None） / BOOLEAN→0/1 / VARCHAR→TEXT / TIMESTAMP→TEXT / ensure_table 幂等重复调用 / ensure_schema 重复调用幂等 / **migrate=True 补列** / migrate=False 不补列 / drop_table / 手写 CREATE TABLE 仍可用（向后兼容） / 上下文管理器嵌套报错语义 / 批量 executemany / executescript 多语句 / 标识符非法名报错 / **谓词下推**：EQ/NEQ/GT/GTE/LT/LTE 六类比较 / LIKE 模糊 / IN 集合匹配（含空列表/非列表值拦截） / IS NULL / IS NOT NULL / 二元操作符漏 value 报错 / 多 Filter AND 组合 / 同字段多 Filter 参数名不冲突 / sort_by + ASC/DESC + limit + offset 组合 / 仅 offset 无 limit 自动补 LIMIT -1 / limit/offset 负数拦截 / 字段名非法注入拦截 / value 参数化（SQL 片段不会被执行） / select_one_where 单行返回与 None / select_one_where 不污染原 Query 对象 / 无 filters 全量返回 / select_where 入参类型检查 / 原 select(where=dict) 等值写法继续可用 / **count**：总行数 / 带条件计数（含 IS NULL）/ 非法表名拦截 / **batch_insert**：基础批量插入 / 空列表返回 0 / 字段不一致拦截 / 非法表名拦截 / **upsert**：插入新行 / 冲突更新且不新增行 / update_columns 限定列 / conflict_keys 缺失拦截 / 非法表名拦截 / **update_where**：比较谓词 GTE / IN 谓词 / SqlExpr 自增表达式 / 空 filters 拦截 / 值参数化防注入 / **delete_where**：比较谓词 LT / IN 谓词 / 空 filters 拦截且不误删 / **多列排序**：sort_by list 升序与降序 / 单字段 str 排序兼容 / 空 sort_by 列表拦截 / busy_timeout 与 timeout 同口径 / SqlExpr 空表达式拦截 / **向量抽象层**：VectorClientBase 12 个抽象方法齐全 / VectorFilter 叶子与 AND 组合编译为 Chroma where / dict 透传 / AND 组合至少 2 子条件拦截 / **ChromaClient**：path 与 host 二选一拦截 / 非法集合名拦截 / 本地持久化全链路（建集→写→按 id 取→近邻 search→where 过滤→按条件删→幂等建集→drop 幂等）/ add 长度一致性与空 ids 拦截 / search 空向量与 n_results=0 拦截 / 带 where 的 count / 非法 metric 拦截 / 建集把 dimension 写入 metadata 并拒绝重复建集 / get_or_create 幂等且保留维度 / add ids 去重拦截 / 批内维度不一致拦截 / 与声明维度不符拦截 / search 维度不符拦截 / add_chunk_size 分块写入 / get limit 校验与 include 开关 / delete ids 优先且精确计数 / 无 metadata 集合的 get_or_create 兜底 / **异常体系**：DatabaseError 继承 ValueError / 8 类继承链（3 层 + 4 场景）/ 底层 SQL 错误映射 DatabaseOperationalError / 入参校验映射 DatabaseValidationError / 非法标识符映射 DatabaseValidationError / 连接失败映射 DatabaseConnectionError / 新异常仍被 except ValueError 捕获 / 顶层 fast_agents 包导出异常 | `python test\test_database.py` |

**关键回归断言**（这几条锁住了核心设计承诺，改代码时必须保留）：

- `test_delete_without_where_raises` —— 锁死「delete 必须带 where」的安全约束，不能因为"方便"去掉检查；
- `test_sql_injection_via_identifier_blocked`（原命名为 `test_illegal_ident_rejected`）—— 锁死 `check_ident` 标识符白名单能拦截 `; DROP TABLE` / `--` / 数字开头 / 空格 / 引号等拼接注入；
- `test_raw_create_table_still_works` —— 锁死向后兼容：上层已有的手写 `db.execute("CREATE TABLE ...")` 必须继续可用，不能因为加了声明式建表就禁止裸 SQL；
- `test_migrate_adds_missing_columns` / `test_migrate_false_does_not_add_columns` —— 两条一起锁住 migrate 的语义：默认不动结构，显式 True 才补列；
- `test_transaction_rollback_on_error` —— 锁死事务原子性，异常时必须回滚干净；
- `test_boolean_mapped_to_integer` / `test_varchar_mapped_to_text` / `test_timestamp_mapped_to_text` —— 锁死 TYPE_TO_SQLITE 映射，未来加多后端时 SQLite 这侧的映射不能悄悄变；
- `test_select_where_value_is_parameterized` —— 锁死谓词下推的值一定走参数化，`value="Alice' OR '1'='1"` 不会被当成 SQL 片段执行；
- `test_select_where_invalid_field_rejected` —— 锁死 select_where 的字段名同样走 `check_ident` 白名单，不能靠 Filter.field 拼接注入；
- `test_select_one_where` + 末尾 `q.limit is None` 断言 —— 锁死 select_one_where 的 limit=1 是内部行为，不污染调用方传入的 Query 对象（避免共享 Query 复用时被意外改 limit）；
- `test_select_where_still_works` —— 锁死原 dict 等值 API 在引入 select_where 后**不被破坏、不被替换**。

**全绿基线**：在 Windows PowerShell + Python 3.10+（仓库 `.python-version` = 3.10）+ 干净 venv（只装 requirements.txt）环境下，`python test\test_database.py` 必须输出末尾显示 `121 passed`（装好 `fast-agents[chroma]` 后）；未装 chromadb 时显示 `104 passed, 17 skipped` 也属正常基线。任何一条 FAILED 都视为环境或代码问题，不应合并。

**跨模块回归**：database 改动后，除了本模块 121 项，还要跑一遍 `test_memory.py` 里涉及 SQLiteMemoryStore 的用例（memory / trace 已基于 SQLiteClient 的 ensure_schema 建表，共用 sqlite3 驱动，方言/默认 PRAGMA 变动可能影响），以及 `test_context.py` / `test_react_agent.py` 全套，确认没有隐式依赖被破坏。

**当前测试与实现的已知差异**（如实写明，不粉饰）：

- `connect()` 作为基类接口目前只在 `__init__` 和 `close()` 后被调用，没有独立单测；由于 `__init__` 已经调用 `_connect`，功能等价，风险低；
- `select_where` 的 `sort_by` 已支持单字段与多字段；「按表达式排序」（如 `ORDER BY random()`）**没有实现**，这是刻意为之的能力边界（见 §5.4 / §9.3）；
- OR / NOT / 括号分组 / JOIN / 聚合 / GROUP BY / 子查询**没有测试也不打算做**，保持零重型依赖；
- 向量侧的 17 项 Chroma 端到端用例依赖可选依赖 `chromadb`，在未安装的环境里会被 `pytest.importorskip` 跳过（不是失败）；纯抽象/编译校验的 6 项向量用例不需要 chromadb，任何环境都会真跑。

---

## 十二、下一步演进路线

按优先级排（与 `docs/演进路线图.md` 编号对齐；database 模块在路线图里独立为 **J 模块（持久化层）**）。

### 12.1 初版已交付

- **初版**（v1.0.0，2026-11-03；**v1.1.0，2026-11-05** 追加 MySQL 后端 + 异常体系扩张）：`fast_agents/database/` **10 个源码文件**，包含**关系型双后端 + 向量型**两套平行后端 + 统一异常体系。
  - **关系型**：`SQLiteClient` 支持连接与上下文事务、基础 SQL 通道（execute / executemany / executescript / query）、三种声明式建表 + 幂等建索引 + `migrate=True` 补列、CRUD（insert / batch_insert / select / select_one / select_where / select_one_where / count / update / delete）、读侧谓词下推（10 种 FilterOp + AND + 单/多列排序 + limit/offset）、写侧条件写入（`upsert` + `update_where` / `delete_where` + `SqlExpr`）、标识符白名单防注入；非内存库默认开启 WAL + foreign_keys，并显式设 `PRAGMA busy_timeout`。
  - **向量型**：`VectorClientBase`（12 个抽象方法，与 `DatabaseClientBase` 平行）+ `VectorFilter` 元数据过滤模型（EQ/NE/GT/…/IN/AND/OR）+ `ChromaClient` 首个向量后端（本地 `PersistentClient` / 远端 `HttpClient` 双模式，`chromadb` 走可选依赖 extras；含"少一次往返 / delete 精确计数 / dimension 真正生效 / add 自动分块 / 关匿名遥测 / 健壮性补强"等优化）。
  - `test/test_database.py` **121/121** 全绿（98 关系型 + 23 向量，其中 17 项需 chromadb）；配套 **6 个**示例（v1.1.0 起新增 `mysql_usage.py`）。

### 12.2 待办（按优先级）

1. **J2 · 连接池 / 线程安全 + busy_timeout 收口**（P2，依赖 G2 异步并发）—— 现状：`SQLiteClient` 是单连接，线程不安全；跨线程用会偶发 `database is locked`。**部分已落地**：`_connect()` 已显式 `PRAGMA busy_timeout = timeout * 1000`（与构造参数同口径，默认 5s；上层传 `timeout=10.0` 即可对齐 10s 口径）。**剩余目标**：用 `threading.local` 给每个线程一个连接（或简单连接池）；单独暴露 `busy_timeout` 参数以便与连接 timeout 解耦。**为什么连接池现在不做**：fast_agents 当前是单线程同步模型（ReActAgent 顺序跑），连接池是 G2 异步并发之后的需求，提前做是过度设计。
2. **J3 · memory/trace/profile 三模块迁移到 SQLiteClient（含 B9 落地）**（**P1**）—— **已落地**：memory 三个 SQLite 后端（`sqlite_store.py` / `session_store.py` / `profile_store.py`）与 `trace/sqlite_store.py` 均已迁移到 `SQLiteClient`（`ensure_schema(..., migrate=True)` 建表 / 迁移 + 统一连接与 PRAGMA），不再各自手写 `CREATE TABLE` + `user_version` 迁移；B9（SQLite WAL + 并发安全）随迁移自然落地。**剩余**：`trace/jsonl_store.py` 等非 SQLite 落盘路径保持原样；后续新增的上层 store 一律走 `DatabaseClientBase` 抽象。
3. **J4 · MySQL / Postgres 后端**（P2，依赖 G4 打包发布）—— **MySQL 已落地**：`fast_agents/database/mysql_client.py`（`MySQLClient`）接口与 `SQLiteClient` 100% 对齐，`pymysql` 走 `pip install fast-agents[mysql]` 可选依赖。**剩余**：`postgres_client.py` 未实现——有了 MySQL 这版样板，照着写 Postgres 即可，上层 0 改动切换。**为什么 Postgres 现在不做**：现阶段目标场景仍是本地 / 小团队 Agent，MySQL 已覆盖「需要网络数据库」的需求；等 G4 PyPI 发布、出现明确的 Postgres 部署诉求时再做。
4. **J5 · 统一异常体系**（**已交付，含扩张**）—— **已落地**：`errors.py` 定义 `DatabaseError` 基类（继承 `ValueError`）+ 7 个语义子类（`DatabaseConnectionError` / `DatabaseOperationalError` / `DatabaseValidationError` 三大层 + `DatabaseIdentifierError` / `DatabaseIntegrityError` / `DatabaseQueryError` / `DatabaseDriverMissingError` 四个场景），`SQLiteClient` / `MySQLClient` / `ChromaClient` 三后端及其依赖的 `schema.check_ident` 已全部按语义接入、不再裸抛驱动原生异常，零破坏。**剩余目标**：待出现更细的分流诉求时再考虑 `NotFoundError` 等；按「没有第二个真实使用方不提前造抽象」的原则暂不加。
5. **J6 · upsert**（P3）—— **单条已落地**：`upsert(table, data, conflict_keys, update_columns=None)` 已提供 `ON CONFLICT DO UPDATE`；`batch_insert` 提供参数化批量插入。**剩余目标**：批量 upsert（一次多条冲突更新），等出现真实批量冲突写场景再做。
6. **J7 · 语义记忆（`memory/semantic_store.py`）接入 `VectorClientBase`**（P2，依赖 J3 节奏；本轮新增编号，`演进路线图.md` 的 J 模块同步随后）—— 现状：database 模块已提供 `VectorClientBase` + `ChromaClient`，但 `memory/` 层尚未接入；本轮刻意**不动** `semantic_store.py`，先把 database 层的向量抽象站稳。目标：让语义记忆通过 `VectorClientBase`（默认 `ChromaClient`）落盘，不在 memory 层直接 `import chromadb`，与关系型一样"只依赖抽象"。**为什么现在不做**：先把 database 层的向量抽象与首个后端交付、验收，再动上层，避免一次改动跨两层、破坏面过大。
7. **J8 · 远端 Chroma Server / 其他向量后端**（P3）—— 现状：`ChromaClient` 已支持远端 `HttpClient` 模式（`host/port/ssl/headers/tenant/database`），但尚未在真实服务端验证；FAISS / Milvus / pgvector 后端未实现。目标：补齐更多向量后端，均实现同一套 `VectorClientBase`。**为什么现在不做**：等出现真实的多机 / 大库检索需求再做，避免为不存在的问题提前造轮子。

当下最明显的空缺：**跨模块迁移已基本收口，剩下 Postgres 后端（J4 剩余）与远端 Chroma 服务端验证（J8）**——database 模块的建表 / CRUD / 读侧谓词下推 / 写侧条件写入（upsert + update_where/delete_where）已足以覆盖 memory / trace 后端的全部现有用法；下一步按 J4 / J7 / J8 推进。

---

## 十三、参考

- [Python `sqlite3` 标准库文档](https://docs.python.org/3/library/sqlite3.html) —— SQLite 后端的底层驱动参考；Row 工厂、事务语义、PRAGMA 说明全部来源于此。
- [PyMySQL 文档](https://pymysql.readthedocs.io/) / [Chroma 文档](https://docs.trychroma.com/) —— MySQL 后端（`%s` 占位符、`ON DUPLICATE KEY UPDATE`、`information_schema`）与向量后端（collection / metadata where / 近邻检索）的驱动参考。
- [PEP 249 – Python Database API Specification v2.0](https://peps.python.org/pep-0249/) —— `connect/close/cursor/execute` 命名和 Connection/Cursor 抽象的来源。
- [SQLite 官方文档（CREATE TABLE / ALTER TABLE / CREATE INDEX / PRAGMA）](https://www.sqlite.org/lang.html) —— SQLite 方言能力边界（ADD COLUMN 限制、INTEGER PRIMARY KEY ROWID 别名、WAL 模式）的权威依据。
- [SQLAlchemy Core – Table / Column / MetaData](https://docs.sqlalchemy.org/en/latest/core/metadata.html) —— 声明式 Schema / Column / Index 模型的设计借鉴，但刻意不引入 SQLAlchemy 依赖。
- [Peewee Schema Migrations](http://docs.peewee-orm.com/en/latest/peewee/playhouse.html#migrate) —— SQLite 迁移能力边界的参考（即便是轻量 ORM，对 SQLite 也只能安全做 ADD COLUMN）。
- [fast_agents 记忆模块编写说明](./记忆模块编写说明.md) —— 上层依赖 database 的典型消费者，短期/长期/画像三层存储都基于 SQLite；database 模块的抽象契约是为它们服务的。
- [fast_agents 工具模块编写说明](./工具模块编写说明.md) / [MCP 模块编写说明](./MCP模块编写说明.md) / [Skill 模块编写说明](./Skill模块编写说明.md) —— 同层级模块文档，本文件的骨架、措辞粒度、代码片段长度都是照着这三份对齐的。
- [fast_agents 演进路线图](./演进路线图.md)（J 模块 持久化层）—— 第十二章的待办项与路线图里的 G / J 系列编号保持一致，避免两处打架。

---

## 附录：发布前自检清单

写完/改完本模块文档后，逐项对照勾选：

- [x] 文档名为 `database模块编写说明.md`，落位 `docs/`（与既有文档同级）；
- [x] 开篇写清了定位、配套代码路径、版本（v1.1.0）、平台（Win/macOS/Linux）、依赖（关系型零新增、仅 sqlite3+pydantic；向量 chromadb 为可选依赖），且有可跳转目录；
- [x] 有更新历史（以初版 v1.0.0 为起点）；后续版本破坏性变更都必须给迁移办法；
- [x] 第二章讲清了分层（门面/抽象/声明模型/查询模型/实现/业务）和 5 条设计原则，后续源码导读的每一处设计都能回指某条原则；
- [x] 第三章包含对比总表（裸 sqlite3/SQLAlchemy/SQLModel/LangChain/CrewAI/fast_agents 六家）+ 异同表（**明确列出了"缺少什么"：多后端、连接池、迁移不完整、OR/JOIN/聚合等**）；
- [x] 第四章有目录结构（单数目录 `database/` + 10 个源码文件逐行职责注释）、分层 ASCII 图、三条主链路数据流图、与 ReActAgent 的接入边界；
- [x] 第五章弹性节认领为「三种建表写法与幂等迁移」，用对比表呈现 dict/list[Column]/Schema 三方案并明确给出选择建议；
- [x] 核心数据模型字段命名与仓库既有约定一致（`nullable` 不是 `allow_null`、`default` 不是 `default_value`、枚举值大写对齐 PEP 435）；
- [x] 源码导读 10 个文件各有「负责什么 / 关键实现 / 为什么这么写 / 改的时候最该看哪里」，关键片段控制在 3–15 行；
- [x] 快速上手命令按 Windows PowerShell 写，提醒了 `test/` 与标准库同名问题；最小可运行示例 ≤25 行、零依赖、不需要 API Key；
- [x] API 速查按类分小节（SQLiteClient 连/DDL/CRUD/事务 + 数据模型速查），无已废弃旧 API 需迁移；
- [x] 第十章三块齐全：设计决策 11 条（每条都是"为什么不是另一种做法"）、常见坑 17 条（每条都是"现象 + 可执行解法"）、跨平台注意事项 6 条（无平台差异的内容不留"无"占位）；
- [x] 跨平台注意事项已并入 10.3（按 v2.0 模板要求不再独立成章）；
- [x] 测试章节列出 121 项覆盖点（98 关系型 + 23 向量）+ 关键回归断言（每条锁死一个设计承诺）+ 全绿基线；未单测 / 刻意不做的能力边界如实写明；
- [x] 演进路线 J2–J8 每条都说明"现状是什么、为什么现在不做"，与《演进路线图.md》的 G / J / B 系列编号一致；
- [x] 章节数量、顺序、编号（开篇 → 一～十三 → 附录）与工具/记忆/Skill/MCP/context 五份成品文档完全一致；
- [x] 全文无"待补充 / TBD"残留，无大段与 README 重复的安装/特性介绍（README 讲怎么用，本文讲怎么实现/怎么扩展，分工明确）；
- [x] 向量后端如实写清了定位：`VectorClientBase` 与 `DatabaseClientBase` 平行、`ChromaClient` 为首个向量后端、`chromadb` 走可选依赖（未装时抛清晰 DatabaseDriverMissingError），并列出相对最小可用版的优化点；
- [x] 向量型与关系型的边界写清楚（两套平行抽象、允许同时继承、向量过滤不支持 LIKE/IS NULL、delete 空条件禁止）；
- [x] 异常体系写清楚：`errors.py` 的基类 + 7 个语义子类、均继承 `ValueError` 保兼容、三后端底层驱动异常与入参校验不再裸抛、`TypeError` 不纳入（J5）；
- [x] 换行符：`.md` 文件在仓库 `.gitattributes` 中声明为 `text eol=crlf`（LF 入库、Windows 检出 CRLF），符合交付要求。
