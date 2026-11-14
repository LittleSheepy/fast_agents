# AGENTS.md —— FastAgents AI 编程约定

> 本文件是本仓库**唯一的 AI 规则事实源**。任何 Coding Agent 开工前先读它。
> 根目录的 `CLAUDE.md` / `GEMINI.md` / `.github/copilot-instructions.md` / `CONVENTIONS.md`
> 只是指向本文件的「薄壳」——**改规则只改本文件**，别处不重复。

---

## 1. 项目速览

- **手写轻量 Agent 框架**，不依赖 LangChain / LlamaIndex 等重型编排库，直接基于 OpenAI 原生 API。
- **当前八大能力**：
  - LLM 客户端、消息系统、工具系统、ReAct Agent、记忆系统、技能系统、MCP 外部接入、上下文工程（context）；
  - 另有 **database（持久化基础服务）** 与 **trace（全链路追踪）** 两个基础设施模块。
- **依赖**：
  - 核心（已有）：`openai>=3.37`、`pydantic>=2.14`、`python-dotenv>=1.2`、`docstring-parser>=0.18`、`pyyaml>=6.0`；
  - 可选（内置搜索用）：`duckduckgo-search>=8.1`（默认、免费无需 key）、`tavily-python>=0.8`（需 `TAVILY_API_KEY`）；
  - "零重型依赖"指**不引编排框架与 SDK**（如 MCP 模块零依赖手写 JSON-RPC、trace 模块仅用标准库+pydantic），非"零第三方包"。
- **Python >= 3.10**；包管理用 uv（仓库自带 `第0步uv_安装.bat` ~ `第3步uv_add.bat`）。
- 仓库：<https://gitee.com/LittleSheepy/fast_agents>（镜像 <https://github.com/LittleSheepy/fast_agents>）。

---

## 2. 开工前必做

1. **先同步最新代码**（clone / pull），重新克隆Gitee官方仓库,基于真实代码动手，不要凭记忆或凭空发明。
2. 通读以下内容，对齐既有风格与约定：
   - `AGENTS.md`：AI规范；
   - `README.md`：特性、项目结构、核心模块、设计取舍、路线图；
   - `docs/xx模块编写说明模版.md`：**所有模块文档的骨架**，新文档必须沿用；
   - 目标模块源码 + 同类已完成模块（`fast_agents/memory`、`tools`、`mcp`、`context`、`trace`、`database`）；
   - `.gitattributes`、`pyproject.toml`、`requirements.txt`、`.python-version`。
3. **优先级：仓库既有约定 > 本文件 > 你的个人偏好**。与现有代码冲突时，以仓库现状为准。

---

## 3. 常用命令（Windows PowerShell）

```powershell
# 首次：按顺序执行仓库自带脚本，装 uv / 建 venv / 装依赖
第0步uv_安装.bat
第1步uv_init.bat
第2步uv_venv.bat
第3步uv_add.bat

# 跑测试（按模块分目录）：
python tests\init\test_init.py
python tests\context\test_context.py
python tests\database\test_database_sqlite.py
python tests\database\test_database.py
python tests\database\test_database_mysql.py
python tests\memory\test_memory.py
python tests\memory\test_memory_chroma_store.py
python tests\tools\test_tools.py
python tests\tools\test_file_tool.py
python tests\trace\test_trace.py
python tests\mcp\test_mcp.py
python tests\agents\test_react_agent.py

# 跑示例（离线可跑，不需要 API Key）
python examples\database\basic_usage.py
python examples\trace\basic_usage.py
```

---

## 4. 硬性约束（红线，违反即不合格）

下列约束按编号列出，自检时按编号勾选。

### 🔴 R1 —— 禁止破坏性重构
- 只允许「微调 + 新增」；
- **不改**现有公开行为、不改既有函数签名（纯**新增可选参数**且默认值保持原行为除外）；
- **不改**已有测试断言；
- 不要"顺手重构"与本次任务无关的代码。

### 🔴 R2 —— 不随意新增重型依赖
- 零重型编排库优先；能用标准库 / pydantic 就不要加新包；
- 可选依赖（如 chromadb、psycopg2、pymysql）必须满足：
  1. **延迟 import**（函数体内 `import`，模块顶层不引）；
  2. **缺失时降级**（`chroma_client=None` 退化为纯词法，见 `memory/chroma_store.py`）；
  3. **异常兜底不阻塞主流程**（try/except 吞错 + logger.warning）。
- 确需新增核心依赖，必须同步 `pyproject.toml`、`requirements.txt`、`uv.lock` 并说明理由。

### 🔴 R3 —— 抽象克制、一点一点来
- **没有第二个真实使用方之前，不提前造抽象层**；
- 先写最小可跑版本，抽象在真实复用出现时再抽。

### 🔴 R4 —— 换行符硬约束（按 `.gitattributes`）
- 入库统一 LF，**Windows 检出为 CRLF**；
- 交付到仓库根目录的文件：
  - **CRLF**（`\r\n`）：`.py` / `.pyi` / `.md` / `.txt` / `.bat` / `.cmd` / `.ps1` / `.psm1`；
  - **LF**（`\n`）：`.sh` / `.json` / `.toml` / `.yml` / `.yaml` / `.cfg` / `.ini` / `.gitignore` / `.gitattributes`；
  - **二进制**（不处理换行）：`.png / .jpg / .pdf / .zip / .db / .sqlite / .pyc` 等。
- 交付前逐字节校验：裸 LF=0（CRLF 文件）、裸 CR=0。

### 🔴 R5 —— SQL 必须参数化
- 任何 SQL 一律用参数绑定（`?` 位置参数或 `:name` 命名参数）防注入；
- 表名 / 列名无法参数化，必须来自内部常量，绝不用用户输入拼接。

### 🔴 R6 —— 打点 / 观测 / 日志不得阻塞主流程
- 参考 `trace/trace_logger.py` 的写法：所有写日志 / 打点 / I/O 必须 `try/except Exception: pass` 或降级，**绝不让观测代码把业务流程打挂**。

### 🔴 R7 —— 未跑通不得交付
- 交付前必须真实运行：对应测试 N/N 全绿 + 示例正常运行；
- 测试与示例必须**离线可跑**，不依赖 API Key / 外部服务（LLM 相关用 fake/mock）。

### 🔴 R8 —— 只给改动文件，路径严格对齐
- 源码按项目路径存放（根目录为 `fast_agents/`、测试在 `tests/`、示例在 `examples/`、文档在 `docs/`）；
- **只给新增/修改的文件**，未改动的文件不要塞进交付；
- **严禁把所有文件堆在同一文件夹**。

### 🔴 R9 —— 时间使用CST时间
- 统一时间用 CST

### 🔴 R10 —— 使用最新代码
- 每次回答一定要重新clone一次仓库。
- 
### 🔴 R11 —— 只考虑windows平台，不考虑跨平台。

---

## 5. 代码风格与结构（约定）

### 5.1 通用代码风格
- 文件首行用**三引号 docstring**说明模块作用与设计要点，**列参考的主流框架**（参考 `tools/tool_base.py`、`memory/memory_store_base.py` 的写法）。
- 普遍使用 `from __future__ import annotations`（本仓库所有新模块默认加）。
- 用现代类型注解；**仅用于类型注解、运行时不用的 import** 放进 `if TYPE_CHECKING:` 块，避免循环导入。
- 数据模型（pydantic v2 `BaseModel`）统一放在模块下的 `models.py`（参考 `trace/models.py`、`memory/models.py`、`mcp/models.py`；注：`database/schema.py` 是表结构 DDL 声明，不属此类）。
- 异常统一放模块自己的 `errors.py` 或 `tool_exception.py`，按"基类 + 具体子类"组织（参考 `mcp/errors.py`、`tools/tool_exception.py`）；**不要把底层库原生异常直接抛给上层**。
- 日志用标准 `logging`：`logger = logging.getLogger(__name__)`，不引 loguru/structlog。
- id 生成：`uuid.uuid4().hex`（短 hex，参考 `trace/models.py:_new_id`、`memory/models.py`）。

### 5.2 目录与命名
- **目录**：单数、小写、单个单词（`memory` / `tools` / `mcp` / `trace` / `database` / `context` / `messages` / `agents` / `skills` / `llm` / `utils`）。
- **文件**：snake_case；**类**：PascalCase；**常量**：UPPER_SNAKE_CASE；**枚举**：`class Xxx(str, Enum)`（参考 `SpanKind`）。
- ⚠️ 注意区分：`fast_agents/skills/`（技能系统**代码模块**）与 `.skills/`（项目级**技能包目录**，如 `web-research`），两者不是一回事。
- ⚠️ 注意区分：`fast_agents/tools/builtin/`（内置工具集）与 `fast_agents/memory/...` 等——内置工具一律放 `tools/builtin/`，工厂函数统一叫 `build_xxx_tools()`（参考 `build_file_tools` / `build_memory_tools` / `build_mcp_tools`）。

### 5.3 模块 `__init__.py` 出口规范
- 模块 `__init__.py` 顶部写模块 docstring，列核心组件；
- 统一 `from .xxx import ...` re-export；
- `__all__` 分组列全，便于 `from fast_agents.xxx import ...`；
- 顶层 `fast_agents/__init__.py` 同步补充新模块的对外导出（分类注释保持现有格式：版本 / Agent 范式 / 工具 / 记忆 / 上下文 / MCP / Database / Trace）。

### 5.4 数据库持久化约定
- 优先复用 `fast_agents/database/SQLiteClient`；不要每个模块自己 `import sqlite3` 重写连接/游标/建表样板；
- 幂等迁移：建表后用 `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` 补列（参考 `memory/sqlite_store.py:_migrate_schema`），升级不丢数据；
- Upsert 优先用 `INSERT OR REPLACE`（当前仓库记忆模块的写法）；需保留 `created_at` 原值、或表有外键/触发器时用 `ON CONFLICT DO UPDATE`。

### 5.5 OpenAI / 网络调用
- 所有 OpenAI 调用走 `fast_agents/llm/LLM`，不要另起 `openai.OpenAI()` 实例；
- 外部 HTTP 请求一律走已有 client / 可选 requests，**不要**在模块顶层硬引第三方 SDK。

### 5.6 钩子与打点
- 工具侧打点优先用 `ToolRegistry.add_hook()` 挂载（参考 `trace/trace_logger.py:tool_hook`），**不要改 ToolRegistry 本身的逻辑**；
- LLM / Agent 侧扩展只加**可选参数**（如 `tracer=None`），默认 `None` 时行为与改动前逐字节一致。

---

## 6. 测试代码风格与结构（约定）

### 6.1 测试入口有：main(测试所有，调用run_all)、main_of_new_add(AI新加测试要放这里一份)
- 每个测试文件有两个入口：`main` 和 `main_of_new_add`
- `main`:测试所有，调用run_all
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))
```
- `main_of_new_add`:AI新加测试也要放这里一份，以便我能只运行到测试的。
```python
def main_of_new_add():
    """ 新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    test_update_delete_data_and_where_keys_are_ident_checked()

    # AI新添加的测试(AI可以删除覆盖)
    test_enter_transaction_failure_raises_operational_error_not_attributeerror() # AI新加的测试
```

### 6.2 不要使用pyterst

### 6.3 尽量不要跨函数，尽量在一个函数中完成测试

### 6.4 按照模块存放tests下不同文件夹

---

## 7. 交付物清单

新模块默认成套交付，缺一不可：

1. **源码**：`fast_agents/<模块>/` 下完整模块，含 `__init__.py`；
2. **测试**：`tests/<模块>/test_<模块>.py`，目录结构镜像源码包；用 `sys.path.insert(0, ...)` 固定写法（参考现有所有 test 文件），**不依赖 pytest**、**全内存或临时目录**、无外部依赖、直接 `python tests/<模块>/test_xxx.py` 可跑；每个测试文件入口统一为 `main(globals())` + `main_of_new_add(globals(), "test_bN_")`（公共运行器在 `tests/_runner.py`，提供 `raises/skip/tmp_path/MonkeyPatch` 等零依赖工具）；给出 N/N 基线；
3. **示例**：`examples/<模块>/` 下至少一个可离线运行的示例；
4. **文档**：`docs/<模块>模块编写说明.md`，**严格按 `docs/xx模块编写说明模版.md` v2.0 骨架**（开篇 + 一～十三 + 附录），起版本 v1.0.0；
5. **导出同步**：如需对外暴露，更新 `fast_agents/__init__.py`；同步 README.md「特性」「项目结构」「文档导航」三栏；同步 `docs/演进路线图.md` 对应条目状态。
6. **依赖同步**：如有新增依赖，同步 `pyproject.toml` / `requirements.txt` / `uv.lock`（本任务预期绝大多数情况下为零新增依赖）。
7. **交付包**：按照仓库目录存放交付文件并打包,未修改的不用打包。打包名字可以参考fast_agents_模块名_xxxx.zip 或者 fast_agents_模块名_xxxx.zip格式
---

## 8. 交付顺序

1. **设计说明**：说清做什么 / 不做什么 / 关键取舍 / 改动范围；
2. **完整源码**：按项目路径给出新增/修改的每个文件内容；
3. **文档**：包括模块文档与（如有）README/路线图的同步更新；
4. **验证结果**：测试 N/N、示例运行输出摘要、CRLF 校验。

---

## 9. 设计约定

- 新模块、功能、文件动笔前，**参考主流框架**（LangChain / LlamaIndex / OpenAI Agents SDK / OpenTelemetry / MCP / Anthropic Skills 等）的对应实现并做对比；
  - **只借鉴设计思路与概念**（如 Trace/Span、ToolBase、Callback Hook），**不照搬 API、不引入其依赖**；
  - 对比写进模块说明文档：整体功能对比写到序号三，单个文件参考写到序号七。
- 新增能力**默认非侵入**（需要用户显式开启/注入才生效），避免影响既有链路。
- 有多种方案时**先摆差异、再找确认**，不要默默拍板。
- 一句话原则：**能跑、可读、零依赖默认路径、有测试、有文档。**
- 代码还没上线：**也就是数据库都是空的，不用为了兼容旧数据而添加兼容代码。**

---

## 10. 参考

- AGENTS.md 开放标准：<https://agents.md/>
- 模块文档骨架：`docs/xx模块编写说明模版.md`
- 后续规划：`docs/演进路线图.md`
- 已完成模块作为风格范本：`fast_agents/tools/`、`fast_agents/memory/`、`fast_agents/trace/`、`fast_agents/database/`、`fast_agents/mcp/`
