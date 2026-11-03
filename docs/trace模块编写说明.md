# fast_agents trace 模块设计与使用说明

> **模块 ID**：trace
> **模块名称**：TraceLogger 全链路记录
> **版本**：v1.0.1
> **对应《演进路线图》**：§十「模块 G：infra」—— **G1 TraceLogger 全链路记录**（v1.0.0 交付 G1 最小可跑闭环；v1.0.1 完成 G1 patch：LLM 打点补 token/model 元数据、修正 tool span 耗时；G2 异步并发 / G3 评估工具 / G4 Web UI 均未做）
> **编写日期**：2026-11-03
> **依赖前置**：database 模块（v1.0.0，`SQLiteClient`/`DatabaseClientBase` 为未来后端预留）、tools 模块（`ToolRegistry.add_hook`）、llm 模块（`LLM.invoke`/`stream_invoke`/`embed_query`/`embed_documents`）、agents/react_agent 模块
> **编写人**：infra 组

---

## 〇、文档元信息

| 字段 | 值 |
|---|---|
| 模块标识 | `fast_agents/trace/` |
| 首版发布仓库提交 | 对应本地仓库 HEAD |
| 适用 fast_agents 版本 | ≥ v0.0.1（`pyproject.toml` 当前版本） |
| 最低 Python 版本 | ≥ 3.10（对齐仓库 `.python-version`） |
| 外部依赖 | **零新增依赖**；仅用标准库 + 仓库已有的 `pydantic>=2.14.1` |
| 可选后端 | `JsonlStore`（当前唯一实现，jsonl 逐行落盘 + 内存模式）；`SQLiteStore` 位已通过 `TraceStoreBase` 抽象预留，待未来 G 系列后续迭代 |
| 文档更新责任 | 每新增一种 SpanKind / 公开 API / 存储后端，必须同步更新本文件 §六、§七、§九、§十 |

---

## 一、模块定位与设计目标

### 1.1 模块一句话定位

**trace 模块是 fast_agents 的可观测性基础设施：在不侵入业务代码、默认零开销的前提下，为 LLM 调用、工具调用、Agent 决策这三类关键路径产出带时间戳/耗时/输入输出的结构化 Span 记录，默认以 jsonl 逐行落盘，支持事后按 trace/session/kind 过滤回放，为后续 G3 评估工具与 G4 Web UI 提供数据底座。**

### 1.2 要解决的核心问题

当 ReActAgent 在真实对话里跑完一轮多步决策，我们希望事后回答这些问题：

- 这次 Agent 调用总共花了多少时间？瓶颈在 LLM 还是工具？
- 每一步决策（think→call tool→observe）的输入输出是什么？LLM 为什么会选中这个工具？
- 某个工具这次被调用了几次？参数是什么？返回了什么？耗时多少？
- 同一个 session（对话）里跨多轮 Agent.run 的所有记录能不能串起来？
- 离线复盘时能不能把整条链路按缩进层级打印出来，一眼看到调用关系？

在没有 trace 模块之前，这些信息散落在各处日志里、无法结构化查询、甚至默认就没有。G1 的目标就是把最小闭环跑通：**定义 Span 模型 → 提供 TraceLogger 记录器 → 默认 jsonl 落盘 → 接入 LLM/tool/Agent → 支持回放**。

### 1.3 设计思路借鉴与边界（必须先讲清楚，避免跑偏）

trace 模块在概念层借鉴了三家主流方案的核心模型，但**只借思想、不引依赖、不照搬 API**：

| 借鉴对象 | 借鉴了什么 | 明确**不**照搬什么 |
|---|---|---|
| **OpenTelemetry** | Trace / Span / SpanKind / Status 的层级关系；parent_id 串起调用树；Start/End 两阶段记录耗时 | 不引入 `opentelemetry-api/sdk`；不做 Context 传播、不做 exporter 抽象、不做 sampler、不做 baggage |
| **LangChain CallbackManager / Tracer** | 在 LLM/Tool/Chain 三个生命周期点上打点的思路；`on_llm_start/end/error`、`on_tool_start/end/error` 这种钩子模型 | 不引入 langchain 依赖；不做异步回调、不做自定义事件冒泡、不做 handler 链多播 |
| **OpenAI Agents SDK Tracing** | Trace 对应一次 Agent.run、Span 对应一次具体调用；trace_id 贯穿整次 run；span_id 唯一标识每次操作；`gen_trace_id/gen_span_id` 短 ID 风格 | 不引入 openai agents sdk；不做远程上报、不做 dashboard、不做 HITL 标注关联 |

**仓库既有约定优先**：

- 工具打点**必须**走 `ToolRegistry.add_hook`（见 `fast_agents/tools/tool_registry.py`），不重写注册表、不破坏其公开方法；
- LLM / ReActAgent 侧只允许**新增可选参数** `tracer=None`，默认 `None` 时行为必须与改动前**逐字节一致**（零破坏性）；
- 存储层抽象 `TraceStoreBase` 对齐 `fast_agents/database/database_client_base.py` 的抽象基类风格（ABC + 抽象方法），为将来切换 `SQLiteClient` 落盘预留好接口位；
- **默认不收集**：框架不会自动创建 tracer——不构造 `TraceLogger`、或不把它传给 LLM/Agent，就零开销、行为与改动前逐字节一致；构造出实例即启用（`enabled=True` 默认），需要显式关闭时传 `enabled=False`。打点通道的任何异常**必须吞掉**，绝不阻塞主业务（打点失败不能让一次 LLM 调用抛错）。

### 1.4 初版（v1.0.0 = G1；v1.0.1 = G1 patch）能力清单

| 能力 | 是否支持 | 说明 |
|---|---|---|
| Span 数据模型（trace_id/span_id/parent_id/session_id/name/kind/输入/输出/状态/起止时间/耗时/错误/元数据）| ✅ | 见 §6.1 `SpanRecord` |
| SpanKind 枚举（llm/tool/decision/agent/custom） | ✅ | 见 §6.2 |
| SpanStatus 枚举（success/error） | ✅ | 见 §6.3 |
| TraceLogger（手动 span、with 上下文、trace() 上下文管理器） | ✅ | 见 §7.4 |
| 默认 jsonl 落盘（路径可配置，支持目录自动创建、append 追加） | ✅ | 见 §7.3 `JsonlStore` |
| 内存模式（`:memory:`，不写磁盘，用于测试/单测） | ✅ | `MEMORY_PATH = ":memory:"` |
| 按 trace_id/session_id/kind 过滤查询 | ✅ | `TraceLogger.read_spans(...)` |
| 按 trace_id 回放（按 start_time 排序、返回可打印 dict 列表） | ✅ | `TraceLogger.replay(trace_id)` |
| attach_to_tool_registry：一行接入工具打点（走 add_hook，不改注册表） | ✅ | `TraceLogger.attach_to_tool_registry(registry)` |
| LLM 门面接入：invoke/stream_invoke/embed_query/embed_documents 可选 tracer 参数 | ✅ | 见 §四接入点 |
| ReActAgent 接入：构造时可选 tracer 参数，自动挂工具钩子 + 自动包 agent/decision span | ✅ | 见 §四接入点 |
| TraceLogger 禁用态零开销（enabled=False 时所有方法 no-op） | ✅ | 见 §3.4 |
| **LLM 打点写入 token/model 元数据（v1.0.1）**：invoke/embed 返回后把 usage（prompt/completion/total tokens）与模型名写进 `Span.metadata` | ✅ | 见 §4.3、§6.1 |
| **tool span 耗时准确（v1.0.1）**：从 `ToolResult.metadata["duration_ms"]` 取实测耗时并回推 `start_time` | ✅ | 见 §4.2、§7.4 |

### 1.5 明确**不做**的事（v1.0.1 红线）

| 不做的事 | 原因 | 计划在哪个版本做 |
|---|---|---|
| G2 异步并发（跨线程/跨协程的 Context 传递） | G1 先跑通同步单线程闭环；async 上下文要用到 contextvars，引入复杂度 | G2（路线图标注） |
| G3 评估工具（基于 trace 自动算准确率/工具命中率/token 消耗） | 评估依赖 LLM 输出结构化，需要 trace 稳定后再做；**token 用量原始数据 v1.0.1 已随 Span 落盘**，G3 可直接聚合 | G3 |
| G4 Web UI（trace 可视化看板） | 需要前端工程，G1 先输出结构化数据 + 命令行回放 | G4 |
| 上报外部平台（LangSmith / Langfuse / Arize Phoenix / OpenTelemetry Collector） | 会引入外部依赖和网络访问，违反零新增依赖与离线可跑原则 | 后续作为可选 exporter 独立模块 |
| 引入 loguru / structlog / OpenTelemetry SDK | 零新增依赖红线 | 永不强制引入，通过扩展 Store 对接 |
| hook 链多播（一个 tracer 派发给多个 handler） | G1 只有一个 store 实现，单播足够 | 有明确需求再加 |
| Span 事件（Span Event，即 span 内打点） | 初版只有 span 起/止两态；事件模型留到 G2 再扩 | G2 |
| 采样（sampler） | G1 要么全记、要么不记；采样策略后面再做 | G2/G3 |
| 流式接口的 token 统计 | `stream_invoke` 底层流式默认不返回 usage（需开 `stream_options.include_usage`，部分兼容端点不支持），为保零破坏 G1 不动流式行为 | G2 |
| 自动 PII 脱敏 | 需要规则集，留到后续 | G3 评估工具时一起做 |

---

## 二、依赖与前置条件

### 2.1 Python 与环境

- Python ≥ 3.10（对齐仓库根 `.python-version`，使用 `match/case` 风格类型注解和 `X | Y` 联合类型语法）；
- 操作系统跨平台（Windows / macOS / Linux），文件路径用 `pathlib.Path` 处理，不写死分隔符；
- 换行符约定：所有 `.py/.md/.txt` 文件必须 **CRLF**（对齐 `.gitattributes` 中 `*.py -text eol=crlf`、`*.md text eol=crlf` 的规定）。

### 2.2 外部依赖（零新增）

trace 模块**零新增第三方依赖**，只使用：

- **标准库**：`abc`、`json`、`pathlib`、`time`、`uuid`、`threading`（仅用 Lock 保证写文件原子）、`typing`、`contextlib`（`contextmanager`）；
- **pydantic**：仓库已有依赖（`pyproject.toml` 中 `pydantic>=2.14.1`），`SpanRecord` 用 `BaseModel` 定义，与 `database/schema.py`、`llm/schemas.py` 风格对齐。

`pyproject.toml` 与 `requirements.txt` **不需要改动**。

### 2.3 仓库内部依赖（已就绪）

| 依赖模块 | 用到的符号 | 用途 |
|---|---|---|
| `fast_agents.database` | `DatabaseClientBase`（风格参考，暂不直接 import） | 将来实现 `SQLiteStore(...)` 时再接入，G1 只做 jsonl |
| `fast_agents.tools.tool_registry` | `ToolRegistry.add_hook` / `ToolResult` / `ToolStatus` | 工具打点通过 hook 挂载，不侵入 `invoke_tool` 逻辑 |
| `fast_agents.llm.llm` | `LLM.invoke` / `stream_invoke` / `embed_query` / `embed_documents` | LLM 门面加可选 `tracer` 参数，内部打点 |
| `fast_agents.agents.react_agent.react_agent` | `ReActAgent.__init__` / `ReActAgent.run` | Agent 侧加可选 `tracer` 参数，自动包 agent/decision span |
| `fast_agents.llm.schemas` | `Message`、`MsgToolCall`（参考字段，不直接构造） | LLM打点时只记录 invoke 的入参/出参，不重建消息 |

---

## 三、模块架构设计

### 3.1 整体分层（三层结构，和 database 模块对齐）

```
┌──────────────────────────────────────────────────────────────┐
│  接入层（Instrumentation）                                     │
│  LLM.tracer / ReActAgent.tracer / ToolRegistry.add_hook       │
│  只做"在正确生命周期调用 tracer.start_span/end_span"，不含业务  │
├──────────────────────────────────────────────────────────────┤
│  核心层（Core API）                                            │
│  TraceLogger —— span 生命周期管理 / 上下文管理器 / 查询 / 回放   │
│  SpanRecord / SpanKind / SpanStatus —— pydantic 数据模型       │
├──────────────────────────────────────────────────────────────┤
│  存储层（Storage，抽象+实现）                                   │
│  TraceStoreBase  (ABC)  ← 抽象接口，对齐 DatabaseClientBase    │
│   └── JsonlStore       ← v1.0.0 唯一实现：jsonl 逐行落盘+内存   │
│   └── （预留）SQLiteStore ← G1 之后接入 database.SQLiteClient   │
└──────────────────────────────────────────────────────────────┘
```

分层原则（和 database 模块一致）：

- **核心层不感知存储后端**：`TraceLogger` 只面向 `TraceStoreBase` 编程，不关心底下是 jsonl 还是 SQLite；
- **接入层不感知存储**：LLM/Agent/tool 侧只拿到 `TraceLogger`（或 `None`），调用其公开方法；
- **存储层不感知上层语义**：store 只负责"写一条 SpanRecord"、"读出 SpanRecord 列表"，不判断 SpanKind。

### 3.2 Trace / Span 概念模型（概念图）

```
[ Trace: 一次 Agent.run 或一次独立链路 ]
│
│   trace_id = trace_xxxx...（`"trace_" + _new_id()`，32 位 hex）
│   session_id = sess_xxxx （跨 run 可共享）
│
├── Span(name="agent.run", kind=agent)               ← 根 span，parent=None
│   ├── Span(name="decision.step1", kind=decision)   ← 第1步决策
│   │     └── Span(name="tool.compute", kind=tool)   ← 该决策调了工具
│   ├── Span(name="decision.step2", kind=decision)   ← 第2步决策
│   │     └── Span(name="llm.chat", kind=llm)        ← 该决策触发了LLM调用
│   │
│   ...
```

- **Trace**：一次完整的逻辑链路（通常对应一次 `Agent.run` 或一次顶层手动 `with tracer.trace()`）。Trace 本身没有独立的表/对象，它由"同一个 `trace_id` 的所有 Span"集合构成，回放时按 `parent_id` 组装成树。
- **Span**：一次具体的可观测操作（一次 LLM 调用 / 一次工具调用 / 一次决策思考）。每个 Span 有唯一 `span_id`、可选 `parent_id`、必须属于某个 `trace_id`。
- **SpanKind**：Span 的类型标签（llm/tool/decision/agent/custom），用于过滤与统计。
- **SpanStatus**：Span 执行结果（success/error）；`status=error` 时 `error` 记录异常文本。

### 3.3 ID 与时戳约定

- **ID 生成**：模块内纯函数 `_new_id()` 返回 `uuid.uuid4().hex`（32 位十六进制，本身不带前缀）；`trace_id` 在其上加前缀 `"trace_"`（即 `f"trace_{_new_id()}"`），`span_id` 直接取 `_new_id()`；
- **session_id**：跨 trace 的会话标识，由调用方在构造 TraceLogger 时传入或通过 `set_session_id()` 切换；不指定时自动生成 `f"sess_{_new_id()[:12]}"`（`sess_` + UUID 前 12 位 hex）；
- **start_time / end_time**：Unix 时间戳，**秒级浮点**（`time.time()`，由 `_now_ts()` 生成）；
- **duration_ms**：`(end_time - start_time) * 1000.0`，毫秒浮点，`SpanRecord.finish()` 时自动计算；
- 所有时间字段均为**挂钟时间**（wall clock），不做单调时钟换算——G1 同步单线程足够；异步/并发场景切到 `time.perf_counter()` 是 G2 的事。

### 3.4 零开销禁用语义

TraceLogger 有实例级 `enabled: bool` 开关（构造默认 `True`）；"默认不收集"是**框架层面**的语义——不构造/不传入 tracer 就没有任何开销。

- `enabled=False` 时，`start_span/end_span/span()/trace()/attach_to_tool_registry/read_spans/replay` 全部为**无副作用的 no-op**：
  - `start_span` 返回一个占位 `SpanRecord`（`trace_id="disabled"`），对它 `finish()` / `end_span()` 都不会写入 store；
  - `attach_to_tool_registry` 直接 return，不注册任何钩子；
  - `trace()` / `span()` 上下文照常可进出，但不产生任何持久化记录；
  - `read_spans` 返回空列表；
- 注意：`enabled=False` 时**底层 store 仍会按 `path` 构造**（沿用 `JsonlStore` 默认行为，文件模式下会创建/打开文件），只是不会写入任何 Span。若希望连文件都不创建，请用 `path=":memory:"` 或直接不构造 tracer；
- 关闭态不产生 ID 生成、序列化与写入开销，保证"关了就真的零成本"。

### 3.5 打点通道的失败安全

打点属于"旁路观测"，任何异常都**绝对不能**让主业务失败。TraceLogger 内所有打点路径（start_span/end_span/钩子回调/with 退出）都包了 `try: ... except Exception: pass`：

- 序列化失败（pydantic 校验不过）→ 吞掉，记录丢失但主流程继续；
- 文件写入失败（磁盘满/权限错）→ 吞掉；
- 钩子回调里用户传了奇怪对象导致 repr 炸了 → 吞掉；
- 但**构造期错误**（如给了非法路径、非法 SpanKind）仍然抛错，避免配置错误被静默。

### 3.6 并发安全（G1 同步，G2 扩异步）

v1.0.0 只承诺**单线程同步场景**下的正确性。`JsonlStore` 内部用一把 `threading.Lock` 保证"同一进程内多线程 append 不混行"，但：

- **不**做跨进程文件锁（多进程同时写同一个 jsonl 可能混行，G1 不处理）；
- **不**做 `contextvars` 跨协程上下文传递（asyncio 下 parent_id 需要 contextvars 才能正确传播，G2 做；v1.0.x 已用 `threading.local` 保证多线程不串栈）。

如果用户在多线程场景用，同一进程内写文件是安全的；跨进程/异步场景目前建议每个 worker 用独立 trace 文件。

---

## 四、接入点与接入方式（不改现有签名，只加可选参数）

### 4.1 接入硬约束（红线，必须遵守）

| 接入点 | 允许的改动 | 不允许的改动 |
|---|---|---|
| `fast_agents/tools/tool_registry.py` | **完全不改源码**；trace 侧通过 `registry.add_hook(hook_fn)` 挂钩子 | 不修改 `invoke_tool`、不修改 `register_tool`、不新增注册表字段 |
| `fast_agents/llm/llm.py`（LLM 门面）| `invoke/stream_invoke/embed_query/embed_documents` 四个方法**仅新增**可选参数 `tracer: "TraceLogger | None" = None`，并新增可选构造参数 `tracer=None`（存到 `self.tracer` 作缺省）；默认 None 时行为与改动前**逐字节一致** | 不改返回值类型、不改既有参数顺序、不打点时不抛新异常 |
| `fast_agents/llm/openai_client.py`（底层 client）| **v1.0.1** 仅新增可选关键字参数 `usage_sink: dict | None = None`，把响应的 usage 回传给上层；默认不传时行为不变 | 不改既有返回值类型/语义；不新增依赖；不把 usage 写进返回的 `Message` |
| `fast_agents/agents/react_agent/react_agent.py` | `__init__` 仅新增可选参数 `tracer=None`，默认 None 时完全不做 trace 相关动作；`run` 内部仅在 tracer 非 None 时包 agent span + decision span | 不改 run 返回值、不改思考循环逻辑、不强制用户传 tracer |
| `fast_agents/agents/agent_base.py` | **完全不改源码** | 基类不感知 tracer，保持 tracer 是 ReActAgent 这一层的可选能力 |

**为避免循环依赖**，llm.py 和 react_agent.py 中对 TraceLogger 的类型注解一律用 **TYPE_CHECKING + 字符串前向引用**：

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fast_agents.trace import TraceLogger
```

### 4.2 工具打点：attach_to_tool_registry

这是最推荐的接入方式，一行代码即可记录**所有**工具调用：

```python
from fast_agents import ToolRegistry, TraceLogger

registry = ToolRegistry()
registry.register_tool(...)   # 正常注册工具

tracer = TraceLogger(path=":memory:", enabled=True, session_id="demo")
tracer.attach_to_tool_registry(registry)   # ← 一行接入，内部走 add_hook
```

`attach_to_tool_registry` 做了什么：

1. 调用 `registry.add_hook(self.tool_hook)`，注册一个 after-execution 钩子；
2. 钩子回调签名是 `(tool_name: str, kwargs: dict, result: ToolResult)`，由 ToolRegistry 在每次 `invoke_tool` 返回后调用（见 `tool_registry.py` 现有逻辑）；
3. 钩子内部：
   - 如果 tracer 已 disabled / closed → 直接 return；
   - 如果当前有活跃 span（即工具是在某个 agent/decision span 内部被调用）→ 自动 `parent_id` = 当前 span；否则 `parent_id=None`（顶层调用工具的情况）；
   - 开一个 kind=tool 的 span，name=`f"tool.{tool_name}"`，输入=工具参数（dict 原样，非 dict 转 `{"raw": str(...)}`），输出=`result.output` 原样，status 由 `result.status` / `result.error` 决定（error → error，否则 success），`error` 在出错时取 `result.error`；工具实测耗时从 `result.metadata["duration_ms"]` 读取并回推 `start_time`；
   - end_span 写 store。

**重要**：`attach_to_tool_registry` **不会修改 ToolRegistry 的任何状态**（不删/不换原有钩子、不包 invoke_tool），纯粹是"多挂一个钩子"，多次调用会注册多个钩子（v1.0.0 不做去重，调用方自己保证只挂一次）。

### 4.3 LLM 打点：可选 tracer 参数

LLM 门面四个对外方法新增可选参数：

```python
class LLM:
    def invoke(self, messages: list[dict], ..., tracer: "TraceLogger | None" = None) -> Message: ...
    def stream_invoke(self, messages: list[dict], ..., tracer: "TraceLogger | None" = None): ...
    def embed_query(self, text: str, ..., tracer: "TraceLogger | None" = None) -> list[float]: ...
    def embed_documents(self, texts: list[str], ..., tracer: "TraceLogger | None" = None) -> list[list[float]]: ...
```

- `effective_tracer = tracer or self.tracer`：方法级 `tracer` 优先，未传则退回构造时的 `self.tracer`；两者都为 None → 完全不打点，直接 `return self._client.xxx(...)`，零额外开销；
- 非 None 时用**上下文管理器** `with effective_tracer.span(name=..., kind=SpanKind.LLM, input={...}, metadata={...}) as s:` 包住原有逻辑；正常退出自动 end（SUCCESS），块内异常自动打 error 并**重新 raise**（不吞主业务异常）；
- 输入快照只记"轻量结构"，不缓存完整 messages：`{"model": used_model, "messages_count": N, "tools_count": M}`；
- 输出摘要同理：invoke 记 `{"role":..., "has_content":..., "tool_calls_count":..., "content_preview": content[:200]}`；
- `stream_invoke` 因为是生成器，改用**手动 `start_span` + 生成器内 `end_span`**：包一层内部生成器边 yield 边累计 chunk，正常耗尽/异常时 `end_span`（output 统计 `chunks` 数与 `content_preview`），不缓存全部 token 以避免内存问题。
- **token / model 元数据（v1.0.1）**：`invoke` / `embed_query` / `embed_documents` 通过可选 `usage_sink` 通道从底层拿回本次响应的 usage，写入 Span 的 `metadata`：`model`（模型名）、`usage`（原始明细）、`prompt_tokens` / `completion_tokens` / `total_tokens`（平铺字段，便于过滤聚合）。拿不到 usage 时不写这些键（不产生 null 噪音）。`stream_invoke` 不产生 token 元数据——底层流式默认不返回 usage（需开 `stream_options.include_usage`，部分兼容端点不支持），为保零破坏 v1.0.1 不动流式行为。

> **ReActAgent 会自动接管 LLM 打点**：`__init__` 里若发现 `self.llm.tracer` 仍是 None（用户未手动设置），就把传入的 tracer 赋给 `self.llm.tracer`；这样 run 循环里 `self.llm.invoke(messages, tools=tools, **kwargs)` 不显式传 tracer 也会自动打点（方法内 `tracer or self.tracer` 兜底）。用户已手动设置 `llm.tracer` 时不覆盖，保留其定制。

### 4.4 ReActAgent 打点：可选 tracer 参数 + 自动包 agent/decision span

ReActAgent 改动两处：

**(1) `__init__(..., tracer=None)`**

- 默认 None → 不挂工具钩子、run 时不包 span（与改动前完全一致）；
- tracer 非 None → 在 `__init__` 末尾调用 `tracer.attach_to_tool_registry(self.tool_registry)`，这样之后所有通过该 agent 的 tool_registry 调用的工具都会被自动打点；并且把 tracer 存到 `self._tracer`。

**(2) `run(...)` 内部包 span**

- 如果 `self.tracer` 非 None（内部取 `_tracer = getattr(self, "tracer", None)`）：
  - 先 `self.tracer.start_trace(name=f"agent.{self.name}.run")` 开一条新 Trace，再手动 `start_span(name="agent.run", kind=SpanKind.AGENT, input={"user_input": input_text[:500], "agent": self.name, "max_steps": ...}, metadata={"agent": self.name})` 作为根 Span；
  - 思考循环的**每一步**都 `start_span(name=f"decision.step{step}", kind=SpanKind.DECISION, input={"step": step, ...}, metadata={"step": step, "agent": self.name})` 开一个 decision 子 Span，结束时按结果 `end_span`：
    - 本步走了工具调用：output=`{"action": "tool_calls", "tool_names": [...]}`；
    - 本步产出最终回答：output=`{"action": "final", "length": len(final_answer)}`；
    - 本步抛异常：`end_span(_dec_span, output=None, error=f"{type(e).__name__}: {e}")` 后重新 raise；
  - 整次 run 结束（或异常）时 `end_span(_agent_span, output={"final_answer_preview": final_answer[:300], "steps": current_step})`，最后 `end_trace()`；
  - LLM span 无需显式传 tracer：`self.llm.invoke(...)` 靠 `self.llm.tracer`（`__init__` 已挂上）自动打点，成为当前 decision Span 的子 Span；
- 如果 `self.tracer` is None：完全走原有逻辑（一行不改）。

这样用户接入最简单：

```python
tracer = TraceLogger(path="logs/run.jsonl", enabled=True)
agent = ReActAgent(name="助手", llm=LLM(), tool_registry=registry, tracer=tracer)
result = agent.run("3 的平方是多少？")   # ← 自动产生 1 个 agent span + N 个 decision span + N 个 tool span
```

---

## 五、核心机制：Span 生命周期、上下文传播与失败安全

### 5.1 Span 三态生命周期

```
                start_span()                   end_span()
   [不存在] ──────────────────► [ACTIVE] ──────────────────► [CLOSED]
                                    │                           │
                                    │ 异常退出 with             │ 写入 store
                                    ▼                           ▼
                                 [CLOSED]                   [持久化]
```

- **不存在**：还没开 span，或 span 已经写完；
- **ACTIVE**：span 已 start、未 end，记录在 `TraceLogger` 线程本地 `span_stack` 的栈顶（见 §5.2）；
- **CLOSED**：`end_span` 被调用，duration_ms 已计算，序列化写入 store。

### 5.2 parent_id 的隐式传播（线程本地栈）

TraceLogger 内部用 `threading.local()` 维护两个栈：`trace_stack`（trace_id 栈）与 `span_stack`（span_id 栈），作为"当前正在进行的 span 链"：

- `start_span(...)` 时：
  1. 生成 span_id；
  2. parent_id = `span_stack` 栈顶的 span_id（如果栈非空），否则 None；
  3. 如果当前 span 没显式传 trace_id，使用栈顶的 trace_id（保持同一 trace），否则自己生成新 trace_id；
  4. 把新 span 压栈；
  5. 返回这个 span 对象。
- `end_span(span, ...)` 时：
  1. 计算 end_time、duration_ms；
  2. 调 `span.finish(...)` 把 output/status/error 等字段写回 span；
  3. 若该 span 正是 `span_stack` 栈顶则 pop（乱序 end 时跳过出栈，不抛错）；
  4. 调 `self._store.write(span)` 持久化；
  5. 返回 `None`（`SpanRecord` 在 start 时已拿到）。

这样用 `with tracer.span(...)` 嵌套时，子 span 自动拿父的 span_id，形成树状关系；最外层用 `with tracer.trace(...) as tid:` 时会压入新 trace_id，作为整条链路的根。

### 5.3 with 语法糖与 trace() 上下文管理器

```python
# 用法 A：手动 start/end（适合跨函数传递 span、或需要精确控制结束时机）
sp = tracer.start_span(name="mystep", kind=SpanKind.CUSTOM, input={"q": 1})
try:
    do_something()
    tracer.end_span(sp, output={"ok": True})
except Exception as e:
    tracer.end_span(sp, error=f"{type(e).__name__}: {e}")   # error 非空自动置 ERROR
    raise

# 用法 B：with tracer.span(...)：自动 start/finish（最常用，推荐）
with tracer.span(name="mystep", kind=SpanKind.CUSTOM, input={"q": 1}) as s:
    do_something()
    s.output = {"ok": True}     # 可以在 with 内给 output 赋值
# ← with 退出时自动 end_span 写入；块内异常自动打 error 并重新抛出

# 用法 C：trace()：开新 Trace（yield 的是 trace_id 字符串）
with tracer.trace(name="agent.run") as tid:
    with tracer.span(name="step1", kind=SpanKind.DECISION) as s1:
        ...
# ← 退出时 end_trace；期间所有 span 共享这条根 trace_id
```

三种用法在接入层按需选择：llm.py 用 `with tracer.span(...)` 包住一次调用；`stream_invoke` 与工具钩子用手动 start/end（需要在生成器耗尽/钩子返回时精确 end）；ReActAgent 在 run 入口 `start_trace()` + `start_span()` 手动包 agent/decision span。

### 5.4 输入输出的截断策略（防 jsonl 爆盘）

LLM 输入（完整 messages 列表）和工具输出（可能是大段文本）在某些场景下会非常大（几万~几十万 token），直接全量写入 jsonl 会让磁盘迅速爆掉。v1.0.x 的约定是**在接入层做摘要**，而不是给 TraceLogger 加截断开关：

- **LLM span**：输入不落完整 messages，只记 `{"model", "messages_count", "tools_count"}`；输出只记 `{"role", "has_content", "tool_calls_count", "content_preview"}`，其中 `content_preview` 截前 200 字符；
- **agent span**：`input` 里 `user_input` 截前 500 字符；`output` 里 `final_answer_preview` 截前 300 字符；
- **decision span**：输入只记 `step` 与首步的 `user_input_for_step`；输出只记 `action` 与 `tool_names` / `length` 这类元信息，不落完整模型输出；
- **tool span**：`input` 记工具参数（dict 原样）、`output` 记 `result.output`（若工具返回大文本，业务侧已有"超长结果卸载"等机制控制）；
- 好处是**零额外配置**：不存在需要关闭的截断参数，接入层各自控制摘要粒度；将来 G3 若要完整落盘，可在接入层或新 store 后端里单独做。

### 5.5 read_spans / replay：查询与回放

- `read_spans(...)` 支持下列过滤条件（任意组合，AND 关系）：
  - `session_id: str | None`
  - `trace_id: str | None`
  - `kind: str | None`（传 `"llm"` / `"tool"` 等字符串值）
  - `limit: int | None`（最多返回条数）
  - 返回顺序为 **store 原始顺序**（`JsonlStore` 即写入顺序）；需要按时间看请用 `replay()`（它按 `start_time` 排序）。**当前不提供时间范围过滤**（时间筛选留待 G2 切 SQLiteStore 建索引后提供）。
- `replay(trace_id=None, session_id=None) -> list[dict]`：
  1. `read_spans(trace_id=..., session_id=...)` 取出目标 span；
  2. 按 `start_time` 升序排序；
  3. 每项转成字段稳定的可打印 dict（`span_id` / `parent_id` / `kind` / `name` / `status` / `duration_ms` / `error` / `input` / `output` / `metadata`）；
  4. 返回该 dict 列表，调用方可自行渲染成缩进树 / 瀑布图（`examples/trace/*.py` 里有打印辅助）。

### 5.6 与数据库模块的衔接（预留位，G1 不实现 SQLiteStore）

`TraceStoreBase` 抽象接口定义与 `database.DatabaseClientBase` 同一风格：

```python
class TraceStoreBase(ABC):
    @abstractmethod
    def write(self, span: SpanRecord) -> None: ...
    @abstractmethod
    def read(self, session_id=None, trace_id=None, kind=None, limit=None) -> list[SpanRecord]: ...
    @abstractmethod
    def close(self) -> None: ...
    def iter_records(self) -> Iterator[SpanRecord]: ...   # 默认基于 read()
    def flush(self) -> None: ...                           # 默认 no-op
```

`JsonlStore` 是 v1.0.0 唯一实现。G1 之后迭代 `SQLiteStore` 时：

1. 新建 `fast_agents/trace/sqlite_store.py`，实现 `TraceStoreBase`；
2. 内部持有一个 `fast_agents.database.SQLiteClient`（或 `DatabaseClientBase`）实例；
3. 建表用 `db.ensure_schema(Schema(...), migrate=True)`，字段和 SpanRecord 一一对应（trace_id/span_id/parent_id/session_id/name/kind/status/start_time/end_time/duration_ms/input/output/error/metadata），对 trace_id/span_id/session_id/kind/start_time 建索引；
4. TraceLogger 构造时增加 `store: TraceStoreBase | None = None` 参数，允许调用方传入自定义 store；不传且 path 以 `".db"`/`".sqlite"` 结尾时自动构造 SQLiteStore（此为 G2 之后计划，v1.0.0 不实现）；
5. 切换后端时上层（LLM/Agent/tool 钩子）代码零改动。

---

## 六、核心数据模型

所有模型都在 `fast_agents/trace/models.py`，用 pydantic `BaseModel` 定义；枚举都继承 `str, Enum`，方便直接打印、序列化、JSON 落盘。

### 6.1 `SpanRecord`（最小记录单元）

```python
class SpanRecord(BaseModel):
    trace_id:    str                                      # 所属 Trace，形如 "trace_<32位hex>"
    span_id:     str = Field(default_factory=_new_id)     # 自身 ID（32 位 hex）
    parent_id:   Optional[str] = None                     # 父 Span ID；根 Span 为 None
    session_id:  Optional[str] = None                     # 会话 ID，跨 trace 聚合用
    name:        str                                      # Span 名称，如 "llm.invoke"、"tool.compute"、"decision.step1"
    kind:        SpanKind                                 # 类型标签
    start_time:  float = Field(default_factory=_now_ts)   # 开始时刻（秒）
    end_time:    Optional[float] = None                   # 结束时刻（秒）
    duration_ms: Optional[float] = None                   # 耗时（毫秒），finish() 时计算
    input:       Any = None                               # 入参（轻量摘要）
    output:      Any = None                               # 出参（轻量摘要）
    status:      SpanStatus = SpanStatus.SUCCESS
    error:       Optional[str] = None                     # status=error 时的异常文本
    metadata:    Dict[str, Any] = Field(default_factory=dict)  # 扩展字段
```

| 字段 | 类型 | 说明 | 必填 |
|---|---|---|---|
| `trace_id` | str | 所属 Trace ID；同一条 trace 的所有 span 共享 | ✅（start_span 自动生成或继承） |
| `span_id` | str | Span 自身全局唯一 ID | ❌ 默认 `_new_id()` 自动生成 |
| `parent_id` | str\|None | 父 Span ID；根 Span 为 None | ❌ 默认 None |
| `session_id` | str\|None | 会话 ID，跨 trace 聚合用（例如一次对话 session 包含多轮 run） | ❌ 构造时默认取 tracer 的 session_id |
| `name` | str | Span 显示名；接入层约定：`llm.invoke`/`llm.stream_invoke`/`llm.embed_query`/`llm.embed_documents`、`tool.<name>`、`decision.step<N>`、`agent.run` | ✅ |
| `kind` | SpanKind | 类型枚举（见 §6.2） | ✅ |
| `start_time` | float | 开始时刻，**秒级** Unix 时间戳（`_now_ts()`） | ❌ 默认 `_now_ts()` |
| `end_time` | float\|None | 结束时刻，`finish()` 时自动填 | ❌ finish() 时赋值 |
| `duration_ms` | float\|None | `(end_time - start_time) * 1000`，`finish()` 时自动算 | ❌ finish() 时赋值 |
| `input` | Any\|None | 入参（轻量摘要，见 §5.4） | ❌ |
| `output` | Any\|None | 出参（轻量摘要） | ❌ |
| `status` | SpanStatus | 执行结果（见 §6.3） | ❌ 默认 SUCCESS（finish/end_span 可覆盖） |
| `error` | str\|None | 异常信息；仅 status=error 时有值 | ❌ |
| `metadata` | dict | 扩展字段字典；v1.0.1 主动写入：llm span 写 model / usage / prompt_tokens / completion_tokens / total_tokens；tool span 写 tool_name / tool_status（无耗时时加 duration_source） | ❌ 默认 {} |

**最小字段集对齐**：用户任务要求的"时间戳 / 类型(llm/tool/decision) / 输入 / 输出 / 耗时 / session_id"全部覆盖，并在其基础上补充了 OpenTelemetry 风格的 `trace_id/span_id/parent_id/name/status`，这也是"借鉴但不照搬"原则的体现。

### 6.2 `SpanKind`（Span 类型枚举）

```python
class SpanKind(str, Enum):
    LLM      = "llm"       # LLM 调用（invoke / stream_invoke / embed_*）
    TOOL     = "tool"      # 工具调用（ToolRegistry.invoke_tool 触发）
    DECISION = "decision"  # Agent 一步决策（think → action/finish）
    AGENT    = "agent"     # Agent 一次 run（根 span）
    CUSTOM   = "custom"    # 用户自定义打点
```

为什么是这 5 种？

- 前 3 种（llm/tool/decision）对应用户任务点名要求的最小类型集；
- `agent` 单独列出是因为一次 Agent.run 是自然的根 span，回放时需要一个起点；
- `custom` 给调用方留扩展口（比如业务方想在自己的工具链里打自定义 span）。

### 6.3 `SpanStatus`（执行结果枚举）

```python
class SpanStatus(str, Enum):
    SUCCESS = "success"
    ERROR   = "error"
```

v1.0.0 只做二元状态。未来 G3 评估工具可能会加 `SKIPPED`/`TIMEOUT` 之类，但初版保持最小。

### 6.4 模块级辅助（纯函数，不依赖 store）

```python
def _new_id() -> str:
    """生成 32 位十六进制短 ID（uuid4().hex）。"""
    return uuid.uuid4().hex

def _now_ts() -> float:
    """当前 Unix 时间戳（秒，浮点）。"""
    return time.time()
```

这两个是模块内部纯函数（前缀 `_`），不属于公开 API。

### 6.5 `TraceStoreBase`（存储抽象基类）

```python
class TraceStoreBase(ABC):
    @abstractmethod
    def write(self, span: SpanRecord) -> None: ...
    @abstractmethod
    def read(
        self,
        session_id: str | None = None,
        trace_id: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[SpanRecord]: ...
    def iter_records(self) -> Iterator[SpanRecord]: ...   # 默认实现，基于 read()
    @abstractmethod
    def close(self) -> None: ...
    def flush(self) -> None: ...                           # 默认 no-op
```

子类（当前只有 `JsonlStore`，未来有 `SQLiteStore`）必须实现 `write` / `read` / `close` 三个抽象方法；`iter_records` / `flush` 已有默认实现。`read` 的 `kind` 参数是字符串值（如 `"tool"`），`limit` 按写入顺序截断。

---

## 七、逐个文件讲解（源码导读）

每个文件固定三段：**它负责什么 → 关键实现 → 为什么这么写**。

### 7.1 `__init__.py` —— 模块出口，统一 re-export

```python
"""
fast_agents.trace —— TraceLogger 全链路记录模块（路线图 G1）。

借鉴 OpenTelemetry Trace/Span 模型、LangChain callback、OpenAI Agents SDK tracing 的思路，
但不引入任何额外依赖；最小字段集覆盖时间戳/类型/输入/输出/耗时/session_id/trace_id/span_id。
"""
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase
from .jsonl_store import JsonlStore, MEMORY_PATH
from .trace_logger import TraceLogger

__all__ = [
    # 核心记录器
    "TraceLogger",
    # 数据模型
    "SpanKind",
    "SpanStatus",
    "SpanRecord",
    # 存储
    "TraceStoreBase",
    "JsonlStore",
    "MEMORY_PATH",
]

__version__ = "1.0.1"
```

- **负责什么**：给上层一个稳定的 import 入口（`from fast_agents.trace import TraceLogger, SpanKind, ...`，或更简单的 `from fast_agents import TraceLogger`，因为根 `__init__.py` 已 re-export），避免上层直接 import 子文件路径。
- **为什么这么写**：对齐 `fast_agents/database/__init__.py`、`fast_agents/tools/__init__.py`、`fast_agents/memory/__init__.py` 的 re-export 模式，`__all__` 显式列出公共 API。
- **要改的时候最该看的地方**：新增公开类（例如未来的 `SQLiteStore`）后在这里加一行导出 + `__all__` 一项，并同步更新根 `fast_agents/__init__.py`。

### 7.2 `models.py` —— 数据模型 + ID/时间纯函数

```python
class SpanRecord(BaseModel):
    trace_id: str
    span_id: str = Field(default_factory=_new_id)
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    name: str
    kind: SpanKind
    start_time: float = Field(default_factory=_now_ts)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    input: Any = None
    output: Any = None
    status: SpanStatus = SpanStatus.SUCCESS
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

- **负责什么**：定义 `SpanRecord`/`SpanKind`/`SpanStatus` 三个公开模型，以及模块内部的 `_new_id` / `_now_ts` 辅助函数。
- **关键实现**：
  - `SpanKind`/`SpanStatus` 都继承 `str, Enum`，这样 `json.dumps(kind)` 直接得到 `"llm"`/`"tool"` 而不是枚举对象；
  - `SpanRecord` 用 pydantic BaseModel，字段可属性赋值（`finish()` 内会写 `self.end_time/duration_ms/output/status/error/metadata`）；
  - `trace_id` 必填，`span_id` / `start_time` 带 `default_factory`，`end_time` / `duration_ms` 由 `finish()` 填充，`start_span` 时不需要；
- **为什么这么写**：
  - 模型字段和 OTel Span 大体对应但精简（不做 attributes/events/links/status_code/resource 这些重型字段），避免 API 膨胀；
  - `metadata` 字段是 dict，v1.0.1 已用于承载 LLM 的 token 用量与模型名（见 §6.1），后续加 retry_count 等**不用改 schema**，保持向后兼容；
  - ID/时间做成独立纯函数（`_new_id` / `_now_ts`），便于复用与单测。

### 7.3 `jsonl_store.py` —— jsonl 逐行落盘实现（v1.0.0 唯一后端）

```python
MEMORY_PATH = ":memory:"

class JsonlStore(TraceStoreBase):
    def __init__(self, path: str | Path | None = MEMORY_PATH, *, append: bool = True, encoding: str = "utf-8"):
        ...
    def write(self, span: SpanRecord) -> None: ...
    def read(self, session_id=None, trace_id=None, kind=None, limit=None) -> list[SpanRecord]: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

- **负责什么**：把 SpanRecord 以"每行一个 JSON"的形式追加写入文本文件；或在内存中用 list 保存（`path=":memory:"`）。
- **关键实现**：
  - **内存模式**：`path` 为 `":memory:"` / `None` 时不打开任何文件，写入只 append 到 `self._mem: list[dict]`；读取直接在内存过滤——单测和 demo 推荐，零磁盘 IO；
  - **文件模式**：
    - 如果 `append=True`（默认）且文件已存在 → 以 `"a"` 追加打开（保证跨进程 run 能续接历史 trace）；
    - 如果 `append=False` → 以 `"w"` 覆盖打开；
    - 写入路径的父目录不存在时自动 `Path.mkdir(parents=True, exist_ok=True)` 创建；
  - **序列化**：`span.model_dump(mode="json", exclude_none=False)` 后再用 `_safe_serialize()` 清洗 `input/output/metadata`（把无法 JSON 化的对象递归转 `str`），最后 `json.dumps(..., ensure_ascii=False)` 写入并补 `\n`；`mode="json"` 是关键，它能让 Enum 被序列化为其 `.value`（即 `"llm"` 而非 `"SpanKind.LLM"`）；
  - **并发**：`threading.Lock()` 保护 write 和 read（读的时候也要锁，避免和写交错导致半行）；
  - **反序列化**：文件模式在**构造时**把已有文件逐行 `json.loads` 读入 `self._mem`；`read()` 再逐条 `SpanRecord(**d)` 还原，遇到坏行（被截断/半行/字段缺失）**跳过**，不影响整体读取；
  - **过滤**：`read()` 在 `self._mem` 上按 session_id/trace_id/kind/limit 过滤，简单可靠，G1 数据量不大足够。
- **为什么这么写**：
  - 选 jsonl 作为默认后端，原因：① 人眼可读（直接 tail 文件就能排查问题）；② append-only 写入天然快，不需要建索引；③ 一行一个 Span 天然容错（一行坏了不影响其它行）；④ 零依赖、零 schema 迁移成本；
  - **不直接用 SQLite**：G1 先把抽象层定好，jsonl 实现最短（<200 行）；等 trace 字段稳定、有查询性能需求再接 SQLite 作为可选后端；
  - **`MEMORY_PATH = ":memory:"`** 是字符串常量（而不是 `None`），为了和 SQLite 惯例（`:memory:` 表示内存库）对齐，将来 SQLiteStore 可以复用同一个 sentinel；
  - **append=True 默认**：典型场景是"应用跑起来后多次 agent.run"，追加写保证跨 run 的所有 trace 都在同一个文件里，便于会话级回放。
- **踩过的坑**：
  - 最初写的时候用 `json.dumps(span.model_dump())` 没有传 `mode="json"`，结果 Enum 被序列化为 `"SpanKind.LLM"` 字符串，反序列化时 pydantic 校验失败、坏行被跳过——表现为"写入成功但读出来是空的"，排查了一轮才发现；**结论：凡是有 Enum/UUID/datetime 的 pydantic 模型要 JSON 序列化，一律用 `model_dump(mode="json")`**。

### 7.4 `trace_logger.py` —— 核心记录器

```python
class TraceLogger:
    def __init__(
        self,
        store: TraceStoreBase | None = None,
        *,
        path: str | Path | None = None,
        session_id: str | None = None,
        enabled: bool = True,
    ): ...
    @property
    def session_id(self) -> str: ...
    def set_session_id(self, session_id: str) -> None: ...
    def start_trace(self, name="root", session_id=None) -> str: ...
    def end_trace(self) -> None: ...
    def trace(self, name="root", session_id=None): ...           # contextmanager，yield trace_id
    def start_span(self, name, kind=SpanKind.CUSTOM, *, input=None, metadata=None, session_id=None) -> SpanRecord: ...
    def end_span(self, span, output=None, status=SpanStatus.SUCCESS, error=None, metadata=None) -> None: ...
    def span(self, name, kind=SpanKind.CUSTOM, *, input=None, metadata=None): ...  # contextmanager，yield SpanRecord
    def tool_hook(self, name, tool_args, result) -> None: ...
    def attach_to_tool_registry(self, registry) -> None: ...
    def read_spans(self, session_id=None, trace_id=None, kind=None, limit=None) -> list[SpanRecord]: ...
    def replay(self, trace_id=None, session_id=None) -> list[dict]: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "TraceLogger": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...
```

- **负责什么**：对外唯一的记录器门面；管理活跃 trace/span 栈、session 上下文、工具钩子注入、查询与回放。
- **关键实现**（按模块分区）：
  - **构造与开关**：构造默认 `enabled=True`；外部传了 `store` 就用它，否则 `JsonlStore(path or MEMORY_PATH, append=True)`；`session_id` 不传则自动 `f"sess_{_new_id()[:12]}"`；`enabled=False` 时打点方法全部走 no-op 分支；
  - **活跃栈**：`self._local = threading.local()`，内部维护 `trace_stack`（trace_id 栈）与 `span_stack`（span_id 栈），`parent_id` / `trace_id` 通过栈顶传播；另用模块级 `ContextVar`（`_active_trace_id` / `_active_span_id`）作兜底；
  - **start_span**：`parent_id` 取自 `span_stack` 栈顶、`trace_id` 取自 `trace_stack` 栈顶（栈空时懒启动一条默认 trace）、把 span_id 入栈、返回 `SpanRecord`；`enabled=False` 时直接返回一个 `trace_id="disabled"` 的占位 SpanRecord；**此时不写 store**（end_span 才写）；
  - **end_span**：`error` 非空且 status 仍为 SUCCESS 时自动改 ERROR → 调 `span.finish(...)` 填 end_time/duration_ms/output/status/error；若 span 正是 `span_stack` 栈顶则出栈；最后 `store.write(span)`，写入异常全部吞掉；
  - **`span()` 上下文管理器**：`@contextmanager` 实现，进入时 `start_span(...)`、yield `SpanRecord`；正常退出时若 span 尚未 finish 就 `end_span`，块内异常则 `end_span(..., error=f"{type(e).__name__}: {e}")` 后重新 raise；**`SpanRecord` 本身不是上下文管理器**，要 with 语法请用 `tracer.span(...)`；
  - **`trace()` 上下文管理器**：`@contextmanager`，进入时 `start_trace(name, session_id)` 生成并压入新 trace_id，退出时 `end_trace()`；**yield 的是 trace_id 字符串**，不是 span；
  - **`attach_to_tool_registry`**：向 `registry.add_hook(self.tool_hook)` 注册钩子；`tool_hook(name, tool_args, result)` 构造 tool Span 并 finish，读 `result.metadata["duration_ms"]` 回推 start_time，所有异常吞掉；钩子签名与 ToolRegistry 约定一致（`(tool_name, tool_args, result: ToolResult)`），不依赖任何私有 API；
  - **`read_spans`**：直接委托给 store（`self._store.read(...)`），返回 `list[SpanRecord]`；
  - **`replay`**：`read(trace_id=... 或 session_id=...)` 取数 → 按 `(start_time, span_id)` 排序 → 每项转成字段稳定的 dict（span_id/parent_id/kind/name/status/duration_ms/error/input/output/metadata）返回 `list[dict]`；不做内建渲染（示例里自带缩进打印辅助）；
  - **`close` / `flush`**：分别委托 `store.close()` / `store.flush()`，异常吞掉；`__enter__/__exit__` 让 `with TraceLogger() as t:` 退出时自动 flush + close；
- **为什么这么写**：
  - **门面模式**：所有公开能力都挂在 TraceLogger 上，上层不需要知道 JsonlStore/栈模型这些内部结构；
  - **默认不收集**：框架不会自动构造 tracer；构造出的实例默认 `enabled=True`，确实不需要时传 `enabled=False`（或干脆不构造）——我们不希望"用户什么都没配就自动在磁盘上生成一堆 jsonl 文件"；这是"最小惊讶"原则；
  - **store 可注入**：允许传自定义 store（例如将来内存测试里的 MockStore、或用户自己的上报后端），同时保持默认用 JsonlStore 简单；
  - **钩子不依赖工具内部**：attach_to_tool_registry 只使用 ToolRegistry 公开的 `add_hook(hook_fn)` 方法，不访问 `_hooks` 私有字段、不 patch invoke_tool，符合"不改注册表逻辑"红线；
  - **SpanRecord 复用为数据对象**：`start_span` 返回的就是 `SpanRecord` 本身（不是单独的 Span 包装类），`span()` 上下文管理器在 with 块内 yield 它，可直接 `s.output = {...}` 赋值，少一层包装；
- **踩过的坑**：
  - `span()` 的 with 块内**抛异常**时，会走 `except` 分支：`end_span(..., error=...)` 标记为 error 后**重新 raise**；正常退出（含 return/break）则视为 SUCCESS；
  - `attach_to_tool_registry` 在 `enabled=False` 时必须直接 return——否则即使 tracer 关闭也会挂上钩子产生开销；
  - `tool_hook` 的耗时不能靠 `getattr(result, "duration_ms", None)` 取——该字段在 `ToolResult.metadata` 里，v1.0.0 曾因此拿不到真实耗时，v1.0.1 已改为读 metadata 并回推 `start_time`。

### 7.5 `llm/llm.py` 与 `react_agent/react_agent.py` 的接入改动

这两个文件属于其他模块，本处只讲"trace 相关改动"部分：

**llm.py 改动**：

- 文件顶部 `if TYPE_CHECKING: from fast_agents.trace import TraceLogger`（延迟导入避免循环依赖）；`__init__` 新增可选参数 `tracer=None` 存到 `self.tracer`；
- 四个公开方法签名新增可选参数 `tracer: "TraceLogger | None" = None`（放在 `tools` 之后）；
- 每个方法开头统一求 `effective_tracer = tracer or self.tracer`，为 None 时**直接返回底层调用**（零开销、与原行为逐字节一致）；
- invoke / embed_query / embed_documents 用上下文管理器包：

```python
usage_sink = {}
with effective_tracer.span(name="llm.invoke", kind=SpanKind.LLM, input=input_snapshot,
                           metadata={"model": used_model, "has_tools": bool(tools)}) as s:
    response = self._client.invoke(messages=messages, tools=tools, usage_sink=usage_sink, **kwargs)
    s.output = {"role": ..., "content_preview": content[:200], ...}
    _merge_usage_metadata(s.metadata, usage_sink)   # v1.0.1：token/model 元数据
    return response
```

- stream_invoke 是生成器，改用**手动 start_span + 生成器内 end_span**（若按上面 try/except 直接包，会在生成器创建时就 end_span，是错的）：

```python
span = effective_tracer.start_span(name="llm.stream_invoke", kind=SpanKind.LLM, input=input_snapshot, metadata={...})
chunks = []
try:
    for chunk in self._client.stream_invoke(...):
        if chunk:
            chunks.append(chunk)
        yield chunk
except Exception as e:
    effective_tracer.end_span(span, output={...}, error=f"{type(e).__name__}: {e}")
    raise
else:
    effective_tracer.end_span(span, output={"chunks": len(chunks), "content_preview": "".join(chunks)[:200], ...})
```

**react_agent.py 改动**：

- 文件顶部同样 TYPE_CHECKING 导入；`__init__` 参数新增 `tracer=None`，末尾：

```python
self.tracer = tracer
if tracer is not None:
    if getattr(self.llm, "tracer", None) is None:   # 用户未定制才挂，避免覆盖
        self.llm.tracer = tracer
    tracer.attach_to_tool_registry(self.tool_registry)
```

- `run` 方法体内**原地插入** `if _tracer is not None:` 分支（`_tracer = getattr(self, "tracer", None)`）：
  - 入口：`_tracer.start_trace(name=f"agent.{self.name}.run")` + `start_span(name="agent.run", kind=SpanKind.AGENT, ...)`；
  - 每步：`start_span(name=f"decision.step{N}", kind=SpanKind.DECISION, ...)`，步末按「走了 tool_calls」/「产出 final」/「抛异常」分别 `end_span`；
  - 出口：`end_span(_agent_span, output={"final_answer_preview": ..., "steps": ...})`；`finally` 里 `end_trace()`；
  - LLM 调用不显式传 tracer（靠 `self.llm.tracer` 自动打点）。
- 非 tracer 路径（`_tracer is None`）下，所有 `if _tracer is not None:` 分支都不执行，业务逻辑与改动前**逐字节一致**；打点语句**全部原地嵌在原有流程里**（并非复制两份循环），改动集中、diff 可逐行 review。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents

# 依赖：trace 模块零新增依赖，只要装好仓库 requirements 即可
pip install -r requirements.txt

# 跑测试（41/41 全过才算装好）
python -m pytest test\test_trace.py -v

# 跑两个示例（全部用内存/临时目录，跑完 _output 里有 jsonl 可查看）
python examples\trace\basic_usage.py
python examples\trace\jsonl_replay.py
```

> **命令必须按 Windows PowerShell 写**：项目里的 `test/` 目录和标准库 `test` 包同名，所以用 `python -m pytest test\test_trace.py` 或 `python test\test_trace.py` 直接执行，不要用 `python -m test.test_trace`（那会尝试跑标准库测试）。

### 8.2 最小可运行示例（零依赖、不需要 API Key、全部内存）

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fast_agents import TraceLogger, SpanKind

tracer = TraceLogger(path=":memory:", session_id="demo-min")   # :memory: 只在内存，零磁盘 IO

with tracer.trace(name="demo") as tid:                        # 开一条 Trace（yield trace_id）
    with tracer.span(name="step1", kind=SpanKind.CUSTOM, input={"msg": "hi"}) as s:
        s.output = {"msg": "hello"}

spans = tracer.read_spans()
print(f"记录了 {len(spans)} 条 span")   # 1 条：step1（trace() 只开 Trace，本身不产生 Span）
for s in spans:
    print(f"  {s.kind.value:<8}  {s.name:<10}  {s.duration_ms:.3f}ms  {s.status.value}")

tracer.close()
```

这段代码 < 20 行，不需要任何外部服务，跑通就说明 trace 模块装好了。

### 8.3 在 ReActAgent 中的接入方式（一行 tracer= 即可）

```python
from fast_agents import LLM, ReActAgent, ToolRegistry, TraceLogger
from fast_agents.tools import StructuredTool

# 构造一个假 LLM（测试/离线场景用），或真实 LLM(需要 API Key)
llm = LLM(...)  # 省略

def compute(x: int) -> int:
    """返回 x 的平方"""
    return x * x

registry = ToolRegistry()
registry.register_tool(StructuredTool.from_function(compute))

# 关键：构造 tracer 并传给 Agent——Agent 自动处理 LLM/tool/decision 打点
tracer = TraceLogger(path="logs/agent_run.jsonl", enabled=True, session_id="chat-001")
agent = ReActAgent(name="助手", llm=llm, tool_registry=registry, tracer=tracer)

result = agent.run("3 的平方是多少？")
print("Agent 输出：", result)

# 事后回放整次 run
spans = tracer.read_spans()
trace_id = spans[0].trace_id
print("\n=== 链路回放 ===")
for row in tracer.replay(trace_id):
    print(row)

tracer.close()
```

### 8.4 典型用法

#### 用法 1：自定义 span 给业务代码打点

```python
with tracer.span(name="pipeline.load_data", kind=SpanKind.CUSTOM) as sp:
    data = load_from_disk(path)
    sp.output = {"rows": len(data), "path": str(path)}

with tracer.span(name="pipeline.predict", kind=SpanKind.CUSTOM) as sp:
    preds = model.predict(data)
    sp.output = {"pred_count": len(preds)}
```

#### 用法 2：工具钩子单独使用（不接 Agent）

```python
registry = ToolRegistry()
registry.register_tool(StructuredTool.from_function(add))

tracer = TraceLogger(path=":memory:", enabled=True)
tracer.attach_to_tool_registry(registry)

# 只要通过 registry.invoke_tool 调用，就会自动打 tool span
result = registry.invoke_tool("add", {"a": 1, "b": 2})
print(tracer.read_spans(kind="tool"))   # 1 条
```

#### 用法 3：LLM 单独打点（不传 Agent）

```python
tracer = TraceLogger(path=":memory:", enabled=True)
llm = LLM()
msg = llm.invoke([{"role": "user", "content": "你好"}], tracer=tracer)
for s in tracer.read_spans(kind=SpanKind.LLM):
    print(s.name, s.duration_ms, "ms")
```

#### 用法 4：按 session / kind 过滤

```python
# 取 session s1 的所有 span
for s in tracer.read_spans(session_id="s1"): ...

# 取全部 tool span（最多 10 条）
for s in tracer.read_spans(kind="tool", limit=10): ...
```

#### 用法 5：关闭后零开销

```python
tracer = TraceLogger(path="logs/dummy.jsonl", enabled=False)   # 显式关闭
tracer.attach_to_tool_registry(registry)   # 不挂钩子
agent = ReActAgent(..., tracer=tracer)     # 不会产生任何 span
result = agent.run("hi")                  # 和没传 tracer 行为完全一致
assert tracer.read_spans() == []
```

#### 用法 6：回放一条 trace

```python
# 假设 agent 刚跑完，取第一条 span 的 trace_id 作为根
spans = tracer.read_spans()
root_trace_id = [s.trace_id for s in spans if s.parent_id is None][0]
rows = tracer.replay(root_trace_id)   # 返回按时间排序的 dict 列表
for row in rows:
    print(row)
```

输出示例（实际效果）：

```
- [   agent] agent.run               20.31ms  success  -> {'final_answer': '3 的平方是 9'}
  - [decision] decision.step1           0.02ms  success  -> {'action': 'call compute', 'args': {'x': 3}}
  - [    tool] tool.compute            20.08ms  success  -> 9
```

### 8.5 持久化文件结构（jsonl）

每次 `TraceLogger(path="xxx.jsonl", enabled=True)` 追加写入的文件格式：

```
{"trace_id":"trace_a1b2...","span_id":"3c4d...","parent_id":null,"session_id":"sess_x9k2...","name":"agent.run","kind":"agent","start_time":1730620800.123456,"end_time":1730620800.143789,"duration_ms":20.333,"input":{"user_input":"3 的平方是多少？","agent":"助手"},"output":{"final_answer_preview":"9","steps":1},"status":"success","error":null,"metadata":{"agent":"助手"}}
{"trace_id":"trace_a1b2...","span_id":"5e6f...","parent_id":"3c4d...","session_id":"sess_x9k2...","name":"decision.step1","kind":"decision","start_time":1730620800.12,"end_time":1730620800.12,"duration_ms":0.02,"input":{"step":1},"output":{"action":"final","length":1},"status":"success","error":null,"metadata":{"step":1}}
...
```

- 每行一个完整 JSON，字段顺序与 SpanRecord 一致（pydantic model_dump 默认按字段声明顺序）；注意时间字段是**秒级**浮点，`duration_ms` 才是毫秒；
- 文件末尾无空行时 append 仍然安全（JsonlStore 写前会先 `write("\n")` 吗？答：不会，每次 writes 都是 `line + "\n"`，每行自带换行，不管上次写在哪；这是 jsonl 的标准写法）；
- 用任意文本编辑器打开、或 `tail -f xxx.jsonl` 都能实时看到新写入的 span；
- 想清空历史直接删除文件即可，下次 append 会自动创建新文件。

---

## 九、测试说明

测试文件：`test/test_trace.py`，pytest 风格，**全部在内存 / 临时目录中跑**，无外部依赖、无网络请求、不需要 API Key。

### 9.1 测试环境准备

```powershell
pip install pytest     # 仓库已有 dev 依赖
python -m pytest test\test_trace.py -v
```

### 9.2 测试用例清单（共 41 个用例，41/41 已通过）

| # | 用例名 | 覆盖内容 |
|---|---|---|
| **数据模型（4）** | | |
| 1 | `test_span_record_defaults` | `SpanRecord` 构造默认值（duration/end_time/output 为 None，metadata 为 {}） |
| 2 | `test_span_record_finish` | `finish()` 写 end_time/duration_ms/output/status，并合并 metadata |
| 3 | `test_span_record_finish_error_sets_status` | 异常路径 status=error、error 字段写入 |
| 4 | `test_span_kind_enum_values` | `SpanKind` 枚举值：llm/tool/decision/agent/custom |
| **JsonlStore（内存 + 文件，7）** | | |
| 5 | `test_jsonl_store_memory_write_read` | 内存模式 write/read 往返、写入顺序 |
| 6 | `test_jsonl_store_filter_by_session_trace_kind` | 按 session_id/trace_id/kind 过滤 |
| 7 | `test_jsonl_store_limit` | limit 截断 |
| 8 | `test_jsonl_store_memory_path_none_equals_memory` | path=None 等同 `:memory:` |
| 9 | `test_jsonl_store_file_write_and_reopen(tmp_path)` | 文件落盘后重新打开可读（跨实例回放） |
| 10 | `test_jsonl_store_file_overwrite_when_append_false(tmp_path)` | append=False 覆盖旧文件 |
| 11 | `test_jsonl_store_file_is_valid_jsonl(tmp_path)` | 每行都是合法 JSON（jsonl 格式） |
| **TraceLogger 生命周期（8）** | | |
| 12 | `test_tracer_disabled_writes_nothing` | enabled=False 下所有 span 操作零写入（内存模式） |
| 13 | `test_tracer_none_zero_overhead_for_llm_via_default` | 不传 tracer 时 LLM 零打点 |
| 14 | `test_start_end_trace_basic` | `start_trace/end_trace` 基本行为、trace_id 生成 |
| 15 | `test_with_trace_context_manager` | `with tracer.trace(...)` 上下文管理器 |
| 16 | `test_with_span_records_on_normal_exit` | `with tracer.span(...)` 正常退出自动 end |
| 17 | `test_with_span_marks_error_on_exception` | with 块内异常标记 ERROR 并重新 raise |
| 18 | `test_parent_child_relationship` | 嵌套 span 的 parent_id 链正确 |
| 19 | `test_manual_start_end_span_duration` | 手动 start/end 的 duration_ms 与字段覆盖 |
| **工具打点（6）** | | |
| 20 | `test_tool_hook_records_success` | 工具成功后写一条 tool Span（输入/输出/耗时） |
| 21 | `test_tool_hook_records_error` | 工具抛错时记录 error Span |
| 22 | `test_tool_hook_uses_real_duration_from_metadata` | **v1.0.1**：耗时取自 `metadata["duration_ms"]` 并回推 start_time |
| 23 | `test_tool_hook_marks_duration_unavailable_when_missing` | **v1.0.1**：工具未上报耗时时降级，并标记 `duration_source` |
| 24 | `test_tool_hook_records_real_duration_via_registry_end_to_end` | **v1.0.1**：真实工具经 registry 执行，耗时 ≈ 实测值 |
| 25 | `test_attach_to_tool_registry_is_noop_when_disabled` | enabled=False 时不挂钩子 |
| **LLM 门面打点（10）** | | |
| 26 | `test_llm_invoke_with_tracer_records_span` | invoke 打点记录 llm Span |
| 27 | `test_llm_invoke_without_tracer_records_nothing` | 不传 tracer 时不打点 |
| 28 | `test_llm_stream_invoke_with_tracer_records_span` | stream_invoke 记录 Span（chunk 数） |
| 29 | `test_llm_embed_query_with_tracer_records_span` | embed_query 记录 Span |
| 30 | `test_llm_embed_documents_with_tracer_records_span` | embed_documents 记录 Span |
| 31 | `test_llm_invoke_records_token_metadata` | **v1.0.1**：invoke 把 usage/token 写进 metadata |
| 32 | `test_llm_invoke_without_usage_adds_no_token_keys` | **v1.0.1**：无 usage 时不写 token 键（不产生 null 噪音） |
| 33 | `test_llm_embed_query_records_token_metadata` | **v1.0.1**：embed_query 写 token 元数据 |
| 34 | `test_llm_embed_documents_records_token_metadata` | **v1.0.1**：embed_documents 写 token 元数据 |
| 35 | `test_llm_stream_invoke_has_no_token_metadata` | **v1.0.1**：流式不产生 token 元数据（Span 仍正常记录） |
| **ReActAgent 接入（2）** | | |
| 36 | `test_react_agent_with_tracer_records_all_spans` | agent/decision/tool/llm 四类 Span 齐全 |
| 37 | `test_react_agent_without_tracer_works_as_before` | 不传 tracer 时行为与改动前一致（零破坏） |
| **回放 / 扩展 / 会话（4）** | | |
| 38 | `test_replay_returns_ordered_printable_list` | replay 返回按时间排序的 dict 列表，字段结构稳定 |
| 39 | `test_custom_store_can_be_injected` | 自定义 TraceStoreBase 子类可注入（SQLite 预留位验证） |
| 40 | `test_set_session_id_and_filter` | set_session_id 切换 + 按 session 过滤 |
| 41 | `test_trace_logger_context_manager_closes` | `with TraceLogger()` 自动 close 底层 store |

### 9.3 跑测试基线

```
$ python -m pytest test/test_trace.py -q
.........................................                                [100%]
41 passed in 1.60s
```

**41/41 全绿才算交付合格**。运行本模块测试不会影响既有 `test_database.py`/`test_tools.py`：已验证既有 103 个用例仍然全过（103 passed），零破坏性。

### 9.4 既有测试回归（零破坏验证）

每次改 LLM/Agent 接入代码后，必须跑：

```powershell
python -m pytest test\test_database.py test\test_tools.py -v
```

基线：**103 passed**。如果失败，说明接入代码破坏了既有行为，必须回退。

---

## 十、API 速查（一页纸）

### 构造

```python
TraceLogger(
    store: TraceStoreBase | None = None,   # 自定义后端；不传则内部构造 JsonlStore
    *,
    path: str | Path | None = None,        # jsonl 文件路径；不传/None 则 ":memory:" 内存模式
    session_id: str | None = None,         # 不传自动生成 sess_xxx
    enabled: bool = True,                  # 实例级开关；False 时所有公开方法 no-op
)
```

> **「默认不收集」的准确含义**：框架默认不收集 trace——不构造 `TraceLogger`、或不把它传给 LLM/Agent，就零开销、行为与改动前逐字节一致。一旦构造出实例即启用（`enabled=True`）；需要显式关闭时传 `enabled=False`。

### Span 生命周期

| 方法 | 作用 |
|---|---|
| `tracer.start_span(name, kind=SpanKind.CUSTOM, *, input=None, metadata=None, session_id=None) -> SpanRecord` | 开一个 span 并入栈，返回 span 对象（未 finish） |
| `tracer.end_span(span, output=None, status=SpanStatus.SUCCESS, error=None, metadata=None) -> None` | 结束 span、算耗时并写入 store（error 非空自动置 ERROR） |
| `with tracer.span(name, kind=SpanKind.CUSTOM, *, input=None, metadata=None) as span:` | with 语法自动 start/finish；块内异常自动标记 ERROR 并重新抛出 |
| `tracer.start_trace(name="root", session_id=None) -> str` / `tracer.end_trace()` | 手动开/闭一条 Trace（返回 trace_id） |
| `with tracer.trace(name="root", session_id=None) as trace_id:` | with 语法开/闭一条 Trace，**yield 的是 trace_id（字符串）** |
| `tracer.set_session_id(sid)` | 切换 session_id，之后新开的 span 用新 sid |
| `tracer.flush()` / `tracer.close()` | 刷盘 / 关闭底层 store |

### 接入

| 方法 | 作用 |
|---|---|
| `tracer.attach_to_tool_registry(registry) -> None` | 给 ToolRegistry 挂工具钩子，自动记录所有 invoke_tool 调用（走 add_hook，不改注册表） |
| `tracer.tool_hook(name, tool_args, result) -> None` | 钩子本体，签名对齐 `ToolRegistry.add_hook`，可手动挂 |

### 查询

```python
tracer.read_spans(
    session_id: str | None = None,
    trace_id: str | None = None,
    kind: str | None = None,
    limit: int | None = None,
) -> list[SpanRecord]   # store 原始顺序（JsonlStore 即写入顺序）
```

### 回放

```python
tracer.replay(trace_id: str | None = None, session_id: str | None = None) -> list[dict]
# 返回按 start_time 排序、字段稳定的可打印 dict 列表（可自行格式化或用示例中的打印辅助）
```

### 枚举

```python
SpanKind.LLM / TOOL / DECISION / AGENT / CUSTOM
SpanStatus.SUCCESS / ERROR
```

### 常量

```python
MEMORY_PATH = ":memory:"   # 传给 path 表示内存模式
```

### 导出（从 fast_agents 根包直接拿）

```python
from fast_agents import (
    TraceLogger, TraceStoreBase, JsonlStore, MEMORY_PATH,
    SpanRecord, SpanKind, SpanStatus,
)
```

---

## 十一、模块文件清单（交付物）

| 文件路径 | 类型 | 说明 |
|---|---|---|
| `fast_agents/trace/__init__.py` | 新增（覆盖原有占位） | 模块出口，re-export 公共符号 |
| `fast_agents/trace/models.py` | 新增 | SpanRecord/SpanKind/SpanStatus + 辅助纯函数 |
| `fast_agents/trace/store_base.py` | 新增 | TraceStoreBase 抽象基类（write/read/iter_records/close/flush） |
| `fast_agents/trace/jsonl_store.py` | 新增 | JsonlStore 实现（jsonl 落盘 + 内存模式）+ `MEMORY_PATH` 常量 |
| `fast_agents/trace/trace_logger.py` | 新增（v1.0.0）/ 修改（v1.0.1） | TraceLogger 核心门面；v1.0.1 修正 `tool_hook` 耗时 |
| `fast_agents/llm/llm.py` | 修改 | invoke/stream_invoke/embed_query/embed_documents 加可选 `tracer` 参数并打点；v1.0.1 补 token/model 元数据 |
| `fast_agents/llm/openai_client.py` | 修改（v1.0.1） | 新增可选 `usage_sink` 通道，把响应 usage 回传上层（不改任何返回值语义） |
| `fast_agents/agents/react_agent/react_agent.py` | 修改 | `__init__` 加 `tracer=None`（非 None 时挂工具钩子、把 tracer 设为 `llm.tracer` 缺省）；`run` 在 tracer 非 None 时包 agent/decision span |
| `fast_agents/__init__.py` | 修改 | 导出 trace 模块公共符号 |
| `test/test_trace.py` | 新增 | 41 个 pytest 用例，全内存/临时目录 |
| `examples/trace/basic_usage.py` | 新增 | 最小示例：memory tracer + 手动 span + 回放（不需要 API Key） |
| `examples/trace/jsonl_replay.py` | 新增 | 落盘示例：工具打点 + 文件落盘 + 跨实例回放（不需要 API Key） |
| `docs/trace模块编写说明.md` | 新增（本文档） | v1.0.1 设计与使用说明 |
| `README.md` | 修改 | 项目结构小节新增 `trace/` 一行说明 |
| `docs/演进路线图.md` | 修改 | G1 状态：v1.0.0 已完成 → v1.0.1 完成 G1 patch |

**不改动**：`pyproject.toml`、`requirements.txt`（零新增依赖）；`tool_registry.py`、`agent_base.py` 等所有其他既有文件保持原样（v1.0.1 仅在 `openai_client.py` 新增可选 `usage_sink` 通道，属向后兼容）。

---

## 十二、已知限制与后续演进

### 12.1 已知限制（v1.0.1 诚实声明）

| 限制 | 影响 | 缓解方式 |
|---|---|---|
| 单线程同步 only；多线程写同一文件靠 Lock 保安全，但跨进程写同一 jsonl 可能混行 | 多 worker 部署时不要共享同一个 jsonl 路径 | 每个 worker 用独立文件（`path=f"logs/{pid}.jsonl"`） |
| asyncio 场景 parent_id 无法跨协程自动传播 | async agent 内嵌的 span 会丢父链 | G2 用 `contextvars.ContextVar` 实现 |
| jsonl 读全量进内存再过滤，trace 量巨大（>100 万条）时慢 | 长会话跑很久后 read_spans 慢 | G2 切 SQLiteStore，对 trace_id/kind/start_time 建索引 |
| 输入输出为轻量摘要（LLM 输入只记计数、agent 输出预览截 300 字符），拿不到完整 LLM 输入输出 | 完整复盘可能丢失长文本细节 | G3 评估工具增加"全量落盘"模式选项 |
| ~~不记录 token 数、model_name~~ **（v1.0.1 已补）**：LLM span 的 metadata 写入 usage / prompt_tokens / completion_tokens / total_tokens + model | 成本分析已可用；但流式调用仍无 usage | 流式 token 统计列 G2 |
| tool span 耗时取自 `ToolResult.metadata["duration_ms"]`，未上报时退化为近零耗时 | 自定义工具若不走 `ToolBase.invoke` 则无准确耗时 | metadata 标 `duration_source=unavailable` 可识别 |
| SpanKind 只有 5 种；没有 Event（span 内的时间点事件）| 无法记录"首个 token 到达时间"、"工具解析开始"等细粒度点 | G2 加 SpanEvent |
| 没有采样（全量记录或完全不记录）| 高流量场景磁盘占用高 | G2/G3 加 Sampler（按比例/按错误率采样） |
| replay 返回 dict 列表，无内建渲染；没有 Web UI | 可视化体验一般 | G4 做 Web UI |
| stream_invoke 只记录 chunk 数量，不记录流式增量内容 | 流式输出过程不可观测 | G2 加 on_chunk 事件 |

### 12.2 后续计划（对齐路线图 G 系列）

| 版本 | 计划内容 | 依赖 | 状态 |
|---|---|---|---|
| **G1 patch（v1.0.1）** | ① LLM 打点补 token/model 元数据（`usage_sink` 通道 + metadata）；② 修正 tool span 耗时（读 `metadata["duration_ms"]` 并回推 start_time） | 无 | ✅ **已完成** |
| G1 patch（v1.0.x 后续）| 输入输出截断可配置、错误处理打磨 | 无 | 待做 |
| G2（v1.1.0） | asyncio 支持（contextvars）、多进程安全文件锁、SQLiteStore 可选后端、采样器、SpanEvent、流式 token 统计 | database 模块 v1.1 加批量插入优化 | 待做 |
| G3（v1.2.0） | 评估工具：基于 trace 自动算工具选择准确率、**token 消耗**、平均 step 数、错误分类；全量落盘模式 | G2 稳定 | 待做（token 原始数据 v1.0.1 已就绪） |
| G4（v1.3.0+）| Web UI：trace 列表、trace 详情树、瀑布图、筛选过滤；可选对接 OpenTelemetry Collector / Langfuse | 前端工程栈确定 | 待做 |

### 12.3 贡献者须知

- **新增 SpanKind**：必须在 `SpanKind` 枚举里加一项，并在本文档 §6.2、§9.2 测试表、§10 API 速查里同步更新；接入层打点时 name 前缀约定（`llm.*/tool.*/decision.*/agent.*`）保持一致。
- **新增 Store 后端**：继承 `TraceStoreBase`，把三个抽象方法实现全；加到 `fast_agents/trace/__init__.py` 和根 `__init__.py` 的导出；加一份 `examples/trace/xxx_store_demo.py`；在 §7 补一小节源码导读。
- **新增接入点**（比如未来给 MCP 客户端、RAG 检索器打点）：必须遵守"只加可选 `tracer=None` 参数"的红线，**绝不**修改既有方法签名的其他参数/返回值；打点通道所有异常必须吞掉。
- **底层新增回传通道**（如 v1.0.1 的 `usage_sink`）：必须是**可选关键字参数**、默认 None 时行为不变，且不得污染透传给 SDK 的 kwargs。
- **改换行符**：所有 `.py/.md/.txt` 文件必须保存为 **CRLF**（`.gitattributes` 强制 eol=crlf）；交付前用 `file` 命令或编辑器检查，不允许出现裸 LF。
- **跑测试**：每次改动必须同时过 `test/test_trace.py`（41/41）、`test/test_database.py` + `test/test_tools.py`（103/103），三者全绿才算通过。
