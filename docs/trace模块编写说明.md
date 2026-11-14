# FastAgents Trace 模块编写说明

本文档面向 fast_agents **Trace（全链路可观测性）**系统，参考 **OpenTelemetry Trace/Span 模型**、**LangChain Callbacks/Tracer**、**OpenAI Agents SDK Tracing** 的核心概念（trace / span / 父子关系 / 钩子），给出本模块的完整说明、源码解析与自定义教程。同时对齐 fast_agents 已交付的 tools / memory / mcp / skills / context / database 模块文档骨架，做到「翻开任何一份模块说明，都知道在哪一节能找到什么」。

> **版本**：v1.2.0 · 2026-11-12 · 路线图 K1 最小闭环 + K3 评估工具/成本日报 + K1 收尾四项：Trace/Span 数据模型 + TraceLogger 记录器 + JsonlStore（默认）+ SQLiteStore（结构化后端）+ ToolRegistry 钩子打点 + LLM/Agent 可选接入 + LLM token/model 元数据 + tool 真实耗时 + 查询/回放 + 后端自动后缀识别 + **TraceEvaluator 离线评估器**（成本/效率/质量三类指标聚合、dict/JSON/文本/CSV 四格式输出）+ **Token 成本日报**（按自然日聚合、模型价目表自动匹配、JSON/文本/CSV 三格式导出）+ **K1 收尾四项**（可配置预览截断长度、`replay`/`build_tree` 树形聚合、`error_type` 异常分类、`perf_counter` 单调时钟耗时）
> **适用版本**：`fast_agents.trace`
> **配套代码**：`fast_agents/trace/`（**9 个**源码文件）+ `tests/trace/`（**5 个测试文件、共 149 项**全绿：`test_trace.py` 74 项 K1 + `test_trace_jsonl_store.py` 12 项 JsonlStore + `test_trace_sqlite_store.py` 11 项 SQLiteStore + `test_trace_eval.py` 33 项 K3 评估器 + `test_trace_cost.py` 19 项 K3 成本日报）+ `examples/trace/`（**6 个**可运行示例）
> **运行环境**：Python 3.10+（仓库 `.python-version` = 3.10），**Windows 平台**（仓库当前唯一目标平台，见 `AGENTS.md` R11）
> **依赖**：**零新增依赖**，仅标准库（`json / os / threading / uuid / time / contextvars / pathlib / contextlib / enum / abc / itertools / sqlite3 / statistics / dataclasses / datetime`）+ 全框架已有的 `pydantic>=2.14.1` + 框架自带 `database.SQLiteClient`；不引入 OpenTelemetry / LangChain / loguru / pandas / numpy 任何外部库
> **本文只写「已经实现的东西」**。「还没做但打算做」的部分放在第十二章，并与 `docs/演进路线图.md` 的 K 模块编号对齐。

---

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：Trace 解决什么问题](#二先想清楚trace-解决什么问题)
- [三、主流框架 Trace 实现对比](#三主流框架-trace-实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：Trace/Span 模型与打点通道](#五核心机制tracespan-模型与打点通道)
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
| **1.2.0** | 2026-11-12 | **K1 收尾四项交付 + 文档对齐最新代码（零新增依赖、零破坏性改动）**：①**输入/输出预览截断长度可配置**——把硬编码在打点接入层的预览长度收编为 `TraceLogger(input_preview_len=..., output_preview_len=...)` 两个可选 keyword 参数 + 门面方法 `truncate_preview(text, default_len, *, is_input=False)`；默认 `None` 时沿用 Agent 输入 500 / Agent 输出 300 / LLM 输出 200，行为**逐字节不变**，配置 `<= 0` 表示不截断；`tests/trace/test_trace.py` 50 → **56 项**，合计 124 → **130/130**。②**`replay()` 返回树形结构 + `build_tree()` 辅助**——`replay()` 返回 dict 追加 `children` 递归字段（同层按 `_create_seq` 排序、悬空父提升为 orphan root、多 root 保留），新增纯只读 `build_tree(trace_id=None, session_id=None)`；`test_trace.py` 56 → **60**，合计 130 → **134/134**。③**统一 `error_type` 异常分类字段**——`SpanRecord` 新增可选 `error_type`，`finish()` / `end_span()` 自动从 `"ExceptionName: msg"` 切取异常类名（ASCII，允许点号限定名），工具错误统一 `"ToolError"`，SQLite schema 加列（老库自动 ALTER 补列、历史数据 None 不回填）；`test_trace.py` 60 → **65**，合计 134 → **139/139**。④**`duration_ms` 改用 `perf_counter()` 单调时钟差值（抗墙钟跳变）**——`TraceLogger.start_span` 新建 Span 时记录私有单调时钟起点 `_mono_start`（不落盘），`finish()` 默认路径走 `perf_counter()` 差值；`start_time` / `end_time` 仍为 `time.time()` 墙钟，显式 `end_time`（后钩子 / 手工精确起止）或直接构造的 `SpanRecord` 仍走墙钟差值，口径与旧版**逐字节一致**；`test_trace.py` 65 → **70**，合计 139 → **144/144**。⑤**文档对齐最新代码 `e2ae4c2`（零源码改动、零依赖变更）**——测试基线 113 → 124，`evaluate()` 的 `limit` / `top_n_traces` 统一为「trace 粒度」截断（按 trace 最近活跃时间倒序取最近 N 条**完整 trace**，不再按 span 条数切片丢 root），序列化兜底函数名 `_safe_serialize` 更正为 `json_safe_dumps` / `json_safe_serialize`。⑥上游并入 2 项回归用例（`test_parent_child_relationship_with_error`、`test_no_root_with_success_span_still_success`）；⑦**内存后端生命周期语义修正**——`InMemoryTraceStore.close()` 改为 no-op（不再清空记录，`TraceLogger` 退出 with 块自动 close 不再丢数据），主动清空职责移交扩展方法 `clear()`；`tests/trace/test_trace.py` 71 → **74 项**，trace 模块当前基线 **74 + 12 + 11 + 33 + 19 = 149/149** 全绿。 | — |
| **1.1.0** | 2026-11-10 | **K3 离线评估与成本日报交付（零新增默认依赖、零破坏性改动）**：①**TraceEvaluator 离线评估器**——新增 `fast_agents/trace/evaluator.py`，交付 `TraceEvaluator` / `TraceMetrics` / `EvaluationReport` 三个公开类，离线消费已落盘 trace（TraceLogger / Store / 文件路径三种构造方式，JsonlStore 与 SQLiteStore 通用），按 trace 聚合成本 / 效率 / 质量三类指标，支持 `session_id` / `trace_id` / `limit` / `top_n_traces` 过滤，提供 `to_dict()` / `to_json()` / `to_text()` 三种输出；新增 `tests/trace/test_trace_eval.py` 21 项，trace 模块合计 67 → **88/88**。②**CSV 明细导出（`to_csv`）**——`EvaluationReport.to_csv(path=None, *, encoding="utf-8-sig")`，固定 25 列，`success` 输出 `true`/`false`、`tool_distribution` 折叠为 `name=count;...`，默认带 BOM 落盘（Windows Excel 中文不乱码）；`test_trace_eval.py` 21 → **27**，合计 88 → **94/94**。③**Token 成本日报（`daily_cost_report`）**——按自然日聚合（默认北京时间 UTC+8，可切时区），内置 22 个常见模型默认价目表（按最长前缀优先匹配），新增 `DayCostMetrics` / `CostDailyReport`，提供 dict/JSON/文本/CSV 四输出；新增 `tests/trace/test_trace_cost.py` 16 项，K1+K3 合计 94 → **110/110**。 | — |
| **1.0.0** | 2026-11-06 | **初版（路线图 K1 最小闭环）**：`fast_agents/trace/` 作为独立子包交付（`__init__.py` / `models.py` / `store_base.py` / `jsonl_store.py` / `sqlite_store.py` / `trace_logger.py` 共 6 个文件）。一次性交付：**SpanRecord** 数据模型（trace_id/span_id/parent_id/session_id/name/kind/start_time/end_time/duration_ms/input/output/status/error/metadata）+ **SpanKind** 枚举（llm/tool/decision/agent/custom）+ **SpanStatus** 枚举（success/error）+ **TraceLogger** 门面（手动 span / with 上下文 / trace() 开新根 / 失败安全 / 幂等 end_span / `read_spans` 多条件过滤 / `replay` 缩进回放 / 后端自动识别）+ **TraceStoreBase** 抽象基类 + **JsonlStore**（文件后端，jsonl 逐行 append 落盘、坏行跳过）+ **InMemoryTraceStore**（纯内存后端）+ **SQLiteStore**（结构化后端，复用框架自带 `database.SQLiteClient`，一条 Span 一行、三索引谓词下推、WAL 并发写、幂等迁移）。**接入**：LLM 门面 `invoke / stream_invoke / embed_query / embed_documents` 仅新增可选 `tracer=None` 打点（stream 走包装生成器），LLM span 写入 `model` + `usage` / `prompt_tokens` / `completion_tokens` / `total_tokens`；ReActAgent 新增可选 `tracer=None`，run 内开 `agent.<name>.run` trace + `agent.run` Span + 每步 `decision.stepN` Span；工具打点走 `ToolRegistry.add_hook` 钩子，读 `ToolResult.metadata["duration_ms"]` 回推真实耗时。**零新增依赖**、**零破坏性改动**：`tests/trace/test_trace.py` **67/67** 全绿；`examples/trace/` 下 4 个离线可跑示例；顶层 `fast_agents/__init__.py` 同步重导出 trace 全部公开符号。 | — |

---

## 二、先想清楚：Trace 解决什么问题

### 2.1 一句话概括

**Trace 模块是 fast_agents 的全链路可观测性底座——给 Agent 一次 run 里发生的每一次 LLM 调用、工具调用、Agent 决策，都打上带父子关系的 Span 记录，默认 JSONL 一行一条落盘，也支持 SQLite 结构化存储方便高效查询，方便事后回放、排错、做成本统计，且对现有代码**零侵入**：不传 tracer 时行为与改动前逐字节一致。**

没有 Trace 的时候，Agent 开发者只能靠散落的 `print` 猜：这一次模型到底调了几次？工具返回了什么？哪一步耗时最长？为什么选了这个工具？总共花了多少 token？Trace 模块就是为这些问题提供统一的、机器可读的答案。

### 2.2 核心概念（对齐 OpenTelemetry，但做减法）

| 概念 | OpenTelemetry 原义 | 本框架对应 | 生命周期 | 典型内容 |
|---|---|---|---|---|
| **Trace** | 一次分布式请求的完整调用链 | 一次 Agent run（或用户手动开启的一条链路）| 从 `start_trace` 到 `end_trace`；`with tracer.trace():` 自动成对 | 整次对话/任务的所有 Span 共享同一个 `trace_id` |
| **Span** | 链路上的单个操作单元 | 一次 LLM 调用 / 一次工具调用 / 一次 decision 决策 / 一次 agent run / 一次自定义打点 | 从 `start_span` 到 `end_span`；`with tracer.span():` 自动成对；`tool_hook` 自动成对 | 输入、输出、起止时间、耗时、状态、错误、扩展元数据 |
| **SpanContext** | 跨进程传递的 trace/span id | 同进程内用**实例级 `contextvars.ContextVar`** 维护当前 trace/span 栈（每个 `TraceLogger` 实例自持一对） | 随调用栈入栈出栈 | 实例间互不串扰；同线程 / 同协程内自动父子；`asyncio.Task` 会拷贝上下文，故协程内自动继承父链 |
| **Exporter / SpanProcessor** | Span 结束后的导出/批处理 | 直接写 `TraceStoreBase.write(span)`，无批处理、无异步队列 | 当前为同步写；异步/批处理留后续版本 | 当前实现 `JsonlStore`（默认）与 `SQLiteStore`（可选）同步写 |

### 2.3 关键设计原则（全篇代码都在落实它们）

1. **默认关闭（零开销）**：不构造 `TraceLogger` 或不传 `tracer=` 给 LLM/Agent，代码路径与改动前逐字节一致——这是路线图 K1 的硬约束，也是模块能放进核心路径的前提。`TraceLogger(enabled=True)` 本身默认启用，但你得**显式构造并传进去**才会产生记录。
2. **零新增依赖**：只用标准库 + 全框架已有的 `pydantic v2` + 框架自带 `database.SQLiteClient`。不引 OpenTelemetry SDK、不引 LangChain Callback、不引 loguru——保持 fast_agents「零重型依赖」的招牌。
3. **旁路不阻塞**：打点通道任何异常（序列化失败、文件写失败、钩子报错）都必须被吞掉，**绝不向上抛**阻塞主业务——这是可观测性模块的铁律。
4. **存储抽象可替换**：`TraceStoreBase` 定义 `write / read / close` 最小契约，默认实现是 `JsonlStore`（一行一个 JSON，最容易被 `jq / pandas / 日志采集器`消费），并已内置 `SQLiteStore`（结构化索引查询、多进程写更稳），未来切换远端上报只改存储实现，上层打点代码零改动。
5. **最小字段集 + 推荐补充**：严格覆盖路线图要求的「时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id」最小集，并额外补充 OpenTelemetry 风格的 `trace_id / span_id / parent_id / name / status / error / metadata`，方便后续做链路还原与成本统计。
6. **接入只走扩展点，不改核心**：工具打点一律走 `ToolRegistry.add_hook`（不改注册表逻辑）；LLM/Agent 只新增**可选参数** `tracer=None`，不改既有函数签名、不改返回值、不改既有测试断言。

---

## 三、主流框架 Trace 实现对比

### 3.1 对比总表

| 方案 | 定位 | 依赖 | 接入方式 | 存储/导出 | 优点 | 缺点 |
|---|---|---|---|---|---|---|
| **OpenTelemetry Python SDK** | 通用可观测性标准 | opentelemetry-api + opentelemetry-sdk + 各 exporter（~MB 级） | TracerProvider / start_as_current_span / Context 传播 | SpanProcessor + Exporter（OTLP/Console/Zipkin…）| 生态最大、标准最完整、跨语言、跨进程 | 对轻量 Agent 框架过重；概念多（SpanKind/Status/Event/Link/Resource）；exporter 生态复杂；对「只想把 LLM/tool 打点落 jsonl」的场景杀鸡用牛刀 |
| **LangChain Callbacks / Tracer** | LangChain 专属可观测性 | langchain-core（内置）| CallbackManager + handler 基类（on_llm_start/on_tool_end 等）| 内置 Console/File/LangSmith Tracer | 与 LLM/Tool/Agent 生命周期绑定最深，开箱即用 | 强绑定 LangChain 抽象体系，拿出来单独用几乎不可能；回调方法众多（start/end/error 三件套），接入面大 |
| **OpenAI Agents SDK Tracing** | OpenAI Agents SDK 专属 | agents-sdk（内置）| `trace()` / `span()` 上下文管理器 + 自动 Agent/LLM/Tool 打点 | 默认本地 trace 文件 + 可选上传 OpenAI 平台 | API 设计简洁（with 语法）、自动打点覆盖好、内置 trace viewer | 强绑定 OpenAI Agents SDK；默认会上报云端（隐私/离线场景需关闭）；本地存储格式不公开稳定 |
| **loguru + 装饰器手写** | 通用日志打点 | loguru | 自己写装饰器/上下文管理器 | 日志文件 | 轻、灵活 | 没有 Trace/Span 父子链概念、没有标准化字段、没有存储抽象、回放/分析全靠自己 grep |
| **fast_agents（本模块）** | **框架内置的最小可跑闭环** | **零新依赖**（标准库 + pydantic + 框架自带 database）| `tracer=` 可选参数注入 + `ToolRegistry.add_hook` 工具钩子 + `with tracer.span(...)` 手动打点 | **默认 InMemoryTraceStore**（`:memory:` 进程内内存，测试友好）；可选 **JsonlStore**（一行一条 JSON 落盘）与 **SQLiteStore**（结构化索引查询、多进程写稳）；存储抽象预留远端上报位 | 零依赖、零侵入、旁路不阻塞；API 克制（一个 TraceLogger 搞定）；字段对齐路线图最小集；状态栈实例级隔离且可随协程传播；支持文件后缀自动识别后端 | 初版不做异步并发写入/批处理、不做 Web UI、不做远端上报；不支持 SpanEvent / Link；无采样 |

### 3.2 关键差异：相同 / 增加 / 缺少

| 维度 | 与三者相同 | 相对最简方案（loguru）增加 | 目前缺少（诚实列缺项） |
|---|---|---|---|
| 核心模型 | Trace/Span 父子关系、SpanKind/Status 分类、metadata 扩展 | trace_id/span_id/parent_id 链路字段、自动计时、自动状态（异常→error）| SpanEvent（时间点事件）、SpanLink（跨 trace 关联）、Resource（进程/主机属性） |
| 接入方式 | with 上下文管理器自动成对、手动 start/end 兜底 | LLM/Agent 可选 `tracer=` 参数自动打点；ToolRegistry 钩子一行接入工具；按文件后缀自动识别存储后端 | 装饰器式打点（`@traced`）、async 场景的写入侧接入（同协程父子链已自动） |
| 存储 | 同步写、落盘可查 | `TraceStoreBase` 抽象、InMemoryTraceStore 纯内存、JsonlStore 零依赖文件落盘、SQLiteStore 结构化索引 | 异步批量写、远端 OTLP/Langfuse exporter |
| 上下文传播 | 当前线程/上下文自动父子 | 实例级 ContextVar 栈：多 tracer 实例互不串扰，且随 `asyncio.Task` 拷贝上下文传播 | 跨线程/跨进程传播、Baggage（透传 KV）|
| 性能 | — | 打点通道异常全吞、不阻塞主业务 | 采样（Sampler）、背压、限速 |
| 消费 | 读取回放 | `read_spans` 按条件过滤 + `replay` 按时间排序平铺；SQLiteStore 谓词下推大 trace 量查询更快 | 瀑布图 UI、Token 成本聚合、工具选择准确率评估（K3/K4）|

一句话收口：**本模块 = OpenTelemetry 的 Trace/Span 核心概念 + OpenAI Agents SDK 简洁的 with API + LangChain 自动打点思路 — 所有重型依赖与复杂抽象，保留 K1 最小可跑闭环 + 结构化存储底座**。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/
└── trace/                              # 单数目录（对齐 framework.md §七）
    ├── __init__.py                     # 模块出口，统一导出公开符号（含 K3 TraceEvaluator 等评估类）
    ├── models.py                       # SpanKind / SpanStatus / SpanRecord 数据模型
    ├── store_base.py                   # TraceStoreBase 抽象基类（存储契约）
    ├── in_memory_trace_store.py        # InMemoryTraceStore 纯内存后端（TraceLogger() / store=":memory:" 的默认落点）
    ├── jsonl_store.py                  # 默认 JsonlStore 落盘实现（一行一 JSON）
    ├── schema.py                       # SQLiteStore 的声明式建表 Schema（表结构 + 索引）
    ├── sqlite_store.py                 # SQLiteStore 结构化落盘实现（一条 Span 一行）
    ├── trace_logger.py                 # TraceLogger 门面：核心记录器、打点入口、栈维护、查询/回放
    └── evaluator.py                    # K3 TraceEvaluator 离线评估器：TraceMetrics / EvaluationReport / dict·JSON·文本·CSV 输出

tests/trace/
├── test_trace.py                       # 74 项测试，全绿；覆盖 K1 打点链路、LLM/Agent/tool 钩子、生命周期、预览截断可配置、replay/build_tree 树形聚合、error_type 自动分类、单调时钟耗时（抗墙钟跳变）、内存后端 close 不清空 / clear 主动清空
├── test_trace_jsonl_store.py           # 12 项测试，全绿；覆盖 JsonlStore 文件后端、序列化兜底、坏行跳过
├── test_trace_sqlite_store.py          # 11 项测试，全绿；覆盖 SQLiteStore 谓词过滤、幂等迁移、upsert、后缀识别
├── test_trace_eval.py                  # 33 项 K3 评估器测试，全绿；覆盖构造/聚合/过滤/输出/CSV/脏数据/零开销
└── test_trace_cost.py                  # 19 项 K3 成本日报测试，全绿；覆盖按日聚合/价目表/时区/CSV 导出

examples/trace/
├── basic_usage.py                      # 最小手动打点 + Agent/LLM/工具全链路示例（离线、无需 API Key）
├── jsonl_demo.py                       # 证明「文件模式不把全量数据留在内存」的内存占用示例（离线）
├── jsonl_replay.py                     # Jsonl 落盘 + 跨实例回放示例（离线、无需 API Key）
├── sqlite_usage.py                     # SQLiteStore 落盘 + 谓词过滤 + 流式遍历示例（离线、无需 API Key）
├── eval_demo.py                        # K3：模拟 3 次 Agent 运行 + 离线评估 + 文本/JSON/CSV 三格式输出示例（离线）
└── cost_daily_demo.py                  # K3：模拟 3 天 Agent 运行 + Token 成本日报 + JSON/CSV 导出示例（离线）
```

（其余 LLM / Agent / ToolRegistry / database 目录在各自模块文档里展开，Trace 侧只通过既有扩展点接入，不改动这些目录的源码。）

### 4.2 分层图（自底向上）

```text
┌─────────────────────────────────────────────────────────────┐
│  使用方：LLM / ReActAgent / 用户自定义代码（可选 tracer=注入）   │ ← 零侵入：不传 tracer = 行为逐字节不变
├─────────────────────────────────────────────────────────────┤
│  TraceLogger 门面：start_trace / start_span / trace / span    │
│   · 实例级 ContextVar 父子栈（trace_stack + span_stack）       │ ← 多 tracer 互不串扰；async 天然传播
│   · tool_hook（对接 ToolRegistry.add_hook）                   │
│   · read_spans / replay / flush / close / __enter__/__exit__  │
├─────────────────────────────────────────────────────────────┤
│  TraceStoreBase 抽象：write / read / iter_records / close     │ ← 存储可替换契约
├──────────────────┬──────────────────────────────────────────┤
│ InMemory（默认）   │  SQLiteStore（结构化后端）                 │ ← 零新依赖
│  · 进程内列表       │  · 复用 database.SQLiteClient              │
│  · clear() 清空     │  · trace_spans 表，一条 Span 一行           │
│  JsonlStore（文件） │  · trace_id/session_id/kind 三索引          │
│  · 一行一 JSON      │  · WAL + busy_timeout 多进程安全             │
│  · append + fsync  │  · upsert 幂等、Schema 自动迁移              │
│  · 读时流式扫描     │  · iter_records 分批流式遍历                 │
└──────────────────┴──────────────────────────────────────────┘
```

### 4.3 数据流与调用链路

**链路 1：手动 span（with 语法）**

```text
with tracer.trace("demo") as tid:                 # trace_logger.py
    with tracer.span("hello", kind=CUSTOM) as s:  # 压栈、parent_id = 当前栈顶
        s.output = "world"                        # 直接赋值
    # __exit__：自动 end_span → finish() → store.write() → 弹栈
```

**链路 2：LLM invoke 打点（传了 tracer 后）**

```text
LLM.invoke(messages, tracer=...)                  # llm.py
  → if tracer:
        with tracer.span("llm.invoke", kind=LLM, input={"messages": ..., "model": ...}) as s:
            resp = client.invoke(..., usage_sink=s.metadata)  # 旁路 dict 收集 usage
        # 退出时 s.metadata 已写入 model / prompt_tokens / completion_tokens / total_tokens
```

**链路 3：ToolRegistry 工具自动打点**

```text
registry.invoke_tool(name, args)                  # tool_registry.py
  → result = tool.run(args)
  → for hook in self._hooks: hook(name, kwargs, result)
        └─ tracer.tool_hook(name, args, result)  # trace_logger.py
              → 读 result.metadata["duration_ms"]  ← tool_base 用 perf_counter 实测写入
              → start_time = now - duration_ms/1000  # 回推开始时刻（后钩子语义）
              → s = start_span("tool.<name>", kind=TOOL, input=args,
                               metadata={"tool_name": name, "tool_status": status_val})
              → s.start_time = start_time
              → end_span(s, output=result.output, status=SUCCESS/ERROR, error=result.error)
```

**链路 4：Agent 整条 run（ReActAgent.run 传了 tracer 后）**

```text
ReActAgent.run(...)                              # react_agent.py
  → if self.tracer:
        self.tracer.start_trace(name=f"agent.{self.name}.run")
        _agent_span = start_span("agent.run", kind=AGENT, input={"user_input": ..., "agent": ..., "max_steps": ...})
  → 每一步 ReAct 循环：
        _dec_span = start_span("decision.stepN", kind=DECISION, input=messages, metadata={...})
        llm.invoke(messages, tools=...)          # 自动产生 llm.invoke span（父是 _dec_span）
        if 要调工具:
            tool_registry.invoke_tool(name, args) # 自动产生 tool.* span（父是 _dec_span）
        end_span(_dec_span, output=parsed, ...)
  → 循环结束：end_span(_agent_span, output={"final_answer_preview": ..., "steps": ...}, status=SUCCESS)
  → 异常：end_span(_agent_span, output=None, error=...)
  → end_trace()
```

### 4.4 与 ReActAgent / LLM / ToolRegistry 的接入点与边界

1. **LLM 侧（`fast_agents/llm/llm.py`）**：
   - `LLM.__init__(self, ..., tracer=None)` 新增可选参数存为 `self.tracer`；
   - `invoke / stream_invoke / embed_query / embed_documents` 各自新增可选 `tracer=None` 参数；**方法级 tracer 优先、实例级 tracer 兜底**，两者都 None 时走原逻辑零开销；
   - span 名：`llm.invoke` / `llm.stream_invoke` / `llm.embed_query` / `llm.embed_documents`；
   - token 元数据：底层 `OpenAIClient` 新增可选关键字 `usage_sink: dict`，LLM 层把 `usage / prompt_tokens / completion_tokens / total_tokens` 合并进 span.metadata；stream 模式无 token 数据。

2. **Agent 侧（`fast_agents/agents/react_agent/react_agent.py`）**：
   - `ReActAgent.__init__(self, ..., tracer=None)` 新增可选参数；
   - 若 tracer 非 None：
     - 若 `self.llm.tracer` 还没被用户手动设置，则自动把 tracer 挂给 LLM（不覆盖用户定制）；
     - 调用 `tracer.attach_to_tool_registry(self.tool_registry)` 一行挂工具钩子；
   - `run()` 内部：开始时开 `agent.<name>.run` trace + `agent.run` Span，每步循环开 `decision.stepN` Span（父是 agent Span），结束时统一 end；异常时 agent span 自动记 ERROR。

3. **工具侧（`fast_agents/tools/tool_registry.py`）**：
   - **不修改 ToolRegistry 源码**，工具打点只走现有 `add_hook(hook)` 扩展点；
   - 钩子签名固定为 `hook(tool_name: str, tool_args: dict, result: ToolResult) -> None`，trace 通过 `tool_hook` 方法对接；
   - 工具真实耗时由 `tool_base.py` 在执行前后用 `time.perf_counter()` 实测，写入 `result.metadata["duration_ms"]`（int 毫秒），trace 侧读取该字段回推 start_time。

4. **存储后端选择逻辑**：
   - `store` 是唯一存储入口：可为 `TraceStoreBase` 实例、路径字符串或 `None`；
   - 传 `TraceStoreBase` 实例时直接使用该实例（最高优先级）；
   - `store=":memory:"` 或 `store=None` → **InMemoryTraceStore**（纯内存，默认）；
   - 其它字符串按后缀自动识别：`.db/.sqlite/.sqlite3` → SQLiteStore；其余（`.jsonl/.ndjson` 等）→ JsonlStore；
   - 传既非字符串又非 `TraceStoreBase` 的类型时抛 `TypeError`。

5. **单一数据源与硬约束**：
   - Trace 是**旁路**，不改变 LLM/Agent/Tool 的返回值、不改变异常传播、不改变消息内容；
   - 打点通道所有异常都必须吞掉，不允许把 trace 的 bug 变成主业务的 bug；
   - 不新增任何构造函数必填参数；`tracer` 一律可选，默认 None。

---

## 五、核心机制：Trace/Span 模型与打点通道

本章是 trace 模块的弹性节（对应模版 §5），讲清楚"Trace 和 Span 到底怎么协作、父子链怎么维护、打点通道怎么保证不阻塞业务"。

### 5.1 Trace/Span 生命周期对比

| 维度 | Trace（一条链路） | Span（一个操作单元）|
|---|---|---|
| 作用域 | 一次 run 内所有 Span 的容器 | 链路上的单个操作 |
| 显式启动 | `start_trace(name)` / `with tracer.trace(name) as tid` | `start_span(name, kind, input=, metadata=)` / `with tracer.span(...) as s` |
| 自动启动 | 首个 start_span 时若没有活跃 trace，会懒启动一条名为 "default" 的 trace | 无 |
| 结束 | `end_trace()` / with 块退出 | `end_span(s, output=, status=, error=, metadata=, end_time=)` / with 块退出（异常自动 ERROR）|
| 数据载体 | 不持久化自己，只作为 trace_id 共享给 Span | `SpanRecord` pydantic 模型，结束时写 store |
| 栈 | `_trace_stack_var`（`ContextVar[tuple]`，元素为 trace_id） | `_span_stack_var`（`ContextVar[tuple]`，元素为 span_id） |

### 5.2 父子链怎么维护

父子关系的核心是**当前活跃 span_id**：

1. **状态唯一载体**：每个 `TraceLogger` 实例在 `__init__` 里自持一对**实例级 `contextvars.ContextVar`**——`_trace_stack_var` / `_span_stack_var`，栈用不可变 `tuple` 表示（ContextVar 推荐不可变值，空栈为 `()`）。实例隔离（多 tracer 互不串扰）+ 上下文隔离（`asyncio.Task` 自动拷贝上下文，协程内天然继承）一次拿全；
2. `start_span` 时，parent_id = `_span_stack_var.get()[-1]`（栈顶即当前活跃 span），空栈时 parent_id 为 None；随后 `_push_span` 把新 span_id 压栈；
3. `end_span` 时，`_pop_span_if_top(span_id)` 仅在栈顶与传入 span 一致时弹出（若调用方手动动过栈，则不做调整）；`with tracer.span(...)` 的 `finally` 里再兜底弹一次，确保无论正常/异常栈都正确弹出；
4. tool_hook 因为是后钩子触发，**调用时上层 decision Span 已经是当前活跃 Span**，所以直接 start_span 出来的 tool Span 会自然成为 decision Span 的子 Span——不需要 tool_hook 自己管父子链。

### 5.3 三种打点方式怎么选

| 方式 | 入口 | 适合场景 | 父子关系 |
|---|---|---|---|
| **with 自动打点** | `with tracer.span(name, kind, input=...) as s:` | 代码块边界清晰的操作（LLM 调用、一段自定义逻辑）| 自动继承当前活跃 span 为父 |
| **手动 start/end** | `s = tracer.start_span(...); ...; tracer.end_span(s, ...)` | 跨函数、跨回调、生成器（如 stream_invoke）需要在多个出口 end 的场景 | 手动 start 时自动设 parent，end 时自动弹栈 |
| **钩子自动打点** | `registry.add_hook(tracer.tool_hook)` / `tracer.attach_to_tool_registry(reg)` | 工具调用，不改工具源码 | 钩子触发时自动成为当前活跃 span 的子 |

**选择建议**：优先用 with 语法；with 覆盖不到的边界（生成器、跨回调）用手动 start/end；工具一律走钩子，不手动给每个工具写 span。

### 5.4 后端自动识别与选择逻辑

TraceLogger 构造时按以下优先级选择存储后端：

1. **最高优先级**：显式传入 `store=TraceStoreBase实例`，直接使用该后端；
2. **内存后端**：`store=":memory:"` 或 `store=None` → InMemoryTraceStore（纯内存，默认）；
3. **自动识别**：传路径字符串时按文件后缀判断：
   - 后缀为 `.db` / `.sqlite` / `.sqlite3` → SQLiteStore
   - 其余后缀（`.jsonl` / `.ndjson` 等）→ JsonlStore
4. **类型校验**：传既非字符串又非 `TraceStoreBase` 的类型时抛 `TypeError`。

### 5.5 工具耗时为什么要回推 start_time

ToolRegistry 的钩子是**后钩子**——工具执行完了才调用 hook，这时已经不知道工具是几点开始的。两种处理方式：

- **方式 A（朴素）**：hook 里 start_span 时取当前时刻当 start_time，finish 时 duration 就是钩子自身的耗时（几乎为 0），完全失真；
- **方式 B（本模块做法）**：
  1. `tool_base.py` 在工具执行前后用 `time.perf_counter()` 实测耗时，写入 `result.metadata["duration_ms"]`（int 毫秒）；
  2. trace 侧在 hook 里读取该值；
  3. 取「当前时刻（≈ 工具执行结束时刻）− duration_ms」回推 start_time；
  4. 构造 span 后覆盖 `s.start_time`，并显式传入 `end_time=hook_end_ts`，使 `end_time - start_time ≡ duration_ms`（严格等于工具自报耗时，零漂移）；
  5. 若工具未上报 duration_ms（第三方工具/自定义工具未写 metadata），退化为近零耗时记录，并在 `span.metadata["duration_source"] = "unavailable"` 标记。

这样保证了：**父子时间区间包含关系正确**（tool span 落在上层 decision span 区间内）、**duration 真实**、**对工具实现零要求**（不报也能跑、只是精度退化为近零）。

### 5.6 打点通道的安全约定（旁路不阻塞）

所有可能抛异常的打点路径都有 try/except 兜底：

- `JsonlStore.write`：整段 try/except，任何序列化/写盘异常都吞；
- `JsonlStore.__init__` / `_iter_file`：读文件时单条 json 解析失败 continue、整体读失败 pass；
- `SQLiteStore.write`：整段 try/except，任何序列化/写库异常都吞；
- `SQLiteStore.read / count / iter_records / flush / close`：异常全吞或返回空结果；
- `TraceLogger.end_span`：store.write 单独 try/except；
- `TraceLogger.tool_hook`：解析 ToolResult 属性的逻辑整段 try/except，失败时退化到 `str(result)`；
- `TraceLogger.attach_to_tool_registry`：add_hook 抛异常（如 registry 为 None/无 add_hook）直接 pass；
- `TraceLogger.flush / close`：底层 store 的 flush/close 异常都吞。

**唯一会抛异常的路径**：构造时 `store` 传入既非字符串也非 `TraceStoreBase` 的类型（抛 `TypeError`）、`SQLiteStore(path="")`（抛 ValueError）。这是刻意为之：错的依赖应该早发现，而不是默默无记录。

---

## 六、核心数据模型

### 6.1 SpanKind（Span 类型枚举）

```python
class SpanKind(str, Enum):
    LLM = "llm"           # 大模型调用（invoke/stream_invoke/embed_query/embed_documents）
    TOOL = "tool"         # 工具调用（通过 ToolRegistry.add_hook 自动打点）
    DECISION = "decision" # Agent 决策/推理步骤（是否调工具、选哪个工具、最终回答）
    AGENT = "agent"       # 一次完整 Agent run
    CUSTOM = "custom"     # 用户自定义打点
```

路线图最小集是 `llm/tool/decision` 三类，额外留 `agent/custom` 作扩展：`agent` 用于包裹整次 run，`custom` 给用户手动打点用。传字符串（如 `"llm"`）给 `start_span(kind=...)` 会自动归一化为枚举，非法字符串退化为 `CUSTOM`。

### 6.2 SpanStatus（执行状态枚举）

```python
class SpanStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
```

最小集只有两种：成功 / 失败。`end_span` 时若 `error` 非空且 status 还是 SUCCESS，会自动置为 ERROR；`with tracer.span(...)` 块内抛异常时自动 ERROR 并把异常信息写入 `s.error` 后重新 raise。

### 6.3 SpanRecord（单条 Span 记录）

```python
class SpanRecord(BaseModel):
    # 追踪链
    trace_id: str                  # 整条链路唯一 id，格式 "trace_" + uuid4.hex
    span_id: str                   # 当前 Span 自身 id（默认 uuid4.hex）
    parent_id: Optional[str]       # 父 Span id；根 Span 为 None
    session_id: Optional[str]      # 会话 id（跨 run 聚合用，默认 sess_ + 12 位 hex）
    # 基本信息
    name: str                      # Span 名称，如 "llm.invoke" / "tool.compute" / "agent.run"
    kind: SpanKind                 # Span 类型
    # 时间
    start_time: float              # 开始时间戳（秒，time.time() 墙钟，用于展示 / 排序）
    end_time: Optional[float]      # 结束时间戳；finish 后自动填充（墙钟）
    duration_ms: Optional[float]   # 耗时（毫秒）；默认由 perf_counter() 单调时钟差值算出（抗墙钟跳变），带下限保护
    # 输入输出
    input: Any                     # 入参（json_safe_serialize 递归处理）
    output: Any                    # 出参（序列化规则同 input）
    # 状态
    status: SpanStatus             # 执行状态，默认 SUCCESS
    error: Optional[str]           # 异常信息（status=error 时填写）
    error_type: Optional[str]      # 异常分类（自动推断异常类名，如 ValueError / ToolError / RuntimeError；显式传入优先）
    # 扩展
    metadata: Dict[str, Any]       # 扩展元数据（模型名、token 用量、工具状态、自定义 KV…）
```

**私有属性（不参与序列化，用户通常无需关心）**：

- `_persisted: bool`（`PrivateAttr`，默认 False）：**幂等落库标记**。`end_span` 成功写入 store 后置 True，同一 Span 再次 `end_span` 直接 return，不会重复写、也不会覆盖首次结果；
- `_create_seq: int`：**全局自增创建序号**，见 6.5。它保证「快速创建的嵌套 Span」在 Windows 低精度时间戳下也能稳定排序，跨进程读回后顺序不变；
- `_mono_start: Optional[float]`（`PrivateAttr`，默认 None）：**单调时钟起点**（`time.perf_counter()` 秒）。仅由 `TraceLogger.start_span` 在「正常创建」时设置，供 `finish()` 计算抗墙钟跳变的耗时；直接构造的 `SpanRecord` / 从存储读回的 Span 保持 None，耗时退化为墙钟差值。

**字段对齐路线图 K1 最小集**：`start_time`（时间戳）/ `kind`（类型）/ `input`（输入）/ `output`（输出）/ `duration_ms`（耗时）/ `session_id`（会话 id），六项全部覆盖。

**finish 方法**：

```python
def finish(self, output=None, status=SpanStatus.SUCCESS, error=None,
           metadata=None, end_time: Optional[float] = None) -> None:
    """标记 Span 结束，自动计算 end_time 与 duration_ms，合并 metadata。

    end_time 为可选精确结束时间戳（秒）；默认 None 时取当前时刻。
    耗时口径：显式传 end_time → 墙钟差值（与旧版一致）；默认路径且 Span 由
    TraceLogger.start_span 正常创建 → perf_counter() 单调时钟差值（抗墙钟跳变）；
    直接构造的 SpanRecord → 退化为墙钟差值。两种口径都做下限保护，不出现负耗时。
    """
```

### 6.4 TraceStoreBase（存储抽象基类）

```python
class TraceStoreBase(ABC):
    @abstractmethod
    def write(self, span: SpanRecord) -> None: ...
    @abstractmethod
    def read(self, session_id=None, trace_id=None, kind=None, limit=None) -> List[SpanRecord]: ...
    def iter_records(self) -> Iterator[SpanRecord]: ...   # 默认基于 read() 遍历，SQLiteStore 覆盖为分批游标
    @abstractmethod
    def close(self) -> None: ...
    def flush(self) -> None: ...                          # 默认 no-op
```

子类契约：`write` 必须保证异常不向上抛；`read` 返回按写入顺序排列的 SpanRecord 列表；`close` 多次调用安全。`count()` 为 SQLiteStore 扩展的便捷方法，基类不强制要求。

### 6.5 辅助函数

| 函数 | 作用 |
|---|---|
| `_new_id()` | 生成 `uuid4().hex`（32 位 hex，不带横线），用作 span_id；trace_id 前缀加 `"trace_"`；session_id 前缀加 `"sess_"` |
| `_now_ts()` | 返回 `time.time()` 秒级墙钟时间戳，**仅用于记录** `start_time` / `end_time`（展示 / 排序 / 落盘）；`duration_ms` 改由 `time.perf_counter()` 单调时钟差值算出（`max(0.0, (perf_counter() - _mono_start) * 1000)`），抗系统墙钟跳变（NTP 校正 / 手动改表），并带下限保护不出现负值 |
| `_next_seq()` | 取一个全局线程安全自增序号（模块级「整数计数器 + `threading.Lock`」），供 `SpanRecord` 记录创建顺序 |
| `_bump_seq(n)` | 把全局计数器推进到至少 `n`，用于读回历史数据后同步计数，避免同一进程续写的新序号与历史冲撞 |

**`_create_seq` 的持久化机制**（`SpanRecord.model_post_init`）：

- **新建场景**：分配一个全局递增序号，**只存进私有属性 `PrivateAttr _create_seq`，绝不写入用户 `metadata`**（`metadata` 始终是用户传入的原样内容）；
- **落盘**：序号作为**独立顶层字段**落盘、与用户 `metadata` 完全隔离——jsonl 每行多一个顶层键 `create_seq`，SQLite 走 `trace_spans` 表的独立 `create_seq` 列（见 `schema.py`）；
- **读回场景**：从顶层 `create_seq` 取值，经 `SpanRecord.model_validate(..., context={"create_seq": n})` 注入私有属性；
- **不做旧数据兼容**：本模块**不读 `metadata` 里的序号**，也没有任何「从旧格式恢复」的兼容分支——按 `AGENTS.md` 约定（代码尚未上线、数据库为空），不为兼容旧数据写兼容代码；缺 `create_seq` 时直接注入 0，排序退化为 `(start_time, span_id)`。

`replay` 排序键为 `(getattr(s, "_create_seq", 0), s.start_time, s.span_id)`：优先按创建序号，缺失时退化为时间 + span_id，彻底摆脱对系统时间精度的依赖。

---

## 七、逐个文件讲解（源码导读）

### 7.1 `__init__.py` —— 模块出口

```python
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase
from .in_memory_trace_store import InMemoryTraceStore
from .jsonl_store import JsonlStore
from .sqlite_store import SQLiteStore
from .trace_logger import TraceLogger, MEMORY_PATH
from .evaluator import (
    TraceEvaluator, TraceMetrics, EvaluationReport,
    DayCostMetrics, CostDailyReport, DEFAULT_PRICE_TABLE,
)

__all__ = ["TraceLogger", "SpanKind", "SpanStatus", "SpanRecord",
           "TraceStoreBase", "JsonlStore", "MEMORY_PATH", "SQLiteStore", "InMemoryTraceStore",
           "TraceEvaluator", "TraceMetrics", "EvaluationReport",
           "DayCostMetrics", "CostDailyReport", "DEFAULT_PRICE_TABLE"]
```

**为什么这么写**：统一导出入口让调用方只需 `from fast_agents.trace import TraceLogger, SQLiteStore`；顶层 `fast_agents/__init__.py` **同步重导出** trace 全部公开符号，所以 `from fast_agents import TraceLogger, SQLiteStore` 也能直接拿到。本模块**不单独维护 `__version__`**（与 tools / memory / mcp 等模块一致，版本号由顶层统一）。注意 `MEMORY_PATH` 是内存模式占位常量，定义在 `trace_logger.py`（**不是** `jsonl_store.py`），由本模块统一重导出。

### 7.2 `models.py` —— 数据模型

**要点**：

- `SpanKind` / `SpanStatus` 继承 `str, Enum`，既可以当枚举用（`SpanKind.LLM`），也可以直接和字符串比较（`s.kind == "llm"`）；pydantic `model_dump(mode="json")` 时自动转成字符串值；
- `SpanRecord` 默认值用 `Field(default_factory=...)`：`span_id` 默认 `_new_id()`（每个新 Span 自动生成），`start_time` 默认 `_now_ts()`（创建时打开始时间），`metadata` 默认空 dict（注意 `default_factory=dict` 而非 `default={}`，避免可变默认值坑）；
- `_persisted` 用 pydantic `PrivateAttr` 声明（默认 False），作为 `end_span` 幂等的落库标记；
- **全局自增创建序号 `_create_seq`**：模块级 `_seq_counter`（整数）+ `_seq_lock`，`_next_seq()` 取号、`_bump_seq()` 在读回历史后把计数器推进到历史最大值之后；`model_post_init` 对「新建」分配序号到私有属性 `PrivateAttr`、对「读回」从顶层 `create_seq` 经 `context` 注入——序号**只存私有属性、以独立顶层键/列落盘，绝不写进（也不从）用户 `metadata` 读写**，既保证跨进程读回后顺序稳定，也彻底解决 Windows 低精度时间戳导致的排序错乱（`replay` 排序优先用它，见 6.5）；
- `finish()` 把「算耗时 / 设 output / 设状态 / 合并 metadata」四件事收口，所有结束 Span 的路径（手动 end_span、with 语法、tool_hook）都走它，保证 duration 计算口径一致；耗时优先用 `perf_counter()` 单调时钟差值（抗墙钟跳变），显式传 `end_time` 时沿用墙钟差值以对齐后钩子精确耗时，并统一做 `max(0.0, ...)` 下限保护，不会出现负耗时；
- 字段特意命名为 `input/output`（单数），对齐路线图最小字段集命名风格、也便于 `s.input` 直接访问；历史上 OpenTelemetry 用 attributes、LangChain 用 inputs/outputs，这里刻意简化为单数。

### 7.3 `store_base.py` —— 存储抽象

**要点**：

- ABC 只定义最小契约：`write / read / close` 三个抽象方法；`iter_records` / `flush` 提供默认实现，子类按需覆盖；
- `read` 的过滤条件做成**命名参数**（session_id/trace_id/kind/limit），默认 None 表示不过滤，让所有后端实现对齐全局过滤语义；kind 按字符串传入（与 SpanKind.value 对齐）而不是枚举，降低后端实现门槛；
- 抽象层不依赖 pydantic 以外的东西，未来要换 SQLite/Postgres/远端后端只需 import 这个基类。

### 7.4 `in_memory_trace_store.py` —— 纯内存后端

**负责什么**：进程内列表存储，进程退出即丢失，是 `TraceLogger()`（不传 store）与 `store=":memory:"` 的默认落点，适合调试、脚本临时打点与单元测试。

**关键实现**：

1. **零构造参数**：`InMemoryTraceStore()` 只维护一个 `self._records: List[SpanRecord]`；
2. **write 异常吞噬**：把 Span 追加进列表，任何异常都吞掉——与 `JsonlStore` / `SQLiteStore` 同一约定，打点通道不阻塞业务；
3. **read 即时过滤**：按 `session_id / trace_id / kind` 顺序过滤、命中 `limit` 即停，按写入顺序返回；
4. **close 为 no-op、清空走 `clear()`**：`close()` 只作占位（多次调用安全）、不再改动数据；需要主动清空记录时调用扩展方法 `clear()`；纯内存后端没有文件句柄需要释放；
5. **为什么单独成文件**：内存后端与文件后端职责完全不同（一个纯 RAM、一个要处理 fsync / 行缓冲 / 坏行跳过），分文件便于各自演进；`MEMORY_PATH = ":memory:"` 占位常量则留在 `trace_logger.py`，因为它是门面层「字符串入口 → 后端」的识别约定，供 `_resolve_store()` 使用。

### 7.5 `jsonl_store.py` —— 默认 JSONL 落盘存储

**关键实现**：

1. **仅文件落盘**：JsonlStore 只做文件模式（内存场景请用 `InMemoryTraceStore`）；`append=True`（默认）时以追加方式打开文件，`append=False` 时清空重写；
2. **不预读历史进内存**：写入直接 append 到文件、**不在内存保留全量镜像**，读取时按需**流式逐行扫描**文件、命中 `limit` 即停，不把全量数据加载到内存（这一点由 `examples/trace/jsonl_demo.py` 实测：写 2000 条 × 约 40KB 后实例仍无内存镜像、进程峰值 RSS 不随数据量增长）；
3. **线程安全**：所有文件句柄读写都走 `self._lock: threading.Lock()`，保证多线程场景不混行；
4. **行缓冲 + fsync**：文件打开用 `buffering=1` 行缓冲、`newline=""` 自己控制换行符；`flush()` 里调用 `os.fsync(fp.fileno())` 保证落盘；
5. **写盘顺序（重要）**：`write()` 调 `span_record_to_dict(span)` 生成「列名 → 值」的行 dict——该函数在 `models.py` 中**先按字段取值、再对三个 `Any` 字段 `input / output / metadata` 用 `json_safe_dumps` 兜底**（= `json.dumps(json_safe_serialize(...))`），**不走** `model_dump(mode="json")`（后者遇到无法识别的对象会整体抛 `PydanticSerializationError`、整条 Span 被静默丢弃）；随后 `json.dumps(row, ensure_ascii=False, default=str)` 把整行写盘。全程只读 span、不修改外部传入的原对象，天然无副作用；
6. **`json_safe_serialize` 逐层兜底**（`fast_agents/utils/json_utils.py`）：`None` 与基本类型（str/int/float/bool）原样返回；pydantic v2 BaseModel 走 `model_dump(mode="json")` 递归（失败退 `str`）；pydantic v1 兜底 `.dict()` 递归；dict / list / tuple 递归；其余再 `json.dumps` 试一遍，不能序列化的一律 `str(obj)`；连 `str()` 都抛异常的极端对象用 `repr()` / 统一占位字符串——保证任何 input/output 都能落库；
7. **为什么 JSONL**：一行一个 JSON，可直接用 `jq` / `pandas.read_json(lines=True)` / `grep` 消费；跨平台换行统一用 `\n`，Windows 上记事本/VSCode 都能正确显示。

**读过滤**：`read()` 流式逐行扫描文件，逐条过滤、try/except 反序列化为 SpanRecord——反序列化失败的脏数据直接跳过，不阻塞其它记录。

### 7.6 `schema.py` —— SQLite 建表 Schema（声明式）

**负责什么**：把 `trace_spans` 表的列、约束与索引用 `database` 模块的声明式 `Schema` / `Column` / `Index` 描述出来，供 `SQLiteStore` 调用 `ensure_schema(migrate=True)` 建表 / 迁移，**不手写建表 SQL**。

**关键实现**：

1. **表结构**：`trace_spans` 表一条 Span 一行，列与 `SpanRecord` 字段一一对应——`id`（INTEGER 自增主键，兼作写入顺序键，等价 jsonl 行序）、`span_id`（TEXT，UNIQUE，防重复写入）、`trace_id` / `parent_id` / `session_id` / `name` / `kind` / `status` / `error`、`start_time` / `end_time` / `duration_ms`（REAL）、`input` / `output` / `metadata`（TEXT，存 `json_safe_dumps` 后的 JSON 串）、`create_seq`（INTEGER，创建序号，排序兜底）；
2. **三索引**：`trace_id` / `session_id` / `kind` 各建一个 `Index`，把 `read()` 的过滤条件交给 `select_where` 做谓词下推；
3. **幂等迁移**：声明式 Schema + `ensure_schema(migrate=True)`，新库一次建齐、老库自动 `ALTER TABLE ADD COLUMN` 补列，升级不丢历史数据；
4. **为什么单独成文件**：表结构是「数据契约」，与 `SQLiteStore` 的读写逻辑解耦——后续加列（如新的排序键 / 采样标记）只改 `schema.py`，读写实现不动。

### 7.7 `sqlite_store.py` —— 结构化 SQLite 落盘存储

**关键实现**：

1. **复用框架已有 `database.SQLiteClient`**：不手写原生 `sqlite3` 连接，直接复用 database 模块统一封装，自动获得 WAL 模式、busy_timeout、参数化查询、声明式幂等建表等能力；
2. **一条 Span 对应表中一行**：落 `trace_spans` 表，`span_id` 加 UNIQUE 约束防止重复写入；自增主键 `id` 保证按写入顺序返回（与 JsonlStore 行序语义完全一致）；
3. **三个索引加速查询**：`trace_id` / `session_id` / `kind` 三个列各建一个索引，`read()` 时把过滤条件编译成 Filter 交给 `select_where` 做谓词下推，大 trace 量下比逐行扫描 jsonl 快很多；
4. **幂等 Schema 迁移**：声明式 Schema + `ensure_schema(migrate=True)`，新库一次建齐表和索引，老库自动 `ALTER TABLE ADD COLUMN` 补列，升级不丢历史 trace 数据；
5. **幂等写入**：写入走 `upsert(conflict_keys=["span_id"])`，同一个 span_id 重复写入时覆盖旧值不增行，重复打点不会产生重复记录；
6. **序列化兜底**：不直接用 `model_dump(mode="json")`（遇到无法识别对象会整体抛异常丢整条 Span），而是调 `span_record_to_dict` 按字段取值、用 `json_safe_dumps` 兜底，与 JsonlStore 序列化口径完全一致，任何 input/output 都能落库；
7. **分批流式遍历**：`iter_records()` 按 500 条一批游标式读取，不一次性把全表加载到内存，适合遍历大规模 trace 数据；
8. **便捷统计**：提供 `count()` 方法快速返回已落盘 Span 总条数，作为后续 K3 评估工具统计的底座；
9. **失败安全**：`write / read / count / iter_records / flush / close` 所有方法都有 try/except 兜底，异常全吞，绝不阻塞主业务；
10. **并发安全**：SQLiteClient 自动开启 WAL 模式和 busy_timeout，多进程并发读写比 JsonlStore 更稳定，适合生产环境持续落盘。

### 7.8 `trace_logger.py` —— 核心记录器

**关键实现**：

1. **构造参数**：`TraceLogger(store=None, *, session_id=None, enabled=True, input_preview_len=None, output_preview_len=None)`；`store` 是唯一存储入口——传 `TraceStoreBase` 实例直接使用；传 `":memory:"`（或 None）用 InMemoryTraceStore；传其它路径字符串按后缀自动选择后端（`.db/.sqlite/.sqlite3` → SQLiteStore，其余 → JsonlStore）；`enabled=False` 时所有方法短路（start_span 返回占位 Span 可无脑 `.finish()`）；
2. **实例级 ContextVar 栈**：`self._trace_stack_var` / `self._span_stack_var` 各持一份 `ContextVar[tuple]` 栈，每个实例 + 每个线程 / asyncio.Task 各持一份，天然隔离；`_stacks()` 读回两条栈，`_push_span()` 压栈，`_pop_span_if_top()` 只在栈顶匹配时弹出；
3. **disabled 占位 Span**：`enabled=False` 时 `start_span` 返回一个 trace_id="disabled" 的占位 SpanRecord，调用方可以无脑 `.finish()`、改 `.output` 等，但 `end_span` 直接 return 不写 store——保证在「想临时关掉 trace 但不改打点代码」时也不报错；
4. **懒启动默认 trace**：`_current_trace_id()` 在 trace 栈空时自动 `start_trace(name="default")`，避免 with 块没包 trace 就 start_span 时 trace_id 为 None；
5. **with 语法的两种粒度**：`trace(name)` yield trace_id，`span(name, kind, input=, metadata=)` yield SpanRecord；with 块内抛异常时自动记 ERROR 并重新 raise；若调用方在块内手动 `.finish()` 过 span（`s.end_time` 非 None），则 `__exit__` 只弹栈 + 写 store，不重复 finish；`finally` 里再兜底弹一次栈，保证父链不错乱；
6. **end_span 幂等**：进入时先判断 `span._persisted`，已落库则直接返回，不重复写、不覆盖首次结果；成功写入后置 `_persisted = True`。这样「with 块内手动 end + 块退出再 end」等重复调用不会产生重复行或 success/error 矛盾记录；
7. **tool_hook**：读 `result.status/error/output/metadata`，从 `metadata["duration_ms"]` 取实测耗时回推 start_time（`start_time = hook_end_ts - duration_ms/1000`，并以同一 `hook_end_ts` 作为 end_time），工具未上报时标记 `duration_source="unavailable"`；tool span 统一命名为 `tool.<name>`，metadata 写入 `tool_name` 和 `tool_status`；
8. **replay**：默认回放规则——有活跃 trace 时回放**当前最内层活跃的 trace**（嵌套场景自动取正在运行的最内层，与 start_span 归属逻辑一致）；无活跃 trace 时自动回放**最近一次结束的 root trace**（parent_id 为 None 的最新 Span 所属 trace）；显式传 `trace_id` / `session_id` 时按参数精确过滤。返回按 `(_create_seq, start_time, span_id)` 排序、可直接 json.dumps 的 dict 列表，所有枚举转字符串值、duration_ms 保留两位小数；
9. **上下文管理器协议**：`__enter__` 返回 self，`__exit__` 先 flush 再 close 并返回 False（不吞异常）——配合 `with TraceLogger(store="...") as tracer:` 自动收尾；
10. **预览截断可配置**：`input_preview_len` / `output_preview_len` 两个可选参数（默认 `None`）控制打点时输入/输出预览的截断长度；门面方法 `truncate_preview(text, default_len, *, is_input=False)` 是唯一出口——未配置时用调用点传入的 `default_len`（与改动前逐字节一致），显式配置后按配置长度截断、`<= 0` 表示不截断，`None`/非字符串原样返回、异常兜底返回原文（R6）。

### 7.9 `evaluator.py` —— 离线评估器（K3）

**负责什么**：离线消费已落盘的 trace 数据，把「一次 Agent run 值不值得、贵不贵、稳不稳」变成可聚合、可对比、可导出的数字。定义三个公开类：

- **`TraceMetrics`**：单条 trace 的指标（成本 / 效率 / 质量 / 工具分布 / 数据质量五组），字段全部可为 None（数据不足时不抛）；
- **`EvaluationReport`**：跨多条 trace 的聚合报告（总数 / 成功率 / 均值 / P95 / 标准差 / 分布），并持有逐条 `traces` 明细；
- **`TraceEvaluator`**：评估器主入口，支持 TraceLogger / Store / SpanRecord 可迭代对象 / 文件路径四种构造方式。

**关键实现**：

1. **多种构造方式**：`TraceEvaluator(source)` 自动识别（有 `.store` + `.read_spans` 当 TraceLogger、有 `.read` 当 Store、否则当可迭代对象）；`TraceEvaluator.from_path(path, backend=None)` 按后缀自动选后端（`.db/.sqlite/.sqlite3` → SQLite，其它 → Jsonl；`:memory:` 或不存在路径返回空报告）；
2. **纯离线、后端无关**：只调用 `store.read()` / `iter_records()`，与在线打点链路完全解耦；JsonlStore 与 SQLiteStore 共用同一套统计逻辑；
3. **成本口径**：从 LLM span 的 `metadata.usage` 累加 `prompt_tokens / completion_tokens / total_tokens`（缺 `total_tokens` 时用 prompt + completion 兜底），并按 `metadata.model` 聚合模型分布；
4. **效率口径**：端到端耗时取「第一条 start ~ 最后一条 end」，另算 LLM / tool / decision 分类耗时、`tool_time_ratio`、`tokens_per_llm_call`，跨 trace 再算均值 / P95（纯标准库线性插值，不引 numpy）/ 标准差；
5. **质量口径**：以 root Span 状态判定整条 trace 成败，统计 error / llm_error / tool_error 计数与 `tool_error_rate`，对错误信息去重；
6. **工具分布**：工具名按 `metadata.tool_name` → `name` 前缀 `tool.` → `name` 三级兜底解析，聚合成 `{tool_name: 调用次数}` 与 `{tool_name: [耗时]}` 明细；
7. **四种输出**：`to_dict()`（程序分析）、`to_json()`（接 dashboard / 前端）、`to_text()`（控制台可读报告）、`to_csv(path=None)`（每条 trace 一行，丢进 Excel / pandas 做二次透视；默认 `utf-8-sig` 落盘防中文乱码，`tool_distribution` 折叠为 `name=count;...`、`error_messages` 折叠为 ` | ` 连接并去掉换行）；
8. **异常吞噬**：任何字段缺失 / 类型异常都不抛，缺字段返回 None 或 0，脏数据整条跳过，评估脚本不会因为一条烂 trace 而崩；
9. **默认关闭**：不构造 `TraceEvaluator` 时零开销，import 不触发任何计算。
10. **Token 成本日报**（`daily_cost_report`）：在 `TraceEvaluator` 上额外提供按自然日聚合 Token 成本的能力——每条 trace 归属到其 root Span `start_time` 所在自然日（默认北京时间 UTC+8），内置 **22 个常见模型**默认价目表（USD/1M tokens，最长前缀优先匹配），未命中走默认兜底；支持 `start_date/end_date/session_id/timezone_offset_hours` 过滤、自定义价目表合并覆盖；输出 `DayCostMetrics`（单日）+ `CostDailyReport`（整报告）两种结构，提供 `to_dict/to_json/to_text/to_csv` 四格式，CSV 每天一行共 21 列，默认 utf-8-sig 落盘 Excel 直读；成本计算公式：`cost = prompt_tokens * input_price / 1e6 + completion_tokens * output_price / 1e6`，全量异常吞噬、脏数据不崩。

**为什么这么写**：评估是「事后分析」而非「在线链路」，所以刻意做成纯离线、只读 store；同时坚守零新增依赖——所有统计量都用 `statistics` + 手写百分位实现，不引 pandas / numpy，保证在离线 / 无网环境也能跑。成本日报是对「每条 trace 一个分数」的进一步汇总，让"今天整体花了多少钱/成功率多少/哪个模型最贵"能一眼看到，而不用每次把 CSV 明细拉进 Excel 透视。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents
python tests\trace\test_trace.py              # K1 打点链路：74/74 全过
python tests\trace\test_trace_jsonl_store.py  # JsonlStore 后端：12/12 全过
python tests\trace\test_trace_sqlite_store.py # SQLiteStore 后端：11/11 全过
python tests\trace\test_trace_eval.py         # K3 评估器：33/33 全过
python tests\trace\test_trace_cost.py         # K3 成本日报：19/19 全过
python examples\trace\basic_usage.py          # 最小手动打点示例
python examples\trace\jsonl_demo.py           # 文件模式不占内存的证明示例
python examples\trace\jsonl_replay.py         # Jsonl 落盘 + 回放示例
python examples\trace\sqlite_usage.py         # SQLite 落盘 + 查询示例
python examples\trace\eval_demo.py            # K3：3 次模拟运行 + 离线评估 + JSON/CSV 导出示例
python examples\trace\cost_daily_demo.py      # K3：模拟 3 天运行 + Token 成本日报 + JSON/CSV 导出示例
```

说明：本仓库测试用**自带的零依赖运行器**（`tests/_runner.py`），**不要用 pytest**；直接用 `python tests\trace\test_trace.py` 执行即可（脚本内置 runner，会打印逐项 PASS/FAIL 与「完成：N 通过」汇总）。K3 评估器测试同理，`python tests\trace\test_trace_eval.py` 独立运行。

### 8.2 最小可运行示例（内存模式 + 手动打点）

```python
from fast_agents.trace import TraceLogger, SpanKind

tracer = TraceLogger()  # 默认 :memory: 内存模式，InMemoryTraceStore

with tracer.trace("demo") as tid:
    with tracer.span("hello", kind=SpanKind.CUSTOM, input={"msg": "hi"}) as s:
        s.output = "world"
    with tracer.span("compute", kind=SpanKind.CUSTOM, input={"x": 1}) as s:
        s.output = 2
        s.metadata["note"] = "simple add"

spans = tracer.read_spans(trace_id=tid)
print(f"trace {tid} 共 {len(spans)} 个 span:")
for s in spans:
    print(f"  [{s.kind.value}] {s.name}  {s.duration_ms:.2f}ms  {s.status.value}")

tracer.close()
```

预期输出（耗时随机器波动）：

```text
trace trace_xxxxxxxx... 共 2 个 span:
  [custom] hello  0.0xms  success
  [custom] compute  0.0xms  success
```

更完整的版本（含 agent/decision/tool 嵌套、replay 打印、enabled=False 关闭）见 `examples/trace/basic_usage.py`。

### 8.3 SQLite 后端快速上手

```python
from fast_agents import TraceLogger, SQLiteStore
from fast_agents.trace import SpanKind

# 方式一：显式传入 SQLiteStore 实例（可自定义参数）
tracer = TraceLogger(store=SQLiteStore("runs/agent_trace.db"))

# 方式二：直接传路径，按后缀自动识别（.db/.sqlite/.sqlite3 → SQLiteStore）
# tracer = TraceLogger(store="runs/agent_trace.db")

with tracer.trace("demo_sqlite") as tid:
    with tracer.span("compute", kind=SpanKind.CUSTOM, input={"x": 2, "y": 3}) as s:
        s.output = 5

print(f"总落盘 Span 数：{tracer.store.count()}")  # SQLiteStore 支持便捷 count
spans = tracer.read_spans(trace_id=tid)
print(f"当前 trace 共 {len(spans)} 个 span")

tracer.close()
# 下次运行重新打开同一路径，历史数据自动保留
```

### 8.4 接到 LLM / Agent 上（自动打点）

**给 LLM 挂 tracer（仅 LLM 打点，不管工具/Agent）**：

```python
from fast_agents import LLM, TraceLogger

tracer = TraceLogger(store="traces/runs.jsonl")  # jsonl 落盘追加
# tracer = TraceLogger(store="traces/runs.db")    # 后缀 .db 自动用 SQLite 后端
llm = LLM(tracer=tracer)
# 之后每次 llm.invoke / stream_invoke / embed_* 都会自动记录一条 llm.* span
```

**给 ReActAgent 挂 tracer（全链路自动打点）**：

```python
from fast_agents import LLM, ReActAgent, ToolRegistry, TraceLogger

tracer = TraceLogger(store="traces/runs.db")  # SQLite 后端，适合生产大量 trace
# tracer= 非 None 时：自动挂 LLM（若 llm.tracer 未被用户设置）+ 自动 attach 工具钩子 + run 内开 agent trace
agent = ReActAgent(name="助手", llm=LLM(), tool_registry=ToolRegistry(), tracer=tracer)
answer = agent.run("你好")
# tracer 里已经有 agent.run / decision.stepN / llm.invoke / tool.* 一整棵树
print(tracer.replay())
tracer.close()
```

**只挂工具打点**：

```python
tracer = TraceLogger()
tracer.attach_to_tool_registry(tool_registry)  # 一行接入；之后 registry.invoke_tool 自动记 tool.* span
```

### 8.5 典型用法四种

- **调试/脚本临时用**：`TraceLogger()`（默认内存模式），跑完 `read_spans()` 直接打印——不落盘、不污染磁盘；
- **开发调试小数据量**：`TraceLogger(store="traces/runs.jsonl")`，Jsonl 追加写入，文本格式直接可读；进程结束 `tracer.close()` 或用 `with TraceLogger(...) as tracer:` 自动 flush + close；
- **生产/大数据量/多进程**：`TraceLogger(store="traces/runs.db")`，SQLite 结构化存储，索引查询快、多进程写稳定，支持 count 统计和流式遍历；
- **跨进程/跨实例读回**：第二次运行时打开同一路径，Jsonl 追加模式或 SQLite 模式都会自动保留历史记录，直接 `read_spans()` / `replay()` 即可。

### 8.6 跑测试

```powershell
python tests\trace\test_trace.py              # K1 打点链路：74 通过
python tests\trace\test_trace_jsonl_store.py  # JsonlStore 后端：12 通过
python tests\trace\test_trace_sqlite_store.py # SQLiteStore 后端：11 通过
python tests\trace\test_trace_eval.py         # K3 评估器：32 通过
python tests\trace\test_trace_cost.py         # K3 成本日报：19 通过
```

预期：各脚本都打印「完成：N 通过，0 跳过，0 失败」；trace 模块合计 **149/149 全绿**。

### 8.7 K3 离线评估最小示例

打点完一批 Agent run 后，**关掉 tracer**，用 `TraceEvaluator` 离线加载落盘文件即可出报告：

```python
from fast_agents.trace import TraceEvaluator

# 业务代码已经跑完并 tracer.close()，现在做离线评估
evaluator = TraceEvaluator.from_path("traces/runs.db")  # .jsonl 也支持，自动识别后端
report = evaluator.evaluate()                            # 评估全部 trace
# 可选过滤：
# report = evaluator.evaluate(session_id="sess_xxx")     # 只评某个会话
# report = evaluator.evaluate(top_n_traces=50)           # 只评最近 50 条 trace（trace 粒度）
# report = evaluator.evaluate(limit=50)                  # limit 同为 trace 粒度：取最近 50 条完整 trace

print(report.to_text())         # 人类可读文本报告（控制台直接打印）
with open("eval_report.json", "w", encoding="utf-8") as f:
    f.write(report.to_json())   # JSON 导出（方便接 dashboard / 前端）
d = report.to_dict()            # Python dict（供程序化分析）
print(f"整体成功率：{d['success_rate']*100:.1f}%，平均耗时 {d['avg_total_duration_ms']:.0f}ms")
report.to_csv("eval_report.csv")  # CSV 明细（每条 trace 一行，Excel / pandas 直接透视）
```

报告里包含：
- **汇总层**：`total_traces` / `successful_traces` / `failed_traces` / `success_rate`；
- **平均指标**：`avg_spans_per_trace` / `avg_llm_calls` / `avg_tool_calls` / `avg_total_duration_ms` / `avg_total_tokens` / `p95_total_duration_ms` / `avg_tool_error_rate` / `avg_llm_error_rate`；
- **分布层**：`tool_distribution` / `tool_error_distribution` / `model_distribution`；
- **明细层**：`traces` 列表，每条 `TraceMetrics` 含成本/效率/质量/工具分布/数据质量五类指标。

更完整的演示见 `examples/trace/eval_demo.py`（模拟 3 次运行 + 打印报告 + 导出 JSON 与 CSV 明细）。

### 8.8 K3 Token 成本日报最小示例

在评估器基础上，再出一份**按天聚合**的成本日报：

```python
from fast_agents.trace import TraceEvaluator

evaluator = TraceEvaluator.from_path("traces/runs.db")

# 默认：按北京时间（UTC+8）自然日聚合，全部 trace
daily = evaluator.daily_cost_report()

# 可选过滤：
# daily = evaluator.daily_cost_report(
#     start_date="2026-11-01",           # 起始日期（含当天）
#     end_date="2026-11-10",             # 结束日期（含当天）
#     session_id="sess_prod",            # 只统计某个会话
#     timezone_offset_hours=8,           # 默认北京时间；设 0 即 UTC
#     price_table={"my-private-model": (0.05, 0.10)},  # 自定义价目表覆盖默认
# )

print(daily.to_text())                  # 控制台可读日报（含合计/每日明细/模型 Top5）
with open("cost_daily.json", "w", encoding="utf-8") as f:
    f.write(daily.to_json())            # JSON 导出（程序化消费）
daily.to_csv("cost_daily.csv")          # CSV 导出（每天一行，Excel 直读）

# 汇总层字段
print(f"总 token：{daily.total_tokens:,}，总成本 ${daily.total_cost_usd:.4f}")
print(f"整体成功率：{daily.overall_success_rate*100:.1f}%")
# 每日迭代
for d in daily.days:
    print(f"  {d.date}  traces={d.trace_count}  tokens={d.total_tokens:,}  "
          f"cost=${d.total_cost_usd:.4f}  success={d.success_rate*100:.0f}%")
    for m, info in d.model_breakdown.items():
        print(f"    - {m}: {info['total_tokens']:,} tokens  ${info['cost']:.4f}")
```

报告包含：
- **整段周期合计**：`total_traces` / `overall_success_rate` / `total_prompt_tokens` / `total_completion_tokens` / `total_tokens` / `total_input_cost_usd` / `total_output_cost_usd` / `total_cost_usd` / `total_llm_calls` / `total_tool_calls` / `total_errors`；
- **每日明细**（`DayCostMetrics`）：日期、trace 数 / 成功率 / LLM & tool 调用数、token 拆分、成本拆分（输入/输出/合计）、平均单 trace 耗时与成本、错误数、`model_breakdown`（每模型调用数/token/成本）、`tool_distribution`；
- **价目表**：`price_table_used` 记录实际使用的价目表（默认 22 个模型 + 自定义覆盖），`price_default` 记录未命中模型的兜底价（默认 $0.50 / $1.50 per 1M）。

**默认价目表覆盖的模型前缀**：`gpt-4o-mini`、`gpt-4o`、`gpt-4-turbo`、`gpt-4`、`gpt-3.5-turbo`、`o1-mini`、`o1`、`o3-mini`、`deepseek-chat`、`deepseek-reasoner`、`claude-3-5-sonnet`、`claude-3-5-haiku`、`claude-3-opus`、`qwen-max`、`qwen-plus`、`qwen-turbo`、`glm-4`、`glm-4-flash`、`doubao-pro`、`doubao-lite`、`ernie-4.0`、`ernie-3.5`——匹配规则是**最长前缀优先**（如 `gpt-4o-mini-2024-07-18` 会匹配到 `gpt-4o-mini` 而非 `gpt-4o`）。想换币种或单价，传 `price_table` 即可覆盖；未命中的模型走 `DEFAULT_PRICE` 兜底，不会漏掉 token 统计。

更完整的演示见 `examples/trace/cost_daily_demo.py`（模拟 3 天 6 次运行 + 文本/JSON/CSV 三格式输出 + 自定义价目表演示）。

---

## 九、API 速查

### 9.1 TraceLogger

**构造**：

```python
TraceLogger(
    store: str | TraceStoreBase | None = None,  # 存储入口：实例 / 路径字符串 / None(:memory:)
    *,
    session_id: str | None = None,              # 会话 id；不传自动生成 sess_xxxxxxxxxxxx
    enabled: bool = True,                       # False 时所有方法短路（占位 Span，不写 store）
    input_preview_len: int | None = None,       # 输入预览截断长度；None 沿用接入点默认（如 500）
    output_preview_len: int | None = None,      # 输出预览截断长度；None 沿用接入点默认（如 300 / 200）
)  # 两个预览长度 <= 0 表示不截断（完整记录）
```

**上下文管理器**：`with TraceLogger(...) as tracer:` → 退出时自动 flush + close。

**Trace 生命周期**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `start_trace(name="root", session_id=None)` | name 默认 "root" | trace_id（`"trace_" + hex`）| 开启一条新 Trace 压栈 |
| `end_trace()` | — | None | 弹栈（实际写入由各 Span.finish 完成）|
| `trace(name="root", session_id=None)` | — | contextmanager，yield trace_id | with 语法自动 start/end |
| `set_session_id(session_id)` | session_id: str | None | 切换 session_id |
| `session_id` (property) | — | str | 当前 session_id |
| `store` (property) | — | TraceStoreBase | 底层后端实例（可访问 count/iter_records 等扩展方法）|

**Span 生命周期**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `start_span(name, kind=CUSTOM, *, input=None, metadata=None, session_id=None)` | kind 可传 SpanKind 或字符串 | SpanRecord | 创建并压栈；parent_id 取当前栈顶；disabled 时返回占位 Span |
| `end_span(span, output=None, status=SUCCESS, error=None, error_type=None, metadata=None, end_time=None)` | error 非空自动 ERROR；`error_type` 自动推断异常类名 | None | **幂等**；finish（含 error_type 推断）→ 弹栈 → 写 store |
| `span(name, kind=CUSTOM, *, input=None, metadata=None)` | — | contextmanager，yield SpanRecord | with 语法；块内异常自动 ERROR 并 raise |

**预览截断（输入/输出长度可配置）**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `truncate_preview(text, default_len, *, is_input=False)` | text: str \| None；default_len: int；is_input=True 取 `input_preview_len`，否则取 `output_preview_len` | str \| None | 按配置截断预览文本：未配置（None）时用 `default_len`（默认路径逐字节不变）；显式配置后按配置长度截断；配置 `<= 0` 不截断；`None`/非字符串原样返回；异常兜底返回原文 |

**工具钩子**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `tool_hook(name, tool_args, result)` | 对齐 `ToolRegistry.add_hook` 签名 | None | 传 hook 给 registry 即可自动打点工具 |
| `attach_to_tool_registry(registry)` | registry: ToolRegistry | None | 一行调用 `registry.add_hook(self.tool_hook)`；enabled=False 时 no-op |

**查询 / 回放**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `read_spans(session_id=None, trace_id=None, kind=None, limit=None)` | kind 传字符串如 `"llm"` | List[SpanRecord] | 从 store 按条件过滤（按写入顺序返回）|
| `replay(trace_id=None, session_id=None)` | 有活跃 trace 取最内层活跃 trace；无则取最近结束的 root trace | List[dict] | 按 `(_create_seq, start_time, span_id)` 排序，dict 可直接 json.dumps |

**生命周期**：

| 方法 | 说明 |
|---|---|
| `flush()` | 刷盘（JsonlStore 会 fsync；SQLiteStore 单语句自动提交，此处 no-op）；异常全吞 |
| `close()` | 关闭底层 store；多次调用安全 |
| `__enter__` / `__exit__` | `with TraceLogger(...) as tracer:` 退出时自动 flush + close |

### 9.2 SpanRecord

| 属性/方法 | 类型/签名 | 说明 |
|---|---|---|
| `trace_id / span_id / parent_id / session_id` | str / str / str? / str? | 追踪链字段 |
| `name` | str | Span 名称 |
| `kind` | SpanKind | 类型（`SpanKind.LLM/TOOL/DECISION/AGENT/CUSTOM` 或字符串值）|
| `start_time / end_time / duration_ms` | float / float? / float? | 时间字段（秒/秒/毫秒）；finish 后自动填 |
| `input / output` | Any | 输入输出（已被 json_safe_dumps 处理成 JSON 安全值）|
| `status / error` | SpanStatus / str? | 执行状态 / 异常信息 |
| `metadata` | dict | 扩展元数据（模型名、token 用量、工具状态、自定义 KV）|
| `finish(output=None, status=SUCCESS, error=None, metadata=None, end_time=None)` | 方法 | 标记结束：填 end_time/duration_ms/output/status/error，合并 metadata；duration_ms 默认用 perf_counter 单调时钟差值（抗墙钟跳变），显式 end_time 时用墙钟差值，负值截断为 0 |
| `_persisted`（PrivateAttr，bool）| 内部标记 | end_span 幂等落库标记，用户无需手动设置 |
| `_create_seq`（PrivateAttr，int）| 内部标记 | 全局自增创建序号，用于跨平台稳定排序，用户无需手动设置 |

**手动修改 span 的正确姿势**：在 with 块内（或 start/end 之间）直接赋值 `s.output = ...` / `s.metadata["k"] = v`，end_span 会把当前字段值写进 store。

### 9.3 SpanKind / SpanStatus

```python
SpanKind.LLM / TOOL / DECISION / AGENT / CUSTOM     # 也可传字符串 "llm"/"tool"/"decision"/"agent"/"custom"
SpanStatus.SUCCESS / ERROR                          # 字符串值 "success"/"error"
```

### 9.4 InMemoryTraceStore / JsonlStore

| 构造参数 | 默认 | 说明 |
|---|---|---|
| `path` | 必填 | JsonlStore 的 jsonl 文件路径（仅文件落盘，内存模式请用 InMemoryTraceStore）|
| `append` | `True` | True 追加（**不预读历史到内存**）；False 清空重写 |
| `encoding` | `"utf-8"` | 文件编码 |

`InMemoryTraceStore()` 无构造参数，进程内列表存储、`close()` 不清空（清空走扩展方法 `clear()`）；是 `TraceLogger()` / `store=":memory:"` 的默认落点。
内存模式占位常量 `MEMORY_PATH = ":memory:"` 定义在 `trace_logger.py`（与 `database.SQLiteClient` 的 `:memory:` 约定对齐），由 `fast_agents.trace` 统一重导出。

**独有扩展方法（非基类强制）**：

| 方法 | 返回 | 说明 |
|---|---|---|
| `clear()` | None | 清空进程内全部记录；`close()` 不清空数据，需要主动重置请调用它 |

### 9.5 SQLiteStore

| 构造参数 | 默认 | 说明 |
|---|---|---|
| `path` | 必填 | SQLite 数据库文件路径；传 `":memory:"` 用内存库（适合测试）|
| `timeout` | `10.0` | SQLite 写锁等待超时秒数（透传给 SQLiteClient，映射 busy_timeout）|

**独有扩展方法（非基类强制）**：

| 方法 | 返回 | 说明 |
|---|---|---|
| `count()` | int | 返回已落盘 Span 总条数（便捷统计）|
| `iter_records()` | Iterator[SpanRecord] | 按 500 条一批分批流式遍历所有记录，不加载全表到内存 |

### 9.6 TraceStoreBase（自定义后端）

实现三个抽象方法即可接入：

```python
class MyStore(TraceStoreBase):
    def write(self, span: SpanRecord) -> None: ...
    def read(self, session_id=None, trace_id=None, kind=None, limit=None) -> List[SpanRecord]: ...
    def close(self) -> None: ...
    # 可选覆盖：iter_records / flush / count
```

传入方式：`TraceLogger(store=MyStore())`。

### 9.7 EvaluationReport（K3 评估报告）

`TraceEvaluator.evaluate(...)` 的返回对象，既有跨 trace 聚合统计，也持有逐条明细。`evaluate` 支持 `session_id` / `trace_id` / `limit` / `top_n_traces` 四个过滤参数，其中 **`limit` 与 `top_n_traces` 同口径**——都按「trace 最近活跃时间倒序取最近 N 条**完整 trace**（连同其全部 span）」截断，**绝不按 span 条数切片**（按 span 切会丢掉 root，把失败 trace 误判成功）；二者默认 None 时评估全部 trace：

```python
report = TraceEvaluator.from_path("traces/runs.db").evaluate()
# 可选过滤：evaluate(session_id=...) / evaluate(trace_id=...) / evaluate(limit=N) / evaluate(top_n_traces=N)

report.total_traces / successful_traces / failed_traces / success_rate
report.avg_spans_per_trace / avg_llm_calls / avg_tool_calls / avg_total_duration_ms / avg_total_tokens
report.p95_total_duration_ms / std_total_duration_ms / avg_tool_error_rate / avg_llm_error_rate
report.tool_distribution / tool_error_distribution / model_distribution
report.traces            # List[TraceMetrics]，逐条明细

report.to_dict()                     # → dict（程序化分析）
report.to_json(indent=2)             # → str（JSON 文本）
report.to_text()                     # → str（控制台可读报告）
report.to_csv(path=None)             # → str（每条 trace 一行 CSV；给 path 则同时落盘）
```

`TraceMetrics` 字段分组：基本（`span_count` / `llm_calls` / `tool_calls` / `decision_calls` / `agent_calls`）、时间（`total_duration_ms` / `llm_total_ms` / `tool_total_ms` / `decision_total_ms` / `tool_time_ratio`）、Token（`prompt_tokens` / `completion_tokens` / `total_tokens` / `tokens_per_llm_call`）、错误（`error_count` / `llm_error_count` / `tool_error_count` / `tool_error_rate` / `success` / `error_messages`）、工具分布（`tool_distribution` / `tool_durations`）、数据质量（`missing_duration_spans` / `spans_without_metadata`）。

**CSV 列**：`to_csv` 输出固定 25 列，列顺序见 `fast_agents.trace.evaluator.CSV_COLUMNS`；`success` 输出 `true` / `false`，`tool_distribution` 形如 `web_search=2;calculator=1`（按次数降序），缺失字段输出空串（不是 `"None"`）。

### 9.8 CostDailyReport / DayCostMetrics（K3 成本日报）

`TraceEvaluator.daily_cost_report(...)` 的返回对象：

```python
daily = TraceEvaluator.from_path("traces/runs.db").daily_cost_report(
    start_date=None, end_date=None,            # 日期范围（YYYY-MM-DD，含两端）
    timezone_offset_hours=8,                   # 默认北京时间；0 = UTC
    session_id=None,                           # 可选会话过滤
    price_table=None,                          # 自定义价目表（与默认表合并覆盖）
)

# 整段周期合计
daily.total_traces / total_llm_calls / total_tool_calls / total_errors
daily.total_prompt_tokens / total_completion_tokens / total_tokens
daily.total_input_cost_usd / total_output_cost_usd / total_cost_usd
daily.overall_success_rate
daily.start_date / end_date / days           # List[DayCostMetrics]，按日期升序
daily.price_table_used / price_default / generated_at

# 单日（DayCostMetrics）
d.date / trace_count / successful_traces / failed_traces / success_rate
d.llm_calls / tool_calls / error_count / llm_error_count / tool_error_count
d.total_duration_ms / avg_duration_ms
d.prompt_tokens / completion_tokens / total_tokens
d.input_cost_usd / output_cost_usd / total_cost_usd / avg_cost_per_trace_usd
d.model_breakdown          # {model: {"calls","prompt_tokens","completion_tokens",
                           #          "total_tokens","input_cost","output_cost","cost"}}
d.tool_distribution        # {tool_name: 调用次数}

# 输出
daily.to_dict()                   # → dict
daily.to_json(indent=2)           # → str（JSON）
daily.to_text(currency="$")       # → str（控制台可读，含每日明细 + 模型成本 Top5）
daily.to_csv(path=None)           # → str（每天一行 CSV；给 path 落盘，默认 utf-8-sig）
```

**成本计算公式**：对每条 LLM Span，`cost = prompt_tokens * input_price / 1_000_000 + completion_tokens * output_price / 1_000_000`（USD）。模型价目按「模型名小写后最长前缀匹配」原则：`gpt-4o-mini-2024-07-18` 命中 `gpt-4o-mini`（长度 11）而非 `gpt-4o`（长度 6）；未命中任何前缀时走 `DEFAULT_PRICE = (0.50, 1.50)`（USD/1M，与 gpt-3.5 同档兜底，保证 token 不会被漏算）。

**日期归属**：每条 trace 归到其**root Span 的 `start_time`** 所在自然日；若该 trace 没有 root，则取其最早的 span `start_time`；若完全没有有效时间戳则跳过（异常吞掉，不影响其它 trace）。时区由 `timezone_offset_hours` 指定，默认 8（北京时间 CST）。

**CSV 列**：`to_csv` 输出固定 21 列，列顺序见 `fast_agents.trace.evaluator.DAILY_CSV_COLUMNS`；`model_breakdown` 形如 `gpt-4o-mini={calls:2,prompt_tokens:3000,cost:0.0009};deepseek-chat={...}`（按 cost 降序），`tool_distribution` 形如 `web_search=3;calculator=1`。

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

1. **为什么不直接用 OpenTelemetry SDK？**——fast_agents 的目标是零重型依赖、能离线跑、能被新手看懂；OTel SDK 带一整套 TracerProvider / SpanProcessor / Exporter / Context 机制，对只想把 LLM/tool 打点落 jsonl 或 sqlite 的场景是过度设计。借鉴它的 Trace/Span 模型足矣。
2. **为什么默认 TraceLogger(enabled=True)，又说"默认关闭"？**——"默认关闭"指的是**框架默认行为**：LLM/Agent 不构造、不传 tracer 时零开销。TraceLogger 自身构造出来默认 enabled=True，因为你既然构造了它一般就是要打点；要临时关闭传 `enabled=False`，此时所有 start_span 返回占位 Span，打点代码无需删改。
3. **为什么 tool_hook 要回推 start_time，而不是在工具调用前插 before_hook？**——现有 `ToolRegistry.add_hook` 只支持**后钩子**（签名 `(name, args, result)`），路线图 K1 明确"不改注册表逻辑"，所以只能从 ToolResult.metadata 读耗时回推；这是硬约束下的最优解。要更精确的前/后双钩子时机，留给后续版本（见第十二章）。
4. **为什么 start_time/end_time 用墙钟 time.time()、而 duration_ms 用 perf_counter() 单调时钟？**——两者职责不同：start_time/end_time 是**给人看的墙钟时间**（能对应到日志时间点，用于展示 / 排序 / 落盘），必须取 `time.time()`；而 duration_ms 是**真实执行时长**，若用 `end_time - start_time` 的墙钟差值，一旦系统墙钟在执行期间被 NTP 校正或手动改表，耗时会失真甚至为负。因此 `finish()` 默认改用 `time.perf_counter()` 单调时钟差值（对墙钟跳变免疫），并做 `max(0.0, ...)` 下限保护。为**零破坏**，只有 `TraceLogger.start_span` 正常创建的 Span 才打单调时钟起点（`_mono_start`）；显式传入 `end_time`（后钩子 / 手工精确起止）或直接构造的 `SpanRecord` 仍走墙钟差值，口径与旧版逐字节一致。
5. **为什么 JsonlStore 不在内存保留全量镜像？**——因为「把整份 jsonl 读进内存」在数据量大时会吃掉大量内存。当前实现：写入直接 append、**不在内存保留镜像**，读取时**按需流式逐行扫描**、命中 limit 即停。这样内存占用与数据量解耦（`examples/trace/jsonl_demo.py` 实测写入 2000 条 × 约 40KB 后实例仍无内存镜像、RSS 不随之增长）；代价是「频繁全量 replay」会重复扫文件，这类大数据量高频查询场景正是 SQLiteStore 索引后端（见 7.7）的用武之地。
6. **为什么 stream_invoke 不统计 token？**——OpenAI 流式接口要拿到 usage 必须请求时加 `stream_options={"include_usage": True}`，这会改变底层响应结构（最后一帧带 usage）；初版不为了统计去改动流式消费逻辑，等确有需求再加（见第十二章）。
7. **为什么 LLM token 要通过 `usage_sink` 这个旁路 dict 回传，而不是改 OpenAIClient 返回值？**——路线图硬约束"不改既有返回值语义"。`usage_sink` 是可选关键字参数，不传就完全不影响原逻辑；传了就在 invoke 结束时把 response.usage 填进去，LLM 层再合并进 span.metadata。
8. **为什么 JSONL 写盘要先兜底三个 Any 字段、再转行 dict？**——`model_dump(mode="json")` 遇到无法识别的自定义对象会整体抛 `PydanticSerializationError`，异常被外层 except 吞掉，整条 Span 就静默丢失了。所以 `write()` 不直接 `model_dump`，而是调 `span_record_to_dict(span)`：对 `input/output/metadata` 逐个用 `json_safe_dumps` 兜底成 JSON 字符串，其余标量字段直接取原值，最后 `json.dumps(row, ensure_ascii=False, default=str)` 落盘——先兜底、后序列化的顺序不能反；该过程只读 span，不修改外部传入的原对象。
9. **为什么 SQLiteStore 也不直接用 model_dump 序列化？**——同第 8 条：`model_dump(mode="json")` 遇到无法识别的对象会整体抛异常、丢整条 Span；SQLiteStore 与 JsonlStore **共用同一个 `span_record_to_dict`**（对 `input/output/metadata` 用 `json_safe_dumps` 兜底），保证任何 input/output 都能落库，两后端序列化口径完全一致。
10. **为什么支持文件后缀自动识别后端？**——让用户无需记忆额外参数，只改文件后缀就能切换后端，零学习成本；同时保留显式参数和 store 注入方式，支持完全自定义。
11. **为什么 count() 不放到基类强制实现？**——count() 是统计便捷方法，不是 Trace 存储的核心契约；基类保持最小接口，简单后端（如内存/远端上报）不需要实现 count 也能正常工作。
12. **为什么要给 SpanRecord 加全局自增创建序号 `_create_seq`？**——Windows 上 `time.time()` 精度只有约 10ms 级，快速创建的嵌套 Span（如外层 agent → 内层 llm）`start_time` 可能完全相同，仅按 `(start_time, span_id)` 排序时顺序不稳定，导致 replay 结果随机器/平台漂移、测试在 Windows 上失败。给每条 Span 分配全局线程安全自增序号，**只存私有属性、以独立顶层字段/列落盘（不碰用户 `metadata`）**，排序优先用它，即可保证任意平台、任意进程重读后顺序都稳定。缺该字段时退化为 `(start_time, span_id)` 排序；本模块不读 `metadata` 里的序号、也不做旧数据兼容。
13. **为什么 end_span 要幂等？**——打点可能在多个出口被触发（如 `with tracer.span(...)` 块内手动 `end_span`、异常路径再 end）。若不幂等，JsonlStore 会落多行、甚至出现同一 span_id 的 `success` 与 `error` 两条矛盾记录；SQLiteStore 因 `upsert` 只留一行，两后端行为不一致。用 `_persisted` 标记做幂等短路，两后端语义统一、统计准确。
14. **为什么 CSV 导出默认 `utf-8-sig`、并把 dict/list 折叠成文本？**——CSV 的典型消费场景是「用 Excel / pandas 打开」，而 Excel 对无 BOM 的 UTF-8 中文会乱码，所以写盘默认带 BOM；同时 CSV 是二维表，塞不下 `tool_distribution` 这种嵌套结构，于是把 dict 折叠成 `name=count;...`、list 折叠成 ` | ` 连接（内部换行折成空格），保证「一分钟能读、一列能筛」。
15. **为什么 `evaluate` 的 `limit` 与 `top_n_traces` 都按 trace 粒度截断？**——**trace 是评估的最小单位**。早期 `limit` 按 **span 条数**切片，会先丢掉序号最小（通常是最先创建的 root）的 span；而失败判定依赖 root 状态，root 一旦缺失，仅凭剩余 span 就判不出失败，于是失败 trace 被误判成功、`success_rate` 与 `error_count` 一并失真，还与本就按 trace 取数的 `top_n_traces` 口径不一致。现统一为「按 trace 最近活跃时间倒序取最近 N 条**完整 trace**（含其全部 span）」，任何进入统计的 trace 都结构完整、root 不丢失。
16. **为什么预览截断长度做成 `TraceLogger` 的可选配置？**——打点会整段记录 LLM 输出 / Agent 输入输出，默认截断（Agent 输入 500 / 输出 300、LLM 输出 200）是为防止 Span 体积失控；但这些数字原本硬编码在各接入层，用户想「完整记录」或「压得更短」都改不了。现收编为 `input_preview_len` / `output_preview_len` 两个可选参数，默认 `None` 时**沿用各接入点原有长度**（行为逐字节不变），显式配置后统一覆盖、`<= 0` 表示不截断；截断逻辑集中在门面方法 `truncate_preview()`，接入层只调这一个出口，异常兜底返回原文（R6）。
16. **为什么 `error_type` 自动推断而不强制调用方传？**——打点接入层（react_agent / llm）统一用 `f"{type(e).__name__}: {e}"` 格式写 error，由 `_infer_error_type` 切取冒号前的异常类名即可；开头必须是 ASCII 字母/下划线，后续只允许 ASCII 字母/数字/下划线/点号（支持 `json.JSONDecodeError`），中文/纯数字/纯符号开头保持 None。工具错误（`tool_hook`）error 是 `ToolResult.error` 直接 str 化，统一标注 `"ToolError"`。老 SQLite 库无 `error_type` 列自动迁移，历史数据 None 不回填（零破坏）。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| 构造了 TraceLogger 但 LLM/Agent 没有任何 span 输出 | 必须把 tracer **显式传给** `LLM(..., tracer=tracer)` 或 `ReActAgent(..., tracer=tracer)`，仅仅构造不会自动全局生效；或者用 `tracer.attach_to_tool_registry(reg)` 手动挂工具 |
| 预期用 SQLite 但实际创建了 jsonl 文件 | 检查文件后缀是否是 `.db/.sqlite/.sqlite3`；后缀不可识别时走 JsonlStore；显式传入 store 实例优先级最高 |
| tool span 的 duration_ms 接近 0 且 metadata 里有 `duration_source=unavailable` | 该工具的 `ToolResult.metadata` 没有 `duration_ms` 字段——自定义工具请用 `ToolBase` 的计时逻辑或手动写入 `result.metadata["duration_ms"]=...`，否则 trace 只能记成近零耗时 |
| 开了 tracer 但异常后 span 没写进去 | 检查是否手动调用了 `start_span` 但忘了 `end_span`（异常路径也要 end）；推荐 with 语法自动成对 |
| with tracer.span(...) 块内异常被吞了 | 没有吞。`span().__exit__` 在记 ERROR 后会**重新 raise**，异常照常传播；只是额外记了一条 ERROR span |
| JsonlStore 第二次启动读不到旧记录 | 检查 `append=True`（默认）并且路径一致；`append=False` 会清空重写；文件模式不预读，但读取会流式扫描文件，路径对就能读回 |
| SQLiteStore 跨 reopen 数据丢失 | 检查路径一致；SQLiteStore 默认持久化，只要路径相同历史数据自动保留；确保进程有目录写权限 |
| read_spans 返回空 | 检查 span 是否真的 end 了（with 块外才会写 store；块内 span.end_time 还是 None）；或者 trace_id/session_id/kind 过滤条件写错 |
| 同一 Span 出现重复记录，或同时有 success 和 error 两条 | 正常路径下 `end_span` 已幂等（`_persisted` 短路），不会重复；若仍出现，请确认没有绕过 `end_span` 直接调 `store.write`。`with tracer.span(...)` 块内手动 finish 后再抛出异常也不会产生矛盾记录 |
| `metadata` 里出现内部键 `_create_seq` | **不会出现**：序号只存私有属性、以独立顶层字段/列落盘，从不写入用户 `metadata`；读回也从顶层 `create_seq` 取，不读 `metadata`。若确实见到该键，说明用的不是当前版本 |
| Windows 上 `replay()` 顺序与预期不一致 | 已由 `_create_seq` 全局自增序号修复；若仍错乱，确认是当前版本（含 `models.py` 的 `_create_seq`） |
| metadata 里的对象变成了字符串 | 对象无法 JSON 序列化时 `json_safe_serialize` 会退化为 `str(obj)`。想保留结构化信息，请传 pydantic 模型 / dict / list / 基本类型 |
| `evaluate(limit=N)` 的成功率 / 错误数与全量评估对不上 | 已修正：`limit` 与 `top_n_traces` 现均为 **trace 粒度**截断（取最近 N 条完整 trace），不再按 span 条数切片丢掉 root；`limit=None` 时不截断，与不传一致。若仍异常，确认用的是当前版本 |
| 内存后端评估前想「重置」数据 | `InMemoryTraceStore.close()` **不再清空**记录——`TraceLogger` 退出 `with` 块时虽会自动 `close()`，但数据仍在，可放心在 `with` 退出后再评估；要主动清空全部记录请调用扩展方法 `clear()` |
| 跨线程丢父链 | 状态栈是**实例级 ContextVar**：同线程/同协程（含 `asyncio.Task`，因其会拷贝上下文）内自动继承父链；但**新建线程不会继承**（`threading.Thread` 不拷贝 Context）。跨线程任务需要在新线程里重新 `start_trace` 或手动指定 parent_id |
| 多进程写同一 jsonl 文件出现混行/损坏 | JsonlStore 的线程锁只保护同进程内多线程；多进程场景请使用 SQLiteStore 后端（自动 WAL + busy_timeout 更安全），或者每个进程写独立文件 |
| finish 传入的 end_time 早于 start_time 时耗时为负？ | 不会。`finish()` 已对 duration_ms 做 `max(0.0, ...)` 下限保护，负值截断为 0。（注：默认路径的耗时来自 `perf_counter()` 单调时钟差值，本身对墙钟跳变免疫；只有显式传 `end_time` 的回推场景才可能遇到「end 早于 start」）|
| 想完整记录超长输入/输出，但预览被截断了？ | 打点默认会截断预览（Agent 输入 500 / 输出 300、LLM 输出 200）。构造 `TraceLogger(input_preview_len=N, output_preview_len=N)` 调整长度，传 `<= 0` 表示不截断（完整记录）；不配置则沿用原有默认，行为逐字节不变 |
| 顶层 `fast_agents` 里 import 不到 SQLiteStore | 顶层 `fast_agents/__init__.py` **已重导出** SQLiteStore，可直接 `from fast_agents import SQLiteStore`；若 import 不到，请确认安装/导入的是本仓库最新版本 |
| `to_csv` 落盘后用 Excel 打开中文乱码 | `to_csv(path)` 写文件默认用 `utf-8-sig`（带 BOM），Excel 双击即可；若自行 `report.to_csv()` 取字符串再手写文件，请显式用 `encoding="utf-8-sig"`。CSV 单元格内换行已被折成空格，无需担心跨行错位 |

### 10.3 平台注意事项（Windows）

> 本模块按 `AGENTS.md` R11 约定**只考虑 Windows 平台**，不做跨平台兼容代码；以下为 Windows 下的实际注意点。

1. **换行符**：JsonlStore 用 `newline=""` 打开文件、写入统一用 `\n`，保证 jsonl 文件格式一致；仓库 `.gitattributes` 配置 `.py/.md/.txt/.bat` 为 CRLF 检出，交付文件需保持 CRLF 行尾（0 裸 LF、0 裸 CR）；
2. **时间精度与排序**：Windows 上 `time.time()` 精度约 10ms，快速创建的嵌套 Span 可能取到相同时间戳，排序不稳定——本模块用全局自增序号 `_create_seq` 兜底，保证顺序稳定；耗时（duration_ms）默认改由 `time.perf_counter()` 单调时钟差值计算，不受墙钟精度与 NTP 校正影响；
3. **路径**：所有路径用 `pathlib` 处理，Windows 下相对/绝对路径、带空格路径均可用，不硬编码 `/`；
4. **文件编码**：jsonl / sqlite 文件统一 `utf-8`；中文/Emoji 都能正确写入（`json.dumps(ensure_ascii=False)`）；
5. **控制台打印**：replay() 返回 dict，打印时若 Windows 控制台中文乱码，执行 `chcp 65001` 或设置 `PYTHONIOENCODING=utf-8`；
6. **文件锁**：JsonlStore 的线程锁只保护**同进程内多线程**；多进程写同一 jsonl 文件可能混行——建议多进程/生产/大数据量场景使用 SQLiteStore 后端；
7. **Python 版本**：仓库 `.python-version` 是 3.10，trace 模块用到的 `str | None` 类型语法要求 Python ≥ 3.10；`from __future__ import annotations` 已在每个文件顶部开启，保证前向引用不报错。

---

## 十一、测试与验证

### 11.1 测试清单

| 测试文件 | 项数 | 覆盖点 | 运行命令 |
|---|---|---|---|
| `tests/trace/test_trace.py` | **74/74** | K1 打点链路（见下表） | `python tests\trace\test_trace.py` |
| `tests/trace/test_trace_jsonl_store.py` | **12/12** | JsonlStore 文件后端、序列化兜底 | `python tests\trace\test_trace_jsonl_store.py` |
| `tests/trace/test_trace_sqlite_store.py` | **11/11** | SQLiteStore 谓词过滤 / 迁移 / 幂等 | `python tests\trace\test_trace_sqlite_store.py` |
| `tests/trace/test_trace_eval.py` | **33/33** | 见下表（K3） | `python tests\trace\test_trace_eval.py` |
| `tests/trace/test_trace_cost.py` | **19/19** | K3 Token 成本日报 | `python tests\trace\test_trace_cost.py` |

> 合计 **149 项全绿**（74 + 12 + 11 + 33 + 19）。

**74 项用例覆盖点**：

1. **Span 数据模型**：默认 span_id 自动生成、start_time 默认当前时刻、finish 自动算 duration_ms（单调时钟差值）、finish 合并 metadata、error 自动置 ERROR、finish 负耗时截断为 0、SpanKind 枚举取值；
2. **存储 InMemoryTraceStore / JsonlStore**（用例位于 `tests/trace/test_trace_jsonl_store.py`）：内存后端写入读回、按 session_id/trace_id/kind 过滤、limit 截断；JsonlStore 文件模式写盘且跨实例读回、append=False 清空重写、落盘文件每行均为合法 JSONL、不累积内存（`test_file_store_does_not_accumulate_in_memory`）、不可序列化 input/output 与极端不可序列化对象也能落库（`test_jsonl_store_handles_unserializable_input_output` / `test_jsonl_store_file_handles_extreme_unserializable` 等）；
3. **存储 SQLiteStore**（用例位于 `tests/trace/test_trace_sqlite_store.py`）：`:memory:` 内存库写入读回、按 session_id/trace_id/kind 谓词过滤、文件模式持久化跨 reopen 读回、幂等 Schema 迁移老库不丢数据、iter_records 分批流式遍历、upsert 同 span_id 幂等不重复、写入异常静默不抛、后缀自动识别 SQLite、显式传入 store 优先级最高；
4. **基础生命周期与开关**：默认构造不报错、enabled=False 不写入、不传 tracer 时 LLM 走原逻辑零分支、with trace 上下文、with span 正常退出记录、with span 异常标记 ERROR；
5. **父子链与时间**：嵌套 span 自动设 parent_id、手动 start/end 成对写入并计算耗时；
6. **幂等与重复调用**：`test_end_span_is_idempotent_no_duplicate_records_jsonl` / `_sqlite`（重复 end 不产生重复记录）、`test_manual_end_span_in_context_normal_exit_does_not_duplicate`（with 块内手动 finish 后再退出不重复）、`test_manual_end_span_in_context_exception_does_not_create_conflicting_records`（手动 finish 后抛异常不产生 success/error 矛盾记录）；
7. **工具钩子 tool_hook**：成功工具生成 `tool.<name>` span 且 metadata 带 tool_name/tool_status、错误工具 status=ERROR 且 error 非空、从 metadata["duration_ms"] 取真实耗时并回推 start_time、缺失耗时标记 duration_source=unavailable、经 ToolRegistry 端到端记录真实耗时、disabled 时 attach 为 no-op；
8. **LLM 打点与 token 元数据**：invoke 带 tracer 记录 span、invoke 不带 tracer 不记录、stream_invoke 记录 span、embed_query / embed_documents 记录 span；invoke 带 usage 时 metadata 含 model/prompt_tokens/completion_tokens/total_tokens/usage、无 usage 时不写这些键、embed_query 记录 token、embed_documents 累加多批次 token、stream_invoke 无 token 元数据；
9. **Agent 接入**：ReActAgent 带 tracer 时记录 agent/decision/llm/tool 多种 span、不带 tracer 时行为与改动前一致；
10. **查询与回放**：`test_replay_uses_inner_most_active_trace_in_nested_context`（嵌套 trace 取最内层活跃）、`test_replay_defaults_to_latest_trace_when_no_active`（无活跃 trace 取最近结束的 root trace）、`test_replay_returns_ordered_printable_list`（顺序稳定、字段完整、可打印）、自定义 store 可注入、set_session_id 切换与会话过滤、TraceLogger 上下文管理器退出时关闭 store；
11. **状态隔离**：多 tracer 实例互不串扰父链（`test_state_is_per_instance_no_cross_pollution`）、状态栈为实例级而非模块级（`test_contextvar_is_instance_level_not_module_level`）；
12. **预览截断可配置**：默认沿用各接入点原有长度（Agent 输入 500 / 输出 300、LLM 输出 200）、显式配置 `input_preview_len`/`output_preview_len` 后按配置截断、`<= 0` 不截断、`None`/非字符串原样返回、Agent 与 LLM 端到端按配置截断；
13. **内存后端生命周期（本次新增）**：`close()` 后记录仍可读出（不清空数据）、`clear()` 清空全部记录且清空后可继续写入、`with TraceLogger(...) as t:` 退出（自动 `close()`）后构造 `TraceEvaluator` 仍能读到数据（原「退出即空报告」坑已消除）。

**K3 评估器 33 项用例覆盖点**（`tests/trace/test_trace_eval.py`）：

1. **构造与空数据**：三种构造方式（TraceLogger / Store / SpanRecord 列表）、空数据返回空报告、`:memory:` 与不存在路径均不抛；
2. **单 trace 指标**：基本计数、端到端耗时、LLM/Tool/Decision 分类耗时累加、token 累加（缺 `total_tokens` 时兜底）、`tool_time_ratio`、`tokens_per_llm_call`；
3. **质量判定**：错误计数、root 错判失败、非 root error 也让 success=False、error_messages 去重、工具名三级解析顺序；
4. **跨 trace 聚合**：avg / std / p95 / success_rate / 工具分布 / 模型分布；
5. **过滤**：`session_id` / `trace_id` / `limit` / `top_n_traces`（`limit` 与 `top_n_traces` 同为 **trace 粒度**截断，取最近 N 条完整 trace，不按 span 切片）；
6. **输出**：`to_dict` / `to_json` roundtrip、`to_text` 含摘要；
7. **CSV 导出（本次新增）**：表头与行数一致、空报告仅表头、落盘 roundtrip（`utf-8-sig`）、`tool_distribution` 折叠为 `web_search=2;calculator=1`、含逗号/换行的错误信息被正确转义且换行折叠、`success` 输出 `true` 且空字段输出空串；
8. **路径后端识别**：`.jsonl` / `.db` roundtrip；
9. **鲁棒性**：脏数据（duration 缺失、metadata 非 dict、usage 非数字）不抛；enabled=False 的 tracer 数据仍可评估；import 不触发计算（零开销）。

### 11.2 关键回归断言

- `test_tool_hook_uses_real_duration_from_metadata` —— 锁住 tool span 必须从 `ToolResult.metadata["duration_ms"]` 取真实耗时并回推 start_time；
- `test_llm_invoke_records_token_metadata` 等 token 用例 —— 锁住 LLM span 必须正确写入/不写入 token 元数据；
- `test_state_is_per_instance_no_cross_pollution` —— 锁住两个 tracer 实例交错打点时，各自根 Span 的 parent_id 必须为 None，不互相串链；
- `test_contextvar_is_instance_level_not_module_level` —— 锁住模块级不再有共享状态、实例上持有自己的 ContextVar 栈；
- `test_end_span_is_idempotent_no_duplicate_records_jsonl` / `_sqlite` —— 锁住重复 end_span 不产生重复记录，且两后端行为一致；
- `test_jsonl_store_file_handles_extreme_unserializable` —— 锁住 JsonlStore 遇到极端不可序列化对象也不丢 Span；
- `test_replay_returns_ordered_printable_list` —— 锁住 replay 输出顺序稳定（依赖 `_create_seq`，Windows 低精度时间戳下也稳定）；
- `test_replay_uses_inner_most_active_trace_in_nested_context` —— 锁住嵌套 trace 时 replay 取最内层活跃 trace；
- `test_to_csv_header_and_row_count` / `test_to_csv_write_file_roundtrip` —— 锁住 K3 CSV 导出表头与列数一致、落盘内容与返回值逐字一致；
- `test_evaluate_limit_keeps_failed_trace_intact` / `test_evaluate_limit_and_top_n_share_trace_granularity` —— 锁住 `evaluate(limit=N)` 必须按 trace 粒度保留整条 trace（含 root）、失败 trace 不被误判成功，且与 `top_n_traces` 口径一致；
- `test_sqlite_store_upsert_is_idempotent` / `test_tracer_auto_detect_sqlite_suffix` —— 锁住 SQLite 幂等写入与后缀自动识别；
- "不传 tracer 时 LLM.invoke 调用链不引入任何 trace 相关分支"——通过 LLM 现有测试全绿间接验证（`test_llm.py` 未改）；
- "ReActAgent 不传 tracer 时行为与改动前一致"——通过 react_agent 现有测试全绿间接验证。

### 11.3 全绿基线

```text
$ python tests\trace\test_trace.py
=== 开始运行测试（共 74 项） ===
  ✅ PASS: test_span_record_defaults
  ...
  ✅ PASS: test_with_trace_context_manager
--- 完成：74 通过，0 跳过，0 失败 ---
```

说明：trace 模块现有 **5 个测试文件、共 149 项全绿**——`test_trace.py` 74/74、`test_trace_jsonl_store.py` 12/12、`test_trace_sqlite_store.py` 11/11、`test_trace_eval.py` 33/33、`test_trace_cost.py` 19/19。本仓库测试用自带的零依赖运行器（`tests/_runner.py`，提供 `raises` / `skip` / `tmp_path` / `MonkeyPatch` 等），**不使用 pytest**。与其他模块的合跑基线以各模块自身 runner 结果为准。

### 11.4 示例验证

```text
$ python examples/trace/basic_usage.py
[1] session_id: sess_xxxxxxxxxxxx
[4] 共记录 4 条 Span

=== demo-pipeline 链路回放（按时间顺序）===
- [   agent] agent.run               20.xxms  success  -> {'final_answer': '3 的平方是 9'}
  - [decision] decision.step1           0.0xms  success  -> {'action': 'call compute', 'args': {'x': 3}}
  - [    tool] tool.compute            20.xxms  success  -> 9

[5] tool 类型 Span 共 1 条
[6] tracer 已关闭
[7] enabled=False 时无任何记录（零开销）

$ python examples/trace/jsonl_demo.py
[1] store._fp is None: False
[2] 实例是否含内存镜像 _mem 属性: False
[3] 写入 2000 条后实例是否含 _mem 镜像: False
[4] 磁盘文件大小:                xx.xx MB
[5] 进程峰值 RSS:                 xx.x MB
[6] 新开 store 后是否含 _mem 镜像: False
[7] 第二次进程峰值 RSS:           xx.x MB
[8] 流式读回记录条数:              2000

$ python examples/trace/jsonl_replay.py
[1] trace 落盘路径: examples/trace/_output/demo.jsonl
[2] 已注册工具: ['add', 'multiply']
[3] 计算结果: (2+3)*4 = 20
[4] 文件已落盘，大小 xxxx 字节
[5] 重新打开后读到 5 条历史 Span
    session=demo-session: 5 条
    kind=agent:    1 条
    kind=decision: 2 条
    kind=tool:     2 条

=== 回放 trace=trace_xxxxxxxx... ===
- [   agent] agent.run               1.0xxms  -> {'final_answer': '(2+3)*4 = 20'}
  - [decision] decision.step1          0.0xms  -> {'action': 'call add', 'args': {'a': 2, 'b': 3}}
  - [    tool] tool.add                0.0xms  -> 5
  - [decision] decision.step2          0.0xms  -> {'action': 'call multiply', 'args': {'a': 5, 'b': 4}}
  - [    tool] tool.multiply           0.0xms  -> 20

$ python examples/trace/sqlite_usage.py
[1] SQLite 落盘路径: examples/trace/_output/demo.db
[2] 已注册工具: ['add', 'multiply']
[3] 计算结果: (2+3)*4 = 20
[4] 总条数 count(): 5
[5] iter_records() 流式遍历： ...
[6] replay trace=trace_xxxxxxxx...
[7] 已关闭，数据库文件大小 xxxxx 字节
[8] 重新打开后读到 5 条历史 Span
[9] TraceLogger(store='...demo_auto.db') 的 store 类型: SQLiteStore

$ python examples/trace/eval_demo.py
Trace 评估报告（K3）
总 Trace 数        : 3
总会话数           : 1
成功 / 失败        : 2 / 1
整体成功率         : 66.7%
（同时导出 eval_report.json + eval_report.csv）

$ python examples/trace/cost_daily_demo.py
Trace Token 成本日报（K3）
统计区间           : 2026-11-09 ~ 2026-11-11  (UTC+8)
总 Trace 数        : 6
整体成功率         : 83.3%
总 Token 消耗      : 22,750 (prompt 18,000 / completion 4,750)
总成本估算         : $0.0660 (输入 $0.0332 / 输出 $0.0328)
（同时导出 cost_daily.json + cost_daily.csv）
```

所有示例均**无需 API Key**，离线可跑（耗时数字随机器波动）。

---

## 十二、下一步演进路线

按优先级排（与《演进路线图.md》§十一 K 模块编号一致；本模块当前版本 v1.2.0，以下为待办能力，不按版本号切分）：

1. **K1 收尾（补丁级）**——✅ **输入输出截断长度可配**、✅ **replay/build_tree 树形聚合**、✅ **error_type 异常分类** 四项**全部交付**（K1 收尾完成）。**duration 改用 perf_counter 单调时钟差值抗墙钟跳变**：`finish()` 默认路径用 `perf_counter()` 差值算 duration_ms，`start_time`/`end_time` 仍为墙钟；显式 `end_time` 或直接构造的 `SpanRecord` 仍走墙钟差值（零破坏）。其中 `error_type` 规则：①显式传入优先；②否则从 `"ExceptionName: msg"` 切取冒号前异常类名（ASCII 标识符，允许点号限定名，如 `json.JSONDecodeError`）；③中文/纯数字/纯符号开头保持 None；工具错误统一 `"ToolError"`；老 SQLite 库自动补列，历史数据保持 None（零破坏）。
2. **K2 异步并发支持**——async 工具/LLM 打点接入、多进程文件锁或文档化「每 worker 独立文件」、Sampler 采样器（按比例/错误率）、SpanEvent 时间点事件（首 token 到达、解析开始）、流式 token 统计（开 `stream_options={"include_usage": True}`）；状态栈已是实例级 ContextVar，跨协程传播已就位，无需返工；依赖 A26 Agent 多工具并发 与 I2 LLM ainvoke 前置落地；
3. **K3 评估工具（已交付基础统计 + CSV 导出 + 成本日报，继续扩展）**——✅ 基础统计聚合（成本/效率/质量三类指标、跨 trace avg/p95/std、工具/模型分布、dict/JSON/文本/CSV 四输出）已交付；✅ CSV 明细导出（`to_csv`，每条 trace 一行 25 列）已交付；✅ **Token 成本日报**（`daily_cost_report`，按自然日聚合、内置 22 个常见模型价目表、最长前缀匹配、dict/JSON/文本/CSV 四输出、可按时区/日期/会话过滤、支持自定义价目表覆盖）已交付。后续增强：工具选择准确率、LLM-as-Judge 打分、双 Agent Win Rate、A/B 对比、按问题集批量回归评估、按问题维度的成本归因；
4. **K4 Web UI 可视化**——瀑布图看板（独立 HTML/SVG 或轻量前端），只读本地 store 数据、不引服务依赖；
5. **K5 外部上报**——可选 OTLP / Langfuse exporter，与「零依赖 / 离线可跑」红线隔离，做成独立可选 exporter、默认不装。

---

## 十三、参考

- [OpenTelemetry Trace API 规范](https://opentelemetry.io/docs/specs/otel/trace/api/) —— 借鉴 Trace/Span/SpanContext/Status 的核心概念与字段划分；
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/) —— 借鉴 on_llm_start / on_tool_end 的事件时机与自动打点思路；
- [OpenAI Agents SDK Tracing](https://openai.github.io/openai-agents-python/tracing/) —— 借鉴 `trace()` / `span()` 简洁的 with 语法 API 设计；
- [fast_agents/database 模块编写说明](file:///docs/database模块编写说明.md) —— 对齐文档骨架；SQLiteClient 是 SQLiteStore 后端的底层依赖；
- [fast_agents/演进路线图.md §十一 模块 K：trace](file:///docs/演进路线图.md) —— K1-K5 模块规划源头。

---

## 附录：发布前自检清单

- [x] 文档名 `trace模块编写说明.md`，落位 `docs/`；
- [x] 开篇写清定位、配套代码路径、版本（v1.2.0）、平台（Windows）与依赖（零新增依赖），且有可跳转目录；
- [x] 更新历史按 `1.x.y` 倒叙记录（同一天合并为一条、每跨一个自然日 `x` 加 1），并列出各版变更；
- [x] 第二章讲清 Trace/Span 概念分层与 6 条设计原则；
- [x] 第三章包含对比总表（5 家）+ 异同表（含"缺少什么"）；
- [x] 第四章有目录结构（单数目录 trace/，9 个源码文件）、分层图、4 条数据流/调用链路图、与 ReActAgent/LLM/ToolRegistry 的接入边界、后端自动识别逻辑；
- [x] 第五章弹性节讲 Trace/Span 生命周期、父子链维护、三种打点方式选择、后端自动识别逻辑、tool_hook 耗时回推机制、打点通道安全约定；
- [x] 数据模型字段命名与代码一致（kind/status/input/output/parent_id/duration_ms，以及 `_persisted` / `_create_seq` 私有属性）；
- [x] 源码导读 9 个文件每个都有「负责什么 / 关键实现 / 为什么这么写」；
- [x] 快速上手命令是 Windows PowerShell、使用仓库自带 runner（`python tests\trace\test_trace.py`，不用 pytest）；最小示例 ≤20 行、无需 API Key；含 SQLite 示例和 4 种典型用法；
- [x] API 速查按类分小节，列出所有对外方法的参数/返回/说明，含 `end_span` 幂等与 `end_time` 参数、SQLiteStore / InMemoryTraceStore API 说明；
- [x] 第十章包含 15 条关键设计决策、19 条常见坑（每条都有可执行解法）、7 条 Windows 平台注意事项；
- [x] 测试章节列出 K1 74 项 + JsonlStore 12 项 + SQLiteStore 11 项 + K3 评估器 33 项 + K3 成本日报 19 项覆盖点 + 关键回归断言 + 全绿基线（149 通过、0 跳过、0 失败）；
- [x] 演进路线与《演进路线图.md》K1-K5 编号一致；待办能力已列入第十二章，版本号按 `1.x.y` 倒叙维护；
- [x] 章节数量、顺序、编号与其余模块文档（tools/memory/skills/mcp/context/database）一致（开篇 → 一～十三 → 附录）；
- [x] 全文无"待补充/TBD"残留，未实现项已列入第十二章；
- [x] 导航目录保留原有结构和条目顺序，锚点保持标准 Markdown 写法，无手动 URL 编码；
- [x] **所有 `.py/.md/.txt/.bat` 为 CRLF 行尾**（交付前逐字节校验 0 裸 LF、0 裸 CR）。
