# FastAgents Trace 模块编写说明

本文档面向 fast_agents **Trace（全链路可观测性）**系统，参考 **OpenTelemetry Trace/Span 模型**、**LangChain Callbacks/Tracer**、**OpenAI Agents SDK Tracing** 的核心概念（trace / span / 父子关系 / 钩子），给出本模块的完整说明、源码解析与自定义教程。同时对齐 fast_agents 已交付的 tools / memory / mcp / skills / context / database 模块文档骨架，做到「翻开任何一份模块说明，都知道在哪一节能找到什么」。

> **版本**：v1.1.0 · 2026-11-05 · P0 SQLite 存储后端交付：Trace/Span 数据模型 + TraceLogger 记录器 + JsonlStore（默认）+ SQLiteStore（新增结构化后端）+ ToolRegistry 钩子打点 + LLM/Agent 可选接入 + LLM token/model 元数据 + tool 真实耗时 + 查询/回放 + 后端自动后缀识别
> **适用版本**：`fast_agents.trace`
> **配套代码**：`fast_agents/trace/`（**6 个**源码文件）+ `test/test_trace.py`（**55 项**用例，全绿）+ `examples/trace/`（**4 个**可运行示例）
> **运行环境**：Python 3.10+（仓库 `.python-version` = 3.10），Windows / macOS / Linux
> **依赖**：**零新增依赖**，仅标准库（`json / os / threading / uuid / time / contextvars / pathlib / contextlib / enum / abc / sqlite3`）+ 全框架已有的 `pydantic>=2.14.1` + 框架自带 `database.SQLiteClient`；不引入 OpenTelemetry / LangChain / loguru 任何外部库
> **本文只写「已经实现的东西」**。「还没做但打算做」的部分放在第十二章，并与 `docs/演进路线图.md` 的 G 模块编号对齐。

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
| **1.1.0** | 2026-11-05 | **P0 SQLite 存储后端落地**：新增 `sqlite_store.py`（第6个源码文件），基于框架自带 `database.SQLiteClient` 实现结构化 SQLite 落盘后端：一条 Span 对应 `trace_spans` 表一行，自增主键保证写入顺序与 JsonlStore 语义一致，`trace_id/session_id/kind` 三个索引支持谓词下推查询，复用 WAL + busy_timeout 保证多进程并发写安全，幂等 Schema 迁移不丢历史数据，同 span_id 重复写幂等覆盖，支持分批流式遍历不加载全表到内存。`TraceLogger` 新增 `backend` 可选参数，支持按文件后缀（.db/.sqlite/.sqlite3）自动识别后端，显式传入 store 实例优先，完全向后兼容（默认仍是 JsonlStore）。**零新增依赖**、**零破坏性改动**：`test/test_trace.py` 新增 12 项 SQLite 测试，总计 **55/55** 全绿；新增 `examples/trace/sqlite_usage.py`、`examples/trace/jsonl_demo.py` 两个离线可跑示例；顶层 `fast_agents/__init__.py` 同步重导出 `SQLiteStore`。 | — |
| **1.0.0** | 2026-11-04 | **初版（路线图 G1 最小闭环）**：`fast_agents/trace/` 作为独立子包交付（`__init__.py` / `models.py` / `store_base.py` / `jsonl_store.py` / `trace_logger.py` 共 5 个文件）。交付：**SpanRecord** 数据模型（trace_id/span_id/parent_id/session_id/name/kind/start_time/end_time/duration_ms/input/output/status/error/metadata）+ **SpanKind** 枚举（llm/tool/decision/agent/custom）+ **SpanStatus** 枚举（success/error）+ **TraceLogger** 门面（手动 span / with 上下文 / trace() 开新根 / 失败安全 / `read_spans` 多条件过滤 / `replay` 缩进回放）+ **TraceStoreBase** 抽象基类 + **JsonlStore**（唯一实现，支持 `":memory:"` 内存模式与 jsonl 逐行 append 落盘，坏行跳过）。**接入**：LLM 门面 `invoke / stream_invoke / embed_query / embed_documents` 仅新增可选 `tracer=None` 打点（stream 走包装生成器），LLM span 写入 `model` + `usage / prompt_tokens / completion_tokens / total_tokens` 元数据；`ReActAgent.__init__` 仅新增可选 `tracer=None`，非 None 时自动 `attach_to_tool_registry` 挂工具钩子、`run` 内自动包 agent/decision 两类 span；工具打点完全走既有 `ToolRegistry.add_hook`，tool span 读 `ToolResult.metadata["duration_ms"]` 记真实耗时并回推 `start_time`，**不修改注册表源码**；根 `fast_agents/__init__.py` 同步重导出 trace 全部公开符号。**零新增依赖**、**零破坏性改动**：`test/test_trace.py` **43/43** 全绿；`examples/trace/basic_usage.py` 与 `examples/trace/jsonl_replay.py` 两个离线可跑示例（不需要 API Key）；`pyproject.toml` / `requirements.txt` 零变更。 | — |

> 后续版本迭代更新时在此表追加，保持「使用者可见的变更」倒序排列；内部重构不在此记录。

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

1. **默认关闭（零开销）**：不构造 `TraceLogger` 或不传 `tracer=` 给 LLM/Agent，代码路径与改动前逐字节一致——这是路线图 G1 的硬约束，也是模块能放进核心路径的前提。`TraceLogger(enabled=True)` 本身默认启用，但你得**显式构造并传进去**才会产生记录。
2. **零新增依赖**：只用标准库 + 全框架已有的 `pydantic v2` + 框架自带 `database.SQLiteClient`。不引 OpenTelemetry SDK、不引 LangChain Callback、不引 loguru——保持 fast_agents「零重型依赖」的招牌。
3. **旁路不阻塞**：打点通道任何异常（序列化失败、文件写失败、钩子报错）都必须被吞掉，**绝不向上抛**阻塞主业务——这是可观测性模块的铁律。
4. **存储抽象可替换**：`TraceStoreBase` 定义 `write / read / close` 最小契约，默认实现是 `JsonlStore`（一行一个 JSON，最容易被 `jq / pandas / 日志采集器`消费），已实现 `SQLiteStore`（结构化索引查询、多进程写更稳），未来切换远端上报只改存储实现，上层打点代码零改动。
5. **最小字段集 + 推荐补充**：严格覆盖路线图要求的「时间戳 / 类型 / 输入 / 输出 / 耗时 / session_id」最小集，并额外补充 OpenTelemetry 风格的 `trace_id / span_id / parent_id / name / status / error / metadata`，方便后续做链路还原与成本统计。
6. **接入只走扩展点，不改核心**：工具打点一律走 `ToolRegistry.add_hook`（不改注册表逻辑）；LLM/Agent 只新增**可选参数** `tracer=None`，不改既有函数签名、不改返回值、不改既有测试断言。

---

## 三、主流框架 Trace 实现对比

### 3.1 对比总表

| 方案 | 定位 | 依赖 | 接入方式 | 存储/导出 | 优点 | 缺点 |
|---|---|---|---|---|---|---|
| **OpenTelemetry Python SDK** | 通用可观测性标准 | opentelemetry-api + opentelemetry-sdk + 各 exporter（~MB 级） | TracerProvider / start_as_current_span / Context 传播 | SpanProcessor + Exporter（OTLP/Console/Zipkin…）| 生态最大、标准最完整、跨语言、跨进程 | 对轻量 Agent 框架过重；概念多（SpanKind/Status/Event/Link/Resource）；exporter 生态复杂；对「只想把 LLM/tool 打点落 jsonl」的场景杀鸡用牛刀 |
| **LangChain Callbacks / Tracer** | LangChain 专属可观测性 | langchain-core（内置）| CallbackManager + handler 基类（on_llm_start/on_tool_end 等）| 内置 Console/File/ LangSmith Tracer | 与 LLM/Tool/Agent 生命周期绑定最深，开箱即用 | 强绑定 LangChain 抽象体系，拿出来单独用几乎不可能；回调方法众多（start/end/error 三件套），接入面大 |
| **OpenAI Agents SDK Tracing** | OpenAI Agents SDK 专属 | agents-sdk（内置）| `trace()` / `span()` 上下文管理器 + 自动 Agent/LLM/Tool 打点 | 默认本地 trace 文件 + 可选上传 OpenAI 平台 | API 设计简洁（with 语法）、自动打点覆盖好、内置 trace viewer | 强绑定 OpenAI Agents SDK；默认会上报云端（隐私/离线场景需关闭）；本地存储格式不公开稳定 |
| **loguru + 装饰器手写** | 通用日志打点 | loguru | 自己写装饰器/上下文管理器 | 日志文件 | 轻、灵活 | 没有 Trace/Span 父子链概念、没有标准化字段、没有存储抽象、回放/分析全靠自己 grep |
| **fast_agents（本模块）** | **框架内置的最小可跑闭环** | **零新依赖**（标准库 + pydantic + 框架自带database）| `tracer=` 可选参数注入 + `ToolRegistry.add_hook` 工具钩子 + `with tracer.span(...)` 手动打点 | **默认 JsonlStore**（一行一条 JSON 落盘，可切 `:memory:`）；可选 **SQLiteStore**（结构化索引查询、多进程写稳）；存储抽象预留远端上报位 | 零依赖、零侵入、旁路不阻塞；API 克制（一个 TraceLogger 搞定）；字段对齐路线图最小集；状态栈实例级隔离且可随协程传播；支持文件后缀自动识别后端 | 初版不做异步并发写入/批处理、不做 Web UI、不做远端上报；不支持 SpanEvent / Link；无采样 |

### 3.2 关键差异：相同 / 增加 / 缺少

| 维度 | 与三者相同 | 相对最简方案（loguru）增加 | 目前缺少（诚实列缺项） |
|---|---|---|---|
| 核心模型 | Trace/Span 父子关系、SpanKind/Status 分类、metadata 扩展 | trace_id/span_id/parent_id 链路字段、自动计时、自动状态（异常→error）| SpanEvent（时间点事件）、SpanLink（跨 trace 关联）、Resource（进程/主机属性） |
| 接入方式 | with 上下文管理器自动成对、手动 start/end 兜底 | LLM/Agent 可选 `tracer=` 参数自动打点；ToolRegistry 钩子一行接入工具；按文件后缀自动识别存储后端 | 装饰器式打点（`@traced`）、async 场景的写入侧接入（同协程父子链已自动） |
| 存储 | 同步写、落盘可查 | `TraceStoreBase` 抽象、JsonlStore 零依赖、SQLiteStore 结构化索引、`:memory:` 测试友好 | 异步批量写、远端 OTLP/Langfuse exporter |
| 上下文传播 | 当前线程/上下文自动父子 | 实例级 ContextVar 栈：多 tracer 实例互不串扰，且随 `asyncio.Task` 拷贝上下文传播 | 跨线程/跨进程传播、Baggage（透传 KV）|
| 性能 | — | 打点通道异常全吞、不阻塞主业务 | 采样（Sampler）、背压、限速 |
| 消费 | 读取回放 | `read_spans` 按条件过滤 + `replay` 按时间排序平铺；SQLiteStore 谓词下推大trace量查询更快 | 瀑布图 UI、Token 成本聚合、工具选择准确率评估（G3/G4）|

一句话收口：**本模块 = OpenTelemetry 的 Trace/Span 核心概念 + OpenAI Agents SDK 简洁的 with API + LangChain 自动打点思路 — 所有重型依赖与复杂抽象，保留 G1 最小可跑闭环 + P0 结构化存储底座**。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/
└── trace/                              # 单数目录（对齐 framework.md §七）
    ├── __init__.py                     # 模块出口，统一导出公开符号
    ├── models.py                       # SpanKind / SpanStatus / SpanRecord 数据模型
    ├── store_base.py                   # TraceStoreBase 抽象基类（存储契约）
    ├── jsonl_store.py                  # 默认 JsonlStore 落盘实现（一行一 JSON）
    ├── sqlite_store.py                 # 新增 SQLiteStore 结构化落盘实现（一条 Span 一行）
    └── trace_logger.py                 # TraceLogger 门面：核心记录器、打点入口、栈维护、查询/回放

test/
└── test_trace.py                       # 55 项测试，全绿；覆盖 JsonlStore 与 SQLiteStore 双后端

examples/trace/
├── basic_usage.py                      # 最小手动打点 + Agent/LLM/工具全链路示例（离线、无需 API Key）
├── jsonl_demo.py                       # JsonlStore 简单落盘示例
├── jsonl_replay.py                     # Jsonl 落盘 + 跨实例回放示例（离线、无需 API Key）
└── sqlite_usage.py                     # SQLiteStore 落盘 + 谓词过滤 + 流式遍历示例（离线、无需 API Key）
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
│  JsonlStore（默认）│  SQLiteStore（可选，v1.1.0新增）            │ ← 零新依赖
│  · :memory: 内存   │  · 复用 database.SQLiteClient              │
│  · 一行一 JSON     │  · trace_spans 表，一条 Span 一行           │
│  · append 追加写   │  · trace_id/session_id/kind 三索引          │
│  · 线程锁 + fsync  │  · WAL + busy_timeout 多进程安全             │
│                   │  · upsert 幂等、Schema 自动迁移              │
│                   │  · iter_records 分批流式遍历                 │
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
   - 显式传入 `store=...` 实例时优先使用，忽略 path/backend 参数；
   - 未显式传 store 时：`backend="jsonl"`/`backend="sqlite"` 显式指定后端；
   - 未显式指定 backend 时，根据 `path` 后缀自动识别：`.jsonl/.ndjson/.json` → JsonlStore；`.db/.sqlite/.sqlite3` → SQLiteStore；
   - 后缀无法识别时默认 JsonlStore，保持向后兼容；
   - `:memory:` 内存模式默认 JsonlStore。

5. **单一数据源与硬约束**：
   - Trace 是**旁路**，不改变 LLM/Agent/Tool 的返回值、不改变异常传播、不改变消息内容；
   - 打点通道所有异常都必须吞掉，不允许把 trace 的 bug 变成主业务的 bug；
   - 不新增任何构造函数必填参数、不改变既有默认值、不改既有测试断言。

---

## 五、核心机制：Trace/Span 模型与打点通道

本章是 trace 模块的弹性节（对应模版 §5），讲清楚"Trace 和 Span 到底怎么协作、父子链怎么维护、打点通道怎么保证不阻塞业务"。

### 5.1 Trace/Span 生命周期对比

| 维度 | Trace（一条链路） | Span（一个操作单元）|
|---|---|---|
| 作用域 | 一次 run 内所有 Span 的容器 | 链路上的单个操作 |
| 显式启动 | `start_trace(name)` / `with tracer.trace(name) as tid` | `start_span(name, kind, input=, metadata=)` / `with tracer.span(...) as s` |
| 自动启动 | 首个 start_span 时若没有活跃 trace，会懒启动一条名为 "default" 的 trace | 无 |
| 结束 | `end_trace()` / with 块退出 | `end_span(s, output=, status=, error=, metadata=)` / with 块退出（异常自动 ERROR）|
| 数据载体 | 不持久化自己，只作为 trace_id 共享给 Span | `SpanRecord` pydantic 模型，结束时写 store |
| 栈 | `_trace_stack_var`（`ContextVar[tuple]`，元素为 trace_id） | `_span_stack_var`（`ContextVar[tuple]`，元素为 span_id） |

### 5.2 父子链怎么维护

父子关系的核心是**当前活跃 span_id**：

1. **状态唯一载体**：每个 `TraceLogger` 实例在 `__init__` 里自持一对**实例级 `contextvars.ContextVar`**——`_trace_stack_var` / `_span_stack_var`，栈用不可变 `tuple` 表示（ContextVar 推荐不可变值，空栈为 `()`）。实例隔离（多 tracer 互不串扰）+ 上下文隔离（`asyncio.Task` 自动拷贝上下文，协程内天然继承）一次拿全；
2. `start_span` 时，parent_id = `_span_stack_var.get()[-1]`（栈顶即当前活跃 span），空栈时 parent_id 为 None；随后 `_push_span` 把新 span_id 压栈；
3. `end_span` 时，`_pop_span_if_top(span_id)` 仅在栈顶与传入 span 一致时弹出（若调用方手动动过栈，则不做调整）；
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

1. **最高优先级**：显式传入 `store=TraceStoreBase实例`，直接使用该后端，忽略 `path` 和 `backend` 参数；
2. **次优先级**：显式传入 `backend="jsonl"` 或 `backend="sqlite"`，按参数选择对应后端；
3. **自动识别**：未显式指定 backend 且 path 不是 `:memory:` 时，根据文件后缀判断：
   - 后缀为 `.jsonl`/`.ndjson`/`.json` → JsonlStore
   - 后缀为 `.db`/`.sqlite`/`.sqlite3` → SQLiteStore
4. **兜底默认**：后缀无法识别或 `path=":memory:"` 时，默认使用 JsonlStore，保持完全向后兼容。

### 5.5 工具耗时为什么要回推 start_time

ToolRegistry 的钩子是**后钩子**——工具执行完了才调用 hook，这时已经不知道工具是几点开始的。两种处理方式：

- **方式 A（朴素）**：hook 里 start_span 时取当前时刻当 start_time，finish 时 duration 就是钩子自身的耗时（几乎为 0），完全失真；
- **方式 B（本模块做法）**：
  1. `tool_base.py` 在工具执行前后用 `time.perf_counter()` 实测耗时，写入 `result.metadata["duration_ms"]`（int 毫秒）；
  2. trace 侧在 hook 里读取该值；
  3. 取「当前时刻（≈ 工具执行结束时刻）− duration_ms」回推 start_time；
  4. 构造 span 后覆盖 `s.start_time`，finish() 算出的 duration_ms ≈ 工具自报耗时；
  5. 若工具未上报 duration_ms（第三方工具/自定义工具未写 metadata），退化为近零耗时记录，并在 `span.metadata["duration_source"] = "unavailable"` 标记。

这样保证了：**父子时间区间包含关系正确**（tool span 落在上层 decision span 区间内）、**duration 真实**、**对工具实现零要求**（不报也能跑、只是精度退化为近零）。

### 5.6 打点通道的安全约定（旁路不阻塞）

所有可能抛异常的打点路径都有 try/except 兜底：

- `JsonlStore.write`：整段 try/except，任何序列化/写盘异常都吞；
- `JsonlStore.__init__`：读已有文件时单条 json 解析失败 continue、整体读失败 pass；
- `SQLiteStore.write`：整段 try/except，任何序列化/写库异常都吞；
- `SQLiteStore.read / count / iter_records / flush / close`：异常全吞或返回空结果；
- `TraceLogger.end_span`：store.write 单独 try/except；
- `TraceLogger.tool_hook`：解析 ToolResult 属性的逻辑整段 try/except，失败时退化到 `str(result)`；
- `TraceLogger.attach_to_tool_registry`：add_hook 抛异常（如 registry 为 None/无 add_hook）直接 pass；
- `TraceLogger.flush / close`：底层 store 的 flush/close 异常都吞。

**唯一会抛异常的路径**：构造时传入非 TraceStoreBase 的 store 实例（类型检查层面就会错）、传入非法 backend 字符串（抛 ValueError）。这是刻意为之：错的依赖应该早发现，而不是默默无记录。

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
    start_time: float              # 开始时间戳（秒，time.time() 墙钟）
    end_time: Optional[float]      # 结束时间戳；finish 后自动填充
    duration_ms: Optional[float]   # 耗时（毫秒）；finish 后自动 (end-start)*1000
    # 输入输出
    input: Any                     # 入参（_safe_serialize 递归处理）
    output: Any                    # 出参（序列化规则同 input）
    # 状态
    status: SpanStatus             # 执行状态，默认 SUCCESS
    error: Optional[str]           # 异常信息（status=error 时填写）
    # 扩展
    metadata: Dict[str, Any]       # 扩展元数据（模型名、token 用量、工具状态、自定义 KV…）
```

**字段对齐路线图 G1 最小集**：`start_time`（时间戳）/ `kind`（类型）/ `input`（输入）/ `output`（输出）/ `duration_ms`（耗时）/ `session_id`（会话 id），六项全部覆盖。

**finish 方法**：

```python
def finish(self, output=None, status=SpanStatus.SUCCESS, error=None, metadata=None) -> None:
    """标记 Span 结束，自动计算 end_time 与 duration_ms，合并 metadata。"""
```

### 6.4 TraceStoreBase（存储抽象基类）

```python
class TraceStoreBase(ABC):
    @abstractmethod
    def write(self, span: SpanRecord) -> None: ...
    @abstractmethod
    def read(self, session_id=None, trace_id=None, kind=None, limit=None) -> List[SpanRecord]: ...
    def iter_records(self) -> Iterator[SpanRecord]: ...   # 默认基于 read() 流式遍历，SQLiteStore 覆盖为分批游标
    @abstractmethod
    def close(self) -> None: ...
    def flush(self) -> None: ...                          # 默认 no-op
```

子类契约：`write` 必须保证异常不向上抛；`read` 返回按写入顺序排列的 SpanRecord 列表；`close` 多次调用安全。`count()` 为 SQLiteStore 扩展的便捷方法，基类不强制要求。

### 6.5 辅助函数

| 函数 | 作用 |
|---|---|
| `_new_id()` | 生成 `uuid4().hex`（32 位 hex，不带横线），用作 span_id；trace_id 前缀加 `"trace_"`；session_id 前缀加 `"sess_"` |
| `_now_ts()` | 返回 `time.time()` 秒级墙钟时间戳，用于 start_time/end_time；`duration_ms` 由 `(end_time - start_time) * 1000` 算出，短时间跨度内墙钟跳变影响可接受 |

---

## 七、逐个文件讲解（源码导读）

### 7.1 `__init__.py` —— 模块出口

```python
from .models import SpanKind, SpanRecord, SpanStatus
from .store_base import TraceStoreBase
from .jsonl_store import JsonlStore, MEMORY_PATH
from .sqlite_store import SQLiteStore
from .trace_logger import TraceLogger

__all__ = ["TraceLogger", "SpanKind", "SpanStatus", "SpanRecord",
           "TraceStoreBase", "JsonlStore", "SQLiteStore", "MEMORY_PATH"]
```

**为什么这么写**：统一导出入口让调用方只需 `from fast_agents.trace import TraceLogger, SQLiteStore`；顶层 `fast_agents/__init__.py` **同步重导出** trace 全部公开符号，所以 `from fast_agents import TraceLogger, SQLiteStore` 也能直接拿到。本模块**不单独维护 `__version__`**（与 tools / memory / mcp 等模块一致，版本号由顶层统一）。

### 7.2 `models.py` —— 数据模型

**要点**：

- `SpanKind` / `SpanStatus` 继承 `str, Enum`，既可以当枚举用（`SpanKind.LLM`），也可以直接和字符串比较（`s.kind == "llm"`）；pydantic `model_dump(mode="json")` 时自动转成字符串值；
- `SpanRecord` 默认值用 `Field(default_factory=...)`：`span_id` 默认 `_new_id()`（每个新 Span 自动生成），`start_time` 默认 `_now_ts()`（创建时打开始时间），`metadata` 默认空 dict（注意 `default_factory=dict` 而非 `default={}`，避免可变默认值坑）；
- `finish()` 把「算耗时 / 设 output / 设状态 / 合并 metadata」四件事收口，所有结束 Span 的路径（手动 end_span、with 语法、tool_hook）都走它，保证 duration 计算口径一致；
- 字段特意命名为 `input/output`（单数），对齐路线图最小字段集命名风格、也便于 `s.input` 直接访问；历史上 OpenTelemetry 用 attributes、LangChain 用 inputs/outputs，这里刻意简化为单数。

### 7.3 `store_base.py` —— 存储抽象

**要点**：

- ABC 只定义最小契约：`write / read / close` 三个抽象方法；`iter_records` / `flush` 提供默认实现，子类按需覆盖；`count()` 为可选扩展方法，基类不强制；
- `read` 的过滤条件做成**命名参数**（session_id/trace_id/kind/limit），默认 None 表示不过滤，让所有后端实现对齐全局过滤语义；kind 按字符串传入（与 SpanKind.value 对齐）而不是枚举，降低后端实现门槛；
- 抽象层不依赖 pydantic 以外的东西，未来要换 SQLite/Postgres/远端后端只需 import 这个基类。

### 7.4 `jsonl_store.py` —— 默认 JSONL 落盘存储

**关键实现**：

1. **:memory: 模式**：`path == ":memory:"`（或 None）时不开文件，所有写入只进 `self._mem: List[dict]`，适合测试/调试；
2. **追加模式**：`append=True`（默认）时打开已有文件，启动时逐行读已存在的 json 装入 `_mem`，保证跨进程/跨 TraceLogger 实例能读回历史（见 `examples/trace/jsonl_replay.py`）；文件模式下写入直接 append 到文件，不会重复扫描全文件；
3. **线程安全**：所有对 `_mem` 和文件句柄的读写都走 `self._lock: threading.Lock()`，保证多线程场景不混行；
4. **行缓冲 + fsync**：文件打开用 `buffering=1` 行缓冲、`newline=""` 自己控制换行符；`flush()` 里调用 `os.fsync(fp.fileno())` 保证落盘；
5. **_safe_serialize 递归兜底**：pydantic v2 BaseModel 走 `model_dump(mode="json")`、v1 兜底走 `.dict()`、dict/list 递归、不能 JSON 序列化的对象一律 `str(obj)`——保证 write 永远不会因为「某个 LLM 返回了一个不可序列化的对象」而抛异常；
6. **为什么 JSONL**：一行一个 JSON，追加写不需要读全文件（只启动时读一次）；可以直接用 `jq` / `pandas.read_json(lines=True)` / `grep` 消费；跨平台换行用 `\n`（统一），Windows 上记事本/VSCode 都能正确显示。

**读过滤**：`read()` 先拿 `_mem` 快照（锁内 copy list），然后锁外逐条过滤、try/except 反序列化为 SpanRecord——反序列化失败的脏数据直接跳过，不阻塞其它记录。

### 7.5 `sqlite_store.py` —— 结构化 SQLite 落盘存储（v1.1.0新增）

**关键实现**：

1. **复用框架已有 `database.SQLiteClient`**：不手写原生 `sqlite3` 连接，直接复用 database 模块统一封装，自动获得 WAL 模式、busy_timeout、参数化查询、声明式幂等建表等能力；
2. **一条 Span 对应表中一行**：落 `trace_spans` 表，`span_id` 加 UNIQUE 约束防止重复写入；自增主键 `id` 保证按写入顺序返回（与 JsonlStore 行序语义完全一致）；
3. **三个索引加速查询**：`trace_id` / `session_id` / `kind` 三个列各建一个索引，`read()` 时把过滤条件编译成 Filter 交给 `select_where` 做谓词下推，大 trace 量下比逐行扫描 jsonl 快很多；
4. **幂等 Schema 迁移**：声明式 Schema + `ensure_schema(migrate=True)`，新库一次建齐表和索引，老库自动 `ALTER TABLE ADD COLUMN` 补列，升级不丢历史 trace 数据；
5. **幂等写入**：写入走 `upsert(conflict_keys=["span_id"])`，同一个 span_id 重复写入时覆盖旧值不增行，重复打点不会产生重复记录；
6. **序列化兜底**：不直接用 `model_dump(mode="json")`（遇到无法识别对象会整体抛异常丢整条 Span），而是按字段取值后用 `_safe_serialize` 兜底，与 JsonlStore 序列化口径完全一致，任何 input/output 都能落库；
7. **分批流式遍历**：`iter_records()` 按 500 条一批游标式读取，不一次性把全表加载到内存，适合遍历大规模 trace 数据；
8. **便捷统计**：提供 `count()` 方法快速返回已落盘 Span 总条数，作为后续 G3 评估工具统计的底座；
9. **失败安全**：`write / read / count / iter_records / flush / close` 所有方法都有 try/except 兜底，异常全吞，绝不阻塞主业务；
10. **并发安全**：SQLiteClient 自动开启 WAL 模式和 busy_timeout，多进程并发读写比 JsonlStore 更稳定，适合生产环境持续落盘。

### 7.6 `trace_logger.py` —— 核心记录器

**关键实现**：

1. **构造参数**：`TraceLogger(store=None, *, path=None, backend=None, session_id=None, enabled=True)`；`store` 自定义后端优先级最高；未传 store 时按 backend 参数或 path 后缀自动选择后端（`.db/.sqlite/.sqlite3` → SQLiteStore，其它默认 JsonlStore）；`enabled=False` 时所有方法短路（start_span 返回占位 Span 可无脑 `.finish()`）；
2. **实例级 ContextVar 栈**：`self._trace_stack_var` / `self._span_stack_var` 各持一份 `ContextVar[tuple]` 栈，每个实例 + 每个线程 / asyncio.Task 各持一份，天然隔离；`_stacks()` 读回两条栈，`_push_span()` 压栈，`_pop_span_if_top()` 只在栈顶匹配时弹出；
3. **disabled 占位 Span**：`enabled=False` 时 `start_span` 返回一个 trace_id="disabled" 的占位 SpanRecord，调用方可以无脑 `.finish()`、改 `.output` 等，但 `end_span` 直接 return 不写 store——保证在「想临时关掉 trace 但不改打点代码」时也不报错；
4. **懒启动默认 trace**：`_current_trace_id()` 在 trace 栈空时自动 `start_trace(name="default")`，避免 with 块没包 trace 就 start_span 时 trace_id 为 None；
5. **with 语法的两种粒度**：`trace(name)` yield trace_id，`span(name, kind, input=, metadata=)` yield SpanRecord；with 块内抛异常时自动记 ERROR 并重新 raise；若调用方在块内手动 `.finish()` 过 span（`s.end_time` 非 None），则 `__exit__` 只弹栈+写 store，不重复 finish；
6. **tool_hook**：读 `result.status/error/output/metadata`，从 `metadata["duration_ms"]` 取实测耗时回推 start_time（`start_time = _now_ts() - duration_ms/1000`），工具未上报时标记 `duration_source="unavailable"`；tool span 统一命名为 `tool.<name>`，metadata 写入 `tool_name` 和 `tool_status`；
7. **replay**：默认回放当前 trace 栈最外层（最早开启的）trace；按 `(start_time, span_id)` 排序后返回可直接 json.dumps 的 dict 列表，所有枚举转字符串值、duration_ms 保留两位小数；
8. **上下文管理器协议**：`__enter__` 返回 self，`__exit__` 先 flush 再 close 并返回 False（不吞异常）——配合 `with TraceLogger(path="...") as tracer:` 自动收尾。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents
python -m pytest test\test_trace.py -v     # 55/55 全过才算装好
python examples\trace\basic_usage.py       # 最小手动打点示例
python examples\trace\jsonl_demo.py        # JsonlStore 简单落盘示例
python examples\trace\jsonl_replay.py      # Jsonl 落盘+回放示例
python examples\trace\sqlite_usage.py      # SQLite 落盘+查询示例
```

说明：本仓库顶层 `test/` 目录与 Python 标准库 `test` 包同名，不要用 `python -m test.test_trace`（会 import 到标准库），用 `pytest test/test_trace.py` 或 `python test\test_trace.py` 直接执行。

### 8.2 最小可运行示例（内存模式 + 手动打点）

```python
from fast_agents.trace import TraceLogger, SpanKind

tracer = TraceLogger()  # 默认 :memory: 内存模式，JsonlStore

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

# 方式二：显式指定 backend 参数
# tracer = TraceLogger(path="runs/agent_trace.db", backend="sqlite")

# 方式三：文件后缀自动识别（.db/.sqlite/.sqlite3 自动用 SQLiteStore）
# tracer = TraceLogger(path="runs/agent_trace.db")

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

tracer = TraceLogger(path="traces/runs.jsonl")  # 默认Jsonl落盘追加
# tracer = TraceLogger(path="traces/runs.db")   # 后缀.db自动用SQLite后端
llm = LLM(tracer=tracer)
# 之后每次 llm.invoke / stream_invoke / embed_* 都会自动记录一条 llm.* span
```

**给 ReActAgent 挂 tracer（全链路自动打点）**：

```python
from fast_agents import LLM, ReActAgent, ToolRegistry, TraceLogger

tracer = TraceLogger(path="traces/runs.db")  # SQLite后端，适合生产大量trace
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
- **开发调试小数据量**：`TraceLogger(path="traces/runs.jsonl")`，Jsonl追加写入，文本格式直接可读；进程结束 `tracer.close()` 或用 `with TraceLogger(...) as tracer:` 自动 flush+close；
- **生产/大数据量/多进程**：`TraceLogger(path="traces/runs.db")`，SQLite结构化存储，索引查询快、多进程写稳定，支持count统计和流式遍历；
- **跨进程/跨实例读回**：第二次运行时打开同一路径，Jsonl append模式或SQLite模式都会自动保留历史记录，直接 `read_spans()` / `replay()` 即可。

### 8.6 跑测试

```powershell
python -m pytest test/test_trace.py -v
```

预期：**55 passed**。

---

## 九、API 速查

### 9.1 TraceLogger

**构造**：

```python
TraceLogger(
    store: TraceStoreBase | None = None,     # 自定义后端实例；显式传入时优先使用，忽略path/backend
    *,
    path: str | None = None,                 # 存储文件路径；默认 ":memory:"；传文件路径时按后缀自动识别后端
    backend: str | None = None,              # 显式指定后端："jsonl" 或 "sqlite"；None时按后缀自动识别，后缀不可识别默认jsonl
    session_id: str | None = None,           # 会话 id；默认自动生成 sess_<12位hex>
    enabled: bool = True,                    # False 时所有打点短路（与不传 tracer 等价）
)
```

**上下文管理器**：`with TraceLogger(...) as tracer:` → 退出时自动 flush+close。

**Trace 生命周期**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `start_trace(name="root", session_id=None)` | name 默认 "root" | trace_id（`"trace_" + hex`）| 开启一条新 Trace 压栈 |
| `end_trace()` | — | None | 弹栈 |
| `trace(name="root", session_id=None)` | — | contextmanager，yield trace_id | with 语法自动 start/end |
| `set_session_id(session_id)` | session_id: str | None | 切换 session_id |
| `session_id` (property) | — | str | 当前 session_id |
| `store` (property) | — | TraceStoreBase | 底层后端实例（可访问count/iter_records等扩展方法） |

**Span 生命周期**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `start_span(name, kind=CUSTOM, *, input=None, metadata=None, session_id=None)` | kind 可传 SpanKind 或字符串 | SpanRecord | 创建并压栈；disabled 时返回占位 Span |
| `end_span(span, output=None, status=SUCCESS, error=None, metadata=None)` | error 非空自动 ERROR | None | finish span → 弹栈 → 写 store |
| `span(name, kind=CUSTOM, *, input=None, metadata=None)` | — | contextmanager，yield SpanRecord | with 语法；异常自动 ERROR 并 raise |

**工具钩子**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `tool_hook(name, tool_args, result)` | 对齐 `ToolRegistry.add_hook` 签名 | None | 传 hook 给 registry 即可自动打点工具 |
| `attach_to_tool_registry(registry)` | registry: ToolRegistry | None | 一行调用 `registry.add_hook(self.tool_hook)` |

**查询 / 回放**：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `read_spans(session_id=None, trace_id=None, kind=None, limit=None)` | kind 传字符串如 `"llm"` | List[SpanRecord] | 从 store 按条件过滤（按写入顺序返回）|
| `replay(trace_id=None, session_id=None)` | 默认回放当前 trace 栈最外层 trace | List[dict] | 按 (start_time, span_id) 排序，dict 可直接 json.dumps |

**生命周期**：

| 方法 | 说明 |
|---|---|
| `flush()` | 刷盘（JsonlStore 会 fsync；SQLiteStore 单语句自动提交，此处 no-op）；异常全吞 |
| `close()` | 关闭底层 store；多次调用安全 |

### 9.2 SpanRecord

| 属性/方法 | 类型/签名 | 说明 |
|---|---|---|
| `trace_id / span_id / parent_id / session_id` | str / str / str? / str? | 追踪链字段 |
| `name` | str | Span 名称 |
| `kind` | SpanKind | 类型（`SpanKind.LLM/TOOL/DECISION/AGENT/CUSTOM` 或字符串值）|
| `start_time / end_time / duration_ms` | float / float? / float? | 时间字段（秒/秒/毫秒）；finish 后自动填 |
| `input / output` | Any | 输入输出（已被 _safe_serialize 处理成 JSON 安全值）|
| `status / error` | SpanStatus / str? | 执行状态 / 异常信息 |
| `metadata` | dict | 扩展元数据（模型名、token 用量、工具状态、自定义 KV）|
| `finish(output=None, status=SUCCESS, error=None, metadata=None)` | 方法 | 标记结束：填 end_time/duration_ms/output/status/error，合并 metadata |

**手动修改 span 的正确姿势**：在 with 块内（或 start/end 之间）直接赋值 `s.output = ...` / `s.metadata["k"] = v`，end_span 会把当前字段值写进 store。

### 9.3 SpanKind / SpanStatus

```python
SpanKind.LLM / TOOL / DECISION / AGENT / CUSTOM     # 也可传字符串 "llm"/"tool"/"decision"/"agent"/"custom"
SpanStatus.SUCCESS / ERROR                          # 字符串值 "success"/"error"
```

### 9.4 JsonlStore / MEMORY_PATH

| 构造参数 | 默认 | 说明 |
|---|---|---|
| `path` | `MEMORY_PATH`（`":memory:"`）| jsonl 文件路径；`:memory:` 或 None 走内存模式 |
| `append` | `True` | True 追加并读回已有文件；False 清空重写 |
| `encoding` | `"utf-8"` | 文件编码 |

`MEMORY_PATH = ":memory:"` 常量，内存模式占位路径（与 `database.SQLiteClient` 的 `:memory:` 约定对齐）。

### 9.5 SQLiteStore（v1.1.0新增）

| 构造参数 | 默认 | 说明 |
|---|---|---|
| `path` | 必填 | SQLite 数据库文件路径；传 `":memory:"` 用内存库（适合测试） |
| `timeout` | `10.0` | SQLite 写锁等待超时秒数（透传给 SQLiteClient，映射 busy_timeout） |

**独有扩展方法（非基类强制）**：

| 方法 | 返回 | 说明 |
|---|---|---|
| `count()` | int | 返回已落盘 Span 总条数（便捷统计） |
| `iter_records()` | Iterator[SpanRecord] | 按500条一批分批流式遍历所有记录，不加载全表到内存 |

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

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

1. **为什么不直接用 OpenTelemetry SDK？**——fast_agents 的目标是零重型依赖、能离线跑、能被新手看懂；OTel SDK 带一整套 TracerProvider / SpanProcessor / Exporter / Context 机制，对只想把 LLM/tool 打点落 jsonl 或 sqlite 的场景是过度设计。借鉴它的 Trace/Span 模型足矣。
2. **为什么默认 TraceLogger(enabled=True)，又说"默认关闭"？**——"默认关闭"指的是**框架默认行为**：LLM/Agent 不构造、不传 tracer 时零开销。TraceLogger 自身构造出来默认 enabled=True，因为你既然构造了它一般就是要打点；要临时关闭传 `enabled=False`，此时所有 start_span 返回占位 Span，打点代码无需删改。
3. **为什么 tool_hook 要回推 start_time，而不是在工具调用前插 before_hook？**——现有 `ToolRegistry.add_hook` 只支持**后钩子**（签名 `(name, args, result)`），路线图 G1 明确"不改注册表逻辑"，所以只能从 ToolResult.metadata 读耗时回推；这是硬约束下的最优解。要更精确的前/后双钩子时机，留给后续版本（见第十二章）。
4. **为什么用墙钟 time.time() 而不是 perf_counter 算 start_time？**——SpanRecord 的 start_time/end_time 是**给人看的墙钟时间**（能对应到日志时间点），duration_ms 由差值算出。短时间操作（LLM/tool 都在秒级以内）NTP 跳变的影响几乎不会出现；长时间 run 跨 NTP 校时窗口可能产生误差，后续版本可把 duration 改用 perf_counter 差值计算（见第十二章）。
5. **为什么 JsonlStore 文件模式要预读全文件进 `_mem`？**——`read()` 需要按条件过滤；不读进内存就得每次 read 时扫全文件，频繁 replay 会重复 IO。初版面向中小数据量（单次开发调试 / 几万条 span）全量缓存可接受；大数据量是 SQLiteStore 的适用场景。
6. **为什么 stream_invoke 不统计 token？**——OpenAI 流式接口要拿到 usage 必须请求时加 `stream_options={"include_usage": True}`，这会改变底层响应结构（最后一帧带 usage）；初版不为了统计去改动流式消费逻辑，等确有需求再加（见第十二章）。
7. **为什么 LLM token 要通过 `usage_sink` 这个旁路 dict 回传，而不是改 OpenAIClient 返回值？**——路线图硬约束"不改既有返回值语义"。`usage_sink` 是可选关键字参数，不传就完全不影响原逻辑；传了就在 invoke 结束时把 response.usage 填进去，LLM 层再合并进 span.metadata。
8. **为什么 SQLiteStore 不直接用 model_dump 序列化？**——`model_dump(mode="json")` 遇到无法识别的自定义对象会整体抛 `PydanticSerializationError`，导致整条 Span 被静默丢弃；按字段逐个取值 + `_safe_serialize` 兜底能保证任何 input/output 都能落库，与 JsonlStore 序列化口径一致。
9. **为什么支持文件后缀自动识别后端？**——让用户无需记忆额外参数，只改文件后缀就能切换后端，零学习成本；同时保留显式参数和store注入方式，支持完全自定义。
10. **为什么 count() 不放到基类强制实现？**——count() 是统计便捷方法，不是 Trace 存储的核心契约；基类保持最小接口，简单后端（如内存/远端上报）不需要实现count也能正常工作。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| 构造了 TraceLogger 但 LLM/Agent 没有任何 span 输出 | 必须把 tracer **显式传给** `LLM(..., tracer=tracer)` 或 `ReActAgent(..., tracer=tracer)`，仅仅构造不会自动全局生效；或者用 `tracer.attach_to_tool_registry(reg)` 手动挂工具 |
| 预期用SQLite但实际创建了jsonl文件 | 检查文件后缀是否是 `.db/.sqlite/.sqlite3`，或者显式传入 `backend="sqlite"`；后缀不可识别时默认JsonlStore；显式传入store实例优先级最高 |
| tool span 的 duration_ms 接近 0 且 metadata 里有 `duration_source=unavailable` | 该工具的 `ToolResult.metadata` 没有 `duration_ms` 字段——自定义工具请用 `ToolBase` 的计时逻辑或手动写入 `result.metadata["duration_ms"]=...`，否则 trace 只能记成近零耗时 |
| 开了 tracer 但异常后 span 没写进去 | 检查是否手动调用了 `start_span` 但忘了 `end_span`（异常路径也要 end）；推荐 with 语法自动成对 |
| with tracer.span(...) 块内异常被吞了 | 没有吞。`span().__exit__` 在记 ERROR 后会**重新 raise**，异常照常传播；只是额外记了一条 ERROR span |
| JsonlStore 第二次启动读不到旧记录 | 检查 `append=True`（默认）并且路径一致；`append=False` 会清空重写 |
| SQLiteStore 跨 reopen 数据丢失 | 检查路径一致；SQLiteStore 默认持久化，只要路径相同历史数据自动保留；确保进程有目录写权限 |
| read_spans 返回空 | 检查 span 是否真的 end 了（with 块外才会写 store；块内 span.end_time 还是 None）；或者 trace_id/session_id/kind 过滤条件写错 |
| metadata 里的对象变成了字符串 | 对象无法 JSON 序列化时 `_safe_serialize` 会退化为 `str(obj)`。想保留结构化信息，请传 pydantic 模型 / dict / list / 基本类型 |
| 跨线程丢父链 | 状态栈是**实例级 ContextVar**：同线程/同协程（含 `asyncio.Task`，因其会拷贝上下文）内自动继承父链；但**新建线程不会继承**（`threading.Thread` 不拷贝 Context）。跨线程任务需要在新线程里重新 `start_trace` 或手动指定 parent_id |
| 多进程写同一jsonl文件出现混行/损坏 | JsonlStore 的线程锁只保护同进程内多线程；多进程场景请使用 SQLiteStore 后端（自动WAL+busy_timeout更安全），或者每个进程写独立文件 |
| 顶层 `fast_agents` 里 import 不到 SQLiteStore | 顶层 `fast_agents/__init__.py` **已重导出** SQLiteStore，可直接 `from fast_agents import SQLiteStore`；若 import 不到，请确认安装/导入的是本仓库最新版本 |

### 10.3 跨平台注意事项

1. **换行符**：JsonlStore 用 `newline=""` 打开文件、写入统一用 `\n`，保证跨平台 jsonl 文件格式一致；所有 `.py/.md/.txt/.bat` 源文件入库 LF、Windows 检出 CRLF（仓库 `.gitattributes` 已配置）；
2. **路径**：`Path(path).parent.mkdir(parents=True, exist_ok=True)`，Windows 下相对/绝对路径、带空格路径均可用；路径分隔符用 pathlib 处理，不硬编码 `/`；
3. **文件编码**：jsonl/sqlite 文件统一 `utf-8`；中文/Emoji 都能正确写入（`json.dumps(ensure_ascii=False)`）；
4. **控制台打印**：replay() 返回 dict，打印时若 Windows 控制台中文乱码，执行 `chcp 65001` 或设置 `PYTHONIOENCODING=utf-8`；
5. **文件锁**：JsonlStore 的线程锁只保护**同进程内多线程**；多进程写同一 jsonl 文件可能混行——建议多进程/生产/大数据量场景使用 SQLiteStore 后端；
6. **Python 版本**：仓库 `.python-version` 是 3.10，trace 模块用到的 `str | None` 类型语法要求 Python ≥ 3.10；`from __future__ import annotations` 已在每个文件顶部开启，保证前向引用不报错。

---

## 十一、测试与验证

### 11.1 测试清单

| 测试文件 | 项数 | 覆盖点 | 运行命令 |
|---|---|---|---|
| `test/test_trace.py` | **55/55** | 见下表 | `python -m pytest test/test_trace.py -v` |

**55 项用例覆盖点**：

1. **Span 数据模型**：默认 span_id 自动生成、start_time 默认当前时刻、finish 自动算 duration_ms、finish 合并 metadata、error 自动置 ERROR、SpanKind 枚举取值；
2. **存储 JsonlStore**：`:memory:` 写入读回、按 session_id/trace_id/kind 过滤、limit 截断、`:memory:` 与 None 等价、文件模式写盘且 append 模式跨实例读回、append=False 清空重写、落盘文件每行均为合法 JSONL；
3. **存储 SQLiteStore（v1.1.0新增）**：内存模式写入读回、按 session_id/trace_id/kind 谓词过滤、文件模式持久化跨reopen读回、幂等Schema迁移老库不丢数据、iter_records分批流式遍历、upsert同span_id幂等不重复、写入异常静默不抛、backend="sqlite"参数正确创建、后缀自动识别SQLite、显式传入store优先级最高；
4. **基础生命周期与开关**：默认构造不报错、enabled=False 不写入、不传 tracer 时 LLM 走原逻辑零分支、with trace 上下文、with span 正常退出记录、with span 异常标记 ERROR；
5. **父子链与时间**：嵌套 span 自动设 parent_id、手动 start/end 成对写入并计算耗时；
6. **工具钩子 tool_hook**：成功工具生成 `tool.<name>` span 且 metadata 带 tool_name/tool_status、错误工具 status=ERROR 且 error 非空、从 metadata["duration_ms"] 取真实耗时并回推 start_time、缺失耗时标记 duration_source=unavailable、经 ToolRegistry 端到端记录真实耗时、disabled 时 attach 为 no-op；
7. **LLM 打点与 token 元数据**：invoke 带 tracer 记录 span、invoke 不带 tracer 不记录、stream_invoke 记录 span、embed_query / embed_documents 记录 span；invoke 带 usage 时 metadata 含 model/prompt_tokens/completion_tokens/total_tokens/usage、无 usage 时不写这些键、embed_query 记录 token、embed_documents 累加多批次 token、stream_invoke 无 token 元数据；
8. **Agent 接入**：ReActAgent 带 tracer 时记录 agent/decision/llm/tool 多种 span、不带 tracer 时行为与改动前一致；
9. **查询与回放**：replay 返回按时间排序、字段完整、可打印的列表、自定义 store 可注入、set_session_id 切换与会话过滤、TraceLogger 上下文管理器退出时关闭 store；
10. **状态隔离**：多 tracer 实例互不串扰父链（`test_state_is_per_instance_no_cross_pollution`）、状态栈为实例级而非模块级（`test_contextvar_is_instance_level_not_module_level`）。

### 11.2 关键回归断言

- `test_tool_hook_uses_real_duration_from_metadata` —— 锁住 tool span 必须从 `ToolResult.metadata["duration_ms"]` 取真实耗时并回推 start_time；
- `test_llm_invoke_records_token_metadata` 等 token 用例 —— 锁住 LLM span 必须正确写入/不写入 token 元数据；
- `test_state_is_per_instance_no_cross_pollution` —— 锁住两个 tracer 实例交错打点时，各自根 Span 的 parent_id 必须为 None，不互相串链；
- `test_contextvar_is_instance_level_not_module_level` —— 锁住模块级不再有共享状态、实例上持有自己的 ContextVar 栈；
- `test_sqlite_store_upsert_is_idempotent` —— 锁住 SQLiteStore 重复写入同span_id不产生重复记录；
- `test_tracer_auto_detect_sqlite_suffix` —— 锁住文件后缀自动识别逻辑正确；
- "不传 tracer 时 LLM.invoke 调用链不引入任何 trace 相关分支"——通过 LLM 现有测试全绿间接验证（`test_llm.py` 未改）；
- "ReActAgent 不传 tracer 时行为与改动前一致"——通过 react_agent 现有测试全绿间接验证。

### 11.3 全绿基线

```text
$ python -m pytest test/test_trace.py -q
.......................................................                              [100%]
55 passed in 3.35s
```

回归基线（与其它模块合跑）：

```text
$ python -m pytest test/test_trace.py test/test_database.py test/test_tools.py -q
299 passed, 17 skipped in 12.1s
```

说明：`test/test_trace.py` 55 项、`test/test_tools.py` 56 项、`test/test_database.py` 188 passed；另有 17 项依赖可选依赖 `chromadb` 的用例在未安装该可选依赖时被 `pytest.importorskip` 跳过（**skip 不是失败**，装上 `pip install fast-agents[chroma]` 后即参与运行）。

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

$ python examples/trace/jsonl_replay.py
[1] trace 落盘路径: examples/trace/_output/demo.jsonl
[2] 已注册工具: ['add', 'multiply']
[3] 计算结果: (2+3)*4 = 20
[4] 文件已落盘，大小 2335 字节
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
[1] SQLite 数据库路径: examples/trace/_output/demo.db
[2] 已注册工具: ['add', 'multiply']
[3] 计算结果: (2+3)*4 = 20
[4] 总落盘 Span 数: 5
[5] 重新打开数据库读回 5 条历史 Span
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
```

所有示例均**无需 API Key**，离线可跑（耗时数字随机器波动）。

---

## 十二、下一步演进路线

按优先级排（与《演进路线图.md》§十 G 模块编号一致）：

1. **v1.1.1 · G1 收尾 patch**——输入输出截断长度可配置（现在接入层硬编码，如 agent 输出预览 300 字符）、`replay` 补 `build_tree()` 树形聚合辅助、统一 `error_type` 分类字段、duration 改用 perf_counter 差值抗墙钟跳变——纯 G1 收尾，零依赖、零破坏；
2. **v1.2.0 · G1.1 数据可消费**：新增 `summarize(trace_id/session_id)` 轻量统计 API（基于 SQLiteStore，按 trace 汇总 token 消耗 / 平均耗时 / step 数 / 工具调用次数 / 错误率 / 工具选择分布），把已写进 metadata 的 token 数据真正消费起来，作为 G3 评估工具的天然底座；
3. **v1.2.0 · G2 异步并发支持**：async 工具/LLM 打点接入、多进程文件锁或文档化「每 worker 独立文件」、Sampler 采样器（按比例/错误率）、SpanEvent 时间点事件（首 token 到达、解析开始）、流式 token 统计（开 `stream_options={"include_usage": True}`）；状态栈已是实例级 ContextVar，跨协程传播已就位，无需返工；
4. **v1.3.0 · G3 评估工具**：工具选择准确率、双 Agent Win Rate、结果对比、Token 成本日报——依赖 v1.2.0 的统计底座；
5. **v1.4.0 · G4 Web UI / 外部上报**：瀑布图看板（独立 HTML/SVG 或轻量前端）、可选 OTLP/Langfuse exporter——与"零依赖/离线可跑"红线隔离，做成独立可选 exporter，默认不装。

---

## 十三、参考

- [OpenTelemetry Trace API 规范](https://opentelemetry.io/docs/specs/otel/trace/api/) —— 借鉴 Trace/Span/SpanContext/Status 的核心概念与字段划分；
- [LangChain Callbacks](https://python.langchain.com/docs/concepts/callbacks/) —— 借鉴 on_llm_start / on_tool_end 的事件时机与自动打点思路；
- [OpenAI Agents SDK Tracing](https://openai.github.io/openai-agents-python/tracing/) —— 借鉴 `trace()` / `span()` 简洁的 with 语法 API 设计；
- [fast_agents/database 模块编写说明](file:///docs/database模块编写说明.md) —— 对齐文档骨架；SQLiteClient 是 SQLiteStore 后端的底层依赖；
- [fast_agents/演进路线图.md §十 模块 G：infra](file:///docs/演进路线图.md) —— G1-G6 模块规划源头。

---

## 附录：发布前自检清单

- [x] 文档名 `trace模块编写说明.md`，落位 `docs/`；
- [x] 开篇写清定位、配套代码路径、版本（v1.1.0）、平台（Win/macOS/Linux）与依赖（零新增依赖），且有可跳转目录；
- [x] 更新历史记录 v1.0.0 初版 + v1.1.0 SQLite 后端变更；
- [x] 第二章讲清 Trace/Span 概念分层与 6 条设计原则；
- [x] 第三章包含对比总表（5 家）+ 异同表（含"缺少什么"）；
- [x] 第四章有目录结构（单数目录 trace/，6个源码文件）、分层图、4 条数据流/调用链路图、与 ReActAgent/LLM/ToolRegistry 的接入边界、后端自动识别逻辑；
- [x] 第五章弹性节讲 Trace/Span 生命周期、父子链维护、三种打点方式选择、后端自动识别逻辑、tool_hook 耗时回推机制、打点通道安全约定；
- [x] 数据模型字段命名与代码一致（kind/status/input/output/parent_id/duration_ms）；
- [x] 源码导读 6 个文件每个都有「负责什么 / 关键实现 / 为什么这么写」；
- [x] 快速上手命令是 Windows PowerShell，且提醒了 `test/` 与标准库同名问题；最小示例 ≤20 行、无需 API Key；新增 SQLite 示例和4种典型用法；
- [x] API 速查按类分小节，列出所有对外方法的参数/返回/说明；新增 SQLiteStore API 和 backend 参数说明；
- [x] 第十章包含 10 条关键设计决策、12 条常见坑（每条都有可执行解法）、6 条跨平台注意事项；
- [x] 测试章节列出 55 项覆盖点 + 关键回归断言 + 全绿基线 55 passed + 合跑 299 passed（+17 skipped）；新增 SQLite 用例和 sqlite_usage 示例验证；
- [x] 演进路线与《演进路线图.md》G1-G4 编号一致；SQLiteStore 已标记为交付完成；
- [x] 章节数量、顺序、编号与其余 6 份模块文档（tools/memory/skills/mcp/context/database）完全一致；
- [x] 全文无"待补充/TBD"残留，未实现项已列入第十二章；
- [x] 导航目录完全保留原有结构和条目顺序，未做任何改动；锚点保持标准Markdown写法，无手动URL编码；
- [x] **所有 `.py/.md/.txt/.bat` 为 CRLF 行尾**（交付前逐字节校验 0 裸 LF、0 裸 CR）。
