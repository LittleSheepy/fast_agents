# FastAgents context 模块编写说明

本文档面向 fast_agents 的**上下文工程（context engineering）**系统，参考 **LangChain 的 `trim_messages` 与 SummarizationNode**、**LlamaIndex 的 Context 构建与召回**、**CrewAI 的上下文传递**、**Anthropic 的 compaction 与 tool result 截断**、**DeepAgents 的「会话状态外化成文件」** 等框架的共性设计，给出本模块的完整说明、源码解析与自定义教程。

> 版本：v1.0 · 2026-10-27 · 首次交付，落地《演进路线图》模块 F 的 F1–F3，铺好 F4 / F5 的地基
> 适用版本：`fast_agents.context`（包内 `__version__ = "1.0.0"`）
> 配套代码：`fast_agents/context/`（7 个文件）+ `examples/context/context_react_agent_demo.py`（5 个示例）+ `test/test_context.py`（16 项），均已实测通过
> 运行环境：Python 3.10+，Windows / macOS / Linux；依赖：**零新增三方依赖**（只用标准库；`tiktoken` 为可选增强，未安装时自动降级为启发式估算）

---

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：context 解决什么问题](#二先想清楚context-解决什么问题)
- [三、主流框架 context 实现对比](#三主流框架-context-实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：五层拼装顺序与三级压缩策略](#五核心机制五层拼装顺序与三级压缩策略)
- [六、核心数据模型](#六核心数据模型)
- [七、逐个文件讲解](#七逐个文件讲解)
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
| 1.0.0 | 2026-10-27 | 首次编写，逐行对齐 `fast_agents/context/` 真实实现；按《XX 模块编写说明》模版 v2.0 重排章节（开篇 → 一～十三 → 附录） | — |

> **破坏性变更：无。** 模块是**纯新增**：新增 `fast_agents/context/` 目录、1 个示例、1 份测试；对既有文件的修改只有两处「追加式」改动——`fast_agents/__init__.py` 追加导出、`fast_agents/agents/react_agent/react_agent.py` 追加 4 个构造参数与 3 个内部方法。**默认开关全部关闭**（`enable_context=False`），不动既有行为，老代码不改一行也能跑。

本次交付同时落地《演进路线图》**模块 F** 的四条任务：

| 编号 | 任务 | 本次状态 |
|---|---|---|
| **F1** | `TokenCounter` + 上下文预算 | ✅ 已实现（`tokens.py`，三级降级计数 + `TokenBudget`） |
| **F2** | 工具结果卸载到磁盘 | ✅ 已实现（`offload.py` + `read_offloaded` 工具） |
| **F3** | 文件系统工具（`ls` / `read` / `write` / `edit`） | ✅ 已实现（`tools.py` 三件套 + `read_offloaded`，`read` 复用既有 `ReadFileTool`） |
| **F4** | 摘要 / 压缩中间件化 | 🟡 部分（三级策略已实现且可开关，但未包成通用中间件，见 [十二章](#十二下一步演进路线)） |
| **F5** | Prompt caching 友好布局 | 🟡 部分（静态在前、动态在后的顺序已就位，未显式打缓存断点，见 [4.4](#44-与-reactagent-的接入点--职责边界)） |

> **相对上一版文档的变化**：上一版 `docs/context模块编写说明.md` 用的是「更新历史 → 主流框架对比 → 模块架构 → 逐个文件讲解 → 上手 → 常见坑 → 演进路线 → 测试」的旧骨架（八节 + 附）。本版按模版 v2.0 重排为「开篇 → 一～十三 → 附录」，**章节数量、顺序、编号与 `docs/记忆模块编写说明.md` 等四份成品完全一致**，内容也补全了旧版缺失的「二、先想清楚」「六、核心数据模型」「九、API 速查」「附录：发布前自检清单」四块。**以本文为准。**

---

## 二、先想清楚：context 解决什么问题

### 2.1 一句话概括

**context 模块负责「在把上下文送进模型之前，算清账、排好序、该砍就砍」。**

它不存历史（那是 `memory` 的事）、不执行工具（那是 `tools` 的事）、不定义消息协议（那是 `messages` 的事）。它只做三件事，按顺序：

| 动作 | 文件 | 回答的问题 |
|------|------|-----------|
| **计量** | `tokens.py` | 这段上下文**有多少 token**？预算还剩多少？ |
| **分层** | `layers.py` | 内容**该放在哪个位置**？超预算时**先砍谁**？ |
| **治理** | `offload.py` / `compressor.py` | 超预算了**怎么办**？卸载？裁剪？还是摘要？ |

三者拼起来，就是门面 `ContextBuilder` 的一次 `build()`。**没有魔法，全是账。**

### 2.2 为什么不能只靠「buffer 按条数裁剪」

模块存在之前，仓库里管上下文的东西有两处，但都**停在局部**：

- `BufferMemory` 的滚动窗口按 **`max_messages`（消息条数）** 裁剪，配一个滚动摘要；
- `ReActAgent._seed_workspace()` 每轮把 buffer 窗口搬进 `self.messages` 工作区。

问题在三处：

1. **「条数」不等于「代价」**。一条消息可能是 3 个 token（`"你好"`），也可能是一整篇 2 万字符的 `read_file` 结果。按条数裁，量大的一条就能把窗口顶爆，而裁掉十条短消息几乎不省 token；
2. **窗口满了没人报警**。没有 token 计数，就不会有「超预算」这个信号，压缩也就无从触发；
3. **大结果无法回收，也无处安放**。一次 `web_search` 返回 5 万字符，塞进 `tool` 消息后，后面十轮都带着它，而其中九轮根本不再看它。

所以 context 的第一条设计原则是：**「裁不裁」的前提是「知道有多少」。** 先有 `TokenCounter`，才谈得上后面的一切。

### 2.3 设计原则

整个模块围绕四条原则展开，后面每一处实现都能回指到其中某一条：

1. **零重型依赖**。与 `memory`（只用 `sqlite3` / `json` / `pathlib`）、`mcp`（自实现 JSON-RPC）一致，context **不做任何硬依赖**：token 计数用 `tiktoken`，**没装就自动降级为启发式估算**；卸载用 `json` + `pathlib`，不引数据库。「装得上、跑得起」优先于「算得准」。
2. **零侵入既有模块**。context **不改 `memory` / `tools` / `messages` 的任何一行**，只从它们的公开接口取数据（`buffer.get_messages()` / `memory.retrieve()` / `ToolResult.output`）。这样即使关掉 context，仓库行为与之前**逐字节一致**。
3. **信息不丢优先于信息变短**。压缩有三条路：**卸载**（丢体积、不丢信息）→ **裁剪**（丢最老的、最不可逆）→ **摘要**（有损、还要多花一次 LLM 调用）。所以顺序固定，把最贵、最不可逆的放最后。
4. **静态在前、动态在后**。拼装顺序不是审美偏好，而是 **prompt caching** 的硬要求：缓存按「前缀匹配」，从请求开头到缓存断点必须逐 token 一致；一旦每轮都在变的记忆被塞进 system 中段，后面全部缓存失效，长会话成本直接翻倍。

### 2.4 概念分层

「上下文」在不同语境里指不同东西。本模块把最顶层拆成五层，各层的数据来源与生命周期如下：

| 层 | 名字 | 数据来源 | 生命周期 | 变化频率 | 典型内容 | 本框架对应 |
|---|---|---|---|---|---|---|
| **L1** | system | `ReActAgent` 构造时生成 | 整个 Agent 生命周期 | 极低（仅加载技能时刷新） | 身份设定、工具清单、技能索引 | `self.system_prompt_template` |
| **L2** | memory | `memory.retrieve()` + `buffer.get_summary()` | 跨轮 | 低 | 长期记忆、会话摘要 | `MemoryManager` |
| **L3** | history | `buffer.get_messages()` | 跨轮，滚动 | 高（每轮增长） | 历轮 user / assistant | `BufferMemory.buffer` |
| **L4** | files | 按需注入 / 卸载回读 | 单轮 | 高 | 文件片段、卸载结果 | `ContextOffloader` |
| **L5** | user | `run()` 入参 | 单轮 | — | 本轮问题 | 调用方 |

> 一句话记忆法：**L1 身份、L2 事实、L3 对话、L4 素材、L5 问题**。前两层相对稳，后三层每轮都在动——所以裁剪的手先伸向 L3。

---

## 三、主流框架 context 实现对比

### 3.1 对比总表

上下文工程是近两年才被单独抽出来的一层。各家叫法不同、边界不同，但要解决的问题高度一致：**窗口有限，而历史、工具结果、文件内容会无界增长，必须有人负责「算账 + 取舍」。**

| 框架 | 核心抽象 | 计量方式 | 压缩 / 裁剪 | 工具结果处理 | 分层与顺序 | 依赖 |
|---|---|---|---|---|---|---|
| **LangChain** | `trim_messages` / `MessagesPlaceholder` | `get_num_tokens_from_messages`（包 tiktoken） | `trim_messages`（按 token 值裁剪）+ `SummaryBufferMemory` | `trim_messages` 可保留最近 N 条含 tool 的组 | 由调用方手工排列，无强制层序 | 重（`langchain-core`） |
| **LangGraph** | `StateGraph` + `SummarizationNode` / Checkpointer | 同上 | 图节点里做「裁剪 + 摘要」，摘要写回 state | 消息列表整体作为 state 处理 | state 即上下文，顺序即写入顺序 | 重（依赖 LangChain 全家桶） |
| **LlamaIndex** | `ChatMemoryBuffer` + `Context` / Node 召回 | tokenizer 可插拔 | 超出 `token_limit` 时按「最近优先」截断 | 检索到的 Node 作为上下文注入 | 检索结果拼进 system / 用户消息 | 中（`llama-index-core`） |
| **CrewAI** | Task / Agent 上下文传递 | 弱（主要靠上下文隔离） | 靠「子 Agent 各持独立上下文」降低压力 | 工具输出进 Agent 自己的上下文 | 无明显分层，靠隔离 | 中 |
| **Anthropic（Claude）** | Compaction + Tool result 截断 | 官方 token 计数 API | **自动 compaction**（超阈值改写历史）+ 手工 `/compact` | tool result 超长截断，可落文件后回读 | 官方建议：稳定内容在前、可变内容在后 | 无（服务端能力） |
| **DeepAgents（LangChain）** | 会话状态外化成文件 + 记忆即工具 | tiktoken | 上下文压缩 + 文件系统 | 大结果写文件，工具按需回读 | 文件为「外部记忆」，按需回读 | 重 |
| **本实现（fast_agents.context）** | `ContextBuilder` 门面 + `ContextWindow` 分层 | `TokenCounter` 三级降级（tiktoken → 通用编码 → 启发式） | **三级策略：卸载 → 裁剪 → 摘要**，可逐项开关 | 超长结果落盘，`read_offloaded` 按行区间回读 | **强制五层顺序**（system → memory → history → files → user） | **零**（只用标准库） |

### 3.2 关键差异

| 维度 | 相同 | 本实现**增加** | 本实现**缺少** |
|---|---|---|---|
| **计量** | 都要先算 token 才谈裁剪 | 三级降级：没装 tiktoken 也能跑，且中文按 CJK 单独处理（1 字≈1 token） | 无精确的、与 provider 对齐的 tokenizer（Claude 用官方 API 计数，本实现是估算） |
| **压缩** | 都支持「裁剪 + 摘要」 | **把「卸载」提到裁剪之前**——先丢体积、再丢信息，最后才有损摘要 | 未做成 LangGraph 那样的「图节点 / 中间件」，仍是构造函数里的方法调用 |
| **工具结果** | 都意识到 tool result 是爆窗主因 | 卸载是**透明**的：`observe()` 未超阈值时零 IO，主循环可以无脑调用 | 未做流式截断（对已在流中的超长输出无能为力） |
| **分层顺序** | 都认同「静态在前」 | 把顺序做成**数据**（`LAYER_ORDER` 常量），因此裁剪器不需要知道谁是 system | 未显式打 cache 断点（没有 `cache_control` 注入点），只把位置留好了 |
| **与记忆的关系** | 都把记忆视作上下文的一部分 | 明确**两条路**：history 只走 buffer、memory 只走检索+摘要，从结构上杜绝重复注入 | 未实现向量召回（预留 `metadata` 扩展位，对齐路线图 B2） |
| **隔离** | CrewAI 用子代理隔离上下文 | —— | 无 sub-agent 隔离（属 `agents` 模块 E3，不在本模块范围） |
| **生态** | 都能与既有工具链协作 | 与 `tools` 的 `ToolResult` 协议、`skills` 的中间件顺序**原生对齐** | 没有跨框架的 serializer / 回调体系 |

**缺少项对应的待办**：tokenizer 精度 → 路线图 F1 延伸；压缩中间件化 → **F4**；缓存断点 → **F5**；语义召回 → **B2**；子代理隔离 → **E3**。详见[第十二章](#十二下一步演进路线)。

### 3.3 一句话收口

**本实现 = LangChain 的「分层 + 可插拔压缩」思路 + Anthropic 的「tool result 截断 + 文件回读」机制 + DeepAgents 的「会话状态外化成文件」实践 − 它们的全部重型依赖 − 它们的黑盒。**

拿走的都是思路，丢掉的是依赖和不可控。代价是功能面窄（没有图编排、没有向量召回），但换来的是**每行都能读懂、随时能改**——这正是 `fast_agents` 这个仓库的定位。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/context/          ← 单数目录，对齐 tools/ memory/ skills/ mcp/
├── __init__.py               模块出口：统一导出「计量 / 分层 / 治理」三组符号
├── tokens.py                 计量：TokenCounter（三级降级）+ TokenBudget + 估算函数
├── layers.py                 分层：ContextLayer / ContextWindow / LAYER_ORDER
├── offload.py                治理·卸载：ContextOffloader（大结果落盘 + 按需回读）
├── compressor.py             治理·压缩：ContextCompressor（三级策略）+ CompressionReport
├── builder.py                门面：ContextBuilder（拼装 + 预算 + 压缩一键走完）
└── tools.py                  工具：read_offloaded + write/edit/list 三件套 + 工厂
```

**同时涉及的既有文件（全部为追加式修改，无删除）**：

```text
fast_agents/__init__.py                                  ← 追加 context 导出（16 个符号）
fast_agents/agents/react_agent/react_agent.py            ← 追加 4 个构造参数 + 3 个内部方法
examples/context/context_react_agent_demo.py             ← 新增示例（5 个）
test/test_context.py                                     ← 新增测试（16 项）
test/test_react_agent.py                                 ← 追加 2 项 context 接入回归
```

> **目录为什么是单数 `context/`**：与 `tools/` `memory/` `skills/` `mcp/` 保持一致（模块目录用单数）。注意区分两件事——`fast_agents/context/` 是**模块目录**（单数），而模块内的 `build_context_tools()` 产出的是一组**工具**（复数概念），两者不是一回事。

### 4.2 分层

模块内部按「出口层 → 门面层 → 能力层 → 基础层」四层组织，依赖方向**只向下**（上层可以调下层，下层绝不回调上层）：

```text
① 出口层     __init__.py
             导出「计量 / 分层 / 治理 / 门面 / 工具」五组符号
                 │
② 门面层     builder.py  ── ContextBuilder
             对外只有 build / observe / record_turn / reset_turn
                 │
③ 能力层     compressor.py ── ContextCompressor      （依赖 tokens + layers + offload）
             offload.py    ── ContextOffloader       （只依赖标准库）
             tools.py      ── 4 个 ToolBase           （依赖 offload + tools 模块的 ToolBase）
                 │
④ 基础层     layers.py     ── ContextLayer / ContextWindow  （依赖 messages + tokens）
             tokens.py     ── TokenCounter / TokenBudget     （零依赖，只依赖标准库）
```

这么分的直接好处：**门面可以换、能力层可以换，而基础层（`tokens` / `layers`）几乎不需要动**。比如你想把压缩换成「只摘要不裁剪」，改的是 `ContextCompressor` 的开关（能力层），`TokenCounter` 和 `LAYER_ORDER` 一行都不用碰。

### 4.3 数据流 / 调用链路

一次完整的 `build()`，数据是这样流的（箭头旁标了方法名，可顺藤摸瓜到源码）：

```text
调用方 / ReActAgent.run()
        │
        ├─ system_prompt ─┐
        ├─ memory packets ┤   参数
        ├─ history ───────┤
        ├─ file_snippets ─┤
        └─ user_input ────┘
                 │
                 ▼
   ContextBuilder.build()                 ← builder.py：五层依次就位
        │
        ├─ L1 system  → ContextLayer(required=True, priority=100)
        ├─ L2 memory  → _render_packets() → _fit_layer(配额 0.25)
        ├─ L3 history → ContextCompressor.compress(配额 = 剩余)
        │                    │
        │                    ├─ ① ContextOffloader.offload()   超长消息 → 落盘，留引用
        │                    ├─ ② _trim_layer()                 成组裁掉最老历史
        │                    └─ ③ _summarize_layer()            LLM 摘要，写回层首
        ├─ L4 files   → _fit_layer(配额 0.25)
        └─ L5 user    → ContextLayer(required=True, priority=100)
                 │
                 ▼
   _enforce_global_budget()               ← 全局兜底：整窗仍超则继续成组丢最老历史
                 │
                 ▼
   ContextWindow                          ← layers.py
        ├─ ordered_layers()  按 LAYER_ORDER 排序
        ├─ flatten()         → List[Message]        （回填 self.messages 工作区）
        └─ to_openai_messages() → List[dict]        （直接喂 LLM）
                 │
                 ▼
            LLM.invoke()
                 │
        ┌────────┴────────┐
        │ 工具返回超长结果  │
        ▼                 │
   ContextBuilder.observe(tool_name, content)     ← 未超阈值原样返回（零 IO）
        │
        ▼
   ContextOffloader 落盘 → 返回「引用文本」放进 tool 消息
        │
        ▼
   模型需要全文时 → 调 read_offloaded(offload_id, start_line, max_lines) 按需回读
```

链路里有两个**必须记住的先后关系**：

1. **`observe()` 发生在技能中间件 `after_tool_call` 之前**。顺序是：`invoke_tool → _render_observation → _offload_observation → skill_middleware.after_tool_call`。如果把卸载放在技能中间件之后，中间件加在观察文本里的「纠错前缀」可能被整段卸载掉，模型反而看不到纠错提示；
2. **`compress()` 只在 history 层做完整三级压缩**。L2 / L4 两层走的是 `_fit_layer()`（只卸载 + 丢最老条目，不做摘要）——因为记忆和文件片段本身就是「可选参考」，没必要为它们多花一次 LLM 调用。

### 4.4 与 ReActAgent 的接入点 / 职责边界

#### 怎么挂进 Agent

context 挂在 `ReActAgent` 上，**与 memory 同级**，都是成员变量：

| 接入项 | 位置 | 说明 |
|---|---|---|
| 构造参数 `enable_context` | `__init__` | 是否启用（**默认 `False`**，保持既有行为不变） |
| 构造参数 `ctx` | `__init__` | 可注入自定义 `ContextBuilder`；不传时自动创建 |
| 构造参数 `context_budget` | `__init__` | 覆盖默认预算（int，token 数） |
| 构造参数 `context_offload_threshold` | `__init__` | 触发卸载的字符数阈值（默认 4000） |
| 成员变量 `self.ctx` | `__init__` | `ContextBuilder` 实例；关闭时为 `None` |
| `_init_context()` | `__init__` 内调用 | 建 builder + 注册 4 个工具 + 刷新 system 提示词 |
| `_fit_context(input_text)` | `run()` 内调用 | 用 builder 重建本轮工作区，写回 `self.messages` |
| `_collect_context_packets(query)` | `_fit_context` 内调用 | 收集「长期记忆 + 会话摘要」，**不含 buffer 历史** |
| `_offload_observation(name, obs)` | 工具调用后 | 超长观察文本卸载；关闭时原样返回 |

启用后构造与运行时会打印：

```text
🧭 上下文工程已启用：预算 27000 token / 计数后端 tiktoken / 卸载阈值 4000 字符
...
🧭 token 预算 475/27000（1.8%，窗口 32000，预留输出 4000）
```

#### 两条职责边界（全篇最值钱的一段）

`ReActAgent` 的既有约定是「**单一数据源**：跨轮历史只存在 `MemoryManager.buffer`，`self.messages` 只是本次 run 的推理工作区」。接入 context 最容易踩坏的就是这条约定，具体是两处：

**边界一：本轮用户输入只能出现一次。**

`run()` 的原有逻辑是：`_seed_workspace()` 搬历史 → 第 1 步 `append_user(user_input)`。而开启 context 后，`_fit_context()` 已经把本轮问题放进了 **L5 user 层**。如果第 1 步再 `append_user` 一次，模型会看到**两条完全相同的 user 消息**——轻则浪费 token，重则让模型以为用户问了两遍。

```python
                # 系统提示词已在构造/seed 时就位，这里只追加本轮用户输入（仅第 1 步追加）
                # 开启 context 时，本轮用户输入已由 _fit_context 放进 user 层，无需重复追加
                if current_step == 1 and not (self.enable_context and self.ctx is not None):
                    self.messages.append_user(user_input)
```

**边界二：长期记忆不再二次拼进 user。**

原有逻辑会把检索到的长期记忆 `_merge_memory()` 拼进本轮用户输入文本。开启 context 后，长期记忆改由 **L2 memory 层**统一承载（`_collect_context_packets()` 产出 packet），因此 `user_input` 要还原成原始输入：

```python
        if self.enable_context and self.ctx is not None:
            self._fit_context(input_text)
            user_input = input_text          # 记忆已进 memory 层，user 只留原始问题
```

这两条边界都有**回归断言**锁死，改动时不会悄悄退化：

- `test_react_agent.py::test_context_user_input_not_duplicated` —— 多工具步场景下，user 输入也只注入一次，不随步数累积；
- `test_react_agent.py::test_context_enabled_builds_and_registers_tools` —— 开启 context 后 4 个工具确实注册、`agent.ctx` 非空。

> **一处已知的粗糙点**：`_init_context()` 里 `workspace="."` 是硬编码（与 `ReadFileTool` 的默认行为一致）。如果你的工作目录不在启动目录，建议用 `ContextBuilder(workspace="D:/code")` 显式构造后通过 `ctx=` 注入。这一点在[第十章](#十设计决策与常见坑)列为坑项。

## 五、核心机制：五层拼装顺序与三级压缩策略

> 本章是 context 模块的弹性节，收拢两块模块特有机制：**分层怎么排**、**超预算怎么砍**。两块合起来回答了本模块最核心的问题——「窗口满了，动谁、怎么动、动到什么程度」。

### 5.1 五层拼装顺序

拼装顺序由 `layers.py` 里的常量 `LAYER_ORDER` 定义（数字越小越靠前）：

```python
LAYER_ORDER: Dict[str, int] = {
    "system": 10,
    "memory": 20,
    "summary": 25,
    "history": 30,
    "files": 40,
    "tool_result": 45,
    "user": 90,
}
```

| 层 | order | required | priority | 内容 | 为什么在这个位置 |
|---|---|---|---|---|---|
| **system** | 10 | ✅ | 100 | 身份 + 工具清单 + 技能索引 | 最稳定，做 prompt caching 的锚点；**绝不裁剪** |
| **memory** | 20 | ❌ | 80 | 长期记忆 + 会话摘要 | 跨轮变化慢，放前面利于缓存 |
| **summary** | 25 | ❌ | — | 压缩产物专用位（预留） | 摘要本质是「历史的替身」，紧贴 history |
| **history** | 30 | ❌ | 60 | 历轮 user / assistant | 唯一持续增长的一层，压缩的主战场 |
| **files** | 40 | ❌ | 40 | 文件片段 / 卸载回读 | 动态、按需，放历史之后 |
| **tool_result** | 45 | ❌ | — | 当轮工具结果（预留） | 与 files 同级，纯动态 |
| **user** | 90 | ✅ | 100 | 本轮问题 | **必须在最后**，紧贴模型待办事项 |

两条不可动摇的规则：

1. **`system` 在最前、`required=True`**。它是整段上下文里最长、最稳的一段，缓存收益最大；同时裁剪器永远不动它。
2. **`user` 在最后、`required=True`**。模型对「离得最近的内容」注意力最集中，把待回答问题压在最底部，指令遵循最稳。

> **为什么顺序要写成「数据」而不是「代码里的 if-else」**：因为这样一来，**裁剪器不需要知道谁是 system、谁是 history**。它只看 `required` 和 `priority` 两个字段做决策。要新增一层（比如 `summary`），只需在 `LAYER_ORDER` 里加一行；要调整优先级，只改数字。这就是「把顺序和优先级做成数据」的收益。

### 5.2 三级压缩策略

当 `history` 层超出配额，`ContextCompressor.compress()` 按固定顺序依次尝试，**上一级解决不了才进下一级**：

| 级别 | 策略 | 手段 | 信息损失 | 代价 | 触发条件 |
|---|---|---|---|---|---|
| **第 1 级** | **卸载**（offload） | 超长消息落盘，上下文里只留引用 id + 预览 | **无损失**（全文可回读） | 一次磁盘写 | `enable_offload` 且传了 `offloader` |
| **第 2 级** | **裁剪**（trim） | 按「原子组」丢掉最老的历史 | **有损失**（不可逆），但保住了 tool 配对 | 极低（纯内存操作） | 卸载后仍超配额，且 `enable_trim` |
| **第 3 级** | **摘要**（summary） | 让 LLM 把被压掉的历史写成一段话，写回层首 | **有损**（细节丢失） | 一次额外 LLM 调用，可能失败 | 裁剪后仍超配额，且有 `llm` |

三条设计取舍：

1. **把「卸载」放在最前，是本实现与多数框架最不一样的地方。** LangChain / LlamaIndex 的主流做法是「裁剪 + 摘要」，而本实现认为：**能靠落盘解决的，绝不用有损手段解决**。一次 `web_search` 的 5 万字符结果落盘后，模型需要时随时 `read_offloaded` 回读——信息零损失，上下文却只剩几十字符；
2. **「裁剪」必须以组为单位，tool 配对绝不拆散。** 这是 OpenAI 的强校验：`tool` 消息必须紧跟在带 `tool_calls` 的 `assistant` 消息之后。乱裁会直接 400。所以裁剪前先调 `_group_messages()` 切原子组；
3. **「摘要」一定要能失败。** 摘要是一次额外 LLM 调用，它自己就可能限流 / 超时 / 抽风。本实现让它**双层降级**：LLM 调用失败 → 退回截断；截断也拿不到东西 → 上层还有 `keep_recent` 保底。

#### 什么场景选哪一级

| 场景 | 建议 | 理由 |
|---|---|---|
| 工具结果动辄几万字符（搜索、抓网页、读大文件） | **开全部三级**（默认） | 卸载吃掉大头，后两级很少走到 |
| 历史以「短问答」为主，几乎没有长结果 | 关卸载、留裁剪 + 摘要 | 卸载没有收益，还平白落盘 |
| 有严格的「中间步骤不能丢」要求 | **只开卸载 + 摘要，关裁剪** | 裁剪不可逆；宁可多花一次 LLM 调用 |
| 没有可用的 LLM（本地小模型 / 离线） | 关摘要 | 摘要需要 LLM；关闭后靠前两级兜底 |
| 预算极紧（4k 窗口） | 三级全开 + 调小 `keep_recent` | 必须榨干每一 token |

### 5.3 卸载与回读：模型看到的是什么

卸载不是「把内容删掉」，而是**把内容搬到一个模型能自己取回来的地方**。`offload()` 落盘后返回的引用文本形如：

```text
[工具结果已卸载] 工具 `web_search` 返回了 52360 字符，全文已写入离线文件，不在上下文中。
offload_id=8f3a1c9d2e4b6a70
内容开头预览：第一条搜索结果：上下文工程（context engineering）是……
如需查看完整内容，调用工具 read_offloaded(tool_name 的 offload_id="8f3a1c9d2e4b6a70"，
可选 start_line/max_lines) 按需读取。
```

这段文字的设计要点有三：**说清事实**（原文多长、已外置）、**给出坐标**（`offload_id`）、**教会方法**（调用哪个工具、支持什么参数）。模型读了这段，就知道「这里有东西、但它不在我眼前、我该用什么取」。

回读 `read_offloaded(offload_id, start_line=1, max_lines=200)` 返回带行号的分段文本：

```text
📦 卸载内容 8f3a1c9d2e4b6a70（来自工具 web_search，共 1873 行，显示 1-200 行）
    1|第一条搜索结果：……
    2|……
  ...|...
  200|……
⚠️ 未读完，如需后续内容把 start_line 设为 201
```

**为什么默认只回读 200 行而不是全文**：如果回读也把 5 万字符全塞回去，上下文立刻又爆了——那就等于没卸载。分段回读让模型自己控制「这次要看多少」，这是「按需」两个字的具体落点。

---

## 六、核心数据模型

context 模块的数据模型**全部用 `dataclasses.dataclass`**，没有用 pydantic。这是刻意的：`memory` / `mcp` 里用 pydantic 的是**需要校验的外部输入**（工具参数、配置），而 context 里的对象都是**模块内部流转的结构**，不涉及不可信输入，用 `dataclass` 更轻、更快、更易读。**需要校验的两个地方另有出口**：工具入参用 pydantic（见 `7.7` 的 `ReadOffloadedInput` 等）。

命名对齐既有约定：工具类照旧用 `args_schema`（不是 `param_model`）、`run()`（不是别的）、`to_openai_tool()`；返回结构化结果照旧用 `ToolResult`。

### 6.1 `TokenBudget`（`tokens.py`）

把「模型窗口」切成「可用预算」和「预留回复空间」。

```python
@dataclass
class TokenBudget:
    max_context_tokens: int = 128_000   # 模型上下文窗口
    reserve_for_output: int = 4_000     # 给模型回复预留的空间
    safety_margin: int = 1_000          # 估算误差的安全余量
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_context_tokens` | `int` | `128_000` | 模型窗口大小（4k / 8k / 32k / 128k / 200k） |
| `reserve_for_output` | `int` | `4_000` | **必须预留**：不预留的话输入占满窗口后模型没余量输出，会「请求成功但回答被截断」或直接 400 |
| `safety_margin` | `int` | `1_000` | 估算误差余量（启发式计数有 ±15% 误差） |

关键公式与四个方法：

```python
available = max_context_tokens - reserve_for_output - safety_margin   # 可放输入的预算
```

| 方法 / 属性 | 返回 | 说明 |
|---|---|---|
| `available` | `int` | 可用于输入的预算（下限保护为 1） |
| `remaining(used)` | `int` | 还能再放多少 |
| `usage_ratio(used)` | `float` | 预算使用率（>1 表示已超） |
| `is_over(used)` | `bool` | 是否超预算 |
| `describe(used)` | `str` | 一行日志摘要 |

### 6.2 `ContextLayer`（`layers.py`）

上下文中的**一层**。它是「分层」概念的最小载体，也是裁剪器唯一需要理解的对象。

```python
@dataclass
class ContextLayer:
    name: str
    messages: List[Message] = field(default_factory=list)
    priority: int = 50
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | `str` | 必填 | 层名（`system` / `memory` / `history` / `files` / `user`） |
| `messages` | `List[Message]` | `[]` | 这一层的消息 |
| `priority` | `int` | `50` | 裁剪优先级，**越大越先保留** |
| `required` | `bool` | `False` | `True` 表示**绝不裁剪**（system / 本轮 user） |
| `metadata` | `Dict` | `{}` | 附加信息，便于调试统计 |

关键属性 / 方法：

| 成员 | 返回 | 说明 |
|---|---|---|
| `order`（property） | `int` | 拼装顺序，查 `LAYER_ORDER`；未知层默认 60（history 与 user 之间） |
| `is_empty()` | `bool` | 层里有没有消息 |
| `token_count(counter)` | `int` | 本层 token 数（按 OpenAI 格式逐条累计） |
| `add(msg)` / `extend(msgs)` / `clear()` | `self` | 就地增删 |

### 6.3 `ContextWindow`（`layers.py`）

上下文窗口：一组按层名索引的 `ContextLayer`。核心动作只有一个——`flatten()`：按层序把各层拼成一串 `Message`。

| 方法 | 返回 | 说明 |
|---|---|---|
| `add_layer(layer)` | `ContextLayer` | 加一层；**同名层会合并**（并取更宽松的 `required` / 更高的 `priority`） |
| `get_layer(name)` | `ContextLayer \| None` | 按名取层 |
| `list_layers()` | `List[ContextLayer]` | 取全部层（添加顺序） |
| `append_message(name, msg, **kw)` | `ContextLayer` | 往指定层追加一条，层不存在自动建 |
| `ordered_layers()` | `List[ContextLayer]` | 按 `order` 排序（稳定排序，同层内保序） |
| `flatten()` | `List[Message]` | 按层序拼成 `Message` 列表 |
| `to_openai_messages()` | `List[dict]` | 按层序拼成 OpenAI 格式，直接喂 LLM |
| `token_count(counter)` | `int` | 整窗 token |
| `layer_stats(counter)` | `List[dict]` | 逐层统计（供打印「上下文构成」） |
| `validate()` | `List[str]` | 层级卫生检查：tool 消息是否有配对 |

`validate()` 是**上线前的最后一道闸门**，它专门检查那条最容易 400 的规则：

```python
        problems: List[str] = []
        previous_had_tool_calls = False
        for message in self.flatten():
            if message.role == "tool" and not previous_had_tool_calls:
                problems.append(
                    f"tool 消息缺少配对的 assistant(tool_calls)：{str(message.content)[:40]!r}"
                )
            previous_had_tool_calls = bool(message.tool_calls)
        return problems
```

### 6.4 `OffloadRecord`（`offload.py`）

一次卸载的元数据，会写进 `index.json`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `offload_id` | `str` | 卸载 id（`uuid4().hex[:16]`，只含十六进制字符） |
| `tool_name` | `str` | 来源工具名（便于溯源） |
| `path` | `str` | 正文文件路径 |
| `original_chars` | `int` | 原始字符数 |
| `summary` | `str` | 摘要（默认取预览前 200 字） |
| `preview` | `str` | 预览（默认前 400 字符） |
| `created_at` | `float` | 创建时间戳（淘汰排序用） |
| `metadata` | `Dict` | 附加元数据 |

提供 `to_dict()` / `from_dict()` 做 `index.json` 的序列化往返；`ensure_ascii=False` 保证中文可读。

### 6.5 `CompressionReport`（`compressor.py`）

一次压缩的结果报告，便于日志与测试断言。

| 字段 | 类型 | 说明 |
|---|---|---|
| `offloaded` | `int` | 被卸载的消息条数 |
| `trimmed` | `int` | 被裁掉的原始消息条数 |
| `summarized` | `bool` | 是否生成了新摘要 |
| `tokens_before` / `tokens_after` | `int` | 压缩前后 token |
| `notes` | `List[str]` | 过程备注 |
| `saved`（property） | `int` | 省下的 token |
| `summary_line()` | `str` | 一行日志，形如「压缩：卸载 2 条 / 裁剪 6 条 / 摘要 否；token 9000 → 2400（省 6600）」 |

### 6.6 `BuildReport`（`builder.py`）

一次 `build()` 的结果报告。

| 字段 | 类型 | 说明 |
|---|---|---|
| `tokens` | `int` | 整窗 token |
| `stats` | `List[dict]` | 逐层统计（同 `layer_stats()`） |
| `compression` | `CompressionReport \| None` | 本次 history 层的压缩报告 |
| `warnings` | `List[str]` | 告警（如「兜底再裁 6 条」「某层即使清空仍超配额」） |
| `describe()` | `str` | 多行报告，形如： |

```text
上下文构成：3 层 / 8 条消息 / 1240 token
  system     1 条   820 token  required
  history    6 条   380 token
  user       1 条    40 token  required
```

---

## 七、逐个文件讲解

### 7.1 `__init__.py` —— 模块出口

**负责什么**：把「计量 / 分层 / 治理 / 门面 / 工具」五组符号统一导出，并按这个分组维护 `__all__`。

```python
from .builder import BuildReport, ContextBuilder
from .compressor import CompressionReport, ContextCompressor, DEFAULT_COMPACT_PROMPT
from .layers import LAYER_ORDER, ContextLayer, ContextWindow
from .offload import ContextOffloader, OffloadRecord
from .tokens import TokenBudget, TokenCounter, count_message_tokens, estimate_text_tokens
from .tools import (EditFileTool, ListFilesTool, ReadOffloadedTool,
                    WriteFileTool, build_context_tools)

__version__ = "1.0.0"
```

**为什么这么写**：`__all__` 的顺序不是随意的，而是**按「计量 → 分层 → 治理 → 门面 → 工具」的概念顺序**排列——读者打开这个文件，就能一眼看出模块的分组结构。另外 `__version__ = "1.0.0"` 是包内版本号，与仓库根 `fast_agents/version.py` 的 `0.0.1` 是两套（前者标模块，后者标框架），别混。

> **你要加东西时最该看这里**：新增可公开的类，必须同时改 `__init__.py` 的 `import` 与 `__all__`，否则 `from fast_agents.context import Xxx` 会失败（对齐 `memory` 模块的约定）。

### 7.2 `tokens.py` —— 计量：先知道有多少，才谈得上怎么裁

**负责什么**：`TokenCounter`（token 计数）+ `TokenBudget`（预算换算）+ 三个估算函数。整个模块的「地基」。

#### 三级降级计数

```python
    @staticmethod
    def _load_encoder(model: str):
        """尝试加载 tiktoken 编码器；任何异常都返回 None（降级）"""
        try:
            import tiktoken  # type: ignore
        except Exception:            # noqa: BLE001 —— 没装就是没装，不能影响框架运行
            return None
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:            # noqa: BLE001 —— 模型名不认识时退回通用编码
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:        # noqa: BLE001
                return None
```

**为什么这么写**：`tiktoken` 是**可选增强**而非硬依赖。三层尝试（按模型找编码 → 退回通用编码 → 放弃），每一层都用 `except` 兜住，**最终一定不会抛异常**。`self.backend` 会记下当前用的是 `"tiktoken"` 还是 `"heuristic"`，日志里能直接看到。

#### 启发式估算：中文单独处理

```python
def estimate_text_tokens(text: str, safety_ratio: float = 1.15) -> int:
    cjk = latin = other = 0
    for ch in text:
        if _is_cjk(ch):                                    # 中日韩：基本 1 字 1 token
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch == "_"):  # 英文/数字：4 字符 ≈ 1 token
            latin += 1
        else:                                              # 标点/空白：2 字符 ≈ 1 token
            other += 1
    return int((cjk + latin / 4.0 + other / 2.0) * safety_ratio) + 1
```

**为什么中文要单独算**：BPE 分词里，**一个汉字基本就是一个 token**，而英文是 4 个字符左右一个 token。如果用「总字符数 / 4」这种粗糙公式估中文，会**严重低估**（中文实际约是估值的 4 倍），结果就是「以为没超、其实已经爆了」。所以这里按 CJK / 拉丁 / 其他三类分开累加，最后乘安全系数 `1.15`，**宁可多算不可少算**。

#### 消息级计数：别漏掉固定开销

```python
_PER_MESSAGE_OVERHEAD = 4    # 每条消息的固定开销
_PER_REQUEST_OVERHEAD = 3    # 每次请求的固定开销
```

`count_message_tokens()` 会把三类东西算进去：**正文**（文本块）、**多模态块**（图片按固定 85 token 粗算）、**function calling**（工具名 + 参数 JSON，再加 6 token 结构开销）。很多人只算 `content`，结果每轮都少算几百上千 token。

`count_tool_schemas()` 单独统计工具 schema 的开销——**这是最容易被忽略的一块**：每轮请求都要带全套 schema，10 个工具轻松吃掉一两千 token。

> **你要改计数精度时最该看这里**：换 tokenizer、调整 `safety_ratio`、改多模态粗算值，都在这个文件。

### 7.3 `layers.py` —— 分层：拼装顺序也是一份数据

**负责什么**：`ContextLayer` / `ContextWindow` / `LAYER_ORDER`。管两件事——**拼装顺序**与**裁剪优先级**。

#### 顺序做成常量

```python
# 层的标准顺序（数字越小越靠前，越靠前越静态、越适合被缓存）
LAYER_ORDER: Dict[str, int] = {
    "system": 10, "memory": 20, "summary": 25,
    "history": 30, "files": 40, "tool_result": 45, "user": 90,
}
```

**为什么这么写**：一旦顺序是常量，`ContextLayer.order` 就只是个查表动作，`ordered_layers()` 只是个 `sorted`。**要加层、要调序，只改这张表**，不碰任何逻辑。

#### 同名层合并而非覆盖

```python
    def add_layer(self, layer: ContextLayer) -> ContextLayer:
        existing = self.get_layer(layer.name)
        if existing is not None:
            existing.messages.extend(layer.messages)
            existing.required = existing.required or layer.required
            existing.priority = max(existing.priority, layer.priority)
            existing.metadata.update(layer.metadata)
            return existing
        self._layers.append(layer)
        return layer
```

**为什么这么写**：合并时取**更宽松的 `required`**、**更高的 `priority`**——这是「安全侧优先」：如果同一层被两次以不同严格度添加，保留更不容裁剪的那个，避免出现「本该 required 的层被裁掉」。

#### 卫生检查

`ContextWindow.validate()` 只做一件事：扫一遍 `flatten()` 的结果，检查有没有**孤立的 tool 消息**。这是 OpenAI 接口的强校验项，也是本模块唯一「宁可报错也不放过」的地方。

> **你要调层序、加新层时最该看这里**：`LAYER_ORDER` 加一行 + 在 builder 的 `build()` 里 `add_layer`。

### 7.4 `offload.py` —— 卸载：大结果落盘，上下文只留引用

**负责什么**：`ContextOffloader`——把超长工具结果写到磁盘，返回一段「引用文本」，并支持按 id / 行区间回读。

#### 落盘布局与路径安全

```text
<workspace>/.fast_agents/context/offload/
├── index.json      # id → 元数据（工具名、原长、摘要、预览）
└── <id>.txt        # 正文
```

```python
_SAFE_ID = re.compile(r"^[0-9a-f]{6,64}$")     # id 只允许十六进制字符
...
        self.workspace = Path(workspace).resolve()
        self.root = (self.workspace / subdir).resolve()
        # 安全兜底：卸载目录必须仍在 workspace 内
        try:
            self.root.relative_to(self.workspace)
        except ValueError:
            raise ValueError("卸载目录必须位于 workspace 内")
```

**为什么这么写**：id 用 `uuid4().hex[:16]`（纯十六进制），再用白名单正则卡一道，**从源头杜绝 `../` 路径穿越**；`read()` 里还会再校验一次目标文件是否仍在 `workspace` 内。这套思路与 `ReadFileTool` 的沙箱一致。

#### 没超阈值 → 零 IO

```python
    def offload(self, tool_name, text, *, threshold=None, summary="", metadata=None) -> str:
        text = text or ""
        if not self.should_offload(text, threshold):
            return text                     # ← 没超阈值：原样返回，不落盘、不写索引
        ...
```

**为什么这么写**：这让 `offload()` **可以无脑在每次工具返回后调用**（Agent 主循环就是这么做的）。绝大多数工具结果都很短，走的是「原样返回」这条零开销路径，只有真正超大时才付出磁盘代价。

#### 有界增长

`max_files=200`（默认）是硬上限：超过时按 `created_at` 淘汰**最早创建**的卸载文件。没有这个上限，一个长跑进程会无声无息把磁盘写满。

#### 回读：行区间 + 可读错误

```python
    def read(self, offload_id, start_line=1, end_line=None, max_lines=200) -> str:
        if not _SAFE_ID.match(offload_id or ""):
            return f"[读取失败] 非法的 offload_id：{offload_id!r}"
        record = self._index.get(offload_id)
        if record is None:
            return f"[读取失败] 未找到 offload_id={offload_id} 的卸载记录（可能已被淘汰）"
        ...
```

**为什么返回错误字符串而不是抛异常**：这个方法会被模型通过 `read_offloaded` 工具调用，**参数是模型给的**。模型完全可能编一个不存在的 id。如果抛异常，工具链就断了；返回可读的错误文本，模型看到「未找到」会自己换个 id 或换思路——**把纠错权交还给模型**。

> **你要调卸载粒度时最该看这里**：`threshold`（触发阈值）、`preview_chars`（预览长度）、`max_files`（上限）。

### 7.5 `compressor.py` —— 治理：三级策略，逐级降级

**负责什么**：`ContextCompressor`——对 `history` 层做三级压缩，产出 `CompressionReport`。

#### 主入口：逐级尝试

```python
        # 第 1 级：卸载超长消息（信息不丢，收益最大）
        if self.enable_offload and self.offloader is not None:
            self._offload_layer(layer, report)

        # 第 2 级：成组裁剪最老的历史
        if self.enable_trim and layer.token_count(counter) > limit:
            self._trim_layer(layer, limit, counter, report)

        # 第 3 级：仍然超预算 → 摘要（把裁掉的内容压成一段话）
        if self.enable_summary and layer.token_count(counter) > limit:
            self._summarize_layer(layer, limit, counter, report)
```

**为什么这么写**：每一级执行完都**重新判断一次是否仍超预算**（注意第 2、3 级的 `if` 条件），所以能自然做到「第一级就够了就不做第二级」。这比「按比例各做一点」更省，也更符合「先便宜后昂贵」的思路。

#### 成组切割：tool 配对绝不拆散

```python
def _group_messages(messages: Sequence[Message]) -> List[List[Message]]:
    """把消息按「原子组」切开
    - assistant(tool_calls) + 后续 tool 消息 → 一组（不可拆）；
    - 其余每条消息各自成组。
    """
    groups: List[List[Message]] = []
    buffer: Optional[List[Message]] = None
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            buffer = [message]; groups.append(buffer); continue
        if message.role == "tool" and buffer is not None:
            buffer.append(message); continue
        buffer = None
        groups.append([message])
    return groups
```

**这是「不许把 tool 配对拆坏」这条硬规则在代码里的落点。** 裁剪和全局兜底都以「组」为单位，所以永远不可能出现「留了 tool 结果、丢了对应的 `assistant(tool_calls)`」这种 400 组合。`test_compressor_trim_keeps_tool_pairing` 专门锁死这一条。

#### 摘要失败必须降级

```python
    def _call_summarize(self, history_text: str) -> str:
        prompt = self.compact_prompt.format(max_words=self.summary_max_words, history=history_text)
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.0)
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"⚠️ 上下文摘要失败，降级为截断：{e}")
        return history_text[: self.summary_max_words * 2]
```

**为什么这么写**：摘要是一次额外 LLM 调用，它自己就可能失败。如果让它抛异常上去，用户会因为「压缩失败」而拿不到任何回答——典型的「为了优化体验反而毁掉体验」。所以是**双层降级**：LLM 失败 → 退回截断；截断也拿不到东西 → 上层还有 `keep_recent` 保底。处理方式与 `BufferMemory._summarize` 一致。

#### 摘要写回层里，而不是丢掉

```python
        layer.messages = [
            Message("system", f"[历史已压缩，供参考]\n{summary}",
                    metadata={"type": "compacted_summary", "required": True})
        ] + keep
```

摘要以一条消息形式**写回 history 层开头**，后面接保留下来的最近若干条。这样下一轮再 build 时这段摘要仍在，不会「压完就忘」。`self._accumulated_summary` 也留了一份，供 Agent 跨轮注入（`ContextBuilder.summary` 属性暴露）。

> **你要换压缩策略时最该看这里**：三个 `enable_*` 开关、`keep_recent`、`compact_prompt`、`summary_max_words`。

### 7.6 `builder.py` —— 门面：把上面四件事串成一次调用

**负责什么**：`ContextBuilder`——对外只暴露少量动作，Agent 侧不需要知道底下分了哪些层、用哪种压缩策略（对齐 `MemoryManager` 的写法）。

#### 三个动作

```python
builder.build(system_prompt=..., history=..., user_input=..., memory_packets=...)  # 拼装 + 裁剪
builder.observe(tool_name, content)          # 工具结果回来后：判断是否卸载（无脑调用也安全）
builder.record_turn(user_input, answer)      # 一轮结束：可选地写回工作区
```

外加 `reset_turn()`（清空本轮工作区）与 `stats()`（概要统计）。

#### build：五层依次就位

```python
        # ---- L1 system：required，不参与裁剪 ----
        if system_prompt:
            system_layer = ContextLayer(name="system", required=True, priority=100)
            system_layer.add(Message("system", system_prompt, metadata={"type": "system", "required": True}))
            self.window.add_layer(system_layer)

        # ---- L2 memory：长期记忆 / 会话摘要 ----
        memory_text = self._render_packets(memory_packets) if memory_packets else ""
        ...

        # ---- L3 history：滚动窗口，最容易爆的一块 ----
        history_layer = ContextLayer(name="history", priority=60)
        for message in history or []:
            if message.role == "system":
                continue        # system 由 L1 统一负责，历史里的 system 一律丢弃，避免重复注入
            history_layer.add(message)
        report = self.compressor.compress(history_layer, self.counter, self._quota_for("history"))
        ...

        # ---- L5 user：required，永远最后 ----
        if user_input:
            user_layer = ContextLayer(name="user", required=True, priority=100)
            user_layer.add(Message("user", user_input, metadata={"type": "user", "required": True}))
            self.window.add_layer(user_layer)
```

**为什么 L3 要主动丢弃 `system` 消息**：历史里可能混进旧的 system（比如上一轮注入过的技能提示）。如果照搬，就会出现「两条 system」，且第二条排在历史中间——既浪费 token，又破坏缓存前缀。**统一由 L1 负责 system，是唯一数据源原则在分层里的体现。**

#### 层配额

```python
        self.layer_quotas = layer_quotas or {
            "system": 0.35, "memory": 0.25, "files": 0.25,
        }
```

**为什么 system 也给配额**：正常情况 system 不会超，但技能索引 + 工具清单在工具很多时可能很大。给个配额是**防御性设计**：一旦超了，`_fit_layer()` 会告警而不是默默爆掉。`history` 没列在配额里，它拿的是**剩余额度**（`_quota_for()` 里算的 `available - 已分配`）。

#### 全局兜底

```python
    def _enforce_global_budget(self, warnings: List[str]) -> None:
        """全局兜底：整体超预算时，从最老的 history 消息成组丢弃"""
        if not self.window.token_count(self.counter) > self.budget.available:
            return
        ...
        groups = _group_messages(history.messages)
        while groups and self.window.token_count(self.counter) > self.budget.available:
            if len(groups) <= self.compressor.keep_recent:
                break
            dropped += len(groups.pop(0))
```

**为什么在层配额之外还要全局兜底**：层配额是按「层」算的，但**总预算**是全局的。「system 用到 35%、memory 也刚好卡到 25%、files 也卡到 25%」——每层都合规，加起来仍可能超。全局兜底就是处理这种「局部合规、整体超标」的情况，并且它**只动 history**（其他层要么 required、要么已经压过）。

> **你要接新层 / 调配额时最该看这里**：`build()` 里的层顺序、`layer_quotas` 默认值。

### 7.7 `tools.py` —— 让模型自己管理上下文

**负责什么**：4 个 `ToolBase` 工具 + 1 个工厂，让 Agent 能**自己读回卸载内容、自己读写文件**。全部继承既有 `ToolBase`，用 pydantic 定义 `args_schema`。

| 工具 | `name` | 入参（`args_schema`） | 作用 |
|---|---|---|---|
| `ReadOffloadedTool` | `read_offloaded` | `offload_id`, `start_line=1`, `max_lines=200` | 按 id 回读卸载内容，支持行区间 |
| `WriteFileTool` | `write_file` | `path`, `content`, `append=False` | 写文件（自动建父目录） |
| `EditFileTool` | `edit_file` | `path`, `old_text`, `new_text=""`, `count=1` | 精确字符串替换 |
| `ListFilesTool` | `list_files` | `path="."`, `recursive=False`, `max_entries=200` | 列目录（等价 `ls`） |

#### 路径沙箱：三个文件工具共用

```python
def _resolve_in_workspace(workspace: Path, path: str) -> Optional[Path]:
    """把相对路径解析到 workspace 内；越界返回 None（调用方负责 fail）"""
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        return None
    return target
```

加一个二进制后缀黑名单，拒绝把文本糊进图片：

```python
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".pdf", ".zip", ".whl", ".tar", ".gz", ".so", ".pyd", ".pyc", ".db",
}
```

**为什么限定在 workspace 内**：对齐《演进路线图》F3 的验收标准——「Agent 能读写**工作目录内**的文件」。给模型文件系统能力，是对齐 Anthropic「会话状态外化成文件」的关键一步，但**必须画死边界**，否则模型可以读写任意路径。

#### 写文件：显式 `newline="\n"`

```python
            with target.open(mode, encoding="utf-8", newline="\n") as f:
                f.write(content)
```

**为什么钉死 `newline="\n"`**：Windows 上 Python 默认会把 `\n` 转成 `\r\n`。如果不显式指定，同一个文件在 Windows 与 Linux 下会产生不同字节，`git status` 满屏假 diff（仓库的 `.gitattributes` 正是为此做了「LF 入库」约定）。工具层在这里与仓库约定保持一致。

#### 工厂：按需组装

```python
def build_context_tools(workspace=".", offloader=None, include_filesystem=True) -> List[ToolBase]:
    tools: List[ToolBase] = []
    if offloader is not None:
        tools.append(ReadOffloadedTool(offloader))          # ← 有卸载器才有回读工具
    if include_filesystem:
        tools.extend([ListFilesTool(workspace), WriteFileTool(workspace), EditFileTool(workspace)])
    return tools
```

**为什么 `offloader` 是可选参数**：没有卸载器时，「回读」这个动作没有意义（没有东西可读），所以按需生成。这一条让工厂在两种场景下都成立——只想要文件工具的、和要全套的。

> **你要加新工具时最该看这里**：照抄任一个类的结构（`name` / `description` / `args_schema` / `run()`），再到工厂里挂上。

### 7.8 `react_agent.py` —— 接入：context 已内置于主类

**负责什么**：`_init_context()` / `_fit_context()` / `_offload_observation()` 三个方法，加上 4 个构造参数。**这是 context 真正「生效」的地方**。

#### 初始化：建 builder + 注册工具

```python
        if ctx is None:
            token_budget = TokenBudget(max_context_tokens=budget) if budget else TokenBudget()
            ctx = ContextBuilder(
                workspace=".",
                budget=token_budget,
                counter=TokenCounter(model=getattr(self.llm, "model", None) or "gpt-4o-mini"),
                offloader=ContextOffloader(workspace=".", threshold=offload_threshold),
                llm=self.llm,
            )
        self.ctx = ctx
        for t in build_context_tools(workspace=".", offloader=ctx.offloader):
            self.tool_registry.register_tool(t)
        self.refresh_system_prompt()
```

**为什么注册完工具要 `refresh_system_prompt()`**：system 提示词里含工具清单。新注册了 4 个工具，必须刷新 system，模型才知道它们存在。这是 `ReActAgent` 里 `add_tool` 一致的做法。

#### 重建本轮工作区

```python
        self.ctx.reset_turn()
        window = self.ctx.build(
            system_prompt=system_prompt, history=history,
            user_input=input_text, memory_packets=packets,
        )
        self.messages.clear()
        for message in window.flatten():
            self.messages.append(message)
```

**为什么要 `reset_turn()` 再 `clear()`**：`reset_turn()` 清的是 builder 内部的 `ContextWindow`，`clear()` 清的是 Agent 的 `self.messages`。两处都不能省——否则上一轮的层结构或消息会残留。走完这一步，`self.messages` 就是**一份按预算裁剪过、层级有序的**工作区。

#### 卸载工具结果

```python
        if self.ctx is None or not observation:
            return observation
        try:
            return self.ctx.observe(tool_name, observation)
        except Exception as e:
            print(f"⚠️ 工具结果卸载失败，保留原文：{e}")
            return observation
```

**为什么这里必须 `try/except` 且「失败退回原文」**：卸载是**优化**，不是**功能**。磁盘写满、权限不足、路径异常都可能让卸载失败——但这种时候对话必须继续，绝不能因为「优化失败」而中断工具链。

> **你要改接入逻辑时最该看这里**：三条打印前缀（`🧭`）能帮你确认 context 是否真的在跑。

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\path\to\fast_agents

.\第1步uv_init.bat      # uv init --python 3.10
.\第2步uv_venv.bat      # uv venv --python 3.10
.\第3步uv_add.bat       # uv add -r requirements.txt
```

依赖清单里**不需要为 context 加任何一行**——本模块零新增依赖。想让计数更准，可选装 `tiktoken`：

```powershell
uv pip install tiktoken      # 可选；不装会自动降级为启发式估算
```

> ⚠️ **`test/` 目录与 Python 标准库的 `test` 包同名**。直接 `python -m pytest` 可能引发导入歧义，本仓库的测试文件都用 `python test\test_xxx.py` **直接运行**（文件内部已做 `sys.path.insert`）。跑 context 测试请照此办理。

### 8.2 三十秒跑通（零依赖、不需要真实 LLM）

```python
from fast_agents.context import ContextBuilder, TokenBudget, TokenCounter

builder = ContextBuilder(
    workspace=".",
    budget=TokenBudget(max_context_tokens=32_000, reserve_for_output=4_000),
    counter=TokenCounter(),
)

window = builder.build(
    system_prompt="你是一个严谨的助手。",
    history=[],                                    # 也可以喂 Message 列表
    user_input="用一句话解释上下文工程。",
)
print(builder.stats())
# {'tokens': 24, 'budget': 27000, 'usage_ratio': 0.00089,
#  'layers': ['system', 'user'], 'offloaded_files': 0}

print(window.to_openai_messages())                 # 直接喂 LLM 的格式
```

### 8.3 接到 ReActAgent 上（推荐）

```python
from fast_agents import LLM, ReActAgent
from fast_agents.tools.builtin import ReadFileTool

llm = LLM(model="gpt-4o-mini", api_key="sk-xxx", base_url="https://api.openai.com/v1")

agent = ReActAgent(
    name="上下文助手",
    llm=llm,
    enable_memory=True,          # 记忆提供 history 与 memory 两层的数据
    enable_context=True,         # ← 打开上下文工程
    context_budget=27_000,       # 可选：覆盖默认预算
    context_offload_threshold=4_000,  # 可选：超过 4000 字符的工具结果落盘
)
agent.add_tool(ReadFileTool(workspace="."))

print(agent.run("帮我把 README 里的安装步骤整理成一份清单文件"))
```

开启后，Agent 会打印每轮的上下文账：

```text
🧭 上下文工程已启用：预算 27000 token / 计数后端 tiktoken / 卸载阈值 4000 字符
🤖 上下文助手 开始处理问题: 帮我把 README 里的安装步骤整理成一份清单文件
🧭 token 预算 475/27000（1.8%，窗口 32000，预留输出 4000）
```

### 8.4 典型用法

**用法 1：只想要一个带预算的上下文（不接 Agent）**

```python
window = builder.build(system_prompt="...", history=history, user_input="...")
report = builder.last_report
print(report.describe())         # 逐层 token 明细
```

**用法 2：工具结果自动卸载 + 按需回读**

```python
observation = builder.observe("web_search", 很长的搜索结果)
# → 超阈值时：落盘 .fast_agents/context/offload/<id>.txt，返回带 offload_id 的引用文本
# → 未超阈值：原样返回，零 IO

# 模型想看全文时（也可以直接调 offloader.read）
print(builder.offloader.read(builder.offloader.last_id, start_line=1, max_lines=200))
```

**用法 3：自定义压缩策略（只卸载 + 摘要，不要裁剪）**

```python
from fast_agents.context import ContextBuilder, ContextCompressor, ContextOffloader

offloader = ContextOffloader(workspace=".", threshold=8_000)
compressor = ContextCompressor(
    llm=llm, offloader=offloader,
    enable_offload=True,     # 卸载：信息不丢
    enable_trim=False,       # ← 关掉裁剪：宁可多花一次 LLM 调用，也不丢中间历史
    enable_summary=True,     # 摘要兜底
    keep_recent=6,
)
builder = ContextBuilder(workspace=".", compressor=compressor, llm=llm)
```

**用法 4：离线演示（不接真实 LLM，看完整链路）**

```powershell
python examples\context\context_react_agent_demo.py
```

5 个示例依次演示「分层拼装 → 卸载 → 三级压缩 → 文件系统工具 → 完整 ReAct 循环」，全部离线可跑：

```text
示例 1  分层拼装与 token 预算    看上下文由哪几层组成、各占多少 token
示例 2  超长工具结果卸载          2 万字符 → 落盘 → 按需 read_offloaded 回读
示例 3  三级压缩                  卸载 → 裁剪 → 摘要，逐级降级
示例 4  文件系统工具              write / list / edit，会话状态外化
示例 5  ReActAgent 接入 context   脚本化假 LLM，离线跑通完整 ReAct 循环
```

示例 5 会打印 Agent 实际送给 LLM 的上下文构成，能直接看到「分层拼装 + 预算裁剪」在真实循环里的效果：

```text
Agent 本轮送进 LLM 的上下文构成：
  第 1 次调用：2 条消息 / 约 677 token
    - [system   ] 你是一个具备推理和行动能力的AI助手。...
    - [user     ] 把「上下文工程 = 计量 + 分层 + 治理」这个结论写进 notes/结论.md
  第 2 次调用：4 条消息 / 约 762 token
    - [system   ] ...
    - [user     ] ...
    - [assistant]  (+tool_calls)
    - [tool     ] ✅ 已写入 notes/结论.md（28 字符，文件当前 56 字节）
```

### 8.5 跑测试

```powershell
python test\test_context.py        # 预期：16/16 通过
python test\test_react_agent.py    # 预期：全部通过（含 2 项 context 专项）
```

---

## 九、API 速查

### 9.1 `TokenCounter` / `TokenBudget`（`tokens.py`）

```python
from fast_agents.context import TokenCounter, TokenBudget, estimate_text_tokens, count_message_tokens

counter = TokenCounter(model="gpt-4o-mini", use_tiktoken=True, safety_ratio=1.15)
counter.backend                       # "tiktoken" 或 "heuristic"
counter.count("你好，世界")            # 文本 → token
counter.count_messages(openai_msgs)   # OpenAI 格式消息列表 → token（含请求固定开销）
counter.count_tool_schemas(tools)     # 工具 schema 列表 → token

budget = TokenBudget(max_context_tokens=32_000, reserve_for_output=4_000, safety_margin=1_000)
budget.available                      # 27000
budget.remaining(used)                # 还能放多少
budget.usage_ratio(used)              # 0~1（>1 超预算）
budget.is_over(used)                  # bool
budget.describe(used)                 # 一行日志

estimate_text_tokens("任意文本")       # 纯启发式估算（无需实例）
count_message_tokens({"role": "user", "content": "hi"})   # 单条消息
```

### 9.2 `ContextLayer` / `ContextWindow`（`layers.py`）

```python
from fast_agents.context import ContextLayer, ContextWindow, LAYER_ORDER

window = ContextWindow()
layer = ContextLayer(name="history", priority=60)
layer.add(Message("user", "你好")).add(Message("assistant", "你好，有什么可以帮你？"))
window.add_layer(layer)
window.append_message("user", Message("user", "再问一句"))   # 层不存在会自动建

window.order                          # LAYER_ORDER（dict）
window.ordered_layers()               # 按 order 排序后的层
window.flatten()                      # List[Message]
window.to_openai_messages()           # List[dict]
window.token_count(counter)           # int
window.layer_stats(counter)           # List[dict]
window.validate()                     # List[str]（空列表 = 没有孤立 tool 消息）
```

### 9.3 `ContextOffloader`（`offload.py`）

```python
from fast_agents.context import ContextOffloader

off = ContextOffloader(workspace=".", subdir=".fast_agents/context/offload",
                       threshold=4_000, preview_chars=400, max_files=200)

off.should_offload(text)              # bool
off.offload("web_search", text)       # → 引用文本（未超阈值时原样返回 text，零 IO）
off.last_id                           # 最近一次卸载的 id
off.read(off.last_id, start_line=1, max_lines=200)   # → 带行号的分段文本
off.list_records()                    # 按创建时间倒序
off.get_record(offload_id)            # OffloadRecord 或 None
len(off)                              # 已卸载文件数
off.clear()                           # 清空全部卸载文件，返回删除数
```

### 9.4 `ContextCompressor`（`compressor.py`）

```python
from fast_agents.context import ContextCompressor, DEFAULT_COMPACT_PROMPT

compressor = ContextCompressor(
    llm=llm, offloader=offloader,
    enable_offload=True, enable_trim=True, enable_summary=True,
    keep_recent=4, summary_max_words=200, compact_prompt=None,   # 不传用 DEFAULT_COMPACT_PROMPT
)
report = compressor.compress(layer, counter, budget)   # → CompressionReport
compressor.accumulated_summary                          # 累积摘要（可跨轮注入）
compressor.reset()                                      # 清空累积摘要
```

### 9.5 `ContextBuilder`（`builder.py`）

```python
from fast_agents.context import ContextBuilder, TokenBudget, TokenCounter, ContextOffloader

builder = ContextBuilder(
    workspace=".", budget=None, counter=None, offloader=None, compressor=None,
    llm=None, offload_threshold=4_000, layer_quotas=None, verbose=False,
)

builder.build(system_prompt=..., history=..., user_input=...,
              memory_packets=..., file_snippets=...)   # → ContextWindow
builder.last_report            # BuildReport（本次 build 的报告）
builder.window                 # 当前 ContextWindow
builder.observe("tool", text)  # → 可能被替换成引用文本的观察文本
builder.record_turn(user_input, answer)   # 把本轮写回 history 层（可选）
builder.reset_turn()           # 清空本轮工作区
builder.summary                # 压缩器累积的摘要
builder.stats()                # {'tokens':…, 'budget':…, 'layers':[…], 'offloaded_files':…}
builder.close()                # 释放资源（当前无长连接，留作扩展位）
```

`memory_packets` 接受 `ContextPacket` / `dict` / `str` 混合列表，内部由 `_render_packets()` 统一渲染。

### 9.6 context 工具与工厂（`tools.py`）

```python
from fast_agents.context import (ReadOffloadedTool, WriteFileTool, EditFileTool,
                                 ListFilesTool, build_context_tools)

tools = build_context_tools(workspace=".", offloader=off, include_filesystem=True)
# → [ReadOffloadedTool, ListFilesTool, WriteFileTool, EditFileTool]
for t in tools:
    agent.add_tool(t)

# 也可以单独构造
ReadOffloadedTool(off)          # name = "read_offloaded"
WriteFileTool("D:/code")        # name = "write_file"
EditFileTool("D:/code")         # name = "edit_file"
ListFilesTool("D:/code")        # name = "list_files"
```

### 9.7 `ReActAgent` 侧参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `ctx` | `None` | 注入自定义 `ContextBuilder`；不传且 `enable_context=True` 时自动创建 |
| `enable_context` | `False` | 是否启用上下文工程 |
| `context_budget` | `None` | 覆盖预算（int token 数）；不传用 `TokenBudget()` 默认 128k |
| `context_offload_threshold` | `4_000` | 触发卸载的字符数阈值 |

> **已废弃写法**：早期版本构造参数叫 `budget` / `offload_threshold`（与 `ReActAgent` 里其他参数命名打架）。现在统一为 `context_budget` / `context_offload_threshold`，语义更清楚。旧写法不再支持，请改过来。

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

**① 为什么先做 `TokenCounter`，而不是先做压缩？**

因为「裁不裁」的前提是「知道有多少」。没有 token 计数，压缩就没有**触发信号**——你只知道消息条数超了，但不知道超的是 token 还是条数。而且计数是最小可用的地基：`TokenCounter` 单独拿出来就能用（算成本、做日志），压缩则必须建在它之上。

**② 为什么「卸载优先」而不是「摘要优先」？**

因为**卸载信息不丢，摘要信息有损**，而摘要还多花一次 LLM 调用（可能失败、可能限流）。能靠落盘解决的问题，不值得用有损手段。LangChain / LlamaIndex 的主流做法是「裁剪 + 摘要」，本实现刻意把「卸载」插到最前面——这是本模块与它们最大的设计分叉。

**③ 为什么要做成「默认关闭」？**

因为 context 会**改变每轮送给 LLM 的上下文构成**，属于影响行为的改动。默认关闭 → 老代码零改动、行为逐字节不变；想用的人显式打开。这与 `enable_mcp` 默认 `False` 是同一个理由：**影响面大的能力不该默认开**。

**④ 为什么零依赖 + 启发式降级，而不是硬依赖 tiktoken？**

`tiktoken` 是 C 扩展，装不上（网络受限、平台不匹配）的平台不少。硬依赖它，等于把「框架能不能跑」绑死在一个增强功能上。所以本实现让它是**可选增强**：装了更准，没装也能跑。**「能用」优先于「精准」**——这与模块整体定位一致（仓库招牌是「零重型依赖」）。

**⑤ 为什么压缩只对 `history` 做摘要？**

`memory` 层和 `files` 层走的是「只卸载 + 丢最老条目」（`_fit_layer`），不做摘要。因为这两层本身就是**可选参考**——长期记忆和文件片段丢了，模型仍能靠 history 与 system 完成任务；而为它们各加一次 LLM 摘要调用，性价比太低。**把最贵的资源（LLM 调用）花在最不可替代的层（history）上。**

**⑥ 为什么把顺序做成 `LAYER_ORDER` 常量，而不是 `build()` 里的 if-else？**

因为顺序一旦是数据，**裁剪器就不需要知道谁是 system**——它只看 `required` 和 `priority`。要加层只改表，要调优先级只改数字。写成 if-else 则每加一层都要改多处判断，迟早写漏。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| 日志显示「计数后端 heuristic」，担心不准 | 没用上 tiktoken（没装 / 加载失败）。功能正常，只是估算有 ±15% 误差。装 `tiktoken` 即可切到 `tiktoken` 后端 |
| 开了 `enable_context=True`，但 `layers` 只有 `['system','user']` | `history` 为空。检查是否同时 `enable_memory=True`（history 来自 `memory.buffer`）；无记忆时自然只有 system + user |
| 卸载文件落在了错误的位置 | 默认 `workspace="."`，即**进程启动目录**。要固定路径，用 `ContextBuilder(workspace="D:/code")` 显式构造后经 `ctx=` 注入 |
| 模型回读时提示「未找到 offload_id」 | 该卸载文件已被淘汰（`max_files` 默认 200，按最早创建淘汰）。调大 `max_files`，或缩短对话轮数 |
| 摘要级似乎没生效 | 摘要需要 `llm`。`ContextCompressor` 没拿到 `llm` 时，第 3 级会失败并降级为截断（会打印「上下文摘要失败，降级为截断」） |
| 上下文里出现两条相同的 user 消息 | 接入边界被破坏。检查是否在 `_fit_context()` 之外又 `append_user()`；回归断言见 `test_context_user_input_not_duplicated` |
| 请求报 400「tool 消息缺少配对」 | 裁剪时把 `assistant(tool_calls)` 与 `tool` 拆开了。裁剪必须走 `_group_messages()` 成组处理；可用 `window.validate()` 提前自检 |
| 注册工具时打印「工具 'write_file' 已存在，将被覆盖」 | Agent 里已有同名工具（如你自己注册过）。`register_tool` 是覆盖语义，确认是否要覆盖 |
| 模型注册了工具却不用 `read_offloaded` | 工具注册后没刷新 system 提示词，模型看不到它。`_init_context()` 里已调 `refresh_system_prompt()`；手工注册时要自己调 |
| Windows 下写文件产生满屏 `git diff` | 写文件时未指定换行。本模块 `WriteFileTool` 已钉死 `newline="\n"`；自写文件工具要照做 |
| 中文内容出现乱码方框 | 读写文件未指定编码。统一 `encoding="utf-8"`，读不可信文件时加 `errors="replace"` |
| `.fast_agents/` 目录被 git 跟踪 | 卸载目录在 workspace 内。建议在 `.gitignore` 加 `.fast_agents/` |
| `layer_quotas` 比例之和不为 1，是否报错 | 不报错。比例之和不必为 1，多余的视为弹性空间；`history` 拿的是剩余额度 |
| 层配额都合规，整窗仍超预算 | 这是「局部合规、整体超标」，由 `_enforce_global_budget()` 兜底（成组丢最老 history），会打印告警 |
| `_seed_workspace()` 与 `_fit_context()` 都跑了，是否重复 | 不重复。`_fit_context()` 用 `reset_turn()` + `clear()` 重建了工作区，会覆盖 seed 的结果。seed 在开启 context 时属于冗余但无害的步骤 |
| 卸载后模型「失忆」，不再提原文内容 | 引用文本里已教会模型用 `read_offloaded`。若模型不会用，可在 system 提示词里补一句「工具结果提示已卸载时，需要全文请调用 read_offloaded」 |
| 技能纠错前缀被卸载吞掉 | 顺序问题。`_offload_observation()` 必须在 `skill_middleware.after_tool_call()` **之前**执行，保证纠错前缀不被整段卸载 |

### 10.3 跨平台注意事项（Windows 专项）

本模块涉及文件读写与路径拼接，**在 Windows 上有实际差异**，逐条给可执行解法：

1. **路径分隔符**：全部用 `pathlib.Path`，**不要手工拼 `"\\"` 或 `"/"`**。`_resolve_in_workspace()` 用 `(workspace / path).resolve()`，跨平台一致；
2. **换行符**：`WriteFileTool` 写文件时显式 `newline="\n"`，避免 Windows 默认转 `\r\n` 造成跨平台字节差异。仓库 `.gitattributes` 已声明「`.py`/`.md`/`.txt`/`.bat` 入库 LF、Windows 检出 CRLF」，工具层与之一致；
3. **编码**：所有文件读写显式 `encoding="utf-8"`；读用户/模型可能给出的任意文件时用 `errors="replace"`，避免一个坏字节导致整次读取抛异常；
4. **卸载目录位置**：默认 `workspace="."` 是**进程启动目录**。在 Windows 上用 IDE 运行时启动目录常是项目根或 IDE 安装目录，务必确认；建议显式指定 `workspace`；
5. **路径空格**：`workspace` 含空格（如 `D:\My Code\...`）时，`pathlib` 本身没问题，但**不要把未加引号的路径拼进 shell 命令**；本模块不调 shell，风险主要在你的自定义工具里；
6. **脚本解释器**：示例与测试都用 `python test\test_xxx.py` 直接跑（不依赖 pytest），`sys.path.insert` 已处理导入路径，与平台无关。

---

## 十一、测试与验证

### 11.1 测试清单

| 测试文件 | 项数 | 覆盖点 | 运行命令 |
|---|---|---|---|
| `test/test_context.py` | 16/16 | 见下 | `python test\test_context.py` |
| `test/test_react_agent.py` | 全绿 | 含 2 项 context 专项 | `python test\test_react_agent.py` |

`test/test_context.py` 覆盖点（不依赖真实 LLM、不依赖网络）：

1. `TokenCounter` 的启发式估算与消息计数（`test_token_counter`）；
2. `TokenBudget` 的预算换算与描述（`test_token_budget`）；
3. `ContextWindow` 的分层顺序与 `flatten`（`test_window_layer_order`）；
4. `ContextWindow` 的 tool 配对校验（`test_window_validate_tool_pairing`）；
5. `ContextOffloader` 的阈值判断（不足阈值不落盘）（`test_offloader_threshold`）；
6. `ContextOffloader` 落盘 + 索引 + 回读（`test_offloader_offload_and_read`）；
7. `ContextOffloader` 的行区间回读（`test_offloader_read_line_range`）；
8. `ContextOffloader` 的文件数上限淘汰（`test_offloader_max_files_eviction`）；
9. `ContextOffloader` 的非法 id / 路径安全（`test_offloader_safety`）；
10. `ContextCompressor` 第 2 级：裁剪最老历史且不拆 tool 配对（`test_compressor_trim_keeps_tool_pairing`）；
11. `ContextCompressor` 第 1 级：超长消息被卸载成引用（`test_compressor_offload_level`）；
12. `ContextCompressor` 第 3 级：LLM 摘要替换历史（`test_compressor_summary_level`）；
13. `ContextBuilder` 的完整分层拼装顺序（`test_builder_pipeline_and_order`）；
14. `ContextBuilder` 的预算裁剪（历史被压掉）（`test_builder_budget_trims`）；
15. context 工具：write / edit / list（`test_context_filesystem_tools`）；
16. context 工具：read_offloaded 与工厂（`test_read_offloaded_tool_and_factory`）。

> 文件头注释把上面的覆盖点归纳成 **15 条编号**（卸载相关的 5 个测试合并在编号 5–9 一条里），实际测试函数是 **16 个**。以 `main()` 里的清单为准。

### 11.2 关键回归断言

以下断言锁死了设计里最容易被后续改动破坏的约定，**改动 context 或 Agent 时必须保持绿**：

| 断言 | 锁住了什么设计 |
|---|---|
| `test_compressor_trim_keeps_tool_pairing` | **tool 配对绝不拆散**：裁剪后不允许出现孤立的 `tool` 消息 |
| `test_window_validate_tool_pairing` | `ContextWindow.validate()` 能正确识别孤立 `tool` 消息 |
| `test_offloader_safety` | **路径安全**：非法 id 被拒、不存在的 id 返回可读错误而非抛异常、卸载目录在 workspace 内 |
| `test_offloader_max_files_eviction` | **有界增长**：超过 `max_files` 按最早创建淘汰 |
| `test_builder_pipeline_and_order` | **五层拼装顺序**：system 在最前、user 在最后 |
| `test_react_agent.py::test_context_user_input_not_duplicated` | 接入边界一：**本轮 user 输入只注入一次**，不随工具步数累积 |
| `test_react_agent.py::test_context_enabled_builds_and_registers_tools` | 接入边界二：开启 context 后 4 个工具确实注册、`agent.ctx` 非空 |

其中 `test_context_user_input_not_duplicated` 是本模块的**头号回归断言**——它锁住「单一数据源」不被后续改动破坏（与记忆模块的 `test_no_duplicate_history_injection` 同一思路）。

### 11.3 全绿基线与手工验证

**全绿基线**（实测结果）：

```text
$ python test\test_context.py
...
✅ context 模块测试全部通过：16/16 项

$ python test\test_react_agent.py
...
🎉 全部测试通过

$ python examples\context\context_react_agent_demo.py
...
全部示例执行完毕        (exit 0)
```

**手工验证清单**（改完代码后自己过一遍）：

- [ ] `python -c "from fast_agents.context import ContextBuilder, TokenCounter, ContextOffloader, ContextCompressor"` 无报错；
- [ ] 开启 `enable_context=True` 后，日志出现「🧭 上下文工程已启用：…」；
- [ ] 同一轮里 `layers` 顺序为 `['system', 'history'/'memory', 'user']`（system 在前、user 在最后）；
- [ ] 造一个超长工具结果，确认落盘到 `.fast_agents/context/offload/` 且 `index.json` 有记录；
- [ ] 用 `read_offloaded` 能回读到内容，且行区间参数生效；
- [ ] 多工具步场景下，送给 LLM 的 `user` 消息只有一条。

> **测试与实现是否同步**：本模块测试与实现**同步**，16/16 全绿，无红着的测试。若你后续改了裁剪逻辑，请同步更新 `test_compressor_trim_keeps_tool_pairing` 的断言，不要直接删断言。

---

## 十二、下一步演进路线

按优先级排（编号与 [`docs/演进路线图.md`](./演进路线图.md) 模块 F / B / E 保持一致）：

1. **[F4] 压缩中间件化** —— 现在三级策略是 `ContextCompressor` 的方法调用，可开关但不能插拔。目标是抽成与 `skills` 的 `SkillMiddleware` 同构的中间件接口（`before_build` / `after_build`），让「专用摘要器」「按主题裁剪器」都能挂上来。**现在不做的原因**：单模块内部调用已够用，抽中间件要先在多个 Agent 上验证接口稳定性；
2. **[F5] Prompt caching 断点显式化** —— 顺序已就位（静态在前），但没打 `cache_control`。落点很小：给 `ContextLayer` 加 `cache_breakpoint: bool`，`flatten()` 时对该层最后一条消息打标即可，**不需要动压缩和裁剪逻辑**。现在不做是因为 OpenAI 侧缓存是自动的，只有接 Anthropic 才需要显式断点；
3. **[F1+] 精确 tokenizer 接入** —— 现在是「tiktoken → 通用编码 → 启发式」三级降级，估算有 ±15% 误差。可增加 provider 级精确计数（如 Anthropic 官方计数接口）作为最高优先级的一级；
4. **[B2] 语义召回层** —— 目前 memory 层只注入「长期记忆 + 会话摘要」，没有向量召回。`ContextPacket.metadata` 已留扩展位。依赖路线图 B2（语义检索）与 `llm.embed()`；
5. **[F2+] 流式截断** —— 现在卸载发生在「工具结果已完整返回」之后，对**流式输出过程中**的超长内容无能为力。可加一个「边流边算 token、超阈值即截断」的中间层；
6. **[F2+] 卸载文件的生命周期管理** —— 现在只有 `max_files` 数量的粗粒度淘汰，没有按时间 / 按会话的清理。长跑服务需要「会话结束时清掉本会话的卸载文件」；
7. **[B7] 双通道注入** —— 记忆层目前是「检索到就注入」。可改为「命中的注入正文、没命中的只注入标题」，进一步压 token；
8. **[E3] 子代理上下文隔离** —— 用独立上下文窗口跑子任务，只把结果返回主 Agent。这属于 `agents` 模块，但它能从根上降低单 Agent 的上下文压力，是长任务的关键能力。

> **当前最明显的空缺**：F4（压缩中间件化）与 F5（缓存断点）都只铺了地基。**能跑、能用、能开关，但还不能像 LangGraph 那样把压缩当成一个可编排的图节点。**

---

## 十三、参考

**仓库内**

- `fast_agents/context/`（7 个文件）—— 本模块全部实现；
- `fast_agents/messages/message.py` / `messages.py`（`Message` / `Messages` / `to_openai_message`）；
- `fast_agents/tools/tool_base.py` / `tool_result.py`（`ToolBase` / `ToolResult` / `ToolStatus`）；
- `fast_agents/memory/manager.py` / `buffer.py`（`MemoryManager` / `BufferMemory` / `ContextPacket`）；
- `fast_agents/agents/react_agent/react_agent.py`（`_init_context` / `_fit_context` / `_offload_observation` / `_seed_workspace`）；
- `examples/context/context_react_agent_demo.py`、`test/test_context.py`、`test/test_react_agent.py`；
- `docs/演进路线图.md`（模块 F / B / E 系列）、`docs/记忆模块编写说明.md`（写法范本）。

**外部框架与标准**

- [LangChain `trim_messages`](https://python.langchain.com/docs/how_to/trim_messages/) —— 借鉴「按 token 值裁剪 + 保留最近若干组」；舍弃其重型 Runnable 抽象；
- [LangGraph `SummarizationNode`](https://langchain-ai.github.io/langgraph/) —— 借鉴「压缩作为图节点、摘要写回 state」；本实现先以方法调用落地；
- [LlamaIndex `ChatMemoryBuffer` / Context](https://docs.llamaindex.ai/) —— 借鉴「门面 + 让检索结果参与预算排队」；舍弃其索引 / 召回引擎；
- [CrewAI](https://github.com/crewAIInc/crewAI) —— 借鉴「上下文隔离也是减压手段」的意识；子代理隔离列入 E3；
- [Anthropic · Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 借鉴「compaction + tool result 截断 + 文件回读 + 静态在前」的实践；
- [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) —— 借鉴「会话状态外化成文件」「记忆即工具」；不加其依赖。

---

## 附录：发布前自检清单

改动 `fast_agents/context/` 之后，逐条打勾：

- [ ] `python -c "from fast_agents.context import ContextBuilder, ContextWindow, ContextOffloader, ContextCompressor, TokenCounter, TokenBudget, build_context_tools"` 无报错；
- [ ] `fast_agents/context/__init__.py` 的 `__all__` 与新增导出保持一致；
- [ ] `fast_agents/__init__.py` 已同步追加新的公开符号；
- [ ] 新增层的名字已加进 `LAYER_ORDER`，并在 `build()` 里就位；
- [ ] 裁剪仍走 `_group_messages()` 成组处理（tool 配对未被拆散）；
- [ ] `ContextWindow.validate()` 对合法上下文返回空列表；
- [ ] 卸载文件仍落在 `workspace` 内（路径穿越仍被拒）；
- [ ] `offload()` 在两处保留「失败退回原文 / 截断」的兜底；
- [ ] 摘要路径仍保留「LLM 失败 → 截断」的双层降级；
- [ ] 新增/改名工具时同步 `build_context_tools()` 与本文 [9.6](#96-context-工具与工厂toolspy)；
- [ ] `ReActAgent` 侧的两条接入边界仍被 `test_react_agent.py` 的两条断言覆盖；
- [ ] 所有 `.py` / `.md` / `.txt` / `.bat` 为 CRLF（或由 `.gitattributes` 统一），`.json` / `.toml` 为 LF；
- [ ] `python test\test_context.py` 与 `python test\test_react_agent.py` 全绿；
- [ ] 本文档 [六章](#六核心数据模型)、[九章](#九api-速查) 已随代码同步；
- [ ] 本文档 [一、更新历史](#一更新历史) 加了一行；
- [ ] 演进路线相关内容与 [`docs/演进路线图.md`](./演进路线图.md) 编号一致，两处不打架；
- [ ] 全文无「待补充 / TBD」残留。

> **一句话总结本模块的设计**：上下文工程 = **计量（知道有多少）+ 分层（知道放哪、先砍谁）+ 治理（超了怎么办）**。三者各占一个文件、各解决一个问题，拼起来就是 `ContextBuilder` 的一次 `build()`——**没有魔法，全是账。**

