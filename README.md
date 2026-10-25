# FastAgents

> 简单、灵活、可扩展的多智能体框架 —— 基于 OpenAI 原生 API 构建

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="version" src="https://img.shields.io/badge/version-0.0.1-green">
<img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

FastAgents 是一个从零手写的轻量 Agent 框架，**不依赖 LangChain 等重型编排库**，直接基于 OpenAI 原生 API 实现。
目前包含四大能力：**LLM 客户端、消息系统、工具系统、ReAct Agent**，并内置了**完整的记忆系统（短期 + 长期）**。

代码全部可用、有测试覆盖，适合作为「理解 Agent 原理」的学习范本，也能直接用于中小型项目。

---

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心模块](#核心模块)
  - [LLM](#1-llm客户端层)
  - [Messages](#2-messages消息系统)
  - [Tools](#3-tools工具系统)
  - [ReActAgent](#4-reactagent智能体)
  - [Memory](#5-memory记忆系统)
- [使用示例](#使用示例)
- [配置](#配置)
- [测试](#测试)
- [设计取舍](#设计取舍)
- [路线图](#路线图)

---

## 特性

| 能力 | 说明 |
|---|---|
| 🧠 **LLM 客户端** | 对 OpenAI SDK 的轻封装，支持 `invoke` / `stream_invoke`，兼容任意 OpenAI 格式的第三方服务（只需改 `base_url`） |
| 💬 **消息系统** | `Message` / `Messages` 双层结构，原生支持 function calling（`tool_calls` / `tool_call_id`），可一键转 OpenAI 格式 |
| 🔧 **工具系统** | `ToolBase` 抽象基类 + `ToolRegistry` 注册表，用 pydantic 定义参数即可自动生成 OpenAI function schema |
| 🤖 **ReAct Agent** | 经典「推理 → 行动 → 观察」循环，支持多步工具调用，自动管理工具链消息 |
| 🧩 **记忆系统** | 短期（滑动窗口 + 滚动摘要）+ 长期（SQLite 持久化，跨会话）+ 偏好无条件召回，支持 Agent 主动读写 |
| 📦 **零重型依赖** | 记忆模块只用 Python 标准库（`sqlite3` / `json` / `pathlib`），不引入向量库或 ORM |

---

## 安装

```bash
git clone https://github.com/LittleSheepy/fast_agents.git
cd fast_agents
pip install -r requirements.txt
```

> **核心运行依赖**其实只有 `openai` + `pydantic` + `python-dotenv`；
> `requirements.txt` 中的其余包（tavily / agentscope / sqlalchemy 等）是为后续的搜索工具、多智能体协作预留的。

**环境要求**：Python 3.10+

---

## 快速开始

最快的上手方式——创建一个能算数的 ReAct Agent：

```python
from fast_agents import LLM, ReActAgent, ToolRegistry
from fast_agents.tools.calculator_tool import CalculatorTool

# 1. 创建 LLM（读取环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID）
llm = LLM()

# 2. 注册工具
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

# 3. 创建 Agent 并运行
agent = ReActAgent("助手", llm=llm, tool_registry=registry)
print(agent.run("帮我算一下 (125 * 8 + 60) / 4 等于多少？"))
```

Agent 会自动完成：分析问题 → 调用 `python_calculator` → 拿到结果 → 给出最终答案。

---

## 项目结构

```
fast_agents/
├── __init__.py                     # 顶层导出：LLM / ReActAgent / ToolRegistry
├── version.py                      # 版本信息
│
├── llm/                            # LLM 客户端层
│   ├── llm.py                      # LLM 统一入口（读环境变量、选择 provider）
│   └── openai_client.py            # OpenAI SDK 封装
│
├── messages/                       # 消息系统
│   ├── message.py                  # 单条消息 Message（含 tool_calls）
│   └── messages.py                 # 消息列表 Messages
│
├── tools/                          # 工具系统
│   ├── tool_base.py                # 工具基类 ToolBase（自动生成 OpenAI schema）
│   ├── tool_registry.py            # 工具注册表 ToolRegistry + global_registry
│   ├── calculator_tool.py          # 内置：计算器工具
│   └── terminal_tool.py            # 内置：命令行工具
│
├── agents/                         # Agent 范式
│   ├── agent_base.py               # Agent 抽象基类
│   └── react_agent/
│       ├── react_agent.py          # ⭐ ReActAgent（工具调用 + 记忆，唯一实现）
│       └── prompt.py               # 默认 ReAct 提示词模板
│
└── memory/                         # ⭐ 记忆系统（9 个文件）
    ├── schema.py                   # 数据模型：MemoryRecord / ContextPacket
    ├── base.py                     # 存储后端抽象接口 MemoryStore（ABC）
    ├── scoring.py                  # 相关性打分：词法 + 时间衰减 + 综合得分
    ├── buffer.py                   # 短期记忆：滑动窗口 + 滚动摘要
    ├── in_memory_store.py          # 长期记忆后端①：纯内存（调试用）
    ├── sqlite_store.py             # 长期记忆后端②：SQLite 持久化（生产用）
    ├── manager.py                  # 门面：MemoryManager 统一编排
    └── tool.py                     # 记忆工具：remember_memory / recall_memory / forget_memory

test/
├── test_tool.py                    # 工具系统测试
├── test_memory.py                  # 记忆模块测试（11 项）
├── test_react_agent.py             # ReActAgent 集成测试（9 项）
└── demo_memory.py                  # 端到端演示：跨会话记忆
```

---

## 核心模块

### 1. LLM（客户端层）

对 OpenAI SDK 的最小封装，支持从环境变量自动读取配置：

```python
from fast_agents import LLM

llm = LLM()                                   # 读环境变量
llm = LLM(model="gpt-4o", api_key="sk-...",   # 或显式传入
          base_url="https://api.openai.com/v1")

resp = llm.invoke([{"role": "user", "content": "你好"}])
print(resp.content)
```

| 环境变量 | 含义 |
|---|---|
| `LLM_API_KEY` | API 密钥 |
| `LLM_BASE_URL` | 接口地址（兼容任意 OpenAI 格式的服务） |
| `LLM_MODEL_ID` | 默认模型名 |

### 2. Messages（消息系统）

`Message` 是单条消息，支持四种角色（`user` / `assistant` / `system` / `tool`）和工具调用字段：

```python
from fast_agents.messages.message import Message, MsgToolCall

msg = Message("assistant", "我来算一下",
              tool_calls=[MsgToolCall(name="calc", arguments='{"x":1}', id="call_1")])
print(msg.to_openai_message())
```

`Messages` 是消息列表容器，提供 `append_system` / `append_user` / `append_ai` 等便捷方法：

```python
from fast_agents.messages import Messages

msgs = Messages()
msgs.append_system("你是一个助手")
msgs.append_user("你好")
print(msgs.to_openai_messages())   # 直接喂给 LLM
```

### 3. Tools（工具系统）

自定义工具只需继承 `ToolBase`，用 pydantic 定义参数模型，框架会自动生成 OpenAI function calling schema：

```python
from pydantic import BaseModel, Field
from fast_agents.tools.tool_base import ToolBase, ToolParameter


class WeatherParams(BaseModel):
    city: str = Field(description="城市名，例如：上海")


class WeatherTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="查询指定城市的天气",
            param_model=WeatherParams,   # ← 参数 schema 由此自动生成
        )

    def run(self, city: str) -> str:
        return f"{city}：晴，25℃"

    def get_parameters(self):
        return [ToolParameter(name="city", type="string",
                              description="城市名", required=True)]
```

注册与使用：

```python
from fast_agents import ToolRegistry
registry = ToolRegistry()
registry.register_tool(WeatherTool())
registry.get_tools_description()      # "- get_weather: 查询指定城市的天气"
registry.execute_tool("get_weather", '{"city": "上海"}')
```

> 小技巧：两个工具可以用 `|` 运算符直接拼成 OpenAI tools 列表：`calc_tool | weather_tool`

### 4. ReActAgent（智能体）

`ReActAgent` 是框架的核心，**同时具备工具调用与记忆能力**（记忆已内置，无需额外子类）。

```python
ReActAgent(
    name="助手",              # 位置参数 1：名称
    llm=llm,                  # 位置参数 2：LLM 实例
    tool_registry=registry,   # 工具注册表（可选）
    system_prompt=None,       # 系统提示词（可选）
    max_steps=10,             # 最大推理步数
    custom_prompt=None,       # 自定义 ReAct 提示词模板
    memory=None,              # 记忆管理器；不传则用纯内存默认实例
    enable_memory=True,       # 默认开启记忆；传 False 退回纯 ReAct
    memory_limit=5,           # 每次注入的长期记忆条数上限
)
```

它的运行循环（`run()`）分四段：

```
① _seed_workspace()              # 重建工作区 = system + buffer 历史
② _retrieve_long_term_memory()   # 检索长期记忆 → 拼进本轮 user 输入
③ ReAct 循环                     # 推理 → 调工具 → 观察 → 再推理 …（tool 链只活在工作区）
④ _remember()                    # 本轮对话沉淀回 buffer（可选抽取长期记忆）
```

四条关键设计约定：

1. **单一数据源**——跨轮对话历史只存在 `MemoryManager.buffer` 里，`self.messages` 只是本次 run 的「工作区」，run 结束即重建，避免历史重复注入；
2. **system 只注入一次**——在 `__init__` 里构建，循环内不重建，system 消息不会越滚越多；
3. **tool 链不进 buffer**——`tool_calls` / `tool` 结果只活在工作区，避免 buffer 裁剪时拆坏 tool 配对（OpenAI 强校验会直接 400）；
4. **记忆是参考信息**——长期记忆随本轮用户输入一起送进 LLM，不提升为系统指令（防 prompt injection）。

### 5. Memory（记忆系统）

记忆分两层，概念上对应「工作记忆」与「长期记忆」：

| 层级 | 类 | 生命周期 | 存储 | 内容 |
|---|---|---|---|---|
| **短期记忆** | `BufferMemory` | 单次会话 | 内存 | 最近 N 轮对话 + 滚动摘要 |
| **长期记忆** | `MemoryStore` 实现 | 跨会话 | SQLite / 内存 | 偏好、事实、情景、程序 |

**① 短期记忆——`BufferMemory`**

滑动窗口 + 滚动摘要：超出窗口的老消息交给 LLM 压成一句摘要，下次一起注入，长对话不再爆 token。

```python
from fast_agents.memory import BufferMemory

buf = BufferMemory(max_messages=20, enable_summary=True, llm=llm)
buf.append_user("我叫小明")
buf.append_ai("你好，小明")
buf.get_messages()            # 窗口内的消息
buf.get_summary()             # 被挤出去内容的摘要
```

**② 长期记忆——`SQLiteMemoryStore`（零依赖持久化）**

```python
from fast_agents.memory import SQLiteMemoryStore, MemoryRecord

store = SQLiteMemoryStore("data/memory.db", user_id="u1", scope="user")
store.add(MemoryRecord(content="用户偏好用中文回答", kind="preference", importance=0.8))
```

两个核心机制：

- **作用域隔离**：`scope="user"` 按用户隔离（同一用户换会话仍记得）；`scope="session"` 按 `user_id + session_id` 双重隔离。所有查询统一走 `_scope_clause()`，杜绝串数据。
- **修订而非覆盖**：`revise()` 把旧记录标为 `superseded`、新记录指向旧 id，历史永不丢失；删除分 `retract()`（留历史）和真删。

```python
old = store.add(MemoryRecord(content="用户住在杭州", kind="fact"))
new = store.revise(old.memory_id, "用户搬到了北京", "user_stated")

new.supersedes                    # == old.memory_id（新版指向旧版）
store.all()                       # 只剩「用户搬到了北京」
store.all(include_inactive=True)  # 历史仍可查，旧记录 status == "superseded"
```

**③ 统一门面——`MemoryManager`**

把短期和长期粘在一起，对外只暴露几个动作：

```python
from fast_agents.memory import MemoryManager, SQLiteMemoryStore

store = SQLiteMemoryStore("data/memory.db", user_id="user_001", scope="user")
mgr = MemoryManager(
    long_term_store=store,
    llm=llm,
    user_id="user_001",      # ⚠️ 必须与 store 的 user_id 一致（见下方提示）
    auto_extract=True,
)

mgr.retrieve("用户偏好什么", limit=5)    # 检索长期记忆
mgr.render_context("问题")               # 检索并渲染成可注入文本
mgr.record(user_input, assistant_output) # 记录一轮对话
mgr.remember("用户喜欢 Python", kind="preference")  # 手动写
mgr.forget(memory_id)                    # 撤回（保留历史）
```

> ⚠️ **注意作用域一致性**：`MemoryManager` 与 `SQLiteMemoryStore` 的 `user_id` / `session_id` **必须相同**。
> 记录落库时用 manager 的值，查询过滤时用 store 的 scope 条件；两者不一致会导致「写进去了却查不出来」。
> 推荐做法：只在一处定义 `user_id`，同时传给两边。

检索用「词法重叠 + 重要性 + 时间衰减」综合打分，不需要向量库：

```
最终得分 = 词法相关度 × 0.6 + 重要性 × 0.3 + 时间衰减 × 0.1
```

> **一个关键细节**：纯词法检索会漏掉「用户偏好中文回答」这类偏好——它和问题字面上无关，但对答案极其重要。
> 所以 `search()` 提供 `include_preferences=True`，**偏好类记忆无条件进候选**。

**④ 记忆工具——让 Agent 主动读写**

```python
from fast_agents.memory import build_memory_tools

for t in build_memory_tools(mgr):        # 一次拿到 remember_memory / recall_memory / forget_memory 三个工具
    registry.register_tool(t)
```

---

## 使用示例

### 示例 1：纯短期记忆

```python
from fast_agents.memory import BufferMemory

buf = BufferMemory(max_messages=10, enable_summary=True, llm=llm)
buf.append_user("你好")
buf.append_ai("你好，有什么可以帮你？")
messages = buf.to_openai_messages()   # 直接喂给 LLM
```

### 示例 2：跨会话长期记忆

```python
from fast_agents.memory import MemoryManager, SQLiteMemoryStore

mgr = MemoryManager(
    long_term_store=SQLiteMemoryStore("data/memory.db", user_id="user_001", scope="user"),
    llm=llm, user_id="user_001", auto_extract=True,
)

# 会话 1
mgr.record("我是做后端的，平时用 Go", "了解了")

# 会话 2（换个进程照样能想起）
print(mgr.render_context("帮我推荐点学习资料"))
```

### 示例 3：Agent + 工具 + 记忆（完整形态）

```python
from fast_agents import LLM, ReActAgent, ToolRegistry
from fast_agents.memory import MemoryManager, SQLiteMemoryStore, build_memory_tools
from fast_agents.tools.calculator_tool import CalculatorTool

llm = LLM()

# 记忆管理器（user_id 与 store 保持一致）
mgr = MemoryManager(
    long_term_store=SQLiteMemoryStore("data/memory.db", user_id="user_001", scope="user"),
    llm=llm, user_id="user_001", auto_extract=True,
)

# 工具注册表：计算器 + 记忆工具
registry = ToolRegistry()
registry.register_tool(CalculatorTool())
for t in build_memory_tools(mgr):
    registry.register_tool(t)

# 记忆已内置于 ReActAgent
agent = ReActAgent("助手", llm=llm, tool_registry=registry, memory=mgr, memory_limit=5)

agent.run("记一下，我下周要去上海出差")
print(agent.run("我下周去哪来着？"))     # Agent 会自己调 recall_memory 工具
```

### 示例 4：运行中动态加工具

```python
from fast_agents.tools.calculator_tool import CalculatorTool

agent = ReActAgent("助手", llm=llm)
agent.add_tool(CalculatorTool())    # 加完后提示词里的工具清单会自动刷新
agent.run("算一下 123 * 456")
```

---

## 配置

在项目根目录（或上层任意目录）创建 `.env`：

```env
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini
```

框架启动时会自动加载。若你的 `.env` 不在默认查找路径，可手动指定：

```python
from dotenv import load_dotenv
load_dotenv("/your/path/.env")
```

> ⚠️ 注意：请勿在代码里写死绝对路径（如 `load_dotenv("D:/xxx/.env")`），否则换台机器或别人 clone 后会直接报错。用 `load_dotenv()` 让它在当前目录和上层自动查找即可。

---

## 测试

仓库自带三个可运行文件，**不依赖真实 LLM 也能跑**（用脚本化的 FakeLLM）：

```bash
python test/test_tool.py         # 工具系统
python test/test_memory.py       # 记忆模块（11 项）
python test/test_react_agent.py  # ReActAgent 集成（9 项）
python test/demo_memory.py       # 端到端：跨会话记忆 + 记忆修订
```

---

## 设计取舍

| 决策点 | 选择 | 理由 |
|---|---|---|
| 底层 SDK | OpenAI 原生 API | 不引入 LangChain，逻辑透明、可控性强 |
| 短期记忆 | 内存滑动窗口 + LLM 滚动摘要 | 比纯 TTL 更贴合 ReAct 场景 |
| 长期存储 | SQLite（标准库） | 零依赖，和项目轻量风格一致 |
| 检索方式 | 词法 + 重要性 + 时间衰减 | 先跑通闭环；接口已留好可换 embedding |
| 作用域 | `scope="user"` / `"session"` | 长期记忆默认跨会话共享 |
| 写入策略 | 修订保留历史（supersede） | 审计友好，历史可追溯 |
| 记忆注入位置 | 长期记忆随本轮 user 输入 | 防注入 + 缓存友好 |
| 对 Agent 暴露 | 自动注入 + 三个记忆工具 | 兼顾自动与可控 |

---

## 路线图

按**模块**推进，完整版（含优先级 / 依赖 / 验收标准 / 关键设计）见 [`演进路线图.md`](./演进路线图.md)。

| 模块 | 下一步重点 |
|---|---|
| 🧠 **memory** | `SessionStore` 短期落盘、语义检索（embedding）、用户画像、双通道注入 |
| 🔧 **tools** | 统一返回协议 `ToolResponse`、`SearchTool`、`RAGTool`、工具结果截断 |
| 🧩 **skill** | 从零建技能系统：`SKILL.md` + 三级渐进式加载 |
| 🔌 **mcp** | 接入 MCP：stdio / HTTP transport + `MCPTool` 适配器 |
| 🤖 **agents** | `PlanExecuteAgent`、`ReflectionAgent`、子代理、多智能体协作 |
| 🪟 **context** | token 预算、工具结果卸载、文件系统工具 |
| 🛠 **infra** | `TraceLogger`、评估工具、异步、打包发布 |

**P0 · 先做**（补齐「用得下去」的短板）：`SearchTool` · `SessionStore` · `TokenCounter` · `llm.embed()`

---

## 参考

本框架在记忆系统的设计上参考了以下开源项目：

- [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) —— 记忆即工具 + 上下文压缩
- [jjyaoao/HelloAgents](https://github.com/jjyaoao/HelloAgents) —— 审计式持久化（修订而非覆盖）
- [LangGraph](https://github.com/langchain-ai/langgraph) —— Checkpointer 的会话/用户两级隔离

详细的记忆模块实现说明（含架构图、逐文件讲解、避坑清单）见 [`记忆模块编写说明.md`](./记忆模块编写说明.md)；
面向后续建设的分模块规划（skill / mcp / Planner 等）见 [`演进路线图.md`](./演进路线图.md)。

---

## License

MIT © LittleSheepy
