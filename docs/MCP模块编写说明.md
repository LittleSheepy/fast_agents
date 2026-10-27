# MCP 模块编写说明

> **对应代码**：`fast_agents/mcp/` + `fast_agents/tools/builtin/mcp_tool.py` + `agents/react_agent/` 中 MCP 相关部分
> **读者**：想让 Agent 接入外部 MCP 工具服务器、或想知道这套零依赖 MCP 是怎么实现的人
> **本文只写「已经实现的东西」**。「还没做但打算做」的部分放在 `docs/演进路线图.md`（D 系列），本文不重复。

---

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：MCP 解决什么问题](#二先想清楚mcp-解决什么问题)
- [三、主流框架 / 协议实现对比](#三主流框架--协议实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：细粒度 vs 粗粒度展开](#五核心机制细粒度-vs-粗粒度展开)
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
| 3.0.0 | 2026-10-27 | 重写。以「MCP 是成员变量、有生命周期」为前提重构：新增 `MCPRegistry` / `MCPClient` / `MCPServerTool`（A1 结构化门脸）、`MCPConfig` 声明式配置、`mcp_tool.py` 三个管理工具 | — |
| 2.0.0 | 2026-10-27 | 引入 `MCPServerTool`、`auto_expand` 钩子（**已废弃**） | — |
| 1.0.0 | 2026-10-27 | 首版：JSON-RPC over stdio/SSE 传输层 + 工具适配 | — |

> **与 `fb/MCP模块编写说明.md` 的区别**：那份描述的是 v2（含 `auto_expand` 钩子、`MCPServerTool` 旧签名）。v3 已删除 `auto_expand`，把「工具展开」交给 `MCPRegistry.register_into(expand=...)`，并让 `MCPServerTool` 改用 A1 结构化入参。**以本文为准。**

---

## 二、先想清楚：MCP 解决什么问题

### 2.1 一句话概括

**MCP（Model Context Protocol）是一套让 Agent 连接「外部工具服务器」的标准协议。**

你写 Agent 时总会遇到这种需求：

- 想让 Agent 操作文件系统；
- 想让 Agent 查数据库、调 GitHub、连 Slack；
- 想让 Agent 用别人已经写好的工具，不想自己重写一遍。

每接一个，就要写一遍「参数怎么传、错误怎么处理、结果怎么翻译成文本」。MCP 的价值在于：**只要对方实现了 MCP server，你这边就只需要「连上去」**——工具清单自动发现、参数 schema 自动转换、调用结果自动翻译。

### 2.2 三个概念

| 概念 | 说明 |
|------|------|
| **MCP server** | 一个独立的进程（stdio）或服务（SSE），对外声明「我提供这些工具」，并接受调用 |
| **MCP client** | 你这边连接 server 的那一端，负责握手、列工具、调工具 |
| **Transport** | 通信通道。目前两种：`stdio`（子进程管道）、`sse`（HTTP Server-Sent Events） |

### 2.3 协议本体

MCP 是**基于 JSON-RPC 2.0** 的。一次交互长这样：

```jsonc
// 客户端 → 服务端
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2024-11-05", "capabilities": {},
    "clientInfo": {"name": "fast-agents", "version": "0.1.0"}}}

// 服务端 → 客户端
{"jsonrpc": "2.0", "id": 1, "result": {
    "protocolVersion": "2024-11-05",
    "serverInfo": {"name": "my-server", "version": "1.0.0"},
    "capabilities": {"tools": {}}}}

// 客户端发通知（无需响应）
{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}

// 列工具
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

// 调工具
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "read_file", "arguments": {"path": "a.txt"}}}
```

本模块只实现了 **`tools`** 这一条链路（`initialize` / `tools/list` / `tools/call`）。MCP 规范里的 `resources` / `prompts` / `sampling` 都没做——`ServerCapabilities` 里虽然预留了字段，但只做解析不做利用。

### 2.4 和「技能」「本地工具」的区别

三者都是给 Agent 加能力，但定位完全不同：

| | 本地工具（`tools/`） | 技能（`skills/`） | MCP（`mcp/`） |
|---|---|---|---|
| **本质** | 一个 Python 函数 | 提示词 + 脚本 + 参考文档的包 | 协议，连外部 server |
| **谁执行** | 本进程内 | 本进程 / 本地子进程 | **外部进程或远程服务** |
| **怎么接入** | `registry.register_tool()` | 三级渐进式加载 | 连接 → `list_tools` → 注册 |
| **上下文成本** | 中（schema 常驻工具表） | 极低（只有索引常驻） | 中（展开后的 schema 常驻） |
| **适合** | 原子能力（计算、读文件） | 复杂作业流程、作业规范 | 复用外部生态、跨语言能力 |
| **生命周期** | 无状态 | 无状态 | **有状态**（子进程 / 连接要关） |

> 最后一行是关键：**MCP 是有生命周期的**。所以 `ReActAgent` 把它做成成员变量（`self.mcp_registry`）而不是无状态工具列表，并在 `close()` 里统一回收。这个设计在 `react_agent.py` 的文件头注释里写明了。

---

## 三、主流框架 / 协议实现对比

### 3.1 对比总表

| 实现 | 协议支持 | 传输 | 是否依赖官方 SDK | 工具暴露形态 |
|------|---------|------|-----------------|-------------|
| **官方 MCP Python SDK** | 完整（tools/resources/prompts/sampling/roots） | stdio / SSE / streamable-http | — | 由你的框架决定 |
| **LangChain** `langchain-mcp-adapters` | tools（+ 部分 resources/prompts） | stdio / SSE / streamable-http | 依赖 | `MultiServerMCPClient` → `BaseTool` 列表 |
| **LlamaIndex** `llama-index-tools-mcp` | tools | stdio / SSE | 依赖 | `FunctionTool` 列表 |
| **OpenAI Agents SDK** | tools | stdio / SSE | 依赖 | `MCPUtil.to_function_tool()` |
| **CrewAI** | tools | stdio | 依赖 | `MCPServerAdapter` → `BaseTool` 列表 |
| **fast_agents** | **仅 tools** | **stdio / SSE** | **零依赖（纯标准库 + 可选 requests）** | 两种：**细粒度** `MCPToolAdapter` / **粗粒度** `MCPServerTool` |

### 3.2 四个关键差异

**① 零依赖，不装 `mcp` 官方 SDK**

`transports.py` 的文件头写得很直白：「不依赖官方 mcp SDK，标准库实现」。因此：

- `StdioTransport` 用 `subprocess.Popen` + 一个后台读线程 + `queue.Queue`；
- `SSETransport` 用 `requests`（**可选依赖**，没装也只在 `connect()` 时给友好报错）；
- JSON-RPC 的收发、id 配对、超时轮询全在 `MCPClient` 里手写（约 60 行）。

好处：装包体积小、不会被 SDK 版本升级打乱。代价：**协议覆盖不全**（只有 tools），也没有官方 SDK 的 `streamable-http` 传输、OAuth、resumability 等。

**② 两种工具暴露形态，这是本模块最有特色的设计**

大多数框架只有一种做法：**一个远端工具 → 一个本地工具**（对齐 LangChain 的 `load_mcp_tools`）。

`fast_agents` 提供了第二种：**一个 server → 一个「门脸工具」**（`MCPServerTool`），模型通过 `action=list_tools` / `action=call_tool` 操作。

为什么需要？**因为工具表长度是有预算的**。假设你连了 5 个 server，每个 server 有 20 个工具，细粒度展开就是 **100 个工具**塞进 prompt——模型的工具选择准确率会明显下降，token 也烧得厉害。粗粒度下只有 **5 个**。

详见 [第五章](#五核心机制细粒度-vs-粗粒度展开)。

**③ 两个形态的「细粒度」实现得比大多数框架细**

`_schema_to_pydantic()` 把 MCP 的 JSON Schema 转成 pydantic 模型时，处理了：

- `type` 是**数组**（如 `["string", "null"]`）的情况；
- `required` 列表 → 决定字段是否 `Optional`；
- `enum` → 转成 `Literal[...]`，让模型看到合法取值；
- `default` → 写进字段默认值；
- `description` → 写进字段描述。

所以导出的 OpenAI / Anthropic 工具 schema 里，每个参数的**类型、描述、必填、枚举**都是完整的。这直接影响调用准确率。

**④ 声明式配置（`MCPConfig`）**

不用手工 `registry.add_stdio(...)`，直接在 `ReActAgent(...)` 里写 `mcp_configs=[...]`：

```python
MCPConfig.stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
MCPConfig.stdio_cmd("demo", "python examples/mcp/echo_server.py")
MCPConfig.sse("weather", "https://example.com/mcp/sse")
```

对齐 Claude Desktop 的 `mcpServers` 配置风格。

### 3.3 为什么本模块「有生命周期」而工具没有

LangChain 的 MCP 适配器里，connection 是 `async with` 出来的；CrewAI 的 `MCPServerAdapter` 也是 context manager。它们的共同点是：**连接由使用方管理**。

`fast_agents` 没有强制 context manager，而是把 `MCPRegistry` 挂成 Agent 的成员变量，并：

- `__init__` 里 `enable_mcp=True` 才连（默认关，避免默认开子进程/网络）；
- `close()` 显式关；
- `__init__` 注册 `atexit` **兜底**（`_safe_close`）——防止异常路径忘记关导致子进程僵尸；
- `__del__` 里也调一次。

也就是说，**大多数情况下你不用管它**，进程退出会自动清理。

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/
├── mcp/
│   ├── __init__.py         # 统一导出
│   ├── errors.py           # 三层异常：连接 / 协议 / 业务
│   ├── models.py           # CallResult / InitializeResult / ServerCapabilities / ToolInfo / MCPConfig
│   ├── transports.py       # Transport 抽象 + StdioTransport + SSETransport + make_stdio
│   ├── client.py           # MCPClient：握手 / list_tools / call_tool / JSON-RPC 收发
│   ├── registry.py         # MCPRegistry：多 server 管理 + 工具展开/卸载
│   └── tool.py             # MCPToolAdapter（细粒度） / MCPServerTool（粗粒度）
├── tools/builtin/
│   └── mcp_tool.py         # list_mcp_servers / connect_mcp_server / disconnect_mcp_server
└── agents/react_agent/
    └── react_agent.py      # enable_mcp / mcp_configs / mcp_expand_mode / _init_mcp / close

examples/mcp/
├── echo_server.py             # 零依赖示例 server（calculate + echo）
├── mcp_quickstart.py          # 最小上手脚本
└── mcp_react_agent_demo.py    # 5 个场景的 Agent 集成示例

test/test_mcp.py               # 10 个用例，含 1 个 MCPConfig/schema 导出
```

### 4.2 分层

```text
┌────────────────────────────────────────────────────────────────────┐
│  ReActAgent（react_agent.py）                                       │
│  enable_mcp / mcp_configs / mcp_expand_mode → _init_mcp()           │
│  close() / _safe_close()（atexit）/ __del__                          │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ 持有
┌──────────────────────────▼─────────────────────────────────────────┐
│  MCPRegistry（registry.py）—— 多 server 管家                        │
│  注册：add_config / add_stdio / add_stdio_cmd / add_sse             │
│  生命周期：connect / connect_all / close / close_all                │
│  展开：register_into(expand=fine|coarse) / unregister_from           │
│  动态：connect_and_expand / close_and_remove                         │
└────────┬───────────────────────────────────┬───────────────────────┘
         │ 每个 server 一个                   │ 展开成
┌────────▼──────────────┐        ┌───────────▼────────────────────────┐
│  MCPClient（client.py）│        │  tool.py                            │
│  connect / initialize  │        │  MCPToolAdapter（细粒度，1:1）       │
│  list_tools            │        │  MCPServerTool（粗粒度，1 server:1） │
│  call_tool             │        └───────────┬────────────────────────┘
│  _request / _notify    │                    │ 注册进
│  _read_loop            │                    ▼
└────────┬──────────────┘        ┌──────────────────────────┐
         │ 用                    │  ToolRegistry（tools 模块）│
┌────────▼──────────────┐        └──────────────────────────┘
│  Transport（transports.py）                                  │
│  StdioTransport（子进程）/ SSETransport（HTTP）              │
└────────┬──────────────────────────────────────────────────────┘
         │ JSON-RPC 2.0 over stdin/stdout 或 HTTP SSE
         ▼
    外部 MCP Server（Python / Node / 任何语言）
```

### 4.3 一次完整的接入与调用

```text
构造 Agent（enable_mcp=True, mcp_configs=[...], mcp_expand_mode="fine"）
  │
  ├─ _init_mcp()
  │    ├─ MCPRegistry() + add_configs(mcp_configs)
  │    ├─ connect_all()
  │    │    └─ 每个 server：MCPClient.connect()
  │    │         ├─ transport.connect()（启动子进程 / 建 SSE）
  │    │         ├─ _request("initialize", …)  ← 握手
  │    │         ├─ _notify("notifications/initialized")
  │    │         └─ _request("tools/list")     → self._tools[name]
  │    ├─ registry.register_into(tool_registry, expand=mode)
  │    │    ├─ fine  → 每个远端工具一个 MCPToolAdapter
  │    │    └─ coarse→ 每个 server 一个 MCPServerTool
  │    └─ 注册 3 个管理工具（list / connect / disconnect_mcp_server）
  │
  └─ agent.run("...")
       │
       ├─ 模型选中 fs__read_file（细粒度）或 mcp__fs + action=call_tool（粗粒度）
       │
       ├─ MCPToolAdapter.run(**kwargs) / MCPServerTool.run(action, tool_name, arguments)
       │    └─ client.call_tool(name, args)
       │         ├─ _request("tools/call", {...}, timeout=60)
       │         │    └─ transport.send() → 等 _read_loop 把响应按 id 填回
       │         └─ CallResult.as_text() → 纯文本
       │
       └─ ToolResult.ok(text) / ToolResult.fail(text) → 回填给模型

agent.close()
  └─ registry.close_all() → 每个 client.close() → transport.close()（terminate 子进程）
```

---

## 五、核心机制：细粒度 vs 粗粒度展开

**这是本模块最需要理解的一个决策。** 两种模式二选一，全局生效（由 `mcp_expand_mode` 决定）。

### 5.1 对比

| | 细粒度（`"fine"`，默认） | 粗粒度（`"coarse"`） |
|---|---|---|
| 工具名 | `{server}__{tool}`，如 `fs__read_file` | `mcp__{server}`，如 `mcp__fs` |
| 一个 server 注册几个工具 | 有多少远端工具就注册几个 | **永远 1 个** |
| 参数 schema | **完整**（type/description/required/enum 全在） | 固定三参数：`action` / `tool_name` / `arguments` |
| 模型调用方式 | 直接点名 `fs__read_file(path=...)` | 先 `action="list_tools"` 看清单，再 `action="call_tool"` |
| 上下文占用 | 高（工具数量 × schema 长度） | 低（常量） |
| 调用往返 | 1 次 | 首次 2 次（先 list 后 call） |
| 调用准确率 | 高（模型看得到参数细节） | 略低（参数要模型自己从文本描述里推断） |
| 适合 | 工具少（< 20 个）、常用 | 工具多、server 多、想省 token |

### 5.2 细粒度：`MCPToolAdapter`

```python
class MCPToolAdapter(ToolBase):
    tags = ["mcp"]

    def __init__(self, client, server_name, tool_info, prefix_sep="__"):
        name = f"{server_name}{prefix_sep}{tool_info.name}"          # fs__read_file
        description = f"[MCP:{server_name}] {tool_info.description}"  # 加来源前缀
        args_schema = _schema_to_pydantic(tool_info.input_schema) \
            if tool_info.has_structured_input else None
        super().__init__(name=name, description=description, args_schema=args_schema)

    def run(self, **kwargs):
        result = self.client.call_tool(self.tool_info.name, kwargs)   # 注意用原始名
        text = result.as_text()
        if result.is_error:
            return ToolResult.fail(text)
        return ToolResult.ok(text, artifact={...})
```

三处值得注意：

1. **工具名加了 `[MCP:{server}]` 前缀**，写在 `description` 里。这样模型知道这个工具来自哪里，多 server 时不会混淆；
2. **调用时用原始工具名**（`tool_info.name`），不是加了前缀的名字——前缀只是本地注册用；
3. **`has_structured_input` 为假时 `args_schema=None`**，退回 `ToolBase` 的默认行为。

`has_structured_input` 的定义在 `models.py`：

```python
@property
def has_structured_input(self) -> bool:
    props = self.input_schema.get("properties", {})
    return bool(props) and set(props.keys()) != {"input"}
```

即「有属性，且不是只有一个叫 `input` 的属性」。后者是 MCP 里常见的简化写法（整个工具只收一个字符串）。

### 5.3 粗粒度：`MCPServerTool`（A1 方案）

入参模型只有三个字段：

```python
class MCPServerCallInput(BaseModel):
    action: str                                    # "list_tools" 或 "call_tool"
    tool_name: Optional[str] = None                # action=call_tool 时必填
    arguments: Optional[Dict[str, Any]] = None     # action=call_tool 时必填
```

`run()` 的分支：

```text
action 归一化（strip + lower）
  ├─ "list_tools" → 返回 _list_tools_text()
  │     每个子工具一行：**名字**：描述
  │        - 参数名 (类型, 必填/可选): 描述
  ├─ 其他 → ToolResult.fail("不支持的 action: '…'")
  └─ "call_tool"
       ├─ 没给 tool_name → fail（提示先 list_tools）
       ├─ tool_name 不在索引里 → fail（列出可用名字）
       ├─ 子工具不是结构化入参 且 arguments 是裸字符串 → 自动包装成 {"input": args}
       └─ client.call_tool(tool_name, args)
```

**`_list_tools_text()` 是粗粒度模式的关键**：它把每个子工具的**参数名、类型、必填性、描述**都渲染成文本给模型看。没有这一步，粗粒度模式下模型根本不知道子工具该怎么传参——这是它和「简单地把所有工具塞进一个 dict」的本质差别。

> **A1 方案的含义**：`action` / `tool_name` / `arguments` 是三个**平铺**的字段，而不是 `action` 枚举里嵌 `tool_name`。这样做的原因是**兼容面最广**——很多模型的 function calling 对「嵌套 oneOf/anyOf」处理得不好，平铺字段最稳。

### 5.4 怎么选

| 场景 | 建议 |
|------|------|
| 1~2 个 server，合计 < 20 个工具 | **fine**（默认，调用最准） |
| server 多 / 工具多 / prompt 预算紧 | **coarse** |
| 混合 | 暂无支持。`mcp_expand_mode` 是全局的，**不支持按 server 分别设置** |

> ⚠️ **运行时动态连接时可以按次指定**：`connect_mcp_server` 工具有 `expand` 参数（见 [7.5](#75-toolsbuiltinmcp_toolpy)），所以「预连接的用 coarse、临时接的用 fine」是可以做到的。但预连接的 server 全部用同一个 `mcp_expand_mode`。

### 5.5 工具名的生成规则

集中在 `MCPRegistry._tool_names_for()`：

```python
if expand == "fine":
    return [f"{server_name}{prefix_sep}{t.name}" for t in tools]   # fs__read_file
if expand == "coarse":
    return [f"mcp{prefix_sep}{server_name}"]                       # mcp__fs
raise MCPError(f"expand 必须是 'fine' 或 'coarse'，收到：{expand}")
```

`prefix_sep` 默认 `"__"`（双下划线）——**不是单下划线**。原因：MCP 工具名本身可能含单下划线（`read_file`），用单下划线做分隔符会导致名字歧义。

---

## 六、核心数据模型

### 6.1 `ToolInfo`（`models.py`）

`tools/list` 返回的工具元信息。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `str` | — | 必填，远端工具名 |
| `description` | `str` | `""` | 工具说明 |
| `input_schema` | `Dict` | `{"type": "object", "properties": {}}` | 入参 JSON Schema |

属性：`has_structured_input`（见 [5.2](#52-细粒度mcptooladapter)）。

### 6.2 `CallResult`（`models.py`）

`tools/call` 的返回。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `content` | `List[Dict]` | `[]` | MCP 内容块列表 |
| `is_error` | `bool` | `False` | 对应响应里的 `isError` |
| `structured_content` | `Optional[Any]` | `None` | 对应 `structuredContent` |

`as_text()` 的翻译规则：

| `content` 里的 `type` | 渲染成 |
|----------------------|--------|
| `text` | 原文 |
| `image` | `[图片：{mimeType}]` |
| `resource` | `[资源] {resource.uri}` |
| 其他 | `str(item)` |
| 空 | `(无输出)` |

> 注意：`image` 只渲染成一行占位文字，**不会真的把图片传给模型**。这是当前实现的边界。

### 6.3 `ServerCapabilities` / `InitializeResult`（`models.py`）

```python
class ServerCapabilities(BaseModel):
    tools / resources / prompts / logging: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = {}          # 其他未列出的能力塞这里

class InitializeResult(BaseModel):
    protocol_version: str = "2024-11-05"
    server_info: Dict[str, Any] = {}    # name / version 等
    capabilities: ServerCapabilities
    instructions: str = ""              # server 给的额外说明（当前未使用）
```

### 6.4 `MCPConfig`（`models.py`）

声明式配置。三个工厂方法：

```python
MCPConfig.stdio(name, command, args=None, env=None, cwd=None)
MCPConfig.sse(name, url, headers=None, timeout=30.0)
MCPConfig.stdio_cmd(name, command_str, cwd=None)
```

字段：

| 字段 | 适用 | 说明 |
|------|------|------|
| `name` | 通用 | server 别名（后续 disconnect / 工具名前缀都用它） |
| `transport` | 通用 | `"stdio"` 或 `"sse"` |
| `command` / `args` / `env` / `cwd` | stdio | 命令式 |
| `command_str` | stdio | 完整命令串，自动 `shlex.split` |
| `url` / `headers` / `timeout` | sse | 地址、请求头、超时 |

> `command` 和 `command_str` **二选一**：`MCPRegistry.add_config()` 里优先看 `command_str`，都没有就 `raise MCPError(f"MCPConfig '{config.name}' (stdio) 需要 command 或 command_str")`。

### 6.5 `Transport`（`transports.py`，抽象基类）

```python
class Transport(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def send(self, message: Dict[str, Any]) -> None: ...
    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def close(self) -> None: ...
```

四个方法，语义清晰。要实现新传输（比如 `streamable-http`）只需继承它。

### 6.6 异常体系（`errors.py`）

三层，全部继承 `MCPError`：

| 异常 | 层 | 什么时候抛 |
|------|-----|-----------|
| `MCPError` | 基类 | 配置错误（`expand` 非法、server 未注册） |
| `MCPConnectionError` | 连接层 | 找不到命令、启动子进程失败、握手超时、请求超时、传输未连接 |
| `MCPProtocolError` | 协议层 | 消息不是合法 JSON-RPC / SSE 握手失败 / SSE POST 4xx+ |
| `MCPCallError` | 业务层 | 远端返回 `error`（含 `code` / `message`） |

> **注意一个易混点**：`_request()` 超时抛的是 **`MCPConnectionError`**（不是 `MCPProtocolError`），因为「没等到响应」本质是连接问题。`MCPToolAdapter.run()` 只把 `MCPCallError` 转成 `ToolResult.fail`，**其他异常会直接冒泡**——这是刻意的（连接断了不该伪装成工具返回错误）。

---

## 七、逐个文件讲解

### 7.1 `__init__.py`

```python
__all__ = [
    "MCPError", "MCPConnectionError", "MCPProtocolError", "MCPCallError",
    "CallResult", "InitializeResult", "ServerCapabilities", "ToolInfo", "MCPConfig",
    "Transport", "StdioTransport", "SSETransport", "make_stdio",
    "MCPClient", "MCPRegistry",
    "MCPToolAdapter", "MCPServerTool",
]
```

> **注意：三个管理工具不在 `mcp/` 里导出。** 它们住在 `fast_agents/tools/builtin/mcp_tool.py`，从 `fast_agents.tools.builtin` 导出。理由见 [10.9](#109-坑管理工具的导入路径)。

### 7.2 `errors.py`

见 [6.6](#66-异常体系errorspy)。19 行，纯类定义。

### 7.3 `models.py`

见 [第六章](#六核心数据模型)。104 行。

### 7.4 `transports.py`

#### `StdioTransport`

```python
StdioTransport(command, args=None, env=None, cwd=None)
```

- **环境变量强制注入**：`PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`，避免子进程在 Windows 上按 GBK 输出；
- **`bufsize=0`**：不缓冲，保证消息及时到达；
- **`creationflags=_CREATE_NO_WINDOW`**：Windows 上不弹黑框（值 `0x08000000`，非 Windows 平台为 `0`）；
- **`stdin/stdout/stderr` 都是 `PIPE`**；
- **后台读线程** `_read_loop`：每次 `read(4096)`，累积到 `buf`，按 `\n` 切分，`json.loads` 后塞进 `queue.Queue(maxsize=256)`；解析失败的**静默跳过**（`continue`）；
- `send()`：`json.dumps(msg, ensure_ascii=False) + "\n"` → UTF-8 编码 → 写 stdin → `flush()`；
- `recv(timeout)`：`queue.get(timeout=timeout)`，`queue.Empty` 返回 `None`；
- `close()`：`terminate()` → `wait(timeout=3)` → 失败则 `kill()`。

> **一条读线程 + 一个队列**是这套实现的核心。它让 `recv()` 变成非阻塞可超时，`MCPClient` 才能在一个循环里做「轮询等 id 匹配」。

#### `SSETransport`

```python
SSETransport(url, headers=None, timeout=30.0)
```

- `connect()` 里 `import requests`，**没装就抛 `MCPConnectionError("SSE 传输需要 requests：pip install requests")`**；
- `GET url`（`stream=True`）→ 起 `_read_sse` 线程；
- **握手靠 `endpoint` 事件**：第一件事等 `queue` 里出现 `{"_event": "endpoint", "endpoint": ...}`，拿到后才算连上；相对 URL 会 `urljoin` 补全；
- `_read_sse` 按 SSE 规范解析：空行 = 事件结束、`:` 开头 = 注释、`event:` / `data:` 分别解析，多行 `data` 用 `\n` 拼接；
- `send()` 是 **HTTP POST 到 `_post_url`**（不是长连接写），`>= 400` 抛 `MCPProtocolError`；
- `recv()` **会跳过 `_event` 类消息**（递归调用直到拿到真消息）。

> **概念澄清**：这里的「SSE」其实是一个**混用**——下行用 SSE 长连接，上行用普通 HTTP POST。这是 MCP 早期 SSE 传输的规范做法（后来被 `streamable-http` 取代）。

#### `make_stdio`

```python
def make_stdio(command_str: str, cwd: Optional[str] = None) -> StdioTransport:
    parts = shlex.split(command_str, posix=(sys.platform != "win32"))
    if not parts:
        raise MCPConnectionError("命令不能为空")
    return StdioTransport(command=parts[0], args=parts[1:], cwd=cwd)
```

> **`posix=(sys.platform != "win32")`** 这一行是 Windows 兼容的关键。`posix=True`（Unix 默认）会把反斜杠当转义符，Windows 路径 `D:\code` 会被吃掉；所以 Windows 上必须用 `posix=False`。

### 7.5 `client.py`

**构造**：

```python
MCPClient(transport, client_name="fast-agents", client_version="0.1.0")
PROTOCOL_VERSION = "2024-11-05"
```

**生命周期**：

| 方法 | 说明 |
|------|------|
| `connect(timeout=10.0)` | `transport.connect()` → 起 `_read_loop` 线程 → `_request("initialize", …)` → 构建 `InitializeResult` → `_notify("notifications/initialized")` → 返回 |
| `close()` | `connected = False` + `transport.close()`（异常吞掉） |

**工具操作**：

| 方法 | 说明 |
|------|------|
| `list_tools(timeout=10.0)` | 未连接 → `MCPConnectionError`；`_request("tools/list")` → 转成 `List[ToolInfo]` |
| `call_tool(name, arguments=None, timeout=60.0)` | 未连接 → `MCPConnectionError`；`_request("tools/call", {...})` → `CallResult`。**默认超时 60 秒**（工具调用可能很慢，比握手宽得多） |

**底层 RPC**：

- `_next_id()`：加锁自增；
- `_request(method, params, timeout)`：生成 id → 记进 `_request_id_to_response[id] = None` → `send()` → **每 10ms 轮询一次**该 dict → 拿到就 pop 并检查 `error` 字段 → 超时则 pop 并抛 `MCPConnectionError`；
- `_notify(method, params)`：只 `send()`，不等响应，异常吞掉；
- `_read_loop()`：`recv(timeout=0.2)` 轮询；带 `id` 且含 `result`/`error` 的 → 按 id 填回 dict；**server 主动发的请求统一回 `-32601 Method not found`**。

> **`_request` 用的是轮询（polling）而非 `Event`**。`self._event` 在构造里建了但**没被用到**——是一处遗留代码。功能上没问题（10ms 粒度够用），但可以优化成 `threading.Event.wait()`。

### 7.6 `registry.py`

**注册**：

| 方法 | 说明 |
|------|------|
| `add_config(config)` | 按 `MCPConfig` 分派到下面两个；stdio 时 `command_str` 优先 |
| `add_configs(configs)` | 循环 `add_config` |
| `add_stdio(name, command, args, env, cwd)` | 存一个 **factory**（闭包），不立即连接 |
| `add_stdio_cmd(name, command_str, cwd)` | 用 `make_stdio` 建好 transport，再包成 factory |
| `add_sse(name, url, headers, timeout)` | 同上 |

> 关键设计：**注册只存 factory，不连接**。连接是 `connect()` 的事。这让「注册很多 server 但只连其中几个」变得自然。

**生命周期**：

| 方法 | 说明 |
|------|------|
| `connect(name, timeout=10.0)` | **幂等**：已在 `_clients` 里直接返回；否则 factory 造 client → `connect()` → `list_tools()` → 记日志 |
| `connect_all(timeout=10.0)` | 遍历 factories；**单个失败只 `logger.warning`，不中断** |
| `close(name)` | 关 client + 清 `_tools` / `_registered_names` / `_expand_mode` / `_prefix_sep` |
| `close_all()` | 遍历 `close()`，置 `_closed = True` |

**查询**：

- `clients` / `factories`：返回**副本**（`dict(...)`），防止外部改内部状态；
- `get_client(name)`：取单个；
- `list_tools(server=None)`：不传返回全部 `{name: [ToolInfo]}`。

**展开**：

| 方法 | 说明 |
|------|------|
| `register_into(registry, prefix_sep="__", expand="fine")` | 遍历已连接 server；**先 `_unregister_one` 卸旧的**（避免重复注册）→ `_expand_one` → 记录 `_registered_names` / `_expand_mode` / `_prefix_sep` |
| `unregister_from(registry, server_name=None)` | 卸载（**不关连接**）。不传 `server_name` 卸载全部 |
| `_unregister_one(registry, server_name)` | 先按 `_registered_names` 精确卸；**再扫前缀兜底**（`{server}__` 和 `mcp__{server}`），清理「用过别的模式注册」的残留 |
| `connect_and_expand(name, registry, expand="fine", prefix_sep="__", timeout=10.0)` | 连接 + 立即展开，返回工具名列表。供 `connect_mcp_server` 用 |
| `close_and_remove(name, registry)` | 卸载 + 关连接 |

> **`_unregister_one` 的前缀兜底**值得单独说：它保证「同一个 server 先 fine 后 coarse」或反过来时，不会在工具表里留垃圾。

### 7.7 `tool.py`

#### `_schema_to_pydantic(schema)`

把 JSON Schema 转 pydantic 模型。转换映射：

| JSON Schema `type` | Python 类型 |
|-------------------|-------------|
| `string` | `str` |
| `integer` | `int` |
| `number` | `float` |
| `boolean` | `bool` |
| `array` | `list` |
| `object` | `dict` |
| 未知 / 缺失 | `str`（兜底） |

三个额外处理：

1. **`type` 是列表**（如 `["string", "null"]`）：取第一个非 `null` 类型；若含 `null` 且不在 `required` 里，包成 `Optional`；
2. **不在 `required` 里** → 包成 `Optional`，且 `default` 取 schema 里的值，没有就 `None`；
3. **有 `enum`** → 覆盖成 `Literal[tuple(enum)]`。

模型名固定 `"MCPArgs"`，用 `create_model("MCPArgs", **fields)` 动态建。

> **局限**：只处理**一层**属性。嵌套 `object`（如 `{"type": "object", "properties": {...}}`）不会递归展开，只会变成 `dict`。这是当前实现为控制复杂度做的取舍。

#### `MCPToolAdapter`

见 [5.2](#52-细粒度mcptooladapter)。`run()` 逻辑：

```python
try:
    result = self.client.call_tool(self.tool_info.name, kwargs)
except MCPCallError as e:
    return ToolResult.fail(str(e))
text = result.as_text()
if result.is_error:
    return ToolResult.fail(text)
return ToolResult.ok(text, artifact={"content": [c for c in result.content if c.get("type") == "text"]})
```

> `artifact` 只保留 `text` 类内容块。

#### `MCPServerTool`

见 [5.3](#53-粗粒度mcpservertoola1-方案)。

### 7.8 `tools/builtin/mcp_tool.py`

三个管理工具，**对齐 `memory_tool.py` 的三工具模式**（都是「构造时持 Agent 引用」）。

| 工具 | `name` | 入参 | 说明 |
|------|--------|------|------|
| `ListMcpServersTool` | `list_mcp_servers` | 无（`ListMcpInput` 空模型） | 列出已注册 / 已连接的 server。已连接显示 ✅ + server 名 + 版本 + 工具名；未连接显示 ⏳ |
| `ConnectMcpServerTool` | `connect_mcp_server` | `name` / `target` / `transport="auto"` / `expand="fine"` | 运行时动态连接 |
| `DisconnectMcpServerTool` | `disconnect_mcp_server` | `name` | 断开并移除工具 |

**`ConnectMcpServerTool.run()` 的校验链**（五道）：

```text
1. registry 为空 → fail("MCP 系统未启用（enable_mcp=False）")
2. name 非法（只允许字母/数字/_/-） → fail
3. name 已连接 → fail（提示先 disconnect）
4. expand 不是 fine/coarse → fail
5. transport 归一化：
     "auto" → target 以 http(s):// 开头则 sse，否则 stdio
     然后 add_sse / add_stdio_cmd
     异常 → registry._factories.pop(name, None) 回滚 + fail
成功后 refresh_system_prompt()，返回注册的工具名列表
```

> 第 5 步的**回滚**很关键：连接失败后把 factory 清掉，避免留一个「注册了但连不上」的脏状态。

`build_mcp_tools(agent)` 返回三个工具的列表，风格与 `build_memory_tools(manager)` 一致：

```python
return [
    ListMcpServersTool(agent),
    ConnectMcpServerTool(agent),
    DisconnectMcpServerTool(agent),
]
```

### 7.9 `agents/react_agent/react_agent.py` 的 MCP 部分

**构造参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `mcp_registry` | `Optional[MCPRegistry]` | `None` | 传入则复用；不传且 `enable_mcp=True` 时自动建空注册表 |
| `mcp_configs` | `Optional[List[MCPConfig]]` | `None` | 声明式配置 |
| `enable_mcp` | `bool` | **`False`** | 是否启用 MCP |
| `mcp_expand_mode` | `str` | `"fine"` | 展开模式 |

**`_init_mcp(mcp_registry, mcp_configs, expand_mode)` 的四步**（对齐 `_init_skills` 的结构）：

```text
1. 校验 expand_mode ∈ {fine, coarse}，否则 raise ValueError
2. registry = mcp_registry or MCPRegistry()
   if mcp_configs: registry.add_configs(mcp_configs)
3. registry.connect_all()
   registry.register_into(self.tool_registry, expand=expand_mode)
   self.mcp_registry = registry
4. for t in build_mcp_tools(self): self.tool_registry.register_tool(t)
   打印连接结果（连了几个 / 一个都没连上）
```

**资源回收三重保险**：

| 机制 | 位置 | 作用 |
|------|------|------|
| `close()` | 显式调用 | `registry.close_all()`，`try/except` 包住 |
| `_safe_close()` | `atexit` 注册 | 正常退出兜底 |
| `__del__()` | GC 时 | 最后一道兜底，`try/except` 全吞 |

> `_safe_close` 的 docstring 写得很清楚：「防止异常路径上忘记 `close` 导致子进程僵尸」。这是**子进程场景必须做的**——不然 Agent 崩了会留一堆孤儿进程。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
cd D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents
.\第1步uv_init.bat
.\第2步uv_venv.bat
.\第3步uv_add.bat
```

> **SSE 传输需要额外装 `requests`**（`requirements.txt` 里没有）：
> ```powershell
> uv pip install requests
> ```
> 只用 stdio 则不需要。

### 8.2 三十秒跑通（零依赖、不需要真实 LLM）

```powershell
python examples\mcp\mcp_quickstart.py
```

这个脚本会启动 `examples/mcp/echo_server.py`（一个零依赖的示例 server，提供 `calculate` 和 `echo` 两个工具），然后依次演示：连接 → 列工具 → 调工具 → 关闭。

### 8.3 接到 ReActAgent 上（细粒度）

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fast_agents.agents.react_agent.react_agent import ReActAgent
from fast_agents.mcp import MCPConfig
from fast_agents.tools import ToolRegistry

# echo_server 路径按实际调整
ECHO_CMD = f'"{sys.executable}" examples/mcp/echo_server.py'

agent = ReActAgent(
    name="助手",
    llm=your_llm,
    tool_registry=ToolRegistry(),
    enable_memory=False,
    enable_skills=False,
    enable_mcp=True,
    mcp_configs=[MCPConfig.stdio_cmd("demo", ECHO_CMD)],
    mcp_expand_mode="fine",
)

print("已注册工具：", agent.tool_registry.list_names())
# → ['demo__calculate', 'demo__echo', 'list_mcp_servers', 'connect_mcp_server', 'disconnect_mcp_server']

print(agent.run("用 MCP 算 sqrt(2)*2，再回显 3 次「细粒度展开」，前缀 [FINE]。"))
agent.close()
```

### 8.4 换粗粒度

只改一行：

```python
agent = ReActAgent(
    ...,
    mcp_expand_mode="coarse",   # 工具表里只有 mcp__demo
)
```

模型的调用会从 `demo__echo(message=..., prefix=..., repeat=...)` 变成两步：

```text
① mcp__demo(action="list_tools")
   → "## demo 提供的子工具：\n- **echo**：…\n    - message (string, 必填): …"

② mcp__demo(action="call_tool", tool_name="echo",
             arguments={"message": "粗粒度展开", "prefix": "[COARSE] ", "repeat": 2})
```

### 8.5 运行时动态连接

不在构造时配，让模型自己决定：

```python
from fast_agents.tools import ToolRegistry
from fast_agents.agents.react_agent.react_agent import ReActAgent

agent = ReActAgent(
    name="助手",
    llm=your_llm,
    tool_registry=ToolRegistry(),
    enable_mcp=True,          # 不传 mcp_configs，注册表为空
    mcp_expand_mode="fine",
)
# 模型现在有 connect_mcp_server，可以自己说：
#   「先连接 MCP server：name=fs，target=uvx -y @modelcontextprotocol/server-filesystem D:/code」
print(agent.run("我需要操作 D:/code 目录下的文件，先把这个能力接进来。"))
agent.close()
```

### 8.6 跑测试

```powershell
python test\test_mcp.py
```

预期 **10 个用例全过**（`echo_server.py` 是零依赖的，测试不需要外部服务）。

---

## 九、API 速查

### 9.1 异常

```python
from fast_agents.mcp import MCPError, MCPConnectionError, MCPProtocolError, MCPCallError
```

### 9.2 数据模型

```python
from fast_agents.mcp import CallResult, InitializeResult, ServerCapabilities, ToolInfo, MCPConfig

ToolInfo(name="read_file", description="读文件",
         input_schema={"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]})
ToolInfo(...).has_structured_input     # bool

CallResult(...).as_text()              # str

MCPConfig.stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
MCPConfig.stdio_cmd("demo", "python server.py", cwd="D:/proj")
MCPConfig.sse("remote", "https://example.com/mcp/sse", headers={"Authorization": "Bearer x"})
```

### 9.3 传输

```python
from fast_agents.mcp import Transport, StdioTransport, SSETransport, make_stdio

StdioTransport(command="uvx", args=["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
StdioTransport(command="python", args=["server.py"], env={"MY_VAR": "1"}, cwd="D:/proj")
SSETransport(url="https://example.com/mcp/sse", headers={}, timeout=30.0)
make_stdio("uvx -y @modelcontextprotocol/server-filesystem D:/code")

t.connect(); t.send({...}); t.recv(timeout=5); t.close()
```

### 9.4 客户端

```python
from fast_agents.mcp import MCPClient

c = MCPClient(StdioTransport("python", ["server.py"]))
init = c.connect(timeout=10.0)     # -> InitializeResult
c.connected                        # bool
init.server_info                   # {"name": ..., "version": ...}
init.capabilities.tools            # dict 或 None

tools = c.list_tools(timeout=10.0)              # List[ToolInfo]
res = c.call_tool("echo", {"message": "hi"}, timeout=60.0)   # CallResult
c.close()
```

### 9.5 注册表

```python
from fast_agents.mcp import MCPRegistry, MCPConfig

reg = MCPRegistry()

# 注册（不连接）
reg.add_config(MCPConfig.stdio_cmd("demo", "python server.py"))
reg.add_configs([cfg1, cfg2])
reg.add_stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
reg.add_stdio_cmd("demo", "python server.py", cwd="D:/proj")
reg.add_sse("remote", "https://example.com/mcp/sse", headers=None, timeout=30.0)

# 生命周期
reg.connect("demo", timeout=10.0)     # -> MCPClient（幂等）
reg.connect_all(timeout=10.0)         # 失败只 warning
reg.close("demo")
reg.close_all()

# 查询
reg.clients                            # Dict[str, MCPClient]（副本）
reg.factories                          # Dict[str, Callable]（副本）
reg.get_client("demo")
reg.list_tools()                       # Dict[str, List[ToolInfo]]
reg.list_tools("demo")                 # {"demo": [...]}

# 展开
reg.register_into(tool_registry, prefix_sep="__", expand="fine")
reg.unregister_from(tool_registry)                 # 全部卸载
reg.unregister_from(tool_registry, "demo")         # 卸载单个

# 动态
names = reg.connect_and_expand("demo", tool_registry, expand="fine")   # List[str]
removed = reg.close_and_remove("demo", tool_registry)                  # List[str]
```

### 9.6 工具适配器

```python
from fast_agents.mcp import MCPToolAdapter, MCPServerTool

MCPToolAdapter(client=..., server_name="fs", tool_info=info, prefix_sep="__")
# name = "fs__read_file"，tags = ["mcp"]

MCPServerTool(client=..., server_name="fs", tools=[info1, info2], prefix_sep="__")
# name = "mcp__fs"，tags = ["mcp"]
```

### 9.7 管理工具

```python
from fast_agents.tools.builtin import (
    ListMcpServersTool, ConnectMcpServerTool, DisconnectMcpServerTool, build_mcp_tools,
)

build_mcp_tools(agent)      # -> [三个工具的列表]
```

### 9.8 Agent 侧参数

```python
ReActAgent(
    ...,
    mcp_registry=None,            # Optional[MCPRegistry]
    mcp_configs=None,             # Optional[List[MCPConfig]]
    enable_mcp=False,             # bool
    mcp_expand_mode="fine",       # "fine" | "coarse"
)
agent.mcp_registry                # Optional[MCPRegistry]
agent.close()                     # 关闭所有连接
```

---

## 十、设计决策与常见坑

### 10.1 为什么 `enable_mcp` 默认 False

因为它会**开子进程或建网络连接**。默认打开意味着：

- 用户 `import` 一下、建个 Agent，后台就多了几个进程；
- 在受限环境（无网络、无权限）会直接报错或卡住；
- 调试纯对话逻辑时被无关的连接日志干扰。

所以默认关、显式开。**技能（`enable_skills`）和记忆（`enable_memory`）也遵循同一原则。**

### 10.2 坑：`connect_all()` 失败是静默的

```python
def connect_all(self, timeout=10.0):
    for name in list(self._factories.keys()):
        try:
            self.connect(name, timeout=timeout)
        except Exception as e:
            logger.warning("MCP[%s] 连接失败：%s", name, e)
```

**只打 `logger.warning`，不抛异常、不中断。** 好处是「一个 server 挂了不影响其他」，坏处是**如果你的 logging 没配，什么都看不到**——模型会发现某个工具不存在，而你不知道为什么。

**建议**：调试时给 `logging.basicConfig(level=logging.INFO)`；或改用 `_init_mcp` 后检查 `len(registry.clients)`。`_init_mcp` 末尾会打印连接结果，这是最直接的信号。

### 10.3 坑：细粒度下工具表会膨胀

见 [5.4](#54-怎么选)。**这是这套设计里最需要主动决策的一件事**：

- 连 3 个 server，每个 15 个工具 → 45 个工具进 prompt；
- 加上技能三件套 + 记忆三件套 + 管理三件套 → 已经 50+。

模型的工具选择准确率会随工具数量上升而下降，而且 token 消耗很实在。

**建议**：工具超过 20 个就考虑 `coarse`；或者干脆不预连接，让模型用 `connect_mcp_server` 按需接（用完 `disconnect_mcp_server` 断开）。

### 10.4 坑：粗粒度模式下模型可能不先 `list_tools`

`MCPServerTool` 的 `description` 里明确写了「先用 `action='list_tools'` 查看子工具清单，再用 `action='call_tool'` 调用指定子工具」，但**模型不一定会照做**——它可能直接瞎猜 `tool_name`。

代码里已经做了两层兜底：

- 没给 `tool_name` → `fail("action=call_tool 时必须提供 tool_name；可先调用 list_tools 查看可用子工具")`；
- `tool_name` 不存在 → `fail("子工具 'x' 不存在于 fs；可用：read_file, write_file")`——**把可用列表直接告诉模型**。

这两条提示语很关键，模型看到后一般会自我纠正。

### 10.5 坑：`_schema_to_pydantic` 只展开一层

嵌套 `object`（如 `{"type": "object", "properties": {"config": {"type": "object", "properties": {...}}}}`）只会变成 `dict`，不给模型看子字段。

**后果**：模型在传嵌套参数时没有 schema 引导，容易传错结构。粗粒度模式下 `_list_tools_text()` 也是只列一层参数。

**建议**：如果接的 server 有复杂嵌套入参，优先确认它的 `description` 里有没有文字说明；没有的话，考虑给那个工具单独写一个本地包装工具。

### 10.6 坑：`artifact` 只留 text，图片不传

`MCPToolAdapter.run()` 里：

```python
return ToolResult.ok(text, artifact={"content": [c for c in result.content if c.get("type") == "text"]})
```

`CallResult.as_text()` 遇到 `image` 只渲染 `[图片：image/png]`。**也就是说 MCP 返回的图片，模型看不到内容。**

**建议**：多模态需求目前要自己在 `CallResult` 之后补一步处理（比如把 base64 提出来转成本框架的图片消息），不能指望默认链路。

### 10.7 坑：子进程的 stderr 没人读

`StdioTransport.connect()` 里：

```python
stderr=subprocess.PIPE,
```

但**代码里没有任何地方读 `proc.stderr`**。操作系统管道有缓冲区（Windows 约 4KB / Unix 约 64KB），子进程如果往 stderr 写很多东西，**写满后会阻塞**，表现出来就是「工具调用莫名其妙卡住」。

**后果**：接一个日志很啰嗦的 server，跑一会儿就假死。

**建议**（按代价从低到高）：

1. 短期：给 server 加 `--quiet` 或关掉详细日志；
2. 中期：起一个线程读 `stderr` 并写进 `logging`（几行代码）；
3. 或者把 `stderr` 改成 `subprocess.DEVNULL`（要求 server 的日志不重要）。

### 10.8 坑：`MCPClient._event` 建了没用

```python
self._event: threading.Event = threading.Event()
```

构造里有，但 **`_request()` 全程用 `time.sleep(0.01)` 轮询**，从没 `self._event.wait()` / `.set()`。

功能上没问题（10ms 粒度足够），但：

- 每次请求最多多等 10ms；
- 高并发下是纯 CPU 浪费（虽然 MCP 场景基本是串行的）。

**建议**：改成 `Event` 或 `condition` 通知，或在 timeout 内用指数退避。

### 10.9 坑：管理工具的导入路径

三个管理工具定义在 `fast_agents/tools/builtin/mcp_tool.py`，**不在 `fast_agents/mcp/`**。所以：

```python
from fast_agents.mcp import build_mcp_tools                 # ❌ ImportError
from fast_agents.tools.builtin import build_mcp_tools       # ✅
```

**为什么这么放**：`mcp/` 是**协议实现层**，`tools/builtin/` 是**工具层**。三个管理工具是「工具」，跟 `memory_tool.py` 同层，不属于协议实现。这样分层更干净，也避免 `mcp/` 反向依赖 `tools/` 造成循环导入（`tool.py` 已经 `from ..tools.tool_base import ToolBase` 了）。

### 10.10 坑：`register_into` 会重复注册工具名

`register_into` 里对每个 server **先 `_unregister_one` 再 `_expand_one`**，所以重复调用是安全的（不会报「工具名已存在」）。

但要注意：**`_unregister_one` 只清它自己记录过的名字**（`_registered_names`），加前缀兜底扫描。如果你手动往 `tool_registry` 里注册过同名工具，它**不会**被清掉，然后 `register_tool` 可能报重名。

**建议**：MCP 工具名前缀 `{server}__` / `mcp__` 尽量不要和本地工具撞名。

### 10.11 坑：只有 `tools`，没有 `resources` / `prompts`

`ServerCapabilities` 解析了 `resources` / `prompts` / `logging`，但**代码里没有任何地方用它们**。`MCPClient` 只有 `list_tools` / `call_tool`。

**后果**：如果某个 server 的核心能力是「提供资源」（比如文件内容以 resource 形式暴露）而不是「提供工具」，接进来会发现没东西可用。

**建议**：接之前先确认对方声明的是 `capabilities.tools` 非空。

### 10.12 跨平台注意事项（Windows 专项）

本模块对 Windows 做了**五处专门处理**，逐条列出：

**① 防黑框（`_CREATE_NO_WINDOW`）**

```python
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
```

只在 Windows 生效。`subprocess.Popen(creationflags=_CREATE_NO_WINDOW)` 让子进程不弹控制台窗口。**没有这个，每次连一个 stdio server 都会闪一下黑框**——GUI 程序里尤其明显。

**② 强制 UTF-8**

```python
merged_env["PYTHONIOENCODING"] = "utf-8"
merged_env["PYTHONUTF8"] = "1"
```

Windows 简体中文环境的默认代码页是 GBK。如果子进程是 Python 写的，不强制 UTF-8 的话，它写出来的中文 JSON 是 GBK 字节，父进程按 UTF-8 解会乱码——**表现为工具返回一堆问号或直接 JSON 解析失败**。

> ⚠️ 但这只对 **Python 子进程**有效。如果 server 是 Node / Go / Rust 写的，得靠它自己输出 UTF-8（MCP 规范要求如此）。

**③ `shlex.split` 用 `posix=False`**

```python
parts = shlex.split(command_str, posix=(sys.platform != "win32"))
```

Windows 路径里的反斜杠在 `posix=True` 下会被当转义符吃掉。所以 Windows 必须 `posix=False`。

**④ 路径写法**

`MCPConfig.stdio_cmd("fs", "uvx -y @modelcontextprotocol/server-filesystem D:/code")`——**推荐用正斜杠**（`D:/code`）。原因：

- 反斜杠在命令串里要写两次（`D:\\code`）才不会被 `shlex` 吃掉；
- 正斜杠 Windows 也认，少一层心智负担；
- 如果路径**含空格**（如 `D:/My Code`），整个参数要加引号：`"... \"D:/My Code\""`，这时 `posix=False` 的 `shlex` 会把引号剥掉。

**⑤ `npx` / `uvx` 的可执行文件后缀**

`StdioTransport` 用 `subprocess.Popen([command, *args])`，**没有 `shell=True`**。这意味着 Windows 上：

- `uvx` → 可能实际是 `uvx.exe`，Python 的 `Popen` 会尝试补 `.exe`，通常能成功；
- **`npx` → 实际是 `npx.cmd`（批处理），`Popen` 直接调用会 `FileNotFoundError`**。

报错长这样：

```text
MCPConnectionError: 找不到命令：npx（请确认已安装）
```

**三种修法**：

```python
# 法一：显式写 .cmd（最简单）
MCPConfig.stdio_cmd("fs", "npx.cmd -y @modelcontextprotocol/server-filesystem D:/code")

# 法二：用 cmd /c 包一层
MCPConfig.stdio("fs", "cmd", ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", "D:/code"])

# 法三：改用 uvx（Python 生态，通常没这个问题）
MCPConfig.stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
```

**⑥ 换行符与编码（db / 配置无影响）**

JSON-RPC 消息用 `\n` 分隔，`_read_loop` 里按 `b"\n"` 切分。Windows 上子进程如果输出 `\r\n`，`line.strip()` 会把 `\r` 一起去掉，**不影响解析**。

**⑦ 测试可直接跑 Windows**

`echo_server.py` 是纯标准库、零依赖、无第三方包的示例 server，`test/test_mcp.py` 在 Windows 上可直接 `python test\test_mcp.py` 跑通，不需要额外配置。

### 10.13 坑：`MCPRegistry._closed` 只写不读

`close_all()` 里 `self._closed = True`，但**没有任何地方检查过它**。所以 `close_all()` 之后再 `connect()` 一个新 server，仍然能成功——`_closed` 更像是个语义标记。

**建议**：如果要在关闭后禁止再连，需要在 `add_*` / `connect` 里加 `if self._closed: raise ...`。

### 10.14 坑：调用超时是固定 60 秒

`MCPClient.call_tool(timeout=60.0)` 是硬编码默认值，**没有从配置读**。有些工具（比如跑一个长脚本）会超过 60 秒。

**建议**：目前只能改源码或用 `MCPToolAdapter` 时自己调 `client.call_tool(..., timeout=xxx)`。如果这是常见需求，考虑把超时提成 `MCPConfig` 的字段。

---

## 十一、测试与验证

### 11.1 现状

`test/test_mcp.py` **10 个用例，全部通过**（用零依赖的 `echo_server.py`）：

| # | 用例 | 验证点 |
|---|------|--------|
| 1 | `test_1_connect_and_list_tools` | 握手成功 + `list_tools` 返回正确数量 |
| 2 | `test_2_call_tool` | `call_tool` 返回内容正确 |
| 3 | `test_3_call_nonexistent_tool_raises` | 调不存在的工具抛 `MCPCallError` |
| 4 | `test_4_tool_adapter_schema_and_call` | 细粒度适配器的 schema 正确 + 调用正常 |
| 5 | `test_5_coarse_server_tool_a1` | 粗粒度门脸工具（A1 结构）的 `list_tools` / `call_tool` |
| 6 | `test_6_registry_register_into_fine` | `register_into(expand="fine")` |
| 6b | `test_6b_registry_register_into_coarse` | `register_into(expand="coarse")` |
| 7 | `test_7_unregister_from` | 卸载工具（含前缀兜底） |
| 8 | `test_8_dynamic_connect_and_close` | `connect_and_expand` / `close_and_remove` |
| 9 | `test_9_mcp_config_and_schema_export` | `MCPConfig` 三个工厂 + schema 导出正常 |

另外 `test/test_react_agent.py` 里有 `enable_mcp` 相关用例，也是通过的。

### 11.2 手工验证

```powershell
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from fast_agents.mcp import MCPRegistry, MCPConfig

reg = MCPRegistry()
reg.add_config(MCPConfig.stdio_cmd('demo', f'\"{sys.executable}\" examples/mcp/echo_server.py'))
reg.connect_all()

print('已连接:', list(reg.clients.keys()))
print('工具:', [t.name for t in reg.list_tools('demo')['demo']])

c = reg.get_client('demo')
r = c.call_tool('echo', {'message': 'hello', 'prefix': '[X] ', 'repeat': 2})
print('调用结果:', r.as_text())
print('is_error:', r.is_error)

reg.close_all()
"
```

### 11.3 检查清单

- [ ] `enable_mcp=True` 时才连外部 server；
- [ ] `mcp_expand_mode` 与工具数量匹配（>20 个考虑 `coarse`）；
- [ ] 工程里 `close()` 被调到（或依赖 `atexit` / `__del__`）；
- [ ] `logging` 有配（否则 `connect_all` 的失败你看不见）；
- [ ] 工具名不和本地工具撞（前缀 `{server}__` / `mcp__`）；
- [ ] 接 npx 类 server 时用了 `.cmd` 或 `cmd /c`；
- [ ] 子进程 stderr 不会写爆（见 [10.7](#107-坑子进程的-stderr-没人读)）；
- [ ] 需要 SSE 时装了 `requests`；
- [ ] 不依赖 `resources` / `prompts`（当前不支持）。

---

## 十二、下一步演进路线

只列方向，编号以 `docs/演进路线图.md` 的 D 系列为准：

- 支持 `streamable-http` 传输（替代/补充 SSE）；
- 支持 `resources` / `prompts`（现在只解析不使用）；
- `stderr` 独立读取线程，避免子进程阻塞（[10.7](#107-坑子进程的-stderr-没人读)）；
- 超时可配置（提到 `MCPConfig` 上）（[10.14](#1014-坑调用超时是固定-60-秒)）；
- `_schema_to_pydantic` 支持嵌套 `object`（[10.5](#105-坑_schema_to_pydantic-只展开一层)）；
- 图片内容透传（多模态）（[10.6](#106-坑artifact-只留-text图片不传)）；
- `_request` 用 `Event` 替代轮询（[10.8](#108-坑mcpclient_event-建了没用)）；
- `close_all()` 后用 `_closed` 真正拦截再连接（[10.13](#1013-坑mcpregistry_closed-只写不读)）；
- 按 server 分别指定展开模式（[5.4](#54-怎么选)）。

---

## 十三、参考

**仓库内**

- `fast_agents/mcp/`（7 个文件）
- `fast_agents/tools/builtin/mcp_tool.py`（三个管理工具）
- `fast_agents/agents/react_agent/react_agent.py`（`_init_mcp` / `close` / `_safe_close`）
- `examples/mcp/`（`echo_server.py` / `mcp_quickstart.py` / `mcp_react_agent_demo.py`）
- `test/test_mcp.py`
- `docs/演进路线图.md`（D 系列）

**外部**

- MCP 官方规范：`modelcontextprotocol.io`（协议版本 `2024-11-05`）
- 官方 Python SDK：`mcp`（本模块**未使用**，用作协议对照）
- LangChain：`langchain-mcp-adapters`（`MultiServerMCPClient`）
- LlamaIndex：`llama-index-tools-mcp`
- OpenAI Agents SDK：`MCPUtil`
- CrewAI：`MCPServerAdapter`

---

## 附录：发布前自检清单

改动 `fast_agents/mcp/` 或 `tools/builtin/mcp_tool.py` 之后，逐条打勾：

- [ ] `python -c "from fast_agents.mcp import MCPRegistry, MCPClient, MCPToolAdapter, MCPServerTool, MCPConfig, StdioTransport, SSETransport"` 无报错；
- [ ] `python test\test_mcp.py` 10/10 通过；
- [ ] `python test\test_react_agent.py` 全过（含 MCP 用例）；
- [ ] `enable_mcp=False` 时不会创建任何子进程 / 网络连接；
- [ ] `fine` / `coarse` 两种 `register_into` 都能正确展开，且切换时**不留残留工具**；
- [ ] 新增 `expand` 取值时同步改了 `_tool_names_for` 和 `_expand_one` 的校验；
- [ ] 新增 `Transport` 实现时实现了全部四个抽象方法；
- [ ] 新增异常类型时挂在 `MCPError` 下，并在本文件 [6.6](#66-异常体系errorspy) 登记；
- [ ] Windows 上仍能用 `python test\test_mcp.py` 跑通（无黑框、无乱码、无 `FileNotFoundError`）；
- [ ] `mcp/__init__.py` 的 `__all__` 与新导出保持一致；
- [ ] `close()` / `_safe_close()` / `__del__` 三重回收仍有效（无子进程泄漏）；
- [ ] 本文档 [六章](#六核心数据模型)、[九章](#九api-速查) 已同步；
- [ ] 本文档 [一、更新历史](#一更新历史) 加了一行。
```
