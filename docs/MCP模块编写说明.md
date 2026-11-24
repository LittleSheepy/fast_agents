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
| 3.3.6 | 2026-11-24 | **P4 轻微问题批量打磨（审计编号 5–10 一次修完，零新增依赖、默认行为不变）**：①`GetPromptResult.as_messages(role_prefix)` 是**死参数**（从未被读取）——现让它生效（`True` 时 content 前缀 `[role] `），默认值由 `True` 改为 `False`，因参数此前无效，**无参调用输出与旧版逐字节一致**；②`client.py` 顶层 `MCPProtocolError` 死 import 删除；③`MCPRegistry._closed` **只写不读**——现 `close_all()` 之后 `connect()` 直接抛 `MCPError`（`connect_all` / `connect_and_expand` 走既有异常兜底，降级为 warning / 友好失败）；④`_unregister_one` 的兜底前缀扫描**硬编码 `"__"`**，自定义 `prefix_sep` 注册的历史残留清不掉——改用该 server 上次展开记录的 `prefix_sep`；⑤`StreamableHTTPTransport._json_resolved` **只增不清**（长驻进程慢速泄漏）——响应被 `recv()` 消费即 `discard` 该 id，`close()` 整表 clear 兜底；⑥`list_changed` 刷新后已注册的粗粒度 `MCPServerTool._tools_index` **永久停在旧快照**，新增子工具永远报「不存在」——新增 `MCPServerTool.update_tools()`，registry 用 `_coarse_tools` 留引用并在刷新时同步；⑦**顺带挖出的根因**：`list_changed` 处理器由客户端读线程同步回调，而刷新自身要再发请求、响应只有读线程消费 → **自死锁直到超时，v3.2 的自动刷新从未在任何传输上生效过**（探针实证），现由 `_schedule_refresh` 甩独立 daemon 线程。新增回归用例 `test_polish_p1`–`test_polish_p5`，tests/mcp 全量 65/65 全绿 | — |
| 3.3.5 | 2026-11-24 | **缺陷修复（声明式配置 env 静默丢弃，零新增依赖、仅纯新增可选参数）**：`MCPRegistry.add_config` 走 stdio `command_str` 分支时此前调 `add_stdio_cmd(name, command_str, cwd)` 把 `config.env` 丢弃——`StdioTransport` 本身支持 env，缺的是中间透传环节，声明式写法的环境变量从未生效（探针 P-D 实证）；现 `make_stdio_transport` / `add_stdio_cmd` / `MCPConfig.stdio_cmd` 三层增加可选参数 `env=None` 并透传，`add_config` 传入 `config.env`；默认 `None` 行为逐字节不变。新增回归用例 `test_registry_r1`，tests/mcp 全量 60/60 全绿 | — |
| 3.3.4 | 2026-11-24 | **缺陷修复（Schema 转换缓存键，零新增依赖、零签名变更）**：`_schema_to_pydantic` 嵌套对象缓存键原为 `str(prop["properties"])`，不含同级 `required` 与对象自身约束——同一 schema 里两个 properties 完全一致、必填集合不同的嵌套对象会复用同一模型，必填字段被静默降级为可选（校验绕过）或可选字段被误强制必填（合法调用被拒），双向错乱（探针 P-C 实证）；缓存键改为整个 prop 的规范化序列化（`json.dumps(sort_keys=True)`），结构等价 ⇔ 键等价，完全相同结构的缓存复用仍生效。新增回归用例 `test_schema_c12`，tests/mcp 全量 59/59 全绿 | — |
| 3.3.3 | 2026-11-24 | **缺陷修复（资源模板方法名，零新增依赖、零破坏性变更）**：`MCPClient.list_resource_templates()` 此前误复用 `resources/list`——合规服务器不会在该响应里返回 `resourceTemplates`，导致本方法恒为空（此前仓内示例 server 同样不规范地把模板混在 `resources/list` 里返回，两边联合掩盖问题，`test_d2`/`test_d8` 一直绿）；现改走规范方法 `resources/templates/list`，并对「声明了 resources 能力但未实现本方法（JSON-RPC -32601）」的服务器降级返回空列表，不打断 `registry.connect` 等上层流程；`MCPCallError` 新增可选 `error_code` 参数（默认 `None`，既有单参构造不变）供调用方区分错误码；`examples/mcp/resources_prompts_server.py` 同步修正为模板独立方法返回。新增回归用例 `test_d11`，tests/mcp 全量 58/58 全绿 | — |
| 3.3.2 | 2026-11-24 | **缺陷修复（stdio 传输层，零新增依赖、零签名变更）**：Windows 上 `make_stdio_transport` 的 `shlex.split(posix=False)` 会把引号字符残留在 token 里，`'"exe" "script"'` 形式的命令串（含空格路径的标准写法）启动时程序名带字面引号交给 `CreateProcess`，必然 `FileNotFoundError`（`test_8_dynamic_connect_and_close` 此前因此确定性红灯）；现在切分后剥离 token 首尾成对双引号并还原 `\"`，空 / 纯引号命令提前抛 `MCPConnectionError`；非 Windows 走 `posix=True` 路径行为不变（引号本就被 shlex 剥离）。新增回归用例 `test_transport_t8` | — |
| 3.2.1 | 2026-11-18 | **传输层拆分为包（纯结构调整，零依赖变更）**：`fast_agents/mcp/transports.py`（453 行）拆成 `transports/` 包——`transport_base.py`（抽象基类）、`stdio_transport.py`（`StdioTransport` + 工厂）、`sse_transport.py`（`SSETransport`），`fast_agents.mcp.transports` 导入路径不变；按仓库 `*_base` 命名约定，`Transport` → `TransportBase`、`make_stdio` → `make_stdio_transport`（`fast_agents` 与 `fast_agents.mcp` 两处顶层同名导出随之更名，属破坏性重命名，旧名不保留）。测试 48/48 不变 | — |
| 3.2.0 | 2026-11-17 | **新增 `resources` / `prompts` 支持**：`MCPClient` 增加 `list_resources()` / `list_resource_templates()` / `read_resource(uri)` / `list_prompts()` / `get_prompt(name, arguments)` 五个方法（服务器未声明对应能力时自动返回空列表、不发请求）；`models.py` 增加 `ResourceInfo` / `ResourceTemplateInfo` / `ReadResourceResult` / `TextResourceContents` / `BlobResourceContents` / `PromptInfo` / `PromptArgument` / `PromptMessage` / `GetPromptResult` 九个模型；`MCPRegistry` 在 `connect()` 时缓存 resources / templates / prompts 并注册 `notifications/*/list_changed` 自动刷新处理器，新增对应查询与转发方法；粗粒度 `MCPServerTool` 的 `action` 从 2 个扩展到 6 个（新增 `list_resources` / `read_resource` / `list_prompts` / `get_prompt`）；`CallResult` 新增 `images()` 提取图片内容。测试 38 → 48 | — |
| 3.1.0 | 2026-11-17 | **稳定性版本（对外 API、签名、既有测试断言均不变）**：①`MCPClient` 内部升级——`threading.Event` 唤醒替代 10ms 忙等轮询；引入状态机（`disconnected/connecting/connected/closing/closed`）使 `connect()` / `close()` 幂等且线程安全；新增只读属性 `state` 与 `on_notification()`——服务器推送的 notification 不再被丢弃，改由注册的 handler 分发；`close()` 会唤醒在途请求并等待读线程优雅退出；删除死代码 `_event`。②传输层健壮性——`StdioTransport` 新增 stderr 后台读线程 + `get_last_stderr()`、子进程异常退出主动抛错、`send()` 检查进程存活、`close()` 优雅退出；`SSETransport` 修复 `recv()` 递归隐患（改循环）、队列满丢最旧不阻塞读线程、流断开投递 disconnected 哨兵、POST 错误分级（5xx → 连接错 / 4xx → 协议错）、SSE 事件去缓冲（`chunk_size=1` + `Accept-Encoding: identity`）。③`_schema_to_pydantic` 增强——支持 array `items`、嵌套 `object` 递归、`enum`、`default`（含 falsy 值修正）、`minimum` / `maximum` / `minLength` / `maxLength` / `pattern` / `minItems` / `maxItems`、`anyOf` / `oneOf` 含 `null` 转 `Optional`。同时修复测试里示例服务器路径 bug（`parents[1]`→`parents[2]`，该 bug 曾使 10 个 MCP 用例整体失败）。测试 10 → 38 | — |
| 3.0.0 | 2026-10-27 | 重写。以「MCP 是成员变量、有生命周期」为前提重构：新增 `MCPRegistry` / `MCPClient` / `MCPServerTool`（A1 结构化门脸）、`MCPConfig` 声明式配置、`mcp_tool.py` 三个管理工具 | — |
| 2.0.0 | 2026-10-27 | 引入 `MCPServerTool`、`auto_expand` 钩子（**已废弃**） | — |
| 1.0.0 | 2026-10-27 | 首版：JSON-RPC over stdio/SSE 传输层 + 工具适配 | — |

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
| **MCP client** | 你这边连接 server 的那一端，负责握手，列工具 / 资源 / 提示模板，并调用工具、读资源、渲染模板 |
| **TransportBase** | 通信通道。目前两种：`stdio`（子进程管道）、`sse`（HTTP Server-Sent Events） |

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

本模块实现了 **`tools`** / **`resources`** / **`prompts`** 三条链路（`initialize` / `tools/list` / `tools/call` / `resources/list` / `resources/read` / `prompts/list` / `prompts/get`）。MCP 规范里的 `sampling` / `roots` 仍未做——`ServerCapabilities` 里预留了字段，但只做解析不做利用。

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
| **fast_agents** | **tools / resources / prompts** | **stdio / SSE** | **零依赖（纯标准库 + 可选 requests）** | 两种：**细粒度** `MCPToolAdapter` / **粗粒度** `MCPServerTool` |

### 3.2 四个关键差异

**① 零依赖，不装 `mcp` 官方 SDK**

`transports/` 包的文件头写得很直白：「不依赖官方 mcp SDK，标准库实现」。因此：

- `StdioTransport` 用 `subprocess.Popen` + 一个后台读线程 + `queue.Queue`；
- `SSETransport` 用 `requests`（**可选依赖**，没装也只在 `connect()` 时给友好报错）；
- JSON-RPC 的收发、id 配对、超时轮询全在 `MCPClient` 里手写（约 130 行）。

好处：装包体积小、不会被 SDK 版本升级打乱。代价：**协议覆盖不全**（`sampling` / `roots` 未做），也没有官方 SDK 的 `streamable-http` 传输、OAuth、resumability 等。

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
- `default` → 写进字段默认值（含 `False` / `0` / `""` / `[]` 等 falsy 值，不再被误判为 `None`）；
- `description` → 写进字段描述；
- `array` 的 `items` → 转成 `list[元素类型]`（含对象数组递归）；嵌套 `object` → 递归生成子模型；
- 数值 / 字符串 / 数组校验（`minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum` / `minLength` / `maxLength` / `pattern` / `minItems` / `maxItems`）→ 对应 pydantic 的 `ge` / `le` / `gt` / `lt` / `min_length` / `max_length` / `pattern`；
- `anyOf` / `oneOf` 含 `null` → `Optional[T]`；
- 不认识的类型 → 降级为 `Any`，打 debug 日志但不抛异常、不阻塞工具注册。

所以导出的 OpenAI / Anthropic 工具 schema 里，每个参数的**类型、描述、必填、枚举、嵌套结构、校验范围**都是完整的。这直接影响调用准确率。

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
│   ├── models.py           # CallResult / InitializeResult / ServerCapabilities / ToolInfo / MCPConfig + Resources/Prompts 九个模型
│   ├── transports/         # 传输层包（TransportBase 抽象 + StdioTransport / SSETransport + make_stdio_transport）
│   │   ├── __init__.py         # 包导出（TransportBase / StdioTransport / SSETransport / make_stdio_transport）
│   │   ├── transport_base.py   # TransportBase 抽象基类（原 Transport）
│   │   ├── stdio_transport.py  # StdioTransport + make_stdio_transport
│   │   └── sse_transport.py    # SSETransport
│   ├── client.py           # MCPClient：握手 / list_tools / call_tool / JSON-RPC 收发
│   ├── registry.py         # MCPRegistry：多 server 管理 + 工具展开/卸载
│   └── tool.py             # MCPToolAdapter（细粒度） / MCPServerTool（粗粒度）
├── tools/builtin/
│   └── mcp_tool.py         # list_mcp_servers / connect_mcp_server / disconnect_mcp_server
└── agents/react_agent/
    └── react_agent.py      # enable_mcp / mcp_configs / mcp_expand_mode / _init_mcp / close

examples/mcp/
├── echo_server.py                 # 零依赖示例 server（calculate + echo）
├── notify_server.py               # 会主动推送 notification 的示例 server
├── crash_server.py                # 崩溃 / 大量 stderr 示例 server（传输层健壮性验证）
├── sse_test_server.py             # 零依赖 SSE 测试 server
├── complex_schema_server.py       # 复杂入参示例 server（嵌套 / 枚举 / 校验）
├── resources_prompts_server.py    # 提供 Tools + Resources + Prompts 完整能力的示例 server
├── mcp_quickstart.py              # 最小上手脚本
├── mcp_react_agent_demo.py        # 5 个场景的 Agent 集成示例
├── mcp_client_features_demo.py    # v3.1 客户端能力（通知/状态/幂等/并发）
├── mcp_transport_stability_demo.py# v3.1 传输层稳定性演示
├── mcp_complex_schema_demo.py     # v3.1 复杂 Schema 能力演示
└── mcp_resources_prompts_demo.py  # v3.2 Resources / Prompts 演示

tests/mcp/test_mcp.py          # 48 个用例，含 MCPConfig/schema 导出 + v3.1 客户端/传输/schema 专项 + v3.2 Resources/Prompts 专项
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
│  list_tools/call_tool  │        │  MCPServerTool（粗粒度，1 server:1） │
│  resources / prompts   │        └───────────┬────────────────────────┘
│  _request / _notify    │                    │ 注册进
│  _read_loop            │                    ▼
└────────┬──────────────┘        ┌──────────────────────────┐
         │ 用                    │  ToolRegistry（tools 模块）│
┌────────▼──────────────┐        └──────────────────────────┘
│  TransportBase（transports/）                                │
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
  │    │         └─ _request("tools/list")     → self._tools[name]（+resources/prompts）
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

入参模型字段：

```python
class MCPServerCallInput(BaseModel):
    action: str                                    # 6 个：list_tools/call_tool/list_resources/read_resource/list_prompts/get_prompt
    tool_name: Optional[str] = None                # action=call_tool 时必填
    arguments: Optional[Dict[str, Any]] = None     # action=call_tool / get_prompt 时传入
    resource_uri: Optional[str] = None             # action=read_resource 时必填
    prompt_name: Optional[str] = None              # action=get_prompt 时必填
```

`run()` 的分支：

```text
action 归一化（strip + lower）
  ├─ "list_tools" → 返回 _list_tools_text()
  │     每个子工具一行：**名字**：描述
  │        - 参数名 (类型, 必填/可选): 描述
  ├─ "list_resources" → _list_resources_text()（资源 + 参数化模板）
  ├─ "read_resource"  → 需 resource_uri，client.read_resource(uri)
  ├─ "list_prompts"   → _list_prompts_text()
  ├─ "get_prompt"     → 需 prompt_name，client.get_prompt(name, arguments)
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
| `image` | `[图片：{mimeType}, 长度={base64长度}]` |
| `resource` | `[资源] {resource.uri}` |
| 其他 | `str(item)` |
| 空 | `(无输出)` |

另有 `images()` 方法（v3.2 新增），把 `image` 类内容块提取成 `[{"mime": ..., "data": ...}]`，方便多模态链路取用。

> 注意：`as_text()` 里 `image` 仍只渲染成一行占位文字；要拿到图片本体需用 `images()` 自行接入多模态消息。

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
MCPConfig.stdio_cmd(name, command_str, cwd=None, env=None)
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

### 6.5 Resources / Prompts 数据模型（`models.py`）

`resources` 与 `prompts` 的协议模型（v3.2 新增）：

| 模型 | 对应协议 | 关键字段 |
|------|---------|---------|
| `ResourceInfo` | `resources/list` | `uri` / `name` / `description` / `mimeType` / `size` |
| `ResourceTemplateInfo` | `resources/list`（模板） | `uriTemplate` / `name` / `description` / `mimeType` |
| `TextResourceContents` / `BlobResourceContents` | `resources/read` | `uri` / `mimeType` / `text` 或 `blob`(base64) |
| `ReadResourceResult` | `resources/read` | `contents`（列表），`as_text()` 汇总文本 |
| `PromptArgument` | `prompts/list` | `name` / `description` / `required` |
| `PromptInfo` | `prompts/list` | `name` / `description` / `arguments` |
| `PromptMessage` | `prompts/get` | `role` / `content`，`as_text()` 渲染 |
| `GetPromptResult` | `prompts/get` | `description` / `messages`，`as_messages()` 转 `{role, content}`；`as_messages(role_prefix=True)` 在 content 前附 `[role] ` 前缀（该参数 v3.3.6 起真正生效，默认 `False` 保持标准消息） |

### 6.6 `TransportBase`（`transports/transport_base.py`，抽象基类）

```python
class TransportBase(ABC):
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

### 6.7 异常体系（`errors.py`）

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
    "ResourceInfo", "ResourceTemplateInfo", "ReadResourceResult",
    "PromptInfo", "PromptArgument", "PromptMessage", "GetPromptResult",
    "TextResourceContents", "BlobResourceContents",
    "TransportBase", "StdioTransport", "SSETransport", "make_stdio_transport",
    "MCPClient", "MCPRegistry",
    "MCPToolAdapter", "MCPServerTool",
]
```

> **注意：三个管理工具不在 `mcp/` 里导出。** 它们住在 `fast_agents/tools/builtin/mcp_tool.py`，从 `fast_agents.tools.builtin` 导出。理由见 [10.9](#109-坑管理工具的导入路径)。

### 7.2 `errors.py`

见 [6.7](#67-异常体系errorspy)。19 行，纯类定义。

### 7.3 `models.py`

见 [第六章](#六核心数据模型)。217 行（v3.2 增加 Resources / Prompts 模型）。

### 7.4 `transports/`（传输层包）

| 文件 | 内容 |
|------|------|
| `transport_base.py` | `TransportBase` 抽象基类（原 `Transport`，按仓库 `*_base` 命名约定更名） |
| `stdio_transport.py` | `StdioTransport` + `make_stdio_transport` 工厂 |
| `sse_transport.py` | `SSETransport` |
| `streamable_http_transport.py` | `StreamableHTTPTransport` + `make_streamable_http_transport` 工厂（v3.3.0 随 D6 交付：MCP 2025 规范单端点 HTTP，响应 Content-Type 分流 JSON 直返 / SSE 流，`Mcp-Session-Id` 会话 + `close()` 发 DELETE） |

> 原单文件 `transports.py`（453 行）拆成包后，`fast_agents.mcp.transports` 导入路径与 `from fast_agents.mcp import StdioTransport` 等用法不变；仅抽象基类与工厂函数更名——`Transport` → `TransportBase`、`make_stdio` → `make_stdio_transport`。

#### `StdioTransport`（`stdio_transport.py`）

```python
StdioTransport(command, args=None, env=None, cwd=None)
```

- **环境变量强制注入**：`PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`，避免子进程在 Windows 上按 GBK 输出；
- **`bufsize=0`**：不缓冲，保证消息及时到达；
- **`creationflags=_CREATE_NO_WINDOW`**：Windows 上不弹黑框（值 `0x08000000`，非 Windows 平台为 `0`）；
- **`stdin/stdout/stderr` 都是 `PIPE`**；
- **stdout 后台读线程** `_read_stdout_loop`：每次 `read(4096)`，累积到 `buf`，按 `\n` 切分，`json.loads` 后塞进 `queue.Queue(maxsize=256)`；解析失败的**静默跳过**（`continue`）；读完后检查子进程退出码，非零则向队列投递 `disconnected` 哨兵并附 stderr 尾部；
- **stderr 后台读线程** `_read_stderr_loop`（v3.1 新增）：持续消费 `proc.stderr`（避免子进程 stderr 写满管道阻塞），最近若干行保存在环形缓冲，`get_last_stderr(n=30)` 取用；
- `send()`（v3.1 起写前检查子进程存活，已退出则抛带 exit code + stderr 尾部的 `MCPConnectionError`）：`json.dumps(msg, ensure_ascii=False) + "\n"` → UTF-8 编码 → 写 stdin → `flush()`；
- `recv(timeout)`：`queue.get(timeout=timeout)`，`queue.Empty` 返回 `None`；遇 `disconnected` 哨兵抛 `MCPConnectionError`；
- `close()`（v3.1 起优雅退出）：先关 stdin，`terminate()` → 等待 → 失败则 `kill()`，并等待 stdout / stderr 读线程退出（≤1s），避免僵尸进程与遗留读线程。

> **两条后台读线程（stdout + stderr）+ 一个队列**是这套实现的核心：`recv()` 变成非阻塞可超时，`MCPClient` 用「每请求一个 `threading.Event`」按 id 唤醒（见 [7.5](#75-clientpy)）。

#### `SSETransport`（`sse_transport.py`）

```python
SSETransport(url, headers=None, timeout=30.0)
```

- `connect()` 里 `import requests`，**没装就抛 `MCPConnectionError("SSE 传输需要 requests：pip install requests")`**；
- `GET url`（`stream=True`）→ 起 `_read_sse` 线程；
- **握手靠 `endpoint` 事件**：第一件事等 `queue` 里出现 `{"_event": "endpoint", "endpoint": ...}`，拿到后才算连上；相对 URL 会 `urljoin` 补全；
- `_read_sse` 按 SSE 规范解析：空行 = 事件结束、`:` 开头 = 注释、`event:` / `data:` 分别解析，多行 `data` 用 `\n` 拼接；
- `send()` 是 **HTTP POST 到 `_post_url`**（不是长连接写），错误分级：`>= 500` 抛 `MCPConnectionError`（可重试）、`>= 400` 抛 `MCPProtocolError`（客户端错）；
- `recv()` **循环跳过 `_event` 类消息**（v3.1 由递归改循环，避免栈溢出），遇 `disconnected` 哨兵抛 `MCPConnectionError`；
- `_safe_enqueue()`：队列满时**丢最旧消息**，永不阻塞读线程；
- 去缓冲：`iter_lines(chunk_size=1)` + 请求头 `Accept-Encoding: identity`，避免 SSE 事件被攒住而延迟到达；
- `close()` 等待读线程退出（≤2s）。

> **概念澄清**：这里的「SSE」其实是一个**混用**——下行用 SSE 长连接，上行用普通 HTTP POST。这是 MCP 早期 SSE 传输的规范做法（后来被 `streamable-http` 取代）。

#### `StreamableHTTPTransport`（`streamable_http_transport.py`，v3.3.0）

```python
StreamableHTTPTransport(url, headers=None, timeout=30.0)
```

- `connect()` **不发请求**，只校验 URL + 建 session（真正握手由上层 `initialize` 触发，那时才能拿到 `Mcp-Session-Id`）；`requests` 延迟 import，未安装时首次使用抛 `MCPConnectionError`；
- `send()` POST 同一端点，按响应 `Content-Type` 分流：`application/json` 直接当本次响应入队（**不建后台线程**）；`text/event-stream` 才起读线程持续解析并入队；
- **`_json_resolved` 记账**：JSON 直返已处理过的请求 id 登记在此，避免同一响应被 SSE 读线程重复分发；**v3.3.6 修正**：该集合此前只增不清（长驻进程慢速泄漏），现在响应被 `recv()` 消费即 `discard(msg_id)`（取 `_resolve_lock`），`close()` 整表 clear 兜底；
- 流断开时读线程用 `Last-Event-ID` 续传重连，多次失败后投递 `disconnected` 哨兵，与 `SSETransport` 行为一致。

#### `make_stdio_transport`（`stdio_transport.py`）

```python
def _unquote_token(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] == '"':
        # posix=False 保留引号；剥首尾成对双引号并还原内部的 \"
        return token[1:-1].replace('\\"', '"')
    return token

def make_stdio_transport(command_str: str, cwd: Optional[str] = None,
                         env: Optional[Dict[str, str]] = None) -> StdioTransport:
    parts = [_unquote_token(p) for p in shlex.split(command_str, posix=(sys.platform != "win32"))]
    if not parts or not parts[0]:
        raise MCPConnectionError("命令不能为空")
    return StdioTransport(command=parts[0], args=parts[1:], cwd=cwd, env=env)
```

> **`posix=(sys.platform != "win32")`** 这一行是 Windows 兼容的关键。`posix=True`（Unix 默认）会把反斜杠当转义符，Windows 路径 `D:\code` 会被吃掉；所以 Windows 上必须用 `posix=False`。
>
> **v3.3.2 补充**：`posix=False` 的代价是引号字符会残留在 token 里（posix 模式才会剥离），带字面引号的程序名交给 `Popen` 直接 `FileNotFoundError`——所以切分后要用 `_unquote_token` 剥离首尾成对双引号。
>
> **v3.3.5 补充**：新增可选 `env` 参数透传 `StdioTransport`（合并进 `os.environ` 后再覆盖），补齐 `command_str` 形态此前没有 env 入口的缺口，见 [10.15](#1015-坑声明式配置-command_str-形态的-env-被静默丢弃v335-已修复)。

### 7.5 `client.py`

**构造**：

```python
MCPClient(transport, client_name="fast-agents", client_version="0.1.0")
PROTOCOL_VERSION = "2024-11-05"
```

**状态与生命周期**：

| 成员 | 说明 |
|------|------|
| `state`（只读属性） | 连接状态字符串：`disconnected` → `connecting` → `connected` → `closing` → `closed` |
| `connected`（只读属性） | `state == "connected"` 的语法糖，兼容旧代码对 `client.connected` 的**读取**（不再可赋值） |
| `on_notification(method, handler)` | 注册服务器推送 notification 的处理器；handler 在读线程中同步调用，应快速返回 |
| `connect(timeout=10.0)` | **幂等**：已连接时直接返回既有 `InitializeResult`；`transport.connect()` → 起 `_read_loop` 线程 → `_request("initialize", …)` → 构建 `InitializeResult` → `_notify("notifications/initialized")`；失败回滚为 `disconnected` |
| `close()` | **幂等且线程安全**：置 `closing` → 唤醒全部在途请求（抛 `MCPConnectionError`）→ `transport.close()` → `join` 读线程（≤2s）→ 置 `closed` |

**工具操作**：

| 方法 | 说明 |
|------|------|
| `list_tools(timeout=10.0)` | 未连接 → `MCPConnectionError`；`_request("tools/list")` → 转成 `List[ToolInfo]` |
| `call_tool(name, arguments=None, timeout=60.0)` | 未连接 → `MCPConnectionError`；`_request("tools/call", {...})` → `CallResult`。**默认超时 60 秒**（工具调用可能很慢，比握手宽得多） |

**资源与提示模板操作（v3.2 新增）**：

| 方法 | 说明 |
|------|------|
| `list_resources(timeout=10.0)` | 列出资源；服务器未声明 `resources` 能力时直接返回 `[]`（不发请求） |
| `list_resource_templates(timeout=10.0)` | 列出参数化资源模板（`resourceTemplates`）；走规范方法 `resources/templates/list`（v3.3.3），服务器未实现该方法（-32601）时降级返回 `[]` |
| `read_resource(uri, timeout=30.0)` | 按 URI 读取资源 → `ReadResourceResult`（文本 / 二进制 blob 均支持） |
| `list_prompts(timeout=10.0)` | 列出提示模板；服务器未声明 `prompts` 能力时直接返回 `[]` |
| `get_prompt(name, arguments=None, timeout=30.0)` | 渲染提示模板 → `GetPromptResult`（`as_messages()` 可直接注入对话） |

**底层 RPC**：

- `_next_id()`：加锁自增，保证并发下 id 唯一；
- `_request(method, params, timeout)`：生成 id → 记进 `_pending[id] = _PendingRequest()` → `send()` → **`pending.event.wait(timeout)`（释放 GIL，无忙等）** → 被读线程 `set()` 唤醒后取回响应 → 检查 `error` 字段；超时或被 `close()` 唤醒则抛 `MCPConnectionError`（并区分「超时」与「连接已关闭」两种原因）；
- `_notify(method, params)`：只 `send()`，不等响应，异常吞掉；
- `_read_loop()`：`recv(timeout=0.2)` 轮询，分三类处理——**响应（有 `id` 且有 `result`/`error`）** → 按 id 唤醒对应请求线程；**server→client 请求（有 `id`）** → 回 `-32601 Method not found`；**notification（无 `id`）** → 分发给 `on_notification` 注册的 handler。读线程异常退出时会唤醒所有在途请求。

> notification 的 handler 在读线程中同步执行，抛出的异常会被捕获并记 debug 日志，**不会打断读线程**——耗时逻辑请自行起线程处理。

### 7.6 `registry.py`

**注册**：

| 方法 | 说明 |
|------|------|
| `add_config(config)` | 按 `MCPConfig` 分派到下面两个；stdio 时 `command_str` 优先 |
| `add_configs(configs)` | 循环 `add_config` |
| `add_stdio(name, command, args, env, cwd)` | 存一个 **factory**（闭包），不立即连接 |
| `add_stdio_cmd(name, command_str, cwd, env)` | 用 `make_stdio_transport` 建好 transport，再包成 factory（`env` 为 v3.3.5 新增可选参数） |
| `add_sse(name, url, headers, timeout)` | 同上 |

> 关键设计：**注册只存 factory，不连接**。连接是 `connect()` 的事。这让「注册很多 server 但只连其中几个」变得自然。

**生命周期**：

| 方法 | 说明 |
|------|------|
| `connect(name, timeout=10.0)` | **幂等**：已在 `_clients` 里直接返回；否则 factory 造 client → `connect()` → `list_tools()` / `list_resources()` / `list_resource_templates()` / `list_prompts()` → 注册 `notifications/*/list_changed` 自动刷新处理器 → 记日志；`close_all()` 之后调用抛 `MCPError`（v3.3.6，`_closed` 从只写不读变为真正拦截） |
| `connect_all(timeout=10.0)` | 遍历 factories；**单个失败只 `logger.warning`，不中断** |
| `close(name)` | 关 client + 清 `_tools` / `_resources` / `_resource_templates` / `_prompts` / `_registered_names` / `_expand_mode` / `_prefix_sep` / `_coarse_tools` |
| `close_all()` | 遍历 `close()`，置 `_closed = True`（v3.3.6 起 `connect()` 会检查它，关闭后拒绝再连） |

**查询**：

- `clients` / `factories`：返回**副本**（`dict(...)`），防止外部改内部状态；
- `get_client(name)`：取单个；
- `list_tools(server=None)`：不传返回全部 `{name: [ToolInfo]}`；
- `list_resources` / `list_resource_templates` / `list_prompts`（v3.2 新增）：同样不传返回全部，传单个返回 `{server: [...]}`；
- `read_resource(server_name, uri)` / `get_prompt(server_name, prompt_name, arguments)`（v3.2 新增）：转发到对应 client，未连接则抛 `MCPError`。

**list_changed 自动刷新（v3.2 新增）**：`connect()` 时若服务器声明 `listChanged`，为工具 / 资源 / 提示分别注册 `notifications/tools|resources|prompts/list_changed` 处理器，收到推送即重拉对应列表刷新本地缓存（只刷缓存、不自动重注册工具，避免打断正在执行的调用）。

> **v3.3.6 两处修正**：①**根因**——处理器由客户端读线程同步回调，而 `_refresh_*` 内部要再发一次请求，响应只有读线程自己能消费，同步执行就是**自死锁直到 5s 超时**（探针实证：v3.2 起这条自动刷新链路从未真正生效过）；现统一走 `_schedule_refresh()`，把刷新甩到独立 daemon 线程，处理器立即返回。②刷新时同步已注册**粗粒度门脸**的工具索引（`_coarse_tools` → `MCPServerTool.update_tools()`），否则门脸永远只认注册那一刻的子工具列表。

**展开**：

| 方法 | 说明 |
|------|------|
| `register_into(registry, prefix_sep="__", expand="fine")` | 遍历已连接 server；**先 `_unregister_one` 卸旧的**（避免重复注册）→ `_expand_one` → 记录 `_registered_names` / `_expand_mode` / `_prefix_sep` |
| `unregister_from(registry, server_name=None)` | 卸载（**不关连接**）。不传 `server_name` 卸载全部 |
| `_unregister_one(registry, server_name)` | 先按 `_registered_names` 精确卸；**再扫前缀兜底**（`{server}{sep}` 和 `mcp{sep}{server}`），清理「用过别的模式注册」的残留；`sep` 取该 server 上次展开记录的 `_prefix_sep`（默认 `"__"`，v3.3.6 前硬编码） |
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
| `array` | `list[元素类型]`（有 `items`）或 `list` |
| `object` | 递归生成的嵌套模型 |
| 未知 / 缺失 | `Any`（降级，打 debug 日志） |

额外处理（v3.1 起增强）：

1. **`type` 是列表**（如 `["string", "null"]`）：取第一个非 `null` 类型；若含 `null` 且不在 `required` 里，包成 `Optional`；
2. **不在 `required` 里** → 包成 `Optional`，`default` 取 schema 里的值（**含 `False` / `0` / `""` / `[]` 等 falsy 值，不再被误判为 `None`**），没有就 `None`；
3. **有 `enum`** → 覆盖成 `Literal[tuple(enum)]`；
4. **`array`**：有 `items` 则递归转成 `list[元素类型]`（支持对象数组）；
5. **嵌套 `object`**：递归生成子模型（整个 prop 的规范化序列化等价才走缓存复用；v3.3.4 前键只含 `str(properties)`，同构不同必填会串模型，见 [10.5](#105-坑schema_to_pydantic-只展开一层v31-已修复)）；
6. **校验约束**：`minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` → `ge`/`le`/`gt`/`lt`；`minLength`/`maxLength`/`pattern` → `min_length`/`max_length`/`pattern`；`minItems`/`maxItems` → `min_items`/`max_items`；
7. **`anyOf`/`oneOf` 含 `null`** → `Optional[T]`。

顶层模型名 `"MCPArgs"`，嵌套子模型名 `"MCPArgs_{字段}"`，用 `create_model` 动态建。

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

**`update_tools(tools)`（v3.3.6 新增）**：原地刷新门脸内部的 `_tools_list` / `_tools_index` 快照。`MCPRegistry` 收到 `tools/list_changed` 时**不会**重注册工具（避免打断在途调用），但会把新列表喂给已注册的门脸，使服务器新增的子工具立刻可通过 `action=call_tool` 调用——修复「门脸只认注册那一刻的旧列表」的索引过期问题。

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
python tests\mcp\test_mcp.py
```

预期 **48 个用例全过**（`echo_server.py` / `notify_server.py` 等示例 server 都是零依赖的，测试不需要外部服务）。

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
from fast_agents.mcp import TransportBase, StdioTransport, SSETransport, make_stdio_transport

StdioTransport(command="uvx", args=["-y", "@modelcontextprotocol/server-filesystem", "D:/code"])
StdioTransport(command="python", args=["server.py"], env={"MY_VAR": "1"}, cwd="D:/proj")
SSETransport(url="https://example.com/mcp/sse", headers={}, timeout=30.0)
make_stdio_transport("uvx -y @modelcontextprotocol/server-filesystem D:/code")

t.connect(); t.send({...}); t.recv(timeout=5); t.close()
```

### 9.4 客户端

```python
from fast_agents.mcp import MCPClient

c = MCPClient(StdioTransport("python", ["server.py"]))

# 可选：在 connect() 之前注册，才能收到握手后立即推送的 notification
c.on_notification("notifications/tools/list_changed", lambda p: print("工具变了", p))

init = c.connect(timeout=10.0)     # -> InitializeResult（幂等，重复调用不重复握手）
c.state                            # "disconnected"/"connecting"/"connected"/"closing"/"closed"
c.connected                        # bool（等价 state == "connected"）
init.server_info                   # {"name": ..., "version": ...}
init.capabilities.tools            # dict 或 None

tools = c.list_tools(timeout=10.0)              # List[ToolInfo]
res = c.call_tool("echo", {"message": "hi"}, timeout=60.0)   # CallResult

# 资源与提示模板（v3.2；服务器未声明对应能力时返回空列表）
resources = c.list_resources()                  # List[ResourceInfo]
templates = c.list_resource_templates()         # List[ResourceTemplateInfo]
content = c.read_resource("file:///a.txt")      # ReadResourceResult
prompts = c.list_prompts()                      # List[PromptInfo]
rendered = c.get_prompt("code_review", {"lang": "python"})   # GetPromptResult
rendered.as_messages()                          # [{"role": ..., "content": ...}]
rendered.as_messages(role_prefix=True)          # content 前缀 "[role] "（v3.3.6 起参数真正生效）
c.close()                          # 幂等：重复调用、多线程并发调用都安全
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
reg.list_resources()                   # Dict[str, List[ResourceInfo]]（v3.2）
reg.list_resource_templates()          # Dict[str, List[ResourceTemplateInfo]]（v3.2）
reg.list_prompts()                     # Dict[str, List[PromptInfo]]（v3.2）
reg.read_resource("demo", "file:///a.txt")              # ReadResourceResult（v3.2）
reg.get_prompt("demo", "review", {"lang": "py"})        # GetPromptResult（v3.2）

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
# action 可选：list_tools / call_tool / list_resources / read_resource / list_prompts / get_prompt
# （后 4 个为 v3.2 新增；入参另有 resource_uri / prompt_name）
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

### 10.5 坑：`_schema_to_pydantic` 只展开一层（v3.1 已修复）

v3.0 里嵌套 `object` 只会变成 `dict`、`array` 只变成裸 `list`，不给模型看子字段 / 元素类型。

**v3.1 已增强**：支持嵌套 `object` 递归展开、`array` 的 `items`（含对象数组）、`enum`、`default`（含 falsy 值）、数值 / 字符串 / 数组校验，并以 `test_schema_c1`–`c11` 锁定。

**v3.3.4 补记（同一函数的另一个坑）**：嵌套模型的缓存键此前只含 `str(properties)`，不含同级 `required`——同 schema 里两个同构但必填集合不同的嵌套对象会串用同一模型，双向都错（必填被降级放行 / 可选被强制拦截，探针 P-C 实证）；现键改为整个 prop 的规范化序列化，以 `test_schema_c12` 锁定。

> 仍存的一点边界：粗粒度模式下 `_list_tools_text()` 只列一层参数，嵌套字段的细节不会出现在文本描述里——此时优先看 server 自己给的 `description`。

### 10.6 坑：`artifact` 只留 text，图片不传（部分缓解）

`MCPToolAdapter.run()` 里：

```python
return ToolResult.ok(text, artifact={"content": [c for c in result.content if c.get("type") == "text"]})
```

`CallResult.as_text()` 遇到 `image` 只渲染一行占位文字。**默认链路仍不会把图片本体塞给模型。**

**v3.2 缓解**：`CallResult` 新增 `images()`，把 `image` 类内容块提取成 `[{"mime": ..., "data": ...}]`；`as_text()` 的 `image` 渲染为 `[图片：{mime}, 长度={n}]`（带长度信息）。

**建议**：多模态需求用 `result.images()` 取到 base64 后，自行业务侧转成本框架的图片消息。

### 10.7 坑：子进程的 stderr 没人读（v3.1 已修复）

v3.0 里 `StdioTransport.connect()` 起了 `stderr=PIPE` 但从不读取，子进程 stderr 写满管道（Windows 约 4KB / Unix 约 64KB）后会阻塞，表现为「工具调用莫名其妙卡住」。

**v3.1 已修复**：新增 stderr 后台读线程持续消费，最近若干行存入环形缓冲，`get_last_stderr()` 可取用；子进程异常退出时，错误消息会自动附上 stderr 尾部。`test_transport_t1` / `t2` 已锁定「捕获 stderr」「大量 stderr 不阻塞」。

### 10.8 坑：`MCPClient._event` 建了没用（v3.1 已修复）

v3.0 里 `_event` 建了却从没被用到，`_request()` 全程用 `time.sleep(0.01)` 轮询该 dict。

v3.1 已改为**每个 pending 请求一个 `threading.Event`**：读线程收到响应后 `set()` 唤醒对应请求线程，彻底消除忙等，并删掉了这个从没用过的 `_event` 字段。v3.1 用内存假传输补了并发按 id 分发的回归测试（`test_client_b5`）。

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

### 10.11 坑：`resources` / `prompts` 支持（v3.2 已补齐）

v3.0 里 `ServerCapabilities` 解析了 `resources` / `prompts` 但代码从不使用，遇到「以资源为主」的 server 会没东西可用。

**v3.2 已补齐**：`MCPClient` 新增 `list_resources` / `list_resource_templates` / `read_resource` / `list_prompts` / `get_prompt`；服务器未声明对应能力时这些方法自动返回空列表（不发请求），不会报错。粗粒度 `MCPServerTool` 也补了对应 action。

**v3.3.3 修正**：`list_resource_templates` 此前误调 `resources/list`，对合规服务器恒为空；现改调规范的 `resources/templates/list`，服务器返回 -32601 时降级空列表。接真实 server 时模板能拿到的前提从此成立。

> 仍未做：`logging` / `sampling` / `roots`。接 server 前可先看它 `capabilities` 里声明了哪几类。

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
parts = [_unquote_token(p) for p in shlex.split(command_str, posix=(sys.platform != "win32"))]
```

Windows 路径里的反斜杠在 `posix=True` 下会被当转义符吃掉。所以 Windows 必须 `posix=False`。

但 `posix=False` 会把**引号字符保留在 token 里**——修复前 `'"D:/Program Files/node.exe" server.js'` 切分出的程序名是带字面引号的 `"D:/Program Files/node.exe"`，`Popen` 必然 `FileNotFoundError`（v3.3.2 修复：切分后用 `_unquote_token` 剥离 token 首尾成对双引号并还原 `\"`，`test_transport_t8` 已锁定）。

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

`echo_server.py` / `notify_server.py` 是纯标准库、零依赖、无第三方包的示例 server，`tests/mcp/test_mcp.py` 在 Windows 上可直接 `python tests\mcp\test_mcp.py` 跑通，不需要额外配置。

### 10.13 坑：`MCPRegistry._closed` 只写不读（v3.3.6 已修复）

`close_all()` 里 `self._closed = True`，但**没有任何地方检查过它**。所以 `close_all()` 之后再 `connect()` 一个新 server，仍然能成功——`_closed` 更像是个语义标记。

**v3.3.6 已修复**：`connect()` 开头检查 `if self._closed: raise MCPError(...)`，关闭后拒绝再连并提示“请新建 MCPRegistry 实例”；`connect_all()` 的单服失败降级（`logger.warning`）与 `connect_mcp_server` 工具的 `try/except` 均沿用现有入口，不会把主流程打挂。以 `test_polish_p2` 锁定（含反向对照：**未** `close_all()` 时单个 `close()` 后重连仍然合法）。

### 10.14 坑：调用超时是固定 60 秒

`MCPClient.call_tool(timeout=60.0)` 是硬编码默认值，**没有从配置读**。有些工具（比如跑一个长脚本）会超过 60 秒。

**建议**：目前只能改源码或用 `MCPToolAdapter` 时自己调 `client.call_tool(..., timeout=xxx)`。如果这是常见需求，考虑把超时提成 `MCPConfig` 的字段。

### 10.15 坑：声明式配置 `command_str` 形态的 env 被静默丢弃（v3.3.5 已修复）

`MCPConfig` 有 `env` 字段、`StdioTransport` 也支持 env，但 `add_config` 的 stdio `command_str` 分支此前只传 `name / command_str / cwd`，`config.env` 在中间层被丢弃——不报错、不日志，声明式配的环境变量永远到不了子进程（命令式 `command` 分支反而正常，因此更容易误判为“都支持”）。探针 P-D 实证。

**v3.3.5 已修复**：`make_stdio_transport` / `add_stdio_cmd` / `MCPConfig.stdio_cmd` 三层增加可选 `env=None` 并透传，以 `test_registry_r1` 锁定。

### 10.16 坑：`list_changed` 刷新 handler 在读线程里同步执行 → 自死锁（v3.3.6 已修复）

这类 bug 隐蔽在“功能看起来已实现”之下：`connect()` 确实注册了 `notifications/*/list_changed` 处理器，代码路径也完整，但：

```text
读线程 recv() → 发现 notification → 同步回调 handler
                                          ↓
                              client.list_tools(timeout=5.0)
                                          ↓
                    新请求的响应必须经读线程消费，而读线程正阻塞在 handler 里
                                          ↓
                              白等到 5s 超时 → “刷新工具列表失败”
```

探针实证：修复前刷新永远报「请求 tools/list 超时」，**v3.2 写的自动刷新功能在任何传输上从未生效过**（包括单测能跑通的内存假传输）。修法很简单也很关键：handler 只做“派活”，刷新 body 丢给独立 daemon 线程（`_schedule_refresh`）。

> **可复用的教训**：在“回调运行在唯一消费者线程”的架构里，回调内部**绝不能再走同一条需要该线程参与的同步请求链路**，否则不是慢而是必然超时；且这类失败只记 warning，不抓探针很容易当作“已实现”。对应回归用例见 §11.1 的 `test_polish_p5`。

### 10.17 其余四处小坑（v3.3.6 一并打磨）

| 位置 | 坑 | 修复 |
|------|----|------|
| `models.py::GetPromptResult.as_messages(role_prefix=True)` | 参数**从未被读取**，调用方以为生效 | 让 `True` 真正拼上 `[role] ` 前缀；默认值 `True`→`False`（因原本无效，无参输出逐字节不变），`test_polish_p1` |
| `client.py` 顶层 `from .errors import ... MCPProtocolError` | 全文件未使用的**死 import** | 删引（`errors.py` 仍定义并对外导出该类，其它传输层在用） |
| `registry.py::_unregister_one` 兜底扫描 | 前缀硬编码 `"__"`，自定义 `prefix_sep` 的残留清不掉 | 改用 `_prefix_sep` 记录的该 server 实际分隔符，`test_polish_p3` |
| `transports/streamable_http_transport.py::_json_resolved` | 只增不清，长驻进程慢速泄漏（每响应 id 一条） | `recv()` 消费到响应时 `discard(msg_id)`（取 `_resolve_lock`），`close()` 整表 clear 兜底，`test_polish_p4` |

这四项都不影响正确性（死参数 / 死代码 / 极端路径 / 微泄漏），但属于“读到就会误判”的坑，故一并清掉并用 `test_polish_p*` 锁定。

---

## 十一、测试与验证

### 11.1 现状

`tests/mcp/test_mcp.py` **65 项用例，全部通过**（v3.3.6）：前 10 个走子进程 + 零依赖的 `echo_server.py`；`b1`–`b10` 走**内存假传输** `FakeTransport`，覆盖 v3.1 客户端内部机制；`t1`–`t8` 覆盖传输层健壮性（stderr / SSE / 引号剥离）；`c1`–`c12` 覆盖复杂 Schema 转换；`d1`–`d11` 覆盖 v3.2 的 Resources / Prompts；`r1` 覆盖 Registry 声明式配置 env 透传；`p1`–`p5` 覆盖 v3.3.6 的 P4 打磨与 `list_changed` 刷新根因。

原有 10 项（子进程 + `echo_server.py`）：

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

v3.1 新增 10 项（内存假传输，不起子进程）：

| # | 用例 | 验证点 |
|---|------|--------|
| b1 | `test_client_b1_state_machine_and_idempotent_connect` | `state` 状态机 + `connect()` 幂等（第二次不重复握手） |
| b2 | `test_client_b2_close_idempotent` | `close()` 幂等；未连接时 `close()` 也安全 |
| b3 | `test_client_b3_notification_dispatch` | `on_notification()` 收到服务器推送的 notification |
| b4 | `test_client_b4_ignore_notification_and_refuse_server_request` | 未注册 notification 被忽略；server→client 请求回 `-32601` |
| b5 | `test_client_b5_concurrent_requests_demux_by_id` | 6 路并发 `call_tool` 按 id 正确分发、不串号 |
| b6 | `test_client_b6_close_wakes_pending_request` | `close()` 立即唤醒在途请求而非干等超时 |
| b7 | `test_client_b7_timeout_message_and_unknown_response` | 超时报「超时」而非「已关闭」；未知 id 响应被忽略 |
| b8 | `test_client_b8_connect_failure_rolls_back_state` | `transport.connect()` 抛错 → 状态回滚为 `disconnected` |
| b9 | `test_client_b9_handshake_timeout_rolls_back` | 握手超时 → 状态回滚 + 报「超时」 |
| b10 | `test_client_b10_handler_exception_isolated` | handler 抛异常不打断读线程，后续请求照常 |

v3.1 传输层新增 8 项（`t1`–`t8`，t8 为 v3.3.2 缺陷回归）：

| # | 用例 | 验证点 |
|---|------|--------|
| t1 | `test_transport_t1_stdio_captures_stderr_and_reports_exit` | stderr 被捕获、子进程异常退出报错带 stderr 尾部 |
| t2 | `test_transport_t2_large_stderr_does_not_deadlock` | 大量 stderr 输出不阻塞子进程 |
| t3 | `test_transport_t3_stdio_send_after_exit_raises_immediately` | 子进程退出后 `send()` 立即抛错 |
| t4 | `test_transport_t4_sse_normal_flow` | SSE 正常收发 |
| t5 | `test_transport_t5_sse_disconnect_wakes_pending` | SSE 断连后等待中的请求被唤醒报错 |
| t6 | `test_transport_t6_sse_connect_bad_url_fails_fast` | 坏 URL 快速失败 |
| t7 | `test_transport_t7_stdio_close_idempotent_and_no_zombie` | `close()` 幂等且无僵尸进程 |
| t8 | `test_transport_t8_make_stdio_transport_strips_wrapping_quotes` | Windows 残留引号剥离（v3.3.2 修复回归）：token 逐字节等于原路径 + 真实拉起握手 + 空命令友好报错 |

v3.1 Schema 新增 11 项（`c1`–`c11`）+ v3.3.4 回归 1 项（`c12`）：

| # | 用例 | 验证点 |
|---|------|--------|
| c1 | `test_schema_c1_array_with_items_type` | `array` + `items` → `list[T]` |
| c2 | `test_schema_c2_nested_object` | 嵌套 `object` 递归展开 |
| c3 | `test_schema_c3_enum_type` | `enum` → `Literal[...]` |
| c4 | `test_schema_c4_default_values_including_falsy` | `default` 含 falsy 值不再误判为 `None` |
| c5 | `test_schema_c5_numeric_validation` | 数值校验 `ge`/`le`/`gt`/`lt` |
| c6 | `test_schema_c6_string_validation` | `minLength`/`maxLength`/`pattern` |
| c7 | `test_schema_c7_anyof_null_becomes_optional` | `anyOf` 含 `null` → `Optional[T]` |
| c8 | `test_schema_c8_array_validation_min_max_items` | `minItems`/`maxItems` |
| c9 | `test_schema_c9_unknown_type_fallback_to_any` | 未知类型降级为 `Any` |
| c10 | `test_schema_c10_array_of_nested_objects` | 对象数组递归 |
| c11 | `test_schema_c11_empty_schema_compatibility` | 空 schema 兼容 |
| c12 | `test_schema_c12_nested_objects_same_shape_different_required` | 嵌套对象缓存键含 `required` 等全量语义；双向错乱回归 + 同构缓存仍复用（v3.3.4） |

v3.3.5 Registry env 透传 1 项（`r1`）：

| # | 用例 | 验证点 |
|---|------|--------|
| r1 | `test_registry_r1_command_str_env_transparent_to_subprocess` | `add_stdio_cmd(env=)` / `MCPConfig.stdio_cmd(env=)` → `add_config` 透传到 transport；不传 env 行为不变；带 env 真实拉起握手成功 |

v3.3.6 P4 批量打磨 5 项（`p1`–`p5`）：

| # | 用例 | 验证点 |
|---|------|--------|
| p1 | `test_polish_p1_as_messages_role_prefix_takes_effect` | `as_messages()` 无参输出与旧版一致；`role_prefix=True` 时 content 带 `[role] ` 前缀而 role 字段仍独立 |
| p2 | `test_polish_p2_closed_registry_blocks_connect` | `close_all()` 后 `connect()` 抛 `MCPError`、`connect_all()` 只降级 warning 不产生新连接；反向对照：单个 `close()` 后重连仍合法 |
| p3 | `test_polish_p3_unregister_fallback_honors_custom_prefix_sep` | 抹掉 `_registered_names` 记账只走兜底扫描，验证 `prefix_sep="_"` 注册的残留能被清 |
| p4 | `test_polish_p4_json_resolved_pruned_on_recv` | 响应被 `recv()` 消费后 `_json_resolved` 里对应 id 立即移除（不起真服务，直接推队列） |
| p5 | `test_polish_p5_list_changed_refresh_works_and_updates_coarse_gate` | **两层**：① `list_changed` 刷新不再在读线程自死锁（缓存真的拉到新列表）；②粗粒度门脸索引同步，新增子工具可直接 `call_tool` |

v3.2 新增 10 项（`d1`–`d10`，Resources / Prompts）+ v3.3.3 回归 1 项（`d11`）：

| # | 用例 | 验证点 |
|---|------|--------|
| d1 | `test_d1_list_resources` | `list_resources()` |
| d2 | `test_d2_list_resource_templates` | `list_resource_templates()` |
| d3 | `test_d3_read_text_resource` | `read_resource()` 读文本资源 |
| d4 | `test_d4_read_nonexistent_resource_raises` | 读不存在的资源抛错 |
| d5 | `test_d5_list_prompts` | `list_prompts()` |
| d6 | `test_d6_get_prompt_with_arguments` | `get_prompt()` 带参数渲染 |
| d7 | `test_d7_get_prompt_missing_required_arg_raises` | 缺必填参数报错 |
| d8 | `test_d8_registry_caches_resources_prompts_on_connect` | Registry 连接时缓存资源 / 提示 |
| d9 | `test_d9_coarse_tool_supports_resource_prompt_actions` | 粗粒度工具支持 4 个新 action |
| d10 | `test_d10_unsupported_server_returns_empty_lists` | 无能力服务器返回空列表 |
| d11 | `test_d11_list_resource_templates_uses_spec_method` | 模板走规范方法 `resources/templates/list`；-32601 降级空列表；`MCPCallError.error_code`（v3.3.3） |

> 说明：测试目录下暂无「ReActAgent + MCP」的自动化集成用例（`tests/trace/test_trace.py` 仅以 `enable_mcp=False` 构造 Agent）。Agent 侧的 MCP 接入（`enable_mcp` / `mcp_expand_mode` / 动态 connect）目前由 `examples/mcp/mcp_react_agent_demo.py` 手工演示覆盖——这是一处可补的覆盖缺口。

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

v3.1 的客户端能力（notification / 状态 / 幂等 / 并发）可直接跑示例，离线零依赖：

```powershell
python examples\mcp\mcp_client_features_demo.py
```

预期：握手后收到 `notifications/message` 与 `notifications/tools/list_changed`；4 路并发调用各回各的结果；二次 `connect()` 幂等（不重复握手）、二次 `close()` 幂等。

v3.2 的资源 / 提示模板能力可跑：

```powershell
python examples\mcp\mcp_resources_prompts_demo.py
```

预期：列出资源与模板、读取文本资源、列出提示模板、带参数渲染提示模板。传输层健壮性与复杂 Schema 分别可跑 `mcp_transport_stability_demo.py` / `mcp_complex_schema_demo.py`。

### 11.3 检查清单

- [ ] `enable_mcp=True` 时才连外部 server；
- [ ] `mcp_expand_mode` 与工具数量匹配（>20 个考虑 `coarse`）；
- [ ] 工程里 `close()` 被调到（或依赖 `atexit` / `__del__`）；
- [ ] `logging` 有配（否则 `connect_all` 的失败你看不见）；
- [ ] 工具名不和本地工具撞（前缀 `{server}__` / `mcp__`）；
- [ ] 接 npx 类 server 时用了 `.cmd` 或 `cmd /c`；
- [ ] 子进程 stderr 不会写爆（见 [10.7](#107-坑子进程的-stderr-没人读)）；
- [ ] 需要 SSE 时装了 `requests`；
- [ ] 若使用 `resources` / `prompts`，确认 server 的 `capabilities` 已声明对应能力（未声明时相关方法返回空列表）。

---

## 十二、下一步演进路线

编号与《演进路线图.md》模块 D 保持一致（D 系列）。

### 12.1 已交付

| 编号 | 任务 | 完成版本 | 落点 |
|------|------|---------|------|
| ✅ **D1** | `MCPClient`（stdio transport） | v3.0.0 | `client.py` + `transports/`（另含 SSE） |
| ✅ **D2** | MCP tool → `ToolBase` 适配器 | v3.0.0 | `tool.py`：`MCPToolAdapter`（细粒度）+ `MCPServerTool`（粗粒度） |
| ✅ **D4** | `.mcp.json` 配置加载 | v3.0.0 | `MCPConfig` 声明式配置 + `MCPRegistry` |
| ✅ **D10** | `stderr` 消费 | v3.1.0 | `aa79b44` 交付：stderr 后台读线程 + `get_last_stderr()`，不再阻塞子进程 |
| ✅ **D7** | `resources` / `prompts` 支持 | v3.2.0 | `bff658e` 交付：`MCPClient` 五个资源 / 提示方法 + Registry 连接缓存与 `list_changed` 自动刷新 + 粗粒度 4 个新 action（刷新链路的读线程自死锁 v3.3.6 修完，见 [10.16](#1016-坑list_changed-刷新-handler-在读线程里同步执行--自死锁v336-已修复)） |
| ✅ **D12** | `close_all()` 后用 `_closed` 拦截再连接 | v3.3.6 | `registry.connect()` 开头检查 `_closed`，`test_polish_p2` 锁定（见 [10.13](#1013-坑mcpregistry_closed-只写不读v336-已修复)） |
| ✅ **D6** | `streamable-http` 传输 | v3.3.0 | `transports/streamable_http_transport.py`：`StreamableHTTPTransport` + `MCPRegistry.add_streamable_http()` + `MCPConfig.streamable_http()` + 测试 `test_streamable_http_e1~e4`（见 [7.4](#74-transports传输层包)） |

### 12.2 待办（按优先级）

| 编号 | 任务 | 优先级 | 依赖 | 一句话说明 |
|------|------|--------|------|-----------|
| **D8** | 图片内容透传（多模态） | P2 | — | `CallResult.images()` 已能取出图片，仍需接入多模态链路（[10.6](#106-坑artifact-只留-text图片不传)） |
| **D9** | 连接自动重连 / 心跳保活 | P2 | — | 生命周期已有（挂成 Agent 成员 + `atexit` 兜底），断线后单次调用失败可恢复的自动重连未做 |
| **D11** | 超时可配置 | P2 | — | 当前调用超时固定 60 秒，提到 `MCPConfig` 上（[10.14](#1014-坑调用超时是固定-60-秒)） |
| **D13** | 按 server 分别指定展开模式 | P2 | — | 细粒度 / 粗粒度目前全局统一（[5.4](#54-怎么选)） |

> v3.1 已完成 `stderr` 独立读取线程与 `_schema_to_pydantic` 嵌套展开，v3.2 已完成 `resources` / `prompts`，v3.3.0 已完成 D6（`streamable-http`），v3.3.6 已完成 D12（`_closed` 拦截），故已从待办表中移除。

---

## 十三、参考

**仓库内**

- `fast_agents/mcp/`（7 个文件）
- `fast_agents/tools/builtin/mcp_tool.py`（三个管理工具）
- `fast_agents/agents/react_agent/react_agent.py`（`_init_mcp` / `close` / `_safe_close`）
- `examples/mcp/`（6 个 server：`echo_server.py` / `notify_server.py` / `crash_server.py` / `sse_test_server.py` / `complex_schema_server.py` / `resources_prompts_server.py` + 6 个演示脚本）
- `tests/mcp/test_mcp.py`
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
- [ ] `python tests\mcp\test_mcp.py` 48/48 通过；
- [ ] `python tests\agents\test_react_agent.py` 全过（含 MCP 用例）；
- [ ] `enable_mcp=False` 时不会创建任何子进程 / 网络连接；
- [ ] `fine` / `coarse` 两种 `register_into` 都能正确展开，且切换时**不留残留工具**；
- [ ] 新增 `expand` 取值时同步改了 `_tool_names_for` 和 `_expand_one` 的校验；
- [ ] 新增 `TransportBase` 实现时实现了全部四个抽象方法；
- [ ] 新增异常类型时挂在 `MCPError` 下，并在本文件 [6.7](#67-异常体系errorspy) 登记；
- [ ] Windows 上仍能用 `python tests\mcp\test_mcp.py` 跑通（无黑框、无乱码、无 `FileNotFoundError`）；
- [ ] `mcp/__init__.py` 的 `__all__` 与新导出保持一致；
- [ ] `close()` / `_safe_close()` / `__del__` 三重回收仍有效（无子进程泄漏）；
- [ ] 本文档 [六章](#六核心数据模型)、[九章](#九api-速查) 已同步；
- [ ] 本文档 [一、更新历史](#一更新历史) 加了一行。
```
