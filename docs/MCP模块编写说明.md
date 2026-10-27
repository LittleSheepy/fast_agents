# MCP 模块编写说明（fast_agents）

> 适配你的代码风格（对齐 `tools/` `memory/` `skills/` 的分层），Windows 兼容，零新增依赖（SSE 可选 requests）。

---

## 一、模块架构

```
fast_agents/mcp/               ← 单数目录，对齐 tools/ memory/ skills/
├── __init__.py                模块出口
├── errors.py                  三层异常：MCPConnectionError / MCPProtocolError / MCPCallError
├── models.py                  pydantic 模型：ToolInfo / CallResult / InitializeResult / ServerCapabilities
├── transports.py              Transport 抽象 + StdioTransport + SSETransport + make_stdio 便捷工厂
├── client.py                  单个 MCP server 客户端（握手 / tools/list / tools/call）
├── registry.py                多 server 注册表：add_stdio / add_sse / connect_all / register_into / close_all
└── tool.py                    两种工具适配器：MCPToolAdapter（细） + MCPServerTool（粗）

fast_agents/tools/builtin/mcp_tool.py    ← MCP 通用管理工具（与 memory_tool.py 同层级、同风格）
├── ListMcpServersTool         list_mcp_servers
├── ConnectMcpServerTool       connect_mcp_server（运行时动态连接，自动展开工具并刷新 prompt）
├── DisconnectMcpServerTool    disconnect_mcp_server
└── build_mcp_tools(registry, tool_registry, refresh_prompt)   工厂函数
```

数据流：

```
ReActAgent
   ↓  tool_registry.invoke_tool("fs__read_file", {...})
ToolRegistry
   ↓  命中 MCPToolAdapter 实例
MCPToolAdapter.run()
   ↓  self.client.call_tool(...)
MCPClient._request("tools/call", ...)   ← JSON-RPC 2.0
   ↓  self.transport.send/recv
Transport（StdioTransport / SSETransport）
   ↓
子进程 / HTTP SSE 远端 server
```

核心设计原则：**MCP 工具对模型来说就是普通 ToolBase 实例**——注册进 ToolRegistry 后和本地 `python_calculator` 没有区别，LLM 无感知。

---

## 二、两种展开方式对比

| 维度 | `MCPToolAdapter`（细粒度，`expand=True`，**推荐**） | `MCPServerTool`（粗粒度，`expand=False`） |
|---|---|---|
| 注册粒度 | 每个远端工具一个 ToolBase | 每个 server 一个聚合工具 |
| 工具名 | `<server>__<tool>`（如 `fs__read_file`） | `mcp_<server>`（如 `mcp_fs`） |
| args_schema | 从远端 `inputSchema` 动态生成 pydantic 模型，**每个字段带类型/描述/enum/必填** | 单个 `payload: dict` 参数，**靠你 `react_agent.py` 的 auto_expand 钩子再展开一层** |
| 多参数工具 | ✅ 原生支持，模型能准确传 `message/prefix/repeat` | ⚠️ 只有单 `input` 字符串，多参数会丢 |
| schema 导出 to_openai/to_anthropic | ✅ 完整 | 仅描述级 |
| 模型调用准确率 | 高 | 一般 |

粗粒度模式保留是为了兼容你 `react_agent.py` L139-155 已写好的 `auto_expand` 钩子——你哪天想删那段钩子，只要始终走 `expand=True`，就不会受影响。

---

## 三、快速上手（Windows PowerShell）

```powershell
cd D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents

# 1. 跑单元测试（9/9 全过才算装好）
python test\test_mcp.py

# 2. 跑示例（连接内置 echo_server，验证链路）
python examples\mcp\mcp_quickstart.py
```

### 最小可运行例子

```python
import sys
from fast_agents import LLM, ReActAgent, MCPRegistry
from fast_agents.tools import builtin
from fast_agents.tools.builtin.mcp_tool import build_mcp_tools

llm = LLM(model="...", api_key="...", base_url="...")

mcp = MCPRegistry()
# 启动本地 echo demo（零依赖，验证用）
mcp.add_stdio_cmd("demo", f'"{sys.executable}" "examples/mcp/echo_server.py"')
# 生产可换成官方 server：
# mcp.add_stdio("fs", "uvx", ["mcp-server-filesystem", r"D:\00sheepy"])
# mcp.add_sse("remote", "https://your-host/mcp")

agent = ReActAgent(name="助手", llm=llm, enable_memory=True, enable_skills=True)
agent.add_tool(builtin.CalculatorTool())

mcp.connect_all()
mcp.register_into(agent.tool_registry)           # 细粒度展开（推荐）
agent.refresh_system_prompt()

# 【可选】让模型能在运行时自己管理 MCP 连接：
for t in build_mcp_tools(mcp, agent.tool_registry, refresh_prompt=agent.refresh_system_prompt):
    agent.add_tool(t)
agent.refresh_system_prompt()

print(agent.run("帮我算 3*4+1"))
mcp.close_all()
```

---

## 四、Windows 专项

1. **路径空格**：用 `add_stdio_cmd(name, command_str)` 传整段命令字符串（如 `"python" "D:/my dir/server.py" "arg with space"`），内部会用 `shlex.split(command_str, posix=False)` 按 Windows 规则拆。避免直接在命令行/字符串里裸拼含空格的路径。
2. **防黑框**：启动子进程用了 `CREATE_NO_WINDOW`（0x08000000），不会弹出黑色命令行窗口。
3. **编码**：子进程强制注入 `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1`，并在 stderr 读取时 `errors="replace"`，避免 GBK 控制台导致的乱码。
4. **SSE 依赖**：`SSETransport` 只有用到时才会 import `requests`，没装会给清晰报错 `pip install requests`；stdio 模式零依赖。

---

## 五、写一个自己的 MCP Server（零依赖模板）

新建 `my_server.py`：

```python
import json, sys, math

def _send(msg): sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n"); sys.stdout.flush()
def _recv():
    line = sys.stdin.readline()
    return json.loads(line) if line else None

TOOLS = [{
    "name": "square",
    "description": "计算一个数的平方",
    "inputSchema": {
        "type": "object",
        "properties": {"x": {"type": "number", "description": "要平方的数"}},
        "required": ["x"],
    },
}]

def main():
    while True:
        msg = _recv()
        if msg is None: break
        mid, method = msg.get("id"), msg.get("method", "")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "my-server", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            if name == "square":
                x = float(args["x"])
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"{x}² = {x*x}"}],
                    "isError": False,
                }})
            else:
                _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown {name}"}})
        elif mid is not None:
            _send({"jsonrpc": "2.0", "id": mid, "result": {}})  # ping 等

if __name__ == "__main__":
    main()
```

接入：

```python
mcp.add_stdio_cmd("my", f'"{sys.executable}" "my_server.py"')
mcp.connect_all()
mcp.register_into(agent.tool_registry)
# 模型就能调用 my__square(x=5) → 25
```

---

## 六、API 速查

### MCPRegistry

| 方法 | 说明 |
|---|---|
| `add_stdio(name, command, args, env=None, cwd=None)` | 注册 stdio server |
| `add_stdio_cmd(name, command_str)` | 注册 stdio server，传整段命令字符串 |
| `add_sse(name, url, headers=None, timeout=30)` | 注册 SSE server |
| `connect(name, timeout=10)` | 连接单个 server，返回 `MCPClient` |
| `connect_all(timeout=10)` | 批量连接 |
| `register_into(registry, prefix_sep="__", expand=True)` | 把工具展开到 ToolRegistry |
| `close(name)` / `close_all()` | 断开 |
| `list_tools(server=None)` | 查询已发现的工具 |

### MCPClient

| 方法 | 说明 |
|---|---|
| `connect(timeout=10)` → `InitializeResult` | 握手（initialize + initialized 通知） |
| `list_tools(timeout=10)` → `List[ToolInfo]` | 列工具 |
| `call_tool(name, arguments, timeout=60)` → `CallResult` | 调用工具；`CallResult.as_text()` 拿拼接后的文本 |
| `close()` | 关闭 |

### builtin 管理工具（`fast_agents.tools.builtin.mcp_tool`）

| 工具名 | 作用 |
|---|---|
| `list_mcp_servers` | 查看已连接的 MCP server 及其工具 |
| `connect_mcp_server` | 运行时用命令串/SSE URL 动态连一个新 server |
| `disconnect_mcp_server` | 断开并从 ToolRegistry 中移除相关工具 |

工厂：`build_mcp_tools(registry, tool_registry, refresh_prompt=agent.refresh_system_prompt)`。

---

## 七、源码逐文件导读

### 1. `errors.py`
三层异常，方便上层按粒度 catch：连接层（进程启动失败/握手超时）、协议层（消息不符合 JSON-RPC）、业务层（远端工具报错）。

### 2. `models.py`
纯 pydantic 数据类，无逻辑。注意 `ToolInfo.has_structured_input`——如果远端 tool 的 schema 只有一个 `input` 字段，我们就认为它是单入参"伪结构化"工具，`MCPToolAdapter` 会回退到单字符串参数。

### 3. `transports.py`
- `Transport` ABC 定义 `connect/send/recv/close` 四个方法；
- `StdioTransport`：子进程 + 后台读线程，按 `\n` 拆帧进队列；Windows 用 `CREATE_NO_WINDOW` + UTF-8 环境变量；
- `SSETransport`：用 `requests` 流式读 SSE，解析 `event:`/`data:` 帧；先收 `endpoint` 事件拿到 POST URL，后续 send 走 HTTP POST；
- `make_stdio(command_str)`：便捷工厂，自动判断 `posix=False` 适配 Windows。

### 4. `client.py`
核心 RPC 逻辑：`_next_id()` 生成递增 id；`_request()` 发消息→轮询 `_request_id_to_response` 字典等响应；`_read_loop` 后台线程把 `id` 对应的响应塞回字典，遇到 server 主动发的方法请求返回 `-32601 Method not found`（合规但极简）。

### 5. `registry.py`
多 server 门面。`register_into(registry, expand=True)` 的细粒度路径会给每个远端 tool 构造 `MCPToolAdapter` 并注册到你的 ToolRegistry；`expand=False` 则构造一个 `MCPServerTool`。工具名用 `<server><sep><tool>`（默认 `__`）避免冲突。

### 6. `tool.py`
- `_schema_to_pydantic(schema)`：把 JSON Schema 转成 pydantic 模型（支持 type/array/object/string/integer/number/boolean/enum/default/required）——这是细粒度能准确展开多参数工具的关键；
- `MCPToolAdapter`：单工具适配器，`run(**kwargs)` 直接把 kwargs 当 arguments 传给远端；
- `MCPServerTool`：聚合工具，暴露 `auto_expand=True` 和 `_available_tools` 属性，兼容你 `react_agent.py` 已有的钩子；`run` 接收 `{"action": "list_tools"}` 或 `{"action": "call_tool", "tool_name":, "arguments":}` 两种 payload。

### 7. `fast_agents/tools/builtin/mcp_tool.py`
完全对齐 `memory_tool.py` 的风格（三工具 + 工厂函数 + `ToolBase` 子类 + pydantic Input），注入 MCPRegistry 和 ToolRegistry 就能工作，不耦合 ReActAgent——通过 `refresh_prompt` 回调通知 prompt 刷新。

---

## 八、测试

`test/test_mcp.py` 共 9 项，覆盖：
1. 握手与工具发现
2. echo/calculate 工具调用
3. 不存在工具抛 MCPCallError
4. MCPToolAdapter args_schema 生成（多参数、required/default）
5. MCPServerTool auto_expand 兼容
6. MCPRegistry 批量展开到 ToolRegistry
7. builtin list 工具
8. builtin connect/disconnect 动态链路（真启动子进程再断开）
9. OpenAI/Anthropic schema 导出

运行：`python test/test_mcp.py`

---

## 九、扩展路线（暂未实现，需要时再加）

- **资源（resources）**：MCP 还定义了 `resources/list`/`resources/read`，可做成 `read_mcp_resource` builtin 工具；
- **提示模板（prompts）**：`prompts/list`/`prompts/get`，用于从远端拉提示词模板；
- **进度/日志**：`$/progress`、`logging/setLevel` 通知；
- **tools/listChanged** 增量更新：当前每次 connect 全量 list，长连接场景可监听 `notifications/tools/list_changed` 自动重 list；
- **采样回调（sampling/createMessage）**：允许 server 反向调用 LLM，需要在 `_read_loop` 里处理 server 发来的请求，涉及 LLM 回调注入，按需再加。

---

## 十、常见问题排查

| 现象 | 原因 / 解决 |
|---|---|
| `找不到命令：uvx` | 没装 uv；先 `pip install uv` 或换成 `npx -y @modelcontextprotocol/server-xxx` |
| Windows 弹出黑框 | 确认用的是本模块的 `StdioTransport`（已加 CREATE_NO_WINDOW）；如果你自己构造 subprocess 别忘了加 `creationflags=0x08000000` |
| 中文乱码 | 已设 PYTHONIOENCODING=utf-8；若 server 自己强制用 GBK，在 `add_stdio(env={"PYTHONIOENCODING":"gbk"})` 覆盖 |
| 工具被展开但模型总传错参数 | 大概率用了粗粒度模式（`expand=False`）；改成 `expand=True` 走细粒度，让模型看到完整 args_schema |
| 工具名冲突 | 不同 server 都叫 `read_file` 时会变成 `fs__read_file` / `github__read_file`；想换分隔符传 `prefix_sep="_"` |
| SSE 握手卡住 | 检查 URL 是否正确、是否需要 headers（如 Authorization）；服务端必须先 `event: endpoint` 再发 POST 地址 |
| connect 超时 | 子进程没起来或没往 stdout 输出 JSON-RPC 消息；先直接命令行手动启动 server 看有没有报错 |
