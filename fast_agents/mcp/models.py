"""MCP 数据模型"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ServerCapabilities(BaseModel):
    """服务端在 initialize 响应里声明的能力"""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class InitializeResult(BaseModel):
    """initialize 握手结果"""
    protocol_version: str = "2024-11-05"
    server_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    instructions: str = ""


class ToolInfo(BaseModel):
    """MCP 工具元信息（来自 tools/list）"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @property
    def has_structured_input(self) -> bool:
        props = self.input_schema.get("properties", {})
        return bool(props) and set(props.keys()) != {"input"}


class CallResult(BaseModel):
    """tools/call 返回结果"""
    content: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    structured_content: Optional[Any] = None

    def as_text(self) -> str:
        parts = []
        for item in self.content:
            t = item.get("type", "")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                mime = item.get("mimeType", "image")
                data = item.get("data", "")
                parts.append(f"[图片：{mime}, 长度={len(data)}]")
            elif t == "resource":
                res = item.get("resource", {})
                parts.append(f"[资源] {res.get('uri', '')}")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(无输出)"

    def images(self) -> List[Dict[str, str]]:
        """提取返回里的图片内容列表，每个元素含 mime/data(base64)。"""
        out = []
        for item in self.content:
            if item.get("type") == "image":
                out.append({
                    "mime": item.get("mimeType", "image/png"),
                    "data": item.get("data", ""),
                })
        return out

    def raw_content(self) -> List[Dict[str, Any]]:
        """返回 MCP 原始 content 列表（含 text/image/resource 等），供上层做多模态透传。"""
        return list(self.content)


# ==================== Resources（资源）相关模型 ====================


class TextResourceContents(BaseModel):
    """文本资源内容"""
    uri: str
    mimeType: str = "text/plain"
    text: str


class BlobResourceContents(BaseModel):
    """二进制资源内容"""
    uri: str
    mimeType: str = "application/octet-stream"
    blob: str  # base64


class ResourceInfo(BaseModel):
    """resources/list 返回的资源元信息"""
    uri: str
    name: str
    description: str = ""
    mimeType: Optional[str] = None
    size: Optional[int] = None


class ResourceTemplateInfo(BaseModel):
    """resources/list 返回的资源模板（参数化 URI）"""
    uriTemplate: str
    name: str
    description: str = ""
    mimeType: Optional[str] = None


class ReadResourceResult(BaseModel):
    """resources/read 返回结果"""
    contents: List[Dict[str, Any]] = Field(default_factory=list)

    def as_text(self) -> str:
        parts = []
        for c in self.contents:
            if "text" in c:
                parts.append(c.get("text", ""))
            else:
                mime = c.get("mimeType", "binary")
                uri = c.get("uri", "")
                parts.append(f"[二进制资源：{uri} ({mime})]")
        return "\n".join(parts) if parts else "(无内容)"


# ==================== Prompts（提示模板）相关模型 ====================


class PromptArgument(BaseModel):
    """提示模板参数定义"""
    name: str
    description: str = ""
    required: bool = False


class PromptMessage(BaseModel):
    """get_prompt 返回的单条消息"""
    role: Literal["user", "assistant"]
    content: Dict[str, Any]

    def as_text(self) -> str:
        t = self.content.get("type", "")
        if t == "text":
            return self.content.get("text", "")
        elif t == "resource":
            res = self.content.get("resource", {})
            return f"[资源] {res.get('uri', '')}"
        elif t == "image":
            return f"[图片：{self.content.get('mimeType', '')}]"
        return str(self.content)


class PromptInfo(BaseModel):
    """prompts/list 返回的提示模板元信息"""
    name: str
    description: str = ""
    arguments: List[PromptArgument] = Field(default_factory=list)


class GetPromptResult(BaseModel):
    """prompts/get 返回结果"""
    description: str = ""
    messages: List[PromptMessage] = Field(default_factory=list)

    def as_messages(self, role_prefix: bool = False) -> List[Dict[str, str]]:
        """转成标准 {role, content} 消息列表，可直接塞给 LLM。

        role_prefix（v3.3.6 修复：此前参数从未被使用，是死参数）：
        True 时在 content 前附加 "[role] " 前缀，适合把消息展平成纯文本日志的场景；
        False（默认）保持标准消息——与旧版无参调用逐字节一致。
        """
        out = []
        for m in self.messages:
            text = m.as_text()
            if role_prefix:
                text = f"[{m.role}] {text}"
            out.append({"role": m.role, "content": text})
        return out

    def as_text(self) -> str:
        return "\n".join(m.as_text() for m in self.messages)


class MCPConfig(BaseModel):
    """MCP server 的声明式配置，让用户不用手工调用 MCPRegistry.add_xxx 就能注册 server。

    用法示例::

        from fast_agents.mcp import MCPConfig
        agent = ReActAgent(
            ...,
            enable_mcp=True,
            mcp_configs=[
                MCPConfig.stdio("fs", "uvx", ["-y", "@modelcontextprotocol/server-filesystem", "D:/code"]),
                MCPConfig.sse("weather", "https://example.com/mcp/sse"),
                MCPConfig.stdio_cmd("calculator", "python calc_server.py"),
            ],
        )
    """

    name: str = Field(description="MCP server 别名")
    transport: Literal["stdio", "sse", "streamable_http"] = Field(description="传输类型")
    # stdio 必填
    command: Optional[str] = Field(default=None, description="stdio：可执行文件路径/命令名")
    args: List[str] = Field(default_factory=list, description="stdio：命令参数列表")
    env: Optional[Dict[str, str]] = Field(default=None, description="stdio：额外环境变量")
    cwd: Optional[str] = Field(default=None, description="stdio：子进程工作目录")
    command_str: Optional[str] = Field(default=None, description="stdio：直接传完整命令串（自动 shlex.split，Windows 用 posix=False）")
    # sse / streamable_http 必填
    url: Optional[str] = Field(default=None, description="sse/streamable_http：服务端地址")
    headers: Optional[Dict[str, str]] = Field(default=None, description="sse/streamable_http：HTTP 请求头")
    timeout: float = Field(default=30.0, description="连接超时秒数")

    @classmethod
    def stdio(cls, name: str, command: str, args: Optional[List[str]] = None,
              env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> "MCPConfig":
        return cls(name=name, transport="stdio", command=command, args=args or [], env=env, cwd=cwd)

    @classmethod
    def sse(cls, name: str, url: str, headers: Optional[Dict[str, str]] = None,
            timeout: float = 30.0) -> "MCPConfig":
        return cls(name=name, transport="sse", url=url, headers=headers, timeout=timeout)

    @classmethod
    def streamable_http(cls, name: str, url: str, headers: Optional[Dict[str, str]] = None,
                        timeout: float = 30.0) -> "MCPConfig":
        """声明式构造一个 Streamable HTTP 类型的 server（单端点 HTTP，MCP 2025 规范）。"""
        return cls(name=name, transport="streamable_http", url=url, headers=headers, timeout=timeout)

    @classmethod
    def stdio_cmd(cls, name: str, command_str: str, cwd: Optional[str] = None,
                  env: Optional[Dict[str, str]] = None) -> "MCPConfig":
        """声明式构造 command_str 形式的 stdio server（v3.3.5 新增可选 env 参数，默认 None 行为不变）。"""
        return cls(name=name, transport="stdio", command_str=command_str, cwd=cwd, env=env)

