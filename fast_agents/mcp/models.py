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
                parts.append(f"[图片：{mime}]")
            elif t == "resource":
                parts.append(f"[资源] {item.get('resource', {}).get('uri', '')}")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(无输出)"


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
    transport: Literal["stdio", "sse"] = Field(description="传输类型")
    # stdio 必填
    command: Optional[str] = Field(default=None, description="stdio：可执行文件路径/命令名")
    args: List[str] = Field(default_factory=list, description="stdio：命令参数列表")
    env: Optional[Dict[str, str]] = Field(default=None, description="stdio：额外环境变量")
    cwd: Optional[str] = Field(default=None, description="stdio：子进程工作目录")
    command_str: Optional[str] = Field(default=None, description="stdio：直接传完整命令串（自动 shlex.split，Windows 用 posix=False）")
    # sse 必填
    url: Optional[str] = Field(default=None, description="sse：服务端地址")
    headers: Optional[Dict[str, str]] = Field(default=None, description="sse：HTTP 请求头")
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
    def stdio_cmd(cls, name: str, command_str: str, cwd: Optional[str] = None) -> "MCPConfig":
        return cls(name=name, transport="stdio", command_str=command_str, cwd=cwd)

