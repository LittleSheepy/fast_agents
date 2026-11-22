#!/usr/bin/env python3
"""MCP 示例服务器：演示 Resources（资源）和 Prompts（提示模板）能力，零依赖 stdlib。

运行方式：python resources_prompts_server.py
客户端连接后可以：
- 调用 echo 工具
- 列出/读取静态资源和参数化资源
- 列出/渲染提示模板
"""

import json
import sys
import os

# 内置静态资源
STATIC_RESOURCES = {
    "docs://getting_started": {
        "name": "快速开始指南",
        "description": "Fast Agents 框架快速入门文档",
        "mimeType": "text/markdown",
        "text": """# Fast Agents 快速开始

## 安装
```bash
pip install fast-agents
```

## 5分钟创建你的第一个Agent
```python
from fast_agents import ReActAgent
agent = ReActAgent(model="qwen-plus")
result = agent.run("你好，请介绍一下你自己")
print(result)
```

## 核心概念
- ToolBase: 自定义工具基类
- ToolRegistry: 工具注册表
- MCPRegistry: MCP 服务器管理
- ReActAgent: 思考-行动-观察循环Agent
""",
    },
    "docs://mcp_guide": {
        "name": "MCP 使用指南",
        "description": "MCP 模块接入外部工具服务器教程",
        "mimeType": "text/markdown",
        "text": """# MCP 模块使用指南

## 细粒度展开（推荐）
每个MCP工具独立注册到工具表，模型调用准确率最高。

## 粗粒度展开
每个MCP服务器一个门脸工具，适合工具数量多的场景。

## 支持的能力
- Tools: 工具调用
- Resources: 资源读取
- Prompts: 提示模板
""",
    },
}

# 参数化资源模板
RESOURCE_TEMPLATES = [
    {
        "uriTemplate": "file://{path}",
        "name": "本地文件",
        "description": "读取当前目录下的文件内容",
        "mimeType": "text/plain",
    }
]

# 内置提示模板
PROMPTS = {
    "code_review": {
        "name": "code_review",
        "description": "代码审查提示模板，生成规范的代码审查意见",
        "arguments": [
            {"name": "language", "description": "编程语言", "required": False},
            {"name": "code", "description": "要审查的代码内容", "required": True},
        ],
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "请作为资深{language}工程师对以下代码进行审查，指出潜在bug、性能问题、代码风格问题并给出改进建议：\n\n{code}",
                }
            }
        ],
    },
    "explain_concept": {
        "name": "explain_concept",
        "description": "概念解释提示模板，用通俗易懂的方式解释技术概念",
        "arguments": [
            {"name": "concept", "description": "要解释的技术概念", "required": True},
            {"name": "level", "description": "解释深度：beginner/intermediate/expert", "required": False},
        ],
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "请用{level}水平的语言解释一下「{concept}」这个概念，尽量举例子说明。",
                }
            }
        ],
    },
}


def send_json(data):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    # 握手阶段
    line = sys.stdin.readline()
    if not line:
        return
    req = json.loads(line)
    if req.get("method") == "initialize":
        send_json({
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "resources-prompts-demo", "version": "1.0.0"},
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True, "subscribe": False},
                    "prompts": {"listChanged": True},
                },
                "instructions": "这是一个演示MCP Resources和Prompts能力的示例服务器",
            }
        })

    # initialized 通知
    line = sys.stdin.readline()
    if not line:
        return

    # 主循环
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        # ── 工具相关 ──
        if method == "tools/list":
            send_json({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "回显输入的内容，用于测试连通性",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string", "description": "要回显的内容"}},
                                "required": ["message"],
                            },
                        }
                    ]
                }
            })
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            if tool_name == "echo":
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Echo: {args.get('message', '')}"}],
                        "isError": False,
                    }
                })
            else:
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                })

        # ── 资源相关 ──
        elif method == "resources/list":
            resources_list = []
            for uri, res in STATIC_RESOURCES.items():
                resources_list.append({
                    "uri": uri,
                    "name": res["name"],
                    "description": res["description"],
                    "mimeType": res["mimeType"],
                })
            send_json({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": resources_list,
                    "resourceTemplates": RESOURCE_TEMPLATES,
                }
            })
        elif method == "resources/read":
            uri = params.get("uri", "")
            # 静态资源
            if uri in STATIC_RESOURCES:
                res = STATIC_RESOURCES[uri]
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [{
                            "uri": uri,
                            "mimeType": res["mimeType"],
                            "text": res["text"],
                        }]
                    }
                })
            # 参数化 file:// URI：读取本地文件
            elif uri.startswith("file://"):
                path = uri[len("file://"):]
                # 安全限制：只能读当前目录下的文件
                path = os.path.basename(path)
                if os.path.exists(path) and os.path.isfile(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read(2000)  # 最多读2KB
                        send_json({
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "contents": [{
                                    "uri": uri,
                                    "mimeType": "text/plain",
                                    "text": content,
                                }]
                            }
                        })
                    except Exception as e:
                        send_json({
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": -32000, "message": f"读取文件失败: {e}"},
                        })
                else:
                    send_json({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32000, "message": f"文件不存在: {path}"},
                    })
            else:
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": f"未知资源URI: {uri}"},
                })

        # ── 提示模板相关 ──
        elif method == "prompts/list":
            prompts_list = []
            for name, p in PROMPTS.items():
                prompts_list.append({
                    "name": p["name"],
                    "description": p["description"],
                    "arguments": p["arguments"],
                })
            send_json({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"prompts": prompts_list}
            })
        elif method == "prompts/get":
            prompt_name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            if prompt_name not in PROMPTS:
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown prompt: {prompt_name}"},
                })
                continue
            p = PROMPTS[prompt_name]
            # 检查必填参数
            for arg in p["arguments"]:
                if arg["required"] and arg["name"] not in args:
                    send_json({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32602, "message": f"缺少必填参数: {arg['name']}"},
                    })
                    break
            else:
                # 渲染消息模板（简单的 {var} 替换）
                messages = []
                for m in p["messages"]:
                    content_text = m["content"]["text"]
                    for k, v in args.items():
                        content_text = content_text.replace(f"{{{k}}}", str(v))
                    # 未提供的参数保留占位符
                    messages.append({
                        "role": m["role"],
                        "content": {"type": "text", "text": content_text},
                    })
                send_json({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "description": p["description"],
                        "messages": messages,
                    }
                })

        # 忽略其他通知
        elif "id" not in msg:
            continue
        else:
            send_json({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()
