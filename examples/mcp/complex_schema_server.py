#!/usr/bin/env python3
"""
MCP 复杂参数Schema示例服务器
演示：嵌套对象、数组、枚举、默认值、数值/字符串校验等复杂JSON Schema场景
用于测试 _schema_to_pydantic 增强后的参数校验能力
零依赖：仅使用Python标准库，通过stdio通信
"""
import json
import sys


def send_message(msg: dict) -> None:
    """发送JSON-RPC消息到stdout"""
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def main() -> None:
    """MCP服务器主循环，实现复杂参数示例工具"""
    buffer = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        buffer += line
        if buffer.endswith("\n"):
            line = buffer.strip()
            buffer = ""
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {})

            # 初始化
            if method == "initialize":
                send_message({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "complex-schema-demo", "version": "1.0.0"},
                        "capabilities": {"tools": {"listChanged": True}},
                    }
                })
                # initialized通知
                initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                send_message(initialized)
                continue

            # tools/list
            if method == "tools/list":
                send_message({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "create_user",
                                "description": "创建用户，演示嵌套对象+枚举+默认值+数值校验",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "username": {
                                            "type": "string",
                                            "minLength": 3,
                                            "maxLength": 20,
                                            "description": "用户名，3-20字符"
                                        },
                                        "age": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 150,
                                            "description": "年龄，0-150"
                                        },
                                        "role": {
                                            "enum": ["admin", "user", "guest"],
                                            "default": "user",
                                            "description": "用户角色，默认user"
                                        },
                                        "profile": {
                                            "type": "object",
                                            "properties": {
                                                "email": {
                                                    "type": "string",
                                                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                                                    "description": "邮箱地址"
                                                },
                                                "phone": {"type": "string", "description": "手机号"},
                                                "notify": {
                                                    "type": "boolean",
                                                    "default": True,
                                                    "description": "是否开启通知，默认开启"
                                                }
                                            },
                                            "required": ["email"]
                                        },
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 0,
                                            "maxItems": 10,
                                            "default": [],
                                            "description": "用户标签，最多10个"
                                        },
                                        "verbose": {
                                            "type": "boolean",
                                            "default": False,
                                            "description": "是否返回详细信息，默认False"
                                        }
                                    },
                                    "required": ["username", "age"]
                                }
                            },
                            {
                                "name": "batch_add_items",
                                "description": "批量添加条目，演示对象数组+数值校验",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "items": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer", "minimum": 1},
                                                    "name": {"type": "string"},
                                                    "price": {
                                                        "type": "number",
                                                        "exclusiveMinimum": 0,
                                                        "description": "单价，必须大于0"
                                                    },
                                                    "quantity": {
                                                        "type": "integer",
                                                        "minimum": 1,
                                                        "default": 1
                                                    }
                                                },
                                                "required": ["id", "name", "price"]
                                            },
                                            "minItems": 1,
                                            "maxItems": 100
                                        },
                                        "note": {
                                            "anyOf": [
                                                {"type": "string"},
                                                {"type": "null"}
                                            ],
                                            "description": "备注，可空"
                                        }
                                    },
                                    "required": ["items"]
                                }
                            }
                        ]
                    }
                })
                continue

            # tools/call
            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result_content = []
                if tool_name == "create_user":
                    result_content.append({
                        "type": "text",
                        "text": f"✅ 创建用户成功：\n"
                                f"用户名: {arguments.get('username')}\n"
                                f"年龄: {arguments.get('age')}\n"
                                f"角色: {arguments.get('role', 'user')}\n"
                                f"邮箱: {arguments.get('profile', {}).get('email')}\n"
                                f"标签: {arguments.get('tags', [])}\n"
                                f"详细模式: {arguments.get('verbose', False)}"
                    })
                elif tool_name == "batch_add_items":
                    items = arguments.get("items", [])
                    total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
                    result_content.append({
                        "type": "text",
                        "text": f"✅ 批量添加{len(items)}个条目成功，总金额：{total:.2f}元\n"
                                f"备注：{arguments.get('note', '无')}"
                    })
                else:
                    result_content.append({"type": "text", "text": f"未知工具：{tool_name}"})
                send_message({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": result_content,
                        "isError": False
                    }
                })
                continue


if __name__ == "__main__":
    main()
