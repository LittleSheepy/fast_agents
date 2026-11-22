#!/usr/bin/env python3
"""MCP Resources/Prompts 功能演示脚本。

运行方式：python mcp_resources_prompts_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fast_agents.mcp import MCPRegistry, MCPConfig


def main():
    print("=" * 60)
    print("MCP Resources / Prompts 功能演示")
    print("=" * 60)

    # 初始化注册表，连接示例服务器
    reg = MCPRegistry()
    server_path = os.path.join(os.path.dirname(__file__), "resources_prompts_server.py")
    reg.add_stdio("demo", sys.executable, [server_path])

    print("\n1. 连接服务器...")
    reg.connect_all()
    client = reg.get_client("demo")
    assert client is not None
    caps = client.initialize_result.capabilities
    print(f"   服务器能力：")
    print(f"     - tools: {caps.tools is not None}")
    print(f"     - resources: {caps.resources is not None}")
    print(f"     - prompts: {caps.prompts is not None}")

    print("\n2. 列出工具...")
    tools = client.list_tools()
    for t in tools:
        print(f"   - {t.name}: {t.description}")

    print("\n3. 调用echo工具测试...")
    result = client.call_tool("echo", {"message": "Hello MCP!"})
    print(f"   返回：{result.as_text()}")

    print("\n4. 列出资源...")
    resources = client.list_resources()
    for r in resources:
        print(f"   - {r.name}: {r.uri} ({r.mimeType})")

    print("\n5. 列出资源模板...")
    templates = client.list_resource_templates()
    for t in templates:
        print(f"   - {t.name}: {t.uriTemplate}")

    print("\n6. 读取静态资源（快速开始指南）...")
    res = client.read_resource("docs://getting_started")
    text = res.as_text()
    print(f"   资源内容（前200字）：{text[:200]}...")

    print("\n7. 列出提示模板...")
    prompts = client.list_prompts()
    for p in prompts:
        args_desc = ", ".join([f"{a.name}({'必填' if a.required else '可选'})" for a in p.arguments])
        print(f"   - {p.name}: {p.description}（参数：{args_desc}）")

    print("\n8. 渲染提示模板（code_review，审查Python代码）...")
    prompt_result = client.get_prompt("code_review", {
        "language": "Python",
        "code": "def add(a,b): return a+b",
    })
    messages = prompt_result.as_messages()
    for m in messages:
        print(f"   [{m['role']}]: {m['content'][:150]}...")

    print("\n9. 粗粒度工具Resources/Prompts action测试...")
    from fast_agents.mcp.tool import MCPServerTool
    tool = MCPServerTool(client=client, server_name="demo", tools=tools)
    res = tool.run(action="list_resources")
    print(f"   list_resources 结果前300字：{res.output[:300]}...")
    res = tool.run(action="list_prompts")
    print(f"   list_prompts 结果前200字：{res.output[:200]}...")

    print("\n10. Registry便捷方法测试...")
    all_resources = reg.list_resources()
    all_prompts = reg.list_prompts()
    print(f"    全部服务器资源数：{sum(len(v) for v in all_resources.values())}")
    print(f"    全部服务器提示模板数：{sum(len(v) for v in all_prompts.values())}")

    # 关闭连接
    reg.close_all()
    print("\n" + "=" * 60)
    print("演示完成，所有功能正常！")
    print("=" * 60)


if __name__ == "__main__":
    main()
