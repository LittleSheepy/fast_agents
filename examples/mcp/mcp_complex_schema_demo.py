#!/usr/bin/env python3
"""
MCP复杂Schema增强功能演示
演示 _schema_to_pydantic 新增的能力：嵌套对象、数组、枚举、默认值、各类校验等
运行方式：python examples/mcp/mcp_complex_schema_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.mcp import MCPClient, MCPToolAdapter
from pydantic import ValidationError


def demo_schema_capabilities():
    print("=" * 60)
    print("MCP _schema_to_pydantic 增强功能演示")
    print("=" * 60)

    # 1. 基础类型校验
    print("\n1. 基础类型+枚举校验：")
    client = MCPClient(command=sys.executable, args=[
        str(Path(__file__).parent / "complex_schema_server.py")
    ])
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        print(f"   ✅ 发现工具：{[t['name'] for t in tools]}")

        # 获取create_user工具的adapter
        for t in tools:
            if t["name"] == "create_user":
                adapter = MCPToolAdapter(
                    mcp_client=client,
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema"),
                    server_name="demo"
                )
                # 测试默认值
                print("\n   测试默认值（不传verbose/role/tags/profile.notify）：")
                args = {
                    "username": "zhangsan",
                    "age": 25,
                    "profile": {"email": "zhangsan@example.com"}
                }
                validated = adapter.args_schema(**args)
                print(f"      role默认值: {validated.role} (预期: user)")
                print(f"      verbose默认值: {validated.verbose} (预期: False)")
                print(f"      tags默认值: {validated.tags} (预期: [])")
                print(f"      notify默认值: {validated.profile.notify} (预期: True)")

                # 测试数值校验
                print("\n   测试数值校验（age=-1，应该报错）：")
                try:
                    adapter.args_schema(username="zhangsan", age=-1, profile={"email": "a@b.com"})
                    print("      ❌ 校验失败：负数年龄未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 测试字符串长度校验
                print("\n   测试字符串长度校验（username='ab'，应该报错）：")
                try:
                    adapter.args_schema(username="ab", age=25, profile={"email": "a@b.com"})
                    print("      ❌ 校验失败：短用户名未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 测试邮箱格式校验
                print("\n   测试正则校验（邮箱格式错误，应该报错）：")
                try:
                    adapter.args_schema(username="zhangsan", age=25, profile={"email": "invalid-email"})
                    print("      ❌ 校验失败：非法邮箱未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 测试枚举校验
                print("\n   测试枚举校验（role='superadmin'，应该报错）：")
                try:
                    adapter.args_schema(username="zhangsan", age=25, role="superadmin", profile={"email": "a@b.com"})
                    print("      ❌ 校验失败：非法角色未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 正常调用
                print("\n   正常调用create_user工具：")
                result = adapter.run(**args)
                print(f"      调用结果：\n{result.output}")

        # 测试对象数组工具
        for t in tools:
            if t["name"] == "batch_add_items":
                adapter = MCPToolAdapter(
                    mcp_client=client,
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema"),
                    server_name="demo"
                )
                print("\n2. 嵌套对象数组校验：")
                # 正常调用
                args = {
                    "items": [
                        {"id": 1, "name": "商品A", "price": 99.9, "quantity": 2},
                        {"id": 2, "name": "商品B", "price": 59.9}
                    ],
                    "note": None
                }
                print("   测试对象数组+可选字段默认值（quantity默认1）：")
                validated = adapter.args_schema(**args)
                print(f"      商品B quantity默认值: {validated.items[1].quantity} (预期: 1)")
                print(f"      note允许为None: {validated.note} (预期: None)")

                # 测试price必须大于0（exclusiveMinimum）
                print("\n   测试数值校验（price=0，应该报错）：")
                try:
                    adapter.args_schema(items=[{"id": 1, "name": "A", "price": 0}])
                    print("      ❌ 校验失败：0价格未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 测试minItems校验
                print("\n   测试数组长度校验（空数组，应该报错）：")
                try:
                    adapter.args_schema(items=[])
                    print("      ❌ 校验失败：空数组未被拦截")
                except ValidationError as e:
                    print(f"      ✅ 校验生效：{e.errors()[0]['msg']}")

                # 正常调用
                print("\n   正常调用batch_add_items工具：")
                result = adapter.run(**args)
                print(f"      调用结果：\n{result.output}")

    finally:
        client.close()

    print("\n" + "=" * 60)
    print("所有演示完成！_schema_to_pydantic增强功能正常工作。")
    print("=" * 60)


if __name__ == "__main__":
    demo_schema_capabilities()
