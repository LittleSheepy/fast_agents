"""
C6-2 沙箱演示脚本：接收参数，验证参数类型转换效果
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--city", default="上海")
    args = parser.parse_args()
    
    print(f"你好 {args.name}！")
    print(f"你今年 {args.age} 岁，来自 {args.city}。")
    
    # 验证网络被禁用
    proxy = os.environ.get("HTTP_PROXY", "")
    net_disabled = os.environ.get("SKILL_SANDBOX_NETWORK_DISABLED", "0")
    print(f"[沙箱] HTTP_PROXY={proxy}, 网络禁用标记={net_disabled}")
    
    # 验证类型转换正确
    assert isinstance(args.age, int), f"age应为int，实际是{type(args.age)}"
    print(f"[沙箱] age类型校验通过: {args.age} ({type(args.age).__name__})")


if __name__ == "__main__":
    main()
