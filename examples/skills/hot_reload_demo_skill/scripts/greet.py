#!/usr/bin/env python3
"""热重载演示脚本：打印欢迎信息"""
import sys
import time

def main():
    print("=" * 50)
    print("  欢迎使用 FastAgents 技能热重载功能 (C14)")
    print(f"  当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print("\n提示: 修改SKILL.md或scripts/下的文件后，")
    print("热重载会自动检测变更并重新加载技能，无需重启Agent。\n")

if __name__ == "__main__":
    main()
