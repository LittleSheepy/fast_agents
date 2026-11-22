"""
示例1：代码型Skill定义与使用
演示如何用继承BaseSkill和@skill装饰器两种方式定义代码型技能
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.skills import BaseSkill, skill, SkillRegistry


# 方式1：继承BaseSkill定义
class DataAnalysisSkill(BaseSkill):
    name = "data-analysis"
    description = "分析CSV数据并生成简单统计，支持求和、平均值计算"
    version = "1.0.0"
    tags = ["data", "analysis", "math"]
    
    def get_instructions(self) -> str:
        return """
        数据分析执行流程：
        1. 接收数字列表作为输入
        2. 计算总和、平均值、最大值、最小值
        3. 返回统计结果
        """
    
    def execute(self, numbers: list[float], **kwargs) -> dict:
        """执行简单统计分析"""
        if not numbers:
            return {"error": "数字列表不能为空"}
        
        total = sum(numbers)
        avg = total / len(numbers)
        return {
            "count": len(numbers),
            "sum": total,
            "average": avg,
            "max": max(numbers),
            "min": min(numbers),
        }


# 方式2：@skill装饰器快速定义
@skill(
    name="greeting",
    description="生成不同语言的问候语",
    tags=["greeting", "i18n"],
    allowed_tools=[]
)
def generate_greeting(name: str, language: str = "zh") -> str:
    """
    问候语生成规则：
    - zh: 中文 "你好，{name}！"
    - en: 英文 "Hello, {name}!"
    - jp: 日文 "こんにちは、{name}！"
    """
    greetings = {
        "zh": f"你好，{name}！",
        "en": f"Hello, {name}!",
        "jp": f"こんにちは、{name}！",
    }
    return greetings.get(language, f"Hello, {name}!")


def main():
    print("=== 代码型Skill使用示例 ===")
    print()
    
    # 创建注册表
    # 演示场景只使用代码型技能，关闭默认路径扫描避免无关冲突告警
    registry = SkillRegistry(extra_skill_paths=[], include_default_paths=False)
    
    # 注册技能
    registry.register(DataAnalysisSkill)
    registry.register(generate_greeting)
    
    # 列出所有技能
    print("已注册的技能：")
    for s in registry.list_skills():
        print(f"  - {s['name']} [{s['skill_type']}]: {s['description']}")
    print()
    
    # 执行技能
    print("执行data-analysis技能：")
    result = registry.execute_skill("data-analysis", numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"  统计结果: {result}")
    print()
    
    print("执行greeting技能：")
    result = registry.execute_skill("greeting", name="小明", language="zh")
    print(f"  问候语: {result}")
    result = registry.execute_skill("greeting", name="Alice", language="en")
    print(f"  问候语: {result}")
    print()
    
    # 获取技能索引（注入System Prompt用）
    print("技能索引（Level 1常驻上下文）：")
    print(registry.get_index_prompt())
    print()
    
    print("✅ 示例运行完成！")


if __name__ == "__main__":
    main()
