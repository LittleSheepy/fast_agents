"""
示例2：文件型Skill加载与三级渐进式加载演示
演示项目根目录.skills下的web-research技能的三级加载流程
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.skills import SkillRegistry


def main():
    print("=== 文件型Skill三级渐进式加载示例 ===")
    print()
    
    # 创建注册表（自动扫描项目根目录.skills/）
    registry = SkillRegistry()
    
    print("【Level 1：索引加载】所有技能元数据常驻System Prompt")
    index_prompt = registry.get_index_prompt()
    print(index_prompt)
    print()
    
    # 列出所有发现的技能
    all_skills = registry.list_skills()
    print(f"共发现 {len(all_skills)} 个技能：")
    for s in all_skills:
        print(f"  - {s['name']} [{s['skill_type']}]: {s['description']}")
    print()
    
    print("【Level 2：指令加载】按需加载web-research完整执行流程")
    load_result = registry.load_skill("web-research")
    if load_result.success:
        print(f"✅ 加载成功！")
        print(f"  可用脚本: {[p.name for p in load_result.scripts]}")
        print(f"  可用参考文档: {[p.name for p in load_result.references]}")
        print()
        print("生成的完整Skill Prompt：")
        print(load_result.full_prompt[:500] + "\n...（截断，完整内容超过500字符）")
    else:
        print(f"❌ 加载失败: {load_result.errors}")
    print()
    
    print("【Level 3：资源读取】按需读取参考文档source-check.md")
    ref_content = registry.read_reference("web-research", "source-check.md")
    print(f"参考文档source-check.md前300字符：")
    print(ref_content[:300] + "..." if len(ref_content) > 300 else ref_content)
    print()
    
    print("✅ 文件型Skill三级加载流程演示完成！")


if __name__ == "__main__":
    main()
