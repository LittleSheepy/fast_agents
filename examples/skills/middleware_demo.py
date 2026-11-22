"""
示例3：SkillMiddleware中间件集成演示
演示中间件如何自动注入Prompt、管理已加载技能的LRU生命周期
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.skills import SkillRegistry, SkillMiddleware, skill


@skill(name="translate", description="简单中英文互译", tags=["translate", "i18n"])
def translate(text: str, to_lang: str = "en") -> str:
    """翻译执行规则：简单演示用"""
    demo_trans = {
        "你好": {"en": "Hello", "jp": "こんにちは"},
        "谢谢": {"en": "Thank you", "jp": "ありがとう"},
    }
    return demo_trans.get(text, {}).get(to_lang, f"[{to_lang}] {text}")


def main():
    print("=== SkillMiddleware中间件使用示例 ===")
    print()
    
    # 创建注册表并注册技能
    # 演示场景只使用代码型技能，关闭默认路径扫描避免无关冲突告警
    registry = SkillRegistry(extra_skill_paths=[], include_default_paths=False)
    registry.register(translate)
    
    # 创建中间件：最多同时加载2个技能，每轮对话结束后自动卸载
    middleware = SkillMiddleware(
        registry=registry,
        auto_inject_index=True,
        unload_after_turn=False,
        max_loaded_skills=2,
    )
    
    base_prompt = "你是一个有用的AI助手。"
    
    print("1. Agent运行前，中间件自动注入技能索引到System Prompt：")
    system_prompt = middleware.before_agent_run(base_prompt)
    print(system_prompt)
    print()
    
    print("2. 模拟LLM调用load_skill加载技能，中间件自动记录已加载技能：")
    load_result = registry.load_skill("translate")
    processed_result = middleware.after_tool_call(
        "load_skill", 
        {"skill_name": "translate"}, 
        load_result.full_prompt
    )
    print(f"已加载技能列表: {middleware.get_loaded_skills()}")
    print(f"技能调用统计: {middleware.get_stats()}")
    print()
    
    print("3. 再次注入System Prompt时，会自动包含已加载技能的完整指令：")
    system_prompt_with_loaded = middleware.before_agent_run(base_prompt)
    assert "已加载的技能详情" in system_prompt_with_loaded
    print("  ✅ System Prompt已包含已加载技能详情")
    print()
    
    print("4. 手动卸载技能：")
    middleware.unload_skill("translate")
    print(f"卸载后已加载技能列表: {middleware.get_loaded_skills()}")
    print()
    
    print("5. 测试LRU淘汰：超过max_loaded_skills自动卸载最早加载的")
    # 再注册几个测试技能
    for i in range(3):
        @skill(name=f"test-skill-{i}", description=f"测试技能{i}")
        def test_func():
            return f"skill {i} result"
        # 注意：这里闭包变量问题，实际使用时单独定义
        registry.register(test_func)
        result = registry.load_skill(f"test-skill-{i}")
        middleware.after_tool_call("load_skill", {"skill_name": f"test-skill-{i}"}, result.full_prompt)
        print(f"  加载test-skill-{i}后，已加载: {middleware.get_loaded_skills()}")
    print()
    
    print("6. 清空所有已加载技能：")
    middleware.clear_loaded_skills()
    print(f"清空后已加载技能列表: {middleware.get_loaded_skills()}")
    print()
    
    print("7. 获取Skill工具（可直接注册到ToolRegistry）：")
    tools = middleware.get_tools()
    print(f"技能提供的工具数量: {len(tools)}")
    for t in tools:
        print(f"  - {t.name}: {t.description[:50]}...")
    print()
    
    print("✅ SkillMiddleware示例运行完成！")


if __name__ == "__main__":
    main()
