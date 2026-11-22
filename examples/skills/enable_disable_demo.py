"""
C11 技能启用/禁用功能示例
演示单个技能禁用/启用、按tag批量禁用/启用、禁用全部/启用全部
离线运行，无需API Key
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import SkillRegistry, SkillMiddleware, skill


def demo():
    print("=" * 60)
    print("  C11 技能启用/禁用功能演示")
    print("=" * 60)
    
    # 1. 注册几个测试技能
    # 演示场景只使用代码型技能，关闭默认路径扫描避免无关冲突告警
    registry = SkillRegistry(include_default_paths=False)
    
    @skill(name="web-search", description="网络搜索技能", tags=["web", "search"])
    def web_search(): return "搜索结果"
    
    @skill(name="code-exec", description="代码执行技能", tags=["code", "exec"])
    def code_exec(): return "执行结果"
    
    @skill(name="file-read", description="文件读取技能", tags=["file", "read"])
    def file_read(): return "文件内容"
    
    @skill(name="data-analysis", description="数据分析技能", tags=["data", "analysis"])
    def data_analysis(): return "分析报告"
    
    registry.register(web_search)
    registry.register(code_exec)
    registry.register(file_read)
    registry.register(data_analysis)
    
    middleware = SkillMiddleware(registry=registry, auto_inject_index=True)
    
    print("\n--- 初始状态：所有技能可见 ---")
    print("可用技能索引：")
    for s in registry.list_skills():
        status = "❌ 禁用" if s["disabled"] else "✅ 启用"
        print(f"  {status} {s['name']} (tags: {', '.join(s['tags'])})")
    
    # 2. 禁用单个技能
    print("\n--- 禁用 code-exec 技能 ---")
    result = registry.disable_skill("code-exec")
    print(f"  结果: {result}")
    print("  索引中code-exec可见?", "code-exec" in registry.get_index_prompt())
    
    # 3. 尝试加载被禁用的技能
    print("\n--- 尝试加载被禁用的 code-exec ---")
    load_result = registry.load_skill("code-exec")
    print(f"  加载成功? {load_result.success}")
    print(f"  错误信息: {'; '.join(load_result.errors)}")
    
    # 4. 重新启用
    print("\n--- 启用 code-exec ---")
    result = registry.enable_skill("code-exec")
    print(f"  结果: {result}")
    load_result = registry.load_skill("code-exec")
    print(f"  加载成功? {load_result.success}")
    
    # 5. 按tag禁用（禁用所有code类技能）
    print("\n--- 按tag禁用：禁用所有tag含'code'的技能 ---")
    result = registry.disable_skills_by_tag("code")
    print(f"  结果: {result}")
    for s in registry.list_skills():
        status = "❌ 禁用" if s["disabled"] else "✅ 启用"
        print(f"  {status} {s['name']}")
    
    # 6. 按tag启用
    print("\n--- 按tag启用：恢复tag='code'的技能 ---")
    result = registry.enable_skills_by_tag("code")
    print(f"  结果: {result}")
    
    # 7. 禁用所有
    print("\n--- 禁用所有技能 ---")
    result = registry.disable_all_skills()
    print(f"  结果: {result}")
    prompt = registry.get_index_prompt()
    print(f"  索引内容: {prompt.strip()}")
    
    # 8. 启用所有
    print("\n--- 启用所有技能 ---")
    result = registry.enable_all_skills()
    print(f"  结果: {result}")
    
    # 9. 演示禁用已加载技能自动卸载
    print("\n--- 演示：禁用已加载技能会自动卸载 ---")
    # 创建带工具白名单的场景
    from fast_agents.tools import ToolRegistry
    tool_reg = ToolRegistry()
    def search_tool(): return "search"
    tool_reg.register_tool(search_tool)
    for t in registry.build_skill_tools():
        tool_reg.register_tool(t)
    
    mw = SkillMiddleware(registry=registry)
    mw.bind_tool_registry(tool_reg)
    
    # 加载web-search
    res = registry.load_skill("web-search")
    mw.after_tool_call("load_skill", {"skill_name": "web-search"}, res.full_prompt)
    print(f"  加载web-search后，已加载: {mw.get_loaded_skills()}")
    
    # 禁用web-search
    registry.disable_skill("web-search")
    print(f"  禁用web-search后，已加载: {mw.get_loaded_skills()}")
    print(f"  白名单状态: {'无过滤(全部可用)' if mw._current_allowed_tools is None else '有限制'}")
    
    # 恢复
    registry.enable_skill("web-search")
    
    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo()
