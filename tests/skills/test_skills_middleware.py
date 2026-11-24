"""
Skill模块单元测试 —— 中间件（SkillMiddleware）
覆盖：技能索引注入 System Prompt、已加载技能 LRU 缓存、每轮结束自动卸载、
       allowed_tools 工具白名单过滤（C10）、
       技能热重载线程的启停与手动重载（C14）
说明：由 test_skills.py 按组件拆分而来，用例体与断言逐字节保持不变；
      _make_skill_dir 辅助函数因跨组件复用，本文件保留一份副本以便独立运行。
全离线运行，无外部依赖，不依赖 pytest。
运行：python tests/skills/test_skills_middleware.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import (
    skill,
    SkillRegistry,
    SkillMiddleware,
)


# ==================== 辅助函数（与 test_skills.py 同源副本） ====================

def _make_skill_dir(tmpdir: Path, name: str, frontmatter: dict, instructions: str = "test") -> Path:
    """在临时目录下创建一个文件型Skill目录"""
    import yaml
    skill_dir = tmpdir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    content = "---\n" + yaml.dump(frontmatter, allow_unicode=True) + "---\n" + instructions
    skill_file.write_text(content, encoding="utf-8")
    return skill_dir


def test_b12_middleware_inject_index_prompt():
    """测试中间件注入System Prompt"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        skill_dir = tmp / "middleware-test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: middleware-test-skill
description: 中间件测试技能
---
执行流程
""", encoding="utf-8")
        
        registry = SkillRegistry(extra_skill_paths=[tmp])
        middleware = SkillMiddleware(registry=registry)
        
        prompt = middleware.before_agent_run("你是一个助手")
        assert "可用技能" in prompt
        assert "middleware-test-skill" in prompt
        assert "中间件测试技能" in prompt
        print("✅ test_b12_middleware_inject_index_prompt 通过")


def test_b13_middleware_lru_cache():
    """测试中间件LRU缓存管理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 创建3个测试skill
        for i in range(3):
            skill_dir = tmp / f"lru-skill-{i}"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"""---
name: lru-skill-{i}
description: LRU测试技能{i}
---
skill {i} content
""", encoding="utf-8")
        
        registry = SkillRegistry(extra_skill_paths=[tmp])
        middleware = SkillMiddleware(registry=registry, max_loaded_skills=2)
        
        # 加载第一个
        result1 = registry.load_skill("lru-skill-0")
        middleware.after_tool_call("load_skill", {"skill_name": "lru-skill-0"}, result1.full_prompt)
        assert "lru-skill-0" in middleware.get_loaded_skills()
        
        # 加载第二个
        result2 = registry.load_skill("lru-skill-1")
        middleware.after_tool_call("load_skill", {"skill_name": "lru-skill-1"}, result2.full_prompt)
        assert len(middleware.get_loaded_skills()) == 2
        
        # 加载第三个，应该挤出第一个
        result3 = registry.load_skill("lru-skill-2")
        middleware.after_tool_call("load_skill", {"skill_name": "lru-skill-2"}, result3.full_prompt)
        loaded = middleware.get_loaded_skills()
        assert len(loaded) == 2
        assert "lru-skill-0" not in loaded
        assert "lru-skill-1" in loaded
        assert "lru-skill-2" in loaded
        
        # 测试卸载
        assert middleware.unload_skill("lru-skill-1") is True
        assert "lru-skill-1" not in middleware.get_loaded_skills()
        
        # 测试清空
        middleware.clear_loaded_skills()
        assert len(middleware.get_loaded_skills()) == 0
        print("✅ test_b13_middleware_lru_cache 通过")


def test_b14_middleware_unload_after_turn():
    """测试每轮结束后自动卸载"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="auto-unload", description="自动卸载测试")
    def auto_unload():
        return "ok"
    
    registry.register(auto_unload)
    middleware = SkillMiddleware(registry=registry, unload_after_turn=True)
    
    result = registry.load_skill("auto-unload")
    middleware.after_tool_call("load_skill", {"skill_name": "auto-unload"}, result.full_prompt)
    assert len(middleware.get_loaded_skills()) == 1
    
    middleware.after_agent_turn()
    assert len(middleware.get_loaded_skills()) == 0
    print("✅ test_b14_middleware_unload_after_turn 通过")


# ========== C10 allowed_tools 功能测试 ==========

def test_c10_1_no_skill_no_filter():
    """C10: 未加载任何技能时，白名单为None，不做过滤"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    # 注册几个测试工具
    def tool_a(): return "a"
    def tool_b(): return "b"
    tool_reg.register_tool(tool_a)
    tool_reg.register_tool(tool_b)
    
    middleware.bind_tool_registry(tool_reg)
    
    # 未加载技能 -> 无过滤
    assert middleware._current_allowed_tools is None
    assert middleware.get_filtered_tools() is None
    assert middleware.get_filtered_tools_description() is None
    assert middleware.is_tool_allowed("tool_a") is True
    assert middleware.is_tool_allowed("tool_b") is True
    assert middleware.is_tool_allowed("any_other_tool") is True
    print("✅ test_c10_1_no_skill_no_filter 通过")


def test_c10_2_load_skill_with_allowed_tools():
    """C10: 加载声明了allowed_tools的技能后，白名单生效"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    def web_search(): return "search result"
    def read_file(): return "file content"
    def write_file(): return "written"
    def calc(): return 42
    tool_reg.register_tool(web_search)
    tool_reg.register_tool(read_file)
    tool_reg.register_tool(write_file)
    tool_reg.register_tool(calc)
    # 注册技能内置工具
    for t in registry.build_skill_tools():
        tool_reg.register_tool(t)
    
    middleware.bind_tool_registry(tool_reg)
    
    # 注册一个限制只能用web_search和read_file的技能
    @skill(
        name="web-only-skill",
        description="只允许网络搜索和读文件的技能",
        allowed_tools=["web_search", "read_file"]
    )
    def web_only():
        return "web only"
    registry.register(web_only)
    
    # 加载技能
    result = registry.load_skill("web-only-skill")
    middleware.after_tool_call("load_skill", {"skill_name": "web-only-skill"}, result.full_prompt)
    
    # 验证白名单生效
    assert middleware._current_allowed_tools is not None
    assert middleware.is_tool_allowed("web_search") is True
    assert middleware.is_tool_allowed("read_file") is True
    assert middleware.is_tool_allowed("write_file") is False
    assert middleware.is_tool_allowed("calc") is False
    # 技能内置工具始终放行
    assert middleware.is_tool_allowed("load_skill") is True
    assert middleware.is_tool_allowed("list_loaded_skills") is True
    
    # 验证过滤后的工具列表只有web_search/read_file + 内置工具
    filtered = middleware.get_filtered_tools()
    filtered_names = {t.name for t in filtered}
    assert "web_search" in filtered_names
    assert "read_file" in filtered_names
    assert "write_file" not in filtered_names
    assert "calc" not in filtered_names
    assert "load_skill" in filtered_names
    
    # 验证工具描述里有工具限制提示
    suffix = middleware.get_system_prompt_suffix()
    assert "工具使用限制" in suffix
    assert "web_search" in suffix
    assert "read_file" in suffix
    print("✅ test_c10_2_load_skill_with_allowed_tools 通过")


def test_c10_3_skill_unload_restores_all_tools():
    """C10: 卸载技能后白名单清空，恢复所有工具可用"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    def tool_x(): return "x"
    def tool_y(): return "y"
    tool_reg.register_tool(tool_x)
    tool_reg.register_tool(tool_y)
    
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="restrict-x", description="只允许x", allowed_tools=["tool_x"])
    def restrict_x():
        return "x only"
    registry.register(restrict_x)
    
    # 加载 -> 白名单生效
    result = registry.load_skill("restrict-x")
    middleware.after_tool_call("load_skill", {"skill_name": "restrict-x"}, result.full_prompt)
    assert middleware.is_tool_allowed("tool_y") is False
    
    # 手动卸载 -> 恢复
    middleware.unload_skill("restrict-x")
    assert middleware._current_allowed_tools is None
    assert middleware.is_tool_allowed("tool_y") is True
    assert middleware.is_tool_allowed("tool_x") is True
    print("✅ test_c10_3_skill_unload_restores_all_tools 通过")


def test_c10_4_multiple_skills_union():
    """C10: 多个技能同时加载时，allowed_tools取并集"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    for name in ["search", "read", "write", "calc", "delete"]:
        def make_tool(n=name):
            def _tool(): return n
            _tool.__name__ = n
            return _tool
        tool_reg.register_tool(make_tool())
    
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="skill-a", description="技能A", allowed_tools=["search", "read"])
    def skill_a(): return "a"
    @skill(name="skill-b", description="技能B", allowed_tools=["read", "write", "calc"])
    def skill_b(): return "b"
    registry.register(skill_a)
    registry.register(skill_b)
    
    # 加载A -> 只能用search/read
    res_a = registry.load_skill("skill-a")
    middleware.after_tool_call("load_skill", {"skill_name": "skill-a"}, res_a.full_prompt)
    assert middleware.is_tool_allowed("write") is False
    assert middleware.is_tool_allowed("calc") is False
    
    # 再加载B -> 并集search/read/write/calc
    res_b = registry.load_skill("skill-b")
    middleware.after_tool_call("load_skill", {"skill_name": "skill-b"}, res_b.full_prompt)
    assert middleware.is_tool_allowed("search") is True
    assert middleware.is_tool_allowed("read") is True
    assert middleware.is_tool_allowed("write") is True
    assert middleware.is_tool_allowed("calc") is True
    assert middleware.is_tool_allowed("delete") is False  # delete两个技能都没声明，不可用
    
    # 卸载A -> 只剩B的白名单
    middleware.unload_skill("skill-a")
    assert middleware.is_tool_allowed("search") is False  # A卸载后search被移除
    assert middleware.is_tool_allowed("read") is True
    assert middleware.is_tool_allowed("write") is True
    assert middleware.is_tool_allowed("calc") is True
    print("✅ test_c10_4_multiple_skills_union 通过")


def test_c10_5_wildcard_allows_all():
    """C10: 技能声明allowed_tools: ["*"]时，放行所有工具"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    def any_tool(): return "any"
    tool_reg.register_tool(any_tool)
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="wildcard-skill", description="通配符技能", allowed_tools=["*"])
    def wildcard_skill(): return "all tools"
    registry.register(wildcard_skill)
    
    result = registry.load_skill("wildcard-skill")
    middleware.after_tool_call("load_skill", {"skill_name": "wildcard-skill"}, result.full_prompt)
    
    # "*" -> 白名单为None，放行所有
    assert middleware._current_allowed_tools is None
    assert middleware.is_tool_allowed("any_tool") is True
    assert middleware.is_tool_allowed("any_other_tool_that_doesnt_exist") is True
    print("✅ test_c10_5_wildcard_allows_all 通过")


def test_c10_6_clear_loaded_resets():
    """C10: clear_loaded_skills清空所有已加载技能后，白名单重置为无过滤"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry, max_loaded_skills=10)
    
    tool_reg = ToolRegistry()
    def only_tool(): return "only"
    def other_tool(): return "other"
    tool_reg.register_tool(only_tool)
    tool_reg.register_tool(other_tool)
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="restricted", description="受限技能", allowed_tools=["only_tool"])
    def restricted(): return "ok"
    registry.register(restricted)
    
    res = registry.load_skill("restricted")
    middleware.after_tool_call("load_skill", {"skill_name": "restricted"}, res.full_prompt)
    assert middleware.is_tool_allowed("other_tool") is False
    
    # 清空 -> 恢复所有
    middleware.clear_loaded_skills()
    assert middleware._current_allowed_tools is None
    assert middleware.is_tool_allowed("other_tool") is True
    assert len(middleware.get_loaded_skills()) == 0
    print("✅ test_c10_6_clear_loaded_resets 通过")


def test_c10_7_list_loaded_skills_tool():
    """C10: list_loaded_skills工具返回正确的加载状态"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    # 先注册技能工具
    for t in registry.build_skill_tools():
        tool_reg.register_tool(t)
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="tool-listed", description="测试列表工具", allowed_tools=["web_search"])
    def listed(): return "ok"
    registry.register(listed)
    
    # 未加载时调用
    list_tool = tool_reg.get_tool("list_loaded_skills")
    result_obj = list_tool.invoke()
    result = result_obj.output if hasattr(result_obj, 'output') else str(result_obj)
    assert "当前没有加载任何技能" in result
    
    # 加载后调用
    res = registry.load_skill("tool-listed")
    middleware.after_tool_call("load_skill", {"skill_name": "tool-listed"}, res.full_prompt)
    result_obj = list_tool.invoke()
    result = result_obj.output if hasattr(result_obj, 'output') else str(result_obj)
    assert "tool-listed" in result
    assert "web_search" in result
    assert "当前可使用工具" in result
    print("✅ test_c10_7_list_loaded_skills_tool 通过")


# ========== C14 技能热重载：线程启停与手动重载 ==========

def test_c14_6_enable_disable_hot_reload():
    """【C14】启用和关闭热重载不报错，线程正常启动停止"""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "hr-skill",
            {"name":"hr-skill","description":"hot reload test","version":"1.0.0"}, "hr content")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        # 启用
        msg = mw.enable_hot_reload(interval=1.0)
        assert "已启用技能热重载" in msg
        assert mw._hot_reload_running is True
        assert mw._hot_reload_thread is not None
        assert mw._hot_reload_thread.is_alive()
        # 等待线程运行一下
        time.sleep(0.5)
        assert mw._hot_reload_thread.is_alive()
        # 关闭
        msg2 = mw.disable_hot_reload()
        assert "已关闭" in msg2
        assert mw._hot_reload_running is False
        # 等待线程结束
        time.sleep(1.5)
        assert not mw._hot_reload_thread.is_alive()
    print("✅ test_c14_6_enable_disable_hot_reload 通过")


def test_c14_7_reload_skills_now_manual_trigger():
    """【C14】手动reload_skills_now能检测并应用变更"""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "manual-skill",
            {"name":"manual-skill","description":"original","version":"1.0.0"}, "original")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        # 初始化mtime快照（模拟热重载已启用的状态）
        mw._last_mtime_map = reg.loader._build_dir_mtime_map()
        # 修改技能
        time.sleep(0.1)
        (tmpdir / "manual-skill" / "SKILL.md").write_text(
            "---\nname: manual-skill\ndescription: changed\nversion: 1.0.1\n---\nchanged\n",
            encoding="utf-8")
        # 手动触发重载
        msg = mw.reload_skills_now()
        assert "修改技能" in msg
        assert "manual-skill" in msg
        # 元数据已更新
        meta = reg.get_metadata("manual-skill")
        assert meta.description == "changed"
    print("✅ test_c14_7_reload_skills_now_manual_trigger 通过")


def test_c14_8_hot_reload_default_disabled():
    """【C14】热重载默认关闭，不启动后台线程"""
    reg = SkillRegistry(extra_skill_paths=[])
    mw = SkillMiddleware(registry=reg)
    assert mw._hot_reload_running is False
    assert mw._hot_reload_thread is None
    print("✅ test_c14_8_hot_reload_default_disabled 通过")


# ==================== 入口 ====================


def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """ 新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
