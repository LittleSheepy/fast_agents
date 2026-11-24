"""
Skill模块单元测试 —— 注册器（SkillRegistry）
覆盖：代码型技能注册 / 反注册、便捷方法与工具工厂、索引提示与技能列表、
       启用禁用（C11）、依赖预检与版本比较（C12）、脚本沙箱运行时校验（C6-2）、
       静态资产读取（C15）、同名技能冲突按来源优先级消解（C8）
说明：由 test_skills.py 按组件拆分而来，用例体与断言逐字节保持不变；
      _make_* 辅助函数因跨组件复用，本文件保留一份副本以便独立运行。
全离线运行，无外部依赖，不依赖 pytest。
运行：python tests/skills/test_skills_registry.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import (
    BaseSkill,
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


def _make_script_skill(tmpdir: Path, name: str, scripts_cfg: list, script_name: str, script_content: str, allow_network: bool = False) -> Path:
    """辅助：创建带scripts目录与脚本的文件型Skill
    【C18方案B】scripts参数Schema不再写SKILL.md frontmatter，而是写到scripts/<脚本名>.schema.json
    """
    import json
    skill_dir = tmpdir / name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    # frontmatter不再包含scripts段和沙箱配置，对齐Anthropic规范，仅保留核心字段
    fm = {
        "name": name, "description": f"{name} desc", "version": "1.0.0",
    }
    (skill_dir / "SKILL.md").write_text(
        "---\n" + json.dumps(fm, ensure_ascii=False, indent=2) + "\n---\ninstructions", encoding="utf-8"
    )
    script_path = scripts_dir / script_name
    script_path.write_text(script_content, encoding="utf-8")
    # 把scripts_cfg写到对应脚本的.schema.json文件
    for script_cfg in scripts_cfg:
        s_name = script_cfg.get("name", script_name)
        if not s_name.endswith(".schema.json"):
            schema_name = s_name.replace(".py", ".schema.json")
        else:
            schema_name = s_name
        schema_path = scripts_dir / schema_name
        # 写入schema，和脚本同目录
        schema_path.write_text(json.dumps(script_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return skill_dir


def _make_asset_skill(root: Path) -> SkillRegistry:
    """构造一个含assets目录的临时技能，返回已注册该路径的SkillRegistry"""
    from fast_agents.skills import SkillRegistry
    skill_dir = root / "asset-demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: asset-demo\ndescription: 资产演示\nversion: 1.0.0\n---\n# 演示\n请使用assets中的资源。\n",
        encoding="utf-8",
    )
    assets = skill_dir / "assets"
    assets.mkdir()
    # 文本资产
    (assets / "hello.txt").write_text("hello world\n你好", encoding="utf-8")
    # 子目录下的文本资产
    (assets / "templates").mkdir()
    (assets / "templates" / "tpl.json").write_text('{"msg":"hi"}', encoding="utf-8")
    # 下划线开头的内部文件不应被暴露
    (assets / "_internal.txt").write_text("should not list", encoding="utf-8")
    # 二进制资产
    (assets / "img.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE-PNG-BYTES")
    # 大文本（模拟超过上限的文本）
    big = assets / "big.txt"
    big.write_text("x" * (200 * 1024 + 10), encoding="utf-8")

    reg = SkillRegistry(extra_skill_paths=[root])
    # 强制全量重扫（绕过C17缓存）：先失效索引缓存再scan
    reg.loader.invalidate_index_cache()
    reg.scan()
    return reg


def _make_conflict_skill_dir(path: Path, name: str, version: str, content_tag: str = ""):
    """创建同名技能目录（用于冲突测试）"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} {content_tag}\nversion: {version}\n---\n# {name}\n{content_tag}\n",
        encoding="utf-8",
    )


def test_b5_registry_register_code_skill():
    """测试注册表注册代码型Skill"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    class MySkill(BaseSkill):
        name = "my-code-skill"
        description = "我的代码技能"
        def execute(self, **kwargs):
            return "code skill result"
    
    # 注册类
    registry.register(MySkill)
    assert "my-code-skill" in registry.get_all_metadata()
    
    # 注册实例
    class MySkill2(BaseSkill):
        name = "my-code-skill-2"
        description = "我的代码技能2"
        def execute(self, **kwargs):
            return "code skill 2 result"
    registry.register(MySkill2())
    assert "my-code-skill-2" in registry.get_all_metadata()
    
    # 执行
    result = registry.execute_skill("my-code-skill")
    assert result == "code skill result"
    print("✅ test_b5_registry_register_code_skill 通过")


def test_b6_registry_unregister():
    """测试注销技能"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="to-remove", description="to remove")
    def to_remove():
        return "remove me"
    
    registry.register(to_remove)
    assert "to-remove" in registry.get_all_metadata()
    
    assert registry.unregister("to-remove") is True
    assert "to-remove" not in registry.get_all_metadata()
    assert registry.unregister("not-exist") is False
    print("✅ test_b6_registry_unregister 通过")


def test_b15_default_skill_paths():
    """测试默认路径自动扫描项目根.skills"""
    # 框架根目录的.skills应该能被扫描到（web-research）
    registry = SkillRegistry()
    all_meta = registry.get_all_metadata()
    # 项目根目录下的web-research技能应该被发现
    assert "web-research" in all_meta
    assert all_meta["web-research"].source in ["builtin", "project"]
    print("✅ test_b15_default_skill_paths 通过")


def test_b16_build_skill_tools():
    """测试build_skill_tools工厂方法返回有效工具"""
    registry = SkillRegistry(extra_skill_paths=[])
    tools = registry.build_skill_tools()
    assert len(tools) == 13  # 4基础(load/read_ref/read_asset/run/list) + 4启用禁用 + 1依赖预检 + 3热重载
    tool_names = [t.name for t in tools]
    assert "load_skill" in tool_names
    assert "read_skill_reference" in tool_names
    assert "read_skill_asset" in tool_names  # 【C15】
    assert "run_skill_script" in tool_names
    assert "list_loaded_skills" in tool_names
    # C11 新增
    assert "disable_skill" in tool_names
    assert "enable_skill" in tool_names
    assert "disable_skills_by_tag" in tool_names
    assert "enable_skills_by_tag" in tool_names
    # C12 新增
    assert "check_skill_dependencies" in tool_names
    # C14 新增热重载工具
    assert "enable_skill_hot_reload" in tool_names
    assert "disable_skill_hot_reload" in tool_names
    assert "reload_skills_now" in tool_names
    print("✅ test_b16_build_skill_tools 通过")


def test_b17_get_index_prompt():
    """测试索引提示生成"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="index-test", description="索引测试技能", tags=["test"])
    def index_test():
        return "ok"
    
    registry.register(index_test)
    prompt = registry.get_index_prompt()
    assert "可用技能" in prompt
    assert "index-test" in prompt
    assert "索引测试技能" in prompt
    assert "[code]" in prompt
    assert "技能使用方法" in prompt
    print("✅ test_b17_get_index_prompt 通过")


def test_b18_list_skills():
    """测试列出所有技能"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="list-test-1", description="列表测试1")
    def t1(): pass
    
    @skill(name="list-test-2", description="列表测试2")
    def t2(): pass
    
    registry.register(t1)
    registry.register(t2)
    
    skills = registry.list_skills()
    names = [s["name"] for s in skills]
    assert "list-test-1" in names
    assert "list-test-2" in names
    print("✅ test_b18_list_skills 通过")


# ========== C11 技能启用/禁用功能测试 ==========

def test_c11_1_disable_single_skill_hides_from_index():
    """C11: 禁用单个技能后，该技能从索引prompt中隐藏"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="visible-skill", description="可见技能", tags=["test"])
    def visible(): return "ok"
    @skill(name="hidden-skill", description="要被隐藏的技能", tags=["test"])
    def hidden(): return "ok"
    registry.register(visible)
    registry.register(hidden)
    
    # 禁用前两个都可见
    prompt_before = registry.get_index_prompt()
    assert "visible-skill" in prompt_before
    assert "hidden-skill" in prompt_before
    
    # 禁用hidden-skill
    result = registry.disable_skill("hidden-skill")
    assert "已禁用" in result
    
    # 禁用后索引中只有visible-skill
    prompt_after = registry.get_index_prompt()
    assert "visible-skill" in prompt_after
    assert "hidden-skill" not in prompt_after
    
    # is_skill_disabled检查
    assert registry.is_skill_disabled("hidden-skill") is True
    assert registry.is_skill_disabled("visible-skill") is False
    print("✅ test_c11_1_disable_single_skill_hides_from_index 通过")


def test_c11_2_disabled_skill_cannot_be_loaded():
    """C11: 被禁用的技能无法被load_skill加载"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="blocked-skill", description="被封锁的技能")
    def blocked(): return "should not run"
    registry.register(blocked)
    
    registry.disable_skill("blocked-skill")
    
    result = registry.load_skill("blocked-skill")
    assert result.success is False
    assert any("已被禁用" in e for e in result.errors)
    print("✅ test_c11_2_disabled_skill_cannot_be_loaded 通过")


def test_c11_3_enable_skill_restores():
    """C11: 启用技能后恢复可见和可加载"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="toggle-skill", description="可切换的技能")
    def toggle(): return "toggle ok"
    registry.register(toggle)
    
    # 禁用
    registry.disable_skill("toggle-skill")
    assert registry.is_skill_disabled("toggle-skill") is True
    
    # 启用
    result = registry.enable_skill("toggle-skill")
    assert "已启用" in result
    
    # 恢复后可见且可加载
    assert registry.is_skill_disabled("toggle-skill") is False
    prompt = registry.get_index_prompt()
    assert "toggle-skill" in prompt
    load_result = registry.load_skill("toggle-skill")
    assert load_result.success is True
    print("✅ test_c11_3_enable_skill_restores 通过")


def test_c11_4_disable_by_tag():
    """C11: 按tag批量禁用技能"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="dev-a", description="开发技能A", tags=["dev", "test"])
    def dev_a(): return "a"
    @skill(name="dev-b", description="开发技能B", tags=["dev"])
    def dev_b(): return "b"
    @skill(name="prod-skill", description="生产技能", tags=["prod"])
    def prod(): return "prod"
    registry.register(dev_a)
    registry.register(dev_b)
    registry.register(prod)
    
    # 按dev tag禁用
    result = registry.disable_skills_by_tag("dev")
    assert "dev-a" in result
    assert "dev-b" in result
    assert "prod-skill" not in result
    
    # dev-a和dev-b被禁用，prod不受影响
    assert registry.is_skill_disabled("dev-a") is True
    assert registry.is_skill_disabled("dev-b") is True
    assert registry.is_skill_disabled("prod-skill") is False
    
    # 索引中只剩prod-skill
    prompt = registry.get_index_prompt()
    assert "dev-a" not in prompt
    assert "dev-b" not in prompt
    assert "prod-skill" in prompt
    
    # 按tag启用
    result2 = registry.enable_skills_by_tag("dev")
    assert "dev-a" in result2
    assert "dev-b" in result2
    assert registry.is_skill_disabled("dev-a") is False
    assert registry.is_skill_disabled("dev-b") is False
    print("✅ test_c11_4_disable_by_tag 通过")


def test_c11_5_disable_all_and_enable_all():
    """C11: 禁用所有/启用所有"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="s1", description="技能1")
    def s1(): return "1"
    @skill(name="s2", description="技能2")
    def s2(): return "2"
    registry.register(s1)
    registry.register(s2)
    
    # 禁用所有
    result = registry.disable_all_skills()
    assert "已禁用全部" in result
    assert registry.is_skill_disabled("s1") is True
    assert registry.is_skill_disabled("s2") is True
    
    # 索引中s1和s2不可见（可能还有其他内置技能也被禁用）
    prompt = registry.get_index_prompt()
    assert "s1" not in prompt
    assert "s2" not in prompt
    
    # 启用所有
    result2 = registry.enable_all_skills()
    assert "已启用所有" in result2
    assert registry.is_skill_disabled("s1") is False
    assert registry.is_skill_disabled("s2") is False
    
    prompt2 = registry.get_index_prompt()
    assert "s1" in prompt2
    assert "s2" in prompt2
    print("✅ test_c11_5_disable_all_and_enable_all 通过")


def test_c11_6_disable_loaded_skill_auto_unloads():
    """C11: 禁用一个已加载的技能时，自动卸载并重算白名单"""
    from fast_agents.tools import ToolRegistry
    registry = SkillRegistry(extra_skill_paths=[])
    middleware = SkillMiddleware(registry=registry)
    
    tool_reg = ToolRegistry()
    def web_search(): return "search"
    tool_reg.register_tool(web_search)
    for t in registry.build_skill_tools():
        tool_reg.register_tool(t)
    middleware.bind_tool_registry(tool_reg)
    
    @skill(name="auto-unload-on-disable", description="禁用时自动卸载", allowed_tools=["web_search"])
    def auto_unload(): return "ok"
    registry.register(auto_unload)
    
    # 加载技能
    res = registry.load_skill("auto-unload-on-disable")
    middleware.after_tool_call("load_skill", {"skill_name": "auto-unload-on-disable"}, res.full_prompt)
    assert "auto-unload-on-disable" in middleware.get_loaded_skills()
    assert middleware.is_tool_allowed("web_search") is True
    
    # 禁用该技能 -> 应自动卸载，白名单恢复
    registry.disable_skill("auto-unload-on-disable")
    assert "auto-unload-on-disable" not in middleware.get_loaded_skills()
    assert middleware._current_allowed_tools is None  # 无加载技能 -> 白名单重置
    assert middleware.is_tool_allowed("web_search") is True  # 恢复可用
    print("✅ test_c11_6_disable_loaded_skill_auto_unloads 通过")


def test_c11_7_disable_nonexistent_skill_returns_error():
    """C11: 禁用/启用不存在的技能返回错误信息"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    result = registry.disable_skill("no-such-thing")
    assert "不存在" in result
    
    result2 = registry.enable_skill("no-such-thing")
    assert "不存在" in result2
    print("✅ test_c11_7_disable_nonexistent_skill_returns_error 通过")


def test_c11_8_list_skills_includes_disabled_status():
    """C11: list_skills结果中包含disabled字段"""
    registry = SkillRegistry(extra_skill_paths=[])
    
    @skill(name="enabled-s", description="启用的")
    def es(): return "e"
    @skill(name="disabled-s", description="禁用的")
    def ds(): return "d"
    registry.register(es)
    registry.register(ds)
    registry.disable_skill("disabled-s")
    
    skills = registry.list_skills()
    skill_map = {s["name"]: s for s in skills}
    assert skill_map["enabled-s"]["disabled"] is False
    assert skill_map["disabled-s"]["disabled"] is True
    print("✅ test_c11_8_list_skills_includes_disabled_status 通过")


# ========== C12 技能依赖检查：注册器依赖预检 ==========

def test_c12_1_version_parse_and_compare():
    """【C12】版本号解析与比较逻辑正确"""
    parse = SkillRegistry._parse_version
    assert parse("1.2.3") == (1, 2, 3)
    assert parse("1.2") == (1, 2, 0)
    assert parse("1") == (1, 0, 0)
    assert parse("2.0.0-beta.1") == (2, 0, 0)
    assert parse(None) == (0, 0, 0)
    
    sat = SkillRegistry._ver_satisfies
    assert sat((1,2,3), '>=', (1,2,0)) is True
    assert sat((1,2,0), '>=', (1,2,3)) is False
    assert sat((1,2,3), '==', (1,2,3)) is True
    assert sat((1,2,3), '<=', (2,0,0)) is True
    print("✅ test_c12_1_version_parse_and_compare 通过")


def test_c12_2_dep_spec_parse():
    """【C12】依赖声明字符串解析正确"""
    parse = SkillRegistry._parse_dep_spec
    assert parse("other-skill") == ("other-skill", None, None)
    assert parse("other-skill>=1.2.0") == ("other-skill", ">=", (1,2,0))
    assert parse("core==2.0.0") == ("core", "==", (2,0,0))
    assert parse("base<3.0") == ("base", "<", (3,0,0))
    print("✅ test_c12_2_dep_spec_parse 通过")


def test_c12_5_missing_tool_blocks_load():
    """【C12】依赖工具缺失时，load_skill返回失败并给出明确提示"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fm = {
            "name": "need-tool-skill",
            "description": "需要不存在工具的技能",
            "version": "1.0.0",
            "allowed_tools": ["nonexistent_tool_xyz"],
        }
        _make_skill_dir(tmpdir, "need-tool-skill", fm, "instr")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        
        # 构造一个只含skill内置工具的假tool_registry
        class FakeTR:
            def list_names(self): return ["load_skill"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        result = reg.load_skill("need-tool-skill")
        assert result.success is False
        err_text = "; ".join(result.errors)
        assert "nonexistent_tool_xyz" in err_text
        assert "缺失依赖工具" in err_text
    print("✅ test_c12_5_missing_tool_blocks_load 通过")


def test_c12_6_missing_skill_dependency_blocks_load():
    """【C12】依赖的技能不存在时，加载失败"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fm = {
            "name": "needs-dep",
            "description": "依赖不存在技能的技能",
            "version": "1.0.0",
            "requires": ["nonexistent-skill-abc"],
        }
        _make_skill_dir(tmpdir, "needs-dep", fm, "instr")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill", "read_skill_reference", "run_skill_script", "list_loaded_skills"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        result = reg.load_skill("needs-dep")
        assert result.success is False
        err_text = "; ".join(result.errors)
        assert "nonexistent-skill-abc" in err_text
        assert "未找到" in err_text
    print("✅ test_c12_6_missing_skill_dependency_blocks_load 通过")


def test_c12_7_version_conflict_blocks_load():
    """【C12】依赖技能版本不满足时，加载失败"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # base-skill 版本 0.5.0
        _make_skill_dir(tmpdir, "base-skill",
            {"name":"base-skill","description":"base","version":"0.5.0"}, "instr")
        # app-skill 要求 base-skill>=1.0.0
        _make_skill_dir(tmpdir, "app-skill",
            {"name":"app-skill","description":"app","version":"1.0.0",
             "requires":["base-skill>=1.0.0"]}, "instr")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        result = reg.load_skill("app-skill")
        assert result.success is False
        err_text = "; ".join(result.errors)
        assert "版本不满足" in err_text or "版本" in err_text
    print("✅ test_c12_7_version_conflict_blocks_load 通过")


def test_c12_8_satisfied_dependencies_load_success():
    """【C12】依赖全部满足时可以正常加载"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "base-skill",
            {"name":"base-skill","description":"base","version":"1.2.0"}, "base instructions")
        _make_skill_dir(tmpdir, "app-skill",
            {"name":"app-skill","description":"app","version":"1.0.0",
             "requires":["base-skill>=1.0.0"], "allowed_tools":[]}, "app instructions")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        # 先加载依赖
        base_result = reg.load_skill("base-skill")
        assert base_result.success, f"base-skill 加载失败: {base_result.errors}"
        
        # 再加载app-skill应该成功（依赖已满足）
        app_result = reg.load_skill("app-skill")
        assert app_result.success, f"app-skill 加载失败: {app_result.errors}"
        assert "app instructions" in app_result.full_prompt
    print("✅ test_c12_8_satisfied_dependencies_load_success 通过")


def test_c12_9_circular_dependency_detected():
    """【C12】循环依赖（a->b->a）能被检测到"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "a-skill",
            {"name":"a-skill","description":"a","version":"1.0.0","requires":["b-skill"]}, "a")
        _make_skill_dir(tmpdir, "b-skill",
            {"name":"b-skill","description":"b","version":"1.0.0","requires":["a-skill"]}, "b")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        result = reg.load_skill("a-skill")
        assert result.success is False
        err_text = "; ".join(result.errors)
        assert "循环依赖" in err_text
    print("✅ test_c12_9_circular_dependency_detected 通过")


def test_c12_10_check_skill_dependencies_api():
    """【C12】check_skill_dependencies公开API返回正确结构"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "good-skill",
            {"name":"good-skill","description":"g","version":"1.0.0","allowed_tools":[]}, "g")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        # 不存在的技能
        r = reg.check_skill_dependencies("nonexistent")
        assert r['ok'] is False
        assert any("不存在" in e for e in r['errors'])
        
        # 正常技能
        r = reg.check_skill_dependencies("good-skill")
        assert r['ok'] is True
        assert r['errors'] == []
    print("✅ test_c12_10_check_skill_dependencies_api 通过")


def test_c12_11_disabled_dependency_blocks_load():
    """【C12】依赖的技能被禁用时，加载失败并提示"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "base-skill",
            {"name":"base-skill","description":"base","version":"1.0.0"}, "b")
        _make_skill_dir(tmpdir, "top-skill",
            {"name":"top-skill","description":"top","version":"1.0.0","requires":["base-skill"]}, "t")
        
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())
        
        reg.disable_skill("base-skill")
        result = reg.load_skill("top-skill")
        assert result.success is False
        assert any("禁用" in e for e in result.errors)
    print("✅ test_c12_11_disabled_dependency_blocks_load 通过")


def test_c12_12_check_skill_dependencies_keys_consistent():
    """【C12】check_skill_dependencies 两分支返回的 key 完全一致（含 skill_name/missing_tools/missing_skills）"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "key-demo",
            {"name":"key-demo","description":"k","version":"1.0.0","allowed_tools":[]}, "k")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())

        expected = {'skill_name', 'ok', 'errors', 'warnings', 'missing_tools', 'missing_skills'}

        # 分支一：技能不存在
        r_none = reg.check_skill_dependencies("nonexistent")
        assert set(r_none.keys()) == expected, r_none.keys()
        assert r_none['skill_name'] == "nonexistent"
        assert r_none['ok'] is False
        assert r_none['missing_tools'] == []
        assert r_none['missing_skills'] == []

        # 分支二：技能存在且依赖满足
        r_ok = reg.check_skill_dependencies("key-demo")
        assert set(r_ok.keys()) == expected, r_ok.keys()
        assert r_ok['skill_name'] == "key-demo"
        assert r_ok['ok'] is True
    print("✅ test_c12_12_check_skill_dependencies_keys_consistent 通过")


def test_c12_13_check_skill_dependencies_missing_lists():
    """【C12】missing_tools/missing_skills 返回真实缺失清单，且与 errors 一致"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fm = {
            "name": "top-demo",
            "description": "依赖缺失示例",
            "version": "1.0.0",
            "requires": ["ghost-skill"],
            "allowed_tools": ["web_search"],
        }
        _make_skill_dir(tmpdir, "top-demo", fm, "instr")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())

        r = reg.check_skill_dependencies("top-demo")
        assert r['ok'] is False
        assert "web_search" in r['missing_tools'], r['missing_tools']
        assert "ghost-skill" in r['missing_skills'], r['missing_skills']
        # 结构化清单应与 errors 文本一致
        assert any("web_search" in e for e in r['errors'])
        assert any("ghost-skill" in e for e in r['errors'])
    print("✅ test_c12_13_check_skill_dependencies_missing_lists 通过")


def test_c12_14_disabled_dependency_in_missing_skills():
    """【C12】依赖技能被禁用时，也应出现在 missing_skills 清单中"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "base-dep",
            {"name":"base-dep","description":"b","version":"1.0.0"}, "b")
        _make_skill_dir(tmpdir, "top-dep",
            {"name":"top-dep","description":"t","version":"1.0.0","requires":["base-dep"]}, "t")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        class FakeTR:
            def list_names(self): return ["load_skill","read_skill_reference","run_skill_script","list_loaded_skills","disable_skill","enable_skill","disable_skills_by_tag","enable_skills_by_tag","check_skill_dependencies"]
            def get_tool(self, name): return None
        mw.bind_tool_registry(FakeTR())

        reg.disable_skill("base-dep")
        r = reg.check_skill_dependencies("top-dep")
        assert r['ok'] is False
        assert "base-dep" in r['missing_skills'], r['missing_skills']
    print("✅ test_c12_14_disabled_dependency_in_missing_skills 通过")


# ========== C6-2 脚本沙箱增强：运行时参数校验与拦截 ==========

def test_c62_7_script_param_type_auto_conversion_at_runtime():
    """【C6-2】脚本执行时int/bool/str自动类型转换并正确传给脚本"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "greet-skill", [
            {"name": "greet.py", "params": [
                {"name": "name", "type": "str", "required": True, "max_length": 50},
                {"name": "age", "type": "int", "required": True, "min": 0, "max": 150},
                {"name": "verbose", "type": "bool", "default": False},
            ]}
        ], "greet.py",
        'import argparse\n'
        'p=argparse.ArgumentParser()\n'
        'p.add_argument("--name")\n'
        'p.add_argument("--age",type=int)\n'
        'p.add_argument("--verbose",action="store_true")\n'
        'a=p.parse_args()\n'
        'print(f"OK name={a.name} age={a.age} verbose={a.verbose}")\n'
        )
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        # age传字符串，应自动转int；verbose缺省，应走默认False
        out = reg.run_script("greet-skill", "greet.py", {"name": "张三", "age": "30"})
        assert "name=张三" in out
        assert "age=30" in out
        assert "verbose=False" in out or "verbose=False" in out
        assert "参数校验失败" not in out
    print("✅ test_c62_7_script_param_type_auto_conversion_at_runtime 通过")


def test_c62_8_missing_required_param_blocks_execution():
    """【C6-2】缺必填参数时直接拦截，不启动脚本进程"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "req-skill", [
            {"name": "run.py", "params": [{"name": "title", "type": "str", "required": True}]}
        ], "run.py", "print('should not run')")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        out = reg.run_script("req-skill", "run.py", {})
        assert "参数校验失败" in out
        assert "title" in out
        assert "should not run" not in out
    print("✅ test_c62_8_missing_required_param_blocks_execution 通过")


def test_c62_9_injection_chars_blocked_without_schema():
    """【C6-2】即使脚本未声明Schema，str参数仍默认拦截命令注入字符"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # 不声明scripts_schema
        import yaml
        skill_dir = tmpdir / "plain-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: plain-skill\ndescription: plain\n---\nx", encoding="utf-8")
        (skill_dir / "scripts" / "echo.py").write_text("import argparse; p=argparse.ArgumentParser(); p.add_argument('--msg'); print(p.parse_args().msg)", encoding="utf-8")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        out = reg.run_script("plain-skill", "echo.py", {"msg": "hello; dir"})
        assert "非法字符" in out
        assert "hello" not in out
    print("✅ test_c62_9_injection_chars_blocked_without_schema 通过")


def test_c62_10_sensitive_path_access_blocked():
    """【C6-2】参数指向Windows系统敏感目录时沙箱拦截"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "path-skill", [
            {"name": "write.py", "params": [{"name": "target", "type": "str", "required": True}]}
        ], "write.py", "print('script ran')")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        # 用正斜杠形式模拟系统路径（避免反斜杠先被注入字符检查拦截）
        out = reg.run_script("path-skill", "write.py", {"target": "C:/Windows/System32/hack"})
        assert "沙箱拦截" in out or "敏感路径" in out
        assert "script ran" not in out
    print("✅ test_c62_10_sensitive_path_access_blocked 通过")


def test_c62_11_network_disabled_by_default():
    """【C6-2】【C18更新】默认禁用网络由loader/registry全局配置控制，环境变量中HTTP_PROXY指向无效地址"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "net-skill", [
            {"name": "check.py"}
        ], "check.py",
        'import os\nprint(f"HTTP_PROXY={os.environ.get(\'HTTP_PROXY\',\'\')}")\nprint(f"NET_DISABLED={os.environ.get(\'SKILL_SANDBOX_NETWORK_DISABLED\',\'0\')}")\n',
        allow_network=False)
        # 默认default_allow_network=False，断网
        reg = SkillRegistry(extra_skill_paths=[tmpdir], default_allow_network=False)
        out = reg.run_script("net-skill", "check.py")
        assert "127.0.0.1" in out, "默认应设置无效代理阻断网络"
        assert "NET_DISABLED=1" in out
        # 全局开启allow_network时，应透传网络
        reg2 = SkillRegistry(extra_skill_paths=[tmpdir], default_allow_network=True)
        out2 = reg2.run_script("net-skill", "check.py")
        assert "NET_DISABLED=1" not in out2
    print("✅ test_c62_11_network_disabled_by_default 通过")


def test_c62_12_enum_value_rejected_at_runtime():
    """【C6-2】枚举值不在白名单内时拦截执行"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "enum-skill", [
            {"name": "run.py", "params": [
                {"name": "format", "type": "enum", "enum": ["md", "txt"], "default": "md"}
            ]}
        ], "run.py", "print('ran')")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        out = reg.run_script("enum-skill", "run.py", {"format": "pdf"})
        assert "必须是以下值之一" in out
        assert "ran" not in out
    print("✅ test_c62_12_enum_value_rejected_at_runtime 通过")


# ========== C15 静态资产：read_asset 读取 ==========

def test_c15_2_read_text_asset_returns_content():
    """【C15】读取文本资产应返回内容（markdown代码块包裹）"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        out = reg.read_asset("asset-demo", "hello.txt")
        assert "静态资产: asset-demo/assets/hello.txt" in out
        assert "hello world" in out
        assert "你好" in out
        # 子目录资产
        out2 = reg.read_asset("asset-demo", "templates/tpl.json")
        assert '{"msg":"hi"}' in out2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_2_read_text_asset_returns_content 通过")


def test_c15_3_read_binary_asset_returns_meta_only():
    """【C15】读取二进制资产只返回元信息与绝对路径，不返回正文"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        out = reg.read_asset("asset-demo", "img.png")
        assert "二进制" in out
        assert "绝对路径" in out
        # 不应包含原始字节
        assert "FAKE-PNG-BYTES" not in out
        assert "不能直接作为文本注入对话" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_3_read_binary_asset_returns_meta_only 通过")


def test_c15_4_oversize_text_asset_returns_meta():
    """【C15】超过 200KB 的文本资产不直接返回内容，仅返回元信息"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        out = reg.read_asset("asset-demo", "big.txt")
        assert "超过单次读取上限" in out
        # 内容应未被完整注入（只有文件开头不会出现全量'x'）
        assert out.count("x") < 1000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_4_oversize_text_asset_returns_meta 通过")


def test_c15_5_path_traversal_blocked():
    """【C15】路径穿越应被拦截"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        out = reg.read_asset("asset-demo", "../SKILL.md")
        assert "非法" in out and "assets目录外" in out
        # 绝对路径同样被拦截
        out2 = reg.read_asset("asset-demo", "C:/Windows/System32/cmd.exe")
        assert "非法" in out2 or "不存在" in out2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_5_path_traversal_blocked 通过")


def test_c15_6_asset_not_found():
    """【C15】不存在的资产返回明确错误信息"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        out = reg.read_asset("asset-demo", "no_such_file.txt")
        assert "不存在" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_6_asset_not_found 通过")


# ========== C8 同名技能冲突消解（按来源优先级覆盖） ==========

def test_c8_1_priority_override_project_overrides_builtin():
    """【C8】项目级技能(project)应覆盖框架内置(builtin)同名技能"""
    import shutil
    # 不依赖仓库自带.skills，自建builtin和project两个层级
    tmp_builtin = Path(tempfile.mkdtemp())
    tmp_proj = Path(tempfile.mkdtemp())
    try:
        name = "builtin-demo"
        _make_conflict_skill_dir(tmp_builtin / name, name, "0.1.0", "[builtin]")
        _make_conflict_skill_dir(tmp_proj / name, name, "1.0.0", "[project]")
        reg = SkillRegistry()
        reg.loader.skill_paths = []  # 清空默认路径
        reg.loader.add_skill_path(tmp_builtin, source="builtin")
        reg.loader.add_skill_path(tmp_proj, source="project")
        reg.loader.invalidate_index_cache()
        meta = reg.get_metadata(name)
        assert meta.source == "project"
        assert "[project]" in meta.description
    finally:
        shutil.rmtree(tmp_builtin, ignore_errors=True)
        shutil.rmtree(tmp_proj, ignore_errors=True)
    print("✅ test_c8_1_priority_override_project_overrides_builtin 通过")


def test_c8_2_priority_custom_overrides_project():
    """【C8】自定义路径(custom)应覆盖项目级(project)同名技能"""
    import shutil
    tmp_proj = Path(tempfile.mkdtemp())
    tmp_custom = Path(tempfile.mkdtemp())
    try:
        name = "conflict-demo"
        # 手动把proj路径作为project源、custom作为custom源注入
        _make_conflict_skill_dir(tmp_proj / name, name, "1.0.0", "[project]")
        _make_conflict_skill_dir(tmp_custom / name, name, "2.0.0", "[custom]")
        reg = SkillRegistry()
        # 清空默认路径，避免仓库自带.skills干扰
        reg.loader.skill_paths = [(p, s) for p, s in reg.loader.skill_paths if s != "project"]
        reg.loader.add_skill_path(tmp_proj, source="project")
        reg.loader.add_skill_path(tmp_custom, source="custom")
        reg.loader.invalidate_index_cache()
        meta = reg.get_metadata(name)
        assert meta.source == "custom"
        assert "[custom]" in meta.description
    finally:
        shutil.rmtree(tmp_proj, ignore_errors=True)
        shutil.rmtree(tmp_custom, ignore_errors=True)
    print("✅ test_c8_2_priority_custom_overrides_project 通过")


def test_c8_3_code_skill_overrides_file_skill():
    """【C8】代码型注册技能应覆盖文件型技能（最高优先级）"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        sk_name = "code-override-demo"
        _make_conflict_skill_dir(tmp / sk_name, sk_name, "1.0.0", "[file]")

        class CodeSkill(BaseSkill):
            name = sk_name
            description = sk_name + " [code]"
            version = "2.0.0"
            tags = []
            def execute(self, **kwargs):
                return "code executed"

        reg = SkillRegistry(extra_skill_paths=[tmp])
        # 清空默认project路径避免干扰
        reg.loader.skill_paths = [(p, s) for p, s in reg.loader.skill_paths if s != "project"]
        reg.loader.invalidate_index_cache()
        # 先确认文件型存在
        assert reg.get_metadata(sk_name).source != "code"
        # 注册代码型
        reg.register(CodeSkill)
        meta = reg.get_metadata(sk_name)
        assert meta.source == "code"
        assert "[code]" in meta.description
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c8_3_code_skill_overrides_file_skill 通过")


def test_c8_4_get_skill_source_api():
    """【C8】get_skill_source 应返回正确来源标签"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        name = "source-test"
        _make_conflict_skill_dir(tmp / name, name, "1.0.0", "")
        reg = SkillRegistry(extra_skill_paths=[tmp])
        reg.loader.skill_paths = [(p, s) for p, s in reg.loader.skill_paths if s != "project"]
        reg.loader.invalidate_index_cache()
        assert reg.get_skill_source(name) == "custom"
        assert reg.get_skill_source("not-exist-xxx") is None

        class CodeSkill(BaseSkill):
            name = "code-src"
            description = "code src"
            version = "1.0.0"
            tags = []
            def execute(self, **kwargs): return ""
        reg.register(CodeSkill)
        assert reg.get_skill_source("code-src") == "code"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c8_4_get_skill_source_api 通过")


def test_c8_5_list_conflicts_api():
    """【C8】list_skill_conflicts 应返回同名冲突条目"""
    import shutil
    tmp_proj = Path(tempfile.mkdtemp())
    tmp_custom = Path(tempfile.mkdtemp())
    try:
        name = "conflict-list"
        _make_conflict_skill_dir(tmp_proj / name, name, "1.0.0", "[project]")
        _make_conflict_skill_dir(tmp_custom / name, name, "2.0.0", "[custom]")
        reg = SkillRegistry()
        # 清空默认路径，避免仓库自带.skills干扰
        reg.loader.skill_paths = []
        reg.loader.add_skill_path(tmp_proj, source="project")
        reg.loader.add_skill_path(tmp_custom, source="custom")
        reg.loader.invalidate_index_cache()
        conflicts = reg.list_skill_conflicts()
        names = [c["skill_name"] for c in conflicts]
        assert name in names
        entry = next(c for c in conflicts if c["skill_name"] == name)
        assert entry["new_source"] == "custom"
        assert entry["old_source"] == "project"
        assert entry["winner"] == "new"
    finally:
        shutil.rmtree(tmp_proj, ignore_errors=True)
        shutil.rmtree(tmp_custom, ignore_errors=True)
    print("✅ test_c8_5_list_conflicts_api 通过")


# ==================== 入口 ====================


def test_bug10_disabled_skill_blocked_on_level3_entries():
    """【Bug10修复】技能被禁用后，Level 3 入口（run_script / read_reference / read_asset）
    不得再执行脚本或读出内容；registry API 与 LLM 工具两条路径都要拦，重新启用后恢复。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        skill_dir = _make_script_skill(tmpdir, "bug10-skill", [], "danger.py", "print('BOOM-RAN')")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "note.md").write_text("SECRET-NOTE\n", encoding="utf-8")
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "data.txt").write_text("SECRET-ASSET\n", encoding="utf-8")

        reg = SkillRegistry(extra_skill_paths=[tmpdir], include_default_paths=False)
        reg.scan()
        # 禁用前：三个入口正常工作（行为不回退）
        assert "BOOM-RAN" in reg.run_script("bug10-skill", "danger.py", {"a": "x"})
        assert "SECRET-NOTE" in reg.read_reference("bug10-skill", "note.md")
        assert "SECRET-ASSET" in reg.read_asset("bug10-skill", "data.txt")

        assert "已禁用" in reg.disable_skill("bug10-skill")
        # 禁用后：不得执行、不得泄内容，并给出与 load_skill 同句式的提示
        for out in (
            reg.run_script("bug10-skill", "danger.py", {"a": "x"}),
            reg.read_reference("bug10-skill", "note.md"),
            reg.read_asset("bug10-skill", "data.txt"),
        ):
            assert "BOOM-RAN" not in out and "SECRET-" not in out, f"禁用后仍执行/读出: {out}"
            assert "已被禁用" in out and "enable_skill" in out, f"禁用提示文案不对: {out}"

        # LLM 工具层同样拦截（工具直转 registry 方法）
        tools = {f.__name__: f for f in reg.get_tools()}
        assert "BOOM-RAN" not in tools["run_skill_script"]("bug10-skill", "danger.py", {"a": "x"})
        assert "SECRET-" not in tools["read_skill_reference"]("bug10-skill", "note.md")
        assert "SECRET-" not in tools["read_skill_asset"]("bug10-skill", "data.txt")

        # 重新启用后恢复可用
        reg.enable_skill("bug10-skill")
        assert "BOOM-RAN" in reg.run_script("bug10-skill", "danger.py", {"a": "x"})
    print("✅ test_bug10_disabled_skill_blocked_on_level3_entries 通过")


def test_bug10_disabled_code_skill_blocked_on_execute_and_tag_disable():
    """【Bug10修复】代码型技能的 execute_skill / aexecute_skill 受禁用约束（同步与异步一致），
    按 tag 批量禁用同样拦截；不存在的技能与未禁用技能行为不变。"""
    import asyncio
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "tagged-s", {
            "name": "tagged-s", "description": "带标签技能", "version": "1.0.0", "tags": ["dev"],
        })
        reg = SkillRegistry(extra_skill_paths=[tmpdir], include_default_paths=False)
        reg.scan()

        @skill(name="bug10-code", description="代码型技能")
        def bug10_code():
            return "CODE-EXECUTED"

        reg.register(bug10_code)
        # 未禁用：正常执行
        assert reg.execute_skill("bug10-code") == "CODE-EXECUTED"
        assert asyncio.run(reg.aexecute_skill("bug10-code")) == "CODE-EXECUTED"
        # 单个禁用：同步与异步入口都拦
        reg.disable_skill("bug10-code")
        assert "CODE-EXECUTED" not in str(reg.execute_skill("bug10-code"))
        assert "CODE-EXECUTED" not in str(asyncio.run(reg.aexecute_skill("bug10-code")))
        assert "已被禁用" in str(reg.execute_skill("bug10-code"))
        reg.enable_skill("bug10-code")
        assert reg.execute_skill("bug10-code") == "CODE-EXECUTED"
        # 按 tag 禁用：Level 3 文件型入口同样拦截
        assert "tagged-s" in reg.disable_skills_by_tag("dev")
        assert "已被禁用" in reg.read_reference("tagged-s", "a.md")
        assert "已被禁用" in str(reg.execute_skill("tagged-s"))
        reg.enable_skills_by_tag("dev")
        assert "已被禁用" not in reg.read_reference("tagged-s", "a.md")
        # 不存在的技能：不得误报为“已被禁用”，保持原有不存在文案
        ghost = reg.run_script("no-such-skill", "x.py", {})
        assert "已被禁用" not in ghost and "不存在" in ghost, f"不存在技能文案变了: {ghost}"
    print("✅ test_bug10_disabled_code_skill_blocked_on_execute_and_tag_disable 通过")


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
    test_bug10_disabled_skill_blocked_on_level3_entries()
    test_bug10_disabled_code_skill_blocked_on_execute_and_tag_disable()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
