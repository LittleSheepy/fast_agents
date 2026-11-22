"""
Skill模块单元测试
覆盖：BaseSkill/@skill装饰器、元数据校验、注册/扫描/加载、中间件、安全加固
全离线运行，无外部依赖
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import (
    BaseSkill, skill,
    SkillRegistry, SkillLoader, SkillMiddleware,
    SkillMetadata, SkillLoadResult,
    SkillNotFoundError, SkillLoadError,
    ScriptParam, ScriptSchema,  # 【C6-2】
)


def test_b1_base_skill_basic():
    """测试BaseSkill基类基本功能"""
    class TestSkill(BaseSkill):
        name = "test-skill"
        description = "测试技能"
        version = "1.0.0"
        tags = ["test"]
        
        def execute(self, **kwargs):
            return "执行成功"
    
    s = TestSkill()
    assert s.name == "test-skill"
    assert s.metadata.name == "test-skill"
    assert s.metadata.description == "测试技能"
    assert s.metadata.tags == ["test"]
    assert s.execute() == "执行成功"
    assert "执行test-skill技能" in s.get_instructions()
    print("✅ test_b1_base_skill_basic 通过")


def test_b2_base_skill_validation():
    """测试BaseSkill元数据校验"""
    # name为空应该报错
    try:
        class BadSkill1(BaseSkill):
            name = ""
            description = "test"
            def execute(self, **kwargs): pass
        BadSkill1()
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "必须定义name属性" in str(e)
    
    # description为空应该报错
    try:
        class BadSkill2(BaseSkill):
            name = "bad-skill"
            description = ""
            def execute(self, **kwargs): pass
        BadSkill2()
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "必须定义description属性" in str(e)
    
    # 非法name格式应该报错
    try:
        class BadSkill3(BaseSkill):
            name = "Bad-Skill"  # 大写字母
            description = "test"
            def execute(self, **kwargs): pass
        BadSkill3()
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "元数据校验失败" in str(e)
    print("✅ test_b2_base_skill_validation 通过")


def test_b3_skill_decorator():
    """测试@skill装饰器"""
    @skill(
        name="calc",
        description="执行计算",
        tags=["math"]
    )
    def calc(a: int, b: int) -> int:
        """加法计算：返回a+b"""
        return a + b
    
    # 装饰后的函数可以直接调用
    assert calc(1, 2) == 3
    
    # 可以通过skill_class创建实例
    s = calc.skill_class()
    assert s.name == "calc"
    assert "加法计算" in s.get_instructions()
    assert s.execute(a=3, b=4) == 7
    print("✅ test_b3_skill_decorator 通过")


def test_b4_skill_metadata_validate():
    """测试SkillMetadata校验规则"""
    # 合法元数据
    meta = SkillMetadata(name="valid-name", description="valid desc")
    errors = meta.validate()
    assert len(errors) == 0
    
    # name超长
    meta2 = SkillMetadata(name="a"*65, description="desc")
    errors = meta2.validate()
    assert any("长度不能超过64" in e for e in errors)
    
    # description超长
    meta3 = SkillMetadata(name="test", description="a"*1025)
    errors = meta3.validate()
    assert any("长度不能超过1024" in e for e in errors)
    
    # 连续连字符
    meta4 = SkillMetadata(name="test--name", description="desc")
    errors = meta4.validate()
    assert any("连续连字符" in e for e in errors)
    
    # 以连字符开头
    meta5 = SkillMetadata(name="-test", description="desc")
    errors = meta5.validate()
    assert len(errors) > 0
    print("✅ test_b4_skill_metadata_validate 通过")


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


def test_b7_loader_scan_file_skill():
    """测试加载器扫描文件型Skill"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 创建一个测试skill
        skill_dir = tmp / "test-file-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test-file-skill
description: 测试文件技能
version: "1.0"
tags: [test]
---

# 测试文件技能

这是测试文件技能的正文。
""", encoding="utf-8")
        
        loader = SkillLoader(extra_paths=[tmp])
        meta = loader.scan_skills()
        assert "test-file-skill" in meta
        assert meta["test-file-skill"].description == "测试文件技能"
        assert meta["test-file-skill"].skill_type == "file"
        
        # 加载完整skill
        result = loader.load_skill("test-file-skill")
        assert result.success is True
        assert "测试文件技能的正文" in result.instructions
        assert "Skill: test-file-skill" in result.full_prompt
        print("✅ test_b7_loader_scan_file_skill 通过")


def test_b8_loader_skill_not_found():
    """测试加载不存在的Skill"""
    loader = SkillLoader(extra_paths=[])
    result = loader.load_skill("not-exist-skill")
    assert result.success is False
    assert any("不存在" in e for e in result.errors)
    print("✅ test_b8_loader_skill_not_found 通过")


def test_b9_script_path_traversal_protection():
    """测试脚本执行路径穿越防护"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        skill_dir = tmp / "safe-skill"
        skill_dir.mkdir()
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        
        # 创建SKILL.md
        (skill_dir / "SKILL.md").write_text("""---
name: safe-skill
description: 安全测试技能
---
test
""", encoding="utf-8")
        
        loader = SkillLoader(extra_paths=[tmp])
        loader.scan_skills()
        
        # 尝试路径穿越
        result = loader.run_script("safe-skill", "../../../etc/passwd")
        assert "非法脚本路径" in result
        print("✅ test_b9_script_path_traversal_protection 通过")


def test_b10_script_extension_validation():
    """测试脚本扩展名校验"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        skill_dir = tmp / "ext-skill"
        skill_dir.mkdir()
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        
        # 创建非法扩展名文件
        (scripts_dir / "dangerous.exe").write_text("malicious", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text("""---
name: ext-skill
description: 扩展名测试
---
test
""", encoding="utf-8")
        
        loader = SkillLoader(extra_paths=[tmp])
        loader.scan_skills()
        
        result = loader.run_script("ext-skill", "dangerous.exe")
        assert "不支持执行该类型脚本" in result
        print("✅ test_b10_script_extension_validation 通过")


def test_b11_reference_path_traversal_protection():
    """测试参考文档路径穿越防护"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        skill_dir = tmp / "ref-safe-skill"
        skill_dir.mkdir()
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text("""---
name: ref-safe-skill
description: 参考文档安全测试
---
test
""", encoding="utf-8")
        
        loader = SkillLoader(extra_paths=[tmp])
        loader.scan_skills()
        
        result = loader.read_reference("ref-safe-skill", "../../../../windows/system32/config/sam")
        assert "非法参考文档路径" in result
        print("✅ test_b11_reference_path_traversal_protection 通过")


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


def _make_skill_dir(tmpdir: Path, name: str, frontmatter: dict, instructions: str = "test") -> Path:
    """在临时目录下创建一个文件型Skill目录"""
    import yaml
    skill_dir = tmpdir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    content = "---\n" + yaml.dump(frontmatter, allow_unicode=True) + "---\n" + instructions
    skill_file.write_text(content, encoding="utf-8")
    return skill_dir


def test_c12_3_metadata_requires_field_validation():
    """【C12】元数据requires字段格式校验正确"""
    # 合法格式
    meta_ok = SkillMetadata(name="ok-skill", description="ok", requires=["dep-a", "dep-b>=1.0.0"])
    assert meta_ok.validate() == []
    
    # 非法格式（带大写）
    meta_bad = SkillMetadata(name="bad-skill", description="bad", requires=["BadDep"])
    errs = meta_bad.validate()
    assert any("requires" in e for e in errs)
    
    # 版本号非法
    meta_ver = SkillMetadata(name="ver-skill", description="v", min_fast_agents_version="not-a-version")
    errs = meta_ver.validate()
    assert any("min_fast_agents_version" in e for e in errs)
    print("✅ test_c12_3_metadata_requires_field_validation 通过")


def test_c12_4_loader_parses_requires_fields():
    """【C12】loader能正确解析YAML中的requires/requires_tools/min_version字段"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fm = {
            "name": "dep-skill",
            "description": "带依赖的技能",
            "version": "1.0.0",
            "requires": ["base-skill>=1.0.0"],
            "requires_tools": ["web_search"],
            "min_fast_agents_version": "0.1.0",
        }
        _make_skill_dir(tmpdir, "dep-skill", fm, "instructions")
        loader = SkillLoader(extra_paths=[tmpdir])
        idx = loader.scan_skills()
        assert "dep-skill" in idx
        meta = idx["dep-skill"]
        assert meta.requires == ["base-skill>=1.0.0"]
        assert meta.requires_tools == ["web_search"]
        assert meta.min_fast_agents_version == "0.1.0"
    print("✅ test_c12_4_loader_parses_requires_fields 通过")


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


# ===================== C6-2 脚本沙箱增强测试 =====================

def test_c62_1_script_param_int_validation():
    """【C6-2】ScriptParam int类型转换与范围校验"""
    p = ScriptParam(name="age", type="int", required=True, min=0, max=150)
    ok, err, val = p.validate_value("28")
    assert ok is True and val == 28 and isinstance(val, int)
    ok, err, val = p.validate_value("abc")
    assert ok is False and "类型错误" in err
    ok, err, val = p.validate_value("999")
    assert ok is False and "最大值" in err
    ok, err, val = p.validate_value("-1")
    assert ok is False and "最小值" in err
    ok, err, val = p.validate_value(None)
    assert ok is False and "必填" in err
    print("✅ test_c62_1_script_param_int_validation 通过")


def test_c62_2_script_param_bool_conversion():
    """【C6-2】ScriptParam bool类型自动转换"""
    p = ScriptParam(name="verbose", type="bool")
    for true_val in [True, "true", "1", "yes", "on", "Y"]:
        ok, _, val = p.validate_value(true_val)
        assert ok and val is True, f"{true_val}应转True"
    for false_val in [False, "false", "0", "no", "off", "N"]:
        ok, _, val = p.validate_value(false_val)
        assert ok and val is False, f"{false_val}应转False"
    ok, err, _ = p.validate_value("maybe")
    assert ok is False and "布尔值" in err
    print("✅ test_c62_2_script_param_bool_conversion 通过")


def test_c62_3_script_param_enum_validation():
    """【C6-2】ScriptParam enum枚举校验"""
    p = ScriptParam(name="fmt", type="enum", enum=["md", "txt", "json"], default="md")
    ok, _, val = p.validate_value("md")
    assert ok and val == "md"
    ok, err, _ = p.validate_value("pdf")
    assert ok is False and "必须是以下值之一" in err
    ok, _, val = p.validate_value(None)
    assert ok and val == "md", "缺省值应取default"
    print("✅ test_c62_3_script_param_enum_validation 通过")


def test_c62_4_script_param_str_injection_protection():
    """【C6-2】str参数命令注入字符自动拦截"""
    p = ScriptParam(name="name", type="str", required=True, max_length=50)
    for bad in ["a;rm -rf /", "b&whoami", "c|dir", "d`cmd`", "e$(x)", "f(g)", "h>out", "i<in", "j\\path", "k\nl"]:
        ok, err, _ = p.validate_value(bad)
        assert ok is False and "非法字符" in err, f"'{bad}'应被拦截"
    ok, _, val = p.validate_value("张三")
    assert ok and val == "张三"
    ok, err, _ = p.validate_value("a" * 100)
    assert ok is False and "长度" in err
    print("✅ test_c62_4_script_param_str_injection_protection 通过")


def test_c62_5_script_schema_validate_args():
    """【C6-2】ScriptSchema整体参数校验（必填+类型+默认值）"""
    schema = ScriptSchema(script_name="greet.py", params=[
        ScriptParam(name="name", type="str", required=True, max_length=50),
        ScriptParam(name="age", type="int", required=True, min=0, max=150),
        ScriptParam(name="city", type="str", default="上海"),
    ])
    ok, errs, out = schema.validate_args({"name": "张三", "age": "28"})
    assert ok is True
    assert out["name"] == "张三"
    assert out["age"] == 28
    assert out["city"] == "上海", "缺省参数应补默认值"
    # 缺必填
    ok, errs, _ = schema.validate_args({"name": "李四"})
    assert ok is False and any("age" in e for e in errs)
    print("✅ test_c62_5_script_schema_validate_args 通过")


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


def test_c62_6_loader_parses_scripts_schema_from_json():
    """【C6-2】【C18更新】SkillLoader能从scripts/<脚本名>.schema.json解析参数Schema（不再读frontmatter）"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_script_skill(tmpdir, "calc-skill", [
            {"name": "add.py", "description": "加法", "params": [
                {"name": "a", "type": "int", "required": True},
                {"name": "b", "type": "int", "required": True},
            ]}
        ], "add.py", "import argparse; p=argparse.ArgumentParser(); p.add_argument('--a',type=int); p.add_argument('--b',type=int); a=p.parse_args(); print(a.a+a.b)")
        loader = SkillLoader(extra_paths=[tmpdir])
        idx = loader.scan_skills()
        meta = idx["calc-skill"]
        # 【C18】SkillMetadata不再包含scripts_schema字段，schema只在加载后SkillLoadResult里
        assert not hasattr(meta, "scripts_schema") or not meta.scripts_schema
        # 加载技能后从scripts目录读取schema
        result = loader.load_skill("calc-skill")
        assert "add.py" in result.scripts_schemas or "add" in result.scripts_schemas
        schema = result.get_script_schema("add.py")
        assert schema is not None
        assert len(schema.params) == 2
        assert schema.params[0].name == "a" and schema.params[0].type == "int"
        # full_prompt应展示参数说明
        assert "--a (int)" in result.full_prompt
    print("✅ test_c62_6_loader_parses_scripts_schema_from_json 通过")


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


def test_c18_1_schema_invalid_json_ignored():
    """【C18】无效的schema.json文件自动忽略，不影响技能加载"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        skill_dir = tmpdir / "bad-schema-skill"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: bad-schema-skill\ndescription: test\n---\nx", encoding="utf-8")
        (scripts_dir / "run.py").write_text("print('ok')", encoding="utf-8")
        # 写一个损坏的json文件
        (scripts_dir / "run.schema.json").write_text("{invalid json", encoding="utf-8")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        result = reg.load_skill("bad-schema-skill")
        assert result.success is True
        # schema解析失败被忽略，脚本仍可执行
        out = reg.run_script("bad-schema-skill", "run.py", {})
        assert "ok" in out
    print("✅ test_c18_1_schema_invalid_json_ignored 通过")


def test_c18_2_frontmatter_scripts_ignored():
    """【C18】SKILL.md frontmatter中即使写了scripts段，也不会被解析（完全忽略，仅读schema.json）"""
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        skill_dir = tmpdir / "old-schema-skill"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        # frontmatter里故意写旧版的scripts段
        fm = {"name":"old-schema-skill", "description":"test", "scripts": [{"name":"run.py", "params":[{"name":"x", "required": True}]}]}
        (skill_dir / "SKILL.md").write_text("---\n"+yaml.dump(fm, allow_unicode=True)+"---\nx", encoding="utf-8")
        (scripts_dir / "run.py").write_text("print('ok')", encoding="utf-8")
        # 不写schema.json
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        result = reg.load_skill("old-schema-skill")
        assert result.success is True
        assert len(result.scripts_schemas) == 0, "frontmatter中的scripts段应被忽略，不解析"
        # 没有schema时，参数不会被校验，命令注入防护仍生效
        out = reg.run_script("old-schema-skill", "run.py", {"x": "hello;dir"})
        assert "非法字符" in out
    print("✅ test_c18_2_frontmatter_scripts_ignored 通过")


def test_c18_3_skill_metadata_no_scripts_or_sandbox_fields():
    """【C18】SkillMetadata不再包含scripts_schema/allow_network/allow_write_paths/env_whitelist字段"""
    meta = SkillMetadata(name="test", description="test")
    assert not hasattr(meta, "scripts_schema") or meta.scripts_schema == {}
    assert not hasattr(meta, "allow_network")
    assert not hasattr(meta, "allow_write_paths")
    assert not hasattr(meta, "env_whitelist")
    d = meta.to_dict()
    assert "scripts_schema" not in d
    assert "allow_network" not in d
    assert "allow_write_paths" not in d
    assert "env_whitelist" not in d
    print("✅ test_c18_3_skill_metadata_no_scripts_or_sandbox_fields 通过")


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


# ========== 【C14】技能热重载测试 ==========

def test_c14_1_mtime_map_builds():
    """【C14】mtime快照能正确构建"""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "demo-skill",
            {"name":"demo-skill","description":"demo","version":"1.0.0"}, "demo content")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        # 构建mtime快照
        mtime_map = reg.loader._build_dir_mtime_map()
        assert "demo-skill" in mtime_map
        assert mtime_map["demo-skill"] > 0
    print("✅ test_c14_1_mtime_map_builds 通过")


def test_c14_2_detect_modified_skill():
    """【C14】修改SKILL.md后能检测到变更"""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "edit-skill",
            {"name":"edit-skill","description":"original","version":"1.0.0"}, "original content")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        old_map = reg.loader._build_dir_mtime_map()
        # 等待一点时间确保mtime变化
        time.sleep(0.1)
        # 修改SKILL.md
        skill_md = tmpdir / "edit-skill" / "SKILL.md"
        new_content = "---\nname: edit-skill\ndescription: modified\nversion: 1.0.1\n---\nmodified content\n"
        skill_md.write_text(new_content, encoding="utf-8")
        # 检测变更
        changes = reg.loader.detect_changes(old_map)
        assert "edit-skill" in changes["modified"]
    print("✅ test_c14_2_detect_modified_skill 通过")


def test_c14_3_reload_skill_updates_metadata():
    """【C14】reload_skill能更新元数据和内容缓存"""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "reload-skill",
            {"name":"reload-skill","description":"original","version":"1.0.0"}, "v1 content")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        # 先加载技能
        result = reg.load_skill("reload-skill")
        assert result.success
        # "original" 在 metadata.description / full_prompt 中，不是 result 字符串本身
        assert result.metadata is not None and "original" in result.metadata.description
        assert reg.is_skill_loaded("reload-skill")
        # 修改文件
        time.sleep(0.1)
        skill_md = tmpdir / "reload-skill" / "SKILL.md"
        skill_md.write_text(
            "---\nname: reload-skill\ndescription: reloaded\nversion: 2.0.0\n---\nv2 new content\n",
            encoding="utf-8")
        # 调用reload_skill
        new_meta = reg.loader.reload_skill("reload-skill")
        assert new_meta is not None
        assert new_meta.description == "reloaded"
        assert new_meta.version == "2.0.0"
        # 原已加载缓存被清理
        assert "reload-skill" not in reg.loader._loaded_skills
    print("✅ test_c14_3_reload_skill_updates_metadata 通过")


def test_c14_4_detect_new_skill_added():
    """【C14】增量重扫能发现新增技能"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "first-skill",
            {"name":"first-skill","description":"first","version":"1.0.0"}, "first")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        # 初始扫描
        reg.scan_skills()
        assert "first-skill" in [m.name for m in reg.list_skills_metadata()]
        assert "second-skill" not in [m.name for m in reg.list_skills_metadata()]
        # 添加第二个技能
        _make_skill_dir(tmpdir, "second-skill",
            {"name":"second-skill","description":"second","version":"1.0.0"}, "second")
        # 增量重扫
        result = reg.loader.rescan_incremental()
        assert "second-skill" in result["added"]
        assert "second-skill" in [m.name for m in reg.list_skills_metadata()]
    print("✅ test_c14_4_detect_new_skill_added 通过")


def test_c14_5_detect_removed_skill():
    """【C14】检测到技能目录删除后，自动清理索引和已加载状态"""
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _make_skill_dir(tmpdir, "del-skill",
            {"name":"del-skill","description":"to delete","version":"1.0.0"}, "content")
        reg = SkillRegistry(extra_skill_paths=[tmpdir])
        mw = SkillMiddleware(registry=reg)
        # 先加载
        reg.load_skill("del-skill")
        assert reg.is_skill_loaded("del-skill")
        # 删除目录
        shutil.rmtree(tmpdir / "del-skill")
        # 检测变更
        old_map = reg.loader._build_dir_mtime_map()
        # 强制刷新（因为目录已经删了，新map不会有del-skill）
        new_map = reg.loader._build_dir_mtime_map()
        changes = reg.loader.detect_changes({"del-skill": old_map.get("del-skill", 0)})
        assert "del-skill" in changes["removed"]
        # 调用_on_skill_removed清理
        mw._on_skill_removed("del-skill")
        assert "del-skill" not in [m.name for m in reg.list_skills_metadata()]
        assert not reg.is_skill_loaded("del-skill")
    print("✅ test_c14_5_detect_removed_skill 通过")


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


def run_all():
    """运行所有测试"""
    tests = [
        test_b1_base_skill_basic,
        test_b2_base_skill_validation,
        test_b3_skill_decorator,
        test_b4_skill_metadata_validate,
        test_b5_registry_register_code_skill,
        test_b6_registry_unregister,
        test_b7_loader_scan_file_skill,
        test_b8_loader_skill_not_found,
        test_b9_script_path_traversal_protection,
        test_b10_script_extension_validation,
        test_b11_reference_path_traversal_protection,
        test_b12_middleware_inject_index_prompt,
        test_b13_middleware_lru_cache,
        test_b14_middleware_unload_after_turn,
        test_b15_default_skill_paths,
        test_b16_build_skill_tools,
        test_b17_get_index_prompt,
        test_b18_list_skills,
        # C10 allowed_tools
        test_c10_1_no_skill_no_filter,
        test_c10_2_load_skill_with_allowed_tools,
        test_c10_3_skill_unload_restores_all_tools,
        test_c10_4_multiple_skills_union,
        test_c10_5_wildcard_allows_all,
        test_c10_6_clear_loaded_resets,
        test_c10_7_list_loaded_skills_tool,
        # C11 enable/disable
        test_c11_1_disable_single_skill_hides_from_index,
        test_c11_2_disabled_skill_cannot_be_loaded,
        test_c11_3_enable_skill_restores,
        test_c11_4_disable_by_tag,
        test_c11_5_disable_all_and_enable_all,
        test_c11_6_disable_loaded_skill_auto_unloads,
        test_c11_7_disable_nonexistent_skill_returns_error,
        test_c11_8_list_skills_includes_disabled_status,
        # C12 dependencies
        test_c12_1_version_parse_and_compare,
        test_c12_2_dep_spec_parse,
        test_c12_3_metadata_requires_field_validation,
        test_c12_4_loader_parses_requires_fields,
        test_c12_5_missing_tool_blocks_load,
        test_c12_6_missing_skill_dependency_blocks_load,
        test_c12_7_version_conflict_blocks_load,
        test_c12_8_satisfied_dependencies_load_success,
        test_c12_9_circular_dependency_detected,
        test_c12_10_check_skill_dependencies_api,
        test_c12_11_disabled_dependency_blocks_load,
        test_c12_12_check_skill_dependencies_keys_consistent,
        test_c12_13_check_skill_dependencies_missing_lists,
        test_c12_14_disabled_dependency_in_missing_skills,
        # C6-2 脚本沙箱增强
        test_c62_1_script_param_int_validation,
        test_c62_2_script_param_bool_conversion,
        test_c62_3_script_param_enum_validation,
        test_c62_4_script_param_str_injection_protection,
        test_c62_5_script_schema_validate_args,
        test_c62_6_loader_parses_scripts_schema_from_json,  # 【C18更新】从schema.json读取
        test_c62_7_script_param_type_auto_conversion_at_runtime,
        test_c62_8_missing_required_param_blocks_execution,
        test_c62_9_injection_chars_blocked_without_schema,
        test_c62_10_sensitive_path_access_blocked,
        test_c62_11_network_disabled_by_default,
        test_c62_12_enum_value_rejected_at_runtime,
        # C18 方案B：schema外置 + 沙箱全局配置
        test_c18_1_schema_invalid_json_ignored,
        test_c18_2_frontmatter_scripts_ignored,
        test_c18_3_skill_metadata_no_scripts_or_sandbox_fields,
        # C14 技能热重载
        test_c14_1_mtime_map_builds,
        test_c14_2_detect_modified_skill,
        test_c14_3_reload_skill_updates_metadata,
        test_c14_4_detect_new_skill_added,
        test_c14_5_detect_removed_skill,
        test_c14_6_enable_disable_hot_reload,
        test_c14_7_reload_skills_now_manual_trigger,
        test_c14_8_hot_reload_default_disabled,
        # C17 性能优化：索引缓存与1+N扫描治理
        test_c17_1_index_cache_skips_rescan_when_unchanged,
        test_c17_2_get_enabled_metadata_no_nplus1_scan,
        test_c17_3_invalidate_cache_forces_rescan,
        # C15 静态资产读取
        test_c15_1_assets_scanned_into_metadata,
        test_c15_2_read_text_asset_returns_content,
        test_c15_3_read_binary_asset_returns_meta_only,
        test_c15_4_oversize_text_asset_returns_meta,
        test_c15_5_path_traversal_blocked,
        test_c15_6_asset_not_found,
        # C8 同名技能冲突消解（按来源优先级覆盖）
        test_c8_1_priority_override_project_overrides_builtin,
        test_c8_2_priority_custom_overrides_project,
        test_c8_3_code_skill_overrides_file_skill,
        test_c8_4_get_skill_source_api,
        test_c8_5_list_conflicts_api,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    total = len(tests)
    passed = total - failed
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    if failed == 0:
        print("✅ 所有测试通过！")
    else:
        print(f"❌ {failed} 个测试失败")

def test_c17_1_index_cache_skips_rescan_when_unchanged():
    """【C17 性能优化】get_index_prompt在根目录未变时不应触发实际磁盘扫描
    通过给 _parse_skill_metadata 打桩统计真正执行解析的次数，验证缓存命中。
    """
    from fast_agents.skills import SkillRegistry
    reg = SkillRegistry()

    # 创建3个测试技能目录
    tmp_root = Path(tempfile.mkdtemp())
    for sname in ["sk-a", "sk-b", "sk-c"]:
        d = tmp_root / sname
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {sname}\ndescription: desc of {sname}\nversion: 1.0.0\n---\n# {sname}\n",
            encoding="utf-8",
        )
    reg.add_skill_path(tmp_root)

    # 统计真正执行全量重建的次数：慢路径每解析一个技能目录都会调用一次 _parse_skill_metadata，
    # 缓存命中时完全不会调用，因此用它作为判断“是否真的重扫”的探针（比观察 keys 变化更精确）。
    real_rescan_count = [0]
    orig = reg.loader._parse_skill_metadata

    def counting_parse(*args, **kwargs):
        real_rescan_count[0] += 1
        return orig(*args, **kwargs)

    reg.loader._parse_skill_metadata = counting_parse
    try:
        # 第一次：应执行全量重建
        real_rescan_count[0] = 0
        p1 = reg.get_index_prompt()
        assert "sk-a" in p1 and "sk-b" in p1 and "sk-c" in p1, "第一次调用应返回所有技能"
        assert real_rescan_count[0] >= 3, f"第一次get_index_prompt应真实解析技能，实际{real_rescan_count[0]}"

        # 后续100次：根目录mtime未变，应0次真实重扫
        real_rescan_count[0] = 0
        for _ in range(100):
            p = reg.get_index_prompt()
        assert real_rescan_count[0] == 0, f"后续调用应全部命中缓存，实际重扫{real_rescan_count[0]}次"
        # 内容也应一致
        assert p == p1, "缓存命中时返回内容应完全一致"
    finally:
        reg.loader._parse_skill_metadata = orig
        import shutil as _shutil
        _shutil.rmtree(tmp_root, ignore_errors=True)
    print("✅ test_c17_1_index_cache_skips_rescan_when_unchanged 通过")


def test_c17_2_get_enabled_metadata_no_nplus1_scan():
    """【C17 性能优化】get_enabled_metadata不应为每个技能单独触发一次扫描
    验证：调用一次get_enabled_metadata期间，真正的全量重扫次数=0（缓存命中）或1（首次）。
    """
    from fast_agents.skills import SkillRegistry
    reg = SkillRegistry()

    tmp_root = Path(tempfile.mkdtemp())
    expected_names = [f"sk-{i}" for i in range(6)]
    for sname in expected_names:
        d = tmp_root / sname
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {sname}\ndescription: desc\nversion: 1.0.0\ntags: [group]\n---\n# {sname}\n",
            encoding="utf-8",
        )
    reg.add_skill_path(tmp_root)

    # 先跑一次确保缓存建立
    reg.get_index_prompt()

    # 打桩统计get_all_metadata真正触发重扫的次数
    real_rescan = [0]
    orig_scan = reg.loader.scan_skills

    def counting_scan(force=False):
        # 只在真正重建时计数：通过观察前后_index变化判断
        before = set(reg.loader._skill_index.keys())
        ret = orig_scan(force=force)
        after = set(reg.loader._skill_index.keys())
        if before != after:
            real_rescan[0] += 1
        return ret

    reg.loader.scan_skills = counting_scan
    try:
        real_rescan[0] = 0
        enabled = reg.get_enabled_metadata()
        # 缓存命中应0次真实重扫
        assert real_rescan[0] == 0, f"get_enabled_metadata缓存命中时不应重扫，实际{real_rescan[0]}次"
        # 应包含我们在tmp_root创建的全部技能（不要求只有它们，cwd下.skills可能有别的）
        for name in expected_names:
            assert name in enabled, f"应返回技能 {name}"

        # 调用list_skills也不应重复扫描
        real_rescan[0] = 0
        lst = reg.list_skills()
        assert real_rescan[0] == 0, f"list_skills缓存命中时不应重扫，实际{real_rescan[0]}次"
        listed_names = {d["name"] for d in lst}
        for name in expected_names:
            assert name in listed_names, f"list_skills应包含 {name}"
    finally:
        reg.loader.scan_skills = orig_scan
        import shutil as _shutil
        _shutil.rmtree(tmp_root, ignore_errors=True)
    print("✅ test_c17_2_get_enabled_metadata_no_nplus1_scan 通过")


def test_c17_3_invalidate_cache_forces_rescan():
    """【C17 性能优化】调用invalidate_index_cache后下一次应强制重扫"""
    from fast_agents.skills import SkillRegistry
    reg = SkillRegistry()

    tmp_root = Path(tempfile.mkdtemp())
    (tmp_root / "only-one").mkdir()
    (tmp_root / "only-one" / "SKILL.md").write_text(
        "---\nname: only-one\ndescription: only\nversion: 1.0.0\n---\n# x\n",
        encoding="utf-8",
    )
    reg.add_skill_path(tmp_root)

    # 第一次建立缓存
    reg.get_index_prompt()
    assert "only-one" in reg.get_index_prompt()

    # 新增一个技能目录（绕过add_skill_path直接在磁盘创建）
    new_dir = tmp_root / "new-one"
    new_dir.mkdir()
    (new_dir / "SKILL.md").write_text(
        "---\nname: new-one\ndescription: new\nversion: 1.0.0\n---\n# x\n",
        encoding="utf-8",
    )

    # 没失效缓存前：缓存命中，看不到新技能（因为根目录mtime没变——Windows上新建子目录会更新父目录mtime，
    # 不过为了兼容不同OS行为，这里主动调invalidate）
    reg.loader.invalidate_index_cache()
    p_after = reg.get_index_prompt()
    assert "only-one" in p_after and "new-one" in p_after, "失效缓存后应能看到新技能"

    import shutil as _shutil
    _shutil.rmtree(tmp_root, ignore_errors=True)
    print("✅ test_c17_3_invalidate_cache_forces_rescan 通过")


# ============ 【C15 静态资产】测试 ============

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


def test_c15_1_assets_scanned_into_metadata():
    """【C15·方案A】加载技能时assets目录应被扫描，但仅进入SkillLoadResult（不进SkillMetadata常驻索引）"""
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = _make_asset_skill(tmp)
        meta = reg.get_metadata("asset-demo")
        assert meta is not None, "asset-demo 应被扫描到"
        # 方案A：assets 不进入 SkillMetadata 常驻索引，仅在 Level 2 load_skill 后出现在 SkillLoadResult
        assert not hasattr(meta, "assets"), "方案A：SkillMetadata 不应包含 assets 字段（避免常驻 token 膨胀）"
        # 二进制标记与清单校验在加载后的 SkillLoadResult 上完成
        loaded = reg.load_skill("asset-demo")
        assert loaded.success
        names = [n for n, _sz, _is_bin in loaded.assets]
        # 文本文件
        assert "hello.txt" in names
        assert "templates/tpl.json" in names
        # 二进制
        assert "img.png" in names
        assert "big.txt" in names
        # 下划线开头被忽略
        assert "_internal.txt" not in names
        # 二进制标记
        bin_map = {n: is_bin for n, _sz, is_bin in loaded.assets}
        assert bin_map["img.png"] is True
        assert bin_map["hello.txt"] is False
        assert bin_map["templates/tpl.json"] is False
        # full_prompt 应包含"静态资源"章节
        prompt = loaded.full_prompt
        assert "静态资源" in prompt
        assert "文本资源" in prompt
        assert "二进制资源" in prompt
        assert "hello.txt" in prompt
        assert "img.png" in prompt
        assert "_internal.txt" not in prompt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("✅ test_c15_1_assets_scanned_into_metadata 通过")


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


def _make_conflict_skill_dir(path: Path, name: str, version: str, content_tag: str = ""):
    """创建同名技能目录（用于冲突测试）"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} {content_tag}\nversion: {version}\n---\n# {name}\n{content_tag}\n",
        encoding="utf-8",
    )


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


def main():
    from tests._runner import run_all as runner_run_all
    raise SystemExit(runner_run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑"""
    # 人工添加测针对测试：AI不要删除覆盖
    
    # AI新添加的测试(AI可以删除覆盖)
    failed = run_all()
    raise SystemExit(failed)


if __name__ == "__main__":
    #main()
    main_of_new_add()
