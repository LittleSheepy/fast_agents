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
    assert len(tools) == 4  # v1.2.0 C10 新增list_loaded_skills
    tool_names = [t.name for t in tools]
    assert "load_skill" in tool_names
    assert "read_skill_reference" in tool_names
    assert "run_skill_script" in tool_names
    assert "list_loaded_skills" in tool_names
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
    assert "[代码]" in prompt
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
    return failed


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
