"""
Skill模块单元测试 —— 基类与模型层（BaseSkill / @skill / SkillMetadata / ScriptParam / ScriptSchema）
覆盖：基类元数据校验、@skill 装饰器、SkillMetadata 字段格式校验、脚本参数 Schema 的模型级校验
说明：本文件已按组件拆分，其余用例迁至同目录：
      test_skills_loader.py     —— SkillLoader 扫描 / 解析 / 沙箱路径防护 / 热重载检测 / 索引缓存
      test_skills_middleware.py —— SkillMiddleware 索引注入 / LRU 缓存 / allowed_tools / 热重载线程
      test_skills_registry.py   —— SkillRegistry 注册 / 启用禁用 / 依赖预检 / 资产读取 / 同名冲突
运行：python tests/skills/test_skills.py
全离线运行，无外部依赖，不依赖 pytest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import (
    BaseSkill, skill,
    SkillMetadata,
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


def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑"""
    # 人工添加测针对测试：AI不要删除覆盖

    # AI新添加的测试(AI可以删除覆盖)


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
