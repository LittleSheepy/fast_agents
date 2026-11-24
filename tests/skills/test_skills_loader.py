"""
Skill模块单元测试 —— 加载器（SkillLoader）
覆盖：技能目录扫描、SKILL.md frontmatter 解析、脚本参数 schema.json 解析与容错（C6-2 / C18）、
       脚本 / 参考文档 / 资产路径穿越防护与扩展名校验（B 组、C15）、
       热重载的变更检测（C14：mtime 快照、detect_changes、reload_skill、rescan_incremental）、
       索引缓存与 1+N 扫描治理（C17）
说明：由 test_skills.py 按组件拆分而来，用例体与断言逐字节保持不变；
      _make_* 辅助函数因跨组件复用，本文件保留一份副本以便独立运行。
      部分用例通过 SkillRegistry 门面调用，但断言的是 SkillLoader 的扫描 / 解析 / 沙箱行为。
全离线运行，无外部依赖，不依赖 pytest。
运行：python tests/skills/test_skills_loader.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import (
    SkillRegistry,
    SkillLoader,
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


# ========== C12 技能依赖检查：loader 解析 requires 字段 ==========

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


# ========== C6-2 脚本沙箱增强：loader 解析 scripts/*.schema.json ==========

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


# ========== C18 方案B：schema 外置与容错 ==========

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


# ========== C14 技能热重载：变更检测 ==========

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


# ========== C17 性能优化：索引缓存与 1+N 扫描治理 ==========

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


# ========== C15 静态资产：assets 扫描 ==========

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
