"""技能（Skill）模块单元测试

覆盖：
1. frontmatter 极简解析器：引号 / 行内列表 / 块列表 / 一层嵌套 map / 块标量 / 数字布尔；
2. 技能名合法性校验；
3. 技能扫描与解析：正常技能 + 缺 description 的坏技能（只记错误，不阻断其它技能）；
4. 路径沙箱：绝对路径、../ 越权、未知技能；
5. 三个技能工具：list_skills / read_skill / run_skill_script（含脚本白名单）；
6. 三级渐进式加载：catalog 只有摘要；read_skill 才拿到正文；resource 才拿到参考资料。
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.skill import (
    SkillLoader,
    SkillRegistry,
    is_valid_skill_name,
    split_frontmatter,
)
from fast_agents.skill.errors import SkillNotFoundError, SkillPermissionError

_HELLO_SCRIPT = (
    "# -*- coding: utf-8 -*-\n"
    "import sys\n"
    "print('技能脚本运行成功，参数：', sys.argv[1:])\n"
)


# ---------------- 辅助 ----------------

def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_skill(
    root: Path,
    name: str,
    *,
    description: str = "演示技能：把任务拆成三步",
    body: str = "# 流程\n\n1. 第一步\n2. 第二步\n",
    references=None,
    scripts=None,
) -> Path:
    """在 root 下造一个技能包，返回技能目录"""
    skill_dir = root / name
    frontmatter = f"---\nname: {name}\ndescription: {description}\n---\n"
    _write(skill_dir / "SKILL.md", frontmatter + body)
    for relative_path, content in (references or {}).items():
        _write(skill_dir / relative_path, content)
    for relative_path, content in (scripts or {}).items():
        _write(skill_dir / relative_path, content)
    return skill_dir


# ---------------- 1. 解析器 ----------------

def test_frontmatter_parser():
    """frontmatter 的六种写法都要能解析，且没有 frontmatter 时安全降级"""
    text = (
        "---\n"
        'name: "web-research"\n'
        "version: 1.0.0\n"
        "enabled: true\n"
        "count: 3\n"
        "allowed-tools: [read_file, terminal]\n"
        "tags:\n"
        "  - research\n"
        "  - report\n"
        "metadata:\n"
        "  author: sheepy\n"
        "  category: research\n"
        "note: |\n"
        "  第一行\n"
        "  第二行\n"
        "---\n"
        "# 正文\n内容\n"
    )
    frontmatter, body, has_frontmatter = split_frontmatter(text)

    assert has_frontmatter is True
    assert frontmatter["name"] == "web-research", frontmatter
    assert frontmatter["version"] == "1.0.0"
    assert frontmatter["enabled"] is True
    assert frontmatter["count"] == 3
    assert frontmatter["allowed-tools"] == ["read_file", "terminal"]
    assert frontmatter["tags"] == ["research", "report"]
    assert frontmatter["metadata"] == {"author": "sheepy", "category": "research"}
    assert frontmatter["note"] == "第一行\n第二行"
    assert body.startswith("# 正文")

    # 没有 frontmatter：不报错，整体当正文
    frontmatter2, body2, has2 = split_frontmatter("# 只有正文")
    assert has2 is False and frontmatter2 == {} and body2 == "# 只有正文"
    print("✅ test_frontmatter_parser passed")


def test_skill_name_validation():
    assert is_valid_skill_name("web-research")
    assert is_valid_skill_name("中文技能名")
    assert not is_valid_skill_name("")
    assert not is_valid_skill_name("a/b")
    assert not is_valid_skill_name("a\\b")
    assert not is_valid_skill_name(".hidden")
    assert not is_valid_skill_name("x" * 65)
    print("✅ test_skill_name_validation passed")


# ---------------- 2. 发现与解析 ----------------

def test_discover_and_load():
    """一个坏技能包只记录错误，不影响其它技能加载"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        _make_skill(
            root,
            "demo-flow",
            references={"references/tips.md": "# 提示\n注意先备份\n"},
            scripts={"scripts/hello.py": _HELLO_SCRIPT},
        )
        _make_skill(root, "bad-skill", description="")  # 缺 description

        loader = SkillLoader([root])
        skills, errors = loader.discover()

        assert [s.name for s in skills] == ["demo-flow"], [s.name for s in skills]
        assert len(errors) == 1, errors
        assert "description" in errors[0].message

        skill = skills[0]
        assert skill.description.startswith("演示技能")
        assert skill.instructions.startswith("# 流程")
        assert skill.source == "extra"  # 临时目录既不是 ./skills 也不是 ~/.fast_agents/skills
        assert skill.resources_of("references") == ["references/tips.md"]
        assert skill.resources_of("scripts") == ["scripts/hello.py"]
        assert "参考资料" in skill.resource_summary()
    print("✅ test_discover_and_load passed")


# ---------------- 3. 沙箱 ----------------

def test_resource_sandbox():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        _make_skill(root, "demo-flow", references={"references/tips.md": "参考资料正文"})
        registry = SkillRegistry([root])

        assert "参考资料正文" in registry.read_resource("demo-flow", "references/tips.md")
        # 技能名大小写不敏感
        assert registry.get("DEMO-FLOW").name == "demo-flow"

        for bad in ("../secret.md", "../../secret.md", "C:/Windows/win.ini", "/etc/passwd"):
            try:
                registry.read_resource("demo-flow", bad)
            except SkillPermissionError:
                continue
            raise AssertionError(f"越权路径未被拦截：{bad}")

        try:
            registry.get("不存在的技能")
        except SkillNotFoundError as exc:
            assert "可用技能" in str(exc)
        else:
            raise AssertionError("未知技能应抛 SkillNotFoundError")
    print("✅ test_resource_sandbox passed")


# ---------------- 4. 工具 ----------------

def test_skill_tools():
    """三个工具的三级渐进式加载行为 + 错误都要转成 ToolResult.fail"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        _make_skill(
            root,
            "demo-flow",
            references={"references/tips.md": "参考资料正文"},
            scripts={"scripts/hello.py": _HELLO_SCRIPT, "scripts/notes.txt": "这不是脚本"},
        )
        registry = SkillRegistry([root])
        tools = {tool.name: tool for tool in registry.get_tools(enable_script=True)}
        assert set(tools) == {"list_skills", "read_skill", "run_skill_script"}

        # ① 发现层：只有名称 + 用途，不含正文
        catalog = tools["list_skills"].invoke({"keyword": ""}).output
        assert "demo-flow" in catalog and "演示技能" in catalog
        assert "1. 第一步" not in catalog, "catalog 不能包含技能正文"

        # ② 激活层：read_skill 才拿到正文
        activated = tools["read_skill"].invoke({"name": "demo-flow"})
        assert activated.succeeded
        assert "1. 第一步" in activated.output
        assert activated.artifact["skill"] == "demo-flow"
        assert registry.active_skills == ["demo-flow"]

        # ③ 执行层：读参考资料
        resource = tools["read_skill"].invoke({"name": "demo-flow", "resource": "references/tips.md"})
        assert resource.succeeded and "参考资料正文" in resource.output

        # ③ 执行层：跑脚本
        script_result = tools["run_skill_script"].invoke(
            {"name": "demo-flow", "script": "scripts/hello.py"}
        )
        assert script_result.succeeded, script_result.output
        assert "技能脚本运行成功" in script_result.output

        # 错误路径：全部变成 ToolResult.fail，不抛异常
        assert tools["read_skill"].invoke({"name": "不存在"}).failed
        assert tools["run_skill_script"].invoke({"name": "demo-flow", "script": "../escape.py"}).failed

        unsupported = tools["run_skill_script"].invoke(
            {"name": "demo-flow", "script": "scripts/notes.txt"}
        )
        assert unsupported.failed and "不支持的脚本类型" in unsupported.output
    print("✅ test_skill_tools passed")


def test_script_cwd_and_env():
    """脚本工作目录默认跟随调用方，技能目录通过 SKILL_DIR 暴露；script_cwd='skill' 时固定在技能目录"""
    where = (
        "# -*- coding: utf-8 -*-\n"
        "import os\n"
        "print('CWD=' + os.getcwd())\n"
        "print('SKILL_DIR=' + os.environ.get('SKILL_DIR', ''))\n"
        "print('SKILL_NAME=' + os.environ.get('SKILL_NAME', ''))\n"
        "print('UTF8=' + os.environ.get('PYTHONUTF8', ''))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        skill_dir = _make_skill(root, "cwd-demo", scripts={"scripts/where.py": where})

        def pick(stdout: str, key: str) -> Path:
            line = [l for l in stdout.splitlines() if l.startswith(key + "=")][0]
            return Path(line.split("=", 1)[1])

        # 默认 caller 模式：模型传来的相对路径按 Agent 当前目录解析
        registry = SkillRegistry([root])
        info = registry.run_script("cwd-demo", "scripts/where.py")
        assert info["returncode"] == 0, info["stderr"]
        assert pick(info["stdout"], "CWD").resolve() == Path(os.getcwd()).resolve(), info["stdout"]
        assert pick(info["stdout"], "SKILL_DIR").resolve() == skill_dir.resolve(), info["stdout"]
        assert "SKILL_NAME=cwd-demo" in info["stdout"]
        assert "UTF8=1" in info["stdout"]

        # skill 模式：脚本固定在技能目录内，可直接用 assets/ 这类相对路径
        skill_mode = SkillRegistry(loader=SkillLoader([root], script_cwd="skill"))
        info2 = skill_mode.run_script("cwd-demo", "scripts/where.py")
        assert info2["returncode"] == 0, info2["stderr"]
        assert pick(info2["stdout"], "CWD").resolve() == skill_dir.resolve(), info2["stdout"]

        # 非法取值必须早失败，而不是运行期才炸
        try:
            SkillLoader([root], script_cwd="somewhere")
        except ValueError:
            pass
        else:
            raise AssertionError("script_cwd 非法取值应抛 ValueError")
    print("✅ test_script_cwd_and_env passed")


def test_empty_registry_catalog():
    """没有任何技能时，catalog 必须是可读的占位文案（提示词里不能留悬空标题）"""
    with tempfile.TemporaryDirectory() as tmp:
        registry = SkillRegistry([Path(tmp) / "nothing"])
        assert registry.get_catalog() == "（暂无可用技能）"
        assert registry.get_tools()[0].invoke({"keyword": ""}).output == "（暂无可用技能）"
    print("✅ test_empty_registry_catalog passed")


if __name__ == "__main__":
    test_frontmatter_parser()
    test_skill_name_validation()
    test_discover_and_load()
    test_resource_sandbox()
    test_skill_tools()
    test_script_cwd_and_env()
    test_empty_registry_catalog()
    print("\n🎉 全部测试通过")
