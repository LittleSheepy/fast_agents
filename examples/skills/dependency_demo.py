"""
C12 技能依赖检查演示
展示：缺失工具/缺失技能/版本冲突/循环依赖/依赖预检等场景
全离线可运行
"""
import os
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import SkillRegistry, SkillMiddleware


def make_skill(base: Path, name: str, meta: dict, instructions: str = "执行指令示例"):
    """在临时目录下创建一个文件型技能"""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n" + yaml.dump(meta, allow_unicode=True) + "---\n" + instructions,
        encoding="utf-8"
    )


class FakeToolRegistry:
    """模拟Agent的ToolRegistry"""
    TOOLS = [
        "load_skill", "read_skill_reference", "run_skill_script", "list_loaded_skills",
        "disable_skill", "enable_skill", "disable_skills_by_tag", "enable_skills_by_tag",
        "check_skill_dependencies", "web_search", "calculator",
    ]
    def list_names(self): return list(self.TOOLS)
    def get_tool(self, name): return None


def main():
    print("=" * 60)
    print("  C12 技能依赖检查演示")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        
        # ---- 场景1：正常技能（无依赖） ----
        make_skill(tmpdir, "hello", {
            "name": "hello",
            "description": "一个无依赖的简单技能",
            "version": "1.0.0",
        })
        
        # ---- 场景2：依赖缺失工具 ----
        make_skill(tmpdir, "need-tool", {
            "name": "need-tool",
            "description": "依赖不存在工具的技能",
            "version": "1.0.0",
            "allowed_tools": ["nonexistent_tool_xyz"],
        })
        
        # ---- 场景3：依赖另一个技能 ----
        make_skill(tmpdir, "base-util", {
            "name": "base-util",
            "description": "基础工具技能",
            "version": "1.5.0",
        })
        make_skill(tmpdir, "top-app", {
            "name": "top-app",
            "description": "上层应用技能，依赖base-util>=1.0.0和web_search工具",
            "version": "1.0.0",
            "requires": ["base-util>=1.0.0"],
            "requires_tools": ["web_search"],
        })
        
        # ---- 场景4：版本冲突 ----
        make_skill(tmpdir, "old-base", {
            "name": "old-base",
            "description": "老版本基础",
            "version": "0.5.0",
        })
        make_skill(tmpdir, "needs-new", {
            "name": "needs-new",
            "description": "需要新版本old-base",
            "version": "1.0.0",
            "requires": ["old-base>=2.0.0"],
        })
        
        # ---- 场景5：循环依赖 ----
        make_skill(tmpdir, "alpha", {
            "name": "alpha",
            "description": "alpha",
            "version": "1.0.0",
            "requires": ["beta"],
        })
        make_skill(tmpdir, "beta", {
            "name": "beta",
            "description": "beta",
            "version": "1.0.0",
            "requires": ["alpha"],
        })
        
        # 演示场景只扫描临时目录内的技能，关闭默认路径避免无关冲突告警
        reg = SkillRegistry(extra_skill_paths=[tmpdir], include_default_paths=False)
        mw = SkillMiddleware(registry=reg)
        mw.bind_tool_registry(FakeToolRegistry())
        
        # 1. 无依赖的技能直接加载
        print("\n[1] 无依赖技能直接加载:")
        r = reg.load_skill("hello")
        print(f"    success={r.success}")
        
        # 2. 缺失工具
        print("\n[2] 缺失工具:")
        r = reg.load_skill("need-tool")
        print(f"    success={r.success}")
        for e in r.errors:
            print(f"    · {e[:90]}")
        
        # 3. 依赖预检（不加载）
        print("\n[3] 依赖预检 top-app（未加载base-util前）:")
        info = reg.check_skill_dependencies("top-app")
        print(f"    ok={info['ok']}")
        for e in info['errors']:
            print(f"    · {e[:90]}")
        
        # 4. 先加载依赖，再加载top-app
        print("\n[4] 先加载 base-util，再加载 top-app:")
        reg.load_skill("base-util")
        r = reg.load_skill("top-app")
        print(f"    success={r.success}")
        
        # 5. 版本冲突
        print("\n[5] 版本冲突 (needs-new要求old-base>=2.0.0但实际是0.5.0):")
        r = reg.load_skill("needs-new")
        print(f"    success={r.success}")
        for e in r.errors[:2]:
            print(f"    · {e[:100]}")
        
        # 6. 循环依赖
        print("\n[6] 循环依赖检测 (alpha<->beta):")
        r = reg.load_skill("alpha")
        print(f"    success={r.success}")
        for e in r.errors[:2]:
            print(f"    · {e[:100]}")
    
    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
