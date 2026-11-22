"""
脚本沙箱增强演示（对应最新实现：C6-2 参数Schema + C18方案B 外置与宿主层配置）
离线可运行：展示参数Schema外置解析、类型自动转换、参数校验、命令注入防护、敏感路径拦截、网络禁用

【C18方案B】写法要点：
- 脚本参数Schema：放在 scripts/<脚本名>.schema.json，与同名脚本同目录，加载技能时自动扫描解析；
- 沙箱策略：由 SkillRegistry / SkillLoader / SkillMiddleware 初始化参数统一配置，
  不再写在 SKILL.md 的 frontmatter 里。

运行方式：python examples/skills/script_sandbox_demo.py
"""
import os
import sys
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.skills import SkillRegistry


def main():
    print("=" * 60)
    print("脚本沙箱增强演示（C6-2 + C18方案B）")
    print("=" * 60)

    # 创建临时技能目录作为扫描路径（把演示skill复制过去）
    demo_skill_src = Path(__file__).parent / "script_sandbox_skill"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_skills = Path(tmpdir) / ".skills"
        shutil.copytree(demo_skill_src, tmp_skills / "script-sandbox-demo")

        # 切换工作目录到临时目录，使.skills被自动扫描
        old_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            # 【C18方案B】沙箱策略在宿主层统一配置，不再由技能自己声明
            registry = SkillRegistry(
                default_allow_network=False,    # 默认断网
                default_allow_write_paths=[],   # 额外可写路径（技能自身 tmp/ 会自动加入白名单）
                default_env_whitelist=[],       # 允许透传给脚本的环境变量白名单
            )

            # ---------- 场景0：一级元数据极简 + 脚本Schema外置 ----------
            print("\n--- 场景0：一级元数据极简 & 脚本Schema外置 ---")
            index = registry.loader.scan_skills()
            meta = index.get("script-sandbox-demo")
            assert meta is not None, "技能应被扫描到"
            field_names = {f.name for f in fields(meta)}
            print(f"技能: {meta.name} v{meta.version}")
            print(f"元数据字段: {', '.join(sorted(field_names))}")
            assert "allow_network" not in field_names, "沙箱字段不应再出现在元数据里"
            assert "allow_write_paths" not in field_names, "沙箱字段不应再出现在元数据里"
            assert "scripts_schema" not in field_names, "脚本Schema不应再出现在元数据里"
            print("[OK] 一级元数据已回归极简（无沙箱配置、无脚本Schema）")

            result = registry.loader.load_skill("script-sandbox-demo")
            schemas = result.scripts_schemas
            print("\n从 scripts/ 目录自动扫描到的参数Schema：")
            for sname in sorted({s.script_name for s in schemas.values()}):
                s = schemas[sname]
                desc = ", ".join(
                    f"{p.name}{'*' if p.required else ''}({p.type})" for p in s.params
                )
                print(f"  - {sname}: {desc}  （* 为必填）")
            assert "greet.py" in schemas, "greet.schema.json 应被扫描到"
            assert "export_report.py" in schemas, "export_report.schema.json 应被扫描到"
            print("[OK] scripts/<脚本名>.schema.json 自动发现并解析成功")

            skill_dir = Path(result.metadata.skill_path)
            print("\n宿主层沙箱配置：")
            print(f"  default_allow_network = {registry.loader.default_allow_network}（默认断网）")
            print(f"  可写白名单 = {registry.loader._get_allow_write_paths(skill_dir)}")

            # ---------- 场景1：正常参数调用（类型自动转换） ----------
            print("\n--- 场景1：合法参数，类型自动转换 ---")
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "张三", "age": "28", "city": "北京"}  # age传字符串，自动转int
            )
            print(result)
            assert "你好 张三" in result, "正常调用应输出问候"
            assert "28" in result, "age应被自动转换为28"
            assert "网络禁用标记=1" in result, "默认应禁用网络"
            print("[OK] 类型自动转换成功，沙箱环境变量已注入")

            # ---------- 场景2：缺失必填参数 ----------
            print("\n--- 场景2：缺失必填参数 → 校验失败 ---")
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "李四"}  # 缺age
            )
            print(result)
            assert "参数校验失败" in result and "age" in result, "应提示缺失age"
            print("[OK] 必填参数校验生效")

            # ---------- 场景3：类型错误 ----------
            print("\n--- 场景3：参数类型错误 → 校验失败 ---")
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "王五", "age": "abc"}  # age传非数字
            )
            print(result)
            assert "类型错误" in result, "应提示类型错误"
            print("[OK] 类型校验生效")

            # ---------- 场景4：数值越界 ----------
            print("\n--- 场景4：数值越界 → 校验失败 ---")
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "赵六", "age": 999}  # age超过max=150
            )
            print(result)
            assert "最大值" in result, "应提示越界"
            print("[OK] 数值范围校验生效")

            # ---------- 场景5：命令注入字符拦截 ----------
            print("\n--- 场景5：参数含命令注入字符 → 拦截 ---")
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "test; rm -rf /", "age": 20}
            )
            print(result)
            assert "非法字符" in result, "应拦截注入字符"
            print("[OK] 命令注入防护生效")

            # ---------- 场景6：枚举类型校验 ----------
            print("\n--- 场景6：枚举值非法 → 校验失败 ---")
            result = registry.run_script(
                "script-sandbox-demo", "export_report.py",
                {"title": "测试报告", "format": "pdf"}  # pdf不在枚举里
            )
            print(result)
            assert "必须是以下值之一" in result, "应提示枚举值错误"
            print("[OK] 枚举校验生效")

            # ---------- 场景7：枚举合法值正常通过 ----------
            print("\n--- 场景7：合法枚举值 → 执行成功 ---")
            result = registry.run_script(
                "script-sandbox-demo", "export_report.py",
                {"title": "测试报告", "format": "json"}
            )
            print(result)
            assert "报告已导出" in result, "应成功导出"
            print("[OK] 枚举合法值执行成功")

            # ---------- 场景8：敏感路径拦截 ----------
            print("\n--- 场景8：参数指向系统敏感路径 → 拦截 ---")
            # 注意：用正斜杠路径，反斜杠会先被注入字符检查拦截
            result = registry.run_script(
                "script-sandbox-demo", "greet.py",
                {"name": "C:/Windows/System32/cmd", "age": 20}
            )
            print(result)
            assert "敏感路径" in result or "沙箱拦截" in result, "应拦截敏感路径"
            print("[OK] 敏感路径访问拦截生效")

            print("\n" + "=" * 60)
            print("全部沙箱能力验证通过！")
            print("=" * 60)

        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    main()
