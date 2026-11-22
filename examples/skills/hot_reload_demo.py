"""
【C14】技能热重载功能演示
演示如何启用技能热重载、自动检测文件变更、手动触发重载
"""
import sys
import time
import tempfile
import shutil
from pathlib import Path

# 自举路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.skills.registry import SkillRegistry
from fast_agents.skills.middleware import SkillMiddleware


def print_sep(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print_sep("【C14】技能热重载功能演示")
    
    # 使用临时目录模拟.skills目录
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        skills_dir = tmpdir / ".skills"
        skills_dir.mkdir()
        
        # 复制示例技能到临时目录
        demo_src = Path(__file__).parent / "hot_reload_demo_skill"
        demo_dst = skills_dir / "demo-skill"
        shutil.copytree(demo_src, demo_dst)
        
        print(f"\n[1] 初始化技能注册表，扫描技能目录: {skills_dir}")
        # 演示场景只扫描临时目录，关闭默认路径避免无关冲突告警
        reg = SkillRegistry(extra_skill_paths=[skills_dir], include_default_paths=False)
        mw = SkillMiddleware(registry=reg, auto_inject_index=False)
        
        # 初始扫描
        skills = reg.list_skills()
        print(f"    发现技能: {[s['name'] for s in skills]}")
        meta = reg.get_metadata("demo-skill")
        print(f"    技能版本: {meta.version}")
        print(f"    技能描述: {meta.description}")
        
        print_sep("[2] 演示手动重载 (reload_skills_now)")
        print("    修改SKILL.md版本号从1.0.0改为2.0.0...")
        time.sleep(0.2)
        
        # 初始化mtime快照（模拟热重载运行状态）
        mw._last_mtime_map = reg.loader._build_dir_mtime_map()
        
        # 修改SKILL.md
        skill_md = demo_dst / "SKILL.md"
        old_content = skill_md.read_text(encoding="utf-8")
        new_content = old_content.replace("version: 1.0.0", "version: 2.0.0")
        new_content = new_content.replace("v1.0.0", "v2.0.0")
        skill_md.write_text(new_content, encoding="utf-8")
        
        time.sleep(0.1)
        result = mw.reload_skills_now()
        print(f"    重载结果:\n{result}")
        
        # 验证元数据已更新
        meta = reg.get_metadata("demo-skill")
        print(f"    更新后版本: {meta.version}")
        
        print_sep("[3] 演示启用后台热重载线程")
        msg = mw.enable_hot_reload(interval=1.0)
        print(f"    {msg}")
        print(f"    后台线程运行中: {mw._hot_reload_thread.is_alive()}")
        
        print("\n    修改SKILL.md版本号从2.0.0改为3.0.0...")
        time.sleep(0.2)
        new_content2 = new_content.replace("version: 2.0.0", "version: 3.0.0")
        new_content2 = new_content2.replace("v2.0.0", "v3.0.0")
        skill_md.write_text(new_content2, encoding="utf-8")
        
        print("    等待后台轮询检测变更（最多等待2秒）...")
        time.sleep(2.5)
        
        meta = reg.get_metadata("demo-skill")
        print(f"    自动更新后版本: {meta.version}")
        
        print_sep("[4] 演示新增技能自动发现")
        print("    新增一个技能目录 new-skill ...")
        time.sleep(0.2)
        new_skill_dir = skills_dir / "new-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text(
            "---\nname: new-skill\ndescription: 新增的演示技能\nversion: 1.0.0\n---\n\n这是运行中新增的技能。\n",
            encoding="utf-8"
        )
        
        print("    等待后台检测...")
        time.sleep(2.5)
        
        skills = reg.list_skills()
        skill_names = [s['name'] for s in skills]
        print(f"    检测后技能列表: {skill_names}")
        assert "new-skill" in skill_names, "新增技能应该被自动发现"
        
        print_sep("[5] 演示删除技能自动清理")
        print("    删除 new-skill 目录...")
        time.sleep(0.2)
        shutil.rmtree(new_skill_dir)
        
        print("    等待后台检测...")
        time.sleep(2.5)
        
        skills = reg.list_skills()
        skill_names = [s['name'] for s in skills]
        print(f"    检测后技能列表: {skill_names}")
        assert "new-skill" not in skill_names, "删除的技能应该被自动清理"
        
        print_sep("[6] 关闭热重载")
        msg = mw.disable_hot_reload()
        print(f"    {msg}")
        time.sleep(1.5)
        thread_stopped = mw._hot_reload_thread is None or not mw._hot_reload_thread.is_alive()
        print(f"    后台线程已停止: {thread_stopped}")
        
        print_sep("✅ 演示完成")
        print("""
C14技能热重载功能总结：
  ✅ 默认关闭，零运行时开销
  ✅ 标准库实现，零新增依赖
  ✅ mtime增量检测，性能开销极小
  ✅ 自动识别新增/修改/删除三类变更
  ✅ 修改已加载技能自动卸载，下次load_skill拿到新版本
  ✅ 删除技能自动清理索引、禁用状态、已加载状态
  ✅ 后台守护线程daemon=True，异常兜底，不影响主流程
  ✅ 三个LLM管理工具：启用/关闭/手动重载，Agent可自主控制
        """)


if __name__ == "__main__":
    main()
