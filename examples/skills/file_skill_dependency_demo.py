"""
示例：文件型 Skill 依赖工具演示（以 .skills/web-research 为例）
演示流程：
  1. 未注册依赖工具 → 加载失败（提示缺失 web_search / read_file）
  2. 注册两个 mock 工具（web_search / read_file）后 → 加载成功
  3. 按需读取参考文档（source-check.md / report-template.md）
  4. 调用 check_skill_dependencies 工具查看依赖状态

说明：本示例使用轻量 MockToolRegistry，不依赖完整 Agent 启动；
      真实使用场景下，SkillMiddleware.bind_tool_registry(tool_registry) 会自动绑定。
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.skills import SkillRegistry, SkillMiddleware


# ---------------------------------------------------------------------------
# 轻量 Mock：只实现 SkillMiddleware 依赖检查会用到的 list_names()
# ---------------------------------------------------------------------------
class MockToolRegistry:
    """模拟 Agent 的工具注册表，仅提供工具名查询，用于演示依赖检查。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """注册一个工具（按名字），用于演示依赖补齐。"""
        self._tools[name] = fn

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def list_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def get_tool(self, name: str) -> Optional[Any]:
        """bind_tool_registry 会调用本方法查找 list_loaded_skills；
        演示场景不需要替换内置工具的实现，统一返回 None 即可。"""
        return None

    def invoke(self, name: str, **kwargs: Any) -> str:
        if name not in self._tools:
            return f"[mock] 工具 {name} 未注册"
        return self._tools[name](**kwargs)


# ---------------------------------------------------------------------------
# 两个 mock 工具：与 web-research 声明的 allowed-tools 一一对应
# ---------------------------------------------------------------------------
def mock_web_search(query: str, **_: Any) -> str:
    """模拟联网搜索，返回固定格式的结果列表。"""
    return (
        f"[mock web_search] 查询：{query}\n"
        "  1) 示例来源A（官方）：关于该主题的公开数据\n"
        "  2) 示例来源B（主流媒体）：相关报道\n"
        "  3) 示例来源C（企业公告）：一手信息"
    )


def mock_read_file(file_path: str, **_: Any) -> str:
    """模拟文件读取，返回占位内容；真实环境下由框架的 read_file 工具实现。"""
    p = Path(file_path)
    if p.exists() and p.is_file():
        return f"[mock read_file] {p.name}（{p.stat().st_size} 字节）\n" + p.read_text(encoding="utf-8")[:500]
    return f"[mock read_file] 文件不存在：{file_path}"


def main() -> None:
    print("=== 文件型 Skill 依赖工具演示（web-research）===  ")
    print()

    # ---- 准备注册表 + 中间件 + 工具注册表 ----
    # 注意：这里 include_default_paths=True，会自动扫描项目根目录 .skills/
    registry = SkillRegistry(include_default_paths=True)
    middleware = SkillMiddleware(registry=registry, auto_inject_index=False)

    tool_reg = MockToolRegistry()
    middleware.bind_tool_registry(tool_reg)  # 绑定后，依赖检查才能看到已注册的工具

    # ---- Level 1：索引扫描 ----
    print("【Level 1】索引扫描")
    all_meta = registry.get_all_metadata()
    wr = all_meta.get("web-research")
    assert wr is not None, "项目根目录 .skills/ 下应存在 web-research 技能"
    print(f"  发现技能: {wr.name} v{wr.version}")
    print(f"  声明依赖工具 (allowed-tools): {wr.allowed_tools}")
    print(f"  当前已注册工具: {tool_reg.list_names()}")
    print()

    # ---- 场景 A：未注册任何依赖工具 → 加载失败 ----
    print("【场景 A】未注册 web_search / read_file → 加载失败（预期）")
    result_a = registry.load_skill("web-research")
    if result_a.success:
        print("  ❌ 预期应失败，却加载成功了，请检查依赖检查逻辑")
    else:
        print("  ✅ 加载失败，错误信息：")
        for err in result_a.errors:
            print(f"     - {err}")
    print()

    # ---- 用 check_skill_dependencies 预检（LLM 工具视角）----
    print("【预检】调用 check_skill_dependencies 查看详细状态")
    dep_report = registry.check_skill_dependencies("web-research")
    print(f"  ok: {dep_report.get('ok')}")
    for err in dep_report.get("errors", []):
        print(f"  - {err}")
    print()

    # ---- 场景 B：注册 mock 工具后 → 加载成功 ----
    print("【场景 B】注册 web_search / read_file 两个 mock 工具 → 加载成功")
    tool_reg.register("web_search", mock_web_search)
    tool_reg.register("read_file", mock_read_file)
    print(f"  当前已注册工具: {tool_reg.list_names()}")

    result_b = registry.load_skill("web-research")
    assert result_b.success, f"应加载成功，实际错误: {result_b.errors}"
    print("  ✅ 加载成功")
    print(f"     可用脚本: {[p.name for p in result_b.scripts]}")
    print(f"     可用参考文档: {[p.name for p in result_b.references]}")
    print(f"     可用静态资产: {[a[0] for a in result_b.assets]}")
    print()

    # ---- Level 3：按需读取参考文档（不要一次性全读）----
    print("【Level 3】按需读取参考文档")
    print("  --- source-check.md（前 300 字符）---")
    src = registry.read_reference("web-research", "source-check.md")
    print("  " + src[:300].replace("\n", "\n  "))
    if len(src) > 300:
        print("  ...（截断，完整内容按需读取）")
    print()

    # ---- 演示 mock 工具实际可被调用 ----
    print("【演示】mock 工具可被 LLM 直接调用")
    print(tool_reg.invoke("web_search", query="2026年中国新能源汽车销量"))
    print()

    # ---- 场景 C：注销一个工具后，check_skill_dependencies 能重新发现缺失 ----
    print("【场景 C】注销 web_search → 依赖预检再次报告缺失")
    tool_reg.unregister("web_search")
    dep2 = registry.check_skill_dependencies("web-research")
    print(f"  ok: {dep2.get('ok')}")
    for err in dep2.get("errors", []):
        print(f"  - {err}")
    print("  ✅ 依赖检查是实时的，注销后立刻反映")
    print()

    print("=== 演示完成 ===")
    print("提示：")
    print("  - 真实 Agent 场景下，web_search / read_file 由框架内置或你自己注册到 ToolRegistry；")
    print("  - SkillMiddleware.bind_tool_registry(tool_registry) 在 ReActAgent 初始化时自动完成；")
    print("  - 技能通过 allowed-tools 声明所需工具，宿主按需补齐，避免无权限工具被误调用。")


if __name__ == "__main__":
    main()
