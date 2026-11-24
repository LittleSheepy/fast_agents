"""Tools 模块单元测试 —— ToolRegistry 注册器

从 test_tools.py 拆分而来，聚焦 ToolRegistry 自身的注册 / 清理 / 隔离行为：
    批量注册 + hook + tag 过滤 + JSON 参数（第 6 节），以及 A22 清理与隔离。

运行：python tests/tools/test_tool_registry.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools import ToolRegistry, tool, builtin


# ─────────────────────────────────────────────────────────
# 6. 注册表批量操作 / hook / tag 过滤
# ─────────────────────────────────────────────────────────

def test_registry_batch_and_hook():
    reg = ToolRegistry()
    reg.register_tools(
        builtin.CalculatorTool(),
        builtin.ReadTool(workspace="/tmp"),
    )
    assert len(reg) == 2
    assert reg.has_tool("python_calculator")
    assert "read_file" in reg.list_names()

    # tag 过滤
    math_tools = reg.filter_by_tag("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "python_calculator"

    # hook
    calls = []
    reg.add_hook(lambda name, args, res: calls.append((name, res.status.value)))
    reg.invoke_tool("python_calculator", {"expression": "1+1"})
    assert len(calls) == 1
    assert calls[0][0] == "python_calculator"
    assert calls[0][1] == "success"

    # JSON 字符串参数（模拟 LLM 传入）
    r = reg.invoke_tool("python_calculator", '{"expression": "2*3"}')
    assert r.output.endswith("6") or "= 6" in r.output

    # 找不到工具
    r = reg.invoke_tool("nonexistent", {})
    assert r.failed
    print("✅ Registry 批量/hook/tag/JSON参数 通过")


# ── A22 ToolRegistry 清理与隔离 ──────────────────────────

def test_register_strict_mode_raises_on_duplicate():
    """strict=True 时同名工具重复注册应抛出 ValueError。"""
    reg = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    reg.register_tool(add)
    try:
        reg.register_tool(add, strict=True)
        assert False, "strict 模式下重复注册应抛出异常"
    except ValueError as e:
        assert "已存在" in str(e)
    print("✅ strict 模式重复注册抛出 ValueError 通过")


def test_register_non_strict_mode_warns_and_overwrites():
    """默认非严格模式保持原有行为：打印 warning 后静默覆盖。"""
    reg = ToolRegistry()

    @tool
    def greet() -> str:
        return "v1"

    @tool
    def greet_v2() -> str:
        return "v2"
    greet_v2.name = "greet"  # 同名不同实现

    reg.register_tool(greet)
    reg.register_tool(greet_v2)  # 默认 strict=False
    r = reg.invoke_tool("greet")
    assert r.succeeded
    assert r.output == "v2"
    print("✅ 非严格模式覆盖行为保持向后兼容通过")


def test_clear_resets_approval_callback():
    """clear() 必须同时重置 _tools、_hooks 和 _approval_callback。"""
    reg = ToolRegistry()

    @tool(metadata={"requires_approval": True})
    def dangerous_op() -> str:
        return "executed"

    reg.register_tool(dangerous_op)
    called = {"count": 0}

    def always_reject(name, args):
        called["count"] += 1
        return False, None, "拒绝"

    reg.set_approval_callback(always_reject)

    # 注册回调后执行应该被拦截
    r1 = reg.invoke_tool("dangerous_op")
    assert not r1.succeeded
    assert called["count"] == 1

    # 清空后重新注册同名工具，不应该再触发旧的审批回调
    reg.clear()
    reg.register_tool(dangerous_op)
    r2 = reg.invoke_tool("dangerous_op")
    assert r2.succeeded, f"clear 后审批回调未重置，仍被拦截：{r2.error}"
    assert called["count"] == 1, "clear 后旧审批回调不应再被调用"
    print("✅ clear() 重置审批回调修复通过")


def test_no_thread_local_dead_code():
    """确认 ToolRegistry 实例不再持有 _thread_local 死代码属性。"""
    reg = ToolRegistry()
    assert not hasattr(reg, "_thread_local"), "_thread_local 死代码应已删除"
    print("✅ 移除 _thread_local 死代码通过")


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    test_registry_batch_and_hook()  # Registry
    test_register_strict_mode_raises_on_duplicate()  # A22
    test_register_non_strict_mode_warns_and_overwrites()  # A22
    test_clear_resets_approval_callback()  # A22
    test_no_thread_local_dead_code()  # A22


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
