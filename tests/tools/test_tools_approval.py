"""Tools 模块单元测试 —— 人工审批（A9）

从 test_tools.py 拆分而来，聚焦 ToolRegistry 的人工审批链路：
    通过 / 拒绝 / 改写参数 / 回调异常 / 无回调放行 / 未标记不触发 / skip_approval 跳过。

运行：python tests/tools/test_tools_approval.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools import ToolBase, ToolRegistry


def _approval_tool(name: str = "approval_tool"):
    """构造一个标记了 requires_approval 的测试工具。"""

    class _ApprovalTool(ToolBase):
        def run(self, expression: str = "1+1"):
            return f"expr={expression}"

    return _ApprovalTool(
        name=name,
        description="需要审批的测试工具",
        metadata={"requires_approval": True},
    )


def test_approval_approved():
    """A9：审批通过后正常执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(lambda name, args: (True, None, ""))
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.succeeded, r.output
    assert r.output == "expr=1+1"
    print("✅ A9 审批通过后正常执行通过")


def test_approval_rejected():
    """A9：审批拒绝时工具不执行，返回失败并带上原因。"""
    executed = []

    class SpyTool(ToolBase):
        name = "spy_tool"
        description = "记录是否执行过的审批工具"
        metadata = {"requires_approval": True}

        def run(self, expression: str = "1+1"):
            executed.append(expression)
            return f"expr={expression}"

    reg = ToolRegistry()
    reg.register_tool(SpyTool())
    reg.set_approval_callback(lambda name, args: (False, None, "不安全"))
    r = reg.invoke_tool("spy_tool", {"expression": "1+1"})
    assert r.failed
    assert "被拒绝" in r.output
    assert "不安全" in r.output
    assert executed == [], "被拒绝的工具不应执行"
    print("✅ A9 审批拒绝：不执行 + 失败带原因 通过")


def test_approval_modified_args():
    """A9：审批回调可改写参数，工具用改写后的参数执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(lambda name, args: (True, {"expression": "2+2"}, ""))
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.succeeded, r.output
    assert r.output == "expr=2+2", f"未使用改写后的参数: {r.output!r}"
    print("✅ A9 审批改写参数生效通过")


def test_approval_callback_exception():
    """A9：审批回调抛异常时返回失败，不抛穿到调用方。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())

    def boom(name, args):
        raise RuntimeError("回调炸了")

    reg.set_approval_callback(boom)
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.failed
    assert "审批流程异常" in r.output
    print("✅ A9 审批回调异常：失败返回不抛穿 通过")


def test_approval_skipped_without_callback():
    """A9：未设置审批回调时，即使工具标记了 requires_approval 也直接执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    r = reg.invoke_tool("approval_tool", {"expression": "3+3"})
    assert r.succeeded, r.output
    assert r.output == "expr=3+3"
    print("✅ A9 无回调时直接执行通过")


def test_approval_not_triggered_for_unmarked_tool():
    """A9：未标记 requires_approval 的工具不触发审批回调。"""
    calls = []

    class PlainTool(ToolBase):
        name = "plain_tool"
        description = "无需审批的测试工具"

        def run(self):
            return "plain ok"

    reg = ToolRegistry()
    reg.register_tool(PlainTool())
    reg.set_approval_callback(
        lambda name, args: (calls.append(name), (False, None, "x"))[1]
    )
    r = reg.invoke_tool("plain_tool", {})
    assert r.succeeded
    assert calls == [], f"未标记的工具不该触发审批，实际调用 {calls}"
    print("✅ A9 未标记工具不触发审批通过")


def test_approval_skip_flag_bypasses_callback():
    """A9：skip_approval=True 时跳过审批（并发内部路径依赖此语义）。"""
    calls = []
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(
        lambda name, args: (calls.append(name), (False, None, "rejected"))[1]
    )
    r = reg.invoke_tool("approval_tool", {"expression": "4+4"}, skip_approval=True)
    assert r.succeeded, r.output
    assert r.output == "expr=4+4"
    assert calls == [], "skip_approval=True 不应触发回调"
    print("✅ A9 skip_approval 跳过审批通过")


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
    test_approval_approved()  # A9
    test_approval_rejected()  # A9
    test_approval_modified_args()  # A9
    test_approval_callback_exception()  # A9
    test_approval_skipped_without_callback()  # A9
    test_approval_not_triggered_for_unmarked_tool()  # A9
    test_approval_skip_flag_bypasses_callback()  # A9


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
