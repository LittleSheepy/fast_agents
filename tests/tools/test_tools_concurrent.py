"""Tools 模块单元测试 —— 并发执行（A8 / A16）

从 test_tools.py 拆分而来，聚焦 ToolRegistry.invoke_tools_concurrent 的并发语义：
    并行提速 / 空调用 / 单工具异常隔离 / 整体超时 / 同名多次 / 审批前置 / 线程不套娃 / 超时标记。

运行：python tests/tools/test_tools_concurrent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools import ToolBase, ToolRegistry, tool


def _sleep_tool(name: str, seconds: float):
    """构造一个耗时 seconds 的测试工具（并发测试需要不同名的工具共存）。"""
    import time

    class _SleepTool(ToolBase):
        def run(self):
            time.sleep(seconds)
            return f"{name} done"

    return _SleepTool(name=name, description=f"耗时 {seconds}s 的测试工具")


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


# ─────────────────────────────────────────────────────────
# A8：并发执行
# ─────────────────────────────────────────────────────────

def test_concurrent_runs_in_parallel():
    """A8：多个无依赖工具并发执行，总耗时取最大值而非累加；返回 List 按 calls 顺序一一对应。"""
    import time
    reg = ToolRegistry(max_workers=4, default_timeout=300)
    reg.register_tools(
        _sleep_tool("s1", 0.3),
        _sleep_tool("s2", 0.3),
        _sleep_tool("s3", 0.3),
    )
    calls = [("s1", None), ("s2", None), ("s3", None)]
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls)
    elapsed = time.perf_counter() - start
    assert len(results) == 3, f"返回长度应与 calls 相等（3），实际 {len(results)}"
    assert isinstance(results, list), f"返回类型应为 List，实际 {type(results).__name__}"
    assert all(r.succeeded for r in results), results
    # 串行约需 0.9s；并发应接近 0.3s，上界留足抖动余量
    assert elapsed < 0.7, f"并发未生效，总耗时 {elapsed:.3f}s（串行约 0.9s）"
    print(f"✅ A8 并发总耗时 {elapsed:.3f}s（< 0.7s）通过")


def test_concurrent_empty_calls_returns_empty():
    """A8：空调用列表返回空列表，不报错。"""
    reg = ToolRegistry()
    assert reg.invoke_tools_concurrent([]) == []
    print("✅ A8 空调用列表返回 [] 通过")


def test_concurrent_exception_isolated():
    """A8/A16①：单个工具抛异常不影响其它工具，异常那个以失败返回（按 calls 顺序一一对应）。"""

    class BoomTool(ToolBase):
        name = "boom_tool"
        description = "必然抛异常的测试工具"

        def run(self):
            raise ValueError("故意炸")

    reg = ToolRegistry()
    reg.register_tools(_sleep_tool("ok_tool", 0.05), BoomTool())
    calls = [("ok_tool", None), ("boom_tool", None)]
    results = reg.invoke_tools_concurrent(calls)
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["ok_tool"].succeeded
    assert name_to_result["boom_tool"].failed
    assert "故意炸" in name_to_result["boom_tool"].output
    print("✅ A8/A16① 单工具异常被隔离通过")


def test_concurrent_partial_timeout():
    """A8/A16④：整体超时后，已完成的保留结果，未完成的标记超时失败并带 timeout_abandoned=True。"""
    reg = ToolRegistry(max_workers=4)
    reg.register_tools(_sleep_tool("fast_tool", 0.05), _sleep_tool("slow_tool", 0.5))
    calls = [("fast_tool", None), ("slow_tool", None)]
    import time
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls, timeout=0.2)
    elapsed = time.perf_counter() - start
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["fast_tool"].succeeded, results
    assert name_to_result["slow_tool"].failed
    assert "超时" in name_to_result["slow_tool"].output
    assert name_to_result["slow_tool"].metadata.get("timeout_abandoned") is True, \
        f"超时结果应标记 timeout_abandoned=True，实际 metadata={name_to_result['slow_tool'].metadata}"
    # A16③-b：超时后立即返回，不再等慢线程跑完（慢工具 0.5s，若仍阻塞就会接近 0.5s）
    assert elapsed < 0.35, f"超时后仍阻塞 {elapsed:.3f}s，shutdown(wait=False) 未生效"
    print(f"✅ A8/A16④ 整体超时：快的保留、慢的标记超时且快速返回（{elapsed:.3f}s）通过")


# ─────────────────────────────────────────────────────────
# A16: 并发语义加固（List返回、审批前置、线程不套娃、超时标记）
# ─────────────────────────────────────────────────────────

def test_concurrent_same_tool_twice_preserves_both_results():
    """A16①（A15 转正）：同一工具在一次并发里调用两次，两条结果都保留，顺序与输入一致。"""
    @tool
    def echo2(text: str) -> str:
        """回显文本"""
        return f"ECHO={text}"

    reg = ToolRegistry()
    reg.register_tool(echo2)
    calls = [("echo2", {"text": "A"}), ("echo2", {"text": "B"})]
    results = reg.invoke_tools_concurrent(calls)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].succeeded and results[1].succeeded
    assert results[0].output == "ECHO=A", f"第0位结果错误：{results[0].output!r}"
    assert results[1].output == "ECHO=B", f"第1位结果错误：{results[1].output!r}"
    print("✅ A16① 同名工具并发两条结果都在、顺序正确通过")


def test_concurrent_approval_called_per_tool_on_main_thread():
    """A16②：审批在主线程集中触发，每个需审批工具恰好触发一次回调，不进入子线程重复审批。"""
    import threading

    approvals: List[Tuple[str, dict]] = []
    approval_threads: List[str] = []
    main_thread = threading.current_thread().name

    def _cb(name, args):
        approvals.append((name, args))
        approval_threads.append(threading.current_thread().name)
        return (True, None, "")

    tool_a = _approval_tool("app_a")
    tool_b = _approval_tool("app_b")

    class PlainTool(ToolBase):
        name = "plain_tool"
        description = "无需审批"
        def run(self):
            return "ok"

    reg = ToolRegistry()
    reg.register_tools(tool_a, tool_b, PlainTool())
    reg.set_approval_callback(_cb)
    results = reg.invoke_tools_concurrent([
        ("app_a", {"expression": "1+1"}),
        ("plain_tool", None),
        ("app_b", {"expression": "2+2"}),
    ])
    assert len(results) == 3
    assert all(r.succeeded for r in results), results
    # 仅两个需审批工具触发回调
    assert len(approvals) == 2, f"应触发2次审批，实际 {len(approvals)}: {approvals}"
    approved_names = {n for n, _ in approvals}
    assert approved_names == {"app_a", "app_b"}
    # 全部审批都在主线程（不在子工作线程）
    assert all(t == main_thread for t in approval_threads), \
        f"审批应在主线程 {main_thread}，实际线程: {approval_threads}"
    print("✅ A16② 审批主线程集中触发、每工具一次通过")


def test_concurrent_approval_rejected_skips_execution():
    """A16②：被拒绝的工具不进入线程池执行，其余工具正常完成。"""
    executed: List[str] = []

    class SpyA(ToolBase):
        name = "spy_a"
        description = "spy a"
        metadata = {"requires_approval": True}
        def run(self):
            executed.append("a")
            return "a done"

    class SpyB(ToolBase):
        name = "spy_b"
        description = "spy b"
        metadata = {"requires_approval": True}
        def run(self):
            executed.append("b")
            return "b done"

    reg = ToolRegistry()
    reg.register_tools(SpyA(), SpyB())
    reg.set_approval_callback(lambda name, args: (False, None, "no") if name == "spy_a" else (True, None, ""))
    results = reg.invoke_tools_concurrent([("spy_a", None), ("spy_b", None)])
    name_to_result = {"spy_a": results[0], "spy_b": results[1]}
    assert name_to_result["spy_a"].failed
    assert "被拒绝" in name_to_result["spy_a"].output
    assert name_to_result["spy_b"].succeeded
    assert executed == ["b"], f"被拒工具不应执行，实际执行了 {executed}"
    print("✅ A16② 被拒工具不执行、其余正常通过")


def test_concurrent_timeout_returns_promptly():
    """A16③-b：整体超时后 shutdown(wait=False) 立即返回，不再等慢线程跑完。"""
    import time
    reg = ToolRegistry(max_workers=4)
    reg.register_tools(_sleep_tool("q_fast", 0.03), _sleep_tool("q_slow", 0.6))
    calls = [("q_fast", None), ("q_slow", None)]
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls, timeout=0.2)
    elapsed = time.perf_counter() - start
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["q_fast"].succeeded
    assert name_to_result["q_slow"].failed
    assert elapsed < 0.35, f"超时后仍阻塞 {elapsed:.3f}s，shutdown(wait=False) 未生效"
    print(f"✅ A16③-b 超时立即返回（{elapsed:.3f}s）通过")


def test_concurrent_no_nested_executor():
    """A16③-a：并发入口子任务直接调用 tool.invoke，不再经过 invoke_tool 的内层超时池；
    线程峰值应 ≈ 基线 + 并发 worker 数 + 1（主线程），不再出现线程套娃。
    """
    import threading, time
    peak = [0]
    active = [0]
    lock = threading.Lock()

    class CountTool(ToolBase):
        name = "count_tool"
        description = "统计并发线程峰值"
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx
            self.name = f"count_{idx}"
            self.description = f"计数器 {idx}"
        def run(self):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.15)
            with lock:
                active[0] -= 1
            return f"c{self.idx}"

    # 启动前后台稳定，确保不混入其他线程
    baseline = threading.active_count()
    reg = ToolRegistry(max_workers=3)
    reg.register_tools(CountTool(1), CountTool(2), CountTool(3))
    results = reg.invoke_tools_concurrent([
        ("count_1", None), ("count_2", None), ("count_3", None),
    ])
    assert all(r.succeeded for r in results), results
    # 修复前内层套池：3 个外层 worker + 3 个内层超时线程 = 峰值至少 baseline+6；
    # 修复后子任务直接执行：峰值 = baseline + 3（3 个 worker），不会再多出内层线程
    assert peak[0] <= baseline + 3 + 1, (
        f"线程套娃未消除：基线 {baseline}，峰值 {peak[0]}，差值 {peak[0] - baseline}"
    )
    print(f"✅ A16③-a 并发无嵌套池（基线 {baseline}，峰值 {peak[0]}）通过")


def test_timeout_abandoned_metadata_marker():
    """A16④：超时结果 metadata 携带 timeout_abandoned=True，便于下游识别。"""
    reg = ToolRegistry()
    reg.register_tool(_sleep_tool("slow_marker", 0.5))
    r = reg.invoke_tool("slow_marker", {}, timeout=0.1)
    assert r.failed
    assert r.metadata.get("timeout_abandoned") is True, f"缺 timeout_abandoned 标记：{r.metadata}"
    print("✅ A16④ 单次超时结果带 timeout_abandoned=True 通过")


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
    test_concurrent_runs_in_parallel()  # A8
    test_concurrent_empty_calls_returns_empty()  # A8
    test_concurrent_exception_isolated()  # A8/A16
    test_concurrent_partial_timeout()  # A8/A16
    test_concurrent_same_tool_twice_preserves_both_results()  # A16
    test_concurrent_approval_called_per_tool_on_main_thread()  # A16
    test_concurrent_approval_rejected_skips_execution()  # A16
    test_concurrent_timeout_returns_promptly()  # A16
    test_concurrent_no_nested_executor()  # A16
    test_timeout_abandoned_metadata_marker()  # A16


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
