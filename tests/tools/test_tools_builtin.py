"""Tools 模块单元测试 —— 内置工具

从 test_tools.py 拆分而来，聚焦 tools/builtin 下的内置工具：
    CalculatorTool / TerminalTool / ReadTool / WriteTool（含平台守卫与截断委托）。

运行：python tests/tools/test_tools_builtin.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools import builtin


# ─────────────────────────────────────────────────────────
# 5. 内置工具
# ─────────────────────────────────────────────────────────

def test_calculator():
    calc = builtin.CalculatorTool()

    cases = [
        ("2+3*4", "14"),
        ("sqrt(16)", "4"),
        ("sin(pi/2)", "1"),
        ("2**10", "1024"),
        ("2^10", "8"),
        ("(3+5)*2", "16"),
    ]
    for expr, expected in cases:
        r = calc.invoke(expression=expr)
        assert r.succeeded, f"{expr} -> {r.output}"
        # 结果以 "expr = " 开头，取等号后
        got = r.output.split("=", 1)[-1].strip()
        assert got == expected, f"{expr} 期望 {expected}，得到 {got}"

    # 错误用例
    r = calc.invoke(expression="1/0")
    assert r.failed
    assert "除零" in r.output

    r = calc.invoke(expression="import os")
    assert r.failed
    print("✅ CalculatorTool 通过")


def test_terminal():
    """TerminalTool：Windows 专用只读命令白名单 + 沙箱 + cd 校验"""
    if os.name != "nt":
        print("⏭️  test_terminal 跳过（TerminalTool 是 Windows 专用，当前非 Windows）")
        return
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hello.txt").write_text("hello fast_agents\n", encoding="utf-8")
        term = builtin.TerminalTool(workspace=tmp)

        # 1) 白名单内的只读命令
        r = term.invoke(command="dir")
        assert r.succeeded, r.output
        assert "hello.txt" in r.output

        r = term.invoke(command="type hello.txt")
        assert r.succeeded, r.output
        assert "hello fast_agents" in r.output

        r = term.invoke(command="echo hi")
        assert r.succeeded, r.output

        # 2) 白名单外的命令被拒（pwd / ls / rm 都不是 Windows 白名单命令）
        r = term.invoke(command="rm -rf /")
        assert r.failed
        assert "不允许" in r.output

        # 3) 已从白名单移除的写命令（del / copy / move / ren / md）也被拒
        r = term.invoke(command="del hello.txt")
        assert r.failed
        assert "不允许" in r.output

        # 4) cd 单独执行：持久切换当前目录；越界则拒绝
        (Path(tmp) / "sub").mkdir()
        r = term.invoke(command="cd sub")
        assert r.succeeded, r.output
        assert "sub" in r.output

        r = term.invoke(command="cd ..\\..\\..")
        assert r.failed
        assert "工作目录外" in r.output

        # 5) cd 与其它命令拼接 → 直接拒绝（防链中 cd 绕过沙箱）
        r = term.invoke(command="dir && cd sub")
        assert r.failed
        assert "拼接" in r.output
    print("✅ TerminalTool 通过")


def test_read_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        rf = builtin.ReadTool(workspace=tmp)

        # 创建测试文件
        f = Path(tmp) / "test.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")

        r = rf.invoke(path="test.txt")
        assert r.succeeded
        assert "共 50 行" in r.output
        assert "line 1" in r.output
        assert "line 50" in r.output

        # 按行号
        r = rf.invoke(path="test.txt", start_line=3, end_line=5, max_lines=10)
        assert "line 3" in r.output
        assert "line 5" in r.output
        assert "line 1" not in r.output

        # 路径穿越拦截
        r = rf.invoke(path="../etc/passwd")
        assert r.failed
        assert "工作目录外" in r.output
    print("✅ ReadTool 通过")


def test_write_append_preserves_existing_content():
    """A6：write_file 的 append=True 追加而非覆盖。"""
    import tempfile
    from pathlib import Path
    from fast_agents.tools.builtin.file_tool import WriteTool

    with tempfile.TemporaryDirectory() as tmp:
        wt = WriteTool(workspace=tmp)
        wt.invoke(path="log.txt", content="line1")
        r = wt.invoke(path="log.txt", content="line2", append=True)
        assert r.succeeded, r.output
        content = (Path(tmp) / "log.txt").read_text(encoding="utf-8")
        assert "line1" in content and "line2" in content, f"追加丢失原内容: {content!r}"
        assert content.index("line1") < content.index("line2"), "追加内容顺序错误"
    print("✅ write_file append=True 追加不覆盖通过")


def test_terminal_platform_guard():
    """N3：非 Windows 平台构造 TerminalTool 应明确报错，而非静默假成功。"""
    import tempfile
    from fast_agents.tools.tool_exception import ToolInitializationError

    if os.name == "nt":
        with tempfile.TemporaryDirectory() as tmp:
            term = builtin.TerminalTool(workspace=tmp)
            assert term is not None
        print("✅ TerminalTool 平台守卫通过（Windows：正常构造）")
        return

    with tempfile.TemporaryDirectory() as tmp:
        try:
            builtin.TerminalTool(workspace=tmp)
        except ToolInitializationError as e:
            assert "Windows" in str(e), f"错误信息未说明平台限制: {e}"
            print("✅ TerminalTool 平台守卫通过（非 Windows：明确报错）")
            return
    raise AssertionError("非 Windows 平台构造 TerminalTool 应抛 ToolInitializationError，实际未报错")


def test_terminal_truncation_delegated_to_base():
    """N2：terminal 输出过长由基类统一截断为 TRUNCATED，不再手动二次截断。"""
    import tempfile
    import unittest.mock as mock
    import fast_agents.tools.builtin.terminal_tool as tt

    long_output = "line\n" * 2000
    with mock.patch.object(tt, "_is_windows", return_value=True), \
         mock.patch.object(tt, "route_and_run", return_value=(0, long_output, "", "cmd")):
        with tempfile.TemporaryDirectory() as tmp:
            term = tt.TerminalTool(workspace=tmp, max_output_chars=100)
            r = term.invoke(command="dir")

    assert r.succeeded, r.output
    assert r.is_truncated, f"超长输出应被基类截断为 TRUNCATED，实际 status={r.status}"
    assert "输出已截断" in r.output, "缺少基类截断标记"
    assert "输出过长，已截断前" not in r.output, "仍存在旧的手动截断标记（二次截断未清除）"
    print("✅ terminal 输出截断统一由基类处理通过")


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
    test_calculator()  # builtin
    test_terminal()  # builtin
    test_read_file()  # builtin
    test_write_append_preserves_existing_content()  # A6
    test_terminal_platform_guard()  # N3
    test_terminal_truncation_delegated_to_base()  # N2


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
