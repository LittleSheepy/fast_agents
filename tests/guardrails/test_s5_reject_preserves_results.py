#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5: run_*_guardrails 遇 reject 时不丢弃已通过结果的回归测试

验证 bug 修复前：某护栏 reject 后前面已通过项的 _pass_stats 与 trace 全部丢失；
验证 bug 修复后：exception 携带 e.results，middleware 补记录，统计准确。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.guardrails import (
    GuardrailMiddleware, GuardrailResult,
    InputGuardrail, OutputGuardrail, ToolCallGuardrail,
    guardrail_creator,
)
from fast_agents.guardrails.models import ToolCallInfo
from fast_agents.guardrails.preset import BasicInputSafetyGuardrail


# ---------- 辅助：构造一个"前面通过、后面拦截"的护栏链 ----------
class PassInputA(InputGuardrail):
    description = "pass A"
    name = "pass_a"
    priority = 10
    def check(self, input_data, context=None):
        return GuardrailResult.allow(metadata={"marker": "A"})


class RejectInputB(InputGuardrail):
    description = "reject B"
    name = "reject_b"
    priority = 20
    def check(self, input_data, context=None):
        return GuardrailResult.reject("B blocks it")


class PassInputC(InputGuardrail):
    description = "pass C"
    """优先级比 B 低，B 拦截后它不应被执行，但前面的 A 必须被记录"""
    name = "pass_c"
    priority = 30
    def check(self, input_data, context=None):
        return GuardrailResult.allow(metadata={"marker": "C"})


class PassOutputA(OutputGuardrail):
    description = "pass out A"
    name = "pass_out_a"
    priority = 10
    def check(self, input_data, context=None):
        return GuardrailResult.allow(metadata={"marker": "OA"})


class RejectOutputB(OutputGuardrail):
    description = "reject out B"
    name = "reject_out_b"
    priority = 20
    def check(self, input_data, context=None):
        return GuardrailResult.reject("OB blocks it")


class PassToolA(ToolCallGuardrail):
    description = "pass tool A"
    name = "pass_tool_a"
    priority = 10
    def check(self, input_data, context=None):
        return GuardrailResult.allow(metadata={"marker": "TA"})


class RejectToolB(ToolCallGuardrail):
    description = "reject tool B"
    name = "reject_tool_b"
    priority = 20
    def check(self, input_data, context=None):
        return GuardrailResult.reject("TB blocks it")


# ---------- 测试用例 ----------
def test_s5_1_input_reject_records_preceding_passes():
    """输入护栏：B 拦截时 A 必须被统计到 _pass_stats，不能丢。"""
    mw = GuardrailMiddleware(enable_presets=False)
    mw.add_guardrail(PassInputA())
    mw.add_guardrail(RejectInputB())
    mw.add_guardrail(PassInputC())

    result = mw.before_agent_run("hello")

    # 必须拦截
    assert result is not None, "应被 B 拦截"
    assert result.phase == "input"

    stats = mw.get_stats()
    # A 通过 → 计数 1（pass 按护栏名一级键）
    assert stats["pass"].get("pass_a") == 1, \
        f"pass_a 应被记录为通过，实际 stats={stats}"
    # B 拦截 → reject 计数 >= 1（正常拦截至少 1）
    assert stats["reject"].get("reject_b", 0) >= 1, \
        f"reject_b 应被记录为拦截，实际 stats={stats}"
    # C 因 B 先拦截没被执行 → 不出现
    assert stats["pass"].get("pass_c") is None, \
        f"pass_c 不应执行，实际 stats={stats}"


def test_s5_2_output_reject_records_preceding_passes():
    """输出护栏：B 拦截时 A 必须被记录。"""
    mw = GuardrailMiddleware(enable_presets=False)
    mw.add_guardrail(PassOutputA())
    mw.add_guardrail(RejectOutputB())

    result = mw.after_agent_run("some output")
    assert result is not None and result.phase == "output"

    stats = mw.get_stats()
    assert stats["pass"].get("pass_out_a") == 1, \
        f"pass_out_a 应被记录为通过，实际 stats={stats}"
    assert stats["reject"].get("reject_out_b", 0) >= 1


def test_s5_3_tool_pre_reject_records_preceding_passes():
    """工具前置护栏：B 拦截时 A 必须被记录，且 before_tool_call 返回拦截。"""
    mw = GuardrailMiddleware(enable_presets=False)
    mw.add_guardrail(PassToolA())
    mw.add_guardrail(RejectToolB())

    info = ToolCallInfo(tool_name="any_tool", tool_args={})
    result = mw.before_tool_call(info.tool_name, info.tool_args)
    # Bug3 协议升级：拦截返回 ToolDecision(False, 原因)，`not result` 与旧 `result is False` 语义等价
    assert not result, "工具前置应被拦截"

    stats = mw.get_stats()
    assert stats["pass"].get("pass_tool_a") == 1, \
        f"pass_tool_a 应被记录为通过，实际 stats={stats}"
    assert stats["reject"].get("reject_tool_b", 0) >= 1
    # 拦截原因已写入
    assert "TB blocks it" in mw._last_tool_reject_reason
    # Bug3：同样的原因必须随返回值携带（不依赖共享属性）
    assert "TB blocks it" in result[1], f"Bug3：拦截原因应随返回值携带，实际 {result!r}"


def test_s5_4_all_pass_no_regression():
    """全部通过场景：正常统计，不丢数，也不多拦截。"""
    mw = GuardrailMiddleware(enable_presets=False)
    mw.add_guardrail(PassInputA())

    result = mw.before_agent_run("hello")
    assert result is None, "应放行"

    stats = mw.get_stats()
    assert stats["pass"].get("pass_a") == 1


def test_s5_5_async_input_reject_records_passes():
    """异步 before_agent_run：B 拦截时 A 必须被记录（验证 async 路径也补上了）。"""
    import asyncio
    mw = GuardrailMiddleware(enable_presets=False)
    mw.add_guardrail(PassInputA())
    mw.add_guardrail(RejectInputB())

    result = asyncio.run(mw.abefore_agent_run("hello"))
    assert result is not None and result.phase == "input"

    stats = mw.get_stats()
    assert stats["pass"].get("pass_a") == 1, \
        f"pass_a 异步路径应被记录为通过，实际 stats={stats}"
    assert stats["reject"].get("reject_b", 0) >= 1


# ---------- 入口（对齐仓库现有测试风格） ----------
def main_of_new_add():
    """S5 专项新增用例"""
    test_s5_1_input_reject_records_preceding_passes()
    test_s5_2_output_reject_records_preceding_passes()
    test_s5_3_tool_pre_reject_records_preceding_passes()
    test_s5_4_all_pass_no_regression()
    test_s5_5_async_input_reject_records_passes()
    print("S5 全部用例通过")


def main():
    """入口：先跑新增用例"""
    main_of_new_add()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
