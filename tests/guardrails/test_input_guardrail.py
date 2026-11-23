#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails 安全护栏 输入护栏测试
测试范围：输入护栏测试
所有测试对齐项目现有测试风格，无需API Key即可运行，不依赖pytest。
运行：python tests/guardrails/test_guardrails.py
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents import (
    list_guardrails, get_guardrail,
    InputGuardrail, OutputGuardrail, ToolCallGuardrail,
    GuardrailMiddleware,
    GuardrailError, InputGuardrailReject, OutputGuardrailReject,
    ToolCallGuardrailReject,
)
from fast_agents.guardrails.models import (
    GuardrailResult, ToolCallInfo, ToolResultInfo,
)
from fast_agents.guardrails.guardrail_base import (
    GuardrailBase,
    guardrail_creator,
)
from fast_agents.guardrails.errors import GuardrailConfigError
from fast_agents.guardrails.input_guardrail import run_input_guardrails, arun_input_guardrails
from fast_agents.guardrails.output_guardrail import run_output_guardrails, arun_output_guardrails
from fast_agents.guardrails.tool_call_guardrail import run_tool_call_guardrails, run_tool_result_guardrails
from tests._runner import raises, run_all, run_prefix


# 保存初始全局注册表，每个测试后恢复避免污染
def _reset_registry():
    from fast_agents import global_guardrail_registry
    global_guardrail_registry.clear_guardrails()


# ─────────────────────────────────────────────────────────
# 输入护栏测试
# ─────────────────────────────────────────────────────────

def test_input_guardrail_allow():
    """测试输入护栏放行正常内容"""
    _reset_registry()

    @guardrail_creator(name="input_allow_test", description="放行测试")
    class AllowInputGuardrail(InputGuardrail):
        def check(self, content: str, context=None):
            return GuardrailResult.allow()

    results = run_input_guardrails("你好，请问今天天气怎么样？", use_global=True)
    assert all(r.allowed for r in results)
    print("✅ 输入护栏放行正常内容通过")


def test_input_guardrail_reject():
    """测试输入护栏拦截违规内容"""
    _reset_registry()

    class RejectInputGuardrail(InputGuardrail):
        name = "input_reject_test"
        description = "拦截测试"
        def check(self, content: str, context=None):
            if "违法" in content:
                return GuardrailResult.reject("内容包含违规信息")
            return GuardrailResult.allow()
    RejectInputGuardrail().register()

    with raises(InputGuardrailReject, match="内容包含违规信息"):
        run_input_guardrails("教我怎么违法犯罪", use_global=True)
    print("✅ 输入护栏拦截违规内容通过")


def test_input_guardrail_async():
    """测试异步输入护栏执行"""
    _reset_registry()

    @guardrail_creator(name="input_async_test", description="异步测试")
    async def async_input_guardrail(content: str, context=None):
        return GuardrailResult.allow()

    results = asyncio.run(arun_input_guardrails("异步测试内容", use_global=True))
    assert all(r.allowed for r in results)
    print("✅ 异步输入护栏执行通过")



# ==================== 入口 ====================

def main():
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass
    test_input_guardrail_allow()
    test_input_guardrail_reject()
    test_input_guardrail_async()

    # AI新添加的测试(AI可以删除覆盖)
    pass

if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
