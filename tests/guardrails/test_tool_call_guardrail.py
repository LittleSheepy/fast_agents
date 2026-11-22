#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails 安全护栏 工具调用护栏测试
测试范围：工具调用护栏测试
所有测试对齐项目现有测试风格，无需API Key即可运行，不依赖pytest。
运行：python tests/guardrails/test_guardrails.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.guardrails.tool_call_guardrail import (
    ToolCallGuardrail, run_tool_call_guardrails, run_tool_result_guardrails,
)
from fast_agents.guardrails.models import (
    GuardrailResult, ToolCallInfo, ToolResultInfo,
)
from fast_agents.guardrails.errors import ToolCallGuardrailReject
from fast_agents.guardrails.guardrail_base import guardrail_creator
from tests._runner import raises, run_all


# 保存初始全局注册表，每个测试后恢复避免污染
def _reset_registry():
    from fast_agents.guardrails.guardrail_base import _GLOBAL_GUARDRAIL_REGISTRY
    _GLOBAL_GUARDRAIL_REGISTRY.clear()


def test_tool_call_allow():
    """测试工具调用放行正常操作"""
    _reset_registry()

    @guardrail_creator(name="tool_call_allow_test", description="工具放行测试")
    class AllowToolGuardrail(ToolCallGuardrail):
        def before_call(self, tool_call: ToolCallInfo, context=None):
            return GuardrailResult.allow()

    tool_info = ToolCallInfo(tool_name="calculator", tool_args={"expression": "1+1"})
    results = run_tool_call_guardrails(tool_info, use_global=True)
    assert all(r.allowed for r in results)
    print("✅ 工具调用放行正常操作通过")


def test_tool_call_reject_high_risk():
    """测试工具调用拦截高危操作"""
    _reset_registry()

    class RejectToolGuardrail(ToolCallGuardrail):
        name = "tool_call_reject_test"
        description = "高危工具拦截测试"
        def before_call(self, tool_call: ToolCallInfo, context=None):
            if tool_call.tool_name == "shell_exec" and "rm -rf" in str(tool_call.tool_args):
                return GuardrailResult.reject("禁止执行高危删除命令")
            return GuardrailResult.allow()
    RejectToolGuardrail().register()

    tool_info = ToolCallInfo(tool_name="shell_exec", tool_args={"command": "rm -rf /"})
    with raises(ToolCallGuardrailReject, match="禁止执行高危删除命令"):
        run_tool_call_guardrails(tool_info, use_global=True)
    print("✅ 工具调用拦截高危操作通过")


def test_tool_result_sensitive_data_block():
    """测试工具返回结果拦截敏感数据泄露"""
    _reset_registry()

    class BlockResultGuardrail(ToolCallGuardrail):
        name = "tool_result_block_test"
        description = "结果拦截测试"
        check_result = True
        def before_call(self, tool_call: ToolCallInfo, context=None):
            return GuardrailResult.allow()
        def after_call(self, tool_result: ToolResultInfo, context=None):
            if "密码" in str(tool_result.output):
                return GuardrailResult.reject("工具结果包含敏感密码信息")
            return GuardrailResult.allow()
    BlockResultGuardrail().register()

    tool_result = ToolResultInfo(tool_name="db_query", success=True, output="用户密码是admin123")
    with raises(ToolCallGuardrailReject, match="敏感密码信息"):
        run_tool_result_guardrails(tool_result, use_global=True)
    print("✅ 工具返回结果拦截敏感数据泄露通过")



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

    # AI新添加的测试(AI可以删除覆盖)
    pass

if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
