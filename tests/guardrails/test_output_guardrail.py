#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails 安全护栏 输出护栏测试
测试范围：输出护栏测试
所有测试对齐项目现有测试风格，无需API Key即可运行，不依赖pytest。
运行：python tests/guardrails/test_guardrails.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.guardrails.models import GuardrailResult
from fast_agents.guardrails.guardrail_base import guardrail_creator
from fast_agents.guardrails.errors import OutputGuardrailReject
from fast_agents.guardrails.output_guardrail import OutputGuardrail, run_output_guardrails
from tests._runner import raises, run_all


# 保存初始全局注册表，每个测试后恢复避免污染
def _reset_registry():
    from fast_agents import global_guardrail_registry
    global_guardrail_registry.clear_guardrails()


def test_output_guardrail_allow():
    """测试输出护栏放行正常内容"""
    _reset_registry()

    @guardrail_creator(name="output_allow_test", description="输出放行测试")
    class AllowOutputGuardrail(OutputGuardrail):
        def check(self, content: str, context=None):
            return GuardrailResult.allow()

    final_output, results = run_output_guardrails("今天天气很好，适合出门。", use_global=True)
    assert all(r.allowed for r in results)
    assert final_output == "今天天气很好，适合出门。"
    print("✅ 输出护栏放行正常内容通过")


def test_output_guardrail_reject():
    """测试输出护栏拦截敏感输出"""
    _reset_registry()

    class RejectOutputGuardrail(OutputGuardrail):
        name = "output_reject_test"
        description = "输出拦截测试"
        def check(self, content: str, context=None):
            if "身份证号" in content:
                return GuardrailResult.reject("禁止输出敏感个人信息")
            return GuardrailResult.allow()
    RejectOutputGuardrail().register()

    with raises(OutputGuardrailReject, match="禁止输出敏感个人信息"):
        run_output_guardrails("用户身份证号是110101199001011234", use_global=True)
    print("✅ 输出护栏拦截敏感输出通过")


def test_output_guardrail_modify_content():
    """测试输出护栏修改内容（脱敏场景）"""
    _reset_registry()

    class ModifyOutputGuardrail(OutputGuardrail):
        name = "output_modify_test"
        description = "输出修改测试"
        allow_modify = True
        def check(self, content: str, context=None):
            if "手机号" in content:
                return self.modify("用户手机号已脱敏：138****5678", reason="自动脱敏手机号")
            return GuardrailResult.allow()
    ModifyOutputGuardrail().register()

    final_output, results = run_output_guardrails("用户手机号是13812345678", use_global=True)
    assert "138****5678" in final_output
    assert "13812345678" not in final_output
    print("✅ 输出护栏修改内容（脱敏）通过")




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
    test_output_guardrail_allow()
    test_output_guardrail_reject()
    test_output_guardrail_modify_content()

    # AI新添加的测试(AI可以删除覆盖)
    pass

if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
