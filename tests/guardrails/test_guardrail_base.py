#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails 安全护栏模块 GuardrailBase测试
测试范围：基类功能
所有测试对齐项目现有测试风格，无需API Key即可运行，不依赖pytest。
运行：python tests/guardrails/test_guardrail_base.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.guardrails.models import GuardrailResult
from fast_agents.guardrails.guardrail_base import (
    GuardrailBase, guardrail_creator,
    list_guardrails, get_guardrail,
)
from fast_agents.guardrails.errors import GuardrailConfigError
from tests._runner import raises, run_all


# 保存初始全局注册表，每个测试后恢复避免污染
def _reset_registry():
    from fast_agents import global_guardrail_registry
    global_guardrail_registry.clear_guardrails()


# ─────────────────────────────────────────────────────────
# 基类核心功能测试
# ─────────────────────────────────────────────────────────
def test_guardrail_result_structure():
    """测试 GuardrailResult 结构体字段正确性"""
    result = GuardrailResult(allowed=True, reason="测试通过")
    assert result.allowed is True
    assert result.reason == "测试通过"
    assert result.metadata == {}

    # 测试工厂方法
    allow_result = GuardrailResult.allow(metadata={"test": True})
    assert allow_result.allowed is True
    reject_result = GuardrailResult.reject("内容违规", metadata={"keyword": "test"})
    assert reject_result.allowed is False
    assert reject_result.reason == "内容违规"
    print("✅ GuardrailResult 结构体正确通过")


def test_guardrail_base_init_validation():
    """测试 GuardrailBase 初始化校验：name和description必填"""
    # 缺少name应该抛出错误
    with raises(GuardrailConfigError, match="护栏必须有 name"):
        class NoNameGuardrail(GuardrailBase):
            description = "测试无name护栏"
            def check(self, input_data, context=None):
                return GuardrailResult.allow()
        NoNameGuardrail()

    # 缺少description应该抛出错误
    with raises(GuardrailConfigError, match="必须有 description"):
        class NoDescGuardrail(GuardrailBase):
            name = "no_desc"
            def check(self, input_data, context=None):
                return GuardrailResult.allow()
        NoDescGuardrail()
    print("✅ GuardrailBase 初始化校验通过")

def test_guardrail_to_dict():
    """测试护栏序列化方法"""
    _reset_registry()

    @guardrail_creator(name="dict_test_guardrail", description="序列化测试", priority=30, tags=["test"])
    def dict_test(input_data, context=None):
        return GuardrailResult.allow()

    guardrail = get_guardrail("dict_test_guardrail")
    info = guardrail.to_dict()
    assert info["name"] == "dict_test_guardrail"
    assert info["priority"] == 30
    assert info["enabled"] is True
    assert "test" in info["tags"]
    print("✅ 护栏序列化方法通过")


def test_guardrail_enabled_toggle():
    """测试护栏启用/禁用开关"""
    _reset_registry()

    @guardrail_creator(name="enabled_test", description="启用开关测试")
    def enabled_guardrail(input_data, context=None):
        return GuardrailResult.reject("应该被拦截")

    # 默认启用，应该拦截
    g = get_guardrail("enabled_test")
    result = g.invoke("test")
    assert result.allowed is False

    # 禁用后直接放行
    g.enabled = False
    result = g.invoke("test")
    assert result.allowed is True
    print("✅ 护栏启用/禁用开关通过")


def test_guardrail_fail_open():
    """测试护栏自身异常时的姿态：默认 fail-closed（安全组件业界 posture）。"""
    _reset_registry()

    @guardrail_creator(name="error_test", description="异常测试")
    def error_guardrail(input_data, context=None):
        raise RuntimeError("护栏执行出错")

    g = get_guardrail("error_test")
    # 默认 fail_open=False：护栏异常 = 未知风险 → 拦截，走完整 reject 打点链路
    result = g.invoke("test")
    assert result.allowed is False, "默认 fail-closed：护栏异常应拦截"
    assert "guardrail_error" in result.metadata
    assert "duration_ms" in result.metadata, "异常路径也必须打 duration_ms，避免统计断层"
    assert result.metadata["guardrail_name"] == "error_test"

    # 显式 fail_open=True 时放行
    g_fail_open = get_guardrail("error_test").clone()
    g_fail_open.fail_open = True
    result = g_fail_open.invoke("test")
    assert result.allowed is True
    assert "guardrail_error" in result.metadata
    print("✅ 护栏异常默认 fail-closed、fail_open=True 放行通过")


def test_guardrail_creator_decorator():
    """测试 @guardrail_creator 装饰器注册功能"""
    _reset_registry()
    initial_count = len(list_guardrails())

    @guardrail_creator(name="test_custom_guardrail", description="自定义测试护栏", priority=50)
    def custom_guardrail(input_data, context=None):
        return GuardrailResult(allowed=True, reason="自定义测试护栏通过")

    # 验证注册成功
    assert len(list_guardrails()) == initial_count + 1
    guardrail_names = [g.name for g in list_guardrails()]
    assert "test_custom_guardrail" in guardrail_names

    # 验证获取实例
    guardrail = get_guardrail("test_custom_guardrail")
    assert guardrail is not None
    assert guardrail.name == "test_custom_guardrail"
    assert guardrail.priority == 50

    # 验证执行逻辑正确
    result = guardrail.invoke("test input")
    assert result.allowed is True
    print("✅ @guardrail_creator 装饰器注册功能通过")


def test_guardrail_priority_sorting():
    """测试护栏按优先级排序执行"""
    _reset_registry()

    @guardrail_creator(name="priority_100_test", description="priority 100", priority=100)
    def priority_100(input_data, context=None):
        return GuardrailResult.allow()

    @guardrail_creator(name="priority_1_test", description="priority 1", priority=1)
    def priority_1(input_data, context=None):
        return GuardrailResult.allow()

    guardrails = list_guardrails()
    # 优先级数字越小越先执行，priority 1应该在priority 100前面
    idx_1 = next(i for i, g in enumerate(guardrails) if g.name == "priority_1_test")
    idx_100 = next(i for i, g in enumerate(guardrails) if g.name == "priority_100_test")
    assert idx_1 < idx_100
    print("✅ 护栏优先级排序执行通过")



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
    test_guardrail_enabled_toggle()

    # AI新添加的测试(AI可以删除覆盖)
    pass

if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
