#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails 注册表分离测试：GuardrailRegistry / global_guardrail_registry
测试范围：
  R1  门面 / 顶层 / 旧别名指向同一 GuardrailRegistry 单例
  R2  语义 API：register / get（不存在返回 None 不抛）/ is_registered / list（按 priority 升序）
  R3  unregister 返回 bool、可传实例或名称
  R4  批量：clear / update / extend
  R5  dict-like 兼容层：in / len / values / copy / update / clear
  R6  装饰器注册后可在注册表查到；GuardrailMiddleware 预设原型登记在注册表
运行：python tests/guardrails/test_guardrail_registry.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fast_agents
from fast_agents.guardrails import (
    GuardrailRegistry,
    global_guardrail_registry,
    guardrail_creator,
    GuardrailResult,
    GuardrailMiddleware,
)
from fast_agents.guardrails.guardrail_base import (
    _GLOBAL_GUARDRAIL_REGISTRY,
    get_global_guardrail_registry,
    get_guardrail,
    list_guardrails,
    clear_guardrails,
    update_global_guardrail_registry,
)
from tests._runner import run_all


def _reset():
    global_guardrail_registry.clear_guardrails()


def _make(name, priority=100):
    @guardrail_creator(name=name, description="测试护栏", priority=priority)
    def _g(input_data, context=None):
        return GuardrailResult.allow()
    return _g


def test_R1_registry_singleton_and_aliases():
    _reset()
    assert isinstance(global_guardrail_registry, GuardrailRegistry)
    assert _GLOBAL_GUARDRAIL_REGISTRY is global_guardrail_registry
    assert get_global_guardrail_registry() is global_guardrail_registry
    assert fast_agents.global_guardrail_registry is global_guardrail_registry
    assert fast_agents.GuardrailRegistry is GuardrailRegistry
    print("✅ R1 单例与别名一致")


def test_R2_register_lookup_and_list_order():
    _reset()
    g10 = _make("r_p10", priority=10)
    _make("r_p50", priority=50)
    assert global_guardrail_registry.is_registered("r_p10")
    assert global_guardrail_registry.is_registered("r_p50")
    assert global_guardrail_registry.get_guardrail("r_p10") is g10
    assert global_guardrail_registry.get_guardrail("__missing__") is None
    assert get_guardrail("__missing__") is None  # 不抛 KeyError
    names = [g.name for g in global_guardrail_registry.list_guardrails()]
    assert names.index("r_p10") < names.index("r_p50"), names
    func_names = [g.name for g in list_guardrails()]
    assert func_names.index("r_p10") < func_names.index("r_p50")
    print("✅ R2 注册 / 查找 / priority 排序")


def test_R3_unregister_returns_bool():
    _reset()
    _make("r_unreg")
    assert global_guardrail_registry.unregister_guardrail("r_unreg") is True
    assert global_guardrail_registry.unregister_guardrail("r_unreg") is False
    g2 = _make("r_unreg2")
    assert global_guardrail_registry.unregister_guardrail(g2) is True
    assert global_guardrail_registry.unregister_guardrail(g2) is False
    print("✅ R3 unregister 返回 bool，支持名称 / 实例")


def test_R4_batch_ops():
    _reset()
    _make("r_a")
    global_guardrail_registry.clear_guardrails()
    assert len(global_guardrail_registry) == 0
    a = _make("r_u1")
    b = _make("r_u2")
    global_guardrail_registry.update_guardrails({"x1": a, "x2": b})
    assert global_guardrail_registry.get_guardrail("x1") is a
    assert global_guardrail_registry.get_guardrail("x2") is b
    clear_guardrails()
    c = _make("r_e1")
    d = _make("r_e2")
    global_guardrail_registry.extend_guardrails([c, d])
    assert global_guardrail_registry.is_registered("r_e1")
    assert global_guardrail_registry.is_registered("r_e2")
    update_global_guardrail_registry({"r_e1": c})
    assert global_guardrail_registry.get_guardrail("r_e1") is c
    print("✅ R4 clear / update / extend 批量操作")


def test_R5_dict_like_compat():
    _reset()
    g = _make("r_compat")
    assert "r_compat" in _GLOBAL_GUARDRAIL_REGISTRY
    assert len(_GLOBAL_GUARDRAIL_REGISTRY) == 1
    assert list(_GLOBAL_GUARDRAIL_REGISTRY.values())[0] is g
    snap = _GLOBAL_GUARDRAIL_REGISTRY.copy()
    assert isinstance(snap, dict) and "r_compat" in snap
    _GLOBAL_GUARDRAIL_REGISTRY.clear()
    assert len(_GLOBAL_GUARDRAIL_REGISTRY) == 0
    _GLOBAL_GUARDRAIL_REGISTRY.update(snap)
    assert _GLOBAL_GUARDRAIL_REGISTRY.get("r_compat") is g
    print("✅ R5 dict-like 兼容层")


def test_R6_creator_and_presets_registered():
    _reset()
    _make("r_deco")
    assert global_guardrail_registry.get_guardrail("r_deco") is not None
    _reset()
    GuardrailMiddleware()
    for pn in ("basic_input_safety", "basic_output_safety", "prompt_injection", "basic_permission"):
        assert global_guardrail_registry.is_registered(pn), "%s 未登记到全局注册表" % pn
    _reset()
    print("✅ R6 装饰器 / 中间件预设登记")


def main():
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    test_R1_registry_singleton_and_aliases()
    test_R2_register_lookup_and_list_order()
    test_R3_unregister_returns_bool()
    test_R4_batch_ops()
    test_R5_dict_like_compat()
    test_R6_creator_and_presets_registered()


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
