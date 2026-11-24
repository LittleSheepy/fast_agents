#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guardrails S4 bug 复现测试：tool_registry 晚注入不传播到 preset 副本
测试范围：
  s4_1  ReActAgent 晚注入后，preset 副本的 tool_registry 应为注入的 registry，而不是 None
  s4_2  注册了 requires_approval=True 的工具，BasicPermissionGuardrail.before_call 应拦截
  s4_3  构造 middleware 时显式传 tool_registry=... 的对照用例（应通过，证明路径代码本身正确）
  s4_4  动态给 middleware.tool_registry 赋值时，应同步传播到已挂载的 BasicPermissionGuardrail
  s4_5  端到端：经 middleware.before_tool_call 对 requires_approval 工具应拦截并携带 reject 原因
  s4_6  A2：先注入 tool_registry、后 add_guardrail 的权限护栏副本也要拿到它（单一事实源不能被时序击败）
运行：python tests/guardrails/test_s4_tool_registry_injection.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents import (
    GuardrailMiddleware,
    ToolRegistry,
    ToolBase,
)
from fast_agents.guardrails.preset import BasicPermissionGuardrail
from fast_agents.guardrails.models import ToolCallInfo
from tests._runner import run_all, run_prefix


# ==================== 测试工具 ====================

class _DangerousTool(ToolBase):
    name = "dangerous_db_op"
    description = "高危数据库操作"
    metadata = {"requires_approval": True}

    def run(self, *args, **kwargs):
        return "executed"


class _AnotherDangerousTool(ToolBase):
    name = "my_secret_op"
    description = "敏感操作"
    metadata = {"requires_approval": True}

    def run(self, *args, **kwargs):
        return "secret"


# ==================== 测试用例 ====================

def test_s4_1_preset_tool_registry_is_none_when_agent_injects_late():
    """S4_1：先构造 middleware（不传 tool_registry），再赋值 tool_registry 模拟 ReActAgent 晚注入，
    preset 副本里的 basic_permission 应拿到注入后的 registry，而不是 None。"""
    mw = GuardrailMiddleware(presets=("basic_permission",))
    # 模拟 react_agent.py:263 的晚注入
    registry = ToolRegistry()
    registry.register_tool(_DangerousTool())
    mw.tool_registry = registry

    bp = [g for g in mw.get_guardrails() if isinstance(g, BasicPermissionGuardrail) and g.name == "basic_permission"]
    assert len(bp) == 1, f"期望有 1 个 basic_permission 实例，实际 {len(bp)}"
    inst = bp[0]
    assert inst.tool_registry is registry, (
        f"preset 副本的 tool_registry 应为晚注入的 registry，实际为 {inst.tool_registry!r}"
    )


def test_s4_2_requires_approval_path_actually_blocks():
    """S4_2：在晚注入场景下，直接调用 BasicPermissionGuardrail.before_call，
    requires_approval=True 的工具应被拦截（走单一事实源分支 2），而不是错误放行。"""
    mw = GuardrailMiddleware(presets=("basic_permission",))
    registry = ToolRegistry()
    registry.register_tool(_AnotherDangerousTool())
    mw.tool_registry = registry  # 模拟 ReActAgent 晚注入

    bp = [g for g in mw.get_guardrails() if isinstance(g, BasicPermissionGuardrail) and g.name == "basic_permission"][0]
    info = ToolCallInfo(
        tool_name="my_secret_op",
        tool_args={"sql": "drop table users"},  # B3 修复暴露：字段名应为 tool_args，原文误写为 arguments
        call_id="call_1",
    )
    result = bp.before_call(info, context={})
    assert not result.allowed, (
        f"requires_approval=True 的工具应被单一事实源路径拦截，但实际 allowed=True, metadata={result.metadata}"
    )
    assert result.metadata.get("source") == "tool_registry_metadata"


def test_s4_3_explicit_tool_registry_in_constructor_still_works():
    """S4_3（对照）：构造 middleware 时显式传 tool_registry=...，single source 路径生效。
    这个用例在 bug 下也应通过，用于反向证明 before_call 逻辑本身是对的。"""
    registry = ToolRegistry()
    registry.register_tool(_DangerousTool())
    mw = GuardrailMiddleware(presets=("basic_permission",), tool_registry=registry)

    bp = [g for g in mw.get_guardrails() if isinstance(g, BasicPermissionGuardrail) and g.name == "basic_permission"][0]
    assert bp.tool_registry is registry
    info = ToolCallInfo(tool_name="dangerous_db_op", tool_args={}, call_id="call_2")  # B3 修复暴露：字段名应为 tool_args
    result = bp.before_call(info, context={})
    assert not result.allowed
    assert result.metadata.get("source") == "tool_registry_metadata"


def test_s4_4_setter_propagates_to_existing_presets_and_external_guards():
    """S4_4：动态给 middleware.tool_registry 赋值时，应同步传播到
    所有已挂载的 BasicPermissionGuardrail（preset 副本 + 外部 add_guardrail 注册的实例）。"""
    mw = GuardrailMiddleware(presets=("basic_permission",))
    # 外部再注册一个自定义 BasicPermissionGuardrail（不传 tool_registry）
    external = BasicPermissionGuardrail(name="my_custom_perm", blocked_tool_names=[])
    mw.add_guardrail(external)

    registry_v1 = ToolRegistry()
    registry_v1.register_tool(_DangerousTool())
    mw.tool_registry = registry_v1

    for inst in mw._instance_guardrails:
        if isinstance(inst, BasicPermissionGuardrail):
            assert inst.tool_registry is registry_v1, (
                f"BasicPermissionGuardrail(name={inst.name!r}) 的 tool_registry 未同步到 registry_v1"
            )

    # 二次替换：setter 应能持续广播
    registry_v2 = ToolRegistry()
    registry_v2.register_tool(_AnotherDangerousTool())
    mw.tool_registry = registry_v2
    for inst in mw._instance_guardrails:
        if isinstance(inst, BasicPermissionGuardrail):
            assert inst.tool_registry is registry_v2, (
                f"BasicPermissionGuardrail(name={inst.name!r}) 的 tool_registry 未同步到 registry_v2"
            )


def test_s4_5_manual_tool_registry_assignment_before_run_blocks_in_agent_loop():
    """S4_5：端到端模拟主循环——构造 middleware 不传 registry、手动赋值后再调 before_tool_call，
    requires_approval 工具应被拦截（原因随返回值携带并写入 _last_tool_reject_reason）。"""
    mw = GuardrailMiddleware(presets=("basic_permission",))
    registry = ToolRegistry()
    registry.register_tool(_AnotherDangerousTool())
    mw.tool_registry = registry  # 模拟主循环晚注入

    ok = mw.before_tool_call(
        tool_name="my_secret_op",
        tool_args={"op": "exfiltrate"},
        context={},
    )
    # Bug3 协议升级：拦截返回 ToolDecision(False, 原因)，`not ok` 与旧 `ok is False` 语义等价
    assert not ok, (
        f"middleware.before_tool_call 对 requires_approval=True 工具应拦截，实际返回 {ok!r}；"
        f"_last_tool_reject_reason={mw._last_tool_reject_reason!r}"
    )
    assert ok[1] and "my_secret_op" in ok[1], f"Bug3：拦截原因必须随返回值携带，实际 {ok!r}"
    assert mw._last_tool_reject_reason is not None
    assert "my_secret_op" in mw._last_tool_reject_reason


def test_s4_6_add_guardrail_after_late_injection_still_gets_registry():
    """A2：广播不能只发生在 setter 那一刻——先注入 tool_registry、后 add_guardrail 的
    权限护栏副本也要拿到它，否则 requires_approval=True 的高危工具被静默放行。

    本用例故意不挂 basic_permission 预设，让「晚挂载的那一个实例」成为唯一判定者，
    否则预设副本已被 setter 广播过、拦截会先发生，测不出这条死角。
    """
    from fast_agents.guardrails.preset import PromptInjectionGuardrail

    registry = ToolRegistry()
    registry.register_tool(_DangerousTool())  # dangerous_db_op: requires_approval=True

    # 时序：建 middleware（不挂任何权限护栏）→ 注入 registry → 再挂载权限护栏
    mw = GuardrailMiddleware(auto_register_presets=False)
    mw.tool_registry = registry
    external = BasicPermissionGuardrail(name="late_perm", blocked_tool_names=set())
    mw.add_guardrail(external)

    mounted = [g for g in mw.get_guardrails() if isinstance(g, BasicPermissionGuardrail)]
    assert len(mounted) == 1, f"应只挂载 1 个权限护栏，实际 {len(mounted)}"
    inst = mounted[0]
    assert inst.tool_registry is registry, (
        f"晚挂载的权限护栏必须补拿到 middleware 已有的 tool_registry，实际 {inst.tool_registry!r}"
    )
    # clone 隔离约定：注入只作用于副本，绝不回写传入的原型
    assert external.tool_registry is None, "不得把 registry 回写到用户传入的原型实例上"

    # 行为层：单一事实源分支必须真正生效。说明：传空集合会被当“未传”而回退默认危险表（已登记的 B1），
    # 故拦住 dangerous_db_op 的只能是 requires_approval 分支——它不在内置精确名单里。
    decision = mw.before_tool_call(tool_name="dangerous_db_op", tool_args={"sql": "drop table users"})
    assert not decision, (
        f"requires_approval=True 的工具应被拦截，实际放行 {decision!r}；"
        f"原因={mw._last_tool_reject_reason!r}"
    )
    assert "requires_approval" in decision[1], f"拦截原因应来自单一事实源，实际：{decision[1]!r}"

    # 实例自己显式带了 registry 时不得被覆盖（本次有意选择的非破坏规则）
    own = ToolRegistry()
    own.register_tool(_AnotherDangerousTool())
    mw.add_guardrail(BasicPermissionGuardrail(name="own_perm", blocked_tool_names=set(), tool_registry=own))
    own_inst = [g for g in mw.get_guardrails() if g.name == "own_perm"][0]
    assert own_inst.tool_registry is own, f"显式配置的 registry 不该被 middleware 覆盖：{own_inst.tool_registry!r}"

    # setter 既有语义不变（R1 守护）：后赋值仍是无条件广播，会覆盖已挂载实例（含自带 registry 的）
    registry_v2 = ToolRegistry()
    registry_v2.register_tool(_AnotherDangerousTool())
    mw.tool_registry = registry_v2
    assert [g for g in mw.get_guardrails() if g.name == "own_perm"][0].tool_registry is registry_v2, (
        "setter 的无条件广播语义不得因 A2 修复而改变"
    )

    # 零回归：middleware 自己没有 registry 时，新挂载实例不受影响，关键词兜底照旧
    mw2 = GuardrailMiddleware(auto_register_presets=False)
    mw2.add_guardrail(BasicPermissionGuardrail(name="no_reg_perm"))
    inst2 = [g for g in mw2.get_guardrails() if g.name == "no_reg_perm"][0]
    assert inst2.tool_registry is None, f"middleware 无 registry 时不得凭空注入：{inst2.tool_registry!r}"
    assert not mw2.before_tool_call(tool_name="delete_file", tool_args={}), "精确名单兜底不得回退"

    # 不误伤其它类型护栏：非权限护栏身上不该出现 tool_registry 属性
    mw3 = GuardrailMiddleware(auto_register_presets=False)
    mw3.tool_registry = registry
    mw3.add_guardrail(PromptInjectionGuardrail())
    inj = [g for g in mw3.get_guardrails() if g.name == "prompt_injection"][0]
    assert not hasattr(inj, "tool_registry"), "不得给非 BasicPermissionGuardrail 的实例塞私有字段"


# ==================== 入口（对齐 tests/guardrails/test_guardrails.py 风格） ====================

def main():
    """跑全部测试。"""
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """跑本轮新增的 S4 测试。"""
    test_s4_1_preset_tool_registry_is_none_when_agent_injects_late()
    test_s4_2_requires_approval_path_actually_blocks()
    test_s4_3_explicit_tool_registry_in_constructor_still_works()
    test_s4_4_setter_propagates_to_existing_presets_and_external_guards()
    test_s4_5_manual_tool_registry_assignment_before_run_blocks_in_agent_loop()
    test_s4_6_add_guardrail_after_late_injection_still_gets_registry()  # A2 晚挂载的权限护栏也要拿到 tool_registry


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
