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
        arguments={"sql": "drop table users"},
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
    info = ToolCallInfo(tool_name="dangerous_db_op", arguments={}, call_id="call_2")
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


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
