"""
预设包初始化，导出所有开箱即用护栏

- BasicInputSafetyGuardrail：输入侧关键词安全（v1.0.0+ 拆分为独立类）
- BasicOutputSafetyGuardrail：输出侧关键词安全
- BasicSafetyGuardrail：向后兼容工厂函数，返回 input+output 双实例列表
- PromptInjectionGuardrail：Prompt 注入检测
- BasicPermissionGuardrail：权限护栏（L9 对齐单一事实源，优先读 ToolRegistry.requires_approval）
"""
from .safety_guardrail import (
    BasicSafetyGuardrail,
    BasicInputSafetyGuardrail,
    BasicOutputSafetyGuardrail,
)
from .injection_guardrail import PromptInjectionGuardrail
from .permission_guardrail import BasicPermissionGuardrail

__all__ = [
    "BasicSafetyGuardrail",
    "BasicInputSafetyGuardrail",
    "BasicOutputSafetyGuardrail",
    "PromptInjectionGuardrail",
    "BasicPermissionGuardrail",
]
