"""
Guardrails 模块异常体系（基类 + 具体子类）。

参考同仓库规范：
- llm/errors.py、skills/errors.py——「基类 + 具体子类 + 集中放在模块自己的 errors.py」
- 所有子类都继承自 GuardrailError，上层统一用 except GuardrailError 捕获
- 异常保留触发的护栏名、拦截原因等上下文信息，便于上层定位
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class GuardrailError(Exception):
    """Guardrails 模块所有异常的基类。

    继承 Exception（不挂 ValueError）：护栏拦截属于安全策略判定，
    语义上不是「值不合法」，独立谱系可避免与无关的 except ValueError 相互误捕。
    """

    def __init__(
        self,
        message: str,
        *,
        guardrail_name: str = "",
        details: Optional[Dict[str, Any]] = None,
        results: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.guardrail_name = guardrail_name
        self.details = details or {}
        # S5: 附加 reject 抛出之前已跑完的护栏结果，供上层补记统计与 trace，
        # 避免 _pass_stats / span 因局部变量丢失而残缺。
        self.results: List[Any] = results or []


class InputGuardrailReject(GuardrailError):
    """输入护栏拦截：用户输入触发安全规则，拒绝处理。"""
    pass


class OutputGuardrailReject(GuardrailError):
    """输出护栏拦截：Agent 输出触发安全规则，拦截并返回提示。"""
    pass


class ToolCallGuardrailReject(GuardrailError):
    """工具调用护栏拦截：工具入参/动作触发安全规则，禁止执行。"""
    pass


class GuardrailConfigError(GuardrailError):
    """护栏配置错误：参数缺失、规则格式非法、依赖未安装等配置类问题。"""
    pass
