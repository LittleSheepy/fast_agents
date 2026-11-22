"""护栏基类 GuardrailBase

参考 ToolBase、LangChain BaseGuardrail、OpenAI Agents SDK GuardrailFunctionOutput 的共同设计：

命名约定（对齐项目现有规范）：
1. 统一抽象基类：GuardrailBase，子类必须实现 check 方法
2. 结构化返回值：check() 返回 GuardrailResult，明确是否触发拦截、原因说明
3. 同步/异步双入口：check / acheck（子类实现），invoke / ainvoke（带错误处理的公共入口）
4. 三种子类划分：InputGuardrail / OutputGuardrail / ToolCallGuardrail，分别对应三个检查阶段
5. 元数据支持：name / description / priority，用于注册和排序
6. 统一日志：通过 logger 输出拦截事件，对接 Trace 体系
7. hooks 机制：before_check / after_check / on_error
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import time as _time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from .errors import GuardrailConfigError, GuardrailError
from .models import GuardrailResult

logger = logging.getLogger(__name__)


def _normalize_to_text(input_data: Any) -> Any:
    """把多模态 List[ContentPart] 输入归一化为纯文本，供字符串类护栏安全处理。

    - str 原样返回；
    - list/List[ContentPart]（E8 多模态）调用 messages.parts.content_to_text 转成纯文本
      （图片自动转成「[图片]」占位符），避免 preset 里 ``if not isinstance(str): return allow()``
      直接放行多模态消息导致绕过；
    - dict（OpenAI message 风格 {"role":..., "content":...}）：取 content 字段递归归一化；
    - 其它类型（MsgToolCall、数字等）原样返回，由具体护栏自行判断。
    """
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        try:
            from ..messages.parts import content_to_text
            return content_to_text(input_data)
        except Exception:
            # content_to_text 不认识的 list 元素，退回 str()
            parts = []
            for item in input_data:
                if hasattr(item, "text"):
                    parts.append(getattr(item, "text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
    if isinstance(input_data, dict) and "content" in input_data:
        return _normalize_to_text(input_data.get("content", ""))
    return input_data


# 全局护栏原型注册表，和 ToolRegistry 风格保持一致。
#
# ⚠️ 隔离语义（对齐路线图 A34「global_registry 隔离」）：
#   这里存的是护栏「原型」（类级别默认实例或 @guardrail_creator 产生的样板实例），
#   仅供发现（list_guardrails）/查找（get_guardrail）/作为 add_guardrail 的克隆源。
#   GuardrailMiddleware 在 add_guardrail / _load_presets 时会对原型做 deepcopy，
#   产生本实例独占的副本；enable/disable/remove/配置修改都只改副本，绝不回写原型。
#   因此多 Agent 之间的护栏状态完全隔离，不会出现"A 禁用 B 也失效"的问题。
#
#   顶层 run_input_guardrails / run_output_guardrails / run_tool_call_guardrails
#   在 guardrails=None 时不再隐式扫全局表，直接返回空结果——不指定就不执行，
#   避免独立调用时意外跑了别的模块/测试注册的护栏。
_GLOBAL_GUARDRAIL_REGISTRY: Dict[str, "GuardrailBase"] = {}


class GuardrailBase(ABC):
    """护栏抽象基类。

    子类至少要设置 ``name`` / ``description`` 两个类属性，并实现 ``check()`` 方法。
    ``priority`` 控制执行顺序，数字越小优先级越高，默认 100。
    """

    name: str = ""
    description: str = ""
    priority: int = 100
    enabled: bool = True
    # 护栏自身异常时的姿态：默认 False（fail-closed，出错即拒）——安全组件的业界默认 posture。
    # 优先级（从高到低）：
    #   1. invoke(..., fail_open=X) 调用方显式传参（最高，覆盖所有默认）；
    #   2. 实例字段（挂载时由 GuardrailMiddleware 对未显式设置的实例注入统一姿态）；
    #   3. 类默认 False。
    # 单独使用时可在构造器显式设置 fail_open=True/False；挂载到 GuardrailMiddleware 后，
    # 未显式设置的实例会继承 middleware 的统一姿态，显式设置的保留用户值。
    fail_open: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
        fail_open: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name if name else type(self).name
        self.description = description if description else type(self).description
        self.priority = priority if priority is not None else type(self).priority
        self.enabled = enabled if enabled is not None else type(self).enabled
        # fail_open 解析优先级：invoke(..., fail_open=X) 显式传参 > 实例字段 > 类默认。
        # 构造时记「是否显式设置」标记，用于 GuardrailMiddleware 挂载时判断能否用统一姿态覆盖：
        # 未显式设置→继承 middleware 默认；显式设置→保留用户值，middleware 不覆盖。
        self.fail_open = fail_open if fail_open is not None else type(self).fail_open
        self._fail_open_explicit: bool = fail_open is not None
        self.tags: List[str] = list(tags) if tags is not None else list(type(self).tags if hasattr(type(self), 'tags') else [])
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else dict(type(self).metadata if hasattr(type(self), 'metadata') else {})

        if not self.name:
            raise GuardrailConfigError("护栏必须有 name")
        if not self.description:
            raise GuardrailConfigError(f"护栏 '{self.name}' 必须有 description")

    @abstractmethod
    def check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """同步检查（子类必须实现）。

        Args:
            input_data: 待检查的输入数据，根据护栏类型不同可能是：
                - InputGuardrail: 用户输入文本或消息列表
                - OutputGuardrail: Agent 输出文本或消息列表
                - ToolCallGuardrail: 工具调用信息（工具名、参数）
            context: 上下文信息，包含 agent_id、session_id、trace_id 等运行时信息

        Returns:
            GuardrailResult 检查结果
        """

    async def acheck(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """异步检查，默认在线程池中执行同步 check()，子类可覆盖实现真正异步逻辑。"""
        return await asyncio.to_thread(self._check_safe, input_data, context)

    def _check_safe(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        return self.check(input_data, context)

    # ── 公共入口（同步/异步都带错误处理、日志、Trace 打点）────────────────

    def invoke(self, input_data: Any, context: Optional[Dict[str, Any]] = None, *, fail_open: Optional[bool] = None) -> GuardrailResult:
        """执行护栏检查（同步入口）。

        Args:
            fail_open: 显式覆盖本次调用的 fail_open 姿态，优先级高于实例字段。
                None=使用实例字段（挂载时已含 GuardrailMiddleware 注入的统一姿态）；
                True=护栏自身出错时放行（fail-open）；
                False=出错即拒（fail-closed，安全组件业界默认 posture）。
        """
        if not self.enabled:
            return GuardrailResult.allow()

        start_time = _time.perf_counter()
        self._before_check(input_data, context)
        effective_fail_open = self.fail_open if fail_open is None else fail_open
        # L6：多模态输入归一化，避免 List[ContentPart] 直接绕过所有 isinstance(str) 检查
        check_input = _normalize_to_text(input_data)
        try:
            result = self.check(check_input, context)
        except GuardrailError as e:
            # 护栏作者显式抛出的拦截/配置异常——统一走 reject 路径打点，
            # 不再直接 raise 导致跳过 _after_check / duration_ms / 统计 / trace。
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            result = GuardrailResult(
                allowed=False,
                reason=str(e),
                metadata={
                    "duration_ms": duration_ms,
                    "guardrail_name": self.name,
                    "rejected_via_exception": True,
                    "exception_type": type(e).__name__,
                    **(e.details if hasattr(e, "details") and isinstance(getattr(e, "details", None), dict) else {}),
                },
            )
            logger.warning("[guardrail:%s] 拦截触发（抛异常）: %s", self.name, result.reason)
            self._after_check(input_data, context, result)
            # 标记为异常路径抛出，由聚合层（run_input_guardrails 等）转换成 XxxGuardrailReject
            result.metadata["_raise_reject"] = True
            return result
        except Exception as e:
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._on_error(e, input_data, context)
            err_meta = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": str(e),
                "guardrail_error_type": type(e).__name__,
                "is_error": True,
            }
            if effective_fail_open:
                logger.warning(
                    "[guardrail:%s] 检查异常（fail_open=True，放行）: %s: %s",
                    self.name, type(e).__name__, e,
                )
                result = GuardrailResult(allowed=True, metadata=err_meta)
            else:
                # fail-closed：护栏自身异常视为"拒绝"，返回 GuardrailResult(allowed=False)
                # 让中间件/上层可按统一 reject 链路打点，并由中间件自行决定是返回
                # _GuardrailBlocked 还是包装成 GuardrailError 上抛。
                logger.warning(
                    "[guardrail:%s] 检查异常（fail_open=False，拦截）: %s: %s",
                    self.name, type(e).__name__, e,
                )
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' 执行异常，fail_open=False 拦截请求：{type(e).__name__}: {e}",
                    metadata=err_meta,
                )
            self._after_check(input_data, context, result)
            return result

        duration_ms = int((_time.perf_counter() - start_time) * 1000)
        result.metadata["duration_ms"] = duration_ms
        result.metadata["guardrail_name"] = self.name

        if not result.allowed:
            logger.warning(
                "[guardrail:%s] 拦截触发: %s",
                self.name, result.reason,
            )
        else:
            logger.debug(
                "[guardrail:%s] 检查通过，耗时 %dms",
                self.name, duration_ms,
            )

        self._after_check(input_data, context, result)
        return result

    async def ainvoke(self, input_data: Any, context: Optional[Dict[str, Any]] = None, *, fail_open: Optional[bool] = None) -> GuardrailResult:
        """执行护栏检查（异步入口）。

        fail_open 语义与 invoke 一致：显式传参最高优先，None 时使用实例字段。
        """
        if not self.enabled:
            return GuardrailResult.allow()

        start_time = _time.perf_counter()
        self._before_check(input_data, context)
        effective_fail_open = self.fail_open if fail_open is None else fail_open
        check_input = _normalize_to_text(input_data)
        try:
            result = await self.acheck(check_input, context)
        except GuardrailError as e:
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            result = GuardrailResult(
                allowed=False,
                reason=str(e),
                metadata={
                    "duration_ms": duration_ms,
                    "guardrail_name": self.name,
                    "rejected_via_exception": True,
                    "exception_type": type(e).__name__,
                    **(e.details if hasattr(e, "details") and isinstance(getattr(e, "details", None), dict) else {}),
                },
            )
            logger.warning("[guardrail:%s] 拦截触发（抛异常，异步）: %s", self.name, result.reason)
            self._after_check(input_data, context, result)
            result.metadata["_raise_reject"] = True
            return result
        except Exception as e:
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._on_error(e, input_data, context)
            err_meta = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": str(e),
                "guardrail_error_type": type(e).__name__,
                "is_error": True,
            }
            if effective_fail_open:
                logger.warning(
                    "[guardrail:%s] 异步检查异常（fail_open=True，放行）: %s: %s",
                    self.name, type(e).__name__, e,
                )
                result = GuardrailResult(allowed=True, metadata=err_meta)
            else:
                logger.warning(
                    "[guardrail:%s] 异步检查异常（fail_open=False，拦截）: %s: %s",
                    self.name, type(e).__name__, e,
                )
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' 执行异常，fail_open=False 拦截请求：{type(e).__name__}: {e}",
                    metadata=err_meta,
                )
            self._after_check(input_data, context, result)
            return result

        duration_ms = int((_time.perf_counter() - start_time) * 1000)
        result.metadata["duration_ms"] = duration_ms
        result.metadata["guardrail_name"] = self.name

        if not result.allowed:
            logger.warning(
                "[guardrail:%s] 拦截触发: %s",
                self.name, result.reason,
            )
        else:
            logger.debug(
                "[guardrail:%s] 检查通过，耗时 %dms",
                self.name, duration_ms,
            )

        self._after_check(input_data, context, result)
        return result

    # ── 内部钩子 ─────────────────────────────────────────────

    def _before_check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> None:
        pass

    def _after_check(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]],
        result: GuardrailResult,
    ) -> None:
        pass

    def _on_error(
        self,
        error: Exception,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    # ── 注册与序列化 ───────────────────────────────────────────────

    def register(self) -> None:
        """把当前实例作为「原型」登记到全局表。

        注意：原型仅供查找/发现/克隆，GuardrailMiddleware 不会直接运行它，
        而是在 add_guardrail / _load_presets 时调用 clone() 产生实例副本。
        """
        if self.name in _GLOBAL_GUARDRAIL_REGISTRY:
            logger.warning("[guardrail] 覆盖已注册的同名护栏原型: %s", self.name)
        _GLOBAL_GUARDRAIL_REGISTRY[self.name] = self

    def clone(self) -> "GuardrailBase":
        """克隆一份独立副本，供多实例隔离使用。

        默认用 deepcopy 深拷贝全部属性；如果子类持有不可深拷贝的资源
        （如编译好的正则、外部连接句柄等），应覆盖此方法做定制拷贝。
        """
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于 API 展示和 Trace 上报"""
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "enabled": self.enabled,
            "fail_open": self.fail_open,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"Guardrail(name={self.name}, enabled={self.enabled})"

    def __repr__(self) -> str:
        return f"<Guardrail '{self.name}' priority={self.priority}>"


def guardrail_creator(
    name: Optional[str] = None,
    description: Optional[str] = None,
    priority: int = 100,
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., GuardrailResult]], GuardrailBase]:
    """护栏创建装饰器，对齐 @tool_creator 的使用风格。

    可以直接装饰一个普通函数，自动转换为 GuardrailBase 子类实例。

    Example:
        @guardrail_creator(name="no_injection", description="检测prompt注入")
        def check_injection(input_data: str, context=None) -> GuardrailResult:
            if "忽略之前的指令" in input_data:
                return GuardrailResult.reject("检测到提示注入尝试")
            return GuardrailResult.allow()
    """
    def decorator(func: Callable[..., GuardrailResult]) -> GuardrailBase:
        # 从函数签名和docstring自动推断信息
        _name = name or func.__name__
        _description = description or inspect.getdoc(func) or f"Guardrail: {_name}"

        class _FunctionGuardrail(GuardrailBase):
            def __init__(self):
                super().__init__(
                    name=_name,
                    description=_description,
                    priority=priority,
                    tags=tags or [],
                )
                self.func = func

            def check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
                return self.func(input_data, context)

            async def acheck(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
                if asyncio.iscoroutinefunction(self.func):
                    return await self.func(input_data, context)
                return await super().acheck(input_data, context)

        instance = _FunctionGuardrail()
        instance.register()
        return instance

    return decorator


def list_guardrails() -> List[GuardrailBase]:
    """列出所有已注册的护栏「原型」快照。

    注意：返回的对象是原型，仅供发现/展示/作为克隆源使用，
    不要直接对其执行 enable/disable 期待影响运行时——
    真正运行时由 GuardrailMiddleware 通过 clone() 产生独立副本。
    """
    return sorted(_GLOBAL_GUARDRAIL_REGISTRY.values(), key=lambda g: g.priority)


def get_guardrail(name: str) -> Optional[GuardrailBase]:
    """根据名称获取护栏「原型」，用于 add_guardrail 前的查找；
    中间件拿到后会自动 clone() 一份实例副本，不会直接运行原型。"""
    return _GLOBAL_GUARDRAIL_REGISTRY.get(name)
