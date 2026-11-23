"""护栏基类 GuardrailBase

参考 ToolBase、LangChain BaseGuardrail、OpenAI Agents SDK GuardrailFunctionOutput 的共同设计：

命名约定（对齐项目现有规范）：
1. 统一抽象基类：GuardrailBase，子类必须实现 check 方法
2. 结构化返回值：check() 返回 GuardrailResult，明确是否触发拦截、原因说明
3. 同步/异步双入口：check / acheck（子类实现），invoke / ainvoke（带错误处理的公共入口）
4. 三种子类划分：InputGuardrail / OutputGuardrail / ToolCallGuardrail，分别对应三个检查阶段
5. 元数据支持：name / description / priority / enabled / fail_open / tags / metadata，用于注册和排序
   （Bug6：全部在 GuardrailBase 类体显式声明默认值，子类只覆盖自己需要的即可）
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
from .guardrail_registry import GuardrailRegistry, global_guardrail_registry

logger = logging.getLogger(__name__)


def to_scan_text(input_data: Any) -> str:
    """把任意输入归一化为「可扫描纯文本」，供内容类护栏（Input/Output 系）正则匹配。

    C2 修复背景：invoke() 的 content_to_text 只负责消息形态输入
    （str / None / List[ContentPart] / 含 ``content`` 键的 dict）转文本，
    其余结构化输入（普通 dict / list / 自定义对象）会原样透传给 check()；
    此前预设内容护栏以 ``if not isinstance(input_data, str): return allow()``
    开头，导致 {"query": "如何制造炸弹"} 这类载荷整体绕过扫描。

    规则：
    - str 原样返回（既有文本路径逐字节不变）；
    - None → 空串；
    - dict：递归收集字符串 key 与全部 value；list/tuple/set/frozenset：逐项递归；
    - 其他对象：降级 str(obj)，保证不存在「零扫描盲区」。
    只做扁平化文本拼接、不解释结构语义。
    ⚠️ Bug5 纠正：_normalize_input=True（Input/Output 系默认）时，invoke() 传入 check() 的
    是本函数扁平化后的字符串，并非原始 input_data；需要结构化解析的自定义护栏应设
    _normalize_input=False（如 ToolCallGuardrail），此时 check() 收到未经扁平化的原对象。
    """
    if isinstance(input_data, str):
        return input_data
    if input_data is None:
        return ""
    if isinstance(input_data, dict):
        parts: List[str] = []
        for key, value in input_data.items():
            if isinstance(key, str):
                parts.append(key)
            parts.append(to_scan_text(value))
        return " ".join(p for p in parts if p)
    if isinstance(input_data, (list, tuple, set, frozenset)):
        return " ".join(p for p in (to_scan_text(item) for item in input_data) if p)
    return str(input_data)


class GuardrailBase(ABC):
    """护栏抽象基类。

    子类至少要设置 ``name`` / ``description`` 两个类属性，并实现 ``check()`` 方法。
    ``priority`` 控制执行顺序，数字越小优先级越高，默认 100。
    ``tags`` / ``metadata`` 同样支持类级声明（如 ``tags = ["audit"]``）或构造器传参，
    两者都会被拷贝成实例属性，修改一个实例不会影响其它实例或类默认值。
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
    # Bug6 修复：tags / metadata 与 name/description/priority/enabled/fail_open 一样在类体显式声明。
    # 此前基类没这两个类属性，__init__ 只能靠 `hasattr(type(self), 'tags')` 隔空探测子类有没有写，
    # 带来两个实际问题：
    #   1) 子类覆盖 __init__ 又漏调 super().__init__() 时，priority/enabled/fail_open 能从类默认值
    #      兜住，而 tags/metadata 直接 AttributeError（to_dict() / clone() 立即报错）；
    #   2) 本包发布 `py.typed`，字段契约不在类上声明，静态检查与 IDE 补全都看不到「护栏可以有 tags」。
    # 注意：类级值只是共享的「兜底常量」，__init__ 一律 list()/dict() 拷贝成实例属性，
    # 因此实例各自 append/update 不会互相污染，也不会改到这里的类默认值。
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    # Bug1 修复：是否在 invoke/ainvoke 中自动将 input_data 文本化（to_scan_text）。
    # InputGuardrail / OutputGuardrail 等基于正则匹配的护栏需要 True（默认），
    # 确保 dict/list 等结构化输入被全量扁平化，不因 content_to_text 的 "content" 提取而丢失字段。
    # ToolCallGuardrail 需要 False —— 它依赖原始 pydantic 对象（ToolCallInfo/ToolResultInfo）的 isinstance 分发。
    _normalize_input: bool = True

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
        # Bug6 修复：类上已声明 tags/metadata 默认值，无需再 `hasattr(type(self), ...)` 兜底；
        # 依旧做 list()/dict() 浅拷贝，保证多个实例之间、实例与类默认值之间互不串扰。
        self.tags: List[str] = list(tags) if tags is not None else list(type(self).tags)
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else dict(type(self).metadata)

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
        self._safe_before_check(input_data, context)
        effective_fail_open = self.fail_open if fail_open is None else fail_open
        # Bug1 修复：用 to_scan_text 全量扁平化替代 content_to_text，
        # 避免 {"content": "安全", "query": "恶意"} 中非 content 字段绕过扫描。
        # to_scan_text 对 str 是恒等变换，不改变既有文本路径行为。
        # ToolCallGuardrail 设置 _normalize_input=False 跳过（它需要原始 pydantic 对象做 isinstance 分发）。
        check_input = to_scan_text(input_data) if self._normalize_input else input_data
        try:
            result = self.check(check_input, context)
        except GuardrailConfigError as e:
            # P5 修复：配置错误（如 allow_modify=False 却调用 modify()）不是「内容命中」，
            # 不能和「护栏作者主动抛的拦截异常」共用一条 reject 通道——否则开发者把护栏配错，
            # 表现却像用户内容被安全拦截，既误导终端用户、又让配置 bug 无法在 error 通道暴露。
            # 这里按「组件故障」观测：is_error=True，并尊重 fail_open 姿态（放行/拦截由姿态决定）。
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._safe_on_error(e, input_data, context)
            cfg_meta = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": str(e),
                "guardrail_error_type": type(e).__name__,
                "is_error": True,
                "config_error": True,
            }
            if effective_fail_open:
                logger.warning(
                    "[guardrail:%s] 配置错误（fail_open=True，放行）: %s", self.name, e,
                )
                result = GuardrailResult(allowed=True, metadata=cfg_meta)
            else:
                logger.error(
                    "[guardrail:%s] 配置错误（fail_open=False，拦截）: %s", self.name, e,
                )
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' 配置错误，fail_open=False 拦截请求：{e}",
                    metadata=cfg_meta,
                )
            self._safe_after_check(input_data, context, result)
            return result
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
            self._safe_after_check(input_data, context, result)
            # Bug4 修复：删除无人消费的控制流标记 `_raise_reject`。聚合层（run_input_guardrails 等）
            # 只看 `not result.allowed` 就转成 XxxGuardrailReject，从来不读这个标记；
            # 留着会让人误以为「异常式拦截」与「返回式拦截」是两条链路、要靠该标记才能往下传导。
            # 拦截来源已由上方 metadata 的 rejected_via_exception / exception_type 表达（仅诊断用），
            # 并经 e.details 透传给上层回调，不参与任何控制流判定。
            return result
        except Exception as e:
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._safe_on_error(e, input_data, context)
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
            self._safe_after_check(input_data, context, result)
            return result

        # Bug1 修复：对 check() 返回值做防御性类型守卫。
        # 子类若忘记 return（隐式返回 None）或误返回其他类型，此前会直接
        # AttributeError 崩溃 invoke() 本身——该崩溃发生在 try/except 块**外**，
        # 不受任何 except 兜底，违反 R6「观测/检查代码不得打挂主流程」。
        # 这里把它降级为「组件故障」路径，尊重 effective_fail_open 姿态决定放行或拦截。
        if not isinstance(result, GuardrailResult):
            _invalid_msg = (
                f"护栏 '{self.name}' 的 check() 返回了非法类型 {type(result).__name__}"
                f"（期望 GuardrailResult），视为组件异常"
            )
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            inv_meta: Dict[str, Any] = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": _invalid_msg,
                "guardrail_error_type": "InvalidCheckReturn",
                "is_error": True,
            }
            if effective_fail_open:
                logger.warning("[guardrail:%s] %s（fail_open=True，放行）", self.name, _invalid_msg)
                result = GuardrailResult(allowed=True, metadata=inv_meta)
            else:
                logger.error("[guardrail:%s] %s（fail_open=False，拦截）", self.name, _invalid_msg)
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' check() 返回类型非法，fail_open=False 拦截请求",
                    metadata=inv_meta,
                )
            self._safe_after_check(input_data, context, result)
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

        self._safe_after_check(input_data, context, result)
        return result

    async def ainvoke(self, input_data: Any, context: Optional[Dict[str, Any]] = None, *, fail_open: Optional[bool] = None) -> GuardrailResult:
        """执行护栏检查（异步入口）。

        fail_open 语义与 invoke 一致：显式传参最高优先，None 时使用实例字段。
        """
        if not self.enabled:
            return GuardrailResult.allow()

        start_time = _time.perf_counter()
        self._safe_before_check(input_data, context)
        effective_fail_open = self.fail_open if fail_open is None else fail_open
        # Bug1 修复：异步入口同样使用 to_scan_text 替代 content_to_text
        check_input = to_scan_text(input_data) if self._normalize_input else input_data
        try:
            result = await self.acheck(check_input, context)
        except GuardrailConfigError as e:
            # P5 修复：异步入口与同步 invoke 语义一致——配置错误按组件故障观测，
            # is_error=True 且尊重 fail_open 姿态，不再伪装成内容命中。
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._safe_on_error(e, input_data, context)
            cfg_meta = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": str(e),
                "guardrail_error_type": type(e).__name__,
                "is_error": True,
                "config_error": True,
            }
            if effective_fail_open:
                logger.warning(
                    "[guardrail:%s] 异步配置错误（fail_open=True，放行）: %s", self.name, e,
                )
                result = GuardrailResult(allowed=True, metadata=cfg_meta)
            else:
                logger.error(
                    "[guardrail:%s] 异步配置错误（fail_open=False，拦截）: %s", self.name, e,
                )
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' 配置错误，fail_open=False 拦截请求：{e}",
                    metadata=cfg_meta,
                )
            self._safe_after_check(input_data, context, result)
            return result
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
            self._safe_after_check(input_data, context, result)
            # Bug4 修复：异步入口同样不再写入无人消费的 `_raise_reject` 标记，语义与同步 invoke 一致。
            return result
        except Exception as e:
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            self._safe_on_error(e, input_data, context)
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
            self._safe_after_check(input_data, context, result)
            return result

        # Bug1 修复（异步入口）：与同步 invoke 语义完全一致——
        # acheck() 返回非法类型时视为组件故障，不再裸 AttributeError 崩溃 ainvoke。
        if not isinstance(result, GuardrailResult):
            _invalid_msg = (
                f"护栏 '{self.name}' 的 acheck() 返回了非法类型 {type(result).__name__}"
                f"（期望 GuardrailResult），视为组件异常"
            )
            duration_ms = int((_time.perf_counter() - start_time) * 1000)
            inv_meta: Dict[str, Any] = {
                "duration_ms": duration_ms,
                "guardrail_name": self.name,
                "guardrail_error": _invalid_msg,
                "guardrail_error_type": "InvalidCheckReturn",
                "is_error": True,
            }
            if effective_fail_open:
                logger.warning("[guardrail:%s] %s（fail_open=True，放行）", self.name, _invalid_msg)
                result = GuardrailResult(allowed=True, metadata=inv_meta)
            else:
                logger.error("[guardrail:%s] %s（fail_open=False，拦截）", self.name, _invalid_msg)
                result = GuardrailResult(
                    allowed=False,
                    reason=f"护栏 '{self.name}' acheck() 返回类型非法，fail_open=False 拦截请求",
                    metadata=inv_meta,
                )
            self._safe_after_check(input_data, context, result)
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

        self._safe_after_check(input_data, context, result)
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

    # ── Bug2/3 修复：钩子安全包装层 ────────────────────────────────────
    # 三个钩子 _before_check / _after_check / _on_error 是子类可重写的观测点，
    # 其调用原本不在 try/except 保护范围内，若子类重写实现抛异常，
    # 会打挂 invoke()/ainvoke() 本身（违反 R6）。
    # 修复后所有钩子调用走 _safe_xxx 包装，失败仅打 warning，绝不阻断主流程。
    # _safe_before_check 失败时跳过钩子继续执行 check()；
    # _safe_after_check 失败时跳过钩子仍然 return result（拦截/放行结果不受影响）；
    # _safe_on_error 失败时跳过钩子继续走 fail_open/fail_closed 逻辑。

    def _safe_before_check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._before_check(input_data, context)
        except Exception as e:
            logger.warning(
                "[guardrail:%s] _before_check 钩子异常（已忽略，不影响主流程）: %s: %s",
                self.name, type(e).__name__, e,
            )

    def _safe_after_check(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]],
        result: GuardrailResult,
    ) -> None:
        try:
            self._after_check(input_data, context, result)
        except Exception as e:
            logger.warning(
                "[guardrail:%s] _after_check 钩子异常（已忽略，不影响拦截结果）: %s: %s",
                self.name, type(e).__name__, e,
            )

    def _safe_on_error(
        self,
        error: Exception,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self._on_error(error, input_data, context)
        except Exception as e:
            logger.warning(
                "[guardrail:%s] _on_error 钩子异常（已忽略，不影响 fail_open 判定）: %s: %s",
                self.name, type(e).__name__, e,
            )

    # ── 注册与序列化 ───────────────────────────────────────────────

    def register(self) -> None:
        """把当前实例作为「原型」登记到全局表。

        注意：原型仅供查找/发现/克隆，GuardrailMiddleware 不会直接运行它，
        而是在 add_guardrail / _load_presets 时调用 clone() 产生实例副本。
        """
        global_guardrail_registry.register_guardrail(self)

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
    guardrail_type: str = "input",
) -> Callable[[Callable[..., GuardrailResult]], GuardrailBase]:
    """护栏创建装饰器，对齐 @tool_creator 的使用风格。

    可以直接装饰一个普通函数，自动转换为**对应类型**的护栏子类实例并注册到全局表。

    P1 修复背景：此前生成的 _FunctionGuardrail 只继承 GuardrailBase，不属于
    InputGuardrail / OutputGuardrail / ToolCallGuardrail 任一子类，会被
    GuardrailMiddleware 的类型分派与 use_global=True 的 isinstance 扫描静默剔除
    ——「注册成功但永远不执行」，与本函数 docstring 承诺的“自动注册为输入护栏”矛盾。
    现在通过 guardrail_type 参数决定继承哪个基类，让函数式护栏真正参与检查。

    Args:
        name / description / priority / tags: 语义不变；缺省时从函数名与 docstring 推断。
        guardrail_type: "input"（默认）/ "output" / "tool_call"，决定生成的护栏继承的基类、
            进而决定参与哪个阶段的检查；非法值在装饰器**创建时**即抛 GuardrailConfigError。
            被装饰函数签名统一为 func(input_data, context) -> GuardrailResult：
            - input / output：input_data 为待检查的用户输入 / Agent 最终输出；
            - tool_call：input_data 为 ToolCallInfo（before_call）或 ToolResultInfo（after_call），
              沿用 ToolCallGuardrail.check 的类型分发，后置检查仍受 check_result 开关约束
              （默认 False 只跑前置；需要检查工具结果时显式置 instance.check_result = True）。

    Example:
        @guardrail_creator(name="no_injection", description="检测prompt注入")
        def check_injection(input_data: str, context=None) -> GuardrailResult:
            if "忽略之前的指令" in input_data:
                return GuardrailResult.reject("检测到提示注入尝试")
            return GuardrailResult.allow()

        @guardrail_creator(name="tool_pre_check", guardrail_type="tool_call")
        def check_tool(tool_call, context=None) -> GuardrailResult:
            if tool_call.tool_name == "shell_exec":
                return GuardrailResult.reject("禁止执行 shell")
            return GuardrailResult.allow()

    ⚠️ 装饰 GuardrailBase 子类（class）时原样返回、不做包装（P1 同步修复：此前会被
    误包装成 _FunctionGuardrail，把类当函数调用导致每次检查都报配置错误），
    护栏类型由类自身的继承决定，此时 guardrail_type 参数不生效（显式传入会告警）。
    """
    # D3：非法类型值早失败——装饰器创建时即抛错，而不是等到运行时被兜底伪装成拦截
    if guardrail_type not in ("input", "output", "tool_call"):
        raise GuardrailConfigError(
            f"guardrail_type 必须是 'input' / 'output' / 'tool_call' 之一，收到：{guardrail_type!r}"
        )

    def decorator(func: Callable[..., GuardrailResult]) -> GuardrailBase:
        # 延迟 import：三个子类模块均依赖本模块，顶层 import 会形成循环导入
        from .input_guardrail import InputGuardrail
        from .output_guardrail import OutputGuardrail
        from .tool_call_guardrail import ToolCallGuardrail

        # P1：传入的已经是 Guardrail 类则原样返回，类型由类的继承决定，不做函数式包装
        if isinstance(func, type) and issubclass(func, GuardrailBase):
            if guardrail_type != "input":
                logger.warning(
                    "@guardrail_creator 装饰类 '%s' 时 guardrail_type=%r 不生效，护栏类型由类自身的继承决定",
                    func.__name__, guardrail_type,
                )
            return func

        # 从函数签名和docstring自动推断信息
        _name = name or func.__name__
        _description = description or inspect.getdoc(func) or f"Guardrail: {_name}"

        if guardrail_type == "tool_call":
            # 不覆盖 check()，沿用 ToolCallGuardrail.check 按输入类型的 before/after 分发，
            # 两个钩子都委托给同一个函数，由用户在函数内自行判断被检查对象
            class _FunctionGuardrail(ToolCallGuardrail):
                def __init__(self):
                    super().__init__(
                        name=_name,
                        description=_description,
                        priority=priority,
                        tags=tags or [],
                    )
                    self.func = func

                def before_call(self, tool_call, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
                    return self.func(tool_call, context)

                def after_call(self, tool_result, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
                    return self.func(tool_result, context)
        else:
            _base_cls = OutputGuardrail if guardrail_type == "output" else InputGuardrail

            class _FunctionGuardrail(_base_cls):
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
    return global_guardrail_registry.list_guardrails()


def get_guardrail(name: str) -> Optional[GuardrailBase]:
    """根据名称获取护栏「原型」，用于 add_guardrail 前的查找；
    中间件拿到后会自动 clone() 一份实例副本，不会直接运行原型。"""
    return global_guardrail_registry.get_guardrail(name)


def clear_guardrails() -> None:
    """清空全局原型注册表（测试隔离常用）。"""
    global_guardrail_registry.clear_guardrails()


def get_global_guardrail_registry() -> "GuardrailRegistry":
    """返回全局护栏原型注册表对象（GuardrailRegistry 单例）。"""
    return global_guardrail_registry


def update_global_guardrail_registry(reg) -> None:
    """用一份 {name: 原型} 映射批量覆盖全局注册表。"""
    global_guardrail_registry.update_guardrails(reg)


# ── 向后兼容别名 ──────────────────────────────────────────────
# 历史上全局原型表是模块级裸 dict（_GLOBAL_GUARDRAIL_REGISTRY），
# 现已抽取为 GuardrailRegistry 单例 global_guardrail_registry。
# 保留旧名指向同一对象，兼容存量代码：示例 / 测试里的 .clear() / .copy() / .update() /
# .values() 与 `name in ...` 判定，均由 GuardrailRegistry 的 dict-like 兼容层支持。
_GLOBAL_GUARDRAIL_REGISTRY = global_guardrail_registry




