"""
Guardrail中间件：将安全护栏系统集成到Agent请求流程
在用户输入、工具调用、Agent输出三个节点自动挂载护栏检查，零侵入接入现有执行流

设计要点（对齐 AGENTS.md §5.6 / R3 单一事实源）：
- 继承 ``AgentMiddleware`` ABC，钩子签名统一，避免与 SkillMiddleware 同名钩子签名冲突；
- before_tool_call / after_tool_call / after_agent_run 的返回值语义与 AgentMiddleware
  契约完全一致（Blocked/Modified/None/True/False/str）；
- **唯一打点入口**复用 ``guardrail_trace.log_guardrail_event``，Span 使用
  ``SpanKind.GUARDRAIL`` 专用种类，不再在 middleware 内联写 start_span/end_span，
  消除 L8 字段漂移；
- 工具危险判定默认读 ``ToolRegistry`` 的 ``metadata["requires_approval"]``
  单一事实源（对齐路线图 A9/C6），预设 ``BasicPermissionGuardrail`` 也遵循该约定；
- 默认 fail_open=False（fail-closed，安全组件业界默认 posture）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from ..agents.middleware import AgentMiddleware
from .guardrail_base import GuardrailBase, _GLOBAL_GUARDRAIL_REGISTRY, _normalize_to_text
from .models import GuardrailResult, ToolCallInfo, ToolResultInfo
from .input_guardrail import InputGuardrail, run_input_guardrails, arun_input_guardrails
from .output_guardrail import OutputGuardrail, run_output_guardrails, arun_output_guardrails
from .tool_call_guardrail import (
    ToolCallGuardrail, ToolCallInfo, ToolResultInfo,
    run_tool_call_guardrails, run_tool_result_guardrails,
    arun_tool_call_guardrails, arun_tool_result_guardrails,
)
from .errors import InputGuardrailReject, OutputGuardrailReject, ToolCallGuardrailReject
from .guardrail_trace import log_guardrail_event

_logger = logging.getLogger(__name__)


@dataclass
class _GuardrailBlocked:
    """护栏拦截信号：phase=input/output，message 为给用户的提示文本。

    中间件通过返回此结构化对象告诉 ReActAgent「这一轮被拦截了」，
    不再依赖对返回字符串做中文子串匹配，避免假阳性/假阴性/死条件。

    ``reason_kind`` 用于区分两类拦截来源，供上层观测/调试：
    - ``"rule"``：规则正常命中（默认），护栏按预期拦截；
    - ``"error"``：护栏自身异常（fail-closed），属"组件故障被硬短路"，
      此时 ``detail`` 携带原始异常信息，仅打日志/打点，不回显给终端用户。
    """

    phase: str
    message: str
    reason_kind: str = "rule"
    detail: str = ""


@dataclass
class _GuardrailModified:
    """护栏修改信号：phase=output 时携带修改后的最终文本（脱敏/改写）。"""

    phase: str
    content: str


class GuardrailMiddleware(AgentMiddleware):
    """
    Guardrail中间件：在Agent全流程自动执行护栏检查

    核心功能：
    1. before_agent_run：用户输入进入推理前执行输入护栏检查
    2. before_tool_call：工具调用前执行工具前置护栏（高危操作拦截、参数校验）
    3. after_tool_call：工具返回后执行工具后置护栏（敏感数据脱敏、结果过滤）
    4. after_agent_run：Agent最终输出前执行输出护栏（输出内容安全、敏感信息脱敏）
    5. 全链路Trace打点，统一 SpanKind.GUARDRAIL，支持拦截率/误判率统计
    """

    def __init__(
        self,
        guardrails: Optional[List[GuardrailBase]] = None,
        presets: Optional[List[str]] = None,
        auto_register_presets: bool = True,
        enable_presets: Optional[bool] = None,
        on_input_reject: Optional[Callable[[InputGuardrailReject], str]] = None,
        on_output_reject: Optional[Callable[[OutputGuardrailReject], str]] = None,
        on_tool_reject: Optional[Callable[[ToolCallGuardrailReject], str]] = None,
        tracer: Any = None,
        fail_open: bool = False,
        tool_registry: Any = None,
    ):
        """
        Args:
            guardrails: 用户显式注册的护栏实例列表；传入后会自动 clone() 一份副本，
                避免多 Agent 共享同一实例导致 enable/disable 串扰。
            presets: 要加载的预设护栏名称列表，None 表示加载全部默认预设。
            auto_register_presets: 是否加载默认预设护栏（默认 True）。
                ``enable_presets`` 是向后兼容的别名。
            enable_presets: 向后兼容参数；非 None 时覆盖 auto_register_presets。
            on_input_reject: 输入拦截回调，返回给用户的提示文本。
            on_output_reject: 输出拦截回调。
            on_tool_reject: 工具调用拦截回调。
            tracer: TraceLogger 实例，用于全链路打点；None 则零开销不打点。
            fail_open: 护栏自身抛异常时的姿态，默认 False（fail-closed，出错即拒）；
                传 True 表示容错放行（fail-open）。
            tool_registry: ``ToolRegistry`` 实例。传入后，权限护栏会优先读取
                ``tool.metadata["requires_approval"]`` 这一单一事实源（对齐路线图
                A9 审批 / C6 沙箱），而不是自造一份危险关键词清单；
                None 时回退到预设关键词表（向后兼容）。
        """
        super().__init__()
        self.tracer = tracer
        self.on_input_reject = on_input_reject
        self.on_output_reject = on_output_reject
        self.on_tool_reject = on_tool_reject
        self.fail_open = fail_open
        # 私有字段：通过 property setter 统一注入，setter 会把 tool_registry 广播到
        # 所有已挂载的 BasicPermissionGuardrail 实例（含 preset 副本），解决 ReActAgent
        # 晚注入（react_agent.py:261-263 m.tool_registry = self.tool_registry）时，preset
        # 构造期拿到 None 后未同步更新的问题（S4）。
        self._tool_registry: Any = None
        # 兼容 enable_presets 别名
        if enable_presets is not None:
            auto_register_presets = enable_presets
        # 实例级护栏副本列表（隔离语义：多 Agent 之间互不影响）
        self._instance_guardrails: List[GuardrailBase] = []
        # 拦截/通过统计，支持观测/监控接入（按护栏名计数）；
        # ``stats["registered"]/stats["reject"]/stats["pass"]`` 作为总览字段。
        self._reject_stats: Dict[str, int] = {}
        self._pass_stats: Dict[str, int] = {}
        self._pass_count: int = 0
        # 最近一次工具拦截原因，供 Agent 把「拒绝原因」作为 observation 喂回 LLM
        self._last_tool_reject_reason: Optional[str] = None

        # 加载显式传入的护栏（clone 副本）
        if guardrails:
            for g in guardrails:
                self.add_guardrail(g)

        # 加载预设护栏（此时 _tool_registry 仍是 None；预设构造不依赖 registry，
        # 后续 setter 会把 registry 广播到所有已挂载的 BasicPermissionGuardrail 实例）
        if auto_register_presets:
            self._load_presets(presets)

        # 构造器显式传入 tool_registry 时，通过 setter 统一广播（必须在 _load_presets 之后）
        if tool_registry is not None:
            self.tool_registry = tool_registry

    # ==================== tool_registry 晚注入传播（S4） ====================

    @property
    def tool_registry(self) -> Any:
        """当前绑定的 ToolRegistry 实例。"""
        return self._tool_registry

    @tool_registry.setter
    def tool_registry(self, value: Any) -> None:
        """赋值 tool_registry 时，同步广播到所有已挂载的 BasicPermissionGuardrail 实例
        （含 preset 副本与外部 add_guardrail 注册的实例）。

        解决 ReActAgent 构造 middleware 之后才把 self.tool_registry 注入（react_agent.py:261-263），
        导致 preset 副本里 tool_registry=None、单一事实源路径永远不触发的问题。
        """
        from .preset import BasicPermissionGuardrail as _BPG
        self._tool_registry = value
        # 广播到已挂载实例（含 preset 副本）
        for inst in self._instance_guardrails:
            if isinstance(inst, _BPG):
                inst.tool_registry = value
        _logger.debug("GuardrailMiddleware.tool_registry 已广播到 %d 个 BasicPermissionGuardrail 实例",
                      sum(1 for i in self._instance_guardrails if isinstance(i, _BPG)))

    def _load_presets(self, names: Optional[List[str]] = None) -> None:
        """加载预设护栏。如果指定 names 则只加载指定的，否则加载全部默认预设。"""
        from .preset import (
            BasicInputSafetyGuardrail, BasicOutputSafetyGuardrail,
            PromptInjectionGuardrail, BasicPermissionGuardrail,
        )

        # 默认预设集合：先注册原型到全局注册表，再往实例挂载 clone 副本
        _prototypes = {
            "basic_input_safety": BasicInputSafetyGuardrail(),
            "basic_output_safety": BasicOutputSafetyGuardrail(),
            "prompt_injection": PromptInjectionGuardrail(),
            "basic_permission": BasicPermissionGuardrail(
                tool_registry=self.tool_registry,
            ),
        }
        for pn, proto in _prototypes.items():
            # 已存在就不重复注册，避免覆盖用户自定义的同名原型
            if pn not in _GLOBAL_GUARDRAIL_REGISTRY:
                proto.register()

        preset_names = names if names is not None else list(_prototypes.keys())
        for preset_name in preset_names:
            if preset_name in _prototypes:
                # 预设也走 add_guardrail（clone + fail_open 同步），保证实例隔离
                self.add_guardrail(_prototypes[preset_name], skip_exist_check=True)
            else:
                _logger.warning("未知预设护栏: %s", preset_name)

    def add_guardrail(self, guardrail: GuardrailBase, skip_exist_check: bool = False) -> None:
        """添加一个护栏（自动 clone 副本，实现多 Agent 隔离）。

        Args:
            guardrail: 原型或现有实例；内部会 clone() 一份副本再挂载，
                所以对返回对象 enable/disable 不会影响其他 Agent。
            skip_exist_check: 内部 _load_presets 调用时跳过同名检查，避免重复告警。
        """
        if not isinstance(guardrail, GuardrailBase):
            raise TypeError(f"add_guardrail 只接受 GuardrailBase 实例，收到 {type(guardrail)}")

        # 先按同名移除旧的，避免重复挂载
        existing_idx = None
        for i, g in enumerate(self._instance_guardrails):
            if g.name == guardrail.name:
                existing_idx = i
                break
        if existing_idx is not None:
            self._instance_guardrails.pop(existing_idx)
            if not skip_exist_check:
                _logger.warning("[guardrail] 替换同名护栏实例: %s", guardrail.name)

        # clone 一份独立副本，避免多 Agent 共享实例导致 enable/disable/fail_open 串扰。
        # fail_open 继承规则（对齐 _fail_open_explicit 标记语义）：
        #   1) 若实例在构造时显式设置了 fail_open（True/False 均可），保留用户值，middleware 不覆盖；
        #   2) 若实例未显式设置（仅用类默认 False），则继承 middleware 的统一姿态 self.fail_open。
        # 注意：构造之后再手动赋值 instance.fail_open = X 不会被 _fail_open_explicit 识别，
        # 请在构造器中显式传参；clone() 走 deepcopy，标记会随副本一起复制。
        instance = guardrail.clone()
        if not getattr(instance, "_fail_open_explicit", False):
            instance.fail_open = self.fail_open
        self._instance_guardrails.append(instance)

    def remove_guardrail(self, name: str) -> bool:
        """按名称移除一个护栏，返回是否真的移除过。"""
        for i, g in enumerate(self._instance_guardrails):
            if g.name == name:
                self._instance_guardrails.pop(i)
                return True
        return False

    def enable_guardrail(self, name: str) -> bool:
        for g in self._instance_guardrails:
            if g.name == name:
                g.enabled = True
                return True
        return False

    def disable_guardrail(self, name: str) -> bool:
        for g in self._instance_guardrails:
            if g.name == name:
                g.enabled = False
                return True
        return False

    def get_guardrails(self) -> List[GuardrailBase]:
        """返回当前实例挂载的护栏副本快照（只用于观测/调试，不直接返回可变引用）。"""
        return list(self._instance_guardrails)

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息：
        - ``registered``：当前挂载的护栏名列表（兼容旧测试与监控）；
        - ``reject``：累计拦截总数；
        - ``pass``：累计放行总数；
        - 其它键：按护栏名拆分的拦截次数（``{name: count}``）。
        """
        stats: Dict[str, Any] = {
            "registered": [g.name for g in self._instance_guardrails],
            "reject": dict(self._reject_stats),
            "pass": dict(self._pass_stats),
        }
        # 总览字段（与老测试/监控兼容）
        stats["reject_count"] = sum(self._reject_stats.values())
        stats["pass_count"] = self._pass_count
        return stats

    def _get_guardrails_by_type(self, guardrail_type: type) -> List[GuardrailBase]:
        return [g for g in self._instance_guardrails if isinstance(g, guardrail_type)]

    def _record_result(
        self,
        guardrail_name: str,
        result: GuardrailResult,
        phase: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """统一记录一次护栏检查结果：统计 + Trace 打点（复用 log_guardrail_event 唯一入口）。"""
        if not result.allowed:
            self._reject_stats[guardrail_name] = self._reject_stats.get(guardrail_name, 0) + 1
        else:
            self._pass_count += 1
            self._pass_stats[guardrail_name] = self._pass_stats.get(guardrail_name, 0) + 1
        # L7/L8：唯一打点入口，SpanKind=GUARDRAIL，字段统一，不重复写 start_span/end_span
        log_guardrail_event(
            self.tracer,
            guardrail_name=guardrail_name,
            result=result,
            phase=phase,
            context=context,
        )

    def _record_guardrail_error(
        self,
        guardrail_name: str,
        phase: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """护栏组件自身异常（fail-closed）的统一记录入口。

        与 _record_result 的区别：
        - 不把异常伪装成"规则命中"，独立计数到 ``__error__`` 键，避免故障率被淹没；
        - log_guardrail_event 的 metadata 显式标 ``is_error=True``、``reason_kind="error"``，
          与规则命中走同一个 SpanKind.GUARDRAIL 打点通道，但字段可被观测面区分；
        - 调用方只负责构造结构化 _GuardrailBlocked（或 False/str 等原签名返回）返回主循环，
          不再 raise GuardrailError，避免被主循环 except Exception 静默吞掉。
        """
        _err_name = guardrail_name or "__guardrail_error__"
        # 错误独立计数，不与规则命中的 reject_count 混在一起
        self._reject_stats[f"{_err_name}__error"] = (
            self._reject_stats.get(f"{_err_name}__error", 0) + 1
        )
        err_result = GuardrailResult(
            allowed=False,
            reason=f"{guardrail_name} 组件异常（fail-closed）: {error}",
            metadata={
                "guardrail_name": _err_name,
                "is_error": True,
                "reason_kind": "error",
                "error_type": type(error).__name__,
            },
        )
        log_guardrail_event(
            self.tracer,
            guardrail_name=_err_name,
            result=err_result,
            phase=phase,
            context=context,
        )

    # ── AgentMiddleware 钩子实现 ──────────────────────────────

    def before_agent_run(
        self,
        user_input: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[_GuardrailBlocked]:
        """用户输入进入推理前：执行输入护栏检查。

        S1（fail-closed 承诺真实兑现）：无论护栏内部是规则命中还是自身崩溃，
        本钩子都只返回 ``_GuardrailBlocked``（拦截）或 ``None``（放行），
        **不再向外抛异常**。主循环在 L5 之后不再用 ``except Exception`` 吞护栏异常，
        这里统一把 fail-closed 的组件故障转成结构化拦截返回，避免静默放行。
        """
        try:
            input_guardrails = self._get_guardrails_by_type(InputGuardrail)
            results = run_input_guardrails(
                user_input, context, guardrails=input_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="input_check", context=context)
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛 InputGuardrailReject，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "input_check", RuntimeError(_detail), context=context)
                    _logger.error("输入护栏 %s 执行异常（fail-closed，已拦截请求）: %s", _gname, _detail)
                    return _GuardrailBlocked(
                        phase="input",
                        message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                        reason_kind="error",
                        detail=_detail,
                    )
            return None  # 放行
        except InputGuardrailReject as e:
            _details = e.details or {}
            # S5：先补记录异常抛出之前已通过的护栏结果，避免 _pass_stats / trace span 丢失
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="input_check", context=context)
            # is_error=True 表示护栏自身崩溃（fail-closed 才会走到这里，fail-open 在 invoke 里已 allowed=True），
            # 不看 self.fail_open——实例级姿态优先，这里结构化拦截 + 独立错误打点
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "input_check", e, context=context)
                _logger.error("输入护栏 %s 执行异常（fail-closed，已拦截请求）: %s", e.guardrail_name, e)
                return _GuardrailBlocked(
                    phase="input",
                    message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                    reason_kind="error",
                    detail=str(e),
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="input_check", context=context)
            if self.on_input_reject:
                msg = self.on_input_reject(e)
            else:
                msg = f"请求被安全护栏拦截：{e}"
            return _GuardrailBlocked(phase="input", message=msg)
        except Exception as e:
            if self.fail_open:
                _logger.warning("输入护栏整体异常（fail_open=True，放行）: %s", e)
                return None
            # fail-closed：组件自身异常（KeyError/TypeError 等），结构化返回拦截
            self._record_guardrail_error("__middleware__", "input_check", e, context=context)
            _logger.error("输入护栏整体执行异常（fail-closed，已拦截请求）: %s", e)
            return _GuardrailBlocked(
                phase="input",
                message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                reason_kind="error",
                detail=str(e),
            )

    async def abefore_agent_run(
        self,
        user_input: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[_GuardrailBlocked]:
        """异步版本，语义同 before_agent_run（S1：不抛异常，只结构化返回）。"""
        try:
            input_guardrails = self._get_guardrails_by_type(InputGuardrail)
            results = await arun_input_guardrails(
                user_input, context, guardrails=input_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="input_check", context=context)
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "input_check", RuntimeError(_detail),
                                                 context={**(context or {})})
                    _logger.error("输入护栏（异步） %s 执行异常（fail-closed，已拦截请求）: %s", _gname, _detail)
                    return _GuardrailBlocked(
                        phase="input",
                        message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                        reason_kind="error",
                        detail=_detail,
                    )
            return None
        except InputGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="input_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "input_check", e, context=context)
                _logger.error("输入护栏（异步）%s 执行异常（fail-closed，已拦截请求）: %s", e.guardrail_name, e)
                return _GuardrailBlocked(
                    phase="input",
                    message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                    reason_kind="error",
                    detail=str(e),
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="input_check", context=context)
            if self.on_input_reject:
                msg = self.on_input_reject(e)
            else:
                msg = f"请求被安全护栏拦截：{e}"
            return _GuardrailBlocked(phase="input", message=msg)
        except Exception as e:
            if self.fail_open:
                _logger.warning("输入护栏整体异步异常（fail_open=True，放行）: %s", e)
                return None
            self._record_guardrail_error("__middleware__", "input_check", e, context=context)
            _logger.error("输入护栏（异步）整体执行异常（fail-closed，已拦截请求）: %s", e)
            return _GuardrailBlocked(
                phase="input",
                message="请求被安全护栏拦截：输入安全检查发生内部错误，为保证安全已拒绝本次请求。",
                reason_kind="error",
                detail=str(e),
            )

    def before_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """工具调用前：执行工具前置护栏。返回 True=放行，False=拦截（原因写入 _last_tool_reject_reason）。

        S1：护栏自身崩溃（fail-closed）不再 raise，改为返回 False + 把错误文案写入
        ``_last_tool_reject_reason``，主循环读取该属性走正常拦截分支，确保工具不会被静默放行。
        """
        self._last_tool_reject_reason = None
        try:
            pre_guardrails = self._get_guardrails_by_type(ToolCallGuardrail)
            tool_call_info = ToolCallInfo(
                tool_name=tool_name,
                tool_args=dict(tool_args) if tool_args else {},
                tool_type=kwargs.get("tool_type", "local"),
                source=kwargs.get("source", ""),
                call_id=kwargs.get("call_id", ""),
            )
            results = run_tool_call_guardrails(
                tool_call_info, context, guardrails=pre_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="tool_pre_check",
                                    context={**(context or {}), "tool_name": tool_name})
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "tool_pre_check", RuntimeError(_detail),
                                                 context={**(context or {}), "tool_name": tool_name})
                    _logger.error("工具前置护栏 %s 执行异常（fail-closed，已拦截调用 %s）: %s", _gname, tool_name, _detail)
                    return False
            return True
        except ToolCallGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="tool_pre_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "tool_pre_check", e,
                                             context={**(context or {}), "tool_name": tool_name})
                _logger.error("工具前置护栏 %s 执行异常（fail-closed，已拦截调用 %s）: %s",
                              e.guardrail_name, tool_name, e)
                self._last_tool_reject_reason = (
                    f"工具 {tool_name} 调用被安全护栏拦截：工具安全检查发生内部错误，为保证安全已拒绝本次调用。"
                )
                return False
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="tool_pre_check",
                                context={**(context or {}), "tool_name": tool_name})
            if self.on_tool_reject:
                self._last_tool_reject_reason = self.on_tool_reject(e)
            else:
                self._last_tool_reject_reason = f"工具调用被安全护栏拦截：{e}"
            return False
        except Exception as e:
            if self.fail_open:
                _logger.warning("[guardrail] 工具前置护栏整体异常（fail_open=True，放行 %s）: %s", tool_name, e)
                return True
            # fail-closed：组件崩溃 → 返回 False（拦截），写入固定文案，主循环会读 _last_tool_reject_reason
            self._record_guardrail_error("__middleware__", "tool_pre_check", e,
                                         context={**(context or {}), "tool_name": tool_name})
            _logger.error("[guardrail] 工具前置护栏整体异常（fail-closed，已拦截调用 %s）: %s", tool_name, e)
            self._last_tool_reject_reason = (
                f"工具 {tool_name} 调用被安全护栏拦截：工具安全检查发生内部错误，为保证安全已拒绝本次调用。"
            )
            return False

    async def abefore_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        self._last_tool_reject_reason = None
        try:
            pre_guardrails = self._get_guardrails_by_type(ToolCallGuardrail)
            tool_call_info = ToolCallInfo(
                tool_name=tool_name,
                tool_args=dict(tool_args) if tool_args else {},
                tool_type=kwargs.get("tool_type", "local"),
                source=kwargs.get("source", ""),
                call_id=kwargs.get("call_id", ""),
            )
            results = await arun_tool_call_guardrails(
                tool_call_info, context, guardrails=pre_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="tool_pre_check",
                                    context={**(context or {}), "tool_name": tool_name})
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "tool_pre_check", RuntimeError(_detail),
                                                 context={**(context or {}), "tool_name": tool_name})
                    _logger.error("[guardrail] 工具前置护栏（异步） %s 执行异常（fail-closed，已拦截调用 %s）: %s", _gname, tool_name, _detail)
                    return False
            return True
        except ToolCallGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="tool_pre_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "tool_pre_check", e,
                                             context={**(context or {}), "tool_name": tool_name})
                _logger.error("[guardrail] 工具前置护栏（异步）%s 异常（fail-closed，已拦截调用 %s）: %s",
                              e.guardrail_name, tool_name, e)
                self._last_tool_reject_reason = (
                    f"工具 {tool_name} 调用被安全护栏拦截：工具安全检查发生内部错误，为保证安全已拒绝本次调用。"
                )
                return False
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="tool_pre_check",
                                context={**(context or {}), "tool_name": tool_name})
            if self.on_tool_reject:
                self._last_tool_reject_reason = self.on_tool_reject(e)
            else:
                self._last_tool_reject_reason = f"工具调用被安全护栏拦截：{e}"
            return False
        except Exception as e:
            if self.fail_open:
                _logger.warning("[guardrail] 工具前置护栏异步整体异常（fail_open=True，放行 %s）: %s", tool_name, e)
                return True
            self._record_guardrail_error("__middleware__", "tool_pre_check", e,
                                         context={**(context or {}), "tool_name": tool_name})
            _logger.error("[guardrail] 工具前置护栏（异步）整体异常（fail-closed，已拦截调用 %s）: %s", tool_name, e)
            self._last_tool_reject_reason = (
                f"工具 {tool_name} 调用被安全护栏拦截：工具安全检查发生内部错误，为保证安全已拒绝本次调用。"
            )
            return False

    def after_tool_call(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str = "",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """工具返回后：执行工具后置护栏（脱敏/结果过滤）。返回新的 observation 字符串或 None。

        S1：fail-closed 路径不抛异常，而是返回一段拦截占位 observation 替换掉真实输出，
        主循环把它写回 tool 消息喂给 LLM，既不静默放行真实结果、也不崩溃流程。
        """
        try:
            post_guardrails = [g for g in self._get_guardrails_by_type(ToolCallGuardrail) if g.check_result]
            result_info = ToolResultInfo(
                tool_name=tool_name,
                success=success,
                output=output or "",
                error=error or "",
                metadata=kwargs.get("metadata", {}),
            )
            new_result, results = run_tool_result_guardrails(
                result_info, context, guardrails=post_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="tool_post_check",
                                    context={**(context or {}), "tool_name": tool_name, "success": success})
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "tool_post_check", RuntimeError(_detail),
                                                 context={**(context or {}), "tool_name": tool_name, "success": success})
                    _logger.error("工具后置护栏 %s 执行异常（fail-closed，已拦截结果 %s）: %s", _gname, tool_name, _detail)
                    return "工具执行结果被安全护栏拦截：结果检查发生内部错误，为保证安全已拦截。"
            # 未修改时返回原始 output（保持向后兼容）；修改后返回新的 output 字符串
            new_output = getattr(new_result, "output", new_result) if new_result is not None else output
            return new_output or output
        except ToolCallGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="tool_post_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "tool_post_check", e,
                                             context={**(context or {}), "tool_name": tool_name})
                _logger.error("[guardrail] 工具后置护栏 %s 异常（fail-closed，已拦截结果 %s）: %s",
                              e.guardrail_name, tool_name, e)
                return (
                    f"工具 {tool_name} 执行结果被安全护栏拦截：结果安全检查发生内部错误，"
                    f"为保证安全已屏蔽原始结果。"
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="tool_post_check",
                                context={**(context or {}), "tool_name": tool_name})
            if self.on_tool_reject:
                return self.on_tool_reject(e)
            return f"工具执行结果被安全护栏拦截：{e}"
        except Exception as e:
            if self.fail_open:
                _logger.warning("[guardrail] 工具后置护栏异常（fail_open=True，放行 %s）: %s", tool_name, e)
                return output or ""
            self._record_guardrail_error("__middleware__", "tool_post_check", e,
                                         context={**(context or {}), "tool_name": tool_name})
            _logger.error("[guardrail] 工具后置护栏整体异常（fail-closed，已拦截结果 %s）: %s", tool_name, e)
            return (
                f"工具 {tool_name} 执行结果被安全护栏拦截：结果安全检查发生内部错误，"
                f"为保证安全已屏蔽原始结果。"
            )

    async def aafter_tool_call(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str = "",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        try:
            post_guardrails = [g for g in self._get_guardrails_by_type(ToolCallGuardrail) if g.check_result]
            result_info = ToolResultInfo(
                tool_name=tool_name,
                success=success,
                output=output or "",
                error=error or "",
                metadata=kwargs.get("metadata", {}),
            )
            new_result, results = await arun_tool_result_guardrails(
                result_info, context, guardrails=post_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="tool_post_check",
                                    context={**(context or {}), "tool_name": tool_name, "success": success})
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "tool_post_check", RuntimeError(_detail),
                                                 context={**(context or {}), "tool_name": tool_name, "success": success})
                    _logger.error("[guardrail] 工具后置护栏（异步） %s 执行异常（fail-closed，已拦截结果 %s）: %s", _gname, tool_name, _detail)
                    return "工具执行结果被安全护栏拦截：结果检查发生内部错误，为保证安全已拦截。"
            new_output = getattr(new_result, "output", new_result) if new_result is not None else output
            return new_output or output
        except ToolCallGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="tool_post_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "tool_post_check", e,
                                             context={**(context or {}), "tool_name": tool_name})
                _logger.error("[guardrail] 工具后置护栏（异步）%s 异常（fail-closed，已拦截结果 %s）: %s",
                              e.guardrail_name, tool_name, e)
                return (
                    f"工具 {tool_name} 执行结果被安全护栏拦截：结果安全检查发生内部错误，"
                    f"为保证安全已屏蔽原始结果。"
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="tool_post_check",
                                context={**(context or {}), "tool_name": tool_name})
            if self.on_tool_reject:
                return self.on_tool_reject(e)
            return f"工具执行结果被安全护栏拦截：{e}"
        except Exception as e:
            if self.fail_open:
                _logger.warning("[guardrail] 工具后置护栏异步异常（fail_open=True，放行 %s）: %s", tool_name, e)
                return output or ""
            self._record_guardrail_error("__middleware__", "tool_post_check", e,
                                         context={**(context or {}), "tool_name": tool_name})
            _logger.error("[guardrail] 工具后置护栏（异步）整体异常（fail-closed，已拦截结果 %s）: %s", tool_name, e)
            return (
                f"工具 {tool_name} 执行结果被安全护栏拦截：结果安全检查发生内部错误，"
                f"为保证安全已屏蔽原始结果。"
            )

    def after_agent_run(
        self,
        agent_output: Union[str, List[Any], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union[_GuardrailBlocked, _GuardrailModified]]:
        """Agent最终输出后：执行输出护栏，支持拦截或修改（脱敏）。

        S1：fail-closed 路径不抛异常，直接返回 _GuardrailBlocked(phase="output", reason_kind="error")，
        主循环会用 message 替换 final_answer，保证拦截文案能走到用户侧。
        """
        try:
            output_guardrails = self._get_guardrails_by_type(OutputGuardrail)
            final_output, results = run_output_guardrails(
                agent_output, context, guardrails=output_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="output_check", context=context)
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "output_check", RuntimeError(_detail),
                                                 context={**(context or {})})
                    _logger.error("输出护栏 %s 执行异常（fail-closed，已拦截输出）: %s", _gname, _detail)
                    return _GuardrailBlocked(
                        phase="output",
                        message="请求被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拦截输出。",
                        reason_kind="error",
                        detail=_detail,
                    )
            if final_output == agent_output:
                return None
            return _GuardrailModified(phase="output", content=final_output)
        except OutputGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="output_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "output_check", e, context=context)
                _logger.error("输出护栏 %s 执行异常（fail-closed，已拦截输出）: %s", e.guardrail_name, e)
                return _GuardrailBlocked(
                    phase="output",
                    message="输出被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拒绝展示原始输出。",
                    reason_kind="error",
                    detail=str(e),
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="output_check", context=context)
            if self.on_output_reject:
                msg = self.on_output_reject(e)
            else:
                msg = f"输出被安全护栏拦截：{e}"
            return _GuardrailBlocked(phase="output", message=msg)
        except Exception as e:
            if self.fail_open:
                _logger.warning("输出护栏执行异常（fail_open=True，放行）: %s", e)
                return None
            self._record_guardrail_error("__middleware__", "output_check", e, context=context)
            _logger.error("输出护栏整体执行异常（fail-closed，已拦截输出）: %s", e)
            return _GuardrailBlocked(
                phase="output",
                message="输出被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拒绝展示原始输出。",
                reason_kind="error",
                detail=str(e),
            )

    async def aafter_agent_run(
        self,
        agent_output: Union[str, List[Any], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union[_GuardrailBlocked, _GuardrailModified]]:
        """异步版本：Agent最终输出后检查，返回语义同 after_agent_run（S1：不抛异常）。"""
        try:
            output_guardrails = self._get_guardrails_by_type(OutputGuardrail)
            final_output, results = await arun_output_guardrails(
                agent_output, context, guardrails=output_guardrails, fail_open=None,
            )
            for r in results:
                self._record_result(r.metadata.get("guardrail_name", ""), r, phase="output_check", context=context)
                # 防御性双重检查：聚合层理论上会在第一条 not allowed 处抛对应 Reject 异常，
                # 这里保留显式的 is_error 判定，防止未来聚合层行为变化时静默放行。
                # 姿态判断交给 r.allowed（invoke 已把实例级 fail_open 编码进来），
                # 不再用 self.fail_open 二次覆盖，避免实例显式 True 被 middleware 压回 False。
                if r.metadata.get("is_error") and not r.allowed:
                    _gname = r.metadata.get("guardrail_name", "")
                    _detail = str(r.reason)
                    self._record_guardrail_error(_gname, "output_check", RuntimeError(_detail),
                                                 context={**(context or {})})
                    _logger.error("输出护栏（异步） %s 执行异常（fail-closed，已拦截输出）: %s", _gname, _detail)
                    return _GuardrailBlocked(
                        phase="output",
                        message="请求被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拦截输出。",
                        reason_kind="error",
                        detail=_detail,
                    )
            if final_output == agent_output:
                return None
            return _GuardrailModified(phase="output", content=final_output)
        except OutputGuardrailReject as e:
            _details = e.details or {}
            # S5：补记录异常抛出之前已通过的护栏结果
            for _r in getattr(e, "results", []) or []:
                self._record_result(_r.metadata.get("guardrail_name", ""), _r, phase="output_check", context=context)
            if _details.get("is_error"):
                self._record_guardrail_error(e.guardrail_name, "output_check", e, context=context)
                _logger.error("输出护栏（异步）%s 异常（fail-closed，已拦截输出）: %s", e.guardrail_name, e)
                return _GuardrailBlocked(
                    phase="output",
                    message="输出被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拒绝展示原始输出。",
                    reason_kind="error",
                    detail=str(e),
                )
            reject_result = GuardrailResult(
                allowed=False, reason=str(e),
                metadata={"guardrail_name": e.guardrail_name, "duration_ms": _details.get("duration_ms"), **_details},
            )
            self._record_result(e.guardrail_name, reject_result, phase="output_check", context=context)
            if self.on_output_reject:
                msg = self.on_output_reject(e)
            else:
                msg = f"输出被安全护栏拦截：{e}"
            return _GuardrailBlocked(phase="output", message=msg)
        except Exception as e:
            if self.fail_open:
                _logger.warning("输出护栏异步执行异常（fail_open=True，放行）: %s", e)
                return None
            self._record_guardrail_error("__middleware__", "output_check", e, context=context)
            _logger.error("输出护栏（异步）整体执行异常（fail-closed，已拦截输出）: %s", e)
            return _GuardrailBlocked(
                phase="output",
                message="输出被安全护栏拦截：输出安全检查发生内部错误，为保证安全已拒绝展示原始输出。",
                reason_kind="error",
                detail=str(e),
            )

    def after_agent_turn(self) -> None:
        """每轮对话结束后的清理工作，对齐SkillMiddleware接口"""
        pass
