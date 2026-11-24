"""Agent 中间件协议基类（AgentMiddleware ABC）。

参考 FastAPI/Starlette Middleware、LangChain BaseCallbackHandler、
OpenTelemetry Middleware 的共同设计：所有挂载到 ReActAgent ``middleware=[...]``
列表的对象必须继承此 ABC，以统一钩子签名，避免鸭子类型分发导致的
「同名钩子不同签名、塞错中间件静默跑错」问题（参见 AGENTS.md §5.6 / R3）。

设计要点：
- 钩子签名对齐 GuardrailMiddleware（新引入的安全中间件）而非 SkillMiddleware：
  SkillMiddleware 走 ReActAgent 专属的 ``self.skill_middleware`` 通道，有自己
  的 after_tool_call(name, args, obs) 三参签名，与通用中间件的
  after_tool_call(name, success, output, error, ...) 四参签名不同；
  SkillMiddleware **不**继承本 ABC，也就不能被误塞进 ``middleware=[]``。
- 所有钩子提供默认空实现，子类只需要覆盖需要的方法；
- 同步/异步成对，默认异步通过 asyncio.to_thread 转同步，减少子类实现负担；
- 返回值语义用 dataclass 明确（GuardrailBlocked/GuardrailModified 在
  guardrail_middleware 中定义，这里只声明 Any 返回类型以避免循环依赖）。
"""
from __future__ import annotations

import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional, Union


class AgentMiddleware(ABC):
    """ReActAgent 通用中间件基类。

    任何要传给 ``ReActAgent(middleware=[...])`` 的中间件都必须继承此类。
    ReActAgent 在启动时会对 ``middleware`` 列表做 ``isinstance`` 校验，
    不符合的会直接抛出 ``TypeError``，避免误塞 SkillMiddleware 等
    同名钩子但签名不兼容的对象导致静默跑错。

    钩子调用时机与返回值语义：

    - ``before_agent_run(user_input, context=None)``
      在用户输入进入推理循环前调用（输入护栏、审计等场景）。
      返回值：
        * ``None`` 放行；
        * ``GuardrailBlocked(phase="input", message=...)`` 拦截本轮，
          message 作为最终回答直接返回给用户。

    - ``before_tool_call(tool_name, tool_args, context=None, **kwargs)``
      每次工具调用前调用。
      返回值（Bug3 协议升级，兼容两种形态）：
        * ``True`` 允许调用；
        * ``False`` 拦截调用，Agent 会把 ``_last_tool_reject_reason``
          作为工具结果喂回 LLM（不终止本轮对话）；
        * ``(allowed, reason)`` 二元组（推荐）：拦截原因随返回值携带，
          并发调用同一中间件实例时不存在共享属性竞态；旧式 bool 实现仍兼容。
          实现类需重写 ``__bool__``，保证 ``if not ret:`` 对拦截元组同样成立。

    - ``after_tool_call(tool_name, success, output="", error="", context=None, **kwargs)``
      工具返回结果后调用（结果脱敏、审计、内容过滤等）。
      返回值：
        * ``None`` 或原样字符串 → 结果不变；
        * ``str`` → 替换工具 observation（脱敏/改写）。

    - ``after_agent_run(agent_output, context=None)``
      Agent 产出最终回答、返回给用户前调用（输出护栏、后处理）。
      返回值：
        * ``None`` 放行；
        * ``GuardrailBlocked(phase="output", message=...)`` 拦截，message 替换最终答案；
        * ``GuardrailModified(phase="output", content=...)`` 修改/脱敏，content 替换最终答案。

    - ``after_agent_turn()`` / ``aafter_agent_turn()``
      本轮对话结束后（finally 块中）调用，用于清理状态、卸载上下文、打点收尾等。
      无返回值，异常会被 ReActAgent 兜底吞掉打 warning，绝不阻塞主流程。

    同步/异步成对性（E10）：以上五个钩子**全部**同时提供 ``a*`` 异步版本。
    ReActAgent 的异步主循环 ``arun_stream`` 只 await ``a*`` 版本，同步主循环
    ``run_stream`` 只调同步版本；两条链路各调用一次，不会重复触发。
    """

    # ── 输入检查 ─────────────────────────────────────────────

    def before_agent_run(
        self,
        user_input: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Agent 运行前钩子。默认放行，子类覆盖实现拦截/改写逻辑。"""
        return None

    async def abefore_agent_run(
        self,
        user_input: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """异步版本，默认在线程池中调用同步 before_agent_run。"""
        return await asyncio.to_thread(self.before_agent_run, user_input, context)

    # ── 工具前置检查 ─────────────────────────────────────────────

    def before_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[bool, "tuple[bool, str]"]:
        """工具调用前钩子。返回 True 放行，False/(False, 原因) 拦截。默认放行。

        Bug3：推荐返回 ``(allowed, reason)`` 二元组把拦截原因随返回值携带（线程安全），
        旧式 bool 返回仍兼容（原因回退 ``_last_tool_reject_reason`` 属性/默认文案）。
        """
        return True

    async def abefore_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[bool, "tuple[bool, str]"]:
        """异步版本。"""
        return await asyncio.to_thread(self.before_tool_call, tool_name, tool_args, context, **kwargs)

    # ── 工具后置检查 ─────────────────────────────────────────────

    def after_tool_call(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str = "",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """工具结果后钩子。返回 None/原串不改；返回 str 则替换 observation。"""
        return None

    async def aafter_tool_call(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: str = "",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """异步版本。"""
        return await asyncio.to_thread(
            self.after_tool_call, tool_name, success, output, error, context, **kwargs
        )

    # ── 输出检查 ─────────────────────────────────────────────

    def after_agent_run(
        self,
        agent_output: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Agent 最终输出钩子。默认放行。"""
        return None

    async def aafter_agent_run(
        self,
        agent_output: Union[str, List[Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """异步版本。"""
        return await asyncio.to_thread(self.after_agent_run, agent_output, context)

    # ── 轮次收尾 ─────────────────────────────────────────────

    def after_agent_turn(self) -> None:
        """每轮结束后钩子，默认空操作。异常会被 Agent 兜底吞掉。"""
        return None

    async def aafter_agent_turn(self) -> None:
        """``after_agent_turn`` 的异步版本，默认在线程池中调用同步实现。

        E10 补齐：本钩子原先只有同步版，导致异步主循环要么静默漏调轮次收尾
        （技能 unload_after_turn 卸载、状态清理），要么在事件循环里直跑同步钩子。
        现与其他四个钩子保持成对，ReActAgent.arun_stream 在 finally 里 await 本方法。

        与同步版一致：实现**不得抛出异常**给主循环；即便抛出，Agent 侧也只记
        warning 后继续收尾（R6：观测/收尾代码不得打挂业务流程）。
        """
        await asyncio.to_thread(self.after_agent_turn)
        return None
