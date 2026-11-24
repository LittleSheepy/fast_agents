"""ReActAgent 异步主循环（E10 / A1 修复）端到端测试

A1 的原缺陷：LLM / Tools / Guardrails / Middleware 四层各自实现了异步能力，
但 Agent 层只有同步 run/run_stream，导致四层异步代码端到端不可达（约 850 行死代码）。
本文件把「异步链路真的被接通」这件事逐条钉死，覆盖：

1. arun 与 run 结果一致（同构性）；
2. 中间件 a* 钩子确实被 await（含本轮新补的 aafter_agent_turn），且同步链路只调同步钩子；
3. Guardrails 整条 arun_* 链路在异步主循环里端到端生效（输入拦截 / 工具前置拦截 /
   输出护栏后的正文同时体现在 end 事件与 arun() 返回值上；A2 的剩余限制仅在
   流式路径——text 增量在护栏前已发出，同步异步同病）；
4. 工具异步执行：原生 async 工具直接 await、纯同步工具降级线程池、超时/审批/hooks 语义对齐同步版；
5. @tool 装饰 async def 的同步路径显式报错（E10-b），不再返回协程对象乱码；
6. parallel_tool_calls 异步并发、return_direct 不进下一轮推理（与同步版同批全执行语义对齐）；
7. duck-typed LLM（无 astream_invoke）回退、AgentBase 默认异步入口不破坏既有子类；
8. R6：收尾钩子抛异常不得打挂业务流程。

全部离线可跑，不依赖 API Key / 外部服务；不使用 pytest。
运行：python tests/agents/test_react_agent_async.py
"""

import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.llm.models import StreamChunk
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.memory import MemoryManager
from fast_agents.tools.tool_base import ToolBase
from fast_agents.tools.tool_registry import ToolRegistry
from fast_agents.tools.structured_tool import StructuredTool, tool
from fast_agents.agents.agent_base import AgentBase
from fast_agents.agents.middleware import AgentMiddleware
from fast_agents.agents.react_agent.react_agent import ReActAgent
from fast_agents.guardrails import GuardrailMiddleware, InputGuardrail, OutputGuardrail
from fast_agents.guardrails.models import GuardrailResult


# ==================== 公共测试替身 ====================

class MockAsyncLLM:
    """脚本化假 LLM：同时提供同步 invoke/stream_invoke 与异步 astream_invoke。

    用于对比「同一条脚本走同步链路与异步链路，喂给模型的消息与产出完全一致」。
    """

    def __init__(self, script):
        self.script = script
        self.calls = []       # 每次 LLM 调用实际收到的 messages
        self.i = 0
        self.sync_stream_calls = 0
        self.async_stream_calls = 0

    def _next(self, messages):
        self.calls.append(list(messages))
        response = self.script[self.i]
        self.i += 1
        return response

    @staticmethod
    def _to_chunks(response):
        content = response.content or ""
        if content:
            yield StreamChunk(type="text", text=content)
        tcs = getattr(response, "tool_calls", None) or []
        yield StreamChunk(
            type="end",
            content=content,
            tool_calls=tcs,
            finish_reason="tool_calls" if tcs else "stop",
        )

    def invoke(self, messages, tools=None, **kwargs):
        return self._next(messages)

    def stream_invoke(self, messages, tools=None, **kwargs):
        self.sync_stream_calls += 1
        yield from self._to_chunks(self._next(messages))

    async def astream_invoke(self, messages, tools=None, **kwargs):
        self.async_stream_calls += 1
        response = self._next(messages)
        for chunk in self._to_chunks(response):
            yield chunk


class MockSyncOnlyLLM:
    """duck-typed 假 LLM：只有同步接口，用于验证异步主循环的回退分支。"""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.i = 0
        self.sync_stream_calls = 0

    def _next(self, messages):
        self.calls.append(list(messages))
        r = self.script[self.i]
        self.i += 1
        return r

    def invoke(self, messages, tools=None, **kwargs):
        return self._next(messages)

    def stream_invoke(self, messages, tools=None, **kwargs):
        self.sync_stream_calls += 1
        for chunk in MockAsyncLLM._to_chunks(self._next(messages)):
            yield chunk


class HookProbeMiddleware(AgentMiddleware):
    """把五个钩子的「同步版 / 异步版」调用分别记账，用于证明两条链路各走各的接口。"""

    def __init__(self, block_tool_names=(), tool_reject_reason="参数命中黑名单", mask_observation=None):
        self.sync_calls = []
        self.async_calls = []
        self._block_tool_names = set(block_tool_names)
        self._tool_reject_reason = tool_reject_reason
        self._mask_observation = mask_observation

    # 输入
    def before_agent_run(self, user_input, context=None):
        self.sync_calls.append("before_agent_run")
        return None

    async def abefore_agent_run(self, user_input, context=None):
        self.async_calls.append("abefore_agent_run")
        return None

    # 工具前置
    def before_tool_call(self, tool_name, tool_args, context=None, **kwargs):
        self.sync_calls.append("before_tool_call")
        if tool_name in self._block_tool_names:
            return (False, self._tool_reject_reason)
        return True

    async def abefore_tool_call(self, tool_name, tool_args, context=None, **kwargs):
        self.async_calls.append("abefore_tool_call")
        if tool_name in self._block_tool_names:
            return (False, self._tool_reject_reason)
        return True

    # 工具后置
    def after_tool_call(self, tool_name, success, output="", error="", context=None, **kwargs):
        self.sync_calls.append("after_tool_call")
        return self._mask_observation

    async def aafter_tool_call(self, tool_name, success, output="", error="", context=None, **kwargs):
        self.async_calls.append("aafter_tool_call")
        return self._mask_observation

    # 输出
    def after_agent_run(self, agent_output, context=None):
        self.sync_calls.append("after_agent_run")
        return None

    async def aafter_agent_run(self, agent_output, context=None):
        self.async_calls.append("aafter_agent_run")
        return None

    # 轮次收尾
    def after_agent_turn(self):
        self.sync_calls.append("after_agent_turn")
        return None

    async def aafter_agent_turn(self):
        self.async_calls.append("aafter_agent_turn")
        return None


def _tc(call_id, name, arguments):
    """构造一条 tool_call（arguments 用 JSON 字符串，贴近真实 OpenAI 返回）。"""
    return MsgToolCall(id=call_id, name=name, arguments=arguments)


def _make_memory(llm):
    """关掉摘要，避免假 LLM 被记忆模块额外消费掉一步脚本。"""
    return MemoryManager(llm=llm, enable_summary=False, max_messages=50)


def _tool_observations(call_messages):
    """从发给 LLM 的扁平消息里取出所有 role=tool 的 content。"""
    return [str(m.get("content", "") or "") for m in call_messages if m.get("role") == "tool"]


def _reset_guardrail_registry():
    from fast_agents import global_guardrail_registry
    global_guardrail_registry.clear_guardrails()


# ==================== 1. 同构性：arun == run ====================

async def test_arun_result_matches_run():
    """同一脚本分别喂同步 run 与异步 arun，最终答案与喂给 LLM 的消息完全一致。"""
    script_a = [Message("assistant", "同步答案：从 LangGraph 入手。")]
    script_b = [Message("assistant", "同步答案：从 LangGraph 入手。")]

    llm_run = MockAsyncLLM(script_a)
    agent_run = ReActAgent("助手", llm_run, memory=_make_memory(llm_run), enable_skills=False)
    sync_answer = agent_run.run("推荐学习路线")

    llm_arun = MockAsyncLLM(script_b)
    agent_arun = ReActAgent("助手", llm_arun, memory=_make_memory(llm_arun), enable_skills=False)
    async_answer = await agent_arun.arun("推荐学习路线")

    assert async_answer == sync_answer == "同步答案：从 LangGraph 入手。"
    assert llm_arun.async_stream_calls == 1, "异步链路没有走 llm.astream_invoke"
    assert llm_arun.sync_stream_calls == 0, "异步链路不应调同步 stream_invoke"
    # 两条链路喂给 LLM 的消息条数一致（同构性）
    assert len(llm_arun.calls[0]) == len(llm_run.calls[0])
    print("✅ test_arun_result_matches_run passed")


async def test_arun_tool_path_matches_run_and_writes_memory():
    """工具链路：异步 arun 与同步 run 的工具观察、记忆沉淀、system 条数一致。"""
    def _registry():
        reg = ToolRegistry()
        reg.register_tool(StructuredTool.from_function(
            func=lambda expression: str(eval(expression)),
            name="py_calc", description="计算表达式",
        ))
        return reg

    script_sync = [
        Message("assistant", "", tool_calls=[_tc("call_1", "py_calc", '{"expression": "2+3*4"}')]),
        Message("assistant", "结果是 14。"),
    ]
    llm_sync = MockAsyncLLM(script_sync)
    mem_sync = _make_memory(llm_sync)
    agent_sync = ReActAgent("助手", llm_sync, tool_registry=_registry(), memory=mem_sync,
                            enable_skills=False)
    answer_sync = agent_sync.run("2+3*4 等于多少")

    script_async = [
        Message("assistant", "", tool_calls=[_tc("call_1", "py_calc", '{"expression": "2+3*4"}')]),
        Message("assistant", "结果是 14。"),
    ]
    llm_async = MockAsyncLLM(script_async)
    mem_async = _make_memory(llm_async)
    agent_async = ReActAgent("助手", llm_async, tool_registry=_registry(), memory=mem_async,
                             enable_skills=False)
    answer_async = await agent_async.arun("2+3*4 等于多少")

    assert answer_async == answer_sync == "结果是 14。"
    obs_sync = _tool_observations(llm_sync.calls[1])
    obs_async = _tool_observations(llm_async.calls[1])
    assert obs_async == obs_sync == ["14"], f"工具观察文本应一致：{obs_async} vs {obs_sync}"
    assert len([m for m in agent_async.messages.get_messages() if m.role == "system"]) == 1
    assert len(mem_async.buffer) == len(mem_sync.buffer) == 2, "异步链路记忆沉淀应与同步一致"
    print("✅ test_arun_tool_path_matches_run_and_writes_memory passed")


# ==================== 2. 中间件 a* 钩子接通（A1 核心断言） ====================

async def test_async_path_only_awaits_async_hooks():
    """A1 核心：异步主循环必须 await 五个 a* 钩子（含本轮新补的 aafter_agent_turn），
    且绝不在事件循环里直跑同步钩子。"""
    probe = HookProbeMiddleware()
    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "noop", "{}")]),
        Message("assistant", "搞定"),
    ])
    reg = ToolRegistry()
    reg.register_tool(StructuredTool.from_function(func=lambda: "ok", name="noop", description="空操作"))
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, middleware=[probe])

    answer = await agent.arun("跑一下")

    assert answer == "搞定"
    assert probe.sync_calls == [], f"异步链路不应触发同步钩子：{probe.sync_calls}"
    assert probe.async_calls == [
        "abefore_agent_run", "abefore_tool_call", "aafter_tool_call",
        "aafter_agent_run", "aafter_agent_turn",
    ], f"a* 钩子调用序列不符：{probe.async_calls}"
    print("✅ test_async_path_only_awaits_async_hooks passed")


def test_sync_path_only_calls_sync_hooks():
    """非侵入验证：同步 run 仍只调同步钩子，一个 a* 都不碰（行为与改动前逐字一致）。"""
    probe = HookProbeMiddleware()
    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "noop", "{}")]),
        Message("assistant", "搞定"),
    ])
    reg = ToolRegistry()
    reg.register_tool(StructuredTool.from_function(func=lambda: "ok", name="noop", description="空操作"))
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, middleware=[probe])

    answer = agent.run("跑一下")

    assert answer == "搞定"
    assert probe.async_calls == [], f"同步链路不应触发异步钩子：{probe.async_calls}"
    assert probe.sync_calls == [
        "before_agent_run", "before_tool_call", "after_tool_call",
        "after_agent_run", "after_agent_turn",
    ], f"同步钩子调用序列不符：{probe.sync_calls}"
    print("✅ test_sync_path_only_calls_sync_hooks passed")


async def test_aafter_agent_turn_exception_does_not_break_answer():
    """R6：轮次收尾钩子抛异常只能记 warning，不得打挂已产出的答案。"""

    class _BoomTurn(AgentMiddleware):
        def __init__(self):
            self.called = 0

        async def aafter_agent_turn(self):
            self.called += 1
            raise RuntimeError("收尾炸了")

    boom = _BoomTurn()
    llm = MockAsyncLLM([Message("assistant", "仍然返回答案")])
    agent = ReActAgent("助手", llm, memory=_make_memory(llm), enable_skills=False,
                       middleware=[boom])
    answer = await agent.arun("你好")

    assert answer == "仍然返回答案"
    assert boom.called == 1
    print("✅ test_aafter_agent_turn_exception_does_not_break_answer passed")


async def test_middleware_to_thread_default_hooks_work():
    """只实现同步钩子的中间件挂到异步链路上：默认 a* 实现应经 to_thread 转发，
    既不能漏调，也不能在事件循环里直跑。"""

    class _SyncOnly(AgentMiddleware):
        def __init__(self):
            self.seen_thread_id = None

        def before_tool_call(self, tool_name, tool_args, context=None, **kwargs):
            self.seen_thread_id = threading.get_ident()
            return True

    mw = _SyncOnly()
    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "t1", "{}")]),
        Message("assistant", "done"),
    ])
    reg = ToolRegistry()
    reg.register_tool(StructuredTool.from_function(func=lambda: "v", name="t1", description="工具1"))
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, middleware=[mw])

    assert await agent.arun("跑") == "done"
    assert mw.seen_thread_id is not None, "默认 abefore_tool_call 未转发到同步实现"
    assert mw.seen_thread_id != threading.get_ident(), "to_thread 转发应离开事件循环线程"
    print("✅ test_middleware_to_thread_default_hooks_work passed")


# ==================== 3. Guardrails 异步链路端到端 ====================

class _BlockKeywordInput(InputGuardrail):
    name = "async_block_keyword_input"
    description = "含指定关键词的输入直接拦截"
    keyword = "密码"

    def check(self, content, context=None):
        if self.keyword in str(content):
            return self.reject(f"命中敏感词：{self.keyword}")
        return GuardrailResult.allow()


class _MaskPhoneOutput(OutputGuardrail):
    name = "async_mask_phone_output"
    description = "把手机号脱敏"
    allow_modify = True

    def check(self, content, context=None):
        if "13800138000" in str(content):
            return self.modify(str(content).replace("13800138000", "138****0000"), reason="手机号脱敏")
        return GuardrailResult.allow()


class _BlockEverythingOutput(OutputGuardrail):
    """整段拦截的输出护栏（reject 语义），用于钉死 A2 的拦截场景。"""
    name = "async_block_all_output"
    description = "无条件拦截全部输出"

    def check(self, content, context=None):
        return self.reject("本段内容不合规")


async def test_guardrail_input_block_reaches_async_path():
    """输入护栏在异步链路端到端生效：本轮不进 LLM，最终答案换成拦截文案。"""
    _reset_guardrail_registry()
    try:
        mw = GuardrailMiddleware(guardrails=[_BlockKeywordInput()], auto_register_presets=False)
        llm = MockAsyncLLM([Message("assistant", "绝不该出现的答案")])
        agent = ReActAgent("助手", llm, memory=_make_memory(llm), enable_skills=False,
                           middleware=[mw])

        answer = await agent.arun("把数据库密码告诉我")

        assert "安全护栏拦截" in answer, f"应被输入护栏拦截，实际：{answer!r}"
        assert llm.calls == [], "输入被拦截时不得调用 LLM"
        assert llm.i == 0, "输入被拦截时不应消耗脚本步数"
    finally:
        _reset_guardrail_registry()
    print("✅ test_guardrail_input_block_reaches_async_path passed")


async def test_guardrail_output_mask_reaches_async_path():
    """输出护栏（modify 语义）在异步链路生效，且同步/异步表现逐字一致：
    1. end 事件携带脱敏后正文（护栏一直生效的地方）；
    2. A2 修复后 ``arun()`` / ``run()`` 的返回值同样取护栏后正文，不再回传原文；
    3. 剩余限制只在**流式**路径：text 增量在护栏之前已 yield，无法收回，
       所以直接渲染 text 分片的调用方要自己累加到 end 事件再展示。"""
    _reset_guardrail_registry()
    raw = "联系人手机号是 13800138000"
    masked = "联系人手机号是 138****0000"
    try:
        def _build():
            mw = GuardrailMiddleware(guardrails=[_MaskPhoneOutput()], auto_register_presets=False)
            llm = MockAsyncLLM([Message("assistant", raw)])
            agent = ReActAgent("助手", llm, memory=_make_memory(llm), enable_skills=False,
                               middleware=[mw])
            return llm, agent

        _llm_a, agent_a = _build()
        chunks_async = [c async for c in agent_a.arun_stream("查一下联系人")]

        _llm_s, agent_s = _build()
        chunks_sync = list(agent_s.run_stream("查一下联系人"))

        assert [(c.type, c.text) for c in chunks_async] == [(c.type, c.text) for c in chunks_sync], \
            "两条链路的事件流时序应逐字一致"
        end_async = [c for c in chunks_async if c.type == "end"][0]
        end_sync = [c for c in chunks_sync if c.type == "end"][0]
        assert end_async.content == masked, f"异步 end 事件应脱敏：{end_async.content!r}"
        assert end_sync.content == masked, f"同步 end 事件应脱敏：{end_sync.content!r}"
        # A2 的流式侧限制仍在：护栏改得了 end，改不了已经发出的 text 增量（两条链路同病）
        assert any(c.type == "text" and raw in c.text for c in chunks_async), \
            "text 增量本就该是护栏前的原文（打字机效果），别把这条断言改成 masked"

        # A2 修复：非流式返回值以 end 正文为准，脱敏结果真正交到调用方手上
        _llm_a2, agent_a2 = _build()
        assert await agent_a2.arun("查一下联系人") == masked, "arun() 仍返回未脱敏原文（A2 未修）"
        _llm_s2, agent_s2 = _build()
        assert agent_s2.run("查一下联系人") == masked, "run() 仍返回未脱敏原文（A2 未修）"
    finally:
        _reset_guardrail_registry()
    print("✅ test_guardrail_output_mask_reaches_async_path passed")


async def test_guardrail_output_block_returns_block_message_both_paths():
    """A2 修复（拦截场景，同步/异步对称）：输出被整段拦截时
    ``run()`` / ``arun()`` 必须返回拦截文案且不得含模型原文；
    同时钉住 ``end.finish_reason == "guardrail_block"`` 这个供前端撤销展示的标记。

    拦截文案由 GuardrailMiddleware 统一加前缀（「输出被安全护栏拦截：」 + reject 原因），
    所以本条只断言「含拦截原因 + 不含原文」，不把前缀写死进断言。"""
    _reset_guardrail_registry()
    raw = "这整段都不应该被输出"
    blocked_reason = "本段内容不合规"
    try:
        def _build():
            mw = GuardrailMiddleware(guardrails=[_BlockEverythingOutput()], auto_register_presets=False)
            llm = MockAsyncLLM([Message("assistant", raw)])
            return ReActAgent("助手", llm, memory=_make_memory(llm), enable_skills=False,
                              middleware=[mw])

        answer_async = await _build().arun("讲点违规的")
        assert raw not in answer_async and blocked_reason in answer_async, \
            f"arun() 未返回拦截文案（A2 未修）：{answer_async!r}"
        answer_sync = _build().run("讲点违规的")
        assert answer_sync == answer_async, f"两条链路拦截结果应一致：{answer_sync!r}"

        # 流式侧：text 增量仍是原文（无法收回），但 end 事件带回了拦截文案 + 专用标记
        chunks = [c async for c in _build().arun_stream("讲点违规的")]
        assert any(c.type == "text" and raw in (c.text or "") for c in chunks), \
            "流式 text 增量本就该是护栏前的原文"
        end = [c for c in chunks if c.type == "end"][0]
        assert end.content == answer_sync, f"end 事件应为拦截文案：{end.content!r}"
        assert end.finish_reason == "guardrail_block", \
            f"拦截时 end.finish_reason 应为 guardrail_block，实际 {end.finish_reason!r}"
    finally:
        _reset_guardrail_registry()
    print("✅ test_guardrail_output_block_returns_block_message_both_paths passed")


async def test_guardrail_preset_blocks_tool_call_in_async_path():
    """预设权限护栏的高危命令拦截，在异步主循环里端到端生效：工具本体不执行。"""
    _reset_guardrail_registry()
    try:
        executed = []

        @tool
        def shell_exec(command: str) -> str:
            """执行 shell 命令"""
            executed.append(command)
            return f"ran: {command}"

        mw = GuardrailMiddleware(auto_register_presets=True)
        llm = MockAsyncLLM([
            Message("assistant", "", tool_calls=[
                _tc("c1", "shell_exec", '{"command": "rm -rf /"}')]),
            Message("assistant", "该操作被安全策略拒绝，我不再执行。"),
        ])
        reg = ToolRegistry()
        reg.register_tool(shell_exec)
        agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                           enable_skills=False, middleware=[mw])

        answer = await agent.arun("帮我清空根目录")

        assert executed == [], f"高危命令应被护栏拦截，实际执行了：{executed}"
        obs = _tool_observations(llm.calls[1])
        assert len(obs) == 1 and "拦截" in obs[0], f"第二轮应看到拦截 observation：{obs}"
        assert "被安全策略拒绝" in answer
    finally:
        _reset_guardrail_registry()
    print("✅ test_guardrail_preset_blocks_tool_call_in_async_path passed")


# ==================== 4. 工具异步执行语义 ====================

async def test_native_async_tool_is_awaited_not_run_in_thread():
    """@tool 装饰 async def：异步路径直接 await 本体（在事件循环线程里），
    同步路径显式报错给出路（E10-b），不再返回协程对象乱码。"""
    called = {"async_in_loop": None, "thread_id": None}

    @tool
    async def async_fetch(query: str) -> str:
        """异步检索工具"""
        called["thread_id"] = threading.get_ident()
        called["async_in_loop"] = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return f"result:{query}"

    reg = ToolRegistry()
    reg.register_tool(async_fetch)

    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "async_fetch", '{"query": "abc"}')]),
        Message("assistant", "拿到结果"),
    ])
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False)
    answer = await agent.arun("搜一下 abc")

    assert answer == "拿到结果"
    assert called["async_in_loop"] is not None, "async 工具未被 await"
    assert called["thread_id"] == threading.get_ident(), "原生 async 工具不应被丢进线程池"
    obs = _tool_observations(llm.calls[1])
    assert obs == ["result:abc"], f"async 工具结果应进 observation：{obs}"

    # 同步路径：显式失败并给出出路，而不是把协程对象当结果
    llm2 = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "async_fetch", '{"query": "abc"}')]),
        Message("assistant", "同步路径拿不到结果"),
    ])
    agent2 = ReActAgent("助手", llm2, tool_registry=reg, memory=_make_memory(llm2),
                        enable_skills=False)
    answer2 = agent2.run("搜一下 abc")
    obs2 = _tool_observations(llm2.calls[1])
    assert answer2 == "同步路径拿不到结果"
    assert "async 函数" in obs2[0] and "<coroutine object" not in obs2[0], \
        f"同步路径应明确报错给出出路：{obs2}"
    print("✅ test_native_async_tool_is_awaited_not_run_in_thread passed")


async def test_sync_only_tool_downgrades_to_thread():
    """只实现同步 run() 的工具走异步路径：自动降级 asyncio.to_thread，不堵事件循环。"""
    seen = {}

    class _SyncTool(ToolBase):
        name = "sync_only"
        description = "只有同步实现的工具"

        def run(self, text: str = "") -> str:
            seen["thread_id"] = threading.get_ident()
            time.sleep(0.02)
            return f"echo:{text}"

    reg = ToolRegistry()
    reg.register_tool(_SyncTool())
    result = await reg.ainvoke_tool("sync_only", {"text": "hi"})

    assert result.succeeded, result.error
    assert result.output == "echo:hi"
    assert seen["thread_id"] != threading.get_ident(), "同步工具应在线程池里执行"
    assert "duration_ms" in result.metadata, "异步入口应保留与同步版一致的耗时元数据"
    print("✅ test_sync_only_tool_downgrades_to_thread passed")


async def test_ainvoke_tool_timeout_marks_abandoned():
    """ainvoke_tool 超时语义与同步版一致：放弃等待 + metadata['timeout_abandoned']=True。"""
    @tool
    async def slow(q: str) -> str:
        """慢工具"""
        await asyncio.sleep(5)
        return "不该返回"

    reg = ToolRegistry(default_timeout=0.1)
    reg.register_tool(slow)
    result = await reg.ainvoke_tool("slow", {"q": "x"})

    assert result.failed
    assert "超时" in result.error, f"应提示超时：{result.error}"
    assert result.metadata.get("timeout_abandoned") is True
    print("✅ test_ainvoke_tool_timeout_marks_abandoned passed")


async def test_ainvoke_tool_not_found_and_hooks():
    """ainvoke_tool 的「未找到工具」文案与 hooks 触发行为逐字对齐同步版。"""
    reg = ToolRegistry()
    missing = await reg.ainvoke_tool("不存在的工具", {})
    assert not missing.succeeded
    assert "未找到名为 '不存在的工具' 的工具" in missing.output, missing.output

    seen = []
    reg.add_hook(lambda name, args, result: seen.append((name, dict(args), result.output)))

    @tool
    def echo(msg: str) -> str:
        """回显"""
        return f"E:{msg}"

    reg.register_tool(echo)
    ok = await reg.ainvoke_tool("echo", {"msg": "hi"})
    assert ok.succeeded and ok.output == "E:hi"
    assert seen == [("echo", {"msg": "hi"}, "E:hi")], f"hooks 应在调用方上下文触发一次：{seen}"
    print("✅ test_ainvoke_tool_not_found_and_hooks passed")


async def test_ainvoke_tools_concurrent_positions_and_duplicate_tool():
    """ainvoke_tools_concurrent：结果与 calls 严格一一对应，同一工具重复出现各占一位。"""
    reg = ToolRegistry()

    @tool
    async def add(a: int, b: int) -> int:
        """求和"""
        await asyncio.sleep(0.01 * (3 - a))  # 让先提交的更晚完成，验证顺序不受完成先后影响
        return a + b

    reg.register_tool(add)
    results = await reg.ainvoke_tools_concurrent([
        ("add", {"a": 1, "b": 10}),
        ("add", {"a": 3, "b": 30}),
        ("add", {"a": 2, "b": 20}),
        ("no_such", {}),
    ])

    assert len(results) == 4
    assert [r.output for r in results[:3]] == ["11", "33", "22"], \
        f"结果顺序必须与 calls 对齐：{[r.output for r in results]}"
    assert results[3].failed and "未找到名为" in results[3].output
    print("✅ test_ainvoke_tools_concurrent_positions_and_duplicate_tool passed")


async def test_ainvoke_tools_concurrent_respects_approval_callback():
    """审批回调在异步并发路径同样前置生效：被拒工具不进任务表、本体不执行。"""
    reg = ToolRegistry()
    executed = []

    @tool
    def drop(table: str) -> str:
        """删表"""
        executed.append(table)
        return f"dropped:{table}"

    drop.metadata["requires_approval"] = True
    reg.register_tool(drop)
    reg.set_approval_callback(lambda name, args: (False, None, "运维未批准"))

    results = await reg.ainvoke_tools_concurrent([("drop", {"table": "users"})])
    assert results[0].failed and "执行被拒绝" in results[0].output, results[0].output
    assert executed == [], "被拒工具不得执行"

    # 单工具入口同样走审批
    single = await reg.ainvoke_tool("drop", {"table": "users"})
    assert single.failed and "执行被拒绝" in single.output
    assert executed == []
    print("✅ test_ainvoke_tools_concurrent_respects_approval_callback passed")


# ==================== 5. 并发 / 惰性短路 ====================

async def test_parallel_tool_calls_run_concurrently_in_async_path():
    """parallel_tool_calls=True：异步分支一次性并发，总耗时接近最慢单个工具。"""
    reg = ToolRegistry()

    @tool
    async def nap(ms: float) -> str:
        """睡一会儿"""
        await asyncio.sleep(ms / 1000)
        return f"done:{int(ms)}"

    reg.register_tool(nap)
    script = [
        Message("assistant", "", tool_calls=[
            _tc("c1", "nap", '{"ms": 250}'),
            _tc("c2", "nap", '{"ms": 250}'),
        ]),
        Message("assistant", "两个都完成"),
    ]
    llm = MockAsyncLLM(script)
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, parallel_tool_calls=True)

    started = time.perf_counter()
    answer = await agent.arun("并行跑")
    elapsed = time.perf_counter() - started

    assert answer == "两个都完成"
    assert elapsed < 0.45, f"并发未生效，耗时 {elapsed:.3f}s（顺序执行应约 0.5s）"
    obs = _tool_observations(llm.calls[1])
    assert obs == ["done:250", "done:250"], f"两个工具都应执行且顺序对齐：{obs}"
    print("✅ test_parallel_tool_calls_run_concurrently_in_async_path passed")


async def test_return_direct_skips_next_round_with_sync_parity():
    """return_direct 语义对齐：命中后**不再进下一轮推理**、直接拿工具结果作最终答案；
    同批其余工具仍会被执行完（这是同步版既有语义，异步路径逐字保留，不擅自修正）。
    """
    def _build():
        executed = []

        class _First(ToolBase):
            name = "first"
            description = "直接返回型工具"
            return_direct = True

            def run(self) -> str:
                executed.append("first")
                return "直达答案"

        class _Second(ToolBase):
            name = "second"
            description = "普通工具"

            def run(self) -> str:
                executed.append("second")
                return "同步执行的其余工具"

        reg = ToolRegistry()
        reg.register_tool(_First())
        reg.register_tool(_Second())
        script = [Message("assistant", "", tool_calls=[
            _tc("c1", "first", "{}"),
            _tc("c2", "second", "{}"),
        ])]
        llm = MockAsyncLLM(script)
        agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                           enable_skills=False)
        return executed, llm, agent

    executed_async, llm_async, agent_async = _build()
    answer_async = await agent_async.arun("先跑 first")
    assert answer_async == "直达答案"
    assert executed_async == ["first", "second"], f"同批工具应全部执行：{executed_async}"
    assert llm_async.i == 1, "return_direct 后不应再进下一轮推理"

    executed_sync, llm_sync, agent_sync = _build()
    answer_sync = agent_sync.run("先跑 first")
    assert answer_sync == answer_async == "直达答案"
    assert executed_sync == executed_async, f"两条链路工具执行集合应一致：{executed_sync}"
    assert llm_sync.i == llm_async.i == 1
    print("✅ test_return_direct_skips_next_round_with_sync_parity passed")


async def test_return_direct_answer_still_passes_output_guardrail():
    """C3：return_direct 的工具原始输出同样要过输出护栏，不得绕过。"""
    _reset_guardrail_registry()
    try:
        class _Leak(ToolBase):
            name = "leaky"
            description = "返回敏感串的工具"
            return_direct = True

            def run(self) -> str:
                return "密钥是 13800138000"

        reg = ToolRegistry()
        reg.register_tool(_Leak())
        mw = GuardrailMiddleware(guardrails=[_MaskPhoneOutput()], auto_register_presets=False)
        llm = MockAsyncLLM([Message("assistant", "", tool_calls=[_tc("c1", "leaky", "{}")])])
        agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                           enable_skills=False, middleware=[mw])

        answer = await agent.arun("取密钥")
        assert answer == "密钥是 138****0000", f"return_direct 输出未过护栏：{answer!r}"
    finally:
        _reset_guardrail_registry()
    print("✅ test_return_direct_answer_still_passes_output_guardrail passed")


async def test_blocked_tool_placeholder_covers_rest_of_batch():
    """N2：本批任一工具被拦则整批不执行，未执行的 tool_call 必须补占位 tool 消息。"""
    probe = HookProbeMiddleware(block_tool_names=("dangerous",))
    executed = []

    @tool
    def dangerous(sql: str) -> str:
        """危险操作"""
        executed.append(sql)
        return "ran"

    @tool
    def safe_read(path: str) -> str:
        """读文件"""
        executed.append(path)
        return "read"

    reg = ToolRegistry()
    reg.register_tool(dangerous)
    reg.register_tool(safe_read)

    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[
            _tc("c1", "dangerous", '{"sql": "drop table t"}'),
            _tc("c2", "safe_read", '{"path": "a.txt"}'),
        ]),
        Message("assistant", "已跳过"),
    ])
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, middleware=[probe])

    answer = await agent.arun("执行这两个操作")
    assert answer == "已跳过"
    assert executed == [], f"被拦批次不得执行任何工具：{executed}"
    obs = _tool_observations(llm.calls[1])
    assert len(obs) == 2, f"两个 tool_call 都应有对应消息：{obs}"
    assert "参数命中黑名单" in obs[0], f"拦截原因应随 observation 携带：{obs[0]}"
    assert "未执行" in obs[1] and "safe_read" in obs[1], f"第二个应补占位：{obs[1]}"
    print("✅ test_blocked_tool_placeholder_covers_rest_of_batch passed")


async def test_tool_result_masked_by_async_after_tool_hook():
    """工具后置护栏（返回 str 替换 observation）在异步链路生效。"""
    probe = HookProbeMiddleware(mask_observation="【已脱敏】")
    executed = []

    @tool
    def read_secret(name: str) -> str:
        """读敏感值"""
        executed.append(name)
        return "super-secret-value"

    reg = ToolRegistry()
    reg.register_tool(read_secret)
    llm = MockAsyncLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "read_secret", '{"name": "token"}')]),
        Message("assistant", "好"),
    ])
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False, middleware=[probe])

    assert await agent.arun("读 token") == "好"
    obs = _tool_observations(llm.calls[1])
    assert obs == ["【已脱敏】"], f"observation 应被异步后置钩子替换：{obs}"
    assert "super-secret-value" not in str(llm.calls[1])
    print("✅ test_tool_result_masked_by_async_after_tool_hook passed")


# ==================== 6. 回退分支与基类默认实现 ====================

async def test_llm_without_astream_invoke_falls_back():
    """注入只实现同步 stream_invoke 的 duck-typed LLM：异步主循环回退直跑，不丢块不错序。"""
    llm = MockSyncOnlyLLM([
        Message("assistant", "", tool_calls=[_tc("c1", "t1", "{}")]),
        Message("assistant", "回退成功"),
    ])
    reg = ToolRegistry()
    reg.register_tool(StructuredTool.from_function(func=lambda: "v", name="t1", description="工具1"))
    agent = ReActAgent("助手", llm, tool_registry=reg, memory=_make_memory(llm),
                       enable_skills=False)

    answer = await agent.arun("跑")
    assert answer == "回退成功"
    assert llm.sync_stream_calls == 2, "回退分支应逐轮使用同步 stream_invoke"
    print("✅ test_llm_without_astream_invoke_falls_back passed")


async def test_structured_tool_dual_impl_prefers_coroutine():
    """同时给 func(同步) + coroutine(异步) 时：同步走 func、异步走 coroutine，互不串。"""
    tool_obj = StructuredTool.from_function(
        func=lambda q: f"sync:{q}",
        name="dual",
        description="双实现工具",
        coroutine=lambda q: asyncio.sleep(0, result=f"async:{q}"),
    )
    reg = ToolRegistry()
    reg.register_tool(tool_obj)

    assert reg.invoke_tool("dual", {"q": "x"}).output == "sync:x"
    assert (await reg.ainvoke_tool("dual", {"q": "x"})).output == "async:x"
    print("✅ test_structured_tool_dual_impl_prefers_coroutine passed")


async def test_agent_base_async_defaults_are_not_abstract():
    """AgentBase.arun/arun_stream 必须**非抽象**：只实现同步接口的老子类仍可实例化，
    且默认异步入口能透传同步事件、聚合出与 run 一致的答案。"""
    assert not getattr(AgentBase.arun, "__isabstractmethod__", False), "arun 不能是抽象方法"
    assert not getattr(AgentBase.arun_stream, "__isabstractmethod__", False), "arun_stream 不能是抽象方法"

    class _LegacyAgent(AgentBase):
        """改动前就存在的、只实现同步接口的子类。"""

        def __init__(self):
            super().__init__("legacy", llm=None)
            self.stream_seen = 0

        def run(self, input_text, **kwargs) -> str:
            return "".join(c.text for c in self.run_stream(input_text, **kwargs) if c.type == "text")

        def run_stream(self, input_text, **kwargs):
            self.stream_seen += 1
            yield StreamChunk(type="text", text="-parts:")
            yield StreamChunk(type="text", text=input_text)
            yield StreamChunk(type="end", content="end兜底")

    agent = _LegacyAgent()
    sync_answer = agent.run("hello")
    async_answer = await agent.arun("hello")

    assert sync_answer == "-parts:hello"
    assert async_answer == sync_answer, f"默认 arun 应与 run 聚合结果一致：{async_answer!r}"
    assert agent.stream_seen == 2
    print("✅ test_agent_base_async_defaults_are_not_abstract passed")


def test_existing_sync_only_subclass_still_instantiable():
    """R1 回归：未覆盖异步方法的 AgentBase 子类实例化不报 TypeError。"""
    class _Minimal(AgentBase):
        def run(self, input_text, **kwargs) -> str:
            return "x"

        def run_stream(self, input_text, **kwargs):
            yield StreamChunk(type="end", content="x")

    agent = _Minimal("最小", None)   # 不传任何异步实现也能实例化
    assert agent.run("q") == "x"
    print("✅ test_existing_sync_only_subclass_still_instantiable passed")


# ==================== 入口 ====================

def main():
    """跑全部测试（tests/_runner.py 会自动识别 async 测试并 asyncio.run 执行）。"""
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """ 新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    # ⚠ 本文件用例多为 async def，直接调用只会得到未 await 的协程；
    #   tests/_runner.py 会自动 asyncio.run，这里手工调用需自己包一层。
    asyncio.run(test_guardrail_output_mask_reaches_async_path())  # A2 修复：arun()/run() 返回护栏后正文
    asyncio.run(test_guardrail_output_block_returns_block_message_both_paths())  # A2 修复：拦截场景同步/异步对称
    asyncio.run(test_async_path_only_awaits_async_hooks())
    asyncio.run(test_arun_result_matches_run())
    asyncio.run(test_guardrail_preset_blocks_tool_call_in_async_path())
    asyncio.run(test_native_async_tool_is_awaited_not_run_in_thread())
    asyncio.run(test_return_direct_skips_next_round_with_sync_parity())
    asyncio.run(test_agent_base_async_defaults_are_not_abstract())


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
