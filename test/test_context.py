"""context 模块测试：不依赖真实 LLM、不依赖网络

覆盖（15 项）：
1.  TokenCounter 的启发式估算与消息计数
2.  TokenBudget 的预算换算与描述
3.  ContextWindow 的分层顺序与 flatten
4.  ContextWindow 的 tool 配对校验
5.  ContextOffloader 的阈值判断（不足阈值不落盘）
6.  ContextOffloader 落盘 + 索引 + 回读
7.  ContextOffloader 的行区间回读
8.  ContextOffloader 的文件数上限淘汰
9.  ContextOffloader 的非法 id / 路径安全
10. ContextCompressor 第 2 级：裁剪最老历史且不拆 tool 配对
11. ContextCompressor 第 1 级：超长消息被卸载成引用
12. ContextCompressor 第 3 级：LLM 摘要替换历史
13. ContextBuilder 的完整分层拼装顺序
14. ContextBuilder 的预算裁剪（历史被压掉）
15. context 工具：write / edit / list / read_offloaded
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.context import (
    ContextBuilder,
    ContextOffloader,
    ContextWindow,
    ContextCompressor,
    ContextLayer,
    TokenBudget,
    TokenCounter,
    EditFileTool,
    ListFilesTool,
    ReadOffloadedTool,
    WriteFileTool,
    build_context_tools,
    estimate_text_tokens,
)
from fast_agents.messages.message import Message, MsgToolCall


# ============================================================
# 测试辅助
# ============================================================

class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """假 LLM：invoke 永远返回固定摘要文本"""

    def __init__(self, reply="（这是摘要）"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return FakeResponse(self.reply)


def _msg(role, content, tool_calls=None, metadata=None):
    """构造 Message；注意 Message.__init__ 只透传 tool_calls / timestamp / metadata"""
    kwargs = {}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    if metadata:
        kwargs["metadata"] = metadata
    return Message(role, content, **kwargs)


def _tool_call(cid="call_1", name="calc", arguments="{}"):
    """构造一个合法的 MsgToolCall"""
    return MsgToolCall(id=cid, name=name, arguments=arguments)


# ============================================================
# 1. TokenCounter
# ============================================================

def test_token_counter():
    counter = TokenCounter(use_tiktoken=False)
    assert counter.backend == "heuristic", counter.backend

    # 中文：一字 ≈ 1 token（含安全系数，会略大于字数）
    zh = counter.count("你好世界")
    assert 4 <= zh <= 8, f"中文计数异常: {zh}"

    # 英文：4 字符 ≈ 1 token
    en = counter.count("hello world this is a test")
    assert 5 <= en <= 12, f"英文计数异常: {en}"

    # 空串必须是 0
    assert counter.count("") == 0, "空串应当为 0"

    # 纯函数版本与实例一致（同为启发式时）
    assert estimate_text_tokens("你好世界", 1.0) >= 4

    # 消息计数：含 role 与固定开销
    n = counter.count_messages([{"role": "user", "content": "你好"}])
    assert n >= 4, f"消息计数异常: {n}"

    # 工具 schema 计数
    schema_tokens = counter.count_tool_schemas([
        {"type": "function", "function": {"name": "calc", "description": "计算", "parameters": {}}}
    ])
    assert schema_tokens > 0, "工具 schema 计数应当 > 0"

    print("✅ test_token_counter 通过")


# ============================================================
# 2. TokenBudget
# ============================================================

def test_token_budget():
    budget = TokenBudget(max_context_tokens=10_000, reserve_for_output=1_000, safety_margin=500)
    assert budget.available == 8_500, budget.available
    assert budget.remaining(3_500) == 5_000
    assert budget.is_over(8_600) is True
    assert budget.is_over(8_400) is False
    assert 0.0 < budget.usage_ratio(4_250) < 1.0
    assert "8500" in budget.describe(1_000).replace("_", "")

    # 极端配置也不允许出现 0 或负数预算
    tiny = TokenBudget(max_context_tokens=100, reserve_for_output=500)
    assert tiny.available >= 1, "可用预算下限保护失效"

    print("✅ test_token_budget 通过")


# ============================================================
# 3. ContextWindow 分层顺序
# ============================================================

def test_window_layer_order():
    window = ContextWindow()
    # 故意乱序添加，验证 flatten 会按「静态→动态」重排
    window.append_message("user", _msg("user", "用户问题"))
    window.append_message("system", _msg("system", "你是助手"))
    window.append_message("memory", _msg("user", "[记忆] 用户偏好中文"))
    window.append_message("history", _msg("assistant", "上一轮回答"))

    flat = window.flatten()
    roles = [(m.role, m.content) for m in flat]
    assert roles[0][1] == "你是助手", f"system 必须排最前: {roles}"
    assert roles[1][1].startswith("[记忆]"), f"memory 应排在 system 之后: {roles}"
    assert roles[2][1] == "上一轮回答", f"history 应在 memory 之后: {roles}"
    assert roles[-1][1] == "用户问题", f"user 必须排最后: {roles}"

    # OpenAI 格式输出长度 = 消息条数
    assert len(window.to_openai_messages()) == 4
    # token 统计与层统计
    assert window.token_count(TokenCounter(use_tiktoken=False)) > 0
    stats = window.layer_stats(TokenCounter(use_tiktoken=False))
    assert [s["name"] for s in stats] == ["system", "memory", "history", "user"], stats

    print("✅ test_window_layer_order 通过")


# ============================================================
# 4. ContextWindow tool 配对校验
# ============================================================

def test_window_validate_tool_pairing():
    ok_window = ContextWindow()
    ok_window.append_message(
        "history",
        _msg("assistant", "", tool_calls=[_tool_call()]),
        priority=60,
    )
    ok_window.append_message("history", _msg("tool", "结果", metadata={"tool_call_id": "call_1"}))
    assert ok_window.validate() == [], ok_window.validate()

    # 孤立 tool 消息应当被检出
    bad_window = ContextWindow()
    bad_window.append_message("history", _msg("tool", "孤立结果"))
    problems = bad_window.validate()
    assert len(problems) == 1, problems
    assert "tool" in problems[0]

    print("✅ test_window_validate_tool_pairing 通过")


# ============================================================
# 5-9. ContextOffloader
# ============================================================

def test_offloader_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=100)
        short = "短文本"
        assert offloader.should_offload(short) is False
        # 未超阈值时原样返回，且不产生文件
        assert offloader.offload("calc", short) == short
        assert len(offloader) == 0, "未超阈值不应产生卸载记录"
    print("✅ test_offloader_threshold 通过")


def test_offloader_offload_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=50, preview_chars=20)
        long_text = "行1\n" + "\n".join(f"内容{i}" for i in range(100))
        reference = offloader.offload("web_search", long_text)

        # 引用文本必须给出 id 与回读提示，且不含全文
        assert "已卸载" in reference
        assert "offload_id=" in reference
        assert "read_offloaded" in reference
        assert len(reference) < len(long_text), "引用文本应当显著短于原文"

        # 正文确实落盘
        record = offloader.get_record(offloader.last_id)
        assert record is not None
        assert record.tool_name == "web_search"
        assert record.original_chars == len(long_text)
        assert os.path.exists(record.path)
        assert len(offloader) == 1

        # 回读能拿到原文（截断到行区间内）
        content = offloader.read(offloader.last_id, max_lines=200)
        assert "行1" in content, content[:200]
        assert "仅作参考" not in content
    print("✅ test_offloader_offload_and_read 通过")


def test_offloader_read_line_range():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=10)
        text = "\n".join(f"第{i}行" for i in range(1, 51))
        offloader.offload("terminal", text)
        picked = offloader.read(offloader.last_id, start_line=10, max_lines=3)
        assert "第10行" in picked and "第12行" in picked, picked
        assert "第13行" not in picked, "行区间未生效"
        assert "未读完" in picked, "应当提示还有后续内容"
    print("✅ test_offloader_read_line_range 通过")


def test_offloader_max_files_eviction():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=10, max_files=3)
        ids = []
        for i in range(5):
            offloader.offload("tool", f"这是一段足够长的内容，编号 {i}，" + "x" * 50)
            ids.append(offloader.last_id)
        assert len(offloader) == 3, f"应当只保留 3 个文件，实际 {len(offloader)}"
        # 最早的应被淘汰
        assert offloader.get_record(ids[0]) is None, "最早的卸载记录应被淘汰"
        assert offloader.get_record(ids[-1]) is not None
    print("✅ test_offloader_max_files_eviction 通过")


def test_offloader_safety():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=10)
        # 非法 id：路径穿越字符必须被拒绝
        bad = offloader.read("../../etc/passwd")
        assert "非法" in bad, bad
        # 不存在的 id：返回可读错误而非抛异常
        missing = offloader.read("deadbeefdeadbeef")
        assert "未找到" in missing, missing
        # 卸载目录必须在 workspace 内
        assert str(offloader.root).startswith(str(offloader.workspace))
    print("✅ test_offloader_safety 通过")


# ============================================================
# 10-12. ContextCompressor
# ============================================================

def test_compressor_trim_keeps_tool_pairing():
    with tempfile.TemporaryDirectory() as tmp:
        counter = TokenCounter(use_tiktoken=False)
        compressor = ContextCompressor(enable_offload=False, enable_summary=False, keep_recent=1)

        layer = ContextLayer(name="history")
        # 造 10 轮历史，其中第 2 轮带 tool 配对
        for i in range(10):
            layer.add(_msg("user", f"问题{i}" + "字" * 30))
            layer.add(_msg("assistant", f"回答{i}" + "字" * 30))
        # 手工插入一组 tool 配对（放在最前面，最可能被裁到）
        layer.messages.insert(0, _msg("tool", "工具结果", metadata={"tool_call_id": "c1"}))
        layer.messages.insert(0, _msg("assistant", "", tool_calls=[_tool_call("c1")]))

        report = compressor.compress(layer, counter, 200)
        assert report.trimmed > 0, "应当发生裁剪"
        assert layer.token_count(counter) <= 200 or len(layer) <= 2

        # 关键断言：剩余消息里不允许出现「孤立的 tool 消息」
        previous_has_calls = False
        for m in layer.messages:
            if m.role == "tool":
                assert previous_has_calls, "裁剪后出现孤立 tool 消息（配对被拆坏）"
            previous_has_calls = bool(m.tool_calls)
    print("✅ test_compressor_trim_keeps_tool_pairing 通过")


def test_compressor_offload_level():
    with tempfile.TemporaryDirectory() as tmp:
        counter = TokenCounter(use_tiktoken=False)
        offloader = ContextOffloader(workspace=tmp, threshold=100)
        compressor = ContextCompressor(
            offloader=offloader, enable_trim=False, enable_summary=False
        )

        layer = ContextLayer(name="history")
        layer.add(_msg("tool", "超长内容：" + "啊" * 5_000, metadata={"name": "read_file"}))
        report = compressor.compress(layer, counter, 100)

        assert report.offloaded == 1, report
        assert "已卸载" in layer.messages[0].content, layer.messages[0].content[:100]
        assert len(offloader) == 1
    print("✅ test_compressor_offload_level 通过")


def test_compressor_summary_level():
    counter = TokenCounter(use_tiktoken=False)
    llm = FakeLLM("【压缩后的摘要】用户在调研上下文工程")
    compressor = ContextCompressor(
        llm=llm, enable_offload=False, enable_trim=False, enable_summary=True, keep_recent=1
    )

    layer = ContextLayer(name="history")
    for i in range(8):
        layer.add(_msg("user", f"很长的问题{i}" + "字" * 100))
        layer.add(_msg("assistant", f"很长的回答{i}" + "字" * 100))

    report = compressor.compress(layer, counter, 200)
    assert report.summarized is True, "应当触发摘要"
    assert llm.calls, "摘要必须真的调用了 LLM"

    # 摘要以 required 的 system 消息形式放回层里
    first = layer.messages[0]
    assert first.role == "system", first.role
    assert "压缩后的摘要" in first.content
    assert compressor.accumulated_summary.startswith("【压缩后的摘要】")
    # 最近的历史必须保留
    assert layer.messages[-1].content.startswith("很长的回答7")
    print("✅ test_compressor_summary_level 通过")


# ============================================================
# 13-14. ContextBuilder
# ============================================================

def test_builder_pipeline_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        builder = ContextBuilder(
            workspace=tmp,
            budget=TokenBudget(max_context_tokens=100_000),
            counter=TokenCounter(use_tiktoken=False),
            llm=FakeLLM(),
        )
        window = builder.build(
            system_prompt="你是 FastAgents 助手",
            history=[
                _msg("user", "我之前问过天气"),
                _msg("assistant", "今天晴"),
            ],
            user_input="帮我总结一下",
            memory_packets=["[preference] 用户偏好中文回答"],
        )
        flat = window.flatten()
        assert flat[0].role == "system" and flat[0].content.startswith("你是")
        assert flat[-1].content == "帮我总结一下"
        assert any("[preference]" in (m.content or "") for m in flat), "记忆包未注入"
        assert builder.last_report.tokens > 0
        stats = builder.stats()
        assert stats["usage_ratio"] < 1.0
        assert "system" in stats["layers"] and "user" in stats["layers"]
        # 历史里的 system 不应被重复带进来
        assert sum(1 for m in flat if m.role == "system") == 1, "system 重复注入"
    print("✅ test_builder_pipeline_and_order 通过")


def test_builder_budget_trims():
    with tempfile.TemporaryDirectory() as tmp:
        # 故意给一个极小的窗口，逼出裁剪
        builder = ContextBuilder(
            workspace=tmp,
            budget=TokenBudget(max_context_tokens=2_000, reserve_for_output=500, safety_margin=100),
            counter=TokenCounter(use_tiktoken=False),
            llm=FakeLLM("摘要"),
            layer_quotas={"system": 0.2, "memory": 0.1, "files": 0.1},
        )
        long_history = []
        for i in range(30):
            long_history.append(_msg("user", f"问题{i}" + "字" * 100))
            long_history.append(_msg("assistant", f"回答{i}" + "字" * 100))

        window = builder.build(
            system_prompt="你是助手",
            history=long_history,
            user_input="最后一问",
        )
        # system 与 user 必须活着
        roles = [m.role for m in window.flatten()]
        assert roles[0] == "system", roles
        assert window.flatten()[-1].content == "最后一问"
        # 总 token 不能超过预算
        assert window.token_count(builder.counter) <= builder.budget.available, (
            f"{window.token_count(builder.counter)} > {builder.budget.available}"
        )
        # 历史被压掉了一部分
        assert builder.last_report.compression is not None
    print("✅ test_builder_budget_trims 通过")


# ============================================================
# 15. context 工具
# ============================================================

def test_context_filesystem_tools():
    with tempfile.TemporaryDirectory() as tmp:
        write_tool = WriteFileTool(tmp)
        list_tool = ListFilesTool(tmp)
        edit_tool = EditFileTool(tmp)

        # 写文件
        r = write_tool.invoke({"path": "notes/a.md", "content": "第一行\n第二行\n"})
        assert r.succeeded, r.error
        assert os.path.exists(os.path.join(tmp, "notes", "a.md"))

        # 列目录
        r = list_tool.invoke({"path": "notes"})
        assert r.succeeded and "a.md" in r.output, r.output

        # 编辑文件（精确替换）
        r = edit_tool.invoke({"path": "notes/a.md", "old_text": "第二行", "new_text": "第二行(改)"})
        assert r.succeeded, r.error
        content = open(os.path.join(tmp, "notes", "a.md"), encoding="utf-8").read()
        assert "第二行(改)" in content

        # 替换不到时必须给出可读失败，而不是静默成功
        r = edit_tool.invoke({"path": "notes/a.md", "old_text": "不存在的文本", "new_text": "x"})
        assert not r.succeeded, "不存在的原文应当失败"
        assert "未找到" in (r.error or ""), r.error

        # 沙箱：越界路径必须拒绝
        r = write_tool.invoke({"path": "../escape.txt", "content": "x"})
        assert not r.succeeded and "工作目录外" in (r.error or ""), r.error
    print("✅ test_context_filesystem_tools 通过")


def test_read_offloaded_tool_and_factory():
    with tempfile.TemporaryDirectory() as tmp:
        offloader = ContextOffloader(workspace=tmp, threshold=20)
        text = "\n".join(f"数据行{i}" for i in range(1, 31))
        offloader.offload("web_search", text)

        tool = ReadOffloadedTool(offloader)
        r = tool.invoke({"offload_id": offloader.last_id, "start_line": 5, "max_lines": 2})
        assert r.succeeded, r.error
        assert "数据行5" in r.output, r.output

        # 不存在的 id → 失败而不是异常
        r = tool.invoke({"offload_id": "ffffffffffffffff"})
        assert not r.succeeded, "不存在的 id 应当失败"

        # 工厂：传 offloader 时包含 read_offloaded + 文件三件套
        tools = build_context_tools(workspace=tmp, offloader=offloader)
        names = {t.name for t in tools}
        assert names == {"read_offloaded", "list_files", "write_file", "edit_file"}, names

        # 工具 schema 能被正常导出（说明 args_schema 定义合法）
        for t in tools:
            schema = t.to_openai_tool()
            assert schema["function"]["name"] == t.name
    print("✅ test_read_offloaded_tool_and_factory 通过")


# ============================================================
# 入口
# ============================================================

def main():
    tests = [
        test_token_counter,
        test_token_budget,
        test_window_layer_order,
        test_window_validate_tool_pairing,
        test_offloader_threshold,
        test_offloader_offload_and_read,
        test_offloader_read_line_range,
        test_offloader_max_files_eviction,
        test_offloader_safety,
        test_compressor_trim_keeps_tool_pairing,
        test_compressor_offload_level,
        test_compressor_summary_level,
        test_builder_pipeline_and_order,
        test_builder_budget_trims,
        test_context_filesystem_tools,
        test_read_offloaded_tool_and_factory,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n{'=' * 60}")
    print(f"✅ context 模块测试全部通过：{passed}/{len(tests)} 项")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
