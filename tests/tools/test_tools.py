"""Tools 模块单元测试

运行：python test/test_tools.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools import (
    ToolBase, StructuredTool, tool,
    ToolResult, ToolStatus, ToolException, ToolRegistry,
    builtin,
)
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# 1. 四种工具定义方式
# ─────────────────────────────────────────────────────────

def test_1_class_inheritance():
    """方式1：继承 ToolBase 定义工具"""
    class GreetInput(BaseModel):
        name: str = Field(description="姓名")

    class GreetTool(ToolBase):
        name = "greet"
        description = "打招呼"
        args_schema = GreetInput

        def run(self, name: str):
            return f"Hello, {name}!"

    t = GreetTool()
    schema = t.input_schema
    assert "name" in schema["properties"]
    assert "name" in schema["required"]

    r = t.invoke(name="world")
    assert r.succeeded
    assert r.output == "Hello, world!"
    print("✅ 方式1 继承 ToolBase 通过")


def test_2_decorator():
    """方式2：@tool 装饰器"""
    @tool
    def echo(text: str) -> str:
        """回显文本
        Args:
            text: 要回显的文本
        """
        return f"echo: {text}"

    assert echo.name == "echo"
    assert "回显文本" in echo.description
    schema = echo.input_schema
    assert "text" in schema["properties"]

    r = echo.invoke(text="hi")
    assert r.output == "echo: hi"
    print("✅ 方式2 @tool 装饰器通过")


def test_3_from_function():
    """方式3：StructuredTool.from_function"""
    def add(a: int, b: int = 0) -> int:
        """加法
        Args:
            a: 第一个数
            b: 第二个数
        """
        return a + b

    t = StructuredTool.from_function(add)
    assert t.name == "add"
    r = t.invoke(a=3, b=4)
    assert r.output == "7"
    print("✅ 方式3 StructuredTool.from_function 通过")


def test_4_register_plain_function():
    """方式4：ToolRegistry 直接注册裸函数（自动包装）"""
    reg = ToolRegistry()

    def multiply(a: int, b: int) -> int:
        """乘法"""
        return a * b

    reg.register_tool(multiply)
    assert "multiply" in reg
    r = reg.invoke_tool("multiply", {"a": 6, "b": 7})
    assert r.output == "42"
    print("✅ 方式4 register_tool 直接收函数 通过")


# ─────────────────────────────────────────────────────────
# 2. 参数校验
# ─────────────────────────────────────────────────────────

def test_validation():
    @tool
    def say(name: str, times: int = 1) -> str:
        """打招呼
        Args:
            name: 名字
            times: 次数
        """
        return " ".join([f"hi {name}"] * times)

    # 缺必填参数
    r = say.invoke()
    assert r.failed
    assert "参数错误" in r.output
    assert "name" in r.output

    # 正常
    r = say.invoke(name="bob")
    assert r.succeeded
    assert r.output == "hi bob"

    print("✅ 参数校验（缺必填失败、正常通过）通过")


# ─────────────────────────────────────────────────────────
# 3. ToolResult / 截断 / 错误处理
# ─────────────────────────────────────────────────────────

def test_result_types():
    @tool(max_output_chars=20)
    def long_text(n: int = 100) -> str:
        """返回长文本"""
        return "A" * n

    r = long_text.invoke(n=100)
    assert r.status == ToolStatus.TRUNCATED
    assert "截断" in r.output

    @tool
    def boom(should_fail: bool = False) -> str:
        """可能失败的工具"""
        if should_fail:
            raise ValueError("故意失败")
        return "ok"

    # handle_tool_error=True（默认）：异常被捕获
    r = boom.invoke(should_fail=True)
    assert r.failed
    assert "故意失败" in r.output

    # handle_tool_error=False：异常抛出
    boom.handle_tool_error = False
    try:
        boom.invoke(should_fail=True)
        assert False, "应该抛出异常"
    except ValueError:
        pass
    finally:
        boom.handle_tool_error = True

    print("✅ ToolResult（成功/失败/截断/异常捕获）通过")


def test_tool_result_ok_fail():
    ok = ToolResult.ok("hello")
    assert ok.succeeded and not ok.failed
    fail = ToolResult.fail("坏了")
    assert fail.failed
    assert str(fail) == "错误：坏了"
    tr = ToolResult.truncated("abc" * 10, 100, 30)
    assert tr.status == ToolStatus.TRUNCATED
    print("✅ ToolResult 构造器（ok/fail/truncated）通过")


# ─────────────────────────────────────────────────────────
# 4. Schema 导出（多 provider）
# ─────────────────────────────────────────────────────────

def test_schema_multi_provider():
    calc = builtin.CalculatorTool()
    oa = calc.to_openai_tool()
    assert oa["type"] == "function"
    assert oa["function"]["name"] == "python_calculator"
    assert "parameters" in oa["function"]
    assert "expression" in oa["function"]["parameters"]["properties"]

    an = calc.to_anthropic_tool()
    assert an["name"] == "python_calculator"
    assert "input_schema" in an
    assert "expression" in an["input_schema"]["properties"]

    gm = calc.to_gemini_tool()
    assert gm["name"] == "python_calculator"
    assert "parameters" in gm

    reg = ToolRegistry()
    reg.register_tool(calc)
    assert len(reg.to_openai_tools()) == 1
    assert len(reg.to_anthropic_tools()) == 1
    assert len(reg.to_gemini_tools()) == 1
    print("✅ 多 provider schema（OpenAI/Anthropic/Gemini）通过")


# （test_calculator / test_terminal / test_read_file 已拆分至 test_tools_builtin.py）
# （test_registry_batch_and_hook 已拆分至 test_tool_registry.py）


# ─────────────────────────────────────────────────────────
# 7. + 运算符兼容
# ─────────────────────────────────────────────────────────

def test_plus_operator():
    calc = builtin.CalculatorTool()

    # tool + tool
    lst = calc + calc
    assert isinstance(lst, list)
    assert len(lst) == 2
    assert lst[0]["function"]["name"] == "python_calculator"

    # list + tool
    lst2 = [] + calc
    assert len(lst2) == 1
    print("✅ + 运算符（tool+tool / list+tool）通过")


# ─────────────────────────────────────────────────────────
# A24: 截断时保留原有artifact与metadata，不覆盖下游所需数据
# ─────────────────────────────────────────────────────────
def test_truncation_preserves_dict_artifact():
    """截断时保留原有dict artifact，追加截断标记，不覆盖content_hash等乐观锁凭证"""
    class LongDictTool(ToolBase):
        name = "long_dict"
        description = "返回长文本+dict artifact"
        max_output_chars = 50

        def run(self, **kwargs):
            return ToolResult.ok(
                output="a" * 200,
                artifact={"content_hash": "abc123", "size": 200, "line_count": 5},
                metadata={"source": "test_file"},
            )

    t = LongDictTool()
    res = t.invoke()
    assert res.status == ToolStatus.TRUNCATED
    # 原有artifact字段保留
    assert res.artifact["content_hash"] == "abc123"
    assert res.artifact["size"] == 200
    assert res.artifact["line_count"] == 5
    # 追加截断标记
    assert res.artifact["truncated"] is True
    assert res.artifact["original_length"] == 200
    assert res.artifact["limit"] == 50
    # 原有metadata保留
    assert res.metadata["source"] == "test_file"
    assert res.metadata["truncated"] is True
    assert "duration_ms" in res.metadata
    print("✅ 截断保留dict artifact（含乐观锁字段）通过")


def test_truncation_preserves_non_dict_artifact():
    """非dict类型artifact会被包在data字段里，不会丢失"""
    class LongListTool(ToolBase):
        name = "long_list"
        description = "返回长文本+列表artifact"
        max_output_chars = 30

        def run(self, **kwargs):
            return ToolResult.ok(
                output="b" * 100,
                artifact=[1, 2, 3, 4],
            )

    t = LongListTool()
    res = t.invoke()
    assert res.status == ToolStatus.TRUNCATED
    # 原有数据保存在data字段
    assert res.artifact["data"] == [1, 2, 3, 4]
    assert res.artifact["truncated"] is True
    print("✅ 截断保留非dict artifact通过")


def test_read_large_file_preserves_hash_for_optimistic_lock():
    """读大文件被截断后仍返回content_hash，乐观锁凭证不丢失（真实场景验证）"""
    from fast_agents.tools.builtin.file_tool import ReadTool, _sha256_bytes
    import tempfile, os

    class SmallMaxReadTool(ReadTool):
        """测试用子类，降低截断阈值"""
        max_output_chars = 20_000

    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "big.txt")
        content = "x" * 200_000  # 20万字符，远大于2万阈值
        raw_bytes = content.encode("utf-8")
        with open(fpath, "wb") as f:
            f.write(raw_bytes)
        expected_hash = _sha256_bytes(raw_bytes, encoding="utf-8")  # 取前16位短hash，与file_tool口径一致

        reader = SmallMaxReadTool(workspace=tmp)
        res = reader.invoke(path="big.txt")
        assert res.status == ToolStatus.TRUNCATED
        # 最关键：乐观锁需要的content_hash必须保留，否则后续write_file无法做冲突检测
        assert res.artifact["content_hash"] == expected_hash
        assert res.artifact["truncated"] is True
        assert "duration_ms" in res.metadata
    print("✅ 大文件截断保留content_hash（乐观锁凭证不丢失）通过")


# ─────────────────────────────────────────────────────────
# 8. 嵌套模型 $ref 展开（回归：带 description 的 $ref 必须内联）
# ─────────────────────────────────────────────────────────
def test_nested_model_ref_expansion():
    """回归测试：工具参数里出现嵌套 BaseModel / Enum 时，schema 必须内联，不能残留 $ref"""
    from enum import Enum
    from typing import Optional

    class Unit(str, Enum):
        celsius = "celsius"
        fahrenheit = "fahrenheit"

    class Location(BaseModel):
        city: str = Field(description="城市名")
        country: Optional[str] = Field(default=None, description="国家")

    class WeatherInput(BaseModel):
        location: Location = Field(description="地点")
        units: Unit = Field(default=Unit.celsius, description="温度单位")

    class WeatherTool(ToolBase):
        name = "get_weather"
        description = "查询天气"
        args_schema = WeatherInput

        def run(self, location, units=Unit.celsius):
            return f"{location.city}: {units}"

    schema = WeatherTool().input_schema
    blob = json.dumps(schema, ensure_ascii=False)

    assert "$ref" not in blob, f"schema 不应残留 $ref: {blob}"
    assert "$defs" not in blob, f"schema 不应残留 $defs: {blob}"

    loc = schema["properties"]["location"]
    assert "properties" in loc, f"嵌套模型 location 未被展开: {loc}"
    assert "city" in loc["properties"], f"嵌套字段丢失: {loc}"
    assert loc["properties"]["city"]["description"] == "城市名"
    assert loc["description"] == "地点", f"兄弟键 description 应保留: {loc}"

    units = schema["properties"]["units"]
    assert units.get("enum") == ["celsius", "fahrenheit"], f"Enum 未展开: {units}"

    assert "title" not in blob, f"title 应被剥离: {blob}"

    print("✅ 嵌套模型 $ref 展开通过（嵌套 BaseModel + Enum 均内联，无残留 $ref/$defs）")


# ──────────────────────────────────────────────────────────────────────────
# A25-1: 结果序列化（非字符串优先 JSON，不产出 Python repr）
# ──────────────────────────────────────────────────────────────────────────
def test_coerce_dict_to_json():
    """dict 返回值应产出合法 JSON（双引号、小写 true/null），而非 Python repr。"""

    class DictTool(ToolBase):
        name = "dict_tool"
        description = "返回 dict 的测试工具"
        args_schema = None

        def run(self):
            return {"ok": True, "count": None, "items": [1, 2]}

    r = DictTool().invoke({})
    # 能被 json.loads 解析即证明是合法 JSON
    parsed = json.loads(r.output)
    assert parsed == {"ok": True, "count": None, "items": [1, 2]}, f"输出不是合法 JSON: {r.output!r}"
    # 确保是双引号 JSON 不是 Python repr（repr 里 True/None 是大写，但 json.loads 都能吃，
    # 所以用一个不含中文/特殊字符的简单 case 查单引号出现即可）
    assert "'" not in r.output, f"输出是 Python repr 而非 JSON: {r.output!r}"
    print("✅ dict → JSON 序列化通过")


def test_coerce_ensure_ascii_false():
    """中文字符不应被转义为 \\uXXXX，保持可读性。"""

    class CnTool(ToolBase):
        name = "cn_tool"
        description = "返回中文 dict"
        args_schema = None

        def run(self):
            return {"name": "张三"}

    r = CnTool().invoke({})
    assert "张三" in r.output, f"中文被转义: {r.output!r}"
    print("✅ 中文不转义（ensure_ascii=False）通过")


def test_coerce_list_to_json():
    """list 返回值也应是 JSON。"""

    class ListTool(ToolBase):
        name = "list_tool"
        description = "返回列表"
        args_schema = None

        def run(self):
            return [1, 2, 3]

    r = ListTool().invoke({})
    assert json.loads(r.output) == [1, 2, 3], f"list 输出不是合法 JSON: {r.output!r}"
    print("✅ list → JSON 序列化通过")


def test_coerce_none_and_str_unchanged():
    """None → 空串、str → 原样，行为保持兼容。"""

    class NoneTool(ToolBase):
        name = "none_tool"
        description = "返回 None"
        args_schema = None

        def run(self):
            return None

    class StrTool(ToolBase):
        name = "str_tool"
        description = "返回字符串"
        args_schema = None

        def run(self):
            return "hello world"

    assert NoneTool().invoke({}).output == ""
    assert StrTool().invoke({}).output == "hello world"
    print("✅ None / str 原始行为保持兼容通过")


def test_coerce_fallback_for_non_serializable():
    """循环引用等无法 JSON 序列化的对象应兜底到 str()，不抛错。"""

    class Bad:
        def __str__(self):
            return "<bad>"

    class BadTool(ToolBase):
        name = "bad_tool"
        description = "返回含自引用的不可序列化对象"
        args_schema = None

        def run(self):
            d = {}
            d["self"] = d  # 循环引用
            d["obj"] = Bad()
            return d

    r = BadTool().invoke({})
    # 兜底路径不抛错即可，输出格式是 str() 结果
    assert isinstance(r.output, str) and r.failed is False
    print("✅ 不可序列化对象回退 str() 不抛错通过")


# ──────────────────────────────────────────────────────────────────────────
# A25-2: 截断语义（truncated 算 succeeded，但 is_truncated 为 True）
# ──────────────────────────────────────────────────────────────────────────
def test_truncated_semantics():
    """截断态：succeeded=True、failed=False、is_truncated=True。"""
    from fast_agents.tools.tool_result import ToolResult, ToolStatus

    tr = ToolResult.truncated("abc", original_length=100, limit=3)
    assert tr.status == ToolStatus.TRUNCATED
    assert tr.succeeded is True, "截断应视为成功（输出被裁剪，而非失败）"
    assert tr.failed is False, "截断不应判为失败"
    assert tr.is_truncated is True
    print("✅ 截断语义 succeeded/is_truncated 通过")


def test_success_is_not_truncated():
    """普通成功：is_truncated=False。"""
    r = ToolResult.ok("hello")
    assert r.succeeded is True
    assert r.failed is False
    assert r.is_truncated is False

    err_result = ToolResult.fail("oops")
    assert err_result.failed is True
    assert err_result.succeeded is False
    assert err_result.is_truncated is False
    print("✅ SUCCESS/ERROR 态 is_truncated=False 通过")


# ──────────────────────────────────────────────────────────────────────────
# A15: A8 并发执行 / A9 人工审批 / A10 超时熔断 测试护栏
# ──────────────────────────────────────────────────────────────────────────

def _sleep_tool(name: str, seconds: float):
    """构造一个耗时 seconds 的测试工具（并发测试需要不同名的工具共存）。"""
    import time

    class _SleepTool(ToolBase):
        def run(self):
            time.sleep(seconds)
            return f"{name} done"

    return _SleepTool(name=name, description=f"耗时 {seconds}s 的测试工具")


# （A8 并发测试 test_concurrent_* 已拆分至 test_tools_concurrent.py）
# （A9 人工审批测试 test_approval_* 已拆分至 test_tools_approval.py）
# （_approval_tool 助手随之拆出，但 _sleep_tool 仍由本文件的 A10/A16/A26 测试使用，故保留）


def test_timeout_returns_failure():
    """A10：超过单次超时返回明确失败。"""
    reg = ToolRegistry()
    reg.register_tool(_sleep_tool("slow_timeout_tool", 0.5))
    r = reg.invoke_tool("slow_timeout_tool", {}, timeout=0.1)
    assert r.failed
    assert "超时" in r.output
    print("✅ A10 超时返回明确失败通过")


def test_timeout_not_triggered_when_fast():
    """A10：快速工具不受超时影响（防误伤）。"""
    reg = ToolRegistry()
    reg.register_tool(_sleep_tool("fast_timeout_tool", 0.02))
    r = reg.invoke_tool("fast_timeout_tool", {}, timeout=5)
    assert r.succeeded, r.output
    print("✅ A10 快工具不被超时误伤通过")


def test_duration_ms_is_int():
    """A10：结果 metadata 携带整型 duration_ms。"""
    calc = builtin.CalculatorTool()
    r = calc.invoke(expression="1+1")
    d = r.metadata.get("duration_ms")
    assert isinstance(d, int), f"duration_ms 应为 int，实际 {type(d).__name__}"
    assert d >= 0
    print("✅ duration_ms 为 int 且 >= 0 通过")


# （test_write_append_preserves_existing_content / test_terminal_platform_guard / test_terminal_truncation_delegated_to_base 已拆分至 test_tools_builtin.py）


# ─────────────────────────────────────────────────────────
# A17：@tool / from_function 透传 metadata / handle_tool_error
# ─────────────────────────────────────────────────────────

def test_decorator_metadata_requires_approval():
    """A17：@tool(metadata=...) 应透传到 StructuredTool.metadata，审批可识别。"""
    @tool(metadata={"requires_approval": True})
    def risky_op(action: str) -> str:
        """需要审批的操作"""
        return f"executed {action}"

    assert risky_op.metadata.get("requires_approval") is True, "metadata 未透传到工具"
    reg = ToolRegistry()
    reg.register_tool(risky_op)
    # 无回调时 skip_approval 路径直接执行（与 A9 既有语义一致）
    r = reg.invoke_tool("risky_op", {"action": "x"}, skip_approval=True)
    assert r.succeeded, r.output
    print("✅ @tool 装饰器 metadata 透传通过")


def test_from_function_metadata_and_error_flag():
    """A17：from_function 可声明 metadata；handle_tool_error=False 时异常向外抛出。"""
    def lookup(key: str) -> str:
        """按 key 查值"""
        if key == "bad":
            raise ValueError("查不到")
        return f"v={key}"

    t_no_catch = StructuredTool.from_function(
        lookup, name="lookup_nocatch",
        metadata={"category": "read"},
        handle_tool_error=False,
    )
    assert t_no_catch.metadata.get("category") == "read"
    # 正常路径
    assert t_no_catch.invoke(key="ok").succeeded
    # handle_tool_error=False 时异常向外抛出，不被兜底为 ToolResult.fail
    raised = False
    try:
        t_no_catch.invoke(key="bad")
    except ValueError as e:
        raised = True
        assert "查不到" in str(e)
    assert raised, "handle_tool_error=False 时异常应向外抛出"

    # handle_tool_error=True（默认）：异常被兜底为失败
    t_catch = StructuredTool.from_function(lookup, name="lookup_catch")
    r = t_catch.invoke(key="bad")
    assert r.failed
    assert "查不到" in r.output
    print("✅ StructuredTool.from_function metadata + handle_tool_error 通过")


def test_tool_return_direct_flag_set():
    """A17/A18：@tool / from_function 传 return_direct=True 时工具实例 return_direct 为 True。"""
    @tool(return_direct=True)
    def direct_answer(q: str) -> str:
        """直接给出最终答案"""
        return f"答案是 {q}"

    assert direct_answer.return_direct is True
    t2 = StructuredTool.from_function(direct_answer.func, return_direct=True,
                                      name="da2", description="d")
    assert t2.return_direct is True
    print("✅ @tool / from_function return_direct 参数透传通过")


# ─────────────────────────────────────────────────────────
# A33: 工具入参解析统一（invoke / ainvoke / invoke_tool 共用同一实现）
# ─────────────────────────────────────────────────────────

def _mk_echo_tool():
    """构造一个用于入参解析测试的回显工具。"""
    @tool
    def echo(text: str = "") -> str:
        """回显文本
        Args:
            text: 要回显的文本
        """
        return f"echo:{text}"
    return echo


def test_merge_invalid_json_same_message():
    """A33：invoke 与 invoke_tool 对非法 JSON 的报错文案一致，且都带工具名。"""
    echo = _mk_echo_tool()
    reg = ToolRegistry()
    reg.register_tool(echo)

    r1 = echo.invoke("不是json")
    r2 = reg.invoke_tool("echo", "不是json")
    assert r1.failed and r2.failed
    assert r1.output == r2.output, f"两条路径文案不一致：{r1.output!r} vs {r2.output!r}"
    assert "工具 'echo'" in r1.output, f"错误信息应带工具名：{r1.output!r}"
    assert "不是合法 JSON" in r1.output
    print("✅ A33 非法 JSON 文案双路径一致通过")


def test_merge_json_array_rejected():
    """A33：JSON 数组不是合法入参，两条路径都提示“必须是 JSON 对象”。"""
    echo = _mk_echo_tool()
    reg = ToolRegistry()
    reg.register_tool(echo)

    r1 = echo.invoke("[1, 2]")
    r2 = reg.invoke_tool("echo", "[1, 2]")
    for r in (r1, r2):
        assert r.failed
        assert "必须是 JSON 对象" in r.output, r.output
    assert r1.output == r2.output
    print("✅ A33 JSON 数组被拒且文案一致通过")


def test_merge_non_dict_type_rejected():
    """A33：非 str/dict 的 tool_input 报类型错误，两条路径文案一致。"""
    echo = _mk_echo_tool()
    reg = ToolRegistry()
    reg.register_tool(echo)

    r1 = echo.invoke(123)
    r2 = reg.invoke_tool("echo", 123)
    for r in (r1, r2):
        assert r.failed
        assert "必须是 dict / JSON 字符串 / None" in r.output, r.output
        assert "int" in r.output
    assert r1.output == r2.output
    print("✅ A33 非法类型报错且文案一致通过")


def test_merge_empty_string_is_empty_dict():
    """A33：空字符串等价于无入参（回归保护）。"""
    echo = _mk_echo_tool()
    reg = ToolRegistry()
    reg.register_tool(echo)

    assert echo.invoke("").output == echo.invoke().output
    assert reg.invoke_tool("echo", "").output == reg.invoke_tool("echo").output
    assert echo.invoke("").output == "echo:"
    print("✅ A33 空字符串等价无入参通过")


def test_merge_kwargs_override_tool_input():
    """A33：kwargs 优先级高于 tool_input 中的同名键（回归保护）。"""
    echo = _mk_echo_tool()
    reg = ToolRegistry()
    reg.register_tool(echo)

    assert echo.invoke({"text": "a"}, text="b").output == "echo:b"
    assert echo.invoke('{"text": "a"}', text="b").output == "echo:b"
    assert reg.invoke_tool("echo", {"text": "a"}, text="b").output == "echo:b"
    print("✅ A33 kwargs 覆盖 tool_input 通过")


def test_merge_ainvoke_consistent():
    """A33：ainvoke 与 invoke 的解析行为一致（三条路径共用同一实现）。"""
    import asyncio
    echo = _mk_echo_tool()

    async def _run():
        ok = await echo.ainvoke('{"text": "hi"}')
        bad = await echo.ainvoke("不是json")
        return ok, bad

    ok, bad = asyncio.run(_run())
    assert ok.output == echo.invoke('{"text": "hi"}').output
    assert bad.output == echo.invoke("不是json").output
    print("✅ A33 ainvoke 与 invoke 行为一致通过")


# ─────────────────────────────────────────────────────────
# （A16 并发语义加固相关测试已拆分至 test_tools_concurrent.py）
# （A22 ToolRegistry 清理与隔离相关测试已拆分至 test_tool_registry.py）
# ─────────────────────────────────────────────────────────


def test_a23_builtin_lazy_import():
    """A23：builtin 延迟导入——用占位父包隔离验证，导入/访问基础工具不拉 memory_tool/mcp_tool，访问后按需加载，API 兼容。"""
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import os, sys, types\n"
        "root = os.environ['FA_ROOT']\n"
        "fa = types.ModuleType('fast_agents'); fa.__path__ = [os.path.join(root, 'fast_agents')]\n"
        "tl = types.ModuleType('fast_agents.tools'); tl.__path__ = [os.path.join(root, 'fast_agents', 'tools')]\n"
        "sys.modules['fast_agents'] = fa; sys.modules['fast_agents.tools'] = tl\n"
        "def _loaded():\n"
        "    return [m for m in sys.modules if m.startswith('fast_agents.tools.builtin.') and (m.endswith('memory_tool') or m.endswith('mcp_tool'))]\n"
        "import fast_agents.tools.builtin as b\n"
        "assert not _loaded(), '导入 builtin 时 memory_tool/mcp_tool 被提前加载'\n"
        "b.CalculatorTool\n"
        "assert not _loaded(), '访问 CalculatorTool 却加载了 memory_tool/mcp_tool'\n"
        "b.RememberTool\n"
        "assert any(m.endswith('memory_tool') for m in _loaded()), '访问 RememberTool 未按需加载 memory_tool'\n"
        "assert 'RememberTool' in dir(b), 'dir() 未包含延迟导出名'\n"
        "assert set(b.__all__) <= set(b._LAZY_IMPORTS), '__all__ 未全部由延迟映射覆盖'\n"
        "from fast_agents.tools.builtin import CalculatorTool, build_mcp_tools\n"
        "print('LAZY_OK')\n"
    )
    env = dict(os.environ)
    env["FA_ROOT"] = str(repo_root)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo_root), env=env)
    assert r.returncode == 0 and "LAZY_OK" in r.stdout, f"A23 延迟导入校验失败:\n{r.stdout}\n{r.stderr}"
    print("✅ A23 builtin 延迟导入（隔离验证：不提前拉起 memory/mcp、按需加载、API 兼容）通过")


# ─────────────────────────────────────────────────────────
# A19：异步入口现代化（asyncio.to_thread，消除 DeprecationWarning）
# ─────────────────────────────────────────────────────────

def test_a19_arun_no_deprecation_warning_and_result():
    """A19：ToolBase.arun 已改用 asyncio.to_thread——不再触发
    asyncio.get_event_loop() 的 DeprecationWarning，且返回与同步路径一致。"""
    import asyncio
    import warnings

    class EchoTool(ToolBase):
        name = "a19_echo"
        description = "回显文本"
        def run(self, text: str = ""):
            return f"ECHO={text}"

    async def _drive():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = await EchoTool().arun(text="hi")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        return r, dep

    result, dep = asyncio.run(_drive())
    assert not dep, f"arun 触发了 DeprecationWarning: {[str(w.message) for w in dep]}"
    assert str(result) == "ECHO=hi", f"arun 结果异常: {result!r}"
    print("✅ A19 arun 走 asyncio.to_thread、无 DeprecationWarning、结果一致通过")


# ─────────────────────────────────────────────────────────
# A11：Tavily provider 缺依赖 / 缺 Key 的明确指引
# ─────────────────────────────────────────────────────────

def test_a11_tavily_missing_key_guide():
    """A11：Tavily 缺 API Key 时返回明确配置指引，而非笼统「搜索失败」"""
    import sys as _sys
    import types as _types
    from fast_agents.tools.builtin.search_tool import SearchTool

    # 注入 tavily 桩模块：让依赖检查通过，从而走到「缺 Key」分支
    stub = _types.ModuleType("tavily")

    class _StubClient:
        def __init__(self, *a, **k):
            raise AssertionError("缺 Key 时不应构造客户端")

    stub.TavilyClient = _StubClient

    old_mod = _sys.modules.get("tavily")
    saved_key = os.environ.pop("TAVILY_API_KEY", None)
    _sys.modules["tavily"] = stub
    try:
        out = str(SearchTool(provider="tavily", api_key=None).run("测试关键词"))
    finally:
        if old_mod is not None:
            _sys.modules["tavily"] = old_mod
        else:
            _sys.modules.pop("tavily", None)
        if saved_key is not None:
            os.environ["TAVILY_API_KEY"] = saved_key

    assert "TAVILY_API_KEY" in out, f"缺 Key 未给出明确指引: {out!r}"
    assert "duckduckgo" in out, f"指引应含免 Key 备选方案: {out!r}"
    print("✅ A11 Tavily 缺 Key 返回明确指引通过")


def test_a11_tavily_with_key_ready():
    """A11：提供 Key 时前置检查放行，能正常返回结果（不误报缺 Key）"""
    import sys as _sys
    import types as _types
    from fast_agents.tools.builtin.search_tool import SearchTool

    stub = _types.ModuleType("tavily")
    captured = {}

    class _StubClient:
        def __init__(self, api_key=None, **k):
            captured["api_key"] = api_key

        def search(self, query, max_results=5):
            return {"results": [{"title": "标题", "url": "http://example.com/x", "content": "摘要"}]}

    stub.TavilyClient = _StubClient

    old_mod = _sys.modules.get("tavily")
    _sys.modules["tavily"] = stub
    try:
        res = SearchTool(provider="tavily", api_key="demo-key").run("测试")
    finally:
        if old_mod is not None:
            _sys.modules["tavily"] = old_mod
        else:
            _sys.modules.pop("tavily", None)

    assert "http://example.com/x" in str(res), f"有 Key 时应正常返回结果: {res!r}"
    assert captured.get("api_key") == "demo-key", f"Key 未透传给客户端: {captured!r}"
    print("✅ A11 Tavily 有 Key 正常返回通过")


# ─────────────────────────────────────────────────────────
# A26：并发工具执行时 hooks 可回调用方线程触发
# ─────────────────────────────────────────────────────────

def test_a26_concurrent_hooks_can_run_in_caller_thread():
    """A26：invoke_tools_concurrent 新增 hooks_in_caller_thread 开关。

    - 默认 False：hook 仍在子线程触发（与改动前行为一致）；
    - True：hook 收回调用方线程，按 calls 顺序各触发一次；
    - 两种情况返回结果都仍与 calls 严格一一对应，且 hook 收到的 result 与返回一致。
    """
    import threading

    main_tid = threading.get_ident()

    def _build():
        reg = ToolRegistry(max_workers=4)
        reg.register_tools(_sleep_tool("h1", 0.05), _sleep_tool("h2", 0.05))
        return reg

    # 默认：hook 在子线程触发
    reg = _build()
    default_seen = []
    reg.add_hook(lambda name, args, res: default_seen.append((name, threading.get_ident())))
    results = reg.invoke_tools_concurrent([("h1", None), ("h2", None)])
    assert [r.output for r in results] == ["h1 done", "h2 done"], results
    assert len(default_seen) == 2, f"默认模式 hook 应触发 2 次，实际 {len(default_seen)}"
    assert not any(tid == main_tid for _, tid in default_seen), \
        "默认模式下 hook 不应跑在调用方线程（须与改动前一致）"

    # 开启开关：hook 回到调用方线程，且顺序与 calls 一致
    reg2 = _build()
    caller_seen = []

    def _hook(name, args, res):
        caller_seen.append((name, threading.get_ident(), res.output))

    reg2.add_hook(_hook)
    results2 = reg2.invoke_tools_concurrent(
        [("h2", None), ("h1", None)], hooks_in_caller_thread=True,
    )
    assert [r.output for r in results2] == ["h2 done", "h1 done"], results2
    assert [n for n, _, _ in caller_seen] == ["h2", "h1"], \
        f"hook 顺序应与 calls 一致，实际 {[n for n, _, _ in caller_seen]}"
    assert all(tid == main_tid for _, tid, _ in caller_seen), \
        "开启 hooks_in_caller_thread 后 hook 必须在调用方线程执行"
    assert [out for _, _, out in caller_seen] == [r.output for r in results2], \
        "hook 拿到的 result 应与返回列表一一对应"
    print("✅ A26 并发 hooks 可回调用方线程、按序触发通过")


def test_a26_caller_thread_hooks_keep_trace_parent_chain():
    """A26：hook 收在调用方线程后，tracer 的工具 Span 能正常落库并保持父链。

    动机：tracer 的 Span 栈是实例级 ``contextvars``（线程绑定），且底层 SQLite
    连接禁止跨线程使用——hook 若在子线程触发，工具 Span 会既丢父链又不落库。
    """
    import tempfile
    from fast_agents.trace import TraceLogger

    reg = ToolRegistry(max_workers=4)
    reg.register_tools(_sleep_tool("t1", 0.05), _sleep_tool("t2", 0.05))

    with tempfile.TemporaryDirectory() as tmp:
        tracer = TraceLogger(store=os.path.join(tmp, "trace.db"))
        tracer.start_trace(name="parent.trace")
        parent_trace_id = tracer._current_trace_id()
        tracer.attach_to_tool_registry(reg)

        reg.invoke_tools_concurrent(
            [("t1", None), ("t2", None)], hooks_in_caller_thread=True,
        )
        spans = tracer.read_spans()
        tracer.close()  # 先关闭 SQLite 连接，否则 Windows 下 TemporaryDirectory 清理会因 trace.db 被占用抛 WinError 267

    tool_spans = [s for s in spans if s.name.startswith("tool.")]
    assert len(tool_spans) == 2, f"应有 2 条工具 Span 落库，实际 {len(tool_spans)}：{[s.name for s in spans]}"
    assert {s.trace_id for s in tool_spans} == {parent_trace_id}, \
        f"工具 Span 丢失父链：{[ (s.name, s.trace_id) for s in tool_spans ]}"
    print("✅ A26 hook 回主线程后工具 Span 正常落库且父链正确通过")


# ─────────────────────────────────────────────────────────
# 搜索 provider 为可选依赖（extras: search）——缺包时返回安装提示而非抛异常
# ─────────────────────────────────────────────────────────

def test_search_missing_duckduckgo_dependency_returns_hint():
    """未装 duckduckgo-search 时，SearchTool 返回安装提示而非抛异常"""
    import sys as _sys
    from fast_agents.tools.builtin.search_tool import SearchTool

    saved = _sys.modules.get("duckduckgo_search")
    _sys.modules["duckduckgo_search"] = None  # 令 `import duckduckgo_search` 触发 ImportError
    try:
        res = SearchTool(provider="duckduckgo").run("测试关键词")
    finally:
        if saved is not None:
            _sys.modules["duckduckgo_search"] = saved
        else:
            _sys.modules.pop("duckduckgo_search", None)

    assert isinstance(res, ToolResult) and res.is_error, f"缺包应返回失败结果而非异常: {res!r}"
    out = str(res)
    assert "duckduckgo-search" in out, f"应提示安装包名 duckduckgo-search: {out!r}"
    assert "pip install" in out, f"应给出 pip 安装指引: {out!r}"
    print("✅ 缺 duckduckgo-search 返回安装提示通过")


def test_search_missing_tavily_dependency_returns_hint():
    """未装 tavily-python 时，SearchTool 前置检查返回安装提示而非抛异常"""
    import sys as _sys
    from fast_agents.tools.builtin.search_tool import SearchTool

    saved = _sys.modules.get("tavily")
    _sys.modules["tavily"] = None
    try:
        res = SearchTool(provider="tavily", api_key="demo-key").run("测试")
    finally:
        if saved is not None:
            _sys.modules["tavily"] = saved
        else:
            _sys.modules.pop("tavily", None)

    assert isinstance(res, ToolResult) and res.is_error, f"缺包应返回失败结果而非异常: {res!r}"
    out = str(res)
    assert "tavily-python" in out, f"应提示安装包名 tavily-python: {out!r}"
    assert "pip install" in out, f"应给出 pip 安装指引: {out!r}"
    print("✅ 缺 tavily-python 返回安装提示通过")


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    test_a19_arun_no_deprecation_warning_and_result()  # A19
    test_a11_tavily_missing_key_guide()  # A11
    test_a11_tavily_with_key_ready()  # A11
    test_a26_concurrent_hooks_can_run_in_caller_thread()  # A26
    test_a26_caller_thread_hooks_keep_trace_parent_chain()  # A26
    test_search_missing_duckduckgo_dependency_returns_hint()  # 可选依赖
    test_search_missing_tavily_dependency_returns_hint()  # 可选依赖


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
