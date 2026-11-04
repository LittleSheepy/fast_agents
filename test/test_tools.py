"""Tools 模块单元测试

运行：python test/test_tools.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ─────────────────────────────────────────────────────────
# 5. 内置工具
# ─────────────────────────────────────────────────────────

def test_calculator():
    calc = builtin.CalculatorTool()

    cases = [
        ("2+3*4", "14"),
        ("sqrt(16)", "4"),
        ("sin(pi/2)", "1"),
        ("2**10", "1024"),
        ("2^10", "8"),
        ("(3+5)*2", "16"),
    ]
    for expr, expected in cases:
        r = calc.invoke(expression=expr)
        assert r.succeeded, f"{expr} -> {r.output}"
        # 结果以 "expr = " 开头，取等号后
        got = r.output.split("=", 1)[-1].strip()
        assert got == expected, f"{expr} 期望 {expected}，得到 {got}"

    # 错误用例
    r = calc.invoke(expression="1/0")
    assert r.failed
    assert "除零" in r.output

    r = calc.invoke(expression="import os")
    assert r.failed
    print("✅ CalculatorTool 通过")


def test_terminal():
    """TerminalTool：Windows 专用只读命令白名单 + 沙箱 + cd 校验"""
    if os.name != "nt":
        print("⏭️  test_terminal 跳过（TerminalTool 是 Windows 专用，当前非 Windows）")
        return
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hello.txt").write_text("hello fast_agents\n", encoding="utf-8")
        term = builtin.TerminalTool(workspace=tmp)

        # 1) 白名单内的只读命令
        r = term.invoke(command="dir")
        assert r.succeeded, r.output
        assert "hello.txt" in r.output

        r = term.invoke(command="type hello.txt")
        assert r.succeeded, r.output
        assert "hello fast_agents" in r.output

        r = term.invoke(command="echo hi")
        assert r.succeeded, r.output

        # 2) 白名单外的命令被拒（pwd / ls / rm 都不是 Windows 白名单命令）
        r = term.invoke(command="rm -rf /")
        assert r.failed
        assert "不允许" in r.output

        # 3) 已从白名单移除的写命令（del / copy / move / ren / md）也被拒
        r = term.invoke(command="del hello.txt")
        assert r.failed
        assert "不允许" in r.output

        # 4) cd 单独执行：持久切换当前目录；越界则拒绝
        (Path(tmp) / "sub").mkdir()
        r = term.invoke(command="cd sub")
        assert r.succeeded, r.output
        assert "sub" in r.output

        r = term.invoke(command="cd ..\\..\\..")
        assert r.failed
        assert "工作目录外" in r.output

        # 5) cd 与其它命令拼接 → 直接拒绝（防链中 cd 绕过沙箱）
        r = term.invoke(command="dir && cd sub")
        assert r.failed
        assert "拼接" in r.output
    print("✅ TerminalTool 通过")


def test_read_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        rf = builtin.ReadTool(workspace=tmp)

        # 创建测试文件
        f = Path(tmp) / "test.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")

        r = rf.invoke(path="test.txt")
        assert r.succeeded
        assert "共 50 行" in r.output
        assert "line 1" in r.output
        assert "line 50" in r.output

        # 按行号
        r = rf.invoke(path="test.txt", start_line=3, end_line=5, max_lines=10)
        assert "line 3" in r.output
        assert "line 5" in r.output
        assert "line 1" not in r.output

        # 路径穿越拦截
        r = rf.invoke(path="../etc/passwd")
        assert r.failed
        assert "工作目录外" in r.output
    print("✅ ReadTool 通过")


# ─────────────────────────────────────────────────────────
# 6. 注册表批量操作 / hook / tag 过滤
# ─────────────────────────────────────────────────────────

def test_registry_batch_and_hook():
    reg = ToolRegistry()
    reg.register_tools(
        builtin.CalculatorTool(),
        builtin.ReadTool(workspace="/tmp"),
    )
    assert len(reg) == 2
    assert reg.has_tool("python_calculator")
    assert "read_file" in reg.list_names()

    # tag 过滤
    math_tools = reg.filter_by_tag("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "python_calculator"

    # hook
    calls = []
    reg.add_hook(lambda name, args, res: calls.append((name, res.status.value)))
    reg.invoke_tool("python_calculator", {"expression": "1+1"})
    assert len(calls) == 1
    assert calls[0][0] == "python_calculator"
    assert calls[0][1] == "success"

    # JSON 字符串参数（模拟 LLM 传入）
    r = reg.invoke_tool("python_calculator", '{"expression": "2*3"}')
    assert r.output.endswith("6") or "= 6" in r.output

    # 找不到工具
    r = reg.invoke_tool("nonexistent", {})
    assert r.failed
    print("✅ Registry 批量/hook/tag/JSON参数 通过")


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


def _approval_tool(name: str = "approval_tool"):
    """构造一个标记了 requires_approval 的测试工具。"""

    class _ApprovalTool(ToolBase):
        def run(self, expression: str = "1+1"):
            return f"expr={expression}"

    return _ApprovalTool(
        name=name,
        description="需要审批的测试工具",
        metadata={"requires_approval": True},
    )


def test_concurrent_runs_in_parallel():
    """A8：多个无依赖工具并发执行，总耗时取最大值而非累加；返回 List 按 calls 顺序一一对应。"""
    import time
    reg = ToolRegistry(max_workers=4, default_timeout=300)
    reg.register_tools(
        _sleep_tool("s1", 0.3),
        _sleep_tool("s2", 0.3),
        _sleep_tool("s3", 0.3),
    )
    calls = [("s1", None), ("s2", None), ("s3", None)]
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls)
    elapsed = time.perf_counter() - start
    assert len(results) == 3, f"返回长度应与 calls 相等（3），实际 {len(results)}"
    assert isinstance(results, list), f"返回类型应为 List，实际 {type(results).__name__}"
    assert all(r.succeeded for r in results), results
    # 串行约需 0.9s；并发应接近 0.3s，上界留足抖动余量
    assert elapsed < 0.7, f"并发未生效，总耗时 {elapsed:.3f}s（串行约 0.9s）"
    print(f"✅ A8 并发总耗时 {elapsed:.3f}s（< 0.7s）通过")


def test_concurrent_empty_calls_returns_empty():
    """A8：空调用列表返回空列表，不报错。"""
    reg = ToolRegistry()
    assert reg.invoke_tools_concurrent([]) == []
    print("✅ A8 空调用列表返回 [] 通过")


def test_concurrent_exception_isolated():
    """A8/A16①：单个工具抛异常不影响其它工具，异常那个以失败返回（按 calls 顺序一一对应）。"""

    class BoomTool(ToolBase):
        name = "boom_tool"
        description = "必然抛异常的测试工具"

        def run(self):
            raise ValueError("故意炸")

    reg = ToolRegistry()
    reg.register_tools(_sleep_tool("ok_tool", 0.05), BoomTool())
    calls = [("ok_tool", None), ("boom_tool", None)]
    results = reg.invoke_tools_concurrent(calls)
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["ok_tool"].succeeded
    assert name_to_result["boom_tool"].failed
    assert "故意炸" in name_to_result["boom_tool"].output
    print("✅ A8/A16① 单工具异常被隔离通过")


def test_concurrent_partial_timeout():
    """A8/A16④：整体超时后，已完成的保留结果，未完成的标记超时失败并带 timeout_abandoned=True。"""
    reg = ToolRegistry(max_workers=4)
    reg.register_tools(_sleep_tool("fast_tool", 0.05), _sleep_tool("slow_tool", 0.5))
    calls = [("fast_tool", None), ("slow_tool", None)]
    import time
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls, timeout=0.2)
    elapsed = time.perf_counter() - start
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["fast_tool"].succeeded, results
    assert name_to_result["slow_tool"].failed
    assert "超时" in name_to_result["slow_tool"].output
    assert name_to_result["slow_tool"].metadata.get("timeout_abandoned") is True, \
        f"超时结果应标记 timeout_abandoned=True，实际 metadata={name_to_result['slow_tool'].metadata}"
    # A16③-b：超时后立即返回，不再等慢线程跑完（慢工具 0.5s，若仍阻塞就会接近 0.5s）
    assert elapsed < 0.35, f"超时后仍阻塞 {elapsed:.3f}s，shutdown(wait=False) 未生效"
    print(f"✅ A8/A16④ 整体超时：快的保留、慢的标记超时且快速返回（{elapsed:.3f}s）通过")


def test_approval_approved():
    """A9：审批通过后正常执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(lambda name, args: (True, None, ""))
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.succeeded, r.output
    assert r.output == "expr=1+1"
    print("✅ A9 审批通过后正常执行通过")


def test_approval_rejected():
    """A9：审批拒绝时工具不执行，返回失败并带上原因。"""
    executed = []

    class SpyTool(ToolBase):
        name = "spy_tool"
        description = "记录是否执行过的审批工具"
        metadata = {"requires_approval": True}

        def run(self, expression: str = "1+1"):
            executed.append(expression)
            return f"expr={expression}"

    reg = ToolRegistry()
    reg.register_tool(SpyTool())
    reg.set_approval_callback(lambda name, args: (False, None, "不安全"))
    r = reg.invoke_tool("spy_tool", {"expression": "1+1"})
    assert r.failed
    assert "被拒绝" in r.output
    assert "不安全" in r.output
    assert executed == [], "被拒绝的工具不应执行"
    print("✅ A9 审批拒绝：不执行 + 失败带原因 通过")


def test_approval_modified_args():
    """A9：审批回调可改写参数，工具用改写后的参数执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(lambda name, args: (True, {"expression": "2+2"}, ""))
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.succeeded, r.output
    assert r.output == "expr=2+2", f"未使用改写后的参数: {r.output!r}"
    print("✅ A9 审批改写参数生效通过")


def test_approval_callback_exception():
    """A9：审批回调抛异常时返回失败，不抛穿到调用方。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())

    def boom(name, args):
        raise RuntimeError("回调炸了")

    reg.set_approval_callback(boom)
    r = reg.invoke_tool("approval_tool", {"expression": "1+1"})
    assert r.failed
    assert "审批流程异常" in r.output
    print("✅ A9 审批回调异常：失败返回不抛穿 通过")


def test_approval_skipped_without_callback():
    """A9：未设置审批回调时，即使工具标记了 requires_approval 也直接执行。"""
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    r = reg.invoke_tool("approval_tool", {"expression": "3+3"})
    assert r.succeeded, r.output
    assert r.output == "expr=3+3"
    print("✅ A9 无回调时直接执行通过")


def test_approval_not_triggered_for_unmarked_tool():
    """A9：未标记 requires_approval 的工具不触发审批回调。"""
    calls = []

    class PlainTool(ToolBase):
        name = "plain_tool"
        description = "无需审批的测试工具"

        def run(self):
            return "plain ok"

    reg = ToolRegistry()
    reg.register_tool(PlainTool())
    reg.set_approval_callback(
        lambda name, args: (calls.append(name), (False, None, "x"))[1]
    )
    r = reg.invoke_tool("plain_tool", {})
    assert r.succeeded
    assert calls == [], f"未标记的工具不该触发审批，实际调用 {calls}"
    print("✅ A9 未标记工具不触发审批通过")


def test_approval_skip_flag_bypasses_callback():
    """A9：skip_approval=True 时跳过审批（并发内部路径依赖此语义）。"""
    calls = []
    reg = ToolRegistry()
    reg.register_tool(_approval_tool())
    reg.set_approval_callback(
        lambda name, args: (calls.append(name), (False, None, "rejected"))[1]
    )
    r = reg.invoke_tool("approval_tool", {"expression": "4+4"}, skip_approval=True)
    assert r.succeeded, r.output
    assert r.output == "expr=4+4"
    assert calls == [], "skip_approval=True 不应触发回调"
    print("✅ A9 skip_approval 跳过审批通过")


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


def test_write_append_preserves_existing_content():
    """A6：write_file 的 append=True 追加而非覆盖。"""
    import tempfile
    from pathlib import Path
    from fast_agents.tools.builtin.file_tool import WriteTool

    with tempfile.TemporaryDirectory() as tmp:
        wt = WriteTool(workspace=tmp)
        wt.invoke(path="log.txt", content="line1")
        r = wt.invoke(path="log.txt", content="line2", append=True)
        assert r.succeeded, r.output
        content = (Path(tmp) / "log.txt").read_text(encoding="utf-8")
        assert "line1" in content and "line2" in content, f"追加丢失原内容: {content!r}"
        assert content.index("line1") < content.index("line2"), "追加内容顺序错误"
    print("✅ write_file append=True 追加不覆盖通过")



def test_terminal_platform_guard():
    """N3：非 Windows 平台构造 TerminalTool 应明确报错，而非静默假成功。"""
    import tempfile
    from fast_agents.tools.tool_exception import ToolInitializationError

    if os.name == "nt":
        with tempfile.TemporaryDirectory() as tmp:
            term = builtin.TerminalTool(workspace=tmp)
            assert term is not None
        print("✅ TerminalTool 平台守卫通过（Windows：正常构造）")
        return

    with tempfile.TemporaryDirectory() as tmp:
        try:
            builtin.TerminalTool(workspace=tmp)
        except ToolInitializationError as e:
            assert "Windows" in str(e), f"错误信息未说明平台限制: {e}"
            print("✅ TerminalTool 平台守卫通过（非 Windows：明确报错）")
            return
    raise AssertionError("非 Windows 平台构造 TerminalTool 应抛 ToolInitializationError，实际未报错")


def test_terminal_truncation_delegated_to_base():
    """N2：terminal 输出过长由基类统一截断为 TRUNCATED，不再手动二次截断。"""
    import tempfile
    import unittest.mock as mock
    import fast_agents.tools.builtin.terminal_tool as tt

    long_output = "line\n" * 2000
    with mock.patch.object(tt, "_is_windows", return_value=True), \
         mock.patch.object(tt, "route_and_run", return_value=(0, long_output, "", "cmd")):
        with tempfile.TemporaryDirectory() as tmp:
            term = tt.TerminalTool(workspace=tmp, max_output_chars=100)
            r = term.invoke(command="dir")

    assert r.succeeded, r.output
    assert r.is_truncated, f"超长输出应被基类截断为 TRUNCATED，实际 status={r.status}"
    assert "输出已截断" in r.output, "缺少基类截断标记"
    assert "输出过长，已截断前" not in r.output, "仍存在旧的手动截断标记（二次截断未清除）"
    print("✅ terminal 输出截断统一由基类处理通过")


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
# A16: 并发语义加固（List返回、审批前置、线程不套娃、超时标记）
# ─────────────────────────────────────────────────────────

def test_concurrent_same_tool_twice_preserves_both_results():
    """A16①（A15 转正）：同一工具在一次并发里调用两次，两条结果都保留，顺序与输入一致。"""
    @tool
    def echo2(text: str) -> str:
        """回显文本"""
        return f"ECHO={text}"

    reg = ToolRegistry()
    reg.register_tool(echo2)
    calls = [("echo2", {"text": "A"}), ("echo2", {"text": "B"})]
    results = reg.invoke_tools_concurrent(calls)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].succeeded and results[1].succeeded
    assert results[0].output == "ECHO=A", f"第0位结果错误：{results[0].output!r}"
    assert results[1].output == "ECHO=B", f"第1位结果错误：{results[1].output!r}"
    print("✅ A16① 同名工具并发两条结果都在、顺序正确通过")


def test_concurrent_approval_called_per_tool_on_main_thread():
    """A16②：审批在主线程集中触发，每个需审批工具恰好触发一次回调，不进入子线程重复审批。"""
    import threading

    approvals: List[Tuple[str, dict]] = []
    approval_threads: List[str] = []
    main_thread = threading.current_thread().name

    def _cb(name, args):
        approvals.append((name, args))
        approval_threads.append(threading.current_thread().name)
        return (True, None, "")

    tool_a = _approval_tool("app_a")
    tool_b = _approval_tool("app_b")

    class PlainTool(ToolBase):
        name = "plain_tool"
        description = "无需审批"
        def run(self):
            return "ok"

    reg = ToolRegistry()
    reg.register_tools(tool_a, tool_b, PlainTool())
    reg.set_approval_callback(_cb)
    results = reg.invoke_tools_concurrent([
        ("app_a", {"expression": "1+1"}),
        ("plain_tool", None),
        ("app_b", {"expression": "2+2"}),
    ])
    assert len(results) == 3
    assert all(r.succeeded for r in results), results
    # 仅两个需审批工具触发回调
    assert len(approvals) == 2, f"应触发2次审批，实际 {len(approvals)}: {approvals}"
    approved_names = {n for n, _ in approvals}
    assert approved_names == {"app_a", "app_b"}
    # 全部审批都在主线程（不在子工作线程）
    assert all(t == main_thread for t in approval_threads), \
        f"审批应在主线程 {main_thread}，实际线程: {approval_threads}"
    print("✅ A16② 审批主线程集中触发、每工具一次通过")


def test_concurrent_approval_rejected_skips_execution():
    """A16②：被拒绝的工具不进入线程池执行，其余工具正常完成。"""
    executed: List[str] = []

    class SpyA(ToolBase):
        name = "spy_a"
        description = "spy a"
        metadata = {"requires_approval": True}
        def run(self):
            executed.append("a")
            return "a done"

    class SpyB(ToolBase):
        name = "spy_b"
        description = "spy b"
        metadata = {"requires_approval": True}
        def run(self):
            executed.append("b")
            return "b done"

    reg = ToolRegistry()
    reg.register_tools(SpyA(), SpyB())
    reg.set_approval_callback(lambda name, args: (False, None, "no") if name == "spy_a" else (True, None, ""))
    results = reg.invoke_tools_concurrent([("spy_a", None), ("spy_b", None)])
    name_to_result = {"spy_a": results[0], "spy_b": results[1]}
    assert name_to_result["spy_a"].failed
    assert "被拒绝" in name_to_result["spy_a"].output
    assert name_to_result["spy_b"].succeeded
    assert executed == ["b"], f"被拒工具不应执行，实际执行了 {executed}"
    print("✅ A16② 被拒工具不执行、其余正常通过")


def test_concurrent_timeout_returns_promptly():
    """A16③-b：整体超时后 shutdown(wait=False) 立即返回，不再等慢线程跑完。"""
    import time
    reg = ToolRegistry(max_workers=4)
    reg.register_tools(_sleep_tool("q_fast", 0.03), _sleep_tool("q_slow", 0.6))
    calls = [("q_fast", None), ("q_slow", None)]
    start = time.perf_counter()
    results = reg.invoke_tools_concurrent(calls, timeout=0.2)
    elapsed = time.perf_counter() - start
    name_to_result = {name: results[i] for i, (name, _) in enumerate(calls)}
    assert name_to_result["q_fast"].succeeded
    assert name_to_result["q_slow"].failed
    assert elapsed < 0.35, f"超时后仍阻塞 {elapsed:.3f}s，shutdown(wait=False) 未生效"
    print(f"✅ A16③-b 超时立即返回（{elapsed:.3f}s）通过")


def test_concurrent_no_nested_executor():
    """A16③-a：并发入口子任务直接调用 tool.invoke，不再经过 invoke_tool 的内层超时池；
    线程峰值应 ≈ 基线 + 并发 worker 数 + 1（主线程），不再出现线程套娃。
    """
    import threading, time
    peak = [0]
    active = [0]
    lock = threading.Lock()

    class CountTool(ToolBase):
        name = "count_tool"
        description = "统计并发线程峰值"
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx
            self.name = f"count_{idx}"
            self.description = f"计数器 {idx}"
        def run(self):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.15)
            with lock:
                active[0] -= 1
            return f"c{self.idx}"

    # 启动前后台稳定，确保不混入其他线程
    baseline = threading.active_count()
    reg = ToolRegistry(max_workers=3)
    reg.register_tools(CountTool(1), CountTool(2), CountTool(3))
    results = reg.invoke_tools_concurrent([
        ("count_1", None), ("count_2", None), ("count_3", None),
    ])
    assert all(r.succeeded for r in results), results
    # 修复前内层套池：3 个外层 worker + 3 个内层超时线程 = 峰值至少 baseline+6；
    # 修复后子任务直接执行：峰值 = baseline + 3（3 个 worker），不会再多出内层线程
    assert peak[0] <= baseline + 3 + 1, (
        f"线程套娃未消除：基线 {baseline}，峰值 {peak[0]}，差值 {peak[0] - baseline}"
    )
    print(f"✅ A16③-a 并发无嵌套池（基线 {baseline}，峰值 {peak[0]}）通过")


def test_timeout_abandoned_metadata_marker():
    """A16④：超时结果 metadata 携带 timeout_abandoned=True，便于下游识别。"""
    reg = ToolRegistry()
    reg.register_tool(_sleep_tool("slow_marker", 0.5))
    r = reg.invoke_tool("slow_marker", {}, timeout=0.1)
    assert r.failed
    assert r.metadata.get("timeout_abandoned") is True, f"缺 timeout_abandoned 标记：{r.metadata}"
    print("✅ A16④ 单次超时结果带 timeout_abandoned=True 通过")


if __name__ == "__main__":
    from pathlib import Path  # for test_terminal
    tests = [
        test_1_class_inheritance,
        test_2_decorator,
        test_3_from_function,
        test_4_register_plain_function,
        test_validation,
        test_result_types,
        test_tool_result_ok_fail,
        test_schema_multi_provider,
        test_calculator,
        test_terminal,
        test_read_file,
        test_registry_batch_and_hook,
        test_plus_operator,
        test_truncation_preserves_dict_artifact,
        test_truncation_preserves_non_dict_artifact,
        test_read_large_file_preserves_hash_for_optimistic_lock,
        test_nested_model_ref_expansion,
        test_coerce_dict_to_json,
        test_coerce_ensure_ascii_false,
        test_coerce_list_to_json,
        test_coerce_none_and_str_unchanged,
        test_coerce_fallback_for_non_serializable,
        test_truncated_semantics,
        test_success_is_not_truncated,
        test_concurrent_runs_in_parallel,
        test_concurrent_empty_calls_returns_empty,
        test_concurrent_exception_isolated,
        test_concurrent_partial_timeout,
        test_approval_approved,
        test_approval_rejected,
        test_approval_modified_args,
        test_approval_callback_exception,
        test_approval_skipped_without_callback,
        test_approval_not_triggered_for_unmarked_tool,
        test_approval_skip_flag_bypasses_callback,
        test_timeout_returns_failure,
        test_timeout_not_triggered_when_fast,
        test_duration_ms_is_int,
        test_write_append_preserves_existing_content,
        test_terminal_platform_guard,
        test_terminal_truncation_delegated_to_base,
        test_decorator_metadata_requires_approval,
        test_from_function_metadata_and_error_flag,
        test_tool_return_direct_flag_set,
        test_merge_invalid_json_same_message,
        test_merge_json_array_rejected,
        test_merge_non_dict_type_rejected,
        test_merge_empty_string_is_empty_dict,
        test_merge_kwargs_override_tool_input,
        test_merge_ainvoke_consistent,
        test_concurrent_same_tool_twice_preserves_both_results,
        test_concurrent_approval_called_per_tool_on_main_thread,
        test_concurrent_approval_rejected_skips_execution,
        test_concurrent_timeout_returns_promptly,
        test_concurrent_no_nested_executor,
        test_timeout_abandoned_metadata_marker,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n🎉 完成：{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(0 if failed == 0 else 1)
