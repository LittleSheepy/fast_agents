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
        ("2^10", "1024"),
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
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        term = builtin.TerminalTool(workspace=tmp)

        r = term.invoke(command="pwd")
        assert r.succeeded
        assert tmp in r.output

        r = term.invoke(command="ls")
        assert r.succeeded

        # 禁止危险命令
        r = term.invoke(command="rm -rf /")
        assert r.failed
        assert "不允许" in r.output

        # cd
        (Path(tmp) / "sub").mkdir()
        r = term.invoke(command="cd sub")
        assert r.succeeded
        r = term.invoke(command="pwd")
        assert "sub" in r.output
    print("✅ TerminalTool 通过")


def test_read_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        rf = builtin.ReadFileTool(workspace=tmp)

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
    print("✅ ReadFileTool 通过")


# ─────────────────────────────────────────────────────────
# 6. 注册表批量操作 / hook / tag 过滤
# ─────────────────────────────────────────────────────────

def test_registry_batch_and_hook():
    reg = ToolRegistry()
    reg.register_tools(
        builtin.CalculatorTool(),
        builtin.ReadFileTool(workspace="/tmp"),
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
        test_nested_model_ref_expansion,
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
