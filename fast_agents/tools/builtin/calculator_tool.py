"""安全的数学计算器（AST 白名单解析，不执行任意代码）"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="要计算的数学表达式，支持 +-*/^ 和函数 sqrt/sin/cos/tan/log/exp/abs/round/max/min，如 '2+3*4'、'sqrt(16)'、'sin(pi/2)'",
    )


class CalculatorTool(ToolBase):
    name = "python_calculator"
    description = (
        "执行数学计算。支持基本运算（+ - * / ^）和常用数学函数 "
        "（sqrt, sin, cos, tan, log, exp, abs, round, max, min），常量 pi、e。"
        "适用于需要精确数值答案的场景。"
    )
    args_schema = CalculatorInput
    tags = ["builtin", "math"]

    # 二元运算符
    _BINOPS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.BitXor: operator.xor,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    }

    # 单元运算符
    _UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
    _FUNCS = {
        "abs": abs, "round": round, "max": max, "min": min, "sum": sum,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
        "ceil": math.ceil, "floor": math.floor,
    }
    _CONSTS = {"pi": math.pi, "e": math.e}

    def run(self, expression: str) -> Union[str, ToolResult]:
        """
        执行计算

        Args:
            expression: 要计算的数学表达式，支持基本运算和数学函数

        Returns:
            计算结果
        """
        if not expression or not expression.strip():
            return ToolResult.fail("表达式不能为空")
        expr = expression.strip()
        try:
            tree = ast.parse(expr, mode="eval")
            value = self._eval(tree.body)
        except ZeroDivisionError:
            return ToolResult.fail("除零错误：请检查表达式分母")
        except (SyntaxError, ValueError) as e:
            return ToolResult.fail(f"表达式语法错误：{e}。请使用合法的数学表达式，如 '2+3*4'")
        except Exception as e:
            return ToolResult.fail(f"计算失败：{e}")
        if isinstance(value, float) and value.is_integer():
            formatted = str(int(value))
        else:
            formatted = str(value)
        return ToolResult.ok(f"{expression} = {formatted}", artifact={"value": value, "expression": expression})

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的字面量：{node.value!r}")
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._BINOPS:
                raise ValueError(f"不支持的运算符：{op_type.__name__}")
            return self._BINOPS[op_type](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._UNARYOPS:
                raise ValueError(f"不支持的一元运算符：{op_type.__name__}")
            return self._UNARYOPS[op_type](self._eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持普通函数调用")
            fname = node.func.id
            if fname not in self._FUNCS:
                raise ValueError(f"不支持的函数：{fname}")
            return self._FUNCS[fname](*[self._eval(a) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in self._CONSTS:
                return self._CONSTS[node.id]
            raise ValueError(f"未定义的变量：{node.id}")
        raise ValueError(f"不支持的表达式节点：{type(node).__name__}")
