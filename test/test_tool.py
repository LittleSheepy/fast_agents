from fast_agents.tools.calculator_tool import CalculatorTool, calculate
from fast_agents.tools.terminal_tool import TerminalTool

def test_CalculatorTool():
    calculator = CalculatorTool()
    openai_like_tool = calculator.build_openai_like_tool()
    print(openai_like_tool)

def test_calculate():
    result = calculate("1+1+4")
    print(result)

def test_tool_or_tool():
    calculator_tool = CalculatorTool()
    terminal_tool = TerminalTool()
    tools = calculator_tool + terminal_tool
    print(tools)



if __name__ == '__main__':
    print("")
    test_tool_or_tool()

