"""
代码型Skill使用示例
展示两种定义方式：继承BaseSkill类、使用@skill装饰器
"""
from fast_agents.skills import BaseSkill, skill, SkillRegistry


# ========== 方式1：使用@skill装饰器（最简单）==========

@skill(
    name="calculate",
    description="执行安全的数学计算，支持加减乘除、幂运算、常用数学函数。当用户需要计算数学表达式时使用。",
    tags=["math", "calculation"],
    version="1.0.0"
)
def calculate(expression: str) -> str:
    """
    数学计算执行规范：
    
    流程：
    1. 检查表达式是否包含危险内容（import、os、sys、__、exec、eval等关键字直接拒绝）
    2. 仅允许以下字符：数字 0-9、运算符 +-*/^().、以及以下函数名：sqrt, sin, cos, tan, log, abs, round
    3. 使用安全的命名空间执行计算
    4. 返回计算结果
    
    注意：
    - 禁止执行任何与数学计算无关的代码
    - 结果保留4位小数
    - 出错时返回明确的错误信息
    """
    import math
    
    # 安全校验
    dangerous = ["import", "os", "sys", "__", "exec", "eval", "open", "file", "compile"]
    for d in dangerous:
        if d in expression.lower():
            return f"计算错误：表达式包含禁止的关键字 '{d}'"
    
    # 允许的函数
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # 替换^为**（幂运算）
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {expression} = {round(result, 4)}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ========== 方式2：继承BaseSkill类（适合复杂Skill）==========

class DataAnalysisSkill(BaseSkill):
    name = "simple-data-analysis"
    description = "简单的CSV数据分析，计算统计指标（均值、最大值、最小值、中位数），无需依赖pandas"
    version = "1.0.0"
    author = "fast-agents"
    tags = ["data", "analysis", "csv"]
    
    def get_instructions(self) -> str:
        return """
        CSV数据分析执行流程：
        
        1. 首先确认CSV文件路径，如果用户没提供，询问用户
        2. 使用文件读取工具读取CSV内容
        3. 解析CSV数据（第一行为表头）
        4. 对数值列计算以下统计指标：
           - 计数(count)
           - 均值(mean)
           - 最大值(max)
           - 最小值(min)
           - 中位数(median)
        5. 将结果以Markdown表格形式输出
        6. 如果有非数值列，统计其唯一值数量
        
        注意：如果CSV文件超过1000行，提示用户数据量较大是否继续。
        """
    
    def execute(self, csv_content: str, **kwargs) -> str:
        """执行简单的CSV数据分析"""
        import csv
        import io
        from statistics import mean, median
        
        f = io.StringIO(csv_content)
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if not rows:
            return "没有数据"
        
        result = ["# 数据分析结果\n"]
        result.append(f"共 {len(rows)} 条数据\n")
        
        # 分析每一列
        for col in reader.fieldnames:
            values = [row[col] for row in rows if row[col].strip()]
            result.append(f"## 列: {col}")
            
            # 尝试数值分析
            try:
                num_values = [float(v) for v in values]
                result.append("| 指标 | 值 |")
                result.append("|------|----|")
                result.append(f"| 计数 | {len(num_values)} |")
                result.append(f"| 均值 | {round(mean(num_values), 2)} |")
                result.append(f"| 最大值 | {max(num_values)} |")
                result.append(f"| 最小值 | {min(num_values)} |")
                result.append(f"| 中位数 | {median(num_values)} |")
            except ValueError:
                # 非数值列
                unique = len(set(values))
                result.append(f"- 计数: {len(values)}")
                result.append(f"- 唯一值数量: {unique}")
                if unique <= 10:
                    from collections import Counter
                    cnt = Counter(values)
                    result.append("- 值分布:")
                    for v, c in cnt.most_common():
                        result.append(f"  - {v}: {c}次")
            
            result.append("")
        
        return "\n".join(result)


# ========== 注册到SkillRegistry ==========

def setup_skills():
    registry = SkillRegistry()
    
    # 注册代码型Skill
    registry.register(calculate)  # @skill装饰的函数
    registry.register(DataAnalysisSkill)  # BaseSkill子类
    
    # 扫描文件型Skill（自动扫描./skills/和~/.fast_agents/skills/）
    registry.scan()
    tools = registry.get_tools()
    skils = registry.list_skills()
    prompts = registry.get_index_prompt()
    # 添加自定义Skill路径
    # registry.add_skill_path("D:/my-custom-skills")
    
    # 列出所有可用Skill
    print("已加载的Skill:")
    for skill_meta in registry.list_skills():
        print(f"  - {skill_meta['name']} ({skill_meta['skill_type']}): {skill_meta['description'][:50]}...")
    
    return registry


if __name__ == "__main__":
    setup_skills()
    
    # 测试执行代码型Skill
    registry = setup_skills()
    print("\n测试calculate Skill:")
    print(registry.execute_skill("calculate", expression="2 + 3 * 4"))
    print(registry.execute_skill("calculate", expression="sqrt(16) + 2^3"))
