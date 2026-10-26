"""
Skill基类与装饰器定义
支持两种Skill定义方式：
1. 继承BaseSkill，代码型Skill
2. 使用@skill装饰器，快速定义Skill
"""
from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .models import SkillMetadata, SkillLoadResult


class BaseSkill:
    """
    所有代码型Skill的基类
    
    用法:
        class MyDataAnalysisSkill(BaseSkill):
            name = "data-analysis"
            description = "分析CSV数据并生成统计报告，处理数据清洗、计算、可视化"
            version = "1.0.0"
            tags = ["data", "analysis", "csv"]
            
            def execute(self, **kwargs) -> str:
                # Skill执行逻辑
                return "分析结果"
            
            def get_instructions(self) -> str:
                return '''
                数据分析执行流程：
                1. 首先读取CSV文件
                2. 清洗数据（处理缺失值、异常值）
                3. 计算统计指标
                4. 生成报告
                '''
    """
    
    # 类属性，子类必须覆盖
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: List[str] = []
    tags: List[str] = []
    extra_metadata: Dict[str, Any] = {}
    
    def __init__(self):
        if not self.name:
            raise ValueError(f"Skill类 {self.__class__.__name__} 必须定义name属性")
        if not self.description:
            raise ValueError(f"Skill类 {self.__class__.__name__} 必须定义description属性")
        
        self._metadata = SkillMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            license=self.license,
            compatibility=self.compatibility,
            allowed_tools=self.allowed_tools,
            tags=self.tags,
            metadata=self.extra_metadata,
            skill_type="code",
            source="code",
        )
        
        # 校验元数据
        errors = self._metadata.validate()
        if errors:
            raise ValueError(f"Skill元数据校验失败: {'; '.join(errors)}")
    
    @property
    def metadata(self) -> SkillMetadata:
        return self._metadata
    
    def get_instructions(self) -> str:
        """获取Skill的执行指令（Level 2加载时调用），子类可覆盖"""
        # 默认从execute方法的docstring获取
        if self.execute.__doc__:
            return inspect.cleandoc(self.execute.__doc__)
        return f"执行{self.name}技能"
    
    def execute(self, **kwargs) -> Any:
        """执行Skill的核心逻辑，子类必须实现"""
        raise NotImplementedError("子类必须实现execute方法")
    
    async def aexecute(self, **kwargs) -> Any:
        """异步执行，子类可覆盖"""
        return self.execute(**kwargs)
    
    def load(self) -> SkillLoadResult:
        """加载Skill，返回加载结果"""
        return SkillLoadResult(
            success=True,
            skill_name=self.name,
            metadata=self._metadata,
            instructions=self.get_instructions(),
        )
    
    def run_script(self, script_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """运行Skill关联的脚本（代码型Skill通常不使用，文件型Skill使用）"""
        raise NotImplementedError("代码型Skill不支持run_script")
    
    def read_reference(self, ref_name: str) -> str:
        """读取参考文档（代码型Skill通常不使用，文件型Skill使用）"""
        raise NotImplementedError("代码型Skill不支持read_reference")


def skill(
    name: str,
    description: str,
    version: str = "1.0.0",
    tags: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
    **extra_metadata
):
    """
    @skill装饰器：快速定义代码型Skill
    
    用法:
        @skill(
            name="calculate",
            description="执行数学计算，支持加减乘除、复杂表达式求值",
            tags=["math", "calculation"]
        )
        def calculate(expression: str) -> str:
            \"\"\"
            计算执行流程：
            1. 解析数学表达式
            2. 安全执行计算
            3. 返回结果
            \"\"\"
            # 注意：实际使用时要做安全校验
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
    """
    def decorator(func: Callable) -> BaseSkill:
        # 获取函数docstring作为instructions
        docstring = inspect.cleandoc(func.__doc__) if func.__doc__ else f"执行{name}技能"
        
        # 创建动态Skill类
        class DecoratedSkill(BaseSkill):
            def __init__(self):
                self.name = name
                self.description = description
                self.version = version
                self.tags = tags or []
                self.allowed_tools = allowed_tools or []
                self.extra_metadata = extra_metadata
                super().__init__()
                
                self._func = func
                self._instructions = docstring
            
            def get_instructions(self) -> str:
                return self._instructions
            
            def execute(self, **kwargs) -> Any:
                return self._func(**kwargs)
            
            async def aexecute(self, **kwargs) -> Any:
                if inspect.iscoroutinefunction(self._func):
                    return await self._func(**kwargs)
                return self._func(**kwargs)
        
        # 保持原函数名
        DecoratedSkill.__name__ = func.__name__
        DecoratedSkill.__qualname__ = func.__qualname__
        DecoratedSkill.__module__ = func.__module__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            skill_instance = DecoratedSkill()
            return skill_instance.execute(*args, **kwargs)
        
        # 附加Skill类到wrapper，方便注册
        wrapper.skill_class = DecoratedSkill
        wrapper.skill_metadata = DecoratedSkill().metadata
        
        return wrapper
    
    return decorator
