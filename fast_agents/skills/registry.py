"""
Skill注册表：统一管理代码型Skill和文件型Skill
提供注册、查找、加载、列出等功能
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base import BaseSkill
from .loader import SkillLoader
from .models import SkillLoadResult, SkillMetadata


class SkillRegistry:
    """
    Skill注册表：统一管理所有类型的Skill
    
    支持:
        - 代码型Skill：通过register()注册BaseSkill子类或@skill装饰的函数
        - 文件型Skill：通过SkillLoader自动扫描SKILL.md目录
    """
    
    def __init__(self, extra_skill_paths: Optional[List[Path]] = None):
        self._code_skills: Dict[str, BaseSkill] = {}
        self._code_metadata: Dict[str, SkillMetadata] = {}
        self.loader = SkillLoader(extra_paths=extra_skill_paths)
        self._tools = None  # 延迟初始化
    
    def register(self, skill: Union[BaseSkill, Type[BaseSkill], Any]) -> None:
        """
        注册代码型Skill
        
        支持三种方式:
            1. 传入BaseSkill实例: registry.register(MySkill())
            2. 传入BaseSkill子类: registry.register(MySkill)
            3. 传入@skill装饰的函数: registry.register(my_skill_func)
        """
        skill_instance: Optional[BaseSkill] = None
        
        # 处理@skill装饰的函数
        if hasattr(skill, 'skill_class'):
            skill_instance = skill.skill_class()
        # 处理BaseSkill子类
        elif isinstance(skill, type) and issubclass(skill, BaseSkill):
            skill_instance = skill()
        # 处理BaseSkill实例
        elif isinstance(skill, BaseSkill):
            skill_instance = skill
        else:
            raise TypeError(f"无法注册类型 {type(skill)}，请传入BaseSkill子类/实例或@skill装饰的函数")
        
        name = skill_instance.metadata.name
        if name in self._code_skills:
            print(f"[SkillRegistry] 警告: Skill '{name}' 已存在，将被覆盖")
        
        self._code_skills[name] = skill_instance
        self._code_metadata[name] = skill_instance.metadata
    
    def unregister(self, skill_name: str) -> bool:
        """注销代码型Skill"""
        if skill_name in self._code_skills:
            del self._code_skills[skill_name]
            del self._code_metadata[skill_name]
            return True
        return False
    
    def add_skill_path(self, path: Union[str, Path], source: str = "custom") -> None:
        """添加文件型Skill扫描路径"""
        self.loader.add_skill_path(Path(path), source=source)
    
    def scan(self) -> None:
        """扫描文件型Skill，重建索引"""
        self.loader.scan_skills()
    
    def get_all_metadata(self) -> Dict[str, SkillMetadata]:
        """获取所有Skill的元数据（代码型+文件型）"""
        # 确保文件型已扫描
        file_metadata = self.loader.scan_skills()
        
        # 合并：代码型优先？还是文件型优先？这里采用后注册优先
        all_metadata = {}
        all_metadata.update(file_metadata)
        all_metadata.update(self._code_metadata)
        return all_metadata
    
    def get_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """获取指定Skill的元数据"""
        all_meta = self.get_all_metadata()
        return all_meta.get(skill_name)
    
    def load_skill(self, skill_name: str) -> SkillLoadResult:
        """加载完整Skill内容（Level 2）"""
        # 先检查代码型Skill
        if skill_name in self._code_skills:
            return self._code_skills[skill_name].load()
        
        # 再检查文件型Skill
        return self.loader.load_skill(skill_name)
    
    def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """执行代码型Skill"""
        if skill_name not in self._code_skills:
            # 文件型Skill不直接execute，按指令执行即可
            return f"文件型Skill '{skill_name}' 请按照加载的指令执行，可调用run_script运行脚本"
        
        return self._code_skills[skill_name].execute(**kwargs)
    
    async def aexecute_skill(self, skill_name: str, **kwargs) -> Any:
        """异步执行代码型Skill"""
        if skill_name not in self._code_skills:
            return self.execute_skill(skill_name, **kwargs)
        
        return await self._code_skills[skill_name].aexecute(**kwargs)
    
    def read_reference(self, skill_name: str, ref_name: str) -> str:
        """读取Skill参考文档"""
        if skill_name in self._code_skills:
            return f"代码型Skill '{skill_name}' 不支持参考文档"
        return self.loader.read_reference(skill_name, ref_name)
    
    def run_script(self, skill_name: str, script_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """运行Skill脚本"""
        if skill_name in self._code_skills:
            return f"代码型Skill '{skill_name}' 不支持脚本执行"
        return self.loader.run_script(skill_name, script_name, args)
    
    def get_index_prompt(self) -> str:
        """获取注入System Prompt的Skill索引文本（Level 1常驻）"""
        all_meta = self.get_all_metadata()
        
        if not all_meta:
            return "## 可用技能\n当前没有可用的技能。"
        
        lines = ["## 可用技能（Skills）", ""]
        lines.append("你可以使用以下技能完成特定任务，采用渐进式加载机制：")
        lines.append("")
        
        for name, meta in sorted(all_meta.items()):
            type_tag = " [代码]" if meta.skill_type == "code" else " [文件]"
            lines.append(f"- **{name}**{type_tag}: {meta.description}")
        
        lines.extend([
            "",
            "### 技能使用方法",
            "1. 根据用户需求判断是否需要使用某个技能",
            "2. 使用 `load_skill` 工具加载对应技能的完整指令",
            "3. 按照加载后的执行流程完成任务",
            "4. 需要参考文档时使用 `read_skill_reference`，需要执行脚本时使用 `run_skill_script`",
            "",
            "注意：不要猜测技能的内部流程，务必先加载再执行。",
        ])
        
        return "\n".join(lines)
    
    def get_tools(self) -> List[Any]:
        """
        获取Skill相关的工具，供Agent绑定使用
        返回三个工具：load_skill, read_skill_reference, run_skill_script
        """
        if self._tools is None:
            self._tools = self._create_skill_tools()
        return self._tools
    
    def _create_skill_tools(self) -> List[Any]:
        """创建Skill系统内置的三个工具"""
        # 这里不依赖你的Tool基类，而是返回可被你现有tools系统包装的函数
        # 你可以根据你现有的tool装饰器/基类进行适配
        
        def load_skill(skill_name: str) -> str:
            """
            加载指定技能的完整执行指令。当你判断需要使用某个技能时，首先调用此工具获取详细的执行流程。
            
            Args:
                skill_name: 要加载的技能名称，必须是上面列表中存在的技能名
            """
            result = self.load_skill(skill_name)
            if not result.success:
                return f"加载技能失败: {'; '.join(result.errors)}"
            return result.full_prompt
        
        def read_skill_reference(skill_name: str, reference_name: str) -> str:
            """
            读取技能的参考文档内容。当技能指令中提到需要参考某个文档时，调用此工具获取详细内容。
            
            Args:
                skill_name: 所属技能名称
                reference_name: 参考文档的文件名
            """
            return self.read_reference(skill_name, reference_name)
        
        def run_skill_script(skill_name: str, script_name: str, args: Optional[Dict[str, Any]] = None) -> str:
            """
            执行技能关联的脚本。当技能指令要求运行某个脚本时使用。
            
            Args:
                skill_name: 所属技能名称
                script_name: 脚本文件名（含扩展名）
                args: 传递给脚本的参数，键值对形式
            """
            return self.run_script(skill_name, script_name, args)
        
        return [load_skill, read_skill_reference, run_skill_script]
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有可用Skill信息"""
        all_meta = self.get_all_metadata()
        return [meta.to_dict() for meta in all_meta.values()]
