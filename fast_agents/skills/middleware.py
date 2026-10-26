"""
Skill中间件：将Skill系统集成到Agent请求流程中
自动注入Skill索引到System Prompt，管理Skill上下文生命周期
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .registry import SkillRegistry


class SkillMiddleware:
    """
    Skill中间件：在Agent请求前后处理Skill相关逻辑
    
    核心功能：
    1. 启动时自动注入Skill索引到System Prompt（Level 1）
    2. 对话中自动清理已加载的Skill上下文（可选：用完即卸载，节省token）
    3. 绑定Skill工具到Agent
    4. 记录Skill调用统计
    """
    
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        auto_inject_index: bool = True,
        unload_after_turn: bool = False,
        max_loaded_skills: int = 5,
    ):
        """
        Args:
            registry: Skill注册表实例，None则自动创建
            auto_inject_index: 是否自动注入Skill索引到System Prompt
            unload_after_turn: 是否每轮对话后卸载已加载的Skill（节省上下文）
            max_loaded_skills: 同时加载的最大Skill数量，超过则卸载最早加载的
        """
        self.registry = registry or SkillRegistry()
        self.auto_inject_index = auto_inject_index
        self.unload_after_turn = unload_after_turn
        self.max_loaded_skills = max_loaded_skills
        
        # 当前已加载的Skill列表（LRU顺序）
        self._loaded_skills: List[str] = []
        # 已加载Skill的完整内容缓存
        self._loaded_content: Dict[str, str] = {}
        # 调用统计
        self._call_stats: Dict[str, int] = {}
    
    def get_system_prompt_suffix(self) -> str:
        """
        获取追加到System Prompt末尾的Skill相关内容
        在Agent构建system prompt时调用
        """
        if not self.auto_inject_index:
            return ""
        
        parts = []
        
        # 1. 注入Skill索引（Level 1）
        index_prompt = self.registry.get_index_prompt()
        parts.append(index_prompt)
        
        # 2. 注入已加载的Skill内容（Level 2）
        if self._loaded_skills:
            parts.extend([
                "",
                "## 已加载的技能详情",
                "",
                "以下技能已加载，请严格按照其中的指令执行：",
                "",
            ])
            for skill_name in self._loaded_skills:
                if skill_name in self._loaded_content:
                    parts.append(self._loaded_content[skill_name])
                    parts.append("")
        
        return "\n".join(parts)
    
    def before_agent_run(self, system_prompt: str) -> str:
        """Agent运行前调用：注入Skill内容到system prompt"""
        suffix = self.get_system_prompt_suffix()
        if suffix:
            return system_prompt + "\n\n" + suffix
        return system_prompt
    
    def after_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_result: str) -> str:
        """工具调用后调用：处理load_skill等工具的结果"""
        if tool_name == "load_skill":
            skill_name = tool_args.get("skill_name", "")
            if skill_name and not tool_result.startswith("加载技能失败"):
                self._mark_skill_loaded(skill_name, tool_result)
        
        elif tool_name == "run_skill_script":
            skill_name = tool_args.get("skill_name", "")
            if skill_name:
                self._call_stats[skill_name] = self._call_stats.get(skill_name, 0) + 1
        
        return tool_result
    
    def after_agent_turn(self) -> None:
        """每轮Agent对话结束后调用"""
        if self.unload_after_turn:
            self.clear_loaded_skills()
    
    def _mark_skill_loaded(self, skill_name: str, content: str) -> None:
        """标记Skill已加载，管理LRU缓存"""
        # 如果已加载，移到末尾
        if skill_name in self._loaded_skills:
            self._loaded_skills.remove(skill_name)
        
        self._loaded_skills.append(skill_name)
        self._loaded_content[skill_name] = content
        self._call_stats[skill_name] = self._call_stats.get(skill_name, 0) + 1
        
        # 超过最大数量时，卸载最早加载的
        while len(self._loaded_skills) > self.max_loaded_skills:
            oldest = self._loaded_skills.pop(0)
            del self._loaded_content[oldest]
    
    def unload_skill(self, skill_name: str) -> bool:
        """手动卸载已加载的Skill"""
        if skill_name in self._loaded_skills:
            self._loaded_skills.remove(skill_name)
            del self._loaded_content[skill_name]
            return True
        return False
    
    def clear_loaded_skills(self) -> None:
        """清空所有已加载的Skill"""
        self._loaded_skills.clear()
        self._loaded_content.clear()
    
    def get_loaded_skills(self) -> List[str]:
        """获取当前已加载的Skill列表"""
        return self._loaded_skills.copy()
    
    def get_tools(self) -> List[Any]:
        """获取Skill系统提供的工具列表，注册到Agent的tool系统"""
        return self.registry.get_tools()
    
    def get_stats(self) -> Dict[str, int]:
        """获取Skill调用统计"""
        return self._call_stats.copy()
