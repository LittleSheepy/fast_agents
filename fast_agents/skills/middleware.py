"""
Skill中间件：将Skill系统集成到Agent请求流程中
自动注入Skill索引到System Prompt，管理Skill上下文生命周期与allowed_tools工具白名单
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from .registry import SkillRegistry


# Skill系统内置工具名，加载/过滤时始终放行（否则会把技能加载工具自己封掉）
SKILL_BUILTIN_TOOLS = {"load_skill", "read_skill_reference", "run_skill_script", "list_loaded_skills"}


class SkillMiddleware:
    """
    Skill中间件：在Agent请求前后处理Skill相关逻辑
    
    核心功能：
    1. 启动时自动注入Skill索引到System Prompt（Level 1）
    2. 对话中自动清理已加载的Skill上下文（可选：用完即卸载，节省token）
    3. 绑定Skill工具到Agent
    4. 记录Skill调用统计
    5. 【C10】根据Skill声明的allowed_tools自动过滤LLM可见工具列表，技能卸载后自动恢复
    """
    
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        auto_inject_index: bool = True,
        unload_after_turn: bool = False,
        max_loaded_skills: int = 5,
        tracer = None,
        strict_tool_filter: bool = False,
    ):
        """
        Args:
            registry: Skill注册表实例，None则自动创建
            auto_inject_index: 是否自动注入Skill索引到System Prompt
            unload_after_turn: 是否每轮对话后卸载已加载的Skill（节省上下文）
            max_loaded_skills: 同时加载的最大Skill数量，超过则卸载最早加载的
            tracer: TraceLogger实例，用于技能操作打点
            strict_tool_filter: True=严格模式，调用不在白名单内的工具直接拦截并返回错误；
                False=仅过滤LLM可见的工具清单（system prompt + tool schema），实际调用时不拦截
                （默认False，给LLM自由度，同时避免误拦截非预期工具调用）
        """
        self.registry = registry or SkillRegistry()
        self.auto_inject_index = auto_inject_index
        self.unload_after_turn = unload_after_turn
        self.max_loaded_skills = max_loaded_skills
        self.strict_tool_filter = strict_tool_filter
        
        # 注入tracer
        if tracer is not None:
            self.registry.set_tracer(tracer)
        
        # 当前已加载的Skill列表（LRU顺序）
        self._loaded_skills: List[str] = []
        # 已加载Skill的完整内容缓存
        self._loaded_content: Dict[str, str] = {}
        # 调用统计
        self._call_stats: Dict[str, int] = {}
        
        # 【C10】工具白名单状态
        # 已加载技能的allowed_tools映射：skill_name -> set(allowed_tool_names)
        self._skill_allowed_tools: Dict[str, Set[str]] = {}
        # 当前计算出的工具白名单（None表示无过滤，放行所有工具）
        self._current_allowed_tools: Optional[Set[str]] = None
        # 绑定的ToolRegistry引用（延迟绑定，在bind_tool_registry时设置）
        self._tool_registry: Optional[Any] = None
        # 原始工具列表备份（绑定registry时记录一次，用于恢复）
        self._original_tool_names: Optional[Set[str]] = None
    
    def bind_tool_registry(self, tool_registry: Any) -> None:
        """
        【C10】绑定Agent的ToolRegistry实例，用于工具过滤
        ReActAgent._init_skills后会调用此方法完成绑定，同时替换list_loaded_skills工具的占位实现
        """
        self._tool_registry = tool_registry
        # 记录原始工具列表
        if tool_registry is not None:
            self._original_tool_names = set(tool_registry.list_names())
            
            # 替换list_loaded_skills工具的占位实现为真实版本
            list_tool = tool_registry.get_tool("list_loaded_skills")
            if list_tool is not None:
                from ..tools import StructuredTool
                def _list_loaded_skills_impl() -> str:
                    """列出当前已加载的技能，以及每个技能允许使用的工具列表"""
                    if not self._loaded_skills:
                        return "当前没有加载任何技能，可以使用全部工具。"
                    lines = ["当前已加载技能列表：", ""]
                    for skill_name in self._loaded_skills:
                        meta = self.registry.get_metadata(skill_name)
                        allowed = meta.allowed_tools if meta else []
                        if "*" in allowed:
                            allowed_text = "允许全部工具"
                        elif allowed:
                            allowed_text = f"允许工具: {', '.join(sorted(allowed))}"
                        else:
                            allowed_text = "无工具限制"
                        lines.append(f"- {skill_name}: {allowed_text}")
                    lines.append("")
                    if self._current_allowed_tools is not None:
                        non_builtin = sorted(n for n in self._current_allowed_tools if n not in SKILL_BUILTIN_TOOLS)
                        lines.append(f"当前可使用工具（白名单合并结果）：{', '.join(non_builtin)}")
                    else:
                        lines.append("当前可使用工具：全部工具（无技能限制）")
                    return "\n".join(lines)
                
                # 替换工具的func
                list_tool.func = _list_loaded_skills_impl
            
            # 绑定后立即重新计算白名单（可能已有预加载技能）
            self._recompute_tool_whitelist()
    
    def _recompute_tool_whitelist(self) -> None:
        """
        【C10】根据当前已加载技能重新计算工具白名单
        规则：
        1. 没有任何已加载技能 -> 白名单为None，放行所有工具
        2. 多个技能同时加载 -> 所有技能allowed_tools的并集
        3. 任何技能声明allowed_tools: ["*"] -> 白名单为None，放行所有工具
        4. SKILL_BUILTIN_TOOLS中的工具始终放行
        """
        if not self._skill_allowed_tools:
            self._current_allowed_tools = None
            return
        
        merged: Set[str] = set()
        allow_all = False
        for tool_set in self._skill_allowed_tools.values():
            if "*" in tool_set:
                allow_all = True
                break
            merged.update(tool_set)
        
        if allow_all:
            self._current_allowed_tools = None
        else:
            # 始终放通技能内置工具
            merged.update(SKILL_BUILTIN_TOOLS)
            self._current_allowed_tools = merged
    
    def get_filtered_tools(self) -> Optional[List[Any]]:
        """
        【C10】获取过滤后的工具列表
        返回None表示不需要过滤，使用原始全部工具；否则返回过滤后的ToolBase列表
        """
        if self._tool_registry is None or self._current_allowed_tools is None:
            return None
        
        return [
            t for t in self._tool_registry.list_tools()
            if t.name in self._current_allowed_tools
        ]
    
    def get_filtered_tools_description(self) -> Optional[str]:
        """
        【C10】获取过滤后的工具描述文本（用于注入system prompt）
        返回None表示使用原始描述
        """
        filtered = self.get_filtered_tools()
        if filtered is None:
            return None
        
        if not filtered:
            return "当前技能限制下暂无可用工具"
        
        return "\n".join(f"- {t.name}: {t.description}" for t in filtered)
    
    def get_filtered_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """【C10】获取过滤后的OpenAI格式tools schema，返回None表示不过滤"""
        filtered = self.get_filtered_tools()
        if filtered is None:
            return None
        return [t.to_openai_tool() for t in filtered]
    
    def get_filtered_anthropic_tools(self) -> Optional[List[Dict[str, Any]]]:
        """【C10】获取过滤后的Anthropic格式tools schema，返回None表示不过滤"""
        filtered = self.get_filtered_tools()
        if filtered is None:
            return None
        return [t.to_anthropic_tool() for t in filtered]
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        【C10】检查工具是否在当前白名单中
        strict_tool_filter=True时，invoke_tool前会调用此方法做拦截
        """
        if self._current_allowed_tools is None:
            return True
        return tool_name in self._current_allowed_tools
    
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
            
            # 【C10】工具限制提示
            if self._current_allowed_tools is not None:
                allowed_list = sorted(n for n in self._current_allowed_tools if n not in SKILL_BUILTIN_TOOLS)
                if allowed_list:
                    parts.extend([
                        "### ⚠️ 工具使用限制",
                        f"当前已加载技能限制你只能使用以下工具：{', '.join(allowed_list)}",
                        "请不要尝试调用列表之外的工具，它们当前不可用。",
                        "",
                    ])
        
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
        """标记Skill已加载，管理LRU缓存，并同步更新工具白名单"""
        # 如果已加载，移到末尾
        if skill_name in self._loaded_skills:
            self._loaded_skills.remove(skill_name)
        
        self._loaded_skills.append(skill_name)
        self._loaded_content[skill_name] = content
        self._call_stats[skill_name] = self._call_stats.get(skill_name, 0) + 1
        
        # 【C10】提取该技能的allowed_tools并更新白名单
        meta = self.registry.get_metadata(skill_name)
        if meta is not None and meta.allowed_tools:
            self._skill_allowed_tools[skill_name] = set(meta.allowed_tools)
            self._recompute_tool_whitelist()
        
        # 超过最大数量时，卸载最早加载的
        while len(self._loaded_skills) > self.max_loaded_skills:
            oldest = self._loaded_skills.pop(0)
            self._unload_skill_internal(oldest)
    
    def _unload_skill_internal(self, skill_name: str) -> None:
        """内部卸载技能：清理缓存 + 更新白名单"""
        if skill_name in self._loaded_content:
            del self._loaded_content[skill_name]
        # 【C10】移除该技能的allowed_tools记录并重算白名单
        if skill_name in self._skill_allowed_tools:
            del self._skill_allowed_tools[skill_name]
            self._recompute_tool_whitelist()
    
    def unload_skill(self, skill_name: str) -> bool:
        """手动卸载已加载的Skill"""
        if skill_name in self._loaded_skills:
            self._loaded_skills.remove(skill_name)
            self._unload_skill_internal(skill_name)
            return True
        return False
    
    def clear_loaded_skills(self) -> None:
        """清空所有已加载的Skill，工具白名单同步恢复为放行全部"""
        self._loaded_skills.clear()
        self._loaded_content.clear()
        # 【C10】清空白名单映射，恢复无过滤状态
        self._skill_allowed_tools.clear()
        self._current_allowed_tools = None
    
    def get_loaded_skills(self) -> List[str]:
        """获取当前已加载的Skill列表"""
        return self._loaded_skills.copy()
    
    def get_tools(self) -> List[Any]:
        """获取Skill系统提供的工具列表，注册到Agent的tool系统"""
        return self.registry.build_skill_tools()
    
    def get_stats(self) -> Dict[str, int]:
        """获取Skill调用统计"""
        return self._call_stats.copy()
