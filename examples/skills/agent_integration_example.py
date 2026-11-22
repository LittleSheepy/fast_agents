"""
fast_agents Skill系统集成示例
展示如何将Skill模块集成到你现有的Agent中
"""
from fast_agents.skills import SkillRegistry, SkillMiddleware


class YourExistingAgent:
    """
    这里模拟你已有的Agent类，展示接入点
    你只需要在3个地方加入少量代码即可
    """
    
    def __init__(self, system_prompt: str, tools=None, memory=None):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.memory = memory
        
        # ========== 接入点1：初始化Skill系统 ==========
        # 创建Skill注册表，自动扫描项目级和用户级skills目录
        self.skill_registry = SkillRegistry()
        
        # 可选：注册自定义代码型Skill
        from code_skill_example import calculate, DataAnalysisSkill
        self.skill_registry.register(calculate)
        self.skill_registry.register(DataAnalysisSkill)
        
        # 创建Skill中间件
        self.skill_middleware = SkillMiddleware(
            registry=self.skill_registry,
            auto_inject_index=True,      # 自动注入Skill索引到system prompt
            unload_after_turn=False,     # 多轮对话可保留已加载的Skill
            max_loaded_skills=3,         # 最多同时加载3个Skill，避免上下文过长
        )
        
        # ========== 接入点2：注册Skill内置工具到你的tool系统 ==========
        # Skill系统提供3个内置工具：load_skill, read_skill_reference, run_skill_script
        skill_tools = self.skill_middleware.get_tools()
        self.tools.extend(skill_tools)
        
        print(f"Skill系统初始化完成，共发现 {len(self.skill_registry.list_skills())} 个技能")
    
    def _build_messages(self, user_input: str):
        """构建发送给LLM的消息列表（你已有的方法）"""
        messages = []
        
        # ========== 接入点3：构建system prompt时注入Skill内容 ==========
        # 在原有system prompt后追加Skill索引和已加载Skill内容
        full_system_prompt = self.skill_middleware.before_agent_run(self.system_prompt)
        messages.append({"role": "system", "content": full_system_prompt})
        
        # ... 你原有的memory加载逻辑 ...
        if self.memory:
            history = self.memory.get_history()
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_input})
        return messages
    
    def _handle_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """处理工具调用（你已有的方法）"""
        # ... 你原有的tool调用逻辑 ...
        
        # ========== 接入点4：工具调用后通知Skill中间件 ==========
        # 这一步很重要：它会自动记录已加载的Skill，追加到下一轮prompt中
        # result = ... 你原来执行tool得到结果
        # 假设你执行完工具得到result后：
        result = "工具执行结果"  # 替换为你的实际结果
        
        # 通知Skill中间件处理结果（自动捕获load_skill事件）
        result = self.skill_middleware.after_tool_call(tool_name, tool_args, result)
        
        return result
    
    def chat(self, user_input: str) -> str:
        """对话入口（你已有的方法）"""
        messages = self._build_messages(user_input)
        
        # ... 你原有的LLM调用、ReAct循环、工具调用逻辑 ...
        # 示例伪代码：
        # while True:
        #     response = llm.chat(messages)
        #     if response.has_tool_calls:
        #         for tool_call in response.tool_calls:
        #             tool_result = self._handle_tool_call(tool_call.name, tool_call.args)
        #             messages.append({"role": "tool", "content": tool_result})
        #     else:
        #         answer = response.content
        #         break
        
        # ========== 接入点5：每轮结束后处理 ==========
        self.skill_middleware.after_agent_turn()
        
        return "Agent回复"


# ========== 系统提示词优化建议 ==========
"""
为了让Skill机制更好地工作，建议在你的Base System Prompt中加入以下段落：

---

## 技能使用规范

你拥有一系列可复用的专业技能（Skills）来帮助你更好地完成任务。

### 技能使用原则
1. **先索引后加载**：系统提示词中只显示技能名称和描述，看到匹配的技能时，先调用`load_skill`加载完整指令，不要猜测技能的执行流程。
2. **严格遵循指令**：加载技能后，严格按照其中的步骤和规范执行，不要跳步或省略步骤。
3. **用完即走**：技能指令只在当前任务中有效，完成任务后无需专门提及加载了什么技能。
4. **脚本执行**：如果技能中要求运行脚本，使用`run_skill_script`工具，参数按照技能说明传递。
5. **参考文档**：如果需要查看参考文档细节，使用`read_skill_reference`工具按需读取。

### 判断是否使用技能的标准
- 如果用户的请求匹配技能描述中的触发场景，必须加载对应技能。
- 简单问题、通用知识问答无需加载技能。
- 一个任务可能需要多个技能配合，按需依次加载即可。
- 如果拿不准是否该用，可以先加载技能看看具体流程是否匹配。

---
"""


if __name__ == "__main__":
    # 使用示例
    SYSTEM_PROMPT = """你是一个有用的AI助手，可以使用各种工具和技能来帮助用户完成任务。
请用中文回答用户的问题。"""
    
    agent = YourExistingAgent(system_prompt=SYSTEM_PROMPT)
