"""ReAct Agent实现 - 推理与行动结合的智能体"""
import re
from typing import Optional, List, Tuple

from .prompt import DEFAULT_REACT_PROMPT
from ...agents.agent_base import AgentBase
from ...llm import LLM
from ...messages import Messages
from ...messages.message import Message, MsgToolResult
from ...tools.tool_registry import ToolRegistry


class ReActAgent(AgentBase):
    """
    ReAct (Reasoning and Acting) Agent
    
    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    
    这是一个经典的Agent范式，特别适合需要外部信息的任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: LLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的工具注册表）
            system_prompt: 系统提示词
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        super().__init__(name, llm, system_prompt)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.messages: Messages = Messages()

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT
        self.system_prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def add_tool(self, tool):
        """
        添加工具到工具注册表
        支持MCP工具的自动展开

        Args:
            tool: 工具实例(可以是普通Tool或MCPTool)
        """
        # 检查是否是MCP工具
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            # MCP工具会自动展开为多个工具
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    # 创建包装工具
                    name = f"{tool.name}_{mcp_tool['name']}"
                    description = mcp_tool.get('description', '')
                    func = lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                        "action": "call_tool",
                        "tool_name": tn,
                        "arguments": {"input": input_text}
                    })
                    self.tool_registry.register_function(name, description, func)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(tool._available_tools)} 个独立工具")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent
        
        Args:
            input_text: 用户问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        self.current_history = []
        current_step = 0
        
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 构建提示词
            tools = self.tool_registry.to_openai_tools()
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            system_prompt = self.system_prompt_template.format(tools=tools_desc)
            
            # 调用LLM
            self.messages.append_system(system_prompt)
            self.messages.append_user(input_text)
            messages = self.messages.to_openai_messages()

            response : Message = self.llm.invoke(messages, tools=tools, **kwargs)

            self.messages.append(response)

            # 判断是否执行工具
            if response.tool_calls:
                message = Message("tool")
                for tool in response.tool_calls:
                    tool_result = self.tool_registry.execute_tool(tool.name, tool.arguments)
                    msg_tool_result = MsgToolResult(name=tool.name, result=tool_result, id=tool.id)
                    message.tool_results.append(msg_tool_result)
                self.messages.append(message)
                continue
            else:
                answer = response.content
                self.messages.append_ai(answer)
                print(f"🎉 最终答案: {answer}")
                return answer

        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        
        # 保存到历史记录
        self.add_message(Message("user", input_text))
        self.add_message(Message("assistant", final_answer))
        
        return final_answer
