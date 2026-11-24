"""Agent基类

E10 异步契约：除同步 ``run`` / ``run_stream`` 外，统一提供**非抽象**的
``arun`` / ``arun_stream`` 默认实现，使 Agent 层可被上层（FastAPI / Jupyter /
异步服务）直接 ``await``，不再被迫用 ``asyncio.run(agent.run(...))`` 包一层
（那种包法在已有事件循环的环境里会直接 RuntimeError）。

默认实现不阻塞子类的选择权：
- 子类没写异步主循环时，``arun_stream`` 透传同步 ``run_stream``（行为等同 ``run``，
  只是签名可 await，LLM 调用期间会占用事件循环）；
- 子类有原生异步主循环时（如 ``ReActAgent.arun_stream``）直接覆盖即可。

⚠️ 两个异步方法**不能**加 ``@abstractmethod``：否则现存所有只实现了同步接口的
AgentBase 子类（含用户自定义）实例化时直接 TypeError，属破坏性改动（R1）。
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Optional

from ..messages import Messages
from ..messages.message import Message
from ..llm import LLM
from ..llm.models import StreamChunk


class AgentBase(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, llm: LLM, system_prompt: Optional[str] = None):
        self.name = name
        self.llm : LLM = llm
        self.system_prompt = system_prompt
        self.messages_simple: Messages = Messages()

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent，返回最终答案字符串"""
        pass
    
    @abstractmethod
    def run_stream(self, input_text: str, **kwargs) -> Iterator[StreamChunk]:
        """流式运行Agent，按StreamChunk契约产出事件，支持打字机效果"""
        pass

    # ---------- 异步入口（E10） ----------
    # 注意：以下两个方法刻意保持**非抽象**，给只实现同步接口的子类兜底。

    async def arun(self, input_text: str, **kwargs) -> str:
        """异步运行Agent，返回最终答案字符串。

        默认实现与 ``run()`` 复用 ``run_stream()`` 的写法逐字对齐：聚合 text 增量
        拼接为最终答案；本轮无 text 增量时（如 return_direct 短路）改用 end 事件
        携带的完整正文。子类若实现了原生异步主循环，应连同 ``arun_stream`` 一起覆盖。

        ⚠ 本基类不知道输出护栏的存在，因此**刻意不**把「end 正文优先」的护栏策略
        （A2）下沉到默认实现——否则会与子类自定义 ``run()`` 的聚合语义不一致。
        挂了输出护栏的 Agent 请像 ``ReActAgent.arun`` 那样覆盖本方法，返回护栏后正文。
        """
        final_text_parts: List[str] = []
        end_content: Optional[str] = None
        async for chunk in self.arun_stream(input_text, **kwargs):
            if chunk.type == "text":
                final_text_parts.append(chunk.text)
            elif chunk.type == "end":
                end_content = chunk.content
        if final_text_parts:
            return "".join(final_text_parts)
        return end_content or ""

    async def arun_stream(self, input_text: str, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式运行Agent，按 StreamChunk 契约产出事件。

        默认实现把同步 ``run_stream`` 的事件逐条透传，保证「任何 Agent 都能被 await」
        这条底线；代价是同步实现里的阻塞 I/O（LLM 调用）会占住事件循环，行为与
        ``run()`` 在事件循环里直跑一致。需要真异步并发的子类请覆盖本方法
        （参照 ``ReActAgent.arun_stream``：LLM 走 ``astream_invoke``、中间件钩子走
        ``a*`` 版本、工具走 ``ToolRegistry.ainvoke_tool``）。
        """
        for chunk in self.run_stream(input_text, **kwargs):
            yield chunk
    
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self.messages_simple.append(message)
    
    def clear_history(self):
        """清空历史记录"""
        self.messages_simple.clear()
    
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self.messages_simple.get_messages()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    def __repr__(self) -> str:
        return self.__str__()
