"""短期记忆（Working Memory / Buffer）

职责：只管「当前这次会话聊到哪了」，不负责跨会话。
- 保存最近 N 轮对话消息；
- 超出窗口时，把被挤出去的消息交给 LLM 压成一句摘要（滚动摘要），
  这样既不丢信息，又不让 prompt 无限膨胀。

对应 HelloAgents 的 working memory（working_memory_capacity + TTL），
但这里用「消息窗口 + 摘要」而不是纯 TTL，因为对 ReAct Agent 更实用。

注意：本模块不落盘。会话结束即释放，真正的持久化交给长期记忆。
"""

from typing import List, Optional

from ..messages.message import Message
from .schema import ContextPacket


# 与 prompt.py 一样，摘要提示词写成模块级常量，方便用户自定义
DEFAULT_SUMMARY_PROMPT = """请把下面这段对话压缩成不超过 {max_words} 个字的中文摘要。

要求：
1. 保留用户的身份信息、明确表达的偏好、正在进行的任务目标；
2. 保留关键结论和已确认的事实，去掉寒暄和重复内容；
3. 只输出摘要正文，不要任何解释、前缀或 Markdown 标记。

对话内容：
{history}
"""


class BufferMemory:
    """会话级短期记忆：滑动窗口 + 滚动摘要"""

    def __init__(
        self,
        max_messages: int = 20,
        enable_summary: bool = True,
        summary_max_words: int = 200,
        llm=None,
        summary_prompt: Optional[str] = None,
    ):
        """
        Args:
            max_messages: 窗口内最多保留多少条消息（user+assistant 各算一条）
            enable_summary: 是否开启滚动摘要（关闭则直接丢弃最老的消息）
            summary_max_words: 摘要字数上限
            llm: LLM 实例，仅在 enable_summary=True 时需要
            summary_prompt: 自定义摘要提示词
        """
        if max_messages < 2:
            raise ValueError("max_messages 至少为 2")
        self.max_messages = max_messages
        self.enable_summary = enable_summary
        self.summary_max_words = summary_max_words
        self.llm = llm
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT

        self._messages: List[Message] = []
        self._summary: str = ""

    # ---------- 写入 ----------

    def append(self, message: Message) -> None:
        """追加一条消息，必要时触发压缩"""
        self._messages.append(message)
        self._maybe_compress()

    def append_user(self, content: str) -> None:
        self.append(Message("user", content))

    def append_ai(self, content: str) -> None:
        self.append(Message("assistant", content))

    def _maybe_compress(self) -> None:
        """超出窗口时，把溢出的最老消息吃掉并更新摘要"""
        if len(self._messages) <= self.max_messages:
            return
        overflow_count = len(self._messages) - self.max_messages
        overflow = self._messages[:overflow_count]
        self._messages = self._messages[overflow_count:]
        if self.enable_summary:
            self._summary = self._summarize(overflow, self._summary)

    def _summarize(self, messages: List[Message], previous: str) -> str:
        """调用 LLM 生成摘要；调用失败时降级为拼接截断，保证不中断主流程"""
        history = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)
        if previous:
            history = f"（此前摘要：{previous}）\n{history}"
        if not self.llm:
            # 没有 llm 时走降级路径：截断保存，至少不丢关键片段
            return history[: self.summary_max_words * 2]
        prompt = self.summary_prompt.format(
            max_words=self.summary_max_words, history=history
        )
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            text = (response.content or "").strip()
            return text or previous
        except Exception as e:  # noqa: BLE001 —— 摘要失败不能拖垮对话
            print(f"⚠️ 摘要生成失败，降级为截断保存: {e}")
            return history[: self.summary_max_words * 2]

    # ---------- 读取 ----------

    def get_messages(self) -> List[Message]:
        """返回窗口内的消息副本"""
        return list(self._messages)

    def get_summary(self) -> str:
        return self._summary

    def to_openai_messages(self) -> List[dict]:
        """转成 OpenAI messages 列表（供 Agent 直接喂给 LLM）"""
        result: List[dict] = []
        if self._summary:
            result.append(
                {
                    "role": "user",
                    "content": f"[历史摘要，供参考]\n{self._summary}",
                }
            )
        result.extend(m.to_openai_message() for m in self._messages)
        return result

    def to_context_packet(self) -> List[ContextPacket]:
        """转成 ContextPacket，交给 ContextBuilder 统一编排预算"""
        packets: List[ContextPacket] = []
        if self._summary:
            packets.append(
                ContextPacket(
                    content=f"[历史摘要] {self._summary}",
                    metadata={"type": "summary", "source": "buffer", "required": False},
                )
            )
        for m in self._messages:
            packets.append(
                ContextPacket(
                    content=f"{m.role}: {m.content}",
                    metadata={"type": "history", "source": "buffer", "required": False},
                )
            )
        return packets

    def clear(self) -> None:
        self._messages = []
        self._summary = ""

    def __len__(self) -> int:
        return len(self._messages)
