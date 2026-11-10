"""短期记忆（Working Memory / Buffer）

职责：只管「当前这次会话聊到哪了」，不负责跨会话。
- 保存最近 N 轮对话消息（内部用 Messages 容器承载）；
- 超出窗口时，把被挤出去的消息交给 LLM 压成一句摘要（滚动摘要），
  这样既不丢信息，又不让 prompt 无限膨胀；
- 可选传入 persist_store（SessionStore）实现短期落盘：
  append 同步落库、构造时 load 回填、进程重启能恢复上次会话（对应演进路线 B1）。

对应 HelloAgents 的 working memory（working_memory_capacity + TTL），
但这里用「消息窗口 + 摘要」而不是纯 TTL，因为对 ReAct Agent 更实用。
"""

from typing import List, Optional

from ..messages import Message, Messages
from .models import ContextPacket
from .session_store import SessionStore


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
    """会话级短期记忆：滑动窗口 + 滚动摘要

    内部消息存储委托给 Messages 容器（纯粹的 Message 列表 + OpenAI 格式转换），
    窗口裁剪、滚动摘要、持久化等「记忆」逻辑全部在这里实现，不侵入 Messages。
    """

    def __init__(
        self,
        max_messages: int = 20,
        enable_summary: bool = True,
        summary_max_words: int = 200,
        llm=None,
        summary_prompt: Optional[str] = None,
        persist_store: Optional[SessionStore] = None,
    ):
        """
        Args:
            max_messages: 窗口内最多保留多少条消息（user+assistant 各算一条）
            enable_summary: 是否开启滚动摘要（关闭则直接丢弃最老的消息）
            summary_max_words: 摘要字数上限
            llm: LLM 实例，仅在 enable_summary=True 时需要
            summary_prompt: 自定义摘要提示词
            persist_store: 可选的短期持久化后端（SQLiteSessionStore / InMemorySessionStore）；
                传入后：
                1. 构造时自动从 persist_store.load() 回填历史消息与摘要；
                2. 每次 append / 压缩产生新摘要时，同步落库；
                3. clear() 时同步清空持久化层。
                不传则保持原有纯内存行为（进程退出即丢）。
        """
        if max_messages < 2:
            raise ValueError("max_messages 至少为 2")
        self.max_messages = max_messages
        self.enable_summary = enable_summary
        self.summary_max_words = summary_max_words
        self.llm = llm
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self.persist_store = persist_store
        self._messages: Messages = Messages()
        self._summary: str = ""
        # 累计 append 过的消息条数（B10 增量抽取水位线用）：只增不减，窗口裁剪不会让它回退
        self._total_appended: int = 0
        # 启动时先从持久化层回填（没有 persist_store 时保持空）
        if persist_store is not None:
            loaded = persist_store.load()
            self._messages.extend(loaded)
            self._total_appended = len(loaded)
            self._summary: str = persist_store.load_summary() or ""
            # 回填后如果超过窗口，立刻裁剪保摘要（溢出消息可能已被摘要过，不重新摘要）
            if len(self._messages) > self.max_messages:
                overflow_count = len(self._messages) - self.max_messages
                self._messages = Messages(self._messages[overflow_count:])

    # ---------- 写入 ----------

    def append(self, message: Message) -> None:
        """追加一条消息，必要时触发压缩；若开启了持久化则同步落库"""
        self._messages.append(message)
        self._total_appended += 1
        if self.persist_store is not None:
            # 持久化失败不能拖垮主流程（和摘要失败同样的容错策略）
            try:
                self.persist_store.append(message)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ 短期记忆持久化 append 失败（已写入内存）: {e}")
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
        overflow: List[Message] = self._messages[:overflow_count]
        # 用切片重建 Messages，窗口裁剪逻辑保留在记忆模块
        self._messages = Messages(self._messages[overflow_count:])
        if self.enable_summary:
            self._summary = self._summarize(overflow, self._summary)
            if self.persist_store is not None:
                try:
                    self.persist_store.save_summary(self._summary)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ 短期记忆持久化 save_summary 失败: {e}")

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
        return self._messages.get_messages()

    # Buffer里加这个方法，完全内部处理裁剪偏移
    def get_messages_since(self, global_offset: int) -> List[Message]:
        """返回全局累计序号 >= global_offset 的、当前窗口内仍留存的消息
        global_offset 是 _total_appended 坐标系下的序号，和窗口裁剪无关
        """
        if global_offset >= self._total_appended:
            return []
        # 全局序号转窗口内下标：窗口里最老的消息对应的全局序号 = 总累计 - 当前窗口长度
        window_start_global_seq = self._total_appended - len(self._messages)
        start_idx_in_window = max(0, global_offset - window_start_global_seq)
        return self.get_messages()[start_idx_in_window:]

    def get_summary(self) -> str:
        return self._summary

    def get_total_appended(self) -> int:
        """自创建（或上次 clear 之后）以来累计 append 过的消息条数

        与 len(self) 不同：滑动窗口裁剪最老消息时它只增不减、不会回退。
        MemoryManager 用它作为增量抽取水位线，从而窗口写满后仍能持续抽取新消息。
        """
        return self._total_appended

    def to_openai_messages(self) -> List[dict]:
        """转成 OpenAI messages 列表（供 Agent 直接喂给 LLM）

        摘要作为一条 user 消息前缀注入，再拼接窗口内所有消息；
        消息的 OpenAI 格式转换委托给 Messages.to_openai_messages()，
        避免重复实现 to_openai_message 的遍历与拉平逻辑。
        """
        result: List[dict] = []
        if self._summary:
            result.append(
                {
                    "role": "user",
                    "content": f"[历史摘要，供参考]\n{self._summary}",
                }
            )
        result.extend(self._messages.to_openai_messages())
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
        self._messages.clear()
        self._summary = ""
        self._total_appended = 0
        if self.persist_store is not None:
            try:
                self.persist_store.clear()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ 短期记忆持久化 clear 失败: {e}")

    def __len__(self) -> int:
        return len(self._messages)
