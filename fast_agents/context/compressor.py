"""上下文压缩策略：卸载 → 裁剪 → 摘要

预算超了怎么办？不同框架各有各的招（LangChain 的 ``trim_messages``、
LangGraph 的 SummarizationNode、Anthropic 的 compaction）。这里把它做成
**可插拔的三级策略**，由 ``ContextCompressor`` 串起来调用：

    第 1 级  卸载 offload   —— 把超长 tool 结果写到磁盘，上下文只留引用（最优，信息不丢）
    第 2 级  裁剪 trim      —— 从最老的历史开始成对丢弃 user/assistant（快，会丢信息）
    第 3 级  摘要 summarize —— 把丢弃的历史交给 LLM 压成一段话（慢，但保住要点）

设计约束（都是踩过的坑）：
- **tool 消息绝不能单独丢**：``assistant(tool_calls)`` 与 ``tool`` 结果必须成对出现，
  否则 OpenAI 强校验直接 400。所以裁剪以「消息组（turn）」为单位；
- **system 与本轮 user 不参与压缩**：它们由 required 标记保护，压缩器直接跳过；
- **摘要失败降级**：LLM 调用异常时退回「截断保存」，绝不让摘要拖垮整个对话
  （与 ``BufferMemory._summarize`` 的处理保持一致）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..messages.message import Message
from .layers import ContextLayer
from .offload import ContextOffloader
from .tokens import TokenCounter

__all__ = ["CompressionReport", "ContextCompressor", "DEFAULT_COMPACT_PROMPT"]


DEFAULT_COMPACT_PROMPT = """请把下面这段历史对话压缩成不超过 {max_words} 个字的中文摘要。

要求：
1. 保留任务目标、已确认的事实与关键结论；
2. 保留用户明确表达的偏好与约束；
3. 去掉寒暄、重复内容与已完成的中间步骤；
4. 只输出摘要正文，不要解释、不要 Markdown 标记。

历史内容：
{history}
"""


@dataclass
class CompressionReport:
    """一次压缩的结果报告（便于日志/测试断言）"""

    offloaded: int = 0          # 被卸载的 tool 消息条数
    trimmed: int = 0            # 被裁掉的原始消息条数
    summarized: bool = False    # 是否生成了新的摘要
    tokens_before: int = 0
    tokens_after: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def summary_line(self) -> str:
        return (
            f"压缩：卸载 {self.offloaded} 条 / 裁剪 {self.trimmed} 条 / "
            f"摘要 {'是' if self.summarized else '否'}；"
            f"token {self.tokens_before} → {self.tokens_after}（省 {self.saved}）"
        )


class ContextCompressor:
    """三级压缩策略的执行者

    典型用法（不需要手动调，``ContextBuilder`` 里会自动走）::

        compressor = ContextCompressor(llm=llm, offloader=offloader)
        report = compressor.compress(history_layer, counter, budget)
    """

    def __init__(
        self,
        llm: Any = None,
        offloader: Optional[ContextOffloader] = None,
        enable_offload: bool = True,
        enable_trim: bool = True,
        enable_summary: bool = True,
        summary_max_words: int = 200,
        compact_prompt: Optional[str] = None,
        keep_recent: int = 4,
    ):
        """
        Args:
            llm: LLM 实例，摘要需要；不传时摘要策略自动跳过
            offloader: 卸载器；不传时不做卸载
            enable_offload / enable_trim / enable_summary: 三个策略的开关（可插拔）
            summary_max_words: 摘要字数上限
            compact_prompt: 自定义摘要提示词
            keep_recent: 裁剪时至少保留最近几条 user/assistant 消息
        """
        self.llm = llm
        self.offloader = offloader
        self.enable_offload = enable_offload
        self.enable_trim = enable_trim
        self.enable_summary = enable_summary
        self.summary_max_words = summary_max_words
        self.compact_prompt = compact_prompt or DEFAULT_COMPACT_PROMPT
        self.keep_recent = max(0, keep_recent)
        # 累积摘要（上一轮压出来的），下一轮会重新注入
        self._accumulated_summary: str = ""

    # ---------- 对外入口 ----------

    def compress(
        self,
        layer: ContextLayer,
        counter: TokenCounter,
        budget: Any,
    ) -> CompressionReport:
        """对某一层做压缩，直到其 token 落在「该层配额」之内

        Args:
            layer: 要压缩的层（通常是 history）
            counter: token 计数器
            budget: TokenBudget 或直接给一个 token 上限（int）

        Returns:
            CompressionReport
        """
        limit = budget.available if hasattr(budget, "available") else int(budget)
        report = CompressionReport()
        report.tokens_before = layer.token_count(counter)

        if report.tokens_before <= limit:
            report.tokens_after = report.tokens_before
            return report

        # 第 1 级：卸载超长消息（信息不丢，收益最大）
        if self.enable_offload and self.offloader is not None:
            self._offload_layer(layer, report)

        # 第 2 级：成组裁剪最老的历史
        if self.enable_trim and layer.token_count(counter) > limit:
            self._trim_layer(layer, limit, counter, report)

        # 第 3 级：仍然超预算 → 摘要（把裁掉的内容压成一段话）
        if self.enable_summary and layer.token_count(counter) > limit:
            self._summarize_layer(layer, limit, counter, report)

        report.tokens_after = layer.token_count(counter)
        return report

    # ---------- 第 1 级：卸载 ----------

    def _offload_layer(self, layer: ContextLayer, report: CompressionReport) -> None:
        """把层里超长的消息替换成「离线引用」"""
        offloader = self.offloader
        assert offloader is not None
        for message in layer.messages:
            content = message.content or ""
            if not offloader.should_offload(content):
                continue
            # system / 本轮 user 不卸载：它们是指令，外置会让模型看不到关键要求
            if message.role == "system":
                continue
            reference = offloader.offload(
                tool_name=(message.metadata or {}).get("name", message.role),
                text=content,
                metadata={"original_chars": len(content), "role": message.role},
            )
            if reference != content:
                message.content = reference
                report.offloaded += 1
                report.notes.append(f"卸载 1 条（原 {len(content)} 字符）")

    # ---------- 第 2 级：裁剪 ----------

    def _trim_layer(
        self,
        layer: ContextLayer,
        limit: int,
        counter: TokenCounter,
        report: CompressionReport,
    ) -> None:
        """从最老的消息开始成组丢弃，直到低于 limit

        成组 = 一个 ``assistant(tool_calls)`` + 紧随其后的若干 ``tool`` 结果，
        以及普通的 ``user``/``assistant`` 单条。绝不允许拆散 tool 配对。
        """
        groups = _group_messages(layer.messages)
        protected_tail = max(0, self.keep_recent)

        while groups and layer.token_count(counter) > limit:
            if len(groups) <= protected_tail:
                break
            # 只从「组」的开头丢，且不丢 required 的组
            head = groups[0]
            if any(m.role == "system" for m in head):
                break
            groups.pop(0)
            report.trimmed += len(head)

        if report.trimmed:
            report.notes.append(f"裁剪 {report.trimmed} 条最老消息")
            layer.messages = [m for group in groups for m in group]

    # ---------- 第 3 级：摘要 ----------

    def _summarize_layer(
        self,
        layer: ContextLayer,
        limit: int,
        counter: TokenCounter,
        report: CompressionReport,
    ) -> None:
        """把除最近若干条以外的历史压成一段摘要，替换回层里"""
        if self.llm is None:
            return
        groups = _group_messages(layer.messages)
        if len(groups) <= self.keep_recent:
            return

        to_compress = groups[: len(groups) - self.keep_recent]
        keep = [m for group in groups[len(groups) - self.keep_recent:] for m in group]
        history_text = "\n".join(
            f"{m.role}: {m.content}" for group in to_compress for m in group if m.content
        )
        if not history_text.strip():
            return

        summary = self._call_summarize(history_text)
        if not summary:
            return

        self._accumulated_summary = summary
        layer.messages = [
            Message(
                "system",
                f"[历史已压缩，供参考]\n{summary}",
                metadata={"type": "compacted_summary", "required": True},
            )
        ] + keep
        report.summarized = True
        report.trimmed += sum(len(g) for g in to_compress)
        report.notes.append(f"摘要压缩 {len(to_compress)} 组历史")

    def _call_summarize(self, history_text: str) -> str:
        """调用 LLM 生成摘要；失败降级为截断，绝不抛异常"""
        prompt = self.compact_prompt.format(
            max_words=self.summary_max_words, history=history_text
        )
        try:
            response = self.llm.invoke(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001 —— 摘要失败不能拖垮对话
            print(f"⚠️ 上下文摘要失败，降级为截断：{e}")
        return history_text[: self.summary_max_words * 2]

    # ---------- 属性 ----------

    @property
    def accumulated_summary(self) -> str:
        """累积摘要（可被 Agent 取走，作为跨轮的会话摘要注入）"""
        return self._accumulated_summary

    def reset(self) -> None:
        self._accumulated_summary = ""

    def __repr__(self) -> str:
        flags = []
        if self.enable_offload:
            flags.append("offload")
        if self.enable_trim:
            flags.append("trim")
        if self.enable_summary:
            flags.append("summary")
        return f"<ContextCompressor strategies={'+'.join(flags) or 'none'}>"


# ============================================================
# 工具函数
# ============================================================

def _group_messages(messages: Sequence[Message]) -> List[List[Message]]:
    """把消息按「原子组」切开

    - ``assistant(tool_calls)`` + 后续 ``tool`` 消息 → 一组（不可拆）；
    - 其余每条消息各自成组。

    这是「不许把 tool 配对拆坏」这条硬规则在代码里的落点。
    """
    groups: List[List[Message]] = []
    buffer: Optional[List[Message]] = None
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            buffer = [message]
            groups.append(buffer)
            continue
        if message.role == "tool" and buffer is not None:
            buffer.append(message)
            continue
        buffer = None
        groups.append([message])
    return groups
