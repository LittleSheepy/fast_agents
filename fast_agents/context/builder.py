"""上下文构建器（ContextBuilder）—— 模块门面

对齐 ``MemoryManager`` 的写法：对外只暴露少量动作，Agent 侧不需要知道
底下分了哪些层、用哪种压缩策略。

    build()     把 system / 记忆 / 历史 / 文件 / 本轮问题 分层拼装 + 预算裁剪
    observe()   工具结果回来后，判断是否触发卸载
    record()    一轮结束后，把本轮消息写回工作区与摘要

与 memory 的衔接（两条路，都不侵入 memory 代码）：
1. **被动**：memory 的 ``MemoryManager.build_context()`` 已产出 ``ContextPacket``，
   builder 直接把这些包渲染成一段文字塞进 memory 层；
2. **主动**：调用方也可以把任意 ``ContextPacket`` 列表喂给 ``extra_packets``，
   让 builder 统一按预算排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..messages.message import Message
from .compressor import CompressionReport, ContextCompressor
from .layers import ContextLayer, ContextWindow
from .offload import ContextOffloader
from .tokens import TokenBudget, TokenCounter

__all__ = ["ContextBuilder", "BuildReport"]


@dataclass
class BuildReport:
    """一次 build() 的结果报告（给日志与测试用）"""

    tokens: int = 0
    stats: List[Dict[str, Any]] = None  # type: ignore[assignment]
    compression: Optional[CompressionReport] = None
    warnings: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.stats = self.stats or []
        self.warnings = self.warnings or []

    def describe(self) -> str:
        """多行打印，形如::

            上下文构成：3 层 / 8 条消息 / 1240 token
              system     1 条   820 token  required
              history    6 条   320 token
              user       1 条   100 token  required
        """
        lines = [f"上下文构成：{len(self.stats)} 层 / "
                 f"{sum(s['messages'] for s in self.stats)} 条消息 / {self.tokens} token"]
        for s in self.stats:
            flag = "  required" if s["required"] else ""
            lines.append(f"  {s['name']:<10} {s['messages']:>2} 条 {s['tokens']:>6} token{flag}")
        if self.compression is not None and (
            self.compression.offloaded or self.compression.trimmed
        ):
            lines.append("  " + self.compression.summary_line())
        for w in self.warnings:
            lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)


class ContextBuilder:
    """上下文工程门面：分层 → 拼装 → 预算 → 压缩"""

    def __init__(
        self,
        *,
        workspace: str = ".",
        budget: Optional[TokenBudget] = None,
        counter: Optional[TokenCounter] = None,
        offloader: Optional[ContextOffloader] = None,
        compressor: Optional[ContextCompressor] = None,
        llm: Any = None,
        offload_threshold: int = 4_000,
        layer_quotas: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            workspace: 工作目录（卸载文件与文件工具都以它为根）
            budget: token 预算；不传用默认 128k 窗口
            counter: token 计数器；不传自动探测 tiktoken
            offloader: 卸载器；不传自动按 workspace 建一个
            compressor: 压缩器；不传自动组装三级策略
            llm: LLM 实例（摘要用），压缩器未显式传入时会用它
            offload_threshold: 触发卸载的字符数阈值
            layer_quotas: 各层 token 配额比例，默认 system 0.35 / memory 0.25 /
                files 0.25 / history 剩余。比例之和不必为 1，多余的视为弹性空间
            verbose: 是否打印每轮的上下文构成
        """
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.budget = budget or TokenBudget()
        self.counter = counter or TokenCounter()
        self.offloader = offloader or ContextOffloader(
            workspace=str(self.workspace), threshold=offload_threshold
        )
        self.compressor = compressor or ContextCompressor(
            llm=llm, offloader=self.offloader
        )
        self.layer_quotas = layer_quotas or {
            "system": 0.35,
            "memory": 0.25,
            "files": 0.25,
        }
        self.verbose = verbose
        self.window = ContextWindow()
        self.last_report: Optional[BuildReport] = None

    # ---------- 内部：配额换算 ----------

    def _quota_for(self, layer_name: str) -> int:
        """把比例换算成绝对 token 配额；未配置的层分剩余额度"""
        if layer_name in self.layer_quotas:
            return int(self.budget.available * self.layer_quotas[layer_name])
        fixed = sum(
            int(self.budget.available * r) for r in self.layer_quotas.values()
        )
        return max(1, self.budget.available - fixed)

    # ---------- 核心：拼装 ----------

    def build(
        self,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[Iterable[Message]] = None,
        user_input: Optional[str] = None,
        memory_packets: Optional[Sequence[Any]] = None,
        file_snippets: Optional[Sequence[str]] = None,
        extra_packets: Optional[Sequence[Any]] = None,
    ) -> ContextWindow:
        """按层拼装上下文并按预算裁剪，返回可直接喂给 LLM 的 ContextWindow

        Args:
            system_prompt: L1 system 文本（静态，跨轮不变 → prompt caching 锚点）
            history: L3 历史消息（通常是 ``MemoryManager.buffer.get_messages()``）
            user_input: L5 本轮用户输入（永远排在最后）
            memory_packets: 记忆包（``ContextPacket`` 或 ``{"content": ...}`` dict）
            file_snippets: 需要注入的文件片段（读到/卸载回读的内容）
            extra_packets: 其他需要参与预算排队的文本包

        Returns:
            ContextWindow（层级有序，已超预算的部分已按策略压缩）
        """
        self.window = ContextWindow()
        warnings: List[str] = []

        # ---- L1 system：required，不参与裁剪 ----
        if system_prompt:
            system_layer = ContextLayer(name="system", required=True, priority=100)
            system_layer.add(
                Message("system", system_prompt, metadata={"type": "system", "required": True})
            )
            self.window.add_layer(system_layer)

        # ---- L2 memory：长期记忆 / 会话摘要 ----
        memory_text = self._render_packets(memory_packets) if memory_packets else ""
        if memory_text:
            memory_layer = ContextLayer(name="memory", priority=80)
            memory_layer.add(
                Message(
                    "user",
                    f"[参考记忆，非用户本轮发言]\n{memory_text}",
                    metadata={"type": "memory", "source": "memory"},
                )
            )
            memory_layer.token_count  # noqa: B018 —— 触发一次计数便于阅读
            self._fit_layer(memory_layer, self._quota_for("memory"), self.counter, warnings)
            if not memory_layer.is_empty():
                self.window.add_layer(memory_layer)

        # ---- L3 history：滚动窗口，最容易爆的一块 ----
        history_layer = ContextLayer(name="history", priority=60)
        for message in history or []:
            if message.role == "system":
                continue  # system 由 L1 统一负责，历史里的 system 一律丢弃，避免重复注入
            history_layer.add(message)
        report = self.compressor.compress(
            history_layer, self.counter, self._quota_for("history")
        )
        if not history_layer.is_empty():
            self.window.add_layer(history_layer)

        # ---- L4 files：卸载回读 / 文件片段 ----
        if file_snippets:
            files_layer = ContextLayer(name="files", priority=40)
            for snippet in file_snippets:
                files_layer.add(
                    Message(
                        "user",
                        f"[文件内容参考]\n{snippet}",
                        metadata={"type": "file", "source": "filesystem"},
                    )
                )
            self._fit_layer(files_layer, self._quota_for("files"), self.counter, warnings)
            if not files_layer.is_empty():
                self.window.add_layer(files_layer)

        # ---- L5 user：required，永远最后 ----
        if user_input:
            user_layer = ContextLayer(name="user", required=True, priority=100)
            user_layer.add(Message("user", user_input, metadata={"type": "user", "required": True}))
            self.window.add_layer(user_layer)

        # ---- 全局兜底：极端情况下整体仍超预算，从最老的 history 继续丢 ----
        self._enforce_global_budget(warnings)

        tokens = self.window.token_count(self.counter)
        self.last_report = BuildReport(
            tokens=tokens,
            stats=self.window.layer_stats(self.counter),
            compression=report,
            warnings=warnings,
        )
        if self.verbose:
            print("🧭 " + self.last_report.describe())
        return self.window

    # ---------- 层内配额裁剪 ----------

    def _fit_layer(
        self,
        layer: ContextLayer,
        limit: int,
        counter: TokenCounter,
        warnings: List[str],
    ) -> None:
        """把非 history 层压到配额以内：优先卸载，其次丢最老的条目"""
        if layer.token_count(counter) <= limit:
            return
        self.compressor._offload_layer(layer, CompressionReport())  # noqa: SLF001 —— 内部策略复用
        while layer.messages and layer.token_count(counter) > limit:
            layer.messages.pop(0)
        if layer.token_count(counter) > limit:
            warnings.append(f"{layer.name} 层即使清空仍超配额（配额 {limit} token）")

    def _enforce_global_budget(self, warnings: List[str]) -> None:
        """全局兜底：整体超预算时，从最老的 history 消息成组丢弃"""
        if not self.window.token_count(self.counter) > self.budget.available:
            return
        history = self.window.get_layer("history")
        if history is None:
            warnings.append("已无可压缩的历史层，上下文可能超出模型窗口")
            return
        from .compressor import _group_messages  # 局部导入，避免循环引用困扰

        groups = _group_messages(history.messages)
        dropped = 0
        while groups and self.window.token_count(self.counter) > self.budget.available:
            if len(groups) <= self.compressor.keep_recent:
                break
            dropped += len(groups.pop(0))
        history.messages = [m for g in groups for m in g]
        if dropped:
            warnings.append(f"全局兜底：再裁剪 {dropped} 条最老历史")

    # ---------- 渲染 ----------

    @staticmethod
    def _render_packets(packets: Sequence[Any]) -> str:
        """把 ContextPacket / dict / str 统一渲染成一段文本"""
        lines: List[str] = []
        for packet in packets:
            if packet is None:
                continue
            if isinstance(packet, str):
                lines.append(packet)
                continue
            if isinstance(packet, dict):
                content = packet.get("content", "")
            else:
                content = getattr(packet, "content", "")
            if content:
                lines.append(str(content))
        return "\n".join(f"- {line}" for line in lines)

    # ---------- 工具结果：观察与卸载 ----------

    def observe(self, tool_name: str, content: str) -> str:
        """工具结果回来后调一次：超长就卸载，返回可放进 tool 消息的文本

        Agent 主循环里每次工具执行完都会走这里，因此必须「无脑调用也安全」：
        没超阈值时原样返回，不产生任何 IO。
        """
        return self.offloader.offload(tool_name, content)

    # ---------- 收尾：写回与重置 ----------

    def record_turn(self, user_input: str, answer: str) -> None:
        """一轮结束后记录到工作区（Agent 可用它维护自己的短期历史副本）

        注意：**跨轮历史仍然只由 memory.buffer 持有**（单一数据源原则），
        这里只往 history 层追加，不涉及持久化，避免又造出第二份历史。
        """
        history = self.window.get_layer("history")
        if history is None:
            history = ContextLayer(name="history", priority=60)
            self.window.add_layer(history)
        history.add(Message("user", user_input, metadata={"type": "history", "source": "turn"}))
        history.add(Message("assistant", answer, metadata={"type": "history", "source": "turn"}))

    def reset_turn(self) -> None:
        """清空本轮工作区（每轮 run 开始时调用，避免上一轮残留）"""
        self.window = ContextWindow()

    # ---------- 状态 ----------

    @property
    def summary(self) -> str:
        """压缩器累积的摘要（可注入下一轮 memory 层）"""
        return self.compressor.accumulated_summary

    def stats(self) -> Dict[str, Any]:
        """当前上下文的概要统计"""
        return {
            "tokens": self.window.token_count(self.counter),
            "budget": self.budget.available,
            "usage_ratio": self.budget.usage_ratio(self.window.token_count(self.counter)),
            "layers": [layer.name for layer in self.window.ordered_layers()],
            "offloaded_files": len(self.offloader),
        }

    def close(self) -> None:
        """释放资源（当前无长连接，留作扩展位，与 MemoryManager.close 对称）"""
        return None

    def __repr__(self) -> str:
        return (
            f"<ContextBuilder layers={[l.name for l in self.window.ordered_layers()]} "
            f"budget={self.budget.available} counter={self.counter.backend}>"
        )
