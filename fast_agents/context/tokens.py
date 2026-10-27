"""Token 计数与上下文预算

为什么先做这个：**「裁不裁」的前提是「知道有多少」**。
现在的 buffer 按「消息条数」裁剪，但一条消息可能是 3 个 token，
也可能是一整篇文档。只有把 token 数算出来，才能按**预算**而不是条数管理上下文。

计数策略（三级降级，全部零新增依赖）：
1. tiktoken：装了就用，拿真实 BPE 分词（可选依赖，不装不影响框架运行）；
2. 启发式估算：中文字符 1 字 ≈ 1 token，英文/数字按 4 字符 ≈ 1 token，
   ASCII 标点按 2 字符 ≈ 1 token，再叠一个 safety_ratio 安全系数；
3. 注意：估算永远只是估算，所以要留余量（默认 1.15 倍），而不是卡着上限算。

对齐参考：LangChain 的 ``get_num_tokens_from_messages``、Anthropic 的 token 预算实践。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "TokenCounter",
    "TokenBudget",
    "estimate_text_tokens",
    "count_message_tokens",
]


# ============================================================
# 文本级估算
# ============================================================

def _is_cjk(ch: str) -> bool:
    """判断是否为 CJK 字符（中文/日文/韩文），这类字符在 BPE 里基本一字一 token"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一表意文字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0xF900 <= code <= 0xFAFF   # 兼容表意文字
        or 0x3040 <= code <= 0x30FF   # 日文假名
        or 0xAC00 <= code <= 0xD7AF   # 韩文
    )


def estimate_text_tokens(text: str, safety_ratio: float = 1.15) -> int:
    """纯启发式估算一段文本的 token 数（不依赖任何第三方库）

    规则（实测对 GPT 系列中文文本误差约 ±15%）：
    - 每个 CJK 字符 ≈ 1 token；
    - 英文单词 / 数字：4 个字符 ≈ 1 token；
    - 其余符号：2 个字符 ≈ 1 token。

    Args:
        text: 待估算文本
        safety_ratio: 安全系数，估算值会乘以它，宁可多算不可少算

    Returns:
        估算的 token 数（至少 0）
    """
    if not text:
        return 0
    cjk = 0
    latin = 0
    other = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            latin += 1
        else:
            other += 1
    tokens = cjk + latin / 4.0 + other / 2.0
    return int(tokens * safety_ratio) + 1


# ============================================================
# 消息级计数
# ============================================================

# OpenAI 每条消息的固定开销（role / 分隔符），和 LangChain 的做法一致
_PER_MESSAGE_OVERHEAD = 4
# 每次请求的固定开销
_PER_REQUEST_OVERHEAD = 3


def count_message_tokens(message: Dict[str, Any], counter: Optional["TokenCounter"] = None) -> int:
    """计算「一条已转成 OpenAI 格式的 dict 消息」的 token 数

    Args:
        message: 形如 ``{"role": "user", "content": "..."}`` 的 dict
        counter: 文本计数器；不传则用启发式估算

    Returns:
        该消息占用的 token 数（含固定开销）
    """
    counter = counter or TokenCounter(use_tiktoken=False)
    total = _PER_MESSAGE_OVERHEAD

    content = message.get("content")
    if isinstance(content, str):
        total += counter.count(content)
    elif isinstance(content, list):
        # 多模态内容块：只统计文本块，图片按固定 85 token 粗算
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                total += counter.count(str(block.get("text", "")))
            else:
                total += 85

    # function calling：工具名 + 参数 JSON 也要算进去
    for tool_call in message.get("tool_calls") or []:
        fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        total += counter.count(str(fn.get("name", "")))
        total += counter.count(str(fn.get("arguments", "")))
        total += 6

    return total


# ============================================================
# TokenCounter
# ============================================================

class TokenCounter:
    """token 计数器：优先 tiktoken，缺失时自动降级为启发式估算

    使用方式::

        counter = TokenCounter()                 # 自动探测 tiktoken
        counter = TokenCounter(use_tiktoken=False)  # 强制走估算
        counter.count("你好，世界")               # → int
        counter.count_messages(openai_messages)  # → int
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_tiktoken: bool = True,
        safety_ratio: float = 1.15,
    ):
        """
        Args:
            model: 模型名，用于匹配 tiktoken 的编码表
            use_tiktoken: 是否尝试使用 tiktoken（没装或加载失败会自动降级）
            safety_ratio: 启发式估算的安全系数
        """
        self.model = model
        self.safety_ratio = safety_ratio
        self._encoder = None
        self.backend = "heuristic"
        if use_tiktoken:
            self._encoder = self._load_encoder(model)
            if self._encoder is not None:
                self.backend = "tiktoken"

    @staticmethod
    def _load_encoder(model: str):
        """尝试加载 tiktoken 编码器；任何异常都返回 None（降级）"""
        try:
            import tiktoken  # type: ignore
        except Exception:  # noqa: BLE001 —— 没装就是没装，不能影响框架运行
            return None
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:  # noqa: BLE001 —— 模型名不认识时退回通用编码
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                return None

    # ---------- 计数 ----------

    def count(self, text: str) -> int:
        """计算一段文本的 token 数"""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:  # noqa: BLE001 —— 编码失败不能中断主流程
                pass
        return estimate_text_tokens(text, self.safety_ratio)

    def count_messages(self, messages: Iterable[Dict[str, Any]]) -> int:
        """计算一批 OpenAI 格式消息的 token 总数（含请求固定开销）"""
        total = _PER_REQUEST_OVERHEAD
        for m in messages:
            if isinstance(m, dict):
                total += count_message_tokens(m, self)
        return total

    def count_tool_schemas(self, tools: Optional[Sequence[Dict[str, Any]]]) -> int:
        """计算 tools（OpenAI function schema 列表）占用的 token

        工具 schema 是常驻 system 之外的一块大开销，容易被忽略：
        每轮请求都要带全套 schema，10 个工具轻松吃掉一两千 token。
        """
        if not tools:
            return 0
        try:
            raw = json.dumps(list(tools), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            raw = str(tools)
        return self.count(raw)

    def __repr__(self) -> str:
        return f"<TokenCounter backend={self.backend} model={self.model}>"


# ============================================================
# TokenBudget
# ============================================================

@dataclass
class TokenBudget:
    """上下文预算：把「模型窗口」切成「可用预算」和「预留回复空间」

    公式：``available = max_context_tokens - reserve_for_output - safety_margin``

    常见窗口：4k / 8k / 32k / 128k / 200k。
    留出回复空间很关键——不预留的话，输入占满窗口后模型没有余量输出，
    会出现「请求成功但回答被截断」或直接 400 的情况。
    """

    max_context_tokens: int = 128_000
    reserve_for_output: int = 4_000
    safety_margin: int = 1_000

    @property
    def available(self) -> int:
        """可用于「输入上下文」的 token 预算，下限保护为 1"""
        return max(1, self.max_context_tokens - self.reserve_for_output - self.safety_margin)

    def remaining(self, used: int) -> int:
        """还能再放多少 token"""
        return max(0, self.available - used)

    def usage_ratio(self, used: int) -> float:
        """预算使用率 0~1（超过 1 表示已超预算）"""
        return used / self.available if self.available else 1.0

    def is_over(self, used: int) -> bool:
        return used > self.available

    def describe(self, used: int) -> str:
        """给日志用的一行摘要"""
        return (
            f"token 预算 {used}/{self.available}"
            f"（{self.usage_ratio(used):.1%}，窗口 {self.max_context_tokens}，"
            f"预留输出 {self.reserve_for_output}）"
        )
