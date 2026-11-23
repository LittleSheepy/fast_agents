"""Prompt注入检测护栏 PromptInjectionGuardrail

检测常见的 Prompt 注入攻击模式，如「忽略之前的指令」「你现在是…」、
「不要遵守系统提示」，以及英文的 "ignore previous instructions"、
"do anything now"、"jailbreak" 等经典注入语句。

设计要点（L7 修订，回应「只覆盖中文、不做 lower、加空格/同形字即可绕过」的评审意见）：

1. **中英双语模式表**：新增英文注入特征（指令覆盖 / 越狱 / 系统提示套取），
   不再只覆盖中文。
2. **文本归一化**：匹配前先做 NFKC 归一化 + 转小写 + 去零宽字符，
   因此 "IGNORE PREVIOUS INSTRUCTIONS"、全角同形字 "ｉｇｎｏｒｅ" 都能命中。
3. **压缩视图**：在归一化文本基础上，再去掉空白与常见插入分隔符
   （空格 / 制表符 / `.` `-` `_` `*` 等）得到「压缩串」，并对压缩串再做一遍匹配。
   这样 "i g n o r e   previous instructions"、中文「忽 略 之 前 的 指 令」
   这类加空格 / 加符号的绕过也会被识别。

⚠️ 与所有基于规则的安全组件一样，本护栏无法覆盖语义改写型注入（例如换用完全
不同的措辞诱导模型），只作为「高召回、低成本」的确定性第一道防线。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from ..input_guardrail import InputGuardrail
from ..guardrail_base import to_scan_text
from ..models import GuardrailResult


# 常见 Prompt 注入特征模式。
#
# 说明：英文模式刻意写成「紧凑形式」（词与词之间不带空格），例如
# ``ignore.{0,20}?previous.{0,20}?instructions``，配合下面的压缩视图即可同时
# 覆盖 "ignore previous instructions" 与 "ignorepreviousinstructions"；
# 中英模式统一在「归一化串」与「压缩串」两个视图上匹配。
DEFAULT_INJECTION_PATTERNS = [
    # ── 中文：直接指令覆盖类 ──────────────────────────────
    r"忽略(之前|上面|以上|先前|所有|一切|掉)?的?(指令|提示|规则|要求|设定)",
    r"忘记(之前|上面|以上|先前|所有)的?(对话|内容|提示|指令)",
    r"无视(以上|之前|所有|一切)?(限制|规则|指令|提示)",
    r"不要(遵守|遵循|按照|理会)(系统|开发者|之前的)?(提示|指令|规则)",
    r"你现在(要|必须|应该|需要)?(扮演|充当|成为)",
    r"你不再是(助手|ai|人工智能|机器人|语言模型)",
    # ── 中文：越狱类 ─────────────────────────────────────
    r"越狱(模式|提示|prompt)",
    r"以开发者模式(回答|回复)",
    # ── 英文：指令覆盖类 ─────────────────────────────────
    r"ignore.{0,20}?(?:all|the|any)?(?:previous|above|prior|earlier|preceding).{0,20}?(?:instructions?|rules?|prompts?|commands?)",
    r"disregard.{0,20}?(?:all|the|any)?(?:previous|above|prior).{0,20}?(?:instructions?|rules?|prompts?)",
    r"forget.{0,20}?(?:all|the)?(?:previous|above|prior|earlier).{0,20}?(?:instructions?|rules?|prompts?|conversation|context)",
    # ── 英文：越狱类 ─────────────────────────────────────
    r"doanythingnow",
    r"jailbreak(?:mode|prompt|now)?",
    r"(?:^|[^a-z])dan(?:mode|prompt)",
    r"youarenow(?:a|an|the)?(?:unrestricted|unfiltered|uncensored|jailbroken)",
    # ── 英文：套取系统提示类 ──────────────────────────────
    r"reveal(?:your|the)?(?:system)?(?:prompt|instructions?)",
    r"(?:print|show|output|repeat|disclose|leak).{0,10}?(?:your|the)?systemprompt",
    # ── 间接注入特征 ─────────────────────────────────────
    r"[\[【]\s*系统指令\s*[\]】]",
    r"[\[【]\s*system\s*[\]】]",
    r"<\|im_start\|>system",
]


# 归一化时要去除的零宽 / 不可见控制字符（常被用来拆断关键词）
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
# 压缩视图时要去除的空白与常见插入分隔符
_SEPARATOR_RE = re.compile(r"[\s._\-*~`·•|/\\^]+")


def _normalize_text(text: str) -> str:
    """NFKC 归一化 + 转小写 + 去零宽字符：对抗全角同形字与大小写绕过。"""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return normalized.lower()


def _squeeze_text(normalized: str) -> str:
    """在归一化文本基础上再去掉空白与插入分隔符：对抗「加空格 / 加符号」绕过。"""
    return _SEPARATOR_RE.sub("", normalized)


class PromptInjectionGuardrail(InputGuardrail):
    """Prompt 注入攻击检测护栏，基于规则匹配常见注入模式。

    优先级高于普通内容安全（priority=20），专门防御提示注入攻击。
    默认开启中英双语规则检测，可扩展自定义模式。
    """

    name = "prompt_injection"
    description = "Prompt注入攻击检测，拦截尝试覆盖系统指令、越狱等恶意输入"
    priority = 20

    def __init__(
        self,
        extra_patterns: Optional[List[str]] = None,
        block_on_match: bool = True,
        **_guardrail_kwargs,
    ):
        super().__init__(**_guardrail_kwargs)
        patterns = DEFAULT_INJECTION_PATTERNS.copy()
        if extra_patterns:
            patterns.extend(extra_patterns)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.block_on_match = block_on_match

    def check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        # C2：非 str 的结构化输入（dict/list 等）递归展开为可扫描文本，不再
        # 无脑放行；str 输入经 to_scan_text 原样返回，与修复前行为逐字节一致。
        scan_text = to_scan_text(input_data)
        if not scan_text:
            return GuardrailResult.allow()

        # 两个视图：归一化串（保留词间空格）/ 压缩串（去掉全部空白与分隔符）
        normalized = _normalize_text(scan_text)
        squeezed = _squeeze_text(normalized)

        matched: List[str] = []
        for pattern in self._compiled:
            if pattern.search(normalized) or pattern.search(squeezed):
                if pattern.pattern not in matched:
                    matched.append(pattern.pattern)

        if matched:
            if self.block_on_match:
                return GuardrailResult.reject(
                    "检测到疑似提示注入攻击尝试",
                    metadata={"matched_patterns": matched},
                )
            return GuardrailResult.allow(
                metadata={"injection_suspected": True, "matches": matched},
            )

        return GuardrailResult.allow()

    def clone(self) -> "PromptInjectionGuardrail":
        """克隆时重新编译正则；基类字段全部透传，避免被重置为类默认（S3 修复）。"""
        patterns = [p.pattern for p in self._compiled]
        base_patterns = list(DEFAULT_INJECTION_PATTERNS)
        extra = [p for p in patterns if p not in base_patterns]
        return PromptInjectionGuardrail(
            extra_patterns=extra or None,
            block_on_match=self.block_on_match,
            name=self.name,
            description=self.description,
            priority=self.priority,
            enabled=self.enabled,
            fail_open=self.fail_open if getattr(self, "_fail_open_explicit", False) else None,
            tags=list(self.tags) if self.tags else None,
            metadata=dict(self.metadata) if self.metadata else None,
        )
