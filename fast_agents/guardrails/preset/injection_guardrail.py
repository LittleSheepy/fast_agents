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

4. **逐句匹配（B4 RC1 修订，回应「注入规则误拦正常文本」的审计意见）**：带间隙的英文
   模式（如 `ignore.{0,20}?previous.{0,20}?instructions`）原先在整段文本上匹配，会把
   「Don\'t forget the previous chapters. Instructions for installation are below.」这种
   分属两句的无关子句粘连成误报。现按句子边界切分再逐句匹配：
   英文 `.` `!` `?` **后跟空白才切**（`jail.break` 不切，`chapters. Instructions` 切），
   中文 `。` `！` `？` 直接切；跨句误报被拆开；因逐句是整段的子集，句内注入与加空格 / 加符号 / 全角绕过均不受影响。

5. **词融合防护（B4 RC2 修订）**：`jailbreak(?:mode|prompt|now)?` 等纯单词紧凑模式
   原本在完整压缩视图（去空格 + 去分隔符）里匹配，导致普通英文短语
   `jail  break the habit` 被误融合成 `jailbreak` 而误拦。现对这些模式改用「非空格
   压缩」视图（仅去 `.` `-` `_` `*` 等非空白分隔符、保留空格），因此：
   `jail-break` / `jail.break` 仍能匹配；`jail  break` 不再误触。
   ⚠️ 已知取舍：`do anything now`（纯空格分隔）仍由完整压缩视图捕获，因为现有测试
   L7 断言该短语单独出现就是 DAN 攻击签名，不可放宽。
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
# 压缩视图时要去除的空白与常见插入分隔符（完整压缩，用于跨词间隙模式）
_SEPARATOR_RE = re.compile(r"[\s._\-*~`·•|/\\^]+")
# B4 RC2 修复：非空格压缩视图——仅去非空白分隔符（`.` `-` `_` `*` 等），保留词间空格。
# 用途：专供 jailbreak 类紧凑单词模式使用，避免 `jail break the habit`（普通英文短语）
# 被全压缩视图误融合为 `jailbreak` 而误拦；`jail-break`、`jail.break` 等真实绕过仍能被捕获。
_NO_SPACE_SEPARATOR_RE = re.compile(r"[._\-*~`·•|/\\^]+")
# B4 RC1 修复：句子切分边界。带间隙的英文模式（如 `ignore.{0,20}?previous.{0,20}?instructions`）
# 在整段文本上会把「……chapters. Instructions for installation……」这种相邻但无关的两个
# 子句粘连成注入特征，造成对正常叙述 / 教程文本的误拦。先按**句末标点**切分、再逐句匹配。
# 切分规则：
# - 英文 `.` `!` `?` 后必须跟空白才切（`chapters. Instructions` 切，`jail.break` 不切）
# - 中文 `。` `！` `？` 直接切（中文词内不会出现这些字符）
# ❗ **不切裸换行**：「ignore all\nprevious instructions」靠完整压缩视图处理，不能靠切分漏放。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])")


def _normalize_text(text: str) -> str:
    """NFKC 归一化 + 转小写 + 去零宽字符：对抗全角同形字与大小写绕过。"""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return normalized.lower()


def _squeeze_text(normalized: str) -> str:
    """在归一化文本基础上再去掉空白与插入分隔符：对抗「加空格 / 加符号」绕过。"""
    return _SEPARATOR_RE.sub("", normalized)


def _squeeze_text_nonspace(normalized: str) -> str:
    """非空格压缩：仅去非空白分隔符（`.` `-` `_` `*` 等），保留词间空格。
    B4 RC2 修复：防止 `jail break`（含空格英文词组）被误融合为 `jailbreak`。
    真实绕过 `jail-break` / `jail.break` 经此视图仍会变成 `jailbreak` 而被捕获。"""
    return _NO_SPACE_SEPARATOR_RE.sub("", normalized)


def _split_sentences(text: str) -> List[str]:
    """按句末终止符切分为子句，用于限制带间隙模式跨句粘连（B4）。

    切分规则：
    - 英文 `.` `!` `?` 后跟空白才切（`chapters. Instructions` 切，`jail.break` 不切）
    - 中文 `。` `！` `？` 直接切
    - **不切裸换行**：换行绕过靠完整压缩视图处理

    若切分后无有效子段则退回原文，绝不因切分而漏扫。
    """
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts or [text]


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

        # B4 RC1 修复：按句末标点切分后逐句匹配，避免带间隙模式跨句把无关子句粘成误报。
        # 逐句匹配是整段匹配的子集（只会减少命中），因此既能拆掉跨句误报，
        # 又不会把句内注入（含 i g n o r e 这类加空格绕过）漏放。
        segments = _split_sentences(normalized)

        # B4 RC2 修复：jailbreak 类纯单词紧凑模式改用「非空格压缩」视图，
        # 防止 jail break（普通英文短语）被压缩成 jailbreak 而误拦；
        # jail-break / jail.break 等真实绕过仍能命中。
        # doanythingnow 保持完整压缩（现有测试 L7 断言 `do anything now` 应被拦截，R1 不变更）。
        _WORD_FUSION_PATTERNS = {
            r"jailbreak(?:mode|prompt|now)?",
        }

        matched: List[str] = []
        for pattern in self._compiled:
            if pattern.pattern in _WORD_FUSION_PATTERNS:
                # RC2：只匹配归一化视图 + 非空格压缩视图，不用完整压缩
                hit = any(
                    pattern.search(seg) or pattern.search(_squeeze_text_nonspace(seg))
                    for seg in segments
                )
            else:
                hit = any(
                    pattern.search(seg) or pattern.search(_squeeze_text(seg))
                    for seg in segments
                )
            if hit and pattern.pattern not in matched:
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
