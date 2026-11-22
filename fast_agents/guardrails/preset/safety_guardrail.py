"""基础内容安全护栏 BasicSafetyGuardrail

基于关键词 / 正则规则的基础内容安全检查，作为默认最低安全基线。

设计要点（L7 修订，回应「误判率被低估」的评审意见）：

1. **输入侧与输出侧使用两套独立模式表**，不再共用一份：
   - ``DEFAULT_BLOCKED_PATTERNS``（输入侧）：识别用户「索取」违法 / 危险信息的请求；
   - ``DEFAULT_OUTPUT_BLOCKED_PATTERNS``（输出侧）：只识别模型「确实在产出」
     可执行危害信息的表述，例如「制造毒品的具体步骤」。
   这样模型正常解释「如何制造毒品是违法的」不会再被输出侧误拦——该句既不含
   「的步骤 / 方法 / 教程」这类产出特征，也不会命中输出侧模式。

2. **输出侧叠加「缓和语境豁免」**：命中模式的前后窗口内若出现
   「违法 / 禁止 / 抱歉 / 无法提供」等否定、警示、拒绝类词，则视为科普、拒答
   或风险提示，予以放行。例如「制造毒品的方法有多种，但都是违法的」不会拦截。

3. **误判率「极低」的表述已删除**：本护栏是高召回、低成本的确定性基线，不是
   高精度分类器。输入侧对「如何制造毒品」这类索取请求刻意保守（宁可误拦），
   输出侧则刻意收窄以免误伤正常解释；两侧都存在残余误判（漏拦与误拦），生产
   环境建议叠加模型判定类护栏，不要单独依赖本规则。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..input_guardrail import InputGuardrail
from ..output_guardrail import OutputGuardrail
from ..models import GuardrailResult


# ── 输入侧模式表：识别用户「索取」违法 / 危险信息的请求 ─────────────────────
# 语义是「请求获取」，因此对「如何制造毒品」这类索取句刻意保守，宁可误拦。
DEFAULT_BLOCKED_PATTERNS: List[Tuple[str, str]] = [
    # 违法犯罪相关
    (r"如何(制造|制作|合成)(毒品|炸弹|枪支|毒药|冰毒|海洛因)", "违法内容请求"),
    (r"(教|教我|告诉我)(一下)?(怎么|如何|怎样)?(制造|制作|合成)(毒品|炸弹|枪支|毒药)", "违法内容请求"),
    # 自伤相关
    (r"(自杀|自残|轻生)的?(方法|方式|教程|步骤)", "危险内容请求"),
    # 明显的恶意指令
    (r"写(一个|一段|个)?(病毒|木马|恶意软件|勒索软件|钓鱼程序)", "恶意软件制作请求"),
    (r"(黑客|入侵|攻击|ddos)(教程|方法|步骤)", "网络攻击教程请求"),
]


# ── 输出侧模式表：只识别「确实在产出」可执行危害信息的表述 ───────────────────
# 与输入侧的关键区别：必须带有「产出 / 教学 / 提供」的结构特征（步骤、方法、
# 代码、脚本等），单纯的提及、引用、反驳、科普都不应命中。
DEFAULT_OUTPUT_BLOCKED_PATTERNS: List[Tuple[str, str]] = [
    (r"(制造|制作|合成)(毒品|炸弹|枪支|毒药|冰毒|海洛因)的?(具体|详细|完整|全部|大致)?(步骤|方法|教程|流程|配方|做法)",
     "输出疑似提供危险物制作步骤"),
    (r"(自杀|自残|轻生)的?(具体|详细|完整)?(步骤|方法|教程|方式)", "输出疑似提供自伤方法"),
    (r"(写|生成|提供|给出|给你|输出|实现|附上)(一个|一段|个)?(病毒|木马|勒索软件|恶意软件|钓鱼程序)的?(代码|程序|脚本)?",
     "输出疑似提供恶意软件代码"),
    (r"(入侵|攻击|ddos)(目标|服务器|网站|系统).{0,20}?(步骤|方法|指令|脚本)", "输出疑似提供网络攻击步骤"),
]


# 输出侧「缓和语境」关键词：命中附近出现这些否定 / 警示 / 拒绝类词，
# 视为科普、拒答或风险提示，不拦截。刻意不收「不能 / 不可 / 危险」这类
# 既可能出现在拒答、也可能被恶意输出套用的弱词，降低被反向利用的空间。
_MITIGATION_KEYWORDS: Tuple[str, ...] = (
    "违法", "非法", "犯法", "犯罪", "禁止", "严禁",
    "请勿", "切勿", "不应", "不该", "不予", "拒绝",
    "抱歉", "无法提供", "不能提供", "不提供", "不建议", "不能给你", "无法给你",
    "不会提供", "不便提供", "恕不",
    "防范", "警惕", "举报", "法律", "合规",
)


def _has_mitigating_context(text: str, start: int, end: int, window: int) -> bool:
    """判断命中片段的前后窗口内是否存在缓和语境关键词。

    Args:
        text: 已小写化的待检文本。
        start / end: 命中片段在 text 中的起止下标。
        window: 前后各自扩展的字符数（命中片段本身也算在窗口内）。
    """
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    segment = text[lo:hi]
    return any(kw in segment for kw in _MITIGATION_KEYWORDS)


class BasicInputSafetyGuardrail(InputGuardrail):
    """输入阶段基础内容安全护栏，拦截用户输入中的明显违规内容。

    基于正则关键词匹配，速度极快，作为默认输入安全基线。
    高优先级（priority=10），最早执行。

    ⚠️ 定位为「高召回」的确定性基线：对「如何制造毒品」这类索取请求刻意保守
    （宁可误拦），难免存在误判，不可单独作为唯一安全手段。
    """

    name = "basic_input_safety"
    description = "基础输入内容安全护栏，拦截暴力、违法、恶意请求等明显违规内容"
    priority = 10

    def __init__(
        self,
        blocked_patterns: Optional[List[tuple]] = None,
        **_guardrail_kwargs,
    ):
        super().__init__(**_guardrail_kwargs)
        self.patterns = blocked_patterns or DEFAULT_BLOCKED_PATTERNS
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), reason)
            for pattern, reason in self.patterns
        ]

    def check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        if not isinstance(input_data, str):
            return GuardrailResult.allow()

        text = input_data.lower()
        for pattern, reason in self._compiled:
            if pattern.search(text):
                return GuardrailResult.reject(f"内容违规：{reason}")

        return GuardrailResult.allow()

    def clone(self) -> "BasicInputSafetyGuardrail":
        """克隆时重新编译正则，避免 deepcopy 正则对象的潜在兼容问题。

        基类字段（name/description/priority/enabled/fail_open/tags/metadata）
        必须显式传入构造器，否则会被重置为类默认值——这是 S3 bug 的修复。
        ``_fail_open_explicit`` 语义保留：原实例未显式设置时传 None，让
        GuardrailMiddleware 挂载阶段按统一姿态覆盖；显式设置时传原值保留标记。
        """
        return BasicInputSafetyGuardrail(
            blocked_patterns=self.patterns,
            name=self.name,
            description=self.description,
            priority=self.priority,
            enabled=self.enabled,
            fail_open=self.fail_open if getattr(self, "_fail_open_explicit", False) else None,
            tags=list(self.tags) if self.tags else None,
            metadata=dict(self.metadata) if self.metadata else None,
        )


class BasicOutputSafetyGuardrail(OutputGuardrail):
    """输出阶段基础内容安全护栏，拦截模型输出中的明显违规内容。

    与输入侧的最大区别：**不再复用输入侧的「索取请求」模式表**，改用输出侧
    专用的 ``DEFAULT_OUTPUT_BLOCKED_PATTERNS``，只拦截「确实在产出」可执行危害
    信息的表述（如「制造毒品的具体步骤」），因此模型正常解释
    「如何制造毒品是违法的」不会被误拦。

    在命中模式前后 ``mitigation_window`` 字符内出现否定 / 警示 / 拒绝类关键词时，
    判定为科普、拒答或风险提示，予以放行。

    优先级（priority=10），最早执行。
    """

    name = "basic_output_safety"
    description = "基础输出内容安全护栏，拦截输出中的违规、恶意、违法内容"
    priority = 10
    # 缓和语境检测窗口（字符数）
    mitigation_window = 30

    def __init__(
        self,
        blocked_patterns: Optional[List[tuple]] = None,
        mitigation_window: Optional[int] = None,
        allow_modify: Optional[bool] = None,
        **_guardrail_kwargs,
    ):
        super().__init__(allow_modify=allow_modify, **_guardrail_kwargs)
        # 默认使用输出侧专用模式表，不再复用输入侧 DEFAULT_BLOCKED_PATTERNS
        self.patterns = blocked_patterns or DEFAULT_OUTPUT_BLOCKED_PATTERNS
        self.mitigation_window = (
            type(self).mitigation_window if mitigation_window is None else mitigation_window
        )
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), reason)
            for pattern, reason in self.patterns
        ]

    def check(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        if not isinstance(input_data, str):
            return GuardrailResult.allow()

        text = input_data.lower()
        for pattern, reason in self._compiled:
            for match in pattern.finditer(text):
                # 命中处附近有缓和语境（违法 / 拒绝 / 抱歉…）→ 视为科普或拒答，放行
                if _has_mitigating_context(text, match.start(), match.end(), self.mitigation_window):
                    continue
                return GuardrailResult.reject(f"输出内容违规：{reason}")

        return GuardrailResult.allow()

    def clone(self) -> "BasicOutputSafetyGuardrail":
        """克隆时重新编译正则；基类字段全部透传，避免被重置为类默认（S3 修复）。"""
        return BasicOutputSafetyGuardrail(
            blocked_patterns=self.patterns,
            mitigation_window=self.mitigation_window,
            allow_modify=getattr(self, "allow_modify", False),
            name=self.name,
            description=self.description,
            priority=self.priority,
            enabled=self.enabled,
            fail_open=self.fail_open if getattr(self, "_fail_open_explicit", False) else None,
            tags=list(self.tags) if self.tags else None,
            metadata=dict(self.metadata) if self.metadata else None,
        )


# 向后兼容别名：BasicSafetyGuardrail 默认是输入+输出两个独立实例的组合
# 避免多继承MRO问题，check_output死参数废弃。
def BasicSafetyGuardrail(
    blocked_patterns: Optional[List[tuple]] = None,
    check_output: bool = True,
    output_patterns: Optional[List[tuple]] = None,
):
    """向后兼容工厂函数，返回输入+输出安全护栏实例列表。

    Args:
        blocked_patterns: 仅在**输入侧**覆盖模式表（不传用 DEFAULT_BLOCKED_PATTERNS）。
        check_output: 是否同时挂输出侧护栏。
        output_patterns: 仅在**输出侧**覆盖模式表（不传用 DEFAULT_OUTPUT_BLOCKED_PATTERNS）。
            提供该参数是为了让输入 / 输出两侧可以各配一套模式表——旧版把
            blocked_patterns 同时喂给两侧，正是「输出侧误拦正常解释」的根因。
    """
    instances = [BasicInputSafetyGuardrail(blocked_patterns=blocked_patterns)]
    if check_output:
        instances.append(BasicOutputSafetyGuardrail(blocked_patterns=output_patterns))
    return instances
