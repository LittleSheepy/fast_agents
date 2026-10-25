"""记忆相关性打分工具

为什么不直接上向量检索？
- 你现在的框架还依赖 OpenAI 原生 API，没有 embedding 链路；
- 能不能先跑通记忆闭环，比检索精度更重要。

所以这里先用「词法重叠 + 重要性 + 时间衰减」这套无依赖打分，
接口留好，后续换成向量检索时只替换 search() 的实现即可。
（HelloAgents 的做法也是这样：先用 SQLite 打底，语义检索作为可选增强层。）
"""

import re
import math
from datetime import datetime
from typing import List

# 中文按字也算词，所以用「英文词 + 单字」混合切分
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")

# 常见中文停用字，避免"的了吗我你"这类字制造虚假相关
_STOP = set("的了吗呢吧啊哦嗯我你他她它是在有和就都很也把被这那之与及或")


def lexical_terms(text: str) -> List[str]:
    """把文本切成词项列表"""
    if not text:
        return []
    terms = []
    for token in _WORD_RE.findall(text.lower()):
        if len(token) == 1 and token in _STOP:
            continue
        terms.append(token)
    return terms


def lexical_score(query: str, content: str) -> float:
    """词法相关度：命中词项占查询词项的比例（0~1）"""
    q_terms = set(lexical_terms(query))
    if not q_terms:
        return 0.0
    c_terms = set(lexical_terms(content))
    return len(q_terms & c_terms) / len(q_terms)


def time_decay(updated_at: datetime, half_life_days: float = 30.0) -> float:
    """时间衰减因子：越旧的记忆权重越低，半衰期默认 30 天"""
    if half_life_days <= 0:
        return 1.0
    delta_days = (datetime.now() - updated_at).total_seconds() / 86400.0
    if delta_days <= 0:
        return 1.0
    return math.pow(0.5, delta_days / half_life_days)


def combined_score(query: str, content: str, importance: float, updated_at: datetime,
                   *, relevance_weight: float = 0.6,
                   importance_weight: float = 0.3,
                   recency_weight: float = 0.1) -> float:
    """综合得分 = 相关度 * w1 + 重要性 * w2 + 时间新鲜度 * w3

    三个权重之和应为 1.0。
    """
    total = relevance_weight + importance_weight + recency_weight
    if abs(total - 1.0) > 1e-6:
        raise ValueError("三个权重之和必须等于 1.0")
    relevance = lexical_score(query, content)
    recency = time_decay(updated_at)
    return (
        relevance * relevance_weight
        + importance * importance_weight
        + recency * recency_weight
    )
