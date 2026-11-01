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
from typing import Iterable, List, Optional

from .schema import MemoryRecord

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


# ---------- 供 MemoryStoreBase.search() 默认实现复用的统一打分与排序 ----------

_PREF_RELEVANCE_FLOOR = 0.3  # 偏好类记忆的「角色权重保底」：哪怕词法零命中，
#                              也给一个基础相关度，保证用户偏好能常驻召回。


def score_record(query: str, record: MemoryRecord) -> float:
    """对单条 MemoryRecord 计算统一得分（v1.3.0 / B7）

    三类权重：
    - 词法相关度 × 0.6（必须命中至少一个查询词才入候选，偏好类例外——
      相关度低于 _PREF_RELEVANCE_FLOOR 时抬到保底值，等价于"角色权重"）；
    - 重要性 × 0.3；
    - 新鲜度（30 天半衰期）× 0.1；
    偏好类与非偏好类共用同一套公式，不再有两套独立权重。
    """
    relevance = lexical_score(query, record.content)
    if record.kind == "preference" and relevance < _PREF_RELEVANCE_FLOOR:
        relevance = _PREF_RELEVANCE_FLOOR
    recency = time_decay(record.updated_at)
    return relevance * 0.6 + record.importance * 0.3 + recency * 0.1


def rank_records(
    query: str,
    records: Iterable[MemoryRecord],
    limit: int = 5,
    *,
    include_preferences: bool = True,
) -> List[MemoryRecord]:
    """统一的「打分 → 过滤 → 排序 → 截断」流程

    供 MemoryStoreBase.search() 默认实现直接调用；
    所有后端共享同一套公式与 tie-breaker（-score → -importance → -updated_at），
    保证同一批记忆、同一 query 在任意后端上召回顺序一致。
    """
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit 必须是正整数")

    rec_list = list(records)

    # 空查询：按最近更新（→ 重要性兜底）截断
    if not query.strip():
        sorted_all = sorted(
            rec_list,
            key=lambda r: (r.updated_at, r.importance),
            reverse=True,
        )
        return sorted_all[:limit]

    scored: list[tuple[float, MemoryRecord]] = []
    for r in rec_list:
        is_pref = r.kind == "preference"
        rel = lexical_score(query, r.content)

        if is_pref:
            if not include_preferences:
                continue
            # 偏好类：保底相关度，始终入候选
            score = score_record(query, r)
            scored.append((score, r))
        else:
            # 普通记忆：至少命中一个查询词才入候选
            if rel > 0:
                score = score_record(query, r)
                scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], -x[1].importance, -x[1].updated_at.timestamp()))
    return [r for _, r in scored[:limit]]
