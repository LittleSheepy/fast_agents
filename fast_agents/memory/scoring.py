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

from fast_agents.utils.time_utils import to_cst, cstnow
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
    delta_days = (cstnow() - to_cst(updated_at)).total_seconds() / 86400.0
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


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度（纯 Python 实现，不依赖 numpy）

    用于语义检索通道的向量打分。返回值 0~1（负值截断为 0）。
    长度不一致时按较短长度截断，避免维度错误把流程打断。
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0 or nb <= 0:
        return 0.0
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    return max(0.0, min(1.0, sim))


# ---- 记忆合并相似度（B5）----
# 合并关心的是「两条记忆是不是同一件事」，与检索相关度不是一回事，故单独一组权重。
# 取值与 SemanticMemoryStore 的 hybrid_weight 口径一致（语义 0.7 / 词法 0.3），不新造参数。
MERGE_SEMANTIC_WEIGHT = 0.7
MERGE_LEXICAL_WEIGHT = 0.3


def merge_similarity(lexical: float, semantic: Optional[float] = None) -> float:
    """合并用相似度：语义 × 0.7 + 词法 × 0.3

    与 score_record 里「相关度」最关键的差别是**语义缺失时怎么处理**：
    - semantic 为 None（后端没有向量能力，或批量 embedding 失败）时，
      **直接退化为纯词法**，绝不能把语义当成 0；
    - 否则两条内容完全相同的记忆也会因拿不到向量而只剩 0.3 倍词法分，
      永远够不到合并阈值，合并功能会静默失效。

    入参均应为 0~1，返回值截断到 0~1。
    """
    if semantic is None:
        return max(0.0, min(1.0, lexical))
    sim = MERGE_SEMANTIC_WEIGHT * semantic + MERGE_LEXICAL_WEIGHT * lexical
    return max(0.0, min(1.0, sim))


def score_record(query: str, record: "MemoryRecord", semantic: Optional[float] = None, *, hybrid_weight: float = 0.7) -> float:
    """单条记忆MemoryRecord计算统一得分

    三类权重：
    - 词法相关度 × 0.6（必须命中至少一个查询词才入候选，偏好类例外——
      相关度低于 _PREF_RELEVANCE_FLOOR 时抬到保底值，等价于"角色权重"）；
    - 重要性 × 0.3；
    - 新鲜度（30 天半衰期）× 0.1；
    偏好类与非偏好类共用同一套公式，不再有两套独立权重。
    - semantic 为 None 时：退化为纯词法打分（兼容 v1.3.0 行为）；
    - semantic 传入时：relevance = hybrid_weight * semantic + (1 - hybrid_weight) * lexical，
      默认 hybrid_weight=0.7，即偏语义（用户确认）；
    - 偏好类仍走「relevance 保底 _PREF_RELEVANCE_FLOOR」的角色权重逻辑。
    """
    from .schema import MemoryRecord  # 局部导入避免循环
    _PREF_RELEVANCE_FLOOR = 0.3
    lex = lexical_score(query, record.content)
    if semantic is None:
        relevance = lex
    else:
        relevance = hybrid_weight * semantic + (1.0 - hybrid_weight) * lex
    if record.kind == "preference" and relevance < _PREF_RELEVANCE_FLOOR:
        relevance = _PREF_RELEVANCE_FLOOR
    recency = time_decay(record.updated_at)
    return relevance * 0.6 + record.importance * 0.3 + recency * 0.1


def rank_records(query, records, limit, *, include_preferences=True,
                 semantic_getter=None, hybrid_weight: float = 0.7,
                 semantic_floor: float = 0.3) -> List["MemoryRecord"]:
    """统一打分 / 排序 / 截断流水线（B2/v1.4.0 支持语义通道）

    semantic_getter: 可选回调 `(MemoryRecord) -> Optional[float]`，
        返回该条记忆的语义分；返回 None 表示该条无向量（降级为纯词法）。
    semantic_floor: 非偏好类记忆在「词法=0 且语义分 < 该阈值」时被过滤掉，
        既没词法命中也没语义相近的不进候选。
    其余参数与 v1.3.0 一致。
    """
    from .schema import MemoryRecord
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not isinstance(limit, int) or limit <= 0:
        raise TypeError("limit 必须是正整数")

    rec_list = list(records)

    # 空查询：按最近更新（→ 重要性兜底）截断
    if not query.strip():
        sorted_all = sorted(
            rec_list,
            key=lambda r: (to_cst(r.updated_at), r.importance),
            reverse=True,
        )
        return sorted_all[:limit]

    scored: list[tuple[float, MemoryRecord]] = []
    for r in rec_list:
        if not include_preferences and r.kind == "preference":
            continue

        sem = semantic_getter(r) if semantic_getter is not None else None
        lex = lexical_score(query, r.content)
        if r.kind != "preference":
            eff_sem = sem if sem is not None else 0.0
            if lex <= 0 and eff_sem < semantic_floor:
                continue  # 词法零命中 + 语义不够近，丢掉
        score = score_record(query, r, semantic=sem, hybrid_weight=hybrid_weight)
        scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], -x[1].importance, -to_cst(x[1].updated_at).timestamp()))
    return [r for _, r in scored[:limit]]
