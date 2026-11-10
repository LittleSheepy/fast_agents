"""TraceEvaluator —— trace 数据离线评估器（路线图 K3）。

基于已落盘的 trace 数据（JsonlStore / SQLiteStore 均可）做离线批量聚合分析，
输出每条 trace 的细粒度指标 + 跨 trace 的聚合统计，用于评估 Agent 的：

- 成本（token 消耗 / LLM 调用次数）
- 效率（平均 step 数 / 端到端耗时 / tool 耗时占比）
- 质量（错误率 / 工具失败率 / 异常分类）
- 工具使用（被调工具分布 / 命中率 / 平均工具耗时）

设计原则：
- **纯离线**：只消费 TraceStoreBase.read() / iter_records() 返回的 SpanRecord，
  不与在线打点链路耦合，不影响业务主流程；
- **零新增默认依赖**：只用标准库（json / statistics / dataclasses / typing），
  不引入 pandas / numpy 等重依赖；
- **异常吞噬**：评估器内部任何计算异常都不抛出，缺字段时返回 None 并打标记，
  绝不因为 trace 数据异常而让评估脚本崩掉；
- **后端无关**：无论 JsonlStore 还是 SQLiteStore，只要实现 TraceStoreBase 即可评估，
  也可直接传入 TraceLogger 实例，自动取其 store；
- **可序列化**：所有结果都是 dataclass → dict → json 友好结构，可直接
  ``json.dumps(evaluator.evaluate().to_dict())``。
"""
from __future__ import annotations

import csv
import io
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Union

from .models import SpanKind, SpanRecord, SpanStatus


# ---------------------------------------------------------------------------
# 单条 Trace 指标
# ---------------------------------------------------------------------------
@dataclass
class TraceMetrics:
    """单条 Trace（一次 Agent run）的评估指标。

    所有字段均可为 None（数据不足时），绝不抛异常。
    """

    trace_id: str
    session_id: Optional[str] = None

    # ---------- 基本 ----------
    span_count: int = 0                 # 总 Span 数
    llm_calls: int = 0                  # LLM 调用次数
    tool_calls: int = 0                 # 工具调用次数
    decision_calls: int = 0             # 决策步次数
    agent_calls: int = 0                # 完整 Agent run 次数

    # ---------- 时间 ----------
    total_duration_ms: Optional[float] = None      # 端到端耗时（第一条 start ~ 最后一条 end）
    llm_total_ms: float = 0.0                      # LLM 总耗时
    tool_total_ms: float = 0.0                     # 工具总耗时
    decision_total_ms: float = 0.0                 # 决策总耗时
    tool_time_ratio: Optional[float] = None        # 工具耗时占比 = tool_total / total

    # ---------- Token（需要 LLM span metadata 带 usage 字段） ----------
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_llm_call: Optional[float] = None    # 平均每次 LLM 调用 token 数

    # ---------- 错误 ----------
    error_count: int = 0               # 总错误 Span 数（status=error）
    llm_error_count: int = 0           # LLM 调用错误数
    tool_error_count: int = 0          # 工具调用错误数
    tool_error_rate: Optional[float] = None   # tool_error / tool_calls（无 tool_calls 时 None）
    success: bool = True               # 整条 trace 是否成功（root Span status=success 且无未捕获异常）
    error_messages: List[str] = field(default_factory=list)   # 去重的 error 信息列表

    # ---------- 工具分布 ----------
    tool_distribution: Dict[str, int] = field(default_factory=dict)   # {tool_name: 调用次数}
    tool_durations: Dict[str, List[float]] = field(default_factory=dict)  # {tool_name: [duration_ms,...]}

    # ---------- 数据质量 ----------
    missing_duration_spans: int = 0    # duration_ms 缺失的 Span 数（打点覆盖度）
    spans_without_metadata: int = 0    # metadata 为空的 Span 数

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 把 tool_durations 的 list 转成均值/最大/最小三个标量（便于 JSON）
        td_summary: Dict[str, Dict[str, float]] = {}
        for name, durations in self.tool_durations.items():
            if not durations:
                continue
            td_summary[name] = {
                "count": len(durations),
                "avg_ms": round(statistics.fmean(durations), 2) if durations else 0.0,
                "max_ms": round(max(durations), 2),
                "min_ms": round(min(durations), 2),
                "total_ms": round(sum(durations), 2),
            }
        d["tool_durations_summary"] = td_summary
        d.pop("tool_durations", None)
        # 数值字段 round 到 2 位
        for k in ("total_duration_ms", "llm_total_ms", "tool_total_ms",
                  "decision_total_ms", "tool_time_ratio", "tokens_per_llm_call",
                  "tool_error_rate"):
            v = d.get(k)
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


# ---------------------------------------------------------------------------
# CSV 导出列定义（顺序稳定，便于 Excel / pandas 直接读取）
# ---------------------------------------------------------------------------
CSV_COLUMNS: List[str] = [
    "trace_id",
    "session_id",
    "success",
    "span_count",
    "llm_calls",
    "tool_calls",
    "decision_calls",
    "agent_calls",
    "total_duration_ms",
    "llm_total_ms",
    "tool_total_ms",
    "decision_total_ms",
    "tool_time_ratio",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tokens_per_llm_call",
    "error_count",
    "llm_error_count",
    "tool_error_count",
    "tool_error_rate",
    "missing_duration_spans",
    "spans_without_metadata",
    "tool_distribution",
    "error_messages",
]


# ---------------------------------------------------------------------------
# 聚合报告（多条 Trace 汇总）
# ---------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """跨多条 Trace 的聚合评估报告。"""

    total_traces: int = 0
    total_sessions: int = 0
    successful_traces: int = 0
    failed_traces: int = 0
    success_rate: Optional[float] = None

    # 平均值 / 标准差 / 最大 / 最小
    avg_spans_per_trace: Optional[float] = None
    avg_llm_calls: Optional[float] = None
    avg_tool_calls: Optional[float] = None
    avg_total_duration_ms: Optional[float] = None
    avg_total_tokens: Optional[float] = None

    std_total_duration_ms: Optional[float] = None
    p95_total_duration_ms: Optional[float] = None

    avg_tool_error_rate: Optional[float] = None
    avg_llm_error_rate: Optional[float] = None

    # 全局工具分布（跨所有 trace 聚合）
    tool_distribution: Dict[str, int] = field(default_factory=dict)
    tool_error_distribution: Dict[str, int] = field(default_factory=dict)

    # 模型分布（metadata.model）
    model_distribution: Dict[str, int] = field(default_factory=dict)

    # 单条 trace 明细
    traces: List[TraceMetrics] = field(default_factory=list)

    # 生成时间（ISO 格式，北京时间由调用方负责传入；这里只记录）
    generated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traces"] = [t.to_dict() for t in self.traces]
        for k in ("success_rate", "avg_spans_per_trace", "avg_llm_calls",
                  "avg_tool_calls", "avg_total_duration_ms", "avg_total_tokens",
                  "std_total_duration_ms", "p95_total_duration_ms",
                  "avg_tool_error_rate", "avg_llm_error_rate"):
            v = d.get(k)
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii, default=str)

    def to_text(self) -> str:
        """生成人类可读的纯文本摘要（对齐 examples/eval_demo.py 的打印输出）。"""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("Trace 评估报告（K3）")
        lines.append("=" * 60)
        lines.append(f"总 Trace 数        : {self.total_traces}")
        lines.append(f"总会话数           : {self.total_sessions}")
        lines.append(f"成功 / 失败        : {self.successful_traces} / {self.failed_traces}")
        if self.success_rate is not None:
            lines.append(f"整体成功率         : {self.success_rate*100:.1f}%")
        lines.append("-" * 60)
        lines.append("【单 Trace 平均】")
        if self.avg_spans_per_trace is not None:
            lines.append(f"  平均 Span 数     : {self.avg_spans_per_trace:.2f}")
        if self.avg_llm_calls is not None:
            lines.append(f"  平均 LLM 调用数  : {self.avg_llm_calls:.2f}")
        if self.avg_tool_calls is not None:
            lines.append(f"  平均工具调用数   : {self.avg_tool_calls:.2f}")
        if self.avg_total_duration_ms is not None:
            lines.append(f"  平均端到端耗时   : {self.avg_total_duration_ms:.1f} ms")
        if self.avg_total_tokens is not None:
            lines.append(f"  平均 Token 消耗  : {self.avg_total_tokens:.0f}")
        if self.p95_total_duration_ms is not None:
            lines.append(f"  P95 端到端耗时   : {self.p95_total_duration_ms:.1f} ms")
        if self.avg_tool_error_rate is not None:
            lines.append(f"  平均工具错误率   : {self.avg_tool_error_rate*100:.1f}%")
        lines.append("-" * 60)
        if self.tool_distribution:
            lines.append("【工具调用分布（跨全部 trace）】")
            total_tools = sum(self.tool_distribution.values()) or 1
            for name, cnt in sorted(self.tool_distribution.items(), key=lambda x: -x[1]):
                err = self.tool_error_distribution.get(name, 0)
                rate = err / cnt * 100 if cnt else 0.0
                lines.append(f"  {name:<25s} {cnt:>4d} 次  (错误 {err:>3d}, {rate:.1f}%)")
            lines.append(f"  {'(合计)':<25s} {total_tools:>4d} 次")
        if self.model_distribution:
            lines.append("-" * 60)
            lines.append("【模型分布】")
            for model, cnt in sorted(self.model_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"  {model:<30s} {cnt:>4d} 次")
        if self.traces:
            lines.append("-" * 60)
            lines.append("【单 Trace 明细】")
            for i, t in enumerate(self.traces, 1):
                dur = f"{t.total_duration_ms:.1f}ms" if t.total_duration_ms is not None else "N/A"
                tok = f"{t.total_tokens}tok" if t.total_tokens else "N/A"
                mark = "OK " if t.success else "ERR"
                lines.append(
                    f"  [{i:02d}] {mark} {t.trace_id[:16]}... "
                    f"spans={t.span_count:>3d} llm={t.llm_calls:>2d} "
                    f"tool={t.tool_calls:>2d} dur={dur:>10s} tok={tok:>8s}"
                )
                if t.error_messages:
                    for msg in t.error_messages[:3]:
                        lines.append(f"         ↳ {msg[:80]}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_csv(self, path: Optional[str] = None, *, encoding: str = "utf-8-sig") -> str:
        """把「每条 trace 一行」的明细导出为 CSV 文本（K3）。

        与 :meth:`to_json` / :meth:`to_text` 互补：JSON 适合程序消费、文本适合控制台
        直读，CSV 则适合**丢进 Excel / pandas 做二次透视**（如按 session、按天做成本日报）。
        每行对应 ``traces`` 列表里的一条 trace，列顺序固定为 :data:`CSV_COLUMNS`。

        Args:
            path: 可选。给定则同时把 CSV 写入该文件；不传只返回字符串、不落盘。
            encoding: 写文件时使用的编码（仅 ``path`` 非空生效）。默认 ``utf-8-sig``
                （带 BOM），这样 Windows 下 Excel 双击打开中文不乱码。

        Returns:
            CSV 文本（不含 BOM），无论是否写文件都会返回；无明细时只含表头行。
            内部任何异常都被吞掉——写文件失败不影响返回值，也不会向上抛。
        """
        buf = io.StringIO()
        try:
            writer = csv.writer(buf)
            writer.writerow(CSV_COLUMNS)
            for t in self.traces:
                try:
                    row = [_csv_cell(getattr(t, col, None)) for col in CSV_COLUMNS]
                except Exception:
                    row = ["" for _ in CSV_COLUMNS]
                writer.writerow(row)
        except Exception:
            pass
        text = buf.getvalue()
        if path:
            try:
                with open(path, "w", encoding=encoding, newline="") as f:
                    f.write(text)
            except Exception:
                pass
        return text


# ---------------------------------------------------------------------------
# 评估器主类
# ---------------------------------------------------------------------------
class TraceEvaluator:
    """Trace 数据离线评估器。

    用法::

        # 方式 1：从 TraceLogger 评估
        evaluator = TraceEvaluator(tracer)
        report = evaluator.evaluate()
        print(report.to_text())

        # 方式 2：从 Store 路径评估
        evaluator = TraceEvaluator.from_path("logs/agent_run.db")
        report = evaluator.evaluate(session_id="sess_xxx")

        # 方式 3：导出 JSON
        with open("eval_report.json", "w", encoding="utf-8") as f:
            f.write(report.to_json())

    Args:
        source: TraceLogger 实例 / TraceStoreBase 实例 / SpanRecord 可迭代对象
                （三种之一）。传入 TraceLogger 时自动取其 ``.store``；
                传入 Store 时直接 ``store.read()``；传入可迭代对象时直接消费。
    """

    def __init__(self, source: Any):
        self._spans: Optional[List[SpanRecord]] = None
        self._source_desc: str = "iterable"
        try:
            # 优先识别 TraceLogger（有 .store / .read_spans）
            if hasattr(source, "store") and hasattr(source, "read_spans"):
                self._spans = list(source.store.iter_records() if hasattr(source.store, "iter_records") else source.store.read())
                self._source_desc = f"TraceLogger(store={type(source.store).__name__})"
            elif hasattr(source, "read") and callable(source.read):
                # TraceStoreBase
                self._spans = list(source.iter_records() if hasattr(source, "iter_records") else source.read())
                self._source_desc = f"Store({type(source).__name__})"
            elif hasattr(source, "__iter__"):
                # 直接给 SpanRecord 列表 / 生成器
                self._spans = [s for s in source if isinstance(s, SpanRecord)]
                self._source_desc = "iterable[SpanRecord]"
            else:
                self._spans = []
        except Exception:
            # 读取失败不抛异常，返回空报告
            self._spans = []

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        backend: Optional[str] = None,
    ) -> "TraceEvaluator":
        """从文件路径构造评估器（自动识别后端：.db/.sqlite/.sqlite3 → SQLite，其它 → Jsonl）。

        Args:
            path: trace 落盘文件路径；支持 ``:memory:``（返回空评估器）。
            backend: 显式指定 ``"jsonl"`` 或 ``"sqlite"``；默认按后缀自动识别。
        """
        from .jsonl_store import MEMORY_PATH, JsonlStore
        from .sqlite_store import SQLiteStore

        if path == MEMORY_PATH:
            return cls([])
        store: Any
        try:
            use_sqlite = False
            if backend is not None:
                use_sqlite = backend.lower() == "sqlite"
            else:
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                use_sqlite = ext in {"db", "sqlite", "sqlite3"}
            if use_sqlite:
                store = SQLiteStore(path)
            else:
                store = JsonlStore(path, append=True)
            return cls(store)
        except Exception:
            # 文件不存在 / 损坏等，返回空评估器
            return cls([])

    # ---------- 核心：单 trace 指标计算 ----------
    def _compute_trace_metrics(self, trace_id: str, spans: List[SpanRecord]) -> TraceMetrics:
        """计算单条 trace 的指标；任何异常都不抛出。"""
        m = TraceMetrics(trace_id=trace_id)
        if not spans:
            return m

        try:
            m.session_id = spans[0].session_id
            m.span_count = len(spans)

            # 找 root Span（parent_id 为 None）以判定整条 trace 成功与否
            root_spans = [s for s in spans if s.parent_id is None]
            if root_spans:
                # 取最先创建的 root 作为代表
                root = min(root_spans, key=lambda s: getattr(s, "_create_seq", 0) or s.start_time)
                m.success = (root.status == SpanStatus.SUCCESS and not root.error)
            else:
                # 没有 root：只要没有 error Span 就算成功（防御性兜底）
                m.success = all(s.status != SpanStatus.ERROR for s in spans)

            min_start: Optional[float] = None
            max_end: Optional[float] = None

            for s in spans:
                # 时间边界
                try:
                    st = s.start_time
                    et = s.end_time if s.end_time is not None else st
                    if min_start is None or st < min_start:
                        min_start = st
                    if max_end is None or et > max_end:
                        max_end = et
                except Exception:
                    pass

                # 按 kind 计数与耗时累加
                try:
                    kind_str = s.kind.value if isinstance(s.kind, SpanKind) else str(s.kind)
                    dur = s.duration_ms if s.duration_ms is not None else 0.0
                    if kind_str == SpanKind.LLM.value:
                        m.llm_calls += 1
                        m.llm_total_ms += dur
                    elif kind_str == SpanKind.TOOL.value:
                        m.tool_calls += 1
                        m.tool_total_ms += dur
                        # 工具名：优先 metadata.tool_name，其次从 name 里拆 "tool.<name>"
                        tname = None
                        if isinstance(s.metadata, dict):
                            tname = s.metadata.get("tool_name")
                        if not tname and isinstance(s.name, str) and s.name.startswith("tool."):
                            tname = s.name[len("tool."):]
                        if not tname:
                            tname = s.name or "unknown"
                        m.tool_distribution[tname] = m.tool_distribution.get(tname, 0) + 1
                        m.tool_durations.setdefault(tname, []).append(float(dur or 0.0))
                        if s.status == SpanStatus.ERROR:
                            m.tool_error_count += 1
                        # 注：per-trace 的「每工具错误数」不在这里维护，
                        # 在 evaluate() 聚合阶段统一从 spans 重扫，避免在 TraceMetrics 里再开一个字段
                    elif kind_str == SpanKind.DECISION.value:
                        m.decision_calls += 1
                        m.decision_total_ms += dur
                    elif kind_str == SpanKind.AGENT.value:
                        m.agent_calls += 1
                except Exception:
                    pass

                # Token 累加：优先 metadata.usage.{prompt,completion,total}_tokens
                try:
                    meta = s.metadata if isinstance(s.metadata, dict) else {}
                    usage = meta.get("usage") if isinstance(meta, dict) else None
                    if isinstance(usage, dict):
                        pt = _safe_int(usage.get("prompt_tokens"))
                        ct = _safe_int(usage.get("completion_tokens"))
                        tt = _safe_int(usage.get("total_tokens"))
                        m.prompt_tokens += pt
                        m.completion_tokens += ct
                        # total_tokens 字段若缺失，用 prompt+completion 兜底
                        m.total_tokens += tt if tt else (pt + ct)
                    # 模型分布（trace 级别只记第一次出现，聚合阶段再汇总）
                    model_name = meta.get("model") if isinstance(meta, dict) else None
                    if model_name:
                        m._models_seen = getattr(m, "_models_seen", set()) | {str(model_name)}
                except Exception:
                    pass

                # 错误
                try:
                    if s.duration_ms is None:
                        m.missing_duration_spans += 1
                    if not s.metadata:
                        m.spans_without_metadata += 1
                    if s.status == SpanStatus.ERROR:
                        m.error_count += 1
                        kind_str = s.kind.value if isinstance(s.kind, SpanKind) else str(s.kind)
                        if kind_str == SpanKind.LLM.value:
                            m.llm_error_count += 1
                        if s.error:
                            err_str = str(s.error)
                            if err_str not in m.error_messages:
                                m.error_messages.append(err_str)
                        if root_spans is None or s not in root_spans:
                            # 非 root span 出错也会导致整条 trace 的 success 标记为 False
                            m.success = False
                except Exception:
                    pass

            # 端到端耗时
            if min_start is not None and max_end is not None and max_end >= min_start:
                m.total_duration_ms = (max_end - min_start) * 1000.0
            # 工具耗时占比
            if m.total_duration_ms and m.total_duration_ms > 0:
                m.tool_time_ratio = m.tool_total_ms / m.total_duration_ms
            # 平均每次 LLM 调用 token
            if m.llm_calls > 0 and m.total_tokens > 0:
                m.tokens_per_llm_call = m.total_tokens / m.llm_calls
            # 工具错误率
            if m.tool_calls > 0:
                m.tool_error_rate = m.tool_error_count / m.tool_calls
        except Exception:
            # 计算过程中任何异常都不抛出，保留已算出的字段
            pass

        return m

    # ---------- 对外主入口 ----------
    def evaluate(
        self,
        *,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: Optional[int] = None,
        top_n_traces: Optional[int] = None,
    ) -> EvaluationReport:
        """执行评估并返回 :class:`EvaluationReport`。

        Args:
            session_id: 可选，按会话过滤。
            trace_id:   可选，只评估单条 trace（优先级最高）。
            limit:      可选，从 store 最多读取多少条 Span（避免大库一次性拉爆内存）。
            top_n_traces: 可选，只评估「最近 N 条 trace」（按 root Span 时间倒序），
                          不传则评估全部 trace。
        """
        from datetime import datetime, timezone, timedelta
        CST = timezone(timedelta(hours=8))

        report = EvaluationReport()
        try:
            report.generated_at = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")
        except Exception:
            report.generated_at = None

        spans = self._spans or []
        # 过滤
        try:
            if session_id is not None:
                spans = [s for s in spans if s.session_id == session_id]
            if trace_id is not None:
                spans = [s for s in spans if s.trace_id == trace_id]
            if limit is not None and limit > 0:
                # 按创建序号/时间倒序取最近 limit 条
                spans = sorted(
                    spans,
                    key=lambda s: (getattr(s, "_create_seq", 0), s.start_time),
                    reverse=True,
                )[:limit]
        except Exception:
            pass

        if not spans:
            return report

        # 按 trace_id 分组
        groups: Dict[str, List[SpanRecord]] = {}
        for s in spans:
            try:
                groups.setdefault(s.trace_id, []).append(s)
            except Exception:
                continue

        # 若指定 top_n_traces，按"最近 root span"的时间倒序取前 N 条 trace
        if top_n_traces is not None and top_n_traces > 0 and trace_id is None:
            def _trace_latest_time(tid: str) -> float:
                t_spans = groups[tid]
                roots = [x for x in t_spans if x.parent_id is None]
                pool = roots if roots else t_spans
                return max((x.end_time or x.start_time or 0.0) for x in pool)
            sorted_tids = sorted(groups.keys(), key=_trace_latest_time, reverse=True)
            selected_tids = sorted_tids[:top_n_traces]
            groups = {tid: groups[tid] for tid in selected_tids}

        # 计算每条 trace 指标
        trace_metrics: List[TraceMetrics] = []
        sessions = set()
        for tid, t_spans in groups.items():
            # 组内按创建序号排序（稳定顺序，便于找 root）
            try:
                t_spans.sort(key=lambda s: (getattr(s, "_create_seq", 0), s.start_time, s.span_id))
            except Exception:
                pass
            tm = self._compute_trace_metrics(tid, t_spans)
            trace_metrics.append(tm)
            if tm.session_id:
                sessions.add(tm.session_id)

        # 按 trace 的最新 span 时间倒序排列（最新的在前）
        def _trace_sort_key(tm: TraceMetrics):
            # 取该 trace 所有 span 的最大 end_time
            t_spans = groups.get(tm.trace_id, [])
            latest = 0.0
            for s in t_spans:
                try:
                    et = s.end_time or s.start_time or 0.0
                    if et > latest:
                        latest = et
                except Exception:
                    continue
            return latest
        trace_metrics.sort(key=_trace_sort_key, reverse=True)

        report.traces = trace_metrics
        report.total_traces = len(trace_metrics)
        report.total_sessions = len(sessions)
        report.successful_traces = sum(1 for t in trace_metrics if t.success)
        report.failed_traces = report.total_traces - report.successful_traces
        if report.total_traces > 0:
            report.success_rate = report.successful_traces / report.total_traces

        # ---------- 聚合数值 ----------
        try:
            if trace_metrics:
                spans_counts = [t.span_count for t in trace_metrics]
                llm_counts = [t.llm_calls for t in trace_metrics]
                tool_counts = [t.tool_calls for t in trace_metrics]
                report.avg_spans_per_trace = statistics.fmean(spans_counts)
                report.avg_llm_calls = statistics.fmean(llm_counts)
                report.avg_tool_calls = statistics.fmean(tool_counts)

                durations = [t.total_duration_ms for t in trace_metrics if t.total_duration_ms is not None]
                if durations:
                    report.avg_total_duration_ms = statistics.fmean(durations)
                    if len(durations) > 1:
                        report.std_total_duration_ms = statistics.pstdev(durations)
                    report.p95_total_duration_ms = _percentile(durations, 95)

                tokens = [t.total_tokens for t in trace_metrics if t.total_tokens > 0]
                if tokens:
                    report.avg_total_tokens = statistics.fmean(tokens)

                # 工具 / LLM 错误率的聚合（跨 trace 平均）
                ter = [t.tool_error_rate for t in trace_metrics if t.tool_error_rate is not None]
                if ter:
                    report.avg_tool_error_rate = statistics.fmean(ter)
                ler_denom = [t.llm_calls for t in trace_metrics if t.llm_calls > 0]
                ler_num = [t.llm_error_count for t in trace_metrics if t.llm_calls > 0]
                if ler_denom:
                    report.avg_llm_error_rate = sum(ler_num) / sum(ler_denom)

                # 工具分布聚合
                td: Dict[str, int] = {}
                ted: Dict[str, int] = {}
                md: Dict[str, int] = {}
                for t in trace_metrics:
                    for name, cnt in t.tool_distribution.items():
                        td[name] = td.get(name, 0) + cnt
                    # tool_error_count 我们在 _compute_trace_metrics 里没存 per-tool 错误，
                    # 这里需要从原始 spans 重扫一遍（Spans 就在 groups 里）
                    for s in groups.get(t.trace_id, []):
                        try:
                            kind_str = s.kind.value if isinstance(s.kind, SpanKind) else str(s.kind)
                            if kind_str == SpanKind.TOOL.value and s.status == SpanStatus.ERROR:
                                tname = None
                                if isinstance(s.metadata, dict):
                                    tname = s.metadata.get("tool_name")
                                if not tname and isinstance(s.name, str) and s.name.startswith("tool."):
                                    tname = s.name[len("tool."):]
                                if tname:
                                    ted[tname] = ted.get(tname, 0) + 1
                            # 模型分布
                            if kind_str == SpanKind.LLM.value and isinstance(s.metadata, dict):
                                model_name = s.metadata.get("model")
                                if model_name:
                                    md[str(model_name)] = md.get(str(model_name), 0) + 1
                        except Exception:
                            pass
                report.tool_distribution = td
                report.tool_error_distribution = ted
                report.model_distribution = md
        except Exception:
            pass

        return report


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------
def _safe_int(v: Any) -> int:
    """把任意值安全转 int；失败返回 0。"""
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0


def _percentile(values: List[float], pct: float) -> float:
    """计算百分位数（线性插值，纯标准库实现，无需 numpy）。

    Args:
        values: 数值列表。
        pct: 0-100 的百分位（如 95 表示 P95）。
    """
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    k = (len(vs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return float(vs[f])
    d0 = vs[f] * (c - k)
    d1 = vs[c] * (k - f)
    return float(d0 + d1)


def _csv_scalar(v: Any) -> str:
    """把单个标量转成 CSV 单元格字符串（内部换行折成空格，绝不抛异常）。"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        s = str(v)
    except Exception:
        return ""
    return s.replace("\r", " ").replace("\n", " ")


def _csv_cell(v: Any) -> str:
    """把任意指标值转成 CSV 单元格字符串（dict / list 折叠为可读文本，绝不抛异常）。

    - ``bool`` → ``"true"`` / ``"false"``（Excel / pandas 可识别为布尔）；
    - ``dict`` → ``"key=val;key2=val2"``，按值降序（如工具分布从高到低）；
    - ``list`` / ``tuple`` / ``set`` → ``" | "`` 连接；
    - ``None`` → 空串。
    """
    try:
        if isinstance(v, dict):
            def _sort_key(kv: Any) -> Any:
                val = kv[1]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return -val
                return 0

            items = sorted(v.items(), key=_sort_key)
            return ";".join(f"{_csv_scalar(k)}={_csv_scalar(val)}" for k, val in items)
        if isinstance(v, (list, tuple, set)):
            return " | ".join(_csv_scalar(x) for x in v)
        return _csv_scalar(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return ""


__all__ = [
    "TraceMetrics",
    "EvaluationReport",
    "TraceEvaluator",
]
