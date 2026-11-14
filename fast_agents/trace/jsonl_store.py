"""JsonlStore —— 默认 JSONL 落盘存储后端。

一行一个 JSON 对象（jsonl 格式），零依赖（仅标准库 json + pathlib）；
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .models import SpanKind, SpanRecord, _now_ts, span_record_to_dict, dict_to_span_record
from .store_base import TraceStoreBase
from ..utils.json_utils import json_safe_serialize, json_safe_load




def _match(d: dict, session_id, trace_id, kind) -> bool:
    """单条记录是否匹配过滤条件。"""
    if session_id is not None and d.get("session_id") != session_id:
        return False
    if trace_id is not None and d.get("trace_id") != trace_id:
        return False
    if kind is not None and d.get("kind") != kind:
        return False
    return True


class JsonlStore(TraceStoreBase):
    """JSONL 一行一条 Span 的存储后端。

    Args:
        path: jsonl 文件路径（必填，仅支持文件落盘，内存模式请用 InMemoryTraceStore）。
        append: True 表示追加到已有文件（默认）；False 表示清空重写。
        encoding: 文件编码，默认 ``utf-8``。

    设计：
    - 文件模式：写入时直接落盘、**不在内存保留全量镜像**（避免随文件线性涨内存）；
      读取时按需流式扫描文件，命中 ``limit`` 即返回，不预加载。
    """

    def __init__(self, path: str, *, append: bool = True, encoding: str = "utf-8"):
        self.path = path
        self.append = append
        self.encoding = encoding
        self._lock = threading.Lock()
        self._fp = None
        self._closed = False
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        # buffering=1 行缓冲；newline="" 让 json 写入跨平台一致，由我们控制 \n
        self._fp = open(p, mode=mode, encoding=encoding, buffering=1, newline="")
        # 文件模式下不再预读历史到内存：读时按需流式扫描，避免启动即占满内存。

    # ---------- TraceStoreBase 接口 ----------
    def write(self, span: SpanRecord) -> None:
        """写入一条 Span。任何异常都会被吞掉（打点不能阻塞业务）。"""
        if self._closed:
            return
        try:
            row = span_record_to_dict(span)
            line = json.dumps(row, ensure_ascii=False, default=str)
            with self._lock:
                if self._fp is not None:
                    self._fp.write(line + "\n")
        except Exception:
            # 打点通道任何异常都不向上抛，避免影响主业务
            pass

    def read(
        self,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SpanRecord]:
        """按条件过滤读取 Span 列表（按写入顺序返回）。"""
        results: List[SpanRecord] = []
        with self._lock:
            # 流式打开文件逐行扫描，命中 limit 即停，不预加载
            if not self.path:
                return results
            p = Path(self.path)
            if not p.exists():
                return results

            try:
                with open(p, mode="r", encoding=self.encoding, newline="") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            span_dict = json_safe_load(line)

                            if not _match(span_dict, session_id, trace_id, kind):
                                continue
                            results.append(dict_to_span_record(span_dict))
                            if limit is not None and len(results) >= limit:
                                break

                        except Exception:
                            continue
            except Exception:
                return results
        return results

    def flush(self) -> None:
        """刷盘（文件模式）。"""
        if self._fp is not None:
            try:
                self._fp.flush()
                os.fsync(self._fp.fileno())
            except Exception:
                pass

    def close(self) -> None:
        """关闭文件句柄，多次调用安全。"""
        if self._closed:
            return
        self._closed = True
        if self._fp is not None:
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None


__all__ = ["JsonlStore"]
