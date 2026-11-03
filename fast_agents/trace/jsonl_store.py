"""JsonlStore —— 默认 JSONL 落盘存储后端。

一行一个 JSON 对象（jsonl 格式），零依赖（仅标准库 json + pathlib）；
同时支持 ``:memory:`` 模式（仅在内存 list 里，不落盘，适合测试/调试）。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, List, Optional

from .models import SpanKind, SpanRecord, _now_ts
from .store_base import TraceStoreBase


# 内存模式占位路径（与 SQLiteClient ``":memory:"`` 约定对齐）
MEMORY_PATH = ":memory:"


def _safe_serialize(obj: Any) -> Any:
    """把任意对象转成 JSON 可序列化的形式。

    规则：
    - ``None`` / 基本类型（str/int/float/bool）原样返回；
    - pydantic v2 BaseModel → model_dump(mode="json")；
    - dict / list / tuple → 递归处理；
    - 其它对象 → str(obj)（兜底，保证不抛异常）。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # pydantic v2 BaseModel
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _safe_serialize(obj.model_dump(mode="json"))
        except Exception:
            return str(obj)
    # pydantic v1 兜底（.dict()）
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _safe_serialize(obj.dict())
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError, OverflowError):
        return str(obj)


class JsonlStore(TraceStoreBase):
    """JSONL 一行一条 Span 的存储后端。

    Args:
        path: jsonl 文件路径；传 ``":memory:"`` 或 ``None`` 则只在内存里保存（不落盘）。
        append: True 表示追加到已有文件（默认）；False 表示清空重写。
        encoding: 文件编码，默认 ``utf-8``。
    """

    def __init__(
        self,
        path: Optional[str] = MEMORY_PATH,
        *,
        append: bool = True,
        encoding: str = "utf-8",
    ):
        self.path = path
        self.append = append
        self.encoding = encoding
        self._lock = threading.Lock()
        self._mem: List[dict] = []  # 内存模式下的缓冲；文件模式下也用于 read() 读已写入内容
        self._fp = None
        self._closed = False

        if path and path != MEMORY_PATH:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            # buffering=1 行缓冲；newline="" 让 json 写入跨平台一致，由我们控制 \n
            self._fp = open(p, mode=mode, encoding=encoding, buffering=1, newline="")
            # 读一遍已有文件，把历史记录装入 _mem，便于 read() 查询
            if append and p.exists():
                try:
                    with open(p, mode="r", encoding=encoding, newline="") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                self._mem.append(json.loads(line))
                            except Exception:
                                continue
                except Exception:
                    pass

    # ---------- TraceStoreBase 接口 ----------
    def write(self, span: SpanRecord) -> None:
        """写入一条 Span。任何异常都会被吞掉（打点不能阻塞业务）。"""
        if self._closed:
            return
        try:
            # 使用 model_dump(mode="json")，pydantic v2 会自动把 enum/datetime/UUID 转成 JSON 安全形式
            data = span.model_dump(mode="json", exclude_none=False)
            data["input"] = _safe_serialize(data.get("input"))
            data["output"] = _safe_serialize(data.get("output"))
            data["metadata"] = _safe_serialize(data.get("metadata") or {})
            # 兜底：kind/status 保证是字符串值（pydantic mode="json" 已经处理，这里再守一次）
            k = data.get("kind")
            if hasattr(k, "value"):
                data["kind"] = k.value
            st = data.get("status")
            if hasattr(st, "value"):
                data["status"] = st.value
            line = json.dumps(data, ensure_ascii=False, default=str)
            with self._lock:
                self._mem.append(data)
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
            items = list(self._mem)
        for d in items:
            if session_id is not None and d.get("session_id") != session_id:
                continue
            if trace_id is not None and d.get("trace_id") != trace_id:
                continue
            if kind is not None and d.get("kind") != kind:
                continue
            try:
                results.append(SpanRecord(**d))
            except Exception:
                continue
            if limit is not None and len(results) >= limit:
                break
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["JsonlStore", "MEMORY_PATH"]
