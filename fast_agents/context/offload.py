"""工具结果卸载（offload）与按需回读

问题：一次 ``read_file`` 或 ``web_search`` 可能返回几万字符，
直接塞进 tool 消息，几轮下来上下文就爆了——而且后面几轮根本不再需要全文。

方案（对齐 Anthropic「tool result 截断 + 文件回读」与 DeepAgents 的
「会话状态外化成文件」）：**大结果落盘，上下文里只留摘要 + 引用 id**，
模型真想看全文时，再通过 `read_offloaded` 工具按 id 回读（可指定行区间）。

落盘布局::

    <root>/                 # 默认 <workspace>/.fast_agents/context/offload
    ├── index.json          # 索引：id → 元数据（工具名、原始长度、摘要、预览）
    └── <id>.txt            # 正文

三个设计取舍：
1. **零依赖**：用 `pathlib` + `json`，不引任何数据库；
2. **有界增长**：超过 ``max_files`` 时按「最早创建」淘汰，避免无限膨胀；
3. **路径安全**：id 只允许十六进制字符，杜绝 ``../`` 穿越（参考 ReadFileTool 的沙箱思路）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

__all__ = ["OffloadRecord", "ContextOffloader"]

# id 白名单：只放十六进制与小写字母数字，杜绝路径穿越
_SAFE_ID = re.compile(r"^[0-9a-f]{6,64}$")


@dataclass
class OffloadRecord:
    """一次卸载的元数据（会写进 index.json）"""

    offload_id: str
    tool_name: str
    path: str
    original_chars: int
    summary: str = ""
    preview: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump() if hasattr(self, "model_dump") else dict(self.__dict__)
        data["created_at"] = self.created_at
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OffloadRecord":
        return cls(
            offload_id=data["offload_id"],
            tool_name=data.get("tool_name", ""),
            path=data.get("path", ""),
            original_chars=int(data.get("original_chars", 0)),
            summary=data.get("summary", ""),
            preview=data.get("preview", ""),
            created_at=float(data.get("created_at", time.time())),
            metadata=data.get("metadata", {}) or {},
        )


class ContextOffloader:
    """把超长工具结果卸载到磁盘，并支持按 id / 行区间回读

    使用方式::

        offloader = ContextOffloader(workspace=".")
        result = offloader.offload("web_search", 长文本, threshold=4000)
        print(result)          # ← 替换 tool 消息内容的简短引用文本
        full = offloader.read(offloader.last_id, start_line=1, max_lines=200)
    """

    def __init__(
        self,
        workspace: str = ".",
        subdir: str = ".fast_agents/context/offload",
        threshold: int = 4_000,
        preview_chars: int = 400,
        max_files: int = 200,
    ):
        """
        Args:
            workspace: 工作目录根（卸载文件只会落在它下面）
            subdir: 卸载目录相对路径
            threshold: 超过多少字符触发卸载
            preview_chars: 引用文本里保留多少字符的预览
            max_files: 卸载文件数量上限，超出按「最早创建」淘汰
        """
        self.workspace = Path(workspace).resolve()
        self.root = (self.workspace / subdir).resolve()
        # 安全兜底：卸载目录必须仍在 workspace 内
        try:
            self.root.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover —— 正常配置不会触发
            raise ValueError(f"卸载目录必须位于工作目录内：{self.root}") from exc
        self.root.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.preview_chars = preview_chars
        self.max_files = max_files
        self._index_path = self.root / "index.json"
        self._index: Dict[str, OffloadRecord] = self._load_index()
        # 最近一次卸载的 id，方便调用方紧接着回读
        self.last_id: Optional[str] = None

    # ---------- 索引 ----------

    def _load_index(self) -> Dict[str, OffloadRecord]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 索引损坏不能让框架挂掉
            return {}
        records: Dict[str, OffloadRecord] = {}
        for key, value in (raw or {}).items():
            try:
                records[key] = OffloadRecord.from_dict(value)
            except Exception:  # noqa: BLE001
                continue
        return records

    def _save_index(self) -> None:
        try:
            data = {k: v.to_dict() for k, v in self._index.items()}
            self._index_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 —— 索引写失败不影响正文已落盘
            pass

    # ---------- 卸载 ----------

    def should_offload(self, text: str, threshold: Optional[int] = None) -> bool:
        """这段文本是否该被卸载"""
        return len(text or "") > (self.threshold if threshold is None else threshold)

    def offload(
        self,
        tool_name: str,
        text: str,
        *,
        threshold: Optional[int] = None,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """把超长结果写盘，返回「替代原文」的简短引用文本

        文本没超过阈值时原样返回，不做任何 IO —— 所以可以无脑在
        Agent 每次工具返回后都调一次。

        Args:
            tool_name: 来源工具名（写进索引，便于溯源）
            text: 原始结果文本
            threshold: 本次的触发阈值；不传用实例默认值
            summary: 可选摘要（没有时自动取预览）
            metadata: 附加元数据

        Returns:
            可直接作为 tool 消息 content 的引用文本
        """
        text = text or ""
        if not self.should_offload(text, threshold):
            return text

        offload_id = uuid4().hex[:16]
        path = self.root / f"{offload_id}.txt"
        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:  # noqa: BLE001 —— 落盘失败就退回截断，不能让对话中断
            return self._fallback_truncate(text, str(e))

        preview = text[: self.preview_chars]
        record = OffloadRecord(
            offload_id=offload_id,
            tool_name=tool_name,
            path=str(path),
            original_chars=len(text),
            summary=summary or preview.replace("\n", " ")[:200],
            preview=preview,
            metadata=metadata or {},
        )
        self._index[offload_id] = record
        self.last_id = offload_id
        self._save_index()
        self._enforce_limit()

        return self._render_reference(record)

    def _render_reference(self, record: OffloadRecord) -> str:
        """渲染模型看到的引用文本：明确告知「内容已外置 + 怎么取回」"""
        return (
            f"[工具结果已卸载] 工具 `{record.tool_name}` 返回了 {record.original_chars} 字符，"
            f"全文已写入离线文件，不在上下文中。\n"
            f"offload_id={record.offload_id}\n"
            f"内容开头预览：{record.preview[:200]}\n"
            f"如需查看完整内容，调用工具 read_offloaded(tool_name 的 offload_id="
            f"\"{record.offload_id}\"，可选 start_line/max_lines) 按需读取。"
        )

    @staticmethod
    def _fallback_truncate(text: str, reason: str) -> str:
        limit = 2_000
        note = f"\n\n[⚠️ 结果卸载失败（{reason}），已截断为前 {limit} 字符]"
        return text[:limit] + note

    def _enforce_limit(self) -> None:
        """超出文件数上限时，淘汰最早创建的卸载文件"""
        if self.max_files <= 0 or len(self._index) <= self.max_files:
            return
        ordered = sorted(self._index.values(), key=lambda r: r.created_at)
        for record in ordered[: len(self._index) - self.max_files]:
            try:
                Path(record.path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            self._index.pop(record.offload_id, None)
        self._save_index()

    # ---------- 回读 ----------

    def get_record(self, offload_id: str) -> Optional[OffloadRecord]:
        return self._index.get(offload_id)

    def read(
        self,
        offload_id: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_lines: int = 200,
    ) -> str:
        """按 id 回读卸载内容（支持行区间，行为对齐 ReadFileTool）

        Returns:
            带行号的文本；id 不存在 / 文件丢失时返回可读的错误提示（而不是抛异常）
        """
        if not _SAFE_ID.match(offload_id or ""):
            return f"[读取失败] 非法的 offload_id：{offload_id!r}"
        record = self._index.get(offload_id)
        if record is None:
            return f"[读取失败] 未找到 offload_id={offload_id} 的卸载记录（可能已被淘汰）"
        path = Path(record.path)
        try:
            path.relative_to(self.workspace)
        except ValueError:
            return "[读取失败] 卸载文件路径越界，已拒绝读取"
        if not path.exists():
            return f"[读取失败] 卸载文件已不存在：{path.name}"

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:  # noqa: BLE001
            return f"[读取失败] {e}"

        total = len(lines)
        start = max(1, start_line)
        stop = min(end_line or total, total, start + max(1, max_lines) - 1)
        selected = lines[start - 1: stop]
        numbered = "\n".join(f"{i:>5}|{line}" for i, line in enumerate(selected, start))
        header = (
            f"📦 卸载内容 {offload_id}"
            f"（来自工具 {record.tool_name}，共 {total} 行，显示 {start}-{stop} 行）"
        )
        if stop < total:
            header += f"\n⚠️ 未读完，如需后续内容把 start_line 设为 {stop + 1}"
        return f"{header}\n{numbered}"

    def list_records(self) -> List[OffloadRecord]:
        """按创建时间倒序列出全部卸载记录"""
        return sorted(self._index.values(), key=lambda r: r.created_at, reverse=True)

    def clear(self) -> int:
        """清空全部卸载文件与索引，返回删除的文件数"""
        count = 0
        for record in list(self._index.values()):
            try:
                Path(record.path).unlink(missing_ok=True)
                count += 1
            except Exception:  # noqa: BLE001
                pass
        self._index.clear()
        self.last_id = None
        self._save_index()
        return count

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"<ContextOffloader root={self.root} files={len(self._index)} threshold={self.threshold}>"
