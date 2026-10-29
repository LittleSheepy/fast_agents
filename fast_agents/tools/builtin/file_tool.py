"""文件操作工具 - 支持乐观锁机制

提供标准的文件读写编辑能力，与已有 :class:`ReadFileTool` 相比，本模块
所有工具都在 ``artifact`` 里返回文件元数据（大小、mtime、内容 hash），
写入/编辑类工具接受这些元数据作为「乐观锁凭证」，从而避免并发或多轮
对话下的覆盖写丢数据问题：

- ReadTool:    读取文件 + 元数据缓存（返回 size/mtime/content_hash，可作为写凭证）
- WriteTool:   写入文件 + 冲突检测（expected_* 乐观锁）+ 原子写入（tmp + fsync + replace）
- EditTool:    精确替换 + 冲突检测（old_text 唯一性 + 锁凭证）+ 备份
- MultiEditTool: 批量替换 + 原子性保证（全部替换验证通过后一次性落盘）+ 备份

乐观锁工作方式：
    1. ReadTool 读文件时返回 artifact = {path, size, mtime, content_hash, ...}
    2. 调用 WriteTool/EditTool 时，把上面的 content_hash 作为 expected_hash 传入
    3. 工具在落盘前会重新 stat 文件并计算 hash，若与期望值不一致 → ToolResult.fail
       并返回当前文件的最新元数据，调用方可以选择重新读取再合并
    4. expected_hash 为空表示「不校验」（首次创建 / 强制覆盖语义）
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from ..tool_base import ToolBase
from ..tool_result import ToolResult


# ──────────────────────────────────────────────
# 公共常量与元数据工具
# ──────────────────────────────────────────────

_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pyc", ".pyo", ".class",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".mov",
    ".xlsx", ".xls", ".doc", ".ppt",
}


@dataclass
class FileMeta:
    """文件元数据（乐观锁凭证的内部结构，同时也会原样放进 ToolResult.artifact）。"""
    path: str                       # 相对 workspace 的路径
    abs_path: str                   # 绝对路径
    exists: bool
    size: int = 0
    mtime: float = 0.0
    content_hash: str = ""          # sha256 前 16 位，短但对碰撞已足够
    line_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "abs_path": self.abs_path,
            "exists": self.exists,
            "size": self.size,
            "mtime": self.mtime,
            "content_hash": self.content_hash,
            "line_count": self.line_count,
        }


def _is_binary_ext(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTS


def _sha256_hex(content: bytes, short: bool = True) -> str:
    h = hashlib.sha256(content).hexdigest()
    return h[:16] if short else h
def _normalize_newlines(text: str) -> str:
    """把各种换行符（\\r\\n / \\r / \\n）统一成 \\n。
    乐观锁的 hash 必须基于"归一化后的内容"计算，否则在 Windows 下
    编辑器/Path.write_text 默认把 \\n 写成 \\r\\n，会让 hash 在内容
    未变时仍然不匹配，出现误报冲突。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")
def _sha256_text(text: str, short: bool = True, encoding: str = "utf-8") -> str:
    """对归一化后的文本算 hash——乐观锁唯一可信的计算口径。"""
    normalized = _normalize_newlines(text)
    return _sha256_hex(normalized.encode(encoding), short=short)
def _sha256_bytes(raw: bytes, *, encoding: str = "utf-8", short: bool = True) -> str:
    """对原始字节算"归一化文本"的 hash：先解码、再归一化换行、再编码。
    用于在没有现成 str 时（_stat_file 只读 bytes）保持与 _sha256_text 一致。
    """
    try:
        text = raw.decode(encoding, errors="replace")
    except Exception:
        return _sha256_hex(raw, short=short)  # 编码异常时退化为字节 hash（二进制兜底）
    return _sha256_text(text, short=short, encoding=encoding)


def _stat_file(abs_path: Path, rel_path: str, *, read_content: bool = False,
               encoding: str = "utf-8") -> FileMeta:
    """读取文件元数据；read_content=True 时额外计算 hash 与行数。
    hash 统一按"归一化换行后的文本"计算，跨平台稳定。
    """
    if not abs_path.exists():
        return FileMeta(path=rel_path, abs_path=str(abs_path), exists=False)
    st = abs_path.stat()
    meta = FileMeta(
        path=rel_path,
        abs_path=str(abs_path),
        exists=True,
        size=st.st_size,
        mtime=st.st_mtime,
    )
    if read_content:
        try:
            raw = abs_path.read_bytes()
            meta.content_hash = _sha256_bytes(raw, encoding=encoding)
            try:
                meta.line_count = len(raw.decode(encoding, errors="replace").splitlines())
            except Exception:
                meta.line_count = 0
        except Exception:
            pass
    return meta


def _check_lock(abs_path: Path, rel_path: str,
                expected_hash: Optional[str],
                expected_mtime: Optional[float],
                expected_size: Optional[int],
                encoding: str = "utf-8") -> Tuple[Optional[ToolResult], FileMeta]:
    """乐观锁校验。返回 (错误结果或 None, 当前文件元数据)。"""
    current = _stat_file(abs_path, rel_path, read_content=True, encoding=encoding)
    # 文件不存在时，只有「期望它也不存在」才算通过
    if not current.exists:
        if expected_hash or expected_mtime is not None or expected_size is not None:
            return ToolResult.fail(
                f"乐观锁冲突：文件 {rel_path} 不存在，但传入了期望版本凭证（文件可能已被删除）。"
            ), current
        return None, current
    if expected_hash and expected_hash != current.content_hash:
        return ToolResult.fail(
            f"乐观锁冲突：文件 {rel_path} 内容已被修改（期望 hash={expected_hash}，"
            f"当前 hash={current.content_hash}，size={current.size}，mtime={current.mtime}）。"
            f"请先重新读取文件确认最新内容，再决定是否覆盖。"
        ), current
    if expected_size is not None and expected_size != current.size:
        return ToolResult.fail(
            f"乐观锁冲突：文件 {rel_path} 大小不匹配（期望 {expected_size}，当前 {current.size}）。"
        ), current
    if expected_mtime is not None and abs(expected_mtime - current.mtime) > 1e-6:
        return ToolResult.fail(
            f"乐观锁冲突：文件 {rel_path} 修改时间不匹配（期望 {expected_mtime}，当前 {current.mtime}）。"
        ), current
    return None, current


def _atomic_write(abs_path: Path, content: str, encoding: str = "utf-8",
                  normalize_newlines: bool = True) -> None:
    """原子写入：同目录临时文件 + fsync + os.replace。跨平台安全。

    normalize_newlines=True（默认）：写入前把所有换行统一为 \\n，且以 binary
    模式打开 fd，避免 Windows text mode 把 \\n 自动转成 \\r\\n。这样磁盘字节
    在任何平台上都与"归一化文本"完全一致，乐观锁的 hash + size 才能稳定。
    """
    if normalize_newlines:
        content = _normalize_newlines(content)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{abs_path.name}.", suffix=".tmp", dir=str(abs_path.parent)
    )
    try:
        # binary 模式打开，绕开 OS newline 转换
        with os.fdopen(fd, "wb") as f:
            f.write(content.encode(encoding))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_path)
    except Exception:
        # 失败时清理临时文件，不让 .xxx.tmp 残留
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _make_backup(abs_path: Path) -> str:
    """把当前文件复制一份到同目录 .bak.<timestamp>，返回备份绝对路径。"""
    if not abs_path.exists():
        return ""
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = abs_path.with_name(f"{abs_path.name}.bak.{ts}")
    # 避免同一秒内覆盖
    i = 1
    while backup_path.exists():
        backup_path = abs_path.with_name(f"{abs_path.name}.bak.{ts}-{i}")
        i += 1
    shutil.copy2(abs_path, backup_path)
    return str(backup_path)


# ──────────────────────────────────────────────
# ReadTool
# ──────────────────────────────────────────────

class ReadInput(BaseModel):
    path: str = Field(description="要读取的文件路径（相对于工作目录）")
    start_line: int = Field(default=1, ge=1, description="起始行号（从 1 开始）")
    end_line: Optional[int] = Field(default=None, ge=1, description="结束行号；不填则读到末尾或到 max_lines")
    max_lines: int = Field(default=200, ge=1, le=2000, description="最多读取行数（防止大文件爆上下文）")
    use_cache: bool = Field(default=True, description="是否使用元数据缓存；同路径重复读取且 mtime 未变时复用缓存")


class ReadTool(ToolBase):
    """读取文件并返回元数据（size/mtime/content_hash），这些字段可作为后续
    WriteTool / EditTool 的乐观锁凭证。
    """

    name = "read_file"
    description = (
        "读取工作目录内的文本文件，支持行号区间。返回内容同时附带文件元数据"
        "（size、mtime、content_hash），可作为 write_file / edit_file 的乐观锁凭证。"
        "大文件会自动截断，二进制文件无法读取。"
    )
    args_schema = ReadInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = "."):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        # 元数据缓存：abs_path(str) -> FileMeta（带 content_hash）
        self._meta_cache: Dict[str, FileMeta] = {}
        # 内容缓存：abs_path(str) -> (mtime, lines)
        self._content_cache: Dict[str, Tuple[float, List[str]]] = {}

    def _resolve(self, rel: str) -> Tuple[Optional[ToolResult], Path, str]:
        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {target}"), target, rel
        return None, target, rel

    def run(self, path: str, start_line: int = 1, end_line: Optional[int] = None,
            max_lines: int = 200, use_cache: bool = True) -> Union[str, ToolResult]:
        err, target, rel = self._resolve(path)
        if err is not None:
            return err
        if not target.exists():
            self._invalidate(str(target))
            return ToolResult.fail(f"文件不存在: {path}")
        if not target.is_file():
            return ToolResult.fail(f"不是文件: {path}（可能是目录）")
        if _is_binary_ext(target):
            return ToolResult.fail(f"看起来是二进制文件（{target.suffix}），拒绝读取")

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult.fail(f"读取失败：{e}")
        text = _normalize_newlines(text)
        lines = text.splitlines()
        total = len(lines)

        # 更新缓存
        st = target.stat()
        content_hash = _sha256_text(text)
        meta = FileMeta(
            path=rel, abs_path=str(target), exists=True,
            size=st.st_size, mtime=st.st_mtime,
            content_hash=content_hash, line_count=total,
        )
        self._meta_cache[str(target)] = meta
        if use_cache:
            self._content_cache[str(target)] = (st.st_mtime, lines)

        start = max(1, start_line)
        stop = min(end_line or (start + max_lines - 1), total, start + max_lines - 1)
        selected = lines[start - 1:stop]
        numbered = "\n".join(f"{i:>5}|{line}" for i, line in enumerate(selected, start))
        header = (
            f"📄 {path}  (共 {total} 行，显示 {start}-{stop} 行)\n"
            f"   size={meta.size}B  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.mtime))}"
            f"  hash={meta.content_hash}"
        )
        if stop < (end_line if end_line else total):
            header += f"\n⚠️ 未读完（最多 {max_lines} 行）。后续内容请把 start_line 设为 {stop + 1}"

        return ToolResult.ok(
            f"{header}\n{numbered}",
            artifact={
                **meta.to_dict(),
                "start": start, "stop": stop, "total_lines": total,
            },
        )

    def get_meta(self, path: str) -> Optional[FileMeta]:
        """获取某路径最近一次读取留下的元数据（供 Agent/其他工具直接取凭证用）。"""
        target = (self.workspace / path).resolve()
        return self._meta_cache.get(str(target))

    def _invalidate(self, abs_path: str) -> None:
        self._meta_cache.pop(abs_path, None)
        self._content_cache.pop(abs_path, None)

    def invalidate(self, path: str) -> None:
        """外部写入/修改后调用，强制使缓存失效。"""
        target = (self.workspace / path).resolve()
        self._invalidate(str(target))


# ──────────────────────────────────────────────
# WriteTool
# ──────────────────────────────────────────────

class WriteInput(BaseModel):
    path: str = Field(description="要写入的文件路径（相对于工作目录）")
    content: str = Field(description="要写入的完整文件内容")
    overwrite: bool = Field(default=True, description="文件已存在时是否允许覆盖；False 且文件存在则报错")
    expected_hash: Optional[str] = Field(default=None, description="乐观锁：期望的文件 hash（来自上次读取的 artifact.content_hash）")
    expected_size: Optional[int] = Field(default=None, ge=0, description="乐观锁：期望的文件字节数")
    expected_mtime: Optional[float] = Field(default=None, description="乐观锁：期望的文件修改时间戳")
    backup: bool = Field(default=False, description="覆盖写入前是否先备份原文件到 .bak.<timestamp>")
    encoding: str = Field(default="utf-8", description="写入编码")

    @field_validator("encoding")
    @classmethod
    def _check_encoding(cls, v: str) -> str:
        try:
            "test".encode(v)
        except LookupError:
            raise ValueError(f"不支持的编码: {v}")
        return v


class WriteTool(ToolBase):
    """原子写入文件，支持乐观锁冲突检测。

    原子性：先写入同目录临时文件，fsync 落盘，再用 os.replace 原子替换，
    进程被 kill 时不会出现半写文件。
    """

    name = "write_file"
    description = (
        "向工作目录内写入文件（覆盖创建或新建）。支持乐观锁冲突检测：传入"
        " expected_hash 后，若文件被别人改过会拒绝写入，避免覆盖他人修改。"
        "写入采用临时文件 + 原子替换，进程崩溃也不会留下半写文件。"
    )
    args_schema = WriteInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = ".", read_tool: Optional[ReadTool] = None):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._read_tool = read_tool  # 可选：用于写入后联动失效缓存

    def _resolve(self, rel: str) -> Tuple[Optional[ToolResult], Path, str]:
        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {target}"), target, rel
        return None, target, rel

    def run(self, path: str, content: str, overwrite: bool = True,
            expected_hash: Optional[str] = None, expected_size: Optional[int] = None,
            expected_mtime: Optional[float] = None, backup: bool = False,
            encoding: str = "utf-8") -> Union[str, ToolResult]:
        err, target, rel = self._resolve(path)
        if err is not None:
            return err
        if _is_binary_ext(target):
            return ToolResult.fail(f"目标扩展名 {target.suffix} 看起来是二进制，拒绝写入文本内容")
        if target.exists() and target.is_dir():
            return ToolResult.fail(f"{path} 是目录，无法写入文件")

        # 乐观锁校验（含 overwrite 语义）
        if target.exists():
            if not overwrite and not (expected_hash or expected_size is not None or expected_mtime is not None):
                return ToolResult.fail(f"文件 {path} 已存在且 overwrite=False，拒绝覆盖")
            lock_err, current = _check_lock(target, rel, expected_hash, expected_mtime, expected_size, encoding=encoding)
            if lock_err is not None:
                return lock_err
            if backup:
                backup_path = _make_backup(target)
                backup_note = f"，原文件已备份到 {Path(backup_path).name}" if backup_path else ""
            else:
                backup_note = ""
        else:
            # 文件不存在：expected_* 必须都为空（表示「期望它不存在」）
            if expected_hash or expected_size is not None or expected_mtime is not None:
                return ToolResult.fail(
                    f"文件 {path} 不存在，但传入了期望版本凭证（文件可能已被删除或尚未创建）。"
                    f"要创建新文件，请不传 expected_* 参数。"
                )
            backup_note = ""

        try:
            _atomic_write(target, content, encoding=encoding)
        except Exception as e:
            return ToolResult.fail(f"写入失败：{e}")

        # 失效缓存
        if self._read_tool is not None:
            self._read_tool.invalidate(rel)

        new_meta = _stat_file(target, rel, read_content=True, encoding=encoding)
        action = "覆盖写入" if (target.exists() and backup_note == "" and new_meta.size > 0 and
                              expected_hash is not None) else ("创建" if not backup_note and "备份" not in backup_note else "覆盖写入")
        if backup_note:
            action = "覆盖写入"
        elif not target.exists():
            pass
        else:
            # 重新判断：备份与否都属于覆盖或创建
            if not content:
                # 空写入仍可能是覆盖
                pass
        # 上面那堆判断太绕，直接用存在性 + 之前状态：
        existed_before = backup_note != "" or (expected_hash is not None) or (expected_size is not None) or (expected_mtime is not None) or (target.exists() and overwrite)
        action_word = "覆盖写入" if existed_before else "创建"
        # 注意：上面的判断在新建分支走不到这里，所以这里统一用更简单的口径：
        # 若之前已存在 → 覆盖；否则 → 创建。用 backup_note 作为已存在的可靠信号，
        # 再补一个 expected_hash 存在即说明之前读过（即文件存在）。
        existed_before = bool(backup_note) or bool(expected_hash) or expected_size is not None or expected_mtime is not None
        # 再补：文件现在 size 与第一次 stat 已无法区分（已写入），因此无法从目标本身判断，
        # 但 overwrite=False 且走到这里的分支一定是新建——overwrite=True 我们就统一表述为「写入」。
        action_word = "覆盖写入" if existed_before else ("创建" if not overwrite else "写入")

        return ToolResult.ok(
            f"✅ {action_word}成功：{path}（{new_meta.size} 字节，{new_meta.line_count} 行{backup_note}）\n"
            f"   新 hash={new_meta.content_hash}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(new_meta.mtime))}",
            artifact=new_meta.to_dict(),
        )


# ──────────────────────────────────────────────
# EditTool
# ──────────────────────────────────────────────

class EditInput(BaseModel):
    path: str = Field(description="要编辑的文件路径（相对于工作目录）")
    old_text: str = Field(description="要被替换的原文片段；必须在文件中精确出现且唯一")
    new_text: str = Field(description="替换后的新文本")
    expected_hash: Optional[str] = Field(default=None, description="乐观锁：期望的文件 hash")
    expected_size: Optional[int] = Field(default=None, ge=0, description="乐观锁：期望的文件字节数")
    expected_mtime: Optional[float] = Field(default=None, description="乐观锁：期望的文件修改时间戳")
    backup: bool = Field(default=True, description="编辑前是否备份原文件（默认 True，推荐保留）")
    append_if_missing: bool = Field(default=False, description="old_text 找不到时，是否改为追加到文件末尾（仅当文件存在时）")
    encoding: str = Field(default="utf-8", description="文件编码")

    @field_validator("encoding")
    @classmethod
    def _check_encoding(cls, v: str) -> str:
        try:
            "test".encode(v)
        except LookupError:
            raise ValueError(f"不支持的编码: {v}")
        return v

    @field_validator("old_text")
    @classmethod
    def _check_old_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("old_text 不能为空；要整体覆盖请使用 write_file")
        return v


class EditTool(ToolBase):
    """精确替换文件片段。约束：

    1. ``old_text`` 必须在文件中**精确出现且唯一**（避免改错位置）；
    2. 通过 ``expected_*`` 做乐观锁校验，文件被改过直接拒绝；
    3. 默认会在同目录留一个 ``.bak.<timestamp>`` 备份；
    4. 替换后采用原子写回，避免半写。
    """

    name = "edit_file"
    description = (
        "精确替换文件中的一段文本：把 old_text 替换为 new_text。要求 old_text 在文件中"
        "精确出现且仅出现一次（避免改错位置）。默认会先备份原文件。支持乐观锁："
        "传入 expected_hash 时，文件若被改过会拒绝写入。如需批量修改，请用 multi_edit_file。"
    )
    args_schema = EditInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = ".", read_tool: Optional[ReadTool] = None):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._read_tool = read_tool

    def _resolve(self, rel: str) -> Tuple[Optional[ToolResult], Path, str]:
        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {target}"), target, rel
        return None, target, rel

    @staticmethod
    def _find_all(text: str, needle: str) -> List[int]:
        if not needle:
            return []
        hits: List[int] = []
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx == -1:
                break
            hits.append(idx)
            start = idx + 1
        return hits

    def run(self, path: str, old_text: str, new_text: str,
            expected_hash: Optional[str] = None, expected_size: Optional[int] = None,
            expected_mtime: Optional[float] = None, backup: bool = True,
            append_if_missing: bool = False, encoding: str = "utf-8") -> Union[str, ToolResult]:
        err, target, rel = self._resolve(path)
        if err is not None:
            return err
        if _is_binary_ext(target):
            return ToolResult.fail(f"目标扩展名 {target.suffix} 看起来是二进制，拒绝编辑")
        if not target.exists():
            return ToolResult.fail(f"文件不存在: {path}（新建请用 write_file）")
        if not target.is_file():
            return ToolResult.fail(f"{path} 不是文件")

        # 乐观锁
        lock_err, current = _check_lock(target, rel, expected_hash, expected_mtime, expected_size, encoding=encoding)
        if lock_err is not None:
            return lock_err

        try:
            original = target.read_text(encoding=encoding, errors="replace")
        except Exception as e:
            return ToolResult.fail(f"读取原文件失败：{e}")
        # 所有匹配/替换基于归一化后的文本，避免 Windows CRLF 让 old_text 匹配不上
        original = _normalize_newlines(original)
        old_text_n = _normalize_newlines(old_text)
        new_text_n = _normalize_newlines(new_text)

        hits = self._find_all(original, old_text_n)
        if len(hits) == 0:
            if append_if_missing:
                # 追加：保持换行友好
                appended = original
                if appended and not appended.endswith("\n"):
                    appended += "\n"
                appended += new_text
                if not appended.endswith("\n"):
                    appended += "\n"
                new_content = appended
                action = "追加（old_text 未找到，append_if_missing=True）"
            else:
                # 为方便排错，返回 old_text 附近内容的提示
                snippet = old_text_n[:80].replace("\n", "\\n")
                return ToolResult.fail(
                    f"编辑失败：old_text 在文件 {path} 中未找到。"
                    f"请用 read_file 重新读取当前内容，确认 old_text 精确匹配（含空格/缩进/换行）。"
                    f"未匹配片段（前 80 字符）: {snippet!r}"
                )
        elif len(hits) > 1:
            preview_lines = []
            lines = original.splitlines(keepends=True)
            # 定位每行起点
            line_starts = [0]
            for ln in lines:
                line_starts.append(line_starts[-1] + len(ln))
            for pos in hits[:5]:
                # 找所在行号
                ln_no = 1
                for i, ls in enumerate(line_starts):
                    if ls <= pos:
                        ln_no = i + 1
                    else:
                        break
                col = pos - line_starts[ln_no - 1] + 1
                preview_lines.append(f"  - 第 {ln_no} 行第 {col} 列")
            more = f"（另有 {len(hits) - 5} 处）" if len(hits) > 5 else ""
            return ToolResult.fail(
                f"编辑失败：old_text 在文件 {path} 中出现了 {len(hits)} 次，不够唯一；"
                f"edit_file 要求 old_text 在文件中仅出现一次以避免改错位置。"
                f"请提供包含更多上下文的 old_text，或使用 multi_edit_file 批量替换。\n"
                f"出现位置：\n" + "\n".join(preview_lines) + more
            )
        else:
            pos = hits[0]
            new_content = original[:pos] + new_text_n + original[pos + len(old_text_n):]
            action = "替换"

        # 备份 + 原子写回
        backup_path = ""
        if backup and target.exists():
            backup_path = _make_backup(target)
        try:
            _atomic_write(target, new_content, encoding=encoding)
        except Exception as e:
            return ToolResult.fail(f"写回失败：{e}")

        # 失效缓存
        if self._read_tool is not None:
            self._read_tool.invalidate(rel)

        after_meta = _stat_file(target, rel, read_content=True, encoding=encoding)
        delta = after_meta.size - current.size
        delta_line = after_meta.line_count - (current.line_count or 0)
        backup_note = f"，原文件已备份到 {Path(backup_path).name}" if backup_path else ""
        return ToolResult.ok(
            f"✅ {action}成功：{path}（{current.size}B → {after_meta.size}B，"
            f"{'+' if delta >= 0 else ''}{delta}B, {'+' if delta_line >= 0 else ''}{delta_line} 行{backup_note}）\n"
            f"   新 hash={after_meta.content_hash}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(after_meta.mtime))}",
            artifact={
                **after_meta.to_dict(),
                "old_hash": current.content_hash,
                "old_size": current.size,
                "replaced_count": 1 if action == "替换" else 0,
                "backup_path": backup_path,
            },
        )


# ──────────────────────────────────────────────
# MultiEditTool
# ──────────────────────────────────────────────

class EditSpec(BaseModel):
    old_text: str = Field(description="要被替换的原文片段")
    new_text: str = Field(description="替换后的新文本")


class MultiEditInput(BaseModel):
    path: str = Field(description="要编辑的文件路径（相对于工作目录）")
    edits: List[EditSpec] = Field(description="批量替换列表，按顺序执行")
    expected_hash: Optional[str] = Field(default=None, description="乐观锁：期望的文件 hash")
    expected_size: Optional[int] = Field(default=None, ge=0, description="乐观锁：期望的文件字节数")
    expected_mtime: Optional[float] = Field(default=None, description="乐观锁：期望的文件修改时间戳")
    backup: bool = Field(default=True, description="编辑前是否备份原文件（默认 True）")
    dry_run: bool = Field(default=False, description="只校验是否可替换，不落盘（用于预检查）")
    encoding: str = Field(default="utf-8", description="文件编码")

    @field_validator("edits")
    @classmethod
    def _check_edits(cls, v: List[EditSpec]) -> List[EditSpec]:
        if not v:
            raise ValueError("edits 不能为空；要整体覆盖请用 write_file")
        for i, e in enumerate(v):
            if not e.old_text:
                raise ValueError(f"第 {i + 1} 条 edit 的 old_text 不能为空")
        return v

    @field_validator("encoding")
    @classmethod
    def _check_encoding(cls, v: str) -> str:
        try:
            "test".encode(v)
        except LookupError:
            raise ValueError(f"不支持的编码: {v}")
        return v


class MultiEditTool(ToolBase):
    """批量替换。原子性保证：

    1. 先把所有替换在内存中按顺序模拟一遍，任何一处 old_text 找不到 / 不唯一
       都直接返回错误，文件**一行都不动**；
    2. 全部验证通过后，做一次备份（如果 backup=True），再用原子写入整文件写回；
    3. 任何一步失败，原文件保持不变（tmp 写失败会被清理，replace 不会触发）。
    """

    name = "multi_edit_file"
    description = (
        "对同一个文件执行多段替换（edits 列表按顺序生效）。原子性：先在内存中"
        "验证所有 old_text 都能唯一匹配，有任何一处失败就整批放弃，原文件不改。"
        "验证通过后备份（默认开启）再原子写回。需要做多处修改时优先用它，"
        "避免多次 edit_file 之间被其他写入打断。"
    )
    args_schema = MultiEditInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = ".", read_tool: Optional[ReadTool] = None):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._read_tool = read_tool

    def _resolve(self, rel: str) -> Tuple[Optional[ToolResult], Path, str]:
        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {target}"), target, rel
        return None, target, rel

    def run(self, path: str, edits: List[Union[EditSpec, Dict[str, str]]],
            expected_hash: Optional[str] = None, expected_size: Optional[int] = None,
            expected_mtime: Optional[float] = None, backup: bool = True,
            dry_run: bool = False, encoding: str = "utf-8") -> Union[str, ToolResult]:
        err, target, rel = self._resolve(path)
        if err is not None:
            return err
        # 把 dict 形式的 edits 规整成 EditSpec（防止 LLM 直接传 dict 列表）
        normalized: List[EditSpec] = []
        for i, e in enumerate(edits):
            if isinstance(e, EditSpec):
                normalized.append(e)
            elif isinstance(e, dict):
                try:
                    normalized.append(EditSpec(**e))
                except Exception as ex:
                    return ToolResult.fail(f"第 {i + 1} 条 edit 格式错误：{ex}")
            else:
                return ToolResult.fail(f"第 {i + 1} 条 edit 必须是对象，收到 {type(e).__name__}")
        edits = normalized

        if _is_binary_ext(target):
            return ToolResult.fail(f"目标扩展名 {target.suffix} 看起来是二进制，拒绝编辑")
        if not target.exists():
            return ToolResult.fail(f"文件不存在: {path}（新建请用 write_file）")
        if not target.is_file():
            return ToolResult.fail(f"{path} 不是文件")

        # 乐观锁
        lock_err, current = _check_lock(target, rel, expected_hash, expected_mtime, expected_size, encoding=encoding)
        if lock_err is not None:
            return lock_err

        try:
            original = target.read_text(encoding=encoding, errors="replace")
        except Exception as e:
            return ToolResult.fail(f"读取原文件失败：{e}")
        # 跨平台换行归一化
        original = _normalize_newlines(original)

        # 在内存中按顺序模拟全部替换；每步都检查「唯一匹配」
        applied_details: List[str] = []
        current_text = original
        for i, e in enumerate(edits):
            e_old = _normalize_newlines(e.old_text)
            e_new = _normalize_newlines(e.new_text)
            hits = EditTool._find_all(current_text, e_old)
            if len(hits) == 0:
                snippet = e_old[:60].replace("\n", "\\n")
                return ToolResult.fail(
                    f"批量编辑失败：第 {i + 1}/{len(edits)} 条 old_text 在当前内容中未找到"
                    f"（前序替换已改变上下文，或原文不存在）。\n"
                    f"未匹配片段（前 60 字符）: {snippet!r}\n"
                    f"整批已回滚，原文件未改动。"
                )
            if len(hits) > 1:
                return ToolResult.fail(
                    f"批量编辑失败：第 {i + 1}/{len(edits)} 条 old_text 在当前内容中出现 {len(hits)} 次，不够唯一。"
                    f"请补充更多上下文后重试。整批已回滚，原文件未改动。"
                )
            pos = hits[0]
            current_text = current_text[:pos] + e_new + current_text[pos + len(e_old):]
            # 粗略行号：基于原始 text 估算（前序替换会让行号漂移，这里显示「替换起始位置」）
            line_no = original.count("\n", 0, pos) + 1
            applied_details.append(
                f"  #{i + 1} 第 ~{line_no} 行：{len(e_old)} 字符 → {len(e_new)} 字符"
            )

        if dry_run:
            return ToolResult.ok(
                f"🔍 dry-run 通过：{path} 的 {len(edits)} 处替换均可唯一匹配，未落盘。\n"
                + "\n".join(applied_details),
                artifact={"dry_run": True, "edit_count": len(edits), "path": rel},
            )

        # 备份 + 原子写回
        backup_path = _make_backup(target) if backup else ""
        try:
            _atomic_write(target, current_text, encoding=encoding)
        except Exception as e:
            return ToolResult.fail(f"写回失败：{e}")

        if self._read_tool is not None:
            self._read_tool.invalidate(rel)

        after_meta = _stat_file(target, rel, read_content=True, encoding=encoding)
        delta = after_meta.size - current.size
        backup_note = f"，原文件已备份到 {Path(backup_path).name}" if backup_path else ""
        return ToolResult.ok(
            f"✅ 批量替换成功：{path} 共 {len(edits)} 处（{current.size}B → {after_meta.size}B，"
            f"{'+' if delta >= 0 else ''}{delta}B{backup_note}）\n"
            + "\n".join(applied_details)
            + f"\n   新 hash={after_meta.content_hash}  "
            f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(after_meta.mtime))}",
            artifact={
                **after_meta.to_dict(),
                "old_hash": current.content_hash,
                "old_size": current.size,
                "edit_count": len(edits),
                "backup_path": backup_path,
            },
        )


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def build_file_tools(workspace: str = ".", *, enable_backup: bool = True) -> List[ToolBase]:
    """一次性构造四个文件工具，共享同一个 ReadTool 元数据缓存。

    返回顺序：read_file, write_file, edit_file, multi_edit_file。Write/Edit/MultiEdit
    会自动持有 ReadTool 的引用，写入后联动失效缓存，避免 Agent 读到陈旧元数据。
    """
    reader = ReadTool(workspace=workspace)
    writer = WriteTool(workspace=workspace, read_tool=reader)
    editor = EditTool(workspace=workspace, read_tool=reader)
    multi = MultiEditTool(workspace=workspace, read_tool=reader)
    # 默认备份开关仅影响 EditTool/MultiEditTool 的默认值——参数层不强制，
    # 这里只是让工厂使用者可以统一开关，真正的 backup 还是调用方传入
    return [reader, writer, editor, multi]


__all__ = [
    "FileMeta",
    "ReadTool", "WriteTool", "EditTool", "MultiEditTool",
    "ReadInput", "WriteInput", "EditInput", "MultiEditInput", "EditSpec",
    "build_file_tools",
]
