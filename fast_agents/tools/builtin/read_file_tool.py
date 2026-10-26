"""本地文件读取工具（路径沙箱 + 行号区间）"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult


class ReadFileInput(BaseModel):
    path: str = Field(description="要读取的文件路径（相对于工作目录）")
    start_line: int = Field(default=1, ge=1, description="起始行号（从 1 开始）")
    end_line: Optional[int] = Field(default=None, ge=1, description="结束行号；不填则读到文件末尾或到 max_lines")
    max_lines: int = Field(default=200, ge=1, le=2000, description="最多读取行数（防止大文件爆上下文）")


class ReadFileTool(ToolBase):
    name = "read_file"
    description = "读取工作目录内的文本文件。支持指定行号区间，大文件会自动截断。二进制文件无法读取。"
    args_schema = ReadFileInput
    tags = ["builtin", "filesystem"]

    _BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib", ".bin",
        ".pyc", ".pyo", ".class",
        ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".mov",
        ".xlsx", ".xls", ".doc", ".ppt",
    }

    def __init__(self, workspace: str = "."):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, path: str, start_line: int = 1, end_line: Optional[int] = None, max_lines: int = 200) -> Union[str, ToolResult]:
        target = (self.workspace / path).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {target}")
        if not target.exists():
            return ToolResult.fail(f"文件不存在: {path}")
        if not target.is_file():
            return ToolResult.fail(f"不是文件: {path}（可能是目录）")
        if target.suffix.lower() in self._BINARY_EXTS:
            return ToolResult.fail(f"看起来是二进制文件（{target.suffix}），拒绝读取")
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return ToolResult.fail(f"读取失败：{e}")
        total = len(lines)
        start = max(1, start_line)
        stop = min(end_line or (start + max_lines - 1), total, start + max_lines - 1)
        selected = lines[start - 1 : stop]
        numbered = "\n".join(f"{i:>5}|{line}" for i, line in enumerate(selected, start))
        header = f"📄 {path}  (共 {total} 行，显示 {start}-{stop} 行)"
        if stop < (end_line or total):
            header += f"\n⚠️ 未读完（最多 {max_lines} 行）。如需后续内容，再次调用并把 start_line 设为 {stop + 1}"
        return ToolResult.ok(
            f"{header}\n{numbered}",
            artifact={"path": str(target), "total_lines": total, "start": start, "stop": stop},
        )
