"""上下文相关的工具：卸载回读 + 文件系统四件套

这些工具让「ager 自己管理自己的上下文」成为可能：

- ``ReadOffloadedTool``  按 offload_id 回读被卸载的工具结果（配合 ContextOffloader）
- ``WriteFileTool``      写文件（会话状态外化）
- ``EditFileTool``       精确替换文件片段
- ``ListFilesTool``      列目录（``ls``）

已有 ``ReadFileTool``（tools/builtin/read_file_tool.py）承担读文件，
这里补上其余三个，凑齐路线图 F3 要求的 ``ls / read / write / edit`` 四件套。

安全边界与 ``ReadFileTool`` / ``TerminalTool`` 完全一致：**一切路径限定在 workspace 内**，
越界直接返回 ``ToolResult.fail``，不做任何 IO。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from ..tools.tool_base import ToolBase
from ..tools.tool_result import ToolResult
from .offload import ContextOffloader

__all__ = [
    "ReadOffloadedTool",
    "WriteFileTool",
    "EditFileTool",
    "ListFilesTool",
    "build_context_tools",
]

# 二进制后缀：拒绝写入/编辑，避免把文本内容糊进图片里
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".pdf", ".zip", ".whl", ".tar", ".gz", ".so", ".pyd", ".pyc", ".db",
}


def _resolve_in_workspace(workspace: Path, path: str) -> Optional[Path]:
    """把相对路径解析到 workspace 内；越界返回 None（调用方负责 fail）"""
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        return None
    return target


# ============================================================
# 1. 卸载结果回读
# ============================================================

class ReadOffloadedInput(BaseModel):
    offload_id: str = Field(description="卸载结果的 id（工具结果被卸载时会在引用文本里给出）")
    start_line: int = Field(default=1, ge=1, description="起始行号，从 1 开始")
    max_lines: int = Field(default=200, ge=1, le=2000, description="最多读取多少行")


class ReadOffloadedTool(ToolBase):
    name = "read_offloaded"
    description = (
        "按 offload_id 回读之前被卸载到磁盘的工具结果全文。"
        "当上下文里出现「[工具结果已卸载]」提示、而你需要查看完整内容时调用。"
        "支持 start_line / max_lines 分段读取，避免一次读回太多又爆上下文。"
    )
    args_schema = ReadOffloadedInput
    tags = ["context", "filesystem"]

    def __init__(self, offloader: ContextOffloader):
        super().__init__()
        self.offloader = offloader

    def run(
        self,
        offload_id: str,
        start_line: int = 1,
        max_lines: int = 200,
    ) -> Union[str, ToolResult]:
        text = self.offloader.read(offload_id, start_line=start_line, max_lines=max_lines)
        if text.startswith("[读取失败]"):
            return ToolResult.fail(text)
        return ToolResult.ok(text, artifact={"offload_id": offload_id})


# ============================================================
# 2. 写文件
# ============================================================

class WriteFileInput(BaseModel):
    path: str = Field(description="相对工作目录的文件路径，如 'notes/todo.md'")
    content: str = Field(description="要写入的完整文本内容")
    append: bool = Field(default=False, description="是否追加到文件末尾（默认覆盖）")


class WriteFileTool(ToolBase):
    name = "write_file"
    description = (
        "把文本写入工作目录内的文件（自动创建父目录）。"
        "适合把长任务的中间结果、草稿、清单落盘，避免全部堆在上下文里。"
        "默认覆盖写入，append=True 时追加。"
    )
    args_schema = WriteFileInput
    tags = ["context", "filesystem"]

    def __init__(self, workspace: str = "."):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, path: str, content: str, append: bool = False) -> Union[str, ToolResult]:
        target = _resolve_in_workspace(self.workspace, path)
        if target is None:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {path}")
        if target.suffix.lower() in _BINARY_EXTS:
            return ToolResult.fail(f"拒绝以文本方式写入二进制类型文件（{target.suffix}）")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8", newline="\n") as f:
                f.write(content)
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(f"写入失败：{e}")
        action = "追加" if append else "写入"
        size = target.stat().st_size
        return ToolResult.ok(
            f"✅ 已{action} {path}（{len(content)} 字符，文件当前 {size} 字节）",
            artifact={"path": str(target), "chars": len(content), "append": append},
        )


# ============================================================
# 3. 编辑文件
# ============================================================

class EditFileInput(BaseModel):
    path: str = Field(description="要编辑的相对路径")
    old_text: str = Field(description="要被替换的原文（必须与文件内容完全一致，含缩进）")
    new_text: str = Field(default="", description="替换后的新文本；传空字符串表示删除该片段")
    count: int = Field(default=1, ge=0, description="替换次数；0 表示替换全部出现")


class EditFileTool(ToolBase):
    name = "edit_file"
    description = (
        "在文件里做精确的字符串替换（不是模糊匹配）。"
        "使用前应先用 read_file 确认原文，old_text 必须与文件内容逐字符一致。"
        "适合小步修改文件，比整篇重写更省 token。"
    )
    args_schema = EditFileInput
    tags = ["context", "filesystem"]

    def __init__(self, workspace: str = "."):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        path: str,
        old_text: str,
        new_text: str = "",
        count: int = 1,
    ) -> Union[str, ToolResult]:
        target = _resolve_in_workspace(self.workspace, path)
        if target is None:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {path}")
        if not target.exists() or not target.is_file():
            return ToolResult.fail(f"文件不存在: {path}")
        if target.suffix.lower() in _BINARY_EXTS:
            return ToolResult.fail(f"拒绝编辑二进制类型文件（{target.suffix}）")
        try:
            original = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(f"读取失败：{e}")

        occurrences = original.count(old_text)
        if occurrences == 0:
            return ToolResult.fail(
                "未找到要替换的原文。请先用 read_file 查看文件，"
                "确保 old_text 与文件内容逐字符一致（含缩进与空行）。"
            )
        limit = occurrences if count == 0 else count
        replaced = original.replace(old_text, new_text, limit)
        try:
            target.write_text(replaced, encoding="utf-8", newline="\n")
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(f"写入失败：{e}")
        return ToolResult.ok(
            f"✅ 已修改 {path}（替换 {min(limit, occurrences)} 处，原文件共 {occurrences} 处匹配）",
            artifact={"path": str(target), "replaced": min(limit, occurrences)},
        )


# ============================================================
# 4. 列目录
# ============================================================

class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="相对工作目录的目录路径，默认当前目录")
    recursive: bool = Field(default=False, description="是否递归列出子目录")
    max_entries: int = Field(default=200, ge=1, le=1000, description="最多列出多少条")


class ListFilesTool(ToolBase):
    name = "list_files"
    description = (
        "列出工作目录内的文件与子目录（等价于 ls）。"
        "不确定文件放在哪、或想先看清目录结构时先调它。"
    )
    args_schema = ListFilesInput
    tags = ["context", "filesystem"]

    def __init__(self, workspace: str = "."):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> Union[str, ToolResult]:
        target = _resolve_in_workspace(self.workspace, path)
        if target is None:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {path}")
        if not target.exists():
            return ToolResult.fail(f"路径不存在: {path}")
        if not target.is_dir():
            return ToolResult.ok(f"📄 {path} 是一个文件（{target.stat().st_size} 字节），不是目录")

        entries: List[str] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        try:
            for item in sorted(iterator, key=lambda p: (p.is_file(), str(p).lower())):
                rel = item.relative_to(target)
                if item.is_dir():
                    entries.append(f"[DIR ] {rel}/")
                else:
                    entries.append(f"[FILE] {rel}  ({item.stat().st_size} 字节)")
                if len(entries) >= max_entries:
                    entries.append(f"...（已达 {max_entries} 条上限，更多内容请缩小目录范围）")
                    break
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(f"列目录失败：{e}")

        if not entries:
            return ToolResult.ok(f"📁 {path} 是空目录")
        header = f"📁 {path} 目录{'（递归）' if recursive else ''}，共 {len(entries)} 条："
        return ToolResult.ok(
            header + "\n" + "\n".join(entries),
            artifact={"path": str(target), "entries": len(entries)},
        )


# ============================================================
# 工厂
# ============================================================

def build_context_tools(
    workspace: str = ".",
    offloader: Optional[ContextOffloader] = None,
    include_filesystem: bool = True,
) -> List[ToolBase]:
    """按需组装 context 模块自带的工具

    Args:
        workspace: 工作目录
        offloader: 卸载器实例；传入时才会生成 ``read_offloaded`` 工具
        include_filesystem: 是否包含 write/edit/list 三件套

    Returns:
        ToolBase 列表，逐个 ``agent.add_tool(t)`` 即可
    """
    tools: List[ToolBase] = []
    if offloader is not None:
        tools.append(ReadOffloadedTool(offloader))
    if include_filesystem:
        tools.extend(
            [
                ListFilesTool(workspace),
                WriteFileTool(workspace),
                EditFileTool(workspace),
            ]
        )
    return tools
