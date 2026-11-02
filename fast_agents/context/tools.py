"""上下文相关的工具：卸载回读 + 目录列表

这些工具让「Agent 自己管理自己的上下文」成为可能：

- ``ReadOffloadedTool``  按 offload_id 回读被卸载的工具结果（配合 ContextOffloader）
- ``ListFilesTool``      列目录（``ls``）

**文件读写编辑不在这里**：``read_file`` / ``write_file`` / ``edit_file`` / ``multi_edit_file``
统一由 ``tools/builtin/file_tool.py`` 提供（带乐观锁、原子写入、备份、审批标记）。
历史上本模块曾有一份同名的精简实现（``WriteFileTool`` / ``EditFileTool``），与前者工具名完全
相同（都是 ``write_file`` / ``edit_file``），同时注册会互相静默覆盖——`ReActAgent._init_context`
会自动补齐 ``build_file_tools()`` 四件套，无需在此重复定义。

安全边界：**一切路径限定在 workspace 内**，越界直接返回 ``ToolResult.fail``，不做任何 IO。
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
    "ListFilesTool",
    "build_context_tools",
]


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
# 2. 列目录
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
        include_filesystem: 是否包含 ``list_files`` 目录工具（文件读写编辑由
            ``tools/builtin/file_tool.py`` 统一提供，不在此处注册）

    Returns:
        ToolBase 列表，逐个 ``agent.add_tool(t)`` 即可。
        ``ReActAgent._init_context`` 会在本工厂返回的工具之后自动补齐
        ``build_file_tools()`` 的 ``read_file`` / ``write_file`` / ``edit_file`` / ``multi_edit_file``
        四件套，且对已存在的同名工具不做覆盖，避免顶掉用户预注册的实现。
    """
    tools: List[ToolBase] = []
    if offloader is not None:
        tools.append(ReadOffloadedTool(offloader))
    if include_filesystem:
        tools.append(ListFilesTool(workspace))
    return tools
