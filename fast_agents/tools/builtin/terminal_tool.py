"""安全的命令行工具"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult


_READONLY_COMMANDS: Set[str] = {
    "ls", "dir", "pwd", "tree", "file", "stat", "du", "df",
    "cat", "head", "tail", "less", "more", "wc",
    "find", "grep", "egrep", "fgrep", "rg",
    "sort", "uniq", "cut", "awk", "sed", "diff",
    "echo", "which", "whereis", "whoami",
    "cd",
}
_EXEC_COMMANDS: Set[str] = {"python", "python3", "node", "bash", "sh", "pip"}


class TerminalInput(BaseModel):
    command: str = Field(description="要执行的 shell 命令，如 'ls -la'、'cat README.md'")


class TerminalTool(ToolBase):
    name = "terminal"
    description = (
        "执行 shell 命令查看文件与目录。默认支持只读命令（ls, cat, head, tail, grep, find, pwd, wc 等）。"
        "工作目录被限制在指定沙箱内，无法访问上级。"
    )
    args_schema = TerminalInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = ".", timeout: int = 30, max_output_chars: int = 20_000,
                 allow_exec: bool = False, allow_cd: bool = True, allowed_commands: Optional[Set[str]] = None):
        super().__init__(max_output_chars=max_output_chars)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.allow_cd = allow_cd
        self._cwd = self.workspace
        allowed = set(_READONLY_COMMANDS)
        if allow_exec:
            allowed |= _EXEC_COMMANDS
        if allowed_commands:
            allowed |= allowed_commands
        self.allowed_commands: Set[str] = allowed

    def run(self, command: str) -> Union[str, ToolResult]:
        if not command or not command.strip():
            return ToolResult.fail("命令不能为空")
        try:
            parts = shlex.split(command.strip())
        except ValueError as e:
            return ToolResult.fail(f"命令解析失败：{e}")
        if not parts:
            return ToolResult.fail("命令不能为空")
        base = parts[0]
        if base not in self.allowed_commands:
            hint = "（如需启用代码执行，创建 TerminalTool 时传 allow_exec=True）" if base in _EXEC_COMMANDS else ""
            return ToolResult.fail(f"不允许的命令: {base}\n允许的命令: {', '.join(sorted(self.allowed_commands))}{hint}")
        if base == "cd":
            return self._handle_cd(parts)
        return self._exec(parts)

    def _handle_cd(self, parts: List[str]) -> ToolResult:
        if not self.allow_cd:
            return ToolResult.fail("cd 命令已禁用")
        target = parts[1] if len(parts) > 1 else str(self.workspace)
        if target == "~":
            new_dir = self.workspace
        elif target.startswith("/"):
            new_dir = (self.workspace / target.lstrip("/")).resolve()
        else:
            new_dir = (self._cwd / target).resolve()
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {new_dir}")
        if not new_dir.exists():
            return ToolResult.fail(f"目录不存在: {new_dir}")
        if not new_dir.is_dir():
            return ToolResult.fail(f"不是目录: {new_dir}")
        self._cwd = new_dir
        return ToolResult.ok(f"✅ 当前目录: {self._cwd.relative_to(self.workspace)}")

    def _exec(self, parts: List[str]) -> ToolResult:
        needs_shell = any(sym in " ".join(parts) for sym in ("|", ">", "<", "*", "?", "&&", ";"))
        try:
            if needs_shell:
                proc = subprocess.run(
                    " ".join(shlex.quote(p) for p in parts), shell=True, cwd=str(self._cwd),
                    capture_output=True, text=True, timeout=self.timeout,
                    env={**os.environ, "HOME": str(self.workspace)},
                )
            else:
                proc = subprocess.run(
                    parts, shell=False, cwd=str(self._cwd), capture_output=True,
                    text=True, timeout=self.timeout, env={**os.environ, "HOME": str(self.workspace)},
                )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"命令执行超时（> {self.timeout}s）")
        except FileNotFoundError:
            return ToolResult.fail(f"命令未找到：{parts[0]}，请确认已安装")
        except Exception as e:
            return ToolResult.fail(f"执行失败：{e}")
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        if proc.returncode != 0:
            output = f"[返回码 {proc.returncode}]\n{output}"
        return ToolResult.ok(output or "（命令执行成功，无输出）")

    def reset(self) -> None:
        self._cwd = self.workspace
