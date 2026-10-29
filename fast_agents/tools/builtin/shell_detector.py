"""shell_detector.py
Shell 探测与路由模块：
1. 优先用 shellingham 探测当前终端环境；
2. 再用命令文本特征判断 CMD / PowerShell；
3. 最后用父进程/环境变量兜底探测。
仅支持 Windows，去掉跨平台 bash 相关逻辑。
"""

import os
import re
import shutil
import subprocess
from typing import Literal, Tuple

ShellType = Literal["cmd", "powershell", "unknown"]


def detect_parent_shell() -> Tuple[ShellType, str]:
    """通过父进程判断当前shell类型"""
    try:
        import psutil

        parent = psutil.Process(os.getpid()).parent()
        if parent is not None:
            pname = parent.name().lower()
            exe_path = parent.exe().lower() if hasattr(parent, "exe") else pname
            if "powershell" in pname or "pwsh" in pname:
                return "powershell", exe_path
            if pname == "cmd.exe" or "cmd.exe" in exe_path:
                return "cmd", exe_path
    except Exception:
        pass
    return "unknown", ""


def detect_shell_by_shellingham() -> Tuple[ShellType, str]:
    """通过 shellingham 库探测shell"""
    try:
        import shellingham
        name, path = shellingham.detect_shell()
        name_lower = name.lower()
        if name_lower in ("powershell", "pwsh"):
            return "powershell", path
        if name_lower == "cmd":
            return "cmd", path
        return "unknown", path
    except ImportError:
        pass
    except Exception:
        pass
    return "unknown", ""


def detect_available_shells() -> dict[str, bool]:
    """检测系统可用的shell"""
    return {
        "cmd": bool(shutil.which("cmd.exe")),
        "powershell": bool(shutil.which("pwsh") or shutil.which("powershell.exe")),
    }


def detect_shell_from_env() -> Tuple[ShellType, str]:
    """通过环境变量判断shell"""
    powershell_path = shutil.which("pwsh") or shutil.which("powershell.exe")
    if powershell_path and "PSMODULEPATH" in os.environ:
        return "powershell", powershell_path
    
    comspec = os.environ.get("COMSPEC", "").lower()
    if "powershell" in comspec:
        return "powershell", comspec

    cmd_path = shutil.which("cmd.exe")
    if cmd_path and "cmd.exe" in comspec:
        return "cmd", cmd_path

    return "unknown", ""


def detect_shell() -> Tuple[ShellType, str]:
    """多层级探测当前 Shell，返回(shell类型, 路径)"""
    # 1. shellingham 库优先
    name, path = detect_shell_by_shellingham()
    if name != "unknown":
        return name, path

    # 2. 父进程探测兜底
    name, path = detect_parent_shell()
    if name != "unknown":
        return name, path

    # 3. 环境变量兜底
    name, path = detect_shell_from_env()
    if name != "unknown":
        return name, path

    # 4. Windows 默认优先 powershell
    ps_path = shutil.which("powershell.exe")
    if ps_path:
        return "powershell", ps_path
    cmd_path = shutil.which("cmd.exe")
    if cmd_path:
        return "cmd", cmd_path

    return "unknown", ""


def guess_shell_from_command(command: str) -> ShellType:
    """根据命令文本特征猜测 CMD / PowerShell，用于路由选择"""
    cmd = command.strip()
    if not cmd:
        return "unknown"

    ps_score = 0
    cmd_score = 0

    # PowerShell 强特征
    if re.search(r'\$[A-Za-z_\{]', cmd):
        ps_score += 3
    if re.search(r'\$env:', cmd, re.I):
        ps_score += 4
    if re.search(r'\bGet-|Set-|New-|Remove-|Invoke-|Test-|Stop-|Start-|Restart-', cmd):
        ps_score += 3
    if re.search(r'\bWhere-Object\b|\bForEach-Object\b|\bSelect-String\b', cmd):
        ps_score += 2
    if re.search(r'\bImport-Module\b|\bGet-Module\b|\bSet-ExecutionPolicy\b', cmd):
        ps_score += 2
    if re.search(r'--%(\s|$)', cmd):
        ps_score += 3
    if re.search(r'`[nt"\'$]', cmd):
        ps_score += 2

    # CMD 强特征
    if re.search(r'%[A-Za-z_][A-Za-z0-9_]*%', cmd):
        cmd_score += 3
    if re.search(r'\bset\s+\w+=', cmd, re.I):
        cmd_score += 2
    if re.search(r'\bsetlocal\b|\bendlocal\b|\benabledelayedexpansion\b', cmd, re.I):
        cmd_score += 3
    if re.search(r'\bgoto\s+', cmd, re.I):
        cmd_score += 3
    if re.search(r'\b@echo\s+off\b', cmd, re.I):
        cmd_score += 3
    if re.search(r'\bcall\s+', cmd, re.I):
        cmd_score += 2

    # 弱特征
    if re.search(r'\btype\b|\bfindstr\b', cmd):
        cmd_score += 1
    if re.search(r'\bGet-ChildItem\b|\bSet-Location\b|\bGet-Content\b', cmd):
        ps_score += 1

    # 短命令别名
    ps_aliases = {"pwd", "ls", "cat", "rm", "cp", "mv", "ps", "touch", "which", "mkdir", "rmdir"}
    cmd_builtins = {"dir", "copy", "del", "ren", "move", "findstr", "cls", "echo", "cd", "md", "rd"}

    first_token = cmd.split()[0].lower()
    if first_token in ps_aliases and first_token not in cmd_builtins:
        ps_score += 1
    elif first_token in cmd_builtins and first_token not in ps_aliases:
        cmd_score += 1

    if ps_score > cmd_score + 1:
        return "powershell"
    if cmd_score > ps_score + 1:
        return "cmd"
    return "unknown"


def run_command(
    command: str,
    shell_type: ShellType = "auto",
    needs_shell: bool = False,
    **kwargs,
):
    """执行命令，自动选择 CMD / PowerShell。

    Args:
        command: 要执行的命令字符串。
        shell_type: "auto" / "cmd" / "powershell"。
        needs_shell: 是否需要 shell=True（由安全检查模块传入，不下发给 subprocess）。
        **kwargs: 其余参数原样透传给 subprocess.run（如 cwd、timeout、env 等），
                  以后新增底层参数无需再改本函数签名。

    Returns:
        subprocess.CompletedProcess
    """
    if shell_type == "auto":
        guessed = guess_shell_from_command(command)
        if guessed in ("cmd", "powershell"):
            effective_shell = guessed
        else:
            detected_shell, _ = detect_shell()
            effective_shell = detected_shell if detected_shell in ("cmd", "powershell") else "powershell"
    else:
        effective_shell = shell_type

    # 组装默认参数（timeout 默认 10s；cwd / env 等由调用方按需透传）
    opts = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 10,
        "stdin": subprocess.DEVNULL,
    }
    
    # 调用方传入的 kwargs（cwd / env / timeout 等）优先级更高，统一在此合并
    opts.update(kwargs)

    # 如果不需要shell，直接用分词后的argv执行，更安全
    if not needs_shell:
        from .command_splitter import split_windows
        try:
            argv = split_windows(command)
            if not argv:
                raise ValueError("empty command")
            # 去掉参数外层的引号（Windows argv不需要保留引号）
            clean_argv = []
            for part in argv:
                if len(part) >= 2 and part.startswith('"') and part.endswith('"'):
                    clean_argv.append(part[1:-1])
                else:
                    clean_argv.append(part)
            return subprocess.run(clean_argv, shell=False, **opts)
        except Exception:
            # 分词失败就回退到shell模式
            pass

    # 需要shell的情况：按目标shell构造argv
    if effective_shell == "cmd":
        argv_or_cmd = ["cmd", "/c", command]
    else:  # powershell
        argv_or_cmd = ["powershell", "-NoProfile", "-Command", command]
    
    return subprocess.run(argv_or_cmd, shell=False, **opts)


def route_and_run(command: str, needs_shell: bool = False, **kwargs) -> Tuple[int, str, str, ShellType]:
    """路由并执行：先试首选 Shell，失败则回退到另一个。

    Args:
        command: 要执行的命令字符串。
        needs_shell: 是否需要 shell=True（由安全检查模块传入，不下发给 subprocess）。
        **kwargs: 其余参数原样透传给 run_command / subprocess.run（如 cwd、timeout、env 等）。
    """
    detected_shell, _ = detect_shell()
    guessed = guess_shell_from_command(command)

    # 确定首选 Shell
    if guessed in ("cmd", "powershell"):
        preferred = guessed
    elif detected_shell in ("cmd", "powershell"):
        preferred = detected_shell
    else:
        preferred = "powershell"

    fallback = "cmd" if preferred == "powershell" else "powershell"

    try:
        proc = run_command(command, shell_type=preferred, needs_shell=needs_shell, **kwargs)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
        if rc == 0 or not err.strip():
            return rc, out, err, preferred
    except Exception as e:
        rc, out, err = -1, "", str(e)

    # 回退尝试另一个shell
    try:
        proc = run_command(command, shell_type=fallback, needs_shell=needs_shell, **kwargs)
        rc2, out2, err2 = proc.returncode, proc.stdout, proc.stderr
        if rc2 == 0 or not err2.strip():
            return rc2, out2, err2, fallback
    except Exception:
        pass

    # 都失败，返回首选的结果
    return rc, out, err, preferred


if __name__ == "__main__":
    shell_name, shell_path = detect_shell()
    print(f"Detected shell: {shell_name} -> {shell_path}")

    test_commands = [
        "echo %USERNAME%",
        "echo $env:USERNAME",
        "dir",
    ]

    for cmd in test_commands:
        guessed = guess_shell_from_command(cmd)
        rc, out, err, used = route_and_run(cmd)
        print(f"\nCommand : {cmd}")
        print(f"Guessed : {guessed}")
        print(f"Used    : {used}")
        print(f"RC      : {rc}")
        print(f"Output  : {out.strip()}")
        if err.strip():
            print(f"Error   : {err.strip()}")
