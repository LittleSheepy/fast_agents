"""command_splitter.py
Windows命令分词与多命令拆分，保留原作者实现，仅补充安全校验功能
"""

from typing import List, Tuple


def split_windows(command: str) -> List[str]:
    """Windows 安全的简单命令分词，不破坏反斜杠路径。

    规则：
    - 双引号包裹的参数视为一个整体（引号保留）
    - 其余按空白分割
    - 不做反斜杠转义处理（与 POSIX shlex 的关键区别）
    """
    parts: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == '"' and not in_quotes:
            in_quotes = True
            current.append(ch)
        elif ch == '"' and in_quotes:
            in_quotes = False
            current.append(ch)
        elif ch in (' ', '\t', '\n') and not in_quotes:
            if current:
                parts.append(''.join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append(''.join(current))
    return parts


def split_windows_commands(command: str) -> List[str]:
    """把含 CMD 操作符的命令行拆成多条独立命令字符串。

    识别的操作符：&&、||、|、&
    - 引号内部的操作符不会被拆分
    - 操作符本身不会出现在结果中
    - 空命令片段自动过滤
    """
    tokens: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0

    while i < len(command):
        ch = command[i]

        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
            i += 1
            continue

        if not in_quotes:
            # 尝试匹配双字符操作符 &&、||
            if i + 1 < len(command) and command[i : i + 2] in ("&&", "||"):
                if current:
                    tokens.append(''.join(current).strip())
                    current = []
                i += 2
                continue

            # 匹配单字符操作符 |、&（注意 & 后面紧跟 & 已在上面处理）
            if ch in ("|", "&"):
                if current:
                    tokens.append(''.join(current).strip())
                    current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    if current:
        stripped = ''.join(current).strip()
        if stripped:
            tokens.append(stripped)

    return tokens


# ========== 补充：安全校验功能 ==========

# 危险命令黑名单（坚决不允许执行）
DANGEROUS_COMMANDS = frozenset({
    # 格式化/磁盘操作
    "format", "fdisk", "diskpart", "convert",
    # 删除/销毁
    "del", "/s", "/q", "/f", "rmdir", "/s", "/q", "rd",
    "erase", "wipe",
    # 注册表危险操作
    "reg", "delete",
    # 系统关闭
    "shutdown", "reboot", "restart", "logoff",
    # 进程杀除
    "taskkill", "/im", "/f", "tskill",
    # 其他危险操作
    "net", "user", "localgroup", "administrators",
    "cipher", "/w",  # 擦除磁盘空闲空间
    "takeown", "/f",  # 夺取文件所有权
    "icacls", "/grant",  # 修改权限
})

# 危险参数模式
DANGEROUS_ARGS_PATTERNS = frozenset({
    "/s", "/q", "/f", "/s /q", "/q /s", "/f /s",  # del/rmdir 递归强制删除
    "-rf", "-fr", "-r -f",  # 类unix递归强制删除
    "/im", "/f",  # taskkill
    "/w",  # cipher擦除
    "-c", "-e",  # 传递代码给解释器
    "/c", "del", "/c", "format", "/c", "shutdown",
})


def validate_command_safety(cmd_parts: List[str]) -> Tuple[bool, str]:
    """校验单条分词后的命令是否安全，返回(是否安全, 错误信息)"""
    if not cmd_parts:
        return False, "空命令"

    base_cmd = cmd_parts[0].lower().strip('"').strip("'")

    # 1. 直接禁止的危险命令
    if base_cmd in DANGEROUS_COMMANDS and base_cmd not in {"del", "rd", "rmdir", "taskkill", "reg"}:
        # 上面单独判断的命令带危险参数才禁止
        return False, f"禁止执行危险命令: {base_cmd}"

    # 2. 检查删除类命令的危险参数
    if base_cmd in {"del", "erase", "rd", "rmdir"}:
        args_lower = " ".join(p.lower() for p in cmd_parts[1:])
        if "/s" in args_lower or "/q" in args_lower or "/f" in args_lower:
            return False, f"禁止执行递归/强制删除操作: {' '.join(cmd_parts)}"

    # 3. 检查taskkill强制杀进程
    if base_cmd == "taskkill" or base_cmd == "tskill":
        args_lower = " ".join(p.lower() for p in cmd_parts[1:])
        if "/f" in args_lower or "/im" in args_lower:
            return False, f"禁止强制杀进程: {' '.join(cmd_parts)}"

    # 4. 禁止通过cmd/c或powershell/c绕过白名单执行嵌套命令
    if base_cmd in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}:
        args_lower = " ".join(p.lower() for p in cmd_parts[1:])
        if "/c" in args_lower or "-command" in args_lower or "-c" in args_lower:
            # 提取嵌套命令，检查是否在白名单内（这里直接拒绝嵌套执行危险命令）
            nested_cmd = args_lower.replace("/c", "").replace("-command", "").replace("-c", "").strip()
            if nested_cmd:
                nested_base = nested_cmd.split()[0] if nested_cmd.split() else ""
                if nested_base.lower() in DANGEROUS_COMMANDS:
                    return False, f"禁止通过嵌套Shell执行危险命令: {nested_base}"

    return True, ""


if __name__ == "__main__":
    # 测试原拆分逻辑
    SPLIT_WINDOWS_COMMANDS_CASES = {
        "dir C:\\Users": [["dir", "C:\\Users"]],
        "dir C:\\Users && echo done": [
            ["dir", "C:\\Users"],
            ["echo", "done"],
        ],
        "echo \"a && b\"": [["echo", "\"a && b\""]],
    }

    print("=== 分词测试 ===")
    for cmd, expected in SPLIT_WINDOWS_COMMANDS_CASES.items():
        # 先拆多命令
        cmds = split_windows_commands(cmd)
        result = [split_windows(c) for c in cmds]
        print(f"{cmd!r} => {result}")

    print("\n=== 安全校验测试 ===")
    test_cmds = [
        ["dir", "C:\\Users"],
        ["del", "/s", "/q", "C:\\"],
        ["format", "C:"],
        ["taskkill", "/f", "/im", "explorer.exe"],
        ["echo", "hello"],
    ]
    for parts in test_cmds:
        ok, msg = validate_command_safety(parts)
        print(f"{'✅' if ok else '❌'} {parts} => {msg if msg else '安全'}")
