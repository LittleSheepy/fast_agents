"""shell_check.py
判断命令是否需要 shell=True 执行
仅支持 Windows，修复原实现中不考虑引号的问题
"""

from typing import List


# Windows CMD / PowerShell 通用 shell 元字符（引号外出现才真正需要 shell）
WIN_SHELL_METACHARS = frozenset((
    "|", ">", ">>", "<",      # 管道 / 重定向
    "&&", "||", "&",          # 命令连接控制符
    "^", "%", "(", ")",       # cmd 特有：转义 / 变量 / 分组
))


def need_shell(command: str) -> bool:
    """
    判断命令串里是否在「双引号之外」出现了 shell 元字符。
    只有真正需要 shell 解释的语法才会返回 True，减少 shell=True 攻击面。
    
    修复原实现 any(sym in command) 的漏洞：
    比如 echo "a && b" 引号内的 && 不会触发需要shell
    """
    if not command or not command.strip():
        return False

    in_quotes = False
    i = 0
    n = len(command)
    
    while i < n:
        ch = command[i]
        
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        
        if not in_quotes:
            # 先匹配双字符操作符 && ||
            if i + 1 < n and command[i:i+2] in ("&&", "||", ">>"):
                return True
            # 再匹配单字符操作符
            if ch in ("|", ">", "<", "&", "^", "%", "(", ")"):
                return True
        
        i += 1

    return False


def cmd_need_shell(cmd_parts: List[str]) -> bool:
    """对已经分词好的命令列表，直接判断是否需要 shell"""
    if not cmd_parts:
        return False
    return need_shell(" ".join(cmd_parts))


if __name__ == '__main__':
    test_cases = [
        ("dir C:\\Users", False),
        ("echo hello", False),
        ("dir | findstr .py", True),
        ("echo \"a && b\"", False),  # 引号内的&&不需要shell
        ("echo a > out.txt", True),
        ("cmd1 && cmd2", True),
        ("\"C:\\Program Files\\app.exe\"", False),
    ]
    for cmd, expected in test_cases:
        result = need_shell(cmd)
        print(f"[{result == expected}] {cmd!r} => need_shell={result}")
