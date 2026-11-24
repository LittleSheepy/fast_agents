"""基础权限控制护栏 BasicPermissionGuardrail（L9 单一事实源对齐版）

设计要点（对齐路线图 A9 审批 / C6 沙箱的「单一事实源」约定）：
- 危险工具判定**优先读取 ToolRegistry 元数据 ``requires_approval=True``**（这是
  工具模块、C6 沙箱脚本执行已经统一使用的标记位，如 file_tool 的写入/删除、
  terminal_tool 的 allow_exec=True 时），避免护栏自造一份关键词表和两边冲突；
- 内置一份「精确工具名」危险工具清单（而非子串匹配），解决 ``execute_query``
  被 ``execute`` 误伤、``remove_duplicate`` 被 ``remove`` 误伤的误判问题；
- 仍保留关键词表作为**兜底**（仅在没有 tool_registry 传入时启用子串匹配，
  且全部对 tool_name 做词边界/分隔符分割后精确比较，不再无脑 ``keyword in name``）；
- 敏感路径参数扫描保留，但可被显式关闭；扫描会**逐层展开**嵌套参数（B2 修复），
  不再只看顶层 str 值。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..tool_call_guardrail import ToolCallGuardrail
from ..models import GuardrailResult, ToolCallInfo


# 默认高危工具名（**精确匹配**，非子串）。只有完全等于列表中名字的工具才会被拦截。
# 这样 execute_query / remove_duplicate / execute_business_logic 等正常工具不会被误伤。
DEFAULT_DANGEROUS_TOOL_NAMES: Set[str] = {
    # 系统命令执行类
    "exec", "execute", "shell", "shell_exec", "exec_command", "execute_command",
    "bash", "command", "cmd", "powershell", "run_command", "run_shell",
    "run_bash", "run_cmd", "terminal",
    "system_exec", "os_system", "subprocess_run",
    # 删除/格式化类
    "delete_file", "remove_file", "delete_dir", "remove_dir",
    "rm", "rmdir", "del", "erase",
    "format_disk", "fdisk", "mkfs",
    # 权限提升类
    "chmod", "chown", "sudo", "su",
}

# 兜底关键词（仅在没有传入 tool_registry 时启用，做词边界分割后比较）。
# 用下划线/连字符/驼峰拆词后再比，避免 "execute" 命中 "execute_query"。
_DEFAULT_FALLBACK_KEYWORDS: Set[str] = DEFAULT_DANGEROUS_TOOL_NAMES

# 默认敏感路径关键词（参数里扫描）
DEFAULT_SENSITIVE_PATHS = {
    "/etc/passwd", "/etc/shadow", "/root/.ssh",
    "~/.ssh", "id_rsa", "id_dsa",
    "/proc", "/sys",
    "C:\\Windows\\System32",
}


def _split_tool_name(name: str) -> Set[str]:
    """把工具名按下划线/连字符/驼峰拆成词集合，用于兜底关键词比较。

    例: ``execute_query`` → {"execute", "query"}；``runShellCommand`` → {"run", "shell", "command"}。
    """
    # 先把驼峰拆成下划线分隔
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # 再按下划线/连字符拆
    parts = re.split(r"[_\-]+", s2)
    return {p.lower() for p in parts if p}


# 参数展开的容器类型（set / frozenset 无下标顺序，仅用于定位路径不稳定，不影响拦截判定）
_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)


def _iter_arg_strings(value: Any, prefix: str = "") -> Iterator[Tuple[str, str]]:
    """展开工具参数为 ``(定位路径, 字符串内容)`` 二元组序列（B2 修复）。

    修复前 ``before_call`` 只对 ``tool_args.items()`` 的顶层 str 值做敏感路径匹配，
    于是把路径藏进一层嵌套就完全绕过，探针实测全部静默放行：
    ``{"payload": {"path": "/etc/passwd"}}``、``{"paths": ["/etc/shadow"]}``、
    ``{"a": {"b": [{"c": "/proc/self/environ"}]}}``。

    展开规则：
    - dict：**键与值都扫**（``{"files": {"/etc/passwd": "r"}}`` 这类路径写在键上的同样要拦），
      定位路径用 ``父.子``；
    - list / tuple / set / frozenset：逐元素扫，定位路径带下标 ``paths[0]``；
    - str 与 ``os.PathLike``：作为字符串产出（文件类工具传 Path 对象很常见，
      ``skills/loader.py`` 的同类检查也显式接受 ``(str, Path)``）；
    - 其余标量（int / float / bool / None）跳过，与修复前一致；
    - 自引用参数按「当前分支已访问容器 id」剪枝，不会死循环。

    两个刻意的设计取舍：
    1. **用显式栈而非递归**：护栏自己报 RecursionError 会按 fail_open/fail_closed
       把业务打成放行或全部拦截，观测/检查代码绝不能把主流程打挂；
    2. **不设深度上限**：深度上限等于新递一个「藏到第 N+1 层就过关」的绕过口，
       而栈遍历本身没有栈深风险，所以不限深度。

    Args:
        value: 参数根对象（正常是 dict，但非 dict 也能安全展开，不再因 ``.items()`` 报错）。
        prefix: 定位路径前缀，顶层传空串使**顶层参数的 arg_name 与修复前完全一致**
            （直接是参数名），既有 ``metadata["arg_name"]`` 契约不变。

    Yields:
        ``(arg_name, text)``：按先序遍历产出，与修复前的顶层扫描顺序一致，
        多个命中时报告的仍是第一个。
    """
    # 栈元素：(待展值, 定位路径, 当前分支已访问的容器 id 元组)
    stack: List[Tuple[Any, str, Tuple[int, ...]]] = [(value, prefix, ())]
    while stack:
        current, loc, branch_ids = stack.pop()
        if isinstance(current, str):
            yield loc, current
            continue
        if isinstance(current, os.PathLike):
            try:
                yield loc, os.fspath(current)
            except Exception:
                # 异常的 PathLike 实现不参与判定，也不能把主流程打挂
                pass
            continue
        next_ids = branch_ids
        if isinstance(current, _CONTAINER_TYPES):
            container_id = id(current)
            if container_id in branch_ids:
                continue  # 自引用剪枝
            # 只禁止「同一条分支上重复出现」，不影响兄弟分支合法引用同一对象
            next_ids = branch_ids + (container_id,)
        children: List[Tuple[Any, str, Tuple[int, ...]]] = []
        if isinstance(current, dict):
            for key, val in current.items():
                child_loc = f"{loc}.{key}" if loc else str(key)
                if isinstance(key, str):
                    children.append((key, child_loc, next_ids))
                children.append((val, child_loc, next_ids))
        elif isinstance(current, _CONTAINER_TYPES):
            for index, item in enumerate(current):
                children.append((item, f"{loc}[{index}]", next_ids))
        # 倒序入栈使弹出顺序等于原始顺序（保持先序，与修复前的顶层扫描行为一致）
        stack.extend(reversed(children))


class BasicPermissionGuardrail(ToolCallGuardrail):
    """基础工具权限护栏，拦截未授权的高危工具调用。

    判定顺序：
    1. 白名单 ``allowed_dangerous_tools`` 直接放行；
    2. 如果传入了 ``tool_registry``：读取该工具的 ``metadata["requires_approval"]``，
       为 True 则拦截（对齐 A9 / C6 单一事实源）；
    3. 否则用内置**精确**危险工具名清单匹配；
    4. 兜底关键词（拆词后精确比较，仅在 ``fallback_keyword_match=True`` 时启用，默认关闭）；
    5. 参数里扫描敏感路径（逐层展开嵌套 dict / list / tuple / set、字典键与 Path 对象，B2）。

    高优先级（priority=15），在所有工具调用前最先执行。
    """

    name = "basic_permission"
    description = "基础工具权限控制，优先读取 ToolRegistry.requires_approval 单一事实源，拦截高危系统命令、敏感文件访问等危险操作"
    priority = 15

    def __init__(
        self,
        blocked_tool_names: Optional[Set[str]] = None,
        sensitive_paths: Optional[Set[str]] = None,
        allowed_dangerous_tools: Optional[Set[str]] = None,
        tool_registry: Any = None,
        fallback_keyword_match: bool = False,
        check_sensitive_paths: bool = True,
        **_guardrail_kwargs,
    ):
        """
        Args:
            blocked_tool_names: 额外要精确拦截的工具名集合（完全等于名字才拦）。
            sensitive_paths: 敏感路径关键词集合。
            allowed_dangerous_tools: 白名单工具名（即便出现在危险清单 / requires_approval=True 也放行）。
            tool_registry: ToolRegistry 实例；传入时优先读 metadata["requires_approval"]
                单一事实源，不再依赖关键词猜测。GuardrailMiddleware 会自动注入。
            fallback_keyword_match: 是否启用拆词兜底关键词匹配（默认 False，避免误判
                execute_query / remove_duplicate 这类正常工具）。
            check_sensitive_paths: 是否扫描工具参数里的敏感路径（默认 True）。
        """
        super().__init__(**_guardrail_kwargs)
        # B1 修复：改用 `is not None` 判断，避免空集合被 `or` 运算符替换为默认值
        self.blocked_names: Set[str] = set(n.lower() for n in (blocked_tool_names if blocked_tool_names is not None else DEFAULT_DANGEROUS_TOOL_NAMES))
        self.sensitive_paths: Set[str] = set(sensitive_paths if sensitive_paths is not None else DEFAULT_SENSITIVE_PATHS)
        self.allowed_tools: Set[str] = set(n.lower() for n in (allowed_dangerous_tools if allowed_dangerous_tools is not None else set()))
        self.tool_registry = tool_registry
        self.fallback_keyword_match = fallback_keyword_match
        self.check_sensitive_paths = check_sensitive_paths

    def _lookup_tool_meta(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """从 tool_registry 读取工具 metadata；拿不到时返回 None。"""
        if self.tool_registry is None:
            return None
        try:
            tool = None
            # 兼容 ToolRegistry 常见的几种获取接口
            for attr in ("get_tool", "get", "_tools"):
                obj = getattr(self.tool_registry, attr, None)
                if obj is None:
                    continue
                if callable(obj) and attr != "_tools":
                    try:
                        tool = obj(tool_name)
                    except Exception:
                        tool = None
                    if tool is not None:
                        break
                elif isinstance(obj, dict):
                    tool = obj.get(tool_name)
                    if tool is not None:
                        break
            if tool is None:
                return None
            # ToolBase 实例或函数包装；取 metadata
            meta = getattr(tool, "metadata", None)
            if isinstance(meta, dict):
                return meta
            return None
        except Exception:
            return None

    def before_call(self, tool_call: ToolCallInfo, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        tool_name = tool_call.tool_name
        tool_name_lower = tool_name.lower()

        # 1) 白名单工具直接放行
        if tool_name_lower in self.allowed_tools:
            return GuardrailResult.allow(metadata={"match": "whitelist"})

        # 2) 优先：读 ToolRegistry 的 requires_approval（单一事实源，对齐 A9/C6）
        #    requires_approval=True 一旦命中直接拦截，不再走后续步骤。
        tool_meta = self._lookup_tool_meta(tool_name)
        if tool_meta is not None and tool_meta.get("requires_approval", False):
            return GuardrailResult.reject(
                f"工具 '{tool_name}' 被标记 requires_approval=True，默认被权限护栏拦截（如需执行请走人工审批或将其加入 allowed_dangerous_tools）",
                metadata={
                    "risk_type": "requires_approval",
                    "source": "tool_registry_metadata",
                },
            )

        # 3) Bug4 修复：blocked_names 精确黑名单提升为**无条件兜底**。
        #    此前它仅在 else 分支（tool_meta is None，即 registry 里查不到该工具）时生效，
        #    形成安全死角：工具已注册到 registry 但未打 requires_approval=True 时，
        #    tool_meta is not None 进入 if 分支 → requires_approval=False → 不拦截，
        #    直接跳过 else 里的 blocked_names，delete_file / shell_exec 等高危工具被静默放行。
        #    修复后无论有无 registry，工具名命中精确黑名单就拦截。
        #    白名单（步骤 1）和 requires_approval（步骤 2）已各自短路，
        #    本步骤不改变单一事实源的优先级语义。
        if tool_name_lower in self.blocked_names:
            return GuardrailResult.reject(
                f"工具 '{tool_call.tool_name}' 属于高危操作（精确名匹配），默认被权限护栏拦截",
                metadata={"risk_type": "dangerous_tool", "match": "exact_name"},
            )

        # 4) 可选兜底：拆词后关键词匹配（仅 registry 里查不到该工具时启用，默认关闭）。
        #    保留原设计意图：有 tool_registry 信息时不靠拆词猜测，避免把正式注册的业务
        #    工具误伤；只有完全没有任何 registry 线索的场景，才降级使用关键词启发式。
        if tool_meta is None and self.fallback_keyword_match:
            words = _split_tool_name(tool_name)
            hit = _DEFAULT_FALLBACK_KEYWORDS & words
            if hit:
                return GuardrailResult.reject(
                    f"工具 '{tool_call.tool_name}' 包含高危关键词 {sorted(hit)}，默认被权限护栏拦截",
                    metadata={"risk_type": "dangerous_tool", "match": "fallback_keyword", "matched": sorted(hit)},
                )

        # 5) 检查参数中的敏感路径（B2 修复：不再只看顶层 str 值，嵌套参数同样扫到；
        #    顶层命中的 reason 文案与 metadata 字段与修复前逐字节一致）
        if self.check_sensitive_paths:
            for arg_name, arg_text in _iter_arg_strings(tool_call.tool_args):
                arg_lower = arg_text.lower()
                for path in self.sensitive_paths:
                    if path.lower() in arg_lower:
                        return GuardrailResult.reject(
                            f"工具参数中包含敏感路径访问：{path}",
                            metadata={"risk_type": "sensitive_path", "path": path, "arg_name": arg_name},
                        )

        return GuardrailResult.allow()

    def clone(self) -> "BasicPermissionGuardrail":
        """克隆时完整复制业务字段 + 基类字段（S3 修复）。"""
        return BasicPermissionGuardrail(
            blocked_tool_names=set(self.blocked_names),
            sensitive_paths=set(self.sensitive_paths),
            allowed_dangerous_tools=set(self.allowed_tools),
            tool_registry=self.tool_registry,
            fallback_keyword_match=self.fallback_keyword_match,
            check_sensitive_paths=self.check_sensitive_paths,
            name=self.name,
            description=self.description,
            priority=self.priority,
            enabled=self.enabled,
            fail_open=self.fail_open if getattr(self, "_fail_open_explicit", False) else None,
            tags=list(self.tags) if self.tags else None,
            metadata=dict(self.metadata) if self.metadata else None,
        )
