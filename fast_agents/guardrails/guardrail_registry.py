"""全局护栏原型注册表 GuardrailRegistry

把原先定义在 guardrail_base.py 里的模块级裸 dict（_GLOBAL_GUARDRAIL_REGISTRY）
抽取为独立注册表对象，风格对齐 tools/tool_registry.py 的 ToolRegistry：
统一提供 注册 / 反注册 / 按名查找 / 排序列举 / 批量更新 等能力。

⚠️ 隔离语义（对齐路线图 A34「global_registry 隔离」）：
  这里存的是护栏「原型」（类级别默认实例或 @guardrail_creator 产生的样板实例），
  仅供发现（list_guardrails）/查找（get_guardrail）/作为 add_guardrail 的克隆源。
  GuardrailMiddleware 在 add_guardrail / _load_presets 时会对原型做 deepcopy，
  产生本实例独占的副本；enable/disable/remove/配置修改都只改副本，绝不回写原型。
  因此多 Agent 之间的护栏状态完全隔离，不会出现"A 禁用 B 也失效"的问题。

  顶层 run_input_guardrails / run_output_guardrails / run_tool_call_guardrails
  在 guardrails=None 时不再隐式扫全局表，直接返回空结果——不指定就不执行，
  避免独立调用时意外跑了别的模块/测试注册的护栏。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

if TYPE_CHECKING:  # 仅为类型注解，避免与 guardrail_base 形成循环导入
    from .guardrail_base import GuardrailBase

logger = logging.getLogger(__name__)


class GuardrailRegistry:
    """护栏原型注册表（用法风格对齐 ToolRegistry）。

    对外提供语义化方法（register_guardrail / get_guardrail / ...）；
    同时兼容历史 `_GLOBAL_GUARDRAIL_REGISTRY` 裸 dict 的用法（`in` / `.get()` /
    `.values()` / `.clear()` / `.update()` / `.copy()`），方便存量代码平滑迁移。
    """

    def __init__(self) -> None:
        self._guardrails: Dict[str, "GuardrailBase"] = {}

    # ── 注册 / 反注册 ────────────────────────────────────────

    def register_guardrail(self, guardrail: "GuardrailBase") -> "GuardrailBase":
        """登记一个护栏原型；同名时打印 warning 后覆盖（对齐 ToolRegistry.register_tool）。"""
        name = getattr(guardrail, "name", None)
        if not name:
            raise ValueError(f"无法注册 {guardrail!r}：护栏必须有 name")
        if name in self._guardrails:
            logger.warning("[guardrail] 覆盖已注册的同名护栏原型: %s", name)
        self._guardrails[name] = guardrail
        return guardrail

    def unregister_guardrail(self, guardrail_or_name: Union[str, "GuardrailBase"]) -> bool:
        """按实例或名称移除原型；不存在时返回 False（不抛 KeyError）。"""
        name = (
            guardrail_or_name
            if isinstance(guardrail_or_name, str)
            else getattr(guardrail_or_name, "name", None)
        )
        if name and name in self._guardrails:
            del self._guardrails[name]
            return True
        return False

    # ── 查询 ─────────────────────────────────────────────────

    def list_guardrails(self) -> List["GuardrailBase"]:
        """按 priority 升序返回全部原型（数字越小优先级越高，越靠前）。"""
        return sorted(self._guardrails.values(), key=lambda g: g.priority)

    def get_guardrail(self, name: str) -> Optional["GuardrailBase"]:
        """按名查找原型；不存在返回 None（不抛 KeyError）。"""
        return self._guardrails.get(name)

    def is_registered(self, name: str) -> bool:
        """判断某名称是否已登记（等价于 `name in registry`）。"""
        return name in self._guardrails

    # ── 批量 / 清空 ──────────────────────────────────────────

    def clear_guardrails(self) -> None:
        """清空全部原型（测试隔离常用）。"""
        self._guardrails.clear()

    def update_guardrails(self, guardrails: Dict[str, "GuardrailBase"]) -> None:
        """用一份 {name: 原型} 映射批量覆盖/新增。"""
        self._guardrails.update(guardrails)

    def extend_guardrails(self, guardrails: List["GuardrailBase"]) -> None:
        """批量登记一组原型。"""
        for guardrail in guardrails:
            self.register_guardrail(guardrail)

    # ── dict-like 兼容层（历史 `_GLOBAL_GUARDRAIL_REGISTRY` 是裸 dict）──

    def __contains__(self, name: object) -> bool:
        return name in self._guardrails

    def __len__(self) -> int:
        return len(self._guardrails)

    def __iter__(self) -> Iterator[str]:
        return iter(self._guardrails)

    def __getitem__(self, name: str) -> "GuardrailBase":
        return self._guardrails[name]

    def __setitem__(self, name: str, guardrail: "GuardrailBase") -> None:
        self._guardrails[name] = guardrail

    def __delitem__(self, name: str) -> None:
        del self._guardrails[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._guardrails.get(name, default)

    def keys(self):
        return self._guardrails.keys()

    def values(self):
        return self._guardrails.values()

    def items(self):
        return self._guardrails.items()

    def clear(self) -> None:
        self._guardrails.clear()

    def update(self, other) -> None:
        self._guardrails.update(other)

    def copy(self) -> Dict[str, "GuardrailBase"]:
        """返回一份浅拷贝快照（dict），兼容历史 `.copy()` 用法。"""
        return dict(self._guardrails)


# 全局护栏原型注册表（模块级单例）
global_guardrail_registry: GuardrailRegistry = GuardrailRegistry()
