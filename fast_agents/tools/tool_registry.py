"""工具注册表 ToolRegistry

统一管理一组工具，提供：注册/反注册、按名查找、执行（invoke_tool）、
多工具并发执行、多 provider schema 导出、hook 回调、审批回调。
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .structured_tool import StructuredTool
from .tool_base import ToolBase, _merge_tool_input
from .tool_result import ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# 审批回调签名：(tool_name, tool_args) -> (approved: bool, modified_args: Optional[dict] = None, reason: str = "")
ApprovalCallback = Callable[[str, dict], Tuple[bool, Optional[dict], str]]


class ToolRegistry:
    def __init__(self, *, max_workers: int = 4, default_timeout: Optional[float] = 300):
        self._tools: Dict[str, ToolBase] = {}
        self._hooks: List[Callable[[str, dict, ToolResult], None]] = []
        self._approval_callback: Optional[ApprovalCallback] = None
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self._thread_local = threading.local()

    def set_approval_callback(self, callback: Optional[ApprovalCallback]) -> None:
        """设置人工审批回调。设置后，所有标记为 requires_approval=True 的工具执行前会触发回调。
        回调返回 (approved: bool, modified_args: Optional[dict], reason: str)。"""
        self._approval_callback = callback

    # ── 注册 / 反注册 ────────────────────────────────────────

    def register_tool(self, tool: Union[ToolBase, Callable]) -> ToolBase:
        """注册一个工具；裸函数会被自动包装成 StructuredTool。"""
        if callable(tool) and not isinstance(tool, ToolBase):
            tool = StructuredTool.from_function(tool)
        if not isinstance(tool, ToolBase):
            raise TypeError(f"无法注册 {type(tool)}：必须是 ToolBase 实例或可调用函数")
        if tool.name in self._tools:
            logger.warning("工具 '%s' 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        return tool

    def register_tools(self, *tools: Union[ToolBase, Callable]) -> List[ToolBase]:
        return [self.register_tool(t) for t in tools]

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ── 查询 ─────────────────────────────────────────────────

    def get_tool(self, name: str) -> Optional[ToolBase]:
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools(self) -> List[ToolBase]:
        return list(self._tools.values())

    def filter_by_tag(self, tag: str) -> List[ToolBase]:
        return [t for t in self._tools.values() if tag in t.tags]

    def get_tools_description(self) -> str:
        if not self._tools:
            return "暂无可用工具"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    # ── 单工具执行 ──────────────────────────────────────────

    def invoke_tool(
        self,
        name: str,
        tool_input: Union[str, Dict[str, Any], None] = None,
        *,
        timeout: Optional[float] = None,
        skip_approval: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """按名执行工具，统一返回 ToolResult。

        Args:
            name: 工具名。
            tool_input: JSON 字符串或 dict；支持同时用 kwargs 追加/覆盖。
            timeout: 单次调用超时（秒），None 用 default_timeout。

                ⚠️ 超时语义：超时表示「调用方放弃等待」，**不代表工具执行终止**。
                Python 线程无法被强制中断，超时返回失败后，底层 ``run()`` 仍可能
                在后台继续执行并产生副作用。对写文件 / 发请求这类有副作用的工具，
                超时后不得假定操作未发生；需要「不重复执行」保障时应结合幂等键。
                超时结果会在 ``metadata['timeout_abandoned']`` 上标记为 ``True``。
            skip_approval: 是否跳过审批流程（内部并发预审时使用）。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"未找到名为 '{name}' 的工具。可用工具：{', '.join(self.list_names())}")
        # 解析入参（与 ToolBase.invoke / ainvoke 共用同一实现，避免三处重复）
        kwargs, parse_error = _merge_tool_input(tool_input, kwargs, tool_name=name)
        if parse_error is not None:
            return ToolResult.fail(parse_error)

        # ── 审批流程 ──
        tool_meta = tool.metadata if isinstance(tool.metadata, dict) else {}
        requires_approval = tool_meta.get("requires_approval", False)
        if not skip_approval and self._approval_callback is not None and requires_approval:
            try:
                approved, modified_args, reason = self._approval_callback(name, kwargs)
                if not approved:
                    return ToolResult.fail(f"工具 '{name}' 执行被拒绝：{reason or '用户未批准'}")
                if modified_args is not None:
                    kwargs = modified_args
            except Exception as e:
                logger.exception("审批回调执行失败: %s", e)
                return ToolResult.fail(f"工具 '{name}' 审批流程异常：{e}")

        # ── 超时控制 ──
        actual_timeout = timeout if timeout is not None else self.default_timeout
        if actual_timeout is not None and actual_timeout > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.invoke, **kwargs)
                try:
                    result = future.result(timeout=actual_timeout)
                except concurrent.futures.TimeoutError:
                    result = ToolResult.fail(
                        f"工具 '{name}' 执行超时（> {actual_timeout}s）",
                        metadata={"timeout_abandoned": True},
                    )
        else:
            result = tool.invoke(**kwargs)

        for hook in self._hooks:
            try:
                hook(name, kwargs, result)
            except Exception as e:
                logger.exception("工具 hook 失败: %s", e)
        return result

    # ── 并发工具执行 ──────────────────────────────────────────

    def invoke_tools_concurrent(
        self,
        calls: List[Tuple[str, Union[str, Dict[str, Any], None]]],
        *,
        timeout: Optional[float] = None,
    ) -> List[ToolResult]:
        """并发执行多个无依赖工具，**返回结果与 calls 一一对应（顺序严格匹配）**。

        审批在主线程集中预审，被拒绝的工具直接返回失败结果、不进线程池；
        子任务**直接调用** ``tool.invoke``，不经过 ``invoke_tool`` 的超时包装，
        避免「外层并发池 + 内层超时池」线程套娃。

        Args:
            calls: [(name, tool_input), ...] 列表。同一个工具可以重复出现在 calls 里，
                每次调用独立占一个结果位，不会互相覆盖。
            timeout: 整体超时（秒），None 用注册表默认值。

                ⚠️ 超时语义：整体超时表示「调用方放弃等待剩余未完成的子任务」，
                **不代表这些子任务的执行被终止**。Python 线程无法被强制中断，
                超时返回后底层 ``run()`` 仍可能在后台继续执行并产生副作用。
                未完成的子任务结果会在 ``metadata['timeout_abandoned']`` 上
                标记为 ``True``。

        Returns:
            ``List[ToolResult]``，**长度与 calls 严格相等**，第 i 个结果对应 calls[i]。
        """
        if not calls:
            return []
        actual_timeout = timeout if timeout is not None else self.default_timeout
        n = len(calls)
        results: List[Optional[ToolResult]] = [None] * n

        # ── 主线程集中预审：解析入参 + 审批，被拒绝的直接返回失败，不进线程池 ──
        def _parse_args(name, inp, idx):
            tool = self._tools.get(name)
            if tool is None:
                return None, ToolResult.fail(
                    f"未找到名为 '{name}' 的工具。可用工具：{', '.join(self.list_names())}"
                )
            kwargs, parse_error = _merge_tool_input(inp, {}, tool_name=name)
            if parse_error is not None:
                return None, ToolResult.fail(parse_error)
            return (tool, kwargs), None

        pending: List[Tuple[int, ToolBase, dict]] = []  # (索引, 工具, kwargs)
        for i, (name, inp) in enumerate(calls):
            parsed, err = _parse_args(name, inp, i)
            if err is not None:
                results[i] = err
                continue
            tool, kwargs = parsed
            # 审批（若需要）
            tool_meta = tool.metadata if isinstance(tool.metadata, dict) else {}
            requires_approval = tool_meta.get("requires_approval", False)
            if self._approval_callback is not None and requires_approval:
                try:
                    approved, modified_args, reason = self._approval_callback(name, kwargs)
                    if not approved:
                        results[i] = ToolResult.fail(
                            f"工具 '{name}' 执行被拒绝：{reason or '用户未批准'}"
                        )
                        continue
                    if modified_args is not None:
                        kwargs = modified_args
                except Exception as e:
                    logger.exception("审批回调执行失败: %s", e)
                    results[i] = ToolResult.fail(f"工具 '{name}' 审批流程异常：{e}")
                    continue
            pending.append((i, tool, kwargs))

        if not pending:
            return results  # type: ignore[return-value]

        # ── 子线程：直接调用 tool.invoke，不走 invoke_tool 的超时包装 ──
        def _run_one(idx: int, tool: ToolBase, kwargs: dict) -> Tuple[int, ToolResult]:
            try:
                r = tool.invoke(**kwargs)
            except Exception as e:
                r = ToolResult.fail(f"工具 '{tool.name}' 并发执行失败：{e}")
            # hooks（每个结果触发一次）
            for hook in self._hooks:
                try:
                    hook(tool.name, kwargs, r)
                except Exception as e:
                    logger.exception("工具 hook 失败: %s", e)
            return idx, r

        max_workers = min(self.max_workers, len(pending))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {executor.submit(_run_one, i, t, k): i for i, t, k in pending}
            try:
                for future in concurrent.futures.as_completed(futures, timeout=actual_timeout):
                    try:
                        i, r = future.result()
                        results[i] = r
                    except Exception as e:
                        i = futures[future]
                        results[i] = ToolResult.fail(f"并发执行内部异常：{e}")
            except concurrent.futures.TimeoutError:
                for future, i in futures.items():
                    if results[i] is None:
                        future.cancel()
                        # 用 name 生成错误信息
                        name = calls[i][0]
                        results[i] = ToolResult.fail(
                            f"工具 '{name}' 执行超时（整体超时 {actual_timeout}s）",
                            metadata={"timeout_abandoned": True},
                        )
        finally:
            # 关键：wait=False → 超时立即返回，不在 __exit__ 处等所有子线程跑完
            executor.shutdown(wait=False, cancel_futures=True)

        return results  # type: ignore[return-value]

    # ── Schema 导出（多 provider）────────────────────────────

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def to_gemini_tools(self) -> List[Dict[str, Any]]:
        return [t.to_gemini_tool() for t in self._tools.values()]

    # ── Hook ─────────────────────────────────────────────────

    def add_hook(self, hook: Callable[[str, dict, ToolResult], None]) -> None:
        self._hooks.append(hook)

    # ── 杂项 ─────────────────────────────────────────────────

    def clear(self) -> None:
        self._tools.clear()
        self._hooks.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self)}>"

# 全局工具注册表
global_registry = ToolRegistry()
