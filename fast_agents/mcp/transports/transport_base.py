"""MCP 传输层抽象基类 —— TransportBase。

按仓库既有命名约定（tool_base.py 的 ToolBase、memory_store_base.py 的 MemoryStoreBase），
抽象基类一律以 Base 结尾，故原 `Transport` 更名为 `TransportBase`。

传输层只做一件事：把「一条 JSON-RPC 消息」抽象成 connect / send / recv / close
四个动作，具体怎么走（子进程管道、HTTP SSE、streamable-http）由子类决定。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TransportBase(ABC):
    """传输层抽象：收发 JSON-RPC 消息"""

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def send(self, message: Dict[str, Any]) -> None: ...
    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def close(self) -> None: ...
