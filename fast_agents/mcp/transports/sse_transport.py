"""SSE 传输 —— SSETransport（HTTP Server-Sent Events）。

不依赖官方 mcp SDK，标准库 + requests（可选依赖）实现。

v3.1 稳定性升级（对外 API 完全兼容）：
- SSETransport：修复 recv() 对 _event 递归调用的栈溢出风险，改为循环取消息；
  队列满时丢最旧消息而非阻塞读线程；SSE 流异常断开时向队列注入哨兵并唤醒，
  避免上层无限等待超时；POST 错误按 HTTP 状态区分 ProtocolError/ConnectionError；
  close() 等待读线程退出。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from ..errors import MCPConnectionError, MCPProtocolError
from .transport_base import TransportBase


class SSETransport(TransportBase):
    """SSE 传输：通过 HTTP Server-Sent Events 连远程 MCP 服务器。

    依赖 requests；未安装时在 connect 时报友好错误。
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._post_url: Optional[str] = None
        self._session = None
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self._closed = False
        self._reader_exited = threading.Event()
        self._disconnect_error: Optional[str] = None

    def connect(self) -> None:
        try:
            import requests
        except ImportError as e:
            raise MCPConnectionError("SSE 传输需要 requests：pip install requests") from e
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        # 禁用自动解压，避免 urllib3 内部缓冲导致 SSE 事件延迟
        self._session.headers.setdefault("Accept-Encoding", "identity")
        self._closed = False
        self._reader_exited.clear()
        self._disconnect_error = None
        try:
            resp = self._session.get(self.url, stream=True, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            self._session.close()
            self._session = None
            raise MCPConnectionError(f"SSE 连接失败：{e}") from e
        self._thread = threading.Thread(
            target=self._read_sse, args=(resp,), name=f"sse-reader-{id(self):x}", daemon=True,
        )
        self._thread.start()
        # 等到 endpoint 事件（循环跳过无关内部事件；超时则报错）
        deadline = time.time() + self.timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                self.close()
                raise MCPProtocolError("SSE 握手失败：等待 endpoint 事件超时")
            try:
                ev = self._queue.get(timeout=min(remain, 0.5))
            except queue.Empty:
                if self._disconnect_error:
                    self.close()
                    raise MCPConnectionError(self._disconnect_error)
                continue
            if isinstance(ev, dict) and ev.get("_event") == "endpoint":
                self._post_url = ev["endpoint"]
                if not self._post_url.startswith("http"):
                    from urllib.parse import urljoin
                    self._post_url = urljoin(self.url, self._post_url)
                return
            if isinstance(ev, dict) and ev.get("_event") == "disconnected":
                self.close()
                raise MCPConnectionError(ev.get("error", "SSE 连接已断开"))
            # 其他内部事件/普通消息握手阶段不应出现，忽略即可

    def _read_sse(self, resp) -> None:
        event_type = "message"
        data_buf: List[str] = []
        err_tail = None
        endpoint_pending: Optional[str] = None
        try:
            # chunk_size=1 避免 urllib3/requests 内部缓冲延迟，SSE 事件要即时拿到
            for raw in resp.iter_lines(decode_unicode=True, chunk_size=1):
                if self._closed:
                    break
                if raw is None:
                    continue
                line = raw.rstrip("\r")
                if line == "":
                    # 一个事件结束
                    if endpoint_pending is not None:
                        self._safe_enqueue({"_event": "endpoint", "endpoint": endpoint_pending})
                        endpoint_pending = None
                    if data_buf:
                        payload = "\n".join(data_buf)
                        data_buf = []
                        try:
                            msg = json.loads(payload)
                            self._safe_enqueue(msg)
                        except json.JSONDecodeError:
                            pass
                    event_type = "message"
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].lstrip(" ")
                    if event_type == "endpoint":
                        # endpoint 事件：等空行结束时再投递，避免半行
                        endpoint_pending = data
                    else:
                        data_buf.append(data)
                # 其他字段忽略
        except Exception as e:
            err_tail = f"SSE 流读取异常：{e}"
        finally:
            try:
                resp.close()
            except Exception:
                pass
            self._reader_exited.set()
            if not self._closed:
                # 流提前结束：向队列塞 disconnected 哨兵
                self._safe_enqueue({
                    "_event": "disconnected",
                    "error": err_tail or "SSE 连接被服务器关闭",
                })

    def _safe_enqueue(self, msg: Dict[str, Any]) -> None:
        """入队：队列满时丢最旧消息，永远不阻塞读线程。"""
        if msg.get("_event") == "disconnected":
            self._disconnect_error = msg.get("error", "SSE 连接已断开")
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                pass

    def send(self, message: Dict[str, Any]) -> None:
        if not self._session or not self._post_url:
            raise MCPConnectionError("SSE 未连接")
        if self._disconnect_error:
            raise MCPConnectionError(self._disconnect_error)
        data = json.dumps(message, ensure_ascii=False)
        try:
            resp = self._session.post(
                self._post_url, data=data.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except Exception as e:
            raise MCPConnectionError(f"SSE POST 请求失败：{e}") from e
        if resp.status_code >= 500:
            raise MCPConnectionError(f"SSE POST 服务端错误：HTTP {resp.status_code} {resp.text[:200]}")
        if resp.status_code >= 400:
            raise MCPProtocolError(f"SSE POST 失败：HTTP {resp.status_code} {resp.text[:200]}")

    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """接收消息。循环跳过 _event 内部事件，避免递归导致栈溢出。"""
        if self._disconnect_error and self._queue.empty() and self._closed:
            raise MCPConnectionError(self._disconnect_error)
        deadline = time.time() + (timeout if timeout is not None else float("inf"))
        while True:
            remain = deadline - time.time()
            if timeout is not None and remain <= 0:
                return None
            try:
                msg = self._queue.get(timeout=min(remain, 0.5) if timeout is not None else 0.5)
            except queue.Empty:
                if self._disconnect_error and self._closed:
                    raise MCPConnectionError(self._disconnect_error)
                if timeout is None:
                    continue
                return None
            if not isinstance(msg, dict) or not msg.get("_event"):
                return msg
            if msg.get("_event") == "disconnected":
                raise MCPConnectionError(msg.get("error", "SSE 连接已断开"))
            # 其他内部事件（endpoint 等）继续取下一条

    def close(self) -> None:
        self._closed = True
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
        # 等待读线程退出
        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._session = None
