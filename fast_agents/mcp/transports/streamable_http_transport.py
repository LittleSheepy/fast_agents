"""Streamable HTTP 传输 —— StreamableHTTPTransport（MCP 2025 规范单端点 HTTP）。

不依赖官方 mcp SDK，标准库 + requests（可选依赖）实现。

设计要点：
- 单端点（如 /mcp），所有请求统一 POST；响应 Content-Type 决定解析方式：
  * application/json ：直接作为一次 JSON-RPC 响应，无需后台线程。
  * text/event-stream：升级为 SSE 流，后台线程持续读取，消息入队。
- 会话：服务端在 initialize 响应头 `Mcp-Session-Id` 返回会话 ID，后续请求全部带上；
  close() 时向该端点发送 DELETE 结束会话（符合规范）。
- 断线重连：读线程在流断开时尝试 GET（Last-Event-ID 续传），多次失败后通过队列注入
  disconnected 哨兵，唤醒所有等待中的请求，与 SSETransport 行为保持一致。
- 仅在「响应为 SSE 流」时才启动读线程，JSON 直返时不创建线程，轻量高效。

依赖 requests；未安装时在 connect() 后首次 send 前抛友好错误（延迟检测，便于 import 不报错）。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from ..errors import MCPConnectionError, MCPProtocolError
from .transport_base import TransportBase


class StreamableHTTPTransport(TransportBase):
    """Streamable HTTP 传输：单端点 HTTP 连远程 MCP 服务器（MCP 2025 规范）。

    会话流程：
      1. connect() 仅校验 URL 与创建 session，不发任何请求（真正的握手由 MCPClient
         的 initialize 触发，此时才能拿到 Mcp-Session-Id）。
      2. send() 发起 POST：
         - 如果响应是 JSON，直接作为 pending 请求的响应（不启动后台读线程）。
         - 如果响应是 text/event-stream，把响应交给后台读线程持续解析，消息入队。
      3. 服务端主动推送（notifications）通过流回来，由读线程入队。
      4. close() 发送 DELETE 结束会话、等待读线程退出。
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout

        # 会话 ID，initialize 响应头返回后填入
        self._session_id: Optional[str] = None
        # 上一个 event id，用于 SSE 断线续传
        self._last_event_id: Optional[str] = None

        self._session = None
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self._closed = False
        self._reader_exited = threading.Event()
        self._reader_running = False
        self._disconnect_error: Optional[str] = None

        # 记录 POST 请求 id 到是否已通过 JSON 直返处理，避免被读线程重复分发
        self._json_resolved: set = set()
        self._resolve_lock = threading.Lock()

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self) -> None:
        """连接：只做 session 准备，真正握手由上层 initialize 触发。"""
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise MCPConnectionError("Streamable HTTP 传输需要 requests：pip install requests") from e

        import requests as req
        self._session = req.Session()
        self._session.headers.update(self.headers)
        # 声明支持两种响应：短请求返回 JSON，长任务返回 SSE
        self._session.headers["Accept"] = "application/json, text/event-stream"
        self._session.headers.setdefault("Accept-Encoding", "identity")
        self._closed = False
        self._reader_exited.set()  # 初始无读线程
        self._reader_running = False
        self._disconnect_error = None
        self._session_id = None
        self._last_event_id = None
        self._json_resolved.clear()

    def close(self) -> None:
        self._closed = True
        # 向端点发送 DELETE 终止会话（规范要求）
        if self._session is not None:
            try:
                hdrs = {}
                if self._session_id:
                    hdrs["Mcp-Session-Id"] = self._session_id
                self._session.delete(self.url, headers=hdrs, timeout=self.timeout)
            except Exception:
                pass
            try:
                self._session.close()
            except Exception:
                pass
        # 等待读线程退出
        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._session = None
        self._thread = None

    # ── 收发 ─────────────────────────────────────────────────

    def send(self, message: Dict[str, Any]) -> None:
        if not self._session:
            raise MCPConnectionError("Streamable HTTP 未连接")
        if self._disconnect_error:
            raise MCPConnectionError(self._disconnect_error)

        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        data = json.dumps(message, ensure_ascii=False)

        try:
            resp = self._session.post(
                self.url, data=data.encode("utf-8"),
                headers=headers, timeout=self.timeout, stream=True,
            )
        except Exception as e:
            raise MCPConnectionError(f"Streamable HTTP POST 请求失败：{e}") from e

        # 提取/更新会话 ID（initialize 响应会返回）
        sid = resp.headers.get("Mcp-Session-Id")
        if sid and sid != self._session_id:
            self._session_id = sid

        # 错误响应直接抛
        if resp.status_code >= 500:
            raise MCPConnectionError(
                f"Streamable HTTP POST 服务端错误：HTTP {resp.status_code} {resp.text[:200]}"
            )
        if resp.status_code == 404:
            raise MCPConnectionError(f"Streamable HTTP 端点不存在：{self.url}")
        if resp.status_code >= 400:
            raise MCPProtocolError(
                f"Streamable HTTP POST 失败：HTTP {resp.status_code} {resp.text[:200]}"
            )

        ctype = resp.headers.get("Content-Type", "").lower()
        if "text/event-stream" in ctype:
            # 长任务/通知流：交给后台读线程处理；若读线程尚未启动则启动
            self._ensure_reader(resp)
        else:
            # 短请求：直接读取整个 body 作为 JSON 响应，立即分发
            self._handle_json_response(resp, message)

    def _handle_json_response(self, resp, original_msg: Dict[str, Any]) -> None:
        """处理 application/json 响应：解析并直接入队（或回写 pending）。"""
        try:
            raw = resp.text
            resp.close()
            if not raw.strip():
                # 空响应（例如 notification 的 POST 通常无响应体）
                return
            msg = json.loads(raw)
        except Exception as e:
            raise MCPProtocolError(f"Streamable HTTP JSON 响应解析失败：{e}") from e
        # 标记该响应已被 JSON 路径处理，避免读线程重复分发（虽然此时无线程，防御性）
        req_id = msg.get("id")
        if req_id is not None:
            with self._resolve_lock:
                self._json_resolved.add(req_id)
        self._safe_enqueue(msg)

    def _ensure_reader(self, initial_resp) -> None:
        """确保后台读线程在跑；若尚未启动则启动。"""
        if self._reader_running:
            # 读线程已在，initial_resp 交给它读；因为 stream=True，resp 会在 iter_lines 中消费
            # 但若已经有读线程，这里的 initial_resp 实际是本次请求的响应，直接交给新的读线程
            # （理论上不会走到——单端点 Streamable HTTP 一次 POST 一个流，不并发多路复用）
            # 安全起见，这里直接让当前线程读 initial_resp 并继续保持原读线程
            threading.Thread(
                target=self._read_sse, args=(initial_resp,),
                name=f"sh-reader-tmp-{id(self):x}", daemon=True,
            ).start()
            return
        self._reader_exited.clear()
        self._reader_running = True
        self._thread = threading.Thread(
            target=self._read_sse, args=(initial_resp, True),
            name=f"sh-reader-{id(self):x}", daemon=True,
        )
        self._thread.start()

    def _read_sse(self, resp, may_reconnect: bool = False) -> None:
        """读 SSE 流。may_reconnect=True 时断开后自动 GET 续传（仅主读线程）。

        设计：
        - 若响应带 Content-Length（一次性事件后关闭），直接 read 完整 body 再解析，
          避免 iter_content(chunk_size=1) 在 Connection: close + Content-Length 下
          的兼容性问题。
        - 若为分块/无限流，使用 iter_content 分块读入，按 \n 切行（不使用 iter_lines，
          因为它会在 chunk 边界截断多字节 UTF-8 字符）。
        """
        err_tail = None
        try:
            while not self._closed:
                err_tail = None
                try:
                    event_type = "message"
                    data_buf: List[str] = []
                    event_id: Optional[str] = None

                    clen = resp.headers.get("Content-Length")
                    te = resp.headers.get("Transfer-Encoding", "").lower()
                    is_finite = clen is not None and "chunked" not in te
                    if is_finite:
                        # 一次性读完整个 body（走 raw，避开 stream 模式下 content 的缓冲问题）
                        try:
                            n = int(clen)
                        except ValueError:
                            n = 0
                        body = resp.raw.read(n) if n > 0 else b""
                        # 读残留
                        try:
                            extra = resp.raw.read(1024)
                            if extra:
                                body += extra
                        except Exception:
                            pass
                        lines = body.split(b"\n")
                        chunks = [(ln + b"\n") for ln in lines[:-1]] + ([lines[-1]] if lines and lines[-1] else [])
                    else:
                        chunks = resp.iter_content(chunk_size=4096)

                    buf = b""
                    for chunk in chunks:
                        if self._closed:
                            break
                        if not chunk:
                            continue
                        buf += chunk
                        while b"\n" in buf:
                            line_bytes, buf = buf.split(b"\n", 1)
                            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                            if line == "":
                                if event_id is not None:
                                    self._last_event_id = event_id
                                if data_buf:
                                    payload = "\n".join(data_buf)
                                    data_buf = []
                                    try:
                                        msg = json.loads(payload)
                                        req_id = msg.get("id")
                                        if req_id is not None:
                                            with self._resolve_lock:
                                                if req_id in self._json_resolved:
                                                    continue
                                        self._safe_enqueue(msg)
                                    except json.JSONDecodeError:
                                        pass
                                event_type = "message"
                                event_id = None
                                continue
                            if line.startswith(":"):
                                continue
                            if line.startswith("event:"):
                                event_type = line[6:].strip()
                            elif line.startswith("id:"):
                                event_id = line[3:].strip()
                            elif line.startswith("data:"):
                                data_buf.append(line[5:].lstrip(" "))
                    # buf 里剩余的不完整行（无 \n 结尾）忽略
                except Exception as e:
                    err_tail = f"Streamable HTTP 流读取异常：{e}"
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass

                if not may_reconnect or self._closed:
                    break

                # 重连：用 GET 续传
                if not self._session:
                    break
                try:
                    time.sleep(0.5)
                    headers_get = {"Accept": "text/event-stream"}
                    if self._session_id:
                        headers_get["Mcp-Session-Id"] = self._session_id
                    if self._last_event_id:
                        headers_get["Last-Event-ID"] = self._last_event_id
                    resp = self._session.get(
                        self.url, headers=headers_get,
                        stream=True, timeout=self.timeout,
                    )
                    if resp.status_code >= 400:
                        err_tail = f"Streamable HTTP GET 重连失败：HTTP {resp.status_code}"
                        break
                    sid = resp.headers.get("Mcp-Session-Id")
                    if sid:
                        self._session_id = sid
                    continue
                except Exception as e:
                    err_tail = f"Streamable HTTP GET 重连异常：{e}"
                    break
        finally:
            if may_reconnect:
                self._reader_running = False
                self._reader_exited.set()
                if not self._closed:
                    self._safe_enqueue({
                        "_event": "disconnected",
                        "error": err_tail or "Streamable HTTP 连接被服务器关闭",
                    })

    def _safe_enqueue(self, msg: Dict[str, Any]) -> None:
        """入队：队列满时丢最旧消息，永远不阻塞读线程。与 SSETransport 行为一致。"""
        if msg.get("_event") == "disconnected":
            self._disconnect_error = msg.get("error", "Streamable HTTP 连接已断开")
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

    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """接收消息。循环跳过内部事件，避免栈溢出。"""
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
                if self._disconnect_error and self._closed and not self._reader_running:
                    raise MCPConnectionError(self._disconnect_error)
                if timeout is None:
                    continue
                return None
            if not isinstance(msg, dict) or not msg.get("_event"):
                # v3.3.6 修复：响应被消费后清掉 JSON 直返登记的 id 标记，
                # 此前 _json_resolved 只增不清，长驻进程会慢速泄漏（close() 时整表 clear 兜底）。
                msg_id = msg.get("id") if isinstance(msg, dict) else None
                if msg_id is not None and ("result" in msg or "error" in msg):
                    with self._resolve_lock:
                        self._json_resolved.discard(msg_id)
                return msg
            if msg.get("_event") == "disconnected":
                raise MCPConnectionError(msg.get("error", "Streamable HTTP 连接已断开"))


def make_streamable_http_transport(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> StreamableHTTPTransport:
    """工厂函数：构造一个 StreamableHTTPTransport（命名与 make_stdio_transport 对齐）。"""
    return StreamableHTTPTransport(url=url, headers=headers, timeout=timeout)
