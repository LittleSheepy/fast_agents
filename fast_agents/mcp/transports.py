"""MCP 传输层 —— 基于 JSON-RPC 2.0 over stdio / SSE。

不依赖官方 mcp SDK，标准库实现：StdioTransport（子进程）+ SSETransport（HTTP）。
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .errors import MCPConnectionError, MCPProtocolError

# Windows 平台防黑框常量
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class Transport(ABC):
    """传输层抽象：收发 JSON-RPC 消息"""

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def send(self, message: Dict[str, Any]) -> None: ...
    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def close(self) -> None: ...


class StdioTransport(Transport):
    """stdio 传输：启动一个子进程，通过 stdin/stdout 进行 JSON-RPC 通信。

    Windows 专项：CREATE_NO_WINDOW 防黑框；强制 PYTHONIOENCODING=utf-8。
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.command = command
        self.args = args or []
        self.cwd = cwd
        merged_env = dict(os.environ)
        merged_env["PYTHONIOENCODING"] = "utf-8"
        merged_env["PYTHONUTF8"] = "1"
        if env:
            merged_env.update(env)
        self.env = merged_env
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self._closed = False

    def connect(self) -> None:
        cmd = [self.command, *self.args]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
                bufsize=0,
                creationflags=_CREATE_NO_WINDOW,
            )
        except FileNotFoundError as e:
            raise MCPConnectionError(f"找不到命令：{self.command}（请确认已安装）") from e
        except Exception as e:
            raise MCPConnectionError(f"启动子进程失败：{e}") from e
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        buf = b""
        while not self._closed:
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                self._queue.put(msg)

    def send(self, message: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPConnectionError("传输未连接")
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


class SSETransport(Transport):
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

    def connect(self) -> None:
        try:
            import requests
        except ImportError as e:
            raise MCPConnectionError("SSE 传输需要 requests：pip install requests") from e
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        resp = self._session.get(self.url, stream=True, timeout=self.timeout)
        resp.raise_for_status()
        self._thread = threading.Thread(target=self._read_sse, args=(resp,), daemon=True)
        self._thread.start()
        # 等到 endpoint 事件
        ev = self._queue.get(timeout=self.timeout)
        if isinstance(ev, dict) and ev.get("_event") == "endpoint":
            self._post_url = ev["endpoint"]
            if not self._post_url.startswith("http"):
                from urllib.parse import urljoin
                self._post_url = urljoin(self.url, self._post_url)
        else:
            raise MCPProtocolError("SSE 握手失败：未收到 endpoint 事件")

    def _read_sse(self, resp) -> None:
        event_type = "message"
        data_buf = []
        for raw in resp.iter_lines(decode_unicode=True):
            if self._closed:
                break
            if raw is None:
                continue
            line = raw.rstrip("\r")
            if line == "":
                if data_buf:
                    payload = "\n".join(data_buf)
                    data_buf = []
                    try:
                        msg = json.loads(payload)
                        self._queue.put(msg)
                    except json.JSONDecodeError:
                        pass
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].lstrip(" ")
                if event_type == "endpoint":
                    self._queue.put({"_event": "endpoint", "endpoint": data})
                else:
                    data_buf.append(data)
            # 其他字段忽略

    def send(self, message: Dict[str, Any]) -> None:
        if not self._session or not self._post_url:
            raise MCPConnectionError("SSE 未连接")
        data = json.dumps(message, ensure_ascii=False)
        resp = self._session.post(
            self._post_url, data=data.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise MCPProtocolError(f"SSE POST 失败：HTTP {resp.status_code} {resp.text[:200]}")

    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        try:
            msg = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(msg, dict) and msg.get("_event"):
            return self.recv(timeout=timeout)
        return msg

    def close(self) -> None:
        self._closed = True
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass


def make_stdio(command_str: str, cwd: Optional[str] = None) -> StdioTransport:
    """便捷工厂：把 'python server.py --arg x' 拆成命令+参数，Windows 用 posix=False。"""
    parts = shlex.split(command_str, posix=(sys.platform != "win32"))
    if not parts:
        raise MCPConnectionError("命令不能为空")
    return StdioTransport(command=parts[0], args=parts[1:], cwd=cwd)
