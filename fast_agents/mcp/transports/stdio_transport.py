"""stdio 传输 —— StdioTransport（子进程管道）+ make_stdio_transport 工厂。

不依赖官方 mcp SDK，标准库实现。

v3.1 稳定性升级（对外 API 完全兼容）：
- StdioTransport：新增 stderr 后台读线程（避免子进程 stderr 缓冲区写满卡死），
  提供 get_last_stderr() 用于错误排查；子进程异常退出时抛 MCPConnectionError；
  send() 检测 stdin 已关闭场景；close() 等待读线程优雅退出、kill 后 wait() 防止僵尸。
"""

from __future__ import annotations

import collections
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional

from ..errors import MCPConnectionError
from .transport_base import TransportBase

# Windows 平台防黑框常量
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# 保留最近 N 行 stderr 用于错误排查
_STDERR_RING_MAX = 200


class StdioTransport(TransportBase):
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
        self._stderr_reader: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=256)
        self._stderr_buf: Deque[str] = collections.deque(maxlen=_STDERR_RING_MAX)
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._reader_exited = threading.Event()

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
        self._closed = False
        self._reader_exited.clear()
        self._reader = threading.Thread(
            target=self._read_stdout_loop, name=f"stdio-stdout-{id(self):x}", daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr_loop, name=f"stdio-stderr-{id(self):x}", daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()

    def _read_stdout_loop(self) -> None:
        assert self._proc and self._proc.stdout
        buf = b""
        try:
            while not self._closed:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break  # stdout 关闭，子进程退出
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
                    self._safe_enqueue(msg)
        except Exception:
            pass  # 读异常交给外层检查进程退出码
        finally:
            # 子进程 stdout 读完了，检查是否异常退出
            self._reader_exited.set()
            if not self._closed and self._proc is not None:
                rc = self._proc.poll()
                if rc is not None and rc != 0:
                    err_tail = self.get_last_stderr()
                    detail = f"子进程意外退出（exit={rc}）。stderr 尾部：\n{err_tail}" if err_tail \
                        else f"子进程意外退出（exit={rc}），无 stderr 输出"
                    self._safe_enqueue({"_event": "disconnected", "error": detail})

    def _read_stderr_loop(self) -> None:
        """后台读 stderr，避免管道写满阻塞子进程；最近若干行保存在 ring buffer 里。"""
        assert self._proc and self._proc.stderr
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    line = repr(raw)
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_buf.append(line)
        except Exception:
            pass

    def _safe_enqueue(self, msg: Dict[str, Any]) -> None:
        """入队：队列满时丢最旧消息，永远不阻塞读线程。"""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢最旧
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                pass

    def get_last_stderr(self, n: int = 30) -> str:
        """返回最近 n 行 stderr，用于错误排查。"""
        with self._stderr_lock:
            tail = list(self._stderr_buf)[-n:]
        return "\n".join(tail)

    def send(self, message: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPConnectionError("传输未连接")
        if self._proc.poll() is not None:
            # 子进程已经退出，给出带 stderr 的清晰错误
            err_tail = self.get_last_stderr()
            detail = f"子进程已退出（exit={self._proc.returncode}）。"
            if err_tail:
                detail += f" stderr 尾部：\n{err_tail}"
            raise MCPConnectionError(detail)
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPConnectionError(f"写入子进程 stdin 失败（子进程可能已退出）：{e}") from e

    def recv(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """接收消息。循环跳过内部事件消息，避免递归导致栈溢出。"""
        deadline = time.time() + (timeout if timeout is not None else float("inf"))
        while True:
            remain = deadline - time.time()
            if timeout is not None and remain <= 0:
                return None
            try:
                msg = self._queue.get(timeout=min(remain, 0.5) if timeout is not None else 0.5)
            except queue.Empty:
                # 检查进程是否已经退出
                if self._proc is not None and self._proc.poll() is not None:
                    err_tail = self.get_last_stderr()
                    detail = f"子进程已退出（exit={self._proc.returncode}）"
                    if err_tail:
                        detail += f"。stderr 尾部：\n{err_tail}"
                    raise MCPConnectionError(detail)
                if timeout is None:
                    continue
                return None
            if not isinstance(msg, dict) or not msg.get("_event"):
                return msg
            if msg.get("_event") == "disconnected":
                raise MCPConnectionError(msg.get("error", "MCP stdio 连接已断开"))
            # 其他内部事件（理论上 stdio 不会有）继续取下一条

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        # 关闭 stdin，让子进程有机会正常退出
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        # terminate + 等待
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # 等待读线程退出（最多 1 秒）
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=1.0)
        if self._stderr_reader is not None and self._stderr_reader.is_alive():
            self._stderr_reader.join(timeout=0.5)


def make_stdio_transport(command_str: str, cwd: Optional[str] = None) -> StdioTransport:
    """便捷工厂：把 'python server.py --arg x' 拆成命令+参数，Windows 用 posix=False。"""
    parts = shlex.split(command_str, posix=(sys.platform != "win32"))
    if not parts:
        raise MCPConnectionError("命令不能为空")
    return StdioTransport(command=parts[0], args=parts[1:], cwd=cwd)
