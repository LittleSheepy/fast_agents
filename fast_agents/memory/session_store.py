"""短期会话持久化后端（SessionStore）

对应演进路线图 B1：`buffer` 纯内存进程重启即丢，新增这一层后
「append 同步落库 / 构造时 load 回填」，上层（MemoryManager、ReActAgent）
接口不变，进程重启能恢复上次会话的消息与摘要。

设计原则（对齐 long_term / MemoryStore 的已有风格）：
1. 零重型依赖：只用标准库 sqlite3 + json，与 SQLiteMemoryStore 同款；
2. 作用域隔离：所有读写都强制带 user_id + session_id，杜绝串会话；
3. seq 保序：消息用自增 seq 列，恢复时严格按 seq 排序，避免时间戳同毫秒乱序；
4. 上层接口稳定：BufferMemory 只多一个可选 `persist_store` 参数，不传完全无变化。

为什么是独立表、不是塞进 memories 表：
- memories 表存的是「抽取后的长期记忆」（MemoryRecord），字段（kind/status/importance/supersedes...）
  和短期消息完全不匹配，塞进去会让长期记忆表污染；
- 短期消息量大、生命周期短、需要按 seq 顺序快速回放，单独一张表性能更好、清档更方便；
- LangChain 的 `ConversationBufferMemory` / ChatHistory 也是单独的消息表，语义清晰。
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from ..messages import Message


class SessionStore(ABC):
    """短期会话持久化抽象接口

    所有方法都以 session_id 为主键（作用域隔离在构造时由 user_id + session_id 决定）。
    接口刻意精简：只有 load / append / clear / close 四个，BufferMemory 不需要更多。
    """

    @abstractmethod
    def load(self) -> List[Message]:
        """按 seq 顺序读回本会话全部消息，空会话返回 []"""

    @abstractmethod
    def append(self, message: Message) -> None:
        """追加一条消息（自动分配 seq）"""

    @abstractmethod
    def clear(self) -> None:
        """清空本会话的全部消息（开启新会话/重置时调用）"""

    def save_summary(self, summary: str) -> None:
        """持久化当前滚动摘要（可选实现；不支持的后端默认 no-op）"""

    def load_summary(self) -> str:
        """读回上次持久化的摘要（可选实现；默认空串）"""
        return ""

    def close(self) -> None:
        """释放资源"""
        return None


class SQLiteSessionStore(SessionStore):
    """SQLite 版 SessionStore：零依赖、按 user_id + session_id 隔离

    表结构：
        sessions(seq INTEGER PK AUTOINCREMENT,
                 user_id TEXT, session_id TEXT,
                 role TEXT, content TEXT,
                 tool_calls TEXT, tool_results TEXT,
                 timestamp TEXT, metadata TEXT)
        session_meta(user_id, session_id, summary TEXT, updated_at TEXT,
                     PRIMARY KEY(user_id, session_id))
    """

    def __init__(
        self,
        path: str,
        user_id: str = "default",
        session_id: str = "default",
    ):
        """
        Args:
            path: SQLite 文件路径；与长期记忆共用同一个 .db 文件即可（会自动建表）
            user_id / session_id: 作用域键，强制隔离，不允许空串
        """
        if str(path) == ":memory:":
            raise ValueError("SessionStore 要做持久化，请传文件路径；临时会话不传 persist_store 即可")
        if not all(isinstance(v, str) and v.strip() for v in (user_id, session_id)):
            raise ValueError("user_id 和 session_id 必须是非空字符串")
        self.path = Path(path)
        self.user_id = user_id
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL DEFAULT '',
                    tool_calls  TEXT NOT NULL DEFAULT '[]',
                    tool_results TEXT NOT NULL DEFAULT '[]',
                    timestamp   TEXT NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_scope
                    ON sessions(user_id, session_id, seq);

                CREATE TABLE IF NOT EXISTS session_meta (
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    summary     TEXT NOT NULL DEFAULT '',
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY(user_id, session_id)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    # ---------- 序列化 ----------

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        """把一行 sessions 表记录还原为 Message（含 tool_calls / tool_results / metadata）"""
        try:
            tool_calls_raw = json.loads(row["tool_calls"] or "[]")
        except json.JSONDecodeError:
            tool_calls_raw = []
        try:
            tool_results_raw = json.loads(row["tool_results"] or "[]")
        except json.JSONDecodeError:
            tool_results_raw = []
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            ts = datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else datetime.now()
        except ValueError:
            ts = datetime.now()

        # 还原 MsgToolCall / MsgToolResult（Message.__init__ 的 kwargs 里直接传 dict 列表即可，
        # Pydantic 会自动转成对应的 BaseModel 实例）
        msg = Message(
            role=row["role"],
            content=row["content"] or "",
            timestamp=ts,
            metadata=metadata,
            tool_calls=tool_calls_raw,
        )
        # tool_results 不在 Message.__init__ 的显式 kwargs 里，通过对象属性赋值补上
        # （查看 message.py 定义：tool_results 是 Message 的 BaseModel 字段，可直接赋值）
        from ..messages.message import MsgToolResult
        msg.tool_results = [
            MsgToolResult(**tr) if isinstance(tr, dict) else tr
            for tr in tool_results_raw
        ]
        return msg

    @staticmethod
    def _message_to_params(user_id: str, session_id: str, m: Message) -> tuple:
        return (
            user_id, session_id,
            m.role, m.content or "",
            json.dumps(
                [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in (m.tool_calls or [])],
                ensure_ascii=False,
            ),
            json.dumps(
                [tr.model_dump() if hasattr(tr, "model_dump") else tr for tr in (m.tool_results or [])],
                ensure_ascii=False,
            ),
            (m.timestamp or datetime.now()).isoformat(),
            json.dumps(m.metadata or {}, ensure_ascii=False, default=str),
        )

    # ---------- 接口实现 ----------

    def load(self) -> List[Message]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM sessions WHERE user_id=? AND session_id=? ORDER BY seq ASC",
                (self.user_id, self.session_id),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def append(self, message: Message) -> None:
        params = self._message_to_params(self.user_id, self.session_id, message)
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions(user_id,session_id,role,content,tool_calls,tool_results,timestamp,metadata)"
                " VALUES(?,?,?,?,?,?,?,?)",
                params,
            )

    def clear(self) -> None:
        with self._connect() as db:
            db.execute(
                "DELETE FROM sessions WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            )
            db.execute(
                "DELETE FROM session_meta WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            )

    def save_summary(self, summary: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO session_meta(user_id,session_id,summary,updated_at) VALUES(?,?,?,?)"
                " ON CONFLICT(user_id,session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
                (self.user_id, self.session_id, summary or "", datetime.now().isoformat()),
            )

    def load_summary(self) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT summary FROM session_meta WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            ).fetchone()
        return row["summary"] if row else ""


class InMemorySessionStore(SessionStore):
    """纯内存版 SessionStore，用于单元测试 / 不想真落盘但想走统一接口的场景

    注意：它和「不传 persist_store」的区别是——它会保留 append 的消息，
    但仍然是进程内的，进程退出即丢。主要用于测试 BufferMemory 的持久化路径
    而无需真的建 SQLite 文件。
    """

    def __init__(self):
        self._messages: List[Message] = []
        self._summary: str = ""

    def load(self) -> List[Message]:
        return list(self._messages)

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def save_summary(self, summary: str) -> None:
        self._summary = summary or ""

    def load_summary(self) -> str:
        return self._summary
