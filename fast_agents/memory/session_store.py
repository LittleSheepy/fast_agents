"""短期会话持久化后端（SessionStore）

对应演进路线图 B1 + J3：`buffer` 纯内存进程重启即丢，新增这一层后
「append 同步落库 / 构造时 load 回填」，上层（MemoryManager、ReActAgent）
接口不变，进程重启能恢复上次会话的消息与摘要。

设计原则（对齐 long_term / MemoryStore 的已有风格）：
1. 零重型依赖：底层复用 `fast_agents.database.SQLiteClient`，统一管理连接、
   WAL、busy_timeout、外键、参数化查询、幂等建表，不再手写 sqlite3；
2. 作用域隔离：所有读写都强制带 user_id + session_id，杜绝串会话；
3. seq 保序：消息用自增 seq 列，恢复时严格按 seq 排序，避免时间戳同毫秒乱序；
4. 上层接口稳定：BufferMemory 只多一个可选 `persist_store` 参数，不传完全无变化。

为什么是独立表、不是塞进 memories 表：
- memories 表存的是「抽取后的长期记忆」（MemoryRecord），字段（kind/status/importance/supersedes...）
  和短期消息完全不匹配，塞进去会让长期记忆表污染；
- 短期消息量大、生命周期短、需要按 seq 顺序快速回放，单独一张表性能更好、清档更方便；
- LangChain 的 `ConversationBufferMemory` / ChatHistory 也是单独的消息表，语义清晰。

关于 `session_meta` 的唯一约束：
- sessions 表用 SQLiteClient 的 dict 形式声明式建表（seq 需要 AUTOINCREMENT 修饰词，
  不在 ColumnType 枚举内，故用最宽松的 dict 形式）；
- session_meta 的 (user_id, session_id) 唯一性用 `Index(unique=True)` 表达：单列主键
  可由 Column.primary_key 声明，但表级复合主键无法声明；而复合唯一索引与复合主键在
  本场景（作用域隔离 + upsert 冲突判定）等价，且 SQLiteClient.upsert 只要求冲突列上
  有 PRIMARY KEY / UNIQUE 约束。因此 session_meta 表零裸 SQL，建表与索引都走
  Schema / Column / Index 声明式完成。
"""

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from ..database import Column, ColumnType, Filter, FilterOp, Index, Query, Schema, SQLiteClient, SortOrder
from ..messages import Message
from fast_agents.utils.time_utils import to_cst, cstnow


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

    底层复用 `SQLiteClient`，自动获得：
    - PRAGMA journal_mode=WAL（非内存库，并发读写更友好）；
    - PRAGMA busy_timeout（写锁等待，避免并发写锁死）；
    - PRAGMA foreign_keys=ON；
    - 参数化查询与标识符白名单校验。

    表结构：
        sessions(seq INTEGER PK AUTOINCREMENT,
                 user_id TEXT, session_id TEXT,
                 role TEXT, content TEXT,
                 tool_calls TEXT, tool_results TEXT,
                 timestamp TEXT, metadata TEXT)
        session_meta(user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                     summary TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                     唯一性由 UNIQUE INDEX(user_id, session_id) 保证)
    """

    _SESSIONS_TABLE = "sessions"
    _META_TABLE = "session_meta"
    _IDX_SCOPE = "idx_sessions_scope"
    _META_IDX_UNIQUE = "uq_session_meta_scope"

    # session_meta 表结构：完全声明式（Schema / Column / Index），无裸 SQL。
    # 复合唯一性用 Index(unique=True) 表达——等价于原表级复合主键，
    # 且满足 SQLiteClient.upsert 对冲突列「必须有 PRIMARY KEY / UNIQUE 约束」的要求。
    _META_SCHEMA = Schema(
        table_name=_META_TABLE,
        columns=[
            Column(name="user_id", type=ColumnType.TEXT, nullable=False),
            Column(name="session_id", type=ColumnType.TEXT, nullable=False),
            Column(name="summary", type=ColumnType.TEXT, nullable=False, default=""),
            Column(name="updated_at", type=ColumnType.TEXT, nullable=False),
        ],
        indexes=[
            Index(name=_META_IDX_UNIQUE, columns=["user_id", "session_id"], unique=True),
        ],
    )

    def __init__(
        self,
        path: str,
        user_id: str = "default",
        session_id: str = "default",
        *,
        timeout: float = 10.0,
    ):
        """
        Args:
            path: SQLite 文件路径；与长期记忆共用同一个 .db 文件即可（会自动建表）
            user_id / session_id: 作用域键，强制隔离，不允许空串
            timeout: SQLite 写锁等待超时秒数（透传给 SQLiteClient，默认 10s）
        """
        if str(path) == ":memory:":
            raise ValueError("SessionStore 要做持久化，请传文件路径；临时会话不传 persist_store 即可")
        if not all(isinstance(v, str) and v.strip() for v in (user_id, session_id)):
            raise ValueError("user_id 和 session_id 必须是非空字符串")
        self.path = Path(path)
        self.user_id = user_id
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # 复用 database 模块统一的 SQLiteClient，不再手写 sqlite3.connect/close
        self._db = SQLiteClient(str(self.path), timeout=timeout)

        # ---- 建表 ----
        # sessions 表：用 dict 形式声明式建表（列定义直接信任书写的类型片段，
        # 因为 AUTOINCREMENT 这类修饰词不在 Schema.Column 的标准类型枚举里）
        self._db.ensure_table(self._SESSIONS_TABLE, {
            "seq":          "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id":      "TEXT NOT NULL",
            "session_id":   "TEXT NOT NULL",
            "role":         "TEXT NOT NULL",
            "content":      "TEXT NOT NULL DEFAULT ''",
            "tool_calls":   "TEXT NOT NULL DEFAULT '[]'",
            "tool_results": "TEXT NOT NULL DEFAULT '[]'",
            "timestamp":    "TEXT NOT NULL",
            "metadata":     "TEXT NOT NULL DEFAULT '{}'",
        })

        # 幂等建索引：CREATE INDEX IF NOT EXISTS 天然幂等，每次调用都安全
        self._db.execute(
            f"CREATE INDEX IF NOT EXISTS {self._IDX_SCOPE} "
            f"ON {self._SESSIONS_TABLE}(user_id, session_id, seq)"
        )

        # session_meta 表：走声明式 Schema 建表（含复合唯一索引），零裸 SQL。
        # migrate=True 兼容旧库——旧表若是裸 DDL 建的复合主键版本，这里只幂等补索引。
        self._db.ensure_schema(self._META_SCHEMA, migrate=True)

    @contextmanager
    def _connect(self) -> Iterator:
        """兼容桥：若其他模块（如 semantic_store 借道 inner）需要底层 sqlite3 连接，
        通过 ``with self._db:`` 显式包裹一个事务，yield 底层 connection；
        正常退出 commit，异常自动 rollback——与原版语义完全一致。

        注意：SQLiteClient 持有单连接且默认开事务，这里直接复用它的上下文管理；
        新版本记忆模块内部方法不再调用 _connect，全部走 SQLiteClient 的声明式 API。
        """
        with self._db:
            yield self._db.conn

    # ---------- 序列化 ----------

    @staticmethod
    def _row_to_message(row) -> Message:
        """把一行 sessions 表记录还原为 Message（含 tool_calls / tool_results / metadata）

        入参接受 sqlite3.Row 或 dict（SQLiteClient.select 返回 dict），两种都兼容。
        """
        def _get(k, default=None):
            if isinstance(row, dict):
                return row.get(k, default)
            return row[k] if k in row.keys() else default

        try:
            tool_calls_raw = json.loads(_get("tool_calls") or "[]")
        except json.JSONDecodeError:
            tool_calls_raw = []
        try:
            tool_results_raw = json.loads(_get("tool_results") or "[]")
        except json.JSONDecodeError:
            tool_results_raw = []
        try:
            metadata = json.loads(_get("metadata") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            ts_raw = _get("timestamp")
            ts = to_cst(datetime.fromisoformat(ts_raw)) if ts_raw else cstnow()
        except (ValueError, TypeError):
            ts = cstnow()

        # 还原 MsgToolCall / MsgToolResult（Message.__init__ 的 kwargs 里直接传 dict 列表即可，
        # Pydantic 会自动转成对应的 BaseModel 实例）
        msg = Message(
            role=_get("role"),
            content=_get("content") or "",
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
    def _message_to_row(user_id: str, session_id: str, m: Message) -> dict:
        """把 Message 序列化成 sessions 表行字典（给 SQLiteClient.insert 用）"""
        return {
            "user_id": user_id,
            "session_id": session_id,
            "role": m.role,
            "content": m.content or "",
            "tool_calls": json.dumps(
                [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in (m.tool_calls or [])],
                ensure_ascii=False,
            ),
            "tool_results": json.dumps(
                [tr.model_dump() if hasattr(tr, "model_dump") else tr for tr in (m.tool_results or [])],
                ensure_ascii=False,
            ),
            "timestamp": to_cst(m.timestamp or cstnow()).isoformat(),
            "metadata": json.dumps(m.metadata or {}, ensure_ascii=False, default=str),
        }

    def _scope_filters(self) -> List[Filter]:
        """返回本会话的作用域过滤条件（user_id + session_id），统一给 Query 用"""
        return [
            Filter(field="user_id", op=FilterOp.EQ, value=self.user_id),
            Filter(field="session_id", op=FilterOp.EQ, value=self.session_id),
        ]

    # ---------- 接口实现 ----------

    def load(self) -> List[Message]:
        rows = self._db.select_where(Query(
            table_name=self._SESSIONS_TABLE,
            filters=self._scope_filters(),
            sort_by="seq",
            sort_order=SortOrder.ASC,
        ))
        return [self._row_to_message(r) for r in rows]

    def append(self, message: Message) -> None:
        row = self._message_to_row(self.user_id, self.session_id, message)
        self._db.insert(self._SESSIONS_TABLE, row)

    def clear(self) -> None:
        scope_q = Query(table_name=self._SESSIONS_TABLE, filters=self._scope_filters())
        meta_q = Query(table_name=self._META_TABLE, filters=self._scope_filters())
        self._db.delete_where(self._SESSIONS_TABLE, scope_q)
        self._db.delete_where(self._META_TABLE, meta_q)

    def save_summary(self, summary: str) -> None:
        # 走 SQLiteClient.upsert 表达 INSERT ... ON CONFLICT DO UPDATE，
        # conflict_keys 是复合唯一键 (user_id, session_id)（由 _META_SCHEMA 的唯一索引保证），
        # 冲突时更新 summary 和 updated_at
        self._db.upsert(
            self._META_TABLE,
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "summary": summary or "",
                "updated_at": cstnow().isoformat(),
            },
            conflict_keys=["user_id", "session_id"],
            update_columns=["summary", "updated_at"],
        )

    def load_summary(self) -> str:
        row = self._db.select_one_where(Query(
            table_name=self._META_TABLE,
            filters=self._scope_filters(),
        ))
        return row["summary"] if row else ""

    def close(self) -> None:
        self._db.close()
        return None


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
