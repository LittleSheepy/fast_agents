"""SQLite 长期记忆后端

设计参考 HelloAgents 的 MemoryStore，核心是三条原则：
1. 作用域隔离：每条记录都带 user_id / session_id，查询永远带范围过滤，
   杜绝 A 的记忆被 B 读到；
2. 修订而非覆盖：改一条记忆时，把旧的标记为 superseded，另插一条新记录，
   历史永不丢失（审计友好）；
3. 显式删除：retract 只改状态保留历史，delete 才真删。

只用标准库 sqlite3，不引入 SQLAlchemy，保持和你现有 requirements 一致的轻量风格。
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional
from contextlib import contextmanager

from .base import MemoryStore
from .schema import MemoryRecord
from .scoring import combined_score, lexical_score


class SQLiteMemoryStore(MemoryStore):
    """按 user_id + session_id 隔离的持久化记忆"""

    def __init__(
        self,
        path: str,
        user_id: str = "default",
        session_id: str = "default",
        scope: str = "user",
    ):
        """
        Args:
            scope: 作用域策略
                - "user"（默认，推荐用于长期记忆）：按 user_id 隔离，
                  同一用户的不同会话共享记忆（所以换个 session_id 还能想起你）；
                - "session"：按 user_id + session_id 双重隔离，
                  记忆只在单次会话内可见（适合「本窗口内的临时结论」）。
        """
        if str(path) == ":memory:":
            raise ValueError("请提供持久化文件路径；临时记忆请用 InMemoryStore")
        if not all(isinstance(v, str) and v.strip() for v in (user_id, session_id)):
            raise ValueError("user_id 和 session_id 必须是非空字符串")
        if scope not in ("user", "session"):
            raise ValueError("scope 必须是 'user' 或 'session'")
        self.path = Path(path)
        self.user_id = user_id
        self.session_id = session_id
        self.scope = scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id   TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    kind        TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    importance  REAL NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    supersedes  TEXT,
                    metadata    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scope
                    ON memories(user_id, session_id, status);
                """
            )

    def _scope_clause(self) -> tuple:
        """返回 (WHERE 片段, 参数列表)，所有读写都必须带上它做隔离"""
        if self.scope == "user":
            return "user_id=?", [self.user_id]
        return "user_id=? AND session_id=?", [self.user_id, self.session_id]

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
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            source=row["source"],
            kind=row["kind"],
            status=row["status"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            supersedes=row["supersedes"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _to_params(r: MemoryRecord) -> tuple:
        return (
            r.memory_id, r.user_id, r.session_id, r.content, r.source, r.kind,
            r.status, r.importance, r.created_at.isoformat(), r.updated_at.isoformat(),
            r.supersedes, json.dumps(r.metadata, ensure_ascii=False),
        )

    # ---------- 接口实现 ----------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record.user_id = record.user_id or self.user_id
        record.session_id = record.session_id or self.session_id
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                self._to_params(record),
            )
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        clause, params = self._scope_clause()
        with self._connect() as db:
            row = db.execute(
                f"SELECT * FROM memories WHERE memory_id=? AND {clause}",
                [memory_id, *params],
            ).fetchone()
        return self._row_to_record(row) if row else None

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        clause, params = self._scope_clause()
        sql = f"SELECT * FROM memories WHERE {clause}"
        if not include_inactive:
            sql += " AND status='active'"
        sql += " ORDER BY updated_at DESC, memory_id"
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(
        self, query: str, limit: int = 5, *, include_preferences: bool = True
    ) -> List[MemoryRecord]:
        """按综合得分排序返回有效记忆；空查询返回最近记录

        include_preferences=True 时，偏好类（preference）记忆无条件进入候选，
        因为「用户偏好」这类信息几乎对所有问题都相关，不该被词法重叠卡掉。
        这也是很多商业产品的做法：偏好常驻上下文，事实/情景按需召回。
        """
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit 必须是正整数")

        records = self.all()
        if not query.strip():
            return records[:limit]

        scored = []
        for r in records:
            if r.kind == "preference" and include_preferences:
                # 偏好类：即使字面不相关也给一个基础分，靠重要性排序
                base = r.importance * 0.5 + lexical_score(query, r.content) * 0.5
                scored.append((base, r))
                continue
            if lexical_score(query, r.content) > 0:
                s = combined_score(query, r.content, r.importance, r.updated_at)
                scored.append((s, r))
        scored.sort(key=lambda x: (-x[0], -x[1].importance))
        return [r for _, r in scored[:limit]]

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if self.get(record.memory_id) is None:
            raise KeyError(record.memory_id)
        record.updated_at = datetime.now()
        clause, params = self._scope_clause()
        with self._connect() as db:
            db.execute(
                f"UPDATE memories SET content=?, source=?, kind=?, status=?, importance=?,"
                f" updated_at=?, metadata=? WHERE memory_id=? AND {clause}",
                (
                    record.content, record.source, record.kind, record.status,
                    record.importance, record.updated_at.isoformat(),
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.memory_id, *params,
                ),
            )
        return record

    def revise(self, memory_id: str, content: str, source: str) -> MemoryRecord:
        """修订一条记忆：旧版标记 superseded，追加新版并指向旧版"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status != "active":
            raise ValueError("只能修订有效记忆，请先重新读取")
        clause, params = self._scope_clause()
        with self._connect() as db:
            db.execute(
                f"UPDATE memories SET status='superseded', updated_at=? "
                f"WHERE memory_id=? AND {clause}",
                (datetime.now().isoformat(), memory_id, *params),
            )
        new = MemoryRecord(
            user_id=old.user_id, session_id=old.session_id, content=content,
            source=source, kind=old.kind, importance=old.importance,
            supersedes=memory_id, metadata=dict(old.metadata),
        )
        return self.add(new)

    def retract(self, memory_id: str) -> MemoryRecord:
        """撤回一条记忆（保留历史，幂等）"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status == "active":
            old.status = "retracted"
            old.updated_at = datetime.now()
            self.update(old)
        return old

    def delete(self, memory_id: str) -> bool:
        clause, params = self._scope_clause()
        with self._connect() as db:
            cur = db.execute(
                f"DELETE FROM memories WHERE memory_id=? AND {clause}",
                [memory_id, *params],
            )
        return cur.rowcount > 0

    def clear(self) -> None:
        clause, params = self._scope_clause()
        with self._connect() as db:
            db.execute(f"DELETE FROM memories WHERE {clause}", params)

    def close(self) -> None:
        return None
