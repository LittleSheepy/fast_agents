"""SQLite 长期记忆后端

设计参考 HelloAgents 的 MemoryStore，核心是三条原则：
1. 作用域隔离：每条记录都带 user_id / session_id，查询永远带范围过滤，
   杜绝 A 的记忆被 B 读到；
2. 修订而非覆盖：改一条记忆时，把旧的标记为 superseded，另插一条新记录，
   历史永不丢失（审计友好）；
3. 显式删除：retract 只改状态保留历史，delete 才真删。

B5 新增两个记账字段 access_count / last_accessed_at：旧库在 __init__ 里通过
``ensure_schema(migrate=True)`` 自动 ALTER TABLE 补齐（幂等），无需手动迁移；
命中记账走 touch()，**不刷新 updated_at**，以免破坏时间衰减打分。

底层复用 ``fast_agents.database.SQLiteClient``：
- 自动开启 WAL + busy_timeout，并发读写更友好（B9 落地）；
- 建表 / 索引 / 常规 CRUD 全部走声明式 API，不再手写 ``CREATE TABLE`` /
  ``PRAGMA table_info`` / ``ALTER TABLE`` 样板；
- 标识符白名单 + 参数化查询，杜绝 SQL 注入；
- 只有少数 SQLite 特有写法（比如 ``INSERT OR REPLACE``、``revise`` 的两步事务）
  通过 ``db.execute()`` 逃生舱保留；``touch()`` 利用 ``update_where`` + ``SqlExpr``
  实现批量自增，不需要手写 ``IN (?,?,….)``。

search 不在此处实现，由 MemoryStoreBase 通过 all() + scoring.rank_records
提供默认实现，保证与 InMemoryStore 打分口径一致（B7）。后续若要做 SQL 侧
预过滤 / 全文索引 / FTS5 向量召回，可直接 override search()。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from contextlib import contextmanager

from ..database import (
    Column,
    ColumnType,
    Filter,
    FilterOp,
    Index,
    Query,
    Schema,
    SqlExpr,
    SQLiteClient,
    SortOrder,
)
from fast_agents.utils.time_utils import to_cst, to_cst_iso, cstnow
from .memory_store_base import MemoryStoreBase
from .models import MemoryRecord


class SQLiteMemoryStore(MemoryStoreBase):
    """按 user_id + session_id 隔离的持久化记忆（底层复用 SQLiteClient）"""

    _TABLE = "memories"
    _IDX_SCOPE = "idx_scope"

    # 声明式表结构，供 ensure_schema(..., migrate=True) 使用；
    # 缺列由 SQLiteClient 自动补 ADD COLUMN，新库一次建齐。
    _SCHEMA = Schema(
        table_name=_TABLE,
        columns=[
            Column(name="memory_id",        type=ColumnType.TEXT, primary_key=True),
            Column(name="user_id",          type=ColumnType.TEXT, nullable=False),
            Column(name="session_id",       type=ColumnType.TEXT, nullable=False),
            Column(name="content",          type=ColumnType.TEXT, nullable=False),
            Column(name="source",           type=ColumnType.TEXT, nullable=False),
            Column(name="kind",             type=ColumnType.TEXT, nullable=False),
            Column(name="status",           type=ColumnType.TEXT, nullable=False),
            Column(name="importance",       type=ColumnType.REAL, nullable=False),
            Column(name="created_at",       type=ColumnType.TIMESTAMP, nullable=False),
            Column(name="updated_at",       type=ColumnType.TIMESTAMP, nullable=False),
            Column(name="supersedes",       type=ColumnType.TEXT),
            # B5 新增列：新库直接建，老库 ensure_schema(migrate=True) 自动补
            Column(name="access_count",     type=ColumnType.INTEGER, nullable=False, default=0),
            Column(name="last_accessed_at", type=ColumnType.TIMESTAMP),
            # B12 新增：可选有效期；新库直接建，老库 ensure_schema(migrate=True) 自动补
            Column(name="expires_at",       type=ColumnType.TIMESTAMP),
            Column(name="metadata",         type=ColumnType.TEXT, nullable=False),
        ],
        indexes=[
            Index(name=_IDX_SCOPE, columns=["user_id", "session_id", "status"]),
        ],
    )

    def __init__(
        self,
        path: str,
        user_id: str = "default_user",
        session_id: str = "default_session",
        scope: str = "user",
        *,
        timeout: float = 10.0,
    ):
        """
        Args:
            scope: 作用域策略
                - "user"（默认，推荐用于长期记忆）：按 user_id 隔离，
                  同一用户的不同会话共享记忆（所以换个 session_id 还能想起你）；
                - "session"：按 user_id + session_id 双重隔离，
                  记忆只在单次会话内可见（适合「本窗口内的临时结论」）。
            timeout: SQLite 写锁等待超时秒数（透传给 SQLiteClient；同时映射到 busy_timeout）。
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

        # 复用 database 模块统一的 SQLiteClient：自动 WAL / busy_timeout / 外键 / 参数化
        self._db = SQLiteClient(str(self.path), timeout=timeout)
        # 幂等建表 + 幂等建索引 + 老库自动补列（B5 的 access_count / last_accessed_at）
        self._db.ensure_schema(self._SCHEMA, migrate=True)

    # ---------- 兼容桥（供 SemanticMemoryStore 等同类内部使用）----------
    # semantic_store 的向量表建在本类的 SQLite 连接里（同库一张独立表），
    # 它通过 inner._connect() 拿到底层 sqlite3.Connection、通过 inner._scope_clause()
    # 拿到 (WHERE 片段, 参数) 做作用域过滤。为避免破坏这个既有耦合，保留这两个
    # 私有方法：它们是实现细节，不是公开 API，不要在第三方代码里调用。

    @contextmanager
    def _connect(self):
        """返回底层 sqlite3.Connection 的上下文管理器（供 semantic_store 等内部复用）。

        保持与旧实现一致的事务语义：进入块时 BEGIN，正常退出 COMMIT，异常 ROLLBACK——
        这是依赖 SQLiteClient 的 ``with self._db`` 显式事务语义实现的；块内 yield
        的就是底层连接对象，第三方代码可直接在上面 execute()。
        """
        with self._db:
            yield self._db.conn

    def _scope_clause(self) -> tuple:
        """返回 (WHERE 片段, 参数列表)，与旧版语义一致，供 semantic_store 复用。"""
        if self.scope == "user":
            return "user_id=?", [self.user_id]
        return "user_id=? AND session_id=?", [self.user_id, self.session_id]

    # ---------- 作用域过滤（所有读写都走这一组 Filter）----------

    def _scope_filters(self) -> List[Filter]:
        """返回作用域对应的 Filter 列表，所有 CRUD 都带上做隔离。"""
        if self.scope == "user":
            return [Filter(field="user_id", op=FilterOp.EQ, value=self.user_id)]
        return [
            Filter(field="user_id", op=FilterOp.EQ, value=self.user_id),
            Filter(field="session_id", op=FilterOp.EQ, value=self.session_id),
        ]

    def _scope_where(self, extra: Optional[List[Filter]] = None) -> List[Filter]:
        """scope + 额外条件拼在一起，构造 query.filters 用。"""
        return self._scope_filters() + list(extra or [])

    def _scope_eq(self, *, scope_only: bool = False) -> dict:
        """等值 where 字典（供 select/update/delete 的等值 API 使用）。"""
        w: dict = {"user_id": self.user_id}
        if self.scope == "session":
            w["session_id"] = self.session_id
        return w

    # ---------- 序列化 ----------

    @staticmethod
    def _row_to_record(row: dict) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            source=row["source"],
            kind=row["kind"],
            status=row["status"],
            importance=row["importance"],
            created_at=to_cst(datetime.fromisoformat(row["created_at"])),
            updated_at=to_cst(datetime.fromisoformat(row["updated_at"])),
            supersedes=row["supersedes"],
            access_count=row["access_count"] or 0,
            last_accessed_at=(
                to_cst(datetime.fromisoformat(row["last_accessed_at"]))
                if row["last_accessed_at"] else None
            ),
            expires_at=(
                to_cst(datetime.fromisoformat(row["expires_at"]))
                if row.get("expires_at") else None
            ),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _record_to_row(r: MemoryRecord) -> dict:
        """把 MemoryRecord 转成数据库行字典（字段名与表列一一对应）。"""
        return {
            "memory_id":   r.memory_id,
            "user_id":     r.user_id,
            "session_id":  r.session_id,
            "content":     r.content,
            "source":      r.source,
            "kind":        r.kind,
            "status":      r.status,
            "importance":  r.importance,
            "created_at":  to_cst_iso(r.created_at),
            "updated_at":  to_cst_iso(r.updated_at),
            "supersedes":  r.supersedes,
            "access_count":     r.access_count,
            "last_accessed_at": to_cst_iso(r.last_accessed_at),
            "expires_at":       to_cst_iso(r.expires_at),
            "metadata":    json.dumps(r.metadata, ensure_ascii=False),
        }

    # ---------- 接口实现 ----------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record.user_id = record.user_id or self.user_id
        record.session_id = record.session_id or self.session_id
        # INSERT OR REPLACE：同 memory_id 覆盖，行为与原实现一致；
        # upsert 的 conflict_keys 取主键 memory_id，冲突时更新除主键外的所有列。
        self._db.upsert(self._TABLE, self._record_to_row(record), conflict_keys=["memory_id"])
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._db.select_one_where(Query(
            table_name=self._TABLE,
            filters=self._scope_where([
                Filter(field="memory_id", op=FilterOp.EQ, value=memory_id),
            ]),
        ))
        return self._row_to_record(row) if row else None

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        filters = self._scope_filters()
        if not include_inactive:
            filters.append(Filter(field="status", op=FilterOp.EQ, value="active"))
        rows = self._db.select_where(Query(
            table_name=self._TABLE,
            filters=filters,
            sort_by=["updated_at", "memory_id"],
            sort_order=SortOrder.DESC,
        ))
        return [self._row_to_record(r) for r in rows]

    def search(
        self, query: str, limit: int = 5, *, include_preferences: bool = True
    ) -> List[MemoryRecord]:
        """按综合得分排序返回有效记忆；空查询返回最近记录

        v1.3.0（B7）起默认由 MemoryStoreBase 通过 rank_records 实现，
        这里保留方法 docstring 作为 SQLite 版的语义说明，直接委托父类。
        """
        return super().search(query, limit, include_preferences=include_preferences)

    def touch(self, memory_ids: Iterable[str]) -> int:
        """命中记账：直接 UPDATE 自增，**不触碰 updated_at**

        覆盖基类默认实现的两个理由：
        1. 省掉一次读（基类要走 get + update 两条路径）；
        2. 基类的 update() 会把 updated_at 刷成当前时间，而 updated_at 参与时间衰减
           打分——命中一次就刷新，会让「越旧越轻」失效，这里必须绕开。

        同一次命中的多个 id 通过 ``update_where + FilterOp.IN`` 批量写回：
        逐条执行会发 N 条 SQL、吃 N 份事务开销，而 touch 恰好在「每轮检索」的热路径上。
        """
        ids = [mid for mid in memory_ids if mid]
        if not ids:
            return 0
        now = cstnow().isoformat()
        return self._db.update_where(
            self._TABLE,
            {
                "access_count":     SqlExpr("access_count + 1"),
                "last_accessed_at": now,
            },
            Query(
                table_name=self._TABLE,
                filters=self._scope_where([
                    Filter(field="memory_id", op=FilterOp.IN, value=ids),
                ]),
            ),
        )

    def update(self, record: MemoryRecord) -> MemoryRecord:
        if self.get(record.memory_id) is None:
            raise KeyError(record.memory_id)
        record.updated_at = cstnow()
        self._db.update_where(
            self._TABLE,
            {
                "content":          record.content,
                "source":           record.source,
                "kind":             record.kind,
                "status":           record.status,
                "importance":       record.importance,
                "updated_at":       record.updated_at.isoformat(),
                "access_count":     record.access_count,
                "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else None,
                "expires_at":       record.expires_at.isoformat() if record.expires_at else None,
                "metadata":         json.dumps(record.metadata, ensure_ascii=False),
            },
            Query(
                table_name=self._TABLE,
                filters=self._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=record.memory_id),
                ]),
            ),
        )
        return record

    def revise(self, memory_id: str, content: str, source: str) -> MemoryRecord:
        """修订一条记忆：旧版标记 superseded，追加新版并指向旧版

        两步必须在同一个事务里：要么都成功，要么都回滚；用 SQLiteClient 的
        ``with db:`` 显式事务上下文（BEGIN ... 正常 commit / 异常 rollback）。
        """
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status != "active":
            raise ValueError("只能修订有效记忆，请先重新读取")
        now_iso = cstnow().isoformat()
        with self._db:
            self._db.update_where(
                self._TABLE,
                {"status": "superseded", "updated_at": now_iso},
                Query(
                    table_name=self._TABLE,
                    filters=self._scope_where([
                        Filter(field="memory_id", op=FilterOp.EQ, value=memory_id),
                    ]),
                ),
            )
            new = MemoryRecord(
                user_id=old.user_id, session_id=old.session_id, content=content,
                source=source, kind=old.kind, importance=old.importance,
                supersedes=memory_id, metadata=dict(old.metadata),
            )
            self._db.upsert(self._TABLE, self._record_to_row(new), conflict_keys=["memory_id"])
        return new

    def retract(self, memory_id: str) -> MemoryRecord:
        """撤回一条记忆（保留历史，幂等）"""
        old = self.get(memory_id)
        if old is None:
            raise KeyError(memory_id)
        if old.status == "active":
            old.status = "retracted"
            old.updated_at = cstnow()
            self.update(old)
        return old

    def delete(self, memory_id: str) -> bool:
        return self._db.delete_where(
            self._TABLE,
            Query(
                table_name=self._TABLE,
                filters=self._scope_where([
                    Filter(field="memory_id", op=FilterOp.EQ, value=memory_id),
                ]),
            ),
        ) > 0

    def clear(self) -> None:
        # delete_where 要求 filters 非空；用 scope_filters 保证只清当前作用域，不会误删他用户数据。
        self._db.delete_where(
            self._TABLE,
            Query(table_name=self._TABLE, filters=self._scope_filters()),
        )

    def close(self) -> None:
        self._db.close()
        return None
