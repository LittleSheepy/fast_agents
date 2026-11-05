"""数据库客户端抽象基类。

参考主流框架：
- PEP 249（Python DB-API 2.0）的 Connection / Cursor 分层——把「连接」与「执行」拆开，
  本模块的 connect / close 对应其 Connection 生命周期；
- SQLAlchemy Core 的 Engine / Connection / Dialect 多后端抽象——同一套上层接口、多种
  方言实现（sqlite / mysql / postgresql ...），本模块取其「最小接口集」的克制版本；
- 同仓库 ``memory/memory_store_base.py``、``tools/tool_base.py`` 的「抽象基类 +
  子类实现最小接口集」组织方式。

统一约定所有后端（SQLite / MySQL / Postgres / ...）必须实现的最小接口集：
- 连接管理：connect / close
- DDL（建/删/检查表）：create_table / ensure_table / ensure_schema / table_exists / drop_table
- DML（增删改查）：insert / select / select_one / update / delete
- 事务：__enter__ / __exit__（with 语句）

所有参数都用「原生 Python 类型 + schema.py 里的声明式模型」表达，
上层（memory / logs / trace 等）只依赖这个抽象，不直接 import sqlite3。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schema import Schema


class DatabaseClientBase(ABC):
    """数据库客户端抽象基类。子类必须实现下面所有方法。"""

    def __init__(self, timeout: float = 5.0):
        self._conn = None
        self.timeout = timeout

    # ---------- 连接管理 ----------
    @abstractmethod
    def _connect(self) -> None: ...

    def connect(self) -> None:
        if self._conn is None:
            self._connect()

    @abstractmethod
    def _close(self) -> None: ...

    def close(self) -> None:
        """关闭连接。"""
        if self._conn is not None:
            self._close()
            self._conn = None

    # ---------- DDL：建表 / 删表 / 查表 ----------
    @abstractmethod
    def table_exists(self, table: str) -> bool:
        """判断表是否已存在。"""

    @abstractmethod
    def create_table(
        self,
        table: str,
        columns: dict[str, str] | list[Any] | Schema,
        if_not_exists: bool = True,
    ) -> None:
        """建表。

        Args:
            table: 表名。
            columns: 列定义，支持三种形式，按方便程度任选：
                1. ``{列名: SQL类型字符串}`` —— 最简，例如 ``{"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL"}``；
                2. ``list[Column]`` —— 使用 ``schema.Column`` 声明式定义，自动处理 NOT NULL / DEFAULT / UNIQUE；
                3. ``Schema`` 对象 —— 一次性声明列 + 索引（推荐工程化使用）。
            if_not_exists: 是否加上 ``IF NOT EXISTS``（默认 True，幂等）。
        """

    @abstractmethod
    def ensure_table(self, table: str, columns: dict[str, str] | list[Any] | Schema) -> None:
        """幂等建表（等价于 ``create_table(..., if_not_exists=True)``）。

        语义：表存在就跳过、不存在就创建；**不会**修改已有表结构。
        如果需要「缺列就补列」的迁移，请显式调用 ``ensure_schema``（支持 ALTER 迁移的后端）。
        """

    @abstractmethod
    def ensure_schema(self, schema: Schema, migrate: bool = False) -> None:
        """按 ``Schema`` 对象幂等建表 + 建索引。

        Args:
            schema: ``schema.Schema`` 声明式模型，含 table_name / columns / indexes。
            migrate: 为 True 时尝试补列（仅 SQLite 支持 ADD COLUMN，已存在的列自动跳过）；
                     默认 False，只建表 + 建索引，不改动已有表结构（最安全）。
        """

    @abstractmethod
    def drop_table(self, table: str, if_exists: bool = True) -> None:
        """删表。"""

    # ---------- DML：CRUD ----------
    @abstractmethod
    def insert(self, table: str, data: dict) -> int: ...

    @abstractmethod
    def batch_insert(self, table: str, rows: list[dict]) -> int:
        """批量插入多行，返回插入行数；空列表直接返回0。"""

    @abstractmethod
    def select(self, table: str, where: dict | None = None, order_by: str | None = None, limit: int | None = None) -> list[dict]: ...

    @abstractmethod
    def select_one(self, table: str, where: dict) -> dict | None: ...

    @abstractmethod
    def select_where(self, query: Any) -> list[dict]:
        """按声明式 Query 对象查询，支持 EQ/NEQ/GT/GTE/LT/LTE/LIKE/IN/IS[NOT]_NULL。"""

    @abstractmethod
    def select_one_where(self, query: Any) -> dict | None:
        """select_where 的单行版本，limit=1，无结果返回 None。"""

    @abstractmethod
    def count(self, table: str, where: dict | None = None) -> int:
        """按等值条件统计行数，where=None 返回表总行数。"""

    @abstractmethod
    def update(self, table: str, data: dict, where: dict) -> int: ...

    @abstractmethod
    def delete(self, table: str, where: dict) -> int: ...

    # ---------- 事务：with 语句 ----------
    @abstractmethod
    def __enter__(self) -> "DatabaseClientBase":
        """进入事务上下文（``with db:`` 时自动调用）。

        约定：最外层调用开启事务，嵌套重入应「加入父事务」（由实现决定是空操作、
        savepoint 还是深度计数），保证多段写入要么全成功、要么全回滚。
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """退出事务上下文：正常提交、异常回滚。

        返回值 False 表示异常继续向外抛出（不吞异常）；实现必须保证无论 commit /
        rollback 自身是否抛错，都复位内部事务状态，避免留下悬挂事务。
        """
