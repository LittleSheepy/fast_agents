"""向量数据库客户端抽象基类。

本基类与 ``DatabaseClientBase`` 平行：
- DatabaseClientBase 面向关系型（表/列/事务/精确谓词）；
- VectorClientBase 面向向量型（collection/id/embedding/metadata/近邻检索）。

设计原则：
1. 方法命名对齐项目既有风格（动词主导、与 DatabaseClientBase 对应方法保持平行）：
   - 集合管理：collection_exists / create_collection / get_or_create_collection / drop_collection
   - 写入：add（按 id upsert，覆盖语义）
   - 删除：delete（按 id 或 metadata 过滤删除）
   - 取：get（按 id / metadata 精确取，非近邻）
   - 近邻检索：search（按 embedding 近邻搜索，返回带 distance 的结果）
   - 计数：count
2. 安全约定：delete 空条件禁止（防止误删全量），与 SQLiteClient.delete_where 一致。
3. 不强行加事务 / DDL 迁移 / 外键等关系型能力：向量数据库普遍不支持，基类不承诺。
4. 允许一个后端同时继承 ``DatabaseClientBase`` 和 ``VectorClientBase``
   （如未来 Postgres + pgvector），上层用 ``isinstance`` 判断即可。

::: module-author fast_agents
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .vector_query import VectorWhere


class VectorClientBase(ABC):
    """向量数据库客户端抽象基类。

    子类需实现全部抽象方法；连接生命周期建议在 ``__init__`` 中懒加载，
    首次调用需要连接的方法时再 ``connect()``。
    """

    # ---------- 连接生命周期 ----------
    @abstractmethod
    def connect(self) -> None:
        """建立到底层向量数据库的连接；幂等。"""

    @abstractmethod
    def close(self) -> None:
        """释放连接 / 资源；幂等。"""

    @abstractmethod
    def ping(self) -> bool:
        """连通性检查，成功返回 True。"""

    # ---------- 集合（Collection）管理：对应关系型的「表」 ----------
    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """判断集合是否存在。"""

    @abstractmethod
    def create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        """创建集合；集合已存在时抛出异常（非幂等）。

        Args:
            name: 集合名，需经合法标识符校验。
            dimension: 向量维度；部分后端（如 Milvus）必填，Chroma 可省略。
            metric: 距离度量，支持 ``cosine`` / ``ip``（内积）/ ``l2``，默认 ``cosine``。
        """

    @abstractmethod
    def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        """幂等：存在即跳过、不存在则创建；对应 SQLiteClient.ensure_table 语义。"""

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """删除集合（含其下所有向量）；不存在时静默通过。"""

    # ---------- 写入 ----------
    @abstractmethod
    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """按 id 写入（upsert 语义）：已存在则覆盖、不存在则新增。返回写入条数。

        三个内容字段（embeddings / documents / metadatas）都可省略，但至少有一个非空，
        且提供了的列表长度必须与 ``ids`` 一致。主键就是 ``id``（单字段），不需要
        像关系型 upsert 那样显式传 conflict_keys。
        """

    # ---------- 删除 ----------
    @abstractmethod
    def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
        where: VectorWhere = None,
    ) -> int:
        """按 id 或 metadata 条件删除，返回影响条数。

        安全约定：``ids`` 和 ``where`` 至少要有一个非空；禁止传入空条件误删全量
        （与 SQLiteClient.delete_where 一致，更严格者可以允许空条件，但基类不允许）。
        """

    # ---------- 精确取（非近邻） ----------
    @abstractmethod
    def get(
        self,
        collection: str,
        ids: list[str] | None = None,
        where: VectorWhere = None,
        limit: int | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """按 id 或 metadata 过滤取记录，返回 ``[{id, document?, metadata?, embedding?}]``。

        Args:
            include_embeddings: 默认 False 不返回向量（省流量 / 内存），需要时显式打开。
        """

    # ---------- 近邻检索 ----------
    @abstractmethod
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        *,
        n_results: int = 10,
        where: VectorWhere = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """按单个向量做近邻检索，返回 ``[{id, document?, metadata?, distance}]``，
        按距离升序（最相似的在前）。

        Args:
            n_results: 返回条数上限；若实际结果不足则返回实际条数。
            where: 元数据前置过滤（在向量近邻之前做 hard filter）。
            include_embeddings: 是否在结果中附带向量，默认 False。
        """

    @abstractmethod
    def count(self, collection: str, where: VectorWhere = None) -> int:
        """统计集合内符合条件的条数；``where`` 为 None 时返回全量条数。"""


__all__ = ["VectorClientBase"]
