"""结构化用户画像存储（B3）

为什么要单独一层：
- 长期记忆（`MemoryRecord`）记的是「碎片」——可能被遗忘、修订、检索，带 kind/status；
- 用户画像是「档案」——姓名、身份、所在地、语言、稳定偏好，是**一份**、**覆盖式更新**、
  几乎每轮都要用的东西。把它塞进 `memories` 表会让检索链路和画像混在一起，
  既重复打分又不好维护。
所以这里单独建一层：抽象接口 + SQLite 实现 + 内存实现，
与 `memory_store_base.py` / `session_store.py` 保持同一种风格（抽象 + 双实现，零重型依赖）。

设计约定：
1. `get / merge / set_field` 三个原子操作，`user_id` 由调用方显式传入（画像天然按用户）；
2. 字段级覆盖式合并：同名 key 后写覆盖前写，两个 dict 做浅合并，**不做版本链**；
3. `get` 到不存在的用户返回**空 `UserProfile`**，不抛异常（上层调用零特判）；
4. 独立表 `user_profiles`，不污染 `memories` 主表；可以和长期记忆 / 短期会话共用同一个 .db 文件。
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..database import SQLiteClient
from fast_agents.utils.time_utils import cstnow
from .schema import UserProfile

# 允许直接写到 UserProfile 顶层的字段白名单
_TOP_LEVEL_FIELDS = ("name", "role", "location", "language")

# 键值类字段的路径前缀（set_field 用）
_PREF_PREFIX = "preferences."
_FACT_PREFIX = "facts."


def build_field_patch(field: str, value: str) -> Dict[str, Any]:
    """把「字段路径 + 值」翻译成 patch 字典

    支持两种写法：
    - 顶层字段：`name` / `role` / `location` / `language`
      → `{"name": "张三"}`
    - 键值字段：`preferences.<键>` / `facts.<键>`
      → `{"stable_preferences": {"<键>": "值"}}`

    非法的字段名直接 `raise ValueError`，避免静默写进莫名其妙的键。
    """
    name = (field or "").strip()
    if not name:
        raise ValueError("field 不能为空")
    if name in _TOP_LEVEL_FIELDS:
        return {name: value}
    if name.startswith(_PREF_PREFIX):
        key = name[len(_PREF_PREFIX):].strip()
        if not key:
            raise ValueError("preferences. 后面要跟具体的偏好名，例如 preferences.编辑器")
        return {"stable_preferences": {key: value}}
    if name.startswith(_FACT_PREFIX):
        key = name[len(_FACT_PREFIX):].strip()
        if not key:
            raise ValueError("facts. 后面要跟具体的事实名，例如 facts.公司")
        return {"stable_facts": {key: value}}
    raise ValueError(
        f"未知画像字段：{field!r}（支持 name/role/location/language，"
        f"或 {_PREF_PREFIX}<键> / {_FACT_PREFIX}<键>）"
    )


class ProfileStore(ABC):
    """用户画像存储抽象接口

    接口刻意精简——只有 get / merge / set_field 三个动作，
    `MemoryManager` 不需要更多；`set_field` 有默认实现（走 merge），
    子类只要实现 `get` 和 `merge` 即可。
    """

    @abstractmethod
    def get(self, user_id: str) -> UserProfile:
        """读取画像；不存在时返回空 `UserProfile`，不抛异常"""

    @abstractmethod
    def merge(
        self, user_id: str, patch: Dict[str, Any], *, source: str = "manual"
    ) -> UserProfile:
        """按 patch 覆盖式合并画像并落库，返回合并后的完整画像"""

    def set_field(
        self, user_id: str, field: str, value: str, *, source: str = "manual"
    ) -> UserProfile:
        """更新单个字段（默认实现：翻译成 patch 后走 merge）"""
        return self.merge(user_id, build_field_patch(field, value), source=source)

    def close(self) -> None:
        """释放资源（子类按需实现）"""
        return None


class SQLiteProfileStore(ProfileStore):
    """SQLite 版画像存储：零依赖、单独一张表，底层复用 `SQLiteClient`

    表结构：
        user_profiles(user_id TEXT PRIMARY KEY,
                      profile  TEXT NOT NULL,   -- UserProfile 的 JSON
                      source   TEXT NOT NULL,   -- 最后一次写入的来源
                      updated_at TEXT NOT NULL)
    """

    def __init__(self, path: str):
        """
        Args:
            path: SQLite 文件路径；与长期记忆 / 短期会话共用同一个 .db 文件即可（会自动建表）
        """
        if str(path) == ":memory:":
            raise ValueError("ProfileStore 要做持久化，请传文件路径；临时画像请用 InMemoryProfileStore")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 复用 database 模块统一的 SQLiteClient：自动 WAL / 外键 / 参数化 / 标识符校验
        self._db = SQLiteClient(str(self.path))
        # 幂等建表，表结构与原手写 CREATE TABLE 完全一致，已有数据无缝兼容
        self._db.ensure_table("user_profiles", {
            "user_id":    "TEXT PRIMARY KEY",
            "profile":    "TEXT NOT NULL",
            "source":     "TEXT NOT NULL",
            "updated_at": "TEXT NOT NULL",
        })

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须是非空字符串")

    def get(self, user_id: str) -> UserProfile:
        self._validate_user_id(user_id)
        row = self._db.select_one("user_profiles", {"user_id": user_id})
        if row is None:
            return UserProfile(user_id=user_id)
        try:
            data = json.loads(row["profile"] or "{}")
        except json.JSONDecodeError:
            data = {}
        data["user_id"] = user_id
        return UserProfile(**data)

    def merge(
        self, user_id: str, patch: Dict[str, Any], *, source: str = "manual"
    ) -> UserProfile:
        self._validate_user_id(user_id)
        profile = self.get(user_id)
        profile.merge_patch(patch or {})
        profile.updated_at = cstnow()
        # 保留原 INSERT OR REPLACE 语义，行为和原实现完全一致
        self._db.execute(
            "INSERT OR REPLACE INTO user_profiles(user_id, profile, source, updated_at)"
            " VALUES(?,?,?,?)",
            (
                user_id,
                json.dumps(profile.to_dict(), ensure_ascii=False),
                source,
                profile.updated_at.isoformat(),
            ),
        )
        return profile

    def close(self) -> None:
        self._db.close()
        return None


class InMemoryProfileStore(ProfileStore):
    """纯内存版画像存储，用于单元测试 / 不想落盘的场景

    进程退出即丢，接口与 `SQLiteProfileStore` 完全一致，可直接互换。
    """

    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._sources: Dict[str, str] = {}

    def get(self, user_id: str) -> UserProfile:
        profile = self._profiles.get(user_id)
        if profile is None:
            return UserProfile(user_id=user_id)
        # 返回副本，避免调用方直接改到内部状态
        return UserProfile(**profile.to_dict())

    def merge(
        self, user_id: str, patch: Dict[str, Any], *, source: str = "manual"
    ) -> UserProfile:
        profile = self.get(user_id)
        profile.merge_patch(patch or {})
        profile.updated_at = cstnow()
        self._profiles[user_id] = UserProfile(**profile.to_dict())
        self._sources[user_id] = source
        return self._profiles[user_id]

    def clear(self) -> None:
        self._profiles.clear()
        self._sources.clear()

    def close(self) -> None:
        return None
