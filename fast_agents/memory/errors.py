"""memory 模块异常体系"""

from __future__ import annotations


class MemoryError(Exception):
    """memory 模块异常基类。"""


class MemoryConflictError(MemoryError):
    """写入新记忆时检测到与已有记忆冲突时抛出（当前版本未对外抛出，保留给未来显式冲突场景使用）。

    属性：
    - new_record：试图写入的新记忆（尚未入库）
    - conflicts：[(existing_record, conflict_type), ...]，列出所有冲突的旧记忆
    """

    def __init__(self, new_record: "MemoryRecord", conflicts: list):
        self.new_record = new_record
        self.conflicts = conflicts
        types = sorted({ctype for _, ctype in conflicts})
        preview = "; ".join(f"{ctype}:{rec.content[:30]}" for rec, ctype in conflicts[:3])
        super().__init__(
            f"记忆冲突（{','.join(types)}），新内容「{new_record.content[:30]}」"
            f"与已有 {len(conflicts)} 条记忆冲突：{preview}"
        )
