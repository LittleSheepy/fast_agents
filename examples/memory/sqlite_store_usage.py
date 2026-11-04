"""memory 模块 SQLite 后端使用示例（J3：已接入 database.SQLiteClient）。

演示 SQLiteMemoryStore 的整行覆盖写入、命中记账（含作用域隔离）、修订留痕，
以及 SQLiteSessionStore 的短期会话持久化与摘要 upsert。全部落在临时目录，跑完自动清理。
运行方式：在项目根目录执行 `python examples/memory/sqlite_store_usage.py`
"""

import os
import tempfile

from fast_agents.memory import MemoryRecord, SQLiteMemoryStore, SQLiteSessionStore
from fast_agents.messages import Message


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "memory.db")

        # ---------- 1. 长期记忆：写入 / 读取 ----------
        store = SQLiteMemoryStore(db_path, user_id="u1", session_id="s1")
        r = store.add(MemoryRecord(content="用户喜欢 Python", kind="fact", source="chat"))
        print("[1] add 一条 ->", store.get(r.memory_id).content)

        # ---------- 2. 整行覆盖（等价原 INSERT OR REPLACE）----------
        r2 = MemoryRecord(
            memory_id=r.memory_id, user_id="u1", session_id="s1",
            content="用户喜欢 Python 和 Rust", kind="fact", source="chat",
        )
        store.add(r2)
        print("[2] 同 id 再 add（整行覆盖）->", store.get(r.memory_id).content)

        # ---------- 3. 命中记账：IN 批量 + 不动 updated_at ----------
        before = store.get(r.memory_id).updated_at
        n = store.touch([r.memory_id])
        got = store.get(r.memory_id)
        print(
            f"[3] touch 命中 {n} 条 -> access_count={got.access_count}, "
            f"updated_at 未变={got.updated_at == before}"
        )

        # ---------- 4. 作用域隔离：别人的 id 不会被记账 ----------
        other = SQLiteMemoryStore(db_path, user_id="u2", session_id="s1")
        ro = other.add(MemoryRecord(content="u2 的记忆", kind="fact", source="chat"))
        hit = store.touch([r.memory_id, ro.memory_id])   # 故意混入 u2 的 id
        print(
            f"[4] 混入他人 id -> 只命中 {hit} 条；u2 的 access_count="
            f"{other.get(ro.memory_id).access_count}"
        )

        # ---------- 5. 修订：旧版 superseded，历史保留 ----------
        store.revise(r.memory_id, "用户喜欢 Python、Rust", source="chat")
        print(
            f"[5] revise -> 有效记忆 {len(store.all())} 条，"
            f"含历史共 {len(store.all(include_inactive=True))} 条"
        )

        # ---------- 6. 短期会话：落盘 + 摘要（upsert 覆盖）----------
        ss = SQLiteSessionStore(db_path, user_id="u1", session_id="s1")
        ss.append(Message(role="user", content="你好"))
        ss.append(Message(role="assistant", content="你好呀"))
        ss.save_summary("第一版摘要")
        ss.save_summary("第二版摘要")          # 走 upsert，覆盖上一版
        print(f"[6] 会话消息 {[m.content for m in ss.load()]}；摘要={ss.load_summary()!r}")

        # ---------- 7. 清档 ----------
        ss.clear()
        store.clear()
        print(f"[7] clear 后：会话 {ss.load()}，长期记忆 {store.all()}")

        store.close()
        other.close()
        ss.close()


if __name__ == "__main__":
    main()
