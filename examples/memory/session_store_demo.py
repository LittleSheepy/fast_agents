"""memory.session_store 快速上手示例（短期会话持久化）。

演示 SessionStore 的最小闭环：
  1. InMemorySessionStore：最小用法（纯内存，进程退出即丢）；
  2. SQLiteSessionStore：append 落盘、按 seq 保序回读；
  3. 作用域隔离：user_id / session_id 不同则互不可见；
  4. BufferMemory 集成：只多传一个 persist_store 参数即获得落盘能力；
  5. 摘要持久化：save_summary / load_summary（同 key 覆盖）；
  6. 跨进程恢复：关掉再打开同一个 .db，历史消息与摘要自动回填
     （真实进程重启与这里的「close 后重新构造」等价：数据只在 .db 文件里）；
  7. clear：一键清空本会话（连带摘要），不影响其他会话。

运行方式：在项目根目录执行
    python examples/memory/session_store_demo.py

无需任何 API Key，完全离线可跑；跑完自动删除示例产生的 .db 文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python examples/memory/xxx.py 运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents.memory import (  # noqa: E402
    BufferMemory,
    InMemorySessionStore,
    SQLiteSessionStore,
)
from fast_agents.messages import Message  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "_demo_sessions.db"


def _title(n: int, text: str) -> None:
    print(f"\n{'=' * 62}\n[{n}] {text}\n{'=' * 62}")


def main() -> None:
    # ---------- 1. InMemorySessionStore：最小用法 ----------
    _title(1, "InMemorySessionStore：最小用法（纯内存，不落盘）")
    mem_store = InMemorySessionStore()
    mem_store.append(Message("user", "今天天气怎么样？"))
    mem_store.append(Message("assistant", "上海今天多云，18~24℃。"))
    mem_store.save_summary("用户问了天气")
    print(f"    load() 回读 {len(mem_store.load())} 条消息：")
    for m in mem_store.load():
        print(f"      {m.role:>9}: {m.content}")
    print(f"    load_summary(): {mem_store.load_summary()}")
    print("    注意：进程退出后这些数据就没了（内存实现）。")

    # ---------- 2. SQLiteSessionStore：落盘 ----------
    _title(2, "SQLiteSessionStore：append 落盘、按 seq 保序回读")
    DB_PATH.unlink(missing_ok=True)
    store = SQLiteSessionStore(DB_PATH, user_id="u1", session_id="s1")
    print(f"    落盘文件: {DB_PATH.name}")
    store.append(Message("user", "帮我记一下：周五下午三点开会"))
    store.append(Message("assistant", "好的，已记住：周五 15:00 开会。"))
    store.append(Message("user", "顺便记下会议室是 3 楼 A 厅"))
    for m in store.load():
        print(f"      {m.role:>9}: {m.content}")
    print("    ↑ 顺序与 append 完全一致（seq 自增保序，不依赖时间戳）")

    # ---------- 3. 作用域隔离 ----------
    _title(3, "作用域隔离：user_id / session_id 不同则互不可见")
    other = SQLiteSessionStore(DB_PATH, user_id="u2", session_id="s1")
    print(f"    同一文件、另一个 user（u2）看到的消息数: {len(other.load())}")
    other.append(Message("user", "我是 u2 的消息"))
    print(f"    写入 u2 之后，u1 仍看到: {len(store.load())} 条（未被串扰）")
    other.close()

    # ---------- 4. BufferMemory 集成 ----------
    _title(4, "BufferMemory 集成：多传一个 persist_store 即获得落盘能力")
    buf_store = SQLiteSessionStore(DB_PATH, user_id="u1", session_id="s2")
    # enable_summary=False → 不需要 LLM，超出窗口的老消息直接丢弃
    buf = BufferMemory(max_messages=4, enable_summary=False, persist_store=buf_store)
    for i in range(1, 5):
        buf.append_user(f"第 {i} 轮用户提问")
        buf.append_ai(f"第 {i} 轮助手回答")
    print(f"    append 8 条（窗口上限 4），当前内存窗口: {len(buf)} 条")
    print(f"    但落盘保留全部: {len(buf_store.load())} 条")
    print("    ↑ 窗口只裁剪「送进上下文」的部分，落盘是完整历史")

    # ---------- 5. 摘要持久化 ----------
    _title(5, "摘要持久化：save_summary / load_summary")
    print("    摘要由 BufferMemory 压缩时自动落盘；这里手动演示读写：")
    buf_store.save_summary("用户在连续追问某个话题")
    print(f"    load_summary(): {buf_store.load_summary()}")
    buf_store.save_summary("摘要被更新了（upsert，同 key 覆盖）")
    print(f"    再次 load_summary(): {buf_store.load_summary()}")

    # ---------- 6. 跨进程恢复 ----------
    _title(6, "跨进程恢复：关掉再打开同一个 .db，自动回填")
    before = len(buf_store.load())
    buf_store.close()                       # 模拟进程退出
    print(f"    close() 前落盘消息数: {before}")
    # 重新构造一个全新的 store：真实重启等价于此（数据只在 .db 文件里）
    reopened = SQLiteSessionStore(DB_PATH, user_id="u1", session_id="s2")
    buf2 = BufferMemory(max_messages=4, enable_summary=False, persist_store=reopened)
    print(f"    重新构造后：落盘 {len(reopened.load())} 条，"
          f"摘要「{reopened.load_summary()}」，窗口回填 {len(buf2)} 条")
    print("    ↑ 历史与摘要都来自磁盘，不是内存残留")
    reopened.close()

    # ---------- 7. clear ----------
    _title(7, "clear()：一键清空本会话（连带摘要）")
    cleaner = SQLiteSessionStore(DB_PATH, user_id="u1", session_id="s2")
    cleaner.clear()
    print(f"    clear 后消息数: {len(cleaner.load())}，摘要: 「{cleaner.load_summary()}」")
    print(f"    其他会话不受影响（u1/s1 仍有 {len(store.load())} 条）")
    cleaner.close()
    store.close()

    # 收尾：删除示例产生的 .db（含 WAL / SHM 附属文件）
    for suffix in ("", "-wal", "-shm"):
        Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
    print(f"\n示例结束，已清理临时文件 {DB_PATH.name}")


if __name__ == "__main__":
    main()
