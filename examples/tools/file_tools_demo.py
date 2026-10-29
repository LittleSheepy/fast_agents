"""file_tool 使用示例：乐观锁驱动的协作写文件

运行：python examples/tools/file_tools_demo.py

这个示例**不需要任何 API key 或 LLM**，用纯本地代码模拟一个真实场景：
  你（Agent）和同事（"外部进程"）轮流编辑同一份 todo.md，
  通过乐观锁保证不会在不知道的情况下互相覆盖。

演示链路：
  1. 创建 todo.md                  → write_file（overwrite=False 防重名）
  2. 读取 todo.md                  → read_file，拿到 hash/size 凭证
  3. 带锁编辑第 1 项               → edit_file（expected_hash=刚读到的 hash），成功
  4. 模拟「同事」悄悄改了第 3 项    → 直接 Path.write_text 绕过工具（模拟并发）
  5. 用旧 hash 再编辑              → 乐观锁**拦截**，返回失败 + 当前元数据
  6. 重新读取 + 正确编辑           → read_file 拿新 hash → edit_file 成功
  7. MultiEdit dry_run 预检查      → 不落盘，只验证 3 处替换都可唯一匹配
  8. MultiEdit 批量替换落盘        → 原子写回，任何一处失败都整批回滚
  9. 检查备份文件                  → 演示 .bak.<timestamp> 自动留底
"""

import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.tools.builtin.file_tool import build_file_tools


def section(title: str) -> None:
    print("\n" + "─" * 68)
    print(f"  {title}")
    print("─" * 68)


def show_result(label: str, r) -> None:
    icon = "✅" if r.succeeded else "❌"
    print(f"{icon} {label}")
    for line in r.output.splitlines():
        print(f"     {line}")
    if r.artifact:
        keys = ["size", "line_count", "content_hash", "edit_count", "dry_run"]
        shown = {k: r.artifact[k] for k in keys if k in r.artifact}
        if shown:
            print(f"     → artifact: {shown}")


def main() -> None:
    with tempfile.TemporaryDirectory() as ws_str:
        ws = Path(ws_str)
        print(f"📂 工作目录：{ws}")

        reader, writer, editor, multi = build_file_tools(str(ws), enable_backup=True)

        # ── 1. 创建文件 ─────────────────────────────────
        section("1. 创建 todo.md（overwrite=False，已存在则拒绝）")
        initial = (
            "# TODO\n"
            "\n"
            "- [ ] 写文档\n"
            "- [ ] 写测试\n"
            "- [ ] 写示例\n"
        )
        r = writer.invoke(path="todo.md", content=initial, overwrite=False)
        show_result("write_file(overwrite=False)", r)
        assert r.succeeded

        # 再创建同名文件应被拦截
        r = writer.invoke(path="todo.md", content="duplicate", overwrite=False)
        show_result("重复创建（应失败）", r)
        assert r.failed

        # ── 2. 读取文件拿凭证 ──────────────────────────
        section("2. 读取 todo.md，拿到乐观锁凭证（hash/size/mtime）")
        r = reader.invoke(path="todo.md")
        show_result("read_file", r)
        meta = r.artifact
        print(f"   🗝️  凭证：hash={meta['content_hash']} size={meta['size']}")

        # ── 3. 带锁编辑第 1 项 ─────────────────────────
        section("3. 带锁把「写文档」改成「写文档 ✅」")
        r = editor.invoke(
            path="todo.md",
            old_text="- [ ] 写文档",
            new_text="- [x] 写文档",
            expected_hash=meta["content_hash"],
            expected_size=meta["size"],
            backup=True,
        )
        show_result("edit_file(带 expected_hash)", r)
        assert r.succeeded

        # ── 4. 模拟并发：同事悄悄改了第 3 项 ────────────
        section("4. （模拟并发）同事直接往磁盘写了「写示例 ✅」")
        time.sleep(0.05)  # 让 mtime 变化
        current_text = ws.joinpath("todo.md").read_text(encoding="utf-8")
        hacked = current_text.replace("- [ ] 写示例", "- [x] 写示例（同事）")
        ws.joinpath("todo.md").write_text(hacked, encoding="utf-8")
        print("     💥 磁盘文件已被外部改动（没走 edit_file）")

        # ── 5. 用旧 hash 再编辑 → 应被拦截 ─────────────
        section("5. 用旧 hash 再编辑「写测试」→ 乐观锁拦截（防止覆盖同事）")
        stale_hash = meta["content_hash"]
        r = editor.invoke(
            path="todo.md",
            old_text="- [ ] 写测试",
            new_text="- [x] 写测试",
            expected_hash=stale_hash,
        )
        show_result("edit_file(旧 hash，应失败)", r)
        assert r.failed
        assert "乐观锁冲突" in r.output

        # ── 6. 重新读取 + 正确编辑 ─────────────────────
        section("6. 重新读取拿新 hash，再编辑「写测试」→ 成功")
        r = reader.invoke(path="todo.md")
        show_result("read_file（重新读取）", r)
        new_meta = r.artifact
        print(f"   🗝️  新凭证：hash={new_meta['content_hash']} size={new_meta['size']}")

        r = editor.invoke(
            path="todo.md",
            old_text="- [ ] 写测试",
            new_text="- [x] 写测试（review by me）",
            expected_hash=new_meta["content_hash"],
            expected_size=new_meta["size"],
            backup=True,
        )
        show_result("edit_file(新 hash)", r)
        assert r.succeeded

        # ── 7. MultiEdit dry_run 预检查 ───────────────
        section("7. MultiEdit dry_run：一次性改标题 + 加结尾，只预检查不落盘")
        r = multi.invoke(
            path="todo.md",
            edits=[
                {"old_text": "# TODO",         "new_text": "# TODO（2026-10-29 更新）"},
                {"old_text": "（review by me）", "new_text": "（review by me ✅）"},
                {"old_text": "（同事）",         "new_text": "（同事 ✅）"},
            ],
            dry_run=True,
        )
        show_result("multi_edit(dry_run=True)", r)
        assert r.succeeded and r.artifact["dry_run"] is True
        # dry-run 后文件不应被改动——再次读一下确认
        before_hash = reader.invoke(path="todo.md").artifact["content_hash"]

        # ── 8. MultiEdit 真正落盘 ─────────────────────
        section("8. MultiEdit 真正落盘（原子写回，3 处一次性改完）")
        r = multi.invoke(
            path="todo.md",
            edits=[
                {"old_text": "# TODO",         "new_text": "# TODO（2026-10-29 更新）"},
                {"old_text": "（review by me）", "new_text": "（review by me ✅）"},
                {"old_text": "（同事）",         "new_text": "（同事 ✅）"},
            ],
            backup=True,
        )
        show_result("multi_edit(正式写入)", r)
        assert r.succeeded
        after_hash = reader.invoke(path="todo.md").artifact["content_hash"]
        assert after_hash != before_hash, "正式写入后 hash 应变化"

        # ── 9. 最终文件 + 备份 ────────────────────────
        section("9. 最终文件内容 + 自动产生的备份")
        final = ws.joinpath("todo.md").read_text(encoding="utf-8")
        print("────── todo.md ──────")
        for line in final.splitlines():
            print(f"   {line}")
        print("────── end ──────────")

        backups = sorted(ws.glob("todo.md.bak.*"))
        print(f"\n📦 工作目录下的备份文件（{len(backups)} 个）：")
        for b in backups:
            st = b.stat()
            print(f"   - {b.name}  ({st.st_size} 字节)")

        print("\n🎉 示例全部跑通。")
        print("   关键结论：乐观锁让 Agent 在「读-思考-写」之间即便文件被外部改动，")
        print("   也不会静默覆盖；备份文件则给了「误改回滚」的最后一道保险。")


if __name__ == "__main__":
    main()
