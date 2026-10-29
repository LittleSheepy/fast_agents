"""file_tool 模块单元测试

运行：python test/test_file_tool.py

覆盖场景：
  1. 工厂函数返回四个工具 + 共享缓存
  2. 基本读写与 artifact 元数据
  3. overwrite=False 拦截已存在文件
  4. 乐观锁命中（hash 匹配）写入成功
  5. 乐观锁未命中（外部已改）返回失败 + 最新元数据
  6. 原子写入：成功和失败场景都不留 tmp 文件
  7. 备份文件命名：.bak.<timestamp> 与同秒自动编号
  8. MultiEdit 一处失败则整体回滚（原文件不动）
  9. MultiEdit dry_run 不落盘
 10. 路径越权拦截（.. 与绝对路径）
 11. EditTool old_text 必须唯一，非唯一报错
 12. 写入后联动失效 ReadTool 缓存
 13. MultiEdit 接受 dict 形式 edits（LLM 友好）
"""

import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.tools.builtin.file_tool import (
    ReadTool, WriteTool, EditTool, MultiEditTool, build_file_tools,
)


# ─────────────────────────────────────────────────────────
# 1. 工厂函数
# ─────────────────────────────────────────────────────────

def test_1_build_file_tools_factory():
    """build_file_tools 应返回 4 个工具，且后三者共享同一个 ReadTool 缓存引用"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, editor, multi = build_file_tools(ws)
        assert isinstance(reader, ReadTool)
        assert isinstance(writer, WriteTool)
        assert isinstance(editor, EditTool)
        assert isinstance(multi, MultiEditTool)

        # 共享同一个 ReadTool 实例，缓存联动生效
        assert writer._read_tool is reader
        assert editor._read_tool is reader
        assert multi._read_tool is reader

        # 工具名
        assert reader.name == "read_file"
        assert writer.name == "write_file"
        assert editor.name == "edit_file"
        assert multi.name == "multi_edit_file"
    print("✅ 1. build_file_tools 工厂返回 4 工具且共享 ReadTool 通过")


# ─────────────────────────────────────────────────────────
# 2. 基本读写 + artifact
# ─────────────────────────────────────────────────────────

def test_2_read_write_basic():
    """新建文件 → 写入 → 读取，artifact 应含 size/mtime/content_hash 等元数据"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, _ = build_file_tools(ws)

        # 写一个新文件（不传 expected_*）
        content = "line1\nline2\nline3\n"
        r = writer.invoke(path="a.txt", content=content)
        assert r.succeeded, f"写入失败: {r.output}"
        art = r.artifact
        assert art is not None
        assert art["path"] == "a.txt"
        assert art["exists"] is True
        assert art["size"] == len(content.encode("utf-8"))
        assert art["line_count"] == 3
        assert len(art["content_hash"]) == 16  # sha256 前 16 位

        # 文件确实落盘
        assert (Path(ws) / "a.txt").exists()
        assert (Path(ws) / "a.txt").read_text(encoding="utf-8") == content

        # 读取
        r = reader.invoke(path="a.txt")
        assert r.succeeded, f"读取失败: {r.output}"
        # 读取 artifact 包含 start/stop/total_lines
        art = r.artifact
        assert art["start"] == 1
        assert art["stop"] == 3
        assert art["total_lines"] == 3
        # 行号标注
        assert "line1" in r.output and "1|" in r.output
    print("✅ 2. 基本读写 + artifact 元数据通过")


# ─────────────────────────────────────────────────────────
# 3. overwrite=False 拦截
# ─────────────────────────────────────────────────────────

def test_3_overwrite_false_rejects_existing():
    """overwrite=False 时，已存在的文件应被拒绝写入；新文件允许创建"""
    with tempfile.TemporaryDirectory() as ws:
        _, writer, _, _ = build_file_tools(ws)

        # 先创建
        r = writer.invoke(path="exist.txt", content="v1", overwrite=False)
        assert r.succeeded, f"新建应成功: {r.output}"

        # overwrite=False 再写 → 应失败
        r = writer.invoke(path="exist.txt", content="v2", overwrite=False)
        assert r.failed
        assert "overwrite=False" in r.output or "拒绝覆盖" in r.output
        # 原内容不变
        assert (Path(ws) / "exist.txt").read_text() == "v1"

        # overwrite=True 正常覆盖（不带锁）
        r = writer.invoke(path="exist.txt", content="v2", overwrite=True)
        assert r.succeeded
        assert (Path(ws) / "exist.txt").read_text() == "v2"
    print("✅ 3. overwrite=False 拦截已存在文件通过")


# ─────────────────────────────────────────────────────────
# 4. 乐观锁命中（hash 匹配）→ 写入成功
# ─────────────────────────────────────────────────────────

def test_4_optimistic_lock_hit_succeeds():
    """先用 reader 读文件拿 hash，再把 hash 作为 expected_hash 传入 writer → 应写入成功"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, _ = build_file_tools(ws)

        # 初始化
        writer.invoke(path="b.txt", content="version-1\n")

        # 读 → 拿到凭证
        r = reader.invoke(path="b.txt")
        assert r.succeeded
        h = r.artifact["content_hash"]
        s = r.artifact["size"]
        m = r.artifact["mtime"]

        # 正确 hash 写入 → 成功
        r = writer.invoke(
            path="b.txt", content="version-2\n",
            expected_hash=h, expected_size=s, expected_mtime=m,
        )
        assert r.succeeded, f"正确凭证应写入成功: {r.output}"
        assert (Path(ws) / "b.txt").read_text() == "version-2\n"

        # artifact 返回新 hash（与旧不同）
        assert r.artifact["content_hash"] != h
    print("✅ 4. 乐观锁命中（hash 匹配）写入成功通过")


# ─────────────────────────────────────────────────────────
# 5. 乐观锁未命中（外部已改）→ 失败 + 返回最新元数据
# ─────────────────────────────────────────────────────────

def test_5_optimistic_lock_miss_fails():
    """外部进程/其他工具已改动文件后，用旧 hash 写入应失败，并返回最新元数据"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, _ = build_file_tools(ws)

        writer.invoke(path="c.txt", content="original\n")
        r = reader.invoke(path="c.txt")
        old_hash = r.artifact["content_hash"]
        old_size = r.artifact["size"]

        # 模拟「别人」改了文件
        time.sleep(0.01)  # 确保 mtime 变化
        (Path(ws) / "c.txt").write_text("hacked-by-others\n", encoding="utf-8")
        new_hash_from_disk = reader.invoke(path="c.txt").artifact["content_hash"]
        assert new_hash_from_disk != old_hash  # 前置条件

        # 用旧 hash 写入 → 必须失败
        r = writer.invoke(
            path="c.txt", content="overwrite-attempt\n",
            expected_hash=old_hash, expected_size=old_size,
        )
        assert r.failed, "外部已改时乐观锁应拦截"
        assert "乐观锁冲突" in r.output
        # 文件保持被改后的版本，没有被覆盖
        assert (Path(ws) / "c.txt").read_text() == "hacked-by-others\n"
    print("✅ 5. 乐观锁未命中（外部已改）拦截成功通过")


# ─────────────────────────────────────────────────────────
# 6. 原子写入：不留 tmp 文件（成功 + 失败两种场景）
# ─────────────────────────────────────────────────────────

def test_6_atomic_write_no_tmp_left():
    """写入成功时工作目录下不应残留 .xxx.tmp 文件；写入失败（二进制扩展名）tmp 也应被清理"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, _ = build_file_tools(ws)
        ws_path = Path(ws)

        # 成功写入
        r = writer.invoke(path="clean.txt", content="hello\n")
        assert r.succeeded

        # 不应有任何 .tmp 文件残留
        leftovers = list(ws_path.glob(".*.tmp")) + list(ws_path.glob("*.tmp"))
        assert leftovers == [], f"原子写入应清理 tmp，但残留: {leftovers}"

        # 写二进制扩展名（应失败）→ 也不应有 tmp
        r = writer.invoke(path="should_fail.png", content="not an image")
        assert r.failed
        leftovers = list(ws_path.glob(".*.tmp")) + list(ws_path.glob("*.tmp"))
        assert leftovers == [], f"失败路径也应清理 tmp，但残留: {leftovers}"
        assert not (ws_path / "should_fail.png").exists()
    print("✅ 6. 原子写入无 tmp 残留（成功+失败）通过")


# ─────────────────────────────────────────────────────────
# 7. 备份文件命名：.bak.<timestamp> 与同秒自动编号
# ─────────────────────────────────────────────────────────

def test_7_backup_naming_pattern():
    """backup=True 时应生成 <name>.bak.YYYYMMDD-HHMMSS 格式备份；同秒多次备份自动加 -N 后缀"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, editor, _ = build_file_tools(ws)
        ws_path = Path(ws)

        # 先写入初始内容
        writer.invoke(path="note.txt", content="v1\n")
        r1 = reader.invoke(path="note.txt")
        h1 = r1.artifact["content_hash"]

        # 第一次备份写入
        r = writer.invoke(path="note.txt", content="v2\n", expected_hash=h1, backup=True)
        assert r.succeeded

        # 备份文件存在，命名符合格式
        backups = sorted(ws_path.glob("note.txt.bak.*"))
        assert len(backups) == 1, f"应存在 1 个备份，实际: {backups}"
        name = backups[0].name
        # 形如 note.txt.bak.20261029-143025
        import re
        assert re.match(r"^note\.txt\.bak\.\d{8}-\d{6}$", name), f"备份命名不符合: {name}"
        assert backups[0].read_text() == "v1\n"  # 备份是 v1 内容

        # 同秒内再做两次备份（利用 editor）→ 应产生 -1 / -2 后缀
        # 注意：为了同秒命中编号逻辑，直接在 1 秒内快速做两次编辑
        r2 = reader.invoke(path="note.txt")
        h2 = r2.artifact["content_hash"]
        r = editor.invoke(path="note.txt", old_text="v2", new_text="v3",
                          expected_hash=h2, backup=True)
        assert r.succeeded

        r3 = reader.invoke(path="note.txt")
        h3 = r3.artifact["content_hash"]
        r = editor.invoke(path="note.txt", old_text="v3", new_text="v4",
                          expected_hash=h3, backup=True)
        assert r.succeeded

        # 至少有 3 个备份（v1/v2/v3），新增的要么是标准时间戳，要么是 -N 后缀
        all_backups = sorted(ws_path.glob("note.txt.bak.*"))
        assert len(all_backups) >= 3, f"应至少 3 个备份，实际: {all_backups}"
        for b in all_backups:
            assert re.match(
                r"^note\.txt\.bak\.\d{8}-\d{6}(-\d+)?$", b.name
            ), f"备份命名不符合: {b.name}"

        # 最终文件是 v4
        assert ws_path.joinpath("note.txt").read_text() == "v4\n"
    print("✅ 7. 备份文件命名（时间戳 + 同秒 -N 后缀）通过")


# ─────────────────────────────────────────────────────────
# 8. MultiEdit 一处失败 → 整体回滚，原文件不动
# ─────────────────────────────────────────────────────────

def test_8_multi_edit_rollback_on_failure():
    """MultiEdit 任意一条 old_text 找不到时，整批放弃，原文件内容、大小、hash 完全不变"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, multi = build_file_tools(ws)
        ws_path = Path(ws)

        original = "name: Alice\nage: 30\ncity: Shanghai\n"
        writer.invoke(path="user.yaml", content=original)
        before = reader.invoke(path="user.yaml").artifact
        before_mtime = ws_path.joinpath("user.yaml").stat().st_mtime
        time.sleep(0.01)  # 给 mtime 变化留窗口，以便后续判断是否落盘

        # 第 1 条能匹配，第 2 条故意写错（不存在）→ 必须整体回滚
        r = multi.invoke(
            path="user.yaml",
            edits=[
                {"old_text": "Alice", "new_text": "Bob"},      # 能匹配
                {"old_text": "NOT_EXIST_KEY", "new_text": "x"},  # 失败
            ],
            backup=True,  # 即使开启备份，回滚时也不应生成备份
        )
        assert r.failed, f"第 2 条找不到应失败: {r.output}"
        assert "整批已回滚" in r.output

        # 原文件一字未动
        assert ws_path.joinpath("user.yaml").read_text() == original
        after_stat = ws_path.joinpath("user.yaml").stat()
        assert after_stat.st_size == before["size"], "文件大小应保持不变"
        assert after_stat.st_mtime == before_mtime, "文件 mtime 应保持不变（未落盘）"

        # 不应有任何备份文件生成（回滚 = 未落盘 = 不应备份）
        assert list(ws_path.glob("user.yaml.bak.*")) == []
    print("✅ 8. MultiEdit 一处失败则整体回滚（原文件不动、无备份）通过")


# ─────────────────────────────────────────────────────────
# 9. MultiEdit dry_run 不落盘
# ─────────────────────────────────────────────────────────

def test_9_multi_edit_dry_run_no_change():
    """dry_run=True 时，即使全部 old_text 都能匹配，也不应写入磁盘，mtime/size 不变，无备份"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, _, multi = build_file_tools(ws)
        ws_path = Path(ws)

        original = "a=1\nb=2\nc=3\n"
        writer.invoke(path="cfg.txt", content=original)
        before_stat = ws_path.joinpath("cfg.txt").stat()
        before_hash = reader.invoke(path="cfg.txt").artifact["content_hash"]

        # dry-run：三条都能匹配
        r = multi.invoke(
            path="cfg.txt",
            edits=[
                {"old_text": "a=1", "new_text": "a=10"},
                {"old_text": "b=2", "new_text": "b=20"},
                {"old_text": "c=3", "new_text": "c=30"},
            ],
            dry_run=True,
            backup=True,
        )
        assert r.succeeded, f"dry-run 应通过预校验: {r.output}"
        assert "dry-run 通过" in r.output
        assert r.artifact["dry_run"] is True
        assert r.artifact["edit_count"] == 3

        # 文件内容、大小、mtime 都不变
        assert ws_path.joinpath("cfg.txt").read_text() == original
        after_stat = ws_path.joinpath("cfg.txt").stat()
        assert after_stat.st_size == before_stat.st_size
        assert after_stat.st_mtime == before_stat.st_mtime
        assert reader.invoke(path="cfg.txt").artifact["content_hash"] == before_hash

        # 无备份文件
        assert list(ws_path.glob("cfg.txt.bak.*")) == []
    print("✅ 9. MultiEdit dry_run 不落盘（mtime/大小/hash 不变）通过")


# ─────────────────────────────────────────────────────────
# 10. 路径越权拦截（.. 与绝对路径）
# ─────────────────────────────────────────────────────────

def test_10_path_traversal_blocked():
    """.. 跳出 workspace、绝对路径指向外部，都应被四个工具拦截"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, editor, multi = build_file_tools(ws)
        ws_path = Path(ws)

        # 先在 workspace 外建一个文件，看会不会被读/写
        outside = ws_path.parent / "outside_secret.txt"
        outside.write_text("TOP SECRET", encoding="utf-8")

        # reader 用 .. 越权
        r = reader.invoke(path="../outside_secret.txt")
        assert r.failed and "工作目录外" in r.output
        # reader 绝对路径
        r = reader.invoke(path=str(outside))
        assert r.failed and "工作目录外" in r.output

        # writer 用 .. 越权
        r = writer.invoke(path="../hack.txt", content="bad")
        assert r.failed and "工作目录外" in r.output
        assert not (ws_path.parent / "hack.txt").exists(), "越权写入应被拒绝"

        # writer 绝对路径
        r = writer.invoke(path=str(ws_path.parent / "hack2.txt"), content="bad")
        assert r.failed and "工作目录外" in r.output
        assert not (ws_path.parent / "hack2.txt").exists()

        # editor 越权（先造个合法文件，再尝试用 .. 路径去改外部）
        writer.invoke(path="d.txt", content="hello")
        r = editor.invoke(path="../outside_secret.txt", old_text="TOP", new_text="X")
        assert r.failed and "工作目录外" in r.output

        # multi 越权
        r = multi.invoke(path="../outside_secret.txt",
                         edits=[{"old_text": "TOP", "new_text": "X"}])
        assert r.failed and "工作目录外" in r.output

        # 清理外部文件
        outside.unlink()
    print("✅ 10. 路径越权拦截（.. 和绝对路径均拒绝）通过")


# ─────────────────────────────────────────────────────────
# 11. EditTool old_text 必须唯一
# ─────────────────────────────────────────────────────────

def test_11_edit_requires_unique_match():
    """EditTool 在 old_text 出现多次时应失败，并指出多处位置；出现 0 次也失败"""
    with tempfile.TemporaryDirectory() as ws:
        _, writer, editor, _ = build_file_tools(ws)

        writer.invoke(path="dup.txt", content="foo\nfoo\nbar\n")

        # 出现 2 次 → 失败
        r = editor.invoke(path="dup.txt", old_text="foo", new_text="FOO")
        assert r.failed
        assert "出现了 2 次" in r.output

        # 出现 0 次 → 失败
        r = editor.invoke(path="dup.txt", old_text="nonexistent", new_text="X")
        assert r.failed
        assert "未找到" in r.output

        # 唯一匹配 → 成功
        r = editor.invoke(path="dup.txt", old_text="bar", new_text="BAR")
        assert r.succeeded
        assert "替换成功" in r.output
    print("✅ 11. EditTool old_text 唯一性校验通过")


# ─────────────────────────────────────────────────────────
# 12. 写入后联动失效 ReadTool 缓存
# ─────────────────────────────────────────────────────────

def test_12_cache_invalidation_after_write():
    """通过 WriteTool/EditTool 写入后，ReadTool 的内容缓存应失效，下一次 read 能看到新内容"""
    with tempfile.TemporaryDirectory() as ws:
        reader, writer, editor, _ = build_file_tools(ws)

        writer.invoke(path="cache.txt", content="v1\n")

        # 第一次 read：把内容装进缓存
        r1 = reader.invoke(path="cache.txt", use_cache=True)
        assert r1.succeeded and "v1" in r1.output

        # 直接从磁盘被其他方改了（模拟外部写入）—— 这种情况缓存会陈旧，
        # 但通过 writer/editor 走的写入会主动 invalidate
        h1 = r1.artifact["content_hash"]
        r = editor.invoke(path="cache.txt", old_text="v1", new_text="v2",
                          expected_hash=h1)
        assert r.succeeded

        # 再次 read，use_cache=True 时也必须读到新内容（缓存已被失效）
        r2 = reader.invoke(path="cache.txt", use_cache=True)
        assert "v2" in r2.output
        assert "v1" not in r2.output
    print("✅ 12. 写入后联动失效 ReadTool 缓存通过")


# ─────────────────────────────────────────────────────────
# 13. MultiEdit 接受 dict 形式 edits（LLM 友好）
# ─────────────────────────────────────────────────────────

def test_13_multi_edit_accepts_dict_edits():
    """LLM 常直接输出 dict 列表，MultiEdit 应正常接受并执行"""
    with tempfile.TemporaryDirectory() as ws:
        _, writer, _, multi = build_file_tools(ws)

        writer.invoke(path="dict.txt", content="alpha beta gamma\n")

        r = multi.invoke(
            path="dict.txt",
            edits=[
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "beta",  "new_text": "BETA"},
            ],
            backup=False,
        )
        assert r.succeeded
        result_text = (Path(ws) / "dict.txt").read_text()
        assert result_text == "ALPHA BETA gamma\n"
        assert r.artifact["edit_count"] == 2
    print("✅ 13. MultiEdit 接受 dict 形式 edits 通过")


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_1_build_file_tools_factory,
        test_2_read_write_basic,
        test_3_overwrite_false_rejects_existing,
        test_4_optimistic_lock_hit_succeeds,
        test_5_optimistic_lock_miss_fails,
        test_6_atomic_write_no_tmp_left,
        test_7_backup_naming_pattern,
        test_8_multi_edit_rollback_on_failure,
        test_9_multi_edit_dry_run_no_change,
        test_10_path_traversal_blocked,
        test_11_edit_requires_unique_match,
        test_12_cache_invalidation_after_write,
        test_13_multi_edit_accepts_dict_edits,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n🎉 完成：{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(0 if failed == 0 else 1)
