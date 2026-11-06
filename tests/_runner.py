"""零依赖测试运行器 + 最小化测试工具。

只提供本仓库测试真实用到的能力，不复刻 pytest 的全部功能：
    - run_all(ns)：收集并运行 ns 中所有 test_ 开头的函数，按名字排序；
        * 自动识别参数签名，对应名字的本地函数若是 generator 则作为 yield fixture
          （yield 前构造、yield 后 teardown，和 pytest 的 @pytest.fixture 语义一致）；
        * 本地函数若名为 tmp_path/monkeypatch 则直接由运行器提供，不必自己定义。
    - run_prefix(ns, prefix)：只跑 ns 中以 prefix 开头的测试，供 AI 新写测试时快速验证。
    - raises(exc, match=None)：断言代码块抛出指定异常，用法 with raises(ValueError): ...
    - importorskip(name)：可选依赖缺失时抛出 Skip，run_all 会把它计为 skipped 而非 fail。
    - tmp_path：返回一个 with 可用的临时目录上下文，退出自动清理。
    - monkeypatch：返回一个轻量 MonkeyPatch 上下文，setattr/delattr/setitem/delitem 自动还原。
"""
from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path


# ==================== Skip 标记（可选依赖未安装时跳过，不计失败） ====================

class Skip(Exception):
    """用 raise Skip(...) 标记跳过，run_all 会把它当成 skipped。"""

    def __init__(self, reason: str = ""):
        super().__init__(reason)
        self.reason = reason


def importorskip(name: str):
    """尝试 import 模块，失败则 raise Skip（等价 pytest.importorskip，但不依赖 pytest）。"""
    try:
        return __import__(name)
    except ImportError as e:
        raise Skip(f"未安装可选依赖 {name}，跳过：{e}")


# ==================== raises ====================

class _RaisesCtx:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type if isinstance(exc_type, tuple) else (exc_type,)
        self.match = match
        self.exception = None
        self.value = None  # 兼容 pytest 用法 ei.value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, _tb):
        if exc_type is None:
            raise AssertionError(
                f"期望抛出 {'/'.join(e.__name__ for e in self.exc_type)}，但没有抛出任何异常"
            )
        if not issubclass(exc_type, self.exc_type):
            return False  # 不吞，外层看到真实异常
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(f"异常信息期望包含 {self.match!r}，实际为 {str(exc_val)!r}")
        self.exception = exc_val
        self.value = exc_val
        return True


def raises(exc_type, match=None):
    return _RaisesCtx(exc_type, match)


# ==================== tmp_path ====================

@contextmanager
def tmp_path():
    """临时目录上下文，退出递归删除。返回 pathlib.Path。"""
    d = tempfile.mkdtemp(prefix="fa_test_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ==================== monkeypatch（轻量） ====================

class _MonkeyPatch:
    def __init__(self):
        self._undos = []

    def setattr(self, obj, name, value):
        old = getattr(obj, name, _MISSING)
        setattr(obj, name, value)
        self._undos.append(("attr", obj, name, old))

    def delattr(self, obj, name):
        old = getattr(obj, name, _MISSING)
        object.__delattr__(obj, name)
        self._undos.append(("attr", obj, name, old))

    def setitem(self, mapping, key, value):
        existed = key in mapping
        old = mapping[key] if existed else _MISSING
        mapping[key] = value
        self._undos.append(("item", mapping, key, old, existed))

    def delitem(self, mapping, key):
        existed = key in mapping
        old = mapping[key] if existed else _MISSING
        if existed:
            del mapping[key]
        self._undos.append(("item", mapping, key, old, existed))

    def undo(self):
        for entry in reversed(self._undos):
            kind = entry[0]
            if kind == "attr":
                _, obj, name, old = entry
                if old is _MISSING:
                    try:
                        object.__delattr__(obj, name)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, name, old)
            elif kind == "item":
                _, mapping, key, old, existed = entry
                if existed:
                    mapping[key] = old
                else:
                    mapping.pop(key, None)
        self._undos.clear()


_MISSING = object()


@contextmanager
def monkeypatch():
    mp = _MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()


# ==================== 收集与运行 ====================

def _collect_tests(ns, prefix=None):
    items = []
    for name in sorted(ns.keys()):
        if not name.startswith("test_") or not callable(ns[name]):
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        items.append((name, ns[name]))
    return items


def _build_kwargs(fn, ns):
    """根据测试函数签名构造 kwargs：fixture 从 ns 取 generator，tmp_path/monkeypatch 由本模块提供。"""
    sig = inspect.signature(fn)
    kwargs = {}
    enter_cms = []      # with 上下文，跑完后 __exit__（teardown）
    for pname, param in sig.parameters.items():
        if pname == "tmp_path":
            cm = tmp_path()
            enter_cms.append(cm)
            kwargs[pname] = cm.__enter__()
        elif pname == "monkeypatch":
            cm = monkeypatch()
            enter_cms.append(cm)
            kwargs[pname] = cm.__enter__()
        else:
            fixture_fn = ns.get(pname)
            if fixture_fn is None or not callable(fixture_fn):
                raise TypeError(f"fixture {pname!r} 未在模块中定义")
            gen = fixture_fn()
            kwargs[pname] = next(gen)
            # 记录 teardown（倒序执行）
            enter_cms.append(_GeneratorCtx(gen))
    return kwargs, enter_cms


class _GeneratorCtx:
    """把 generator fixture 包装成上下文，__exit__ 时继续 next() 触发 teardown。"""
    def __init__(self, gen):
        self.gen = gen
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            next(self.gen)
        except StopIteration:
            pass
        else:
            # fixture 里 yield 超过一次，按 pytest 惯例忽略
            pass
        return False


def _run(items, label):
    total = len(items)
    passed = failed = skipped = 0
    errors = []
    print(f"=== {label}（共 {total} 项） ===")
    for name, fn in items:
        cms = []
        try:
            kwargs, cms = _build_kwargs(fn, fn.__globals__)
            fn(**kwargs)
            passed += 1
            print(f"  ✅ PASS: {name}")
        except Skip as s:
            skipped += 1
            print(f"  ⏭  SKIP: {name}  [{s.reason}]")
        except AssertionError as e:
            failed += 1
            errors.append((name, e))
            print(f"  ❌ FAIL: {name}  {e}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  💥 ERROR: {name}  {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            # 倒序 teardown（tmp_path / monkeypatch / generator fixtures）
            for cm in reversed(cms):
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    traceback.print_exc()
    print(f"--- 完成：{passed} 通过，{skipped} 跳过，{failed} 失败 ---")
    if errors:
        print("失败详情：")
        for name, err in errors:
            print(f"\n  [{name}] {type(err).__name__}: {err}")
        return 1
    return 0


def run_all(namespace):
    """跑当前模块里全部 test_* 函数。"""
    items = _collect_tests(namespace)
    return _run(items, "开始运行测试")


def run_prefix(namespace, prefix):
    """只跑以 prefix 开头的 test_* 函数。"""
    items = _collect_tests(namespace, prefix=prefix)
    return _run(items, f"运行前缀 {prefix!r} 的测试")


if __name__ == "__main__":
    raise SystemExit(0)
