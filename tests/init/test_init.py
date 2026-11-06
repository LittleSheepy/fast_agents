"""顶层包（fast_agents/__init__.py）的基础测试（无需 pytest）。

只验证「包本身的结构」，不碰业务流程，全部用标准库断言：
    - 版本与元信息（__version__ / __author__ / __email__ / __description__）
    - __all__ 声明规范（存在、非空、无重复、全是字符串）
    - __all__ 中每个符号都能从包顶层访问到（防「声明了但没 import」）
    - 主要子模块都能正常导入（防 __init__ 导入链断裂）

运行方式（Windows PowerShell，在项目根目录）:
    python tests\\init\\test_init.py
    （无需安装 pytest，纯标准库）
"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fast_agents


# ==================== 受测常量 ====================

# 顶层包声明的所有子模块（新增模块时同步维护这一份清单）
SUB_MODULES = [
    "fast_agents.agents",
    "fast_agents.context",
    "fast_agents.database",
    "fast_agents.llm",
    "fast_agents.mcp",
    "fast_agents.memory",
    "fast_agents.messages",
    "fast_agents.skills",
    "fast_agents.tools",
    "fast_agents.trace",
    "fast_agents.utils",
    "fast_agents.version",
]


# ==================== 版本 / 元信息 ====================

def test_version_is_nonempty_string():
    """__version__ 存在且是非空字符串。"""
    assert isinstance(fast_agents.__version__, str), "__version__ 不是字符串"
    assert fast_agents.__version__.strip(), "__version__ 为空"


def test_version_matches_version_module():
    """包顶层暴露的版本应与 fast_agents.version 里的一致（单一来源，别两处各写一份）。"""
    from fast_agents import version as version_module

    assert fast_agents.__version__ == version_module.__version__, (
        f"__init__ 的版本 {fast_agents.__version__} 与 version.py 的 "
        f"{version_module.__version__} 不一致"
    )


def test_package_metadata_present():
    """作者 / 邮箱 / 描述都已定义且非空。"""
    for attr in ("__author__", "__email__", "__description__"):
        value = getattr(fast_agents, attr, None)
        assert isinstance(value, str) and value.strip(), f"{attr} 缺失或为空"


def test_installed_metadata_version_matches():
    """若包已以 fast-agents 安装，则安装元数据版本应与 __version__ 一致。

    这些测试是源码直跑（靠 sys.path 引导）的，没 pip install 时直接 return 跳过。
    """
    from tests._runner import Skip
    try:
        installed = metadata.version("fast-agents")
    except metadata.PackageNotFoundError:
        raise Skip("未以 fast-agents 安装，跳过安装元数据校验")
    assert installed == fast_agents.__version__, (
        f"pyproject 里的版本 {installed} 与 __version__ {fast_agents.__version__} 不一致"
    )


# ==================== __all__ 规范 ====================

def test_all_is_well_formed():
    """__all__ 存在、非空、全是字符串、且没有重复项。"""
    all_names = getattr(fast_agents, "__all__", None)
    assert isinstance(all_names, list) and all_names, "__all__ 缺失、不是 list 或为空"
    assert all(isinstance(n, str) for n in all_names), "__all__ 里必须全是字符串"
    duplicates = sorted({n for n in all_names if all_names.count(n) > 1})
    assert not duplicates, f"__all__ 存在重复项：{duplicates}"


def test_all_symbols_are_accessible():
    """__all__ 里声明的每个名字，都能从 fast_agents 顶层直接拿到。"""
    missing = [n for n in fast_agents.__all__ if not hasattr(fast_agents, n)]
    assert not missing, f"__all__ 声明了但顶层访问不到：{missing}"


# ==================== 子模块导入 ====================

def test_submodules_importable():
    """各主要子模块可正常导入（导入链没断）。"""
    for module_name in SUB_MODULES:
        assert importlib.import_module(module_name) is not None, f"{module_name} 导入失败"


# ==================== 入口 ====================

def main():
    from tests._runner import run_all
    raise SystemExit(run_all(globals()))


def main_of_new_add():
    """新添加的测试在这里跑
    1.跑新加的测试
    2.修改bug可以跑针对性测试
    """
    # 人工添加测针对测试：AI不要删除覆盖
    pass

    # AI新添加的测试(AI可以删除覆盖)
    pass


if __name__ == "__main__":
    print("")
    main_of_new_add()
    main()
