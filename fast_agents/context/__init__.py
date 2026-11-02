"""context 模块 —— 上下文工程（token 预算 / 分层拼装 / 压缩 / 卸载 / 目录浏览）

分层与数据流：

    system + 记忆 + 历史 + 文件 + 本轮问题
        ↓  ContextBuilder.build()          ← 分层拼装
    ContextWindow（按层序排列 + 带预算）
        ↓  ContextCompressor.compress()    ← 超预算则：卸载 → 裁剪 → 摘要
    Messages → LLM
        ↓  工具返回超长结果
    ContextOffloader.offload()             ← 落盘，只留引用；模型可 read_offloaded 回读

对外只暴露三个概念：
- **计量**：``TokenCounter`` / ``TokenBudget`` —— 先知道有多少，才谈得上怎么裁；
- **分层**：``ContextWindow`` / ``ContextLayer`` —— 静态在前、动态在后，缓存友好；
- **治理**：``ContextCompressor`` / ``ContextOffloader`` —— 三级策略 + 按需回读。

外加一组让 Agent 自己管理上下文的工具（``build_context_tools``），
与 ``memory_tool`` / ``mcp_tool`` 同层级、同风格。

文件读写编辑（``read_file`` / ``write_file`` / ``edit_file`` / ``multi_edit_file``）
统一由 ``tools/builtin/file_tool.py`` 提供（带乐观锁、原子写入、备份、审批标记）。
``ReActAgent._init_context`` 会在注册完本模块工具后自动补齐 file_tool 四件套。
"""

from .builder import BuildReport, ContextBuilder
from .compressor import CompressionReport, ContextCompressor, DEFAULT_COMPACT_PROMPT
from .layers import LAYER_ORDER, ContextLayer, ContextWindow
from .offload import ContextOffloader, OffloadRecord
from .tokens import TokenBudget, TokenCounter, count_message_tokens, estimate_text_tokens
from .tools import (
    ListFilesTool,
    ReadOffloadedTool,
    build_context_tools,
)

__all__ = [
    # 计量
    "TokenCounter", "TokenBudget", "estimate_text_tokens", "count_message_tokens",
    # 分层
    "ContextLayer", "ContextWindow", "LAYER_ORDER",
    # 治理
    "ContextCompressor", "CompressionReport", "DEFAULT_COMPACT_PROMPT",
    "ContextOffloader", "OffloadRecord",
    # 门面
    "ContextBuilder", "BuildReport",
    # 工具
    "ReadOffloadedTool", "ListFilesTool",
    "build_context_tools",
]

__version__ = "1.1.0"
