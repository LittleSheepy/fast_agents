"""记忆模块测试：不依赖真实 LLM，用假 LLM 验证记忆闭环

覆盖：
1.
"""

import os
import re
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fast_agents.memory import (
    BufferMemory,
    ChromaMemoryStore,
    InMemoryProfileStore,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    SemanticMemoryStore,
    SQLiteMemoryStore,
    SQLiteProfileStore,
    SQLiteSessionStore,
    UserProfile,
    lexical_score,
    merge_similarity,
)
from fast_agents.tools.builtin import (
    RememberTool, RecallTool, ForgetTool, UpdateProfileTool,
)
from fast_agents.messages.message import Message


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """假 LLM：invoke 永远返回固定内容，用来测试不依赖网络的路径"""

    def __init__(self, reply="（假模型回复）"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return FakeMessage(self.reply)


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
