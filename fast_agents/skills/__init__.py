"""
Skill模块 - 为fast_agents提供可复用的技能能力
遵循Anthropic开放SKILL.md标准，支持渐进式加载
【C15】静态资产(assets)读取工具
【C8】同名技能冲突消解（按来源优先级覆盖）
"""

from .base import BaseSkill, skill
from .errors import (
    AssetNotFoundError,
    SkillError, SkillNotFoundError, SkillLoadError, SkillValidationError,
    SkillDependencyError, SkillCircularDependencyError, SkillConflictError,
    ScriptExecutionError, ReferenceNotFoundError,
    ScriptParamValidationError, ScriptSandboxViolationError,
)
from .loader import SkillLoader
from .middleware import SkillMiddleware
from .models import SkillMetadata, SkillLoadResult, ScriptParam, ScriptSchema
from .registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "skill",
    "SkillRegistry",
    "SkillMetadata",
    "SkillLoadResult",
    "SkillLoader",
    "SkillMiddleware",
    # 【C6-2】脚本参数Schema
    "ScriptParam",
    "ScriptSchema",
    # 异常类
    "SkillError",
    "SkillNotFoundError",
    "SkillLoadError",
    "SkillValidationError",
    "SkillDependencyError",
    "SkillCircularDependencyError",
    "SkillConflictError",  # 【C8】
    "ScriptExecutionError",
    "ReferenceNotFoundError",
    "AssetNotFoundError",  # 【C15】
    "ScriptParamValidationError",
    "ScriptSandboxViolationError",
]
