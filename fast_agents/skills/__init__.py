"""
Skill模块 - 为fast_agents提供可复用的技能能力
遵循Anthropic开放SKILL.md标准，支持渐进式加载
"""

from .base import BaseSkill, skill
from .errors import SkillError, SkillNotFoundError, SkillLoadError, SkillValidationError, ScriptExecutionError, ReferenceNotFoundError
from .loader import SkillLoader
from .middleware import SkillMiddleware
from .models import SkillMetadata, SkillLoadResult
from .registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "skill",
    "SkillRegistry",
    "SkillMetadata",
    "SkillLoadResult",
    "SkillLoader",
    "SkillMiddleware",
    # 新增异常类
    "SkillError",
    "SkillNotFoundError",
    "SkillLoadError",
    "SkillValidationError",
    "ScriptExecutionError",
    "ReferenceNotFoundError",
]

