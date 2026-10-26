"""
Skill模块 - 为fast_agents提供可复用的技能能力
遵循Anthropic开放SKILL.md标准，支持渐进式加载
"""

from .base import BaseSkill, skill
from .registry import SkillRegistry
from .models import SkillMetadata, SkillLoadResult
from .loader import SkillLoader
from .middleware import SkillMiddleware

__all__ = [
    "BaseSkill",
    "skill",
    "SkillRegistry",
    "SkillMetadata",
    "SkillLoadResult",
    "SkillLoader",
    "SkillMiddleware",
]

__version__ = "1.0.0"
