"""
Skill数据模型定义
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SkillMetadata:
    """Skill元数据，对应SKILL.md的YAML frontmatter"""
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时字段（不来自frontmatter）
    skill_path: Optional[Path] = None
    skill_type: str = "file"  # file / code
    source: str = "project"  # project / user / code / plugin
    
    def validate(self) -> List[str]:
        """校验元数据合法性，返回错误列表"""
        errors = []
        
        # name校验规则：小写字母、数字、连字符，1-64字符，不能以连字符开头/结尾，不能连续连字符
        if not self.name:
            errors.append("name不能为空")
        else:
            if len(self.name) > 64:
                errors.append(f"name长度不能超过64字符，当前{len(self.name)}")
            if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', self.name):
                errors.append("name只能包含小写字母、数字、连字符(-)，不能以连字符开头/结尾，不能有连续连字符")
            if '--' in self.name:
                errors.append("name不能包含连续连字符(--)")
        
        # description校验
        if not self.description:
            errors.append("description不能为空")
        elif len(self.description) > 1024:
            errors.append(f"description长度不能超过1024字符，当前{len(self.description)}")
        
        return errors
    
    def to_index_text(self) -> str:
        """生成用于常驻上下文的索引文本（Level 1），每个Skill约100token"""
        return f"- {self.name}: {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "compatibility": self.compatibility,
            "allowed_tools": self.allowed_tools,
            "tags": self.tags,
            "metadata": self.metadata,
            "skill_type": self.skill_type,
            "source": self.source,
        }


@dataclass
class SkillLoadResult:
    """Skill加载结果"""
    success: bool
    skill_name: str
    metadata: Optional[SkillMetadata] = None
    content: Optional[str] = None
    instructions: Optional[str] = None
    scripts: List[Path] = field(default_factory=list)
    references: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def full_prompt(self) -> str:
        """获取完整的Skill提示词（Level 2加载后使用）"""
        if not self.success or not self.instructions:
            return ""
        
        parts = [
            f"# Skill: {self.skill_name}",
            f"",
            f"## 技能说明",
            f"{self.metadata.description if self.metadata else ''}",
            f"",
            f"## 执行指令",
            f"{self.instructions}",
        ]
        
        if self.scripts:
            parts.extend([
                f"",
                f"## 可用脚本",
                *[f"- {p.name}" for p in self.scripts]
            ])
        
        if self.references:
            parts.extend([
                f"",
                f"## 参考文档",
                *[f"- {p.name}" for p in self.references]
            ])
        
        return "\n".join(parts)
