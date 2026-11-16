"""
Skill模块异常定义
统一基类 + 具体子类，不直接抛底层原生异常给上层
"""
from __future__ import annotations


class SkillError(Exception):
    """Skill模块所有异常的基类"""
    pass


class SkillNotFoundError(SkillError):
    """技能不存在"""
    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        super().__init__(f"Skill '{skill_name}' not found")


class SkillLoadError(SkillError):
    """技能加载失败"""
    def __init__(self, skill_name: str, reason: str):
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Load skill '{skill_name}' failed: {reason}")


class SkillValidationError(SkillError):
    """技能元数据校验失败"""
    def __init__(self, skill_name: str, errors: list[str]):
        self.skill_name = skill_name
        self.errors = errors
        super().__init__(f"Skill '{skill_name}' validation failed: {'; '.join(errors)}")


class ScriptExecutionError(SkillError):
    """脚本执行失败"""
    def __init__(self, skill_name: str, script_name: str, reason: str):
        self.skill_name = skill_name
        self.script_name = script_name
        self.reason = reason
        super().__init__(f"Execute script '{script_name}' in skill '{skill_name}' failed: {reason}")


class ReferenceNotFoundError(SkillError):
    """参考文档不存在"""
    def __init__(self, skill_name: str, ref_name: str):
        self.skill_name = skill_name
        self.ref_name = ref_name
        super().__init__(f"Reference '{ref_name}' not found in skill '{skill_name}'")
