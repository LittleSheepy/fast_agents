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


class SkillDependencyError(SkillError):
    """【C12】技能依赖缺失（工具不存在 / 技能依赖未加载 / 版本不兼容）"""
    def __init__(self, skill_name: str, missing_tools: list[str] = None,
                 missing_skills: list[str] = None,
                 version_conflicts: list[str] = None,
                 details: str = ""):
        self.skill_name = skill_name
        self.missing_tools = missing_tools or []
        self.missing_skills = missing_skills or []
        self.version_conflicts = version_conflicts or []
        parts = []
        if self.missing_tools:
            parts.append(f"缺失工具: {', '.join(self.missing_tools)}")
        if self.missing_skills:
            parts.append(f"依赖技能未加载: {', '.join(self.missing_skills)}")
        if self.version_conflicts:
            parts.append(f"版本不兼容: {'; '.join(self.version_conflicts)}")
        if details:
            parts.append(details)
        msg = f"Skill '{skill_name}' 依赖检查失败: " + "; ".join(parts) if parts else f"Skill '{skill_name}' 依赖检查失败"
        super().__init__(msg)


class SkillCircularDependencyError(SkillError):
    """【C12】技能循环依赖"""
    def __init__(self, chain: list[str]):
        self.chain = chain
        super().__init__(f"检测到技能循环依赖: {' -> '.join(chain)}")


class ScriptExecutionError(SkillError):
    """脚本执行失败"""
    def __init__(self, skill_name: str, script_name: str, reason: str):
        self.skill_name = skill_name
        self.script_name = script_name
        self.reason = reason
        super().__init__(f"Execute script '{script_name}' in skill '{skill_name}' failed: {reason}")


class ScriptParamValidationError(SkillError):
    """【C6-2】脚本参数校验失败（类型不匹配/缺必填/越界/枚举不合法）"""
    def __init__(self, skill_name: str, script_name: str, errors: list[str]):
        self.skill_name = skill_name
        self.script_name = script_name
        self.errors = errors
        super().__init__(f"Script '{script_name}' 参数校验失败: {'; '.join(errors)}")


class ScriptSandboxViolationError(SkillError):
    """【C6-2】脚本沙箱违规（访问网络/写入未授权路径/非法命令）"""
    def __init__(self, skill_name: str, script_name: str, reason: str):
        self.skill_name = skill_name
        self.script_name = script_name
        self.reason = reason
        super().__init__(f"Script '{script_name}' 触发沙箱拦截: {reason}")


class ReferenceNotFoundError(SkillError):
    """参考文档不存在"""
    def __init__(self, skill_name: str, ref_name: str):
        self.skill_name = skill_name
        self.ref_name = ref_name
        super().__init__(f"Reference '{ref_name}' not found in skill '{skill_name}'")


class AssetNotFoundError(SkillError):
    """【C15】静态资产不存在"""
    def __init__(self, skill_name: str, asset_name: str):
        self.skill_name = skill_name
        self.asset_name = asset_name
        super().__init__(f"Asset '{asset_name}' not found in skill '{skill_name}'")


class SkillConflictError(SkillError):
    """【C8】技能同名冲突：多个来源存在同名技能，按优先级覆盖时抛出警告"""
    def __init__(self, skill_name: str, old_source: str, new_source: str, old_path: str, new_path: str):
        self.skill_name = skill_name
        self.old_source = old_source
        self.new_source = new_source
        self.old_path = old_path
        self.new_path = new_path
        msg = (f"技能 '{skill_name}' 存在同名冲突："
               f"[{old_source}] {old_path} 被 [{new_source}] {new_path} 覆盖（按来源优先级规则）")
        super().__init__(msg)
