"""
Skill数据模型定义
【C6-2】新增ScriptParam / ScriptSchema，支持脚本参数Schema声明与自动校验
【C15】新增assets字段：静态资产清单（图片/模板/样例数据等，相对路径+大小+类型）
【C18】方案B调整：scripts_schema外置到scripts/<脚本名>.schema.json，沙箱三字段上移到loader全局配置，不再出现在SkillMetadata中
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ScriptParam:
    """【C6-2】单个脚本参数的Schema定义
    
    Attributes:
        name: 参数名（对应命令行 --name）
        type: 参数类型 str / int / float / bool / enum
        required: 是否必填
        default: 默认值
        description: 参数说明
        enum: 枚举允许值列表（type=enum时使用）
        min/max: int/float类型的数值上下界
        max_length: str类型的最大长度（防注入超长参数）
    """
    name: str
    type: str = "str"
    required: bool = False
    default: Any = None
    description: str = ""
    enum: Optional[List[Any]] = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    max_length: Optional[int] = None
    
    def validate_value(self, value: Any) -> tuple[bool, str, Any]:
        """校验并转换单个参数值，返回 (ok, 错误信息, 转换后的值)"""
        # None且有默认值时使用默认值
        if value is None:
            if self.default is not None:
                return True, "", self.default
            if self.required:
                return False, f"参数 '{self.name}' 是必填项", None
            return True, "", None
        
        # 类型转换与校验
        try:
            if self.type == "int":
                converted = int(value)
                if self.min is not None and converted < self.min:
                    return False, f"参数 '{self.name}' 最小值为 {self.min}", None
                if self.max is not None and converted > self.max:
                    return False, f"参数 '{self.name}' 最大值为 {self.max}", None
            elif self.type == "float":
                converted = float(value)
                if self.min is not None and converted < self.min:
                    return False, f"参数 '{self.name}' 最小值为 {self.min}", None
                if self.max is not None and converted > self.max:
                    return False, f"参数 '{self.name}' 最大值为 {self.max}", None
            elif self.type == "bool":
                if isinstance(value, bool):
                    converted = value
                else:
                    s = str(value).strip().lower()
                    if s in ("true", "1", "yes", "y", "on"):
                        converted = True
                    elif s in ("false", "0", "no", "n", "off"):
                        converted = False
                    else:
                        return False, f"参数 '{self.name}' 必须是布尔值（true/false/1/0）", None
            elif self.type == "enum":
                converted = str(value)
                if self.enum and converted not in self.enum:
                    return False, f"参数 '{self.name}' 必须是以下值之一：{', '.join(str(v) for v in self.enum)}", None
            elif self.type == "str":
                converted = str(value)
                if self.max_length is not None and len(converted) > self.max_length:
                    return False, f"参数 '{self.name}' 长度不能超过 {self.max_length} 字符", None
                # 防注入：禁止参数中出现换行/分号/&/|等命令注入字符
                _DANGEROUS_CHARS = ['\n', '\r', ';', '&', '|', '`', '$', '(', ')', '<', '>', '\\']
                for ch in _DANGEROUS_CHARS:
                    if ch in converted:
                        return False, f"参数 '{self.name}' 包含非法字符 '{ch}'（命令注入防护）", None
            else:
                converted = str(value)
        except (ValueError, TypeError) as e:
            return False, f"参数 '{self.name}' 类型错误，期望 {self.type}", None
        
        return True, "", converted


@dataclass
class ScriptSchema:
    """【C6-2】单个脚本的参数Schema定义
    
    Attributes:
        script_name: 脚本文件名（如 export_report.py）
        description: 脚本功能说明
        params: 参数列表
        timeout: 超时秒数（覆盖全局默认）
    """
    script_name: str
    description: str = ""
    params: List[ScriptParam] = field(default_factory=list)
    timeout: Optional[int] = None
    
    def validate_args(self, args: Optional[Dict[str, Any]]) -> tuple[bool, List[str], Dict[str, Any]]:
        """校验并转换所有参数，返回 (ok, 错误列表, 转换后参数)"""
        errors: List[str] = []
        converted: Dict[str, Any] = {}
        args = args or {}
        
        param_map = {p.name: p for p in self.params}
        
        # 1. 检查必填参数是否都传了
        for p in self.params:
            if p.required and p.name not in args:
                if p.default is None:
                    errors.append(f"缺少必填参数 '{p.name}'")
        
        # 2. 检查传入的未知参数
        for k in args.keys():
            if k not in param_map:
                # 未知参数默认放行但给出警告，不阻断执行
                converted[k] = args[k]
        
        # 3. 逐个校验已知参数
        for p in self.params:
            val = args.get(p.name)
            ok, err, cval = p.validate_value(val)
            if not ok:
                errors.append(err)
            elif cval is not None:
                converted[p.name] = cval
            elif p.default is not None and p.name not in args:
                converted[p.name] = p.default
        
        return len(errors) == 0, errors, converted


@dataclass
class SkillMetadata:
    """Skill元数据，对应SKILL.md的YAML frontmatter
    
    【C12】新增依赖字段：
        requires: 依赖的其他技能列表（字符串列表，支持版本约束如 "other-skill>=1.2.0"）
        requires_tools: 依赖的工具名列表（除了allowed_tools声明外，运行时必须存在的工具）
        min_fast_agents_version: 最低需要的fast_agents框架版本
        max_fast_agents_version: 最高兼容的fast_agents框架版本（可选）
    
    【C18方案B调整】脚本参数Schema、沙箱配置全部从元数据中移除：
        - scripts_schema: 外置到scripts/<脚本名>.schema.json，加载技能时扫描scripts目录读取
        - allow_network/allow_write_paths/env_whitelist: 上移到SkillLoader初始化全局配置，宿主统一管理
        一级元数据仅保留name/description等核心字段，对齐Anthropic Agent Skills规范，常驻token控制在~100/skill
    """
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)            # 【C12】技能依赖
    requires_tools: List[str] = field(default_factory=list)     # 【C12】工具依赖
    min_fast_agents_version: Optional[str] = None               # 【C12】最低框架版本
    max_fast_agents_version: Optional[str] = None               # 【C12】最高兼容框架版本
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
        
        # 【C12】requires字段格式校验
        for dep in self.requires:
            if not isinstance(dep, str):
                errors.append(f"requires 项必须是字符串，收到 {type(dep).__name__}")
                continue
            # 简单校验格式：skill_name 或 skill_name>=x.y.z / skill_name==x.y.z / skill_name<=x.y.z
            if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\s*(>=|<=|==|>|<)\s*\d+(\.\d+(\.\d+)?)?)?$', dep):
                errors.append(f"requires 项 '{dep}' 格式不合法，应为 'skill-name' 或 'skill-name>=1.0.0'")
        
        # 【C12】版本号格式校验
        for ver_field_name, ver_val in [
            ("min_fast_agents_version", self.min_fast_agents_version),
            ("max_fast_agents_version", self.max_fast_agents_version),
        ]:
            if ver_val is not None:
                if not re.match(r'^\d+(\.\d+){0,2}$', str(ver_val)):
                    errors.append(f"{ver_field_name} 版本号格式不合法，应为 x.y.z 格式")
        
        # version字段本身校验
        if self.version and not re.match(r'^\d+(\.\d+){0,2}(-[a-zA-Z0-9.]+)?$', self.version):
            errors.append(f"version 格式不合法，应为语义化版本号如 1.0.0")
        
        return errors
    
    def to_index_text(self) -> str:
        """生成用于常驻上下文的索引文本（Level 1），每个Skill约100token"""
        return f"- **{self.name}** [{self.skill_type}]: {self.description}"

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
            "requires": self.requires,
            "requires_tools": self.requires_tools,
            "min_fast_agents_version": self.min_fast_agents_version,
            "max_fast_agents_version": self.max_fast_agents_version,
            "metadata": self.metadata,
            "skill_type": self.skill_type,
            "source": self.source,
        }


@dataclass
class SkillLoadResult:
    """Skill加载结果
    
    【C18方案B】scripts_schemas存放scripts/目录下扫描到的参数Schema（<脚本名>.schema.json解析得到）
    """
    success: bool
    skill_name: str
    metadata: Optional[SkillMetadata] = None
    content: Optional[str] = None
    instructions: Optional[str] = None
    scripts: List[Path] = field(default_factory=list)
    references: List[Path] = field(default_factory=list)
    assets: List[Tuple[str, int, bool]] = field(default_factory=list)  # 【C15】静态资产清单（相对路径，大小，是否二进制）
    scripts_schemas: Dict[str, ScriptSchema] = field(default_factory=dict)  # 【C18】外置脚本参数Schema（脚本名 -> Schema）
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_script_schema(self, script_name: str) -> Optional[ScriptSchema]:
        """【C18】获取指定脚本的Schema，支持模糊匹配（不带扩展名也能匹配）"""
        if script_name in self.scripts_schemas:
            return self.scripts_schemas[script_name]
        # 模糊匹配：去掉扩展名
        name_no_ext = Path(script_name).stem
        for sname, schema in self.scripts_schemas.items():
            if Path(sname).stem == name_no_ext:
                return schema
        return None
    
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
            ])
            for p in self.scripts:
                script_line = f"- {p.name}"
                # 如果有对应schema，追加参数说明
                schema = self.get_script_schema(p.name)
                if schema:
                    if schema.description:
                        script_line += f": {schema.description}"
                    if schema.timeout:
                        script_line += f"（超时 {schema.timeout}s）"
                    parts.append(script_line)
                    # 列出参数
                    if schema.params:
                        for param in schema.params:
                            req_mark = "*" if param.required else " "
                            type_mark = f"({param.type})"
                            default_mark = f"，默认: {param.default}" if param.default is not None else ""
                            enum_mark = f"，可选: {param.enum}" if param.enum else ""
                            parts.append(f"  {req_mark} --{param.name} {type_mark}: {param.description}{default_mark}{enum_mark}")
                else:
                    parts.append(script_line)
                    parts.append(f"  （无参数schema，调用时参考正文说明）")
        
        if self.references:
            parts.extend([
                f"",
                f"## 参考文档",
                *[f"- {p.name}" for p in self.references]
            ])
        
        # 【C15】静态资产清单：区分文本/二进制，提示LLM二进制文件只能通过read_skill_asset获取路径/元信息，不要读正文
        if self.assets:
            text_assets = [n for n, _sz, is_bin in self.assets if not is_bin]
            bin_assets = [n for n, _sz, is_bin in self.assets if is_bin]
            parts.append("")
            parts.append("## 静态资源（assets/）")
            if text_assets:
                parts.append("### 文本资源（可用read_skill_asset读取内容）")
                parts.extend(f"- {n}" for n in text_assets)
            if bin_assets:
                if text_assets:
                    parts.append("")
                parts.append("### 二进制资源（图片/字体/二进制数据，不可直接返回内容，read_skill_asset仅返回元信息与绝对路径，可交由脚本使用）")
                parts.extend(f"- {n}" for n in bin_assets)
        
        return "\n".join(parts)
