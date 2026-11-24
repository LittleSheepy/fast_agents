"""
Skill加载器：负责扫描、解析、加载文件型Skill（SKILL.md格式）
遵循Anthropic开放标准，兼容Windows和Unix路径
【C6-2】脚本沙箱增强：参数Schema校验、类型自动转换、命令注入防护、网络/路径访问白名单
【C15】静态资产(assets)读取：assets目录扫描+路径穿越防护+文本/二进制分流
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .errors import (
    AssetNotFoundError,
    SkillLoadError, SkillNotFoundError,
    ScriptExecutionError, ReferenceNotFoundError,
    ScriptParamValidationError, ScriptSandboxViolationError,
)
from .models import SkillMetadata, SkillLoadResult, ScriptParam, ScriptSchema

# Trace 相关延迟导入（避免 skills → trace 反向强依赖；不传 tracer 时零开销）
_SpanKind = None
_SpanStatus = None


def _ensure_trace_imports() -> None:
    """延迟导入 SpanKind/SpanStatus，保证不传 tracer 时不触发 trace 模块加载。"""
    global _SpanKind, _SpanStatus
    if _SpanKind is None or _SpanStatus is None:
        from fast_agents.trace import SpanKind, SpanStatus
        _SpanKind, _SpanStatus = SpanKind, SpanStatus


# 【C15】可作为文本直接读取的资产扩展名；其余按二进制处理
_TEXT_ASSET_EXTENSIONS = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".xml", ".html", ".htm", ".css", ".js", ".py", ".bat", ".cmd",
    ".ps1", ".ini", ".cfg", ".toml", ".log", ".sql",
}
# 【C15】单个文本资产的最大字节数（防止把大文件塞进上下文）
_MAX_TEXT_ASSET_BYTES = 200 * 1024  # 200KB
# 【C15】二进制资产预览字节数（默认不返回内容，仅返回元信息与绝对路径）
_BIN_ASSET_PREVIEW_BYTES = 0


class SkillLoader:
    """
    Skill加载器：扫描目录，解析SKILL.md，加载技能内容
    
    支持扫描路径（优先级从高到低）:
        - 代码型注册/自定义路径(custom/code): 最高优先级，后注册覆盖前者
        - 项目级: ./.skills/ (project)
        - 用户级: ~/.fast_agents/skills/ (user)
        - 框架内置: fast_agents包内的.skills/ (builtin)
    【C8】同名技能冲突时，高优先级来源覆盖低优先级，打印警告日志
    """
    
    SKILL_FILENAMES = ["SKILL.md", "skill.md", "Skills.md", "skills.md"]
    SCRIPTS_DIR = "scripts"
    REFERENCES_DIR = "references"
    ASSETS_DIR = "assets"
    
    # 【C8】技能来源优先级：数值越大优先级越高，冲突时高优先级覆盖低优先级
    SOURCE_PRIORITY = {
        "builtin": 0,   # 框架内置，最低优先级
        "user": 1,      # 用户全局目录
        "project": 2,   # 当前项目目录
        "custom": 3,    # 显式add_skill_path添加的自定义路径
        "code": 4,      # 代码型注册(BaseSkill/@skill)，最高优先级
    }
    
    # 脚本安全配置
    ALLOWED_SCRIPT_EXTENSIONS = {".py", ".ps1", ".bat", ".cmd"}
    DEFAULT_SCRIPT_TIMEOUT = 30
    MAX_SCRIPT_OUTPUT_CHARS = 2000
    
    # 【C6-2】敏感路径（绝对路径前缀），脚本禁止读写
    SENSITIVE_PATH_PREFIXES = [
        "C:\\Windows\\System32",
        "C:\\Windows\\SysWOW64",
        "C:\\Users",  # 用户目录默认禁止，allow_write_paths显式放开才允许
    ]
    # 【C6-2】禁止出现的危险命令片段（Windows平台）
    DANGEROUS_COMMANDS = [
        "del /f /s /q", "rmdir /s /q", "format ", "shutdown ",
        "reg delete", "reg add", "net user", "net localgroup",
        "powershell -enc", "Invoke-Expression", "IEX ",
    ]
    # 【Bug9修复】Windows 扩展长度 / 设备命名前缀（\?\ 与 \.\），归一化敏感路径前先剥掉一层
    _EXTENDED_PATH_PREFIXES = ("\\\\?\\", "\\\\.\\")
    # 【Bug11修复】命令注入字符表（与 models.ScriptParam.validate_value 里的 _DANGEROUS_CHARS 保持同一集合）
    INJECTION_DANGEROUS_CHARS = ['\n', '\r', ';', '&', '|', '`', '$', '(', ')', '<', '>', '\\']
    
    def __init__(
        self,
        extra_paths: Optional[List[Path]] = None,
        default_allow_network: bool = False,
        default_allow_write_paths: Optional[List[str]] = None,
        default_env_whitelist: Optional[List[str]] = None,
        default_script_timeout: int = DEFAULT_SCRIPT_TIMEOUT,
        include_default_paths: bool = True,
    ):
        self.skill_paths: List[Tuple[Path, str]] = []  # (path, source_label)
        self._loaded_skills: Dict[str, SkillLoadResult] = {}
        self._skill_index: Dict[str, SkillMetadata] = {}
        self._tracer = None  # 延迟注入trace
        
        # 【C18】沙箱全局默认配置
        self.default_allow_network: bool = default_allow_network
        self.default_allow_write_paths: List[str] = list(default_allow_write_paths or [])
        self.default_env_whitelist: List[str] = list(default_env_whitelist or [])
        self.DEFAULT_SCRIPT_TIMEOUT = default_script_timeout
        
        # 【性能优化】索引缓存：避免 get_index_prompt / is_skill_disabled 等调用反复全量重扫
        # 索引缓存有效性直接由 _skill_index 是否为空表达（空即需要重扫），无需额外标志位；
        # _last_roots_mtime 记录扫描根目录mtime快照（用于自动检测根目录级变更：新增/删除子目录）
        self._last_roots_mtime: float = 0.0

        # 添加默认路径（可关闭，单元测试/独立演示场景使用）
        if include_default_paths:
            self._add_default_paths()

        # 添加自定义路径
        if extra_paths:
            for p in extra_paths:
                self.add_skill_path(p, source="custom")

    def invalidate_index_cache(self) -> None:
        """主动失效技能索引缓存：直接清空已缓存的索引，下次调用 scan_skills/get_all_metadata 时会强制重扫。
        外部调用方在明确知道目录发生变更（如添加路径、热重载reload后）时可显式调用。
        """
        self._skill_index.clear()
    
    def set_tracer(self, tracer) -> None:
        """注入trace打点器，可选"""
        self._tracer = tracer
    
    def _add_default_paths(self):
        """添加默认扫描路径（project/builtin/user 三级），同一物理目录不重复添加"""
        added: set[str] = set()

        def _add(p: Path, source: str):
            rp = str(Path(p).resolve())
            if rp in added:
                return
            added.add(rp)
            self.add_skill_path(p, source=source)

        # 1. 项目级: 当前工作目录下的.skills（统一标准路径）
        project_skills = Path.cwd() / ".skills"
        _add(project_skills, source="project")

        # 2. 框架内置: fast_agents包内的.skills（与 project 同目录时不重复添加）
        try:
            framework_root = Path(__file__).resolve().parents[2]
            builtin_skills = framework_root / ".skills"
            _add(builtin_skills, source="builtin")
        except Exception:
            pass

        # 3. 用户级: ~/.fast_agents/skills/
        user_home = Path.home()
        user_skills = user_home / ".fast_agents" / "skills"
        _add(user_skills, source="user")
    
    def add_skill_path(self, path: Path, source: str = "custom"):
        """添加Skill扫描路径；添加后会主动失效索引缓存，下次访问时自动重扫"""
        path = Path(path).resolve()
        if path.exists() and path.is_dir():
            self.skill_paths.append((path, source))
            # 添加新路径后需要失效缓存（可能有新技能出现）
            self.invalidate_index_cache()
    
    def _compute_roots_mtime(self) -> float:
        """计算所有技能扫描根目录自身的mtime最大值。
        根目录mtime变化意味着子目录/文件在根级发生新增/删除/重命名，需要重建索引。
        子目录内文件的修改不影响根目录mtime，由 C14 的 _get_skill_dir_mtime（递归rglob）在热重载场景处理。
        """
        max_mtime = 0.0
        for base_path, _source in self.skill_paths:
            if not base_path.exists():
                continue
            try:
                mt = base_path.stat().st_mtime
                if mt > max_mtime:
                    max_mtime = mt
            except OSError:
                pass
        return max_mtime

    def scan_skills(self, force: bool = False) -> Dict[str, SkillMetadata]:
        """
        扫描所有路径下的Skill，构建索引（Level 1: 仅加载元数据）
        返回skill_name -> SkillMetadata的映射
        【性能优化】默认 force=False 时：索引非空且根目录mtime未变则直接返回缓存；
        仅当force=True或根目录mtime变化时才真正执行全量重扫。
        子目录内文件的内容修改由热重载（_check_and_apply_changes -> reload_skill）单独处理。
        """
        if not force and self._skill_index:
            # 快速路径：检查根目录mtime是否变化（新增/删除子目录）
            current_roots_mtime = self._compute_roots_mtime()
            if abs(current_roots_mtime - self._last_roots_mtime) < 0.001:
                return self._skill_index

        # 慢路径：全量重建索引
        self._skill_index.clear()
        self._loaded_skills.clear()

        # 【C8】按来源优先级从低到高排序，确保后扫描的高优先级技能覆盖低优先级同名技能
        sorted_paths = sorted(
            self.skill_paths,
            key=lambda item: self.SOURCE_PRIORITY.get(item[1], 2)  # 未知来源默认project优先级
        )

        for base_path, source in sorted_paths:
            if not base_path.exists():
                continue

            # 遍历base_path下的每个子目录
            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                # 查找SKILL.md文件
                skill_file = self._find_skill_file(skill_dir)
                if not skill_file:
                    continue

                # 解析元数据
                try:
                    metadata = self._parse_skill_metadata(skill_file, skill_dir, source)
                    if metadata:
                        # 校验
                        errors = metadata.validate()
                        if errors:
                            print(f"[SkillLoader] 警告: Skill {skill_dir.name} 元数据校验失败: {'; '.join(errors)}")
                            continue

                        # 检查name与目录名是否一致
                        if metadata.name != skill_dir.name:
                            print(f"[SkillLoader] 警告: Skill {skill_dir.name} 的name字段({metadata.name})与目录名不一致，已自动修正")
                            metadata.name = skill_dir.name

                        # 【C8】同名技能冲突检测：按优先级覆盖，打印警告
                        if metadata.name in self._skill_index:
                            old_meta = self._skill_index[metadata.name]
                            old_priority = self.SOURCE_PRIORITY.get(old_meta.source, 2)
                            new_priority = self.SOURCE_PRIORITY.get(source, 2)
                            if new_priority >= old_priority:
                                # 高优先级覆盖低优先级
                                old_path = str(getattr(old_meta, 'skill_path', 'unknown'))
                                new_path = str(metadata.skill_path)
                                print(
                                    f"[SkillLoader] 警告: 技能 '{metadata.name}' 存在同名冲突："
                                    f"[{old_meta.source}] {old_path} 被 [{source}] {new_path} 覆盖"
                                )
                                self._skill_index[metadata.name] = metadata
                            else:
                                # 低优先级不覆盖高优先级，保留原有
                                old_path = str(getattr(old_meta, 'skill_path', 'unknown'))
                                new_path = str(metadata.skill_path)
                                print(
                                    f"[SkillLoader] 警告: 技能 '{metadata.name}' 存在同名冲突："
                                    f"[{source}] {new_path} 优先级低于 [{old_meta.source}] {old_path}，保留高优先级版本"
                                )
                        else:
                            self._skill_index[metadata.name] = metadata
                except Exception as e:
                    print(f"[SkillLoader] 加载Skill {skill_dir.name} 失败: {e}")

        # 重扫完成后更新根目录mtime快照（供下次快速路径判断是否命中缓存）
        self._last_roots_mtime = self._compute_roots_mtime()
        return self._skill_index
    
    def _find_skill_file(self, skill_dir: Path) -> Optional[Path]:
        """在目录中查找SKILL.md文件"""
        for filename in self.SKILL_FILENAMES:
            candidate = skill_dir / filename
            if candidate.exists() and candidate.is_file():
                return candidate
        return None
    
    def _parse_script_param(self, raw: Dict[str, Any]) -> Optional[ScriptParam]:
        """【C6-2】解析单个脚本参数定义"""
        if not isinstance(raw, dict) or "name" not in raw:
            return None
        ptype = str(raw.get("type", "str")).lower()
        if ptype not in ("str", "int", "float", "bool", "boolean", "enum"):
            ptype = "str"
        if ptype == "boolean":
            ptype = "bool"
        return ScriptParam(
            name=str(raw["name"]),
            type=ptype,
            required=bool(raw.get("required", False)),
            default=raw.get("default"),
            description=str(raw.get("description", "")),
            enum=raw.get("enum"),
            min=raw.get("min"),
            max=raw.get("max"),
            max_length=raw.get("max_length"),
        )
    
    def _parse_script_schema_json(self, schema_file: Path) -> Optional[ScriptSchema]:
        """【C18方案B】解析scripts/目录下单个 <脚本名>.schema.json 文件，返回ScriptSchema；解析失败返回None"""
        try:
            content = schema_file.read_text(encoding="utf-8")
            raw = json.loads(content)
            if not isinstance(raw, dict):
                print(f"[SkillLoader] 警告: 脚本schema文件 {schema_file} 不是JSON对象，已忽略")
                return None
            # 脚本名优先取schema里的name字段，否则取schema文件名（去掉.schema.json后缀）
            sname = raw.get("name") or raw.get("script_name") or schema_file.name.replace(".schema.json", "")
            params_raw = raw.get("params") or raw.get("args") or raw.get("arguments") or []
            params: List[ScriptParam] = []
            if isinstance(params_raw, list):
                for pr in params_raw:
                    sp = self._parse_script_param(pr)
                    if sp is not None:
                        params.append(sp)
            return ScriptSchema(
                script_name=str(sname),
                description=str(raw.get("description", "")),
                params=params,
                timeout=raw.get("timeout"),
            )
        except Exception as e:
            print(f"[SkillLoader] 警告: 解析脚本schema文件 {schema_file} 失败: {e}，已忽略")
            return None
    
    def _parse_scripts_schema(self, raw_scripts: Any) -> Dict[str, ScriptSchema]:
        """【C6-2】【C18已废弃仅保留兼容】解析YAML中的scripts参数Schema定义（方案B不再使用frontmatter scripts段）"""
        result: Dict[str, ScriptSchema] = {}
        if not isinstance(raw_scripts, (list, dict)):
            return result
        
        items = raw_scripts if isinstance(raw_scripts, list) else [
            {"name": k, **v} if isinstance(v, dict) else {"name": k}
            for k, v in raw_scripts.items()
        ]
        
        for item in items:
            if not isinstance(item, dict):
                continue
            sname = item.get("name") or item.get("script_name") or item.get("file")
            if not sname:
                continue
            params_raw = item.get("params") or item.get("args") or item.get("arguments") or []
            params: List[ScriptParam] = []
            if isinstance(params_raw, list):
                for pr in params_raw:
                    sp = self._parse_script_param(pr)
                    if sp is not None:
                        params.append(sp)
            schema = ScriptSchema(
                script_name=str(sname),
                description=str(item.get("description", "")),
                params=params,
                timeout=item.get("timeout"),
            )
            # 同时存带扩展名和不带扩展名两个key，方便模糊匹配
            result[str(sname)] = schema
            stem = Path(str(sname)).stem
            if stem != str(sname):
                result[stem] = schema
        return result
    
    @staticmethod
    def _as_str_list(raw: Any) -> List[str]:
        """【Bug4修复】把 YAML 标量字符串/列表统一归一为 list[str]。

        SKILL.md 中 allowed-tools / tags 允许写成 YAML 列表或逗号分隔字符串（Anthropic 惯例）；
        若直接拿裸字符串，下游 set(str) 会退化为单字符集合，导致工具白名单误拦/标签批量开关失效。
        注：条目按逗号/分号切分，工具名与本框架标识符一致不含空格，故不在条目内拆分。
        """
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            return [t.strip() for t in re.split(r'[,;]+', s) if t.strip()]
        return [str(raw)]

    def _parse_skill_metadata(
        self, skill_file: Path, skill_dir: Path, source: str
    ) -> Optional[SkillMetadata]:
        """解析SKILL.md的YAML frontmatter元数据"""
        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[SkillLoader] 读取{skill_file}失败: {e}")
            return None
        
        # 解析YAML frontmatter
        frontmatter, body = self._split_frontmatter(content)
        if not frontmatter:
            # 没有frontmatter，使用默认值
            metadata = SkillMetadata(
                name=skill_dir.name,
                description=f"{skill_dir.name}技能",
                skill_path=skill_dir,
                skill_type="file",
                source=source,
            )
            return metadata
        
        try:
            meta_dict = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            print(f"[SkillLoader] 解析{skill_file}YAML失败: {e}")
            return None
        
        if not isinstance(meta_dict, dict):
            return None
        
        # 构建SkillMetadata
        # 【C12】支持多种YAML键名写法：requires / dependencies / depends_on
        requires = meta_dict.get("requires",
                     meta_dict.get("dependencies",
                     meta_dict.get("depends_on", [])))
        if not isinstance(requires, list):
            requires = [requires] if requires else []
        
        requires_tools = meta_dict.get("requires_tools",
                       meta_dict.get("required_tools",
                       meta_dict.get("needs_tools", [])))
        if not isinstance(requires_tools, list):
            requires_tools = [requires_tools] if requires_tools else []
        
        min_ver = meta_dict.get("min_fast_agents_version",
                  meta_dict.get("min_version",
                  meta_dict.get("requires_framework_version", None)))
        max_ver = meta_dict.get("max_fast_agents_version",
                  meta_dict.get("max_version", None))
        
        metadata = SkillMetadata(
            name=meta_dict.get("name", skill_dir.name),
            description=meta_dict.get("description", ""),
            version=str(meta_dict.get("version", "1.0.0")),
            author=meta_dict.get("author"),
            license=meta_dict.get("license"),
            compatibility=meta_dict.get("compatibility"),
            allowed_tools=self._as_str_list(meta_dict.get("allowed-tools", meta_dict.get("allowed_tools", []))),
            tags=self._as_str_list(meta_dict.get("tags", [])),
            requires=requires,
            requires_tools=requires_tools,
            min_fast_agents_version=min_ver,
            max_fast_agents_version=max_ver,
            metadata=meta_dict.get("metadata", {}),
            skill_path=skill_dir,
            skill_type="file",
            source=source,
        )
        
        return metadata
    
    def _split_frontmatter(self, content: str) -> Tuple[Optional[str], str]:
        """分离YAML frontmatter和正文"""
        # 匹配开头的--- ... ---块
        pattern = r'^---\s*\n(.*?)\n---\s*\n?'
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            return None, content
        
        frontmatter = match.group(1)
        body = content[match.end():]
        return frontmatter, body
    
    def load_skill(self, skill_name: str) -> SkillLoadResult:
        """
        加载完整Skill内容（Level 2: 加载完整SKILL.md指令）
        """
        span = None
        try:
            # Trace打点开始
            if self._tracer is not None:
                try:
                    _ensure_trace_imports()
                    span = self._tracer.start_span(
                        name=f"skill.load.{skill_name}",
                        kind=_SpanKind.SKILL,
                        attributes={"skill.name": skill_name, "skill.op": "load"}
                    )
                except Exception:
                    span = None
            
            # 先确保索引已构建
            if not self._skill_index:
                self.scan_skills()
            
            # 检查是否已加载
            if skill_name in self._loaded_skills:
                if span:
                    try:
                        span.set_attribute("skill.cached", True)
                        span.finish()
                    except Exception:
                        pass
                return self._loaded_skills[skill_name]
            
            # 查找元数据
            metadata = self._skill_index.get(skill_name)
            if not metadata:
                result = SkillLoadResult(
                    success=False,
                    skill_name=skill_name,
                    errors=[f"Skill '{skill_name}' 不存在"],
                )
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=f"Skill not found: {skill_name}")
                    except Exception:
                        pass
                return result
            
            skill_dir = metadata.skill_path
            if not skill_dir or not skill_dir.exists():
                result = SkillLoadResult(
                    success=False,
                    skill_name=skill_name,
                    errors=[f"Skill目录不存在: {skill_dir}"],
                )
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=f"Skill dir not found: {skill_dir}")
                    except Exception:
                        pass
                return result
            
            skill_file = self._find_skill_file(skill_dir)
            if not skill_file:
                result = SkillLoadResult(
                    success=False,
                    skill_name=skill_name,
                    errors=["SKILL.md文件不存在"],
                )
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error="SKILL.md not found")
                    except Exception:
                        pass
                return result
            
            try:
                content = skill_file.read_text(encoding="utf-8")
                _, instructions = self._split_frontmatter(content)
                instructions = instructions.strip()
                
                # 扫描scripts目录
                scripts = []
                scripts_schemas: Dict[str, ScriptSchema] = {}
                scripts_dir = skill_dir / self.SCRIPTS_DIR
                if scripts_dir.exists() and scripts_dir.is_dir():
                    for p in scripts_dir.iterdir():
                        if not p.is_file() or p.name.startswith("_"):
                            continue
                        # 【C18方案B】识别.schema.json文件，作为对应脚本的参数Schema
                        if p.name.endswith(".schema.json"):
                            schema = self._parse_script_schema_json(p)
                            if schema is not None:
                                # 同时存带扩展名和不带扩展名两个key，方便模糊匹配
                                scripts_schemas[schema.script_name] = schema
                                stem = Path(schema.script_name).stem
                                if stem != schema.script_name:
                                    scripts_schemas[stem] = schema
                            continue
                        # 普通可执行脚本文件
                        if p.suffix.lower() in self.ALLOWED_SCRIPT_EXTENSIONS:
                            scripts.append(p)
                
                # 扫描references目录
                references = []
                refs_dir = skill_dir / self.REFERENCES_DIR
                if refs_dir.exists() and refs_dir.is_dir():
                    references = [
                        p for p in refs_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".json", ".yaml", ".yml"]
                    ]
                
                # 【C15】扫描assets目录（递归，文本/二进制分流；以下划线开头的视为内部文件）
                # 方案A：assets清单只进入SkillLoadResult（Level 2加载后通过full_prompt展示），不进入SkillMetadata常驻索引
                assets: List[Tuple[str, int, bool]] = []
                assets_dir = skill_dir / self.ASSETS_DIR
                if assets_dir.exists() and assets_dir.is_dir():
                    for ap in sorted(assets_dir.rglob("*")):
                        if not ap.is_file():
                            continue
                        rel_parts = ap.relative_to(assets_dir).parts
                        if any(part.startswith("_") for part in rel_parts):
                            continue
                        rel_str = "/".join(rel_parts)
                        try:
                            sz = ap.stat().st_size
                        except OSError:
                            sz = 0
                        is_bin = ap.suffix.lower() not in _TEXT_ASSET_EXTENSIONS
                        assets.append((rel_str, sz, is_bin))
                
                result = SkillLoadResult(
                    success=True,
                    skill_name=skill_name,
                    metadata=metadata,
                    content=content,
                    instructions=instructions,
                    scripts=scripts,
                    references=references,
                    assets=assets,
                    scripts_schemas=scripts_schemas,
                )
                
                self._loaded_skills[skill_name] = result
                
                if span:
                    try:
                        span.set_attribute("skill.cached", False)
                        span.set_attribute("skill.scripts_count", len(scripts))
                        span.set_attribute("skill.references_count", len(references))
                        span.set_attribute("skill.assets_count", len(assets))
                        span.set_attribute("skill.scripts_schemas_count", len(scripts_schemas))
                        span.set_attribute("skill.sandbox.allow_network", self.default_allow_network)
                        span.finish()
                    except Exception:
                        pass
                
                return result
            
            except Exception as e:
                result = SkillLoadResult(
                    success=False,
                    skill_name=skill_name,
                    errors=[f"加载Skill失败: {str(e)}"],
                )
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=str(e))
                    except Exception:
                        pass
                return result
        except Exception:
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=["加载Skill失败: 内部错误"],
            )
    
    def read_reference(self, skill_name: str, ref_name: str) -> str:
        """读取Skill参考文档（Level 3: 按需读取）"""
        span = None
        try:
            # Trace打点开始
            if self._tracer is not None:
                try:
                    _ensure_trace_imports()
                    span = self._tracer.start_span(
                        name=f"skill.read_reference.{skill_name}",
                        kind=_SpanKind.SKILL,
                        attributes={"skill.name": skill_name, "skill.reference": ref_name, "skill.op": "read_reference"}
                    )
                except Exception:
                    span = None
            
            skill_result = self.load_skill(skill_name)
            if not skill_result.success:
                error_msg = f"读取参考文档失败: {'; '.join(skill_result.errors)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 查找匹配的参考文件
            skill_dir = skill_result.metadata.skill_path
            refs_base_dir = (skill_dir / self.REFERENCES_DIR).resolve()
            
            # 安全：路径穿越防护（用 relative_to 严格判定归属，避免同前缀兄弟目录如 references_evil 绕过 startswith 检查）
            ref_path = (refs_base_dir / ref_name).resolve()
            try:
                ref_path.relative_to(refs_base_dir)
            except ValueError:
                error_msg = "非法参考文档路径：禁止访问references目录外的文件"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            if not ref_path.exists():
                # 尝试模糊匹配
                if refs_base_dir.exists():
                    for p in refs_base_dir.iterdir():
                        if p.name.lower() == ref_name.lower() or ref_name.lower() in p.name.lower():
                            ref_path = p
                            break
            
            if not ref_path.exists():
                error_msg = f"参考文档 '{ref_name}' 不存在"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            try:
                content = ref_path.read_text(encoding="utf-8")
                if span:
                    try:
                        span.set_attribute("reference.size_bytes", len(content.encode('utf-8')))
                        span.finish()
                    except Exception:
                        pass
                return content
            except Exception as e:
                error_msg = f"读取参考文档失败: {str(e)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
        except Exception:
            return "读取参考文档失败: 内部错误"
    
    def read_asset(self, skill_name: str, asset_name: str) -> str:
        """【C15】读取Skill静态资产
        
        设计：
        - 路径穿越防护：asset_name resolve后必须仍在assets/目录下
        - 文本/二进制分流：文本类扩展名且未超 200KB 时直接返回 UTF-8 内容；
          二进制资产不返回正文，仅返回元信息（类型/大小/绝对路径），交由脚本/用户处理
        - 模糊匹配：大小写不敏感 + 子串匹配（与read_reference保持一致）
        """
        span = None
        try:
            if self._tracer is not None:
                try:
                    _ensure_trace_imports()
                    span = self._tracer.start_span(
                        name=f"skill.read_asset.{skill_name}",
                        kind=_SpanKind.SKILL,
                        attributes={"skill.name": skill_name, "skill.asset": asset_name, "skill.op": "read_asset"}
                    )
                except Exception:
                    span = None
            
            skill_result = self.load_skill(skill_name)
            if not skill_result.success:
                error_msg = f"读取静态资产失败: {'; '.join(skill_result.errors)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            skill_dir = skill_result.metadata.skill_path
            assets_base_dir = (skill_dir / self.ASSETS_DIR).resolve()
            
            # 路径穿越防护
            asset_path = (assets_base_dir / asset_name).resolve()
            try:
                asset_path.relative_to(assets_base_dir)
            except ValueError:
                error_msg = "非法静态资产路径：禁止访问assets目录外的文件"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            if not asset_path.exists():
                # 模糊匹配：大小写不敏感 + 子串
                if assets_base_dir.exists():
                    for p in assets_base_dir.rglob("*"):
                        if not p.is_file():
                            continue
                        rel = p.relative_to(assets_base_dir).as_posix()
                        if (p.name.lower() == asset_name.lower()
                                or rel.lower() == asset_name.lower()
                                or asset_name.lower() in rel.lower()):
                            asset_path = p
                            break
            
            if not asset_path.exists() or not asset_path.is_file():
                error_msg = f"静态资产 '{asset_name}' 不存在"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            rel_path = asset_path.relative_to(assets_base_dir).as_posix()
            try:
                size = asset_path.stat().st_size
            except OSError:
                size = 0
            suffix = asset_path.suffix.lower()
            is_binary = suffix not in _TEXT_ASSET_EXTENSIONS
            
            if is_binary:
                # 二进制资产：只返回元信息与绝对路径，不返回正文（这是预期行为，不记错误）
                if span:
                    try:
                        span.set_attribute("asset.size_bytes", size)
                        span.set_attribute("asset.is_binary", True)
                        span.finish()
                    except Exception:
                        pass
                return (
                    f"# 静态资产: {skill_name}/assets/{rel_path}\n"
                    f"\n"
                    f"- 类型：二进制（{suffix or '无扩展名'}）\n"
                    f"- 大小：{size} 字节\n"
                    f"- 绝对路径：{asset_path}\n"
                    f"\n"
                    f"提示：该文件为二进制格式，不能直接作为文本注入对话；"
                    f"如需要处理，请通过 `run_skill_script` 调用技能脚本来读写或处理该文件。"
                )
            
            # 文本资产：大小上限保护
            if size > _MAX_TEXT_ASSET_BYTES:
                if span:
                    try:
                        span.set_attribute("asset.size_bytes", size)
                        span.set_attribute("asset.is_binary", False)
                        span.set_attribute("asset.too_large", True)
                        span.finish()
                    except Exception:
                        pass
                return (
                    f"# 静态资产: {skill_name}/assets/{rel_path}\n"
                    f"\n"
                    f"- 类型：文本（{suffix}）\n"
                    f"- 大小：{size} 字节（超过单次读取上限 {_MAX_TEXT_ASSET_BYTES} 字节）\n"
                    f"- 绝对路径：{asset_path}\n"
                    f"\n"
                    f"提示：文件过大，未直接返回内容以避免撑爆上下文；"
                    f"如需局部内容，请通过 `run_skill_script` 让脚本读取所需片段。"
                )
            
            try:
                content = asset_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # 扩展名是文本但实际是二进制（或非UTF-8），降级为元信息返回
                if span:
                    try:
                        span.set_attribute("asset.size_bytes", size)
                        span.set_attribute("asset.decode_error", True)
                        span.finish(status=_SpanStatus.ERROR, error="文本资产无法以UTF-8解码")
                    except Exception:
                        pass
                return (
                    f"# 静态资产: {skill_name}/assets/{rel_path}\n"
                    f"\n"
                    f"- 类型：扩展名是文本（{suffix}）但无法以 UTF-8 解码，按二进制处理\n"
                    f"- 大小：{size} 字节\n"
                    f"- 绝对路径：{asset_path}\n"
                )
            
            if span:
                try:
                    span.set_attribute("asset.size_bytes", size)
                    span.set_attribute("asset.is_binary", False)
                    span.finish()
                except Exception:
                    pass
            return (
                f"# 静态资产: {skill_name}/assets/{rel_path}\n"
                f"\n"
                f"```\n{content}\n```"
            )
        except Exception:
            return "读取静态资产失败: 内部错误"
    
    def _build_sandbox_env(self, skill_dir: Path) -> Dict[str, str]:
        """【C6-2】【C18重构】构造沙箱环境变量：全局白名单透传 + 网络禁用，不再从SkillMetadata读取配置"""
        env: Dict[str, str] = {}
        
        # 白名单环境变量透传（使用全局默认配置）
        for key in self.default_env_whitelist:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        
        # 默认透传PATH、SYSTEMROOT等Windows运行必需变量
        for must_have in ("PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "PYTHONIOENCODING"):
            val = os.environ.get(must_have)
            if val is not None and must_have not in env:
                env[must_have] = val
        env["PYTHONIOENCODING"] = "utf-8"
        
        # 网络禁用：通过环境变量拦截Python requests/urllib和常用库（使用全局默认配置）
        if not self.default_allow_network:
            env["HTTP_PROXY"] = "http://127.0.0.1:1"  # 指向无效代理
            env["HTTPS_PROXY"] = "http://127.0.0.1:1"
            env["NO_PROXY"] = ""
            env["http_proxy"] = "http://127.0.0.1:1"
            env["https_proxy"] = "http://127.0.0.1:1"
            env["SKILL_SANDBOX_NETWORK_DISABLED"] = "1"
        
        return env
    
    def _get_allow_write_paths(self, skill_dir: Path) -> List[str]:
        """【C18】获取当前技能的可写路径白名单：全局配置 + 技能自身tmp/目录"""
        allow_list = list(self.default_allow_write_paths)
        # 默认允许写入技能目录下的tmp/
        default_tmp = str((skill_dir / "tmp").resolve())
        if default_tmp not in allow_list:
            allow_list.append(default_tmp)
        return allow_list
    
    @staticmethod
    def _iter_arg_strings(value: Any, prefix: str = ""):
        """【Bug7修复】显式栈把嵌套参数逐层展开为 (定位路径, 字符串) 对。

        与 guardrails 模块 B2 修复模式对齐：dict 键与值都展开，list/tuple/set/frozenset 逐元素展开，
        os.PathLike 经 os.fspath 转字符串，其余标量跳过；
        用显式栈而非递归（无 RecursionError 风险，绝不打挂主流程）；
        不设深度上限（避免制造「藏到第 N+1 层就过关」的新绕过口）；
        环保护用「当前分支祖先 id 元组」剪枝（不用全局 visited，兄弟分支合法复用同一对象不会被误剪）。
        """
        stack: List[Tuple[Any, str, Tuple[int, ...]]] = [(value, prefix, ())]
        while stack:
            cur, path, anc_ids = stack.pop()
            if id(cur) in anc_ids:
                continue  # 环保护：当前分支祖先链已含同一对象则不再进入
            if isinstance(cur, str):
                yield path, cur
            elif isinstance(cur, os.PathLike):
                try:
                    yield path, os.fspath(cur)
                except Exception:
                    pass
            elif isinstance(cur, dict):
                branch_ids = anc_ids + (id(cur),)
                for k, v in cur.items():
                    key_path = f"{path}.{k}" if path else str(k)
                    if isinstance(k, str):
                        stack.append((k, key_path, branch_ids))
                    stack.append((v, key_path, branch_ids))
            elif isinstance(cur, (list, tuple, set, frozenset)):
                branch_ids = anc_ids + (id(cur),)
                for i, item in enumerate(cur):
                    stack.append((item, f"{path}[{i}]", branch_ids))
            # 其余标量（int / float / bool / None 等）跳过

    @classmethod
    def _normalize_path_candidate(cls, value: Any, skill_dir_norm: Optional[str] = None) -> Optional[str]:
        """【Bug9修复】把参数字符串归一化为「可比较的路径形态」，供敏感路径前缀判定使用。

        修复前的判定只做 lower() + 反斜杠转斜杠 + startswith，因此下列常见写法能整体绕过沙箱：
            1. `C:\\\\Windows\\\\System32\\\\x`（双反斜杠，JSON 双重转义 / 字符串字面量形态）
            2. `\\\\?\\C:\\Windows\\System32\\x`（Windows 扩展长度前缀，绕过斜杠归一）
            3. `C:\\Windows\\Temp\\..\\System32\\x`（.. 混排，字面量不以敏感前缀开头）
            4. `C:\\PROGRA~1\\..\\Windows\\System32\\x`（8.3 短名 + ..）
            5. `..\\..\\..\\..\\..\\Windows\\System32\\x`（相对路径向上跳出技能目录）
        归一化步骤：剥 file:// 与 \\\\?\\ 前缀 → 分隔符统一为反斜杠 → ntpath.normpath 折叠 `..`/`.`/重复分隔符
            → 再回到「小写 + 正斜杠」形态。
        相对路径按脚本工作目录（技能目录）解析，但**只在对 `..` 向上跳出技能目录时**才纳入判定：
            留在技能目录内的相对路径（如 templates/a.md、tmp/out.csv）本就是沙箱允许的读写区，不判。
        返回 None 表示「不必检查」：空串、不含分隔符的普通文本（表达式/文件名/工具名等）、目录内相对路径。
        """
        s = str(value).strip()
        if not s:
            return None
        # 1. file:// URI → 取本地路径部分（支持 file:///C:/x、file://C:/x、file://localhost/C:/x）
        if s.lower().startswith("file://"):
            rest = s[len("file://"):]
            m = re.match(r'^/?(?:localhost)?[\\/]?([a-zA-Z]:[\\/].+)$', rest, re.IGNORECASE)
            s = m.group(1) if m else rest.lstrip("/\\")
        # 2. 扩展长度 / 设备前缀（只剥一层，避免把 \?\ 藏进深层再绕）
        for ext in cls._EXTENDED_PATH_PREFIXES:
            if s.startswith(ext):
                s = s[len(ext):]
                break
        # 3. 分隔符统一成反斜杠，交给 ntpath 做折叠
        s = s.replace("/", "\\")
        # 4. 绝对路径 / UNC：直接归一；相对路径：先折叠再判断是否逃出了技能目录
        if not os.path.splitdrive(s)[0] and not s.startswith("\\"):
            norm_rel = os.path.normpath(s)
            if not (norm_rel == ".." or norm_rel.startswith(".." + os.sep)):
                return None  # 目录内相对路径，沙箱本就允许，不做敏感判定
            if skill_dir_norm:
                s = os.path.join(str(skill_dir_norm).replace("/", "\\"), norm_rel)
            else:
                s = norm_rel
        s = os.path.normpath(s)
        return s.lower().replace("\\", "/")

    def _check_sensitive_path_access(self, args: Dict[str, Any], skill_dir: Path) -> Optional[str]:
        """【C6-2】【C18重构】检查参数中是否包含敏感路径，返回错误信息或None，使用全局白名单+技能tmp目录
        【Bug7修复】顶层-only 扫描改为经 _iter_arg_strings 逐层展开嵌套结构，
        堵住 {"payload": {"path": "C:/Windows/System32/..."}} 这类参数绕过；
        【Bug9修复】判定前先做路径归一化（_normalize_path_candidate），堵住双反斜杠 / \\\\?\ 前缀 /
        `..` 混排 / 8.3 短名 / 逃出技能目录的相对路径 五类绕过；
        命中后的错误文案保持回显**原始值**，与归一化改动前逐字节一致。
        注：前缀比较仍沿用原有 startswith 语义（不引入目录边界切分），以免把 `C:\\Users` 这类宽口径
        防护改松——本次改动只收紧防护面，不放松任何既有拦截。
        """
        # 【Bug9修复】可写白名单也归一化后再比较，避免白名单用反斜杠写、参数用正斜杠写导致豁免失效
        allow_list = [
            n for n in (
                self._normalize_path_candidate(p) for p in self._get_allow_write_paths(skill_dir)
            ) if n
        ]
        # 敏感路径统一为小写正斜杠形式匹配（常量表本身也归一化，消除表内反斜杠写法与参数形态的差异）
        sensitive_prefixes = [
            n for n in (
                self._normalize_path_candidate(p) for p in self.SENSITIVE_PATH_PREFIXES
            ) if n
        ]
        # 额外加入常见Windows系统路径的正斜杠/反斜杠变体
        extra_sensitive = [
            "c:/windows/system32", "c:/windows/syswow64", "c:/users",
        ]
        sensitive_prefixes.extend(extra_sensitive)
        # 技能目录归一化形态：用于把「向上逃逸」的相对路径锚定到脚本真实工作目录
        skill_dir_norm = self._normalize_path_candidate(skill_dir)

        for arg_path, v in self._iter_arg_strings(args):
            s = self._normalize_path_candidate(v, skill_dir_norm)
            if s is None:
                continue
            # 检查是否命中敏感路径
            for prefix in sensitive_prefixes:
                if s.startswith(prefix):
                    in_allow = any(s.startswith(allowed) for allowed in allow_list)
                    if not in_allow:
                        return f"参数 '{arg_path}' 指向敏感路径 '{v}'，不在写入白名单内（沙箱拦截）"
        return None
    
    def _check_injection_chars(self, args: Dict[str, Any], reason: str) -> Optional[str]:
        """【Bug11修复】对参数字符串做命令注入字符检查，返回错误文案或 None。

        参数集合由调用方给出（无Schema路径走全量 args，有Schema路径只走未声明的键），
        reason 为文案括号内的口径说明，便于区分「未声明Schema」与「Schema未声明该参数」两种场景。
        沿用 _iter_arg_strings 逐层展开（嵌套 dict/list 里的字符串同样覆盖）。
        """
        for loc, v in self._iter_arg_strings(args):
            for ch in self.INJECTION_DANGEROUS_CHARS:
                if ch in v:
                    return f"参数 '{loc}' 包含非法字符 '{ch}'（{reason}）"
        return None

    def run_script(
        self,
        skill_name: str,
        script_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        执行Skill脚本（Level 3: 按需执行）
        安全加固（已有）：路径穿越防护、扩展名白名单、输出截断
        【C6-2】新增：参数Schema校验+类型转换、命令注入防护、敏感路径拦截、网络白名单、可写路径白名单
        timeout=None 时取实例配置 self.DEFAULT_SCRIPT_TIMEOUT（旧写法默认值在类定义期绑死为类常量30，导致构造参数 default_script_timeout 失效）
        """
        # 【Bug3修复】运行期解析默认超时，让构造传入的 default_script_timeout 真正生效；显式传入 timeout 时行为与改动前一致
        if timeout is None:
            timeout = self.DEFAULT_SCRIPT_TIMEOUT
        span = None
        try:
            # Trace打点开始
            if self._tracer is not None:
                try:
                    _ensure_trace_imports()
                    span = self._tracer.start_span(
                        name=f"skill.run_script.{skill_name}",
                        kind=_SpanKind.SKILL,
                        attributes={"skill.name": skill_name, "skill.script": script_name, "skill.op": "run_script"}
                    )
                except Exception:
                    span = None
            
            skill_result = self.load_skill(skill_name)
            if not skill_result.success:
                error_msg = f"执行脚本失败: {'; '.join(skill_result.errors)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            metadata = skill_result.metadata
            skill_dir = metadata.skill_path
            scripts_base_dir = (skill_dir / self.SCRIPTS_DIR).resolve()
            
            # 安全：路径穿越防护（用 relative_to 严格判定归属，避免同前缀兄弟目录如 scripts_evil 绕过 startswith 检查）
            script_path = (scripts_base_dir / script_name).resolve()
            try:
                script_path.relative_to(scripts_base_dir)
            except ValueError:
                error_msg = "非法脚本路径：禁止访问scripts目录外的文件"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 尝试添加扩展名查找
            if not script_path.exists():
                for ext in self.ALLOWED_SCRIPT_EXTENSIONS:
                    candidate = script_path.with_suffix(ext)
                    if candidate.exists():
                        script_path = candidate
                        break
            
            if not script_path.exists():
                error_msg = f"脚本 '{script_name}' 不存在"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 安全：校验扩展名
            suffix = script_path.suffix.lower()
            if suffix not in self.ALLOWED_SCRIPT_EXTENSIONS:
                error_msg = f"不支持执行该类型脚本：{suffix}，仅允许 {', '.join(self.ALLOWED_SCRIPT_EXTENSIONS)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # ========== 【C18方案B】参数Schema校验与类型转换（schema来自scripts/&lt;脚本名&gt;.schema.json） ==========
            args = args or {}
            validated_args = args
            schema = skill_result.get_script_schema(script_path.name) or skill_result.get_script_schema(Path(script_name).stem)
            if schema is not None:
                ok, errors, converted = schema.validate_args(args)
                if not ok:
                    error_msg = "参数校验失败：\n- " + "\n- ".join(errors)
                    if span:
                        try:
                            span.set_attribute("script.validation_failed", True)
                            span.finish(status=_SpanStatus.ERROR, error=error_msg)
                        except Exception:
                            pass
                    return error_msg
                validated_args = converted
                # 覆盖timeout为脚本声明的值
                if schema.timeout is not None:
                    timeout = schema.timeout
                # 【Bug11修复】Schema 只管得到「已声明参数」（ScriptParam.validate_value 内含注入字符检查），
                # 而未知参数在 validate_args 里是原样放行进 converted 的，修复前这些值完全跳过注入字符检查，
                # 造成「声明了Schema反而比不声明更松」：{"unknown": "a;b|c&d"} 能直通脚本 argv。
                # 这里对未声明的键补上与无Schema路径同强度的检查（已声明参数不重复审，避免改变现有校验次序/文案）。
                declared_names = {p.name for p in schema.params}
                undeclared_args = {k: v for k, v in args.items() if k not in declared_names}
                if undeclared_args:
                    inj_err = self._check_injection_chars(
                        undeclared_args, "命令注入防护，Schema未声明的参数同样禁止特殊字符"
                    )
                    if inj_err:
                        if span:
                            try:
                                span.set_attribute("script.injection_blocked", True)
                                span.finish(status=_SpanStatus.ERROR, error=inj_err)
                            except Exception:
                                pass
                        return inj_err
            else:
                # 无Schema时也做基本的命令注入字符检查
                for k, v in args.items():
                    if isinstance(v, str):
                        for ch in self.INJECTION_DANGEROUS_CHARS:
                            if ch in v:
                                error_msg = f"参数 '{k}' 包含非法字符 '{ch}'（命令注入防护，未声明Schema的参数默认禁止特殊字符）"
                                if span:
                                    try:
                                        span.set_attribute("script.injection_blocked", True)
                                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                                    except Exception:
                                        pass
                                return error_msg
            
            # ========== 【C6-2】新增：敏感路径访问检查（【C18】使用全局白名单+技能tmp目录） ==========
            path_err = self._check_sensitive_path_access(validated_args, skill_dir)
            if path_err:
                if span:
                    try:
                        span.set_attribute("script.sandbox_blocked", True)
                        span.finish(status=_SpanStatus.ERROR, error=path_err)
                    except Exception:
                        pass
                return f"沙箱拦截：{path_err}"
            
            # ========== 【C6-2】新增：禁止危险命令片段（bat/cmd场景） ==========
            if suffix in (".bat", ".cmd", ".ps1"):
                try:
                    script_content = script_path.read_text(encoding="utf-8", errors="replace")
                    content_lower = script_content.lower()
                    for dangerous in self.DANGEROUS_COMMANDS:
                        if dangerous.lower() in content_lower:
                            error_msg = f"脚本包含危险命令 '{dangerous}'，沙箱拦截"
                            if span:
                                try:
                                    span.set_attribute("script.dangerous_command_blocked", True)
                                    span.finish(status=_SpanStatus.ERROR, error=error_msg)
                                except Exception:
                                    pass
                            return f"沙箱拦截：{error_msg}"
                except Exception:
                    pass
            
            # 构建命令
            cmd = []
            
            # Windows平台专用：根据扩展名选择解释器
            if suffix == ".py":
                cmd = [sys.executable, str(script_path)]
            elif suffix in (".ps1", ".bat", ".cmd"):
                cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
            else:
                # 理论上不会走到这里，上面已经校验扩展名
                cmd = [str(script_path)]
            
            # 添加参数（使用校验+转换后的参数）
            for key, value in validated_args.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            # ========== 【C6-2】沙箱环境变量（【C18】使用全局默认配置） ==========
            sandbox_env = self._build_sandbox_env(skill_dir)
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(skill_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                    env=sandbox_env,  # 【C6-2】使用沙箱化的环境变量
                )
                
                output = []
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                
                # 输出截断
                if len(stdout) > self.MAX_SCRIPT_OUTPUT_CHARS:
                    stdout = stdout[:self.MAX_SCRIPT_OUTPUT_CHARS] + f"\n\n[输出已截断，原始长度{len(result.stdout)}字符]"
                if len(stderr) > self.MAX_SCRIPT_OUTPUT_CHARS // 2:
                    stderr = stderr[:self.MAX_SCRIPT_OUTPUT_CHARS // 2] + f"\n\n[stderr输出已截断]"
                
                if stdout:
                    output.append(stdout)
                if stderr:
                    output.append(f"[stderr]\n{stderr}")
                if result.returncode != 0:
                    output.append(f"[退出码: {result.returncode}]")
                    if span:
                        try:
                            span.set_attribute("script.exit_code", result.returncode)
                        except Exception:
                            pass
                
                final_output = "\n".join(output).strip()
                
                if span:
                    try:
                        span.set_attribute("script.timeout", timeout)
                        span.set_attribute("script.args_count", len(validated_args))
                        span.set_attribute("script.allow_network", self.default_allow_network)
                        if result.returncode != 0:
                            span.finish(status=_SpanStatus.ERROR, error=f"脚本退出码非零: {result.returncode}")
                        else:
                            span.finish()
                    except Exception:
                        pass
                
                return final_output
            
            except subprocess.TimeoutExpired:
                error_msg = f"脚本执行超时（{timeout}秒）"
                if span:
                    try:
                        span.set_attribute("script.timeout_expired", True)
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
            except Exception as e:
                error_msg = f"执行脚本失败: {str(e)}"
                if span:
                    try:
                        span.finish(status=_SpanStatus.ERROR, error=error_msg)
                    except Exception:
                        pass
                return error_msg
        except Exception:
            # 打点兜底：绝不让观测代码打挂主流程
            return "执行脚本失败: 内部错误"
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有已发现的Skill"""
        if not self._skill_index:
            self.scan_skills()
        
        return [meta.to_dict() for meta in self._skill_index.values()]

    # ========== 【C14】技能热重载相关方法 ==========

    def _get_skill_dir_mtime(self, skill_dir: Path) -> float:
        """【C14】递归计算技能目录下所有文件的最大mtime，用于检测变更"""
        max_mtime = 0.0
        try:
            for file in skill_dir.rglob("*"):
                if file.is_file():
                    try:
                        mtime = file.stat().st_mtime
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return max_mtime

    def _build_dir_mtime_map(self) -> Dict[str, float]:
        """【C14】构建当前所有已发现技能目录的mtime快照：skill_name -> max_mtime
        【Bug6修复】无条件先调 scan_skills()（非 force）：索引非空且根目录 mtime 未变时直接命中缓存返回
        （仅对扫描根目录 stat，零遍历开销）；新增/删除/重命名技能目录使根 mtime 变化时自动重建索引。
        修复原「索引为空才扫描」导致热重载轮询线程永远发现不了新增技能的问题
        （new_mtime_map 基于旧索引构建，新增目录不可见，与 enable_hot_reload 的承诺不符）。
        """
        mtime_map: Dict[str, float] = {}
        # 确保索引为最新（根目录mtime未变时命中缓存直接返回）
        self.scan_skills()
        for name, meta in self._skill_index.items():
            skill_dir = getattr(meta, 'skill_path', None)
            if skill_dir and Path(skill_dir).exists():
                mtime_map[name] = self._get_skill_dir_mtime(Path(skill_dir))
        return mtime_map

    def detect_changes(self, old_mtime_map: Dict[str, float]) -> Dict[str, List[str]]:
        """【C14】对比新旧mtime快照，检测变更，返回三类变更列表"""
        new_mtime_map = self._build_dir_mtime_map()

        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []

        # 检测新增和修改
        for name, new_mtime in new_mtime_map.items():
            if name not in old_mtime_map:
                added.append(name)
            elif abs(new_mtime - old_mtime_map[name]) > 0.001:
                modified.append(name)

        # 检测删除
        for name in old_mtime_map:
            if name not in new_mtime_map:
                removed.append(name)

        return {
            "added": added,
            "modified": modified,
            "removed": removed,
            "new_mtime_map": new_mtime_map,
        }

    def reload_skill(self, skill_name: str) -> Optional[SkillMetadata]:
        """【C14】重新解析单个技能的元数据，返回新元数据；技能不存在或解析失败返回None"""
        # 找到该技能对应的目录（优先从现有元数据取）
        old_meta = self._skill_index.get(skill_name)
        if old_meta is None:
            return None

        skill_dir_path = getattr(old_meta, 'skill_path', None)
        if skill_dir_path is None:
            return None
        skill_dir = Path(skill_dir_path)
        if not skill_dir.exists():
            return None

        # 查找SKILL.md
        skill_file = self._find_skill_file(skill_dir)
        if not skill_file:
            return None

        try:
            new_meta = self._parse_skill_metadata(skill_file, skill_dir, old_meta.source)
            if new_meta is None:
                return None
            errors = new_meta.validate()
            if errors:
                print(f"[SkillLoader][热重载] 警告: 技能 '{skill_name}' 新元数据校验失败，保持旧版本: {'; '.join(errors)}")
                return None
            # 确保name与目录名一致
            if new_meta.name != skill_dir.name:
                new_meta.name = skill_dir.name
            # 更新索引
            self._skill_index[skill_name] = new_meta
            # 清理该技能已加载的内容缓存（由调用方middleware负责后续处理）
            if skill_name in self._loaded_skills:
                del self._loaded_skills[skill_name]
            # 单个技能reload不标记整个缓存dirty（只更新了这一个），但刷新根目录mtime快照防止后续误判
            self._last_roots_mtime = self._compute_roots_mtime()
            return new_meta
        except Exception as e:
            print(f"[SkillLoader][热重载] 重新加载技能 '{skill_name}' 失败: {e}")
            return None

    def rescan_incremental(self) -> Dict[str, List[str]]:
        """【C14】增量重扫：发现新增技能、检测删除技能；不修改现有正常技能的元数据
        返回新增/删除的技能名列表
        """
        before_names = set(self._skill_index.keys())
        # 全量重扫（force=True，会清空_index重建，因此先备份）
        old_meta_backup = dict(self._skill_index)
        self._skill_index.clear()
        self.scan_skills(force=True)
        after_names = set(self._skill_index.keys())

        added = sorted(after_names - before_names)
        removed = sorted(before_names - after_names)

        # 对于修改的技能——scan_skills已经重新解析了它们，因此只要在before和after中都存在，
        # 就已经拿到新元数据了；detect_changes靠mtime判断是否真正修改了文件
        return {"added": added, "removed": removed}
