"""
Skill加载器：负责扫描、解析、加载文件型Skill（SKILL.md格式）
遵循Anthropic开放标准，兼容Windows和Unix路径
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .errors import SkillLoadError, SkillNotFoundError, ScriptExecutionError, ReferenceNotFoundError
from .models import SkillMetadata, SkillLoadResult


class SkillLoader:
    """
    Skill加载器：扫描目录，解析SKILL.md，加载技能内容
    
    支持扫描路径（优先级从高到低）:
        - 自定义路径: 通过add_skill_path添加
        - 项目级: ./.skills/
        - 用户级: ~/.fast_agents/skills/
        - 框架内置: fast_agents包内的.skills/
    """
    
    SKILL_FILENAMES = ["SKILL.md", "skill.md", "Skills.md", "skills.md"]
    SCRIPTS_DIR = "scripts"
    REFERENCES_DIR = "references"
    ASSETS_DIR = "assets"
    
    # 脚本安全配置
    ALLOWED_SCRIPT_EXTENSIONS = {".py", ".ps1", ".bat", ".cmd"}
    DEFAULT_SCRIPT_TIMEOUT = 30
    MAX_SCRIPT_OUTPUT_CHARS = 2000
    
    def __init__(self, extra_paths: Optional[List[Path]] = None):
        self.skill_paths: List[Tuple[Path, str]] = []  # (path, source_label)
        self._loaded_skills: Dict[str, SkillLoadResult] = {}
        self._skill_index: Dict[str, SkillMetadata] = {}
        self._tracer = None  # 延迟注入trace
        
        # 添加默认路径
        self._add_default_paths()
        
        # 添加自定义路径
        if extra_paths:
            for p in extra_paths:
                self.add_skill_path(p, source="custom")
    
    def set_tracer(self, tracer) -> None:
        """注入trace打点器，可选"""
        self._tracer = tracer
    
    def _add_default_paths(self):
        """添加默认扫描路径"""
        # 1. 项目级: 当前工作目录下的.skills（统一标准路径）
        project_skills = Path.cwd() / ".skills"
        self.add_skill_path(project_skills, source="project")
        
        # 2. 框架内置: fast_agents包内的.skills
        try:
            framework_root = Path(__file__).resolve().parents[2]
            builtin_skills = framework_root / ".skills"
            self.add_skill_path(builtin_skills, source="builtin")
        except Exception:
            pass
        
        # 3. 用户级: ~/.fast_agents/skills/
        user_home = Path.home()
        user_skills = user_home / ".fast_agents" / "skills"
        self.add_skill_path(user_skills, source="user")
    
    def add_skill_path(self, path: Path, source: str = "custom"):
        """添加Skill扫描路径"""
        path = Path(path).resolve()
        if path.exists() and path.is_dir():
            self.skill_paths.append((path, source))
    
    def scan_skills(self) -> Dict[str, SkillMetadata]:
        """
        扫描所有路径下的Skill，构建索引（Level 1: 仅加载元数据）
        返回skill_name -> SkillMetadata的映射
        """
        self._skill_index.clear()
        self._loaded_skills.clear()
        
        for base_path, source in self.skill_paths:
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
                        
                        self._skill_index[metadata.name] = metadata
                except Exception as e:
                    print(f"[SkillLoader] 加载Skill {skill_dir.name} 失败: {e}")
        
        return self._skill_index
    
    def _find_skill_file(self, skill_dir: Path) -> Optional[Path]:
        """在目录中查找SKILL.md文件"""
        for filename in self.SKILL_FILENAMES:
            candidate = skill_dir / filename
            if candidate.exists() and candidate.is_file():
                return candidate
        return None
    
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
        metadata = SkillMetadata(
            name=meta_dict.get("name", skill_dir.name),
            description=meta_dict.get("description", ""),
            version=str(meta_dict.get("version", "1.0.0")),
            author=meta_dict.get("author"),
            license=meta_dict.get("license"),
            compatibility=meta_dict.get("compatibility"),
            allowed_tools=meta_dict.get("allowed-tools", meta_dict.get("allowed_tools", [])),
            tags=meta_dict.get("tags", []),
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
                    span = self._tracer.start_span(
                        name=f"skill.load.{skill_name}",
                        kind="skill",
                        attributes={
                            "skill.name": skill_name,
                        }
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
                        span.finish(error=f"Skill not found: {skill_name}")
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
                        span.finish(error=f"Skill dir not found: {skill_dir}")
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
                        span.finish(error="SKILL.md not found")
                    except Exception:
                        pass
                return result
            
            try:
                content = skill_file.read_text(encoding="utf-8")
                _, instructions = self._split_frontmatter(content)
                instructions = instructions.strip()
                
                # 扫描scripts目录
                scripts = []
                scripts_dir = skill_dir / self.SCRIPTS_DIR
                if scripts_dir.exists() and scripts_dir.is_dir():
                    scripts = [
                        p for p in scripts_dir.iterdir()
                        if p.is_file() and not p.name.startswith("_")
                    ]
                
                # 扫描references目录
                references = []
                refs_dir = skill_dir / self.REFERENCES_DIR
                if refs_dir.exists() and refs_dir.is_dir():
                    references = [
                        p for p in refs_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".json", ".yaml", ".yml"]
                    ]
                
                result = SkillLoadResult(
                    success=True,
                    skill_name=skill_name,
                    metadata=metadata,
                    content=content,
                    instructions=instructions,
                    scripts=scripts,
                    references=references,
                )
                
                self._loaded_skills[skill_name] = result
                
                if span:
                    try:
                        span.set_attribute("skill.cached", False)
                        span.set_attribute("skill.scripts_count", len(scripts))
                        span.set_attribute("skill.references_count", len(references))
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
                        span.finish(error=str(e))
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
        """
        读取Skill参考文档（Level 3: 按需读取）
        """
        span = None
        try:
            # Trace打点开始
            if self._tracer is not None:
                try:
                    span = self._tracer.start_span(
                        name=f"skill.read_reference.{skill_name}",
                        kind="skill",
                        attributes={
                            "skill.name": skill_name,
                            "skill.reference": ref_name,
                        }
                    )
                except Exception:
                    span = None
            
            skill_result = self.load_skill(skill_name)
            if not skill_result.success:
                error_msg = f"读取参考文档失败: {'; '.join(skill_result.errors)}"
                if span:
                    try:
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 查找匹配的参考文件
            skill_dir = skill_result.metadata.skill_path
            refs_base_dir = (skill_dir / self.REFERENCES_DIR).resolve()
            
            # 安全：路径穿越防护
            ref_path = (refs_base_dir / ref_name).resolve()
            if not str(ref_path).startswith(str(refs_base_dir)):
                error_msg = f"非法参考文档路径：禁止访问references目录外的文件"
                if span:
                    try:
                        span.finish(error=error_msg)
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
                        span.finish(error=error_msg)
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
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
        except Exception:
            return f"读取参考文档失败: 内部错误"
    
    def run_script(
        self,
        skill_name: str,
        script_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_SCRIPT_TIMEOUT,
    ) -> str:
        """
        执行Skill脚本（Level 3: 按需执行）
        安全加固：严格路径校验、禁止路径穿越、仅允许指定扩展名、输出截断
        """
        span = None
        try:
            # Trace打点开始
            if self._tracer is not None:
                try:
                    span = self._tracer.start_span(
                        name=f"skill.run_script.{skill_name}",
                        kind="skill",
                        attributes={
                            "skill.name": skill_name,
                            "skill.script": script_name,
                        }
                    )
                except Exception:
                    span = None
            
            skill_result = self.load_skill(skill_name)
            if not skill_result.success:
                error_msg = f"执行脚本失败: {'; '.join(skill_result.errors)}"
                if span:
                    try:
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            skill_dir = skill_result.metadata.skill_path
            scripts_base_dir = (skill_dir / self.SCRIPTS_DIR).resolve()
            
            # 安全：路径穿越防护
            script_path = (scripts_base_dir / script_name).resolve()
            if not str(script_path).startswith(str(scripts_base_dir)):
                error_msg = f"非法脚本路径：禁止访问scripts目录外的文件"
                if span:
                    try:
                        span.finish(error=error_msg)
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
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 安全：校验扩展名
            suffix = script_path.suffix.lower()
            if suffix not in self.ALLOWED_SCRIPT_EXTENSIONS:
                error_msg = f"不支持执行该类型脚本：{suffix}，仅允许 {', '.join(self.ALLOWED_SCRIPT_EXTENSIONS)}"
                if span:
                    try:
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
            
            # 构建命令
            args = args or {}
            cmd = []
            
            # Windows平台专用：根据扩展名选择解释器
            if suffix == ".py":
                cmd = [sys.executable, str(script_path)]
            elif suffix in [".ps1", ".bat", ".cmd"]:
                cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
            else:
                # 理论上不会走到这里，上面已经校验扩展名
                cmd = [str(script_path)]
            
            # 添加参数：--key value格式
            for key, value in args.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value is not False and value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(skill_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
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
                        span.finish()
                    except Exception:
                        pass
                
                return final_output
            
            except subprocess.TimeoutExpired:
                error_msg = f"脚本执行超时（{timeout}秒）"
                if span:
                    try:
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
            except Exception as e:
                error_msg = f"执行脚本失败: {str(e)}"
                if span:
                    try:
                        span.finish(error=error_msg)
                    except Exception:
                        pass
                return error_msg
        except Exception:
            # 打点兜底：绝不让观测代码打挂主流程
            return f"执行脚本失败: 内部错误"
    
    def get_skill_index_text(self) -> str:
        """获取所有Skill的索引文本（注入system prompt的Level 1内容）"""
        if not self._skill_index:
            self.scan_skills()
        
        if not self._skill_index:
            return "当前没有可用的技能。"
        
        lines = ["## 可用技能列表", ""]
        for name, meta in sorted(self._skill_index.items()):
            lines.append(meta.to_index_text())
        
        lines.extend([
            "",
            "提示：当你判断需要使用某个技能时，使用 load_skill 工具加载完整技能指令。",
            "技能采用渐进式加载：当前只显示技能名称和描述，加载后可看到完整执行流程。",
        ])
        
        return "\n".join(lines)
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有已发现的Skill"""
        if not self._skill_index:
            self.scan_skills()
        
        return [meta.to_dict() for meta in self._skill_index.values()]
