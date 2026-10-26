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

from .models import SkillMetadata, SkillLoadResult


class SkillLoader:
    """
    Skill加载器：扫描目录，解析SKILL.md，加载技能内容
    
    支持扫描路径:
        - 项目级: ./skills/
        - 用户级: ~/.fast_agents/skills/
        - 自定义路径: 通过add_skill_path添加
    """
    
    SKILL_FILENAMES = ["SKILL.md", "skill.md", "Skills.md", "skills.md"]
    SCRIPTS_DIR = "scripts"
    REFERENCES_DIR = "references"
    ASSETS_DIR = "assets"
    
    def __init__(self, extra_paths: Optional[List[Path]] = None):
        self.skill_paths: List[Tuple[Path, str]] = []  # (path, source_label)
        self._loaded_skills: Dict[str, SkillLoadResult] = {}
        self._skill_index: Dict[str, SkillMetadata] = {}
        
        # 添加默认路径
        self._add_default_paths()
        
        # 添加自定义路径
        if extra_paths:
            for p in extra_paths:
                self.add_skill_path(p, source="custom")
    
    def _add_default_paths(self):
        """添加默认扫描路径"""
        # 项目级: 当前工作目录下的skills
        project_skills = Path.cwd() / "skills"
        self.add_skill_path(project_skills, source="project")
        
        # 用户级: ~/.fast_agents/skills/
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
        # 先确保索引已构建
        if not self._skill_index:
            self.scan_skills()
        
        # 检查是否已加载
        if skill_name in self._loaded_skills:
            return self._loaded_skills[skill_name]
        
        # 查找元数据
        metadata = self._skill_index.get(skill_name)
        if not metadata:
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=[f"Skill '{skill_name}' 不存在"],
            )
        
        skill_dir = metadata.skill_path
        if not skill_dir or not skill_dir.exists():
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=[f"Skill目录不存在: {skill_dir}"],
            )
        
        skill_file = self._find_skill_file(skill_dir)
        if not skill_file:
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=["SKILL.md文件不存在"],
            )
        
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
            return result
        
        except Exception as e:
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=[f"加载Skill失败: {str(e)}"],
            )
    
    def read_reference(self, skill_name: str, ref_name: str) -> str:
        """
        读取Skill参考文档（Level 3: 按需读取）
        """
        skill_result = self.load_skill(skill_name)
        if not skill_result.success:
            return f"读取参考文档失败: {'; '.join(skill_result.errors)}"
        
        # 查找匹配的参考文件
        skill_dir = skill_result.metadata.skill_path
        ref_path = skill_dir / self.REFERENCES_DIR / ref_name
        
        if not ref_path.exists():
            # 尝试模糊匹配
            refs_dir = skill_dir / self.REFERENCES_DIR
            if refs_dir.exists():
                for p in refs_dir.iterdir():
                    if p.name.lower() == ref_name.lower() or ref_name.lower() in p.name.lower():
                        ref_path = p
                        break
        
        if not ref_path.exists():
            return f"参考文档 '{ref_name}' 不存在"
        
        try:
            return ref_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"读取参考文档失败: {str(e)}"
    
    def run_script(
        self,
        skill_name: str,
        script_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> str:
        """
        执行Skill脚本（Level 3: 按需执行）
        """
        skill_result = self.load_skill(skill_name)
        if not skill_result.success:
            return f"执行脚本失败: {'; '.join(skill_result.errors)}"
        
        skill_dir = skill_result.metadata.skill_path
        script_path = skill_dir / self.SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            # 尝试添加扩展名查找
            for ext in [".py", ".ps1", ".bat", ".cmd", ".sh", ""]:
                candidate = script_path.with_suffix(ext)
                if candidate.exists():
                    script_path = candidate
                    break
        
        if not script_path.exists():
            return f"脚本 '{script_name}' 不存在"
        
        # 构建命令
        args = args or {}
        cmd = []
        
        # 根据扩展名选择解释器
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            cmd = [sys.executable, str(script_path)]
        elif suffix in [".ps1", ".bat", ".cmd"]:
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
        elif suffix in [".sh"]:
            cmd = ["bash", str(script_path)]
        else:
            # 可执行文件直接运行
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
            if result.stdout:
                output.append(result.stdout)
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr}")
            if result.returncode != 0:
                output.append(f"[退出码: {result.returncode}]")
            
            return "\n".join(output).strip()
        
        except subprocess.TimeoutExpired:
            return f"脚本执行超时（{timeout}秒）"
        except Exception as e:
            return f"执行脚本失败: {str(e)}"
    
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
