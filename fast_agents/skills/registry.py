"""
Skill注册表：统一管理代码型Skill和文件型Skill
提供注册、查找、加载、列出、启用/禁用等功能
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from .base import BaseSkill
from .errors import SkillNotFoundError
from .loader import SkillLoader
from .models import SkillLoadResult, SkillMetadata


class SkillRegistry:
    """
    Skill注册表：统一管理所有类型的Skill
    
    支持:
        - 代码型Skill：通过register()注册BaseSkill子类或@skill装饰的函数
        - 文件型Skill：通过SkillLoader自动扫描SKILL.md目录
        - 【C11】运行时启用/禁用：disable_skill/enable_skill可动态开关技能，
          禁用后从索引隐藏、不可加载；按tag批量开关
    """
    
    def __init__(
        self,
        extra_skill_paths: Optional[List[Path]] = None,
        default_allow_network: bool = False,
        default_allow_write_paths: Optional[List[str]] = None,
        default_env_whitelist: Optional[List[str]] = None,
        default_script_timeout: int = 30,
        include_default_paths: bool = True,
    ):
        self._code_skills: Dict[str, BaseSkill] = {}
        self._code_metadata: Dict[str, SkillMetadata] = {}
        self.loader: SkillLoader = SkillLoader(
            extra_paths=extra_skill_paths,
            default_allow_network=default_allow_network,
            default_allow_write_paths=default_allow_write_paths,
            default_env_whitelist=default_env_whitelist,
            default_script_timeout=default_script_timeout,
            include_default_paths=include_default_paths,
        )
        self._tools: Optional[List[Callable]] = None
        self._skill_tools: Optional[List[Any]] = None  # 包装后的ToolBase工具
        self._tracer = None
        # 【C11】禁用状态：skill_name -> True 表示被禁用
        self._disabled_skills: set[str] = set()
        # 【C11】按tag禁用：tag名集合，带这些tag的技能全部禁用
        self._disabled_tags: set[str] = set()
        # 【C11】中间件引用（延迟绑定，由middleware在初始化时设置）
        self._middleware: Optional[Any] = None
    
    def set_tracer(self, tracer) -> None:
        """注入trace打点器，透传到loader"""
        self._tracer = tracer
        self.loader.set_tracer(tracer)
    
    def set_middleware(self, middleware) -> None:
        """【C11】绑定中间件引用，启用/禁用操作时通知中间件同步状态"""
        self._middleware = middleware
    
    # ========== 【C11】启用/禁用API ==========
    
    def is_skill_disabled(self, skill_name: str, _all_meta: Optional[Dict[str, SkillMetadata]] = None) -> bool:
        """检查技能是否被禁用（单个禁用或tag禁用）
        【性能优化】支持调用方传入已查询好的_all_meta，避免内部再调get_metadata触发一次扫描。
        """
        if skill_name in self._disabled_skills:
            return True
        if _all_meta is not None:
            meta = _all_meta.get(skill_name)
        else:
            meta = self.get_metadata(skill_name)
        if meta is not None and meta.tags:
            for tag in meta.tags:
                if tag in self._disabled_tags:
                    return True
        return False
    
    def disable_skill(self, skill_name: str) -> str:
        """禁用单个技能：从索引隐藏、不可加载、若已加载则自动卸载"""
        all_meta = self.get_all_metadata()
        if skill_name not in all_meta:
            return f"禁用失败: 技能 '{skill_name}' 不存在"
        
        self._disabled_skills.add(skill_name)
        
        # 通知中间件卸载已加载的该技能并重算白名单
        if self._middleware is not None:
            self._middleware._on_skill_disabled(skill_name)
        
        return f"已禁用技能 '{skill_name}'，该技能已从索引中隐藏，无法被加载。"
    
    def enable_skill(self, skill_name: str) -> str:
        """启用单个技能：恢复索引可见和可加载"""
        all_meta = self.get_all_metadata()
        if skill_name not in all_meta:
            return f"启用失败: 技能 '{skill_name}' 不存在"
        
        self._disabled_skills.discard(skill_name)
        
        # 通知中间件重算白名单（如果该技能已加载则不需要额外操作，未加载则索引自动恢复）
        if self._middleware is not None:
            self._middleware._on_skill_enabled(skill_name)
        
        return f"已启用技能 '{skill_name}'，该技能现在可以正常加载使用。"
    
    def disable_skills_by_tag(self, tag: str) -> str:
        """按标签禁用一批技能：所有带有该tag的技能都被禁用"""
        self._disabled_tags.add(tag)
        
        # 找到所有带该tag的技能，已加载的要卸载
        disabled_names = []
        all_meta = self.get_all_metadata()
        for name, meta in all_meta.items():
            if tag in (meta.tags or []):
                disabled_names.append(name)
                if self._middleware is not None:
                    self._middleware._on_skill_disabled(name)
        
        if disabled_names:
            return f"已按标签 '{tag}' 禁用 {len(disabled_names)} 个技能: {', '.join(sorted(disabled_names))}"
        return f"已添加禁用标签 '{tag}'，当前没有发现带有该标签的技能。"
    
    def enable_skills_by_tag(self, tag: str) -> str:
        """按标签启用一批技能：移除该tag的禁用标记"""
        if tag not in self._disabled_tags:
            return f"标签 '{tag}' 当前不在禁用列表中。"
        
        self._disabled_tags.discard(tag)
        
        # 通知中间件重算白名单
        if self._middleware is not None:
            self._middleware._on_tag_enabled(tag)
        
        # 找到被该tag禁用的技能名用于回复（注意：单个被disable_skill禁用的不会恢复）
        enabled_names = []
        all_meta = self.get_all_metadata()
        for name, meta in all_meta.items():
            if tag in (meta.tags or []) and name not in self._disabled_skills:
                enabled_names.append(name)
        
        if enabled_names:
            return f"已按标签 '{tag}' 启用 {len(enabled_names)} 个技能: {', '.join(sorted(enabled_names))}"
        return f"已移除禁用标签 '{tag}'。"
    
    def disable_all_skills(self) -> str:
        """禁用所有技能：清空索引，已加载的全部卸载（技能管理工具仍可用）"""
        all_meta = self.get_all_metadata()
        skill_names = list(all_meta.keys())
        
        # 标记所有技能为禁用
        self._disabled_skills.update(skill_names)
        
        # 通知中间件卸载所有
        if self._middleware is not None:
            self._middleware._on_all_skills_disabled()
        
        return f"已禁用全部 {len(skill_names)} 个技能，技能管理工具仍可用于恢复。"
    
    def enable_all_skills(self) -> str:
        """启用所有技能：清空所有禁用状态"""
        self._disabled_skills.clear()
        self._disabled_tags.clear()
        
        if self._middleware is not None:
            self._middleware._on_all_skills_enabled()
        
        return "已启用所有技能，所有禁用状态已清除。"
    
    def get_disabled_skills(self) -> Dict[str, Any]:
        """获取当前禁用状态信息"""
        return {
            "disabled_skills": sorted(self._disabled_skills),
            "disabled_tags": sorted(self._disabled_tags),
        }
    
    def get_skill_source(self, skill_name: str) -> Optional[str]:
        """【C8】查询技能来源（builtin/user/project/custom/code），不存在返回None"""
        meta = self.get_metadata(skill_name)
        if meta is None:
            return None
        return getattr(meta, 'source', None)
    
    def list_skill_conflicts(self) -> List[Dict[str, Any]]:
        """【C8】列出所有同名技能冲突（仅返回被覆盖的低优先级条目，用于调试/排查）"""
        conflicts: List[Dict[str, Any]] = []
        # 收集所有文件型技能（按来源优先级排序，低到高）
        file_meta = self.loader.scan_skills()
        seen: Dict[str, Dict[str, Any]] = {}
        sorted_paths = sorted(
            self.loader.skill_paths,
            key=lambda item: self.loader.SOURCE_PRIORITY.get(item[1], 2)
        )
        for base_path, source in sorted_paths:
            if not base_path.exists():
                continue
            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = self.loader._find_skill_file(skill_dir)
                if not skill_file:
                    continue
                try:
                    meta = self.loader._parse_skill_metadata(skill_file, skill_dir, source)
                    if not meta:
                        continue
                    if meta.name != skill_dir.name:
                        meta.name = skill_dir.name
                    if meta.name in seen:
                        # 低优先级被高优先级覆盖
                        old = seen[meta.name]
                        conflicts.append({
                            "skill_name": meta.name,
                            "old_source": old["source"],
                            "old_path": old["path"],
                            "new_source": source,
                            "new_path": str(skill_dir),
                            "winner": "new" if self.loader.SOURCE_PRIORITY.get(source, 2) >= self.loader.SOURCE_PRIORITY.get(old["source"], 2) else "old",
                        })
                    else:
                        seen[meta.name] = {"source": source, "path": str(skill_dir)}
                except Exception:
                    pass
        # 代码型覆盖文件型
        for name, code_meta in self._code_metadata.items():
            if name in seen:
                old = seen[name]
                conflicts.append({
                    "skill_name": name,
                    "old_source": old["source"],
                    "old_path": old["path"],
                    "new_source": "code",
                    "new_path": "(代码型注册)",
                    "winner": "new",
                })
        return conflicts
    
    def register(self, skill: Union[BaseSkill, Type[BaseSkill], Any]) -> None:
        """
        注册代码型Skill
        
        支持三种方式:
            1. 传入BaseSkill实例: registry.register(MySkill())
            2. 传入BaseSkill子类: registry.register(MySkill)
            3. 传入@skill装饰的函数: registry.register(my_skill_func)
        【C8】代码型技能优先级最高(source="code")，同名会覆盖所有文件型技能，打印冲突警告
        """
        skill_instance: Optional[BaseSkill] = None
        
        # 处理@skill装饰的函数
        if hasattr(skill, 'skill_class'):
            skill_instance = skill.skill_class()
        # 处理BaseSkill子类
        elif isinstance(skill, type) and issubclass(skill, BaseSkill):
            skill_instance = skill()
        # 处理BaseSkill实例
        elif isinstance(skill, BaseSkill):
            skill_instance = skill
        else:
            raise TypeError(f"无法注册类型 {type(skill)}，请传入BaseSkill子类/实例或@skill装饰的函数")
        
        name = skill_instance.metadata.name
        # 【C8】代码型技能统一标记source="code"（最高优先级）
        if not hasattr(skill_instance.metadata, 'source') or not skill_instance.metadata.source:
            skill_instance.metadata.source = "code"
        else:
            skill_instance.metadata.source = "code"
        
        if name in self._code_skills:
            print(f"[SkillRegistry] 警告: 代码型Skill '{name}' 已存在，将被覆盖")
        
        # 【C8】代码型技能覆盖文件型技能警告
        all_file_meta = self.loader.scan_skills()
        if name in all_file_meta:
            old_meta = all_file_meta[name]
            old_path = str(getattr(old_meta, 'skill_path', 'unknown'))
            print(
                f"[SkillRegistry] 警告: 技能 '{name}' 存在同名冲突："
                f"文件型 [{old_meta.source}] {old_path} 被 代码型注册技能 覆盖（代码型优先级最高）"
            )
        
        self._code_skills[name] = skill_instance
        self._code_metadata[name] = skill_instance.metadata
    
    def unregister(self, skill_name: str) -> bool:
        """注销代码型Skill"""
        if skill_name in self._code_skills:
            del self._code_skills[skill_name]
            del self._code_metadata[skill_name]
            return True
        return False
    
    def add_skill_path(self, path: Union[str, Path], source: str = "custom") -> None:
        """添加文件型Skill扫描路径"""
        self.loader.add_skill_path(Path(path), source=source)
    
    def scan(self) -> None:
        """扫描文件型Skill，重建索引（强制）"""
        self.loader.invalidate_index_cache()
        self.loader.scan_skills(force=True)

    def scan_skills(self) -> Dict[str, "SkillMetadata"]:
        """【便捷别名】扫描文件型Skill并返回元数据字典。等价于 scan() + get_all_metadata()，方便外部/测试直接调用。"""
        self.scan()
        return self.get_all_metadata()

    def is_skill_loaded(self, skill_name: str) -> bool:
        """查询指定技能是否已加载（Level 2 内容已读入内存）。
        合并 loader 侧的文件型缓存与 middleware 侧的 LRU 已加载名单，给外部/测试一个统一入口。
        """
        # 文件型：loader._loaded_skills 记录了已经做过 Level 2 加载的结果
        if hasattr(self.loader, "_loaded_skills") and skill_name in self.loader._loaded_skills:
            return True
        # middleware 的 LRU 名单（当前 Agent 会话中已实际加载过）
        if self._middleware is not None:
            loaded = getattr(self._middleware, "_loaded_skills", None)
            if loaded is not None and skill_name in loaded:
                return True
        # 代码型：注册过即视为"可用"，但代码型不存在 Level 2 加载流程，按 loader 缓存不存在即未加载
        return False

    def get_all_metadata(self) -> Dict[str, SkillMetadata]:
        """获取所有Skill的元数据（代码型+文件型），包含已禁用的技能（内部管理用）
        【性能优化】默认走缓存：loader.scan_skills 仅在根目录mtime变化时才真正重扫。
        """
        # 确保文件型已扫描（mtime未变则走缓存，不会做磁盘I/O）
        file_metadata = self.loader.scan_skills()

        # 合并：代码型优先？还是文件型优先？这里采用后注册优先
        all_metadata = {}
        all_metadata.update(file_metadata)
        all_metadata.update(self._code_metadata)
        return all_metadata

    def get_enabled_metadata(self) -> Dict[str, SkillMetadata]:
        """【C11】获取所有未被禁用的Skill元数据（用于给LLM展示索引）
        【性能优化】先一次性拿到all_meta，再传入is_skill_disabled，避免其内部重复扫描。
        """
        all_meta = self.get_all_metadata()
        return {
            name: meta for name, meta in all_meta.items()
            if not self.is_skill_disabled(name, _all_meta=all_meta)
        }

    def get_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """获取指定Skill的元数据（查全量，不受启用/禁用影响）
        【性能优化】支持传入已查询好的_all_meta避免重复扫描。
        """
        return self.get_all_metadata().get(skill_name)
    
    # ========== 【C12】依赖检查与版本比对工具 ==========
    
    @staticmethod
    def _parse_version(v: str) -> tuple:
        """解析版本号 '1.2.3' -> (1,2,3)；缺省位补0；非数字部分截断"""
        if v is None:
            return (0, 0, 0)
        # 去掉 pre-release 后缀（-xxx）
        v = str(v).strip().split('-')[0].split('+')[0]
        parts = []
        for p in v.split('.'):
            m = re.match(r'^(\d+)', p)
            parts.append(int(m.group(1)) if m else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])
    
    @staticmethod
    def _parse_dep_spec(dep_str: str) -> tuple:
        """解析依赖声明 'other-skill>=1.2.0' -> ('other-skill', '>=', (1,2,0))；无约束返回 (name, None, None)"""
        m = re.match(r'^([a-z0-9][a-z0-9-]*[a-z0-9]?)\s*(>=|<=|==|>|<)\s*(\d+(?:\.\d+){0,2})\s*$', dep_str.strip())
        if not m:
            return (dep_str.strip(), None, None)
        name, op, ver = m.group(1), m.group(2), m.group(3)
        return (name, op, SkillRegistry._parse_version(ver))
    
    @staticmethod
    def _ver_satisfies(actual: tuple, op: str, required: tuple) -> bool:
        """判断 actual op required 是否成立"""
        if op == '>=': return actual >= required
        if op == '<=': return actual <= required
        if op == '==': return actual == required
        if op == '>':  return actual > required
        if op == '<':  return actual < required
        return True
    
    def _collect_dependency_report(
        self, skill_name: str, metadata: SkillMetadata,
        _chain: Optional[List[str]] = None,
        _all_meta: Optional[Dict[str, SkillMetadata]] = None,
    ) -> Dict[str, List[str]]:
        """
        【C12】递归收集技能依赖检查的完整报告，返回结构化字典：
        {'errors': [...], 'missing_tools': [...], 'missing_skills': [...]}
        - errors: 人类可读的错误描述（与旧 _check_dependencies 返回值逐项一致）
        - missing_tools: 未注册的依赖工具名（去重排序）
        - missing_skills: 未满足的依赖技能名（未找到/已被禁用/版本不满足，去重排序）
        检查项：
        1. requires 声明的其他技能是否存在/已加载、版本是否满足
        2. requires_tools 声明的工具（含allowed_tools）是否已注册
        3. 框架版本兼容性（min/max_fast_agents_version）
        4. 循环依赖检测（DFS路径记录）
        【性能优化】支持传入_all_meta复用已查好的元数据，避免递归时重复扫描。
        """
        errors: List[str] = []
        missing_tools: List[str] = []
        missing_skills: List[str] = []
        _chain = _chain or []

        # 循环依赖检测
        if skill_name in _chain:
            cycle = _chain[_chain.index(skill_name):] + [skill_name]
            errors.append(f"检测到循环依赖: {' -> '.join(cycle)}")
            return {'errors': errors, 'missing_tools': missing_tools, 'missing_skills': missing_skills}
        current_chain = _chain + [skill_name]

        # 1. 框架版本检查
        try:
            from ..version import __version__ as framework_version
        except Exception:
            framework_version = "0.0.0"
        fw_ver = self._parse_version(framework_version)

        if metadata.min_fast_agents_version:
            min_v = self._parse_version(metadata.min_fast_agents_version)
            if fw_ver < min_v:
                errors.append(
                    f"该技能需要 fast_agents >= {metadata.min_fast_agents_version}，"
                    f"当前版本为 {framework_version}，请升级框架"
                )
        if metadata.max_fast_agents_version:
            max_v = self._parse_version(metadata.max_fast_agents_version)
            if fw_ver > max_v:
                errors.append(
                    f"该技能要求 fast_agents <= {metadata.max_fast_agents_version}，"
                    f"当前版本为 {framework_version}，存在兼容性风险"
                )

        # 2. 工具依赖检查（allowed_tools + requires_tools 合并去重）
        required_tools_set = set(metadata.allowed_tools or []) | set(metadata.requires_tools or [])
        required_tools_set.discard("*")  # 通配符表示全部，不需要检查

        available_tools: Set[str] = set()
        if self._middleware is not None and getattr(self._middleware, '_tool_registry', None) is not None:
            available_tools = set(self._middleware._tool_registry.list_names())
        # 技能系统内置工具默认可用
        available_tools |= {
            "load_skill", "read_skill_reference", "run_skill_script",
            "list_loaded_skills", "disable_skill", "enable_skill",
            "disable_skills_by_tag", "enable_skills_by_tag",
            "check_skill_dependencies",  # 【C12】预检工具
            "enable_skill_hot_reload", "disable_skill_hot_reload", "reload_skills_now",  # 【C14】热重载管理工具始终可用
        }

        missing_tools = sorted(t for t in required_tools_set if t not in available_tools)
        if missing_tools:
            errors.append(
                f"缺失依赖工具: {', '.join(missing_tools)}；"
                f"请确保这些工具已注册到 Agent"
            )

        # 3. 技能依赖检查（递归）
        loaded_skill_names: Set[str] = {skill_name}
        if self._middleware is not None:
            loaded_skill_names |= set(getattr(self._middleware, '_loaded_skills', []))

        # 【性能优化】复用_all_meta避免递归时重复扫描
        if _all_meta is None:
            _all_meta = self.get_all_metadata()

        for dep_spec in (metadata.requires or []):
            dep_name, op, required_ver = self._parse_dep_spec(dep_spec)

            # 依赖的技能是否在索引中
            dep_meta = _all_meta.get(dep_name)
            if dep_meta is None:
                errors.append(f"依赖技能 '{dep_name}' 未找到，请确保该技能已安装到技能目录")
                missing_skills.append(dep_name)
                continue

            # 依赖的技能是否被禁用（复用_all_meta）
            if self.is_skill_disabled(dep_name, _all_meta=_all_meta):
                errors.append(f"依赖技能 '{dep_name}' 当前已被禁用，请先启用它")
                missing_skills.append(dep_name)
                continue

            # 版本约束检查
            if op is not None and required_ver is not None:
                dep_ver = self._parse_version(dep_meta.version)
                if not self._ver_satisfies(dep_ver, op, required_ver):
                    op_display = {'==': '==', '>=': '≥', '<=': '≤', '>': '>', '<': '<'}.get(op, op)
                    errors.append(
                        f"依赖技能 '{dep_name}' 版本不满足要求：需要 {op_display} "
                        f"{'.'.join(str(x) for x in required_ver)}，当前版本为 {dep_meta.version}"
                    )
                    missing_skills.append(dep_name)

            # 递归检查依赖的依赖（若未加载才递归；已加载的视为通过，避免重复）
            if dep_name not in loaded_skill_names:
                sub_report = self._collect_dependency_report(dep_name, dep_meta, current_chain, _all_meta=_all_meta)
                errors.extend(sub_report['errors'])
                missing_tools.extend(sub_report['missing_tools'])
                missing_skills.extend(sub_report['missing_skills'])

        return {
            'errors': errors,
            'missing_tools': sorted(set(missing_tools)),
            'missing_skills': sorted(set(missing_skills)),
        }
    
    def _check_dependencies(
        self, skill_name: str, metadata: SkillMetadata,
        _chain: Optional[List[str]] = None,
        _all_meta: Optional[Dict[str, SkillMetadata]] = None,
    ) -> List[str]:
        """
        【C12】递归检查技能依赖，返回错误信息列表（空列表表示全部通过）
        检查项：
        1. requires 声明的其他技能是否存在/已加载、版本是否满足
        2. requires_tools 声明的工具（含allowed_tools）是否已注册
        3. 框架版本兼容性（min/max_fast_agents_version）
        4. 循环依赖检测（DFS路径记录）
        【性能优化】支持传入_all_meta复用已查好的元数据，避免递归时重复扫描。
        【C12】内部复用 _collect_dependency_report 只取 errors，函数签名与返回类型保持不变。
        """
        return self._collect_dependency_report(
            skill_name, metadata, _chain=_chain, _all_meta=_all_meta
        )['errors']

    def check_skill_dependencies(self, skill_name: str) -> Dict[str, Any]:
        """
        【C12】公开API：预检技能依赖，不实际加载。
        【C12】两个分支统一返回 6 个 key，结构稳定：
        {'skill_name': str, 'ok': bool, 'errors': [...], 'warnings': [...],
         'missing_tools': [...], 'missing_skills': [...]}
        - missing_tools: 未注册的依赖工具名（去重排序）
        - missing_skills: 未满足的依赖技能名（未找到/被禁用/版本不满足，去重排序）
        """

        result = {
            'skill_name': skill_name,
            'ok': True,
            'errors': [],
            'warnings': [],
            'missing_tools': [],
            'missing_skills': [],
        }
        all_meta = self.get_all_metadata()
        meta = all_meta.get(skill_name)
        if meta is None:
            result['ok'] = False
            result["errors"] = [f"技能 '{skill_name}' 不存在"]
            return result

        report = self._collect_dependency_report(skill_name, meta, _all_meta=all_meta)
        errors = report['errors']
        if errors and len(errors) > 0:
            result['ok'] = False
            result["errors"] = errors
            result['missing_tools'] = report['missing_tools']
            result['missing_skills'] = report['missing_skills']
        return result

    def load_skill(self, skill_name: str) -> SkillLoadResult:
        """加载完整Skill内容（Level 2）；【C11】被禁用的技能不可加载；【C12】加载前自动做依赖检查"""
        # 一次性拿到全量元数据，后续复用避免重复扫描
        all_meta = self.get_all_metadata()

        # 【C11】禁用检查（复用all_meta避免再次扫描）
        if self.is_skill_disabled(skill_name, _all_meta=all_meta):
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=[f"技能 '{skill_name}' 当前已被禁用，无法加载。请先使用 enable_skill 启用它。"],
            )

        # 先检查代码型Skill
        if skill_name in self._code_skills:
            code_meta = self._code_metadata.get(skill_name)
            if code_meta is not None:
                dep_errors = self._check_dependencies(skill_name, code_meta, _all_meta=all_meta)
                if dep_errors:
                    hint = "\n提示：可调用 check_skill_dependencies 查看详细依赖状态。"
                    return SkillLoadResult(
                        success=False,
                        skill_name=skill_name,
                        errors=dep_errors + [hint],
                    )
            return self._code_skills[skill_name].load()

        # 再检查文件型Skill
        # 【C12】先查元数据存在
        meta = all_meta.get(skill_name)
        if meta is None:
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                errors=[f"技能 '{skill_name}' 不存在"],
            )

        # 【C12】依赖检查
        dep_errors = self._check_dependencies(skill_name, meta, _all_meta=all_meta)
        if dep_errors:
            hint = "\n提示：可调用 check_skill_dependencies 工具查看详细依赖状态；\n" \
                   "若依赖其他技能，请先使用 load_skill 加载依赖技能。"
            return SkillLoadResult(
                success=False,
                skill_name=skill_name,
                metadata=meta,
                errors=dep_errors + [hint],
            )

        return self.loader.load_skill(skill_name)
    
    def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """执行代码型Skill"""
        if skill_name not in self._code_skills:
            # 文件型Skill不直接execute，按指令执行即可
            return f"文件型Skill '{skill_name}' 请按照加载的指令执行，可调用run_script运行脚本"
        
        return self._code_skills[skill_name].execute(**kwargs)
    
    async def aexecute_skill(self, skill_name: str, **kwargs) -> Any:
        """异步执行代码型Skill"""
        if skill_name not in self._code_skills:
            return self.execute_skill(skill_name, **kwargs)
        
        return await self._code_skills[skill_name].aexecute(**kwargs)
    
    def read_reference(self, skill_name: str, ref_name: str) -> str:
        """读取Skill参考文档"""
        if skill_name in self._code_skills:
            return f"代码型Skill '{skill_name}' 不支持参考文档"
        return self.loader.read_reference(skill_name, ref_name)
    
    def read_asset(self, skill_name: str, asset_name: str) -> str:
        """【C15】读取Skill静态资产（文本返回内容，二进制仅返回元信息）"""
        if skill_name in self._code_skills:
            return f"代码型Skill '{skill_name}' 不支持静态资产"
        return self.loader.read_asset(skill_name, asset_name)
    
    def run_script(self, skill_name: str, script_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """运行Skill脚本"""
        if skill_name in self._code_skills:
            return f"代码型Skill '{skill_name}' 不支持脚本执行"
        return self.loader.run_script(skill_name, script_name, args)
    
    def get_index_prompt(self) -> str:
        """获取注入System Prompt的Skill索引文本（Level 1常驻）；【C11】只展示启用的技能"""
        all_meta = self.get_enabled_metadata()
        
        if not all_meta:
            return "## 可用技能\n当前没有可用的技能。"
        
        lines = ["## 可用技能（Skills）", ""]
        lines.append("你可以使用以下技能完成特定任务，采用渐进式加载机制：")
        lines.append("")

        for name, meta in sorted(all_meta.items()):
            lines.append(meta.to_index_text())

        lines.extend([
            "",
            "### 技能使用方法",
            "1. 根据用户需求判断是否需要使用某个技能",
            "2. 使用 `load_skill` 工具加载对应技能的完整指令",
            "3. 按照加载后的执行流程完成任务",
            "4. 需要参考文档时使用 `read_skill_reference`，需要执行脚本时使用 `run_skill_script`，需要读取静态资源（模板/样例/配置等）时使用 `read_skill_asset`",
            "",
            "注意：不要猜测技能的内部流程，务必先加载再执行。二进制资产（图片/字体/二进制数据）不可直接注入对话，请通过脚本处理。",
        ])
        
        return "\n".join(lines)
    
    def get_tools(self) -> List[Any]:
        """
        获取Skill相关的工具函数（裸函数），包含：
        load_skill, read_skill_reference, read_skill_asset, run_skill_script, list_loaded_skills,
        disable_skill/enable_skill/disable_skills_by_tag/enable_skills_by_tag,
        check_skill_dependencies,
        enable_skill_hot_reload/disable_skill_hot_reload/reload_skills_now
        注意：如需符合ToolBase规范的工具对象，请使用build_skill_tools()
        """
        if self._tools is None:
            self._tools = self._create_skill_functions()
        return self._tools
    
    def _create_skill_functions(self) -> List[Callable]:
        """创建Skill系统内置的工具函数（裸函数）；【C11】新增4个启用/禁用管理工具"""
        
        def load_skill(skill_name: str) -> str:
            """
            加载指定技能的完整执行指令。当你判断需要使用某个技能时，首先调用此工具获取详细的执行流程。
            
            Args:
                skill_name: 要加载的技能名称，必须是上面列表中存在的技能名
            """
            result = self.load_skill(skill_name)
            if not result.success:
                return f"加载技能失败: {'; '.join(result.errors)}"
            return result.full_prompt
        
        def read_skill_reference(skill_name: str, reference_name: str) -> str:
            """
            读取技能的参考文档内容。当技能指令中提到需要参考某个文档时，调用此工具获取详细内容。
            
            Args:
                skill_name: 所属技能名称
                reference_name: 参考文档的文件名
            """
            return self.read_reference(skill_name, reference_name)
        
        # 【C15】静态资产读取
        def read_skill_asset(skill_name: str, asset_name: str) -> str:
            """
            读取技能的静态资源（assets/目录下的文件）。
            文本类资源（.md/.txt/.json/.yaml/.csv/.py等）直接返回内容；
            二进制资源（图片/字体/二进制数据）仅返回元信息与绝对路径，不返回正文，需通过脚本处理。
            
            Args:
                skill_name: 所属技能名称
                asset_name: 资产相对路径（支持子目录，如 templates/report.html）
            """
            return self.read_asset(skill_name, asset_name)
        
        def run_skill_script(skill_name: str, script_name: str, args: Optional[Dict[str, Any]] = None) -> str:
            """
            执行技能关联的脚本。当技能指令要求运行某个脚本时使用。
            
            Args:
                skill_name: 所属技能名称
                script_name: 脚本文件名（含扩展名）
                args: 传递给脚本的参数，键值对形式
            """
            return self.run_script(skill_name, script_name, args)
        
        def list_loaded_skills() -> str:
            """
            列出当前已加载的技能，以及每个技能允许使用的工具列表。
            用于确认当前工具使用权限，避免调用不被允许的工具。
            """
            # 这个函数实际会被middleware替换，这里只保留占位符
            return "请通过SkillMiddleware获取已加载技能列表"
        
        # 【C11】启用/禁用管理工具
        def disable_skill(skill_name: str) -> str:
            """
            禁用指定技能。禁用后该技能从可用技能列表中隐藏，无法被加载；若已加载则自动卸载。
            适用于确认某个技能不适用于当前任务场景时临时关闭。
            
            Args:
                skill_name: 要禁用的技能名称
            """
            return self.disable_skill(skill_name)
        
        def enable_skill(skill_name: str) -> str:
            """
            启用之前被禁用的技能。启用后技能恢复可见、可正常加载使用。
            
            Args:
                skill_name: 要启用的技能名称
            """
            return self.enable_skill(skill_name)
        
        def disable_skills_by_tag(tag: str) -> str:
            """
            按标签批量禁用技能。所有带有指定标签的技能都会被禁用。
            适用于一次性关闭某一类技能（如测试类、实验类技能）。
            
            Args:
                tag: 要禁用的技能标签名
            """
            return self.disable_skills_by_tag(tag)
        
        def enable_skills_by_tag(tag: str) -> str:
            """
            按标签批量启用技能。移除之前通过disable_skills_by_tag设置的标签禁用。
            
            Args:
                tag: 要启用的技能标签名
            """
            return self.enable_skills_by_tag(tag)
        
        # 【C12】依赖预检工具
        def check_skill_dependencies(skill_name: str) -> str:
            """
            预检某个技能的依赖是否满足（工具、其他技能、框架版本），不实际加载技能。
            加载技能失败提示依赖缺失时，可先用本工具查看详细状态。
            
            Args:
                skill_name: 要检查的技能名称
            """
            result = self.check_skill_dependencies(skill_name)
            if result['ok']:
                return f"技能 '{skill_name}' 依赖检查全部通过，可以正常加载。"
            lines = [f"技能 '{skill_name}' 依赖检查未通过："]
            lines.extend(f"- {e}" for e in result['errors'])
            # 【C12】附加结构化摘要，方便直接取用缺失清单，不用从长句里回抠工具名/技能名
            if result['missing_tools']:
                lines.append(f"缺失工具: {', '.join(result['missing_tools'])}")
            if result['missing_skills']:
                lines.append(f"缺失/未满足的技能: {', '.join(result['missing_skills'])}")
            return "\n".join(lines)
        
        # 【C14】热重载管理工具
        def enable_skill_hot_reload(interval: float = 2.0) -> str:
            """
            启用技能热重载：开发模式下后台监控技能目录文件变更，SKILL.md/scripts/references修改后自动重载，无需重启Agent。
            
            Args:
                interval: 轮询间隔秒数，默认2秒，范围1-60秒
            """
            if self._middleware is None:
                return "启用失败: SkillMiddleware未初始化，热重载功能不可用"
            return self._middleware.enable_hot_reload(interval)
        
        def disable_skill_hot_reload() -> str:
            """
            关闭技能热重载，停止后台文件监控线程。生产环境部署时建议关闭。
            """
            if self._middleware is None:
                return "热重载未启用"
            return self._middleware.disable_hot_reload()
        
        def reload_skills_now() -> str:
            """
            手动立即触发一次技能重扫，立即检查新增/修改/删除的技能并应用变更，不用等待轮询间隔。
            当你新添加了技能目录或修改了技能文件想立刻生效时使用。
            """
            if self._middleware is None:
                return "重载失败: SkillMiddleware未初始化"
            return self._middleware.reload_skills_now()
        
        return [load_skill, read_skill_reference, read_skill_asset, run_skill_script, list_loaded_skills,
                disable_skill, enable_skill, disable_skills_by_tag, enable_skills_by_tag,
                check_skill_dependencies,
                enable_skill_hot_reload, disable_skill_hot_reload, reload_skills_now]
    
    def build_skill_tools(self) -> List[Any]:
        """
        构建符合框架ToolBase规范的Skill工具列表，和build_memory_tools()风格对齐
        直接返回可注册到ToolRegistry的工具对象，无需手动包装
        """
        if self._skill_tools is None:
            # 延迟导入，避免循环依赖
            from ..tools import StructuredTool
            funcs = self._create_skill_functions()
            self._skill_tools = [StructuredTool.from_function(f) for f in funcs]
        return self._skill_tools
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有可用Skill信息（含启用/禁用状态）"""
        all_meta = self.get_all_metadata()
        result = []
        for meta in all_meta.values():
            d = meta.to_dict()
            d["disabled"] = self.is_skill_disabled(meta.name, _all_meta=all_meta)
            result.append(d)
        return result

    def list_skills_metadata(self) -> List["SkillMetadata"]:
        """【便捷别名】列出所有可用Skill的元数据对象列表（测试/调试用），等价于 get_all_metadata().values()。"""
        return list(self.get_all_metadata().values())
