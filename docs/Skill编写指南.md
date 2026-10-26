# fast_agents Skill 编写指南

## 📖 概述

Skill（技能）是 fast_agents 的核心能力扩展机制，遵循 Anthropic 开放 SKILL.md 标准（2025.12 发布，已被 Claude Code、OpenAI Codex、GitHub Copilot、Microsoft Agent Framework 等 33+ 产品采纳）。

### 核心设计理念

Skill 不是普通的 Tool 函数，而是**封装了专业知识和标准作业流程（SOP）的能力包**。采用业界标准的**三级渐进式加载机制**，完美解决"上下文窗口有限"与"能力无限扩展"之间的矛盾：

| 加载级别 | 时机 | 内容 | Token开销 | 作用 |
|---------|------|------|----------|------|
| Level 1 | 会话启动时 | name + description | ~100 token/Skill | 意图匹配，Agent知道有什么技能 |
| Level 2 | 判断需要时 | 完整SKILL.md指令 | 按需加载 | 获得完整执行流程和规范 |
| Level 3 | 执行需要时 | references文档 + scripts脚本 | 按需加载 | 获取细节数据、执行外部脚本 |

### Skill vs Tool 的区别

| 维度 | Tool（你已有的工具） | Skill（新的技能系统） |
|------|---------------------|----------------------|
| 本质 | 原子化执行接口 | 标准化工作流程/专家知识 |
| 职责 | 执行单个动作（调API、读文件、执行命令） | 编排多个Tool完成复杂任务，包含业务规则和SOP |
| 比喻 | 手脚 | 大脑的方法论 |
| 开发方式 | Python代码 | Markdown文档 + 可选脚本，非程序员也能写 |
| 上下文 | 全量描述常驻 | 渐进式加载，大幅节省token |

---

## 📁 目录结构

### Skill存放路径（按优先级从高到低）
1. **项目级**：`./skills/` 目录（在你的项目根目录下创建，可随Git共享给团队）
2. **用户级**：`C:/Users/&lt;你的用户名&gt;/.fast_agents/skills/` （Windows）或 `~/.fast_agents/skills/`（Mac/Linux，所有项目通用）
3. **代码注册**：通过 `registry.register()` 在代码中注册
4. **自定义路径**：通过 `registry.add_skill_path()` 添加

### 单个Skill的标准结构

一个Skill就是一个独立文件夹，文件夹名就是技能名（必须小写，用连字符分隔）：

```
skills/
└── skill-name-here/           # 技能根目录，名称必须是kebab-case
    ├── SKILL.md               # ✅ 必需：技能主文件（元数据+执行指令）
    ├── scripts/               # 📂 可选：可执行脚本目录
    │   ├── helper.py          # Python脚本
    │   └── validate.bat       # Windows批处理/PowerShell脚本也支持
    ├── references/            # 📂 可选：参考文档目录（按需加载）
    │   ├── policy.md          # 政策规范
    │   └── templates.md       # 模板示例
    └── assets/                # 📂 可选：静态资源目录
        └── template.html      # 模板文件等
```

&gt; ✅ 最小可用：只需要一个SKILL.md文件即可工作，其他都是可选的。

---

## ✍️ SKILL.md 编写规范

SKILL.md 是技能的核心文件，由两部分组成：
- **YAML Frontmatter**：文件开头 `---` 包裹的元数据
- **Markdown Body**：给AI看的执行指令正文

### 1. Frontmatter 元数据字段

```yaml
---
name: skill-name                # ✅ 必填，技能唯一标识
description: 技能描述            # ✅ 必填，最关键字段！决定Agent何时触发
version: "1.0"                  # 可选，版本号
author: your-name               # 可选，作者
license: MIT                    # 可选，许可证
compatibility: Windows/macOS/Linux  # 可选，兼容性说明
tags: [tag1, tag2]              # 可选，标签，用于分类
allowed-tools: [Bash, Read]     # 可选，预授权可用的工具列表
metadata:                       # 可选，自定义元数据
  key: value
---
```

#### 字段详细说明

| 字段 | 必填 | 规范 |
|------|------|------|
| **name** | ✅ | 1-64字符，只能小写字母a-z、数字0-9、连字符-；不能以连字符开头/结尾，不能有连续连字符；必须与文件夹名完全一致 |
| **description** | ✅ | 1-1024字符，写清楚**"做什么"+"什么时候用"**，一定要包含触发关键词！这是Agent选择技能的唯一依据。 |

#### ✅ 好的description示例：
```yaml
description: 生成规范的Git提交信息并执行提交。当用户说"提交代码"、"git commit"、"生成commit信息"时使用，遵循Conventional Commits规范。
```
说明：清晰说了功能+触发场景+关键词，Agent一看就知道什么时候用。

#### ❌ 差的description示例：
```yaml
description: 帮助处理git相关的事情
```
说明：太模糊，没有触发关键词，Agent不知道什么时候该调用。

---

### 2. Markdown 正文编写指南

正文是给AI看的"标准作业手册"，告诉AI"按什么步骤、什么规范、输出什么格式"。建议结构：

```markdown
# 技能名称（清晰易懂）

## 适用场景
明确告诉AI什么时候该用这个技能，什么时候不该用。

## 执行流程
分步骤写清楚执行顺序，编号1/2/3/4：
1. 第一步做什么（检查什么、调用什么工具）
2. 第二步做什么
3. 第三步做什么
...

## 输出格式
明确要求输出的格式、规范、示例。

## 注意事项
列出边界情况、禁止事项、异常处理方式。
```

#### 正文编写最佳实践：

1. **步骤要具体**：不要写"分析数据"，要写"执行git diff --cached查看已暂存的更改内容"
2. **明确终止条件**：比如"如果没有暂存文件，提示用户先git add，终止流程"
3. **明确禁止事项**：比如"不要自动执行git add .添加所有文件"
4. **给出反例**：容易出错的地方，给出错误示例和正确示例
5. **长度控制**：建议正文控制在500行以内，太长的内容拆分到references目录按需加载
6. **引用文件**：引用参考文档直接写文件名，Agent会用read_skill_reference加载
7. **脚本调用**：说明脚本的调用方式和参数要求

---

## 🚀 快速上手：10分钟写第一个Skill

我们来写一个"代码评审"技能，你只需要3步：

### Step 1：创建目录和SKILL.md
在项目根目录下创建：
```
skills/
└── code-review/
    └── SKILL.md
```

### Step 2：编写SKILL.md
```markdown
---
name: code-review
description: 审查代码质量，发现bug、安全漏洞、性能问题和代码规范问题。当用户说"代码评审"、"review代码"、"帮我看看这段代码"时使用。
version: "1.0"
tags: ["code", "review", "quality"]
allowed-tools: [Read, Write]
---

# 代码评审技能

## 执行流程

严格按照以下Checklist执行评审，覆盖所有维度：

### 1. 功能性检查
- [ ] 代码逻辑是否正确，能否实现需求目标
- [ ] 有没有明显的bug（边界条件、空指针、除以零等）
- [ ] 错误处理是否完善，异常是否被捕获

### 2. 安全性检查
- [ ] 有没有SQL注入、XSS等安全漏洞
- [ ] 用户输入是否做了校验和转义
- [ ] 敏感信息（密钥、密码）是否硬编码

### 3. 性能检查
- [ ] 有没有N+1查询问题
- [ ] 循环中有没有重复的数据库/网络调用
- [ ] 大数据量操作有没有做分页/流式处理

### 4. 代码规范
- [ ] 命名是否清晰易懂
- [ ] 函数是否过长（建议不超过50行）
- [ ] 有没有重复代码可以抽取复用
- [ ] 注释是否充分且准确

## 输出格式

评审结果按以下格式输出：

```
## 代码评审结果

### 🔴 严重问题（必须修复）
1. **&lt;问题位置&gt;**：&lt;问题描述&gt;
   - 影响：&lt;会导致什么后果&gt;
   - 建议修复方案：&lt;具体怎么改&gt;

### 🟡 建议优化
1. **&lt;问题位置&gt;**：&lt;问题描述&gt;
   - 建议：&lt;优化方案&gt;

### 🟢 亮点
1. &lt;写得好的地方&gt;
```

## 注意事项
- 不要只说"这里不好"，要给出具体的修改建议
- 严重问题必须给出代码示例说明怎么改
- 如果代码整体质量不错，要肯定做得好的地方
- 不要吹毛求疵，重点关注真正影响正确性、安全性、可维护性的问题
```

### Step 3：重启Agent即可使用
重启你的Agent，它会自动发现这个新技能。当你说"帮我review一下这个文件"，Agent会自动加载这个技能并按照标准流程执行评审！

---

## 💻 代码型Skill开发

除了文件型Skill，你也可以用Python代码定义Skill，适合包含复杂执行逻辑的场景。两种方式二选一即可，也可以混合使用。

### 方式1：@skill装饰器（最简单）

```python
from fast_agents.skills import skill

@skill(
    name="calculate",
    description="执行安全的数学计算，支持加减乘除、幂运算、常用数学函数。当用户需要计算数学表达式时使用。",
    tags=["math", "calculation"]
)
def calculate(expression: str) -&gt; str:
    """
    数学计算执行规范：
    1. 检查表达式是否包含危险内容
    2. 使用安全命名空间执行计算
    3. 返回计算结果，保留4位小数
    """
    import math
    # ... 你的计算逻辑 ...
    return "结果"
```

### 方式2：继承BaseSkill类（适合复杂场景）

```python
from fast_agents.skills import BaseSkill

class DataAnalysisSkill(BaseSkill):
    name = "data-analysis"
    description = "CSV数据分析，计算统计指标"
    version = "1.0.0"
    tags = ["data", "analysis"]
    
    def get_instructions(self) -&gt; str:
        # 返回给Agent看的执行流程指令
        return """
        数据分析流程：
        1. 读取CSV文件
        2. 计算统计指标
        3. 输出Markdown表格结果
        """
    
    def execute(self, csv_content: str, **kwargs) -&gt; str:
        # 实际执行逻辑
        return "分析结果"
```

### 注册代码型Skill

```python
from fast_agents.skills import SkillRegistry

registry = SkillRegistry()
registry.register(calculate)
registry.register(DataAnalysisSkill)
registry.scan()  # 同时扫描文件型Skill
```

---

## 🔌 集成到你的Agent

只需要3步即可把Skill系统集成到你已有的Agent代码中：

### 1. 初始化Skill系统
```python
from fast_agents.skills import SkillRegistry, SkillMiddleware

# 在Agent.__init__中添加：
self.skill_registry = SkillRegistry()
self.skill_middleware = SkillMiddleware(self.skill_registry)

# 注册Skill内置3个工具到你的tool系统
self.tools.extend(self.skill_middleware.get_tools())
```

### 2. 构建System Prompt时注入内容
```python
# 在构建system prompt时：
full_system_prompt = self.skill_middleware.before_agent_run(original_system_prompt)
```

### 3. 工具调用后通知中间件
```python
# 在执行完工具调用，得到result后：
result = self.skill_middleware.after_tool_call(tool_name, tool_args, result)
```

---

## 🎯 系统提示词优化

为了让Skill机制更好地工作，请把以下内容加入到你的基础System Prompt中：

```markdown
## 技能（Skills）使用规范

你拥有一系列可复用的专业技能来帮助你更高质量、更规范地完成任务。所有技能采用渐进式加载机制。

### 使用流程
1. **匹配识别**：根据用户问题，在"可用技能列表"中找到最匹配的技能
2. **加载指令**：调用`load_skill`工具加载该技能的完整执行流程——**不要猜测技能内部流程，必须先加载再执行**
3. **严格执行**：按照加载后的指令步骤、规范、输出格式执行任务
4. **按需读取**：如果技能提到需要参考文档，调用`read_skill_reference`读取；需要执行脚本，调用`run_skill_script`

### 重要原则
- ✅ 匹配场景必须使用：如果用户需求符合某技能描述，必须加载并遵循该技能
- ❌ 不要凭空创造流程：不要自创步骤，严格按照技能中写的流程执行
- 📝 不要提及技能本身：正常回复用户即可，不需要说"我正在使用xx技能"
- ⚠️ 不确定就加载：拿不准是否适用时，可以先加载看看具体流程再决定
- 🚫 不要过度使用：简单问题、通用知识问答无需加载技能

### 工具说明
- `load_skill(skill_name)`：加载技能完整指令
- `read_skill_reference(skill_name, reference_name)`：读取技能参考文档
- `run_skill_script(skill_name, script_name, args)`：执行技能关联脚本
```

---

## 💡 最佳实践与避坑指南

### ✅ 推荐做法
1. **Skill粒度要合适**：一个Skill解决一类问题，不要太大也不要太碎
   - 好：git-commit、code-review、weekly-report
   - 不好：git-helper（太大）、add-file-to-git（太碎）

2. **description一定要写好触发关键词**
   - 想想用户会怎么说，把那些说法都写进description里
   - 例如：用户说"提交代码"、"commit一下"、"生成commit信息"都应该触发git-commit

3. **步骤要具体到工具调用级别**
   - 不要写"查看更改"，要写"执行`git diff --cached`查看暂存区更改"

4. **明确禁止事项**
   - 把"不要做什么"写清楚，比如"不要自动执行git add ."

5. **长内容拆分到references**
   - SKILL.md正文不要超过500行，详细表格、模板、案例放到references目录按需加载

### ❌ 常见坑
1. **name与文件夹名不一致**：必须完全一致（小写连字符），否则扫描不到
2. **description写太简单**："处理pdf"这种描述Agent根本不知道什么时候触发
3. **正文写得太笼统**：没有具体步骤，Agent还是不知道怎么做
4. **全量加载太多Skill**：不要一次加载所有Skill全文，会把上下文撑爆，用默认的渐进式加载就好
5. **脚本没有异常处理**：scripts里的脚本要做好错误捕获，返回清晰的错误信息

---

## 📚 内置API参考

### SkillRegistry 主要方法
```python
registry = SkillRegistry(extra_skill_paths=None)  # 创建注册表
registry.register(skill)                         # 注册代码型Skill
registry.add_skill_path(path)                    # 添加文件型Skill扫描路径
registry.scan()                                  # 扫描文件型Skill
registry.load_skill(name)                        # 加载完整Skill
registry.execute_skill(name, **kwargs)           # 执行代码型Skill
registry.get_index_prompt()                      # 获取Level 1索引提示词
registry.list_skills()                           # 列出所有Skill
registry.get_tools()                             # 获取3个内置工具
```

### SkillMiddleware 配置参数
```python
middleware = SkillMiddleware(
    registry=registry,
    auto_inject_index=True,      # 自动注入索引到system prompt
    unload_after_turn=False,     # 每轮结束后卸载已加载Skill（省token，但多轮对话要重新加载）
    max_loaded_skills=5,         # 最多同时加载的Skill数量，LRU淘汰
)
```

---

## 🎉 为什么这套架构比LangChain传统Tool更好？

1. **省Token**：100个Skill只占~10k token索引，全量注入Tool描述要50k+ token
2. **行为更稳定**：SOP固化下来，Agent每次执行步骤一致，不会漏步骤
3. **易开发维护**：写Markdown比写Python代码简单，产品/运营也能写Skill
4. **可复用可分享**：一个文件夹就是一个Skill，直接拷贝就能用，跨团队跨项目共享
5. **开放标准**：遵循Anthropic开放标准，写的Skill未来可以直接在Claude Code、Copilot等产品中使用
6. **和现有Tool完美互补**：Tool做原子动作，Skill做流程编排，各司其职
