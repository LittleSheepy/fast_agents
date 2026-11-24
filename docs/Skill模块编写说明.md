# Skill 模块编写说明

> **对应代码**：`fast_agents/skills/`（已同步至最新代码，2026-11-17）
> **读者**：想给 FastAgents 增加技能、或想照着这套思路给自己的 Agent 加技能机制的人
> **本文只写「已经实现的东西」**。「还没做但打算做」的部分一律放在 `docs/演进路线图.md`，本文不重复。

---

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：Skill 解决什么问题](#二先想清楚skill-解决什么问题)
- [三、主流框架 Skill 实现对比](#三主流框架-skill-实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：三级渐进式加载](#五核心机制三级渐进式加载)
- [六、核心数据模型](#六核心数据模型)
- [七、逐个文件讲解](#七逐个文件讲解)
- [八、快速上手](#八快速上手)
- [九、API 速查](#九api-速查)
- [十、设计决策与常见坑](#十设计决策与常见坑)
- [十一、测试与验证](#十一测试与验证)
- [十二、下一步演进路线](#十二下一步演进路线)
- [十三、参考](#十三参考)
- [附录：发布前自检清单](#附录发布前自检清单)

---

## 一、更新历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| 1.9.1 | 2026-11-18 | 【K1】技能trace打点融合修正：对齐trace模块在94bf92b新增的`attributes`/`set_attribute`/`SpanKind.SKILL`/finish自动落盘能力——①四个入口（`load_skill`/`read_reference`/`read_asset`/`run_script`）的`kind`由字符串`"skill"`改为枚举`SpanKind.SKILL`，统一使用`attributes`承载框架标准属性（`skill.name`/`skill.op`/`asset.size_bytes`/`script.exit_code`等，命名空间点号风格），`metadata`不再承载skill.*键；②所有错误路径（技能不存在、路径穿越、文件缺失、参数校验失败、命令注入拦截、敏感路径拦截、危险命令拦截、脚本退出码非0、超时、异常）显式传`status=SpanStatus.ERROR`，修复原代码中`error`不自动翻转status导致错误span被记为SUCCESS的bug；③每个span补`skill.op`标识操作类型（`load`/`read_reference`/`read_asset`/`run_script`），便于评估器按op聚合；④`SpanKind`/`SpanStatus`改为延迟导入（模块级`_ensure_trace_imports()`），不传tracer时零开销且避免skills→trace反向强依赖；测试基线保持**84/84**全绿；零新增第三方依赖 | wxy |
| 1.9.0 | 2026-11-18 | 【C8】同名技能冲突消解落地（对齐 Claude Code 分层作用域）：新增来源优先级 `builtin` < `user` < `project` < `custom` < `code`，同名技能按优先级高者覆盖低者、扫描时打印覆盖警告；代码型技能（`register()` / `BaseSkill` / `@skill`）恒为最高优先级；新增 `SkillConflictError` 异常与 `get_skill_source()` / `list_skill_conflicts()` 两个调试 API。顺带刷齐文档欠账：§四 异常类计数 10→11、§五 assets 文本扩展名补 `.tsv`/`.htm`、§七 `__init__.py` 导出补 `AssetNotFoundError` / `SkillConflictError`、§八 补 `scripts/<脚本名>.schema.json` 示例、§九 `get_tools()` 12→13 件套、§十一 测试基线 60/64→84/84、§十二 补 C18/C8 归档、§十三 参考补 `file_skill_dependency_demo.py`、附录 import 补 `AssetNotFoundError` | wxy |
| 1.8.1 | 2026-11-18 | 【C12】`check_skill_dependencies` 返回结构统一（方案B）：技能不存在与正常两个分支统一返回 6 个 key（`skill_name`/`ok`/`errors`/`warnings`/`missing_tools`/`missing_skills`）；新增 `_collect_dependency_report()` 结构化收集，`_check_dependencies` 收敛为薄封装（签名与返回类型 `List[str]` 不变），`load_skill` 两处调用零改动；`missing_tools`/`missing_skills` 返回真实去重清单（未注册工具 / 未找到·被禁用·版本不满足的技能）；LLM 工具返回文案附结构化摘要；新增 3 项测试，测试总数 81→**84/84** 全绿 | wxy |
| 1.8.0 | 2026-11-18 | 【C18】方案B对齐主流规范，一级元数据回归极简：① **脚本参数Schema从SKILL.md frontmatter外移到`scripts/&lt;脚本名&gt;.schema.json`**（和同名`.py`脚本同目录，改完脚本顺手改schema，脚本和schema文件绑定便于维护），frontmatter不再写长scripts段；② **沙箱三字段（`allow_network`/`allow_write_paths`/`env_whitelist`）从`SkillMetadata`上移到SkillLoader/Registry/Middleware初始化全局配置**，宿主统一管理沙箱策略，每个技能默认追加自身`tmp/`到可写路径白名单；③ 保留参数校验全部能力（类型/必填/enum/范围校验、类型自动转换、命令注入防护、敏感路径拦截），仅调整schema读取位置；④ `SkillMetadata`一级元数据仅保留`name/description/version/tags/requires`等核心字段，对齐Anthropic Agent Skills规范，单技能常驻token约100；新增3项C18测试，测试总数76/76全绿；工具总数保持13个不变 | wxy |
| 1.7.0 | 2026-11-18 | 【C15】静态资产（assets）完整读取（遵循三级渐进式披露，对齐 Anthropic/Microsoft 主流实践）：`load_skill` 扫描 `assets/` 目录（文本/二进制分流）；新增 `read_skill_asset` LLM 工具（路径穿越防护、文本按 UTF-8 返回内容、二进制/超大文件仅返回元信息）；**`assets` 清单只进入 `SkillLoadResult`**（Level 2 加载后通过 `full_prompt` 展示），不进入 `SkillMetadata` 常驻索引，避免随 function schema 常驻 token 膨胀；`full_prompt` 分"文本资源/二进制资源"两段展示；工具总数 12→13；6 项 C15 测试通过 | wxy |
| 1.6.2 | 2026-11-18 | 【文档】同步死代码 `get_skill_index_text()` 删除（commit `27e2d7a`）：7.4 节介绍改为已移除说明、10.8 坑标注已清理、示例与 C15 条目收窄 | — |
| 1.6.1 | 2026-11-17 | 【文档】第十二章「下一步演进路线」按《xx模块编写说明模版.md》改为 **12.1 已交付 / 12.2 待办（按优先级）** 两张表，编号对齐《演进路线图.md》C 系列 | — |
| 1.6.0 | 2026-11-17 | 【C14】技能热重载：mtime 增量检测 + 后台监控线程 + 三个管理工具 | — |
| 1.5.0 | 2026-11-17 | 【C6-2】脚本沙箱增强：参数 Schema 校验与类型转换、命令注入防护、敏感路径拦截、网络默认禁用、危险命令扫描 | — |
| 1.4.0 | 2026-11-17 | 【C12】技能依赖与前置检查：工具 / 技能 / 框架版本 / 循环依赖校验 + `check_skill_dependencies` 预检 | — |
| 1.3.0 | 2026-11-17 | 【C11】技能运行时启用 / 禁用 + 按 tag 批量开关（8 个 registry API + 4 个 LLM 工具） | — |
| 1.2.0 | 2026-11-17 | 【C10】`allowed_tools` 落地生效：加载技能后按白名单过滤 LLM 可见工具，卸载自动恢复全量 | — |
| 1.1.0 | 2026-11-17 | 【P0】接入 trace 打点、统一异常体系 `errors.py`、`build_skill_tools()` 工具工厂、默认扫描路径统一 `.skills`、脚本执行安全加固 | — |
| 1.0.0 | 2026-10-27 | **初版**：首次编写，逐行对齐 `fast_agents/skills/` 真实实现 | — |

> 约定：本文档当前版本为**初版**（v1.0.0）；以后每次改动 `fast_agents/skills/` 的对外行为（新增工具、改元数据字段、改加载层级），都要在表格**最上方**补一行新版本——更新记录按版本号**倒序**维护，最新的排在最上。

---

## 二、先想清楚：Skill 解决什么问题

### 2.1 一句话概括

**Tool 是「一个动作」，Skill 是「一套流程」。**

- 工具层解决的是「Agent 能不能做某件事」。比如 `CalculatorTool` 能算数、`ReadTool` 能读文件。
- 技能层解决的是「Agent 知不知道这件事该按什么规矩做」。比如「生成周报」这件事本身不需要新工具，它需要的是**一套步骤、格式要求和质量规范**。

把流程直接写进 System Prompt 会立刻遇到两个瓶颈：

1. **上下文被撑爆**。你有 30 个流程，每个 800 字，全塞进 System Prompt 就是 24000 字，而且每轮对话都要重发一遍。模型的注意力会被稀释，出现「指令漂移」——前面说过的规矩，聊到后面就忘了。
2. **没法复用和分发**。流程硬编码在某个项目的提示词里，换一个项目就得再抄一遍，团队之间也没法共享。

### 2.2 Skill 的答案：流程外置 + 按需加载

`fast_agents/skills/` 给出的答案是两件事：

- **把流程外置成文件夹**。一个技能 = 一个文件夹，里面放一份 `SKILL.md`（元数据 + 执行指令），外加可选的 `scripts/`（脚本）、`references/`（参考文档）、`assets/`（静态资源）。
- **把加载拆成三级，按需付费**。平时上下文里只留「技能名 + 一句话描述」的索引；模型判断需要了，才去加载完整指令；指令里提到某个细节文档，才再去读那份文档。

### 2.3 和 tools 模块的分工

| | `fast_agents/tools/` | `fast_agents/skills/` |
|---|---|---|
| 单元形态 | 一个函数 / 一个类 | 一个文件夹（至少一份 `SKILL.md`） |
| 谁来决定用不用 | 模型通过 Function Calling 直接调用 | 模型读了索引后，主动调 `load_skill` |
| 运行时做什么 | 真的执行代码、拿到返回值 | 加载文本指令（脚本可选） |
| 上下文成本 | 每个工具的 schema 常驻 | 索引常驻（约 100 字/个），正文按需 |
| 适合的东西 | 「能做什么」 | 「怎么做才规范」 |

这两者是**互补**关系，不是替代关系。技能指令里可以要求模型去调工具，技能的 `scripts/` 里也可以放脚本去调工具背后的能力。

```text
用户提问
   │
   ├─ 命中某个技能的描述 → load_skill 拿完整流程
   │        │
   │        └─ 流程里要求「用 CalculatorTool 算一下」
   │                 │
   │                 └─ 模型发起 Function Calling → CalculatorTool.run()
   │
   └─ 没命中任何技能 → 直接按普通 ReAct 流程回答
```

---

## 三、主流框架 Skill 实现对比

这一节回答一个问题：**别的框架怎么做的，我们为什么这么做。**

### 3.1 对比总表

| 框架 | 对应概念的官方叫法 | 载体形态 | 发现方式 | 典型加载策略 |
|------|------------------|---------|---------|------------|
| **Anthropic Claude Agent Skills** | Skill | 文件夹：`SKILL.md` + `scripts/` + `references/` + `assets/` | 扫描技能目录，元数据常驻 | **三级渐进式披露**（元数据 → 正文 → 资源） |
| **Semantic Kernel** | Plugin（`KernelPlugin`） | Python/C# 类 + `[KernelFunction]`；也支持目录、OpenAPI 文档 | 代码侧 `add_plugin` / `from_directory` / `from_openapi` | 全部函数的元数据一次性交给模型 |
| **CrewAI** | Knowledge（+ Tools / Memory） | 向量库（默认 ChromaDB），源可以是字符串、txt、PDF、CSV、Excel、JSON、URL | `knowledge_sources=[...]` | 检索式：按相关度召回片段 |
| **LangChain** | **没有原生 Skill**；只有 Tool（`BaseTool` / `StructuredTool`） | Python 类 | 代码硬注册进 Agent 的 `tools` 列表 | 全部工具的 schema 一次性交给模型 |
| **AutoGen** | 文档中出现 Skill 叫法 | `SKILL.md` | 运行时解析 | 运行时动态解析 |

### 3.2 逐个说清楚

**① Anthropic Agent Skills —— 我们的直接对标对象**

它是 2025 年 10 月随 Claude 生态推出的能力封装机制，2025 年 12 月被确立为**开放标准**。官方开源仓库是 `github.com/anthropics/skills`。

核心设计只有两点：

- **文件结构**：一个技能就是一个文件夹，`SKILL.md` 是唯一必需项，头部 YAML frontmatter 写元数据，正文写执行指令。
- **渐进式披露（Progressive Disclosure）**：分三层按需加载。
  - 第一层：frontmatter 元数据，**始终**在上下文里，一个技能约 100 token。
  - 第二层：`SKILL.md` 正文，模型判断相关时才读。
  - 第三层：`references/`、`scripts/`、`assets/`，**真正需要时**才碰。

官方建议 `SKILL.md` 正文控制在 5000 字以内，详细的参考资料一律拆到 `references/` 里去。frontmatter 里的 `name` 硬性要求 kebab-case（小写字母 + 连字符）。

`fast_agents/skills/` 的取舍是：**结构照抄，加载机制自己实现**。所以你会看到 `SKILL_FILENAMES`、`SCRIPTS_DIR`、`REFERENCES_DIR`、`ASSETS_DIR` 这四个常量，以及 `SkillLoadResult` 里的 `instructions` / `scripts` / `references` 三个字段——都是这条脉络。

**② Semantic Kernel —— 「插件」路线**

SK 用 `KernelPlugin` 表示一组函数，每个函数标 `[KernelFunction]`，靠 name + description + 参数元数据让模型理解。加载方式有三种：从对象加载（`from_object`）、从目录加载（`from_directory`，解析 `.py` / `.yaml` / `skprompt.txt` + `config.json`）、从 OpenAPI 文档加载（`from_openapi`）。

关键差别：**SK 的插件是"代码 + 元数据"，我们的技能是"文档 + 可选代码"**。SK 里不会写「步骤 1、步骤 2、步骤 3」这种给模型看的自然语言流程，它的行为规范藏在函数签名和 description 里。

**③ CrewAI —— 「知识」路线**

CrewAI 的概念矩阵是 Agent（角色）+ Task + Tools + Process + Knowledge + Memory。其中 Knowledge 是 RAG：把外部资料（PDF、CSV、网页……）灌进向量库，任务执行时按相关度召回片段。

差别在于：**CrewAI 的 Knowledge 解决的是"事实从哪来"，我们的 Skill 解决的是"流程按什么规矩走"**。前者是检索，后者是作业手册。两者可以共存——技能指令里完全可以要求模型去检索知识库。

**④ LangChain —— 没有 Skill，只有 Tool**

这点要单独强调，因为网上文章经常把 LangChain 的 Tool 说成 Skill。**LangChain 官方并没有 Skill 这个概念**，它的能力单元就是 `Tool`（`BaseTool` / `StructuredTool`）和 `@tool` 装饰器，用 `AgentExecutor` 或 LangGraph 串起来。社区文章里讲的 "LangChain Skill"，通常是指开发者自己把若干相关工具打包成一个模块的**工程实践**，不是框架级特性。

对比下来，LangChain 的路线是把所有工具的 JSON Schema 一次性交给模型（和我们 `to_openai_tools()` 做的事一样），区别是它多了一大堆抽象层：`BaseTool` → `CallbackHandler` → `AgentExecutor` → `Memory` → `AgentAction/AgentFinish`。抽象层的代价是调用链变长、出问题难定位。

**⑤ AutoGen —— 也叫 Skill**

AutoGen 生态中出现了以 `SKILL.md` 为载体的 Skill 叫法，运行时动态解析文档。思路和 Anthropic 路线一致。

### 3.3 一张图看懂三者关系

有一篇 2026 年的分析给了一个很好用的三分法，这里借用一下：

- **Skill = 合同**：位于语义层，面向 LLM 的自主决策，回答「这个能力能做什么、什么时候做、边界在哪、返回什么」。
- **Tool = 接口**：位于执行层，面向代码调用，有明确的入参出参。
- **Plugin = 装配**：位于运行时层，决定哪些能力在什么环境下被加载、如何热插拔。

这三者不是互斥的，而是**同一个业务逻辑的三层投影**。`fast_agents` 的现状是：

| 层 | fast_agents 里的对应物 |
|---|---|
| 语义层（合同） | `fast_agents/skills/`——`SKILL.md` + `BaseSkill` |
| 执行层（接口） | `fast_agents/tools/`——`ToolBase` / `StructuredTool` |
| 运行时层（装配） | `SkillMiddleware` + `SkillLoader` 的多路径扫描 |

### 3.4 结论：fast_agents 选的路

| 决策点 | 选择 | 理由 |
|---|---|---|
| 载体 | 文件夹 + `SKILL.md` | 跟开放标准对齐，文档和代码解耦，非程序员也能写技能 |
| 加载 | 三级渐进式 | 索引常驻、正文按需，上下文成本可控 |
| 脚本 | 支持 `scripts/`（多语言） | 流程里需要确定性计算时，交给脚本而不是让模型算 |
| 编码型技能 | 额外支持 `BaseSkill` / `@skill` | 给纯代码场景留后门，不必为了用技能而伪造一个文件夹 |
| 与工具的关系 | 技能工具本身也是工具 | 复用既有 `tools/` 注册表，不另起一套 |

---

## 四、模块架构

### 4.1 目录结构

```text
fast_agents/
└── skills/                          # ← 模块目录（单数 skills 这个包名本身就是复数，注意区分）
    ├── __init__.py                  # 统一导出（技能类 + 模型 + 11 个异常类）
    ├── models.py                    # SkillMetadata / SkillLoadResult / ScriptParam / ScriptSchema
    ├── base.py                      # BaseSkill 基类 + @skill 装饰器
    ├── errors.py                    # 统一异常体系（SkillError 及 11 个子类）
    ├── loader.py                    # 文件型技能：扫描 / 解析 / 加载 / 沙箱跑脚本 / 热重载检测
    ├── registry.py                  # 代码型 + 文件型统一注册表 + 13 个内置工具 + build_skill_tools()
    ├── middleware.py                # 注入 system prompt + LRU 生命周期 + 工具白名单 + 热重载线程
    ├── agent_integration_example.py # 接入已有 Agent 的 5 个接入点示范
    └── code_skill_example.py        # 代码型技能两种写法的完整示例
```

> **命名易混点**：`fast_agents/skills/` 是**代码模块**（Python 包）；而技能包存放的目录（形如 `.skills/demo-skill/`）是**数据目录**。文档里说到「技能目录」时指的是后者。

### 4.2 分层

```text
┌───────────────────────────────────────────────────────────┐
│  Agent 层（fast_agents/agents/react_agent/react_agent.py） │
│  _init_skills() / refresh_system_prompt() / run()          │
└───────────────┬───────────────────────────────────────────┘
                │ 使用
┌───────────────▼───────────────────────────────────────────┐
│  SkillMiddleware（middleware.py）                          │
│  before_agent_run() → 拼 system prompt                     │
│  after_tool_call()  → 捕获 load_skill 事件                 │
│  after_agent_turn() → 可选地卸载已加载技能                 │
└───────────────┬───────────────────────────────────────────┘
                │ 依赖
┌───────────────▼───────────────────────────────────────────┐
│  SkillRegistry（registry.py）                              │
│  register() / load_skill() / get_index_prompt() / get_tools()│
└───────┬─────────────────────────────┬─────────────────────┘
        │ 代码型                       │ 文件型
┌───────▼──────────────┐   ┌──────────▼─────────────────────┐
│  BaseSkill / @skill  │   │  SkillLoader（loader.py）       │
│  （base.py）          │   │  scan_skills() / load_skill()  │
│                      │   │  read_reference() / run_script()│
└───────┬──────────────┘   └──────────┬─────────────────────┘
        │                             │
        └──────────┬──────────────────┘
                   ▼
        ┌──────────────────────────┐
        │  models.py               │
        │  SkillMetadata           │
        │  SkillLoadResult         │
        └──────────────────────────┘
```

### 4.3 三条主线

模块源码共 9 个文件（含 `errors.py` 与两个示例），但真正跑起来是三条链路，读代码时按链路读最快。

**链路 A：索引注入（每次构造 / 刷新 system prompt 时）**

```text
SkillMiddleware.before_agent_run(system_prompt)
  └─ get_system_prompt_suffix()
       ├─ registry.get_index_prompt()        # Level 1：所有技能的 name + description
       └─ if self._loaded_skills:            # Level 2：已加载技能的完整内容
            └─ self._loaded_content[name]
```

**链路 B：技能加载（模型调用 `load_skill` 时）**

```text
模型决定用某技能
  └─ 调 load_skill(skill_name)
       ├─ SkillRegistry.load_skill(name)
       │    ├─ 代码型？→ BaseSkill.load()
       │    └─ 文件型？→ SkillLoader.load_skill(name) → SkillLoadResult.full_prompt
       └─ 工具返回文本
            └─ SkillMiddleware.after_tool_call("load_skill", {...}, result)
                 └─ _mark_skill_loaded()  # 记进 LRU，下一轮进 system prompt
                      └─ Agent.refresh_system_prompt()  # 把 Level 2 内容拼进去
```

**链路 C：扫描发现（`scan()` 时）**

```text
SkillRegistry.scan()
  └─ SkillLoader.scan_skills()
       ├─ 遍历 self.skill_paths（默认 + 自定义）
       ├─ 每个子目录找 SKILL.md（SKILL.md / skill.md / Skills.md / skills.md 都认）
       ├─ _parse_skill_metadata()：拆 YAML frontmatter → SkillMetadata
       ├─ metadata.validate()：name / description 校验
       └─ name 与目录名不一致 → 打印告警并强制改成目录名
```

---

## 五、核心机制：三级渐进式加载

这是整个模块的灵魂，也是省 token 的关键。三级分别是什么、由谁触发、代价多大：

| 级别 | 内容 | 何时进入上下文 | 谁触发 | 大致成本 |
|------|------|--------------|--------|---------|
| **Level 1** | 技能名 + 描述（元数据） | 每次构造 / 刷新 system prompt | `SkillRegistry.get_index_prompt()` | 约 100 字 / 技能 |
| **Level 2** | `SKILL.md` 正文（执行指令） | 模型调 `load_skill` 之后 | 模型主动决策 | 视正文长度，官方建议 ≤5000 字 |
| **Level 3** | `references/` 文档、`scripts/` 脚本、`assets/` 静态资源 | 指令里提到、模型按需读取/执行 | `read_skill_reference` / `run_skill_script` / `read_skill_asset` | 单份文档大小 / 文本资产 200KB 上限 |

### 5.1 Level 1：索引常驻

`SkillRegistry.get_index_prompt()` 生成的字面量长这样：

```text
## 可用技能（Skills）

你可以使用以下技能完成特定任务，采用渐进式加载机制：

- **demo-skill** [文件]: 这是一个演示技能，用于说明三级渐进式加载……

### 技能使用方法
1. 根据用户需求判断是否需要使用某个技能
2. 使用 `load_skill` 工具加载对应技能的完整指令
3. 按照加载后的执行流程完成任务
4. 需要参考文档时使用 `read_skill_reference`，需要执行脚本时使用 `run_skill_script`，需要读取静态资源（模板/样例/配置等文本文件）时使用 `read_skill_asset`（二进制资产仅返回元信息，需通过脚本处理）

注意：不要猜测技能的内部流程，务必先加载再执行。
```

没有技能时返回的是 `"## 可用技能\n当前没有可用的技能。"`（注意：这条**不带**后面的使用方法段落）。技能名后面会带 ` [代码]` 或 ` [文件]` 标签，用于区分技能类型。

### 5.2 Level 2：正文按需

模型调 `load_skill("demo-skill")` 后，拿到的是 `SkillLoadResult.full_prompt`，它是**拼装出来的**，不是原始文件内容：

```text
# Skill: demo-skill

## 技能说明
<frontmatter 里的 description>

## 执行指令
<SKILL.md 去掉 frontmatter 之后的正文>

## 可用脚本
- hello.py

## 参考文档
- notes.md

## 静态资源（assets/）      # 仅当技能带 assets/ 时出现，文本与二进制分两组
- templates/tpl.json
```

> **【C15】**技能带 `assets/` 时，`full_prompt` 会在末尾追加「## 静态资源（assets/）」段，文本资源与二进制资源分两组列出。

模型在下一步收到这段文本，同时 `SkillMiddleware` 把它记进 `_loaded_content`，下一轮 `refresh_system_prompt()` 时它会作为「## 已加载的技能详情」追加到 system prompt 末尾——**这样后续轮次不用重复调 `load_skill`**。

### 5.3 Level 3：资源按需

- `read_skill_reference(skill_name, reference_name)`：先精确匹配 `references/<name>`，匹配不到再做一次模糊匹配（大小写不敏感 + 子串包含）。
- `read_skill_asset(skill_name, asset_name)`：【C15】读取 `assets/` 下的静态资源。**文本类扩展名**（`.md/.txt/.json/.yaml/.yml/.csv/.tsv/.xml/.html/.htm/.css/.js/.py/.bat/.cmd/.ps1/.ini/.cfg/.toml/.log/.sql`）且不超过 **200KB** 时按 UTF-8 返回正文；二进制资源（图片/字体/可执行文件等）或超过上限的大文本仅返回元信息（类型/大小/绝对路径），不返回正文，需交由 `run_skill_script` 里的脚本处理。路径穿越防护与 `read_skill_reference` 一致：`..`、绝对路径都会被拦截。支持子目录（传 `templates/report.html` 这类相对路径即可）。
- `run_skill_script(skill_name, script_name, args)`：按扩展名挑解释器，见 [10.9](#109-跨平台注意事项)。

### 5.4 LRU 管理

`SkillMiddleware` 用 `_loaded_skills`（列表，维护 LRU 顺序）+ `_loaded_content`（名字 → 内容）两个结构。每加载一个技能就追加到列表尾；一旦超过 `max_loaded_skills`（默认 5），就从列表头弹出最旧的，同时删掉它的内容缓存。

```text
_reloaded_skills = ["skill-a", "skill-b"]      # a 更早加载
load_skill("skill-c")
  → ["skill-a", "skill-b", "skill-c"]
再 load_skill("skill-a")                        # 命中已加载，先移除再追加（刷新 LRU）
  → ["skill-b", "skill-c", "skill-a"]
```

> **【C10】加载的第二重效果**：技能被加载后，`allowed_tools` 会对 LLM 可见的工具做一次白名单过滤（取所有已加载技能 `allowed_tools` 的并集）；全部卸载或没有任何技能声明 `allowed_tools` 时自动恢复全量工具，详见 [7.6](#76-middlewarepy--注入与生命周期)。

---

## 六、核心数据模型

### 6.1 `SkillMetadata`（`models.py`）

对应 `SKILL.md` 头部的 YAML frontmatter，`@dataclass`。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必填 | 技能唯一标识，必须与目录名一致 |
| `description` | `str` | 必填 | **最关键字段**，决定模型何时触发这个技能 |
| `version` | `str` | `"1.0.0"` | 版本号 |
| `author` | `Optional[str]` | `None` | 作者 |
| `license` | `Optional[str]` | `None` | 许可证 |
| `compatibility` | `Optional[str]` | `None` | 兼容性说明 |
| `allowed_tools` | `List[str]` | `[]` | 预授权可用工具列表（**【C10】加载技能后据此过滤 LLM 可见工具**） |
| `tags` | `List[str]` | `[]` | 分类标签 |
| `requires` | `List[str]` | `[]` | **【C12】**依赖的其他技能，支持版本约束如 `other-skill>=1.2.0` |
| `requires_tools` | `List[str]` | `[]` | **【C12】**运行时必须存在的工具名 |
| `min_fast_agents_version` | `Optional[str]` | `None` | **【C12】**最低框架版本 |
| `max_fast_agents_version` | `Optional[str]` | `None` | **【C12】**最高兼容框架版本 |
| `metadata` | `Dict[str, Any]` | `{}` | 自定义元数据 |
| `skill_path` | `Optional[Path]` | `None` | **运行时字段**，技能目录路径 |
| `skill_type` | `str` | `"file"` | **运行时字段**，`file` / `code` |
| `source` | `str` | `"project"` | **运行时字段**，`project` / `user` / `builtin` / `custom` / `code` |

> **【C18方案B对齐主流规范说明】**
> 1. `scripts_schema`（脚本参数Schema）已从SkillMetadata移除，外置到`scripts/<脚本名>.schema.json`文件，与同名`.py`脚本同目录，Level 2加载技能后扫描scripts目录读取，仅在需要运行脚本时校验参数，不随Level 1索引发送；
> 2. `allow_network`/`allow_write_paths`/`env_whitelist`（沙箱三字段）已从SkillMetadata移除，改为SkillLoader/SkillRegistry/SkillMiddleware初始化时的全局配置参数，宿主统一管理沙箱策略，每个技能默认追加自身`tmp/`目录到可写白名单；
> 3. 一级元数据仅保留核心字段，对齐Anthropic Agent Skills规范，单技能常驻索引token约100，避免SKILL.md开头过长。

`validate()` 返回错误字符串列表（空列表 = 通过）：

- `name` 非空，长度 ≤ 64；
- `name` 必须匹配正则 `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`——即只能小写字母 / 数字 / 连字符，不能以连字符开头结尾；
- `name` 不能含连续连字符 `--`（注意：这条和上一条正则其实重叠，正则已经拦住了）；
- `description` 非空，长度 ≤ 1024；
- **【C12】**`requires` 每项须为 `skill-name` 或 `skill-name>=1.0.0` 形式（支持 `>=` / `<=` / `==` / `>` / `<`）；
- **【C12】**`min_fast_agents_version` / `max_fast_agents_version` 须为 `x.y.z` 形式；
- `version` 须为语义化版本号（如 `1.0.0`，可带 `-suffix`）。

### 6.2 `SkillLoadResult`（`models.py`）

一次加载的结果，`@dataclass`。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `success` | `bool` | 必填 | 是否成功 |
| `skill_name` | `str` | 必填 | 技能名 |
| `metadata` | `Optional[SkillMetadata]` | `None` | 元数据 |
| `content` | `Optional[str]` | `None` | `SKILL.md` 原始全文（含 frontmatter） |
| `instructions` | `Optional[str]` | `None` | 去掉 frontmatter 的正文 |
| `scripts` | `List[Path]` | `[]` | `scripts/` 下的文件（**排除** `_` 开头的和 `.schema.json` schema文件） |
| `references` | `List[Path]` | `[]` | `references/` 下的文件（只保留 `.md/.txt/.json/.yaml/.yml`） |
| `assets` | `List[Tuple[str, int, bool]]` | `[]` | 【C15】`assets/` 下所有文件清单（递归，跳过 `_` 开头）：每个元素是 `(相对路径, 大小字节, 是否二进制)` |
| `scripts_schemas` | `Dict[str, ScriptSchema]` | `{}` | 【C18方案B】从`scripts/&lt;脚本名&gt;.schema.json`解析得到的参数Schema，脚本执行前自动校验参数，full_prompt会展示每个脚本的参数说明 |
| `errors` | `List[str]` | `[]` | 错误信息 |
| `warnings` | `List[str]` | `[]` | 警告信息 |

`full_prompt` 是 `@property`，按 [5.2](#52-level-2正文按需) 的格式拼装；`success=False` 或没有 `instructions` 时返回空字符串。

---

## 七、逐个文件讲解

### 7.1 `__init__.py`

对外导出的全部符号：

```python
from .base import BaseSkill, skill
from .errors import (
    SkillError, SkillNotFoundError, SkillLoadError, SkillValidationError,
    SkillDependencyError, SkillCircularDependencyError,
    ScriptExecutionError, ReferenceNotFoundError,
    ScriptParamValidationError, ScriptSandboxViolationError,
    AssetNotFoundError, SkillConflictError,
)
from .loader import SkillLoader
from .middleware import SkillMiddleware
from .models import SkillMetadata, SkillLoadResult, ScriptParam, ScriptSchema
from .registry import SkillRegistry

__all__ = [
    "BaseSkill", "skill", "SkillRegistry",
    "SkillMetadata", "SkillLoadResult", "SkillLoader", "SkillMiddleware",
    "ScriptParam", "ScriptSchema",
    "SkillError", "SkillNotFoundError", "SkillLoadError", "SkillValidationError",
    "SkillDependencyError", "SkillCircularDependencyError",
    "ScriptExecutionError", "ReferenceNotFoundError",
    "ScriptParamValidationError", "ScriptSandboxViolationError",
    "AssetNotFoundError", "SkillConflictError",
]
```

> 注意两点：① `skills/__init__.py` **目前不导出 `__version__`**（子包不单独维护版本，`fast_agents/version.py` 的 `__version__` 是框架顶层版本）；② 11 个异常类统一在 `errors.py` 定义、从这里导出，见 [7.9](#79-errorspy--统一异常体系)。

### 7.2 `models.py`

四个 dataclass：核心模型 `SkillMetadata` / `SkillLoadResult`（字段见 [第六章](#六核心数据模型)），以及 **【C6-2】**新增的 `ScriptParam` / `ScriptSchema`——前者描述一个脚本参数（`name` / `type` / `required` / `default` / `enum` / `min` / `max` / `max_length`，带 `validate_value()`），后者描述一个脚本（`script_name` / `description` / `params` / `timeout`，带 `validate_args()`）。

### 7.3 `base.py` —— 代码型技能

**`BaseSkill` 类属性**（子类覆盖）：

```python
name: str = ""
description: str = ""
version: str = "1.0.0"
author: Optional[str] = None
license: Optional[str] = None
compatibility: Optional[str] = None
allowed_tools: List[str] = []
tags: List[str] = []
extra_metadata: Dict[str, Any] = {}
```

> 注意坑：`BaseSkill` 上的自定义元数据字段叫 **`extra_metadata`**，但 `SkillMetadata` 上的字段叫 **`metadata`**。`__init__` 里做的是 `metadata=self.extra_metadata`。写子类时用错名字，数据会静默丢失。

**`__init__` 做的事**：校验 `name` / `description` 非空 → 构造 `SkillMetadata`（`skill_type="code"`，`source="code"`）→ 跑 `validate()`，有错就 `raise ValueError`。

**方法**：

| 方法 | 说明 |
|------|------|
| `metadata`（property） | 取构造好的 `SkillMetadata` |
| `get_instructions()` | 返回要注入的执行指令。默认取 `self.execute` 的 docstring（经 `inspect.cleandoc` 清理），子类可覆盖 |
| `execute(**kwargs)` | **核心逻辑，子类必须实现**，基类直接 `raise NotImplementedError` |
| `aexecute(**kwargs)` | 异步版，默认直接 `return self.execute(**kwargs)` |
| `load()` | 返回 `SkillLoadResult(success=True, ..., instructions=self.get_instructions())` |
| `run_script()` / `read_reference()` | 代码型技能不支持，`raise NotImplementedError` |

**`@skill` 装饰器**：

```python
@skill(
    name="calculate",
    description="执行安全的数学计算……当用户需要计算数学表达式时使用。",
    tags=["math", "calculation"],
    version="1.0.0",
)
def calculate(expression: str) -> str:
    """这里是会自动变成 instructions 的 docstring。"""
    ...
```

装饰器内部动态创建一个 `DecoratedSkill(BaseSkill)` 子类，把装饰器参数填进类属性，并且**把被装饰函数的 docstring 作为 `get_instructions()` 的返回值**。装饰后的对象上会挂一个 `skill_class` 属性——这正是 `SkillRegistry.register()` 用来识别装饰器产物的标记（`hasattr(skill, 'skill_class')`）。

### 7.4 `loader.py` —— 文件型技能

**类常量**：

```python
SKILL_FILENAMES = ["SKILL.md", "skill.md", "Skills.md", "skills.md"]
SCRIPTS_DIR = "scripts"
REFERENCES_DIR = "references"
ASSETS_DIR = "assets"
```

**构造与默认路径**：`SkillLoader(extra_paths=None, default_allow_network=False, default_allow_write_paths=None, default_env_whitelist=None, default_script_timeout=30)`。默认加三条（优先级从高到低）：

1. `Path.cwd() / ".skills"`，`source="project"`（**【C7】已统一为带点的 `.skills`**）；
2. 框架内置：包根 `fast_agents/.skills`，`source="builtin"`；
3. `~/.fast_agents/skills`，`source="user"`。

`extra_paths` 里的路径按 `source="custom"` 追加。`add_skill_path()` 会做 `resolve()`，并且**只有目录存在才会真的加进去**（不存在就静默跳过）。

【C18方案B】沙箱全局配置参数说明：
- `default_allow_network`：全局默认是否允许脚本访问网络，默认`False`（断网，通过设置无效代理拦截）
- `default_allow_write_paths`：全局可写路径白名单，每个技能加载时自动追加自身`tmp/`目录到白名单
- `default_env_whitelist`：全局允许透传给脚本的环境变量白名单，默认透传PATH/SystemRoot等运行必需变量
- `default_script_timeout`：脚本默认超时时间（秒），默认30秒，可被schema.json中声明的timeout覆盖

**`scan_skills()`**：清空索引 → 遍历所有 `skill_paths` → 对每个子目录找 `SKILL.md` → 解析元数据 → 校验 → 如果 `metadata.name != skill_dir.name` 就打印警告并**强制改成目录名**。返回 `Dict[str, SkillMetadata]`。

**`_split_frontmatter()`**：正则 `^---\s*\n(.*?)\n---\s*\n?`（`re.DOTALL`），拆出 YAML 块和正文。拆不到就返回 `(None, content)`——此时元数据走默认值（name 取目录名，description 取 `f"{目录名}技能"`）。

**`load_skill()`**：索引为空时先 `scan_skills()` → 命中缓存直接返回 → 找不到返回 `success=False` → 读文件、拆 frontmatter、扫 `scripts/`（排除 `_` 开头）、`references/`（限定后缀）和 `assets/`（递归 + 文本/二进制分流，排除 `_` 开头）→ 组装 `SkillLoadResult` 并缓存。【C15 方案A】`assets` 清单**只进入 SkillLoadResult**（Level 2 加载后通过 full_prompt 展示给 LLM），不进入 SkillMetadata 常驻索引，避免随 function schema 常驻 token 膨胀，对齐 Anthropic / Microsoft Agent Skills 规范的三级渐进式披露原则。

**`read_reference()`**：精确匹配失败后做模糊匹配（小写比较 + 子串包含）。

**`run_script()`**：按扩展名挑解释器 —— `.py` 用 `sys.executable`；`.ps1/.bat/.cmd` 统一走 `powershell -ExecutionPolicy Bypass -File`。脚本扩展名先过白名单 `{".py", ".ps1", ".bat", ".cmd"}`，路径穿越直接拒。参数拼成 `--key value`（`True` 只拼 `--key`，`False`/`None` 跳过）。默认 `timeout=30` 秒，`cwd` 设为技能目录，stdout/stderr 合并返回，非零退出码会附 `[退出码: N]`。

**【C6-2】沙箱加固**（加在原有的路径穿越防护 / 扩展名白名单 / 输出截断之上）：

- **参数 Schema 校验**：命中 `scripts_schema` 时按 `ScriptParam` 做类型转换与范围 / 枚举 / 长度校验，不通过返回 `参数校验失败`；未声明 Schema 的参数默认做命令注入字符检查；
- **敏感路径拦截**：参数命中 `SENSITIVE_PATH_PREFIXES`（`C:\Windows\System32` / `SysWOW64` / `Users` 等）且不在 `allow_write_paths` 内时返回 `沙箱拦截`；
- **危险命令扫描**：`.bat/.cmd/.ps1` 内容命中 `DANGEROUS_COMMANDS`（`format` / `reg delete` / `net user` / `Invoke-Expression` 等）直接拦截；
- **环境变量沙箱**：仅透传 `PATH` 等必需变量 + `env_whitelist`；`allow_network=False`（默认）时把 `HTTP_PROXY`/`HTTPS_PROXY` 指向无效地址断网；
- **输出截断**：stdout 超 `MAX_SCRIPT_OUTPUT_CHARS`（2000 字符）、stderr 超其一半时截断并附提示。

索引文本统一由 `registry.get_index_prompt()` 生成（内部调用 `SkillMetadata.to_index_text()`）；原 `SkillLoader.get_skill_index_text()` 已于 commit `27e2d7a` 删除，见 [10.8](#108-坑有两套索引文本已清理)。

### 7.5 `registry.py` —— 统一注册表

**构造**：`SkillRegistry(extra_skill_paths=None, default_allow_network=False, default_allow_write_paths=None, default_env_whitelist=None, default_script_timeout=30, include_default_paths=True)`，内部立刻建一个 `SkillLoader`。后四个沙箱参数（**【C18】**）直接透传给 loader，宿主统一管理沙箱策略；`include_default_paths=False` 可跳过默认扫描路径（测试 / 独立示例用）。

**`register(skill)`** 支持三种入参，靠特征分派：

| 传入 | 判定依据 | 处理 |
|------|---------|------|
| `@skill` 装饰后的函数 | `hasattr(skill, 'skill_class')` | `skill.skill_class()` 实例化 |
| `BaseSkill` 子类 | `isinstance(skill, type) and issubclass(skill, BaseSkill)` | 直接 `skill()` |
| `BaseSkill` 实例 | `isinstance(skill, BaseSkill)` | 直接用 |
| 其他 | — | `raise TypeError` |

注册时若同名已存在，打印警告并覆盖。**【C8】**文件型技能之间按来源优先级（`builtin` < `user` < `project` < `custom` < `code`）覆盖同名者，代码型技能（`register()` / `BaseSkill` / `@skill`）恒为最高优先级。

**核心方法**：

| 方法 | 说明 |
|------|------|
| `register` / `unregister` | 代码型技能的增删 |
| `add_skill_path(path, source)` / `scan()` | 转调 loader |
| `get_all_metadata()` | **文件型 + 代码型合并，代码型覆盖同名文件型** |
| `get_metadata(name)` | 单个查 |
| `load_skill(name)` | 先查代码型，再交 loader |
| `execute_skill` / `aexecute_skill` | 只对代码型有效；文件型返回一句「请按加载的指令执行」 |
| `read_reference` / `run_script` | 只对文件型有效；代码型返回「不支持」 |
| `get_index_prompt()` | Level 1 索引文本（**已剔除被禁用的技能**） |
| `get_all_metadata()` / `get_enabled_metadata()` | 全部 / 仅启用中的技能元数据 |
| `is_skill_disabled(name)` | **【C11】**单个技能是否被禁用 |
| `disable_skill` / `enable_skill` | **【C11】**禁用 / 启用单个技能 |
| `disable_skills_by_tag` / `enable_skills_by_tag` | **【C11】**按 tag 批量禁用 / 启用 |
| `disable_all_skills` / `enable_all_skills` | **【C11】**批量开关全部技能 |
| `get_disabled_skills()` | **【C11】**当前被禁用的技能清单 |
| `check_skill_dependencies(name)` | **【C12】**依赖预检，统一返回 6 键：`{'skill_name','ok','errors','warnings','missing_tools','missing_skills'}` |
| `set_middleware(mw)` / `set_tracer(tracer)` | 绑定中间件（启用禁用通知）与 trace 打点 |
| `get_tools()` | 13 个内置工具（**裸函数**，懒加载缓存） |
| `build_skill_tools()` | 13 个内置工具的 `StructuredTool` 版（**可直接注册进 `ToolRegistry`**） |
| `list_skills()` | `List[Dict]`，来自 `metadata.to_dict()`，含 `disabled` 状态位 |
| `get_skill_source(name)` | **【C8】**查询技能来源标签（`builtin` / `user` / `project` / `custom` / `code`），不存在返回 `None` |
| `list_skill_conflicts()` | **【C8】**列出被高优先级来源覆盖的同名技能条目（调试 / 排查用） |

**13 个内置工具**（`_create_skill_functions()`）返回的是**裸 Python 函数**，不是 `ToolBase` 子类；`build_skill_tools()` 再统一包成 `StructuredTool`：

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_skill` | `(skill_name: str) -> str` | 返回 `SkillLoadResult.full_prompt` |
| `read_skill_reference` | `(skill_name: str, reference_name: str) -> str` | 转 `loader.read_reference` |
| `read_skill_asset` | `(skill_name: str, asset_name: str) -> str` | **【C15】**转 `loader.read_asset`（文本返回内容，二进制仅返回元信息） |
| `run_skill_script` | `(skill_name: str, script_name: str, args: Optional[Dict] = None) -> str` | 转 `loader.run_script` |
| `list_loaded_skills` | `() -> str` | **【P0】**列出当前已加载技能（绑 `ToolRegistry` 后替换为真实实现） |
| `disable_skill` / `enable_skill` | `(skill_name: str) -> str` | **【C11】**禁用 / 启用单个技能 |
| `disable_skills_by_tag` / `enable_skills_by_tag` | `(tag: str) -> str` | **【C11】**按 tag 批量开关 |
| `check_skill_dependencies` | `(skill_name: str) -> str` | **【C12】**加载前预检依赖 |
| `enable_skill_hot_reload` | `(interval: float = 2.0) -> str` | **【C14】**开启热重载后台轮询 |
| `disable_skill_hot_reload` | `() -> str` | **【C14】**关闭热重载 |
| `reload_skills_now` | `() -> str` | **【C14】**立即手动重扫一次 |

> 这些函数的 **docstring 就是给模型看的工具描述**，靠 `StructuredTool.from_function` 自动抽取成 schema。所以改这些函数时，docstring 不能乱写。

### 7.6 `middleware.py` —— 注入与生命周期

构造参数：

```python
SkillMiddleware(
    registry: Optional[SkillRegistry] = None,   # None 则自建一个
    auto_inject_index: bool = True,             # 是否自动注入 Level 1 索引
    unload_after_turn: bool = False,            # 是否每轮结束就卸载已加载技能
    max_loaded_skills: int = 5,                 # LRU 上限
    tracer=None,                                # 【P0】TraceLogger，技能操作打点
    strict_tool_filter: bool = False,           # 【C10】True=调用越权工具直接拦截
)
```

内部状态：`_loaded_skills`（LRU 列表）、`_loaded_content`（名字 → 内容）、`_call_stats`（调用计数）。

| 方法 | 何时调 | 干什么 |
|------|--------|--------|
| `get_system_prompt_suffix()` | 被 `before_agent_run` 调 | 拼 Level 1 索引 + Level 2 已加载详情 |
| `before_agent_run(system_prompt)` | Agent 构造 / 刷新时 | 把 suffix 追加到 system prompt 末尾 |
| `after_tool_call(tool_name, tool_args, tool_result)` | **每次工具调用后** | 若工具是 `load_skill` 且结果不是「加载技能失败」，就记进 LRU；若是 `run_skill_script` 就累加计数 |
| `after_agent_turn()` | 每轮对话结束 | 若 `unload_after_turn=True`，清空已加载技能 |
| `_mark_skill_loaded(name, content)` | 内部 | 维护 LRU，超限弹出最旧的 |
| `unload_skill(name)` / `clear_loaded_skills()` | 手动 | 卸载单个 / 全部 |
| `get_loaded_skills()` / `get_stats()` / `get_tools()` | — | 查看状态 / 取 13 个内置工具 |
| `bind_tool_registry(tr)` | `ReActAgent._init_skills` 时 | **【C10】**绑定工具表，算出并应用 `allowed_tools` 白名单 |
| `get_filtered_tools()` / `get_filtered_tools_description()` | 取工具表 / 刷提示词时 | **【C10】**返回白名单过滤后的工具列表 / 工具描述文本 |
| `get_filtered_openai_tools()` / `get_filtered_anthropic_tools()` | 组装 LLM 请求时 | **【C10】**返回过滤后的 OpenAI / Anthropic tools schema |
| `is_tool_allowed(name)` | — | **【C10】**该工具是否在白名单内 |
| `enable_hot_reload(interval)` / `disable_hot_reload()` / `reload_skills_now()` | 手动 | **【C14】**开启 / 关闭 / 立即触发技能热重载 |

> `after_tool_call` 里的判定字符串是硬编码的中文 `"加载技能失败"`，与 `load_skill` 工具返回的错误前缀**必须保持一致**，改一处要改两处。

### 7.7 `agent_integration_example.py` —— 5 个接入点

这个文件是给「已有自己的 Agent 类」的人看的，用伪代码标出 5 个接入位置：

| 接入点 | 位置 | 做什么 |
|--------|------|--------|
| 1 | `__init__` | 建 `SkillRegistry()` + `SkillMiddleware(...)`，可选 `register()` 代码型技能 |
| 2 | `__init__` | `skill_middleware.get_tools()` 拿到工具（**新代码建议改用 `registry.build_skill_tools()`**，它直接返回 `StructuredTool`，无需手动包装） |
| 3 | 构建 messages 时 | `skill_middleware.before_agent_run(system_prompt)` |
| 4 | 工具调用后 | `skill_middleware.after_tool_call(name, args, result)` |
| 5 | 每轮结束 | `skill_middleware.after_agent_turn()` |

文件末尾还附了一段**推荐的系统提示词「技能使用规范」**，那 5 条原则和 `prompt.py` 里 `DEFAULT_REACT_PROMPT` 的内容是同一套。（示例文件里是 5 条，`prompt.py` 里精简成 4 条。）

> 这个文件里的 `YourExistingAgent` 是**示意用的假类**，且它 `from code_skill_example import ...` 用的是**顶层导入**（`code_skill_example` 而不是 `fast_agents.skills.code_skill_example`），只在把示例文件单独拿出来跑时成立，不要照抄进正式代码。

### 7.8 `code_skill_example.py` —— 两种代码型写法

给了一个可直接跑的例子：

- **`@skill` 版**：`calculate(expression)`，先做危险关键字黑名单校验（`import`/`os`/`sys`/`__`/`exec`/`eval`/`open`/`file`/`compile`），再把 `^` 替换成 `**`，最后 `eval(expr, {"__builtins__": {}}, safe_dict)`。
- **`BaseSkill` 版**：`DataAnalysisSkill`，演示覆盖 `get_instructions()`。

### 7.9 `errors.py` —— 统一异常体系

**【P0】新增**，把原先散落的字符串错误前缀收敛成异常类：根类 `SkillError(Exception)`，下设 11 个子类。

| 异常类 | 触发场景 |
|--------|---------|
| `SkillNotFoundError` | 技能不存在 |
| `SkillLoadError` | 加载 / 解析失败 |
| `SkillValidationError` | 元数据校验不通过 |
| `SkillDependencyError` | 依赖的技能 / 工具缺失或版本不满足（C12） |
| `SkillCircularDependencyError` | 技能依赖成环（C12） |
| `ScriptExecutionError` | 脚本执行失败 / 超时 |
| `ReferenceNotFoundError` | `read_reference` 找不到文档 |
| `AssetNotFoundError` | **【C15】**`read_asset` 找不到静态资产 |
| `SkillConflictError` | **【C8】**同名技能冲突（按来源优先级覆盖时告警） |
| `ScriptParamValidationError` | 脚本参数校验失败（C6-2） |
| `ScriptSandboxViolationError` | 触发沙箱拦截（C6-2） |
| `SkillError` | 其余技能相关错误的兜底基类 |

> 对外仍以「返回错误字符串」为主（工具返回值不变），异常体系服务于内部抛出与上层捕获。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

```powershell
# 假定仓库根目录是 D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents
cd D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents

# 用仓库自带的三个 bat 初始化（uv 路线）
.\第1步uv_init.bat
.\第2步uv_venv.bat
.\第3步uv_add.bat
```

> **必须确认装了 `pyyaml`**：`skills/loader.py` 顶部 `import yaml`，但 `requirements.txt` 与 `pyproject.toml` 里**都没有列出它**。如果启动时报 `ModuleNotFoundError: No module named 'yaml'`，执行：
>
> ```powershell
> uv add pyyaml
> # 或者
> pip install pyyaml
> ```

### 8.2 写第一个技能

在项目根目录建 `.skills/hello-world/`（`ReActAgent` 默认扫 `.skills`）：

```text
.skills/
└── hello-world/
    ├── SKILL.md
    ├── references/
    │   └── style.md
    ├── scripts/
    │   └── greet.py
    └── assets/                 # 【C15】可选：静态资源（模板/样例/图片等）
        └── templates/
            └── tpl.json
```

`.skills/hello-world/SKILL.md`：

```markdown
---
name: hello-world
description: 生成带问候语的欢迎文案。当用户要求"生成欢迎语""写个问候""做个开场白"时使用。
version: "1.0.0"
tags: [text, greeting]
---

# 欢迎文案生成

## 适用场景
用户需要一段欢迎语 / 开场白 / 问候文案时使用。

## 执行流程
1. 询问或推断使用场景（会议开场、邮件开头、社群欢迎等）。
2. 按场景选择语气：正式场合用「尊敬的」，社群场景用轻松口吻。
3. 生成 2 个备选，标注各自适合的场景。
4. 参考 `references/style.md` 里的措辞偏好。

## 输出格式
- 每个备选一行，前面标 `【备选 1】` / `【备选 2】`。
- 末尾用一句话说明推荐哪一个。

## 禁止事项
- 不要输出超过 100 字的文案。
```

`.skills/hello-world/scripts/greet.py`：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="朋友")
args = parser.parse_args()
print(f"你好，{args.name}！欢迎使用技能系统。")
```

> 脚本参数按 `--key value` 拼，所以必须用 `argparse` 风格的命令行参数解析。

**【C18】脚本的参数契约（类型 / 必填 / 枚举 / 范围）不再写在 `SKILL.md` frontmatter，而是放在与脚本同目录、同名的 `scripts/greet.schema.json` 里**。加载技能时自动扫描解析，`run_script()` 执行前按它做参数校验与类型转换；不写该文件则跳过前置校验（脚本自身的 `argparse` 仍会兜住）：

`.skills/hello-world/scripts/greet.schema.json`：

```json
{
  "name": "greet.py",
  "description": "生成欢迎语",
  "timeout": 10,
  "params": [
    {"name": "name", "type": "str", "required": true, "description": "称呼"},
    {"name": "formal", "type": "bool", "default": false, "description": "是否正式语气"}
  ]
}
```

> schema 内的 `params` 支持 `type`（`str` / `int` / `float` / `bool` / `enum`）、`required`、`default`、`enum`、`min` / `max`、`max_length`；解析结果存于 `SkillLoadResult.scripts_schemas`。

### 8.3 让它跑起来

```powershell
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from fast_agents.skills import SkillRegistry

r = SkillRegistry()
r.add_skill_path('.skills', source='project')
r.scan()
print('发现技能:', [s['name'] for s in r.list_skills()])
res = r.load_skill('hello-world')
print('加载成功:', res.success)
print('脚本:', [p.name for p in res.scripts])
print('参考:', [p.name for p in res.references])
print(r.run_script('hello-world', 'greet.py', {'name': '小明'}))
"
```

预期输出里能看到 `hello-world`、`['greet.py']`、`['style.md']`，最后一行是 `你好，小明！欢迎使用技能系统。`。

> `add_skill_path` 传相对路径时内部会 `resolve()`，相对的是**当前工作目录**，所以要在仓库根目录下执行。

### 8.4 接到 ReActAgent 上

```python
from fast_agents.skills import SkillRegistry
from fast_agents.tools.tool_registry import ToolRegistry
from fast_agents.agents.react_agent.react_agent import ReActAgent

registry = SkillRegistry()
registry.add_skill_path(".skills", source="project")
registry.scan()

agent = ReActAgent(
    name="助手",
    llm=your_llm,
    tool_registry=ToolRegistry(),
    skill_registry=registry,   # 不传的话，Agent 会自动新建并扫描 .skills/
    enable_skills=True,
)
print(agent.run("帮我写一段社群欢迎语"))
```

`ReActAgent` 在 `enable_skills=True` 时会自动：扫描技能目录 → 建 `SkillMiddleware` → 调 `registry.build_skill_tools()` 拿到 13 个 `StructuredTool` 注册进 `tool_registry` → 绑定工具表算 `allowed_tools` 白名单 → 在 system prompt 里注入索引。发现技能时会打印 `🧩 技能系统已启用，发现 N 个可用技能: ...`。

### 8.5 跑测试

```powershell
# 测试目录已改名为 tests/，按脚本直接执行，不要用 -m
# 技能测试已按组件拆为四个入口（合计 84 项）
python tests\skills\test_skills.py            # 基类与模型层 11 项
python tests\skills\test_skills_loader.py     # 加载器 18 项
python tests\skills\test_skills_middleware.py # 中间件 13 项
python tests\skills\test_skills_registry.py   # 注册器 42 项
python tests\agents\test_react_agent.py
```

---

## 九、API 速查

### 9.1 `SkillRegistry`

```python
from fast_agents.skills import SkillRegistry

registry = SkillRegistry(extra_skill_paths=None)   # 沙箱策略可传 default_allow_network / default_allow_write_paths / default_env_whitelist（C18）；include_default_paths=False 跳过默认扫描路径

# 注册 / 注销代码型技能
registry.register(MySkill())        # 实例
registry.register(MySkill)          # 类
registry.register(my_skill_func)    # @skill 装饰后的函数
registry.unregister("my-skill")     # -> bool

# 路径与扫描
registry.add_skill_path(".skills", source="project")
registry.scan()

# 查询
registry.get_all_metadata()                 # Dict[str, SkillMetadata]
registry.get_metadata("demo-skill")         # SkillMetadata | None
registry.list_skills()                      # List[Dict]

# 加载与执行
result = registry.load_skill("demo-skill")  # SkillLoadResult
registry.execute_skill("my-code-skill", **kwargs)          # 仅代码型
await registry.aexecute_skill("my-code-skill", **kwargs)   # 仅代码型
registry.read_reference("demo-skill", "notes.md")          # 仅文件型
registry.run_script("demo-skill", "hello.py", {"k": "v"})  # 仅文件型

# 启用 / 禁用（C11）
registry.disable_skill("demo-skill")
registry.enable_skill("demo-skill")
registry.disable_skills_by_tag("experimental")
registry.get_disabled_skills()                # Dict[str, Any]
registry.list_skills()                        # 含 disabled 状态位

# 依赖预检（C12）
registry.check_skill_dependencies("demo-skill")   # -> {"skill_name": str, "ok": bool, "errors": [...], "warnings": [...], "missing_tools": [...], "missing_skills": [...]}

# 同名冲突（C8）
registry.get_skill_source("demo-skill")           # -> "project" | "builtin" | "custom" | "code" | None
registry.list_skill_conflicts()                   # -> List[Dict]，被覆盖的低优先级条目

# 给 Agent 用
registry.get_index_prompt()   # Level 1 文本（已剔除被禁用技能）
registry.get_tools()          # 13 个裸函数
registry.build_skill_tools()  # 13 个 StructuredTool（可直接注册）
```

### 9.2 `SkillLoader`

```python
from fast_agents.skills import SkillLoader

loader = SkillLoader(extra_paths=None)
loader.add_skill_path(Path(".skills"), source="project")
loader.scan_skills()                       # Dict[str, SkillMetadata]
loader.load_skill("demo-skill")            # SkillLoadResult
loader.read_reference("demo-skill", "notes.md")
loader.run_script("demo-skill", "hello.py", {"name": "x"}, timeout=30)
loader.list_skills()                       # List[Dict]

# 热重载（C14）
loader.detect_changes(old_mtime_map)       # 增量变更检测
loader.reload_skill("demo-skill")          # 单个技能重载
loader.rescan_incremental()                # 增量重扫（新增 / 修改 / 删除）
```

### 9.3 `SkillMiddleware`

```python
from fast_agents.skills import SkillMiddleware

mw = SkillMiddleware(
    registry=registry,
    auto_inject_index=True,
    unload_after_turn=False,
    max_loaded_skills=5,
    tracer=None,
    strict_tool_filter=False,
)

prompt = mw.before_agent_run(system_prompt)                       # 注入
result = mw.after_tool_call(name, args, result)                   # 捕获 load_skill
mw.after_agent_turn()                                             # 可选卸载
mw.get_loaded_skills()                                            # List[str]
mw.get_stats()                                                    # Dict[str, int]
mw.get_tools()                                                    # 13 件套（裸函数）
mw.unload_skill("demo-skill")                                     # -> bool
mw.clear_loaded_skills()

# 工具白名单（C10）
mw.bind_tool_registry(tool_registry)     # 绑定后自动算白名单
mw.get_filtered_tools_description()      # 过滤后的工具描述文本
mw.is_tool_allowed("calculator")         # -> bool

# 热重载（C14）
mw.enable_hot_reload(interval=2.0)
mw.disable_hot_reload()
mw.reload_skills_now()
```

### 9.4 `BaseSkill` / `@skill`

```python
from fast_agents.skills import BaseSkill, skill


class MySkill(BaseSkill):
    name = "my-skill"                  # 必须 kebab-case
    description = "做什么 + 什么时候用"
    version = "1.0.0"
    tags = ["demo"]
    extra_metadata = {"owner": "me"}   # 注意字段名是 extra_metadata

    def execute(self, **kwargs) -> str:
        return "结果"

    def get_instructions(self) -> str:
        return "第一步……第二步……"


@skill(name="my-func-skill", description="……", tags=["demo"])
def my_func(x: str) -> str:
    """会被当成 instructions 的 docstring。"""
    return f"收到 {x}"
```

### 9.5 模型对象

```python
from fast_agents.skills import SkillMetadata, SkillLoadResult

meta.validate()        # -> List[str]，空列表表示通过
meta.to_index_text()   # -> "- **name** [code]: description"
meta.to_dict()         # -> Dict

result.full_prompt     # -> str（拼装好的 Level 2 文本）
```

---

## 十、设计决策与常见坑

### 10.1 为什么用「文件夹 + SKILL.md」而不是「纯 Python 类」

因为**写流程的人不一定是写代码的人**。产品、运营、领域专家可以直接改 `SKILL.md`，不用碰 Python。同时文档能进 Git、能 review、能 diff，比把流程写死在代码字符串里好维护得多。

代价是要处理 YAML 解析、编码、路径这些琐事——所以 `loader.py` 是模块里最长的一个文件。

### 10.2 为什么 Level 1 索引和 Level 2 正文要分开

Level 1 是**每个技能都要常驻**的成本，所以它必须极短（只有名字 + 描述）。Level 2 只在真正用到时才进来。如果把正文也常驻，30 个技能就把上下文吃干净了。

**推论**：`description` 是整个技能里最重要的字段——它是模型**唯一的**选择依据。写得含糊，技能等于不存在。

好的写法：

```yaml
description: 生成规范的 Git 提交信息并执行提交。当用户说"提交代码""git commit""生成commit信息"时使用，遵循 Conventional Commits 规范。
```

差的写法：

```yaml
description: 帮助处理 git 相关的事情
```

### 10.3 坑：默认扫描路径有两套（已修复，C7）

历史问题是 `SkillLoader._add_default_paths()` 扫 `cwd/skills`（无点），而 `ReActAgent` 扫 `.skills`（有点）。

**现已统一**：`loader._add_default_paths()` 按优先级加三条 —— `cwd/.skills`（`project`）→ 包根 `fast_agents/.skills`（`builtin`）→ `~/.fast_agents/skills`（`user`）；`ReActAgent._PROJECT_SKILLS_DIRS` 仍是 `fast_agents/.skills` + 项目根 `.skills`，两者现在指向同一套带点的 `.skills`。

### 10.4 坑：`get_tools()` 返回的是裸函数

`registry.get_tools()` 返回 13 个**普通函数**，不是 `ToolBase` 实例；直接塞进 `tool_registry`（需要 `ToolBase`）会失败。

**现在有两条路**：① 用 `registry.build_skill_tools()`，它把同样 13 个函数统一包成 `StructuredTool`，可直接 `register_tool()`（`ReActAgent._init_skills()` 走的正是这条）；② 自己用 `StructuredTool.from_function(fn, name=fn.__name__)` 包一层。

`agent_integration_example.py` 里 `self.tools.extend(skill_tools)` 这种写法已过时——它假设工具列表接受裸函数，只在自建 Agent 时才成立。

### 10.5 坑：`name` 必须和目录名一致

`scan_skills()` 里发现不一致时会打印警告并**强制把 `metadata.name` 改成目录名**。也就是说 frontmatter 里写的 `name` 会**被静默覆盖**——如果你靠 `name` 做索引，务必让两者一致。

### 10.6 坑：二进制资产不要尝试直接读正文

**（已在 v1.7.0/C15 落地，以下说明更新为行为约束，不再是未实现）**

`load_skill()` 会递归扫描 `assets/` 目录，并在 `full_prompt` 里把"文本资源"和"二进制资源"分两段列出；`read_skill_asset` 对二进制资源（图片/字体/可执行/非 UTF-8 编码的文本等）**不会返回正文**，只返回元信息（类型/大小/绝对路径）。如果你期望 LLM 直接读图片内容，会拿到的是提示"该文件为二进制格式，请通过脚本处理"。

使用约束：
- 文本类资源（`.md/.txt/.json/.yaml/.yml/.csv/.py` 等）单次读取上限 **200KB**，超过也只返回元信息，需要局部内容请写脚本读取。
- 二进制资源必须通过 `run_skill_script` 交给脚本处理（读字节、转格式、OCR、裁剪等），不要指望模型直接解析。
- 以下划线开头的文件（如 `_cache.dat`、`templates/_partial.html`）视为内部文件，不会出现在 `assets` 清单里，也不会被 `read_skill_asset` 读到。

### 10.7 坑：`pyyaml` 没写进依赖（已修复）

历史问题：`loader.py` 顶部 `import yaml`，但依赖清单里没有 `pyyaml`，全新环境一 import `fast_agents.skills`（`registry` → `loader` → `yaml`）就 `ModuleNotFoundError: No module named 'yaml'`。

**现已写入**：`pyproject.toml` 的 `dependencies` 含 `pyyaml>=6.0.3`，全新环境 `uv sync` 后可直接 import `fast_agents.skills`。

### 10.8 坑：有两套索引文本（已清理）

历史问题：`SkillRegistry.get_index_prompt()`（标题 `## 可用技能（Skills）`，Agent 实际在用）与 `SkillLoader.get_skill_index_text()`（标题 `## 可用技能列表`，无人调用）两套索引文本并存。

**现已删除**：commit `27e2d7a` 移除了 `get_skill_index_text()`，索引文本统一由 `get_index_prompt()` 生成（内部调用 `SkillMetadata.to_index_text()`）；改提示词风格只需改这一处。

### 10.9 跨平台注意事项

`skills/` 模块是为跨平台设计的（`loader.py` 文件头注释也写了「兼容 Windows 和 Unix 路径」），但有些行为在三个平台上表现不同，尤其在 Windows 上。

**① 路径分隔符**

全模块统一用 `pathlib.Path`，没有字符串拼 `\` 或 `/`，这一点是对的。要小心的只有一处：`add_skill_path()` 会 `resolve()`，相对路径是**相对当前工作目录**的。在 Windows 上从别的盘符、别的目录启动脚本时，`.` 的含义会变。**建议一律传绝对路径**，或者在启动时先固定 `os.chdir(仓库根)`。

**② 脚本解释器（`run_script()` 的真实分支）**

```python
suffix = script_path.suffix.lower()
if suffix == ".py":
    cmd = [sys.executable, str(script_path)]
elif suffix in [".ps1", ".bat", ".cmd"]:
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
elif suffix in [".sh"]:
    cmd = ["bash", str(script_path)]
else:
    cmd = [str(script_path)]
```

- **Windows**：`powershell` 这个名字在 Windows PowerShell 5.x 和 PowerShell 7 上都能解析，没问题。但 `-ExecutionPolicy Bypass` 是必需的，否则默认策略会拦下 `.ps1`。
- **`.sh` 在 Windows 上会失败**，除非机器上有 `bash`（Git Bash / WSL）。技能里如果要放 shell 脚本，注意标注只支持 Unix。
- **macOS / Linux**：`.ps1` 基本不可用；`.py` 走 `sys.executable` 没问题。
- **跨平台技能包的建议**：优先用 `.py`，它三个平台都能跑。

**③ 编码**

`load_skill()` 读文件写死了 `encoding="utf-8"`，`run_script()` 用 `encoding="utf-8", errors="replace"`。这在 Windows 上有两个隐患：

- 你如果用记事本 / 某些编辑器把 `SKILL.md` 存成 **GBK**，会直接 `UnicodeDecodeError`（被 `load_skill` 的 try/except 兜住，变成 `加载Skill失败`）。**统一存 UTF-8**。
- 脚本输出里的中文，在 Windows 控制台默认 GBK 代码页下可能出现乱码。`errors="replace"` 会让它变成替换符而不是抛异常，但可读性差。脚本里可以自己加 `sys.stdout.reconfigure(encoding="utf-8")`。

**④ 换行符**

`_split_frontmatter()` 的正则是 `^---\s*\n(.*?)\n---\s*\n?`。`\s` 能吃掉 `\r`，所以 **CRLF 和 LF 都能正常解析**。但如果用 `\r\n` 存文件而某些编辑器插入 BOM，首行的 `---` 前面会有 `\ufeff`，正则就匹配不上了（会退化成「没有 frontmatter」，元数据全部走默认值）。**用 UTF-8 不带 BOM 保存**。

**⑤ 保留字与大小写**

`SKILL_FILENAMES` 里同时列了 `SKILL.md` 和 `skill.md`，但没用大小写不敏感比较。在 Windows 上文件系统本身不区分大小写，所以 `Skill.md` 也能被 `SKILL.md` 匹配到；在 Linux 上就不行了。**统一写成全大写的 `SKILL.md`**，最稳。

**⑥ 路径长度**

Windows 默认单条路径上限 260 字符。技能目录如果嵌得很深（`D:\...\fast_agents\.skills\very-long-skill-name\references\some-deep-file.md`），加上项目路径本身就很长，有触顶风险。技能名尽量短。

---

## 十一、测试与验证

### 11.1 现状

技能模块的测试已按组件拆为四个入口（合计 84 项），基类与模型层留在 `tests/skills/test_skills.py`，扫描/加载、中间件、注册器各自的用例分到同目录三个文件。

| 测试文件 | 结果 | 说明 |
|---------|------|------|
| `tests/skills/test_skills.py` | ✅ 11/11 | 基类与模型层：`BaseSkill` 元数据校验、`@skill` 装饰器、`SkillMetadata` 字段格式（含 C12 `requires`、C18 字段瘦身）、`ScriptParam` / `ScriptSchema` 模型级校验 |
| `tests/skills/test_skills_loader.py` | ✅ 18/18 | 加载器：目录扫描、frontmatter 与 `scripts/*.schema.json` 解析及容错（C18）、脚本/参考文档/资产路径穿越防护与扩展名校验、C14 热重载的变更检测（mtime 快照 / `detect_changes` / `reload_skill` / `rescan_incremental`）、C17 索引缓存与 1+N 扫描治理 |
| `tests/skills/test_skills_middleware.py` | ✅ 13/13 | 中间件：索引注入 System Prompt、已加载技能 LRU 缓存与每轮自动卸载、C10 `allowed_tools` 工具白名单、C14 热重载线程启停与手动重载 |
| `tests/skills/test_skills_registry.py` | ✅ 42/42 | 注册器：代码型技能注册/反注册、便捷方法与 13 件工具工厂、C11 启用禁用、C12 依赖预检与版本比较、C6-2 沙箱运行时校验、C15 `read_asset`、C8 同名冲突消解 |
| `tests/agents/test_react_agent.py` | ✅ 全过 | 含技能启用 / 禁用两条用例 |
| `tests/mcp/test_mcp.py` | ✅ 全过 | MCP 专项 |

> 早期 4 项 C14 热重用例的写法问题已随 `registry` 补 `scan_skills()` / `is_skill_loaded()` / `list_skills_metadata()` 便捷方法与 `middleware` 保留线程引用后全部修复。
> 拆分只做了用例搬家：用例体与断言逐字节不变，`_make_skill_dir` / `_make_script_skill` / `_make_asset_skill` / `_make_conflict_skill_dir` 四个辅助函数因跨组件复用，在各文件保留同源副本，以保证每个文件可单独运行。

> 测试目录已从 `test/` 改名为 `tests/`（避开与 Python 标准库 `test` 包同名），按脚本直接执行、不要用 `-m`；`tests/_runner.py` 是零依赖运行器。每个测试文件各自在开头写 `sys.path.insert(0, 父目录)` 来保证能 import 到 `fast_agents`。

### 11.2 手工验证一份技能包

写完技能后，按下面顺序过一遍（Windows PowerShell，在仓库根执行）：

```powershell
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from fast_agents.skills import SkillRegistry

r = SkillRegistry()
r.add_skill_path('.skills', source='project')
r.scan()

# 1. 索引发现了几个
print('技能数:', len(r.list_skills()))

# 2. 元数据校验过了没（name 被改成目录名说明 frontmatter 里写错了）
for s in r.list_skills():
    print(s['name'], '|', s['skill_type'], '|', s['source'])

# 3. Level 1 文本长什么样
print(r.get_index_prompt())

# 4. Level 2 能否加载
res = r.load_skill('你的技能名')
print('success =', res.success, 'errors =', res.errors)
print(res.full_prompt[:200])

# 5. Level 3 参考文档与脚本
print(r.read_reference('你的技能名', 'notes.md')[:100])
print(r.run_script('你的技能名', 'hello.py', {'name': 'x'}))
"
```

### 11.3 检查清单

- [ ] `SKILL.md` 以 UTF-8 无 BOM 保存；
- [ ] frontmatter 的 `name` 与目录名完全一致；
- [ ] `name` 满足 `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`；
- [ ] `description` 同时写了「做什么」和「什么时候用」，且包含触发关键词；
- [ ] `description` 不超过 1024 字符；
- [ ] `scripts/` 里的脚本用 `argparse` 风格 `--key value`；
- [ ] 脚本输出编码在 Windows 控制台下不炸；
- [ ] 技能指令里引用的 `references/` 文件名与实际文件一致。

---

## 十二、下一步演进路线

编号以 `docs/演进路线图.md` 的 C 系列为准（其 §六 模块 C），两处编号保持一致。**P0 优化与 C6-2 / C7 / C8 / C10 / C11 / C12 / C14 / C15 / C17 / C18 十轮增强已全部落地**；下面 12.1 逐条给出完成版本与落点，12.2 只列**仍未做**的部分。

### 12.1 已交付

| 编号 | 任务 | 完成版本 | 落点 |
|------|------|---------|------|
| ✅ **C1** | Skill 数据模型 + `SKILL.md` 解析 | v1.0.0 | `models.py`（`SkillMetadata` / `SkillLoadResult`）+ `loader._parse_skill_metadata`（YAML front matter） |
| ✅ **C2** | `SkillRegistry` 扫描目录发现技能 | v1.0.0 | `registry.py` + `loader.scan_skills()` |
| ✅ **C3** | 三级渐进式加载 | v1.0.0 | `loader.py` 索引 / 正文 / 资源三级（见 [第五章](#五核心机制三级渐进式加载)） |
| ✅ **C4** | 技能工具（基础三件套） | v1.0.0 | `registry._create_skill_functions()` 的 `load_skill` / `read_skill_reference` / `run_skill_script`；现已扩展到 **13 个** |
| ✅ **C5** | 示例技能包 | v1.0.0 | `examples/skills/`（`code_skill_demo.py` / `file_skill_demo.py` / `middleware_demo.py` 等） |
| ✅ **P0** | 模块治理批（trace 打点 / 统一异常 / 工具工厂 / 路径统一 / 脚本安全加固） | v1.1.0 | 新增 `errors.py`；`build_skill_tools()`；`run_script()` 路径穿越与扩展名加固、输出截断、可配超时 |
| ✅ **C7** | 统一默认扫描路径 | v1.1.0 | `loader._add_default_paths()`：`cwd/.skills`（project）→ 包根 `fast_agents/.skills`（builtin）→ `~/.fast_agents/skills`（user） |
| ✅ **C10** | `allowed_tools` 落地生效 | v1.2.0 | `middleware.py` 白名单过滤（`get_filtered_tools()` / `get_filtered_openai_tools()` / `is_tool_allowed()`）+ `react_agent.py` 两处入口 |
| ✅ **C11** | 技能运行时启用 / 禁用 | v1.3.0 | `registry.py` 8 个 API + `middleware.py` 事件回调 + 4 个 LLM 工具 |
| ✅ **C12** | 技能依赖与前置检查 | v1.4.0 / v1.8.1 | `models.py` 4 个字段 + `registry.py` 校验（工具 / 技能 / 框架版本 / 循环依赖）+ `errors.py` 2 个异常 + `check_skill_dependencies`；v1.8.1 起 `check_skill_dependencies` 两分支统一 6 键，`missing_tools`/`missing_skills` 返回结构化清单 |
| ✅ **C6-2** | `scripts/` 脚本沙箱增强 | v1.5.0 | `loader.run_script()` 九层加固（参数 Schema / 注入拦截 / 敏感路径 / 危险命令扫描 / 环境隔离等）+ `models.py`（`ScriptParam` / `ScriptSchema`） |
| ✅ **C14** | 技能热重载 | v1.6.0 | `loader.py` mtime 增量检测（`detect_changes` / `reload_skill` / `rescan_incremental`）+ `middleware.py` 后台线程 + 3 个 LLM 工具，默认关闭 |
| ✅ **C15** | `assets/` 静态资产完整读取 | v1.7.0 | `load_skill()` 递归扫描 `assets/`（文本/二进制分流）+ `read_skill_asset` LLM 工具（路径穿越防护、文本返回内容、二进制/超大文件仅返回元信息）；**方案A：assets 清单只进 `SkillLoadResult`**，不进 `SkillMetadata` 常驻索引，对齐 Anthropic/Microsoft 主流实践；工具总数 12→13 |
| ✅ **C17** | 索引缓存与 1+N 扫描治理 | v1.7.0 | `loader.scan_skills()` mtime 门控命中直接返回 `_skill_index`；`registry` 全链路复用元数据字典消除 N 次重复扫描；缓存命中率高时热路径从 ~30ms 降到 ~0.05ms |
| ✅ **C18** | 一级元数据极简（对齐主流规范） | v1.8.0 | 脚本参数 Schema 从 frontmatter 外移到 `scripts/<脚本名>.schema.json`（与脚本同目录、解析结果入 `SkillLoadResult.scripts_schemas`）；沙箱三字段（`allow_network` / `allow_write_paths` / `env_whitelist`）上移到 `SkillLoader` / `SkillRegistry` / `SkillMiddleware` 初始化全局配置；`SkillMetadata` 回归 `name` / `description` / `version` / `tags` 等核心字段，单技能常驻 token 约 100 |
| ✅ **C8** | 同名技能冲突消解 | v1.9.0 | `loader` 按来源优先级（`builtin` < `user` < `project` < `custom` < `code`）覆盖同名技能并打印警告；新增 `SkillConflictError` + `get_skill_source()` / `list_skill_conflicts()`；多版本共存未做（待 C9 之后） |

### 12.2 待办（按优先级）

| 编号 | 任务 | 优先级 | 依赖 | 一句话说明 |
|------|------|--------|------|-----------|
| **C9** | 技能市场 / 远程安装 | **P2** | C12（已完成） | 从 URL / Git 拉取技能包，按 `name` 校验后安装 |
| **C16** | 技能执行结果缓存 | **P2** | C3 | 相同参数脚本结果缓存，减少重复 IO 与进程启动开销 |
| **C8** | 技能多版本共存 | **P3** | C9（技能市场） | 同名技能冲突消解已随 C8 落地（v1.9.0）；剩「同一技能多版本共存 + 版本约束自动选版本」，**排在 C9 技能市场之后**——单机技能目录下无多来源冲突，无前置需求 |

---

## 十三、参考

**仓库内**

- `fast_agents/skills/`（`__init__.py` / `models.py` / `base.py` / `errors.py` / `loader.py` / `registry.py` / `middleware.py`）
- `fast_agents/skills/agent_integration_example.py`、`fast_agents/skills/code_skill_example.py`
- `fast_agents/agents/react_agent/react_agent.py`、`fast_agents/agents/react_agent/prompt.py`
- `docs/演进路线图.md`（C 系列条目）
- `tests/skills/`（`test_skills.py` 基类与模型层 / `test_skills_loader.py` 加载器 / `test_skills_middleware.py` 中间件 / `test_skills_registry.py` 注册器）、`tests/agents/test_react_agent.py`
- `examples/skills/`（`code_skill_demo.py` / `file_skill_demo.py` / `middleware_demo.py` / `enable_disable_demo.py` / `dependency_demo.py` / `file_skill_dependency_demo.py` / `script_sandbox_demo.py` / `hot_reload_demo.py` 等）

**外部**

- Anthropic 官方技能开源仓库：<https://github.com/anthropics/skills>
- Anthropic Agent Skills 规范（`SKILL.md` + YAML frontmatter + 三级渐进式披露）
- Semantic Kernel `KernelPlugin` 文档：<https://learn.microsoft.com/python/api/semantic-kernel/semantic_kernel.functions.kernelplugin>
- CrewAI Knowledge 文档

---

## 附录：发布前自检清单

改动 `fast_agents/skills/` 之后，逐条打勾：

- [ ] `python -c "from fast_agents.skills import SkillRegistry, SkillLoader, SkillMiddleware, BaseSkill, skill, ScriptParam, ScriptSchema, SkillError, AssetNotFoundError, SkillConflictError"` 无报错；
- [ ] `tests/skills/` 下四个入口全过（`test_skills.py` 11 / `test_skills_loader.py` 18 / `test_skills_middleware.py` 13 / `test_skills_registry.py` 42）；`python tests/agents/test_react_agent.py` 全过（含两条技能用例）；
- [ ] 新增 / 修改的技能包能被 `scan()` 发现，且 `name` 没被强制改写（没打印告警）；
- [ ] `get_index_prompt()` 的输出格式没被意外破坏（标题、`[代码]`/`[文件]` 标签、使用方法段落）；
- [ ] `full_prompt` 的五个段落（`# Skill:` / `## 技能说明` / `## 执行指令` / `## 可用脚本` / `## 参考文档`）顺序与内容正确；
- [ ] 13 个内置工具的 **docstring** 仍然准确（它们是给模型看的工具描述）；
- [ ] `SkillMiddleware.after_tool_call()` 里判定的错误前缀 `"加载技能失败"` 与 `load_skill` 实际返回的前缀一致；
- [ ] `SKILL_FILENAMES` / `SCRIPTS_DIR` / `REFERENCES_DIR` / `ASSETS_DIR` 常量没被改坏；
- [ ] 新增字段后同步更新了 `SkillMetadata.to_dict()`；
- [ ] 本文档的 [第六章](#六核心数据模型)、[第九章](#九api-速查) 已同步更新；
- [ ] 本文档 [一、更新历史](#一更新历史) 加了一行。
