# FastAgents Guardrails（安全护栏）模块编写说明

本文档面向 fast_agents Guardrails 安全护栏系统，参考 **LangChain Guardrails**、**OpenAI Agents SDK Guardrails**、**LlamaIndex Guardrails**、**Anthropic Claude Guardrails**、**Microsoft MCP 安全规范** 等框架的共性设计，给出本模块的完整说明、源码解析与自定义教程。

> 版本：v1.0.10 · 2026-11-24 · 跨模块行为登记：输出护栏与 Agent 取值的时序说清（agents 域 A2 已由路线图 v2.74.0 修掉，`guardrails/` 本体代码零改动）
> 适用版本：`fast_agents.guardrails` v1.0.10
> 配套代码：`fast_agents/guardrails/`（9 个核心文件 + 3 个预设护栏）+ `tests/guardrails/`（11 个文件共 125 项测试全绿：45 主用例 + 12 中间件 + 19 真实 ReActAgent 端到端 + 16 模拟 Agent 链路 + 33 分项）+ `examples/guardrails/guardrails_basic_demo.py`（无API Key可运行示例），均已实测通过。
> 运行环境：Python 3.10+，Windows / macOS / Linux；依赖：零新增依赖，完全复用现有核心抽象。

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：Guardrails 解决什么问题](#二先想清楚guardrails-解决什么问题)
- [三、主流框架 Guardrails 实现对比](#三主流框架-guardrails-实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：三层防护全链路安全](#五核心机制三层防护全链路安全)
- [六、核心数据模型](#六核心数据模型)
- [七、逐个文件讲解（源码导读）](#七逐个文件讲解源码导读)
- [八、快速上手](#八快速上手)
- [九、API 速查](#九api-速查)
- [十、设计决策与常见坑](#十设计决策与常见坑)
- [十一、测试与验证](#十一测试与验证)
- [十二、下一步演进路线](#十二下一步演进路线)
- [十三、参考](#十三参考)
- [附录：发布前自检清单](#附录发布前自检清单)

---

## 一、更新历史

| 版本 | 日期 | 变更内容 | 破坏性 |
|------|------|----------|--------|
| v1.0.10 | 2026-11-24 | **输出护栏与 Agent 取值的时序说清（本轮 `guardrails/` 源码零改动，属跨模块行为登记）**：第三轮架构评审的 **agents 域 A2**（与本文档 v1.0.5 那个 guardrails 域 A2 不同编号空间）已于路线图 v2.74.0 修掉：`ReActAgent.run` / `arun` 旧写法优先拼接流式 `text` 增量，而输出护栏在正文产出之后才跑，导致脱敏（modify）与拦截（reject）两种结果**都只能改写 `end` 事件、非流式返回值依旧拿到原文**（同步 / 异步两条链路同病，本文档旧版 §5.3 所称「Agent 最终结果返回给用户前」实际仅对 `end` 事件成立）。现 `run` / `arun` 改取护栏后正文；**剩下的只是流式侧物理限制**：`text` 分片已发出收不回，需强护栏语义请用 `run` / `arun`，或累加到 `end` 再展示 | 低：护栏本体行为逐字节不变；行为变化在 Agent 侧（`run` 返回值不再含工具轮次过渡语），详 §5.3 与 §10.2 第 12 条 |
| v1.0.9 | 2026-11-24 | **B4 修复（注入护栏误拦正常文本）**：RC1 跨句粘连——`_SENTENCE_SPLIT_RE` 按英文句末终止符 + 后跟空白（`chapters. Instructions`）或中文 `。！？` 切分句子，逐句匹配而非整段匹配，防止带间隙模式把无关子句粘连成注入特征；RC2 词融合——`jailbreak(?:mode|prompt|now)?` 改用非空格压缩视图（`_squeeze_text_nonspace`，仅去分隔符保留空格），防止 `jail  break the habit` 被误融合为 `jailbreak` 而误拦；`test_s4_tool_registry_injection.py` 顺带修复：B3 的 `extra="forbid"` 暴露出该文件两处 `ToolCallInfo(arguments=...)` 字段名拼错（应为 `tool_args=`）。**doanythingnow 保留完整压缩视图**：现有测试 L7 断言 `do anything now` 应拦截（DAN 攻击签名），R1 不变更 | **行为变化（正向安全加固）**：跨句正常叙述文本不再误拦；jailbreak 纯空格分隔误拦消除；各类绕过样本（含换行/加符号/全角）全部仍拦截；test_s4 调用处字段名拼错改为正确值 |
| v1.0.8 | 2026-11-24 | **B1/B3 修复**：`BasicPermissionGuardrail` 构造器的集合参数改用 `is not None` 判断，避免空集合被 `or` 运算符替换为默认值（修复前传 `blocked_tool_names=set()` 想关闭拦截时，默认危险工具名照样生效）；`models.py` 三个安全关键模型（`GuardrailResult` / `ToolCallInfo` / `ToolResultInfo`）的 `extra` 配置从 `"allow"` 改为 `"forbid"`，防止拼错字段名（如 `alowd=True` / `tool_ars={...}`）被静默接受而导致检查逻辑被绕过 | **行为变化（正向安全加固）**：B1 影响传空集合的场景用户（此前被默认值覆盖，现保留空集合）；B3 影响传拼错字段名的代码（此前静默接受，现立即抛 `ValidationError`）。正确用法不受影响 |
| v1.0.7 | 2026-11-24 | **B2 修复**：`BasicPermissionGuardrail` 第 5 步的敏感路径参数扫描改为**逐层展开**。修复前只遍历 `tool_args.items()` 的**顶层 str 值**，路径换个位置就绕开：`{"payload": {"path": "/etc/passwd"}}`、`{"paths": ["/etc/shadow"]}`、`{"a": {"b": [{"c": "/proc/self/environ"}]}}`、路径写在字典键上 `{"files": {"/etc/passwd": "r"}}`、传 `Path` 对象——探针实测 6 种嵌套形态**全部 ALLOWED**；且仓库此前没有任何测试喂过敏感路径参数（只测工具名拦截），故长期存活。现按先序展开 dict（**键与值都扫**）/ list / tuple / set / frozenset / `os.PathLike`；用**显式栈而非递归**（护栏自己抛 RecursionError 会按 fail_open 把业务打成放行或全拦），**不设深度上限**（深度上限等于新递一个「藏到第 N+1 层就过关」的绕过口）；自引用参数按「当前分支已访问容器 id」剪枝，不死循环；`tool_args` 为非 dict（list / str / None / int / set）时不再因 `.items()` 抛 `AttributeError`。**测试口径修正**：v1.0.6 行及其之前所写「116 项 / 28 分项」为漏计——`test_tool_call_guardrail.py`（3 项）从未计入分项、`test_s5_reject_preserves_results.py` 实为 5 项（记成 3），按 `run_all` 执行计数 v1.0.6 真实为 **121 项**；历史行不做追溯篡改，自本行起以执行计数为准 | **行为变化（正向安全修复）**：嵌套参数里的敏感路径此前放行、现在拦截。顶层命中的 `reason` 文案与 `metadata["arg_name"]`（直接是参数名）逐字节不变；`check_sensitive_paths=False` 语义不变（彻底不扫） |
| v1.0.6 | 2026-11-24 | **B5 修复**：`BasicOutputSafetyGuardrail` 的缓和语境豁免改为**只在命中片段所处子句内取证**。修复前 `_has_mitigating_context` 只取「命中片段前后 `mitigation_window`（默认 30）字符」，30 字符足以横跨整句、上一句甚至下一个括号，于是最常见的绕过形态——「免责声明单独成句 + 大段可执行危害步骤」——被一句轻描淡写整体放行（实测：`"请勿模仿：以下是制造毒品的具体步骤\n第一步……"`、`"……具体步骤：1. 备料……（本内容违法）"` 均 ALLOWED）。现在取证范围 = 子句边界 ∩ [start-window, end+window]，子句边界为句末标点/冒号分号/换行/括号，**逗号顿号故意不当边界**（正常科普正是用逗号把警示语挂在同句），`mitigation_window` 保留原参数与默认值、语义改为上限 | **行为变化（正向安全修复）**：跨句/跨括号免责声明不再豁免危害输出；同句内自带警示语的科普、拒答逐字节不变。代价：同子句内无表内缓和词的新闻/司法通报体（「已被依法刑事拘留」）转为误拦，本次有意不扩充词表（扩词同时放大绕过面） |
| v1.0.5 | 2026-11-24 | **A2 修复**：`add_guardrail()` 挂载时补注入 `tool_registry`。修复前广播只写在 `tool_registry` 的 setter 里，仅覆盖「赋值那一瞬间已挂载」的实例；先注入 registry、后 `add_guardrail(BasicPermissionGuardrail(...))` 时，新副本的 `tool_registry` 永远是 None → 读 `metadata["requires_approval"]` 的单一事实源分支从不执行，高危工具静默放行（S4 同一条死角在「先注入、后挂载」方向上的延伸）。现仅当实例自己没带 registry 时才注入（不把用户显式配置的 registry 覆盖掉），`self._tool_registry` 为 None 时不执行也不 import preset 包；setter 的无条件广播语义逐字节不变 | **行为变化（正向修复）**：仅影响「已有 tool_registry 后晚挂载权限护栏」这一种时序；无该时序时行为逐字节一致 |
| v1.0.4 | 2026-11-24 | **A1 修复**：`GuardrailMiddleware(guardrails=[...])` 显式传入的护栏与自动加载的预设**同名**时，不再被预设静默替换。修复前加载顺序为「先挂显式、后加载预设」，`_load_presets` 内部经 `add_guardrail(..., skip_exist_check=True)` 的同名替换逻辑把用户实例 `pop` 掉，又因 `skip_exist_check=True` 连 warning 都不打——用户自定义同名护栏（尤其是覆盖 `prompt_injection` 这类预设名的场景）**完全不生效且无任何线索**；修复后确立「**显式优先**」：同名预设让位（跳过加载）并打一条 info 日志留痕，其余预设照常加载；构造完成后走公开 `add_guardrail()` 的显式注册语义不变（后注册覆盖先注册、仍打 warning） | **行为变化（正向修复）**：仅影响「构造器显式护栏与预设同名」这一种配置，此前被静默丢弃的用户规则恢复生效；无参冲突时行为逐字节一致 |
| v1.0.3 | 2026-11-23 | P1 修复：`@guardrail_creator` 新增 `guardrail_type` 参数（`input` 默认 / `output` / `tool_call`），生成的护栏按类型继承对应基类、真正参与 `GuardrailMiddleware` 阶段分派与 `use_global=True` 扫描（此前「注册成功却永不执行」）；`tool_call` 型沿用 before/after 分发且后置受 `check_result` 约束；装饰 GuardrailBase 子类恢复原样返回（此前被误包装、检查即报配置错误）；非法 `guardrail_type` 创建时即抛 `GuardrailConfigError` | **行为变化（正向修复）**：此前静默不执行的函数式护栏现在会在对应阶段真正执行 |
| v1.0.2 | 2026-11-23 | 四个统计/观测层缺陷修复（均不改放行/拦截控制流）：**P2**（观测）护栏 fail-closed 崩溃在 `except XxxGuardrailReject` 补记循环里被 `_record_result` 计一次、又由 `_record_guardrail_error` 计一次，同一次崩溃产生 `{name:1, name__error:1}` 双计数并发两个 span——补记循环对 `is_error` 的拒绝项改为 `continue`，交由 `_record_guardrail_error` 单记（C1 对规则命中路径的去重逻辑保留不变）；**P3**（统计）禁用护栏仍被 invoke，命中 `if not self.enabled: return allow()` 早退分支不写 `guardrail_name`，导致 `stats["pass"][""] += 1` 空名污染 pass_count——`_get_guardrails_by_type` 增加 `and g.enabled` 过滤，禁用护栏彻底不参与执行；**P4**（观测）fail-open 护栏崩溃 `allowed=True, is_error=True` 被当成完全健康的通过静默放行、运维无感知——`_record_result` 对任何 `is_error` 结果按 `{name}__error` 补记一条独立 error 计数（放行判定不变）；**P5**（语义）`GuardrailConfigError`（如 `allow_modify=False` 却调用 `modify()`）继承自 `GuardrailError`，被 invoke 的 `except GuardrailError` 当作普通内容拦截（`reason_kind="rule"`），把配置 bug 伪装成安全拦截——拆出独立 `except GuardrailConfigError` 分支按组件故障观测（`is_error=True` + 尊重 fail_open 姿态）。同步补 4 个行为回归测试到 `main_of_new_add`。P1（`@guardrail_creator` 装饰器护栏因不属三类子类，被 `_get_guardrails_by_type`/`run_*` 的 `isinstance` 过滤静默忽略，等于挂载即失效）属真实功能行为变更且装饰器分类方案需先定，本轮缓做、登记演进路线图 | 低：P2/P3/P4/P5 只改统计与观测维度，放行/拦截决策与钩子返回值语义逐字节不变；既有断言多用 `>= 1`（C1/N4/test_L6 等）不受影响 |
| v1.0.1 | 2026-11-23 | 六个缺陷修复：**Bug1**（安全）`GuardrailBase.invoke/ainvoke` 改用 `to_scan_text` 全量扁平化，含 `content` 键的 dict 不再只扫描 content 值而绕过其余字段（`ToolCallGuardrail` 靠 `_normalize_input=False` 保留原始 pydantic 对象）；**Bug2**（可观测）`log_guardrail_event` 仅在真实拦截时置 span ERROR，fail-open 放行不再因空 `error=""` 被误标为错误；**Bug3**（并发正确性）`before_tool_call` 拦截时返回 `ToolDecision(False, 原因)` 二元组，拦截原因随返回值携带，不再依赖实例共享属性 `_last_tool_reject_reason`；**Bug4**（死代码）删除 `invoke/ainvoke` 写入但从未被读取的控制流标记 `metadata["_raise_reject"]`，并修正声称它驱动聚合层的误导性注释；**Bug5**（文档与行为不一致）`run_input_guardrails` 等聚合函数 docstring 声称「返回所有结果 / 经过所有护栏处理后的最终输出」，而实现是 **fail-fast**（命中第一条 allowed=False 即 raise、剩余护栏不跑）——逐个修正 8 个执行函数（含 4 个异步版）与两个护栏基类的 docstring；**Bug6**（字段契约不对称）`tags`/`metadata` 是 `GuardrailBase` 上唯一没在类体声明默认值的元数据字段，`__init__` 只能靠 `hasattr(type(self), ...)` 探测子类——子类覆盖 `__init__` 且漏调 `super()` 时 `to_dict()`/`clone()` 直接 `AttributeError`（已实跑复现），且发 `py.typed` 后静态检查看不到字段契约；改为类体声明 `tags: List[str] = []` / `metadata: Dict[str, Any] = {}` 并删掉 hasattr 分支（类值仅作兜底、`__init__` 依旧逐实例拷贝）；附带修正文档里“基类有 `unregister()`”的不存在 API 描述 | 低：钩子返回值语义扩展，旧式 `bool` 实现与主循环 `if not ret:` 判定均仍兼容（`ToolDecision` 重写 `__bool__`）；仅直接断言 `is False` 的测试需改为 `not ret`；Bug4 只移除私有键，Bug5 只改注释/docstring，Bug6 只补齐原本就等价的类默认值（无子类声明 tags 时新旧结果完全相同），三者公开行为逐字节不变 |
| v1.0.0 | 2026-11-19 | 初版落地：核心基类GuardrailBase、三层护栏（输入/输出/工具调用）、中间件接入、三个预设规则、全局注册表、装饰器、异常体系、可观测埋点、63项单元测试、使用示例、模块文档 | 无 |


---

## 二、先想清楚：Guardrails 解决什么问题

大语言模型本身没有安全边界，在生产环境使用Agent会面临三类核心安全风险：
1. **输入侧风险**：Prompt注入、越狱攻击、违规内容请求，诱导模型执行恶意操作
2. **工具侧风险**：高危命令执行、敏感文件访问、未授权操作、越权调用工具
3. **输出侧风险**：敏感数据泄露（身份证/手机号/密码）、违规内容输出、幻觉导致的错误信息

Guardrails模块作为独立于LLM的确定性防护层，完全规避Prompt防御的脆弱性，用代码逻辑在Agent全链路设置安全检查点，在风险操作发生前直接拦截，不需要依赖大模型的概率判断，从根本上解决Agent的安全问题。

设计目标：
- **零侵入**：通过中间件接入现有ReActAgent，不需要修改任何核心执行流代码
- **零新增依赖**：完全复用现有抽象，不引入任何重型安全依赖
- **可扩展**：支持自定义护栏规则，通过继承基类或装饰器快速扩展
- **高性能**：默认并行执行检查，不显著增加请求延迟
- **容错性**：支持fail_open模式，护栏异常不影响主流程可用性
- **可观测**：所有拦截事件自动对接Trace体系，可审计可回溯

---

## 三、主流框架 Guardrails 实现对比

| 框架 | 输入护栏 | 输出护栏 | 工具护栏 | 中间件集成 | 可扩展 | 零依赖 |
|------|----------|----------|----------|------------|--------|--------|
| LangChain Guardrails | ✅ | ✅ | ✅ | ✅（LCEL） | ✅ | ❌ 依赖多 |
| OpenAI Agents SDK | ✅ | ✅ | ✅ | ✅（原生） | ✅ | ❌ 绑定OpenAI生态 |
| LlamaIndex Guardrails | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ 依赖Nvidia NeMo |
| Anthropic Claude Guardrails | ✅ | ✅ | ✅ | ❌ 需手动接入 | ❌ 闭源 | ❌ |
| **本实现** | ✅ | ✅ | ✅ | ✅（对齐SkillMiddleware接口） | ✅ | ✅ 零新增依赖 |

本实现的取舍：
- 对齐OpenAI Agents SDK的三层护栏划分，命名保持一致
- 对齐项目现有中间件接口规范，和SkillMiddleware完全兼容，接入成本为0
- 优先实现规则型基础护栏，小模型语义护栏留待后续版本扩展
- 完全复用现有Trace、Registry等核心抽象，不重复造轮子

---

## 四、模块架构

### 4.1 目录结构
```
fast_agents/guardrails/
├── __init__.py                    # 模块导出
├── errors.py                      # 统一异常体系
├── guardrail_base.py              # GuardrailBase基类 + 全局注册表 + @guardrail_creator装饰器
├── input_guardrail.py             # InputGuardrail输入护栏基类 + 执行函数
├── output_guardrail.py            # OutputGuardrail输出护栏基类 + 执行函数
├── tool_call_guardrail.py         # ToolCallGuardrail工具护栏基类 + 前后置执行函数
├── guardrail_middleware.py        # GuardrailMiddleware中间件，零侵入接入Agent
├── guardrail_trace.py             # Trace可观测埋点
└── preset/                        # 开箱即用预设护栏
    ├── __init__.py
    ├── safety_guardrail.py        # 基础内容安全检查
    ├── injection_guardrail.py     # Prompt注入检测
    └── permission_guardrail.py    # 高危工具权限控制
```

### 4.2 数据流
```
用户输入 → InputGuardrails（输入检查）→ LLM推理 → 工具调用 → ToolCallGuardrails（前置检查）→ 工具执行 → ToolCallGuardrails（后置结果检查）→ LLM继续推理 → OutputGuardrails（输出检查）→ 返回用户
                                                         ↓ 拦截                                               ↓ 拦截                                   ↓ 拦截
                                                    返回拦截提示                                         返回拦截提示                            返回拦截提示
```

### 4.3 与其他模块的关系
- 依赖：`messages`、`tools`、`trace`、`errors` 等核心基础模块
- 被依赖：`ReActAgent` 通过中间件接口调用，零侵入
- 兼容：完全兼容现有MCP、Skill、Memory等所有模块，工具调用全链路覆盖

---

## 五、核心机制：三层防护全链路安全

Guardrails采用三层防护架构，覆盖Agent执行全流程的每个风险点：

### 5.1 输入护栏（InputGuardrail）
- 执行时机：用户输入进入LLM推理前
- 作用：拦截Prompt注入、违规内容、恶意请求，提前阻断不浪费LLM调用成本
- 典型场景：注入攻击检测、敏感词过滤、请求长度限制、身份权限校验

### 5.2 工具调用护栏（ToolCallGuardrail）
- 执行时机：工具调用前（前置检查）+ 工具返回结果后（后置检查）
- 作用：拦截高危操作、检查工具入参合法性、防止工具返回结果泄露敏感数据
- 覆盖范围：本地Tool、MCP服务、Skill动作所有工具调用
- 典型场景：高危命令拦截（rm -rf）、敏感路径访问拦截、返回结果敏感数据检测

### 5.3 输出护栏（OutputGuardrail）
- 执行时机：Agent最终结果返回给用户前
- 作用：过滤敏感信息、违规内容，支持内容自动脱敏/修改
- 典型场景：手机号/身份证自动脱敏、违规内容拦截、输出格式校验
- ⚠ **它在主循环的哪个位置生效**（路线图 v2.74.0 实测说清，agents 域 A2）：护栏跑在本轮正文已产出**之后**，
  改写结果落在 `run_stream` / `arun_stream` 的 `end` 事件与 `run` / `arun` 的返回值上；
  记忆沉淀（`_remember`）、trace span、`🎉 最终答案` 打印一直用的都是护栏后正文。
  已确认拦住的两条绕过口子：①`return_direct` 工具直返结果同样过护栏（C3）；②护栏钩子自身抛异常按
  fail-closed 兜底替为拦截文案（S1），不把模型原始输出透给用户。
  仍保留的限制只有流式：`text` 增量为了打字机效果在护栏之前 `yield`，收不回，详见 §10.2 第 12 条

---

## 六、核心数据模型

### 6.1 GuardrailResult
护栏检查返回的统一结构体：
| 字段 | 类型 | 说明 |
|------|------|------|
| allowed | bool | 是否允许通过，False表示触发拦截 |
| reason | str | 拦截原因说明，放行时可为空 |
| metadata | Dict[str, Any] | 附加元数据，用于日志和Trace上报 |

工厂方法：
- `GuardrailResult.allow(metadata=None)`：构造放行结果
- `GuardrailResult.reject(reason, metadata=None)`：构造拦截结果

`metadata` 由 `invoke()`/`ainvoke()` 统一补齐的约定字段（均为**只供观测的诊断字段，不参与控制流**）：

| 键 | 写入时机 | 说明 |
|------|----------|------|
| `duration_ms` | 所有路径 | 本次检查耗时（毫秒） |
| `guardrail_name` | 所有路径 | 护栏名，中间件据此归类统计与打点 |
| `is_error` / `guardrail_error` / `guardrail_error_type` | 护栏自身异常 | 区分「组件故障 fail-closed」与「规则命中」，中间件只读 `is_error` |
| `rejected_via_exception` / `exception_type` | 作者主动抛 `GuardrailError` | 标记拦截来自异常式写法，仅供诊断 |
| （护栏作者自带键） | `raise XxxReject(details=...)` | `details` 被合并进 metadata，再经 `e.details` 透传回上层 |

⚠️ **Bug4 约定**：往 `metadata` 里只写「给人看 / 给回调看」的诊断字段，**不写下划线开头的私有控制流标记**。
历史上 `invoke()` 曾写 `metadata["_raise_reject"] = True` 并声称「由聚合层转换成 XxxGuardrailReject」，
但聚合层只看 `not result.allowed`、从不读它，而 metadata 会作为 `e.details` 透传给用户回调，
等于把私有字段泄漏到公开面——现已删除。

### 6.2 ToolCallInfo
工具调用信息结构体，作为工具护栏前置检查输入：
| 字段 | 类型 | 说明 |
|------|------|------|
| tool_name | str | 工具名称 |
| tool_args | Dict[str, Any] | 工具调用参数 |
| tool_type | str | 工具类型：local/mcp/skill |
| source | str | 工具来源（MCP server名/Skill名） |
| call_id | str | 工具调用ID，用于链路追踪 |

### 6.3 ToolResultInfo
工具执行结果结构体，作为工具护栏后置检查输入：
| 字段 | 类型 | 说明 |
|------|------|------|
| tool_name | str | 工具名称 |
| success | bool | 工具是否执行成功 |
| output | str | 工具返回输出内容 |
| error | str | 工具执行错误信息 |
| metadata | Dict[str, Any] | 附加元数据 |

### 6.4 异常类
所有护栏异常统一继承`GuardrailError`：
- `InputGuardrailReject`：输入护栏拦截时抛出
- `OutputGuardrailReject`：输出护栏拦截时抛出
- `ToolCallGuardrailReject`：工具护栏拦截时抛出
- `GuardrailConfigError`：护栏配置错误（如重复注册同名护栏）

---

## 七、逐个文件讲解（源码导读）

### 7.1 guardrail_base.py
核心基类与基础设施：
- `GuardrailBase`：所有护栏的抽象基类，定义统一接口
  - 核心方法：`check()`（子类实现同步检查）、`acheck()`（子类实现异步检查）、`invoke()`（带错误处理的同步入口）、`ainvoke()`（带错误处理的异步入口）
    —— Bug1 修复：`invoke`/`ainvoke` 在调用 `check()` 前用 `to_scan_text()` 而非 `content_to_text()`
    归一化输入：前者对任意 dict/list/对象全量扁平化，不会丢掉非 `content` 字段；
    类开关 `_normalize_input: bool = True`（默认全量扫描），依赖原始 pydantic 对象做 isinstance 分发的
    护栏（如 `ToolCallGuardrail`）覆盖为 `False`，拿到未文本化的原对象。
    —— Bug4 修复：「作者主动抛 `GuardrailError`」分支归一为 `GuardrailResult(allowed=False)` 后，
    不再往 metadata 里写无人消费的控制流标记 `_raise_reject`（聚合层只看 `not allowed`）；
    异常式与返回式两种拦截写法经 `invoke` 后链路完全一致（已由 L7 测试锁定）。
  - 元数据：`name`/`description`/`priority`/`enabled`/`fail_open`/`tags`/`metadata`，用于注册、排序与扩展
    —— Bug6 修复：这七个字段全部在 `GuardrailBase` **类体显式声明默认值**，子类只覆盖自己需要的即可
    （可以直接写 `tags = ["audit"]`）。此前 `tags`/`metadata` 是唯一没声明的两个，`__init__` 只能靠
    `hasattr(type(self), 'tags')` 探测子类：子类一旦覆盖 `__init__` 又漏调 `super().__init__()`，
    其余字段能从类默认值兜住，而 `tags`/`metadata` 直接 `AttributeError`（`to_dict()` / `clone()` 就炸）。
    类级值只作共享兜底常量，`__init__` 一律 `list()`/`dict()` 拷贝成实例属性，实例之间、实例与类之间不串扰。
  - hooks机制：`before_check()`/`after_check()`/`on_error()`，可扩展自定义逻辑
  - 注册：实例只有 `register()`（把当前实例登记为全局原型）；**基类没有 `unregister()`**，
    反注册在注册表层：`global_guardrail_registry.unregister_guardrail(实例或名称)`；
    中间件层面的移除用 `GuardrailMiddleware.remove_guardrail(name)`
- 全局注册表：`_GLOBAL_GUARDRAIL_REGISTRY`，存储所有已注册护栏实例
- `@guardrail_creator`装饰器：快速注册函数式护栏，对齐`@tool_creator`使用方式；v1.0.3 起支持 `guardrail_type="input"/"output"/"tool_call"`（默认 input）决定继承基类与参与阶段；装饰 GuardrailBase 子类时原样返回（类型由类的继承决定）
- 工具函数：`list_guardrails()`（按优先级排序列出所有护栏）、`get_guardrail(name)`（按名称获取护栏实例）、`to_scan_text(input_data)`（C2：把任意结构化输入 dict/list/对象递归展开为可扫描纯文本，供内容类护栏使用；str 原样返回，不影响既有文本路径）

### 7.2 input_guardrail.py
输入护栏实现：
- `InputGuardrail`：继承GuardrailBase，所有输入检查护栏的基类
- 执行函数：`run_input_guardrails()`（同步）、`arun_input_guardrails()`（异步），按 priority 升序**逐个**执行；**fail-fast**：命中第一条拦截即抛 `InputGuardrailReject`、剩余护栏不再跑，因此只有全部放行时才会返回结果列表（已完成的部分挂在 `e.results`，见 S5）

### 7.3 output_guardrail.py
输出护栏实现：
- `OutputGuardrail`：继承GuardrailBase，所有输出检查护栏的基类，支持`allow_modify`开关允许修改输出内容
  - 快捷方法：`reject()`（返回 `allowed=False` 的结果；在 `check()` 里直接 `raise OutputGuardrailReject` 也支持）、
    `modify()`（需 `allow_modify=True`，把新内容写进 `metadata["modified_output"]` 后返回放行结果）
- 执行函数：`run_output_guardrails()`（同步）、`arun_output_guardrails()`（异步），按 priority 升序**逐个**执行并自动叠加 `modify()` 的改写结果；同样是 **fail-fast**——一旦拦截即抛 `OutputGuardrailReject`，**排在后面的脱敏/改写护栏没机会执行**（有意的 fail-fast 姿态，不是遗漏）

### 7.4 tool_call_guardrail.py
工具调用护栏实现：
- `ToolCallGuardrail`：继承GuardrailBase，所有工具检查护栏的基类，支持`check_result`开关控制是否执行后置结果检查
  - `_normalize_input = False`（Bug1）：`check()` 依赖 `isinstance(input_data, ToolCallInfo/ToolResultInfo)` 分发，
    因此不能被基类文本化，必须拿到原始 pydantic 对象
  - 钩子方法：`before_call()`（前置检查，子类实现）、`after_call()`（后置检查，子类实现）
- 执行函数（全部 **fail-fast**，与 Bug5 修正后的 docstring 一致）：
  - 前置：`run_tool_call_guardrails()`/`arun_tool_call_guardrails()`，工具调用前执行检查，全部放行才返回结果列表
  - 后置：`run_tool_result_guardrails()`/`arun_tool_result_guardrails()`，工具返回结果后执行检查，
    返回 `(current_result, results)`，`current_result` 为依次叠加护栏改写后的结果
  - 命中**第一条** `allowed=False` 即抛 `ToolCallGuardrailReject`，剩余护栏不再执行；
    `details` 额外携带 `tool_name`（前置还带 `tool_args`），`results` 只含已执行项（见 S5）

### 7.5 guardrail_middleware.py
中间件实现，对齐SkillMiddleware生命周期接口：
- 核心方法：
  - `before_agent_run()`：Agent运行前执行输入护栏检查
  - `before_tool_call()`：工具调用前执行工具前置护栏检查。放行返回 `True`；拦截返回
    `ToolDecision(False, 原因)` 二元组（Bug3 修复）——拦截原因随返回值携带，同一 middleware 实例被
    多线程 / 多 Agent 并发调用时，主循环读到的永远是本次调用自己的原因，消除共享属性竞态；
    `_last_tool_reject_reason` 仍同步写入，但仅供观测/调试与旧式 `bool` 调用方回退，不再是正确性依赖。
    `ToolDecision` 继承 `tuple` 并重写 `__bool__` 委派给 `allowed`，因此旧调用方的
    `if not ret:` 判定对拦截元组同样成立（裸 tuple 是 truthy，不重写会把 `(False, "原因")` 误判为放行）。
    拦截（或钩子自身抛异常被主循环 fail-closed 兜底）时，主循环会为本批次**每一个**
    `tool_call` 各生成一条 tool 消息：被拦的那条写拦截理由，同批其余未执行的写「未执行」占位，
    以保证 assistant 消息声明的 tool_call_id 与 tool 消息严格配对（缺一条即 OpenAI 返回 400）。
    主循环用 `_coerce_tool_args()` 把工具参数统一解析成 dict 后再传给钩子，保证护栏检查与实际
    执行用同一份参数（避免「护栏看 dict、工具收字符串」的不一致漏洞）。
    拦截统计：`run_*_guardrails` 抛出的 `ToolCallGuardrailReject` 异常已携带完整 `e.results`
    （含拒绝项），中间件只遍历 `e.results` 记录一次，不再另造 `reject_result` 重复计数。
    同一「不重复计数」规则现覆盖全部四类钩子（C1 修复，同步/异步严格一致）：
    `before_agent_run`/`after_agent_run` 的 `InputGuardrailReject`/`OutputGuardrailReject`
    分支也已只遍历 `e.results` 补记一次，每次拦截统计计 1 次、发 1 个 span
  - `after_tool_call()`：工具返回后执行工具后置护栏检查；结果为 `str` 时替换 observation，
    为 `None` 时表示未修改、主循环保留原始渲染结果
  - `after_agent_run()`：Agent返回结果前执行输出护栏检查，返回处理后的结果。
    主循环里**两条产出最终答案的路径都会走它**（C3 修复）：①模型直接给出文本；
    ②`return_direct` 工具命中后直接短路的工具输出。修复前只有路径①调用本钩子，
    `return_direct` 的工具原始输出会绕过全部输出护栏原样透传给用户；现两段判定收敛为
    `ReActAgent._apply_output_guardrails()` 单一方法，两条路径共用，新增产出路径时只需调该方法。
- 功能特性：
  - `auto_register_presets`：初始化时自动加载三个预设护栏
  - **tool_registry 全时序注入（A2）**：无论 `tool_registry` 是先还是后于 `add_guardrail()` 设置，
    权限护栏副本都能读到它；实例自己显式带了 registry 时不覆盖，`remove_guardrail()` 后重挂同样生效
  - **显式优先（A1）**：构造器 `guardrails` 中与预设**同名**的护栏胜出，对应预设跳过加载并打 info 日志；
    仅约束构造期，构造完成后用公开 `add_guardrail()` 注册同名护栏仍会替换预设副本并打 warning
  - `fail_open`：护栏执行出错时是否放行，默认True（容错优先）
  - 动态管理：`add_guardrail()`/`remove_guardrail()`/`enable_guardrail()`/`disable_guardrail()`
  - 自定义回调：`on_input_reject`/`on_output_reject`/`on_tool_reject`，支持自定义拦截返回内容
  - 统计功能：`get_stats()`返回拦截/放行统计数据

### 7.6 guardrail_trace.py
可观测埋点：
- `log_guardrail_event()`：记录护栏检查事件到TraceLogger，包含护栏名称、检查结果、拦截原因、上下文信息
- 所有护栏通过/拦截事件自动上报，可在Trace链路中查询审计

### 7.7 preset/ 预设护栏
- `safety_guardrail.py`：基础内容安全检查。输入侧（`DEFAULT_BLOCKED_PATTERNS`）识别用户「索取」违法/危险信息的请求；输出侧改用独立模式表（`DEFAULT_OUTPUT_BLOCKED_PATTERNS`），只拦截「确实在产出」可执行危害信息的表述，并叠加缓和语境豁免（**命中片段所处子句内**出现「违法/禁止/抱歉/无法提供」等否定、警示、拒答词则放行；v1.0.6 / B5 后豁免不跨子句外溢，「请勿模仿：」这类单独成句 / 成括号的免责声明挡不住整段危害步骤，逗号 / 顿号故意不算边界以保住同句科普），避免模型正常解释「如何制造毒品是违法的」被误拦。属高召回确定性基线，非高精度分类器。C2 修复后，非 str 的结构化输入（如 `{"query": "如何制造炸弹"}`）不再直接放行，而是经 `to_scan_text` 递归展开后照常扫描。
- `injection_guardrail.py`：Prompt注入检测。中英双语规则表（中文「忽略之前的指令」「越狱」等，英文 `ignore previous instructions` / `do anything now` / `jailbreak` 等），匹配前先做 NFKC 归一化 + 转小写 + 去零宽字符，并在「去空白/分隔符」的压缩串上二次匹配，可对抗大小写、全角同形字与加空格绕过；C2 修复后同样覆盖 dict/list 结构化载荷（先 `to_scan_text` 展开再进入双视图匹配）。
- `permission_guardrail.py`：高危工具权限控制，优先读 `ToolRegistry` 的 `requires_approval` 元数据（单一事实源），默认拦截 shell 执行、敏感路径写入、危险命令等高危操作。敏感路径扫查（第 5 步）自 v1.0.7 / B2 起不再只看顶层字符串参数，而是按先序**逐层展开** `tool_args`：dict 的键与值都扫、list / tuple / set / frozenset 逐元素扫、`os.PathLike` 经 `os.fspath` 扫；实现用**显式栈 + 分支 id 元组剪枝**（不用递归、不设深度上限），避免护栏自身抛 `RecursionError` / 死循环把主流程打挂；命中时 `metadata["arg_name"]` 给出定位路径（顶层仍为裸参数名，嵌套形如 `payload.path` / `paths[0]`）。

### 7.8 errors.py
统一异常体系，每个模块独立异常类，对齐项目现有errors.py规范。

---

## 八、快速上手

### 8.1 最简使用：一行启用所有预设安全护栏
```python
from fast_agents import ReActAgent, GuardrailMiddleware

# 只需在middleware中加入GuardrailMiddleware即可启用所有预设安全护栏
agent = ReActAgent(
    llm=llm,
    tools=tools,
    middleware=[GuardrailMiddleware(auto_register_presets=True)]
)

# 之后正常调用agent.run()即可，所有护栏自动生效
result = agent.run("帮我写一个Python排序算法")
```

### 8.2 自定义输入护栏
```python
from fast_agents import InputGuardrail, GuardrailResult, GuardrailMiddleware

class KeywordBlockGuardrail(InputGuardrail):
    name = "keyword_block"
    description = "拦截包含特定关键词的请求"
    priority = 20
    BLOCKED_KEYWORDS = ["内部机密", "保密文件"]

    def check(self, content: str, context=None, **kwargs) -> GuardrailResult:
        for keyword in self.BLOCKED_KEYWORDS:
            if keyword in content:
                return GuardrailResult.reject(f"请求包含禁止访问的关键词：{keyword}")
        return GuardrailResult.allow()

# 使用自定义护栏
middleware = GuardrailMiddleware(auto_register_presets=False)
middleware.add_guardrail(KeywordBlockGuardrail())
```

### 8.3 自定义输出护栏（自动脱敏手机号）
```python
import re
from fast_agents import OutputGuardrail, GuardrailResult

class PhoneDesensitizeGuardrail(OutputGuardrail):
    name = "phone_desensitize"
    description = "自动脱敏输出中的手机号"
    priority = 25
    allow_modify = True  # 允许修改输出内容
    PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")

    def check(self, content: str, context=None, **kwargs) -> GuardrailResult:
        def replace_phone(match):
            phone = match.group()
            return phone[:3] + "****" + phone[7:]
        modified = self.PHONE_PATTERN.sub(replace_phone, content)
        if modified != content:
            return self.modify(modified, reason="已自动脱敏手机号")
        return GuardrailResult.allow()
```

### 8.4 自定义工具护栏（拦截高危文件删除）
```python
from fast_agents import ToolCallGuardrail, GuardrailResult, ToolCallInfo

class FileDeleteProtectGuardrail(ToolCallGuardrail):
    name = "file_delete_protect"
    description = "禁止删除系统关键目录文件"
    priority = 15
    PROTECTED_PATHS = ["/etc", "/usr", "C:\\Windows"]

    def before_call(self, tool_call: ToolCallInfo, context=None, **kwargs) -> GuardrailResult:
        if tool_call.tool_name == "delete_file":
            file_path = tool_call.tool_args.get("file_path", "")
            for protected_path in self.PROTECTED_PATHS:
                if file_path.startswith(protected_path):
                    return GuardrailResult.reject(f"禁止删除系统保护目录 {protected_path} 下的文件")
        return GuardrailResult.allow()
```

更多完整示例见 `examples/guardrails/guardrails_basic_demo.py`，无需API Key即可运行。

---

## 九、API 速查

### GuardrailBase 基类
| 方法/属性 | 说明 |
|-----------|------|
| `name: str` | 护栏唯一名称 |
| `description: str` | 护栏描述 |
| `priority: int` | 执行优先级，数字越小越先执行 |
| `enabled: bool` | 是否启用 |
| `check(input_data, context=None)` | 同步检查逻辑，子类实现 |
| `acheck(input_data, context=None)` | 异步检查逻辑，子类实现 |
| `invoke(input_data, context=None)` | 同步执行入口，带错误处理 |
| `ainvoke(input_data, context=None)` | 异步执行入口，带错误处理 |
| `register()` | 注册到全局注册表（基类只有这个方法） |
| `clone()` | 深拷贝独立副本，供多实例隔离 |
| `to_dict()` | 序列化为字典（含 `tags`/`metadata`） |

> 反注册不在 `GuardrailBase` 上：请用 `global_guardrail_registry.unregister_guardrail(实例或名称)`（不存在时返回 `False`，不抛 KeyError）。

### GuardrailMiddleware 中间件
| 方法/参数 | 说明 |
|-----------|------|
| `__init__(auto_register_presets=True, fail_open=True, tracer=None)` | 构造函数；`guardrails` 显式传入的护栏与预设同名时，显式优先、预设让位（A1） |
| `add_guardrail(guardrail)` | 动态添加护栏；middleware 已有 `tool_registry` 时，新挂载且自身未带 registry 的权限护栏副本会补拿到它（A2） |
| `remove_guardrail(name)` | 移除护栏 |
| `enable_guardrail(name)` | 启用护栏 |
| `disable_guardrail(name)` | 禁用护栏 |
| `before_agent_run(user_input, context=None)` | 输入检查，返回处理后的输入 |
| `before_tool_call(tool_name, tool_args, context=None)` | 工具前置检查；放行返回 `True`，拦截返回 `ToolDecision(False, 原因)`（旧式 `bool` 实现仍兼容）；拦截时主循环为整批 tool_call 补齐响应消息 |
| `after_tool_call(tool_name, success, output="", error="", context=None, **kwargs)` | 工具后置检查；`str` 替换 observation，`None` 表示未修改、保持原结果 |
| `after_agent_run(agent_output, context=None)` | 输出检查，返回处理后的输出 |
| `get_stats()` | 获取拦截/放行统计 |

### ToolDecision（Bug3：线程安全的拦截返回值）
| 字段/方法 | 说明 |
|-----------|------|
| `ToolDecision(allowed, reason="")` | `before_tool_call` 的二元组返回值，继承自 `tuple`，可直接解包为 `(allowed, reason)` |
| `.allowed` | 等价于 `self[0]`，`False` 表示拦截 |
| `.reason` | 等价于 `self[1]`，拦截原因文本，随返回值一同传递 |
| `__bool__` | 重写为 `bool(allowed)`，保证 `if not ret:` / `assert not ret` 对拦截元组成立 |

---

## 十、设计决策与常见坑

### 10.1 设计决策
1. **为什么护栏不依赖LLM判断？**
   - LLM判断本身存在被注入绕过的风险，确定性规则是安全防护的最后一道防线
   - LLM判断延迟高、成本高，不适合作为高频调用的基础防护
   - 小模型语义护栏作为可选扩展放在后续版本，不影响基础能力

2. **为什么采用中间件接入而不是修改Agent核心代码？**
   - 零侵入，不破坏现有执行流，用户可以选择是否启用护栏
   - 对齐项目现有SkillMiddleware的接口规范，学习成本为0
   - 方便插拔，测试/调试时可以随时移除中间件

3. **为什么默认fail_open=True？**
   - 护栏本身的bug不应该导致整个服务不可用，容错优先
   - 生产环境如果需要严格安全模式，可以手动设置fail_open=False

4. **为什么默认优先级数字越小越先执行？**
   - 对齐其他框架通用设计，高危拦截规则优先级更高（数字更小），先执行快速拦截减少不必要的计算

### 10.2 常见坑
1. **输出护栏修改内容必须设置allow_modify=True**
   - 否则调用modify()方法会抛出异常，避免未经允许修改用户输出

2. **工具后置检查必须设置check_result=True**
   - 默认只执行前置检查，需要检查工具返回结果时必须显式开启，避免不必要的性能开销

3. **护栏名称必须全局唯一**
   - 重复注册同名护栏会抛出GuardrailConfigError，建议命名采用`模块_功能`格式避免冲突

4. **自定义异步护栏必须实现acheck方法**
   - 默认acheck会调用同步check，异步场景下建议重写实现真正的异步逻辑，避免阻塞事件循环

5. **写自己的中间件时，别用裸 tuple 做 `before_tool_call` 的拦截返回值**
   - 裸 `(False, "原因")` 是 truthy，`if not ret:` 会把它当成放行，直接打开安全口子；
     要么返回 `ToolDecision(False, "原因")`（已重写 `__bool__`），要么返回纯 `bool`
   - 主循环侧不依赖 `bool(tuple)`，而是取 `_ret[0]` 判定，两种写法都能正确拦截

6. **多个 Agent 共用一个 `GuardrailMiddleware` 实例时，不要读 `_last_tool_reject_reason` 做业务判断**
   - 该属性是实例级共享状态，并发下会被别的线程覆盖；拦截原因请取返回值 `ToolDecision.reason`

7. **不要指望 `run_*_guardrails` 能拿到全部护栏结果**（Bug5）
   - 它是 **fail-fast**：命中第一条拦截即抛异常，后面的护栏根本不跑，`e.results` 只含已执行项；
     需要「跑完全部再汇总」时请自己逐个 `g.invoke(...)` 收集
   - 同理，把“后置脱敏/改写护栏”的 `priority` 排在“拦截护栏”之后，一旦拦截就不会脱敏——顺序要按意图设计

8. **别把自定义护栏起名成预设同名，否则你以为在“覆盖预设”（v1.0.4 / A1 已修正为显式优先）**
   - 现行为：`GuardrailMiddleware(guardrails=[我的护栏])` 中若 `我的护栏.name == "prompt_injection"`，
     内置同名预设**不会加载**，你的规则说了算（有 info 日志可查）；其余预设照常生效
   - 想要“预设 + 自定义”共存时，给自定义护栏取另一个名字（建议 `模块_功能` 格式）；
     想要完全接管某一预设的行为时，才复用它的名
   - 注意区分：构造**完成后**调 `add_guardrail()` 属显式替换，会顶掉已挂载的同名预设并打 warning

9. **`tool_registry` 与 `add_guardrail()` 的先后次序不再影响拦截（v1.0.5 / A2）**
   - 修复前：`tool_registry` 只在被赋值的**那一刻**广播；已经赋值之后再 `add_guardrail` 进来的
     `BasicPermissionGuardrail` 副本永远拿不到 registry，`requires_approval=True` 的工具直接放行
   - 现行为：两种时序都生效；注入只作用于 clone 副本，绝不回写你传入的原型实例
   - 想要护栏用自己的 registry（而非 middleware 的）：在构造护栏时显式传 `tool_registry=...`，
     `add_guardrail()` 不会覆盖它；但后续给 `middleware.tool_registry` 赋值仍会无条件广播覆盖（setter 旧语义）

10. **别指望靠一句免责声明放行危害输出（v1.0.6 / B5）**
   - 豁免词只在**命中片段所处的那个子句**内生效：「请勿模仿：」「（本内容违法）」「免责声明：本文仅供学术参考。」
     这类单独成句 / 成括号的话只修饰它自己，不再把后面的整段步骤一并洗白
   - 确实要保留旧的跨句豁免（如接入的是自带全文声明的第三方内容）：自己的护栏类覆写
     `_has_mitigating_context` 即可，不要改预设
   - 收紧的另一面是误拦：同子句内没有表内缓和词的新闻 / 司法通报体（「已被依法刑事拘留」）会被拦。
     词表刻意保持现状——扩词会同时放大绕过面；真要兼顾，叠一层模型判定类护栏而不是改词表
   - `mitigation_window` 仍然是「前后各 N 字符」，但现在只是**上限**：调大它不能退回跨句豁免，
     设 0 则只看命中片段自身

11. **不要把路径藏进嵌套参数就以为看不见（v1.0.7 / B2）**
   - 敏感路径扫描会逐层展开：`{"payload": {"path": "/etc/passwd"}}`、`{"paths": ["/etc/shadow"]}`、
     路径写在**字典键**上 `{"files": {"/etc/passwd": "r"}}`、传 `Path` 对象，如今都拦得住
   - 但它是**子串表**不是路径解析器，仍有漏检面（本版不处理）：变形写法（`/etc//passwd`、
     `/etc/./passwd`、URL 编码、八进制/符号链）、拼接后才出现的路径、非字符串载体（bytes / base64）
     扫不到；要堵这些需接入真实文件系统做 `realpath` 判定，超出护栏层职责
   - ⚠ **已知未修缺陷（分隔符归一化）**：表内 unix 条目全是正斜杠，而 Windows 上
     `os.fspath(Path("/etc/shadow"))` 会被展开为 `\etc\shadow`，与表内条目对不上而漏拦；
     顶层字符串写成 `"\\etc\\shadow"` 同样漏——这是修复前后**都存在**的老问题，不属 B2（嵌套）范畴，
     且归一化会把 `/sys` 这类短条目的子串误伤面放大，因此单列而不并入本次
   - 关闭扫描请用 `check_sensitive_paths=False`，语义不变（彻底不扫，不是只关嵌套）

12. **别以为挂了输出护栏就能拦回已经流出去的正文（路线图 v2.74.0 / agents 域 A2）**
    - 护栏能改的只有两处：`run` / `arun` 的**返回值**、`run_stream` / `arun_stream` 的 **`end` 事件正文**；
      `text` 增量在护栏之前就已 `yield`，物理上收不回（同步与异步两条链路逐字一致）
    - 旧版除了流式这一条，还多一个真正的漏洞：`run` / `arun` 优先拼接 `text` 增量，于是非流式调用方
      拿到的就是脱敏 / 拦截前的原文——modify 与 reject 两种语义均实测泄露过，现 v2.74.0 已修
    - 调用方对策：需要强护栏语义请用 `run` / `arun`；要保留打字机效果又必须脱敏，则累加 `text` 到
      `end` 事件后一次性展示；拦得住的场景可凭 `end.finish_reason == "guardrail_block"` 清空已展示内容，
      而**脱敏（modify）场景没有任何事件级标记**（`StreamChunk` 无 `metadata` 字段可挂）
    - 自定义 `AgentBase` 子类时注意：护栏策略只落在 `ReActAgent`，基类默认的 `arun` 仍是“聚合 `text`
      增量”的原始语义（基类不知道护栏的存在）——自己写了主循环又挂护栏的，请像 `ReActAgent.arun`
      那样覆盖并取护栏后正文

### 10.3 跨平台注意事项
- 敏感路径判定是**关键词子串匹配**，不做 `os.path` 归一化也不做 `realpath`：表内 unix 条目为正斜杠
  字面量，Windows 上反斜杠写法与 `Path` 对象展开结果对不上→会漏拦（见 §10.2 第 11 条，尚未修复）
- 正则表达式默认开启re.IGNORECASE，大小写不敏感匹配
- 代码本身无平台特定依赖，Windows/macOS/Linux 均可运行；但权限护栏的路径表需按目标平台自行补齐

---

## 十一、测试与验证

### 11.1 测试覆盖范围
单元测试共 **125 项**（`tests/guardrails/` 下 11 个文件：45 主用例 + 12 中间件 + 19 真实 ReActAgent 端到端
+ 16 模拟 Agent 链路 + 33 分项），覆盖：
- 基类功能：GuardrailResult结构体、装饰器注册、优先级排序、序列化
- 三类护栏：输入/输出/工具护栏的放行/拦截/修改逻辑
- 预设护栏：注册、Prompt注入检测、高危工具拦截、正常内容放行
- 中间件：初始化、动态添加/移除护栏、fail_open模式、统计功能
- 异常体系：继承关系、属性正确性
- 缺陷回归（v1.0.1）：
  - **Bug1** `test_BUG1_dict_with_content_key_does_not_bypass_scan`：含 `content` 键的 dict 其余字段不得绕过扫描（输入/输出两侧）
  - **Bug2** fail-open 放行的护栏 span 必须为 SUCCESS，不被空 `error=""` 误标为 ERROR
  - **Bug3** `test_BUG3_reject_reason_travels_with_return_value`：`ToolDecision` 的 tuple/`__bool__` 协议语义、
    拦截原因随返回值携带、8 线程共用同一 middleware 不串扰、放行路径仍返回裸 `True`
  - **Bug4** `test_BUG4_no_dead_control_flag_in_reject_metadata`：`metadata`/`e.details` 不再出现下划线开头的
    私有控制键，诊断字段保留，异常式与返回式拦截行为逐项一致
  - **Bug5** `test_BUG5_run_guardrails_fail_fast_matches_docstring`：输入/输出/工具前置/工具后置四层聚合函数
    均为 fail-fast（拦截项之后的护栏不被调用、`results` 只含已执行项、全放行才返回列表且叠加 `modify()`），
    同步与异步语义严格一致——把修正后的 docstring 锁在测试里，防止再次漂移
  - **Bug6** `test_BUG6_tags_metadata_are_declared_class_attributes`：`tags`/`metadata` 与其他五个元数据字段
    一样在类体声明并出现在 `__annotations__`；子类覆盖 `__init__` 漏调 `super()` 时 `to_dict()`/`clone()`
    不再 `AttributeError`（修复前已实跑复现）；实例级 `append`/`update` 不串扰其它实例、不回污染类默认值；
    构造器显式传参优先级最高
- 缺陷回归（v1.0.3）：
  - **P1** `test_P1_function_guardrail_participates_in_dispatch`：`@guardrail_creator` 生成的函数式护栏按
    `guardrail_type` 继承对应基类，中间件输入/工具阶段与 `use_global=True` 扫描均真正执行；非法类型值创建时
    即抛 `GuardrailConfigError`；装饰类原样返回不被误包装。同步翻转 `test_guardrails_sim_agent.py` S15
    （「现状记录 not_dispatched」→ dispatched），修正 `test_tool_call_guardrail.py::test_tool_call_allow`
    空转断言为显式注册 + 真实执行
- 缺陷回归（v1.0.4）：
  - **A1** `test_A1_explicit_guardrail_wins_over_same_name_preset`：故意不关预设、只传一个与
    `prompt_injection` 同名的自定义输入护栏，断言：挂载的是用户实例副本（非预设类）、共 4 个护栏
    （其余预设照常）、自定义规则真正拦截、内置英文注入规则因让位而不再拦截、让位必有 info 日志；
    显式 `presets=[...]` 名单同样让位；构造后公开 `add_guardrail()` 仍替换同名预设并打 warning（R1 守护）
- 缺陷回归（v1.0.5）：
  - **A2** `test_s4_6_add_guardrail_after_late_injection_still_gets_registry`（`test_s4_tool_registry_injection.py`）：
    故意不挂 `basic_permission` 预设、走「建 middleware → 注入 registry → 后 `add_guardrail` 权限护栏」时序，
    断言副本 `tool_registry is registry`、`requires_approval=True` 工具真正被拦且原因来自单一事实源、
    不回写原型；并守护：实例自带 registry 不被 `add_guardrail` 覆盖、setter 的无条件广播不变、
    middleware 无 registry 时不凭空注入且精确名单兜底照旧、非权限护栏不被塞私有字段
- 缺陷回归（v1.0.6）：
  - **B5** `test_B5_output_mitigation_does_not_leak_across_clauses`：以「建护栏 → 逐句判定」的行为断言驱动，
    断言 3 条修复前实测 ALLOWED 的跨子句豁免样本（前缀「请勿模仿：」、后缀「（本内容违法）」、
    感叹号另起一句的「警告：这是违法行为」）全部转为拦截，并保留一条修复前后都拦得住的对照样本；
    同时守护同子句科普 / 拒答的 4 条放行样本不回退、`mitigation_window=100000` 不能退回跨句豁免、
    `mitigation_window=0` 只看命中片段自身、走 `GuardrailMiddleware.after_agent_run` 的预设链路同样拦得住；
    并把「同子句无表内缓和词的新闻体被误拦」作为**已知代价**钉进断言，防止无声漂移
- 缺陷回归（v1.0.7）：
  - **B2** `test_B2_nested_tool_args_sensitive_path_still_blocked`：一个函数内 10 个子块——
    6 种修复前实测 ALLOWED 的嵌套形态（内嵌 dict / list 元素 / 多层混合 / 路径写在字典键上 /
    先序取首个命中 / `Path` 对象）全部转为拦截并校验 `arg_name` 定位路径；同时守住契约不回退：
    顶层命中的 `arg_name` 仍为裸参数名且 `reason` 文案逐字节一致、正常嵌套参数（`system` /
    `path_style` / `--syslog` / 标量）不误伤、12 层与 40 层深嵌套仍拦得住（无深度上限）、
    自引用参数 `cyc["self"] = cyc` 不崩且正常放行、`tool_args` 为非 dict（list / str / None / int / set）
    不再抛 `AttributeError`、`check_sensitive_paths=False` 仍彻底不扫、`clone()` 副本保留新能力、
    走 `GuardrailMiddleware.before_tool_call` 的预设链路同样拦得住（`ToolDecision` 为假且原因含「敏感路径」）

    > 口径说明：本项不断言「正斜杠 `Path` 在 Windows 上能命中」——`os.fspath` 会把它展开为反斜杠而与
    > 表内条目对不上，属修复前后都存在的分隔符缺陷（§10.2 第 11 条），**不把它固化成期望行为**
- 缺陷回归（v1.0.8）：
  - **B1** `test_B1_empty_set_not_replaced_by_default`：传空集合时不被 `or` 运算符替换为默认值
    （修复前 `blocked_tool_names or DEFAULT_DANGEROUS_TOOL_NAMES` 使空集合被当未传，用户显式传 `set()`
    想关闭拦截拦截时，默认危险工具名照样生效）；同时守护传 `None` 时使用默认值、传非空集合时保留用户值
  - **B3** `test_B3_extra_fields_forbidden_in_models`：三个安全关键模型（`GuardrailResult` / `ToolCallInfo` /
    `ToolResultInfo`）使用 `extra="forbid"`，拼错字段名（如 `alowd=True` / `tool_ars={...}` / `sucess=True`）
    立即抛 `ValidationError`；同时守护正确字段名能正常构造
- 缺陷回归（v1.0.9）：
  - **B4-RC1** `test_B4_injection_false_positive_fixed`：跨句误拦样本（如「Don\'t forget the previous chapters. Instructions
    for installation are below.」）应转放行——句子切分后逐句匹配不再粘连无关子句
  - **B4-RC2**：词融合误拦样本（如「jail  break the habit」）应转放行——`jailbreak` 模式改用非空格压缩视图
  - **安全守护**：18 条绕过样本（含换行绕过/全角绕过/分隔符绕过/中文绕过/jail-break-mode）全部仍拦截
  - **test_s4 字段名拼错修复**：`ToolCallInfo(arguments=...)` 改为 `ToolCallInfo(tool_args=...)` （B3 的 `extra="forbid"` 暴露出该拼写缺陷）

### 11.2 运行测试命令
本仓库测试不依赖 pytest，逐文件直接运行即可：
```powershell
cd fast_agents
python tests\guardrails\test_guardrails.py
python tests\guardrails\test_guardrails_real_agent.py
python tests\guardrails\test_guardrails_sim_agent.py
```

### 11.3 运行示例验证
```bash
python examples/guardrails/guardrails_basic_demo.py
```
无需API Key，所有示例均可直接运行，预期输出所有拦截/放行效果。

---

## 十二、下一步演进路线

Guardrails模块采用分阶段迭代，L1初版已完成，后续规划：
- **L2** 小模型语义护栏：替代纯规则匹配，降低误报率，支持更复杂的语义风险识别
- **L3** 敏感数据识别与自动脱敏：支持身份证/银行卡/地址/邮箱等多类型PII自动识别脱敏
- **L4** RBAC细粒度权限控制：基于角色/租户的工具访问控制，支持白名单/黑名单配置
- **L5** 对话上下文风险检测：跨轮次越狱检测、间接注入识别、上下文投毒防御
- **L6** 护栏规则配置中心：支持热更新规则，无需重启服务即可调整安全策略
- **L7** 攻击样本库自动更新：对抗新型注入攻击，持续更新检测规则
- **L8** 多租户护栏隔离：不同租户/用户组使用不同安全策略
- **L9** 护栏效果统计与误报反馈闭环：自动统计误报率，支持用户反馈优化规则准确率

---

## 十三、参考

- [OpenAI Agents SDK Guardrails](https://platform.openai.com/docs/guides/guardrails) —— 三层护栏划分、GuardrailFunctionOutput设计来源
- [LangChain Guardrails](https://python.langchain.com/docs/concepts/guardrails/) —— 中间件集成方式参考
- [LlamaIndex Guardrails](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/guardrails/) —— 预设规则设计参考
- [Anthropic Claude Guardrails](https://docs.anthropic.com/en/docs/agents-and-tools/guardrails) —— 安全最佳实践参考
- [Model Context Protocol Security Specification](https://modelcontextprotocol.io/specification/2025-03-26/security) —— MCP工具调用安全规范参考
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) —— LLM安全风险分类参考

---

## 附录：发布前自检清单

- [x] 所有文件CRLF换行符合要求
- [x] 命名完全对齐项目现有风格（GuardrailBase对标ToolBase、@guardrail_creator对标@tool_creator、GuardrailMiddleware对标SkillMiddleware）
- [x] 零新增依赖，不引入任何第三方包
- [x] 零破坏性改动，现有代码完全不受影响
- [x] 顶层__init__.py已完成所有类和装饰器导出
- [x] 单元测试63/63全部通过，无失败无错误
- [x] 示例代码无需API Key即可正常运行，演示所有核心功能
- [x] 模块文档完全按照v2.0模板编写，所有必备章节齐全
- [x] 演进路线图已更新，新增L系列模块编号和规划
- [x] 所有异常类放在errors.py，对齐项目现有规范
- [x] 自动对接现有Trace体系，可观测性完整
- [x] 预设护栏开箱可用，默认配置即可提供基础安全防护
