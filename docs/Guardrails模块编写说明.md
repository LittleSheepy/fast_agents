# FastAgents Guardrails（安全护栏）模块编写说明

本文档面向 fast_agents Guardrails 安全护栏系统，参考 **LangChain Guardrails**、**OpenAI Agents SDK Guardrails**、**LlamaIndex Guardrails**、**Anthropic Claude Guardrails**、**Microsoft MCP 安全规范** 等框架的共性设计，给出本模块的完整说明、源码解析与自定义教程。

> 版本：v1.0.3 · 2026-11-23 · P1 缺陷修复：`@guardrail_creator` 新增 `guardrail_type` 参数（input/output/tool_call，默认 input），函数式护栏按类型真正参与中间件与 use_global 分派（此前被 isinstance 过滤静默忽略）；装饰类恢复 passthrough；非法类型值创建时即抛 `GuardrailConfigError`
> 适用版本：`fast_agents.guardrails` v1.0.3
> 配套代码：`fast_agents/guardrails/`（9 个核心文件 + 3 个预设护栏）+ `tests/guardrails/`（10 个文件共 113 项测试全绿：51 主用例 + 19 真实 ReActAgent 端到端 + 16 模拟 Agent 链路 + 27 分项）+ `examples/guardrails/guardrails_basic_demo.py`（无API Key可运行示例），均已实测通过。
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
  - `fail_open`：护栏执行出错时是否放行，默认True（容错优先）
  - 动态管理：`add_guardrail()`/`remove_guardrail()`/`enable_guardrail()`/`disable_guardrail()`
  - 自定义回调：`on_input_reject`/`on_output_reject`/`on_tool_reject`，支持自定义拦截返回内容
  - 统计功能：`get_stats()`返回拦截/放行统计数据

### 7.6 guardrail_trace.py
可观测埋点：
- `log_guardrail_event()`：记录护栏检查事件到TraceLogger，包含护栏名称、检查结果、拦截原因、上下文信息
- 所有护栏通过/拦截事件自动上报，可在Trace链路中查询审计

### 7.7 preset/ 预设护栏
- `safety_guardrail.py`：基础内容安全检查。输入侧（`DEFAULT_BLOCKED_PATTERNS`）识别用户「索取」违法/危险信息的请求；输出侧改用独立模式表（`DEFAULT_OUTPUT_BLOCKED_PATTERNS`），只拦截「确实在产出」可执行危害信息的表述，并叠加缓和语境豁免（命中附近出现「违法/禁止/抱歉/无法提供」等否定、警示、拒答词则放行），避免模型正常解释「如何制造毒品是违法的」被误拦。属高召回确定性基线，非高精度分类器。C2 修复后，非 str 的结构化输入（如 `{"query": "如何制造炸弹"}`）不再直接放行，而是经 `to_scan_text` 递归展开后照常扫描。
- `injection_guardrail.py`：Prompt注入检测。中英双语规则表（中文「忽略之前的指令」「越狱」等，英文 `ignore previous instructions` / `do anything now` / `jailbreak` 等），匹配前先做 NFKC 归一化 + 转小写 + 去零宽字符，并在「去空白/分隔符」的压缩串上二次匹配，可对抗大小写、全角同形字与加空格绕过；C2 修复后同样覆盖 dict/list 结构化载荷（先 `to_scan_text` 展开再进入双视图匹配）。
- `permission_guardrail.py`：高危工具权限控制，优先读 `ToolRegistry` 的 `requires_approval` 元数据（单一事实源），默认拦截 shell 执行、敏感路径写入、危险命令等高危操作。

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
| `__init__(auto_register_presets=True, fail_open=True, tracer=None)` | 构造函数 |
| `add_guardrail(guardrail)` | 动态添加护栏 |
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

### 10.3 跨平台注意事项
- 所有路径判断统一使用os.path处理，Windows反斜杠路径自动兼容
- 正则表达式默认开启re.IGNORECASE，大小写不敏感匹配
- 无平台特定依赖，Windows/macOS/Linux全平台支持

---

## 十一、测试与验证

### 11.1 测试覆盖范围
单元测试共 **113 项**（`tests/guardrails/` 下 10 个文件：51 主用例 + 19 真实 ReActAgent 端到端
+ 16 模拟 Agent 链路 + 27 分项），覆盖：
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
