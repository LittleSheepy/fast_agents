# FastAgents LLM模块编写说明

本文档面向 fast_agents LLM（大语言模型）系统，参考 **LangChain BaseLanguageModel / Embeddings**、**LlamaIndex LLM 抽象**、**LiteLLM 多 provider 适配**、**OpenAI 官方 Python SDK** 等框架的共性设计，给出本模块的完整说明、源码解析与自定义教程。

> 版本：v1.5.0 · 2026-11-19 · 完成 I3 多provider原生适配：新增 `AnthropicClient`/`AsyncAnthropicClient`（Claude原生API）、`GeminiClient`/`AsyncGeminiClient`（Google Gemini原生API），同步/异步客户端接口与OpenAI版本完全对齐；门面类`LLM`新增`provider`参数（支持openai/anthropic/gemini，大小写不敏感），上层调用API完全无感；第三方SDK作为可选依赖，未安装时给出明确安装提示；零新增默认依赖，既有代码100%兼容
> 适用版本：`fast_agents.llm`
> 配套代码：`fast_agents/llm/`（12个文件：`__init__.py` / `llm.py` / `models.py` / `openai_client.py` / `async_openai_client.py` / `anthropic_client.py` / `async_anthropic_client.py` / `gemini_client.py` / `async_gemini_client.py` / `errors.py` / `retry.py` / `retry_errors.py`）+ 本文档
> 运行环境：Python 3.10+，**Windows 平台**（仓库当前唯一目标平台，见 `AGENTS.md` R11）；依赖：`openai>=1.0`（项目主依赖，**零新增默认依赖**——Anthropic/Gemini SDK作为可选依赖，通过`pip install fast-agents[anthropic]`/`pip install fast-agents[gemini]`安装）

## 目录

- [一、更新历史](#一更新历史)
- [二、先想清楚：LLM 解决什么问题](#二先想清楚llm-解决什么问题)
- [三、主流框架 LLM 实现对比](#三主流框架-llm-实现对比)
- [四、模块架构](#四模块架构)
- [五、核心机制：Chat 与 Embedding 双能力分层](#五核心机制chat-与-embedding-双能力分层)
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

> **v1.5.0 更新摘要**（2026-11-19，I3 多provider原生适配）
> 1. **新增Anthropic原生客户端** —— 独立文件`anthropic_client.py`/`async_anthropic_client.py`，基于官方`anthropic` SDK实现，支持Claude 3/3.5系列模型；同步/异步方法签名、返回结构（Message/StreamChunk）、异常映射、usage回填与OpenAI版本完全对齐；自动转换消息格式（抽离system prompt、tool角色映射、工具schema转换）；流式调用支持思考过程输出与工具调用分片聚合；结构化输出通过prompt注入JSON Schema实现，与OpenAI结构化输出调用方式完全一致；
> 2. **新增Gemini原生客户端** —— 独立文件`gemini_client.py`/`async_gemini_client.py`，基于官方`google-generativeai` SDK实现，支持Gemini 2.0系列模型；同步/异步方法与OpenAI版本完全对齐；自动转换消息格式（assistant→model角色映射、system_instruction抽离、function declaration转换）；流式调用支持工具调用分片；支持原生`response_mime_type="application/json"`配合prompt约束实现结构化输出；内置text-embedding-004 embedding模型支持，批量请求自动分片并发；
> 3. **门面类多provider路由** —— `LLM`构造函数新增`provider`参数（默认`"openai"`），支持`openai`/`anthropic`/`gemini`三种值，大小写不敏感；根据provider自动实例化对应同步/异步客户端，上层调用API（invoke/stream_invoke/ainvoke等）完全一致，切换模型厂商仅需修改provider和对应API Key，无需修改业务代码；非法provider抛出明确`LLMConfigError`提示；
> 4. **可选依赖机制** —— Anthropic/Gemini SDK不作为默认依赖安装，未安装对应SDK时调用对应provider会抛出明确的安装提示（如`pip install fast-agents[anthropic]`）；pyproject.toml新增`anthropic`/`gemini`/`all_providers`三个可选依赖组，按需安装即可；
> 5. **新增测试与示例** —— `tests/llm/test_multi_provider.py`（8项离线测试，覆盖provider路由、接口对齐、依赖缺失提示、消息格式转换、大小写不敏感）；`examples/llm/multi_provider_demo.py`演示三种provider的调用方式、异步并发、跨provider结构化输出；
> 6. **零默认依赖、零回归** —— 所有第三方SDK均为可选依赖，默认安装仅保留原有openai依赖；既有53项OpenAI相关测试全部通过，13项异步测试全部通过，向后100%兼容；
> 7. **设计决策说明** —— 多provider客户端采用独立文件组织（与OpenAI客户端平级），而非塞在同一个文件：与主流框架（LangChain/LiteLLM/OpenAI SDK）的文件组织方式一致，不同厂商的SDK导入、消息格式转换、错误映射逻辑完全独立，便于后续扩展更多provider，也避免单文件过大；客户端之间复用共用工具函数，零代码重复。

> **v1.4.0 更新摘要**（2026-11-19，I2 异步能力）
> 1. **新增异步底层客户端 `AsyncOpenAIClient`** —— 独立文件 `async_openai_client.py`，继承 `openai.AsyncOpenAI`，与同步 `OpenAIClient` 完全对等，实现 5 个异步方法：`ainvoke`/`astream_invoke`/`ainvoke_structured`/`aembed_query`/`aembed_documents`；所有方法签名、参数语义、错误映射、usage 回填逻辑与同步版本逐字节对齐；`aembed_documents` 使用 `asyncio.Semaphore(4)` 做协程级并发，与同步版本线程池 `max_workers=4` 并发度对齐；复用同步客户端的私有工具函数 `_dump_usage`/`_merge_usage`/`_classify_openai_error`/`_EMBED_BATCH_SIZE`，零代码重复；
> 2. **新增异步重试执行器** —— `retry.py` 新增 `arun_with_retry`/`arun_stream_with_retry`，与同步版本共用同一套 `RetryConfig`/`CircuitBreakerRegistry`（`threading.Lock` 在协程内并发安全，无需单独实现 asyncio.Lock 版熔断器）；退避使用 `asyncio.sleep` 不阻塞事件循环；首块前重试、开始 yield 后不重试的语义与同步版一致；支持 fn 返回 coroutine 或直接返回 async generator 两种写法；
> 3. **门面类 `LLM` 新增 5 个异步方法** —— `ainvoke`/`astream_invoke`/`ainvoke_structured`/`aembed_query`/`aembed_documents`，方法签名与同步版本完全对齐；异步 client 通过 `asyncio.Lock` 做协程安全的**懒加载**（首次 await 时初始化，后续复用同一实例），避免同步方法用户为不需要的异步 HTTP 客户端付出初始化开销；tracer span 命名对应 `llm.ainvoke`/`llm.astream_invoke`/`llm.ainvoke_structured`/`llm.aembed_query`/`llm.aembed_documents`，metadata 额外标记 `async=True` 与 `retry_attempts`；
> 4. **测试运行器自动支持异步测试** —— `tests/_runner.py` 新增 asyncio 支持：自动识别 `async def test_xxx` 测试函数，用 `asyncio.run` 执行，零配置无需改动现有测试；
> 5. **新增测试与示例** —— `tests/llm/test_async.py`（13 项离线测试，覆盖异步 client 五方法、门面懒加载、重试器、并发调用）；`examples/llm/async_demo.py` 完整演示异步非流式、流式、结构化、Embedding、gather 并发五种用法；
> 6. **零新增依赖、零回归** —— 仅新增 `import asyncio` 和 `import inspect` 标准库依赖；既有同步方法代码一字未改，53 项同步测试全部通过，向后完全兼容；
> 7. **设计决策说明** —— 异步 client 单独建文件而非塞在 `openai_client.py`：因为 `openai.OpenAI` 与 `openai.AsyncOpenAI` 是两个独立类、底层 HTTP 客户端完全分离（同步 `httpx.Client` vs 异步 `httpx.AsyncClient`），与 OpenAI 官方 SDK、LiteLLM 的文件组织方式一致；不做"单类同时支持同步异步"的组合模式封装，保持与现有继承式设计的一致性，不引入不必要的抽象层。

> **v1.3.1 更新摘要**（2026-11-19，`stream_invoke` 参数透传修复）
> 1. **修复 `OpenAIClient.stream_invoke` 丢失额外 kwargs 的 bug** —— 其 `create_kwargs` 原先只显式带 `model` / `messages` / `tools` / `temperature` / `max_tokens` / `stream`，缺少 `invoke` 里的 `**{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}` 一行，导致经 `stream_invoke` 传入的 `top_p` / `frequency_penalty` / `presence_penalty` / `stop` / `response_format` 等被静默丢弃；由于 Agent 流式链路（`run_stream`）走的就是 `stream_invoke`，这些参数在流式模式下全部失效。现补齐该行，行为与 `invoke` 对齐；
> 2. **`temperature` / `max_tokens` 语义不变** —— 仍为「未传取实例默认、传入则单次覆盖生效」，透传行排除这两个键只是为避免与显式赋值重复；
> 3. **新增测试** —— `tests/llm/test_stream_kwargs.py`（4 项）：`stream_invoke` 额外参数透传、`temperature`/`max_tokens` 解析、`invoke` 回归、`invoke` 与 `stream_invoke` 透传一致性。

> **v1.3.0 更新摘要**（历史，I4 重试/熔断）
> 1. **新增指数退避重试** —— `retry.py` 实现 `RetryConfig` / `run_with_retry` / `run_stream_with_retry`；Full Jitter 算法（`random.uniform(0, min(base*2**attempt, max_delay))`），参考 AWS / tenacity / LangChain 标准做法，避免"惊群"；按 `e.retryable` 属性分流：连接/超时/限流/5xx（`retryable=True`）自动重试，鉴权/配置/参数/结构化输出错误（`retryable=False`）直接透传；默认 `max_retries=2`（最多 3 次尝试）、`base_delay=1.0s`、`max_delay=30.0s`；
> 2. **新增线程安全断路器（Circuit Breaker）** —— `CircuitBreakerRegistry` 按 `(base_url, model)` 维度独立维护状态机（CLOSED → OPEN → HALF_OPEN → CLOSED），默认 `failure_threshold=5`、`recovery_timeout=30s`；连续失败达阈值后进入 OPEN 快速拒绝（不真正发请求），冷却时间过半自动放行一条探测（HALF_OPEN），探测成功回 CLOSED、失败重新 OPEN；使用 `threading.Lock` 保证跨线程安全；
> 3. **流式调用重试边界** —— `run_stream_with_retry` 仅在「发起 HTTP 请求、获取首块之前」的网络错误可重试；一旦开始 yield chunks 给消费方，后续错误直接冒泡不重试，避免文本片段重复/错位；
> 4. **新增两个业务异常** —— `retry_errors.py` 新增 `LLMRetryExhaustedError`（重试耗尽）与 `LLMCircuitOpenError`（断路器打开），均 `retryable=False`，统一导出在 `fast_agents.llm` 顶层；
> 5. **LLM 门面集成** —— `LLM.__init__` 新增可选 `retry: Optional[RetryConfig] = None` 参数，**默认开启**（传 `None` 可完全关闭）；四个核心方法（`invoke` / `stream_invoke` / `invoke_structured` / `embed_query` / `embed_documents`）全部接入重试链路；tracer span 的 `metadata` 新增 `retry_attempts` 字段（实际重试次数），便于 K3 成本评估分析；
> 6. **零新增依赖** —— 仅使用标准库；`retry=None` 路径完全零开销；`RetryConfig.disabled()` 快速关闭；`CircuitBreakerRegistry.status(key)` / `.reset(key)` 暴露监控/重置接口；
> 7. **向后兼容** —— 既有 `invoke(...)` 调用方零代码改动，即可自动获得重试/熔断保护；上层若已自行写重试逻辑，传 `retry=None` 即可关闭本模块内置重试。

> **v1.2.0 更新摘要**（本次同步，E9 结构化输出）
> 1. **新增 `invoke_structured` 方法** —— `OpenAIClient` 与门面 `LLM` 均新增结构化输出入口；参数 `response_format` 接收 pydantic `BaseModel` 子类，内部自动通过 `model.model_json_schema()` 生成 JSON Schema，以 `response_format={"type": "json_schema", ...}` 协议传给端点，再把响应 JSON 解析为 pydantic 实例返回；
> 2. **新增 `LLMStructuredOutputError` 异常** —— JSON 解析失败 / pydantic 校验不通过 / 响应内容为空均抛此异常，`retryable=False`（格式错误重试无意义，应调整 prompt 或 schema）；网络 / API 层错误仍走既有分层异常；
> 3. **支持 `strict` 模式开关** —— `strict=False`（默认）宽容模式，兼容绝大多数 OpenAI 兼容端点；`strict=True` 开启严格约束（要求端点支持 Structured Outputs），不兼容端点会返回 400；
> 4. **门面层带可选 `tracer` 打点** —— 传 `tracer` 时记录 `llm.invoke_structured` span，`metadata` 写 `response_format` 名 + `strict` + token 用量；默认 `None` 零开销；
> 5. **向后兼容** —— 纯新增方法，既有 `invoke` / `stream_invoke` / `embed_query` / `embed_documents` 签名与行为逐字节不变。

> **v1.1.2 更新摘要**（历史，异常体系规范化）
> 1. **新增 `errors.py` 分层异常体系** —— 严格对齐 `AGENTS.md` §5.1「异常统一放 errors.py，不要把底层库原生异常直接抛给上层」规范；按 **连接/超时/限流/服务端/鉴权/配置/模型/参数/API/响应** 拆分 11 个具体子类，统一继承基类 `LLMError`（独立谱系挂在 `Exception` 下，不挂 `ValueError`，避免被无关的 `except ValueError` 误捕）；
> 2. **每个异常带 `retryable: bool` 属性** —— 连接失败 / 超时 / 限流 / 5xx 为 `True`（可重试），鉴权 / 配置 / 参数非法 / 模型不存在为 `False`（不可重试），直接供 I4（重试 / 熔断）按类型判断，上层无需再按字符串猜；
> 3. **OpenAI SDK 原生异常统一映射** —— 新增私有辅助 `_classify_openai_error()`，把 `APIConnectionError / APITimeoutError / RateLimitError / AuthenticationError / PermissionDeniedError / NotFoundError / BadRequestError / APIStatusError / APIError` 一一映射到对应子类；
> 4. **业务层包装与异常链保留** —— Chat 链路最外层包装为 `LLMInvokeError`，Embedding 链路最外层包装为 `LLMEmbeddingError`，统一使用 `raise ... from e` 保留完整原始异常链，上层通过 `__cause__` 可访问底层 SDK 异常与 HTTP 状态码；
> 5. **客户端初始化失败单独分类** —— `OpenAIClient.__init__` 中 SDK 初始化失败（base_url 非法、api_key 格式错误等）现在抛出 `LLMConfigError`，而不是之前的裸 SDK 异常；入参校验类错误（`TypeError` / `ValueError`，如传了非 str 的 text）**仍走原生异常**，不纳入 LLMError 体系（避免吞掉调用方编程错误）；
> 6. **向后兼容** —— 上层原有 `except Exception:` 兜底逻辑仍可正常捕获新异常（因为所有子类都继承自 Exception）；推荐改为 `except LLMError as e:` 并按 `e.retryable` 分流。

> **v1.1.1 更新摘要**（2026-11-18，协议合规修复）
> 1. **`extra_body` 由硬编码改为可配置** —— `OpenAIClient` 与门面 `LLM` 均新增可选构造参数 `extra_body: Optional[dict] = None`；`invoke` / `stream_invoke` 原先无条件硬编码的 `extra_body={"enable_thinking": False}` 已移除，改为「实例级默认 + 单次调用覆盖」两级合并（新增私有辅助 `OpenAIClient._resolve_extra_body()`），**两者都为空时请求体完全不携带 `extra_body`**；
> 2. **原因** —— `enable_thinking` 是阿里云百炼（DashScope）/ Qwen 系私有参数，`extra_body` 会被 SDK 原样合进请求体；作为「基于 OpenAI 原生 API」的框架，默认路径携带该字段在对接 `api.openai.com` 或严格校验未知字段的兼容端点时会返回 400（`Unrecognized request argument supplied: enable_thinking`）；
> 3. **顺带修复 `stream_invoke` 的遗留 bug** —— 原先 `max_tokens` 硬编码为 `self.max_tokens`，无法像 `invoke` 那样单次调用覆盖，现已统一支持 `kwargs` 覆盖；
> 4. **向后兼容** —— 需要关闭思考模式的用户显式传 `LLM(extra_body={"enable_thinking": False})` 即可保留原行为；不传时上层代码零改动，仅请求体更干净。

> **v1.1.0 更新摘要**（历史）
> 1. **新增可选 `tracer` 打点** —— `LLM` 门面新增 `tracer: Optional[TraceLogger] = None` 构造参数，`invoke` / `stream_invoke` / `embed_query` / `embed_documents` 四个方法各新增可选 `tracer` 形参；非 `None` 时分别记录 `llm.invoke` / `llm.stream_invoke` / `llm.embed_query` / `llm.embed_documents` 四条 Span（`SpanKind.LLM`），默认 `None` 时行为与 v1.0.0 逐字节一致；
> 2. **LLM span 写入 token / model 元数据** —— 底层 `OpenAIClient` 新增可选关键字 `usage_sink`（不改返回值语义），门面经 `_merge_usage_metadata()` 把 `usage` 明细与 `prompt_tokens` / `completion_tokens` / `total_tokens` 平铺写入 span 的 `metadata`（`stream_invoke` 无 token）；配套 `_dump_usage()` / `_merge_usage()`，`embed_documents` 多批次累加；
> 3. **口径修正** —— 运行环境收敛为 Windows 平台（对齐 `AGENTS.md` R11）；补记底层默认 `temperature=0.2` / `max_tokens=1024`。

> **v1.0.0 更新摘要**（历史首版）
> 1. **新增 `embed_query` / `embed_documents`** —— 补齐语义检索前置能力，命名对齐 LangChain；所有新代码通过 `LLM` 实例直接调用，`import` 路径不变（仍然是 `from fast_agents.llm import LLM`），无破坏性变更；
> 2. **新增 `embedding_model` 配置** —— chat 模型与 embedding 模型独立，通过构造参数 `embedding_model=` 或环境变量 `LLM_EMBED_MODEL` 指定，默认 `text-embedding-3-small`；
> 3. **`embed_documents` 自动按 2048 分片并发** —— 调用方无需关心 OpenAI API 单批上限，返回顺序与输入严格一致。

---

## 二、先想清楚：LLM 解决什么问题

LLM 模块是整个 fast_agents 的「大脑底座」——上层所有模块（agents / memory / tools / context / skills / mcp）都要通过它把文本喂给模型、再把模型输出拿回来。它要回答三个问题：

1. **对话**：怎么把一组 message 发给模型，拿到回复（含 tool_calls），流式/非流式都要支持；
2. **结构化输出**：怎么让模型按指定 schema 返回 JSON，并自动解析为 pydantic 实例（E9）；
3. **向量化**：怎么把文本转成稠密向量（embedding），为下游的语义检索（memory 的语义记忆、tools 的 RAG）打地基。

### 2.1 能力分层

| 能力 | 输入 | 输出 | 典型调用方 | 本框架对应方法 |
|---|---|---|---|---|
| 非流式对话 | `list[message]` + 可选 tools | 完整 `Message`（含 tool_calls） | Agent 单轮收尾、工具挑选后总结 | `LLM.invoke()` |
| 流式对话 | `list[message]` + 可选 tools | `Iterator[str]` 文本片段流 | 实时对话交互、CLI demo | `LLM.stream_invoke()` |
| **结构化输出** | `list[message]` + pydantic 模型 | pydantic 实例 | Agent 计划产出（E1 前置）、信息抽取、表单解析 | `LLM.invoke_structured()` |
| 查询向量 | 单条 `str`（用户问题） | 一维 `list[float]` | 语义检索 query 侧 | `LLM.embed_query()` |
| 文档向量 | 批量 `list[str]`（文档块） | 二维 `list[list[float]]` | 知识库入库、语义记忆写入 | `LLM.embed_documents()` |

> v1.4.0（I2）起，上述五类能力各有完整的异步对等版本：`ainvoke` / `astream_invoke` / `ainvoke_structured` / `aembed_query` / `aembed_documents`，参数语义与同步版一致，详见 §6.2 / §7.7 / §8.10 / §9.1。

### 2.2 关键设计原则

1. **零重型依赖**：不引入 langchain / litellm / sentence-transformers，只依赖 `openai` SDK（仓库已有）；第三方兼容端点（DeepSeek / Qwen / 豆包 / DashScope 等）通过 `base_url` + `api_key` 走 OpenAI 兼容协议；
2. **门面-客户端两层**：`LLM` 是给上层用的薄门面（只负责读环境变量、选 provider、方法代理），真正的 HTTP 调用在 `OpenAIClient` 里；后续加 Anthropic / Gemini 直连，只需新增 client 文件、不动门面签名；
3. **显式区分 query / document embedding**：方法名直接锁死语义，避免「单条批量共用一个方法、调用方自己想清楚语义」带来的混淆；对接 BGE / E5 / DashScope 等需要 query/document 不同前缀或 `input_type` 的模型时，内部实现可加分支，对外接口不用改；
4. **环境变量优先、构造参数覆盖**：`LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_EMBED_MODEL` 四个环境变量是默认配置来源，构造函数传参可覆盖；
5. **Chat 模型与 Embedding 模型解耦**：两者独立配置——很多团队用 DeepSeek-V3 做对话、用 `text-embedding-3-small` 或 DashScope 做向量化，不能绑死同一个模型名；
6. **错误分层可分流**：所有对外方法统一抛出 `fast_agents.llm.errors` 下的分层异常（基类 `LLMError`），不再抛裸 `Exception`；每个子类带 `retryable` 属性，上层（Agent、tools、I4 重试/熔断）可直接按类型 + 属性分流：连接/超时/限流/5xx 可重试，鉴权/配置/参数非法/模型不存在直接上报用户，不用再按字符串猜；原始 SDK 异常通过 `__cause__` 保留完整链路。

---

## 三、主流框架 LLM 实现对比

### 3.1 对比总表

| 框架 | 实现方式 | 依赖 | 优点 | 缺点 |
|---|---|---|---|---|
| LangChain | `BaseChatModel` / `Embeddings` 抽象基类 + 数十个 provider 实现，方法 `invoke` / `ainvoke` / `embed_query` / `embed_documents` | 重（含 langchain-core + provider 包） | 抽象统一、provider 全、生态最丰富 | 抽象层过厚、调试链路长、依赖地狱、升级易破 |
| LlamaIndex | `LLM` / `BaseEmbedding` 独立抽象，和索引/检索耦合 | 中 | RAG 场景深度优化、embedding 抽象细 | 对话侧偏弱、和 RAG 强绑定、做通用 Agent 不直观 |
| LiteLLM | 统一所有 provider 到 OpenAI 格式，`completion()` / `embedding()` 一个函数走天下 | 中（litellm 单包） | provider 最全、自动兜底重试/限流 | 黑盒过重、把 openai SDK 再包一层、出问题不好查 |
| OpenAI 官方 SDK | 直接 `client.chat.completions.create` / `client.embeddings.create` | 轻（仅 openai） | 最薄、官方维护、错误信息最原始 | chat 和 embeddings 两套独立 API、无门面、无分片/重试 |
| **fast_agents（本模块）** | 门面 `LLM` + 同步/异步双客户端（分别继承 SDK `OpenAI` / `AsyncOpenAI`），同步方法 `invoke` / `stream_invoke` / `invoke_structured` / `embed_query` / `embed_documents` + v1.4.0 异步对等 `a*` 五方法 | **轻（仅 openai；Anthropic/Gemini SDK 为可选依赖）** | 门面薄、零额外默认依赖、query/document 显式区分、自动分片并发、环境变量开箱即用、可选 `tracer` 打点 + token 用量元数据、结构化输出（E9）、重试/熔断（v1.3.0）、全量异步（v1.4.0）、Anthropic/Gemini 原生适配（v1.5.0） | 无本地 embedding、无速率限制/排队、无响应缓存 |

### 3.2 关键差异

| 维度 | 相同点（与主流一致） | 本模块刻意简化 | 目前缺什么（诚实告知） |
|---|---|---|---|
| 对话方法命名 | `invoke` 返回完整响应；v1.4.0 补 `ainvoke`（与 LangChain 异步命名一致） | 不做 `batch` 对象抽象；流式逐段 yield `StreamChunk`（SSE 事件抽象归 Agent 层 E7） | 工具调用严格模式、流式结构化输出 |
| 结构化输出 | `invoke_structured` + pydantic 模型 → 实例（取 OpenAI Agents SDK `output_type` / LangChain `with_structured_output` 思路） | 不做 forced tool-call 兜底路线（待端点实测后定） | 强制工具调用兜底、流式结构化 |
| Embedding 命名 | `embed_query` / `embed_documents`（取 LangChain），v1.4.0 补异步对等 `aembed_*` | 不抽象 `Embeddings` 基类、不做本地模型适配器 | 本地 embedding（sentence-transformers）、embedding 缓存 |
| Provider 支持 | OpenAI 协议全兼容（任意 base_url 中转）；v1.5.0 起 Anthropic / Gemini 原生 SDK 适配（方向同 LiteLLM） | 第三方 SDK 全部是可选依赖，未安装不可用、不强制捆绑 | 多 provider 自动降级 |
| 工程能力 | 超时可配、异常可 catch、**可选 `tracer` 打点 + token 用量元数据**；v1.3.0 内置指数退避重试 + 熔断；v1.4.0 异步方法共用同一套重试配置与断路器 | 不做速率限制/排队 | 限流、响应缓存 |

**一句话收口**：本实现 = **OpenAI 官方 SDK 的原生调用** + **LangChain 的命名约定（`embed_query` / `embed_documents`）** + **门面薄封装（环境变量 + provider 分发 + 分片并发）**，刻意不引入重型抽象。

---

## 四、模块架构

### 4.1 目录结构

```
fast_agents/llm/                ← 单数目录，对齐 tools/ memory/ skills/ mcp/
├── __init__.py                 模块出口，统一导出 LLM / errors 分层异常 / retry 相关类 / StreamChunk
├── llm.py                      门面类 LLM：读环境变量、provider 分发、透传方法、可选 tracer 打点、默认接入 I4 重试/熔断；
│                                 v1.4.0（I2）新增 5 个异步方法（a* 系列），异步 client 用 asyncio.Lock 协程安全懒加载
├── models.py                   StreamChunk 流式输出契约（dataclass，text / tool_call / end 三态，同步/异步流式共用，E7）
├── openai_client.py            OpenAIClient（继承官方 OpenAI）：实际同步 HTTP 调用，含 chat / structured / embedding 接口
├── async_openai_client.py      AsyncOpenAIClient（继承官方 AsyncOpenAI）：I2 核心（v1.4.0 新增），5 个方法
│                                 ainvoke / astream_invoke / ainvoke_structured / aembed_query / aembed_documents，
│                                 aembed_documents 用 asyncio.Semaphore(4) 协程级并发，复用同步客户端工具函数零代码重复
├── anthropic_client.py / async_anthropic_client.py   I3（v1.5.0）：Anthropic Claude 原生 API 客户端（同步/异步成对，SDK 可选依赖）
├── gemini_client.py / async_gemini_client.py         I3（v1.5.0）：Google Gemini 原生 API 客户端（同步/异步成对，SDK 可选依赖）
├── errors.py                   LLMError 基类 + 12 个具体子类（v1.2.0 新增 LLMStructuredOutputError），对齐 AGENTS.md §5.1 异常规范
├── retry.py                    I4 核心：RetryConfig（指数退避 Full Jitter）+ CircuitBreakerRegistry（线程安全断路器）
│                                 + run_with_retry / run_stream_with_retry 同步执行器（v1.3.0）
│                                 + arun_with_retry / arun_stream_with_retry 异步执行器（v1.4.0，与同步共用 RetryConfig/断路器）
└── retry_errors.py             I4 专属异常：LLMRetryExhaustedError（重试耗尽）+ LLMCircuitOpenError（断路器打开）（v1.3.0 新增）
```

### 4.2 分层

```
┌──────────────────────────────────────────────┐
│  上层调用方（agents / memory / tools / ...）  │
│  只依赖 fast_agents.llm.LLM 这一个入口         │
└──────────────────┬───────────────────────────┘
                   │  同步：invoke / stream_invoke /
                   │        invoke_structured / embed_query / embed_documents
                   │  异步（v1.4.0）：ainvoke / astream_invoke /
                   │        ainvoke_structured / aembed_query / aembed_documents
                   ▼
┌──────────────────────────────────────────────┐
│  门面层：llm.py · LLM                         │
│  - 读取环境变量（LLM_MODEL_ID / LLM_API_KEY /  │
│    LLM_BASE_URL / LLM_EMBED_MODEL / timeout） │
│  - 按 provider 分发同步/异步 client（v1.5.0）  │
│  - 异步 client 懒加载（asyncio.Lock，协程安全） │
│  - tracer 打点 + 重试/熔断横切逻辑             │
└──────────┬────────────────┬───────────────────┘
           │ 同步路径         │ 异步路径（首次 await 时建连）
           ▼                ▼
┌────────────────────┐  ┌──────────────────────────┐
│ openai_client.py   │  │ async_openai_client.py   │
│ OpenAIClient       │  │ AsyncOpenAIClient（v1.4.0）│
│ - 继承 openai.OpenAI│  │ - 继承 openai.AsyncOpenAI │
│   （httpx.Client）  │  │   （httpx.AsyncClient）   │
│ - invoke / embed_* │  │ - ainvoke / aembed_*     │
│ - 线程池并发分片    │  │ - Semaphore(4) 协程并发   │
└────────────────────┘  └──────────────────────────┘
   （anthropic / gemini 的同步与异步客户端同构存在，v1.5.0，按 provider 路由）
```

依赖方向严格自上而下：上层 → 门面 → 具体 client；client 不反向依赖门面，不同 client 之间无相互依赖。

### 4.3 调用链路

**对话（非流式）**：
```
Agent → LLM.invoke(messages, tools)
         └── OpenAIClient.invoke()
               └── self.chat.completions.create(model, messages, tools, stream=False)
                     └── 包装成 Message（含 tool_calls）→ 返回
```

**Embedding 文档批量**：
```
MemoryManager / RAGTool → LLM.embed_documents(texts)
                            └── OpenAIClient.embed_documents(texts)
                                  ├── 按 _EMBED_BATCH_SIZE=2048 切片
                                  ├── ThreadPoolExecutor 并发请求每片
                                  │     └── self.embeddings.create(model, input=batch)
                                  ├── 按 resp.data[*].index 排序（对齐输入顺序）
                                  └── 拼接成 list[list[float]] → 返回
```

**异步非流式对话（v1.4.0，I2）**：
```
Agent(async) → await LLM.ainvoke(messages, tools)
                └── _get_async_client()：首次调用时 asyncio.Lock 协程安全懒加载，后续复用同一实例
                      └── arun_with_retry(...)：与同步共用同一套 RetryConfig / 断路器，退避用 asyncio.sleep
                            └── AsyncOpenAIClient.ainvoke()
                                  └── self.chat.completions.create(..., stream=False) → 包装成 Message → 返回
```

箭头旁标注的就是源码里的真实方法名，可直接顺藤摸瓜。

### 4.4 与 ReActAgent 的接入点 / 职责边界

- **接入方式**：`ReActAgent.__init__` 接收 `llm: LLM` 参数并保存为成员；所有对模型的调用（包括思考、tool_calls 解析、滚动摘要、未来的 embedding）都走 `self.llm`，不允许直接 new 一个 OpenAI client；
- **单一数据源约定**：对话历史由 `memory.BufferMemory` 维护，LLM 不持有任何会话状态——它是纯函数式的「输入消息 → 输出消息/向量」，保证可替换、可 mock、可测试；
- **边界清晰**：LLM 自身不做 token 计数（归 context 模块）、不做向量存储（归 memory / RAG 工具）——它只负责「把东西发出去、把结果拿回来」；重试熔断（I4）自 v1.3.0 起下沉到 LLM 层内部（见 §7.5），v1.4.0（I2）起异步方法与同步方法共用同一套重试配置与断路器实例；唯一新增的横切能力是**可选** `tracer`：默认 `None` 时零开销、行为与改动前逐字节一致，传入后才记录 span 与 token 元数据（见 §7.2 / §9.1 / §10.1）。

---

## 五、核心机制：Chat 与 Embedding 双能力分层

LLM 模块最核心的设计取舍是：**为什么 Chat 和 Embedding 不拆成两个类，而是同一个 LLM 门面下的两组方法？**

| 维度 | 方案A：拆两个独立类（ChatLLM + Embedder） | 方案B：同一个 LLM 门面下的两组方法（本模块选择） |
|---|---|---|
| 概念清晰 | ✅ 两类能力完全分离，单一职责 | ⚠️ 一个类四种方法，需要文档说明 |
| 配置共享 | ❌ api_key / base_url / timeout 要配两遍 | ✅ 复用同一个 client、同一份鉴权 |
| 上层使用 | ❌ Agent 里要持两个成员：`self.chat_llm` + `self.embedder` | ✅ Agent 只持一个 `self.llm`，调哪个方法调什么 |
| 模型解耦 | ✅ 天然支持不同模型 | ✅ 通过 `model` / `embedding_model` 两个参数实现同样的解耦（不绑死） |
| 后续扩展本地模型 | ✅ 可以有 LocalEmbedder 独立实现 | ✅ 同样可以：`LocalEmbeddingClient` 实现相同 `embed_*` 方法，LLM 门面在 provider 处分发 |
| 项目一致性 | ❌ 目前仓库里其他模块都是「一个模块一个门面类」风格，拆两类会显得突兀 | ✅ 和 `ToolRegistry` / `MemoryManager` / `SkillRegistry` 统一为「单门面」风格 |

**结论（倾向方案B）**：在 fast_agents 当前体量下，**一个 LLM 门面 + 两组方法** 是最贴合「零重型依赖、每个模块独立目录」风格的选择。真正需要拆的时候，再抽 `BaseEmbeddings` 基类也不迟——那时 `embed_query` / `embed_documents` 的方法签名已经稳定，上层调用代码一行都不用改。

**第二个取舍：为什么分 `embed_query` 和 `embed_documents`，而不是一个 `embed(texts)` 靠参数单/批量自动判断？**

- 对 OpenAI 自己的 embedding API 来说，单条和批量确实是同一个接口，没必要分方法；
- 但对 **BGE / E5 / DashScope text-embedding-v3** 这类模型，query 和 document 需要不同的前缀（BGE 要加 `"为这个句子生成表示以用于检索相关文章："`；DashScope v3 要传 `input_type="query"` / `"document"`），如果接口不分，调用方很容易在入库时误用 query 模式、在检索时误用 document 模式，**错误是静默的**——效果差但不报错；
- LangChain 选择分成两个方法，就是把这种「工程语义差异」写进类型签名，让调用方在写代码时就被提醒、让后续接非对称模型时接口不用变；
- 本框架直接采用 LangChain 的命名。

---

## 六、核心数据模型

LLM 模块本身没有 pydantic 模型（输入输出都是原生 Python 类型 + `fast_agents.messages.Message`），但有几个关键的类型约定需要明确：

### 6.1 LLM 门面类构造参数

| 参数 | 类型 | 默认值 | 说明 | 必填 |
|---|---|---|---|---|
| `model` | `Optional[str]` | `$LLM_MODEL_ID` | Chat 模型名，如 `gpt-4o-mini` / `deepseek-chat` | 否（环境变量兜底） |
| `api_key` | `Optional[str]` | `$LLM_API_KEY` | API 密钥 | 否（环境变量兜底） |
| `base_url` | `Optional[str]` | `$LLM_BASE_URL` | API 端点，兼容第三方中转（如 `https://api.deepseek.com`） | 否（默认 OpenAI 官方） |
| `provider` | `str` | `"openai"` | 客户端分发：`openai` / `anthropic` / `gemini`（大小写不敏感，v1.5.0 已落地）；同步/异步方法自动对齐同一 provider；非法值抛 `LLMConfigError` | 否 |
| `timeout` | `Optional[int]` | `180`（client 层默认） | 请求超时秒数 | 否 |
| `embedding_model` | `Optional[str]` | `$LLM_EMBED_MODEL`，否则 `"text-embedding-3-small"` | Embedding 模型名，与 chat 模型独立 | 否 |
| `extra_body` | `Optional[dict]` | `None` | **v1.1.1 新增**：provider 私有请求体扩展字段（如 DashScope / Qwen 的 `enable_thinking`）；默认 `None` 时**不发送**该字段，保证对原生 OpenAI 端点严格兼容 | 否 |
| `tracer` | `Optional[TraceLogger]` | `None` | **v1.1.0 新增**：可选打点器；非 `None` 时同步/异步各方法每调用记一条 LLM span（异步 span metadata 额外标 `async=True`，v1.4.0），默认 `None` 零开销 | 否 |
| `retry` | `Optional[RetryConfig]` | `None` | **v1.3.0 新增（I4）**：重试/熔断配置；不传时自动启用默认保守配置（max_retries=2）；传 `RetryConfig(...)` 自定义、`RetryConfig.disabled()` 关闭；v1.4.0 起异步方法共用同一实例 | 否 |

### 6.2 方法签名（核心契约）

```python
class LLM:
    # ---- Chat（同步）----
    def invoke(self, messages: list[dict[str, Any]], tools=None, tracer=None, **kwargs) -> Message: ...
    def stream_invoke(self, messages: list[dict[str, Any]], tools=None, tracer=None, **kwargs) -> Iterator[StreamChunk]: ...
    # ---- Structured Output（v1.2.0 新增）----
    def invoke_structured(self, messages: list[dict[str, Any]], response_format: type, *, strict=False, tracer=None, **kwargs) -> BaseModel: ...
    # ---- Embeddings（同步）----
    def embed_query(self, text: str, tracer=None, **kwargs) -> list[float]: ...
    def embed_documents(self, texts: list[str], tracer=None, **kwargs) -> list[list[float]]: ...

    # ---- 异步对等版本（v1.4.0 新增，I2）：命名 = 同步方法加 a 前缀，参数语义与同步版逐一对齐 ----
    async def ainvoke(self, messages: list[dict[str, Any]], tools=None, tracer=None, **kwargs) -> Message: ...
    async def astream_invoke(self, messages: list[dict[str, Any]], tools=None, tracer=None, **kwargs) -> AsyncIterator[StreamChunk]: ...
    async def ainvoke_structured(self, messages: list[dict[str, Any]], response_format: type, *, strict=False, tracer=None, **kwargs) -> BaseModel: ...
    async def aembed_query(self, text: str, tracer=None, **kwargs) -> list[float]: ...
    async def aembed_documents(self, texts: list[str], tracer=None, **kwargs) -> list[list[float]]: ...
```

**命名约定锁死**：
- 对话方法用 `invoke` / `stream_invoke`——对齐工具模块的 `ToolBase.invoke()` 命名风格（动词 `invoke` 是本框架「一次调用」的统一动词）；
- Embedding 方法用 `embed_query` / `embed_documents`——对齐 LangChain，**不使用** `embed` / `encode` / `get_embedding` 等模糊命名；
- 返回原生 `list[float]`——不引入 numpy / torch 依赖，下游要算余弦相似度自己用标准库 `math` 即可（参考 `memory/scoring.py` 全标准库实现的风格）；
- 所有同步/异步方法统一多一个**可选** `tracer: Optional[TraceLogger] = None`（v1.1.0）——默认 `None` 时零开销、行为不变；传入后记一条 `llm.*` span 并在 `metadata` 写 token 用量；
- 异步方法用 `a` 前缀（v1.4.0，I2）——`ainvoke` / `astream_invoke` / `ainvoke_structured` / `aembed_query` / `aembed_documents`，对齐 LangChain「同步名 + a 前缀」的异步命名惯例；参数、返回值类型与同步版完全一致，差别仅在需要 `await` / `async for`。

---

## 七、逐个文件讲解（源码导读）

### 7.1 `__init__.py` —— 模块出口

```python
"""LLM模块。

对外暴露：
- LLM：统一门面类，上层唯一入口；
- errors：LLM 分层异常体系（含基类与所有可重试/不可重试子类），
  上层做重试 / 熔断 / 用户提示分流时直接从此处导入。
"""
from .llm import LLM
from . import errors
from .errors import (LLMError, LLMConnectionError, LLMTimeoutError,
                     LLMRateLimitError, LLMServerError, LLMAuthenticationError,
                     LLMConfigError, LLMModelNotFoundError, LLMBadRequestError,
                     LLMApiError, LLMResponseError, LLMInvokeError, LLMEmbeddingError)

__all__ = ["LLM", "errors", "LLMError", ...]
```

**为什么这么写**：和仓库其他模块保持一致（`database/__init__.py` 导出 `DatabaseError` 等、`mcp/__init__.py` 导出 `MCPError`），对外永远只暴露一个门面类 `LLM` 加一套异常类型；内部 client 类不进 `__all__`——上层不应该直接 `from fast_agents.llm import OpenAIClient`。上层做错误处理时统一从 `fast_agents.llm` 或 `fast_agents.llm.errors` 导入异常，无需关心底层是哪个 client 抛出的。

### 7.2 `llm.py` —— LLM 门面

```python
class LLM:
    def __init__(self, model=None, api_key=None, base_url=None,
                 provider="openai", timeout=None, embedding_model=None,
                 extra_body=None,                    # v1.1.1：provider 私有扩展，默认 None 不发送
                 tracer=None):                       # v1.1.0：可选打点，默认 None 零开销
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("LLM_EMBED_MODEL")
        self.extra_body = extra_body                 # None 时不发送任何扩展字段
        self.tracer = tracer                         # None 时行为与改动前逐字节一致
        ...
        self._client = self._create_client()

    def embed_query(self, text: str, tracer=None, **kwargs) -> list[float]:
        effective = tracer or self.tracer
        if effective is None:                        # 零开销路径：直接透传
            return self._client.embed_query(text, **kwargs)
        ...                                          # 否则记 llm.embed_query span

    def embed_documents(self, texts: list[str], tracer=None, **kwargs) -> list[list[float]]:
        effective = tracer or self.tracer
        if effective is None:
            return self._client.embed_documents(texts, **kwargs)
        ...                                          # 否则记 llm.embed_documents span
```

**关键实现要点**：
- **环境变量 vs 构造参数**：用 `or` 实现「构造参数优先，环境变量兜底」。注意 `or` 对空字符串同样 fallback，这是刻意的——如果用户显式传 `model=""`，视为没传、走环境变量；
- **provider 分发位**：`_create_client()` 虽然目前所有分支都返回 `OpenAIClient`，但 `if self.provider == "openai"` / `else` 的分支结构先搭好，将来加 Anthropic client 时只需加一个 `elif self.provider == "anthropic"` 分支，门面类的构造签名和方法签名都不动；
- **方法透传**：所有对外方法都是「薄代理」——拿到 kwargs 直接透给 client，不在门面做业务逻辑。这样的好处是：门面永远不需要因为底层参数调整而改；
- **可选打点（tracer）**：用 `tracer or self.tracer` 取有效值，都为 `None` 时**直接把请求透传给 client、不构造任何 span**（零开销）；只有传入后才用 `with effective_tracer.span(...)`（或 `stream_invoke` 的 `start_span` / `end_span`）记录 LLM span，并在正常/异常分支回填 `output` 与 token 元数据。默认 `None` 保证对既有链路零侵入。

**要自己改时最该看的地方**：想加超时重试、响应缓存、trace 打点，都应该在这一层做（装饰器或包裹 client 调用），不要改 `openai_client.py`——因为 client 将来可能有多个（Anthropic、本地），横切能力应该加在门面或更外层。

### 7.3 `openai_client.py` —— OpenAI 协议实际实现

这是最核心的文件，`OpenAIClient` 直接继承官方 `openai.OpenAI`，复用它的 HTTP 连接池、鉴权、重试等底层能力，只在上层封装我们自己的方法签名。

#### 7.3.1 构造与模型分离

```python
self.model = model or os.getenv("LLM_MODEL_ID")
self.embedding_model = (
    embedding_model
    or os.getenv("LLM_EMBED_MODEL")
    or "text-embedding-3-small"
)
```

**为什么要分开两个模型字段**：默认 embedding 模型用 `text-embedding-3-small`（OpenAI 当前性价比最高、1536 维）；而 chat 的 `self.model` **没有硬编码默认值**，强制用户通过环境变量或构造参数指定——因为 chat 模型的选择差异太大（gpt-4o / gpt-4o-mini / deepseek-chat / qwen-plus…），硬选一个默认值反而容易误导。

#### 7.3.2 chat 方法

`invoke` / `stream_invoke` 的底层调用（含 `usage_sink` 回填与**可配置**的 `extra_body`）：

```python
chat_model = kwargs.pop("model", self.model)
extra_body = self._resolve_extra_body(kwargs)                  # 实例默认 + 单次覆盖
response = self.chat.completions.create(
    model=chat_model, messages=messages, tools=tools,
    temperature=kwargs.get('temperature', self.temperature),   # 默认 0.2
    max_tokens=kwargs.get('max_tokens', self.max_tokens),       # 默认 1024
    stream=False,
    **({"extra_body": extra_body} if extra_body else {}),       # 为空则完全不带该字段
)
if usage_sink is not None:                                     # 打点用，可选
    usage_sink.update(_dump_usage(response.usage))
```

**为什么允许通过 kwargs 覆盖 model**：上层偶尔需要对单次调用切换模型（比如摘要用小模型、复杂推理用大模型），不应该为此 new 一个新 LLM 实例。`kwargs.pop` 保证这个参数不会被重复传给底层 SDK（SDK 签名里 `model` 是位置/关键字参数，重复传会报错）。

**为什么默认 `temperature=0.2` / `max_tokens=1024`**：`OpenAIClient.__init__` 从构造 kwargs 读取这两个默认值，偏向「确定性 + 短回复」，适合 Agent 的工具挑选与摘要；需要更长或更发散时，`invoke` 可按次传 `temperature=` / `max_tokens=` 覆盖（`stream_invoke` 仅 `temperature` 可覆盖、`max_tokens` 固定取实例值）。

**为什么 `extra_body` 由硬编码改为可配置（v1.1.1）**：部分国内推理模型（Qwen3 等）默认开启思考模式，会在正文前吐 `reasoning_content`、拖慢首字延迟，需要 `extra_body={"enable_thinking": False}` 来关闭——但 `enable_thinking` 是**阿里云百炼（DashScope）/ Qwen 系的私有参数**，`extra_body` 会被 SDK 原样合进请求体，对接 `api.openai.com` 或严格校验未知字段的兼容端点时会直接 400（`Unrecognized request argument supplied: enable_thinking`）。因此改为**默认不发送**：不传 `extra_body` 时请求体保持纯净；需要关闭思考模式时显式传 `LLM(extra_body={"enable_thinking": False})`（或单次调用 `invoke(..., extra_body={...})`）。`stream_invoke` 仍兼容处理 `reasoning_content`（打印到控制台但不进入 yield），防止端点忽略该开关时正文流被思考片段污染。

#### 7.3.3 embed_query

```python
def embed_query(self, text: str, **kwargs) -> list[float]:
    if not isinstance(text, str):
        raise TypeError(...)
    if not text.strip():
        raise ValueError("embed_query 收到空字符串，无法生成 embedding")
    embed_model = kwargs.pop("model", self.embedding_model)
    resp = self.embeddings.create(model=embed_model, input=[text], **kwargs)
    return list(resp.data[0].embedding)
```

**为什么做入参校验**：OpenAI SDK 对空串会返回 400，但错误信息是 HTTP 层的，定位慢；在入口先给清晰的 TypeError/ValueError，调用方调试更快。

**为什么 `input=[text]` 而不是 `input=text`**：OpenAI API 既接受 str 也接受 list[str]，但统一包成 list 让返回结构稳定（始终是 `data[0]`），不用处理两种返回形态。

#### 7.3.4 embed_documents（分片 + 并发）

```python
_EMBED_BATCH_SIZE = 2048

def _one_batch(batch: list[str]):
    resp = self.embeddings.create(model=embed_model, input=batch, **kwargs)
    ordered = sorted(resp.data, key=lambda d: d.index)
    vectors = [list(d.embedding) for d in ordered]
    return vectors, _dump_usage(getattr(resp, "usage", None))   # 打点用 usage（可选）

batches = [cleaned[i : i + _EMBED_BATCH_SIZE]
           for i in range(0, len(cleaned), _EMBED_BATCH_SIZE)]
if len(batches) == 1:
    return _one_batch(batches[0])
with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
    for part in pool.map(_one_batch, batches):
        results.append(part)
```

**为什么分片**：OpenAI embedding API 单次请求 input 上限 2048 条（官方文档约束），一次塞几千段文档会直接 400；调用方（RAG 入库、记忆批量写入）通常有成百上千段文本，不能让上层自己管批次。

**为什么并发**：分批次串行调用 N 次的总耗时 ≈ N × 单次 RTT；线程池并发后总耗时 ≈ max(单次 RTT)。用 `ThreadPoolExecutor` 是因为 `openai` SDK 底层是 httpx，IO 密集，线程池就能拿到并发收益，不需要 asyncio。`max_workers=min(4, len(batches))` 是保守值——避免瞬间发太多请求触发速率限制。

**为什么要 `sorted(resp.data, key=lambda d: d.index)`**：OpenAI API 文档**不保证**返回的 `data` 顺序与输入顺序一致（实际实现通常一致，但契约上不保证）；按 `index` 排序是官方推荐的对齐方式，能避免极端情况下向量错位这种极难排查的 bug。

**空串 / 非法项处理**：清洗规则是 `cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]`——**空串、纯空白串、非字符串项一律替换为 `" "`（单空格）而不是丢弃**——因为 RAG/记忆入库时，输入 texts 通常和一批 chunk 对象一一对应，如果中间丢弃会导致返回向量数量和输入不一致，上层索引错位；单空格的 embedding 是一个合法向量，且和任何语义文本的相似度都极低，不会污染检索结果。

**要自己改时最该看的地方**：接 BGE / E5 / DashScope v3 等非对称模型时，就在 `embed_query` 和 `embed_documents` 内部分别加前缀或 `input_type` 参数，对外签名不变。加重试（指数退避）、加缓存（按 text hash 缓存向量）也在这个文件里做。

#### 7.3.5 异常映射（v1.1.2）

```python
from openai import (APIConnectionError, APITimeoutError, RateLimitError,
                    AuthenticationError, PermissionDeniedError,
                    NotFoundError, BadRequestError, APIStatusError, APIError)
from .errors import (LLMConnectionError, LLMTimeoutError, LLMRateLimitError,
                     LLMServerError, LLMAuthenticationError, LLMModelNotFoundError,
                     LLMBadRequestError, LLMApiError)

def _classify_openai_error(e: Exception) -> LLMError:
    if isinstance(e, APITimeoutError):       return LLMTimeoutError(...)
    if isinstance(e, APIConnectionError):    return LLMConnectionError(...)
    if isinstance(e, RateLimitError):        return LLMRateLimitError(...)
    if isinstance(e, (AuthenticationError, PermissionDeniedError)):
                                             return LLMAuthenticationError(...)
    if isinstance(e, NotFoundError):         return LLMModelNotFoundError(...)
    if isinstance(e, BadRequestError):       return LLMBadRequestError(...)
    if isinstance(e, APIStatusError):
        if 500 <= (e.status_code or 0) < 600: return LLMServerError(...)
        return LLMApiError(...)
    if isinstance(e, APIError):              return LLMApiError(...)
    return LLMApiError(...)
```

**为什么不在 client 里直接 `except Exception as e: raise LLMError(str(e))`**：那样会把所有错误压扁成同一个类型，I4（重试/熔断）无法区分"网络抖动该重试"还是"API Key 错了不该重试"。按 SDK 原生异常类型一一映射，每个子类带 `retryable` 属性，上层一行 `if err.retryable: retry()` 就能分流。

**为什么 Chat / Embedding 最外层再包一层 `LLMInvokeError` / `LLMEmbeddingError`**：`_classify_openai_error` 负责按 HTTP 层原因分类；业务层再按"是在调用哪个方法时发生的"包一层，上层 `except LLMInvokeError` / `except LLMEmbeddingError` 可以分别针对对话链路与向量链路做不同降级（比如对话失败走兜底回复、embedding 失败走关键词检索回退）。两层异常通过 `raise ... from mapped` 串成异常链，`err.__cause__` 可访问底层具体原因，`err.original_error` 可访问最原始的 SDK 异常。

**入参校验为什么不纳入 LLMError 体系**：`TypeError` / `ValueError` 是**调用方编程错误**（传了非 str 的 text、传了 int 当 messages），不是服务端/网络问题；如果把它们包进 LLMError，上层会误以为是可重试/可上报用户的运行时错误，反而掩盖 bug。因此这两类错误原样透传，不被 LLMError 吞掉。

### 7.4 `errors.py` —— 分层异常体系（v1.1.2 新增）

严格对齐 `AGENTS.md` §5.1「异常统一放 errors.py，不要把底层库原生异常直接抛给上层」的规范，参考同仓库 `database/errors.py` / `mcp/errors.py` 的「基类 + 具体子类 + retryable 属性」组织方式。

#### 7.4.1 异常类层级

```
Exception
└── LLMError                              # 基类，retryable=False，带 original_error 属性
    ├── LLMConnectionError                # 连接失败（DNS、拒连、TLS）        [retryable=True]
    ├── LLMTimeoutError                   # 请求超时                          [retryable=True]
    ├── LLMRateLimitError                 # 429 限流、配额耗尽                 [retryable=True]
    ├── LLMServerError                    # 5xx 服务端错误                     [retryable=True]
    ├── LLMAuthenticationError            # 401/403 鉴权失败                   [retryable=False]
    ├── LLMConfigError                    # 配置缺失/非法、SDK 初始化失败       [retryable=False]
    │   └── LLMModelNotFoundError         # 404 模型不存在                     [retryable=False]
    ├── LLMBadRequestError                # 400 请求参数非法（未知字段等）      [retryable=False]
    ├── LLMApiError                       # 其它未分类 API 错误                [retryable=False]
    ├── LLMResponseError                  # 响应格式异常（空向量、缺字段）      [retryable=False]
    ├── LLMInvokeError                    # Chat 链路业务包装                  [retryable=False，见 __cause__]
    ├── LLMStructuredOutputError          # 结构化输出业务包装（v1.2.0 新增）  [retryable=False]
    └── LLMEmbeddingError                 # Embedding 链路业务包装             [retryable=False，见 __cause__]
```

#### 7.4.2 设计要点

- **独立谱系挂在 `Exception` 下**：不挂 `ValueError`。因为 LLM 错误绝大多数是网络/鉴权/服务端问题，语义上不是"值不合法"；挂 `ValueError` 会被业务代码里无处不在的 `except ValueError`（如 `int()`、`datetime.fromisoformat`）误捕，反而埋 bug。
- **`retryable` 作为类属性**：I4（重试/熔断）可直接 `if err.retryable: backoff_retry()`，不用再维护一份"哪些异常该重试"的白名单；业务层包装类（`LLMInvokeError` / `LLMEmbeddingError`）默认 `retryable=False`，但上层可通过 `err.__cause__.retryable` 取到底层具体原因的可重试性。
- **`original_error` 保留原始 SDK 异常**：`__cause__` 是 Python 异常链的隐式链，`original_error` 是显式字段——两者同时存在，便于序列化（把异常打进 trace span 时直接取 `original_error` 更稳定）。
- **配置错误在构造期暴露**：`OpenAIClient.__init__` 里调用 `super().__init__()` 失败（如 `base_url` 格式非法）会被单独包装成 `LLMConfigError`，而不是之前那样裸抛 SDK 异常；这让"启动时就报错、提示检查环境变量"和"运行中网络抖动重试"两类错误在类型上就能区分。

### 7.5 `retry.py` —— I4 指数退避重试 + 断路器（v1.3.0 新增）

零新增依赖（仅用标准库 `time` / `random` / `threading` / `functools`），实现三大组件：

1. **`RetryConfig`**：不可变配置对象，含四个参数：
   - `max_retries`：最大重试次数（不含首次调用），默认 `2`（总尝试 3 次）；
   - `base_delay`：基础退避时间（秒），默认 `1.0`；
   - `max_delay`：退避上限（秒），默认 `30.0`，防止等待过长；
   - `retry_on`：扩展钩子——额外的可重试异常元组；默认仅按 `e.retryable` 判断；
   - `breaker`：自定义断路器注册表，不传则用模块级默认实例。
   - 提供 `backoff(attempt)` Full Jitter 退避算法（参考 AWS 架构博客），返回 `random.uniform(0, min(base*2**attempt, max_delay))`；`should_retry(exc)` 统一判断；类方法 `disabled()` 返回一个「完全关闭重试、断路器阈值极高」的配置，方便排障。

2. **`CircuitBreakerRegistry`**：按 `(base_url, model)` 维度隔离的线程安全断路器，内部状态机：
   - `CLOSED`（正常放行）→ 连续失败达 `failure_threshold` 次 → `OPEN`（快速拒绝，抛 `LLMCircuitOpenError`）；
   - `OPEN` → 冷却 `recovery_timeout` 秒后 → `HALF_OPEN`（放行一条探测请求）；
   - `HALF_OPEN` 探测成功 → `CLOSED`；探测失败 → 立即回 `OPEN` 并重置冷却计时；
   - 每个 key 独立一把 `threading.Lock`，注册表级另一把锁保证创建新 key 线程安全。
   - 提供 `status(key)` 返回当前状态（含 `seconds_remaining`）、`reset(key=None)` 重置状态。

3. **执行器**：
   - `run_with_retry(fn, retry, circuit_key, args, kwargs)`：同步版本。`retry=None` 走零开销路径直接调用；断路器检查 → 循环调用 → 可重试异常按 Full Jitter 退避 sleep 后 continue → 非可重试异常直接透传 → 耗尽抛 `LLMRetryExhaustedError`；成功记 `on_success`、失败记 `on_failure`。返回 `(result, attempts_used)` 元组。
   - `run_stream_with_retry(...)`：流式版本。**只在生成器创建/首块获取阶段可重试**——因为流一旦开始 yield 给消费方，重试会导致文本片段重复错位；首块成功后剩余块迭代中再出错，断路器记一次失败但不重试。返回 `(iterator, attempts_used)` 元组。
   - `arun_with_retry(...)` / `arun_stream_with_retry(...)`（**v1.4.0 新增，I2**）：异步执行器，与同步版共用同一套 `RetryConfig` / `CircuitBreakerRegistry`（单事件循环内协程并发，`threading.Lock` 临界区无挂起点、不会死锁，无需另做 asyncio.Lock 版断路器）；退避用 `asyncio.sleep` 不阻塞事件循环；支持 fn 返回 coroutine 或直接返回 async generator 两种写法；流式重试边界与同步版一致（仅首块前可重试）。

关键设计决策：
- **max_retries=0 不包 LLMRetryExhaustedError**：只有真正做过重试才抛「重试耗尽」；关闭重试时直接透传原异常，语义纯粹。
- **不可重试异常也计数断路器**：鉴权风暴等永久错误也应触发熔断（避免反复撞墙），因此 `on_failure` 对所有异常（除开入参错误 TypeError/ValueError）都计数。
- **失败安全**：retry/breaker 自身异常不会阻塞主流程（实现为纯函数 + Lock，没有外部副作用点）。

### 7.6 `retry_errors.py` —— I4 专属异常（v1.3.0 新增）

两个异常子类，均 `retryable=False`（因为重试已经尽力/被判定为危险）：
- `LLMRetryExhaustedError`：重试达到 `max_retries` 上限仍未成功时抛出，`original_error` 保留最后一次失败的异常；
- `LLMCircuitOpenError`：断路器 OPEN、请求被快速拒绝时抛出。

### 7.7 `async_openai_client.py` —— I2 异步底座（v1.4.0 新增）

`AsyncOpenAIClient` 继承官方 `openai.AsyncOpenAI`，与同步 `OpenAIClient` 完全对等，实现 5 个异步方法：

```python
class AsyncOpenAIClient(AsyncOpenAI):
    async def ainvoke(self, messages, tools=None, *, usage_sink=None, **kwargs) -> Message: ...
    async def astream_invoke(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]: ...
    async def ainvoke_structured(self, messages, response_format, *, strict=False, **kwargs) -> BaseModel: ...
    async def aembed_query(self, text, *, usage_sink=None, **kwargs) -> list[float]: ...
    async def aembed_documents(self, texts, *, usage_sink=None, **kwargs) -> list[list[float]]: ...
```

**关键实现要点**：
- **零代码重复**：复用 `openai_client.py` 的模块级工具函数 `_dump_usage` / `_merge_usage` / `_classify_openai_error` / `_EMBED_BATCH_SIZE`，异常映射与 token 用量回填口径两个文件同源；
- **`aembed_documents` 分片并发**：用 `asyncio.Semaphore(4)` 替代线程池，与同步版 `ThreadPoolExecutor(max_workers=4)` 并发度对齐，同样按 `data[*].index` 排序对齐输入顺序，空串清洗规则与同步版一致；
- **异步重试**：由门面层的 `arun_with_retry` / `arun_stream_with_retry`（见 §7.5）包裹，与同步版共用 `RetryConfig` / 断路器，退避用 `asyncio.sleep`；流式「首块前可重试、开始 yield 后不重试」语义与同步一致；
- **门面懒加载**：`LLM._get_async_client()` 用 `asyncio.Lock` 保证首次 `await` 调用时才初始化对应 provider 的异步 client，后续复用同一实例——只用同步方法的用户不付出异步 client 的任何初始化开销；
- **打点**：异步 span 命名对应 `llm.ainvoke` / `llm.astream_invoke` / `llm.ainvoke_structured` / `llm.aembed_query` / `llm.aembed_documents`，`metadata` 额外标记 `async=True`（与同步记录在同一模型下易于区分过滤），`retry_attempts` 与 token 用量口径同同步版。

**设计取舍（对齐主流框架做法）**：`openai.OpenAI` 与 `openai.AsyncOpenAI` 是两个独立类（底层 `httpx.Client` / `httpx.AsyncClient` 完全分离），LangChain / LiteLLM / OpenAI 官方 SDK 同样是同步异步两套并行实现而非单类组合模式，本模块沿用该形态，不引入额外抽象层（符合 `AGENTS.md` R3 抽象克制）。测试见 `tests/llm/test_async.py`（13 项离线全绿，`tests/_runner.py` 自动识别 `async def test_*` 并用 `asyncio.run` 执行），示例见 `examples/llm/async_demo.py`。

---

## 八、快速上手

### 8.1 环境准备（Windows PowerShell）

在项目根目录新建 `.env` 文件：

```env
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini
LLM_EMBED_MODEL=text-embedding-3-small
```

> 如果用第三方中转（例如 DeepSeek / 阿里通义 / 本地 vLLM），改 `LLM_BASE_URL` 即可，接口都兼容 OpenAI 协议。

验证安装：

```powershell
cd D:\path\to\fast_agents
python -c "from fast_agents.llm import LLM; print('OK, methods:', [m for m in dir(LLM) if not m.startswith('_')])"
```

预期输出：
```
OK, methods: ['aembed_documents', 'aembed_query', 'ainvoke', 'ainvoke_structured', 'astream_invoke', 'embed_documents', 'embed_query', 'invoke', 'invoke_structured', 'stream_invoke']
```

### 8.2 最小可运行示例（需要 API Key）

```python
from dotenv import load_dotenv
from fast_agents.llm import LLM

load_dotenv()
llm = LLM()

# 1) 非流式对话
msg = llm.invoke([{"role": "user", "content": "1+1=?"}])
print("chat:", msg.content)

# 2) 单条 query 向量
qv = llm.embed_query("今天天气怎么样")
print(f"query vector dim = {len(qv)}")   # 1536

# 3) 批量文档向量
dvs = llm.embed_documents(["段落一", "段落二", "段落三"])
print(f"got {len(dvs)} vectors, dim = {len(dvs[0])}")
```

### 8.3 结构化输出（v1.2.0+，需要 API Key）

定义 pydantic 模型作为输出 schema，`invoke_structured` 自动约束模型输出并解析为实例：

```python
from pydantic import BaseModel, Field
from fast_agents.llm import LLM, LLMStructuredOutputError

llm = LLM()

class CalendarEvent(BaseModel):
    """日程事件。"""
    name: str = Field(description="事件名称")
    date: str = Field(description="日期")
    participants: list[str] = Field(description="参与者")

try:
    event = llm.invoke_structured(
        messages=[{"role": "user", "content": "Alice 和 Bob 周五去科学展"}],
        response_format=CalendarEvent,
    )
    print(event.name)          # "科学展"
    print(event.participants)  # ["Alice", "Bob"]
except LLMStructuredOutputError as e:
    print(f"结构化输出失败：{e}，retryable={e.retryable}")
```

> `strict=False`（默认）宽容模式，兼容绝大多数端点；`strict=True` 开启严格约束（要求端点支持 Structured Outputs）。

### 8.4 接到 ReActAgent 上

LLM 实例直接传给 Agent 构造函数即可——这是仓库既有用法，embedding 能力是 LLM 实例自带的，不需要额外参数：

```python
from fast_agents.llm import LLM
from fast_agents.agents.react_agent.react_agent import ReActAgent
from fast_agents.tools.tool_registry import ToolRegistry

llm = LLM()
tool_registry = ToolRegistry()
agent = ReActAgent(llm=llm, tool_registry=tool_registry)
# agent 内部在需要做语义检索时会调用 self.llm.embed_query / embed_documents
```

### 8.5 典型用法

**用法 1：对话用 DeepSeek、向量化用 OpenAI（混合模型）**

```python
llm = LLM(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-deepseek-xxx",
    embedding_model="text-embedding-3-small",   # embedding 仍然走 OpenAI
)
```

**用法 2：对话 + 向量化都走国内中转端点**

```python
llm = LLM(
    model="qwen-plus",
    embedding_model="text-embedding-v3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-dashscope-xxx",
)
```

**用法 3：单次调用覆盖模型（临时切模型）**

```python
# 默认用 gpt-4o-mini，但这次用 gpt-4o
msg = llm.invoke(messages, model="gpt-4o")
```

### 8.6 跑自测

直接运行 client 文件末尾的 `__main__` 块，会依次跑流式对话 + `embed_query` + `embed_documents`：

```powershell
python fast_agents\llm\openai_client.py
```

### 8.7 接入 TraceLogger（可选，v1.1.0+）

不传 `tracer` 时行为与旧版完全一致；传入后每次调用记录一条 LLM span（含 token 用量），可直接接上 trace 模块的落盘 / 回放：

```python
from fast_agents.llm import LLM
from fast_agents.trace import TraceLogger

tracer = TraceLogger(path="logs/llm_run.jsonl", session_id="chat-001")
llm = LLM(tracer=tracer)                      # ①实例级：所有调用自动打点
msg = llm.invoke([{"role": "user", "content": "你好"}])

# ②单次覆盖（不传则回退实例级 tracer）：
llm.embed_query("今天天气怎么样", tracer=tracer)

# 查回放
for row in tracer.replay(tracer.read_spans(kind="llm")[0].trace_id):
    print(row)
```

> `tracer=None`（默认）时**不构造任何 span**，对既有链路零开销——这是本模块「新增能力默认非侵入」的硬承诺（对齐 `AGENTS.md` §9）。

### 8.8 异常捕获与重试分流（v1.1.2+）

所有对外方法抛出的运行时错误均为 `LLMError` 的子类，每个子类带 `retryable` 属性；I4（重试/熔断）可直接按该属性分流，不用再按字符串猜错误类型：

```python
import time
from fast_agents.llm import (LLM, LLMError, LLMAuthenticationError,
                             LLMConfigError, LLMModelNotFoundError,
                             LLMRateLimitError)

llm = LLM()

def invoke_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except (LLMAuthenticationError, LLMConfigError, LLMModelNotFoundError) as e:
            # 不可重试：API Key 错 / 配置错 / 模型不存在，直接上报用户
            raise RuntimeError(f"LLM 配置错误，请检查环境变量：{e}") from e
        except LLMRateLimitError as e:
            # 限流：指数退避
            time.sleep(2 ** attempt)
            continue
        except LLMError as e:
            if e.retryable:
                time.sleep(2 ** attempt)
                continue
            raise  # 其他不可重试错误直接抛
    raise RuntimeError(f"LLM 调用失败，已重试 {max_retries} 次")

msg = invoke_with_retry([{"role": "user", "content": "1+1=?"}])
print(msg.content)
```

> **向后兼容提示**：v1.1.1 及之前的版本所有异常都是裸 `Exception("LLM调用失败: ...")`；v1.1.2 起所有运行时错误都是 `LLMError` 子类，但因为 `LLMError` 继承自 `Exception`，上层原来写的 `except Exception:` 兜底逻辑**无需改动**即可继续工作，只是现在可以更精细地按类型分流。
>
> **不要捕获的异常**：入参错误（传了非 str 的 text、传了 int 当 messages）仍然抛原生 `TypeError` / `ValueError`，这是编程错误，上层不应该 catch 后重试——让它在开发期直接爆出来即可。

### 8.9 重试/熔断（v1.3.0+，I4，默认开启）

v1.3.0 起 LLM 默认开启**指数退避重试 + 断路器**，无需手动写循环：

```python
from fast_agents.llm import LLM, RetryConfig, CircuitBreakerRegistry

# 1) 默认：max_retries=2（最多 3 次尝试）、Full Jitter 指数退避、全局断路器
llm = LLM(model="qwen3-14b-instruct")

# 2) 调参：最多重试 3 次、基础退避 2s、熔断阈值 10 次、冷却 60s
llm = LLM(
    model="qwen3-14b-instruct",
    retry=RetryConfig(
        max_retries=3,
        base_delay=2.0,
        max_delay=60.0,
        breaker=CircuitBreakerRegistry(
            failure_threshold=10,
            recovery_timeout=60.0,
        ),
    ),
)

# 3) 完全关闭（零开销路径，适用于已有外层重试或排障）
llm = LLM(model="qwen3-14b-instruct", retry=None)
# 或
llm = LLM(model="qwen3-14b-instruct", retry=RetryConfig.disabled())
```

可重试与不可重试行为：
- **自动重试**：`LLMConnectionError`（连接失败）/`LLMTimeoutError`（超时）/`LLMRateLimitError`（429 限流）/`LLMServerError`（5xx 服务端错误）——这四个 `retryable=True`；
- **直接透传不重试**：`LLMAuthenticationError` / `LLMConfigError` / `LLMModelNotFoundError` / `LLMBadRequestError` / `LLMStructuredOutputError` 等 `retryable=False` 的异常；
- **流式调用边界**：`stream_invoke` 只在「发起 HTTP 请求、拿到首块之前」可重试；一旦开始 yield chunks，后续错误直接冒泡（避免输出片段重复错位）；
- **断路器维度**：按 `(base_url, model)` 独立统计，默认连续 5 次失败后 OPEN 30 秒，期间请求快速抛 `LLMCircuitOpenError`，不真正触达下游；
- **重试耗尽**：抛出 `LLMRetryExhaustedError`，`original_error` 保留最后一次失败的异常；
- **trace 打点**：开启 tracer 时，span 的 `metadata.retry_attempts` 字段会记录本次业务调用实际重试次数（0 表示首次就成功），便于成本分析。

离线示例见 `examples/llm/retry_demo.py`。

### 8.10 异步调用（v1.4.0+，I2）

5 个 `a*` 方法与同步版签名完全一致，在事件循环里使用；底层异步 client 首次 `await` 时才懒加载：

```python
import asyncio
from fast_agents.llm import LLM

async def main():
    llm = LLM()

    # 1) 异步非流式对话
    msg = await llm.ainvoke([{"role": "user", "content": "1+1=?"}])
    print("chat:", msg.content)

    # 2) 异步流式：async for 遍历 StreamChunk
    async for chunk in llm.astream_invoke([{"role": "user", "content": "写一首短诗"}]):
        if chunk.type == "text":
            print(chunk.text, end="", flush=True)

    # 3) 异步结构化输出（response_format 为 pydantic 模型类，同 §8.3 的 CalendarEvent）
    event = await llm.ainvoke_structured(
        [{"role": "user", "content": "Alice 和 Bob 周五去科学展"}],
        CalendarEvent,
    )

    # 4) 异步 embedding + gather 并发（分片/顺序对齐语义与同步一致）
    vectors = await llm.aembed_documents(["段落一", "段落二", "段落三"])
    q1, q2 = await asyncio.gather(
        llm.aembed_query("今天天气怎么样"),
        llm.aembed_query("明天会下雨吗"),
    )

asyncio.run(main())
```

> - 同步用户零成本：只有真正 `await` 过 `a*` 方法才会创建异步 client（`asyncio.Lock` 协程安全懒加载）；
> - 重试/熔断/tracer 与同步共用同一套配置（见 §8.7 / §8.9），异步 span 的 `metadata` 额外标记 `async=True`；
> - 完整示例见 `examples/llm/async_demo.py`（覆盖非流式/流式/结构化/Embedding/gather 并发五种用法）；离线测试见 `tests/llm/test_async.py`（13 项全绿）。

---

## 九、API 速查

### 9.1 LLM（门面类）

构造：

```python
from fast_agents.llm import LLM
llm = LLM(
    model=None,                 # chat 模型名，默认读 $LLM_MODEL_ID
    api_key=None,               # 默认读 $LLM_API_KEY
    base_url=None,              # 默认读 $LLM_BASE_URL
    provider="openai",          # v1.5.0：openai / anthropic / gemini（大小写不敏感），同步/异步自动对齐
    timeout=None,               # 默认 180 秒（client 层）
    embedding_model=None,       # embedding 模型，默认读 $LLM_EMBED_MODEL，否则 text-embedding-3-small
    extra_body=None,            # 可选；provider 私有请求体扩展（如 {"enable_thinking": False}），默认 None 不发送
    tracer=None,                # 可选打点器；非 None 时同步/异步各方法各记一条 llm.* span（异步额外标 async=True）
    retry=None,                 # v1.3.0 新增；RetryConfig 实例；默认 None 时自动启用默认配置（max_retries=2）；显式传 RetryConfig.disabled() 或 RetryConfig(max_retries=0) 关闭
)
```

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `invoke(messages, tools=None, tracer=None, **kwargs)` | `messages: list[dict]`（OpenAI message 格式），`tools`（OpenAI tools schema 列表），`tracer`（可选 `TraceLogger`） | `Message`（含 `role`/`content`/`tool_calls`） | 非流式对话，kwargs 可覆盖 `model`/`temperature`/`max_tokens`/`extra_body`；传 `tracer` 时记 `llm.invoke` span |
| `stream_invoke(messages, tools=None, tracer=None, **kwargs)` | 同上 | `Iterator[StreamChunk]` | 流式对话，逐段 yield StreamChunk（text/tool_call/end 三态，E7；`reasoning_content` 打印到控制台但不进入 yield）；kwargs 可覆盖 `model`/`temperature`/`extra_body`（`max_tokens` 固定取实例默认值、不支持单次覆盖）；传 `tracer` 时记 `llm.stream_invoke` span |
| `invoke_structured(messages, response_format, *, strict=False, tracer=None, **kwargs)` | `messages: list[dict]`，`response_format: type[BaseModel]`（pydantic 模型类），`strict: bool`（默认 False 宽容模式），`tracer`（可选 `TraceLogger`） | pydantic 实例 | **v1.2.0 新增**：结构化输出，通过 `response_format=json_schema` 约束模型输出，自动解析为 pydantic 实例；kwargs 可覆盖 `model`/`temperature`/`max_tokens`；传 `tracer` 时记 `llm.invoke_structured` span |
| `embed_query(text, tracer=None, **kwargs)` | `text: str`，`tracer`（可选 `TraceLogger`） | `list[float]` | 单条查询向量；kwargs 可覆盖 `model`（embedding 模型名）；传 `tracer` 时记 `llm.embed_query` span |
| `embed_documents(texts, tracer=None, **kwargs)` | `texts: list[str]`，`tracer`（可选 `TraceLogger`） | `list[list[float]]` | 批量文档向量；内部按 2048 分片并发；顺序与输入严格一致；kwargs 可覆盖 `model`；传 `tracer` 时记 `llm.embed_documents` span |
| `async ainvoke(messages, tools=None, tracer=None, **kwargs)` | 同 `invoke` | `Message` | **v1.4.0（I2）**：异步非流式对话，参数语义与 `invoke` 完全一致；重试/断路器与同步共用同一套配置；传 `tracer` 时记 `llm.ainvoke` span，metadata 含 `async=True` |
| `async astream_invoke(messages, tools=None, tracer=None, **kwargs)` | 同上 | `AsyncIterator[StreamChunk]` | **v1.4.0（I2）**：异步流式，`async for` 遍历；仅首块前可重试，语义同同步版；span 名 `llm.astream_invoke` |
| `async ainvoke_structured(messages, response_format, *, strict=False, tracer=None, **kwargs)` | 同 `invoke_structured` | pydantic 实例 | **v1.4.0（I2）**：异步结构化输出；span 名 `llm.ainvoke_structured` |
| `async aembed_query(text, tracer=None, **kwargs)` | 同 `embed_query` | `list[float]` | **v1.4.0（I2）**：异步单条查询向量；空串/类型防御与同步一致；span 名 `llm.aembed_query` |
| `async aembed_documents(texts, tracer=None, **kwargs)` | 同 `embed_documents` | `list[list[float]]` | **v1.4.0（I2）**：异步批量文档向量；内部按 2048 分片 + `asyncio.Semaphore(4)` 协程级并发；顺序与输入严格一致；span 名 `llm.aembed_documents` |

### 9.2 OpenAIClient / AsyncOpenAIClient（底层，一般不直接使用）

`OpenAIClient` 继承官方 `openai.OpenAI`，拥有 SDK 全部能力（`chat.completions` / `embeddings` / `files` / 等）。本模块自定义的公开方法（`invoke` / `stream_invoke` / `embed_query` / `embed_documents`）整体签名与 `LLM` 对齐，但比门面多一个**关键字专用**的 `usage_sink: Optional[dict] = None`（供打点回填 token 用量，默认 `None` 时不产生任何副作用）；门面层则在方法外多一层可选 `tracer` 打点。构造参数另支持 `extra_body: Optional[dict] = None`（provider 私有请求体扩展，默认 `None` 不发送，v1.1.1）。

`AsyncOpenAIClient`（**v1.4.0 新增**）继承官方 `openai.AsyncOpenAI`，5 个 `a*` 方法与 `OpenAIClient` 的同步方法一一对齐，同样多一个关键字专用的 `usage_sink`；两者复用同一套异常映射与工具函数（见 §7.7）。上层无需直接实例化：`LLM` 门面在首次调用异步方法时自动懒加载对应 provider 的异步 client。

### 9.3 已废弃/预留

- 目前没有已废弃的旧写法；
- `provider` 参数已于 v1.5.0（I3）落地为三路原生分发（openai / anthropic / gemini）；非 `openai` 分支需安装对应可选 SDK（`pip install fast-agents[anthropic]` / `pip install fast-agents[gemini]`），未安装时抛出带明确安装提示的 `LLMConfigError`。

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

1. **为什么继承而不是组合官方 `OpenAI` 类？**
   官方 SDK 的 `OpenAI` 类已经封装了连接池、httpx client、auth、重试、流式解析等大量细节；继承它比组合它少写几十个方法代理，且能直接调用 `self.embeddings.create(...)` / `self.chat.completions.create(...)`，可读性更好。风险点是 SDK 大版本升级可能改内部实现，但 openai SDK 从 v1 起公开 API 稳定，风险可控。

2. **为什么 `embed_query` 收 `str` 而不是 `list[str]`？**
   方法名里的 `query` 是单数——查询就是「用户的一个问题」，不是一批问题。批量查询在工程上非常罕见（RAG 是一次 query 对一批 doc）；批量能力放在 `embed_documents`。如果上层真的需要批量 query，用 `embed_documents` 即可——底层实现完全能跑，只是语义上要知道「query 前缀/input_type」不会被自动加上（将来加非对称模型时这里要注意）。

3. **为什么默认开启重试/熔断？为什么只重试 2 次？**
   v1.3.0 前此问题答案是"不内置"；v1.3.0（I4）改为默认开启。理由：
   - **裸 `Exception` 已被分层异常体系（v1.1.2）取代**：每个异常有明确的 `retryable` 属性，重试判断不再是"按字符串猜状态码"，可以安全地做类型分流；
   - **默认值保守**：`max_retries=2`（总尝试 3 次）+ Full Jitter 指数退避（基础 1s、上限 30s）+ 断路器阈值 5 次冷却 30s，对实时对话场景（需要快速失败）和批处理场景（需要更强韧性）都不至于过度；
   - **零开销关闭路径**：`retry=None` 完全不走重试/熔断逻辑，和 v1.2.0 行为字节一致，方便排障或接入外层已有重试策略；
   - **流式调用有重试边界**：只在 HTTP 首包前重试，yield 出去之后不再重试，避免输出片段重复错位。
   策略可通过 `RetryConfig(...)` 完全自定义；`CircuitBreakerRegistry` 可跨实例共享，方便把多个 LLM 实例纳入同一个熔断窗口。

4. **为什么不在 LLM 里做 embedding 缓存？**
   缓存的 key 是 `(model, text_hash)`，但缓存存储位置（内存/磁盘/redis）和失效策略是上层场景决定的（短期会话可以内存缓存、跨进程需要磁盘、多租户需要 redis）。LLM 层做缓存意味着强绑一种策略，反而阻碍上层。将来可以加一个可选的 `cache:` 构造参数，传入一个有 `get/set` 方法的对象即可，默认 None。

5. **为什么 chat 模型没有硬编码默认值，embedding 模型有？**
   chat 模型的选择直接决定回复质量、价格、速度，差异极大（几元到几百元每百万 token），硬选默认会让没配环境变量的用户在跑完才发现用错模型、产生意料外费用；embedding 模型相对稳定（`text-embedding-3-small` 便宜、足够快、维度统一），且嵌入质量稍差只会影响检索召回率，不会产生安全/费用风险，所以给一个开箱即用的默认是合适的。

6. **为什么打点用「可选 `tracer` 参数」而不是内置一个 Logger？**
   内置 Logger 会把 LLM 和具体观测后端（jsonl / sqlite / 云上报）绑死，且违背「新增能力默认非侵入」；改成可选 `tracer=None` 后，默认路径零开销、零依赖，需要观测时再传入 trace 模块的 `TraceLogger` 实例即可。token 用量走同一套机制——底层只把 `response.usage` 回填进一个调用方提供的 dict（`usage_sink`），不改变任何既有返回值语义，`tracer=None` 时这段代码根本不执行。

7. **为什么结构化输出选 `response_format=json_schema` 而不是强制工具调用兜底？**
   两条路线各有适用场景：`response_format` 是 OpenAI 原生协议，绝大多数兼容端点（DeepSeek / Qwen / 豆包）均已支持，语义最清晰（模型直接输出 JSON）；强制工具调用路线（注册一个「返回结果」工具，`tool_choice=required`）适合端点不支持 `response_format` 的降级场景。当前先做路线 A（`response_format`），`strict=False` 默认宽容模式保证最大兼容性；路线 B 待出现真实不兼容端点时再加，符合 R3「没有第二个真实使用方之前不提前造抽象」。

8. **为什么异步客户端单独建文件、且懒加载？（v1.4.0，I2）**
   三个理由：①`openai.OpenAI` 与 `openai.AsyncOpenAI` 是两个独立类，底层 HTTP 栈（`httpx.Client` / `httpx.AsyncClient`）完全分离，LangChain / LiteLLM / 官方 SDK 的主流做法同样是同步异步两套并行实现而非单类组合；②**懒加载**：异步 client 只在首次 `await a*` 时才创建（`asyncio.Lock` 协程安全），纯同步用户零成本，符合「新增能力默认非侵入」；③**配置复用**：异步重试执行器与同步完全共用同一套 `RetryConfig` / 断路器实例（单事件循环内 `threading.Lock` 临界区无挂起点、不会死锁），避免同步异步两套配置各管各的产生漂移。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| 抛 `LLMAuthenticationError`（401/403） | API Key 错或过期。检查 `.env` 的 `LLM_API_KEY`，或显式传 `api_key=` |
| 抛 `LLMModelNotFoundError`（404） | chat / embedding 模型名错（第三方中转不一定支持 `gpt-4o-mini` 或 `text-embedding-3-small`）。第三方中转要显式传 `model=` / `embedding_model=` 指定该平台支持的模型名 |
| 抛 `LLMBadRequestError`（400）| 请求参数非法：常见是对接原生 OpenAI 端点但传了 provider 私有参数（如 `extra_body={"enable_thinking": False}`）、或单条文本超过 embedding 上限（text-embedding-3 单条约 8191 token）；RAG 场景必须先分 chunk，不能把整篇文档直接塞给 `embed_documents` |
| 抛 `LLMRateLimitError`（429）| 触发限流/配额耗尽：可退避重试；批量 embedding 时内部已按 2048 分片 + 最多 4 线程并发，仍触发需自行控速 |
| 抛 `LLMConnectionError` / `LLMTimeoutError` | 网络不通或端点不可达：检查 `LLM_BASE_URL`、本地代理、服务端是否在线；属于可重试错误 |
| 抛 `LLMServerError`（5xx） | 服务端临时过载/维护：退避后重试 |
| 抛 `LLMConfigError`（初始化时） | `base_url` 格式非法或 SDK 初始化失败：检查环境变量拼写 |
| 抛 `TypeError: embed_query 要求 text 为 str` / `ValueError: embed_query 收到空字符串` | 入参错误（编程问题），不是运行时网络错误；调用方应修复参数，不要 catch 后重试 |
| 中文乱码 / Windows 控制台打印乱码 | 在脚本开头加 `import sys; sys.stdout.reconfigure(encoding='utf-8')`，或 PowerShell 先执行 `chcp 65001` |
| 调用很慢、批量 embed 耗时明显 | 单次 batch 内部已并发；若总文本量很大（> 1万条），建议调用方自己分批分段入库，控制并发 |
| 第三方中转端点 embedding 路径不是 `/v1/embeddings` | 少数中转端点改了路径；这种情况只能换端点或手动改 `base_url`，OpenAI SDK 没有路径重写开关 |
| 抛 `LLMStructuredOutputError`（JSON 解析失败） | 端点返回的内容不是合法 JSON：可能是模型不支持 `response_format`、或 prompt 不够明确；检查端点是否支持 json_schema 模式，或尝试 `strict=False` |
| 抛 `LLMRetryExhaustedError`（重试耗尽） | 默认 3 次尝试（首次 + 2 次重试）全部失败；`e.original_error` 可拿到最后一次失败的异常；常见于端点持续不可用，检查服务端状态或调高 `max_retries` |
| 抛 `LLMCircuitOpenError`（断路器 OPEN） | 该 `(base_url, model)` 维度连续失败达阈值（默认 5 次），断路器打开后 30 秒内快速拒绝请求、不真正发请求；等冷却时间过了会自动放行探测请求，成功则恢复。可调用 `RetryConfig.breaker.status(key)` 查看剩余时间，或 `.reset(key)` 手动重置 |
| 抛 `LLMStructuredOutputError`（pydantic 校验失败） | JSON 合法但字段类型/必填项不匹配：检查 pydantic 模型定义与 prompt 描述是否一致；可选字段应给默认值 |
| 抛 `LLMStructuredOutputError`（400 BadRequest） | 端点不支持 `response_format` 的 `json_schema` 类型：换用支持 Structured Outputs 的端点，或降级为普通 `invoke` + 手动解析 |
| 流式调用里 `top_p` / `stop` / `response_format` 等参数不生效（v1.3.1 前） | 旧版 `OpenAIClient.stream_invoke` 未把额外 kwargs 透传给 `chat.completions.create`（`invoke` 会透传），Agent 流式链路走 `stream_invoke` 时这些参数被静默丢弃；v1.3.1 起已与 `invoke` 对齐，升级即可修复 |
| 同步代码里想调用异步方法（v1.4.0） | `ainvoke` 等必须在事件循环内 `await`；同步上下文用 `asyncio.run(...)` 包一层，漏写 `await` 只会拿到 coroutine 对象、不会真正发出请求；`astream_invoke` 用 `async for` 遍历，不能 `await` 后当迭代器用 |

### 10.3 跨平台注意事项（Windows 专项）

1. **换行符**：本模块所有 `.py` 文件均为 CRLF（`\r\n`），直接复制落盘即可，Windows Git 若开了 `autocrlf` 也不会重复转；
2. **`.env` 文件编码**：Windows 下用记事本保存 `.env` 时要选 **UTF-8**（不要选 UTF-16 或 ANSI），否则 `python-dotenv` 读取中文会乱码；
3. **控制台编码**：`stream_invoke` 里有 print 中文字符，PowerShell 默认 GBK 编码可能乱码，启动前执行 `chcp 65001` 或在代码里 `sys.stdout.reconfigure(encoding='utf-8')`；
4. **路径空格**：若 `.env` 放在含空格的路径下，`load_dotenv()` 自动向上查找是正常的，但不要在 `base_url` 里拼 Windows 本地路径（`base_url` 是 URL 不是文件路径）；
5. **代理**：公司内网需要走 HTTP 代理时，`openai` SDK 会自动读 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量，不需要额外配置。

---

## 十一、测试与验证

本模块当前测试文件：`tests/llm/test_openai_client.py`（工具调用解析兼容性 + 流式聚合，7/7）+ `tests/llm/test_invoke_structured.py`（E9 结构化输出，19/19）+ `tests/llm/test_retry.py`（I4 重试/熔断，16/16）+ `tests/llm/test_stream_kwargs.py`（v1.3.1 参数透传，4/4）+ `tests/llm/test_async.py`（I2 异步，13/13，v1.4.0）+ `tests/llm/test_multi_provider.py`（I3 多 provider，8/8，v1.5.0）+ `tests/llm/test_multimodal_messages.py`。全部离线可跑、不依赖 API Key（LLM 调用用 fake/mock）。自测通过以下方式：

| 验证项 | 方式 | 预期结果 |
|---|---|---|
| 模块导入 | `python -c "from fast_agents.llm import LLM, LLMError"` | 无报错，`LLM` 与异常类均可导入 |
| 方法挂载 | `[m for m in dir(LLM) if not m.startswith('_')]` | 含同步五方法 `invoke` / `stream_invoke` / `invoke_structured` / `embed_query` / `embed_documents` + 异步五方法 `ainvoke` / `astream_invoke` / `ainvoke_structured` / `aembed_query` / `aembed_documents`（v1.4.0） |
| 语法检查 | `ast.parse` 十二个文件（含 async_openai_client.py 等） | 无 SyntaxError |
| 真实 API 联通 | `python fast_agents\llm\openai_client.py` | 流式打印对话结果，输出两个向量的维度信息 |
| 空串防御 | 传入 `embed_query("")` / `embed_query("   ")` | 抛出 `ValueError`，不发请求 |
| 类型防御 | 传入 `embed_query(123)` / `embed_documents("not a list")` | 抛出 `TypeError` |
| 分片并发 | `embed_documents` 传 3000 条短文本（需 key） | 返回 3000 个向量，顺序与输入一一对应 |
| 下游接入 | ReActAgent 构造时传 `llm=LLM()` | 既有对话/工具调用流程不受影响 |
| 打点零侵入 | `LLM()`（不传 `tracer`）调用四个方法 | 不构造任何 span、行为与旧版逐字节一致 |
| 打点落盘 | `LLM(tracer=TraceLogger(path=..., enabled=True))` 调 `invoke` | `read_spans(kind="llm")` 能读到 `llm.invoke` span，`metadata` 含 `total_tokens`（端点回填 usage 时） |
| **异常分类（v1.1.2）** | monkeypatch `chat.completions.create` 抛各 SDK 原生异常 | `APIConnectionError → LLMConnectionError(retryable=True)`、`RateLimitError → LLMRateLimitError(retryable=True)`、`AuthenticationError → LLMAuthenticationError(retryable=False)`、`BadRequestError → LLMBadRequestError(retryable=False)`；所有异常都继承自 `LLMError`，且保留原始异常链（`__cause__` 可访问） |
| **结构化输出（v1.2.0）** | `tests/llm/test_invoke_structured.py`（19 项全绿） | 正常解析 / 嵌套模型 / 可选字段 / JSON 解析失败 / 空内容 / pydantic 校验失败 / 非 pydantic 类型 / response_format 正确传递 / strict 模式 / usage_sink 回填 / 门面层代理 / tracer 打点 / retryable=False |
| **重试/熔断（v1.3.0，I4）** | `tests/llm/test_retry.py`（16 项全绿） | 首次成功不重试 / 可重试错误指数退避至成功 / 重试耗尽抛 LLMRetryExhaustedError / 不可重试错误不重试 / retry=None 零开销 / RetryConfig.disabled() / 参数校验 / Full Jitter backoff 范围 / 断路器 OPEN 快速拒绝 / recovery_timeout 后 HALF_OPEN 成功回 CLOSED / HALF_OPEN 探测失败重开 / (base_url,model) 维度隔离 / reset 重置 / 流式首块成功后正常迭代 / 流式首块失败可重试 / 流式中途失败不重试；离线示例 `examples/llm/retry_demo.py` 跑通 6 个场景 |
| **流式参数透传（v1.3.1）** | `tests/llm/test_stream_kwargs.py`（4 项全绿） | `stream_invoke` 把 `top_p` / `frequency_penalty` / `presence_penalty` / `stop` / `response_format` 透传给 `chat.completions.create`；`temperature` / `max_tokens` 未传取实例默认、传入则单次覆盖；`invoke` 透传行为不变；`invoke` 与 `stream_invoke` 对同一组额外 kwargs 的处理一致 |
| **异步（v1.4.0，I2）** | `tests/llm/test_async.py`（13 项全绿，`tests/_runner.py` 自动识别 `async def test_*` 并用 `asyncio.run` 执行） | 异步 client 五方法与同步语义对齐 / 门面懒加载协程安全（多次并发调用只建一个实例） / `arun_with_retry` 重试耗尽与退避 / `asyncio.gather` 并发顺序对齐 / tracer 异步 span 记 `async=True`；离线示例 `examples/llm/async_demo.py` 跑通五种用法 |
| **无裸 Exception** | `grep -n "raise Exception" fast_agents/llm/*.py` | 结果为 0（所有对外方法不再抛裸 `Exception`） |

**关键回归断言（如果后续补 `test_llm.py`，这几条必须锁住）**：

1. `embed_query("")` 必须抛 `ValueError`，不能发 HTTP 请求（防止被计费）；
2. `embed_documents([])` 返回 `[]`，不报错；
3. `embed_documents(["a", "", "b"])` 返回长度为 3 的列表（空串不被丢弃，防止顺序错位）；
4. 传入 `model=` kwarg 能正确覆盖默认模型（通过 monkeypatch `embeddings.create` 断言入参）；
5. `LLM` 门面的四个方法都把 kwargs 透传给 client，不在中间吃掉参数；
6. `LLM()`（`tracer=None`）对四个方法的调用次数 = 底层 client 调用次数，不产生额外 span（零开销承诺）；
7. 传入 `tracer=` 时，`invoke` 落一条 `llm.invoke` span、`metadata` 含 `model` 与（端点回填时）`total_tokens`；
8. `stream_invoke` 传 `tracer=` 时，span 在生成器耗尽后 `end`，`output.chunks` 等于实际 yield 次数；
9. **（v1.1.2）** 所有对外方法在 SDK 抛错时抛出 `LLMError` 子类而非裸 `Exception`；每个子类的 `retryable` 属性正确（连接/超时/限流/5xx 为 True，其余为 False）；
10. **（v1.1.2）** SDK 原生异常可通过 `err.__cause__` 访问（异常链未断）；客户端初始化失败（如 `base_url=""`）抛 `LLMConfigError`。
11. **（v1.3.0，I4）** 默认构造 `LLM()` 应自动对可重试错误做指数退避重试（最多 2 次）；显式传 `retry=None` 时不重试、不包 `LLMRetryExhaustedError`；连续 5 次可重试失败后断路器进入 OPEN，第 6 次请求快速抛 `LLMCircuitOpenError` 而不真正调用底层；流式 `stream_invoke` 在首块获取成功后，迭代中途失败**不重试**；tracer span 的 `metadata.retry_attempts` 正确记录实际重试次数；两个新异常类 `LLMRetryExhaustedError` / `LLMCircuitOpenError` 均 `retryable=False`。
12. **（v1.4.0，I2）** 异步门面方法与同步版语义逐一对齐：从未调用过 `a*` 时不实例化异步 client（懒加载承诺）；并发首次调用只创建一个实例（`asyncio.Lock` 协程安全）；传 `tracer=` 时 span 命名 `llm.ainvoke` 等且 `metadata.async` 为 `True`；同步与异步共用同一 `RetryConfig` 与断路器实例（熔断计数互相感知）。

---

## 十二、下一步演进路线

编号与《演进路线图.md》模块 I 保持一致。

### 12.1 已交付

| 编号 | 任务 | 完成版本 | 落点 |
|------|------|---------|------|
| ✅ **I1** | `embed_query` / `embed_documents` | v1.0.0 | `llm.py`：单条查询侧 + 批量文档侧（自动 2048 分片 + 线程池并发 + `data[*].index` 顺序对齐），chat / embedding 模型独立配置 |
| ✅ **I5** | Token 用量统计 + TraceLogger 打点 | v1.1.0 | `LLM` 与四个方法新增可选 `tracer`，记录 `llm.invoke` / `llm.stream_invoke` / `llm.embed_query` / `llm.embed_documents` 四条 Span，`metadata` 写入 `model` + token 用量（见 §7.2 / §9.1 / §10.1） |
| ✅ **I5-补** | 分层异常体系（errors.py） | v1.1.2 | 新增 `fast_agents/llm/errors.py`（11 个子类 + `retryable` 属性）；所有对外方法不再抛裸 `Exception`，严格对齐 `AGENTS.md` §5.1；OpenAI SDK 原生异常按类型一一映射，业务层再包 `LLMInvokeError` / `LLMEmbeddingError`；为 I4 重试/熔断按类型分流提供依据 |
| ✅ **E9 / I8** | Structured Output（`invoke_structured`） | v1.2.0 | `OpenAIClient.invoke_structured` + `LLM.invoke_structured`：通过 `response_format=json_schema` 协议约束输出，自动解析为 pydantic 实例；新增 `LLMStructuredOutputError` 异常；支持 `strict` 模式开关；门面层带可选 `tracer` 打点；为 E1 PlanExecuteAgent 前置 |
| ✅ **I4** | 指数退避重试 + 断路器（Retry / Circuit Breaker） | v1.3.0 | `retry.py` 实现 `RetryConfig`（Full Jitter 指数退避）+ `CircuitBreakerRegistry`（按 (base_url,model) 维度隔离的线程安全断路器）+ `run_with_retry` / `run_stream_with_retry` 执行器；`retry_errors.py` 新增 `LLMRetryExhaustedError` / `LLMCircuitOpenError`；四个核心方法默认开启（max_retries=2），可通过 `retry=None` 完全关闭；流式仅在首块前重试；tracer metadata 写 `retry_attempts`；零新增依赖；为 E1 PlanExecuteAgent / A10 工具超时熔断打下基础 |
| ✅ **I2** | 异步能力（`ainvoke` / `astream_invoke` / `ainvoke_structured` / `aembed_query` / `aembed_documents`） | v1.4.0 | `async_openai_client.py`（`AsyncOpenAIClient` 继承官方 `AsyncOpenAI`，接口与同步版逐字节对齐）+ 门面 5 个 `a*` 方法（`asyncio.Lock` 协程安全懒加载）+ `retry.py` 异步执行器 `arun_with_retry` / `arun_stream_with_retry`（与同步共用同一套 RetryConfig/断路器）；`aembed_documents` 用 `asyncio.Semaphore(4)` 对齐同步线程池并发度；tracer 异步 span 标记 `async=True`；`tests/llm/test_async.py` 13/13；零新增依赖；解锁 K2 trace 异步打点前置（见 §7.7） |
| ✅ **I3** | Anthropic / Gemini 原生适配 | v1.5.0 | `anthropic_client.py` / `async_anthropic_client.py` / `gemini_client.py` / `async_gemini_client.py` 四个客户端，同步/异步接口与 OpenAI 版本完全对齐；门面 `provider` 三路分发（大小写不敏感）；第三方 SDK 为可选依赖，未安装抛带安装提示的 `LLMConfigError`；`tests/llm/test_multi_provider.py` 8/8 |

### 12.2 待办（按优先级）

| 编号 | 任务 | 优先级 | 依赖 | 一句话说明 |
|------|------|--------|------|-----------|
| **I6** | `LocalEmbeddingClient`（sentence-transformers） | P2 | — | 现状只能调云端 API；sentence-transformers 依赖 torch（约 2GB），违反「零重型依赖」原则，可作为可选 extra（`pip install fast-agents[local]`） |
| **I7** | 可选 embedding 缓存 | P2 | — | 相同文本重复调用会重复请求、重复计费；缓存后端（内存 / 磁盘 / redis）依赖场景，先不内置，可通过「LLM 接受可选 cache 参数」实现可插拔 |

---

## 十三、参考

- [LangChain Embeddings 抽象](https://python.langchain.com/docs/concepts/embedding_models/) —— 借鉴了 `embed_query` / `embed_documents` 的命名与语义区分；
- [OpenAI 官方 Python SDK](https://github.com/openai/openai-python) —— 本模块的 `OpenAIClient` 直接继承它的 `OpenAI` 类，复用 HTTP 栈；
- [OpenAI Embeddings API 文档](https://platform.openai.com/docs/api-reference/embeddings) —— 输入上限 2048、向量维度、`data[*].index` 顺序约定的权威来源；
- [BGE 模型使用说明](https://github.com/FlagOpen/FlagEmbedding) —— 理解 query/document 非对称前缀需求（为什么要分两个方法）；
- [DashScope 文本向量 v3](https://help.aliyun.com/zh/model-studio/user-guide/embedding) —— 理解 `input_type=query/document` 的非对称需求；
- [LiteLLM](https://github.com/BerriAI/litellm) —— 多 provider 统一抽象的参考实现，本模块刻意没引入它，保持零重型依赖。

---

## 附录：发布前自检清单

- [x] 文档名 `LLM模块编写说明.md`，落位 `docs/`；
- [x] 开篇写清了定位、配套代码路径、版本、平台与依赖，且有可跳转目录；
- [x] 有更新历史，且破坏性变更都给了迁移办法（v1.1.1 移除硬编码 `enable_thinking`，已给出「显式传 `extra_body`」的迁移办法；v1.1.2 异常体系升级为分层异常，已说明向后兼容 `except Exception`；v1.2.0 纯新增方法，既有链路零改动；v1.4.0 纯新增异步 `a*` 方法，同步链路行为零改动）；
- [x] 第二章讲清了五类能力分层与 7 条设计原则（含 v1.1.2 错误分层可分流原则 + v1.2.0 结构化输出路线选择）；
- [x] 第三章包含对比总表 5 家 + 异同表（含「缺少什么」），已同步结构化输出 / 重试熔断（v1.3.0）/ 异步（v1.4.0）/ 多 provider（v1.5.0）现状；
- [x] 第四章有目录结构（单数目录 + 逐文件职责，含 `errors.py` / `async_openai_client.py` 等 12 个文件）、分层图（同步/异步双路径）、调用链路图（含异步链路）、与 Agent 接入边界；
- [x] 第五章（弹性节）讲清了「为什么 Chat/Embedding 在同一个门面」+「为什么分 query/documents 两个方法」；
- [x] 第六章数据模型字段命名与仓库既有约定一致（`invoke` 动词、门面类单名 `LLM`）；
- [x] 源码导读**七个文件**都有「负责什么 / 关键实现 / 为什么」，覆盖 v1.1.0 `tracer`、v1.1.1 `extra_body`、v1.1.2 异常映射 / errors.py、v1.2.0 `invoke_structured`、v1.3.0 retry/retry_errors（指数退避 Full Jitter + 线程安全断路器 + 流式重试边界）与 v1.4.0 async_openai_client.py（异步五方法 / 懒加载 / Semaphore 并发 / 异步重试）；
- [x] 快速上手是 Windows PowerShell，且如实说明 `tests/` 目录结构现状；
- [x] 快速上手新增 §8.3 结构化输出示例 + §8.8 异常捕获与重试分流示例（v1.1.2）+ §8.9 重试/熔断（v1.3.0，默认开启、调参与关闭方式）+ §8.10 异步调用（v1.4.0，五类 `a*` 用法 + gather 并发）；
- [x] API 速查完整列出构造参数（含 `extra_body` / `tracer` / `retry`）与同步/异步共十个方法（含 `invoke_structured`、`a*` 系列与 `tracer` 形参）与异常类清单；
- [x] 第十章三块齐全：**9 条**设计决策（新增 v1.3.0 为什么默认开启重试、v1.4.0 为什么异步单独建文件且懒加载）、**18 条**常见坑（新增重试耗尽 / 断路器 OPEN / v1.3.1 流式参数透传 / v1.4.0 同步代码调异步方法四条）、5 条 Windows 专项；
- [x] 测试章节列出 16+ 项验证点 + **12 条**关键回归断言，含 `tests/llm/test_invoke_structured.py` 19/19、`tests/llm/test_retry.py` 16/16、`tests/llm/test_async.py` 13/13（v1.4.0）；
- [x] 演进路线列出已交付项（含 v1.2.0 E9/I8 结构化输出、**v1.3.0 I4 重试/熔断**、**v1.4.0 I2 异步**、**v1.5.0 I3 多 provider**）与待办，与《演进路线图.md》编号对齐；
- [x] **章节数量、顺序、编号与其余模块文档完全一致**（开篇→一～十三→附录）；
- [x] 全文无「待补充 / TBD」残留；
- [x] 交付的 `.py` / `.md` 文件为 CRLF 换行符（`.gitattributes` 约定）；
- [x] **零新增依赖**：retry.py 仅使用标准库 `time` / `random` / `threading` / `functools`，v1.4.0 异步实现仅新增标准库 `asyncio` / `inspect`，`pyproject.toml` / `requirements.txt` 无变更。
