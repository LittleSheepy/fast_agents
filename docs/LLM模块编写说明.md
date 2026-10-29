# FastAgents LLM模块编写说明

本文档面向 fast_agents LLM（大语言模型）系统，参考 **LangChain BaseLanguageModel / Embeddings**、**LlamaIndex LLM 抽象**、**LiteLLM 多 provider 适配**、**OpenAI 官方 Python SDK** 等框架的共性设计，给出本模块的完整说明、源码解析与自定义教程。

> 版本：v1.0 · 2026-10-29 · 首版交付：补齐 `embed_query` / `embed_documents`，语义检索链路打通
> 适用版本：`fast_agents.llm`
> 配套代码：`fast_agents/llm/`（3 个文件：`__init__.py` / `llm.py` / `openai_client.py`）+ 本文档
> 运行环境：Python 3.10+，Windows / macOS / Linux；依赖：`openai>=1.0`（项目主依赖，零新增依赖）

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

> **v1.0 更新摘要**（本次同步）
> 1. **新增 `embed_query` / `embed_documents`** —— 补齐语义检索前置能力，命名对齐 LangChain；所有新代码通过 `LLM` 实例直接调用，`import` 路径不变（仍然是 `from fast_agents.llm import LLM`），无破坏性变更；
> 2. **新增 `embedding_model` 配置** —— chat 模型与 embedding 模型独立，通过构造参数 `embedding_model=` 或环境变量 `LLM_EMBED_MODEL` 指定，默认 `text-embedding-3-small`；
> 3. **`embed_documents` 自动按 2048 分片并发** —— 调用方无需关心 OpenAI API 单批上限，返回顺序与输入严格一致。

---

## 二、先想清楚：LLM 解决什么问题

LLM 模块是整个 fast_agents 的「大脑底座」——上层所有模块（agents / memory / tools / context / skills / mcp）都要通过它把文本喂给模型、再把模型输出拿回来。它要回答两个问题：

1. **对话**：怎么把一组 message 发给模型，拿到回复（含 tool_calls），流式/非流式都要支持；
2. **向量化**：怎么把文本转成稠密向量（embedding），为下游的语义检索（memory 的语义记忆、tools 的 RAG）打地基。

### 2.1 能力分层

| 能力 | 输入 | 输出 | 典型调用方 | 本框架对应方法 |
|---|---|---|---|---|
| 非流式对话 | `list[message]` + 可选 tools | 完整 `Message`（含 tool_calls） | Agent 单轮收尾、工具挑选后总结 | `LLM.invoke()` |
| 流式对话 | `list[message]` + 可选 tools | `Iterator[str]` 文本片段流 | 实时对话交互、CLI demo | `LLM.stream_invoke()` |
| 查询向量 | 单条 `str`（用户问题） | 一维 `list[float]` | 语义检索 query 侧 | `LLM.embed_query()` |
| 文档向量 | 批量 `list[str]`（文档块） | 二维 `list[list[float]]` | 知识库入库、语义记忆写入 | `LLM.embed_documents()` |

### 2.2 关键设计原则

1. **零重型依赖**：不引入 langchain / litellm / sentence-transformers，只依赖 `openai` SDK（仓库已有）；第三方兼容端点（DeepSeek / Qwen / 豆包 / DashScope 等）通过 `base_url` + `api_key` 走 OpenAI 兼容协议；
2. **门面-客户端两层**：`LLM` 是给上层用的薄门面（只负责读环境变量、选 provider、方法代理），真正的 HTTP 调用在 `OpenAIClient` 里；后续加 Anthropic / Gemini 直连，只需新增 client 文件、不动门面签名；
3. **显式区分 query / document embedding**：方法名直接锁死语义，避免「单条批量共用一个方法、调用方自己想清楚语义」带来的混淆；对接 BGE / E5 / DashScope 等需要 query/document 不同前缀或 `input_type` 的模型时，内部实现可加分支，对外接口不用改；
4. **环境变量优先、构造参数覆盖**：`LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_EMBED_MODEL` 四个环境变量是默认配置来源，构造函数传参可覆盖；
5. **Chat 模型与 Embedding 模型解耦**：两者独立配置——很多团队用 DeepSeek-V3 做对话、用 `text-embedding-3-small` 或 DashScope 做向量化，不能绑死同一个模型名；
6. **错误可捕获**：所有对外方法统一抛出 `Exception("...调用失败: ...")`，上层（Agent、tools）可直接 catch 后做重试或提示；不静默吞错。

---

## 三、主流框架 LLM 实现对比

### 3.1 对比总表

| 框架 | 实现方式 | 依赖 | 优点 | 缺点 |
|---|---|---|---|---|
| LangChain | `BaseChatModel` / `Embeddings` 抽象基类 + 数十个 provider 实现，方法 `invoke` / `ainvoke` / `embed_query` / `embed_documents` | 重（含 langchain-core + provider 包） | 抽象统一、provider 全、生态最丰富 | 抽象层过厚、调试链路长、依赖地狱、升级易破 |
| LlamaIndex | `LLM` / `BaseEmbedding` 独立抽象，和索引/检索耦合 | 中 | RAG 场景深度优化、embedding 抽象细 | 对话侧偏弱、和 RAG 强绑定、做通用 Agent 不直观 |
| LiteLLM | 统一所有 provider 到 OpenAI 格式，`completion()` / `embedding()` 一个函数走天下 | 中（litellm 单包） | provider 最全、自动兜底重试/限流 | 黑盒过重、把 openai SDK 再包一层、出问题不好查 |
| OpenAI 官方 SDK | 直接 `client.chat.completions.create` / `client.embeddings.create` | 轻（仅 openai） | 最薄、官方维护、错误信息最原始 | chat 和 embeddings 两套独立 API、无门面、无分片/重试 |
| **fast_agents（本模块）** | 门面 `LLM` + `OpenAIClient`（继承 SDK `OpenAI` 类），方法 `invoke` / `stream_invoke` / `embed_query` / `embed_documents` | **轻（仅 openai）** | 门面薄、零额外依赖、query/document 显式区分、自动分片并发、环境变量开箱即用 | 目前只原生支持 OpenAI 兼容协议、无重试/熔断/异步、无本地 embedding |

### 3.2 关键差异

| 维度 | 相同点（与主流一致） | 本模块刻意简化 | 目前缺什么（诚实告知） |
|---|---|---|---|
| 对话方法命名 | `invoke` 返回完整响应 | 不做 `ainvoke` / `batch` / `stream` 的 SSE 对象抽象，流式直接 yield str | 异步、结构化输出（response_format）、工具调用严格模式 |
| Embedding 命名 | `embed_query` / `embed_documents`（取 LangChain） | 不抽象 `Embeddings` 基类、不做本地模型适配器 | 异步 `aembed_*`、本地 embedding（sentence-transformers）、缓存 |
| Provider 支持 | OpenAI 协议全兼容（任意 base_url 中转） | 不做 Anthropic / Gemini 原生 SDK 适配 | Anthropic 原生、Gemini 原生、多 provider 自动降级 |
| 工程能力 | 超时可配、异常可 catch | 不做自动重试/熔断/限流/全链路 trace | 指数退避重试、熔断、Token 用量统计、TraceLogger 打点 |

**一句话收口**：本实现 = **OpenAI 官方 SDK 的原生调用** + **LangChain 的命名约定（`embed_query` / `embed_documents`）** + **门面薄封装（环境变量 + provider 分发 + 分片并发）**，刻意不引入重型抽象。

---

## 四、模块架构

### 4.1 目录结构

```
fast_agents/llm/                ← 单数目录，对齐 tools/ memory/ skills/ mcp/
├── __init__.py                 模块出口，统一导出 LLM
├── llm.py                      门面类 LLM：读环境变量、选 client、透传方法
└── openai_client.py            OpenAIClient（继承官方 OpenAI）：实际 HTTP 调用，含 chat 与 embedding 两套接口
```

全部为既有文件，本次**只修改、不新增**文件：在 `openai_client.py` 内新增 embedding 实现与分片并发逻辑，在 `llm.py` 内新增透传方法与 `embedding_model` 参数。

### 4.2 分层

```
┌──────────────────────────────────────────────┐
│  上层调用方（agents / memory / tools / ...）  │
│  只依赖 fast_agents.llm.LLM 这一个入口         │
└──────────────────┬───────────────────────────┘
                   │  LLM.invoke / stream_invoke /
                   │  embed_query / embed_documents
                   ▼
┌──────────────────────────────────────────────┐
│  门面层：llm.py · LLM                         │
│  - 读取环境变量（LLM_MODEL_ID / LLM_API_KEY /  │
│    LLM_BASE_URL / LLM_EMBED_MODEL / timeout） │
│  - 根据 provider 分发到具体 client            │
│  - 方法全部代理给 self._client                │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  实现层：openai_client.py · OpenAIClient      │
│  - 继承官方 openai.OpenAI，直接用它的 HTTP 栈  │
│  - invoke：chat.completions.create（非流式）  │
│  - stream_invoke：chat.completions.create（流）│
│  - embed_query：embeddings.create（单条）     │
│  - embed_documents：embeddings.create（分片+并发）│
└──────────────────────────────────────────────┘
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

箭头旁标注的就是源码里的真实方法名，可直接顺藤摸瓜。

### 4.4 与 ReActAgent 的接入点 / 职责边界

- **接入方式**：`ReActAgent.__init__` 接收 `llm: LLM` 参数并保存为成员；所有对模型的调用（包括思考、tool_calls 解析、滚动摘要、未来的 embedding）都走 `self.llm`，不允许直接 new 一个 OpenAI client；
- **单一数据源约定**：对话历史由 `memory.BufferMemory` 维护，LLM 不持有任何会话状态——它是纯函数式的「输入消息 → 输出消息/向量」，保证可替换、可 mock、可测试；
- **边界清晰**：LLM 不做 token 计数（归 context 模块）、不做重试熔断（归 infra 模块）、不做向量存储（归 memory / RAG 工具）——它只负责「把东西发出去、把结果拿回来」。

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
| `provider` | `str` | `"openai"` | 预留 provider 分发位；目前所有值都走 OpenAI 兼容协议 | 否 |
| `timeout` | `Optional[int]` | `180`（client 层默认） | 请求超时秒数 | 否 |
| `embedding_model` | `Optional[str]` | `$LLM_EMBED_MODEL`，否则 `"text-embedding-3-small"` | Embedding 模型名，与 chat 模型独立 | 否 |

### 6.2 方法签名（核心契约）

```python
class LLM:
    # ---- Chat ----
    def invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Message: ...
    def stream_invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Iterator[str]: ...
    # ---- Embeddings ----
    def embed_query(self, text: str, **kwargs) -> list[float]: ...
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]: ...
```

**命名约定锁死**：
- 对话方法用 `invoke` / `stream_invoke`——对齐工具模块的 `ToolBase.invoke()` 命名风格（动词 `invoke` 是本框架「一次调用」的统一动词）；
- Embedding 方法用 `embed_query` / `embed_documents`——对齐 LangChain，**不使用** `embed` / `encode` / `get_embedding` 等模糊命名；
- 返回原生 `list[float]`——不引入 numpy / torch 依赖，下游要算余弦相似度自己用标准库 `math` 即可（参考 `memory/scoring.py` 全标准库实现的风格）。

---

## 七、逐个文件讲解（源码导读）

### 7.1 `__init__.py` —— 模块出口

```python
"""LLM模块"""
from .llm import LLM

__all__ = [
    "LLM",
]
```

**为什么这么写**：和仓库其他模块保持一致（`tools/__init__.py` 导出 `ToolRegistry` 等、`memory/__init__.py` 导出 `MemoryManager`），对外永远只暴露一个门面类，内部 client 类不进 `__all__`——上层不应该直接 `from fast_agents.llm import OpenAIClient`。如果需要换 client，在 `llm.py` 的 `_create_client()` 里改即可，不影响上层 import。

### 7.2 `llm.py` —— LLM 门面

```python
class LLM:
    def __init__(self, model=None, api_key=None, base_url=None,
                 provider="openai", timeout=None, embedding_model=None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("LLM_EMBED_MODEL")
        ...
        self._client = self._create_client()

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return self._client.embed_query(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self._client.embed_documents(texts, **kwargs)
```

**关键实现要点**：
- **环境变量 vs 构造参数**：用 `or` 实现「构造参数优先，环境变量兜底」。注意 `or` 对空字符串同样 fallback，这是刻意的——如果用户显式传 `model=""`，视为没传、走环境变量；
- **provider 分发位**：`_create_client()` 虽然目前所有分支都返回 `OpenAIClient`，但 `if self.provider == "openai"` / `else` 的分支结构先搭好，将来加 Anthropic client 时只需加一个 `elif self.provider == "anthropic"` 分支，门面类的构造签名和方法签名都不动；
- **方法透传**：所有对外方法都是「薄代理」——拿到 kwargs 直接透给 client，不在门面做业务逻辑。这样的好处是：门面永远不需要因为底层参数调整而改。

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

`invoke` / `stream_invoke` 是已有代码，本次改动只加了一行：

```python
chat_model = kwargs.pop("model", self.model)
response = self.chat.completions.create(model=chat_model, ...)
```

**为什么允许通过 kwargs 覆盖 model**：上层偶尔需要对单次调用切换模型（比如摘要用小模型、复杂推理用大模型），不应该为此 new 一个新 LLM 实例。`kwargs.pop` 保证这个参数不会被重复传给底层 SDK（SDK 签名里 `model` 是位置/关键字参数，重复传会报错）。

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

def _one_batch(batch: list[str]) -> list[list[float]]:
    resp = self.embeddings.create(model=embed_model, input=batch, **kwargs)
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [list(d.embedding) for d in ordered]

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

**空串处理**：空串替换为 `" "`（单空格）而不是丢弃——因为 RAG/记忆入库时，输入 texts 通常和一批 chunk 对象一一对应，如果中间丢弃空串会导致返回向量数量和输入不一致，上层索引错位；单空格的 embedding 是一个合法向量，且和任何语义文本的相似度都极低，不会污染检索结果。

**要自己改时最该看的地方**：接 BGE / E5 / DashScope v3 等非对称模型时，就在 `embed_query` 和 `embed_documents` 内部分别加前缀或 `input_type` 参数，对外签名不变。加重试（指数退避）、加缓存（按 text hash 缓存向量）也在这个文件里做。

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
OK, methods: ['embed_documents', 'embed_query', 'invoke', 'stream_invoke']
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

### 8.3 接到 ReActAgent 上

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

### 8.4 典型用法

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

### 8.5 跑自测

直接运行 client 文件末尾的 `__main__` 块，会依次跑流式对话 + `embed_query` + `embed_documents`：

```powershell
python fast_agents\llm\openai_client.py
```

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
    provider="openai",          # 预留，目前都走 OpenAI 兼容
    timeout=None,               # 默认 180 秒（client 层）
    embedding_model=None,       # embedding 模型，默认读 $LLM_EMBED_MODEL，否则 text-embedding-3-small
)
```

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `invoke(messages, tools=None, **kwargs)` | `messages: list[dict]`（OpenAI message 格式），`tools`（OpenAI tools schema 列表） | `Message`（含 `role`/`content`/`tool_calls`） | 非流式对话，kwargs 可覆盖 `model`/`temperature`/`max_tokens` |
| `stream_invoke(messages, tools=None, **kwargs)` | 同上 | `Iterator[str]` | 流式对话，逐段 yield 正文（思考片段打印到控制台但不进入 yield） |
| `embed_query(text, **kwargs)` | `text: str` | `list[float]` | 单条查询向量；kwargs 可覆盖 `model`（embedding 模型名） |
| `embed_documents(texts, **kwargs)` | `texts: list[str]` | `list[list[float]]` | 批量文档向量；内部按 2048 分片并发；顺序与输入严格一致；kwargs 可覆盖 `model` |

### 9.2 OpenAIClient（底层，一般不直接使用）

继承官方 `openai.OpenAI`，拥有 SDK 全部能力（`chat.completions` / `embeddings` / `files` / 等）。在本模块自定义的公开方法上，签名与 `LLM` 完全一致——`LLM` 只是代理。

### 9.3 已废弃/预留

- 目前没有已废弃的旧写法；
- `provider` 参数的非 `"openai"` 分支目前和 `"openai"` 行为一致，预留作未来 Anthropic / Gemini 原生分发。

---

## 十、设计决策与常见坑

### 10.1 关键设计决策（为什么）

1. **为什么继承而不是组合官方 `OpenAI` 类？**
   官方 SDK 的 `OpenAI` 类已经封装了连接池、httpx client、auth、重试、流式解析等大量细节；继承它比组合它少写几十个方法代理，且能直接调用 `self.embeddings.create(...)` / `self.chat.completions.create(...)`，可读性更好。风险点是 SDK 大版本升级可能改内部实现，但 openai SDK 从 v1 起公开 API 稳定，风险可控。

2. **为什么 `embed_query` 收 `str` 而不是 `list[str]`？**
   方法名里的 `query` 是单数——查询就是「用户的一个问题」，不是一批问题。批量查询在工程上非常罕见（RAG 是一次 query 对一批 doc）；批量能力放在 `embed_documents`。如果上层真的需要批量 query，用 `embed_documents` 即可——底层实现完全能跑，只是语义上要知道「query 前缀/input_type」不会被自动加上（将来加非对称模型时这里要注意）。

3. **为什么不默认做重试/熔断？**
   重试策略（退避基数、最大次数、对哪些状态码重试）和熔断阈值是强场景相关的：批处理入库可以激进重试、实时对话要快速失败。过早内置一个「万能重试」反而会让上层想自定义时绕不开。重试属于 infra 层横切能力，留给 G1 `TraceLogger` + G2 异步并发一起做。

4. **为什么不在 LLM 里做 embedding 缓存？**
   缓存的 key 是 `(model, text_hash)`，但缓存存储位置（内存/磁盘/redis）和失效策略是上层场景决定的（短期会话可以内存缓存、跨进程需要磁盘、多租户需要 redis）。LLM 层做缓存意味着强绑一种策略，反而阻碍上层。将来可以加一个可选的 `cache:` 构造参数，传入一个有 `get/set` 方法的对象即可，默认 None。

5. **为什么 chat 模型没有硬编码默认值，embedding 模型有？**
   chat 模型的选择直接决定回复质量、价格、速度，差异极大（几元到几百元每百万 token），硬选默认会让没配环境变量的用户在跑完才发现用错模型、产生意料外费用；embedding 模型相对稳定（`text-embedding-3-small` 便宜、足够快、维度统一），且嵌入质量稍差只会影响检索召回率，不会产生安全/费用风险，所以给一个开箱即用的默认是合适的。

### 10.2 常见坑与排查

| 现象 | 原因 / 解决 |
|---|---|
| `Exception: LLM调用失败: ... 401 ...` | API Key 错或过期。检查 `.env` 的 `LLM_API_KEY`，或显式传 `api_key=` |
| `Exception: LLM调用失败: ... model not found` | chat 模型名错（第三方中转不一定支持 `gpt-4o-mini`），或 embedding 模型名在第三方端点不可用。第三方中转要显式传 `embedding_model=` 指定该平台支持的 embedding 模型名 |
| `Exception: Embedding 调用失败（documents）: ... maximum context length ...` / `400 input length exceeded` | 单条文本过长（OpenAI text-embedding-3 单条上限约 8191 token）。RAG 场景必须先分 chunk，不能把整篇文档直接塞给 `embed_documents` |
| `embed_documents` 返回顺序和输入对不上 | 老版本没按 `data[*].index` 排序会偶发；请确保使用本版本 |
| 中文乱码 / Windows 控制台打印乱码 | 在脚本开头加 `import sys; sys.stdout.reconfigure(encoding='utf-8')`，或 PowerShell 先执行 `chcp 65001` |
| 调用很慢、批量 embed 耗时明显 | 单次 batch 内部已并发；若总文本量很大（> 1万条），建议调用方自己分批分段入库，控制并发 |
| 第三方中转端点 embedding 路径不是 `/v1/embeddings` | 少数中转端点改了路径；这种情况只能换端点或手动改 `base_url`，OpenAI SDK 没有路径重写开关 |

### 10.3 跨平台注意事项（Windows 专项）

1. **换行符**：本模块所有 `.py` 文件均为 CRLF（`\r\n`），直接复制落盘即可，Windows Git 若开了 `autocrlf` 也不会重复转；
2. **`.env` 文件编码**：Windows 下用记事本保存 `.env` 时要选 **UTF-8**（不要选 UTF-16 或 ANSI），否则 `python-dotenv` 读取中文会乱码；
3. **控制台编码**：`stream_invoke` 里有 print 中文字符，PowerShell 默认 GBK 编码可能乱码，启动前执行 `chcp 65001` 或在代码里 `sys.stdout.reconfigure(encoding='utf-8')`；
4. **路径空格**：若 `.env` 放在含空格的路径下，`load_dotenv()` 自动向上查找是正常的，但不要在 `base_url` 里拼 Windows 本地路径（`base_url` 是 URL 不是文件路径）；
5. **代理**：公司内网需要走 HTTP 代理时，`openai` SDK 会自动读 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量，不需要额外配置。

---

## 十一、测试与验证

本模块当前没有独立的 `test/test_llm.py`（仓库基线无此文件），自测通过以下方式：

| 验证项 | 方式 | 预期结果 |
|---|---|---|
| 模块导入 | `python -c "from fast_agents.llm import LLM"` | 无报错，`LLM` 可导入 |
| 方法挂载 | `[m for m in dir(LLM) if not m.startswith('_')]` | 含 `invoke` / `stream_invoke` / `embed_query` / `embed_documents` 四个方法 |
| 语法检查 | `ast.parse` 三个文件 | 无 SyntaxError |
| 真实 API 联通 | `python fast_agents\llm\openai_client.py` | 流式打印对话结果，输出两个向量的维度信息 |
| 空串防御 | 传入 `embed_query("")` / `embed_query("   ")` | 抛出 `ValueError`，不发请求 |
| 类型防御 | 传入 `embed_query(123)` / `embed_documents("not a list")` | 抛出 `TypeError` |
| 分片并发 | `embed_documents` 传 3000 条短文本（需 key） | 返回 3000 个向量，顺序与输入一一对应 |
| 下游接入 | ReActAgent 构造时传 `llm=LLM()` | 既有对话/工具调用流程不受影响 |

**关键回归断言（如果后续补 `test_llm.py`，这几条必须锁住）**：

1. `embed_query("")` 必须抛 `ValueError`，不能发 HTTP 请求（防止被计费）；
2. `embed_documents([])` 返回 `[]`，不报错；
3. `embed_documents(["a", "", "b"])` 返回长度为 3 的列表（空串不被丢弃，防止顺序错位）；
4. 传入 `model=` kwarg 能正确覆盖默认模型（通过 monkeypatch `embeddings.create` 断言入参）；
5. `LLM` 门面的四个方法都把 kwargs 透传给 client，不在中间吃掉参数。

---

## 十二、下一步演进路线

按优先级排（对齐《演进路线图.md》）：

1. **[G2 异步] `aembed_query` / `aembed_documents` / `ainvoke`** —— 现状：所有方法是同步的，靠线程池拿到有限并发；原因：上游 memory/tools/agents 也都是同步的，先打通语义检索闭环比异步优先级高；待 infra G2 异步并发启动时一起做；
2. **[provider 扩展] Anthropic / Gemini 原生适配** —— 现状：所有 provider 都靠 OpenAI 兼容协议；原因：国内端点基本都兼容 OpenAI 协议，原生适配需求不强；待有明确不兼容的端点需求时再加；
3. **[本地模型] LocalEmbeddingClient（sentence-transformers）** —— 现状：只能调云端 API；原因：sentence-transformers 依赖 torch（~2GB），违反「零重型依赖」原则；可以作为可选 extra（`pip install fast-agents[local]`），不做默认依赖；
4. **[G1 可观测] Token 用量统计 + TraceLogger 打点** —— 现状：调用后不统计 token 用量、不打 trace；原因：需要 infra G1 先落 `TraceLogger`；届时在 `invoke` / `embed_*` 三个方法入口加装饰器，记录入参、耗时、token、响应状态，落 jsonl；
5. **[健壮性] 指数退避重试 + 熔断** —— 现状：无重试，网络抖动直接抛错；原因：和 infra G2/G1 一起做更合适，需要全局的重试策略配置位；
6. **[缓存] 可选 embedding 缓存** —— 现状：相同文本重复调用会重复请求、重复计费；原因：缓存后端选择（内存/磁盘/redis）依赖场景，先不内置；可以通过「LLM 接受可选 cache 参数」实现可插拔。

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
- [x] 有更新历史，且破坏性变更都给了迁移办法（本次是纯新增，无破坏）；
- [x] 第二章讲清了四类能力分层与 6 条设计原则；
- [x] 第三章包含对比总表 5 家 + 异同表（含「缺少什么」）；
- [x] 第四章有目录结构（单数目录 + 逐文件职责）、分层图、两张调用链路图、与 Agent 接入边界；
- [x] 第五章（弹性节）讲清了「为什么 Chat/Embedding 在同一个门面」+「为什么分 query/documents 两个方法」；
- [x] 第六章数据模型字段命名与仓库既有约定一致（`invoke` 动词、门面类单名 `LLM`）；
- [x] 源码导读三个文件都有「负责什么 / 关键实现 / 为什么」；
- [x] 快速上手命令是 Windows PowerShell，且提醒了 `test/` 与标准库同名问题（虽然本模块暂无 test 文件）；
- [x] API 速查完整列出构造参数与四个方法；
- [x] 第十章三块齐全：5 条设计决策、7 条常见坑、5 条 Windows 专项；
- [x] 测试章节列出 8 项验证点 + 5 条关键回归断言，如实说明当前没有独立 test_llm.py；
- [x] 演进路线列出 6 条待办，与《演进路线图.md》G 系列编号对齐；
- [x] **章节数量、顺序、编号与其余四份文档完全一致**（开篇→一～十三→附录）；
- [x] 全文无「待补充 / TBD」残留；
- [x] 所有 `.py` 文件已转换为 CRLF 换行符。
