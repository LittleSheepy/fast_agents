"""LLM 模块异常体系（基类 + 具体子类，文本生成与 Embedding 共用）。

参考主流框架与同仓库规范：
- OpenAI Python SDK 的 ``openai.APIError`` 层级——按「连接/超时/限流/认证/API 错误」
  拆分，天然对应「要不要重试」的决策；
- 同仓库 ``database/errors.py``、``mcp/errors.py``——「基类 + 具体子类 +
  集中放在模块自己的 errors.py」的组织方式，不把底层 SDK 原生异常直接抛给上层。

设计要点：
- :class:`LLMError` **继承 ``Exception``**（而非 ``ValueError``）：LLM 错误涵盖连接失败、
  超时、限流、鉴权失败、响应格式异常等，语义上并非「值不合法」；独立谱系可避免与
  无关的 ``except ValueError`` 相互误捕。上层统一用 ``except LLMError`` 或其子类分流。
- 每个具体异常带类属性 ``retryable: bool``，直接供 I4（重试 / 熔断）判断：
  * 可重试：连接失败、超时、限流、服务端 5xx、临时过载 → ``retryable=True``
  * 不可重试：认证失败、配置错误、参数非法、模型不存在 → ``retryable=False``
- 统一保留原始异常链（``raise Xxx(...) from e``），底层 SDK 异常被包裹但不丢失栈信息；
  上层如需访问原始异常，可通过 ``__cause__`` 拿到。
- 不吞掉上层传参错误（如把 int 当 prompt 传入产生的 ``TypeError``），这类错误仍走原生
  异常，不纳入本体系。
"""
from __future__ import annotations


class LLMError(Exception):
    """LLM 模块异常基类。

    继承 ``Exception``（不挂 ``ValueError``）：LLM 错误多为网络、鉴权、服务端类问题，
    语义上不是「值不合法」，独立谱系可避免与无关的 ``except ValueError`` 相互误捕。
    上层统一捕获本类或其具体子类来分流；所有子类默认 ``retryable=False``，可重试
    子类显式覆盖为 ``True``。
    """

    retryable: bool = False

    def __init__(self, message: str, *, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


# ---------- 连接 / 网络层（可重试） ----------


class LLMConnectionError(LLMError):
    """连接层：无法连接到 LLM 服务端点（DNS 失败、连接被拒、TLS 握手失败等）。"""

    retryable: bool = True


class LLMTimeoutError(LLMError):
    """连接层：请求超时（连接超时、读取超时）。"""

    retryable: bool = True


# ---------- 限流 / 服务端临时错误（可重试） ----------


class LLMRateLimitError(LLMError):
    """服务端：触发限流（429）、配额耗尽。通常退避后可重试。"""

    retryable: bool = True


class LLMServerError(LLMError):
    """服务端：5xx 错误（服务端过载、内部错误、维护中）。"""

    retryable: bool = True


# ---------- 认证 / 配置 / 权限（不可重试） ----------


class LLMAuthenticationError(LLMError):
    """鉴权层：API Key 无效、签名错误、权限不足（401/403）。"""

    retryable: bool = False


class LLMConfigError(LLMError):
    """配置层：base_url / model 等配置缺失或非法、SDK 初始化失败。"""

    retryable: bool = False


class LLMModelNotFoundError(LLMConfigError):
    """配置层：请求的模型不存在或无权访问（404 model_not_found）。"""

    retryable: bool = False


# ---------- API / 响应层 ----------


class LLMBadRequestError(LLMError):
    """API 层：请求参数非法（400），如未知字段、上下文超长等。"""

    retryable: bool = False


class LLMApiError(LLMError):
    """API 层：其它未分类的服务端返回错误（含响应解析失败）。"""

    retryable: bool = False


class LLMResponseError(LLMError):
    """响应层：服务端返回格式不符合预期（缺字段、JSON 解析失败等）。"""

    retryable: bool = False


# ---------- 业务层（按方法区分） ----------


class LLMInvokeError(LLMError):
    """业务层：文本生成（invoke / stream_invoke）过程中的错误包装。"""

    retryable: bool = False  # 具体可重试性由内部原始异常决定，见 __cause__


class LLMStructuredOutputError(LLMError):
    """业务层：结构化输出（invoke_structured）过程中的错误包装。

    涵盖 JSON 解析失败、pydantic 校验不通过、端点不支持 response_format 等场景。
    retryable=False —— 格式/校验错误重试不会改变结果，应由上层调整 prompt 或 schema。
    """

    retryable: bool = False


class LLMEmbeddingError(LLMError):
    """业务层：文本向量化（embed_query / embed_documents）过程中的错误包装。"""

    retryable: bool = False  # 同上
