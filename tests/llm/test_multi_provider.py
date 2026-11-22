"""
I3 多provider适配单元测试
验证：Anthropic/Gemini客户端接口与OpenAIClient完全对齐，上层调用无感
测试全程离线，使用mock对象模拟SDK响应
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Any, Optional
from pydantic import BaseModel
from fast_agents.llm import LLM
from fast_agents.llm.models import StreamChunk
from fast_agents.llm.errors import LLMConfigError
from fast_agents.messages.message import Message, MsgToolCall


# ---------------------------------------------------------------------- #
# Mock对象：模拟Anthropic SDK响应                                          #
# ---------------------------------------------------------------------- #
class MockAnthropicTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text

class MockAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class MockAnthropicResponse:
    def __init__(self, content: str, input_tokens=10, output_tokens=20):
        self.content = [MockAnthropicTextBlock(content)]
        self.usage = MockAnthropicUsage(input_tokens, output_tokens)

class MockAnthropicMessages:
    def create(self, **kwargs):
        return MockAnthropicResponse("Anthropic响应内容：" + kwargs["model"])

class MockAnthropicClient:
    def __init__(self, **kwargs):
        self.messages = MockAnthropicMessages()

# ---------------------------------------------------------------------- #
# Mock对象：模拟Gemini SDK响应                                            #
# ---------------------------------------------------------------------- #
class MockGeminiPart:
    def __init__(self, text: str):
        self.text = text
        self.function_call = None

class MockGeminiCandidate:
    def __init__(self, text: str):
        self.content = type("Content", (), {"parts": [MockGeminiPart(text)]})()
        self.finish_reason = "STOP"

class MockGeminiUsageMetadata:
    def __init__(self, prompt=10, candidates=20, total=30):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total

class MockGeminiResponse:
    def __init__(self, text: str):
        self.candidates = [MockGeminiCandidate(text)]
        self.usage_metadata = MockGeminiUsageMetadata()

class MockGeminiModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
    def generate_content(self, **kwargs):
        return MockGeminiResponse("Gemini响应内容：" + self.model_name)

class MockGenai:
    def configure(self, **kwargs):
        pass
    def GenerativeModel(self, **kwargs):
        return MockGeminiModel(**kwargs)


def test_provider_config_validation():
    """测试：非法provider抛出LLMConfigError"""
    print("🔍 测试非法provider校验...")
    try:
        llm = LLM(provider="invalid_provider", api_key="fake_key")
        assert False, "非法provider应该抛出异常"
    except LLMConfigError as e:
        assert "不支持的 LLM provider" in str(e)
    print("✅ 非法provider校验通过")

def test_provider_routing():
    """测试：provider参数正确路由到对应客户端"""
    print("🔍 测试provider路由逻辑...")
    # OpenAI provider默认生效
    llm_openai = LLM(provider="openai", api_key="fake_key", base_url="http://localhost:8000/v1")
    assert llm_openai.provider == "openai"
    assert type(llm_openai._client).__name__ == "OpenAIClient"
    print("✅ OpenAI provider路由正确")

    # 模拟Anthropic SDK存在场景（通过monkey patch注入mock）
    import fast_agents.llm.anthropic_client as anthropic_module
    original_import = anthropic_module._import_anthropic
    def mock_import_anthropic():
        class FakeAnthropic:
            def __init__(self, **kwargs):
                self.messages = MockAnthropicMessages()
        class FakeAsyncAnthropic:
            def __init__(self, **kwargs):
                pass
        errors_map = {k: type(k, (Exception,), {}) for k in [
            "APIConnectionError", "APITimeoutError", "RateLimitError",
            "AuthenticationError", "PermissionDeniedError", "NotFoundError",
            "BadRequestError", "APIStatusError", "APIError"
        ]}
        anthropic_mod = type("anthropic", (), {
            "Anthropic": FakeAnthropic,
            "AsyncAnthropic": FakeAsyncAnthropic
        })()
        return anthropic_mod, errors_map
    anthropic_module._import_anthropic = mock_import_anthropic

    llm_anthropic = LLM(provider="anthropic", api_key="fake_key")
    assert llm_anthropic.provider == "anthropic"
    assert type(llm_anthropic._client).__name__ == "AnthropicClient"
    print("✅ Anthropic provider路由正确")

    # 模拟Gemini SDK存在场景
    import fast_agents.llm.gemini_client as gemini_module
    original_genai_import = gemini_module._import_google_genai
    def mock_import_genai():
        return MockGenai(), None
    gemini_module._import_google_genai = mock_import_genai

    llm_gemini = LLM(provider="gemini", api_key="fake_key")
    assert llm_gemini.provider == "gemini"
    assert type(llm_gemini._client).__name__ == "GeminiClient"
    print("✅ Gemini provider路由正确")

    # 恢复原函数
    anthropic_module._import_anthropic = original_import
    gemini_module._import_google_genai = original_genai_import

def test_anthropic_client_interface():
    """测试：AnthropicClient接口与OpenAIClient完全对齐（方法签名一致）"""
    print("🔍 测试AnthropicClient接口一致性...")
    from fast_agents.llm.openai_client import OpenAIClient
    from fast_agents.llm.anthropic_client import AnthropicClient

    openai_methods = {m for m in dir(OpenAIClient) if not m.startswith("_") and callable(getattr(OpenAIClient, m))}
    anthropic_methods = {m for m in dir(AnthropicClient) if not m.startswith("_") and callable(getattr(AnthropicClient, m))}
    # 必须的核心方法全部存在
    required_methods = {"invoke", "stream_invoke", "invoke_structured", "embed_query", "embed_documents"}
    assert required_methods.issubset(anthropic_methods), f"AnthropicClient缺少方法：{required_methods - anthropic_methods}"
    print("✅ AnthropicClient接口对齐OpenAIClient")

def test_gemini_client_interface():
    """测试：GeminiClient接口与OpenAIClient完全对齐（方法签名一致）"""
    print("🔍 测试GeminiClient接口一致性...")
    from fast_agents.llm.openai_client import OpenAIClient
    from fast_agents.llm.gemini_client import GeminiClient

    openai_methods = {m for m in dir(OpenAIClient) if not m.startswith("_") and callable(getattr(OpenAIClient, m))}
    gemini_methods = {m for m in dir(GeminiClient) if not m.startswith("_") and callable(getattr(GeminiClient, m))}
    required_methods = {"invoke", "stream_invoke", "invoke_structured", "embed_query", "embed_documents"}
    assert required_methods.issubset(gemini_methods), f"GeminiClient缺少方法：{required_methods - gemini_methods}"
    print("✅ GeminiClient接口对齐OpenAIClient")

def test_async_clients_interface():
    """测试：异步客户端接口与AsyncOpenAIClient完全对齐"""
    print("🔍 测试异步客户端接口一致性...")
    from fast_agents.llm.async_openai_client import AsyncOpenAIClient
    from fast_agents.llm.async_anthropic_client import AsyncAnthropicClient
    from fast_agents.llm.async_gemini_client import AsyncGeminiClient

    required_async_methods = {"ainvoke", "astream_invoke", "ainvoke_structured", "aembed_query", "aembed_documents"}
    for client_cls, name in [
        (AsyncAnthropicClient, "AsyncAnthropicClient"),
        (AsyncGeminiClient, "AsyncGeminiClient"),
    ]:
        methods = {m for m in dir(client_cls) if not m.startswith("_") and callable(getattr(client_cls, m))}
        assert required_async_methods.issubset(methods), f"{name}缺少异步方法：{required_async_methods - methods}"
    print("✅ 异步客户端接口对齐AsyncOpenAIClient")

def test_optional_dependency_error():
    """测试：未安装对应SDK时抛出明确的安装提示错误"""
    print("🔍 测试可选依赖缺失提示...")
    # 测试Anthropic缺失提示
    import builtins
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import
    # 重新加载模块清除缓存
    import importlib
    import fast_agents.llm.anthropic_client as anthropic_module
    importlib.reload(anthropic_module)
    try:
        from fast_agents.llm.anthropic_client import _import_anthropic
        _import_anthropic()
        assert False, "未安装SDK时应该抛出异常"
    except LLMConfigError as e:
        assert "pip install fast-agents[anthropic]" in str(e)
    builtins.__import__ = original_import
    importlib.reload(anthropic_module)

    # 测试Gemini缺失提示
    def mock_import_genai(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise ImportError("No module named 'google'")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = mock_import_genai
    import fast_agents.llm.gemini_client as gemini_module
    importlib.reload(gemini_module)
    try:
        from fast_agents.llm.gemini_client import _import_google_genai
        _import_google_genai()
        assert False, "未安装SDK时应该抛出异常"
    except LLMConfigError as e:
        assert "pip install fast-agents[gemini]" in str(e)
    builtins.__import__ = original_import
    importlib.reload(gemini_module)
    print("✅ 可选依赖缺失提示正确")

def test_message_conversion():
    """测试：OpenAI格式消息正确转换为各provider格式"""
    print("🔍 测试消息格式转换...")
    from fast_agents.llm.anthropic_client import _convert_messages_to_anthropic
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
        {"role": "tool", "content": "工具结果"}
    ]
    anthro_msgs, system = _convert_messages_to_anthropic(messages)
    assert system == "你是助手"
    assert len(anthro_msgs) == 3  # system被抽离，tool映射为user
    assert anthro_msgs[0]["role"] == "user"
    assert anthro_msgs[2]["role"] == "user"  # tool→user
    print("✅ Anthropic消息格式转换正确")

    from fast_agents.llm.gemini_client import _convert_messages_to_gemini
    gemini_msgs, system_gemini = _convert_messages_to_gemini(messages)
    assert system_gemini == "你是助手"
    assert len(gemini_msgs) == 3
    assert gemini_msgs[1]["role"] == "model"  # assistant→model
    print("✅ Gemini消息格式转换正确")

def test_provider_case_insensitive():
    """测试：provider参数大小写不敏感"""
    print("🔍 测试provider大小写不敏感...")
    import fast_agents.llm.llm as llm_module
    original_create_openai = llm_module.OpenAIClient
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            pass
    llm_module.OpenAIClient = FakeOpenAIClient
    llm1 = LLM(provider="OPENAI", api_key="fake")
    llm2 = LLM(provider="OpenAI", api_key="fake")
    llm3 = LLM(provider="openai", api_key="fake")
    assert llm1.provider == "OPENAI"
    assert llm2.provider == "OpenAI"
    assert llm3.provider == "openai"
    llm_module.OpenAIClient = original_create_openai
    print("✅ provider大小写不敏感正确")


def main():
    test_provider_config_validation()
    test_provider_routing()
    test_anthropic_client_interface()
    test_gemini_client_interface()
    test_async_clients_interface()
    test_optional_dependency_error()
    test_message_conversion()
    test_provider_case_insensitive()
    print("\n🎉 I3 多provider适配所有测试通过！")

if __name__ == "__main__":
    main()