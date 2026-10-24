import os
from typing import Optional

from .openai_client import OpenAIClient
from ..messages.message import Message



class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: Optional[int] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout
        self.provider = provider
        self._client = self._create_client()

    def _create_client(self):
        """创建OpenAI客户端"""
        if self.provider == "openai":
            return OpenAIClient(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            return OpenAIClient(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Message:
        return self._client.invoke(messages=messages, tools=tools, **kwargs)

    def stream_invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> str:
        return self._client.stream_invoke(messages=messages, tools=tools, **kwargs)