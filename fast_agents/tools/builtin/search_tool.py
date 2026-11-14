"""联网搜索工具（DuckDuckGo / Tavily）"""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=20, description="返回结果条数（1-20，默认 5）")


class SearchTool(ToolBase):
    name = "web_search"
    description = "联网搜索网页，返回标题、链接和摘要。用于获取实时信息、最新新闻、未知事实、文档查询等。"
    args_schema = SearchInput
    tags = ["builtin", "search"]

    # provider 选型：
    # - "duckduckgo"（默认）：免 API Key，开箱即用；依赖 `duckduckgo-search`。
    # - "tavily"：结果质量更稳、更贴合问答，但需要 API Key；依赖 `tavily-python`。
    #   Key 可显式传入（api_key="..."），也可通过环境变量 TAVILY_API_KEY 提供。

    def __init__(self, provider: Literal["duckduckgo", "tavily"] = "duckduckgo",
                 api_key: Optional[str] = None, max_output_chars: int = 8000):
        super().__init__(max_output_chars=max_output_chars)
        self.provider = provider
        self.api_key = api_key

    def run(self, query: str, max_results: int = 5) -> Union[str, ToolResult]:
        if not query.strip():
            return ToolResult.fail("搜索关键词不能为空")
        if self.provider == "tavily":
            # 前置检查：缺依赖 / 缺 Key 时给出明确指引，避免被下方通用 except
            # 笼统包成「搜索失败：...」而看不出真正原因。
            guide = self._tavily_precheck()
            if guide is not None:
                return ToolResult.fail(guide)
        try:
            if self.provider == "duckduckgo":
                results = self._ddg(query, max_results)
            else:
                results = self._tavily(query, max_results)
        except ImportError as e:
            pkg = "duckduckgo-search" if self.provider == "duckduckgo" else "tavily-python"
            return ToolResult.fail(f"缺少依赖：{e.name}。请安装：pip install {pkg}")
        except Exception as e:
            return ToolResult.fail(f"搜索失败：{e}")
        if not results:
            return ToolResult.ok("未找到相关结果，可以尝试换一个关键词。")
        lines = [f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)]
        return ToolResult.ok("\n\n".join(lines), artifact={"query": query, "results": results})

    def _tavily_precheck(self) -> Optional[str]:
        """Tavily provider 前置检查。

        返回 None 表示可以继续执行；返回字符串表示不可用，字符串即为给用户的提示。
        依次检查：依赖是否安装 → API Key 是否就绪。
        """
        try:
            import tavily  # noqa: F401
        except ImportError as e:
            return f"缺少依赖：{e.name}。请安装：pip install tavily-python"
        if not (self.api_key or os.environ.get("TAVILY_API_KEY")):
            return ("未配置 Tavily API Key：请设置环境变量 TAVILY_API_KEY，"
                    "或构造 SearchTool(provider='tavily', api_key='...')；"
                    "免 Key 场景可改用 provider='duckduckgo'。")
        return None

    @staticmethod
    def _ddg(query: str, max_results: int) -> List[dict]:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw]

    def _tavily(self, query: str, max_results: int) -> List[dict]:
        from tavily import TavilyClient
        api_key = self.api_key or os.environ.get("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        resp = client.search(query, max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in resp.get("results", [])]
