"""联网搜索工具（DuckDuckGo / Tavily）"""

from __future__ import annotations

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

    def __init__(self, provider: Literal["duckduckgo", "tavily"] = "duckduckgo",
                 api_key: Optional[str] = None, max_output_chars: int = 8000):
        super().__init__(max_output_chars=max_output_chars)
        self.provider = provider
        self.api_key = api_key

    def run(self, query: str, max_results: int = 5) -> Union[str, ToolResult]:
        if not query.strip():
            return ToolResult.fail("搜索关键词不能为空")
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

    @staticmethod
    def _ddg(query: str, max_results: int) -> List[dict]:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw]

    def _tavily(self, query: str, max_results: int) -> List[dict]:
        from tavily import TavilyClient
        client = TavilyClient(api_key=self.api_key)
        resp = client.search(query, max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in resp.get("results", [])]
