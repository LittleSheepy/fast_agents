"""
I3 多provider使用示例
演示：OpenAI/Anthropic/Gemini三种provider的初始化与调用方式，上层API完全一致
"""
import os
import asyncio
from dotenv import load_dotenv
from fast_agents.llm import LLM

load_dotenv()


def demo_openai():
    """OpenAI兼容协议调用（默认provider，支持所有国内中转端点）"""
    print("=== 1. OpenAI兼容协议示例 ===")
    llm = LLM(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",  # 也可以填国内中转地址
    )
    resp = llm.invoke(messages=[{"role": "user", "content": "1+1=?"}])
    print(f"响应：{resp.content}\n")


def demo_anthropic():
    """Anthropic Claude原生API调用"""
    print("=== 2. Anthropic Claude示例 ===")
    llm = LLM(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    resp = llm.invoke(messages=[{"role": "user", "content": "1+1=?"}])
    print(f"响应：{resp.content}\n")


def demo_gemini():
    """Google Gemini原生API调用"""
    print("=== 3. Google Gemini示例 ===")
    llm = LLM(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    resp = llm.invoke(messages=[{"role": "user", "content": "1+1=?"}])
    print(f"响应：{resp.content}\n")


async def demo_async():
    """异步多provider并发调用示例"""
    print("=== 4. 异步并发调用示例 ===")
    # 初始化不同provider实例
    llm_openai = LLM(provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm_anthropic = LLM(provider="anthropic", model="claude-3-5-sonnet-latest", api_key=os.getenv("ANTHROPIC_API_KEY"))
    llm_gemini = LLM(provider="gemini", model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    # asyncio.gather并发调用三个模型
    tasks = [
        llm_openai.ainvoke(messages=[{"role": "user", "content": "用一句话介绍OpenAI"}]),
        llm_anthropic.ainvoke(messages=[{"role": "user", "content": "用一句话介绍Anthropic"}]),
        llm_gemini.ainvoke(messages=[{"role": "user", "content": "用一句话介绍Google Gemini"}]),
    ]
    results = await asyncio.gather(*tasks)
    for provider, resp in zip(["OpenAI", "Anthropic", "Gemini"], results):
        print(f"{provider}响应：{resp.content[:50]}...")
    print()


def demo_structured_output():
    """结构化输出跨provider示例"""
    print("=== 5. 跨provider结构化输出示例 ===")
    from pydantic import BaseModel, Field
    class Joke(BaseModel):
        setup: str = Field(description="笑话的提问部分")
        punchline: str = Field(description="笑话的包袱部分")

    # 任意provider都支持相同的结构化输出调用方式
    llm = LLM(provider=os.getenv("DEFAULT_PROVIDER", "openai"))
    joke = llm.invoke_structured(
        messages=[{"role": "user", "content": "讲一个关于程序员的冷笑话"}],
        response_format=Joke,
    )
    print(f"提问：{joke.setup}")
    print(f"包袱：{joke.punchline}\n")


if __name__ == "__main__":
    # 注：需要对应provider的API Key才能运行真实调用，仅作示例参考
    print("💡 I3 多provider示例：不同大模型厂商的调用方式完全统一\n")
    # 注释掉真实调用，仅展示结构
    # demo_openai()
    # demo_anthropic()
    # demo_gemini()
    # asyncio.run(demo_async())
    # demo_structured_output()
    print("✅ 示例代码结构正确，填写对应API Key即可运行")