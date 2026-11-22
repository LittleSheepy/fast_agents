"""I2 异步 LLM 调用示例：展示在 asyncio 事件循环中并发调用 LLM 的正确用法。

特性演示：
1. 异步非流式 ainvoke
2. 异步流式 astream_invoke（async for 逐段输出）
3. 异步结构化输出 ainvoke_structured
4. 异步 Embedding aembed_query / aembed_documents
5. 并发调用多个 ainvoke（gather）

运行方式（需要配置真实 API Key）：
    export LLM_API_KEY=sk-xxx
    export LLM_BASE_URL=https://api.openai.com/v1
    python examples/llm/async_demo.py
"""
import asyncio
import sys
import os

# 允许从仓库根目录直接运行本示例
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv()

from fast_agents.llm import LLM
from fast_agents.messages.message import Message
from pydantic import BaseModel


class MathAnswer(BaseModel):
    expression: str
    result: int
    explanation: str


async def demo_ainvoke(llm: LLM):
    print("\n=== 1. 异步非流式 ainvoke ===")
    msg = await llm.ainvoke(messages=[
        {"role": "user", "content": "1+1等于几？请只回答数字。"}
    ])
    print(f"回复：{msg.content}")
    return msg


async def demo_astream(llm: LLM):
    print("\n=== 2. 异步流式 astream_invoke ===")
    print("回复：", end="", flush=True)
    final_content = ""
    async for chunk in llm.astream_invoke(messages=[
        {"role": "user", "content": "用三句话解释什么是异步编程。"}
    ]):
        if chunk.type == "text":
            print(chunk.text, end="", flush=True)
            final_content += chunk.text
        elif chunk.type == "end":
            print()
            print(f"[finish_reason={chunk.finish_reason}, 总长度={len(final_content)}]")
    return final_content


async def demo_ainvoke_structured(llm: LLM):
    print("\n=== 3. 异步结构化输出 ainvoke_structured ===")
    result = await llm.ainvoke_structured(
        messages=[
            {"role": "user", "content": "计算 25 * 4 + 10，并给出表达式、结果和一句话解释。"}
        ],
        response_format=MathAnswer,
    )
    print(f"解析结果：{result.model_dump_json(indent=2, ensure_ascii=False)}")
    return result


async def demo_aembed(llm: LLM):
    print("\n=== 4. 异步 Embedding ===")
    qv = await llm.aembed_query("什么是异步编程？")
    print(f"单条 query embedding 维度：{len(qv)}，前 5 维：{[round(x, 4) for x in qv[:5]]}")

    docs = ["异步编程不阻塞主线程", "asyncio 是 Python 标准库", "并发不等于并行"]
    dvs = await llm.aembed_documents(docs)
    print(f"批量 documents embedding：{len(dvs)} 条，每条维度 {len(dvs[0]) if dvs else 0}")
    return qv, dvs


async def demo_concurrent(llm: LLM):
    print("\n=== 5. 并发调用（asyncio.gather）===")
    questions = ["1+1=?", "2+3=?", "5*7=?", "100/4=?", "2**10=?"]
    print(f"同时问 {len(questions)} 个问题，并发请求：")
    tasks = [
        llm.ainvoke(messages=[{"role": "user", "content": q + " 只回答数字。"}])
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    for q, r in zip(questions, results):
        print(f"  {q} → {r.content.strip()}")
    return results


async def main():
    # 初始化 LLM 门面（默认开启重试/熔断，异步 client 首次调用时懒加载）
    llm = LLM(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        temperature=0.0,  # 数学题用确定性输出
    )

    await demo_ainvoke(llm)
    await demo_astream(llm)
    await demo_ainvoke_structured(llm)
    await demo_aembed(llm)
    await demo_concurrent(llm)

    print("\n✅ 所有异步示例运行完成！")


if __name__ == "__main__":
    asyncio.run(main())
