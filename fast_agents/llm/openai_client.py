import os
from typing import Optional, Iterator

from openai import OpenAI
from ..messages.message import Message, MsgToolCall


class OpenAIClient(OpenAI):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            timeout = 180,
            **kwargs
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 1024)

    def invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Message:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            response = self.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']},
                extra_body={"enable_thinking": False}
            )
            message = response.choices[0].message
            response_msg = Message(role="assistant", content=message.content)
            if response.choices[0].finish_reason == "tool_calls":
                for tool in message.tool_calls:
                    msg_tool_call = MsgToolCall(name=tool.function.name, arguments=tool.function.arguments, id=tool.id)
                    response_msg.tool_calls.append(msg_tool_call)

            return response_msg
        except Exception as e:
            raise Exception(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], tools=None, **kwargs) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=self.max_tokens,
                stream=True,
                extra_body={"enable_thinking": False}
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            think_print_flg = False
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 先处理思考过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not think_print_flg:
                        print("[思考]")
                        think_print_flg = True
                    print(f"{delta.reasoning_content}", end="", flush=True)
                    continue  # 思考片段不塞进正文流
                print()  # 在流式输出结束后换行

                content = delta.content or ""
                if content:
                    # print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise Exception(f"LLM调用失败: {str(e)}")

if __name__ == '__main__':
    from dotenv import load_dotenv

    # load_dotenv("D:/00sheepy/.env")
    load_dotenv()  # 自动向上查找 .env，不再硬编码绝对路径
    openai_client = OpenAIClient()
    result = openai_client.stream_invoke(messages=[{"role": "user", "content": "1+1=?"}])
    for content in result:
        print(content, end="", flush=True)

    # print(result)