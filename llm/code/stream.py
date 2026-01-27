from openai import OpenAI

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="93a67648-c2cd-4a51-99ba-c51114b537ee",
)


def llm_stream(messages, max_new_tokens=8192, buffer_size=10):
    """
    流式生成响应，支持缓冲以提升前端显示流畅度。

    Args:
        messages: 对话消息列表
        max_new_tokens: 最大生成 token 数
        buffer_size: 缓冲大小（字符数），累积到该大小或遇到换行符时 yield

    Yields:
        str: 累积的文本块
    """
    response = client.chat.completions.create(
        model="ep-20251209150604-gxb42",
        messages=messages,
        stream=True,
        max_tokens=max_new_tokens,
    )

    buffer = ""
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue

        if "reasoning_content" in delta:
            text = delta.reasoning_content or ""
        else:
            text = delta.content or ""

        if text:
            buffer += text
            if len(buffer) >= buffer_size or "\n" in text:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer


if __name__ == "__main__":
    prompt = '小说《倚天屠龙记》的作者是谁？'
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    stream_res = llm_stream(messages)
    full_content = ''
    for chunk in stream_res:
        full_content += chunk
        print(chunk, end='', flush=True)
