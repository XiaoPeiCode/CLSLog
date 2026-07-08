import asyncio
import os
import platform

import httpx
from openai import AsyncOpenAI

_clients = {}


def get_openai_client(llm_config=None):
    llm_config = llm_config or {}
    api_key = (
        llm_config.get('api_key')
        or os.getenv('CLSLOG_LLM_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or os.getenv('DASHSCOPE_API_KEY')
        or os.getenv('QWEN_API_KEY')
        or ''
    )
    base_url = llm_config.get('base_url') or os.getenv('CLSLOG_LLM_BASE_URL') or 'https://api.openai.com/v1'
    cache_key = (base_url, api_key[:8] if api_key else '')
    if cache_key not in _clients:
        http_client = httpx.AsyncClient(trust_env=False, timeout=120.0)
        _clients[cache_key] = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    return _clients[cache_key]


async def chat_async_task(messages, llm_config):
    client = get_openai_client(llm_config)
    params = build_chat_params(llm_config)
    response = await client.chat.completions.create(
        messages=messages,
        **params,
    )
    return response


async def chat_async(messages_list, llm_config):
    tasks = [chat_async_task(messages, llm_config) for messages in messages_list]
    return await asyncio.gather(*tasks)


def build_chat_params(llm_config):
    params = {
        'model': llm_config.get('model', 'gpt-4o-mini'),
        'temperature': llm_config.get('temperature', 0.01),
        'top_p': llm_config.get('top_p', 0.95),
    }
    if llm_config.get('response_format', True):
        params['response_format'] = {'type': 'json_object'}
    if llm_config.get('max_tokens'):
        params['max_tokens'] = llm_config['max_tokens']
    return params


if __name__ == '__main__':
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    demo_config = {
        'model': os.getenv('CLSLOG_LLM_MODEL', 'deepseek-v4-pro'),
        'base_url': os.getenv('CLSLOG_LLM_BASE_URL', 'https://aiapi.opl.chat/v1'),
        'api_key': os.getenv('CLSLOG_LLM_API_KEY', ''),
        'temperature': 0.01,
    }
    result = asyncio.run(chat_async_task(
        [{'role': 'user', 'content': 'Return JSON: {"ok": true}'}],
        demo_config,
    ))
    print(result.choices[0].message.content)
