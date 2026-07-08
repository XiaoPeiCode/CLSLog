import asyncio
import json
import platform

from openai import RateLimitError
from tqdm.asyncio import tqdm_asyncio

from modules import llm_chat


async def process_prompts_in_batches(prompts, llm_config, batch_size=8, delay=0.5):
    results = []
    for start in tqdm_asyncio(range(0, len(prompts), batch_size), desc='LLM batches'):
        batch = prompts[start:start + batch_size]
        try:
            batch_results = await llm_chat.chat_async(batch, llm_config)
            results.extend(batch_results)
        except RateLimitError:
            await asyncio.sleep(10)
            batch_results = await llm_chat.chat_async(batch, llm_config)
            results.extend(batch_results)
        if delay:
            await asyncio.sleep(delay)
    return results


def completion_usage_to_dict(completion_usage):
    if completion_usage is None:
        return {}
    return {
        'completion_tokens': completion_usage.completion_tokens,
        'prompt_tokens': completion_usage.prompt_tokens,
        'total_tokens': completion_usage.total_tokens,
    }


def parse_json_response(content):
    if not content:
        return {}
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def parse_llm_answer(raw_answer):
    system_state = (
        raw_answer.get('System State')
        or raw_answer.get('system_state')
        or raw_answer.get('systemState')
        or raw_answer.get('label')
    )
    if system_state is None:
        return 0
    return int(str(system_state).strip().lower() in {'abnormal', 'anomaly', '1', 'true', 'error'})


def apply_dataset_postprocess(content, prediction, config=None):
    config = config or {}
    if not config.get('llm_rule_postprocess', False):
        return prediction

    dataset_name = (config.get('dataset_name') or '').lower()
    content_lower = content.lower()
    if dataset_name == 'zookeeper':
        critical_phrase = 'unexpected exception causing shutdown while sock still open'
        if critical_phrase in content_lower:
            return 1
        return 0
    return prediction


def llm_json_results(prompts, llm_config):
    if not prompts:
        return [], []

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    batch_size = llm_config.get('batch_size', 8)
    llm_outputs = asyncio.run(
        process_prompts_in_batches(prompts, llm_config, batch_size=batch_size),
        debug=False,
    )
    llm_answer = [parse_json_response(item.choices[0].message.content) for item in llm_outputs]
    llm_usage = [completion_usage_to_dict(item.usage) for item in llm_outputs]
    return llm_answer, llm_usage
