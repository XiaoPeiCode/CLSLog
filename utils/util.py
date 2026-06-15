import json
import os

import yaml


def load_prompts(prompt_dir):
    prompts = {}
    for filename in os.listdir(prompt_dir):
        if not filename.endswith(('.yaml', '.yml')):
            continue
        prompt_name = os.path.splitext(filename)[0]
        with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as file:
            prompts[prompt_name] = yaml.safe_load(file)
    return prompts


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
