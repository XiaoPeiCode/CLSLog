"""
Preprocess LogHub datasets (BGL / Zookeeper) for CLSLog experiments.

Usage:
    python demo/loghub_data_process_demo.py --dataset BGL
    python demo/loghub_data_process_demo.py --dataset Zookeeper
"""

import argparse
import json
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.loghub_preprocessing import load_loghub_data


def build_config(dataset_name):
    config_path = os.path.join(ROOT, 'config', f'clslog_{dataset_name.lower()}.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BGL', choices=['BGL', 'Zookeeper'])
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--no_cache', action='store_true')
    args = parser.parse_args()

    config = build_config(args.dataset)
    if args.sample_size is not None:
        config['sample_size'] = args.sample_size

    results = load_loghub_data(config, use_cache=not args.no_cache)
    summary = {
        dataset: {
            'num_sequences': len(df),
            'num_normal': int((df['label'] == 0).sum()),
            'num_abnormal': int((df['label'] == 1).sum()),
        }
        for dataset, df in results.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
