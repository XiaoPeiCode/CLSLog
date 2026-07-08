"""Hyperparameter search for Zookeeper SM-only pipeline (valid-driven selection)."""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CLSLog import (
    apply_hdbscan,
    generate_siamese_embeddings,
    load_config,
    set_seed,
    sm_predict,
    train_siamese_network,
)
from utils.loghub_preprocessing import get_device, load_loghub_data


def _scores_and_labels(train_embeddings, train_labels, eval_embeddings, k):
    result = sm_predict(train_embeddings, train_labels, eval_embeddings, k)
    scores = np.asarray([d['anomaly_score'] for d in result['details']])
    return scores, result['predictions']


def _metrics_from_scores(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    return {
        'f1': float(f1_score(labels, preds, pos_label=1)),
        'precision': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'recall': float(recall_score(labels, preds, pos_label=1, zero_division=0)),
    }


def _select_k_and_threshold(train_emb, train_labels, valid_emb, valid_labels, k_candidates):
    best = {'f1': -1.0, 'k': 5, 'threshold': 0.0}
    for k in k_candidates:
        scores, _ = _scores_and_labels(train_emb, train_labels, valid_emb, k)
        thresholds = np.unique(np.round(scores, 4))
        if len(thresholds) > 200:
            thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 40))
        thresholds = np.concatenate([thresholds, [0.0]])
        for threshold in thresholds:
            metrics = _metrics_from_scores(scores, valid_labels, threshold)
            if metrics['f1'] > best['f1']:
                best = {'f1': metrics['f1'], 'k': k, 'threshold': float(threshold), **metrics}
    return best


def _prepare_embeddings(siamese_model, train_emb, valid_emb, test_emb):
    if siamese_model is None:
        return (
            normalize(train_emb, norm='l2'),
            normalize(valid_emb, norm='l2'),
            normalize(test_emb, norm='l2'),
        )
    return (
        normalize(generate_siamese_embeddings(siamese_model, train_emb), norm='l2'),
        normalize(generate_siamese_embeddings(siamese_model, valid_emb), norm='l2'),
        normalize(generate_siamese_embeddings(siamese_model, test_emb), norm='l2'),
    )


def run_trial(raw_train, raw_valid, raw_test, train_labels, valid_labels, test_labels, trial, device):
    cfg = trial['train_config']
    siamese_model = None
    if cfg.get('use_siamese_network', True):
        siamese_model = train_siamese_network(
            raw_train.copy(), train_labels.copy(),
            raw_valid.copy(), valid_labels.copy(),
            cfg, device,
        )
    train_emb, valid_emb, test_emb = _prepare_embeddings(
        siamese_model, raw_train, raw_valid, raw_test,
    )
    train_lbl = train_labels.copy()
    if cfg.get('use_hdbscan', False):
        train_emb, train_lbl = apply_hdbscan(
            train_emb, train_lbl,
            min_cluster_size=cfg.get('min_cluster_size', 15),
            min_samples=cfg.get('min_samples', 1),
        )

    selection = _select_k_and_threshold(
        train_emb, train_lbl, valid_emb, valid_labels, trial['k_candidates'],
    )
    test_scores, _ = _scores_and_labels(train_emb, train_lbl, test_emb, selection['k'])
    test_metrics = _metrics_from_scores(test_scores, test_labels, selection['threshold'])
    return {
        **trial['params'],
        'valid_f1': selection['f1'],
        'valid_precision': selection['precision'],
        'valid_recall': selection['recall'],
        'best_k': selection['k'],
        'score_threshold': selection['threshold'],
        'train_bank_size': int(len(train_emb)),
        'test_metrics': test_metrics,
    }


def build_trials(base_config):
    k_candidates = list(range(1, 21)) + [25, 30, 40, 50]
    train_grid = [
        {'cl_num_epochs': 20, 'batch_size': 128, 'learning_rate': 0.001, 'use_hdbscan': False},
        {'cl_num_epochs': 30, 'batch_size': 128, 'learning_rate': 0.001, 'use_hdbscan': False},
        {'cl_num_epochs': 50, 'batch_size': 256, 'learning_rate': 0.001, 'use_hdbscan': False},
        {'cl_num_epochs': 30, 'batch_size': 64, 'learning_rate': 0.0003, 'use_hdbscan': False},
        {'cl_num_epochs': 30, 'batch_size': 128, 'learning_rate': 0.003, 'use_hdbscan': False},
        {'cl_num_epochs': 30, 'batch_size': 128, 'learning_rate': 0.001, 'use_hdbscan': True, 'min_cluster_size': 10, 'min_samples': 1},
        {'cl_num_epochs': 30, 'batch_size': 128, 'learning_rate': 0.001, 'use_hdbscan': True, 'min_cluster_size': 15, 'min_samples': 1},
        {'cl_num_epochs': 30, 'batch_size': 128, 'learning_rate': 0.001, 'use_hdbscan': True, 'min_cluster_size': 5, 'min_samples': 1},
        {'cl_num_epochs': 20, 'batch_size': 128, 'learning_rate': 0.001, 'use_siamese_network': False, 'use_hdbscan': False},
    ]
    trials = []
    for params in train_grid:
        train_config = copy.deepcopy(base_config)
        train_config.update(params)
        train_config['eval_in_train'] = True
        trials.append({
            'params': params,
            'train_config': train_config,
            'k_candidates': k_candidates,
        })
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/clslog_zookeeper.yaml')
    parser.add_argument('--output', default='./results/clslog/zookeeper_tuning/tuning_results.json')
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    base_config = load_config(args.config)
    base_config['use_cache'] = True

    print(f'Loading data on {device}...')
    data = load_loghub_data(base_config, use_cache=True)
    raw_train = np.vstack(data['train']['embedding'].values)
    raw_valid = np.vstack(data['valid']['embedding'].values)
    raw_test = np.vstack(data['test']['embedding'].values)
    train_labels = data['train']['label'].values
    valid_labels = data['valid']['label'].values
    test_labels = data['test']['label'].values

    trials = build_trials(base_config)
    results = []
    best_valid = {'valid_f1': -1.0}
    start = time.time()

    for idx, trial in enumerate(trials, 1):
        print(f'\n=== Trial {idx}/{len(trials)}: {trial["params"]} ===')
        try:
            result = run_trial(
                raw_train, raw_valid, raw_test,
                train_labels, valid_labels, test_labels,
                trial, device,
            )
            results.append(result)
            print(
                f'valid F1={result["valid_f1"]:.4f}, test F1={result["test_metrics"]["f1"]:.4f}, '
                f'k={result["best_k"]}, threshold={result["score_threshold"]:.4f}'
            )
            if result['valid_f1'] > best_valid.get('valid_f1', -1):
                best_valid = result
        except Exception as exc:
            print(f'Trial failed: {exc}')
            results.append({'params': trial['params'], 'error': str(exc)})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = {
        'dataset': 'Zookeeper',
        'elapsed_seconds': time.time() - start,
        'trials': results,
        'best_by_valid': best_valid,
    }
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)

    print('\n=== Best config (selected on valid) ===')
    print(json.dumps(best_valid, indent=2))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
