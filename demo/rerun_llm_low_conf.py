"""Re-run LLM on exported low-confidence cases."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CLSLog import (
    format_sm_result,
    load_config,
)
from modules import llm_utils
from utils import util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/clslog_zookeeper_llm.yaml')
    parser.add_argument('--cases', default='./results/clslog/zookeeper_clslog/low_confidence_cases.json')
    parser.add_argument('--output', default='./results/clslog/zookeeper_clslog/llm_rerun_records.json')
    args = parser.parse_args()

    config = load_config(args.config)
    cases = util.load_json(args.cases)
    prompt_dict = util.load_prompts('./prompt/')
    prompt_name = config.get('llm_prompt', 'clslog_evol_detect')
    prompt_cfg = prompt_dict[prompt_name]
    llm_config = dict(config.get('LLM', {}))

    confidence_threshold = config.get('confidence_threshold') or 0.999986
    prompt_messages = []
    for case in cases:
        sm_detail = {
            'sm_prediction': case['sm_prediction'],
            'anomaly_score': case['anomaly_score'],
            'confidence': case['confidence'],
            'negative_count': case['topk_normal_count'],
            'positive_count': case['topk_abnormal_count'],
        }
        user_prompt = prompt_cfg['user_prompt'].format(
            dataset_name=config.get('dataset_name', 'LogHub'),
            confidence=case['confidence'],
            confidence_threshold=confidence_threshold,
            topk_context='(omitted in rerun script)',
            sm_result=format_sm_result(sm_detail),
            input_logs=case['content'],
        )
        prompt_messages.append([
            {'role': 'system', 'content': prompt_cfg['system_prompt']},
            {'role': 'user', 'content': user_prompt},
        ])

    print(f'Calling LLM for {len(prompt_messages)} cases...')
    llm_answers, llm_usages = llm_utils.llm_json_results(prompt_messages, llm_config)

    records = []
    labels = []
    preds = []
    for case, answer, usage in zip(cases, llm_answers, llm_usages):
        raw_pred = llm_utils.parse_llm_answer(answer)
        pred = llm_utils.apply_dataset_postprocess(case['content'], raw_pred, config)
        label = case['true_label']
        labels.append(label)
        preds.append(pred)
        records.append({
            'index': case['test_index'],
            'true_label': label,
            'llm_raw_prediction': raw_pred,
            'llm_prediction': pred,
            'llm_answer': answer,
            'llm_usage': usage,
            'is_correct': pred == label,
        })

    labels = np.asarray(labels)
    preds = np.asarray(preds)
    metrics = {
        'f1': float(2 * (preds & labels).sum() / max((preds.sum() + labels.sum()), 1)),
        'accuracy': float((preds == labels).mean()),
        'correct': int((preds == labels).sum()),
        'total': len(labels),
    }
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = 2 * precision * recall / max(precision + recall, 1e-9)

    raw_preds = np.asarray([r['llm_raw_prediction'] for r in records])
    raw_tp = int(((raw_preds == 1) & (labels == 1)).sum())
    raw_fp = int(((raw_preds == 1) & (labels == 0)).sum())
    raw_fn = int(((raw_preds == 0) & (labels == 1)).sum())
    raw_precision = raw_tp / max(raw_tp + raw_fp, 1)
    raw_recall = raw_tp / max(raw_tp + raw_fn, 1)
    raw_metrics = {
        'accuracy': float((raw_preds == labels).mean()),
        'normal_accuracy': float((raw_preds[labels == 0] == 0).mean()) if (labels == 0).any() else None,
        'abnormal_recall': float((raw_preds[labels == 1] == 1).mean()) if (labels == 1).any() else None,
        'precision': raw_precision,
        'recall': raw_recall,
        'f1': 2 * raw_precision * raw_recall / max(raw_precision + raw_recall, 1e-9),
    }

    util.save_json({'metrics': metrics, 'raw_metrics': raw_metrics, 'records': records}, args.output)
    print('Raw LLM (prompt only):', json.dumps(raw_metrics, indent=2))
    print('Final (with optional postprocess):', json.dumps(metrics, indent=2))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
