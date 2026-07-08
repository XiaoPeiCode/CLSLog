"""CLSLog: small-model + LLM cascade for log anomaly detection."""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import wandb
except ImportError:
    wandb = None
import yaml
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from utils import util
from utils.cluster_utils import apply_hdbscan
from utils.loghub_preprocessing import get_device, load_loghub_data


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    llm_local_path = os.path.join(config_dir, 'llm_local.yaml')
    if os.path.exists(llm_local_path):
        with open(llm_local_path, 'r', encoding='utf-8') as file:
            llm_local = yaml.safe_load(file) or {}
        config.setdefault('LLM', {})
        config['LLM'].update({k: v for k, v in llm_local.items() if v is not None})
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)


class SiameseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        anchor = self.embeddings[idx]
        same_class_indices = np.where(self.labels == self.labels[idx])[0]
        if len(same_class_indices) > 1:
            positive_idx = random.choice(same_class_indices[same_class_indices != idx])
        else:
            positive_idx = (idx + 1) % len(self.embeddings)
        positive = self.embeddings[positive_idx]
        label = float(self.labels[idx] != self.labels[positive_idx])
        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def contrastive_loss(output1, output2, label, margin=1.0):
    distance = nn.functional.pairwise_distance(output1, output2)
    return torch.mean((1 - label) * torch.pow(distance, 2) +
                      label * torch.pow(torch.clamp(margin - distance, min=0.0), 2))


def generate_siamese_embeddings(siamese_model, embeddings, batch_size=512):
    siamese_model.eval()
    device = next(siamese_model.parameters()).device
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    transformed = []
    with torch.no_grad():
        for batch in loader:
            transformed.append(siamese_model.forward_once(batch[0].to(device)).cpu().numpy())
    return np.vstack(transformed)


def train_siamese_network(train_embeddings, train_labels, eval_embeddings, eval_labels, config, device):
    embedding_dim = train_embeddings.shape[1]
    model = SiameseNetwork(embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    loader = DataLoader(
        SiameseDataset(train_embeddings, train_labels),
        batch_size=config['batch_size'],
        shuffle=True,
    )

    if wandb is not None:
        wandb.init(
            project=config.get('wandb_project', 'CLSLog'),
            mode=config.get('wandb_mode', 'disabled'),
            name=f"Train_SM({config['dataset_name']})",
            tags=['CLSLog', 'Train_SM', config['dataset_name']],
        )

    best_f1 = 0.0
    best_state = None
    for epoch in range(config['cl_num_epochs']):
        model.train()
        total_loss = 0.0
        for anchor, positive, label in loader:
            optimizer.zero_grad()
            output1, output2 = model(anchor.to(device), positive.to(device))
            loss = contrastive_loss(output1, output2, label.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if config.get('eval_in_train', True):
            f1, pr, re = evaluate_sm(model, train_embeddings, train_labels, eval_embeddings, eval_labels, config)
            eval_name = config.get('eval_split', 'valid')
            print(f"Epoch [{epoch + 1}/{config['cl_num_epochs']}], Loss: {total_loss / len(loader):.4f}, "
                  f'{eval_name} F1: {f1:.4f}, Pr: {pr:.4f}, Re: {re:.4f}')
            if wandb is not None:
                wandb.log({'epoch': epoch + 1, 'loss': total_loss / len(loader), 'eval_f1': f1, 'eval_pr': pr, 'eval_re': re})
            if f1 >= best_f1:
                best_f1 = f1
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    if wandb is not None:
        wandb.finish()
    print(f'Best SM F1 on {config.get("eval_split", "valid")} during training: {best_f1:.4f}')
    return model


def compute_sm_outputs(similarities, neighbor_labels):
    confidence = float(np.mean(similarities))
    anomaly_score = float(np.sum(similarities * (2 * neighbor_labels - 1)))
    sm_prediction = int(anomaly_score > 0)
    positive_count = int(np.sum(neighbor_labels))
    negative_count = int(len(neighbor_labels) - positive_count)
    return {
        'confidence': confidence,
        'anomaly_score': anomaly_score,
        'sm_prediction': sm_prediction,
        'positive_count': positive_count,
        'negative_count': negative_count,
    }


def evaluate_sm(siamese_model, train_embeddings, train_labels, test_embeddings, test_labels, config):
    transformed_train = generate_siamese_embeddings(siamese_model, train_embeddings)
    transformed_test = generate_siamese_embeddings(siamese_model, test_embeddings)
    if config.get('use_hdbscan', False):
        transformed_train, train_labels = apply_hdbscan(
            transformed_train,
            train_labels,
            min_cluster_size=config.get('min_cluster_size', 15),
            min_samples=config.get('min_samples', 1),
        )

    transformed_train = normalize(transformed_train, norm='l2')
    transformed_test = normalize(transformed_test, norm='l2')
    predictions = sm_predict(transformed_train, train_labels, transformed_test, config['k'])['predictions']
    return (
        f1_score(test_labels, predictions, pos_label=1),
        precision_score(test_labels, predictions, pos_label=1),
        recall_score(test_labels, predictions, pos_label=1),
    )


def sm_predict(train_embeddings, train_labels, test_embeddings, k, score_threshold=0.0):
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(train_embeddings)
    distances, indices = nbrs.kneighbors(test_embeddings)
    similarities = 1 - distances

    predictions = []
    confidences = []
    details = []
    for i in range(len(test_embeddings)):
        neighbor_labels = train_labels[indices[i]]
        outputs = compute_sm_outputs(similarities[i], neighbor_labels)
        outputs['sm_prediction'] = int(outputs['anomaly_score'] > score_threshold)
        predictions.append(outputs['sm_prediction'])
        confidences.append(outputs['confidence'])
        details.append(outputs)

    return {
        'predictions': np.asarray(predictions),
        'confidences': np.asarray(confidences),
        'indices': indices,
        'similarities': similarities,
        'details': details,
    }


def tune_k_and_threshold(train_embeddings, train_labels, valid_embeddings, valid_labels, k_candidates=None):
    if k_candidates is None:
        k_candidates = list(range(1, 21)) + [25, 30, 40, 50]
    best = {'f1': -1.0, 'k': 5, 'score_threshold': 0.0}
    for k in k_candidates:
        valid_sm = sm_predict(train_embeddings, train_labels, valid_embeddings, k, score_threshold=0.0)
        scores = np.asarray([d['anomaly_score'] for d in valid_sm['details']])
        thresholds = np.unique(np.round(scores, 4))
        if len(thresholds) > 200:
            thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 40))
        thresholds = np.concatenate([thresholds, [0.0]])
        for threshold in thresholds:
            preds = (scores > threshold).astype(int)
            if valid_labels.sum() == 0 or preds.sum() == 0:
                continue
            f1 = f1_score(valid_labels, preds, pos_label=1)
            if f1 > best['f1']:
                best = {
                    'f1': float(f1),
                    'k': int(k),
                    'score_threshold': float(threshold),
                    'precision': float(precision_score(valid_labels, preds, pos_label=1, zero_division=0)),
                    'recall': float(recall_score(valid_labels, preds, pos_label=1, zero_division=0)),
                }
    return best


def select_confidence_threshold(valid_confidences, valid_labels, valid_predictions, default=0.95):
    """Select mu on valid so that SM-routed (high-confidence) samples maximize F1."""
    if len(valid_confidences) == 0:
        return default

    candidates = np.unique(valid_confidences)
    if len(candidates) > 50:
        candidates = np.quantile(valid_confidences, np.linspace(0.05, 0.95, 40))

    best_threshold = default
    best_f1 = -1.0
    for threshold in candidates:
        high_conf_mask = valid_confidences > threshold
        if high_conf_mask.sum() == 0:
            continue
        routed_labels = valid_labels[high_conf_mask]
        routed_preds = valid_predictions[high_conf_mask]
        if routed_labels.sum() == 0:
            continue
        f1 = f1_score(routed_labels, routed_preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def _metrics_if_valid(true_labels, predictions):
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    if len(true_labels) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 0}
    if true_labels.sum() == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'support': int(len(true_labels)),
            'note': 'no abnormal samples in subset',
        }
    return {
        'f1': float(f1_score(true_labels, predictions, pos_label=1)),
        'precision': float(precision_score(true_labels, predictions, pos_label=1, zero_division=0)),
        'recall': float(recall_score(true_labels, predictions, pos_label=1, zero_division=0)),
        'support': int(len(true_labels)),
    }


def evaluate_subgroup(true_labels, predictions, result_dir, tag):
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    metrics = _metrics_if_valid(true_labels, predictions)
    print(f'\n=== {tag} (n={len(true_labels)}, abnormal={int(true_labels.sum())}) ===')
    if len(true_labels) == 0:
        print('Empty subset, skip classification report.')
        return metrics

    report = classification_report(true_labels, predictions, digits=4, zero_division=0)
    print(report)
    print(
        f'{tag} F1: {metrics["f1"]:.4f}, Precision: {metrics["precision"]:.4f}, '
        f'Recall: {metrics["recall"]:.4f}'
    )

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{tag}_evaluation.txt'), 'w', encoding='utf-8') as file:
        file.write(report + '\n')
        file.write(
            f'F1: {metrics["f1"]:.4f}\nPrecision: {metrics["precision"]:.4f}\n'
            f'Recall: {metrics["recall"]:.4f}\nSupport: {metrics["support"]}\n'
        )
    return metrics


def analyze_confidence_routing(test_labels, test_sm, confidence_threshold):
    """
    Split test samples by confidence mu:
    - high-confidence: C(si) > mu  (non-evolutionary, SM direct)
    - low-confidence:  C(si) <= mu (evolutionary / uncertain, LLM candidate)
    """
    confidences = test_sm['confidences']
    predictions = test_sm['predictions']
    high_conf_mask = confidences > confidence_threshold
    low_conf_mask = ~high_conf_mask

    return {
        'confidence_threshold': float(confidence_threshold),
        'high_confidence': {
            'description': 'C(si) > mu, non-evolutionary logs, routed to SM',
            'count': int(high_conf_mask.sum()),
            'ratio': float(high_conf_mask.mean()),
            'label_stats': _split_label_stats(test_labels[high_conf_mask]),
            'sm_metrics': _metrics_if_valid(test_labels[high_conf_mask], predictions[high_conf_mask]),
        },
        'low_confidence': {
            'description': 'C(si) <= mu, evolutionary/uncertain logs, routed to LLM',
            'count': int(low_conf_mask.sum()),
            'ratio': float(low_conf_mask.mean()),
            'label_stats': _split_label_stats(test_labels[low_conf_mask]),
            'sm_metrics': _metrics_if_valid(test_labels[low_conf_mask], predictions[low_conf_mask]),
        },
        'high_conf_mask': high_conf_mask,
        'low_conf_mask': low_conf_mask,
    }


def print_routing_summary(routing):
    print('\n========== Confidence Routing Summary ==========')
    print(f"Confidence threshold mu: {routing['confidence_threshold']:.4f}")
    for key in ('high_confidence', 'low_confidence'):
        part = routing[key]
        sm = part['sm_metrics']
        print(
            f"\n{key}: count={part['count']} ({part['ratio']:.1%}), "
            f"labels={part['label_stats']}"
        )
        print(
            f"  SM F1={sm['f1']:.4f}, Pr={sm['precision']:.4f}, Re={sm['recall']:.4f}"
        )
    print('================================================\n')


def export_routing_cases(test_df, test_labels, test_sm, routing, result_dir, tag_prefix='low_confidence'):
    """Persist routed samples with SM outputs for downstream LLM or analysis."""
    mask = routing['low_conf_mask'] if tag_prefix == 'low_confidence' else routing['high_conf_mask']
    indices = np.where(mask)[0]
    records = []
    for idx in indices:
        detail = test_sm['details'][idx]
        records.append({
            'test_index': int(idx),
            'true_label': int(test_labels[idx]),
            'true_label_name': 'Abnormal' if test_labels[idx] == 1 else 'Normal',
            'sm_prediction': int(detail['sm_prediction']),
            'sm_prediction_name': 'Abnormal' if detail['sm_prediction'] == 1 else 'Normal',
            'confidence': float(detail['confidence']),
            'anomaly_score': float(detail['anomaly_score']),
            'topk_normal_count': detail['negative_count'],
            'topk_abnormal_count': detail['positive_count'],
            'is_correct': bool(detail['sm_prediction'] == test_labels[idx]),
            'content': test_df['content'].iloc[idx],
        })

    export_path = os.path.join(result_dir, f'{tag_prefix}_cases.json')
    util.save_json(records, export_path)
    print(f'Exported {len(records)} {tag_prefix} cases to {export_path}')

    abnormal_records = [r for r in records if r['true_label'] == 1]
    if abnormal_records:
        abnormal_path = os.path.join(result_dir, f'{tag_prefix}_abnormal_cases.json')
        util.save_json(abnormal_records, abnormal_path)
        print(f'Exported {len(abnormal_records)} abnormal {tag_prefix} cases to {abnormal_path}')
    return records


def format_topk_context(neighbor_contents, neighbor_labels, similarities):
    lines = []
    for content, label, sim in zip(neighbor_contents, neighbor_labels, similarities):
        state = 'Abnormal' if label == 1 else 'Normal'
        lines.append(f'- Similarity: {sim:.4f}, Label: {state}, Logs:\n{content}')
    return '\n'.join(lines)


def format_sm_result(outputs):
    state = 'Abnormal' if outputs['sm_prediction'] == 1 else 'Normal'
    return (
        f"Small model prediction: {state}\n"
        f"Anomaly score: {outputs['anomaly_score']:.4f}\n"
        f"Confidence (mean top-k similarity): {outputs['confidence']:.4f}\n"
        f"Top-k label distribution: normal={outputs['negative_count']}, abnormal={outputs['positive_count']}"
    )


def llm_predict_low_confidence_samples(
    low_conf_indices,
    test_contents,
    test_labels,
    train_contents,
    train_labels,
    sm_details,
    neighbor_indices,
    similarities,
    prompt_dict,
    config,
    confidence_threshold,
):
    llm_predictions = {}
    llm_records = []
    if not config.get('use_large_model', False):
        print('use_large_model=False, low-confidence samples fall back to SM predictions.')
        for idx in low_conf_indices:
            llm_predictions[idx] = sm_details[idx]['sm_prediction']
        return llm_predictions, llm_records

    from modules import llm_utils

    prompt_name = config.get('llm_prompt', 'clslog_evol_detect')
    if prompt_name not in prompt_dict and 'clslog_detect' in prompt_dict:
        prompt_name = 'clslog_detect'
    prompt_cfg = prompt_dict[prompt_name]
    llm_config = dict(config.get('LLM', {}))
    llm_config.setdefault('batch_size', 8)

    prompt_messages = []
    index_order = []
    for idx in low_conf_indices:
        neighbors = neighbor_indices[idx]
        topk_context = format_topk_context(
            train_contents[neighbors],
            train_labels[neighbors],
            similarities[idx],
        )
        sm_result = format_sm_result(sm_details[idx])
        user_prompt = prompt_cfg['user_prompt'].format(
            dataset_name=config.get('dataset_name', 'LogHub'),
            confidence=sm_details[idx]['confidence'],
            confidence_threshold=confidence_threshold,
            topk_context=topk_context,
            sm_result=sm_result,
            input_logs=test_contents[idx],
        )
        prompt_messages.append([
            {'role': 'system', 'content': prompt_cfg['system_prompt']},
            {'role': 'user', 'content': user_prompt},
        ])
        index_order.append(int(idx))

    print(f'Calling LLM for {len(prompt_messages)} low-confidence samples...')
    llm_answers, llm_usages = llm_utils.llm_json_results(prompt_messages, llm_config)
    for idx, answer, usage in zip(index_order, llm_answers, llm_usages):
        raw_prediction = llm_utils.parse_llm_answer(answer)
        prediction = llm_utils.apply_dataset_postprocess(
            test_contents[idx],
            raw_prediction,
            config,
        )
        llm_predictions[idx] = prediction
        llm_records.append({
            'index': idx,
            'true_label': int(test_labels[idx]),
            'sm_prediction': int(sm_details[idx]['sm_prediction']),
            'llm_raw_prediction': raw_prediction,
            'llm_prediction': prediction,
            'confidence': float(sm_details[idx]['confidence']),
            'content': test_contents[idx],
            'llm_answer': answer,
            'llm_usage': usage,
            'is_correct': bool(prediction == test_labels[idx]),
        })
    return llm_predictions, llm_records


def evaluate_predictions(true_labels, predictions, result_dir, tag='clslog'):
    report = classification_report(true_labels, predictions, digits=4)
    f1 = f1_score(true_labels, predictions, pos_label=1)
    pr = precision_score(true_labels, predictions, pos_label=1)
    re = recall_score(true_labels, predictions, pos_label=1)
    print(report)
    print(f'{tag} F1: {f1:.4f}, Precision: {pr:.4f}, Recall: {re:.4f}')

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{tag}_evaluation.txt'), 'w', encoding='utf-8') as file:
        file.write(report + '\n')
        file.write(f'F1: {f1:.4f}\nPrecision: {pr:.4f}\nRecall: {re:.4f}\n')
    return {'f1': f1, 'precision': pr, 'recall': re}


def _split_label_stats(labels):
    labels = np.asarray(labels)
    return {
        'normal': int((labels == 0).sum()),
        'abnormal': int((labels == 1).sum()),
    }


def main(config):
    set_seed(config.get('random_seed', 42))
    device = get_device()
    print(f'Using device: {device}')
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    sm_only_mode = config.get('sm_only_mode', False)
    enable_routing = config.get('enable_confidence_routing', True)

    data = load_loghub_data(config, use_cache=config.get('use_cache', True))
    train_df, valid_df, test_df = data['train'], data['valid'], data['test']

    train_embeddings = np.vstack(train_df['embedding'].values)
    valid_embeddings = np.vstack(valid_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)
    train_labels = train_df['label'].values
    valid_labels = valid_df['label'].values
    test_labels = test_df['label'].values
    train_contents = train_df['content'].values
    test_contents = test_df['content'].values

    print(f'Train sequences: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}')
    print(f'Test label stats: {_split_label_stats(test_labels)}')

    siamese_model = None
    if config.get('use_siamese_network', True):
        start_time = time.time()
        siamese_model = train_siamese_network(
            train_embeddings, train_labels, valid_embeddings, valid_labels, config, device
        )
        print(f'Siamese training time: {time.time() - start_time:.2f}s')
        train_embeddings = normalize(generate_siamese_embeddings(siamese_model, train_embeddings), norm='l2')
        valid_embeddings = normalize(generate_siamese_embeddings(siamese_model, valid_embeddings), norm='l2')
        test_embeddings = normalize(generate_siamese_embeddings(siamese_model, test_embeddings), norm='l2')

    if config.get('use_hdbscan', False):
        train_embeddings, train_labels = apply_hdbscan(
            train_embeddings,
            train_labels,
            min_cluster_size=config.get('min_cluster_size', 15),
            min_samples=config.get('min_samples', 1),
        )

    k = config['k']
    score_threshold = config.get('score_threshold') or 0.0
    if config.get('tune_k_on_valid', True):
        tuned = tune_k_and_threshold(
            train_embeddings, train_labels, valid_embeddings, valid_labels,
            k_candidates=config.get('k_candidates'),
        )
        k = tuned['k']
        score_threshold = tuned['score_threshold']
        print(f'Tuned on valid: k={k}, score_threshold={score_threshold:.4f}, valid F1={tuned["f1"]:.4f}')

    valid_sm = sm_predict(train_embeddings, train_labels, valid_embeddings, k, score_threshold=score_threshold)
    test_sm = sm_predict(train_embeddings, train_labels, test_embeddings, k, score_threshold=score_threshold)
    sm_metrics = evaluate_predictions(test_labels, test_sm['predictions'], result_dir, tag='trained_sm_only')

    confidence_threshold = config.get('confidence_threshold')
    if confidence_threshold is None and enable_routing:
        confidence_threshold = select_confidence_threshold(
            valid_sm['confidences'],
            valid_labels,
            valid_sm['predictions'],
        )
    elif confidence_threshold is None:
        confidence_threshold = 0.95

    routing = None
    high_conf_metrics = None
    low_conf_metrics = None
    if enable_routing:
        routing = analyze_confidence_routing(test_labels, test_sm, confidence_threshold)
        print_routing_summary(routing)
        high_conf_metrics = evaluate_subgroup(
            test_labels[routing['high_conf_mask']],
            test_sm['predictions'][routing['high_conf_mask']],
            result_dir,
            tag='sm_high_confidence',
        )
        low_conf_metrics = evaluate_subgroup(
            test_labels[routing['low_conf_mask']],
            test_sm['predictions'][routing['low_conf_mask']],
            result_dir,
            tag='sm_low_confidence',
        )
        routing_export = {
            'confidence_threshold': routing['confidence_threshold'],
            'high_confidence': {
                k: v for k, v in routing['high_confidence'].items() if k != 'description'
            },
            'low_confidence': {
                k: v for k, v in routing['low_confidence'].items() if k != 'description'
            },
        }
        util.save_json(routing_export, os.path.join(result_dir, 'routing_analysis.json'))
        export_routing_cases(test_df, test_labels, test_sm, routing, result_dir, 'low_confidence')
        export_routing_cases(test_df, test_labels, test_sm, routing, result_dir, 'high_confidence')

    final_predictions = test_sm['predictions'].copy()
    llm_records = []
    llm_low_metrics = None
    if enable_routing and not sm_only_mode:
        low_conf_indices = np.where(routing['low_conf_mask'])[0]
        prompt_dict = util.load_prompts('./prompt/')
        llm_predictions, llm_records = llm_predict_low_confidence_samples(
            low_conf_indices,
            test_contents,
            test_labels,
            train_contents,
            train_labels,
            test_sm['details'],
            test_sm['indices'],
            test_sm['similarities'],
            prompt_dict,
            config,
            confidence_threshold,
        )
        for idx, pred in llm_predictions.items():
            final_predictions[idx] = pred
        if config.get('use_large_model', False) and llm_records:
            low_conf_llm_preds = np.asarray([llm_predictions[i] for i in low_conf_indices])
            llm_low_metrics = evaluate_subgroup(
                test_labels[low_conf_indices],
                low_conf_llm_preds,
                result_dir,
                tag='clslog_low_confidence',
            )
            util.save_json(llm_records, os.path.join(result_dir, 'llm_low_confidence_records.json'))

    clslog_metrics = evaluate_predictions(test_labels, final_predictions, result_dir, tag='clslog')

    summary = {
        'dataset': config['dataset_name'],
        'mode': 'sm_only' if sm_only_mode else 'clslog',
        'split_method': config.get('split_method', 'stratified'),
        'k': k,
        'score_threshold': score_threshold,
        'confidence_threshold': confidence_threshold,
        'enable_confidence_routing': enable_routing,
        'sm_only_metrics': sm_metrics,
        'metrics': clslog_metrics if not sm_only_mode else sm_metrics,
        'train_size': int(len(train_df)),
        'valid_size': int(len(valid_df)),
        'test_size': int(len(test_df)),
        'train_label_stats': _split_label_stats(train_labels),
        'valid_label_stats': _split_label_stats(valid_labels),
        'test_label_stats': _split_label_stats(test_labels),
    }
    if routing is not None:
        summary['routing'] = {
            'high_confidence': routing['high_confidence'],
            'low_confidence': routing['low_confidence'],
        }
        summary['sm_high_confidence_metrics'] = high_conf_metrics
        summary['sm_low_confidence_metrics'] = low_conf_metrics
        summary['llm_ratio'] = routing['low_confidence']['ratio']
        summary['high_conf_ratio'] = routing['high_confidence']['ratio']
        if config.get('use_large_model', False) and llm_records:
            summary['clslog_low_confidence_metrics'] = llm_low_metrics

    util.save_json(summary, os.path.join(result_dir, 'clslog_summary.json'))
    if llm_records:
        util.save_json(llm_records, os.path.join(result_dir, 'llm_records.json'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLSLog on LogHub datasets.')
    parser.add_argument('--config', type=str, default='./config/clslog_bgl_demo.yaml')
    args = parser.parse_args()
    main(load_config(args.config))
