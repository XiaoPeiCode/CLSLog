"""
CLSLog: Collaborating Large and Small Models for Log-based Anomaly Detection.

This is the initial cascade version (FSE Companion 2025):
1. Train a Siamese small model with contrastive loss on historical logs.
2. Downsample the training set with HDBSCAN.
3. Route high-confidence samples to the small model; low-confidence samples to the LLM.
4. Inject top-k context and SM scores into the LLM prompt for knowledge enhancement.
"""

import argparse
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

from modules import llm_utils
from utils import util
from utils.cluster_utils import apply_hdbscan
from utils.loghub_preprocessing import load_loghub_data


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


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


def train_siamese_network(train_embeddings, train_labels, test_embeddings, test_labels, config, device):
    embedding_dim = train_embeddings.shape[1]
    model = SiameseNetwork(embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            f1, pr, re = evaluate_sm(model, train_embeddings, train_labels, test_embeddings, test_labels, config)
            print(f"Epoch [{epoch + 1}/{config['cl_num_epochs']}], Loss: {total_loss / len(loader):.4f}, "
                  f"test F1: {f1:.4f}, Pr: {pr:.4f}, Re: {re:.4f}")
            if wandb is not None:
                wandb.log({'epoch': epoch + 1, 'loss': total_loss / len(loader), 'test_f1': f1, 'test_pr': pr, 'test_re': re})
            best_f1 = max(best_f1, f1)

    if wandb is not None:
        wandb.finish()
    print(f'Best SM F1 on test split during training: {best_f1:.4f}')
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
    if config.get('use_hdbscan', True):
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


def sm_predict(train_embeddings, train_labels, test_embeddings, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(train_embeddings)
    distances, indices = nbrs.kneighbors(test_embeddings)
    similarities = 1 - distances

    predictions = []
    confidences = []
    routed_to_llm = []
    details = []
    for i in range(len(test_embeddings)):
        neighbor_labels = train_labels[indices[i]]
        outputs = compute_sm_outputs(similarities[i], neighbor_labels)
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


def select_confidence_threshold(valid_confidences, valid_labels, valid_predictions, default=0.95):
    if len(valid_confidences) == 0:
        return default

    best_threshold = default
    best_f1 = -1.0
    for threshold in np.linspace(max(0.5, valid_confidences.min()), min(1.0, valid_confidences.max()), 20):
        routed_predictions = valid_predictions.copy()
        routed_predictions[valid_confidences <= threshold] = -1
        sm_mask = routed_predictions != -1
        if sm_mask.sum() == 0:
            continue
        f1 = f1_score(valid_labels[sm_mask], routed_predictions[sm_mask], pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)


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
):
    llm_predictions = {}
    llm_records = []
    if not config.get('use_large_model', False):
        print('use_large_model=False, low-confidence samples fall back to SM predictions.')
        for idx in low_conf_indices:
            llm_predictions[idx] = sm_details[idx]['sm_prediction']
        return llm_predictions, llm_records

    for idx in tqdm(low_conf_indices, desc='LLM inference'):
        neighbors = neighbor_indices[idx]
        topk_context = format_topk_context(
            train_contents[neighbors],
            train_labels[neighbors],
            similarities[idx],
        )
        sm_result = format_sm_result(sm_details[idx])
        user_prompt = prompt_dict['clslog_detect']['user_prompt'].format(
            topk_context=topk_context,
            sm_result=sm_result,
            input_logs=test_contents[idx],
        )
        prompt_message = [
            {'role': 'system', 'content': prompt_dict['clslog_detect']['system_prompt']},
            {'role': 'user', 'content': user_prompt},
        ]
        llm_answer, llm_usage = llm_utils.llm_json_results([prompt_message], config['LLM'])
        system_state = llm_answer[0].get('System State') or llm_answer[0].get('system_state')
        prediction = int(str(system_state).lower() == 'abnormal')
        llm_predictions[idx] = prediction
        llm_records.append({
            'index': int(idx),
            'true_label': int(test_labels[idx]),
            'prediction': prediction,
            'llm_answer': llm_answer[0],
            'llm_usage': llm_usage,
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


def main(config):
    set_seed(config.get('random_seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)

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

    siamese_model = None
    if config.get('use_siamese_network', True):
        start_time = time.time()
        siamese_model = train_siamese_network(
            train_embeddings, train_labels, test_embeddings, test_labels, config, device
        )
        print(f'Siamese training time: {time.time() - start_time:.2f}s')
        train_embeddings = normalize(generate_siamese_embeddings(siamese_model, train_embeddings), norm='l2')
        valid_embeddings = normalize(generate_siamese_embeddings(siamese_model, valid_embeddings), norm='l2')
        test_embeddings = normalize(generate_siamese_embeddings(siamese_model, test_embeddings), norm='l2')

    if config.get('use_hdbscan', True):
        train_embeddings, train_labels = apply_hdbscan(
            train_embeddings,
            train_labels,
            min_cluster_size=config.get('min_cluster_size', 15),
            min_samples=config.get('min_samples', 1),
        )

    sm_only = sm_predict(train_embeddings, train_labels, test_embeddings, config['k'])
    evaluate_predictions(test_labels, sm_only['predictions'], result_dir, tag='trained_sm_only')

    valid_sm = sm_predict(train_embeddings, train_labels, valid_embeddings, config['k'])
    confidence_threshold = config.get('confidence_threshold')
    if confidence_threshold is None:
        confidence_threshold = select_confidence_threshold(
            valid_sm['confidences'],
            valid_labels,
            valid_sm['predictions'],
        )
    print(f'Confidence threshold mu: {confidence_threshold:.4f}')

    test_sm = sm_predict(train_embeddings, train_labels, test_embeddings, config['k'])
    high_conf_mask = test_sm['confidences'] > confidence_threshold
    low_conf_indices = np.where(~high_conf_mask)[0]
    final_predictions = test_sm['predictions'].copy()

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
    )
    for idx, pred in llm_predictions.items():
        final_predictions[idx] = pred

    clslog_metrics = evaluate_predictions(test_labels, final_predictions, result_dir, tag='clslog')
    llm_ratio = len(low_conf_indices) / len(test_labels)
    summary = {
        'dataset': config['dataset_name'],
        'confidence_threshold': confidence_threshold,
        'llm_ratio': llm_ratio,
        'high_conf_ratio': float(high_conf_mask.mean()),
        'metrics': clslog_metrics,
        'train_size': int(len(train_df)),
        'valid_size': int(len(valid_df)),
        'test_size': int(len(test_df)),
    }
    util.save_json(summary, os.path.join(result_dir, 'clslog_summary.json'))
    if llm_records:
        util.save_json(llm_records, os.path.join(result_dir, 'llm_records.json'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLSLog on LogHub datasets.')
    parser.add_argument('--config', type=str, default='./config/clslog_bgl.yaml')
    args = parser.parse_args()
    main(load_config(args.config))
