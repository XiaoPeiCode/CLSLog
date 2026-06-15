"""
LogHub data preprocessing for CLSLog.

Reference: LogCAE/utils/preprocessing.py and utils/SlidingWindow.py
CLSLog uses raw log Content (no log parsing dependency) with BERT embeddings.
"""

import os
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

LOGCAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LogCAE'))
if LOGCAE_ROOT not in sys.path:
    sys.path.insert(0, LOGCAE_ROOT)

from utils import preprocessing as logcae_preprocessing  # noqa: E402


class LogDataset(Dataset):
    def __init__(self, log_messages, tokenizer, max_length=128):
        self.log_messages = log_messages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.log_messages)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.log_messages[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )


def ensure_structured_log(dataset_name, dataset_dir='dataset', structured_log_path=None):
    if structured_log_path and os.path.exists(structured_log_path):
        return structured_log_path
    structured_path = os.path.join(dataset_dir, dataset_name, f'{dataset_name}.log_structured.csv')
    if not os.path.exists(structured_path):
        logcae_preprocessing.parsing(dataset_name, dataset_dir)
    return structured_path


def load_structured_log(structured_path, dataset_name):
    df = pd.read_csv(structured_path)
    if dataset_name == 'Zookeeper':
        df['Label'] = df['Level'].apply(lambda x: int(x == 'ERROR'))
    else:
        df['Label'] = df['Label'].apply(lambda x: int(x != '-'))
    return df


def _parse_datetime(df, dataset_name):
    if dataset_name == 'BGL':
        if 'Timestamp' in df.columns:
            return pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
        return pd.to_datetime(df['Date'], format='%Y.%m.%d', errors='coerce')
    if dataset_name == 'Zookeeper':
        if 'Date' in df.columns and 'Time' in df.columns:
            return pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        if 'LineId' in df.columns:
            return pd.to_numeric(df['LineId'], errors='coerce')
    raise ValueError(f'Unsupported datetime columns for dataset: {dataset_name}')


def evolutionary_time_split(df, dataset_name, train_ratio=0.8, gap_days=14):
    """
    Split logs chronologically with a gap between train and test periods.
    Earlier logs are used for training; test logs start at least gap_days later.
    """
    df = df.copy()
    df['__datetime__'] = _parse_datetime(df, dataset_name)
    df = df.dropna(subset=['__datetime__']).sort_values('__datetime__').reset_index(drop=True)

    if len(df) == 0:
        raise ValueError('No valid timestamps found for evolutionary split.')

    start_time = df['__datetime__'].iloc[0]
    end_time = df['__datetime__'].iloc[-1]
    train_end_time = start_time + (end_time - start_time) * train_ratio
    test_start_time = train_end_time + timedelta(days=gap_days)

    train_df = df[df['__datetime__'] <= train_end_time].drop(columns='__datetime__')
    test_df = df[df['__datetime__'] >= test_start_time].drop(columns='__datetime__')

    if len(train_df) == 0 or len(test_df) == 0:
        split_index = int(len(df) * train_ratio)
        gap_index = min(len(df), split_index + max(1, int(len(df) * 0.05)))
        train_df = df.iloc[:split_index].drop(columns='__datetime__')
        test_df = df.iloc[gap_index:].drop(columns='__datetime__')

    label_col = 'Label' if 'Label' in train_df.columns else 'label'
    stratify = train_df[label_col] if train_df[label_col].nunique() > 1 else None
    valid_df, train_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=stratify)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def generate_bert_embeddings(log_messages, tokenizer, model, device, batch_size=32, desc='Generating Embeddings'):
    dataset = LogDataset(log_messages, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.extend(output.last_hidden_state[:, 0, :].cpu().numpy())
    return np.asarray(embeddings)


def apply_sliding_window(df, window_size=10, step_size=10, embedding_method='mean'):
    windowed_embeddings = []
    windowed_labels = []
    contents = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        if embedding_method == 'mean':
            window_embedding = np.mean(np.vstack(window['embedding'].values), axis=0)
        elif embedding_method == 'individual':
            window_embedding = np.concatenate(window['embedding'].values, axis=0)
        else:
            raise ValueError("embedding_method must be 'mean' or 'individual'")

        content = ''.join(f' - {msg};\n' for msg in window['log_message'].values)
        windowed_embeddings.append(window_embedding)
        windowed_labels.append(int(window['label'].max()))
        contents.append(content)

    return pd.DataFrame({'embedding': windowed_embeddings, 'label': windowed_labels, 'content': contents})


def _build_sequence_dataframe(log_df, tokenizer, model, device, config, split_name):
    embeddings = generate_bert_embeddings(
        log_df['log_message'].tolist(),
        tokenizer,
        model,
        device,
        batch_size=config.get('emb_batch_size', 64),
        desc=f'BERT embeddings ({split_name})',
    )
    seq_df = log_df.copy()
    seq_df['embedding'] = list(embeddings)
    return apply_sliding_window(
        seq_df,
        window_size=config['window_size'],
        step_size=config['step_size'],
        embedding_method=config.get('embedding_method', 'mean'),
    )


def load_loghub_data(config, use_cache=True):
    dataset_name = config['dataset_name']
    dataset_dir = config.get('dataset_dir', 'dataset')
    cache_dir = config.get('cache_dir', './cache_dir/loghub')
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(
        cache_dir,
        f'{dataset_name}_w{config["window_size"]}_s{config["step_size"]}_{config.get("embedding_method", "mean")}.pkl',
    )
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            return pickle.load(file)

    structured_path = ensure_structured_log(
        dataset_name,
        dataset_dir,
        structured_log_path=config.get('structured_log_path'),
    )
    structured_df = load_structured_log(structured_path, dataset_name)

    if config.get('sample_size'):
        structured_df = structured_df.iloc[: config['sample_size']]

    log_df = pd.DataFrame({
        'log_message': structured_df['Content'],
        'label': structured_df['Label'],
    })

    split_input = structured_df.copy()
    split_input['log_message'] = structured_df['Content']
    split_input['label'] = structured_df['Label']

    if config.get('split_method', 'evolutionary') == 'evolutionary':
        train_df, valid_df, test_df = evolutionary_time_split(
            split_input,
            dataset_name,
            train_ratio=config.get('train_ratio', 0.8),
            gap_days=config.get('gap_days', 14),
        )
    else:
        train_valid_df, test_df = train_test_split(
            log_df,
            test_size=config.get('test_ratio', 0.2),
            random_state=42,
            stratify=log_df['label'] if log_df['label'].nunique() > 1 else None,
        )
        train_df, valid_df = train_test_split(
            train_valid_df,
            test_size=0.1,
            random_state=42,
            stratify=train_valid_df['label'] if train_valid_df['label'].nunique() > 1 else None,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'], cache_dir=config.get('cache_dir', './cache_dir'))
    model = BertModel.from_pretrained(config['bert_model'], cache_dir=config.get('cache_dir', './cache_dir'))
    model = model.to(device)

    results = {}
    for split_name, split_df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        split_log_df = pd.DataFrame({
            'log_message': split_df['log_message'] if 'log_message' in split_df else split_df['Content'],
            'label': split_df['label'] if 'label' in split_df else split_df['Label'],
        })
        results[split_name] = _build_sequence_dataframe(split_log_df, tokenizer, model, device, config, split_name)

    with open(cache_file, 'wb') as file:
        pickle.dump(results, file)
    return results
