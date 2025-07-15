# 简单的日志异常检测 Pipeline 使用 BERT 和 KNN 或孪生网络（启用多 GPU 加速）
import json
import time
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
import yaml
import argparse
import torch.nn as nn
import torch.optim as optim
from modules import llm_utils
import random
import ast
import hdbscan
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from modules.AutoEncoder import AE, train

from utils.cluster_utils import apply_faiss_hdbscan
from utils.loadEvlog import load_evlog_data
# 读取配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# 是否启用缓存
use_cache = True

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()
# 封装数据加载函数
def load_structured_log(file_path, dataset_name,sample_size=None):
    """
    返回内容包括：
    """
    structured_df = pd.read_csv(file_path)

    if sample_size:
        structured_df = structured_df.iloc[:sample_size, :]
    
    if dataset_name == "Zookeeper":
        structured_df["Label"] = structured_df["Level"].apply(lambda x: int(x == 'ERROR'))
#       df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    else:
        structured_df["Label"] = structured_df["Label"].apply(lambda x: int(x != '-'))
    return structured_df

# 封装数据分割函数
def split_dataset(log_df, method='random', test_size=0.2):
    print(f"split_method:{method}")
    if method == 'random':
        return train_test_split(log_df, test_size=test_size, random_state=42)
    elif method == 'sequential':
        split_index = int(len(log_df) * (1 - test_size))
        return log_df.iloc[:split_index], log_df.iloc[split_index:]
    else:
        raise ValueError("Invalid split method. Choose 'random' or 'sequential'.")

# 使用滑动窗口处理数据集
def apply_sliding_window(df, window_size=10, step_size=10, embedding_method='individual'):
    windowed_embeddings = []
    windowed_labels = []
    contents = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        # 将每个日志事件的嵌入向量堆叠起来，形成窗口的嵌入向量
        if embedding_method == 'individual':
            window_embedding = np.concatenate(window['embedding'].values, axis=0)
        elif embedding_method == 'mean':
            window_embedding = np.mean(np.vstack(window['embedding'].values), axis=0)  # 计算均值 embedding
        else:
            raise ValueError("Invalid embedding_method. Choose 'individual' or 'mean'.")
            
        content = ""
        for k in window['log_message'].values:
            content =   content  + " - " +  k + '；\n'
        windowed_embeddings.append(window_embedding)
        windowed_labels.append(window['label'].max())  # 如果窗口中有异常，则标记为异常
        contents.append(content)
    return pd.DataFrame({'embedding': windowed_embeddings, 'label': windowed_labels,'content':contents})

# 封装 BERT 嵌入生成函数
def generate_embeddings(log_messages, tokenizer, model, batch_size=32, desc="Generating Embeddings"):
    log_dataset = LogDataset(log_messages, tokenizer)
    log_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in tqdm(log_loader, desc=desc):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

# 定义一个 Dataset 类用于批量处理日志消息
class LogDataset(Dataset):
    def __init__(self, log_messages, tokenizer):
        self.log_messages = log_messages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.log_messages)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.log_messages[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return tokens

# 封装评估函数
def evaluate_model(test_df,predicted_label='predicted_label',label_col='label',file='evaluation_results.txt'):
    print("Classification Report:")
    classification_rep = classification_report(test_df[label_col], test_df[predicted_label],digits=4)
    print(classification_rep)

    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(test_df[label_col], test_df[predicted_label])
    print(conf_matrix)

    # 计算异常样本的 Recall, Precision 和 F1 分数
    anomaly_recall = recall_score(test_df[label_col], test_df[predicted_label], pos_label=1)
    anomaly_precision = precision_score(test_df[label_col], test_df[predicted_label], pos_label=1)
    anomaly_f1 = f1_score(test_df[label_col], test_df[predicted_label], pos_label=1)

    print(f"Anomaly Recall: {anomaly_recall:.4f}")
    print(f"Anomaly Precision: {anomaly_precision:.4f}")
    print(f"Anomaly F1 Score: {anomaly_f1:.4f}")

    # 保存评估结果到文件
    eval_result_path = os.path.join(result_dir, file)
    with open(eval_result_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_rep + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix) + "\n")
        f.write(f"Anomaly Recall: {anomaly_recall:.4f}\n")
        f.write(f"Anomaly Precision: {anomaly_precision:.4f}\n")
        f.write(f"Anomaly F1 Score: {anomaly_f1:.4f}\n")

# 定义孪生网络（Siamese Network）
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        return self.fc1(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# 对比损失函数
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# 训练孪生网络

def train_siamese_network(train_embeddings, train_labels, test_embeddings,test_labels,embedding_dim, num_epochs=10, batch_size=256):
    siamese_model = SiameseNetwork(embedding_dim).to(device)
    optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)
    train_dataset = SiameseDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    wandb.init(
    project= wandb_project, # train_df: 'spark2'
    mode=config['wandb_mode'],

    name=f"Train_SM({config['train_df']},{config['test_df']},{num_epochs},{batch_size})",
    tags=['Train_SM', config['dataset_name']],
    config={
        "component": "Train_SM",
        "dataset": config['dataset_name'],
        'param_name': {
            # 数据集划分配置
            'train_df': config['train_df'],
            'test_df': config['test_df'],
            'num_epoch':num_epochs,
            'batch_size':batch_size,
            'config':config
            }
        }
    )
                
    best_f1,best_epoch=0,0
    for epoch in range(num_epochs):
        total_loss = 0
        siamese_model.train()
        for anchor, positive, label in train_loader:
            optimizer.zero_grad()
            output1, output2 = siamese_model(anchor.to(device), positive.to(device))
            loss = contrastive_loss(output1, output2, label.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        siamese_model.eval()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
        
        # cloned_train_embeddings = train_embeddings.copy()
        # cloned_test_embeddings = test_embeddings.copy()
        cloned_test_embeddings = test_embeddings
        cloned_train_embeddings = train_embeddings

        # f1,pr,re = evaluate_knn(siamese_model,cloned_train_embeddings,train_labels,cloned_train_embeddings,train_labels)
        # print(f"train f1:{f1:.4f},pr:{pr:.4f},re:{re:.4f}")

        if config['eval_in_train']:
            f1,pr,re = evaluate_knn(siamese_model,cloned_train_embeddings,train_labels,cloned_test_embeddings,test_labels,config['k'])
            print(f"test f1:{f1:.4f},pr:{pr:.4f},re:{re:.4f}")
  
            wandb.log({"epoch":epoch+1,"Loss": total_loss / len(train_loader), "test f1": f1, "test pr": pr, "test re": re})
            if best_f1 < f1:
                best_f1 = f1
                best_epoch = epoch
    
    K_list = [1,2,3,4,5,6,7,8,9]
    f1s,prs,res = evaluate_knn(siamese_model,cloned_train_embeddings,train_labels,cloned_test_embeddings,test_labels,K_list)
    for i in range(len(K_list)):
        wandb.log({'epoch':K_list[i],"k":K_list[i],"f1 (k)":f1s[i],"pr(k)":prs[i],"re (k)":res[i]})
    print(f"best f1:{best_f1},best epoch:{best_epoch}")
    wandb.finish()

    return siamese_model

def calAnomalyScore(similarities,neighbor_labels):
    score = 0
    for s,l in zip(similarities,neighbor_labels):
        if l == 0:
         score -= s 
    else: # 大于0表示异常
        score += s
    return score
def identifyEvLogs(similarities,neighbor_labels,score_threshold=1,similarity_threshold=0.99):
    score = 0
    for s,l in zip(similarities,neighbor_labels):
        if l == 0:
            score -= s 
        else: # 大于0表示异常
            score += s

    if abs(score) <= abs(score_threshold) or np.mean(similarities) < similarity_threshold:
        return True
    else:
        return False
    
def knn_anomaly_detection(train_embeddings, train_labels, test_embeddings, k, fuzzy_frac, similarity_threshold=0.99):
    print("start knn_anomaly_detection")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(train_embeddings)
    distances, indices = nbrs.kneighbors(test_embeddings)
    similarities = 1 - distances  # 计算余弦相似度

    knn_predictions, uncertain_samples_index, uncertain_knn_predictions = [], [], []
    # for i, neighbors in enumerate(tqdm(indices, desc="KNN Predictions")):
    for i, neighbors in enumerate(indices):
        neighbor_labels = train_labels[neighbors]
        positive_count, negative_count = np.sum(neighbor_labels), len(neighbor_labels) - np.sum(neighbor_labels)
        
        anomaly_prediction = int(positive_count > k / 2)
        knn_predictions.append(anomaly_prediction)

        similarity_sum = np.sum(similarities[i])

        # 识别模糊样本和概念漂移日志
        if identifyEvLogs(similarities[i],neighbor_labels):
            uncertain_samples_index.append(i)
            uncertain_knn_predictions.append(anomaly_prediction)

    return knn_predictions, uncertain_samples_index, uncertain_knn_predictions

def apply_hdbscan(train_embeddings, train_labels, min_cluster_size=15, min_samples=1, cache_path=None):
    """
    使用 HDBSCAN 进行聚类，并同步更新 train_labels
    :param train_embeddings: 原始训练集的嵌入 (numpy array)
    :param train_labels: 原始训练集的标签 (numpy array)
    :param min_cluster_size: HDBSCAN 最小簇大小
    :param min_samples: HDBSCAN 最小样本数
    :param cache_path: 如果提供，则尝试从缓存加载 HDBSCAN 结果，避免重复计算
    :return: (HDBSCAN 处理后的 train_embeddings, train_labels)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached HDBSCAN results from {cache_path}")
        return joblib.load(cache_path)

    print("Applying HDBSCAN clustering for downsampling...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(train_embeddings)

    # 仅保留核心点
    # 等于-1，表示不是核心点
    mask = (cluster_labels != -1) | (train_labels == 1)
    filtered_embeddings = train_embeddings[mask]
    filtered_labels = train_labels[mask]  # 确保标签也对应减少

    if cache_path:
        joblib.dump((filtered_embeddings, filtered_labels), cache_path)
    
    if config['cluster_detail']:
        # wan
        print(f"Original samples: {train_embeddings.shape[0]}")
        print(f"Reduced samples: {filtered_embeddings.shape[0]} ({100 * filtered_embeddings.shape[0] / train_embeddings.shape[0]:.2f}% retained)")
        
        # 统计正常/异常样本比例
        normal_before = np.sum(train_labels == 0)
        abnormal_before = np.sum(train_labels == 1)
        normal_after = np.sum(filtered_labels == 0)
        abnormal_after = np.sum(filtered_labels == 1)

        print(f"Before HDBSCAN: Normal={normal_before}, Abnormal={abnormal_before}")
        print(f"After HDBSCAN: Normal={normal_after}, Abnormal={abnormal_after}")

    print(f"Reduced training embeddings from {train_embeddings.shape[0]} to {filtered_embeddings.shape[0]} using HDBSCAN")
    return filtered_embeddings, filtered_labels

def evaluate_knn(siamese_model, train_dataset,train_labels,test_dataset,test_labels,K=5,batch_size=512,):
    siamese_model.eval()

    train_embeddings = generate_siamese_embeddings(siamese_model, train_dataset, batch_size=batch_size)
    test_embeddings = generate_siamese_embeddings(siamese_model, test_dataset, batch_size=batch_size)
    
    if config['use_hdbscan']:
        # train_embeddings, train_labels = apply_hdbscan(train_embeddings, train_labels,min_cluster_size=config['min_cluster_size'], min_samples=config['min_samples'])
        train_embeddings, train_labels = apply_faiss_hdbscan(train_embeddings, train_labels,min_cluster_size=config['min_cluster_size'], min_samples=config['min_samples'])

    if isinstance(K,list): # 选择最佳的k
        best_f1,best_k = 0,-1
        best_pr,best_re = 0,0
        prs,res,f1s = [],[],[]
        for k in K:
            knn_predictions, uncertain_samples, uncertain_knn_predictions = knn_anomaly_detection(
            train_embeddings, train_labels, test_embeddings,k, config["fuzzy_frac"], config["similarity_threshold"]
        )
            
            re = recall_score(test_labels,knn_predictions, pos_label=1)
            pr = precision_score(test_labels, knn_predictions, pos_label=1)
            f1 = f1_score(test_labels,knn_predictions, pos_label=1)
            print(f"k:{k},f1:{f1},pr:{pr},re:{re}")
            
            prs.append(pr)
            res.append(re)
            f1s.append(f1)
            if f1 > best_f1:
                best_f1 = f1
                best_pr = best_pr
                best_re = best_re
                best_k = k
        print(f"best k:{best_k},f1:{best_f1},pr:{best_pr},re:{best_re}")
        
        return f1s,prs,res

    else:

        knn_predictions, uncertain_samples, uncertain_knn_predictions = knn_anomaly_detection(
            train_embeddings, train_labels, test_embeddings, K, config["fuzzy_frac"], config["similarity_threshold"]
        )
        re = recall_score(test_labels,knn_predictions, pos_label=1)
        pr = precision_score(test_labels, knn_predictions, pos_label=1)
        f1 = f1_score(test_labels,knn_predictions, pos_label=1)

        return f1,pr,re

from torch.utils.data import TensorDataset, DataLoader

def generate_siamese_embeddings(siamese_model, embeddings, batch_size=512):
    """
    采用 GPU 批量计算方式生成孪生网络的嵌入，以充分利用 GPU 并行计算能力。
    :param siamese_model: 训练好的孪生网络
    :param embeddings: 原始嵌入，numpy 数组格式 (N, embedding_dim)
    :param batch_size: 批量大小，默认为 512
    :return: 经过孪生网络转换后的嵌入
    """
    device = next(siamese_model.parameters()).device  # 获取模型所在设备
    siamese_model.eval()  # 进入评估模式
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)  # 将数据加载到 GPU
    dataset = TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 批量加载数据

    transformed_embeddings = []
    with torch.no_grad():  # 关闭梯度计算，加速推理
        # for batch in tqdm(dataloader, desc="Generating Siamese Embeddings"):
        for batch in dataloader:
    
            batch_embeddings = batch[0]
            transformed_batch = siamese_model.forward_once(batch_embeddings)  # 批量计算
            transformed_embeddings.append(transformed_batch.cpu().numpy())  # 移回 CPU 并转换为 numpy 数组

    return np.vstack(transformed_embeddings)  # 合并所有批次的嵌入


# def eval
# 定义孪生网络的数据集类
class SiameseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        anchor = self.embeddings[idx]
        positive_idx = (idx + 1) % len(self.embeddings)  # 简单地选择下一个样本作为正样本
        positive = self.embeddings[positive_idx]
        label = self.labels[idx]
        return torch.tensor(anchor, dtype=torch.float), torch.tensor(positive, dtype=torch.float), torch.tensor(label, dtype=torch.float)


def stats_summary(data_list):
    """仅用Python内置库实现"""
    if data_list is None and len(data_list) > 0:
        return "Empty list provided"
    
    sorted_data = sorted(data_list)
    n = len(sorted_data)
    
    def get_quantile(p):
        idx = int(p * (n - 1))
        return sorted_data[idx]
    
    stats = {
        'mean': sum(data_list) / n,
        'min': sorted_data[0],
        'max': sorted_data[-1],
        'Q10': get_quantile(0.1),
        'Q20': get_quantile(0.2),
        'Q30': get_quantile(0.3),
        'Q40': get_quantile(0.4),
        'Q50': get_quantile(0.5),
        'Q60': get_quantile(0.6),
        'Q70': get_quantile(0.7),
        'Q80': get_quantile(0.8),
        'Q90': get_quantile(0.9),
        'Q95': get_quantile(0.95)

    }
    return ' '.join(f"{k}:{v:.5f}" for k, v in stats.items())


def main(config):
    global result_dir
    result_dir = config['result_dir'] + f"/{config['train_df']}_{config['test_df']}_{config['use_large_model']}/"
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查是否有可用的 GPU
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    dfs = load_evlog_data(file_path=f"/home/xiaopei/XPLog/Dataset/Logevol/{config['train_df']}",use_cache=True)    # 嵌入已经生成，可以直接使用，无需重复生成
    train_df,_,valid_df = dfs['train'],dfs['test'],dfs['valid']
    # dfs = load_evlog_data(file_path="/home/xiaopei/XPLog/Dataset/Logevol/spark3",use_cache=True)    # 嵌入已经生成，可以直接使用，无需重复生成
    # _,test_df,valid_df = dfs['train'],dfs['test'],dfs['valid']
    dfs = load_evlog_data(file_path=f"/home/xiaopei/XPLog/Dataset/Logevol/{config['test_df']}",use_cache=True)    # 嵌入已经生成，可以直接使用，无需重复生成
    _,test_df,_ = dfs['train'],dfs['test'],dfs['valid']
    
    # test_df = pd.read_csv("/home/xiaopei/XPLog/Dataset/Logevol/spark2/test_df.csv")
    # test_df['embedding'] = test_df['embedding'].apply(ast.literal_eval)

    # train_df = test_df
    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)
    valid_embeddings = np.vstack(valid_df['embedding'].values)
    
    train_labels = train_df['label'].values
    test_labels = test_df['label'].values
    valid_labels = valid_df['label'].values
    train_contents = train_df['content'].values

    print(f"training embeddings from {train_embeddings.shape[0]}")
    print(f"test embeddings from {test_embeddings.shape[0]}")

    
    # 训练自编器
    """
    config:
    AE:
        hidden_dims: [64,128,14]
        model_path: "./cache_dir/model/bgl_detection.pkl"
        batch_size: 256
        lr : 0.001
        epoch: 30
    """
    def train_AE(train_embeddings,config):
        # wandb.init(project= wandb_project, # train_df: 'spark2'
        #     name=f"AE ({config['train_df']},{config['test_df']})",
        #     tags=['coordinator', config['dataset_name']],
        #     config={
        #         "component": "Train AE",
        #         "dataset": config['dataset_name'],
        #         'param_name': {
        #             # 数据集划分配置
        #             'train_df': config['train_df'],
        #             'test_df': config['test_df'],
        #             'config':config
        #             }
        #         })
        print("**********AE train Start ************")

        train_dataset_ae = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float), torch.zeros(len(train_embeddings)))  # 自编码器不需要目标值，但可以用作占位符
        train_dataloader = DataLoader(train_dataset_ae, batch_size=config['AE']['batch_size'], shuffle=True, pin_memory=True)
        
        model = AE(input_dim=train_embeddings.shape[1], hidden_dims=config["AE"]["hidden_dims"])
        optimizer = optim.Adam(model.parameters(), lr=config["AE"]["lr"])

        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"torch.cuda.device_count():{torch.cuda.device_count()}")
            model = nn.DataParallel(model)

    #     # train
        train_losses = []
        for epoch in tqdm(range(config["AE"]["epoch"]), desc="Training: "):
            # logging.info(f"--------Epoch {epoch + 1}/{epochs}-------")
            train_losses = []
            train_loss, y_preds = train(model, train_dataloader, optimizer, device)
            # wandb.log({'train_loss':train_loss})
            train_losses.append(train_loss)
            print(f"Epoch {epoch + 1}/{config['AE']['epoch']}, Loss: {train_loss:.8f}")
        # wandb.finish()
        # 保存模型
        # torch.save(model.state_dict(), "autoencoder_model.pth")
        # print("模型已保存到 autoencoder_model.pth")
        print("**********AE train end ************")
        return model
    
    def infer_AE(model, test_embeddings, batch_size=512):
        """
        计算测试嵌入的自编码器重构损失
        :param model: 训练好的自编码器模型
        :param test_embeddings: 测试嵌入数组
        :param batch_size: 批量大小
        :return: 测试嵌入的重构损失列表
        """
        device = next(model.parameters()).device  # 获取模型所在设备
        model.eval()  # 设置模型为评估模式
        test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
        dataset = TensorDataset(test_embeddings_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        losses = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(device)
                res_dict = model(inputs)
                batch_loss = res_dict["y_pred"]
                losses.extend(batch_loss.cpu().numpy().tolist())  # 将损失从 GPU 移回 CPU 并转换为列表

        return losses


          # 纯粹的检KNN，输入train_embeddings, train_labels, test_embeddings,k)
    def knn_eval(train_embeddings, train_labels, test_embeddings, test_labels,k):
        if len(train_embeddings) == 0 or len(test_embeddings) == 0:
            print("train_embeddings or test_embeddings is empty, returning empty results.")
            return [],0,0,0,0
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(train_embeddings)
        distances, indices = nbrs.kneighbors(test_embeddings)
        
        # 向量化计算邻居标签的统计量
        neighbor_labels = train_labels[indices]  # 形状: (n_test_samples, k)
        positive_counts = np.sum(neighbor_labels, axis=1)  # 每个测试样本的正类邻居数
        # 向量化预测（无需循环）
        knn_predictions = (positive_counts > k / 2).astype(int)
    
        re = recall_score(test_labels, knn_predictions, pos_label=1)
        pr = precision_score(test_labels, knn_predictions, pos_label=1)
        f1 = f1_score(test_labels, knn_predictions, pos_label=1)
        acc = np.mean(test_labels == knn_predictions)  # Calculate accuracy
        return knn_predictions,f1,pr,re,acc


     
    # 给定一个阈值，输出两边的结果，还有对应的数量；测试集和label，loss
    def as_threshold_sensitivity_analysis(valid_losses, test_losses,test_embeddings, test_labels, train_embeddings, train_labels, config,save_path):
        import matplotlib.pyplot as plt
        knn_predictions, f1_all, pr_all, re_all,acc = knn_eval(train_embeddings, train_labels, test_embeddings, test_labels, config['k'])

        print(f"ALL f1:{f1_all:.4f}, pr:{pr_all:.4f}, re:{re_all:.4f},acc:{acc:.4f}")

        percentiles = np.arange(0, 99, 5)  # 从10%到90%，每隔10%进行分析
        results = []
        high_loss_f1_scores = []
        low_loss_f1_scores = []

        mode = 'test'
        if len(valid_losses) == len(test_losses):
            mode = 'valid'
        for p in percentiles:
            threshold = np.percentile(valid_losses, p)
            low_loss_indices = [i for i, loss in enumerate(test_losses) if loss < threshold]
            high_loss_indices = [i for i, loss in enumerate(test_losses) if loss >= threshold]

            low_loss_embeddings = test_embeddings[low_loss_indices]
            high_loss_embeddings = test_embeddings[high_loss_indices]
            low_loss_labels = test_labels[low_loss_indices]
            high_loss_labels = test_labels[high_loss_indices]

            knn_predictions, f1_low, pr_low, re_low,acc_low = knn_eval(train_embeddings, train_labels, low_loss_embeddings, low_loss_labels, config['k'])
            knn_predictions, f1_high, pr_high,re_high,acc_high = knn_eval(train_embeddings, train_labels, high_loss_embeddings, high_loss_labels, config['k'])

            results.append({
                'percentile': p,
                'low_loss_f1': f1_low,
                'low_loss_pr': pr_low,
                'low_loss_re': re_low,
                'high_loss_f1': f1_high,
                'high_loss_pr': pr_high,
                'high_loss_re': re_high,
            })

            total_samples = len(test_embeddings)
            high_loss_f1_scores.append(f1_high)
            low_loss_f1_scores.append(f1_low)
            low_loss_ratio = len(low_loss_indices) / total_samples * 100
            high_loss_ratio = len(high_loss_indices) / total_samples * 100

            print(f"Percentile {p}%:")
            print(f"Low Loss samples: {len(low_loss_indices)} ({low_loss_ratio:.2f}%)")
            print(f"High Loss samples: {len(high_loss_indices)} ({high_loss_ratio:.2f}%)")
            print(f"Low Loss f1:{f1_low:.4f}, pr:{pr_low:.4f}, re:{re_low:.4f},acc:{acc_low:.4f}")
            print(f"High Loss f1:{f1_high:.4f}, pr:{pr_high:.4f}, re:{re_high:.4f},acc:{acc_high:.4f}")

        # def plot_and_save_list(x,y,y1,save_path):
        plt.figure(figsize=(10, 6))
        # if x:
        plt.plot(percentiles, high_loss_f1_scores, marker='o', label='high_loss_f1_scores')
        plt.plot(percentiles, low_loss_f1_scores, marker='o', label=' low_loss_f1_scores')

        # else:
        #     plt.plot(y, marker='o', label='F1 Score')
        plt.xlabel('Percentile')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Sensitivity Analysis')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        # plot_and_save_list(percentiles,high_loss_f1_scores,low_loss_f1_scores,save_path)
        return results,high_loss_f1_scores
        
    # AE_model = train_AE(train_embeddings=train_embeddings, config=config)
    # valid_losses = infer_AE(AE_model, valid_embeddings, batch_size=512)
    # test_losses = infer_AE(AE_model, test_embeddings, batch_size=512)
    
    # save_path = "cor_sensi_AE_valid.png"
    # as_threshold_sensitivity_analysis(valid_losses,valid_losses, valid_embeddings, valid_labels, train_embeddings, train_labels, config,save_path)
    #         # 检查是否使用孪生网络
    # save_path = "cor_sensi_AE_test.png"
    # as_threshold_sensitivity_analysis(valid_losses,test_losses, test_embeddings, test_labels, train_embeddings, train_labels, config,save_path)

    original_train_embeddings = train_embeddings.copy()
    original_test_embeddings = test_embeddings.copy()
    original_valid_embeddings = valid_embeddings.copy()
    if config.get('use_siamese_network', False):
        print("训练孪生网络")
        embedding_dim = train_embeddings.shape[1]
        siamese_model = train_siamese_network(train_embeddings, train_df['label'].values,test_embeddings,test_df['label'].values, embedding_dim,num_epochs=config["cl_num_epochs"],
                                              batch_size=config['batch_size'])
        # 使用孪生网络生成新的嵌入
        train_embeddings = generate_siamese_embeddings(siamese_model, train_embeddings, batch_size=512)
        test_embeddings = generate_siamese_embeddings(siamese_model, test_embeddings, batch_size=512)
        valid_embeddings = generate_siamese_embeddings(siamese_model, valid_embeddings, batch_size=512)

        train_embeddings = normalize(train_embeddings, norm='l2')  # 若未归一化，需添加此步骤
        test_embeddings = normalize(test_embeddings, norm='l2')  # 若未归一化，需添加此步骤
        valid_embeddings = normalize(valid_embeddings, norm='l2')  # 若未归一化，需添加此步骤

        # train_embeddings = [siamese_model.forward_once(torch.tensor(e, dtype=torch.float).to(device)).cpu().detach().numpy() for e in train_embeddings]
        # test_embeddings = [siamese_model.forward_once(torch.tensor(e, dtype=torch.float).to(device)).cpu().detach().numpy() for e in test_embeddings]
        # train_embeddings = np.vstack(train_embeddings)
        # test_embeddings = np.vstack(test_embeddings)
        
    AE_model = train_AE(train_embeddings=original_train_embeddings, config=config)
    valid_losses = infer_AE(AE_model, original_valid_embeddings, batch_size=512)
    test_losses = infer_AE(AE_model, original_test_embeddings, batch_size=512)
    save_path = f"cor_sensi_AE_CL_valid_{config['test_df']}.png"
    as_threshold_sensitivity_analysis(valid_losses,valid_losses, valid_embeddings, valid_labels, train_embeddings, train_labels, config,save_path)
            # 检查是否使用孪生网络
    save_path = f"cor_sensi_AE_CL_test_{config['test_df']}.png"
    as_threshold_sensitivity_analysis(valid_losses,test_losses, test_embeddings, test_labels, train_embeddings, train_labels, config,save_path)

     # 第三步：KNN 异常检测

    # 1. 计算训练集相似度阈值（排除自身）
    nbrs_train = NearestNeighbors(n_neighbors=2, metric='cosine').fit(train_embeddings)
    distances_train, _ = nbrs_train.kneighbors(train_embeddings)
    train_max_similars = 1 - distances_train[:, 1]  # 取第二近邻（排除自身）
    train_similars = 1 - distances_train  # 取第二近邻（排除自身）
    # train_max_similars = np.max([i for i in train_similars[i]])  # 取第二近邻（排除自身）
    print(stats_summary([np.max(i) for i in train_similars]))


    # 2. 构建KNN模型（余弦相似度）
    nbrs_test = NearestNeighbors(n_neighbors=config['k'], metric='cosine').fit(train_embeddings)
    test_distances, test_indices = nbrs_test.kneighbors(test_embeddings)
    test_similarities = 1 - test_distances  # 转换为相似度值[1,6](@ref)
    test_max_similars = [np.max(i) for i in test_similarities]  # 取第二近邻（排除自身）


      # 确定Fuzzy Logs阈值（基于训练集信息熵的75%分位数）
    train_entropies = [entropy(np.bincount(train_labels[nbrs_test.kneighbors([emb])[1][0]])) 
                      for emb in train_embeddings]
       # 确定Fuzzy Logs阈值（基于训练集信息熵的75%分位数）
    test_entropies = [entropy(np.bincount(train_labels[nbrs_test.kneighbors([emb])[1][0]])) 
                      for emb in test_embeddings]
       
    print("train_max_similars")
    print(stats_summary(train_max_similars))
    print("test_similarities")
    print(stats_summary( 1 - test_distances[:,1]))
    print("train_entropies")
    print(stats_summary(train_entropies))
    
    


    # 测试选择最佳的参数
    wandb.init(
    project= wandb_project, # train_df: 'spark2'
    mode=config['wandb_mode'],
    name=f"coordinator ({config['train_df']},{config['test_df']},evlog,fuzzy log)",
    tags=['coordinator', config['dataset_name']],
    config={
        "component": "Train_SM",
        "dataset": config['dataset_name'],
        'param_name': {
            # 数据集划分配置
            'train_df': config['train_df'],
            'test_df': config['test_df'],
            'config':config
            }
        }
    )

    th_list = np.arange(0,1,0.05)
    # 理论上，相似度越低，判断效果越差； （低->高）
    # 理论上，信息熵越大，不确定性越高，预测效果越低（高->低）
    fuzzy_log_thresholds = np.arange(0.9,1,0.001)

    tag_3 =[]
    for i in len(fuzzy_log_thresholds):
        tag_1,tag_2 = [],[]
        # fuzzy_log_threshold = np.quantile(train_entropies, th)  # 自动适应数据分布test_entropies
        # evlog_threshold = np.quantile(train_max_similars, th) test_max_similars
        
        fuzzy_log_threshold = np.quantile(test_entropies, fuzzy_log_thresholds[i])  # 自动适应数据分布
        evlog_threshold = np.quantile(test_max_similars, th) 

        for i in range(len(test_embeddings)):
            # test_similarities = 1 - test_distances  # 转换为相似度值[1,6](@ref)
            # max_sim = np.max(test_similarities[i])
            max_sim = np.max(test_similarities[i])
            neighbor_labels = train_labels[test_indices[i]]
            
            ent = entropy(np.bincount(neighbor_labels))
            pos_count = np.sum(neighbor_labels == 1)
            neg_count = np.sum(neighbor_labels == 0)
            label_diff = abs(pos_count - neg_count)
            
            if max_sim <= evlog_threshold:
                tag_1.append(i)  # 演化日志
                
            if ent >= fuzzy_log_threshold:  # 允许差距阈值 #  or label_diff <= 1
                tag_2.append(i)  # 模糊样本
            if label_diff <= 1 and th == th_list[0]:
                tag_3.append(i)
            
        print(f" start Q{th},fuzzy_log_threshold:{fuzzy_log_threshold},evlog_threshold:{evlog_threshold}")
        knn_predictions,f1_1,pr_1,re_1 = knn_eval(train_embeddings,train_labels, test_embeddings[tag_1], test_labels[tag_1],config['k'])
        print(f"Evlogs candinate knn evluate result({len(knn_predictions)}):")
        print(f"f1:{f1_1:.5f},pr:{pr_1:.5f},re:{re_1:.5f}")
        
        knn_predictions,f1_2,pr_2,re_2 = knn_eval(train_embeddings,train_labels, test_embeddings[tag_2], test_labels[tag_2],config['k'])
        print(f"Fuzzy Logs knn evluate result({len(knn_predictions)}):")
        print(f"f1:{f1_2:.5f},pr:{pr_2:.5f},re:{re_2:.5f}")

        print("end\n")
        tag1_ratio = len(tag_1) / len(test_embeddings) * 100
        tag2_ratio = len(tag_2) / len(test_embeddings) * 100

        wandb.log({"epoch":th,"threshold":th,'evlog_threshold':evlog_threshold,'fuzzy_log_threshold':fuzzy_log_threshold,
                   "EvLogs ratio(%)":tag1_ratio,"EvLogs F1":f1_1,"EvLogs Pr":pr_1,"Evlogs Re":re_1,
                   "FyLogs ratio(%)":tag2_ratio,"FyLogs F1":f1_2,"FyLogs Pr":pr_2,"FyLogs Re":re_2,
                   })

    knn_predictions,f1_3,pr_3,re_3 = knn_eval(train_embeddings,train_labels, test_embeddings[tag_3], test_labels[tag_3],config['k'])
    print(f"Lable diff <= 1 knn evluate result({len(knn_predictions)}):")
    print(f"f1:{f1_3:.5f},pr:{pr_3:.5f},re:{re_3:.5f}")
        
    wandb.finish()
    
    fuzzy_log_threshold = np.quantile(train_entropies, 0.001)  # 自动适应数据分布
    evlog_threshold = np.quantile(train_max_similars, 0.80)
    # 3. 分类逻辑
    tags = []    
    for i in range(len(test_embeddings)):
        max_sim = np.max(test_similarities[i])
        neighbor_labels = train_labels[test_indices[i]]
        
        ent = entropy(np.bincount(neighbor_labels))

        pos_count = np.sum(neighbor_labels == 1)
        neg_count = np.sum(neighbor_labels == 0)
        label_diff = abs(pos_count - neg_count)
        
        if max_sim < evlog_threshold:
            tags.append(1)  # 演化日志
        elif ent >= fuzzy_log_threshold or label_diff <= 1:  # 允许差距阈值
            tags.append(2)  # 模糊样本
        else:
            tags.append(3)  # 普通样本
    # 4. 按tag分类结果
    tag1_indices = np.where(np.array(tags) == 1)[0].tolist()
    tag2_indices = np.where(np.array(tags) == 2)[0].tolist()
    tag3_indices = np.where(np.array(tags) == 3)[0].tolist()

    print(f"演化日志样本数: {len(tag1_indices)}")
    print(f"模糊样本数: {len(tag2_indices)}")
    print(f"普通样本数: {len(tag3_indices)}")
            
    # 5.对每种分类进行决策
   
    knn_predictions,f1,pr,re = knn_eval(train_embeddings, train_labels, test_embeddings, test_labels,config['k'])
    print(f"ALL Logs knn evluate result({len(knn_predictions)}):")
    print(f"f1:{f1:.5f},pr:{pr:.5f},re:{re:.5f}")
    
    knn_predictions,f1,pr,re = knn_eval(train_embeddings,train_labels, test_embeddings[tag1_indices], test_labels[tag1_indices],config['k'])
    print(f"Evlogs candinate knn evluate result({len(knn_predictions)}):")
    print(f"f1:{f1:.5f},pr:{pr:.5f},re:{re:.5f}")
    
    knn_predictions,f1,pr,re = knn_eval(train_embeddings,train_labels, test_embeddings[tag2_indices], test_labels[tag2_indices],config['k'])
    print(f"Fuzzy Logs knn evluate result({len(knn_predictions)}):")
    print(f"f1:{f1:.5f},pr:{pr:.5f},re:{re:.5f}")
    
    knn_predictions,f1,pr,re = knn_eval(train_embeddings,train_labels, test_embeddings[tag3_indices], test_labels[tag3_indices],config['k'])
    print(f"Other Logs knn evluate result({len(knn_predictions)}):")
    print(f"f1:{f1:.5f},pr:{pr:.5f},re:{re:.5f}")
       



    
    # KNN 异常检测
    k = config['k']
    nbrs = NearestNeighbors(n_neighbors=config['k'], metric='cosine').fit(train_embeddings)
    # 进行 KNN 查询，使用 tqdm 展示进度条
    distances, indices = nbrs.kneighbors(test_embeddings) # (m, k)
    similarities = 1 - distances  # 余弦相似度
    similarity_values = similarities.flatten()  # 展平所有相似度值为一维数组
    similarity_threshold = np.percentile(similarity_values, 5)  # 计算Q90阈值
    print(f"similarity_threshold:{similarity_threshold}")
    print(f"Calculated similarity threshold (Q5): {similarity_threshold}")
    temp_df = pd.DataFrame()
    temp_df['all_similarities'] = similarity_values
    print(temp_df.describe())
    
    uncertain_knn_predictions = []
    knn_predictions = []
    uncertain_samples = []
    scores = []
    for i, neighbors in enumerate(tqdm(indices, desc="Calculating Predictions")):
        neighbor_labels = train_labels[neighbors]
        positive_count = np.sum(neighbor_labels)
        negative_count = len(neighbor_labels) - positive_count
        score = calAnomalyScore(similarities[i],neighbor_labels)
        scores.append(abs(score))
        anomaly_prediction = int(np.sum(neighbor_labels) > k / 2)
        similar_samples = train_contents[neighbors]
        knn_predictions.append(anomaly_prediction)

        similarity_sum = np.sum(similarities[i])
        sum_sim_th = config['similarity_threshold'] * config['k']
        # 如果正负样本比例接近，或者相似度之和低于阈值，标记为模糊样本或演化日志
        similarity_threshold = 0.99
        if identifyEvLogs(similarities[i],neighbor_labels,score_threshold=2,similarity_threshold=similarity_threshold):
            uncertain_samples.append((i, test_df.iloc[i, test_df.columns.get_loc('content')], test_df.iloc[i, test_df.columns.get_loc('label')],
                                    similar_samples, similarities[i], neighbor_labels))
            uncertain_knn_predictions.append(anomaly_prediction)

    temp_df = pd.DataFrame()
    temp_df['scores'] = scores
    print(temp_df.describe())

    
    # 统计模糊样本的比例
    uncertain_ratio = len(uncertain_samples) / len(test_df)
    print(f"Uncertain sample ratio: {(uncertain_ratio * 100):.2f} % ({len(uncertain_samples)}/ {len(test_df)})")
    
    # knn_predictions, uncertain_samples_index, uncertain_knn_predictions = knn_anomaly_detection(
    #     train_embeddings, train_df['label'].values, test_embeddings, config['k'], config["fuzzy_frac"], config["similarity_threshold"]
    # )


    test_df['cl_knn_predicted_label'] = knn_predictions
    # test_df.to_csv(os.path.join(result_dir, 'knn_anomaly_detection_results.csv'), index=False)
    evaluate_model(test_df,predicted_label='cl_knn_predicted_label',file='cl_knn_evaluation_results.txt')


    # 对模糊样本进行检测
    # for i in uncertain_samples_index:
    #     uncertain_samples.append((i, test_df.iloc[i, test_df.columns.get_loc('content')], test_df.iloc[i, test_df.columns.get_loc('label')],
    #                         similar_samples, similarities[i], neighbor_labels))

    uncertain_samples_df = pd.DataFrame(uncertain_samples, columns=['index', 'content', 'label', 'similar_samples', 'similarities', 'neighbor_labels'])
    uncertain_samples_df['knn_label']= uncertain_knn_predictions

    # 模糊样本KNN的结果
    print("-------uncertain_samples KNN-------")
    evaluate_model(uncertain_samples_df,predicted_label='knn_label',file='uncertain_samples_knn.txt')

    # 对于模糊样本，使用大模型进行进一步处理
    if config.get('use_large_model', False):
        # 获得prompts
        cot_prompts = []
        our_prompts = []

        labels = []
        for i, content,label,similar_samples, similarities,neighbor_labels in uncertain_samples:
            top_k_logs = "<----start top_k_logs---->"
            k = 1
            for similar_sample,similarity,neighbor_label in zip(similar_samples, similarities,neighbor_labels):
                if neighbor_label==0:
                    system_state = "Normal"
                else:
                    system_state = "Abnormal"
                item = f" ### Top-{k} samples:\n####similarity:{similarity}\n "
                item += f"#### The system state of this log sequence :{system_state}\n"

                item += f"#### logs:\n{similar_sample}"
                k = k + 1

                top_k_logs += item
            top_k_logs += '<--- end top_k_logs ---> '
            labels.append(label)
            our_prompt_content = prompt_config["our_prompt"].format(logs=content,file_system=config['dataset_name'], top_k_logs=top_k_logs)
            cot_prompt_content =  prompt_config["cot_prompt"].format(logs=content,file_system=config['dataset_name'])
            our_prompt = [{"role":"system","content":prompt_config["system_prompt"]},
                            {"role": "user", "content":our_prompt_content} ]
            cot_prompt = [{"role":"system","content":prompt_config["system_prompt"]},
                            {"role": "user", "content":cot_prompt_content}]
            our_prompts.append(our_prompt)
            cot_prompts.append(cot_prompt)
            # print(our_prompt_content)
            # print(cot_prompt_content)
        
        start_time = time.perf_counter()       
        our_results,llm_usage,llm_answers= llm_utils.get_json_results(our_prompts,prompt_config["LLM"])
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"execution_time:{execution_time}s")
        # print(cot_results)
        # print(our_results)
        # print(labels)
        with open(os.path.join(result_dir, 'qa_logs.txt'),'w') as f:
           f.write(f"execution_time:{execution_time}s")
           for i in range(len(cot_prompts)):
                f.write(json.dumps(cot_prompts[i], ensure_ascii=False, indent=4))
                f.write(f"\n------Label:{labels[i]}--------\n")
                f.write(json.dumps(llm_answers[i], ensure_ascii=False, indent=4))
                f.write("\n--------------\n")

        llm_usage_file = os.path.join(result_dir, 'our_llm_usage.json')
        with open(llm_usage_file,'w') as f:
           json.dump(llm_usage,f)

        our_results["label"] = labels 
        our_results['knn_label'] = uncertain_samples_df['knn_label']
        evaluate_model(our_results,predicted_label='predicted_label',file='uncertain_our_evaluation_results.txt')
           
        cot_results,llm_usage,llm_answers = llm_utils.get_json_results(cot_prompts,prompt_config["LLM"])

        cot_results["label"] = labels 
        cot_results['knn_label'] = uncertain_samples_df['knn_label']

        uncertain_samples_df["label"] = labels
        cot_results.to_csv(os.path.join(result_dir, 'cot_results.csv'), index=False)
        our_results.to_csv(os.path.join(result_dir, 'our_results.csv'), index=False)

        
        evaluate_model(cot_results,predicted_label='predicted_label',file='uncertain_cot_evaluation_results.txt')
        



                    
            
        # 得到结果
        # 解析结果进行统计prompts


    # 保存测试结果

    # 将预测结果添加到 DataFrame 中
    
    test_df['cl_knn_predicted_label'] = knn_predictions
    # test_df['knn_llm_predicted_label'] = final_predictions

    # test_df.to_csv( os.path.join(result_dir, 'anomaly_detection_results.csv'), index=False)

    # 评估模型性能
    evaluate_model(test_df,predicted_label='cl_knn_predicted_label',file='cl_knn_evaluation_results.txt')
    # evaluate_model(test_df,predicted_label='knn_llm_predicted_label',file='cl_knn_llm_evaluation_results.txt')


    # 输出结果
    # print(test_df[['log_message', 'label', 'predicted_label']])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--config', type=str, default='./config/evlog.yaml', help="Path to the configuration file")
    parser.add_argument('--config', type=str, default='./config/spark3.yaml', help="Path to the configuration file")
    # parser.add_argument('--config', type=str, default='./config/collaborlog_hadoop3.yaml', help="Path to the configuration file")

    args = parser.parse_args()
    config = load_config(args.config)
    prompt_config = load_config('./config/prompt/prompt.yaml')

    wandb_project = "CollaborLog_7_8_test"

    print(json.dumps(config,indent=2))
    main(config)
