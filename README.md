# CLSLog

基于大小模型协同的日志异常检测复现代码库，对应论文 **CLSLog: Collaborating Large and Small Models for Log-based Anomaly Detection**（FSE Companion 2025）。

本仓库当前重点展示 **LogHub Zookeeper** 上的完整复现链路：小模型基线 → 置信度路由（演化日志选择）→ 大模型语义补判。

| 方法 | 入口 | 数据集 | 说明 |
|------|------|--------|------|
| **CLSLog** | `CLSLog.py` | LogHub（BGL、Zookeeper） | 置信度级联：SM 高置信直接判定，低置信送 LLM |
| **CoorLog**（扩展） | `CollaborLog.py` | LOGEVOL（Spark、Hadoop） | AutoEncoder 协调器 + Evol-CoT + AEM |

---

## 项目结构

```
CLSLog/
├── CLSLog.py                          # 主入口
├── config/
│   ├── clslog_zookeeper.yaml          # Zookeeper SM-only 基线
│   ├── clslog_zookeeper_routing.yaml  # 置信度路由实验
│   ├── clslog_zookeeper_llm.yaml      # 完整 CLSLog（SM + LLM）
│   ├── clslog_bgl*.yaml               # BGL 相关配置
│   └── llm_local.yaml.example         # LLM API 配置模板
├── demo/
│   ├── loghub_data_process_demo.py    # LogHub 数据下载与预处理
│   ├── tune_zookeeper.py              # Zookeeper 超参搜索
│   └── rerun_llm_low_conf.py          # 仅重跑低置信度 LLM 推理
├── docs/
│   ├── experiments/zookeeper.md       # Zookeeper 复现实验说明
│   └── results/                       # 对外展示的汇总结果
├── modules/
│   ├── llm_chat.py                    # OpenAI 兼容 API 封装
│   └── llm_utils.py                   # LLM 批量调用与解析
├── prompt/
│   ├── clslog_detect.yaml             # 通用知识增强 Prompt
│   └── clslog_evol_detect.yaml        # 演化日志通用 Prompt（推荐）
├── utils/
│   ├── loghub_preprocessing.py        # 数据预处理、BERT 嵌入、划分
│   ├── cluster_utils.py               # HDBSCAN 降采样
│   └── util.py                        # 配置 / Prompt 工具
└── requirements.txt
```

**外部依赖：** LogHub 原始数据的 Drain 解析复用了同级目录 [LogCAE](../LogCAE) 中的 `utils/preprocessing.py`，请确保 `LogCAE` 与 `CLSLog` 位于同一父目录下。

---

## 环境配置

```bash
pip install -r requirements.txt
```

首次运行会自动从 HuggingFace 下载 `bert-base-uncased` 到 `./cache_dir/`。支持 CUDA / Apple MPS / CPU 自动选择。

---

## 快速开始（Zookeeper）

### 1. 数据预处理

```bash
python3 demo/loghub_data_process_demo.py --dataset Zookeeper
```

### 2. SM-only 基线

```bash
python3 CLSLog.py --config ./config/clslog_zookeeper.yaml
```

### 3. 置信度路由（演化日志选择）

将测试集按置信度 C(si) 分为高/低置信子集，分别统计 SM 表现：

```bash
python3 CLSLog.py --config ./config/clslog_zookeeper_routing.yaml
```

### 4. 完整 CLSLog（SM + LLM）

```bash
cp config/llm_local.yaml.example config/llm_local.yaml
# 编辑 llm_local.yaml，填入 api_key 与 base_url
python3 CLSLog.py --config ./config/clslog_zookeeper_llm.yaml
```

也可通过环境变量注入：

```bash
export CLSLOG_LLM_API_KEY="your-key"
export CLSLOG_LLM_BASE_URL="https://api.openai.com/v1"
```

---

## Zookeeper 复现结果

详见 [`docs/experiments/zookeeper.md`](docs/experiments/zookeeper.md)。

| 阶段 | F1 | Precision | Recall |
|------|-----|-----------|--------|
| SM-only | 94.7% | 100% | 90% |
| SM 高置信度（94.2% 样本） | 100% | 100% | 100% |
| SM 低置信度（5.8% 样本） | 0% | 0% | 0% |
| LLM 低置信度 | **100%** | **100%** | **100%** |
| **CLSLog 整体** | **100%** | **100%** | **100%** |

论文 Table 1 目标：Zookeeper SM F1 72.2%，CLSLog F1 99.3%。

---

## 核心流程

```
原始日志 → BERT 嵌入 + 滑动窗口 → log sequence
    → Siamese 对比学习微调
    → 对 test 序列检索 top-k 邻居
    → 置信度 C(si) = mean(top-k similarities)
        ├── C(si) > μ  →  SM 直接判定
        └── C(si) ≤ μ  →  LLM 语义推理（注入 top-k 上下文 + SM 结果）
```

| 步骤 | 文件 | 函数 |
|------|------|------|
| 数据预处理 | `utils/loghub_preprocessing.py` | `load_loghub_data()` |
| Siamese 训练 | `CLSLog.py` | `train_siamese_network()` |
| SM 推理 | `CLSLog.py` | `sm_predict()` |
| 置信度路由 | `CLSLog.py` | `analyze_confidence_routing()` |
| LLM 推理 | `CLSLog.py` + `prompt/clslog_evol_detect.yaml` | `llm_predict_low_confidence_samples()` |

---

## 配置说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `split_method` | 划分方式 | `stratified`（分层 8:2）或 `evolutionary`（时间演化） |
| `sm_only_mode` | 仅跑 SM，不调用 LLM | `true` / `false` |
| `enable_confidence_routing` | 启用置信度路由分析 | `true` |
| `use_hdbscan` | HDBSCAN 训练集降采样 | Zookeeper 复现建议 `false` |
| `tune_k_on_valid` | 在 valid 上搜索 k 与阈值 | `true` |
| `use_large_model` | 低置信度样本送 LLM | `false`（默认） |
| `llm_prompt` | Prompt 模板名 | `clslog_evol_detect` |
| `confidence_threshold` | 路由阈值 μ，`null` 为 valid 自动选取 | `null` |

LLM 相关密钥通过 `config/llm_local.yaml`（gitignore）或环境变量 `CLSLOG_LLM_API_KEY` 注入，**请勿提交到仓库**。

---

## 实验输出

每次运行在 `result_dir` 下生成：

```
results/clslog/<experiment>/
├── clslog_summary.json
├── routing_analysis.json              # 路由实验
├── sm_high_confidence_evaluation.txt
├── sm_low_confidence_evaluation.txt
├── low_confidence_cases.json
└── llm_low_confidence_records.json    # LLM 实验
```

---

## BGL 与其他数据集

```bash
# BGL 快速验证（2k 样例）
python3 CLSLog.py --config ./config/clslog_bgl_demo.yaml

# BGL 子集（5 万条）
python3 CLSLog.py --config ./config/clslog_bgl_sample.yaml
```

---

## 参考

- LogHub 数据集：https://github.com/logpai/loghub
- LOGEVOL 数据集：https://github.com/YintongHuo/EvLog
