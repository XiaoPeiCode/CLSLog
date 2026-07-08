# CLSLog

基于大小模型协同的日志异常检测复现代码库，对应论文 **CLSLog: Collaborating Large and Small Models for Log-based Anomaly Detection**（FSE Companion 2025）。

本仓库为**完全独立**的开源实现，不依赖 LogCAE、CoorLog 或其他外部项目。数据下载、Drain 日志解析、BERT 嵌入、Siamese 小模型、置信度路由与大模型推理均可在本仓库内完成。

当前重点展示 **LogHub Zookeeper** 上的完整复现链路：小模型基线 → 置信度路由（演化日志选择）→ 大模型语义补判。

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
├── dataset/example/BGL/               # 内置 BGL 2k 快速验证数据
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
│   ├── drain.py                       # Drain 日志解析（MIT, LogPAI）
│   ├── log_parsing.py                 # LogHub 下载与解析
│   ├── loghub_preprocessing.py        # BERT 嵌入、滑动窗口、划分
│   ├── cluster_utils.py               # HDBSCAN 降采样
│   └── util.py                        # 配置 / Prompt 工具
└── requirements.txt
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

首次运行会自动从 HuggingFace 下载 `bert-base-uncased` 到 `./cache_dir/`。支持 CUDA / Apple MPS / CPU 自动选择。

---

## 快速开始

### 1. 零下载 smoke test（内置 BGL 2k 样例，约 1 分钟）

```bash
python3 CLSLog.py --config ./config/clslog_bgl_demo.yaml
```

### 2. LogHub 全量数据预处理

自动从 Zenodo 下载并用内置 Drain 解析器生成 structured CSV：

```bash
python3 demo/loghub_data_process_demo.py --dataset Zookeeper
python3 demo/loghub_data_process_demo.py --dataset BGL
```

### 3. Zookeeper 完整复现

```bash
# SM-only 基线
python3 CLSLog.py --config ./config/clslog_zookeeper.yaml

# 置信度路由（演化日志选择）
python3 CLSLog.py --config ./config/clslog_zookeeper_routing.yaml

# 完整 CLSLog（需配置 LLM）
cp config/llm_local.yaml.example config/llm_local.yaml
python3 CLSLog.py --config ./config/clslog_zookeeper_llm.yaml
```

LLM 密钥通过 `config/llm_local.yaml`（gitignore）或环境变量 `CLSLOG_LLM_API_KEY` 注入。

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
原始日志 → Drain 解析 → BERT 嵌入 + 滑动窗口 → log sequence
    → Siamese 对比学习微调
    → 对 test 序列检索 top-k 邻居
    → 置信度 C(si) = mean(top-k similarities)
        ├── C(si) > μ  →  SM 直接判定
        └── C(si) ≤ μ  →  LLM 语义推理（注入 top-k 上下文 + SM 结果）
```

---

## 配置说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `split_method` | 划分方式 | `stratified` 或 `evolutionary` |
| `sm_only_mode` | 仅跑 SM | `true` / `false` |
| `enable_confidence_routing` | 置信度路由分析 | `true` |
| `use_hdbscan` | HDBSCAN 降采样 | Zookeeper 建议 `false` |
| `use_large_model` | 低置信度送 LLM | `false`（默认） |
| `llm_prompt` | Prompt 模板 | `clslog_evol_detect` |
| `structured_log_path` | 跳过下载，直接用已有 CSV | demo 配置已内置 |

---

## 第三方组件

- **Drain 日志解析器**：`utils/drain.py`，源自 [LogPAI/logparser](https://github.com/logpai/logparser)（MIT License）
- **LogHub 数据集**：https://github.com/logpai/loghub

---

## 参考

- LOGEVOL / CoorLog 等扩展工作不在本仓库维护范围内
