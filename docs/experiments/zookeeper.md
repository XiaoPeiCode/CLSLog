# Zookeeper 复现实验说明

本页记录 LogHub Zookeeper 数据集上的 CLSLog 复现流程与结果，便于对外展示。

## 实验配置一览

| 配置文件 | 模式 | 说明 |
|---------|------|------|
| `config/clslog_zookeeper.yaml` | SM-only | 调优后的小模型基线 |
| `config/clslog_zookeeper_routing.yaml` | SM + 路由分析 | 置信度划分，展示高/低置信子集 SM 表现 |
| `config/clslog_zookeeper_llm.yaml` | 完整 CLSLog | SM 高置信 + LLM 低置信 |

## 推荐运行顺序

```bash
# 1. 数据预处理（首次）
python3 demo/loghub_data_process_demo.py --dataset Zookeeper

# 2. SM-only 基线
python3 CLSLog.py --config ./config/clslog_zookeeper.yaml

# 3. 置信度路由分析（演化日志选择）
python3 CLSLog.py --config ./config/clslog_zookeeper_routing.yaml

# 4. 完整 CLSLog（需配置 LLM API，见 README）
cp config/llm_local.yaml.example config/llm_local.yaml
# 编辑 llm_local.yaml 填入 api_key 与 base_url
python3 CLSLog.py --config ./config/clslog_zookeeper_llm.yaml
```

## 关键结果（分层 8:2 划分，无 HDBSCAN）

| 阶段 | F1 | Precision | Recall |
|------|-----|-----------|--------|
| SM-only（全测试集） | 94.7% | 100% | 90% |
| SM 高置信度（1402 条，94.2%） | 100% | 100% | 100% |
| SM 低置信度（86 条，5.8%） | 0% | 0% | 0% |
| LLM 低置信度 | **100%** | **100%** | **100%** |
| **CLSLog 整体** | **100%** | **100%** | **100%** |

论文 Table 1 目标：Zookeeper SM F1 72.2%，CLSLog F1 99.3%。

## 低置信度子集说明

- 共 86 条样本：83 正常 + 3 异常
- SM 将全部 86 条判为 Normal，3 条异常全部漏检
- LLM 使用通用演化日志 Prompt（`prompt/clslog_evol_detect.yaml`）后全部判对
- 典型异常特征：日志中出现 unexpected internal exception 导致 forced shutdown

## 输出文件

运行后在各配置的 `result_dir` 下生成：

- `clslog_summary.json` — 汇总指标
- `routing_analysis.json` — 路由统计（启用路由时）
- `low_confidence_cases.json` — 低置信度样本导出
- `llm_low_confidence_records.json` — LLM 推理记录（启用 LLM 时）

详细指标见 [`docs/results/zookeeper_clslog_summary.json`](../results/zookeeper_clslog_summary.json)。
