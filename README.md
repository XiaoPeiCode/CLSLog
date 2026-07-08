# CLSLog

Log-based anomaly detection with small-model + LLM cascade (LogHub BGL / Zookeeper).

## Install

```bash
pip install -r requirements.txt
```

## Usage

Quick test (bundled BGL 2k sample):

```bash
python3 CLSLog.py --config ./config/clslog_bgl_demo.yaml
```

Download and preprocess LogHub data:

```bash
python3 demo/loghub_data_process_demo.py --dataset Zookeeper
python3 demo/loghub_data_process_demo.py --dataset BGL
```

Run on Zookeeper:

```bash
python3 CLSLog.py --config ./config/clslog_zookeeper.yaml
```

Enable LLM for low-confidence samples — set in `config/clslog_zookeeper.yaml`:

```yaml
sm_only_mode: false
use_large_model: true
```

Then configure API credentials:

```bash
cp config/llm_local.yaml.example config/llm_local.yaml
# set api_key and base_url
python3 CLSLog.py --config ./config/clslog_zookeeper.yaml
```

Or use environment variables:

```bash
export CLSLOG_LLM_API_KEY="your-key"
export CLSLOG_LLM_BASE_URL="https://api.openai.com/v1"
```

Optional utilities:

```bash
python3 demo/tune_zookeeper.py
python3 demo/rerun_llm_low_conf.py
```

Results are written to the `result_dir` path in each config file.
