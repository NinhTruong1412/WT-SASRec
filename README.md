# Sequential Recommendation Baselines

Thesis project: **"Improving Sequential Recommendation by Incorporating Watch-Time Signals"**

Runs three standard sequential recommendation baselines using [RecBole](https://recbole.io/) on real streaming watch history data.

---

## Models

| Model | Architecture | Paper |
|-------|-------------|-------|
| **GRU4Rec** | RNN (GRU) | Hidasi et al., 2016 |
| **SASRec** | Self-Attention Transformer | Kang & McAuley, 2018 |
| **BERT4Rec** | Bidirectional Transformer (Cloze) | Sun et al., 2019 |

---

## Dataset

- **Source**: BigQuery export of user watch history
- **Raw**: 112,399 users, 857,422 interactions
- **Filtered** (≥5 interactions/user): ~23,171 users, ~722,538 interactions
- **Fields**: `user_id`, `item_id` (content), `timestamp`, `watch_time` (seconds)
- **Split**: Leave-one-out — last item → test, 2nd-last → val, rest → train

---

## Requirements

- Python 3.10+
- PyTorch 2.10+ (CPU or GPU — **MPS auto-detected on Apple Silicon**)
- RecBole 1.2.0

---

## Setup

```bash
# 1. Install dependencies + apply compatibility patches
make setup

# 2. Preprocess raw parquet → RecBole format
make preprocess
```

---

## Usage

```bash
# Quick verification (2000 users, 3 epochs — runs in ~2 min)
make sample

# Full training — all 3 models (50 epochs, early stopping)
make train

# Full training — individual models
make train-gru4rec
make train-sasrec
make train-bert4rec

# Clean up logs, checkpoints, results
make clean-logs

# Clean everything including preprocessed data
make clean-all
```

---

## Project Structure

```
Sequential Model/
├── Makefile                        # All commands
├── requirements.txt
├── preprocess.py                   # Parquet → RecBole .inter conversion
├── run_baseline.py                 # Training + evaluation runner
│
├── configs/
│   ├── dataset.yaml                # Full dataset config (train/val/test split, metrics)
│   ├── sample_dataset.yaml         # Sample config (2000 users, 3 epochs)
│   ├── gru4rec.yaml
│   ├── sasrec.yaml
│   └── bert4rec.yaml
│
├── data/
│   ├── thesis_dataset/             # Full preprocessed data (.inter)
│   └── thesis_sample/              # 2000-user sample for quick tests
│
├── log/                            # RecBole training logs (per model)
├── saved/                          # Best model checkpoints (.pth)
├── results/                        # Evaluation results (.json + .txt)
└── log_tensorboard/                # TensorBoard event files
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of test items retrieved in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rank-aware |
| **MRR@K** | Mean Reciprocal Rank |
| **Hit@K** | Binary hit rate at top-K |

K = 5, 10, 20 reported. `NDCG@10` used as validation metric.

---

## Sample Results (2000 users, 3 epochs)

| Metric | GRU4Rec | SASRec | BERT4Rec |
|--------|---------|--------|----------|
| Recall@10 | 0.3695 | **0.6975** | 0.3335 |
| NDCG@10 | 0.1886 | **0.5057** | 0.1967 |
| MRR@10 | 0.1317 | **0.4432** | 0.1529 |

> Note: These are 3-epoch results on a 2000-user sample. Full training will produce higher metrics.

---

## GPU Support

| Hardware | Backend | Config |
|----------|---------|--------|
| Apple Silicon (M1/M2/M3) | MPS | `use_gpu: true` (auto-detected) |
| NVIDIA GPU | CUDA | `use_gpu: true` (auto-detected) |
| CPU fallback | CPU | `use_gpu: false` |

---

## Next Steps (Thesis)

1. Run full training: `make train`
2. Use these baselines as comparison benchmarks
3. Implement watch-time weighted SASRec (Direction 1 from proposal)
4. Compare proposed model vs. baselines on HR@10, NDCG@10, MRR
