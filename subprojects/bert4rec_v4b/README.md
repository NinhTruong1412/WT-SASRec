# BERT4Rec + WTCausalBERT4RecV4b

Isolated subproject for the two-model workflow around:

- `BERT4Rec`
- `WTCausalBERT4RecV4b`
- full parquet-to-RecBole data preparation
- tracked training runs and best-checkpoint summaries
- set-based intersection analysis/report generation

This subproject is intentionally narrower than the root repository. It keeps only the surfaces needed to reproduce the `BERT4Rec` vs `WTCausalBERT4RecV4b` pipeline without pulling in the whole experiment tree.

## Layout

```text
subprojects/bert4rec_v4b/
├── artifacts/best_models.json          # committed reference manifest for current best checkpoints
├── configs/                            # dataset + model configs
├── models/                             # minimal V4b dependency chain
├── tests/                              # subproject tests
├── compat.py                           # runtime compatibility patches
├── preprocess.py                       # parquet -> RecBole .inter
├── progress_trainer.py                 # Trainer subclass with JSONL progress tracking
├── train_models.py                     # focused BERT4Rec / V4b training runner
├── run_set_intersection_analysis.py    # generic set-based report generator
├── run_intersection_report.py          # BERT4Rec vs V4b prediction + report pipeline
└── training_progress.jsonl             # generated live batch progress log
```

## Quick start

```bash
cd subprojects/bert4rec_v4b
make setup
```

### 1. Prepare local full data

Place parquet shards under `raw_data/` inside this subproject, then run:

```bash
make preprocess
```

That writes:

- `data/thesis_dataset/thesis_dataset.inter`

### 2. Train the two models

```bash
make train
```

Outputs are written locally under:

- `training_progress.jsonl`
- `log/`
- `saved/`
- `results/`

The training runner also writes a JSON + Markdown summary with:

- per-seed metrics
- best validation score per run
- best checkpoint per model

### 3. Generate the set-intersection report

Use the committed reference manifest against the shared repository data:

```bash
make report
```

Or run directly with custom checkpoints:

```bash
python run_intersection_report.py \
  --data-path /path/to/data \
  --bert-checkpoint /path/to/BERT4Rec.pth \
  --v4b-checkpoint /path/to/WTCausalBERT4RecV4b.pth
```

This writes:

- a raw prediction file suitable for later reuse
- JSON analysis output
- the Markdown **Set-Based Intersection Analysis Report**

## Reusing shared repository assets

If you want a smoke run without creating local data first, the subproject can reuse the repository-level assets:

```bash
make smoke
```

That uses:

- shared data path: `../../data`
- shared mini dataset config: local `configs/mini_dataset.yaml`

## Reference best checkpoints

The committed reference manifest is `artifacts/best_models.json`.

It records the currently selected repository-level checkpoints:

- `saved/BERT4Rec-Apr-10-2026_18-20-11.pth`
- `saved/WTCausalBERT4RecV4b-Apr-12-2026_12-19-54.pth`

Those paths are stored as reproducibility metadata only. Raw checkpoint binaries are not committed by this subproject.
