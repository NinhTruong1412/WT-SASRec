.PHONY: help setup preprocess sample train train-gru4rec train-sasrec train-bert4rec \
        sample-weighted train-weighted \
        train-wt-gru4rec train-wt-sasrec train-wt-bert4rec \
        clean-logs clean-all

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON     := python3
DATA_FILE  := bigquery-export_user_item_interaction_2026_03_27_user_item_interaction-part-000000000000.parquet
INTER_FILE := data/thesis_dataset/thesis_dataset.inter

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Sequential Recommendation Baselines"
	@echo "  ====================================="
	@echo ""
	@echo "  Setup"
	@echo "    make setup          Install all dependencies"
	@echo "    make preprocess     Convert parquet → RecBole .inter format"
	@echo ""
	@echo "  Training — Baselines"
	@echo "    make sample         Quick verification, baselines (2000 users, 3 epochs)"
	@echo "    make train          Full training — all 3 baselines (GRU4Rec, SASRec, BERT4Rec)"
	@echo "    make train-gru4rec  Full training — GRU4Rec only"
	@echo "    make train-sasrec   Full training — SASRec only"
	@echo "    make train-bert4rec Full training — BERT4Rec only"
	@echo ""
	@echo "  Training — Watch-Time Models"
	@echo "    make sample-weighted    Quick verification, WT models (2000 users, 3 epochs)"
	@echo "    make train-weighted     Full training — all 3 WT models"
	@echo "    make train-wt-gru4rec   Full training — WTGru4Rec only"
	@echo "    make train-wt-sasrec    Full training — WTSASRec only"
	@echo "    make train-wt-bert4rec  Full training — WTBert4Rec only"
	@echo ""
	@echo "  Housekeeping"
	@echo "    make clean-logs     Remove logs, checkpoints, results"
	@echo "    make clean-all      Remove logs + preprocessed data"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "→ Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "→ Applying RecBole compatibility patches..."
	$(PYTHON) -c "\
import site, os; \
sp = site.getsitepackages()[0]; \
cfg = os.path.join(sp, 'recbole/config/configurator.py'); \
qs  = os.path.join(sp, 'recbole/quick_start/quick_start.py'); \
tr  = os.path.join(sp, 'recbole/trainer/trainer.py'); \
uts = os.path.join(sp, 'recbole/utils/utils.py'); \
\
# ── quick_start: make ray optional ──\
txt = open(qs).read(); \
old = 'from ray import tune'; \
new = 'try:\n    from ray import tune\nexcept ImportError:\n    tune = None'; \
open(qs, 'w').write(txt.replace(old, new)) if old in txt else None; \
\
# ── configurator: fix np legacy aliases ──\
txt = open(cfg).read(); \
old = '        np.bool = np.bool_\n        np.int = np.int_\n        np.float = np.float_\n        np.complex = np.complex_\n        np.object = np.object_\n        np.str = np.str_\n        np.long = np.int_\n        np.unicode = np.unicode_'; \
new = '        for alias, replacement in [(\"bool\",\"bool_\"),(\"int\",\"int_\"),(\"float\",\"float64\"),(\"complex\",\"complex128\"),(\"object\",\"object_\"),(\"str\",\"str_\"),(\"long\",\"int_\"),(\"unicode\",\"str_\")]:\n            if not hasattr(np, alias):\n                try: setattr(np, alias, getattr(np, replacement))\n                except AttributeError: pass'; \
open(cfg, 'w').write(txt.replace(old, new)) if old in txt else None; \
\
# ── configurator: add MPS device support ──\
txt = open(cfg).read(); \
old = '            self.final_config_dict[\"device\"] = (\n                torch.device(\"cpu\")\n                if len(gpu_id) == 0 or not torch.cuda.is_available()\n                else torch.device(\"cuda\")\n            )'; \
new = '            if len(gpu_id) > 0 and torch.cuda.is_available():\n                self.final_config_dict[\"device\"] = torch.device(\"cuda\")\n            elif len(gpu_id) > 0 and torch.backends.mps.is_available():\n                self.final_config_dict[\"device\"] = torch.device(\"mps\")\n            else:\n                self.final_config_dict[\"device\"] = torch.device(\"cpu\")'; \
open(cfg, 'w').write(txt.replace(old, new)) if old in txt else None; \
\
# ── trainer: fix torch.load weights_only ──\
txt = open(tr).read().replace('torch.load(resume_file, map_location=self.device)', 'torch.load(resume_file, map_location=self.device, weights_only=False)').replace('torch.load(checkpoint_file, map_location=self.device)', 'torch.load(checkpoint_file, map_location=self.device, weights_only=False)'); \
open(tr, 'w').write(txt); \
print('Patches applied OK')"
	@echo "✓ Setup complete."

# ── Data Preprocessing ────────────────────────────────────────────────────────
preprocess: $(INTER_FILE)

$(INTER_FILE): $(DATA_FILE)
	@echo "→ Preprocessing data..."
	$(PYTHON) preprocess.py
	@echo "✓ Data ready at $(INTER_FILE)"

# ── Training ─────────────────────────────────────────────────────────────────
sample: $(INTER_FILE)
	@echo "→ Running sample verification (2000 users, 3 epochs)..."
	$(PYTHON) run_baseline.py --sample

train: $(INTER_FILE)
	@echo "→ Running full training — all 3 models..."
	$(PYTHON) run_baseline.py

train-gru4rec: $(INTER_FILE)
	@echo "→ Training GRU4Rec (full data)..."
	$(PYTHON) run_baseline.py --model GRU4Rec

train-sasrec: $(INTER_FILE)
	@echo "→ Training SASRec (full data)..."
	$(PYTHON) run_baseline.py --model SASRec

train-bert4rec: $(INTER_FILE)
	@echo "→ Training BERT4Rec (full data)..."
	$(PYTHON) run_baseline.py --model BERT4Rec

# ── Watch-Time Models ─────────────────────────────────────────────────────────
sample-weighted: $(INTER_FILE)
	@echo "→ Running sample verification — Watch-Time models (2000 users, 3 epochs)..."
	$(PYTHON) run_baseline.py --sample --weighted

train-weighted: $(INTER_FILE)
	@echo "→ Running full training — all 3 Watch-Time models..."
	$(PYTHON) run_baseline.py --weighted

train-wt-gru4rec: $(INTER_FILE)
	@echo "→ Training WTGru4Rec (full data)..."
	$(PYTHON) run_baseline.py --model WTGru4Rec --weighted

train-wt-sasrec: $(INTER_FILE)
	@echo "→ Training WTSASRec (full data)..."
	$(PYTHON) run_baseline.py --model WTSASRec --weighted

train-wt-bert4rec: $(INTER_FILE)
	@echo "→ Training WTBert4Rec (full data)..."
	$(PYTHON) run_baseline.py --model WTBert4Rec --weighted

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean-logs:
	@echo "→ Removing logs, checkpoints, results..."
	rm -rf log/* log_tensorboard/* results/* saved/*
	@echo "✓ Cleaned."

clean-all: clean-logs
	@echo "→ Removing preprocessed data..."
	rm -rf data/thesis_dataset data/thesis_sample
	@echo "✓ All clean."
