.PHONY: help setup preprocess train train-bert4rec train-v4b smoke report clean

PYTHON ?= python3

help:
	@echo ""
	@echo "  BERT4Rec + WTCausalBERT4RecV4b subproject"
	@echo "  =========================================="
	@echo ""
	@echo "  Setup"
	@echo "    make setup           Install subproject dependencies"
	@echo ""
	@echo "  Data"
	@echo "    make preprocess      Build local data/thesis_dataset/thesis_dataset.inter"
	@echo ""
	@echo "  Training"
	@echo "    make train           Train BERT4Rec and WTCausalBERT4RecV4b"
	@echo "    make train-bert4rec  Train only BERT4Rec"
	@echo "    make train-v4b       Train only WTCausalBERT4RecV4b"
	@echo "    make smoke           Quick 1-epoch smoke run on shared mini_dataset"
	@echo ""
	@echo "  Analysis"
	@echo "    make report          Build the set-intersection report from the committed best-model manifest"
	@echo ""
	@echo "  Housekeeping"
	@echo "    make clean           Remove local outputs"
	@echo ""

setup:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) preprocess.py

train:
	$(PYTHON) train_models.py --model both

train-bert4rec:
	$(PYTHON) train_models.py --model BERT4Rec

train-v4b:
	$(PYTHON) train_models.py --model WTCausalBERT4RecV4b

smoke:
	$(PYTHON) train_models.py --model both --dataset-config configs/mini_dataset.yaml --data-path ../../data --epochs 1 --stopping-step 1 --seeds 42 --show-progress false

report:
	$(PYTHON) run_intersection_report.py --data-path ../../data

clean:
	rm -rf data raw_data log log_tensorboard results saved training_progress.jsonl
