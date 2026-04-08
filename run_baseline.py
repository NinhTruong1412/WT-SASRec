"""
run_baseline.py
---------------
Runs sequential recommendation baselines and watch-time enhanced models
using RecBole and saves results to results/.

Usage:
    python3 run_baseline.py                  # all 3 baselines, full data
    python3 run_baseline.py --sample         # all 3 baselines, 2000-user sample
    python3 run_baseline.py --weighted       # all 3 WT models, full data
    python3 run_baseline.py --sample --weighted   # all 3 WT models, sample
    python3 run_baseline.py --model GRU4Rec  # single baseline model
    python3 run_baseline.py --model WTSASRec --weighted  # single WT model
"""

import argparse
import json
import os
import sys
from datetime import datetime
from logging import getLogger

# Allow importing custom models from ./models/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger, get_trainer, init_seed, set_color, get_flops, get_environment
)


BASE_CONFIG   = "configs/dataset.yaml"
SAMPLE_CONFIG = "configs/sample_dataset.yaml"
RESULTS_DIR   = "results"

BASELINE_MODELS = ["GRU4Rec", "SASRec", "BERT4Rec"]
WEIGHTED_MODELS = ["WTGru4Rec", "WTSASRec", "WTBert4Rec"]

BASELINE_CONFIGS = {
    "GRU4Rec":  "configs/gru4rec.yaml",
    "SASRec":   "configs/sasrec.yaml",
    "BERT4Rec": "configs/bert4rec.yaml",
}
WEIGHTED_CONFIGS = {
    "WTGru4Rec":  "configs/wt_gru4rec.yaml",
    "WTSASRec":   "configs/wt_sasrec.yaml",
    "WTBert4Rec": "configs/wt_bert4rec.yaml",
}

# Maps each WT model → parent RecBole model name (for Config to load correct defaults)
WT_BASE_MODEL = {
    "WTGru4Rec":  "GRU4Rec",
    "WTSASRec":   "SASRec",
    "WTBert4Rec": "BERT4Rec",
}

# Pairs for side-by-side comparison
MODEL_PAIRS = [("GRU4Rec", "WTGru4Rec"), ("SASRec", "WTSASRec"), ("BERT4Rec", "WTBert4Rec")]


def _load_wt_model_class(model_name: str):
    """Dynamically import a watch-time model class from models/."""
    from models import WTGru4Rec, WTSASRec, WTBert4Rec
    return {"WTGru4Rec": WTGru4Rec, "WTSASRec": WTSASRec, "WTBert4Rec": WTBert4Rec}[model_name]


def run_custom_model(model_class, config_file_list: list,
                     base_model_name: str = None, saved: bool = True) -> dict:
    """Run a custom (non-RecBole-registry) model class — mirrors run_recbole logic.

    base_model_name: RecBole parent model name (e.g. "BERT4Rec") used so that
    RecBole loads the correct default property YAML (which sets fields like
    MASK_ITEM_SEQ, POS_ITEMS etc.) before overriding with config_file_list.
    """
    config = Config(
        model=base_model_name or model_class.__name__,
        config_file_list=config_file_list,
    )
    # Override model name so logs/checkpoints use the WT model name
    config.final_config_dict["model"] = model_class.__name__
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    dataset    = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops     = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return {"test_result": test_result, "best_valid_score": best_valid_score,
            "best_valid_result": best_valid_result}


def run_model(model_name: str, base_config: str, model_configs: dict) -> dict:
    """Run a single RecBole model and return its test metrics."""
    print(f"\n{'='*60}")
    print(f"  Running {model_name}")
    print(f"{'='*60}\n")

    if model_name in WEIGHTED_CONFIGS:
        model_arg = _load_wt_model_class(model_name)
        result = run_custom_model(
            model_class=model_arg,
            config_file_list=[base_config, model_configs[model_name]],
            base_model_name=WT_BASE_MODEL.get(model_name),
        )
    else:
        result = run_recbole(
            model=model_name,
            config_file_list=[base_config, model_configs[model_name]],
        )
    return result.get("test_result", {})


def _fmt(val):
    return f"{float(val):.4f}" if val != "N/A" else "N/A"


def save_results(all_results: dict, label: str = "RESULTS"):
    """Save results to JSON + text and print a comparison table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(RESULTS_DIR, f"{label.lower()}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(
            {k: {m: float(v) for m, v in r.items() if isinstance(v, (int, float))}
             for k, r in all_results.items()},
            f, indent=2
        )
    print(f"\nResults saved to: {json_path}")

    metrics = ["recall@5", "recall@10", "recall@20",
               "hit@10", "ndcg@5", "ndcg@10", "ndcg@20",
               "mrr@10"]
    models  = list(all_results.keys())
    col_w   = 14
    sep     = "=" * (16 + col_w * len(models))

    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"{'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in models))
    print("-" * (16 + col_w * len(models)))
    for metric in metrics:
        row = f"{metric:<16}"
        for m in models:
            v = all_results.get(m, {}).get(metric, "N/A")
            row += f"{_fmt(v):>{col_w}}"
        print(row)
    print(sep)

    txt_path = os.path.join(RESULTS_DIR, f"{label.lower()}_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(f"{label}\n{sep}\n")
        f.write(f"{'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in models) + "\n")
        f.write("-" * (16 + col_w * len(models)) + "\n")
        for metric in metrics:
            row = f"{metric:<16}"
            for m in models:
                v = all_results.get(m, {}).get(metric, "N/A")
                row += f"{_fmt(v):>{col_w}}"
            f.write(row + "\n")
        f.write(sep + "\n")
    print(f"Text summary saved to: {txt_path}")


def save_comparison(baseline_results: dict, weighted_results: dict):
    """Print a side-by-side baseline vs watch-time comparison table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = ["recall@10", "ndcg@10", "mrr@10", "hit@10"]
    col_w   = 10

    headers = ["Metric"] + [f"{b}\n{w}" for b, w in MODEL_PAIRS]
    sep     = "=" * (16 + (col_w * 2 + 3) * 3)

    print(f"\n{sep}")
    print("  BASELINE vs WATCH-TIME COMPARISON")
    print(sep)
    hdr = f"{'Metric':<16}"
    for base, wt in MODEL_PAIRS:
        hdr += f"  {base:>{col_w}} {wt:>{col_w}}"
    print(hdr)
    print("-" * (16 + (col_w * 2 + 3) * 3))

    for metric in metrics:
        row = f"{metric:<16}"
        for base, wt in MODEL_PAIRS:
            bv = baseline_results.get(base, {}).get(metric, "N/A")
            wv = weighted_results.get(wt, {}).get(metric, "N/A")
            row += f"  {_fmt(bv):>{col_w}} {_fmt(wv):>{col_w}}"
        print(row)
    print(sep)

    txt_path = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("BASELINE vs WATCH-TIME COMPARISON\n")
        for metric in metrics:
            row = f"{metric:<16}"
            for base, wt in MODEL_PAIRS:
                bv = baseline_results.get(base, {}).get(metric, "N/A")
                wv = weighted_results.get(wt, {}).get(metric, "N/A")
                row += f"  {_fmt(bv):>{col_w}} {_fmt(wv):>{col_w}}"
            f.write(row + "\n")
    print(f"Comparison saved to: {txt_path}")


def main():
    all_baseline_choices = BASELINE_MODELS + ["all"]
    all_wt_choices       = WEIGHTED_MODELS + ["all"]

    parser = argparse.ArgumentParser(description="Run RecBole sequential recommendation models")
    parser.add_argument("--model", default="all",
                        help="Model name or 'all' (default: all)")
    parser.add_argument("--sample", action="store_true",
                        help="Use 2000-user sample (3 epochs) for quick verification")
    parser.add_argument("--weighted", action="store_true",
                        help="Run watch-time enhanced models instead of baselines")
    args = parser.parse_args()

    base_config  = SAMPLE_CONFIG if args.sample else BASE_CONFIG
    model_pool   = WEIGHTED_MODELS if args.weighted else BASELINE_MODELS
    model_cfgs   = WEIGHTED_CONFIGS if args.weighted else BASELINE_CONFIGS
    label        = "WATCH-TIME MODELS" if args.weighted else "BASELINE MODELS"

    if args.sample:
        tag = "⚡ SAMPLE MODE (2000 users, 3 epochs)"
        print(f"{tag} — {'Watch-Time' if args.weighted else 'Baseline'} models\n")

    models_to_run = model_pool if args.model == "all" else [args.model]
    all_results   = {}

    for model_name in models_to_run:
        try:
            metrics = run_model(model_name, base_config, model_cfgs)
            all_results[model_name] = metrics
            print(f"\n✓ {model_name} complete. Test metrics:")
            for k, v in sorted(metrics.items()):
                print(f"    {k}: {float(v):.4f}" if not isinstance(v, str) else f"    {k}: {v}")
        except Exception as exc:
            import traceback
            print(f"\n✗ {model_name} failed: {exc}")
            traceback.print_exc()
            all_results[model_name] = {"error": str(exc)}

    if len(models_to_run) > 1:
        save_results(all_results, label)


if __name__ == "__main__":
    main()

