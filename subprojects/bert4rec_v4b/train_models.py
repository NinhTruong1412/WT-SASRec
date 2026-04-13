#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from logging import FileHandler, getLogger
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from compat import apply_runtime_patches

apply_runtime_patches()

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.utils import init_logger, init_seed

from models import WTCausalBERT4RecV4b
from progress_trainer import TrackedTrainer


MODEL_REGISTRY = {
    "BERT4Rec": BERT4Rec,
    "WTCausalBERT4RecV4b": WTCausalBERT4RecV4b,
}

MODEL_CONFIGS = {
    "BERT4Rec": PROJECT_ROOT / "configs" / "bert4rec.yaml",
    "WTCausalBERT4RecV4b": PROJECT_ROOT / "configs" / "wt_causal_bert4rec_v4b.yaml",
}

MODEL_BASE_NAMES = {
    "BERT4Rec": "BERT4Rec",
    "WTCausalBERT4RecV4b": "BERT4Rec",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_project_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def reset_logger_handlers() -> None:
    logger = getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def extract_log_files() -> list[str]:
    logger = getLogger()
    log_files: list[str] = []
    for handler in logger.handlers:
        if isinstance(handler, FileHandler):
            log_files.append(to_project_relative(handler.baseFilename))
    return sorted(set(log_files))


def build_config(model_name: str, dataset_config: Path, data_path: Path, seed: int, args: argparse.Namespace) -> Config:
    overrides = {
        "seed": seed,
        "data_path": str(data_path.resolve()),
        "checkpoint_dir": str((PROJECT_ROOT / "saved").resolve()),
        "training_progress_file": str((PROJECT_ROOT / "training_progress.jsonl").resolve()),
        "progress_log_interval": args.progress_log_interval,
        "show_progress": args.show_progress,
        "log_wandb": False,
        "save_dataset": False,
        "save_dataloaders": False,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.stopping_step is not None:
        overrides["stopping_step"] = args.stopping_step
    if args.eval_step is not None:
        overrides["eval_step"] = args.eval_step
    if args.train_batch_size is not None:
        overrides["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        overrides["eval_batch_size"] = args.eval_batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    config = Config(
        model=MODEL_BASE_NAMES[model_name],
        config_file_list=[str(dataset_config), str(MODEL_CONFIGS[model_name])],
        config_dict=overrides,
    )
    config.final_config_dict["model"] = model_name
    return config


def run_one(model_name: str, dataset_config: Path, data_path: Path, seed: int, args: argparse.Namespace) -> dict:
    reset_logger_handlers()

    config = build_config(model_name=model_name, dataset_config=dataset_config, data_path=data_path, seed=seed, args=args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    log_files = extract_log_files()

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = MODEL_REGISTRY[model_name](config, dataset).to(config["device"])
    trainer = TrackedTrainer(config, model)

    started_at = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=True,
        saved=True,
        show_progress=args.show_progress,
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=False,
    )
    elapsed = time.time() - started_at

    return {
        "model": model_name,
        "seed": seed,
        "dataset": config["dataset"],
        "dataset_config": to_project_relative(dataset_config),
        "data_path": str(data_path.resolve()),
        "checkpoint": to_project_relative(trainer.saved_model_file),
        "log_files": log_files,
        "training_progress_file": "training_progress.jsonl",
        "best_valid_score": round(float(best_valid_score), 6),
        "best_valid_result": {k: float(v) for k, v in (best_valid_result or {}).items()},
        "test_result": {k: float(v) for k, v in test_result.items()},
        "train_time_s": round(elapsed, 1),
    }


def select_best_runs(runs: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for run in runs:
        current = best.get(run["model"])
        if current is None or run["best_valid_score"] > current["best_valid_score"]:
            best[run["model"]] = run
    return best


def build_markdown_report(runs: list[dict], best_runs: dict[str, dict]) -> str:
    lines = []
    lines.append("# BERT4Rec vs WTCausalBERT4RecV4b Training Summary")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}_")
    lines.append("")
    lines.append("## Per-run metrics")
    lines.append("")
    lines.append("| Model | Seed | Best valid NDCG@10 | Test NDCG@10 | Recall@10 | MRR@10 | Checkpoint |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in runs:
        metrics = run["test_result"]
        lines.append(
            f"| {run['model']} | {run['seed']} | {run['best_valid_score']:.6f} | "
            f"{metrics.get('ndcg@10', 0.0):.6f} | {metrics.get('recall@10', 0.0):.6f} | "
            f"{metrics.get('mrr@10', 0.0):.6f} | `{run['checkpoint']}` |"
        )
    lines.append("")
    lines.append("## Best checkpoint per model")
    lines.append("")
    lines.append("| Model | Seed | Best valid NDCG@10 | Checkpoint |")
    lines.append("| --- | ---: | ---: | --- |")
    for model_name in sorted(best_runs):
        run = best_runs[model_name]
        lines.append(
            f"| {model_name} | {run['seed']} | {run['best_valid_score']:.6f} | `{run['checkpoint']}` |"
        )
    return "\n".join(lines)


def save_outputs(runs: list[dict], dataset_config: Path) -> tuple[Path, Path]:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_runs = select_best_runs(runs)

    payload = {
        "generated_at": timestamp,
        "dataset_config": to_project_relative(dataset_config),
        "runs": runs,
        "best_by_model": best_runs,
    }

    json_path = results_dir / f"train_summary_{timestamp}.json"
    md_path = results_dir / f"train_summary_{timestamp}.md"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    with md_path.open("w") as f:
        f.write(build_markdown_report(runs, best_runs))
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT4Rec and WTCausalBERT4RecV4b in the extracted subproject.")
    parser.add_argument(
        "--model",
        default="both",
        choices=["both", "BERT4Rec", "WTCausalBERT4RecV4b"],
        help="Model selection.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Dataset config path relative to the subproject root unless absolute.",
    )
    parser.add_argument(
        "--data-path",
        default="data",
        help="Directory containing RecBole dataset folders.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stopping-step", type=int, default=None)
    parser.add_argument("--eval-step", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=10)
    parser.add_argument("--show-progress", type=parse_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = resolve_path(args.dataset_config)
    data_path = resolve_path(args.data_path)
    models = ["BERT4Rec", "WTCausalBERT4RecV4b"] if args.model == "both" else [args.model]

    runs = []
    for model_name in models:
        for seed in args.seeds:
            print(f"\n=== Training {model_name} | seed={seed} ===")
            runs.append(run_one(model_name=model_name, dataset_config=dataset_config, data_path=data_path, seed=seed, args=args))

    json_path, md_path = save_outputs(runs, dataset_config)
    print(f"\nSaved JSON summary: {json_path}")
    print(f"Saved Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
