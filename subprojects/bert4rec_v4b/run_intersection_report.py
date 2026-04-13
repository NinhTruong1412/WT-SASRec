#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from compat import apply_runtime_patches

apply_runtime_patches()

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.utils import init_seed

from models import WTCausalBERT4RecV4b
from run_set_intersection_analysis import AnalysisConfig, analyze_set_intersections, generate_markdown_report, save_outputs


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def topk_items_and_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_idx = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row_idx, idx], axis=1)
    topk_idx = idx[row_idx, order]
    topk_scores = scores[row_idx, topk_idx]
    return topk_idx, topk_scores


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_config(dataset_config: Path, model_config: Path, data_path: Path, seed: int, model_name: str) -> Config:
    config = Config(
        model="BERT4Rec",
        config_file_list=[str(dataset_config), str(model_config)],
        config_dict={
            "seed": seed,
            "data_path": str(data_path.resolve()),
            "show_progress": False,
            "log_wandb": False,
        },
    )
    config.final_config_dict["model"] = model_name
    return config


def load_model(model_cls, checkpoint_path: Path, config: Config, dataset):
    model = model_cls(config, dataset).to(config["device"])
    checkpoint = torch.load(str(checkpoint_path), map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    return model


def masked_full_sort_scores(model, interaction, history_index, total_items: int, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        scores = model.full_sort_predict(interaction.to(device))
    scores = scores.view(-1, total_items)
    scores[:, 0] = -np.inf
    if history_index is not None:
        scores[history_index] = -np.inf
    return scores.detach().cpu().numpy().astype(np.float32)


def collect_prediction_rows(
    bert_model,
    v4b_model,
    test_data,
    config: Config,
    topk_depth: int,
) -> list[dict]:
    rows: list[dict] = []
    uid_field = config["USER_ID_FIELD"]
    total_items = test_data._dataset.item_num
    device = config["device"]

    for batched_data in test_data:
        interaction, history_index, positive_u, positive_i = batched_data
        bert_scores = masked_full_sort_scores(
            bert_model, interaction, history_index, total_items, device
        )
        v4b_scores = masked_full_sort_scores(
            v4b_model, interaction, history_index, total_items, device
        )

        bert_items, bert_top_scores = topk_items_and_scores(bert_scores, topk_depth)
        v4b_items, v4b_top_scores = topk_items_and_scores(v4b_scores, topk_depth)

        user_ids = interaction[uid_field].cpu().tolist()
        gt_map: dict[int, list[int]] = defaultdict(list)
        for row_idx, item_id in zip(positive_u.cpu().tolist(), positive_i.cpu().tolist()):
            gt_map[int(row_idx)].append(int(item_id))

        for row_idx, user_id in enumerate(user_ids):
            rows.append(
                {
                    "user_id": int(user_id),
                    "ground_truth_items": gt_map.get(row_idx, []),
                    "model_A_top10_preds": bert_items[row_idx].astype(int).tolist(),
                    "model_A_scores": bert_top_scores[row_idx].astype(float).tolist(),
                    "model_B_top10_preds": v4b_items[row_idx].astype(int).tolist(),
                    "model_B_scores": v4b_top_scores[row_idx].astype(float).tolist(),
                }
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a BERT4Rec vs WTCausalBERT4RecV4b set-intersection report.")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-depth", type=int, default=20)
    parser.add_argument("--eval-ks", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--bert-checkpoint", default=None)
    parser.add_argument("--v4b-checkpoint", default=None)
    parser.add_argument(
        "--best-manifest",
        default="artifacts/best_models.json",
        help="Reference manifest used when explicit checkpoints are not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(resolve_project_path(args.best_manifest))

    bert_checkpoint = (
        resolve_project_path(args.bert_checkpoint)
        if args.bert_checkpoint
        else resolve_repo_path(manifest["models"]["BERT4Rec"]["checkpoint_repo_path"])
    )
    v4b_checkpoint = (
        resolve_project_path(args.v4b_checkpoint)
        if args.v4b_checkpoint
        else resolve_repo_path(manifest["models"]["WTCausalBERT4RecV4b"]["checkpoint_repo_path"])
    )

    dataset_config = resolve_project_path(args.dataset_config)
    data_path = resolve_project_path(args.data_path)

    bert_config = build_config(
        dataset_config=dataset_config,
        model_config=PROJECT_ROOT / "configs" / "bert4rec.yaml",
        data_path=data_path,
        seed=args.seed,
        model_name="BERT4Rec",
    )
    init_seed(bert_config["seed"], bert_config["reproducibility"])
    dataset = create_dataset(bert_config)
    _, _, test_data = data_preparation(bert_config, dataset)

    v4b_config = build_config(
        dataset_config=dataset_config,
        model_config=PROJECT_ROOT / "configs" / "wt_causal_bert4rec_v4b.yaml",
        data_path=data_path,
        seed=args.seed,
        model_name="WTCausalBERT4RecV4b",
    )

    bert_model = load_model(BERT4Rec, bert_checkpoint, bert_config, dataset)
    v4b_model = load_model(WTCausalBERT4RecV4b, v4b_checkpoint, v4b_config, dataset)

    rows = collect_prediction_rows(
        bert_model=bert_model,
        v4b_model=v4b_model,
        test_data=test_data,
        config=bert_config,
        topk_depth=args.topk_depth,
    )

    results = analyze_set_intersections(
        pd.DataFrame(rows),
        config=AnalysisConfig(eval_ks=tuple(args.eval_ks), ranking_methods=("avg_score", "borda")),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / f"bert4rec_v4b_predictions_{timestamp}.jsonl"
    pd.DataFrame(rows).to_json(predictions_path, orient="records", lines=True)

    report_body = generate_markdown_report(results)
    preface = [
        "# BERT4Rec vs WTCausalBERT4RecV4b",
        "",
        f"- Dataset config: `{dataset_config}`",
        f"- Data path: `{data_path}`",
        f"- BERT4Rec checkpoint: `{bert_checkpoint}`",
        f"- WTCausalBERT4RecV4b checkpoint: `{v4b_checkpoint}`",
        f"- Raw predictions: `{predictions_path}`",
        "",
    ]
    report = "\n".join(preface) + report_body
    output_prefix = results_dir / f"bert4rec_v4b_set_intersection_{timestamp}"
    json_path, md_path = save_outputs(results, report, output_prefix=output_prefix)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
