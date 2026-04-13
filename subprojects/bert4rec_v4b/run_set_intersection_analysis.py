#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - optional dependency
    binomtest = None


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

METHOD_LABELS = {
    "avg_score": "Average normalized scores",
    "borda": "Borda / average-rank fusion",
}

CONFIG_LABELS = {
    "consensus": "Strict Consensus (X)",
    "baseline": "Baseline (X∪Y = A)",
    "proposed": "Proposed (X∪Z = B)",
    "union": "Union / Ensemble (X∪Y∪Z)",
}


@dataclass(frozen=True)
class ColumnConfig:
    user_id: str = "user_id"
    ground_truth: str = "ground_truth_items"
    model_a_preds: str = "model_A_top10_preds"
    model_a_scores: str = "model_A_scores"
    model_b_preds: str = "model_B_top10_preds"
    model_b_scores: str = "model_B_scores"


@dataclass(frozen=True)
class AnalysisConfig:
    eval_ks: tuple[int, ...] = (10, 20)
    ranking_methods: tuple[str, ...] = ("avg_score", "borda")
    missing_score: float = 0.0


@dataclass
class NormalizedRow:
    user_id: Hashable
    ground_truth: tuple[Hashable, ...]
    model_a_items: np.ndarray
    model_a_scores: np.ndarray
    model_b_items: np.ndarray
    model_b_scores: np.ndarray


@dataclass
class MetricAccumulator:
    hit_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    ndcg_sum: float = 0.0
    count: int = 0

    def update(self, ranked_items: Sequence[Hashable], gt_items: set[Hashable], k: int) -> None:
        ranked_k = list(ranked_items[:k])
        rels = [1.0 if item in gt_items else 0.0 for item in ranked_k]
        hits = int(sum(rels))

        self.count += 1
        self.hit_sum += 1.0 if hits > 0 else 0.0
        self.precision_sum += hits / float(k)
        self.recall_sum += hits / float(max(len(gt_items), 1))

        dcg = 0.0
        for idx, rel in enumerate(rels):
            if rel:
                dcg += 1.0 / math.log2(idx + 2)
        ideal_hits = min(len(gt_items), k)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
        self.ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    def result(self) -> dict[str, float]:
        denom = float(max(self.count, 1))
        return {
            "precision": round(self.precision_sum / denom, 6),
            "recall": round(self.recall_sum / denom, 6),
            "hit_rate": round(self.hit_sum / denom, 6),
            "ndcg": round(self.ndcg_sum / denom, 6),
        }


@dataclass
class SizeAccumulator:
    sizes: list[int] = field(default_factory=list)

    def update(self, size: int) -> None:
        self.sizes.append(int(size))

    def result(self) -> dict[str, float]:
        if not self.sizes:
            return {"avg_size": 0.0, "pct_non_empty": 0.0}
        arr = np.asarray(self.sizes, dtype=np.float64)
        return {
            "avg_size": round(float(arr.mean()), 4),
            "pct_non_empty": round(float((arr > 0).mean() * 100.0), 2),
        }


@dataclass
class DisagreementAccumulator:
    y_sizes: list[int] = field(default_factory=list)
    z_sizes: list[int] = field(default_factory=list)
    y_hits: int = 0
    z_hits: int = 0
    y_only_hits: int = 0
    z_only_hits: int = 0
    both_hit: int = 0
    neither_hit: int = 0
    count: int = 0

    def update(self, y_items: Sequence[Hashable], z_items: Sequence[Hashable], gt_items: set[Hashable], k: int) -> None:
        y_ranked = list(y_items[:k])
        z_ranked = list(z_items[:k])
        y_hit = any(item in gt_items for item in y_ranked)
        z_hit = any(item in gt_items for item in z_ranked)

        self.count += 1
        self.y_sizes.append(len(y_ranked))
        self.z_sizes.append(len(z_ranked))
        self.y_hits += int(y_hit)
        self.z_hits += int(z_hit)

        if y_hit and not z_hit:
            self.y_only_hits += 1
        elif z_hit and not y_hit:
            self.z_only_hits += 1
        elif y_hit and z_hit:
            self.both_hit += 1
        else:
            self.neither_hit += 1

    def result(self) -> dict[str, float | int | str]:
        denom = float(max(self.count, 1))
        discordant = self.y_only_hits + self.z_only_hits
        p_value = paired_binomial_pvalue_greater(self.z_only_hits, discordant)

        if self.z_hits > self.y_hits:
            winner = "Z (Model B unique items)"
        elif self.y_hits > self.z_hits:
            winner = "Y (Model A unique items)"
        else:
            winner = "Tie"

        return {
            "avg_size_y": round(float(np.mean(self.y_sizes)) if self.y_sizes else 0.0, 4),
            "avg_size_z": round(float(np.mean(self.z_sizes)) if self.z_sizes else 0.0, 4),
            "hit_rate_y": round(self.y_hits / denom, 6),
            "hit_rate_z": round(self.z_hits / denom, 6),
            "delta_hit_rate_z_minus_y": round((self.z_hits - self.y_hits) / denom, 6),
            "y_only_hits": self.y_only_hits,
            "z_only_hits": self.z_only_hits,
            "both_hit": self.both_hit,
            "neither_hit": self.neither_hit,
            "discordant_pairs": discordant,
            "p_value_z_greater_y": round(p_value, 12),
            "winner": winner,
        }


def paired_binomial_pvalue_greater(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(successes, trials, 0.5, alternative="greater").pvalue)

    mean = trials * 0.5
    std = math.sqrt(trials * 0.25)
    if std == 0:
        return 1.0
    z = (successes - mean) / std
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return pd.DataFrame(data["records"])
        if "data" in data and isinstance(data["data"], list):
            return pd.DataFrame(data["data"])
        return pd.DataFrame([data])
    raise TypeError("Input must be a pandas DataFrame, list of dicts, or dict.")


def load_input_dataframe(path: Path, input_format: str = "auto") -> pd.DataFrame:
    fmt = input_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix == ".parquet":
            fmt = "parquet"
        elif suffix in {".pkl", ".pickle"}:
            fmt = "pickle"
        else:
            raise ValueError(f"Cannot infer input format from suffix: {path.suffix}")

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "pickle":
        return pd.read_pickle(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    if fmt == "json":
        with path.open() as f:
            payload = json.load(f)
        return ensure_dataframe(payload)
    raise ValueError(f"Unsupported input format: {input_format}")


def to_python_scalar(value: Any) -> Hashable:
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set, pd.Series)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
                    return list(parsed)
                return [parsed]
            except Exception:
                continue
        if "," in stripped:
            return [token.strip() for token in stripped.split(",") if token.strip()]
        return [stripped]
    if pd.isna(value):
        return []
    return [value]


def coerce_items(value: Any, field_name: str) -> list[Hashable]:
    items = [to_python_scalar(item) for item in parse_sequence(value)]
    if not items:
        raise ValueError(f"{field_name} must contain at least one item.")
    return items


def coerce_scores(value: Any, field_name: str) -> np.ndarray:
    raw_scores = parse_sequence(value)
    if not raw_scores:
        raise ValueError(f"{field_name} must contain at least one score.")
    try:
        return np.asarray([float(score) for score in raw_scores], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def ensure_unique_predictions(items: list[Hashable], field_name: str) -> None:
    seen: set[Hashable] = set()
    duplicates: list[Hashable] = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate recommendation items: {duplicates[:5]}")


def normalize_score_vector(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max > score_min:
        return (scores - score_min) / (score_max - score_min)
    if len(scores) == 1:
        return np.array([1.0], dtype=np.float64)
    return np.linspace(1.0, 0.0, num=len(scores), dtype=np.float64)


def normalize_input_rows(df: pd.DataFrame, columns: ColumnConfig, required_depth: int) -> list[NormalizedRow]:
    required = [
        columns.user_id,
        columns.ground_truth,
        columns.model_a_preds,
        columns.model_a_scores,
        columns.model_b_preds,
        columns.model_b_scores,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    subset = df[required].rename(
        columns={
            columns.user_id: "user_id",
            columns.ground_truth: "ground_truth",
            columns.model_a_preds: "model_a_preds",
            columns.model_a_scores: "model_a_scores",
            columns.model_b_preds: "model_b_preds",
            columns.model_b_scores: "model_b_scores",
        }
    )

    rows: list[NormalizedRow] = []
    for row in subset.itertuples(index=False, name="PredictionRow"):
        gt_items = tuple(coerce_items(row.ground_truth, "ground_truth_items"))
        model_a_items = coerce_items(row.model_a_preds, "model_A predictions")
        model_a_scores = coerce_scores(row.model_a_scores, "model_A scores")
        model_b_items = coerce_items(row.model_b_preds, "model_B predictions")
        model_b_scores = coerce_scores(row.model_b_scores, "model_B scores")

        if len(model_a_items) != len(model_a_scores):
            raise ValueError("model_A predictions and scores must have the same length.")
        if len(model_b_items) != len(model_b_scores):
            raise ValueError("model_B predictions and scores must have the same length.")
        if len(model_a_items) < required_depth:
            raise ValueError(
                f"model_A predictions must have at least {required_depth} items for exact evaluation."
            )
        if len(model_b_items) < required_depth:
            raise ValueError(
                f"model_B predictions must have at least {required_depth} items for exact evaluation."
            )

        ensure_unique_predictions(model_a_items[:required_depth], "model_A predictions")
        ensure_unique_predictions(model_b_items[:required_depth], "model_B predictions")

        rows.append(
            NormalizedRow(
                user_id=to_python_scalar(row.user_id),
                ground_truth=gt_items,
                model_a_items=np.asarray(model_a_items[:required_depth], dtype=object),
                model_a_scores=np.asarray(model_a_scores[:required_depth], dtype=np.float64),
                model_b_items=np.asarray(model_b_items[:required_depth], dtype=object),
                model_b_scores=np.asarray(model_b_scores[:required_depth], dtype=np.float64),
            )
        )
    return rows


def build_rank_map(items: Sequence[Hashable]) -> dict[Hashable, int]:
    return {item: idx + 1 for idx, item in enumerate(items)}


def build_score_map(items: Sequence[Hashable], scores: np.ndarray) -> dict[Hashable, float]:
    normalized_scores = normalize_score_vector(scores)
    return {item: float(score) for item, score in zip(items, normalized_scores)}


def rank_items_by_avg_score(
    items: Sequence[Hashable],
    score_map_a: dict[Hashable, float],
    score_map_b: dict[Hashable, float],
    rank_map_a: dict[Hashable, int],
    rank_map_b: dict[Hashable, int],
    max_rank: int,
    missing_score: float,
) -> list[Hashable]:
    def sort_key(item: Hashable) -> tuple[float, int, float, str]:
        score_a = score_map_a.get(item, missing_score)
        score_b = score_map_b.get(item, missing_score)
        combined_score = score_a + score_b
        present_count = int(item in score_map_a) + int(item in score_map_b)
        avg_rank = (rank_map_a.get(item, max_rank) + rank_map_b.get(item, max_rank)) / 2.0
        return (-combined_score, -present_count, avg_rank, str(item))

    return sorted(items, key=sort_key)


def rank_items_by_borda(
    items: Sequence[Hashable],
    score_map_a: dict[Hashable, float],
    score_map_b: dict[Hashable, float],
    rank_map_a: dict[Hashable, int],
    rank_map_b: dict[Hashable, int],
    max_rank: int,
) -> list[Hashable]:
    def sort_key(item: Hashable) -> tuple[float, float, int, str]:
        rank_sum = rank_map_a.get(item, max_rank) + rank_map_b.get(item, max_rank)
        combined_score = score_map_a.get(item, 0.0) + score_map_b.get(item, 0.0)
        present_count = int(item in rank_map_a) + int(item in rank_map_b)
        return (rank_sum, -combined_score, -present_count, str(item))

    return sorted(items, key=sort_key)


def rerank_items(
    items: Sequence[Hashable],
    method: str,
    score_map_a: dict[Hashable, float],
    score_map_b: dict[Hashable, float],
    rank_map_a: dict[Hashable, int],
    rank_map_b: dict[Hashable, int],
    k: int,
    missing_score: float,
) -> list[Hashable]:
    if method == "avg_score":
        return rank_items_by_avg_score(
            items=items,
            score_map_a=score_map_a,
            score_map_b=score_map_b,
            rank_map_a=rank_map_a,
            rank_map_b=rank_map_b,
            max_rank=k + 1,
            missing_score=missing_score,
        )
    if method == "borda":
        return rank_items_by_borda(
            items=items,
            score_map_a=score_map_a,
            score_map_b=score_map_b,
            rank_map_a=rank_map_a,
            rank_map_b=rank_map_b,
            max_rank=k + 1,
        )
    raise ValueError(f"Unsupported ranking method: {method}")


def init_metric_accumulators(ranking_methods: Sequence[str], eval_ks: Sequence[int]) -> dict[str, dict[int, dict[str, MetricAccumulator]]]:
    return {
        method: {
            k: {
                "consensus": MetricAccumulator(),
                "baseline": MetricAccumulator(),
                "proposed": MetricAccumulator(),
                "union": MetricAccumulator(),
            }
            for k in eval_ks
        }
        for method in ranking_methods
    }


def analyze_set_intersections(
    data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    columns: ColumnConfig | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    columns = columns or ColumnConfig()
    config = config or AnalysisConfig()

    eval_ks = tuple(sorted(set(config.eval_ks)))
    if not eval_ks:
        raise ValueError("At least one evaluation cutoff is required.")
    if min(eval_ks) <= 0:
        raise ValueError("Evaluation cutoffs must be positive integers.")

    df = ensure_dataframe(data)
    rows = normalize_input_rows(df, columns=columns, required_depth=max(eval_ks))

    metric_accs = init_metric_accumulators(config.ranking_methods, eval_ks)
    set_size_accs = {
        k: {"X": SizeAccumulator(), "Y": SizeAccumulator(), "Z": SizeAccumulator()}
        for k in eval_ks
    }
    disagreement_accs = {k: DisagreementAccumulator() for k in eval_ks}

    for row in rows:
        gt_items = set(row.ground_truth)
        for k in eval_ks:
            a_items = [to_python_scalar(item) for item in row.model_a_items[:k].tolist()]
            b_items = [to_python_scalar(item) for item in row.model_b_items[:k].tolist()]
            a_scores = row.model_a_scores[:k]
            b_scores = row.model_b_scores[:k]

            set_a = set(a_items)
            set_b = set(b_items)

            x_items = [item for item in a_items if item in set_b]
            y_items = [item for item in a_items if item not in set_b]
            z_items = [item for item in b_items if item not in set_a]
            union_items = list(dict.fromkeys(a_items + b_items))

            set_size_accs[k]["X"].update(len(x_items))
            set_size_accs[k]["Y"].update(len(y_items))
            set_size_accs[k]["Z"].update(len(z_items))
            disagreement_accs[k].update(y_items=y_items, z_items=z_items, gt_items=gt_items, k=k)

            rank_map_a = build_rank_map(a_items)
            rank_map_b = build_rank_map(b_items)
            score_map_a = build_score_map(a_items, a_scores)
            score_map_b = build_score_map(b_items, b_scores)

            for method in config.ranking_methods:
                consensus_ranked = rerank_items(
                    items=x_items,
                    method=method,
                    score_map_a=score_map_a,
                    score_map_b=score_map_b,
                    rank_map_a=rank_map_a,
                    rank_map_b=rank_map_b,
                    k=k,
                    missing_score=config.missing_score,
                )
                union_ranked = rerank_items(
                    items=union_items,
                    method=method,
                    score_map_a=score_map_a,
                    score_map_b=score_map_b,
                    rank_map_a=rank_map_a,
                    rank_map_b=rank_map_b,
                    k=k,
                    missing_score=config.missing_score,
                )

                metric_accs[method][k]["consensus"].update(consensus_ranked, gt_items, k)
                metric_accs[method][k]["baseline"].update(a_items, gt_items, k)
                metric_accs[method][k]["proposed"].update(b_items, gt_items, k)
                metric_accs[method][k]["union"].update(union_ranked, gt_items, k)

    metrics = {
        method: {
            str(k): {
                config_name: acc.result()
                for config_name, acc in metric_accs[method][k].items()
            }
            for k in eval_ks
        }
        for method in config.ranking_methods
    }
    set_sizes = {
        str(k): {set_name: acc.result() for set_name, acc in set_size_accs[k].items()}
        for k in eval_ks
    }
    disagreements = {str(k): disagreement_accs[k].result() for k in eval_ks}

    return {
        "metadata": {
            "n_users": len(rows),
            "eval_ks": list(eval_ks),
            "ranking_methods": list(config.ranking_methods),
            "columns": {
                "user_id": columns.user_id,
                "ground_truth": columns.ground_truth,
                "model_a_preds": columns.model_a_preds,
                "model_a_scores": columns.model_a_scores,
                "model_b_preds": columns.model_b_preds,
                "model_b_scores": columns.model_b_scores,
            },
        },
        "set_size_stats": set_sizes,
        "metrics": metrics,
        "pure_disagreements": disagreements,
    }


def generate_markdown_report(results: dict[str, Any]) -> str:
    eval_ks = results["metadata"]["eval_ks"]
    ranking_methods = results["metadata"]["ranking_methods"]

    lines: list[str] = []
    lines.append("# Set-Based Intersection Analysis Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}_")
    lines.append(f"_Users evaluated: {results['metadata']['n_users']:,}_")
    lines.append("")
    lines.append("## Set Definitions")
    lines.append("")
    lines.append("| Symbol | Meaning |")
    lines.append("| --- | --- |")
    lines.append("| `X` | Consensus items recommended by both models |")
    lines.append("| `Y` | Items recommended only by Model A |")
    lines.append("| `Z` | Items recommended only by Model B |")
    lines.append("| `X ∪ Y` | Baseline ranking from Model A |")
    lines.append("| `X ∪ Z` | Proposed ranking from Model B |")
    lines.append("| `X ∪ Y ∪ Z` | Union / ensemble of both models |")
    lines.append("")

    lines.append("## Average Set Sizes")
    lines.append("")
    lines.append("| K | Avg \\|X\\| | Avg \\|Y\\| | Avg \\|Z\\| | X non-empty % | Y non-empty % | Z non-empty % |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in eval_ks:
        stats = results["set_size_stats"][str(k)]
        lines.append(
            f"| {k} | {stats['X']['avg_size']:.4f} | {stats['Y']['avg_size']:.4f} | "
            f"{stats['Z']['avg_size']:.4f} | {stats['X']['pct_non_empty']:.2f} | "
            f"{stats['Y']['pct_non_empty']:.2f} | {stats['Z']['pct_non_empty']:.2f} |"
        )
    lines.append("")

    for method in ranking_methods:
        lines.append(f"## Configuration Metrics - {METHOD_LABELS[method]}")
        lines.append("")
        for k in eval_ks:
            lines.append(f"### @{k}")
            lines.append("")
            lines.append(f"| Configuration | Precision@{k} | Recall@{k} | Hit Rate@{k} | NDCG@{k} |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for config_name in ("consensus", "baseline", "proposed", "union"):
                row = results["metrics"][method][str(k)][config_name]
                lines.append(
                    f"| {CONFIG_LABELS[config_name]} | {row['precision']:.6f} | {row['recall']:.6f} | "
                    f"{row['hit_rate']:.6f} | {row['ndcg']:.6f} |"
                )
            lines.append("")

    lines.append("## Pure Disagreements (Y vs Z)")
    lines.append("")
    lines.append(
        "| K | Avg \\|Y\\| | Avg \\|Z\\| | Hit Rate Y | Hit Rate Z | Delta (Z-Y) | "
        "Y-only hits | Z-only hits | Both hit | Neither hit | p-value (Z>Y) | Winner |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for k in eval_ks:
        stats = results["pure_disagreements"][str(k)]
        lines.append(
            f"| {k} | {stats['avg_size_y']:.4f} | {stats['avg_size_z']:.4f} | "
            f"{stats['hit_rate_y']:.6f} | {stats['hit_rate_z']:.6f} | "
            f"{stats['delta_hit_rate_z_minus_y']:.6f} | {stats['y_only_hits']} | "
            f"{stats['z_only_hits']} | {stats['both_hit']} | {stats['neither_hit']} | "
            f"{stats['p_value_z_greater_y']:.6g} | {stats['winner']} |"
        )
    lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- `Baseline (X∪Y = A)` and `Proposed (X∪Z = B)` use each model's direct ranked list at cutoff `K`.")
    lines.append("- `Strict Consensus (X)` and `Union / Ensemble (X∪Y∪Z)` are the only configurations affected by the fusion method.")
    lines.append("- The disagreement summary compares whether Model B's unique items (`Z`) capture ground-truth items more often than Model A's unique items (`Y`).")
    return "\n".join(lines)


def save_outputs(results: dict[str, Any], report: str, output_prefix: Path | None = None) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_prefix is None:
        output_prefix = RESULTS_DIR / f"set_intersection_analysis_{timestamp}"
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    with md_path.open("w") as f:
        f.write(report)

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set-based intersection analysis for two recommendation models.")
    parser.add_argument("--input", type=Path, required=True, help="Input file path (json/jsonl/csv/parquet/pickle).")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "json", "jsonl", "csv", "parquet", "pickle"],
        help="Override input format detection.",
    )
    parser.add_argument("--eval-ks", nargs="+", type=int, default=[10, 20], help="Evaluation cutoffs.")
    parser.add_argument(
        "--ranking-method",
        default="both",
        choices=["avg_score", "borda", "both"],
        help="Fusion method for consensus/union rankings.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix without extension.",
    )
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--gt-col", default="ground_truth_items")
    parser.add_argument("--model-a-preds-col", default="model_A_top10_preds")
    parser.add_argument("--model-a-scores-col", default="model_A_scores")
    parser.add_argument("--model-b-preds-col", default="model_B_top10_preds")
    parser.add_argument("--model-b-scores-col", default="model_B_scores")
    parser.add_argument(
        "--missing-score",
        type=float,
        default=0.0,
        help="Default normalized score assigned to an item missing from one model under avg_score fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ranking_methods = ("avg_score", "borda") if args.ranking_method == "both" else (args.ranking_method,)
    columns = ColumnConfig(
        user_id=args.user_col,
        ground_truth=args.gt_col,
        model_a_preds=args.model_a_preds_col,
        model_a_scores=args.model_a_scores_col,
        model_b_preds=args.model_b_preds_col,
        model_b_scores=args.model_b_scores_col,
    )
    config = AnalysisConfig(
        eval_ks=tuple(args.eval_ks),
        ranking_methods=ranking_methods,
        missing_score=args.missing_score,
    )

    df = load_input_dataframe(args.input, args.input_format)
    results = analyze_set_intersections(df, columns=columns, config=config)
    report = generate_markdown_report(results)
    json_path, md_path = save_outputs(results, report, args.output_prefix)

    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
