from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_set_intersection_analysis import AnalysisConfig, analyze_set_intersections


def _descending_scores(length: int) -> list[float]:
    return [float(length - idx) for idx in range(length)]


def test_analyze_set_intersections_reports_expected_core_metrics():
    df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "ground_truth_items": [100],
                "model_A_top10_preds": list(range(1, 21)),
                "model_A_scores": _descending_scores(20),
                "model_B_top10_preds": [1, 2, 100, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "model_B_scores": _descending_scores(20),
            },
            {
                "user_id": "u2",
                "ground_truth_items": [200],
                "model_A_top10_preds": [200, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                "model_A_scores": _descending_scores(20),
                "model_B_top10_preds": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                "model_B_scores": _descending_scores(20),
            },
        ]
    )

    results = analyze_set_intersections(
        df,
        config=AnalysisConfig(eval_ks=(10, 20), ranking_methods=("avg_score", "borda")),
    )

    size_stats_10 = results["set_size_stats"]["10"]
    assert size_stats_10["X"]["avg_size"] == 9.0
    assert size_stats_10["Y"]["avg_size"] == 1.0
    assert size_stats_10["Z"]["avg_size"] == 1.0

    avg_score_10 = results["metrics"]["avg_score"]["10"]
    borda_10 = results["metrics"]["borda"]["10"]

    assert avg_score_10["baseline"]["hit_rate"] == 0.5
    assert avg_score_10["proposed"]["hit_rate"] == 0.5
    assert avg_score_10["consensus"]["hit_rate"] == 0.0
    assert avg_score_10["union"]["hit_rate"] == 1.0
    assert avg_score_10["baseline"]["precision"] == 0.05
    assert avg_score_10["union"]["precision"] == 0.1

    assert borda_10["baseline"]["hit_rate"] == 0.5
    assert borda_10["proposed"]["hit_rate"] == 0.5
    assert borda_10["consensus"]["hit_rate"] == 0.0
    assert borda_10["union"]["hit_rate"] == 1.0

    disagreement_10 = results["pure_disagreements"]["10"]
    assert disagreement_10["hit_rate_y"] == 0.5
    assert disagreement_10["hit_rate_z"] == 0.5
    assert disagreement_10["y_only_hits"] == 1
    assert disagreement_10["z_only_hits"] == 1
    assert disagreement_10["both_hit"] == 0
    assert disagreement_10["neither_hit"] == 0

    size_stats_20 = results["set_size_stats"]["20"]
    assert size_stats_20["X"]["avg_size"] == 19.0
    assert size_stats_20["Y"]["avg_size"] == 1.0
    assert size_stats_20["Z"]["avg_size"] == 1.0


def test_analyze_set_intersections_requires_enough_depth_for_max_k():
    df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "ground_truth_items": [1],
                "model_A_top10_preds": list(range(10)),
                "model_A_scores": _descending_scores(10),
                "model_B_top10_preds": list(range(10)),
                "model_B_scores": _descending_scores(10),
            }
        ]
    )

    with pytest.raises(ValueError, match="at least 20 items"):
        analyze_set_intersections(
            df,
            config=AnalysisConfig(eval_ks=(10, 20), ranking_methods=("avg_score",)),
        )
