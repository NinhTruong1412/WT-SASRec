"""
generate_report.py
------------------
Reads the latest SASRec and WTSASRec result JSON files from results/
and produces a final comparison report.
"""

import os, json, glob
from datetime import datetime

RESULTS_DIR = "results"

METRICS = [
    "recall@5", "recall@10", "recall@20",
    "ndcg@5",   "ndcg@10",   "ndcg@20",
    "mrr@10",   "hit@10",
]

def load_latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)), reverse=True)
    if not files:
        return None, None
    path = files[0]
    with open(path) as f:
        data = json.load(f)
    return path, data

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "N/A"

def main():
    sasrec_path,   sasrec_data   = load_latest("baseline_models_*.json")
    wtsasrec_path, wtsasrec_data = load_latest("watch-time_models_*.json")

    # Also try single-model runs stored under individual keys
    sasrec_metrics   = None
    wtsasrec_metrics = None

    if sasrec_data:
        sasrec_metrics = sasrec_data.get("SASRec", sasrec_data)
    if wtsasrec_data:
        wtsasrec_metrics = wtsasrec_data.get("WTSASRec", wtsasrec_data)

    # Fallback: look for any json with SASRec or WTSASRec key
    if sasrec_metrics is None or wtsasrec_metrics is None:
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), reverse=True):
            with open(f) as fp:
                d = json.load(fp)
            if sasrec_metrics is None and "SASRec" in d:
                sasrec_metrics = d["SASRec"]
                sasrec_path = f
            if wtsasrec_metrics is None and "WTSASRec" in d:
                wtsasrec_metrics = d["WTSASRec"]
                wtsasrec_path = f

    if not sasrec_metrics or not wtsasrec_metrics:
        print("Could not find result files for both models.")
        print(f"  SASRec results:   {sasrec_path}")
        print(f"  WTSASRec results: {wtsasrec_path}")
        return

    col_w = 12
    sep   = "=" * (16 + col_w * 3)
    models = ["SASRec", "WTSASRec", "Δ (WT - Base)"]

    lines = []
    lines.append(sep)
    lines.append("  SASRec vs WTSASRec — Final Comparison Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append(f"{'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in models))
    lines.append("-" * (16 + col_w * 3))
    for metric in METRICS:
        bv = sasrec_metrics.get(metric, None)
        wv = wtsasrec_metrics.get(metric, None)
        delta = "N/A"
        if bv is not None and wv is not None:
            delta = f"{float(wv) - float(bv):+.4f}"
        row = f"{metric:<16}{fmt(bv):>{col_w}}{fmt(wv):>{col_w}}{delta:>{col_w}}"
        lines.append(row)
    lines.append(sep)
    lines.append(f"  Source SASRec:   {sasrec_path}")
    lines.append(f"  Source WTSASRec: {wtsasrec_path}")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)

    out = os.path.join(RESULTS_DIR, f"comparison_sasrec_vs_wtsasrec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {out}")

if __name__ == "__main__":
    main()
