"""
preprocess.py
-------------
Converts the raw BigQuery parquet export into RecBole .inter format.

Input : bigquery-export_...parquet
Output: data/thesis_dataset/thesis_dataset.inter

RecBole sequential models require a tab-separated .inter file with columns:
    user_id:token   item_id:token   timestamp:float   watch_time:float
"""

import os
import pandas as pd

RAW_FILE = "bigquery-export_user_item_interaction_2026_03_27_user_item_interaction-part-000000000000.parquet"
OUTPUT_DIR = "data/thesis_dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "thesis_dataset.inter")
MIN_INTERACTIONS = 5  # drop users with fewer interactions (standard sequential rec practice)


def main():
    print("Loading parquet file...")
    df_raw = pd.read_parquet(RAW_FILE)
    print(f"  Loaded {len(df_raw):,} users")

    # ── Explode nested contents array into flat rows ──────────────────────────
    print("Exploding nested interactions...")
    records = []
    for _, row in df_raw.iterrows():
        user_id = row["user_id"]
        for item in row["contents"]:
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item["content_id"],
                    "timestamp": item["min_wt_timestamp"].timestamp(),
                    "watch_time": float(item.get("watch_time") or 0.0),
                }
            )

    df = pd.DataFrame(records)
    print(f"  Total raw interactions: {len(df):,}")

    # ── Filter users with fewer than MIN_INTERACTIONS ─────────────────────────
    user_counts = df.groupby("user_id")["item_id"].count()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    df = df[df["user_id"].isin(valid_users)].copy()
    print(
        f"  After filtering (>= {MIN_INTERACTIONS} interactions): "
        f"{df['user_id'].nunique():,} users, {len(df):,} interactions"
    )

    # ── Sort by user then timestamp (sequential order) ────────────────────────
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── Write RecBole .inter file ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # RecBole header uses field:type notation
    df_out = df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "watch_time": "watch_time:float",
        }
    )
    df_out.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"  Saved to: {OUTPUT_FILE}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n── Dataset Summary ────────────────────────────────")
    print(f"  Users        : {df['user_id'].nunique():,}")
    print(f"  Items        : {df['item_id'].nunique():,}")
    print(f"  Interactions : {len(df):,}")
    print(f"  Avg seq len  : {len(df) / df['user_id'].nunique():.1f}")
    print(f"  Date range   : {pd.to_datetime(df['timestamp'], unit='s').min()} "
          f"→ {pd.to_datetime(df['timestamp'], unit='s').max()}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
