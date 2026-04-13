#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def clean_watch_time(watch_time: float, duration: float) -> float:
    watch_time = max(float(watch_time), 0.0)
    if duration > 0:
        watch_time = min(watch_time, float(duration))
    return watch_time


def explode_parquet(path: Path) -> list[dict[str, float | str]]:
    df_raw = pd.read_parquet(path)
    records: list[dict[str, float | str]] = []
    for _, row in df_raw.iterrows():
        user_id = row["user_id"]
        for item in row["contents"]:
            raw_watch_time = float(item.get("watch_time") or 0.0)
            duration = float(item.get("runtime") or 0.0)
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item["content_id"],
                    "timestamp": item["min_wt_timestamp"].timestamp(),
                    "watch_time": clean_watch_time(raw_watch_time, duration),
                    "duration": duration,
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw parquet shards into a RecBole .inter dataset.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "raw_data",
        help="Directory containing raw parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "thesis_dataset",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--dataset",
        default="thesis_dataset",
        help="Dataset name used to build the <dataset>.inter output file.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Drop users with fewer than this many interactions.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional limit for quick smoke preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_file = output_dir / f"{args.dataset}.inter"

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if args.max_shards is not None:
        parquet_files = parquet_files[: args.max_shards]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found in {raw_dir}")

    print(f"Found {len(parquet_files)} parquet shards in '{raw_dir}'")

    all_records: list[dict[str, float | str]] = []
    for idx, path in enumerate(parquet_files, start=1):
        print(f"  [{idx:02d}/{len(parquet_files)}] Loading {path.name} ...", end=" ")
        records = explode_parquet(path)
        all_records.extend(records)
        print(f"{len(records):,} interactions")

    df = pd.DataFrame(all_records)
    print(f"\nTotal raw interactions: {len(df):,} from {df['user_id'].nunique():,} users")
    print(f"  watch_time=0 (after cleaning) : {(df['watch_time'] == 0).sum():,}")
    print(f"  duration=0                    : {(df['duration'] == 0).sum():,}")

    user_counts = df.groupby("user_id")["item_id"].count()
    valid_users = user_counts[user_counts >= args.min_interactions].index
    df = df[df["user_id"].isin(valid_users)].copy()
    print(
        f"After filtering (>= {args.min_interactions} interactions): "
        f"{df['user_id'].nunique():,} users, {len(df):,} interactions"
    )

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "watch_time": "watch_time:float",
            "duration": "duration:float",
        }
    )
    df_out.to_csv(output_file, sep="\t", index=False)

    print(f"\nSaved to: {output_file}")
    print("\n-- Dataset Summary ---------------------------------------------------")
    print(f"  Users           : {df['user_id'].nunique():,}")
    print(f"  Items           : {df['item_id'].nunique():,}")
    print(f"  Interactions    : {len(df):,}")
    print(f"  Avg seq len     : {len(df) / df['user_id'].nunique():.1f}")
    print(f"  watch_time=0    : {(df['watch_time'] == 0).sum():,}")
    print(f"  duration=0      : {(df['duration'] == 0).sum():,}")
    print(
        f"  Date range      : {pd.to_datetime(df['timestamp'], unit='s').min()} "
        f"-> {pd.to_datetime(df['timestamp'], unit='s').max()}"
    )
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()
