from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


MERGE_KEYS = ["station_id", "year", "month"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    angle = 2 * math.pi * df["month"] / 12.0
    df["month_sin"] = angle.map(math.sin)
    df["month_cos"] = angle.map(math.cos)
    return df


def add_lagged_features(df: pd.DataFrame, numeric_cols: list[str], scale: int, max_lag: int) -> pd.DataFrame:
    df = df.sort_values(["station_id", "year", "month"]).copy()
    groups = df.groupby("station_id", group_keys=False)

    for col in numeric_cols:
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag{lag}"] = groups[col].shift(lag)
        df[f"{col}_roll{scale}_mean"] = groups[col].transform(lambda s: s.rolling(scale, min_periods=1).mean())
        if "prec" in col.lower() or col.lower() == "pr":
            df[f"{col}_roll{scale}_sum"] = groups[col].transform(lambda s: s.rolling(scale, min_periods=1).sum())

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build station-month training samples for one SPEI timescale.")
    parser.add_argument("label_csv", type=Path, help="Monthly station SPEI labels")
    parser.add_argument("dynamic_csv", type=Path, help="Station-month dynamic predictors")
    parser.add_argument("static_csv", type=Path, help="Station static predictors")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("--scale", type=int, required=True, help="SPEI timescale in months")
    parser.add_argument("--extra-path", action="append", type=Path, default=[], help="Optional extra tables to merge")
    parser.add_argument("--max-lag", type=int, default=None, help="Maximum lag count; default is scale - 1")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = pd.read_csv(args.label_csv)
    labels = labels[labels["scale"] == args.scale].copy()
    labels["station_id"] = labels["station_id"].astype(str)
    if args.year_min is not None:
        labels = labels[labels["year"] >= args.year_min].copy()
    if args.year_max is not None:
        labels = labels[labels["year"] <= args.year_max].copy()

    dynamic = pd.read_csv(args.dynamic_csv)
    dynamic["station_id"] = dynamic["station_id"].astype(str)
    dynamic_source_cols = [
        col
        for col in dynamic.columns
        if col not in set(MERGE_KEYS) and col not in {"lat", "lon"} and pd.api.types.is_numeric_dtype(dynamic[col])
    ]
    dynamic_rename = {col: f"dyn_{col}" for col in dynamic.columns if col not in set(MERGE_KEYS)}
    dynamic = dynamic.rename(columns=dynamic_rename)

    static = pd.read_csv(args.static_csv)
    static["station_id"] = static["station_id"].astype(str)

    df = labels.merge(dynamic, on=MERGE_KEYS, how="inner")
    df = df.merge(static, on="station_id", how="left")

    time_varying_cols = [dynamic_rename[col] for col in dynamic_source_cols]
    for extra_path in args.extra_path:
        extra = pd.read_csv(extra_path)
        extra["station_id"] = extra["station_id"].astype(str)
        join_keys = [key for key in MERGE_KEYS if key in extra.columns]
        extra_numeric = [col for col in extra.columns if col not in set(join_keys) and pd.api.types.is_numeric_dtype(extra[col])]
        df = df.merge(extra, on=join_keys, how="left")
        time_varying_cols.extend(extra_numeric)

    df = add_calendar_features(df)
    max_lag = args.max_lag if args.max_lag is not None else max(1, args.scale - 1)
    df = add_lagged_features(df, sorted(set(time_varying_cols)), args.scale, max_lag)
    df = df.dropna(subset=["spei"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
