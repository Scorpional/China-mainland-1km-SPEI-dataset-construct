from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def anomaly_source_columns(df: pd.DataFrame, scale: int) -> list[str]:
    suffixes = (f"_roll{scale}_mean", f"_roll{scale}_sum")
    return sorted(
        name
        for name in df.columns
        if name.startswith("dyn_")
        and not name.startswith(("dyn_lon", "dyn_lat", "dyn_time"))
        and pd.api.types.is_numeric_dtype(df[name])
        and "_lag" not in name
        and ("_roll" not in name or name.endswith(suffixes))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add monthly station climatology and anomaly predictors."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--scale", type=int, required=True, choices=[1, 3, 6, 12, 24])
    parser.add_argument("--baseline-start", type=int, default=1979)
    parser.add_argument("--baseline-end", type=int, default=2018)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv, dtype={"station_id": str})
    source_columns = anomaly_source_columns(frame, args.scale)
    if not source_columns:
        raise ValueError("No current or rolling dynamic predictors were found")
    baseline = frame[frame["year"].between(args.baseline_start, args.baseline_end)]
    climatology = baseline.groupby(["station_id", "month"], as_index=False)[source_columns].mean()
    climatology = climatology.rename(
        columns={name: f"clim_{name}" for name in source_columns}
    )
    output = frame.merge(
        climatology, on=["station_id", "month"], how="left", validate="many_to_one"
    )
    for name in source_columns:
        output[f"anom_{name}"] = output[name] - output[f"clim_{name}"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output)} rows with climatology and anomaly predictors")


if __name__ == "__main__":
    main()
