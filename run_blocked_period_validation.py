from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from add_station_climatology_anomalies import anomaly_source_columns
from validation_common import (
    DEFAULT_FILE_TEMPLATE,
    DEFAULT_SCALES,
    build_model,
    load_training_table,
    regression_metrics,
    select_feature_columns,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_DIR = ROOT / "derived_data" / "training_samples"
DEFAULT_OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "blocked_period_validation"
DEFAULT_TEST_PERIODS = ("1999-2003", "2004-2008", "2009-2013", "2014-2018")


def parse_period(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid period '{value}'; expected YYYY-YYYY") from exc
    if start > end:
        raise argparse.ArgumentTypeError(f"Invalid period '{value}'; start is after end")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward blocked-period validation. Each test period contains complete years, and training uses only "
            "years before the test period."
        )
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-template", default=DEFAULT_FILE_TEMPLATE)
    parser.add_argument("--timescales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--test-periods", nargs="+", default=list(DEFAULT_TEST_PERIODS))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def rebuild_train_period_anomalies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Estimate station-month climatology only from the training period."""
    source_columns = anomaly_source_columns(train_df, scale)
    climatology = train_df.groupby(["station_id", "month"], as_index=False)[source_columns].mean()
    climatology = climatology.rename(columns={name: f"clim_{name}" for name in source_columns})
    derived_columns = [
        name
        for source in source_columns
        for name in [f"clim_{source}", f"anom_{source}"]
    ]

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.drop(columns=[name for name in derived_columns if name in frame.columns]).copy()
        result["_source_order"] = range(len(result))
        result = result.merge(climatology, on=["station_id", "month"], how="left", validate="many_to_one")
        result = result.sort_values("_source_order").drop(columns="_source_order").reset_index(drop=True)
        for source in source_columns:
            result[f"anom_{source}"] = result[source] - result[f"clim_{source}"]
        return result

    revised_train = attach(train_df)
    revised_test = attach(test_df)
    missing = revised_test[[f"clim_{name}" for name in source_columns]].isna().sum().sum()
    if int(missing) > 0:
        raise ValueError(
            f"Training-period climatology is unavailable for {int(missing)} test feature values in SPEI-{scale}"
        )
    return revised_train, revised_test, source_columns


def main() -> None:
    args = parse_args()
    periods = [parse_period(value) for value in args.test_periods]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for scale in args.timescales:
        df = load_training_table(args.training_dir / args.file_template.format(scale=scale))
        features = select_feature_columns(df, drop_soil=True)
        for fold, (test_start, test_end) in enumerate(periods, start=1):
            train_df = df[df["year"] < test_start].copy()
            test_df = df[(df["year"] >= test_start) & (df["year"] <= test_end)].copy()
            if train_df.empty or test_df.empty:
                print(f"Skip SPEI-{scale} {test_start}-{test_end}: empty training or test data")
                continue
            train_years = set(train_df["year"].astype(int))
            test_years = set(test_df["year"].astype(int))
            if train_years.intersection(test_years):
                raise RuntimeError(f"Year leakage in SPEI-{scale} fold {fold}")
            if max(train_years) >= min(test_years):
                raise RuntimeError(f"Training is not strictly earlier than testing in SPEI-{scale} fold {fold}")

            train_df, test_df, anomaly_sources = rebuild_train_period_anomalies(
                train_df,
                test_df,
                scale,
            )

            print(
                f"SPEI-{scale} fold {fold}: train={min(train_years)}-{max(train_years)}, "
                f"test={test_start}-{test_end}, train_samples={len(train_df)}, test_samples={len(test_df)}"
            )
            model = build_model(args.random_state)
            model.fit(train_df[features], train_df["spei"])
            pred = model.predict(test_df[features])
            metrics = regression_metrics(test_df["spei"].to_numpy(), pred)
            fold_rows.append(
                {
                    "timescale": f"SPEI-{scale}",
                    "fold": fold,
                    "train_start": int(min(train_years)),
                    "train_end": int(max(train_years)),
                    "test_start": test_start,
                    "test_end": test_end,
                    "n_train_years": len(train_years),
                    "n_test_years": len(test_years),
                    "n_train_samples": len(train_df),
                    "n_test_samples": len(test_df),
                    "n_train_period_anomaly_sources": len(anomaly_sources),
                    **metrics,
                }
            )
            out = test_df[["station_id", "year", "month", "spei"]].copy()
            out.insert(0, "timescale", f"SPEI-{scale}")
            out.insert(1, "fold", fold)
            out.insert(2, "test_period", f"{test_start}-{test_end}")
            out = out.rename(columns={"spei": "obs"})
            out["pred"] = pred
            out["error"] = out["pred"] - out["obs"]
            prediction_frames.append(out)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for timescale, group in fold_df.groupby("timescale", sort=False):
        pooled = pred_df[pred_df["timescale"] == timescale]
        pooled_metrics = regression_metrics(pooled["obs"].to_numpy(), pooled["pred"].to_numpy())
        row: dict[str, object] = {"timescale": timescale}
        for metric in ["R2", "RMSE", "MAE", "Bias", "Pearson_r"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_pooled"] = pooled_metrics[metric]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    fold_df.to_csv(args.output_dir / "blocked_period_fold_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "blocked_period_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "blocked_period_predictions.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
