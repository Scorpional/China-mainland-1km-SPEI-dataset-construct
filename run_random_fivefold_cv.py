from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

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
DEFAULT_OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "random_fivefold_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random five-fold cross-validation for revised SPEI station-month samples."
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-template", default=DEFAULT_FILE_TEMPLATE)
    parser.add_argument("--timescales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for scale in args.timescales:
        path = args.training_dir / args.file_template.format(scale=scale)
        df = load_training_table(path)
        features = select_feature_columns(df, drop_soil=True)
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            model = build_model(args.random_state)
            model.fit(train_df[features], train_df["spei"])
            pred = model.predict(test_df[features])
            metrics = regression_metrics(test_df["spei"].to_numpy(), pred)
            fold_rows.append(
                {
                    "timescale": f"SPEI-{scale}",
                    "fold": fold,
                    "n_train_samples": len(train_df),
                    "n_test_samples": len(test_df),
                    **metrics,
                }
            )

            out = test_df[["station_id", "year", "month", "spei"]].copy()
            out.insert(0, "timescale", f"SPEI-{scale}")
            out.insert(1, "fold", fold)
            out = out.rename(columns={"spei": "obs"})
            out["pred"] = pred
            out["error"] = out["pred"] - out["obs"]
            prediction_frames.append(out)
            print(
                f"SPEI-{scale} fold {fold}: train_samples={len(train_df)}, "
                f"test_samples={len(test_df)}, R2={metrics['R2']:.4f}"
            )

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

    fold_df.to_csv(args.output_dir / "random_fivefold_fold_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "random_fivefold_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "random_fivefold_predictions.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
