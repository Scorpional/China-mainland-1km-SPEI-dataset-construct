from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from validation_common import (
    build_model,
    load_training_table,
    regression_metrics,
    select_feature_columns,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_DIR = ROOT / "derived_data" / "training_samples"
DEFAULT_OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "station_groupkfold_cv"
DEFAULT_FILE_TEMPLATE = "spei{scale:02d}_training_samples_1979-2018.csv"
DEFAULT_SCALES = (1, 3, 6, 12, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold out-of-station cross-validation with LightGBM_Leaf63."
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-template", type=str, default=DEFAULT_FILE_TEMPLATE)
    parser.add_argument("--timescales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def load_training_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training sample file not found: {csv_path}")
    return load_training_table(csv_path)


def build_station_folds(
    station_ids: pd.Series,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    _ = random_state  # kept for interface compatibility; GroupKFold itself is deterministic
    unique_station_ids = pd.Index(sorted(station_ids.astype(str).unique()))
    if len(unique_station_ids) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique stations, got {len(unique_station_ids)}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    dummy_x = np.zeros((len(station_ids), 1), dtype=np.float32)
    dummy_y = np.zeros(len(station_ids), dtype=np.float32)
    for train_idx, test_idx in splitter.split(dummy_x, dummy_y, station_ids.to_numpy()):
        folds.append((train_idx, test_idx))
    return folds


def compute_main_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)


def save_outputs(
    output_dir: Path,
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(output_dir / "out_station_cv_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "out_station_cv_summary.csv", index=False)
    pred_df.to_csv(output_dir / "out_station_cv_predictions.csv", index=False)


def plot_outputs(output_dir: Path, pred_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for timescale, group in pred_df.groupby("timescale", sort=False):
        obs = group["obs"].to_numpy()
        pred = group["pred"].to_numpy()
        error = group["error"].to_numpy()

        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        hb = ax.hexbin(obs, pred, gridsize=55, mincnt=1, cmap="viridis")
        lim_min = float(min(obs.min(), pred.min()))
        lim_max = float(max(obs.max(), pred.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Observed SPEI")
        ax.set_ylabel("Predicted SPEI")
        ax.set_title(f"{timescale} obs-pred density")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Count")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{timescale.lower().replace('-', '')}_obs_pred_hexbin.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.hist(error, bins=60, color="#4C78A8", edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Residual (pred - obs)")
        ax.set_ylabel("Count")
        ax.set_title(f"{timescale} residual histogram")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{timescale.lower().replace('-', '')}_residual_hist.png", dpi=300)
        plt.close(fig)

    metric_names = ["R2_mean", "RMSE_mean", "MAE_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    x = np.arange(len(summary_df))
    labels = summary_df["timescale"].tolist()
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, metric_name, color in zip(axes, metric_names, colors):
        ax.bar(x, summary_df[metric_name], color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(metric_name.replace("_mean", ""))
    fig.tight_layout()
    fig.savefig(plot_dir / "timescale_summary_metrics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    fold_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for scale in args.timescales:
        csv_path = args.training_dir / args.file_template.format(scale=scale)
        df = load_training_data(csv_path)
        feature_cols = select_feature_columns(df, drop_soil=True)
        X = df[feature_cols]
        y = df["spei"].to_numpy()
        station_ids = df["station_id"]
        timescale = f"SPEI-{scale}"
        folds = build_station_folds(station_ids, args.folds, args.random_state)

        print(f"\n=== {timescale} ===")
        print(f"Input file: {csv_path}")
        print(f"Features used: {len(feature_cols)}")

        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            train_station_ids = set(station_ids.iloc[train_idx])
            test_station_ids = set(station_ids.iloc[test_idx])
            intersection = train_station_ids.intersection(test_station_ids)
            if intersection:
                overlap_preview = ", ".join(sorted(intersection)[:10])
                raise RuntimeError(
                    f"{timescale} fold {fold} has station leakage: {overlap_preview}"
                )

            n_train_stations = len(train_station_ids)
            n_test_stations = len(test_station_ids)
            n_train_samples = len(train_idx)
            n_test_samples = len(test_idx)

            print(
                f"Fold {fold}: "
                f"train_stations={n_train_stations}, "
                f"test_stations={n_test_stations}, "
                f"train_samples={n_train_samples}, "
                f"test_samples={n_test_samples}"
            )

            pipe = build_model(args.random_state)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            obs = y[test_idx]
            metric_row = compute_main_metrics(obs, pred)
            fold_row: dict[str, float | int | str] = {
                "timescale": timescale,
                "fold": fold,
                "n_train_stations": n_train_stations,
                "n_test_stations": n_test_stations,
                "n_train_samples": n_train_samples,
                "n_test_samples": n_test_samples,
                **metric_row,
            }
            fold_rows.append(fold_row)

            test_meta = df.iloc[test_idx][["station_id", "year", "month", "spei"]].copy()
            test_meta.insert(0, "timescale", timescale)
            test_meta.insert(1, "fold", fold)
            test_meta = test_meta.rename(columns={"spei": "obs"})
            test_meta["pred"] = pred
            test_meta["error"] = test_meta["pred"] - test_meta["obs"]
            prediction_frames.append(test_meta)

        scale_fold_df = pd.DataFrame([row for row in fold_rows if row["timescale"] == timescale])
        summary_rows.append(
            {
                "timescale": timescale,
                "R2_mean": scale_fold_df["R2"].mean(),
                "R2_std": scale_fold_df["R2"].std(ddof=1),
                "RMSE_mean": scale_fold_df["RMSE"].mean(),
                "RMSE_std": scale_fold_df["RMSE"].std(ddof=1),
                "MAE_mean": scale_fold_df["MAE"].mean(),
                "MAE_std": scale_fold_df["MAE"].std(ddof=1),
                "Bias_mean": scale_fold_df["Bias"].mean(),
                "Bias_std": scale_fold_df["Bias"].std(ddof=1),
                "Pearson_r_mean": scale_fold_df["Pearson_r"].mean(),
                "Pearson_r_std": scale_fold_df["Pearson_r"].std(ddof=1),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    save_outputs(args.output_dir, fold_df, summary_df, pred_df)

    if args.make_plots and not pred_df.empty:
        plot_outputs(args.output_dir, pred_df, summary_df)

    print("\nTimescale | R2 | RMSE | MAE | Bias")
    for _, row in summary_df.iterrows():
        print(
            f"{row['timescale']:<9} | "
            f"{row['R2_mean']:.4f} | "
            f"{row['RMSE_mean']:.4f} | "
            f"{row['MAE_mean']:.4f} | "
            f"{row['Bias_mean']:.4f}"
        )

    print(f"\nSaved fold metrics to: {args.output_dir / 'out_station_cv_fold_metrics.csv'}")
    print(f"Saved summary metrics to: {args.output_dir / 'out_station_cv_summary.csv'}")
    print(f"Saved predictions to: {args.output_dir / 'out_station_cv_predictions.csv'}")

    # Out-of-station validation differs from random five-fold validation because
    # all samples from a test station are held out together. It evaluates transfer
    # to unseen station IDs, but nearby training stations can still remain; strict
    # geographic transfer is assessed separately with spatial blocks and buffers.


if __name__ == "__main__":
    main()
