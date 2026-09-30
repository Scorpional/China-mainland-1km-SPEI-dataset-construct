from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.warp import transform
from sklearn.model_selection import GroupKFold

from validation_common import (
    DEFAULT_FILE_TEMPLATE,
    DEFAULT_SCALES,
    build_model,
    haversine_distance_matrix_km,
    load_training_table,
    regression_metrics,
    select_feature_columns,
    station_coordinates,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_DIR = ROOT / "derived_data" / "training_samples"
DEFAULT_OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "spatial_block_buffer_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-fold spatial-block cross-validation with an optional distance buffer."
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-template", default=DEFAULT_FILE_TEMPLATE)
    parser.add_argument("--timescales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-km", type=float, default=200.0)
    parser.add_argument("--buffer-km", type=float, default=100.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--drop-terrain",
        action="store_true",
        help="Exclude canonical NASADEM terrain predictors for ablation analysis.",
    )
    parser.add_argument(
        "--drop-coordinates",
        action="store_true",
        help="Exclude latitude and longitude predictors to test axis-aligned spatial artifacts.",
    )
    return parser.parse_args()


def build_spatial_folds(
    coords: pd.DataFrame,
    n_splits: int,
    block_km: float,
) -> tuple[pd.DataFrame, list[tuple[int, set[str], set[str]]]]:
    if block_km <= 0:
        raise ValueError("block-km must be positive")
    china_albers = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=m +no_defs"
    x, y = transform(
        "EPSG:4326",
        china_albers,
        coords["lon"].astype(float).tolist(),
        coords["lat"].astype(float).tolist(),
    )
    result = coords.copy()
    result["x_m"] = x
    result["y_m"] = y
    block_m = block_km * 1000.0
    result["block_x"] = np.floor((result["x_m"] - result["x_m"].min()) / block_m).astype(int)
    result["block_y"] = np.floor((result["y_m"] - result["y_m"].min()) / block_m).astype(int)
    result["spatial_block"] = result["block_x"].astype(str) + "_" + result["block_y"].astype(str)
    if result["spatial_block"].nunique() < n_splits:
        raise ValueError("The selected block size produces fewer spatial blocks than folds")

    splitter = GroupKFold(n_splits=n_splits)
    split_rows: list[tuple[int, set[str], set[str]]] = []
    result["fold"] = 0
    dummy = np.zeros((len(result), 1), dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(dummy, groups=result["spatial_block"].to_numpy()), start=1
    ):
        train_ids = set(result.iloc[train_idx]["station_id"])
        test_ids = set(result.iloc[test_idx]["station_id"])
        if train_ids.intersection(test_ids):
            raise RuntimeError(f"Station leakage before buffering in fold {fold}")
        result.loc[result.index[test_idx], "fold"] = fold
        split_rows.append((fold, train_ids, test_ids))
    return result, split_rows


def apply_distance_buffer(
    coords: pd.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    buffer_km: float,
) -> tuple[set[str], pd.DataFrame]:
    train = coords[coords["station_id"].isin(train_ids)].reset_index(drop=True)
    test = coords[coords["station_id"].isin(test_ids)].reset_index(drop=True)
    distances = haversine_distance_matrix_km(
        test["lon"].to_numpy(),
        test["lat"].to_numpy(),
        train["lon"].to_numpy(),
        train["lat"].to_numpy(),
    )
    min_to_test = distances.min(axis=0)
    keep_train = min_to_test >= buffer_km if buffer_km > 0 else np.ones(len(train), dtype=bool)
    buffered_train = set(train.loc[keep_train, "station_id"])
    if not buffered_train:
        raise ValueError("Distance buffer removed all training stations")

    kept = train[train["station_id"].isin(buffered_train)].reset_index(drop=True)
    kept_distances = haversine_distance_matrix_km(
        test["lon"].to_numpy(),
        test["lat"].to_numpy(),
        kept["lon"].to_numpy(),
        kept["lat"].to_numpy(),
    )
    distance_df = test[["station_id", "lon", "lat"]].copy()
    distance_df["nearest_training_station_km"] = kept_distances.min(axis=1)
    return buffered_train, distance_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    distance_frames: list[pd.DataFrame] = []

    for scale in args.timescales:
        path = args.training_dir / args.file_template.format(scale=scale)
        df = load_training_table(path)
        features = select_feature_columns(
            df,
            drop_soil=True,
            drop_terrain=args.drop_terrain,
            drop_coordinates=args.drop_coordinates,
        )
        coords = station_coordinates(df)
        assignments, folds = build_spatial_folds(coords, args.folds, args.block_km)
        assignments.insert(0, "timescale", f"SPEI-{scale}")
        assignment_frames.append(assignments)

        for fold, raw_train_ids, test_ids in folds:
            train_ids, distance_df = apply_distance_buffer(coords, raw_train_ids, test_ids, args.buffer_km)
            if train_ids.intersection(test_ids):
                raise RuntimeError(f"Station leakage after buffering in SPEI-{scale} fold {fold}")
            if args.buffer_km > 0 and distance_df["nearest_training_station_km"].min() + 1e-6 < args.buffer_km:
                raise RuntimeError(f"Distance-buffer violation in SPEI-{scale} fold {fold}")

            train_mask = df["station_id"].isin(train_ids)
            test_mask = df["station_id"].isin(test_ids)
            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()
            print(
                f"SPEI-{scale} fold {fold}: blocks={assignments.loc[assignments['fold'] == fold, 'spatial_block'].nunique()}, "
                f"train_stations={len(train_ids)}, test_stations={len(test_ids)}, "
                f"train_samples={len(train_df)}, test_samples={len(test_df)}, "
                f"min_distance_km={distance_df['nearest_training_station_km'].min():.1f}"
            )

            model = build_model(args.random_state)
            model.fit(train_df[features], train_df["spei"])
            pred = model.predict(test_df[features])
            metrics = regression_metrics(test_df["spei"].to_numpy(), pred)
            fold_metrics.append(
                {
                    "timescale": f"SPEI-{scale}",
                    "fold": fold,
                    "block_km": args.block_km,
                    "buffer_km": args.buffer_km,
                    "n_train_stations_before_buffer": len(raw_train_ids),
                    "n_train_stations": len(train_ids),
                    "n_test_stations": len(test_ids),
                    "n_train_samples": len(train_df),
                    "n_test_samples": len(test_df),
                    "min_test_to_train_km": float(distance_df["nearest_training_station_km"].min()),
                    **metrics,
                }
            )
            out = test_df[["station_id", "year", "month", "spei"]].copy()
            out.insert(0, "timescale", f"SPEI-{scale}")
            out.insert(1, "fold", fold)
            out = out.rename(columns={"spei": "obs"})
            out["pred"] = pred
            out["error"] = out["pred"] - out["obs"]
            predictions.append(out)

            distance_df.insert(0, "timescale", f"SPEI-{scale}")
            distance_df.insert(1, "fold", fold)
            distance_frames.append(distance_df)

    fold_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(predictions, ignore_index=True)
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

    fold_df.to_csv(args.output_dir / "spatial_block_cv_fold_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "spatial_block_cv_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "spatial_block_cv_predictions.csv", index=False)
    pd.concat(assignment_frames, ignore_index=True).to_csv(
        args.output_dir / "spatial_block_station_assignments.csv", index=False
    )
    pd.concat(distance_frames, ignore_index=True).to_csv(
        args.output_dir / "spatial_block_test_station_distances.csv", index=False
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
