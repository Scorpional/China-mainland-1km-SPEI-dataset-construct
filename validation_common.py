from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


DEFAULT_SCALES = (1, 3, 6, 12, 24)
DEFAULT_FILE_TEMPLATE = "spei{scale:02d}_training_samples_1979-2018.csv"
EXCLUDED_FEATURE_COLS = {
    "spei",
    "scale",
    "station_id",
    "month_end_date",
    "source_date",
    "date",
    "year_month",
    "time",
    "fold",
    "station_elevation_m",
    "Elevation(m)",
    "dyn_lon",
    "dyn_lat",
    "station_lon",
    "station_lat",
    "cmfd_grid_lon",
    "cmfd_grid_lat",
}
MODEL_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 0,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def load_training_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str})
    required = {"station_id", "year", "month", "spei"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df["station_id"] = df["station_id"].astype(str)
    df = df.dropna(subset=["station_id", "year", "month", "spei"]).reset_index(drop=True)
    return df


def select_feature_columns(
    df: pd.DataFrame,
    drop_soil: bool = True,
    drop_terrain: bool = False,
    drop_coordinates: bool = False,
) -> list[str]:
    features = [
        name
        for name in df.columns
        if name not in EXCLUDED_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[name])
    ]
    if drop_soil:
        features = [name for name in features if not name.startswith("soil_")]
    if drop_terrain:
        features = [name for name in features if not name.startswith("dem_")]
    if drop_coordinates:
        features = [name for name in features if name not in {"lat", "lon"}]
    if not features:
        raise ValueError("No numeric model features were selected")
    return features


def build_model(random_state: int = 42) -> Pipeline:
    params = dict(MODEL_PARAMS)
    params["random_state"] = random_state
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")
    return Pipeline(
        [
            ("imputer", imputer),
            ("model", LGBMRegressor(**params)),
        ]
    )


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if obs.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Pearson_r": np.nan}
    pearson = float(np.corrcoef(obs, pred)[0, 1]) if obs.size > 1 else np.nan
    return {
        "R2": float(r2_score(obs, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(obs, pred))),
        "MAE": float(mean_absolute_error(obs, pred)),
        "Bias": float(np.mean(pred - obs)),
        "Pearson_r": pearson,
    }


def station_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    lon_col = next((name for name in ["station_lon", "lon", "dyn_lon"] if name in df.columns), None)
    lat_col = next((name for name in ["station_lat", "lat", "dyn_lat"] if name in df.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError("Could not identify station longitude and latitude columns")
    coords = df[["station_id", lon_col, lat_col]].rename(columns={lon_col: "lon", lat_col: "lat"}).copy()
    coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    spread = coords.groupby("station_id")[["lon", "lat"]].nunique(dropna=False)
    inconsistent = spread[(spread["lon"] > 1) | (spread["lat"] > 1)]
    if not inconsistent.empty:
        raise ValueError(f"Station coordinates vary over time for: {inconsistent.index[:10].tolist()}")
    coords = coords.drop_duplicates("station_id").dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return coords


def haversine_distance_matrix_km(
    lon_a: np.ndarray,
    lat_a: np.ndarray,
    lon_b: np.ndarray,
    lat_b: np.ndarray,
) -> np.ndarray:
    lon1 = np.radians(np.asarray(lon_a, dtype=float))[:, None]
    lat1 = np.radians(np.asarray(lat_a, dtype=float))[:, None]
    lon2 = np.radians(np.asarray(lon_b, dtype=float))[None, :]
    lat2 = np.radians(np.asarray(lat_b, dtype=float))[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
