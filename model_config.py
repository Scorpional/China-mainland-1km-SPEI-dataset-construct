from __future__ import annotations

import pandas as pd


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

EXCLUDED_COLUMNS = {
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


def select_production_features(frame: pd.DataFrame) -> list[str]:
    features = [
        name
        for name in frame.columns
        if name not in EXCLUDED_COLUMNS
        and not name.startswith("soil_")
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    if not features:
        raise ValueError("No numeric production features were selected")
    return features
