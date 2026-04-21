from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMRegressor
from netCDF4 import Dataset
from rasterio.windows import Window
from rasterio.warp import transform
from sklearn.impute import SimpleImputer


ROOT = Path(r"D:\GitRepository\bte")
CMFD_DIR = ROOT / "data_downloads" / "cmfd" / "Data_forcing_01mo_010deg"
STATIC_DIR = ROOT / "derived_data" / "grid_1km_china" / "static"
TRAIN_DIR = ROOT / "derived_data" / "training_samples"
CACHE_DIR = ROOT / "derived_data" / "cache" / "cmfd_climatology"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CMFD_FILES = {
    "lrad": CMFD_DIR / "lrad_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "prec": CMFD_DIR / "prec_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "pres": CMFD_DIR / "pres_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "shum": CMFD_DIR / "shum_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "srad": CMFD_DIR / "srad_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "temp": CMFD_DIR / "temp_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
    "wind": CMFD_DIR / "wind_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc",
}

TRAINING_TABLES = {
    1: TRAIN_DIR / "spei01_demsoil_nomodis_anomaly_training_samples_1979-2018.csv",
    3: TRAIN_DIR / "spei03_demsoil_nomodis_anomaly_training_samples_1979-2018.csv",
    6: TRAIN_DIR / "spei06_demsoil_nomodis_anomaly_training_samples_1979-2018.csv",
    12: TRAIN_DIR / "spei12_demsoil_nomodis_anomaly_training_samples_1979-2018.csv",
    24: TRAIN_DIR / "spei24_demsoil_nomodis_anomaly_training_samples_1979-2018.csv",
}

MODEL_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

STATIC_RASTERS = {
    "dem_elev_m": STATIC_DIR / "dem_elev_m.tif",
    "dem_relief_1km": STATIC_DIR / "dem_relief_1km.tif",
    "dem_std_1km": STATIC_DIR / "dem_std_1km.tif",
    "soil_bdod_0_5cm": STATIC_DIR / "soil_bdod_0_5cm.tif",
    "soil_clay_0_5cm": STATIC_DIR / "soil_clay_0_5cm.tif",
    "soil_phh2o_0_5cm": STATIC_DIR / "soil_phh2o_0_5cm.tif",
    "soil_sand_0_5cm": STATIC_DIR / "soil_sand_0_5cm.tif",
    "soil_silt_0_5cm": STATIC_DIR / "soil_silt_0_5cm.tif",
    "soil_soc_0_5cm": STATIC_DIR / "soil_soc_0_5cm.tif",
}


def parse_month(month_text: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month_text}-01")


def fit_model(scale: int) -> tuple[LGBMRegressor, SimpleImputer, list[str]]:
    df = pd.read_csv(TRAINING_TABLES[scale])
    feature_cols = [
        col
        for col in df.columns
        if col not in {"spei", "scale", "station_id", "month_end_date", "time"} and pd.api.types.is_numeric_dtype(df[col])
    ]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(df[feature_cols])
    y_train = df["spei"].to_numpy()
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)
    return model, imputer, feature_cols


def open_cmfd() -> tuple[dict[str, Dataset], np.ndarray, np.ndarray, pd.DatetimeIndex]:
    datasets = {name: Dataset(path) for name, path in CMFD_FILES.items()}
    lons = datasets["prec"].variables["lon"][:].astype("float32")
    lats = datasets["prec"].variables["lat"][:].astype("float32")
    times = pd.date_range("1979-01-01", periods=480, freq="MS")
    return datasets, lons, lats, times


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.cumsum(arr, axis=0, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    out[: window - 1] = np.nan
    out[window - 1] = cumsum[window - 1]
    out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def load_climatology(scale: int, var_name: str) -> dict[str, np.ndarray]:
    cache_path = CACHE_DIR / f"scale_{scale:02d}_{var_name}_climatology.npz"
    if cache_path.exists():
        obj = np.load(cache_path)
        return {key: obj[key] for key in obj.files}

    with Dataset(CMFD_FILES[var_name]) as ds:
        arr = ds.variables[var_name][:].astype("float32")

    current = np.stack([np.nanmean(arr[month::12], axis=0) for month in range(12)], axis=0).astype("float32")
    rolling_mean = rolling_sum(arr, scale) / float(scale)
    rolling_mean_clim = np.stack(
        [
            np.nanmean(
                rolling_mean[np.array([idx for idx in range(month, arr.shape[0], 12) if idx >= scale - 1])],
                axis=0,
            )
            for month in range(12)
        ],
        axis=0,
    ).astype("float32")

    payload = {"current": current, "roll_mean": rolling_mean_clim}
    if var_name == "prec":
        rolling_prec_sum = rolling_sum(arr, scale)
        rolling_sum_clim = np.stack(
            [
                np.nanmean(
                    rolling_prec_sum[np.array([idx for idx in range(month, arr.shape[0], 12) if idx >= scale - 1])],
                    axis=0,
                )
                for month in range(12)
            ],
            axis=0,
        ).astype("float32")
        payload["roll_sum"] = rolling_sum_clim

    np.savez_compressed(cache_path, **payload)
    return payload


def get_dynamic_arrays(
    datasets: dict[str, Dataset],
    time_index: int,
    scale: int,
    climatology: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    month_index = time_index % 12
    data: dict[str, np.ndarray] = {}
    for var_name, ds in datasets.items():
        var = ds.variables[var_name]
        current = var[time_index].astype("float32")
        data[f"dyn_{var_name}"] = current
        data[f"clim_dyn_{var_name}"] = climatology[var_name]["current"][month_index]
        data[f"anom_dyn_{var_name}"] = current - data[f"clim_dyn_{var_name}"]

        for lag in range(1, scale):
            data[f"dyn_{var_name}_lag{lag}"] = var[time_index - lag].astype("float32")

        window_arr = var[time_index - scale + 1 : time_index + 1].astype("float32")
        rolling_mean = window_arr.mean(axis=0).astype("float32")
        data[f"dyn_{var_name}_roll{scale}_mean"] = rolling_mean
        data[f"clim_dyn_{var_name}_roll{scale}_mean"] = climatology[var_name]["roll_mean"][month_index]
        data[f"anom_dyn_{var_name}_roll{scale}_mean"] = rolling_mean - data[f"clim_dyn_{var_name}_roll{scale}_mean"]

        if var_name == "prec":
            rolling_sum_arr = window_arr.sum(axis=0).astype("float32")
            data[f"dyn_{var_name}_roll{scale}_sum"] = rolling_sum_arr
            data[f"clim_dyn_{var_name}_roll{scale}_sum"] = climatology[var_name]["roll_sum"][month_index]
            data[f"anom_dyn_{var_name}_roll{scale}_sum"] = rolling_sum_arr - data[f"clim_dyn_{var_name}_roll{scale}_sum"]

    return data


def block_lon_lat(src: rasterio.DatasetReader, row0: int, row1: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(row0, row1)
    cols = np.arange(src.width)
    xs = src.transform.c + (cols + 0.5) * src.transform.a
    ys = src.transform.f + (rows + 0.5) * src.transform.e
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = transform(src.crs, "EPSG:4326", xx.ravel().tolist(), yy.ravel().tolist())
    return np.asarray(lon, dtype="float32").reshape(xx.shape), np.asarray(lat, dtype="float32").reshape(xx.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final LightGBM model and reconstruct 1 km monthly SPEI grids.")
    parser.add_argument("--scale", type=int, required=True, choices=[1, 3, 6, 12, 24])
    parser.add_argument("--start", required=True, help="Start month in YYYY-MM")
    parser.add_argument("--end", required=True, help="End month in YYYY-MM")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output raster directory")
    parser.add_argument("--row-chunk", type=int, default=64, help="Row block size for tiled prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, imputer, feature_cols = fit_model(args.scale)
    datasets, cmfd_lons, cmfd_lats, times = open_cmfd()
    climatology = {name: load_climatology(args.scale, name) for name in CMFD_FILES}
    start_month = parse_month(args.start)
    end_month = parse_month(args.end)
    target_times = [ts for ts in times if start_month <= ts <= end_month and ts >= times[args.scale - 1]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    static_handles = {name: rasterio.open(path) for name, path in STATIC_RASTERS.items()}
    ref = static_handles["soil_clay_0_5cm"]

    lon0 = float(cmfd_lons[0])
    lat0 = float(cmfd_lats[0])
    step_lon = float(cmfd_lons[1] - cmfd_lons[0])
    step_lat = float(cmfd_lats[1] - cmfd_lats[0])

    meta = ref.meta.copy()
    meta.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw", tiled=True)

    try:
        for ts in target_times:
            time_index = int(np.where(times == ts)[0][0])
            dynamic_arrays = get_dynamic_arrays(datasets, time_index, args.scale, climatology)
            out_path = args.out_dir / f"spei_{args.scale:02d}mo_{ts.strftime('%Y-%m')}.tif"
            with rasterio.open(out_path, "w", **meta) as dst:
                for row0 in range(0, ref.height, args.row_chunk):
                    row1 = min(row0 + args.row_chunk, ref.height)
                    window = Window(0, row0, ref.width, row1 - row0)

                    static_block: dict[str, np.ndarray] = {}
                    for name, src in static_handles.items():
                        arr = src.read(1, window=window).astype("float32")
                        arr[np.isclose(arr, src.nodata)] = np.nan
                        static_block[name] = arr

                    valid_mask = np.isfinite(static_block["soil_clay_0_5cm"])
                    out_arr = np.full((row1 - row0, ref.width), -9999.0, dtype="float32")
                    if not valid_mask.any():
                        dst.write(out_arr, 1, window=window)
                        continue

                    lon, lat = block_lon_lat(ref, row0, row1)
                    lon_idx = np.clip(np.rint((lon - lon0) / step_lon).astype(int), 0, len(cmfd_lons) - 1)
                    lat_idx = np.clip(np.rint((lat - lat0) / step_lat).astype(int), 0, len(cmfd_lats) - 1)

                    features: dict[str, np.ndarray] = {}
                    features["year"] = np.full(valid_mask.sum(), ts.year, dtype="float32")
                    features["month"] = np.full(valid_mask.sum(), ts.month, dtype="float32")
                    features["month_sin"] = np.full(valid_mask.sum(), np.sin(2 * np.pi * ts.month / 12.0), dtype="float32")
                    features["month_cos"] = np.full(valid_mask.sum(), np.cos(2 * np.pi * ts.month / 12.0), dtype="float32")
                    features["lat"] = lat[valid_mask]
                    features["lon"] = lon[valid_mask]
                    features["station_lat"] = lat[valid_mask]
                    features["station_lon"] = lon[valid_mask]
                    features["cmfd_grid_lon"] = cmfd_lons[lon_idx[valid_mask]]
                    features["cmfd_grid_lat"] = cmfd_lats[lat_idx[valid_mask]]

                    elevation = static_block["dem_elev_m"][valid_mask]
                    features["Elevation(m)"] = elevation
                    features["station_elevation_m"] = elevation

                    for name, arr in static_block.items():
                        features[name] = arr[valid_mask]
                    for name, arr in dynamic_arrays.items():
                        features[name] = arr[lat_idx[valid_mask], lon_idx[valid_mask]]

                    frame = pd.DataFrame(
                        {
                            col: features.get(col, np.full(valid_mask.sum(), np.nan, dtype="float32"))
                            for col in feature_cols
                        }
                    )
                    x_pred = pd.DataFrame(imputer.transform(frame), columns=feature_cols)
                    pred = model.predict(x_pred).astype("float32")
                    out_arr[valid_mask] = pred
                    dst.write(out_arr, 1, window=window)

            print(f"Wrote {out_path}")
    finally:
        for ds in datasets.values():
            ds.close()
        for src in static_handles.values():
            src.close()


if __name__ == "__main__":
    main()
