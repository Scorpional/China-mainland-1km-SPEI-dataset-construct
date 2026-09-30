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
from scipy.ndimage import distance_transform_edt
from sklearn.impute import SimpleImputer

from cmfd_monthly_units import convert_cmfd_monthly_values
from cmfd_spatial_sampling import prepare_regular_grid_sampling, sample_regular_grid
from model_config import MODEL_PARAMS, select_production_features


CMFD_VARIABLES = ("lrad", "prec", "pres", "shum", "srad", "temp", "wind")
TERRAIN_FILES = {
    "dem_elev_m": "dem_elev_m.tif",
    "dem_relief_1km": "dem_relief_1km.tif",
    "dem_std_1km": "dem_std_1km.tif",
}
LAND_MASK_FILE = "china_land_mask_1km.tif"
CMFD_TIMES = pd.date_range("1979-01-01", periods=480, freq="MS")


def resolve_cmfd_files(cmfd_dir: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for variable in CMFD_VARIABLES:
        matches = sorted(cmfd_dir.glob(f"{variable}_*.nc"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one NetCDF for {variable} in {cmfd_dir}, found {matches}"
            )
        output[variable] = matches[0]
    return output


def fit_model(training_csv: Path) -> tuple[LGBMRegressor, SimpleImputer, list[str]]:
    frame = pd.read_csv(training_csv, dtype={"station_id": str})
    if "spei" not in frame:
        raise ValueError(f"Missing target column 'spei' in {training_csv}")
    feature_columns = select_production_features(frame)
    imputer = SimpleImputer(strategy="median")
    predictors = imputer.fit_transform(frame[feature_columns])
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(predictors, frame["spei"].to_numpy())
    return model, imputer, feature_columns


def open_cmfd(files: dict[str, Path]) -> tuple[dict[str, Dataset], np.ndarray, np.ndarray]:
    datasets = {name: Dataset(path) for name, path in files.items()}
    lons = np.asarray(datasets["prec"].variables["lon"][:], dtype=np.float32)
    lats = np.asarray(datasets["prec"].variables["lat"][:], dtype=np.float32)
    for name, dataset in datasets.items():
        if dataset.variables[name].shape[0] != len(CMFD_TIMES):
            raise ValueError(f"Unexpected CMFD time dimension for {name}")
    return datasets, lons, lats


def rolling_sum(array: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(array, axis=0, dtype=np.float32)
    output = np.empty_like(array, dtype=np.float32)
    output[: window - 1] = np.nan
    output[window - 1] = cumulative[window - 1]
    output[window:] = cumulative[window:] - cumulative[:-window]
    return output


def load_climatology(
    scale: int,
    variable: str,
    cmfd_file: Path,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"scale_{scale:02d}_{variable}_climatology.npz"
    if cache.exists():
        stored = np.load(cache)
        return {name: stored[name] for name in stored.files}

    with Dataset(cmfd_file) as dataset:
        values = convert_cmfd_monthly_values(
            variable, dataset.variables[variable][:], CMFD_TIMES
        )
    current = np.stack(
        [np.nanmean(values[month::12], axis=0) for month in range(12)], axis=0
    ).astype(np.float32)
    rolling = rolling_sum(values, scale)
    rolling_mean = rolling / float(scale)
    rolling_mean_climatology = np.stack(
        [
            np.nanmean(
                rolling_mean[
                    np.asarray(
                        [index for index in range(month, len(CMFD_TIMES), 12) if index >= scale - 1]
                    )
                ],
                axis=0,
            )
            for month in range(12)
        ],
        axis=0,
    ).astype(np.float32)
    payload = {"current": current, "roll_mean": rolling_mean_climatology}
    if variable == "prec":
        payload["roll_sum"] = np.stack(
            [
                np.nanmean(
                    rolling[
                        np.asarray(
                            [index for index in range(month, len(CMFD_TIMES), 12) if index >= scale - 1]
                        )
                    ],
                    axis=0,
                )
                for month in range(12)
            ],
            axis=0,
        ).astype(np.float32)
    np.savez_compressed(cache, **payload)
    return payload


def month_arrays(
    datasets: dict[str, Dataset],
    time_index: int,
    scale: int,
    climatology: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    month_index = time_index % 12
    output: dict[str, np.ndarray] = {}
    for variable, dataset in datasets.items():
        source = dataset.variables[variable]
        current = convert_cmfd_monthly_values(
            variable, source[time_index], CMFD_TIMES[time_index]
        )
        output[f"dyn_{variable}"] = current
        output[f"clim_dyn_{variable}"] = climatology[variable]["current"][month_index]
        output[f"anom_dyn_{variable}"] = current - output[f"clim_dyn_{variable}"]

        for lag in range(1, scale):
            output[f"dyn_{variable}_lag{lag}"] = convert_cmfd_monthly_values(
                variable,
                source[time_index - lag],
                CMFD_TIMES[time_index - lag],
            )

        window = convert_cmfd_monthly_values(
            variable,
            source[time_index - scale + 1 : time_index + 1],
            CMFD_TIMES[time_index - scale + 1 : time_index + 1],
        )
        rolling_mean = window.mean(axis=0).astype(np.float32)
        mean_name = f"dyn_{variable}_roll{scale}_mean"
        output[mean_name] = rolling_mean
        output[f"clim_{mean_name}"] = climatology[variable]["roll_mean"][month_index]
        output[f"anom_{mean_name}"] = rolling_mean - output[f"clim_{mean_name}"]
        if variable == "prec":
            rolling_total = window.sum(axis=0).astype(np.float32)
            sum_name = f"dyn_{variable}_roll{scale}_sum"
            output[sum_name] = rolling_total
            output[f"clim_{sum_name}"] = climatology[variable]["roll_sum"][month_index]
            output[f"anom_{sum_name}"] = rolling_total - output[f"clim_{sum_name}"]
    return output


def nearest_valid_grid(array: np.ndarray) -> np.ndarray:
    values = np.ma.filled(np.ma.asarray(array, dtype=np.float32), np.nan)
    valid = np.isfinite(values)
    if valid.all() or not valid.any():
        return values
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def block_coordinates(
    reference: rasterio.DatasetReader,
    row_start: int,
    row_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(row_start, row_stop)
    columns = np.arange(reference.width)
    xs = reference.transform.c + (columns + 0.5) * reference.transform.a
    ys = reference.transform.f + (rows + 0.5) * reference.transform.e
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = transform(
        reference.crs, "EPSG:4326", xx.ravel().tolist(), yy.ravel().tolist()
    )
    shape = xx.shape
    return np.asarray(lon, dtype=np.float32).reshape(shape), np.asarray(lat, dtype=np.float32).reshape(shape)


def assert_alignment(handles: dict[str, rasterio.DatasetReader]) -> None:
    items = list(handles.items())
    reference_name, reference = items[0]
    geometry = (reference.width, reference.height, reference.crs, reference.transform)
    for name, source in items[1:]:
        if (source.width, source.height, source.crs, source.transform) != geometry:
            raise ValueError(f"Raster {name} is not aligned with {reference_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the final no-soil LightGBM model and reconstruct monthly 1 km SPEI grids."
    )
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("cmfd_dir", type=Path)
    parser.add_argument("static_dir", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--scale", type=int, required=True, choices=[1, 3, 6, 12, 24])
    parser.add_argument("--start", required=True, help="First requested month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last requested month, YYYY-MM")
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, imputer, feature_columns = fit_model(args.training_csv)
    if any(name.startswith("soil_") for name in feature_columns):
        raise AssertionError("Soil predictors must not enter the version 2 production model")

    cmfd_files = resolve_cmfd_files(args.cmfd_dir)
    datasets, cmfd_lons, cmfd_lats = open_cmfd(cmfd_files)
    climatology = {
        variable: load_climatology(
            args.scale, variable, cmfd_files[variable], args.cache_dir
        )
        for variable in CMFD_VARIABLES
    }

    terrain_paths = {
        name: args.static_dir / filename for name, filename in TERRAIN_FILES.items()
    }
    mask_path = args.static_dir / LAND_MASK_FILE
    missing = [str(path) for path in [*terrain_paths.values(), mask_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing canonical 1 km rasters: {missing}")
    terrain = {name: rasterio.open(path) for name, path in terrain_paths.items()}
    mask = rasterio.open(mask_path)
    assert_alignment({**terrain, "land_mask": mask})

    start = pd.Timestamp(f"{args.start}-01")
    end = pd.Timestamp(f"{args.end}-01")
    target_times = [
        timestamp
        for timestamp in CMFD_TIMES
        if start <= timestamp <= end and timestamp >= CMFD_TIMES[args.scale - 1]
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = mask.meta.copy()
    metadata.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw", tiled=True)

    try:
        for timestamp in target_times:
            output_path = args.out_dir / f"spei_{args.scale:02d}mo_{timestamp:%Y-%m}.tif"
            if output_path.exists():
                if not args.skip_existing:
                    raise FileExistsError(f"Refusing to overwrite {output_path}")
                with rasterio.open(output_path) as existing:
                    if existing.tags().get("generation_complete") == "true":
                        print(f"Skipped {output_path}", flush=True)
                        continue
                raise RuntimeError(f"Existing raster is not marked complete: {output_path}")

            time_index = int(np.where(CMFD_TIMES == timestamp)[0][0])
            dynamic = month_arrays(
                datasets, time_index, args.scale, climatology
            )
            dynamic_fallback = {
                name: nearest_valid_grid(values) for name, values in dynamic.items()
            }

            with rasterio.open(output_path, "w", **metadata) as destination:
                for row_start in range(0, mask.height, args.row_chunk):
                    row_stop = min(row_start + args.row_chunk, mask.height)
                    window = Window(0, row_start, mask.width, row_stop - row_start)
                    valid = mask.read(1, window=window) == 1
                    output = np.full(valid.shape, -9999.0, dtype=np.float32)
                    if not valid.any():
                        destination.write(output, 1, window=window)
                        continue

                    lon, lat = block_coordinates(mask, row_start, row_stop)
                    sample_lon = lon[valid]
                    sample_lat = lat[valid]
                    sampling_plan = prepare_regular_grid_sampling(
                        sample_lon, sample_lat, cmfd_lons, cmfd_lats
                    )
                    features: dict[str, np.ndarray] = {
                        "year": np.full(valid.sum(), timestamp.year, dtype=np.float32),
                        "month": np.full(valid.sum(), timestamp.month, dtype=np.float32),
                        "month_sin": np.full(valid.sum(), np.sin(2 * np.pi * timestamp.month / 12.0), dtype=np.float32),
                        "month_cos": np.full(valid.sum(), np.cos(2 * np.pi * timestamp.month / 12.0), dtype=np.float32),
                        "lon": sample_lon,
                        "lat": sample_lat,
                    }
                    for name, source in terrain.items():
                        array = source.read(1, window=window).astype(np.float32)
                        if source.nodata is not None:
                            array[np.isclose(array, source.nodata)] = np.nan
                        features[name] = array[valid]
                    for name, values in dynamic.items():
                        features[name] = sample_regular_grid(
                            values,
                            sample_lon,
                            sample_lat,
                            cmfd_lons,
                            cmfd_lats,
                            fallback_array=dynamic_fallback[name],
                            sampling_plan=sampling_plan,
                        )

                    missing_features = [name for name in feature_columns if name not in features]
                    if missing_features:
                        raise KeyError(f"Inference is missing training features: {missing_features}")
                    frame = pd.DataFrame({name: features[name] for name in feature_columns})
                    predictors = imputer.transform(frame)
                    prediction = model.predict(
                        predictors, num_threads=args.prediction_threads
                    ).astype(np.float32)
                    output[valid] = prediction
                    destination.write(output, 1, window=window)
                destination.update_tags(
                    generation_complete="true",
                    product_version="2",
                    model="LightGBM_Leaf63",
                    cmfd_spatial_method="bilinear",
                    soil_predictors="excluded",
                    spei_timescale_months=str(args.scale),
                    year_month=timestamp.strftime("%Y-%m"),
                )
            print(f"Wrote {output_path}", flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()
        for source in terrain.values():
            source.close()
        mask.close()


if __name__ == "__main__":
    main()
