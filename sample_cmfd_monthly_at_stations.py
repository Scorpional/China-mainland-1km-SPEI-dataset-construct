from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

from cmfd_monthly_units import convert_cmfd_monthly_values
from cmfd_spatial_sampling import sample_regular_grid


def detect_station_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    id_col = next((name for name in ["station_id", "StationID", "station_ID", "ID", "id"] if name in df), None)
    lon_col = next((name for name in ["lon", "longitude", "lontitude(Decimal Degrees)"] if name in df), None)
    lat_col = next((name for name in ["lat", "latitude", "latitude(Decimal Degrees)"] if name in df), None)
    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"Could not detect station columns in {list(df.columns)}")
    return id_col, lon_col, lat_col


def nearest_valid_grid(array: np.ndarray) -> np.ndarray:
    values = np.ma.filled(np.ma.asarray(array, dtype=np.float32), np.nan)
    valid = np.isfinite(values)
    if valid.all() or not valid.any():
        return values
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def select_data_variable(dataset: xr.Dataset) -> str:
    candidates = [
        name
        for name, variable in dataset.data_vars.items()
        if {"time", "lat", "lon"}.issubset(variable.dims)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one time-lat-lon variable, found {candidates}")
    return candidates[0]


def sample_file(path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    with xr.open_dataset(path) as dataset:
        variable_name = select_data_variable(dataset)
        variable = dataset[variable_name].transpose("time", "lat", "lon")
        values = np.ma.filled(np.ma.asarray(variable.values, dtype=np.float32), np.nan)
        lons = np.asarray(dataset["lon"].values, dtype=float)
        lats = np.asarray(dataset["lat"].values, dtype=float)
        sample_lons = stations["lon"].to_numpy(dtype=float)
        sample_lats = stations["lat"].to_numpy(dtype=float)
        fallback = np.stack([nearest_valid_grid(layer) for layer in values], axis=0)
        sampled = sample_regular_grid(
            values,
            sample_lons,
            sample_lats,
            lons,
            lats,
            fallback_array=fallback,
        )
        times = pd.DatetimeIndex(pd.to_datetime(dataset["time"].values))
        sampled = convert_cmfd_monthly_values(variable_name, sampled, times)

    frame = pd.DataFrame(
        {
            "station_id": np.tile(stations["station_id"].to_numpy(), len(times)),
            "time": np.repeat(times.to_numpy(), len(stations)),
            variable_name: sampled.reshape(-1),
        }
    )
    frame["year"] = pd.DatetimeIndex(frame["time"]).year
    frame["month"] = pd.DatetimeIndex(frame["time"]).month
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample monthly CMFD fields at stations using NaN-aware bilinear interpolation."
    )
    parser.add_argument("station_csv", type=Path)
    parser.add_argument("cmfd_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--glob", default="*.nc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.station_csv)
    id_col, lon_col, lat_col = detect_station_columns(source)
    stations = source.rename(
        columns={id_col: "station_id", lon_col: "lon", lat_col: "lat"}
    )[["station_id", "lon", "lat"]].copy()
    stations["station_id"] = stations["station_id"].astype(str)

    files = sorted(args.cmfd_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No CMFD NetCDF files found in {args.cmfd_dir}")
    merged: pd.DataFrame | None = None
    for path in files:
        sampled = sample_file(path, stations)
        keys = ["station_id", "time", "year", "month"]
        merged = sampled if merged is None else merged.merge(sampled, on=keys, how="outer", validate="one_to_one")
    if merged is None:
        raise RuntimeError("No CMFD station features were generated")
    merged = merged.merge(stations, on="station_id", how="left", validate="many_to_one")
    merged = merged.sort_values(["station_id", "time"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(merged)} station-month rows to {args.output_csv}")


if __name__ == "__main__":
    main()
