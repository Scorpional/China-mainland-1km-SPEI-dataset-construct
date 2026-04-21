from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


def detect_station_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    id_col = next((c for c in ["station_id", "StationID", "station_ID", "ID", "id"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "lontitude(Decimal Degrees)", "缁忓害"] if c in df.columns), None)
    lat_col = next((c for c in ["lat", "latitude", "latitude(Decimal Degrees)", "绾害"] if c in df.columns), None)
    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"Could not detect station id/lon/lat columns in {list(df.columns)}")
    return id_col, lon_col, lat_col


def select_data_var(dataset: xr.Dataset) -> str:
    for name in dataset.data_vars:
        if {"time", "lat", "lon"} & set(dataset[name].dims):
            return name
    raise ValueError(f"No usable data variable found in {list(dataset.data_vars)}")


def sample_dataset(nc_path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    dataset = xr.open_dataset(nc_path)
    var_name = select_data_var(dataset)
    data_var = dataset[var_name]

    rows: list[pd.DataFrame] = []
    for station in stations.itertuples(index=False):
        series = (
            data_var.sel(lon=float(station.lon), lat=float(station.lat), method="nearest")
            .to_series()
            .reset_index()
            .rename(columns={var_name: var_name, "time": "time"})
        )
        series["station_id"] = station.station_id
        rows.append(series[["station_id", "time", var_name]])

    sampled = pd.concat(rows, ignore_index=True)
    sampled["time"] = pd.to_datetime(sampled["time"])
    sampled["year"] = sampled["time"].dt.year
    sampled["month"] = sampled["time"].dt.month
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample monthly CMFD NetCDF variables at station locations.")
    parser.add_argument("station_csv", type=Path, help="Station table with coordinates")
    parser.add_argument("cmfd_dir", type=Path, help="Directory containing monthly CMFD NetCDF files")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("--glob", default="*.nc", help="Input NetCDF glob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations_raw = pd.read_csv(args.station_csv)
    id_col, lon_col, lat_col = detect_station_columns(stations_raw)
    stations = stations_raw.rename(columns={id_col: "station_id", lon_col: "lon", lat_col: "lat"})[
        ["station_id", "lon", "lat"]
    ].copy()
    stations["station_id"] = stations["station_id"].astype(str)

    files = sorted(args.cmfd_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {args.cmfd_dir}")

    merged: pd.DataFrame | None = None
    for nc_path in files:
        sampled = sample_dataset(nc_path, stations)
        if merged is None:
            merged = sampled
        else:
            merged = merged.merge(sampled, on=["station_id", "time", "year", "month"], how="outer")

    if merged is None:
        raise RuntimeError("No CMFD samples were generated")

    merged = merged.merge(stations, on="station_id", how="left")
    merged = merged.sort_values(["station_id", "time"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(merged)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
