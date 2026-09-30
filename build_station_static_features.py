from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


TERRAIN_RASTERS = {
    "dem_elev_m": "dem_elev_m.tif",
    "dem_relief_1km": "dem_relief_1km.tif",
    "dem_std_1km": "dem_std_1km.tif",
}


def detect_station_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    id_col = next((name for name in ["station_id", "StationID", "station_ID", "ID", "id"] if name in df), None)
    lon_col = next((name for name in ["lon", "longitude", "lontitude(Decimal Degrees)"] if name in df), None)
    lat_col = next((name for name in ["lat", "latitude", "latitude(Decimal Degrees)"] if name in df), None)
    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"Could not detect station columns in {list(df.columns)}")
    return id_col, lon_col, lat_col


def nearest_valid_value(
    source: rasterio.DatasetReader,
    row_index: int,
    column_index: int,
) -> tuple[float, float]:
    for radius in [0, 2, 5, 10, 25, 50, 100, 200]:
        row0 = max(row_index - radius, 0)
        row1 = min(row_index + radius + 1, source.height)
        col0 = max(column_index - radius, 0)
        col1 = min(column_index + radius + 1, source.width)
        if row0 >= row1 or col0 >= col1:
            continue
        array = source.read(1, window=((row0, row1), (col0, col1))).astype(float)
        valid = np.isfinite(array)
        if source.nodata is not None:
            valid &= ~np.isclose(array, source.nodata)
        if not valid.any():
            continue
        yy, xx = np.indices(array.shape)
        distance_sq = (yy - (row_index - row0)) ** 2 + (xx - (column_index - col0)) ** 2
        distance_sq = np.where(valid, distance_sq, np.inf)
        nearest = np.unravel_index(np.argmin(distance_sq), array.shape)
        return float(array[nearest]), float(np.sqrt(distance_sq[nearest]))
    return float("nan"), float("nan")


def sample_terrain(
    stations: pd.DataFrame,
    static_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {name: static_dir / filename for name, filename in TERRAIN_RASTERS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing canonical terrain rasters: {missing}")

    output = stations.copy()
    audit_rows: list[dict[str, object]] = []
    reference_geometry: tuple[object, ...] | None = None
    for feature, path in paths.items():
        with rasterio.open(path) as source:
            geometry = (source.width, source.height, source.crs, source.transform)
            if reference_geometry is None:
                reference_geometry = geometry
            elif geometry != reference_geometry:
                raise ValueError(f"Canonical terrain raster is not aligned: {path}")
            xs, ys = transform(
                "EPSG:4326",
                source.crs,
                stations["lon"].astype(float).tolist(),
                stations["lat"].astype(float).tolist(),
            )
            values: list[float] = []
            distances: list[float] = []
            for x, y in zip(xs, ys, strict=True):
                row_index, column_index = source.index(x, y)
                value, distance = nearest_valid_value(source, row_index, column_index)
                values.append(value)
                distances.append(distance)
            output[feature] = values
            audit_rows.extend(
                {
                    "station_id": station_id,
                    "feature": feature,
                    "fallback_distance_pixels": distance,
                    "used_fallback": bool(np.isfinite(distance) and distance > 0),
                    "missing_after_fallback": bool(not np.isfinite(value)),
                }
                for station_id, value, distance in zip(
                    stations["station_id"], values, distances, strict=True
                )
            )
    return output, pd.DataFrame(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the canonical 1 km NASADEM terrain rasters at station locations."
    )
    parser.add_argument("station_csv", type=Path)
    parser.add_argument("static_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.station_csv)
    id_col, lon_col, lat_col = detect_station_columns(source)
    stations = source.rename(
        columns={id_col: "station_id", lon_col: "lon", lat_col: "lat"}
    )[["station_id", "lon", "lat"]].copy()
    stations["station_id"] = stations["station_id"].astype(str)
    output, audit = sample_terrain(stations, args.static_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    audit_path = args.audit_csv or args.output_csv.with_name(f"{args.output_csv.stem}_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"Wrote {len(output)} station terrain rows to {args.output_csv}")


if __name__ == "__main__":
    main()
