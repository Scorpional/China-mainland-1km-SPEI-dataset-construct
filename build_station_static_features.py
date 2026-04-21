from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


def detect_station_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    id_col = next((c for c in ["station_id", "station_ID", "StationID", "ID", "id"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "LONGITUDE", "lontitude(Decimal Degrees)"] if c in df.columns), None)
    lat_col = next((c for c in ["lat", "latitude", "LATITUDE", "latitude(Decimal Degrees)"] if c in df.columns), None)
    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(f"Could not detect station id/lon/lat columns in {list(df.columns)}")
    return id_col, lon_col, lat_col


def nasadem_tile_name(lon: float, lat: float) -> str:
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    lat_prefix = "n" if lat_floor >= 0 else "s"
    lon_prefix = "e" if lon_floor >= 0 else "w"
    return f"{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}"


def extract_hgt(tile: str, dem_dir: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    hgt_path = cache_dir / f"{tile}.hgt"
    if hgt_path.exists() and hgt_path.stat().st_size > 0:
        return hgt_path

    zip_path = dem_dir / f"NASADEM_HGT_{tile}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        hgt_path.write_bytes(zf.read(f"{tile}.hgt"))
    return hgt_path


def sample_dem_features(stations: pd.DataFrame, dem_dir: Path, cache_dir: Path, window_radius: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for tile, subset in stations.groupby("dem_tile", sort=True):
        hgt_path = extract_hgt(tile, dem_dir=dem_dir, cache_dir=cache_dir)
        with rasterio.open(hgt_path) as src:
            arr = src.read(1)
            nodata = src.nodata
            for station in subset.itertuples(index=False):
                row_idx, col_idx = src.index(float(station.station_lon), float(station.station_lat))
                elev = float("nan")
                relief = float("nan")
                std = float("nan")

                if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                    value = float(arr[row_idx, col_idx])
                    if nodata is None or not np.isclose(value, nodata):
                        elev = value

                    row0 = max(row_idx - window_radius, 0)
                    row1 = min(row_idx + window_radius + 1, src.height)
                    col0 = max(col_idx - window_radius, 0)
                    col1 = min(col_idx + window_radius + 1, src.width)
                    window = arr[row0:row1, col0:col1].astype(float)
                    if nodata is not None:
                        window = window[~np.isclose(window, nodata)]
                    if window.size:
                        relief = float(window.max() - window.min())
                        std = float(window.std())

                rows.append(
                    {
                        "station_id": str(station.station_id),
                        "dem_elev_m": elev,
                        "dem_relief_1km": relief,
                        "dem_std_1km": std,
                    }
                )

    return pd.DataFrame(rows)


def sample_soil_raster(raster_path: Path, stations: pd.DataFrame, search_radius: int) -> list[float]:
    with rasterio.open(raster_path) as src:
        xs, ys = transform("EPSG:4326", src.crs, stations["station_lon"].tolist(), stations["station_lat"].tolist())
        nodata = src.nodata
        out_values: list[float] = []

        for x, y in zip(xs, ys, strict=False):
            row_idx, col_idx = src.index(x, y)
            if row_idx < 0 or col_idx < 0 or row_idx >= src.height or col_idx >= src.width:
                out_values.append(float("nan"))
                continue

            row0 = max(row_idx - search_radius, 0)
            row1 = min(row_idx + search_radius + 1, src.height)
            col0 = max(col_idx - search_radius, 0)
            col1 = min(col_idx + search_radius + 1, src.width)
            window = src.read(1, window=((row0, row1), (col0, col1))).astype(float)

            valid_mask = np.isfinite(window)
            if nodata is not None:
                valid_mask &= ~np.isclose(window, nodata)

            center_y = row_idx - row0
            center_x = col_idx - col0
            if 0 <= center_y < window.shape[0] and 0 <= center_x < window.shape[1] and valid_mask[center_y, center_x]:
                out_values.append(float(window[center_y, center_x]))
                continue

            if not valid_mask.any():
                out_values.append(float("nan"))
                continue

            yy, xx = np.indices(window.shape)
            distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
            distance[~valid_mask] = distance.max() + 1
            nearest_distance = distance.min()
            nearest_values = window[(distance == nearest_distance) & valid_mask]
            out_values.append(float(nearest_values.mean()))

    return out_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build station static features from NASADEM and SoilGrids.")
    parser.add_argument("station_csv", type=Path, help="Station table with coordinates")
    parser.add_argument("dem_dir", type=Path, help="Directory containing NASADEM zip tiles")
    parser.add_argument("soil_dir", type=Path, help="Directory containing SoilGrids rasters")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("--window-radius", type=int, default=17, help="Neighborhood radius for terrain metrics")
    parser.add_argument("--soil-search-radius", type=int, default=2, help="Search radius for nearest valid soil pixel")
    parser.add_argument("--cache-dir", type=Path, default=Path("derived_data/cache/nasadem_hgt"), help="Cache directory for extracted HGT files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.station_csv)
    id_col, lon_col, lat_col = detect_station_columns(raw)

    stations = pd.DataFrame(
        {
            "station_id": raw[id_col].astype(str),
            "station_lat": raw[lat_col].astype(float),
            "station_lon": raw[lon_col].astype(float),
            "station_name": raw["station_name"] if "station_name" in raw.columns else "",
            "province": raw["Province"] if "Province" in raw.columns else raw.get("province", ""),
            "station_elevation_m": raw["Elevation(m)"] if "Elevation(m)" in raw.columns else raw.get("station_elevation_m", np.nan),
        }
    )
    stations["dem_tile"] = [nasadem_tile_name(lon, lat) for lon, lat in zip(stations["station_lon"], stations["station_lat"], strict=False)]

    dem_df = sample_dem_features(stations, dem_dir=args.dem_dir, cache_dir=args.cache_dir, window_radius=args.window_radius)
    output = stations.merge(dem_df, on="station_id", how="left").drop(columns=["dem_tile"])

    soil_files = {
        "bdod_0-5cm_mean_1000.tif": "soil_bdod_0_5cm",
        "clay_0-5cm_mean_1000.tif": "soil_clay_0_5cm",
        "phh2o_0-5cm_mean_1000.tif": "soil_phh2o_0_5cm",
        "sand_0-5cm_mean_1000.tif": "soil_sand_0_5cm",
        "silt_0-5cm_mean_1000.tif": "soil_silt_0_5cm",
        "soc_0-5cm_mean_1000.tif": "soil_soc_0_5cm",
    }
    for filename, out_col in soil_files.items():
        raster_path = args.soil_dir / filename
        if raster_path.exists():
            output[out_col] = sample_soil_raster(raster_path, output, search_radius=args.soil_search_radius)
        else:
            output[out_col] = float("nan")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
