from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCALE_DIR_SUFFIX = "monthscale"


def normalize_lookup_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for source in ["station_id", "StationID", "station_ID", "ID"]:
        if source in df.columns:
            rename_map[source] = "station_id"
    for source in ["lon", "longitude", "lontitude(Decimal Degrees)", "缁忓害"]:
        if source in df.columns:
            rename_map[source] = "lon"
    for source in ["lat", "latitude", "latitude(Decimal Degrees)", "绾害"]:
        if source in df.columns:
            rename_map[source] = "lat"
    return df.rename(columns=rename_map)


def read_station_daily_spei(file_path: Path, scale: int) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None, names=["year", "doy", "spei"])
    df["station_id"] = file_path.stem
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + df["doy"].astype(int).astype(str).str.zfill(3),
        format="%Y%j",
    )
    month_end = df.groupby([df["date"].dt.year, df["date"].dt.month], as_index=False).tail(1).copy()
    month_end["year"] = month_end["date"].dt.year
    month_end["month"] = month_end["date"].dt.month
    month_end["month_end_date"] = month_end["date"].dt.strftime("%Y-%m-%d")
    month_end["scale"] = scale
    return month_end[["station_id", "year", "month", "month_end_date", "scale", "spei"]]


def detect_scale(scale_dir: Path) -> int:
    return int(scale_dir.name.split("_")[-1].replace(SCALE_DIR_SUFFIX, ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract month-end SPEI labels from daily station files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing daily_spei_GEV_*monthscale folders")
    parser.add_argument("lookup_csv", type=Path, help="Station lookup table")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("--glob", default="*.csv", help="Input file glob inside each scale folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale_dirs = sorted(
        path for path in args.input_dir.iterdir() if path.is_dir() and path.name.endswith(SCALE_DIR_SUFFIX)
    )
    if not scale_dirs:
        raise FileNotFoundError(f"No scale folders found in {args.input_dir}")

    frames: list[pd.DataFrame] = []
    for scale_dir in scale_dirs:
        scale = detect_scale(scale_dir)
        for file_path in sorted(scale_dir.glob(args.glob)):
            frames.append(read_station_daily_spei(file_path, scale))

    result = pd.concat(frames, ignore_index=True)
    lookup = normalize_lookup_columns(pd.read_csv(args.lookup_csv))
    if "station_id" in lookup.columns:
        result["station_id"] = result["station_id"].astype(str)
        lookup["station_id"] = lookup["station_id"].astype(str)
        result = result.merge(lookup, on="station_id", how="left")

    result = result.sort_values(["scale", "station_id", "year", "month"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
