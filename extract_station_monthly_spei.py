from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def read_station_daily_spei(
    file_path: Path,
    scale: int,
    monthly_rule: str = "strict_calendar_end",
    calendar: str = "noleap_365",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path, header=None, names=["year", "doy", "spei"])
    df["station_id"] = file_path.stem
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
    df["spei"] = pd.to_numeric(df["spei"], errors="coerce")
    if calendar == "gregorian":
        df["date"] = pd.to_datetime(
            df["year"].astype("Int64").astype(str) + df["doy"].astype("Int64").astype(str).str.zfill(3),
            format="%Y%j",
            errors="coerce",
        )
    elif calendar == "noleap_365":
        valid_doy = df["doy"].between(1, 365)
        template_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        template_date.loc[valid_doy] = pd.Timestamp("2001-01-01") + pd.to_timedelta(
            df.loc[valid_doy, "doy"] - 1,
            unit="D",
        )
        df["date"] = pd.to_datetime(
            {
                "year": df["year"],
                "month": template_date.dt.month,
                "day": template_date.dt.day,
            },
            errors="coerce",
        )
    else:
        raise ValueError(f"Unsupported source calendar: {calendar}")
    invalid_date_rows = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).sort_values("date").copy()
    df["period"] = df["date"].dt.to_period("M")

    if monthly_rule not in {"strict_calendar_end", "last_record"}:
        raise ValueError(f"Unsupported monthly rule: {monthly_rule}")

    if calendar == "noleap_365":
        noleap_month_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        df["expected_month_end"] = pd.to_datetime(
            {
                "year": df["date"].dt.year,
                "month": df["date"].dt.month,
                "day": df["date"].dt.month.map(noleap_month_days),
            }
        )
    else:
        df["expected_month_end"] = df["period"].dt.to_timestamp(how="end").dt.normalize()
    df["is_month_end"] = df["date"] == df["expected_month_end"]
    df["is_duplicate_date_row"] = df.duplicated(subset=["date"], keep=False)

    grouped = df.groupby("period", sort=True)
    stats = grouped.agg(
        n_daily_records=("date", "size"),
        n_month_end_records=("is_month_end", "sum"),
        n_duplicate_date_rows=("is_duplicate_date_row", "sum"),
    ).reset_index()
    last = grouped.tail(1)[["period", "date", "spei", "expected_month_end"]].rename(
        columns={"date": "last_record_date", "spei": "last_record_spei"}
    )
    strict = (
        df[df["is_month_end"]]
        .groupby("period", sort=True)
        .tail(1)[["period", "date", "spei"]]
        .rename(columns={"date": "strict_source_date", "spei": "strict_month_end_spei"})
    )
    audit = stats.merge(last, on="period", how="left").merge(strict, on="period", how="left")
    audit["station_id"] = file_path.stem
    audit["scale"] = scale
    audit["source_calendar"] = calendar
    audit["year"] = audit["period"].dt.year.astype(int)
    audit["month"] = audit["period"].dt.month.astype(int)
    audit["last_record_is_month_end"] = audit["last_record_date"] == audit["expected_month_end"]
    audit["has_month_end_record"] = audit["n_month_end_records"] > 0
    audit["month_end_spei_valid"] = audit["strict_month_end_spei"].notna()
    audit["invalid_date_rows_in_file"] = invalid_date_rows

    if monthly_rule == "strict_calendar_end":
        source_date = audit["strict_source_date"]
        spei = audit["strict_month_end_spei"]
    else:
        source_date = audit["last_record_date"]
        spei = audit["last_record_spei"]
    labels = pd.DataFrame(
        {
            "station_id": file_path.stem,
            "year": audit["year"],
            "month": audit["month"],
            "month_end_date": audit["expected_month_end"].dt.strftime("%Y-%m-%d"),
            "source_date": source_date.dt.strftime("%Y-%m-%d").fillna(""),
            "scale": scale,
            "spei": spei,
        }
    )
    audit["expected_month_end"] = audit["expected_month_end"].dt.strftime("%Y-%m-%d")
    audit["last_record_date"] = audit["last_record_date"].dt.strftime("%Y-%m-%d")
    audit["strict_source_date"] = audit["strict_source_date"].dt.strftime("%Y-%m-%d")
    audit = audit[
        [
            "station_id",
            "scale",
            "source_calendar",
            "year",
            "month",
            "expected_month_end",
            "last_record_date",
            "last_record_is_month_end",
            "has_month_end_record",
            "month_end_spei_valid",
            "last_record_spei",
            "strict_month_end_spei",
            "n_daily_records",
            "n_month_end_records",
            "n_duplicate_date_rows",
            "invalid_date_rows_in_file",
        ]
    ]
    return labels, audit


def detect_scale(scale_dir: Path) -> int:
    return int(scale_dir.name.split("_")[-1].replace(SCALE_DIR_SUFFIX, ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract month-end SPEI labels from daily station files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing daily_spei_GEV_*monthscale folders")
    parser.add_argument("lookup_csv", type=Path, help="Station lookup table")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("--glob", default="*.csv", help="Input file glob inside each scale folder")
    parser.add_argument(
        "--monthly-rule",
        choices=["strict_calendar_end", "last_record"],
        default="strict_calendar_end",
        help="Use the fixed calendar month end (default) or reproduce the legacy last-record rule",
    )
    parser.add_argument(
        "--calendar",
        choices=["noleap_365", "gregorian"],
        default="noleap_365",
        help="Calendar used by the daily source files; the supplied station dataset contains 365 DOY records per year",
    )
    parser.add_argument("--audit-csv", type=Path, help="Optional station-month audit output")
    parser.add_argument("--summary-csv", type=Path, help="Optional scale-level audit summary")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale_dirs = sorted(
        path for path in args.input_dir.iterdir() if path.is_dir() and path.name.endswith(SCALE_DIR_SUFFIX)
    )
    if not scale_dirs:
        raise FileNotFoundError(f"No scale folders found in {args.input_dir}")

    frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for scale_dir in scale_dirs:
        scale = detect_scale(scale_dir)
        for file_path in sorted(scale_dir.glob(args.glob)):
            if file_path.suffix.lower() != ".csv":
                continue
            labels, audit = read_station_daily_spei(file_path, scale, args.monthly_rule, args.calendar)
            frames.append(labels)
            audit_frames.append(audit)

    result = pd.concat(frames, ignore_index=True)
    audit = pd.concat(audit_frames, ignore_index=True)
    if args.start_year is not None:
        result = result[result["year"] >= args.start_year].copy()
        audit = audit[audit["year"] >= args.start_year].copy()
    if args.end_year is not None:
        result = result[result["year"] <= args.end_year].copy()
        audit = audit[audit["year"] <= args.end_year].copy()
    lookup = normalize_lookup_columns(pd.read_csv(args.lookup_csv))
    if "station_id" in lookup.columns:
        result["station_id"] = result["station_id"].astype(str)
        lookup["station_id"] = lookup["station_id"].astype(str)
        result = result.merge(lookup, on="station_id", how="left")

    result = result.sort_values(["scale", "station_id", "year", "month"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    audit_csv = args.audit_csv or args.output_csv.with_name(f"{args.output_csv.stem}_audit.csv")
    summary_csv = args.summary_csv or args.output_csv.with_name(f"{args.output_csv.stem}_audit_summary.csv")
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv, index=False)

    summary = (
        audit.groupby("scale", as_index=False)
        .agg(
            n_station_months=("station_id", "size"),
            n_stations=("station_id", "nunique"),
            n_last_record_not_month_end=("last_record_is_month_end", lambda s: int((~s).sum())),
            n_missing_month_end_record=("has_month_end_record", lambda s: int((~s).sum())),
            n_invalid_month_end_spei=("month_end_spei_valid", lambda s: int((~s).sum())),
            n_duplicate_date_rows=("n_duplicate_date_rows", "sum"),
        )
        .sort_values("scale")
    )
    summary["n_valid_strict_labels"] = summary["n_station_months"] - summary["n_invalid_month_end_spei"]
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {len(result)} rows to {args.output_csv}")
    print(f"Wrote audit rows to {audit_csv}")
    print(f"Wrote audit summary to {summary_csv}")


if __name__ == "__main__":
    main()
