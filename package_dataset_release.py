from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import rasterio


ROOT = Path(r"D:\GitRepository\bte")
PRED_ROOT = ROOT / "derived_data" / "grid_1km_china" / "predictions"
TABLE_ROOT = ROOT / "derived_data" / "manuscript_submission" / "tables"
FIG_ROOT = ROOT / "derived_data" / "manuscript_submission" / "figures"

PACKAGE_NAME = "China_1km_monthly_multiscale_SPEI_1979_2018_v1"
SCALES = ["spei01", "spei03", "spei06", "spei12", "spei24"]
SCALE_LABELS = {
    "spei01": "SPEI01",
    "spei03": "SPEI03",
    "spei06": "SPEI06",
    "spei12": "SPEI12",
    "spei24": "SPEI24",
}
SUMMARY_TABLES = {
    "Table1_main_randomkfold_metrics.csv": "validation_main_5fold_random.csv",
    "TableS1_grouped_validation_metrics.csv": "validation_grouped.csv",
    "TableS2_anomaly_vs_baseline_improvement.csv": "anomaly_vs_baseline_improvement.csv",
}
PREVIEW_FILES = {
    FIG_ROOT / "Figure2_station_map.jpg": "Figure2_station_map.jpg",
    FIG_ROOT / "Figure3_spatial_patterns.jpg": "Figure3_spatial_patterns.jpg",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_scale(scale_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    tif_files = sorted(scale_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF found in {scale_dir}")

    rows: list[dict[str, object]] = []
    ref_meta = None
    all_aligned = True
    for tif in tif_files:
        with rasterio.open(tif) as ds:
            transform = ds.transform
            meta = {
                "width": ds.width,
                "height": ds.height,
                "crs": str(ds.crs),
                "transform_a": transform.a,
                "transform_b": transform.b,
                "transform_c": transform.c,
                "transform_d": transform.d,
                "transform_e": transform.e,
                "transform_f": transform.f,
                "nodata": ds.nodata,
                "dtype": ds.dtypes[0],
            }
            if ref_meta is None:
                ref_meta = meta
            elif meta != ref_meta:
                all_aligned = False
            rows.append(
                {
                    "scale_dir": scale_dir.name,
                    "filename": tif.name,
                    "month": tif.stem.split("_")[-1],
                    "size_bytes": tif.stat().st_size,
                    **meta,
                }
            )

    summary = {
        "scale_dir": scale_dir.name,
        "n_files": len(tif_files),
        "first_file": tif_files[0].name,
        "last_file": tif_files[-1].name,
        "first_month": tif_files[0].stem.split("_")[-1],
        "last_month": tif_files[-1].stem.split("_")[-1],
        "total_size_gb": round(sum(row["size_bytes"] for row in rows) / (1024**3), 2),
        "all_aligned_within_scale": all_aligned,
        **{key: value for key, value in ref_meta.items()},
    }
    return rows, summary


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for scale in SCALES:
        rows, summary = inventory_scale(PRED_ROOT / scale)
        inventory_rows.extend(rows)
        summary_rows.append(summary)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["scale_dir", "month"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("scale_dir").reset_index(drop=True)
    return inventory_df, summary_df


def write_alignment_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    compare_cols = [
        "width",
        "height",
        "crs",
        "transform_a",
        "transform_b",
        "transform_c",
        "transform_d",
        "transform_e",
        "transform_f",
        "nodata",
        "dtype",
    ]
    first = summary_df.iloc[0]
    same_grid = True
    for _, row in summary_df.iloc[1:].iterrows():
        if any(row[col] != first[col] for col in compare_cols):
            same_grid = False
            break

    lines = [
        "Raster alignment check",
        "",
        f"Package: {PACKAGE_NAME}",
        f"Scales checked: {', '.join(summary_df['scale_dir'].tolist())}",
        f"Same raster grid across all scales: {'YES' if same_grid else 'NO'}",
        "",
        "Reference raster grid:",
    ]
    for col in compare_cols:
        lines.append(f"- {col}: {first[col]}")
    lines.extend(["", "Within-scale alignment:"])
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['scale_dir']}: {'YES' if row['all_aligned_within_scale'] else 'NO'}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_docs(pkg_dir: Path, summary_df: pd.DataFrame) -> None:
    docs_dir = pkg_dir / "docs"
    metadata_dir = pkg_dir / "metadata"
    preview_dir = pkg_dir / "preview"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    total_size = summary_df["total_size_gb"].sum()
    first_months = ", ".join(f"{row.scale_dir}:{row.first_month}" for row in summary_df.itertuples())

    readme = [
        f"# {PACKAGE_NAME}",
        "",
        "This directory is the final publication package for the dataset release.",
        "",
        "## Product summary",
        "- Spatial coverage: mainland China",
        "- Spatial resolution: 1 km",
        f"- Temporal coverage by timescale: {first_months} to 2018-12",
        "- Timescales included: SPEI-1, SPEI-3, SPEI-6, SPEI-12, SPEI-24",
        "- Data format inside archives: GeoTIFF (float32)",
        f"- Approximate total raster volume: {total_size:.2f} GB",
        "",
        "## Upload target",
        "- Upload the archive files in `archives/`.",
        "- Use the checksum file in `metadata/` for verification.",
    ]
    (docs_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    upload_list = [
        "Upload file list",
        "",
        "Upload the following files from `archives/`:",
        "",
    ]
    for scale in SCALES:
        label = SCALE_LABELS[scale]
        row = summary_df[summary_df["scale_dir"] == scale].iloc[0]
        upload_list.append(f"- China_1km_monthly_multiscale_SPEI_v1_{label}_{row['first_month']}_{row['last_month']}.tar")
    upload_list.append("- China_1km_monthly_multiscale_SPEI_v1_docs_metadata.zip")
    (docs_dir / "UPLOAD_FILE_LIST.txt").write_text("\n".join(upload_list), encoding="utf-8")

    for src_name, dst_name in SUMMARY_TABLES.items():
        src = TABLE_ROOT / src_name
        if src.exists():
            shutil.copy2(src, metadata_dir / dst_name)

    for src, dst_name in PREVIEW_FILES.items():
        if src.exists():
            shutil.copy2(src, preview_dir / dst_name)


def write_scale_manifest(scale: str, scale_df: pd.DataFrame, output_path: Path) -> None:
    label = f"SPEI-{int(scale[-2:])}"
    lines = [
        f"{label} archive note",
        "",
        f"Archive content: all monthly GeoTIFF grids for {label} over mainland China.",
        f"File count: {len(scale_df)}",
        f"First month: {scale_df['month'].min()}",
        f"Last month: {scale_df['month'].max()}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_scale_archive(pkg_dir: Path, scale: str, scale_df: pd.DataFrame) -> Path:
    archives_dir = pkg_dir / "archives"
    temp_dir = pkg_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    note_path = temp_dir / f"{scale}_README.txt"
    manifest_path = temp_dir / f"{scale}_manifest.csv"
    write_scale_manifest(scale, scale_df, note_path)
    scale_df.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)

    time_range = f"{scale_df['month'].min()}_{scale_df['month'].max()}"
    archive_name = f"China_1km_monthly_multiscale_SPEI_v1_{SCALE_LABELS[scale]}_{time_range}.tar"
    archive_path = archives_dir / archive_name
    with tarfile.open(archive_path, mode="w") as tf:
        tf.add(note_path, arcname=f"{PACKAGE_NAME}/README_{scale}.txt")
        tf.add(manifest_path, arcname=f"{PACKAGE_NAME}/metadata/{scale}_manifest.csv")
        for row in scale_df.itertuples(index=False):
            src = PRED_ROOT / scale / row.filename
            tf.add(src, arcname=f"{PACKAGE_NAME}/data/{scale}/{row.filename}")
    return archive_path


def build_docs_archive(pkg_dir: Path) -> Path:
    archive_path = pkg_dir / "archives" / "China_1km_monthly_multiscale_SPEI_v1_docs_metadata.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for subdir in ["docs", "metadata", "preview"]:
            for path in sorted((pkg_dir / subdir).rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=f"{PACKAGE_NAME}/{subdir}/{path.relative_to(pkg_dir / subdir)}")
    return archive_path


def write_archive_manifest(pkg_dir: Path, archive_paths: list[Path]) -> None:
    rows = [
        {
            "archive_name": path.name,
            "size_bytes": path.stat().st_size,
            "size_gb": round(path.stat().st_size / (1024**3), 3),
            "sha256": sha256sum(path),
        }
        for path in archive_paths
    ]
    df = pd.DataFrame(rows).sort_values("archive_name").reset_index(drop=True)
    df.to_csv(pkg_dir / "metadata" / "archive_manifest.csv", index=False)
    checksum_lines = [f"{row.sha256} *{row.archive_name}" for row in df.itertuples(index=False)]
    (pkg_dir / "metadata" / "archive_checksums_sha256.txt").write_text("\n".join(checksum_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the final dataset release for repository upload.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "derived_data" / "release_uploads", help="Output root directory")
    parser.add_argument("--skip-archives", action="store_true", help="Only build docs and metadata")
    parser.add_argument("--only-scale", choices=SCALES, help="Build one scale archive only")
    parser.add_argument("--docs-only", action="store_true", help="Build the documentation archive only")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing package directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkg_dir = args.out_root / PACKAGE_NAME
    if pkg_dir.exists() and not args.resume:
        shutil.rmtree(pkg_dir)

    for subdir in ["archives", "metadata", "docs", "preview"]:
        (pkg_dir / subdir).mkdir(parents=True, exist_ok=True)

    inventory_path = pkg_dir / "metadata" / "raster_inventory.csv"
    summary_path = pkg_dir / "metadata" / "product_summary.csv"
    if inventory_path.exists() and summary_path.exists() and args.resume:
        inventory_df = pd.read_csv(inventory_path)
        summary_df = pd.read_csv(summary_path)
    else:
        inventory_df, summary_df = build_inventory()
        inventory_df.to_csv(inventory_path, index=False, quoting=csv.QUOTE_MINIMAL)
        summary_df.to_csv(summary_path, index=False)

    write_alignment_report(summary_df, pkg_dir / "metadata" / "alignment_check.txt")
    write_docs(pkg_dir, summary_df)

    if not args.skip_archives:
        target_scales = [args.only_scale] if args.only_scale else SCALES
        if args.docs_only:
            target_scales = []

        for scale in target_scales:
            scale_df = inventory_df[inventory_df["scale_dir"] == scale].copy()
            archive_path = build_scale_archive(pkg_dir, scale, scale_df)
            print(f"Built {archive_path}")

        if args.docs_only or args.only_scale is None:
            docs_archive = build_docs_archive(pkg_dir)
            print(f"Built {docs_archive}")

        archives = sorted((pkg_dir / "archives").glob("*.tar"))
        docs_zip = pkg_dir / "archives" / "China_1km_monthly_multiscale_SPEI_v1_docs_metadata.zip"
        if docs_zip.exists():
            archives.append(docs_zip)
        if archives:
            write_archive_manifest(pkg_dir, archives)

    temp_dir = pkg_dir / "_temp"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            pass

    print(pkg_dir)


if __name__ == "__main__":
    main()
