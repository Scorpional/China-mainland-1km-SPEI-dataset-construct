from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import rasterio


PACKAGE_NAME = "China_1km_monthly_multiscale_SPEI_1979_2018_v2"
PACKAGE_VERSION = "2.0.0"
DATASET_DOI = "https://doi.org/10.57760/sciencedb.34500"
CODE_URL = "https://github.com/Scorpional/China-mainland-1km-SPEI-dataset-construct"
SCALE_CONFIG = {
    "spei01": ("SPEI01", 1, "1979-01", "2018-12", 480),
    "spei03": ("SPEI03", 3, "1979-03", "2018-12", 478),
    "spei06": ("SPEI06", 6, "1979-06", "2018-12", 475),
    "spei12": ("SPEI12", 12, "1979-12", "2018-12", 469),
    "spei24": ("SPEI24", 24, "1980-12", "2018-12", 457),
}
VALIDATION_FILES = {
    "final_nosoil/random_fivefold_cv/random_fivefold_summary.csv": "random_fivefold_summary.csv",
    "final_nosoil/station_groupkfold_cv/out_station_cv_summary.csv": "station_groupkfold_summary.csv",
    "final_nosoil/spatial_block_buffer_cv/spatial_block_cv_summary.csv": "spatial_block_cv_summary.csv",
    "final_nosoil/blocked_period_validation/blocked_period_summary.csv": "blocked_period_summary.csv",
    "final_nosoil/drought_condition_validation_spatial/drought_condition_validation_summary.csv": "drought_condition_validation_summary.csv",
    "reference_product_consistency_all_scales/reference_product_consistency_summary.csv": "reference_product_consistency_summary.csv",
    "soil_ablation/soil_ablation_delta_summary.csv": "soil_ablation_delta_summary.csv",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_months(start: str, end: str) -> list[str]:
    return pd.period_range(start=start, end=end, freq="M").astype(str).tolist()


def inspect_scale(grid_root: Path, scale_dir: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    label, months, start, end, expected_count = SCALE_CONFIG[scale_dir]
    files = sorted((grid_root / scale_dir).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files found in {grid_root / scale_dir}")

    rows: list[dict[str, object]] = []
    geometries: set[tuple[object, ...]] = set()
    observed_months: list[str] = []
    for path in files:
        month = path.stem.rsplit("_", 1)[-1]
        observed_months.append(month)
        with rasterio.open(path) as src:
            transform = src.transform
            tags = src.tags()
            geometry = (
                src.width,
                src.height,
                str(src.crs),
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
                src.nodata,
                src.dtypes[0],
            )
            geometries.add(geometry)
            rows.append(
                {
                    "timescale": f"SPEI-{months}",
                    "month": month,
                    "file": f"{scale_dir}/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "transform_a": transform.a,
                    "transform_b": transform.b,
                    "transform_c": transform.c,
                    "transform_d": transform.d,
                    "transform_e": transform.e,
                    "transform_f": transform.f,
                    "pixel_size_x": abs(transform.a),
                    "pixel_size_y": abs(transform.e),
                    "nodata": src.nodata,
                    "dtype": src.dtypes[0],
                    "compression": str(src.compression).split(".")[-1].lower(),
                    "cmfd_spatial_method": tags.get("cmfd_spatial_method", ""),
                    "soil_predictors": tags.get("soil_predictors", ""),
                    "generation_complete": tags.get("generation_complete", ""),
                }
            )

    expected = expected_months(start, end)
    missing = sorted(set(expected).difference(observed_months))
    extra = sorted(set(observed_months).difference(expected))
    if len(files) != expected_count or missing or extra:
        raise ValueError(
            f"{label} coverage mismatch: files={len(files)}, expected={expected_count}, "
            f"missing={missing}, extra={extra}"
        )
    if len(geometries) != 1:
        raise ValueError(f"Raster geometry is inconsistent within {label}")
    if any(row["cmfd_spatial_method"] != "bilinear" for row in rows):
        raise ValueError(f"Not all {label} rasters carry the bilinear CMFD tag")
    if any(row["soil_predictors"] != "excluded" for row in rows):
        raise ValueError(f"Not all {label} rasters carry the excluded-soil tag")
    if any(row["generation_complete"] != "true" for row in rows):
        raise ValueError(f"Not all {label} rasters are marked complete")

    first = rows[0]
    summary = {
        "timescale": f"SPEI-{months}",
        "scale_dir": scale_dir,
        "n_files": len(files),
        "first_month": start,
        "last_month": end,
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "total_size_gib": round(sum(int(row["size_bytes"]) for row in rows) / 1024**3, 3),
        "width": first["width"],
        "height": first["height"],
        "crs": first["crs"],
        "pixel_size_m": first["pixel_size_x"],
        "nodata": first["nodata"],
        "dtype": first["dtype"],
        "compression": first["compression"],
        "cmfd_spatial_method": first["cmfd_spatial_method"],
        "soil_predictors": first["soil_predictors"],
        "missing_months": 0,
    }
    return rows, summary


def build_inventory(grid_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for scale_dir in SCALE_CONFIG:
        scale_rows, summary = inspect_scale(grid_root, scale_dir)
        rows.extend(scale_rows)
        summaries.append(summary)
    inventory = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)

    geometry_columns = [
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
    if len(inventory[geometry_columns].drop_duplicates()) != 1:
        raise ValueError("Raster geometry differs among SPEI timescales")
    return inventory, summary


def copy_optional(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_documentation(package_dir: Path, summary: pd.DataFrame) -> None:
    docs = package_dir / "docs"
    total_files = int(summary["n_files"].sum())
    total_gib = float(summary["total_size_gib"].sum())
    coverage = ", ".join(
        f"{row.timescale}: {row.first_month} to {row.last_month}"
        for row in summary.itertuples(index=False)
    )
    root_readme = f"""# {PACKAGE_NAME}

This directory contains the upload-ready release of the China 1 km monthly
multi-timescale SPEI dataset over mainland China.

- Dataset DOI: {DATASET_DOI}
- Source code: {CODE_URL}
- Data files: one TAR archive per SPEI timescale
- Documentation: one ZIP archive containing metadata, validation summaries,
  release notes, and preview figures

The `v1` release is retained separately. Version 2 must not be mixed with
version 1 because the reconstruction workflow and raster values differ.
"""
    (package_dir / "README.md").write_text(root_readme, encoding="utf-8")

    readme = f"""# China 1 km Monthly Multi-timescale SPEI Dataset, Version 2

## Product summary

- Spatial coverage: mainland China
- Target grid: 1,000 m, ESRI:54052
- Raster dimensions: 5,251 columns x 3,869 rows
- Temporal resolution: monthly
- SPEI timescales: 1, 3, 6, 12, and 24 months
- Temporal coverage: {coverage}
- Raster count: {total_files}
- Unarchived raster volume: {total_gib:.2f} GiB
- Data type: Float32 GeoTIFF with LZW compression
- NoData value: -9999

## Reconstruction represented by version 2

Version 2 uses exact month-end station SPEI labels on the source 365-day
no-leap calendar, NaN-aware bilinear mapping of monthly 0.1-degree CMFD
forcing, and a canonical 1 km NASADEM terrain stack shared by station training
and nationwide inference. SoilGrids predictors were evaluated in an ablation
experiment but are excluded from the released production model.

## File organization

- `archives/`: one data archive per SPEI timescale and one documentation ZIP.
- `metadata/`: raster inventory, grid metadata, validation summaries, and checksums.
- `docs/`: release notes, citation and data-availability text, and usage notes.
- `preview/`: station and spatial-pattern figures for visual inspection.

Each data archive contains monthly GeoTIFFs under
`{PACKAGE_NAME}/data/<scale>/`. Filenames follow
`spei_<NN>mo_<YYYY-MM>.tif`.
"""
    (docs / "README.md").write_text(readme, encoding="utf-8")

    readme_cn = f"""# 中国大陆 1 km 月尺度多时间尺度 SPEI 数据集（版本 2）

版本 2 包含 SPEI-1、SPEI-3、SPEI-6、SPEI-12 和 SPEI-24，共 {total_files} 个
月尺度 Float32 GeoTIFF。目标网格为 ESRI:54052 投影下的 1000 m 网格，
NoData 为 -9999。

本版本采用固定月末站点标签、CMFD 缺失值感知双线性空间映射，以及训练和
全国推理共用的 NASADEM 1 km 地形栅格。SoilGrids 仅用于消融试验，未进入
最终生产模型。
"""
    (docs / "README_CN.md").write_text(readme_cn, encoding="utf-8")

    release_notes = """Version 2 release notes

Changes relative to version 1:
- Rebuilt monthly station labels using an exact source-calendar month-end rule.
- Replaced nearest-cell CMFD assignment with NaN-aware bilinear mapping.
- Used the same canonical 1 km NASADEM terrain rasters for station training and inference.
- Excluded all SoilGrids predictors from final model fitting and nationwide inference.
- Regenerated all five SPEI timescales and completed raster-level quality control.
- Updated spatial-block, blocked-period, drought-threshold, uncertainty, and reference-product validation.

Version 2 supersedes version 1 for scientific use.
"""
    (docs / "RELEASE_NOTES.txt").write_text(release_notes, encoding="utf-8")

    citation = f"""Dataset citation

Please cite the associated article and the dataset record:
Peng, H.; Li, W.; Zhang, K.; Wu, C.; Liu, H.; Chu, Y.; Chen, X.
China 1 km Monthly Multi-timescale SPEI Dataset over Mainland China (1979-2018), version 2.0.0.
Science Data Bank. {DATASET_DOI}

Code: {CODE_URL}
"""
    (docs / "CITATION.txt").write_text(citation, encoding="utf-8")

    availability = (
        "The dataset generated in this study is openly available in Science Data Bank at "
        f"{DATASET_DOI}. The source code is available at {CODE_URL}.\n"
    )
    (docs / "DATA_AVAILABILITY.txt").write_text(availability, encoding="utf-8")

    sciencedb = f"""# ScienceDB Submission Metadata

## Dataset title
China 1 km Monthly Multi-timescale SPEI Dataset over Mainland China (1979-2018), Version 2

## Creators
Haoxiang Peng; Wenxing Li; Kun Zhang; Chengrong Wu; Hui Liu; Yihang Chu; Xidong Chen

## Dataset description
This dataset provides monthly gridded Standardized Precipitation Evapotranspiration
Index (SPEI) data over mainland China at 1 km spatial resolution for SPEI-1,
SPEI-3, SPEI-6, SPEI-12, and SPEI-24. The product was reconstructed from exact
month-end SPEI observations at 427 stations, monthly CMFD meteorological forcing,
and NASADEM terrain predictors using timescale-specific LightGBM models. CMFD
fields were mapped using NaN-aware bilinear interpolation. Soil properties are
not inputs to the released version 2 production model.

## Keywords
SPEI; drought; mainland China; 1 km; monthly; GeoTIFF; CMFD; NASADEM; LightGBM

## Subject classification
Earth Sciences; Climatology; Hydrology; Remote Sensing and GIS

## Spatial information
Mainland China; ESRI:54052; 1,000 m pixels; 5,251 x 3,869 cells

## Temporal information
Monthly, 1979-2018, with scale-dependent spin-up periods documented in the metadata.
"""
    (docs / "ScienceDB_submission_metadata_EN.md").write_text(sciencedb, encoding="utf-8")

    upload_lines = ["Version 2 upload file list", ""]
    for scale_dir, (label, _, start, end, _) in SCALE_CONFIG.items():
        upload_lines.append(f"- China_1km_monthly_multiscale_SPEI_v2_{label}_{start}_{end}.tar")
    upload_lines.append("- China_1km_monthly_multiscale_SPEI_v2_docs_metadata.zip")
    upload_lines.append("- upload_checksums_sha256.txt")
    (docs / "UPLOAD_FILE_LIST.txt").write_text("\n".join(upload_lines), encoding="utf-8")


def write_machine_metadata(
    package_dir: Path,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    first = inventory.iloc[0]
    width = int(first["width"])
    height = int(first["height"])
    transform = [
        float(first["transform_a"]),
        float(first["transform_b"]),
        float(first["transform_c"]),
        float(first["transform_d"]),
        float(first["transform_e"]),
        float(first["transform_f"]),
    ]
    left = transform[2]
    top = transform[5]
    right = left + transform[0] * width
    bottom = top + transform[4] * height
    metadata = {
        "title": "China 1 km Monthly Multi-timescale SPEI Dataset over Mainland China (1979-2018)",
        "version": PACKAGE_VERSION,
        "doi": DATASET_DOI,
        "code_repository": CODE_URL,
        "creators": [
            "Haoxiang Peng",
            "Wenxing Li",
            "Kun Zhang",
            "Chengrong Wu",
            "Hui Liu",
            "Yihang Chu",
            "Xidong Chen",
        ],
        "variable": {
            "name": "SPEI",
            "long_name": "Standardized Precipitation Evapotranspiration Index",
            "unit": "dimensionless",
            "interpretation": "Negative values indicate drier conditions and positive values indicate wetter conditions.",
            "dtype": str(first["dtype"]),
            "nodata": float(first["nodata"]),
        },
        "grid": {
            "crs": str(first["crs"]),
            "pixel_size_m": float(first["pixel_size_x"]),
            "width": width,
            "height": height,
            "affine_coefficients_a_b_c_d_e_f": transform,
            "bounds": {
                "left": round(left, 6),
                "bottom": round(bottom, 6),
                "right": round(right, 6),
                "top": round(top, 6),
            },
        },
        "temporal_resolution": "monthly",
        "timescales": [
            {
                "name": str(row.timescale),
                "start": str(row.first_month),
                "end": str(row.last_month),
                "file_count": int(row.n_files),
            }
            for row in summary.itertuples(index=False)
        ],
        "file_format": "GeoTIFF",
        "compression": str(first["compression"]),
        "file_naming": "spei_<NN>mo_<YYYY-MM>.tif",
        "raster_count": int(summary["n_files"].sum()),
    }
    output = package_dir / "metadata" / "dataset_metadata.json"
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_alignment_report(inventory: pd.DataFrame, output_path: Path) -> None:
    first = inventory.iloc[0]
    lines = [
        "Version 2 raster alignment and provenance check",
        "",
        f"Raster files checked: {len(inventory)}",
        "Common grid across all files: YES",
        f"Shape: {first['height']} rows x {first['width']} columns",
        f"CRS: {first['crs']}",
        f"Pixel size: {first['pixel_size_x']} m x {first['pixel_size_y']} m",
        f"NoData: {first['nodata']}",
        f"Data type: {first['dtype']}",
        f"Compression: {first['compression']}",
        "CMFD spatial method: NaN-aware bilinear mapping",
        "Soil predictors in released model: excluded",
        "Missing months: none within each documented scale-specific period",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_scale_archive(
    package_dir: Path,
    grid_root: Path,
    scale_dir: str,
    scale_inventory: pd.DataFrame,
) -> Path:
    label, months, start, end, _ = SCALE_CONFIG[scale_dir]
    archive = package_dir / "archives" / f"China_1km_monthly_multiscale_SPEI_v2_{label}_{start}_{end}.tar"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    note = (
        f"SPEI-{months} monthly GeoTIFF archive, version 2\n"
        f"Coverage: {start} to {end}\n"
        f"Files: {len(scale_inventory)}\n"
    )
    note_path = package_dir / "metadata" / f"{scale_dir}_README.txt"
    manifest_path = package_dir / "metadata" / f"{scale_dir}_manifest.csv"
    note_path.write_text(note, encoding="utf-8")
    scale_inventory.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with tarfile.open(archive, "w") as handle:
        handle.add(note_path, arcname=f"{PACKAGE_NAME}/README_{scale_dir}.txt")
        handle.add(manifest_path, arcname=f"{PACKAGE_NAME}/metadata/{scale_dir}_manifest.csv")
        for row in scale_inventory.itertuples(index=False):
            source = grid_root / row.file
            handle.add(source, arcname=f"{PACKAGE_NAME}/data/{row.file}")
    return archive


def update_archive_manifest(package_dir: Path) -> None:
    archives = sorted((package_dir / "archives").glob("*.tar"))
    manifest_path = package_dir / "metadata" / "archive_manifest.csv"
    previous: dict[tuple[str, int], str] = {}
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        previous = {
            (str(row.archive_name), int(row.size_bytes)): str(row.sha256)
            for row in old.itertuples(index=False)
        }
    rows = []
    for path in archives:
        size = path.stat().st_size
        checksum = previous.get((path.name, size)) or sha256sum(path)
        rows.append(
            {
                "archive_name": path.name,
                "size_bytes": size,
                "size_gib": round(size / 1024**3, 3),
                "sha256": checksum,
            }
        )
    pd.DataFrame(rows, columns=["archive_name", "size_bytes", "size_gib", "sha256"]).to_csv(
        manifest_path, index=False
    )
    checksum_lines = [f"{row['sha256']} *{row['archive_name']}" for row in rows]
    (package_dir / "metadata" / "archive_checksums_sha256.txt").write_text(
        "\n".join(checksum_lines), encoding="utf-8"
    )


def build_docs_archive(package_dir: Path) -> Path:
    archive = package_dir / "archives" / "China_1km_monthly_multiscale_SPEI_v2_docs_metadata.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as handle:
        handle.write(package_dir / "README.md", arcname=f"{PACKAGE_NAME}/README.md")
        for subdir in ["docs", "metadata", "preview"]:
            for path in sorted((package_dir / subdir).rglob("*")):
                if path.is_file():
                    handle.write(path, arcname=f"{PACKAGE_NAME}/{subdir}/{path.relative_to(package_dir / subdir)}")
    return archive


def write_upload_checksums(package_dir: Path, docs_archive: Path) -> None:
    manifest = pd.read_csv(package_dir / "metadata" / "archive_manifest.csv")
    lines = [
        f"{row.sha256} *{row.archive_name}"
        for row in manifest.itertuples(index=False)
    ]
    lines.append(f"{sha256sum(docs_archive)} *{docs_archive.name}")
    (package_dir / "upload_checksums_sha256.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the version 2 dataset release package.")
    parser.add_argument("grid_root", type=Path, help="Directory containing spei01, spei03, spei06, spei12, and spei24")
    parser.add_argument("output_root", type=Path, help="Parent directory for the version 2 package")
    parser.add_argument("--results-root", type=Path, help="Experiment-results directory used to collect validation summaries")
    parser.add_argument("--feature-schema-dir", type=Path, help="Directory containing the final feature-schema CSV files")
    parser.add_argument("--preview", action="append", type=Path, default=[], help="Preview image to include; may be repeated")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing version 2 package directory")
    parser.add_argument("--build-scale", choices=list(SCALE_CONFIG), help="Build one large scale archive")
    parser.add_argument("--build-all", action="store_true", help="Build all five large scale archives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_scale and args.build_all:
        raise ValueError("Use either --build-scale or --build-all, not both")
    package_dir = args.output_root / PACKAGE_NAME
    if package_dir.exists() and not args.resume:
        raise FileExistsError(f"Package already exists; use --resume: {package_dir}")
    for name in ["archives", "docs", "metadata", "preview"]:
        (package_dir / name).mkdir(parents=True, exist_ok=True)

    inventory_path = package_dir / "metadata" / "raster_inventory.csv"
    summary_path = package_dir / "metadata" / "product_summary.csv"
    if args.resume and inventory_path.exists() and summary_path.exists():
        inventory = pd.read_csv(inventory_path)
        summary = pd.read_csv(summary_path)
    else:
        inventory, summary = build_inventory(args.grid_root)
        inventory.to_csv(inventory_path, index=False)
        summary.to_csv(summary_path, index=False)
    write_alignment_report(inventory, package_dir / "metadata" / "alignment_check.txt")
    write_documentation(package_dir, summary)
    write_machine_metadata(package_dir, inventory, summary)

    if args.results_root:
        for relative_source, destination in VALIDATION_FILES.items():
            copy_optional(args.results_root / relative_source, package_dir / "metadata" / destination)
    if args.feature_schema_dir:
        for source in args.feature_schema_dir.glob("*.csv"):
            copy_optional(source, package_dir / "metadata" / source.name)
    for source in args.preview:
        copy_optional(source, package_dir / "preview" / source.name)

    target_scales = list(SCALE_CONFIG) if args.build_all else ([args.build_scale] if args.build_scale else [])
    for scale_dir in target_scales:
        scale_inventory = inventory[inventory["file"].str.startswith(f"{scale_dir}/")].copy()
        archive = build_scale_archive(package_dir, args.grid_root, scale_dir, scale_inventory)
        print(f"Built {archive}", flush=True)

    update_archive_manifest(package_dir)
    docs_archive = build_docs_archive(package_dir)
    write_upload_checksums(package_dir, docs_archive)
    print(f"Built {docs_archive}")
    print(f"Version 2 package: {package_dir}")


if __name__ == "__main__":
    main()
