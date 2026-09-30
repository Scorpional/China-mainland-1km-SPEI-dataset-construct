from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tarfile
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from remotezip import RemoteZip
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_geom

from plot_style import set_style


ROOT = Path(__file__).resolve().parent
OUR_DIR = ROOT / "derived_data" / "final_grid_1km"
OUR_ARCHIVE_DIR = (
    ROOT
    / "derived_data"
    / "release_uploads"
    / "China_1km_monthly_multiscale_SPEI_1979_2018_v2"
    / "archives"
)
COMPARISON_DIR = ROOT / "derived_data" / "comparison_products"
ZHANG_NC = ROOT / "data_downloads" / "comparison" / "zhang_monthly" / "Monthly_SPEI_GEV_3M_1979-2018.nc"
HSPEI_URL_MANIFEST = ROOT / "derived_data" / "metadata" / "hspei_urls" / "V6.txt"
OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "reference_product_consistency_validation"
DEFAULT_SCALES = (1, 3, 6, 12, 24)
CORR_NODATA = -9999.0
REMOTE_REQUEST_TIMEOUT = (30.0, 180.0)
MONTH_RE = re.compile(r"(?P<month>\d{4}-\d{2})")
TIF_PRODUCT_RE = re.compile(
    r"(?P<prefix>[a-zA-Z0-9]+)_spei(?P<scale>\d{2})_(?P<month>\d{4}-\d{2})\.tif$",
    re.IGNORECASE,
)
OUR_FILE_RE = re.compile(r"spei_(?P<scale>\d{2})mo_(?P<month>\d{4}-\d{2})\.tif$", re.IGNORECASE)

# This experiment complements the representative-year map figures by checking
# whether Our SPEI remains consistent with public reference products across the
# full overlapping period, shared SPEI timescales, and common valid pixels.


@dataclass
class ReferenceProduct:
    key: str
    name: str
    source_type: str
    scales: list[int]
    month_maps: dict[int, dict[str, Path | None]]
    nc_path: Path | None = None
    remote_zip_urls: dict[int, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whole-period consistency between our 1 km monthly SPEI dataset "
            "and public reference SPEI products."
        )
    )
    parser.add_argument("--our-dir", type=Path, default=OUR_DIR)
    parser.add_argument(
        "--our-archive-dir",
        type=Path,
        default=OUR_ARCHIVE_DIR,
        help="Fallback directory containing one uncompressed TAR archive per SPEI timescale.",
    )
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--zhang-nc", type=Path, default=ZHANG_NC)
    parser.add_argument(
        "--admin0",
        type=Path,
        required=True,
        help="Mainland-China boundary GeoJSON in WGS84 coordinates.",
    )
    parser.add_argument(
        "--admin1",
        type=Path,
        required=True,
        help="Province-level boundary GeoJSON in WGS84 coordinates.",
    )
    parser.add_argument("--hspei-url-manifest", type=Path, default=HSPEI_URL_MANIFEST)
    parser.add_argument(
        "--include-hspei-remote",
        action="store_true",
        help="Read HSPEI GeoTIFF members on demand with HTTP range requests; full ZIP files are not downloaded.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timescales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--min-months-for-corr", type=int, default=3)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume completed product-timescale pairs from checkpoints.")
    parser.add_argument(
        "--reference-products",
        nargs="+",
        help="Optional reference keys or names to run, for example hspei or zhang_spei.",
    )
    parser.add_argument(
        "--remote-request-delay",
        type=float,
        default=1.0,
        help="Pause between remotely streamed monthly rasters to reduce HTTP rate limiting.",
    )
    return parser.parse_args()


def read_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_polygons(feature_collection: dict):
    for feature in feature_collection.get("features", []):
        geom = feature["geometry"]
        props = feature.get("properties", {})
        yield geom, props


def scale_label(scale: int) -> str:
    return f"SPEI-{scale}"


def scale_dirname(scale: int) -> str:
    return f"spei{scale:02d}"


def humanize_product(prefix: str) -> str:
    lower = prefix.lower()
    if "gpr" in lower:
        return "GPR-SPEI"
    if "hspei" in lower:
        return "HSPEI"
    if "zhang" in lower:
        return "Zhang-SPEI"
    return prefix


def our_month_map(our_dir: Path, archive_dir: Path | None = None) -> dict[int, dict[str, Path | str]]:
    month_maps: dict[int, dict[str, Path | str]] = {}
    for scale in DEFAULT_SCALES:
        scale_path = our_dir / scale_dirname(scale)
        if not scale_path.exists():
            continue
        mapping: dict[str, Path] = {}
        for tif_path in sorted(scale_path.glob("spei_*mo_*.tif")):
            match = OUR_FILE_RE.match(tif_path.name)
            if not match:
                continue
            month = match.group("month")
            mapping[month] = tif_path
        if mapping:
            month_maps[scale] = mapping

    if month_maps or archive_dir is None or not archive_dir.exists():
        return month_maps

    for scale in DEFAULT_SCALES:
        archives = sorted(archive_dir.glob(f"*SPEI{scale:02d}_*.tar"))
        if len(archives) != 1:
            continue
        tar_path = archives[0].resolve()
        mapping: dict[str, Path | str] = {}
        with tarfile.open(tar_path, mode="r:") as archive:
            for member in archive:
                filename = Path(member.name).name
                match = OUR_FILE_RE.match(filename)
                if not member.isfile() or match is None or int(match.group("scale")) != scale:
                    continue
                vsi_path = f"/vsitar/{tar_path.as_posix()}/{member.name}"
                mapping[match.group("month")] = vsi_path
        if mapping:
            month_maps[scale] = mapping
    return month_maps


def discover_hspei_remote(
    manifest_path: Path,
    requested_scales: list[int],
) -> ReferenceProduct:
    if not manifest_path.exists():
        raise FileNotFoundError(f"HSPEI URL manifest not found: {manifest_path}")

    zip_urls: dict[int, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        file_name = parse_qs(urlparse(line).query).get("fileName", [""])[0]
        match = re.fullmatch(r"HSPEI_(\d+)\.zip", file_name, flags=re.IGNORECASE)
        if match:
            zip_urls[int(match.group(1))] = line

    month_maps: dict[int, dict[str, Path | None]] = {}
    selected_urls: dict[int, str] = {}
    for scale in requested_scales:
        remote_url = zip_urls.get(scale)
        if remote_url is None:
            continue
        mapping: dict[str, Path | None] = {}
        with RemoteZip(
            remote_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=REMOTE_REQUEST_TIMEOUT,
        ) as archive:
            for info in archive.infolist():
                match = re.search(
                    rf"(?:^|/)SPEI_{scale}_(\d{{4}})(\d{{2}})\.tif$",
                    info.filename,
                    flags=re.IGNORECASE,
                )
                if match:
                    mapping[f"{match.group(1)}-{match.group(2)}"] = Path(info.filename)
        if mapping:
            month_maps[scale] = mapping
            selected_urls[scale] = remote_url

    if not month_maps:
        raise ValueError("No requested HSPEI timescales were found in the remote ZIP manifests")
    return ReferenceProduct(
        key="hspei",
        name="HSPEI",
        source_type="remote_zip_tif",
        scales=sorted(month_maps),
        month_maps=month_maps,
        remote_zip_urls=selected_urls,
    )


def discover_reference_products(
    comparison_dir: Path,
    zhang_nc: Path,
    include_hspei_remote: bool = False,
    hspei_url_manifest: Path = HSPEI_URL_MANIFEST,
    requested_scales: list[int] | None = None,
) -> list[ReferenceProduct]:
    products: list[ReferenceProduct] = []

    if zhang_nc.exists():
        products.append(
            ReferenceProduct(
                key="zhang_spei",
                name="Zhang-SPEI",
                source_type="zhang_nc",
                scales=[3],
                month_maps={3: {}},
                nc_path=zhang_nc,
            )
        )

    grouped: dict[str, dict[int, dict[str, Path | None]]] = {}
    for tif_path in sorted(comparison_dir.glob("*.tif")):
        match = TIF_PRODUCT_RE.match(tif_path.name)
        if not match:
            continue
        prefix = match.group("prefix")
        if prefix.lower().startswith("zhang"):
            continue
        scale = int(match.group("scale"))
        month = match.group("month")
        grouped.setdefault(prefix, {}).setdefault(scale, {})[month] = tif_path

    for prefix, scale_map in sorted(grouped.items()):
        products.append(
            ReferenceProduct(
                key=prefix.lower(),
                name=humanize_product(prefix),
                source_type="monthly_tif",
                scales=sorted(scale_map),
                month_maps=scale_map,
                nc_path=None,
            )
        )

    if include_hspei_remote:
        products.append(
            discover_hspei_remote(
                hspei_url_manifest,
                requested_scales or list(DEFAULT_SCALES),
            )
        )

    return products


def month_list_from_range(start: str, end: str) -> list[str]:
    start_ts = pd.Period(start, freq="M")
    end_ts = pd.Period(end, freq="M")
    return [str(period) for period in pd.period_range(start=start_ts, end=end_ts, freq="M")]


def zhang_month_map(zhang_nc: Path) -> dict[str, None]:
    with rasterio.open(zhang_nc) as ds:
        end_month = pd.Period("1979-01", freq="M") + (ds.count - 1)
        months = month_list_from_range("1979-01", str(end_month))
    return {month: None for month in months}


def month_to_zhang_band(month: str) -> int:
    target = pd.Period(month, freq="M")
    start = pd.Period("1979-01", freq="M")
    return int(target.ordinal - start.ordinal) + 1


def choose_resampling(reference_name: str, reference_shape: tuple[int, int]) -> Resampling:
    if reference_name == "Zhang-SPEI":
        return Resampling.average
    return Resampling.bilinear


def reference_crs(reference: ReferenceProduct, ref_ds):
    if reference.source_type == "zhang_nc":
        return "EPSG:4326"
    return ref_ds.crs


def iter_windows(height: int, width: int, block_size: int):
    for row_off in range(0, height, block_size):
        row_size = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            col_size = min(block_size, width - col_off)
            yield rasterio.windows.Window(col_off, row_off, col_size, row_size)


def create_region_rasters(
    transform,
    width: int,
    height: int,
    target_crs,
    admin0_path: Path,
    admin1_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    admin0 = read_geojson(admin0_path)
    admin1 = read_geojson(admin1_path)

    mainland_shapes = [
        (transform_geom("EPSG:4326", target_crs, geom), 1)
        for geom, _ in iter_polygons(admin0)
    ]
    mainland_mask = rasterize(
        mainland_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    province_shapes = []
    province_names: dict[int, str] = {}
    for idx, (geom, props) in enumerate(iter_polygons(admin1), start=1):
        province_shapes.append((transform_geom("EPSG:4326", target_crs, geom), idx))
        province_names[idx] = str(props.get("name", f"region_{idx}"))

    province_ids = rasterize(
        province_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int16",
        all_touched=False,
    )
    return mainland_mask, province_ids, province_names


def metric_from_sums(count: int, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, sum_abs_err: float, sum_sq_err: float, sum_err: float) -> dict[str, float]:
    if count == 0:
        return {"Pearson_r": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "Bias": float("nan")}
    rmse = math.sqrt(sum_sq_err / count)
    mae = sum_abs_err / count
    bias = sum_err / count
    num = count * sum_xy - sum_x * sum_y
    den_x = count * sum_x2 - sum_x * sum_x
    den_y = count * sum_y2 - sum_y * sum_y
    if den_x <= 0 or den_y <= 0:
        pearson_r = float("nan")
    else:
        pearson_r = num / math.sqrt(den_x * den_y)
    return {
        "Pearson_r": float(pearson_r),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
    }


def save_plot_multi(fig: plt.Figure, out_stem: Path, dpi: int = 320) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".jpg"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def open_reference_dataset(
    reference: ReferenceProduct,
    scale: int,
    month: str,
    stack: ExitStack,
    remote_archive: RemoteZip | None = None,
):
    if reference.source_type == "zhang_nc":
        if reference.nc_path is None:
            raise ValueError("Missing Zhang NetCDF path")
        return stack.enter_context(rasterio.open(reference.nc_path))
    month_path = reference.month_maps[scale][month]
    if month_path is None:
        raise ValueError(f"Missing monthly raster path for {reference.name} {scale_label(scale)} {month}")
    if reference.source_type == "remote_zip_tif":
        if reference.remote_zip_urls is None or scale not in reference.remote_zip_urls:
            raise ValueError(f"Missing remote ZIP URL for {reference.name} {scale_label(scale)}")
        archive = remote_archive
        if archive is None:
            archive = stack.enter_context(
                RemoteZip(
                    reference.remote_zip_urls[scale],
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=REMOTE_REQUEST_TIMEOUT,
                )
            )
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                if attempt == 0:
                    payload = archive.read(month_path.as_posix())
                else:
                    retry_archive = stack.enter_context(
                        RemoteZip(
                            reference.remote_zip_urls[scale],
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=REMOTE_REQUEST_TIMEOUT,
                        )
                    )
                    payload = retry_archive.read(month_path.as_posix())
                break
            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    raise
                delay = 30.0 * (2 ** attempt)
                print(
                    f"Remote read failed for {reference.name} {scale_label(scale)} {month}; "
                    f"retry in {delay:.0f} s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        else:
            raise RuntimeError("Remote ZIP retry loop exited unexpectedly") from last_error
        memory_file = stack.enter_context(MemoryFile(payload))
        return stack.enter_context(memory_file.open())
    return stack.enter_context(rasterio.open(month_path))


def read_reference_window(ds, reference: ReferenceProduct, month: str, window) -> np.ndarray:
    if reference.source_type == "zhang_nc":
        band = month_to_zhang_band(month)
        arr = ds.read(band, window=window).astype("float32")
        nodata = ds.nodata if ds.nodata is not None else -99999.0
        arr[arr == nodata] = np.nan
        arr *= 0.0001
        return arr

    arr = ds.read(1, window=window).astype("float32")
    nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def read_our_on_reference_grid(
    our_path: Path | str,
    ref_ds,
    dst_crs,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.open(our_path) as src:
        dest = np.full((ref_ds.height, ref_ds.width), CORR_NODATA, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_ds.transform,
            dst_crs=dst_crs,
            dst_nodata=CORR_NODATA,
            resampling=resampling,
        )
        dest[dest == CORR_NODATA] = np.nan
        return dest


def init_region_accumulators(region_names: dict[int, str]) -> dict[int, dict[str, float | int]]:
    return {
        region_id: {
            "count": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_y2": 0.0,
            "sum_xy": 0.0,
            "sum_abs_err": 0.0,
            "sum_sq_err": 0.0,
            "sum_err": 0.0,
        }
        for region_id in region_names
    }


def update_region_accumulators(
    region_accums: dict[int, dict[str, float | int]],
    province_ids: np.ndarray,
    our_arr: np.ndarray,
    ref_arr: np.ndarray,
    mask: np.ndarray,
    n_regions: int,
) -> None:
    valid_ids = province_ids[mask]
    if valid_ids.size == 0:
        return
    valid = valid_ids > 0
    if not np.any(valid):
        return

    ids = valid_ids[valid].astype(np.int32)
    x = our_arr[mask][valid].astype("float64")
    y = ref_arr[mask][valid].astype("float64")
    err = x - y

    count_inc = np.bincount(ids, minlength=n_regions + 1)
    sum_x_inc = np.bincount(ids, weights=x, minlength=n_regions + 1)
    sum_y_inc = np.bincount(ids, weights=y, minlength=n_regions + 1)
    sum_x2_inc = np.bincount(ids, weights=x * x, minlength=n_regions + 1)
    sum_y2_inc = np.bincount(ids, weights=y * y, minlength=n_regions + 1)
    sum_xy_inc = np.bincount(ids, weights=x * y, minlength=n_regions + 1)
    sum_abs_err_inc = np.bincount(ids, weights=np.abs(err), minlength=n_regions + 1)
    sum_sq_err_inc = np.bincount(ids, weights=err * err, minlength=n_regions + 1)
    sum_err_inc = np.bincount(ids, weights=err, minlength=n_regions + 1)

    for region_id, accum in region_accums.items():
        accum["count"] += int(count_inc[region_id])
        accum["sum_x"] += float(sum_x_inc[region_id])
        accum["sum_y"] += float(sum_y_inc[region_id])
        accum["sum_x2"] += float(sum_x2_inc[region_id])
        accum["sum_y2"] += float(sum_y2_inc[region_id])
        accum["sum_xy"] += float(sum_xy_inc[region_id])
        accum["sum_abs_err"] += float(sum_abs_err_inc[region_id])
        accum["sum_sq_err"] += float(sum_sq_err_inc[region_id])
        accum["sum_err"] += float(sum_err_inc[region_id])


def create_memmaps(tmp_dir: Path, shape: tuple[int, int], mode: str = "w+"):
    return {
        "count": np.memmap(tmp_dir / "count.dat", mode=mode, dtype="uint16", shape=shape),
        "sum_x": np.memmap(tmp_dir / "sum_x.dat", mode=mode, dtype="float32", shape=shape),
        "sum_y": np.memmap(tmp_dir / "sum_y.dat", mode=mode, dtype="float32", shape=shape),
        "sum_x2": np.memmap(tmp_dir / "sum_x2.dat", mode=mode, dtype="float32", shape=shape),
        "sum_y2": np.memmap(tmp_dir / "sum_y2.dat", mode=mode, dtype="float32", shape=shape),
        "sum_xy": np.memmap(tmp_dir / "sum_xy.dat", mode=mode, dtype="float32", shape=shape),
    }


def flush_memmaps(memmaps: dict[str, np.memmap]) -> None:
    for arr in memmaps.values():
        arr.flush()


def write_month_checkpoint(
    state_path: Path,
    common_months: list[str],
    completed_months: int,
    shape: tuple[int, int],
    global_sums: dict[str, float | int],
    region_accums: dict[int, dict[str, float | int]],
    mean_rows: list[dict[str, float | int | str]],
) -> None:
    state = {
        "common_months": common_months,
        "completed_months": completed_months,
        "shape": list(shape),
        "global_sums": global_sums,
        "region_accums": {str(key): value for key, value in region_accums.items()},
        "mean_rows": mean_rows,
    }
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(state_path)


def compute_corr_tif(
    memmaps: dict[str, np.memmap],
    out_path: Path,
    profile: dict,
    block_size: int,
    min_months_for_corr: int,
) -> int:
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=CORR_NODATA,
        compress="lzw",
        predictor=2,
    )
    height = profile["height"]
    width = profile["width"]
    valid_pixel_count = 0

    with rasterio.open(out_path, "w", **profile) as dst:
        for window in iter_windows(height, width, block_size):
            row0 = int(window.row_off)
            row1 = row0 + int(window.height)
            col0 = int(window.col_off)
            col1 = col0 + int(window.width)

            count = np.asarray(memmaps["count"][row0:row1, col0:col1], dtype="float64")
            sum_x = np.asarray(memmaps["sum_x"][row0:row1, col0:col1], dtype="float64")
            sum_y = np.asarray(memmaps["sum_y"][row0:row1, col0:col1], dtype="float64")
            sum_x2 = np.asarray(memmaps["sum_x2"][row0:row1, col0:col1], dtype="float64")
            sum_y2 = np.asarray(memmaps["sum_y2"][row0:row1, col0:col1], dtype="float64")
            sum_xy = np.asarray(memmaps["sum_xy"][row0:row1, col0:col1], dtype="float64")

            corr = np.full(count.shape, CORR_NODATA, dtype="float32")
            valid = count >= min_months_for_corr
            if np.any(valid):
                num = count * sum_xy - sum_x * sum_y
                den_x = count * sum_x2 - sum_x * sum_x
                den_y = count * sum_y2 - sum_y * sum_y
                denom = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0))
                valid &= denom > 0
                corr_valid = np.full(count.shape, np.nan, dtype="float64")
                corr_valid[valid] = num[valid] / denom[valid]
                corr[valid] = corr_valid[valid].astype("float32")
                valid_pixel_count += int(np.count_nonzero(count > 0))
            else:
                valid_pixel_count += int(np.count_nonzero(count > 0))

            dst.write(corr, 1, window=window)

    return valid_pixel_count


def build_reference_profile(reference: ReferenceProduct, scale: int, sample_month: str) -> dict:
    with ExitStack() as stack:
        ref_ds = open_reference_dataset(reference, scale, sample_month, stack)
        profile = ref_ds.profile.copy()
        if reference.source_type == "zhang_nc":
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=CORR_NODATA,
                compress="lzw",
                predictor=2,
                crs="EPSG:4326",
            )
        return profile


def common_months_for_product(
    our_months: dict[int, dict[str, Path | str]],
    reference: ReferenceProduct,
    scale: int,
) -> list[str]:
    if scale not in our_months:
        return []
    if reference.source_type == "zhang_nc":
        ref_months = zhang_month_map(reference.nc_path) if reference.nc_path is not None else {}
    else:
        ref_months = reference.month_maps.get(scale, {})
    months = sorted(set(our_months[scale]).intersection(ref_months))
    return months


def write_time_series_plot(out_dir: Path, mean_df: pd.DataFrame, reference_name: str, scale: int) -> None:
    sub = mean_df[(mean_df["reference_product"] == reference_name) & (mean_df["timescale"] == scale_label(scale))].copy()
    if sub.empty:
        return
    sub["date"] = pd.to_datetime(sub["month"] + "-01")

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.plot(sub["date"], sub["our_mean"], color="#1d4ed8", linewidth=1.1, label="Our SPEI")
    ax.plot(sub["date"], sub["reference_mean"], color="#d97706", linewidth=1.0, linestyle="--", label=reference_name)
    ax.set_ylabel(scale_label(scale))
    ax.set_xlabel("Year")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_title(f"National monthly mean comparison: {reference_name}, {scale_label(scale)}")
    fig.tight_layout()
    stem = out_dir / f"{reference_name.lower().replace(' ', '_').replace('-', '_')}_{scale:02d}_national_mean_timeseries"
    save_plot_multi(fig, stem)


def write_region_boxplot(out_dir: Path, region_df: pd.DataFrame, scale: int) -> None:
    sub = region_df[region_df["timescale"] == scale_label(scale)].copy()
    if sub.empty:
        return
    order = sorted(sub["reference_product"].unique())
    data = [sub.loc[sub["reference_product"] == name, "Pearson_r"].dropna().to_numpy() for name in order]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.boxplot(data, tick_labels=order, patch_artist=True)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Regional consistency by reference product: {scale_label(scale)}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_plot_multi(fig, out_dir / f"region_boxplot_{scale:02d}")


def write_corr_map_preview(out_dir: Path, corr_tif: Path, reference_name: str, scale: int) -> None:
    with rasterio.open(corr_tif) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    ax.set_title(f"Grid correlation: {reference_name}, {scale_label(scale)}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    save_plot_multi(
        fig,
        out_dir / f"{reference_name.lower().replace(' ', '_').replace('-', '_')}_{scale:02d}_corr_map",
    )


def run_validation(
    our_months: dict[int, dict[str, Path | str]],
    reference: ReferenceProduct,
    scale: int,
    common_months: list[str],
    output_dir: Path,
    block_size: int,
    min_months_for_corr: int,
    make_plots: bool,
    remote_request_delay: float,
    admin0_path: Path,
    admin1_path: Path,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    sample_month = common_months[0]
    profile = build_reference_profile(reference, scale, sample_month)
    height = profile["height"]
    width = profile["width"]
    transform = profile["transform"]
    resampling = choose_resampling(reference.name, (height, width))
    mainland_mask, province_ids, province_names = create_region_rasters(
        transform,
        width,
        height,
        profile["crs"],
        admin0_path,
        admin1_path,
    )
    n_regions = len(province_names)
    corr_tif = output_dir / f"reference_product_grid_correlation_{reference.key}_{scale:02d}.tif"
    common_period = f"{common_months[0]} to {common_months[-1]}"

    progress_dir = output_dir / f"progress_{reference.key}_{scale:02d}"
    state_path = progress_dir / "state.json"
    progress_dir.mkdir(parents=True, exist_ok=True)
    completed_months = 0

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("common_months") != common_months or state.get("shape") != [height, width]:
            raise ValueError(f"Incompatible month checkpoint: {state_path}")
        completed_months = int(state["completed_months"])
        global_sums = state["global_sums"]
        region_accums = {int(key): value for key, value in state["region_accums"].items()}
        mean_rows = state["mean_rows"]
        memmaps = create_memmaps(progress_dir, (height, width), mode="r+")
        print(
            f"Resume month checkpoint for {reference.name} {scale_label(scale)}: "
            f"{completed_months}/{len(common_months)} months",
            flush=True,
        )
    else:
        global_sums = {
            "count": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_y2": 0.0,
            "sum_xy": 0.0,
            "sum_abs_err": 0.0,
            "sum_sq_err": 0.0,
            "sum_err": 0.0,
        }
        region_accums = init_region_accumulators(province_names)
        mean_rows: list[dict[str, float | int | str]] = []
        memmaps = create_memmaps(progress_dir, (height, width))

    global_count = int(global_sums["count"])
    global_sum_x = float(global_sums["sum_x"])
    global_sum_y = float(global_sums["sum_y"])
    global_sum_x2 = float(global_sums["sum_x2"])
    global_sum_y2 = float(global_sums["sum_y2"])
    global_sum_xy = float(global_sums["sum_xy"])
    global_sum_abs_err = float(global_sums["sum_abs_err"])
    global_sum_sq_err = float(global_sums["sum_sq_err"])
    global_sum_err = float(global_sums["sum_err"])

    try:
        with ExitStack() as product_stack:
            remote_archive = None
            if reference.source_type == "remote_zip_tif":
                if reference.remote_zip_urls is None or scale not in reference.remote_zip_urls:
                    raise ValueError(f"Missing remote ZIP URL for {reference.name} {scale_label(scale)}")
                remote_archive = product_stack.enter_context(
                    RemoteZip(
                        reference.remote_zip_urls[scale],
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=REMOTE_REQUEST_TIMEOUT,
                    )
                )

            for month_index, month in enumerate(common_months[completed_months:], start=completed_months):
                with ExitStack() as stack:
                    ref_ds = open_reference_dataset(
                        reference,
                        scale,
                        month,
                        stack,
                        remote_archive=remote_archive,
                    )
                    dst_crs = reference_crs(reference, ref_ds)
                    our_full = read_our_on_reference_grid(
                        our_months[scale][month],
                        ref_ds,
                        dst_crs,
                        resampling,
                    )

                    month_sum_our = 0.0
                    month_sum_ref = 0.0
                    month_count = 0

                    for window in iter_windows(height, width, block_size):
                        row0 = int(window.row_off)
                        row1 = row0 + int(window.height)
                        col0 = int(window.col_off)
                        col1 = col0 + int(window.width)

                        ref_arr = read_reference_window(ref_ds, reference, month, window)
                        our_arr = our_full[row0:row1, col0:col1]

                        local_mainland = mainland_mask[row0:row1, col0:col1]
                        local_province = province_ids[row0:row1, col0:col1]
                        pair_mask = local_mainland & np.isfinite(our_arr) & np.isfinite(ref_arr)
                        if not np.any(pair_mask):
                            continue

                        x = our_arr[pair_mask].astype("float64")
                        y = ref_arr[pair_mask].astype("float64")
                        err = x - y
                        n = x.size

                        global_count += n
                        global_sum_x += float(x.sum())
                        global_sum_y += float(y.sum())
                        global_sum_x2 += float((x * x).sum())
                        global_sum_y2 += float((y * y).sum())
                        global_sum_xy += float((x * y).sum())
                        global_sum_abs_err += float(np.abs(err).sum())
                        global_sum_sq_err += float((err * err).sum())
                        global_sum_err += float(err.sum())

                        month_sum_our += float(x.sum())
                        month_sum_ref += float(y.sum())
                        month_count += n

                        update_region_accumulators(
                            region_accums,
                            local_province,
                            our_arr,
                            ref_arr,
                            pair_mask,
                            n_regions,
                        )

                        memmaps["count"][row0:row1, col0:col1] += pair_mask.astype("uint16")
                        x_full = np.where(pair_mask, our_arr, 0.0).astype("float32")
                        y_full = np.where(pair_mask, ref_arr, 0.0).astype("float32")
                        memmaps["sum_x"][row0:row1, col0:col1] += x_full
                        memmaps["sum_y"][row0:row1, col0:col1] += y_full
                        memmaps["sum_x2"][row0:row1, col0:col1] += x_full * x_full
                        memmaps["sum_y2"][row0:row1, col0:col1] += y_full * y_full
                        memmaps["sum_xy"][row0:row1, col0:col1] += x_full * y_full

                    if month_count > 0:
                        mean_rows.append(
                            {
                                "reference_product": reference.name,
                                "common_period": common_period,
                                "timescale": scale_label(scale),
                                "month": month,
                                "our_mean": month_sum_our / month_count,
                                "reference_mean": month_sum_ref / month_count,
                            }
                        )
                flush_memmaps(memmaps)
                write_month_checkpoint(
                    state_path=state_path,
                    common_months=common_months,
                    completed_months=month_index + 1,
                    shape=(height, width),
                    global_sums={
                        "count": global_count,
                        "sum_x": global_sum_x,
                        "sum_y": global_sum_y,
                        "sum_x2": global_sum_x2,
                        "sum_y2": global_sum_y2,
                        "sum_xy": global_sum_xy,
                        "sum_abs_err": global_sum_abs_err,
                        "sum_sq_err": global_sum_sq_err,
                        "sum_err": global_sum_err,
                    },
                    region_accums=region_accums,
                    mean_rows=mean_rows,
                )
                print(
                    f"Completed {reference.name} {scale_label(scale)} {month} "
                    f"({month_index + 1}/{len(common_months)})",
                    flush=True,
                )
                if reference.source_type == "remote_zip_tif" and remote_request_delay > 0:
                    time.sleep(remote_request_delay)

        flush_memmaps(memmaps)
        n_valid_pixels = compute_corr_tif(memmaps, corr_tif, profile, block_size, min_months_for_corr)
    finally:
        for key, arr in list(memmaps.items()):
            mmap_handle = getattr(arr, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
            del memmaps[key]

    summary_metrics = metric_from_sums(
        global_count,
        global_sum_x,
        global_sum_y,
        global_sum_x2,
        global_sum_y2,
        global_sum_xy,
        global_sum_abs_err,
        global_sum_sq_err,
        global_sum_err,
    )
    summary_row: dict[str, float | int | str] = {
        "reference_product": reference.name,
        "common_period": common_period,
        "timescale": scale_label(scale),
        "n_months": len(common_months),
        "n_valid_pixels": n_valid_pixels,
        **summary_metrics,
    }

    region_rows: list[dict[str, float | int | str]] = []
    for region_id, region_name in province_names.items():
        accum = region_accums[region_id]
        if int(accum["count"]) == 0:
            continue
        region_metrics = metric_from_sums(
            int(accum["count"]),
            float(accum["sum_x"]),
            float(accum["sum_y"]),
            float(accum["sum_x2"]),
            float(accum["sum_y2"]),
            float(accum["sum_xy"]),
            float(accum["sum_abs_err"]),
            float(accum["sum_sq_err"]),
            float(accum["sum_err"]),
        )
        region_rows.append(
            {
                "reference_product": reference.name,
                "region": region_name,
                "timescale": scale_label(scale),
                "common_period": common_period,
                "n_months": len(common_months),
                **region_metrics,
            }
        )

    if make_plots:
        mean_df = pd.DataFrame(mean_rows)
        write_time_series_plot(output_dir, mean_df, reference.name, scale)
        write_corr_map_preview(output_dir, corr_tif, reference.name, scale)

    return summary_row, region_rows, mean_rows


def main() -> None:
    args = parse_args()
    set_style()
    for boundary_path in (args.admin0, args.admin1):
        if not boundary_path.is_file():
            raise FileNotFoundError(f"Boundary GeoJSON not found: {boundary_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    our_months = our_month_map(args.our_dir, args.our_archive_dir)
    if not our_months:
        raise FileNotFoundError(
            f"No Our SPEI monthly grids found under {args.our_dir} or {args.our_archive_dir}"
        )

    references = discover_reference_products(
        args.comparison_dir,
        args.zhang_nc,
        include_hspei_remote=args.include_hspei_remote,
        hspei_url_manifest=args.hspei_url_manifest,
        requested_scales=args.timescales,
    )
    if args.reference_products:
        requested = {value.lower() for value in args.reference_products}
        references = [
            reference
            for reference in references
            if reference.key.lower() in requested or reference.name.lower() in requested
        ]
    if not references:
        raise FileNotFoundError("No reference SPEI products were discovered.")

    checkpoint_summary = args.output_dir / "reference_product_consistency_summary_checkpoint.csv"
    checkpoint_region = args.output_dir / "reference_product_consistency_by_region_checkpoint.csv"
    checkpoint_mean = args.output_dir / "reference_product_consistency_monthly_means_checkpoint.csv"
    summary_rows: list[dict[str, float | int | str]] = []
    region_rows: list[dict[str, float | int | str]] = []
    mean_rows: list[dict[str, float | int | str]] = []
    if args.resume and checkpoint_summary.exists():
        summary_rows = pd.read_csv(checkpoint_summary).to_dict("records")
        if checkpoint_region.exists():
            region_rows = pd.read_csv(checkpoint_region).to_dict("records")
        if checkpoint_mean.exists():
            mean_rows = pd.read_csv(checkpoint_mean).to_dict("records")
    completed = {(str(row["reference_product"]), str(row["timescale"])) for row in summary_rows}

    for reference in references:
        common_scales = sorted(set(reference.scales).intersection(args.timescales))
        if not common_scales:
            print(f"Skip {reference.name}: no shared timescales with requested set.")
            continue

        for scale in common_scales:
            result_key = (reference.name, scale_label(scale))
            if result_key in completed:
                print(f"Resume: skip completed {reference.name} {scale_label(scale)}")
                continue
            common_months = common_months_for_product(our_months, reference, scale)
            if not common_months:
                print(f"Skip {reference.name} {scale_label(scale)}: no common months.")
                continue

            print(
                f"Run {reference.name} {scale_label(scale)}: "
                f"{common_months[0]} to {common_months[-1]} "
                f"({len(common_months)} months)"
            )
            summary_row, current_region_rows, current_mean_rows = run_validation(
                our_months=our_months,
                reference=reference,
                scale=scale,
                common_months=common_months,
                output_dir=args.output_dir,
                block_size=args.block_size,
                min_months_for_corr=args.min_months_for_corr,
                make_plots=args.make_plots,
                remote_request_delay=args.remote_request_delay,
                admin0_path=args.admin0,
                admin1_path=args.admin1,
            )
            summary_rows.append(summary_row)
            region_rows.extend(current_region_rows)
            mean_rows.extend(current_mean_rows)
            pd.DataFrame(summary_rows).to_csv(
                args.output_dir / "reference_product_consistency_summary_checkpoint.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(region_rows).to_csv(
                args.output_dir / "reference_product_consistency_by_region_checkpoint.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(mean_rows).to_csv(
                args.output_dir / "reference_product_consistency_monthly_means_checkpoint.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary_df = pd.DataFrame(summary_rows)
    region_df = pd.DataFrame(region_rows)
    mean_df = pd.DataFrame(mean_rows)

    summary_path = args.output_dir / "reference_product_consistency_summary.csv"
    region_path = args.output_dir / "reference_product_consistency_by_region.csv"
    mean_path = args.output_dir / "reference_product_consistency_monthly_means.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    region_df.to_csv(region_path, index=False, encoding="utf-8-sig")
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    if args.make_plots and not region_df.empty:
        for timescale_label in sorted(region_df["timescale"].unique()):
            scale = int(timescale_label.split("-")[1])
            write_region_boxplot(args.output_dir, region_df, scale)

    print("\nReference product consistency summary")
    if summary_df.empty:
        print("No comparable product/scale pairs were found.")
    else:
        print(summary_df.to_string(index=False))

    print(f"\nWrote summary table: {summary_path}")
    print(f"Wrote regional table: {region_path}")
    print(f"Wrote monthly means: {mean_path}")
    print(
        "\nThis experiment validates whether Our SPEI remains consistent with public "
        "reference SPEI products over the full overlapping period and across shared "
        "timescales, rather than only in a few representative snapshot years."
    )


if __name__ == "__main__":
    main()
