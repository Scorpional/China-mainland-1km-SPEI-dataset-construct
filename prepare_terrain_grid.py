from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from scipy.ndimage import distance_transform_edt


TARGET_CRS = "ESRI:54052"
TARGET_BOUNDS = (7082250.0, 2025000.0, 12333250.0, 5894000.0)
TARGET_WIDTH = 5251
TARGET_HEIGHT = 3869


def read_boundary(path: Path) -> list[dict]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("type") == "FeatureCollection":
        return [feature["geometry"] for feature in content["features"]]
    if content.get("type") == "Feature":
        return [content["geometry"]]
    return [content]


def build_vrt(nasadem_dir: Path, output_dir: Path) -> Path:
    sources: list[str] = []
    for archive in sorted(nasadem_dir.glob("NASADEM_HGT_*.zip")):
        with zipfile.ZipFile(archive) as handle:
            hgt_files = [name for name in handle.namelist() if name.lower().endswith(".hgt")]
        if len(hgt_files) != 1:
            raise ValueError(f"Expected one HGT file in {archive}, found {hgt_files}")
        sources.append(f"/vsizip/{archive.as_posix()}/{hgt_files[0]}")
    if not sources:
        raise FileNotFoundError(f"No NASADEM HGT archives found in {nasadem_dir}")
    vrt_path = output_dir / "nasadem_china.vrt"
    dataset = gdal.BuildVRT(str(vrt_path), sources)
    if dataset is None:
        raise RuntimeError("GDAL failed to build the NASADEM VRT")
    dataset = None
    return vrt_path


def warp_terrain_products(vrt_path: Path, boundary: Path, output_dir: Path) -> None:
    targets = {
        "dem_mean_1km.tif": "average",
        "dem_max_1km.tif": "max",
        "dem_min_1km.tif": "min",
        "dem_rms_1km.tif": "rms",
    }
    for filename, algorithm in targets.items():
        path = output_dir / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        options = gdal.WarpOptions(
            format="GTiff",
            outputBounds=TARGET_BOUNDS,
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            dstSRS=TARGET_CRS,
            resampleAlg=algorithm,
            cutlineDSName=str(boundary),
            cropToCutline=False,
            dstNodata=-9999.0,
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            multithread=True,
        )
        dataset = gdal.Warp(str(path), str(vrt_path), options=options)
        if dataset is None:
            raise RuntimeError(f"GDAL failed to generate {path}")
        dataset = None
        print(f"Wrote {path}")


def derive_terrain_features(output_dir: Path) -> None:
    source_names = ["dem_mean_1km.tif", "dem_max_1km.tif", "dem_min_1km.tif", "dem_rms_1km.tif"]
    sources = [rasterio.open(output_dir / name) for name in source_names]
    try:
        arrays = [source.read(1).astype(np.float32) for source in sources]
        nodata = sources[0].nodata
        for array, source in zip(arrays, sources, strict=True):
            if source.nodata is not None:
                array[np.isclose(array, source.nodata)] = np.nan
        mean, maximum, minimum, rms = arrays
        features = {
            "dem_elev_m.tif": mean,
            "dem_relief_1km.tif": maximum - minimum,
            "dem_std_1km.tif": np.sqrt(np.maximum(rms**2 - mean**2, 0.0)),
        }
        metadata = sources[0].meta.copy()
        metadata.update(dtype="float32", nodata=-9999.0, compress="lzw", tiled=True)
        for filename, values in features.items():
            path = output_dir / filename
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            output = np.where(np.isfinite(values), values, -9999.0).astype(np.float32)
            with rasterio.open(path, "w", **metadata) as destination:
                destination.write(output, 1)
    finally:
        for source in sources:
            source.close()


def build_mask_and_fill(boundary: Path, output_dir: Path) -> None:
    shapes = read_boundary(boundary)
    reference_path = output_dir / "dem_elev_m.tif"
    with rasterio.open(reference_path) as reference:
        projected = [transform_geom("EPSG:4326", reference.crs, shape) for shape in shapes]
        land = geometry_mask(
            projected,
            out_shape=(reference.height, reference.width),
            transform=reference.transform,
            invert=True,
            all_touched=False,
        )
        mask_metadata = reference.meta.copy()
        mask_metadata.update(dtype="uint8", nodata=0, count=1, compress="lzw", tiled=True)
    with rasterio.open(output_dir / "china_land_mask_1km.tif", "w", **mask_metadata) as destination:
        destination.write(land.astype(np.uint8), 1)

    for filename in ["dem_elev_m.tif", "dem_relief_1km.tif", "dem_std_1km.tif"]:
        path = output_dir / filename
        with rasterio.open(path, "r+") as dataset:
            values = dataset.read(1)
            valid = np.isfinite(values) & ~np.isclose(values, dataset.nodata)
            fill = land & ~valid
            if fill.any():
                nearest = distance_transform_edt(
                    ~valid, return_distances=False, return_indices=True
                )
                values[fill] = values[nearest[0][fill], nearest[1][fill]]
                dataset.write(values, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical 1 km NASADEM terrain stack used by version 2."
    )
    parser.add_argument("nasadem_dir", type=Path)
    parser.add_argument("china_boundary_geojson", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    vrt = build_vrt(args.nasadem_dir, args.output_dir)
    warp_terrain_products(vrt, args.china_boundary_geojson, args.output_dir)
    derive_terrain_features(args.output_dir)
    build_mask_and_fill(args.china_boundary_geojson, args.output_dir)
    print(f"Canonical terrain stack: {args.output_dir}")


if __name__ == "__main__":
    main()
