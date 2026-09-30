from __future__ import annotations

import numpy as np


def prepare_regular_grid_sampling(
    sample_lons: np.ndarray,
    sample_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
) -> tuple[np.ndarray, ...]:
    sample_lons = np.asarray(sample_lons, dtype=float)
    sample_lats = np.asarray(sample_lats, dtype=float)
    grid_lons = np.asarray(grid_lons, dtype=float)
    grid_lats = np.asarray(grid_lats, dtype=float)
    if sample_lons.shape != sample_lats.shape:
        raise ValueError("Longitude and latitude arrays must have the same shape")
    if grid_lons.ndim != 1 or grid_lats.ndim != 1:
        raise ValueError("CMFD longitude and latitude coordinates must be one-dimensional")

    lon_pos = (sample_lons - grid_lons[0]) / (grid_lons[1] - grid_lons[0])
    lat_pos = (sample_lats - grid_lats[0]) / (grid_lats[1] - grid_lats[0])
    lon_nearest = np.clip(np.rint(lon_pos).astype(int), 0, len(grid_lons) - 1)
    lat_nearest = np.clip(np.rint(lat_pos).astype(int), 0, len(grid_lats) - 1)
    lon_low = np.clip(np.floor(lon_pos).astype(int), 0, len(grid_lons) - 2)
    lat_low = np.clip(np.floor(lat_pos).astype(int), 0, len(grid_lats) - 2)
    lon_weight = np.clip(lon_pos - lon_low, 0.0, 1.0)
    lat_weight = np.clip(lat_pos - lat_low, 0.0, 1.0)
    return lon_nearest, lat_nearest, lon_low, lat_low, lon_weight, lat_weight


def sample_regular_grid(
    array: np.ndarray,
    sample_lons: np.ndarray,
    sample_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    fallback_array: np.ndarray | None = None,
    fallback_values: np.ndarray | None = None,
    sampling_plan: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    """Apply NaN-aware bilinear sampling to the final two array dimensions."""
    values = np.ma.filled(np.ma.asarray(array, dtype=np.float32), np.nan)
    if fallback_array is not None and fallback_values is not None:
        raise ValueError("Specify either fallback_array or fallback_values")
    plan = sampling_plan or prepare_regular_grid_sampling(
        sample_lons, sample_lats, grid_lons, grid_lats
    )
    lon_nearest, lat_nearest, lon_low, lat_low, lon_weight, lat_weight = plan
    corners = np.stack(
        [
            values[..., lat_low, lon_low],
            values[..., lat_low, lon_low + 1],
            values[..., lat_low + 1, lon_low],
            values[..., lat_low + 1, lon_low + 1],
        ],
        axis=-1,
    )
    weights = np.stack(
        [
            (1.0 - lon_weight) * (1.0 - lat_weight),
            lon_weight * (1.0 - lat_weight),
            (1.0 - lon_weight) * lat_weight,
            lon_weight * lat_weight,
        ],
        axis=-1,
    )
    finite = np.isfinite(corners)
    numerator = np.where(finite, corners * weights, 0.0).sum(axis=-1)
    denominator = np.where(finite, weights, 0.0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sampled = np.where(denominator > 0.0, numerator / denominator, np.nan)
    missing = denominator <= 0.0

    if fallback_array is not None:
        fallback = np.ma.filled(
            np.ma.asarray(fallback_array, dtype=np.float32), np.nan
        )[..., lat_nearest, lon_nearest]
        sampled = np.where(missing, fallback, sampled)
    elif fallback_values is not None:
        fallback = np.asarray(fallback_values, dtype=np.float32)
        if fallback.shape != sampled.shape:
            raise ValueError("Fallback values do not match sampled output shape")
        sampled = np.where(missing, fallback, sampled)
    return np.asarray(sampled, dtype=np.float32)
