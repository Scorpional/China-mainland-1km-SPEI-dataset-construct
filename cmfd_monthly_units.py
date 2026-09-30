from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def convert_cmfd_monthly_values(
    variable_name: str,
    values: np.ndarray,
    timestamps: Sequence[object] | object,
) -> np.ndarray:
    """Convert CMFD precipitation rates to monthly totals."""
    array = np.ma.filled(np.ma.asarray(values, dtype=np.float32), np.nan)
    if variable_name.lower() != "prec":
        return np.asarray(array, dtype=np.float32)

    time_index = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(timestamps)))
    factors = time_index.days_in_month.to_numpy(dtype=np.float32) * 24.0
    if factors.size == 1:
        return np.asarray(array * factors[0], dtype=np.float32)
    if array.ndim == 0 or array.shape[0] != factors.size:
        raise ValueError("Precipitation time dimension does not match the timestamps")
    shape = (factors.size,) + (1,) * (array.ndim - 1)
    return np.asarray(array * factors.reshape(shape), dtype=np.float32)
