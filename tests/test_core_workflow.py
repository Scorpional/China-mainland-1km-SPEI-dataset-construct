from __future__ import annotations

import numpy as np
import pandas as pd

from build_training_samples import add_lagged_features
from cmfd_monthly_units import convert_cmfd_monthly_values
from cmfd_spatial_sampling import sample_regular_grid
from extract_station_monthly_spei import read_station_daily_spei
from model_config import select_production_features


def test_nan_aware_bilinear_sampling() -> None:
    grid = np.array([[1.0, 3.0], [5.0, np.nan]], dtype=np.float32)
    sampled = sample_regular_grid(
        grid,
        np.array([0.5]),
        np.array([0.5]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    assert np.isclose(sampled[0], 3.0)


def test_precipitation_rate_to_monthly_total() -> None:
    values = np.array([1.0, 1.0], dtype=np.float32)
    converted = convert_cmfd_monthly_values(
        "prec", values, pd.to_datetime(["2001-01-01", "2001-02-01"])
    )
    assert np.allclose(converted, [31 * 24, 28 * 24])


def test_strict_noleap_month_end(tmp_path) -> None:
    daily = tmp_path / "station_001.csv"
    frame = pd.DataFrame(
        {
            "year": [2000, 2000, 2000],
            "doy": [58, 59, 60],
            "spei": [-0.2, -0.5, 0.3],
        }
    )
    frame.to_csv(daily, index=False, header=False)
    labels, audit = read_station_daily_spei(
        daily, scale=1, monthly_rule="strict_calendar_end", calendar="noleap_365"
    )
    february = labels[(labels["year"] == 2000) & (labels["month"] == 2)].iloc[0]
    assert february["source_date"] == "2000-02-28"
    assert np.isclose(february["spei"], -0.5)
    assert bool(audit.loc[audit["month"] == 2, "has_month_end_record"].iloc[0])


def test_complete_temporal_windows() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["1"] * 3,
            "year": [1979] * 3,
            "month": [1, 2, 3],
            "dyn_prec": [1.0, 2.0, 3.0],
        }
    )
    output, history = add_lagged_features(frame, ["dyn_prec"], scale=3, max_lag=2)
    assert output.loc[0, history].isna().any()
    assert output.loc[1, history].isna().any()
    assert not output.loc[2, history].isna().any()
    assert np.isclose(output.loc[2, "dyn_prec_roll3_sum"], 6.0)


def test_production_features_exclude_soil_and_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["1"],
            "spei": [0.0],
            "year": [2000],
            "dyn_prec": [10.0],
            "dem_elev_m": [100.0],
            "soil_clay_0_5cm": [20.0],
        }
    )
    assert select_production_features(frame) == ["year", "dyn_prec", "dem_elev_m"]
