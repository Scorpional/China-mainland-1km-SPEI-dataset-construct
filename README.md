# China 1 km Monthly Multi-timescale SPEI Dataset

Production workflow for version 2 of the China 1 km monthly multi-timescale
SPEI dataset over mainland China (1979-2018).

- Dataset: https://doi.org/10.57760/sciencedb.34500
- Timescales: SPEI-1, SPEI-3, SPEI-6, SPEI-12, and SPEI-24
- Output: monthly Float32 GeoTIFF, 1,000 m, ESRI:54052
- Model: one LightGBM regressor per timescale

## Version 2 method

Version 2 implements the workflow used for the revised dataset:

1. Extract the exact final day of each month from the source 365-day no-leap
   station SPEI records. No fallback to an earlier daily record is allowed.
2. Convert monthly CMFD precipitation rates to monthly totals and sample all
   seven CMFD variables with NaN-aware bilinear interpolation.
3. Build a canonical 1 km NASADEM terrain stack containing mean elevation,
   relief, and elevation standard deviation.
4. Sample the same canonical terrain rasters at stations and use them during
   nationwide inference, avoiding a train-inference terrain mismatch.
5. Construct complete timescale-specific lag and rolling features, monthly
   climatologies, and departures from climatology.
6. Fit the LightGBM Leaf63 model and infer the mainland-China grid in row blocks.

SoilGrids variables were tested separately in an ablation experiment but are
not predictors in the version 2 production models or released rasters.

## Scripts

- `extract_station_monthly_spei.py`: fixed month-end station labels and audit files.
- `sample_cmfd_monthly_at_stations.py`: NaN-aware bilinear CMFD station sampling.
- `prepare_terrain_grid.py`: canonical 1 km NASADEM terrain rasters and land mask.
- `build_station_static_features.py`: sample canonical terrain at stations.
- `build_training_samples.py`: merge labels and predictors and add temporal memory.
- `add_station_climatology_anomalies.py`: add monthly climatology and anomaly features.
- `reconstruct_monthly_spei_grids.py`: fit and apply the final no-soil model.
- `run_random_fivefold_cv.py`: sample-wise random five-fold validation.
- `run_out_station_cv_lightgbm.py`: station-grouped five-fold validation.
- `run_spatial_block_buffer_cv.py`: spatial-block validation with a distance buffer.
- `run_blocked_period_validation.py`: forward validation on complete held-out periods.
- `run_drought_condition_validation.py`: drought-threshold detection from held-out predictions.
- `run_reference_product_consistency_validation.py`: whole-period comparison with public products.
- `package_dataset_release.py`: validate and package the version 2 release.

Compact tables from the revision experiments are provided in `results/`. Full
predictions, intermediate arrays, third-party products, and raw data are not
tracked because of their size or redistribution terms.

## Installation

The exact environment used for the version 2 release is recorded in
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate spei-reconstruction-v2
```

Alternatively, install the unpinned runtime requirements in an existing
environment:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

GDAL must match the GDAL library installed on the host system. A Conda
environment is recommended when installing `gdal` and `rasterio` on Windows.

## Processing order

The following commands show the required order. Replace paths with local data
locations.

```bash
python extract_station_monthly_spei.py DAILY_SPEI_DIR STATION_LOOKUP.csv monthly_labels.csv \
  --calendar noleap_365 --monthly-rule strict_calendar_end \
  --start-year 1979 --end-year 2018

python sample_cmfd_monthly_at_stations.py STATION_LOOKUP.csv CMFD_MONTHLY_DIR cmfd_station.csv

python prepare_terrain_grid.py NASADEM_HGT_DIR china_admin0.geojson terrain_1km

python build_station_static_features.py STATION_LOOKUP.csv terrain_1km station_terrain.csv

python build_training_samples.py monthly_labels.csv cmfd_station.csv station_terrain.csv \
  derived_data/training_samples/spei03_training_samples_base_1979-2018.csv \
  --scale 3 --year-min 1979 --year-max 2018

python add_station_climatology_anomalies.py \
  derived_data/training_samples/spei03_training_samples_base_1979-2018.csv \
  derived_data/training_samples/spei03_training_samples_1979-2018.csv --scale 3

python reconstruct_monthly_spei_grids.py \
  derived_data/training_samples/spei03_training_samples_1979-2018.csv \
  CMFD_MONTHLY_DIR terrain_1km cache output/spei03 \
  --scale 3 --start 1979-03 --end 2018-12 --skip-existing
```

Repeat the final three timescale-specific steps for scales 1, 3, 6, 12, and 24.

## Model parameters

The fixed parameters are defined in `model_config.py`: 800 trees, learning
rate 0.03, 63 leaves, minimum child size 30, feature fraction 0.8, L2
regularization 0.5, and random seed 42. Median imputation is fitted only on
the model-training table.

## Validation

The production workflow uses random five-fold validation as the primary model
performance summary and adds stricter tests for spatial and temporal transfer:

```bash
python run_random_fivefold_cv.py
python run_out_station_cv_lightgbm.py
python run_spatial_block_buffer_cv.py --block-km 200 --buffer-km 100
python run_blocked_period_validation.py

python run_drought_condition_validation.py \
  --input-csv PATH_TO_SPATIAL_OOF_PREDICTIONS.csv

python run_reference_product_consistency_validation.py \
  --admin0 PATH_TO_MAINLAND_BOUNDARY.geojson \
  --admin1 PATH_TO_PROVINCE_BOUNDARIES.geojson \
  --comparison-dir PATH_TO_REFERENCE_PRODUCTS \
  --zhang-nc PATH_TO_ZHANG_SPEI.nc --resume
```

Administrative GeoJSON inputs are expected in WGS84 and are transformed to the
reference-product grid before rasterization. They are supplied explicitly
because boundary redistribution rights depend on the selected data source.

## Version 2 release packaging

```bash
python package_dataset_release.py FINAL_GRID_ROOT RELEASE_OUTPUT_ROOT \
  --results-root EXPERIMENT_RESULTS \
  --feature-schema-dir FEATURE_SCHEMA \
  --build-all
```

The packaging script refuses to overwrite existing data archives. It verifies
the documented monthly coverage, common grid geometry, bilinear-CMFD metadata,
excluded-soil metadata, and completion tags before creating the archives.

## Data provenance

- Station SPEI: National Cryosphere Desert Data Center,
  https://www.ncdc.ac.cn/portal/metadata/1f5961a2-fe24-46ea-8dff-18cc54fa7433
- CMFD: National Tibetan Plateau / Third Pole Environment Data Center,
  https://data.tpdc.ac.cn/zh-hans/data/e60dfd96-5fd8-493f-beae-e8e5d24dece4
- NASADEM: NASA JPL NASADEM_HGT.001,
  https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001

Raw third-party data are not redistributed in this repository.
