# SPEI Dataset Pipeline

Core production scripts for the China 1 km monthly multiscale SPEI dataset workflow.

## Workflow

1. `extract_station_monthly_spei.py`
   Extract month-end SPEI labels from daily station files.

2. `sample_cmfd_monthly_at_stations.py`
   Sample monthly CMFD variables at station locations.

3. `build_station_static_features.py`
   Build NASADEM and SoilGrids predictors for each station.

4. `build_training_samples.py`
   Join labels and predictors, then generate lagged and rolling features for one SPEI scale.

5. `reconstruct_monthly_spei_grids.py`
   Train the final LightGBM model and reconstruct 1 km monthly grids.

6. `package_dataset_release.py`
   Build the publication package and archive files for repository upload.

## Output

- Monthly station labels
- Station-based dynamic and static predictors
- Training tables for each SPEI timescale
- Monthly 1 km GeoTIFF grids
- Repository-ready release archives

## Notes

- The raster product is distributed as GeoTIFF files inside archive packages.
- This directory is the cleaned code handoff for the final workflow.
