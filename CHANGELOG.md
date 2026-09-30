# Changelog

## Version 2

- Enforced exact source-calendar month-end station labels.
- Added label audit outputs for missing, duplicate, and invalid daily records.
- Replaced nearest-cell CMFD assignment with NaN-aware bilinear interpolation.
- Converted monthly precipitation rates to monthly accumulation.
- Unified station and nationwide terrain predictors on the same NASADEM 1 km grid.
- Excluded SoilGrids variables from final model fitting and inference.
- Enforced complete lag and rolling windows for each SPEI timescale.
- Added safe, resumable version 2 release packaging and raster metadata checks.

## Version 1

- Initial public production workflow and dataset release.
