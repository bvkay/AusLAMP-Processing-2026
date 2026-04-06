# Archived Notebooks

These notebooks represent the initial exploratory data analysis phase of the AusLAMP Victoria project (April 2026).

## Archived Files

- **old_01_metadata_exploration.ipynb** - Initial site metadata extraction from EDL raw MiniSEED files
- **old_02_timeseries_qaqc.ipynb** - Batch QA/QC and ASCII export (71/81 sites successful)
- **old_03_debug_problem_sites.ipynb** - Diagnostics for sites with loading errors
- **old_04_reexport_problem_sites.ipynb** - Attempted re-export with robust error handling
- **old_05_combined_dataset_analysis.ipynb** - Combined EDL + LEMI-424 analysis with AGRF comparison

## Why Archived?

These notebooks were based on:
1. Reading MiniSEED files directly with ObsPy
2. Exporting to ASCII (.dat files)
3. Inline LEMI-424 reader code (not modularized)
4. Manual metadata extraction from multiple sources

## New Approach (Current)

The refactored codebase (April 2026) uses:
1. **GA's MTH5 files as primary data source** (both EDL and LEMI-424)
2. **Unified MTH5 reader** (`src/readers/mth5_reader.py`)
3. **Correct Bartington Mag-03 calibration** (GA's calibration was incorrect)
4. **Modular architecture** with proper testing
5. **SQLite metadata database** instead of fragmented CSVs

## Key Findings from Archived Work

### Data Quality Issues
- 10 EDL sites had loading errors (channel length mismatches, very short recordings)
- Problem sites: VIC046, VIC049, VIC054, VIC059, VIC060, VIC061, VIC063, VIC068, VIC074, VIC076, VIC078b, VIC100

### Calibration Discovery
- GA's MTH5 files contain **uncalibrated** EDL data (raw µV)
- All channels incorrectly labeled "units: celsius degrees"
- LEMI-424 data is **already calibrated** in nT and µV/m

### Dataset Summary
- **EDL**: 81 sites, 10 Hz, ~25 days median duration (2014)
- **LEMI-424**: 19 sites, 1 Hz, ~55 days median duration (2016-2017)
- **Total**: 100 MT sites across Victoria

### AGRF Comparison
- Mean total field residual: -656 nT (measurements vs AGRF25 at 2020)
- Residuals indicate secular variation over 4-6 years
- Individual residuals may also reflect crustal magnetic anomalies

## Reference

These notebooks are preserved for reproducibility and to document the evolution of the analysis workflow. For current analysis, see the notebooks in the parent directory.
