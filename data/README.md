# Data Directory

This directory contains local project data. All subdirectories are git-ignored.

## Structure

```
data/
├── raw/                        # Reference data (tracked in git)
│   └── eCat_150056_AGRF25/    # Australian Geomagnetic Reference Field model
│
├── processed/                  # Generated metadata and QA/QC results (git-ignored)
│   └── metadata_all_sites.csv # Combined metadata from all 100 sites
│
└── mt_results/                 # MT processing outputs (git-ignored)
    └── (future: impedance tensors, transfer functions)
```

## External Data Sources

The actual MT time series data lives on external storage (E: drive):
- `E:\MT_Timeseries_DATA\MT_AusLAMP_GA\EDL_MTH5\` (81 EDL sites)
- `E:\MT_Timeseries_DATA\MT_AusLAMP_GA\LEMI_MTH5\` (19 LEMI-424 sites)

Configure path with environment variable:
```bash
export AUSLAMP_DATA_ROOT="/path/to/data"
```

## AGRF Model

**Australian Geomagnetic Reference Field 2025** (AGRF25)
- Location: `raw/eCat_150056_AGRF25/`
- Executable: `AGRF25S.EXE`
- Valid: 2020-2030 (optimized for 2025-2030)
- Source: Geoscience Australia eCatalog Record 150056

Used via Python wrapper: `src/models/agrf.py` (to be created)
