# AusLAMP Victoria Magnetotelluric Data Analysis

Magnetotelluric (MT) data analysis for 100 sites across Victoria, Australia:
- **81 EDL sites** (Earth Data Logger with Bartington Mag-03, 10 Hz, 2014)
- **19 LEMI-424 sites** (LEMI-424 fluxgate, 1 Hz, 2016-2017)

**Data source**: Geoscience Australia MTH5 files
**GitHub**: https://github.com/bvkay/AusLAMP-Vic

---

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate AusLAMP_GA

# Register Jupyter kernel
python -m ipykernel install --user --name=AusLAMP_GA --display-name="Python (AusLAMP_GA)"
```

### 2. Run Demo Notebook

```bash
cd notebooks
jupyter notebook 01_quick_start_demo.ipynb
```

Select **Kernel → Change kernel → Python (AusLAMP_GA)** in Jupyter.

---

## Project Structure

```
AusLAMP-Vic/
├── src/
│   ├── config.py                    # Path configuration
│   ├── readers/
│   │   ├── __init__.py
│   │   └── mth5_reader.py          # Unified MTH5 reader (EDL + LEMI-424)
│   └── (future: qaqc/, processing/, plotting/)
│
├── notebooks/
│   ├── 01_quick_start_demo.ipynb   # Demonstrates unified reader
│   └── (future: metadata, QA/QC, MT processing)
│
├── notebooks_archive/               # Original exploratory notebooks (pre-refactor)
│   ├── README.md                    # Documentation of archived work
│   └── old_*.ipynb                  # 5 archived notebooks
│
├── scripts/                         # Standalone Python scripts
│   ├── explore_mth5_structure.py
│   └── test_mth5_direct.py
│
├── data/                            # Local processed data (git-ignored)
│   ├── raw/
│   │   └── eCat_150056_AGRF25/     # AGRF25 geomagnetic model
│   └── processed/
│       └── metadata_all_sites.csv  # Combined metadata (future)
│
├── outputs/                         # Figures and reports (git-ignored)
│   ├── figures/
│   └── reports/
│
├── tests/                           # Unit tests (future)
├── environment.yml                  # Conda environment
├── .gitignore
└── README.md                        # This file
```

---

## Data Location

**External data** (E: drive, read-only):
- `E:\MT_Timeseries_DATA\MT_AusLAMP_GA\EDL_MTH5\` — 81 EDL MTH5 files
- `E:\MT_Timeseries_DATA\MT_AusLAMP_GA\LEMI_MTH5\` — 19 LEMI-424 MTH5 files

**Local reference data**:
- `data/raw/eCat_150056_AGRF25/` — Australian Geomagnetic Reference Field model

Override data location with environment variable:
```bash
export AUSLAMP_DATA_ROOT="/path/to/data"
```

---

## Key Features

### Unified MTH5 Reader
- **Single interface** for both EDL and LEMI-424 instruments
- **Automatic instrument detection** and calibration
- **Correct Bartington Mag-03 calibration** for EDL (GA's is wrong)
- **Read directly from GA's MTH5 files** (no ASCII conversion needed)
- **Handles multiple runs** (LEMI-424 long deployments)

### Example Usage

```python
from src.readers import read_station, get_station_metadata

# Read EDL station
df_edl = read_station('E:/MT_Timeseries_DATA/.../EDL_MTH5/VIC001.h5')
# Returns: DataFrame with BX, BY, BZ in nT (calibrated)

# Read LEMI-424 station
df_lemi = read_station('E:/MT_Timeseries_DATA/.../LEMI_MTH5/VIC065R.h5')
# Returns: DataFrame with BX, BY, BZ in nT (already calibrated)

# Get metadata without loading full dataset
metadata = get_station_metadata('VIC001.h5')
```

---

## MTH5 Data Format

### EDL (Bartington Mag-03)
- **Raw data**: Uncalibrated µV in MTH5 (GA labeled it wrong as "celsius degrees")
- **Calibration** (applied by reader):
  - BX, BY: 0.007 nT/µV
  - BZ: 0.0175 nT/µV (includes 2.5× voltage divider)
- **Sample rate**: 10 Hz
- **Channels**: BX, BY, BZ (magnetic), EX, EY (electric)

### LEMI-424
- **Raw data**: Already calibrated in nT and µV/m
- **Sample rate**: 1 Hz
- **Channels**: BX, BY, BZ (magnetic), EX, EY (electric), temperature_e, temperature_h
- **Multiple runs**: Long deployments with battery swaps

---

## Dataset Summary

| Instrument | Sites | Sample Rate | Duration (median) | Year |
|------------|-------|-------------|-------------------|------|
| EDL        | 81    | 10 Hz       | ~25 days          | 2014 |
| LEMI-424   | 19    | 1 Hz        | ~55 days          | 2016-2017 |
| **Total**  | **100** | —         | —                 | —    |

---

## Refactoring History

**April 2026**: Major refactoring to use GA's MTH5 files as primary data source.

**Before** (archived in `notebooks_archive/`):
- Read MiniSEED files with ObsPy
- Exported to ASCII (.dat files)
- Inline LEMI-424 reader in notebooks
- Manual metadata extraction

**After** (current):
- Read MTH5 files directly with h5py
- Unified reader for both instruments
- Modular architecture with proper config
- Correct calibration applied

See [`notebooks_archive/README.md`](notebooks_archive/README.md) for details.

---

## Dependencies

**Core**: Python 3.11, numpy, scipy, pandas, matplotlib
**Data I/O**: h5py, xarray, obspy
**Geospatial**: cartopy, geopandas, folium
**Testing**: pytest
**MT libraries**: mth5, mt-metadata (optional, not currently used)

See [`environment.yml`](environment.yml) for complete specification.

---

## Next Steps

1. **Build metadata CSV** — Extract metadata from all 100 MTH5 files
2. **QA/QC analysis** — Statistics, outlier detection, data quality flags
3. **Site visualization** — Location maps, recording timelines, AGRF comparison
4. **MT processing** — Remote reference selection, impedance tensor estimation
5. **Unit tests** — Test suite for readers and processing functions

---

## Documentation

See project-specific documentation (git-ignored):
- **CLAUDE.md** — Detailed project context for AI assistant
- **notebooks_archive/README.md** — History of exploratory analysis phase

---

## Citation

Data provided by Geoscience Australia:
- **EDL Survey**: AusLAMP Victoria, 2014 (eCat Record 2018.021)
- **LEMI-424 Survey**: AusLAMP Victoria, 2016-2017

**AGRF25 Model**: Geoscience Australia, Australian Geomagnetic Reference Field 2025
DOI: 10.11636/Record.2020.XXX (check GA website for correct DOI)

---

## License

[Specify license - typically CC-BY-4.0 for GA data]

---

## Contact

Project maintained by: [Your name/organization]
For data access: Geoscience Australia Client Services (ClientServices@ga.gov.au)
