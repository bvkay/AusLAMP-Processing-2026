"""
Configuration management for AusLAMP Victoria MT project.

Environment variables can override default paths:
- AUSLAMP_DATA_ROOT: Root directory for MT data (default: E:\\MT_Timeseries_DATA\\MT_AusLAMP_GA)
- AUSLAMP_AGRF_DIR: Directory containing AGRF25S.EXE (default: project_root/data/raw/eCat_150056_AGRF25)
"""

from pathlib import Path
import os

# Allow environment override for data root
DATA_ROOT = Path(os.getenv('AUSLAMP_DATA_ROOT', r'E:\MT_Timeseries_DATA\MT_AusLAMP_GA'))

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


class DataPaths:
    """Centralized path configuration for all data sources."""

    # Raw data (read-only, on external drive)
    EDL_MTH5 = DATA_ROOT / 'EDL_MTH5'
    LEMI_MTH5 = DATA_ROOT / 'LEMI_MTH5'

    # Legacy raw data (if needed)
    LEMI_RAW = DATA_ROOT / 'LEMI_raw'  # Original TXT files (superseded by LEMI_MTH5)

    # Deprecated/legacy paths (if still needed)
    EDL_RAW_MINISEED = DATA_ROOT / 'EDL_raw'  # Original MiniSEED files
    EDL_EXPORTED_ASCII = DATA_ROOT / 'EDL_2026'  # ASCII exports (no longer primary)

    # Project data directories (read-write, in project)
    PROJECT_DATA = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = PROJECT_DATA / 'raw'
    PROCESSED_DATA_DIR = PROJECT_DATA / 'processed'
    MT_RESULTS_DIR = PROJECT_DATA / 'mt_results'

    # AGRF model (can be in project or specified separately)
    AGRF_DIR = Path(os.getenv(
        'AUSLAMP_AGRF_DIR',
        RAW_DATA_DIR / 'eCat_150056_AGRF25'
    ))

    # Processed data products
    METADATA_DB = PROCESSED_DATA_DIR / 'metadata.db'
    QAQC_DIR = PROCESSED_DATA_DIR / 'qaqc'
    QAQC_SUMMARY = QAQC_DIR / 'qaqc_summary.csv'

    # Output directories
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'
    FIGURES_DIR = OUTPUT_DIR / 'figures'
    REPORTS_DIR = OUTPUT_DIR / 'reports'

    @classmethod
    def create_directories(cls):
        """Create all necessary project directories if they don't exist."""
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR'):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ {attr_name}: {path}")

    @classmethod
    def validate_external_data(cls):
        """Check that external data sources (E: drive) are accessible."""
        missing = []

        if not cls.EDL_MTH5.exists():
            missing.append(f"EDL_MTH5: {cls.EDL_MTH5}")

        if not cls.LEMI_MTH5.exists():
            missing.append(f"LEMI_MTH5: {cls.LEMI_MTH5}")

        if missing:
            print("WARNING: External data directories not found:")
            for path in missing:
                print(f"  ✗ {path}")
            print("\nSet AUSLAMP_DATA_ROOT environment variable if data is in different location.")
            return False

        print("✓ External data directories accessible")
        return True


class CalibrationConstants:
    """Hardware calibration constants for magnetometers."""

    # Bartington Mag-03 Fluxgate Magnetometer (used with EDL)
    # These are the CORRECT coefficients (not GA's incorrect values)
    BARTINGTON_MAG03 = {
        'BX': 0.007,   # nT per μV (70,000 nT over ±10V)
        'BY': 0.007,   # nT per μV (70,000 nT over ±10V)
        'BZ': 0.0175   # nT per μV (70,000 nT over ±4V, includes 2.5× voltage divider)
    }

    # LEMI-424 (data already calibrated in TXT files)
    LEMI424_CALIBRATED = True


class CoordinateSystems:
    """Coordinate system conventions and transformations."""

    # EDL sensor orientation (from MTH5 metadata)
    EDL_ORIENTATION_FRAME = 'Geomagnetic'  # Per MTH5: orientation.reference_frame
    EDL_ORIENTATION_METHOD = 'L shaped'    # Per MTH5: orientation.method

    # Standard geomagnetic convention (INTERMAGNET, IGRF)
    GEOMAGNETIC_CONVENTION = {
        'X': 'North',
        'Y': 'East',
        'Z': 'Down (positive downward)'
    }

    # Note: Some EDL sites have swapped channels
    # VIC001: "channels 0-2 (magnetic fields) and 3-5 (electric fields) were swapped"
    # This is documented in MTH5 station.comments attribute


if __name__ == '__main__':
    print("AusLAMP Victoria MT Project - Configuration")
    print("=" * 80)
    print(f"\nData root: {DATA_ROOT}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"\nExternal data sources:")
    print(f"  EDL MTH5: {DataPaths.EDL_MTH5}")
    print(f"  LEMI raw: {DataPaths.LEMI_RAW}")
    print(f"\nAGRF model:")
    print(f"  {DataPaths.AGRF_DIR}")

    print(f"\nValidating external data...")
    DataPaths.validate_external_data()

    print(f"\nCreating project directories...")
    DataPaths.create_directories()

    print(f"\nCalibration constants:")
    print(f"  Bartington Mag-03:")
    for ch, cal in CalibrationConstants.BARTINGTON_MAG03.items():
        print(f"    {ch}: {cal} nT/μV")
