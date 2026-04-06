"""
Unified MTH5 reader for AusLAMP Victoria EDL and LEMI-424 data.

Reads GA's MTH5 files with proper calibration:
- EDL: Applies correct Bartington Mag-03 calibration (GA's is wrong)
- LEMI-424: Data already calibrated, use as-is

Author: Claude Code
Date: 2026-04-06
"""

from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import warnings

# Import calibration constants from config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CalibrationConstants, DataPaths


def detect_instrument_type(mth5_file: Union[str, Path]) -> str:
    """
    Detect instrument type from MTH5 file.

    Parameters
    ----------
    mth5_file : str or Path
        Path to MTH5 file

    Returns
    -------
    str
        'EDL' or 'LEMI-424'
    """
    mth5_file = Path(mth5_file)

    with h5py.File(mth5_file, 'r') as f:
        # Check station metadata
        stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
        stations = f[stations_path]
        station_name = list(stations.keys())[0]
        station = stations[station_name]

        # Get first run
        runs = [k for k in station.keys() if k not in ['Transfer_Functions']]
        if not runs:
            raise ValueError(f"No runs found in {mth5_file}")

        run = station[runs[0]]

        # Check first channel for sensor manufacturer
        channels = [k for k in run.keys() if k in ['bx', 'by', 'bz', 'ex', 'ey']]
        if channels:
            ch = run[channels[0]]
            manufacturer = ch.attrs.get('sensor.manufacturer', b'unknown')
            if isinstance(manufacturer, bytes):
                manufacturer = manufacturer.decode()

            if 'LEMI' in manufacturer.upper():
                return 'LEMI-424'
            elif manufacturer == 'none':
                return 'EDL'

    # Fallback: check filename
    if 'R.h5' in mth5_file.name or 'LEMI' in str(mth5_file):
        return 'LEMI-424'
    else:
        return 'EDL'


def _decode_attr(attr):
    """Helper to decode HDF5 attributes (handles both bytes and str)."""
    if isinstance(attr, bytes):
        return attr.decode()
    elif hasattr(attr, '__iter__') and not isinstance(attr, str):
        return [item.decode() if isinstance(item, bytes) else item for item in attr]
    return attr


def get_station_metadata(mth5_file: Union[str, Path]) -> Dict:
    """
    Extract station metadata from MTH5 file.

    Parameters
    ----------
    mth5_file : str or Path
        Path to MTH5 file

    Returns
    -------
    dict
        Station metadata including:
        - station_id, instrument, latitude, longitude, elevation
        - declination, start_time, end_time, sample_rate
        - comments, channels_recorded
    """
    mth5_file = Path(mth5_file)

    with h5py.File(mth5_file, 'r') as f:
        stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
        stations = f[stations_path]
        station_name = list(stations.keys())[0]
        station = stations[station_name]

        # Extract station attributes
        metadata = {
            'station_id': _decode_attr(station.attrs.get('id', station_name)),
            'latitude': station.attrs.get('location.latitude', np.nan),
            'longitude': station.attrs.get('location.longitude', np.nan),
            'elevation': station.attrs.get('location.elevation', np.nan),
            'declination': station.attrs.get('location.declination.value', np.nan),
            'declination_model': _decode_attr(station.attrs.get('location.declination.model', 'unknown')),
            'comments': _decode_attr(station.attrs.get('comments', '')),
            'channels_recorded': _decode_attr(station.attrs.get('channels_recorded', [])),
        }

        # Get time range from runs
        runs = [k for k in station.keys() if k not in ['Transfer_Functions']]
        if runs:
            first_run = station[runs[0]]
            last_run = station[runs[-1]]

            start_time = _decode_attr(first_run.attrs.get('time_period.start', ''))
            end_time = _decode_attr(last_run.attrs.get('time_period.end', ''))
            sample_rate = first_run.attrs.get('sample_rate', np.nan)

            metadata['start_time'] = pd.Timestamp(start_time) if start_time else None
            metadata['end_time'] = pd.Timestamp(end_time) if end_time else None
            metadata['sample_rate'] = sample_rate
            metadata['n_runs'] = len(runs)

        # Detect instrument
        metadata['instrument'] = detect_instrument_type(mth5_file)

    return metadata


def get_all_metadata(mth5_dir: Union[str, Path], pattern: str = '*.h5') -> pd.DataFrame:
    """
    Extract metadata from all MTH5 files in a directory.

    Parameters
    ----------
    mth5_dir : str or Path
        Directory containing MTH5 files
    pattern : str, optional
        Glob pattern for MTH5 files (default: '*.h5')

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per station
    """
    mth5_dir = Path(mth5_dir)
    mth5_files = sorted(mth5_dir.glob(pattern))

    if not mth5_files:
        raise FileNotFoundError(f"No MTH5 files found in {mth5_dir} matching {pattern}")

    print(f"Found {len(mth5_files)} MTH5 files in {mth5_dir.name}")

    metadata_list = []
    for mth5_file in mth5_files:
        try:
            metadata = get_station_metadata(mth5_file)
            metadata['mth5_file'] = str(mth5_file)
            metadata_list.append(metadata)
        except Exception as e:
            warnings.warn(f"Failed to read {mth5_file.name}: {e}")

    df = pd.DataFrame(metadata_list)

    # Calculate duration
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['duration_days'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 86400

    print(f"Successfully extracted metadata from {len(df)} stations")

    return df


def read_station(
    mth5_file: Union[str, Path],
    channels: Union[str, List[str]] = 'magnetic',
    run_index: Optional[int] = None,
    time_slice: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    calibrate: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Read MT station data from GA MTH5 file.

    Automatically detects instrument type and applies appropriate calibration:
    - EDL: Applies correct Bartington Mag-03 calibration
    - LEMI-424: Uses pre-calibrated data

    Parameters
    ----------
    mth5_file : str or Path
        Path to MTH5 file
    channels : str or list, optional
        Channels to load:
        - 'all': All available channels
        - 'magnetic': BX, BY, BZ (default)
        - 'electric': EX, EY
        - List of channel names: ['BX', 'BY', ...]
    run_index : int, optional
        If multiple runs, load specific run (0-indexed). Default: load all runs concatenated
    time_slice : tuple of pd.Timestamp, optional
        (start, end) to load only part of data
    calibrate : bool, optional
        Apply calibration (default: True)
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with calibrated data:
        - Magnetic: nT (BX, BY, BZ)
        - Electric: µV/m (EX, EY) for LEMI, µV for EDL
        - Index: DatetimeIndex with timezone-aware timestamps
    """
    mth5_file = Path(mth5_file)

    if not mth5_file.exists():
        raise FileNotFoundError(f"MTH5 file not found: {mth5_file}")

    # Detect instrument type
    instrument = detect_instrument_type(mth5_file)

    if verbose:
        print(f"Reading {mth5_file.name} (Instrument: {instrument})")

    # Determine which channels to load
    if channels == 'all':
        requested_channels = ['bx', 'by', 'bz', 'ex', 'ey']
    elif channels == 'magnetic':
        requested_channels = ['bx', 'by', 'bz']
    elif channels == 'electric':
        requested_channels = ['ex', 'ey']
    elif isinstance(channels, list):
        requested_channels = [ch.lower() for ch in channels]
    else:
        raise ValueError(f"Invalid channels parameter: {channels}")

    with h5py.File(mth5_file, 'r') as f:
        # Navigate to station
        stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
        stations = f[stations_path]
        station_name = list(stations.keys())[0]
        station = stations[station_name]

        # Get runs
        runs = [k for k in station.keys() if k not in ['Transfer_Functions']]
        if not runs:
            raise ValueError(f"No runs found in {mth5_file}")

        # Select run(s) to load
        if run_index is not None:
            runs_to_load = [runs[run_index]]
        else:
            runs_to_load = runs

        if verbose:
            print(f"  Loading {len(runs_to_load)} run(s): {runs_to_load}")

        # Load data from each run
        run_dataframes = []

        for run_name in runs_to_load:
            run = station[run_name]

            # Get sample rate and time info
            sample_rate = float(run.attrs.get('sample_rate', 10.0))
            start_time_str = _decode_attr(run.attrs.get('time_period.start'))
            start_time = pd.Timestamp(start_time_str)

            # Load each channel
            channel_data = {}
            n_samples = None

            for ch_name in requested_channels:
                if ch_name not in run:
                    if verbose:
                        warnings.warn(f"Channel {ch_name} not found in run {run_name}, skipping")
                    continue

                ch_dataset = run[ch_name]
                ch_raw = ch_dataset[:]

                # Track number of samples (should be same for all channels)
                if n_samples is None:
                    n_samples = len(ch_raw)
                elif len(ch_raw) != n_samples:
                    warnings.warn(
                        f"Channel {ch_name} has {len(ch_raw)} samples, expected {n_samples}. "
                        f"Truncating to shortest."
                    )
                    n_samples = min(n_samples, len(ch_raw))

                # Apply calibration
                if calibrate:
                    if instrument == 'EDL':
                        # EDL: raw data in µV, need Bartington Mag-03 calibration
                        if ch_name in ['bx', 'by', 'bz']:
                            cal_factor = CalibrationConstants.BARTINGTON_MAG03[ch_name.upper()]
                            ch_calibrated = ch_raw * cal_factor  # µV → nT
                        else:
                            # Electric channels: keep as µV (need dipole length for mV/km)
                            ch_calibrated = ch_raw
                    else:  # LEMI-424
                        # Already calibrated: magnetic in nT, electric in µV/m
                        ch_calibrated = ch_raw
                else:
                    ch_calibrated = ch_raw

                channel_data[ch_name.upper()] = ch_calibrated[:n_samples]

            if not channel_data:
                raise ValueError(f"No valid channels loaded from run {run_name}")

            # Create time index
            time_index = pd.date_range(
                start=start_time,
                periods=n_samples,
                freq=pd.Timedelta(seconds=1/sample_rate)
            )

            # Create DataFrame for this run
            df_run = pd.DataFrame(channel_data, index=time_index)
            df_run.index.name = 'time'

            # Apply time slice if requested
            if time_slice is not None:
                df_run = df_run.loc[time_slice[0]:time_slice[1]]

            run_dataframes.append(df_run)

            if verbose:
                print(f"    Run {run_name}: {len(df_run):,} samples at {sample_rate} Hz")

    # Concatenate all runs
    df = pd.concat(run_dataframes)

    if verbose:
        print(f"  Total: {len(df):,} samples, {list(df.columns)}")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")

    return df


def read_all_stations(
    mth5_dir: Union[str, Path],
    pattern: str = '*.h5',
    channels: Union[str, List[str]] = 'magnetic',
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Read all MT stations from a directory of MTH5 files.

    Parameters
    ----------
    mth5_dir : str or Path
        Directory containing MTH5 files
    pattern : str, optional
        Glob pattern for MTH5 files (default: '*.h5')
    channels : str or list, optional
        Channels to load (default: 'magnetic')
    max_samples : int, optional
        Load only first N samples per station (for quick QA/QC)
    verbose : bool, optional
        Print progress messages (default: True)

    Returns
    -------
    dict
        Dictionary mapping station_id -> DataFrame
    """
    mth5_dir = Path(mth5_dir)
    mth5_files = sorted(mth5_dir.glob(pattern))

    if not mth5_files:
        raise FileNotFoundError(f"No MTH5 files found in {mth5_dir} matching {pattern}")

    if verbose:
        print(f"Loading {len(mth5_files)} stations from {mth5_dir.name}")
        print("=" * 80)

    stations = {}

    for i, mth5_file in enumerate(mth5_files, 1):
        station_id = mth5_file.stem  # VIC001, VIC065R, etc.

        try:
            df = read_station(mth5_file, channels=channels, verbose=False)

            # Apply sample limit if requested
            if max_samples is not None:
                df = df.iloc[:max_samples]

            stations[station_id] = df

            if verbose:
                duration = (df.index[-1] - df.index[0]).total_seconds() / 86400
                print(f"[{i}/{len(mth5_files)}] {station_id}: {len(df):,} samples, "
                      f"{duration:.1f} days, {list(df.columns)}")

        except Exception as e:
            if verbose:
                print(f"[{i}/{len(mth5_files)}] {station_id}: ERROR - {e}")

    if verbose:
        print("=" * 80)
        print(f"Successfully loaded {len(stations)}/{len(mth5_files)} stations")

    return stations


if __name__ == '__main__':
    """Test the reader with sample files."""

    print("Testing MTH5 Reader")
    print("=" * 80)

    # Test EDL file
    print("\n1. Testing EDL (VIC001):")
    edl_file = DataPaths.EDL_MTH5 / 'VIC001.h5'
    if edl_file.exists():
        metadata = get_station_metadata(edl_file)
        print(f"   Metadata: {metadata['station_id']}, {metadata['instrument']}, "
              f"{metadata['latitude']:.4f}, {metadata['longitude']:.4f}")

        df_edl = read_station(edl_file, channels='magnetic', verbose=True)
        print(f"\n   Sample data:")
        print(df_edl.head())
        print(f"\n   Statistics:")
        print(df_edl.describe())

    # Test LEMI file
    print("\n2. Testing LEMI-424 (VIC065R):")
    lemi_file = DataPaths.EDL_MTH5.parent / 'LEMI_MTH5' / 'VIC065R.h5'
    if lemi_file.exists():
        metadata = get_station_metadata(lemi_file)
        print(f"   Metadata: {metadata['station_id']}, {metadata['instrument']}, "
              f"{metadata['latitude']:.4f}, {metadata['longitude']:.4f}")

        df_lemi = read_station(lemi_file, channels='magnetic', verbose=True)
        print(f"\n   Sample data:")
        print(df_lemi.head())
        print(f"\n   Statistics:")
        print(df_lemi.describe())

    print("\n✓ MTH5 reader tests complete")
