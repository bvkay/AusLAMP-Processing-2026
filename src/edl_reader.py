"""
Optimized reader for Earth Data Logger (EDL) PR6-24 magnetometer data.

This version:
- Reads MiniSEED format files (SEED v2.3 with Steim1 compression) using ObsPy
- Scans directory once for all channels
- Sorts by filename timestamps (handles year rollover correctly)
- Loads all channels per timestamp together
- Extracts GPS metadata (lat/lon/elevation) from .gps files
- Applies hardware calibration for Bartington Mag-03 sensors

Based on: https://github.com/bvkay/mth5/blob/Lemi423/mth5/io/uoa/pr624.py
EDL Manual: Data stored as 24-bit samples in MiniSEED format (microvolts per bit)
"""

import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict
import re
from obspy import read as obspy_read


# Hardware calibration constants for Bartington Mag-03
CALIBRATION = {
    'BX': 0.007,   # nT per μV
    'BY': 0.007,   # nT per μV
    'BZ': 0.0175   # nT per μV (includes voltage divider correction: 0.007 × 2.5)
}

# GPS week rollover correction: 1024 weeks = 7168 days
GPS_WEEK_ROLLOVER_DAYS = 7168


def parse_edl_timestamp(filename, apply_gps_rollover=False):
    """
    Extract timestamp from EDL filename.

    Filename format: {STATION}_YYMMDDHHMMSS.{CHANNEL}
    Example: VIC001_140411060000.BX -> 2014-04-11 06:00:00

    Parameters
    ----------
    filename : str
        EDL filename
    apply_gps_rollover : bool, optional
        Apply GPS week rollover correction (+7168 days). Default: False.
        Set to True if data was collected with GPS week rollover bug.
        Victorian AusLAMP data (2014-2016) does NOT need this correction.

    Returns
    -------
    pd.Timestamp
        Start time extracted from filename
    """
    # Extract timestamp portion (YYMMDDHHMMSS) - always last 12 digits before extension
    # Handles both formats: VIC001_140411060000.BX and VIC002140513000000.BX
    basename = Path(filename).stem

    # Extract last 12 digits from basename
    match = re.search(r'(\d{12})$', basename)
    if not match:
        raise ValueError(f"Cannot extract timestamp from filename: {filename}")
    timestamp_str = match.group(1)

    # Parse: assume YY < 50 means 20YY, else 19YY
    year = int(timestamp_str[0:2])
    year = 2000 + year if year < 50 else 1900 + year
    month = int(timestamp_str[2:4])
    day = int(timestamp_str[4:6])
    hour = int(timestamp_str[6:8])
    minute = int(timestamp_str[8:10])
    second = int(timestamp_str[10:12])

    timestamp = pd.Timestamp(year=year, month=month, day=day,
                            hour=hour, minute=minute, second=second)

    # Apply GPS week rollover correction if requested
    if apply_gps_rollover:
        timestamp = timestamp + pd.Timedelta(days=GPS_WEEK_ROLLOVER_DAYS)

    return timestamp


def read_edl_station(data_dir, sample_rate=None, channels='all', verbose=False):
    """
    Read EDL MiniSEED files from a station directory.

    Expected structure:
    - Day folders (001-366): data/337/VIC001_041202060000.BX
    - Flat directory: data/VIC001_041202060000.BX

    Parameters
    ----------
    data_dir : str or Path
        Directory containing EDL data files
    sample_rate : float, optional
        Expected sample rate in Hz (default: None, auto-detect from data)
        Set to 10.0 for Victorian AusLAMP data if auto-detection fails
    channels : str or list, optional
        Channels to load: 'all', 'magnetic' (BX,BY,BZ), 'electric' (EX,EY),
        or list like ['BX', 'BY', 'BZ', 'EX', 'EY']. Default: 'all'
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with requested channels
        Magnetic channels in nT (calibrated), electric in μV (raw ADC)

    Notes
    -----
    EDL stores data in MiniSEED format (SEED v2.3, Steim1 compression).
    Data values are in microvolts (μV). Calibration is applied for magnetic channels.
    """
    data_dir = Path(data_dir)

    # Determine which channels to load
    if channels == 'all':
        requested_channels = ['BX', 'BY', 'BZ', 'EX', 'EY', 'TP']
    elif channels == 'magnetic':
        requested_channels = ['BX', 'BY', 'BZ']
    elif channels == 'electric':
        requested_channels = ['EX', 'EY']
    elif isinstance(channels, list):
        requested_channels = [ch.upper() for ch in channels]
    else:
        raise ValueError(f"Invalid channels parameter: {channels}")

    # Find all channel files
    all_files = list(data_dir.rglob("*.[BbEeTtXxYyZzPp][XxYyZzPp]"))

    if not all_files:
        raise FileNotFoundError(f"No EDL data files found in {data_dir}")

    if verbose:
        print(f"Found {len(all_files)} files in {data_dir.name}")

    # Group files by timestamp and channel
    files_by_time = defaultdict(dict)

    for filepath in all_files:
        channel = filepath.suffix[1:].upper()  # .BX -> BX, .EX -> EX, .TP -> TP
        if channel not in requested_channels:
            continue

        timestamp = parse_edl_timestamp(filepath.name)
        files_by_time[timestamp][channel] = filepath

    # Sort timestamps
    sorted_times = sorted(files_by_time.keys())

    if verbose:
        print(f"Time range: {sorted_times[0]} to {sorted_times[-1]}")
        print(f"Loading {len(sorted_times)} time segments...")

    # Load data using ObsPy (MiniSEED reader)
    channel_streams = {ch: [] for ch in requested_channels}

    for timestamp in sorted_times:
        files = files_by_time[timestamp]

        for channel in requested_channels:
            if channel not in files:
                continue

            try:
                # Read MiniSEED file with ObsPy
                stream = obspy_read(str(files[channel]))

                if len(stream) > 0:
                    trace = stream[0]  # Get first (and usually only) trace
                    channel_streams[channel].append(trace)

                    if verbose and timestamp == sorted_times[0]:
                        print(f"  {channel}: {trace.stats.npts} samples at {trace.stats.sampling_rate} Hz")

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load {channel} at {timestamp}: {e}")
                continue

    # Merge all traces for each channel
    df_data = {}

    for channel in requested_channels:
        if not channel_streams[channel]:
            continue

        try:
            # Merge all traces for this channel
            from obspy import Stream
            stream = Stream(traces=channel_streams[channel])
            stream.merge(method=1, fill_value=0)  # Merge with zero-fill for gaps

            if len(stream) > 0:
                trace = stream[0]

                # Extract data (in microvolts from EDL)
                data_uv = trace.data

                # Apply calibration
                if channel in ['BX', 'BY', 'BZ']:
                    # Convert μV to nT using Bartington Mag-03 calibration
                    data_calibrated = data_uv * CALIBRATION[channel]
                elif channel in ['EX', 'EY']:
                    # Keep as μV for now (needs dipole length for mV/km conversion)
                    data_calibrated = data_uv
                else:
                    # TP (temperature) - keep raw
                    data_calibrated = data_uv

                df_data[channel] = data_calibrated

                # Use ObsPy's time index for the first channel
                if 'time_index' not in locals():
                    time_index = pd.DatetimeIndex([trace.stats.starttime.datetime +
                                                   pd.Timedelta(seconds=i/trace.stats.sampling_rate)
                                                   for i in range(len(data_calibrated))])
                    detected_sample_rate = trace.stats.sampling_rate

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to merge {channel}: {e}")
            continue

    if not df_data:
        raise ValueError(f"No valid data loaded from {data_dir}")

    # Create DataFrame
    df = pd.DataFrame(df_data, index=time_index)
    df.index.name = 'timestamp'

    if verbose:
        print(f"Loaded {len(df):,} samples at {detected_sample_rate} Hz")
        print(f"Channels: {list(df.columns)}")
        if sample_rate and abs(detected_sample_rate - sample_rate) > 0.01:
            print(f"Warning: Expected {sample_rate} Hz, got {detected_sample_rate} Hz")

    return df


def read_edl_parallel_test(base_dir, sample_rate=10.0, verbose=True):
    """
    Read all EDL instruments from parallel test.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing instrument folders (e.g., R01, R02, ...)
    sample_rate : float, optional
        Sample rate in Hz (default: 10.0 Hz for AusLAMP Victorian data)
    verbose : bool, optional
        Print progress messages (default: True)

    Returns
    -------
    dict
        Dictionary mapping instrument ID -> DataFrame
        Each DataFrame has columns [Bx, By, Bz] in nT
    """
    base_dir = Path(base_dir)

    # Find all R* folders (exclude Lemi-424 folders)
    data_folders = sorted([d for d in base_dir.glob("R*")
                          if d.is_dir() and not d.name.startswith('DATA')])

    if verbose:
        print(f"Found {len(data_folders)} EDL instruments")

    instruments = {}
    for folder in data_folders:
        instrument_id = folder.name
        try:
            df = read_edl_station(folder, sample_rate=sample_rate, verbose=verbose)
            instruments[instrument_id] = df
            if verbose:
                print(f"✓ {instrument_id}: {len(df):,} samples, "
                      f"{df.index[0]} to {df.index[-1]}\n")
        except Exception as e:
            if verbose:
                print(f"✗ {instrument_id}: {e}\n")

    return instruments


def parse_gps_file(gps_file_path):
    """
    Parse GPS coordinates from EDL .gps file.

    Extracts latitude, longitude, and elevation from GPS messages in the file.

    Expected message formats:
    - >RAL12032+00046+00012;*43< (Altitude: second field)
    - >RPV11580-3452843+1378762700000012;*75< (Lat/Lon: first two fields after RPV)

    Parameters
    ----------
    gps_file_path : str or Path
        Path to .gps file

    Returns
    -------
    dict
        Dictionary with keys: 'latitude', 'longitude', 'elevation' (all in decimal degrees/meters)
        Returns NaN for any values not found
    """
    gps_data = {
        'latitude': np.nan,
        'longitude': np.nan,
        'elevation': np.nan
    }

    gps_file_path = Path(gps_file_path)
    if not gps_file_path.exists():
        return gps_data

    try:
        with open(gps_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 100:  # Only read first 100 lines
                    break

                # Parse altitude from RAL message
                # Format: >RAL12032+00046+00012;*43<
                if '>RAL' in line:
                    match = re.search(r'>RAL\d+([+-]\d+)', line)
                    if match:
                        gps_data['elevation'] = float(match.group(1))

                # Parse lat/lon from RPV message
                # Format: >RPV11580-3452843+1378762700000012;*75<
                if '>RPV' in line:
                    match = re.search(r'>RPV\d+([+-]\d+)([+-]\d+)', line)
                    if match and len(match.groups()) == 2:
                        lat_raw = float(match.group(1))
                        lon_raw = float(match.group(2))

                        # Latitude: divide by 1e5
                        gps_data['latitude'] = lat_raw / 1e5

                        # Longitude: divide by 1e7, then correct if out of range
                        gps_data['longitude'] = lon_raw / 1e7

                        # Ensure longitude is in valid range (-180 to +180)
                        while abs(gps_data['longitude']) > 180:
                            gps_data['longitude'] = gps_data['longitude'] / 10

                        # Ensure latitude is in valid range (-90 to +90)
                        while abs(gps_data['latitude']) > 90:
                            gps_data['latitude'] = gps_data['latitude'] / 10

                # Stop if we've found all values
                if not np.isnan(gps_data['latitude']) and \
                   not np.isnan(gps_data['longitude']) and \
                   not np.isnan(gps_data['elevation']):
                    break

    except Exception as e:
        print(f"Warning: Error parsing GPS file {gps_file_path}: {e}")

    return gps_data


def get_station_metadata(station_dir):
    """
    Extract metadata for an EDL station from directory structure and GPS files.

    Parameters
    ----------
    station_dir : str or Path
        Directory containing station data

    Returns
    -------
    dict
        Dictionary with metadata:
        - station_id: Station identifier (e.g., 'VIC001')
        - latitude, longitude, elevation: GPS coordinates
        - start_time, end_time: Recording period
        - n_days: Number of day folders
        - n_files: Total number of data files
    """
    station_dir = Path(station_dir)

    metadata = {
        'station_id': station_dir.name,
        'latitude': np.nan,
        'longitude': np.nan,
        'elevation': np.nan,
        'start_time': None,
        'end_time': None,
        'n_days': 0,
        'n_files': 0
    }

    # Find GPS file
    gps_files = list(station_dir.rglob("*.gps"))
    if gps_files:
        gps_data = parse_gps_file(gps_files[0])
        metadata.update(gps_data)

    # Count day folders and files
    day_folders = [d for d in station_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    metadata['n_days'] = len(day_folders)

    # Find all data files to get time range
    all_files = list(station_dir.rglob("*.[BbXxYyZz][XxYyZz]"))
    metadata['n_files'] = len(all_files)

    if all_files:
        try:
            # Extract timestamps and find first/last files
            file_times = []
            for f in all_files:
                try:
                    ts = parse_edl_timestamp(f.name)
                    file_times.append((ts, f))
                except:
                    continue

            if file_times:
                # Sort by timestamp
                file_times.sort(key=lambda x: x[0])

                # Start time: first file's timestamp
                metadata['start_time'] = file_times[0][0]

                # End time: last file's timestamp + duration of that file
                last_time, last_file = file_times[-1]

                try:
                    # Try to determine file duration
                    # EDL files are binary: 24-bit samples = 3 bytes per sample
                    file_size = last_file.stat().st_size
                    n_samples = file_size // 3  # 3 bytes per 24-bit sample

                    # Calculate duration at 10 Hz
                    duration_seconds = n_samples / 10.0
                    metadata['end_time'] = last_time + pd.Timedelta(seconds=duration_seconds)
                except:
                    # If we can't determine file size, just use the start time of last file
                    metadata['end_time'] = last_time

        except Exception as e:
            print(f"Warning: Could not determine time range for {station_dir.name}: {e}")

    return metadata


def get_all_stations_metadata(base_dir, station_pattern="VIC*"):
    """
    Get metadata for all stations in a directory.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing station folders
    station_pattern : str, optional
        Glob pattern for station folders (default: "VIC*")

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per station, columns from get_station_metadata()
    """
    base_dir = Path(base_dir)

    # Find all matching station directories
    station_dirs = sorted([d for d in base_dir.glob(station_pattern)
                          if d.is_dir() and not d.name.endswith('.zip')])

    metadata_list = []
    for station_dir in station_dirs:
        print(f"Reading metadata: {station_dir.name}...")
        metadata = get_station_metadata(station_dir)
        metadata_list.append(metadata)

    df = pd.DataFrame(metadata_list)

    # Calculate recording duration
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['duration_days'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 86400

    return df
