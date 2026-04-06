"""
Extract metadata from MTH5 files.

MTH5 is an HDF5-based format for magnetotelluric data. This module extracts
station-level metadata (coordinates, elevation, declination) from MTH5 files.
"""

import h5py
from pathlib import Path
import pandas as pd


def extract_station_metadata(mth5_file):
    """
    Extract station metadata from a single MTH5 file.

    Parameters
    ----------
    mth5_file : Path or str
        Path to MTH5 (.h5) file

    Returns
    -------
    dict or None
        Dictionary with metadata fields, or None if extraction fails:
        - station_id: Station identifier (e.g., 'VIC001')
        - latitude: Decimal degrees
        - longitude: Decimal degrees
        - elevation: Meters
        - declination: Degrees (magnetic declination)
        - declination_model: Model used (e.g., 'AGRF')
        - start_time: ISO 8601 timestamp string
        - end_time: ISO 8601 timestamp string
    """
    mth5_file = Path(mth5_file)

    try:
        with h5py.File(mth5_file, 'r') as f:
            # Navigate to stations
            stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
            if stations_path not in f:
                print(f'WARNING: {mth5_file.name} does not have expected path {stations_path}')
                return None

            stations = f[stations_path]

            # Get first (and usually only) station
            station_names = list(stations.keys())
            if len(station_names) == 0:
                print(f'WARNING: {mth5_file.name} has no stations')
                return None

            station_name = station_names[0]
            station = stations[station_name]

            # Extract metadata from attributes
            attrs = station.attrs

            metadata = {
                'station_id': attrs.get('id', station_name),
                'latitude': attrs.get('location.latitude', None),
                'longitude': attrs.get('location.longitude', None),
                'elevation': attrs.get('location.elevation', None),
                'declination': attrs.get('location.declination.value', None),
                'declination_model': attrs.get('location.declination.model', None),
                'start_time': attrs.get('time_period.start', None),
                'end_time': attrs.get('time_period.end', None),
                'acquired_by': attrs.get('acquired_by.name', None),
                'data_type': attrs.get('data_type', None),
                'geographic_name': attrs.get('geographic_name', None),
                'orientation_method': attrs.get('orientation.method', None),
                'orientation_frame': attrs.get('orientation.reference_frame', None),
                'comments': attrs.get('comments', None),
                'channels_recorded': attrs.get('channels_recorded', None),
            }

            # Decode bytes to strings or convert arrays to comma-separated strings
            for key, val in metadata.items():
                if isinstance(val, bytes):
                    metadata[key] = val.decode('utf-8')
                elif hasattr(val, '__iter__') and not isinstance(val, str):
                    # Convert numpy arrays or lists to comma-separated strings
                    try:
                        # Handle byte arrays in the list
                        decoded_items = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in val]
                        metadata[key] = ','.join(decoded_items)
                    except:
                        metadata[key] = str(val)

            return metadata

    except Exception as e:
        print(f'ERROR reading {mth5_file.name}: {e}')
        return None


def extract_all_metadata(mth5_dir, pattern='VIC*.h5'):
    """
    Extract metadata from all MTH5 files in a directory.

    Parameters
    ----------
    mth5_dir : Path or str
        Directory containing MTH5 files
    pattern : str, optional
        Glob pattern for MTH5 files (default: 'VIC*.h5')

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per station, columns:
        - station_id, latitude, longitude, elevation, declination,
          declination_model, start_time, end_time, acquired_by,
          data_type, geographic_name, orientation_method, orientation_frame,
          comments, channels_recorded
    """
    mth5_dir = Path(mth5_dir)

    if not mth5_dir.exists():
        raise FileNotFoundError(f'MTH5 directory not found: {mth5_dir}')

    mth5_files = sorted(mth5_dir.glob(pattern))
    print(f'Found {len(mth5_files)} MTH5 files in {mth5_dir}')

    metadata_list = []

    for mth5_file in mth5_files:
        metadata = extract_station_metadata(mth5_file)
        if metadata is not None:
            metadata_list.append(metadata)

    print(f'Successfully extracted metadata from {len(metadata_list)} files')

    return pd.DataFrame(metadata_list)
