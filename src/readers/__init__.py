"""
MT data readers for AusLAMP Victoria project.

Provides unified interface for reading both EDL and LEMI-424 data from MTH5 files.
"""

from .mth5_reader import (
    read_station,
    read_all_stations,
    get_station_metadata,
    get_all_metadata,
    detect_instrument_type
)

__all__ = [
    'read_station',
    'read_all_stations',
    'get_station_metadata',
    'get_all_metadata',
    'detect_instrument_type'
]
