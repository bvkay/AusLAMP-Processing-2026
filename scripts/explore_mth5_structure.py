"""
Explore MTH5 file structure to determine if we can use GA's MTH5 files directly.

Key questions:
1. Is time series data stored in MTH5 (not just metadata)?
2. Are calibration coefficients applied or stored separately?
3. Can we re-calibrate the data with correct Bartington Mag-03 coefficients?
"""

import h5py
import numpy as np
from pathlib import Path

# Path to a sample MTH5 file
mth5_file = Path(r'E:\MT_Timeseries_DATA\MT_AusLAMP_GA\EDL_MTH5\VIC001.h5')

def explore_h5_structure(h5_obj, prefix='', max_depth=5):
    """Recursively print HDF5 structure."""
    if max_depth == 0:
        return

    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ (Group)")
            explore_h5_structure(item, prefix + '  ', max_depth - 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}{key} (Dataset) shape={item.shape}, dtype={item.dtype}")
            # If small dataset, show attributes
            if item.size < 10:
                print(f"{prefix}  -> values: {item[()]}")

def main():
    print(f"Exploring: {mth5_file}")
    print("=" * 80)

    with h5py.File(mth5_file, 'r') as f:
        print("\n1. TOP-LEVEL STRUCTURE")
        print("-" * 80)
        explore_h5_structure(f, max_depth=3)

        # Navigate to station data
        stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
        if stations_path in f:
            stations = f[stations_path]
            station_name = list(stations.keys())[0]
            station = stations[station_name]

            print(f"\n2. STATION: {station_name}")
            print("-" * 80)
            print("Attributes:")
            for attr_name, attr_value in station.attrs.items():
                print(f"  {attr_name}: {attr_value}")

            # Look for time series data
            print(f"\n3. LOOKING FOR TIME SERIES DATA")
            print("-" * 80)

            # Check for Runs (time series segments)
            if 'runs' in station or 'Runs' in station:
                runs_group = station.get('runs', station.get('Runs'))
                print(f"Found 'runs' group with {len(runs_group)} runs")

                # Examine first run
                first_run = list(runs_group.keys())[0]
                run = runs_group[first_run]
                print(f"\nFirst run: {first_run}")
                print(f"Run structure:")
                explore_h5_structure(run, '  ', max_depth=3)

                # Look for channel data
                for possible_channel in ['BX', 'BY', 'BZ', 'EX', 'EY', 'bx', 'by', 'bz']:
                    if possible_channel in run:
                        channel = run[possible_channel]
                        print(f"\n  Channel '{possible_channel}':")
                        print(f"    Shape: {channel.shape}")
                        print(f"    Dtype: {channel.dtype}")
                        print(f"    First 10 samples: {channel[:10]}")
                        print(f"    Attributes:")
                        for attr_name, attr_value in channel.attrs.items():
                            print(f"      {attr_name}: {attr_value}")
                        break
            else:
                print("No 'runs' group found - data may not be in MTH5")
                print("\nAvailable groups in station:")
                for key in station.keys():
                    print(f"  {key}")

        print("\n4. CALIBRATION INFORMATION")
        print("-" * 80)
        # Look for calibration data
        if stations_path in f:
            station = f[stations_path][list(f[stations_path].keys())[0]]

            # Check for sensor information
            for possible_path in ['sensors', 'Sensors', 'channels', 'Channels']:
                if possible_path in station:
                    print(f"Found '{possible_path}' group")
                    sensor_group = station[possible_path]
                    explore_h5_structure(sensor_group, '  ', max_depth=2)

if __name__ == '__main__':
    main()
