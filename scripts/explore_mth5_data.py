"""
Deep dive into MTH5 time series data and calibration info.
"""

import h5py
import numpy as np
from pathlib import Path

mth5_file = Path(r'E:\MT_Timeseries_DATA\MT_AusLAMP_GA\EDL_MTH5\VIC001.h5')

def main():
    print(f"Exploring time series data in: {mth5_file.name}")
    print("=" * 80)

    with h5py.File(mth5_file, 'r') as f:
        # Navigate to station
        station = f['Experiment/Surveys/AusLAMP_VIC/Stations/VIC001']

        # Get the run (named by timestamp)
        run_name = '20140411060000'
        if run_name in station:
            run = station[run_name]

            print(f"\nRUN: {run_name}")
            print("-" * 80)
            print("Run attributes:")
            for attr_name, attr_value in run.attrs.items():
                print(f"  {attr_name}: {attr_value}")

            print(f"\nRun contents:")
            for key in run.keys():
                print(f"  {key}")

            # Check each channel
            print(f"\nCHANNEL DATA:")
            print("-" * 80)

            for channel_name in ['bx', 'by', 'bz', 'ex', 'ey']:
                if channel_name in run:
                    channel = run[channel_name]

                    print(f"\n{channel_name.upper()}:")
                    print(f"  Shape: {channel.shape}")
                    print(f"  Dtype: {channel.dtype}")
                    print(f"  Size: {channel.size:,} samples")
                    print(f"  First 5 values: {channel[:5]}")
                    print(f"  Mean: {channel[:].mean():.2f}")
                    print(f"  Std: {channel[:].std():.2f}")

                    print(f"  Attributes:")
                    for attr_name in channel.attrs.keys():
                        attr_value = channel.attrs[attr_name]
                        # Truncate long strings
                        if isinstance(attr_value, (bytes, str)) and len(str(attr_value)) > 100:
                            print(f"    {attr_name}: {str(attr_value)[:100]}...")
                        else:
                            print(f"    {attr_name}: {attr_value}")

                    # Check for calibration info
                    print(f"  Looking for calibration...")
                    for attr in ['calibration', 'calibration_applied', 'sensor',
                                'units', 'scale_factor', 'gain']:
                        if attr in channel.attrs:
                            print(f"    FOUND: {attr} = {channel.attrs[attr]}")

            # Check if there's a separate calibration group
            print(f"\n\nCALIBRATION/SENSOR INFO:")
            print("-" * 80)

            # Check station-level sensor info
            if 'sensors' in station or 'Sensors' in station:
                sensors = station.get('sensors', station.get('Sensors'))
                print("Found sensor group at station level:")
                for sensor_name in sensors.keys():
                    print(f"  {sensor_name}")
                    sensor = sensors[sensor_name]
                    for attr_name, attr_value in sensor.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
            else:
                print("No sensor group found at station level")

            # Check run-level sensor info
            for possible in ['sensors', 'Sensors', 'channels', 'Channels']:
                if possible in run:
                    print(f"\nFound '{possible}' in run:")
                    group = run[possible]
                    for key in group.keys():
                        print(f"  {key}")

            # Look for filter/calibration in run
            for key in run.keys():
                if 'filter' in key.lower() or 'calib' in key.lower():
                    print(f"\nFound potential calibration: {key}")
                    item = run[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  Dataset shape: {item.shape}")
                    elif isinstance(item, h5py.Group):
                        print(f"  Group contents: {list(item.keys())}")

if __name__ == '__main__':
    main()
