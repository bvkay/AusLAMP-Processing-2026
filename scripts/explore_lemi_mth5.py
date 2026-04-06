"""
Explore LEMI-424 MTH5 structure to see if data is calibrated.
"""

from pathlib import Path
import h5py
import numpy as np

lemi_mth5 = Path(r'E:\MT_Timeseries_DATA\MT_AusLAMP_GA\LEMI_MTH5\VIC065R.h5')

print("=" * 80)
print(f"Exploring LEMI-424 MTH5: {lemi_mth5.name}")
print("=" * 80)

with h5py.File(lemi_mth5, 'r') as f:
    # Check structure
    print("\n1. Top-level groups:")
    print(f"   {list(f.keys())}")

    # Navigate to station
    stations_path = 'Experiment/Surveys/AusLAMP_VIC/Stations'
    if stations_path in f:
        stations = f[stations_path]
        station_name = list(stations.keys())[0]
        station = stations[station_name]

        print(f"\n2. Station: {station_name}")
        print("   Key attributes:")
        for attr in ['id', 'location.latitude', 'location.longitude',
                     'location.elevation', 'location.declination.value',
                     'comments', 'channels_recorded']:
            if attr in station.attrs:
                val = station.attrs[attr]
                # Decode bytes
                if isinstance(val, bytes):
                    val = val.decode()
                elif hasattr(val, '__iter__') and not isinstance(val, str):
                    val = [v.decode() if isinstance(v, bytes) else v for v in val]
                print(f"     {attr}: {val}")

        # Check runs
        run_list = [k for k in station.keys() if k not in ['Transfer_Functions']]
        print(f"\n3. Runs: {run_list}")

        if run_list:
            run_name = run_list[0]
            run = station[run_name]

            print(f"\n4. Run: {run_name}")
            print(f"   Channels: {list(run.keys())}")

            # Check BX channel
            if 'bx' in run:
                bx = run['bx']
                print(f"\n5. BX Channel:")
                print(f"   Shape: {bx.shape}")
                print(f"   Dtype: {bx.dtype}")
                print(f"   Sample rate: {bx.attrs.get('sample_rate', 'N/A')} Hz")

                # Get units
                units_attr = bx.attrs.get('units', b'unknown')
                if isinstance(units_attr, bytes):
                    units = units_attr.decode()
                else:
                    units = units_attr
                print(f"   Units: {units}")

                # Check sensor info
                print(f"   Sensor manufacturer: {bx.attrs.get('sensor.manufacturer', b'unknown')}")
                print(f"   Sensor type: {bx.attrs.get('sensor.type', b'unknown')}")

                # Sample data
                sample = bx[:100]
                print(f"\n6. Sample data (first 100 points):")
                print(f"   Mean: {sample.mean():.2f}")
                print(f"   Std: {sample.std():.2f}")
                print(f"   Min: {sample.min():.2f}")
                print(f"   Max: {sample.max():.2f}")
                print(f"   First 5: {sample[:5]}")

                # Determine if calibrated
                print(f"\n7. Calibration status:")
                if abs(sample.mean()) < 100:
                    print(f"   ⚠️  Data appears UNCALIBRATED (mean ~ {sample.mean():.0f})")
                    print(f"   Expected: ~20,000-25,000 nT for Victoria")
                elif 20000 < abs(sample.mean()) < 60000:
                    print(f"   ✓ Data appears CALIBRATED (mean ~ {sample.mean():.0f} nT)")
                    print(f"   This matches expected field strength for Victoria")
                else:
                    print(f"   ❓ Unclear calibration status (mean = {sample.mean():.0f})")

            # Check all channels
            print(f"\n8. All channels overview:")
            for ch_name in ['bx', 'by', 'bz', 'ex', 'ey']:
                if ch_name in run:
                    ch = run[ch_name]
                    sample = ch[:1000]

                    # Get units
                    units_attr = ch.attrs.get('units', b'unknown')
                    if isinstance(units_attr, bytes):
                        units = units_attr.decode()
                    else:
                        units = units_attr

                    print(f"   {ch_name.upper()}: mean={sample.mean():.2f}, "
                          f"std={sample.std():.2f}, units={units}")

print("\n" + "=" * 80)
print("Conclusion: Check if LEMI data is already calibrated in nT")
