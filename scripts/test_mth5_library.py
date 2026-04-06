"""
Test the mth5 library with GA's MTH5 files.

Key questions:
1. How to read station/run/channel data?
2. Does mth5 return xarray or pandas DataFrame?
3. Can we apply custom calibration coefficients?
4. How to handle the incorrect GA calibration?
"""

from pathlib import Path
from mth5.mth5 import MTH5
import numpy as np

# Test file
mth5_file = Path(r'E:\MT_Timeseries_DATA\MT_AusLAMP_GA\EDL_MTH5\VIC001.h5')

print("=" * 80)
print("Testing mth5 library with GA MTH5 files")
print("=" * 80)

# Open MTH5 file
print(f"\n1. Opening: {mth5_file.name}")
mth5_obj = MTH5(file_version='0.2.0')
mth5_obj.open_mth5(mth5_file, 'r')

# Explore structure
print(f"\n2. Surveys in file: {mth5_obj.surveys_group.groups_list}")

# Get station
survey = mth5_obj.get_survey('AusLAMP_VIC')
print(f"\n3. Stations in survey: {survey.stations_group.groups_list[:5]}...")  # First 5

station = survey.get_station('VIC001')
print(f"\n4. Station metadata:")
print(f"   ID: {station.metadata.id}")
print(f"   Location: {station.metadata.location.latitude}, {station.metadata.location.longitude}")
print(f"   Elevation: {station.metadata.location.elevation} m")
print(f"   Declination: {station.metadata.location.declination.value}°")
print(f"   Comments: {station.metadata.comments}")

# Get runs
print(f"\n5. Runs in station: {station.run_list}")

run = station.get_run('20140411060000')
print(f"\n6. Run metadata:")
print(f"   Sample rate: {run.metadata.sample_rate} Hz")
print(f"   Start: {run.metadata.time_period.start}")
print(f"   End: {run.metadata.time_period.end}")

# Get channel data
print(f"\n7. Channels in run: {run.channel_list}")

bx_channel = run.get_channel('bx')
print(f"\n8. BX Channel:")
print(f"   Type: {type(bx_channel)}")
print(f"   Metadata type: {bx_channel.metadata.type}")
print(f"   Units: {bx_channel.metadata.units}")
print(f"   Sample rate: {bx_channel.sample_rate}")

# Get the actual data
print(f"\n9. Accessing time series data:")
bx_data = bx_channel.to_xarray()
print(f"   Type: {type(bx_data)}")
print(f"   Shape: {bx_data.shape}")
print(f"   Dimensions: {bx_data.dims}")
print(f"   Data type: {bx_data.dtype}")

# Check raw values
print(f"\n10. Raw data values:")
print(f"    First 5 samples: {bx_data.values[:5]}")
print(f"    Mean: {bx_data.values.mean():.2f}")
print(f"    Std: {bx_data.values.std():.2f}")
print(f"    Min: {bx_data.values.min()}")
print(f"    Max: {bx_data.values.max()}")

# Compare with our calibration
print(f"\n11. Applying correct Bartington Mag-03 calibration:")
BARTINGTON_BX_CALIBRATION = 0.007  # nT per μV
bx_calibrated = bx_data.values * BARTINGTON_BX_CALIBRATION
print(f"    Mean (calibrated): {bx_calibrated.mean():.2f} nT")
print(f"    Std (calibrated): {bx_calibrated.std():.2f} nT")
print(f"    First 5 samples: {bx_calibrated[:5]}")

# Check if we can convert to pandas
print(f"\n12. Converting to pandas DataFrame:")
bx_df = bx_data.to_pandas()
print(f"    Type: {type(bx_df)}")
print(f"    Shape: {bx_df.shape}")
print(f"    Index type: {type(bx_df.index)}")
print(f"    First 5 rows:")
print(bx_df.head())

# Check all channels
print(f"\n13. Loading all magnetic channels:")
channels = {}
for ch_name in ['bx', 'by', 'bz']:
    ch = run.get_channel(ch_name)
    ch_data = ch.to_xarray()
    channels[ch_name.upper()] = ch_data.values
    print(f"    {ch_name.upper()}: {len(ch_data):,} samples, mean={ch_data.values.mean():.1f}")

# Calibrate all channels
print(f"\n14. Calibrated magnetic field values:")
CALIBRATION = {'BX': 0.007, 'BY': 0.007, 'BZ': 0.0175}
for ch_name, raw_data in channels.items():
    calibrated = raw_data * CALIBRATION[ch_name]
    print(f"    {ch_name}: {calibrated.mean():.2f} nT ± {calibrated.std():.2f} nT")

# Check horizontal field
bx_cal = channels['BX'] * CALIBRATION['BX']
by_cal = channels['BY'] * CALIBRATION['BY']
h_field = np.sqrt(bx_cal**2 + by_cal**2)
print(f"\n15. Horizontal field strength: {h_field.mean():.2f} nT")

mth5_obj.close_mth5()
print(f"\n✓ Successfully read and calibrated MTH5 data using mth5 library")
