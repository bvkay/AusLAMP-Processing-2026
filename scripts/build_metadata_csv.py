"""
Build combined metadata CSV for all 100 MT sites (EDL + LEMI-424).

Extracts metadata from all MTH5 files without loading full time series.
Outputs: data/processed/metadata_all_sites.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from readers import get_all_metadata
from config import DataPaths
import pandas as pd


def main():
    print("=" * 80)
    print("Building Metadata CSV for AusLAMP Victoria")
    print("=" * 80)

    # Ensure output directory exists
    DataPaths.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load EDL metadata
    print("\n1. Loading EDL metadata...")
    print(f"   Source: {DataPaths.EDL_MTH5}")
    df_edl = get_all_metadata(DataPaths.EDL_MTH5, pattern='VIC*.h5')
    print(f"   ✓ {len(df_edl)} EDL sites")

    # Load LEMI-424 metadata
    print("\n2. Loading LEMI-424 metadata...")
    print(f"   Source: {DataPaths.LEMI_MTH5}")
    df_lemi = get_all_metadata(DataPaths.LEMI_MTH5, pattern='VIC*R.h5')
    print(f"   ✓ {len(df_lemi)} LEMI-424 sites")

    # Combine
    print("\n3. Combining datasets...")
    df_all = pd.concat([df_edl, df_lemi], ignore_index=True)

    # Sort by station_id
    df_all = df_all.sort_values('station_id').reset_index(drop=True)

    print(f"   ✓ Total: {len(df_all)} sites")
    print(f"     - EDL: {len(df_all[df_all['instrument']=='EDL'])}")
    print(f"     - LEMI-424: {len(df_all[df_all['instrument']=='LEMI-424'])}")

    # Summary statistics
    print("\n4. Dataset Summary:")
    print(f"\n   Recording Duration by Instrument:")
    duration_stats = df_all.groupby('instrument')['duration_days'].describe()
    print(duration_stats)

    print(f"\n   Date Range:")
    print(f"     Earliest start: {df_all['start_time'].min()}")
    print(f"     Latest end: {df_all['end_time'].max()}")

    print(f"\n   Sample Rates:")
    print(df_all.groupby('instrument')['sample_rate'].value_counts())

    # Save to CSV
    output_file = DataPaths.PROCESSED_DATA_DIR / 'metadata_all_sites.csv'
    df_all.to_csv(output_file, index=False)

    print(f"\n5. Saved to: {output_file}")
    print(f"   Columns: {', '.join(df_all.columns)}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")

    # Show preview
    print("\n6. Preview (first 10 sites):")
    print(df_all[['station_id', 'instrument', 'latitude', 'longitude',
                  'duration_days', 'sample_rate']].head(10).to_string())

    print("\n" + "=" * 80)
    print("✓ Metadata CSV build complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
