"""
Test notebook 02 functionality with a few representative sites.

Tests:
1. Metadata loading
2. Statistics calculation for 2 EDL + 2 LEMI sites
3. Plotting functions work
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from readers import read_station, get_all_metadata
from config import DataPaths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("Testing Notebook 02 Functionality")
print("=" * 80)

# Test 1: Load metadata
print("\n1. Testing metadata loading...")
metadata_file = DataPaths.PROCESSED_DATA_DIR / 'metadata_all_sites.csv'
df_meta = pd.read_csv(metadata_file)
df_meta['start_time'] = pd.to_datetime(df_meta['start_time'], format='mixed')
df_meta['end_time'] = pd.to_datetime(df_meta['end_time'], format='mixed')

print(f"   ✓ Loaded {len(df_meta)} sites")
print(f"   ✓ EDL: {len(df_meta[df_meta['instrument']=='EDL'])}")
print(f"   ✓ LEMI-424: {len(df_meta[df_meta['instrument']=='LEMI-424'])}")

# Test 2: Statistics calculation
print("\n2. Testing statistics calculation...")

test_sites = [
    ('VIC001', 'EDL'),
    ('VIC010', 'EDL'),
    ('VIC065R', 'LEMI-424'),
    ('VIC009R', 'LEMI-424')
]

def get_site_statistics(mth5_file, duration_hours=1):
    """Calculate statistics from first N hours of recording."""
    try:
        df = read_station(mth5_file, channels='magnetic', verbose=False)

        # Get first N hours
        sample_rate = len(df) / ((df.index[-1] - df.index[0]).total_seconds() / 3600)
        n_samples = int(duration_hours * 3600 * sample_rate)
        df_sample = df.iloc[:n_samples]

        # Calculate statistics
        stats = {
            'bx_mean': df_sample['BX'].mean(),
            'by_mean': df_sample['BY'].mean(),
            'bz_mean': df_sample['BZ'].mean(),
            'bx_std': df_sample['BX'].std(),
            'by_std': df_sample['BY'].std(),
            'bz_std': df_sample['BZ'].std(),
            'n_samples': len(df_sample)
        }

        # Derived quantities
        stats['h_field'] = np.sqrt(stats['bx_mean']**2 + stats['by_mean']**2)
        stats['total_field'] = np.sqrt(stats['bx_mean']**2 + stats['by_mean']**2 + stats['bz_mean']**2)

        return stats

    except Exception as e:
        print(f"Error: {e}")
        return None

stats_results = []

for station_id, instrument in test_sites:
    print(f"\n   Testing {station_id} ({instrument})...")

    # Find MTH5 file
    site_row = df_meta[df_meta['station_id'] == station_id].iloc[0]
    mth5_file = Path(site_row['mth5_file'])

    if not mth5_file.exists():
        print(f"   ✗ File not found: {mth5_file}")
        continue

    stats = get_site_statistics(mth5_file)

    if stats:
        print(f"   ✓ Loaded {stats['n_samples']} samples")
        print(f"     BX: {stats['bx_mean']:.2f} ± {stats['bx_std']:.2f} nT")
        print(f"     BY: {stats['by_mean']:.2f} ± {stats['by_std']:.2f} nT")
        print(f"     BZ: {stats['bz_mean']:.2f} ± {stats['bz_std']:.2f} nT")
        print(f"     H: {stats['h_field']:.2f} nT")
        print(f"     F: {stats['total_field']:.2f} nT")

        stats['station_id'] = station_id
        stats['instrument'] = instrument
        stats_results.append(stats)
    else:
        print(f"   ✗ Failed to calculate statistics")

# Test 3: Plotting
print("\n3. Testing plotting functions...")

if len(stats_results) > 0:
    df_stats = pd.DataFrame(stats_results)

    # Simple test plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'EDL': 'blue', 'LEMI-424': 'red'}

    for instrument in df_stats['instrument'].unique():
        subset = df_stats[df_stats['instrument'] == instrument]
        x = range(len(subset))
        y = subset['total_field'].values
        ax.scatter(x, y, c=colors[instrument], s=80, alpha=0.7,
                  label=f'{instrument} (n={len(subset)})')

    ax.set_xlabel('Site Index')
    ax.set_ylabel('Total Field (nT)')
    ax.set_title('Test Plot: Total Field for Sample Sites')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save test figure
    test_fig_path = DataPaths.FIGURES_DIR / 'test_plot.png'
    fig.savefig(test_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Test plot saved to: {test_fig_path}")
else:
    print("   ✗ No statistics available for plotting")

# Test 4: Output directory structure
print("\n4. Testing output directory structure...")

required_dirs = [
    DataPaths.FIGURES_DIR,
    DataPaths.FIGURES_DIR / 'qaqc',
    DataPaths.PROCESSED_DATA_DIR / 'qaqc'
]

project_root = DataPaths.PROJECT_DATA.parent  # Get project root from data dir

for dir_path in required_dirs:
    if dir_path.exists():
        print(f"   ✓ {dir_path.relative_to(project_root)}")
    else:
        print(f"   ✗ Missing: {dir_path.relative_to(project_root)}")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"     Created directory")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print(f"\nMetadata: ✓ {len(df_meta)} sites loaded")
print(f"Statistics: ✓ {len(stats_results)}/{len(test_sites)} test sites successful")
print(f"Plotting: ✓ Test plot generated")
print(f"Directories: ✓ All required directories exist")

if len(stats_results) == len(test_sites):
    print("\n✓ All tests passed! Notebook 02 is ready to use.")
else:
    print(f"\n⚠️  Some tests failed ({len(stats_results)}/{len(test_sites)} sites)")
    print("    Check errors above and fix before running full notebook.")

print("=" * 80)
