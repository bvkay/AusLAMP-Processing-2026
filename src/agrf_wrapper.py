"""
AGRF25 Python Wrapper

Calls the compiled AGRF25S.EXE executable to compute Australian Geomagnetic
Reference Field predictions.

AGRF25 = IGRF-14 + Regional Residual Field (spherical cap harmonic model)
Valid: 2020-2030 (optimized for 2025-2030)
Coverage: Spherical cap 28° radius centered on -24°N, 135°E

Author: Claude Code
Date: 2026-04-01
"""

import subprocess
from pathlib import Path
import re
from typing import Dict, Union, Optional
import tempfile


def compute_agrf25(
    lat: float,
    lon: float,
    alt_km: float,
    year: float,
    agrf_dir: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Compute AGRF25 field components by calling AGRF25S.EXE

    Parameters
    ----------
    lat : float
        Geodetic latitude (degrees, negative for south)
    lon : float
        Geodetic longitude (degrees, positive for east)
    alt_km : float
        Height above mean sea level (km)
    year : float
        Decimal year (e.g., 2024.5 for mid-2024)
    agrf_dir : str or Path, optional
        Directory containing AGRF25S.EXE and COEF25
        Default: '../data/eCat_150056_AGRF25' relative to this file

    Returns
    -------
    dict
        Field components with keys:
        - X : float - North component (nT)
        - Y : float - East component (nT)
        - Z : float - Down component (nT)
        - H : float - Horizontal intensity (nT)
        - F : float - Total intensity (nT)
        - D : float - Declination (degrees)
        - I : float - Inclination (degrees)

    Raises
    ------
    FileNotFoundError
        If AGRF25S.EXE not found
    ValueError
        If AGRF output cannot be parsed
    RuntimeError
        If AGRF25S.EXE execution fails

    Notes
    -----
    AGRF25 uses NED coordinate convention (same as IGRF/INTERMAGNET):
    - X = North component
    - Y = East component
    - Z = Down component (positive downward)

    In southern hemisphere, Z is typically negative (field points upward).

    Examples
    --------
    >>> result = compute_agrf25(-34.5, 138.5, 0.3, 2024.5)
    >>> print(f"X={result['X']:.1f} nT, Y={result['Y']:.1f} nT, Z={result['Z']:.1f} nT")
    X=23206.0 nT, Y=3280.0 nT, Z=-54122.0 nT
    """
    # Resolve AGRF directory
    if agrf_dir is None:
        agrf_dir = Path(__file__).parent.parent / 'data' / 'eCat_150056_AGRF25'
    else:
        agrf_dir = Path(agrf_dir)

    exe_path = agrf_dir / 'AGRF25S.EXE'

    if not exe_path.exists():
        raise FileNotFoundError(
            f"AGRF25S.EXE not found at {exe_path}\n"
            f"Expected directory: {agrf_dir}\n"
            f"Download from: https://geomag.ga.gov.au/agrf.html"
        )

    # Check year validity
    if year < 2020 or year > 2030:
        import warnings
        warnings.warn(
            f"AGRF25 is optimized for 2020-2030, but year={year} was requested. "
            f"Predictions may be less accurate outside this range."
        )

    # Convert decimal degrees to deg/min/sec
    def decimal_to_dms(decimal: float) -> tuple:
        """Convert decimal degrees to (degrees, minutes, seconds)"""
        degrees = int(decimal)
        minutes_decimal = abs(decimal - degrees) * 60
        minutes = int(minutes_decimal)
        seconds = (minutes_decimal - minutes) * 60
        return degrees, minutes, seconds

    lat_d, lat_m, lat_s = decimal_to_dms(lat)
    lon_d, lon_m, lon_s = decimal_to_dms(lon)

    # Create unique temporary output filename
    import time
    import os
    timestamp = int(time.time() * 1000000)  # microseconds
    output_filename = f"agrf_tmp_{os.getpid()}_{timestamp}.txt"
    output_file = agrf_dir / output_filename

    try:
        # Prepare input for interactive program
        # Sequence: output_file, model(1=AGRF), year, site_name, lat, lon, alt, more?(0=quit), enter_to_finish
        input_text = f"""{output_filename}
1
{year}
Site
{lat_d} {lat_m} {lat_s}
{lon_d} {lon_m} {lon_s}
{alt_km}
0

"""

        # Run AGRF25S.EXE with piped input
        result = subprocess.run(
            [str(exe_path)],
            input=input_text,
            capture_output=True,
            text=True,
            cwd=str(agrf_dir),
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"AGRF25S.EXE failed with return code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Parse output from stdout
        # Looking for lines like: "Field =   23206.0  3280.0 -54122.0  23439.8  58419.7    8.03  -66.59"
        output = result.stdout

        field_match = re.search(
            r'Field\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
            output
        )

        if field_match:
            X, Y, Z, H, F, D, I = map(float, field_match.groups())

            return {
                'X': X,  # nT, North component
                'Y': Y,  # nT, East component
                'Z': Z,  # nT, Down component
                'H': H,  # nT, Horizontal intensity
                'F': F,  # nT, Total intensity
                'D': D,  # degrees, Declination
                'I': I   # degrees, Inclination
            }
        else:
            raise ValueError(
                f"Could not parse AGRF25 output. Expected 'Field = X Y Z H F D I' line.\n"
                f"Output received:\n{output}"
            )

    finally:
        # Clean up temporary output file
        try:
            if output_file.exists():
                output_file.unlink()
        except Exception:
            pass  # Ignore cleanup errors


def compute_agrf25_batch(
    sites: list,
    agrf_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute AGRF25 for multiple sites

    Parameters
    ----------
    sites : list of dict
        List of site dictionaries with keys: 'name', 'lat', 'lon', 'alt_km', 'year'
    agrf_dir : str or Path, optional
        Directory containing AGRF25S.EXE and COEF25

    Returns
    -------
    dict
        Dictionary mapping site names to AGRF results

    Examples
    --------
    >>> sites = [
    ...     {'name': 'Site1', 'lat': -34.5, 'lon': 138.5, 'alt_km': 0.3, 'year': 2024.5},
    ...     {'name': 'Site2', 'lat': -35.0, 'lon': 139.0, 'alt_km': 0.2, 'year': 2024.5}
    ... ]
    >>> results = compute_agrf25_batch(sites)
    >>> print(results['Site1']['X'])
    23206.0
    """
    results = {}

    for site in sites:
        try:
            result = compute_agrf25(
                lat=site['lat'],
                lon=site['lon'],
                alt_km=site['alt_km'],
                year=site['year'],
                agrf_dir=agrf_dir
            )
            results[site['name']] = result
        except Exception as e:
            print(f"Warning: Failed to compute AGRF for {site['name']}: {e}")
            results[site['name']] = None

    return results


if __name__ == '__main__':
    # Test with parallel test site
    print("Testing AGRF25 wrapper...")
    print("=" * 60)

    result = compute_agrf25(
        lat=-34.5,
        lon=138.5,
        alt_km=0.3,
        year=2024.81  # Oct 2024
    )

    print("Parallel test site (-34.5°S, 138.5°E, 300m, Oct 2024):")
    print(f"  X (North): {result['X']:>8.1f} nT")
    print(f"  Y (East):  {result['Y']:>8.1f} nT")
    print(f"  Z (Down):  {result['Z']:>8.1f} nT")
    print(f"  H:         {result['H']:>8.1f} nT")
    print(f"  F:         {result['F']:>8.1f} nT")
    print(f"  D:         {result['D']:>8.2f}°")
    print(f"  I:         {result['I']:>8.2f}°")
