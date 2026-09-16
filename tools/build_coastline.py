"""Build auslamp_proc/data/coastline_au.npz once, so the workbooks draw a map with no GIS package.

Source: the Natural Earth 1:50 m physical coastline, the copy cartopy keeps at
C:/Users/joint/.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp
(located by D:/BEN/MTH5_Aurora_mt-io_2026/scripts/qc/coastline.py).

Clip: lon 108-160 deg, lat -48 to -8 deg. Output: two float32 arrays, lon and lat, with NaN between parts.

Budget: the file must stay under MAX_BYTES = 300 KB. Vertices are dropped at every DECIMATE-th, doubling
DECIMATE until the file fits, and a part left with fewer than 2 vertices is discarded. The step used and the
byte size are printed.

Run it with an environment that has geopandas; the frozen analysis environment is read-only and will do:

    C:/Users/joint/anaconda3/envs/mt_aurora_2026/python.exe tools/build_coastline.py

The package reads the npz only; the shapefile is not needed at run time.

@author: ben kay (ben@auscope.org.au)
"""
import sys
from pathlib import Path

import numpy as np

SHP = Path(r"C:/Users/joint/.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp")
OUT = Path(__file__).resolve().parent.parent / "auslamp_proc" / "data" / "coastline_au.npz"
LON0, LON1, LAT0, LAT1 = 108.0, 160.0, -48.0, -8.0
DECIMATE = 1          # raised below until the file fits the budget
MAX_BYTES = 300 * 1024


def parts_from_shapefile():
    """[(lon, lat)] arrays, one per LineString that touches the clip box."""
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.read_file(SHP)
    clip = gdf.clip(box(LON0, LAT0, LON1, LAT1))
    out = []
    for geom in clip.geometry:
        if geom is None or geom.is_empty:
            continue
        pieces = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for g in pieces:
            xy = np.asarray(g.coords)
            if len(xy) >= 2:
                out.append((xy[:, 0], xy[:, 1]))
    return out


def pack(parts, step):
    lon, lat = [], []
    kept = 0
    for x, y in parts:
        xs, ys = x[::step], y[::step]
        if len(xs) < 2:
            continue
        kept += 1
        lon.append(np.r_[xs, np.nan])
        lat.append(np.r_[ys, np.nan])
    return (np.concatenate(lon).astype(np.float32),
            np.concatenate(lat).astype(np.float32), kept)


def main():
    if not SHP.exists():
        print("NOT FOUND:", SHP)
        return 1
    parts = parts_from_shapefile()
    print("parts touching lon %g-%g, lat %g..%g: %d" % (LON0, LON1, LAT0, LAT1, len(parts)))
    step = DECIMATE
    while True:
        lon, lat, kept = pack(parts, step)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT, lon=lon, lat=lat)
        size = OUT.stat().st_size
        print("  step %d: %d parts, %d vertices, %.1f KB" % (step, kept, len(lon), size / 1024))
        if size <= MAX_BYTES or step >= 16:
            break
        step *= 2
    print("wrote %s (%d bytes, every %d%s vertex kept)"
          % (OUT, OUT.stat().st_size, step, "" if step == 1 else "th"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
