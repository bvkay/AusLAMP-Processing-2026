"""IGRF declination, great-circle distance and the INTERMAGNET observatory list.

The declination is recorded in every product's header and never applied. Products are served in geomagnetic
north, with each site's horizontal magnetics rotated so the mean Hy is zero; applying IGRF as well would
rotate them twice.

ppigrf.igrf takes longitude FIRST, height in KILOMETRES, and returns (Be, Bn, Bu) = east, north, up. The
convention used here is X = north, Y = east, Z = down, so Z = -Bu. igrf() applies both conversions.
Ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/vic_figures.py:31-36 (igrf).

OBSERVATORIES holds the six Australian INTERMAGNET observatories serving one-second data, used for distance
only. Each pair was read on 2026-09-16 from the 'Geodetic Latitude' and 'Geodetic Longitude' header lines of
one IAGA-2002 day (2014-01-01, publicationState=best-avail) fetched from the INTERMAGNET GIN; the parquet
archive carries no header. These values supersede the table in
D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/obs_1sec.py:46-53, which differs by 51 km at GNG (Gingin,
not Gnangara) and by 0.2-0.6 km at CNB, CTA, KDU and LRM.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# code: (name, latitude deg, longitude deg) -- INTERMAGNET GIN IAGA-2002 headers, read 2026-09-16
OBSERVATORIES = {
    "ASP": ("Alice Springs, Australia", -23.762, 133.883),
    "CNB": ("Canberra, Australia", -35.320, 149.360),
    "CTA": ("Charters Towers, Australia", -20.090, 146.264),
    "GNG": ("Gingin, Australia", -31.356, 115.715),
    "KDU": ("Kakadu, Australia", -12.690, 132.470),
    "LRM": ("Learmonth, Australia", -22.220, 114.100),
}


def igrf(lat, lon, elev_m, when):
    """The IGRF field at a point and time as X (north), Y (east), Z (down), H, F in nT and D in degrees.

    `when` is a datetime. `elev_m` is metres above sea level; ppigrf takes kilometres and the conversion is
    applied here.
    """
    import ppigrf
    # ppigrf compares `when` against a naive DatetimeIndex of coefficient epochs and raises on a tz-aware
    # datetime. All times in this package are UTC, so the tzinfo is dropped.
    if getattr(when, "tzinfo", None) is not None:
        when = when.replace(tzinfo=None)
    Be, Bn, Bu = ppigrf.igrf(lon, lat, float(elev_m) / 1000.0, when)   # lon FIRST, height in km
    X, Y, Z = float(np.squeeze(Bn)), float(np.squeeze(Be)), float(-np.squeeze(Bu))
    return dict(X=X, Y=Y, Z=Z,
                H=float(np.hypot(X, Y)),
                F=float(np.sqrt(X * X + Y * Y + Z * Z)),
                D=float(np.degrees(np.arctan2(Y, X))))


def declination(lat, lon, elev_m, when) -> float:
    """IGRF declination in degrees east of true north at a point and time. Recorded, never applied."""
    return igrf(lat, lon, elev_m, when)["D"]


def distance_km(a, b) -> float:
    """Great-circle km between two (lat, lon) pairs, or between two observatory codes."""
    if isinstance(a, str):
        a = OBSERVATORIES[a.upper()][1:]
    if isinstance(b, str):
        b = OBSERVATORIES[b.upper()][1:]
    la1, lo1, la2, lo2 = np.radians([a[0], a[1], b[0], b[1]])
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


def nearest_observatory(lat, lon):
    """(code, km) of the closest observatory in OBSERVATORIES."""
    d = {c: distance_km((lat, lon), (v[1], v[2])) for c, v in OBSERVATORIES.items()}
    code = min(d, key=d.get)
    return code, d[code]


def observatory_distances(lat, lon):
    """[(code, name, km)] for every observatory, nearest first."""
    out = [(c, v[0], distance_km((lat, lon), (v[1], v[2]))) for c, v in OBSERVATORIES.items()]
    return sorted(out, key=lambda t: t[2])


def scatter_m(lats, lons) -> float:
    """95th percentile spread of a cloud of fixes about its own median, in metres."""
    la = np.asarray(lats, float)
    lo = np.asarray(lons, float)
    if la.size < 2:
        return float("nan")
    dy = (la - np.median(la)) * 111320.0
    dx = (lo - np.median(lo)) * 111320.0 * np.cos(np.radians(np.median(la)))
    return float(np.percentile(np.hypot(dx, dy), 95))
