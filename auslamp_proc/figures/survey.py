"""The two survey figures: the site map and the deployment register.

The map draws the coastline from auslamp_proc/data/coastline_au.npz, the Natural Earth 1:50 m line clipped to
lon 108-160 deg and lat -48 to -8 deg as two float32 arrays with NaN between parts (17.2 KB, 2,962 vertices),
built once by tools/build_coastline.py. The environment that runs the workbooks needs no GIS package.

The frame is the sites' own bounding box padded by 10 per cent of its span or 0.5 deg, whichever is larger, so
the survey fills the picture. An observatory inside that frame is drawn; one outside it is named in a one-line
box in the lower left with its bearing in degrees east of north and its distance in km from the survey
centroid, which keeps a 1,000 km observatory from setting the scale of a 3 deg survey.

Aspect is set to 1 / cos(mean latitude) so a degree of longitude is drawn at its real length.

Site labels are the digits of the site name and what follows them: VIC073 is drawn as 073. Full names overlap
at a hundred sites.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .common import finish

DATA = Path(__file__).resolve().parent.parent / "data"
COASTLINE = DATA / "coastline_au.npz"

EDL_COLOUR = "C0"
LEMI_COLOUR = "C1"
OBS_COLOUR = "k"

FRAME_PAD_FRACTION = 0.10
FRAME_PAD_MIN_DEG = 0.5

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def bearing_deg(a, b) -> float:
    """The initial great-circle bearing from (lat, lon) a to b, in degrees east of north, 0 to 360."""
    la1, lo1, la2, lo2 = np.radians([a[0], a[1], b[0], b[1]])
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def compass_point(bearing: float) -> str:
    """A bearing in degrees to one of the 16 compass points."""
    return COMPASS[int(round((bearing % 360.0) / 22.5)) % 16]


def site_frame(lats, lons, pad_fraction=FRAME_PAD_FRACTION, pad_min_deg=FRAME_PAD_MIN_DEG):
    """(lon_min, lon_max, lat_min, lat_max) of the sites padded by pad_fraction of the span or pad_min_deg."""
    la = np.asarray([v for v in lats if np.isfinite(v)], float)
    lo = np.asarray([v for v in lons if np.isfinite(v)], float)
    if not la.size or not lo.size:
        return None
    pad_x = max(pad_fraction * (lo.max() - lo.min()), pad_min_deg)
    pad_y = max(pad_fraction * (la.max() - la.min()), pad_min_deg)
    return lo.min() - pad_x, lo.max() + pad_x, la.min() - pad_y, la.max() + pad_y


def load_coastline(path=None):
    """(lon, lat) float32 arrays with NaN between parts, or (None, None) if the file is absent."""
    p = Path(path) if path else COASTLINE
    if not p.exists():
        return None, None
    z = np.load(p)
    return z["lon"], z["lat"]


def short_label(site: str) -> str:
    """'VIC073' -> '073', 'VIC016R' -> '016R', 'Q57N' -> '57N': the first digit onward."""
    m = re.search(r"\d.*$", str(site))
    return m.group(0) if m else str(site)


def map(sites, observatories, out, title="", caption="", coastline=None, figsize=(9.0, 8.0),
        dpi=110,
        label_col="site", colour_by="instrument"):
    """Sites as points with short labels, one colour per instrument, observatories as black triangles.

    `sites` is a frame with site, lat, lon and the `colour_by` column; `observatories` is [(code, name, lat,
    lon)]. The frame is the sites' bounding box padded by 10 per cent or 0.5 deg, whichever is larger; an
    observatory outside it is not drawn but is named in a one-line box with its bearing and distance from the
    survey centroid. Returns the figure.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    lon, lat = load_coastline(coastline)
    if lon is not None:
        ax.plot(lon, lat, color="0.55", lw=0.7, zorder=1)

    s = sites.dropna(subset=["lat", "lon"])
    kinds = list(dict.fromkeys(s[colour_by])) if colour_by in s.columns else [""]
    palette = {k: (EDL_COLOUR if i == 0 else LEMI_COLOUR if i == 1 else "C%d" % (i + 1))
               for i, k in enumerate(kinds)}
    for k in kinds:
        sub = s[s[colour_by] == k] if colour_by in s.columns else s
        ax.plot(sub.lon, sub.lat, "o", ms=5, color=palette[k], label="%s (%d)" % (k, len(sub)), zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(short_label(r[label_col]), (r.lon, r.lat), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, color="0.25", zorder=4)

    # the frame is the sites', not the sites plus the observatories: a 1,000 km observatory otherwise sets the
    # scale and the survey shrinks into a corner
    from ..geo import distance_km
    x0, x1, y0, y1 = site_frame(s.lat.values, s.lon.values)
    centroid = (float(np.mean(s.lat.values)), float(np.mean(s.lon.values)))
    outside = []
    for code, name, olat, olon in observatories:
        if x0 <= olon <= x1 and y0 <= olat <= y1:
            ax.plot(olon, olat, "^", ms=9, color=OBS_COLOUR, zorder=5)
            ax.annotate(code, (olon, olat), xytext=(5, -9), textcoords="offset points",
                        fontsize=8, color=OBS_COLOUR, zorder=5)
        else:
            bd = bearing_deg(centroid, (olat, olon))
            outside.append("%s %03.0f deg %s %.0f km"
                           % (code, bd, compass_point(bd), distance_km(centroid, (olat, olon))))
    if outside:
        ax.text(0.01, 0.01, "outside the frame, from the survey centroid: " + "; ".join(outside),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.2",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9), zorder=6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.radians(centroid[0])))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("")
    finish(fig, title, caption or (
        "Every site as a point labelled with the digits of its name, one colour per instrument, and each "
        "observatory inside the frame as a black triangle. The frame is the sites' own bounding box padded "
        "by 10 per cent of its span or 0.5 deg, whichever is larger, and the aspect is 1 / cos(mean "
        "latitude), so a degree of longitude is drawn at its real length. An observatory outside the frame "
        "is named in the corner box with its bearing in degrees east of north and its distance in km from "
        "the survey centroid, because a 1,000 km observatory drawn on the map would set the scale. The "
        "coastline is the Natural Earth 1:50 m line stored as two arrays in the package."), out, dpi=dpi)
    return fig


def register(spans, groups, active, out, title="", caption="", figsize=(11.0, 11.0), dpi=110,
             colour_by="instrument", group_spans=True):
    """A Gantt of the records sorted by start, over the sites-active-per-day curve.

    One bar per site, coloured by instrument; each group's common window as a light span behind the bars; the
    lower panel counts the sites recording each day.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    sp = spans.sort_values("start").reset_index(drop=True)
    # constrained layout: tight_layout warns on a shared-x pair and may place it wrongly
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, layout="constrained",
        gridspec_kw=dict(height_ratios=[4.2, 1.0]))

    kinds = list(dict.fromkeys(sp[colour_by])) if colour_by in sp.columns else [""]
    palette = {k: (EDL_COLOUR if i == 0 else LEMI_COLOUR if i == 1 else "C%d" % (i + 1))
               for i, k in enumerate(kinds)}

    if group_spans and groups is not None and len(groups):
        for _, g in groups.iterrows():
            if not str(g.get("common_start", "")) or float(g.get("common_days", 0) or 0) <= 0:
                continue
            ax.axvspan(pd.Timestamp(g["common_start"]), pd.Timestamp(g["common_end"]),
                       color="0.85", zorder=0)

    for i, r in sp.iterrows():
        c = palette.get(r[colour_by], "C0") if colour_by in sp.columns else "C0"
        ax.barh(i, (r.end - r.start).total_seconds() / 86400.0, left=r.start, height=0.72,
                color=c, zorder=3)
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels(sp.site, fontsize=5.5)
    ax.set_ylim(-1, len(sp))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title(title)
    handles = [plt.Line2D([], [], color=palette[k], lw=6,
                          label="%s (%d)" % (k, int((sp[colour_by] == k).sum()))) for k in kinds]
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    ax2.fill_between(active.index, active.values, step="mid", color="0.35", alpha=0.8)
    ax2.set_ylabel("sites active")
    ax2.grid(alpha=0.25, lw=0.5)
    # day of month in the tick: a survey inside one month labelled %Y-%m repeats the same month on every tick
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.set_title("")
    finish(fig, title, caption or (
        "One bar per site over the days it recorded, sorted by start and coloured by instrument, with each "
        "overlap group's common window as a light span behind the bars; the lower panel counts the sites "
        "recording each day. A group is the pool a reference is drawn from, so a site's bar has to overlap "
        "a candidate's for the two to be a pair at all. What to change: MIN_OVERLAP_DAYS sets how much "
        "overlap two sites need to share a group."), out, dpi=dpi)
    return fig
