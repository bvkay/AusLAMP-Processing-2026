"""The two survey figures: the site map and the deployment register.

The map draws the coastline from auslamp_proc/data/coastline_au.npz, the Natural Earth 1:50 m line clipped to
lon 108-160 deg and lat -48 to -8 deg as two float32 arrays with NaN between parts (17.2 KB, 2,962 vertices),
built once by tools/build_coastline.py. The environment that runs the workbooks needs no GIS package.

Aspect is set to 1 / cos(mean latitude) so a degree of longitude is drawn at its real length.

Site labels are the digits of the site name and what follows them: VIC073 is drawn as 073. Full names overlap
at a hundred sites.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
COASTLINE = DATA / "coastline_au.npz"

EDL_COLOUR = "C0"
LEMI_COLOUR = "C1"
OBS_COLOUR = "k"


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


def map(sites, observatories, out, title="", coastline=None, figsize=(9.0, 8.0), dpi=110,
        label_col="site", colour_by="instrument"):
    """Sites as points with short labels, one colour per instrument, observatories as black triangles.

    `sites` is a frame with site, lat, lon and the `colour_by` column; `observatories` is [(code, name, lat,
    lon)]. Returns the figure.
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

    for code, name, olat, olon in observatories:
        ax.plot(olon, olat, "^", ms=9, color=OBS_COLOUR, zorder=5)
        ax.annotate(code, (olon, olat), xytext=(5, -9), textcoords="offset points",
                    fontsize=8, color=OBS_COLOUR, zorder=5)

    la = np.r_[s.lat.values, [o[2] for o in observatories]]
    lo = np.r_[s.lon.values, [o[3] for o in observatories]]
    pad = 0.6
    ax.set_xlim(lo.min() - pad, lo.max() + pad)
    ax.set_ylim(la.min() - pad, la.max() + pad)
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(la)))))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    return fig


def register(spans, groups, active, out, title="", figsize=(11.0, 11.0), dpi=110,
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
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    fig.savefig(out, dpi=dpi)
    return fig
