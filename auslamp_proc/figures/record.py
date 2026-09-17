"""Figure 01: the whole record as laid, in eight panels, with the IGRF values beside the measured medians.

Panels: Hx, Hy, Hz, Ex, Ey, then H = sqrt(Hx^2 + Hy^2), F = sqrt(Hx^2 + Hy^2 + Hz^2), then the hourly sensor
angle atan2(Hy, Hx). Each of the first seven is the per-minute mean drawn over the per-minute range, its
y-limits the 1st to 99th percentile of the channel with a 5 per cent pad, and its median printed with the
channel's range; the magnetic panels also carry the IGRF value at the site at the record midpoint and the
ratio of the median to it. figsize (15, 16), dpi 110.

The record is the cache: no sign, no rotation and no notch. What the figure is read for is a reversed axis
(Bx below zero), a dead channel, a gain (F against IGRF), a turn of the sensor (the hourly angle), a rail and
a step.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from ..geo import igrf as igrf_at
from .common import finish

DAY = 86400
CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")
COLOURS = {"Hx": "C0", "Hy": "C1", "Hz": "C2", "Ex": "C3", "Ey": "C4"}
UNIT = {"Hx": "nT", "Hy": "nT", "Hz": "nT", "Ex": "mV/km", "Ey": "mV/km"}


def iso(t) -> str:
    """A unix second as 'YYYY-MM-DD HH:MM' UTC."""
    return datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


def day_axis(t0, n, fs=1.0):
    """(days since t0 per sample, tick positions in days, tick labels as MM-DD)."""
    days = np.arange(n) / (fs * DAY)
    start = datetime.fromtimestamp(int(t0), timezone.utc)
    n_days = int(days[-1]) + 1 if n else 1
    step = max(1, n_days // 8)
    ticks = list(range(0, n_days + 1, step))
    labels = [(start + timedelta(days=d)).strftime("%m-%d") for d in ticks]
    return days, ticks, labels


def minute_stats(x, m=60):
    """(mean, min, max) of each block of m samples."""
    n = len(x) // m * m
    a = np.asarray(x[:n], float).reshape(-1, m)
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=1), np.nanmin(a, axis=1), np.nanmax(a, axis=1)


def record(t0, arrays, out, site="", survey="", lat=np.nan, lon=np.nan, elev_m=0.0, fs=1.0,
           figsize=(15, 16), dpi=110):
    """Draw figure 01 for one site and save it to `out`. Returns (figure, one summary line)."""
    import matplotlib.pyplot as plt

    arr = {c: np.asarray(arrays[c], float) for c in CHANNELS}
    n = len(arr["Hx"])
    m_per_minute = int(round(60 * fs))
    ig = igrf_at(lat, lon, elev_m if np.isfinite(elev_m) else 0.0,
                 datetime.fromtimestamp(t0 + n / fs / 2, timezone.utc)) if np.isfinite(lat) else None
    H = np.hypot(arr["Hx"], arr["Hy"])
    F = np.sqrt(arr["Hx"] ** 2 + arr["Hy"] ** 2 + arr["Hz"] ** 2)

    m = int(round(3600 * fs))
    nh = n // m * m
    with np.errstate(all="ignore"):
        hx_h = np.nanmean(arr["Hx"][:nh].reshape(-1, m), axis=1)
        hy_h = np.nanmean(arr["Hy"][:nh].reshape(-1, m), axis=1)
    ang = np.degrees(np.arctan2(hy_h, hx_h))
    t_ang = (np.arange(len(ang)) * m + m / 2) / (fs * DAY)
    _, ticks, labels = day_axis(t0, n, fs)

    fig, axes = plt.subplots(8, 1, figsize=figsize, sharex=True)
    panels = [("Hx", arr["Hx"], UNIT["Hx"], "X"), ("Hy", arr["Hy"], UNIT["Hy"], "Y"),
              ("Hz", arr["Hz"], UNIT["Hz"], "Z"), ("Ex", arr["Ex"], UNIT["Ex"], None),
              ("Ey", arr["Ey"], UNIT["Ey"], None),
              ("H = sqrt(Hx2+Hy2)", H, "nT", "H"), ("F = |B|", F, "nT", "F")]
    for ax, (name, x, unit, key) in zip(axes[:7], panels):
        mid, lo, hi = minute_stats(x, m_per_minute)
        t = (np.arange(len(mid)) * 60 + 30) / DAY
        col = COLOURS.get(name, "0.3")
        ax.fill_between(t, lo, hi, lw=0, alpha=0.3, color=col, rasterized=True)
        ax.plot(t, mid, lw=0.7, color=col)
        finite = np.isfinite(x)
        med = float(np.nanmedian(x)) if finite.any() else np.nan
        sd = float(np.nanstd(x)) if finite.any() else np.nan
        if finite.any():
            p_lo, p_hi = np.nanpercentile(x, (1, 99))
            pad = 0.05 * (p_hi - p_lo + 1e-9)
            ax.set_ylim(p_lo - pad, p_hi + pad)
        ax.axhline(med, color="k", lw=0.6, ls=":")
        txt = ("median %.1f %s, std %.1f, range %.4g .. %.4g"
               % (med, unit, sd, np.nanmin(x) if finite.any() else np.nan,
                  np.nanmax(x) if finite.any() else np.nan))
        if ig is not None and key:
            txt += "   |   IGRF %s = %.0f nT (ratio %.3f)" % (key, ig[key], med / ig[key] if ig[key] else np.nan)
        ax.text(0.005, 0.93, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        ax.set_ylabel("%s\n(%s)" % (name, unit), fontsize=8)
        ax.grid(alpha=0.25)

    ax = axes[7]
    ax.plot(t_ang, ang, lw=0.8, color="C5")
    amed = float(np.nanmedian(ang)) if np.isfinite(ang).any() else np.nan
    ax.axhline(amed, color="k", lw=0.6, ls=":")
    ax.axhline(0, color="0.6", lw=0.5)
    txt = "sensor angle atan2(Hy, Hx), hourly: median %+.2f deg (0 = laid to the field's north)" % amed
    if ig is not None:
        txt += "   |   IGRF declination %+.2f deg (a sensor laid to true north)" % ig["D"]
        ax.axhline(ig["D"], color="tab:red", lw=0.6, ls="--")
    ax.text(0.005, 0.93, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="0.2",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    fin = np.isfinite(ang)
    ax.set_ylim((np.nanmin(ang) - 2, np.nanmax(ang) + 2) if fin.any() else (-30, 30))
    ax.set_ylabel("angle\n(deg)", fontsize=8)
    ax.grid(alpha=0.25)

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("days from %s UTC" % iso(t0))
    finish(fig, "%s %s: the whole record as laid" % (survey, site),
           "The %g Hz cache as laid over %.1f days from %s UTC, with no sign, no rotation and no notch. "
           "Each of the first seven panels is the per-minute mean drawn over the per-minute range, its y "
           "limits the 1st to 99th percentile of the channel with a 5 per cent pad and its median printed "
           "with the channel's range; H = sqrt(Hx^2 + Hy^2) and F = sqrt(Hx^2 + Hy^2 + Hz^2) are the field "
           "strengths, and the last panel is the sensor angle atan2(Hy, Hx) per hour. The magnetic panels "
           "carry the IGRF value at the site at the record midpoint and the ratio of the median to it."
           % (fs, n / fs / DAY, iso(t0)), out, dpi=dpi)
    line = ("%s: %.1f d; median Hx %.0f Hy %.0f Hz %.0f nT, H %.0f F %.0f nT, Ex %.2f Ey %.2f mV/km, "
            "angle %+.1f deg" % (site, n / fs / DAY, np.nanmedian(arr["Hx"]), np.nanmedian(arr["Hy"]),
                                 np.nanmedian(arr["Hz"]), np.nanmedian(H), np.nanmedian(F),
                                 np.nanmedian(arr["Ex"]), np.nanmedian(arr["Ey"]), amed))
    return fig, line
