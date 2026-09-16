"""Figures 02-05: band coherence, the coherence maps, the Welch spectra and the spectrograms.

Ported from D:/BEN/MTH5_Aurora_mt-io_2026/student_pack_v2/qld_student.py (PAIRS :431, PAIR_LABEL :432,
auto_nperseg :435, coherence_map :439, coherence_levels :469, band_from_levels :492, levels_to_grid :514,
plot_coherence_maps :526, welch :417, power_map :564, power_levels :600, plot_spectrograms :617,
spectrogram_table :669) and from scripts/processing/wamt_site_figures.py:60-99, which lays out 02 and 04.
qld_student hard-codes 1 Hz; every function here takes `fs` and every time axis is in seconds, so the same
code draws a 10 Hz record.

The parameters, which are the same at every survey:

    windows      60 min every 30 min at the base level
    levels       each factor of 4 in period gets its own level: the segment is as long as its longest period,
                 the window is 6 segments and the step is half a segment
    periods      2 s to 20,000 s, 8 log bins per decade
    nperseg      the largest power of two leaving 4 segments in the base window (512 at 3600 s and 1 Hz)
    windows      the mean is removed per window; a window holding any non-finite sample is left NaN
    bins         the DC bin is dropped; a log bin holding no FFT harmonic is dropped, not drawn as a hole
    smoothing    3 h running median along time on the maps, 12 h on the band lines
    bands        5-20, 20-200, 100-1000 and 1000-10000 s
    pairs        Bx-Ey, By-Ex, Bx-By, Ex-Ey
    colour       viridis 0 to 1 on the coherence maps with gouraud shading, guide lines at 20, 200, 1000 and
                 10,000 s; the spectrograms in dB of absolute power density with limits at the 2nd and 98th
                 percentile of the channel

Figure 04 is one Welch spectrum per channel over the longest finite run, nperseg 4096, the DC bin dropped,
log-log, the 10-1000 s band shaded.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .record import DAY, day_axis, iso

CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")
UNIT = {"Hx": "nT", "Hy": "nT", "Hz": "nT", "Ex": "mV/km", "Ey": "mV/km"}

PAIRS = (("Hx", "Ey"), ("Hy", "Ex"), ("Hx", "Hy"), ("Ex", "Ey"))
PAIR_LABEL = {("Hx", "Ey"): "Bx-Ey", ("Hy", "Ex"): "By-Ex", ("Hx", "Hy"): "Bx-By", ("Ex", "Ey"): "Ex-Ey"}

BANDS_S = ((5, 20, "5-20 s"), (20, 200, "20-200 s"), (100, 1000, "100-1000 s"), (1000, 10000, "1000-10000 s"))
GUIDE_S = (20, 200, 1000, 10000)
PER_DECADE = 8
MIN_SEGMENTS = 4
LONG_SEGMENTS = 6
LEVEL_FACTOR = 4
SMOOTH_H = 3.0
LINE_SMOOTH_H = 12.0
PMIN_S = 2.0
PMAX_S = 20000.0
SPECTRA_NPERSEG = 4096


# ---------------------------------------------------------------- the level ladder

def auto_nperseg(win_s, fs=1.0, min_segments=MIN_SEGMENTS) -> int:
    """The largest power of two leaving `min_segments` segments in a window of win_s seconds."""
    return int(2 ** np.floor(np.log2(win_s * fs / min_segments)))


def levels_plan(fs=1.0, win_s=3600, step_s=1800, pmin=PMIN_S, pmax=PMAX_S,
                min_segments=MIN_SEGMENTS, long_segments=LONG_SEGMENTS, factor=LEVEL_FACTOR):
    """The ladder the maps are built on, one row per level, without reading any data.

    Each row is (nperseg samples, segment length s, window s, step s, period floor s, period ceiling s). The
    base level uses the caller's window and step; every level above it takes a segment as long as its longest
    period, a window of `long_segments` = 6 segments and a step of half a segment.
    """
    n0 = auto_nperseg(win_s, fs, min_segments)
    rows = []
    nper, lo = n0, pmin
    while True:
        hi = min(nper / fs, pmax)
        w, st = ((win_s, step_s) if nper == n0
                 else (int(long_segments * nper / fs), max(int(step_s), int(nper / fs / 2))))
        rows.append(dict(level=len(rows), nperseg=int(nper), segment_s=round(nper / fs, 1),
                         window_s=int(w), step_s=int(st),
                         period_floor_s=round(lo, 1), period_ceiling_s=round(hi, 1)))
        if hi >= pmax:
            break
        lo, nper = hi, nper * factor
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- coherence

def coherence_map(x, y, fs=1.0, win_s=3600, step_s=1800, nperseg=None, per_decade=PER_DECADE,
                  pmin=4.0, pmax=None):
    """Squared coherence per window averaged into log-period bins.

    Returns (window centres in seconds from the first sample, bin centre periods, coherence[window, period]).
    A window holding any non-finite sample is left NaN; the DC bin is dropped; a log bin holding no FFT
    harmonic is dropped rather than drawn as a hole.
    """
    from scipy.signal import coherence as _coherence
    nperseg = int(nperseg or auto_nperseg(win_s, fs))
    pmax = float(pmax or nperseg / fs)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    w, st = int(round(win_s * fs)), int(round(step_s * fs))
    starts = np.arange(0, max(1, n - w + 1), st, dtype=int)
    periods = 1.0 / np.fft.rfftfreq(nperseg, 1.0 / fs)[1:]
    edges = np.geomspace(pmin, pmax, int(round(np.log10(pmax / pmin) * per_decade)) + 1)
    cent = np.sqrt(edges[:-1] * edges[1:])
    idx = np.digitize(periods, edges) - 1
    ok = (idx >= 0) & (idx < len(cent))
    cnt = np.bincount(idx[ok], minlength=len(cent)).astype(float)
    out = np.full((len(starts), len(cent)), np.nan)
    for k, s in enumerate(starts):
        xs, ys = x[s:s + w], y[s:s + w]
        # a record shorter than one window of this level: scipy would shorten nperseg and return a spectrum
        # of a different length from the bin index built above
        if len(xs) < nperseg or not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            continue
        _, c = _coherence(xs - xs.mean(), ys - ys.mean(), fs=fs, nperseg=nperseg)
        c = c[1:]
        acc = np.bincount(idx[ok], weights=c[ok], minlength=len(cent))
        with np.errstate(invalid="ignore"):
            out[k] = acc / cnt
    pop = cnt > 0
    return (starts + w / 2.0) / fs, cent[pop], out[:, pop]


def coherence_levels(x, y, fs=1.0, win_s=3600, step_s=1800, pmin=PMIN_S, pmax=PMAX_S,
                     per_decade=PER_DECADE, min_segments=MIN_SEGMENTS, long_segments=LONG_SEGMENTS,
                     factor=LEVEL_FACTOR):
    """coherence_map once per level of the ladder. Each entry is (t_s, periods, coh, nperseg, win_s, step_s)."""
    levels = []
    for row in levels_plan(fs, win_s, step_s, pmin, pmax, min_segments, long_segments,
                           factor).itertuples():
        tc, per, coh = coherence_map(x, y, fs, row.window_s, row.step_s, nperseg=row.nperseg,
                                     per_decade=per_decade, pmin=row.period_floor_s,
                                     pmax=row.period_ceiling_s)
        levels.append((tc, per, coh, row.nperseg, row.window_s, row.step_s))
    return levels


def band_from_map(periods, coh, lo_s, hi_s):
    m = (periods >= lo_s) & (periods <= hi_s)
    if not m.any():
        return np.full(coh.shape[0], np.nan)
    return np.nanmean(coh[:, m], axis=1)


def band_from_levels(levels, lo_s, hi_s, t_base):
    """One band line on the base level's time grid, averaged over every level that reaches the band."""
    out = []
    for tc, per, coh, *_ in levels:
        v = band_from_map(per, coh, lo_s, hi_s)
        if np.isfinite(v).any():
            idx = np.clip(np.searchsorted(tc, t_base), 0, len(tc) - 1)
            out.append(v[idx])
    if not out:
        return np.full(len(t_base), np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(np.vstack(out), axis=0)


def levels_to_grid(levels, t_base):
    """Every level interpolated (nearest window) onto the base time grid, stacked into one (time, period) image."""
    per_all, cols = [], []
    for tc, per, coh, *_ in levels:
        idx = np.clip(np.searchsorted(tc, t_base), 0, len(tc) - 1)
        per_all.append(per)
        cols.append(coh[idx])
    per = np.concatenate(per_all)
    img = np.concatenate(cols, axis=1)
    o = np.argsort(per)
    return per[o], img[:, o]


def coherence_maps(t0, arrays, out, fs=1.0, pairs=PAIRS, win_s=3600, step_s=1800, pmin=PMIN_S,
                   pmax=PMAX_S, smooth_h=SMOOTH_H, title="", figsize=(14, 11), dpi=110):
    """Figure 03: one image per pair, every level on the base time grid, gouraud shaded.

    Returns (figure, {pair: levels}); the levels are what figure 02's band lines are read from.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    n = len(arrays[pairs[0][0]])
    _, ticks, labels = day_axis(t0, n, fs)
    maps = {}
    pc = None
    for ax, (a, b) in zip(axes, pairs):
        levels = coherence_levels(arrays[a], arrays[b], fs=fs, win_s=win_s, step_s=step_s,
                                  pmin=pmin, pmax=pmax)
        maps[(a, b)] = levels
        tc = levels[0][0]
        per, img = levels_to_grid(levels, tc)
        if smooth_h:
            w = max(1, int(round(smooth_h * 3600.0 / step_s)))
            img = pd.DataFrame(img).rolling(w, center=True, min_periods=1).median().to_numpy()
        pc = ax.pcolormesh(tc / DAY, per, img.T, vmin=0, vmax=1, cmap="viridis", shading="gouraud")
        ax.set_yscale("log")
        ax.set_ylim(per.min(), per.max())
        ax.set_ylabel("%s\nperiod (s)" % PAIR_LABEL.get((a, b), a + "-" + b))
        for p in GUIDE_S:
            ax.axhline(p, color="w", lw=0.5, ls=":", alpha=0.7)
    fig.colorbar(pc, ax=axes.tolist(), label="squared coherence", fraction=0.02, pad=0.01)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("days from %s UTC" % iso(t0))
    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=dpi)
    return fig, maps


def coherence_bands(t0, maps, out, n, fs=1.0, step_s=1800, win_min=60, pairs=PAIRS,
                    smooth_h=LINE_SMOOTH_H, title="", figsize=(14, 9), dpi=110):
    """Figure 02: the four pairs' band coherence against time, one line per band, 12 h running median.

    Ported from wamt_site_figures.py:66-82. `maps` is what coherence_maps returned, so the lines and the
    images below them are the same numbers.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    _, ticks, labels = day_axis(t0, n, fs)
    w = max(1, int(round(smooth_h * 3600.0 / step_s)))
    for ax, pair in zip(axes, pairs):
        tc = maps[pair][0][0]
        for lo, hi, lab in BANDS_S:
            v = band_from_levels(maps[pair], lo, hi, tc)
            v = pd.Series(v).rolling(w, center=True, min_periods=1).median().to_numpy()
            ax.plot(tc / DAY, v, lw=0.8, alpha=0.55, label=lab)
        ax.axhline(0.5, color="0.5", ls="--", lw=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("coh " + PAIR_LABEL[pair])
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper right", ncol=4, framealpha=0.8)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("days from %s UTC" % iso(t0))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    return fig


def band_table(maps, pairs=PAIRS) -> pd.DataFrame:
    """The median band coherence per pair per band over the record, from the same levels figure 02 draws."""
    rows = []
    for pair in pairs:
        tc = maps[pair][0][0]
        row = {"pair": PAIR_LABEL.get(pair, pair[0] + "-" + pair[1])}
        for lo, hi, lab in BANDS_S:
            v = band_from_levels(maps[pair], lo, hi, tc)
            row[lab] = round(float(np.nanmedian(v)), 3) if np.isfinite(v).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


# ---------------------------------------------------------------- spectra

def welch(x, fs=1.0, nperseg=SPECTRA_NPERSEG):
    """The Welch spectrum of the longest finite run of a channel, the DC bin dropped. Ported from :417."""
    from scipy.signal import welch as _welch
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    if not fin.any():
        return np.array([]), np.array([])
    if not fin.all():
        edges = np.flatnonzero(np.diff(np.r_[0, fin.view(np.int8), 0]))
        runs = list(zip(edges[0::2], edges[1::2]))
        a, b = max(runs, key=lambda r: r[1] - r[0])
        x = x[a:b]
    if len(x) < 16:
        return np.array([]), np.array([])
    x = x - x.mean()
    f, p = _welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return f[1:], p[1:]


def fall_band(fs=1.0):
    """(low frequency Hz, high frequency Hz) the decades of fall are measured over at this rate.

    The rule is stated over 1 mHz to 3 Hz, which is a 10 Hz record. On a 1 Hz cache 3 Hz is
    above the Nyquist frequency, so the top of the band is min(3 Hz, 0.4 x fs) and the band actually used is
    reported beside every number read from it.
    """
    return 1e-3, min(3.0, 0.4 * fs)


def spectra(arrays, out, fs=1.0, channels=CHANNELS, nperseg=SPECTRA_NPERSEG, title="",
            figsize=(10, 5.5), dpi=110, shade=(10, 1000)):
    """Figure 04: one Welch spectrum per channel on log-log axes, the 10-1000 s band shaded.

    Returns (figure, a frame of the decades of fall of each channel over the band fall_band(fs) gives). A
    sound electric line falls 4.6-5.6 decades over 1 mHz to 3 Hz; under 1 decade is an
    open input. The band used is a column of the frame, because it depends on the rate.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    f_lo, f_hi = fall_band(fs)
    band_label = "%.4g mHz - %.4g Hz" % (f_lo * 1000, f_hi)
    rows = []
    for ch in channels:
        f, p = welch(arrays[ch], fs=fs, nperseg=nperseg)
        if not len(f):
            rows.append(dict(channel=ch, decades_of_fall=np.nan, fall_band=band_label,
                             power_at_100s=np.nan))
            continue
        ax.loglog(1 / f, p, lw=0.9, label=ch)
        lo = (f >= 0.8 * f_lo) & (f <= 1.2 * f_lo)
        hi = (f >= 0.8 * f_hi) & (f <= f_hi)
        fall = (np.log10(np.median(p[lo])) - np.log10(np.median(p[hi]))) if lo.any() and hi.any() else np.nan
        band = (f >= 1 / 120.0) & (f <= 1 / 80.0)
        rows.append(dict(channel=ch, decades_of_fall=round(float(fall), 2) if np.isfinite(fall) else np.nan,
                         fall_band=band_label,
                         power_at_100s=float(np.median(p[band])) if band.any() else np.nan))
    ax.axvspan(shade[0], shade[1], color="gold", alpha=0.12)
    ax.axvline(2.0 / fs, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel="period (s)", ylabel="power (nT^2/Hz, (mV/km)^2/Hz)", title=title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    return fig, pd.DataFrame(rows).set_index("channel")


# ---------------------------------------------------------------- spectrograms

def power_map(x, fs=1.0, win_s=3600, step_s=1800, nperseg=None, per_decade=PER_DECADE, pmin=4.0, pmax=None):
    """Welch power density per window averaged into log-period bins. Ported from :564.

    A window less than half finite is left NaN; the finite samples' mean fills the rest and is removed.
    """
    from scipy.signal import welch as _welch
    nperseg = int(nperseg or auto_nperseg(win_s, fs))
    pmax = float(pmax or nperseg / fs)
    x = np.asarray(x, float)
    w, st = int(round(win_s * fs)), int(round(step_s * fs))
    starts = np.arange(0, max(1, len(x) - w + 1), st, dtype=int)
    per = 1.0 / np.fft.rfftfreq(nperseg, 1.0 / fs)[1:]
    nb = max(1, int(round(per_decade * np.log10(pmax / pmin))))
    edges = np.logspace(np.log10(pmin), np.log10(pmax), nb + 1)
    which = np.digitize(per, edges) - 1
    inb = (which >= 0) & (which < nb)
    counts = np.bincount(which[inb], minlength=nb)
    P = np.full((len(starts), nb), np.nan)
    for i, s in enumerate(starts):
        seg = x[s:s + w]
        fin = np.isfinite(seg)
        if fin.mean() < 0.5:
            continue
        m = seg[fin].mean()
        seg = np.where(fin, seg, m) - m
        _, p = _welch(seg, fs=fs, nperseg=min(nperseg, len(seg)))
        p = p[1:]
        if len(p) != len(per):
            continue
        sums = np.bincount(which[inb], weights=p[inb], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            P[i] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    keep = counts > 0
    centres = np.sqrt(edges[:-1] * edges[1:])
    return (starts + w / 2.0) / fs, centres[keep], P[:, keep]


def power_levels(x, fs=1.0, win_s=3600, step_s=1800, pmin=PMIN_S, pmax=PMAX_S, per_decade=PER_DECADE,
                 min_segments=MIN_SEGMENTS, long_segments=LONG_SEGMENTS, factor=LEVEL_FACTOR):
    """power_map once per level, the same ladder as coherence_levels."""
    levels = []
    for row in levels_plan(fs, win_s, step_s, pmin, pmax, min_segments, long_segments,
                           factor).itertuples():
        tc, per, P = power_map(x, fs, row.window_s, row.step_s, nperseg=row.nperseg,
                               per_decade=per_decade, pmin=row.period_floor_s, pmax=row.period_ceiling_s)
        levels.append((tc, per, P, row.nperseg, row.window_s, row.step_s))
    return levels


def spectrograms(t0, arrays, out, fs=1.0, channels=CHANNELS, win_s=3600, step_s=1800, pmin=PMIN_S,
                 pmax=PMAX_S, smooth_h=SMOOTH_H, title="", figsize=(14, 13), dpi=110):
    """Figure 05: one image per channel in dB of absolute power density, limits at the 2nd and 98th percentile.

    Returns (figure, {channel: levels}); spectrogram_table reads the base level of those maps.
    """
    import matplotlib.pyplot as plt
    channels = [c for c in channels if c in arrays]
    fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    n = len(arrays[channels[0]])
    _, ticks, labels = day_axis(t0, n, fs)
    maps = {}
    for ax, ch in zip(axes, channels):
        levels = power_levels(arrays[ch], fs=fs, win_s=win_s, step_s=step_s, pmin=pmin, pmax=pmax)
        maps[ch] = levels
        tc = levels[0][0]
        per, img = levels_to_grid(levels, tc)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(img)
        if smooth_h:
            w = max(1, int(round(smooth_h * 3600.0 / step_s)))
            db = pd.DataFrame(db).rolling(w, center=True, min_periods=1).median().to_numpy()
        vmin, vmax = np.nanpercentile(db, [2, 98]) if np.isfinite(db).any() else (-1.0, 1.0)
        pc = ax.pcolormesh(tc / DAY, per, db.T, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
        ax.set_yscale("log")
        ax.set_ylim(per.min(), per.max())
        ax.set_ylabel("%s\nperiod (s)" % ch)
        for p in GUIDE_S:
            ax.axhline(p, color="w", lw=0.5, ls=":", alpha=0.7)
        fig.colorbar(pc, ax=ax, label="dB (%s)^2/Hz" % UNIT.get(ch, ""), fraction=0.02, pad=0.01)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("days from %s UTC" % iso(t0))
    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=dpi)
    return fig, maps


def spectrogram_table(maps, lo_s=20.0, hi_s=200.0, db_away=10.0) -> pd.DataFrame:
    """Per channel from the base level: the median power in the band in dB and the fraction of windows
    more than `db_away` = 10 dB below it (quiet or dead) or above it (bursts and storms). Ported from :669."""
    rows = []
    for ch, levels in maps.items():
        tc, per, P = levels[0][:3]
        m = (per >= lo_s) & (per <= hi_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            band = 10.0 * np.log10(np.nanmean(P[:, m], axis=1))
        med = np.nanmedian(band)
        fin = np.isfinite(band)
        rows.append({"channel": ch,
                     "median dB at %g-%g s" % (lo_s, hi_s): round(float(med), 1),
                     "windows %g dB below" % db_away: round(float(np.mean(band[fin] < med - db_away)), 3)
                     if fin.any() else np.nan,
                     "windows %g dB above" % db_away: round(float(np.mean(band[fin] > med + db_away)), 3)
                     if fin.any() else np.nan,
                     "windows": int(fin.sum())})
    return pd.DataFrame(rows).set_index("channel")
