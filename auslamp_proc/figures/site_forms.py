"""The figures of workbook 05: what each method did to one site, beside the table that scored it.

One helper per figure, each taking what its section already computed and writing a PNG at dpi 110 into the
run folder. Nothing is estimated here: every number drawn is one the section printed.

Each figure carries a SHORT title -- the site and what the figure is, one line that fits at any width -- and
a CAPTION under the axes in smaller text, wrapped to the figure's width, carrying what was done and with
which values. Both go through figures.common.finish, which reserves the caption's space before the save.

The rest of the conventions are the package's. Period is a log x axis labelled `period (s)`; apparent
resistivity is log in Ohm.m; phase runs 0-90 deg with the yx panel labelled `+ 180 deg`; time series carry
`channel (unit)` in nT and mV/km against `days from <t0> UTC`, drawn as a per-minute mean over a per-minute
envelope; series are C0..C9, masked spans grey, the xy and yx component spans blue and red at low alpha;
grids at alpha 0.25. The transfer-function panels are drawn through figures.products.tf_panels and dressed by
its own _dress, so a form page and a workbook 04 page read the same way.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .common import finish
from .products import _dress, tf_panels
from .record import minute_stats

DPI = 110
DAY = 86400.0
GRID_ALPHA = 0.25
SPAN_COLOUR = {"xy": "tab:blue", "yx": "tab:red"}
UNIT = {"Hx": "nT", "Hy": "nT", "Hz": "nT", "Ex": "mV/km", "Ey": "mV/km"}


def _fig(nrows=1, ncols=1, figsize=(13, 6), **kw):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, ax


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- section 2

def daily_magnetics(dm, site, out, neighbours=(), f_tolerance=0.05, factor=3.0):
    """Three strips per day: the field against IGRF and the Hx and Hz fluctuation ratios."""
    fig, ax = _fig(3, 1, figsize=(13, 7.5), sharex=True)
    x = np.arange(len(dm))
    ok = dm.sound.to_numpy(bool)
    judged = dm.judged.to_numpy(bool)
    # a list, not a numpy string array: "0.6" written into a <U2 array truncates to "0." and draws black
    colour = ["0.6" if not j else ("C2" if s else "C3") for s, j in zip(ok, judged)]
    ax[0].axhspan(1 - f_tolerance, 1 + f_tolerance, color="C2", alpha=0.12, lw=0)
    ax[0].bar(x, dm.F_ratio.to_numpy(float) - 1.0, bottom=1.0, color=colour, width=0.8)
    ax[0].axhline(1.0, color="0.4", lw=0.8)
    ax[0].set(ylabel="F / F IGRF", ylim=(0.8, 1.2))
    for k, (num, den, lab) in enumerate((("hx_std", "nb_hx_std", "Hx / the neighbours'"),
                                         ("hz_std", "nb_hz_std", "Hz / the neighbours'"))):
        r = dm[num].to_numpy(float) / np.where(dm[den].to_numpy(float) > 0, dm[den].to_numpy(float), np.nan)
        a = ax[k + 1]
        a.axhspan(1.0 / factor, factor, color="C2", alpha=0.12, lw=0)
        a.bar(x, r, color=colour, width=0.8)
        a.axhline(1.0, color="0.4", lw=0.8)
        a.set(ylabel=lab, yscale="log")
    for a in ax:
        a.grid(alpha=GRID_ALPHA)
    step = max(1, len(dm) // 20)
    ax[-1].set_xticks(x[::step])
    ax[-1].set_xticklabels([d[5:] for d in dm.day][::step], rotation=90, fontsize=7)
    ax[-1].set_xlabel("UTC day")
    return finish(fig, "%s: the magnetics day by day" % site,
                  "Each UTC day's median total field over IGRF, and the standard deviation of its Hx and Hz "
                  "after a 3,000 s high-pass over the median of the nearest sites recording that day (%s). "
                  "A day is sound (green) where the field is within %.0f per cent of IGRF and both "
                  "fluctuations sit within a factor %g of the neighbours'; a day outside either band is red, "
                  "and a day with no neighbour is grey and judged on the field alone. %d of %d day(s) are "
                  "sound and %d were judged against a neighbour."
                  % (", ".join(neighbours) or "no neighbour", 100 * f_tolerance, factor,
                     int(ok.sum()), len(dm), int(judged.sum())), out)


def fleet_bars(ft, site, out, near_km=150.0, floor=0.3):
    """One bar per pair of the fleet table, with the shifted pair of each neighbour beside it."""
    t = ft.get("table")
    fig, ax = _fig(2, 1, figsize=(13, 7), sharex=False)
    if t is None or not len(t):
        ax[0].text(0.5, 0.5, "no pair scored", ha="center")
        return finish(fig, "%s: the fleet" % site,
                      "No other site within %g km covers the stretch." % near_km, out)
    mine = t[(t.a == site) | (t.b == site)]
    ctrl = t[(t.a != site) & (t.b != site)]
    order = list(mine.itertuples()) + list(ctrl.itertuples())
    labels = ["%s-%s" % (r.a, r.b) for r in order]
    x = np.arange(len(order))
    shifted = {r["site"]: r for r in (ft.get("shifted_table").to_dict("records")
                                      if ft.get("shifted_table") is not None
                                      and len(ft["shifted_table"]) else [])}
    shift_h = float(ft.get("shift_s", 43200.0)) / 3600.0
    for k, chan in enumerate(("Hx", "Hy")):
        a = ax[k]
        vals = [getattr(r, chan) for r in order]
        cols = ["C0" if (r.a == site or r.b == site) else "0.65" for r in order]
        a.bar(x - 0.2, vals, width=0.4, color=cols, label="the pair")
        sh = []
        for r in order:
            other = r.b if r.a == site else (r.a if r.b == site else None)
            sh.append(shifted.get(other, {}).get(chan, np.nan) if other else np.nan)
        a.bar(x + 0.2, sh, width=0.4, color="C3", alpha=0.75, hatch="//",
              label="shifted %g h" % shift_h)
        a.axhline(floor, color="C3", ls=":", lw=1)
        a.axhline(0.8 * (ctrl[chan].median() if len(ctrl) else np.nan), color="C2", ls="--", lw=1)
        a.set(ylabel="%s coherence" % chan, ylim=(0, 1))
        a.grid(alpha=GRID_ALPHA)
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=90, fontsize=7)
        if k == 0:
            a.legend(fontsize=8, loc="lower left", ncol=2)
    return finish(fig, "%s: the fleet at 100-1000 s" % site,
                  "Every pair of the site and the sites covering %s .. %s, blue where the site is in the "
                  "pair and grey for the control pairs. The hatched bar beside each of the site's own pairs "
                  "is that same pair with the neighbour's record taken %g h later, which is the negative "
                  "control: two records of the same fleet that cannot share a field. The dotted line is the "
                  "%.1f the shifted pairs must stay under and the dashed line the 0.8 of the control pairs' "
                  "median the site's own must reach. The site reads Hx %.2f and Hy %.2f; the shifted pairs "
                  "read Hx %.2f and Hy %.2f over %d pair(s)."
                  % (_iso(ft.get("t_start", 0)), _iso(ft.get("t_end", 0)), shift_h, floor,
                     ft.get("site_hx", np.nan), ft.get("site_hy", np.nan), ft.get("shifted_hx", np.nan),
                     ft.get("shifted_hy", np.nan), ft.get("n_shifted", 0)), out)


def clock_lags(ck, site, out, pass_s=10.0, edge_fraction=0.95, peak_ratio=1.5, maxlag_s=43200.0):
    """The per-day lag, the peak correlation and the peak's stand above that day's own other lags.

    A counted day is filled and a day refused by one of the guards is open, with the edge band shaded so an
    edge hit is read as the window and not as a lag.
    """
    t = ck.get("table")
    fig, ax = _fig(3, 1, figsize=(13, 8), sharex=True)
    if t is None or not len(t):
        ax[0].text(0.5, 0.5, "no day was timed", ha="center")
        return finish(fig, "%s: the clock" % site,
                      "No day was timed: %s." % ck.get("reason", "no reference covered the record"), out)
    x = np.arange(len(t))
    counted = t.counted.to_numpy(bool)
    lim = 1.1 * maxlag_s
    ax[0].axhspan(edge_fraction * maxlag_s, lim, color="C3", alpha=0.12, lw=0, label="the search edge")
    ax[0].axhspan(-lim, -edge_fraction * maxlag_s, color="C3", alpha=0.12, lw=0)
    ax[0].axhspan(-pass_s, pass_s, color="C2", alpha=0.25, lw=0, label="+-%.0f s" % pass_s)
    ax[0].scatter(x[counted], t.lag_s.to_numpy(float)[counted], s=20, color="C0", label="counted")
    ax[0].scatter(x[~counted], t.lag_s.to_numpy(float)[~counted], s=20, facecolors="none", color="0.6",
                  label="refused by a guard")
    if np.isfinite(ck.get("median_lag_s", np.nan)):
        ax[0].axhline(ck["median_lag_s"], color="C3", ls="--", lw=1,
                      label="median %+.2f s" % ck["median_lag_s"])
    ax[0].set(ylabel="lag (s)", yscale="symlog", ylim=(-lim, lim))
    ax[0].legend(fontsize=7, loc="lower left", ncol=5)
    ax[1].bar(x, t.peak_r.to_numpy(float), color=np.where(counted, "C0", "0.75"), width=0.8)
    ax[1].axhline(0.5, color="C3", ls=":", lw=1)
    ax[1].set(ylabel="peak correlation", ylim=(-1, 1))
    if "peak_over_floor" in t.columns:
        v = t.peak_over_floor.to_numpy(float)
        ax[2].bar(x, np.where(np.isfinite(v), v, 0.0), color=np.where(counted, "C0", "0.75"), width=0.8)
        ax[2].axhline(peak_ratio, color="C3", ls=":", lw=1)
        ax[2].set(ylabel="peak over the day's other lags")
    for a in ax:
        a.grid(alpha=GRID_ALPHA)
    step = max(1, len(t) // 20)
    ax[2].set_xticks(x[::step])
    ax[2].set_xticklabels([d[5:] for d in t.day][::step], rotation=90, fontsize=7)
    ax[2].set_xlabel("UTC day")
    band = ck.get("band_s") or [5, 20]
    return finish(fig, "%s: the clock against %s" % (site, ck.get("ref")),
                  "The lag of the peak cross-correlation of the despiked, %g-%g s band-passed Hx against "
                  "%s's, searched to +-%g h and refined by a parabola through the peak's two neighbours. A "
                  "day is counted (filled) only where the total field is within 20 per cent of IGRF, the "
                  "peak correlation reaches 0.5, it stands %.1fx above the median correlation over that "
                  "day's other lags, and it is not inside the shaded edge; the green band is the +-%.0f s "
                  "the median must fall in. %d day(s) of %d counted%s."
                  % (band[0], band[1], ck.get("ref"), maxlag_s / 3600.0, peak_ratio, pass_s,
                     int(ck.get("n_days", 0)), len(t),
                     (", median %+.2f s" % ck["median_lag_s"]) if ck.get("judged")
                     else ", which is under the 5 the test needs and leaves the clock UNJUDGED"), out)


# ---------------------------------------------------------------- section 3

def quality_images(qm, site, out, thr=0.5):
    """The quality map as two images: hourly 4-50 s, and daily 50-1000 s local and observatory."""
    fig, ax = _fig(2, 1, figsize=(13, 7))
    h = np.asarray(qm["hourly"], float).T
    im = ax[0].imshow(h, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
                      extent=[0, len(qm["hourly"]) / 24.0, -0.5, 1.5])
    ax[0].set(yticks=[0, 1], yticklabels=["Ex with Hy", "Ey with Hx"],
              ylabel="hourly 4-50 s", xlabel="")
    fig.colorbar(im, ax=ax[0], pad=0.01)
    d = np.asarray(qm["daily"], float).T
    im2 = ax[1].imshow(d, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
                       extent=[0, len(qm["daily"]), -0.5, 3.5])
    ax[1].set(yticks=[0, 1, 2, 3],
              yticklabels=["Ex local", "Ey local", "Ex %s" % qm["observatory"],
                           "Ey %s" % qm["observatory"]],
              ylabel="daily 50-1000 s", xlabel="days from %s UTC" % _iso(qm["t0"]))
    fig.colorbar(im2, ax=ax[1], pad=0.01)
    return finish(fig, "%s: the quality map" % site,
                  "Above, the bias-corrected coherence of each electric line with the H it couples to over "
                  "4-50 s, one value an hour across %d hour(s). Below, the bias-corrected multiple coherence "
                  "of each line with the local pair and with %s over 50-1000 s, one value a UTC day across "
                  "%d day(s). The day mask of the next cell is taken on the two %s rows, which carry no "
                  "local magnetic noise, at or above %.2f."
                  % (qm["hours"], qm["observatory"], qm["n_days"], qm["observatory"], thr), out)


def record_spans(t0, arrays, out, site="", channels=("Hx", "Ex", "Ey"), spans=(), title="", caption="",
                 fs=1.0, figsize=(13, 7)):
    """The record as a per-minute mean over its per-minute envelope, with named spans drawn over it.

    `spans` is [(label, colour, hatch, [(t_a, t_b), ...])] in unix seconds.
    """
    fig, ax = _fig(len(channels), 1, figsize=figsize, sharex=True)
    ax = np.atleast_1d(ax)
    n = len(arrays[channels[0]])
    per = int(round(60 * fs))
    days = (np.arange(n // per) * per) / (fs * DAY)
    for k, ch in enumerate(channels):
        m, lo, hi = minute_stats(np.asarray(arrays[ch], float), per)
        a = ax[k]
        a.fill_between(days[:len(m)], lo, hi, color="0.8", lw=0)
        a.plot(days[:len(m)], m, color="C%d" % k, lw=0.6)
        fin = np.isfinite(arrays[ch])
        if fin.any():
            v = np.asarray(arrays[ch], float)[fin]
            a.set_ylim(np.percentile(v, 1), np.percentile(v, 99))
        a.set(ylabel="%s (%s)" % (ch, UNIT.get(ch, "")))
        a.grid(alpha=GRID_ALPHA)
    for label, colour, hatch, items in spans:
        for j, (ta, tb) in enumerate(items):
            for a in ax:
                a.axvspan((ta - t0) / DAY, (tb - t0) / DAY, color=colour, alpha=0.18, lw=0,
                          hatch=hatch, label=(label if j == 0 and a is ax[0] else None))
    if spans:
        ax[0].legend(fontsize=8, loc="upper right", ncol=len(spans))
    ax[-1].set_xlabel("days from %s UTC" % _iso(t0))
    return finish(fig, title or "%s: the record and the selections" % site, caption, out)


def mask_spans(keep, t0, fs=1.0, max_spans=400):
    """[(t_a, t_b)] of the True runs of a keep mask, in unix seconds."""
    k = np.asarray(keep, bool)
    d = np.diff(np.concatenate(([0], k.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = [(t0 + a / fs, t0 + b / fs) for a, b in zip(st, en)]
    return out[:max_spans]


# ---------------------------------------------------------------- the product panels

def form_panels(curves, site, out, title="", caption="", period_range=None, figsize=(13, 8),
                tipper=False):
    """Several transfer functions on the rho and phase panels of a page.

    `curves` is [(label, TFData, colour, linestyle)]. The panels and their limits are workbook 04's, so a
    form page and a product page read the same way.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2 + (1 if tipper else 0), 2, figsize=figsize, sharex=True)
    ax_rho_xy, ax_rho_yx = axes[0]
    ax_ph_xy, ax_ph_yx = axes[1]
    ax_tzx, ax_tzy = (axes[2] if tipper else (None, None))
    rho = []
    for label, tf, colour, ls in curves:
        if tf is None:
            continue
        rho += tf_panels(ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy, tf, label,
                         colour=colour, ls=ls, marker="o", period_range=period_range, bars=True)
    _dress((ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy), rho, period_range)
    ax_rho_xy.legend(fontsize=8, loc="best")
    return finish(fig, title or site, caption, out)


def rate_panels(curves, site, out, join_s=16.0, bands=((8.0, 16.0), (32.0, 100.0)),
                period_range=(0.6, 2000.0), title="", caption="", figsize=(13, 8)):
    """The 10 Hz forms against the 1 Hz baseline, with the join period and the two ruled bands drawn.

    `curves` is [(label, TFData, colour, linestyle)] as for form_panels -- the 1 Hz baseline solid and the
    10 Hz forms dashed. The vertical line is the period the short end would join the 1 Hz row at and the
    shaded columns are the two bands the step at that join is scored on, so the figure shows the 10 Hz
    product over the part of the spectrum a splice would ever use.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    ax_rho_xy, ax_rho_yx = axes[0]
    ax_ph_xy, ax_ph_yx = axes[1]
    rho = []
    for label, tf, colour, ls in curves:
        if tf is None:
            continue
        rho += tf_panels(ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, None, None, tf, label,
                         colour=colour, ls=ls, marker="o", period_range=period_range, bars=True)
    _dress((ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, None, None), rho, period_range)
    for a in (ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx):
        for k, (lo, hi) in enumerate(bands):
            a.axvspan(lo, hi, color="C7", alpha=0.13, lw=0)
        a.axvline(join_s, color="0.3", ls="-.", lw=1)
    ax_rho_xy.legend(fontsize=8, loc="best")
    return finish(fig, title or "%s: the 10 Hz forms against the 1 Hz baseline" % site, caption, out)


# ---------------------------------------------------------------- section 5

def hour_scores(hours, site, out, comps=("xy", "yx"), contig_pick=None, figsize=(13, 7)):
    """The hour score per component with the kept, random and contiguous selections as three rows of spans."""
    comps = [c for c in comps if c in hours]
    fig, ax = _fig(max(1, len(comps)), 1, figsize=figsize, sharex=True)
    ax = np.atleast_1d(ax)
    band, frac, seed, n_sel = (0, 0), 0.0, 0, 0
    for k, comp in enumerate(comps):
        b = hours[comp]
        band, frac, seed, n_sel = b["band_s"], b["fraction"], b["seed"], b["n_selected"]
        a = ax[k]
        t = np.asarray(b["centres"], float) / DAY
        a.plot(t, b["score"], color="0.4", lw=0.5, label="the hour score")
        if np.isfinite(b["threshold"]):
            a.axhline(b["threshold"], color="C0", ls="--", lw=1, label="threshold %.3f" % b["threshold"])
        a.set(ylabel="%s coherence" % comp, ylim=(0, max(0.2, float(np.nanmax(b["score"])) * 1.05)))
        a.grid(alpha=GRID_ALPHA)
        top = a.get_ylim()[1]
        rows = [("kept", b["keep"], SPAN_COLOUR[comp], None),
                ("random", b["random_keep"], "0.4", None)]
        h = (contig_pick or {}).get(comp)
        if h and h in b["contiguous"]:
            rows.append(("contiguous %d h" % h, b["contiguous"][h]["keep"], "C2", "//"))
        for j, (label, keep, colour, hatch) in enumerate(rows):
            lo = top * (0.97 - 0.05 * j)
            hi = top * (1.0 - 0.05 * j)
            kk = np.asarray(keep, bool)
            d = np.diff(np.concatenate(([0], kk.view(np.int8), [0])))
            first = True
            for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                a.fill_between([s0 / DAY, s1 / DAY], lo, hi, color=colour, alpha=0.7, lw=0, hatch=hatch,
                               label=(label if first else None))
                first = False
        a.legend(fontsize=7, loc="lower left", ncol=5)
    ax[-1].set_xlabel("days from the record start")
    return finish(fig, "%s: the best hours and their controls" % site,
                  "The coherence of each component's pair over %g-%g s, one value an hour on "
                  "non-overlapping windows, with three rows of spans above it: the best %.0f per cent of the "
                  "candidate hours, the same number drawn at random from the same pool under seed %d, and "
                  "the contiguous windows tiled to the same duration. The three cost the same and only the "
                  "top row was chosen on the score, so what separates their products is the choosing. %d "
                  "window(s) were selected."
                  % (band[0], band[1], 100 * frac, seed, n_sel), out)


# ---------------------------------------------------------------- section 6

def residual_panels(sv, site, me, built, out, days=3, elines=None, band_s=(20.0, 200.0), nperseg=4096,
                    rate=1, figsize=(13, 5)):
    """The residual coherence and the complex gain against period, for the site and the built control.

    The gain is drawn over the L_N / L_E the arms predict, so the band the criterion reads is the same on
    every site whatever its arm lengths.
    """
    from ..look import highpass
    from ..site.centre import GAIN_HI, GAIN_LO, RESID_COH_MIN, _cross
    from ..site.masks import load_raw, prepare
    fig, ax = _fig(1, 2, figsize=figsize)
    b, a = highpass(float(rate), 3000.0)
    t0, own = load_raw(sv, site, rate, ("Ex", "Ey", "Hx", "Hy"))
    other = None
    if built and built.get("neighbour"):
        tn, other = load_raw(sv, built["neighbour"], rate, ("Ey",))
    table = elines[elines.coh_Ex_Ey.notna()].sort_values("coh_Ex_Ey", ascending=False) \
        if elines is not None and len(elines) else None
    day = None
    if table is not None:
        for r in table.itertuples():
            i0, i1 = int(float(r.t_start) - t0), int(float(r.t_end) - t0)
            if 0 <= i0 and i1 <= len(own["Ex"]):
                day = (r.day, i0, i1, float(r.t_start), float(r.t_end))
                break
    if day is None:
        ax[0].text(0.5, 0.5, "no usable day", ha="center")
        return finish(fig, "%s: the shared-centre residual test" % site,
                      "No day of the record carries all four channels over a whole UTC day.", out)
    _lab, i0, i1, ta, _tb = day
    g_expected = float((me or {}).get("gain_expected", 1.0) or 1.0)
    if not np.isfinite(g_expected) or g_expected <= 0:
        g_expected = 1.0
    pairs = [("%s Ex with its own Ey" % site, own["Ex"][i0:i1], own["Ey"][i0:i1], "C0")]
    if other is not None:
        j0 = int(ta - tn)
        if 0 <= j0 and j0 + (i1 - i0) <= len(other["Ey"]):
            pairs.append(("%s Ex with %s Ey, the built control" % (site, built["neighbour"]),
                          own["Ex"][i0:i1], other["Ey"][j0:j0 + (i1 - i0)], "C3"))
    for label, ex, ey, colour in pairs:
        p = {k: prepare(v, b, a) for k, v in (("ex", ex), ("ey", ey),
                                              ("hx", own["Hx"][i0:i1]), ("hy", own["Hy"][i0:i1]))}
        if any(v is None for v in p.values()):
            continue
        f, S = _cross(p, nperseg, float(rate))
        sel = (f > 1.0 / band_s[1]) & (f < 1.0 / band_s[0])
        Shh = np.stack([np.stack([S[("hx", "hx")], S[("hx", "hy")]], -1),
                        np.stack([S[("hy", "hx")], S[("hy", "hy")]], -1)], -2)[sel]
        try:
            Sinv = np.linalg.inv(Shh)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Shh)

        def resid(e1, e2):
            seh = np.stack([S[(e1, "hx")], S[(e1, "hy")]], -1)[sel]
            she = np.stack([S[("hx", e2)], S[("hy", e2)]], -1)[sel]
            return S[(e1, e2)][sel] - np.einsum("ni,nij,nj->n", seh, Sinv, she)

        rxx, ryy, rxy = resid("ex", "ex").real, resid("ey", "ey").real, resid("ex", "ey")
        per = 1.0 / f[sel]
        ax[0].plot(per, np.abs(rxy) ** 2 / (rxx * ryy), color=colour, lw=0.9, label=label)
        ax[1].plot(per, np.abs(np.conj(rxy) / rxx) / g_expected, color=colour, lw=0.9)
    ax[0].axhline(RESID_COH_MIN, color="0.3", ls="--", lw=1)
    ax[0].set(xscale="log", ylim=(0, 1), xlabel="period (s)", ylabel="residual coherence",
              title="the residuals after the H-explained part")
    ax[1].axhspan(GAIN_LO, GAIN_HI, color="C2", alpha=0.15, lw=0)
    ax[1].set(xscale="log", yscale="log", xlabel="period (s)",
              ylabel="|g| / (L_N / L_E)", title="the gain over what the arms predict")
    for a_ in ax:
        a_.grid(alpha=GRID_ALPHA, which="both")
    ax[0].legend(fontsize=8, loc="lower left")
    return finish(fig, "%s: the shared-centre residual test" % site,
                  "On %s over %g-%g s, the part of each line that (Hx, Hy) explains is removed frequency bin "
                  "by frequency bin and what is left is read: the coherence of the two residuals on the "
                  "left, and on the right the complex gain divided by the L_N / L_E = %.3f the arm lengths "
                  "predict, so 1 is what a shared centre would give. The model holds where the coherence is "
                  "at least %.1f (dashed) and the scaled gain lies in %.2f-%.2f (green). The site reads "
                  "%.2f and %.2f. The built control pairs the site's own Ex with %s's Ey against the site's "
                  "own H: two lines that cannot share a centre, and it reads %.2f and %.2f."
                  % (day[0], band_s[0], band_s[1], g_expected, RESID_COH_MIN, GAIN_LO, GAIN_HI,
                     (me or {}).get("resid_coh", np.nan), (me or {}).get("gain_ratio", np.nan),
                     (built or {}).get("neighbour", "no neighbour"),
                     (built or {}).get("resid_coh", np.nan), (built or {}).get("gain_ratio", np.nan)), out)


def diagonal_day(sv, site, out, elines=None, rate=1, figsize=(13, 6)):
    """One quiet day of the two signed lines, with the two arm-length diagonals underneath."""
    from ..process import frame as FR
    from ..site.centre import arm_lengths, diagonal_angle, diagonal_length, diagonals
    from ..site.masks import load_raw
    fig, ax = _fig(2, 1, figsize=figsize, sharex=True)
    t0, arr = load_raw(sv, site, rate, ("Ex", "Ey"))
    arms = arm_lengths(sv, site)
    ln, le = (float(arms["L_N"]), float(arms["L_E"])) if arms.get("known") else (1.0, 1.0)
    theta, dsep = diagonal_angle(ln, le), diagonal_length(ln, le)
    try:
        drow = sv.decision(site)
    except KeyError:
        drow = None
    s_ex, _d1 = FR.read_sign(None if drow is None else drow.get("sign_ex"))
    s_ey, _d2 = FR.read_sign(None if drow is None else drow.get("sign_ey"))
    i0, i1, label = 0, min(len(arr["Ex"]), int(DAY * rate)), "the first day"
    if elines is not None and len(elines):
        t = elines[elines.coh_Ex_Ey.notna()].sort_values("coh_Ex_Ey", ascending=False)
        for r in t.itertuples():
            a0, a1 = int(float(r.t_start) - t0), int(float(r.t_end) - t0)
            if 0 <= a0 and a1 <= len(arr["Ex"]):
                i0, i1, label = a0, a1, "%s (Ex-Ey coherence %.2f)" % (r.day, float(r.coh_Ex_Ey))
                break
    ex = np.asarray(arr["Ex"][i0:i1], float) * s_ex
    ey = np.asarray(arr["Ey"][i0:i1], float) * s_ey
    hrs = np.arange(len(ex)) / (3600.0 * rate)
    ax[0].plot(hrs, ex - np.nanmean(ex), color="C3", lw=0.5, label="Ex, signed")
    ax[0].plot(hrs, ey - np.nanmean(ey), color="C4", lw=0.5, label="Ey, signed")
    ax[0].set(ylabel="E (mV/km)")
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    e_d, e_v = diagonals(ex, ey, ln, le)
    ax[1].plot(hrs, e_d - np.nanmean(e_d), color="C0", lw=0.5, label="Ex' = (L_N Ex - L_E Ey)/d")
    ax[1].plot(hrs, e_v - np.nanmean(e_v), color="0.6", lw=0.5, label="Ey' = (L_E Ex + L_N Ey)/d")
    ax[1].set(ylabel="E' (mV/km)", xlabel="hours from the day's start")
    ax[1].legend(fontsize=8, loc="upper right", ncol=2)
    for a in ax:
        a.grid(alpha=GRID_ALPHA)
    return finish(fig, "%s: the two lines and the two arm diagonals" % site,
                  "One day, %s: above the two physically signed lines, below the two combinations the arm "
                  "lengths define. Ex' is the voltage between the two arm electrodes over their separation "
                  "d = %.2f m and carries no centre; Ey' is the orthogonal row and carries it doubled. With "
                  "L_N = %.4g m and L_E = %.4g m the diagonal lies at %+.2f deg from north, which at equal "
                  "arms would be -45 deg. The common motion of the two lines above is what the weighted "
                  "difference below has cancelled.%s"
                  % (label, dsep, ln, le, theta,
                     "" if arms.get("known") else
                     " The arms are not on file here, so the figure is drawn at equal lengths."), out)


# ---------------------------------------------------------------- section 7

def notch_spectra(sv, site, decision, out, freqs=(1.0, 2.0), rate=10, nperseg=1 << 16, figsize=(13, 7)):
    """The spectrum of each channel around each tone, before and after the notch, on the worst day."""
    from scipy import signal
    from ..site.variants import cache_path, day_bounds, runs_of
    chans = sorted(set(decision.channel)) if decision is not None and len(decision) else []
    fig, ax = _fig(max(1, len(chans)), len(freqs), figsize=figsize, squeeze=False)
    src, var = cache_path(sv.cfg["work_root"], site, rate), cache_path(sv.cfg["work_root"], site, rate,
                                                                      "notched")
    if not chans or not src.exists() or not var.exists():
        ax[0][0].text(0.5, 0.5, "no notched variant", ha="center")
        return finish(fig, "%s: the tone before and after the notch" % site,
                      "No notched variant was written for this site.", out)
    z0 = np.load(src, allow_pickle=False)
    z1 = np.load(var, allow_pickle=False)
    t0 = float(np.asarray(z0["t0"]).ravel()[0])
    fired = []
    for r, ch in enumerate(chans):
        rows = decision[decision.channel == ch]
        worst = str(rows.sort_values("worst_ratio").iloc[-1].worst_day)
        if bool(rows.fire.any()):
            fired.append(ch)
        x0 = np.asarray(z0[ch], np.float64)
        x1 = np.asarray(z1[ch], np.float64)
        i0 = i1 = None
        for a0, a1, lab in day_bounds(t0, len(x0), float(rate)):
            if lab == worst:
                rr = runs_of(np.isfinite(x0[a0:a1]))
                if rr:
                    s0, s1 = max(rr, key=lambda t: t[1] - t[0])
                    i0, i1 = a0 + s0, a0 + s1
                break
        if i0 is None:
            continue
        for y, lab, col in ((x0[i0:i1], "before", "C3"), (x1[i0:i1], "after", "C0")):
            g = y[np.isfinite(y)]
            if g.size < 2 * nperseg:
                continue
            f, P = signal.welch(g - g.mean(), fs=float(rate), nperseg=nperseg, noverlap=nperseg // 2,
                                detrend="linear")
            for k, f0 in enumerate(freqs):
                m = (f > f0 * 0.8) & (f < f0 * 1.2)
                ax[r][k].semilogy(f[m], P[m], color=col, lw=0.9, label=lab)
        for k, f0 in enumerate(freqs):
            did = bool(rows[rows.f0 == f0].fire.any())
            ax[r][k].axvline(f0, color="0.4", ls=":", lw=1)
            ax[r][k].set(xlabel="frequency (Hz)" if r == len(chans) - 1 else "",
                         ylabel="%s power" % ch if k == 0 else "",
                         title="%s at %.3f Hz, %s%s" % (ch, f0, worst, "" if did else " (did not fire)"))
            ax[r][k].grid(alpha=GRID_ALPHA, which="both")
            if r == 0 and k == 0:
                ax[r][k].legend(fontsize=8)
    z0.close()
    z1.close()
    return finish(fig, "%s: the tone before and after the notch" % site,
                  "One panel per channel and per tone, each on that channel's own worst day, before the "
                  "notch in red and after it in blue. The filter is scipy's iirnotch at Q = 100 applied with "
                  "filtfilt and cut at every gap of 60 s or more, so it is zero phase and cannot move the "
                  "record outside the two notch bands. %s fired at a worst-day ratio above the threshold; a "
                  "channel that did not fire is copied into the variant byte for byte, which the sha256 "
                  "check of the table beside this figure proves."
                  % (", ".join(fired) or "No channel"), out)


def ratio_panel(ratio, site, out, tone_bands=(), refused=(), figsize=(13, 5)):
    """The per-period ratio of each variant's product to the original's, by band.

    `refused` is [(form, sentence)] for a variant the run floor refused before any pass was run: it has no
    row in the table, and the sentence stands in the caption where its points would have been.
    """
    gone = " ".join("%s has no points: %s." % (n, w) for n, w in (refused or ()))
    fig, ax = _fig(1, 1, figsize=figsize)
    if ratio is None or not len(ratio):
        ax.text(0.5, 0.5, "no 10 Hz pair to compare", ha="center")
        return finish(fig, "%s: what each 10 Hz variant did to the product" % site,
                      ("No 10 Hz pair of products was made, so nothing can be compared. " + gone).strip(),
                      out)
    bands = list(dict.fromkeys(ratio.band))
    x = np.arange(len(bands))
    for k, (form, grp) in enumerate(ratio.groupby("form", sort=True)):
        for j, comp in enumerate(("xy", "yx")):
            g = grp[grp.component == comp].set_index("band").reindex(bands)
            ax.plot(x + 0.06 * (2 * k + j - 1.5), g.rho_ratio.to_numpy(float), "o",
                    color="C%d" % k, mfc=("none" if comp == "yx" else None), ms=6,
                    label="%s %s" % (form, comp))
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axhspan(0.975, 1.025, color="C2", alpha=0.15, lw=0)
    for k, b in enumerate(bands):
        if b in tone_bands:
            ax.axvspan(k - 0.5, k + 0.5, color="C3", alpha=0.10, lw=0)
    ax.set(xticks=x, yscale="log", ylabel="variant rho / original rho")
    ax.set_xticklabels(bands, rotation=0, fontsize=8)
    ax.grid(alpha=GRID_ALPHA, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    return finish(fig, "%s: what each 10 Hz variant did to the product" % site,
                  "The median apparent-resistivity ratio of each variant's product to the pass on the "
                  "original cache, band by band, both passes using the same reference and the same "
                  "parameters so the variant is the only difference. The shaded columns (%s) hold the "
                  "1.000 Hz tone and its 2.000 Hz harmonic, which the notch is there to move; the green "
                  "band is the 2.5 per cent every other band must stay inside for the variant to be a "
                  "surgical change rather than a new answer. %s"
                  % (", ".join(tone_bands) or "none", gone), out)


# ---------------------------------------------------------------- section 8

def candidate_bars(cands, site, out, threshold=0.5, donor_gate=0.8, figsize=(13, 5)):
    """Horizontal bars of the site's own coherence with each candidate, per channel, with the thresholds."""
    fig, ax = _fig(1, 2, figsize=figsize, sharey=True)
    if cands is None or not len(cands):
        ax[0].text(0.5, 0.5, "no candidate", ha="center")
        return finish(fig, "%s: the replacement candidates" % site,
                      "No neighbouring site covers enough of the record to be a candidate.", out)
    y = np.arange(len(cands))[::-1]
    for k, chan in enumerate(("Hx", "Hy")):
        a = ax[k]
        a.barh(y + 0.2, cands["own_%s_coh" % chan].to_numpy(float), height=0.38, color="C0",
               label="the site's own %s" % chan)
        a.barh(y - 0.2, cands["donor_%s_coh" % chan].to_numpy(float), height=0.38, color="0.65",
               hatch="//", label="the candidate against a third site")
        a.axvline(threshold, color="C3", ls=":", lw=1)
        a.axvline(donor_gate, color="C2", ls="--", lw=1)
        a.set(xlim=(0, 1), xlabel="%s coherence, 100-1000 s" % chan)
        a.grid(alpha=GRID_ALPHA, axis="x")
        if k == 0:
            a.set_yticks(y)
            a.set_yticklabels(["%s (%.0f km)" % (r.candidate, r.km) for r in cands.itertuples()],
                              fontsize=8)
            a.legend(fontsize=8, loc="lower right")
    fires = [r.candidate for r in cands.itertuples() if r.rule_fires]
    return finish(fig, "%s: the replacement candidates" % site,
                  "The whole-record mean coherence of the site's own channel with each candidate over "
                  "100-1000 s, both series despiked, and beside it the candidate's own coherence with the "
                  "best third site, which says whether a low reading is the site's channel or the "
                  "candidate's. The rule replaces the worse channel only, and only where its coherence is "
                  "under the dotted %.2f and under 70 per cent of the other channel's, with the candidate "
                  "itself reaching the dashed %.2f against that third site. The rule fires at %s."
                  % (threshold, donor_gate, ", ".join(fires) or "no candidate"), out)


# ---------------------------------------------------------------- section 10

def forms_bars(table, site, out, band=(10, 1000), margin=0.2, figsize=(13, 8)):
    """One horizontal bar per form on the 10-1000 s bar, its controls beside it, candidates marked."""
    fig, ax = _fig(1, 1, figsize=figsize)
    d = table[np.isfinite(table.bar_10_1000)] if table is not None and len(table) else None
    if d is None or not len(d):
        ax.text(0.5, 0.5, "no form carries a bar", ha="center")
        return finish(fig, "%s: every form on the bar" % site,
                      "No form in the run folder carries a bar.", out)
    y = np.arange(len(d))[::-1]
    colours = ["C2" if c else "C0" for c in d.candidate]
    ax.barh(y, d.bar_10_1000.to_numpy(float), height=0.6, color=colours)
    by_form = dict(zip(d.form, d.bar_10_1000))
    for k, r in enumerate(d.itertuples()):
        for c in str(r.controls or "").split(";"):
            if c and c in by_form and np.isfinite(by_form[c]):
                ax.plot([by_form[c]], [y[k]], "d", color="C3", ms=7,
                        label=("its control" if k == 0 else None))
        if r.candidate:
            ax.text(r.bar_10_1000 * 1.05, y[k], "candidate", va="center", fontsize=7, color="C2")
    ax.set(xscale="log", xlabel="the %g-%g s bar (median relative impedance error)" % band, yticks=y)
    ax.set_yticklabels(d.form, fontsize=8)
    ax.grid(alpha=GRID_ALPHA, axis="x", which="both")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], fontsize=8, loc="lower right")
    return finish(fig, "%s: every form on the bar" % site,
                  "The median relative impedance error over %g-%g s of every form in the run folder, with "
                  "each form's own control marked as a red diamond beside it. A form is a candidate for the "
                  "next workbook (green) only where it beats every control it carries by at least %.0f per "
                  "cent; %d of %d form(s) are. A form sitting on the whole record's own bar bought "
                  "efficiency and not a different answer, which is a reading and not a failure."
                  % (band[0], band[1], 100 * margin, int(d.candidate.sum()), len(d)), out)
