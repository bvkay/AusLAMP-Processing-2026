"""The figures of workbook 05: what each method did to one site, beside the table that scored it.

One helper per figure, each taking what its section already computed and writing a PNG at dpi 110 into the
run folder. Nothing is estimated here: every number drawn is one the section printed.

Each figure carries a short title -- the site and what the figure is, one line that fits at any width -- and
a caption under the axes in smaller text, wrapped to the figure's width, carrying what was done and with
which values. Both go through figures.common.finish, which reserves the caption's space before the save.

The rest of the conventions are the package's. Period is a log x axis labelled `period (s)`; apparent
resistivity is log in Ohm.m; phase runs 0-90 deg with the yx panel labelled `+ 180 deg`; time series carry
`channel (unit)` in nT and mV/km against `days from <t0> UTC`, drawn as a per-minute mean over a per-minute
envelope; series are C0..C9, masked spans grey, the xy and yx component spans blue and red at low alpha;
grids at alpha 0.25. The transfer-function panels are drawn through
figures.transfer_functions.tf_panels and dressed by its own _dress, so a form page and a workbook 04 page
read the same way.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .common import finish
from .transfer_functions import _dress, tf_panels
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
                     else ", which is under the 5 the test needs and leaves the clock unjudged"), out)
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
# -------------------------------------------------------- the transfer-function panels

def form_panels(curves, site, out, title="", caption="", period_range=None, figsize=(13, 8),
                tipper=False, comps=("xy", "yx")):
    """Several transfer functions on the rho and phase panels of a page.

    `curves` is [(label, TFData, colour, linestyle)]. The panels and their limits are workbook 04's, so a
    form page and a transfer-function page read the same way. A form is drawn for comparison; not a
    transfer function.

    `comps` names the two components the panel titles carry. Every curve on one page has to be in one frame
    for the page to mean anything, so a page drawn in a turned frame passes the turned names -- x'y' and
    y'x' for the arm diagonal -- and every tensor on it is turned into that frame before it is drawn.
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
    _dress((ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy), rho, period_range, comps=comps)
    ax_rho_xy.legend(fontsize=8, loc="best")
    return finish(fig, title or site, caption, out)


def rate_panels(curves, site, out, join_s=16.0, bands=((8.0, 16.0), (32.0, 100.0)),
                period_range=(0.6, 2000.0), title="", caption="", figsize=(13, 8)):
    """The 10 Hz forms against the 1 Hz baseline, with the join period and the two ruled bands drawn.

    `curves` is [(label, TFData, colour, linestyle)] as for form_panels -- the 1 Hz baseline solid and the
    10 Hz forms dashed, each drawn for comparison; not a transfer function. The vertical line is the period
    the short end would join the 1 Hz row at and the shaded columns are the two bands the step at that join
    is scored on, so the figure shows the 10 Hz transfer function over the part of the spectrum a splice
    would ever use.
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


# ---------------------------------------------------------------- section 11

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


# ---------------------------------------------------------------- the recipe

def recipe_spans(coh, rows, site, out, coh_min=0.5, title="", caption="", figsize=(13, 7)):
    """The hourly coherence of each recorded line with its H, and each row's stretch drawn over it.

    `coh` is process.selection.site_scores's table and `rows` is [(label, colour, hatch, [(t_a, t_b), ...])] in
    unix seconds, the rows of spans drawn above the series in the order given.
    """
    pairs = (("xy", "Ex with Hy"), ("yx", "Ey with Hx"))
    if coh is None or not len(coh):
        fig, ax = _fig(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "the record holds no whole UTC hour to score", ha="center")
        return finish(fig, title or "%s: the recipe's stretches" % site, caption, out)
    t0 = float(coh.t_start.iloc[0])
    days = (np.asarray(coh.t_start, float) - t0) / DAY
    fig, ax = _fig(len(pairs), 1, figsize=figsize, sharex=True)
    ax = np.atleast_1d(ax)
    top = 1.02 + 0.055 * max(1, len(rows)) + 0.02
    for k, (comp, label) in enumerate(pairs):
        a = ax[k]
        a.plot(days, np.asarray(coh["coh_%s" % comp], float), color=SPAN_COLOUR[comp], lw=0.6, label=label)
        a.axhline(float(coh_min), color="0.3", ls="--", lw=1, label="%.2f" % float(coh_min))
        a.set(ylabel="%s coherence" % label, ylim=(0, top), yticks=[0, 0.25, 0.5, 0.75, 1.0])
        a.grid(alpha=GRID_ALPHA)
        for j, (name, colour, hatch, items) in enumerate(rows):
            lo, hi = 1.02 + 0.055 * j, 1.06 + 0.055 * j
            for i, (ta, tb) in enumerate(items):
                a.fill_between([(ta - t0) / DAY, (tb - t0) / DAY], lo, hi, color=colour, alpha=0.75, lw=0,
                               hatch=hatch, label=(name if (i == 0 and k == 0) else None))
        a.legend(fontsize=7, loc="lower left", ncol=2 + len(rows))
    ax[-1].set_xlabel("days from %s UTC" % _iso(t0))
    return finish(fig, title or "%s: the recipe's stretches over the coherence they were chosen on" % site,
                  caption, out)


def recipe_transfer_function(curves, site, out, join_s=None, title="", caption="", period_range=(1, 50000),
                   figsize=(13, 8)):
    """The assembled transfer function against the baseline and the controls, with the y row's last period.

    `curves` is [(label, TFData, colour, linestyle)] as for form_panels. The vertical line is the longest
    period the y row reaches: above it the assembled file carries the EDI empty value on that row, and the
    two rows stop being the same measurement of the same span of time.
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
    if join_s and np.isfinite(join_s):
        for a in (ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx):
            a.axvline(float(join_s), color="0.3", ls="-.", lw=1)
    ax_rho_xy.legend(fontsize=8, loc="best")
    return finish(fig, title or "%s: the assembled recipe" % site, caption, out)
