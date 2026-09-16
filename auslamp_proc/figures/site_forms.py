"""The figures of workbook 05: what each method did to one site, beside the table that scored it.

One helper per figure, each taking what its section already computed and writing a PNG at dpi 110 into the
run folder. Nothing is estimated here: every number drawn is one the section printed.

The conventions are the package's. Period is a log x axis labelled `period (s)`; apparent resistivity is log
in Ohm.m; phase runs 0-90 deg with the yx panel labelled `+ 180 deg`; time series carry `channel (unit)` in
nT and mV/km against `days from <t0> UTC`, drawn as a per-minute mean over a per-minute envelope; series are
C0..C9, masked spans grey, the xy and yx component spans blue and red at low alpha; grids at alpha 0.25. The
transfer-function panels are drawn through figures.products.tf_panels and dressed by its own _dress, so a
form page and a workbook 04 page read the same way.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


def _save(fig, out, tight=True):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- section 2

def daily_magnetics(dm, site, out, neighbours=(), f_tolerance=0.05, factor=3.0):
    """Three strips per day: the field against IGRF and the Hx and Hz fluctuation ratios against the
    neighbours', with the pass band shaded and a day outside any band in red."""
    fig, ax = _fig(3, 1, figsize=(13, 7.5), sharex=True)
    x = np.arange(len(dm))
    ok = dm.sound.to_numpy(bool)
    judged = dm.judged.to_numpy(bool)
    colour = np.where(ok, "C2", "C3")
    colour[~judged] = "0.6"
    ax[0].axhspan(1 - f_tolerance, 1 + f_tolerance, color="C2", alpha=0.12, lw=0)
    ax[0].bar(x, dm.F_ratio.to_numpy(float) - 1.0, bottom=1.0, color=colour, width=0.8)
    ax[0].axhline(1.0, color="0.4", lw=0.8)
    ax[0].set(ylabel="F / F IGRF", ylim=(0.8, 1.2))
    for k, (num, den, lab) in enumerate((("hx_std", "nb_hx_std", "Hx fluctuation / the neighbours'"),
                                         ("hz_std", "nb_hz_std", "Hz fluctuation / the neighbours'"))):
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
    ax[0].set_title("%s: the daily magnetics test against %s -- green sound, red not, grey judged on the "
                    "field alone (%d of %d day(s) sound)"
                    % (site, ", ".join(neighbours) or "no neighbour", int(ok.sum()), len(dm)))
    return _save(fig, out)


def fleet_bars(ft, site, out, near_km=150.0, floor=0.3):
    """One bar per pair of the fleet table, the site's own pairs first, with the shifted pair beside each."""
    t = ft.get("table")
    fig, ax = _fig(2, 1, figsize=(13, 7), sharex=False)
    if t is None or not len(t):
        ax[0].text(0.5, 0.5, "no pair scored", ha="center")
        return _save(fig, out)
    mine = t[(t.a == site) | (t.b == site)]
    ctrl = t[(t.a != site) & (t.b != site)]
    order = list(mine.itertuples()) + list(ctrl.itertuples())
    labels = ["%s-%s" % (r.a, r.b) for r in order]
    x = np.arange(len(order))
    shifted = {r["site"]: r for r in (ft.get("shifted_table").to_dict("records")
                                      if ft.get("shifted_table") is not None
                                      and len(ft["shifted_table"]) else [])}
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
              label="the same pair, shifted %g h" % (ft.get("shift_s", 43200.0) / 3600.0))
        a.axhline(floor, color="C3", ls=":", lw=1)
        a.axhline(0.8 * (ctrl[chan].median() if len(ctrl) else np.nan), color="C2", ls="--", lw=1)
        a.set(ylabel="%s coherence" % chan, ylim=(0, 1))
        a.grid(alpha=GRID_ALPHA)
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=90, fontsize=7)
        if k == 0:
            a.legend(fontsize=8, loc="lower left", ncol=2)
    ax[0].set_title("%s: the fleet at 100-1000 s over %s .. %s -- blue the site's own pairs, grey the control "
                    "pairs, hatched the shifted control (dotted %.1f, dashed 0.8 of the control median)"
                    % (site, _iso(ft.get("t_start", 0)), _iso(ft.get("t_end", 0)), floor))
    return _save(fig, out)


def clock_lags(ck, site, out, pass_s=10.0, edge_fraction=0.95, peak_ratio=1.5, maxlag_s=43200.0):
    """The per-day lag, the peak correlation and the peak's stand above that day's own other lags.

    A counted day is filled and a day refused by one of the guards is open, with the edge band shaded so an
    edge hit is read as the window and not as a lag.
    """
    t = ck.get("table")
    fig, ax = _fig(3, 1, figsize=(13, 8), sharex=True)
    if t is None or not len(t):
        ax[0].text(0.5, 0.5, "no day was timed (%s)" % ck.get("reason", ""), ha="center")
        return _save(fig, out)
    x = np.arange(len(t))
    counted = t.counted.to_numpy(bool)
    lim = 1.1 * maxlag_s
    ax[0].axhspan(edge_fraction * maxlag_s, lim, color="C3", alpha=0.12, lw=0, label="the search edge")
    ax[0].axhspan(-lim, -edge_fraction * maxlag_s, color="C3", alpha=0.12, lw=0)
    ax[0].axhspan(-pass_s, pass_s, color="C2", alpha=0.25, lw=0, label="+-%.0f s, the pass band" % pass_s)
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
    ax[0].set_title("%s: the clock against %s on the %g-%g s band -- a day counts only where the field is "
                    "within 20 per cent of IGRF, the peak reaches 0.5, it stands %.1fx above that day's own "
                    "other lags and it is not in the shaded edge"
                    % (site, ck.get("ref"), band[0], band[1], peak_ratio))
    return _save(fig, out)


# ---------------------------------------------------------------- section 3

def quality_images(qm, site, out, thr=0.5):
    """The quality map as two images: the hourly 4-50 s coherence and the daily 50-1000 s multiple
    coherence, local and observatory."""
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
    ax[0].set_title("%s: the quality map -- the day mask is taken on the two %s rows, which carry no local "
                    "magnetic noise, at or above %.2f" % (site, qm["observatory"], thr))
    return _save(fig, out)


def record_spans(t0, arrays, out, site="", channels=("Hx", "Ex", "Ey"), spans=(), title="", fs=1.0,
                 figsize=(13, 7)):
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
    ax[0].set_title(title or "%s: the record with the selections drawn over it" % site)
    return _save(fig, out)


def mask_spans(keep, t0, fs=1.0, max_spans=400):
    """[(t_a, t_b)] of the True runs of a keep mask, in unix seconds."""
    k = np.asarray(keep, bool)
    d = np.diff(np.concatenate(([0], k.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = [(t0 + a / fs, t0 + b / fs) for a, b in zip(st, en)]
    return out[:max_spans]


# ---------------------------------------------------------------- the product panels

def form_panels(curves, site, out, title="", period_range=None, figsize=(13, 8), tipper=False):
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
    fig.suptitle(title or site, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out, tight=False)


# ---------------------------------------------------------------- section 5

def hour_scores(hours, site, out, comps=("xy", "yx"), contig_pick=None, figsize=(13, 7)):
    """The hour score per component with the kept, random and contiguous selections as three rows of spans."""
    comps = [c for c in comps if c in hours]
    fig, ax = _fig(max(1, len(comps)), 1, figsize=figsize, sharex=True)
    ax = np.atleast_1d(ax)
    for k, comp in enumerate(comps):
        b = hours[comp]
        a = ax[k]
        t = np.asarray(b["centres"], float) / DAY
        a.plot(t, b["score"], color="0.4", lw=0.5, label="the hour score, %g-%g s"
               % (b["band_s"][0], b["band_s"][1]))
        if np.isfinite(b["threshold"]):
            a.axhline(b["threshold"], color="C0", ls="--", lw=1,
                      label="the selection's threshold %.3f" % b["threshold"])
        a.set(ylabel="%s coherence" % comp, ylim=(0, max(0.2, float(np.nanmax(b["score"])) * 1.05)))
        a.grid(alpha=GRID_ALPHA)
        top = a.get_ylim()[1]
        rows = [("kept", b["keep"], SPAN_COLOUR[comp], None),
                ("random, seed %d" % b["seed"], b["random_keep"], "0.4", None)]
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
    ax[0].set_title("%s: the best hours, with the kept, random and contiguous selections as three rows of "
                    "spans -- the three cost the same and only the top row was chosen on the score" % site)
    return _save(fig, out)


# ---------------------------------------------------------------- section 6

def residual_panels(sv, site, me, built, out, days=3, elines=None, band_s=(20.0, 200.0), nperseg=4096,
                    rate=1, figsize=(13, 5)):
    """The residual coherence and the complex gain against frequency, for the site and the built control.

    The bands the criterion reads are drawn: 0.9 on the coherence and 0.85-1.18 on the gain.
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
        return _save(fig, out)
    _lab, i0, i1, ta, _tb = day
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
        ax[1].plot(per, np.abs(np.conj(rxy) / rxx), color=colour, lw=0.9)
    ax[0].axhline(RESID_COH_MIN, color="0.3", ls="--", lw=1)
    ax[0].set(xscale="log", ylim=(0, 1), xlabel="period (s)", ylabel="residual coherence",
              title="the residuals after the H-explained part (the model needs >= %.1f)" % RESID_COH_MIN)
    ax[1].axhspan(GAIN_LO, GAIN_HI, color="C2", alpha=0.15, lw=0)
    ax[1].set(xscale="log", yscale="log", xlabel="period (s)", ylabel="|g| = |S_ry,rx / S_rx,rx|",
              title="the complex gain (the model needs %.2f-%.2f)" % (GAIN_LO, GAIN_HI))
    for a_ in ax:
        a_.grid(alpha=GRID_ALPHA, which="both")
    ax[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("%s: the shared-centre residual test on %s, with the built control beside it"
                 % (site, day[0]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, tight=False)


def diagonal_day(sv, site, out, elines=None, rate=1, figsize=(13, 6)):
    """One quiet day of Ex and Ey with the north-west diagonal Ex' = (Ex - Ey)/sqrt2 underneath."""
    from ..process import frame as FR
    from ..site.masks import load_raw
    fig, ax = _fig(2, 1, figsize=figsize, sharex=True)
    t0, arr = load_raw(sv, site, rate, ("Ex", "Ey"))
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
                i0, i1, label = a0, a1, "%s, Ex-Ey coherence %.2f" % (r.day, float(r.coh_Ex_Ey))
                break
    ex = np.asarray(arr["Ex"][i0:i1], float) * s_ex
    ey = np.asarray(arr["Ey"][i0:i1], float) * s_ey
    hrs = np.arange(len(ex)) / (3600.0 * rate)
    ax[0].plot(hrs, ex - np.nanmean(ex), color="C3", lw=0.5, label="Ex, signed")
    ax[0].plot(hrs, ey - np.nanmean(ey), color="C4", lw=0.5, label="Ey, signed")
    ax[0].set(ylabel="E (mV/km)")
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    d = (ex - ey) / np.sqrt(2.0)
    s = (ex + ey) / np.sqrt(2.0)
    ax[1].plot(hrs, d - np.nanmean(d), color="C0", lw=0.5,
               label="Ex' = (Ex - Ey)/sqrt2, the north-west diagonal")
    ax[1].plot(hrs, s - np.nanmean(s), color="0.6", lw=0.5,
               label="Ey' = (Ex + Ey)/sqrt2, the centre doubled")
    ax[1].set(ylabel="E' (mV/km)", xlabel="hours from the day's start")
    ax[1].legend(fontsize=8, loc="upper right", ncol=2)
    for a in ax:
        a.grid(alpha=GRID_ALPHA)
    ax[0].set_title("%s: %s -- the two lines above, the two diagonals below; the difference is what the "
                    "shared centre cancels" % (site, label))
    return _save(fig, out)


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
        return _save(fig, out)
    z0 = np.load(src, allow_pickle=False)
    z1 = np.load(var, allow_pickle=False)
    t0 = float(np.asarray(z0["t0"]).ravel()[0])
    for r, ch in enumerate(chans):
        rows = decision[decision.channel == ch]
        worst = str(rows.sort_values("worst_ratio").iloc[-1].worst_day)
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
        for c, (y, lab, col) in enumerate(((x0[i0:i1], "before", "C3"), (x1[i0:i1], "after", "C0"))):
            g = y[np.isfinite(y)]
            if g.size < 2 * nperseg:
                continue
            f, P = signal.welch(g - g.mean(), fs=float(rate), nperseg=nperseg, noverlap=nperseg // 2,
                                detrend="linear")
            for k, f0 in enumerate(freqs):
                m = (f > f0 * 0.8) & (f < f0 * 1.2)
                ax[r][k].semilogy(f[m], P[m], color=col, lw=0.9, label=lab)
        for k, f0 in enumerate(freqs):
            fired = bool(rows[rows.f0 == f0].fire.any())
            ax[r][k].axvline(f0, color="0.4", ls=":", lw=1)
            ax[r][k].set(xlabel="frequency (Hz)" if r == len(chans) - 1 else "",
                         ylabel="%s power" % ch if k == 0 else "",
                         title="%s at %.3f Hz, worst day %s%s" % (ch, f0, worst,
                                                                  "" if fired else " (did not fire)"))
            ax[r][k].grid(alpha=GRID_ALPHA, which="both")
            if r == 0 and k == 0:
                ax[r][k].legend(fontsize=8)
    z0.close()
    z1.close()
    fig.suptitle("%s: the tone before and after the notch, each channel on its own worst day" % site,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, tight=False)


def ratio_panel(ratio, site, out, tone_bands=(), figsize=(13, 5)):
    """The per-period ratio of each variant's product to the original's, by band."""
    fig, ax = _fig(1, 1, figsize=figsize)
    if ratio is None or not len(ratio):
        ax.text(0.5, 0.5, "no 10 Hz pair to compare", ha="center")
        return _save(fig, out)
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
    ax.set(xticks=x, yscale="log", ylabel="variant rho / original rho",
           title="%s: what each 10 Hz variant did to the product, band by band -- the shaded columns hold "
                 "the tone and its harmonic, the green band is the 2.5 per cent the rest must stay inside"
                 % site)
    ax.set_xticklabels(bands, rotation=0, fontsize=8)
    ax.grid(alpha=GRID_ALPHA, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    return _save(fig, out)


def spike_map(side, site, out, figsize=(13, 6)):
    """The spike screen's blanked count per channel and per UTC day."""
    import pandas as pd
    fig, ax = _fig(2, 1, figsize=figsize)
    per_channel = pd.DataFrame(side.get("per_channel") or [])
    per_day = pd.DataFrame(side.get("per_day") or [])
    if not len(per_channel):
        ax[0].text(0.5, 0.5, "no spike variant", ha="center")
        return _save(fig, out)
    ax[0].bar(per_channel.channel, 100 * per_channel.fraction.to_numpy(float), color="C0")
    ax[0].set(ylabel="per cent of samples blanked")
    ax[0].grid(alpha=GRID_ALPHA)
    if len(per_day):
        days = list(dict.fromkeys(per_day.day))
        x = np.arange(len(days))
        for k, ch in enumerate(per_channel.channel):
            g = per_day[per_day.channel == ch].set_index("day").reindex(days)
            ax[1].plot(x, g.blanked.to_numpy(float), lw=0.8, color="C%d" % k, label=ch)
        step = max(1, len(days) // 20)
        ax[1].set_xticks(x[::step])
        ax[1].set_xticklabels([d[5:] for d in days][::step], rotation=90, fontsize=7)
        ax[1].set(ylabel="samples blanked", xlabel="UTC day", yscale="symlog")
        ax[1].grid(alpha=GRID_ALPHA)
        ax[1].legend(fontsize=8, ncol=5)
    c = side.get("control", {})
    ax[0].set_title("%s: the spike screen at k = %g -- the control is the quietest day %s (%s blanked) "
                    "against the noisiest %s (%s)"
                    % (site, side.get("k", np.nan), c.get("quietest_day"), c.get("quietest_blanked"),
                       c.get("noisiest_day"), c.get("noisiest_blanked")))
    return _save(fig, out)


# ---------------------------------------------------------------- section 8

def candidate_bars(cands, site, out, threshold=0.5, donor_gate=0.8, figsize=(13, 5)):
    """Horizontal bars of the site's own coherence with each candidate, per channel, with the thresholds."""
    fig, ax = _fig(1, 2, figsize=figsize, sharey=True)
    if cands is None or not len(cands):
        ax[0].text(0.5, 0.5, "no candidate", ha="center")
        return _save(fig, out)
    y = np.arange(len(cands))[::-1]
    for k, chan in enumerate(("Hx", "Hy")):
        a = ax[k]
        a.barh(y + 0.2, cands["own_%s_coh" % chan].to_numpy(float), height=0.38, color="C0",
               label="the site's own %s" % chan)
        a.barh(y - 0.2, cands["donor_%s_coh" % chan].to_numpy(float), height=0.38, color="0.65",
               hatch="//", label="the candidate against its third site")
        a.axvline(threshold, color="C3", ls=":", lw=1)
        a.axvline(donor_gate, color="C2", ls="--", lw=1)
        a.set(xlim=(0, 1), xlabel="%s coherence, 100-1000 s" % chan)
        a.grid(alpha=GRID_ALPHA, axis="x")
        if k == 0:
            a.set_yticks(y)
            a.set_yticklabels(["%s (%.0f km)" % (r.candidate, r.km) for r in cands.itertuples()],
                              fontsize=8)
            a.legend(fontsize=8, loc="lower right")
    fig.suptitle("%s: the replacement candidates -- the dotted line is the %.2f the own channel must fall "
                 "below, the dashed line the %.2f the candidate must reach against a third site"
                 % (site, threshold, donor_gate), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out, tight=False)


# ---------------------------------------------------------------- section 10

def forms_bars(table, site, out, band=(10, 1000), margin=0.2, figsize=(13, 8)):
    """One horizontal bar per form on the 10-1000 s bar, its controls beside it, candidates marked."""
    fig, ax = _fig(1, 1, figsize=figsize)
    d = table[np.isfinite(table.bar_10_1000)] if table is not None and len(table) else None
    if d is None or not len(d):
        ax.text(0.5, 0.5, "no form carries a bar", ha="center")
        return _save(fig, out)
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
    ax.set(xscale="log", xlabel="the %g-%g s bar (median relative impedance error)" % band,
           yticks=y, title="%s: every form on the bar, its control as a red diamond; a form is a candidate "
                           "only where it beats every control by %.0f per cent" % (site, 100 * margin))
    ax.set_yticklabels(d.form, fontsize=8)
    ax.grid(alpha=GRID_ALPHA, axis="x", which="both")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], fontsize=8, loc="lower right")
    return _save(fig, out)
