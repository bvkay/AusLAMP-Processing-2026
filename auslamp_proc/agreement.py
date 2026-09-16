"""Two transfer functions on one period grid, and what their difference per decade says.

The grid for a comparison between two of our own products is the ten-per-decade grid T = 10^(k/10) from
3.16 s (k = 5) to 50,119 s (k = 47), ported from scripts/qc/edi_resample.py:69-71
(D:/BEN/MTH5_Aurora_mt-io_2026). A comparison against a source outside the run is scored on OUR periods, so
nothing of ours is moved.

on_grid interpolates linearly in log10 period, of log10 |Z| and of the unwrapped phase, over the source's own
valid nodes. There is no extrapolation and no bridging of a hole wider than 0.30 decades, both from the same
module: a grid period outside the source's first and last valid node of that component, or between two nodes
further apart than that, gets no value.

per_decade reports, per band and per off-diagonal component, the median apparent-resistivity ratio
(|Z_a|^2 / |Z_b|^2) and the median phase difference in degrees, with the count the medians were taken over.
The bands are 5-10, 10-100, 100-1000 and 1000-10000 s.

reading is the scale / frame / fault call of scripts/processing/vic_vs_ga.py:51-99, with its thresholds as
arguments: two curves agree within AGREE_RHO and AGREE_PHASE_DEG; a difference is a SCALE where the ratio's
75th over its 25th percentile is at most SPREAD_MAX and the phase agrees, which is a dipole length or a gain;
a FRAME where turning by the declination brings the ratio inside TURNED_LO..TURNED_HI with the phase
agreeing; anything else is a FAULT. It is a reading and never a check.

smoothness is the per-curve score of scripts/qc/curve_consistency.py: each point against a weighted quadratic
through its neighbours, the residual over the combined sigma, a jump above JUMP_Z; the phase step per decade;
and the Weidelt rho-phase residual, which is read together with the jumps and never alone.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .products import COMPONENTS, OFF_DIAGONAL, TFData, rho_phase

# the ten-per-decade grid, 3.16 s to 50,119 s
K_LO, K_HI, PER_DECADE = 5, 47, 10
GRID = 10.0 ** (np.arange(K_LO, K_HI + 1) / float(PER_DECADE))

GAP_DEX_MAX = 0.30            # do not bridge a hole in the source wider than this, in decades
BANDS = ((5.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, 10000.0))
AGREE_BAND = (5.0, 200.0)     # the band the agreement rule is read over

AGREE_RHO = 0.20              # agreement: the rho ratio within this fraction of one
AGREE_PHASE_DEG = 5.0         # ... and the phase difference within this many degrees
SPREAD_MAX = 1.3              # a scale: the ratio's 75th over its 25th percentile at most this
TURNED_LO, TURNED_HI = 0.8, 1.25   # a frame: the turned ratio inside this range

JUMP_Z = 4.0                  # |z| above this is a jump
SMOOTH_HALF = 3               # neighbours each side the quadratic is fitted to


def band_label(lo, hi) -> str:
    return "%g-%g s" % (lo, hi)


# ------------------------------------------------------------------ one curve onto another's periods

def _interp_component(p_src, z_src, e_src, periods):
    """(Z, error) of one component on `periods`, linear in log period of log|Z| and the unwrapped phase."""
    out = np.full(len(periods), np.nan + 1j * np.nan, complex)
    oute = np.full(len(periods), np.nan)
    ok = np.isfinite(p_src) & (p_src > 0) & np.isfinite(z_src) & (np.abs(z_src) > 0)
    if ok.sum() < 2:
        return out, oute
    x = np.log10(p_src[ok])
    mag = np.log10(np.abs(z_src[ok]))
    ang = np.unwrap(np.angle(z_src[ok]))
    err = np.asarray(e_src, float)[ok]
    lg_err = np.log10(np.where(np.isfinite(err) & (err > 0), err, np.nan))
    xq = np.log10(np.asarray(periods, float))
    inside = (xq >= x[0]) & (xq <= x[-1])                      # no extrapolation
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    gap = x[idx + 1] - x[idx]
    inside &= gap <= GAP_DEX_MAX                               # no bridging of a hole in the source
    m = np.interp(xq, x, mag)
    a = np.interp(xq, x, ang)
    out[inside] = (10.0 ** m[inside]) * np.exp(1j * a[inside])
    if np.isfinite(lg_err).any():
        good = np.isfinite(lg_err)
        e = np.interp(xq, x[good], lg_err[good]) if good.sum() >= 2 else np.full(len(xq), np.nan)
        oute[inside] = 10.0 ** e[inside]
    return out, oute


def on_grid(a: TFData, b: TFData, periods=None) -> TFData:
    """`b` interpolated onto `a`'s periods (or onto `periods`), component by component.

    Nothing of `a` is moved. Pass periods=GRID for a run-against-run comparison, where neither curve is the
    one the other is judged against.
    """
    per = np.asarray(a.period, float) if periods is None else np.asarray(periods, float)
    z = np.full((len(per), 2, 2), np.nan + 1j * np.nan, complex)
    ze = np.full((len(per), 2, 2), np.nan)
    for c, (i, j) in COMPONENTS.items():
        z[:, i, j], ze[:, i, j] = _interp_component(b.period, b.z[:, i, j], b.z_err[:, i, j], per)
    t = te = None
    if b.t is not None:
        shape = np.asarray(b.t).shape
        t = np.full((len(per),) + shape[1:], np.nan + 1j * np.nan, complex)
        te = np.full((len(per),) + shape[1:], np.nan)
        for i in range(shape[1]):
            for j in range(shape[2]):
                # real and imaginary parts, linearly in log period: a tipper part crosses zero and changes
                # sign, which log10 of a magnitude cannot carry
                t[:, i, j], te[:, i, j] = _interp_linear(b.period, b.t[:, i, j],
                                                         np.asarray(b.t_err)[:, i, j], per)
    return TFData(per, z, ze, t, te, dict(b.meta, on_grid_of=("GRID" if periods is not None else "a")))


def _interp_linear(p_src, v_src, e_src, periods):
    """(value, error) of one complex quantity on `periods`, real and imaginary parts linear in log period."""
    out = np.full(len(periods), np.nan + 1j * np.nan, complex)
    oute = np.full(len(periods), np.nan)
    ok = np.isfinite(p_src) & (p_src > 0) & np.isfinite(v_src)
    if ok.sum() < 2:
        return out, oute
    x = np.log10(np.asarray(p_src, float)[ok])
    xq = np.log10(np.asarray(periods, float))
    inside = (xq >= x[0]) & (xq <= x[-1])
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    inside &= (x[idx + 1] - x[idx]) <= GAP_DEX_MAX
    re_ = np.interp(xq, x, np.real(v_src[ok]))
    im_ = np.interp(xq, x, np.imag(v_src[ok]))
    out[inside] = re_[inside] + 1j * im_[inside]
    e = np.asarray(e_src, float)[ok]
    if np.isfinite(e).sum() >= 2:
        g = np.isfinite(e)
        oute[inside] = np.interp(xq, x[g], e[g])[inside]
    return out, oute


# ------------------------------------------------------------------ the per-decade numbers

def _pair(a: TFData, b_on_a: TFData, comp: str, lo: float, hi: float):
    """(ratio array, phase-difference array) of one component over one band, a over b."""
    i, j = COMPONENTS[comp]
    p = np.asarray(a.period, float)
    za, zb = a.z[:, i, j], b_on_a.z[:, i, j]
    m = (p >= lo) & (p < hi) & np.isfinite(za) & np.isfinite(zb) & (np.abs(zb) > 0)
    if not m.any():
        return np.zeros(0), np.zeros(0)
    ratio = (np.abs(za[m]) ** 2) / (np.abs(zb[m]) ** 2)
    dphi = (np.degrees(np.angle(za[m])) - np.degrees(np.angle(zb[m])) + 180.0) % 360.0 - 180.0
    return ratio, dphi


def per_decade(a: TFData, b: TFData, bands=BANDS, components=OFF_DIAGONAL, periods=None) -> pd.DataFrame:
    """Rows of (band, component, median rho ratio, median phase difference in deg, n), a over b."""
    b_on_a = on_grid(a, b, periods=periods)
    ref = a if periods is None else on_grid(a, a, periods=periods)
    rows = []
    for lo, hi in bands:
        for comp in components:
            ratio, dphi = _pair(ref, b_on_a, comp, lo, hi)
            rows.append(dict(band=band_label(lo, hi), component=comp,
                             rho_ratio=(float(np.median(ratio)) if len(ratio) else np.nan),
                             phase_diff_deg=(float(np.median(dphi)) if len(dphi) else np.nan),
                             n=int(len(ratio))))
    return pd.DataFrame(rows)


def agrees(row, agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE_DEG) -> bool:
    """True where one per-decade row is inside the agreement rule."""
    r, d = row.get("rho_ratio"), row.get("phase_diff_deg")
    if not (np.isfinite(r) and np.isfinite(d)) or not row.get("n"):
        return False
    return bool(abs(r - 1.0) <= agree_rho and abs(d) <= agree_phase)


def band_stats(a: TFData, b: TFData, comp: str, lo=AGREE_BAND[0], hi=AGREE_BAND[1], periods=None) -> dict:
    """The median ratio, its spread, the median phase difference and the count over one band."""
    b_on_a = on_grid(a, b, periods=periods)
    ref = a if periods is None else on_grid(a, a, periods=periods)
    ratio, dphi = _pair(ref, b_on_a, comp, lo, hi)
    if len(ratio) < 4:
        return dict(rho_ratio=np.nan, spread=np.nan, phase_diff_deg=np.nan, n=int(len(ratio)))
    return dict(rho_ratio=float(np.median(ratio)),
                spread=float(np.percentile(ratio, 75) / max(np.percentile(ratio, 25), 1e-30)),
                phase_diff_deg=float(np.median(dphi)), n=int(len(ratio)))


# ------------------------------------------------------------------ the comparisons a workbook draws

def kind_vs_kind(products: pd.DataFrame, read=None, bands=BANDS, agree_rho=AGREE_RHO,
                 agree_phase=AGREE_PHASE_DEG, agree_band=AGREE_BAND) -> pd.DataFrame:
    """Every pair of kinds of one site and rate, per band, with the agreement call over `agree_band`.

    `read` is a function from a path to a TFData; the default reads each file once per call.
    """
    from .products import read_tf
    read = read or read_tf
    rows = []
    for (site, rate, run), grp in products.groupby(["site", "rate_hz", "run"], sort=True):
        avail = {r.kind: r.path for r in grp.itertuples() if r.on_disk}
        keys = list(avail)
        for m in range(len(keys)):
            for n in range(m + 1, len(keys)):
                ka, kb = keys[m], keys[n]
                try:
                    a, b = read(avail[ka]), read(avail[kb])
                except Exception as exc:
                    rows.append(dict(site=site, run=run, rate_hz=rate, kind_a=ka, kind_b=kb,
                                     band="", component="", rho_ratio=np.nan, phase_diff_deg=np.nan, n=0,
                                     agrees=False, error="%s: %s" % (type(exc).__name__, str(exc)[:80])))
                    continue
                per = per_decade(a, b, bands=bands)
                for _, r in per.iterrows():
                    rows.append(dict(site=site, run=run, rate_hz=rate, kind_a=ka, kind_b=kb,
                                     band=r.band, component=r.component, rho_ratio=r.rho_ratio,
                                     phase_diff_deg=r.phase_diff_deg, n=int(r.n), agrees=False, error=""))
                for comp in OFF_DIAGONAL:
                    s = band_stats(a, b, comp, agree_band[0], agree_band[1])
                    rows.append(dict(site=site, run=run, rate_hz=rate, kind_a=ka, kind_b=kb,
                                     band=band_label(*agree_band), component=comp,
                                     rho_ratio=s["rho_ratio"], phase_diff_deg=s["phase_diff_deg"],
                                     n=s["n"], agrees=agrees(s, agree_rho, agree_phase), error=""))
    return pd.DataFrame(rows)


def run_vs_run(products: pd.DataFrame, read=None, bands=BANDS) -> pd.DataFrame:
    """The same product (site, kind, rate) across two runs, on the ten-per-decade grid."""
    from .products import read_tf
    read = read or read_tf
    rows = []
    for (site, kind, rate), grp in products.groupby(["site", "kind", "rate_hz"], sort=True):
        runs = [(r.run, r.stamp, r.path) for r in grp.itertuples() if r.on_disk]
        if len(runs) < 2:
            continue
        for m in range(len(runs)):
            for n in range(m + 1, len(runs)):
                (ra, sa, pa), (rb, sb, pb) = runs[m], runs[n]
                a, b = read(pa), read(pb)
                per = per_decade(a, b, bands=bands, periods=GRID)
                for _, r in per.iterrows():
                    rows.append(dict(site=site, kind=kind, rate_hz=rate, run_a="%s_%s" % (ra, sa),
                                     run_b="%s_%s" % (rb, sb), band=r.band, component=r.component,
                                     rho_ratio=r.rho_ratio, phase_diff_deg=r.phase_diff_deg, n=int(r.n)))
    return pd.DataFrame(rows)


def versus_comparison(products: pd.DataFrame, source: dict, declinations: dict, read=None, bands=BANDS,
                      agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE_DEG,
                      agree_band=AGREE_BAND) -> pd.DataFrame:
    """Ours against one declared source, per site, kind and band, with the scale / frame / fault reading.

    `declinations` is {site: declination_deg} from sites.csv; it is what turns a geographic source into our
    frame and what the frame reading tries as an explanation for a disagreement.
    """
    from .products import load_comparison, read_tf, turn_to_our_frame
    read = read or read_tf
    rows = []
    for site, grp in products.groupby("site", sort=True):
        dec = declinations.get(site)
        try:
            theirs = load_comparison(source, site, dec)
        except Exception as exc:
            rows.append(dict(site=site, kind="", rate_hz=np.nan, band="", component="", rho_ratio=np.nan,
                             phase_diff_deg=np.nan, n=0, spread=np.nan, rho_ratio_turned=np.nan,
                             phase_diff_turned_deg=np.nan, reading="",
                             error="%s: %s" % (type(exc).__name__, str(exc)[:100]), their_file=""))
            continue
        if not theirs:
            continue
        for r in grp.itertuples():
            if not r.on_disk:
                continue
            tf_them = theirs.get(r.kind, theirs.get(""))
            if tf_them is None:
                continue
            ours = read(r.path)
            for lo, hi in bands:
                for comp in OFF_DIAGONAL:
                    s = band_stats(ours, tf_them, comp, lo, hi)
                    rows.append(dict(site=site, kind=r.kind, rate_hz=r.rate_hz, band=band_label(lo, hi),
                                     component=comp, rho_ratio=s["rho_ratio"], spread=s["spread"],
                                     phase_diff_deg=s["phase_diff_deg"], n=s["n"],
                                     rho_ratio_turned=np.nan, phase_diff_turned_deg=np.nan,
                                     reading="", error="", their_file=str(tf_them.meta.get("path", ""))))
            for comp in OFF_DIAGONAL:
                s = band_stats(ours, tf_them, comp, agree_band[0], agree_band[1])
                turned = dict(rho_ratio=np.nan, phase_diff_deg=np.nan, spread=np.nan, n=0)
                if dec is not None and np.isfinite(float(dec)):
                    turned = band_stats(ours, turn_to_our_frame(tf_them, float(dec)), comp,
                                        agree_band[0], agree_band[1])
                rows.append(dict(site=site, kind=r.kind, rate_hz=r.rate_hz,
                                 band=band_label(*agree_band), component=comp,
                                 rho_ratio=s["rho_ratio"], spread=s["spread"],
                                 phase_diff_deg=s["phase_diff_deg"], n=s["n"],
                                 rho_ratio_turned=turned["rho_ratio"],
                                 phase_diff_turned_deg=turned["phase_diff_deg"],
                                 reading=reading(s, turned, agree_rho, agree_phase), error="",
                                 their_file=str(tf_them.meta.get("path", ""))))
    return pd.DataFrame(rows)


def reading(now: dict, turned: dict, agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE_DEG,
            spread_max=SPREAD_MAX, turned_lo=TURNED_LO, turned_hi=TURNED_HI) -> str:
    """scale / frame / fault, or agreement. A reading of what a difference looks like, never a check.

    A constant rho ratio with the phase untouched is a scale -- a dipole length or a gain. A disagreement the
    declination turn removes is a frame. The rest is a fault: the ratio wanders with period, or the phase
    disagrees and the turn does not fix it.
    """
    r, sp, d = now.get("rho_ratio"), now.get("spread"), now.get("phase_diff_deg")
    if not (np.isfinite(r) and np.isfinite(d)):
        return "not scored"
    if abs(r - 1.0) <= agree_rho and abs(d) <= agree_phase:
        return "agrees"
    rt, dt = turned.get("rho_ratio"), turned.get("phase_diff_deg")
    if np.isfinite(rt) and np.isfinite(dt) and turned_lo <= rt <= turned_hi and abs(dt) <= agree_phase:
        return "FRAME (turned %.2f, %+.1f deg)" % (rt, dt)
    if np.isfinite(sp) and sp <= spread_max and abs(d) <= agree_phase:
        return "SCALE x%.2f in rho (a dipole or a gain)" % r
    return "FAULT (ratio %.2f, spread %.2f, %+.1f deg)" % (r, sp if np.isfinite(sp) else np.nan, d)


# ------------------------------------------------------------------ smoothness, per curve

def _local_outlier_z(x, y, s, half=SMOOTH_HALF):
    """Each point's residual from a weighted quadratic through its neighbours, over the combined sigma.

    The point itself is left out of the fit, so curvature is absorbed by the quadratic and a single-period
    spike or a step stands out. NaN where fewer than four neighbours have finite bars.
    """
    n = len(x)
    z = np.full(n, np.nan)
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    for k in range(n):
        idx = [i for i in range(max(0, k - half), min(n, k + half + 1))
               if i != k and np.isfinite(s[i]) and s[i] > 0 and np.isfinite(y[i])]
        if len(idx) < 4 or not (np.isfinite(s[k]) and s[k] > 0 and np.isfinite(y[k])):
            continue
        xx = np.vstack([np.ones(len(idx)), x[idx] - x[k], (x[idx] - x[k]) ** 2]).T
        w = 1.0 / s[idx]
        a = xx * w[:, None]
        b = y[idx] * w
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        try:
            s_fit = np.sqrt(max(np.linalg.inv(a.T @ a)[0, 0], 0.0))
        except np.linalg.LinAlgError:
            s_fit = 0.0
        z[k] = (y[k] - coef[0]) / np.sqrt(s[k] ** 2 + s_fit ** 2)
    return z


def smoothness(tf: TFData, components=OFF_DIAGONAL, jump_z=JUMP_Z, period_range=None) -> pd.DataFrame:
    """One row per component: the jumps per decade, the largest phase step per decade, the rho-phase residual.

    A jump is a point off the local smooth trend by more than `jump_z` of its own bar. The phase step is the
    adjacent difference taken the short way round the circle, so a curve crossing +-180 deg is not read as a
    360 deg step. The rho-phase residual is the Weidelt local 1-D relation
    phi_pred = 45 deg x (1 - d log rho / d log T) against the measured phase; it runs 10-15 deg at a 2-D or
    3-D site, so it is read together with the jumps and never alone.
    """
    rows = []
    for comp in components:
        rho, rho_err, ph, ph_err = rho_phase(tf.period, tf.z, tf.z_err, comp)
        ok = np.isfinite(rho) & (rho > 0) & np.isfinite(ph)
        if period_range is not None:
            pp = np.asarray(tf.period, float)
            ok &= (pp >= period_range[0]) & (pp <= period_range[1])
        p = np.asarray(tf.period, float)[ok]
        if len(p) < 5:
            rows.append(dict(component=comp, n_periods=int(len(p)), n_jumps=0, jumps_per_decade=np.nan,
                             worst_z=np.nan, worst_z_period_s=np.nan, max_phase_step_deg=np.nan,
                             max_phase_step_deg_per_decade=np.nan,
                             median_abs_dphi_deg=np.nan, jumps=""))
            continue
        lr = np.log10(rho[ok])
        lt = np.log10(p)
        s_lr = (rho_err[ok] / rho[ok]) / np.log(10)          # sigma of log10 rho from the rho bar
        s_ph = ph_err[ok]
        z_lr = _local_outlier_z(lt, lr, s_lr)
        z_ph = _local_outlier_z(lt, ph[ok], s_ph)
        worst = np.fmax(np.abs(z_lr), np.abs(z_ph))
        jumps = [(float(p[k]), float(worst[k])) for k in range(len(p))
                 if np.isfinite(worst[k]) and worst[k] > jump_z]
        decades = max(lt[-1] - lt[0], 1e-9)
        # the difference taken the short way round the circle: a curve crossing +-180 deg steps by a few
        # degrees, not by 360
        dph = (np.diff(ph[ok]) + 180.0) % 360.0 - 180.0
        # the rate is floored at 0.05 decades of spacing: two periods 0.005 decades apart turn a 20 deg step
        # into 4,000 deg a decade, which is a property of the period grid and not of the curve
        step = np.abs(dph) / np.maximum(np.diff(lt), 0.05)
        slope = np.gradient(lr, lt)
        dphi = ph[ok] - 45.0 * (1.0 - slope)
        rows.append(dict(component=comp, n_periods=int(len(p)), n_jumps=len(jumps),
                         jumps_per_decade=round(len(jumps) / decades, 3),
                         worst_z=(float(np.nanmax(worst)) if np.isfinite(worst).any() else np.nan),
                         worst_z_period_s=(float(p[int(np.nanargmax(worst))])
                                           if np.isfinite(worst).any() else np.nan),
                         max_phase_step_deg=(float(np.max(np.abs(dph))) if len(dph) else np.nan),
                         max_phase_step_deg_per_decade=(float(np.max(step)) if len(step) else np.nan),
                         median_abs_dphi_deg=float(np.median(np.abs(dphi))),
                         jumps="; ".join("%.0f s z %.1f" % j for j in jumps[:6])))
    return pd.DataFrame(rows)


def rate_step(a: TFData, b: TFData, below=(8.0, 16.0), above=(32.0, 100.0),
              components=OFF_DIAGONAL) -> pd.DataFrame:
    """The level step between two rates of one site and kind, on the band each side of the 16 s join.

    `a` is the 1 Hz curve and `b` the 10 Hz curve. The step is reported as a per cent difference in apparent
    resistivity, which is what the splice rule of the later workbook is read against.
    """
    rows = []
    for comp in components:
        lo = band_stats(a, b, comp, below[0], below[1])
        hi = band_stats(a, b, comp, above[0], above[1])
        rows.append(dict(component=comp,
                         below_band=band_label(*below), below_pct=100.0 * (lo["rho_ratio"] - 1.0),
                         below_phase_deg=lo["phase_diff_deg"], n_below=lo["n"],
                         above_band=band_label(*above), above_pct=100.0 * (hi["rho_ratio"] - 1.0),
                         above_phase_deg=hi["phase_diff_deg"], n_above=hi["n"]))
    return pd.DataFrame(rows)
