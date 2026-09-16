"""Is the record usable, when, and which hours or days a component is estimated over.

Six tests and three selections, each ported with its criterion from the frozen Victoria and WA tools under
D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing.

    daily_magnetics_test   per UTC day the total field against IGRF within 5 per cent AND the high-passed
                           Hx and Hz standard deviations within a factor 3 of the median of the k nearest
                           sites recording that day (vic_windows.cmd_days :150-168)
    fleet_test             over one stretch the 100-1000 s Hx-Hx and Hy-Hy coherence of the site against the
                           sites within near_km, read against the control pairs at the same distances at 0.8
                           of their median; a site incoherent with every other site under 0.3 is named and
                           dropped from the control; a negative-control stretch must read under 0.3
                           (vic_windows.cmd_fleet :171-209)
    clock_test             per day the lag of the peak normalised cross-correlation of the despiked,
                           high-passed Hx against a reference site to +-12 h, counted only where F is within
                           20 per cent of IGRF and the peak correlation is at least 0.5; within 10 s passes
                           (vic_windows.cmd_clock :247-270)
    quality_map            the hourly 4-50 s coherence of each line with the H it couples to, the daily
                           50-1000 s multiple coherence with the local pair and with the observatory pair,
                           and one 1000-10000 s number on the record decimated to 0.1 Hz
                           (vic_quality.cmd_map :116-170, multi_coh :37-78, pair_coh :81-101)
    day_mask               whole days kept on the observatory multiple coherence at or above thr, with a
                           random control of the same day count from a named seed (vic_quality.cmd_mask
                           :198-229)
    component_mask         the running median over smooth_h hours of the component's own coherence below
                           coh_min masks the hour; a masked run shorter than min_hole_h is given back
                           (Processing_Run/wamt_tools.sustained_low :518-543)
    best_hours             the best fraction of candidate hour windows by the band coherence of the
                           component's pair, with the random control of the same size from the same pool and
                           the matched-duration contiguous controls (vic_best_hours :41-95,
                           qld_student.select_hours :918-933, vic_short_w2m.cmd_masks :313-360)
    window_from_days       the longest run of sound days of one line as a proposed window
                           (vic_add_windows :30-83)

The multiple coherence is bias-corrected as g2c = (g2 - p/nu) / (1 - p/nu) with p = 2 predictors and
nu = 0.82 x the number of 50 per cent overlapping Hann segments, because a few-segment estimate saturates at
1 or sits at the bias floor; the magnetic cross-spectral matrix is inverted by pseudo-inverse where a flat
channel makes it singular.

Selecting on the target's own E-H coherence uses the target's own response and favours the hours where the
linear model already fits, so the thresholds sit well below live (0.5 where a live line reads above 0.85) and
the component mask requires a sustained run.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import geo
from ..look import despike, highpass
from ..raw import cache

DAY = 86400.0

# vic_windows.py:57-59, :125-128
HIGHPASS_S = 3000.0
FLEET_BAND_S = (100.0, 1000.0)
FLEET_NPERSEG = 8192
F_TOLERANCE = 0.05                  # a day's total field within 5 per cent of IGRF
FLUCTUATION_FACTOR = 3.0            # ... and Hx, Hz within a factor 3 of the neighbours' median
FLEET_NEAR_KM = 150.0
FLEET_CONTROL_FRACTION = 0.8        # the site's median at or above this share of the control pairs' median
FLEET_JUNK = 0.3                    # a site incoherent with every other site below this is not a control
NEGATIVE_MAX = 0.3                  # the negative control must read under this
SHIFT_CONTROL_S = 12 * 3600.0       # the neighbour's record is taken this much later for the shifted pair

# vic_windows.py:247-270
CLOCK_MAXLAG_S = 12 * 3600
CLOCK_MIN_CORR = 0.5
CLOCK_F_TOLERANCE = 0.20
CLOCK_PASS_S = 10.0
CLOCK_BAND_S = (5.0, 20.0)          # the band the lag is measured in, process.align.SHORT_BAND
CLOCK_EDGE_FRACTION = 0.95          # a peak this far out in the search window is an edge hit, not a lag
CLOCK_PEAK_RATIO = 1.5              # ... and a peak must stand this far above the day's own other lags
CLOCK_MIN_DAYS = 5                  # below this many counted days the clock limb is UNJUDGED

# vic_quality.py:37-101, :144-155
MULTI_P = 2.0                       # predictors: Hx and Hy
NU_FACTOR = 0.82                    # effective independent segments per 50 per cent overlapping Hann window
HOURLY_BAND_S = (4.0, 50.0)
HOURLY_NPERSEG = 512
DAILY_BAND_S = (50.0, 1000.0)
DAILY_NPERSEG = 4096
VLONG_BAND_S = (1000.0, 10000.0)
VLONG_DECIMATE = 10

# vic_quality.py:198-229
DAY_THR = 0.5
DAY_MIN_KEPT = 3                    # a mask keeping fewer days than this starves the long bands

# Processing_Run/make_notebooks.py:815
COH_BAND_S = (20.0, 200.0)
COH_MIN = 0.5
SMOOTH_H = 6.0
MIN_HOLE_H = 6.0
WIN_S = 3600.0
STEP_S = 1800.0
COH_NPERSEG = 1024

# vic_best_hours.py:41-44
HOURS_BAND_S = (2.0, 50.0)
HOURS_FRACTION = 0.25
HOURS_NPERSEG = 256
HOURS_MIN_CANDIDATES = 24           # below this the weak days are admitted and the selection is flagged thin
CONTIG_HOURS = (2, 4, 6, 24)
# A random split of a scored pool separates kept from dropped by a small amount half the time, so "the
# random control does not separate" is read as a fraction of the ranked selection's own separation: the
# control separates where its kept-minus-dropped exceeds this share of the selection's.
RANDOM_GAP_MAX = 0.2

COMPONENT_PAIR = {"xy": ("Ex", "Hy"), "yx": ("Ey", "Hx")}
COMPONENT_LINE = {"xy": "Ex", "yx": "Ey"}
H = ("Hx", "Hy")


# ---------------------------------------------------------------- reading a record

def load_raw(sv, site, rate=1, channels=("Hx", "Hy", "Hz", "Ex", "Ey")):
    """(t0, {channel: array}) of one site's cache as laid: no sign, no rotation.

    The tests below read the record the instrument wrote. A sign flips a coherence's phase and not its
    magnitude, and the frame turn mixes Hx into Hy, so a fleet coherence measured on rotated pairs is a
    different number at every site; these tests are measured as laid.
    """
    t0, arrays, _meta = cache.load(site, sv.cfg["work_root"], rate)
    return int(t0), {c: arrays[c] for c in channels if c in arrays}


def span(sv, site, rate=1):
    """(t0, n) of one site's cache. Ported from process.references.Store.window (:187-193)."""
    from pathlib import Path
    z = np.load(Path(sv.cfg["work_root"]) / ("cache_%dhz" % int(rate)) / ("%s.npz" % site),
                allow_pickle=False)
    t0, n = int(z["t0"][0]), int(len(z["Hx"]))
    z.close()
    return t0, n


def site_position(sv, site):
    r = sv.site(site)
    return float(r.lat), float(r.lon)


def distance_km(sv, a, b) -> float:
    return geo.distance_km(site_position(sv, a), site_position(sv, b))


def prepare(x, b, a, min_finite=0.8):
    """One stretch despiked, interpolated over its gaps, demeaned and high-passed, or None.

    Ported from vic_windows.segment (:113-122). One logger spike inside one Welch segment puts a whole day's
    coherence at zero, so the despike runs before the filter; a stretch under min_finite is refused rather
    than filtered, because filtering it would make a number out of interpolation.
    """
    from scipy import signal
    y, _n = despike(np.asarray(x, float))
    fin = np.isfinite(y)
    if not len(fin) or fin.mean() < min_finite or fin.sum() < 16:
        return None
    idx = np.arange(len(y))
    y = np.interp(idx, idx[fin], y[fin])
    return signal.filtfilt(b, a, y - y.mean())


def band_coherence(x, y, fs=1.0, band_s=FLEET_BAND_S, nperseg=FLEET_NPERSEG) -> float:
    """The median squared coherence of two series over a period band. Ported from vic_windows.coherence."""
    from scipy import signal
    n = int(min(nperseg, len(x), len(y)))
    f, C = signal.coherence(x, y, fs=fs, nperseg=n, noverlap=n // 2)
    sel = (f > 1.0 / band_s[1]) & (f < 1.0 / band_s[0])
    return float(np.median(C[sel])) if sel.any() else float("nan")


def _iso_day(t) -> str:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- the magnetics by day

def daily_field(sv, site, b, a) -> dict:
    """{day: (F/F_igrf, high-passed Hx std, high-passed Hz std)} or {day: None}.

    Ported from vic_windows.daily_field (:131-147). A day carrying fewer than 3600 finite samples on all
    three magnetic channels is not scored.
    """
    from scipy import signal
    t0, arr = load_raw(sv, site, 1, ("Hx", "Hy", "Hz"))
    n = len(arr["Hx"])
    lat, lon = site_position(sv, site)
    Fi = geo.igrf(lat, lon, 0.0, datetime.fromtimestamp(t0 + n / 2.0, tz=timezone.utc))["F"]
    out = {}
    d0 = int(t0 // DAY)
    for d in range(int(np.ceil(n / DAY)) + 1):
        ds = (d0 + d) * DAY
        i0, i1 = int(max(0, ds - t0)), int(min(n, ds + DAY - t0))
        if i1 - i0 < 3600:
            continue
        hx, hy, hz = arr["Hx"][i0:i1], arr["Hy"][i0:i1], arr["Hz"][i0:i1]
        fin = np.isfinite(hx) & np.isfinite(hy) & np.isfinite(hz)
        key = _iso_day(ds)
        if fin.sum() < 3600:
            out[key] = None
            continue
        F = float(np.nanmedian(np.sqrt(hx[fin] ** 2 + hy[fin] ** 2 + hz[fin] ** 2)))
        out[key] = (F / Fi,
                    float(np.std(signal.filtfilt(b, a, hx[fin] - hx[fin].mean()))),
                    float(np.std(signal.filtfilt(b, a, hz[fin] - hz[fin].mean()))))
    return out


def daily_magnetics_test(sv, site, k_nearest=2, sites=None) -> pd.DataFrame:
    """One row per UTC day: the field ratio, the two fluctuations, the neighbours' medians and the call.

    A day is sound when |F/F_igrf - 1| < 0.05 AND the high-passed Hx and Hz standard deviations sit within a
    factor 3 of the median of the k nearest sites recording that day. Without a neighbour that day only F is
    judged and the row says so (vic_windows.cmd_days :150-168).
    """
    b, a = highpass(1.0, HIGHPASS_S)
    pool = [s for s in (sites if sites is not None else list(sv.sites.site)) if s != site]
    t0, n = span(sv, site, 1)
    t1 = t0 + n
    covering = []
    for s in pool:
        try:
            st, sn = span(sv, s, 1)
        except Exception:
            continue
        if st + sn > t0 and st < t1:
            covering.append(s)
    near = sorted(covering, key=lambda s: distance_km(sv, site, s))[:int(k_nearest)]
    me = daily_field(sv, site, b, a)
    others = {s: daily_field(sv, s, b, a) for s in near}
    rows = []
    for day in sorted(me):
        if me[day] is None:
            rows.append(dict(day=day, F_ratio=np.nan, hx_std=np.nan, hz_std=np.nan,
                             n_neighbours=0, nb_F_ratio=np.nan, nb_hx_std=np.nan, nb_hz_std=np.nan,
                             sound=False, judged=False, reason="no data"))
            continue
        f_ok = abs(me[day][0] - 1.0) < F_TOLERANCE
        ref = [others[s][day] for s in near if others[s].get(day)]
        if not ref:
            rows.append(dict(day=day, F_ratio=me[day][0], hx_std=me[day][1], hz_std=me[day][2],
                             n_neighbours=0, nb_F_ratio=np.nan, nb_hx_std=np.nan, nb_hz_std=np.nan,
                             sound=bool(f_ok), judged=False,
                             reason="F only: no neighbour recorded that day"))
            continue
        rf = float(np.median([r[0] for r in ref]))
        rx = float(np.median([r[1] for r in ref]))
        rz = float(np.median([r[2] for r in ref]))
        ok = bool(f_ok and rx / FLUCTUATION_FACTOR <= me[day][1] <= rx * FLUCTUATION_FACTOR
                  and rz / FLUCTUATION_FACTOR <= me[day][2] <= rz * FLUCTUATION_FACTOR)
        why = [] if ok else ([] if f_ok else ["F %.3f of IGRF" % me[day][0]])
        if not ok and f_ok:
            why.append("Hx %.1f against %.1f, Hz %.1f against %.1f"
                       % (me[day][1], rx, me[day][2], rz))
        rows.append(dict(day=day, F_ratio=me[day][0], hx_std=me[day][1], hz_std=me[day][2],
                         n_neighbours=len(ref), nb_F_ratio=rf, nb_hx_std=rx, nb_hz_std=rz,
                         sound=ok, judged=True, reason="; ".join(why)))
    out = pd.DataFrame(rows)
    out.attrs["neighbours"] = ["%s (%.0f km)" % (s, distance_km(sv, site, s)) for s in near]
    return out


# ---------------------------------------------------------------- the fleet

def fleet_test(sv, site, t_start, t_end, negative=None, near_km=FLEET_NEAR_KM, sites=None,
               max_sites=8, shift_s=SHIFT_CONTROL_S) -> dict:
    """The site's 100-1000 s Hx-Hx and Hy-Hy coherence against the fleet, read against two controls.

    Every pair of the site and the sites covering the stretch is scored; a site whose median Hx coherence
    with the others is under 0.3 is named and dropped from the control, because a dead sensor or a clock
    hours out is not a control. The POSITIVE limb passes when the site's median with the sites within
    `near_km` is at least 0.8 of the control pairs' median on both channels (vic_windows.cmd_fleet :171-209).

    The NEGATIVE limb is the SHIFTED PAIR: the same stretch, the same estimator, the site's Hx against each
    neighbour's Hx taken `shift_s` = 12 h later (and Hy likewise), medianed over the pairs. It must read
    under 0.3. This replaces the frozen tool's negative stretch of junk magnetics (Ben, after the Q53N run
    of 2026-09-17): a stretch of junk does not exist at a sound site, so a criterion written on one fails by
    construction wherever the magnetics are good and says nothing about the site, while the shifted pair
    exists at every site and tests what the criterion is about -- whether the estimator can tell a coherent
    pair from an incoherent one over this stretch, at this band, with these records.

    `negative` stays available as (t_start, t_end) of a stretch expected to fail, and is reported beside the
    shifted control where a caller supplies one.
    """
    b, a = highpass(1.0, HIGHPASS_S)
    ta, tb = int(t_start), int(t_end)
    pool = [s for s in (sites if sites is not None else list(sv.sites.site)) if s != site]
    covering = []
    for s in pool:
        try:
            st, sn = span(sv, s, 1)
        except Exception:
            continue
        if st <= ta and st + sn >= tb:
            covering.append(s)
    covering = sorted(covering, key=lambda s: distance_km(sv, site, s))[:int(max_sites)]
    chosen = [site] + covering
    segs = {}
    for s in chosen:
        try:
            t0, arr = load_raw(sv, s, 1, H)
        except Exception:
            continue
        for c in H:
            i0, i1 = ta - t0, tb - t0
            segs[(s, c)] = (None if i0 < 0 or i1 > len(arr[c])
                            else prepare(arr[c][i0:i1], b, a))
        del arr
    rows = []
    for i, p in enumerate(chosen):
        for q in chosen[i + 1:]:
            if any(segs.get((s, c)) is None for s in (p, q) for c in H):
                continue
            rows.append(dict(a=p, b=q, km=distance_km(sv, p, q),
                             Hx=band_coherence(segs[(p, "Hx")], segs[(q, "Hx")]),
                             Hy=band_coherence(segs[(p, "Hy")], segs[(q, "Hy")])))
    if not rows:
        return dict(site=site, ok=False, judged=False, table=pd.DataFrame(rows),
                    reason="no other site covers the stretch with 80 per cent finite data",
                    negative_hx=np.nan, negative_hy=np.nan, junk=[], shifted_hx=np.nan,
                    shifted_hy=np.nan, n_shifted=0, shifted_judged=False, shift_s=float(shift_s),
                    positive_ok=False, shifted_ok=False, negative_judged=False)
    t = pd.DataFrame(rows).sort_values("km").reset_index(drop=True)
    per_site = {s: float(np.median([r.Hx for r in t.itertuples() if s in (r.a, r.b)]))
                for s in chosen[1:]}
    junk = sorted(s for s, v in per_site.items() if v < FLEET_JUNK)
    mine = t[(t.a == site) | (t.b == site)]
    ctrl = t[(t.a != site) & (t.b != site) & ~t.a.isin(junk) & ~t.b.isin(junk)]
    note = ""
    near, cnear = mine[mine.km <= near_km], ctrl[ctrl.km <= near_km]
    if near.empty or cnear.empty:
        near, cnear = mine, ctrl
        note = "no pair within %.0f km on one side: all distances used" % near_km
    positive_ok = bool(len(near) and len(cnear)
                       and near.Hx.median() >= FLEET_CONTROL_FRACTION * cnear.Hx.median()
                       and near.Hy.median() >= FLEET_CONTROL_FRACTION * cnear.Hy.median())

    # the shifted pair: the same stretch of the site against each neighbour's record taken shift_s later,
    # or earlier where the neighbour's record does not reach that far
    sx, sy, shifted_rows = np.nan, np.nan, []
    for s in covering:
        if segs.get((site, "Hx")) is None or segs.get((site, "Hy")) is None:
            break
        try:
            t0s, arr = load_raw(sv, s, 1, H)
        except Exception:
            continue
        got = {}
        for sign in (+1, -1):
            j0, j1 = int(ta + sign * shift_s - t0s), int(tb + sign * shift_s - t0s)
            if j0 < 0 or j1 > len(arr["Hx"]):
                continue
            cand = {c: prepare(arr[c][j0:j1], b, a) for c in H}
            if all(v is not None for v in cand.values()):
                got = dict(cand, sign=sign)
                break
        del arr
        if not got:
            continue
        shifted_rows.append(dict(site=s, offset_h=got["sign"] * shift_s / 3600.0,
                                 Hx=band_coherence(segs[(site, "Hx")], got["Hx"]),
                                 Hy=band_coherence(segs[(site, "Hy")], got["Hy"])))
    if shifted_rows:
        sx = float(np.median([r["Hx"] for r in shifted_rows]))
        sy = float(np.median([r["Hy"] for r in shifted_rows]))
    shifted_ok = bool(shifted_rows and sx < NEGATIVE_MAX and sy < NEGATIVE_MAX)

    ok = bool(positive_ok and shifted_ok)
    nx = ny = np.nan
    neg_sites = []
    if negative:
        na, nb = int(negative[0]), int(negative[1])
        vals = []
        # the sites that cover the NEGATIVE stretch, which is a different window from the test stretch and
        # need not be covered by the same sites
        for s in pool:
            try:
                st, sn = span(sv, s, 1)
            except Exception:
                continue
            if st <= na and st + sn >= nb:
                neg_sites.append(s)
        neg_sites = sorted(neg_sites, key=lambda s: distance_km(sv, site, s))[:int(max_sites)]
        t_own, own = load_raw(sv, site, 1, H)
        x = [(None if na - t_own < 0 or nb - t_own > len(own[c])
              else prepare(own[c][na - t_own:nb - t_own], b, a)) for c in H]
        del own
        for s in neg_sites:
            ts, arr = load_raw(sv, s, 1, H)
            y = [(None if na - ts < 0 or nb - ts > len(arr[c])
                  else prepare(arr[c][na - ts:nb - ts], b, a)) for c in H]
            del arr
            if all(v is not None for v in x + y):
                vals.append((band_coherence(x[0], y[0]), band_coherence(x[1], y[1])))
        if vals:
            nx = float(np.median([v[0] for v in vals]))
            ny = float(np.median([v[1] for v in vals]))
            ok = bool(ok and nx < NEGATIVE_MAX and ny < NEGATIVE_MAX)
    return dict(site=site, ok=ok, judged=True, table=t, junk=junk, note=note,
                positive_ok=positive_ok, shifted_ok=shifted_ok, shifted_hx=sx, shifted_hy=sy,
                n_shifted=len(shifted_rows), shifted_judged=bool(shifted_rows),
                shifted_table=pd.DataFrame(shifted_rows), shift_s=float(shift_s),
                negative_sites=neg_sites, n_pairs_site=int(len(near)), n_pairs_control=int(len(cnear)),
                site_hx=float(near.Hx.median()) if len(near) else np.nan,
                site_hy=float(near.Hy.median()) if len(near) else np.nan,
                control_hx=float(cnear.Hx.median()) if len(cnear) else np.nan,
                control_hy=float(cnear.Hy.median()) if len(cnear) else np.nan,
                negative_hx=nx, negative_hy=ny,
                negative_judged=bool(negative and np.isfinite(nx)),
                t_start=ta, t_end=tb)


# ---------------------------------------------------------------- the clock

def clock_prepare(x, fs=1.0, band_s=CLOCK_BAND_S, min_finite=0.8):
    """One stretch despiked, interpolated, demeaned and BAND-PASSED over band_s, or None.

    The band is process.align.SHORT_BAND, 5-20 s. The rest of the package high-passes at 3,000 s for
    coherence, which leaves the daily variation in the series, and the daily variation is a half-day
    sinusoid: correlated against a neighbour it peaks at +-12 h as readily as at zero (Ben, 2026-09-17, from
    Q17's clock figure). A band with no diurnal content in it has no such peak to offer.
    """
    from ..process import align
    y, _n = despike(np.asarray(x, float))
    fin = np.isfinite(y)
    if not len(fin) or fin.mean() < min_finite or fin.sum() < 16:
        return None
    idx = np.arange(len(y))
    y = np.interp(idx, idx[fin], y[fin])
    return align.bandpass(y - y.mean(), float(fs), (1.0 / band_s[1], 1.0 / band_s[0]))


def clock_test(sv, site, ref=None, sites=None, maxlag_s=CLOCK_MAXLAG_S, band_s=CLOCK_BAND_S) -> dict:
    """Per UTC day the lag of the peak cross-correlation of the 5-20 s band-passed Hx, to +-12 h.

    The band is process.align.SHORT_BAND and not the 3,000 s high-pass the coherence tests use: the daily
    variation the high-pass leaves in is a half-day sinusoid, and correlated against a neighbour it peaks at
    the edge of a +-12 h search as readily as at zero. The window stays at +-12 h so an hour-scale offset is
    still found (Ben, 2026-09-17).

    A day is counted only where all four of these hold, and the lag is refined to a fraction of a sample by a
    parabola through the peak and its two neighbours:

        the site's total field is within 20 per cent of IGRF     junk magnetics have nothing to time
        the peak correlation reaches 0.5                          a railed step correlates with anything
        the peak stands 1.5x above the day's own other lags       a broad rise is not a peak
        the peak is not within 5 per cent of the search edge      an edge hit is the window, not a lag

    Fewer than CLOCK_MIN_DAYS = 5 counted days leaves the clock UNJUDGED and no median is reported: at Q17
    two days survived the old rule at +-30,000 s with correlations of 0.52-0.62 against a floor of 0.3-0.4,
    which is a daily-variation correlation at a half-day shift and not a clock, while the campaign measured
    that site at +0.4 s on this band (vic_windows.cmd_clock :247-270).

    The sign is the frozen tool's: a positive lag means the site's samples must move LATER.
    """
    from scipy import signal
    b, a = highpass(1.0, HIGHPASS_S)
    t0, n = span(sv, site, 1)
    t1 = t0 + n
    L = int(maxlag_s)
    pool = [s for s in (sites if sites is not None else list(sv.sites.site)) if s != site]
    if ref is None:
        covering, overlaps = [], []
        for s in pool:
            try:
                st, sn = span(sv, s, 1)
            except Exception:
                continue
            over = max(0, min(st + sn, t1) - max(st, t0))
            if over > 2 * L:
                overlaps.append((over, s))
            if st <= t0 + L and st + sn >= t1 - L:
                covering.append(s)
        if covering:
            ref = sorted(covering, key=lambda s: distance_km(sv, site, s))[0]
        elif overlaps:
            # no site brackets the record with 12 h either side, so the reference is the one that shares
            # the most of it; the days neither record covers are skipped below rather than timed
            ref = max(overlaps)[1]
        else:
            return dict(site=site, ref=None, judged=False, median_lag_s=np.nan, n_days=0, ok=False,
                        table=pd.DataFrame(),
                        reason="no site shares more than %g h of the record" % (2 * L / 3600.0))
    ts, tn = span(sv, ref, 1)
    _t0, own = load_raw(sv, site, 1, ("Hx",))
    _ts, other = load_raw(sv, ref, 1, ("Hx",))
    field = daily_field(sv, site, b, a)
    rows, lags = [], []
    for ds in range(int(t0 // DAY) * int(DAY) + int(DAY), int(t1 - DAY), int(DAY)):
        i0, i1 = ds - t0, ds + int(DAY) - t0
        j0, j1 = ds - L - ts, ds + int(DAY) + L - ts
        if i0 < 0 or i1 > n or j0 < 0 or j1 > tn:
            continue
        x = clock_prepare(own["Hx"][i0:i1], 1.0, band_s)
        y = clock_prepare(other["Hx"][j0:j1], 1.0, band_s)
        if x is None or y is None:
            continue
        c = signal.correlate(y, x, mode="valid", method="fft") / (
            len(x) * np.std(x) * np.std(y[L:L + int(DAY)]) + 1e-12)
        k = int(np.argmax(c))
        lag = float(k - L) + _parabolic(c, k)
        floor = float(np.median(np.abs(c)))
        ratio = float(c[k] / floor) if floor > 0 else np.inf
        edge = bool(abs(k - L) >= CLOCK_EDGE_FRACTION * L)
        f = field.get(_iso_day(ds))
        f_ok = f is not None and abs(f[0] - 1.0) < CLOCK_F_TOLERANCE
        counted = bool(f_ok and c[k] >= CLOCK_MIN_CORR and ratio >= CLOCK_PEAK_RATIO and not edge)
        if counted:
            lags.append(lag)
        why = []
        if not f_ok:
            why.append("F off IGRF")
        if c[k] < CLOCK_MIN_CORR:
            why.append("peak under %.1f" % CLOCK_MIN_CORR)
        if ratio < CLOCK_PEAK_RATIO:
            why.append("peak %.1fx the day's own floor, under %.1f" % (ratio, CLOCK_PEAK_RATIO))
        if edge:
            why.append("an edge hit at %.0f s of a %.0f s search" % (k - L, L))
        rows.append(dict(day=_iso_day(ds), lag_s=round(lag, 2), peak_r=float(c[k]),
                         peak_over_floor=round(ratio, 2), r_at_zero=float(c[L]), edge=edge,
                         F_ratio=(float(f[0]) if f else np.nan), counted=counted,
                         reason="; ".join(why)))
    table = pd.DataFrame(rows)
    judged = len(lags) >= CLOCK_MIN_DAYS
    med = float(np.median(lags)) if judged else np.nan
    return dict(site=site, ref=ref, judged=judged, median_lag_s=med, n_days=len(lags),
                table=table, km=distance_km(sv, site, ref), band_s=list(band_s),
                ok=bool(judged and abs(med) <= CLOCK_PASS_S),
                reason="" if judged else
                       "only %d day(s) of %d carry a peak that is above %.1f, stands %.1fx above the day's "
                       "own other lags and is not within %.0f per cent of the %g h search edge"
                       % (len(lags), len(rows), CLOCK_MIN_CORR, CLOCK_PEAK_RATIO,
                          100 * (1 - CLOCK_EDGE_FRACTION), L / 3600.0))


def _parabolic(c, k) -> float:
    """The sub-sample offset of a correlation peak from a parabola through it and its two neighbours.

    The same refinement process.align.xcorr_peak (:67-71) makes, so a lag of a fraction of a sample -- the
    0.4 s the campaign measured at one Queensland site -- is not rounded away by the sample grid.
    """
    if k <= 0 or k >= len(c) - 1:
        return 0.0
    y0, y1, y2 = float(c[k - 1]), float(c[k]), float(c[k + 1])
    d = y0 - 2 * y1 + y2
    return float(np.clip(0.5 * (y0 - y2) / d, -1.0, 1.0)) if d != 0 else 0.0


# ---------------------------------------------------------------- the quality map

def _cross(ch, nperseg, fs=1.0):
    """The cross-spectral matrix of the named channels on their FIRST DIFFERENCE.

    The difference whitens the red spectrum, so a band's coherence is not dominated by the longest periods
    inside it (vic_quality.multi_coh :54-60).
    """
    from scipy import signal
    keys = list(ch)
    d = {k: np.diff(np.asarray(v, float)) for k, v in ch.items()}
    S, f = {}, None
    for i, p in enumerate(keys):
        for q in keys[i:]:
            f, P = signal.csd(d[p], d[q], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
            S[(p, q)] = P
            S[(q, p)] = np.conj(P)
    return f, S


def multi_coh(ch, lo_s, hi_s, nperseg, fs=1.0):
    """(Ex, Ey) multiple coherence with (Hx, Hy) over a band, bias-corrected. vic_quality.multi_coh :37-78.

    g2c = (g2 - p/nu) / (1 - p/nu) with p = 2 and nu = 0.82 x the number of segments; NaN where the piece
    gives fewer than four windows or nu is at or under 3, because such an estimate saturates at 1 or sits on
    the bias floor. The magnetic matrix is inverted by pseudo-inverse where a flat channel makes it singular.
    """
    n = len(ch["Ex"])
    if n < 4 * nperseg:
        return np.nan, np.nan
    f, S = _cross({k: ch[k] for k in ("Ex", "Ey", "Hx", "Hy")}, nperseg, fs)
    sel = (f > 1.0 / hi_s) & (f < 1.0 / lo_s)
    if sel.sum() < 2:
        return np.nan, np.nan
    nseg = max(1, (n - nperseg) // (nperseg // 2) + 1)
    nu = NU_FACTOR * nseg
    if nu <= 3:
        return np.nan, np.nan
    Shh = np.stack([np.stack([S[("Hx", "Hx")], S[("Hx", "Hy")]], -1),
                    np.stack([S[("Hy", "Hx")], S[("Hy", "Hy")]], -1)], -2)[sel]
    try:
        Sinv = np.linalg.inv(Shh)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(Shh)
    out = []
    for e in ("Ex", "Ey"):
        seh = np.stack([S[(e, "Hx")], S[(e, "Hy")]], -1)[sel]
        she = np.stack([S[("Hx", e)], S[("Hy", e)]], -1)[sel]
        pred = np.einsum("ni,nij,nj->n", seh, Sinv, she).real
        g2 = float(np.median(pred / np.maximum(S[(e, e)][sel].real, 1e-30)))
        out.append(float(np.clip((g2 - MULTI_P / nu) / (1.0 - MULTI_P / nu), 0.0, 1.0)))
    return out[0], out[1]


def pair_coh(x, y, lo_s, hi_s, nperseg, fs=1.0) -> float:
    """Bias-corrected ordinary coherence on the first difference, medianed over four octave sub-bands.

    Most Fourier bins of a wide band sit at its short-period end, so a plain median over 4-50 s is the short
    end's number (vic_quality.pair_coh :81-101).
    """
    from scipy import signal
    n = int(min(len(x), len(y)))
    if n < 4 * nperseg:
        return np.nan
    dx, dy = np.diff(np.asarray(x, float)[:n]), np.diff(np.asarray(y, float)[:n])
    f, C = signal.coherence(dx, dy, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear")
    nseg = max(1, (n - nperseg) // (nperseg // 2) + 1)
    nu = NU_FACTOR * nseg
    if nu <= 2:
        return np.nan
    vals = []
    edges = np.geomspace(lo_s, hi_s, 5)
    for a, b in zip(edges[:-1], edges[1:]):
        s = (f > 1.0 / b) & (f < 1.0 / a)
        if s.sum():
            vals.append(float(np.median(C[s])))
    if not vals:
        return np.nan
    g2 = float(np.median(vals))
    return float(np.clip((g2 - 1.0 / nu) / (1.0 - 1.0 / nu), 0.0, 1.0))


def _clean_series(arr, channels):
    """Every named channel despiked and interpolated over its gaps, for the spectral estimates."""
    out = {}
    for k in channels:
        y, _n = despike(np.asarray(arr[k], float))
        fin = np.isfinite(y)
        idx = np.arange(len(y))
        out[k] = np.interp(idx, idx[fin], y[fin]) if fin.any() else np.zeros(len(y))
    return out


def observatory_pair(sv, site, t0, n, rate=1):
    """({'Hx','Hy'} of the observatory reference on the site's grid, its name), or (None, '').

    The observatory reference store workbook 03 built is the observatory record in its own mean-field frame
    on the target's grid, which is the pair the quality map's second multiple coherence needs. Where no store
    has been built the archive is read directly.
    """
    from ..process import references as REF
    try:
        rt0, h, _mask, info = REF.load_reference("obs", site, rate, sv.cfg["work_root"])
        if rt0 == t0 and len(h["Hx"]) >= n:
            return {c: h[c][:n] for c in H}, str(info.get("observatory") or "observatory")
    except Exception:
        pass
    code = str((sv.cfg.get("observatory") or {}).get("code") or "")
    if not code:
        return None, ""
    try:
        from .. import observatory as OB
        got = OB.load(code, t0, n, (sv.cfg.get("observatory") or {}).get("archive"))
        return {c: np.asarray(got[c], float) for c in H}, code
    except Exception:
        return None, ""


def quality_map(sv, site, rate=1) -> dict:
    """The hourly, daily and whole-record coherence map of one site's two electric lines.

    hourly: one value per hour per line at 4-50 s against the H it couples to. daily: four values per UTC
    day -- the multiple coherence of Ex and Ey with the LOCAL pair, then with the OBSERVATORY pair, at
    50-1000 s. vlong: one (Ex, Ey) pair at 1000-10000 s on the whole record decimated to 0.1 Hz. The
    observatory columns carry no local magnetic noise, which is why the day mask is taken on them
    (vic_quality.cmd_map :116-170, :208-213).
    """
    from scipy import signal
    t0, arr = load_raw(sv, site, rate, ("Ex", "Ey", "Hx", "Hy"))
    clean = _clean_series(arr, ("Ex", "Ey", "Hx", "Hy"))
    n = len(clean["Ex"])
    obs, obs_name = observatory_pair(sv, site, t0, n, rate)
    obs_clean = _clean_series(obs, H) if obs is not None else None
    nh, nd = n // 3600, n // int(DAY)
    hourly = np.full((nh, 2), np.nan)
    for h in range(nh):
        sl = slice(h * 3600, (h + 1) * 3600)
        hourly[h, 0] = pair_coh(clean["Ex"][sl], clean["Hy"][sl], *HOURLY_BAND_S, nperseg=HOURLY_NPERSEG)
        hourly[h, 1] = pair_coh(clean["Ey"][sl], clean["Hx"][sl], *HOURLY_BAND_S, nperseg=HOURLY_NPERSEG)
    daily = np.full((max(nd, 1), 4), np.nan)
    for d in range(nd):
        sl = slice(d * int(DAY), (d + 1) * int(DAY))
        daily[d, 0], daily[d, 1] = multi_coh({k: v[sl] for k, v in clean.items()},
                                             *DAILY_BAND_S, nperseg=DAILY_NPERSEG)
        if obs_clean is not None:
            piece = dict(Ex=clean["Ex"][sl], Ey=clean["Ey"][sl],
                         Hx=obs_clean["Hx"][sl], Hy=obs_clean["Hy"][sl])
            daily[d, 2], daily[d, 3] = multi_coh(piece, *DAILY_BAND_S, nperseg=DAILY_NPERSEG)
    dec = {k: signal.decimate(v, VLONG_DECIMATE, ftype="fir", zero_phase=True) for k, v in clean.items()}
    vlong = multi_coh(dec, *VLONG_BAND_S, nperseg=DAILY_NPERSEG, fs=1.0 / VLONG_DECIMATE)
    days = [_iso_day(t0 + d * DAY) for d in range(nd)]
    return dict(site=site, t0=t0, n=n, hourly=hourly, daily=daily, vlong=vlong, days=days,
                observatory=obs_name, hours=nh, n_days=nd,
                columns=("Ex local", "Ey local", "Ex %s" % (obs_name or "observatory"),
                         "Ey %s" % (obs_name or "observatory")))


def quality_row(qm: dict) -> dict:
    """One row of a quality map: the medians per line and the fraction of hours and days over a half."""
    hx, dy = qm["hourly"], qm["daily"]
    out = dict(site=qm["site"], hours=qm["hours"], days=qm["n_days"], observatory=qm["observatory"])
    for k, col in (("Ex_hourly", 0), ("Ey_hourly", 1)):
        out["%s_med" % k] = float(np.nanmedian(hx[:, col])) if len(hx) else np.nan
        out["%s_over_0p5" % k] = float(np.nanmean(hx[:, col] > 0.5)) if len(hx) else np.nan
    for k, col in (("Ex_local", 0), ("Ey_local", 1), ("Ex_obs", 2), ("Ey_obs", 3)):
        out["%s_med" % k] = float(np.nanmedian(dy[:, col])) if len(dy) else np.nan
        out["%s_days_over_0p5" % k] = float(np.nanmean(dy[:, col] > 0.5)) if len(dy) else np.nan
    out["Ex_vlong"], out["Ey_vlong"] = qm["vlong"]
    return out


# ---------------------------------------------------------------- the day mask and its control

def day_mask(qm: dict, comp: str, thr=DAY_THR, seed=20260916, column="observatory"):
    """(keep, control_keep, rows) -- whole days kept on one line's daily coherence, and the random control.

    `column` is "observatory" (the default, the columns that carry no local magnetic noise) or "local". The
    control keeps the SAME number of days drawn without replacement from the SAME pool of scored days under
    the named seed, so a claim that the selection helped has something to fail against
    (vic_quality.cmd_mask :198-229).
    """
    col = {("xy", "observatory"): 2, ("yx", "observatory"): 3,
           ("xy", "local"): 0, ("yx", "local"): 1}[(comp, column)]
    daily = np.asarray(qm["daily"], float)
    vals = daily[:, col] if daily.shape[1] > col else np.full(len(daily), np.nan)
    scored = np.isfinite(vals)
    good = scored & (vals >= float(thr))
    n_days, n_keep = int(len(vals)), int(good.sum())
    rng = np.random.default_rng(int(seed))
    pool = np.flatnonzero(scored)
    ctrl_days = (rng.choice(pool, size=min(n_keep, len(pool)), replace=False) if n_keep and len(pool)
                 else np.array([], int))
    n = int(qm["n"])

    def _mask(days):
        m = np.zeros(n, bool)
        for d in np.asarray(days, int):
            a = int(d) * int(DAY)
            m[max(0, a):max(0, a + int(DAY))] = True
        return m

    rows = dict(component=comp, column=column, threshold=float(thr), seed=int(seed),
                days_scored=int(scored.sum()), days_total=n_days, days_kept=n_keep,
                days_control=int(len(ctrl_days)),
                kept_days=" ".join(qm["days"][d] for d in np.flatnonzero(good) if d < len(qm["days"])),
                control_days=" ".join(qm["days"][d] for d in sorted(ctrl_days) if d < len(qm["days"])),
                median_kept=float(np.nanmedian(vals[good])) if n_keep else np.nan,
                median_dropped=(float(np.nanmedian(vals[scored & ~good]))
                                if (scored & ~good).any() else np.nan),
                pool="the days the map scored, the selection included",
                enough=bool(n_keep >= DAY_MIN_KEPT))
    return _mask(np.flatnonzero(good)), _mask(ctrl_days), rows


# ---------------------------------------------------------------- the component mask

def sustained_low(tc, v, step_s, win_s, n_samples, thresh=COH_MIN, smooth_h=SMOOTH_H,
                  min_hole_h=MIN_HOLE_H):
    """(keep, [(i0, i1)]) from a per-window series: a sustained low run is masked, a short one given back.

    The running median over smooth_h hours below thresh masks the window's samples; a masked run shorter
    than min_hole_h hours is given back, because a quiet hour of source is not a fault and a run of them is.
    NaN windows are not counted as low. Ported from Processing_Run/wamt_tools.sustained_low (:518-543).
    """
    v = np.asarray(v, float)
    k = max(1, int(round(float(smooth_h) * 3600.0 / float(step_s))))
    sm = pd.Series(v).rolling(k, center=True, min_periods=1).median().to_numpy()
    low = np.isfinite(sm) & (sm < float(thresh))
    bad = np.zeros(int(n_samples), bool)
    for t, b in zip(np.asarray(tc, float), low):
        if b:
            i0, i1 = int(max(0, t - win_s / 2)), int(min(n_samples, t + win_s / 2))
            bad[i0:i1] = True
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.view(np.int8), [0]))))
    keep = np.ones(int(n_samples), bool)
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if (b - a) >= float(min_hole_h) * 3600.0:
            keep[a:b] = False
            out.append((int(a), int(b)))
    return keep, out


def coherence_series(sv, site, comp, band_s=COH_BAND_S, win_s=WIN_S, step_s=STEP_S, nperseg=COH_NPERSEG,
                     rate=1, arrays=None, t0=None):
    """(window centres in samples from t0, the band coherence of the component's pair per window).

    The line is scored against the H it couples to -- Ex with Hy, Ey with Hx -- which is the direct
    measurement of whether that line is following the field.
    """
    line, hchan = COMPONENT_PAIR[comp]
    if arrays is None:
        t0, arr = load_raw(sv, site, rate, (line, hchan))
    else:
        arr = arrays
    clean = _clean_series(arr, (line, hchan))
    n = len(clean[line])
    fs = float(rate)
    w, st = int(win_s * fs), int(step_s * fs)
    tc, vals = [], []
    for i in range(0, max(0, n - w + 1), st):
        tc.append(i + w / 2.0)
        vals.append(band_coherence(clean[line][i:i + w], clean[hchan][i:i + w], fs, band_s,
                                   min(nperseg, w)))
    return np.asarray(tc, float), np.asarray(vals, float), int(n), int(t0 if t0 is not None else 0)


def component_mask(sv, site, comp, coh_band=COH_BAND_S, coh_min=COH_MIN, smooth_h=SMOOTH_H,
                   min_hole_h=MIN_HOLE_H, rate=1, arrays=None, t0=None) -> dict:
    """The per-component coherence mask with its own parameters recorded beside it."""
    tc, v, n, t0 = coherence_series(sv, site, comp, coh_band, rate=rate, arrays=arrays, t0=t0)
    keep, ivl = sustained_low(tc, v, STEP_S, WIN_S, n, coh_min, smooth_h, min_hole_h)
    return dict(component=comp, keep=keep, intervals=ivl, t0=t0, n=n, centres=tc, values=v,
                coh_params=dict(band_s=list(coh_band), coh_min=float(coh_min), smooth_h=float(smooth_h),
                                min_hole_h=float(min_hole_h), win_s=WIN_S, step_s=STEP_S),
                masked_frac=float(1.0 - keep.mean()) if n else 0.0,
                median_coherence=float(np.nanmedian(v)) if len(v) else np.nan)


# ---------------------------------------------------------------- the best hours and their controls

def select_hours(tc, score, fraction, win_s, n, rng=None):
    """(keep, threshold, count) from the best `fraction` of windows by score, or a random `fraction`.

    One function makes the selection and its control, so the two differ only in how the windows are chosen.
    Ported from D:/BEN/MTH5_Aurora_mt-io_2026/student_pack_v2/qld_student.select_hours (:918-933).
    """
    v = np.asarray(score, float)
    ok = np.isfinite(v)
    k = int(round(float(fraction) * ok.sum()))
    if rng is not None:
        chosen = rng.choice(np.flatnonzero(ok), size=k, replace=False) if k else np.array([], int)
        thr = np.nan
    else:
        order = np.argsort(-np.where(ok, v, -np.inf))
        chosen = order[:k]
        thr = float(v[chosen].min()) if k else np.nan
    keep = np.zeros(int(n), bool)
    for i in chosen:
        a, b = int(max(0, tc[i] - win_s / 2)), int(min(n, tc[i] + win_s / 2))
        keep[a:b] = True
    return keep, thr, int(k)


def contiguous_windows(tc, score, hours, n, n_match, win_s=WIN_S):
    """(keep, rows) for one contiguous window length, tiled from the first whole hour.

    Windows of `hours` hours are tiled without overlap, every window scored by the MEAN of its hours, the
    top ones taken until the kept duration matches the scattered selection's to within one window. A single
    Aurora window at the deepest decimation level is 65,536 s, so scattered hours can never reach the long
    periods and only a contiguous selection at the same cost decides a long-period claim
    (vic_short_w2m.cmd_masks :358-388).
    """
    v = np.asarray(score, float)
    step = int(round(win_s))
    per = int(hours)
    idx = np.arange(len(v))
    wins = []
    for a in range(0, len(v) - per + 1, per):
        block = idx[a:a + per]
        if len(block) < per or not np.isfinite(v[block]).all():
            continue
        wins.append((float(np.mean(v[block])), block))
    if not wins:
        return np.zeros(int(n), bool), dict(window_hours=per, n_windows=0, hours=0, kept=np.nan,
                                            dropped=np.nan, discriminates=False, matches=False,
                                            n_available=0)
    wins.sort(key=lambda x: -x[0])
    ntop = max(1, int(round(n_match / float(per))))
    top = wins[:ntop]
    keep = np.zeros(int(n), bool)
    for _s, block in top:
        for i in block:
            a, b = int(max(0, tc[i] - win_s / 2)), int(min(n, tc[i] + win_s / 2))
            keep[a:b] = True
    kept = float(np.mean([s for s, _ in top]))
    dropped = float(np.mean([s for s, _ in wins[ntop:]])) if ntop < len(wins) else np.nan
    hours_kept = int(sum(len(b) for _s, b in top))
    return keep, dict(window_hours=per, n_windows=len(top), hours=hours_kept, kept=kept, dropped=dropped,
                      discriminates=bool(not np.isfinite(dropped) or kept > dropped),
                      matches=bool(abs(hours_kept - n_match) <= per), n_available=len(wins))


def best_hours(sv, site, comp, elines=None, band=HOURS_BAND_S, fraction=HOURS_FRACTION, seed=20260916,
               contig_hours=CONTIG_HOURS, rate=1, states=("sound",), arrays=None, t0=None,
               step_s=WIN_S) -> dict:
    """The selection, its random control and its matched-duration contiguous controls, with the statistic.

    The candidate pool is the hour windows whose day is in `states` in the site's elines table; where that
    leaves fewer than 24 candidates the weak days are admitted and the selection is flagged thin
    (vic_best_hours :62-79). The random control draws the same number of windows without replacement from
    the same pool under the named seed and is expected NOT to separate kept from dropped.

    The scoring windows do not overlap (`step_s` defaults to the window length), so the same number of
    windows is the same duration and the selection, its random control and its contiguous controls all cost
    the same. Overlapping windows would make a clustered selection cover less time than a scattered one of
    the same size and the comparison would not be at the same cost.
    """
    line, hchan = COMPONENT_PAIR[comp]
    if arrays is None:
        t0, arr = load_raw(sv, site, rate, (line, hchan))
    else:
        arr = arrays
    clean = _clean_series(arr, (line, hchan))
    n = len(clean[line])
    fs = float(rate)
    w, st = int(WIN_S * fs), int(float(step_s) * fs)
    tc, score = [], []
    for i in range(0, max(0, n - w + 1), st):
        tc.append(i + w / 2.0)
        score.append(band_coherence(clean[line][i:i + w], clean[hchan][i:i + w], fs, band,
                                    min(HOURS_NPERSEG, w)))
    tc, score = np.asarray(tc, float), np.asarray(score, float)
    tu = (t0 or 0) + tc / fs
    state = np.full(len(tc), "gap", dtype=object)
    if elines is not None and len(elines):
        col = "%s_state" % line
        for r in elines.itertuples():
            m = (tu >= float(r.t_start)) & (tu < float(r.t_end))
            state[m] = getattr(r, col)
    admitted = "days in state %s" % ", ".join(states)
    cand = np.isin(state, list(states)) & np.isfinite(score)
    if cand.sum() < HOURS_MIN_CANDIDATES:
        wider = tuple(states) + ("weak",)
        cand = np.isin(state, list(wider)) & np.isfinite(score)
        admitted = "weak days admitted (fewer than %d candidates on %s)" % (HOURS_MIN_CANDIDATES,
                                                                           ", ".join(states))
    n_cand = int(cand.sum())
    sc = np.where(cand, score, np.nan)
    keep, thr, k = select_hours(tc, sc, fraction, WIN_S * fs, n)
    rnd, _t, k_r = select_hours(tc, sc, fraction, WIN_S * fs, n, rng=np.random.default_rng(int(seed)))

    def _stats(mask):
        chosen = np.zeros(len(tc), bool)
        for i in np.flatnonzero(cand):
            a, b = int(max(0, tc[i] - WIN_S * fs / 2)), int(min(n, tc[i] + WIN_S * fs / 2))
            chosen[i] = bool(mask[a:b].mean() > 0.5) if b > a else False
        kept = float(np.nanmedian(sc[chosen])) if chosen.any() else np.nan
        drop = float(np.nanmedian(sc[cand & ~chosen])) if (cand & ~chosen).any() else np.nan
        return kept, drop, int(chosen.sum())

    kept_med, drop_med, n_sel = _stats(keep)
    r_kept, r_drop, n_rnd = _stats(rnd)
    gap = kept_med - drop_med
    r_gap = r_kept - r_drop
    contig = {}
    for hours in contig_hours:
        m, row = contiguous_windows(tc, sc, hours, n, k, WIN_S * fs)
        contig[int(hours)] = dict(keep=m, **row)
    return dict(site=site, component=comp, band_s=list(band), fraction=float(fraction), seed=int(seed),
                admitted=admitted, n_candidates=n_cand, threshold=thr, n_selected=k,
                thin=bool(n_cand < HOURS_MIN_CANDIDATES),
                keep=keep, random_keep=rnd, contiguous=contig, centres=tc, score=sc,
                days_kept=float(keep.sum() / (DAY * fs)), days_random=float(rnd.sum() / (DAY * fs)),
                median_kept=kept_med, median_dropped=drop_med,
                random_median_kept=r_kept, random_median_dropped=r_drop,
                gap=gap, random_gap=r_gap,
                random_gap_fraction=(float(r_gap / gap) if np.isfinite(gap) and gap > 0 else np.nan),
                random_gap_max=RANDOM_GAP_MAX,
                separates=bool(np.isfinite(kept_med) and np.isfinite(drop_med) and kept_med > drop_med),
                random_separates=bool(np.isfinite(r_gap) and np.isfinite(gap) and gap > 0
                                      and r_gap > RANDOM_GAP_MAX * gap),
                pool="the candidate hour windows, the selection included")


# ---------------------------------------------------------------- windows

def window_from_days(elines: pd.DataFrame, comp: str, states=("sound",), min_days=0.5,
                     fallback_states=("sound", "weak")) -> dict:
    """The longest run of days of one line in `states`, as a proposed window with its reason.

    Where no run in `states` reaches min_days the fallback states are tried and the reason says so, which is
    vic_best_hours' rule for a pool too thin to select from (:78-79) applied to a window. Returns a dict with
    t_start, t_end, days and reason, or days = 0 where the line has no run at all.
    """
    line = COMPONENT_LINE[comp]
    col = "%s_state" % line
    if elines is None or not len(elines) or col not in elines.columns:
        return dict(component=comp, t_start=None, t_end=None, days=0.0, n_days=0,
                    reason="no elines table", states="")

    states_col = list(elines[col])
    t_start = [int(v) for v in elines.t_start]
    t_end = [int(v) for v in elines.t_end]

    def _longest(want):
        best = (0, None, None)
        cur = 0
        for i, s in enumerate(states_col):
            if s in want:
                cur += 1
                if cur > best[0]:
                    best = (cur, t_start[i - cur + 1], t_end[i])
            else:
                cur = 0
        return best

    for want, tag in ((tuple(states), "sound"), (tuple(fallback_states), "fallback")):
        n_days, ta, tb = _longest(want)
        if ta is not None and (tb - ta) / DAY >= float(min_days):
            reason = ("the longest run of %s days of %s: %d day(s), %s to %s"
                      % ("+".join(want), line, n_days, _iso_day(ta), _iso_day(tb)))
            if tag == "fallback":
                reason += ("; no run of %s days reaches the %.2f d floor, so the weak days are admitted and "
                           "the window is a proposal, not a finding" % ("+".join(states), min_days))
            return dict(component=comp, t_start=int(ta), t_end=int(tb), days=round((tb - ta) / DAY, 3),
                        n_days=int(n_days), reason=reason, states="+".join(want))
    return dict(component=comp, t_start=None, t_end=None, days=0.0, n_days=0,
                reason="no run of %s days reaches the %.2f d floor" % ("+".join(states), min_days),
                states="")


def random_block(t0, n, length_samples, seed, exclude=None, fs=1.0):
    """(i0, i1) of a block of the same length placed at random elsewhere in the record.

    The control a window is read against: a selection that buys nothing beyond its length is one a block of
    the same length placed anywhere would buy. `exclude` is the (i0, i1) of the window itself, which the
    draw avoids where the record is long enough to hold both.
    """
    n, L = int(n), int(length_samples)
    if L >= n:
        return 0, n
    rng = np.random.default_rng(int(seed))
    starts = np.arange(0, n - L + 1)
    if exclude is not None:
        a, b = int(exclude[0]), int(exclude[1])
        free = starts[(starts + L <= a) | (starts >= b)]
        if len(free):
            starts = free
    i0 = int(rng.choice(starts))
    return i0, i0 + L


def write_windows(decisions: pd.DataFrame, site: str, windows: dict) -> pd.DataFrame:
    """The decisions.csv `windows` cell of one site, as JSON per component with t_start, t_end and reason.

    The frame is returned and never written: decisions are the analyst's, and workbook 05 prints the cell it
    proposes unless WRITE_DECISIONS says otherwise.
    """
    import json
    out = decisions.copy()
    cell = json.dumps({c: dict(t_start=_iso_day(w["t_start"]) if w.get("t_start") else None,
                               t_end=_iso_day(w["t_end"]) if w.get("t_end") else None,
                               days=w.get("days"), reason=w.get("reason"))
                       for c, w in windows.items() if w.get("t_start")}, sort_keys=True)
    if "site" in out.columns and (out.site == site).any():
        out.loc[out.site == site, "windows"] = cell if cell != "{}" else "decide"
    return out


def windows_table(site: str, run: str, windows: dict, rows: list) -> pd.DataFrame:
    """The sidecar table <work_root>/<site>/windows.csv: one row per component with its evidence.

    The columns are vic_add_windows' (:72-74) with `rule` carrying the evidence, the controls and the
    numbers, so the reason a window was chosen is in the table and not only in the notebook.
    """
    out = []
    for comp, w in sorted(windows.items()):
        extra = next((r for r in rows if r.get("component") == comp), {})
        out.append(dict(site=site, run=run, channel=COMPONENT_LINE[comp], arm="", component=comp,
                        t_start=(_iso_day(w["t_start"]) if w.get("t_start") else ""),
                        t_end=(_iso_day(w["t_end"]) if w.get("t_end") else ""),
                        days=w.get("days"), fraction=extra.get("fraction"),
                        n_live_days=w.get("n_days"), record_days=extra.get("record_days"),
                        in_use=extra.get("in_use", False),
                        rule="%s | %s" % (w.get("reason", ""), extra.get("evidence", ""))))
    return pd.DataFrame(out)
