"""Is the record usable, when, and which hours or days a component is estimated over.

The tests and the selections, each with the criterion it applies and the values that criterion carries.

    daily_magnetics_test   per UTC day the total field against IGRF within 5 per cent and the high-passed
                           Hx and Hz standard deviations within a factor 3 of the median of the k nearest
                           sites recording that day (vic_windows.cmd_days :150-168)
    clock_test             per day the lag of the peak normalised cross-correlation of the despiked,
                           high-passed Hx against a reference site to +-12 h, counted only where F is within
                           20 per cent of IGRF and the peak correlation is at least 0.5; within 10 s passes
                           (vic_windows.cmd_clock :247-270)
    pair_coh               the coherence of one pair of channels over a band, bias-corrected
                           (vic_quality.pair_coh :81-101)
    random_block           a block of a named length placed at random elsewhere in the record under a named
                           seed, the control every selection here is read against
    write_windows          a component's chosen span written back to decisions.csv, and windows_table the row
                           of it a workbook prints

Which hours a component is estimated over is not decided here. The one selection rule of this package is
process.selection: whole UTC hours scored on the decided record, the longest coherent stretch, and its
control on the same hour grid. This module holds the record-level tests the stretch rule reads around.

A coherence is bias-corrected as g2c = (g2 - p/nu) / (1 - p/nu) with p predictors and nu = NU_FACTOR x the
number of 50 per cent overlapping Hann segments, because a few-segment estimate saturates at 1 or sits at the
bias floor; the magnetic cross-spectral matrix is inverted by pseudo-inverse where a flat channel makes it
singular.

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
# vic_windows.py:247-270
CLOCK_MAXLAG_S = 12 * 3600
CLOCK_MIN_CORR = 0.5
CLOCK_F_TOLERANCE = 0.20
CLOCK_PASS_S = 10.0
CLOCK_BAND_S = (5.0, 20.0)          # the band the lag is measured in, process.align.SHORT_BAND
CLOCK_EDGE_FRACTION = 0.95          # a peak this far out in the search window is an edge hit, not a lag
CLOCK_PEAK_RATIO = 1.5              # ... and a peak must stand this far above the day's own other lags
CLOCK_MIN_DAYS = 5                  # below this many counted days the clock limb is not judged
NU_FACTOR = 0.82                    # effective independent segments per 50 per cent overlapping Hann window
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
    """(t0, n) of one site's cache."""
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

    One logger spike inside one Welch segment puts a whole day's coherence at zero, so the despike runs
    before the filter; a stretch under min_finite is refused rather than filtered, because filtering it
    would make a number out of interpolation.
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
    """The median squared coherence of two series over a period band."""
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

    A day carrying fewer than 3600 finite samples on all three magnetic channels is not scored
    (vic_windows.daily_field :131-147).
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

    A day is sound when |F/F_igrf - 1| < 0.05 and the high-passed Hx and Hz standard deviations sit within a
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
# ---------------------------------------------------------------- the clock

def clock_prepare(x, fs=1.0, band_s=CLOCK_BAND_S, min_finite=0.8):
    """One stretch despiked, interpolated, demeaned and band-passed over band_s, or None.

    The band is process.align.SHORT_BAND, 5-20 s, which carries no diurnal content and so offers no
    correlation peak at the +-12 h edge of the clock search.
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

    The band is process.align.SHORT_BAND, 5-20 s, which carries no diurnal content and so offers no
    half-day peak. The window stays at +-12 h so an hour-scale offset is still found (Ben, 2026-09-17).

    A day is counted only where all four of these hold, and the lag is refined to a fraction of a sample by a
    parabola through the peak and its two neighbours:

        the site's total field is within 20 per cent of IGRF     junk magnetics have nothing to time
        the peak correlation reaches 0.5                          a railed step correlates with anything
        the peak stands 1.5x above the day's own other lags       a broad rise is not a peak
        the peak is not within 5 per cent of the search edge      an edge hit is the window, not a lag

    Fewer than CLOCK_MIN_DAYS = 5 counted days leaves the clock unjudged and no median is reported
    (vic_windows.cmd_clock :247-270).

    A positive lag means the site's samples must move later.
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
    """The cross-spectral matrix of the named channels on their first difference.

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
