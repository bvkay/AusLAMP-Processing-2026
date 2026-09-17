"""Member alignment: the delay a stack member is advanced by before it enters the sum.

A delay on a single remote cancels exactly in Z, so a remote site is never shifted and neither is the
observatory. It does not cancel in a stack: members whose delays differ by 2-3 s partly cancel each other
when they are summed, and 2.5 s is 90 degrees at a 10 s period.

The lag is measured on the 5-20 s cross-correlation of the event-free overlap, as the median over up to
MAX_WINDOWS = 8 windows spread across the record -- spread and not the first eight, because a clock that
starts drifting on day 22 would never be seen otherwise. Three gates, all from wamt_align.shift_for:

    the lag has to be measurable at all (peak correlation at least MIN_CORR = 0.35 in a window);
    it has to be one constant across the record, spread <= MAX_SPREAD_S = 1.0 s, because no single shift
        fixes a free-running clock and such a member is worse aligned than not;
    the member has to correlate with the target at 20-200 s at MIN_LONG_CORR = 0.5 at all, or the lag was
        measured on noise.

A refusal returns lag NaN and the caller decides: a stack member that cannot be aligned is dropped, and the
observatory is kept unshifted with the reason recorded. On GPS-disciplined loggers a refusal for want of
windows means the pair is kept at lag 0. Where the 2-day window yields fewer than two windows the batch's
survey.yaml `floors.align_fallback_s` (12 h) is tried instead and the sidecar says which window was used.

The delay itself is applied by shift() as an integer roll and a Lanczos-windowed sinc of LANCZOS_A = 16 taps
either side. The interpolation is local, so an impulsive sample is not spread and a NaN reaches only the
2 x LANCZOS_A samples around it.

The lag is measured after references.EDGE_S = 120 s is NaN inside every record end and every gap end,
because the logger's settling there biases the 5-20 s cross-correlation.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

SHORT_BAND = (1 / 20.0, 1 / 5.0)
LONG_BAND = (1 / 200.0, 1 / 20.0)
WIN_S = 2 * 86400
MAXLAG_S = 30
MAX_WINDOWS = 8
MIN_WINDOWS = 2
MIN_CORR = 0.35
MAX_SPREAD_S = 1.0
MIN_LONG_CORR = 0.5
OBS_MAX_SPREAD_S = 5.0
LANCZOS_A = 16                        # taps either side of the fractional delay
H = ("Hx", "Hy")


def bandpass(x, fs, band=SHORT_BAND):
    from scipy import signal
    b, a = signal.butter(4, [band[0], band[1]], btype="band", fs=fs)
    return signal.filtfilt(b, a, np.asarray(x, float))


def xcorr_peak(a, b, maxlag):
    """(lag in samples, peak normalised correlation); a positive lag means b is late.

    cc[k] = <a[i], b[i + k]>, so a peak at +k puts the target's feature at index i and the member's copy of
    it at i + k. The inputs must be one contiguous stretch: gathering the valid samples and filtering the
    concatenation puts a step at every join and the estimator reads the join.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    L = int(maxlag)
    if len(a) <= 4 * L:
        return np.nan, np.nan
    core = a[L:len(a) - L]
    cc = np.array([float(np.dot(core, b[L + k:len(b) - L + k])) for k in range(-L, L + 1)])
    na = float(np.dot(core, core))
    k = int(np.argmax(cc))
    f = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        d = y0 - 2 * y1 + y2
        f = float(np.clip(0.5 * (y0 - y2) / d, -1.0, 1.0)) if d != 0 else 0.0
    seg = b[L + (k - L):len(b) - L + (k - L)]
    nb = float(np.dot(seg, seg))
    r = cc[k] / np.sqrt(na * nb) if na > 0 and nb > 0 else np.nan
    return float((k - L) + f), float(r)


def _pick_windows(ok, win, max_windows):
    """Up to max_windows disjoint win-long slices, spread over the record rather than taken from its head."""
    from .transients import segments
    got = []
    for a, L in segments(ok, win):
        for k in range(0, L - win + 1, win):
            got.append(a + k)
    if len(got) <= max_windows:
        return got
    idx = np.linspace(0, len(got) - 1, max_windows).round().astype(int)
    return [got[i] for i in sorted(set(idx.tolist()))]


def lag(target_h: dict, member_h: dict, fs, band=SHORT_BAND, maxlag_s=MAXLAG_S, keep=None,
        win_s=WIN_S, max_windows=MAX_WINDOWS, min_corr=MIN_CORR) -> dict:
    """How far the member is delayed behind the target, as the median over event-free windows.

    Both arguments are on one common grid. `spread` is the range of the per-window medians and is the gate
    on a free-running clock: a stable pair is flat and a drifting one is not, and a mean would hide it.
    """
    chans = [c for c in H if c in target_h and c in member_h]
    n = min(min(len(target_h[c]) for c in chans), min(len(member_h[c]) for c in chans))
    ok = np.ones(n, bool) if keep is None else np.asarray(keep, bool)[:n].copy()
    for c in chans:
        ok &= np.isfinite(target_h[c][:n]) & np.isfinite(member_h[c][:n])
    win = int(win_s * fs)
    starts = _pick_windows(ok, win, max_windows)
    rows = []
    for i0 in starts:
        sl = slice(i0, i0 + win)
        for c in chans:
            l, r = xcorr_peak(bandpass(target_h[c][sl], fs, band),
                              bandpass(member_h[c][sl], fs, band), int(maxlag_s * fs))
            if np.isfinite(l) and np.isfinite(r) and r >= min_corr:
                rows.append((i0, c, l / fs, r))
    if not rows:
        return dict(lag_s=np.nan, corr=np.nan, spread=np.nan, n_windows=0, n_offered=len(starts),
                    win_s=float(win_s))
    per_win: dict = {}
    for i0, c, l, r in rows:
        per_win.setdefault(i0, []).append(l)
    wmed = np.array([np.median(v) for v in per_win.values()])
    return dict(lag_s=float(np.median(wmed)), corr=float(np.median([r[3] for r in rows])),
                spread=float(wmed.max() - wmed.min()) if len(wmed) > 1 else 0.0,
                n_windows=len(per_win), n_offered=len(starts), win_s=float(win_s))


def shift(arr, lag_s, fs):
    """`arr` advanced by lag_s seconds: out[i] = arr[i + lag_s * fs], so a member shifted by its own lag
    against the target lines up with it.

    The integer part is a roll with NaN fill and the fractional part a Lanczos-windowed sinc of
    LANCZOS_A = 16 taps either side. The interpolation is local: an impulsive sample stays where it is and a
    NaN reaches only the 2 x LANCZOS_A samples around it. A Fourier phase ramp over a finite run does
    neither -- it spreads one impulsive sample into a sinc train decaying as 1/n over the whole run, and Q45
    (Queensland Phase 3), whose record opens with the logger settling from 16,804 to 33,214 to 29,234 nT
    over about 10 s, goes from 78 to 4,384 steps above 2 nT per million samples under its 0.172 s shift.

    The rms error against an analytic sine shifted 0.37 s is 0.3 per cent of the amplitude at a 2.3 s period
    and 0.09 per cent at 2.5 s, and falls with period; 2.3 s is the shortest band the 1 Hz parameter set
    estimates. There is no run segmentation, no reflection padding and no minimum run.
    """
    if isinstance(arr, dict):
        return {k: shift(v, lag_s, fs) for k, v in arr.items()}
    x = np.asarray(arr, float)
    if not np.isfinite(lag_s) or lag_s == 0.0:
        return x.copy()
    n = len(x)
    d = float(lag_s) * float(fs)
    k = int(np.floor(d))
    f = d - k
    y = np.full(n, np.nan)
    if 0 <= k < n:
        y[:n - k] = x[k:]
    elif -n < k < 0:
        y[-k:] = x[:n + k]
    if f == 0.0:
        return y
    a = int(LANCZOS_A)
    taps = np.arange(-a + 1, a + 1)
    t = taps - f
    w = np.sinc(t) * np.sinc(t / a)
    w /= w.sum()
    good = np.isfinite(y)
    yf = np.where(good, y, 0.0)
    out = np.zeros(n)
    bad = np.zeros(n, bool)
    for tap, wt in zip(taps, w):
        lo, hi = (0, n - tap) if tap >= 0 else (-tap, n)
        if hi <= lo:
            bad[:] = True
            continue
        src = slice(tap, n) if tap >= 0 else slice(0, n + tap)
        out[lo:hi] += wt * yf[src]
        bad[lo:hi] |= ~good[src]
        bad[:lo] = True
        bad[hi:] = True
    out[bad] = np.nan
    return out


def member_lag(target_h: dict, member_h: dict, fs, keep, fallback_s=None) -> dict:
    """The short-band lag of one member, with the 12 h fallback where the 2-day form finds under two windows.

    Ben's ruling of 2026-09-12: a member with fewer than two full 2-day windows is not thereby unusable.
    """
    r = lag(target_h, member_h, fs, SHORT_BAND, keep=keep, win_s=WIN_S)
    if fallback_s and r.get("n_windows", 0) < MIN_WINDOWS:
        r2 = lag(target_h, member_h, fs, SHORT_BAND, keep=keep, win_s=float(fallback_s))
        if r2.get("n_windows", 0) > r.get("n_windows", 0):
            r2["why_window"] = ("fewer than %d windows of %.1f d; measured on %.1f h windows instead"
                                % (MIN_WINDOWS, WIN_S / 86400.0, float(fallback_s) / 3600.0))
            r = r2
    return r


def shift_for(target_h: dict, member_h: dict, fs, keep, fallback_s=None, obs=False) -> tuple:
    """(lag_s, refusal) for one member; the refusal is None where the lag was measured and accepted.

    A refusal returns lag 0.0 and the reason: these are GPS-disciplined loggers, so a member whose lag
    cannot be measured or is not one constant is kept at zero rather than dropped, and the reason goes into
    the reference's sidecar. The three gates are stated in this module's docstring.
    """
    max_spread = OBS_MAX_SPREAD_S if obs else MAX_SPREAD_S
    s = member_lag(target_h, member_h, fs, keep, fallback_s)
    long = lag(target_h, member_h, fs, LONG_BAND, keep=keep, win_s=s.get("win_s", WIN_S))
    if not np.isfinite(s["lag_s"]):
        return 0.0, ("no usable %g-%g s window in %d offered (peak correlation below %.2f)"
                     % (1 / SHORT_BAND[1], 1 / SHORT_BAND[0], s["n_offered"], MIN_CORR))
    if not np.isfinite(long["corr"]) or long["corr"] < MIN_LONG_CORR:
        return 0.0, ("20-200 s peak correlation %s < %.1f: the lag was measured on noise"
                     % ("nan" if not np.isfinite(long["corr"]) else "%.2f" % long["corr"], MIN_LONG_CORR))
    if np.isfinite(s["spread"]) and s["spread"] > max_spread:
        return 0.0, ("lag spread %.2f s over %d windows > %.1f s: not one constant, so no single shift "
                     "fixes it (the measured lag would have been %+.2f s)"
                     % (s["spread"], s["n_windows"], max_spread, s["lag_s"]))
    return float(s["lag_s"]), None
