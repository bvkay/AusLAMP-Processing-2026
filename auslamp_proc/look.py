"""Stage 2, look first: the magnetometer DC test against IGRF and the per-day state of the two electric lines.

The magnetometer test (ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/vic_dc.py:55-58) reads the
record medians Bx, By, Bz as laid against IGRF at the site and the record midpoint, with H = sqrt(Bx^2 + By^2)
and F = sqrt(Bx^2 + By^2 + Bz^2). Its rules, in the order they fire:

    |F / F_igrf - 1| > 0.05                      a gain or a broken axis; a tilt leaves F alone
    else |Bz / Z - 1| > 0.10 or |H / H_igrf - 1| > 0.10   a tilt, reported with its angle
    Bx < 0                                       a reversed north axis
    |atan2(By, Bx)| > 30 deg                     the sensor laid far from north, or the axes exchanged

The electric-line test (ported from vic_windows.py:273-312, cmd_elines) scores each UTC day of at least 6 h.
Per day the 20-200 s squared coherence of Ex with the site's own Hy, of Ey with its Hx and of Ex with Ey
(Welch, nperseg 4096 samples, noverlap 2048), and the standard deviation of each line after a 3,000 s
high-pass (Butterworth order 2). The state per line:

    dead    high-passed std < 0.2 mV/km
    sound   coherence with H >= 0.4 and Ex-Ey coherence < 0.5
    common  Ex-Ey coherence >= 0.6 and coherence with H < 0.3
    weak    otherwise

Every series is despiked before the high-pass (vic_windows.py:95): a sample whose one-second step exceeds 30
times the robust scale of the steps (1.4826 x the median absolute deviation) is blanked with samples i-2 to
i+3 and interpolated, because one logger spike in one Welch segment puts the whole day's coherence at zero.
A day in which any of the four channels is less than 80 per cent finite is scored `gap` and counts as no day.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .geo import igrf

# vic_dc.py:55-58
F_GAIN_TOLERANCE = 0.05
TILT_TOLERANCE = 0.10
ANGLE_TOLERANCE_DEG = 30.0

# vic_windows.py:95 and :294-298
DESPIKE_K = 30.0
HIGHPASS_S = 3000.0
DEAD_STD_MV_PER_KM = 0.2
SOUND_COH = 0.4
SOUND_EX_EY_MAX = 0.5
COMMON_EX_EY = 0.6
COMMON_COH_MAX = 0.3
BAND_S = (20.0, 200.0)
NPERSEG = 4096
NOVERLAP = 2048
MIN_FINITE = 0.8
MIN_DAY_HOURS = 6.0

STATES = ("sound", "weak", "common", "dead", "gap")


# ---------------------------------------------------------------- the magnetometer

def dc_test(arrays, lat, lon, elev_m, when) -> dict:
    """The record's DC field against IGRF, with the four flags of vic_dc.py:55-58.

    `arrays` holds Hx, Hy, Hz in nT as laid (no sign, no rotation). `when` is the record midpoint as a
    datetime. Returns one row: the medians, the IGRF values, the three ratios, the tilt and sensor angles, the
    flags and a verdict of PASS or CHECK.
    """
    bx = float(np.nanmedian(arrays["Hx"]))
    by = float(np.nanmedian(arrays["Hy"]))
    bz = float(np.nanmedian(arrays["Hz"]))
    ig = igrf(lat, lon, elev_m, when)
    X, Y, Z, Hi, Fi = ig["X"], ig["Y"], ig["Z"], ig["H"], ig["F"]
    H = float(np.hypot(bx, by))
    F = float(np.sqrt(bx * bx + by * by + bz * bz))
    ang = float(np.degrees(np.arctan2(by, bx)))
    tilt = float(np.degrees(np.arctan2(H, -bz) - np.arctan2(Hi, -Z)))
    flags = []
    if abs(F / Fi - 1) > F_GAIN_TOLERANCE:
        flags.append("F/Figrf %.2f (gain or a broken axis)" % (F / Fi))
    elif abs(bz / Z - 1) > TILT_TOLERANCE or abs(H / Hi - 1) > TILT_TOLERANCE:
        flags.append("tilt %+.1f deg (H %.2f, Z %.2f, F %.3f)" % (tilt, H / Hi, bz / Z, F / Fi))
    if bx < 0:
        flags.append("Bx negative")
    if abs(ang) > ANGLE_TOLERANCE_DEG:
        flags.append("angle %+.0f deg" % ang)
    return dict(Bx=round(bx, 1), By=round(by, 1), Bz=round(bz, 1), H=round(H, 1), F=round(F, 1),
                igrf_X=round(X, 1), igrf_Y=round(Y, 1), igrf_Z=round(Z, 1), igrf_H=round(Hi, 1),
                igrf_F=round(Fi, 1), igrf_D=round(ig["D"], 2),
                Bx_over_X=round(bx / X, 4), Bz_over_Z=round(bz / Z, 4),
                H_over_Higrf=round(H / Hi, 4), F_over_Figrf=round(F / Fi, 4),
                angle_deg=round(ang, 2), tilt_deg=round(tilt, 2),
                flags="; ".join(flags), verdict="PASS" if not flags else "CHECK")


# ---------------------------------------------------------------- the electric lines

def despike(x, k: float = DESPIKE_K):
    """Blank the samples around any one-sample step beyond k times the robust scale of the steps.

    Ported from vic_windows.py:95. The scale is 1.4826 x the median absolute deviation of the first
    difference; samples i-2 to i+3 around a hit are set to NaN. A quiet day's one-second steps scale at
    0.05-0.1 nT, a storm's sudden commencement at 0.5 nT/s and a logger spike at thousands, so k = 30 sits
    between them. Returns (series, samples blanked).
    """
    x = np.asarray(x, float).copy()
    fin = np.isfinite(x)
    if fin.sum() < 10:
        return x, 0
    d = np.diff(x)
    good = np.isfinite(d)
    if good.sum() < 10:
        return x, 0
    mad = np.median(np.abs(d[good] - np.median(d[good]))) * 1.4826
    bad = np.zeros(len(x), bool)
    for i in np.flatnonzero(good & (np.abs(d) > k * mad + 1e-9)):
        bad[max(0, i - 2):i + 4] = True
    x[bad] = np.nan
    return x, int(bad.sum())


def highpass(fs: float = 1.0, period_s: float = HIGHPASS_S):
    """The order-2 Butterworth high-pass at `period_s`, as vic_windows.hp (:57) builds it, with fs a parameter."""
    from scipy import signal
    return signal.butter(2, (1.0 / period_s) / (fs / 2.0), "high")


def prepare(x, b, a, min_finite: float = MIN_FINITE):
    """(series or None, finite fraction after the despike, samples blanked) for one stretch.

    The stretch is despiked, interpolated over its gaps, demeaned and high-passed. A stretch less than
    `min_finite` = 0.80 finite after the despike returns None: filtering it would make a number out of
    interpolation. The fraction is returned whether or not the stretch is kept, so a caller can say which
    channel refused a day and by how much.
    """
    from scipy import signal
    x, n_blanked = despike(np.asarray(x, float))
    fin = np.isfinite(x)
    frac = float(fin.mean()) if len(fin) else 0.0
    if frac < min_finite or fin.sum() < 16:
        return None, frac, n_blanked
    idx = np.arange(len(x))
    x = np.interp(idx, idx[fin], x[fin])
    return signal.filtfilt(b, a, x - x.mean()), frac, n_blanked


def segment(x, b, a, min_finite: float = MIN_FINITE):
    """One stretch despiked, interpolated, demeaned and high-passed, or None. Ported from vic_windows:113."""
    return prepare(x, b, a, min_finite)[0]


def band_coherence(x, y, fs: float = 1.0, lo_s: float = BAND_S[0], hi_s: float = BAND_S[1],
                   nperseg: int = NPERSEG, noverlap: int = NOVERLAP) -> float:
    """The median squared coherence of two series over lo_s..hi_s. Ported from vic_windows.cmd_elines coh."""
    from scipy import signal
    n = int(min(nperseg, len(x), len(y)))
    f, C = signal.coherence(x, y, fs=fs, nperseg=n, noverlap=min(noverlap, n // 2))
    band = (f > 1.0 / hi_s) & (f < 1.0 / lo_s)
    return float(np.median(C[band])) if band.any() else float("nan")


def e_state(coh_with_h: float, coh_ex_ey: float, std: float) -> str:
    """The per-day state of one electric line. The rule with its values is in the module docstring."""
    if not np.isfinite(std):
        return "gap"
    if std < DEAD_STD_MV_PER_KM:
        return "dead"
    if coh_with_h >= SOUND_COH and coh_ex_ey < SOUND_EX_EY_MAX:
        return "sound"
    if coh_ex_ey >= COMMON_EX_EY and coh_with_h < COMMON_COH_MAX:
        return "common"
    return "weak"


def elines(t0, arrays, fs: float = 1.0, min_day_hours: float = MIN_DAY_HOURS) -> pd.DataFrame:
    """The per-UTC-day table of the two electric lines. Ported from vic_windows.cmd_elines (:273-312).

    `t0` is the first sample's time in seconds since the epoch. Each UTC day holding at least
    `min_day_hours` = 6 h of axis is scored; a day in which any of Hx, Hy, Ex, Ey is under 80 per cent finite
    is scored `gap` on both lines and is not counted as a day. One row per day with the three coherences, the
    two high-passed standard deviations and the two states.
    """
    b, a = highpass(fs)
    n = len(arrays["Hx"])
    t_end = t0 + n / fs
    rows = []
    day = int(t0 // 86400) * 86400
    while day < t_end:
        ta, tb = max(day, t0), min(day + 86400, t_end)
        day += 86400
        if tb - ta < min_day_hours * 3600.0:
            continue
        i0, i1 = int(round((ta - t0) * fs)), int(round((tb - t0) * fs))
        label = datetime.fromtimestamp(ta, tz=timezone.utc).strftime("%Y-%m-%d")
        prepped = {c: prepare(arrays[c][i0:i1], b, a) for c in ("Hx", "Hy", "Ex", "Ey")}
        s = {c: v[0] for c, v in prepped.items()}
        if any(v is None for v in s.values()):
            # a day is refused for no data or for a channel the despike emptied; the line says which and by
            # how much, because the two are different faults and the state is the same word
            why = "; ".join("%s %.2f finite after the despike" % (c, prepped[c][1])
                            for c in ("Hx", "Hy", "Ex", "Ey") if s[c] is None)
            rows.append(dict(day=label, t_start=int(ta), t_end=int(tb), coh_Ex_Hy=np.nan,
                             coh_Ey_Hx=np.nan, coh_Ex_Ey=np.nan, std_Ex=np.nan, std_Ey=np.nan,
                             Ex_state="gap", Ey_state="gap", gap_reason=why))
            continue
        cx = band_coherence(s["Ex"], s["Hy"], fs)
        cy = band_coherence(s["Ey"], s["Hx"], fs)
        ce = band_coherence(s["Ex"], s["Ey"], fs)
        sx, sy = float(np.std(s["Ex"])), float(np.std(s["Ey"]))
        rows.append(dict(day=label, t_start=int(ta), t_end=int(tb),
                         coh_Ex_Hy=round(cx, 3), coh_Ey_Hx=round(cy, 3), coh_Ex_Ey=round(ce, 3),
                         std_Ex=round(sx, 3), std_Ey=round(sy, 3),
                         Ex_state=e_state(cx, ce, sx), Ey_state=e_state(cy, ce, sy), gap_reason=""))
    return pd.DataFrame(rows, columns=["day", "t_start", "t_end", "coh_Ex_Hy", "coh_Ey_Hx", "coh_Ex_Ey",
                                       "std_Ex", "std_Ey", "Ex_state", "Ey_state", "gap_reason"])


def eline_summary(table: pd.DataFrame, site: str = "") -> dict:
    """The state counts and fractions per line over the scored days of one site's elines table.

    A `gap` day is not a scored day: the fractions are over the days that carry four channels at least 80 per
    cent finite. `days_scored` of zero is the UNJUDGED case the workbook's check counts as a failure.
    """
    out = {"site": site, "days_total": int(len(table))}
    scored = table[(table.Ex_state != "gap") | (table.Ey_state != "gap")] if len(table) else table
    out["days_scored"] = int(len(scored))
    for line in ("Ex", "Ey"):
        counts = {s: 0 for s in STATES}
        if len(table):
            for s, c in table["%s_state" % line].value_counts().items():
                counts[s] = int(c)
        for s in STATES:
            out["%s_%s" % (line, s)] = counts[s]
        n = max(1, out["days_scored"])
        out["%s_sound_frac" % line] = round(counts["sound"] / n, 3)
        out["%s_common_frac" % line] = round(counts["common"] / n, 3)
        out["%s_dead_frac" % line] = round(counts["dead"] / n, 3)
        out["%s_weak_frac" % line] = round(counts["weak"] / n, 3)
        out["%s_longest_sound_run_d" % line] = _longest_run(table["%s_state" % line]) if len(table) else 0
    both = int(((table.Ex_state == "sound") & (table.Ey_state == "sound")).sum()) if len(table) else 0
    out["both_sound_days"] = both
    out["both_sound_frac"] = round(both / max(1, out["days_scored"]), 3)
    chans: list[str] = []
    if len(table) and "gap_reason" in table.columns:
        for reason in table.gap_reason:
            chans += [part.strip().split()[0] for part in str(reason).split(";") if part.strip()]
    out["gap_channels"] = " ".join("%s:%d" % (c, chans.count(c)) for c in sorted(set(chans)))
    return out


def _longest_run(states, want: str = "sound") -> int:
    best = cur = 0
    for s in states:
        cur = cur + 1 if s == want else 0
        best = max(best, cur)
    return int(best)
