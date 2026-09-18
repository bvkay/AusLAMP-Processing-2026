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

    dead    high-passed std < DEAD_FRACTION x the line's own median daily std over the scored days
            (0.2, Ben's ruling of 2026-09-16; the source used 0.2 mV/km absolute), or below the absolute
            floor DEAD_ABS_MV_PER_KM where one is set (0 = off). Both are parameters of the workbook,
            because the level of a line depends on the dipole length and the survey.
    sound   coherence with H >= 0.4 and Ex-Ey coherence < 0.5
    common  Ex-Ey coherence >= 0.6 and coherence with H < 0.3
    weak    otherwise

A line dead for the whole record has its median at its own noise floor, so no day falls under 0.2 of it and
the line reads `weak` on its coherence with H; the absolute floor is the switch for that case.

Every series is despiked before the high-pass (vic_windows.py:95): a sample whose one-second step exceeds 30
times the robust scale of the steps (1.4826 x the median absolute deviation) is blanked with samples i-2 to
i+3 and interpolated, because one logger spike in one Welch segment puts the whole day's coherence at zero.
A day in which any of the four channels is less than 80 per cent finite is scored `gap` and counts as no day.

Two of the five signs are decided from these tests, and each writes what it measured into decisions.csv
beside its value. `signs_from_dc` reads sign_hx and sign_hz off the DC ratios against IGRF with the 0.3
floor. `quadrant_sign` reads sign_ex and sign_ey off the phase of a transfer function estimated with a
sound H, which is workbook 03's cell because that is where the transfer function is. Neither guesses: a
ratio inside the floor and a phase that is in neither quadrant leave the sign at `decide`.

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
DEAD_FRACTION = 0.2            # of the line's own median daily high-passed std (Ben, 2026-09-16)
DEAD_ABS_MV_PER_KM = 0.0       # an absolute floor in mV/km; 0 = off
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
def band_coherence(x, y, fs: float = 1.0, lo_s: float = BAND_S[0], hi_s: float = BAND_S[1],
                   nperseg: int = NPERSEG, noverlap: int = NOVERLAP) -> float:
    """The median squared coherence of two series over lo_s..hi_s. Ported from vic_windows.cmd_elines coh."""
    from scipy import signal
    n = int(min(nperseg, len(x), len(y)))
    f, C = signal.coherence(x, y, fs=fs, nperseg=n, noverlap=min(noverlap, n // 2))
    band = (f > 1.0 / hi_s) & (f < 1.0 / lo_s)
    return float(np.median(C[band])) if band.any() else float("nan")


def dead_threshold(stds, dead_fraction: float = DEAD_FRACTION,
                   dead_abs_mv_per_km: float = DEAD_ABS_MV_PER_KM) -> float:
    """The std below which a day of one line is dead: dead_fraction x the median of the line's daily stds,
    or the absolute floor, whichever is larger. NaN when no day was scored."""
    s = np.asarray(stds, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return float("nan")
    return float(max(dead_fraction * np.median(s), dead_abs_mv_per_km))


def e_state(coh_with_h: float, coh_ex_ey: float, std: float, dead_std: float) -> str:
    """The per-day state of one electric line; `dead_std` is dead_threshold() of that line. The rule with its
    values is in the module docstring."""
    if not np.isfinite(std):
        return "gap"
    if np.isfinite(dead_std) and std < dead_std:
        return "dead"
    if coh_with_h >= SOUND_COH and coh_ex_ey < SOUND_EX_EY_MAX:
        return "sound"
    if coh_ex_ey >= COMMON_EX_EY and coh_with_h < COMMON_COH_MAX:
        return "common"
    return "weak"


def elines(t0, arrays, fs: float = 1.0, min_day_hours: float = MIN_DAY_HOURS,
           dead_fraction: float = DEAD_FRACTION, dead_abs_mv_per_km: float = DEAD_ABS_MV_PER_KM) -> pd.DataFrame:
    """The per-UTC-day table of the two electric lines. Ported from vic_windows.cmd_elines (:273-312).

    `t0` is the first sample's time in seconds since the epoch. Each UTC day holding at least
    `min_day_hours` = 6 h of axis is scored; a day in which any of Hx, Hy, Ex, Ey is under 80 per cent finite
    is scored `gap` on both lines and is not counted as a day. One row per day with the three coherences, the
    two high-passed standard deviations and the two states. The dead threshold of each line is
    dead_threshold() over the scored days and is carried in the columns dead_std_Ex and dead_std_Ey.
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
                         Ex_state="", Ey_state="", gap_reason=""))
    # the dead threshold is relative to the line's own record, so the states are assigned once every day is in
    dead = {c: dead_threshold([r["std_%s" % c] for r in rows], dead_fraction, dead_abs_mv_per_km)
            for c in ("Ex", "Ey")}
    for r in rows:
        r["dead_std_Ex"], r["dead_std_Ey"] = round(dead["Ex"], 4), round(dead["Ey"], 4)
        if r["Ex_state"] == "gap":
            continue
        r["Ex_state"] = e_state(r["coh_Ex_Hy"], r["coh_Ex_Ey"], r["std_Ex"], dead["Ex"])
        r["Ey_state"] = e_state(r["coh_Ey_Hx"], r["coh_Ex_Ey"], r["std_Ey"], dead["Ey"])
    return pd.DataFrame(rows, columns=["day", "t_start", "t_end", "coh_Ex_Hy", "coh_Ey_Hx", "coh_Ex_Ey",
                                       "std_Ex", "std_Ey", "dead_std_Ex", "dead_std_Ey", "Ex_state", "Ey_state",
                                       "gap_reason"])


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


# ---------------------------------------------------------------- the signs these tests decide

SIGN_FLOOR = 0.3               # a ratio against IGRF this far from zero is a sign; below it the axis is dead
QUADRANT_BAND_S = (30.0, 1000.0)
QUADRANT_PERIODS = 6           # the periods scored inside that band
QUADRANT_MIN_IN = 4            # of which this many have to sit in the quadrant


def dc_flags(dc_row) -> list:
    """The flags of one dc_test row, `Bx negative` left out, as a list. An empty list is a sound sensor.

    `Bx negative` is the one flag the sign rule exists to answer, so it does not disqualify the row. Every
    other flag does: a gain or a broken axis (F away from IGRF), a tilt, or a sensor laid far from north all
    say the ratio being read is not the component the axis is supposed to carry.
    """
    raw = str(dc_row.get("flags", "") if hasattr(dc_row, "get") else "")
    if raw.strip().lower() in ("", "nan", "none", "<na>"):
        return []
    return [f.strip() for f in raw.split(";") if f.strip() and not f.strip().lower().startswith("bx negative")]


def signs_from_dc(dc_row, floor: float = SIGN_FLOOR) -> dict:
    """{'Hx': (sign or None, reason), 'Hz': (...)} from one dc_test row, by the floor rule.

    A magnetic axis laid the right way up reads the IGRF component it points at, so the sign of the ratio of
    the record median to IGRF is the sign of the channel: Bx/X for the north axis, Bz/Z for the vertical.
    Below `floor` = 0.3 the ratio says nothing -- a dead coil sits near zero and so does a sensor laid at
    right angles to north -- and the sign stays undecided rather than being read off noise.

    A row carrying any flag but `Bx negative` reads no sign at all (`dc_flags`). A magnetometer whose F is
    16 per cent off IGRF, or whose axes are laid 40 deg from north, is not measuring the component the
    ratio is being read as, and a polarity taken off it is a polarity taken off the wrong number. Queensland
    Phase 1's Q65 is that site, and the campaign withheld its sign by hand for the same reason.

    Hy is not decided here. The east component of the field is small and its ratio against IGRF is near zero
    at an Australian site whatever the coil does, so By/Y is not a test; Hy is decided against the
    observatory's east channel and two neighbours, which is a different measurement.
    """
    suspect = dc_flags(dc_row)
    if suspect:
        why = ("the DC test flags %s, so the ratio against IGRF is not this channel's polarity"
               % "; ".join(suspect))
        return {ch: (None, why) for ch in ("Hx", "Hz")}
    out = {}
    for ch, key, against in (("Hx", "Bx_over_X", "IGRF X"), ("Hz", "Bz_over_Z", "IGRF Z")):
        try:
            v = float(dc_row[key])
        except (TypeError, ValueError, KeyError, IndexError):
            v = float("nan")
        if not np.isfinite(v):
            out[ch] = (None, "%s carries no ratio against %s" % (ch, against))
        elif abs(v) < floor:
            out[ch] = (None, "%s/%s = %+.3f is inside the %.1f floor, so the axis says nothing about its "
                             "sign" % (ch, against, v, floor))
        else:
            out[ch] = (1 if v > 0 else -1,
                       "%s/%s = %+.3f, outside the %.1f floor" % (ch, against, v, floor))
    return out


def quadrant_sign(period, phase_deg, line: str = "", band_s=QUADRANT_BAND_S,
                  n_periods: int = QUADRANT_PERIODS, min_in: int = QUADRANT_MIN_IN) -> tuple:
    """(+1, -1 or None, one line saying what was counted) for one electric line, by Ben's quadrant rule.

    `phase_deg` is the phase as `transfer_functions.rho_phase` serves it, which is the xy phase as it stands
    and the yx phase already folded by +180 deg. Over a conductive half space with a sound H both sit in the
    first quadrant; reversing a line's electrode pair turns its row of the tensor by 180 deg and puts the
    phase in the third. So the rule reads:

        0 to 90 deg          the line is the right way round     +1
        -180 to -90 deg      the line is reversed                -1
        anything else        neither quadrant, and no sign is read

    The periods scored are `n_periods` = 6 spread over `band_s` = 30-1000 s, where a long-period record has
    its best signal-to-noise and where the phase is least disturbed by the short-period noise a reference
    cannot cancel; the sign is written only where `min_in` = 4 of the 6 agree. Fewer than 6 finite periods
    in the band leaves the sign undecided, which is the one honest answer: an E line's sign is never read by
    comparing E fields between sites (Ben's rule).
    """
    p = np.asarray(period, float)
    ph = np.asarray(phase_deg, float)
    inside = np.isfinite(p) & np.isfinite(ph) & (p >= band_s[0]) & (p <= band_s[1])
    idx = np.flatnonzero(inside)
    label = "%s over %g-%g s" % (line or "the line", band_s[0], band_s[1])
    if len(idx) < n_periods:
        return None, ("%s: %d finite period(s) in the band, fewer than the %d the rule scores"
                      % (label, len(idx), n_periods))
    pick = idx[np.round(np.linspace(0, len(idx) - 1, n_periods)).astype(int)]
    vals = ph[pick]
    first = int(np.sum((vals >= 0.0) & (vals <= 90.0)))
    third = int(np.sum((vals >= -180.0) & (vals <= -90.0)))
    detail = ("%s: %d of %d period(s) in 0-90 deg and %d in -180 to -90 deg (phase %s deg at %s s)"
              % (label, first, n_periods, third,
                 " ".join("%+.0f" % v for v in vals), " ".join("%.0f" % v for v in p[pick])))
    if first >= min_in:
        return 1, "%s -> +1" % detail
    if third >= min_in:
        return -1, "%s -> -1" % detail
    return None, "%s: neither quadrant holds %d of %d, so no sign is read" % (detail, min_in, n_periods)
