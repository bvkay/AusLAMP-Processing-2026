"""Cache variants: the 1.000 Hz tone notched out, and a spike screen, each written beside the original.

A cache is never edited in place. Each variant is a full npz under <work_root>/cache_<rate>hz_<name>/ with a
JSON sidecar naming what was done to which channel, and the sha256 of every channel so that a channel the
variant did not touch can be shown byte-identical to its source. That identity is what makes the
product-level control meaningful: a pass on the variant differs from a pass on the original only in the
channels that fired.

THE NOTCH (ported from vic_notch.py, constants :45-56). The tone is intermittent, so the decision is taken
PER WORST DAY and not on a whole-record statistic: over seventeen Victoria sites the whole-record ratio
exceeded ten at seven and the worst day at all seventeen. It is taken PER CHANNEL, because the tone sits in
Hx at most sites and in Hy at some. The statistic is the power in the bin at f0 over the median power of its
sidebands at 0.90-0.98 and 1.02-1.10 of f0, on the longest contiguous finite run of each UTC day carrying at
least four Welch windows. The filter is scipy's iirnotch at Q = 100 applied with filtfilt, so it is zero
phase and linear time invariant and cannot change the record outside the two notch bands; it is applied
GAP-AWARE, the record cut at every hole of 60 s or more and each piece filtered with 600 s of padding,
because one filtfilt over an interpolated 17.6 h hole moved a record by 2.15 per cent of its rms.

THE SPIKE SCREEN is new work and not a port: vic_windows.despike (:95-110) is a diagnostic used before a
coherence estimate and was never a processing step. Per channel, the samples i-2 to i+3 around any
first-difference step beyond k x 1.4826 x the median absolute deviation of the steps are blanked and LEFT
NaN -- a cache never carries interpolation -- and the count is written per channel and per UTC day. Its own
control is the count on the quietest day by H variance against the noisiest: a screen that fires as often on
a quiet day as on a loud one is measuring the estimator's own threshold and not the record.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")

# vic_notch.py:48-56
FS = 10.0
FREQS = (1.0, 2.0)
Q = 100.0
NPER = 1 << 16                  # 6553.6 s at 10 Hz
MIN_DAY_SAMPLES = 4 * NPER      # 7.3 h of contiguous finite data before a day may vote
RATIO_FIRE = 10.0
GAP_SPLIT_S = 60.0
MIN_SEG_S = 600.0
PAD_S = 600.0

SPIKE_K = 30.0
SPIKE_BEFORE, SPIKE_AFTER = 2, 3        # samples i-2 to i+3 inclusive


def sha256_array(x) -> str:
    """The sha256 of one channel's bytes, as stored."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def runs_of(mask):
    """(start, stop) of every maximal True run. vic_notch.runs_of (:67-71)."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def tone_ratio(x, f0, fs=FS, nper=NPER) -> float:
    """Power in the bin at f0 over the median power of its sidebands. vic_notch.tone_ratio (:74-86)."""
    from scipy import signal
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < nper * 2:
        return np.nan
    f, P = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear")
    i = int(np.argmin(np.abs(f - f0)))
    side = ((f >= f0 * 0.90) & (f <= f0 * 0.98)) | ((f >= f0 * 1.02) & (f <= f0 * 1.10))
    if side.sum() == 0 or P[i] <= 0:
        return np.nan
    med = float(np.median(P[side]))
    return float(P[i] / med) if med > 0 else np.nan


def day_bounds(t0, n, fs=FS):
    """(index0, index1, label) of every UTC calendar day the record touches. vic_notch.day_bounds."""
    t_start = datetime.fromtimestamp(float(t0), tz=timezone.utc)
    d0 = t_start.replace(hour=0, minute=0, second=0, microsecond=0)
    out, k = [], 0
    while True:
        a = d0 + timedelta(days=k)
        b = a + timedelta(days=1)
        i0 = int(round((a.timestamp() - float(t0)) * fs))
        i1 = int(round((b.timestamp() - float(t0)) * fs))
        if i0 >= n:
            break
        out.append((max(0, i0), min(n, i1), a.strftime("%Y-%m-%d")))
        k += 1
    return out


def cache_path(work_root, site, rate=10, variant="") -> Path:
    name = "cache_%dhz%s" % (int(rate), ("_" + variant) if variant else "")
    return Path(work_root) / name / ("%s.npz" % site)


def notch_census(sv, site, rate=10, freqs=FREQS, ratio_fire=RATIO_FIRE) -> tuple:
    """(the per-day table, the decision table) of the tone at one site.

    One row per channel and UTC day with the ratio at each frequency, taken on the longest contiguous finite
    run of the day; one decision row per channel and frequency naming the worst day, its ratio and whether
    the notch fires (vic_notch.scan_site :107-136, decide :139-155).
    """
    p = cache_path(sv.cfg["work_root"], site, rate)
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame()
    fs = float(rate)
    z = np.load(p, allow_pickle=False)
    t0 = float(np.asarray(z["t0"]).ravel()[0])
    rows = []
    for ch in CHANNELS:
        if ch not in z.files:
            continue
        x = np.asarray(z[ch], np.float64)
        n = x.size
        fin = np.isfinite(x)
        for i0, i1, lab in day_bounds(t0, n, fs):
            rr = runs_of(fin[i0:i1])
            if not rr:
                continue
            a, b = max(rr, key=lambda t: t[1] - t[0])
            if b - a < MIN_DAY_SAMPLES:
                continue
            y = x[i0 + a:i0 + b]
            y = y - np.mean(y)
            row = dict(site=site, channel=ch, day=lab, n_samples=int(b - a))
            for f0 in freqs:
                row["ratio_%gHz" % f0] = tone_ratio(y, f0, fs)
            rows.append(row)
        del x, fin
    z.close()
    scan = pd.DataFrame(rows)
    out = []
    if len(scan):
        for (s, ch), g in scan.groupby(["site", "channel"], sort=False):
            for f0 in freqs:
                v = g["ratio_%gHz" % f0].to_numpy(float)
                ok = np.isfinite(v)
                if not ok.any():
                    out.append(dict(site=s, channel=ch, f0=f0, worst_day="", worst_ratio=np.nan,
                                    median_ratio=np.nan, n_days=0, fire=False))
                    continue
                j = int(np.nanargmax(v))
                out.append(dict(site=s, channel=ch, f0=float(f0),
                                worst_day=str(g["day"].to_numpy()[j]), worst_ratio=float(v[j]),
                                median_ratio=float(np.nanmedian(v[ok])), n_days=int(ok.sum()),
                                fire=bool(v[j] > float(ratio_fire))))
    return scan, pd.DataFrame(out)


def notch_gap_aware(x, freqs, fs=FS, q=Q):
    """(the filtered series, statistics). vic_notch.notch_gap_aware (:158-201).

    The record is cut at every hole of GAP_SPLIT_S or more; each piece shorter than MIN_SEG_S is passed
    through untouched and counted; holes shorter than the split are bridged for the filter and re-blanked.
    """
    from scipy import signal
    x = np.asarray(x, np.float64)
    n = x.size
    fin = np.isfinite(x)
    y = x.copy()
    gap_split = int(round(GAP_SPLIT_S * fs))
    min_seg = int(round(MIN_SEG_S * fs))
    pad = int(round(PAD_S * fs))
    sos = [signal.iirnotch(f0, q, fs) for f0 in freqs]
    holes = [(a, b) for a, b in runs_of(~fin) if b - a >= gap_split]
    cuts = [0]
    for a, b in holes:
        cuts += [a, b]
    cuts.append(n)
    pieces = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts) - 1, 2)] if holes else [(0, n)]
    n_filt = n_skip = 0
    for a, b in pieces:
        if b - a <= 0:
            continue
        sub = x[a:b].copy()
        good = np.isfinite(sub)
        if not good.any() or b - a < min_seg:
            n_skip += b - a
            continue
        if not good.all():
            idx = np.flatnonzero(good)
            sub[~good] = np.interp(np.flatnonzero(~good), idx, sub[idx])
        m = float(np.mean(sub))
        sub = sub - m
        pl = int(min(pad, (b - a) // 3))
        for bb, aa in sos:
            sub = signal.filtfilt(bb, aa, sub, padlen=pl)
        sub = sub + m
        sub[~good] = np.nan
        y[a:b] = sub
        n_filt += b - a
    return y, dict(n_pieces=len(pieces), n_long_gaps=len(holes), samples_filtered=int(n_filt),
                   samples_passed_through=int(n_skip))


def notch_variant(sv, site, decision: pd.DataFrame, rate=10, variant="notched", force=False) -> dict:
    """Write cache_<rate>hz_<variant>/<site>.npz with only the channels that fired filtered.

    A channel that does not fire is copied through unchanged and its sha256 is compared with the source's,
    which is the sidecar's `identical` column. Returns the sidecar dict.
    """
    src = cache_path(sv.cfg["work_root"], site, rate)
    dst = cache_path(sv.cfg["work_root"], site, rate, variant)
    if not src.exists():
        return dict(site=site, status="no cache at %s" % src)
    if dst.exists() and not force:
        side = dst.with_suffix(".json")
        if side.exists():
            return json.loads(side.read_text(encoding="utf-8"))
    fire = {}
    if decision is not None and len(decision):
        d = decision[(decision.site == site) & decision.fire]
        for ch, g in d.groupby("channel"):
            fire[str(ch)] = sorted(float(f) for f in g.f0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(src, allow_pickle=False)
    dat = {k: z[k] for k in z.files}
    z.close()
    before = {ch: sha256_array(dat[ch]) for ch in CHANNELS if ch in dat}
    controls = []
    for ch in CHANNELS:
        if ch not in dat or ch not in fire:
            continue
        x = np.asarray(dat[ch], np.float64)
        y, st = notch_gap_aware(x, fire[ch], float(rate))
        fin = np.isfinite(x)
        rms = float(np.sqrt(np.mean(x[fin] ** 2))) if fin.any() else np.nan
        drms = float(np.sqrt(np.mean((y[fin] - x[fin]) ** 2))) if fin.any() else np.nan
        controls.append(dict(channel=ch, freqs=" ".join("%.3f" % f for f in fire[ch]),
                             tone_before_1Hz=tone_ratio(x, 1.0, float(rate)),
                             tone_after_1Hz=tone_ratio(y, 1.0, float(rate)),
                             tone_after_2Hz=tone_ratio(y, 2.0, float(rate)),
                             rms_change_frac=(drms / rms) if rms else np.nan, **st))
        dat[ch] = y.astype(np.asarray(dat[ch]).dtype)
        del x, y
    note = ("notch Q=%g, gap-aware (split %g s, min piece %g s, pad %g s), per worst day above %g: %s"
            % (Q, GAP_SPLIT_S, MIN_SEG_S, PAD_S, RATIO_FIRE,
               "; ".join("%s at %s Hz" % (c, "/".join("%.3f" % f for f in v))
                         for c, v in sorted(fire.items())) or "nothing fired"))
    dat["notch"] = np.array([note])
    np.savez(dst, **dat)
    after = {ch: sha256_array(dat[ch]) for ch in CHANNELS if ch in dat}
    del dat
    side = dict(site=site, variant=variant, rate_hz=float(rate), source=str(src), path=str(dst),
                built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                fired=sorted(fire), note=note, controls=controls,
                sha256_source={k: v for k, v in sorted(before.items())},
                sha256_variant={k: v for k, v in sorted(after.items())},
                identical={ch: bool(before[ch] == after[ch]) for ch in sorted(before)},
                untouched_identical=all(before[ch] == after[ch] for ch in before if ch not in fire),
                fired_changed=all(before[ch] != after[ch] for ch in fire if ch in before))
    dst.with_suffix(".json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    return side


def spike_counts(x, k=SPIKE_K):
    """(the blanking mask, the robust scale of the steps). The samples i-2 to i+3 around each hit."""
    x = np.asarray(x, float)
    bad = np.zeros(len(x), bool)
    d = np.diff(x)
    good = np.isfinite(d)
    if good.sum() < 10:
        return bad, np.nan
    mad = float(np.median(np.abs(d[good] - np.median(d[good]))) * 1.4826)
    if not np.isfinite(mad) or mad <= 0:
        return bad, mad
    for i in np.flatnonzero(good & (np.abs(d) > float(k) * mad + 1e-9)):
        bad[max(0, i - SPIKE_BEFORE):i + SPIKE_AFTER + 1] = True
    return bad, mad


def spike_variant(sv, site, k=SPIKE_K, rate=10, variant="despiked", force=False) -> dict:
    """Write cache_<rate>hz_<variant>/<site>.npz with the samples around each first-difference step blanked.

    The blanked samples are left NaN: a cache never carries interpolation, and the estimator's own mask is
    what decides what a NaN costs. The count per channel and per UTC day is written into the sidecar, with
    the quietest and the noisiest day by H variance named as the control pair.
    """
    src = cache_path(sv.cfg["work_root"], site, rate)
    dst = cache_path(sv.cfg["work_root"], site, rate, variant)
    if not src.exists():
        return dict(site=site, status="no cache at %s" % src)
    if dst.exists() and not force:
        side = dst.with_suffix(".json")
        if side.exists():
            return json.loads(side.read_text(encoding="utf-8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fs = float(rate)
    z = np.load(src, allow_pickle=False)
    dat = {kk: z[kk] for kk in z.files}
    z.close()
    t0 = float(np.asarray(dat["t0"]).ravel()[0])
    before = {ch: sha256_array(dat[ch]) for ch in CHANNELS if ch in dat}
    bounds = day_bounds(t0, len(np.asarray(dat["Hx"])), fs)
    per_day, per_channel = [], []
    hvar = {}
    for i0, i1, lab in bounds:
        seg = np.asarray(dat["Hx"], np.float64)[i0:i1]
        fin = np.isfinite(seg)
        hvar[lab] = float(np.var(seg[fin])) if fin.sum() > 100 else np.nan
    for ch in CHANNELS:
        if ch not in dat:
            continue
        x = np.asarray(dat[ch], np.float64)
        bad, mad = spike_counts(x, k)
        for i0, i1, lab in bounds:
            per_day.append(dict(channel=ch, day=lab, blanked=int(bad[i0:i1].sum()),
                                samples=int(i1 - i0)))
        per_channel.append(dict(channel=ch, blanked=int(bad.sum()), samples=int(len(x)),
                                fraction=float(bad.mean()), step_scale=mad,
                                threshold=(float(k) * mad if np.isfinite(mad) else np.nan)))
        x[bad] = np.nan
        dat[ch] = x.astype(np.asarray(dat[ch]).dtype)
        del x, bad
    days = pd.DataFrame(per_day)
    quiet = min((d for d in hvar if np.isfinite(hvar[d])), key=lambda d: hvar[d], default="")
    loud = max((d for d in hvar if np.isfinite(hvar[d])), key=lambda d: hvar[d], default="")
    note = ("spike screen: samples i-%d to i+%d around any first difference beyond %g x 1.4826 MAD of the "
            "steps, left NaN" % (SPIKE_BEFORE, SPIKE_AFTER, float(k)))
    dat["despike"] = np.array([note])
    np.savez(dst, **dat)
    after = {ch: sha256_array(dat[ch]) for ch in CHANNELS if ch in dat}
    del dat

    def _day_total(day):
        if not day or not len(days):
            return None
        return int(days[days.day == day].blanked.sum())

    side = dict(site=site, variant=variant, rate_hz=fs, k=float(k), source=str(src), path=str(dst),
                built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), note=note,
                per_channel=per_channel, per_day=days.to_dict("records"),
                control=dict(quietest_day=quiet, quietest_h_variance=hvar.get(quiet),
                             quietest_blanked=_day_total(quiet),
                             noisiest_day=loud, noisiest_h_variance=hvar.get(loud),
                             noisiest_blanked=_day_total(loud),
                             rule="the screen must fire more on the noisiest day than on the quietest"),
                sha256_source={k2: v for k2, v in sorted(before.items())},
                sha256_variant={k2: v for k2, v in sorted(after.items())},
                identical={ch: bool(before[ch] == after[ch]) for ch in sorted(before)})
    dst.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str), encoding="utf-8")
    return side
