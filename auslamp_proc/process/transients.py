"""The transient tail scan, the clean pool test, and the keep mask a pass is run under.

The scan (ported from scripts/qc/qld_donor_tails.py, which reads 0.1-1 Hz on a native 10 Hz record) walks a
site's rotated record in `chunk_s` = 600 s chunks and writes the per-chunk median band power of Hx, Hy, Ex
and Ey to <work_root>/tails/<site>.npz (t0, chunk_s, one array per channel).

The band on a 1 Hz cache is 0.005-0.05 Hz, that is 20-200 s. A cache decimated from 10 Hz has its 2-20 s
band sitting on the anti-alias filter's roll-off, where the statistic is each unit's own noise floor rather
than the field: measured over AusLAMP Queensland Phase 1 on 2026-09-16 the 2-20 s chunk medians span five
decades across 23 fluxgates, where the 20-200 s medians hold inside a factor of 10 at 22 of the 23. A
baseline that is not comparable between sites cannot be judged against a survey median, so the scan is read
in the band the fleet agrees in, which is also the band the remote-site rule scores coherence in.

The event test (ported from qld_transients.py) is fleet-normalised. A chunk is an event when the site's own
band power exceeds `own_ratio` = 100 times its own median on Hx or Hy and the rest of the fleet does not see
the same thing: either fewer than `min_fleet` = 3 other sites were recording then, or this site runs more
than `fleet_excess` = 10 times the median of the others. A substorm lifts every site in the array within the
same ten minutes and is signal; a power-cycle or a vehicle lifts one. The site's own median cannot tell them
apart and the fleet can.

The test needs a fleet to normalise against, so where the set it is given holds fewer than `min_fleet` = 3
other sites with a scan, it is OFF and no chunk is flagged as an event: the site's own threshold alone
cannot tell a substorm from a power-cycle, and applying it alone flags every large chunk of every site and
empties the pool. `fleet_normalised` answers whether it ran, `clean_row` reports it in `note`, and the
baseline limb is scored as usual, being a comparison between sites and not within one. The per-chunk
`min_fleet` term is separate and still applies: inside a survey that has a fleet, an hour when fewer than
three others were recording carries no corroboration either.

The clean test is two limbs and neither implies the other: event fraction <= 0.02 of
the 600 s chunks, and baseline <= 10 times the survey median, on both H channels, judged on the rotated
record. The first catches a site that is fine except for a power-cycle; the second catches a site that never
spikes because it is loud all the time. The baseline denominator is one median over every site with a scan,
so the reference level is a property of the instrument and the band and not of whoever was deployed alongside.

Every threshold is a survey.yaml `pool` value, read by params().

The keep mask of a pass is the finite samples and the H transient intervals of the target and, for the
remote-site kind, those of the remote and the E burst intervals (wamt_run.build_keep :598-647).
The record is never cut: each kept stretch of at least MIN_SEGMENT_S = 3600 s becomes its own Aurora run.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHUNK_S = 600.0
NPERSEG = 512
BAND_HZ = (0.005, 0.05)               # 20-200 s at 1 Hz
OWN_RATIO = 100.0
FLEET_EXCESS = 10.0
MIN_FLEET = 3
EVENT_FRAC_MAX = 0.02
BASELINE_MAX = 10.0
E_BURST_RATIO = 100.0
MIN_SEGMENT_S = 3600.0
SCAN_CHANNELS = ("Hx", "Hy", "Ex", "Ey")
H = ("Hx", "Hy")
E = ("Ex", "Ey")


def params(cfg: dict) -> dict:
    """The pool thresholds from survey.yaml `pool`, each falling back to the value stated in this module."""
    p = (cfg or {}).get("pool") or {}
    return dict(chunk_s=float(p.get("chunk_s", CHUNK_S)),
                nperseg=int(p.get("nperseg", NPERSEG)),
                band_hz=tuple(p.get("band_hz", BAND_HZ)),
                own_ratio=float(p.get("own_ratio", OWN_RATIO)),
                fleet_excess=float(p.get("fleet_excess", FLEET_EXCESS)),
                min_fleet=int(p.get("min_fleet", MIN_FLEET)),
                event_frac_max=float(p.get("event_frac_max", EVENT_FRAC_MAX)),
                baseline_max=float(p.get("baseline_max", BASELINE_MAX)),
                e_burst_ratio=float(p.get("e_burst_ratio", E_BURST_RATIO)),
                min_segment_s=float(p.get("min_segment_s", MIN_SEGMENT_S)))


def tails_dir(work_root) -> Path:
    return Path(work_root) / "tails"


# ------------------------------------------------------------------ the scan

def chunk_power(x, fs: float, chunk_s: float, nperseg: int, band_hz) -> np.ndarray:
    """Per-chunk median band power of one channel; NaN where the chunk is not whole and finite."""
    from scipy.signal import welch
    x = np.asarray(x, float)
    step = int(round(chunk_s * fs))
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    out = []
    for i in range(0, len(x) - step, step):
        seg = x[i:i + step]
        if not np.all(np.isfinite(seg)):
            out.append(np.nan)
            continue
        out.append(float(np.median(welch(seg - seg.mean(), fs, nperseg=nperseg)[1][m])))
    return np.asarray(out, float)


def scan(site: str, t0: int, arrays: dict, work_root, fs: float = 1.0, cfg=None, force=False) -> dict:
    """Write <work_root>/tails/<site>.npz from a rotated record, and return its summary row.

    `arrays` carries the signed, rotated channels. With `force` False a scan already on disk is summarised
    from the file rather than recomputed.
    """
    p = params(cfg or {})
    fn = tails_dir(work_root) / ("%s.npz" % site)
    if fn.exists() and not force:
        return summarise(site, *load_series(site, work_root))
    pw = {c: chunk_power(arrays[c], fs, p["chunk_s"], p["nperseg"], p["band_hz"])
          for c in SCAN_CHANNELS if c in arrays}
    fn.parent.mkdir(parents=True, exist_ok=True)
    tmp = fn.with_suffix(".tmp.npz")
    np.savez(tmp, t0=np.array([int(t0)], np.int64), chunk_s=np.array([float(p["chunk_s"])]),
             band_hz=np.array(list(p["band_hz"]), float), built_utc=np.array([_now()]), **pw)
    tmp.replace(fn)
    return summarise(site, float(t0), float(p["chunk_s"]), pw)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_series(site: str, work_root):
    """(t0, chunk_s, {channel: array}) from <work_root>/tails/<site>.npz, or None where there is none."""
    fn = tails_dir(work_root) / ("%s.npz" % site)
    if not fn.exists():
        return None
    z = np.load(fn, allow_pickle=False)
    got = (float(np.atleast_1d(z["t0"])[0]), float(np.atleast_1d(z["chunk_s"])[0]),
           {c: np.asarray(z[c], float) for c in SCAN_CHANNELS if c in z.files})
    z.close()
    return got


def summarise(site: str, t0: float, chunk_s: float, pw: dict) -> dict:
    """One row per site of what the scan holds: the medians, the worst excursion and the chunk counts."""
    row = dict(site=site, chunks=int(len(pw["Hx"])), days=round(len(pw["Hx"]) * chunk_s / 86400.0, 2))
    for c in SCAN_CHANNELS:
        if c not in pw:
            continue
        x = np.asarray(pw[c], float)
        med = float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")
        r = x / med if np.isfinite(med) and med > 0 else np.full(len(x), np.nan)
        row["%s_median" % c] = med
        row["%s_max_ratio" % c] = float(np.nanmax(r)) if np.isfinite(r).any() else float("nan")
        row["%s_nan_frac" % c] = round(float(np.mean(~np.isfinite(x))), 4)
    return row


# ----------------------------------------------------------- the fleet and the events

def _slot(delta, chunk_s) -> int:
    """Chunk index `delta` seconds after a grid origin; floor, snapping only on an integer number of chunks.

    Rounding would hide a scan that started off the grid: a site half a chunk out would land on whichever
    neighbour is closer and nothing would say so.
    """
    q = float(delta) / float(chunk_s)
    r = int(np.round(q))
    return r if abs(q - r) <= 1e-6 else int(np.floor(q))


def fleet_series(sites, work_root, channels=H):
    """(t0, chunk_s, {channel: median}, n_recording) over a common grid, or None.

    The median is of log10 power: band power spans ten decades across a fleet and a linear median is dragged
    by whichever site is loudest.
    """
    got = [(s,) + load_series(s, work_root) for s in sites if load_series(s, work_root) is not None]
    if not got:
        return None
    chunk_s = got[0][2]
    t_lo = min(g[1] for g in got)
    t_hi = max(g[1] + len(g[3][channels[0]]) * g[2] for g in got)
    ng = int(round((t_hi - t_lo) / chunk_s))
    grids = {c: np.full((len(got), ng), np.nan) for c in channels}
    for k, (_s, t0, cs, pw) in enumerate(got):
        i0 = _slot(t0 - t_lo, cs)
        for c in channels:
            x = np.asarray(pw[c], float)
            j0, j1 = max(0, i0), min(ng, i0 + len(x))
            if j1 > j0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    grids[c][k, j0:j1] = np.log10(x[j0 - i0:j1 - i0])
    with np.errstate(invalid="ignore"):
        med = {c: np.nanmedian(grids[c], axis=0) for c in channels}
    n_rec = np.sum(np.isfinite(grids[channels[0]]), axis=0)
    return t_lo, chunk_s, med, n_rec.astype(int)


def fleet_normalised(site: str, fleet_sites, work_root, cfg=None) -> tuple:
    """(True where the event test can be normalised against a fleet, the others with a scan).

    The test asks whether the rest of the array saw what this site saw, so it needs `min_fleet` others with
    a scan to ask. A run scoped to a handful of sites has none, and the test is off rather than falling back
    on the site's own threshold, which flags every large chunk of every site.
    """
    p = params(cfg or {})
    others = [s for s in fleet_sites if s != site and load_series(s, work_root) is not None]
    return len(others) >= int(p["min_fleet"]), others


def flag_chunks(site: str, fleet_sites, work_root, cfg=None, keep_chunks=None):
    """(t0, chunk_s, bad) -- the per-chunk event flags of one site, before any merging.

    `keep_chunks` is an optional boolean over the site's own chunks; chunks it excludes are never flagged and
    are not counted, which is how a rotation drop is kept out of the site's own score.

    Where the set holds fewer than `min_fleet` other sites with a scan, the test is off and nothing is
    flagged (`fleet_normalised`).
    """
    p = params(cfg or {})
    r = load_series(site, work_root)
    if r is None:
        return None
    t0, chunk_s, pw = r
    on, others = fleet_normalised(site, fleet_sites, work_root, cfg)
    n = len(pw["Hx"])
    bad = np.zeros(n, bool)
    if not on:
        return t0, chunk_s, bad
    fl = fleet_series(others, work_root)
    if fl is None:                               # every other site's scan went missing between the two reads
        return t0, chunk_s, bad
    for c in H:
        x = np.asarray(pw[c], float)
        med = np.nanmedian(x)
        if not np.isfinite(med) or med <= 0:
            continue
        own = np.isfinite(x) & (x > p["own_ratio"] * med)
        if not own.any():
            continue
        gt0, gcs, gmed, gn = fl
        if abs(chunk_s - gcs) <= 1e-9:
            idx = _slot(t0 - gt0, gcs) + np.arange(n)
        else:
            idx = np.array([_slot(t0 + i * chunk_s - gt0, gcs) for i in range(n)])
        ok = (idx >= 0) & (idx < len(gmed[c]))
        fmed = np.full(n, np.nan)
        nrec = np.zeros(n, int)
        fmed[ok] = gmed[c][idx[ok]]
        nrec[ok] = gn[idx[ok]]
        lonely = nrec < p["min_fleet"]
        with np.errstate(over="ignore", invalid="ignore"):
            excess = np.isfinite(fmed) & (x > p["fleet_excess"] * 10.0 ** fmed)
        bad |= own & (lonely | excess)
    if keep_chunks is not None:
        k = np.asarray(keep_chunks, bool)[:n]
        bad[:len(k)] &= k
        bad[len(k):] = False
    return t0, chunk_s, bad


def _intervals(t0, chunk_s, bad, merge_gap_s=0.0) -> list:
    """Flagged chunks as [(start, end)] in unix seconds; adjacent chunks always merge."""
    hits = np.flatnonzero(np.asarray(bad, bool))
    if not len(hits):
        return []
    out, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        gap = (i - prev - 1) * chunk_s
        if gap > 0 and gap >= merge_gap_s:
            out.append((t0 + start * chunk_s, t0 + (prev + 1) * chunk_s))
            start = i
        prev = i
    out.append((t0 + start * chunk_s, t0 + (prev + 1) * chunk_s))
    return [(float(a), float(b)) for a, b in out]


def site_events(site: str, fleet_sites, work_root, cfg=None, merge_gap_s=0.0, keep_chunks=None) -> list:
    """The merged intervals of unix seconds this site is faulty over. The mask's input.

    merge_gap_s defaults to 0.0: only adjacent chunks join, so a quiet chunk between two bursts is kept.
    Bridging an hour welds scattered ten-minute faults into blocks and costs the record between them.
    """
    r = flag_chunks(site, fleet_sites, work_root, cfg, keep_chunks)
    return [] if r is None else _intervals(r[0], r[1], r[2], merge_gap_s)


def e_events(site: str, work_root, cfg=None) -> list:
    """The E burst intervals: chunks where either electric line exceeds e_burst_ratio times its own median.

    No fleet test: an electric burst is local to the site's own electrodes by definition, where a magnetic
    one may be the source field lighting up the whole array.
    """
    p = params(cfg or {})
    r = load_series(site, work_root)
    if r is None:
        return []
    t0, chunk_s, pw = r
    bad = np.zeros(len(pw["Hx"]), bool)
    for c in E:
        if c not in pw:
            continue
        x = np.asarray(pw[c], float)
        med = np.nanmedian(x)
        if not np.isfinite(med) or med <= 0:
            continue
        bad |= np.isfinite(x) & (x > p["e_burst_ratio"] * med)
    return _intervals(t0, chunk_s, bad)


def event_fraction(site: str, fleet_sites, work_root, cfg=None, keep_chunks=None):
    """Fraction of the site's chunks flagged, unmerged, or None where there is no scan.

    Unmerged on purpose: this is the class decision, and merging would charge the site for the quiet hour
    between two spikes, which is not time it was faulty.
    """
    r = flag_chunks(site, fleet_sites, work_root, cfg, keep_chunks)
    if r is None or len(r[2]) == 0:
        return None
    bad = np.asarray(r[2], bool)
    if keep_chunks is not None:
        k = np.asarray(keep_chunks, bool)[:len(bad)]
        return float(bad[:len(k)].mean()) if k.any() else None
    return float(bad.mean())


def site_baseline(site: str, work_root, keep_chunks=None):
    """max(median Hx, median Hy) of the site's band-power series, or None.

    The max and not the mean of the two: one dead coil makes a site useless as a reference and a mean lets
    the sound channel hide it.
    """
    r = load_series(site, work_root)
    if r is None:
        return None
    _t0, _cs, pw = r
    meds = []
    for c in H:
        x = np.asarray(pw[c], float)
        if keep_chunks is not None:
            k = np.asarray(keep_chunks, bool)[:len(x)]
            x = x[:len(k)][k]
        m = np.nanmedian(x) if x.size else np.nan
        if np.isfinite(m) and m > 0:
            meds.append(float(m))
    return max(meds) if meds else None


def survey_baseline(sites, work_root):
    """Median site_baseline over every site with a scan. One denominator for the whole survey.

    Normalising per group would let a noisy group grade itself: where most sites are loud the median is loud
    too and every one of them scores near 1x and passes as normal.
    """
    vals = [b for b in (site_baseline(s, work_root) for s in sites) if b is not None]
    return float(np.median(vals)) if vals else None


def baseline_ratio(site: str, sites, work_root, keep_chunks=None):
    """The site's baseline over the survey-wide reference, or None."""
    b = site_baseline(site, work_root, keep_chunks)
    ref = survey_baseline(sites, work_root)
    if b is None or not ref or not np.isfinite(ref) or ref <= 0:
        return None
    return float(b / ref)


def clean_row(site: str, sites, work_root, cfg=None, keep_chunks=None) -> dict:
    """The clean test of one site with both its numbers and the reason it failed, if it did.

    Where fewer than `min_fleet` other sites carry a scan the event limb is off and `note` says so; the
    site is then judged on its baseline alone, and `judged` asks whether it has a scan at all.
    """
    p = params(cfg or {})
    on, others = fleet_normalised(site, sites, work_root, cfg)
    f = event_fraction(site, sites, work_root, cfg, keep_chunks) if on else None
    b = baseline_ratio(site, sites, work_root, keep_chunks)
    scanned = load_series(site, work_root) is not None
    why, note = [], ""
    if on:
        if f is None:
            why.append("no chunk scored")
        elif f > p["event_frac_max"]:
            why.append("event fraction %.4f > %.2f" % (f, p["event_frac_max"]))
    else:
        note = ("the fleet-normalised event test is off: %d other site(s) carry a scan, below "
                "pool.min_fleet = %d, so this site is judged on its baseline alone"
                % (len(others), int(p["min_fleet"])))
        if not scanned:
            why.append("no scan")
    if b is None:
        why.append("no baseline")
    elif b > p["baseline_max"]:
        why.append("baseline %.1fx > %.0fx the survey median" % (b, p["baseline_max"]))
    return dict(site=site, event_frac=f, baseline=b, clean=not why, reason="; ".join(why),
                judged=(f is not None) if on else scanned, fleet_normalised=on, note=note)


# ----------------------------------------------------------------- the mask

def keep_mask(t0, n, fs, intervals) -> np.ndarray:
    """A boolean of length n, False inside any (start, end) interval of unix seconds."""
    keep = np.ones(int(n), bool)
    for a, b in intervals or []:
        i0 = max(0, int(np.floor((a - t0) * fs)))
        i1 = min(int(n), int(np.ceil((b - t0) * fs)))
        if i1 > i0:
            keep[i0:i1] = False
    return keep


def segments(keep, min_len=0) -> list:
    """[(offset, length)] of the contiguous True runs of at least min_len samples."""
    k = np.asarray(keep, bool)
    d = np.diff(np.concatenate(([0], k.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b - a)) for a, b in zip(st, en) if (b - a) >= min_len]


def floor_dropped_frac(keep, fs, min_segment_s=MIN_SEGMENT_S) -> float:
    """The fraction of the record the mask kept and the run floor then throws away.

    Aurora gets one run per kept stretch and a stretch under min_segment_s yields no window at the deep
    levels, so it is dropped. That is a second loss on top of the mask -- two events forty minutes apart cost
    the forty minutes between them as well as themselves -- and it is measured here so the ledger and the EDI
    can say how much rather than the mask pretending to bridge the gap.
    Ported from qld_campaign.floor_dropped_frac (:1322).
    """
    k = np.asarray(keep, bool)
    if not len(k) or not k.sum():
        return 0.0
    written = sum(L for _, L in segments(k, int(min_segment_s * fs)))
    return float((int(k.sum()) - written) / len(k))


def build_keep(t0, arrays: dict, fs, ev_site, ev_remote=(), ev_e=(), remote_mask=None, extra_mask=None,
               extra_name="") -> tuple:
    """(keep, statistics) for one pass.

    The mask is the finite samples of every local channel and the target's H transient intervals and, for
    the remote-site kind, the remote's own intervals and the E burst intervals. Where a reference carries a
    coverage mask, samples the reference does not cover are dropped as well: a stack is zero there and a
    zero reference contributes nothing but the window it sits in.

    The E burst intervals drop the sample, not just its E: one mask covers all five channels, and the H
    under an E burst is usually sound, so what this costs is variance over a few hours and what it buys is
    that a 100x electric transient is not in the regression.

    `extra_mask` is a selection of hours made outside this pass and named in the decisions table; every
    fraction it costs is reported on its own line so the two masks are never read as one.
    """
    n = len(arrays["Hx"])
    keep = np.ones(n, bool)
    for v in arrays.values():
        keep &= np.isfinite(np.asarray(v, float))
    finite = float(keep.mean())
    keep &= keep_mask(t0, n, fs, ev_site)
    keep &= keep_mask(t0, n, fs, ev_remote)
    keep &= keep_mask(t0, n, fs, ev_e)
    after_mask = float(keep.mean())
    if remote_mask is not None:
        keep &= np.asarray(remote_mask, bool)[:n]
    after_ref = float(keep.mean())
    if extra_mask is not None:
        keep &= np.asarray(extra_mask, bool)[:n]
    stats = dict(n=int(n), finite_frac=round(finite, 5), kept_frac=round(float(keep.mean()), 5),
                 mask_dropped_frac=round(float(1.0 - keep.mean()), 5),
                 transient_dropped_frac=round(float(finite - after_mask), 5),
                 reference_coverage_dropped_frac=round(float(after_mask - after_ref), 5),
                 selection=extra_name or "none: the whole record",
                 selection_dropped_frac=round(float(after_ref - keep.mean()), 5),
                 n_intervals_site=len(list(ev_site)), n_intervals_remote=len(list(ev_remote)),
                 n_intervals_e=len(list(ev_e)))
    return keep, stats
