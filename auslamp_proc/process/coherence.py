"""The coherence estimators the reference choices are made on.

band_coh_clean (ported from qld_campaign.py:332-406) measures a pair over its event-free segments with
nothing concatenated. Masked data is never joined before a coherence estimate: target and donor are cut at
identical instants, so every join puts the same step into both and the estimator reads that shared edge as
shared field. Measured on one Queensland pair over 58 days, a synthetic mask that removed nothing real took
the score from 0.268 to 0.831 at a 50 per cent cut. So the uncut overlap is walked in non-overlapping
`nperseg` segments; a segment is used only where `keep` is True right through it, its periodogram goes into
the running spectra of the chunk it belongs to, a chunk with at least `min_segments` clean segments yields
one coherence, and the score is the median over chunks.

The Welch bias is removed. Coherence from N segments sits high by about (1 - c)^2 / N, so a chunk with 24
clean segments outscores one with 84 on identical fields by 0.02 at c = 0.2, which is enough to reorder
candidates near a 0.3 gate.

`nperseg` is sized for 1 Hz -- 1024 s resolves the 200 s end of the 20-200 s band five bins up -- and is
scaled with fs by nperseg_for().

Two bands are used. The remote-site rule scores 20-200 s at nperseg 1024
and 24 segments a chunk. The stack weights score 100-1000 s at nperseg 4096 and 6 segments a chunk, and the
weight is the member's median coherence with the fleet, never with the target (Ben's rule, 2026-09-06).

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

PAIR_BAND = (1 / 200.0, 1 / 20.0)
PAIR_NPERSEG = 1024
PAIR_MIN_SEGMENTS = 24

WEIGHT_BAND = (1 / 1000.0, 1 / 100.0)
WEIGHT_NPERSEG = 4096
WEIGHT_MIN_SEGMENTS = 6

CHUNK_DAYS = 1.0
MIN_CHUNKS = 4
WEIGHT_RULE = "fleet"                 # the member's standing in the fleet, never its agreement with a target
H = ("Hx", "Hy")


def nperseg_for(nperseg_1hz: int, fs: float) -> int:
    """The segment length at `fs` that spans the same number of seconds as `nperseg_1hz` does at 1 Hz."""
    return int(round(int(nperseg_1hz) * float(fs)))


def band_coh_clean(x, y, fs, keep, band=PAIR_BAND, chunk_days=CHUNK_DAYS, min_chunks=MIN_CHUNKS,
                   nperseg=PAIR_NPERSEG, min_segments=PAIR_MIN_SEGMENTS, return_days=False):
    """Median over chunks of the band-mean squared coherence of the pair's event-free whole segments.

    `nperseg` is in samples at `fs`; pass nperseg_for(1024, fs) to hold the segment's length in seconds.
    Returns the median, or (median, chunks scored) with return_days True; NaN below `min_chunks` chunks.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    k = np.asarray(keep, bool)
    n = min(len(x), len(y), len(k))
    seg = int(nperseg)
    chunk = int(round(chunk_days * 86400 * fs))
    if n < seg or chunk < seg:
        return (np.nan, 0) if return_days else np.nan
    f = np.fft.rfftfreq(seg, 1.0 / fs)
    with np.errstate(divide="ignore"):
        p = np.where(f > 0, 1.0 / np.maximum(f, 1e-12), np.inf)
    m = (p >= 1 / band[1]) & (p <= 1 / band[0])
    if not m.any():
        return (np.nan, 0) if return_days else np.nan
    win = np.hanning(seg)
    acc: dict = {}                                        # chunk -> [Pxx, Pyy, Pxy, count]
    for i in range(0, n - seg + 1, seg):
        if not k[i:i + seg].all():
            continue
        a, b = x[i:i + seg], y[i:i + seg]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        A = np.fft.rfft((a - a.mean()) * win)[m]
        B = np.fft.rfft((b - b.mean()) * win)[m]
        d = acc.setdefault(i // chunk, [0.0, 0.0, 0.0j, 0])
        d[0] = d[0] + np.abs(A) ** 2
        d[1] = d[1] + np.abs(B) ** 2
        d[2] = d[2] + A * np.conj(B)
        d[3] += 1
    vals = []
    for pxx, pyy, pxy, cnt in acc.values():
        if cnt < min_segments:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.abs(pxy) ** 2 / (pxx * pyy)
        c = np.clip(c - (1.0 - c) ** 2 / cnt, 0.0, 1.0)   # the Welch bias, removed
        if np.isfinite(c).any():
            vals.append(float(np.nanmean(c)))
    if len(vals) < min_chunks:
        return (np.nan, 0) if return_days else np.nan
    med = float(np.nanmedian(vals))
    return (med, len(vals)) if return_days else med


def pair_coh(ha: dict, hb: dict, fs, keep, band=PAIR_BAND, nperseg=PAIR_NPERSEG,
             min_segments=PAIR_MIN_SEGMENTS, chunk_days=CHUNK_DAYS, min_chunks=MIN_CHUNKS) -> dict:
    """Mean over Hx and Hy of band_coh_clean on two H records already on one grid.

    `keep` is the union mask of both sites' event intervals, so a chunk either of them flags is dropped from
    both. Returns coh, the chunks scored, the chunk length and the fraction of the overlap the mask removed.
    """
    vals, chunks = [], []
    for c in H:
        v, d = band_coh_clean(ha[c], hb[c], fs, keep, band=band, nperseg=nperseg,
                              chunk_days=chunk_days, min_chunks=min_chunks,
                              min_segments=min_segments, return_days=True)
        vals.append(v)
        chunks.append(d)
    coh = float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
    return dict(coh=coh, chunks=int(min(chunks)), chunk_days=float(chunk_days),
                clean_days=round(float(min(chunks)) * float(chunk_days), 2),
                mask_frac=float(1.0 - np.asarray(keep, bool).mean()))


def fleet_weight_table(pairs: dict, pool, min_chunks=3) -> dict:
    """{site: median coherence with every other member of the pool}, from a {"a:b": {coh, chunks}} table.

    Ben's rule of 2026-09-06: a reference is a measurement of the regional field, and whether it is a good
    one is a question about the reference and the field, not about the target. The target's own pair is one
    of many in the median, so its influence is a fraction of one value.
    """
    out = {}
    for a in pool:
        vals = [v["coh"] for k, v in pairs.items()
                if a in k.split(":") and v.get("coh") is not None and np.isfinite(v["coh"])
                and int(v.get("chunks", 0)) >= min_chunks]
        out[a] = round(float(np.median(vals)), 4) if vals else None
    return out
