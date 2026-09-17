"""The five reference kinds and the store they are written to.

    single station        the site's own H and E; no reference is written
    remote site           one other site's H, whole, on the target's grid
    fleet stack           a coherence-weighted mean of several sites' H, aligned and demeaned
    observatory           an INTERMAGNET one-second record in its own mean-field frame
    stack + observatory   the stack with the observatory as one more member

What each is for. The remote site cancels the noise that is uncorrelated between the two sites, and costs the
variance of one more record. The stack averages down each member's own noise and needs the members aligned,
because delays that differ by 2-3 s partly cancel each other in the sum. The observatory is far and quiet and
reaches the long periods, and is never shifted: a delay on a single reference cancels exactly in Z.

The pool. A member is drawn from the clean pool (transients.clean_row) restricted to the target's overlap
group in <work_root>/survey/deployment_groups.csv and to the sites that have a cache. The remote for a site
is decisions.csv `remote_site` where that cell names a site, and otherwise the five-branch rule.

The five-branch remote-site rule (ported from wamt_remotes.partner :941-1029, itself the Queensland campaign's
rule), with the branch and the reason recorded in every product:

    1  clean, coh >= COH_MIN, overlap >= 90 % of the target's record -> the nearest of them
    2  clean, coh >= COH_MIN                                          -> the longest overlap
    3  clean, coh >= COH_RELAX                                        -> the most coherent
    4  coh >= COH_MIN, not clean                                      -> the fewest events
    5  none of the above                                              -> the nearest by km

A candidate is scored only where its usable overlap reaches overlap_floor = min(MIN_OVERLAP_DAYS,
0.75 x the days the target can use); the coherence is the event-free 20-200 s chunk median (coherence.pair_coh).

The stack (ported from wamt_remotes.stack_weights / accumulate / _finish :1188-1356). The weight is the
member's median coherence with the FLEET at 100-1000 s and never its coherence with the target
(Ben's rule, 2026-09-06): whether a reference is a good measurement of the regional field is a
question about the reference and the field. Members below STACK_CUTOFF = 0.5 are excluded, the best
STACK_MAX = 8 are kept, and a stack with fewer than STACK_MIN = 2 members is refused -- a one-member stack is
a remote site renamed (STACK_MIN_MEMBERS, qld_p1_remotes_v2.py:115). A sample where a member is NaN does not
add to that member's weight there, so the mean is over whoever is sound and the weights renormalise per
sample. Where no member is sound the stack is zero and the mask is False: a zero reference contributes
nothing to either the cross- or the auto-spectrum, so those windows drop out of the estimate instead of
biasing it.

A member passes four steps between its cache and the sum, in this order (Ben's ruling, 2026-09-17). Its
transient chunks are NaN; then EDGE_S = 120 s inside every record end and every gap end is NaN, because
the samples either side of a break carry the logger's settling, which runs at 37x the record's own rate of
steps above 2 nT for the first 60 s and under 2x by the second; then it is shifted by its own lag with the
Lanczos delay of align.shift, which is local and so does not spread a spike; then member_screen bridges the
spikes that are its own and not the fleet's; then level_match sets it on the fleet's own level over
LEVEL_WIN_S = 3600 s, which is what a per-record demeaning cannot do, because a member's own mean is
exactly the quantity that stepped the composite whenever the member set changed.

The stack and its members are measured on the same statistic, steps above STEP_NT = 2.0 nT per million
samples, and the sidecar carries the stack's rate beside both the raw and the screened member medians: a
stack rougher than the median of its own screened members has done the opposite of what averaging is for.

The store is <work_root>/references/<rate>hz/<kind>_<site>.npz (t0, Hx, Hy, mask, coverage, and for the two
stack kinds bridged_Hx and bridged_Hy, the spans the spike screen bridged) beside
<kind>_<site>.json (members, weights, lags, roles, refusals, frame, built_at), with pool.json,
fleet_weights.json, fleet_pairs.json and candidates_<site>.json for the tables the workbook prints. A 10 Hz
store re-reads the 1 Hz specification -- the same pool, the same remote, the same members, weights and lags --
on the 10 Hz grid after the gap-edge screen.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geo, observatory
from ..raw import cache
from . import align, coherence as COH, frame as FR, transients as TR

COH_MIN = 0.5
COH_RELAX = 0.3
MIN_OVERLAP_DAYS = 20.0
STACK_CUTOFF = 0.5
STACK_MAX = 8
STACK_MIN = 2
GAP_EDGE_S = 1.0                      # widened either side of every gap above 1 Hz, the remote path
EDGE_S = 120.0                        # NaN inside every record end and gap end, the member path, every rate
SCREEN_K = 30.0                       # the multiple of the MAD a member's first difference may depart by
SCREEN_FLOOR_NT = 3.0                 # and the floor under that threshold, in nT
SCREEN_MIN_MEMBERS = 2                # members that have to be finite at a sample for the screen to run
SCREEN_PAD = (2, 3)                   # samples flagged before and after a flagged sample
SCREEN_MAX_SPAN_S = 12.0              # a flagged span longer than this is NaN rather than bridged
STEP_NT = 2.0                         # a sample-to-sample step this large is not the field on a quiet day
LEVEL_WIN_S = 3600.0                  # the window a member's offset from the fleet level is taken over
LEVEL_BLOCK_S = 60.0                  # the block each median of that offset is taken over first
LEVEL_BLOCK_MIN_FRAC = 1.0 / 3.0      # of a block that has to be finite for the block to count
LEVEL_MIN_FRAC = 0.1                  # of the window's blocks that have to be finite for an offset
LEVEL_MIN_MEMBERS = 2                 # members that have to be finite for the fleet level to be taken
LEVEL_PASSES = 2                      # 1 is the block median alone; 2 puts a block mean first
REPLACE_TRIES = 10                    # times a store file is offered to os.replace before giving up
REPLACE_WAIT_S = 30.0                 # waited between those tries while another process holds the old file
H = ("Hx", "Hy")
KINDS_WITH_STORE = ("remote", "stack", "obs", "stack_obs")


class NoMembers(RuntimeError):
    """A stack was asked for with nothing to stack."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _replace(src, dst, tries=REPLACE_TRIES, wait_s=REPLACE_WAIT_S) -> None:
    """Move `src` over `dst`, waiting where another process still holds the old file open.

    A reader that has the old file mapped makes the move fail on Windows with PermissionError, and a
    workbook pass may be reading the store while it is rebuilt. The wait is the whole remedy: a store file is
    never truncated in place, so a reader either sees the old file whole or the new one whole.
    """
    for k in range(int(tries)):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError:
            if k == int(tries) - 1:
                raise
            time.sleep(float(wait_s))


def _write_json(path, obj) -> None:
    """Serialise `obj` beside `path` and move it over, so a reader never sees half a sidecar."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    _replace(tmp, path)


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def gap_edge_screen(x, fs, edge_s=GAP_EDGE_S) -> np.ndarray:
    """Widen every gap by `edge_s` seconds at each end, above 1 Hz.

    The samples either side of a gap carry the recorder's settling, which the decimation to 1 Hz averages
    away and the 10 Hz grid does not. The screen widens what is already missing and so cannot break a
    continuous stretch into pieces.

    A per-sample amplitude screen was tried here first -- NaN wherever the absolute first difference
    exceeds 10x the record's median -- and is not used: measured on Q60N at 10 Hz on 2026-09-16 it NaNs
    865,078 of 52.5 M Hx samples scattered through a gapless record, which leaves 248,970 pieces of a
    median 1 s, and the 3,600 s run floor then throws away 64.9 per cent of the record. A screen whose cost
    is paid by the run floor is not a screen. Outliers are handled where they belong, inside the estimator:
    the robust regression down-weights them and the window screen drops a window whose rms first difference
    exceeds 10x the median.
    """
    v = np.asarray(x, float).copy()
    bad = ~np.isfinite(v)
    if not bad.any() or bad.all() or float(fs) <= 1.0:
        return v
    k = int(round(float(edge_s) * float(fs)))
    if k < 1:
        return v
    d = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        v[max(0, a - k):min(len(v), b + k)] = np.nan
    return v


def edge_screen(x, fs, edge_s=EDGE_S) -> np.ndarray:
    """NaN `edge_s` seconds inside every finite run: both record ends and both sides of every gap.

    The samples either side of a break carry the logger's settling, which is a step of thousands of nT at a
    PR6-24 and is not the field: Q45 (Queensland Phase 3) opens at 16,804 nT, reaches 33,214 nT and settles
    at 29,234 nT over about 10 s, and carries 24 steps above 50 nT. The alignment shift then spreads such a
    sample over its neighbours, so it is removed before the shift and not after.

    The screen runs on the stack member path only, at every rate, and takes what is inside a run rather than
    widening what is outside it, which is what gap_edge_screen does above 1 Hz on the remote path.
    """
    v = np.asarray(x, float).copy()
    k = int(round(float(edge_s) * float(fs)))
    if k < 1:
        return v
    bad = ~np.isfinite(v)
    d = np.diff(np.concatenate(([1], bad.view(np.int8), [1])))
    for a, b in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
        v[a:min(b, a + k)] = np.nan
        v[max(a, b - k):b] = np.nan
    return v


def steps_per_million(x, step_nt=STEP_NT) -> float:
    """How many of a channel's sample-to-sample steps exceed `step_nt`, per million steps.

    Taken over the finite samples in order, so a step across a gap counts: the statistic is measured the
    same way on a member and on the stack built from it, and a screen that leaves holes is charged for them.
    """
    v = np.asarray(x, float)
    d = np.diff(v[np.isfinite(v)])
    return 1e6 * float((np.abs(d) > float(step_nt)).sum()) / max(1, len(d))


def member_screen(members: dict, fs, k=SCREEN_K, floor_nt=SCREEN_FLOOR_NT,
                  min_members=SCREEN_MIN_MEMBERS, pad=SCREEN_PAD, max_span_s=SCREEN_MAX_SPAN_S) -> tuple:
    """One channel of every stack member with each member's own spikes bridged over, and the counts.

    D_i is a member's first difference. Two rules, on the number of members finite at the sample:

        three or more   med is the median over them of D; a sample is flagged where |D_i - med| exceeds
                        max(k x MAD(D_i - med), floor_nt). A spike the fleet shares is the field: the median
                        carries it and the departure is what is measured.
        exactly two     the median of two first differences does not name the culprit, so a sample is
                        flagged where |D_i| exceeds max(k x MAD(D_i), floor_nt) AND the other member's |D_j|
                        at that sample is under floor_nt / 2. Where both exceed the floor neither is
                        flagged, because a step both members take is the field.
        fewer than two, or fewer than `min_members`      nothing is flagged.

    The flagged span, pad[0] samples before it to pad[1] after and merged with its neighbours, is replaced by
    a straight line between the finite samples either side of the span inside that member, where the span is
    at most max_span_s seconds and both of those samples are finite, and is NaN otherwise.

    A line and not a hole. A member that goes NaN over a span drops out of the weighted mean there, which
    costs more than the spike did: Q60N reads 4,093 steps above 2 nT per million samples with holes, 1,068
    with the spikes left in, and 338 with the line.

    The counts are per member: the samples flagged by each rule, the spans bridged and left NaN, each rule's
    MAD and threshold in nT and which limb set it, and the bridged spans as [start, end] index pairs.
    """
    names = list(members)
    n = len(members[names[0]])
    d_all = np.vstack([np.diff(np.asarray(members[d], float), prepend=np.nan) for d in names])
    fin_all = np.isfinite(d_all)
    nfin = fin_all.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a sample no member is finite at
        med = np.nanmedian(d_all, axis=0)
    med[nfin < max(3, int(min_members))] = np.nan
    pair = (nfin == 2) if int(min_members) <= 2 else np.zeros(n, bool)
    abs_sum = np.where(fin_all, np.abs(d_all), 0.0).sum(axis=0) if pair.any() else None
    span_max = max(1, int(round(float(max_span_s) * float(fs))))
    out, counts = {}, {}
    for i, name in enumerate(names):
        r = d_all[i] - med
        fin = np.isfinite(r)
        mad = 1.4826 * float(np.median(np.abs(r[fin] - np.median(r[fin])))) if fin.any() else np.nan
        thr = max(float(k) * mad, float(floor_nt)) if np.isfinite(mad) else float(floor_nt)
        hit = fin & (np.abs(r) > thr)
        d_i, fin_i = d_all[i], fin_all[i]
        ad = np.abs(d_i)
        mad_p = (1.4826 * float(np.median(np.abs(d_i[fin_i] - np.median(d_i[fin_i]))))
                 if fin_i.any() else np.nan)
        thr_p = max(float(k) * mad_p, float(floor_nt)) if np.isfinite(mad_p) else float(floor_nt)
        if abs_sum is None:
            hit_p = np.zeros(n, bool)
        else:
            hit_p = pair & fin_i & (ad > thr_p) & ((abs_sum - np.where(fin_i, ad, 0.0)) < 0.5 * floor_nt)
        idx = np.flatnonzero(hit | hit_p)
        x = np.asarray(members[name], float).copy()
        flag = np.zeros(n, bool)
        for j in idx:
            flag[max(0, j - pad[0]):j + pad[1] + 1] = True
        dd = np.diff(np.concatenate(([0], flag.view(np.int8), [0])))
        spans, holes = [], 0
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
            if a >= 1 and b < n and (b - a) <= span_max and np.isfinite(x[a - 1]) and np.isfinite(x[b]):
                x[a:b] = np.linspace(x[a - 1], x[b], b - a + 2)[1:-1]
                spans.append([int(a), int(b)])
            else:
                x[a:b] = np.nan
                holes += 1
        out[name] = x
        counts[name] = dict(flagged=int(len(idx)), flagged_pair=int(np.count_nonzero(hit_p)),
                            bridged=len(spans), nan_spans=int(holes),
                            mad_nt=(None if not np.isfinite(mad) else round(float(mad), 4)),
                            threshold_nt=round(float(thr), 4),
                            threshold_from=("the %g nT floor" % floor_nt
                                            if not np.isfinite(mad) or float(k) * mad <= float(floor_nt)
                                            else "%g x MAD" % k),
                            pair_mad_nt=(None if not np.isfinite(mad_p) else round(float(mad_p), 4)),
                            pair_threshold_nt=round(float(thr_p), 4),
                            pair_threshold_from=("the %g nT floor" % floor_nt
                                                 if not np.isfinite(mad_p) or
                                                 float(k) * mad_p <= float(floor_nt) else "%g x MAD" % k),
                            spans=spans)
    return out, counts


def level_match(members: dict, fs, win_s=LEVEL_WIN_S, min_frac=LEVEL_MIN_FRAC,
                min_members=LEVEL_MIN_MEMBERS, block_s=LEVEL_BLOCK_S, passes=LEVEL_PASSES) -> tuple:
    """One channel of every stack member with its slow offset from the fleet removed, and each offset's rms.

    m(t) is the median across the members finite at t, taken where at least `min_members` are (with two, the
    median of a pair is their mean). The offset o_i of x_i - m is taken over blocks: the median of each
    `block_s` = 60 s block, NaN where fewer than a third of the block is finite; then a centred running
    statistic over win_s / block_s = 60 of those blocks, needing `min_frac` of that many finite; then linear
    interpolation of the block series back to the sample grid, which carries the offset across gaps and
    holds it constant beyond its ends. The member enters the weighted mean as x_i - o_i, less one constant
    common to every member, which is the mean of m over its finite samples: a constant every member shares
    cannot step the level, and it leaves the composite near zero so the store's convention of zero where no
    member is sound stays meaningful.

    Medians and not a mean over samples. A sample mean is the wrong estimator for an offset in the presence
    of excursions: the recorder's settling that survives the edge screen enters a 3,600 s mean at its own
    area over 3,600, which measured 13-22 nT on Hx over a plateau ending exactly at half the window (Ben's
    ruling, 2026-09-17). A median of 60 s blocks discards an excursion shorter than 30 s outright. Blocks and
    not samples because a running statistic over 53,000 blocks is cheap where one over 3 million samples is
    not.

    Two passes: the first takes the running MEAN of the blocks, which puts the members on a common level, and
    the second the running MEDIAN, which is robust on a target that no longer steps. Both are needed because
    m steps by up to 2,788 nT whenever the member set changes, a median follows such a step where a mean
    smooths it, and a single median pass therefore lays 285-406 steps above 2 nT per million into each member
    against 6-27 before it. `passes` = 1 is the median pass alone.

    This is what removes the slow site-specific difference -- DC, drift, Sq amplitude -- that stepped the
    stack's level whenever a member dropped in or out; it changes the stack's content only at periods above
    about win_s and only by the smoothed difference between the weighted mean and the median of the
    coherent members. A member is not demeaned over its own record afterwards: the offset does that job, and
    a per-record mean is exactly the quantity that made the level step.

    A member the fleet never overlaps has no offset to measure and falls back to its own mean, which is the
    only estimate of its level there is; it is reported with the others by its rms.

    Returns ({member: array}, {member: the offset's rms in nT}, the members' established level). The level is
    the fleet median of the matched members, less the same common constant, and is what the observatory is
    matched against one-sidedly in build_stack_obs.
    """
    names = list(members)
    n = len(members[names[0]])
    stack = np.vstack([np.asarray(members[d], float) for d in names])
    off = np.zeros_like(stack)
    block = max(1, int(round(float(block_s) * float(fs))))
    nwin = max(1, int(round(float(win_s) / float(block_s))))
    need = max(1, int(round(float(min_frac) * nwin)))
    base = 0.0
    npass = max(1, int(passes))
    for k in range(npass):
        m = _fleet_level(stack, min_members)
        if k == 0:
            base = float(np.nanmean(m)) if np.isfinite(m).any() else 0.0
        smooth = npass > 1 and k == 0
        for i in range(len(names)):
            b = _block_median(stack[i] - m, block)
            got = _running_mean(b, nwin, need) if smooth else _running_median(b, nwin, need)
            if np.isfinite(got).any():
                o = _blocks_to_samples(got, n, block)
            else:
                g = np.isfinite(stack[i])
                o = np.full(n, float(stack[i][g].mean()) - base if g.any() else 0.0)
            stack[i] -= o
            off[i] += o
        del m
    out = {name: stack[i] - base for i, name in enumerate(names)}
    rms = {name: round(float(np.sqrt(np.mean(off[i] ** 2))), 3) for i, name in enumerate(names)}
    level = (_fleet_level(stack, min_members) - base).astype(np.float32)
    return out, rms, level


def one_sided_offset(x, level, fs, win_s=LEVEL_WIN_S, min_frac=LEVEL_MIN_FRAC, block_s=LEVEL_BLOCK_S,
                     passes=LEVEL_PASSES) -> np.ndarray:
    """The offset of one record from a level that is already established, by level_match's own estimator.

    `level` is the members' matched fleet median and is not recomputed: the record being matched does not
    enter it, so a record whose own level sits thousands of nT away cannot drag what it is matched against.
    The offset is taken over the samples where both are finite, carried across the record's gaps and held
    beyond its ends by the same interpolation from block centres.

    This is how the observatory enters the stack + observatory. It is matched and not demeaned over its own
    record, because a per-record mean is what stepped the composite every time the archive's mask opened or
    closed -- 42 to 70 times over a Queensland window, which put stack + observatory at 74-207 steps above
    2 nT per million against a stack at 9.5-43.5 (2026-09-17).
    """
    x = np.asarray(x, float)
    n = len(x)
    block = max(1, int(round(float(block_s) * float(fs))))
    nwin = max(1, int(round(float(win_s) / float(block_s))))
    need = max(1, int(round(float(min_frac) * nwin)))
    resid = x - np.asarray(level, float)
    out = np.zeros(n)
    npass = max(1, int(passes))
    for k in range(npass):
        b = _block_median(resid, block)
        got = _running_mean(b, nwin, need) if (npass > 1 and k == 0) else _running_median(b, nwin, need)
        if not np.isfinite(got).any():
            break
        step = _blocks_to_samples(got, n, block)
        out += step
        resid -= step
    return out


def _fleet_level(stack, min_members=LEVEL_MIN_MEMBERS) -> np.ndarray:
    """The median across the members finite at each sample, NaN where fewer than min_members are."""
    nfin = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a sample no member is finite at
        m = np.nanmedian(stack, axis=0)
    m[nfin < int(min_members)] = np.nan
    return m


def _block_median(y, block, min_frac=LEVEL_BLOCK_MIN_FRAC) -> np.ndarray:
    """The median of each `block` samples of y, NaN where fewer than `min_frac` of the block is finite.

    The last block is padded with NaN and is judged on the same count as a whole one, so a tail shorter than
    a third of a block does not carry a block of its own.
    """
    y = np.asarray(y, float)
    n = len(y)
    nb = int(np.ceil(n / block)) if n else 0
    if not nb:
        return np.zeros(0)
    pad = nb * block - n
    v = (np.concatenate([y, np.full(pad, np.nan)]) if pad else y).reshape(nb, block)
    cnt = np.isfinite(v).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a block with nothing finite in it
        med = np.nanmedian(v, axis=1)
    med[cnt < max(1, int(round(float(min_frac) * block)))] = np.nan
    return med


def _running_median(b, w, need) -> np.ndarray:
    """The centred running median of the block series b over w blocks, NaN where fewer than `need` finite.

    Only whole windows are taken, so the first and last w // 2 blocks are NaN and the interpolation back to
    the sample grid holds the nearest measured value over them. A truncated window is what biased the sample
    mean this replaces, so none is offered here.
    """
    b = np.asarray(b, float)
    nb = len(b)
    out = np.full(nb, np.nan)
    if nb < w:
        g = np.isfinite(b)
        if g.sum() >= int(need):
            out[:] = float(np.median(b[g]))
        return out
    sw = np.lib.stride_tricks.sliding_window_view(b, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sw, axis=1)
    med[np.isfinite(sw).sum(axis=1) < int(need)] = np.nan
    out[w // 2:w // 2 + len(med)] = med
    return out


def _running_mean(b, w, need) -> np.ndarray:
    """The centred running mean of the block series b over w blocks, NaN where fewer than `need` finite.

    The first pass of the level match, where the target still steps by hundreds of nT and a median would
    follow the step rather than smooth it. The block medians beneath it have already taken out the
    excursions a mean over samples would have been dragged by.
    """
    b = np.asarray(b, float)
    nb = len(b)
    out = np.full(nb, np.nan)
    if nb < w:
        g = np.isfinite(b)
        if g.sum() >= int(need):
            out[:] = float(np.mean(b[g]))
        return out
    sw = np.lib.stride_tricks.sliding_window_view(b, w)
    good = np.isfinite(sw)
    cnt = good.sum(axis=1)
    with np.errstate(invalid="ignore"):
        got = np.where(good, sw, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    got[cnt < int(need)] = np.nan
    out[w // 2:w // 2 + len(got)] = got
    return out


def _blocks_to_samples(b, n, block) -> np.ndarray:
    """The block series on the sample grid, linear between block centres and held beyond the finite ones.

    Zero throughout where nothing is finite: a member the fleet never overlaps has no offset to measure.
    """
    b = np.asarray(b, float)
    g = np.isfinite(b)
    if not g.any():
        return np.zeros(n)
    centres = np.arange(len(b), dtype=float) * block + 0.5 * (block - 1)
    return np.interp(np.arange(n, dtype=float), centres[g], b[g])


class Store:
    """One survey's references at one rate, with the caches the scoring needs.

    `spec_rate` is the rate the decisions are measured at. A 10 Hz store is built with spec_rate 1: the pool,
    the remote, the members, the weights and the lags come from the 1 Hz store and only the arrays are read
    on the 10 Hz grid.
    """

    def __init__(self, survey, sites=None, rate=1, spec_rate=None, rot_cache=None, coh_min=COH_MIN,
                 coh_relax=COH_RELAX, min_overlap_days=MIN_OVERLAP_DAYS, cutoff=STACK_CUTOFF,
                 n_max=STACK_MAX, n_min=STACK_MIN):
        self.sv = survey
        self.cfg = survey.cfg
        self.work = Path(survey.cfg["work_root"])
        self.rate = int(rate)
        self.fs = float(rate)
        self.spec_rate = int(spec_rate if spec_rate is not None else rate)
        self.sites = list(sites) if sites is not None else list(survey.sites.site)
        self.dir = self.work / "references" / ("%dhz" % self.rate)
        self.coh_min = float(coh_min)
        self.coh_relax = float(coh_relax)
        self.min_overlap_days = float(min_overlap_days)
        self.cutoff = float(cutoff)
        self.n_max = int(n_max)
        self.n_min = int(n_min)
        self._rot: dict = {}
        self._order: list = []
        self._max = int(rot_cache if rot_cache is not None else (24 if self.rate == 1 else 2))
        self._events: dict = {}
        self._scores: dict = {}
        self._pool = None
        self._fleet = None
        self._groups = None

    # -------------------------------------------------------------- the record

    def has_cache(self, site: str) -> bool:
        return (self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)).exists()

    def rotated(self, site: str):
        """(t0, {'Hx','Hy'} in the site's own mean-field frame, angle, n) at this store's rate.

        Signs from decisions.csv are applied first, then the rotation; rot_drop days come back NaN and are
        excluded from the mean. Above 1 Hz the gap-edge screen runs before the rotation.
        """
        if site in self._rot:
            self._order.remove(site)
            self._order.append(site)
            return self._rot[site]
        path = self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)
        z = np.load(path, allow_pickle=False)
        t0 = int(z["t0"][0])
        arr = {c: np.asarray(z[c], float) for c in H}
        z.close()
        if self.rate > 1:
            arr = {c: gap_edge_screen(arr[c], self.fs) for c in H}
        dec = self._decision(site)
        arr, _applied, _und = FR.apply_signs(arr, dec)
        regimes = FR.parse_regimes(dec.get("rot_regimes") if dec is not None else "")
        drop = FR.parse_regimes(dec.get("rot_drop") if dec is not None else "")
        arr, ang = FR.rotate_to_mean_field(arr, regimes=regimes, drop=drop, fs=self.fs)
        got = (t0, {c: np.asarray(arr[c], np.float32) for c in H}, ang, len(arr["Hx"]))
        self._rot[site] = got
        self._order.append(site)
        while len(self._order) > self._max:
            self._rot.pop(self._order.pop(0), None)
        return got

    def _decision(self, site: str):
        try:
            return self.sv.decision(site)
        except KeyError:
            return None

    def window(self, site: str) -> tuple:
        """(t0, n) of the site's cache at this rate, read from the npz header alone."""
        path = self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)
        z = np.load(path, allow_pickle=False)
        t0, n = int(z["t0"][0]), int(len(z["Hx"]))
        z.close()
        return t0, n

    def usable_days(self, site: str) -> float:
        _t0, h, _a, _n = self.rotated(site)
        return float(np.isfinite(h["Hx"]).sum()) / (86400.0 * self.fs)

    def events(self, site: str) -> list:
        if site not in self._events:
            self._events[site] = TR.site_events(site, self.sites, self.work, self.cfg)
        return self._events[site]

    def position(self, site: str) -> tuple:
        r = self.sv.site(site)
        return float(r.lat), float(r.lon)

    def distance_km(self, a: str, b: str) -> float:
        return geo.distance_km(self.position(a), self.position(b))

    # ---------------------------------------------------------------- the pool

    def groups(self) -> dict:
        """{site: group} from <work_root>/survey/deployment_groups.csv, written by workbook 01."""
        if self._groups is None:
            p = self.work / "survey" / "deployment_groups.csv"
            if not p.exists():
                raise FileNotFoundError("%s has not been written; run workbook 01 first" % p)
            g = pd.read_csv(p)
            self._groups = {m: r.group for r in g.itertuples() for m in str(r.members).split()}
        return self._groups

    def clean_pool(self, force=False) -> tuple:
        """(the pool, one row per site of the clean test with its reason).

        A site is in the pool where its tail scan is clean and it has a cache at this store's rate. The rows
        name every exclusion.
        """
        if self._pool is not None and not force:
            return self._pool
        rows = []
        for s in self.sites:
            r = TR.clean_row(s, self.sites, self.work, self.cfg)
            if not self.has_cache(s):
                r["clean"] = False
                r["reason"] = "; ".join(x for x in (r["reason"], "no %d Hz cache" % self.rate) if x)
            rows.append(r)
        table = pd.DataFrame(rows)
        pool = [r["site"] for r in rows if r["clean"]]
        self._pool = (pool, table)
        return self._pool

    def members_for(self, target: str) -> tuple:
        """(the candidate members of a target, one line saying where the set came from).

        decisions.csv `stack_members` where the cell names sites, otherwise the clean pool restricted to the
        target's own overlap group.
        """
        dec = self._decision(target)
        cell = str(dec.get("stack_members", "") if dec is not None else "").strip()
        if cell and cell.lower() not in ("decide", "nan", "none"):
            named = [s for s in cell.replace(",", " ").split() if s != target]
            return named, "decisions.csv stack_members"
        pool, _t = self.clean_pool()
        grp = self.groups()
        mine = grp.get(target)
        out = [s for s in pool if s != target and grp.get(s) == mine and self.has_cache(s)]
        return out, "the clean pool inside overlap group %s" % (mine or "-")

    # ------------------------------------------------------------- the scoring

    def overlap(self, a: str, b: str):
        ta, na = self.window(a)
        tb, nb = self.window(b)
        lo, hi = max(ta, tb), min(ta + na / self.fs, tb + nb / self.fs)
        return (lo, hi) if hi > lo else None

    def overlap_floor(self, target: str, own_days: float) -> float:
        """min(MIN_OVERLAP_DAYS, 0.75 x the days the target can use).

        The days the target can use, not its cache length: a cache is as long as the logger ran, not as long
        as the magnetics are sound, and asking a candidate to cover 9.8 days of a 2.6-day usable record
        admits nobody (Ben, 2026-09-11).
        """
        try:
            u = float(self.usable_days(target))
        except Exception:
            u = float("nan")
        if not np.isfinite(u) or u <= 0:
            u = own_days
        return float(min(self.min_overlap_days, 0.75 * min(u, own_days)))

    def _on_grid(self, site: str, t0, n, drop_events=True, lag_s=None, steps_before=None) -> dict:
        """A member's rotated H on the target grid; NaN where it has nothing to say.

        `drop_events` NaNs the member's own transient chunks and then screens EDGE_S = 120 s inside every
        record end and every gap end, which is what a stack member needs: the samples either side of a break
        carry the logger's settling and the shift below would spread them. A remote site is passed through
        whole instead, because the single-reference pass cuts local and remote together on the union of the
        two masks at processing time and NaNing here would hide that cut.

        The order is fixed: the events, then the edge screen, then the shift. The edge screen has to come
        before the shift because the shift reaches LANCZOS_A samples either side of what it is given.

        `steps_before`, where a dict is given, is filled with each channel's steps per million before the
        edge screen and the shift. That is the member as it stands, and the number the stack is judged
        against.
        """
        td, hd, _a, nd = self.rotated(site)
        out = {c: np.full(int(n), np.nan) for c in H}
        fs = self.fs
        lo, hi = max(t0, td), min(t0 + n / fs, td + nd / fs)
        i0 = j0 = L = 0
        if hi > lo:
            i0 = int(round((lo - t0) * fs))
            j0 = int(round((lo - td) * fs))
            L = max(0, min(int(round((hi - lo) * fs)), int(n) - i0, nd - j0))
        if L:
            for c in H:
                out[c][i0:i0 + L] = hd[c][j0:j0 + L]
        if drop_events:
            if L:
                keep = TR.keep_mask(td, nd, fs, self.events(site))
                for c in H:
                    out[c][i0:i0 + L][~keep[j0:j0 + L]] = np.nan
            if steps_before is not None:
                steps_before.update({c: steps_per_million(out[c]) for c in H})
            for c in H:
                out[c] = edge_screen(out[c], fs)
        if lag_s is not None and np.isfinite(lag_s) and lag_s != 0.0:
            out = {c: align.shift(out[c], float(lag_s), fs) for c in H}
        return out

    def pair_coherence(self, a: str, b: str, band=COH.PAIR_BAND, nperseg=COH.PAIR_NPERSEG,
                       min_segments=COH.PAIR_MIN_SEGMENTS) -> dict:
        """coherence.pair_coh over the two sites' overlap, both event lists masked."""
        ta, ha, _aa, na = self.rotated(a)
        tb, hb, _ab, nb = self.rotated(b)
        ov = self.overlap(a, b)
        if ov is None:
            return dict(coh=float("nan"), chunks=0, chunk_days=0.0, clean_days=0.0, mask_frac=float("nan"),
                        overlap_s=0)
        lo, hi = ov
        fs = self.fs
        ia, ib = int(round((lo - ta) * fs)), int(round((lo - tb) * fs))
        L = int(round((hi - lo) * fs))
        keep = TR.keep_mask(lo, L, fs, self.events(a) + self.events(b))
        cd = float(((self.cfg.get("floors") or {}).get("member_chunk_days")) or COH.CHUNK_DAYS)
        mc = 1 if cd < COH.CHUNK_DAYS else COH.MIN_CHUNKS
        ms = max(4, int(round(min_segments * min(1.0, cd / COH.CHUNK_DAYS))))
        r = COH.pair_coh({c: ha[c][ia:ia + L] for c in H}, {c: hb[c][ib:ib + L] for c in H}, fs, keep,
                         band=band, nperseg=COH.nperseg_for(nperseg, fs), min_segments=ms,
                         chunk_days=cd, min_chunks=mc)
        r["overlap_s"] = int(hi - lo)
        return r

    def score_candidates(self, target: str, pool=None, verbose=False, force=False) -> pd.DataFrame:
        """One row per candidate: km, overlap, event fraction, baseline, clean, and the 20-200 s coherence.

        A candidate below the overlap floor is reported with the reason and not scored for coherence. The
        table is kept per target, because the remote-site rule and the stack weights both ask for it.
        """
        if pool is None and target in self._scores and not force:
            return self._scores[target]
        cached = pool is None
        pool = list(pool) if pool is not None else self.members_for(target)[0]
        ta, na = self.window(target)
        own_days = max(na / (86400.0 * self.fs), 1e-9)
        floor = self.overlap_floor(target, own_days)
        clean_tab = self.clean_pool()[1].set_index("site")
        rows = []
        for d in pool:
            if d == target or not self.has_cache(d):
                continue
            ov = self.overlap(target, d)
            row = dict(name=d, km=round(self.distance_km(target, d), 1),
                       own_days=round(own_days, 1), min_overlap_days=round(floor, 1))
            if ov is None:
                row.update(overlap_days=0.0, overlap_frac=0.0, ok=False, why="no overlap",
                           coh=None, chunks=0, clean=False, event_frac=None, baseline=None)
                rows.append(row)
                continue
            ovd = (ov[1] - ov[0]) / 86400.0
            cr = clean_tab.loc[d] if d in clean_tab.index else None
            row.update(overlap_days=round(ovd, 1), overlap_frac=round(ovd / own_days, 3),
                       event_frac=(None if cr is None or cr.event_frac is None else round(float(cr.event_frac), 4)),
                       baseline=(None if cr is None or cr.baseline is None else round(float(cr.baseline), 1)),
                       clean=bool(cr.clean) if cr is not None else False,
                       ok=bool(ovd >= floor))
            if not row["ok"]:
                row.update(why="overlap %.1f d < floor %.1f d" % (ovd, floor), coh=None, chunks=0)
                rows.append(row)
                continue
            r = self.pair_coherence(target, d)
            row.update(why="", coh=(None if not np.isfinite(r["coh"]) else round(r["coh"], 3)),
                       chunks=int(r["chunks"]), chunk_days=r["chunk_days"],
                       clean_days=r["clean_days"], mask_frac=round(float(r["mask_frac"]), 4))
            rows.append(row)
            if verbose:
                print("      %-8s %6.0f km  ov %5.1f d  coh %s" % (d, row["km"], row["overlap_days"],
                                                                   row["coh"]), flush=True)
        cols = ["name", "km", "overlap_days", "overlap_frac", "event_frac", "baseline", "clean", "coh",
                "chunks", "ok", "why", "own_days", "min_overlap_days"]
        t = pd.DataFrame(rows)
        t = (t.reindex(columns=[c for c in cols if c in t.columns] +
                       [c for c in t.columns if c not in cols]) if len(t) else t)
        if cached:
            self._scores[target] = t
            self.dir.mkdir(parents=True, exist_ok=True)
            _write_json(self.dir / ("candidates_%s.json" % target),
                        dict(target=target, band_s=[20, 200], built_at=_now(),
                             rows=t.to_dict("records")))
        return t

    # ------------------------------------------------------------ the remote site

    def remote_site(self, target: str, scores=None, coh_min=None, coh_relax=None) -> dict:
        """The chosen remote, its branch and the reason, or decisions.csv's named site where there is one."""
        coh_min = self.coh_min if coh_min is None else float(coh_min)
        coh_relax = self.coh_relax if coh_relax is None else float(coh_relax)
        dec = self._decision(target)
        named = str(dec.get("remote_site", "") if dec is not None else "").strip()
        scores = self.score_candidates(target) if scores is None else scores
        cands = scores[scores.ok].to_dict("records") if len(scores) else []
        cands.sort(key=lambda c: c["km"])
        rule = self._rule_choice(target, cands, scores, coh_min, coh_relax)
        if named and named.lower() not in ("decide", "nan", "none"):
            row = next((c for c in cands if c["name"] == named), {})
            src = str(dec.get("remote_source", "") if dec is not None else "") or "decisions.csv remote_site"
            return dict(target=target, name=named, branch=0, coh=row.get("coh"),
                        source="decisions.csv", source_detail=src,
                        reason="decisions.csv names %s as the remote (%s); the rule would have chosen %s "
                               "on %s" % (named, src[:120], rule["name"], rule["reason"].split(":")[0]),
                        rule_name=rule["name"], rule_branch=rule["branch"], rule_coh=rule["coh"],
                        candidates=cands)
        rule.update(source="the five-branch rule", source_detail="", rule_name=rule["name"],
                    rule_branch=rule["branch"], rule_coh=rule["coh"], candidates=cands)
        return rule

    def _rule_choice(self, target, cands, scores, coh_min, coh_relax) -> dict:
        n_considered = len(scores)
        n_short = int((~scores.ok).sum()) if len(scores) else 0
        own_days = float(scores.own_days.iloc[0]) if len(scores) else float("nan")
        floor = float(scores.min_overlap_days.iloc[0]) if len(scores) else float("nan")

        def nearest():
            pool = self.members_for(target)[0]
            near = sorted((d for d in pool if d != target and self.has_cache(d)),
                          key=lambda d: self.distance_km(target, d))
            return near[0] if near else None

        return branch_rule(target, cands, n_considered, n_short, own_days, floor,
                           coh_min=coh_min, coh_relax=coh_relax, nearest=nearest)

    # ----------------------------------------------------------- the fleet weights

    def fleet_weights(self, force=False, verbose=False) -> tuple:
        """({member: median 100-1000 s coherence with the rest of the pool}, the per-pair table).

        The quantity the stack weights on. Computed once per store and kept as fleet_weights.json with the
        pairs in fleet_pairs.json, so the workbook can recompute one site's weights from the table it prints.
        """
        if self._fleet is not None and not force:
            return self._fleet
        fw = self.dir / "fleet_weights.json"
        fp = self.dir / "fleet_pairs.json"
        if fw.exists() and fp.exists() and not force:
            self._fleet = (json.loads(fw.read_text(encoding="utf-8")),
                           json.loads(fp.read_text(encoding="utf-8")))
            return self._fleet
        pool, _t = self.clean_pool()
        pairs = {}
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                r = self.pair_coherence(a, b, band=COH.WEIGHT_BAND, nperseg=COH.WEIGHT_NPERSEG,
                                        min_segments=COH.WEIGHT_MIN_SEGMENTS)
                pairs["%s:%s" % (a, b)] = dict(
                    coh=(None if not np.isfinite(r["coh"]) else round(float(r["coh"]), 4)),
                    chunks=int(r["chunks"]))
                if verbose:
                    print("   fleet pair %s-%s coh %s over %d chunks"
                          % (a, b, pairs["%s:%s" % (a, b)]["coh"], r["chunks"]), flush=True)
        weights = COH.fleet_weight_table(pairs, pool)
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(fw, weights)
        _write_json(fp, pairs)
        self._fleet = (weights, pairs)
        return self._fleet

    # ---------------------------------------------------------------- the stack

    def stack_members(self, target: str, cutoff=None, n_max=None, align_members=True,
                      verbose=False) -> tuple:
        """({member: weight}, {member: lag_s}, {member: refusal}, {member: alignment note}).

        The weight is the member's fleet coherence at 100-1000 s. A member is refused where it has no fleet
        coherence, where that coherence is below the cutoff, or where it falls outside the best n_max.
        """
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        cand, _why = self.members_for(target)
        weights, _pairs = self.fleet_weights()
        # the default pool, so the candidate table kept for the remote-site rule answers here as well
        scores = self.score_candidates(target)
        ok = set(scores.name[scores.ok]) if len(scores) else set()
        scored, refused, lags, notes = [], {}, {}, {}
        for d in cand:
            if d not in ok:
                row = scores[scores.name == d]
                refused[d] = str(row.why.iloc[0]) if len(row) else "not scored"
                continue
            w = weights.get(d)
            if w is None or not np.isfinite(w) or w <= 0:
                refused[d] = "no fleet coherence at 100-1000 s (fewer than 3 scored pairs)"
                continue
            if w < cutoff:
                refused[d] = "fleet coherence %.3f < %s" % (w, cutoff)
                continue
            scored.append((float(w), d))
        scored.sort(reverse=True)
        for w, d in scored[n_max:]:
            refused[d] = "fleet coherence %.3f, outside the best %d" % (w, n_max)
        kept = {d: round(w, 4) for w, d in scored[:n_max]}
        if align_members and kept:
            t0, n = self.window(target)
            tt, th, _a, _n = self.rotated(target)
            keep = TR.keep_mask(t0, n, self.fs, self.events(target))
            fb = (self.cfg.get("floors") or {}).get("align_fallback_s")
            for d in list(kept):
                g = self._on_grid(d, t0, n, drop_events=True)
                l, why = align.shift_for({c: np.asarray(th[c], float) for c in H}, g, self.fs, keep,
                                         fallback_s=fb)
                lags[d] = round(float(l), 3)
                if why:
                    notes[d] = "kept at lag 0: %s" % why
                del g
        else:
            lags = {d: 0.0 for d in kept}
        if verbose and kept:
            print("   %s stack members: %s" % (target, ", ".join("%s %.3f (%+.2f s)"
                                                                 % (d, w, lags.get(d, 0.0))
                                                                 for d, w in kept.items())), flush=True)
        return kept, lags, refused, notes

    def accumulate(self, target: str, weights: dict, lags: dict) -> tuple:
        """(t0, n, num, den, count, screen, spans, levels) of the weighted member sum, one read per member.

        `screen` is the spike screen's record for the sidecar, `spans` the spans it bridged, merged over the
        members, for the npz -- at 1 Hz a stack carries up to 33,000 of them and they belong beside the
        arrays rather than in a sidecar a reader opens -- and `levels` the members' established level per
        channel, which build_stack_obs matches the observatory against.

        The members are held together rather than summed one at a time, because the two steps between the
        shift and the sum are both comparisons between them: the spike screen, without which a member's own
        logger spike enters the stack at its weight over the sum of the weights (Q38 carries 1,152 such
        steps per million samples of its own), and the level match, without which the composite's level
        steps every time a member drops in or out.
        """
        t0, n = self.window(target)
        arrays, before = {}, {}
        for d in weights:
            b = {}
            arrays[d] = self._on_grid(d, t0, n, drop_events=True, lag_s=(lags or {}).get(d),
                                      steps_before=b)
            before[d] = b
        screened, counts, offsets, levels = {}, {}, {}, {}
        for c in H:
            got, cn = member_screen({d: arrays[d].pop(c) for d in arrays}, self.fs)
            got, rms, level = level_match(got, self.fs)
            screened[c] = got
            offsets[c] = rms
            levels[c] = level          # what the observatory is matched against in build_stack_obs
            for d, v in cn.items():
                counts.setdefault(d, {})[c] = v
        del arrays
        num = {c: np.zeros(n) for c in H}
        den = {c: np.zeros(n) for c in H}
        cnt = np.zeros(n, np.int16)
        for d, w in weights.items():
            for c in H:
                x = screened[c][d]
                m = np.isfinite(x)
                num[c][m] += w * x[m]
                den[c][m] += w
            cnt += np.isfinite(screened["Hx"][d])
        screen = dict(
            rule="a member sample whose first difference departs from the median over members by more than "
                 "max(%g x MAD, %g nT) is flagged -- where exactly two members are finite, where |D| passes "
                 "that bar and the other member's is under %g nT -- and the span %d samples before it to %d "
                 "after, merged with its neighbours, is bridged by a straight line where the span is at "
                 "most %g s. Each member is then set on the fleet's own level, by the running median "
                 "over %g s of the medians of its %g s blocks."
                 % (SCREEN_K, SCREEN_FLOOR_NT, 0.5 * SCREEN_FLOOR_NT, SCREEN_PAD[0], SCREEN_PAD[1],
                    SCREEN_MAX_SPAN_S, LEVEL_WIN_S, LEVEL_BLOCK_S),
            k=SCREEN_K, floor_nt=SCREEN_FLOOR_NT, min_members=SCREEN_MIN_MEMBERS,
            pad_samples=list(SCREEN_PAD), max_span_s=SCREEN_MAX_SPAN_S, edge_s=EDGE_S, step_nt=STEP_NT,
            level_win_s=LEVEL_WIN_S, level_block_s=LEVEL_BLOCK_S, level_min_frac=LEVEL_MIN_FRAC,
            level_min_members=LEVEL_MIN_MEMBERS, level_passes=LEVEL_PASSES,
            level_pass_statistic=(["running mean of the blocks", "running median of the blocks"]
                                  if LEVEL_PASSES > 1 else ["running median of the blocks"]),
            members={d: {c: dict({k: v for k, v in counts[d][c].items() if k != "spans"},
                                 offset_rms_nt=offsets[c][d],
                                 steps_per_million_before=round(before[d].get(c, float("nan")), 1),
                                 steps_per_million_after=round(steps_per_million(screened[c][d]), 1))
                         for c in H} for d in weights},
            steps_per_million=dict(
                members_median_raw={
                    c: round(float(np.median([before[d].get(c, float("nan")) for d in weights])), 1)
                    for c in H},
                members_median_screened={
                    c: round(float(np.median([steps_per_million(screened[c][d]) for d in weights])), 1)
                    for c in H}))
        spans = {"bridged_%s" % c: _merge_spans([counts[d][c]["spans"] for d in weights]) for c in H}
        return t0, n, num, den, cnt, screen, spans, levels

    @staticmethod
    def finish(num, den, thin=None) -> tuple:
        """(H, coverage per channel, mask): zero and mask False where no member is sound, and where `thin`
        says fewer than STACK_MIN SITE members are.

        `thin` is the per-sample form of the floor the member list is judged on, and it counts site members
        in both stack kinds: a sample carried by one of them is a remote site renamed, and one of them with
        the observatory is a remote site with an observatory. A zero reference contributes nothing to either
        spectrum where it is masked out.
        """
        ok = {c: den[c] > 0 for c in H}
        if thin is not None:
            t = np.asarray(thin, bool)
            ok = {c: ok[c] & ~t for c in H}
        h = {c: np.where(ok[c], num[c] / np.maximum(den[c], 1e-12), 0.0) for c in H}
        cov = {c: round(float(ok[c].mean()), 4) for c in H}
        mask = ok["Hx"] & ok["Hy"]
        return h, cov, mask

    # ---------------------------------------------------------- the observatory

    def observatory_member(self, code: str, t0, n) -> tuple:
        """(H in the observatory's own mean-field frame, its real-data mask, the angle used).

        The archive holds geographic X, Y, Z; the pair is turned into the mean-field frame of the window,
        like every member. The observatory is never shifted.

        The archive is one-second data whatever this store's rate is. On a 10 Hz grid each sample is held
        for ten, which is a zero-order hold and puts images above 0.5 Hz: the observatory kinds are built at
        1 Hz, where the archive is native, and a 10 Hz run takes the single station, the remote site and the
        fleet stack.
        """
        archive = (self.cfg.get("observatory") or {}).get("archive", "")
        n1 = int(round(n / self.fs))                       # the archive is 1 Hz whatever this store's rate is
        d = observatory.load(code, int(t0), n1, archive)
        arr = {c: np.asarray(d[c], float) for c in H}
        turned, ang = FR.rotate_to_mean_field(arr, fs=1.0)
        mask = np.asarray(d["mask"], bool)
        if self.rate != 1:
            rep = int(round(self.fs))
            turned = {c: np.repeat(turned[c], rep)[:int(n)] for c in H}
            mask = np.repeat(mask, rep)[:int(n)]
            for c in H:
                if len(turned[c]) < int(n):
                    turned[c] = np.concatenate([turned[c], np.full(int(n) - len(turned[c]), np.nan)])
            if len(mask) < int(n):
                mask = np.concatenate([mask, np.zeros(int(n) - len(mask), bool)])
        return turned, mask, (ang if isinstance(ang, float) else float("nan"))

    def site_observatory_coh(self, site: str, code: str) -> dict:
        """One site's event-free coherence with the observatory at 100-1000 s, on the same rule as a pair.

        A reading: it says how much of the site's long-period field the observatory shares from several
        hundred kilometres away, which is what the observatory kind can buy and what it cannot.
        """
        t0, n = self.window(site)
        o, omask, _a = self.observatory_member(code, t0, n)
        _ts, hs, _as, _ns = self.rotated(site)
        keep = TR.keep_mask(t0, n, self.fs, self.events(site)) & omask
        r = COH.pair_coh({c: np.asarray(hs[c], float) for c in H}, o, self.fs, keep,
                         band=COH.WEIGHT_BAND, nperseg=COH.nperseg_for(COH.WEIGHT_NPERSEG, self.fs),
                         min_segments=COH.WEIGHT_MIN_SEGMENTS)
        del o
        return r

    def observatory_weight(self, code: str, force=False) -> tuple:
        """(the observatory's fleet coherence at 100-1000 s, the pairs it was measured over).

        Measured on the same rule as a site member and against the same pool, so the observatory enters the
        stack at a weight on the members' own scale rather than at its agreement with any one target.
        """
        p = self.dir / "observatory_weight.json"
        if p.exists() and not force:
            return tuple(json.loads(p.read_text(encoding="utf-8")).values())[:2]
        pool, _t = self.clean_pool()
        vals, pairs = [], {}
        for s in pool:
            t0, n = self.window(s)
            o, omask, _a = self.observatory_member(code, t0, n)
            _ts, hs, _as, _ns = self.rotated(s)
            keep = TR.keep_mask(t0, n, self.fs, self.events(s)) & omask
            r = COH.pair_coh({c: np.asarray(hs[c], float) for c in H}, o, self.fs, keep,
                             band=COH.WEIGHT_BAND, nperseg=COH.nperseg_for(COH.WEIGHT_NPERSEG, self.fs),
                             min_segments=COH.WEIGHT_MIN_SEGMENTS)
            pairs[s] = None if not np.isfinite(r["coh"]) else round(float(r["coh"]), 4)
            if pairs[s] is not None:
                vals.append(pairs[s])
            del o
        w = round(float(np.median(vals)), 4) if vals else None
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(p, dict(weight=w, pairs=pairs, code=code, band_s=[100, 1000], built_at=_now()))
        return w, pairs

    # ----------------------------------------------------------------- the store

    def path(self, kind: str, site: str) -> Path:
        return self.dir / ("%s_%s.npz" % (kind, site))

    def _write(self, kind, site, t0, h, mask, info, extra=None) -> tuple:
        """Write the npz and its sidecar, and return (path, the sidecar as written).

        `extra` is any further array the kind carries; the kinds that pass none write the same six arrays
        they always have.
        """
        p = self.path(kind, site)
        p.parent.mkdir(parents=True, exist_ok=True)
        cov = float(np.asarray(mask, bool).mean())
        arrays = dict(t0=np.array([int(t0)], np.int64), fs=np.array([float(self.fs)]),
                      Hx=np.asarray(h["Hx"], np.float32), Hy=np.asarray(h["Hy"], np.float32),
                      mask=np.asarray(mask, bool), coverage=np.array([cov]))
        arrays.update(extra or {})
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        _replace(tmp, p)
        info = dict(info)
        info.update(kind=kind, target=site, rate_hz=self.fs, npz=p.name, t0=int(t0),
                    n=int(len(h["Hx"])), coverage=round(cov, 4), built_at=_now())
        _write_json(p.with_suffix(".json"), info)
        return p, info

    def build_remote(self, target: str, choice=None) -> tuple:
        """The chosen remote's rotated H on the target grid, finite everywhere, zero outside its record.

        The remote's own transient intervals are left in and written to the sidecar: the single-reference
        pass cuts local and remote together on the union of the two masks, and NaNing here would hide that
        cut from the code that builds the keep mask.
        """
        choice = self.remote_site(target) if choice is None else choice
        name = choice.get("name")
        if not name:
            raise NoMembers("%s: no remote site could be chosen -- %s" % (target, choice.get("reason", "")))
        t0, n = self.window(target)
        h = self._on_grid(name, t0, n, drop_events=False)
        mask = np.isfinite(h["Hx"]) & np.isfinite(h["Hy"])
        for c in H:
            h[c] = np.where(mask, h[c], 0.0)
        ev = [(float(a), float(b)) for a, b in self.events(name) if b > t0 and a < t0 + n / self.fs]
        info = dict(remote=name, branch=choice.get("branch"), coh=choice.get("coh"),
                    source=choice.get("source"), source_detail=choice.get("source_detail"),
                    reason=choice.get("reason"), band_s=[20, 200],
                    rule_name=choice.get("rule_name"), rule_branch=choice.get("rule_branch"),
                    rule_coh=choice.get("rule_coh"),
                    members=[dict(name=name, weight=1.0, lag_s=0.0, role="remote")],
                    weights={name: 1.0}, lags={name: 0.0}, refusals={},
                    remote_events=[[_iso(a), _iso(b)] for a, b in ev],
                    remote_event_frac_in_window=round(
                        float(sum(min(b, t0 + n / self.fs) - max(a, t0) for a, b in ev)) /
                        max(n / self.fs, 1.0), 5),
                    n_candidates=len(choice.get("candidates", [])),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="finite everywhere; zero and mask False outside the remote's own record")
        return self._write("remote", target, t0, h, mask, info)

    def build_stack(self, target: str, members=None, cutoff=None, n_max=None, n_min=None, acc=None,
                    verbose=False) -> tuple:
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        n_min = self.n_min if n_min is None else int(n_min)
        weights, lags, refused, notes = (members if members is not None
                                         else self.stack_members(target, cutoff, n_max, verbose=verbose))
        if len(weights) < n_min:
            raise NoMembers("%s: the fleet stack has %d member(s) and the floor is %d -- a one-member stack "
                            "is a remote site renamed. %s"
                            % (target, len(weights), n_min,
                               " | ".join("%s: %s" % kv for kv in sorted(refused.items())) or
                               "(no candidate scored)"))
        t0, n, num, den, cnt, screen, spans, _levels = (self.accumulate(target, weights, lags)
                                                        if acc is None else acc)
        thin = np.asarray(cnt) < n_min
        h, cov, mask = self.finish(num, den, thin=thin)
        info = dict(weights=weights, lags=lags, refusals=refused, alignment_notes=notes,
                    member_floor=_floor_rows(thin, n_min),
                    members=_member_rows(weights, lags, refused, notes),
                    weight_rule=COH.WEIGHT_RULE, weight_band_s=[100, 1000], align_band_s=[5, 20],
                    cutoff=cutoff, n_max=n_max, n_min=n_min,
                    coverage_by_channel=cov, member_count=_count_profile(cnt, len(weights)),
                    screen=_with_stack_steps(screen, h, mask),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="members edge-screened, time-aligned to the target, spike-screened against each "
                         "other and set on the fleet's own level before averaging; zero and mask False "
                         "where no member is sound")
        return self._write("stack", target, t0, h, mask, info, extra=spans)

    def build_obs(self, target: str, code=None) -> tuple:
        code = code or (self.cfg.get("observatory") or {}).get("code", "")
        t0, n = self.window(target)
        o, omask, ang = self.observatory_member(code, t0, n)
        h = {c: np.where(omask & np.isfinite(o[c]), o[c], 0.0) for c in H}
        w, _pairs = self.observatory_weight(code)
        info = dict(observatory=code, angle_deg=ang, fleet_coh=w,
                    distance_km=round(geo.distance_km(self.position(target), code), 1),
                    members=[dict(name=code, weight=w, lag_s=0.0, role="observatory")],
                    weights={code: w}, lags={code: 0.0}, refusals={},
                    alignment_notes={code: "never shifted: a delay on a single reference cancels in Z"},
                    frame="mean-horizontal-field of the observatory record (geographic X, Y turned)",
                    note="zero and mask False outside the archive's real samples")
        return self._write("obs", target, t0, h, omask, info)

    def build_stack_obs(self, target: str, code=None, members=None, acc=None, cutoff=None, n_max=None,
                        n_min=None) -> tuple:
        """The stack with the observatory as one more member at its own fleet weight, never shifted."""
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        n_min = self.n_min if n_min is None else int(n_min)
        code = code or (self.cfg.get("observatory") or {}).get("code", "")
        weights, lags, refused, notes = (members if members is not None
                                         else self.stack_members(target, cutoff, n_max))
        if len(weights) < n_min:
            raise NoMembers("%s: stack + observatory has %d site member(s) and the floor is %d. %s"
                            % (target, len(weights), n_min,
                               " | ".join("%s: %s" % kv for kv in sorted(refused.items())) or
                               "(no candidate scored)"))
        t0, n, num0, den0, cnt, screen, spans, levels = (self.accumulate(target, weights, lags)
                                                         if acc is None else acc)
        num = {c: num0[c].copy() for c in H}
        den = {c: den0[c].copy() for c in H}
        w_obs, _pairs = self.observatory_weight(code)
        o, omask, ang = self.observatory_member(code, t0, n)
        obs_rows = {}
        if w_obs is not None and np.isfinite(w_obs) and w_obs > 0:
            for c in H:
                x = np.where(np.isfinite(o[c]) & omask, np.asarray(o[c], float), np.nan)
                # matched one-sided against the members' established level, which it never enters
                off = one_sided_offset(x, levels[c], self.fs)
                x = x - off
                m = np.isfinite(x)
                obs_rows[c] = dict(
                    offset_rms_nt=round(float(np.sqrt(np.mean(off ** 2))), 3),
                    mask_edges=int(np.abs(np.diff(np.asarray(omask, bool).astype(np.int8))).sum()),
                    steps_per_million_before=round(steps_per_million(
                        np.where(np.isfinite(o[c]) & omask, np.asarray(o[c], float), np.nan)), 1),
                    steps_per_million_after=round(steps_per_million(x), 1))
                num[c][m] += w_obs * x[m]
                den[c][m] += w_obs
        # the floor counts SITE members only, per sample as per list: one site member and the observatory is
        # a remote site and an observatory, not a stack and one (Ben's ruling, 2026-09-17)
        thin = np.asarray(cnt) < n_min
        h, cov, mask = self.finish(num, den, thin=thin)
        allw = dict(weights)
        allw[code] = w_obs
        info = dict(weights=allw, lags=dict(lags, **{code: 0.0}), refusals=refused,
                    member_floor=_floor_rows(thin, n_min),
                    alignment_notes=dict(notes, **{code: "never shifted"}),
                    members=_member_rows(weights, lags, refused, notes, obs=code, w_obs=w_obs),
                    observatory=code, observatory_weight=w_obs, observatory_angle_deg=ang,
                    observatory_coverage=round(float(omask.mean()), 4),
                    weight_rule=COH.WEIGHT_RULE, weight_band_s=[100, 1000], align_band_s=[5, 20],
                    cutoff=cutoff, n_max=n_max, n_min=n_min,
                    coverage_by_channel=cov, member_count=_count_profile(cnt, len(weights)),
                    screen=_with_stack_steps(dict(screen, observatory={code: obs_rows}), h, mask),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="the observatory enters as one more member at its own fleet weight, is never "
                         "shifted and is not spike-screened, and is set on the members' established level "
                         "one-sidedly, which it never enters; zero and mask False where fewer than n_min "
                         "SITE members are sound, whatever the observatory holds there")
        return self._write("stack_obs", target, t0, h, mask, info, extra=spans)

    def build_site(self, target: str, kinds=KINDS_WITH_STORE, force=False, verbose=False) -> dict:
        """Every asked-for reference for one target, with one member-scoring pass and one member read pass."""
        out = {}
        if "remote" in kinds:
            try:
                if force or not self.path("remote", target).exists():
                    _p, info = self.build_remote(target)
                else:
                    info = json.loads(self.path("remote", target).with_suffix(".json")
                                      .read_text(encoding="utf-8"))
                out["remote"] = info
            except Exception as exc:
                out["remote"] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
        want = [k for k in ("stack", "stack_obs") if k in kinds]
        if want:
            need = force or any(not self.path(k, target).exists() for k in want)
            if need:
                members = self.stack_members(target, verbose=verbose)
                acc = None
                if len(members[0]) >= self.n_min:
                    acc = self.accumulate(target, members[0], members[1])
                for k in want:
                    try:
                        if force or not self.path(k, target).exists():
                            builder = self.build_stack if k == "stack" else self.build_stack_obs
                            _p, info = builder(target, members=members, acc=acc)
                        else:
                            info = json.loads(self.path(k, target).with_suffix(".json")
                                              .read_text(encoding="utf-8"))
                        out[k] = info
                    except Exception as exc:
                        out[k] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
                del acc
            else:
                for k in want:
                    out[k] = json.loads(self.path(k, target).with_suffix(".json")
                                        .read_text(encoding="utf-8"))
        if "obs" in kinds:
            try:
                if force or not self.path("obs", target).exists():
                    _p, info = self.build_obs(target)
                else:
                    info = json.loads(self.path("obs", target).with_suffix(".json")
                                      .read_text(encoding="utf-8"))
                out["obs"] = info
            except Exception as exc:
                out["obs"] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
        return out


def branch_rule(target, cands, n_considered, n_short, own_days, floor, coh_min=COH_MIN,
                coh_relax=COH_RELAX, nearest=None) -> dict:
    """The five-branch remote-site rule on a scored candidate list, with its branch and its reason.

    `cands` are the candidates that cleared the overlap floor, sorted by km. `nearest` is called only where
    nothing cleared the floor, and answers with the nearest site in the pool.
    """
    def desc(c):
        return ("%.0f km, overlap %.1f d (%.0f %% of %.1f d), events %s, coh %s over %d chunks"
                % (c["km"], c["overlap_days"], 100 * c["overlap_frac"], own_days, c["event_frac"],
                   c["coh"], c.get("chunks", 0)))

    if not cands:
        return dict(target=target, name=(nearest() if nearest else None), branch=5, coh=None,
                    reason="branch 5 (nearest by km): no candidate scored -- %d of %d candidates fall "
                           "below the %.1f d overlap floor for a %.1f d record"
                           % (n_short, n_considered, floor, own_days))
    clean = [c for c in cands if c["clean"]]
    coh_ok = {c["name"] for c in cands if c["coh"] is not None and c["coh"] >= coh_min}
    good = [c for c in clean if c["name"] in coh_ok]
    best_clean = max((c["coh"] for c in clean if c["coh"] is not None), default=None)
    full = [c for c in good if c["overlap_frac"] >= 0.9]
    if full:
        b = min(full, key=lambda c: c["km"])
        return dict(target=target, name=b["name"], branch=1, coh=b["coh"],
                    reason="branch 1 (clean, coh >= %s, overlap >= 90 %%): nearest of %d: %s"
                           % (coh_min, len(full), desc(b)))
    if good:
        b = max(good, key=lambda c: c["overlap_days"])
        return dict(target=target, name=b["name"], branch=2, coh=b["coh"],
                    reason="branch 2 (clean, coh >= %s, longest overlap): branch 1 empty, the best overlap "
                           "among the coherent clean candidates is %.0f %%; %s"
                           % (coh_min, 100 * max(c["overlap_frac"] for c in good), desc(b)))
    relaxed = [c for c in clean if c["coh"] is not None and c["coh"] >= coh_relax]
    if relaxed:
        b = max(relaxed, key=lambda c: c["coh"])
        return dict(target=target, name=b["name"], branch=3, coh=b["coh"],
                    reason="branch 3 (clean, gate relaxed %s -> %s): none of the %d clean candidates "
                           "reaches %s (best %s); %s"
                           % (coh_min, coh_relax, len(clean), coh_min, best_clean, desc(b)))
    why12 = ("none of the %d candidates is clean" % len(cands) if not clean else
             "coherence unmeasurable for every clean candidate" if best_clean is None else
             "no clean candidate reaches even %s (best %s)" % (coh_relax, best_clean))
    coherent = [c for c in cands if c["name"] in coh_ok]
    if coherent:
        b = min(coherent, key=lambda c: (c["event_frac"] if c["event_frac"] is not None else 1.0))
        return dict(target=target, name=b["name"], branch=4, coh=b["coh"],
                    reason="branch 4 (fewest events among the %d at coh >= %s): branches 1-3 empty because "
                           "%s; %s" % (len(coherent), coh_min, why12, desc(b)))
    b = cands[0]
    return dict(target=target, name=b["name"], branch=5, coh=b["coh"],
                reason="branch 5 (nearest by km): branches 1-4 empty because %s and no candidate reaches "
                       "coherence %s, and decisions.csv names no remote; %s" % (why12, coh_min, desc(b)))


def _floor_rows(thin, n_min) -> dict:
    """What the per-sample member floor cost: how many samples carried fewer than n_min SITE members."""
    t = np.asarray(thin, bool)
    return dict(n_min=int(n_min), counts="site members only", samples_below=int(t.sum()),
                frac_below=round(float(t.mean()), 5) if len(t) else 0.0,
                note="a sample carried by fewer than n_min site members is zero and its mask False, on the "
                     "same rule the member list is judged on: a one-member stack is a remote site renamed, "
                     "and one site member with the observatory is a remote site with an observatory")


def _merge_spans(per_member: list) -> np.ndarray:
    """Every member's bridged spans on one channel as one sorted (k, 2) int32 array, overlaps merged.

    The union and not one row per member: the figure marks where the screen fired, and at this density the
    marks of two members at the same second are one mark.
    """
    flat = [s for spans in per_member for s in spans]
    if not flat:
        return np.zeros((0, 2), np.int32)
    flat.sort()
    out = [list(flat[0])]
    for a, b in flat[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.asarray(out, np.int32)


def _with_stack_steps(screen: dict, h: dict, mask) -> dict:
    """The screen record with the finished stack's own steps per million added beside the members' median.

    The two numbers the stack is judged on sit in one place, so the check reads them and recomputes nothing.
    """
    out = dict(screen)
    steps = dict(out.get("steps_per_million") or {})
    m = np.asarray(mask, bool)
    steps["stack"] = {c: round(steps_per_million(np.where(m, h[c], np.nan)), 1) for c in H}
    out["steps_per_million"] = steps
    return out


def _member_rows(weights, lags, refusals, notes, obs=None, w_obs=None) -> list:
    rows = [dict(name=d, weight=w, lag_s=(lags or {}).get(d), role="member",
                 note=(notes or {}).get(d)) for d, w in weights.items()]
    if obs is not None:
        rows.append(dict(name=obs, weight=w_obs, lag_s=0.0, role="observatory",
                         note="never shifted"))
    rows += [dict(name=d, weight=None, lag_s=None, role="refused", note=r)
             for d, r in sorted((refusals or {}).items())]
    return rows


def _count_profile(cnt, n_members) -> dict:
    """How ragged the member coverage is, which is what costs a stack.

    Each member is demeaned over its own record, so when the member set changes mid-record the composite's
    baseline steps by the difference of the members' local offsets. Under remote reference that costs
    efficiency and not bias, but a target with a ragged profile is one where the single remote site may beat
    the stack, so the profile is written out.
    """
    c = np.asarray(cnt)
    return dict(n_members=int(n_members), min=int(c.min()), median=int(np.median(c)),
                frac_zero=round(float((c == 0).mean()), 4),
                frac_lt_half=round(float((c < max(1, n_members // 2)).mean()), 4),
                frac_full=round(float((c == n_members).mean()), 4))


def build_store(survey, sites=None, rate=1, kinds=KINDS_WITH_STORE, force=False, verbose=True,
                spec_store=None, store=None) -> dict:
    """Build every reference of a survey at one rate. Returns {site: {kind: sidecar or error}}.

    `spec_store` is a 1 Hz store whose decisions a higher-rate store re-uses: the pool, the remote, the
    members, the weights and the lags come from it and only the arrays are read on this store's grid.
    `store` is an existing Store to build into, so a caller that set its own thresholds keeps them.
    """
    st = store if store is not None else Store(survey, sites, rate)
    st.dir.mkdir(parents=True, exist_ok=True)
    pool, table = st.clean_pool()
    _write_json(st.dir / "pool.json", dict(
        pool=pool, rate_hz=st.fs, thresholds=TR.params(st.cfg), built_at=_now(),
        rows=table.to_dict("records")))
    out = {}
    for s in (sites if sites is not None else st.sites):
        if verbose:
            print("   %s" % s, flush=True)
        if spec_store is not None:
            out[s] = _build_from_spec(st, spec_store, s, kinds, force)
        else:
            out[s] = st.build_site(s, kinds=kinds, force=force, verbose=verbose)
    return out


def _build_from_spec(st: Store, spec: Store, target: str, kinds, force) -> dict:
    """One site's references at st's rate, with every decision taken from the 1 Hz store's sidecars."""
    out = {}
    for kind in kinds:
        p = st.path(kind, target)
        if p.exists() and not force:
            out[kind] = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            continue
        src = spec.path(kind, target).with_suffix(".json")
        if not src.exists():
            out[kind] = dict(error="the %g Hz store holds no %s reference for %s" % (spec.fs, kind, target))
            continue
        spec_info = json.loads(src.read_text(encoding="utf-8"))
        try:
            if kind == "remote":
                choice = dict(name=spec_info["remote"], branch=spec_info.get("branch"),
                              coh=spec_info.get("coh"), source=spec_info.get("source"),
                              source_detail=spec_info.get("source_detail"),
                              reason=spec_info.get("reason"), rule_name=spec_info.get("rule_name"),
                              rule_branch=spec_info.get("rule_branch"),
                              rule_coh=spec_info.get("rule_coh"), candidates=[])
                _q, info = st.build_remote(target, choice=choice)
            elif kind == "obs":
                _q, info = st.build_obs(target)
            else:
                weights = {k: v for k, v in (spec_info.get("weights") or {}).items()
                           if v is not None and k != spec_info.get("observatory")}
                lags = {k: float(v or 0.0) for k, v in (spec_info.get("lags") or {}).items()
                        if k in weights}
                members = (weights, lags, spec_info.get("refusals") or {},
                           spec_info.get("alignment_notes") or {})
                builder = st.build_stack if kind == "stack" else st.build_stack_obs
                _q, info = builder(target, members=members)
            info["specification_from"] = "%g Hz store" % spec.fs
            _write_json(p.with_suffix(".json"), info)
            out[kind] = info
        except Exception as exc:
            out[kind] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
    return out


def load_reference(kind: str, site: str, rate, work_root) -> tuple:
    """(t0, {'Hx','Hy'}, mask, sidecar) for one reference from the store, or Nones for the single station."""
    if kind == "single":
        return None, None, None, dict(kind="single")
    p = Path(work_root) / "references" / ("%dhz" % int(rate)) / ("%s_%s.npz" % (kind, site))
    if not p.exists():
        raise FileNotFoundError("%s has not been built; run the reference store first" % p)
    z = np.load(p, allow_pickle=False)
    t0 = int(z["t0"][0])
    h = {c: np.asarray(z[c], float) for c in H}
    mask = np.asarray(z["mask"], bool)
    z.close()
    info = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    return t0, h, mask, info


def bridged_spans(kind: str, site: str, rate, work_root, channel="Hx") -> np.ndarray:
    """The spans the member spike screen bridged on one channel, as [start, end] pairs on the reference's
    own grid, merged over the members. An empty (0, 2) array where the store carries none.
    """
    p = Path(work_root) / "references" / ("%dhz" % int(rate)) / ("%s_%s.npz" % (kind, site))
    if not p.exists():
        return np.zeros((0, 2), np.int32)
    z = np.load(p, allow_pickle=False)
    key = "bridged_%s" % channel
    got = np.asarray(z[key], np.int32) if key in z.files else np.zeros((0, 2), np.int32)
    z.close()
    return got.reshape(-1, 2)


def reference_station_id(kind: str, info: dict, survey_code="") -> str:
    """The station id a reference is written into the MTH5 under: REMOTE_<site>, STACK, OBS_<code>, STACK_OBS."""
    if kind == "remote":
        return "REMOTE_%s" % info.get("remote", "")
    if kind == "stack":
        return "STACK"
    if kind == "obs":
        return "OBS_%s" % (info.get("observatory") or survey_code or "")
    if kind == "stack_obs":
        return "STACK_OBS"
    return ""


def cache_sidecar(site, work_root) -> dict:
    return cache.sidecar(site, work_root)
