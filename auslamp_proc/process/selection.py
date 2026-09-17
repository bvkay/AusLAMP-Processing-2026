"""Which hours a 10 Hz pass is run on: the hour score, the three selections and the random control.

Ben's ruling, 2026-09-17: the 10 Hz product only needs the most coherent parts, and the full record makes no
sense at 10 Hz. This is a capability of the 10 Hz lane alone; the 1 Hz rule, the 1 Hz names and the 1 Hz path
are not touched by it.

    score_hours       per whole UTC hour, the squared coherence of the two impedance pairs (Ex with Hy, Ey
                      with Hx) over SCORE_BAND_S = 1-30 s, Welch at SCORE_NPERSEG = 2048 samples at 10 Hz
                      with the segment mean removed (scipy detrend "constant"), as the median over the band
                      (site.masks.band_coherence). The two pairs are averaged where both lines are live on
                      that UTC day in workbook 02's elines table and the live pair stands alone where the
                      other line is dead; `dead` is the one state that takes a pair out, so a day workbook
                      02 could not score leaves both pairs in and the finiteness of the hour decides. An
                      hour holding a non-finite sample in a pair scores NaN on it, and an hour with no
                      finite pair scores NaN and is not a candidate. The series is written to
                      <work_root>/<site>/hour_scores_10hz.csv.
    selections        the best SELECT10 = 0.05, 0.10 and 0.25 of the scored hours, keyed f05, f10 and f25,
                      plus RANDOM_FRACTION = 0.25 of the same pool drawn without replacement under
                      SEED = 20260916 and keyed r25. The choosing is workbook 05's
                      site.masks.select_hours, so the selection and its control differ only in how the
                      hours are chosen. A selection that does not beat its random control buys nothing.
    write_selection   the chosen hours per key as <work_root>/<site>/hour_selection_10hz.json, each with
                      its fraction, its seed, its score threshold and its hour intervals in unix seconds.
                      The pass rebuilds the sample mask from the intervals rather than carrying a 52 M
                      element boolean file per selection.

The hours are whole UTC hours, so a kept hour is 3,600 s = 36,000 samples at 10 Hz and meets the
transients.MIN_SEGMENT_S = 3,600 s run floor exactly: an isolated kept hour survives the floor as one Aurora
run, and adjacent kept hours merge into one longer run. A kept hour the transient mask or a gap cuts into is
shorter than the floor and is dropped, which the product's own floor_dropped_frac reports. Scattered hours
are admissible here because the 10 Hz product is the short end, below the 16 s join, and no deep level is
delivered from it.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..raw import cache
from ..site import masks as MK

HOUR_S = 3600.0
SCORE_BAND_S = (1.0, 30.0)
SCORE_NPERSEG_10HZ = 2048           # at 10 Hz; scaled with the rate by nperseg_for()
SELECT10 = (0.05, 0.10, 0.25)
RANDOM_FRACTION = 0.25
SEED = 20260916
PAIRS = (("xy", "Ex", "Hy"), ("yx", "Ey", "Hx"))
DEAD_STATE = "dead"                 # the one elines state that takes a pair out of the score
WHOLE = "whole"

SCORES_NAME = "hour_scores_10hz.csv"
SELECTION_NAME = "hour_selection_10hz.json"

SCORE_COLUMNS = ["hour", "t_start", "t_end", "utc", "score", "score_xy", "score_yx", "pairs",
                 "Ex_state", "Ey_state"]


def key_for(fraction, random=False) -> str:
    """The selection tag of one fraction: f05, f10, f25 for the ranked ones and r25 for the control."""
    return "%s%02d" % ("r" if random else "f", int(round(100 * float(fraction))))


def nperseg_for(fs) -> int:
    """The Welch segment at `fs` spanning the same seconds SCORE_NPERSEG_10HZ does at 10 Hz."""
    return int(round(SCORE_NPERSEG_10HZ * float(fs) / 10.0))


def scores_path(work_root, site) -> Path:
    return Path(work_root) / str(site) / SCORES_NAME


def selection_path(work_root, site) -> Path:
    return Path(work_root) / str(site) / SELECTION_NAME


def _day_states(elines) -> list:
    """[(t_start, t_end, Ex_state, Ey_state)] from an elines table, or an empty list."""
    if elines is None or not len(elines):
        return []
    return [(float(r.t_start), float(r.t_end), str(r.Ex_state), str(r.Ey_state))
            for r in elines.itertuples()]


def _state_at(days, t) -> tuple:
    """The two line states covering time `t`, or ("unknown", "unknown") where no day covers it."""
    for ta, tb, sx, sy in days:
        if ta <= t < tb:
            return sx, sy
    return "unknown", "unknown"


def hour_grid(t0, n, fs, hour_s=HOUR_S) -> tuple:
    """(the first sample of each whole UTC hour the record holds, the unix second it starts at).

    The record starts where the logger started, so the first whole UTC hour is the first hour boundary at or
    after t0 and the last is the last boundary a whole hour before the end.
    """
    step = int(round(hour_s * float(fs)))
    t_first = float(np.ceil(float(t0) / hour_s) * hour_s)
    i0 = int(round((t_first - float(t0)) * float(fs)))
    starts = np.arange(i0, int(n) - step + 1, step, dtype=int)
    return starts, (float(t0) + starts / float(fs))


def score_hours(t0, arrays, fs=10.0, elines=None, band_s=SCORE_BAND_S, nperseg=None,
                hour_s=HOUR_S) -> pd.DataFrame:
    """The hour score table of one record: one row per whole UTC hour.

    `arrays` carries Ex, Ey, Hx and Hy as laid. Each pair is scored by site.masks.band_coherence, the median
    squared coherence over `band_s`, and the row's score is the mean of the pairs whose electric line is not
    dead on that UTC day. An hour holding a non-finite sample in a pair scores NaN on that pair, and an hour
    with no finite pair scores NaN and is not a candidate.
    """
    nperseg = int(nperseg or nperseg_for(fs))
    step = int(round(float(hour_s) * float(fs)))
    starts, t_starts = hour_grid(t0, len(arrays["Hx"]), fs, hour_s)
    days = _day_states(elines)
    rows = []
    for h, (i0, ta) in enumerate(zip(starts, t_starts)):
        sx, sy = _state_at(days, ta + 0.5 * float(hour_s))
        state = {"xy": sx, "yx": sy}
        vals, used = {}, []
        for comp, line, hchan in PAIRS:
            if state[comp] == DEAD_STATE:
                vals[comp] = np.nan
                continue
            e = np.asarray(arrays[line][i0:i0 + step], float)
            b = np.asarray(arrays[hchan][i0:i0 + step], float)
            if not (np.isfinite(e).all() and np.isfinite(b).all()):
                vals[comp] = np.nan
                continue
            v = MK.band_coherence(e, b, float(fs), band_s, nperseg)
            vals[comp] = v
            if np.isfinite(v):
                used.append(v)
        rows.append(dict(hour=int(h), t_start=int(round(ta)), t_end=int(round(ta + float(hour_s))),
                         utc=pd.Timestamp(ta, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                         score=(float(np.mean(used)) if used else np.nan),
                         score_xy=vals.get("xy", np.nan), score_yx=vals.get("yx", np.nan),
                         pairs=len(used), Ex_state=sx, Ey_state=sy))
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def site_scores(sv, site, rate=10, force=False, elines=None) -> pd.DataFrame:
    """One site's hour score table, read from <work_root>/<site>/hour_scores_10hz.csv or computed and written.

    The record is read from the cache as laid: no sign and no frame. A sign flips the phase of a coherence
    and not its magnitude, and the rotation mixes Hx into Hy, so the score of an hour is the same number
    whichever frame the pass is later run in.
    """
    work = Path(sv.cfg["work_root"])
    out = scores_path(work, site)
    if out.exists() and not force:
        return pd.read_csv(out)
    if elines is None:
        p = work / site / "elines.csv"
        elines = pd.read_csv(p) if p.exists() else None
    t0, arrays, _meta = cache.load(site, work, rate)
    # the table is rounded BEFORE it is selected from as well as before it is written, so the selection a
    # fresh computation makes and the selection a later read of the CSV makes are the same set of hours.
    # Two hours whose coherence differs in the eighth decimal rank one way in memory and the other way
    # after a round trip through six decimals, and the two selections then differ by one hour.
    table = score_hours(t0, arrays, float(rate), elines=elines).round(6)
    del arrays
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def selections(table, t0, n, fs=10.0, fractions=SELECT10, random_fraction=RANDOM_FRACTION, seed=SEED,
               hour_s=HOUR_S) -> dict:
    """{tag: {fraction, random, seed, threshold, n_hours, hours}} for one site's score table.

    `hours` is [(t_start, t_end)] in unix seconds. The ranked selections and the random control are both made
    by site.masks.select_hours, the function workbook 05 selects hours with, so the control differs from the
    selection only in how the hours are chosen.
    """
    step = int(round(float(hour_s) * float(fs)))
    tc = np.asarray(table.t_start, float)
    tc = (tc - float(t0)) * float(fs) + step / 2.0          # the centre of each hour, in samples
    score = np.asarray(table.score, float)
    t_start = np.asarray(table.t_start, float)
    out = {}
    wanted = [(f, False) for f in fractions] + ([(random_fraction, True)] if random_fraction else [])
    for fraction, is_random in wanted:
        rng = np.random.default_rng(int(seed)) if is_random else None
        keep, thr, k = MK.select_hours(tc, score, fraction, step, int(n), rng=rng)
        chosen = np.flatnonzero([bool(keep[int(max(0, c - step / 2)):int(min(n, c + step / 2))].all())
                                 for c in tc]) if k else np.array([], int)
        out[key_for(fraction, is_random)] = dict(
            fraction=float(fraction), random=bool(is_random), seed=int(seed),
            threshold=(None if not np.isfinite(thr) else round(float(thr), 6)),
            n_hours=int(k), n_candidates=int(np.isfinite(score).sum()),
            hours=[[int(t_start[i]), int(t_start[i] + hour_s)] for i in chosen])
    return out


def write_selection(work_root, site, sel: dict, band_s=SCORE_BAND_S, hour_s=HOUR_S, rate=10) -> Path:
    """Write <work_root>/<site>/hour_selection_10hz.json and return its path."""
    p = selection_path(work_root, site)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(site=str(site), rate_hz=float(rate), band_s=list(band_s),
                                 hour_s=float(hour_s), nperseg=nperseg_for(rate),
                                 selections=sel), indent=1, default=str), encoding="utf-8")
    return p


def read_selection(work_root, site) -> dict:
    p = selection_path(work_root, site)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def mask_from_hours(hours, t0, n, fs) -> np.ndarray:
    """The sample mask of a list of [t_start, t_end] hour intervals in unix seconds."""
    keep = np.zeros(int(n), bool)
    for ta, tb in hours:
        a = int(round((float(ta) - float(t0)) * float(fs)))
        b = int(round((float(tb) - float(t0)) * float(fs)))
        a, b = max(0, a), min(int(n), b)
        if b > a:
            keep[a:b] = True
    return keep


def build(sv, site, rate=10, fractions=SELECT10, random_fraction=RANDOM_FRACTION, seed=SEED,
          force=False) -> dict:
    """One site's hour scores and its selections, both written. Returns the selection dict."""
    work = Path(sv.cfg["work_root"])
    table = site_scores(sv, site, rate=rate, force=force)
    t0, n = MK.span(sv, site, rate)
    sel = selections(table, t0, n, float(rate), fractions=fractions, random_fraction=random_fraction,
                     seed=seed)
    write_selection(work, site, sel, rate=rate)
    return sel


def summary_rows(work_root, sites, keys=None) -> pd.DataFrame:
    """One row per site and selection tag: the hours scored, the hours kept and the days they come to."""
    rows = []
    for s in sites:
        d = read_selection(work_root, s)
        for tag, v in sorted((d.get("selections") or {}).items()):
            if keys is not None and tag not in keys:
                continue
            rows.append(dict(site=s, selection=tag, fraction=v["fraction"], random=v["random"],
                             seed=v["seed"], threshold=v["threshold"], hours_scored=v["n_candidates"],
                             hours_kept=v["n_hours"], days_kept=round(v["n_hours"] / 24.0, 2)))
    return pd.DataFrame(rows)
