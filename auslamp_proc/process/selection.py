"""The one rule for which hours a pass is run on: the hour score, the longest coherent stretch, its control.

One function in one place, used by workbook 03's 10 Hz pass, by workbook 04's windows and by workbook 04's
recipe, so that every selection of hours in the package is the same selection.

    score_hours       per whole UTC hour of the 1 Hz cache, the median squared coherence of Ex with Hy and of
                      Ey with Hx over SCORE_BAND_S = 20-200 s, Welch at SCORE_SEGMENT_S = 1,024 s segments.
                      An hour holding a non-finite sample in a pair scores NaN on that pair and cannot be
                      inside a stretch. The series is written to <work_root>/<site>/hour_scores.csv.
    a stretch         a contiguous run of whole UTC hours in which every line the caller names reads above
                      WINDOW_COH = 0.5. Both lines are named for a whole selection; a row-wise window uses
                      that row's line alone.
    the selection     the longest stretch, and where it runs longer than STRETCH_MAX_H = 48 h the 48
                      contiguous hours inside it whose mean score is the highest.
    the control       a run of the same number of whole hours placed at random elsewhere in the record
                      under SEED = 20260916 and not overlapping the selection. A selection that does not
                      beat its control bought efficiency and not a different answer.
    the refusal       a chosen stretch is measured against the site's own transient mask and its gaps, and
                      one the mask leaves with no run of transients.MIN_SEGMENT_S = 3,600 s is refused here,
                      before any pass, with the sentence "the mask leaves no run of 3600 s inside the <n> h
                      stretch". The length itself is no bar: a 3 h stretch the mask leaves whole is passed.
                      The control is measured and refused the same way and independently.

The record is read decided. raw.cache.load_decided applies the decisions.csv row through
process.frame.apply_decisions before the hours are scored: e_exchange pairs an electric line with the wrong
magnetic channel and e_shift_s is a quarter of a cycle at 4 s, and both move an E-H coherence.

h_lender IS applied here, where references.Store.rotated leaves it out, because the two answer different
questions: this score asks which hours each electric line followed the magnetic field the pass will actually
use, and at a borrowing site that field is the lender's, while the clean pool asks whether a site's own
record is fit for another site to reference and reads it without the lender.

The hours are scored on the 1 Hz cache whatever rate the pass runs at, because 200 s is measured on the long
record. They are whole UTC hours, so a kept hour is 3,600 s and meets the transients.MIN_SEGMENT_S = 3,600 s
run floor exactly: an isolated kept hour survives as one Aurora run and adjacent kept hours merge into one
longer run.

A pass carries its tag between the rate and the parameter set, <site>_<kind>_10hz_<tag>_<params>.edi, and the
same tag in the ledger's `selection` column and in the file's own `selection=` line. The tags are `stretch`
and `control`; `whole` is the untagged pass over the whole record.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..raw import cache
from ..site import masks as MK
from . import transients as TR

HOUR_S = 3600.0
SCORE_BAND_S = (20.0, 200.0)        # the band the hourly coherence is read over, in s
SCORE_SEGMENT_S = 1024              # the Welch segment of the hourly coherence, in s
SCORE_RATE = 1                      # the rate the hours are scored at, whatever rate the pass runs at
WINDOW_COH = 0.5                    # a line is inside a stretch where its squared coherence is above this
STRETCH_MAX_H = 48                  # a stretch longer than this is cut to its best 48 contiguous hours
SEED = 20260916                     # the draw the control of the same length is placed under

STRETCH = "stretch"                 # the tag of the selected stretch
CONTROL = "control"                 # ... and of the stretch that controls it
WHOLE = "whole"                     # the untagged pass over the whole record
TAGS = (STRETCH, CONTROL)

PAIRS = (("xy", "Ex", "Hy"), ("yx", "Ey", "Hx"))
LINES = ("xy", "yx")                # both recorded lines; a row-wise window names one of them
ROW_LINE = {"x": "xy", "y": "yx"}   # the line each impedance row is estimated on

SCORES_NAME = "hour_scores.csv"
SELECTION_NAME = "hour_selection.json"

SCORE_COLUMNS = ["hour", "t_start", "t_end", "utc", "coh_xy", "coh_yx", "score"]


def _hm(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def nperseg_for(fs) -> int:
    """The Welch segment at `fs` spanning SCORE_SEGMENT_S = 1,024 s."""
    return int(round(float(SCORE_SEGMENT_S) * float(fs)))


def scores_path(work_root, site) -> Path:
    return Path(work_root) / str(site) / SCORES_NAME


def selection_path(work_root, site) -> Path:
    return Path(work_root) / str(site) / SELECTION_NAME


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


# ------------------------------------------------------------------ the hour score

def score_hours(t0, arrays, fs=float(SCORE_RATE), band_s=SCORE_BAND_S, nperseg=None,
                hour_s=HOUR_S) -> pd.DataFrame:
    """One row per whole UTC hour: each recorded line against the magnetic channel it couples to.

    `arrays` carries Ex, Ey, Hx and Hy. Ex is scored against Hy and Ey against Hx, each as the median squared
    coherence over `band_s`. `score` is the mean of the finite pairs and is what ranks the hours inside a
    stretch longer than STRETCH_MAX_H.
    """
    nperseg = int(nperseg or nperseg_for(fs))
    step = int(round(float(hour_s) * float(fs)))
    starts, t_starts = hour_grid(t0, len(arrays["Hx"]), fs, hour_s)
    rows = []
    for h, (i0, ta) in enumerate(zip(starts, t_starts)):
        row = dict(hour=int(h), t_start=int(round(ta)), t_end=int(round(ta + float(hour_s))), utc=_hm(ta))
        got = []
        for comp, line, hchan in PAIRS:
            e = np.asarray(arrays[line][i0:i0 + step], float)
            b = np.asarray(arrays[hchan][i0:i0 + step], float)
            v = (MK.band_coherence(e, b, float(fs), band_s, nperseg)
                 if (np.isfinite(e).all() and np.isfinite(b).all()) else np.nan)
            row["coh_%s" % comp] = v
            if np.isfinite(v):
                got.append(v)
        row["score"] = float(np.mean(got)) if got else np.nan
        rows.append(row)
    out = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    out.attrs["band_s"] = list(band_s)
    out.attrs["nperseg"] = int(nperseg)
    out.attrs["rate_hz"] = float(fs)
    return out


def site_scores(sv, site, rate=SCORE_RATE, force=False, arrays=None, t0=None) -> pd.DataFrame:
    """One site's hour score table, read from <work_root>/<site>/hour_scores.csv or computed and written.

    The record is read decided, through raw.cache.load_decided: the exchange of two electric lines and a
    shift of one of them both move an E-H coherence, so an hour scored on the record as laid is not the hour
    the pass estimates on. `arrays` and `t0` score a record the caller already holds, which is what a variant
    cache needs, and nothing is written for one.
    """
    work = Path(sv.cfg["work_root"])
    out = scores_path(work, site)
    if arrays is not None:
        return score_hours(int(t0), arrays, float(rate)).round(6)
    if out.exists() and not force:
        return pd.read_csv(out)
    t0, arrays, _applied, _rec = cache.load_decided(site, sv, rate=int(rate))
    # the table is rounded before it is read from as well as before it is written, so a fresh computation and
    # a later read of the CSV find the same hours: two hours differing in the eighth decimal rank one way in
    # memory and the other way after a round trip through six decimals
    table = score_hours(int(t0), arrays, float(rate)).round(6)
    del arrays
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


# ------------------------------------------------------------------ the stretch

def _above(table: pd.DataFrame, coh_min, lines) -> np.ndarray:
    """The hours in which every named line reads above `coh_min` and is finite."""
    ok = np.ones(len(table), bool)
    for comp in lines:
        v = np.asarray(table["coh_%s" % comp], float)
        ok &= np.isfinite(v) & (v > float(coh_min))
    return ok


def runs_above(table: pd.DataFrame, coh_min=WINDOW_COH, lines=LINES) -> list:
    """[(hours, t_start, t_end)] of every contiguous run in which every named line is above `coh_min`.

    Longest first, and the earliest of any that tie, so a record carrying several runs of one length returns
    the same one on every call.
    """
    ok = _above(table, coh_min, lines)
    ts = np.asarray(table.t_start, float)
    te = np.asarray(table.t_end, float)
    out, k = [], 0
    while k < len(ok):
        if not ok[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        out.append((int(j - k + 1), int(ts[k]), int(te[j])))
        k = j + 1
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def _row_score(table: pd.DataFrame, lines) -> np.ndarray:
    """The score the hours inside a long stretch are ranked on: the named lines, averaged."""
    cols = [np.asarray(table["coh_%s" % c], float) for c in lines]
    return np.nanmean(np.vstack(cols), axis=0) if cols else np.asarray(table.score, float)


def _best_inside(table: pd.DataFrame, t_start, t_end, max_h, lines):
    """(t_start, t_end) of the `max_h` contiguous hours inside a run whose mean score is the highest."""
    ts = np.asarray(table.t_start, float)
    idx = np.flatnonzero((ts >= float(t_start)) & (ts < float(t_end)))
    v = _row_score(table, lines)[idx]
    k = int(max_h)
    best_i, best_v = 0, -np.inf
    for i in range(0, len(idx) - k + 1):
        m = float(np.mean(v[i:i + k]))
        if m > best_v:
            best_i, best_v = i, m
    a = idx[best_i]
    b = idx[best_i + k - 1]
    return int(ts[a]), int(np.asarray(table.t_end, float)[b]), best_v


def longest_stretch(table: pd.DataFrame, coh_min=WINDOW_COH, lines=LINES, max_h=STRETCH_MAX_H) -> dict:
    """The selection: the longest stretch above `coh_min`, cut to its best `max_h` hours where it is longer.

    The returned dict carries the bounds, the hours, the threshold and the reason, and an empty stretch where
    no hour reads above the threshold on every named line.
    """
    runs = runs_above(table, coh_min, lines)
    scored = int(np.isfinite(_row_score(table, lines)).sum())
    above = int(_above(table, coh_min, lines).sum())
    band = table.attrs.get("band_s", list(SCORE_BAND_S))
    named = " and ".join(lines)
    if not runs:
        return dict(rule=STRETCH, t_start=None, t_end=None, hours=0, n_runs=0, n_hours_above=above,
                    n_hours_scored=scored, coh_min=float(coh_min), lines=list(lines), cut=False,
                    mean_score=float("nan"), runs=[],
                    reason="no hour reads above %.2f on %s over %g-%g s (%d hour(s) scored)"
                           % (coh_min, named, band[0], band[1], scored))
    hours, ta, tb = runs[0]
    cut, mean_score = False, float(np.nanmean(_row_score(
        table, lines)[np.flatnonzero((np.asarray(table.t_start, float) >= ta)
                                     & (np.asarray(table.t_start, float) < tb))]))
    if max_h and hours > int(max_h):
        ta, tb, mean_score = _best_inside(table, ta, tb, int(max_h), lines)
        cut, hours = True, int(max_h)
    tail = ("; the longest run is %d h and the %d contiguous hours inside it with the highest mean score are "
            "taken" % (runs[0][0], int(max_h))) if cut else ""
    return dict(rule=STRETCH, t_start=int(ta), t_end=int(tb), hours=int(hours), n_runs=len(runs),
                n_hours_above=above, n_hours_scored=scored, coh_min=float(coh_min), lines=list(lines),
                cut=cut, mean_score=round(float(mean_score), 6), runs=runs,
                reason="the longest contiguous stretch with %s above %.2f over %g-%g s: %d h, %s to %s UTC, "
                       "of %d hour(s) above on %d scored%s"
                       % (named, coh_min, band[0], band[1], int(hours), _hm(ta), _hm(tb), above, scored,
                          tail))


def control_stretch(t0, n, stretch: dict, seed=SEED, fs=float(SCORE_RATE)) -> dict:
    """A run of the same number of whole UTC hours placed at random elsewhere, not overlapping the selection.

    The draw is made on the same whole-UTC-hour grid the selection is chosen on, so the control is the same
    kind of object as the stretch it controls: a run of whole hours, each 3,600 s, which is the unit both the
    hour score and the transients.MIN_SEGMENT_S run floor are defined on. A control placed at an arbitrary
    sample would start mid-hour and span one more partial hour than the stretch.

    Where the record holds no second placement of that length clear of the selection, no control is returned
    and the reason says so: a stretch that cannot be controlled cannot be judged against one.
    """
    hours = int(round((float(stretch["t_end"]) - float(stretch["t_start"])) / HOUR_S))
    span = hours * HOUR_S
    _starts, t_starts = hour_grid(t0, n, fs)
    end = float(t0) + int(n) / float(fs)
    free = [t for t in t_starts
            if t + span <= end and (t + span <= float(stretch["t_start"])
                                    or t >= float(stretch["t_end"]))]
    if not free:
        return dict(rule=CONTROL, t_start=None, t_end=None, hours=0, seed=int(seed),
                    coh_min=float(stretch.get("coh_min", WINDOW_COH)),
                    lines=list(stretch.get("lines", LINES)),
                    reason="the record holds no second run of %d whole hour(s) clear of the selection, so "
                           "the stretch carries no control" % hours)
    rng = np.random.default_rng(int(seed))
    ta = int(free[int(rng.integers(len(free)))])
    tb = int(ta + span)
    return dict(rule=CONTROL, t_start=ta, t_end=tb, hours=hours, seed=int(seed),
                coh_min=float(stretch.get("coh_min", WINDOW_COH)), lines=list(stretch.get("lines", LINES)),
                reason="a run of the same %d whole hour(s) placed at random elsewhere in the record and not "
                       "overlapping the selection, seed %d: %s to %s UTC"
                       % (hours, int(seed), _hm(ta), _hm(tb)))


def named_stretch(spec: str) -> dict:
    """`window:<ISO UTC start>/<hours>` as a stretch, or a dict carrying the reason it does not parse."""
    body = str(spec).split(":", 1)[1] if ":" in str(spec) else ""
    start, _, hours = body.rpartition("/")
    try:
        ta = int(pd.Timestamp(start.strip(), tz="UTC").timestamp())
        h = float(hours)
        if h <= 0:
            raise ValueError("the hour count is not positive")
    except Exception as exc:
        return dict(rule=str(spec), t_start=None, t_end=None, hours=0,
                    reason="%s does not read as window:<ISO UTC start>/<hours> (%s)" % (spec, exc))
    return dict(rule=str(spec), t_start=ta, t_end=int(ta + h * HOUR_S), hours=float(h),
                reason="the stretch named in the recipe: %.4g h from %s UTC" % (h, _hm(ta)))


# ------------------------------------------------------------------ the mask and the store

def mask_from_hours(hours, t0, n, fs) -> np.ndarray:
    """The sample mask of a list of [t_start, t_end] intervals in unix seconds."""
    keep = np.zeros(int(n), bool)
    for ta, tb in hours:
        a = int(round((float(ta) - float(t0)) * float(fs)))
        b = int(round((float(tb) - float(t0)) * float(fs)))
        a, b = max(0, a), min(int(n), b)
        if b > a:
            keep[a:b] = True
    return keep


def stretch_mask(t0, n, stretch: dict, fs=1.0) -> np.ndarray:
    """The sample mask of one stretch on a record of `n` samples starting at `t0`, at `fs`."""
    if not stretch or stretch.get("t_start") is None:
        return np.zeros(int(n), bool)
    return mask_from_hours([[stretch["t_start"], stretch["t_end"]]], t0, n, fs)


def surviving_runs(t0, n, stretch: dict, keep, fs=float(SCORE_RATE), min_run_s=TR.MIN_SEGMENT_S) -> list:
    """[(offset, length)] of the runs of at least `min_run_s` the mask leaves inside one stretch.

    This is the measurement a pass makes before it estimates: a kept hour the transient mask or a gap cuts
    into falls under the transients.MIN_SEGMENT_S = 3,600 s run floor and is dropped, so a stretch the mask
    leaves with no whole run is a stretch no pass can read, whatever its length.
    """
    inside = stretch_mask(t0, n, stretch, fs) & np.asarray(keep, bool)
    return TR.segments(inside, int(float(min_run_s) * float(fs)))


def local_keep(sv, site, rate=SCORE_RATE):
    """(t0, n, the keep mask) of one site: the record's own gaps and its transient and E-burst intervals.

    The mask is measured at the score's own rate. A gap and an event interval are the same span of time at
    every rate, so the runs it leaves are the runs a pass at any rate starts from; a pass whose own
    reference mask removes more is refused inside the pass, loudly, and that refusal stays.
    """
    from . import frame as FR
    from . import mth5_build
    work = Path(sv.cfg["work_root"])
    d = sv.decision(site)
    t0, arrays, _applied, _rec = cache.load_decided(site, sv, rate=int(rate))
    arrays, _ang = FR.rotate_to_mean_field(arrays, regimes=FR.parse_regimes(d.get("rot_regimes")),
                                           drop=FR.parse_regimes(d.get("rot_drop")), fs=float(rate))
    local = {c: np.asarray(arrays[c], float) for c in mth5_build.LOCAL_CHANNELS}
    del arrays
    keep, _stats = TR.build_keep(t0, local, float(rate),
                                 TR.site_events(site, list(sv.sites.site), work, sv.cfg), (),
                                 TR.e_events(site, work, sv.cfg))
    n = len(local["Hx"])
    del local
    return int(t0), int(n), keep


def _refusal(stretch: dict, runs: list, min_run_s=TR.MIN_SEGMENT_S) -> str:
    """The sentence a stretch the mask empties is refused with, or '' where a run survives."""
    if runs:
        return ""
    return ("the mask leaves no run of %d s inside the %d h stretch"
            % (int(min_run_s), int(stretch.get("hours") or 0)))


def _item(stretch: dict, tag: str, random: bool, seed: int) -> dict:
    """One selection as the store and the pass read it: the bounds, the hour count and the intervals."""
    return dict(tag=str(tag), rule=str(stretch.get("rule", tag)), random=bool(random), seed=int(seed),
                t_start=stretch.get("t_start"), t_end=stretch.get("t_end"),
                n_hours=int(stretch.get("hours") or 0), threshold=stretch.get("coh_min"),
                lines=list(stretch.get("lines", LINES)), cut=bool(stretch.get("cut", False)),
                mean_score=stretch.get("mean_score"), n_hours_scored=stretch.get("n_hours_scored"),
                n_hours_above=stretch.get("n_hours_above"), reason=stretch.get("reason", ""),
                refused="", n_runs_kept=None,
                hours=([[int(stretch["t_start"]), int(stretch["t_end"])]]
                       if stretch.get("t_start") is not None else []))


def _measure(item: dict, stretch: dict, t0, n, keep, fs) -> dict:
    """Apply the mask to one chosen stretch and refuse it where no run of the floor survives.

    A refused item keeps its bounds, so what was chosen is still reported, and carries no hour interval, so
    nothing downstream can ask a pass for it.
    """
    if item.get("t_start") is None or keep is None:
        return item
    runs = surviving_runs(t0, n, stretch, keep, fs)
    item["n_runs_kept"] = len(runs)
    item["kept_seconds"] = int(sum(L for _o, L in runs) / float(fs))
    why = _refusal(stretch, runs)
    if why:
        item.update(refused=why, hours=[], n_hours=0,
                    reason="%s; the stretch was %s to %s UTC"
                           % (why, _hm(stretch["t_start"]), _hm(stretch["t_end"])))
    return item


def select(sv, site, coh_min=WINDOW_COH, lines=LINES, max_h=STRETCH_MAX_H, seed=SEED, rate=SCORE_RATE,
           force=False, keep=None) -> dict:
    """{tag: the selection} for one site: the longest coherent stretch and the control that sizes it.

    The stretch is chosen on the 1 Hz score table and is stated in unix seconds, so the same bounds carry to
    a pass at any rate. Each chosen stretch is then measured against the site's own mask, and one the mask
    leaves with no run of transients.MIN_SEGMENT_S is refused here, before any pass, with the reason. The
    control is measured the same way and refuses the same way; the two are independent, so a site can carry
    a stretch its control cannot match.

    `keep` is the site's mask where the caller already holds it; otherwise it is read here.
    """
    table = site_scores(sv, site, rate=rate, force=force)
    t0, n = MK.span(sv, site, int(rate))
    sel = longest_stretch(table, coh_min=coh_min, lines=lines, max_h=max_h)
    if keep is None and sel.get("t_start") is not None:
        _t0, _n, keep = local_keep(sv, site, rate=rate)
    out = {STRETCH: _measure(_item(sel, STRETCH, False, seed), sel, t0, n, keep, float(rate))}
    if sel.get("t_start") is not None:
        ctrl = control_stretch(t0, n, sel, seed, float(rate))
        out[CONTROL] = _measure(_item(ctrl, CONTROL, True, seed), ctrl, t0, n, keep, float(rate))
    return out


def write_selection(work_root, site, sel, band_s=SCORE_BAND_S, hour_s=HOUR_S, rate=SCORE_RATE) -> Path:
    """Write <work_root>/<site>/hour_selection.json and return its path."""
    p = selection_path(work_root, site)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(site=str(site), rate_hz=float(rate), band_s=list(band_s),
                                 hour_s=float(hour_s), nperseg=nperseg_for(rate),
                                 coh_min=float(WINDOW_COH), stretch_max_h=int(STRETCH_MAX_H),
                                 selections=sel), indent=1, default=str), encoding="utf-8")
    return p


def read_selection(work_root, site) -> dict:
    p = selection_path(work_root, site)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(sv, site, coh_min=WINDOW_COH, lines=LINES, max_h=STRETCH_MAX_H, seed=SEED, rate=SCORE_RATE,
          force=False) -> dict:
    """One site's hour scores and its selection, both written. Returns the selection dict."""
    sel = select(sv, site, coh_min=coh_min, lines=lines, max_h=max_h, seed=seed, rate=rate, force=force)
    write_selection(Path(sv.cfg["work_root"]), site, sel, rate=rate)
    return sel


def summary_rows(work_root, sites, keys=TAGS) -> pd.DataFrame:
    """One row per site and tag: the bounds, the hours kept and the hours the score could judge."""
    rows = []
    for s in sites:
        d = read_selection(work_root, s)
        for tag, v in sorted((d.get("selections") or {}).items()):
            if keys is not None and tag not in keys:
                continue
            rows.append(dict(site=s, selection=tag, random=v.get("random"), seed=v.get("seed"),
                             threshold=v.get("threshold"),
                             start_utc=(_iso(v["t_start"]) if v.get("t_start") else ""),
                             end_utc=(_iso(v["t_end"]) if v.get("t_end") else ""),
                             hours_scored=v.get("n_hours_scored"), hours_above=v.get("n_hours_above"),
                             hours_kept=v.get("n_hours"), cut=v.get("cut"),
                             days_kept=round((v.get("n_hours") or 0) / 24.0, 2)))
    return pd.DataFrame(rows)
