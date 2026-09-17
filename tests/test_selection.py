"""The one selection rule of auslamp_proc.process.selection, on synthetic records and synthetic tables.

Every test states what it would take to fail. None of them reads a survey tree: the hour score is measured on
a record built here, and the stretch rule is measured on tables built here.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from auslamp_proc.process import selection as SEL

HOUR = 3600.0
T0 = 1772920800                                     # 2026-03-07 22:00:00 UTC, a whole UTC hour


def _table(coh_xy, coh_yx, t0=T0, score=None):
    """An hour score table of two given series, as score_hours returns one."""
    n = len(coh_xy)
    t = np.arange(n) * HOUR + float(t0)
    xy = np.asarray(coh_xy, float)
    yx = np.asarray(coh_yx, float)
    out = pd.DataFrame(dict(hour=np.arange(n), t_start=t.astype(int), t_end=(t + HOUR).astype(int),
                            utc=[str(x) for x in t.astype(int)], coh_xy=xy, coh_yx=yx,
                            score=(np.nanmean(np.vstack([xy, yx]), axis=0) if score is None
                                   else np.asarray(score, float))))
    out.attrs["band_s"] = list(SEL.SCORE_BAND_S)
    return out


# ---------------------------------------------------------------- the hour score

def test_the_hour_grid_starts_on_a_whole_utc_hour_and_holds_only_whole_hours():
    """Fails if the first scored hour does not begin on a UTC hour boundary, or if a partial hour at either
    end of the record is scored."""
    t0 = T0 + 900                                   # the logger started a quarter of an hour past the hour
    n = int(5.5 * HOUR)
    starts, t_starts = SEL.hour_grid(t0, n, 1.0)
    assert len(starts) == 4                         # 5.5 h of record, offset, holds four whole UTC hours
    assert all(float(t) % HOUR == 0 for t in t_starts)
    assert t_starts[0] == t0 + (HOUR - 900)
    assert starts[-1] + int(HOUR) <= n


def test_a_coherent_pair_scores_high_and_an_independent_pair_scores_low():
    """Fails if the hour score does not separate a line that follows the field from one that does not. The
    two pairs are built from the same magnetic record, so only the electric line differs."""
    rng = np.random.default_rng(7)
    n = 4 * int(HOUR)
    hy = np.cumsum(rng.normal(size=n))              # a red magnetic series, which is what the field is
    hx = np.cumsum(rng.normal(size=n))
    arrays = dict(Hx=hx, Hy=hy,
                  Ex=3.0 * hy + 0.02 * rng.normal(size=n),   # Ex follows Hy
                  Ey=rng.normal(size=n))                     # Ey is independent of Hx
    t = SEL.score_hours(T0, arrays, fs=1.0)
    assert len(t) == 4
    assert float(np.nanmedian(t.coh_xy)) > 0.9
    assert float(np.nanmedian(t.coh_yx)) < 0.5
    assert np.all(np.isfinite(t.score))


def test_an_hour_holding_a_non_finite_sample_scores_nothing_on_that_pair():
    """Fails if an hour with a hole in one pair is given a coherence rather than left unscored."""
    rng = np.random.default_rng(11)
    n = 3 * int(HOUR)
    hy = np.cumsum(rng.normal(size=n))
    arrays = dict(Hx=np.cumsum(rng.normal(size=n)), Hy=hy, Ex=3.0 * hy, Ey=rng.normal(size=n))
    arrays["Ex"] = arrays["Ex"].copy()
    arrays["Ex"][int(1.5 * HOUR)] = np.nan          # one sample of the second hour
    t = SEL.score_hours(T0, arrays, fs=1.0)
    assert np.isfinite(t.coh_xy.iloc[0]) and np.isfinite(t.coh_xy.iloc[2])
    assert not np.isfinite(t.coh_xy.iloc[1])
    assert np.isfinite(t.coh_yx.iloc[1])            # the other pair is untouched


def test_the_welch_segment_is_1024_s_at_every_rate():
    """Fails if the segment the hourly coherence is read over is not the same duration at 1 Hz and at 10 Hz,
    which would make the band it measures depend on the rate."""
    assert SEL.nperseg_for(1) == 1024
    assert SEL.nperseg_for(10) == 10240
    assert SEL.SCORE_BAND_S == (20.0, 200.0)


# ---------------------------------------------------------------- the stretch

def test_one_30_hour_coherent_stretch_is_found_with_its_bounds():
    """Fails if the rule returns a stretch that is not the longest run with BOTH lines above the threshold,
    or if its bounds are not the first and last hour of that run."""
    xy = np.full(80, 0.2)
    yx = np.full(80, 0.2)
    xy[10:40], yx[10:40] = 0.8, 0.8                 # 30 h with both lines above: the answer
    xy[50:75], yx[50:75] = 0.8, 0.2                 # 25 h with one line above: not a stretch
    xy[76:79], yx[76:79] = 0.9, 0.9                 # 3 h with both above: shorter
    got = SEL.longest_stretch(_table(xy, yx), coh_min=0.5)
    assert got["hours"] == 30
    assert got["t_start"] == T0 + 10 * HOUR
    assert got["t_end"] == T0 + 40 * HOUR
    assert got["n_runs"] == 2
    assert got["n_hours_above"] == 33
    assert got["cut"] is False


def test_a_90_hour_stretch_is_cut_to_its_best_48_contiguous_hours():
    """Fails if a stretch longer than STRETCH_MAX_H is delivered whole, if the cut is not 48 contiguous
    hours, or if the 48 taken are not the window of that length with the highest mean score."""
    xy = np.full(120, 0.2)
    yx = np.full(120, 0.2)
    xy[10:100], yx[10:100] = 0.6, 0.6               # 90 h above the threshold
    xy[60:90], yx[60:90] = 0.95, 0.95               # the best part of it sits at hours 60-90
    got = SEL.longest_stretch(_table(xy, yx), coh_min=0.5, max_h=48)
    assert got["cut"] is True
    assert got["hours"] == 48
    assert (got["t_end"] - got["t_start"]) == 48 * HOUR
    # the best 48 h window must contain the 30 h of 0.95 and sit inside the 90 h run
    assert got["t_start"] >= T0 + 10 * HOUR and got["t_end"] <= T0 + 100 * HOUR
    assert got["t_start"] <= T0 + 60 * HOUR and got["t_end"] >= T0 + 90 * HOUR
    assert "the longest run is 90 h" in got["reason"]


def test_a_row_wise_window_reads_that_row_s_line_alone():
    """Fails if a window asked for one impedance row is gated on both electric lines. The x row is read on
    Ex against Hy and the y row on Ey against Hx, so a record where one line is dead still has a window for
    the other."""
    xy = np.full(40, 0.9)
    yx = np.full(40, 0.1)                           # the y line never rises
    both = SEL.longest_stretch(_table(xy, yx), coh_min=0.5, lines=SEL.LINES)
    assert both["t_start"] is None and both["hours"] == 0
    x_row = SEL.longest_stretch(_table(xy, yx), coh_min=0.5, lines=(SEL.ROW_LINE["x"],))
    assert x_row["hours"] == 40
    y_row = SEL.longest_stretch(_table(xy, yx), coh_min=0.5, lines=(SEL.ROW_LINE["y"],))
    assert y_row["t_start"] is None


def test_an_unscored_hour_cuts_a_stretch_in_two():
    """Fails if an hour that could not be scored is treated as inside a coherent stretch."""
    xy = np.full(20, 0.9)
    yx = np.full(20, 0.9)
    xy[9] = np.nan                                  # one hour holding a non-finite sample
    got = SEL.longest_stretch(_table(xy, yx), coh_min=0.5)
    assert got["hours"] == 10                       # hours 10..19, the longer of the two pieces
    assert got["n_runs"] == 2


def test_no_hour_above_the_threshold_is_a_reason_and_not_a_stretch():
    """Fails if a record with no coherent hour returns a stretch rather than the reason it found none."""
    got = SEL.longest_stretch(_table(np.full(10, 0.1), np.full(10, 0.1)), coh_min=0.5)
    assert got["t_start"] is None and got["hours"] == 0
    assert "no hour reads above" in got["reason"]


def test_a_named_stretch_reads_its_start_and_its_hours():
    """Fails if the named form does not parse, or if a malformed one raises instead of stating the reason."""
    got = SEL.named_stretch("window:2026-03-22 12:00/14")
    assert got["t_start"] == int(pd.Timestamp("2026-03-22 12:00", tz="UTC").timestamp())
    assert got["t_end"] - got["t_start"] == 14 * HOUR
    bad = SEL.named_stretch("window:not-a-time/14")
    assert bad["t_start"] is None and "does not read as" in bad["reason"]


# ---------------------------------------------------------------- the control

def test_the_control_is_the_same_length_lands_elsewhere_and_follows_its_seed():
    """Fails if the control differs in length from the stretch it controls, if it overlaps that stretch
    where the record is long enough to hold both, or if two seeds give the same stretch."""
    n = 60 * 86400
    s = dict(t_start=T0 + 15 * 86400, t_end=T0 + 15 * 86400 + 30 * int(HOUR))
    c = SEL.control_stretch(T0, n, s, seed=SEL.SEED)
    assert (c["t_end"] - c["t_start"]) == (s["t_end"] - s["t_start"])
    assert c["hours"] == 30
    assert c["t_end"] <= s["t_start"] or c["t_start"] >= s["t_end"]
    assert SEL.control_stretch(T0, n, s, seed=1)["t_start"] != c["t_start"]
    assert SEL.control_stretch(T0, n, s, seed=SEL.SEED)["t_start"] == c["t_start"]


def test_the_control_begins_on_a_whole_utc_hour_like_the_stretch_it_controls():
    """Fails if the control is placed at an arbitrary sample rather than on the whole-UTC-hour grid.

    The stretch is a run of whole hours and the run floor is one hour, so a control starting mid-hour is not
    the same kind of object: it spans one more partial hour than the stretch and its mask edges fall inside
    an hour the score was never computed over. Two hundred seeds are drawn, because one seed can land on an
    hour boundary by luck.
    """
    n = 60 * 86400
    t0 = T0 + 1237                                  # the logger did not start on the hour
    s = dict(t_start=T0 + 15 * 86400, t_end=T0 + 15 * 86400 + 7 * int(HOUR))
    for seed in range(200):
        c = SEL.control_stretch(t0, n, s, seed=seed)
        assert c["t_start"] % 3600 == 0, "seed %d starts %d s past the hour" % (seed, c["t_start"] % 3600)
        assert (c["t_end"] - c["t_start"]) % 3600 == 0
        assert c["hours"] == 7


def test_a_record_with_no_room_for_a_control_says_so_and_returns_none():
    """Fails if a stretch that fills its own record is handed a control that overlaps it, or one at all.

    A stretch that cannot be controlled cannot be judged against one, and the rule states that rather than
    drawing a control on top of the selection.
    """
    n = 9 * int(HOUR)
    s = dict(t_start=T0, t_end=T0 + 8 * int(HOUR))
    c = SEL.control_stretch(T0, n, s, seed=SEL.SEED)
    assert c["t_start"] is None and c["hours"] == 0
    assert "no second run" in c["reason"]


def test_the_stretch_mask_covers_exactly_the_stretch_at_both_rates():
    """Fails if the mask of a stretch keeps a different number of samples from the stretch's own length."""
    n_days = 10
    s = dict(t_start=T0 + 86400, t_end=T0 + 86400 + 30 * int(HOUR))
    for fs in (1.0, 10.0):
        m = SEL.stretch_mask(T0, int(n_days * 86400 * fs), s, fs=fs)
        assert int(m.sum()) == int(30 * HOUR * fs)
    empty = SEL.stretch_mask(T0, 100, dict(t_start=None, t_end=None), fs=1.0)
    assert not empty.any()


# ---------------------------------------------------------------- the tags

def test_the_tags_are_stretch_control_and_whole():
    """Fails if the package still offers a fraction tag. The file name, the ledger column and the EDI's own
    selection= line all read one of these three."""
    assert SEL.STRETCH == "stretch"
    assert SEL.CONTROL == "control"
    assert SEL.WHOLE == "whole"
    assert SEL.TAGS == ("stretch", "control")
    for gone in ("f05", "f10", "f25", "r25"):
        assert gone not in SEL.TAGS and gone != SEL.WHOLE


def _keep_with_hole(t0, n, hole=None):
    """A keep mask over `n` samples with one span masked out, as a transient interval would leave it."""
    keep = np.ones(int(n), bool)
    if hole is not None:
        keep[int(hole[0]):int(hole[1])] = False
    return keep


def test_a_stretch_whose_mask_leaves_one_whole_run_is_passed():
    """Fails if a stretch is refused for its length. A 2 h stretch the mask leaves whole holds two runs of
    the 3,600 s floor and is a stretch a pass can read, so the rule passes it."""
    n = 10 * int(HOUR)
    s = dict(t_start=T0 + 2 * int(HOUR), t_end=T0 + 4 * int(HOUR), hours=2)
    runs = SEL.surviving_runs(T0, n, s, _keep_with_hole(T0, n), fs=1.0)
    assert runs, "a clean 2 h stretch left no run"
    assert sum(L for _o, L in runs) == 2 * int(HOUR)
    assert SEL._refusal(s, runs) == ""


def test_a_stretch_whose_mask_leaves_no_whole_run_is_refused_with_that_sentence():
    """Fails if a stretch the mask cuts below the run floor is passed to an estimator, or if the refusal
    does not say what it is.

    The control is the test above: the same 2 h stretch with no hole survives, so the refusal is the mask's
    doing and not the length's.
    """
    n = 10 * int(HOUR)
    s = dict(t_start=T0 + 2 * int(HOUR), t_end=T0 + 4 * int(HOUR), hours=2)
    # one masked minute in the middle leaves two pieces, each under the 3,600 s floor
    keep = _keep_with_hole(T0, n, hole=(3 * int(HOUR) - 30, 3 * int(HOUR) + 30))
    runs = SEL.surviving_runs(T0, n, s, keep, fs=1.0)
    assert not runs, "the mask left a run of the floor where it should have left none"
    why = SEL._refusal(s, runs)
    assert why == "the mask leaves no run of 3600 s inside the 2 h stretch", why


def test_a_long_stretch_the_mask_cuts_into_keeps_the_runs_that_survive():
    """Fails if a stretch is refused because the mask cut it at all, rather than because nothing survived.
    A 6 h stretch cut once in the middle still holds two runs of about 3 h."""
    n = 24 * int(HOUR)
    s = dict(t_start=T0 + 2 * int(HOUR), t_end=T0 + 8 * int(HOUR), hours=6)
    keep = _keep_with_hole(T0, n, hole=(5 * int(HOUR) - 30, 5 * int(HOUR) + 30))
    runs = SEL.surviving_runs(T0, n, s, keep, fs=1.0)
    assert len(runs) == 2
    assert SEL._refusal(s, runs) == ""
