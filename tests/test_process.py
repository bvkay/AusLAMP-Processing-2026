"""The processing package on synthetic records: the frame, the mask, the estimators, the rule, the stack,
the MTH5 and the band objects.

Every test states what would make it fail. Nothing here reads a survey's raw tree or its cache: the records
are built in the test so that a failure names the code and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auslamp_proc.process import aurora_run, coherence as COH, frame as FR, mth5_build
from auslamp_proc.process import references as REF, transients as TR


# ------------------------------------------------------------------ the frame

def test_apply_signs_flips_and_records():
    """Fails if a -1 cell does not flip its channel, or a `decide` cell is not recorded as undecided."""
    n = 100
    a = {c: np.arange(n, dtype=float) + 1.0 for c in FR.CHANNELS}
    row = pd.Series(dict(sign_hx="+1", sign_hy="-1", sign_hz="decide", sign_ex="-1", sign_ey=""))
    out, applied, undecided = FR.apply_signs(a, row)
    assert np.allclose(out["Hx"], a["Hx"])
    assert np.allclose(out["Hy"], -a["Hy"])
    assert np.allclose(out["Ex"], -a["Ex"])
    assert applied == {"Hx": 1.0, "Hy": -1.0, "Hz": 1.0, "Ex": -1.0, "Ey": 1.0}
    assert sorted(undecided) == ["Ey", "Hz"]


def test_rotate_to_mean_field_zeroes_mean_hy_and_keeps_h():
    """Fails if the rotated mean Hy exceeds 1e-9 nT or |H| changes at any sample."""
    rng = np.random.default_rng(3)
    n = 20000
    hx = 25000.0 + 30 * rng.standard_normal(n)
    hy = 4000.0 + 30 * rng.standard_normal(n)
    a = {"Hx": hx, "Hy": hy, "Hz": rng.standard_normal(n)}
    before = np.hypot(hx, hy)
    out, ang = FR.rotate_to_mean_field(a)
    assert abs(float(np.nanmean(out["Hy"]))) < 1e-9
    assert np.max(np.abs(np.hypot(out["Hx"], out["Hy"]) - before)) < 1e-6
    assert abs(ang - np.degrees(np.arctan2(hy.mean(), hx.mean()))) < 1e-3


def test_rotate_regimes_and_drop():
    """Fails if a regime is not turned into its own frame, or a dropped day is not NaN."""
    n = 3 * 86400
    t = np.arange(n, dtype=float)
    hx = np.where(t < 86400, 20000.0, 15000.0) + np.sin(t / 500.0)
    hy = np.where(t < 86400, 3000.0, -6000.0) + np.cos(t / 500.0)
    out, segs = FR.rotate_to_mean_field({"Hx": hx, "Hy": hy}, regimes=((0, 1), (1, None)),
                                        drop=((2, 3),), fs=1.0)
    assert len(segs) == 2
    assert abs(float(np.nanmean(out["Hy"][:86400]))) < 1e-6
    assert abs(float(np.nanmean(out["Hy"][86400:2 * 86400]))) < 1e-6
    assert np.all(~np.isfinite(out["Hx"][2 * 86400:]))


def test_turn_tensor_is_its_own_inverse():
    """Fails if turning a tensor by an angle and back does not reproduce it to 1e-12."""
    rng = np.random.default_rng(11)
    z = rng.standard_normal((7, 2, 2)) + 1j * rng.standard_normal((7, 2, 2))
    t = rng.standard_normal((7, 1, 2)) + 1j * rng.standard_normal((7, 1, 2))
    zr, tr = FR.turn_tensor(z, t, 37.5)
    zb, tb = FR.turn_tensor(zr, tr, -37.5)
    assert np.max(np.abs(zb - z)) < 1e-12
    assert np.max(np.abs(tb - t)) < 1e-12
    assert np.max(np.abs(zr - z)) > 1e-6                 # the turn is not the identity


# ------------------------------------------------------------- mask and segments

def test_segments_and_keep_mask_drop_the_short_piece():
    """Fails if a kept piece under 3600 s is written as a run, or is not counted as an hour-floor loss."""
    fs, n, t0 = 1.0, 20000, 1_700_000_000
    keep = TR.keep_mask(t0, n, fs, [(t0 + 5000, t0 + 6000), (t0 + 8000, t0 + 8600)])
    pieces = TR.segments(keep)
    assert [L for _o, L in pieces] == [5000, 2000, 11400]
    runs = TR.segments(keep, int(3600 * fs))
    assert [L for _o, L in runs] == [5000, 11400]
    lost = TR.floor_dropped_frac(keep, fs)
    assert abs(lost - 2000 / n) < 1e-9


def test_gap_edge_screen_widens_gaps_and_makes_none():
    """Fails if the screen puts a gap into a continuous record, or does not widen an existing one.

    A screen that NaNs scattered samples is paid for by the 3,600 s run floor, which drops every piece it
    leaves shorter than an hour; this one can only widen what is already missing.
    """
    x = np.arange(1000.0)
    x[500] = np.nan
    y = REF.gap_edge_screen(x, 10.0, edge_s=1.0)
    bad = np.flatnonzero(~np.isfinite(y))
    assert bad.min() == 490 and bad.max() == 510
    assert len(TR.segments(np.isfinite(y))) == len(TR.segments(np.isfinite(x)))
    clean = np.arange(1000.0)
    assert np.isfinite(REF.gap_edge_screen(clean, 10.0)).all()
    assert int((~np.isfinite(REF.gap_edge_screen(x, 1.0))).sum()) == 1     # 1 Hz is left alone


# --------------------------------------------------------------- the estimators

def test_welch_bias_correction_lands_near_zero():
    """Fails if the coherence of two independent white series does not land well below the 1/N Welch floor.

    Welch coherence from N segments sits high by about (1 - c)^2 / N, so 40 segments of two independent
    series read about 0.025 rather than 0. The uncorrected value is computed here from scipy on the same
    series as the control, and the corrected one has to be under half of it and under 0.015.
    """
    from scipy import signal
    rng = np.random.default_rng(7)
    n = 40 * 1024
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    keep = np.ones(n, bool)
    got = COH.band_coh_clean(x, y, 1.0, keep, band=(1 / 200.0, 1 / 20.0), chunk_days=0.5, min_chunks=1,
                             nperseg=1024, min_segments=20)
    f, c = signal.coherence(x, y, fs=1.0, nperseg=1024, noverlap=0, window="hann")
    p = 1.0 / f[1:]
    band = (p >= 20.0) & (p <= 200.0)
    uncorrected = float(np.mean(c[1:][band]))
    assert np.isfinite(got)
    assert uncorrected > 0.015, "the Welch floor is not in this control (%.4f)" % uncorrected
    assert got < 0.5 * uncorrected and got < 0.015, \
        "corrected %.4f against an uncorrected %.4f" % (got, uncorrected)


def test_nperseg_scales_with_rate():
    """Fails if a segment does not keep its length in seconds when the rate changes."""
    assert COH.nperseg_for(1024, 1.0) == 1024
    assert COH.nperseg_for(1024, 10.0) == 10240


def test_fleet_weight_is_the_median_over_pairs():
    """Fails if a member's weight is anything but the median of its pairs with the rest of the pool."""
    pairs = {"A:B": dict(coh=0.9, chunks=10), "A:C": dict(coh=0.5, chunks=10),
             "B:C": dict(coh=0.7, chunks=10), "A:D": dict(coh=0.1, chunks=1)}
    w = COH.fleet_weight_table(pairs, ["A", "B", "C", "D"])
    assert w["A"] == pytest.approx(0.7)                  # median of 0.9 and 0.5; the 1-chunk pair is dropped
    assert w["B"] == pytest.approx(0.8)
    assert w["C"] == pytest.approx(0.6)
    assert w["D"] is None


# ------------------------------------------------------------ the five-branch rule

def _cand(name, km, frac, clean, coh, ev=0.001, days=None):
    return dict(name=name, km=km, overlap_days=(days if days is not None else 60 * frac),
                overlap_frac=frac, clean=clean, coh=coh, event_frac=ev, chunks=40)


def _rule(cands, **kw):
    return REF.branch_rule("T", sorted(cands, key=lambda c: c["km"]), len(cands), 0, 60.0, 20.0,
                           nearest=lambda: "NEAR", **kw)


def test_branch_rule_hits_each_branch():
    """Fails if any of the five branches is unreachable or picks another branch's candidate."""
    b1 = _rule([_cand("A", 80, 0.95, True, 0.8), _cand("B", 40, 0.95, True, 0.7)])
    assert (b1["branch"], b1["name"]) == (1, "B")         # clean, coherent, full overlap -> the nearest

    b2 = _rule([_cand("A", 40, 0.5, True, 0.8), _cand("B", 80, 0.7, True, 0.9)])
    assert (b2["branch"], b2["name"]) == (2, "B")         # no full overlap -> the longest

    b3 = _rule([_cand("A", 40, 0.5, True, 0.35), _cand("B", 80, 0.7, True, 0.45)])
    assert (b3["branch"], b3["name"]) == (3, "B")         # nobody at 0.5 -> the most coherent above 0.3

    b4 = _rule([_cand("A", 40, 0.5, False, 0.8, ev=0.30), _cand("B", 80, 0.5, False, 0.9, ev=0.05)])
    assert (b4["branch"], b4["name"]) == (4, "B")         # none clean -> the fewest events among coherent

    b5 = _rule([_cand("A", 40, 0.5, False, 0.1), _cand("B", 80, 0.5, False, 0.2)])
    assert (b5["branch"], b5["name"]) == (5, "A")         # nothing reaches the gate -> the nearest

    empty = _rule([])
    assert (empty["branch"], empty["name"]) == (5, "NEAR")
    for r in (b1, b2, b3, b4, b5, empty):
        assert r["reason"].startswith("branch %d" % r["branch"])


# ------------------------------------------------------------------ the stack

class _FakeSurvey:
    def __init__(self, tmp):
        self.cfg = dict(work_root=str(tmp), name="test")
        self.sites = pd.DataFrame(dict(site=["A", "B", "C"]))

    def decision(self, name):
        raise KeyError(name)


def test_stack_refuses_fewer_than_two_members(tmp_path):
    """Fails if a stack of one member is written: a one-member stack is a remote site renamed."""
    st = REF.Store(_FakeSurvey(tmp_path), ["A", "B", "C"], rate=1)
    with pytest.raises(REF.NoMembers):
        st.build_stack("A", members=({"B": 0.9}, {"B": 0.0}, {"C": "fleet coherence 0.2 < 0.5"}, {}))
    assert REF.STACK_MIN == 2


# ------------------------------------------------------------------- the MTH5

def test_mth5_read_back_equals_the_input(tmp_path):
    """Fails if the MTH5 differs from the arrays it was written from, or its run count differs from the
    kept stretches."""
    fs, t0 = 1.0, 1_700_000_000
    n = 12800
    rng = np.random.default_rng(5)
    local = {c: rng.standard_normal(n) * 10.0 for c in mth5_build.LOCAL_CHANNELS}
    keep = np.ones(n, bool)
    keep[7200:7800] = False                               # leaves 2 h, then 1 h, then an 800 s tail
    keep[11400:12000] = False
    row = pd.Series(dict(site="TST", lat=-25.0, lon=148.0, elev_m=400.0, instrument="PR6-24+Mag-03",
                         serial="12345", sample_rate_hz=1.0, dipole_n_m=10.0, dipole_e_m=11.0,
                         dipole_source="test", position_source="test", declination_deg=9.0))
    h5 = tmp_path / "tst.h5"
    _p, segs = mth5_build.write_h5(h5, "TST", local, None, t0, fs, "TEST", row, keep=keep)
    assert [L for _o, L in segs] == [7200, 3600]          # the 600 s tail is under the hour floor
    back = mth5_build.read_back(h5, "TST", local, segs, "TEST")
    assert back["problems"] == []
    assert back["runs_written"] == len(segs)
    assert back["worst_difference"] < 1e-3
    assert back["samples_compared"] == 5 * sum(L for _o, L in segs)


# ------------------------------------------------------------------- the bands

def test_band_objects_read_through_aurora():
    """Fails if either band file does not give the periods the package states for it.

    The indices in a band file are harmonics of the WINDOW, so a file paired with the wrong window length
    names different periods and nothing says so. This reads both through Aurora's own band machinery.
    """
    t1 = aurora_run.band_table(aurora_run.BANDS["1hz"])
    assert len(t1) == 48 and t1.level.nunique() == 6
    assert t1.centre_s.min() == pytest.approx(2.303, rel=1e-3)
    assert t1.centre_s.max() == pytest.approx(48470.0, rel=1e-3)

    t10 = aurora_run.band_table(aurora_run.BANDS["10hz"])
    assert len(t10) == 31 and t10.level.nunique() == 5
    assert t10.centre_s.min() == pytest.approx(1.073, rel=1e-3)
    assert t10.centre_s.max() == pytest.approx(1211.9, rel=1e-3)

    assert aurora_run.BANDS["1hz"].decimation_factors == [1, 4, 4, 4, 4, 4]
    assert aurora_run.BANDS["10hz"].window == 256
