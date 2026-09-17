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

from auslamp_proc.process import aurora_run, coherence as COH, edi as EDI, frame as FR, mth5_build
from auslamp_proc.process import rate as RATE, references as REF, run as RUN
from auslamp_proc.process import selection as SEL, transients as TR


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


# ------------------------------------------------- the 10 Hz hour selection

def _pair_record(t0, hours, fs=10.0, seed=7):
    """A record of whole hours whose Ex-Hy pair is coherent in the last third and noise before it."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * fs)
    hx = rng.standard_normal(n)
    hy = rng.standard_normal(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    live = slice(int(2 * n / 3), n)
    ex[live] = 5.0 * hy[live] + 0.05 * rng.standard_normal(n - live.start)
    ey[live] = 5.0 * hx[live] + 0.05 * rng.standard_normal(n - live.start)
    return t0, {"Hx": hx, "Hy": hy, "Ex": ex, "Ey": ey}


def test_selection_tags_are_the_four_the_package_names():
    """Fails if a fraction does not key to its documented tag."""
    assert [SEL.key_for(f) for f in SEL.SELECT10] == ["f05", "f10", "f25"]
    assert SEL.key_for(SEL.RANDOM_FRACTION, True) == "r25"
    assert SEL.nperseg_for(10) == SEL.SCORE_NPERSEG_10HZ and SEL.nperseg_for(1) == 205


def test_hour_grid_starts_on_a_whole_utc_hour():
    """Fails if the first scored hour does not begin on a UTC hour boundary, or an hour runs past the end."""
    t0 = 1759017600 + 900                      # a record that starts a quarter past the hour
    n = int(6 * 3600 * 10.0)
    starts, t_starts = SEL.hour_grid(t0, n, 10.0)
    assert np.all(np.asarray(t_starts) % SEL.HOUR_S == 0)
    assert starts[0] == int(2700 * 10)
    assert int(starts[-1]) + int(SEL.HOUR_S * 10) <= n


def test_score_hours_reads_the_coherent_hours_and_drops_a_dead_line():
    """Fails if a coherent hour does not outscore a noise hour, or a `dead` line is still in the average."""
    t0 = 1759017600
    t0, arrays = _pair_record(t0, 6)
    table = SEL.score_hours(t0, arrays, 10.0)
    assert len(table) == 6 and table.pairs.max() == 2
    assert float(table.score.iloc[-1]) > 0.8 > float(table.score.iloc[0])

    days = pd.DataFrame([dict(t_start=t0, t_end=t0 + 86400, Ex_state="dead", Ey_state="sound")])
    dead = SEL.score_hours(t0, arrays, 10.0, elines=days)
    assert set(dead.pairs) == {1}
    assert np.allclose(dead.score.to_numpy(float), table.score_yx.to_numpy(float), equal_nan=True)


def test_selection_keeps_whole_hours_and_the_control_costs_the_same():
    """Fails if a kept hour is not one whole 3,600 s run, or the control keeps a different number of hours."""
    t0 = 1759017600
    t0, arrays = _pair_record(t0, 24)
    n = len(arrays["Hx"])
    table = SEL.score_hours(t0, arrays, 10.0)
    sel = SEL.selections(table, t0, n, 10.0, fractions=(0.25,), random_fraction=0.25, seed=11)
    assert sorted(sel) == ["f25", "r25"]
    assert sel["f25"]["n_hours"] == sel["r25"]["n_hours"] == 6
    assert sel["r25"]["threshold"] is None and sel["f25"]["threshold"] is not None
    keep = SEL.mask_from_hours(sel["f25"]["hours"], t0, n, 10.0)
    runs = TR.segments(keep, int(TR.MIN_SEGMENT_S * 10))
    assert len(runs) == len({tuple(h) for h in sel["f25"]["hours"]}) or all(
        L % int(TR.MIN_SEGMENT_S * 10) == 0 for _o, L in runs)
    assert int(keep.sum()) == 6 * int(SEL.HOUR_S * 10)
    # the ranked selection takes the coherent third and the random control cannot
    assert float(np.mean([table.score[table.t_start == h[0]].iloc[0] for h in sel["f25"]["hours"]])) > \
        float(np.mean([table.score[table.t_start == h[0]].iloc[0] for h in sel["r25"]["hours"]]))


def test_widen_ledger_adds_the_column_and_keeps_the_rows(tmp_path):
    """Fails if widening a ledger written before `selection` existed loses or reorders a row."""
    p = tmp_path / "runs.csv"
    old = [c for c in RUN.RUNS_COLUMNS if c != "selection"]
    before = pd.DataFrame([{c: ("Q49" if c == "site" else 1) for c in old},
                           {c: ("Q50" if c == "site" else 2) for c in old}], columns=old)
    before.to_csv(p, index=False)
    assert RUN.widen_ledger(p) is True
    after = pd.read_csv(p)
    assert list(after.columns) == RUN.RUNS_COLUMNS
    assert list(after.site) == ["Q49", "Q50"] and after.selection.isna().all()
    assert RUN.widen_ledger(p) is False


# --------------------------------------------------- the rate the caveat states

class _TF:
    """The two fields process.rate reads off a transfer function."""

    def __init__(self, period, z):
        self.period = np.asarray(period, float)
        self.z = np.asarray(z, complex)


def _tf(periods, rho_xy, rho_yx=None):
    """A transfer function whose xy and yx apparent resistivities are the ones given."""
    p = np.asarray(periods, float)
    rho_yx = rho_xy if rho_yx is None else rho_yx
    z = np.zeros((len(p), 2, 2), complex)
    z[:, 0, 1] = np.sqrt(np.asarray(rho_xy, float) / (0.2 * p))
    z[:, 1, 0] = np.sqrt(np.asarray(rho_yx, float) / (0.2 * p))
    return _TF(p, z)


def test_rate_departure_reads_the_ratio_and_refuses_to_extrapolate():
    """Fails if a known 5 per cent offset is not read back, or a period outside the 1 Hz row is compared."""
    p = np.array([2.0, 4.0, 8.0, 16.0, 30.0])
    base = _tf(p, np.full(5, 100.0))
    high = _tf(p, np.full(5, 105.0))
    d = RATE.departure(high, base)
    assert d["xy"] == pytest.approx(1.05, rel=1e-6)
    assert d["yx"] == pytest.approx(1.05, rel=1e-6)
    # the 1 Hz row starts at 8 s, so the 4-8 s half of the band has nothing to compare against and the
    # periods below it are dropped rather than measured against the end point numpy.interp clamps to
    short = _tf(np.array([8.0, 16.0, 30.0]), np.full(3, 100.0))
    wild = _tf(p, np.array([1e6, 1e6, 105.0, 105.0, 105.0]))
    assert RATE.departure(wild, short)["xy"] == pytest.approx(1.05, rel=1e-6)


def test_rate_caveat_carries_the_survey_number_or_says_there_is_none():
    """Fails if the caveat quotes a figure without a measurement, or omits the band or the site count."""
    empty = RATE.caveat(None)
    assert "has not been measured" in empty and "per cent" not in empty
    rec = RATE.measure([("A", _tf([4.0, 8.0, 16.0], [100.0] * 3), _tf([4.0, 8.0, 16.0], [100.0] * 3)),
                        ("B", _tf([4.0, 8.0, 16.0], [90.0] * 3), _tf([4.0, 8.0, 16.0], [100.0] * 3))],
                       "remote")
    assert rec["n_sites"] == 2 and rec["n_scored"]["xy"] == 2
    assert rec["median_ratio"]["xy"] == pytest.approx(0.95, rel=1e-3)
    said = RATE.caveat(rec)
    assert "4-32 s" in said and "2 whole-record 10 Hz remote product(s)" in said
    assert "-5.0 per cent" in said and "Victoria" not in said


def test_rate_record_round_trips(tmp_path):
    """Fails if the measurement written for a survey does not read back as the same sentence."""
    rec = RATE.measure([("A", _tf([4.0, 8.0], [96.0] * 2), _tf([4.0, 8.0], [100.0] * 2))], "remote")
    RATE.write_record(tmp_path, rec)
    assert RATE.caveat(RATE.read_record(tmp_path)) == RATE.caveat(rec)
    assert RATE.caveat(RATE.read_record(tmp_path / "nowhere")) == RATE.caveat(None)


def test_run_refuses_the_single_station_with_the_ruling_sentence():
    """Fails if --kinds single is accepted, or the refusal does not say why the kind is gone."""
    assert "single" in RUN.REFUSED_KINDS
    said = RUN.REFUSED_KINDS["single"]
    assert "biases it low" in said and "error bars" in said
    assert RUN.main(["--survey", "queensland_phase1", "--site", "Q49", "--kinds", "single"]) == 2
    with pytest.raises(ValueError, match="not a kind of this package"):
        RUN.one_site("queensland_phase1", "Q49", "first", ["single"], [1], "kaiser20_75")


def test_rewrite_caveat_changes_one_line_and_nothing_else(tmp_path):
    """Fails if the rewrite moves a byte outside the caveat line, or does not report what it replaced."""
    p = tmp_path / "x.edi"
    body = ("\n".join(["  >HEAD", "    DATAID=Q49", "    caveat_10hz=the old sentence",
                       "    other=kept", ">END"]) + "\n").encode("utf-8")
    p.write_bytes(body)
    r = RATE.rewrite_caveat(p, "the new sentence")
    assert r["changed"] and r["old"] == "the old sentence" and r["new"] == "the new sentence"
    after = p.read_bytes()
    strip = lambda b: [ln for ln in b.splitlines(keepends=True) if b"caveat_10hz=" not in ln]  # noqa: E731
    assert strip(after) == strip(body)
    assert RATE.carries_caveat(p, "the new sentence")
    assert RATE.rewrite_caveat(p, "the new sentence")["reason"].startswith("the file already carries")


def test_rewrite_caveat_refuses_what_it_cannot_place(tmp_path):
    """Fails if a file with no caveat line, two of them, or a sentence carrying a newline is rewritten."""
    none = tmp_path / "none.edi"
    none.write_bytes(b"    DATAID=Q49\n>END\n")
    before = none.read_bytes()
    assert RATE.rewrite_caveat(none, "s")["reason"] == "the file carries no caveat_10hz line"
    assert none.read_bytes() == before

    two = tmp_path / "two.edi"
    two.write_bytes(b"    caveat_10hz=a\n    caveat_10hz=b\n")
    before = two.read_bytes()
    assert "2 caveat_10hz lines" in RATE.rewrite_caveat(two, "s")["reason"]
    assert two.read_bytes() == before

    one = tmp_path / "one.edi"
    one.write_bytes(b"    caveat_10hz=a\n")
    before = one.read_bytes()
    assert RATE.rewrite_caveat(one, "two\nlines")["reason"].startswith("the sentence carries a line ending")
    assert one.read_bytes() == before


def test_rewrite_is_recorded_in_the_run_folder(tmp_path):
    """Fails if a rewritten product leaves no record of the sentence it carried before."""
    (tmp_path / "provenance.json").write_text('{"run": "first"}', encoding="utf-8")
    rows = [dict(path=str(tmp_path / "a.edi"), changed=True, old="was", new="is", reason=""),
            dict(path=str(tmp_path / "b.edi"), changed=False, old=None, new=None, reason="no line")]
    RATE.record_rewrite(tmp_path, rows, at="2026-09-17T00:00:00+00:00")
    import json as _json
    d = _json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert len(d[RATE.REWRITE_NAME]) == 1
    assert d[RATE.REWRITE_NAME][0] == dict(at="2026-09-17T00:00:00+00:00", product="a.edi",
                                           old="was", new="is")


def test_read_parameter_finds_the_line_under_its_dotted_name(tmp_path):
    """Fails if a processing_parameters line is missed because the writer emits its full dotted name.

    The EDI writer emits `transfer_function.processing_parameters.selection=...`, so a reader that asks for
    a line beginning with `selection=` finds nothing and reports a product that carries the line as one that
    does not. Section 9's check read it that way and failed 184 sound products before this was fixed.
    """
    p = tmp_path / "x.edi"
    p.write_text("\n".join([
        "  >HEAD",
        "    transfer_function.processing_parameters.selection=f05: the best 5 per cent",
        "    transfer_function.processing_parameters.caveat_10hz=a sentence",
        "    transfer_function.processing_parameters.runs=30 kept stretch(es)",
        "    DATAID=Q49",
        ">END"]) + "\n", encoding="utf-8")
    assert EDI.read_parameter(p, "selection").startswith("f05")
    assert EDI.read_parameter(p, "caveat_10hz") == "a sentence"
    assert EDI.read_parameter(p, "runs").startswith("30 kept")
    assert EDI.read_parameter(p, "DATAID") == "Q49"
    assert EDI.read_parameter(p, "not_there") == ""
    # the key is matched where it sits, not as a substring of a longer name
    assert EDI.read_parameter(p, "election") == ""
