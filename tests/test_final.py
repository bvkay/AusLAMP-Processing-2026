"""The earth rule, the product of record, the splice and the merge, on synthetic transfer functions.

Every test states what would make it fail. Nothing here reads a survey's work root: each transfer function is
built in the test and, where a file is needed, written to an EDI through mt_metadata, so a failure names the
code and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pytest

from auslamp_proc import final as FN, products as PR, readings as RD, splice as SP


# ------------------------------------------------------------------ synthetic tensors

def earth_tensor(n=41, rho=100.0, phase=45.0, bar=0.02, lo=0.5, hi=4.5, flip_yx=False):
    """A one-dimensional tensor of `rho` Ohm.m at `phase` deg over 10^lo to 10^hi s.

    rho = 0.2 T |Z|^2 with Z in mV/km/nT, so |Z| = sqrt(rho / (0.2 T)). The yx element is the negative of
    the xy element, which is what folds its phase into the first quadrant by +180 deg.
    """
    period = 10.0 ** np.linspace(lo, hi, n)
    mag = np.sqrt(rho / (0.2 * period))
    zxy = mag * np.exp(1j * np.radians(phase))
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 1] = zxy
    z[:, 1, 0] = (zxy if flip_yx else -zxy)
    err = float(bar) * np.abs(z)
    err[err == 0] = np.nan
    return period, z, err


def loop_tensor(n=41, lo=0.5, hi=4.5, bar=0.02):
    """An inductive loop: Z proportional to omega with a flat phase, so d log|Z| / d log T is -1.

    rho = 0.2 T |Z|^2 then falls as 1/T, which passes the quadrant, the bar and the held-range guards and
    is what the impedance-slope cut exists to refuse.
    """
    period = 10.0 ** np.linspace(lo, hi, n)
    mag = 100.0 / period
    zxy = mag * np.exp(1j * np.radians(45.0))
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 1] = zxy
    z[:, 1, 0] = -zxy
    return period, z, float(bar) * np.abs(z)


def tf_of(period, z, err, t=None, t_err=None, path="synthetic"):
    return PR.TFData(np.asarray(period, float), np.asarray(z, complex), np.asarray(err, float),
                     t, t_err, dict(path=path, parameters={}))


def write_edi(tmp_path, period, z, err, station="TEST", tipper=None):
    """A synthetic tensor written to an EDI through mt_metadata, as a product on disk would be."""
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

    tf = TF()
    tf.station_metadata.id = station
    tf.survey_metadata.id = "SYNTH"
    tf.station_metadata.location.latitude = -27.0
    tf.station_metadata.location.longitude = 145.0
    order = np.argsort(period)
    tf.period = np.asarray(period, float)[order]
    tf.impedance = xr.DataArray(np.asarray(z)[order], dims=["period", "output", "input"],
                                coords=dict(period=tf.period, output=["ex", "ey"], input=["hx", "hy"]))
    tf.impedance_error = xr.DataArray(np.asarray(err, float)[order], dims=["period", "output", "input"],
                                      coords=dict(period=tf.period, output=["ex", "ey"],
                                                  input=["hx", "hy"]))
    if tipper is not None:
        tf.tipper = xr.DataArray(np.asarray(tipper)[order], dims=["period", "output", "input"],
                                 coords=dict(period=tf.period, output=["hz"], input=["hx", "hy"]))
        tf.tipper_error = xr.DataArray(0.01 * np.ones(np.asarray(tipper).shape, float)[order],
                                       dims=["period", "output", "input"],
                                       coords=dict(period=tf.period, output=["hz"], input=["hx", "hy"]))
    out = tmp_path / ("%s.edi" % station)
    tf.write(fn=str(out), file_type="edi")
    return out


# ------------------------------------------------------------------ the quadrant

def test_quadrant_holds_on_an_earth_and_fails_on_a_flipped_row():
    """Fails if a 45 deg earth is called out of quadrant, or a row with its sign flipped is called in it."""
    p, z, e = earth_tensor()
    tf = tf_of(p, z, e)
    q = RD.quality(tf, "xy")
    assert q["quadrant"] and 0 < q["median_phase_deg"] < 90
    flipped = tf_of(p, -z, e)
    assert not RD.quality(flipped, "xy")["quadrant"]
    assert not RD.earth(RD.quality(flipped, "xy"))[0]


def test_a_flipped_yx_row_leaves_the_quadrant():
    """Fails if the yx row of a tensor whose Zyx has the wrong sign still folds into the first quadrant."""
    p, z, e = earth_tensor(flip_yx=True)
    q = RD.quality(tf_of(p, z, e), "yx")
    assert not q["quadrant"]
    assert not RD.earth(q)[0]


# ------------------------------------------------------------------ the bar guard and the held range

def test_the_bar_guard_refuses_a_row_whose_bar_exceeds_its_own_value():
    """Fails if a row whose median error over |Z| is above BAR_MAX is still called an earth."""
    p, z, e = earth_tensor(bar=1.5)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["bar_10_1000"] > RD.BAR_MAX
    ok, why = RD.earth(q)
    assert not ok and "bar" in why


def test_a_row_that_holds_at_no_period_is_not_an_earth():
    """Fails if a row with every bar above the held ceiling is called an earth on its shape alone."""
    p, z, e = earth_tensor(bar=0.6)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["held_n"] == 0
    ok, why = RD.earth(q)
    assert not ok and "holds at no period" in why


def test_the_rho_bar_is_twice_the_impedance_bar():
    """Fails if the two bars are not named apart: rho = 0.2 T |Z|^2, so its relative error is twice."""
    p, z, e = earth_tensor(bar=0.03)
    tf = tf_of(p, z, e)
    assert RD.rho_bar(tf, "xy") == pytest.approx(2.0 * RD.bar(tf, "xy"), rel=1e-9)


# ------------------------------------------------------------------ the impedance slope

def test_z_slope_refuses_an_inductive_loop_and_passes_an_earth():
    """Fails if a curve with d log|Z| / d log T near -1 passes, or a half-space-like earth near -0.4 fails."""
    p, z, e = loop_tensor()
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["z_slope"] == pytest.approx(-1.0, abs=0.05)
    ok, why = RD.earth(q)
    assert not ok and "inductive loop" in why

    # an earth whose rho rises as T^0.2 gives d log|Z| / d log T = (0.2 - 1) / 2 = -0.4
    period = 10.0 ** np.linspace(0.5, 4.5, 41)
    rho = 100.0 * (period / 100.0) ** 0.2
    mag = np.sqrt(rho / (0.2 * period))
    zz = np.zeros((len(period), 2, 2), complex)
    zz[:, 0, 1] = mag * np.exp(1j * np.radians(36.0))
    zz[:, 1, 0] = -zz[:, 0, 1]
    q2 = RD.quality(tf_of(period, zz, 0.02 * np.abs(zz)), "xy")
    assert q2["z_slope"] == pytest.approx(-0.4, abs=0.02)
    assert RD.earth(q2)[0]


# ------------------------------------------------------------------ the short slope clause

def test_the_short_slope_clause_is_unjudged_outside_the_live_band():
    """Fails if a live band that leaves no part of 2-20 s still produces a band for the clause."""
    assert RD.slope_band((30.0, 3000.0)) is None
    assert RD.slope_band((10.0, 3000.0)) == (10.0, 20.0)
    assert RD.slope_band(None) == RD.SHORT_SLOPE_BAND


def test_an_unjudged_short_slope_does_not_veto():
    """Fails if a row with no short slope measurable is refused for the clause it could not be scored on."""
    p, z, e = earth_tensor(lo=1.5, hi=4.5)      # nothing below 30 s
    q = RD.quality(tf_of(p, z, e), "xy", live=(30.0, 3000.0))
    assert not np.isfinite(q["short_slope"])
    assert RD.earth(q)[0]


# ------------------------------------------------------------------ agreement

def test_agreement_passes_a_ten_per_cent_copy_and_fails_a_double():
    """Fails if a 1.1x copy in rho is called a disagreement, or a 2x copy is called agreement."""
    p, z, e = earth_tensor()
    a = tf_of(p, z, e)
    b = tf_of(p, np.sqrt(1.1) * z, np.sqrt(1.1) * e)      # rho x 1.1
    c = tf_of(p, np.sqrt(2.0) * z, np.sqrt(2.0) * e)      # rho x 2
    assert RD.agree(b, a, "xy")["agrees"]
    assert not RD.agree(c, a, "xy")["agrees"]


def test_agreement_fails_a_phase_shift_beyond_the_tolerance():
    """Fails if a curve 10 deg away in phase at the same level is called agreement."""
    p, z, e = earth_tensor()
    a = tf_of(p, z, e)
    b = tf_of(p, z * np.exp(1j * np.radians(10.0)), e)
    s = RD.agree(b, a, "xy")
    assert s["phase_dev_deg"] == pytest.approx(10.0, abs=0.5)
    assert not s["agrees"]


# ------------------------------------------------------------------ the product of record

def _reading(site, comp, kind, bar, earth=True, agree_n=1, held_hi=1000.0, rate=1.0):
    return dict(site=site, component=comp, kind=kind, kind_word=kind, form="", run="r", stamp="s",
                rate_hz=rate, status="ok", earth=earth, not_earth_because="", agree_n=agree_n,
                agree_kinds="other", bar_10_1000=bar, held_hi_s=held_hi, held_n=10, path="")


def test_product_of_record_takes_the_smallest_bar_among_agreeing_earths():
    """Fails if a smaller bar on a row that is not an earth, or that corroborates nothing, is chosen."""
    import pandas as pd
    t = pd.DataFrame([
        _reading("S1", "xy", "single", 0.001, earth=False, agree_n=0),   # smallest bar, not an earth
        _reading("S1", "xy", "obs", 0.002, earth=True, agree_n=0),       # an earth, corroborates nothing
        _reading("S1", "xy", "remote", 0.004),
        _reading("S1", "xy", "stack", 0.003),
    ])
    rec = RD.product_of_record(t)
    row = rec[(rec.site == "S1") & (rec.component == "xy")].iloc[0]
    assert row.kind == "stack" and row.bar_10_1000 == pytest.approx(0.003)


def test_product_of_record_breaks_a_tie_on_the_longest_period_held():
    """Fails if two rows with the same bar are not separated by the longest period each holds."""
    import pandas as pd
    t = pd.DataFrame([_reading("S2", "yx", "remote", 0.005, held_hi=2000.0),
                      _reading("S2", "yx", "stack", 0.005, held_hi=9000.0)])
    rec = RD.product_of_record(t)
    row = rec[rec.site == "S2"].iloc[0]
    assert row.kind == "stack" and "longest period held" in row.why


def test_product_of_record_is_none_where_no_earth_agrees_with_another_kind():
    """Fails if a component whose earths corroborate nothing is given a product of record anyway."""
    import pandas as pd
    t = pd.DataFrame([_reading("S3", "xy", "single", 0.002, agree_n=0),
                      _reading("S3", "xy", "obs", 0.003, agree_n=0)])
    rec = RD.product_of_record(t)
    row = rec[rec.site == "S3"].iloc[0]
    assert row["product"] == "none" and "none agrees with a product of another kind" in row.why


# ------------------------------------------------------------------ the splice step

def test_the_step_at_the_join_passes_at_one_and_a_half_per_cent_and_fails_at_three():
    """Fails if a 1.5 per cent step is refused or a 3 per cent step is accepted, on either ruled band."""
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=4.5)
    base = tf_of(p, z, e)
    for pct, expect in ((1.5, True), (3.0, False)):
        s = np.sqrt(1.0 + pct / 100.0)
        short = tf_of(p, s * z, s * e)
        st = SP.step_at_join(base, short, "xy")
        assert st["step_pct"] == pytest.approx(pct, abs=0.1)
        assert (abs(st["step_pct"]) <= SP.SPLICE_MAX_STEP_PCT) is expect


def test_the_step_below_the_join_is_measurable_on_a_long_period_grid():
    """Fails if either ruled band comes back unmeasured on the grids this survey actually carries.

    A 1 Hz product carries about seven periods a decade and a 10 Hz product about eight, so the octave
    8-16 s below the join holds two or three periods. A minimum of four points there leaves the rule
    reading one band where it states two, which is how the acceptance would silently halve.
    """
    p1 = np.array([2.303, 2.949, 3.775, 4.833, 6.187, 7.921, 10.14, 12.98, 16.62, 21.28, 27.24, 34.87,
                   44.64, 57.14, 73.14, 93.62, 119.8, 153.4, 196.3])
    p10 = np.array([1.072, 1.320, 1.607, 1.982, 2.449, 3.033, 3.986, 5.146, 6.662, 8.601, 10.84, 13.78,
                    17.13, 20.58, 26.32, 33.69, 43.13, 55.21, 70.68, 90.49, 115.8])

    def curve_at(period, rho):
        mag = np.sqrt(rho / (0.2 * period))
        z = np.zeros((len(period), 2, 2), complex)
        z[:, 0, 1] = mag * np.exp(1j * np.radians(45.0))
        z[:, 1, 0] = -z[:, 0, 1]
        return tf_of(period, z, 0.02 * np.abs(z))

    base = curve_at(p1, 100.0)
    short = curve_at(p10, 100.0 * 1.015)
    st = SP.step_at_join(base, short, "xy")
    assert np.isfinite(st["step_below_pct"]) and st["n_below"] >= 2
    assert np.isfinite(st["step_above_pct"]) and st["n_above"] >= 2
    assert np.isfinite(st["guard_pct"])
    assert st["step_below_pct"] == pytest.approx(1.5, abs=0.2)


def test_the_guard_band_is_measured_and_scored_by_nothing():
    """Fails if a step confined to the 18-36 s guard band changes the step the acceptance reads."""
    p, z, e = earth_tensor(n=81, lo=-0.2, hi=4.5)
    base = tf_of(p, z, e)
    zz = z.copy()
    inside = (p >= SP.SPLICE_GUARD[0]) & (p <= SP.SPLICE_GUARD[1])
    zz[inside] *= np.sqrt(1.30)                      # a 30 per cent bump, inside the guard band only
    st = SP.step_at_join(base, tf_of(p, zz, e), "xy")
    assert st["guard_pct"] == pytest.approx(30.0, abs=1.0)
    assert abs(st["step_pct"]) < 0.5
    assert abs(st["step_pct"]) <= SP.SPLICE_MAX_STEP_PCT


def test_the_control_gate_promotes_only_a_selection_that_beats_its_random_control():
    """Fails if a selection no better than r25 is admitted, or one 30 per cent better is refused.

    The gate reads the 2-16 s impedance bar. A selection must beat the random 25 per cent of hours by 20
    per cent of the control's bar; the whole-record pass is admitted without the gate and r25 is never
    promoted.
    """
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1, bar=0.10)
    shorts = {
        ("single", "r25"): (tf_of(p, z, 0.10 * np.abs(z)), "r25.edi"),
        ("single", "f05"): (tf_of(p, z, 0.07 * np.abs(z)), "f05.edi"),      # 30 per cent better
        ("single", "f10"): (tf_of(p, z, 0.095 * np.abs(z)), "f10.edi"),     # 5 per cent better
        ("single", "whole"): (tf_of(p, z, 0.12 * np.abs(z)), "whole.edi"),
    }
    g = SP.control_gate(shorts, "xy")
    assert g[("single", "f05")]["eligible"]
    assert not g[("single", "f10")]["eligible"]
    assert "does NOT beat" in g[("single", "f10")]["verdict"]
    assert g[("single", "whole")]["eligible"] and "admitted without the gate" in \
        g[("single", "whole")]["verdict"]
    assert not g[("single", "r25")]["eligible"]


def test_the_control_gate_is_unjudged_where_no_random_control_exists():
    """Fails if a selection with no r25 product of its kind is admitted rather than reported UNJUDGED."""
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1)
    shorts = {("remote", "f10"): (tf_of(p, z, e), "f10.edi")}
    g = SP.control_gate(shorts, "xy")
    assert not g[("remote", "f10")]["eligible"]
    assert g[("remote", "f10")]["verdict"].startswith("UNJUDGED")


def test_the_rate_gate_refuses_a_component_before_any_row_is_read():
    """Fails if a component whose survey rate path is outside the ceiling is still offered a 10 Hz row."""
    import pandas as pd
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1)
    shorts = {("single", "whole", ""): (tf_of(p, z, e), "s.edi")}
    reads = pd.DataFrame([dict(site="S", component=c, kind="single", rate_hz=10.0, selection="whole",
                               form="", earth=True) for c in RD.COMPONENTS])
    out = SP.select_rows("S", tf_of(p, z, e), shorts, reads, rate_ok={"xy": False, "yx": True})
    assert out["xy"]["pick"] is None and "10 Hz path sits further" in out["xy"]["why"]
    assert out["yx"]["scored"], "the component the gate passed must still be scored"


def test_a_selection_name_is_read_off_the_product_file_name():
    """Fails if the selection tag is lost, or a whole-record name is read as carrying one."""
    a = PR.parse_product_name("Q73_stack_10hz_f05_kaiser20_75.edi")
    assert a["site"] == "Q73" and a["kind"] == "stack" and a["rate_hz"] == 10.0
    assert a["selection"] == "f05" and a["params"] == "kaiser20_75"
    b = PR.parse_product_name("Q49_stack_obs_1hz_kaiser20_75.edi")
    assert b["kind"] == "stack_obs" and b["selection"] == PR.WHOLE_SELECTION
    assert b["params"] == "kaiser20_75"
    assert PR.parse_product_name("Q53N_whole_remote_1hz_kaiser20_75.edi").get("kind") != "whole"


# ------------------------------------------------------------------ the unspliced rows

def test_an_unspliced_row_comes_back_identical(tmp_path):
    """Fails if any period or value of a component that was not spliced moves, to 1e-12 relative.

    The frozen tool rounded the union grid's VALUES and shifted one period of every unspliced row by
    2.3e-7 of itself (vic_splice test S5); the grid here is deduplicated on a rounded key and keeps the
    original value.
    """
    p, z, e = earth_tensor(n=41, lo=0.4, hi=4.1)
    base = write_edi(tmp_path, p, z, e, "BASE")
    ps, zs, es = earth_tensor(n=31, lo=0.0, hi=3.1)
    short = write_edi(tmp_path, ps, zs, es, "SHORT")
    pick = dict(kind="single", kind_word="single station", file=str(short), step_pct=0.5,
                step_above_pct=0.4, step_below_pct=0.5)
    out = tmp_path / "spliced.edi"
    SP.splice(base, {"xy": pick}, out)
    r = SP.unspliced_unchanged(base, out, spliced=("xy",))
    assert r["worst_period_relative"] <= SP.IDENTITY_TOL
    assert r["worst_value_relative"] <= SP.IDENTITY_TOL
    assert r["ok"]


def test_the_union_grid_keeps_the_original_period_values():
    """Fails if a period of the base grid is moved by the deduplication the union does."""
    base = np.array([2.3034803053585633, 4.0, 8.0, 16.0, 32.0])
    short = np.array([0.7, 1.0, 2.3034803053585633, 4.0, 20.0])
    g = SP.union_grid(base, short, floor=0.6, join=16.0)
    assert 2.3034803053585633 in set(g)
    assert np.all(np.diff(g) > 0)
    assert 20.0 not in set(g)          # above the join: the base row keeps that period
    assert 0.7 in set(g) and 1.0 in set(g)


def test_nothing_below_the_short_floor_is_delivered():
    """Fails if a 10 Hz period under SHORT_FLOOR_S reaches the union grid."""
    g = SP.union_grid(np.array([16.0, 32.0]), np.array([0.2, 0.4, 0.8, 2.0]),
                      floor=SP.SHORT_FLOOR_S, join=16.0)
    assert min(g) >= SP.SHORT_FLOOR_S


# ------------------------------------------------------------------ the merge

def test_merge_reads_back_and_the_two_sources_control_holds(tmp_path):
    """Fails if the written file's rows are not its sources' to 1e-9, or if the yx row is the xy source's.

    The two sources carry different yx rows, so the merged yx row must DIFFER from the xy source's: that is
    the control that the merge took rows from two files and not one.
    """
    p, z_xy, e_xy = earth_tensor(n=41, rho=100.0)
    _p2, z_yx, e_yx = earth_tensor(n=41, rho=250.0)
    tip = np.zeros((len(p), 1, 2), complex)
    tip[:, 0, 0] = 0.1 + 0.05j
    f_xy = write_edi(tmp_path, p, z_xy, e_xy, "SXY", tipper=tip)
    f_yx = write_edi(tmp_path, p, z_yx, e_yx, "SYX")

    class SV:
        cfg = dict(name="synthetic", project="synthetic", author="test")

        @staticmethod
        def site(_name):
            import pandas as pd
            return pd.Series(dict(site="SXY", declination_deg="8.978"))

    picks = {"xy": dict(path=str(f_xy), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar_10_1000=0.02),
             "yx": dict(path=str(f_yx), kind="stack", kind_word="fleet stack", form="", run="r",
                        stamp="s", rate_hz=1.0, bar_10_1000=0.03)}
    rec = FN.merge(SV, "SXY", picks, tmp_path / "final" / "SXY.edi", tipper_from="xy", verbose=False)
    assert rec["written"] and rec["ok"]
    assert rec["worst_readback_relative"] <= FN.READBACK_RTOL
    assert "control PASS" in rec["control"]

    back = PR.read_tf(rec["path"])
    a = PR.read_tf(f_xy)
    b = PR.read_tf(f_yx)
    m = np.isfinite(back.z[:, 0, 1])
    assert np.allclose(back.z[m, 0, 1], a.z[m, 0, 1], rtol=FN.READBACK_RTOL)
    assert np.allclose(back.z[m, 1, 0], b.z[m, 1, 0], rtol=FN.READBACK_RTOL)
    assert not np.allclose(back.z[m, 1, 0], a.z[m, 1, 0], rtol=1e-6)


def test_merge_writes_the_frame_block(tmp_path):
    """Fails if the written final carries no declination line or no angle to geographic north."""
    p, z, e = earth_tensor(n=41)
    f = write_edi(tmp_path, p, z, e, "FRAME")

    class SV:
        cfg = dict(name="synthetic", project="synthetic", author="test")

        @staticmethod
        def site(_name):
            import pandas as pd
            return pd.Series(dict(site="FRAME", declination_deg="8.978"))

    picks = {"xy": dict(path=str(f), kind="single", kind_word="single station", form="", run="r",
                        stamp="s", rate_hz=1.0, bar_10_1000=0.02)}
    rec = FN.merge(SV, "FRAME", picks, tmp_path / "final" / "FRAME.edi", verbose=False)
    kv = PR.read_tf(rec["path"]).meta["parameters"]
    assert "declination_deg" in kv and "RECORDED AND NOT APPLIED" in kv["declination_deg"]
    assert "to_geographic_north_deg" in kv and kv["to_geographic_north_deg"].startswith("-8.978")
    assert "reference_frame" in kv
    assert "no product of record" in kv.get("yx_rows", "")


def test_the_optional_resample_lands_on_the_ten_per_decade_grid(tmp_path):
    """Fails if the optional output is not on the common grid, or if it does not read back as a tensor.

    It is written beside the delivered file and never in its place: every value on it is interpolated.
    """
    from auslamp_proc import agreement as AG
    p, z, e = earth_tensor(n=41, rho=100.0, lo=0.6, hi=4.2)
    f = write_edi(tmp_path, p, z, e, "GRID")
    r = FN.resample(f, tmp_path / "GRID_resampled.edi")
    back = PR.read_tf(r["path"])
    assert len(back.period) == len(AG.GRID)
    assert np.allclose(np.sort(back.period), AG.GRID, rtol=1e-6)
    rho, _e, _ph, _pe = PR.rho_phase(back.period, back.z, back.z_err, "xy")
    per = np.asarray(back.period, float)
    got = np.isfinite(rho)
    assert got.sum() > 20
    assert np.allclose(rho[got], 100.0, rtol=1e-3)
    # nothing is extrapolated: every grid period carrying a value lies inside the source's own range
    # the EDI carries frequency, not period, and the round trip moves a grid value by about 1e-7 of itself
    edge = 1e-6
    assert per[got].min() >= p.min() * (1 - edge) and per[got].max() <= p.max() * (1 + edge)
    assert not np.isfinite(rho[per < p.min() * (1 - edge)]).any()
    kv = back.meta["parameters"]
    assert "resampled" in kv and "not the delivered file" in kv["resampled"]


def test_a_component_with_no_product_of_record_reads_back_empty(tmp_path):
    """Fails if the empty row of a delivered file reads back as a measurement of zero at every period.

    The EDI writer emits the 1e32 empty-data value for a row that carries nothing and the reader hands it
    back as Z = 0 with an error of 0, so a reader that took a zero bar for a measurement would see a
    two-dimensional site where one component was never delivered.
    """
    p, z, e = earth_tensor(n=41)
    f = write_edi(tmp_path, p, z, e, "ONECOMP")

    class SV:
        cfg = dict(name="synthetic", project="synthetic", author="test")

        @staticmethod
        def site(_name):
            import pandas as pd
            return pd.Series(dict(site="ONECOMP", declination_deg="8.978"))

    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar_10_1000=0.02)}
    rec = FN.merge(SV, "ONECOMP", picks, tmp_path / "final" / "ONECOMP.edi", verbose=False)
    back = PR.read_tf(rec["path"])
    assert np.isfinite(back.z[:, 0, 1]).sum() == len(p)
    assert np.isfinite(back.z[:, 1, 0]).sum() == 0
    assert np.isfinite(back.z[:, 1, 1]).sum() == 0


def test_an_info_line_carries_nothing_the_edi_reader_splits_on():
    """Fails if a pipe or a newline survives into a processing_parameters line.

    The EDI reader splits a Comment on the pipe, so a line carrying one comes back as two and the next
    read of the file raises on the fragment.
    """
    out = FN.clean("resampled=linear in log period of log|Z|\nand the unwrapped phase")
    assert "|" not in out and "\n" not in out
    assert out.startswith("resampled=")


def test_align_never_interpolates(tmp_path):
    """Fails if a source on another grid contributes a value at a period it does not carry."""
    p = 10.0 ** np.linspace(0.5, 4.5, 41)
    pc = 10.0 ** np.linspace(0.5, 4.5, 13)
    row = np.ones((len(pc), 2), complex)
    err = np.ones((len(pc), 2))
    out, oerr, n_ok, how = FN.align(p, pc, row, err)
    assert n_ok < len(p) and "aligned by nearest period" in how
    assert np.isfinite(out[:, 0]).sum() == n_ok
    # every value that was taken is one of the source's own, never a blend
    taken = out[np.isfinite(out[:, 0]), 0]
    assert np.allclose(taken, 1.0)
