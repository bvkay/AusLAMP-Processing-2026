"""The three response tests, the transfer function of record, the choice record, the splice, the merge.

Every test states what would make it fail. Nothing here reads a survey's work root: each transfer function is
built in the test and, where a file is needed, written to an EDI through mt_metadata, so a failure names the
code and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pytest

from auslamp_proc import final as FN, transfer_functions as TFN, readings as RD, splice as SP


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


def power_law(n=41, rho0=100.0, slope=1.5, phase=80.0, bar=0.02, lo=0.5, hi=4.5):
    """A curve whose apparent resistivity rises as T^`slope`, with the phase held in its quadrant."""
    period = 10.0 ** np.linspace(lo, hi, n)
    rho = rho0 * (period / 100.0) ** float(slope)
    mag = np.sqrt(rho / (0.2 * period))
    zxy = mag * np.exp(1j * np.radians(phase))
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 1] = zxy
    z[:, 1, 0] = -zxy
    return period, z, float(bar) * np.abs(z)


def tf_of(period, z, err, t=None, t_err=None, path="synthetic"):
    return TFN.TFData(np.asarray(period, float), np.asarray(z, complex), np.asarray(err, float),
                     t, t_err, dict(path=path, parameters={}))


def write_edi(tmp_path, period, z, err, station="TEST", tipper=None):
    """A synthetic tensor written to an EDI through mt_metadata, as a delivered file on disk would be."""
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


class SurveyStub:
    """The two things final.merge asks a survey for: its config and one row of sites.csv."""

    cfg = dict(name="synthetic", project="synthetic", author="test")

    @staticmethod
    def site(name):
        import pandas as pd
        return pd.Series(dict(site=str(name), declination_deg="8.978"))


# ------------------------------------------------------------------ the three response tests

def test_a_clean_one_dimensional_curve_passes_all_three_tests():
    """Fails if a 100 Ohm.m curve at 45 deg with a 2 per cent bar does not pass the three tests."""
    p, z, e = earth_tensor()
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["pass_phase"] and q["pass_slope"] and q["pass_error"]
    assert q["passes"] and q["fails"] == ""
    assert q["phase_frac"] == 1.0 and q["slope_frac"] == 1.0
    assert 0 < q["median_phase_deg"] < 90


def test_a_flipped_sign_fails_the_phase_test():
    """Fails if a row with its sign flipped is still called in quadrant, or the failure is not named."""
    p, z, e = earth_tensor()
    q = RD.quality(tf_of(p, -z, e), "xy")
    assert not q["pass_phase"] and not q["passes"]
    assert q["fails"].startswith("phase") and "sign fault" in q["fails"]


def test_a_flipped_yx_row_fails_the_phase_test():
    """Fails if the yx row of a tensor whose Zyx has the wrong sign still folds into the first quadrant."""
    p, z, e = earth_tensor(flip_yx=True)
    q = RD.quality(tf_of(p, z, e), "yx")
    assert not q["pass_phase"] and not q["passes"]


def test_the_phase_test_tolerates_a_few_periods_out_of_quadrant():
    """Fails if a curve with one period in ten out of quadrant is refused at QUADRANT_MIN = 0.70."""
    p, z, e = earth_tensor(n=41)
    zz = z.copy()
    zz[::10] *= -1                                  # four of the forty-one periods flipped
    q = RD.quality(tf_of(p, zz, e), "xy")
    assert q["phase_frac"] >= RD.QUADRANT_MIN and q["pass_phase"]


def test_a_curve_rising_as_t_to_the_three_halves_fails_the_slope_test():
    """Fails if rho proportional to T^1.5 passes the bound of one decade a decade, or the name is wrong.

    |d log rho / d log T| <= 1 holds for a one-dimensional earth (Weidelt 1972; Parker and Booker 1996);
    the test widens it by SLOPE_TOL for noise, so 1.5 is outside it at every adjacent pair.
    """
    p, z, e = power_law(slope=1.5)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["slope_frac"] == 0.0 and not q["pass_slope"]
    assert not q["passes"] and "slope" in q["fails"]
    assert q["pass_phase"] and q["pass_error"], "only the slope test may refuse this curve"


def test_a_curve_inside_the_bound_passes_the_slope_test():
    """Fails if rho proportional to T^0.9, which a one-dimensional earth can produce, is refused."""
    p, z, e = power_law(slope=0.9, phase=50.0)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["slope_frac"] == 1.0 and q["pass_slope"]


def test_a_bar_of_two_fails_the_error_test():
    """Fails if a curve whose error bars are twice its own impedance is still called sound."""
    p, z, e = earth_tensor(bar=2.0)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["bar"] == pytest.approx(2.0, rel=1e-6) and q["bar"] > RD.BAR_MAX
    assert q["n_periods"] == 0 and not q["pass_error"]
    assert not q["passes"] and "error" in q["fails"]
    assert q["pass_phase"] and q["pass_slope"], "only the error test may refuse this curve"


def test_the_error_test_needs_min_periods_under_the_ceiling():
    """Fails if a curve whose bar is under the ceiling at fewer than MIN_PERIODS of the band passes."""
    p, z, e = earth_tensor(n=41, bar=0.02)
    band = (p >= RD.QUALITY_BAND[0]) & (p <= RD.QUALITY_BAND[1])
    idx = np.where(band)[0]
    ee = e.copy()
    ee[idx[: len(idx) - (RD.MIN_PERIODS - 1)]] *= 200.0     # leaves MIN_PERIODS - 1 under the ceiling
    q = RD.quality(tf_of(p, z, ee), "xy")
    assert q["n_periods"] == RD.MIN_PERIODS - 1 and not q["pass_error"]


def test_the_rho_bar_is_twice_the_impedance_bar():
    """Fails if the two bars are not named apart: rho = 0.2 T |Z|^2, so its relative error is twice."""
    p, z, e = earth_tensor(bar=0.03)
    tf = tf_of(p, z, e)
    assert RD.rho_bar(tf, "xy") == pytest.approx(2.0 * RD.bar(tf, "xy"), rel=1e-9)


def test_the_held_band_is_reported_and_is_not_a_test():
    """Fails if a curve holding at no period is refused, or if its held band is not reported empty.

    The held band is what the delivered file is trimmed to, not a clause: a curve with bars over the held
    ceiling everywhere is still put to the three tests on its own terms.
    """
    p, z, e = earth_tensor(bar=0.6)
    q = RD.quality(tf_of(p, z, e), "xy")
    assert q["held_n"] == 0 and not np.isfinite(q["held_lo_s"])
    assert q["pass_phase"] and q["pass_slope"] and q["pass_error"] and q["passes"]


def test_the_single_station_is_not_a_deliverable_kind():
    """Fails if the single station is among the kinds a transfer function may be delivered on."""
    import pandas as pd
    assert RD.DROPPED_KIND == "single" and "single" not in RD.KINDS
    assert set(RD.KINDS) == {"remote", "stack", "obs", "stack_obs"}
    tfs = pd.DataFrame([dict(site="S", kind=k, rate_hz=1.0, on_disk=True) for k in
                        ("single", "remote", "stack")])
    keep, dropped = RD.deliverable(tfs)
    assert set(keep.kind) == {"remote", "stack"} and list(dropped.kind) == ["single"]


def test_a_form_no_section_of_workbook_04_makes_is_ignored_and_not_read():
    """Fails if a stray form enters the rows a workbook reads, or if a form workbook 04 makes is dropped.

    The set is stated once, in site.forms.FORM_NAMES. `daymask_xy` and `hours_yx_random` are two of the ten
    the removed day-mask and best-hours sections left in Q53N's run folder; `window_xy_control`, `diagonal`
    and `replace_Hx_Q52N` are three the workbook still makes.
    """
    import pandas as pd
    from auslamp_proc.site import forms as FM
    keeps = ["", "whole", "whole10", "diagonal", "recipe", "recipe_x", "window_xy", "window_xy_control",
             "merged_yx", "replace_Hx_Q52N", "replace_Hy_stack", "lender_Q52N"]
    strays = ["daymask_xy", "daymask_yx_control", "hours_xy", "hours_yx_random", "hours_xy_contig2h"]
    assert all(FM.is_form_name(f) for f in keeps if f), "a form the workbook makes is not in the set"
    assert not any(FM.is_form_name(f) for f in strays), "a stray is in the set"
    tfs = pd.DataFrame([dict(site="S", kind="remote", rate_hz=1.0, on_disk=True, form=f)
                        for f in keeps + strays])
    keep, dropped = RD.deliverable(tfs)
    assert list(keep.form) == keeps, "a form the workbook makes was dropped"
    assert list(dropped.form) == strays, "a stray was read"
    assert set(dropped.why) == {"a form no section of workbook 04 makes"}


def test_a_tf_name_is_read_off_its_kind_rate_selection_or_form():
    """Fails if the name the choice cell uses does not pick a row out of the readings table."""
    import pandas as pd
    rows = pd.DataFrame([dict(kind="stack", rate_hz=1.0, selection="whole", form=""),
                         dict(kind="remote", rate_hz=10.0, selection="stretch", form=""),
                         dict(kind="remote", rate_hz=1.0, selection="whole", form="window_yx")])
    got = [RD.tf_key(r) for r in rows.itertuples()]
    assert got == ["stack_1hz", "remote_10hz_stretch", "window_yx"]


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


# ------------------------------------------------------------------ the transfer function of record

def _reading(site, comp, kind, bar, passes=True, agree_n=1, held_hi=1000.0, rate=1.0):
    return dict(site=site, component=comp, kind=kind, kind_word=kind, selection="whole", form="",
                run="r", stamp="s", rate_hz=rate, status="ok",
                transfer_function="%s_%ghz" % (kind, rate),
                passes=passes, fails="", agree_n=agree_n, agree_kinds="other", bar=bar, n_periods=20,
                held_lo_s=10.0, held_hi_s=held_hi, held_n=10, path="")


def test_the_record_takes_the_smallest_bar_among_agreeing_sound_rows():
    """Fails if a smaller bar on a row that fails a test, or corroborates nothing, is chosen."""
    import pandas as pd
    t = pd.DataFrame([
        _reading("S1", "xy", "remote", 0.001, passes=False, agree_n=0),   # smallest bar, fails a test
        _reading("S1", "xy", "obs", 0.002, passes=True, agree_n=0),       # sound, corroborates nothing
        _reading("S1", "xy", "stack_obs", 0.004),
        _reading("S1", "xy", "stack", 0.003),
    ])
    rec = RD.transfer_function_of_record(t)
    row = rec[(rec.site == "S1") & (rec.component == "xy")].iloc[0]
    assert row.kind == "stack" and row.bar == pytest.approx(0.003)
    assert row["transfer_function"] == "stack_1hz"


def test_the_record_breaks_a_tie_on_the_longest_period_held():
    """Fails if two rows with the same bar are not separated by the longest period each holds."""
    import pandas as pd
    t = pd.DataFrame([_reading("S2", "yx", "remote", 0.005, held_hi=2000.0),
                      _reading("S2", "yx", "stack", 0.005, held_hi=9000.0)])
    rec = RD.transfer_function_of_record(t)
    row = rec[rec.site == "S2"].iloc[0]
    assert row.kind == "stack" and "longest period held" in row.why


def test_the_record_is_none_where_nothing_agrees_with_another_kind():
    """Fails if a component whose sound rows corroborate nothing is given a transfer function of record."""
    import pandas as pd
    t = pd.DataFrame([_reading("S3", "xy", "remote", 0.002, agree_n=0),
                      _reading("S3", "xy", "obs", 0.003, agree_n=0)])
    rec = RD.transfer_function_of_record(t)
    row = rec[rec.site == "S3"].iloc[0]
    assert row["transfer_function"] == "none"
    assert "none agrees with a row of another kind" in row.why


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

    A 1 Hz pass carries about seven periods a decade and a 10 Hz pass about eight, so the octave
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
    """Fails if a selection no better than its control is admitted, or one 30 per cent better is refused.

    The gate reads the 2-16 s impedance bar. A selection must beat its random control by 20 per cent of the
    control's bar; the whole-record pass is admitted without the gate and the control itself is never
    promoted. The control's tag is read from SP.CONTROL_SELECTION, so the test states the rule and not the
    spelling of the tag.
    """
    ctrl = SP.CONTROL_SELECTION
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1, bar=0.10)
    shorts = {
        ("remote", ctrl): (tf_of(p, z, 0.10 * np.abs(z)), "remote_control.edi"),
        ("remote", "stretch"): (tf_of(p, z, 0.07 * np.abs(z)), "remote_stretch.edi"),  # 30 per cent better
        ("obs", ctrl): (tf_of(p, z, 0.10 * np.abs(z)), "obs_control.edi"),
        ("obs", "stretch"): (tf_of(p, z, 0.095 * np.abs(z)), "obs_stretch.edi"),       # 5 per cent better
        ("remote", "whole"): (tf_of(p, z, 0.12 * np.abs(z)), "whole.edi"),
    }
    g = SP.control_gate(shorts, "xy")
    assert g[("remote", "stretch")]["eligible"]
    assert not g[("obs", "stretch")]["eligible"]
    assert "does NOT beat" in g[("obs", "stretch")]["verdict"]
    assert g[("remote", "whole")]["eligible"] and "admitted without the gate" in \
        g[("remote", "whole")]["verdict"]
    assert not g[("remote", ctrl)]["eligible"]


def test_the_control_gate_is_unjudged_where_no_random_control_exists():
    """Fails if a selection with no control of its kind is admitted rather than reported UNJUDGED."""
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1)
    shorts = {("remote", "stretch"): (tf_of(p, z, e), "a.edi")}
    g = SP.control_gate(shorts, "xy")
    assert not g[("remote", "stretch")]["eligible"]
    assert g[("remote", "stretch")]["verdict"].startswith("UNJUDGED")


def test_the_rate_gate_refuses_a_component_before_any_row_is_read():
    """Fails if a component whose survey rate path is outside the ceiling is still offered a 10 Hz row."""
    import pandas as pd
    p, z, e = earth_tensor(n=61, lo=-0.2, hi=3.1)
    shorts = {("remote", "whole", ""): (tf_of(p, z, e), "s.edi")}
    reads = pd.DataFrame([dict(site="S", component=c, kind="remote", rate_hz=10.0, selection="whole",
                               form="", passes=True) for c in RD.COMPONENTS])
    out = SP.select_rows("S", tf_of(p, z, e), shorts, reads, rate_ok={"xy": False, "yx": True})
    assert out["xy"]["pick"] is None and "10 Hz path sits further" in out["xy"]["why"]
    assert out["yx"]["scored"], "the component the gate passed must still be scored"


def test_a_selection_name_is_read_off_the_file_name():
    """Fails if the selection tag is lost, or a whole-record name is read as carrying one."""
    a = TFN.parse_tf_name("Q73_stack_10hz_stretch_kaiser20_75.edi")
    assert a["site"] == "Q73" and a["kind"] == "stack" and a["rate_hz"] == 10.0
    assert a["selection"] == "stretch" and a["params"] == "kaiser20_75"
    b = TFN.parse_tf_name("Q49_stack_obs_1hz_kaiser20_75.edi")
    assert b["kind"] == "stack_obs" and b["selection"] == TFN.WHOLE_SELECTION
    assert b["params"] == "kaiser20_75"
    assert TFN.parse_tf_name("Q53N_whole_remote_1hz_kaiser20_75.edi").get("kind") != "whole"


# ------------------------------------------------------------------ the unjoined rows

def test_an_unspliced_row_comes_back_identical(tmp_path):
    """Fails if any period or value of a component that was not spliced moves, to 1e-12 relative.

    The union grid is deduplicated on a rounded key and keeps the original value. Rounding the values
    themselves shifts one period of every unspliced row by 2.3e-7 of itself (vic_splice test S5).
    """
    p, z, e = earth_tensor(n=41, lo=0.4, hi=4.1)
    base = write_edi(tmp_path, p, z, e, "BASE")
    ps, zs, es = earth_tensor(n=31, lo=0.0, hi=3.1)
    short = write_edi(tmp_path, ps, zs, es, "SHORT")
    pick = dict(kind="remote", kind_word="remote site", file=str(short), step_pct=0.5,
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

    picks = {"xy": dict(path=str(f_xy), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02),
             "yx": dict(path=str(f_yx), kind="stack", kind_word="fleet stack", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.03)}
    rec = FN.merge(SurveyStub, "SXY", picks, tmp_path / "final" / "SXY.edi", tipper_from="xy",
                   verbose=False)
    assert rec["written"] and rec["ok"]
    assert rec["worst_readback_relative"] <= FN.READBACK_RTOL
    assert "control PASS" in rec["control"]

    back = TFN.read_tf(rec["path"])
    a = TFN.read_tf(f_xy)
    b = TFN.read_tf(f_yx)
    m = np.isfinite(back.z[:, 0, 1])
    assert np.allclose(back.z[m, 0, 1], a.z[m, 0, 1], rtol=FN.READBACK_RTOL)
    assert np.allclose(back.z[m, 1, 0], b.z[m, 1, 0], rtol=FN.READBACK_RTOL)
    assert not np.allclose(back.z[m, 1, 0], a.z[m, 1, 0], rtol=1e-6)


def test_merge_writes_the_frame_block(tmp_path):
    """Fails if the written final carries no declination line or no angle to geographic north."""
    p, z, e = earth_tensor(n=41)
    f = write_edi(tmp_path, p, z, e, "FRAME")
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    rec = FN.merge(SurveyStub, "FRAME", picks, tmp_path / "final" / "FRAME.edi", verbose=False)
    kv = TFN.read_tf(rec["path"]).meta["parameters"]
    assert "declination_deg" in kv and "RECORDED AND NOT APPLIED" in kv["declination_deg"]
    assert "to_geographic_north_deg" in kv and kv["to_geographic_north_deg"].startswith("-8.978")
    assert "reference_frame" in kv
    assert "no transfer function of record" in kv.get("yx_rows", "")


def test_the_optional_resample_lands_on_the_ten_per_decade_grid(tmp_path):
    """Fails if the optional output is not on the common grid, or if it does not read back as a tensor.

    It is written beside the delivered file and never in its place: every value on it is interpolated.
    """
    from auslamp_proc import agreement as AG
    p, z, e = earth_tensor(n=41, rho=100.0, lo=0.6, hi=4.2)
    f = write_edi(tmp_path, p, z, e, "GRID")
    r = FN.resample(f, tmp_path / "GRID_resampled.edi")
    back = TFN.read_tf(r["path"])
    assert len(back.period) == len(AG.GRID)
    assert np.allclose(np.sort(back.period), AG.GRID, rtol=1e-6)
    rho, _e, _ph, _pe = TFN.rho_phase(back.period, back.z, back.z_err, "xy")
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


def test_a_component_with_no_transfer_function_of_record_reads_back_empty(tmp_path):
    """Fails if the empty row of a delivered file reads back as a measurement of zero at every period.

    The EDI writer emits the 1e32 empty-data value for a row that carries nothing and the reader hands it
    back as Z = 0 with an error of 0, so a reader that took a zero bar for a measurement would see a
    two-dimensional site where one component was never delivered.
    """
    p, z, e = earth_tensor(n=41)
    f = write_edi(tmp_path, p, z, e, "ONECOMP")
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    rec = FN.merge(SurveyStub, "ONECOMP", picks, tmp_path / "final" / "ONECOMP.edi", verbose=False)
    back = TFN.read_tf(rec["path"])
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


# ------------------------------------------------------------------ the trim

def test_the_trim_drops_exactly_the_periods_outside_the_held_band(tmp_path):
    """Fails if a period inside the chosen band is dropped, or one outside it is delivered.

    The periods are compared to 1e-6 relative and not exactly: an EDI carries frequency, so the round trip
    through the writer and the reader moves a period value by about 1e-7 of itself.
    """
    p, z, e = earth_tensor(n=41, lo=0.0, hi=4.5)
    f = write_edi(tmp_path, p, z, e, "TRIM")
    band = (10.0, 1000.0)
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    rec = FN.merge(SurveyStub, "TRIM", picks, tmp_path / "final" / "TRIM.edi",
                   keep_band={"xy": band}, verbose=False)
    back = TFN.read_tf(rec["path"])
    inside = p[(p >= band[0]) & (p <= band[1])]
    outside = p[(p < band[0]) | (p > band[1])]
    assert len(back.period) == len(inside)
    assert np.allclose(np.sort(back.period), np.sort(inside), rtol=1e-6)
    assert rec["n_dropped"] == len(outside)
    assert np.allclose(np.sort(rec["dropped_periods_s"]), np.sort(outside), rtol=1e-6)
    assert "outside the delivered band" in rec["dropped_reason"]


def test_the_manifest_names_the_dropped_periods(tmp_path):
    """Fails if the delivered file's manifest row does not carry the dropped periods and the reason."""
    import pandas as pd
    p, z, e = earth_tensor(n=41, lo=0.0, hi=4.5)
    f = write_edi(tmp_path, p, z, e, "MAN")
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    rec = FN.merge(SurveyStub, "MAN", picks, tmp_path / "final" / "MAN.edi",
                   keep_band={"xy": (10.0, 1000.0)}, verbose=False)
    rec["picks"] = picks
    out = FN.write_record(tmp_path / "survey", pd.DataFrame(columns=RD.RECORD_COLUMNS),
                          pd.DataFrame(columns=RD.READINGS_COLUMNS), [rec], sites=["MAN"])
    row = out["manifest"][out["manifest"].role == "final"].iloc[0]
    assert int(row.n_dropped) == rec["n_dropped"] > 0
    assert len(str(row.dropped_periods_s).split()) == rec["n_dropped"]
    assert "outside the delivered band" in str(row.dropped_reason)
    assert row.sha256 == FN.sha256(rec["path"])


def test_a_band_that_would_empty_the_file_is_not_applied(tmp_path):
    """Fails if a band holding no period of the grid writes a file with nothing in it."""
    p, z, e = earth_tensor(n=41, lo=0.5, hi=4.5)
    f = write_edi(tmp_path, p, z, e, "EMPTY")
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    rec = FN.merge(SurveyStub, "EMPTY", picks, tmp_path / "final" / "EMPTY.edi",
                   keep_band={"xy": (1e5, 2e5)}, verbose=False)
    assert rec["n_dropped"] == 0 and "not trimmed" in rec["dropped_reason"]
    assert len(TFN.read_tf(rec["path"]).period) == len(p)


# ------------------------------------------------------------------ the choice record

def test_the_choice_record_round_trips_and_reproduces_the_merge(tmp_path):
    """Fails if a choice row written is not read back with the same transfer function, band and join.

    The row is the record of a delivery: a re-run that reads it and merges again must land on the same
    file, byte for byte in its impedance rows.
    """
    p, z, e = earth_tensor(n=41, lo=0.0, hi=4.5)
    f = write_edi(tmp_path, p, z, e, "ROUND")
    band = (10.0, 1000.0)
    picks = {"xy": dict(path=str(f), kind="remote", kind_word="remote site", form="", run="r",
                        stamp="s", rate_hz=1.0, bar=0.02)}
    first = FN.merge(SurveyStub, "ROUND", picks, tmp_path / "final" / "ROUND.edi",
                     keep_band={"xy": band}, verbose=False)

    path = tmp_path / "final_choices.csv"
    FN.write_choices(path, [FN.choice_row("ROUND", "xy", "remote_1hz", periods=band, join=None,
                                          chosen_by="analyst", note="the analyst took the remote row")])
    back = FN.read_choices(path)
    row = back[(back.site == "ROUND") & (back.component == "xy")].iloc[0]
    assert row["transfer_function"] == "remote_1hz" and row.chosen_by == "analyst"
    assert (float(row.periods_lo), float(row.periods_hi)) == band
    assert str(row["join"]).strip() in ("", "nan")

    again = FN.merge(SurveyStub, "ROUND", picks, tmp_path / "final" / "ROUND_again.edi",
                     keep_band={"xy": (float(row.periods_lo), float(row.periods_hi))}, verbose=False)
    a, b = TFN.read_tf(first["path"]), TFN.read_tf(again["path"])
    assert np.allclose(a.period, b.period, rtol=1e-12)
    m = np.isfinite(a.z[:, 0, 1])
    assert np.allclose(a.z[m, 0, 1], b.z[m, 0, 1], rtol=1e-12)


def test_an_analyst_row_is_never_overwritten_by_the_rule(tmp_path):
    """Fails if a rule row replaces an analyst row, or if a site this run did not deliver loses its row.

    The workbook delivers one site per run and writes back to the survey's one choice file, so a re-run must
    leave every other site's row as it stands and must never undo a choice an analyst made.
    """
    path = tmp_path / "final_choices.csv"
    FN.write_choices(path, [FN.choice_row("A", "xy", "remote_1hz", chosen_by="analyst", note="mine"),
                            FN.choice_row("B", "xy", "stack_1hz", chosen_by="rule")])
    out = FN.write_choices(path, [FN.choice_row("A", "xy", "stack_1hz", chosen_by="rule"),
                                  FN.choice_row("B", "xy", "obs_1hz", chosen_by="rule"),
                                  FN.choice_row("C", "xy", "obs_1hz", chosen_by="rule")])
    t = out["table"]
    assert out["analyst_kept"] == 1 and out["written"] == 2
    assert t[(t.site == "A")].iloc[0]["transfer_function"] == "remote_1hz"
    assert t[(t.site == "A")].iloc[0].chosen_by == "analyst"
    assert t[(t.site == "B")].iloc[0]["transfer_function"] == "obs_1hz"
    assert set(t.site) == {"A", "B", "C"}


def test_the_manifest_check_catches_a_file_that_changed(tmp_path):
    """Fails if a delivered file rewritten after the manifest was taken still matches its sha256."""
    import pandas as pd
    p, z, e = earth_tensor(n=41)
    f = write_edi(tmp_path, p, z, e, "SHA")
    man = pd.DataFrame([dict(site="SHA", role="final", component="", file=str(f), kind="", form="",
                             run="", stamp="", rate_hz=np.nan, n_periods=len(p), n_dropped=0,
                             dropped_periods_s="", dropped_reason="", bytes=f.stat().st_size,
                             sha256=FN.sha256(f))])
    assert bool(FN.manifest_check(man).iloc[0].sha256_matches)
    write_edi(tmp_path, p, np.sqrt(2.0) * z, e, "SHA")
    row = FN.manifest_check(man).iloc[0]
    assert row.exists and row.readable and not row.sha256_matches
