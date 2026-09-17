"""The arithmetic of auslamp_proc.site, on synthetic records with a known answer.

Every test states what it would take to fail. None of them reads a survey tree: a test that needs the data
to be on this machine is a test that is skipped on every other one.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auslamp_proc.process import frame as FR
from auslamp_proc.site import centre, deliver, forms, masks, replace, variants

RNG = np.random.default_rng(20260916)


# ---------------------------------------------------------------- the selections and their controls

def test_day_mask_control_is_the_same_size_from_the_same_pool():
    """Fails if the control keeps a different number of days from the selection, or draws a day the map did
    not score."""
    n_days, n = 20, 20 * 86400
    daily = np.full((n_days, 4), np.nan)
    daily[:, 2] = np.linspace(0.1, 0.9, n_days)
    daily[3, 2] = np.nan                                     # one day the map could not score
    qm = dict(daily=daily, n=n, days=["d%02d" % i for i in range(n_days)], site="X")
    keep, ctrl, rows = masks.day_mask(qm, "xy", thr=0.5, seed=20260916)
    assert rows["days_kept"] == int(np.sum(np.nan_to_num(daily[:, 2]) >= 0.5))
    assert rows["days_control"] == rows["days_kept"]
    assert keep.sum() == ctrl.sum()
    drawn = {d for d in rows["control_days"].split() if d}
    assert "d03" not in drawn                                 # the unscored day is not in the pool


def test_day_mask_control_is_reproducible_from_its_seed():
    """Fails if two draws under the same seed differ, or if two different seeds give the same draw."""
    daily = np.full((30, 4), np.nan)
    daily[:, 2] = RNG.uniform(0.2, 0.9, 30)
    qm = dict(daily=daily, n=30 * 86400, days=["d%02d" % i for i in range(30)], site="X")
    a = masks.day_mask(qm, "xy", 0.5, seed=1)[2]["control_days"]
    b = masks.day_mask(qm, "xy", 0.5, seed=1)[2]["control_days"]
    c = masks.day_mask(qm, "xy", 0.5, seed=2)[2]["control_days"]
    assert a == b and a != c


def test_select_hours_ranked_takes_the_best_and_random_does_not():
    """Fails if the ranked selection does not keep the highest-scoring windows, or if the random selection
    keeps the same set."""
    n = 100 * 3600
    tc = np.arange(100) * 3600.0 + 1800.0
    score = np.linspace(0.0, 1.0, 100)
    keep, thr, k = masks.select_hours(tc, score, 0.25, 3600, n)
    rnd, thr_r, k_r = masks.select_hours(tc, score, 0.25, 3600, n, rng=np.random.default_rng(7))
    assert k == k_r == 25
    assert thr == pytest.approx(score[75])
    assert np.isnan(thr_r)
    assert keep[-25 * 3600:].all()                            # the top 25 windows are the last ones
    assert not keep[:75 * 3600].any()
    assert not np.array_equal(keep, rnd)


def test_contiguous_control_matches_the_duration_within_one_window():
    """Fails if a contiguous window set's kept duration misses the scattered selection's by more than one
    window."""
    tc = np.arange(96) * 3600.0 + 1800.0
    score = RNG.uniform(0, 1, 96)
    n = 96 * 3600
    _keep, _thr, k = masks.select_hours(tc, score, 0.25, 3600, n)
    for hours in (2, 4, 6, 24):
        _m, row = masks.contiguous_windows(tc, score, hours, n, k)
        assert abs(row["hours"] - k) <= hours, (hours, row)


def test_sustained_low_gives_back_a_hole_shorter_than_min_hole():
    """Fails if a short low run is masked, or a long one is not."""
    step, win, n = 1800.0, 3600.0, 200 * 3600
    tc = np.arange(0, n - int(win), int(step)) + win / 2
    v = np.full(len(tc), 0.9)
    v[10:12] = 0.0                                            # one hour low: given back
    v[100:140] = 0.0                                          # twenty hours low: masked
    keep, ivl = masks.sustained_low(tc, v, step, win, n, thresh=0.5, smooth_h=1.0, min_hole_h=6.0)
    assert len(ivl) == 1
    a, b = ivl[0]
    assert (b - a) >= 6 * 3600
    assert keep[:10 * 3600].all()


def test_window_from_days_takes_the_longest_run_and_falls_back():
    """Fails if a shorter sound run is preferred, or if a table with no sound day yields a window without
    saying it admitted the weak days."""
    t0 = 1_700_000_000
    states = ["weak", "sound", "sound", "weak", "sound", "sound", "sound", "dead"]
    el = pd.DataFrame([dict(day="d%d" % i, t_start=t0 + i * 86400, t_end=t0 + (i + 1) * 86400,
                            Ex_state=s, Ey_state="weak") for i, s in enumerate(states)])
    w = masks.window_from_days(el, "xy")
    assert w["n_days"] == 3 and w["t_start"] == t0 + 4 * 86400
    w2 = masks.window_from_days(el, "yx")
    assert w2["n_days"] == 8 and "weak days are admitted" in w2["reason"]


# ---------------------------------------------------------------- the shared centre and the arm diagonal

def test_ne_algebra_is_exact():
    """Fails if Ex' differs from (Ex - Ey)/sqrt2 or Ey' from (Ex + Ey)/sqrt2 at any finite sample."""
    ex = RNG.normal(size=5000)
    ey = RNG.normal(size=5000)
    exp, eyp = (ex - ey) / centre.SQ2, (ex + ey) / centre.SQ2
    assert np.allclose(exp, (ex - ey) / np.sqrt(2.0), rtol=0, atol=0)
    assert np.allclose(eyp, (ex + ey) / np.sqrt(2.0), rtol=0, atol=0)
    # the inverse: the pair is a rotation of (Ex, Ey) by -45 deg, so the norm is kept
    assert np.allclose(exp ** 2 + eyp ** 2, ex ** 2 + ey ** 2)


def test_equal_arms_reduce_to_the_ported_numbers():
    """Fails if equal arms do not give an expected gain of 1, a diagonal at -45 deg of length L sqrt2, and
    the (Ex -+ Ey)/sqrt2 pair the Victoria tool uses -- the general form must contain the frozen one."""
    assert centre.expected_gain(50.0, 50.0) == 1.0
    assert abs(centre.diagonal_angle(50.0, 50.0) - centre.THETA_NE) < 1e-12
    assert abs(centre.diagonal_length(50.0, 50.0) - 50.0 * np.sqrt(2.0)) < 1e-12
    ex, ey = RNG.normal(size=4000), RNG.normal(size=4000)
    e_d, e_v = centre.diagonals(ex, ey, 50.0, 50.0)
    assert np.allclose(e_d, (ex - ey) / np.sqrt(2.0), rtol=1e-12, atol=0)
    assert np.allclose(e_v, (ex + ey) / np.sqrt(2.0), rtol=1e-12, atol=0)


def test_the_diagonal_direction_and_length_follow_the_arms():
    """Fails if the arm diagonal of a 9 m north arm and a 12 m east arm is not at atan2(-12, 9) from north
    over a separation of 15 m, or if the pair is not a rotation of (Ex, Ey) by that angle."""
    ln, le = 9.0, 12.0
    theta = centre.diagonal_angle(ln, le)
    assert abs(theta - np.degrees(np.arctan2(-le, ln))) < 1e-12
    assert abs(theta - (-53.13010235)) < 1e-6, theta
    assert abs(centre.diagonal_length(ln, le) - 15.0) < 1e-12
    assert abs(theta - centre.THETA_NE) > 8.0, "unequal arms must NOT sit at the equal-arm -45 deg"
    ex, ey = RNG.normal(size=4000), RNG.normal(size=4000)
    e_d, e_v = centre.diagonals(ex, ey, ln, le)
    # a rotation keeps the norm, which is what says the pair is a frame turn and not a rescaling
    assert np.allclose(e_d ** 2 + e_v ** 2, ex ** 2 + ey ** 2)


def test_the_arm_diagonal_cancels_an_injected_centre_and_the_equal_arm_one_does_not():
    """Fails if (L_N Ex - L_E Ey)/d does not remove an injected centre voltage to 1e-9 of the true field at
    unequal arms, or if (Ex - Ey)/sqrt2 -- the equal-arm form -- removes it there."""
    ln, le, n = 9.0, 12.0, 20000
    ex_true, ey_true = RNG.normal(size=n), RNG.normal(size=n)
    v = 5.0 * np.cumsum(RNG.normal(size=n))          # the centre electrode's own voltage
    c = -v / ln
    ex, ey = ex_true + c, ey_true + c * (ln / le)    # s = +1: arms north and east
    d_true = centre.diagonals(ex_true, ey_true, ln, le)[0]
    d_got = centre.diagonals(ex, ey, ln, le)[0]
    assert np.max(np.abs(d_got - d_true)) < 1e-9 * max(1.0, np.max(np.abs(d_true))), \
        np.max(np.abs(d_got - d_true))
    naive = (ex - ey) / np.sqrt(2.0) - (ex_true - ey_true) / np.sqrt(2.0)
    assert np.max(np.abs(naive)) > 1e-3 * np.max(np.abs(c)), "the equal-arm diagonal must leave a residue"


def test_the_residual_test_holds_at_unequal_arms_where_the_equal_arm_form_would_not():
    """Fails if a pair sharing one centre on 9 m and 12 m arms does not read a gain near L_N/L_E = 0.75 with
    a scaled gain inside 0.85-1.18 -- and it fails the other way if 0.75 lies inside the equal-arm band,
    which would mean the change of criterion could not have changed a verdict."""
    ln, le = 9.0, 12.0
    g_expected = centre.expected_gain(ln, le)
    assert abs(g_expected - 0.75) < 1e-12
    n = 4 * centre.NPERSEG
    hx, hy = RNG.normal(size=n), RNG.normal(size=n)
    c = RNG.normal(size=n) * 3.0
    ex = 2.0 * hy + 0.1 * RNG.normal(size=n) + c
    ey = -2.0 * hx + 0.1 * RNG.normal(size=n) + c * (ln / le)
    st = centre.day_stats(ex, ey, hx, hy, band_s=(4.0, 200.0), L_N=ln, L_E=le)
    assert st["coh_r"] >= centre.RESID_COH_MIN, st
    assert abs(st["gain"] - g_expected) < 0.05 * g_expected, st
    ratio = st["gain"] / g_expected
    assert centre.GAIN_LO <= ratio <= centre.GAIN_HI, (ratio, st)
    # the criterion this replaces: |g| judged against 1, which refuses the same site
    assert not (centre.GAIN_LO <= st["gain"] <= centre.GAIN_HI), \
        "the equal-arm criterion must refuse this pair, or the change of criterion is untested"
    # and the diagonal the arms define is the one the H coherence prefers
    assert st["mcoh_diff"] > st["mcoh_sum"], st


def test_the_turn_back_holds_its_invariants_at_the_arm_angle():
    """Fails if the turn-back at theta = atan2(-L_E, L_N) moves an element beyond 1e-6 relative, the
    determinant beyond 1e-5 of ||Z||^2 or the Frobenius norm beyond 1e-6 -- the tolerances are the -45 deg
    ones and must hold at any angle -- or if the trace comes out invariant there."""
    theta = centre.diagonal_angle(9.0, 12.0)
    Z = RNG.normal(size=(40, 2, 2)) + 1j * RNG.normal(size=(40, 2, 2))
    Zr, _ = centre.turn_columns(Z, None, theta)
    inv = centre.turn_invariants(Z, Zr, angle_deg=theta)
    assert inv["ok"], inv
    assert abs(inv["trace_ratio"] - 1.0) > 1e-3, "the trace is NOT an invariant of Z -> Z R^T"
    # and the -45 deg turn-back does NOT verify a turn taken at the arm angle
    assert not centre.turn_invariants(Z, Zr, angle_deg=centre.THETA_NE)["ok"], \
        "checking at the wrong angle must fail, or the angle is not carried into the check"


def test_turn_back_keeps_the_element_determinant_and_norm_and_not_the_trace():
    """Fails if a column-only turn moves an element beyond 1e-6 relative, the determinant beyond 1e-5 of
    ||Z||^2 or the Frobenius norm beyond 1e-6 -- or if the trace comes out invariant, which would mean the
    test cannot tell a column turn from a similarity."""
    p = np.geomspace(1, 10000, 40)
    Z = (RNG.normal(size=(40, 2, 2)) + 1j * RNG.normal(size=(40, 2, 2)))
    Zr, _ = centre.turn_columns(Z, None, centre.THETA_NE)
    inv = centre.turn_invariants(Z, Zr)
    assert inv["ok"], inv
    assert inv["max_element_rel"] <= centre.ELEMENT_RTOL
    assert inv["max_det_scaled"] <= centre.DET_SCALED_TOL
    assert inv["max_frobenius_rel"] <= centre.FROBENIUS_RTOL
    assert abs(inv["trace_ratio"] - 1.0) > 1e-3, "the trace is NOT an invariant of Z -> Z R^T"
    assert len(p) == len(Z)


def test_residual_test_holds_on_a_built_shared_centre_and_not_on_an_independent_pair():
    """Fails if a synthetic pair sharing one centre voltage does not read a residual coherence of at least
    0.9 with a gain near 1 and the sign it was built with, or if an independent pair does."""
    n = 4 * centre.NPERSEG
    hx, hy = RNG.normal(size=n), RNG.normal(size=n)
    for s in (+1.0, -1.0):
        c = RNG.normal(size=n) * 3.0
        ex = 2.0 * hy + 0.1 * RNG.normal(size=n) + c
        ey = -2.0 * hx + 0.1 * RNG.normal(size=n) + s * c
        st = centre.day_stats(ex, ey, hx, hy, band_s=(4.0, 200.0))
        assert st["coh_r"] >= 0.9, (s, st)
        assert centre.GAIN_LO <= st["gain"] <= centre.GAIN_HI, (s, st)
        assert st["s_obs"] == s, (s, st)
    ex = 2.0 * hy + RNG.normal(size=n)
    ey = -2.0 * hx + RNG.normal(size=n)
    st = centre.day_stats(ex, ey, hx, hy, band_s=(4.0, 200.0))
    assert st["coh_r"] < 0.9, st


def test_the_shifted_pair_reads_low_where_the_unshifted_pair_reads_high():
    """Fails if shifting one of two coherent records by 12 h does not drop the 100-1000 s coherence under
    0.3 while the unshifted pair stays above 0.8 -- which is what makes the shifted pair a control."""
    from auslamp_proc.look import highpass
    n = 4 * 86400
    field = np.cumsum(RNG.normal(size=n))                     # one regional field both sites see
    a_rec = field + 0.05 * np.cumsum(RNG.normal(size=n))
    b_rec = field + 0.05 * np.cumsum(RNG.normal(size=n))
    hp = highpass(1.0, masks.HIGHPASS_S)
    win = 2 * 86400
    x = masks.prepare(a_rec[:win], *hp)
    same = masks.prepare(b_rec[:win], *hp)
    shifted = masks.prepare(b_rec[43200:43200 + win], *hp)
    unshifted_coh = masks.band_coherence(x, same, 1.0, masks.FLEET_BAND_S)
    shifted_coh = masks.band_coherence(x, shifted, 1.0, masks.FLEET_BAND_S)
    assert unshifted_coh > 0.8, unshifted_coh
    assert shifted_coh < masks.NEGATIVE_MAX, shifted_coh


def test_the_clock_finds_a_fraction_of_a_second_and_never_an_edge_lag():
    """Fails if a pair carrying a strong daily variation and a true lag of 0.4 s does not come back at
    0.4 s, if the peak lands within 5 per cent of the +-12 h search edge, or if the peak does not stand
    1.5x above that day's own other lags."""
    from scipy import signal
    fs, day, lag_true = 1.0, 86400, 0.4
    L = int(masks.CLOCK_MAXLAG_S)
    n = day + 2 * L
    t = np.arange(n) / fs
    b, a = signal.butter(4, [1 / 20.0, 1 / 5.0], btype="band", fs=fs)
    band = signal.filtfilt(b, a, RNG.normal(size=n))
    diurnal = 60.0 * np.sin(2 * np.pi * t / day) + 20.0 * np.sin(4 * np.pi * t / day)
    ref = band + diurnal
    # the site stamps the same field 0.4 s early, so its samples must move 0.4 s LATER and the estimator's
    # convention (vic_windows.cmd_clock: positive = the site's samples move later) returns +0.4
    site = np.interp(t + lag_true, t, ref)
    x = masks.clock_prepare(site[L:L + day], fs)
    y = masks.clock_prepare(ref, fs)
    c = signal.correlate(y, x, mode="valid", method="fft") / (
        len(x) * np.std(x) * np.std(y[L:L + day]) + 1e-12)
    k = int(np.argmax(c))
    got = float(k - L) + masks._parabolic(c, k)
    assert abs(got - lag_true) < 0.15, got
    assert abs(k - L) < masks.CLOCK_EDGE_FRACTION * L, (k - L, "an edge hit")
    assert c[k] / float(np.median(np.abs(c))) >= masks.CLOCK_PEAK_RATIO, c[k]


def test_the_built_centre_control_does_not_hold_the_model():
    """Fails if two electric lines with independent centre voltages -- the pair the built control is made of
    -- read a residual coherence of 0.9 or more with a gain inside 0.85-1.18, or if a pair sharing one
    centre on the same magnetics does not hold."""
    n = 4 * centre.NPERSEG
    hx, hy = RNG.normal(size=n), RNG.normal(size=n)
    c = RNG.normal(size=n) * 3.0
    shared = centre.day_stats(2.0 * hy + 0.1 * RNG.normal(size=n) + c,
                              -2.0 * hx + 0.1 * RNG.normal(size=n) + c, hx, hy, band_s=(4.0, 200.0))
    assert shared["coh_r"] >= centre.RESID_COH_MIN, shared
    built = centre.day_stats(2.0 * hy + 0.1 * RNG.normal(size=n) + c,
                             -2.0 * hx + 0.1 * RNG.normal(size=n) + RNG.normal(size=n) * 3.0,
                             hx, hy, band_s=(4.0, 200.0))
    holds = bool(built["coh_r"] >= centre.RESID_COH_MIN
                 and centre.GAIN_LO <= built["gain"] <= centre.GAIN_HI)
    assert not holds, built


def test_sign_prediction_is_never_filled_by_convention():
    """Fails if an undecided E sign produces a prediction instead of leaving the site UNJUDGED."""
    s, und = centre.sign_prediction({"sign_ex": "-1", "sign_ey": "-1"})
    assert s == 1.0 and not und
    s, und = centre.sign_prediction({"sign_ex": "+1", "sign_ey": "-1"})
    assert s == -1.0 and not und
    s, und = centre.sign_prediction({"sign_ex": "decide", "sign_ey": "-1"})
    assert np.isnan(s) and und == ["Ex"]


# ---------------------------------------------------------------- the borrowed channel

def test_mixed_basis_reduces_to_the_rotation_and_to_the_identity():
    """Fails if the correction for two borrowed channels is not the rotation R(-t), if the correction for
    none is not the identity, or if the one-of-each case is a rotation."""
    t = 17.5
    both = replace.basis_correction(t, (True, True))
    none = replace.basis_correction(t, (False, False))
    one = replace.basis_correction(t, (True, False))
    assert np.allclose(both, FR.rotation_matrix(-t))
    assert np.allclose(none, np.eye(2))
    assert np.allclose(replace.mixed_basis(t, (True, False)),
                       [[1.0, 0.0], [-np.sin(np.radians(t)), np.cos(np.radians(t))]])
    assert not np.allclose(one @ one.T, np.eye(2)), "a mixed basis is not corrected by a rotation"


def test_the_correction_recovers_the_geomagnetic_tensor():
    """Fails if a tensor estimated on a mixed H basis, corrected, differs from the one estimated on the
    site's own mean-field pair."""
    t = 12.0
    R = FR.rotation_matrix(t)
    Z_geo = RNG.normal(size=(5, 2, 2)) + 1j * RNG.normal(size=(5, 2, 2))
    for borrowed in ((True, False), (False, True), (True, True), (False, False)):
        M = replace.mixed_basis(t, borrowed)
        # E = Z_geo H_geo = Z_geo R(t) H_sensor and H_used = M H_sensor, so Z_used = Z_geo R(t) M^-1
        Z_used = np.einsum("nij,jk,kl->nil", Z_geo, R, np.linalg.inv(M))
        back, _ = replace.correct_basis(Z_used, None, replace.basis_correction(t, borrowed))
        assert np.allclose(back, Z_geo), borrowed


def test_to_sensor_frame_is_the_inverse_of_the_rotation():
    """Fails if turning a mean-field pair into the sensor frame and back does not return it."""
    t = 23.0
    geo = {"Hx": RNG.normal(size=500), "Hy": RNG.normal(size=500)}
    sensor = replace.to_sensor_frame(geo, t)
    back_x, back_y = FR.rotate_pair(sensor["Hx"], sensor["Hy"], t)
    assert np.allclose(back_x, geo["Hx"]) and np.allclose(back_y, geo["Hy"])


def test_lender_in_reference_is_refused():
    """Fails if a lender that is a member of the reference is not caught."""
    info = dict(remote="Q63", members=[dict(name="Q84", role="member"), dict(name="Q50", role="refused")])
    assert replace.lender_in_reference("Q63", info)
    assert replace.lender_in_reference("Q84", info)
    assert not replace.lender_in_reference("Q50", info)
    assert not replace.lender_in_reference("Q99", info)


# ---------------------------------------------------------------- the cache variants

def test_notch_moves_only_the_tone_bin():
    """Fails if the tone is not removed, or if any frequency outside the notch's own skirt -- taken as 10 per
    cent of f0 either side, the width tone_ratio reads its sidebands over -- moves by more than 2.5 per cent
    of its power."""
    from scipy import signal
    fs, n = 10.0, 1 << 19
    t = np.arange(n) / fs
    x = RNG.normal(size=n) + 5.0 * np.sin(2 * np.pi * 1.0 * t)
    before = variants.tone_ratio(x, 1.0, fs)
    y, st = variants.notch_gap_aware(x, (1.0, 2.0), fs)
    after = variants.tone_ratio(y, 1.0, fs)
    assert before > variants.RATIO_FIRE and after < before / 10.0, (before, after)
    f, Pb = signal.welch(x, fs=fs, nperseg=1 << 14)
    _f, Pa = signal.welch(y, fs=fs, nperseg=1 << 14)
    tone = np.zeros(len(f), bool)
    for f0 in (1.0, 2.0):
        tone |= np.abs(f - f0) <= 0.10 * f0
    off = ~tone & (f > 0)
    worst = float(np.max(np.abs(Pa[off] / Pb[off] - 1.0)))
    assert worst <= 0.025, worst
    assert st["samples_passed_through"] == 0


def test_sha256_of_an_unchanged_channel_is_its_source():
    """Fails if the byte identity a variant claims can be true of a changed array."""
    x = RNG.normal(size=1000).astype(np.float32)
    assert variants.sha256_array(x) == variants.sha256_array(x.copy())
    y = x.copy()
    y[10] = np.float32(y[10] + 1.0)
    assert variants.sha256_array(x) != variants.sha256_array(y)


# ---------------------------------------------------------------- the merge and the delivery

def test_merge_component_names_only_its_own_two_rows():
    """Fails if the row map of a component is not that component's two rows."""
    assert forms.ROWS_OF["xy"] == ((0, 1), (0, 0))
    assert forms.ROWS_OF["yx"] == ((1, 0), (1, 1))
    assert set(forms.ROWS_OF["xy"]) & set(forms.ROWS_OF["yx"]) == set()


def test_merge_component_replaces_exactly_two_rows(tmp_path):
    """Fails if a merge changes any row other than the windowed component's two."""
    mt = pytest.importorskip("mt_metadata.transfer_functions.core")
    p = np.geomspace(1.0, 1000.0, 20)
    base = _write_edi(tmp_path / "base.edi", p, 1.0, mt)
    win = _write_edi(tmp_path / "win.edi", p, 2.0, mt)
    got = forms.merge_component(base, win, "xy", out_edi=tmp_path / "merged.edi", verbose=False)
    assert got["untouched_unchanged"], got
    assert set(got["rows_changed"]) <= {"xx", "xy"}, got
    assert got["ok"]


def _write_edi(path, period, scale, mt):
    from mt_metadata.transfer_functions.core import TF
    n = len(period)
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 0] = scale * (1 + 1j)
    z[:, 0, 1] = scale * (10 + 5j)
    z[:, 1, 0] = -(3 + 2j)
    z[:, 1, 1] = (0.5 + 0.1j)
    tf = TF()
    tf.station_metadata.id = "TEST"
    tf.period = period
    tf.impedance = z
    tf.impedance_error = np.full((n, 2, 2), 0.1)
    tf.write(fn=str(path), file_type="edi")
    return path


def test_tipper_refusal_fires_on_hz_equal_to_hx():
    """Fails if a vertical channel that is a copy of Hx is not refused."""
    n = 200000
    hx = np.cumsum(RNG.normal(size=n))
    arr = dict(Hx=hx, Hy=np.cumsum(RNG.normal(size=n)), Hz=hx.copy())
    from auslamp_proc.look import highpass
    b, a = highpass(1.0, 3000.0)
    prep = {c: masks.prepare(arr[c], b, a) for c in arr}
    coh = masks.band_coherence(prep["Hz"], prep["Hx"], 1.0, deliver.LEAK_BAND_S)
    assert coh >= deliver.COPY_COH, coh
    coh_other = masks.band_coherence(prep["Hy"], prep["Hx"], 1.0, deliver.LEAK_BAND_S)
    assert coh_other < deliver.COPY_COH, coh_other


def test_tipper_only_refuses_a_refused_tipper_and_a_product_without_one(tmp_path):
    """Fails if a refused tipper is written anyway, or if a product carrying no tipper yields a file."""
    pytest.importorskip("mt_metadata.transfer_functions.core")
    p = np.geomspace(1.0, 1000.0, 20)
    src = _write_edi(tmp_path / "src.edi", p, 1.0, None)
    got = deliver.tipper_only(src, tmp_path / "a.edi", "remote site",
                              dict(refused=True, reason="Hz reads 1.00 with its own Hx: a copy"))
    assert got["refused"] and not got["written"] and not (tmp_path / "a.edi").exists()
    got = deliver.tipper_only(src, tmp_path / "b.edi", "remote site", dict(refused=False))
    assert got["refused"] and "no tipper" in got["reason"]


def test_beats_needs_the_stated_margin():
    """Fails if a bar equal to its control's, or better by less than the margin, is called a win."""
    assert deliver.beats(0.05, 0.10, 0.20)
    assert not deliver.beats(0.09, 0.10, 0.20)
    assert not deliver.beats(0.10, 0.10, 0.20)
    assert not deliver.beats(np.nan, 0.10, 0.20)


def test_forms_table_refuses_to_promote_a_form_without_a_control():
    """Fails if a form carrying no control is marked a candidate."""
    rows = [dict(site="X", form="whole", kind="remote", rate_hz=1.0, params="k", status="made",
                 product="none.edi", controls="", criterion="", seed=None)]
    t = deliver.forms_table(rows)
    assert not bool(t.candidate.iloc[0])
    assert "no control" in t.verdict.iloc[0]


def test_a_refused_form_and_a_crashed_one_do_not_read_the_same():
    """Fails if a form the floor refused before the pass reads the same on the table as one that crashed --
    the distinction is that the first is a reading with numbers and the second is a failure."""
    why = ("refused: the notched screen leaves 0 run(s) of 3600 s against the remote (the whole record "
           "keeps 55.59 d over 87 run(s))")
    rows = [dict(site="X", form="notched", kind="remote", rate_hz=10.0, params="k", status="refused",
                 product="", controls="whole10", criterion="", seed=None, reason=why, error=""),
            dict(site="X", form="other", kind="remote", rate_hz=10.0, params="k", status="FAILED",
                 product="", controls="whole10", criterion="", seed=None,
                 error="ValueError: aurora returned no transfer function")]
    t = deliver.forms_table(rows).set_index("form")
    assert t.loc["notched", "verdict"] == why and "0 run(s)" in t.loc["notched", "verdict"]
    assert t.loc["other", "verdict"] == "NOT MADE"
    assert not bool(t.loc["notched", "candidate"]) and not bool(t.loc["other", "candidate"])


def test_refusal_sentence_carries_the_numbers_that_refused_the_form():
    """Fails if the sentence a refused form carries does not name the runs it left, the floor they had to
    clear, the reference and what the whole record kept: a refusal without numbers is an opinion."""
    s = forms.refusal_sentence(dict(n_runs=0, min_segment_s=3600.0, kind="remote", days=0.0, empty=True),
                               dict(n_runs=87, days=55.592), what="the notched screen")
    for piece in ("the notched screen", "0 run(s)", "3600 s", "remote", "55.59 d", "87 run(s)"):
        assert piece in s, (piece, s)
