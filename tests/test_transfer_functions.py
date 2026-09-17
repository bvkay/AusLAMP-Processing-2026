"""The reader, the tensor turn and the per-decade comparison, on synthetic transfer functions.

Every test states what would make it fail. Nothing here reads a survey's work root: each transfer function is
built in the test and written to an EDI through mt_metadata, so a failure names the code and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pytest

from auslamp_proc import agreement as AG, transfer_functions as TFN
from auslamp_proc.site import centre as CE


# ------------------------------------------------------------------ a synthetic transfer function

def synthetic(n=24, rho=100.0, shuffle=False, fill_at=()):
    """A one-dimensional tensor of `rho` Ohm.m at 45 deg, with an optional shuffle and 1e32 fill periods."""
    period = 10.0 ** np.linspace(0.5, 4.5, n)
    # rho = 0.2 T |Z|^2 with Z in mV/km/nT, so |Z| = sqrt(rho / (0.2 T)); the 1-D phase is 45 deg
    mag = np.sqrt(rho / (0.2 * period))
    zxy = mag * np.exp(1j * np.radians(45.0))
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 1] = zxy
    z[:, 1, 0] = -zxy
    # the diagonal of a one-dimensional tensor is zero, and its bar has to be a positive number: the EDI
    # writer emits the 1e32 empty-data value for a row carrying nothing and the reader hands that back as
    # Z = 0 with an error of 0, so a zero value with a zero bar is not distinguishable from an empty row in
    # the file at all. An estimator's jackknife bar on a measured zero is small and never zero.
    err = np.maximum(0.02 * np.abs(z), 1e-6 * float(np.abs(z).max()))
    for k in fill_at:
        z[k, 0, 1] = 1e32 + 0j
        err[k, 0, 1] = 1e32
    if shuffle:
        order = np.random.default_rng(7).permutation(n)
        period, z, err = period[order], z[order], err[order]
    return period, z, err


def write_edi(tmp_path, period, z, err, station="TEST"):
    """The synthetic tensor written to an EDI through mt_metadata, as a file on disk would be."""
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
                                      coords=dict(period=tf.period, output=["ex", "ey"], input=["hx", "hy"]))
    out = tmp_path / ("%s.edi" % station)
    tf.write(fn=str(out), file_type="edi")
    return out


# ------------------------------------------------------------------ read_tf

def test_read_tf_sorts_shuffled_periods(tmp_path):
    """Fails if read_tf returns periods that are not strictly increasing, or loses one."""
    period, z, err = synthetic(shuffle=True)
    path = write_edi(tmp_path, period, z, err, "SORTED")
    tf = TFN.read_tf(path)
    assert len(tf.period) == len(period)
    assert np.all(np.diff(tf.period) > 0)


def test_read_tf_drops_duplicate_periods(tmp_path):
    """Fails if a duplicated period survives the read, or the count of them is not recorded."""
    period, z, err = synthetic(n=12)
    period = np.concatenate([period, period[-3:]])
    z = np.concatenate([z, z[-3:]])
    err = np.concatenate([err, err[-3:]])
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "DUP"))
    assert len(tf.period) == len(set(np.round(tf.period, 9)))
    assert tf.meta["n_periods"] <= 12


def test_read_tf_masks_the_fill(tmp_path):
    """Fails if a 1e32 fill reaches the caller, or if it masks a component that carries a measurement."""
    period, z, err = synthetic(n=20, fill_at=(3, 11))
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "FILL"))
    assert not np.isfinite(tf.z[3, 0, 1]) and not np.isfinite(tf.z[11, 0, 1])
    assert np.isfinite(tf.z[3, 1, 0]) and np.isfinite(tf.z[11, 1, 0])
    assert np.nanmax(np.abs(tf.z)) < TFN.FILL


def test_rho_phase_folds_the_yx_quadrant(tmp_path):
    """Fails if the yx phase is not folded into 0-90 deg, or the xy phase is moved."""
    period, z, err = synthetic(n=16, rho=100.0)
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "FOLD"))
    _r, _re, ph_xy, _pe = TFN.rho_phase(tf.period, tf.z, tf.z_err, "xy")
    _r2, _re2, ph_yx, _pe2 = TFN.rho_phase(tf.period, tf.z, tf.z_err, "yx")
    assert np.allclose(ph_xy, 45.0, atol=1e-6)
    assert np.allclose(ph_yx, 45.0, atol=1e-6)


def test_read_tf_gives_the_apparent_resistivity_it_was_built_from(tmp_path):
    """Fails if rho read back from the written file is not the 100 Ohm.m the tensor was built at."""
    period, z, err = synthetic(n=16, rho=100.0)
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "RHO"))
    rho, _e, _p, _pe = TFN.rho_phase(tf.period, tf.z, tf.z_err, "xy")
    assert np.allclose(rho, 100.0, rtol=1e-6)


# ------------------------------------------------------------------ the tensor turn

@pytest.mark.parametrize("angle", [0.0, 8.978, -12.5, 90.0])
def test_the_column_turn_is_orthogonal(angle):
    """Fails if the arm-diagonal turn changes the length of a row of Z or of the tipper by more than 1e-9.

    The turn workbook 04's diagonal frame applies is a turn of the H COLUMNS alone, Z' = Z R^T, because the
    pass on the NE cache already carries the E row turn. A turn of the columns is orthogonal, so the sum of
    the squared magnitudes along each row is invariant; it is not a similarity transform, so Zxy - Zyx and
    Zxx + Zyy are not invariant and are not asserted here. A turn that moved a row's length would be a scale,
    a reflection or an index slip.
    """
    rng = np.random.default_rng(3)
    z = rng.normal(size=(30, 2, 2)) + 1j * rng.normal(size=(30, 2, 2))
    t = rng.normal(size=(30, 1, 2)) + 1j * rng.normal(size=(30, 1, 2))
    zt, tt = CE.turn_columns(z, t, angle)
    for row in (0, 1):
        before = np.abs(z[:, row, 0]) ** 2 + np.abs(z[:, row, 1]) ** 2
        after = np.abs(zt[:, row, 0]) ** 2 + np.abs(zt[:, row, 1]) ** 2
        assert np.allclose(after, before, rtol=1e-9), "row %d" % row
    assert np.allclose(np.abs(tt[:, 0, 0]) ** 2 + np.abs(tt[:, 0, 1]) ** 2,
                       np.abs(t[:, 0, 0]) ** 2 + np.abs(t[:, 0, 1]) ** 2, rtol=1e-9)


def test_turn_and_back_is_the_identity(tmp_path):
    """Fails if turning a tensor and turning it back does not return it to 1e-9 of the tensor's own size.

    The bound is relative to the largest element at each period, not to the element itself: a 1-D tensor has
    Zxx = Zyy = 0 exactly, and a relative bound on zero is a bound no arithmetic can meet.
    """
    period, z, err = synthetic(n=18)
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "TURN"))
    there, _t = CE.turn_columns(tf.z, tf.t, 9.5)
    back, _b = CE.turn_columns(there, None, -9.5)
    ok = np.isfinite(tf.z)
    scale = np.nanmax(np.abs(tf.z).reshape(len(tf.period), -1), axis=1)[:, None, None]
    scale = np.broadcast_to(scale, tf.z.shape)
    assert ok.sum() == tf.z.size
    assert np.all(np.abs(back[ok] - tf.z[ok]) <= 1e-9 * scale[ok])


# ------------------------------------------------------------------ on_grid and per_decade

def test_on_grid_is_exact_at_shared_periods(tmp_path):
    """Fails if interpolating a curve onto its own periods changes any value by more than 1e-9 relative."""
    period, z, err = synthetic(n=20)
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "GRIDA"))
    same = AG.on_grid(tf, tf)
    ok = np.isfinite(tf.z) & np.isfinite(same.z)
    assert ok.sum() >= 2 * len(tf.period) - 4
    assert np.all(np.abs(same.z[ok] - tf.z[ok]) <= 1e-9 * np.abs(tf.z[ok]))


def test_per_decade_against_itself_is_one_and_zero(tmp_path):
    """Fails if a tensor scored against itself gives a rho ratio away from 1 or a phase away from 0."""
    period, z, err = synthetic(n=24)
    tf = TFN.read_tf(write_edi(tmp_path, period, z, err, "SELF"))
    rows = AG.per_decade(tf, tf)
    scored = rows[rows.n > 0]
    assert len(scored) >= 4
    assert np.allclose(scored.rho_ratio.values, 1.0, atol=1e-9)
    assert np.allclose(scored.phase_diff_deg.values, 0.0, atol=1e-9)


def test_per_decade_against_a_doubled_rho_gives_a_ratio_of_two(tmp_path):
    """Fails if a curve at twice the apparent resistivity does not give a ratio of 2.0 with no phase
    change."""
    period, z, err = synthetic(n=24, rho=100.0)
    a = TFN.read_tf(write_edi(tmp_path, period, z, err, "ONEX"))
    b = TFN.read_tf(write_edi(tmp_path, period, z * np.sqrt(2.0), err * np.sqrt(2.0), "TWOX"))
    rows = AG.per_decade(a, b)                       # a over b: a is the 1x curve, b the 2x curve
    scored = rows[rows.n > 0]
    assert np.allclose(scored.rho_ratio.values, 0.5, rtol=1e-6)
    assert np.allclose(scored.phase_diff_deg.values, 0.0, atol=1e-6)
    rows2 = AG.per_decade(b, a)
    scored2 = rows2[rows2.n > 0]
    assert np.allclose(scored2.rho_ratio.values, 2.0, rtol=1e-6)
    now = AG.band_stats(b, a, "xy", 5.0, 200.0)
    assert np.isclose(now["rho_ratio"], 2.0, rtol=1e-6) and now["spread"] <= 1.001


def test_agreement_rule_uses_its_own_thresholds():
    """Fails if a row inside the thresholds is not called agreement, or one outside either of them is."""
    assert AG.agrees(dict(rho_ratio=1.15, phase_diff_deg=3.0, n=12))
    assert not AG.agrees(dict(rho_ratio=1.35, phase_diff_deg=3.0, n=12))
    assert not AG.agrees(dict(rho_ratio=1.05, phase_diff_deg=7.0, n=12))
    assert not AG.agrees(dict(rho_ratio=1.05, phase_diff_deg=1.0, n=0))


def test_smoothness_finds_a_planted_jump(tmp_path):
    """Fails if a single period moved by a factor of three in rho is not counted as a jump, or if the clean
    curve it was planted in scores one."""
    period, z, err = synthetic(n=28)
    clean = TFN.read_tf(write_edi(tmp_path, period, z, err, "CLEAN"))
    z2 = z.copy()
    z2[14, 0, 1] *= np.sqrt(3.0)
    rough = TFN.read_tf(write_edi(tmp_path, period, z2, err, "ROUGH"))
    s_clean = AG.smoothness(clean)
    s_rough = AG.smoothness(rough)
    assert int(s_clean[s_clean.component == "xy"].n_jumps.iloc[0]) == 0
    assert int(s_rough[s_rough.component == "xy"].n_jumps.iloc[0]) >= 1
