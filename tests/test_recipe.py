"""The arithmetic of auslamp_proc.site.recipe, on synthetic tables and synthetic transfer functions.

Every test states what it would take to fail. None of them reads a survey tree, except the one that asks a
survey object with no cache on disk for a 10 Hz row and reads the refusal it gets back.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auslamp_proc.process import selection as SEL
from auslamp_proc.site import masks as MK
from auslamp_proc.site import recipe

HOUR = 3600.0
T0 = 1772920800                                     # 2026-03-07 22:00:00 UTC, a whole UTC hour


def _table(coh_xy, coh_yx, t0=T0):
    """An hour score table of two given series, as process.selection.score_hours returns one."""
    n = len(coh_xy)
    t = np.arange(n) * HOUR + float(t0)
    xy, yx = np.asarray(coh_xy, float), np.asarray(coh_yx, float)
    out = pd.DataFrame(dict(hour=np.arange(n), t_start=t.astype(int), t_end=(t + HOUR).astype(int),
                            utc=[str(x) for x in t.astype(int)], coh_xy=xy, coh_yx=yx,
                            score=np.nanmean(np.vstack([xy, yx]), axis=0)))
    out.attrs["band_s"] = list(SEL.SCORE_BAND_S)
    return out


class _Survey:
    """The two things row_hours asks a survey for: the work root and the span of one cache."""

    def __init__(self, work_root, t0=T0, n=60 * 86400):
        self.cfg = dict(work_root=str(work_root))
        self._span = (int(t0), int(n))


# ---------------------------------------------------------------- the hours a row is passed on

def test_whole_is_the_record_and_carries_no_control():
    """Fails if the whole record is given a window or a control; there is nothing a stretch of the same
    length elsewhere would be compared against."""
    got = recipe.row_hours(None, "X", recipe.HOURS_WHOLE, row="x")
    assert got["rule"] == "whole"
    assert got["window"] is None and got["control"] is None


def test_window_coherent_reads_the_row_s_own_line_through_the_one_rule(monkeypatch, tmp_path):
    """Fails if the recipe chooses hours by any rule other than process.selection.longest_stretch, or if the
    x row is not read on Ex against Hy and the y row on Ey against Hx."""
    xy = np.full(40, 0.9)
    yx = np.full(40, 0.1)                           # the y line never rises
    table = _table(xy, yx)
    monkeypatch.setattr(MK, "span", lambda sv, site, rate=1: (T0, 60 * 86400))
    sv = _Survey(tmp_path)
    x = recipe.row_hours(sv, "X", recipe.HOURS_COHERENT, row="x", table=table, seed=SEL.SEED)
    assert x["line"] == "xy"
    assert x["window"]["hours"] == 40
    assert x["control"]["hours"] == 40
    assert x["control"]["t_end"] <= x["window"]["t_start"] or         x["control"]["t_start"] >= x["window"]["t_end"]
    y = recipe.row_hours(sv, "X", recipe.HOURS_COHERENT, row="y", table=table, seed=SEL.SEED)
    assert y["line"] == "yx"
    assert y["window"]["t_start"] is None and y["control"] is None


def test_a_named_window_is_parsed_and_not_scored(monkeypatch, tmp_path):
    """Fails if a window the caller names is re-chosen by the rule, or if a malformed one raises instead of
    stating the reason."""
    monkeypatch.setattr(MK, "span", lambda sv, site, rate=1: (T0, 60 * 86400))
    got = recipe.row_hours(_Survey(tmp_path), "X", "window:2026-03-22 12:00/14", row="y")
    assert got["window"]["t_end"] - got["window"]["t_start"] == 14 * HOUR
    assert got["control"]["hours"] == 14
    bad = recipe.row_hours(_Survey(tmp_path), "X", "window:not-a-time/14", row="y")
    assert bad["window"]["t_start"] is None and "does not read as" in bad["window"]["reason"]


def test_the_hours_options_are_three_and_a_fourth_is_refused(tmp_path):
    """Fails if a fraction of the scored hours is still an option a row may name."""
    assert recipe.HOURS_OPTIONS == ("whole", "window:coherent", "window:<ISO UTC start>/<hours>")
    for gone in ("f05", "f10", "f25", "r25"):
        with pytest.raises(ValueError):
            recipe.row_hours(_Survey(tmp_path), "X", gone, row="x")


# ---------------------------------------------------------------- the frame

def test_the_frame_is_one_of_two_and_a_third_is_refused():
    """Fails if a frame the recipe does not carry is accepted."""
    with pytest.raises(ValueError):
        recipe.frame_spec(None, "X", "geographic")


# ---------------------------------------------------------------- the missing input

def test_a_missing_10hz_cache_refuses_with_the_reason(tmp_path):
    """Fails if a row whose 10 Hz cache or 10 Hz reference store is not on disk is passed anyway, or if the
    refusal does not name which input is missing."""
    sv = type("S", (), {})()
    sv.cfg = dict(work_root=str(tmp_path))
    why = recipe.missing_inputs(sv, "Q58N", 10, "", "remote")
    assert why.startswith("refused:")
    assert "no 10 Hz cache" in why
    assert "no 10 Hz remote reference store" in why
    assert "the single station is not a kind of this package" in why
    # the cache on disk and the store still missing: the store alone is named
    (tmp_path / "cache_10hz").mkdir()
    (tmp_path / "cache_10hz" / "Q58N.npz").write_bytes(b"")
    why = recipe.missing_inputs(sv, "Q58N", 10, "", "remote")
    assert "no 10 Hz cache" not in why and "no 10 Hz remote reference store" in why
    # the frame's variant of the cache is named on its own
    why = recipe.missing_inputs(sv, "Q58N", 10, "ne", "remote")
    assert "no 10 Hz ne variant cache" in why
    # every input on disk leaves nothing to refuse
    (tmp_path / "cache_10hz_ne").mkdir()
    (tmp_path / "cache_10hz_ne" / "Q58N.npz").write_bytes(b"")
    (tmp_path / "references" / "10hz").mkdir(parents=True)
    (tmp_path / "references" / "10hz" / "remote_Q58N.npz").write_bytes(b"")
    assert recipe.missing_inputs(sv, "Q58N", 10, "ne", "remote") == ""


# ---------------------------------------------------------------- the assembly

def _write_edi(path, period, z, tipper=None, error=0.1):
    from mt_metadata.transfer_functions.core import TF
    tf = TF()
    tf.station_metadata.id = "TEST"
    tf.period = np.asarray(period, float)
    tf.impedance = np.asarray(z, complex)
    tf.impedance_error = np.full(np.shape(z), float(error))
    if tipper is not None:
        tf.tipper = np.asarray(tipper, complex)
        tf.tipper_error = np.full(np.shape(tipper), 0.01)
    tf.write(fn=str(path), file_type="edi")
    return path


def _rows(n, scale):
    z = np.zeros((n, 2, 2), complex)
    z[:, 0, 0] = scale * (1 + 1j)
    z[:, 0, 1] = scale * (10 + 5j)
    z[:, 1, 0] = scale * -(3 + 2j)
    z[:, 1, 1] = scale * (0.5 + 0.1j)
    return z


def test_the_rows_assemble_into_one_tensor_with_the_named_row_s_tipper(tmp_path):
    """Fails if the assembled file does not take its first row from the x pass and its second from the y
    pass, if the tipper is not the named row's, or if a period the y row does not reach carries a number."""
    pytest.importorskip("mt_metadata.transfer_functions.core")
    from auslamp_proc.transfer_functions import read_tf
    p_long = np.geomspace(1.0, 3000.0, 32)
    p_short = np.geomspace(1.0, 100.0, 24)          # the y row stops at 100 s
    tx = np.full((len(p_long), 1, 2), 0.1 + 0.0j)
    ty = np.full((len(p_short), 1, 2), 0.9 + 0.0j)
    x = _write_edi(tmp_path / "x.edi", p_long, _rows(len(p_long), 1.0), tx)
    y = _write_edi(tmp_path / "y.edi", p_short, _rows(len(p_short), 7.0), ty)

    got = recipe.assemble(x, y, tmp_path / "a.edi", tipper="x", verbose=False)
    a = read_tf(tmp_path / "a.edi")
    assert got["tipper_from"] == "x"
    assert np.isclose(np.abs(a.z[0, 0, 1]), np.abs(_rows(1, 1.0)[0, 0, 1]), rtol=1e-3)   # x row kept
    assert np.isclose(np.abs(a.z[0, 1, 0]), np.abs(_rows(1, 7.0)[0, 1, 0]), rtol=1e-3)   # y row written
    above = a.period > 110.0
    assert above.any()
    assert not np.isfinite(a.z[above, 1, 0]).any()                                        # empty above
    assert np.isfinite(a.z[above, 0, 1]).all()                                            # x row measured
    assert np.isclose(np.abs(a.t[0, 0, 0]), 0.1, rtol=1e-3)
    assert got["n_y_periods"] == int(np.isfinite(a.z[:, 1, 0]).sum())

    got = recipe.assemble(x, y, tmp_path / "b.edi", tipper="y", verbose=False)
    b = read_tf(tmp_path / "b.edi")
    assert got["tipper_from"] == "y"
    assert np.isclose(np.abs(b.t[0, 0, 0]), 0.9, rtol=1e-3)


def test_an_identical_grid_is_taken_and_not_interpolated(tmp_path):
    """Fails if two transfer functions on one grid are interpolated onto each other, which would move every value by
    the interpolation's own error where nothing had to move."""
    pytest.importorskip("mt_metadata.transfer_functions.core")
    from auslamp_proc.transfer_functions import read_tf
    p = np.geomspace(1.0, 1000.0, 20)
    x = _write_edi(tmp_path / "x.edi", p, _rows(len(p), 1.0))
    y = _write_edi(tmp_path / "y.edi", p, _rows(len(p), 3.0))
    got = recipe.assemble(x, y, tmp_path / "a.edi", verbose=False)
    assert "the two grids are one" in got["how"]
    assert got["n_y_periods"] == len(p)
    a = read_tf(tmp_path / "a.edi")
    assert np.allclose(np.abs(a.z[:, 1, 0]), np.abs(_rows(len(p), 3.0)[:, 1, 0]), rtol=1e-4)


def test_an_assembled_form_carries_its_own_candidacy_and_its_sentence():
    """Fails if a form assembled from other forms cannot be a candidate without a control of its own, or if
    the rule the caller states does not reach the table's verdict cell."""
    from auslamp_proc.site import deliver
    base = dict(site="X", kind="remote", rate_hz=1.0, params="k", status="made", transfer_function="none.edi",
                controls="", criterion="", seed=None)
    t = deliver.forms_table([dict(base, form="recipe",
                                  candidate_rule=dict(candidate=True,
                                                      verdict="a candidate: the y row beats its control"))])
    assert bool(t.candidate.iloc[0])
    assert t.verdict.iloc[0].startswith("a candidate")
    t = deliver.forms_table([dict(base, form="recipe",
                                  candidate_rule=dict(candidate=False,
                                                      verdict="NOT a candidate: the model does not hold"))])
    assert not bool(t.candidate.iloc[0])
    assert t.verdict.iloc[0].startswith("NOT a candidate")
    # without the rule the same row is read and never promoted, as every form with no control is
    t = deliver.forms_table([dict(base, form="recipe")])
    assert not bool(t.candidate.iloc[0])
    assert "no control" in t.verdict.iloc[0]


def test_the_recipe_lines_carry_the_frame_the_hours_the_rate_and_the_seed():
    """Fails if a header line is not key=value, which the EDI writer raises on, or if the window and the
    control's seed are not in the lines."""
    rows = dict(x=dict(hours="whole", rate=1, seed=20260916),
                y=dict(hours="window:coherent", rate=10, seed=20260917,
                       window=dict(t_start=1774180800, t_end=1774231200),
                       control_window=dict(t_start=1773000000, t_end=1773050400)))
    lines = recipe.recipe_lines(dict(frame="diagonal", tipper="x"), rows)
    assert all("=" in ln for ln in lines)
    joined = " ".join(lines)
    assert "diagonal" in joined and "window:coherent" in joined and "10 Hz" in joined
    assert "20260917" in joined and "2026-03-22T12:00:00" in joined
