"""Tests for the two signs the look tests decide: the DC floor rule and the phase quadrant rule.

Each test names the consequence of its failure. Nothing here reads a real record: the numbers are written in
the test so that a failure names the rule and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

from auslamp_proc import look


def test_the_dc_floor_reads_a_sign_only_outside_the_floor():
    """Fails if a magnetic sign is read off a ratio inside the 0.3 floor. A dead coil sits near zero and so
    does a sensor laid at right angles to north, and a sign taken there is a sign taken off noise."""
    got = look.signs_from_dc({"Bx_over_X": 0.95, "Bz_over_Z": -0.99})
    assert got["Hx"][0] == 1 and got["Hz"][0] == -1
    assert "0.3 floor" in got["Hx"][1] or "0.3" in got["Hx"][1]

    inside = look.signs_from_dc({"Bx_over_X": 0.12, "Bz_over_Z": -0.29})
    assert inside["Hx"][0] is None and inside["Hz"][0] is None
    assert "floor" in inside["Hx"][1]

    absent = look.signs_from_dc({"Bx_over_X": "", "Bz_over_Z": float("nan")})
    assert absent["Hx"][0] is None and absent["Hz"][0] is None


def test_a_flagged_dc_row_reads_no_sign_at_all():
    """Fails if a sign is read from a magnetometer the DC test already called into question.

    A sensor whose F is 16 per cent off IGRF, or one laid far from north, is not measuring the component the
    ratio is being read as, so a polarity taken off it is a polarity taken off the wrong number. `Bx
    negative` is the one flag the rule exists to answer and does not disqualify the row. Queensland Phase 1's
    Q65 is the site this reproduces: its sign_hx was withheld by hand, and its DC row reads
    `F/Figrf 1.16 (gain or a broken axis); Bx negative; angle -18`.
    """
    q65 = {"Bx_over_X": -1.1266, "Bz_over_Z": 1.1871,
           "flags": "F/Figrf 1.16 (gain or a broken axis); Bx negative; angle -18"}
    got = look.signs_from_dc(q65)
    assert got["Hx"][0] is None and got["Hz"][0] is None
    assert "F/Figrf" in got["Hx"][1]
    assert look.dc_flags(q65) == ["F/Figrf 1.16 (gain or a broken axis)", "angle -18"]

    # Bx negative alone is the case the rule answers, and an empty or NaN flags cell is a sound sensor
    only_bx = {"Bx_over_X": -0.98, "Bz_over_Z": 0.99, "flags": "Bx negative"}
    assert look.dc_flags(only_bx) == []
    assert look.signs_from_dc(only_bx)["Hx"][0] == -1
    assert look.dc_flags({"flags": float("nan")}) == []
    assert look.dc_flags({}) == []


def _phase_curve(value_deg, n=12, lo=20.0, hi=2000.0):
    period = np.logspace(np.log10(lo), np.log10(hi), n)
    return period, np.full(n, float(value_deg))


def test_the_quadrant_rule_reads_the_first_quadrant_as_plus_one():
    """Fails if a line whose phase sits in the first quadrant is not read as +1, or one in the third as -1.
    Reversing an electrode pair turns that row of the tensor by 180 deg, which is the whole of this rule."""
    p, ph = _phase_curve(45.0)
    assert look.quadrant_sign(p, ph, "Ex")[0] == 1
    p, ph = _phase_curve(-135.0)
    assert look.quadrant_sign(p, ph, "Ey")[0] == -1


def test_the_quadrant_rule_refuses_a_phase_in_neither_quadrant():
    """Fails if a sign is read where the phase is in neither quadrant: a phase at 135 deg is not a polarity
    question and writing a sign there would put a guess into decisions.csv."""
    p, ph = _phase_curve(135.0)
    value, why = look.quadrant_sign(p, ph, "Ex")
    assert value is None and "neither quadrant" in why


def test_the_quadrant_rule_needs_six_periods_in_the_band():
    """Fails if a sign is read from fewer than the six periods the rule scores, or from periods outside
    30-1000 s: the band is where a long-period record has its best signal-to-noise and the count is what
    keeps one good period from deciding a polarity."""
    p = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 40.0, 100.0])       # only two inside 30-1000 s
    value, why = look.quadrant_sign(p, np.full(len(p), 45.0), "Ex")
    assert value is None and "fewer than the 6" in why

    # four of six in the quadrant is the bar, three is not
    p = np.logspace(np.log10(35), np.log10(900), 6)
    ph = np.array([45.0, 45.0, 45.0, 45.0, 135.0, 135.0])
    assert look.quadrant_sign(p, ph, "Ex")[0] == 1
    ph = np.array([45.0, 45.0, 45.0, 135.0, 135.0, 135.0])
    assert look.quadrant_sign(p, ph, "Ex")[0] is None


def test_the_floor_and_the_band_are_the_values_the_workbooks_state():
    """Fails if the constants move away from the numbers SITES_COLUMNS.md and the workbooks print."""
    assert look.SIGN_FLOOR == 0.3
    assert look.QUADRANT_BAND_S == (30.0, 1000.0)
    assert (look.QUADRANT_PERIODS, look.QUADRANT_MIN_IN) == (6, 4)
