"""Tests for the raw readers and the deployment register.

Each test names the consequence of its failure. The two fixtures are real files copied from the AusLAMP
Victoria release: a 130-byte EDL .gps day file and a 269-byte LEMI .INF power-up block.

@author: ben kay (ben@auscope.org.au)
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from auslamp_proc.raw import discover, edl, lemi
from auslamp_proc import register

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VIC_YEARS = [2013, 2014, 2016, 2017, 2018]
QLD_YEARS = [2025, 2026]


# ---------------------------------------------------------------- file names
# Failure means the site id was taken from the file stem; 30 of the 81 AusLAMP Victoria EDL sites would then
# be filed under a name that is not their folder's.
@pytest.mark.parametrize("folder,name,stem,stamp,ext", [
    ("VIC001", "VIC001_140411060000.BX", "VIC001_", "140411060000", "BX"),   # the common form
    ("VIC003", "VIC003140512000000.gps", "VIC003", "140512000000", "gps"),   # no underscore
    ("VIC073", "VIC73140101000000.BX", "VIC73", "140101000000", "BX"),       # two-digit stem
    ("VIC055", "VIC055_140409030000.by", "VIC055_", "140409030000", "by"),   # lower-case extension
    ("Q44", "Q68N_061023025217.BX", "Q68N_", "061023025217", "BX"),          # the previous site's name
    ("Q64N", "q64n_061011000000.BX", "q64n_", "061011000000", "BX"),         # mixed case
])
def test_split_stem(folder, name, stem, stamp, ext):
    got_stem, got_stamp, got_ext = edl.split_stem(name)
    assert (got_stem, got_stamp, got_ext) == (stem, stamp, ext)


def test_the_site_is_the_folder_not_the_stem(tmp_path):
    """Every file written under a previous deployment's name; the site is still the folder name."""
    site = tmp_path / "VIC073"
    (site / "001").mkdir(parents=True)
    for hh in ("00", "01"):
        for e in ("BX", "BY", "BZ", "EX", "EY"):
            (site / "001" / ("VIC73140101%s0000.%s" % (hh, e))).write_bytes(b"")
    (site / "001" / "VIC73140101000000.gps").write_bytes(b"")
    (site / "config").mkdir()
    row = discover._discover_edl_site("VIC073", site, VIC_YEARS)
    assert row["site"] == "VIC073"
    assert row["stem_styles"] == "VIC73"
    assert row["n_data_files"] == 10
    assert row["first_stamp"] == "140101000000"
    assert row["sidecars"] == "gps:1"
    assert "config" in row["notes"]


def test_a_foreign_stem_far_outside_the_span_is_dropped_and_named(tmp_path):
    """One folder, two deployments: the site's own 2014 hours and a 2013 bench hour under another name.

    Failure moves the site's start by a year and puts it in the wrong group in the deployment register.
    """
    site = tmp_path / "VIC062"
    (site / "170").mkdir(parents=True)
    for e in ("BX", "BY", "BZ", "EX", "EY"):
        (site / "170" / ("VIC062_140619000000.%s" % e)).write_bytes(b"")
        (site / "170" / ("CALP28_130619053859.%s" % e)).write_bytes(b"")
        (site / "170" / ("VIC66140619120000.%s" % e)).write_bytes(b"")   # renamed logger, same day: kept
    row = discover._discover_edl_site("VIC062", site, VIC_YEARS)
    assert row["first_stamp"] == "140619000000"
    assert "CALP28_" not in row["stem_styles"]
    assert "VIC66" in row["stem_styles"]
    assert "CALP28_" in row["notes"]
    assert "foreign stem:5" in row["dropped"]


def test_split_stem_without_a_stamp():
    assert edl.split_stem("miniseed_move.bat")[:2] == (None, None)
    assert edl.split_stem("recorder")[:2] == (None, None)


# ---------------------------------------------------------------- the rollover
# Failure dates a Queensland 2025 record to 2006, outside every survey year, or pushes a Victoria 2014 record
# to 2033.
def test_rollover_applies_when_the_year_precedes_the_survey():
    dt, corr = edl.parse_stamp("060212000000", QLD_YEARS)
    assert dt == datetime(2025, 9, 28, tzinfo=timezone.utc)
    assert corr == "+1024 weeks"


def test_rollover_does_not_apply_inside_the_survey_years():
    dt, corr = edl.parse_stamp("140411060000", VIC_YEARS)
    assert dt == datetime(2014, 4, 11, 6, 0, tzinfo=timezone.utc)
    assert corr == ""


def test_rollover_is_exactly_1024_weeks():
    assert edl.ROLLOVER.days == 7168


# ---------------------------------------------------------------- the .gps decode
# The >RPV field widths are inferred from the files, not specified. Failure means the decode has drifted and
# every EDL position in the survey is wrong.
def test_gps_decode_of_one_day_file():
    fixes = edl.read_gps_file(FIXTURES / "VIC001_140411000000.gps")
    assert len(fixes) == 1
    f = fixes[0]
    assert f["latitude"] == pytest.approx(-38.91852, abs=1e-5)
    assert f["longitude"] == pytest.approx(146.44153, abs=1e-5)
    assert f["elevation"] == pytest.approx(94.0, abs=0.5)
    assert f["n_sat"] == 12


def test_gps_position_over_a_folder_of_one_file(tmp_path):
    (tmp_path / "101").mkdir()
    (tmp_path / "101" / "VIC001_140411000000.gps").write_bytes(
        (FIXTURES / "VIC001_140411000000.gps").read_bytes())
    p = edl.gps_position(tmp_path)
    assert p["lat"] == pytest.approx(-38.91852, abs=1e-5)
    assert p["n_fixes"] == 1
    assert p["min_sat"] == 12
    assert p["n_below_floor"] == 0
    assert p["scatter_m"] == 0.0


def test_gps_position_drops_fixes_below_the_satellite_floor(tmp_path):
    """A two-satellite fix 1 deg away does not move the site and is counted in n_below_floor."""
    (tmp_path / "101").mkdir()
    good = (FIXTURES / "VIC001_140411000000.gps").read_text()
    (tmp_path / "101" / "a_140411000000.gps").write_text(good)
    (tmp_path / "101" / "b_140412000000.gps").write_text(
        ">RAL20714+00094+00012;*4E<\n>RPV20714-3791852+1464415300000002;*76<\n")
    p = edl.gps_position(tmp_path)
    assert p["n_fixes"] == 1
    assert p["n_below_floor"] == 1
    assert p["min_sat"] == 12
    assert p["lat"] == pytest.approx(-38.91852, abs=1e-5)


# ---------------------------------------------------------------- DDMM and the .INF
# Failure puts every LEMI position out by the difference between degrees-minutes and decimal degrees, about
# 60 km at this latitude, and takes the dipole lengths from the wrong power-up block.
@pytest.mark.parametrize("value,hemi,expect", [
    ("3756.1009", "S", -37.935015),
    ("14327.7354", "E", 143.462257),
    ("0000.0000", "N", 0.0),
    ("3756.1009", "N", 37.935015),
])
def test_ddmm(value, hemi, expect):
    assert lemi.ddmm(value, hemi) == pytest.approx(expect, abs=1e-6)


def test_parse_inf_block():
    inf = lemi.parse_inf(FIXTURES / "201712130554.INF")
    assert inf["serial"] == "0070"
    assert inf["firmware"] == "1.1"
    assert inf["L1"] == pytest.approx(100.0)
    assert inf["L2"] == pytest.approx(100.0)
    assert inf["inf_lat"] == pytest.approx(-37.93502, abs=1e-5)
    assert inf["inf_lon"] == pytest.approx(143.46226, abs=1e-5)
    assert inf["inf_alt"] == pytest.approx(190.7)
    assert inf["inf_missing"] is False


def test_site_inf_takes_the_last_block_at_or_before_the_data(tmp_path):
    src = (FIXTURES / "201712130554.INF").read_text()
    (tmp_path / "201712130554.INF").write_text(src)                       # the block in force
    (tmp_path / "201712130001.INF").write_text(src.replace("1.000000e+02", "1.000000e+00"))  # setup, earlier
    (tmp_path / "203305311601.INF").write_text(src.replace("1.000000e+02", "5.000000e+02"))  # lost clock
    gov, allb = lemi.site_inf(tmp_path, "201712130555")
    assert len(allb) == 3
    assert gov["inf_key"] == "201712130554"
    assert gov["L1"] == pytest.approx(100.0)


def test_site_inf_with_no_block_at_all(tmp_path):
    gov, allb = lemi.site_inf(tmp_path, "201712130555")
    assert gov["inf_missing"] is True
    assert allb == []


def test_position_from_rows():
    row = ("2017 12 14 00 00 00 21368.969 -106.883 -55502.355 24.61 24.50 8.628 -498.482 18.486 -6.219 "
           "13.27 187.2 3756.1008 S 14327.7378 E 16 2 0").split()
    p = lemi.position_from_rows([row])
    assert p["lat"] == pytest.approx(-37.935013, abs=1e-5)
    assert p["lon"] == pytest.approx(143.462297, abs=1e-5)
    assert p["elev_m"] == pytest.approx(187.2)
    assert p["n_sat"] == 16
    assert lemi.row_time(row) == datetime(2017, 12, 14, tzinfo=timezone.utc)


# ---------------------------------------------------------------- the register
# Three sites: A and B overlap by 20 days, C starts after B ends. Failure makes the candidate pool for every
# reference choice wrong: a site that never overlapped the target is offered, or one that did is hidden.
def _three_sites():
    return pd.DataFrame([
        dict(site="A", start_utc="2014-01-01", end_utc="2014-02-01", instrument="EDL"),
        dict(site="B", start_utc="2014-01-12", end_utc="2014-02-20", instrument="EDL"),
        dict(site="C", start_utc="2014-03-01", end_utc="2014-04-01", instrument="EDL"),
    ])


def test_spans_and_overlap():
    sp = register.spans(_three_sites())
    assert list(sp.site) == ["A", "B", "C"]
    ov = register.overlap_matrix(sp)
    assert ov.loc["A", "B"] == pytest.approx(20.0, abs=0.01)
    assert ov.loc["A", "C"] == 0.0


def test_groups_at_two_floors():
    sp = register.spans(_three_sites())
    g7 = register.groups(sp, 7)
    assert list(g7.n) == [2, 1]
    assert g7.iloc[0].members == "A B"
    assert g7.iloc[0].common_start.startswith("2014-01-12")
    assert g7.iloc[0].common_end.startswith("2014-02-01")
    g30 = register.groups(sp, 30)          # nothing overlaps by 30 days: three singletons
    assert list(g30.n) == [1, 1, 1]


def test_core_group_is_the_sites_covering_the_fortnight():
    sp = register.spans(_three_sites())
    d = register.core_group("2014-01-20", sp, half_days=7)
    assert d["n_core"] == 2 and d["core"] == "A B"
    assert d["common_start"].startswith("2014-01-12")
    d2 = register.core_group("2014-02-10", sp, half_days=7)
    assert d2["n_core"] == 1 and d2["core"] == "B"


def test_active_per_day():
    a = register.active_per_day(register.spans(_three_sites()))
    assert int(a.loc["2014-01-20"]) == 2
    assert int(a.loc["2014-02-10"]) == 1
    assert int(a.loc["2014-02-25"]) == 0
