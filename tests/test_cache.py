"""Tests for the cache placement, the NaN-honest decimation and the stage-2 look.

Each test names the consequence of its failure. The synthetic records are written as ASCII PR6-24 hourly
files, which is the form all three AusLAMP Queensland phases are in.

@author: ben kay (ben@auscope.org.au)
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from auslamp_proc import look
from auslamp_proc.geo import igrf
from auslamp_proc.raw import cache

INSTRUMENT = "PR6-24+Mag-03"
CONSTANTS = {"uv_per_count": 1.0, "h_uv_per_nt": 142.857, "bz_divider": 0.4, "e_gain": 10,
             "source": "mt_io.uoa.pr624 constants"}
FS = 1.0
HOUR = 3600


def write_site(root, site, hours, fs=FS, duplicate=()):
    """A PR6-24 ASCII tree: <site>/001/<site>_250101hh0000.<BX BY BZ EX EY>, one hour per stamp.

    `hours` are the hours of 2025-01-01 to write. `duplicate` names the hours written a second time under a
    second stem, which is the recorder writing one hour twice.
    """
    day = root / site / "001"
    day.mkdir(parents=True, exist_ok=True)
    n = int(HOUR * fs)
    for h in hours:
        for ch, ext in cache.EDL_EXT.items():
            v = np.arange(n, dtype=float) + 1000 * h + 7 * list(cache.EDL_EXT).index(ch)
            (day / ("%s_250101%02d0000.%s" % (site, h, ext))).write_text(
                "\n".join("%d" % x for x in v) + "\n", encoding="ascii")
            if h in duplicate:
                (day / ("%s250101%02d0000.%s" % (site, h, ext))).write_text(
                    "\n".join("%d" % x for x in v) + "\n", encoding="ascii")
    return root / site


def site_row(site, raw_path):
    return {"site": site, "raw_path": str(raw_path), "instrument": INSTRUMENT,
            "layout": "edl_dayofyear", "sample_rate_hz": FS,
            "dipole_n_m": "10.0", "dipole_e_m": "12.0", "dipole_source": "test"}


def cfg(work_root):
    return {"work_root": str(work_root), "years": [2025], "instruments": {INSTRUMENT: CONSTANTS}}


# ---------------------------------------------------------------- placement
# Failure means an hour is concatenated rather than placed: a missing hour would shift every hour after it,
# and every period read from the record after the gap would be wrong.
def test_missing_hour_lands_at_its_own_start(tmp_path):
    raw = write_site(tmp_path / "raw", "Q99", hours=(0, 1, 3))
    prov = cache.build(site_row("Q99", raw), cfg(tmp_path / "work"), rates=(1,), force=True)

    assert prov["n_raw"] == 4 * HOUR                       # the axis runs to the end of the last hour
    t0, arrays, _ = cache.load("Q99", tmp_path / "work", 1)
    assert t0 == int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    hx = arrays["Hx"]
    assert np.isfinite(hx[0:2 * HOUR]).all()               # hours 0 and 1
    assert np.isnan(hx[2 * HOUR:3 * HOUR]).all()           # the missing hour stays NaN
    assert np.isfinite(hx[3 * HOUR:4 * HOUR]).all()        # the fourth hour at its own start, not shifted
    # the fourth hour's first sample is its own first value, which is what a concatenation would lose
    assert hx[3 * HOUR] == pytest.approx((0 + 1000 * 3) / CONSTANTS["h_uv_per_nt"])

    for ch in cache.CHANNELS:
        s = prov["channels"][ch]
        assert s["files_found"] == 3 and s["files_read"] == 3
        assert s["samples_placed"] == s["samples_read"] == 3 * HOUR
        assert s["overlap_samples"] == 0 and s["files_unplaced"] == 0
    assert prov["verdict_lines"][-1].startswith("reconciliation: PASS")


# Failure means an hour written twice is silently overwritten and the record is reported as complete.
def test_an_overlapping_pair_is_counted_and_fails_the_reconciliation(tmp_path):
    raw = write_site(tmp_path / "raw", "Q98", hours=(0, 1, 2), duplicate=(1,))
    prov = cache.build(site_row("Q98", raw), cfg(tmp_path / "work"), rates=(1,), force=True)

    for ch in cache.CHANNELS:
        s = prov["channels"][ch]
        assert s["files_found"] == 4                       # three hours, one of them written twice
        assert s["overlap_samples"] == HOUR
        assert s["samples_placed"] == 4 * HOUR and s["samples_read"] == 4 * HOUR
    assert prov["verdict_lines"][-1].startswith("reconciliation: FAIL")
    assert all(line.startswith(ch + ": FAIL") for ch, line in zip(cache.CHANNELS, prov["verdict_lines"]))


# Failure means the reconciliation cannot fail at all, which is the vacuous form of vic_cache.py:169.
def test_the_reconciliation_verdict_can_fail():
    good = {ch: dict(files_found=2, files_read=2, samples_read=10, samples_placed=10,
                     overlap_samples=0, files_unplaced=0) for ch in cache.CHANNELS}
    assert cache._verdicts(good)[-1].startswith("reconciliation: PASS")
    for key, value in (("overlap_samples", 3), ("samples_placed", 9), ("files_read", 1),
                       ("files_unplaced", 1)):
        bad = {ch: dict(good[ch]) for ch in cache.CHANNELS}
        bad["Hx"][key] = value
        assert cache._verdicts(bad)[-1].startswith("reconciliation: FAIL"), key


# ---------------------------------------------------------------- decimation
# Failure means a gap is filled with interpolated values that the rest of the series cannot tell from data.
def test_one_nan_blanks_the_output_it_touches_and_nothing_else():
    x = np.sin(np.arange(1000) / 50.0)
    x[25] = np.nan
    y = cache.decimate_nan(x, 10)
    assert len(y) == 100
    assert np.isnan(y[2])
    assert np.isfinite(np.delete(y, 2)).all()


def test_a_whole_gap_blanks_exactly_its_own_blocks():
    x = np.sin(np.arange(1000) / 50.0)
    x[200:260] = np.nan                                     # blocks 20 to 25 inclusive
    y = cache.decimate_nan(x, 10)
    assert np.isnan(y[20:26]).all()
    assert np.isfinite(np.delete(y, np.arange(20, 26))).all()


def test_an_all_nan_channel_decimates_to_all_nan():
    y = cache.decimate_nan(np.full(1000, np.nan), 10)
    assert len(y) == 100 and np.isnan(y).all()


# ---------------------------------------------------------------- the DC test
# Failure means a reversed or mis-gained magnetometer is carried into processing unflagged.
def test_the_dc_test_passes_a_sensor_sitting_on_igrf():
    lat, lon, elev, when = -25.0, 148.0, 400.0, datetime(2025, 10, 15)
    ig = igrf(lat, lon, elev, when)
    arrays = {"Hx": np.full(100, ig["X"]), "Hy": np.full(100, ig["Y"]), "Hz": np.full(100, ig["Z"])}
    row = look.dc_test(arrays, lat, lon, elev, when)
    assert row["flags"] == "" and row["verdict"] == "PASS"
    assert row["F_over_Figrf"] == pytest.approx(1.0, abs=1e-3)


def test_the_dc_test_flags_a_reversed_bx():
    lat, lon, elev, when = -25.0, 148.0, 400.0, datetime(2025, 10, 15)
    ig = igrf(lat, lon, elev, when)
    arrays = {"Hx": np.full(100, -ig["X"]), "Hy": np.full(100, ig["Y"]), "Hz": np.full(100, ig["Z"])}
    row = look.dc_test(arrays, lat, lon, elev, when)
    assert "Bx negative" in row["flags"] and row["verdict"] == "CHECK"
    assert row["F_over_Figrf"] == pytest.approx(1.0, abs=1e-3)   # a reversal leaves the total field alone


def test_the_dc_test_flags_a_gain_before_a_tilt():
    lat, lon, elev, when = -25.0, 148.0, 400.0, datetime(2025, 10, 15)
    ig = igrf(lat, lon, elev, when)
    arrays = {c: np.full(100, 0.9 * ig[k]) for c, k in (("Hx", "X"), ("Hy", "Y"), ("Hz", "Z"))}
    row = look.dc_test(arrays, lat, lon, elev, when)
    assert "gain or a broken axis" in row["flags"]


# ---------------------------------------------------------------- the E-line state machine
def _two_days(seed=3):
    rng = np.random.default_rng(seed)
    n = 2 * 86400
    hx = np.cumsum(rng.normal(0, 1.0, n)) * 0.02
    hy = np.cumsum(rng.normal(0, 1.0, n)) * 0.02
    return n, hx, hy, rng


# Failure means a dead or common-mode electric line is scored as usable and processed.
def test_the_state_machine_reads_sound_dead_and_common():
    n, hx, hy, rng = _two_days()
    t0 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())

    sound = look.elines(t0, {"Hx": hx, "Hy": hy,
                             "Ex": 5.0 * hy + rng.normal(0, 0.05, n),
                             "Ey": 5.0 * hx + rng.normal(0, 0.05, n)})
    assert list(sound.Ex_state) == ["sound"] * len(sound)
    assert list(sound.Ey_state) == ["sound"] * len(sound)
    assert len(sound) == 2

    dead = look.elines(t0, {"Hx": hx, "Hy": hy,
                            "Ex": rng.normal(0, 0.01, n),
                            "Ey": 5.0 * hx + rng.normal(0, 0.05, n)})
    assert list(dead.Ex_state) == ["dead"] * len(dead)
    assert list(dead.Ey_state) == ["sound"] * len(dead)

    shared = np.cumsum(rng.normal(0, 1.0, n)) * 0.05        # one noise on both lines, coherent with no H
    common = look.elines(t0, {"Hx": hx, "Hy": hy,
                              "Ex": shared + rng.normal(0, 0.02, n),
                              "Ey": shared + rng.normal(0, 0.02, n)})
    assert list(common.Ex_state) == ["common"] * len(common)
    assert list(common.Ey_state) == ["common"] * len(common)

    s = look.eline_summary(sound, "SOUND")
    assert s["days_scored"] == 2 and s["Ex_sound_frac"] == 1.0 and s["both_sound_days"] == 2


# Failure means a day whose channels are mostly gap is scored, and UNJUDGED is reported as a state.
def test_a_day_under_eighty_per_cent_finite_is_a_gap_day():
    n, hx, hy, rng = _two_days(seed=5)
    ex = 5.0 * hy + rng.normal(0, 0.05, n)
    ey = 5.0 * hx + rng.normal(0, 0.05, n)
    ex[:60000] = np.nan                                     # 69 per cent of the first day
    t = look.elines(int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()),
                    {"Hx": hx, "Hy": hy, "Ex": ex, "Ey": ey})
    assert t.iloc[0].Ex_state == "gap" and t.iloc[0].Ey_state == "gap"
    assert t.iloc[1].Ex_state == "sound"
    assert look.eline_summary(t)["days_scored"] == 1


# Failure means one spike puts a whole day's coherence at zero and a sound line is reported as weak.
def test_the_despike_blanks_a_spike_and_its_neighbours():
    x = np.sin(np.arange(2000) / 30.0)
    x[1000] += 500.0
    y, n_blanked = look.despike(x)
    assert n_blanked >= 6
    assert np.isnan(y[1000])
    assert np.isfinite(y[:990]).all() and np.isfinite(y[1010:]).all()
