"""Tests for the clean pool on a survey too small to have a fleet, and for the runner's parameter parsing.

Each test names the consequence of its failure. The scans are written in the test, so a failure names the
rule and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from auslamp_proc.process import transients as TR

ROOT = Path(__file__).resolve().parent.parent
CFG = {"pool": {"min_fleet": 3}}


def _scan(work, site, n=400, level=1.0, spikes=()):
    """Write one site's tail scan: a flat band-power series with `spikes` chunks raised 1000x."""
    pw = {c: np.full(n, level) for c in TR.SCAN_CHANNELS}
    for c in ("Hx", "Hy"):
        for i in spikes:
            pw[c][i] = level * 1000.0
    TR.tails_dir(work).mkdir(parents=True, exist_ok=True)
    np.savez(TR.tails_dir(work) / ("%s.npz" % site), t0=np.array([0.0]),
             chunk_s=np.array([600.0]), **pw)


def test_a_pool_of_three_turns_the_event_test_off(tmp_path):
    """Fails if the fleet-normalised event test runs on a survey that has no fleet.

    With two others the test cannot ask whether the array saw what this site saw, and falling back on the
    site's own threshold flags every large chunk of every site: the pool empties, no remote is offered and
    the run ends in a traceback rather than a reading. A run scoped to three sites is the ordinary way a
    student tries the series out, so this is the case the gate exists for.
    """
    sites = ["A", "B", "C"]
    for s in sites:
        _scan(tmp_path, s, spikes=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
    on, others = TR.fleet_normalised("A", sites, tmp_path, CFG)
    assert not on and len(others) == 2

    r = TR.clean_row("A", sites, tmp_path, CFG)
    assert r["event_frac"] is None, "the event limb was scored without a fleet to normalise against"
    assert r["fleet_normalised"] is False
    assert r["judged"], "a site with a scan was reported unjudged"
    assert r["clean"], "a three-site pool was emptied by the event limb"
    assert "min_fleet" in r["note"] and "off" in r["note"]
    assert TR.site_events("A", sites, tmp_path, CFG) == [], "the mask flagged events with no fleet"


def test_a_pool_of_four_scores_the_event_test(tmp_path):
    """Fails if the gate turns the test off where a fleet does exist: three others is the threshold, and a
    site that spikes alone against three quiet neighbours is exactly what the test is for."""
    sites = ["A", "B", "C", "D"]
    _scan(tmp_path, "A", spikes=tuple(range(0, 200, 2)))     # loud, and alone in it
    for s in ("B", "C", "D"):
        _scan(tmp_path, s)
    on, others = TR.fleet_normalised("A", sites, tmp_path, CFG)
    assert on and len(others) == 3
    r = TR.clean_row("A", sites, tmp_path, CFG)
    assert r["event_frac"] is not None and r["event_frac"] > 0.02
    assert not r["clean"] and "event fraction" in r["reason"]
    assert r["fleet_normalised"] and r["note"] == ""


def test_a_fleet_weight_table_written_under_another_pool_is_not_reused(tmp_path):
    """Fails if a cached fleet_weights.json is used where its members are not this store's pool.

    A weight is a median over a member's pairs with the pool it was computed under, so a table written while
    three sites were cached gives each member two pairs, falls under fleet_weight_table's min_chunks and
    comes back None. Reusing it on the full survey refuses every member of every stack for `no fleet
    coherence at 100-1000 s` at sites with a dozen sound pairs, and nothing says why. W_fresh's Phase 3 run
    read exactly that and recorded the fleet stack as unavailable on the survey.
    """
    import json
    import types

    from auslamp_proc.process import references as REF

    st = REF.Store.__new__(REF.Store)
    st.dir = tmp_path / "references" / "1hz"
    st.dir.mkdir(parents=True)
    st._fleet = None
    (st.dir / "fleet_weights.json").write_text(json.dumps({"A": None, "B": None, "C": None}),
                                               encoding="utf-8")
    (st.dir / "fleet_pairs.json").write_text(json.dumps({"A:B": {"coh": None, "chunks": 1}}),
                                             encoding="utf-8")

    # the store's own pool is seven sites; the file on disk knows three
    st.clean_pool = types.MethodType(lambda self: (list("ABCDEFG"), None), st)
    recomputed = {}

    def _scored(self, *a, **k):
        recomputed["ran"] = True
        return {"coh": 0.8, "chunks": 40}

    st.pair_coherence = types.MethodType(_scored, st)
    weights, _pairs = st.fleet_weights()
    assert recomputed.get("ran"), "the stale three-site table was reused over a seven-site pool"
    assert set(weights) == set("ABCDEFG")
    assert all(v is not None for v in weights.values())

    # and a table whose members ARE the pool is reused, because re-scoring it costs minutes
    st._fleet = None
    recomputed.clear()
    st.fleet_weights()
    assert not recomputed.get("ran"), "a table written under this very pool was thrown away"


def test_a_bare_word_set_value_arrives_as_a_string():
    """Fails if --set SITES=largest is written into the copied workbook as the name `largest`.

    PowerShell strips the inner quotes of --set SITES='"largest"', so the documented POSIX quoting arrives
    as a bare word on the shell this repository's own machine defaults to. Writing it straight in is a
    NameError at execution rather than a refusal at the command line.
    """
    sys.path.insert(0, str(ROOT / "generator"))
    import run_workbooks as RW

    assert RW.parameter_value("largest") == repr("largest")
    assert RW.parameter_value('"largest"') == repr("largest")
    assert RW.parameter_value("13") == "13"
    assert RW.parameter_value('["Q32","Q33"]') == repr(["Q32", "Q33"])
    assert RW.parameter_value("True") == "True"
    # a Windows path survives: the replacement is a function, so a backslash is not a regex escape
    assert RW.parameter_value(r"C:\work") == repr("C:\\work")


def test_the_runner_counts_the_verdicts_it_reads():
    """Fails if the runner's summary cannot tell a PASS from a FAIL. A green execution line over a workbook
    carrying FAIL verdicts is what made the workbooks' own verdict machinery invisible."""
    sys.path.insert(0, str(ROOT / "generator"))
    import run_workbooks as RW

    lines = ["VERDICT: PASS -- one", "VERDICT: FAIL -- two", "VERDICT: UNJUDGED -- three",
             "VERDICT: PASS -- four"]
    n_pass, n_fail, n_unj, bad = RW.verdict_counts(lines)
    assert (n_pass, n_fail, n_unj) == (2, 1, 1)
    assert bad == ["VERDICT: FAIL -- two", "VERDICT: UNJUDGED -- three"]
    assert RW.verdict_counts([]) == (0, 0, 0, [])
