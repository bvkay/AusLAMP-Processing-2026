"""Tests for the survey's three tables: the cell grammar, what a re-run may overwrite, and the columns.

Each test names the consequence of its failure. Nothing here reads a real survey: the tables are written in
the test so that a failure names the code and not the data.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import pandas as pd
import yaml

from auslamp_proc import survey as SV

DECISION_COLUMNS_ADDED = ["e_exchange", "e_exchange_source", "e_shift_s", "e_shift_source",
                          "h_exchange", "h_gain", "h_gain_source",
                          "h_lender", "h_lender_channels", "h_lender_source"]


def _blank(site="A"):
    return SV.blank_decisions([site])


def test_the_six_hand_decisions_and_their_sources_are_columns(tmp_path):
    """Fails if a decision the processing applies is not a column of decisions.csv: a decision applied from
    anywhere but the table is a decision nothing records."""
    for c in DECISION_COLUMNS_ADDED:
        assert c in SV.DECISIONS_COLUMNS, c
    assert list(_blank().columns) == SV.DECISIONS_COLUMNS
    assert all(_blank().iloc[0][c] == SV.DECIDE for c in DECISION_COLUMNS_ADDED)


def test_write_table_keeps_a_stored_decide_where_the_caller_computed_nothing(tmp_path):
    """Fails if a workbook that computed nothing blanks an analyst's `decide` or `assume:` cell: that is the
    case the guard exists for."""
    path = tmp_path / "decisions.csv"
    first = _blank()
    first.at[0, "h_gain"] = "assume:1.17"
    SV.write_table(path, first, SV.DECISIONS_COLUMNS)
    incoming = _blank()
    incoming.at[0, "h_gain"] = ""                     # the workbook computed nothing
    incoming.at[0, "e_exchange"] = ""
    line = SV.write_table(path, incoming, SV.DECISIONS_COLUMNS)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.at[0, "h_gain"] == "assume:1.17"
    assert back.at[0, "e_exchange"] == SV.DECIDE
    assert "2 assume:/decide cells kept" in line


def test_write_table_lets_a_caller_with_a_value_fill_a_decide_cell(tmp_path):
    """Fails if a caller holding a measured value cannot write it over a stored `decide` or `assume:`: the
    analyst's own write is exactly that, and a guard that refuses it cannot FILL a cell (D1 stage 1)."""
    path = tmp_path / "decisions.csv"
    first = _blank()
    first.at[0, "h_gain"] = "assume:1.00"
    SV.write_table(path, first, SV.DECISIONS_COLUMNS)
    incoming = _blank()
    incoming.at[0, "h_gain"] = "1.17"
    incoming.at[0, "e_exchange"] = "yes"
    line = SV.write_table(path, incoming, SV.DECISIONS_COLUMNS)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.at[0, "h_gain"] == "1.17"
    assert back.at[0, "e_exchange"] == "yes"
    assert "0 assume:/decide cells kept" in line


def test_none_is_a_decision_and_not_an_empty_cell(tmp_path):
    """Fails if `none` is read as an empty cell: h_lender `none` says the site's own H stands, which is a
    decision, and a guard that read it as silence would revert it to `decide` for ever."""
    path = tmp_path / "decisions.csv"
    SV.write_table(path, _blank(), SV.DECISIONS_COLUMNS)
    incoming = _blank()
    incoming.at[0, "h_lender"] = "none"
    SV.write_table(path, incoming, SV.DECISIONS_COLUMNS)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.at[0, "h_lender"] == "none"
    assert SV.is_empty("none") and not SV.is_blank("none")


def test_the_new_columns_round_trip_through_load_survey(tmp_path, monkeypatch):
    """Fails if a survey written with the new columns does not come back with them, or if a numeric cell
    comes back as a float: `decide` and 1.17 share a column and a dtype guess turns one of them into NaN."""
    folder = tmp_path / "surveys" / "tst"
    folder.mkdir(parents=True)
    (folder / "survey.yaml").write_text(yaml.safe_dump(dict(name="tst", raw_root=str(tmp_path),
                                                            work_root=str(tmp_path), years=[2025])),
                                        encoding="utf-8")
    dec = _blank("Q65")
    dec.at[0, "h_gain"] = "1.17"
    dec.at[0, "h_lender"] = "none"
    dec.at[0, "e_shift_s"] = "+0.95"
    dec.at[0, "e_exchange"] = "yes"
    dec.at[0, "h_lender_channels"] = "Hx Hy Hz"
    SV.write_table(folder / "decisions.csv", dec, SV.DECISIONS_COLUMNS)
    sites = pd.DataFrame([{c: ("Q65" if c == "site" else "") for c in SV.SITES_COLUMNS}])
    SV.write_table(folder / "sites.csv", sites, SV.SITES_COLUMNS)
    monkeypatch.setattr(SV, "SURVEYS", tmp_path / "surveys")
    sv = SV.load_survey("tst")
    row = sv.decision("Q65")
    assert list(sv.decisions.columns) == SV.DECISIONS_COLUMNS
    assert row["h_gain"] == "1.17" and row["e_shift_s"] == "+0.95"
    assert row["e_exchange"] == "yes" and row["h_lender"] == "none"
    assert row["h_lender_channels"] == "Hx Hy Hz"
    # and the package reads them back as the decisions they are
    from auslamp_proc.process import frame as FR
    assert FR.read_number(row["h_gain"], 1.0) == (1.17, True)
    assert FR.read_number(row["e_shift_s"], 0.0) == (0.95, True)
    assert FR.read_yes_no(row["e_exchange"]) == (True, True)
    assert FR.read_lender(row) == ""
    assert FR.read_lender_channels(row) == ["Hx", "Hy", "Hz"]


def test_validate_reports_an_open_decision(tmp_path):
    """Fails if a `decide` in one of the new columns is not reported as an open input: an open decision that
    nothing reports is one nobody closes."""
    sv = SV.Survey("tst", {}, pd.DataFrame(columns=SV.SITES_COLUMNS), _blank("Q65"))
    lines = [l for l in SV.validate(sv) if l.startswith("Q65 decisions")]
    assert lines and "h_gain" in lines[0] and "h_lender" in lines[0] and "e_exchange" in lines[0]
