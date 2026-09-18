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


def test_a_header_with_no_row_is_created_and_not_preserved(tmp_path):
    """Fails if a table holding a header and no row is treated as a table to preserve.

    surveys/_template ships sites.csv and decisions.csv as header-only stubs so that the column lists are
    visible in the template, so a copied template makes both files exist holding nothing. A stub that was
    preserved would leave a new survey with empty tables for ever, which is what workbook 01's "created
    once if it does not exist" branch never firing looks like from the outside.
    """
    path = tmp_path / "decisions.csv"
    path.write_text(",".join(SV.DECISIONS_COLUMNS) + "\n", encoding="utf-8")
    assert SV.is_stub(path, SV.DECISIONS_COLUMNS)
    assert SV.is_stub(tmp_path / "absent.csv", SV.DECISIONS_COLUMNS)
    line = SV.write_table(path, SV.blank_decisions(["A", "B"]), SV.DECISIONS_COLUMNS)
    assert line.startswith("created") and "header and no row" in line
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(back.site) == ["A", "B"] and back.at[0, "sign_hx"] == SV.DECIDE
    assert not SV.is_stub(path, SV.DECISIONS_COLUMNS)


def test_write_signs_writes_only_into_a_decide_cell(tmp_path):
    """Fails if a measured sign overwrites an analyst's value or an `assume:` cell.

    A workbook's test is a reading beside the analyst's cell and never a ruling over it, so write_signs
    writes only where the cell reads `decide`. This is the guard the two sign-writing cells rest on.
    """
    path = tmp_path / "decisions.csv"
    dec = SV.blank_decisions(["A", "B"])
    dec.at[0, "sign_hx"] = "-1"                    # the analyst's own value
    dec.at[0, "sign_hz"] = "assume:+1"             # an assumption somebody stated
    dec.at[1, "h_gain"] = "1.17"                   # a plain value in a column nothing here touches
    dec.at[1, "sign_source"] = "an earlier ruling"
    SV.write_table(path, dec, SV.DECISIONS_COLUMNS)

    measured = {"A": {"sign_hx": (1, "the DC test"), "sign_hz": (1, "the DC test"),
                      "sign_ex": (-1, "the quadrant rule")},
                "B": {"sign_hx": (-1, "the DC test"), "sign_hy": (None, "inside the floor")}}
    line, written = SV.write_signs(tmp_path, measured, dated="2026-09-18")
    back = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("site")

    assert back.at["A", "sign_hx"] == "-1", "an analyst's value was overwritten"
    assert back.at["A", "sign_hz"] == "assume:+1", "an assume: cell was overwritten"
    assert back.at["A", "sign_ex"] == "-1", "a decide cell was not written"
    assert back.at["B", "sign_hx"] == "-1"
    assert back.at["B", "sign_hy"] == SV.DECIDE, "a sign the test could not judge was written anyway"
    assert back.at["B", "h_gain"] == "1.17", "a value in an untouched column was lost"
    assert sorted(written) == ["A sign_ex -1", "B sign_hx -1"]
    assert "the quadrant rule" in back.at["A", "sign_source"] and "2026-09-18" in back.at["A", "sign_source"]
    assert back.at["B", "sign_source"].startswith("an earlier ruling |"), "an existing source was replaced"
    assert line.startswith("rewrote")

    # a second call decides nothing new, so it leaves the file shut rather than rewriting it byte for byte
    before = path.read_bytes()
    line2, written2 = SV.write_signs(tmp_path, measured, dated="2026-09-19")
    assert written2 == [] and "was not opened" in line2
    assert path.read_bytes() == before, "a run that wrote nothing still rewrote the table"


def test_the_dipole_block_reads_a_table_and_a_default(tmp_path):
    """Fails if survey.yaml `dipoles` does not serve both shapes. An EDL raw folder records no arm length,
    so this block is the only path a new EDL survey has from the deployment sheet to sites.csv."""
    (tmp_path / "dipoles.csv").write_text(
        "site,dipole_n_m,dipole_e_m,source\nQ01,10.2,11.2,sheet 2026-09-05\n", encoding="utf-8")
    cfg = {"dipoles": {"table": "dipoles.csv", "default_m": 100, "reason": "the AusLAMP nominal arm"}}
    table = SV.dipole_table(cfg, tmp_path)
    assert list(table.columns) == SV.DIPOLE_TABLE_COLUMNS
    assert table.at[0, "site"] == "Q01" and table.at[0, "source"] == "sheet 2026-09-05"
    assert SV.dipole_default(cfg, "PR6-24+Mag-03") == (100.0, "the AusLAMP nominal arm")
    # the older per-instrument spelling still reads, and an absent block is no default at all
    old = {"dipole_default": {"PR6-24+Mag-03": {"value": 50, "reason": "why"}}}
    assert SV.dipole_default(old, "PR6-24+Mag-03") == (50.0, "why")
    assert SV.dipole_default(old, "LEMI-424") == (None, "")
    assert SV.dipole_default({}, "PR6-24+Mag-03") == (None, "")
    assert not len(SV.dipole_table({}, tmp_path))
