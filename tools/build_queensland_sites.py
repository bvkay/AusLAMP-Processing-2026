"""Merge the AusLAMP Queensland deployment-sheet cells into a phase's sites.csv and write its decisions.csv.

Workbook 01 writes what the raw tree holds: the span, the position from the logger's own GPS, the file count.
The dipole lengths, the arm directions, the recorder serial and the field notes are on the deployment sheets
and nowhere in the raw, so they are merged in here, once per phase, from the tables the sheets were
transcribed into on 2026-09-05:

    Phase 1  work/qld_phase1/phase1_master.csv        ex_dipole_m ey_dipole_m ex_laid ey_laid
                                                      ex_polarity ey_polarity geometry_confidence
                                                      serial serial_how sheet_notes
    Phase 2  QLD_AusLAMP_process.xlsx sheet Phase2    transcribed into SHEET2 and NOTES2 below, because the
                                                      environment carries no xlsx reader; the serial comes
                                                      from work/campaign_v2/phase2_master.csv
    Phase 3  work/qld_phase3/phase3_master.csv        the same columns as Phase 1

    python tools/build_queensland_sites.py queensland_phase1

The rules applied:

    azimuth        the direction as laid: N 0, S 180, E 90, W 270 deg
    sign_ex/ey     the campaign's -s_orient rule, orientation 0 or 90 -> -1 and 180 or 270 -> +1, with
                   qld_phase1.py E_SIGN_OVERRIDE applied (Q84 Ex, Q74N Ey). Phases 1 and 3 carry the rule's
                   result in their master's ex_polarity and ey_polarity columns and it is read from there;
                   Phase 2 has no such column and the rule is applied here to the sheet direction.
    OPEN           Q85N sign_ey, Q63 sign_ey and Q65 sign_hx/sign_hy stay `decide` and carry the open
                   question in flags; the sign is Ben's to rule.
    remote_site    the remote the campaign chose, from the phase's own remotes table, with remote_source
                   naming that table.

The sheet latitude and longitude are NOT written: the position of record is the logger's own GPS, and the
sheet position is compared in workbook 01 through survey.yaml `comparison_positions`.

Run once per phase, after workbook 01 has created sites.csv and decisions.csv.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auslamp_proc.survey import DECIDE, DECISIONS_COLUMNS, SITES_COLUMNS, SURVEYS  # noqa: E402

WORK = Path("D:/BEN/MTH5_Aurora_mt-io_2026/work")
XLSX = "E:/MT_Timeseries_DATA/MT_AusLAMP_QLD/Phase2/QLD_AusLAMP_process.xlsx"

AZIMUTH = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
POLARITY = {0.0: -1, 90.0: -1, 180.0: 1, 270.0: 1}

SIGN_SOURCE = ("campaign rule -s_orient on the sheet direction (0/90 -> -1, 180/270 -> +1); "
               "overrides qld_phase1.py E_SIGN_OVERRIDE")
OPEN_FLAG = ("polarity question open since 2026-09-08 (Q85N Ey +1 by quadrant; Q63 Ey reversed vs sheet; "
             "Q65 magnetometer horizontals exchanged) -- Ben to rule")
# site -> the decisions columns that stay `decide` and carry OPEN_FLAG
OPEN = {"Q85N": ["sign_ey"], "Q63": ["sign_ey"], "Q65": ["sign_hx", "sign_hy"]}

# QLD_AusLAMP_process.xlsx sheet Phase2, columns Ex (m), 'Ex orintation' [sic], Ey (m), Ey Orientation:
# site: (Ex m, Ey m, Ex orientation deg, Ey orientation deg)
SHEET2 = {
    "Q46N": (10.9, 9.4, 180, 270), "Q47N": (9.6, 11.8, 180, 270), "Q48N": (9.2, 10.8, 0, 270),
    "Q56N": (12.0, 10.7, 0, 90), "Q57N": (10.3, 11.9, 0, 270), "Q58N": (9.0, 11.9, 0, 90),
    "Q59N": (12.1, 12.3, 0, 270), "Q66N": (10.2, 10.9, 0, 90), "Q67N": (9.0, 9.8, 180, 270),
    "Q68N": (14.8, 14.3, 0, 90), "Q69N": (12.1, 10.0, 0, 270), "Q70N": (10.0, 10.6, 0, 90),
    "Q71N": (9.4, 9.2, 0, 90), "Q78N": (10.9, 10.5, 0, 90), "Q79N": (11.0, 11.0, 0, 90),
    "Q80N": (9.2, 9.5, 0, 90), "Q81N": (11.0, 12.3, 0, 90), "Q82": (9.2, 9.4, 0, 90),
}
# the same sheet's notes columns, verbatim
NOTES2 = {
    "Q46N": "Low freq noise on Bz - nosiy under 10s",
    "Q47N": "All cable BROKEN, N(box) went to West, E(box) went to South; Ex & Ey has been Swaped",
    "Q48N": "Center been pulled out; NEED detrend",
    "Q56N": "Problem in the middle chunk",
    "Q57N": "Box is burnt. Replaced with center control #29; Remove Spike",
    "Q58N": "Center cable broken; Remove Spike",
    "Q59N": "West and Center been pulled out",
    "Q66N": "North cable broken; Zxy - North cable been broken at the beginning?",
    "Q67N": "Rocky Soil; Noisy from 15/MAR, try to reduce the daily temperature variation",
    "Q68N": "East Cable been chewed by goats",
    "Q71N": "CHECK WITH BART! CABLE PROBLEM; Bx from 06/Apr",
}

PHASES = {
    "queensland_phase1": dict(
        master=WORK / "qld_phase1" / "phase1_master.csv",
        remotes=WORK / "qld_phase1" / "phase1_remotes_table.csv", remote_col="remote_site",
        dipole_date="2026-09-05", sheet="deployment sheet"),
    "queensland_phase2": dict(
        master=WORK / "campaign_v2" / "phase2_master.csv",
        remotes=WORK / "campaign_v2" / "phase2_master.csv", remote_col="p2_remote",
        dipole_date="2026-09-16", sheet="deployment sheet QLD_AusLAMP_process.xlsx sheet Phase2"),
    "queensland_phase3": dict(
        master=WORK / "qld_phase3" / "phase3_master.csv",
        remotes=WORK / "qld_phase3" / "phase3_remotes_table.csv", remote_col="remote_site",
        dipole_date="2026-09-05", sheet="deployment sheet"),
}


def _cell(frame, site, col, default=""):
    if site not in frame.index or col not in frame.columns:
        return default
    v = frame.loc[site, col]
    return default if (v != v or str(v).strip() == "") else v


def sheet_facts(survey: str) -> dict:
    """{site: {dipole_n_m, dipole_e_m, laid_n, laid_e, sign_ex, sign_ey, confidence, serial, serial_source,
    notes}} for one phase, read from that phase's transcription of its deployment sheet."""
    spec = PHASES[survey]
    m = pd.read_csv(spec["master"]).set_index("site")
    out = {}
    for site in m.index:
        if survey == "queensland_phase2":
            if site not in SHEET2:
                continue
            ex, ey, exo, eyo = SHEET2[site]
            laid = {v: k for k, v in AZIMUTH.items()}
            out[site] = dict(
                dipole_n_m=ex, dipole_e_m=ey,
                laid_n=laid[float(exo)], laid_e=laid[float(eyo)],
                sign_ex=POLARITY[float(exo)], sign_ey=POLARITY[float(eyo)],
                confidence="not recorded",
                serial=_cell(m, site, "serial"),
                serial_source="campaign_v2/phase2_master.csv serial column",
                notes=NOTES2.get(site, ""))
        else:
            out[site] = dict(
                dipole_n_m=_cell(m, site, "ex_dipole_m"), dipole_e_m=_cell(m, site, "ey_dipole_m"),
                laid_n=_cell(m, site, "ex_laid"), laid_e=_cell(m, site, "ey_laid"),
                sign_ex=_cell(m, site, "ex_polarity", None), sign_ey=_cell(m, site, "ey_polarity", None),
                confidence=_cell(m, site, "geometry_confidence"),
                serial=_cell(m, site, "serial"),
                serial_source=str(_cell(m, site, "serial_how")),
                notes=str(_cell(m, site, "sheet_notes")))
    return out


def _sign(v):
    """A polarity of +-1 as '+1' or '-1', or 'decide' where the master records none."""
    if v is None or v != v or str(v).strip() == "":
        return DECIDE
    return "%+d" % int(float(v))


def merge_sites(survey: str, facts: dict) -> str:
    path = SURVEYS / survey / "sites.csv"
    if not path.exists():
        raise SystemExit("%s does not exist: run workbook 01 on %s first" % (path, survey))
    spec = PHASES[survey]
    # object dtype: a str-backed column refuses a float, and every cell written below is text
    s = pd.read_csv(path, dtype=str, keep_default_na=False).astype(object)
    filled = {c: 0 for c in ("serial", "dipole_n_m", "azimuth_n_deg", "notes")}
    missing = []
    for i, r in s.iterrows():
        f = facts.get(r["site"])
        if f is None:
            missing.append(r["site"])
            continue
        if str(f["serial"]).strip():
            s.at[i, "serial"] = str(f["serial"]).split(".")[0]
            s.at[i, "serial_source"] = f["serial_source"]
            filled["serial"] += 1
        if str(f["dipole_n_m"]).strip() and str(f["dipole_e_m"]).strip():
            s.at[i, "dipole_n_m"] = "%g" % float(f["dipole_n_m"])
            s.at[i, "dipole_e_m"] = "%g" % float(f["dipole_e_m"])
            s.at[i, "dipole_source"] = ("%s transcribed %s, confidence %s"
                                        % (spec["sheet"], spec["dipole_date"], f["confidence"]))
            filled["dipole_n_m"] += 1
        if f["laid_n"] in AZIMUTH and f["laid_e"] in AZIMUTH:
            s.at[i, "azimuth_n_deg"] = "%g" % AZIMUTH[f["laid_n"]]
            s.at[i, "azimuth_e_deg"] = "%g" % AZIMUTH[f["laid_e"]]
            s.at[i, "azimuth_source"] = "deployment sheet (direction as laid)"
            filled["azimuth_n_deg"] += 1
        if str(f["notes"]).strip():
            s.at[i, "notes"] = str(f["notes"])
            filled["notes"] += 1
    s.reindex(columns=SITES_COLUMNS).to_csv(path, index=False)
    return ("sites.csv: %d rows; serial %d, dipoles %d, azimuths %d, notes %d filled from the sheet%s"
            % (len(s), filled["serial"], filled["dipole_n_m"], filled["azimuth_n_deg"], filled["notes"],
               ("; no sheet row for " + ", ".join(missing)) if missing else ""))


def merge_decisions(survey: str, facts: dict) -> str:
    path = SURVEYS / survey / "decisions.csv"
    if not path.exists():
        raise SystemExit("%s does not exist: run workbook 01 on %s first" % (path, survey))
    spec = PHASES[survey]
    rem = pd.read_csv(spec["remotes"]).set_index("site")
    d = pd.read_csv(path, dtype=str, keep_default_na=False).astype(object)
    n_sign, n_remote, n_open = 0, 0, 0
    for i, r in d.iterrows():
        site = r["site"]
        f = facts.get(site)
        held = OPEN.get(site, [])
        if f is not None:
            for col, val in (("sign_ex", f["sign_ex"]), ("sign_ey", f["sign_ey"])):
                if col in held:
                    continue
                d.at[i, col] = _sign(val)
                n_sign += 1
            if d.at[i, "sign_ex"] != DECIDE or d.at[i, "sign_ey"] != DECIDE:
                d.at[i, "sign_source"] = SIGN_SOURCE
        if held:
            d.at[i, "flags"] = OPEN_FLAG
            n_open += len(held)
        r_site = _cell(rem, site, spec["remote_col"])
        if str(r_site).strip():
            d.at[i, "remote_site"] = str(r_site)
            d.at[i, "remote_source"] = "%s, column %s" % (spec["remotes"].as_posix(), spec["remote_col"])
            n_remote += 1
    d.reindex(columns=DECISIONS_COLUMNS).to_csv(path, index=False)
    return ("decisions.csv: %d rows; %d E signs written, %d left open for Ben, %d remote sites prefilled"
            % (len(d), n_sign, n_open, n_remote))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("survey", choices=sorted(PHASES))
    a = ap.parse_args(argv)
    facts = sheet_facts(a.survey)
    print("%s: %d sites on the sheet (%s)" % (a.survey, len(facts), XLSX if a.survey.endswith("2")
                                              else PHASES[a.survey]["master"]))
    print(merge_sites(a.survey, facts))
    print(merge_decisions(a.survey, facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
