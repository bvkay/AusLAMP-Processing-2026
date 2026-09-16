"""Create surveys/<survey>/decisions.csv with 'decide' in every cell, carrying over signs already decided.

A survey worked on before arrives with decisions already made. This tool writes the blank decisions table and
copies in the five channel signs from a previous table wherever that table records HOW the sign was reached.
A sign whose *_how value is not in --accept stays 'decide'; a convention is not a measurement.

    python tools/prefill_decisions.py victoria \
        E:/MT_Timeseries_DATA/MT_AusLAMP_GA/AusLAMP_Victoria_BK/victoria_master.csv \
        --dated 2026-09-11 --accept decided measured

Run once. It refuses to overwrite an existing decisions.csv unless --force is given.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auslamp_proc.survey import DECIDE, DECISIONS_COLUMNS, SURVEYS, blank_decisions  # noqa: E402

SIGNS = [("sign_hx", "sign_Hx"), ("sign_hy", "sign_Hy"), ("sign_hz", "sign_Hz"),
         ("sign_ex", "sign_Ex"), ("sign_ey", "sign_Ey")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("survey")
    ap.add_argument("master", help="the previous table to carry decisions over from")
    ap.add_argument("--dated", default="", help="the date of that table, written into sign_source")
    ap.add_argument("--accept", nargs="*", default=["decided", "measured"],
                    help="the *_how values that count as a decision; anything else stays 'decide'")
    ap.add_argument("--site-column", default="site")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out_path = SURVEYS / a.survey / "decisions.csv"
    if out_path.exists() and not a.force:
        print("refusing to overwrite %s (use --force)" % out_path)
        return 1
    m = pd.read_csv(a.master)
    dec = blank_decisions(sorted(m[a.site_column].astype(str)))
    dec = dec.set_index("site")
    label = "%s %s" % (Path(a.master).name, a.dated) if a.dated else Path(a.master).name
    n_taken, n_left = 0, 0
    for _, r in m.iterrows():
        site = str(r[a.site_column])
        taken = []
        for col, mcol in SIGNS:
            how = str(r.get(mcol + "_how", "")).strip()
            val = r.get(mcol, None)
            if how and how.split()[0].lower() in [x.lower() for x in a.accept] and pd.notna(val):
                dec.loc[site, col] = "%+d" % int(float(val))
                taken.append("%s %s" % (col, how.split()[0]))
                n_taken += 1
            else:
                n_left += 1
        if taken:
            dec.loc[site, "sign_source"] = "%s, %s" % (label, "; ".join(taken))
    dec = dec.reset_index()[DECISIONS_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dec.to_csv(out_path, index=False)
    n_sites = int((dec.sign_source != DECIDE).sum())
    print("wrote %s: %d rows, %d sign cells carried over at %d sites, %d cells left to decide"
          % (out_path, len(dec), n_taken, n_sites, n_left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
