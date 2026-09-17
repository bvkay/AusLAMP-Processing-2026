"""Reproducibility on halves: the same product re-estimated on each half of its record, and their agreement.

    python -m auslamp_proc.halves --survey queensland_phase1 --site Q49 --kind remote --rate 1 \
        --run halves --stamp 20260917_0600

The check the frozen tools made with re-runs (wamt_esp2026_readings.py and vic_collect.py read the readings
of several run folders and applied one rule across them; D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing).
Here it is run rather than read: a candidate product of record is re-estimated twice, once on each half of
its own record, and the two are put through the agreement rule of `readings.agree` over the quality band. A
product whose two halves disagree was estimated on something that changed inside the record.

The record is not cut. Each half is a keep mask handed to the estimator on top of the transient mask -- the
same `keep_extra` a workbook 05 form uses -- so the mask is applied inside the pass and one Aurora run is
written per kept stretch, exactly as the whole-record product was. The split is by SAMPLE INDEX at the
midpoint of the record, so the two halves are the same length whatever the gaps hold; the days each half
actually keeps after the transient mask are reported beside the verdict, because a record whose second half
is mostly masked reproduces on a shorter record than its first.

The cost is two passes per candidate. The two products land in <work_root>/<site>/halves_<stamp>/ as
<site>_half1_<kind>_<rate>hz_<params>.edi and <site>_half2_..., with the run folder's provenance.json and
halves.csv naming what each was built on.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse                                                              # noqa: E402
import json                                                                  # noqa: E402
import sys                                                                   # noqa: E402
import time                                                                  # noqa: E402
from datetime import datetime, timezone                                      # noqa: E402
from pathlib import Path                                                     # noqa: E402

import numpy as np                                                           # noqa: E402
import pandas as pd                                                          # noqa: E402

from . import products as PR, readings as RD, survey as SV                   # noqa: E402
from .raw import cache                                                       # noqa: E402

RUN_NAME = "halves"
HALVES = ("half1", "half2")
HALVES_COLUMNS = ["site", "kind", "rate_hz", "params", "component", "half1", "half2",
                  "half1_days", "half2_days", "rho_dev", "phase_dev_deg", "rho_ratio", "phase_diff_deg",
                  "n", "reproducible", "why", "seconds"]


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def half_masks(n: int) -> tuple:
    """Two boolean masks over `n` samples: the first half and the second, split at the midpoint index."""
    n = int(n)
    a = np.zeros(n, bool)
    b = np.zeros(n, bool)
    mid = n // 2
    a[:mid] = True
    b[mid:] = True
    return a, b


def record_samples(sv, site, rate=1) -> int:
    """How many samples the site's cache holds at one rate."""
    path = Path(sv.cfg["work_root"]) / ("cache_%dhz" % int(rate)) / ("%s.npz" % site)
    z = np.load(path, allow_pickle=False)
    try:
        return int(len(z[cache.CHANNELS[0]]))
    finally:
        z.close()


def run_dir(work_root, site, stamp) -> Path:
    return Path(work_root) / str(site) / ("%s_%s" % (RUN_NAME, stamp))


def reproducible(a, b, comp: str, band=RD.QUALITY_BAND, agree_rho=RD.AGREE_RHO,
                 agree_phase=RD.AGREE_PHASE) -> dict:
    """The agreement of one component's two half products over the quality band.

    The rule is `readings.agree` read over `band` rather than over the agreement band: the question is
    whether the product reproduces where it is delivered, which is 10-1000 s.
    """
    s = RD.agree(a, b, comp, band[0], band[1], agree_rho, agree_phase)
    s["reproducible"] = bool(s["agrees"])
    s["why"] = ("the two halves agree within %.0f %% in rho and %.1f deg in phase over %g-%g s"
                % (100 * agree_rho, agree_phase, band[0], band[1])) if s["agrees"] else \
               ("the two halves differ by %.0f %% in rho and %.1f deg in phase over %g-%g s"
                % (100 * (s["rho_dev"] if np.isfinite(s["rho_dev"]) else np.nan),
                   s["phase_dev_deg"] if np.isfinite(s["phase_dev_deg"]) else np.nan, band[0], band[1])
                if np.isfinite(s["rho_dev"]) else
                "the two halves carry no common band over %g-%g s" % (band[0], band[1]))
    return s


def split_half(sv, site, kind, rate=1, run=RUN_NAME, stamp=None, params="kaiser20_75", redo=False,
               band=RD.QUALITY_BAND, agree_rho=RD.AGREE_RHO, agree_phase=RD.AGREE_PHASE,
               verbose=True) -> dict:
    """Run both half passes of one product and score them. Returns the run record.

    The two passes go through `site.forms.run_form`, which is the pass workbook 03's CLI runs -- the same
    frame, signs, transient mask, reference store, band file and parameter set -- with one extra keep mask.
    The mask cannot be handed through decisions.csv, which is the analyst's table, so it is passed in.
    """
    from .site import forms as FM

    stamp = stamp or stamp_now()
    out_dir = run_dir(sv.cfg["work_root"], site, stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = record_samples(sv, site, rate)
    masks = dict(zip(HALVES, half_masks(n)))
    t_start = time.time()
    rows, paths = {}, {}
    for name, mask in masks.items():
        row = FM.run_form(sv, site, name, out_dir, kind=kind, rate=rate, params=params,
                          keep_extra=mask, redo=redo, verbose=verbose,
                          keep_name=("%s of %d samples: the %s half of the record by sample index"
                                     % (name, n, "first" if name == "half1" else "second")),
                          criterion=("reproducibility: the two halves must agree within %.0f %% in rho and "
                                     "%.1f deg in phase over %g-%g s"
                                     % (100 * agree_rho, agree_phase, band[0], band[1])),
                          controls=[o for o in HALVES if o != name],
                          extra_lines=["half=%s of the record, %d of %d samples, applied as a keep mask "
                                       "inside the estimator and not as a cut" % (name, int(mask.sum()), n)])
        rows[name] = row
        paths[name] = row.get("product")
    out = dict(site=site, kind=kind, rate_hz=float(rate), params=params, stamp=stamp,
               run_dir=str(out_dir), seconds=round(time.time() - t_start, 1),
               half1_status=rows["half1"].get("status"), half2_status=rows["half2"].get("status"),
               half1_days=rows["half1"].get("days"), half2_days=rows["half2"].get("days"),
               half1=str(paths["half1"]), half2=str(paths["half2"]),
               error="; ".join(str(rows[k].get("error") or "") for k in HALVES).strip("; "),
               components={})
    ok = all(rows[k].get("status") in ("made", "exists") and Path(str(paths[k])).exists() for k in HALVES)
    if ok:
        a, b = PR.read_tf(paths["half1"]), PR.read_tf(paths["half2"])
        for comp in RD.COMPONENTS:
            out["components"][comp] = reproducible(a, b, comp, band, agree_rho, agree_phase)
    return out


def halves_table(records, out_path=None) -> pd.DataFrame:
    """One row per candidate and component: the two halves, their departure and the verdict."""
    rows = []
    for rec in records:
        for comp in RD.COMPONENTS:
            s = (rec.get("components") or {}).get(comp)
            rows.append(dict(site=rec["site"], kind=rec["kind"], rate_hz=rec["rate_hz"],
                             params=rec.get("params", ""), component=comp,
                             half1=Path(str(rec.get("half1"))).name, half2=Path(str(rec.get("half2"))).name,
                             half1_days=rec.get("half1_days"), half2_days=rec.get("half2_days"),
                             rho_dev=(s or {}).get("rho_dev", np.nan),
                             phase_dev_deg=(s or {}).get("phase_dev_deg", np.nan),
                             rho_ratio=(s or {}).get("rho_ratio", np.nan),
                             phase_diff_deg=(s or {}).get("phase_diff_deg", np.nan),
                             n=(s or {}).get("n", 0),
                             reproducible=("UNJUDGED" if s is None else
                                           ("yes" if s["reproducible"] else "no")),
                             why=((rec.get("error") or "one half pass did not produce a file")
                                  if s is None else s["why"]),
                             seconds=rec.get("seconds")))
    t = pd.DataFrame(rows, columns=HALVES_COLUMNS)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out_path, index=False)
    return t


def read_existing(work_root, site, kind, rate=1, params="kaiser20_75", stamp=None,
                  band=RD.QUALITY_BAND, agree_rho=RD.AGREE_RHO, agree_phase=RD.AGREE_PHASE) -> dict | None:
    """The record of a half pass already on disk, or None. What HALVES = False reads instead of running."""
    from .site.forms import product_name

    work = Path(work_root)
    folders = sorted((work / str(site)).glob("%s_*" % RUN_NAME))
    if stamp:
        folders = [f for f in folders if f.name.endswith(stamp)]
    for folder in reversed(folders):
        paths = {k: folder / product_name(site, k, kind, rate, params) for k in HALVES}
        if not all(p.exists() for p in paths.values()):
            continue
        # what each half kept is in the run folder's provenance, written when the pass ran: a record whose
        # second half is mostly masked reproduces on a shorter record than its first, and the verdict says so
        stored = {}
        try:
            doc = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
            stored = {f.get("product"): f for f in (doc.get("forms") or [])}
        except Exception:
            stored = {}
        rec = dict(site=site, kind=kind, rate_hz=float(rate), params=params,
                   stamp=RD.split_run_folder(folder.name)[1], run_dir=str(folder),
                   seconds=sum(float(stored.get(str(paths[k]), {}).get("seconds") or 0.0)
                               for k in HALVES) or None,
                   half1_status="exists", half2_status="exists",
                   half1_days=stored.get(str(paths["half1"]), {}).get("days"),
                   half2_days=stored.get(str(paths["half2"]), {}).get("days"),
                   half1=str(paths["half1"]), half2=str(paths["half2"]), error="", components={})
        a, b = PR.read_tf(paths["half1"]), PR.read_tf(paths["half2"])
        for comp in RD.COMPONENTS:
            rec["components"][comp] = reproducible(a, b, comp, band, agree_rho, agree_phase)
        return rec
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the two half passes of one site's candidates, and their "
                                             "agreement")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--kinds", nargs="+", required=True,
                    help="the reference kinds of this site to re-estimate on each half")
    ap.add_argument("--rate", type=int, default=1)
    ap.add_argument("--run", default=RUN_NAME)
    ap.add_argument("--stamp", default="", help="the launch time in UTC as YYYYmmdd_HHMM, shared by a run")
    ap.add_argument("--params", default="kaiser20_75")
    ap.add_argument("--work-root", default="", help="override survey.yaml work_root")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    sv = SV.load_survey(a.survey)
    if a.work_root:
        sv.cfg["work_root"] = a.work_root
    # one site a lane, its kinds in turn: the two passes of one site share a scratch folder and an MTH5
    # name, so two lanes on one site would take each other's file away
    failed, out = [], []
    for kind in a.kinds:
        rec = split_half(sv, a.site, kind, a.rate, run=a.run, stamp=(a.stamp or None), params=a.params,
                         redo=a.redo, verbose=not a.quiet)
        out.append({k: v for k, v in rec.items() if k != "components"})
        failed += ["%s %s" % (kind, k) for k in HALVES
                   if rec.get("%s_status" % k) not in ("made", "exists")]
    print(json.dumps(out, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
