"""One survey, many sites: the reference store, then one lane per site over the kinds and the rates.

    python -m auslamp_proc.process.batch --survey queensland_phase1 --sites all --lanes 3
    python -m auslamp_proc.process.batch --survey queensland_phase1 --redo --sites Q49 Q84

This is the batch entry of the package. A workbook runs one site at a time; this runs the survey, and the two
share one code path: every lane is `python -m auslamp_proc.process.run` over one site, which is the same CLI a
workbook calls.

Each lane pins its BLAS threads to THREADS_PER_LANE = 3. A lane that takes every core makes three concurrent
lanes slower than one, and the memory a pass peaks at is per lane: over AusLAMP Queensland Phase 1, whose
records run 12-62 days, a 1 Hz pass peaks at 2.5-4.8 GB and takes 140-900 s, so three lanes at 1 Hz is the
default and a 10 Hz stretch of at most 48 h runs two.

The references are built before the lanes and never inside one. A pass reads the store; two transfer functions
of the same kind are then built on the same array. `--redo` rebuilds the store for the named sites as well as
remaking their transfer functions, which is what a changed membership rule or a changed decisions.csv row
needs: `--redo --sites Q49 Q84` rebuilds those sites' references and remakes every transfer function of
theirs, and leaves every other site alone.

The stamp is the launch time in UTC and every site of one batch shares it, so one run folder name covers the
survey. A batch is resumed rather than restarted: where <site>/<run>_<stamp>/ folders already exist the
latest of their stamps is taken up again, and without --redo a transfer function whose EDI is on disk is
reported `exists` and left alone.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import survey as SV
from . import KINDS, aurora_run, references as REF, selection as SEL
from .run import REFUSED_KINDS

THREADS_PER_LANE = 3                # the BLAS threads one lane is pinned to
DEFAULT_LANES = 3                   # concurrent single-site subprocesses at 1 Hz
DEFAULT_KINDS = tuple(k for k in KINDS if k not in REFUSED_KINDS)


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def resume_stamp(work, sites, run_name) -> str:
    """The latest stamp of the run folders already on disk under this run name, or a new one."""
    stamps = sorted({p.name.split("_", 1)[1] for s in sites
                     for p in (Path(work) / s).glob(run_name + "_*") if p.is_dir()})
    return stamps[-1] if stamps else stamp_now()


def chosen_sites(sv, sites, rate=1) -> tuple:
    """(the sites with a cache at `rate`, the sites named that have none)."""
    work = Path(sv.cfg["work_root"])
    named = (list(sv.sites.site) if str(sites).strip().lower() == "all"
             else [str(s) for s in sites])
    have = [s for s in named if (work / ("cache_%dhz" % int(rate)) / ("%s.npz" % s)).exists()]
    return have, [s for s in named if s not in have]


def lane(site, survey_name, run_name, stamp, kinds, rates, params, selections=(), redo=False,
         work_root="", repo=None, threads=THREADS_PER_LANE) -> dict:
    """One subprocess over one site. Returns the row the caller prints."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-m", "auslamp_proc.process.run", "--survey", survey_name, "--site", site,
           "--run", run_name, "--stamp", stamp, "--params", params, "--quiet",
           "--kinds"] + list(kinds) + ["--rates"] + [str(r) for r in rates]
    if selections:
        cmd += ["--selections"] + list(selections)
    if redo:
        cmd.append("--redo")
    if work_root:
        cmd += ["--work-root", str(work_root)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(repo or Path.cwd()), capture_output=True, text=True, env=env)
    return dict(site=site, returncode=r.returncode, seconds=round(time.time() - t, 1),
                stdout=(r.stdout or "").strip()[-300:], stderr=(r.stderr or "").strip()[-400:])


def build_references(sv, sites, rates, kinds, redo=False, verbose=True) -> dict:
    """The reference store for the chosen sites at each rate, the higher rates from the 1 Hz specification."""
    out = {}
    spec = REF.Store(sv, list(sv.sites.site), rate=1)
    for rate in rates:
        want = tuple(k for k in kinds if k in REF.KINDS_WITH_STORE)
        if int(rate) != 1:
            want = tuple(k for k in want if k != "obs")     # the observatory is not a kind above 1 Hz
        st = spec if int(rate) == 1 else REF.Store(sv, list(sv.sites.site), rate=int(rate), spec_rate=1)
        t = time.time()
        got = REF.build_store(sv, sites, rate=int(rate), kinds=want, force=redo, verbose=False,
                              spec_store=(None if int(rate) == 1 else spec), store=st)
        out[int(rate)] = got
        if verbose:
            print("references %2d Hz: %d site(s), %d reference(s), %.1f s"
                  % (int(rate), len(got), sum(len(v) for v in got.values()), time.time() - t), flush=True)
    return out


def build_selections(sv, sites, redo=False, verbose=True, run_name="", stamp="", rates=(10,),
                     kinds=()) -> dict:
    """The hour score and the stretch of each site, written before any lane reads one.

    A pass rebuilds the sample mask from the interval in <work_root>/<site>/hour_selection.json and raises
    where the site carries none, so the stretch is chosen here, once per site, and never inside a lane: two
    lanes scoring the same record would read the same 1 Hz cache twice and write the same file twice.

    A stretch the site's own mask leaves with no run of transients.MIN_SEGMENT_S is refused by the rule
    before any pass, and that refusal is written to the ledger as one row per kind with status `refused` and
    the reason, so a site that made nothing is a stated refusal in the record and not an absence.
    """
    from .run import append_row, RUNS_COLUMNS
    work = Path(sv.cfg["work_root"])
    out, refusals = {}, []
    t = time.time()
    for s in sites:
        sel = SEL.build(sv, s, force=redo)
        out[s] = {k: v.get("n_hours") for k, v in sel.items()}
        mine = [(tag, item["refused"]) for tag, item in sorted(sel.items()) if item.get("refused")]
        refusals += [(s, tag, why) for tag, why in mine]
        if verbose:
            said = ", ".join("%s %s h" % kv for kv in sorted(out[s].items()))
            why = "; ".join("%s refused: %s" % kv for kv in mine)
            print("   stretch %-9s %s%s" % (s, said or "none: no coherent stretch",
                                            ("   -- " + why) if why else ""), flush=True)
    for site, tag, why in refusals:
        for kind in (kinds or ("",)):
            row = {c: None for c in RUNS_COLUMNS}
            row.update(site=site, run=run_name, stamp=stamp, kind=kind,
                       rate_hz=float(max(int(r) for r in rates)), params="", selection=tag,
                       status="refused", error=why)
            append_row(work / "survey" / "runs.csv", row)
    if verbose:
        print("the stretch of %d site(s) in %.1f s; %d refused by the mask before any pass"
              % (len(sites), time.time() - t, len(refusals)), flush=True)
    return out


def run_batch(survey_name, sites="all", lanes=DEFAULT_LANES, run_name="first", stamp="",
              kinds=DEFAULT_KINDS, rates=(1,), params="kaiser20_75", selections=(), redo=False,
              work_root="", references=True, verbose=True) -> dict:
    """The whole batch: the references, the stretches, then `lanes` sites at a time. What it did comes back."""
    import auslamp_proc

    repo = Path(auslamp_proc.__file__).resolve().parent.parent
    sv = SV.load_survey(survey_name)
    if work_root:
        sv.cfg["work_root"] = work_root
    work = Path(sv.cfg["work_root"])
    have, no_cache = chosen_sites(sv, sites, rate=min(int(r) for r in rates))
    stamp = stamp or resume_stamp(work, have, run_name)
    if verbose:
        print("survey     %s" % sv.cfg["name"])
        print("work root  %s" % work)
        print("sites      %d of %d named carry a cache" % (len(have), len(have) + len(no_cache)))
        if no_cache:
            print("no cache   %s -- run workbook 02 over them first" % " ".join(no_cache))
        print("kinds      %s" % " ".join(kinds))
        print("rates      %s Hz, parameter set %s, %d lane(s)"
              % (", ".join(str(r) for r in rates), params, int(lanes)))
        print("selections %s" % (" ".join(selections) or SEL.WHOLE))
        print("run        %s_%s" % (run_name, stamp), flush=True)
    refs = {}
    if references and have:
        refs = build_references(sv, have, rates, kinds, redo=redo, verbose=verbose)
    picked = {}
    if have and [s for s in selections if s != SEL.WHOLE]:
        picked = build_selections(sv, have, redo=redo, verbose=verbose, run_name=run_name, stamp=stamp,
                                  rates=rates, kinds=kinds)
        # a lane is asked only for the tags that survived at its own site: the rule refuses a stretch and
        # its control independently, so a site can carry one and not the other
        per_site = {s: [t for t in selections if t == SEL.WHOLE or picked.get(s, {}).get(t)]
                    for s in have}
        have = [s for s in have if per_site[s]]
        if verbose:
            print("sites      %d carry a stretch and enter the lanes" % len(have), flush=True)
            for s in have:
                if len(per_site[s]) != len([t for t in selections]):
                    print("           %-9s asks for %s alone" % (s, " ".join(per_site[s])), flush=True)
    else:
        per_site = {s: list(selections) for s in have}
    t0 = time.time()
    done, results = 0, []
    with cf.ThreadPoolExecutor(max_workers=int(lanes)) as ex:
        for r in ex.map(lambda s: lane(s, survey_name, run_name, stamp, kinds, rates, params,
                                       selections=per_site.get(s, list(selections)), redo=redo,
                                       work_root=work_root, repo=repo), have):
            done += 1
            results.append(r)
            if verbose:
                print("%2d/%2d %-9s exit %d %7.1f s  %s%s"
                      % (done, len(have), r["site"], r["returncode"], r["seconds"], r["stdout"],
                         ("  || " + r["stderr"]) if r["returncode"] else ""), flush=True)
    wall = time.time() - t0
    failed = [r["site"] for r in results if r["returncode"]]
    if verbose:
        print()
        print("wall time %.1f min over %d site(s) in %d lane(s); %d lane(s) exited non-zero (%s)"
              % (wall / 60.0, len(have), int(lanes), len(failed), ", ".join(failed) or "none"), flush=True)
    return dict(survey=survey_name, run=run_name, stamp=stamp, sites=have, no_cache=no_cache,
                lanes=int(lanes), seconds=round(wall, 1), failed=failed, results=results,
                references={str(k): len(v) for k, v in refs.items()}, selections=picked)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="one survey over many sites: the references, then one lane a site")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--sites", nargs="+", default=["all"], help='"all" or the site names to run')
    ap.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    ap.add_argument("--run", default="first", help="the run name; the folder is <site>/<run>_<stamp>")
    ap.add_argument("--stamp", default="", help="the launch time in UTC as YYYYmmdd_HHMM, shared by a batch")
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS), choices=list(DEFAULT_KINDS))
    ap.add_argument("--rates", nargs="+", type=int, default=[1])
    ap.add_argument("--params", default=aurora_run.DEFAULT_PARAMS,
                    choices=sorted(aurora_run.AURORA_PARAMS))
    ap.add_argument("--selections", nargs="+", default=[], choices=list(SEL.TAGS) + [SEL.WHOLE],
                    help="the stretches each kind is run on: stretch control; none = the whole record")
    ap.add_argument("--work-root", default="", help="override survey.yaml work_root")
    ap.add_argument("--redo", action="store_true",
                    help="rebuild the named sites' references and remake every transfer function of theirs")
    ap.add_argument("--no-references", action="store_true",
                    help="run the lanes on the store as it stands and build nothing")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    sites = "all" if [s.lower() for s in a.sites] == ["all"] else a.sites
    out = run_batch(a.survey, sites=sites, lanes=a.lanes, run_name=a.run, stamp=a.stamp, kinds=a.kinds,
                    rates=a.rates, params=a.params, selections=a.selections, redo=a.redo,
                    work_root=a.work_root, references=not a.no_references, verbose=not a.quiet)
    print(json.dumps({k: out[k] for k in ("survey", "run", "stamp", "lanes", "seconds", "failed")}))
    return 1 if out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
