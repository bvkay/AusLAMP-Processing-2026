"""One site, end to end: the frame, the mask, the MTH5, the Aurora pass, the EDI and the provenance.

    python -m auslamp_proc.process.run --survey queensland_phase1 --site Q49 --run first \
        --kinds remote stack obs stack_obs --rates 1 --params kaiser20_75 [--redo]

Everything lands in <work_root>/<site>/<RUN>_<stamp>/, where the stamp is the launch time in UTC of the whole
run and is passed in with --stamp so every site of one run shares a folder name. Each pass writes
<site>_<kind>_<rate>hz_<params>.edi and .xml beside log.txt and provenance.json, and appends one row to
<work_root>/survey/runs.csv.

The CLI is resumable: with --redo absent, a transfer function whose EDI is already on disk is left alone and
reported as `exists`. A pass that fails is caught, its error is written into its ledger row, and the next
one runs.

A decisions.csv `keep_mask` cell naming a boolean .npy applies that selection of hours on top of the
transient mask. A cell naming a file that is not there is said loudly and the pass runs on the whole record,
because a selection silently ignored produces a whole-record transfer function in a folder named after
the selection.

`--selections` names the stretches of process.selection a pass is run on, one transfer function each:
`stretch` is the longest run of whole UTC hours in which both recorded lines read above 0.5 at 20-200 s, and
`control` is a run of the same length placed at random elsewhere in the record. A selected transfer function
carries its tag between the rate and the parameter set, `<site>_<kind>_10hz_<tag>_<params>.edi`, and the same
tag in the ledger's `selection` column and in the EDI's own `selection=` line. The default is the whole
record, whose name carries no tag, so the 1 Hz names and the 1 Hz path are unchanged.

`single` is refused as a kind: noise in H biases the single station low and its error bars carry no sign of
that bias (Ben's ruling, 2026-09-17). The key stays in process.KINDS so a file already written under it
still reads.

The BLAS thread count is pinned to 3 before numpy is imported. A lane that takes every core makes three
concurrent lanes slower than one, and the memory a pass peaks at is per lane.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse                                                              # noqa: E402
import json                                                                  # noqa: E402
import shutil                                                                # noqa: E402
import sys                                                                   # noqa: E402
import time                                                                  # noqa: E402
import traceback                                                             # noqa: E402
from datetime import datetime, timezone                                      # noqa: E402
from pathlib import Path                                                     # noqa: E402

import numpy as np                                                           # noqa: E402
import pandas as pd                                                          # noqa: E402

from .. import survey as SV                                                  # noqa: E402
from ..raw import cache                                                      # noqa: E402
from . import KINDS, aurora_run, edi as EDI, frame as FR, mth5_build, provenance as PROV  # noqa: E402
from . import rate as RATE, references as REF, selection as SEL, transients as TR  # noqa: E402

RUNS_COLUMNS = ["site", "run", "stamp", "kind", "rate_hz", "params", "selection", "remote", "members",
                "n_runs", "mask_dropped_frac", "floor_dropped_frac", "seconds", "peak_rss_mb", "status",
                "error", "edi", "xml"]
WHOLE = SEL.WHOLE                       # the untagged pass over the whole record
SINGLE_REFUSED = ("the single station is not a kind of this package: noise in H biases it low and its error "
                  "bars carry no sign of that bias (Ben's ruling, 2026-09-17)")
REFUSED_KINDS = {"single": SINGLE_REFUSED}


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def rss_mb() -> tuple:
    """(resident, peak resident) of this lane in MB, or (nan, nan) where psutil cannot say.

    The peak is the process's own peak since it started, so the second pass of a site reports the peak of
    the first as well: the column is what the lane cost by then, not what that one pass cost alone.
    """
    try:
        import psutil
        m = psutil.Process().memory_info()
        return m.rss / 2 ** 20, getattr(m, "peak_wset", m.rss) / 2 ** 20
    except Exception:
        return float("nan"), float("nan")


def widen_ledger(path):
    """Give a ledger written before a column existed that column, empty, and leave the rows alone.

    The ledger is appended to, so a file whose header is narrower than RUNS_COLUMNS would take rows with
    more fields than names and no reader could split it. The rewrite goes through a temporary file and one
    replace, so a reader sees either the old file or the new one.
    """
    path = Path(path)
    if not path.exists():
        return False
    d = pd.read_csv(path)
    if list(d.columns) == RUNS_COLUMNS:
        return False
    for c in RUNS_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    tmp = path.with_suffix(".rewrite")
    d[RUNS_COLUMNS].to_csv(tmp, index=False)
    os.replace(str(tmp), str(path))
    return True


def append_row(path, row):
    """Append one ledger row under a lock, so three lanes writing at once do not interleave a line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    for _ in range(600):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        lock.unlink(missing_ok=True)
    try:
        widen_ledger(path)
        header = not path.exists()
        pd.DataFrame([row], columns=RUNS_COLUMNS).to_csv(path, mode="a", header=header, index=False)
    finally:
        lock.unlink(missing_ok=True)


def _external_mask(dec_row, work, site, n, fs, say):
    """(the sample mask decisions.csv `keep_mask` names, its name), or (None, "").

    The cell is a path to a boolean .npy, absolute or relative to the work root, and a selection of hours
    made outside this run -- the best hours by coherence, a hand-drawn window. A cell that names a file that
    is not there is said loudly and the pass runs on the whole record: a selection silently ignored produces
    a whole-record transfer function sitting in a folder named after the selection.

    A mask built at 1 Hz is repeated over each second's samples where the pass runs faster.
    """
    cell = str(dec_row.get("keep_mask", "") if dec_row is not None else "").strip()
    if not cell or cell.lower() in ("decide", "nan", "none"):
        return None, ""
    p = Path(cell)
    if not p.is_absolute():
        p = Path(work) / cell
    if not p.exists():
        say("!! %s: decisions.csv keep_mask names %s and it is not there; this pass uses the WHOLE RECORD, "
            "not the selection" % (site, p))
        return None, ""
    m = np.load(p).astype(bool)
    if len(m) < n and fs > 1.0:
        m = np.repeat(m, int(round(fs)))
    if len(m) < n:
        say("!! %s: the keep mask %s is %d samples and the record is %d; it is NOT applied"
            % (site, p.name, len(m), n))
        return None, ""
    say("   %s: keep mask %s applied, %.1f %% of the record" % (site, p.name, 100 * m[:n].mean()))
    return m[:n], p.name


def _selection_mask(work, site, tag, t0, n, fs, say):
    """(the sample mask of one named stretch, its record), or (None, {}) for the whole record.

    The stretch is read from <work_root>/<site>/hour_selection.json, which process.selection wrote, and the
    mask is rebuilt from its interval. A tag with no entry there is raised rather than passed over: a
    selection silently ignored produces a whole-record transfer function under a name that says otherwise.
    """
    tag = str(tag or "").strip()
    if not tag or tag == WHOLE:
        return None, {}
    d = SEL.read_selection(work, site)
    item = (d.get("selections") or {}).get(tag)
    if item is None:
        raise FileNotFoundError("%s carries no %s stretch in %s; build the hour scores first"
                                % (site, tag, SEL.selection_path(work, site)))
    m = SEL.mask_from_hours(item["hours"], t0, n, fs)
    say("   %s: %s keeps %d hour(s), %.2f %% of the record, from %s UTC"
        % (site, tag, item["n_hours"], 100 * m.mean(), item.get("t_start")))
    return m, item


def load_local(sv, site, rate):
    """(t0, the decided and rotated five channels, the angle, the signs, the undecided channels, the record).

    Every decisions.csv decision is applied by raw.cache.load_decided, which calls the one place that
    applies one, process.frame.apply_decisions: the exchanges, the gain, the E shift, the signs and the
    lender's channels, in that fixed order, before the rotation. The record it returns carries the applied
    decisions as strings, which the EDI and provenance.json both carry verbatim.

    Above 1 Hz the horizontal magnetics go through the gap-edge screen first, the same one a member of the
    reference store goes through, so a site's own H and its H as somebody's reference are one record. It
    only widens what is already missing, so it cannot break a continuous stretch into pieces the 3,600 s
    run floor would then drop. The electric lines are not screened: their bursts are masked by interval
    from the tail scan, which is the mask the file's own header reports.
    """
    dec = sv.decision(site)
    t0, arrays, decisions_applied, rec = cache.load_decided(site, sv, rate)
    applied, undecided = rec["signs"], rec["undecided"]
    if int(rate) > 1:
        for c in ("Hx", "Hy"):
            arrays[c] = REF.gap_edge_screen(arrays[c], float(rate))
    regimes = FR.parse_regimes(dec.get("rot_regimes"))
    drop = FR.parse_regimes(dec.get("rot_drop"))
    turned, ang = FR.rotate_to_mean_field(arrays, regimes=regimes, drop=drop, fs=float(rate))
    local = {c: np.asarray(turned[c], float) for c in mth5_build.LOCAL_CHANNELS}
    rec["decisions_applied"] = list(decisions_applied)
    return t0, local, ang, applied, undecided, rec


def one_site(survey_name, site, run_name, kinds, rates, params_name, stamp=None, redo=False,
             work_root=None, verbose=True, selections=None) -> list:
    """Every asked-for transfer function of one site. Returns the ledger rows it wrote.

    `selections` is the stretches each kind is run on, one transfer function each: None or an empty list is
    the whole record, whose file carries no tag in its name.
    """
    refused = [k for k in kinds if k in REFUSED_KINDS]
    if refused:
        raise ValueError("; ".join(REFUSED_KINDS[k] for k in refused))
    aurora_run.silence_loggers()
    import aurora
    sv = SV.load_survey(survey_name)
    if work_root:
        sv.cfg["work_root"] = work_root
    work = Path(sv.cfg["work_root"])
    stamp = stamp or stamp_now()
    out_dir = work / site / ("%s_%s" % (run_name, stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "a", encoding="utf-8")

    def say(msg):
        line = "%s %s" % (datetime.now(timezone.utc).isoformat(timespec="seconds"), msg)
        log.write(line + "\n")
        log.flush()
        if verbose:
            print(line, flush=True)

    sels = [str(s) for s in (selections or [])] or [""]
    rows, ref_info_all, mask_all, tfs = [], {}, {}, []
    site_row = sv.site(site)
    dec_row = sv.decision(site)
    sidecar = cache.sidecar(site, work)
    # a resumed run rewrites this folder's provenance, so what an earlier pass recorded about a file
    # already on disk is carried over rather than blanked
    prev = PROV.read(out_dir / "provenance.json")
    prev_tfs = {(p.get("kind"), float(p.get("rate_hz", 0)), p.get("selection") or WHOLE): p
                     for p in (prev.get("transfer_functions") or [])}
    try:
        for rate in rates:
            t0, local, ang, applied, undecided, rec = load_local(sv, site, rate)
            decisions_applied = list(rec.get("decisions_applied") or [])
            open_decisions = list(rec.get("open") or [])
            n = len(local["Hx"])
            fs = float(rate)
            say("%s: %d samples at %g Hz from cache_%dhz, rotation %s deg, signs %s, undecided %s"
                % (site, n, fs, int(rate), ang,
                   " ".join("%s%+d" % (k, v) for k, v in sorted(applied.items())),
                   ", ".join(undecided) or "none"))
            for _d in decisions_applied:
                say("   decision=%s" % _d)
            if TR.load_series(site, work) is None:
                say("!! %s has no tail scan under %s: the transient mask is empty and the pool cannot judge "
                    "it" % (site, TR.tails_dir(work)))
            ev_site = TR.site_events(site, list(sv.sites.site), work, sv.cfg)
            ev_e = TR.e_events(site, work, sv.cfg)
            extra_mask, extra_name = _external_mask(dec_row, work, site, n, fs, say)
            for kind in kinds:
                ref_held = {}                   # the reference array, read once and used by every selection
                for sel in sels:
                    t_start = time.time()
                    tag = "" if sel in ("", WHOLE) else "%s_" % sel
                    label = "%s %s" % (kind, sel or WHOLE)
                    edi_out = out_dir / ("%s_%s_%dhz_%s%s.edi"
                                         % (site, kind, int(rate), tag, params_name))
                    key = "%s_%dhz%s" % (kind, int(rate), ("_%s" % sel) if tag else "")
                    row = dict(site=site, run=run_name, stamp=stamp, kind=kind, rate_hz=float(rate),
                               params=params_name, selection=(sel or WHOLE), remote=None, members=None,
                               n_runs=None, mask_dropped_frac=None, floor_dropped_frac=None, seconds=None,
                               peak_rss_mb=None, status=None, error=None, edi=str(edi_out), xml=None)
                    if edi_out.exists() and not redo:
                        row.update(status="exists", xml=str(edi_out.with_suffix(".xml"))
                                   if edi_out.with_suffix(".xml").exists() else None)
                        rows.append(row)
                        old = prev_tfs.get((kind, float(rate), sel or WHOLE))
                        if old:
                            tfs.append(old)
                        else:
                            # a pass over a subset of the kinds rewrites this folder's provenance, and a
                            # file it did not iterate would drop out of `tfs` and be lost from the
                            # record. What the file itself can still say is written instead of nothing.
                            tfs.append(dict(kind=kind, rate_hz=float(rate), params=params_name,
                                            selection=(sel or WHOLE), edi=str(edi_out),
                                            xml=row["xml"], tipper=EDI.has_tipper(edi_out),
                                            carried="read off the file; this pass did not make it"))
                        if key in (prev.get("mask") or {}):
                            mask_all[key] = prev["mask"][key]
                        if key in (prev.get("references") or {}):
                            ref_info_all[key] = prev["references"][key]
                        say("   %-18s exists" % label)
                        continue
                    try:
                        if kind not in ref_held:
                            ref_held[kind] = REF.load_reference(kind, site, rate, work)
                        rt0, rh, rmask, info = ref_held[kind]
                        ref_info_all[key] = {k: v for k, v in info.items() if k != "mask"}
                        if rh is not None and rt0 != t0:
                            raise ValueError("%s %s: the reference starts at %d and the record at %d"
                                             % (site, kind, rt0, t0))
                        sel_mask, sel_meta = _selection_mask(work, site, sel, t0, n, fs, say)
                        pass_mask, pass_name = extra_mask, extra_name
                        if sel_mask is not None:
                            pass_mask = sel_mask if extra_mask is None else (extra_mask & sel_mask)
                            pass_name = "%s: %d contiguous hour(s) from %s, chosen on the %g-%g s E-H " \
                                        "coherence of both recorded lines above %s%s" \
                                        % (sel, sel_meta["n_hours"], sel_meta.get("t_start"),
                                           SEL.SCORE_BAND_S[0], SEL.SCORE_BAND_S[1],
                                           sel_meta.get("threshold"),
                                           (" -- placed at random under seed %d as the control"
                                            % sel_meta["seed"]) if sel_meta["random"] else "")
                        ev_rem = []
                        if kind == "remote":
                            ev_rem = [(float(pd.Timestamp(a).timestamp()),
                                       float(pd.Timestamp(b).timestamp()))
                                      for a, b in (info.get("remote_events") or [])]
                        keep, stats = TR.build_keep(t0, local, fs, ev_site, ev_rem, ev_e,
                                                    remote_mask=(None if rmask is None else rmask[:n]),
                                                    extra_mask=pass_mask, extra_name=pass_name)
                        floor = TR.floor_dropped_frac(keep, fs)
                        h5 = scratch / ("%s_%s_%dhz%s.h5" % (site, kind, int(rate),
                                                             ("_%s" % sel) if tag else ""))
                        rid = REF.reference_station_id(kind, info,
                                                       (sv.cfg.get("observatory") or {}).get("code", ""))
                        ref_row = None
                        if kind == "remote":
                            try:
                                ref_row = sv.site(info["remote"])
                            except KeyError:
                                ref_row = None
                        _p, segs = mth5_build.write_h5(
                            h5, site, local, (None if rh is None else (rid, rh)), t0, fs,
                            sv.cfg["name"], site_row, keep=keep, reference_row=ref_row)
                        raw = scratch / ("raw_%s_%dhz%s.edi" % (kind, int(rate),
                                                                ("_%s" % sel) if tag else ""))
                        aurora_run.run_pass(h5, site, (rid or None), rate, params_name, raw)
                        bs = aurora_run.bands_for(rate)
                        plines = EDI.reference_lines(kind, info)
                        plines += EDI.mask_lines(stats, len(segs), floor)
                        plines += EDI.sign_lines(applied, undecided)
                        plines += ["decision=%s" % d for d in decisions_applied]
                        plines += ["h_rotation_deg=%s" % ang,
                                   "sample_rate_hz=%g" % fs,
                                   "parameter_set=%s (%s)" % (params_name,
                                                              ", ".join("%s=%s" % kv for kv in
                                                                        aurora_run.AURORA_PARAMS[params_name].items())),
                                   "band_file=%s, %d levels, window %d samples"
                                   % (bs.file.name, bs.levels, bs.window),
                                   "engine=Aurora %s" % aurora.__version__,
                                   "cache=%s built %s; notch: %s; signs in the cache: %s; frame in the "
                                   "cache: %s"
                                   % (sidecar.get("builder"), sidecar.get("built_utc"),
                                      sidecar.get("notch_applied", "none"),
                                      sidecar.get("signs_applied", "none"),
                                      sidecar.get("frame_applied", "none"))]
                        if int(rate) == 10:
                            plines.append("caveat_10hz=%s" % RATE.caveat(RATE.read_record(work)))
                        _o, missed, xml, xml_err = EDI.finish_edi(
                            raw, edi_out, site_row, dec_row, sv.cfg, kind, info, plines,
                            (t0, t0 + n / fs), ang, "Aurora", aurora.__version__,
                            remote_ids=([rid] if rid else []), rate=fs)
                        raw.unlink(missing_ok=True)
                        left = mth5_build.remove(h5)
                        if left:
                            say("   %s" % left)
                        res, peak = rss_mb()
                        members = ";".join("%s:%s" % (m.get("name"), m.get("weight"))
                                           for m in (info.get("members") or [])
                                           if m.get("role") != "refused")
                        row.update(remote=rid or None, members=members or None, n_runs=len(segs),
                                   mask_dropped_frac=round(stats["mask_dropped_frac"], 5),
                                   floor_dropped_frac=round(float(floor), 5),
                                   seconds=round(time.time() - t_start, 1), peak_rss_mb=round(peak, 1),
                                   status="made", xml=(str(xml) if xml else None),
                                   error=(None if not xml_err else "xml: %s" % xml_err))
                        mask_all[key] = dict(stats, runs=len(segs),
                                             floor_dropped_frac=round(float(floor), 5))
                        tfs.append(dict(kind=kind, rate_hz=float(rate), params=params_name,
                                             selection=(sel or WHOLE), selection_detail=sel_meta,
                                             edi=str(edi_out), xml=(str(xml) if xml else None),
                                             seconds=row["seconds"], peak_rss_mb=row["peak_rss_mb"],
                                             n_runs=len(segs), tipper=EDI.has_tipper(edi_out),
                                             metadata_missed=missed[:5], xml_error=xml_err))
                        say("   %-18s made in %5.1f s, %d runs, mask drops %.2f %%, peak %.0f MB%s"
                            % (label, row["seconds"], len(segs), 100 * stats["mask_dropped_frac"], peak,
                               "" if not xml_err else "; the XML was not written (%s)" % xml_err))
                    except Exception as exc:
                        row.update(status="FAILED", error="%s: %s" % (type(exc).__name__, str(exc)[:400]),
                                   seconds=round(time.time() - t_start, 1))
                        say("   %-18s FAILED: %s" % (label, row["error"]))
                        say(traceback.format_exc(limit=10))
                    rows.append(row)
                ref_held.clear()
            del local
        undec = [c for c in undecided]
        pool_file = work / "references" / ("%dhz" % int(rates[0])) / "pool.json"
        pool = json.loads(pool_file.read_text(encoding="utf-8")) if pool_file.exists() else {}
        PROV.write(out_dir / "provenance.json", sv.cfg, site_row, dec_row, ref_info_all,
                   aurora_run.bands_for(rates[0]), params_name,
                   aurora_run.AURORA_PARAMS[params_name], rates[0], run_name, stamp, tfs, mask_all,
                   sidecar, "Aurora", aurora.__version__,
                   extra_caveats=PROV.caveats(site_row, applied, undec, ref_info_all, max(rates),
                                              work_root=work, open_decisions=open_decisions),
                   decisions_applied=decisions_applied,
                   pool=pool, rot_segments=FR.parse_regimes(dec_row.get("rot_regimes")),
                   rot_drop=FR.parse_regimes(dec_row.get("rot_drop")),
                   selection=SEL.read_selection(work, site) if sels != [""] else {})
    finally:
        log.close()
        shutil.rmtree(scratch, ignore_errors=True)
    for row in rows:
        append_row(work / "survey" / "runs.csv", row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one site end to end: MTH5, Aurora, EDI and provenance")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run", default="first", help="the run name; the folder is <run>_<stamp>")
    ap.add_argument("--stamp", default="", help="the launch time in UTC as YYYYmmdd_HHMM, shared by a run")
    ap.add_argument("--kinds", nargs="+", default=[k for k in KINDS if k not in REFUSED_KINDS],
                    choices=list(KINDS))
    ap.add_argument("--rates", nargs="+", type=int, default=[1])
    ap.add_argument("--params", default=aurora_run.DEFAULT_PARAMS, choices=sorted(aurora_run.AURORA_PARAMS))
    ap.add_argument("--work-root", default="", help="override survey.yaml work_root")
    ap.add_argument("--selections", nargs="+", default=[], choices=list(SEL.TAGS) + [SEL.WHOLE],
                    help="the stretches each kind is run on: stretch control; none = the whole record")
    ap.add_argument("--redo", action="store_true",
                    help="remake a transfer function whose EDI is already on disk")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    refused = [k for k in a.kinds if k in REFUSED_KINDS]
    if refused:
        print("\n".join(REFUSED_KINDS[k] for k in refused))
        return 2
    rows = one_site(a.survey, a.site, a.run, a.kinds, a.rates, a.params, stamp=(a.stamp or None),
                    redo=a.redo, work_root=(a.work_root or None), verbose=not a.quiet,
                    selections=a.selections)
    failed = [r for r in rows if r["status"] == "FAILED"]
    print(json.dumps(dict(site=a.site, made=sum(r["status"] == "made" for r in rows),
                          exists=sum(r["status"] == "exists" for r in rows), failed=len(failed))))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
