"""One site, end to end: the frame, the mask, the MTH5, the Aurora pass, the EDI and the provenance.

    python -m auslamp_proc.process.run --survey queensland_phase1 --site Q49 --run first \
        --kinds single remote stack obs stack_obs --rates 1 --params kaiser20_75 [--redo]

Everything lands in <work_root>/<site>/<RUN>_<stamp>/, where the stamp is the launch time in UTC of the whole
run and is passed in with --stamp so every site of one run shares a folder name. Each product writes
<site>_<kind>_<rate>hz_<params>.edi and .xml beside log.txt and provenance.json, and appends one row to
<work_root>/survey/runs.csv.

The CLI is resumable: with --redo absent, a product whose EDI is already on disk is left alone and reported
as `exists`. A product that fails is caught, its error is written into its ledger row, and the next product
runs.

A decisions.csv `keep_mask` cell naming a boolean .npy applies that selection of hours on top of the
transient mask. A cell naming a file that is not there is said loudly and the pass runs on the whole record,
because a selection silently ignored produces a whole-record product in a folder named after the selection.

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
from . import references as REF, transients as TR                            # noqa: E402

RUNS_COLUMNS = ["site", "run", "stamp", "kind", "rate_hz", "params", "remote", "members", "n_runs",
                "mask_dropped_frac", "floor_dropped_frac", "seconds", "peak_rss_mb", "status", "error",
                "edi", "xml"]


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def rss_mb() -> tuple:
    """(resident, peak resident) of this lane in MB, or (nan, nan) where psutil cannot say.

    The peak is the process's own peak since it started, so the second product of a site reports the peak of
    the first as well: the column is what the lane cost by then, not what that one product cost alone.
    """
    try:
        import psutil
        m = psutil.Process().memory_info()
        return m.rss / 2 ** 20, getattr(m, "peak_wset", m.rss) / 2 ** 20
    except Exception:
        return float("nan"), float("nan")


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
        header = not path.exists()
        pd.DataFrame([row], columns=RUNS_COLUMNS).to_csv(path, mode="a", header=header, index=False)
    finally:
        lock.unlink(missing_ok=True)


def _external_mask(dec_row, work, site, n, fs, say):
    """(the sample mask decisions.csv `keep_mask` names, its name), or (None, "").

    The cell is a path to a boolean .npy, absolute or relative to the work root, and a selection of hours
    made outside this run -- the best hours by coherence, a hand-drawn window. A cell that names a file that
    is not there is said loudly and the pass runs on the whole record: a selection silently ignored produces
    a whole-record product sitting in a folder named after the selection.

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


def load_local(sv, site, rate):
    """(t0, the signed and rotated five channels, the angle, the signs applied, the undecided channels).

    Above 1 Hz the horizontal magnetics go through the gap-edge screen first, the same one a member of the
    reference store goes through, so a site's own H and its H as somebody's reference are one record. It
    only widens what is already missing, so it cannot break a continuous stretch into pieces the 3,600 s
    run floor would then drop. The electric lines are not screened: their bursts are masked by interval
    from the tail scan, which is the mask the product's header reports.
    """
    work = Path(sv.cfg["work_root"])
    t0, arrays, meta = cache.load(site, work, rate)
    dec = sv.decision(site)
    arrays, applied, undecided = FR.apply_signs(arrays, dec)
    if int(rate) > 1:
        for c in ("Hx", "Hy"):
            arrays[c] = REF.gap_edge_screen(arrays[c], float(rate))
    regimes = FR.parse_regimes(dec.get("rot_regimes"))
    drop = FR.parse_regimes(dec.get("rot_drop"))
    turned, ang = FR.rotate_to_mean_field(arrays, regimes=regimes, drop=drop, fs=float(rate))
    local = {c: np.asarray(turned[c], float) for c in mth5_build.LOCAL_CHANNELS}
    return t0, local, ang, applied, undecided, meta


def one_site(survey_name, site, run_name, kinds, rates, params_name, stamp=None, redo=False,
             work_root=None, verbose=True) -> list:
    """Every asked-for product of one site. Returns the ledger rows it wrote."""
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

    rows, ref_info_all, mask_all, products = [], {}, {}, []
    site_row = sv.site(site)
    dec_row = sv.decision(site)
    sidecar = cache.sidecar(site, work)
    # a resumed run rewrites this folder's provenance, so what an earlier pass recorded about a product
    # already on disk is carried over rather than blanked
    prev = PROV.read(out_dir / "provenance.json")
    prev_products = {(p.get("kind"), float(p.get("rate_hz", 0))): p for p in (prev.get("products") or [])}
    try:
        for rate in rates:
            t0, local, ang, applied, undecided, _meta = load_local(sv, site, rate)
            n = len(local["Hx"])
            fs = float(rate)
            say("%s: %d samples at %g Hz from cache_%dhz, rotation %s deg, signs %s, undecided %s"
                % (site, n, fs, int(rate), ang,
                   " ".join("%s%+d" % (k, v) for k, v in sorted(applied.items())),
                   ", ".join(undecided) or "none"))
            if TR.load_series(site, work) is None:
                say("!! %s has no tail scan under %s: the transient mask is empty and the pool cannot judge "
                    "it" % (site, TR.tails_dir(work)))
            ev_site = TR.site_events(site, list(sv.sites.site), work, sv.cfg)
            ev_e = TR.e_events(site, work, sv.cfg)
            extra_mask, extra_name = _external_mask(dec_row, work, site, n, fs, say)
            for kind in kinds:
                t_start = time.time()
                edi_out = out_dir / ("%s_%s_%dhz_%s.edi" % (site, kind, int(rate), params_name))
                row = dict(site=site, run=run_name, stamp=stamp, kind=kind, rate_hz=float(rate),
                           params=params_name, remote=None, members=None, n_runs=None,
                           mask_dropped_frac=None, floor_dropped_frac=None, seconds=None,
                           peak_rss_mb=None, status=None, error=None, edi=str(edi_out), xml=None)
                if edi_out.exists() and not redo:
                    row.update(status="exists", xml=str(edi_out.with_suffix(".xml"))
                               if edi_out.with_suffix(".xml").exists() else None)
                    rows.append(row)
                    old = prev_products.get((kind, float(rate)))
                    if old:
                        products.append(old)
                        key = "%s_%dhz" % (kind, int(rate))
                        if key in (prev.get("mask") or {}):
                            mask_all[key] = prev["mask"][key]
                        if key in (prev.get("references") or {}):
                            ref_info_all[key] = prev["references"][key]
                    say("   %-10s exists" % kind)
                    continue
                try:
                    rt0, rh, rmask, info = REF.load_reference(kind, site, rate, work)
                    ref_info_all["%s_%dhz" % (kind, int(rate))] = {
                        k: v for k, v in info.items() if k != "mask"}
                    if rh is not None and rt0 != t0:
                        raise ValueError("%s %s: the reference starts at %d and the record at %d"
                                         % (site, kind, rt0, t0))
                    ev_rem = []
                    if kind == "remote":
                        ev_rem = [(float(pd.Timestamp(a).timestamp()), float(pd.Timestamp(b).timestamp()))
                                  for a, b in (info.get("remote_events") or [])]
                    keep, stats = TR.build_keep(t0, local, fs, ev_site, ev_rem, ev_e,
                                                remote_mask=(None if rmask is None else rmask[:n]),
                                                extra_mask=extra_mask, extra_name=extra_name)
                    floor = TR.floor_dropped_frac(keep, fs)
                    h5 = scratch / ("%s_%s_%dhz.h5" % (site, kind, int(rate)))
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
                    raw = scratch / ("raw_%s_%dhz.edi" % (kind, int(rate)))
                    aurora_run.run_pass(h5, site, (rid or None), rate, params_name, raw)
                    bs = aurora_run.bands_for(rate)
                    plines = EDI.reference_lines(kind, info)
                    plines += EDI.mask_lines(stats, len(segs), floor)
                    plines += EDI.sign_lines(applied, undecided)
                    plines += ["h_rotation_deg=%s" % ang,
                               "sample_rate_hz=%g" % fs,
                               "parameter_set=%s (%s)" % (params_name,
                                                          ", ".join("%s=%s" % kv for kv in
                                                                    aurora_run.AURORA_PARAMS[params_name].items())),
                               "band_file=%s, %d levels, window %d samples"
                               % (bs.file.name, bs.levels, bs.window),
                               "engine=Aurora %s" % aurora.__version__,
                               "cache=%s built %s; notch: %s; signs in the cache: %s; frame in the cache: %s"
                               % (sidecar.get("builder"), sidecar.get("built_utc"),
                                  sidecar.get("notch_applied", "none"),
                                  sidecar.get("signs_applied", "none"),
                                  sidecar.get("frame_applied", "none"))]
                    if int(rate) == 10:
                        plines.append("caveat_10hz=%s" % EDI.TEN_HZ_CAVEAT)
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
                    mask_all["%s_%dhz" % (kind, int(rate))] = dict(stats, runs=len(segs),
                                                                   floor_dropped_frac=round(float(floor), 5))
                    products.append(dict(kind=kind, rate_hz=float(rate), params=params_name,
                                         edi=str(edi_out), xml=(str(xml) if xml else None),
                                         seconds=row["seconds"], peak_rss_mb=row["peak_rss_mb"],
                                         n_runs=len(segs), tipper=EDI.has_tipper(edi_out),
                                         metadata_missed=missed[:5], xml_error=xml_err))
                    say("   %-10s made in %5.1f s, %d runs, mask drops %.2f %%, peak %.0f MB%s"
                        % (kind, row["seconds"], len(segs), 100 * stats["mask_dropped_frac"], peak,
                           "" if not xml_err else "; the XML was not written (%s)" % xml_err))
                except Exception as exc:
                    row.update(status="FAILED", error="%s: %s" % (type(exc).__name__, str(exc)[:400]),
                               seconds=round(time.time() - t_start, 1))
                    say("   %-10s FAILED: %s" % (kind, row["error"]))
                    say(traceback.format_exc(limit=10))
                rows.append(row)
            del local
        undec = [c for c in undecided]
        pool_file = work / "references" / ("%dhz" % int(rates[0])) / "pool.json"
        pool = json.loads(pool_file.read_text(encoding="utf-8")) if pool_file.exists() else {}
        PROV.write(out_dir / "provenance.json", sv.cfg, site_row, dec_row, ref_info_all,
                   aurora_run.bands_for(rates[0]), params_name,
                   aurora_run.AURORA_PARAMS[params_name], rates[0], run_name, stamp, products, mask_all,
                   sidecar, "Aurora", aurora.__version__,
                   extra_caveats=PROV.caveats(site_row, applied, undec, ref_info_all, max(rates)),
                   pool=pool, rot_segments=FR.parse_regimes(dec_row.get("rot_regimes")),
                   rot_drop=FR.parse_regimes(dec_row.get("rot_drop")))
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
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--rates", nargs="+", type=int, default=[1])
    ap.add_argument("--params", default=aurora_run.DEFAULT_PARAMS, choices=sorted(aurora_run.AURORA_PARAMS))
    ap.add_argument("--work-root", default="", help="override survey.yaml work_root")
    ap.add_argument("--redo", action="store_true", help="remake a product whose EDI is already on disk")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    rows = one_site(a.survey, a.site, a.run, a.kinds, a.rates, a.params, stamp=(a.stamp or None),
                    redo=a.redo, work_root=(a.work_root or None), verbose=not a.quiet)
    failed = [r for r in rows if r["status"] == "FAILED"]
    print(json.dumps(dict(site=a.site, made=sum(r["status"] == "made" for r in rows),
                          exists=sum(r["status"] == "exists" for r in rows), failed=len(failed))))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
