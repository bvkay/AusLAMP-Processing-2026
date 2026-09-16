"""One form of one site as a product in the run layout, and the merge of a windowed row into the whole.

A form is one pass over the same site with one thing changed: a day mask, a window, a selection of hours, a
variant cache, a borrowed magnetic channel. Every form lands in the same run folder as
<site>_<form>_<kind>_<rate>hz_<params>.edi, carries the same header a workbook 03 product carries, and adds
processing_parameters lines naming the form, the mask or window it was built on, the seed of its control and
the control products it is read against. provenance.json in the run folder holds one entry per form.

The passes are the ones the package already runs: the frame and the signs from decisions.csv
(process.frame), the transient and E-burst mask (process.transients.build_keep), one Aurora run per kept
stretch of at least 3600 s (process.mth5_build), the same reference store workbook 03 built
(process.references), the same band file and parameter set (process.aurora_run). Only the record, the mask
or the local magnetics differ, so two forms are comparable.

The windowed pass (ported from wamt_run.window_slice :472-476, windowed_local :514-537, window_pass :781-807
and the rule at :447-455). Everything is sliced to the window, H included: the point of a window is that this
component's estimate sees only the days its electrode was alive, and an estimator handed a longer H than E
would be given NaN over the rest. The whole record supplies the healthy row and the tipper, the window supplies the
other row, and BOTH windows are in the provenance. A window shorter than survey.yaml
`floors.min_window_days` is refused.

merge_component (ported from wamt_run.merge_component :540-584) replaces exactly the two impedance rows of
one component in the whole-record file from the windowed file: the station block, the position, the tipper
and every other row carry across untouched. Both products come off the same band file, so the expected answer
is the identity, and the grid is checked because a silent half-period shift is what a band file can produce.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import survey as SV  # noqa: F401  (the caller's Survey object type)
from ..process import aurora_run, edi as EDI, frame as FR, mth5_build
from ..process import provenance as PROV, references as REF, transients as TR
from ..raw.cache import CHANNELS

ROWS_OF = {"xy": ((0, 1), (0, 0)), "yx": ((1, 0), (1, 1))}
ELEMENTS = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
ELEMENT_OF = {v: k for k, v in ELEMENTS.items()}
GRID_RTOL = 1e-9
DEFAULT_MIN_WINDOW_DAYS = 0.5


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def min_window_days(sv) -> float:
    return float((sv.cfg.get("floors") or {}).get("min_window_days", DEFAULT_MIN_WINDOW_DAYS))


def cache_dir(work_root, rate=1, variant="") -> Path:
    return Path(work_root) / ("cache_%dhz%s" % (int(rate), ("_" + variant) if variant else ""))


def load_local(sv, site, rate=1, variant="", apply_e_signs=True):
    """(t0, the five channels signed and rotated, the angle, the signs applied, the undecided channels).

    `variant` names a cache beside the original -- ne, notched, despiked -- which is read in its place.
    `apply_e_signs` is False for a variant whose electric channels are already signed (the NE cache), because
    signing them twice would undo the difference the variant exists for.
    """
    z = np.load(cache_dir(sv.cfg["work_root"], rate, variant) / ("%s.npz" % site), allow_pickle=False)
    t0 = int(z["t0"][0])
    arrays = {c: np.asarray(z[c], np.float64) for c in CHANNELS}
    z.close()
    try:
        dec = sv.decision(site)
    except KeyError:
        dec = None
    if not apply_e_signs and dec is not None:
        dec = dict(dec)
        dec["sign_ex"] = "+1"
        dec["sign_ey"] = "+1"
    arrays, applied, undecided = FR.apply_signs(arrays, dec)
    if int(rate) > 1:
        for c in ("Hx", "Hy"):
            arrays[c] = REF.gap_edge_screen(arrays[c], float(rate))
    regimes = FR.parse_regimes(None if dec is None else dec.get("rot_regimes"))
    drop = FR.parse_regimes(None if dec is None else dec.get("rot_drop"))
    turned, ang = FR.rotate_to_mean_field(arrays, regimes=regimes, drop=drop, fs=float(rate))
    local = {c: np.asarray(turned[c], float) for c in mth5_build.LOCAL_CHANNELS}
    return t0, local, ang, applied, undecided


def mean_angle(ang) -> float:
    """One angle from what rotate_to_mean_field returned: a float, or the first regime's angle."""
    if isinstance(ang, (int, float)):
        return float(ang)
    try:
        return float(ang[0][2])
    except Exception:
        return float("nan")


def product_name(site, form, kind, rate, params) -> str:
    return "%s_%s_%s_%dhz_%s.edi" % (site, form, kind, int(rate), params)


def record_form(out_dir, entry: dict) -> Path:
    """Append one form's entry to the run folder's provenance.json, keeping what is already there."""
    p = Path(out_dir) / "provenance.json"
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    forms = [f for f in (doc.get("forms") or []) if f.get("product") != entry.get("product")]
    forms.append(entry)
    doc["forms"] = forms
    doc.setdefault("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return p


def run_form(sv, site, form, out_dir, kind="remote", rate=1, params="kaiser20_75",
             keep_extra=None, keep_name="", window=None, variant="", apply_e_signs=True,
             local_h=None, correction=None, turn_ne=False, seed=None, controls=(),
             criterion="", extra_lines=(), lender=None, redo=False, verbose=True) -> dict:
    """One form of one site as a product. Returns the row the forms table is built from.

    `keep_extra` is a boolean over the record's samples -- a day mask, a selection of hours -- applied on top
    of the transient mask and reported on its own line. `window` is (t_start, t_end) in unix seconds and
    slices everything, H included. `variant` names a cache beside the original. `local_h` replaces Hx, Hy
    after the frame is applied, and `correction` is the matrix the tensor's H basis is right-multiplied by
    afterwards. `turn_ne` completes the north-minus-east turn on the written file. `lender` names the site a
    borrowed channel came from and is refused where it is a member of the reference this pass reads.
    """
    aurora_run.silence_loggers()
    import aurora
    work = Path(sv.cfg["work_root"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edi_out = out_dir / product_name(site, form, kind, rate, params)
    row = dict(site=site, form=form, kind=kind, rate_hz=float(rate), params=params,
               product=str(edi_out), controls=";".join(controls), criterion=criterion,
               seed=(None if seed is None else int(seed)), status=None, error=None, seconds=None)
    if edi_out.exists() and not redo:
        # what the pass measured when it made this product is carried forward from the run folder's
        # provenance, so a resumed run reports the same numbers rather than blanks: without it the
        # turn-back invariants of a form already on disk would come back empty and the check that reads
        # them could not fail
        row["status"] = "exists"
        for old in (PROV.read(out_dir / "provenance.json").get("forms") or []):
            if old.get("product") == str(edi_out):
                for k in ("turn", "n_runs", "days", "record_days", "kept_frac", "mask_dropped_frac",
                          "selection_dropped_frac", "floor_dropped_frac", "rotation_deg", "remote",
                          "seconds"):
                    if old.get(k) is not None:
                        row[k] = old[k]
                break
        return row
    t_start = time.time()
    scratch = out_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        t0, local, ang, applied, undecided = load_local(sv, site, rate, variant, apply_e_signs)
        n = len(local["Hx"])
        fs = float(rate)
        if local_h is not None:
            for c in ("Hx", "Hy"):
                local[c] = np.asarray(local_h[c], float)[:n]
        rt0, rh, rmask, info = REF.load_reference(kind, site, rate, work)
        if rh is not None and rt0 != t0:
            raise ValueError("%s %s: the reference starts at %d and the record at %d" % (site, kind, rt0, t0))
        if lender:
            from .replace import lender_in_reference
            if lender_in_reference(lender, info):
                raise ValueError("the lender %s is a member of the %s reference this pass reads; a reference "
                                 "sharing a channel with the local H compares a channel with itself"
                                 % (lender, kind))
        ev_site = TR.site_events(site, list(sv.sites.site), work, sv.cfg)
        ev_e = TR.e_events(site, work, sv.cfg)
        ev_rem = []
        if kind == "remote":
            ev_rem = [(float(pd.Timestamp(a).timestamp()), float(pd.Timestamp(b).timestamp()))
                      for a, b in (info.get("remote_events") or [])]
        extra = None if keep_extra is None else np.asarray(keep_extra, bool)
        if extra is not None and len(extra) < n and fs > 1.0:
            extra = np.repeat(extra, int(round(fs)))
        if extra is not None and len(extra) < n:
            extra = np.concatenate([extra, np.zeros(n - len(extra), bool)])
        keep, stats = TR.build_keep(t0, local, fs, ev_site, ev_rem, ev_e,
                                    remote_mask=(None if rmask is None else rmask[:n]),
                                    extra_mask=(None if extra is None else extra[:n]),
                                    extra_name=keep_name)
        i0, i1, window_note = 0, n, ""
        if window is not None:
            i0, i1 = window_slice(t0, n, window, fs)
            days = (i1 - i0) / 86400.0 / fs
            floor = min_window_days(sv)
            if days < floor:
                raise ValueError("%s %s: the window is %.2f d, under the %g d floor" % (site, form, days, floor))
            window_note = ("window %s..%s (%.2f d of %.2f d); everything is sliced to it, H included"
                           % (_iso(window[0]), _iso(window[1]), days, n / 86400.0 / fs))
            local = {c: v[i0:i1] for c, v in local.items()}
            keep = keep[i0:i1].copy()
            for v in local.values():
                keep &= np.isfinite(v)
            if rh is not None:
                rh = {c: rh[c][i0:i1] for c in rh}
            t0 = int(t0 + i0 / fs)
            n = i1 - i0
        if not keep.any():
            raise ValueError("%s %s: the mask leaves nothing" % (site, form))
        floor = TR.floor_dropped_frac(keep, fs)
        h5 = scratch / ("%s_%s_%dhz.h5" % (site, form, int(rate)))
        rid = REF.reference_station_id(kind, info, (sv.cfg.get("observatory") or {}).get("code", ""))
        ref_row = None
        if kind == "remote":
            try:
                ref_row = sv.site(info["remote"])
            except KeyError:
                ref_row = None
        site_row = sv.site(site)
        dec_row = sv.decision(site)
        _p, segs = mth5_build.write_h5(h5, site, local, (None if rh is None else (rid, rh)), t0, fs,
                                       sv.cfg["name"], site_row, keep=keep, reference_row=ref_row)
        raw = scratch / ("raw_%s_%dhz.edi" % (form, int(rate)))
        aurora_run.run_pass(h5, site, (rid or None), rate, params, raw)
        bs = aurora_run.bands_for(rate)
        plines = EDI.reference_lines(kind, info)
        plines += EDI.mask_lines(stats, len(segs), floor)
        plines += EDI.sign_lines(applied, undecided)
        plines += ["form=%s" % form,
                   "form_selection=%s" % (keep_name or "none: the whole record"),
                   "form_window=%s" % (window_note or "none: the whole record"),
                   "form_controls=%s" % (", ".join(controls) or "none"),
                   "form_seed=%s" % ("none" if seed is None else str(int(seed))),
                   "form_criterion=%s" % (criterion or "none stated"),
                   "cache=%s" % cache_dir(work, rate, variant).name,
                   "h_rotation_deg=%s" % ang,
                   "sample_rate_hz=%g" % fs,
                   "parameter_set=%s (%s)" % (params, ", ".join("%s=%s" % kv for kv in
                                                                aurora_run.AURORA_PARAMS[params].items())),
                   "band_file=%s, %d levels, window %d samples" % (bs.file.name, bs.levels, bs.window),
                   "engine=Aurora %s" % aurora.__version__]
        plines += [str(x) for x in extra_lines]
        if int(rate) == 10:
            plines.append("caveat_10hz=%s" % EDI.TEN_HZ_CAVEAT)
        _o, missed, xml, xml_err = EDI.finish_edi(
            raw, edi_out, site_row, dec_row, sv.cfg, kind, info, plines,
            (t0, t0 + n / fs), ang, "Aurora", aurora.__version__,
            remote_ids=([rid] if rid else []), rate=fs)
        raw.unlink(missing_ok=True)
        mth5_build.remove(h5)
        turn = None
        if turn_ne:
            from .centre import turn_edi
            turn = turn_edi(edi_out, note="form %s" % form)
        elif correction is not None:
            from .replace import correct_edi
            turn = correct_edi(edi_out, np.asarray(correction, float), note="form %s" % form)
        row.update(status="made", seconds=round(time.time() - t_start, 1), n_runs=len(segs),
                   kept_frac=stats["kept_frac"], mask_dropped_frac=stats["mask_dropped_frac"],
                   selection_dropped_frac=stats["selection_dropped_frac"],
                   floor_dropped_frac=round(float(floor), 5),
                   days=round(float(keep.sum()) / 86400.0 / fs, 3),
                   record_days=round(n / 86400.0 / fs, 3),
                   xml=(str(xml) if xml else None), turn=turn, remote=(rid or None),
                   rotation_deg=mean_angle(ang), metadata_missed=missed[:3], xml_error=xml_err)
        if verbose:
            print("   %-22s made in %5.1f s, %2d run(s), %.2f d kept, mask drops %.1f %%"
                  % (form, row["seconds"], len(segs), n * stats["kept_frac"] / 86400.0 / fs,
                     100 * stats["mask_dropped_frac"]), flush=True)
    except Exception as exc:
        row.update(status="FAILED", error="%s: %s" % (type(exc).__name__, str(exc)[:400]),
                   seconds=round(time.time() - t_start, 1))
        if verbose:
            print("   %-22s FAILED: %s" % (form, row["error"]), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record_form(out_dir, dict(row, recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return row


# ---------------------------------------------------------------- the window

def window_slice(t0, n, window, fs=1.0):
    """(i0, i1) of one window inside this record, clipped. wamt_run.window_slice (:472-476).

    The end second is inclusive: a window written as two calendar days holds both of them.
    """
    a, b = float(window[0]), float(window[1])
    i0 = max(0, min(int(n), int(round((a - t0) * fs))))
    i1 = max(0, min(int(n), int(round((b + 1 - t0) * fs))))
    return int(i0), int(i1)


def _match(pf, pe, rtol=GRID_RTOL):
    """(indices of pe for each pf, the worst relative difference) where the two grids are the same, else None."""
    pf, pe = np.asarray(pf, float), np.asarray(pe, float)
    if not len(pf) or not len(pe):
        return None
    idx = np.array([int(np.argmin(np.abs(np.log(pe) - np.log(t)))) for t in pf])
    k = float(np.max(np.abs(pe[idx] / pf - 1.0)))
    return (idx, k) if k <= rtol else None


def _interp_to(pf, pe, ze, ee):
    """(Z, error, inside) of the windowed product on the whole-record grid, linear in log period."""
    pf, pe = np.asarray(pf, float), np.asarray(pe, float)
    inside = (pf >= pe.min()) & (pf <= pe.max())
    zi = np.full((len(pf), 2, 2), np.nan + 1j * np.nan, complex)
    ei = np.full((len(pf), 2, 2), np.nan)
    lx, lq = np.log10(pe), np.log10(pf)
    for i in range(2):
        for j in range(2):
            zi[inside, i, j] = (np.interp(lq[inside], lx, np.real(ze[:, i, j]))
                                + 1j * np.interp(lq[inside], lx, np.imag(ze[:, i, j])))
            ei[inside, i, j] = np.interp(lq[inside], lx, np.asarray(ee, float)[:, i, j])
    return zi, ei, inside


def merge_component(base_edi, win_edi, comp, out_edi=None, verbose=True) -> dict:
    """Replace one component's two impedance rows of `base_edi` from `win_edi`.

    The whole-record file is the base, so the station block, the position and the tipper carry across
    untouched and only the row the dead electrode ruined is replaced. Returns the row count changed, the
    grid note and the rows that were NOT touched, so the check can score the merge on what it left alone.
    """
    from mt_metadata.transfer_functions.core import TF
    base_edi = Path(base_edi)
    out_edi = Path(out_edi) if out_edi else base_edi
    if out_edi != base_edi:
        shutil.copyfile(base_edi, out_edi)
    tf = TF(fn=str(out_edi))
    tf.read()
    pf = np.asarray(tf.period, float)
    zf = np.array(tf.impedance.values)
    ef = np.array(tf.impedance_error.values)
    before = zf.copy()
    te = TF(fn=str(win_edi))
    te.read()
    pe = np.asarray(te.period, float)
    ze = np.array(te.impedance.values)
    ee = np.array(te.impedance_error.values)
    m = _match(pf, pe)
    if m is not None:
        idx, kmax = m
        zi, ei = ze[idx], ee[idx]
        how = "grids identical to %.1e relative" % kmax
    else:
        zi, ei, inside = _interp_to(pf, pe, ze, ee)
        how = ("grids differ (%d against %d periods): the windowed %s was interpolated in log period onto "
               "the whole-record grid and the %d period(s) outside the window's grid carry no %s"
               % (len(pf), len(pe), comp, int((~inside).sum()), comp))
    for i, j in ROWS_OF[comp]:
        zf[:, i, j] = zi[:, i, j]
        ef[:, i, j] = ei[:, i, j]
    tf.impedance = zf
    tf.impedance_error = ef
    try:
        st = tf.station_metadata
        pp = list(st.transfer_function.processing_parameters or [])
        pp.append("merged=%s rows (Z%s, Z%s) from the windowed pass %s; every other row and the tipper are "
                  "the whole record's. %s"
                  % (comp, "xx" if comp == "xy" else "yx", "xy" if comp == "xy" else "yy",
                     Path(win_edi).name, how))
        st.transfer_function.processing_parameters = pp
    except Exception:
        pass
    try:
        tf.write(fn=str(out_edi), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(out_edi), file_type="edi")
    changed = [name for name, (i, j) in ELEMENTS.items() if not _same(before[:, i, j], zf[:, i, j])]
    expected = sorted(ELEMENT_OF[c] for c in ROWS_OF[comp])
    untouched_same = all(name in expected for name in changed)
    if verbose:
        print("   merged %s from the window (%s); rows changed: %s" % (comp, how, ", ".join(changed) or "none"))
    return dict(path=str(out_edi), component=comp, how=how, rows_changed=sorted(changed),
                rows_expected=expected, untouched_unchanged=bool(untouched_same), ok=bool(untouched_same))


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool(np.array_equal(fa, fb) and np.allclose(a[fa], b[fb], rtol=0, atol=0))


def write_run_provenance(sv, site, out_dir, run_name, stamp, rate, params, kind, forms_rows) -> Path:
    """The run folder's provenance.json, with the survey, the two table rows and every form recorded."""
    import aurora
    work = Path(sv.cfg["work_root"])
    site_row = sv.site(site)
    dec_row = sv.decision(site)
    try:
        _t0, _h, _m, info = REF.load_reference(kind, site, rate, work)
    except Exception:
        info = {"kind": kind}
    from ..raw import cache as CA
    doc = PROV.write(Path(out_dir) / "provenance.json", sv.cfg, site_row, dec_row,
                     {"%s_%dhz" % (kind, int(rate)): {k: v for k, v in info.items() if k != "mask"}},
                     aurora_run.bands_for(rate), params, aurora_run.AURORA_PARAMS[params], rate,
                     run_name, stamp, [], {}, CA.sidecar(site, work), "Aurora", aurora.__version__,
                     extra_caveats=["workbook 05: every form in this folder is one pass over the same site "
                                    "with one thing changed; the forms table states the criterion and the "
                                    "control each was read against"],
                     rot_segments=FR.parse_regimes(dec_row.get("rot_regimes")),
                     rot_drop=FR.parse_regimes(dec_row.get("rot_drop")))
    p = Path(out_dir) / "provenance.json"
    # what an earlier pass measured about a form is kept where this run only found it on disk, so the
    # measurements a form was made with survive a resumed run
    stored = {f.get("product"): f for f in (PROV.read(p).get("forms") or [])}
    rows = []
    for r in forms_rows:
        r = dict(r)
        old = stored.get(r.get("product")) or {}
        for k, v in old.items():
            if r.get(k) is None and v is not None:
                r[k] = v
        rows.append(r)
    doc["forms"] = rows
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return p
