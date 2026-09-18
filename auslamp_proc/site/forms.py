"""One form of one site as a transfer function in the run layout, and the merge of a windowed row.

A form is one pass over the same site with one thing changed: a day mask, a window, a selection of hours, a
variant cache, a borrowed magnetic channel. Every form lands in the same run folder as
<site>_<form>_<kind>_<rate>hz_<params>.edi, carries the same header a workbook 03 pass writes, and adds
processing_parameters lines naming the form, the mask or window it was built on, the seed of its control and
the control it is read against. provenance.json in the run folder holds one entry per form.

The passes are the ones the package already runs: the frame and the signs from decisions.csv
(process.frame), the transient and E-burst mask (process.transients.build_keep), one Aurora run per kept
stretch of at least 3600 s (process.mth5_build), the same reference store workbook 03 built
(process.references), the same band file and parameter set (process.aurora_run). Only the record, the mask
or the local magnetics differ, so two forms are comparable.

The windowed pass (ported from wamt_run.window_slice :472-476, windowed_local :514-537, window_pass :781-807
and the rule at :447-455). Everything is sliced to the window, H included: the point of a window is that this
component's estimate sees only the days its electrode was alive, and an estimator handed a longer H than E
would be given NaN over the rest. The whole record supplies the healthy row and the tipper, the window
supplies the other row, and both windows are in the provenance. A window shorter than survey.yaml
`floors.min_window_days` is refused.

FORM_NAMES states, once, the forms workbook 04 can produce. A file in a run folder whose name carries a form
outside it is a stray left by a workbook that has since been cut, named where it is found and read by
nothing. A windowed form also records the bounds it was estimated on -- the two unix seconds, the hours, the
rule and its threshold -- in the file and in the run folder's provenance, so a later run compares them with
what the rule gives now and remakes the form where they differ instead of reusing it under a rule it never
saw.

merge_component (ported from wamt_run.merge_component :540-584) replaces exactly the two impedance rows of
one component in the whole-record file from the windowed file: the station block, the position, the tipper
and every other row carry across untouched. Both files come off the same band file, so the expected answer is
the identity, and the grid is checked because a silent half-period shift is what a band file can produce.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..process import aurora_run, edi as EDI, frame as FR, mth5_build
from ..process import provenance as PROV, references as REF, transients as TR
from ..raw.cache import CHANNELS

ROWS_OF = {"xy": ((0, 1), (0, 0)), "yx": ((1, 0), (1, 1))}
ELEMENTS = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
ELEMENT_OF = {v: k for k, v in ELEMENTS.items()}
GRID_RTOL = 1e-9
DEFAULT_MIN_WINDOW_DAYS = 0.5

# The forms workbook 04 can produce, stated once. Every name a section of that workbook passes to run_form or
# appends to its own table matches this, and a file in a run folder whose name does not is a stray: a form
# left by a workbook that has since been cut. A stray is named where it is found and read by nothing, because
# the criterion a form was judged on is what makes it readable and a cut section took its criterion with it.
FORM_NAMES = re.compile(r"^(whole|whole10|diagonal|recipe|recipe_[xy]"
                        r"|window_(xy|yx)(_control)?"
                        r"|merged_(xy|yx)"
                        r"|replace_H[xy]_[A-Za-z0-9]+"
                        r"|lender_[A-Za-z0-9]+)$")

# the keys a windowed form's bounds are compared on: the two unix seconds, the whole hours between them, the
# name of the rule that chose them and the threshold that rule read
BOUNDS_KEYS = ("t_start", "t_end", "hours", "rule", "threshold")


def is_form_name(name) -> bool:
    """True where `name` is a form workbook 04 makes. See FORM_NAMES."""
    return bool(FORM_NAMES.match(str(name or "")))


def strays(folder, sites) -> list:
    """[(path, form)] for every EDI of a run folder that its forms.csv does not name and whose form is not
    one workbook 04 makes.

    A file forms.csv names is a form of this run whatever its name parses as: <form>_<kind> is ambiguous
    where a form ends in a kind's own word, and `replace_Hx_stack` against the obs reference reads back as
    the form `replace_Hx` on the kind `stack_obs`. Every other file is read by its name through
    readings.parse_form_name, which is where the package reads a form off one; a name that does not parse as
    a form at all is a workbook 03 transfer function and is not a stray.
    """
    from ..readings import parse_form_name   # readings parses a form's file name; forms.py states the set
    folder = Path(folder)
    named = set()
    if (folder / "forms.csv").exists():
        try:
            named = {str(x) for x in pd.read_csv(folder / "forms.csv").transfer_function.astype(str)}
        except Exception:
            named = set()
    out = []
    for p in sorted(folder.glob("*.edi")):
        if p.name in named:
            continue
        meta = parse_form_name(p, list(sites))
        form = str((meta or {}).get("form") or "")
        if form and not is_form_name(form):
            out.append((p, form))
    return out


# ---------------------------------------------------------------- the bounds a windowed form was made on

def window_bounds(selection, rule=None, threshold=None) -> dict:
    """The bounds a windowed form is estimated on, as the guard compares them.

    A selection dict from process.selection -- longest_stretch or control_stretch -- carries all five keys
    under its own names. A window chosen by another rule states its own: masks.window_from_days names
    neither a rule nor a threshold, so the caller passes the rule it applied and the floor it read. An empty
    selection gives {}, which no recorded bounds can match.
    """
    if not selection or selection.get("t_start") is None:
        return {}
    ta, tb = int(selection["t_start"]), int(selection["t_end"])
    thr = selection.get("coh_min") if threshold is None else threshold
    return dict(t_start=ta, t_end=tb, hours=int(round((tb - ta) / 3600.0)),
                rule=str((selection.get("rule") if rule is None else rule) or ""),
                threshold=round(float("nan") if thr is None else float(thr), 6))


def recorded_bounds(out_dir, site, form, kind, rate, params) -> dict:
    """The bounds the run folder's provenance.json records for one form, or {} where it records none.

    A form written before the guard existed carries none, which is not the same as carrying bounds that
    match: the caller remakes it, because what it was estimated on cannot be read off the file.
    """
    want = str(Path(out_dir) / tf_name(site, form, kind, rate, params))
    for entry in (PROV.read(Path(out_dir) / "provenance.json").get("forms") or []):
        if str(entry.get("transfer_function")) == want:
            return dict(entry.get("bounds") or {})
    return {}


def bounds_note(recorded: dict, wanted: dict) -> str:
    """Empty where the form on disk was estimated on the bounds the rule now gives, else what differs.

    The returned sentence names both sets, because a form is remade on this reading and the reading is what
    says why. Bounds the rule cannot state -- no stretch was found -- differ from anything.
    """
    if not wanted:
        return "the rule finds no stretch for this row now, so no bounds can be matched"
    if not recorded:
        return ("the file on disk records no bounds, so what it was estimated on cannot be read; the rule "
                "now gives %s" % bounds_words(wanted))
    off = [k for k in BOUNDS_KEYS if str(recorded.get(k)) != str(wanted.get(k))]
    if not off:
        return ""
    return ("the file was estimated on %s and the rule now gives %s (%s differ)"
            % (bounds_words(recorded), bounds_words(wanted), ", ".join(off)))


def bounds_words(b: dict) -> str:
    """One set of bounds in words: the span in UTC, the hours, the rule and the threshold."""
    if not b:
        return "no bounds"
    return ("%s..%s, %s h, rule %s at %s"
            % (_iso(b.get("t_start", 0)), _iso(b.get("t_end", 0)), b.get("hours"), b.get("rule"),
               b.get("threshold")))


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def min_window_days(sv) -> float:
    return float((sv.cfg.get("floors") or {}).get("min_window_days", DEFAULT_MIN_WINDOW_DAYS))


def cache_dir(work_root, rate=1, variant="") -> Path:
    return Path(work_root) / ("cache_%dhz%s" % (int(rate), ("_" + variant) if variant else ""))


def load_local(sv, site, rate=1, variant="", apply_e_signs=True, decisions_applied=None):
    """(t0, the five channels decided and rotated, the angle, the signs applied, the undecided channels).

    `variant` names a cache beside the original -- ne, the arm diagonal -- which is read in its place.
    `apply_e_signs` is False for a variant whose electric channels are already signed (the NE cache), because
    signing them twice would undo the difference the variant exists for.

    Every decisions.csv decision is applied by process.frame.apply_decisions, the one place that applies
    one; the unvaried cache goes through raw.cache.load_decided, which is also where a lender's record is
    read. The strings it returns are appended to `decisions_applied` where a list is given, for the form's
    own provenance. A variant cache is read here directly and no lender is substituted into it: workbook 05
    builds a borrowed form itself, from site.replace, and says so.
    """
    from ..raw import cache as CA
    try:
        dec = sv.decision(site)
    except KeyError:
        dec = None
    if not variant:
        t0, arrays, applied_list, rec = CA.load_decided(site, sv, rate)
        applied, undecided = rec["signs"], rec["undecided"]
    else:
        z = np.load(cache_dir(sv.cfg["work_root"], rate, variant) / ("%s.npz" % site), allow_pickle=False)
        t0 = int(z["t0"][0])
        arrays = {c: np.asarray(z[c], np.float64) for c in CHANNELS}
        z.close()
        if not apply_e_signs and dec is not None:
            dec = dict(dec)
            dec["sign_ex"] = "+1"
            dec["sign_ey"] = "+1"
        try:
            srow = sv.site(site)
        except KeyError:
            srow = None
        arrays, applied_list, rec = FR.apply_decisions(arrays, dec, srow, None, float(rate))
        applied, undecided = rec["signs"], rec["undecided"]
        if FR.read_lender(dec):
            applied_list.append("h_lender=%s: the %s variant cache is read with the site's own channels; a "
                                "borrowed form is built by site.replace and named as one"
                                % (FR.read_lender(dec), variant))
    if decisions_applied is not None:
        decisions_applied.extend(applied_list)
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


def tf_name(site, form, kind, rate, params) -> str:
    return "%s_%s_%s_%dhz_%s.edi" % (site, form, kind, int(rate), params)


def record_form(out_dir, entry: dict) -> Path:
    """Append one form's entry to the run folder's provenance.json, keeping what is already there."""
    p = Path(out_dir) / "provenance.json"
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    forms = [f for f in (doc.get("forms") or []) if f.get("transfer_function") != entry.get("transfer_function")]
    forms.append(entry)
    doc["forms"] = forms
    doc.setdefault("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return p


def coverage(sv, site, kind="remote", rate=1, variant="", apply_e_signs=True, keep_extra=None,
             keep_name="", min_segment_s=None) -> dict:
    """What a cache would give Aurora against a reference, measured before any pass is run.

    The same mask the pass builds -- the transient events of the site, of the reference and of the electric
    lines, the reference's own coverage, and any extra selection -- is built here and cut into runs at the
    package's floor. `n_runs` and `days` are what Aurora would actually be handed.

    A screen that blanks scattered samples fragments the record and the floor then throws the pieces away, so
    a variant can leave nothing to pass while its source leaves the whole record. Measuring that first turns
    it into a reading with numbers instead of an exception out of the estimator.
    """
    work = Path(sv.cfg["work_root"])
    fs = float(rate)
    floor_s = float(TR.MIN_SEGMENT_S if min_segment_s is None else min_segment_s)
    t0, local, _ang, _ap, _un = load_local(sv, site, rate, variant, apply_e_signs)
    n = len(local["Hx"])
    rt0, rh, rmask, info = REF.load_reference(kind, site, rate, work)
    if rh is not None and rt0 != t0:
        raise ValueError("%s %s: the reference starts at %d and the record at %d" % (site, kind, rt0, t0))
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
                                extra_mask=(None if extra is None else extra[:n]), extra_name=keep_name)
    segs = TR.segments(keep, int(floor_s * fs))
    days = sum(L for _, L in segs) / (86400.0 * fs)
    return dict(site=site, kind=kind, rate_hz=float(rate), variant=(variant or "the original cache"),
                n_runs=len(segs), days=float(days), min_segment_s=floor_s,
                record_days=float(n / (86400.0 * fs)), kept_frac=stats["kept_frac"],
                floor_dropped_frac=TR.floor_dropped_frac(keep, fs), empty=bool(not segs))


def refusal_sentence(cov: dict, whole: dict, what="the screen") -> str:
    """The sentence a form refused by the floor carries in place of a file."""
    return ("refused: %s leaves %d run(s) of %g s against the %s (the whole record keeps %.2f d over %d "
            "run(s))" % (what, cov.get("n_runs", 0), cov.get("min_segment_s", TR.MIN_SEGMENT_S),
                         cov.get("kind", ""), whole.get("days", float("nan")), whole.get("n_runs", 0)))


def refused_row(sv, site, form, out_dir, kind, rate, params, reason, cov=None, controls=(),
                criterion="") -> dict:
    """The forms-table row of a form that was not passed, with the numbers that refused it.

    `refused` is not `FAILED`: the first is the method's own floor stating what the cache leaves, measured
    before Aurora is called, and the second is an exception out of the estimator.
    """
    row = dict(site=site, form=form, kind=kind, rate_hz=float(rate), params=params,
               transfer_function="", controls=";".join(controls), criterion=criterion, seed=None,
               status="refused", error="", reason=reason, seconds=None,
               days=(None if cov is None else cov.get("days")),
               n_runs=(None if cov is None else cov.get("n_runs")))
    record_form(out_dir, dict(row, transfer_function=str(Path(out_dir) / tf_name(site, form, kind, rate, params))))
    return row


def run_form(sv, site, form, out_dir, kind="remote", rate=1, params="kaiser20_75",
             keep_extra=None, keep_name="", window=None, variant="", apply_e_signs=True,
             local_h=None, correction=None, turn_ne=False, turn_angle_deg=None, seed=None, controls=(),
             criterion="", extra_lines=(), lender=None, bounds=None, redo=False, verbose=True) -> dict:
    """One form of one site as a transfer function. Returns the row the forms table is built from.

    `keep_extra` is a boolean over the record's samples -- a day mask, a selection of hours -- applied on top
    of the transient mask and reported on its own line. `window` is (t_start, t_end) in unix seconds and
    slices everything, H included. `variant` names a cache beside the original. `local_h` replaces Hx, Hy
    after the frame is applied, and `correction` is the matrix the tensor's H basis is right-multiplied by
    afterwards. `turn_ne` completes the arm-diagonal turn on the written file, at `turn_angle_deg`
    (the site's own atan2(-L_E, L_N); None keeps the equal-arm -45 deg). `lender` names the site a
    borrowed channel came from and is refused where it is a member of the reference this pass reads.
    `bounds` is window_bounds of the selection `window` came from and is written into the file and into the
    run folder's provenance, so a later run can read what this pass was estimated on and remake it where the
    rule has since moved.
    """
    aurora_run.silence_loggers()
    import aurora
    work = Path(sv.cfg["work_root"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edi_out = out_dir / tf_name(site, form, kind, rate, params)
    # `transfer_function` is the key forms.csv and provenance.json already carry for the file this pass wrote
    row = dict(site=site, form=form, kind=kind, rate_hz=float(rate), params=params,
               transfer_function=str(edi_out), controls=";".join(controls), criterion=criterion,
               seed=(None if seed is None else int(seed)), status=None, error=None, seconds=None,
               bounds=(dict(bounds) if bounds else None))
    if edi_out.exists() and not redo:
        # what the pass measured when it wrote this file is carried forward from the run folder's
        # provenance, so a resumed run reports the same numbers rather than blanks: without it the
        # turn-back invariants of a form already on disk would come back empty and the check that reads
        # them could not fail
        row["status"] = "exists"
        for old in (PROV.read(out_dir / "provenance.json").get("forms") or []):
            if old.get("transfer_function") == str(edi_out):
                carry = ["turn", "n_runs", "days", "record_days", "kept_frac", "mask_dropped_frac",
                         "selection_dropped_frac", "floor_dropped_frac", "rotation_deg", "remote",
                         "seconds"]
                # the bounds the caller states are the ones it just checked the file against; where it
                # states none, what the file records is carried forward rather than blanked
                if not bounds:
                    carry.append("bounds")
                for k in carry:
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
                   "form_bounds=%s" % (json.dumps(dict(bounds), sort_keys=True) if bounds
                                       else "none: the whole record"),
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
            from .centre import THETA_NE, turn_edi
            turn = turn_edi(edi_out, angle_deg=(THETA_NE if turn_angle_deg is None
                                                else float(turn_angle_deg)),
                            note="form %s" % form)
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
    """(Z, error, inside) of the windowed file on the whole-record grid, linear in log period."""
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
    grid note and the rows that were not touched, so the check can score the merge on what it left alone.
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
    stored = {f.get("transfer_function"): f for f in (PROV.read(p).get("forms") or [])}
    rows = []
    for r in forms_rows:
        r = dict(r)
        old = stored.get(r.get("transfer_function")) or {}
        for k, v in old.items():
            if r.get(k) is None and v is not None:
                r[k] = v
        rows.append(r)
    doc["forms"] = rows
    p.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return p
