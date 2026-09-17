"""One EDI per site: the xy rows from one product, the yx rows from another, the tipper from a named one.

Ported from scripts/processing/wamt_esp2026_products.py:303-447 and its fill_metadata
(D:/BEN/MTH5_Aurora_mt-io_2026), with the delivery record of wamt_best_merge.py and wamt_deliver.py.

THE MERGE. The xy product of record supplies the first impedance row (Zxx, Zxy) and the yx product the
second (Zyx, Zyy), each with its errors; the tipper comes from the product `tipper_from` names. A component
with no product of record leaves its row empty and the file says so, so a reader cannot take an empty row for
a measurement. A product on another period grid is aligned by NEAREST PERIOD within NEAREST_PCT and never
interpolated: an interpolated row is a third curve, not either product.

THE TRIM. `keep_band` is {component: (lo, hi) in s}, the band each row is delivered over -- the chosen
product's held band by default. A value outside its component's band is dropped, and a period left carrying
neither off-diagonal element is dropped from the grid with the tipper on it, so that a delivered file carries
the measurement and nothing else. Every dropped period is named in the manifest with the band that dropped
it. Without `keep_band` every period the sources carry is written and the file says it was not trimmed.

THE CHOICE RECORD. surveys/<survey>/final_choices.csv holds one row per site and component: the product, the
periods, the join, whether the rule or the analyst chose it, a note and the date. A workbook reads it before
it proposes anything, so a re-run reproduces a delivery without choosing again, and writes back the rule's
own rows only where no analyst row stands.

THE FAILURE CRITERION of the merge, which is what `check` measures: the written file must read back with its
Zxy equal to the xy source's and its Zyx equal to the yx source's at every period to READBACK_RTOL relative,
and, where the two sources carry different yx rows, its Zyx must DIFFER from the xy source's. The second is
the control that the rows came from two files; it is n/a where the two sources share the row, which happens
when one product of record is a form merged out of the other.

THE INFO BLOCK carries, as processing_parameters lines: which product each row came from (the kind, the run
and the form), the flags and the notes per component, the frame block (the frame, the declination recorded
and not applied, and the angle to turn the tensor by for geographic north with the transformation), the
splice line where a 10 Hz row is in the file, the cache the product was built from, the 10 Hz caveat, and
the package version and date. Every string comes from survey.yaml, sites.csv and the source products' own
headers; nothing about a survey is written into this module.

THE DELIVERY RECORD. PRODUCTS_OF_RECORD.csv and READINGS.csv are the tables the choice was made on, written
beside the files; FINAL_MANIFEST.csv carries the sha256 of every final and of every product it came from, so
the delivery can be re-read without the files.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import auslamp_proc
from . import agreement as AG, products as PR, readings as RD
from .process import KIND_WORD, edi as EDI

NEAREST_PCT = 1.0                  # a product on another grid is aligned to a period within this per cent
READBACK_RTOL = 1e-9               # the written file must read back equal to its sources within this
GRID_RTOL = 1e-6                   # two grids count as the same within this
EDI_FILL = 1e32                    # the EDI empty-data value an undelivered row is written as

MANIFEST_COLUMNS = ["site", "role", "component", "file", "kind", "form", "run", "stamp", "rate_hz",
                    "n_periods", "n_dropped", "dropped_periods_s", "dropped_reason", "bytes", "sha256"]

CHOICE_COLUMNS = ["site", "component", "product", "periods_lo", "periods_hi", "join", "chosen_by", "note",
                  "date"]
CHOSEN_BY = ("rule", "analyst")


def clean(text) -> str:
    """One processing_parameters line with nothing in it the EDI reader splits on.

    The EDI reader splits a Comment on the pipe and on the newline, so a line carrying either comes back
    as several and the next read raises on the fragment. Ported from wamt_esp2026_products._txt.
    """
    return str(text).replace("|", " ").replace("\n", " ").replace("\r", " ").strip()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(cell, default=float("nan")) -> float:
    s = str(cell).strip()
    if s.lower().startswith("assume:"):
        s = s.split(":", 1)[1]
    try:
        return float(s)
    except ValueError:
        return default


def _load(path):
    """(TF, period, Z, Z error, T, T error) of one product, read through mt_metadata as written."""
    from mt_metadata.transfer_functions.core import TF
    tf = TF(fn=str(path))
    tf.read()
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.impedance.values, complex)
    ze = np.asarray(tf.impedance_error.values, float)
    has = False
    try:
        has = bool(tf.has_tipper() and tf.tipper is not None)
    except Exception:
        has = False
    t = np.asarray(tf.tipper.values, complex) if has else None
    te = np.asarray(tf.tipper_error.values, float) if has else None
    return tf, p, z, ze, t, te


def _match_index(source_period, period) -> tuple:
    """(the nearest index of `source_period` for each of `period`, whether it lands within NEAREST_PCT)."""
    pc = np.asarray(source_period, float)
    p = np.asarray(period, float)
    if not len(pc) or not len(p):
        return np.zeros(len(p), int), np.zeros(len(p), bool)
    idx = np.array([int(np.argmin(np.abs(np.log(pc) - np.log(t)))) for t in p])
    return idx, np.abs(np.log(pc[idx]) - np.log(p)) < np.log1p(NEAREST_PCT / 100.0)


def align(period, source_period, source_row, source_err):
    """One product's two-element row on another grid, by nearest period and never interpolated."""
    p, pc = np.asarray(period, float), np.asarray(source_period, float)
    if len(pc) == len(p) and np.allclose(pc, p, rtol=GRID_RTOL):
        return np.asarray(source_row), np.asarray(source_err), int(len(p)), "grids identical"
    idx = np.array([int(np.argmin(np.abs(np.log(pc) - np.log(t)))) for t in p])
    ok = np.abs(np.log(pc[idx]) - np.log(p)) < np.log1p(NEAREST_PCT / 100.0)
    row = np.full((len(p), 2), np.nan + 0j)
    err = np.full((len(p), 2), np.nan)
    row[ok] = np.asarray(source_row)[idx[ok]]
    err[ok] = np.asarray(source_err)[idx[ok]]
    return row, err, int(ok.sum()), ("grids differ (%d against %d periods): aligned by nearest period "
                                     "within %g per cent, %d matched, the rest empty"
                                     % (len(p), len(pc), NEAREST_PCT, int(ok.sum())))


def source_lines(picks: dict, tipper_from: str, record: pd.DataFrame, site) -> list:
    """The processing_parameters lines naming which product each row came from, with its flags and notes."""
    out = []
    for comp in RD.COMPONENTS:
        pick = picks.get(comp)
        rows = "Z%s, Z%s" % ("xx" if comp == "xy" else "yx", "xy" if comp == "xy" else "yy")
        if pick is None:
            out.append("%s_rows=no product of record (%s empty); see PRODUCTS_OF_RECORD.csv" % (comp, rows))
            continue
        out.append("%s_rows=%s from the %s product (%s), run %s_%s%s at %g Hz, bar %.4f over 10-1000 s; "
                   "file %s"
                   % (comp, rows, pick.get("kind_word") or KIND_WORD.get(pick["kind"], pick["kind"]),
                      pick["kind"], pick.get("run", ""), pick.get("stamp", ""),
                      (", form %s" % pick["form"]) if pick.get("form") else "",
                      float(pick.get("rate_hz") or 0.0), float(pick.get("bar") or np.nan),
                      Path(str(pick["path"])).name))
    out.append("tipper=from the %s product" % tipper_from)
    if record is not None and len(record):
        for r in record[record.site == site].itertuples():
            note, flag = str(getattr(r, "note", "") or ""), str(getattr(r, "flagged", "") or "")
            if note or flag:
                out.append("%s_note=%s%s" % (r.component, ("[%s] " % flag) if flag else "", note))
    return out


def frame_block(site_row, base_meta: dict) -> list:
    """The three frame lines, built from sites.csv and the base product's own rotation line."""
    dec = _num(site_row.declination_deg, None)
    rot = base_meta.get("parameters", {}).get("h_rotation_deg", "")
    try:
        rot_deg = float(str(rot).strip())
    except (TypeError, ValueError):
        rot_deg = None
    return EDI.frame_lines(dec, rot_deg if rot_deg is not None else "per rotation regime")


def carried_lines(base_meta: dict) -> list:
    """The cache the product was built from and the 10 Hz caveat, from the source product's own header.

    The cache line the processing wrote carries a `notch:` clause, which this package no longer has a notch
    to fill: it is dropped here rather than delivered as a statement about a method the package does not
    carry.
    """
    kv = base_meta.get("parameters", {}) or {}
    out = []
    if kv.get("cache"):
        cache = "; ".join(part for part in str(kv["cache"]).split("; ")
                          if not part.strip().lower().startswith("notch:"))
        out.append("cache=%s" % cache)
    if kv.get("caveat_10hz"):
        out.append("caveat_10hz=%s" % kv["caveat_10hz"])
    return out


def package_lines(cfg) -> list:
    return ["package=auslamp_proc %s, workbook 06" % auslamp_proc.__version__,
            "delivered=%s" % datetime.now(timezone.utc).date().isoformat(),
            "delivered_by=%s" % (cfg.get("author") or ""),
            "survey=%s" % (cfg.get("project") or cfg.get("name") or "")]


# ------------------------------------------------------------------ the merge

def trim(period, zm, em, keep_band: dict) -> tuple:
    """(the kept periods, the tensor, its errors, the dropped periods, the reason) under a per-component band.

    A value outside its component's band is dropped from that row; a period left carrying neither
    off-diagonal element is dropped from the grid. A band that would empty the file is not applied, and the
    reason says so rather than writing a file with nothing in it.
    """
    p = np.asarray(period, float)
    z, e = np.asarray(zm).copy(), np.asarray(em, float).copy()
    words = []
    for comp, row in (("xy", 0), ("yx", 1)):
        band = (keep_band or {}).get(comp)
        if not band or not all(np.isfinite([float(band[0]), float(band[1])])):
            continue
        outside = (p < float(band[0])) | (p > float(band[1]))
        z[outside, row, :] = np.nan + 0j
        e[outside, row, :] = np.nan
        words.append("%s %g-%g s" % (comp, float(band[0]), float(band[1])))
    live = np.isfinite(z[:, 0, 1]) | np.isfinite(z[:, 1, 0])
    if not words:
        return p, np.asarray(zm), np.asarray(em, float), np.zeros(0), "not trimmed: no band was given"
    if not live.any():
        return (p, np.asarray(zm), np.asarray(em, float), np.zeros(0),
                "not trimmed: the band (%s) leaves no period carrying either off-diagonal element"
                % "; ".join(words))
    return (p[live], z[live], e[live], p[~live],
            "outside the delivered band (%s), the tipper trimmed with the grid" % "; ".join(words))


def merge(sv, site, picks: dict, out_path, tipper_from="xy", record=None, extra_lines=(),
          keep_band=None, verbose=True) -> dict:
    """One final EDI for one site. Returns the record of what was merged and how the check scored.

    `picks` is {"xy": row, "yx": row} of the record rows chosen, each a mapping carrying `path`, `kind`,
    `form`, `run`, `stamp`, `rate_hz` and `bar`. `tipper_from` is "xy", "yx" or a reference kind; where the
    named product carries no tipper the other pick supplies it and the file says which. `keep_band` is
    {component: (lo, hi) in s}, the band each row is delivered over; None writes every period.
    """
    from mt_metadata.transfer_functions.core import TF

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks = {c: v for c, v in (picks or {}).items() if v and Path(str(v.get("path") or "")).exists()}
    if not picks:
        return dict(site=site, path=None, written=False, reason="no product of record on either component")
    base_comp = "xy" if "xy" in picks else "yx"
    tf0, p, z0, e0, t0, te0 = _load(picks[base_comp]["path"])
    zm = np.full_like(z0, np.nan + 0j)
    em = np.full_like(e0, np.nan)
    src, notes = {}, []
    for comp, row in (("xy", 0), ("yx", 1)):
        if comp not in picks:
            continue
        if comp == base_comp:
            tfc, pc, zc, ec, tc, tec = tf0, p, z0, e0, t0, te0
        else:
            tfc, pc, zc, ec, tc, tec = _load(picks[comp]["path"])
        rw, er, n_ok, how = align(p, pc, zc[:, row, :], ec[:, row, :])
        zm[:, row, :] = rw
        em[:, row, :] = er
        src[comp] = dict(period=pc, z=zc, t=tc, t_err=tec, n_aligned=n_ok, how=how)
        notes.append("%s_grid=%s" % (comp, how))

    n_before = int(len(p))
    p, zm, em, dropped, trim_why = trim(p, zm, em, keep_band)
    if keep_band:
        notes.append("trim=%d of %d period(s) dropped, %s" % (len(dropped), n_before, trim_why))
    else:
        notes.append("trim=every period the sources carry is delivered; the file is not trimmed to a band")

    want = tipper_from if tipper_from in picks else None
    if want is None:
        for comp, pick in picks.items():
            if str(pick.get("kind")) == str(tipper_from):
                want = comp
                break
    order = ([want] if want else []) + [c for c in ("xy", "yx") if c in picks and c != want]
    tip_from = next((c for c in order if src[c]["t"] is not None), None)

    tn = TF()
    tn.survey_metadata = tf0.survey_metadata
    tn.station_metadata = tf0.station_metadata
    tn.period = p
    if np.isfinite(zm).any():
        tn.impedance = zm
        tn.impedance_error = em
    if tip_from is not None:
        pc, tc, tec = src[tip_from]["period"], src[tip_from]["t"], src[tip_from]["t_err"]
        if len(pc) == len(p) and np.allclose(pc, p, rtol=GRID_RTOL):
            tn.tipper = tc
            if tec is not None:
                tn.tipper_error = tec
        else:
            shape = tc.shape[1:]
            tp = np.full((len(p),) + shape, np.nan + 0j)
            tpe = np.full((len(p),) + shape, np.nan)
            for i in range(shape[0]):
                rw, er, _n, _how = align(p, pc, tc[:, i, :], np.asarray(tec, float)[:, i, :])
                tp[:, i, :] = rw
                tpe[:, i, :] = er
            tn.tipper = tp
            tn.tipper_error = tpe

    site_row = sv.site(site)
    lines = source_lines(picks, (KIND_WORD.get(str(picks[tip_from]["kind"]), str(picks[tip_from]["kind"]))
                                 if tip_from else "none: no source carries a tipper"), record, site)
    lines += notes
    lines += frame_block(site_row, PR.read_tf(picks[base_comp]["path"]).meta)
    lines += carried_lines(PR.read_tf(picks[base_comp]["path"]).meta)
    lines += package_lines(sv.cfg)
    lines += [str(x) for x in extra_lines]
    try:
        st = tn.station_metadata
        st.transfer_function.processing_parameters = list(
            st.transfer_function.processing_parameters or []) + [clean(x) for x in lines]
        tn.station_metadata = st
    except Exception:
        pass
    try:
        tn.write(fn=str(out_path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tn.write(fn=str(out_path), file_type="edi")
    xml_out, xml_error = None, ""
    try:
        s = tn.survey_metadata
        s.geographic_name = EDI.XML_ID_BAD.sub(" ", str(s.geographic_name or "")).strip()
        tn.survey_metadata = s
        xml_out = out_path.with_suffix(".xml")
        tn.write(fn=str(xml_out), file_type="emtfxml")
    except Exception as exc:
        xml_out, xml_error = None, "%s: %s" % (type(exc).__name__, str(exc)[:200])

    rec = check(out_path, p, zm, picks, src)
    rec.update(site=site, path=str(out_path), written=True, xml=(str(xml_out) if xml_out else None),
               xml_error=xml_error, tipper_from=(tip_from or ""), n_periods=int(len(p)),
               n_dropped=int(len(dropped)), dropped_periods_s=dropped, dropped_reason=trim_why,
               keep_band=dict(keep_band or {}),
               xy=(picks["xy"]["kind"] if "xy" in picks else ""),
               yx=(picks["yx"]["kind"] if "yx" in picks else ""),
               xy_form=(picks["xy"].get("form", "") if "xy" in picks else ""),
               yx_form=(picks["yx"].get("form", "") if "yx" in picks else ""),
               lines=lines)
    if verbose:
        print("   %-8s %-22s xy %-11s yx %-11s tipper %-3s %s %s"
              % (site, out_path.name, rec["xy"] or "-", rec["yx"] or "-", rec["tipper_from"] or "-",
                 "PASS" if rec["ok"] else "FAIL", rec["control"]), flush=True)
    return rec


def check(out_path, period, zm, picks: dict, src: dict) -> dict:
    """The read-back criterion and the two-source control on one written final."""
    _tb, pb, zb, _eb, _tb2, _te = _load(out_path)
    ok = bool(len(pb) == len(period) and np.allclose(pb, period, rtol=GRID_RTOL))
    worst = 0.0
    for comp, row in (("xy", 0), ("yx", 1)):
        if comp not in picks or not ok:
            continue
        a, c = zb[:, row, :], zm[:, row, :]
        m = np.isfinite(c)
        if not m.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(a[m] - c[m]) / np.maximum(np.abs(c[m]), 1e-30)
        worst = max(worst, float(np.nanmax(rel)))
        ok = ok and bool(np.allclose(a[m], c[m], rtol=READBACK_RTOL, atol=0))
    control = "control n/a (one component has no product of record)"
    if "xy" in picks and "yx" in picks:
        # the sources are indexed by nearest period, not by position: a trimmed file is shorter than the
        # products it came from, and a positional control on it would silently report n/a
        ixy, okxy = _match_index(src["xy"]["period"], period)
        iyx, okyx = _match_index(src["yx"]["period"], period)
        zxy, zyx = src["xy"]["z"][ixy], src["yx"]["z"][iyx]
        m = (okxy & okyx & np.isfinite(zb[:, 1, 0]) & np.isfinite(zxy[:, 1, 0])
             & np.isfinite(zyx[:, 1, 0]))
        if not m.any():
            control = "control n/a (the two sources share no period carrying both yx rows)"
        elif np.allclose(zxy[m, 1, 0], zyx[m, 1, 0], rtol=GRID_RTOL, atol=0):
            control = "control n/a (the two source files share the yx row)"
        else:
            differs = not np.allclose(zb[m, 1, 0], zxy[m, 1, 0], rtol=GRID_RTOL, atol=0)
            control = ("control PASS (the yx row follows the yx source and differs from the xy source's)"
                       if differs else "control FAIL (the yx row equals the xy source's)")
            ok = ok and differs
    return dict(ok=bool(ok), worst_readback_relative=worst, control=control,
                control_ok=("FAIL" not in control))


# ------------------------------------------------------------------ the tipper-only delivery

def tipper_only(sv, site, product_path, out_path, kind="", refusal=None, record=None,
                extra_lines=()) -> dict:
    """A site with no deliverable impedance delivering its tipper, with the frame block on the file.

    `site.deliver.tipper_only` blanks the impedance rows and writes the two INFO lines that say what the
    file is; the frame block and the package lines are added here so that a tipper-only file carries the
    same header as a merged one.
    """
    from .site import deliver as DL

    out_path = Path(out_path)
    base_meta = PR.read_tf(product_path).meta
    lines = (frame_block(sv.site(site), base_meta) + carried_lines(base_meta) + package_lines(sv.cfg)
             + [str(x) for x in extra_lines])
    if record is not None and len(record):
        for r in record[record.site == site].itertuples():
            if str(getattr(r, "note", "") or ""):
                lines.append("%s_note=%s" % (r.component, r.note))
    out = DL.tipper_only(product_path, out_path, kind=kind, refusal=refusal)
    if not out.get("written"):
        return dict(site=site, path=None, written=False, ok=False, reason=out.get("reason", ""))
    from mt_metadata.transfer_functions.core import TF
    tf = TF(fn=str(out_path))
    tf.read()
    try:
        st = tf.station_metadata
        st.transfer_function.processing_parameters = list(
            st.transfer_function.processing_parameters or []) + [clean(x) for x in lines]
        tf.station_metadata = st
        tf.write(fn=str(out_path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except Exception:
        pass
    back = PR.read_tf(out_path)
    return dict(site=site, path=str(out_path), written=True, ok=bool(back.t is not None),
                tipper_from=kind, control="control n/a (a tipper-only delivery has no second source)",
                reason=out.get("reason", ""), n_periods=int(len(back.period)),
                impedance_finite=out.get("impedance_finite", {}), lines=lines)


# ------------------------------------------------------------------ the optional common grid

def resample(final_path, out_path, grid=None) -> dict:
    """The final on the ten-per-decade grid T = 10^(k/10) from 3.16 s to 50,119 s, beside the delivered file.

    Ported from scripts/qc/edi_resample.py:69-71. An optional output and never the delivered file itself:
    every value on it has been interpolated, and the delivered file is the one the estimator wrote.
    """
    from mt_metadata.transfer_functions.core import TF

    grid = AG.GRID if grid is None else np.asarray(grid, float)
    src = PR.read_tf(final_path)
    tf0 = TF(fn=str(final_path))
    tf0.read()
    on = AG.on_grid(src, src, periods=grid)
    tn = TF()
    tn.survey_metadata = tf0.survey_metadata
    tn.station_metadata = tf0.station_metadata
    tn.period = grid
    tn.impedance = on.z
    tn.impedance_error = on.z_err
    if on.t is not None:
        tn.tipper = on.t
        tn.tipper_error = on.t_err
    try:
        st = tn.station_metadata
        st.transfer_function.processing_parameters = list(
            st.transfer_function.processing_parameters or []) + [
            clean("resampled=ten periods a decade, T = 10^(k/10) from %.2f to %.0f s, linear in log "
                  "period of the log impedance magnitude and the unwrapped phase, no extrapolation and no "
                  "bridging of a hole wider than 0.30 decades; an optional output, not the delivered file"
                  % (grid[0], grid[-1]))]
        tn.station_metadata = st
    except Exception:
        pass
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tn.write(fn=str(out_path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tn.write(fn=str(out_path), file_type="edi")
    back = PR.read_tf(out_path)
    return dict(path=str(out_path), n_periods=int(len(back.period)),
                n_finite={c: int(np.isfinite(back.z[:, i, j]).sum())
                          for c, (i, j) in PR.COMPONENTS.items()})


# ------------------------------------------------------------------ the choice record

def choices_path(sv) -> Path:
    """surveys/<survey>/final_choices.csv, beside the survey's own tables."""
    return Path(sv.folder) / "final_choices.csv"


def read_choices(path) -> pd.DataFrame:
    """final_choices.csv, or an empty frame with its columns where the file is not there yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CHOICE_COLUMNS)
    d = pd.read_csv(path, dtype={"product": str, "join": str, "chosen_by": str, "note": str,
                                 "date": str})
    for c in CHOICE_COLUMNS:
        if c not in d.columns:
            d[c] = ""
    d["chosen_by"] = [str(v).strip().lower() if str(v).strip().lower() in CHOSEN_BY else "rule"
                      for v in d.chosen_by]
    return d[CHOICE_COLUMNS]


def choice_row(site, component, product, periods=None, join=None, chosen_by="rule", note="") -> dict:
    """One final_choices.csv row. `periods` is (lo, hi) in s or None, `join` a period in s or None."""
    lo, hi = (periods if periods is not None else (np.nan, np.nan))
    return dict(site=str(site), component=str(component), product=str(product),
                periods_lo=(float(lo) if lo is not None and np.isfinite(float(lo)) else np.nan),
                periods_hi=(float(hi) if hi is not None and np.isfinite(float(hi)) else np.nan),
                join=("" if join is None else ("%g" % float(join))), chosen_by=str(chosen_by),
                note=str(note), date=datetime.now(timezone.utc).date().isoformat())


def write_choices(path, rows, keep_analyst=True) -> dict:
    """Write final_choices.csv, an analyst's row never overwritten by the rule's.

    A row already in the file with `chosen_by` analyst is kept as it stands; every other (site, component)
    is replaced by the row given. Returns the counts, so a workbook can say what it wrote and what it left.
    """
    path = Path(path)
    old = read_choices(path)
    new = pd.DataFrame(list(rows), columns=CHOICE_COLUMNS)
    kept = old[old.chosen_by == "analyst"] if (keep_analyst and len(old)) else old.iloc[0:0]
    locked = {(r.site, r.component) for r in kept.itertuples()}
    fresh = new[[(r.site, r.component) not in locked for r in new.itertuples()]] if len(new) else new
    untouched = old[[(r.site, r.component) not in set(zip(fresh.site, fresh.component))
                     and (r.site, r.component) not in locked for r in old.itertuples()]] \
        if len(old) else old
    out = pd.concat([kept, untouched, fresh], ignore_index=True).sort_values(["site", "component"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out[CHOICE_COLUMNS].to_csv(path, index=False)
    return dict(path=str(path), written=int(len(fresh)), analyst_kept=int(len(kept)),
                rows=int(len(out)), table=out[CHOICE_COLUMNS].reset_index(drop=True))


def manifest_check(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per delivered file: whether it is there, reads as a transfer function, and matches its sha256.

    The three are read from the files themselves and not from the table that names them, so a manifest row
    written against a file that has since changed is what this reports.
    """
    rows = []
    for r in manifest[manifest.role == "final"].itertuples():
        p = Path(str(r.file))
        exists = p.exists()
        digest = sha256(p) if exists else ""
        readable = False
        if exists:
            try:
                readable = bool(len(PR.read_tf(p).period))
            except Exception:
                readable = False
        rows.append(dict(site=r.site, file=str(p), exists=exists, readable=readable,
                         sha256_matches=bool(exists and digest == str(r.sha256)),
                         sha256=digest, sha256_recorded=str(r.sha256)))
    return pd.DataFrame(rows, columns=["site", "file", "exists", "readable", "sha256_matches",
                                       "sha256", "sha256_recorded"])


# ------------------------------------------------------------------ the delivery record

def _carry_other_sites(path, table: pd.DataFrame, sites) -> pd.DataFrame:
    """The table with the rows of `sites` replaced and every other site's rows of the old file kept.

    The workbook delivers one site at a time, and the four tables are the survey's and not the run's: a
    one-site run that wrote its own rows alone would leave the survey with a table of one site.
    """
    if sites is None or not Path(path).exists():
        return table
    try:
        old = pd.read_csv(path)
    except Exception:
        return table
    if "site" not in old.columns:
        return table
    keep = old[~old.site.isin(list(sites))]
    if not len(keep):
        return table
    out = pd.concat([keep, table], ignore_index=True, sort=False)
    return out[[c for c in table.columns if c in out.columns]
               + [c for c in out.columns if c not in table.columns]]


def write_record(out_dir, record: pd.DataFrame, readings: pd.DataFrame, merges, splice=None,
                 sites=None) -> dict:
    """PRODUCTS_OF_RECORD.csv, READINGS.csv, SPLICE.csv and FINAL_MANIFEST.csv under `out_dir`.

    `sites` are the sites this run delivered: their rows replace the old ones and every other site's rows
    stay, so a one-site run adds to the survey's tables rather than replacing them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in (("PRODUCTS_OF_RECORD.csv", record), ("READINGS.csv", readings),
                        ("SPLICE.csv", splice)):
        if table is None:
            continue
        path = out_dir / name
        _carry_other_sites(path, table, sites).to_csv(path, index=False)
        written[name] = str(path)
    rows = []
    for m in merges:
        if not m.get("written") or not m.get("path"):
            continue
        # the delivered file is the spliced one where a 10 Hz row was joined and the merge itself where
        # none was: a manifest naming the unspliced base would carry the sha256 of a file nobody delivers
        p = Path(m.get("delivered") or m["path"])
        gone = list(np.asarray(m.get("dropped_periods_s", []), float))
        rows.append(dict(site=m["site"], role="final", component="", file=str(p), kind="", form="",
                         run="", stamp="", rate_hz=np.nan,
                         n_periods=m.get("n_delivered_periods", m.get("n_periods", np.nan)),
                         n_dropped=int(m.get("n_dropped", 0) or 0),
                         dropped_periods_s=" ".join("%.4g" % t for t in gone),
                         dropped_reason=str(m.get("dropped_reason", "") or ""),
                         bytes=p.stat().st_size, sha256=sha256(p)))
        b = Path(m["path"])
        if b != p and b.exists():
            rows.append(dict(site=m["site"], role="base", component="", file=str(b), kind="", form="",
                             run="", stamp="", rate_hz=1.0, n_periods=m.get("n_periods", np.nan),
                             n_dropped=int(m.get("n_dropped", 0) or 0),
                             dropped_periods_s=" ".join("%.4g" % t for t in gone),
                             dropped_reason=str(m.get("dropped_reason", "") or ""),
                             bytes=b.stat().st_size, sha256=sha256(b)))
        for comp in RD.COMPONENTS + ("tipper",):
            pick = (m.get("picks") or {}).get(comp)
            if not pick:
                continue
            q = Path(str(pick["path"]))
            if not q.exists():
                continue
            rows.append(dict(site=m["site"], role="source", component=comp, file=str(q),
                             kind=pick.get("kind", ""), form=pick.get("form", ""),
                             run=pick.get("run", ""), stamp=pick.get("stamp", ""),
                             rate_hz=pick.get("rate_hz", np.nan), n_periods=np.nan, n_dropped=0,
                             dropped_periods_s="", dropped_reason="", bytes=q.stat().st_size,
                             sha256=sha256(q)))
    man = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    path = out_dir / "FINAL_MANIFEST.csv"
    man = _carry_other_sites(path, man, sites)
    man.to_csv(path, index=False)
    written["FINAL_MANIFEST.csv"] = str(path)
    return dict(written=written, manifest=man)
