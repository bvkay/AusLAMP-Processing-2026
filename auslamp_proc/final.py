"""One EDI per site: the xy rows from one product, the yx rows from another, the tipper from a named one.

Ported from scripts/processing/wamt_esp2026_products.py:303-447 and its fill_metadata
(D:/BEN/MTH5_Aurora_mt-io_2026), with the delivery record of wamt_best_merge.py and wamt_deliver.py.

THE MERGE. The xy product of record supplies the first impedance row (Zxx, Zxy) and the yx product the
second (Zyx, Zyy), each with its errors; the tipper comes from the product `tipper_from` names. A component
with no product of record leaves its row empty and the file says so, so a reader cannot take an empty row for
a measurement. A product on another period grid is aligned by NEAREST PERIOD within NEAREST_PCT and never
interpolated: an interpolated row is a third curve, not either product.

THE FAILURE CRITERION of the merge, which is what `check` measures: the written file must read back with its
Zxy equal to the xy source's and its Zyx equal to the yx source's at every period to READBACK_RTOL relative,
and, where the two sources carry different yx rows, its Zyx must DIFFER from the xy source's. The second is
the control that the rows came from two files; it is n/a where the two sources share the row, which happens
when one product of record is a form merged out of the other.

THE INFO BLOCK carries, as processing_parameters lines: which product each row came from (the kind, the run
and the form), the flags and the notes per component, the frame block (the frame, the declination recorded
and not applied, and the angle to turn the tensor by for geographic north with the transformation), the
splice line where a 10 Hz row is in the file, the notch line the cache recorded, the 10 Hz caveat, and the
package version and date. Every string comes from survey.yaml, sites.csv and the source products' own
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
                    "bytes", "sha256"]


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
                      float(pick.get("rate_hz") or 0.0), float(pick.get("bar_10_1000") or np.nan),
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
    """The cache's notch record and the 10 Hz caveat, carried from the source product's own header."""
    kv = base_meta.get("parameters", {}) or {}
    out = []
    if kv.get("cache"):
        out.append("cache=%s" % kv["cache"])
    if kv.get("caveat_10hz"):
        out.append("caveat_10hz=%s" % kv["caveat_10hz"])
    return out


def package_lines(cfg) -> list:
    return ["package=auslamp_proc %s, workbook 06" % auslamp_proc.__version__,
            "delivered=%s" % datetime.now(timezone.utc).date().isoformat(),
            "delivered_by=%s" % (cfg.get("author") or ""),
            "survey=%s" % (cfg.get("project") or cfg.get("name") or "")]


# ------------------------------------------------------------------ the merge

def merge(sv, site, picks: dict, out_path, tipper_from="xy", record=None, extra_lines=(),
          verbose=True) -> dict:
    """One final EDI for one site. Returns the record of what was merged and how the check scored.

    `picks` is {"xy": row, "yx": row} of the record rows chosen, each a mapping carrying `path`, `kind`,
    `form`, `run`, `stamp`, `rate_hz` and `bar_10_1000`. `tipper_from` is "xy", "yx" or a reference kind;
    where the named product carries no tipper the other pick supplies it and the file says which.
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
        zxy, zyx = src["xy"]["z"], src["yx"]["z"]
        same_grid = zxy.shape[0] == len(period) and zyx.shape[0] == len(period)
        m = ((np.isfinite(zb[:, 1, 0]) & np.isfinite(zxy[:, 1, 0]) & np.isfinite(zyx[:, 1, 0]))
             if same_grid else np.zeros(len(period), bool))
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


# ------------------------------------------------------------------ the delivery record

def write_record(out_dir, record: pd.DataFrame, readings: pd.DataFrame, merges, splice=None) -> dict:
    """PRODUCTS_OF_RECORD.csv, READINGS.csv, SPLICE.csv and FINAL_MANIFEST.csv under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in (("PRODUCTS_OF_RECORD.csv", record), ("READINGS.csv", readings),
                        ("SPLICE.csv", splice)):
        if table is None:
            continue
        path = out_dir / name
        table.to_csv(path, index=False)
        written[name] = str(path)
    rows = []
    for m in merges:
        if not m.get("written") or not m.get("path"):
            continue
        p = Path(m["path"])
        rows.append(dict(site=m["site"], role="final", component="", file=str(p), kind="", form="",
                         run="", stamp="", rate_hz=np.nan, bytes=p.stat().st_size, sha256=sha256(p)))
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
                             rate_hz=pick.get("rate_hz", np.nan), bytes=q.stat().st_size,
                             sha256=sha256(q)))
    man = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    path = out_dir / "FINAL_MANIFEST.csv"
    man.to_csv(path, index=False)
    written["FINAL_MANIFEST.csv"] = str(path)
    return dict(written=written, manifest=man)
