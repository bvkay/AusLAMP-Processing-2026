"""The transfer functions of a run: where they are and what they hold.

find_transfer_functions reads the ledger <work_root>/survey/runs.csv and the run folders
<site>/<run>_<stamp>/ that workbook 03 writes. The ledger is append-only and a resumed run appends a second
row per transfer function, so the rows are reduced to one per (site, run, stamp, kind, rate) before anything
else. runs="latest" keeps, for each run name, only its latest stamp, so a 1 Hz run and a 10 Hz run under
different names are both kept and a re-run of one name does not count twice.

A transfer function also carries the selection of hours it was estimated on, which the file name states and
the ledger's `selection` column names. `whole` is the whole record and carries no tag; `stretch` is the
longest contiguous run of whole UTC hours in which Ex against Hy and Ey against Hx both read above 0.5 in
median squared coherence over 20-200 s, cut to its best 48 contiguous hours where it runs longer; `control`
is a run of the same length placed at random elsewhere in the record and not overlapping it (Ben, 2026-09-17:
the 10 Hz pass runs on the most coherent hours, never the whole record). process.selection holds the rule and
its values. A ledger written before that column existed has none, and every one of its rows is read as the
whole record.

read_tf applies two rules before a curve is used, both ported from scripts/qc/survey_pdf.py:55 read
(D:/BEN/MTH5_Aurora_mt-io_2026):

    the fill      the EDI empty-data value 1e32 is masked per component, together with the project's
                  no-information convention (Z = 0 with an error of 1e9), so a fill never enters a median
    the sort      periods are sorted and duplicates dropped before any interpolation: some writers emit them
                  unsorted and np.interp on an unsorted grid is silently wrong

The yx phase is folded into the first quadrant by +180 deg in rho_phase, which is what every table and panel
reads; read_tf itself returns the tensor as written.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .process import KINDS

# the EDI empty-data value, and the project's Z = 0 with an error of 1e9 for a period carrying no information
FILL = 1e30
NO_INFO_ERR = 1e9

COMPONENTS = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
OFF_DIAGONAL = ("xy", "yx")
TIPPER_COMPONENTS = {"zx": (0, 0), "zy": (0, 1)}

TF_COLUMNS = ["site", "run", "stamp", "kind", "rate_hz", "params", "selection", "path", "xml",
                   "provenance", "remote", "members", "n_runs", "seconds", "peak_rss_mb", "status",
                   "on_disk"]

# The selection of hours a transfer function was estimated on. A 1 Hz pass is the whole record and carries
# no tag in its file name; a 10 Hz pass carries one, because it runs on the most coherent hours (Ben,
# 2026-09-17). `stretch` is the longest contiguous run of hours both lines score above 0.5 over 20-200 s and
# `control` is a run of the same length drawn at random elsewhere; `whole` is the untagged whole-record pass.
WHOLE_SELECTION = "whole"
SELECTION_TAGS = ("whole", "stretch", "control")
TF_NAME = re.compile(r"^(?P<site>.+?)_(?P<kind>%s)_(?P<rate>\d+)hz_(?P<rest>.+)$"
                          % "|".join(sorted(KINDS, key=len, reverse=True)))

# the processing_parameters keys a page prints under its title, in the order it prints them
METADATA_KEYS = ("reference_kind", "reference_members", "reference_coverage", "reference_weight_rule",
                 "mask", "runs", "selection", "h_rotation_deg", "declination_deg", "sample_rate_hz",
                 "parameter_set", "band_file", "engine", "days", "caveat_10hz")


class TFData(NamedTuple):
    """One transfer function: periods sorted and unique, the tensor, its errors, the tipper and its errors."""
    period: np.ndarray
    z: np.ndarray
    z_err: np.ndarray
    t: np.ndarray
    t_err: np.ndarray
    meta: dict


# ------------------------------------------------------------------ the ledger and the run folders

def ledger(work_root) -> pd.DataFrame:
    """<work_root>/survey/runs.csv with one row per (site, run, stamp, kind, rate), the last one kept.

    The ledger is appended to, and a resumed run writes a second row per transfer function with status
    `exists`, so the raw file holds more rows than there are transfer functions.
    """
    path = Path(work_root) / "survey" / "runs.csv"
    if not path.exists():
        raise FileNotFoundError("%s has not been written; run workbook 03 first" % path)
    d = pd.read_csv(path)
    d["rate_hz"] = d.rate_hz.astype(float)
    # the last row that made the file, else the last row of any status: a resumed run's `exists` row
    # carries no reference, no members and no cost, and keeping it would blank the provenance the run wrote
    d["_made"] = (d.status.astype(str) == "made").astype(int)
    d = d.sort_values(["site", "run", "stamp", "kind", "rate_hz", "_made"], kind="stable")
    keep = d.drop_duplicates(["site", "run", "stamp", "kind", "rate_hz"], keep="last")
    return keep.drop(columns="_made").reset_index(drop=True)


def choose_runs(led: pd.DataFrame, runs="latest") -> list:
    """The (run, stamp) pairs a `runs` parameter names, newest stamp last.

    "latest" keeps the latest stamp of every run name, "all" keeps every pair, and a list of run names keeps
    the latest stamp of each of those.
    """
    pairs = sorted({(r, s) for r, s in zip(led.run, led.stamp)})
    if str(runs).strip().lower() == "all":
        return pairs
    wanted = None if str(runs).strip().lower() == "latest" else [str(x) for x in runs]
    out = []
    for name in sorted({r for r, _ in pairs}):
        if wanted is not None and name not in wanted:
            continue
        out.append((name, max(s for r, s in pairs if r == name)))
    return sorted(out)


def run_folder(work_root, site, run, stamp) -> Path:
    return Path(work_root) / str(site) / ("%s_%s" % (run, stamp))


def tf_path(work_root, site, run, stamp, kind, rate_hz, params, selection="") -> Path:
    """The EDI the run folder holds for one transfer function, built from the folder rule and not the ledger.

    The ledger records the absolute path the run wrote, which is wrong for a work root that has since moved;
    the folder rule is not. A transfer function estimated on a selection of hours carries the tag between the
    rate and the parameter set; the whole record carries none, so an untagged name is unchanged.
    """
    tag = str(selection or "").strip()
    tag = "" if tag in ("", WHOLE_SELECTION, "nan") else (tag + "_")
    return run_folder(work_root, site, run, stamp) / ("%s_%s_%dhz_%s%s.edi"
                                                      % (site, kind, int(float(rate_hz)), tag, params))


def parse_tf_name(path) -> dict:
    """{site, kind, rate_hz, selection, params} read off a transfer function's file name, or an empty dict.

    The parameter set carries an underscore of its own (kaiser20_75), so the selection is recognised as a
    known tag at the head of what follows the rate rather than by splitting on underscores.
    """
    m = TF_NAME.match(Path(path).stem)
    if not m:
        return {}
    rest = m.group("rest")
    sel = WHOLE_SELECTION
    for tag in SELECTION_TAGS:
        if tag != WHOLE_SELECTION and rest.startswith(tag + "_"):
            sel, rest = tag, rest[len(tag) + 1:]
            break
    return dict(site=m.group("site"), kind=m.group("kind"), rate_hz=float(m.group("rate")),
                selection=sel, params=rest)


def find_transfer_functions(survey, sites, runs="latest", kinds="all", rates="all", work_root=None,
                  selections="all") -> pd.DataFrame:
    """One row per transfer function of the chosen sites and runs, from the ledger and the run folders.

    `sites` is a list of site names, `kinds` is "all" or a list of the code keys, `rates` is "all" or a list
    of sample rates in Hz, and `selections` is "all" or a list of the tags. `on_disk` says whether the EDI
    the ledger names is there.

    The ledger's `selection` column names the selection of hours each row was estimated on. A ledger written
    before that column existed has none, and every one of its rows is the whole record.
    """
    work = Path(work_root or survey.cfg["work_root"])
    led = ledger(work)
    if "selection" not in led.columns:
        led = led.assign(selection=WHOLE_SELECTION)
    led["selection"] = [WHOLE_SELECTION if (pd.isna(s) or not str(s).strip()) else str(s).strip()
                        for s in led.selection]
    pairs = choose_runs(led, runs)
    keep = led[[(r, s) in pairs for r, s in zip(led.run, led.stamp)]]
    keep = keep[keep.site.isin(list(sites))]
    if not isinstance(kinds, str):
        keep = keep[keep.kind.isin([str(k) for k in kinds])]
    if not isinstance(rates, str):
        keep = keep[keep.rate_hz.isin([float(x) for x in rates])]
    if not isinstance(selections, str):
        keep = keep[keep.selection.isin([str(x) for x in selections])]
    rows = []
    for r in keep.itertuples():
        p = tf_path(work, r.site, r.run, r.stamp, r.kind, r.rate_hz, r.params, r.selection)
        folder = run_folder(work, r.site, r.run, r.stamp)
        rows.append(dict(site=r.site, run=r.run, stamp=r.stamp, kind=r.kind, rate_hz=float(r.rate_hz),
                         params=r.params, selection=r.selection, path=str(p),
                         xml=str(p.with_suffix(".xml")), provenance=str(folder / "provenance.json"),
                         remote=(None if pd.isna(r.remote) else r.remote),
                         members=(None if pd.isna(r.members) else r.members),
                         n_runs=r.n_runs, seconds=r.seconds, peak_rss_mb=r.peak_rss_mb,
                         status=r.status, on_disk=p.exists()))
    out = pd.DataFrame(rows, columns=TF_COLUMNS)
    order = {k: i for i, k in enumerate(KINDS)}
    if len(out):
        out = out.sort_values(["site", "rate_hz", "run", "kind", "selection"],
                              key=lambda c: c.map(order) if c.name == "kind" else c)
    return out.reset_index(drop=True)


def unledgered(work_root, sites, pairs) -> list:
    """Every EDI sitting in a chosen run folder that the ledger does not name.

    A transfer function on disk with no ledger row is the other half of the question the check asks of a
    ledger row with no file, and neither half is answered by reading the ledger alone.
    """
    work = Path(work_root)
    named = set()
    led = ledger(work)
    if "selection" not in led.columns:
        led = led.assign(selection=WHOLE_SELECTION)
    for r in led.itertuples():
        named.add(str(tf_path(work, r.site, r.run, r.stamp, r.kind, r.rate_hz, r.params,
                                   getattr(r, "selection", ""))).lower())
    out = []
    for site in sites:
        for run, stamp in pairs:
            folder = run_folder(work, site, run, stamp)
            for p in sorted(folder.glob("*.edi")):
                if str(p).lower() not in named:
                    out.append(str(p))
    return out


# ------------------------------------------------------------------ reading a transfer function

def _mask_component(z, e):
    """One component with the EDI fill and the no-information convention masked out.

    A zero with a finite bar between zero and 1e9 is a measurement -- a one-dimensional earth has
    Zxx = Zyy = 0 exactly, and masking it would leave the diagonal empty, which makes a frame turn of that
    period return nothing at all because the turn mixes the four elements. A zero carrying the 1e9 bar, no
    bar at all, or a bar of exactly zero is the no-information convention and is masked.

    The zero bar is the one the EDI round trip produces: the writer emits the 1e32 empty-data value for a
    row that carries nothing, and the reader hands that back as Z = 0 with an error of 0, so a blanked row
    would otherwise read as a measurement of zero at every period. No estimator produces a zero error.
    """
    z = np.asarray(z, complex).copy()
    e = np.asarray(e, float).copy()
    measured_bar = np.isfinite(e) & (e > 0) & (e < NO_INFO_ERR)
    bad = ~np.isfinite(z) | (np.abs(z) >= FILL)
    bad |= np.isfinite(e) & (e >= NO_INFO_ERR)
    bad |= (np.abs(z) == 0.0) & ~measured_bar
    z[bad] = np.nan + 1j * np.nan
    e[bad] = np.nan
    e[~np.isfinite(e) | (np.abs(e) >= FILL)] = np.nan
    return z, e


def read_tf(path) -> TFData:
    """(period, Z, Z_err, T, T_err, meta) from an EDI or an XML, the fill masked and the periods sorted.

    Every component is masked on its own: a period where Zxy is a fill and Zyx is a measurement keeps Zyx.
    Duplicated periods are dropped, the first of each kept. meta carries what the header says and what the
    sort and the mask had to do, so a check can score it.

    A file whose impedance is the empty-data fill at every period and component -- an H-only delivery, where
    the tipper is what is delivered and the two impedance rows carry no measurement -- reads back with no
    impedance at all, because the reader masks the fill and is left with nothing. It comes back here as an
    empty tensor on the file's own periods rather than as an error, so a tipper-only file reads like any
    other.
    """
    from mt_metadata.transfer_functions.core import TF

    tf = TF(fn=str(path))
    tf.read()
    p = np.asarray(tf.period, float)
    z = (np.asarray(tf.impedance.values, complex) if tf.impedance is not None
         else np.full((len(p), 2, 2), np.nan + 1j * np.nan, complex))
    try:
        ze = np.abs(np.asarray(tf.impedance_error.values, float))
    except Exception:
        ze = np.full(z.shape, np.nan)
    t = te = None
    try:
        if tf.has_tipper() and tf.tipper is not None:
            t = np.asarray(tf.tipper.values, complex)
            try:
                te = np.abs(np.asarray(tf.tipper_error.values, float))
            except Exception:
                te = np.full(t.shape, np.nan)
    except Exception:
        t = te = None

    n_source = len(p)
    was_sorted = bool(n_source < 2 or np.all(np.diff(p) > 0))
    order = np.argsort(p, kind="stable")
    p, z, ze = p[order], z[order], ze[order]
    if t is not None:
        t, te = t[order], te[order]
    first = np.concatenate(([True], np.diff(p) > 0)) if len(p) else np.zeros(0, bool)
    n_duplicate = int(len(p) - first.sum())
    p, z, ze = p[first], z[first], ze[first]
    if t is not None:
        t, te = t[first], te[first]

    for c, (i, j) in COMPONENTS.items():
        z[:, i, j], ze[:, i, j] = _mask_component(z[:, i, j], ze[:, i, j])
    if t is not None:
        for c, (i, j) in TIPPER_COMPONENTS.items():
            t[:, i, j], te[:, i, j] = _mask_component(t[:, i, j], te[:, i, j])

    meta = header(tf, path)
    meta.update(n_source_periods=n_source, n_periods=int(len(p)), source_sorted=was_sorted,
                n_duplicate_periods=n_duplicate,
                n_finite={c: int(np.isfinite(z[:, i, j]).sum()) for c, (i, j) in COMPONENTS.items()},
                has_tipper=t is not None)
    return TFData(p, z, ze, t, te, meta)


def header(tf, path) -> dict:
    """What the file's own header says: the station, the frame and every processing_parameters line."""
    st = tf.station_metadata
    lines = list(getattr(st.transfer_function, "processing_parameters", []) or [])
    kv = {}
    for ln in lines:
        if "=" in ln:
            k, _, v = str(ln).partition("=")
            kv.setdefault(k.strip(), v.strip())
    return dict(path=str(path), station=str(st.id), lines=lines, parameters=kv,
                latitude=float(st.location.latitude or 0.0), longitude=float(st.location.longitude or 0.0),
                coordinate_system=str(getattr(st.transfer_function, "coordinate_system", "") or ""),
                remote_references=list(getattr(st.transfer_function, "remote_references", []) or []))


def metadata_lines(meta: dict, keys=METADATA_KEYS, width=110) -> list:
    """The header lines a page prints under its title, one per key that the file carries."""
    kv = meta.get("parameters") or {}
    out = []
    for k in keys:
        if k in kv and str(kv[k]).strip():
            v = str(kv[k]).strip()
            out.append("%s: %s" % (k, v if len(v) <= width else v[:width - 3] + "..."))
    return out


def rho_phase(period, z, z_err, comp: str):
    """(rho in Ohm.m, its error, phase in deg, its error) for one component, the yx phase folded by +180 deg.

    rho = 0.2 T |Z|^2 with Z in mV/km/nT; the errors are propagated from the impedance error.
    """
    i, j = COMPONENTS[comp]
    zz = np.asarray(z)[:, i, j]
    ee = np.asarray(z_err)[:, i, j]
    p = np.asarray(period, float)
    rho = 0.2 * p * np.abs(zz) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = ee / np.abs(zz)
    rho_err = 2 * rho * rel
    ph = np.degrees(np.angle(zz))
    if comp == "yx":
        ph = ph + 180.0
    ph = np.where(ph > 180, ph - 360, ph)
    ph_err = np.degrees(rel)
    return rho, rho_err, ph, ph_err


def tipper_parts(tf: TFData, comp: str):
    """(real, imaginary, error) of one tipper column, or (None, None, None) where the file carries none."""
    if tf.t is None:
        return None, None, None
    i, j = TIPPER_COMPONENTS[comp]
    return np.real(tf.t[:, i, j]), np.imag(tf.t[:, i, j]), np.asarray(tf.t_err)[:, i, j]
