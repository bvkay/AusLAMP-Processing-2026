"""The products of a run: where they are, what they hold, and the comparison sources declared beside them.

find_products reads the ledger <work_root>/survey/runs.csv and the run folders <site>/<run>_<stamp>/ that
workbook 03 writes. The ledger is append-only and a resumed run appends a second row per product, so the rows
are reduced to one per (site, run, stamp, kind, rate) before anything else. runs="latest" keeps, for each run
name, only its latest stamp, so a 1 Hz run and a 10 Hz run under different names are both kept and a re-run of
one name does not count twice.

read_tf applies two rules before a curve is used, both ported from scripts/qc/survey_pdf.py:55 read
(D:/BEN/MTH5_Aurora_mt-io_2026):

    the fill      the EDI empty-data value 1e32 is masked PER COMPONENT, together with the project's
                  no-information convention (Z = 0 with an error of 1e9), so a fill never enters a median
    the sort      periods are sorted and duplicates dropped before any interpolation: some writers emit them
                  unsorted and np.interp on an unsorted grid is silently wrong

The yx phase is folded into the first quadrant by +180 deg in rho_phase, which is what every table and panel
reads; read_tf itself returns the tensor as written.

comparison_sources reads the survey.yaml blocks earlier_processing and release_tensors. Every source declares
a frame and a note and is refused without them. The frames and what is done with each:

    geomagnetic   the mean-field frame, ours: compared as it is
    geographic    true north: turned by the site's declination_deg into our frame (Z' = R Z R^T, T' = T R^T,
                  R = [[cos, sin], [-sin, cos]] at +declination, the inverse of the to_geographic_north_deg
                  angle every product carries); the comparison moves, never our products
    instrument    as laid: compared as it is, with the note printed beside the table

A source may declare rho_factor, the multiple its apparent resistivity is out by; it is applied to the
comparison as sqrt(rho_factor) on Z and stated in every figure title and table that carries it.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .process import KINDS, KIND_WORD
from .process import frame as FR

# the EDI empty-data value, and the project's Z = 0 with an error of 1e9 for a period carrying no information
FILL = 1e30
NO_INFO_ERR = 1e9

COMPONENTS = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
OFF_DIAGONAL = ("xy", "yx")
TIPPER_COMPONENTS = {"zx": (0, 0), "zy": (0, 1)}

PRODUCT_COLUMNS = ["site", "run", "stamp", "kind", "rate_hz", "params", "path", "xml", "provenance",
                   "remote", "members", "n_runs", "seconds", "peak_rss_mb", "status", "on_disk"]

# the processing_parameters keys a page prints under its title, in the order it prints them
METADATA_KEYS = ("reference_kind", "reference_members", "reference_coverage", "reference_weight_rule",
                 "mask", "runs", "selection", "h_rotation_deg", "declination_deg", "sample_rate_hz",
                 "parameter_set", "band_file", "engine", "days", "caveat_10hz")

# the two survey.yaml keys that hold one source each, plus `comparisons:`, a mapping of any further named
# blocks of the same shape
SOURCE_BLOCKS = ("earlier_processing", "release_tensors")
SOURCE_MAPPING = "comparisons"
FRAMES = ("geomagnetic", "geographic", "instrument")


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

    The ledger is appended to, and a resumed run writes a second row per product with status `exists`, so the
    raw file holds more rows than there are products.
    """
    path = Path(work_root) / "survey" / "runs.csv"
    if not path.exists():
        raise FileNotFoundError("%s has not been written; run workbook 03 first" % path)
    d = pd.read_csv(path)
    d["rate_hz"] = d.rate_hz.astype(float)
    # the last row that made the product, else the last row of any status: a resumed run's `exists` row
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


def product_path(work_root, site, run, stamp, kind, rate_hz, params) -> Path:
    """The EDI the run folder holds for one product, built from the folder rule rather than from the ledger.

    The ledger records the absolute path the run wrote, which is wrong for a work root that has since moved;
    the folder rule is not.
    """
    return run_folder(work_root, site, run, stamp) / ("%s_%s_%dhz_%s.edi"
                                                      % (site, kind, int(float(rate_hz)), params))


def find_products(survey, sites, runs="latest", kinds="all", rates="all", work_root=None) -> pd.DataFrame:
    """One row per product of the chosen sites and runs, from the ledger and the run folders.

    `sites` is a list of site names, `kinds` is "all" or a list of the code keys, `rates` is "all" or a list
    of sample rates in Hz. `on_disk` says whether the EDI the ledger names is there.
    """
    work = Path(work_root or survey.cfg["work_root"])
    led = ledger(work)
    pairs = choose_runs(led, runs)
    keep = led[[(r, s) in pairs for r, s in zip(led.run, led.stamp)]]
    keep = keep[keep.site.isin(list(sites))]
    if not isinstance(kinds, str):
        keep = keep[keep.kind.isin([str(k) for k in kinds])]
    if not isinstance(rates, str):
        keep = keep[keep.rate_hz.isin([float(x) for x in rates])]
    rows = []
    for r in keep.itertuples():
        p = product_path(work, r.site, r.run, r.stamp, r.kind, r.rate_hz, r.params)
        folder = run_folder(work, r.site, r.run, r.stamp)
        rows.append(dict(site=r.site, run=r.run, stamp=r.stamp, kind=r.kind, rate_hz=float(r.rate_hz),
                         params=r.params, path=str(p), xml=str(p.with_suffix(".xml")),
                         provenance=str(folder / "provenance.json"),
                         remote=(None if pd.isna(r.remote) else r.remote),
                         members=(None if pd.isna(r.members) else r.members),
                         n_runs=r.n_runs, seconds=r.seconds, peak_rss_mb=r.peak_rss_mb,
                         status=r.status, on_disk=p.exists()))
    out = pd.DataFrame(rows, columns=PRODUCT_COLUMNS)
    order = {k: i for i, k in enumerate(KINDS)}
    if len(out):
        out = out.sort_values(["site", "rate_hz", "run", "kind"],
                              key=lambda c: c.map(order) if c.name == "kind" else c)
    return out.reset_index(drop=True)


def unledgered(work_root, sites, pairs) -> list:
    """Every EDI sitting in a chosen run folder that the ledger does not name.

    A product on disk with no ledger row is the other half of the same question the check asks of a ledger row
    with no file, and neither is answered by reading the ledger alone.
    """
    work = Path(work_root)
    named = set()
    led = ledger(work)
    for r in led.itertuples():
        named.add(str(product_path(work, r.site, r.run, r.stamp, r.kind, r.rate_hz, r.params)).lower())
    out = []
    for site in sites:
        for run, stamp in pairs:
            folder = run_folder(work, site, run, stamp)
            for p in sorted(folder.glob("*.edi")):
                if str(p).lower() not in named:
                    out.append(str(p))
    return out


def run_provenance(path) -> dict:
    """provenance.json of a run folder, or an empty dict."""
    import json
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ reading a transfer function

def _mask_component(z, e):
    """One component with the EDI fill and the no-information convention masked out.

    A zero with a finite bar under 1e9 is a measurement -- a one-dimensional earth has Zxx = Zyy = 0 exactly,
    and masking it would leave the diagonal empty, which makes a frame turn of that period return nothing at
    all because the turn mixes the four elements. A zero carrying the 1e9 bar,
    or no bar at all, is the no-information convention and is masked.
    """
    z = np.asarray(z, complex).copy()
    e = np.asarray(e, float).copy()
    measured_bar = np.isfinite(e) & (e < NO_INFO_ERR)
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
    the tipper is the product and the two rows carry no measurement -- reads back with no impedance at all,
    because the reader masks the fill and is left with nothing. It comes back here as an empty tensor on the
    file's own periods rather than as an error, so a tipper-only product can be read like any other.
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


# ------------------------------------------------------------------ the comparison sources

def _stem_candidates(site: str) -> list:
    """The file stems a site may be delivered under, most specific first.

    Two rules are in use in the folders this package compares against: the leading zero of a three-digit site
    number is dropped (VIC001 -> VIC01) and a trailing N is dropped (Q79N -> Q79).
    """
    s = str(site)
    m = re.match(r"^([A-Za-z]+)(\d+)(.*)$", s)
    if not m:
        return [s]
    head, digits, tail = m.groups()
    numbers = [digits]
    while numbers[-1].startswith("0") and len(numbers[-1]) > 1:      # leading zeros only, never a real digit
        numbers.append(numbers[-1][1:])
    tails = [tail] + ([tail[:-1]] if tail.upper().endswith("N") else [])
    out = ["%s%s%s" % (head, n, t) for t in tails for n in numbers]
    return list(dict.fromkeys(out))


def _folder_index(folder: Path) -> dict:
    return {p.stem.lower(): p for p in sorted(Path(folder).glob("*.edi"))}


def comparison_sources(survey, names="all") -> list:
    """The comparison sources survey.yaml declares, each with its frame and its note.

    A block without a `frame:` or a `note:` is returned with `error` set and the workbook refuses it. A
    comparison whose frame is not declared cannot be turned into ours, and a tensor in an unknown frame drawn
    on our axes is a different object on the same picture.
    """
    if isinstance(names, str) and str(names).strip().lower() == "none":
        return []
    blocks = [(b, survey.cfg.get(b) or {}) for b in SOURCE_BLOCKS]
    blocks += sorted((survey.cfg.get(SOURCE_MAPPING) or {}).items())
    out = []
    for block, spec in blocks:
        if not spec or not spec.get("folder"):
            continue
        if not isinstance(names, str) and block not in list(names):
            continue
        src = dict(name=block, folder=str(spec["folder"]), frame=str(spec.get("frame") or ""),
                   note=str(spec.get("note") or spec.get("release_note") or ""),
                   rho_factor=float(spec.get("rho_factor") or 1.0),
                   layout=str(spec.get("layout") or "<site>.edi"),
                   params=str(spec.get("params") or ""),
                   kind_map=dict(spec.get("kind_map") or {}),
                   exceptions={str(k): str(v) for k, v in (spec.get("exceptions") or {}).items()},
                   error="")
        why = []
        if src["frame"] not in FRAMES:
            why.append("no frame: declaration (one of %s)" % ", ".join(FRAMES))
        if not src["note"]:
            why.append("no note: declaration")
        if not Path(src["folder"]).exists():
            why.append("the folder is not there: %s" % src["folder"])
        src["error"] = "; ".join(why)
        out.append(src)
    return out


def comparison_paths(source: dict, site: str) -> dict:
    """{our kind key: the file} for one site of one source, over the kinds the source actually delivers.

    A layout holding <kind> is a per-kind tree, one folder per reference kind and parameter set; a layout
    without it is one file per site and the kind is "".
    """
    folder = Path(source["folder"])
    if not folder.exists():
        return {}
    layout = source["layout"]
    if "<kind>" in layout:
        out = {}
        for ours in KINDS:
            theirs = (source["kind_map"] or {}).get(ours)
            if not theirs:
                continue
            rel = layout.replace("<kind>", theirs).replace("<params>", source["params"])
            sub = folder / Path(rel).parent
            if not sub.exists():
                continue
            idx = _folder_index(sub)
            for cand in _stem_candidates(site):
                if cand.lower() in idx:
                    out[ours] = idx[cand.lower()]
                    break
        return out
    idx = _folder_index(folder)
    forced = (source.get("exceptions") or {}).get(str(site))
    if forced is not None:
        if not str(forced).strip():
            return {}
        return {"": idx[forced.lower()]} if forced.lower() in idx else {}
    for cand in _stem_candidates(site):
        if cand.lower() in idx:
            return {"": idx[cand.lower()]}
    return {}


def turn_to_our_frame(tf: TFData, declination_deg: float) -> TFData:
    """A tensor in geographic north turned into our geomagnetic frame by +declination.

    The angle is the inverse of the to_geographic_north_deg = -declination line every product carries, so the
    comparison moves and our products never do. The errors are turned in quadrature over |R|.
    """
    d = float(declination_deg)
    # a turn mixes all four elements, so a period missing one of them cannot be turned at all: those periods
    # come back empty rather than as the NaN one masked element would spread over the other three
    whole = np.all(np.isfinite(tf.z.reshape(len(tf.period), -1)), axis=1)
    z, t = FR.turn_tensor(np.where(whole[:, None, None], tf.z, 0.0), tf.t, d)
    z = np.where(whole[:, None, None], z, np.nan + 1j * np.nan)
    r = np.abs(FR.rotation_matrix(d))
    ze = np.sqrt(np.einsum("ij,njk,lk->nil", r ** 2, np.nan_to_num(tf.z_err) ** 2, r ** 2))
    ze = np.where(np.isfinite(z), ze, np.nan)
    te = None
    if tf.t_err is not None:
        flat = np.asarray(tf.t_err).reshape(np.asarray(tf.t_err).shape[0], -1)
        te = np.sqrt((np.nan_to_num(flat) ** 2) @ (r ** 2).T).reshape(np.asarray(tf.t_err).shape)
    meta = dict(tf.meta, turned_deg=round(d, 4), n_turned=int(whole.sum()),
                n_not_turned=int((~whole).sum()),
                frame_note="turned by %+.3f deg from geographic north into our geomagnetic frame" % d)
    return TFData(tf.period, z, ze, t, te, meta)


def scale_rho(tf: TFData, rho_factor: float) -> TFData:
    """A comparison whose apparent resistivity is a known multiple out, corrected on Z by sqrt(rho_factor)."""
    f = float(rho_factor)
    if not np.isfinite(f) or f == 1.0:
        return tf
    s = np.sqrt(f)
    meta = dict(tf.meta, rho_factor=f,
                rho_factor_note="rho multiplied by %.3f (Z by %.4f) as survey.yaml declares" % (f, s))
    return TFData(tf.period, tf.z * s, tf.z_err * s, tf.t, tf.t_err, meta)


def load_comparison(source: dict, site: str, declination_deg=None) -> dict:
    """{kind: TFData in OUR frame} for one site of one source, turned and scaled as the source declares.

    A geographic source is turned by the site's declination; an instrument or geomagnetic source is compared
    as it is. A declared rho_factor is applied to every curve of the source.
    """
    out = {}
    for kind, path in comparison_paths(source, site).items():
        tf = read_tf(path)
        if source["frame"] == "geographic":
            if declination_deg is None or not np.isfinite(float(declination_deg)):
                continue
            tf = turn_to_our_frame(tf, float(declination_deg))
        tf = scale_rho(tf, source.get("rho_factor", 1.0))
        tf.meta.update(source=source["name"], source_frame=source["frame"], source_kind=kind,
                       source_note=source["note"])
        out[kind] = tf
    return out


def kind_words(keys=KINDS) -> pd.DataFrame:
    """The vocabulary table: the word for each reference kind beside the code key file names carry."""
    return pd.DataFrame([dict(word=KIND_WORD[k], key=k) for k in keys])
