"""The earth rule over every product, and the product of record per site and component.

Ported from scripts/processing/wamt_esp2026_readings.py and the quality call it uses,
scripts/qc/qld_salvage.py:177-210 (D:/BEN/MTH5_Aurora_mt-io_2026). Every threshold is an argument with the
frozen value as its default.

THE EARTH RULE, per product and component, over QUALITY_BAND (10-1000 s by default):

    quadrant      the phase in its quadrant: Zxy in (0, 90) deg and Zyx in (-180, -90) deg, which is
                  (0, 90) after the +180 deg fold products.rho_phase applies. Read as the median of the
                  band and the fraction of its periods inside, not as every point: a sign flip moves the
                  median, a few noisy periods at the band edges do not (qld_salvage:201-205).
    continuity    the apparent-resistivity curve continuous: the log-log slope of rho against period
                  within SLOPE_MAX of zero and within SLOPE_PHASE_TOL of the slope the phase implies,
                  1 - phase / 45 deg, scored over the band and again over its short end (lo to 5 lo) with
                  the looser SLOPE_MAX_SHORT and SLOPE_TOL_SHORT (qld_salvage:177-183).
    short slope   the log-log slope of rho over the survey's live short band within SLOPE_TOL. The clause
                  follows the band the record actually carries: on a 1 Hz record whose live band starts at
                  10 s the clause is scored over 10-20 s, and where the live band leaves nothing of 2-20 s
                  the clause is UNJUDGED and does not veto (wamt_esp2026_readings.slope_band).
    z slope       the log-log slope of |Z| against period at or above Z_SLOPE_CUT. An electric field
                  following dB/dt -- an inductive loop, Z proportional to omega with a flat phase -- reads
                  -0.92 to -0.95 and passes quadrant, slope of rho, bar and held range alike; a uniform
                  half-space reads -0.5. One-sided: the shallow side is structure, not a known fault.
    the bar       the median relative error of |Z| over the band. A row whose bar exceeds BAR_MAX is not a
                  measurement. The bar always sits beside the shape call, because the shape rule alone
                  passes on noise.
    held          the periods where the impedance bar is under HELD_BAR_MAX and the phase is in quadrant.
                  A row that holds at no period is not an earth.

One column name is one quantity: `bar_10_1000` is the impedance bar, the median of the error over |Z|, and
`rho_bar_10_1000` is the apparent-resistivity bar, which is twice it. The frozen tool scored the bar guard on
the apparent-resistivity bar at 1.0; BAR_MAX here is the impedance bar's own ceiling and both bars are in the
table, so which quantity a guard reads is never in doubt.

AGREEMENT between two products of one site and component: the median departure in apparent resistivity within
AGREE_RHO and the median departure in phase within AGREE_PHASE over AGREE_BAND (20 per cent and 5 deg over
5-200 s). The statistic is the frozen tool's median of the absolute departures, which refuses a curve that
wobbles about the other by more than the tolerance; the median ratio and the median phase difference that
workbook 04 reports are carried in the same row beside it.

EVERY PRODUCT of a site is scored the same way: workbook 03's reference kinds at both rates, each selection
of hours a 10 Hz pass ran on (whole, f05, f10, f25 and the random control r25), workbook 05's forms, and any
product a run folder holds that no table names. The readings table carries `selection` and `form` beside the
kind, so which pass a row belongs to is never inferred from its bar.

WHICH PRODUCTS MAY BE DELIVERED is forms.csv's call and not this module's. A workbook 03 product may always
be delivered; a form may only where forms.csv marks it a candidate, which it does where the form beats every
control it carries on the bar by the stated margin and is not an inter-site impedance. A form that is not a
candidate is read, scored and reported, is never the product of record, and does not corroborate another row.

THE PRODUCT OF RECORD, per site and component: among the products that are earths AND agree with at least one
product of another reference kind, the one with the smallest bar over QUALITY_BAND; ties are broken by the
longest period held. The choice is made at the delivery rate, which is 1 Hz: a 10 Hz product of a long-period
survey stops near 1,200 s, and its short end enters the delivered file through the splice rather than as the
whole row. Corroboration comes from another kind because two references that share no magnetics
cannot carry the same noise into the estimate. A component with no agreeing earth has no product of record and
the row says which clause emptied it. The release and any earlier processing are never the arbiter: the rule
reads the products' own soundness and their agreement with each other.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import products as PR
from .process import KIND_WORD, KINDS

QUALITY_BAND = (10.0, 1000.0)      # the band the shape, the bar and the choice are read over, in s
SHORT_SLOPE_BAND = (2.0, 20.0)     # the short slope clause's band before the live band clips it, in s
SLOPE_TOL = 1.2                    # |d log rho / d log T| over the short band beyond this is not an earth
AGREE_RHO = 0.20                   # two products agree within this fraction in apparent resistivity
AGREE_PHASE = 5.0                  # ... and this many degrees in phase
AGREE_BAND = (5.0, 200.0)          # ... over this band, in s
Z_SLOPE_CUT = -0.75                # d log|Z| / d log T below this is an inductive loop, not an earth
BAR_MAX = 1.0                      # the median relative error of |Z| over the band, at most this
HELD_BAR_MAX = 0.20                # a period is held where its impedance bar is under this and the phase

QUADRANT_FRAC = 0.70               # the fraction of the band's periods that must sit in the quadrant
SLOPE_MAX = 1.2                    # the continuity clause over the whole band
SLOPE_PHASE_TOL = 1.0              # ... and its departure from the slope the phase implies
SLOPE_MAX_SHORT = 2.0              # the same over the band's short end, looser for its scatter
SLOPE_TOL_SHORT = 1.5
SHORT_END_FACTOR = 5.0             # the short end of the band is lo to SHORT_END_FACTOR x lo

BAR_BANDS = ((2.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, 10000.0))
MIN_POINTS = 4                     # a slope is fitted to at least this many periods
COMPONENTS = ("xy", "yx")

READINGS_COLUMNS = ["site", "component", "kind", "kind_word", "selection", "form", "run", "stamp",
                    "rate_hz", "status", "earth", "not_earth_because", "quadrant", "quadrant_frac",
                    "median_phase_deg", "continuous", "rho_slope", "rho_slope_implied", "short_slope",
                    "short_slope_band_s", "z_slope", "bar_10_1000", "rho_bar_10_1000", "held_lo_s",
                    "held_hi_s", "held_n", "n_periods", "agree_kinds", "agree_n", "reproducible",
                    "reproducible_why", "candidate", "path"]

RECORD_COLUMNS = ["site", "component", "product", "kind", "kind_word", "selection", "form", "run", "stamp",
                  "rate_hz", "bar_10_1000", "held_hi_s", "held_n", "earth", "agree_n", "agree_kinds",
                  "reproducible", "why", "alternatives", "flagged", "note", "path"]


# ------------------------------------------------------------------ the survey's live band

def live_band(sv) -> tuple | None:
    """(lo, hi) in s of the band the survey's records carry, or None where none is declared.

    `live_band_s: [lo, hi]` in survey.yaml is read first. Failing that the liveness bands of the record
    stage, `bands.liveness`, are read as strings of the form "lo-hi" and their outer edges are taken: those
    are the bands the survey declares its channels are judged live over, so a clause scored outside them is
    scored on a band the record does not carry.
    """
    cfg = getattr(sv, "cfg", sv) or {}
    declared = cfg.get("live_band_s")
    if declared:
        lo, hi = float(declared[0]), float(declared[1])
        return (lo, hi) if lo < hi else None
    bands = ((cfg.get("bands") or {}).get("liveness") or [])
    edges = []
    for text in bands:
        try:
            a, _, b = str(text).partition("-")
            edges += [float(a), float(b)]
        except ValueError:
            continue
    return (min(edges), max(edges)) if edges else None


def slope_band(live, band=SHORT_SLOPE_BAND) -> tuple | None:
    """The short slope clause's band clipped to the live band, or None where the clause has no band left.

    Ported from wamt_esp2026_readings.slope_band. A survey with no live band declared keeps the clause's own
    band unchanged, and a clause with no band left is UNJUDGED and must not veto an earth row.
    """
    lo, hi = float(band[0]), float(band[1])
    if live is None:
        return lo, hi
    lo = max(lo, float(live[0]))
    hi = min(hi, float(live[1])) if live[1] else hi
    return (lo, hi) if lo < hi else None


# ------------------------------------------------------------------ one component's curve

def curve(tf, comp: str):
    """(period, rho, rho error, phase in deg, phase error) of one component, the yx phase folded by +180."""
    rho, rho_err, ph, ph_err = PR.rho_phase(tf.period, tf.z, tf.z_err, comp)
    return np.asarray(tf.period, float), rho, rho_err, ph, ph_err


def bar(tf, comp: str, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1]) -> float:
    """The median relative impedance error of one component over a band: the product's bar."""
    i, j = PR.COMPONENTS[comp]
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.z)[:, i, j]
    e = np.asarray(tf.z_err, float)[:, i, j]
    m = (p >= lo) & (p <= hi) & np.isfinite(z) & np.isfinite(e) & (np.abs(z) > 0)
    return float(np.median(e[m] / np.abs(z[m]))) if m.any() else np.nan


def rho_bar(tf, comp: str, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1]) -> float:
    """The median relative apparent-resistivity error over a band: twice the impedance bar."""
    p, rho, rho_err, _ph, _pe = curve(tf, comp)
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0) & np.isfinite(rho_err)
    return float(np.median(rho_err[m] / rho[m])) if m.any() else np.nan


def quadrant(p, ph, lo, hi, frac=QUADRANT_FRAC) -> tuple:
    """(in quadrant, the fraction of the band inside it, the median phase in deg).

    qld_salvage:201-205: the median of the band must sit in (0, 90) deg and at least `frac` of the band's
    periods with it.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(ph)
    if not m.any():
        return False, np.nan, np.nan
    inside = (ph[m] > 0) & (ph[m] < 90)
    med = float(np.median(ph[m]))
    return bool(0 < med < 90 and inside.mean() >= frac), float(inside.mean()), med


def _slope_pair(p, rho, ph, lo, hi, slope_max, tol) -> tuple:
    """(holds, the fitted log-log slope, the slope the phase implies) over one band."""
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0) & np.isfinite(ph)
    if m.sum() < 3:
        return True, np.nan, np.nan
    slope = float(np.polyfit(np.log10(p[m]), np.log10(rho[m]), 1)[0])
    implied = 1.0 - float(np.median(ph[m])) / 45.0
    return bool(abs(slope) <= slope_max and abs(slope - implied) <= tol), slope, implied


def continuous(p, rho, ph, lo, hi, slope_max=SLOPE_MAX, tol=SLOPE_PHASE_TOL,
               slope_max_short=SLOPE_MAX_SHORT, tol_short=SLOPE_TOL_SHORT,
               short_factor=SHORT_END_FACTOR) -> tuple:
    """(the curve is continuous, the slope over the band, the slope the phase implies).

    qld_salvage._earth: scored over the whole band and again over its short end, lo to short_factor x lo,
    where the looser pair of bounds applies. A row of six decades of rho climbing over one decade of period
    is a filter on E and not the field, and the whole-band fit alone passed it.
    """
    whole, slope, implied = _slope_pair(p, rho, ph, lo, hi, slope_max, tol)
    short, _s, _i = _slope_pair(p, rho, ph, lo, short_factor * lo, slope_max_short, tol_short)
    return bool(whole and short), slope, implied


def short_slope(p, rho, lo, hi) -> float:
    """The log-log slope of rho against period over one band, NaN under MIN_POINTS periods.

    wamt_esp2026_readings.short_slope. An earth's slope lies within about +-1; a noise-biased single-station
    row ramps three or four decades over a decade of period.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0)
    return float(np.polyfit(np.log10(p[m]), np.log10(rho[m]), 1)[0]) if m.sum() >= MIN_POINTS else np.nan


def z_slope(p, rho, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1]) -> float:
    """The log-log slope of |Z| against period over one band, NaN under MIN_POINTS periods.

    wamt_esp2026_readings.z_slope: rho = 0.2 T |Z|^2, so log|Z| = (log rho - log 0.2 - log T) / 2 and no
    second read of the file is needed.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0)
    if m.sum() < MIN_POINTS:
        return np.nan
    lz = 0.5 * (np.log10(rho[m]) - np.log10(0.2) - np.log10(p[m]))
    return float(np.polyfit(np.log10(p[m]), lz, 1)[0])


def held(tf, comp: str, held_bar_max=HELD_BAR_MAX) -> tuple:
    """(the shortest period held, the longest, how many) over the whole curve.

    A period is held where its impedance bar is under `held_bar_max` and its folded phase is in (0, 90) deg.
    The frozen tool wrote the same guard as an apparent-resistivity bar under 0.4, which is this one at 0.20.
    """
    i, j = PR.COMPONENTS[comp]
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.z)[:, i, j]
    e = np.asarray(tf.z_err, float)[:, i, j]
    _p, _rho, _re, ph, _pe = curve(tf, comp)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = e / np.abs(z)
    m = (np.isfinite(z) & (np.abs(z) > 0) & np.isfinite(e) & (rel < held_bar_max)
         & np.isfinite(ph) & (ph >= 0) & (ph <= 90))
    return (float(p[m].min()), float(p[m].max()), int(m.sum())) if m.any() else (np.nan, np.nan, 0)


# ------------------------------------------------------------------ the reading and the earth flag

def quality(tf, comp: str, band=QUALITY_BAND, live=None, held_bar_max=HELD_BAR_MAX,
            quadrant_frac=QUADRANT_FRAC, slope_max=SLOPE_MAX, slope_phase_tol=SLOPE_PHASE_TOL,
            slope_max_short=SLOPE_MAX_SHORT, slope_tol_short=SLOPE_TOL_SHORT,
            short_slope_band=SHORT_SLOPE_BAND) -> dict:
    """Every statistic the earth rule reads, for one product and one component.

    `live` is the survey's live band; the short slope clause is scored on `short_slope_band` clipped to it,
    and is NaN where the clip leaves nothing.
    """
    lo, hi = float(band[0]), float(band[1])
    p, rho, rho_err, ph, ph_err = curve(tf, comp)
    inq, frac, med = quadrant(p, ph, lo, hi, quadrant_frac)
    cont, slope, implied = continuous(p, rho, ph, lo, hi, slope_max, slope_phase_tol,
                                      slope_max_short, slope_tol_short)
    ss_band = slope_band(live, short_slope_band)
    ss = short_slope(p, rho, ss_band[0], ss_band[1]) if ss_band else np.nan
    h = held(tf, comp, held_bar_max)
    out = dict(quadrant=inq, quadrant_frac=frac, median_phase_deg=med,
               continuous=cont, rho_slope=slope, rho_slope_implied=implied,
               short_slope=ss,
               short_slope_band_s=("%g-%g" % ss_band if ss_band else "UNJUDGED: outside the live band"),
               z_slope=z_slope(p, rho, lo, hi),
               bar_10_1000=bar(tf, comp, lo, hi), rho_bar_10_1000=rho_bar(tf, comp, lo, hi),
               held_lo_s=h[0], held_hi_s=h[1], held_n=h[2],
               n_periods=int(np.isfinite(rho).sum()))
    for a, b in BAR_BANDS:
        out["bar_%g_%g" % (a, b)] = bar(tf, comp, a, b)
    return out


def earth(reading: dict, slope_tol=SLOPE_TOL, z_slope_cut=Z_SLOPE_CUT, bar_max=BAR_MAX) -> tuple:
    """(the row is an earth, the clauses it failed).

    Every clause of the rule, in the order the frozen tool applied them: quadrant and continuity together,
    the short slope clause where its band survives the live band, the two guards a shape test cannot supply
    (a row that holds at no period, a bar above its own value), and the impedance slope.
    """
    why = []
    if not reading.get("quadrant"):
        why.append("the phase is out of quadrant (median %.0f deg, %.0f %% of the band inside)"
                   % (reading.get("median_phase_deg", np.nan), 100 * (reading.get("quadrant_frac") or 0.0)))
    if not reading.get("continuous"):
        why.append("the rho curve is not continuous (slope %+.2f, the phase implies %+.2f)"
                   % (reading.get("rho_slope", np.nan), reading.get("rho_slope_implied", np.nan)))
    ss = reading.get("short_slope", np.nan)
    if np.isfinite(ss) and abs(ss) > slope_tol:
        why.append("the short slope is %+.2f over %s s, beyond +-%.1f"
                   % (ss, reading.get("short_slope_band_s", ""), slope_tol))
    if not reading.get("held_n"):
        why.append("it holds at no period")
    b = reading.get("bar_10_1000", np.nan)
    if np.isfinite(b) and b > bar_max:
        why.append("the bar is %.2f, above %.2f" % (b, bar_max))
    if not np.isfinite(b):
        why.append("no finite impedance in the band")
    zs = reading.get("z_slope", np.nan)
    if np.isfinite(zs) and zs < z_slope_cut:
        why.append("d log|Z| / d log T is %+.3f, below %+.3f: an inductive loop, not an earth"
                   % (zs, z_slope_cut))
    return (not why), "; ".join(why)


# ------------------------------------------------------------------ agreement between two products

def agree(a, b, comp: str, lo=AGREE_BAND[0], hi=AGREE_BAND[1], agree_rho=AGREE_RHO,
          agree_phase=AGREE_PHASE) -> dict:
    """The departure of `a` from `b` over one band and the agreement call.

    `rho_dev` and `phase_dev_deg` are the medians of the ABSOLUTE departures, which is the statistic the
    frozen tool ruled on: a curve wobbling about the other by more than the tolerance is refused even where
    its median ratio is one. `rho_ratio` and `phase_diff_deg` are the median ratio and the median difference
    workbook 04 reports, carried beside it.
    """
    pa, ra, _ea, pha, _pea = curve(a, comp)
    pb, rb, _eb, phb, _peb = curve(b, comp)
    m = (pa >= lo) & (pa <= hi) & np.isfinite(ra) & (ra > 0) & np.isfinite(pha)
    fb = np.isfinite(rb) & (rb > 0) & np.isfinite(phb)
    if m.sum() < 2 or fb.sum() < 2:
        return dict(rho_dev=np.nan, phase_dev_deg=np.nan, rho_ratio=np.nan, phase_diff_deg=np.nan,
                    n=int(m.sum()), agrees=False)
    order = np.argsort(pb[fb])
    xb, yr, yp = np.log(pb[fb][order]), np.log(rb[fb][order]), phb[fb][order]
    lr = np.interp(np.log(pa[m]), xb, yr)
    lp = np.interp(np.log(pa[m]), xb, yp)
    ratio = np.exp(lr) / ra[m]
    dphi = (lp - pha[m] + 180.0) % 360.0 - 180.0
    dev = float(np.median(np.abs(ratio - 1.0)))
    devp = float(np.median(np.abs(dphi)))
    return dict(rho_dev=dev, phase_dev_deg=devp, rho_ratio=float(np.median(ratio)),
                phase_diff_deg=float(np.median(dphi)), n=int(m.sum()),
                agrees=bool(dev <= agree_rho and devp <= agree_phase))


# ------------------------------------------------------------------ every product of a site

RUN_STAMP = re.compile(r"^(?P<run>.+)_(?P<stamp>\d{8}_\d{4})$")
FORM_NAME = re.compile(r"^(?P<form>.+?)_(?P<kind>%s)_(?P<rate>\d+)hz_(?P<params>.+)$"
                       % "|".join(sorted(KINDS, key=len, reverse=True)))
PRODUCT_COLS = ["site", "run", "stamp", "kind", "rate_hz", "params", "selection", "form", "path",
                "on_disk", "candidate"]


def split_run_folder(name: str) -> tuple:
    """(the run name, the stamp) of a run folder. The stamp carries an underscore of its own, so the split
    is on the trailing YYYYmmdd_HHMM and not on the last underscore."""
    m = RUN_STAMP.match(str(name))
    return (m.group("run"), m.group("stamp")) if m else (str(name), "")


def forms_products(work_root, sites) -> pd.DataFrame:
    """One row per product workbook 05 left, read from the forms.csv of every run folder that has one.

    Workbook 05's products do not sit in the ledger and their file names carry the form, so the folder rule
    the ledger reads by does not name them. forms.csv is the record of what each one is.
    """
    work = Path(work_root)
    rows = []
    for site in sites:
        for table in sorted((work / str(site)).glob("*/forms.csv")):
            folder = table.parent
            run, stamp = split_run_folder(folder.name)
            try:
                t = pd.read_csv(table)
            except Exception:
                continue
            for r in t.itertuples():
                path = folder / str(r.product)
                rows.append(dict(site=str(r.site), run=run, stamp=stamp, kind=str(r.kind),
                                 rate_hz=float(r.rate_hz), params="",
                                 selection=PR.WHOLE_SELECTION, form=str(r.form),
                                 path=str(path), status=str(r.status), candidate=bool(r.candidate),
                                 on_disk=path.exists()))
    return pd.DataFrame(rows, columns=PRODUCT_COLS + ["status"])


def parse_form_name(path, sites) -> dict:
    """{site, form, kind, rate_hz, params} read off a form's file name, or an empty dict.

    The name is <site>_<form>_<kind>_<rate>hz_<params>.edi and both the site and the form carry
    underscores of their own, so the site is matched against the survey's own list, longest first, and the
    kind is matched as one of the five code keys with the longest alternatives tried first.
    """
    stem = Path(path).stem
    for site in sorted(sites, key=len, reverse=True):
        if not stem.startswith("%s_" % site):
            continue
        m = FORM_NAME.match(stem[len(site) + 1:])
        if not m:
            return {}
        return dict(site=site, form=m.group("form"), kind=m.group("kind"),
                    rate_hz=float(m.group("rate")), params=m.group("params"))
    return {}


def stray_products(folders, sites, known) -> pd.DataFrame:
    """Every EDI in the given run folders that no table names, read off its own file name.

    A run folder can hold a product an earlier pass wrote and the folder's forms.csv no longer names -- a
    form whose lender changed between runs leaves the old one behind. The file is still a product of that
    run and is read and scored like any other; `candidate` False keeps it out of the delivery, which is
    forms.csv's call and not this module's.

    `folders` are the run folders the chosen products already come from. A run folder no table references
    at all is a different run and is left to the `runs` parameter, which is what says which runs are read:
    walking every folder under a site would take a run still being written as products of this one.
    """
    rows = []
    for folder in sorted(set(Path(f) for f in folders)):
        run, stamp = split_run_folder(folder.name)
        for p in sorted(folder.glob("*.edi")):
            if str(p).lower() in known:
                continue
            meta = parse_form_name(p, sites) or PR.parse_product_name(p)
            if not meta or meta.get("site") not in set(sites):
                continue
            rows.append(dict(site=meta["site"], run=run, stamp=stamp, kind=meta["kind"],
                             rate_hz=meta["rate_hz"], params=meta["params"],
                             selection=meta.get("selection", PR.WHOLE_SELECTION),
                             form=meta.get("form", ""), path=str(p), on_disk=True,
                             candidate=False))
    return pd.DataFrame(rows, columns=PRODUCT_COLS)


def all_products(sv, sites, runs="all", work_root=None, forms=True, from_names=True) -> pd.DataFrame:
    """Every product of the chosen sites: the ledger's, workbook 05's forms, and any named on disk only.

    `form` is empty for a product workbook 03 made and the form's name for one workbook 05 made;
    `selection` is the selection of hours the pass ran on, `whole` for the whole record; `candidate` is
    forms.csv's call on whether a form may be delivered. A form whose forms.csv row is not `made` or
    `exists` is dropped: it has no file to read.
    """
    work = Path(work_root or sv.cfg["work_root"])
    led = PR.find_products(sv, sites, runs=runs, work_root=work).assign(form="", candidate=True)
    out = [led[PRODUCT_COLS]]
    if forms:
        fm = forms_products(work, sites)
        if len(fm):
            fm = fm[fm.status.isin(["made", "exists"])]
            out.append(fm[PRODUCT_COLS])
    if from_names:
        known = {str(p).lower() for frame in out for p in frame.path}
        folders = {Path(p).parent for frame in out for p in frame.path}
        extra = stray_products(folders, sites, known)
        if len(extra):
            out.append(extra[PRODUCT_COLS])
    d = pd.concat(out, ignore_index=True) if len(out) > 1 else out[0].copy()
    d["selection"] = d.selection.fillna(PR.WHOLE_SELECTION).replace("", PR.WHOLE_SELECTION)
    return d.drop_duplicates(["site", "run", "stamp", "kind", "rate_hz", "selection",
                              "form"]).reset_index(drop=True)


def product_label(row) -> str:
    """The name a table and a figure call one product: the kind, the selection or form, and the rate."""
    form = str(getattr(row, "form", "") or "")
    sel = str(getattr(row, "selection", "") or PR.WHOLE_SELECTION)
    tag = form or ("" if sel in ("", PR.WHOLE_SELECTION) else sel)
    return "%s%s %g Hz" % (row.kind, ("/" + tag) if tag else "", float(row.rate_hz))


# ------------------------------------------------------------------ the readings table

def readings_table(products: pd.DataFrame, read=None, band=QUALITY_BAND, live=None,
                   agree_band=AGREE_BAND, agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE,
                   slope_tol=SLOPE_TOL, z_slope_cut=Z_SLOPE_CUT, bar_max=BAR_MAX,
                   held_bar_max=HELD_BAR_MAX, short_slope_band=SHORT_SLOPE_BAND) -> pd.DataFrame:
    """One row per product and component with every statistic, the earth flag and the agreement count.

    `read` turns a path into a TFData; the default reads each file once per call. The agreement count is the
    number of OTHER reference kinds of the same site and component whose row is an earth and agrees with this
    one over `agree_band`, which is what the product of record is chosen among.
    """
    read = read or PR.read_tf
    rows = []
    for (site,), grp in products.groupby(["site"], sort=True):
        tfs, order = {}, []
        for r in grp.itertuples():
            key = (r.run, r.stamp, r.kind, float(r.rate_hz),
                   str(getattr(r, "selection", "") or PR.WHOLE_SELECTION),
                   str(getattr(r, "form", "") or ""))
            order.append((key, r))
            if not r.on_disk:
                continue
            try:
                tfs[key] = read(r.path)
            except Exception:
                tfs[key] = None
        for comp in COMPONENTS:
            reads = {}
            for key, r in order:
                tf = tfs.get(key)
                if tf is None:
                    reads[key] = None
                    continue
                q = quality(tf, comp, band=band, live=live, held_bar_max=held_bar_max,
                            short_slope_band=short_slope_band)
                is_earth, why = earth(q, slope_tol=slope_tol, z_slope_cut=z_slope_cut, bar_max=bar_max)
                reads[key] = dict(q, earth=is_earth, not_earth_because=why)
            for key, r in order:
                rd = reads.get(key)
                if rd is None:
                    rows.append(dict(site=site, component=comp, kind=r.kind,
                                     kind_word=KIND_WORD.get(r.kind, r.kind),
                                     selection=str(getattr(r, "selection", "") or PR.WHOLE_SELECTION),
                                     form=str(getattr(r, "form", "") or ""), run=r.run, stamp=r.stamp,
                                     rate_hz=float(r.rate_hz),
                                     status=("not on disk" if not r.on_disk else "unreadable"),
                                     earth=False, not_earth_because="the file could not be read",
                                     agree_kinds="", agree_n=0,
                                     candidate=bool(getattr(r, "candidate", True)), path=r.path))
                    continue
                partners = []
                if rd["earth"]:
                    for other, ro in order:
                        # a product corroborates only where it could itself be delivered: a form workbook
                        # 05 did not promote is read and scored, and it does not vouch for another row
                        if other == key or ro.kind == r.kind or not bool(getattr(ro, "candidate", True)):
                            continue
                        od = reads.get(other)
                        if od is None or not od["earth"]:
                            continue
                        a = agree(tfs[key], tfs[other], comp, agree_band[0], agree_band[1],
                                  agree_rho, agree_phase)
                        if a["agrees"]:
                            partners.append(ro.kind)
                rows.append(dict(site=site, component=comp, kind=r.kind,
                                 kind_word=KIND_WORD.get(r.kind, r.kind),
                                 selection=str(getattr(r, "selection", "") or PR.WHOLE_SELECTION),
                                 form=str(getattr(r, "form", "") or ""), run=r.run, stamp=r.stamp,
                                 rate_hz=float(r.rate_hz), status="ok",
                                 agree_kinds=" ".join(sorted(set(partners))),
                                 agree_n=len(set(partners)),
                                 candidate=bool(getattr(r, "candidate", True)), path=r.path, **rd))
    out = pd.DataFrame(rows)
    for c in READINGS_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    front = [c for c in READINGS_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def agreement_matrix(products: pd.DataFrame, read=None, comp="xy", agree_band=AGREE_BAND,
                     agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE) -> pd.DataFrame:
    """Every pair of products of one site and component with the two departures and the agreement call."""
    read = read or PR.read_tf
    rows = []
    for (site,), grp in products.groupby(["site"], sort=True):
        items = [r for r in grp.itertuples() if r.on_disk]
        tfs = {}
        for k, r in enumerate(items):
            try:
                tfs[k] = read(r.path)
            except Exception:
                tfs[k] = None
        for m in range(len(items)):
            for n in range(m + 1, len(items)):
                a, b = items[m], items[n]
                if tfs[m] is None or tfs[n] is None:
                    continue
                s = agree(tfs[m], tfs[n], comp, agree_band[0], agree_band[1],
                          agree_rho, agree_phase)
                rows.append(dict(site=site, component=comp, a=product_label(a), b=product_label(b),
                                 kind_a=a.kind, kind_b=b.kind,
                                 same_kind=bool(a.kind == b.kind), **s))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the product of record

def _rate_words(rates) -> str:
    return "every rate's" if rates is None else " and ".join("%g Hz" % float(r) for r in rates)


def product_of_record(readings: pd.DataFrame, bar_max=BAR_MAX, agree_band=AGREE_BAND,
                      band=QUALITY_BAND, rates=(1.0,)) -> pd.DataFrame:
    """Per site and component the chosen product with the reason, or none with the reason there is none.

    The pool is the rows at the delivery rate that are earths and agree with at least one product of another
    reference kind. The choice inside it is the smallest bar over `band`, ties broken by the longest period
    held. A component whose pool is empty has no product of record, and the row says whether the earths
    did not corroborate each other or there was no earth at all.

    `rates` is the delivery rate or rates the choice is made among; None reads every rate in the table. The
    default is the 1 Hz row, because a 10 Hz product of this survey stops near 1,200 s and the row a
    long-period survey delivers has to cover the delivery band: the 10 Hz product enters through the splice,
    where its short end joins the chosen row at 16 s, and not as the whole row.
    """
    rows = []
    for (site, comp), g in readings.groupby(["site", "component"], sort=True):
        ok = g[(g.status == "ok")]
        if rates is not None:
            ok = ok[ok.rate_hz.isin([float(r) for r in rates])]
        # forms.csv says which forms may be delivered: a form that does not beat its control buys
        # efficiency and not a different answer, and an inter-site impedance is shown and never delivered
        if "candidate" in ok.columns:
            ok = ok[(ok.form == "") | ok.candidate.astype(bool)]
        earths = ok[ok.earth.astype(bool)]
        pool = earths[earths.agree_n > 0]
        alt = "; ".join("%s %.4f" % (product_label(r), r.bar_10_1000) for r in
                        earths.sort_values("bar_10_1000").itertuples()
                        if np.isfinite(r.bar_10_1000)) or "none"
        if len(pool):
            # the smallest bar, ties by the longest period held: two products of one record can carry the
            # same bar to the digit a table prints, and the one that reaches further is the one to deliver
            best = pool.sort_values(["bar_10_1000", "held_hi_s"],
                                    ascending=[True, False]).iloc[0]
            tied = pool[np.isclose(pool.bar_10_1000, best.bar_10_1000, rtol=1e-9, atol=0)]
            why = ("the smallest bar over %g-%g s among the %s earths that agree with another kind over "
                   "%g-%g s%s" % (band[0], band[1], _rate_words(rates), agree_band[0], agree_band[1],
                                  ("; %d tied on the bar and the longest period held broke it"
                                   % len(tied)) if len(tied) > 1 else ""))
            rows.append(dict(site=site, component=comp, product=product_label(best), kind=best.kind,
                             kind_word=best.kind_word,
                             selection=best.get("selection", PR.WHOLE_SELECTION), form=best.form,
                             run=best.run, stamp=best.stamp,
                             rate_hz=best.rate_hz, bar_10_1000=best.bar_10_1000,
                             held_hi_s=best.held_hi_s, held_n=best.held_n, earth=True,
                             agree_n=int(best.agree_n), agree_kinds=best.agree_kinds,
                             reproducible=best.get("reproducible", ""), why=why, alternatives=alt,
                             flagged="", note="", path=best.path))
            continue
        if len(earths):
            why = ("no product of record: %d %s row(s) are earths and none agrees with a product of another "
                   "kind over %g-%g s (%s)"
                   % (len(earths), _rate_words(rates), agree_band[0], agree_band[1],
                      ", ".join(sorted(set(earths.kind)))))
        elif len(ok):
            reasons = sorted({str(r.not_earth_because).split(";")[0] for r in ok.itertuples()})
            why = ("no product of record: no %s row is an earth (%s)"
                   % (_rate_words(rates), "; ".join(reasons[:3])))
        else:
            why = ("no product of record: no %s product of this site and component could be read"
                   % _rate_words(rates))
        rows.append(dict(site=site, component=comp, product="none", kind="", kind_word="", selection="",
                         form="", run="", stamp="", rate_hz=np.nan, bar_10_1000=np.nan, held_hi_s=np.nan,
                         held_n=0, earth=False, agree_n=0, agree_kinds="", reproducible="",
                         why=why, alternatives=alt, flagged="", note="", path=""))
    out = pd.DataFrame(rows)
    for c in RECORD_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[RECORD_COLUMNS]


def candidates(record: pd.DataFrame) -> pd.DataFrame:
    """The distinct products the record chose, one row each: what the split-half passes are run over."""
    cols = ["site", "kind", "rate_hz", "selection", "form", "run", "stamp"]
    chosen = record[record["product"] != "none"]
    if not len(chosen):
        return pd.DataFrame(columns=cols + ["components"])
    g = (chosen.groupby(cols)["component"].apply(lambda v: " ".join(sorted(v))).reset_index()
         .rename(columns={"component": "components"}))
    return g.sort_values(["site", "kind"]).reset_index(drop=True)
