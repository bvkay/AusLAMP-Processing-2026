"""The three response tests over every transfer function, and the transfer function of record per component.

Ported from scripts/processing/wamt_esp2026_readings.py and the quality call it uses,
scripts/qc/qld_salvage.py:177-210 (D:/BEN/MTH5_Aurora_mt-io_2026). Every threshold is an argument with the
frozen value as its default.

The three response tests, per transfer function and component, over QUALITY_BAND (10-1000 s by default):

    the phase test    the phase lies in its quadrant -- 0-90 deg for xy, and for yx after the +180 deg fold
                      rho_phase applies -- at QUADRANT_MIN of the band's periods. A row that fails it carries
                      a sign fault on an E or an H line.
    the slope test    apparent resistivity cannot change faster than the period: |d log rho / d log T| <= 1
                      for a one-dimensional earth (Weidelt 1972; Parker and Booker 1996). The log-log slope
                      between adjacent periods lies inside +-(SLOPE_BOUND + SLOPE_TOL) at SLOPE_MIN of the
                      adjacent pairs. A two- or three-dimensional response can exceed the bound, so the test
                      is a screen carrying a tolerance for noise and not a law.
    the error test    the median relative error of |Z| over the band at or under BAR_MAX, with at least
                      MIN_PERIODS periods in the band carrying an error under BAR_MAX.

A transfer function passes the response tests where all three hold; `fails` names the tests it did not hold.

One column name is one quantity: `bar` is the impedance bar, the median of the error over |Z| across
QUALITY_BAND, and `rho_bar` is the apparent-resistivity bar, which is twice it. `n_periods` is the error
test's count -- the periods of the band whose error is under BAR_MAX -- and `n_finite_rho` is how many
periods of the whole curve carry an apparent resistivity at all.

The held band is the periods where the impedance bar is under HELD_BAR_MAX and the phase is in quadrant. It
is not a test: it is the band the delivered file is trimmed to, so that the file carries the measurement
alone.

The kinds a transfer function may be delivered on are the four references: remote site, fleet stack,
observatory, stack + observatory. The single station is not one of them. Its estimate is biased low by
whatever noise sits in the site's own H, its error bars do not show that bias, and a corroboration count that
included it would count a row against its own noise. `deliverable` splits a frame on that rule and the
workbooks report what it dropped.

Agreement between two transfer functions of one site and component: the median departure in apparent
resistivity within AGREE_RHO and the median departure in phase within AGREE_PHASE over AGREE_BAND (20 per
cent and 5 deg over 5-200 s). The statistic is the median of the absolute departures, which refuses a curve
that wobbles about the other by more than the tolerance; the median ratio and the median phase difference
workbook 04 reports are carried in the same row beside it.

Every transfer function of a site is scored the same way: workbook 03's reference kinds at both rates, each
selection of hours a 10 Hz pass ran on (whole, the longest coherent stretch, and the random control of the
same length), workbook 05's forms, and any file a run folder holds that no table names. The readings table
carries `selection` and `form` beside the kind, so which pass a row belongs to is never inferred from its
bar.

Which transfer functions may be delivered is forms.csv's call and not this module's. A workbook 03 row may
always be delivered; a form may only where forms.csv marks it a candidate, which it does where the form beats
every control it carries on the bar by the stated margin and is not an inter-site impedance. A form that is
not a candidate is read, scored and reported, is never the transfer function of record, and does not
corroborate another row.

The transfer function of record, per site and component: among the rows that pass the three response tests
and agree with at least one row of another reference kind, the one with the smallest bar over QUALITY_BAND;
ties are broken by the longest period held. The choice is made at the delivery rate, which is 1 Hz: a 10 Hz
pass of a long-period survey stops near 1,200 s, and its short end enters the delivered file through the
splice rather than as the whole row. Corroboration comes from another kind because two references that share
no magnetics cannot carry the same noise into the estimate. A component with no agreeing row has none of
record and the row says which test emptied it. The release and any earlier processing are never the arbiter.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import transfer_functions as TFN
from .process import KIND_WORD
from .process import KINDS as ALL_KINDS

QUALITY_BAND = (10.0, 1000.0)      # the band the three tests and the choice are read over, in s
QUADRANT_MIN = 0.70                # the phase test: this fraction of the band's periods inside the quadrant
SLOPE_BOUND = 1.0                  # the slope test: |d log rho / d log T| for a one-dimensional earth
SLOPE_TOL = 0.25                   # ... the tolerance added to the bound for noise
SLOPE_MIN = 0.80                   # ... and the fraction of adjacent pairs that must lie inside it
BAR_MAX = 1.0                      # the error test: the median relative error of |Z| over the band
MIN_PERIODS = 8                    # ... and the periods of the band that must carry an error under it

HELD_BAR_MAX = 0.20                # a period is held where its impedance bar is under this and the phase
                                   # is in quadrant; the held band is what a delivered file is trimmed to

AGREE_RHO = 0.20                   # two curves agree within this fraction in apparent resistivity
AGREE_PHASE = 5.0                  # ... and this many degrees in phase
AGREE_BAND = (5.0, 200.0)          # ... over this band, in s

DROPPED_KIND = "single"            # the single station: read where it sits on disk, never delivered
KINDS = tuple(k for k in ALL_KINDS if k != DROPPED_KIND)

BAR_BANDS = ((2.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, 10000.0))
COMPONENTS = ("xy", "yx")

READINGS_COLUMNS = ["site", "component", "kind", "kind_word", "selection", "form", "run", "stamp",
                    "rate_hz", "status", "transfer_function", "passes", "fails", "pass_phase", "phase_frac",
                    "median_phase_deg", "pass_slope", "slope_frac", "pass_error", "bar", "n_periods",
                    "rho_bar", "n_finite_rho", "held_lo_s", "held_hi_s", "held_n", "agree_kinds",
                    "agree_n", "reproducible", "reproducible_why", "candidate", "path"]

RECORD_COLUMNS = ["site", "component", "transfer_function", "kind", "kind_word", "selection", "form", "run",
                  "stamp", "rate_hz", "bar", "n_periods", "held_lo_s", "held_hi_s", "held_n", "passes",
                  "fails", "agree_n", "agree_kinds", "reproducible", "why", "alternatives", "flagged",
                  "note", "path"]


# ------------------------------------------------------------------ one component's curve

def curve(tf, comp: str):
    """(period, rho, rho error, phase in deg, phase error) of one component, the yx phase folded by +180."""
    rho, rho_err, ph, ph_err = TFN.rho_phase(tf.period, tf.z, tf.z_err, comp)
    return np.asarray(tf.period, float), rho, rho_err, ph, ph_err


def relative_error(tf, comp: str):
    """The error of one component over its own impedance magnitude, NaN where either is not finite."""
    i, j = TFN.COMPONENTS[comp]
    z = np.asarray(tf.z)[:, i, j]
    e = np.asarray(tf.z_err, float)[:, i, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = e / np.abs(z)
    rel[~(np.isfinite(z) & (np.abs(z) > 0) & np.isfinite(e))] = np.nan
    return rel


def bar(tf, comp: str, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1]) -> float:
    """The median relative impedance error of one component over a band: the curve's bar."""
    p = np.asarray(tf.period, float)
    rel = relative_error(tf, comp)
    m = (p >= lo) & (p <= hi) & np.isfinite(rel)
    return float(np.median(rel[m])) if m.any() else np.nan


def rho_bar(tf, comp: str, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1]) -> float:
    """The median relative apparent-resistivity error over a band: twice the impedance bar."""
    p, rho, rho_err, _ph, _pe = curve(tf, comp)
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0) & np.isfinite(rho_err)
    return float(np.median(rho_err[m] / rho[m])) if m.any() else np.nan


# ------------------------------------------------------------------ the three response tests

def phase_test(p, ph, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1], quadrant_min=QUADRANT_MIN) -> tuple:
    """(the test holds, the fraction of the band inside the quadrant, the median phase in deg).

    The phase of a transfer function of an earth lies in the first quadrant, which is (0, 90) deg for Zxy and
    (-180, -90) deg for Zyx, the second folded into the first by the +180 deg rho_phase applies. The fraction
    is read rather than every point, so a few noisy periods at the band edges do not refuse a curve and a
    sign flip, which moves every point, does.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(ph)
    if not m.any():
        return False, np.nan, np.nan
    inside = (ph[m] > 0) & (ph[m] < 90)
    return bool(inside.mean() >= quadrant_min), float(inside.mean()), float(np.median(ph[m]))


def slope_test(p, rho, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1], bound=SLOPE_BOUND, tol=SLOPE_TOL,
               slope_min=SLOPE_MIN) -> tuple:
    """(the test holds, the fraction of adjacent pairs inside the bound, the pairs read).

    Apparent resistivity cannot change faster than the period over a one-dimensional earth:
    |d log rho / d log T| <= 1 (Weidelt 1972; Parker and Booker 1996). A two- or three-dimensional response
    can violate the bound, which is why the test is applied as a screen with `tol` added for noise and a
    fraction of the pairs rather than all of them, and not as a law.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(rho) & (rho > 0)
    if m.sum() < 2:
        return False, np.nan, 0
    x, y = np.log10(p[m]), np.log10(rho[m])
    order = np.argsort(x)
    x, y = x[order], y[order]
    dx = np.diff(x)
    ok = dx > 0
    if not ok.any():
        return False, np.nan, 0
    s = np.diff(y)[ok] / dx[ok]
    inside = np.abs(s) <= (float(bound) + float(tol))
    return bool(inside.mean() >= slope_min), float(inside.mean()), int(inside.size)


def error_test(rel, p, lo=QUALITY_BAND[0], hi=QUALITY_BAND[1], bar_max=BAR_MAX,
               min_periods=MIN_PERIODS) -> tuple:
    """(the test holds, the bar, the periods of the band whose error is under `bar_max`).

    The bar always sits beside the shape tests, because the two shape tests alone pass on noise.
    """
    m = (p >= lo) & (p <= hi) & np.isfinite(rel)
    if not m.any():
        return False, np.nan, 0
    b = float(np.median(rel[m]))
    n = int((rel[m] < bar_max).sum())
    return bool(b <= bar_max and n >= min_periods), b, n


def held(tf, comp: str, held_bar_max=HELD_BAR_MAX) -> tuple:
    """(the shortest period held, the longest, how many) over the whole curve.

    A period is held where its impedance bar is under `held_bar_max` and its folded phase is in (0, 90) deg.
    The same guard written as an apparent-resistivity bar under 0.4 is this one at 0.20.
    """
    p = np.asarray(tf.period, float)
    rel = relative_error(tf, comp)
    _p, _rho, _re, ph, _pe = curve(tf, comp)
    m = np.isfinite(rel) & (rel < held_bar_max) & np.isfinite(ph) & (ph >= 0) & (ph <= 90)
    return (float(p[m].min()), float(p[m].max()), int(m.sum())) if m.any() else (np.nan, np.nan, 0)


def quality(tf, comp: str, band=QUALITY_BAND, quadrant_min=QUADRANT_MIN, slope_bound=SLOPE_BOUND,
            slope_tol=SLOPE_TOL, slope_min=SLOPE_MIN, bar_max=BAR_MAX, min_periods=MIN_PERIODS,
            held_bar_max=HELD_BAR_MAX) -> dict:
    """Every statistic the three response tests read, and the three verdicts, for one curve and component.

    `passes` is the conjunction of the three and `fails` names the tests that did not hold, in the order the
    tests are stated.
    """
    lo, hi = float(band[0]), float(band[1])
    p, rho, _rho_err, ph, _ph_err = curve(tf, comp)
    rel = relative_error(tf, comp)
    ok_phase, phase_frac, median_phase = phase_test(p, ph, lo, hi, quadrant_min)
    ok_slope, slope_frac, n_pairs = slope_test(p, rho, lo, hi, slope_bound, slope_tol, slope_min)
    ok_error, b, n_under = error_test(rel, p, lo, hi, bar_max, min_periods)
    h = held(tf, comp, held_bar_max)
    why = []
    if not ok_phase:
        why.append("phase (median %.0f deg, %.0f %% of %g-%g s inside the quadrant, %.0f %% needed): a sign "
                   "fault on an E or an H line"
                   % (median_phase, 100 * (phase_frac if np.isfinite(phase_frac) else 0.0), lo, hi,
                      100 * quadrant_min))
    if not ok_slope:
        why.append("slope (%.0f %% of the adjacent pairs inside +-%.2f, %.0f %% needed)"
                   % (100 * (slope_frac if np.isfinite(slope_frac) else 0.0), slope_bound + slope_tol,
                      100 * slope_min))
    if not ok_error:
        # "; " separates one failed test from the next, so no test's own sentence may carry one
        why.append("error (bar %s over %g-%g s against a ceiling of %.2f, %d period(s) under it against "
                   "%d needed)" % (("%.3f" % b) if np.isfinite(b) else "not finite", lo, hi, bar_max,
                                   n_under, min_periods))
    out = dict(pass_phase=ok_phase, phase_frac=phase_frac, median_phase_deg=median_phase,
               pass_slope=ok_slope, slope_frac=slope_frac, n_slope_pairs=n_pairs,
               pass_error=ok_error, bar=b, n_periods=n_under,
               rho_bar=rho_bar(tf, comp, lo, hi),
               held_lo_s=h[0], held_hi_s=h[1], held_n=h[2],
               n_finite_rho=int(np.isfinite(rho).sum()),
               passes=bool(ok_phase and ok_slope and ok_error), fails="; ".join(why))
    for a, c in BAR_BANDS:
        out["bar_%g_%g" % (a, c)] = bar(tf, comp, a, c)
    return out


# ------------------------------------------------------------------ agreement between two curves

def agree(a, b, comp: str, lo=AGREE_BAND[0], hi=AGREE_BAND[1], agree_rho=AGREE_RHO,
          agree_phase=AGREE_PHASE) -> dict:
    """The departure of `a` from `b` over one band and the agreement call.

    `rho_dev` and `phase_dev_deg` are the medians of the absolute departures: a curve wobbling about the
    other by more than the tolerance is refused even where its median ratio is one. `rho_ratio` and
    `phase_diff_deg` are the median ratio and the median difference workbook 04 reports, carried beside it.
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


# ------------------------------------------------------------------ every transfer function of a site

RUN_STAMP = re.compile(r"^(?P<run>.+)_(?P<stamp>\d{8}_\d{4})$")
FORM_NAME = re.compile(r"^(?P<form>.+?)_(?P<kind>%s)_(?P<rate>\d+)hz_(?P<params>.+)$"
                       % "|".join(sorted(ALL_KINDS, key=len, reverse=True)))
TF_COLS = ["site", "run", "stamp", "kind", "rate_hz", "params", "selection", "form", "path",
           "on_disk", "candidate"]


def split_run_folder(name: str) -> tuple:
    """(the run name, the stamp) of a run folder. The stamp carries an underscore of its own, so the split
    is on the trailing YYYYmmdd_HHMM and not on the last underscore."""
    m = RUN_STAMP.match(str(name))
    return (m.group("run"), m.group("stamp")) if m else (str(name), "")


def forms_tfs(work_root, sites) -> pd.DataFrame:
    """One row per transfer function workbook 05 left, from the forms.csv of every run folder that has one.

    Workbook 05's files do not sit in the ledger and their names carry the form, so the folder rule the
    ledger reads by does not name them. forms.csv is the record of what each one is.
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
                # forms.csv names the file in its own `transfer_function` column, which is that table's spelling and
                # not this module's
                path = folder / str(r.transfer_function)
                rows.append(dict(site=str(r.site), run=run, stamp=stamp, kind=str(r.kind),
                                 rate_hz=float(r.rate_hz), params="",
                                 selection=TFN.WHOLE_SELECTION, form=str(r.form),
                                 path=str(path), status=str(r.status), candidate=bool(r.candidate),
                                 on_disk=path.exists()))
    return pd.DataFrame(rows, columns=TF_COLS + ["status"])


def parse_form_name(path, sites) -> dict:
    """{site, form, kind, rate_hz, params} read off a form's file name, or an empty dict.

    The name is <site>_<form>_<kind>_<rate>hz_<params>.edi and both the site and the form carry
    underscores of their own, so the site is matched against the survey's own list, longest first, and the
    kind is matched as one of the code keys with the longest alternatives tried first.
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


def stray_tfs(folders, sites, known) -> pd.DataFrame:
    """Every EDI in the given run folders that no table names, read off its own file name.

    A run folder can hold a file an earlier pass wrote and the folder's forms.csv no longer names -- a form
    whose lender changed between runs leaves the old one behind. The file is still a transfer function of
    that run and is read and scored like any other; `candidate` False keeps it out of the delivery, which is
    forms.csv's call and not this module's.

    `folders` are the run folders the chosen rows already come from. A run folder no table references at all
    is a different run and is left to the `runs` parameter, which is what says which runs are read: walking
    every folder under a site would take a run still being written as part of this one.
    """
    rows = []
    for folder in sorted(set(Path(f) for f in folders)):
        run, stamp = split_run_folder(folder.name)
        for p in sorted(folder.glob("*.edi")):
            if str(p).lower() in known:
                continue
            meta = parse_form_name(p, sites) or TFN.parse_tf_name(p)
            if not meta or meta.get("site") not in set(sites):
                continue
            rows.append(dict(site=meta["site"], run=run, stamp=stamp, kind=meta["kind"],
                             rate_hz=meta["rate_hz"], params=meta["params"],
                             selection=meta.get("selection", TFN.WHOLE_SELECTION),
                             form=meta.get("form", ""), path=str(p), on_disk=True,
                             candidate=False))
    return pd.DataFrame(rows, columns=TF_COLS)


def all_tfs(sv, sites, runs="all", work_root=None, forms=True, from_names=True) -> pd.DataFrame:
    """Every transfer function of the chosen sites: the ledger's, workbook 05's forms, and any on disk only.

    `form` is empty for a row workbook 03 made and the form's name for one workbook 05 made; `selection` is
    the selection of hours the pass ran on, `whole` for the whole record; `candidate` is forms.csv's call on
    whether a form may be delivered. A form whose forms.csv row is not `made` or `exists` is dropped: it has
    no file to read.
    """
    work = Path(work_root or sv.cfg["work_root"])
    led = TFN.find_transfer_functions(sv, sites, runs=runs, work_root=work).assign(form="", candidate=True)
    out = [led[TF_COLS]]
    if forms:
        fm = forms_tfs(work, sites)
        if len(fm):
            fm = fm[fm.status.isin(["made", "exists"])]
            out.append(fm[TF_COLS])
    if from_names:
        known = {str(p).lower() for frame in out for p in frame.path}
        folders = {Path(p).parent for frame in out for p in frame.path}
        extra = stray_tfs(folders, sites, known)
        if len(extra):
            out.append(extra[TF_COLS])
    d = pd.concat(out, ignore_index=True) if len(out) > 1 else out[0].copy()
    d["selection"] = d.selection.fillna(TFN.WHOLE_SELECTION).replace("", TFN.WHOLE_SELECTION)
    return d.drop_duplicates(["site", "run", "stamp", "kind", "rate_hz", "selection",
                              "form"]).reset_index(drop=True)


def deliverable(tfs: pd.DataFrame, kinds=KINDS) -> tuple:
    """(the rows of the four reference kinds, the rows of every other kind).

    The second frame is what a workbook reports as read and ignored: a single-station file left on disk by an
    earlier pass is named and not scored, so a file nobody deleted cannot enter a table or a figure.
    """
    if not len(tfs):
        return tfs, tfs
    keep = tfs.kind.isin(list(kinds))
    return tfs[keep].reset_index(drop=True), tfs[~keep].reset_index(drop=True)


def tf_label(row) -> str:
    """The name a table and a figure call one row: the kind, the selection or form, and the rate."""
    form = str(getattr(row, "form", "") or "")
    sel = str(getattr(row, "selection", "") or TFN.WHOLE_SELECTION)
    tag = form or ("" if sel in ("", TFN.WHOLE_SELECTION) else sel)
    return "%s%s %g Hz" % (row.kind, ("/" + tag) if tag else "", float(row.rate_hz))


def tf_key(row) -> str:
    """The name an analyst writes in the choice cell: `stack_1hz`, `remote_10hz_stretch`, or a form's name.

    A workbook 05 form is named by its form, which is what its file name carries and what forms.csv calls
    it; a workbook 03 row is named by its kind, its rate and, where a 10 Hz pass ran on a selection of hours,
    that selection.
    """
    form = str(getattr(row, "form", "") or "")
    if form:
        return form
    sel = str(getattr(row, "selection", "") or TFN.WHOLE_SELECTION)
    tag = "" if sel in ("", TFN.WHOLE_SELECTION, "nan") else ("_" + sel)
    return "%s_%ghz%s" % (row.kind, float(row.rate_hz), tag)


# ------------------------------------------------------------------ the readings table

def readings_table(tfs: pd.DataFrame, read=None, band=QUALITY_BAND, agree_band=AGREE_BAND,
                   agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE, quadrant_min=QUADRANT_MIN,
                   slope_bound=SLOPE_BOUND, slope_tol=SLOPE_TOL, slope_min=SLOPE_MIN, bar_max=BAR_MAX,
                   min_periods=MIN_PERIODS, held_bar_max=HELD_BAR_MAX, kinds=KINDS) -> pd.DataFrame:
    """One row per transfer function and component: every statistic, the three verdicts, the agreement count.

    `read` turns a path into a TFData; the default reads each file once per call. The agreement count is the
    number of other reference kinds of the same site and component whose row passes the three tests and
    agrees with this one over `agree_band`, which is what the transfer function of record is chosen among.
    """
    read = read or TFN.read_tf
    rows = []
    for (site,), grp in tfs.groupby(["site"], sort=True):
        curves, order = {}, []
        for r in grp.itertuples():
            key = (r.run, r.stamp, r.kind, float(r.rate_hz),
                   str(getattr(r, "selection", "") or TFN.WHOLE_SELECTION),
                   str(getattr(r, "form", "") or ""))
            order.append((key, r))
            if not r.on_disk:
                continue
            try:
                curves[key] = read(r.path)
            except Exception:
                curves[key] = None
        for comp in COMPONENTS:
            reads = {}
            for key, r in order:
                tf = curves.get(key)
                if tf is None:
                    reads[key] = None
                    continue
                reads[key] = quality(tf, comp, band=band, quadrant_min=quadrant_min,
                                     slope_bound=slope_bound, slope_tol=slope_tol, slope_min=slope_min,
                                     bar_max=bar_max, min_periods=min_periods, held_bar_max=held_bar_max)
            for key, r in order:
                rd = reads.get(key)
                common = dict(site=site, component=comp, kind=r.kind,
                              kind_word=KIND_WORD.get(r.kind, r.kind),
                              selection=str(getattr(r, "selection", "") or TFN.WHOLE_SELECTION),
                              form=str(getattr(r, "form", "") or ""), run=r.run, stamp=r.stamp,
                              rate_hz=float(r.rate_hz), transfer_function=tf_key(r),
                              candidate=bool(getattr(r, "candidate", True)), path=r.path)
                if rd is None:
                    rows.append(dict(common,
                                     status=("not on disk" if not r.on_disk else "unreadable"),
                                     passes=False, fails="the file could not be read",
                                     agree_kinds="", agree_n=0))
                    continue
                partners = []
                if rd["passes"]:
                    for other, ro in order:
                        # a row corroborates only where it could itself be delivered: a form workbook 05 did
                        # not promote is read and scored, and it does not vouch for another row
                        if (other == key or ro.kind == r.kind or ro.kind not in set(kinds)
                                or not bool(getattr(ro, "candidate", True))):
                            continue
                        od = reads.get(other)
                        if od is None or not od["passes"]:
                            continue
                        a = agree(curves[key], curves[other], comp, agree_band[0], agree_band[1],
                                  agree_rho, agree_phase)
                        if a["agrees"]:
                            partners.append(ro.kind)
                rows.append(dict(common, status="ok", agree_kinds=" ".join(sorted(set(partners))),
                                 agree_n=len(set(partners)), **rd))
    out = pd.DataFrame(rows)
    for c in READINGS_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out = unique_tf_names(out)
    front = [c for c in READINGS_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def unique_tf_names(readings: pd.DataFrame) -> pd.DataFrame:
    """`transfer_function` made unique within a site and component by appending the run on a shared name.

    One row of the table is one file, keyed by its run and its stamp. Two runs of a site can hold the same
    kind, rate and selection -- a workbook 03 short-rate run and a workbook 05 form of the same name -- and a
    choice cell naming it would then be ambiguous, so the name carries the run.
    """
    if not len(readings) or "transfer_function" not in readings.columns:
        return readings
    d = readings.copy()
    key = ["site", "component", "transfer_function"]
    files = d.groupby(key).path.nunique()
    clash = {k for k, n in files.items() if n > 1}
    if not clash:
        return d
    d["transfer_function"] = [
        ("%s@%s" % (r.transfer_function, r.run)
         if (r.site, r.component, r.transfer_function) in clash else r.transfer_function)
        for r in d.itertuples()]
    return d


def agreement_matrix(tfs: pd.DataFrame, read=None, comp="xy", agree_band=AGREE_BAND,
                     agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE) -> pd.DataFrame:
    """Every pair of transfer functions of one site and component: the two departures and the agreement."""
    read = read or TFN.read_tf
    rows = []
    for (site,), grp in tfs.groupby(["site"], sort=True):
        items = [r for r in grp.itertuples() if r.on_disk]
        curves = {}
        for k, r in enumerate(items):
            try:
                curves[k] = read(r.path)
            except Exception:
                curves[k] = None
        for m in range(len(items)):
            for n in range(m + 1, len(items)):
                a, b = items[m], items[n]
                if curves[m] is None or curves[n] is None:
                    continue
                s = agree(curves[m], curves[n], comp, agree_band[0], agree_band[1],
                          agree_rho, agree_phase)
                rows.append(dict(site=site, component=comp, a=tf_label(a), b=tf_label(b),
                                 kind_a=a.kind, kind_b=b.kind,
                                 same_kind=bool(a.kind == b.kind), **s))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the transfer function of record

def _rate_words(rates) -> str:
    return "every rate's" if rates is None else " and ".join("%g Hz" % float(r) for r in rates)


def transfer_function_of_record(readings: pd.DataFrame, agree_band=AGREE_BAND, band=QUALITY_BAND,
                      rates=(1.0,)) -> pd.DataFrame:
    """Per site and component the chosen row with the reason, or none with the reason there is none.

    The pool is the rows at the delivery rate that pass the three response tests and agree with at least one
    row of another reference kind. The choice inside it is the smallest bar over `band`, ties broken by the
    longest period held. A component whose pool is empty has no transfer function of record, and the row says
    whether the passing rows did not corroborate each other or none passed at all.

    `rates` is the delivery rate or rates the choice is made among; None reads every rate in the table. The
    default is the 1 Hz row, because a 10 Hz pass of this survey stops near 1,200 s and the row a long-period
    survey delivers has to cover the delivery band: the 10 Hz row enters through the splice, where its short
    end joins the chosen row at 16 s, and not as the whole row.
    """
    rows = []
    for (site, comp), g in readings.groupby(["site", "component"], sort=True):
        ok = g[(g.status == "ok")]
        if rates is not None:
            ok = ok[ok.rate_hz.isin([float(r) for r in rates])]
        # forms.csv says which forms may be delivered: a form that does not beat its control buys efficiency
        # and not a different answer, and an inter-site impedance is drawn for comparison and not delivered
        if "candidate" in ok.columns:
            ok = ok[(ok.form == "") | ok.candidate.astype(bool)]
        sound = ok[ok.passes.astype(bool)]
        pool = sound[sound.agree_n > 0]
        alt = "; ".join("%s %.4f" % (tf_label(r), r.bar) for r in
                        sound.sort_values("bar").itertuples() if np.isfinite(r.bar)) or "none"
        if len(pool):
            # the smallest bar, ties by the longest period held: two curves of one record can carry the same
            # bar to the digit a table prints, and the one that reaches further is the one to deliver
            best = pool.sort_values(["bar", "held_hi_s"], ascending=[True, False]).iloc[0]
            tied = pool[np.isclose(pool.bar, best.bar, rtol=1e-9, atol=0)]
            why = ("the smallest bar over %g-%g s among the %s rows that pass the three response tests and "
                   "agree with another kind over %g-%g s%s"
                   % (band[0], band[1], _rate_words(rates), agree_band[0], agree_band[1],
                      ("; %d tied on the bar and the longest period held broke it"
                       % len(tied)) if len(tied) > 1 else ""))
            rows.append(dict(site=site, component=comp, transfer_function=best["transfer_function"],
                             kind=best.kind, kind_word=best.kind_word,
                             selection=best.get("selection", TFN.WHOLE_SELECTION), form=best.form,
                             run=best.run, stamp=best.stamp, rate_hz=best.rate_hz, bar=best.bar,
                             n_periods=best.n_periods, held_lo_s=best.held_lo_s,
                             held_hi_s=best.held_hi_s, held_n=best.held_n, passes=True, fails="",
                             agree_n=int(best.agree_n), agree_kinds=best.agree_kinds,
                             reproducible=best.get("reproducible", ""), why=why, alternatives=alt,
                             flagged="", note="", path=best.path))
            continue
        if len(sound):
            why = ("no transfer function of record: %d %s row(s) pass the three response tests and none "
                   "agrees with a row of another kind over %g-%g s (%s)"
                   % (len(sound), _rate_words(rates), agree_band[0], agree_band[1],
                      ", ".join(sorted(set(sound.kind)))))
        elif len(ok):
            reasons = sorted({str(r.fails).split(";")[0] for r in ok.itertuples()})
            why = ("no transfer function of record: no %s row passes the three response tests (%s)"
                   % (_rate_words(rates), "; ".join(reasons[:3])))
        else:
            why = ("no transfer function of record: no %s row of this site and component could be read"
                   % _rate_words(rates))
        rows.append(dict(site=site, component=comp, transfer_function="none", kind="", kind_word="",
                         selection="", form="", run="", stamp="", rate_hz=np.nan, bar=np.nan, n_periods=0,
                         held_lo_s=np.nan, held_hi_s=np.nan, held_n=0, passes=False, fails="",
                         agree_n=0, agree_kinds="", reproducible="", why=why, alternatives=alt,
                         flagged="", note="", path=""))
    out = pd.DataFrame(rows)
    for c in RECORD_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[RECORD_COLUMNS]


def candidates(record: pd.DataFrame) -> pd.DataFrame:
    """The distinct rows the record chose, one each: what the split-half passes are run over."""
    cols = ["site", "kind", "rate_hz", "selection", "form", "run", "stamp"]
    chosen = record[record["transfer_function"] != "none"]
    if not len(chosen):
        return pd.DataFrame(columns=cols + ["components"])
    g = (chosen.groupby(cols)["component"].apply(lambda v: " ".join(sorted(v))).reset_index()
         .rename(columns={"component": "components"}))
    return g.sort_values(["site", "kind"]).reset_index(drop=True)
