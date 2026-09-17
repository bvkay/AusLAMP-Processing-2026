"""The 10 Hz short end joined to the 1 Hz row at 16 s, where the step at the join holds on both ruled bands.

Ported from scripts/processing/vic_splice.py (D:/BEN/MTH5_Aurora_mt-io_2026), with its thresholds as
arguments and their frozen values as defaults.

THE DECOMPOSITION FIRST. Before any row is joined, the 10 Hz path is read against the 1 Hz path of the same
kind over LEVEL_BAND (4-32 s): the median of that ratio over the kinds is the RATE effect and the spread
across kinds at one rate is the KIND effect. A survey whose rate effect exceeds RATE_PATH_MAX_PCT cannot be
spliced at all, because the join would deliver the difference between two processing paths as a bend in the
earth. Aurora at 10 Hz reads about 8 per cent low at 4-32 s against its own 1 Hz transfer function (AusLAMP
Victoria, 2026-09-11), so this is the measurement that decides whether the survey has a short end to deliver.

The gate is read over the whole-record 10 Hz pass and the control stretch, and over no other row: a stretch
of the most coherent hours is not comparable to a 1 Hz transfer function of the whole record, so its
departure over 4-32 s carries the selection as well as the rate, and a gate read over it would refuse the
join for a difference the join does not deliver. Every 10 Hz row is in the decomposition table beside the
gate.

THE ACCEPTANCE. The step at the join is measured on two bands and the worse one governs: STEP_BELOW (8-16 s,
where the delivered row would be the 10 Hz one) and STEP_ABOVE (32-100 s, where it is the 1 Hz one). A row
whose worse step exceeds SPLICE_MAX_STEP_PCT is REPORTED NOT SPLICED and keeps its 1 Hz row untouched. The
16-32 s octave is not scored: SPLICE_GUARD (18-36 s) holds the Earth Data logger's 20.6 s instrument line,
which is not an earth response and is reproducible only to 5-14 per cent between honest processing paths, so
a 2 per cent criterion there would measure the line and not the join. The guard band is measured and printed
per row and scored by nothing.

THE CONTROL GATE, before the step test. The 10 Hz pass runs on the longest coherent stretch and never on the
whole record, so a site carries two 10 Hz transfer functions per kind: `stretch` is the longest run of whole
UTC hours with both recorded lines above 0.5 at 20-200 s, and `control` is a run of the same length placed at
random elsewhere in the record. A stretch is eligible only where it beats that kind's control on the
CONTROL_BAR_BAND (2-16 s) impedance bar by CONTROL_MARGIN: a selection that does not beat a random one of the
same length buys efficiency, not a different answer. The whole-record 10 Hz pass is a control and not a
selection, and is admitted without the gate; the control itself is never promoted. A stretch whose kind has
no control is UNJUDGED on the gate and is not eligible, and the table says so rather than passing it.

THE SELECTION, inside what the control gate and the acceptance leave, among the 10 Hz rows that pass the
three response tests of `readings`:
    xy   on its LEVEL over LEVEL_BAND against the row it will join: the smallest departure from one, and
         inside LEVEL_MAX_PCT
    yx   on its own error bar at the short end: the smallest impedance bar over SHORT_BAR_BAND, and under
         SHORT_BAR_MAX

THE FLOOR. Nothing below SHORT_FLOOR_S is delivered.

THE FILE. The spliced grid is the union of the base file's periods and the 10 Hz periods used. The union is
deduplicated on a ROUNDED KEY and keeps the ORIGINAL period value: rounding the values themselves moved one
base period by 2.3e-7 of itself, and apparent resistivity carries T, so every unspliced row shifted by the
same amount (vic_splice.py, its test S5). `unspliced_unchanged` is that test: every period and every value
of a component that was not spliced must come back from the written file identical to the input.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import transfer_functions as TFN, readings as RD
from .process import KIND_WORD

SPLICE_JOIN_S = 16.0               # the period the 10 Hz row joins the 1 Hz row at, in s
SPLICE_MAX_STEP_PCT = 2.0          # the step at the join, in per cent, on the worse of the two bands
SPLICE_GUARD = (18.0, 36.0)        # measured and printed per row and scored by nothing, in s
SHORT_FLOOR_S = 0.6                # nothing below this is delivered, in s
STEP_BELOW = (8.0, 16.0)           # the band below the join the step is scored on, in s
STEP_ABOVE = (32.0, 100.0)         # ... and the band above it
LEVEL_BAND = (4.0, 32.0)           # the band the xy row is selected on and the rate path is read over, in s
LEVEL_MAX_PCT = 5.0                # the widest level departure an xy row may carry into the selection
SHORT_BAR_BAND = (2.0, 10.0)       # the band the yx row's short-end bar is read over, in s
SHORT_BAR_MAX = 0.25               # the impedance bar there, at most this (the frozen 0.50 on the rho bar)
RATE_PATH_MAX_PCT = 4.0            # a 10 Hz path further than this from its 1 Hz path cannot be joined
MIN_BAND_POINTS = 2                # the periods a band needs each side before its ratio is read
NEAREST_LOG_TOL = 0.01             # a 10 Hz period lands on a grid point within this in natural log
PERIOD_KEY_DP = 6                  # the decimals the union key is rounded to; the VALUE is never rounded
IDENTITY_TOL = 1e-12               # an unspliced period and value must come back within this, relatively

CONTROL_SELECTION = "control"      # the random stretch of the same length a selection must beat
CONTROL_BAR_BAND = (2.0, 16.0)     # the band the control gate reads the bar over, in s
CONTROL_MARGIN = 0.20              # ... and the fraction of the control's bar a selection must beat it by
WHOLE_SELECTION = "whole"          # the whole-record 10 Hz pass: a control, admitted as a candidate

SPLICE_COLUMNS = ["site", "component", "spliced", "kind", "kind_word", "selection", "form",
                  "control_verdict",
                  "bar_2_16", "control_bar_2_16", "step_pct", "step_below_pct", "step_above_pct",
                  "guard_pct", "level_4_32_pct", "short_bar", "passes", "join_s", "shortest_period_s",
                  "n_short_periods", "why", "all_kinds", "file"]


def band_ratio(short, base, comp, lo, hi, min_points=MIN_BAND_POINTS):
    """(the median rho_short / rho_base over lo-hi s, the count), ported from vic_splice.band_ratio.

    The short curve is interpolated onto the BASE file's periods inside the band, linearly in log period of
    log rho, with no extrapolation. The count is what survives, and `min_points` is 2: the bands the step is
    scored on are an octave and a decade wide and a long-period grid carries two or three periods in an
    octave, so a four-point minimum leaves the band below the join unmeasurable and the rule reading one
    band where it states two.
    """
    pb, rb, _eb, _pb, _peb = RD.curve(base, comp)
    ps, rs, _es, _ps, _pes = RD.curve(short, comp)
    mb = (pb >= lo) & (pb <= hi) & np.isfinite(rb) & (rb > 0)
    ms = np.isfinite(rs) & (rs > 0)
    if mb.sum() < min_points or ms.sum() < min_points:
        return np.nan, 0
    order = np.argsort(ps[ms])
    lr = np.interp(np.log(pb[mb]), np.log(ps[ms][order]), np.log(rs[ms][order]),
                   left=np.nan, right=np.nan)
    v = np.exp(lr) / rb[mb]
    v = v[np.isfinite(v)]
    return (float(np.median(v)), int(v.size)) if v.size else (np.nan, 0)


def _ratio_pct(a, b, comp, lo, hi):
    """(100 x (rho_a / rho_b - 1), n) over one band, `a` interpolated onto `b`'s periods there."""
    r, n = band_ratio(a, b, comp, lo, hi)
    return (100.0 * (r - 1.0) if np.isfinite(r) else np.nan), n


# ------------------------------------------------------------------ the rate against the kind

def decompose(tfs: pd.DataFrame, read=None, band=LEVEL_BAND,
              rate_max_pct=RATE_PATH_MAX_PCT) -> pd.DataFrame:
    """Per site, kind, selection and component, the 10 Hz path against the 1 Hz path of the same kind.

    One row per pair that exists, read over `band`. `rate_pct` is the 10 Hz level as a per cent departure
    from the 1 Hz level: the number the acceptance rule of the splice is a guard against. The 1 Hz row of a
    kind is its whole-record pass, so each 10 Hz stretch of that kind is read against the same row.
    """
    read = read or TFN.read_tf
    rows = []
    for site, grp in tfs.groupby("site", sort=True):
        on = grp[grp.on_disk]
        if "selection" not in on.columns:
            on = on.assign(selection=TFN.WHOLE_SELECTION)
        if "form" not in on.columns:
            on = on.assign(form="")
        for kind in sorted(set(on.kind)):
            a = on[(on.kind == kind) & (on.rate_hz == 1.0) & (on.form == "")]
            b = on[(on.kind == kind) & (on.rate_hz == 10.0)]
            if not len(a) or not len(b):
                continue
            one = read(a.iloc[0].path)
            for r in b.itertuples():
                ten = read(r.path)
                for comp in RD.COMPONENTS:
                    pct, n = _ratio_pct(ten, one, comp, band[0], band[1])
                    rows.append(dict(site=site, kind=kind, kind_word=KIND_WORD.get(kind, kind),
                                     selection=str(getattr(r, "selection", TFN.WHOLE_SELECTION)),
                                     form=str(getattr(r, "form", "") or ""),
                                     component=comp, band_s="%g-%g" % band, rate_pct=pct, n=n,
                                     inside=bool(np.isfinite(pct) and abs(pct) <= rate_max_pct)))
    return pd.DataFrame(rows, columns=["site", "kind", "kind_word", "selection", "form", "component",
                                       "band_s", "rate_pct", "n", "inside"])


GATE_SELECTIONS = (WHOLE_SELECTION, CONTROL_SELECTION)   # the passes the rate gate is read over


def rate_versus_kind(dec: pd.DataFrame, rate_max_pct=RATE_PATH_MAX_PCT,
                     gate_selections=GATE_SELECTIONS) -> pd.DataFrame:
    """Per component: the median rate effect over every site and kind, and the spread across kinds.

    The rate effect is what the splice would deliver as a bend; the kind spread is how much of the same
    number is the choice of reference rather than the rate.

    The gate is read over `gate_selections` only -- the whole-record 10 Hz pass and the control stretch. A
    stretch of the most coherent hours is not comparable to a 1 Hz pass of the whole record: its departure
    over 4-32 s carries the selection as well as the rate, and reading the gate over it would refuse the
    splice for a difference the join does not deliver. Every row stays in the table beside the gate,
    `in_gate` saying which ones the median was taken over.
    """
    dec = dec.copy()
    if "selection" not in dec.columns:
        dec["selection"] = WHOLE_SELECTION
    dec["in_gate"] = dec.selection.isin(list(gate_selections))
    rows = []
    for comp, g in dec.groupby("component", sort=True):
        gated = g[g.in_gate]
        vals = np.asarray(gated.rate_pct[np.isfinite(gated.rate_pct)], float)
        spread = gated.groupby(["site"]).rate_pct.apply(
            lambda v: float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).sum() > 1 else np.nan)
        rows.append(dict(component=comp, n=int(len(vals)), n_rows=int(len(g)),
                         gate_selections=" ".join(gate_selections),
                         rate_pct_median=(float(np.median(vals)) if len(vals) else np.nan),
                         rate_pct_worst=(float(vals[int(np.argmax(np.abs(vals)))]) if len(vals)
                                         else np.nan),
                         kind_spread_pct_median=(float(np.nanmedian(spread)) if len(spread) else np.nan),
                         can_splice=bool(len(vals) and abs(float(np.median(vals))) <= rate_max_pct)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the step at the join

def step_at_join(base, short, comp, join=SPLICE_JOIN_S, below=STEP_BELOW, above=STEP_ABOVE,
                 guard=SPLICE_GUARD, level_band=LEVEL_BAND) -> dict:
    """The step the 10 Hz row would make at the join, on both ruled bands, with the guard band beside it.

    `base` is the 1 Hz curve and `short` the 10 Hz one. Every number is the 10 Hz level as a per cent
    departure from the 1 Hz level over that band. `step_pct` is the worse of the two ruled bands, which is
    what the acceptance reads; `guard_pct` is measured and scored by nothing.
    """
    lo_pct, n_lo = _ratio_pct(short, base, comp, below[0], below[1])
    hi_pct, n_hi = _ratio_pct(short, base, comp, above[0], above[1])
    g_pct, n_g = _ratio_pct(short, base, comp, guard[0], guard[1])
    lvl, n_lvl = _ratio_pct(short, base, comp, level_band[0], level_band[1])
    cand = [v for v in (lo_pct, hi_pct) if np.isfinite(v)]
    worst = max(cand, key=abs) if cand else np.nan
    return dict(join_s=float(join), step_pct=worst, step_below_pct=lo_pct, step_above_pct=hi_pct,
                n_below=n_lo, n_above=n_hi, guard_pct=g_pct, n_guard=n_g,
                guard_band_s="%g-%g" % guard, level_4_32_pct=lvl, n_level=n_lvl)


# ------------------------------------------------------------------ the selection

def control_gate(shorts: dict, comp: str, control=CONTROL_SELECTION, band=CONTROL_BAR_BAND,
                 margin=CONTROL_MARGIN, whole=WHOLE_SELECTION) -> dict:
    """{key: the gate's reading} over one site's 10 Hz transfer functions and one component.

    A key is (kind, selection) or (kind, selection, form); a form is controlled by its own kind's and
    form's control stretch, so a variant cache is never judged against the original's control.

    A selection is eligible only where its own bar over `band` is at most (1 - margin) of that control's.
    The whole-record pass carries no selection to control and passes without the gate; the control itself
    is never promoted; a selection with no control of its own is UNJUDGED and not eligible.
    """
    bars = {key: RD.bar(tf, comp, band[0], band[1]) for key, (tf, _p) in shorts.items()}
    out = {}
    for key in shorts:
        kind, sel = key[0], key[1]
        rest = tuple(key[2:])
        own = bars.get(key, np.nan)
        ctrl = bars.get((kind, control) + rest, np.nan)
        if sel == whole:
            out[key] = dict(eligible=True, bar_2_16=own, control_bar_2_16=np.nan,
                            verdict="the whole-record pass: a control, admitted without the gate")
        elif sel == control:
            out[key] = dict(eligible=False, bar_2_16=own, control_bar_2_16=own,
                            verdict="the random control itself: never promoted")
        elif not (np.isfinite(own) and np.isfinite(ctrl) and ctrl > 0):
            out[key] = dict(eligible=False, bar_2_16=own, control_bar_2_16=ctrl,
                            verdict="UNJUDGED: this row has no %s of its own to beat" % control)
        else:
            wins = bool(own <= (1.0 - float(margin)) * ctrl)
            out[key] = dict(
                eligible=wins, bar_2_16=own, control_bar_2_16=ctrl,
                verdict=("beats the %s control on the %g-%g s bar by at least %.0f %% (%.4f against %.4f)"
                         % (control, band[0], band[1], 100 * margin, own, ctrl)) if wins else
                        ("does NOT beat the %s control on the %g-%g s bar by %.0f %% (%.4f against %.4f): "
                         "efficiency, not a different answer"
                         % (control, band[0], band[1], 100 * margin, own, ctrl)))
    return out


def select_rows(site, base, shorts: dict, readings: pd.DataFrame, join=SPLICE_JOIN_S,
                max_step_pct=SPLICE_MAX_STEP_PCT, below=STEP_BELOW, above=STEP_ABOVE, guard=SPLICE_GUARD,
                level_band=LEVEL_BAND, level_max_pct=LEVEL_MAX_PCT, short_bar_band=SHORT_BAR_BAND,
                short_bar_max=SHORT_BAR_MAX, control=CONTROL_SELECTION, control_band=CONTROL_BAR_BAND,
                control_margin=CONTROL_MARGIN, rate_ok=None,
                rate_max_pct=RATE_PATH_MAX_PCT) -> dict:
    """{component: the chosen row or None} with the reason, over one site's 10 Hz transfer functions.

    `shorts` is {(kind, selection, form): (TFData, path)} of the site's 10 Hz transfer functions and
    `readings` is the readings table, which carries the response-test verdict each of them was given. The
    gate runs first, then the acceptance -- the step at the join and the short-end bar guard -- and the
    selection runs inside what the two leave.

    `rate_ok` is {component: the survey's rate path is inside rate_max_pct}, from `rate_versus_kind`. A
    component whose survey-wide 10 Hz path sits further than that from its 1 Hz path is refused before any
    row is read: the join would deliver the difference between two processing paths as a bend in the earth.
    """
    out = {}
    for comp in RD.COMPONENTS:
        if rate_ok is not None and not rate_ok.get(comp, True):
            out[comp] = dict(pick=None, scored=[], all_kinds="",
                             why=("NOT SPLICED: the survey's 10 Hz path sits further than %.0f %% from its "
                                  "1 Hz path over %g-%g s, so no row of this component can be joined"
                                  % (rate_max_pct, level_band[0], level_band[1])))
            continue
        gate = control_gate(shorts, comp, control, control_band, control_margin)
        scored = []
        for key, (tf, path) in sorted(shorts.items()):
            kind, sel = key[0], key[1]
            form = key[2] if len(key) > 2 else ""
            row = readings[(readings.site == site) & (readings.component == comp)
                           & (readings.kind == kind) & (readings.rate_hz == 10.0)
                           & (readings.selection == sel) & (readings.form == form)]
            sound = bool(row.passes.iloc[0]) if len(row) else False
            s = step_at_join(base, tf, comp, join, below, above, guard, level_band)
            g = gate[key]
            s.update(kind=kind, kind_word=KIND_WORD.get(kind, kind), selection=sel, form=form,
                     passes=sound, file=str(path), tf=tf, eligible=g["eligible"],
                     control_verdict=g["verdict"], bar_2_16=g["bar_2_16"],
                     control_bar_2_16=g["control_bar_2_16"],
                     short_bar=RD.bar(tf, comp, short_bar_band[0], short_bar_band[1]))
            scored.append(s)
        all_kinds = "; ".join(
            "%s/%s%s passes=%s control=%s level=%s step=%s bar=%s"
            % (s["kind"], s["selection"], ("/" + s["form"]) if s["form"] else "", s["passes"],
               ("pass" if s["eligible"] else "refused"),
               ("%+.1f%%" % s["level_4_32_pct"]) if np.isfinite(s["level_4_32_pct"]) else "-",
               ("%+.1f%%" % s["step_pct"]) if np.isfinite(s["step_pct"]) else "-",
               ("%.3f" % s["short_bar"]) if np.isfinite(s["short_bar"]) else "-") for s in scored)
        sound = [s for s in scored if s["passes"] and s["eligible"]]
        pool = [s for s in sound
                if np.isfinite(s["step_pct"]) and abs(s["step_pct"]) <= max_step_pct
                and np.isfinite(s["short_bar"]) and s["short_bar"] <= short_bar_max]
        best, why = None, ""
        if not scored:
            why = "no 10 Hz transfer function of this site"
        elif not [s for s in scored if s["passes"]]:
            why = "no 10 Hz row passes the three response tests"
        elif not sound:
            unjudged = [s for s in scored if s["passes"] and "UNJUDGED" in s["control_verdict"]]
            why = ("NOT SPLICED: no 10 Hz row that passes the response tests holds the control gate (%s)"
                   % ("UNJUDGED: no %s control exists yet for %s"
                      % (control, ", ".join(sorted({s["kind"] for s in unjudged}))) if unjudged
                      else "every selection is refused by its random control"))
        elif not pool:
            held = [s for s in sound if np.isfinite(s["step_pct"]) and abs(s["step_pct"]) <= max_step_pct]
            if held:
                bg = min(held, key=lambda s: s["short_bar"] if np.isfinite(s["short_bar"]) else np.inf)
                why = ("NOT SPLICED by the short-end bar guard, not by the step: %s holds the step at "
                       "%+0.1f %% but its %g-%g s impedance bar is %.3f, over the %.2f ceiling"
                       % (bg["kind_word"], bg["step_pct"], short_bar_band[0], short_bar_band[1],
                          bg["short_bar"], short_bar_max))
            else:
                w = min((s for s in sound if np.isfinite(s["step_pct"])),
                        key=lambda s: abs(s["step_pct"]), default=None)
                why = ("NOT SPLICED: no sound kind holds the step at the join inside %.0f %% (best %s, "
                       "%+0.1f %% -- %+0.1f above the join, %+0.1f below it)"
                       % (max_step_pct, w["kind_word"] if w else "none",
                          w["step_pct"] if w else np.nan,
                          w["step_above_pct"] if w else np.nan, w["step_below_pct"] if w else np.nan))
        elif comp == "xy":
            ok = [s for s in pool if np.isfinite(s["level_4_32_pct"])
                  and abs(s["level_4_32_pct"]) <= level_max_pct]
            if ok:
                best = min(ok, key=lambda s: abs(s["level_4_32_pct"]))
                why = ("xy on its %g-%g s level: %s on %s, %+0.1f %% of the row it joins; step at the "
                       "join %+0.1f %%" % (level_band[0], level_band[1], best["kind_word"],
                                           best["selection"], best["level_4_32_pct"], best["step_pct"]))
            else:
                why = ("NOT SPLICED: no row that holds the step also holds its %g-%g s level inside %.0f %%"
                       % (level_band[0], level_band[1], level_max_pct))
        else:
            best = min(pool, key=lambda s: s["short_bar"])
            why = ("yx on its short-end bar: %s on %s, %g-%g s impedance bar %.3f; step at the join "
                   "%+0.1f %%" % (best["kind_word"], best["selection"], short_bar_band[0],
                                  short_bar_band[1], best["short_bar"], best["step_pct"]))
        out[comp] = dict(pick=best, why=why, all_kinds=all_kinds, scored=scored)
    return out


# ------------------------------------------------------------------ the splice itself

def union_grid(base_periods, short_periods, floor=SHORT_FLOOR_S, join=SPLICE_JOIN_S) -> np.ndarray:
    """The base periods with the 10 Hz periods in [floor, join) added, deduplicated on a rounded key.

    The key is rounded and the VALUE is not: rounding the values moved the base file's shortest period by
    2.3e-7 of itself and every unspliced row with it (vic_splice.py test S5).
    """
    keys = {}
    for t in list(np.asarray(base_periods, float)) + [float(t) for t in short_periods
                                                      if floor <= float(t) < join]:
        k = round(float(t), PERIOD_KEY_DP)
        if k not in keys:
            keys[k] = float(t)
    return np.array(sorted(keys.values()), float)


def first_delivered_period(period, z, comp) -> float:
    """The shortest period at which the delivered row actually carries a value.

    Not the shortest period above the floor: the union grid holds base periods the spliced row has
    superseded, and a 10 Hz period only lands on a grid point within NEAREST_LOG_TOL, so reading the floor
    instead of the row's own finite mask put a period in the file that the row did not carry (vic_splice
    test V6).
    """
    i, j = TFN.COMPONENTS[comp]
    m = np.isfinite(np.asarray(z)[:, i, j])
    return float(np.min(np.asarray(period, float)[m])) if m.any() else np.nan


def splice(base_path, picks: dict, out_path, join=SPLICE_JOIN_S, floor=SHORT_FLOOR_S,
           guard=SPLICE_GUARD, tol=NEAREST_LOG_TOL, extra_lines=()) -> dict:
    """Write the spliced file: the 10 Hz row below the join for each picked component, the base above it.

    `picks` is {component: the row select_rows chose}, each carrying `file`. A component with no pick keeps
    its base row at every base period. The diagonal and the tipper stay the base file's, on the base periods.
    A file with no pick at all is copied through unchanged so that not one delivered value can move.
    """
    from mt_metadata.transfer_functions.core import TF

    base_path, out_path = Path(base_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks = {c: v for c, v in (picks or {}).items() if v}
    if not picks:
        if base_path.resolve() != out_path.resolve():
            shutil.copyfile(base_path, out_path)
        return dict(path=str(out_path), spliced=[], periods_added=0, lines=[], copied=True)

    tf0 = TF(fn=str(base_path))
    tf0.read()
    p0 = np.asarray(tf0.period, float)
    z0 = np.asarray(tf0.impedance.values, complex)
    e0 = np.asarray(tf0.impedance_error.values, float)
    has_tip = bool(tf0.has_tipper() and tf0.tipper is not None)
    t0 = np.asarray(tf0.tipper.values, complex) if has_tip else None
    te0 = np.asarray(tf0.tipper_error.values, float) if has_tip else None

    used = []
    shorts = {}
    for comp, pick in picks.items():
        tfm = TF(fn=str(pick["file"]))
        tfm.read()
        pm = np.asarray(tfm.period, float)
        shorts[comp] = (pm, np.asarray(tfm.impedance.values, complex),
                        np.asarray(tfm.impedance_error.values, float))
        used += [t for t in pm if floor <= t < join]
    pnew = union_grid(p0, used, floor, join)
    zn = np.full((len(pnew), 2, 2), np.nan + 0j)
    en = np.full((len(pnew), 2, 2), np.nan)
    i0 = {round(float(t), PERIOD_KEY_DP): k for k, t in enumerate(p0)}

    lines = []
    for comp, (row, col) in (("xy", (0, 1)), ("yx", (1, 0))):
        pick = picks.get(comp)
        for j, t in enumerate(pnew):
            k = i0.get(round(float(t), PERIOD_KEY_DP))
            if k is not None and (pick is None or t >= join):
                zn[j, row, col] = z0[k, row, col]
                en[j, row, col] = e0[k, row, col]
        if pick is None:
            continue
        pm, zm, em = shorts[comp]
        n_short = 0
        for j, t in enumerate(pnew):
            if not (floor <= t < join):
                continue
            k = int(np.argmin(np.abs(np.log(pm) - np.log(t))))
            if abs(np.log(pm[k]) - np.log(t)) < tol:
                zn[j, row, col] = zm[k, row, col]
                en[j, row, col] = em[k, row, col]
                n_short += 1
        pick["n_short_periods"] = n_short
        pick["shortest_period_s"] = first_delivered_period(pnew, zn, comp)
        lines.append("splice_%s=below %g s from the 10 Hz %s transfer function on the %s selection (%s), "
                     "above from the 1 Hz row; step at the join %+0.1f %% (%+0.1f above, %+0.1f below); "
                     "%d period(s) taken; shortest delivered period %.3f s"
                     % (comp, join, pick["kind_word"], pick.get("selection", "whole-record"),
                        Path(pick["file"]).name, pick["step_pct"], pick["step_above_pct"],
                        pick["step_below_pct"], n_short, pick["shortest_period_s"]))
    for row, col in ((0, 0), (1, 1)):
        for j, t in enumerate(pnew):
            k = i0.get(round(float(t), PERIOD_KEY_DP))
            if k is not None:
                zn[j, row, col] = z0[k, row, col]
                en[j, row, col] = e0[k, row, col]
    lines.append("splice_guard=%g-%g s carries the Earth Data logger's 20.6 s instrument line and is "
                 "measured, printed and scored by nothing" % guard)
    lines.append("splice_floor=nothing below %g s is delivered" % floor)
    lines += [str(x) for x in extra_lines]

    tn = TF()
    tn.survey_metadata = tf0.survey_metadata
    tn.station_metadata = tf0.station_metadata
    tn.period = pnew
    tn.impedance = zn
    tn.impedance_error = en
    if has_tip:
        tp = np.full((len(pnew),) + t0.shape[1:], np.nan + 0j)
        tpe = np.full((len(pnew),) + t0.shape[1:], np.nan)
        for j, t in enumerate(pnew):
            k = i0.get(round(float(t), PERIOD_KEY_DP))
            if k is not None:
                tp[j] = t0[k]
                tpe[j] = te0[k]
        tn.tipper = tp
        tn.tipper_error = tpe
    try:
        st = tn.station_metadata
        from .final import clean
        st.transfer_function.processing_parameters = list(
            st.transfer_function.processing_parameters or []) + [clean(x) for x in lines]
        tn.station_metadata = st
    except Exception:
        pass
    try:
        tn.write(fn=str(out_path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tn.write(fn=str(out_path), file_type="edi")
    return dict(path=str(out_path), spliced=sorted(picks), periods_added=int(len(pnew) - len(p0)),
                lines=lines, copied=False)


def unspliced_unchanged(base_path, out_path, spliced=(), tol=IDENTITY_TOL) -> dict:
    """Every period and value of a component that was NOT spliced, read back against the input.

    vic_splice's test S5, which caught a 2.3e-7 shift in every unspliced row. The base periods must all be
    present in the written file and each unspliced component's value at each of them must be identical to
    the input within `tol` relative.
    """
    base, out = TFN.read_tf(base_path), TFN.read_tf(out_path)
    pb = np.asarray(base.period, float)
    po = np.asarray(out.period, float)
    idx = np.array([int(np.argmin(np.abs(po - t))) for t in pb]) if len(po) else np.zeros(0, int)
    d_period = (np.abs(po[idx] / pb - 1.0) if len(idx) else np.zeros(0))
    worst_period = float(np.max(d_period)) if len(d_period) else np.nan
    rows, worst_value, worst_where = [], 0.0, ""
    for comp, (i, j) in TFN.COMPONENTS.items():
        if comp in set(spliced):
            continue
        a = base.z[:, i, j]
        b = out.z[idx, i, j] if len(idx) else np.zeros(0, complex)
        m = np.isfinite(a)
        n_lost = int((m & ~np.isfinite(b)).sum()) if len(b) else int(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(b[m] - a[m]) / np.maximum(np.abs(a[m]), 1e-30) if len(b) else np.zeros(0)
        w = float(np.nanmax(rel)) if len(rel) and np.isfinite(rel).any() else 0.0
        if w > worst_value:
            worst_value, worst_where = w, comp
        rows.append(dict(component=comp, n=int(m.sum()), n_lost=n_lost, worst_relative=w))
    ok = bool(worst_period <= tol and worst_value <= tol
              and not any(r["n_lost"] for r in rows) and len(pb) and len(po) >= len(pb))
    return dict(ok=ok, worst_period_relative=worst_period, worst_value_relative=worst_value,
                worst_component=worst_where, tol=tol, components=rows,
                n_base_periods=int(len(pb)), n_out_periods=int(len(po)))


def splice_table(rows, out_path=None) -> pd.DataFrame:
    """SPLICE.csv: one row per site and component with the kind, both steps and the shortest period."""
    t = pd.DataFrame(rows)
    for c in SPLICE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    t = t[SPLICE_COLUMNS]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out_path, index=False)
    return t
