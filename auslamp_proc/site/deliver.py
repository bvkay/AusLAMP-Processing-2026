"""What a form delivers: the tipper-only product, its refusal test, and the forms table workbook 06 reads.

THE TIPPER-ONLY PRODUCT (ported from wamt_esp2026_products.py:339-401). A site with no deliverable impedance
can still deliver its tipper: the tipper is an H-only quantity and survives two dead electric lines. The
impedance rows stay NaN and the file's INFO lines say so -- "xy rows: no product of record" and "tipper from
the <kind> product" -- so a reader cannot take the empty rows for a measurement.

THE REFUSAL. A tipper is refused where the vertical channel is not measuring the vertical field. Two faults
say so and neither is visible in the tipper itself: Hz a COPY of a horizontal channel, which reads a
coherence of 1.00 with Hx and makes the tipper a re-statement of the horizontal record; and a LEAK, where Hz
carries the site's own horizontal field (0.95 with its own H at 1000-4000 s) while carrying nothing of a
neighbour's vertical field (0.00 with the neighbour's Hz). A real vertical field is coherent with the
neighbour's vertical field at those periods, because the source is regional.

THE FORMS TABLE is <work_root>/<site>/<run>_<stamp>/forms.csv: one row per form with the products it is read
against, the criterion in words, the verdict, and the reading -- the bar over 10-1000 s, the smoothness, the
agreement with the baseline per decade -- and whether the form is a candidate for workbook 06. A form is a
candidate only where it beats every control it carries on the bar by the stated margin: a selection whose
product does not beat its random control buys efficiency, not a different answer, and is not promoted. A form
that borrows both horizontal channels is an inter-site impedance and is never a candidate. A form assembled
from other forms carries `candidate_rule`, a {candidate, verdict} the caller states, because its controls
belong to the rows it was assembled from and this row's own bar cannot be read for them.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import agreement as AG
from ..products import COMPONENTS, OFF_DIAGONAL, read_tf

BAR_BAND_S = (10.0, 1000.0)
AGREE_BAND_S = (100.0, 1000.0)
BAR_MARGIN = 0.20               # a delivered selection must beat its control on the bar by this fraction
DECADES = ((5.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, 10000.0))

# the tipper refusal
EDI_FILL = 1e32                 # the EDI empty-data value a blanked impedance row is written as

COPY_COH = 0.99                 # Hz a copy of a horizontal channel
LEAK_OWN_COH = 0.95             # ... or coherent with its own H at 1000-4000 s
LEAK_NEIGHBOUR_COH = 0.10       # ... while carrying nothing of a neighbour's vertical field
LEAK_BAND_S = (1000.0, 4000.0)


# ---------------------------------------------------------------- the reading

def bar(tf, comp, lo_s=BAR_BAND_S[0], hi_s=BAR_BAND_S[1]) -> float:
    """The median relative impedance error of one component over a band: the product's bar.

    The bar always sits beside the shape call, because the shape rule alone passes on noise.
    """
    i, j = COMPONENTS[comp]
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.z)[:, i, j]
    e = np.asarray(tf.z_err, float)[:, i, j]
    m = (p >= lo_s) & (p <= hi_s) & np.isfinite(z) & np.isfinite(e) & (np.abs(z) > 0)
    return float(np.median(e[m] / np.abs(z[m]))) if m.any() else np.nan


def reading(tf, baseline=None, bands=DECADES, bar_band=BAR_BAND_S, agree_band=AGREE_BAND_S) -> dict:
    """The bar, the smoothness and the agreement with the baseline of one product, per component."""
    out = {}
    sm = AG.smoothness(tf)
    for comp in OFF_DIAGONAL:
        row = sm[sm.component == comp]
        out["bar_%s" % comp] = bar(tf, comp, *bar_band)
        out["jumps_per_decade_%s" % comp] = float(row.jumps_per_decade.iloc[0]) if len(row) else np.nan
        out["max_phase_step_%s" % comp] = (float(row.max_phase_step_deg_per_decade.iloc[0])
                                           if len(row) else np.nan)
        out["periods_%s" % comp] = int(row.n_periods.iloc[0]) if len(row) else 0
        if baseline is not None:
            s = AG.band_stats(tf, baseline, comp, agree_band[0], agree_band[1])
            out["rho_ratio_%s" % comp] = s["rho_ratio"]
            out["phase_diff_%s" % comp] = s["phase_diff_deg"]
            out["n_%s" % comp] = s["n"]
    if baseline is not None:
        per = AG.per_decade(tf, baseline, bands=bands)
        out["per_decade"] = "; ".join(
            "%s %s %.2f %+0.1f deg" % (r.band, r.component, r.rho_ratio, r.phase_diff_deg)
            for r in per.itertuples() if np.isfinite(r.rho_ratio))
    out["bar"] = float(np.nanmin([out.get("bar_xy", np.nan), out.get("bar_yx", np.nan)]))
    return out


def beats(value, control, margin=BAR_MARGIN) -> bool:
    """True where a bar beats its control's by at least `margin` of the control's value (lower is better)."""
    if not (np.isfinite(value) and np.isfinite(control)) or control <= 0:
        return False
    return bool(value <= (1.0 - float(margin)) * control)


# ---------------------------------------------------------------- the forms table

def forms_table(rows, out_path=None, baseline_path=None, bar_band=BAR_BAND_S,
                agree_band=AGREE_BAND_S, margin=BAR_MARGIN) -> pd.DataFrame:
    """One row per form with its controls, criterion, verdict and reading. Written to forms.csv.

    `rows` are the dicts run_form returned, each carrying `controls` as a semicolon-separated list of the
    form names it is read against. A form with a control is a candidate only where it beats every one of them
    on the bar by `margin`; a form with no control (the baseline, a demonstration, an inter-site form) is
    read and never promoted on this table alone.
    """
    made = {r["form"]: r for r in rows if r.get("status") in ("made", "exists")}
    tfs, reads = {}, {}
    base = None
    if baseline_path and Path(baseline_path).exists():
        base = read_tf(baseline_path)
    for form, r in made.items():
        p = Path(r["product"])
        if not p.exists():
            continue
        try:
            tfs[form] = read_tf(p)
        except Exception as exc:
            reads[form] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:80]))
            continue
        reads[form] = reading(tfs[form], base if form != "whole" else None, bar_band=bar_band,
                              agree_band=agree_band)
    out = []
    for r in rows:
        form = r["form"]
        rd = reads.get(form, {})
        controls = [c for c in str(r.get("controls") or "").split(";") if c]
        ctrl_bars = {c: reads.get(c, {}).get("bar", np.nan) for c in controls}
        this_bar = rd.get("bar", np.nan)
        judged = bool(controls) and any(np.isfinite(v) for v in ctrl_bars.values())
        wins = all(beats(this_bar, v, margin) for v in ctrl_bars.values() if np.isfinite(v)) if judged \
            else False
        if r.get("status") == "refused":
            # the method's own floor stating what the cache leaves, measured before the pass: a reading with
            # numbers, and not the same thing as an exception out of the estimator
            verdict = str(r.get("reason") or "refused before the pass")
        elif r.get("status") not in ("made", "exists"):
            verdict = "NOT MADE"
        elif r.get("inter_site"):
            verdict = "shown, never delivered: an inter-site impedance"
        elif r.get("judged_on") and r["judged_on"] != "bar":
            # a form whose criterion is a per-period statement is read against its control period by period
            # and never on the bar, which cannot say whether one band moved and the rest held
            verdict = ("read against %s on the %s, not on the bar"
                       % (";".join(controls) or "its original", r["judged_on"]))
        elif not controls:
            verdict = "read, no control: not promoted on this table"
        elif not judged:
            verdict = "UNJUDGED: no control product was made"
        else:
            verdict = ("beats its control(s) on the %g-%g s bar by at least %.0f %%"
                       % (bar_band[0], bar_band[1], 100 * margin)) if wins else \
                      ("does NOT beat its control(s) on the %g-%g s bar by %.0f %%: efficiency, not a "
                       "different answer" % (bar_band[0], bar_band[1], 100 * margin))
        rule = r.get("candidate_rule") or {}
        if rule:
            # a form assembled from other forms carries its own candidacy rule with the sentence that
            # states it: its controls were the passes its rows were made by, and this row's own bar --
            # the better of its two rows -- cannot say whether each of those rows beat its control
            wins = bool(rule.get("candidate"))
            verdict = str(rule.get("verdict") or verdict)
        out.append(dict(
            site=r.get("site"), form=form, kind=r.get("kind"), rate_hz=r.get("rate_hz"),
            product=Path(str(r.get("product"))).name, status=r.get("status"),
            controls=";".join(controls) or "none",
            control_products=";".join(Path(str(made[c]["product"])).name for c in controls if c in made),
            seed=r.get("seed"), criterion=r.get("criterion") or "",
            bar_10_1000=this_bar,
            control_bar=";".join("%s %.4f" % (c, v) for c, v in ctrl_bars.items() if np.isfinite(v)) or "",
            bar_xy=rd.get("bar_xy"), bar_yx=rd.get("bar_yx"),
            jumps_per_decade_xy=rd.get("jumps_per_decade_xy"),
            jumps_per_decade_yx=rd.get("jumps_per_decade_yx"),
            rho_ratio_xy=rd.get("rho_ratio_xy"), phase_diff_xy=rd.get("phase_diff_xy"),
            rho_ratio_yx=rd.get("rho_ratio_yx"), phase_diff_yx=rd.get("phase_diff_yx"),
            agreement_per_decade=rd.get("per_decade", ""),
            days=r.get("days"), n_runs=r.get("n_runs"), seconds=r.get("seconds"),
            judged_on=r.get("judged_on") or "bar", verdict=verdict,
            candidate=bool(wins and not r.get("inter_site") and r.get("judged_on") in (None, "bar")
                           and r.get("status") in ("made", "exists")),
            error=r.get("error") or rd.get("error") or ""))
    t = pd.DataFrame(out)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out_path, index=False)
    return t


# ---------------------------------------------------------------- the tipper

def tipper_refusal(sv, site, neighbour=None, band_s=LEAK_BAND_S, rate=1) -> dict:
    """The two faults that refuse a tipper: Hz a copy of a horizontal channel, and Hz a leak.

    Returns the three coherences and the call. A tipper is refused where Hz reads at or above 0.99 with its
    own Hx -- the channel is a copy and the tipper re-states the horizontal record -- or where it reads at
    or above 0.95 with its own H at 1000-4000 s while reading under 0.10 with a neighbour's Hz, which is the
    site's own horizontal field leaking into the vertical channel rather than a regional vertical field.
    """
    from .masks import band_coherence, distance_km, load_raw, prepare, span
    from ..look import highpass
    b, a = highpass(float(rate), 3000.0)
    t0, arr = load_raw(sv, site, rate, ("Hx", "Hy", "Hz"))
    n = len(arr["Hz"])
    prep = {c: prepare(arr[c], b, a) for c in ("Hx", "Hy", "Hz")}
    if any(v is None for v in prep.values()):
        return dict(site=site, judged=False, reason="a magnetic channel is under 80 per cent finite")
    coh_hz_hx = band_coherence(prep["Hz"], prep["Hx"], float(rate), band_s)
    coh_hz_h = max(coh_hz_hx, band_coherence(prep["Hz"], prep["Hy"], float(rate), band_s))
    coh_nb = np.nan
    if neighbour is None:
        pool = [s for s in sv.sites.site if s != site]
        if pool:
            neighbour = sorted(pool, key=lambda s: distance_km(sv, site, s))[0]
    if neighbour:
        try:
            tn, nb = load_raw(sv, neighbour, rate, ("Hz",))
            i0 = t0 - tn
            seg = nb["Hz"][i0:i0 + n] if 0 <= i0 and i0 + n <= len(nb["Hz"]) else None
            y = prepare(seg, b, a) if seg is not None else None
            if y is not None:
                coh_nb = band_coherence(prep["Hz"], y, float(rate), band_s)
        except Exception:
            coh_nb = np.nan
    copy = bool(np.isfinite(coh_hz_hx) and coh_hz_hx >= COPY_COH)
    leak = bool(np.isfinite(coh_hz_h) and coh_hz_h >= LEAK_OWN_COH
                and np.isfinite(coh_nb) and coh_nb < LEAK_NEIGHBOUR_COH)
    why = []
    if copy:
        why.append("Hz reads %.2f with its own Hx: a copy" % coh_hz_hx)
    if leak:
        why.append("Hz reads %.2f with its own H at %g-%g s and %.2f with %s's Hz: a leak"
                   % (coh_hz_h, band_s[0], band_s[1], coh_nb, neighbour))
    return dict(site=site, judged=True, neighbour=neighbour, band_s=list(band_s),
                coh_hz_hx=coh_hz_hx, coh_hz_own_h=coh_hz_h, coh_hz_neighbour_hz=coh_nb,
                refused=bool(copy or leak), reason="; ".join(why) or "neither fault fires")


def tipper_only(product, out_path, kind="", refusal=None) -> dict:
    """A tipper-only delivery: the impedance rows blanked, the tipper kept, the INFO lines saying so.

    The impedance rows are written as the EDI empty-data fill so that a reader cannot take them for a
    measurement, and two INFO lines name what the file is.
    """
    from mt_metadata.transfer_functions.core import TF
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if refusal and refusal.get("refused"):
        return dict(path=None, written=False, refused=True, reason=refusal.get("reason"))
    tf = TF(fn=str(product))
    tf.read()
    has_tipper = False
    try:
        has_tipper = bool(tf.has_tipper() and tf.tipper is not None)
    except Exception:
        has_tipper = False
    if not has_tipper:
        return dict(path=None, written=False, refused=True,
                    reason="the product carries no tipper: an H-only delivery has nothing to deliver")
    z = np.asarray(tf.impedance.values, complex)
    ze = np.asarray(tf.impedance_error.values, float)
    # the EDI empty-data value, not NaN: an all-NaN tensor leaves the file with no impedance block at all,
    # which a reader cannot tell from a file that was never given one. products.read_tf masks 1e32 per
    # component, so the rows come back empty and the block still says which periods the tipper covers.
    tf.impedance = np.full_like(z, EDI_FILL + 0j)
    tf.impedance_error = np.full_like(ze, EDI_FILL)
    try:
        st = tf.station_metadata
        pp = list(st.transfer_function.processing_parameters or [])
        # every processing_parameters line must be key=value: the EDI writer splits each one on the first
        # "=" and raises on a line without one
        pp += ["xy_rows=no product of record",
               "yx_rows=no product of record",
               "tipper=from the %s product %s" % (kind or "source", Path(product).name)]
        if refusal:
            pp.append("tipper_refusal_test=%s (Hz with own Hx %.2f, with own H %.2f, with %s's Hz %.2f)"
                      % (refusal.get("reason", ""), refusal.get("coh_hz_hx", float("nan")),
                         refusal.get("coh_hz_own_h", float("nan")), refusal.get("neighbour", "none"),
                         refusal.get("coh_hz_neighbour_hz", float("nan"))))
        st.transfer_function.processing_parameters = pp
    except Exception:
        pass
    try:
        tf.write(fn=str(out_path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(out_path), file_type="edi")
    back = read_tf(out_path)
    finite = {c: int(np.isfinite(back.z[:, i, j]).sum()) for c, (i, j) in COMPONENTS.items()}
    return dict(path=str(out_path), written=True, refused=False,
                impedance_finite=finite, has_tipper=bool(back.t is not None),
                reason="the impedance rows are blank and the tipper is the %s product's" % (kind or "source"))
