"""One composition of the catalogue's methods into one transfer function: a frame, a rule per row, the
assembly.

The catalogue sections of workbook 04 each change one thing and score it. The recipe composes them: a frame
for the whole file, and per impedance row a selection of hours and a sample rate, so that one row can come
from the whole record at 1 Hz and the other from a short window at 10 Hz. The rows are passed separately and
assembled into one tensor, with the tipper taken from the pass the recipe names.

THE FRAME. `native` is the site's own sensor frame after the package's mean-field rotation; `diagonal` is the
arm diagonal of site.centre, so every row's pass reads cache_<rate>hz_ne and is turned back by
theta = atan2(-L_E, L_N) on the written file. Both rows run in the recipe's frame, so the assembled file is
in one frame and no row is turned after the assembly.

THE HOURS of a row, one of three:

    whole               the record.
    window:coherent     the stretch process.selection chooses -- the longest contiguous run of whole UTC
                        hours in which the row's own recorded line reads above WINDOW_COH = 0.5 against the
                        magnetic channel it couples to over 20-200 s, cut to its best STRETCH_MAX_H = 48 h
                        where it runs longer. The x row is read on Ex against Hy and the y row on Ey against
                        Hx. The lines are the recorded ones and not the frame's: the stretch exists to find
                        where that electrode was measuring, which is a fact about the electrode and not
                        about the frame the rows are later combined in. The 1 Hz cache is read whatever the
                        row's rate, because 200 s is measured on the long record.
    window:<start>/<h>  a stretch the caller names, as an ISO UTC start and a count of hours.

    A stretch's control is a stretch of the same length placed at random elsewhere in the record under the
    named seed, so the two cost the same and the difference between their transfer functions is what the
    stretch bought.

THE ASSEMBLY. The x row supplies Zx'x' and Zx'y' and the y row Zy'x' and Zy'y'. The x row's file is the
base, so its period grid, its station block and its position carry through; the y row is matched onto that
grid where the two grids are the same and interpolated in log period where they are not, and a period the y
row's grid does not reach carries the EDI empty-data value 1e32 rather than a number. read_tf masks
that value per component, so the assembled file reads back with the y row empty above the y row's own longest
period and the x row measured throughout.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..process import selection as SEL
from . import masks as MK

HOUR_S = SEL.HOUR_S
EDI_FILL = 1e32                 # the EDI empty-data value a period the y row does not reach is written as

HOURS_WHOLE = "whole"           # the row is passed over the record
HOURS_COHERENT = "window:coherent"   # ... over the stretch the one rule chooses for that row's own line
HOURS_OPTIONS = (HOURS_WHOLE, HOURS_COHERENT, "window:<ISO UTC start>/<hours>")

ROW_COMPONENT = {"x": "xy", "y": "yx"}      # the off-diagonal element each row carries
ROW_ELEMENTS = {"x": ((0, 0), (0, 1)), "y": ((1, 0), (1, 1))}
ROW_LABEL = {"x": "x'", "y": "y'"}
FRAMES = ("native", "diagonal")
GRID_RTOL = 1e-9


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def _hm(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- the hours a row is passed on

def row_hours(sv, site, spec, row="x", seed=SEL.SEED, table=None, coh_min=SEL.WINDOW_COH,
              max_h=SEL.STRETCH_MAX_H, rate=SEL.SCORE_RATE) -> dict:
    """The stretch one row's `hours` names, with its control and the score table it was read from.

    `spec` is `whole`, `window:coherent` or `window:<ISO UTC start>/<hours>`. `window:coherent` calls
    process.selection.longest_stretch on that row's own recorded line, which is the one rule the package
    selects hours with; a named window is parsed and not scored. A row over the whole record carries no
    control, because there is nothing a stretch of the same length elsewhere would be compared with.
    """
    spec = str(spec).strip()
    if spec == HOURS_WHOLE:
        return dict(rule=HOURS_WHOLE, window=None, control=None, table=table, line=None,
                    reason="the whole record")
    line = SEL.ROW_LINE[str(row)]
    if spec == HOURS_COHERENT:
        table = SEL.site_scores(sv, site, rate=rate) if table is None else table
        window = SEL.longest_stretch(table, coh_min=coh_min, lines=(line,), max_h=max_h)
    elif spec.startswith("window:"):
        window = SEL.named_stretch(spec)
    else:
        raise ValueError("%s: the hours are one of %s" % (spec, ", ".join(HOURS_OPTIONS)))
    control = None
    if window.get("t_start") is not None:
        t0, n = MK.span(sv, site, int(rate))
        control = SEL.control_stretch(t0, n, window, seed=int(seed), fs=float(rate))
    return dict(rule=spec, window=window, control=control, table=table, line=line,
                reason=window.get("reason", ""))


# ---------------------------------------------------------------- the frame

def frame_spec(sv, site, frame) -> dict:
    """What a row's pass needs to run in `frame`: the cache variant, the turn and the angle.

    `native` reads the site's own cache and turns nothing. `diagonal` reads cache_<rate>hz_ne, whose electric
    channels are already signed and combined, and turns the written tensor's H columns by
    theta = atan2(-L_E, L_N).
    """
    frame = str(frame)
    if frame not in FRAMES:
        raise ValueError("the frame is %s; it is one of %s" % (frame, ", ".join(FRAMES)))
    if frame == "native":
        return dict(frame="native", variant="", apply_e_signs=True, turn_ne=False, turn_angle_deg=None,
                    theta_deg=0.0, note="the site's own sensor frame after the mean-field rotation")
    from .centre import arm_lengths, diagonal_angle, diagonal_length
    arms = arm_lengths(sv, site)
    if not arms["known"]:
        raise ValueError("%s: the arm lengths are not in sites.csv, so the diagonal has no direction" % site)
    theta = diagonal_angle(arms["L_N"], arms["L_E"])
    return dict(frame="diagonal", variant="ne", apply_e_signs=False, turn_ne=True, turn_angle_deg=theta,
                theta_deg=theta, L_N=arms["L_N"], L_E=arms["L_E"],
                separation_m=diagonal_length(arms["L_N"], arms["L_E"]),
                note="the arm diagonal: x' along (L_N Ex - L_E Ey)/d at %+.2f deg from north with L_N %.4g m "
                     "and L_E %.4g m, y' orthogonal to it" % (theta, arms["L_N"], arms["L_E"]))


# ---------------------------------------------------------------- what a row needs on disk

def missing_inputs(sv, site, rate, variant, kind) -> str:
    """The reason a row cannot be passed, or an empty string where every input is on disk.

    A 10 Hz row needs the 10 Hz cache of the site, the variant of it the frame reads, and the reference store
    of the baseline kind at that rate -- the store is the remote's own 10 Hz record on this site's grid, so a
    missing store is a missing remote cache. The single station is not a kind of this package, so a missing
    store refuses the row rather than falling back to one.
    """
    work = Path(sv.cfg["work_root"])
    why = []
    src = work / ("cache_%dhz" % int(rate)) / ("%s.npz" % site)
    if not src.exists():
        why.append("no %d Hz cache at %s" % (int(rate), src))
    elif variant:
        var = work / ("cache_%dhz_%s" % (int(rate), variant)) / ("%s.npz" % site)
        if not var.exists():
            why.append("no %d Hz %s variant cache at %s" % (int(rate), variant, var))
    store = work / "references" / ("%dhz" % int(rate)) / ("%s_%s.npz" % (kind, site))
    if not store.exists():
        why.append("no %d Hz %s reference store at %s, and the single station is not a kind of this package"
                   % (int(rate), kind, store))
    return "refused: " + "; ".join(why) if why else ""


# ---------------------------------------------------------------- the header lines

def recipe_lines(recipe: dict, rows: dict) -> list:
    """The `recipe=` lines an assembled file carries: one per row, then the tipper and the frame."""
    out = ["recipe_frame=%s" % str(recipe.get("frame"))]
    for key in ("x", "y"):
        r = rows.get(key) or {}
        w = r.get("window") or {}
        c = r.get("control_window") or {}
        out.append("recipe_%s=hours=%s, rate=%g Hz, frame=%s, window=%s, control=%s, seed=%s"
                   % (key, r.get("hours", "?"), float(r.get("rate", 0) or 0), str(recipe.get("frame")),
                      ("%s..%s (%.4g h)" % (_iso(w["t_start"]), _iso(w["t_end"]),
                                            (w["t_end"] - w["t_start"]) / HOUR_S))
                      if w.get("t_start") else "none: the whole record",
                      ("%s..%s" % (_iso(c["t_start"]), _iso(c["t_end"]))) if c.get("t_start")
                      else str(r.get("control") or "none"),
                      str(r.get("seed", "none"))))
    out.append("recipe_tipper=from the pass of the %s row" % str(recipe.get("tipper")))
    return out


# ---------------------------------------------------------------- the assembly

def _match(pf, pe, rtol=GRID_RTOL):
    """(the index of pe nearest each pf, the worst relative difference) where the grids are one, else None."""
    pf, pe = np.asarray(pf, float), np.asarray(pe, float)
    if not len(pf) or not len(pe):
        return None
    idx = np.array([int(np.argmin(np.abs(np.log(pe) - np.log(t)))) for t in pf])
    k = float(np.max(np.abs(pe[idx] / pf - 1.0)))
    return (idx, k) if k <= rtol else None


def _interp_block(pf, pe, ve, ee):
    """(values, errors, inside) of one block interpolated onto pf, linear in log period."""
    pf, pe = np.asarray(pf, float), np.asarray(pe, float)
    ve, ee = np.asarray(ve), np.asarray(ee, float)
    inside = (pf >= pe.min()) & (pf <= pe.max())
    vi = np.full((len(pf),) + ve.shape[1:], np.nan + 1j * np.nan, complex)
    ei = np.full((len(pf),) + ee.shape[1:], np.nan)
    lx, lq = np.log10(pe), np.log10(pf)
    for i in range(ve.shape[1]):
        for j in range(ve.shape[2]):
            vi[inside, i, j] = (np.interp(lq[inside], lx, np.real(ve[:, i, j]))
                                + 1j * np.interp(lq[inside], lx, np.imag(ve[:, i, j])))
            ei[inside, i, j] = np.interp(lq[inside], lx, ee[:, i, j])
    return vi, ei, inside


def assemble(x_edi, y_edi, out_edi, tipper="x", lines=(), fill=EDI_FILL, verbose=True) -> dict:
    """One transfer function from two row passes: the x row's file as the base, the y row's rows written in.

    The x row's file carries the period grid, the station block, the position and every header line, so
    the assembled file is the x row's file with Zy'x' and Zy'y' replaced. A period the y row's own grid does
    not reach carries `fill`, the EDI empty-data value, so a reader cannot take an absent row for a
    measurement. `tipper` names the row the tipper is taken from; the x row's is already in the base.
    """
    from mt_metadata.transfer_functions.core import TF
    out_edi = Path(out_edi)
    out_edi.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(x_edi), str(out_edi))
    tf = TF(fn=str(out_edi))
    tf.read()
    pf = np.asarray(tf.period, float)
    z = np.array(tf.impedance.values)
    e = np.array(tf.impedance_error.values)
    ty = TF(fn=str(y_edi))
    ty.read()
    pe = np.asarray(ty.period, float)
    zy = np.array(ty.impedance.values)
    ey = np.array(ty.impedance_error.values)
    m = _match(pf, pe)
    if m is not None:
        idx, kmax = m
        zi, ei, inside = zy[idx], ey[idx], np.ones(len(pf), bool)
        how = "the two grids are one to %.1e relative" % kmax
    else:
        zi, ei, inside = _interp_block(pf, pe, zy, ey)
        how = ("the grids differ (%d periods against %d): the y row was interpolated in log period onto the "
               "x row's grid" % (len(pf), len(pe)))
    for i, j in ROW_ELEMENTS["y"]:
        ok = inside & np.isfinite(zi[:, i, j]) & np.isfinite(ei[:, i, j])
        z[:, i, j] = np.where(ok, zi[:, i, j], fill + 0j)
        e[:, i, j] = np.where(ok, ei[:, i, j], fill)
    tf.impedance = z
    tf.impedance_error = e
    y_periods = pf[np.isfinite(np.asarray(z, complex)[:, 1, 0])
                   & (np.abs(np.asarray(z, complex)[:, 1, 0]) < fill)]
    took_tipper = False
    if str(tipper) == "y":
        try:
            if ty.has_tipper() and ty.tipper is not None:
                t_src = np.array(ty.tipper.values)
                te_src = np.array(ty.tipper_error.values)
                ti, tei, tin = _interp_block(pf, pe, t_src, te_src)
                tf.tipper = np.where(tin[:, None, None] & np.isfinite(ti), ti, fill + 0j)
                tf.tipper_error = np.where(tin[:, None, None] & np.isfinite(tei), tei, fill)
                took_tipper = True
        except Exception:
            took_tipper = False
    try:
        st = tf.station_metadata
        pp = list(st.transfer_function.processing_parameters or [])
        pp += [str(x) for x in lines]
        pp.append("recipe_assembly=Zy'x' and Zy'y' from %s, %s; %d of %d period(s) carry the y row and the "
                  "rest carry the EDI empty value %g; the tipper is the pass of the %s row"
                  % (Path(y_edi).name, how, int(len(y_periods)), int(len(pf)), fill,
                     "y" if took_tipper else "x"))
        st.transfer_function.processing_parameters = pp
    except Exception:
        pass
    try:
        tf.write(fn=str(out_edi), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(out_edi), file_type="edi")
    if verbose:
        print("   assembled %s: %s; the y row covers %d of %d period(s)"
              % (out_edi.name, how, int(len(y_periods)), int(len(pf))))
    return dict(path=str(out_edi), how=how, n_periods=int(len(pf)), n_y_periods=int(len(y_periods)),
                y_min_s=(float(y_periods.min()) if len(y_periods) else float("nan")),
                y_max_s=(float(y_periods.max()) if len(y_periods) else float("nan")),
                tipper_from=("y" if took_tipper else "x"), fill=float(fill))


def row_comparison(a, b, lo_s=5.0, hi_s=100.0) -> pd.DataFrame:
    """Row by row, the rho ratio and the phase difference of two transfer functions over a band.

    A transfer function against one built outside this package, read on the four elements over one band.
    It is a comparison and never a verdict.
    """
    from .. import agreement as AG
    from ..transfer_functions import COMPONENTS, rho_phase
    bg = AG.on_grid(a, b)
    rows = []
    for comp, (i, j) in COMPONENTS.items():
        ra, _ea, pa, _fa = rho_phase(a.period, a.z, a.z_err, comp)
        rb, _eb, pb, _fb = rho_phase(a.period, bg.z, bg.z_err, comp)
        m = (a.period >= float(lo_s)) & (a.period <= float(hi_s)) & np.isfinite(ra) & np.isfinite(rb)
        rows.append(dict(element="Z%s" % comp, row=("x" if i == 0 else "y"), n=int(m.sum()),
                         rho_ratio=(float(np.median(rb[m] / ra[m])) if m.any() else np.nan),
                         phase_diff_deg=(float(np.median(pb[m] - pa[m])) if m.any() else np.nan)))
    return pd.DataFrame(rows)
