"""One composition of the catalogue's methods into one product: a frame, a rule per row, and the assembly.

The catalogue sections of workbook 05 each change one thing and score it. The recipe composes them: a frame
for the whole product, and per impedance row a selection of hours and a sample rate, so that one row can come
from the whole record at 1 Hz and the other from a short window at 10 Hz. The rows are passed separately and
assembled into one tensor, with the tipper taken from the pass the recipe names.

THE FRAME. `native` is the site's own sensor frame after the package's mean-field rotation; `diagonal` is the
arm diagonal of site.centre, so every row's pass reads cache_<rate>hz_ne and is turned back by
theta = atan2(-L_E, L_N) on the written file. Both rows run in the recipe's frame, so the assembled product
is in one frame and no row is turned after the assembly.

THE HOURS of a row:

    whole             the record.
    f05, f10, f25     workbook 03's hour selections (process.selection), scored on this frame's cache at this
                      row's rate over SCORE_BAND_S = 1-30 s, each with its r25 control -- the same number of
                      hours drawn without replacement from the same pool under the seed.
    window:coherent   the longest contiguous stretch in which both recorded electric lines read a coherence
                      with the magnetic field they couple to -- Ex with Hy, Ey with Hx -- above
                      WINDOW_COH = 0.5 over WINDOW_BAND_S = 20-200 s, per whole UTC hour on the 1 Hz cache as
                      laid. The lines are the recorded ones and not the frame's: the window exists to find
                      where both electrodes were measuring, which is a fact about the electrodes and not
                      about the frame they are later combined in. The 1 Hz cache is read whatever the row's
                      rate, because 200 s is measured on the long record and not on the short one.
    window:<start>/<h>  a stretch the caller names, as an ISO UTC start and a count of hours.

    A window's control is a stretch of the same length placed at random elsewhere in the record under the
    named seed (site.masks.random_block), so the two cost the same and the difference between their products
    is what the window bought.

THE RULE the window rule reproduces (the AusLAMP Queensland Phase 2 campaign at Q58N, 2026-09-07:
PROJECT_JOURNAL.md :1678-1683, Q58N_EXPLANATION_2026-09-07.md :44-46): the longest stretch with both recorded
lines above 0.5 at 20-200 s is 2026-03-22 12:00 to 2026-03-23 02:00 UTC, 14 h.

THE ASSEMBLY. The x row supplies Zx'x' and Zx'y' and the y row Zy'x' and Zy'y'. The x row's product is the
base, so its period grid, its station block and its position carry through; the y row is matched onto that
grid where the two grids are the same and interpolated in log period where they are not, and a period the y
row's grid does not reach carries the EDI empty-data value 1e32 rather than a number. products.read_tf masks
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
from .masks import band_coherence, random_block

WINDOW_COH = 0.5                # both recorded lines must read above this to be inside a coherent window
WINDOW_BAND_S = (20.0, 200.0)   # ... over this band
WINDOW_NPERSEG = 1024           # the Welch segment of the hourly coherence at 1 Hz, masks.COH_NPERSEG
WINDOW_RATE = 1                 # the rate the window is chosen on, whatever rate the row is passed at
HOUR_S = 3600.0
EDI_FILL = 1e32                 # the EDI empty-data value a period the y row does not reach is written as

RECORDED_PAIRS = (("xy", "Ex", "Hy"), ("yx", "Ey", "Hx"))
ROW_COMPONENT = {"x": "xy", "y": "yx"}      # the off-diagonal element each row carries
ROW_ELEMENTS = {"x": ((0, 0), (0, 1)), "y": ((1, 0), (1, 1))}
ROW_LABEL = {"x": "x'", "y": "y'"}
FRAMES = ("native", "diagonal")
SELECTION_KEYS = ("f05", "f10", "f25")
CONTROL_KEY = "r25"
GRID_RTOL = 1e-9


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def _hm(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- the hourly coherence and the window

def hour_coherence(sv, site, rate=WINDOW_RATE, band_s=WINDOW_BAND_S, nperseg=WINDOW_NPERSEG,
                   hour_s=HOUR_S, arrays=None, t0=None) -> pd.DataFrame:
    """One row per whole UTC hour: the coherence of each recorded line with the H it couples to.

    The record is read as laid -- no sign and no frame -- because a sign flips a coherence's phase and not
    its magnitude. An hour holding a non-finite sample in a pair scores NaN on that pair and cannot be inside
    a window, which is what keeps a gap out of a stretch that is called coherent.
    """
    from .masks import load_raw
    if arrays is None:
        t0, arrays = load_raw(sv, site, rate, ("Hx", "Hy", "Ex", "Ey"))
    fs = float(rate)
    n = len(arrays["Hx"])
    step = int(round(float(hour_s) * fs))
    starts, t_starts = SEL.hour_grid(int(t0), n, fs, hour_s)
    rows = []
    for i0, ta in zip(starts, t_starts):
        row = dict(t_start=int(round(ta)), t_end=int(round(ta + float(hour_s))), utc=_hm(ta))
        for comp, line, hchan in RECORDED_PAIRS:
            e = np.asarray(arrays[line][i0:i0 + step], float)
            h = np.asarray(arrays[hchan][i0:i0 + step], float)
            row["coh_%s" % comp] = (band_coherence(e, h, fs, band_s, nperseg)
                                    if (np.isfinite(e).all() and np.isfinite(h).all()) else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows, columns=["t_start", "t_end", "utc", "coh_xy", "coh_yx"])
    out.attrs["band_s"] = list(band_s)
    out.attrs["nperseg"] = int(nperseg)
    out.attrs["rate_hz"] = float(rate)
    return out


def runs_above(table: pd.DataFrame, coh_min=WINDOW_COH) -> list:
    """[(hours, t_start, t_end)] of every contiguous stretch with both recorded lines above `coh_min`.

    Longest first, and the earliest of any that tie, so a record carrying several stretches of one length
    returns the same one on every run.
    """
    a = np.asarray(table.coh_xy, float)
    b = np.asarray(table.coh_yx, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > float(coh_min)) & (b > float(coh_min))
    ts = np.asarray(table.t_start, float)
    te = np.asarray(table.t_end, float)
    out, k = [], 0
    while k < len(ok):
        if not ok[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        out.append((int(j - k + 1), int(ts[k]), int(te[j])))
        k = j + 1
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def coherent_window(table: pd.DataFrame, coh_min=WINDOW_COH) -> dict:
    """The longest contiguous stretch with both recorded lines above `coh_min`, with its reason."""
    runs = runs_above(table, coh_min)
    a = np.asarray(table.coh_xy, float)
    b = np.asarray(table.coh_yx, float)
    scored = int((np.isfinite(a) & np.isfinite(b)).sum())
    above = int(((a > float(coh_min)) & (b > float(coh_min))).sum())
    band = table.attrs.get("band_s", list(WINDOW_BAND_S))
    if not runs:
        return dict(rule="window:coherent", t_start=None, t_end=None, hours=0, n_runs=0,
                    n_hours_above=above, n_hours_scored=scored, coh_min=float(coh_min), runs=[],
                    reason="no hour reads above %.2f on both recorded lines over %g-%g s (%d hour(s) scored)"
                           % (coh_min, band[0], band[1], scored))
    hours, ta, tb = runs[0]
    return dict(rule="window:coherent", t_start=int(ta), t_end=int(tb), hours=int(hours), n_runs=len(runs),
                n_hours_above=above, n_hours_scored=scored, coh_min=float(coh_min), runs=runs,
                reason="the longest contiguous stretch with both recorded lines above %.2f over %g-%g s: "
                       "%d h, %s to %s UTC, of %d hour(s) above on %d scored"
                       % (coh_min, band[0], band[1], hours, _hm(ta), _hm(tb), above, scored))


def named_window(spec: str) -> dict:
    """`window:<ISO UTC start>/<hours>` as a window dict, or a dict carrying the reason it does not parse."""
    body = str(spec).split(":", 1)[1] if ":" in str(spec) else ""
    start, _, hours = body.rpartition("/")
    try:
        ta = int(pd.Timestamp(start.strip(), tz="UTC").timestamp())
        h = float(hours)
        if h <= 0:
            raise ValueError("the hour count is not positive")
    except Exception as exc:
        return dict(rule=str(spec), t_start=None, t_end=None, hours=0,
                    reason="%s does not read as window:<ISO UTC start>/<hours> (%s)" % (spec, exc))
    return dict(rule=str(spec), t_start=ta, t_end=int(ta + h * HOUR_S), hours=float(h),
                reason="the stretch named in the recipe: %.4g h from %s UTC" % (h, _hm(ta)))


def window_mask(t0, n, window, fs=1.0) -> np.ndarray:
    """The sample mask of one window on a record of `n` samples starting at `t0`, at `fs`."""
    keep = np.zeros(int(n), bool)
    a = max(0, int(round((float(window["t_start"]) - float(t0)) * float(fs))))
    b = min(int(n), int(round((float(window["t_end"]) - float(t0)) * float(fs))))
    if b > a:
        keep[a:b] = True
    return keep


def control_window(t0, n, window, seed, fs=1.0) -> dict:
    """A stretch of the same length placed at random elsewhere in the record, under `seed`."""
    length = int(round((float(window["t_end"]) - float(window["t_start"])) * float(fs)))
    a, b = random_block(int(t0), int(n), length, int(seed), fs=float(fs),
                        exclude=(int(round((window["t_start"] - t0) * fs)),
                                 int(round((window["t_end"] - t0) * fs))))
    ta = int(round(t0 + a / float(fs)))
    tb = int(round(t0 + b / float(fs)))
    return dict(t_start=ta, t_end=tb, hours=round((tb - ta) / HOUR_S, 3), seed=int(seed),
                reason="a stretch of the same length placed at random elsewhere in the record, seed %d"
                       % int(seed))


# ---------------------------------------------------------------- the frame and the selections

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


def selection_hours(sv, site, key, rate=1, variant="", seed=SEL.SEED, elines=None,
                    control_key=CONTROL_KEY) -> dict:
    """One of workbook 03's hour selections and its r25 control, scored on this cache at this rate.

    The score table and the selections are computed in memory and never written: the site's own
    hour_scores_10hz.csv and hour_selection_10hz.json belong to workbook 03's 10 Hz lane, and a selection
    scored on a variant cache at another rate is not that file's content.
    """
    key = str(key)
    if key not in SELECTION_KEYS:
        raise ValueError("the selection is %s; it is one of %s" % (key, ", ".join(SELECTION_KEYS)))
    work = Path(sv.cfg["work_root"])
    d = work / ("cache_%dhz%s" % (int(rate), ("_" + variant) if variant else ""))
    z = np.load(d / ("%s.npz" % site), allow_pickle=False)
    t0 = int(z["t0"][0])
    arrays = {c: np.asarray(z[c], float) for c in ("Hx", "Hy", "Ex", "Ey")}
    z.close()
    n = len(arrays["Hx"])
    if elines is None:
        p = work / site / "elines.csv"
        elines = pd.read_csv(p) if p.exists() else None
    table = SEL.score_hours(t0, arrays, float(rate), elines=elines).round(6)
    del arrays
    fraction = float(key[1:]) / 100.0
    sel = SEL.selections(table, t0, n, float(rate), fractions=(fraction,),
                         random_fraction=float(control_key[1:]) / 100.0, seed=int(seed))
    out = {}
    for tag in (key, control_key):
        v = sel[tag]
        out[tag] = dict(tag=tag, fraction=v["fraction"], random=v["random"], seed=v["seed"],
                        threshold=v["threshold"], n_hours=v["n_hours"], n_candidates=v["n_candidates"],
                        hours=v["hours"], days=round(v["n_hours"] / 24.0, 3),
                        keep=SEL.mask_from_hours(v["hours"], t0, n, float(rate)))
    out["t0"] = t0
    out["n"] = n
    out["rate_hz"] = float(rate)
    out["scored"] = int(table.score.notna().sum())
    out["band_s"] = list(SEL.SCORE_BAND_S)
    return out


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
    """The `recipe=` lines an assembled product carries: one per row, then the tipper and the frame."""
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
    out.append("recipe_tipper=from the %s row's pass" % str(recipe.get("tipper")))
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
    """One product from two row passes: the x row's file as the base, the y row's two rows written into it.

    The x row's product carries the period grid, the station block, the position and every header line, so
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
                  "rest carry the EDI empty value %g; the tipper is the %s row's"
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
    """Row by row, the rho ratio and the phase difference of two products over a band.

    The comparison of workbook 05's section 12: a product against one built outside this package, read on the
    four elements over one band. It is a comparison and never a verdict.
    """
    from .. import agreement as AG
    from ..products import COMPONENTS, rho_phase
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
