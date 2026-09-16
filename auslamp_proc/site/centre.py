"""The shared centre electrode, and the north-minus-east diagonal that cancels it.

The EDL L layout is three electrodes: a shared centre C and two arms, so Ex = (V_N - V_C)/L and
Ey = (V_E - V_C)/L and a noisy centre puts one voltage on both lines. With the lines physically signed the
model is

    Ex = Ex_true + c,   Ey = Ey_true + s c,    s = +1 for arms N+E or S+W, -1 for one arm reversed,

and the E signs predict s: the convention (Ex -1, Ey -1) is N+E, each departure to +1 reverses an arm, so
s = +1 for an even number of departures and -1 for an odd one. An undecided sign is never filled by
convention: the site is UNJUDGED on the sign prediction.

The residual test (ported from vic_centre_test.day_stats :80-99, criteria :6-24). On the days of highest
Ex-Ey coherence, at 20-200 s, the part of each line explained by (Hx, Hy) is removed per frequency bin from
the cross-spectra and the residuals are read: the model holds when the residual coherence is at least 0.9 and
the complex gain g = S_ry,rx / S_rx,rx has |g| in 0.85-1.18, and sign(Re g) is the observed s. The clean
diagonal is the one whose multiple coherence with (Hx, Hy) is the higher. The three criteria:

    A  where the model holds and both E signs are decided, the prediction must agree with the observed sign
    B  a CONTROL site whose source Ex-Ey coherence is under 0.35 on every scored day must NOT hold the model
    C  the remedy applies only where the model holds and the clean diagonal by H is the observed one

The remedy (ported from vic_ne_cache :6-13 and vic_ne_rotate :6-21). V_N - V_E = L (Ex - Ey) is the voltage
across the diagonal of length L sqrt 2, so E_x' = (Ex - Ey)/sqrt 2 is the field along the north-west diagonal
and E_y' = (Ex + Ey)/sqrt 2 along the north-east one, carrying the centre doubled. The pair is E in the frame
turned by -45 deg, so a pass on the variant cache gives R Z; turning the H columns as well gives
Z' = (R Z) R^T and T' = T R^T. The x' row is the clean one and the y' row is kept for the record.

The turn-back is checked on three invariants of a column-only turn: the elements to 1e-6 relative (an EDI
carries seven significant digits), |det Z' - det Z| <= 1e-5 ||Z||^2 scaled by the norm because the relative
form is meaningless at a near-singular period, and the Frobenius norm to 1e-6. The trace is NOT an invariant
of a column-only turn and is reported as the counter-example.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..look import highpass
from ..process import frame as FR
from .masks import DAY, load_raw, prepare

NPERSEG = 4096
BAND_S = (20.0, 200.0)
CENTRE_DAYS = 3
RESID_COH_MIN = 0.9
GAIN_LO, GAIN_HI = 0.85, 1.18
CONTROL_EX_EY_MAX = 0.35        # a control site's source Ex-Ey coherence on every scored day
SQ2 = float(np.sqrt(2.0))
THETA_NE = -45.0

# the tolerances of the turn-back, vic_ne_rotate.py:12-21
ELEMENT_RTOL = 1e-6
DET_SCALED_TOL = 1e-5
FROBENIUS_RTOL = 1e-6


def _cross(ch, nperseg=NPERSEG, fs=1.0):
    from scipy import signal
    names = list(ch)
    S, f = {}, None
    for i, p in enumerate(names):
        for q in names[i:]:
            f, P = signal.csd(ch[p], ch[q], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            S[(p, q)] = P
            S[(q, p)] = np.conj(P)
    return f, S


def day_stats(ex, ey, hx, hy, band_s=BAND_S, nperseg=NPERSEG, fs=1.0) -> dict:
    """The residual coherence, the complex gain and the two diagonals' H coherence over one piece.

    S_r1,r2 = S_12 - S_1h Shh^-1 S_h2 is the cross-spectrum of the residuals after the H-explained part is
    removed; the model's observable is their coherence and the gain g = S_ry,rx / S_rx,rx. Ported from
    vic_centre_test.day_stats (:80-99).
    """
    f, S = _cross(dict(ex=ex, ey=ey, hx=hx, hy=hy), nperseg, fs)
    sel = (f > 1.0 / band_s[1]) & (f < 1.0 / band_s[0])
    if sel.sum() < 2:
        return {}
    Shh = np.stack([np.stack([S[("hx", "hx")], S[("hx", "hy")]], -1),
                    np.stack([S[("hy", "hx")], S[("hy", "hy")]], -1)], -2)[sel]
    try:
        Sinv = np.linalg.inv(Shh)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(Shh)

    def seh(e):
        return np.stack([S[(e, "hx")], S[(e, "hy")]], -1)[sel]

    def she(e):
        return np.stack([S[("hx", e)], S[("hy", e)]], -1)[sel]

    def resid(e1, e2):
        return S[(e1, e2)][sel] - np.einsum("ni,nij,nj->n", seh(e1), Sinv, she(e2))

    rxx, ryy, rxy = resid("ex", "ex").real, resid("ey", "ey").real, resid("ex", "ey")
    coh_r = float(np.median(np.abs(rxy) ** 2 / (rxx * ryy)))
    g = np.conj(rxy) / rxx
    gain = float(np.median(np.abs(g)))
    s_obs = float(np.sign(np.median(g.real)))

    def mcoh(w):
        """The multiple coherence of the combination ex + w ey with (hx, hy)."""
        See = (S[("ex", "ex")][sel].real + w * w * S[("ey", "ey")][sel].real
               + 2 * w * S[("ex", "ey")][sel].real)
        Seh = seh("ex") + w * seh("ey")
        She = she("ex") + w * she("ey")
        return float(np.median(np.einsum("ni,nij,nj->n", Seh, Sinv, She).real / See))

    return dict(coh_r=coh_r, gain=gain, s_obs=s_obs, mcoh_diff=mcoh(-1.0), mcoh_sum=mcoh(1.0),
                coh_ExEy=float(np.median(np.abs(S[("ex", "ey")][sel]) ** 2
                                         / (S[("ex", "ex")][sel].real * S[("ey", "ey")][sel].real))))


def sign_prediction(decision_row) -> tuple:
    """(s_pred, [undecided channels]). An undecided sign is never filled by convention.

    vic_centre_test.py:117-124: filling an undecided sign with the wave's majority and then scoring the sign
    prediction against the guess is a test that cannot fail.
    """
    und, vals = [], {}
    for ch, col in (("Ex", "sign_ex"), ("Ey", "sign_ey")):
        v, decided = FR.read_sign(None if decision_row is None else decision_row.get(col))
        vals[ch] = v
        if not decided:
            und.append(ch)
    if und:
        return float("nan"), und
    dep = int(vals["Ex"] > 0) + int(vals["Ey"] > 0)
    return (-1.0 if dep % 2 else 1.0), []


def residual_test(sv, site, days=CENTRE_DAYS, elines=None, band_s=BAND_S, nperseg=NPERSEG,
                  rate=1) -> dict:
    """The shared-centre model at one site, on the `days` days of highest Ex-Ey coherence.

    Returns the medians over those days, whether the model holds, the predicted and observed signs, the two
    diagonals' H coherence, whether the site qualifies as a control (source Ex-Ey coherence under 0.35 on
    every scored day) and whether the remedy applies.
    """
    if elines is None:
        elines = read_elines(sv, site)
    b, a = highpass(float(rate), 3000.0)
    t0, arr = load_raw(sv, site, rate, ("Ex", "Ey", "Hx", "Hy"))
    dec = None
    try:
        dec = sv.decision(site)
    except KeyError:
        dec = None
    s_pred, undecided = sign_prediction(dec)
    stats, used = [], []
    table = elines[elines.coh_Ex_Ey.notna()].sort_values("coh_Ex_Ey", ascending=False) if len(elines) \
        else elines
    for r in table.itertuples():
        if len(stats) >= int(days):
            break
        i0, i1 = int(float(r.t_start) - t0), int(float(r.t_end) - t0)
        if i0 < 0 or i1 > len(arr["Ex"]):
            continue
        p = {k: prepare(arr[k][i0:i1], b, a) for k in ("Ex", "Ey", "Hx", "Hy")}
        if any(v is None for v in p.values()):
            continue
        st = day_stats(p["Ex"], p["Ey"], p["Hx"], p["Hy"], band_s, nperseg, float(rate))
        if not st:
            continue
        st["day"], st["src"] = r.day, float(r.coh_Ex_Ey)
        stats.append(st)
        used.append(r.day)
    if not stats:
        return dict(site=site, judged=False, days="", reason="no usable day",
                    s_pred=s_pred, sign_undecided=" ".join(undecided))
    med = {k: float(np.median([d[k] for d in stats]))
           for k in ("coh_r", "gain", "mcoh_diff", "mcoh_sum", "coh_ExEy", "src")}
    s_obs = float(np.sign(np.sum([d["s_obs"] for d in stats]))) or 1.0
    model = bool(med["coh_r"] >= RESID_COH_MIN and GAIN_LO <= med["gain"] <= GAIN_HI)
    control = bool(max(d["src"] for d in stats) < CONTROL_EX_EY_MAX)
    clean_obs = "diff" if s_obs > 0 else "sum"
    clean_pred = "" if undecided else ("diff" if s_pred > 0 else "sum")
    clean_H = "diff" if med["mcoh_diff"] > med["mcoh_sum"] else "sum"
    return dict(site=site, judged=True, days=" ".join(used), n_days=len(stats),
                src_coh=med["src"], resid_coh=med["coh_r"], resid_gain=med["gain"],
                s_pred=s_pred, s_obs=s_obs, sign_undecided=" ".join(undecided),
                a_judged=bool(not undecided and model),
                mcoh_diff=med["mcoh_diff"], mcoh_sum=med["mcoh_sum"],
                model_holds=model, control=control,
                clean_pred=clean_pred, clean_obs=clean_obs, clean_by_H=clean_H,
                sign_agrees=(None if undecided or not model else bool(s_obs == s_pred)),
                remedy_applicable=bool(model and clean_H == clean_obs))


def read_elines(sv, site) -> pd.DataFrame:
    """<work_root>/<site>/elines.csv, the per-day electric-line table workbook 02 wrote."""
    p = Path(sv.cfg["work_root"]) / site / "elines.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def built_control(sv, site, neighbour=None, days=CENTRE_DAYS, elines=None, band_s=BAND_S,
                  nperseg=NPERSEG, rate=1, members=None) -> dict:
    """The residual test on a pair of lines that CANNOT share a centre electrode, as the control.

    The site's own Ex is paired with the nearest sound site's Ey, both read against the site's own (Hx, Hy)
    over the same days, and the residual test is run on that pair. Two electrodes tens of kilometres apart
    have no common voltage, so the model must NOT hold: a residual coherence at or above 0.9 with a gain
    inside 0.85-1.18 there would mean the test finds a shared centre wherever it looks.

    This is the control the section is judged on (Ben, after the Q53N run of 2026-09-17). A site whose own
    Ex-Ey coherence stays under 0.35 on every day need not exist in a survey -- a one-dimensional earth
    correlates the two lines through the source field alone, and no AusLAMP Queensland Phase 1 site clears
    that ceiling, the lowest daily maximum being 0.354 -- so a criterion written on finding one is UNJUDGED
    wherever the survey is layered. The built pair exists at every site with a neighbour.
    """
    if elines is None:
        elines = read_elines(sv, site)
    pool = [s for s in (members if members is not None else list(sv.sites.site)) if s != site]
    if neighbour is None:
        from .masks import distance_km
        ranked = []
        for s in sorted(pool, key=lambda x: distance_km(sv, site, x)):
            el = read_elines(sv, s)
            if not len(el) or "Ey_state" not in el.columns:
                continue
            if float((el.Ey_state == "dead").mean()) < 0.5:
                ranked.append(s)
        if not ranked:
            return dict(site=site, judged=False, model_holds=None, neighbour=None,
                        reason="no other site carries an Ey that is alive on most days")
        neighbour = ranked[0]
    b, a = highpass(float(rate), 3000.0)
    t0, own = load_raw(sv, site, rate, ("Ex", "Hx", "Hy"))
    tn, other = load_raw(sv, neighbour, rate, ("Ey",))
    table = elines[elines.coh_Ex_Ey.notna()].sort_values("coh_Ex_Ey", ascending=False) if len(elines) \
        else elines
    stats, used = [], []
    for r in table.itertuples():
        if len(stats) >= int(days):
            break
        i0, i1 = int(float(r.t_start) - t0), int(float(r.t_end) - t0)
        j0, j1 = int(float(r.t_start) - tn), int(float(r.t_end) - tn)
        if i0 < 0 or i1 > len(own["Ex"]) or j0 < 0 or j1 > len(other["Ey"]):
            continue
        p = {k: prepare(own[k][i0:i1], b, a) for k in ("Ex", "Hx", "Hy")}
        p["Ey"] = prepare(other["Ey"][j0:j1], b, a)
        if any(v is None for v in p.values()):
            continue
        st = day_stats(p["Ex"], p["Ey"], p["Hx"], p["Hy"], band_s, nperseg, float(rate))
        if not st:
            continue
        st["day"] = r.day
        stats.append(st)
        used.append(r.day)
    if not stats:
        return dict(site=site, judged=False, model_holds=None, neighbour=neighbour,
                    reason="no day carries both the site's Ex and %s's Ey" % neighbour)
    med = {k: float(np.median([d[k] for d in stats]))
           for k in ("coh_r", "gain", "mcoh_diff", "mcoh_sum", "coh_ExEy")}
    holds = bool(med["coh_r"] >= RESID_COH_MIN and GAIN_LO <= med["gain"] <= GAIN_HI)
    return dict(site="%s Ex + %s Ey" % (site, neighbour), neighbour=neighbour, judged=True,
                days=" ".join(used), n_days=len(stats), src_coh=med["coh_ExEy"],
                resid_coh=med["coh_r"], resid_gain=med["gain"], mcoh_diff=med["mcoh_diff"],
                mcoh_sum=med["mcoh_sum"], model_holds=holds,
                reason="the site's own Ex against %s's Ey, both on the site's own H over the same days: "
                       "two electrodes that cannot share a centre" % neighbour)


def control_site(sv, site, members=None) -> dict:
    """The member of the site's group with the lowest Ex-Ey coherence, and its numbers.

    A READING beside built_control, not the criterion. The control is chosen on the MAXIMUM of its daily
    Ex-Ey coherence with the median as the tie-break, and `qualifies` says whether that maximum clears the
    0.35 ceiling; the ceiling is not raised to find one (vic_centre_test :171-175). Where no site qualifies
    the reading says so and the section is judged on the built control instead.
    """
    pool = [s for s in (members if members is not None else list(sv.sites.site)) if s != site]
    rows = []
    for s in pool:
        el = read_elines(sv, s)
        if not len(el) or "coh_Ex_Ey" not in el.columns:
            continue
        v = pd.to_numeric(el.coh_Ex_Ey, errors="coerce")
        if not np.isfinite(v).any():
            continue
        rows.append(dict(site=s, max_coh_Ex_Ey=float(np.nanmax(v)),
                         median_coh_Ex_Ey=float(np.nanmedian(v)), days=int(np.isfinite(v).sum())))
    if not rows:
        return dict(site=None, qualifies=False,
                    reason="no other site in the group carries an elines table")
    t = pd.DataFrame(rows).sort_values(["max_coh_Ex_Ey", "median_coh_Ex_Ey"]).reset_index(drop=True)
    best = t.iloc[0]
    return dict(site=str(best.site), median_coh_Ex_Ey=float(best.median_coh_Ex_Ey),
                max_coh_Ex_Ey=float(best.max_coh_Ex_Ey), table=t,
                qualifies=bool(best.max_coh_Ex_Ey < CONTROL_EX_EY_MAX),
                ceiling=CONTROL_EX_EY_MAX,
                reason="the lowest daily maximum Ex-Ey coherence of the group (%.3f over %d scored days)"
                       % (float(best.max_coh_Ex_Ey), int(best.days)))


# ---------------------------------------------------------------- the variant cache

def ne_variant(sv, site, rate=1, force=False) -> dict:
    """Write cache_<rate>hz_ne/<site>.npz: the same H and t0, Ex and Ey replaced by the two diagonals.

    Ex' = (Ex - Ey)/sqrt 2 is the north-west diagonal, free of the shared centre; Ey' = (Ex + Ey)/sqrt 2 is
    the north-east one, carrying it doubled. The difference is taken on the PHYSICALLY SIGNED lines -- the
    decisions.csv sign of each line is applied first -- because the model Ex = Ex_true + c, Ey = Ey_true + s c
    is stated for signed lines; the sidecar records that, and a pass on this cache does not sign E again.
    The check is on the algebra alone: the written Ex' equals (Ex - Ey)/sqrt 2 of the signed source at every
    finite sample. The coherence part of the frozen check is superseded -- the coherence between the two
    diagonals does not say which one is clean.
    """
    work = Path(sv.cfg["work_root"])
    src = work / ("cache_%dhz" % int(rate)) / ("%s.npz" % site)
    out_dir = work / ("cache_%dhz_ne" % int(rate))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / ("%s.npz" % site)
    try:
        drow = sv.decision(site)
    except KeyError:
        drow = None
    s_ex, _d1 = FR.read_sign(None if drow is None else drow.get("sign_ex"))
    s_ey, _d2 = FR.read_sign(None if drow is None else drow.get("sign_ey"))
    z = np.load(src, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    z.close()
    ex, ey = np.asarray(d["Ex"], float) * s_ex, np.asarray(d["Ey"], float) * s_ey
    exp, eyp = (ex - ey) / SQ2, (ex + ey) / SQ2
    d["Ex"] = exp.astype(np.float32)
    d["Ey"] = eyp.astype(np.float32)
    d["signs_applied"] = np.array(["Ex %+d, Ey %+d applied before the difference" % (s_ex, s_ey)])
    d["reason_x"] = np.array(["Ex' = (Ex - Ey)/sqrt(2): the north-minus-east voltage over the diagonal of "
                              "length L sqrt 2, free of the shared centre electrode"])
    d["reason_y"] = np.array(["Ey' = (Ex + Ey)/sqrt(2): the north-plus-east sum along the north-east "
                              "diagonal, carrying the centre electrode's noise doubled"])
    d["ne_frame"] = np.array(["E in the frame turned -45 deg: x' north-west (the difference), y' north-east "
                              "(the sum); H as the source cache, turned after the pass"])
    # the variant is rewritten only where it is absent or `force` is set, but the algebra is checked against
    # the file on disk either way: a check skipped because the file was already there is not a check
    written = bool(force or not dst.exists())
    if written:
        np.savez(dst, **d)
    fin = np.isfinite(ex) & np.isfinite(ey)
    got = np.asarray(np.load(dst, allow_pickle=False)["Ex"], float)
    exact = bool(np.allclose(got[fin], ((ex[fin] - ey[fin]) / SQ2).astype(np.float32), rtol=0, atol=0))
    return dict(site=site, path=str(dst), written=written, exact=exact, n_finite=int(fin.sum()),
                sign_ex=float(s_ex), sign_ey=float(s_ey),
                sidecar=str(_write_ne_sidecar(dst, site, rate, int(fin.sum()), exact, s_ex, s_ey)))


def _write_ne_sidecar(dst: Path, site, rate, n_finite, exact, s_ex, s_ey) -> Path:
    p = dst.with_suffix(".json")
    p.write_text(json.dumps(dict(
        site=site, rate_hz=float(rate), variant="ne",
        transform="Ex' = (Ex - Ey)/sqrt2, Ey' = (Ex + Ey)/sqrt2; H, t0 and the layout keys unchanged",
        signs_applied=dict(Ex=float(s_ex), Ey=float(s_ey)),
        signs_note="the lines are signed in this cache; a pass on it does not sign E again",
        frame="E turned -45 deg; the tensor a pass on this cache produces is R Z and is turned back",
        n_finite_samples=int(n_finite), algebra_exact=bool(exact)), indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------- the turn-back

def turn_columns(z, t=None, angle_deg=THETA_NE):
    """(Z', T') with the H columns turned: Z' = Z R^T and T' = T R^T, R = [[cos, sin], [-sin, cos]].

    A pass on the NE cache already carries the E row turn, so its impedance is R Z; turning the columns as
    well completes Z' = (R Z) R^T (vic_ne_rotate :41, :52-58).
    """
    R = FR.rotation_matrix(angle_deg)
    zz = np.asarray(z)
    zt = np.einsum("nij,kj->nik", zz, R)
    if t is None:
        return zt, None
    tt = np.asarray(t)
    return zt, np.einsum("nij,kj->nik", tt, R)


def turn_errors(e, angle_deg=THETA_NE):
    """Errors combined in quadrature over the turned columns."""
    R = FR.rotation_matrix(angle_deg)
    return np.sqrt(np.einsum("nij,kj->nik", np.asarray(e, float) ** 2, R ** 2))


def turn_invariants(z_before, z_after) -> dict:
    """The three tolerances of a column-only turn, and the trace as the counter-example.

    The determinant is scaled by ||Z||^2: at a near-singular period the relative form is meaningless (a
    tensor whose |det| is 0.5 per cent of ||Z||^2 reads 1.1e-5 relative and 6e-8 scaled). The trace is not
    an invariant of Z -> Z R^T and its ratio is reported so that the wrong criterion stays visible.
    """
    a, b = np.asarray(z_before, complex), np.asarray(z_after, complex)
    fin = np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
    if not fin.any():
        return dict(n=0, max_element_rel=np.nan, max_det_scaled=np.nan, max_frobenius_rel=np.nan,
                    trace_ratio=np.nan, ok=False)
    A, B = a[fin], b[fin]
    R = FR.rotation_matrix(THETA_NE)
    want = np.einsum("nij,kj->nik", A, R)
    el = float(np.max(np.abs(B - want) / np.maximum(np.abs(want), 1e-30)))
    det = float(np.max(np.abs(np.linalg.det(B) - np.linalg.det(A))
                       / np.maximum(np.linalg.norm(A, axis=(1, 2)) ** 2, 1e-30)))
    fro_a = np.linalg.norm(A, axis=(1, 2))
    fro_b = np.linalg.norm(B, axis=(1, 2))
    fro = float(np.max(np.abs(fro_b - fro_a) / np.maximum(fro_a, 1e-30)))
    tr_a = np.abs(A[:, 0, 0] + A[:, 1, 1])
    tr_b = np.abs(B[:, 0, 0] + B[:, 1, 1])
    return dict(n=int(fin.sum()), max_element_rel=el, max_det_scaled=det, max_frobenius_rel=fro,
                trace_ratio=float(np.median(tr_b / np.maximum(tr_a, 1e-30))),
                ok=bool(el <= ELEMENT_RTOL and det <= DET_SCALED_TOL and fro <= FROBENIUS_RTOL))


def turn_edi(path, angle_deg=THETA_NE, note="") -> dict:
    """Turn a written EDI's H columns in place and read it back against the intended tensor.

    Returns the invariants of the turn plus the read-back element error, so the check is on the file the
    later workbooks read and not on an array in memory.
    """
    from mt_metadata.transfer_functions.core import TF
    path = Path(path)
    tf = TF(fn=str(path))
    tf.read()
    Z = np.asarray(tf.impedance.values, complex)
    Zr, _ = turn_columns(Z, None, angle_deg)
    try:
        Ze = np.asarray(tf.impedance_error.values, float)
        tf.impedance_error = turn_errors(Ze, angle_deg)
    except Exception:
        pass
    tf.impedance = Zr
    try:
        if tf.has_tipper() and tf.tipper is not None:
            T = np.asarray(tf.tipper.values, complex)
            _z, Tr = turn_columns(Z, T, angle_deg)
            tf.tipper = Tr
            try:
                Te = np.asarray(tf.tipper_error.values, float)
                tf.tipper_error = turn_errors(Te, angle_deg)
            except Exception:
                pass
    except Exception:
        pass
    st = tf.station_metadata
    try:
        pp = list(st.transfer_function.processing_parameters or [])
        pp.append("frame=%+.1f deg: x' north-west = (Ex - Ey)/sqrt2 free of the shared centre electrode, "
                  "y' north-east = (Ex + Ey)/sqrt2 carrying it; the H columns turned to match. %s"
                  % (angle_deg, note))
        st.transfer_function.processing_parameters = pp
    except Exception:
        pass
    try:
        tf.write(fn=str(path), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(path), file_type="edi")
    t2 = TF(fn=str(path))
    t2.read()
    Z2 = np.asarray(t2.impedance.values, complex)
    inv = turn_invariants(Z, Z2)
    return dict(path=str(path), **inv)
