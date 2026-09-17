"""The shared centre electrode, and the arm diagonal that cancels it at any pair of arm lengths.

The EDL L layout is three electrodes: a shared centre C, a north arm of length L_N and an east arm of length
L_E, so Ex = (V_N - V_C)/L_N and Ey = (V_E - V_C)/L_E. A noisy centre puts one voltage on both lines, and
because the two lines divide that voltage by different lengths it does not arrive as the same field on both.
With the lines physically signed and a centre voltage n,

    Ex = Ex_true + c,   Ey = Ey_true + s c (L_N / L_E),   c = -n / L_N,

with s = +1 for arms N+E or S+W and -1 where one arm is reversed. The E signs predict s: the convention
(Ex -1, Ey -1) is N+E, each departure to +1 reverses an arm, so s = +1 for an even number of departures and
-1 for an odd one. An undecided sign is never filled by convention: the site is UNJUDGED on the prediction.

The residual test, ported from vic_centre_test.day_stats :80-99 and criteria :6-24 with the arm lengths
carried through the gain. On the days of highest Ex-Ey coherence, at 20-200 s, the part of each line
explained by (Hx, Hy) is removed per frequency bin from the cross-spectra and the residuals are read. The
complex gain g = S_ry,rx / S_rx,rx is then s L_N / L_E, so the model holds where the residual coherence is at
least 0.9 and |g| divided by the expected L_N / L_E lies in 0.85-1.18; sign(Re g) is still the observed s.
The lengths come from sites.csv dipole_n_m and dipole_e_m, and an assume: cell is used and named. The clean
diagonal is the one whose multiple coherence with (Hx, Hy) is the higher. The three criteria:

    A  where the model holds and both E signs are decided, the prediction must agree with the observed sign
    B  a control that cannot share a centre must not hold the model
    C  the remedy applies only where the model holds and the clean diagonal by H is the observed one

The remedy (ported from vic_ne_cache :6-13 and vic_ne_rotate :6-21, generalised to unequal arms). The
voltage between the two arm electrodes carries no centre at any lengths:

    V_N - V_E = L_N Ex - L_E Ey,   d = sqrt(L_N^2 + L_E^2),   E_d = (L_N Ex - L_E Ey) / d,

which is the field along the unit vector (L_N, -L_E)/d in (north, east), that is at
theta = atan2(-L_E, L_N) from north. The orthogonal row E_v = (L_E Ex + L_N Ey) / d carries the centre and is
kept for the record. Equal arms give theta = -45 deg and the pair reduces to (Ex - Ey)/sqrt 2 and
(Ex + Ey)/sqrt 2 exactly. The pair is E in the frame turned by theta, so a pass on the variant cache gives
R(theta) Z; turning the H columns as well gives Z' = (R Z) R^T and T' = T R^T at that same theta. The x' row
is the clean one and the y' row is kept for the record.

The turn-back is checked on three invariants of a column-only turn: the elements to 1e-6 relative (an EDI
carries seven significant digits), |det Z' - det Z| <= 1e-5 ||Z||^2 scaled by the norm because the relative
form is meaningless at a near-singular period, and the Frobenius norm to 1e-6. The trace is not an invariant
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
from .masks import load_raw, prepare

NPERSEG = 4096
BAND_S = (20.0, 200.0)
CENTRE_DAYS = 3
RESID_COH_MIN = 0.9
GAIN_LO, GAIN_HI = 0.85, 1.18   # the bounds on |g| divided by the expected L_N / L_E
CONTROL_EX_EY_MAX = 0.35        # a control site's source Ex-Ey coherence on every scored day
THETA_NE = -45.0                # the equal-arm diagonal; diagonal_angle gives the site's own

# the tolerances of the turn-back, vic_ne_rotate.py:12-21
ELEMENT_RTOL = 1e-6
DET_SCALED_TOL = 1e-5
FROBENIUS_RTOL = 1e-6


def arm_lengths(sv, site) -> dict:
    """(L_N, L_E) from sites.csv dipole_n_m and dipole_e_m, with the source of each named.

    An `assume:<value>` cell is used and flagged, as everywhere else in the package: the expected gain and
    the diagonal's direction both scale with these numbers, so a transfer function built on an assumed arm
    carries the assumption into its provenance.
    """
    from ..raw.cache import dipole_value
    row = sv.site(site)
    ln, ln_kind = dipole_value(row.get("dipole_n_m", ""))
    le, le_kind = dipole_value(row.get("dipole_e_m", ""))
    ok = bool(np.isfinite(ln) and np.isfinite(le) and ln > 0 and le > 0)
    return dict(L_N=float(ln), L_E=float(le), known=ok,
                assumed=sorted(c for c, k in (("Ex", ln_kind), ("Ey", le_kind)) if k == "assume"),
                source=str(row.get("dipole_source", "")),
                note=("L_N %.3g m, L_E %.3g m (%s)" % (ln, le, row.get("dipole_source", "no source"))
                      if ok else "the arm lengths are not in sites.csv"))


def expected_gain(L_N, L_E) -> float:
    """|g| the shared-centre model predicts for the residual test: L_N / L_E.

    One centre voltage divided by two different arm lengths is two different fields, so the residuals of a
    real shared centre come out in the ratio of the lengths and not at one.
    """
    if not (np.isfinite(L_N) and np.isfinite(L_E)) or L_E == 0:
        return float("nan")
    return float(L_N) / float(L_E)


def diagonal_angle(L_N, L_E) -> float:
    """theta in degrees of the arm diagonal from north: atan2(-L_E, L_N). Equal arms give -45 deg."""
    if not (np.isfinite(L_N) and np.isfinite(L_E)):
        return float("nan")
    return float(np.degrees(np.arctan2(-float(L_E), float(L_N))))


def diagonal_length(L_N, L_E) -> float:
    """d = sqrt(L_N^2 + L_E^2), the separation of the two arm electrodes."""
    return float(np.hypot(float(L_N), float(L_E)))


def _arm_fields(arms: dict, g_expected: float) -> dict:
    """The arm columns every centre result carries, so a reader can see what the gain was judged against."""
    return dict(L_N=arms["L_N"], L_E=arms["L_E"], arms_known=arms["known"],
                arms_assumed=" ".join(arms["assumed"]), arm_source=arms["source"][:120],
                gain_expected=g_expected,
                theta_deg=diagonal_angle(arms["L_N"], arms["L_E"]),
                diagonal_m=diagonal_length(arms["L_N"], arms["L_E"]))


def diagonals(ex, ey, L_N, L_E):
    """(E_d, E_v): the field along the arm diagonal and along its orthogonal, at any pair of lengths.

    E_d = (L_N Ex - L_E Ey) / d is the voltage between the two arm electrodes over their separation and
    carries no centre; E_v = (L_E Ex + L_N Ey) / d carries it. The pair is R(theta) applied to (Ex, Ey).
    """
    d = diagonal_length(L_N, L_E)
    ex = np.asarray(ex, float)
    ey = np.asarray(ey, float)
    return (float(L_N) * ex - float(L_E) * ey) / d, (float(L_E) * ex + float(L_N) * ey) / d


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


def day_stats(ex, ey, hx, hy, band_s=BAND_S, nperseg=NPERSEG, fs=1.0, L_N=1.0, L_E=1.0) -> dict:
    """The residual coherence, the complex gain and the two diagonals' H coherence over one piece.

    S_r1,r2 = S_12 - S_1h Shh^-1 S_h2 is the cross-spectrum of the residuals after the H-explained part is
    removed; the model's observable is their coherence and the gain g = S_ry,rx / S_rx,rx, which a real
    shared centre puts at s L_N / L_E. Ported from vic_centre_test.day_stats (:80-99), with the arm lengths
    carried into the two diagonals: the clean one is ex - (L_E/L_N) ey and the one that keeps the centre is
    ex + (L_N/L_E) ey, which at equal arms are ex - ey and ex + ey. A multiple coherence does not change
    when its combination is scaled, so only the ratio of the lengths enters.
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

    ln, le = float(L_N), float(L_E)
    w_diff = -le / ln if ln else -1.0
    w_sum = ln / le if le else 1.0
    return dict(coh_r=coh_r, gain=gain, s_obs=s_obs, mcoh_diff=mcoh(w_diff), mcoh_sum=mcoh(w_sum),
                w_diff=float(w_diff), w_sum=float(w_sum),
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

    Returns the medians over those days, the arm lengths and the gain they predict, whether the model holds,
    the predicted and observed signs, the two diagonals' H coherence, whether the site qualifies as a control
    (source Ex-Ey coherence under 0.35 on every scored day) and whether the remedy applies.
    """
    if elines is None:
        elines = read_elines(sv, site)
    arms = arm_lengths(sv, site)
    ln, le = (arms["L_N"], arms["L_E"]) if arms["known"] else (1.0, 1.0)
    g_expected = expected_gain(ln, le)
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
        st = day_stats(p["Ex"], p["Ey"], p["Hx"], p["Hy"], band_s, nperseg, float(rate), ln, le)
        if not st:
            continue
        st["day"], st["src"] = r.day, float(r.coh_Ex_Ey)
        stats.append(st)
        used.append(r.day)
    if not stats:
        return dict(site=site, judged=False, days="", reason="no usable day",
                    s_pred=s_pred, sign_undecided=" ".join(undecided), **_arm_fields(arms, g_expected))
    med = {k: float(np.median([d[k] for d in stats]))
           for k in ("coh_r", "gain", "mcoh_diff", "mcoh_sum", "coh_ExEy", "src")}
    s_obs = float(np.sign(np.sum([d["s_obs"] for d in stats]))) or 1.0
    ratio = med["gain"] / g_expected if np.isfinite(g_expected) and g_expected else np.nan
    model = bool(med["coh_r"] >= RESID_COH_MIN and np.isfinite(ratio)
                 and GAIN_LO <= ratio <= GAIN_HI)
    control = bool(max(d["src"] for d in stats) < CONTROL_EX_EY_MAX)
    clean_obs = "diff" if s_obs > 0 else "sum"
    clean_pred = "" if undecided else ("diff" if s_pred > 0 else "sum")
    clean_H = "diff" if med["mcoh_diff"] > med["mcoh_sum"] else "sum"
    return dict(site=site, judged=True, days=" ".join(used), n_days=len(stats),
                src_coh=med["src"], resid_coh=med["coh_r"], resid_gain=med["gain"],
                gain_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
                s_pred=s_pred, s_obs=s_obs, sign_undecided=" ".join(undecided),
                a_judged=bool(not undecided and model),
                mcoh_diff=med["mcoh_diff"], mcoh_sum=med["mcoh_sum"],
                model_holds=model, control=control, **_arm_fields(arms, g_expected),
                clean_pred=clean_pred, clean_obs=clean_obs, clean_by_H=clean_H,
                sign_agrees=(None if undecided or not model else bool(s_obs == s_pred)),
                remedy_applicable=bool(model and clean_H == clean_obs))


def read_elines(sv, site) -> pd.DataFrame:
    """<work_root>/<site>/elines.csv, the per-day electric-line table workbook 02 wrote."""
    p = Path(sv.cfg["work_root"]) / site / "elines.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def built_control(sv, site, neighbour=None, days=CENTRE_DAYS, elines=None, band_s=BAND_S,
                  nperseg=NPERSEG, rate=1, members=None) -> dict:
    """The residual test on a pair of lines that cannot share a centre electrode, as the control.

    The site's own Ex is paired with the nearest sound site's Ey, both read against the site's own (Hx, Hy)
    over the same days, and the residual test is run on that pair. Two electrodes tens of kilometres apart
    have no common voltage, so the model must not hold: a residual coherence at or above 0.9 with a gain
    within 0.85-1.18 of the site's own expected L_N / L_E there would mean the test finds a shared centre
    wherever it looks. The control is judged against the site's own expected gain, because the question it
    answers is whether this site's test can fire on a pair that cannot have a shared centre.

    This is the control the section is judged on. A site whose own Ex-Ey coherence stays under
    CONTROL_EX_EY_MAX = 0.35 on every day need not exist in a survey -- a one-dimensional earth correlates
    the two lines through the source field alone, and no AusLAMP Queensland Phase 1 site clears that ceiling,
    the lowest daily maximum being 0.354 (Ben, 2026-09-17) -- so a criterion written on finding one is
    UNJUDGED wherever the survey is layered. The built pair exists at every site with a neighbour.
    """
    if elines is None:
        elines = read_elines(sv, site)
    arms = arm_lengths(sv, site)
    ln, le = (arms["L_N"], arms["L_E"]) if arms["known"] else (1.0, 1.0)
    g_expected = expected_gain(ln, le)
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
        st = day_stats(p["Ex"], p["Ey"], p["Hx"], p["Hy"], band_s, nperseg, float(rate), ln, le)
        if not st:
            continue
        st["day"] = r.day
        stats.append(st)
        used.append(r.day)
    if not stats:
        return dict(site=site, judged=False, model_holds=None, neighbour=neighbour,
                    reason="no day carries both the site's Ex and %s's Ey" % neighbour,
                    **_arm_fields(arms, g_expected))
    med = {k: float(np.median([d[k] for d in stats]))
           for k in ("coh_r", "gain", "mcoh_diff", "mcoh_sum", "coh_ExEy")}
    # the control is judged by the site's own expected gain, because the question it answers is whether this
    # site's test can fire on a pair that cannot have a shared centre
    ratio = med["gain"] / g_expected if np.isfinite(g_expected) and g_expected else np.nan
    holds = bool(med["coh_r"] >= RESID_COH_MIN and np.isfinite(ratio) and GAIN_LO <= ratio <= GAIN_HI)
    return dict(site="%s Ex + %s Ey" % (site, neighbour), neighbour=neighbour, judged=True,
                days=" ".join(used), n_days=len(stats), src_coh=med["coh_ExEy"],
                resid_coh=med["coh_r"], resid_gain=med["gain"],
                gain_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
                mcoh_diff=med["mcoh_diff"], mcoh_sum=med["mcoh_sum"], model_holds=holds,
                **_arm_fields(arms, g_expected),
                reason="the site's own Ex against %s's Ey, both on the site's own H over the same days: "
                       "two electrodes that cannot share a centre" % neighbour)


def control_site(sv, site, members=None) -> dict:
    """The member of the site's group with the lowest Ex-Ey coherence, and its numbers.

    A reading beside built_control, not the criterion. The control is chosen on the maximum of its daily
    Ex-Ey coherence with the median as the second key, and `qualifies` says whether that maximum clears the
    CONTROL_EX_EY_MAX = 0.35 ceiling; the ceiling is not raised to find one (vic_centre_test :171-175). Where
    no site qualifies the reading says so and the section is judged on the built control instead.
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
    """Write cache_<rate>hz_ne/<site>.npz: the same H and t0, Ex and Ey replaced by the two arm diagonals.

    Ex' = (L_N Ex - L_E Ey) / d is the voltage between the two arm electrodes over their separation
    d = sqrt(L_N^2 + L_E^2), free of the shared centre at any pair of lengths; Ey' = (L_E Ex + L_N Ey) / d is
    the orthogonal row and carries the centre. The pair is E turned by theta = atan2(-L_E, L_N), which at
    equal arms is -45 deg and reduces to (Ex - Ey)/sqrt 2 and (Ex + Ey)/sqrt 2 exactly.

    The combination is taken on the physically signed lines -- the decisions.csv sign of each is applied
    first -- because the model Ex = Ex_true + c, Ey = Ey_true + s c L_N/L_E is stated for signed lines; the
    sidecar records that, and a pass on this cache does not sign E again. The check is on the algebra alone:
    the written Ex' equals (L_N Ex - L_E Ey)/d of the signed source at every finite sample. The coherence
    part of the frozen check is superseded -- the coherence between the two diagonals does not say which one
    is clean.
    """
    work = Path(sv.cfg["work_root"])
    src = work / ("cache_%dhz" % int(rate)) / ("%s.npz" % site)
    out_dir = work / ("cache_%dhz_ne" % int(rate))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / ("%s.npz" % site)
    arms = arm_lengths(sv, site)
    if not arms["known"]:
        return dict(site=site, path=str(dst), written=False, exact=None, **_arm_fields(arms, np.nan),
                    reason="the arm lengths are not in sites.csv, so the diagonal has no direction")
    ln, le = arms["L_N"], arms["L_E"]
    theta, dsep = diagonal_angle(ln, le), diagonal_length(ln, le)
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
    e_d, e_v = diagonals(ex, ey, ln, le)
    d["Ex"] = e_d.astype(np.float32)
    d["Ey"] = e_v.astype(np.float32)
    d["signs_applied"] = np.array(["Ex %+d, Ey %+d applied before the combination" % (s_ex, s_ey)])
    d["reason_x"] = np.array(["Ex' = (L_N Ex - L_E Ey)/d with L_N %.4g m, L_E %.4g m, d %.4g m: the voltage "
                              "between the two arm electrodes over their separation, free of the shared "
                              "centre electrode at any pair of lengths" % (ln, le, dsep)])
    d["reason_y"] = np.array(["Ey' = (L_E Ex + L_N Ey)/d: the row orthogonal to the arm diagonal, which "
                              "carries the centre electrode's voltage"])
    d["ne_frame"] = np.array(["E in the frame turned %+.4f deg = atan2(-L_E, L_N): x' along the arm "
                              "diagonal, y' orthogonal to it; H as the source cache, turned after the pass"
                              % theta])
    # a cache built at another frame is not this frame's cache: where the sidecar is missing, carries no arm
    # block (the equal-arm code wrote none) or names other lengths or another theta, the variant is rewritten
    # whether or not `force` is set, and `refreshed` tells the caller to remake the pass that reads it
    refreshed = bool(dst.exists() and _frame_changed(dst.with_suffix(".json"), ln, le, theta))
    # otherwise it is rewritten only where it is absent or `force` is set, and the algebra is checked against
    # the file on disk either way: a check skipped because the file was already there is not a check
    written = bool(force or refreshed or not dst.exists())
    if written:
        np.savez(dst, **d)
    fin = np.isfinite(ex) & np.isfinite(ey)
    got = np.asarray(np.load(dst, allow_pickle=False)["Ex"], float)
    want = diagonals(ex[fin], ey[fin], ln, le)[0].astype(np.float32)
    exact = bool(np.allclose(got[fin], want, rtol=0, atol=0))
    return dict(site=site, path=str(dst), written=written, refreshed=refreshed, exact=exact,
                n_finite=int(fin.sum()),
                sign_ex=float(s_ex), sign_ey=float(s_ey), **_arm_fields(arms, expected_gain(ln, le)),
                sidecar=str(_write_ne_sidecar(dst, site, rate, int(fin.sum()), exact, s_ex, s_ey,
                                              ln, le, theta, dsep)))


def _frame_changed(sidecar: Path, L_N, L_E, theta) -> bool:
    """True where the variant on disk was built at other arm lengths or another frame angle.

    A sidecar written before the lengths were carried has no `arms` block at all, so it reads as changed and
    its cache is rebuilt: the equal-arm combination it holds is this site's only where the arms are equal.
    """
    try:
        a = (json.loads(sidecar.read_text(encoding="utf-8")) or {}).get("arms")
    except Exception:
        return True
    if not isinstance(a, dict):
        return True
    return not (abs(float(a.get("L_N", np.nan)) - float(L_N)) <= 1e-9
                and abs(float(a.get("L_E", np.nan)) - float(L_E)) <= 1e-9
                and abs(float(a.get("theta_deg", np.nan)) - float(theta)) <= 1e-9)


def _write_ne_sidecar(dst: Path, site, rate, n_finite, exact, s_ex, s_ey, L_N, L_E, theta, dsep) -> Path:
    p = dst.with_suffix(".json")
    p.write_text(json.dumps(dict(
        site=site, rate_hz=float(rate), variant="ne",
        arms=dict(L_N=float(L_N), L_E=float(L_E), separation_m=float(dsep), theta_deg=float(theta)),
        transform="Ex' = (L_N Ex - L_E Ey)/d, Ey' = (L_E Ex + L_N Ey)/d with d = sqrt(L_N^2 + L_E^2); "
                  "H, t0 and the layout keys unchanged",
        signs_applied=dict(Ex=float(s_ex), Ey=float(s_ey)),
        signs_note="the lines are signed in this cache; a pass on it does not sign E again",
        frame="E turned %+.4f deg = atan2(-L_E, L_N); the tensor a pass on this cache produces is R(theta) Z "
              "and is turned back by Z' = (R Z) R^T at that theta" % theta,
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


def turn_frame(z, t=None, angle_deg=THETA_NE):
    """(Z', T') of a tensor estimated in the site's own frame, written in the frame at angle_deg.

    turn_columns turns the H columns alone, which is all a pass on the NE cache needs: that cache already
    carries the E row turn, so its estimate is R Z. A tensor estimated in the site's own frame carries
    neither turn, so both are applied here -- the rows put E into the turned frame and the columns put H
    into it -- and Z' = R Z R^T. The tipper's one row is Hz, which does not turn, so T' = T R^T.
    """
    R = FR.rotation_matrix(angle_deg)
    return turn_columns(np.einsum("ik,nkj->nij", R, np.asarray(z)), t, angle_deg)


def turn_frame_errors(e, angle_deg=THETA_NE):
    """Errors combined in quadrature over both turns of turn_frame, as turn_errors does over the columns."""
    R = FR.rotation_matrix(angle_deg)
    rows = np.sqrt(np.einsum("ik,nkj->nij", R ** 2, np.asarray(e, float) ** 2))
    return turn_errors(rows, angle_deg)


def turn_tf(tf, angle_deg=THETA_NE):
    """One read transfer function (transfer_functions.TFData) written in the frame at angle_deg.

    The tensor and the tipper go through turn_frame and their error bars through turn_frame_errors, so a
    transfer function estimated in the site's own frame is drawn and read against a pass whose cache already
    carries the turn, on the same axes and in the same units.
    """
    z, t = turn_frame(tf.z, tf.t, angle_deg)
    te = None if tf.t_err is None else turn_errors(np.asarray(tf.t_err, float), angle_deg)
    return tf._replace(z=z, z_err=turn_frame_errors(tf.z_err, angle_deg), t=t, t_err=te)


def turn_invariants(z_before, z_after, angle_deg=THETA_NE) -> dict:
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
    R = FR.rotation_matrix(float(angle_deg))
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
        pp.append("frame=%+.4f deg = atan2(-L_E, L_N): x' along the arm diagonal, "
                  "(L_N Ex - L_E Ey)/d free of the shared centre electrode, y' orthogonal to it, "
                  "(L_E Ex + L_N Ey)/d carrying it; the H columns turned to match. %s"
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
    inv = turn_invariants(Z, Z2, angle_deg)
    return dict(path=str(path), **inv)
