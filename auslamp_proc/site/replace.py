"""A magnetic channel borrowed from a neighbour, one channel at a time, and the frames that go with it.

The physical claim (wamt_substitute.py:1-14): at long period the horizontal magnetic field is homogeneous
over the site spacing, so a neighbour's H measures the same field; at short period it is not. A borrowed long
end therefore goes UNDER the site's own short periods and the two are spliced where borrowing stops costing.

WHICH CHANNEL, AND FROM WHOM (qld_campaign.py:122-139, :696-777). The WORST channel only, and only where it
is clearly worse than the other: a site decorrelated from its neighbours for any reason -- distance, a quiet
spell, its own noise -- has both channels below any threshold, so "any channel below a threshold" replaces
both and wrecks the site. The own channel's coherence with the candidate is the WHOLE-RECORD mean over
100-1000 s, deliberately not the chunk median used elsewhere: the chunk median is more forgiving and misses
the two sites whose own channels really are bad. The candidate must itself be sound, judged against a THIRD
site at 0.80 -- a donor vouched for by the site it stands in for has been vouched for by nobody
(qld_v2_pairs.py:194-222) -- and a donor cannot also be a member of the stack the pass uses as its reference.
A pair's score is the MIN of its two channels; coverage is reported and not gated.

THE FRAMES (vic_w3_vic040.py:335-365, :427-431). The lender's pair is served in its own mean-horizontal-field
frame, so it is turned by minus the site's angle into the site's SENSOR frame before it stands in for a
sensor-frame channel. The tensor the pass then produces has its H basis in whatever pair was handed in, and
is turned back on the right. With H_used = M H_sensor and H_geo = R(t) H_sensor,

    M's row i = the identity row where channel i is borrowed, row i of R(t) where it is the site's own,
    Z_geo = Z_used M R(-t),   T_geo = T_used M R(-t).

With both channels borrowed M is the identity and the correction is the rotation R(-t); with neither, M is
R(t) and the correction is the identity; with one of each M is [[1, 0], [-sin t, cos t]] (or its transpose
case) and the correction is not a rotation.

WHAT IS NEVER DELIVERED. A form that borrows BOTH horizontal channels is an inter-site impedance -- the
tensor of the site's E on the regional field the neighbour measured -- and is labelled as one, shown and not
delivered (the VIC040 result: an Hx-only replacement sits within 4 per cent and 0.6 deg of the own-H product
where every whole-pair form is 1.6x off). Electrics are never borrowed. A lender leaves every reference the
pass reads, because a reference sharing a channel with the local H compares a channel with itself
(vic_w3_lender.py:1-22).

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..process import frame as FR
from ..process import references as REF
from .masks import H, distance_km, span

# qld_campaign.py:122-125 and wamt_substitute.py:107-117
SUB_BAND_S = (100.0, 1000.0)
SUB_NPERSEG = 16384
SUB_THRESHOLD = 0.5             # the own channel's coherence below which a replacement is considered
SUB_CLEARLY = 0.7               # ... and it must be this much below the other channel's
DONOR_GATE = 0.80               # the donor's coherence against a THIRD site
COVER_GATE = 0.95               # coverage is reported, not gated; this is the level reported against
MIN_OVERLAP_DAYS = 20.0

# wamt_substitute.py:1113-1152
SPLICE_RHO = 0.05
SPLICE_PHASE_DEG = 5.0
SPLICE_SIGMA = 2.0
SPLICE_TEST_HI_S = 3000.0
SPLICE_TEST_LO_S = 10.0


def sub_coh(x, y, fs=1.0, band_s=SUB_BAND_S, nperseg=SUB_NPERSEG, despike_k=30.0) -> float:
    """The WHOLE-RECORD mean coherence over a band. Ported from qld_campaign.sub_coh (:678-694).

    The chunk median is the better statistic for a single pair over a long record and is the wrong one for
    this decision: it is more forgiving and it misses the sites whose own channels really are bad. The chunk
    median stays where it belongs, in the donor weighting.

    Both series are despiked first (look.despike, vic_windows.py:95). A whole-record mean has no median to
    hide behind, so one logger spike sets the whole number: measured on AusLAMP Queensland Phase 1 on
    2026-09-17, Q53N against Q52N reads 0.08 on Hx raw and 0.76 despiked, and 0.06 against 0.90 on Hy, while
    Q53N against Q63 moves 0.957 to 0.968. Without the despike the rule would call every pair uncoherent.
    """
    from scipy.signal import coherence
    from ..look import despike
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if despike_k:
        x, _a = despike(x, despike_k)
        y, _b = despike(y, despike_k)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 * 256:
        return np.nan
    x, y = x[ok], y[ok]
    n = int(min(nperseg, max(256, len(x) // 8)))
    f, c = coherence(x - np.mean(x), y - np.mean(y), fs=fs, nperseg=n)
    m = (f >= 1.0 / band_s[1]) & (f <= 1.0 / band_s[0])
    return float(np.mean(c[m])) if m.any() else np.nan


def rotated_pair(sv, site, rate=1):
    """(t0, {'Hx','Hy'} in the site's own mean-field frame, the angle, n).

    The same object the reference store stacks, so a channel borrowed here and the same site's channel used
    as a reference elsewhere are one record (process.references.Store.rotated :152-179).
    """
    st = REF.Store(sv, [site], rate=int(rate))
    return st.rotated(site)


def on_grid(h, t_src, t0, n):
    """One pair placed on the target's grid, NaN outside the overlap."""
    out = {}
    off = int(t_src - t0)
    for c in H:
        x = np.full(int(n), np.nan)
        a0, a1 = max(0, off), min(int(n), off + len(h[c]))
        if a1 > a0:
            x[a0:a1] = np.asarray(h[c], float)[a0 - off:a1 - off]
        out[c] = x
    return out


def coverage(pair) -> float:
    """The fraction of the target's grid the borrowed pair covers on both channels."""
    ok = np.isfinite(pair["Hx"]) & np.isfinite(pair["Hy"])
    return float(ok.mean()) if len(ok) else 0.0


def candidates(sv, site, chan=None, donors=None, rate=1, band_s=SUB_BAND_S, exclude=(),
               max_donors=6) -> pd.DataFrame:
    """One row per candidate lender: the own channel coherences, the third-site check and the coverage.

    `chan` restricts the table to one channel; with None both are scored and the WORST is the one the rule
    would replace. `exclude` names sites a donor may not be -- the members of the stack the pass uses, and
    the remote site -- because a reference sharing a channel with the local H compares a channel with itself.
    """
    t0, n = span(sv, site, rate)
    own_t0, own, _ang, _n = rotated_pair(sv, site, rate)
    pool = [s for s in (donors if donors is not None else list(sv.sites.site))
            if s != site and s not in set(exclude)]
    pool = sorted(pool, key=lambda s: distance_km(sv, site, s))
    got = []
    for s in pool:
        try:
            ts, h, _a, _nn = rotated_pair(sv, s, rate)
        except Exception:
            continue
        g = on_grid(h, ts, own_t0, n)
        if coverage(g) * n / 86400.0 / float(rate) < MIN_OVERLAP_DAYS:
            continue
        got.append((s, g))
        if len(got) >= int(max_donors):
            break
    rows = []
    for i, (s, g) in enumerate(got):
        # the donor is vouched for by the BEST third site among the other candidates, never by the site it
        # stands in for and never by whichever neighbour happens to come next in the list: a third site that
        # is itself broken refuses every donor, which would make the gate unusable rather than strict
        third, third_coh = "", {}
        best = -1.0
        for j, (s2, g2) in enumerate(got):
            if j == i:
                continue
            cand_coh = {c: sub_coh(g[c], g2[c], float(rate), band_s) for c in H}
            score = float(np.nanmin([cand_coh[c] for c in H]))
            if np.isfinite(score) and score > best:
                best, third, third_coh = score, s2, cand_coh
        row = dict(site=site, candidate=s, km=distance_km(sv, site, s),
                   coverage=round(coverage(g), 4), third_site=third)
        for c in H:
            row["own_%s_coh" % c] = sub_coh(own[c][:n], g[c], float(rate), band_s)
            row["donor_%s_coh" % c] = third_coh.get(c, np.nan)
        cx, cy = row["own_Hx_coh"], row["own_Hy_coh"]
        worst = "Hx" if (np.isfinite(cx) and np.isfinite(cy) and cx < cy) else "Hy"
        row["worst_channel"] = worst
        other = "Hy" if worst == "Hx" else "Hx"
        wv, ov = row["own_%s_coh" % worst], row["own_%s_coh" % other]
        row["pair_score"] = float(np.nanmin([cx, cy]))
        row["donor_sound"] = bool(np.isfinite(row["donor_%s_coh" % worst])
                                  and row["donor_%s_coh" % worst] >= DONOR_GATE)
        row["rule_fires"] = bool(np.isfinite(wv) and np.isfinite(ov) and wv < SUB_THRESHOLD
                                 and wv < SUB_CLEARLY * ov and row["donor_sound"])
        row["covers"] = bool(row["coverage"] >= COVER_GATE)
        row["reason"] = ("own %s %.2f over %g-%g s, %s %.2f (%.2f of the other channel's %.2f); donor "
                         "against its best third site %s %.2f, %s %.2f"
                         % (worst, wv if np.isfinite(wv) else np.nan, band_s[0], band_s[1],
                            "below" if np.isfinite(wv) and wv < SUB_THRESHOLD else "not below",
                            SUB_THRESHOLD, (wv / ov) if np.isfinite(wv) and ov else np.nan,
                            ov if np.isfinite(ov) else np.nan, third or "no third site",
                            row["donor_%s_coh" % worst] if np.isfinite(row["donor_%s_coh" % worst])
                            else np.nan,
                            "at or above" if row["donor_sound"] else "below", DONOR_GATE))
        rows.append(row)
    out = pd.DataFrame(rows)
    if chan and len(out):
        out = out.assign(channel=chan)
    return out


# ---------------------------------------------------------------- the frames

def mixed_basis(theta_deg, borrowed) -> np.ndarray:
    """M with H_used = M H_sensor: the identity row where a channel is borrowed, R(t)'s row where it is own.

    A borrowed channel is handed in already turned into the site's SENSOR frame, so its row of M is the
    identity's; the site's own channel is handed in in its mean-field frame, so its row is R(t)'s.
    """
    R = FR.rotation_matrix(float(theta_deg))
    I = np.eye(2)
    return np.array([I[i] if bool(borrowed[i]) else R[i] for i in (0, 1)], float)


def basis_correction(theta_deg, borrowed) -> np.ndarray:
    """M R(-t): the matrix a tensor's H basis is right-multiplied by to reach the mean-field frame.

    Both channels borrowed gives the rotation R(-t); neither gives the identity; one of each gives
    [[1, 0], [-sin t, cos t]] R(-t), which is not a rotation.
    """
    return mixed_basis(theta_deg, borrowed) @ FR.rotation_matrix(-float(theta_deg))


def to_sensor_frame(pair, theta_deg):
    """A mean-field-frame pair turned by minus the site's angle into the site's sensor frame.

    H_sensor = R(-t) H_geo with R(a) = [[cos, sin], [-sin, cos]] (vic_w3_vic040.py:337-348).
    """
    a = np.radians(float(theta_deg))
    c, s = np.cos(a), np.sin(a)
    hx = c * np.asarray(pair["Hx"], float) - s * np.asarray(pair["Hy"], float)
    hy = s * np.asarray(pair["Hx"], float) + c * np.asarray(pair["Hy"], float)
    return {"Hx": hx, "Hy": hy}


def replace_channel(local, lender_pair, chan, theta_deg, whole_pair=False) -> dict:
    """The local five channels with one (or both) horizontal magnetic channel(s) borrowed.

    `local` is the site's own signed, rotated record; `lender_pair` is the lender's mean-field-frame pair
    already on the site's grid. The borrowed channel is turned into the site's sensor frame; the site's own
    channel is left in its mean-field frame; E and Hz are always the site's own. Returns the arrays, the
    (borrowed_x, borrowed_y) flags, the correction matrix and a note.
    """
    sensor = to_sensor_frame(lender_pair, theta_deg)
    out = {k: np.asarray(v, float) for k, v in local.items()}
    if whole_pair:
        borrowed = (True, True)
        out["Hx"], out["Hy"] = sensor["Hx"], sensor["Hy"]
        note = ("both horizontal channels borrowed and turned into the site's sensor frame by R(%+.4f deg): "
                "the tensor of the site's E on the neighbour's field, an INTER-SITE IMPEDANCE, shown and "
                "never delivered" % -float(theta_deg))
    else:
        borrowed = (chan == "Hx", chan == "Hy")
        out[chan] = sensor[chan]
        note = ("%s borrowed and turned into the site's sensor frame by R(%+.4f deg); the site's own %s kept "
                "in its mean-field frame; E and Hz own"
                % (chan, -float(theta_deg), "Hy" if chan == "Hx" else "Hx"))
    C = basis_correction(theta_deg, borrowed)
    return dict(arrays=out, borrowed=borrowed, correction=C, note=note, theta_deg=float(theta_deg),
                is_rotation=bool(np.allclose(C @ C.T, np.eye(2))),
                inter_site=bool(whole_pair))


def correct_basis(z, t=None, correction=None):
    """(Z, T) with the H basis right-multiplied by the correction matrix."""
    C = np.eye(2) if correction is None else np.asarray(correction, float)
    zz = np.einsum("nij,jk->nik", np.asarray(z), C)
    if t is None:
        return zz, None
    return zz, np.einsum("nij,jk->nik", np.asarray(t), C)


def correct_errors(e, correction):
    """Errors combined in quadrature over the corrected columns."""
    C = np.asarray(correction, float)
    return np.sqrt(np.einsum("nij,jk->nik", np.asarray(e, float) ** 2, C ** 2))


def correct_edi(path, correction, note="") -> dict:
    """Right-multiply a written EDI's H basis in place, and read it back against the intended tensor."""
    from mt_metadata.transfer_functions.core import TF
    from pathlib import Path
    path = Path(path)
    tf = TF(fn=str(path))
    tf.read()
    Z = np.asarray(tf.impedance.values, complex)
    Zr, _ = correct_basis(Z, None, correction)
    try:
        tf.impedance_error = correct_errors(np.asarray(tf.impedance_error.values, float), correction)
    except Exception:
        pass
    tf.impedance = Zr
    try:
        if tf.has_tipper() and tf.tipper is not None:
            T = np.asarray(tf.tipper.values, complex)
            _z, Tr = correct_basis(Z, T, correction)
            tf.tipper = Tr
            try:
                tf.tipper_error = correct_errors(np.asarray(tf.tipper_error.values, float), correction)
            except Exception:
                pass
    except Exception:
        pass
    try:
        st = tf.station_metadata
        pp = list(st.transfer_function.processing_parameters or [])
        pp.append("h_basis_correction=[[%.6f, %.6f], [%.6f, %.6f]] applied on the right. %s"
                  % (correction[0][0], correction[0][1], correction[1][0], correction[1][1], note))
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
    fin = np.isfinite(Zr).all(axis=(1, 2)) & np.isfinite(Z2).all(axis=(1, 2))
    rel = (float(np.max(np.abs(Z2[fin] - Zr[fin]) / np.maximum(np.abs(Zr[fin]), 1e-30)))
           if fin.any() else np.nan)
    det = (float(np.max(np.abs(np.linalg.det(Z2[fin]) - np.linalg.det(Z[fin]) * np.linalg.det(correction))
                        / np.maximum(np.linalg.norm(Z[fin], axis=(1, 2)) ** 2, 1e-30)))
           if fin.any() else np.nan)
    return dict(path=str(path), n=int(fin.sum()), max_element_rel=rel, max_det_scaled=det,
                ok=bool(np.isfinite(rel) and rel <= 1e-6 and np.isfinite(det) and det <= 1e-5))


# ---------------------------------------------------------------- the lender and the reference

def reference_members(info: dict) -> list:
    """Every site the reference of one product is built from, the observatory included."""
    out = []
    if not isinstance(info, dict):
        return out
    if info.get("remote"):
        out.append(str(info["remote"]))
    for m in (info.get("members") or []):
        if m.get("role") != "refused" and m.get("name"):
            out.append(str(m["name"]))
    for k in ("weights",):
        for name in (info.get(k) or {}):
            out.append(str(name))
    return sorted(set(out))


def lender_in_reference(lender, info: dict) -> bool:
    """True where the lender is a member of the reference the pass reads.

    A reference sharing a channel with the local H is comparing a channel with itself, so a form whose
    lender is in its own reference is refused (vic_w3_lender.py:1-22).
    """
    return bool(lender) and str(lender) in set(reference_members(info))


# ---------------------------------------------------------------- the splice

def splice_period(period, rho_a, phase_a, rho_b, phase_b, sigma_a=None, sigma_b=None,
                  rho_tol=SPLICE_RHO, phase_tol=SPLICE_PHASE_DEG, sigma_k=SPLICE_SIGMA,
                  lo_s=SPLICE_TEST_LO_S, hi_s=SPLICE_TEST_HI_S) -> dict:
    """The shortest period from which two curves agree and keep agreeing up to hi_s.

    The looser of two rules decides: within `rho_tol` in apparent resistivity and `phase_tol` in phase, OR
    within `sigma_k` combined sigma. Above SPLICE_TEST_HI_S = 3000 s the two curves stop being compared with
    each other and start being compared with the shorter record's sampling noise, so the test stops there.
    A borrowed long end goes UNDER the site's own short periods, which is what the period returned is for.
    """
    p = np.asarray(period, float)
    ra, rb = np.asarray(rho_a, float), np.asarray(rho_b, float)
    pa, pb = np.asarray(phase_a, float), np.asarray(phase_b, float)
    m = (p >= lo_s) & (p <= hi_s) & np.isfinite(ra) & np.isfinite(rb) & (ra > 0) & (rb > 0) \
        & np.isfinite(pa) & np.isfinite(pb)
    idx = np.flatnonzero(m)
    if not len(idx):
        return dict(t_c=np.nan, n_tested=0, frac_agree=np.nan, rule="no period inside the test band")
    close = (np.abs(ra[idx] / rb[idx] - 1.0) <= rho_tol) & \
            (np.abs(((pa[idx] - pb[idx]) + 180.0) % 360.0 - 180.0) <= phase_tol)
    if sigma_a is not None and sigma_b is not None:
        sa, sb = np.asarray(sigma_a, float)[idx], np.asarray(sigma_b, float)[idx]
        comb = np.sqrt(np.nan_to_num(sa) ** 2 + np.nan_to_num(sb) ** 2)
        within = np.abs(ra[idx] - rb[idx]) <= sigma_k * comb
        close = close | np.where(np.isfinite(comb) & (comb > 0), within, False)
    t_c = np.nan
    for k in range(len(idx)):
        if close[k:].all():
            t_c = float(p[idx[k]])
            break
    return dict(t_c=t_c, n_tested=int(len(idx)), frac_agree=float(close.mean()),
                rule="within %.0f %% in rho and %.0f deg in phase OR within %.0f combined sigma, held to "
                     "%g s" % (100 * rho_tol, phase_tol, sigma_k, hi_s))
