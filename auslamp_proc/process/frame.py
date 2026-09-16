"""Signs and the frame, applied at processing time from decisions.csv.

The cache is the record as laid, so this module is where decisions.csv acts. The order is fixed: signs,
then the rotation, then the reference built from the rotated members, then the MTH5.

Signs. Each channel is multiplied by the decisions.csv cell sign_hx/hy/hz/ex/ey where that cell reads +1 or
-1. A cell reading `decide` is used as +1 and recorded as undecided, and every product built from it carries
a processing_parameters line naming the channels.

Rotation. The target's (Hx, Hy) pair is turned by theta = atan2(mean Hy, mean Hx), which puts the mean Hy at
zero: the mean-field frame, which takes out the fluxgate's hand-compass misalignment. E is left as laid.
Every member of a reference is turned into its own mean-field frame before it is stacked, and the
observatory's geographic X, Y are turned into the mean-field frame of the observatory record.
rot_regimes from decisions.csv splits the record into stretches rotated one at a time, each by its own mean
field, because the field did not move and the instrument did; rot_drop days come back NaN and are excluded
from the mean. The angle used is recorded as h_rotation_deg and the IGRF declination is recorded and not
applied.
Ported from qld_campaign.rotate (:306-309) and wamt_remotes.rot_h (:556-615).

The turn-back. A tensor served in the mean-field frame is turned to another frame by Z' = R Z R^T and
T' = T R^T with R = [[cos, sin], [-sin, cos]]; turn_tensor by the negative of the declination gives
geographic north.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")
H_PAIR = ("Hx", "Hy")
SIGN_COLUMN = {"Hx": "sign_hx", "Hy": "sign_hy", "Hz": "sign_hz", "Ex": "sign_ex", "Ey": "sign_ey"}
DECIDE = "decide"


def read_sign(cell) -> tuple[float, bool]:
    """A decisions.csv sign cell to (value, decided). `decide`, an empty cell and anything unreadable give
    (+1.0, False)."""
    s = str(cell).strip().lower()
    if s in ("", "nan", "none", DECIDE):
        return 1.0, False
    try:
        v = float(s)
    except ValueError:
        return 1.0, False
    if v not in (1.0, -1.0):
        return 1.0, False
    return v, True


def apply_signs(arrays: dict, decisions_row) -> tuple[dict, dict, list]:
    """(arrays with each channel multiplied by its sign, {channel: sign applied}, [undecided channels]).

    `decisions_row` is one row of decisions.csv, as a Series or a dict. A channel absent from the row is
    left alone and counted as undecided.
    """
    out, applied, undecided = {}, {}, []
    for ch in CHANNELS:
        if ch not in arrays:
            continue
        cell = None
        col = SIGN_COLUMN[ch]
        if decisions_row is not None:
            try:
                cell = decisions_row[col]
            except (KeyError, IndexError, TypeError):
                cell = None
        sign, decided = read_sign(cell)
        if not decided:
            undecided.append(ch)
        applied[ch] = float(sign)
        out[ch] = np.asarray(arrays[ch], float) * sign if sign != 1.0 else np.asarray(arrays[ch], float)
    for k, v in arrays.items():                       # anything that is not one of the five passes through
        if k not in out:
            out[k] = v
    return out, applied, undecided


def parse_regimes(cell) -> tuple:
    """A decisions.csv rot_regimes or rot_drop cell to ((day0, day1), ...) in days from the record start.

    The cell is a space- or comma-separated list of `start:end` pairs in days; an open end is written as an
    empty right-hand side, as in `21:`. An empty cell or `decide` gives ().
    """
    s = str(cell).strip()
    if s.lower() in ("", "nan", "none", DECIDE):
        return ()
    out = []
    for item in s.replace(",", " ").split():
        if ":" not in item:
            continue
        a, _, b = item.partition(":")
        try:
            d0 = float(a) if a.strip() else 0.0
            d1 = float(b) if b.strip() else None
        except ValueError:
            continue
        out.append((d0, d1))
    return tuple(out)


def _day_slice(n: int, fs: float, d0: float, d1) -> tuple[int, int]:
    i0 = int(round(float(d0) * 86400.0 * fs))
    i1 = n if d1 is None else min(n, int(round(float(d1) * 86400.0 * fs)))
    return max(0, min(i0, n)), max(0, min(i1, n))


def mean_field_angle(hx, hy) -> float:
    """theta = atan2(mean Hy, mean Hx) in degrees, over the finite samples. NaN where nothing is finite."""
    x, y = np.asarray(hx, float), np.asarray(hy, float)
    g = np.isfinite(x) & np.isfinite(y)
    if not g.any():
        return float("nan")
    return float(np.degrees(np.arctan2(y[g].mean(), x[g].mean())))


def rotate_pair(hx, hy, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(Hx, Hy) turned by angle_deg: Hx' = c Hx + s Hy, Hy' = -s Hx + c Hy."""
    a = np.radians(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    x, y = np.asarray(hx, float), np.asarray(hy, float)
    return c * x + s * y, -s * x + c * y


def rotate_to_mean_field(arrays: dict, regimes=None, drop=None, fs: float = 1.0):
    """(arrays with Hx, Hy in the mean-field frame, the angle used).

    The angle is one float where the record is turned as a whole, and a list of (day0, day1, angle) where
    `regimes` names more than one stretch. `drop` names day ranges the sensor was in transit or dead; those
    samples come back NaN and are excluded from every mean. Hz and the electric channels are not touched.
    """
    out = {k: (np.asarray(v, float).copy() if k in H_PAIR else v) for k, v in arrays.items()}
    n = len(out["Hx"])
    for d0, d1 in (drop or ()):
        i0, i1 = _day_slice(n, fs, d0, d1)
        for c in H_PAIR:
            out[c][i0:i1] = np.nan
    if not regimes:
        ang = mean_field_angle(out["Hx"], out["Hy"])
        if not np.isfinite(ang):
            return out, float("nan")
        out["Hx"], out["Hy"] = rotate_pair(out["Hx"], out["Hy"], ang)
        return out, round(float(ang), 4)
    turned = {c: np.full(n, np.nan) for c in H_PAIR}
    segs = []
    for d0, d1 in regimes:
        i0, i1 = _day_slice(n, fs, d0, d1)
        if i1 <= i0:
            continue
        ang = mean_field_angle(out["Hx"][i0:i1], out["Hy"][i0:i1])
        if not np.isfinite(ang):
            continue
        x, y = rotate_pair(out["Hx"][i0:i1], out["Hy"][i0:i1], ang)
        turned["Hx"][i0:i1], turned["Hy"][i0:i1] = x, y
        segs.append((float(d0), (None if d1 is None else float(d1)), round(float(ang), 4)))
    out["Hx"], out["Hy"] = turned["Hx"], turned["Hy"]
    return out, segs


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """R = [[cos, sin], [-sin, cos]] at angle_deg."""
    a = np.radians(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]], float)


def turn_tensor(z, t=None, angle_deg: float = 0.0):
    """(Z', T') turned by angle_deg: Z' = R Z R^T and T' = T R^T.

    `z` is (n_periods, 2, 2) and `t` is (n_periods, 1, 2) or (n_periods, 2), or None. Turning by minus the
    declination takes a tensor served in geomagnetic north to geographic north.
    """
    r = rotation_matrix(angle_deg)
    zz = np.asarray(z)
    zt = np.einsum("ij,njk,lk->nil", r, zz, r) if zz.ndim == 3 else r @ zz @ r.T
    if t is None:
        return zt, None
    tt = np.asarray(t)
    flat = tt.reshape(tt.shape[0], -1) if tt.ndim == 3 else tt
    turned = flat @ r.T
    return zt, turned.reshape(tt.shape)
