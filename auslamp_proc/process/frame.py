"""The analyst's decisions and the frame, applied at processing time from decisions.csv.

The cache is the record as laid, so this module is where decisions.csv acts. The order is fixed:
apply_decisions, then the rotation, then the reference built from the rotated members, then the MTH5.

apply_decisions is the ONE PLACE. Every reader of a site's own channels calls it and nothing applies a
decision anywhere else: the transfer function path (process.run.load_local through raw.cache.load_decided),
the members and remotes of a reference (process.references.Store.rotated) and workbook 05's forms
(site.forms.load_local). Its order is fixed and tested: e_exchange, h_exchange, h_gain, e_shift_s, the
signs, then the lender's channels. The wiring exchanges come first because they say which line a channel
carried; the gain is a calibration of the channel as wired; the shift is a delay of the E line as wired; the
signs are of the LINE and not of the channel it was recorded on, so they come after the exchanges; the
lender's channels come last because they arrive carrying the LENDER's own decisions and must not take the
borrowing site's.

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
E_PAIR = ("Ex", "Ey")
H_TRIPLE = ("Hx", "Hy", "Hz")
SIGN_COLUMN = {"Hx": "sign_hx", "Hy": "sign_hy", "Hz": "sign_hz", "Ex": "sign_ex", "Ey": "sign_ey"}
DECIDE = "decide"
ASSUME = "assume:"
NOTHING = ("", "nan", "none", "<na>", DECIDE)


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


# ---------------------------------------------------------------- the decision cells

def _cell(row, column):
    """One cell of a decisions.csv row, as a string; '' where the row has no such column."""
    if row is None:
        return ""
    try:
        v = row[column]
    except (KeyError, IndexError, TypeError):
        return ""
    return "" if v is None else str(v)


def read_yes_no(cell) -> tuple[bool, bool]:
    """A yes/no decision cell to (applied, decided). `decide`, empty and anything unreadable give
    (False, False), so an undecided cell changes nothing and is recorded as open."""
    s = str(cell).strip().lower()
    if s in ("yes", "y", "true"):
        return True, True
    if s in ("no", "n", "false"):
        return False, True
    return False, False


def read_number(cell, default=1.0) -> tuple[float, bool]:
    """A numeric decision cell to (value, decided). `decide`, empty and anything unreadable give
    (default, False)."""
    s = str(cell).strip().lower()
    if s.startswith(ASSUME):
        s = s[len(ASSUME):].strip()
    if s in NOTHING:
        return float(default), False
    try:
        v = float(s)
    except ValueError:
        return float(default), False
    if not np.isfinite(v):
        return float(default), False
    return float(v), True


def read_name(cell) -> str:
    """A site-name decision cell to a name, or '' for `none`, `decide`, empty and unreadable."""
    s = str(cell).strip()
    return "" if s.lower() in NOTHING else s


def read_lender(row) -> str:
    """The site decisions.csv h_lender names, or '' where it names none."""
    return read_name(_cell(row, "h_lender"))


def read_lender_channels(row) -> list:
    """The channels h_lender_channels names, defaulting to all three magnetic channels.

    Electrics are never borrowed (site/replace.py:29-34), so a name that is not Hx, Hy or Hz is dropped and
    the caller records it.
    """
    s = str(_cell(row, "h_lender_channels")).strip()
    if s.lower() in NOTHING or s.lower() == "all":
        return list(H_TRIPLE)
    want = [w.strip().capitalize() for w in s.replace(",", " ").split()]
    return [c for c in H_TRIPLE if c in want]


def dipole_pair(site_row) -> tuple[float, float]:
    """(the north arm, the east arm) in metres from a sites.csv row; NaN where the cell says nothing.

    An `assume:<value>` cell is used, as raw.cache.dipole_value uses it: the assumption is already carried
    into the cache and into every product's provenance.
    """
    out = []
    for col in ("dipole_n_m", "dipole_e_m"):
        s = str(_cell(site_row, col)).strip()
        if s.lower().startswith(ASSUME):
            s = s[len(ASSUME):]
        try:
            v = float(s)
        except ValueError:
            v = float("nan")
        out.append(v)
    return out[0], out[1]


def to_sensor_frame(pair, angle_deg):
    """A mean-field-frame pair turned by minus `angle_deg` into a site's sensor frame.

    H_sensor = R(-t) H_meanfield with R(a) = [[cos, sin], [-sin, cos]]. The same formula site.replace
    carries (vic_w3_vic040.py:337-348), kept here because the lender's channels are substituted before any
    rotation and process may not import site.
    """
    a = np.radians(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    x, y = np.asarray(pair["Hx"], float), np.asarray(pair["Hy"], float)
    return {"Hx": c * x - s * y, "Hy": s * x + c * y}


def apply_decisions(arrays: dict, decision_row, site_row=None, lender_arrays=None, fs: float = 1.0):
    """(the arrays with every decisions.csv decision applied, the applied decisions as strings, a record).

    THE ONE PLACE. The order is fixed and tested:

        e_exchange   the two recorded E lines were wired to each other's channel. The channels are swapped
                     and each is rescaled by the ratio of the arm length the cache was built with to the
                     length of the line the channel actually carried. The cache holds mV/km computed with
                     the wrong length for a swapped channel: Ex_cache = V_east / (L_N g), so the true east
                     line is Ex_cache x (L_N / L_E) and the true north line is Ey_cache x (L_E / L_N), and
                     the swap puts each back on its own channel:

                         Ex_out = Ey_cache x (L_E / L_N),   Ey_out = Ex_cache x (L_N / L_E)

                     A missing or non-positive arm length leaves the level alone and is recorded.
        h_exchange   Hx and Hy were wired to each other's channel; the two are swapped.
        h_gain       every magnetic channel is divided by it. The site's magnetometer over-reads by this
                     factor, so Z = E/H is low by it until the division.
        e_shift_s    the E lines are advanced by this many seconds with the Lanczos delay of align.shift:
                     out[t] takes the recorded value at t + s, which is the correction for an E line that
                     LAGS H by s. Both electric channels are shifted at the site's own rate.
        the signs    apply_signs, on the lines as they now stand.
        h_lender     the channels of h_lender_channels are taken from `lender_arrays`, which the caller has
                     already decided, rotated into the LENDER's own mean-field frame and placed on this
                     site's grid. The horizontal pair is turned into the borrowing site's sensor frame by
                     R(-t) with t the site's own mean-field angle, so that the rotation downstream leaves
                     the lender's own mean-field pair; Hz is substituted as it stands. The lender's channels
                     never take the borrowing site's signs.

    `arrays` may hold any subset of the five channels: a decision whose channels are absent is not applied
    and is not recorded as applied. `record` carries the signs, the undecided sign channels, the angle used
    for the lender and the open decisions.
    """
    out = {}
    for k, v in arrays.items():
        out[k] = np.asarray(v, float).copy() if k in CHANNELS else v
    applied: list = []
    rec: dict = dict(e_exchange=False, h_exchange=False, h_gain=1.0, e_shift_s=0.0, lender="",
                     lender_channels=[], lender_angle_deg=float("nan"), lender_lag_s=float("nan"),
                     open=[], notes=[])

    # ---------------------------------------------------------------- e_exchange
    want, decided = read_yes_no(_cell(decision_row, "e_exchange"))
    if not decided:
        rec["open"].append("e_exchange")
    if want and all(c in out for c in E_PAIR):
        ln, le = dipole_pair(site_row)
        ok = np.isfinite(ln) and np.isfinite(le) and ln > 0 and le > 0
        fx, fy = (le / ln, ln / le) if ok else (1.0, 1.0)
        if not ok:
            rec["notes"].append("e_exchange at a site whose arm lengths are not both known: the channels "
                                "are swapped and neither level is rescaled")
        out["Ex"], out["Ey"] = out["Ey"] * fx, out["Ex"] * fy
        rec["e_exchange"] = True
        applied.append("e_exchange=yes: the two recorded E lines were wired to each other's channel; Ex "
                       "takes the Ey channel scaled by L_E/L_N = %s and Ey takes the Ex channel scaled by "
                       "L_N/L_E = %s (L_N %s m, L_E %s m)"
                       % (("%.6f" % fx) if ok else "1 (no arm length)", ("%.6f" % fy) if ok else
                          "1 (no arm length)", ln, le))

    # ---------------------------------------------------------------- h_exchange
    want, decided = read_yes_no(_cell(decision_row, "h_exchange"))
    if not decided:
        rec["open"].append("h_exchange")
    if want and all(c in out for c in H_PAIR):
        out["Hx"], out["Hy"] = out["Hy"], out["Hx"]
        rec["h_exchange"] = True
        applied.append("h_exchange=yes: Hx and Hy were wired to each other's channel; the two are swapped "
                       "before the signs and before the rotation")

    # ---------------------------------------------------------------- h_gain
    gain, decided = read_number(_cell(decision_row, "h_gain"), 1.0)
    if not decided:
        rec["open"].append("h_gain")
    if decided and gain not in (0.0, 1.0) and any(c in out for c in H_TRIPLE):
        got = [c for c in H_TRIPLE if c in out]
        for c in got:
            out[c] = out[c] / gain
        rec["h_gain"] = float(gain)
        applied.append("h_gain=%g: %s divided by it; the magnetometer over-reads by this factor, so the "
                       "impedance is low by it until the division" % (gain, ", ".join(got)))

    # ---------------------------------------------------------------- e_shift_s
    shift_s, decided = read_number(_cell(decision_row, "e_shift_s"), 0.0)
    if not decided:
        rec["open"].append("e_shift_s")
    if decided and shift_s != 0.0 and any(c in out for c in E_PAIR):
        from . import align
        got = [c for c in E_PAIR if c in out]
        for c in got:
            out[c] = align.shift(out[c], float(shift_s), float(fs))
        rec["e_shift_s"] = float(shift_s)
        applied.append("e_shift_s=%+g s: %s advanced by it at %g Hz with the Lanczos delay of "
                       "process.align.shift (out[t] takes the recorded value at t %+g s), the correction "
                       "for an E line that lags H" % (shift_s, ", ".join(got), fs, shift_s))

    # ---------------------------------------------------------------- the signs
    out, signs, undecided = apply_signs(out, decision_row)
    rec["signs"], rec["undecided"] = signs, undecided

    # ---------------------------------------------------------------- h_lender
    name = read_lender(decision_row)
    if str(_cell(decision_row, "h_lender")).strip().lower() in ("", "nan", "<na>", DECIDE):
        rec["open"].append("h_lender")          # `none` is a decision; an empty cell and `decide` are not
    if name:
        rec["lender"] = name
        chans = [c for c in read_lender_channels(decision_row) if c in out]
        if lender_arrays is None:
            rec["notes"].append("h_lender names %s and no lender record was handed in: the site's own "
                                "channels stand" % name)
        elif not chans:
            rec["notes"].append("h_lender names %s and h_lender_channels names no magnetic channel this "
                                "record carries: the site's own channels stand" % name)
        else:
            ang = 0.0
            if all(c in out for c in H_PAIR):
                a = mean_field_angle(out["Hx"], out["Hy"])
                ang = float(a) if np.isfinite(a) else 0.0
            rec["lender_angle_deg"] = ang
            pair = {c: lender_arrays[c] for c in H_PAIR if c in lender_arrays}
            turned = to_sensor_frame(pair, ang) if len(pair) == 2 else pair
            for c in chans:
                if c in ("Hx", "Hy") and c in turned:
                    out[c] = np.asarray(turned[c], float)
                elif c in lender_arrays:
                    out[c] = np.asarray(lender_arrays[c], float)
            rec["lender_channels"] = list(chans)
            applied.append("h_lender=%s: %s taken from %s on this site's grid, carrying %s's own decisions "
                           "and its own mean-field frame; the horizontal pair is turned into this site's "
                           "sensor frame by R(%+.4f deg), so the rotation downstream leaves the lender's "
                           "mean-field pair" % (name, ", ".join(chans), name, name, -ang))
            if all(c in chans for c in H_PAIR):
                applied.append("h_lender=%s: BOTH horizontal channels are borrowed, so the tensor is this "
                               "site's E on the field %s measured -- an inter-site impedance "
                               "(site/replace.py:29-34)" % (name, name))
    for note in rec["notes"]:
        applied.append("note: %s" % note)
    return out, applied, rec


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
