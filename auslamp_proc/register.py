"""The deployment register: spans, sites active per day, overlap, and the two groupings.

A reference can only help a target over the time the two were both recording, so the register is the first
filter on every reference choice made later.

groups(spans, min_overlap_days) is the candidate pool: the connected components of the graph whose edges join
two sites overlapping by at least min_overlap_days. A component's common_start and common_end are the window
every member shares, and are empty where the component is held together by a chain (A-B and B-C overlapping
does not make A and C overlap). A remote site and the fleet stack members for a target are drawn from its
component.

core_group(peak, spans, half_days) is the concurrent core: the sites whose record covers the whole of the
2 * half_days = 14 days centred on `peak`, that is start <= peak - 7 d and end >= peak + 7 d.
Ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/vic_waves_table.py:62-80 (derive). Members are
ordered by start time; ties keep the order of the `order` argument where one is given, otherwise they follow
the order pandas returns.

peaks_from_active takes the peak days from the local maxima of the sites-active-per-day curve. This is a
reading of the curve. It does not recover how any earlier table chose its peaks: in the AusLAMP Victoria table
of 2026-09-10 the peak sits at the start of its own plateau in 8 rows and at the end in 12, so a comparison
against that table takes its peaks as an input.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_DAYS = 7


def spans(sites: pd.DataFrame, start_col="start_utc", end_col="end_utc") -> pd.DataFrame:
    """(site, start, end, days, instrument, layout) for every site with a parseable span, sorted by start."""
    rows = []
    for _, r in sites.iterrows():
        s, e = pd.Timestamp(r[start_col]), pd.Timestamp(r[end_col])
        if pd.isna(s) or pd.isna(e):
            continue
        rows.append(dict(site=r["site"], start=s, end=e,
                         days=float((e - s).total_seconds() / 86400.0),
                         instrument=r.get("instrument", ""), layout=r.get("layout", "")))
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def active_per_day(sp: pd.DataFrame) -> pd.Series:
    """How many sites were recording on each calendar day of the survey."""
    if not len(sp):
        return pd.Series(dtype=int)
    days = pd.date_range(sp.start.min().normalize(), sp.end.max().normalize(), freq="D")
    n = np.zeros(len(days), int)
    idx = {d: i for i, d in enumerate(days)}
    for r in sp.itertuples():
        a, b = idx[r.start.normalize()], idx[r.end.normalize()]
        n[a:b + 1] += 1
    return pd.Series(n, index=days, name="active")


def overlap_matrix(sp: pd.DataFrame) -> pd.DataFrame:
    """Days of overlap between every pair of sites; the diagonal is each site's own length."""
    names = list(sp.site)
    st = sp.start.values.astype("datetime64[s]").astype("int64")
    en = sp.end.values.astype("datetime64[s]").astype("int64")
    lo = np.maximum.outer(st, st)
    hi = np.minimum.outer(en, en)
    ov = np.clip(hi - lo, 0, None) / 86400.0
    return pd.DataFrame(np.round(ov, 2), index=names, columns=names)


def groups(sp: pd.DataFrame, min_overlap_days: float) -> pd.DataFrame:
    """Connected components of the graph 'two sites overlap by >= min_overlap_days'.

    Each row carries the members, their count, and the window every member shares (max start, min end).
    common_days is 0 and the window is empty where the component has no window in common.
    """
    if not len(sp):
        return pd.DataFrame(columns=["group", "n", "common_start", "common_end", "common_days", "members"])
    ov = overlap_matrix(sp).values
    n = len(sp)
    adj = ov >= float(min_overlap_days)
    np.fill_diagonal(adj, False)
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.flatnonzero(adj[k]):
                if j not in seen:
                    seen.add(int(j)); stack.append(int(j))
        comps.append(sorted(comp, key=lambda k: sp.start.iloc[k]))
    rows = []
    for gi, comp in enumerate(sorted(comps, key=lambda c: sp.start.iloc[c[0]]), start=1):
        members = [sp.site.iloc[k] for k in comp]
        cs = max(sp.start.iloc[k] for k in comp)
        ce = min(sp.end.iloc[k] for k in comp)
        cd = (ce - cs).total_seconds() / 86400.0
        # a component held together by a chain has no shared window; it is reported empty, not reversed
        has = cd > 0
        rows.append(dict(group="G%02d" % gi, n=len(comp),
                         common_start=str(cs) if has else "", common_end=str(ce) if has else "",
                         common_days=round(cd, 2) if has else 0.0, members=" ".join(members)))
    return pd.DataFrame(rows)


def core_group(peak, sp: pd.DataFrame, half_days: int = HALF_DAYS, order=()) -> dict:
    """The sites covering the 2 * half_days window centred on `peak`, and the window they share.

    Members are ordered by start time. `order` is a previous table's member order and is kept for the members
    it still names, so a comparison against that table does not turn on a tie-break.
    Ported from vic_waves_table.derive.
    """
    pk = pd.Timestamp(peak)
    S = dict(zip(sp.site, sp.start))
    E = dict(zip(sp.site, sp.end))
    # the spans carry UTC and a peak given as a bare date does not; both are put on the same basis
    tz = next((v.tz for v in S.values()), None)
    if tz is not None and pk.tz is None:
        pk = pk.tz_localize(tz)
    elif tz is None and pk.tz is not None:
        pk = pk.tz_localize(None)
    a, b = pk - pd.Timedelta(days=half_days), pk + pd.Timedelta(days=half_days)
    core = [s for s in S if S[s] <= a and E[s] >= b]
    known = [s for s in order if s in core]
    core = known + sorted((s for s in core if s not in known), key=lambda s: S[s])
    if not core:
        return dict(peak=str(pd.Timestamp(peak).date()), n_core=0, common_start="", common_end="",
                    days=0, core="")
    cs = max(S[s] for s in core)
    ce = min(E[s] for s in core)
    return dict(peak=str(pd.Timestamp(peak).date()), n_core=len(core), common_start=_fmt(cs),
                common_end=_fmt(ce), days=int((ce - cs).total_seconds() // 86400), core=" ".join(core))


def _fmt(ts) -> str:
    """A timestamp as 'YYYY-MM-DD hh:mm:ss[.ffffff]'. All times are UTC, so the offset is dropped."""
    ts = pd.Timestamp(ts)
    return str(ts.tz_localize(None) if ts.tz is not None else ts)


def peaks_from_active(active: pd.Series, min_sites: int = 2) -> list:
    """Local maxima of the sites-active-per-day curve, one day per plateau.

    A plateau is reported once, at its first day. Any day of a plateau gives the same members.
    """
    if not len(active):
        return []
    v = active.values
    out = []
    i = 0
    while i < len(v):
        if v[i] < min_sites:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        left = v[i - 1] if i > 0 else -1
        right = v[j + 1] if j + 1 < len(v) else -1
        if v[i] > left and v[i] > right:
            out.append(active.index[i])
        i = j + 1
    return out


def core_groups(sp: pd.DataFrame, peaks=None, half_days: int = HALF_DAYS,
                by: str = "instrument") -> pd.DataFrame:
    """core_group over a list of peak days, one row per peak and per value of `by`.

    With peaks=None the peaks come from peaks_from_active over the whole register. Rows with no core are
    dropped.
    """
    if peaks is None:
        peaks = peaks_from_active(active_per_day(sp))
    rows = []
    tiers = sorted(sp[by].unique()) if by and by in sp.columns else [""]
    for pk in peaks:
        for t in tiers:
            sub = sp[sp[by] == t] if t else sp
            d = core_group(pk, sub, half_days)
            if d["n_core"]:
                d[by] = t
                rows.append(d)
    cols = ["peak", "n_core", "common_start", "common_end", "days"] + ([by] if by else []) + ["core"]
    return pd.DataFrame(rows, columns=cols)
