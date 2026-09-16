"""The INTERMAGNET one-second archive: coverage, missing days, fetch, and the window loader.

Archive layout: one parquet per observatory-year at <archive>/<CODE>_1sec_<year>.parquet, holding a complete
regular 1 s grid (31,536,000 rows in a common year) with NaN in the gaps, in ZSTD row groups of about 1.05 M
rows carrying full column statistics. Columns: DateTime, BX, BY, BZ (geographic X Y Z in nT), F, pub.

A year file is about 380 MB, so neither coverage nor load reads a whole one.

coverage reads only the row-group statistics in the file footer, 31 groups per year, and counts the rows each
group contributes to each day. A day is present when it holds at least half of its 86,400 samples.

load reads only the row groups whose DateTime statistics overlap the window, and scatters rows by their own
time stamp rather than assuming a contiguous grid.
Ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/obs_1sec.py:108-215 (load). Gaps up to
max_gap_s = 600 s are linearly filled and longer ones stay NaN, with `mask` False wherever a sample is not
real archive data; a gap touching either end of the window is never filled. No demeaning or detrending is
applied: a stack demeans its members before averaging and a single remote is not touched at all.

fetch_missing is the INTERMAGNET GIN request: IAGA-2002, XYZF, publicationState=best-avail, one day per
request, written to <archive>/_parts/<CODE>/ and consolidated into a year file only when every expected day of
that year is present. It defaults to dry_run=True and returns the (day, url) list it would fetch.
Ported from D:/BEN/2025_Ying_GICs_Paper/2025_Ying_GICs_Paper/scripts/fetch_observatory_1sec.py (fetch_day,
part_path, year_path, consolidate).

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GIN_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
CHANS = ("BX", "BY", "BZ")
_OUT = {"BX": "Hx", "BY": "Hy", "BZ": "Hz"}
FS = 1.0
SECONDS_PER_DAY = 86400


def archive_root(cfg: dict | None = None) -> Path:
    """The archive folder from survey.yaml `observatory.archive`, or the machine default."""
    default = Path("E:/MT_Timeseries_DATA/Geomagnetic_Observatory_Data/INTERMAGNET_1sec")
    if not cfg:
        return default
    return Path((cfg.get("observatory") or {}).get("archive") or default)


def year_path(archive, code: str, year: int) -> Path:
    return Path(archive) / ("%s_1sec_%d.parquet" % (code.upper(), year))


def part_path(archive, code: str, period) -> Path:
    p = pd.Timestamp(period)
    return Path(archive) / "_parts" / code.upper() / ("%s_%04d-%02d.parquet" % (code.upper(), p.year, p.month))


def _row_group_day_counts(path: Path) -> dict:
    """{date: rows} from the row-group statistics alone: a footer read, no column data.

    A row group's DateTime min and max bound the days it can hold. Groups are contiguous in this archive, so a
    group's rows are apportioned over the days it spans in proportion to each day's share of that span. The
    count is exact where a group lies inside one day and correct to a few rows at a group boundary.
    """
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(j).name for j in range(md.num_columns)]
    col = names.index("DateTime")
    out: dict = {}
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(col).statistics
        n = md.row_group(i).num_rows
        if st is None or not st.has_min_max:
            continue
        a = pd.Timestamp(st.min)
        b = pd.Timestamp(st.max)
        span = max(1.0, (b - a).total_seconds() + 1.0)
        for d in pd.date_range(a.normalize(), b.normalize(), freq="D"):
            lo = max(a, d)
            hi = min(b, d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            share = max(0.0, (hi - lo).total_seconds() + 1.0) / span
            out[d.date()] = out.get(d.date(), 0) + int(round(n * share))
    return out


def coverage(code: str, start, end, archive=None) -> pd.DataFrame:
    """One row per calendar day of [start, end]: date, n_samples present, whether the year file exists.

    Runs in milliseconds per year: only the parquet footers are read.
    """
    archive = Path(archive) if archive else archive_root()
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    counts: dict = {}
    files: dict = {}
    for y in range(s.year, e.year + 1):
        p = year_path(archive, code, y)
        files[y] = p.exists()
        if p.exists():
            counts.update(_row_group_day_counts(p))
    rows = []
    for d in pd.date_range(s, e, freq="D"):
        n = counts.get(d.date(), 0)
        rows.append(dict(day=d.date(), n_samples=int(n), year_file=files.get(d.year, False),
                         present=bool(n >= 0.5 * SECONDS_PER_DAY)))
    return pd.DataFrame(rows)


def missing_days(code: str, start, end, archive=None) -> list:
    """The calendar days of [start, end] the archive does not hold at least half of."""
    cov = coverage(code, start, end, archive)
    return [d for d, ok in zip(cov.day, cov.present) if not ok]


def gin_url(code: str, day) -> str:
    d = pd.Timestamp(day)
    return (f"{GIN_BASE}?Request=GetData&observatoryIagaCode={code.upper()}"
            f"&dataStartDate={d:%Y-%m-%d}&dataDuration=1&samplesPerDay=Second"
            f"&publicationState=best-avail&format=iaga2002&orientation=XYZF")


def fetch_day(code: str, day, tries: int = 3) -> pd.DataFrame:
    """One observatory-day of IAGA-2002 one-second data as a frame, empty where the GIN serves none.

    Values at or beyond 88888 are the IAGA-2002 fill and are masked to NaN.
    Ported from fetch_observatory_1sec.fetch_day.
    """
    import io
    url = gin_url(code, day)
    last = None
    cols = ["DateTime", "BX", "BY", "BZ", "F", "pub"]
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                txt = r.read().decode("utf-8", errors="replace")
            if "<html" in txt[:400].lower():
                return pd.DataFrame(columns=cols)
            m = re.search(r"^\s*Data Type\s+(\S+)", txt, flags=re.M)
            pub = m.group(1).lower() if m else "unknown"
            rows = [ln for ln in txt.splitlines()
                    if ln and ln[0] not in " #|" and not ln.startswith("DATE")]
            if not rows:
                return pd.DataFrame(columns=cols)
            d = pd.read_csv(io.StringIO("\n".join(rows)), sep=r"\s+", header=None,
                            names=["date", "time", "doy", "BX", "BY", "BZ", "F"], engine="c")
            t = pd.to_datetime(d.date + " " + d.time, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            out = pd.DataFrame({"DateTime": t, "BX": d.BX, "BY": d.BY, "BZ": d.BZ, "F": d.F})
            out[["BX", "BY", "BZ", "F"]] = out[["BX", "BY", "BZ", "F"]].mask(
                out[["BX", "BY", "BZ", "F"]].abs() >= 88888.0)
            out["pub"] = pub
            return out.dropna(subset=["DateTime"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as ex:
            last = ex
            time.sleep(3 * (k + 1))
    raise RuntimeError("%s %s: %s" % (code, pd.Timestamp(day).date(), last))


def fetch_missing(code: str, start, end, archive=None, dry_run: bool = True) -> list:
    """The days of [start, end] the archive lacks, as (day, url).

    With dry_run False each day is fetched into <archive>/_parts/<CODE>/<CODE>_<YYYY-MM>.parquet and a year is
    consolidated into <CODE>_1sec_<year>.parquet only when every expected day of that year is present.
    """
    archive = Path(archive) if archive else archive_root()
    days = missing_days(code, start, end, archive)
    plan = [(d, gin_url(code, d)) for d in days]
    if dry_run or not plan:
        return plan
    by_month: dict = {}
    for d, _ in plan:
        by_month.setdefault(pd.Timestamp(d).to_period("M"), []).append(d)
    for period, ds in sorted(by_month.items()):
        frames = []
        for d in ds:
            got = fetch_day(code, d)
            if len(got):
                frames.append(got)
        if not frames:
            continue
        p = part_path(archive, code, period.start_time)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).sort_values("DateTime").to_parquet(
            p, index=False, compression="zstd")
    for year in sorted({pd.Timestamp(d).year for d, _ in plan}):
        consolidate(code, year, archive)
    return plan


def consolidate(code: str, year: int, archive=None) -> str:
    """Merge <archive>/_parts/<CODE>/<CODE>_<year>-MM.parquet into the year file when the year is complete.

    The year file is rewritten only when every day of the year (to today, for the current year) is present in
    the merge; otherwise the parts are left in place and the reason is returned.
    Ported from fetch_observatory_1sec.consolidate.
    """
    archive = Path(archive) if archive else archive_root()
    parts = sorted((archive / "_parts" / code.upper()).glob("%s_%d-*.parquet" % (code.upper(), year)))
    yp = year_path(archive, code, year)
    frames = [pd.read_parquet(p) for p in parts]
    if yp.exists():
        frames.append(pd.read_parquet(yp))
    if not frames:
        return "%s %d: nothing to consolidate" % (code, year)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("DateTime").sort_values("DateTime")
    have = set(pd.DatetimeIndex(df.DateTime).normalize().date)
    last = min(date(year, 12, 31), datetime.now(timezone.utc).date())
    want = pd.date_range(date(year, 1, 1), last, freq="D").date
    missing = [d for d in want if d not in have]
    if missing:
        return "%s %d: %d days still missing (%s ...); year file untouched" % (
            code, year, len(missing), missing[0])
    df.to_parquet(yp, index=False, compression="zstd")
    for p in parts:
        os.remove(p)
    return "%s %d: consolidated %d rows into %s" % (code, year, len(df), yp.name)


def _runs(bad: np.ndarray):
    d = np.diff(np.r_[0, bad.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill_short(x, mask, max_gap):
    """Linearly fill runs of missing samples up to `max_gap`.

    A gap touching either end of the window is never filled: np.interp would extend the edge value across it.
    """
    bad = ~mask
    if not bad.any():
        return 0, 0, []
    runs = _runs(bad)
    lens = [e - s for s, e in runs]
    fillable = np.zeros(len(x), bool)
    left = []
    for s, e in runs:
        if (e - s) <= max_gap and s > 0 and e < len(x):
            fillable[s:e] = True
        else:
            left.append(e - s)
    n_fill = int(fillable.sum())
    if n_fill and mask.any():
        idx = np.arange(len(x))
        x[fillable] = np.interp(idx[fillable], idx[mask], x[mask])
    return n_fill, (max(left) if left else 0), lens


def load(code: str, t0_unix: int, n_samples: int, archive=None, max_gap_s: float = 600.0) -> dict:
    """One observatory's H over [t0, t0 + n) at 1 Hz, geographic XYZ, nT.

    Returns Hx Hy Hz (NaN in gaps longer than max_gap_s), mask (True where the sample is real archive data),
    t0, fs, pub_counts and gap statistics. A window crossing a year boundary is read from both files.
    Ported from obs_1sec.load.
    """
    import pyarrow.parquet as pq
    archive = Path(archive) if archive else archive_root()
    code = code.upper()
    t0, n = int(t0_unix), int(n_samples)
    out = {c: np.full(n, np.nan) for c in _OUT.values()}
    mask = np.zeros(n, bool)
    pub_counts: dict = {}
    y0 = int(str(np.datetime64(t0, "s"))[:4])
    y1 = int(str(np.datetime64(t0 + n - 1, "s"))[:4])
    files = []
    for y in range(y0, y1 + 1):
        p = year_path(archive, code, y)
        if p.exists():
            files.append(p)
    if not files:
        raise FileNotFoundError("no %s parquet covering %d..%d under %s" % (code, t0, t0 + n, archive))
    for p in files:
        pf = pq.ParquetFile(p)
        md = pf.metadata
        names = [md.schema.column(j).name for j in range(md.num_columns)]
        col = names.index("DateTime")
        has_pub = "pub" in names
        want = []
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(col).statistics
            if st is None or not st.has_min_max:
                want.append(i)
                continue
            a = np.datetime64(st.min, "s").astype("int64").item()
            b = np.datetime64(st.max, "s").astype("int64").item()
            if b >= t0 and a < t0 + n:
                want.append(i)
        for i in want:
            tb = pf.read_row_group(i, columns=["DateTime"] + list(CHANS) + (["pub"] if has_pub else []))
            ts = tb.column("DateTime").to_numpy(zero_copy_only=False)
            ts = ts.astype("datetime64[s]").astype("int64")
            sel = (ts >= t0) & (ts < t0 + n)
            if not sel.any():
                continue
            k = ts[sel] - t0
            good = np.ones(len(k), bool)
            vals = {}
            for c in CHANS:
                v = tb.column(c).to_numpy(zero_copy_only=False).astype(float)[sel]
                vals[c] = v
                good &= np.isfinite(v)
            for c in CHANS:
                out[_OUT[c]][k] = vals[c]
            mask[k] = good
            if has_pub:
                pub = tb.column("pub").to_numpy(zero_copy_only=False)[sel]
                u, cnt = np.unique(pub.astype(str), return_counts=True)
                for a_, b_ in zip(u, cnt):
                    pub_counts[str(a_)] = pub_counts.get(str(a_), 0) + int(b_)
            else:
                pub_counts["unknown"] = pub_counts.get("unknown", 0) + int(sel.sum())
    lens, n_fill, longest = [], 0, 0
    for c in _OUT.values():
        f, lg, ls = _fill_short(out[c], mask, int(max_gap_s))
        n_fill = max(n_fill, f)
        longest = max(longest, lg)
        lens = ls
    unfilled = ~np.isfinite(out["Hx"])
    gaps = dict(n_missing=int((~mask).sum()), n_filled=int(n_fill), n_unfilled=int(unfilled.sum()),
                n_gaps=len(lens), longest_gap_s=int(max(lens) if lens else 0),
                longest_unfilled_gap_s=int(longest), null_fraction=float((~mask).sum() / n))
    return dict(t0=t0, fs=FS, obs=code, n=n, mask=mask, pub_counts=pub_counts, gaps=gaps, **out)
