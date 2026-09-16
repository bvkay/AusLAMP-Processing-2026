"""LEMI-424: the .INF power-up block, the daily TXT head and tail, and the multi-deployment split.

Tree: <site>/DATA<NNNN>/<YYYYMMDDhhmm>.TXT, one file per day, 24 whitespace columns at 1 Hz, with zero, one or
two .INF power-up blocks beside them. DATA<NNNN> is the logger serial and can be a previous logger's.

Columns, read from the files (2017-12-14, VIC016R):

    0-5   year month day hour minute second
    6-8   Bx By Bz            nT
    9-10  temperature of the electronics and of the fluxgate, degC
    11-14 E1 E2 E3 E4         uV/m, already divided by the %L lengths: never divide by L again
    15    battery V
    16    elevation m
    17-18 latitude DDMM.MMMM and its hemisphere letter
    19-20 longitude DDDMM.MMMM and its hemisphere letter
    21-23 satellites, fix type, a spare

Two rules this module applies.

1. The .INF block in force is the last one written at or before the first data file. An earlier block can
   carry setup dipole lengths that never applied to the recording, and a block dated after the first data file
   is rejected (one AusLAMP site elsewhere carries a block stamped 2033 from a lost clock).
   Ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/wamt_esperance.py:140-205 (_ddmm, parse_inf,
   site_inf).
2. One DATA folder can hold more than one deployment. deployments_in_folder clusters the daily files by their
   first row's position and returns the clusters; it does not choose between them. Which cluster is the site
   is recorded in survey.yaml raw_overrides. In AusLAMP Victoria VIC058R's folder holds 2 deployments,
   VIC065R's 4 and VIC067R's 3.

txt_first_last reads the first line and the last 4 KB of a daily file by seek. The files are 13 MB.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TAIL_BYTES = 4096
DEPLOYMENT_SPLIT_KM = 1.0        # first rows more than this far apart are different deployments


def ddmm(value, hemi: str) -> float:
    """'3756.1009' with 'S' -> -37.935015 degrees. Ported from wamt_esperance._ddmm."""
    v = float(value)
    d = int(v // 100)
    return (d + (v - 100 * d) / 60.0) * (-1.0 if str(hemi).upper() in ("S", "W") else 1.0)


def parse_inf(path) -> dict:
    """One `.INF` power-up block. %L values are in scientific notation; %Lat/%Lon are DDMM.MMMM plus a letter.

    `firmware` is normalised: the file writes '%FIRMWARE Ver.1.1' and the value wanted is '1.1'.
    Ported from wamt_esperance.parse_inf.
    """
    path = Path(path)
    txt = path.read_text(errors="ignore")
    m = re.match(r"^(\d{12})", path.name)
    out = {"inf_path": str(path), "inf_key": m.group(1) if m else ""}
    m = re.search(r"%Site\s+(\S+)", txt)
    out["inf_site"] = m.group(1) if m else ""
    m = re.search(r"%LEMI424\s*#(\S+)", txt)
    out["serial"] = m.group(1) if m else ""
    m = re.search(r"%FIRMWARE\s+(\S+)", txt)
    fw = m.group(1) if m else ""
    out["firmware"] = re.sub(r"^Ver\.?", "", fw, flags=re.I)
    for i in (1, 2, 3, 4):
        m = re.search(rf"%L{i}\s*=\s*(\S+)", txt)
        out["L%d" % i] = float(m.group(1)) if m else np.nan
    m = re.search(r"%Lat\s+([\d.]+)\s*,\s*([NS])", txt)
    out["inf_lat"] = ddmm(*m.groups()) if m else np.nan
    m = re.search(r"%Lon\s+([\d.]+)\s*,\s*([EW])", txt)
    out["inf_lon"] = ddmm(*m.groups()) if m else np.nan
    m = re.search(r"%Alt\s+([-\d.]+)", txt)
    out["inf_alt"] = float(m.group(1)) if m else np.nan
    out["inf_missing"] = False
    return out


def _empty_inf() -> dict:
    return dict(inf_path="", inf_key="", inf_site="", serial="", firmware="",
                L1=np.nan, L2=np.nan, L3=np.nan, L4=np.nan,
                inf_lat=np.nan, inf_lon=np.nan, inf_alt=np.nan, inf_missing=True)


def site_inf(datadir, first_key: str) -> tuple[dict, list[dict]]:
    """(the block in force, every block found) for a DATA<NNNN> folder.

    In force = the last block whose stamp is on or before the day of the first data file. A folder with no
    block returns an empty block with inf_missing True; 8 of the 19 AusLAMP Victoria LEMI sites have no .INF
    and take their dipole lengths from the release MTH5.
    Ported from wamt_esperance.site_inf.
    """
    datadir = Path(datadir)
    infs = []
    for f in sorted(datadir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".inf" and not f.name.startswith("._"):
            infs.append(parse_inf(f))
    if not infs:
        return _empty_inf(), []
    named = [i for i in infs if i["inf_site"]]
    pool = named or infs
    ok = [i for i in pool if i["inf_key"] and i["inf_key"][:8] <= str(first_key)[:8]]
    gov = max(ok, key=lambda i: i["inf_key"]) if ok else pool[0]
    return gov, infs


def txt_first_last(path) -> tuple[list[str], list[str]]:
    """(first row, last row) of a daily TXT as lists of fields, by seek.

    The last TAIL_BYTES = 4096 bytes are read and the last complete line of at least 21 fields is taken.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="ignore").split()
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read().decode("ascii", errors="ignore").splitlines()
    last = []
    for line in reversed(tail):
        fields = line.split()
        if len(fields) >= 21:
            last = fields
            break
    return first, last


def row_time(row) -> datetime:
    """The UTC time of one data row (columns 0-5)."""
    y, mo, d, h, mi, s = (int(float(row[i])) for i in range(6))
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def position_from_rows(rows) -> dict:
    """Median position over a list of data rows: latitude, longitude, elevation (m), satellites, scatter (m).

    Columns 17-20 are DDMM.MMMM with a hemisphere letter; column 16 is elevation in m and column 21 the
    satellite count.
    """
    la, lo, el, ns = [], [], [], []
    for r in rows:
        if len(r) < 22:
            continue
        try:
            la.append(ddmm(r[17], r[18]))
            lo.append(ddmm(r[19], r[20]))
            el.append(float(r[16]))
            ns.append(int(float(r[21])))
        except (ValueError, IndexError):
            continue
    if not la:
        return dict(lat=np.nan, lon=np.nan, elev_m=np.nan, n_sat=0, n_rows=0, scatter_m=np.nan)
    la_a, lo_a = np.asarray(la), np.asarray(lo)
    dy = (la_a - np.median(la_a)) * 111320.0
    dx = (lo_a - np.median(lo_a)) * 111320.0 * np.cos(np.radians(np.median(la_a)))
    return dict(lat=float(np.median(la_a)), lon=float(np.median(lo_a)),
                elev_m=float(np.median(el)), n_sat=int(min(ns)) if ns else 0,
                n_rows=len(la),
                scatter_m=float(np.percentile(np.hypot(dx, dy), 95)) if len(la) > 1 else 0.0)


def daily_files(datadir) -> list[Path]:
    """The daily TXT files of one DATA<NNNN> folder, in name order, AppleDouble copies dropped."""
    datadir = Path(datadir)
    return sorted(p for p in datadir.iterdir()
                  if p.is_file() and p.suffix.upper() == ".TXT" and not p.name.startswith("._"))


def deployments_in_folder(datadir, split_km: float = DEPLOYMENT_SPLIT_KM) -> list[dict]:
    """Cluster a DATA folder's daily files into deployments by their first row's position.

    A new cluster starts where a file's first fix is more than `split_km` = 1.0 km from the running cluster's
    median. Returns one dict per cluster with its files, span and position. Which cluster is the site is set in
    survey.yaml raw_overrides, not here.
    """
    from ..geo import distance_km
    out = []
    cur = None
    for p in daily_files(datadir):
        first, _ = txt_first_last(p)
        pos = position_from_rows([first])
        if not np.isfinite(pos["lat"]):
            continue
        if cur is not None and distance_km((cur["lat"], cur["lon"]), (pos["lat"], pos["lon"])) <= split_km:
            cur["files"].append(p)
            cur["lat"] = float(np.median([f[0] for f in cur["_fixes"]] + [pos["lat"]]))
            cur["lon"] = float(np.median([f[1] for f in cur["_fixes"]] + [pos["lon"]]))
            cur["_fixes"].append((pos["lat"], pos["lon"]))
        else:
            cur = dict(files=[p], lat=pos["lat"], lon=pos["lon"], _fixes=[(pos["lat"], pos["lon"])])
            out.append(cur)
    for c in out:
        c.pop("_fixes", None)
        c["n_files"] = len(c["files"])
        c["first"] = c["files"][0].name[:12]
        c["last"] = c["files"][-1].name[:12]
        c["files"] = [str(p) for p in c["files"]]
    return out
