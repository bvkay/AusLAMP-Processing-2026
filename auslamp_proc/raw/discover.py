"""A raw folder to one row per site, read from file names alone.

discover opens no data file. It walks the tree, splits each name into (stem, stamp, extension) and reports
the day folders or daily files, the first and last stamp, every file stem seen, whether days are zipped,
which sidecars are present, and what was dropped and why. Header reads belong to raw.scan, which reads two
files per site.

Layouts:

    edl_dayofyear    <site>/<DDD>/<stem><YYMMDDhhmmss>.<BX BY BZ EX EY TP ...>   PR6-24
    lemi_data_nnnn   <site>/DATA<NNNN>/<YYYYMMDDhhmm>.TXT with 0-2 .INF          LEMI-424, field folders
    lemi_zip         <site>.zip, one per site                                     GA releases; not implemented

Drop rules:

    LAB_* and *_test*      bench recordings, dated years before the deployment
    non-numeric folders    config, log and temp sit beside the day folders
    ._*                    AppleDouble copies
    .bat .lnk              operator scripts left in a day folder
    a stem with no stamp   nothing can be placed on the time axis without one
    a foreign stem whose stamps lie more than FOREIGN_STEM_DAYS = 30 days outside the span of the site's own
    stem files; it is a different deployment left in the folder, and is named in the row's notes

In AusLAMP Victoria the last rule drops VIC062's CALP28_ and HB05_ hours of 2013-06-19, which precede the
site's own record by 356 days, and keeps the four cases where a logger had not yet been renamed: VIC057's
VIC84 day, VIC086's VIC124 day, VIC041's single VIC66 file and VIC100's first 14 minutes as VIC1003. In
AusLAMP Queensland it keeps Q44's first hour, written Q68N_061023025217, and Q55's first hours, written
Q57N_061024.

Extensions are matched case-insensitively. stem_styles names every stem kept; notes names every stem dropped.

A survey whose site folders sit directly under raw_root sets `subdir` to an empty string, and a folder beside
them that is not a site (AusLAMP Queensland Phase 1 ships KML/) is named in the instrument block's `exclude`
list.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from . import edl as edl_raw

FOREIGN_STEM_DAYS = 30

DISCOVERY_COLUMNS = [
    "site", "raw_path", "instrument", "layout",
    "n_day_folders", "n_daily_files", "n_data_files",
    "first_stamp", "last_stamp", "stem_styles", "zipped", "sidecars", "n_inf",
    "dropped", "date_correction", "first_file", "last_file",
]

_DROP_NAME = re.compile(r"^(LAB_)|(.*_test)", re.I)
_DROP_EXT = {"bat", "lnk", "txt~", "db"}


def _drop_reason(name: str) -> str:
    if name.startswith("._"):
        return "AppleDouble"
    if _DROP_NAME.match(name):
        return "LAB_/test"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in _DROP_EXT:
        return "operator script"
    return ""


def _stamp_days_apart(a: str, b: str) -> float:
    """Whole days between two YYMMDDhhmmss stamps, ignoring the rollover (both sides share it)."""
    from datetime import datetime
    def dt(s):
        return datetime(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]),
                        int(s[6:8]), int(s[8:10]), int(s[10:12]))
    return abs((dt(a) - dt(b)).total_seconds()) / 86400.0


def _discover_edl_site(site: str, site_dir: Path, years) -> dict:
    day_folders, zips, non_numeric = [], [], []
    for e in os.scandir(site_dir):
        if e.is_dir():
            (day_folders if e.name.isdigit() else non_numeric).append(e.name)
        elif e.name.lower().endswith(".zip"):
            zips.append(e.name)
        else:
            non_numeric.append("file:" + e.name)

    stems: dict[str, list[str]] = {}
    paths: dict[str, str] = {}
    sidecars: dict[str, int] = {}
    dropped: dict[str, int] = {}
    n_data = 0
    unknown_ext: dict[str, int] = {}
    for d in day_folders:
        for f in os.scandir(site_dir / d):
            if not f.is_file():
                continue
            why = _drop_reason(f.name)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            stem, stamp, ext = edl_raw.split_stem(f.name)
            up = ext.upper()
            if stamp is None:
                dropped["no stamp"] = dropped.get("no stamp", 0) + 1
                continue
            if up in edl_raw.DATA_EXT:
                stems.setdefault(stem, []).append(stamp)
                paths.setdefault(stamp + "|" + stem, f.path)
                n_data += 1
            elif ext in edl_raw.SIDECAR_EXT or up == "TP":
                sidecars[ext] = sidecars.get(ext, 0) + 1
            else:
                unknown_ext[ext] = unknown_ext.get(ext, 0) + 1

    # a foreign stem whose stamps lie more than FOREIGN_STEM_DAYS outside the own-stem span is dropped
    own = [s for s in stems if s.rstrip("_").upper() == site.upper()]
    own_stamps = sorted(x for s in own for x in stems[s])
    kept, foreign_dropped = {}, []
    for s, ss in stems.items():
        if s in own or not own_stamps:
            kept[s] = ss
            continue
        lo, hi = min(ss), max(ss)
        gap = min(_stamp_days_apart(lo, own_stamps[0]), _stamp_days_apart(hi, own_stamps[-1]))
        inside = own_stamps[0] <= lo <= own_stamps[-1] or own_stamps[0] <= hi <= own_stamps[-1]
        if inside or gap <= FOREIGN_STEM_DAYS:
            kept[s] = ss
        else:
            foreign_dropped.append("%s (%d files, %s)" % (s, len(ss), lo))
            dropped["foreign stem"] = dropped.get("foreign stem", 0) + len(ss)
    for e, n in unknown_ext.items():
        dropped["ext .%s" % e] = n

    pairs = sorted((x, s) for s, ss in kept.items() for x in ss)
    all_stamps = [p[0] for p in pairs]
    first = all_stamps[0] if all_stamps else ""
    last = all_stamps[-1] if all_stamps else ""
    first_file = paths.get(pairs[0][0] + "|" + pairs[0][1], "") if pairs else ""
    last_file = paths.get(pairs[-1][0] + "|" + pairs[-1][1], "") if pairs else ""
    corr = ""
    if first:
        _, corr = edl_raw.parse_stamp(first, years)
    notes = []
    if non_numeric:
        notes.append("beside the day folders: " + " ".join(sorted(non_numeric)))
    if foreign_dropped:
        notes.append("dropped as another deployment: " + "; ".join(foreign_dropped))
    return dict(
        site=site, raw_path=str(site_dir), instrument="", layout="edl_dayofyear",
        n_day_folders=len(day_folders), n_daily_files=0, n_data_files=n_data,
        first_stamp=first, last_stamp=last,
        stem_styles=" ".join(sorted(kept)),
        zipped=("%d day zips" % len(zips)) if zips else "",
        sidecars=" ".join("%s:%d" % (k, v) for k, v in sorted(sidecars.items())),
        n_inf=0,
        dropped="; ".join("%s:%d" % (k, v) for k, v in sorted(dropped.items())),
        date_correction=corr,
        first_file=first_file, last_file=last_file,
        notes="; ".join(notes),
    )


def _discover_lemi_site(site: str, site_dir: Path, years) -> dict:
    datadirs, other = [], []
    zips = []
    for e in os.scandir(site_dir):
        if e.is_dir():
            (datadirs if e.name.upper().startswith("DATA") else other).append(e.name)
        elif e.name.lower().endswith(".zip"):
            zips.append(e.name)
        else:
            other.append("file:" + e.name)
    stamps, n_inf, dropped = [], 0, {}
    sidecars: dict[str, int] = {}
    for d in datadirs:
        for f in os.scandir(site_dir / d):
            if not f.is_file():
                continue
            why = _drop_reason(f.name)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            up = f.name.rsplit(".", 1)[-1].upper() if "." in f.name else ""
            if up == "TXT":
                m = re.match(r"^(\d{12})", f.name)
                if m:
                    stamps.append((m.group(1), f.path))
                else:
                    dropped["no stamp"] = dropped.get("no stamp", 0) + 1
            elif up == "INF":
                n_inf += 1
                sidecars["INF"] = sidecars.get("INF", 0) + 1
            else:
                dropped["ext .%s" % up.lower()] = dropped.get("ext .%s" % up.lower(), 0) + 1
    stamps.sort()
    return dict(
        site=site, raw_path=str(site_dir), instrument="", layout="lemi_data_nnnn",
        n_day_folders=len(datadirs), n_daily_files=len(stamps), n_data_files=len(stamps),
        first_stamp=stamps[0][0] if stamps else "", last_stamp=stamps[-1][0] if stamps else "",
        first_file=stamps[0][1] if stamps else "", last_file=stamps[-1][1] if stamps else "",
        stem_styles=" ".join(sorted(datadirs)),
        zipped=("site zip: " + zips[0]) if zips else "",
        sidecars=" ".join("%s:%d" % (k, v) for k, v in sorted(sidecars.items())),
        n_inf=n_inf,
        dropped="; ".join("%s:%d" % (k, v) for k, v in sorted(dropped.items())),
        date_correction="",           # the LEMI stamps a four-digit year and is not rollover-affected
        notes=("beside the DATA folders: " + " ".join(sorted(other))) if other else "",
    )


def discover(raw_root, layout: str, years=(), sites=None, overrides=None, instrument="",
             exclude=()) -> pd.DataFrame:
    """One row per site folder under `raw_root`, read from names only.

    `overrides` maps a site to a folder outside `raw_root`. An overridden site is discovered at that folder and
    the row's notes record it. `exclude` names folders under `raw_root` that are not sites.
    """
    raw_root = Path(raw_root)
    overrides = overrides or {}
    skip = {str(x) for x in (exclude or ())}
    if layout == "lemi_zip":
        raise NotImplementedError(
            "lemi_zip (one <site>.zip per site, read from its namelist) is not implemented: no survey in this "
            "repository ships that layout. The GA state releases do; add it when one arrives.")
    if layout not in ("edl_dayofyear", "lemi_data_nnnn"):
        raise ValueError("unknown layout %r" % layout)
    names = sites if sites is not None else sorted(p.name for p in raw_root.iterdir()
                                                   if p.is_dir() and p.name not in skip)
    rows = []
    for site in names:
        d = Path(overrides[site]) if site in overrides else raw_root / site
        if not d.exists():
            rows.append(dict(site=site, raw_path=str(d), instrument="", layout=layout,
                             n_day_folders=0, n_daily_files=0, n_data_files=0, first_stamp="", last_stamp="",
                             stem_styles="", zipped="", sidecars="", n_inf=0,
                             dropped="folder absent", date_correction="",
                             first_file="", last_file="", notes="NOT FOUND"))
            continue
        row = (_discover_edl_site(site, d, years) if layout == "edl_dayofyear"
               else _discover_lemi_site(site, d, years))
        if site in overrides:
            row["notes"] = ("survey.yaml raw_overrides; "
                            + row.get("notes", "")).strip("; ")
        row["instrument"] = instrument
        rows.append(row)
    return pd.DataFrame(rows, columns=DISCOVERY_COLUMNS + ["notes"])


def discover_survey(cfg: dict, years=(), verbose=False) -> pd.DataFrame:
    """Every instrument block of a survey.yaml, discovered in turn, as one table sorted by site."""
    root = Path(cfg["raw_root"])
    over = cfg.get("raw_overrides") or {}
    out = []
    for instrument, block in (cfg.get("instruments_layout") or {}).items():
        base = root / (block.get("subdir") or "")
        if not base.exists():
            if verbose:
                print("   %s: NOT FOUND %s" % (instrument, base))
            continue
        d = discover(base, block["layout"], years=years, overrides=over, instrument=instrument,
                     exclude=block.get("exclude") or ())
        if verbose:
            print("   %-14s %-14s %3d sites under %s" % (instrument, block["layout"], len(d), base))
        out.append(d)
    return (pd.concat(out, ignore_index=True).sort_values("site").reset_index(drop=True)
            if out else pd.DataFrame(columns=DISCOVERY_COLUMNS + ["notes"]))


def audit_stems(site_dir, layout: str) -> set:
    """Every file stem under a site folder, from a second walk that shares no code with `discover`.

    Used by workbook 01's first check, which compares two independent walks rather than asking the discovery
    table about its own bookkeeping. This walk uses pathlib.rglob and a plain regex; `discover` uses os.scandir
    and `split_stem`.
    """
    site_dir = Path(site_dir)
    out = set()
    if layout == "edl_dayofyear":
        pat = re.compile(r"^(.*?)(\d{12})\.(BX|BY|BZ|EX|EY)$", re.I)
        for p in site_dir.rglob("*"):
            if not p.is_file() or p.name.startswith("._"):
                continue
            m = pat.match(p.name)
            if m and not _DROP_NAME.match(p.name):
                out.add(m.group(1))
    elif layout == "lemi_data_nnnn":
        for p in site_dir.rglob("*"):
            if p.is_dir() and p.name.upper().startswith("DATA"):
                out.add(p.name)
    return out


def audit_bench_files(site_dir, layout: str) -> list:
    """Every LAB_ or *_test* file under a site's day folders, from a walk that shares no code with `discover`.

    Used by workbook 01's first check, which compares this list against the `LAB_/test` count the discovery
    row reports. The walk is pathlib.iterdir over the numeric day folders (or the DATA folders for LEMI) and a
    plain regex; `discover` uses os.scandir and `_drop_reason`. Names are returned sorted.
    """
    site_dir = Path(site_dir)
    if not site_dir.exists():
        return []
    out = []
    for d in site_dir.iterdir():
        if not d.is_dir():
            continue
        if layout == "edl_dayofyear" and not d.name.isdigit():
            continue
        if layout == "lemi_data_nnnn" and not d.name.upper().startswith("DATA"):
            continue
        for p in d.iterdir():
            if p.is_file() and not p.name.startswith("._") and _DROP_NAME.match(p.name):
                out.append(p.name)
    return sorted(out)


def dropped_count(dropped: str, reason: str) -> int:
    """The count `discover` recorded against one drop reason in a row's `dropped` string, 0 where absent."""
    for part in str(dropped).split(";"):
        k, _, v = part.strip().rpartition(":")
        if k == reason:
            return int(v)
    return 0
