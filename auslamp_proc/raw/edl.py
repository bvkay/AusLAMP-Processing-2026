"""Earth Data Logger PR6-24: file stamps, the GPS week rollover, hourly headers and the .gps position.

Tree: <site>/<DDD>/<stem><YYMMDDhhmmss>.<BX BY BZ EX EY TP>, hourly, with .gps .gst .pll once a day and, at
some sites, hourly .ambientTemperature .batteryVolts2 .diskTemperature.

The PR6-24 writes either miniSEED or ASCII (one integer count per line) to the same .BX .BY .BZ .EX .EY
names, so the extension does not say which. AusLAMP Victoria is miniSEED and all three AusLAMP Queensland
phases are ASCII. An ASCII file carries no header: the start time is the stamp in its name, the sample count
is its line count, and the sample rate follows from consecutive file stamps and counts. hour_header reads
whichever format the file is in (`mt_io.uoa.pr624.is_miniseed`, `count_samples`), and sample_rate infers the
rate from the site's own files (`mt_io.uoa.pr624.infer_sample_rate`, which snaps to the rates the recorder can
be set to, EDM 021 1.3, within 2 per cent). Added 2026-09-16: reading only the miniSEED header stopped the
workbook on every Queensland site.

Four rules this module applies.

1. The site id is the folder name, not the file stem. In AusLAMP Victoria the stems run <site>_ at 51 of the
   81 EDL sites, <site> at 15 and a two-digit form (VIC73 for VIC073) at 15, and five sites carry hours
   written under a previous deployment's name. split_stem takes the last 12 digits of the stem as the time
   stamp and returns the prefix separately.
2. Extensions are matched case-insensitively. VIC055 writes 105 hours of By as .by, with no upper-case .BY
   for those hours.
3. GPS week rollover: a stamp whose year precedes the survey's `years` is 1024 weeks (7168 days) early and is
   corrected by parse_stamp. The day folder, the .gps >RTM sentence and the .gst and .pll logs all carry the
   same early date, so the survey years are an input. mt-io 0.0.5 does not apply the correction.
   Ported from D:/BEN/MT_Processing_2026/mt_proc/mt_proc/io/edl.py:46-60, with the 2015 constant replaced by
   the survey's `years`.
4. The position comes from the logger's own .gps, not from release metadata. In AusLAMP Victoria the release
   MTH5 places VIC013 45.2 km from the position its logger recorded.

The .gps format is Earth Data's >R.. sentences, not NMEA. Field widths, inferred from the files:

    >RAL86285+00148+00002;*4E<              5-char stamp, signed 5-digit altitude in m, vertical velocity
    >RPV86285-2004638+1420381000000002;*78< 5-char stamp, lat 1e-5 deg (8), lon 1e-5 deg (9), velocity (6),
                                            satellites (2)

Ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/edl_gps.py (read_gps_file, site_position,
scatter_m). The widths are inferred, not specified, so gps_position is checked against an independent position
on each new fleet.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DATA_EXT = ("BX", "BY", "BZ", "EX", "EY")
SIDECAR_EXT = ("gps", "gst", "pll", "TP", "ambientTemperature", "batteryVolts2", "diskTemperature")

STAMP = re.compile(r"(\d{12})$")
RPV = re.compile(r">RPV(\d{5})([+-]\d{7})([+-]\d{8})(\d{6})(\d{2});")
RAL = re.compile(r">RAL(\d{5})([+-]\d{5})([+-]\d{5});")

ROLLOVER = timedelta(weeks=1024)          # 7168 days
ROLLOVER_LABEL = "+1024 weeks"


def split_stem(name: str):
    """'VIC73140101000000.BX' -> ('VIC73', '140101000000', 'BX'). (None, None, ext) if there is no stamp."""
    if "." not in name:
        return None, None, ""
    stem, ext = name.rsplit(".", 1)
    m = STAMP.search(stem)
    if not m:
        return None, None, ext
    return stem[:m.start()], m.group(1), ext


def parse_stamp(stamp: str, years) -> tuple[datetime, str]:
    """'YYMMDDhhmmss' -> (UTC datetime, correction label).

    The correction is +1024 weeks when the parsed year is earlier than every year in `years`, and '' otherwise.
    A survey straddling a rollover boundary lists both sides in its `years`.
    """
    s = str(stamp)
    dt = datetime(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]),
                  int(s[6:8]), int(s[8:10]), int(s[10:12]), tzinfo=timezone.utc)
    if years and dt.year < min(int(y) for y in years):
        return dt + ROLLOVER, ROLLOVER_LABEL
    return dt, ""


def hour_header(path, years=(), fs=None) -> dict:
    """(start_utc, fs, n) of one hourly data file, whichever format the recorder wrote it in.

    miniSEED: obspy reads the header with headonly=True, about 20 ms for a 49 KB file, and no sample is read.
    The rollover is applied to the header start as well as to the file name; the logger writes the same date
    into both.

    ASCII: there is no header. The start is the 12-digit stamp in the file name, the sample count is the line
    count (`mt_io.uoa.pr624.count_samples`), and the rate is `fs`, which the caller reads from the site's own
    files with sample_rate; with fs None it is returned as NaN.
    """
    from mt_io.uoa import pr624
    path = Path(path)
    if pr624.is_miniseed(path):
        from obspy import read
        tr = read(str(path), headonly=True)[0]
        t0 = tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)
        corr = ""
        if years and t0.year < min(int(y) for y in years):
            t0, corr = t0 + ROLLOVER, ROLLOVER_LABEL
        return dict(start_utc=t0, fs=float(tr.stats.sampling_rate), n=int(tr.stats.npts),
                    date_correction=corr, format="miniSEED", path=str(path))
    _, stamp, _ = split_stem(path.name)
    if stamp is None:
        raise ValueError("%s is ASCII and its name carries no stamp, so it cannot be placed" % path.name)
    t0, corr = parse_stamp(stamp, years)
    return dict(start_utc=t0, fs=float(fs) if fs else np.nan, n=int(pr624.count_samples(path)),
                date_correction=corr, format="ASCII", path=str(path))


def sample_rate(site_dir, channel: str = "BX", n_days: int = 3, max_pairs: int = 25) -> float:
    """The rate a site's own ASCII files were written at, from consecutive file stamps and sample counts.

    A file that runs on into the next holds start(next) - start(this) seconds, so the rate follows from its
    sample count; the modal answer over up to max_pairs pairs is taken and snapped to a rate the recorder can
    be set to (`mt_io.uoa.pr624.infer_sample_rate`, EDM 021 1.3, 2 per cent tolerance). Files are taken from
    n_days = 3 day folders spread evenly over the record, because a setup fragment at the start can overlap
    the hour after it. Returns NaN where no rate is agreed.
    """
    from mt_io.uoa import pr624
    site_dir = Path(site_dir)
    days = sorted(p for p in site_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not days:
        return float("nan")
    step = max(1, len(days) // max(1, n_days))
    files = []
    for d in days[::step][:n_days]:
        files += [p for p in d.iterdir()
                  if p.is_file() and p.suffix.upper() == "." + channel.upper()
                  and not p.name.startswith("._")]
    rate = pr624.infer_sample_rate(files, max_pairs=max_pairs) if len(files) > 1 else None
    return float(rate) if rate else float("nan")


def read_gps_file(path) -> list[dict]:
    """Every fix in one `.gps` file: latitude, longitude, elevation (m), n_sat.

    A fix outside -90..90 or -180..180 is dropped, which covers a truncated sentence at the end of a file.
    Ported from edl_gps.read_gps_file.
    """
    try:
        txt = Path(path).read_text(errors="ignore")
    except OSError:
        return []
    alts = [int(m.group(2)) for m in RAL.finditer(txt)]
    out = []
    for i, m in enumerate(RPV.finditer(txt)):
        lat = int(m.group(2)) / 1e5
        lon = int(m.group(3)) / 1e5
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        out.append({"latitude": lat, "longitude": lon,
                    "elevation": float(alts[i]) if i < len(alts) else np.nan,
                    "n_sat": int(m.group(5))})
    return out


def gps_position(site_dir, max_files: int = 40, min_sat: int = 3) -> dict:
    """The site's position from its own `.gps` files: the median over up to `max_files` daily files.

    The value is the median of the kept fixes, not the first fix: a receiver can still be converging when the
    logger writes its first sentences. Fixes with fewer than `min_sat` = 3 satellites are dropped and counted
    in n_below_floor. scatter_m is the 95th percentile spread of the kept fixes about their own median and is
    the quality flag; the floor applied by workbook 01 is 100 m.
    Ported from edl_gps.site_position and edl_gps.scatter_m.
    """
    files = sorted(Path(site_dir).rglob("*.gps"))
    out = dict(lat=np.nan, lon=np.nan, elev_m=np.nan, n_fixes=0, n_files=len(files),
               scatter_m=np.nan, min_sat=0, max_sat=0, n_below_floor=0)
    if not files:
        return out
    step = max(1, len(files) // max_files)
    used = files[::step]
    allfix = [f for p in used for f in read_gps_file(p)]
    fixes = [f for f in allfix if f["n_sat"] >= min_sat]
    out["n_below_floor"] = len(allfix) - len(fixes)
    if not fixes:
        out["n_files"] = len(used)
        return out
    # min_sat and max_sat are over the kept fixes only; a fix below the floor is counted in n_below_floor
    out["min_sat"] = int(min(f["n_sat"] for f in fixes))
    out["max_sat"] = int(max(f["n_sat"] for f in fixes))
    la = [f["latitude"] for f in fixes]
    lo = [f["longitude"] for f in fixes]
    el = [f["elevation"] for f in fixes if np.isfinite(f["elevation"])]
    dy = (np.asarray(la) - np.median(la)) * 111320.0
    dx = (np.asarray(lo) - np.median(lo)) * 111320.0 * np.cos(np.radians(np.median(la)))
    out.update(lat=float(np.median(la)), lon=float(np.median(lo)),
               elev_m=float(np.median(el)) if el else np.nan,
               n_fixes=len(fixes), n_files=len(used),
               scatter_m=float(np.percentile(np.hypot(dx, dy), 95)) if len(fixes) > 1 else 0.0)
    return out
