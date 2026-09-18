"""The discovery table plus the headers and sidecars: one row per site of what was recorded, where and when.

discover reads names. This module reads the files names cannot answer for:

    the first and the last data file            the span and the sample rate: a miniSEED header carries all
                                                three, an ASCII file none, and for ASCII the rate is inferred
                                                from the site's own consecutive files (raw.edl.sample_rate)
    the .gps day files (EDL), or the first and last row of each daily TXT (LEMI)
                                                 the position, its scatter and the satellite count
    the .INF block in force (LEMI)               the serial, the firmware and the dipole lengths
    the release's own MTH5 station attributes    compared, never used as a value

An EDL folder records no arm length at all. survey.yaml `dipoles` carries what a survey uses instead -- a
CSV of the deployment sheet's lengths, or one default with the reason -- and the default is written here as
`assume:<metres>` while the table is applied by workbook 01, which is where the survey folder is in hand.

Two file headers per site plus up to 40 small sidecars; no time-series body is opened. AusLAMP Victoria (100
sites, 243,157 data files) scans in about 13 s on a warm file cache.

The release MTH5 position is read for comparison only. In AusLAMP Victoria it places VIC013 45.2 km, VIC058R
23.7 km, VIC027 6.0 km and VIC072 4.9 km from the position the site's own instrument recorded.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import edl as edl_raw
from . import lemi as lemi_raw

SCAN_COLUMNS = [
    "site", "instrument", "layout", "raw_path",
    "files", "sample_rate_hz", "start_utc", "end_utc", "days", "date_correction",
    "lat", "lon", "elev_m", "position_source", "position_scatter_m", "n_fixes", "min_sat", "max_sat",
    "n_fixes_below_floor",
    "serial", "serial_source", "firmware",
    "dipole_n_m", "dipole_e_m", "dipole_source",
    "release_lat", "release_lon", "release_elev_m", "release_separation_km",
    "release_serial", "release_dipole_n_m", "release_dipole_e_m",
    "n_deployments", "deployments", "notes",
]


def _tidy_serial(v) -> str:
    """'5437.0' -> '5437', '70' -> '70'. The release writes the logger id as a float string.

    The value is parsed as a number rather than stripped of characters, which would turn '10' into '1'.
    """
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    try:
        f = float(s)
    except ValueError:
        return s
    if not np.isfinite(f):
        return ""
    return str(int(f)) if f == int(f) else s


def mth5_station(path) -> dict:
    """The station attributes of a release MTH5, read with h5py: position, logger id, dipole lengths.

    Attributes only; no dataset is opened.
    """
    import h5py
    out = dict(release_lat=np.nan, release_lon=np.nan, release_elev_m=np.nan,
               release_serial="", release_dipole_n_m=np.nan, release_dipole_e_m=np.nan,
               release_start="", release_end="")
    p = Path(path)
    if not p.exists():
        return out
    with h5py.File(p, "r") as h:
        surveys = h["Experiment/Surveys"]
        for sv in surveys:
            stations = surveys[sv]["Stations"]
            for st in stations:
                a = stations[st].attrs
                out["release_lat"] = float(a.get("location.latitude", np.nan))
                out["release_lon"] = float(a.get("location.longitude", np.nan))
                out["release_elev_m"] = float(a.get("location.elevation", np.nan))
                for run in stations[st]:
                    ra = stations[st][run].attrs
                    out["release_serial"] = str(ra.get("data_logger.id", ""))
                    out["release_start"] = str(ra.get("time_period.start", ""))
                    for ch, key in (("ex", "release_dipole_n_m"), ("ey", "release_dipole_e_m")):
                        if ch in stations[st][run]:
                            v = stations[st][run][ch].attrs.get("dipole_length", np.nan)
                            out[key] = float(v) if v is not None else np.nan
                    break
                break
            break
    return out


def _edl_span(row, years) -> dict:
    """(start, end, fs, correction) from the site's first and last data file.

    A miniSEED file carries the start, the rate and the count in its header. An ASCII file carries none of
    them, so the rate is read from the site's own files first (`edl.sample_rate`) and passed in; the start is
    then the stamp in the name and the count is the line count.
    """
    if not row["first_file"]:
        return dict(start_utc=None, end_utc=None, sample_rate_hz=np.nan, date_correction="")
    a = edl_raw.hour_header(row["first_file"], years)
    fs = None
    if not np.isfinite(a["fs"]):
        fs = edl_raw.sample_rate(row["raw_path"])
        a = edl_raw.hour_header(row["first_file"], years, fs=fs)
    b = edl_raw.hour_header(row["last_file"], years, fs=fs)
    end = (b["start_utc"] + timedelta(seconds=(b["n"] - 1) / b["fs"])
           if np.isfinite(b["fs"]) and b["fs"] else b["start_utc"])
    return dict(start_utc=a["start_utc"], end_utc=end, sample_rate_hz=a["fs"],
                date_correction=a["date_correction"])


def _lemi_span(row) -> dict:
    """(start, end, fs) from the first row of the first daily file and the last row of the last."""
    if not row["first_file"]:
        return dict(start_utc=None, end_utc=None, sample_rate_hz=np.nan, date_correction="")
    f_first, _ = lemi_raw.txt_first_last(row["first_file"])
    f2_first, f2_last = lemi_raw.txt_first_last(row["last_file"])
    start = lemi_raw.row_time(f_first)
    end = lemi_raw.row_time(f2_last) if f2_last else lemi_raw.row_time(f2_first)
    return dict(start_utc=start, end_utc=end, sample_rate_hz=1.0, date_correction="")


def scan_site(row, cfg, years, release_mth5=None, gps_max_files: int = 40, verbose=False) -> dict:
    """One discovery row plus its headers, position, serial and dipoles."""
    site = row["site"]
    inst = row["instrument"]
    layout = row["layout"]
    d = Path(row["raw_path"])
    out = {c: "" for c in SCAN_COLUMNS}
    out.update(site=site, instrument=inst, layout=layout, raw_path=str(d),
               files=int(row["n_data_files"]), notes=row.get("notes", ""))
    out.update(n_deployments=1, deployments="")

    if layout == "edl_dayofyear":
        out.update(_edl_span(row, years))
        pos = edl_raw.gps_position(d, max_files=gps_max_files)
        out.update(lat=pos["lat"], lon=pos["lon"], elev_m=pos["elev_m"],
                   position_scatter_m=pos["scatter_m"], n_fixes=pos["n_fixes"],
                   min_sat=pos["min_sat"], max_sat=pos["max_sat"],
                   n_fixes_below_floor=pos["n_below_floor"],
                   position_source="EDL .gps median over %d daily files, >= 3 satellites" % pos["n_files"])
        # an EDL folder records no arm length. survey.yaml `dipoles` says what to use where nothing does,
        # and workbook 01 writes a measured length from `dipoles.table` over this afterwards
        from ..survey import dipole_default
        value, reason = dipole_default(cfg, inst)
        if value is not None:
            out.update(dipole_n_m="assume:%g" % value, dipole_e_m="assume:%g" % value,
                       dipole_source=reason)
    elif layout == "lemi_data_nnnn":
        out.update(_lemi_span(row))
        datadirs = [p for p in d.iterdir() if p.is_dir() and p.name.upper().startswith("DATA")]
        rows_first = []
        files = []
        for dd in datadirs:
            files += lemi_raw.daily_files(dd)
        files.sort()
        for p in files:
            fr, _ = lemi_raw.txt_first_last(p)
            if fr:
                rows_first.append(fr)
        pos = lemi_raw.position_from_rows(rows_first)
        out.update(lat=pos["lat"], lon=pos["lon"], elev_m=pos["elev_m"],
                   position_scatter_m=pos["scatter_m"], n_fixes=pos["n_rows"],
                   min_sat=pos["n_sat"], max_sat=pos["n_sat"], n_fixes_below_floor=0,
                   position_source="LEMI record GPS, median over the first row of %d daily files" % len(files))
        # a folder holding more than one deployment is reported; the choice is survey.yaml raw_overrides
        deps = lemi_raw.deployments_in_folder(datadirs[0]) if datadirs else []
        out["n_deployments"] = len(deps)
        if len(deps) > 1:
            out["deployments"] = "; ".join("%s..%s n=%d at %.4f,%.4f"
                                           % (c["first"], c["last"], c["n_files"], c["lat"], c["lon"])
                                           for c in deps)
        first_key = Path(row["first_file"]).name[:12] if row["first_file"] else ""
        gov, allb = lemi_raw.site_inf(datadirs[0], first_key) if datadirs else (None, [])
        if gov and not gov["inf_missing"]:
            out.update(serial=gov["serial"], serial_source=".INF %%LEMI424 (%s)" % Path(gov["inf_path"]).name,
                       firmware=gov["firmware"],
                       dipole_n_m=gov["L1"], dipole_e_m=gov["L2"],
                       dipole_source=".INF %%L1/%%L2 (%s)" % Path(gov["inf_path"]).name)
    else:
        raise ValueError("unknown layout %r" % layout)

    if release_mth5:
        rel = mth5_station(Path(release_mth5) / ("%s.h5" % site))
        out.update(rel)
        if np.isfinite(rel["release_lat"]) and np.isfinite(out["lat"] if out["lat"] != "" else np.nan):
            from ..geo import distance_km
            out["release_separation_km"] = distance_km((out["lat"], out["lon"]),
                                                       (rel["release_lat"], rel["release_lon"]))
        # the release supplies the serial and the dipoles only where the site's own files do not
        if not out["serial"] and rel["release_serial"]:
            out.update(serial=_tidy_serial(rel["release_serial"]),
                       serial_source="release MTH5 data_logger.id")
        if out["dipole_n_m"] in ("", None) and np.isfinite(rel["release_dipole_n_m"]):
            out.update(dipole_n_m=rel["release_dipole_n_m"], dipole_e_m=rel["release_dipole_e_m"],
                       dipole_source="release MTH5 dipole_length")

    if out["start_utc"] is not None and out["end_utc"] is not None and out["start_utc"] != "":
        out["days"] = round((out["end_utc"] - out["start_utc"]).total_seconds() / 86400.0, 3)
    if verbose:
        print("   %-9s %s .. %s  %s" % (site, out["start_utc"], out["end_utc"], out["position_source"]))
    return out


def scan(discovery: pd.DataFrame, cfg: dict, years, release_mth5_by_instrument=None,
         verbose=False) -> pd.DataFrame:
    """scan_site over a discovery table. `release_mth5_by_instrument` maps instrument to the MTH5 folder."""
    rel = release_mth5_by_instrument or {}
    rows = []
    for _, r in discovery.iterrows():
        rows.append(scan_site(r, cfg, years, release_mth5=rel.get(r["instrument"]), verbose=verbose))
    return pd.DataFrame(rows, columns=SCAN_COLUMNS)
