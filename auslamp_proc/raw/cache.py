"""The npz cache: every raw file placed on one absolute time axis by its own header start, in physical units.

The cache is the record as laid. No sign, no rotation and no notch are applied here: the frame and the signs
come from decisions.csv at processing time, and a cache carrying them could not be re-read under a different
decision.

Placement (ported from D:/BEN/MTH5_Aurora_mt-io_2026/scripts/processing/vic_cache.py:61-85, read_channel and
place). The axis runs from the earliest file start to the latest file end at the instrument's rate. Each file
is written at index round((start - t0) * fs) from its own start time; files are never concatenated, so a
missing hour leaves NaN and does not shift the hours after it. A file whose samples land on indices already
holding a finite value counts those samples as overlap and the later file wins.

Reconciliation (ported from vic_cache_lemi.py:83, `overlap == 0 and placed == read`; vic_cache.py:169 states
it as a disjunction whose right-hand limb `overlap_samples >= 0` is always true, so that form cannot fail).
The criterion this module applies, per channel:

    samples_placed == samples_read, overlap_samples == 0, files_read == files_found, files_unplaced == 0

Every limb can fail: an hour written twice raises overlap_samples, an unreadable file lowers files_read, a
file whose stamp falls outside the axis raises files_unplaced, and any of the three breaks the equality.

Units. PR6-24 with a Bartington Mag-03, constants from mt_io.uoa.pr624: Hx, Hy nT = uV / 142.857
(BARTINGTON_UV_PER_NT); Hz nT = uV / (142.857 * 0.4) (BZ_DIVIDER_RATIO, the 15k/10k divider); Ex, Ey mV/km =
uV / (L_m * 10) (E_TERMINAL_BOX_GAIN), with L the dipole of that arm from sites.csv. A dipole cell reading
`assume:<value>` is used and the assumption is written into the sidecar. LEMI-424: the reader returns bx, by,
bz in nT and e1, e2 in mV/km already divided by the %L lengths; they are never divided by L again.

Decimation to 1 Hz (ported from vic_cache.py:87-99, decimate_nan). Gaps are bridged by linear interpolation
so the FIR has finite input, the zero-phase FIR decimation runs (scipy.signal.decimate, ftype 'fir',
zero_phase True), and an output sample is set back to NaN where any of its q input samples was not finite.
The blanking is per decimation block of q samples; the anti-alias filter's support is wider than one block,
so an output sample next to a gap carries interpolated input within the filter's half-width.

LEMI gap handling. The mt-io LEMI reader stamps the samples after a gap early by the length of the gap, so
each daily file is placed at its own first-row time and its row count is checked against 86400; a short file
is placed at its first-row time and the rest of that day stays NaN.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import edl as edl_raw
from .discover import FOREIGN_STEM_DAYS, _drop_reason, _stamp_days_apart

BUILDER_VERSION = "cache 1.0 (2026-09-16)"

CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")
EDL_EXT = {"Hx": "BX", "Hy": "BY", "Hz": "BZ", "Ex": "EX", "Ey": "EY"}
LEMI_VAR = {"Hx": "bx", "Hy": "by", "Hz": "bz", "Ex": "e1", "Ey": "e2"}
LEMI_ROWS_PER_DAY = 86400

UNIT = {"Hx": "nT", "Hy": "nT", "Hz": "nT", "Ex": "mV/km", "Ey": "mV/km"}


# ---------------------------------------------------------------- cell values

def dipole_value(cell) -> tuple[float, str]:
    """A sites.csv dipole cell to (metres, source word). `assume:12` returns (12.0, 'assume')."""
    s = str(cell).strip()
    if s.lower().startswith("assume:"):
        return float(s.split(":", 1)[1]), "assume"
    if s in ("", "nan", "decide"):
        return float("nan"), "missing"
    return float(s), "value"


# ---------------------------------------------------------------- file lists

def channel_files(site_dir, ext: str, site: str) -> tuple[list[Path], dict]:
    """Every hourly file of one EDL channel under a site tree, the drop rules of discover.py applied.

    AppleDouble copies, LAB_ and *_test* bench files and operator scripts are dropped (discover._drop_reason),
    a name with no 12-digit stamp is dropped, and a stem that is not the site's own whose stamps lie more than
    FOREIGN_STEM_DAYS = 30 days outside the span of the own-stem files is dropped as another deployment left in
    the folder (discover.py:120-135). Returns (files in stamp order, {reason: count}).
    """
    site_dir = Path(site_dir)
    dropped: dict[str, int] = {}
    found: list[tuple[str, str, Path]] = []            # (stamp, stem, path)
    for root, _dirs, names in os.walk(site_dir):
        for name in names:
            if not name.upper().endswith("." + ext.upper()):
                continue
            why = _drop_reason(name)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            stem, stamp, _ = edl_raw.split_stem(name)
            if stamp is None:
                dropped["no stamp"] = dropped.get("no stamp", 0) + 1
                continue
            found.append((stamp, stem, Path(root) / name))
    own_stamps = sorted(s for s, stem, _ in found if stem.rstrip("_").upper() == site.upper())
    keep = []
    for stamp, stem, path in sorted(found):
        if not own_stamps or stem.rstrip("_").upper() == site.upper():
            keep.append(path)
            continue
        gap = min(_stamp_days_apart(stamp, own_stamps[0]), _stamp_days_apart(stamp, own_stamps[-1]))
        if own_stamps[0] <= stamp <= own_stamps[-1] or gap <= FOREIGN_STEM_DAYS:
            keep.append(path)
        else:
            dropped["foreign stem"] = dropped.get("foreign stem", 0) + 1
    return keep, dropped


# ---------------------------------------------------------------- reading and placing

def read_edl_file(path, fs, years) -> tuple[float, np.ndarray]:
    """(start in seconds since the epoch, samples in uV) of one hourly PR6-24 file.

    miniSEED carries its own start and rate in the header (mt_io.uoa.pr624.read_edl_miniseed). ASCII carries
    neither: the start is the 12-digit stamp in the file name and the rate is `fs`, read from the site's own
    files by raw.edl.sample_rate and checked against survey.yaml in workbook 01. The GPS week rollover is
    applied to both forms (raw.edl.parse_stamp).
    """
    from mt_io.uoa import pr624
    path = Path(path)
    if pr624.is_miniseed(path):
        data, start, rate = pr624.read_edl_miniseed(path)
        if years and start.year < min(int(y) for y in years):
            start = start + edl_raw.ROLLOVER
        if abs(float(rate) - float(fs)) > 1e-6:
            raise ValueError("%s is at %g Hz, not %g Hz" % (path.name, rate, fs))
        return float(start.timestamp()), np.asarray(data, np.float32)
    _, stamp, _ = edl_raw.split_stem(path.name)
    t0, _ = edl_raw.parse_stamp(stamp, years)
    # the ASCII form is one recorder-scaled microVolt value per line, no header (mt_io.uoa.pr624.UoADataReader)
    data = np.loadtxt(path, dtype=np.float32, comments=None)
    return float(t0.timestamp()), np.atleast_1d(data).ravel()


def place(segs, t0: float, n: int, fs: float):
    """Write each (start, samples) segment at its own index on one axis. Ported from vic_cache.place (:74).

    Returns (array, placed, overlap, read, unplaced). `overlap` counts samples written over a finite value,
    which is an hour the recorder wrote twice; the later file wins. `unplaced` counts the files whose samples
    fall outside the axis, which cannot happen on an axis built from the same segments and is reported so that
    a caller passing a fixed axis learns of it.
    """
    x = np.full(n, np.nan, np.float32)
    placed = overlap = read = unplaced = 0
    for ts, data in segs:
        i0 = int(round((ts - t0) * fs))
        i1 = i0 + len(data)
        read += len(data)
        if i0 < 0 or i1 > n:
            unplaced += 1
            continue
        overlap += int(np.isfinite(x[i0:i1]).sum())
        x[i0:i1] = data
        placed += len(data)
    return x, placed, overlap, read, unplaced


def decimate_nan(x, q: int = 10):
    """Zero-phase FIR decimation by q that blanks every output whose q input samples hold a NaN.

    Ported from vic_cache.decimate_nan (:87). Gaps are bridged by linear interpolation so the filter has
    finite input and the outputs are set back to NaN afterwards. The blanking is per block of q inputs; the
    FIR support is wider, so an output next to a gap carries interpolated input within the filter's half-width.
    """
    from scipy import signal
    q = int(q)
    n = len(x) // q * q
    x = np.asarray(x[:n], np.float64)
    bad = ~np.isfinite(x)
    if bad.all():
        return np.full(n // q, np.nan)
    if bad.any():
        idx = np.arange(n)
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    y = signal.decimate(x, q, ftype="fir", zero_phase=True)
    y[bad.reshape(-1, q).any(axis=1)] = np.nan
    return y


def gap_stats(x) -> dict:
    """(nan_fraction, n_gaps, longest_gap_samples) of one channel."""
    fin = np.isfinite(np.asarray(x))
    edges = np.diff(np.concatenate(([0], (~fin).astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return dict(nan_fraction=float(1 - fin.mean()) if len(fin) else 1.0,
                n_gaps=len(runs),
                longest_gap_s=float(max((b - a for a, b in runs), default=0)))


# ---------------------------------------------------------------- the build

def _write_npz(path, t0, fs, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             t0=np.array([int(round(t0))], np.int64), fs=np.array([float(fs)]),
             layout=np.array([meta["layout"]]), chan_x=np.array([meta["chan_x"]]),
             chan_y=np.array([meta["chan_y"]]),
             dipole_n_m=np.array([meta["dipole_n_m"]]), dipole_e_m=np.array([meta["dipole_e_m"]]),
             e_gain=np.array([meta["e_gain"]]), h_uv_per_nt=np.array([meta["h_uv_per_nt"]]),
             bz_divider=np.array([meta["bz_divider"]]),
             **{c: np.asarray(arrays[c], np.float32) for c in CHANNELS})


def _verdicts(stats: dict) -> list[str]:
    """One line per channel stating the reconciliation criterion and its numbers, then the survey-wide line."""
    out = []
    bad = 0
    for ch in CHANNELS:
        s = stats[ch]
        ok = (s["samples_placed"] == s["samples_read"] and s["overlap_samples"] == 0
              and s["files_read"] == s["files_found"] and s["files_unplaced"] == 0)
        bad += not ok
        out.append("%s: %s -- placed %d of %d read, overlap %d, %d of %d files read, %d unplaced"
                   % (ch, "PASS" if ok else "FAIL", s["samples_placed"], s["samples_read"],
                      s["overlap_samples"], s["files_read"], s["files_found"], s["files_unplaced"]))
    out.append("reconciliation: %s -- %d of %d channels place every sample they read with no overlap"
               % ("PASS" if bad == 0 else "FAIL", len(CHANNELS) - bad, len(CHANNELS)))
    return out


def build(site_row, survey_cfg, rates=(1, 10), force=False) -> dict:
    """Build a site's caches and its sidecar. Returns the sidecar dict.

    `site_row` is one row of sites.csv (site, raw_path, instrument, sample_rate_hz, dipole_n_m, dipole_e_m,
    dipole_source); `survey_cfg` is survey.yaml. `rates` names the caches to write: 10 writes
    <work_root>/cache_10hz/<site>.npz for an instrument recording above 1 Hz, and 1 writes
    <work_root>/cache_1hz/<site>.npz with <site>.json beside it. With `force` False a cache already on disk is
    left alone and its sidecar is returned.
    """
    site = str(site_row["site"])
    work = Path(survey_cfg["work_root"])
    layout = str(site_row.get("layout", "")) or "edl_dayofyear"
    out1 = work / "cache_1hz" / ("%s.npz" % site)
    out10 = work / "cache_10hz" / ("%s.npz" % site)
    side = work / "cache_1hz" / ("%s.json" % site)
    want10 = 10 in tuple(rates) and float(site_row.get("sample_rate_hz", 1) or 1) > 1.0
    if not force and side.exists() and out1.exists() and (out10.exists() or not want10):
        return json.loads(side.read_text(encoding="utf-8"))
    if layout.startswith("lemi"):
        return _build_lemi(site_row, survey_cfg, out1, side)
    return _build_edl(site_row, survey_cfg, rates, out1, out10, side, want10)


def _build_edl(site_row, cfg, rates, out1, out10, side, want10) -> dict:
    from mt_io.uoa import pr624
    site = str(site_row["site"])
    site_dir = Path(site_row["raw_path"])
    fs = float(site_row["sample_rate_hz"])
    years = [int(y) for y in (cfg.get("years") or [])]
    const = (cfg.get("instruments") or {}).get(str(site_row["instrument"]), {})
    e_gain = float(const.get("e_gain", pr624.E_TERMINAL_BOX_GAIN))
    h_uv_per_nt = float(const.get("h_uv_per_nt", pr624.BARTINGTON_UV_PER_NT))
    bz_divider = float(const.get("bz_divider", pr624.BZ_DIVIDER_RATIO))
    dn, dn_kind = dipole_value(site_row.get("dipole_n_m", ""))
    de, de_kind = dipole_value(site_row.get("dipole_e_m", ""))
    gains = {"Hx": h_uv_per_nt, "Hy": h_uv_per_nt, "Hz": h_uv_per_nt * bz_divider,
             "Ex": dn * e_gain, "Ey": de * e_gain}

    log: list[str] = []
    segs: dict[str, list] = {}
    stats: dict[str, dict] = {}
    for ch in CHANNELS:
        files, dropped = channel_files(site_dir, EDL_EXT[ch], site)
        got = []
        for f in files:
            try:
                got.append(read_edl_file(f, fs, years))
            except Exception as exc:                       # an unreadable hour is named and counted, not raised
                log.append("unreadable %s: %s" % (f.name, str(exc)[:100]))
        segs[ch] = got
        stats[ch] = dict(files_found=len(files), files_read=len(got),
                         dropped="; ".join("%s:%d" % kv for kv in sorted(dropped.items())))
    allseg = [(t, d) for v in segs.values() for (t, d) in v]
    if not allseg:
        raise RuntimeError("%s: no readable data file under %s" % (site, site_dir))
    t0 = min(t for t, _ in allseg)
    t1 = max(t + len(d) / fs for t, d in allseg)
    n = int(round((t1 - t0) * fs))

    arrays = {}
    for ch in CHANNELS:
        x, placed, overlap, read, unplaced = place(segs[ch], t0, n, fs)
        segs[ch] = None                                    # the segments of one 10 Hz channel are 200 MB
        arrays[ch] = (x / gains[ch]).astype(np.float32) if np.isfinite(gains[ch]) and gains[ch] else x
        stats[ch].update(samples_read=int(read), samples_placed=int(placed), overlap_samples=int(overlap),
                         files_unplaced=int(unplaced), **gap_stats(arrays[ch]))
    del segs

    meta = dict(layout="edl_L", chan_x="EX", chan_y="EY", dipole_n_m=dn, dipole_e_m=de,
                e_gain=e_gain, h_uv_per_nt=h_uv_per_nt, bz_divider=bz_divider)
    if want10:
        _write_npz(out10, t0, fs, arrays, meta)
    q = int(round(fs / 1.0))
    one = {}
    for ch in CHANNELS:
        one[ch] = decimate_nan(arrays[ch], q).astype(np.float32) if q > 1 else arrays[ch]
        arrays[ch] = None
    del arrays
    if 1 in tuple(rates):
        _write_npz(out1, t0, 1.0, one, meta)

    prov = _sidecar(site, site_dir, t0, n, fs, len(one["Hx"]), stats, one, meta, gains,
                    dipole_source=str(site_row.get("dipole_source", "")),
                    dipole_kind={"Ex": dn_kind, "Ey": de_kind},
                    const_source=str(const.get("source", "")), log=log,
                    files_written=[str(p) for p in ([out10] if want10 else []) + ([out1] if 1 in tuple(rates) else [])])
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return prov


def _build_lemi(site_row, cfg, out1, side) -> dict:
    """The LEMI-424 daily TXT files on one 1 Hz axis. Ported from vic_cache_lemi.build (:39-90).

    A file the reader returns at a rate other than 1 Hz is refused and named in the log; it counts as read and
    not placed, so the reconciliation fails on it. Each file is placed at its own first-row time and its row
    count is compared with 86400.
    """
    from mt_io.lemi import read_lemi424
    import pandas as pd
    site = str(site_row["site"])
    site_dir = Path(site_row["raw_path"])
    files = sorted({f.resolve() for f in site_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith("._")},
                   key=lambda f: f.name)
    log, segs, short = [], [], []
    for f in files:
        try:
            r = read_lemi424(f)
            rate = float(r.sample_rate)
            data = {ch: np.asarray(r.dataset[LEMI_VAR[ch]].values, np.float32) for ch in CHANNELS}
            if abs(rate - 1.0) > 1e-6:
                log.append("%s at %g Hz refused" % (f.name, rate))
                segs.append((np.nan, data, rate, f.name))
                continue
            if len(data["Hx"]) != LEMI_ROWS_PER_DAY:
                short.append("%s %d rows" % (f.name, len(data["Hx"])))
            segs.append((float(pd.Timestamp(str(r.start)).timestamp()), data, rate, f.name))
        except Exception as exc:
            log.append("unreadable %s: %s" % (f.name, str(exc)[:100]))
    good = [s for s in segs if np.isfinite(s[0]) and abs(s[2] - 1.0) <= 1e-6]
    if not good:
        raise RuntimeError("%s: no readable 1 Hz daily file under %s" % (site, site_dir))
    t0 = min(s[0] for s in good)
    t1 = max(s[0] + len(s[1]["Hx"]) for s in good)
    n = int(round(t1 - t0))
    arrays = {ch: np.full(n, np.nan, np.float32) for ch in CHANNELS}
    placed = overlap = read = unplaced = 0
    for ts, data, rate, name in segs:
        read += len(data["Hx"])
        if not np.isfinite(ts) or abs(rate - 1.0) > 1e-6:
            unplaced += 1
            continue
        i0 = int(round(ts - t0))
        i1 = i0 + len(data["Hx"])
        if i0 < 0 or i1 > n:
            log.append("%s outside the axis" % name)
            unplaced += 1
            continue
        overlap += int(np.isfinite(arrays["Hx"][i0:i1]).sum())
        for ch in CHANNELS:
            arrays[ch][i0:i1] = data[ch]
        placed += len(data["Hx"])
    if short:
        log.append("short files (placed at their first-row time, the rest of the day NaN): " + "; ".join(short[:20]))
    stats = {ch: dict(files_found=len(files), files_read=len(segs), samples_read=int(read),
                      samples_placed=int(placed), overlap_samples=int(overlap), files_unplaced=int(unplaced),
                      dropped="", **gap_stats(arrays[ch])) for ch in CHANNELS}
    dn, dn_kind = dipole_value(site_row.get("dipole_n_m", ""))
    de, de_kind = dipole_value(site_row.get("dipole_e_m", ""))
    meta = dict(layout="lemi_cross", chan_x="E1", chan_y="E2", dipole_n_m=dn, dipole_e_m=de,
                e_gain=1.0, h_uv_per_nt=np.nan, bz_divider=np.nan)
    _write_npz(out1, t0, 1.0, arrays, meta)
    prov = _sidecar(site, site_dir, t0, n, 1.0, n, stats, arrays, meta,
                    gains={ch: 1.0 for ch in CHANNELS},
                    dipole_source=str(site_row.get("dipole_source", "")),
                    dipole_kind={"Ex": dn_kind, "Ey": de_kind},
                    const_source="mt_io.lemi.read_lemi424 returns nT and mV/km; E is never divided by L again",
                    log=log, files_written=[str(out1)])
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return prov


def _sidecar(site, site_dir, t0, n_raw, fs, n_one, stats, one, meta, gains, dipole_source,
             dipole_kind, const_source, log, files_written) -> dict:
    dc = {ch: (float(np.nanmedian(one[ch])) if np.isfinite(one[ch]).any() else float("nan"))
          for ch in CHANNELS}
    return dict(
        site=site, source=str(site_dir), builder=BUILDER_VERSION,
        built_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        t0=int(round(t0)), t0_iso=datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        fs_raw=float(fs), n_raw=int(n_raw), n_1hz=int(n_one),
        span_days=round(n_raw / fs / 86400.0, 4),
        layout=meta["layout"], chan_x=meta["chan_x"], chan_y=meta["chan_y"],
        dipole_n_m=meta["dipole_n_m"], dipole_e_m=meta["dipole_e_m"],
        dipole_source=dipole_source,
        dipole_assumed=[k for k, v in sorted(dipole_kind.items()) if v == "assume"],
        e_gain=meta["e_gain"], h_uv_per_nt=meta["h_uv_per_nt"], bz_divider=meta["bz_divider"],
        gains_uv_per_unit={k: (float(v) if np.isfinite(v) else None) for k, v in gains.items()},
        constants_source=const_source,
        units={ch: UNIT[ch] for ch in CHANNELS},
        signs_applied="none", frame_applied="none", notch_applied="none",
        channels={ch: {k: (float(v) if isinstance(v, float) else v) for k, v in stats[ch].items()}
                  for ch in CHANNELS},
        dc_1hz={k: (None if not np.isfinite(v) else round(v, 3)) for k, v in dc.items()},
        nan_fraction_1hz={ch: round(float(1 - np.isfinite(one[ch]).mean()), 6) for ch in CHANNELS},
        log=log[:80],
        files_written=files_written,
        verdict_lines=_verdicts(stats),
    )


# ---------------------------------------------------------------- reading back

def load(site, work_root, rate=1):
    """(t0 unix seconds, {channel: float array}, meta) of one cached record.

    `rate` is 1 or 10. meta carries the sidecar where one exists beside the 1 Hz cache, plus the npz scalars.
    """
    work = Path(work_root)
    path = work / ("cache_%dhz" % int(rate)) / ("%s.npz" % site)
    z = np.load(path, allow_pickle=False)
    t0 = int(z["t0"][0])
    arrays = {c: np.asarray(z[c], np.float64) for c in CHANNELS}
    meta = dict(fs=float(z["fs"][0]), layout=str(z["layout"][0]), chan_x=str(z["chan_x"][0]),
                chan_y=str(z["chan_y"][0]), dipole_n_m=float(z["dipole_n_m"][0]),
                dipole_e_m=float(z["dipole_e_m"][0]), e_gain=float(z["e_gain"][0]),
                h_uv_per_nt=float(z["h_uv_per_nt"][0]), bz_divider=float(z["bz_divider"][0]),
                path=str(path))
    z.close()
    side = work / "cache_1hz" / ("%s.json" % site)
    if side.exists():
        meta["sidecar"] = json.loads(side.read_text(encoding="utf-8"))
    return t0, arrays, meta


def sidecar(site, work_root) -> dict:
    """The sidecar of one site, or an empty dict where none has been written."""
    p = Path(work_root) / "cache_1hz" / ("%s.json" % site)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def reconciliation_row(prov: dict) -> list[dict]:
    """One row per channel of a sidecar's placement numbers, for the workbook's table."""
    out = []
    for ch in CHANNELS:
        s = prov["channels"][ch]
        out.append(dict(site=prov["site"], channel=ch,
                        files_found=int(s["files_found"]), files_read=int(s["files_read"]),
                        samples_read=int(s["samples_read"]), samples_placed=int(s["samples_placed"]),
                        overlap_samples=int(s["overlap_samples"]),
                        files_unplaced=int(s["files_unplaced"]),
                        nan_fraction=round(float(s["nan_fraction"]), 4)))
    return out
