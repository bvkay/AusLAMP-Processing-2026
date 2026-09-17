"""The MTH5 Aurora reads: the local station and, where there is one, the reference as a second station.

Ported from qld_campaign.write_h5 (:1436-1480) and qld_one_second.station (:45-71).

One run per kept stretch. Cutting the record and concatenating puts a step at every join, and a step common
to E, H and the reference is coherent between them, so a robust regression fits it rather than down-weighting
it, so cutting the record is the disease. Instead each kept stretch of at least
transients.MIN_SEGMENT_S = 3600 s becomes its own run: local and reference are segmented identically, so
KernelDataset's run-interval intersection pairs them one to one. A stretch under an hour yields no complete
window at the deep decimation levels and is dropped and counted (transients.floor_dropped_frac).

The reference is a second station carrying hx and hy only, on the target's own grid and with the same run
ids, named by its kind: REMOTE_<site>, STACK, OBS_<code> or STACK_OBS.

No decision is applied here. The local channels arrive already decided and rotated: process.run.load_local
reads them through raw.cache.load_decided, which calls process.frame.apply_decisions, the one place that
applies a decisions.csv decision. This module writes what it is handed, and read_back compares the file
with those same arrays.

Units are `nanoTesla` for the magnetic channels and `milliVolt per kilometer` for the electric ones. The
spelling matters: mt_metadata resolves `millivolts per kilometer` to `unknown per kilometer` with a warning
and writes the channel with no unit, where `milliVolt per kilometer` resolves. No filter is attached to any
channel: the cache is already in physical units, so a filter here would be applied a second time.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .transients import MIN_SEGMENT_S, segments

LOCAL_CHANNELS = ("Hx", "Hy", "Hz", "Ex", "Ey")
AZIMUTH = {"hx": 0.0, "hy": 90.0, "hz": 0.0, "ex": 0.0, "ey": 90.0}


def station_objects(site, chans: dict, lat, lon, elev, start, n, fs, dipole_n=None, dipole_e=None):
    """(Station metadata, RunTS) for one stretch of one station. No filters are attached."""
    from mth5.timeseries import ChannelTS, RunTS
    from mt_metadata.timeseries import Electric, Magnetic, Run, Station

    sm = Station()
    sm.id = site
    sm.location.latitude, sm.location.longitude = float(lat), float(lon)
    if elev is not None and np.isfinite(elev):
        sm.location.elevation = float(elev)
    rm = Run()
    rm.id = "001"
    rm.sample_rate = float(fs)
    rm.data_type = "LPMT"
    rm.time_period.start = str(start)
    idx = pd.date_range(start=start, periods=int(n), freq=pd.Timedelta(seconds=1.0 / float(fs)))
    cts = []
    for comp, arr in chans.items():
        if comp.startswith("h"):
            md = Magnetic()
            md.units = "nanoTesla"
            md.sensor.type = "fluxgate"
            md.measurement_azimuth = AZIMUTH[comp]
            md.measurement_tilt = 90.0 if comp == "hz" else 0.0
            ctype = "magnetic"
        else:
            md = Electric()
            md.units = "milliVolt per kilometer"
            md.measurement_azimuth = AZIMUTH[comp]
            length = dipole_n if comp == "ex" else dipole_e
            if length is not None and np.isfinite(length):
                md.dipole_length = float(length)
            ctype = "electric"
        md.component = comp
        md.sample_rate = float(fs)
        md.time_period.start = str(start)
        cts.append(ChannelTS(channel_type=ctype, data=np.asarray(arr, float), channel_metadata=md,
                             station_metadata=sm, run_metadata=rm))
    run = RunTS(array_list=cts, run_metadata=rm, station_metadata=sm)
    run.dataset = run.dataset.assign_coords(time=idx)
    return sm, run


def write_h5(path, site, local: dict, reference, t0, fs, survey_name, site_row, keep=None,
             reference_row=None, min_segment_s=MIN_SEGMENT_S):
    """Write the MTH5 one pass reads. Returns (path, [(offset, length)] of the runs written).

    `reference` is (station id, {'Hx','Hy'}) or None for the single station. `site_row` is the sites.csv row
    of the target, which carries the position and the dipole lengths; `reference_row` is the sites.csv row of
    a remote site where the reference is one, and the target's row otherwise.
    """
    from mth5.mth5 import MTH5
    n = len(local["Hx"])
    segs = [(0, n)] if keep is None else segments(np.asarray(keep, bool)[:n], int(min_segment_s * fs))
    if not segs:
        raise ValueError("%s: the keep mask leaves no run of %g s or longer" % (site, min_segment_s))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    lat, lon = float(site_row.lat), float(site_row.lon)
    elev = float(site_row.elev_m) if str(site_row.elev_m) not in ("", "nan") else float("nan")
    dn, de = _dipoles(site_row)
    stations = [(site, {k.lower(): local[k] for k in LOCAL_CHANNELS}, lat, lon, elev, dn, de)]
    if reference is not None and reference[0] and reference[1] is not None:
        rname, rh = reference
        rr = reference_row if reference_row is not None else site_row
        stations.append((rname, {"hx": rh["Hx"][:n], "hy": rh["Hy"][:n]},
                         float(rr.lat), float(rr.lon), float("nan"), None, None))
    m = MTH5()
    m.open_mth5(str(path), "w")
    try:
        m.add_survey(survey_name)
        for name, data, la, lo, el, d_n, d_e in stations:
            st = None
            for i, (off, ln) in enumerate(segs):
                rid = "%03d" % (i + 1)
                start = pd.Timestamp(int(round(t0 + off / fs)), unit="s")
                sm, run = station_objects(name, {k: v[off:off + ln] for k, v in data.items()},
                                          la, lo, el, start, ln, fs, d_n, d_e)
                run.run_metadata.id = rid
                if st is None:
                    st = m.add_station(name, station_metadata=sm, survey=survey_name)
                st.add_run(rid, run_metadata=run.run_metadata).from_runts(run)
    finally:
        m.close_mth5()
    return path, segs


def _dipoles(row):
    def one(cell):
        s = str(cell).strip()
        if s.lower().startswith("assume:"):
            s = s.split(":", 1)[1]
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return one(row.dipole_n_m), one(row.dipole_e_m)


def remove(path) -> str:
    """Delete a written MTH5, and say so rather than raise where the file is still held open.

    On Windows a handle the HDF5 library has not yet released makes the unlink raise, and a scratch file
    left behind is not worth failing a product for.
    """
    import gc
    gc.collect()
    try:
        Path(path).unlink(missing_ok=True)
        return ""
    except OSError as exc:
        return "%s was not removed: %s" % (Path(path).name, str(exc)[:120])


def read_back(path, site, local: dict, segs, survey_name) -> dict:
    """Compare the MTH5's own samples with the arrays it was written from, run by run and channel by channel.

    Returns the run count, the samples compared, the largest absolute difference and the channels that
    differ. The comparison is on the float32 the file stores, so a difference above 1e-3 nT or mV/km is the
    file not carrying the record it was given.
    """
    from mth5.mth5 import MTH5
    m = MTH5()
    m.open_mth5(str(path), "r")
    try:
        st = m.get_station(site, survey=survey_name)
        runs = [r for r in st.groups_list if r not in ("Transfer_Functions", "Fourier_Coefficients")]
        worst, compared, bad = 0.0, 0, []
        for i, (off, ln) in enumerate(segs):
            rid = "%03d" % (i + 1)
            if rid not in runs:
                bad.append("run %s is not in the file" % rid)
                continue
            rg = st.get_run(rid)
            for ch in LOCAL_CHANNELS:
                got = np.asarray(rg.get_channel(ch.lower()).hdf5_dataset[...], float)
                want = np.asarray(local[ch][off:off + ln], float)
                k = min(len(got), len(want))
                if len(got) != len(want):
                    bad.append("%s %s: %d samples in the file, %d written" % (rid, ch, len(got), len(want)))
                d = np.nanmax(np.abs(got[:k] - want[:k])) if k else 0.0
                compared += k
                worst = max(worst, float(d))
                if d > 1e-3:
                    bad.append("%s %s: largest difference %.6g" % (rid, ch, d))
        return dict(runs_in_file=len([r for r in runs if r.isdigit()]), runs_written=len(segs),
                    samples_compared=int(compared), worst_difference=float(worst), problems=bad)
    finally:
        m.close_mth5()
