"""The Aurora pass: the band files, the parameter sets, and the call itself.

Ported from mundi_process.run_aurora (:508-583) and qld_campaign.AURORA_PARAMS (:265-271).

The engine is RME for a single station and RME_RR wherever a reference station is present. The tipper comes
out of the same pass, because hz is in the local station.

Two traps the call handles. mth5 recomputes the sample rate from the time index and the rounding depends on
record length, so two runs written at the same rate can differ in the twelfth decimal; Aurora then counts
them as mixed rates and refuses the kernel dataset, so the rates are snapped to nine decimals, which leaves
any real difference intact. And Aurora stores the window overlap in SAMPLES, so an override that changes the
window length after a fraction has been converted leaves the old count behind -- 512 samples at an overlap of
192 is 37.5 per cent, not the 75 asked for, silently -- so the overlap is re-derived after every override.

Aurora's default window is a boxcar with 32 of 256 samples of overlap. Boxcar leaks across the spectrum and
12.5 per cent overlap throws away most of the averaging the record could give, so the default parameter set
here is kaiser20_75: a Kaiser window of beta 20 at 75 per cent overlap.

A band file's indices are FFT harmonics of the WINDOW, so a file read at another window length names
different periods and nothing says so. File, level count and window are therefore one object, BANDS.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BANDS_DIR = Path(__file__).resolve().parent.parent / "bands"

AURORA_PARAMS = {
    "kaiser20_75": dict(window_type="kaiser", overlap=0.75, wargs={"beta": 20}),
    "kaiser20_50": dict(window_type="kaiser", overlap=0.50, wargs={"beta": 20}),
    "dpss4_75": dict(window_type="dpss", overlap=0.75, wargs={"NW": 4}),
    "kaiser20_w512": dict(window_type="kaiser", overlap=0.75, wargs={"beta": 20}, window_samples=512),
}
DEFAULT_PARAMS = "kaiser20_75"


@dataclass(frozen=True)
class BandSet:
    """One band file with the level count and the window length its indices are harmonics of."""
    file: Path
    levels: int
    window: int
    rate_hz: float

    @property
    def decimation_factors(self) -> list:
        return [1] + [4] * (self.levels - 1)


BANDS = {
    "1hz": BandSet(BANDS_DIR / "bs_1hz_6lv_48.cfg", 6, 256, 1.0),
    "10hz": BandSet(BANDS_DIR / "bs_10hz_5lv_31.cfg", 5, 256, 10.0),
}


def bands_for(rate) -> BandSet:
    key = "%dhz" % int(rate)
    if key not in BANDS:
        raise KeyError("no band file for %s; the package ships %s" % (key, ", ".join(sorted(BANDS))))
    return BANDS[key]


def band_table(bandset: BandSet, fs=None):
    """The band edges as Aurora computes them: one row per band with its level, harmonics and periods.

    Read through EMTFBandSetupFile.compute_band_edges and Processing.assign_bands, so the table is what
    Aurora will use rather than a reading of the text file.
    """
    import pandas as pd
    from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile
    from mt_metadata.processing.aurora.processing import Processing
    fs = float(bandset.rate_hz if fs is None else fs)
    b = EMTFBandSetupFile(filepath=str(bandset.file), sample_rate=fs)
    dec = bandset.decimation_factors
    edges = b.compute_band_edges(dec, b.num_decimation_levels * [bandset.window])
    p = Processing()
    p.assign_bands(edges, fs, dec, bandset.window)
    rows = []
    for d in p.decimations:
        for bd in d.bands:
            rows.append(dict(level=int(d.decimation.level), fs_hz=float(d.decimation.sample_rate),
                             harmonics="%d-%d" % (bd.index_min, bd.index_max),
                             lower_s=round(1 / bd.frequency_max, 3),
                             centre_s=round(1 / bd.center_frequency, 3),
                             upper_s=round(1 / bd.frequency_min, 3)))
    return pd.DataFrame(rows).sort_values(["level", "centre_s"]).reset_index(drop=True)


def run_aurora(paths, station, remote, bands, levels, out_edi, first_dec=1, window_type=None,
               overlap=None, window_samples=None, overrides=None, out_xml=None):
    """The Aurora call. Returns (the EDI written, the XML written or None, the transfer function).

    `paths` are the MTH5 files, `station` the local station id and `remote` the reference station id or None.
    `overrides` are dotted paths into a decimation level, so a parameter set can reach the stft and
    regression fields without a keyword each.
    """
    from aurora.config.config_creator import ConfigCreator
    from aurora.pipelines.process_mth5 import process_mth5
    from mth5.processing import KernelDataset, RunSummary

    rs = RunSummary()
    rs.from_mth5s([str(p) for p in paths])
    if "sample_rate" in rs.df.columns:
        rs.df["sample_rate"] = rs.df["sample_rate"].round(9)
    kd = KernelDataset()
    kd.from_run_summary(rs.clone() if hasattr(rs, "clone") else rs, station, remote)
    cc = ConfigCreator()
    kw = dict(decimation_factors=[first_dec] + [4] * (int(levels) - 1))
    if bands:
        kw["emtf_band_file"] = str(bands)
    cfg = cc.create_from_kernel_dataset(kd, **kw)
    for d in cfg.decimations:
        d.estimator.engine = "RME_RR" if remote else "RME"
        if window_samples:
            d.stft.window.num_samples = int(window_samples)
        if window_type:
            d.stft.window.type = window_type
        if overlap is not None:
            n = d.stft.window.num_samples
            d.stft.window.overlap = int(round(n * overlap)) if overlap < 1 else int(overlap)
        for key, value in (overrides or {}).items():
            obj, _, leaf = key.rpartition(".")
            target = d
            for part in obj.split("."):
                if part:
                    target = getattr(target, part)
            setattr(target, leaf, value)
        if overlap is not None and overlap < 1:
            n = d.stft.window.num_samples
            want = int(round(n * overlap))
            if d.stft.window.overlap != want:
                d.stft.window.overlap = want
    tf = process_mth5(cfg, kd, units="MT")
    if tf is None or not hasattr(tf, "write"):
        raise ValueError("aurora returned no transfer function")
    tf.station_metadata.id = station
    out_edi = Path(out_edi)
    out_edi.parent.mkdir(parents=True, exist_ok=True)
    # LONG= / REFLONG=, not the writer's default LON=: GeoTools keys on LONG and silently puts a site at
    # 0,0 when it finds LON instead. On a mt_metadata without those keywords the call raises TypeError and
    # the plain write still produces a usable file, whose header the finisher rewrites in any case.
    try:
        tf.write(fn=str(out_edi), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(out_edi), file_type="edi")
    xml = None
    if out_xml is not None:
        try:
            tf.write(fn=str(out_xml), file_type="emtfxml")
            xml = Path(out_xml)
        except Exception:
            xml = None
    return out_edi, xml, tf


def run_pass(h5, site, remote_id, rate, params, out_edi, out_xml=None, bandset=None):
    """One Aurora product. Returns (edi, xml or None, the parameter set used, the band set used)."""
    bs = bandset or bands_for(rate)
    p = AURORA_PARAMS[params]
    edi, xml, _tf = run_aurora([Path(h5)], site, remote_id, bs.file, bs.levels, Path(out_edi),
                               window_type=p["window_type"], overlap=p["overlap"],
                               window_samples=p.get("window_samples"),
                               overrides={"stft.window.additional_args": p["wargs"]},
                               out_xml=out_xml)
    return edi, xml, p, bs


def silence_loggers():
    """Import aurora and mth5, then drop loguru's sinks.

    Both re-arm loguru when they are first imported, which puts tens of thousands of INFO lines into a run.
    The "Window is longer than the time series" ERROR is expected: a one-hour run holds no complete window
    at the deep decimation levels and contributes nothing there.
    """
    import aurora                                       # noqa: F401
    import mth5                                         # noqa: F401
    from loguru import logger
    logger.remove()
