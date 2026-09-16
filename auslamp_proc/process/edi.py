"""The product's header: the survey, the station, the frame and the processing parameters.

Ported from wamt_run.finish_edi (:921-1021) and the frame block of wamt_esp2026_products.py (:41-181).

The EDI writer accepts station_metadata.comments and then drops them, so everything a reader has to see is a
processing_parameters line. The lines carry the kind and its members with their weights and lags, the
reference coverage, the band file with its level count and window length, the rate, the parameter set, the
mask statistics, the signs applied and those still undecided, the rotation angle, the engine and its version,
the cache's builder stamp and notch record, and at 10 Hz the short-end caveat.

The frame is stated in three lines (Ben's ruling, 2026-09-11): the tensor is served in the frame it was
processed in, the IGRF declination is recorded and NOT applied, and the angle to turn the tensor by for true
geographic north is given with the transformation (Z' = R Z R^T, T' = T R^T, R = [[cos, sin], [-sin, cos]]).

The XML twin is written from the same object after the EDI. The EMTFXML writer builds its Survey id from
survey_metadata.geographic_name, and mt_metadata validates an id against ^[a-zA-Z0-9_\\- ]*$, so anything
else in that name -- a comma is enough -- raises and leaves no XML; the name is put through that pattern
before the XML write and the EDI is not touched by it.

Every string comes from survey.yaml and sites.csv; nothing about a survey is written into this module.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TEN_HZ_CAVEAT = ("Aurora at 10 Hz reads about 8 per cent low at 4-32 s against its own 1 Hz product "
                 "(AusLAMP Victoria, 2026-09-11); not spliced")
# what mt_metadata accepts in an id, and so in the survey name the EMTFXML writer builds one from
XML_ID_BAD = re.compile(r"[^A-Za-z0-9_\- ]")
AZIMUTH = {"ex": 0.0, "ey": 90.0, "hx": 0.0, "hy": 90.0, "hz": 0.0}


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def _num(cell, default=float("nan")) -> float:
    s = str(cell).strip()
    if s.lower().startswith("assume:"):
        s = s.split(":", 1)[1]
    try:
        return float(s)
    except ValueError:
        return default


def frame_lines(declination_deg, rotation_deg) -> list:
    """The three processing_parameters lines that say what frame the tensor is in and how to leave it."""
    d = None if declination_deg is None or not np.isfinite(declination_deg) else float(declination_deg)
    return [
        "reference_frame=geomagnetic north: the tensor is served in the frame it was processed in, the "
        "horizontal magnetic pair turned %s into the mean horizontal field frame"
        % (("by %+.3f deg" % rotation_deg) if isinstance(rotation_deg, (int, float))
           and np.isfinite(rotation_deg) else "per rotation regime"),
        "declination_deg=%s" % (("%+.3f east (IGRF), RECORDED AND NOT APPLIED" % d) if d is not None
                                else "not available"),
        "to_geographic_north_deg=%s: turn the tensor by this angle for true geographic north "
        "(Z' = R Z R^T, T' = T R^T, R = [[cos,sin],[-sin,cos]])"
        % (("%+.3f" % -d) if d is not None else "not available"),
    ]


def reference_lines(kind: str, info: dict) -> list:
    """The lines naming the reference kind, its members with their weights and lags, and its coverage."""
    from . import KIND_WORD
    out = ["reference_kind=%s (%s)" % (KIND_WORD.get(kind, kind), kind)]
    if kind == "single":
        out.append("reference_members=none: the single station is the control that says what a reference buys")
        return out
    members = [m for m in (info.get("members") or []) if m.get("role") != "refused"]
    out.append("reference_members=" + ("; ".join(
        "%s weight %s lag %+.2f s%s" % (m.get("name"), m.get("weight"), float(m.get("lag_s") or 0.0),
                                        (" [%s]" % m["note"]) if m.get("note") else "")
        for m in members) or "none"))
    out.append("reference_coverage=%s of the record carries a sound reference" % info.get("coverage"))
    if info.get("weight_rule"):
        out.append("reference_weight_rule=%s: each member is weighted by its median coherence with the "
                   "fleet at %s s, never with the target"
                   % (info["weight_rule"], "-".join(str(x) for x in info.get("weight_band_s", []))))
    if info.get("branch") is not None:
        out.append("remote_choice=branch %s, %s" % (info.get("branch"), str(info.get("reason", ""))[:400]))
    if info.get("refusals"):
        out.append("reference_refusals=" + "; ".join("%s: %s" % kv for kv in sorted(info["refusals"].items())))
    return out


def mask_lines(stats: dict, runs: int, floor_frac: float) -> list:
    return [
        "mask=%d target and %d reference transient interval(s) and %d E burst(s); %.2f %% of the record "
        "dropped, of which %.2f %% is the transient mask and %.2f %% the reference's own coverage"
        % (stats.get("n_intervals_site", 0), stats.get("n_intervals_remote", 0),
           stats.get("n_intervals_e", 0), 100 * stats.get("mask_dropped_frac", 0.0),
           100 * stats.get("transient_dropped_frac", 0.0),
           100 * stats.get("reference_coverage_dropped_frac", 0.0)),
        "runs=%d kept stretch(es) of at least 3600 s became one Aurora run each; the hour floor drops a "
        "further %.2f %% of the record the mask kept" % (runs, 100 * float(floor_frac)),
        "selection=%s, which costs a further %.2f %% of the record"
        % (stats.get("selection", "none: the whole record"),
           100 * stats.get("selection_dropped_frac", 0.0)),
    ]


def sign_lines(applied: dict, undecided: list) -> list:
    return [
        "signs_applied=" + ("; ".join("%s %+d" % (k, int(v)) for k, v in sorted(applied.items())) or "none"),
        "signs_undecided=" + ((", ".join(sorted(undecided)) + ": used as +1 and recorded as undecided")
                              if undecided else "none: every sign in decisions.csv carries a value"),
    ]


def finish_edi(edi_in, out, site_row, decision_row, cfg, kind, info, params_lines, window, rotation_deg,
               engine, engine_version, remote_ids=(), rate=None):
    """Rewrite a raw Aurora EDI with the survey's identity, the station's facts and the provenance.

    Returns (the path written, [the metadata fields that could not be set]). Nothing about the survey is in
    this module: `cfg` is survey.yaml and `site_row` is the sites.csv row.
    """
    from mt_metadata.transfer_functions.core import TF

    a, b = window
    tf = TF(fn=str(edi_in))
    tf.read()
    missed = []

    def _set(obj, path, val):
        try:
            o = obj
            *head, leaf = path.split(".")
            for h in head:
                o = getattr(o, h)
            setattr(o, leaf, val)
        except Exception as exc:
            missed.append("%s: %s: %s" % (path, type(exc).__name__, exc))

    inst = str(site_row.instrument)
    const = (cfg.get("instruments") or {}).get(inst, {})
    s = tf.survey_metadata
    for k, v in (("id", cfg.get("survey_id") or cfg.get("name")),
                 ("name", cfg.get("project") or cfg.get("name")),
                 ("project", cfg.get("project") or cfg.get("name")),
                 ("geographic_name", cfg.get("geographic_name") or ""),
                 ("acquired_by.author", cfg.get("organization") or ""),
                 ("acquired_by.organization", cfg.get("organization") or ""),
                 ("summary", cfg.get("release") or "")):
        _set(s, k, v)

    st = tf.station_metadata
    st.id = str(site_row.site)
    dec_deg = _num(site_row.declination_deg, None)
    for k, v in (("geographic_name", "%s, %s" % (site_row.site, cfg.get("geographic_name") or "")),
                 ("location.latitude", float(site_row.lat)), ("location.longitude", float(site_row.lon)),
                 ("location.elevation", _num(site_row.elev_m, 0.0)), ("location.datum", "WGS84"),
                 ("orientation.reference_frame", "geomagnetic"),
                 ("orientation.method", "compass"),
                 ("time_period.start", _iso(a)), ("time_period.end", _iso(b)),
                 ("provenance.software.name", engine),
                 ("provenance.software.version", str(engine_version)),
                 ("provenance.creator.author", cfg.get("author") or ""),
                 ("provenance.submitter.author", cfg.get("author") or ""),
                 ("acquired_by.author", cfg.get("organization") or ""),
                 ("provenance.creation_time", datetime.now(timezone.utc).isoformat()),
                 ("data_type", "LPMT")):
        _set(st, k, v)
    if dec_deg is not None and np.isfinite(dec_deg):
        mid = _iso(0.5 * (float(a) + float(b)))
        # without an epoch the writer emits a default 1980 time stamp beside the declination, which a
        # reader would take for the epoch the value was computed at
        for k, v in (("location.declination.value", round(float(dec_deg), 3)),
                     ("location.declination.model", "IGRF"),
                     ("location.declination.epoch", mid),
                     ("location.declination.comments",
                      "%s; recorded, not applied: the tensor is served in the geomagnetic frame"
                      % (cfg.get("declination_source") or "IGRF"))):
            _set(st, k, v)
        try:
            st.location.declination.comments.time_stamp = mid
        except Exception:
            pass
    dn, de = _num(site_row.dipole_n_m), _num(site_row.dipole_e_m)
    # the rate the product was processed at, which on a 10 Hz recorder decimated to 1 Hz is not the
    # recorder's own rate in sites.csv
    fs = float(rate) if rate else _num(site_row.sample_rate_hz, 1.0)
    for run in st.runs:
        _set(run, "data_type", "LPMT")
        _set(run, "sample_rate", float(fs))
        _set(run, "data_logger.manufacturer", const.get("logger_manufacturer", ""))
        _set(run, "data_logger.model", const.get("logger_model", inst))
        if str(site_row.serial) not in ("", "nan"):
            _set(run, "data_logger.id", str(site_row.serial))
        for ch in run.channels:
            comp = str(ch.component).lower()
            _set(ch, "measurement_azimuth", AZIMUTH.get(comp, 0.0))
            _set(ch, "translated_azimuth", AZIMUTH.get(comp, 0.0))
            if comp.startswith("e"):
                _set(ch, "units", "milliVolt per kilometer")
                length = dn if comp == "ex" else de
                if np.isfinite(length):
                    _set(ch, "dipole_length", float(length))
            else:
                _set(ch, "units", "nanoTesla")
                _set(ch, "sensor.type", "fluxgate")
                _set(ch, "sensor.manufacturer", const.get("sensor_manufacturer", ""))
                _set(ch, "sensor.model", const.get("sensor_model", ""))

    tfm = st.transfer_function
    plist = list(params_lines)
    plist += ["record.start=%s" % _iso(a), "record.end=%s" % _iso(b),
              "record.days=%.2f" % ((b - a) / 86400.0),
              "site.dipole_n_m=%s" % site_row.dipole_n_m, "site.dipole_e_m=%s" % site_row.dipole_e_m,
              "site.dipole_source=%s" % str(site_row.dipole_source)[:200],
              "site.position_source=%s" % str(site_row.position_source)[:200]]
    plist += frame_lines(dec_deg, rotation_deg)
    for k, v in (("id", str(site_row.site)),
                 ("processed_by.author", cfg.get("author") or ""),
                 ("processed_date", datetime.now(timezone.utc).date().isoformat()),
                 ("processing_type", "robust remote reference" if remote_ids else "robust single station"),
                 ("remote_references", list(remote_ids or [])),
                 ("processing_parameters", plist),
                 ("software.name", engine),
                 ("software.version", str(engine_version)),
                 ("software.author", cfg.get("author") or ""),
                 ("sign_convention", "+"),
                 ("units", "milliVolt per kilometer per nanoTesla"),
                 ("coordinate_system", "geomagnetic")):
        _set(tfm, k, v)
    if not tfm.runs_processed:
        _set(tfm, "runs_processed", [r.id for r in st.runs])
    _set(st, "comments",
         "E as laid in the instrument frame; H turned into the mean horizontal field frame. "
         "Reference: %s. Sample rate %g Hz." % (", ".join(remote_ids) if remote_ids
                                                else "none (single station)", fs))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tf.write(fn=str(out), file_type="edi", longitude_format="LONG", latlon_format="dd")
    except TypeError:
        tf.write(fn=str(out), file_type="edi")
    # The EMTFXML writer builds its Survey id from survey_metadata.geographic_name, and mt_metadata
    # validates an id against ^[a-zA-Z0-9_\- ]*$, so a comma in the name raises and leaves no XML. The EDI
    # is written first and is not touched by this.
    s.geographic_name = XML_ID_BAD.sub(" ", str(s.geographic_name or "")).strip()
    xml_out, xml_error = None, None
    try:
        xml_out = out.with_suffix(".xml")
        tf.write(fn=str(xml_out), file_type="emtfxml")
    except Exception as exc:
        xml_out, xml_error = None, "%s: %s" % (type(exc).__name__, str(exc)[:200])
    return out, missed, xml_out, xml_error


def has_tipper(edi_path) -> bool:
    """True where the written EDI carries a tipper. The tipper comes out of the Aurora pass with hz."""
    txt = Path(edi_path).read_text(encoding="utf-8", errors="ignore")
    return ">TXR" in txt or ">TX.EXP" in txt
