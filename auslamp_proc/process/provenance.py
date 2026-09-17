"""provenance.json: what one run rests on, including the parts that are inputs rather than measurements.

The shape is ported from vic_store_w3.provenance (:38-71). One file per run folder, holding the survey.yaml
fields the run used, the sites.csv and decisions.csv rows verbatim, the reference sidecar of every kind, the
engine and its version, the band object, the parameter set, the mask statistics per product, the rate, the
dates, the machine time and peak resident memory per product, and `caveats`.

`caveats` is never empty when one applies: an undecided sign, an assumed dipole, a reference refused with its
reason, and at 10 Hz the short-end caveat each add a line.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

SURVEY_FIELDS = ("name", "release", "raw_root", "work_root", "years", "frame", "declination_source",
                 "survey_id", "project", "geographic_name", "author", "observatory", "instruments",
                 "floors", "pool")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def caveats(site_row, applied_signs, undecided_signs, references: dict, rate, extra=(),
            work_root=None) -> list:
    """One line per open input. Empty only where nothing is open.

    `work_root` is where the survey's own measured 10 Hz departure is read from; without one the 10 Hz line
    says the departure has not been measured rather than quoting a figure from another survey.
    """
    out = []
    if undecided_signs:
        out.append("signs undecided at %s: used as +1 and recorded" % ", ".join(sorted(undecided_signs)))
    for col, ch in (("dipole_n_m", "Ex"), ("dipole_e_m", "Ey")):
        cell = str(site_row.get(col, "") if hasattr(site_row, "get") else site_row[col])
        if cell.lower().startswith("assume:"):
            out.append("assumed dipole for %s: %s (%s); rho scales with its square"
                       % (ch, cell, str(site_row["dipole_source"])[:120]))
    for kind, info in sorted((references or {}).items()):
        if not isinstance(info, dict):
            continue
        if info.get("error"):
            out.append("the %s reference was refused: %s" % (kind, str(info["error"])[:300]))
        for m, why in sorted((info.get("refusals") or {}).items()):
            out.append("%s reference refused the member %s: %s" % (kind, m, str(why)[:200]))
        for m, note in sorted((info.get("alignment_notes") or {}).items()):
            out.append("%s reference member %s: %s" % (kind, m, str(note)[:200]))
    if int(rate) == 10:
        from . import rate as RATE
        out.append(RATE.caveat(RATE.read_record(work_root) if work_root else None))
    out += [str(x) for x in extra]
    return out


def write(path, survey_cfg, site_row, decision_row, references, bands, params_name, params, rate,
          run_name, stamp, products, mask_stats, cache_sidecar, engine, engine_version, extra_caveats=(),
          pool=None, rot_segments=(), rot_drop=(), weight_rule="fleet", selection=None):
    """Write provenance.json into a run folder and return the dict it holds.

    `selection` is the hour selection record of process.selection where the run was made on one: the band and
    the Welch segment the hours were scored on, and per tag the fraction, the seed, the score threshold and
    the hours kept. It is empty for a whole-record run.
    """
    path = Path(path)
    d = dict(
        built_at=_now(),
        author=survey_cfg.get("author") or "",
        run=run_name,
        stamp=stamp,
        rate_hz=float(rate),
        weight_rule=weight_rule,
        pool=pool or {},
        rot_segments=list(rot_segments),
        rot_drop=list(rot_drop),
        member_weights_are_per_channel=False,
        member_weight_note="a member carries one scalar weight and it is applied to Hx and Hy alike; the "
                           "only per-channel mechanism is NaN, so a site with one sound channel is either "
                           "in the pool whole or out of it whole",
        machine=dict(node=platform.node(), python=platform.python_version(), system=platform.platform()),
        survey={k: survey_cfg.get(k) for k in SURVEY_FIELDS if k in survey_cfg},
        sites_row={k: (None if v is None else str(v)) for k, v in dict(site_row).items()},
        decisions_row=({k: (None if v is None else str(v)) for k, v in dict(decision_row).items()}
                       if decision_row is not None else None),
        references=references,
        engine=dict(name=engine, version=str(engine_version)),
        bands=dict(file=str(bands.file), name=Path(bands.file).name, levels=int(bands.levels),
                   window=int(bands.window), rate_hz=float(bands.rate_hz),
                   decimation_factors=bands.decimation_factors),
        parameter_set=dict(name=params_name, **{k: v for k, v in params.items()}),
        mask=mask_stats,
        selection=selection or {},
        products=products,
        cache=dict(builder=cache_sidecar.get("builder"), built_utc=cache_sidecar.get("built_utc"),
                   notch=cache_sidecar.get("notch_applied", "none"),
                   signs_in_cache=cache_sidecar.get("signs_applied", "none"),
                   frame_in_cache=cache_sidecar.get("frame_applied", "none"),
                   units=cache_sidecar.get("units"), t0_iso=cache_sidecar.get("t0_iso"),
                   span_days=cache_sidecar.get("span_days")),
        caveats=list(extra_caveats),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    return d


def read(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
