"""What the 10 Hz row reads against the 1 Hz row of the same survey, and the caveat that states it.

A caveat has to be true of the survey it sits in (Ben's ruling, 2026-09-17). The departure a 10 Hz transfer
function carries against its own 1 Hz transfer function is a property of the instrument, the cache and the
band file of that survey, so it is measured on the survey's own whole-record transfer functions rather than
quoted from another one.

    departure      the median of rho(10 Hz) / rho(1 Hz) over BAND_S = 4-32 s, per component. The 1 Hz row is
                   interpolated in log period and log rho onto the 10 Hz periods inside the band, and a
                   10 Hz period the 1 Hz row does not reach is dropped rather than compared against the end
                   point numpy.interp would clamp to.
    measure        that median over the whole-record 10 Hz transfer functions of one kind and their 1 Hz
                   twins, with the sites counted. The whole record is the only fair measurement of the rate,
                   because a selection of hours differs from the 1 Hz row in which hours it holds as well as
                   in rate, so the `stretch` and `control` passes are not measured on.
    caveat         the sentence the EDI and the prose carry, built from a measurement or, where the survey
                   has none, saying that it has none. No number from another survey appears in either.

The record is written once per survey to <work_root>/survey/rate_departure.json and read by the pass, so
every 10 Hz transfer function of a survey carries the same measured sentence.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BAND_S = (4.0, 32.0)
RECORD_NAME = "rate_departure.json"
COMPONENTS = ("xy", "yx")


def record_path(work_root) -> Path:
    return Path(work_root) / "survey" / RECORD_NAME


def rho(tf, comp):
    """(period, apparent resistivity) of one component, the non-finite and zero entries dropped."""
    i, j = (0, 1) if comp == "xy" else (1, 0)
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.z, complex)[:, i, j]
    good = np.isfinite(z) & (np.abs(z) > 0) & np.isfinite(p) & (p > 0)
    return p[good], 0.2 * p[good] * np.abs(z[good]) ** 2


def departure(tf10, tf1, band_s=BAND_S) -> dict:
    """{component: the median rho ratio over the band}, NaN where the band holds no comparable period."""
    out = {}
    for comp in COMPONENTS:
        p10, r10 = rho(tf10, comp)
        p1, r1 = rho(tf1, comp)
        out[comp] = float("nan")
        if len(p1) < 2 or not len(p10):
            continue
        m = (p10 >= band_s[0]) & (p10 < band_s[1]) & (p10 >= p1.min()) & (p10 <= p1.max())
        if not m.any():
            continue
        base = np.exp(np.interp(np.log(p10[m]), np.log(p1), np.log(r1)))
        out[comp] = float(np.median(r10[m] / base))
    return out


def measure(pairs, kind, band_s=BAND_S) -> dict:
    """The survey's own departure from `pairs`, a list of (site, the 10 Hz TFData, the 1 Hz TFData).

    Returns the median ratio per component over the sites, the per-site values, the band, the kind the
    measurement was made on and the site count. A pair whose band holds no comparable period is not counted.
    """
    per_site, vals = {}, {c: [] for c in COMPONENTS}
    for site, tf10, tf1 in pairs:
        d = departure(tf10, tf1, band_s)
        per_site[str(site)] = {c: (None if not np.isfinite(d[c]) else round(d[c], 4)) for c in COMPONENTS}
        for c in COMPONENTS:
            if np.isfinite(d[c]):
                vals[c].append(d[c])
    med = {c: (round(float(np.median(vals[c])), 4) if vals[c] else None) for c in COMPONENTS}
    return dict(band_s=list(band_s), kind=str(kind), n_sites=len(per_site),
                n_scored={c: len(vals[c]) for c in COMPONENTS}, median_ratio=med, per_site=per_site)


def caveat(rec=None, band_s=BAND_S) -> str:
    """The sentence every 10 Hz transfer function carries, built from a measurement or saying there is
    none."""
    if not rec or not rec.get("median_ratio") or all(v is None for v in rec["median_ratio"].values()):
        return ("the departure of this survey's 10 Hz row from its own 1 Hz row has not been measured, "
                "because the survey carries no whole-record 10 Hz transfer function to measure it on; "
                "not spliced")
    b = rec.get("band_s") or list(band_s)
    parts = []
    for c in COMPONENTS:
        v = rec["median_ratio"].get(c)
        if v is not None:
            parts.append("%s %+.1f per cent" % (c, 100.0 * (float(v) - 1.0)))
    n = max(rec.get("n_scored", {}).values() or [0]) if rec.get("n_scored") else rec.get("n_sites", 0)
    return ("Aurora at 10 Hz reads %s at %g-%g s against its own 1 Hz transfer function, measured on this "
            "survey's %d whole-record 10 Hz %s transfer function(s); not spliced"
            % (" and ".join(parts), b[0], b[1], int(n), rec.get("kind", "")))


def write_record(work_root, rec) -> Path:
    p = record_path(work_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return p


def read_record(work_root) -> dict:
    p = record_path(work_root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------- the caveat in a finished file

CAVEAT_KEY = b"caveat_10hz="
REWRITE_NAME = "caveat_rewritten_at"


def _split(raw: bytes):
    """(the lines with their endings, the index of the one caveat line), or (the lines, the refusal)."""
    lines = raw.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if CAVEAT_KEY in ln]
    if not hits:
        return lines, "the file carries no caveat_10hz line"
    if len(hits) > 1:
        return lines, ("the file carries %d caveat_10hz lines and which one is the header's is not decided"
                       % len(hits))
    return lines, hits[0]


def rewrite_caveat(path, sentence) -> dict:
    """Replace the one caveat_10hz line of a finished transfer function, and verify no other byte moved.

    The work is done on the file's bytes, so no encoding or newline round trip can alter a character the
    rewrite did not mean to touch. The new line keeps the old line's prefix up to the key and its own line
    ending, and every other line is carried over unread. After the write the file is read back and the two
    are compared with that one line removed from each: where any other byte differs the original bytes are
    put back, `changed` is False and the reason says so.

    Returns {path, changed, old, new, reason}.
    """
    path = Path(path)
    out = dict(path=str(path), changed=False, old=None, new=None, reason="")
    if "\n" in str(sentence) or "\r" in str(sentence):
        out["reason"] = "the sentence carries a line ending and cannot be one header line"
        return out
    if not path.exists():
        out["reason"] = "the file is not there"
        return out
    raw = path.read_bytes()
    lines, where = _split(raw)
    if isinstance(where, str):
        out["reason"] = where
        return out
    head, _key, tail = lines[where].partition(CAVEAT_KEY)
    ending = b""
    for e in (b"\r\n", b"\n", b"\r"):
        if tail.endswith(e):
            ending = e
            break
    out["old"] = tail[:len(tail) - len(ending)].decode("utf-8", "replace")
    out["new"] = str(sentence)
    if out["old"] == out["new"]:
        out["reason"] = "the file already carries the survey's sentence"
        return out
    new_lines = list(lines)
    new_lines[where] = head + CAVEAT_KEY + str(sentence).encode("utf-8") + ending
    without_old = b"".join(lines[:where] + lines[where + 1:])
    path.write_bytes(b"".join(new_lines))
    back_lines, back_where = _split(path.read_bytes())
    if isinstance(back_where, str) or back_where != where or \
            b"".join(back_lines[:where] + back_lines[where + 1:]) != without_old:
        path.write_bytes(raw)
        out["reason"] = "the file differed outside the caveat line after the write, so the original bytes "\
                        "were put back"
        return out
    out["changed"] = True
    return out


def rewrite_all(work_root, sentence, pattern="*/*/*_10hz_*.edi", twin=True) -> list:
    """Rewrite the caveat line of every 10 Hz transfer function under a work root, one record per file.

    The XML twin is scored too and rewritten only where it carries the line; the EMTFXML writer drops the
    processing parameters, so on this package's files it does not carry one, and the record says that rather
    than the caller assuming it.
    """
    rows = []
    for p in sorted(Path(work_root).glob(pattern)):
        rows.append(rewrite_caveat(p, sentence))
        xml = p.with_suffix(".xml")
        if twin and xml.exists():
            r = rewrite_caveat(xml, sentence)
            if r["changed"] or "no caveat_10hz line" not in r["reason"]:
                rows.append(r)
    return rows


def record_rewrite(folder, rows, at=None) -> Path:
    """Append what was rewritten to a run folder's provenance.json, under `caveat_rewritten_at`."""
    from datetime import datetime, timezone
    p = Path(folder) / "provenance.json"
    if not p.exists():
        return p
    d = json.loads(p.read_text(encoding="utf-8"))
    stamp = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = list(d.get(REWRITE_NAME) or [])
    for r in rows:
        if r.get("changed"):
            entries.append(dict(at=stamp, transfer_function=Path(r["path"]).name, old=r["old"],
                                new=r["new"]))
    d[REWRITE_NAME] = entries
    p.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    return p


def carries_caveat(path, sentence) -> bool:
    """True where a written transfer function's caveat_10hz line is the sentence given."""
    p = Path(path)
    if not p.exists():
        return False
    lines, where = _split(p.read_bytes())
    if isinstance(where, str):
        return False
    _h, _k, tail = lines[where].partition(CAVEAT_KEY)
    return tail.decode("utf-8", "replace").strip() == str(sentence).strip()
