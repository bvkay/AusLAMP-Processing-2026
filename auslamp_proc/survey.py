"""Load and write a survey's three tables: survey.yaml, sites.csv and decisions.csv.

A survey lives in surveys/<name>/. survey.yaml holds the survey-level inputs: raw tree, per-instrument layout,
the years the survey ran, the instrument constants with their source, the observatory archive. sites.csv holds
one row per site, each value with its provenance, written by workbook 01. decisions.csv holds one row per site
of what the analyst decides.

Cell grammar:

    a value          measured or read from a file; the neighbouring *_source column names where from
    assume:<value>   used, and carried into every transfer function's provenance until replaced
    decide           not decided; a later workbook measures it and writes it back with its source
    empty            not known, and nothing depends on it

write_table preserves a stored assume: or decide cell where the incoming cell is empty, and lets a caller
holding a value write it: a workbook that computed nothing leaves the cell empty and the analyst's input
survives, and an analyst's own write fills the cell. diff_tables reports the cells that differ between two
versions of a table.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent
SURVEYS = PKG_ROOT / "surveys"

# One row per site of what was recorded, where and when. Written by workbook 01; every value carries a source.
SITES_COLUMNS = [
    "site", "raw_path", "instrument", "serial", "serial_source",
    "layout", "sample_rate_hz",
    "files", "start_utc", "end_utc", "days", "date_correction",
    "lat", "lon", "elev_m", "position_source", "position_scatter_m",
    "dipole_n_m", "dipole_e_m", "dipole_source",
    "azimuth_n_deg", "azimuth_e_deg", "azimuth_source",
    "declination_deg", "observatory", "notes",
]

# One row per site of what the analyst decides. Created by workbook 01 with `decide` in every cell except those
# a survey arrives with already decided; filled by the later workbooks or by hand.
DECISIONS_COLUMNS = [
    "site",
    "sign_hx", "sign_hy", "sign_hz", "sign_ex", "sign_ey", "sign_source",
    "e_exchange", "e_exchange_source", "e_shift_s", "e_shift_source",
    "h_exchange", "h_gain", "h_gain_source",
    "h_lender", "h_lender_channels", "h_lender_source",
    "rot_regimes", "rot_drop", "windows", "keep_mask",
    "remote_site", "remote_source", "stack_members", "flags",
]

DECIDE = "decide"
ASSUME = "assume:"


@dataclass
class Survey:
    """One survey's three tables. cfg is survey.yaml; sites and decisions are DataFrames."""
    name: str
    cfg: dict
    sites: pd.DataFrame
    decisions: pd.DataFrame

    @property
    def folder(self) -> Path:
        return SURVEYS / self.name

    @property
    def raw_root(self) -> Path:
        return Path(self.cfg["raw_root"])

    @property
    def work_root(self) -> Path:
        return Path(self.cfg["work_root"])

    @property
    def years(self) -> list[int]:
        return [int(y) for y in self.cfg["years"]]

    def raw_path(self, site: str) -> Path:
        """The site's raw folder. survey.yaml raw_overrides takes precedence over <raw_root>/<site>."""
        over = (self.cfg.get("raw_overrides") or {}).get(site)
        if over:
            return Path(over)
        row = self.sites[self.sites.site == site]
        if len(row):
            return Path(row.iloc[0]["raw_path"])
        raise KeyError("%s is not in %s" % (site, self.folder / "sites.csv"))

    def instrument_constants(self, instrument: str) -> dict:
        return (self.cfg.get("instruments") or {}).get(instrument, {})

    def site(self, name: str) -> pd.Series:
        row = self.sites[self.sites.site == name]
        if not len(row):
            raise KeyError("%s is not in %s" % (name, self.folder / "sites.csv"))
        return row.iloc[0]

    def decision(self, name: str) -> pd.Series:
        row = self.decisions[self.decisions.site == name]
        if not len(row):
            raise KeyError("%s is not in %s" % (name, self.folder / "decisions.csv"))
        return row.iloc[0]


def load_survey(name: str) -> Survey:
    """Read surveys/<name>/survey.yaml, sites.csv and decisions.csv. A missing table loads as empty."""
    folder = SURVEYS / name
    cfg = yaml.safe_load((folder / "survey.yaml").read_text(encoding="utf-8"))
    sites = _read(folder / "sites.csv", SITES_COLUMNS)
    dec = _read(folder / "decisions.csv", DECISIONS_COLUMNS)
    return Survey(name, cfg, sites, dec)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    # every column as text: 'decide' and 'assume:50' share a column with numbers, and a dtype guess turns one
    # of them into NaN
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate(survey: Survey) -> list[str]:
    """One line per missing column, open decision or assumption. An empty list means no input is open."""
    out = []
    for label, frame, cols in (("sites.csv", survey.sites, SITES_COLUMNS),
                               ("decisions.csv", survey.decisions, DECISIONS_COLUMNS)):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            out.append("%s lacks columns: %s" % (label, ", ".join(missing)))
    for frame, label in ((survey.sites, "sites"), (survey.decisions, "decisions")):
        if "site" not in frame.columns:
            continue
        for _, r in frame.iterrows():
            und = sorted(k for k, v in r.items() if str(v).strip().lower() == DECIDE)
            asm = sorted("%s=%s" % (k, str(v)[len(ASSUME):]) for k, v in r.items()
                         if str(v).lower().startswith(ASSUME))
            if und:
                out.append("%s %s: decide %s" % (r["site"], label, ", ".join(und)))
            if asm:
                out.append("%s %s: assumed %s" % (r["site"], label, ", ".join(asm)))
    return out


def write_table(path: Path, new: pd.DataFrame, columns: list[str], key: str = "site") -> str:
    """Write `new` to `path`, keeping a stored assume: or decide cell only where the incoming cell is empty.

    A cell set to assume:<value> or left at decide is an input and not a computed value, so a workbook
    that computed nothing must not blank it, which is the case where the incoming cell is empty. A caller
    holding an actual value has measured something and writes it, which is how an analyst's own write fills
    a `decide` cell. Returns one line describing what was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # object dtype: a column holds 50.0 at one site and 'assume:50' at the next, which no typed column accepts
    out = new.reindex(columns=columns).astype(object)
    out = out.where(pd.notna(out), "")
    if not path.exists():
        out.to_csv(path, index=False)
        return "created %s (%d rows)" % (path.name, len(out))
    old = _read(path, columns).set_index(key)
    kept = 0
    for i, r in out.iterrows():
        k = r[key]
        if k not in old.index:
            continue
        for c in columns:
            if c == key or c not in old.columns:
                continue
            v = str(old.loc[k, c])
            if v.strip().lower() == DECIDE or v.lower().startswith(ASSUME):
                if is_blank(out.at[i, c]) and str(out.at[i, c]) != v:
                    out.at[i, c] = v
                    kept += 1
    out.to_csv(path, index=False)
    return "rewrote %s (%d rows, %d assume:/decide cells kept)" % (path.name, len(out), kept)


def diff_tables(old: pd.DataFrame, new: pd.DataFrame, key: str = "site") -> pd.DataFrame:
    """The cells that differ between two versions of a table, as (key, column, old, new).

    A row present in only one table is reported with the missing side as '<absent>'.
    """
    o = old.set_index(key).astype(str) if len(old) else pd.DataFrame()
    n = new.set_index(key).astype(str) if len(new) else pd.DataFrame()
    rows = []
    for k in sorted(set(o.index) | set(n.index)):
        if k not in o.index:
            rows.append({key: k, "column": "<row>", "old": "<absent>", "new": "added"})
            continue
        if k not in n.index:
            rows.append({key: k, "column": "<row>", "old": "present", "new": "<absent>"})
            continue
        for c in sorted(set(o.columns) | set(n.columns)):
            a = str(o.loc[k, c]) if c in o.columns else "<absent>"
            b = str(n.loc[k, c]) if c in n.columns else "<absent>"
            if a != b:
                rows.append({key: k, "column": c, "old": a, "new": b})
    return pd.DataFrame(rows, columns=[key, "column", "old", "new"])


def is_blank(v) -> bool:
    """True where a caller supplied nothing at all: None, NaN, or a cell that is empty after a strip.

    Narrower than is_empty on purpose. `none` is a word this grammar uses -- h_lender `none` says the site's
    own H stands, which is a decision and not a silence -- so write_table's guard must not read it as an
    empty cell and revert the stored `decide` over it.
    """
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return str(v).strip().lower() in ("", "nan", "<na>")


def is_empty(v) -> bool:
    """True where a cell holds nothing: an empty string, NaN, or the strings a CSV round trip leaves behind."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return str(v).strip().lower() in ("", "nan", "none", "<na>")


# The columns of sites.csv that no raw file answers for. They are on a deployment sheet or an instrument
# register, so a re-run of the discovery carries them over from the table rather than recomputing them.
SHEET_COLUMNS = ["serial", "serial_source", "dipole_n_m", "dipole_e_m", "dipole_source",
                 "azimuth_n_deg", "azimuth_e_deg", "azimuth_source", "notes"]

OWNED_COLUMNS = ("notes",)


def carry_over(new: pd.DataFrame, old: pd.DataFrame, key: str = "site", columns=None,
               owned=OWNED_COLUMNS):
    """Fill the cells of `new` from `old` that a re-run of the discovery cannot compute.

    A dipole length, an arm bearing as laid and, on a recorder that writes no serial into its files, the
    serial are on a deployment sheet and nowhere in the raw, so re-running the discovery must not blank them:
    where a cell of `new` is empty and `old` holds a value, the old value is kept, and a cell `new` filled is
    left alone. A column of `old` that `new` does not carry at all is added.

    `owned` names the columns sites.csv owns outright, where the old value wins even over one the discovery
    computed: `notes` is free text and belongs to whoever wrote it.

    Returns (frame, ['column: n kept', ...]).
    """
    if not len(old) or key not in old.columns or not len(new):
        return new, []
    cols = [c for c in (columns or list(dict.fromkeys(list(new.columns) + list(old.columns))))
            if c != key and c in old.columns]
    o = old.set_index(key)
    out = new.copy().astype(object)
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    kept: dict = {}
    for i, k in zip(out.index, out[key]):
        if k not in o.index:
            continue
        for c in cols:
            if is_empty(o.loc[k, c]):
                continue
            if is_empty(out.at[i, c]) or (c in owned and str(out.at[i, c]) != str(o.loc[k, c])):
                out.at[i, c] = o.loc[k, c]
                kept[c] = kept.get(c, 0) + 1
    return out, ["%s: %d" % (c, n) for c, n in sorted(kept.items())]


def read_reference(spec: dict, path_key: str = "table", columns_key: str = "columns",
                   select_key: str = "select") -> pd.DataFrame:
    """A table named in survey.yaml, with its columns renamed to ours and its rows filtered.

    `spec` holds the path under `path_key`, a {our name: their name} map under `columns_key`, and an optional
    {column: value} filter under `select_key` for a table that covers more than one survey (one campaign file
    holding every phase). A mapped column the table does not carry is left out; a missing path returns an
    empty frame. The frame is a comparison or a regression and never a source: no value is read from it into
    sites.csv.
    """
    spec = spec or {}
    path = spec.get(path_key) or ""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    for col, val in (spec.get(select_key) or {}).items():
        if col in raw.columns:
            raw = raw[raw[col].astype(str) == str(val)]
    cm = spec.get(columns_key) or {}
    return pd.DataFrame({k: raw[v].values for k, v in cm.items() if v in raw.columns})


def select_sites(survey: Survey, spec, max_sites: int = 0, groups_csv=None) -> tuple[list[str], str]:
    """The sites a workbook was asked for, and one line saying where the set came from.

    `spec` is "all", "largest" (the largest group of the register's deployment_groups.csv), a group name such
    as "G01", or a list of site names. `max_sites` above zero keeps the first that many of the chosen set in
    the register's own order; the line says how many were dropped and names the ones kept.
    """
    known = list(survey.sites.site)
    if isinstance(spec, (list, tuple, set)):
        chosen = [s for s in spec if s in known]
        source = "the list the parameter names"
        missing = [s for s in spec if s not in known]
        if missing:
            source += " (not in sites.csv: %s)" % ", ".join(missing)
    elif str(spec).strip().lower() == "all":
        chosen, source = list(known), "every site in sites.csv"
    else:
        path = Path(groups_csv) if groups_csv else Path(survey.cfg["work_root"]) / "survey" / "deployment_groups.csv"
        if not path.exists():
            raise FileNotFoundError("%s has not been written; run workbook 01 first" % path)
        g = pd.read_csv(path)
        row = g.loc[g.n.astype(int).idxmax()] if str(spec).strip().lower() == "largest" \
            else g[g.group == str(spec)].iloc[0]
        chosen = [s for s in str(row.members).split() if s in known]
        source = "register group %s (%d members, common window %s d)" % (row.group, int(row.n),
                                                                        row.common_days)
    if max_sites and len(chosen) > int(max_sites):
        source += "; capped at %d of %d" % (int(max_sites), len(chosen))
        chosen = chosen[:int(max_sites)]
    return chosen, source


def blank_decisions(sites: list[str]) -> pd.DataFrame:
    """A decisions table with 'decide' in every cell except the site name."""
    rows = [{c: (s if c == "site" else DECIDE) for c in DECISIONS_COLUMNS} for s in sites]
    return pd.DataFrame(rows, columns=DECISIONS_COLUMNS)
