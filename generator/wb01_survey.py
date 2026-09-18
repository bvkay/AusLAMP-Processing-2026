r"""Workbook 01, the survey: the site table, the map, the register and the observatory.

The cells are a Python list of ("md", text) and ("code", source). generator/make_workbooks.py imports the
list from here and writes 01_survey.ipynb.

@author: ben kay (ben@auscope.org.au)
"""

WB01_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/; copy surveys/_template/ for your own survey
RAW_ROOT = None              # None = survey.yaml raw_root; a path here reads a copy of the release instead
WORK_ROOT = None             # None = survey.yaml work_root; everything this workbook writes lands under it
MIN_OVERLAP_DAYS = 7         # the floor a pair must overlap by to share a group; 7 d, raise it to cut the pool
FETCH = False                # True downloads the observatory days the archive lacks; a fetch is thousands of
                             # requests to the INTERMAGNET GIN, so it is never the default
'''

WB01_SETUP = '''import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# mt_io logs one INFO line per site while the sample rate is inferred, which buries the tables below
from loguru import logger as _loguru
_loguru.disable("mt_io")

import numpy as np
import pandas as pd

from auslamp_proc import survey as SV, register, geo, observatory
from auslamp_proc.raw import discover, edl, lemi, scan
from auslamp_proc.figures import survey as FIG

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 250)

sv = SV.load_survey(SURVEY)
if RAW_ROOT:
    sv.cfg["raw_root"] = RAW_ROOT
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
YEARS = sv.years
OUT = Path(sv.cfg["work_root"]) / "survey"
OUT.mkdir(parents=True, exist_ok=True)
WRITTEN = []

print("survey       %s" % sv.cfg["name"])
print("release      %s" % sv.cfg["release"])
print("raw root     %s" % sv.cfg["raw_root"])
print("work root    %s" % OUT)
print("years        %s" % YEARS)
print("frame        %s, declination %s" % (sv.cfg["frame"], sv.cfg["declination_source"]))
for inst, blk in sv.cfg["instruments_layout"].items():
    print("instrument   %-14s %-14s %s Hz under %s" % (inst, blk["layout"], blk["sample_rate_hz"], blk["subdir"]))
for site, path in (sv.cfg.get("raw_overrides") or {}).items():
    print("override     %-9s %s" % (site, path))
'''

WB01 = [
("md", r"""# 01 -- The survey: what was recorded, where, and when

This workbook reads a survey's raw folder and writes the site table the rest of the series works from: one row
per site carrying the instrument, the serial, the dipoles, the position from the logger's own GPS, the span
with the GPS week rollover applied, and the IGRF declination. It draws the survey map and the deployment
register, and it asks the INTERMAGNET archive whether it covers the survey's span.

No time-series body is opened. What is read is file names, the first and the last data file, the `.gps`,
`.gst`, `.pll` and `.INF` sidecars, and the first and last row of each LEMI daily file. A PR6-24 writes either
miniSEED, whose header carries the start, the rate and the sample count, or ASCII, which carries none of them:
for ASCII the start is the stamp in the file name, the count is the line count, and the rate is read from the
site's own consecutive files (`auslamp_proc.raw.edl.sample_rate`).

Nine checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported as FAIL and is
not revised afterwards.

The reference kinds the later workbooks build are named here once and used as words from then on:

| word | what it is | code key |
|---|---|---|
| remote site | one other site's H as the reference | `remote` |
| fleet stack | a coherence-weighted mean of several sites' H | `stack` |
| observatory | an INTERMAGNET one-second record as the reference | `obs` |
| stack + observatory | the stack with the observatory as a member | `stack_obs` |

A group in the deployment register is the pool those choices are drawn from: a reference can only help a
target over the time the two of them were both recording."""),

("code", WB01_PARAMS),

("md", r"""## The survey and its inputs

Everything below comes from `surveys/<SURVEY>/survey.yaml`. Read the instrument lines: each names the folder
its sites live under, the layout its files are written in, and the sample rate the files will be checked
against. An override line names a site whose raw folder is not where the release put it."""),

("code", WB01_SETUP),

("md", r"""## The raw tree

One row per site folder, read from file names alone. `stem_styles` names every file stem found: the site id is
the folder and never the stem. The stems run `<site>_`, `<site>` with no underscore, a two-digit form such as
`VIC73` for `VIC073`, a mixed case such as `q64n_` for `Q64N`, and at a handful of sites the name of the
previous deployment, because the logger had not been renamed when it wrote its first hour. `dropped` counts
what was left out and why (`auslamp_proc.raw.discover`): AppleDouble `._` copies, `LAB_` and `*_test*` bench
files, operator scripts, a file with no 12-digit stamp, and a stem whose stamps sit more than 30 days outside
the span of the site's own-stem files, which is a different deployment left in the folder. `notes` names the
day folders' non-numeric neighbours and every stem dropped that way.

**This check fails if any site folder yields no data files, if a file stem that is not the folder name is not
named in the row, if a site named in survey.yaml `checks.first_data_file` does not open on the file named
there, or if a `LAB_` or `*_test*` file on disk is not counted under `dropped`.** Each limb is scored against
a second walk of the tree that shares no code with the discovery: `discover.audit_stems` for the stems and
`discover.audit_bench_files` for the bench files, both pathlib and a plain regex where the discovery uses
os.scandir and `split_stem`. The third limb is the one that would catch a discovery globbing on the site name:
`checks.first_data_file` names the hour a site opens on that carries another site's name, and a discovery that
matched files by folder name would miss it."""),

("code", '''disc = discover.discover_survey(sv.cfg, years=YEARS, verbose=True)
cols = ["site", "instrument", "n_day_folders", "n_daily_files", "n_data_files", "first_stamp", "last_stamp",
        "stem_styles", "zipped", "sidecars", "n_inf", "dropped"]
print(disc[cols].to_string())
print()
ordinary = disc.notes.str.match(r"^beside the day folders: (config|log|temp)( (config|log|temp))*$")
print("%d sites have nothing beside the day folders but config, log or temp" % int(ordinary.sum()))
notable = disc[(disc.notes != "") & ~ordinary][["site", "notes"]]
print(notable.to_string() if len(notable) else "   none")

empty = sorted(disc[disc.n_data_files == 0].site)
unnamed, bench_bad = [], []
n_bench = 0
for _, r in disc.iterrows():
    on_disk = discover.audit_stems(r.raw_path, r.layout)
    named = set(r.stem_styles.split())
    for s in sorted(on_disk - named):
        if s not in r.notes:                       # a dropped stem is named in notes, not in stem_styles
            unnamed.append("%s: %s" % (r.site, s))
    found = discover.audit_bench_files(r.raw_path, r.layout)
    n_bench += len(found)
    counted = discover.dropped_count(r.dropped, "LAB_/test")
    if len(found) != counted:
        bench_bad.append("%s: %d on disk (%s), %d under dropped"
                         % (r.site, len(found), " ".join(found[:3]) or "none", counted))

want_first = (sv.cfg.get("checks") or {}).get("first_data_file") or {}
first_bad = []
for site, expect in sorted(want_first.items()):
    row = disc[disc.site == site]
    got = Path(row.iloc[0].first_file).name if len(row) and row.iloc[0].first_file else "<none>"
    if not got.startswith(expect):
        first_bad.append("%s: expected %s*, opened on %s" % (site, expect, got))
    else:
        print("first data file: %-9s %s, the name survey.yaml expects" % (site, got))
if n_bench:
    print("%d LAB_/test bench files found on disk and counted under dropped" % n_bench)

n = len(disc)
print()
if n == 0:
    print("VERDICT: UNJUDGED -- no site folder was found under %s" % sv.cfg["raw_root"])
elif empty or unnamed or first_bad or bench_bad:
    print("VERDICT: FAIL -- %d of %d sites yield no data files (%s); %d stems on disk are not named in their "
          "row (%s); %d of %d sites do not open on the file survey.yaml names (%s); %d sites miscount their "
          "bench files (%s)"
          % (len(empty), n, ", ".join(empty) or "none", len(unnamed), "; ".join(unnamed) or "none",
             len(first_bad), len(want_first), "; ".join(first_bad) or "none",
             len(bench_bad), "; ".join(bench_bad) or "none"))
else:
    print("VERDICT: PASS -- %d sites, %d data files, every folder yields data, every one of the %d distinct "
          "stems on disk is named in its row, the %d sites survey.yaml names open on the file it names, and "
          "all %d LAB_/test files on disk are counted under dropped"
          % (n, int(disc.n_data_files.sum()),
             len({s for _, r in disc.iterrows() for s in r.stem_styles.split()}), len(want_first), n_bench))
'''),

("md", r"""## Dates and the GPS week rollover

A logger whose almanac was stale stamps its files 1024 weeks (7168 days) early, and the day folder, the `.gps`
`>RTM` sentence and the `.gst` and `.pll` logs all carry the same early date, so no file on disk holds the
true one. The correction is +1024 weeks wherever the parsed year precedes the survey's `years`
(`auslamp_proc.raw.edl.parse_stamp`), which makes the survey years an input rather than a constant. The
released mt-io 0.0.5 reader does not apply it.

The worked example below is this survey's own first data file, the first row of the discovery table above. It
shows the stamp as the logger wrote it, the stamp the survey's `years` put it at, and the shift between them
in days and weeks. Where a survey needs no correction the example says so and the table's `date_correction`
column stays empty at every site.

Reading the span costs two data files per EDL site, plus up to 75 more where the rate has to be inferred from
ASCII, and the first and last row of two daily files per LEMI site; the positions in the next section cost up
to 40 small sidecars per site. Both are done in the one pass below, which is the slowest cell in the workbook.

**This check fails if any corrected start or end lies outside the survey's years.**"""),

("code", '''sites_scan = scan.scan(disc, sv.cfg, YEARS, release_mth5_by_instrument=sv.cfg.get("release_mth5"))

# a dipole, an arm bearing, a field note and some serials are on a deployment sheet and in no raw file, so a
# cell surveys/<SURVEY>/sites.csv already holds and this pass cannot compute is kept, not blanked
sites_scan, carried = SV.carry_over(sites_scan, sv.sites, columns=SV.SHEET_COLUMNS)
print("carried over from surveys/%s/sites.csv: %s" % (SURVEY, ", ".join(carried) or "nothing"))
print()

ex = disc[disc.first_stamp != ""].iloc[0]                   # <- the survey's own first site and its first file
ex_stamp = ex.first_stamp
raw_dt, _ = edl.parse_stamp(ex_stamp, ())                  # no years: the stamp as the logger wrote it
fix_dt, label = edl.parse_stamp(ex_stamp, YEARS)
print("worked example: %s opens on %s" % (ex.site, Path(ex.first_file).name))
print("               the stamp %s reads %s as written; with years %s the correction is %s -> %s"
      % (ex_stamp, raw_dt, YEARS, label or "none", fix_dt))
print("               the shift is %d days = %d weeks" % ((fix_dt - raw_dt).days, (fix_dt - raw_dt).days // 7))
print()
print(sites_scan[["site", "instrument", "files", "sample_rate_hz", "start_utc", "end_utc", "days",
                  "date_correction"]].to_string())

t0 = pd.to_datetime(sites_scan.start_utc, format="ISO8601", utc=True)
t1 = pd.to_datetime(sites_scan.end_utc, format="ISO8601", utc=True)
outside = [(s, str(a), str(b)) for s, a, b in zip(sites_scan.site, t0, t1)
           if pd.isna(a) or pd.isna(b) or a.year not in YEARS or b.year not in YEARS]
n_corr = int((sites_scan.date_correction != "").sum())
print()
if not len(sites_scan):
    print("VERDICT: UNJUDGED -- no site span was read")
elif outside:
    print("VERDICT: FAIL -- %d of %d spans fall outside the survey years %s: %s"
          % (len(outside), len(sites_scan), YEARS,
             "; ".join("%s %s..%s" % x for x in outside)))
else:
    print("VERDICT: PASS -- all %d spans lie inside %s (%s..%s); the rollover was applied at %d sites"
          % (len(sites_scan), YEARS, t0.min().date(), t1.max().date(), n_corr))
'''),

("md", r"""## Positions

The position of record is the instrument's own. For an EDL site it is the median over the daily `.gps` files
of the fixes with at least 3 satellites (`auslamp_proc.raw.edl.gps_position`): the median and not the first
fix, because a receiver can still be converging when the logger writes its first sentences. For a LEMI-424
site it is the median over the first row of every daily file, which carries a fix of its own.
`position_scatter_m` is the 95th percentile spread of the kept fixes about their own median.

Two outside positions are read for comparison and never used: the release's own MTH5, where the survey has
one, and the deployment sheet or site metadata table named in survey.yaml `comparison_positions`. A separation
of a few metres is the outside table agreeing with the logger; anything above a kilometre is it disagreeing
with the instrument that recorded the data, and every such site is listed below by name with both positions.
An outside table is never the position of record: a sheet is written by hand in the field and a release
position is re-typed from one.

A LEMI DATA folder can hold more than one deployment, because the logger was moved without a new folder being
started. Those folders are reported by name below and are never split automatically: which cluster is the site
is a decision, and it is recorded in survey.yaml `raw_overrides`.

**This check fails if any site's own-GPS scatter exceeds 100 m, or if any site has no fix with at least 3
satellites.**"""),

("code", '''CMP = sv.cfg.get("comparison_positions") or {}                # <- survey.yaml; {} where none exists
cmp_pos = SV.read_reference(CMP)
if len(cmp_pos):
    j_cmp = sites_scan.merge(cmp_pos, on="site", how="left", suffixes=("", "_cmp"))
    sites_scan["comparison_separation_m"] = [
        geo.distance_km((a, b), (x, y)) * 1000 if all(np.isfinite([a, b, x, y])) else np.nan
        for a, b, x, y in zip(pd.to_numeric(j_cmp.lat, errors="coerce"),
                              pd.to_numeric(j_cmp.lon, errors="coerce"),
                              pd.to_numeric(j_cmp.lat_cmp, errors="coerce"),
                              pd.to_numeric(j_cmp.lon_cmp, errors="coerce"))]
    sites_scan["comparison_lat"] = pd.to_numeric(j_cmp.lat_cmp, errors="coerce").values
    sites_scan["comparison_lon"] = pd.to_numeric(j_cmp.lon_cmp, errors="coerce").values
else:
    sites_scan["comparison_separation_m"] = np.nan
    sites_scan["comparison_lat"] = np.nan
    sites_scan["comparison_lon"] = np.nan

print(sites_scan[["site", "instrument", "lat", "lon", "elev_m", "position_scatter_m", "n_fixes",
                  "min_sat", "max_sat", "n_fixes_below_floor", "release_separation_km",
                  "comparison_separation_m"]].to_string())
print()
print("release MTH5 position compared, separation over 1 km:")
far = sites_scan[pd.to_numeric(sites_scan.release_separation_km, errors="coerce") > 1.0]
print(far[["site", "lat", "lon", "release_lat", "release_lon", "release_separation_km"]].to_string()
      if len(far) else "   none")
print()
print("%s compared, separation over 1 km:" % (CMP.get("label") or "no outside position table"))
far2 = sites_scan[pd.to_numeric(sites_scan.comparison_separation_m, errors="coerce") > 1000.0]
print(far2[["site", "lat", "lon", "comparison_lat", "comparison_lon",
            "comparison_separation_m"]].round(6).to_string() if len(far2) else "   none")
if len(cmp_pos):
    cs = pd.to_numeric(sites_scan.comparison_separation_m, errors="coerce")
    print("   %d of %d sites compared, median %.1f m, largest %.1f m at %s"
          % (int(cs.notna().sum()), len(sites_scan), cs.median(), cs.max(),
             sites_scan.site[cs.idxmax()] if cs.notna().any() else "none"))

print()
print("LEMI folders holding more than one deployment (the release folder, before any override):")
found_multi = 0
lemi_block = (sv.cfg["instruments_layout"] or {}).get("LEMI-424") or {}
for _, r in disc[disc.layout == "lemi_data_nnnn"].iterrows():
    rel = Path(sv.cfg["raw_root"]) / (lemi_block.get("subdir") or "") / r.site
    dd = sorted(p for p in rel.iterdir() if p.is_dir() and p.name.upper().startswith("DATA"))
    for d in dd:
        deps = lemi.deployments_in_folder(d)
        if len(deps) > 1:
            found_multi += 1
            print("   %-9s %s: %d deployments" % (r.site, d.name, len(deps)))
            for c in deps:
                print("      %s..%s  %3d files  %.4f, %.4f" % (c["first"], c["last"], c["n_files"],
                                                               c["lat"], c["lon"]))
print("   none" if not found_multi else "")

scat = pd.to_numeric(sites_scan.position_scatter_m, errors="coerce")
sats = pd.to_numeric(sites_scan.n_fixes, errors="coerce")
bad_scatter = [(s, float(v)) for s, v in zip(sites_scan.site, scat) if pd.notna(v) and v > 100.0]
no_fix = [s for s, v in zip(sites_scan.site, sats) if not pd.notna(v) or v < 1]
judged = int(scat.notna().sum())
print()
if judged == 0:
    print("VERDICT: UNJUDGED -- no site produced a position from its own instrument")
elif bad_scatter or no_fix:
    print("VERDICT: FAIL -- %d of %d sites scatter more than 100 m (%s); %d have no fix with 3 satellites (%s)"
          % (len(bad_scatter), judged, "; ".join("%s %.0f m" % x for x in bad_scatter) or "none",
             len(no_fix), ", ".join(no_fix) or "none"))
else:
    print("VERDICT: PASS -- %d of %d sites judged; the largest own-GPS scatter is %.1f m at %s, the floor is "
          "100 m, and every site has at least %d fixes of 3 satellites or more"
          % (judged, len(sites_scan), scat.max(), sites_scan.site[scat.idxmax()], int(sats.min())))
'''),

("md", r"""## Instruments, dipoles and gains

The serial comes from the LEMI `.INF` `%LEMI424 #NNNN` block where the site has one, and from the release
MTH5 `data_logger.id` where it does not; `serial_source` says which.

The dipole lengths come from four places, in this order, and `dipole_source` always says which:

| order | where from | when |
|---|---|---|
| 1 | the cell `surveys/<SURVEY>/sites.csv` already holds | a re-run: an arm length is a field fact and is never recomputed |
| 2 | the LEMI `.INF` `%L1` and `%L2`, or the release MTH5 `dipole_length` | where the instrument or the release wrote one |
| 3 | `survey.yaml` `dipoles.table` | a CSV of `site, dipole_n_m, dipole_e_m, source`; the expected source is the deployment sheet, transcribed with its date |
| 4 | `survey.yaml` `dipoles.default_m` with `dipoles.reason` | nothing above records one: the cell reads `assume:<metres>` and the reason travels with it |

An EDL raw folder records no arm length at all, so an EDL survey is on rows 3 and 4 and there is no other
path from the field to the table. Apparent resistivity goes as the square of the length -- a 100 m guess
against a true 50 m is a factor of four -- so an assumed length is named as an assumption in the cell below,
in `dipole_source`, and in the provenance of every transfer function the site delivers, until a sheet
replaces it.

Azimuths are the arm bearings as the sheet records them where a sheet exists, and otherwise nominal, 0 deg for
the north arm and 90 deg for the east arm in the instrument frame, which is what the logger wrote and not a
measured bearing; `azimuth_source` says which. The instrument constants are printed once, from survey.yaml,
with the source of each.

**This check fails if any site's sample rate was not read from its own files or differs from the rate
survey.yaml expects, or if either dipole cell is empty (neither a measurement nor a stated assumption).**"""),

("code", '''for inst, k in (sv.cfg.get("instruments") or {}).items():
    print("%s" % inst)
    for key, val in k.items():
        print("   %-14s %s" % (key, val))

# nominal only where no sheet bearing was carried over; a sheet bearing is a record and is never overwritten
nominal = {"azimuth_n_deg": 0.0, "azimuth_e_deg": 90.0,
           "azimuth_source": "instrument frame (nominal); no measured bearing in the release"}
for col, val in nominal.items():
    if col not in sites_scan.columns:
        sites_scan[col] = ""
    sites_scan[col] = [val if SV.is_empty(v) else v for v in sites_scan[col]]

# survey.yaml `dipoles`: the deployment sheet's lengths where it names a CSV, and assume:<default> where it
# does not. A cell the instrument, the release or a previous run already filled is left alone, because an
# arm length is a field fact and a re-run of the discovery cannot measure one
DIP = SV.dipole_table(sv.cfg, sv.folder)
dip_by_site = DIP.set_index("site") if len(DIP) else None
from_table, assumed_sites, still_empty = [], [], []
for i, r in sites_scan.iterrows():
    site = r["site"]
    held = not SV.is_empty(r["dipole_n_m"]) and not str(r["dipole_n_m"]).lower().startswith(SV.ASSUME)
    if held:
        continue                                    # rows 1 and 2: already measured, with its own source
    if dip_by_site is not None and site in dip_by_site.index:
        d = dip_by_site.loc[site]
        if not SV.is_empty(d["dipole_n_m"]) and not SV.is_empty(d["dipole_e_m"]):
            sites_scan.at[i, "dipole_n_m"] = "%g" % float(d["dipole_n_m"])
            sites_scan.at[i, "dipole_e_m"] = "%g" % float(d["dipole_e_m"])
            sites_scan.at[i, "dipole_source"] = str(d["source"] or (sv.cfg.get("dipoles") or {}).get("table"))
            from_table.append(site)
            continue
    value, reason = SV.dipole_default(sv.cfg, r["instrument"])   # <- survey.yaml dipoles.default_m
    if value is None:
        still_empty.append(site)
        continue
    sites_scan.at[i, "dipole_n_m"] = "assume:%g" % value
    sites_scan.at[i, "dipole_e_m"] = "assume:%g" % value
    sites_scan.at[i, "dipole_source"] = reason
    assumed_sites.append(site)
print()
print("dipoles: %d site(s) from survey.yaml dipoles.table (%s), %d assumed from dipoles.default_m, %d "
      "already carried a length, %d carry none"
      % (len(from_table), (sv.cfg.get("dipoles") or {}).get("table") or "no table named", len(assumed_sites),
         len(sites_scan) - len(from_table) - len(assumed_sites) - len(still_empty), len(still_empty)))
if assumed_sites:
    print("   ASSUMED, so every apparent resistivity of these sites is on a length nobody recorded:")
    print("   %s" % " ".join(assumed_sites))
    print("   %s" % str(sites_scan.dipole_source[sites_scan.site == assumed_sites[0]].iloc[0])[:150])
if still_empty:
    print("   no length and no default: %s" % " ".join(still_empty))

print()
print(sites_scan[["site", "instrument", "serial", "serial_source", "firmware", "sample_rate_hz",
                  "dipole_n_m", "dipole_e_m", "dipole_source", "azimuth_n_deg",
                  "azimuth_e_deg"]].to_string())

want = {i: float(b["sample_rate_hz"]) for i, b in sv.cfg["instruments_layout"].items()}
rate_bad = [(r.site, r.instrument, r.sample_rate_hz, want.get(r.instrument))
            for _, r in sites_scan.iterrows()
            if not np.isfinite(pd.to_numeric(r.sample_rate_hz, errors="coerce"))
            or float(r.sample_rate_hz) != want.get(r.instrument)]
dip_bad = [r.site for _, r in sites_scan.iterrows()
           if str(r.dipole_n_m).strip() in ("", "nan") or str(r.dipole_e_m).strip() in ("", "nan")]
n_assume = int(sum(str(v).startswith("assume:") for v in sites_scan.dipole_n_m))
print()
if not len(sites_scan):
    print("VERDICT: UNJUDGED -- no site was scanned")
elif rate_bad or dip_bad:
    print("VERDICT: FAIL -- %d sites read a rate that is not the expected one (%s); %d sites have an empty "
          "dipole cell (%s)" % (len(rate_bad), "; ".join("%s %s %s vs %s" % x for x in rate_bad) or "none",
                                len(dip_bad), ", ".join(dip_bad) or "none"))
else:
    print("VERDICT: PASS -- %d sites read the rate survey.yaml expects (%s); every dipole cell carries a value "
          "or a stated assumption (%d sites read assume:)"
          % (len(sites_scan), ", ".join("%s %g Hz" % (i, v) for i, v in want.items()), n_assume))
'''),

("md", r"""## Declination

The IGRF declination at each site at the midpoint of its record, in degrees east of true north
(`auslamp_proc.geo.declination`, which wraps ppigrf; ppigrf takes longitude first and height in kilometres and
returns east, north and up, so X = north, Y = east and Z = -up). It is recorded and not applied: a transfer
function is estimated and served in geomagnetic north, with each site's horizontal magnetics rotated so the
mean Hy is zero, and turning them by IGRF as well would rotate them twice. The value travels with every
delivered file as `declination_deg` and its negative as `to_geographic_north_deg`, the angle a tensor in
geomagnetic north is turned by to reach true geographic north.

The check has two limbs. The first is scored on every survey: a declination is computed from the site's own
position, so a site whose position is missing or not finite gets none, and that is the fault the first limb
catches. The second is scored only where the survey ships a release MTH5 to read a declination out of; the
release computes its value from a different reference field (AGRF), so the two are independent computations of
the same quantity and 2.0 deg is the agreement two reference fields hold to over an Australian survey's epoch.
Where no release value exists the cell below prints `no release value to compare` and the second limb scores
nothing; that is a reading of what the survey ships, and the first limb still scores every site.

**This check fails if any site has no declination, or, where the survey ships a release MTH5, if the IGRF
value differs by more than 2.0 deg from the declination that MTH5 carries.** The verdict names both limbs and
says how many sites each scored."""),

("code", '''mid = t0 + (t1 - t0) / 2
sites_scan["declination_deg"] = [
    round(geo.declination(r.lat, r.lon, r.elev_m if np.isfinite(r.elev_m) else 0.0, m.to_pydatetime()), 3)
    if np.isfinite(r.lat) else np.nan
    for r, m in zip(sites_scan.itertuples(), mid)]

rel_dec = []
for _, r in sites_scan.iterrows():
    folder = (sv.cfg.get("release_mth5") or {}).get(r.instrument, "")
    v = np.nan
    if folder:
        p = Path(folder) / ("%s.h5" % r.site)
        if p.exists():
            import h5py
            with h5py.File(p, "r") as h:
                for sname in h["Experiment/Surveys"]:
                    stg = h["Experiment/Surveys"][sname]["Stations"]
                    for st in stg:
                        v = float(stg[st].attrs.get("location.declination.value", np.nan))
                        break
                    break
    rel_dec.append(v)
sites_scan["release_declination_deg"] = np.round(rel_dec, 3)
d = (sites_scan.declination_deg - sites_scan.release_declination_deg).abs()
print(sites_scan[["site", "lat", "lon", "declination_deg", "release_declination_deg"]].assign(
    difference=np.round(d, 3)).to_string())

missing = list(sites_scan.site[~np.isfinite(sites_scan.declination_deg)])
compared = int(d.notna().sum())
over = [(s, float(v)) for s, v in zip(sites_scan.site, d) if pd.notna(v) and v > 2.0]
limb2 = ("the largest difference from the release's AGRF value over %d sites is %.3f deg at %s, under the "
         "2.0 deg bound" % (compared, d.max(), sites_scan.site[d.idxmax()]) if compared
         else "no release value to compare against, so the second limb scored no site")
print()
print("release declination compared at %d of %d sites" % (compared, len(sites_scan)))
if len(sites_scan) == 0:
    print("VERDICT: UNJUDGED -- no site was scanned, so no declination was computed")
elif missing or over:
    print("VERDICT: FAIL -- %d of %d sites have no declination (%s); %d of %d compared sites differ from the "
          "release by more than 2.0 deg (%s)"
          % (len(missing), len(sites_scan), ", ".join(missing) or "none", len(over), compared,
             "; ".join("%s %.2f deg" % x for x in over) or "none"))
else:
    print("VERDICT: PASS -- all %d sites carry a declination, of %.2f to %.2f deg east; %s"
          % (len(sites_scan), sites_scan.declination_deg.min(), sites_scan.declination_deg.max(), limb2))
'''),

("md", r"""## The site table

`sites_discovered.csv` is written to the work root on every run. `surveys/<SURVEY>/sites.csv` is created once
and then left alone: on a later run the cells that differ are printed and the file is not touched, because a
cell a person has set to `assume:<value>` or left at `decide` is a standing instruction and not a value to be
recomputed (`auslamp_proc.survey.write_table`). `decisions.csv` is created with `decide` in every cell the
same way.

"Created once" means created where the file is absent OR holds a header and no row.
`surveys/_template/` ships both tables as header-only stubs, so that the column lists are visible in the
template, and a copy of the template is therefore a survey in which both files exist holding nothing. A stub
that was preserved would leave a new survey with empty tables for ever, so a header with no row is read as a
file waiting to be written (`auslamp_proc.survey.is_stub`).

The comparison below is a regression against site tables built by other code, named in survey.yaml
`regression`: `regression.sites` carries the positions and `regression.times` the spans, which in some
campaigns are two different files. It is a regression and not a source: no value is read from it into
sites.csv.

A regression table's span means what its builder meant by it: the first and last sample on disk, or the hours a
processing run kept. `regression.times_kind` declares which, and the criterion follows.

| `times_kind` | what the table's start and end are | what the spans are tested for |
|---|---|---|
| `raw_span` | the first and the last sample on disk | equality, within `time_tolerance_h` |
| `processing_window` | the hours a processing run kept | containment: the raw span holds that window |

**This check fails if any site's position differs from the regression table's by more than 20 m, if a site is
in one table and not the other, or if the spans disagree: under `raw_span` when a start or an end differs by
more than 1 hour, and under `processing_window` when the raw span does not contain the window to within 1
hour, that is when the raw start falls more than 1 hour after the window's start or the raw end more than 1
hour before the window's end.** The tolerances are survey.yaml `regression.position_tolerance_m` and
`regression.time_tolerance_h`, and every site that fails is printed with both values.

A raw span that does not hold what a run processed means the run read hours this discovery cannot find, which
is a fault in one of the two, and the check reports it.

The AusLAMP Queensland table is a set of processing windows and its edges are the campaign's own trims: the
tails of Q49, Q77N and Q87 cut where the record died, 14.6, 13.1 and 32.6 days of record each. Those three
sites hold more raw record than the campaign processed, which is containment holding."""),

("code", '''site_table = sites_scan.reindex(columns=SV.SITES_COLUMNS).copy()
site_table["observatory"] = (sv.cfg.get("observatory") or {}).get("code", "")
for c in ("start_utc", "end_utc"):
    site_table[c] = sites_scan[c].astype(str)
disc_path = OUT / "sites_discovered.csv"
site_table.to_csv(disc_path, index=False)
WRITTEN.append(disc_path)
print("wrote %s (%d rows, %d columns)" % (disc_path, len(site_table), len(site_table.columns)))

# a header with no row is a stub and not a table: surveys/_template ships both files that way so that the
# column lists are visible in it, so a copied template is a survey whose tables have yet to be written
if SV.is_stub(sv.folder / "sites.csv", SV.SITES_COLUMNS):
    print(SV.write_table(sv.folder / "sites.csv", site_table, SV.SITES_COLUMNS))
else:
    delta = SV.diff_tables(SV.load_survey(SURVEY).sites, site_table)
    print("surveys/%s/sites.csv exists and is left alone; %d cells differ from this run" % (SURVEY, len(delta)))
    if len(delta):
        print(delta.head(40).to_string())
if SV.is_stub(sv.folder / "decisions.csv", SV.DECISIONS_COLUMNS):
    SV.blank_decisions(list(site_table.site)).to_csv(sv.folder / "decisions.csv", index=False)
    print("created decisions.csv with 'decide' in every cell")
else:
    dec = SV.load_survey(SURVEY).decisions
    print("decisions.csv holds %d rows; %d of %d cells are still 'decide', %d sites carry a sign already "
          "decided" % (len(dec), int((dec == "decide").sum().sum()), dec.size,
                       int((dec.sign_source != "decide").sum())))

# every column the package reads, every cell still open and every stated assumption, one line each; an empty
# list is a survey with no input left to decide
open_lines = SV.validate(SV.load_survey(SURVEY))
print()
print("%d open input(s) across the two tables" % len(open_lines))
for line in open_lines[:20]:
    print("   %s" % line)
if len(open_lines) > 20:
    print("   ... and %d more" % (len(open_lines) - 20))

reg = sv.cfg.get("regression") or {}
if not reg.get("sites"):
    print()
    print("VERDICT: UNJUDGED -- survey.yaml names no regression table")
else:
    ref = SV.read_reference(reg, path_key="sites", columns_key="columns", select_key="select")
    times = SV.read_reference(reg, path_key="times", columns_key="time_columns", select_key="time_select")
    if len(times):                                 # <- regression.times; the spans live in a second table
        ref = ref.merge(times, on="site", how="outer")
    j = site_table.merge(ref, on="site", how="outer", suffixes=("", "_ref"), indicator=True)
    only = j[j._merge != "both"][["site", "_merge"]]
    j = j[j._merge == "both"].copy()
    j["separation_m"] = [geo.distance_km((a, b), (c, d)) * 1000
                         if all(np.isfinite([a, b, c, d])) else np.nan
                         for a, b, c, d in zip(pd.to_numeric(j.lat, errors="coerce"),
                                               pd.to_numeric(j.lon, errors="coerce"),
                                               pd.to_numeric(j.lat_ref, errors="coerce"),
                                               pd.to_numeric(j.lon_ref, errors="coerce"))]
    has_times = "start_utc_ref" in j.columns and "end_utc_ref" in j.columns
    for c in ("start", "end"):
        j["%s_diff_h" % c] = ((pd.to_datetime(j["%s_utc" % c], format="ISO8601", utc=True)
                               - pd.to_datetime(j["%s_utc_ref" % c], format="ISO8601", utc=True)
                               ).dt.total_seconds() / 3600) if has_times else np.nan
    tol_m = float(reg.get("position_tolerance_m", 20))
    tol_h = float(reg.get("time_tolerance_h", 1))
    kind = reg.get("times_kind", "raw_span")       # <- raw_span (equality) | processing_window (containment)
    if kind == "processing_window":
        # the raw span must hold the window: start no later, end no earlier, each within tol_h
        bad = (j.start_diff_h > tol_h) | (j.end_diff_h < -tol_h)
        rule = ("containment within %.0f h: the raw span holds the window the run kept" % tol_h)
    else:
        bad = (j.start_diff_h.abs() > tol_h) | (j.end_diff_h.abs() > tol_h)
        rule = "equality within %.0f h on both the start and the end" % tol_h
    pos_fail = j[j.separation_m > tol_m]
    time_fail = j[bad.fillna(False)]
    print()
    print("regression against %s%s (%s): %d sites in both tables, %d in only one"
          % (Path(reg["sites"]).name,
             (" and %s" % Path(reg["times"]).name) if reg.get("times") else "",
             reg.get("dated", "undated"), len(j), len(only)))
    print("   spans declared %s, so the criterion is %s" % (kind, rule))
    for inst in sorted(j.instrument.unique()):
        s = j[j.instrument == inst]
        print("   position, %-14s %3d sites, median %5.1f m, largest %6.1f m at %s"
              % (inst, len(s), s.separation_m.median(), s.separation_m.max(),
                 s.site[s.separation_m.idxmax()]))
    for c in ("start", "end"):
        d = j["%s_diff_h" % c].abs()
        print("   %-9s largest %.4f h at %s" % (c + ":", d.max(), j.site[d.idxmax()])
              if d.notna().any() else "   %-9s no span in the regression table" % (c + ":"))
    print()
    print("difference per site (ours minus the regression table; under processing_window a negative start and "
          "a positive end are raw record outside the window)")
    print(j[["site", "separation_m", "start_diff_h", "end_diff_h"]].round(3).to_string(index=False))
    if len(pos_fail):
        print()
        print("sites over %.0f m:" % tol_m)
        print(pos_fail[["site", "lat", "lon", "lat_ref", "lon_ref", "separation_m"]].to_string())
    if len(time_fail):
        print()
        print("sites failing the span criterion:")
        print(time_fail[["site", "start_utc", "start_utc_ref", "start_diff_h",
                         "end_utc", "end_utc_ref", "end_diff_h"]].to_string())
    print()
    if len(j) == 0:
        print("VERDICT: UNJUDGED -- no site is in both tables")
    elif len(pos_fail) or len(time_fail) or len(only):
        print("VERDICT: FAIL -- %d of %d sites differ in position by more than %.0f m (%s); %d fail the span "
              "criterion for a %s table, %s (%s); %d sites are in only one table (%s)"
              % (len(pos_fail), len(j), tol_m, ", ".join(pos_fail.site) or "none",
                 len(time_fail), kind, rule, ", ".join(time_fail.site) or "none",
                 len(only), ", ".join(only.site) or "none"))
    elif kind == "processing_window":
        # slack is how much raw record sits outside the window: negative means inside it, within tol_h
        print("VERDICT: PASS -- %d sites agree with %s to within %.0f m (largest %.1f m) and every raw span "
              "contains the window %s holds, the least slack being %.4f h at the start and %.4f h at the end "
              "against a %.0f h tolerance"
              % (len(j), Path(reg["sites"]).name, tol_m, j.separation_m.max(),
                 Path(reg["times"]).name if reg.get("times") else Path(reg["sites"]).name,
                 -j.start_diff_h.max() if has_times else 0.0,
                 j.end_diff_h.min() if has_times else 0.0, tol_h))
    else:
        print("VERDICT: PASS -- %d sites agree with %s to within %.0f m and %.0f h (largest %.1f m, %.4f h)"
              % (len(j), Path(reg["sites"]).name, tol_m, tol_h, j.separation_m.max(),
                 max(j.start_diff_h.abs().max(), j.end_diff_h.abs().max()) if has_times else 0.0))
'''),

("md", r"""## Map

Sites as points labelled with the digits of their names, one colour per instrument, observatories as black
triangles. The coastline is `auslamp_proc/data/coastline_au.npz`, the Natural Earth 1:50 m line clipped to
lon 108-160 deg and lat -48 to -8 deg and stored as two float32 arrays (17.2 KB, 2,962 vertices) so that the
map needs no GIS package; `tools/build_coastline.py` built it once. The aspect is corrected by cos(lat), so a
degree of longitude is drawn at its real length at this latitude.

The frame is the sites' own bounding box padded by 10 per cent of its span or 0.5 deg, whichever is larger
(`auslamp_proc.figures.survey.site_frame`), so the survey fills the picture. An observatory inside that frame
is drawn; one outside it is named in a one-line box in the lower left with its bearing in degrees east of
north and its distance in km from the survey centroid, because a 1,000 km observatory drawn on the map sets
the scale and shrinks a 3 deg survey into a corner.

Look for a site sitting away from the grid: at this survey's spacing a point off the half-degree lattice is a
position to re-read.

**This check fails if fewer points are drawn than there are sites, or if the coastline file is absent.**"""),

("code", '''obs_here = [(c, v[0], v[1], v[2]) for c, v in geo.OBSERVATORIES.items()]
lat0, lon0 = sites_scan.lat.mean(), sites_scan.lon.mean()
near = [o for o in obs_here if geo.distance_km((lat0, lon0), (o[2], o[3])) < 1500]   # <- 1500 km, offered to the map
title = "%s: %d sites, %s" % (sv.cfg["name"], len(sites_scan),
                              " and ".join(sorted(set(sites_scan.instrument))))
map_png = OUT / "map.png"
fig = FIG.map(sites_scan, near, map_png, title=title)
WRITTEN.append(map_png)

frame = FIG.site_frame(sites_scan.lat.values, sites_scan.lon.values)
inside = [o for o in near if frame[0] <= o[3] <= frame[1] and frame[2] <= o[2] <= frame[3]]
in_codes = {o[0] for o in inside}
print("frame lon %.2f .. %.2f, lat %.2f .. %.2f" % frame)
for c, name, olat, olon in near:
    b = FIG.bearing_deg((lat0, lon0), (olat, olon))
    print("   %-4s %-32s %8.1f km  bearing %03.0f deg (%s)  %s"
          % (c, name, geo.distance_km((lat0, lon0), (olat, olon)), b, FIG.compass_point(b),
             "drawn" if c in in_codes else "named in the corner box"))

lon_c, lat_c = FIG.load_coastline()
drawn = int(np.isfinite(pd.to_numeric(sites_scan.lat, errors="coerce")).sum())
print()
if len(sites_scan) == 0:
    print("VERDICT: UNJUDGED -- no site to draw")
elif drawn < len(sites_scan) or lon_c is None:
    print("VERDICT: FAIL -- %d of %d sites have a finite position and were drawn; coastline %s"
          % (drawn, len(sites_scan), "absent" if lon_c is None else "present"))
else:
    print("VERDICT: PASS -- all %d sites drawn inside a frame of %.2f x %.2f deg, %d of the %d observatories "
          "within 1500 km fall in it and are drawn and %d are named in the corner box, coastline %d vertices"
          % (drawn, frame[1] - frame[0], frame[3] - frame[2], len(inside), len(near),
             len(near) - len(inside), len(lon_c)))
'''),

("md", r"""## The deployment register

Sorted by start, one bar per site over the count of sites recording each day, with two tables under it.

The groups table is the candidate pool: the connected components of the graph whose edges join two sites
overlapping by at least `MIN_OVERLAP_DAYS` (`auslamp_proc.register.groups`). `common_days` is the window every
member shares and is empty where the component is held together by a chain: A and B overlapping and B and C
overlapping does not make A and C overlap. A remote site and the fleet stack members for a target are chosen
from its component.

The core table is the concurrent core: for a given day, the sites whose record covers the whole of the 14 days
centred on it, that is start <= peak - 7 d and end >= peak + 7 d (`auslamp_proc.register.core_group`). The
peaks are the local maxima of the sites-active-per-day curve.

The check has two limbs, and each is scored where the survey carries what it needs.

Limb A applies where survey.yaml names `regression.waves`, a core-group table built by other code. **It fails
if the core groups, computed with the rule above at the peaks and tiers that table names and on that table's
own spans, do not reproduce it row for row in n_core, common_start, common_end, days and core.** Timestamps
are compared to the second, the resolution the shipped table is written to. The same rule is run a second time
on the spans this workbook discovered, and those differences are printed below the verdict as a reading, not
as part of the criterion.

Limb B applies where `decisions.csv` already carries a chosen `remote_site`. A remote site can only reference
a target over the time the two were both recording, and the campaign that chose these remotes chose them on
overlap and coherence, neither of which this workbook reads; the register is an independent computation of the
overlap half. **Limb B fails if any site's chosen remote is not in the same overlap group as the site at
`MIN_OVERLAP_DAYS`, or if any site with a chosen remote has no overlap with it at all.** Every pair is printed
with its overlap in days, and every failing pair is named. A survey whose remotes are all still `decide` has
no pair to score and limb B says so.

The verdict is scored on whichever limbs apply, and reports UNJUDGED only where neither does."""),

("code", '''sp = register.spans(site_table)
act = register.active_per_day(sp)
grp = register.groups(sp, MIN_OVERLAP_DAYS)                # <- MIN_OVERLAP_DAYS; raise it to cut the pool
peaks = register.peaks_from_active(act)
core = register.core_groups(sp, peaks=peaks, by="instrument")

reg_png = OUT / "deployment_register.png"
fig = FIG.register(sp, grp, act, reg_png, title="%s: the deployment register" % sv.cfg["name"])
WRITTEN.append(reg_png)

print("groups at MIN_OVERLAP_DAYS = %d d" % MIN_OVERLAP_DAYS)
print(grp.to_string())
print()
print("concurrent core at the local maxima of the active-sites curve (14 d centred on each peak)")
print(core.to_string())
print()
big = grp.loc[grp.n.idxmax()]
members = big.members.split()
ov = register.overlap_matrix(sp)
print("overlap in days within the largest group, %s (%d sites)" % (big.group, len(members)))
print(ov.loc[members, members].to_string())

for name, frame in (("deployment_register.csv", sp), ("deployment_groups.csv", grp),
                    ("deployment_core_groups.csv", core), ("overlap_matrix.csv", ov)):
    p = OUT / name
    frame.to_csv(p, index=(name == "overlap_matrix.csv"))
    WRITTEN.append(p)
    print("wrote %s" % p.name)

# limb B: the remote each site was given must sit in the same overlap group and actually overlap it
group_of = {m: g.group for _, g in grp.iterrows() for m in g.members.split()}
dec_now = SV.load_survey(SURVEY).decisions
pairs, remote_bad = [], []
if "remote_site" in dec_now.columns:
    for _, r in dec_now.iterrows():
        rem = str(r["remote_site"]).strip()
        if not rem or rem.lower() == SV.DECIDE or r["site"] not in ov.index:
            continue
        days = float(ov.loc[r["site"], rem]) if rem in ov.columns else float("nan")
        same = group_of.get(r["site"]) is not None and group_of.get(r["site"]) == group_of.get(rem)
        pairs.append(dict(site=r["site"], remote=rem, overlap_days=round(days, 2),
                          group=group_of.get(r["site"], ""), remote_group=group_of.get(rem, ""),
                          same_group=same))
        if not same or not (days > 0):
            remote_bad.append("%s -> %s, %.2f d overlap, groups %s and %s"
                              % (r["site"], rem, days, group_of.get(r["site"], "-"),
                                 group_of.get(rem, "-")))
print()
if pairs:
    print("the remote each site was given, against the register (limb B)")
    print(pd.DataFrame(pairs).to_string(index=False))
else:
    print("no site carries a chosen remote_site, so limb B scores no pair")

reg = sv.cfg.get("regression") or {}
if not reg.get("waves"):
    print()
    print("limb A: survey.yaml names no core-group table to reproduce, so it scores no row")
    if not pairs:
        print("VERDICT: UNJUDGED -- neither limb has anything to score")
    elif remote_bad:
        print("VERDICT: FAIL -- %d of %d chosen remotes are not in the site's own overlap group at "
              "MIN_OVERLAP_DAYS = %d d or do not overlap it: %s"
              % (len(remote_bad), len(pairs), MIN_OVERLAP_DAYS, "; ".join(remote_bad)))
    else:
        print("VERDICT: PASS -- all %d chosen remotes sit in the site's own overlap group at "
              "MIN_OVERLAP_DAYS = %d d and overlap it, the shortest overlap being %.2f d"
              % (len(pairs), MIN_OVERLAP_DAYS, min(p["overlap_days"] for p in pairs)))
else:
    shipped = pd.read_csv(reg["waves"])
    half = int(reg.get("waves_half_days", 7))
    tier_map = reg.get("waves_tier_map") or {}
    m = pd.read_csv(reg["sites"])
    cm = reg["columns"]
    ref_sp = register.spans(pd.DataFrame({"site": m[cm["site"]], "start_utc": m[cm["start_utc"]],
                                          "end_utc": m[cm["end_utc"]], "instrument": m[cm["tier"]]}))

    def compare(spans_frame, tier_as):
        rows = []
        for _, r in shipped.iterrows():
            sub = spans_frame[spans_frame.instrument == tier_as(r.tier)]
            got = register.core_group(r.peak, sub, half, order=str(r.core).split())
            for c in ("n_core", "days", "core"):
                if str(got[c]) != str(r[c]):
                    rows.append((r.peak, c, str(r[c]), str(got[c])))
            for c in ("common_start", "common_end"):
                a, b = pd.Timestamp(r[c]) if str(r[c]) else pd.NaT, pd.Timestamp(got[c]) if got[c] else pd.NaT
                same = (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b)
                                                       and abs((a - b).total_seconds()) <= 1.0)
                if not same:
                    rows.append((r.peak, c, str(r[c]), str(got[c])))
        return rows

    on_ref = compare(ref_sp, lambda t: t)
    on_ours = compare(sp, lambda t: tier_map.get(t, t))
    print()
    print("on the regression table's own spans: %d of %d rows reproduce"
          % (len(shipped) - len({x[0] for x in on_ref}), len(shipped)))
    for x in on_ref:
        print("   %s %-13s %s -> %s" % x)
    print()
    print("on the spans this workbook discovered: %d of %d rows reproduce (a reading, not the criterion)"
          % (len(shipped) - len({x[0] for x in on_ours}), len(shipped)))
    for x in on_ours:
        print("   %s %-13s %s -> %s" % (x[0], x[1], x[2][:64], x[3][:64]))
    limb_b = ("all %d chosen remotes sit in the site's own overlap group and overlap it, the shortest being "
              "%.2f d" % (len(pairs), min(p["overlap_days"] for p in pairs)) if pairs
              else "no site carries a chosen remote_site, so limb B scored no pair")
    print()
    if len(shipped) == 0 and not pairs:
        print("VERDICT: UNJUDGED -- neither limb has anything to score")
    elif on_ref or remote_bad:
        print("VERDICT: FAIL -- %d of %d rows of %s do not reproduce on its own spans, %d cells differ; %d of "
              "%d chosen remotes fail the overlap rule at MIN_OVERLAP_DAYS = %d d (%s)"
              % (len({x[0] for x in on_ref}), len(shipped), Path(reg["waves"]).name, len(on_ref),
                 len(remote_bad), len(pairs), MIN_OVERLAP_DAYS, "; ".join(remote_bad) or "none"))
    else:
        print("VERDICT: PASS -- all %d rows of %s reproduce on its own spans at half_days = %d; %s"
              % (len(shipped), Path(reg["waves"]).name, half, limb_b))
'''),

("md", r"""## Observatory

An INTERMAGNET one-second observatory is one of the reference kinds, so its record has to cover the survey's
span before a later workbook can offer it. The distances below are to the survey's centroid; the positions are
the `Geodetic Latitude` and `Geodetic Longitude` lines of the observatories' own IAGA-2002 headers, read from
the GIN on 2026-09-16 (`auslamp_proc.geo.OBSERVATORIES`).

Coverage is read from the parquet row-group statistics in the file footer and not from the data
(`auslamp_proc.observatory.coverage`): a year file is 380 MB, and the question is only whether a day is there.
A day is counted present when it holds at least half of its 86,400 samples.

The coverage of every observatory over this survey's span is read, nearest first, so the choice is made on
what the archive holds and not on distance alone. Where `survey.yaml` `observatory.code` still reads
`decide` the cell names the nearest one with full coverage and stops there: `decide` is not an IAGA code,
and asking the archive for an observatory called DECIDE finds nothing and reports it as 44 absent days,
which reads as a data fault rather than as an unfilled field. Write the code the cell names into
`surveys/<SURVEY>/survey.yaml` and run this workbook again.

`observatory.archive` is a path on this machine, like `raw_root` and `work_root`. Where the archive holds
none of the span at any observatory, set `FETCH` True and run this cell again: it downloads the days from
the INTERMAGNET GIN, one request per day.

**This check fails if `survey.yaml` names no observatory code, or if any day of the survey's span is absent
from the archive for the code it names.**"""),

("code", '''code = str((sv.cfg.get("observatory") or {}).get("code", "")).strip()
archive = (sv.cfg.get("observatory") or {}).get("archive", "")
lat0, lon0 = sites_scan.lat.mean(), sites_scan.lon.mean()
start, end = sp.start.min().date(), sp.end.max().date()
print("survey centroid %.3f, %.3f; the span is %s .. %s" % (lat0, lon0, start, end))
print("archive      %s  (a path on THIS machine)" % archive)

# every observatory's coverage over the span, nearest first: a footer read per observatory-year, so the
# choice is made on what the archive holds and not on distance alone
OPEN_CODE = code.lower() in ("", "decide", "none", "nan")
cov_rows, cov_of = [], {}
for c, name, km in geo.observatory_distances(lat0, lon0):
    cv = observatory.coverage(c, start, end, archive)
    cov_of[c] = cv
    cov_rows.append(dict(code=c, observatory=name, km=round(km, 1), days=len(cv),
                         present=int(cv.present.sum()), absent=int((~cv.present).sum()),
                         chosen=("survey.yaml observatory.code" if c == code.upper() else "")))
cov_table = pd.DataFrame(cov_rows)
print()
print("the one-second coverage of every observatory over this survey's span, nearest first")
print(cov_table.to_string(index=False))
full = cov_table[cov_table.absent == 0]
nearest_full = str(full.code.iloc[0]) if len(full) else ""

if OPEN_CODE:
    # `decide` is not an IAGA code. The archive would be asked for an observatory called DECIDE and would
    # report every day absent, which reads as a data fault; the field is unfilled and this says so
    print()
    print("survey.yaml observatory.code reads %r, which is not an IAGA code." % (code or "<empty>"))
    if nearest_full:
        print("Write   code: %s   into surveys/%s/survey.yaml and run this workbook again: %s is the "
              "nearest observatory whose archive covers the whole span, %.1f km from the centroid."
              % (nearest_full, SURVEY, nearest_full, float(full.km.iloc[0])))
    else:
        print("No observatory covers the whole span in this archive. Set FETCH = True and run this cell "
              "again to download the days one of them lacks, or write   code: none   if this survey is to "
              "be processed without an observatory reference.")
    print()
    print("VERDICT: FAIL -- survey.yaml names no observatory: observatory.code reads %r and no code was "
          "read from the archive under it. The choice is %s; %d of the %d observatories cover the whole "
          "span (%s)"
          % (code or "<empty>", nearest_full or "none of them, so FETCH or code: none", len(full),
             len(cov_table), ", ".join(full.code) or "none"))
else:
    cov = cov_of.get(code.upper())
    if cov is None:
        cov = observatory.coverage(code, start, end, archive)
    cov_path = OUT / "observatory_coverage.csv"
    cov.to_csv(cov_path, index=False)
    WRITTEN.append(cov_path)
    print()
    print("%s coverage %s .. %s: %d days, %d present, %d absent"
          % (code, start, end, len(cov), int(cov.present.sum()), int((~cov.present).sum())))
    month = cov.assign(month=pd.to_datetime(cov.day).dt.to_period("M")).groupby("month").agg(
        days=("present", "size"), present=("present", "sum"))
    month["absent"] = month.days - month.present
    print(month.to_string())

    missing = observatory.missing_days(code, start, end, archive)
    plan = observatory.fetch_missing(code, start, end, archive, dry_run=not FETCH)  # <- FETCH downloads them
    print()
    print("days FETCH would download: %d%s" % (len(plan), (" (first %s)" % plan[0][0]) if plan else ""))
    for day, url in plan[:5]:
        print("   %s  %s" % (day, url))
    print()
    if len(cov) == 0:
        print("VERDICT: UNJUDGED -- the survey span produced no days to check")
    elif missing:
        print("VERDICT: FAIL -- %d of %d days of %s .. %s are absent from the %s archive: %s%s"
              % (len(missing), len(cov), start, end, code,
                 ", ".join(str(d) for d in missing[:10]), " ..." if len(missing) > 10 else ""))
    else:
        print("VERDICT: PASS -- all %d days of %s .. %s are present in the %s archive, %.1f km from the "
              "survey centroid; the thinnest day holds %d of 86400 samples"
              % (len(cov), start, end, code, geo.distance_km((lat0, lon0), code), int(cov.n_samples.min())))
'''),

("md", r"""## What was written"""),

("code", '''rows = []
for p in sorted(set(WRITTEN)) + sorted((sv.folder).glob("*.csv")) + sorted((sv.folder).glob("*.yaml")):
    p = Path(p)
    if p.exists():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print(files.to_string(index=False))
print()
print("%d files, %.1f KB" % (len(files), files.kb.sum()))
'''),
]


