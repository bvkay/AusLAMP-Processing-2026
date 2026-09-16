r"""Write the workbooks. Each is a Python list of ("md", text) and ("code", source) cells held here.

    python workbooks/make_workbooks.py            all
    python workbooks/make_workbooks.py 01         some, by prefix

Then execute them, which is what puts the outputs in:

    python workbooks/run_workbooks.py 01

A notebook is never edited in place; the generator is edited and re-run.

Prose rules the cells follow: declarative, present tense, first person plural only where a choice was made;
every number with its unit and its band; every rule with its value and the function that applies it; a
markdown cell says what to look for in the cell below it; every check states its failure criterion in bold
before the cell runs and the cell prints a verdict.

@author: ben kay (ben@auscope.org.au)
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
KERNEL = "auslamp-processing-2026"


# ===================================================================== 01 the survey

WB01_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
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
import matplotlib.pyplot as plt

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
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported FAILED and is
not revised afterwards.

The reference kinds the later workbooks build are named here once and used as words from then on:

| word | what it is | code key |
|---|---|---|
| single station | the site's own H and E | `single` |
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
MTH5 `data_logger.id` where it does not; `serial_source` says which. The dipole lengths come from the `.INF`
`%L1` and `%L2` where they exist and from the release MTH5 `dipole_length` otherwise. Where a survey records
no dipole at all the cell reads `assume:<value>` with its reason, and that assumption is carried into every
product's provenance until a sheet replaces it. An EDL survey holds neither: its dipoles and serials are on
the deployment sheet, they were merged into `surveys/<SURVEY>/sites.csv` with their source, and the cell above
carried them over.

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
returns east, north and up, so X = north, Y = east and Z = -up). It is recorded and not applied: products are
processed and served in geomagnetic north, with each site's horizontal magnetics rotated so the mean Hy is
zero, and turning them by IGRF as well would rotate them twice. The release's tensors, which are in geographic
north, are turned by this value when they are compared in a later workbook.

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
recomputed (`auslamp_proc.survey.write_table`). `decisions.csv` is created with `decide` in every cell if it
does not exist.

The comparison below is a regression against site tables built by other code, named in survey.yaml
`regression`: `regression.sites` carries the positions and `regression.times` the spans, which in some
campaigns are two different files. It is a regression and not a source: no value is read from it into
sites.csv.

A regression table says what its own builder meant by a span, and the two meanings are different quantities.
`regression.times_kind` declares which one the table holds, and the criterion follows from that declaration.

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

Containment is the whole criterion for a processing window and not a weakened equality. A run states which
hours it used; what the raw tree has to answer for is that those hours are in it. A raw span that does not
hold what a run processed means the run read hours this discovery cannot find, which is a fault in one of the
two, and the check reports it.

The AusLAMP Queensland tables are processing windows and their edges are the campaign's own trims: six
transit hours cut from the head of each Phase 2 record, the tails of Phase 1's Q49, Q77N and Q87 cut where
the record died (14.6, 13.1 and 32.6 days), and Phase 3's Q31 and Q55 opened after a leading outage (2.0 and
11.8 days). Those sites hold more raw record than the campaign processed, which is containment holding."""),

("code", '''site_table = sites_scan.reindex(columns=SV.SITES_COLUMNS).copy()
site_table["observatory"] = (sv.cfg.get("observatory") or {}).get("code", "")
for c in ("start_utc", "end_utc"):
    site_table[c] = sites_scan[c].astype(str)
disc_path = OUT / "sites_discovered.csv"
site_table.to_csv(disc_path, index=False)
WRITTEN.append(disc_path)
print("wrote %s (%d rows, %d columns)" % (disc_path, len(site_table), len(site_table.columns)))

if not (sv.folder / "sites.csv").exists():
    print(SV.write_table(sv.folder / "sites.csv", site_table, SV.SITES_COLUMNS))
else:
    delta = SV.diff_tables(SV.load_survey(SURVEY).sites, site_table)
    print("surveys/%s/sites.csv exists and is left alone; %d cells differ from this run" % (SURVEY, len(delta)))
    if len(delta):
        print(delta.head(40).to_string())
if not (sv.folder / "decisions.csv").exists():
    SV.blank_decisions(list(site_table.site)).to_csv(sv.folder / "decisions.csv", index=False)
    print("created decisions.csv with 'decide' in every cell")
else:
    dec = SV.load_survey(SURVEY).decisions
    print("decisions.csv holds %d rows; %d of %d cells are still 'decide', %d sites carry a sign already "
          "decided" % (len(dec), int((dec == "decide").sum().sum()), dec.size,
                       int((dec.sign_source != "decide").sum())))

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

With `FETCH` False the days the archive lacks are listed and nothing is downloaded.

**This check fails if any day of the survey's span is absent from the archive for the chosen code.**"""),

("code", '''code = (sv.cfg.get("observatory") or {}).get("code", "")
archive = (sv.cfg.get("observatory") or {}).get("archive", "")
lat0, lon0 = sites_scan.lat.mean(), sites_scan.lon.mean()
print("survey centroid %.3f, %.3f" % (lat0, lon0))
for c, name, km in geo.observatory_distances(lat0, lon0):
    print("   %-4s %-32s %8.1f km%s" % (c, name, km, "   <- survey.yaml observatory.code" if c == code else ""))

start, end = sp.start.min().date(), sp.end.max().date()
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
plan = observatory.fetch_missing(code, start, end, archive, dry_run=not FETCH)   # <- FETCH True downloads them
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
    print("VERDICT: PASS -- all %d days of %s .. %s are present in the %s archive, %.1f km from the survey "
          "centroid; the thinnest day holds %d of 86400 samples"
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


# ===================================================================== 02 the records

WB02_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
SITES = "all"                 # "all" | "largest" (the register's largest group) | a group name | ["Q49", "Q50"]
MAX_SITES = 0                 # 0 = every chosen site; a cap keeps an example short and names what it kept
WORK_ROOT = None              # None = survey.yaml work_root; every cache and figure lands under it
RATES = [1, 10]               # the caches built; 10 is EDL only, and a student short of disk sets [1]
WIN_MIN, STEP_MIN = 60, 30    # the base window and step of figures 02, 03 and 05, in minutes
PMAX = 20000                  # the longest period the level ladder reaches, in s
REBUILD = False               # True rebuilds every cache from the raw files, which is the slow path
SHOW = "first"                # the site whose five figures are shown below; a site name, or "none"
'''

WB02_SETUP = '''import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

# mt_io logs one INFO line per file read, which is thousands of lines per site
from loguru import logger as _loguru
_loguru.disable("mt_io")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # the figures are written to disk and shown from there, not inline
import matplotlib.pyplot as plt
from IPython.display import Image, display

from auslamp_proc import survey as SV, look
from auslamp_proc.raw import cache
from auslamp_proc.figures import record as FREC, site as FSITE

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

sv = SV.load_survey(SURVEY)
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
WORK = Path(sv.cfg["work_root"])
CHOSEN, WHY = SV.select_sites(sv, SITES, MAX_SITES)
WRITTEN = []
TIMING = {}

def site_dir(s):
    d = WORK / s
    d.mkdir(parents=True, exist_ok=True)
    return d

def load(s):
    """The site's 1 Hz cache and its row. One site is held at a time: a 60 d record is 105 MB a channel."""
    t0, arrays, meta = cache.load(s, WORK, 1)
    return t0, arrays, meta, sv.site(s)

def midpoint(t0, n, fs=1.0):
    return datetime.fromtimestamp(t0 + n / fs / 2, timezone.utc)

def show(s, name):
    p = WORK / s / name
    if SHOW != "none" and p.exists() and s == SHOW_SITE:
        display(Image(filename=str(p)))

SHOW_SITE = "" if (SHOW == "none" or not CHOSEN) else (CHOSEN[0] if SHOW == "first" else SHOW)

print("survey       %s" % sv.cfg["name"])
print("work root    %s" % WORK)
print("sites        %d chosen from %s" % (len(CHOSEN), WHY))
print("             %s" % " ".join(CHOSEN))
print("rates        %s Hz" % ", ".join(str(r) for r in RATES))
print("windows      %d min every %d min, periods 2 .. %g s" % (WIN_MIN, STEP_MIN, PMAX))
print("figures of   %s are shown below; every site's five figures are written to <work_root>/<site>/"
      % (SHOW_SITE or "no site"))
'''

WB02 = [
("md", r"""# 02 -- The records: look before processing

This workbook builds each site's cache and draws its record, its band coherence, its coherence maps, its
spectra and its spectrograms. Nothing is processed here. What it answers is which channels a site actually
recorded, over which days, and in which state: a reversed magnetic axis, a dead electric line, two lines
carrying one noise, a gain, a turn of the sensor, a rail or a step all show in these five pictures before any
transfer function is estimated.

The cache is the record as laid. Every hourly file is placed on one absolute time axis by its own header
start and never concatenated, so a missing hour stays a gap and does not shift the hours after it; gaps are
NaN; the counts are converted to nT and mV/km by the instrument constants and the site's own dipole lengths.
No sign, no rotation and no notch is applied: the frame and the signs come from `decisions.csv` at processing
time, and a cache carrying them could not be re-read under a different decision.

Four checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported FAILED and is
not revised afterwards.

The words for the reference kinds are fixed in workbook 01 and used here as words: single station, remote
site, fleet stack, observatory, stack + observatory, member."""),

("code", WB02_PARAMS),

("md", r"""## The survey and the chosen set

Everything below runs on `surveys/<SURVEY>/` and writes under that survey's `work_root`. `SITES` chooses the
set: `all` is every row of sites.csv, `largest` is the largest group of the deployment register workbook 01
wrote, a group name such as `G01` is that group, and a list is the sites it names. A group is the pool the
reference choices are drawn from, so running a workbook over one group is running it over a set of sites that
were recording at the same time."""),

("code", WB02_SETUP),

("md", r"""## The sites

The chosen set, with the dipole lengths the electric channels are divided by and the source of each. A cell
reading `assume:<value>` is used and carried into every product's provenance until a sheet replaces it, and is
named as an assumption below. The dipoles matter to the level of every apparent resistivity the survey will
produce: rho scales with the square of the dipole length, so a dipole wrong by 20 per cent moves rho by 44 per
cent."""),

("code", '''rows = []
for s in CHOSEN:
    r = sv.site(s)
    dn, kn = cache.dipole_value(r.dipole_n_m)
    de, ke = cache.dipole_value(r.dipole_e_m)
    rows.append(dict(site=s, instrument=r.instrument, sample_rate_hz=r.sample_rate_hz,
                     days=r.days, dipole_n_m=dn, dipole_e_m=de,
                     assumed=" ".join([k for k, v in (("Ex", kn), ("Ey", ke)) if v == "assume"]) or "",
                     dipole_source=str(r.dipole_source)[:70]))
chosen_table = pd.DataFrame(rows)
print(chosen_table.to_string(index=False))
print()
n_assumed = int((chosen_table.assumed != "").sum())
print("%d of %d sites carry an assumed dipole (%s)"
      % (n_assumed, len(chosen_table), ", ".join(chosen_table.site[chosen_table.assumed != ""]) or "none"))
print()
print("the instrument constants the counts are converted by, from survey.yaml")
for inst, k in (sv.cfg.get("instruments") or {}).items():
    print("%s" % inst)
    for key, val in k.items():
        print("   %-14s %s" % (key, val))
'''),

("md", r"""## The cache

Each site is built once and then read from disk. The reconciliation below is the placement's own arithmetic:
every file found on disk, every file read, every sample read, every sample written onto the axis, and every
sample written over a value already there. An hour the recorder wrote twice raises `overlap_samples`; an hour
it could not write leaves NaN and lowers nothing, which is why the NaN fraction is a separate column and not
part of the equality.

`auslamp_proc.raw.cache.build` states the criterion in its own docstring, because the check has to be able to
fail: the form in the script this was ported from reads `placed == read or overlap >= 0`, whose right-hand
limb is true at every site, so it passes on a record it has never tested.

**This check fails if, at any site, samples placed differs from samples read, or any two files overlap, or a
channel is more than 50 per cent NaN, or the rate read differs from the rate survey.yaml gives the
instrument.**"""),

("code", '''recon, prov_all, built = [], {}, []
want_rate = {i: float(b["sample_rate_hz"]) for i, b in sv.cfg["instruments_layout"].items()}
for s in CHOSEN:
    t = time.time()
    row = sv.site(s)
    prov = cache.build(row, sv.cfg, rates=tuple(RATES), force=REBUILD)   # <- REBUILD True re-reads the raw
    TIMING[s] = {"cache_s": round(time.time() - t, 1)}
    prov_all[s] = prov
    built.append("%-9s %6.1f s  %7.3f d  %s" % (s, TIMING[s]["cache_s"], prov["span_days"],
                                                prov["verdict_lines"][-1]))
    recon += cache.reconciliation_row(prov)
    print(built[-1], flush=True)

recon = pd.DataFrame(recon)
print()
print(recon.to_string(index=False))

span = pd.DataFrame([dict(site=s, t0=prov_all[s]["t0_iso"][:16], span_days=prov_all[s]["span_days"],
                          fs_raw=prov_all[s]["fs_raw"], n_1hz=prov_all[s]["n_1hz"],
                          worst_nan=max(prov_all[s]["nan_fraction_1hz"].values()))
                     for s in CHOSEN])
print()
print(span.to_string(index=False))

mismatch = recon[(recon.samples_placed != recon.samples_read) | (recon.overlap_samples > 0)
                 | (recon.files_read != recon.files_found) | (recon.files_unplaced > 0)]
nanny = recon[recon.nan_fraction > 0.5]
rate_bad = [(s, prov_all[s]["fs_raw"], want_rate.get(sv.site(s).instrument))
            for s in CHOSEN if abs(prov_all[s]["fs_raw"] - want_rate.get(sv.site(s).instrument, -1)) > 1e-6]
print()
if not len(recon):
    print("VERDICT: UNJUDGED -- no site was built, so no channel was reconciled")
elif len(mismatch) or len(nanny) or rate_bad:
    print("VERDICT: FAIL -- %d of %d channel rows do not place every sample they read without overlap (%s); "
          "%d channels are more than 50 %% NaN (%s); %d sites read a rate that is not survey.yaml's (%s)"
          % (len(mismatch), len(recon),
             "; ".join("%s %s placed %d of %d, overlap %d" % (r.site, r.channel, r.samples_placed,
                                                              r.samples_read, r.overlap_samples)
                       for r in mismatch.itertuples()) or "none",
             len(nanny), "; ".join("%s %s %.0f %%" % (r.site, r.channel, 100 * r.nan_fraction)
                                   for r in nanny.itertuples()) or "none",
             len(rate_bad), "; ".join("%s %g vs %g Hz" % x for x in rate_bad) or "none"))
else:
    print("VERDICT: PASS -- %d sites, %d channel rows: every sample read was placed (%d in all), no two files "
          "overlap, the worst channel is %.1f %% NaN at %s, and every site reads %s"
          % (len(CHOSEN), len(recon), int(recon.samples_placed.sum()), 100 * recon.nan_fraction.max(),
             recon.site[recon.nan_fraction.idxmax()],
             ", ".join("%g Hz" % v for v in sorted(set(want_rate.values())))))
'''),

("md", r"""## The record (figure 01) and the DC field

Figure 01 is the whole 1 Hz record as laid, in eight panels: the five channels as a per-minute mean drawn over
the per-minute range, then H = sqrt(Hx^2 + Hy^2), F = sqrt(Hx^2 + Hy^2 + Hz^2), and the hourly sensor angle
atan2(Hy, Hx). Each magnetic panel carries the IGRF value at the site at the record midpoint and the ratio of
the measured median to it.

What to look for, in this order:

- a reversed axis: Bx below zero, or a panel that is the mirror of its neighbours';
- a dead channel: a flat line, or a range of a few counts where the others move tens of nT;
- a gain: F away from IGRF by more than 5 per cent, with H and Z away by the same factor;
- a turn of the sensor: a step in the hourly angle, which is the mast being knocked or re-laid, and splits the
  record into rotation regimes;
- a rail: a channel pinned at a constant for hours, which is the amplifier at its limit;
- a step: a jump in an electric line, which is an electrode being disturbed.

The table under the figure is the DC test of every chosen site: the medians against IGRF, the three ratios,
the sensor angle, the tilt and the flags. The rules, in the order they fire, are
`|F/F_igrf - 1| > 0.05` a gain or a broken axis; else `|Bz/Z - 1| > 0.10` or `|H/H_igrf - 1| > 0.10` a tilt;
`Bx < 0` a reversed north axis; `|angle| > 30 deg` a sensor laid far from north or the axes exchanged
(`auslamp_proc.look.dc_test`).

The flags are the finding. What the check tests is that the rule fires where the numbers say it must.

**This check fails if any site's F differs from IGRF by more than 5 per cent without a flag naming it, or if
any Bx is negative without a flag.**"""),

("code", '''dc_rows = []
for s in CHOSEN:
    t = time.time()
    t0, arrays, meta, row = load(s)
    n = len(arrays["Hx"])
    fig, line = FREC.record(t0, arrays, site_dir(s) / "01_record.png", site=s, survey=sv.cfg["name"],
                            lat=float(row.lat), lon=float(row.lon), elev_m=float(row.elev_m), fs=1.0)
    plt.close(fig)
    WRITTEN.append(site_dir(s) / "01_record.png")
    d = look.dc_test(arrays, float(row.lat), float(row.lon), float(row.elev_m), midpoint(t0, n))
    d["site"] = s
    d["days"] = round(n / 86400.0, 2)
    dc_rows.append(d)
    pd.DataFrame([d]).to_csv(site_dir(s) / "dc.csv", index=False)
    WRITTEN.append(site_dir(s) / "dc.csv")
    TIMING[s]["record_s"] = round(time.time() - t, 1)
    del arrays
    print("%-9s %5.1f s  %s" % (s, TIMING[s]["record_s"], line), flush=True)

dc = pd.DataFrame(dc_rows)[["site", "days", "Bx", "By", "Bz", "H", "F", "igrf_X", "igrf_Y", "igrf_Z",
                            "igrf_H", "igrf_F", "Bx_over_X", "Bz_over_Z", "H_over_Higrf", "F_over_Figrf",
                            "angle_deg", "tilt_deg", "flags", "verdict"]]
print()
print(dc.to_string(index=False))
show(SHOW_SITE, "01_record.png")

f_off = dc[(dc.F_over_Figrf - 1).abs() > 0.05]
# dc["flags"], not dc.flags: a DataFrame already carries a `flags` attribute of its own
f_unflagged = f_off[~f_off["flags"].str.contains("F/Figrf", na=False)]
bx_neg = dc[dc.Bx < 0]
bx_unflagged = bx_neg[~bx_neg["flags"].str.contains("Bx negative", na=False)]
print()
print("%d of %d sites carry a flag: %s"
      % (int((dc["flags"] != "").sum()), len(dc),
         "; ".join("%s [%s]" % (r.site, r.flags) for r in dc.itertuples() if r.flags) or "none"))
print()
if not len(dc):
    print("VERDICT: UNJUDGED -- no site was read, so no DC field was scored")
elif len(f_unflagged) or len(bx_unflagged):
    print("VERDICT: FAIL -- %d of %d sites are more than 5 %% off IGRF in F with no flag naming it (%s); "
          "%d sites have Bx below zero with no flag (%s)"
          % (len(f_unflagged), len(dc), ", ".join(f_unflagged.site) or "none",
             len(bx_unflagged), ", ".join(bx_unflagged.site) or "none"))
else:
    print("VERDICT: PASS -- all %d sites scored; the %d sites over 5 %% in F (%s) and the %d with Bx below "
          "zero (%s) each carry the flag the rule fires, and the largest F/F_igrf departure is %.1f %% at %s"
          % (len(dc), len(f_off), ", ".join(f_off.site) or "none", len(bx_neg),
             ", ".join(bx_neg.site) or "none", 100 * (dc.F_over_Figrf - 1).abs().max(),
             dc.site[(dc.F_over_Figrf - 1).abs().idxmax()]))
'''),

("md", r"""## Band coherence (figure 02) and the electric lines

Figure 02 is the squared coherence of four pairs against time, one line per band (5-20, 20-200, 100-1000 and
1000-10000 s) on a 12 h running median. How to read them:

| pair | what it is |
|---|---|
| Bx-Ey and By-Ex | the impedance pairs: this is the coherence the transfer function is estimated from |
| Bx-By | the polarisation of the source: a stretch where it rises to one is a single-polarisation event, and a tensor cannot be resolved there |
| Ex-Ey | the two electric lines against each other: high with Bx-By low is common-mode electric noise, the shared electrode of the L array carrying one signal into both lines |

The table under it is the per-UTC-day state of each line: the 20-200 s coherence of Ex with the site's own Hy
and of Ey with its Hx, of Ex with Ey, and the standard deviation of each line after a 3,000 s high-pass, with
the state `dead` (high-passed std under 0.2 mV/km), `sound` (coherence with H at least 0.4 and Ex-Ey under
0.5), `common` (Ex-Ey at least 0.6 and coherence with H under 0.3) or `weak` (`auslamp_proc.look.elines`).
Every series is despiked before the high-pass: one logger spike inside one Welch segment otherwise puts a
whole day's coherence at zero. A day in which any of the four channels is under 80 per cent finite is scored
`gap` and is not a scored day.

Figure 03's coherence maps are computed in the same pass and written with figure 02, because the band lines
are the same numbers averaged over a band.

**This check fails if the state machine is UNJUDGED at any site, that is if no day was scored.**"""),

("code", '''e_rows, band_rows = [], []
for s in CHOSEN:
    t = time.time()
    t0, arrays, meta, row = load(s)
    n = len(arrays["Hx"])
    el = look.elines(t0, arrays, fs=1.0)
    el.to_csv(site_dir(s) / "elines.csv", index=False)
    WRITTEN.append(site_dir(s) / "elines.csv")
    fig, maps = FSITE.coherence_maps(
        t0, arrays, site_dir(s) / "03_coherence_maps.png", fs=1.0,
        win_s=WIN_MIN * 60, step_s=STEP_MIN * 60, pmax=PMAX,          # <- WIN_MIN, STEP_MIN, PMAX
        title="%s %s: squared coherence, %d min windows every %d min, %g h running median"
              % (sv.cfg["name"], s, WIN_MIN, STEP_MIN, FSITE.SMOOTH_H))
    plt.close(fig)
    fig = FSITE.coherence_bands(t0, maps, site_dir(s) / "02_coherence_bands.png", n, fs=1.0,
                                step_s=STEP_MIN * 60, win_min=WIN_MIN,
                                title="%s %s: band coherence per %d min window, %g h running median"
                                      % (sv.cfg["name"], s, WIN_MIN, FSITE.LINE_SMOOTH_H))
    plt.close(fig)
    WRITTEN += [site_dir(s) / "02_coherence_bands.png", site_dir(s) / "03_coherence_maps.png"]
    bt = FSITE.band_table(maps)
    for pair, r in bt.iterrows():
        band_rows.append(dict(site=s, pair=pair, **{c: r[c] for c in bt.columns}))
    e_rows.append(look.eline_summary(el, s))
    TIMING[s]["coherence_s"] = round(time.time() - t, 1)
    del arrays, maps
    su = e_rows[-1]
    print("%-9s %5.1f s  days %3d scored %3d | Ex sound %.2f common %.2f dead %.2f | Ey sound %.2f "
          "common %.2f dead %.2f" % (s, TIMING[s]["coherence_s"], su["days_total"], su["days_scored"],
                                     su["Ex_sound_frac"], su["Ex_common_frac"], su["Ex_dead_frac"],
                                     su["Ey_sound_frac"], su["Ey_common_frac"], su["Ey_dead_frac"]),
          flush=True)

esum = pd.DataFrame(e_rows)
print()
print(esum[["site", "days_total", "days_scored",
            "Ex_sound", "Ex_weak", "Ex_common", "Ex_dead", "Ex_gap",
            "Ey_sound", "Ey_weak", "Ey_common", "Ey_dead", "Ey_gap",
            "Ex_sound_frac", "Ey_sound_frac", "both_sound_frac",
            "Ex_longest_sound_run_d", "Ey_longest_sound_run_d", "gap_channels"]].to_string(index=False))
print()
print("gap_channels counts the days a channel was under 80 % finite after the despike; a magnetic channel "
      "there is a spiky sensor, not an outage, and is read from the record figure")
print()
if SHOW_SITE and (WORK / SHOW_SITE / "elines.csv").exists():
    print("the day table of %s (the same table is written to every site's elines.csv)" % SHOW_SITE)
    print(pd.read_csv(WORK / SHOW_SITE / "elines.csv").to_string(index=False))
print()
print("median band coherence per pair over the record, 20-200 s")
bandt = pd.DataFrame(band_rows)
print(bandt.pivot(index="site", columns="pair", values="20-200 s").to_string())
print()
print("median band coherence per pair over the record, 100-1000 s")
print(bandt.pivot(index="site", columns="pair", values="100-1000 s").to_string())
show(SHOW_SITE, "02_coherence_bands.png")

unjudged = esum[esum.days_scored == 0]
print()
if not len(esum):
    print("VERDICT: UNJUDGED -- no site was scored at all")
elif len(unjudged):
    print("VERDICT: FAIL -- the state machine scored no day at %d of %d sites (%s)"
          % (len(unjudged), len(esum),
             "; ".join("%s, refused on %s" % (r.site, r.gap_channels or "no channel named")
                       for r in unjudged.itertuples())))
else:
    print("VERDICT: PASS -- every one of the %d sites scored at least one day, the thinnest being %d days at "
          "%s; %d sites have a line sound on more than half their days and %d have a line common-mode on more "
          "than half" % (len(esum), int(esum.days_scored.min()), esum.site[esum.days_scored.idxmin()],
                         int(((esum.Ex_sound_frac > 0.5) | (esum.Ey_sound_frac > 0.5)).sum()),
                         int(((esum.Ex_common_frac > 0.5) | (esum.Ey_common_frac > 0.5)).sum())))
'''),

("md", r"""## Coherence maps (figure 03)

The same coherence as figure 02, drawn as squared coherence against time and period rather than averaged into
four bands. The ladder below is how the map reaches 20,000 s from 60 min windows: the base level takes the
largest power of two leaving 4 segments in the window, and each level above it covers a factor of 4 in period
with a segment as long as its longest period, a window of 6 segments and a step of half a segment. Every level
is drawn on the base level's time grid, so the long-period rows are the same value repeated over the hours
their window covers, which is what makes the top of the map look blocky.

What the map shows that the band lines do not:

- a live line is bright from a few tens of seconds to thousands, and the brightness moves with the source: it
  dims at quiet local night and brightens in a storm;
- a dead line is dark at every period at every hour, and stays dark through a storm that lights its neighbour;
- a polarised stretch is a bright band in Bx-By over the same hours that Bx-Ey or By-Ex goes dark: the source
  was one polarisation, so one of the two impedance pairs had nothing to work with;
- common-mode noise is bright in Ex-Ey and dark in both impedance pairs at the same hours."""),

("code", '''ladder = FSITE.levels_plan(1.0, WIN_MIN * 60, STEP_MIN * 60, pmax=PMAX)
print("the level ladder at 1 Hz with %d min windows every %d min, to %g s" % (WIN_MIN, STEP_MIN, PMAX))
print(ladder.to_string(index=False))
print()
print("%d levels; %d log bins per decade; the DC bin is dropped and a bin holding no FFT harmonic is dropped"
      % (len(ladder), FSITE.PER_DECADE))
print("guide lines at %s s; the maps carry a %g h running median along time and the band lines %g h"
      % (", ".join(str(g) for g in FSITE.GUIDE_S), FSITE.SMOOTH_H, FSITE.LINE_SMOOTH_H))
print()
have3 = [s for s in CHOSEN if (WORK / s / "03_coherence_maps.png").exists()]
print("%d of %d sites have a coherence map on disk" % (len(have3), len(CHOSEN)))
show(SHOW_SITE, "03_coherence_maps.png")
'''),

("md", r"""## Spectra (figure 04)

One Welch spectrum per channel over the longest finite run of the record, 4,096-sample segments, the DC bin
dropped, on log-log axes with the 10-1000 s band shaded: that is where a 1 Hz fluxgate does its work and where
the transfer function will be read.

This section is a reading, not a check. A sound electric line falls 4.6-5.6 decades over 1 mHz to 3 Hz; a flat spectrum, under 1 decade of fall over that span, is an open input, an electrode
that is not connected to the ground. On a 1 Hz cache 3 Hz is above the Nyquist frequency, so the fall is
measured to 0.4 Hz instead and the band used is printed with the numbers; the bound 1 decade still separates a
line that falls from one that does not."""),

("code", '''fall_rows = []
for s in CHOSEN:
    t = time.time()
    t0, arrays, meta, row = load(s)
    fig, fall = FSITE.spectra(arrays, site_dir(s) / "04_spectra.png", fs=1.0,
                              title="%s %s: Welch spectra of the whole record (shaded: 10-1000 s, where a "
                                    "1 Hz fluxgate does its work; dotted: the Nyquist period)"
                                    % (sv.cfg["name"], s))
    plt.close(fig)
    WRITTEN.append(site_dir(s) / "04_spectra.png")
    for ch, r in fall.iterrows():
        fall_rows.append(dict(site=s, channel=ch, decades_of_fall=r.decades_of_fall,
                              power_at_100s=r.power_at_100s, band=r.fall_band))
    TIMING[s]["spectra_s"] = round(time.time() - t, 1)
    del arrays

fallt = pd.DataFrame(fall_rows)
wide = fallt.pivot(index="site", columns="channel", values="decades_of_fall")
print("decades of fall over %s" % fallt.band.iloc[0])
print(wide.to_string())
print()
flat = fallt[(fallt.channel.isin(["Ex", "Ey"])) & (fallt.decades_of_fall < 1.0)]
print("electric lines falling less than 1 decade (an open input, read and not scored): %s"
      % ("; ".join("%s %s %.2f" % (r.site, r.channel, r.decades_of_fall) for r in flat.itertuples())
         or "none"))
show(SHOW_SITE, "04_spectra.png")
'''),

("md", r"""## Spectrograms (figure 05)

The power of each channel against time and period, in dB of absolute power density, with the colour limits at
that channel's own 2nd and 98th percentile, so a dead stretch is a dark band at every period and a storm is
bright across the whole picture. The table under it reads the base level of the same maps: the median power in
the 20-200 s band, and the fraction of windows more than 10 dB below it (an hour the channel was quiet or
dead) or above it (a burst).

A channel whose `windows 10 dB below` is a large fraction died part way through the record, and the days it
died are what the mask in workbook 05 will cut."""),

("code", '''spec_rows = []
for s in CHOSEN:
    t = time.time()
    t0, arrays, meta, row = load(s)
    fig, pmaps = FSITE.spectrograms(t0, arrays, site_dir(s) / "05_spectrograms.png", fs=1.0,
                                    win_s=WIN_MIN * 60, step_s=STEP_MIN * 60, pmax=PMAX,
                                    title="%s %s: power per channel, %d min windows every %d min, %g h "
                                          "running median" % (sv.cfg["name"], s, WIN_MIN, STEP_MIN,
                                                              FSITE.SMOOTH_H))
    plt.close(fig)
    tab = FSITE.spectrogram_table(pmaps)
    tab.to_csv(site_dir(s) / "05_spectrograms.csv")
    WRITTEN += [site_dir(s) / "05_spectrograms.png", site_dir(s) / "05_spectrograms.csv"]
    for ch, r in tab.iterrows():
        spec_rows.append(dict(site=s, channel=ch, median_db=r["median dB at 20-200 s"],
                              frac_10db_below=r["windows 10 dB below"],
                              frac_10db_above=r["windows 10 dB above"], windows=r["windows"]))
    TIMING[s]["spectrogram_s"] = round(time.time() - t, 1)
    del arrays, pmaps
    print("%-9s %5.1f s  %s" % (s, TIMING[s]["spectrogram_s"],
                                " ".join("%s %.1f dB" % (r["channel"], r["median_db"])
                                         for r in spec_rows[-5:])), flush=True)

spect = pd.DataFrame(spec_rows)
print()
print("median power at 20-200 s (dB)")
print(spect.pivot(index="site", columns="channel", values="median_db").to_string())
print()
print("fraction of windows 10 dB below the median at 20-200 s")
print(spect.pivot(index="site", columns="channel", values="frac_10db_below").to_string())
show(SHOW_SITE, "05_spectrograms.png")
'''),

("md", r"""## The survey table

One row per site of what the five figures showed: the total field against IGRF and the flags the DC rule
fired, the fraction of scored days each electric line was sound, common-mode, dead or weak, and the median
power at 20-200 s per channel. It is written to `<work_root>/survey/records_summary.csv`, and it is what
workbook 03 reads when it chooses which sites and which days to process.

**This check fails if any site in the chosen set has no figure 01, 02, 03, 04 or 05 on disk.**"""),

("code", '''FIGS = ["01_record.png", "02_coherence_bands.png", "03_coherence_maps.png", "04_spectra.png",
        "05_spectrograms.png"]
db = spect.pivot(index="site", columns="channel", values="median_db")
summary = (dc.set_index("site")[["days", "F_over_Figrf", "H_over_Higrf", "Bz_over_Z", "angle_deg", "flags"]]
           .join(esum.set_index("site")[["days_scored", "Ex_sound_frac", "Ex_weak_frac", "Ex_common_frac",
                                         "Ex_dead_frac", "Ey_sound_frac", "Ey_weak_frac", "Ey_common_frac",
                                         "Ey_dead_frac", "both_sound_frac"]])
           .join(db.rename(columns=lambda c: "db_20_200_%s" % c)))
summary["figures_on_disk"] = [sum((WORK / s / f).exists() for f in FIGS) for s in summary.index]
summary = summary.reset_index()
out_csv = WORK / "survey" / "records_summary.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_csv, index=False)
WRITTEN.append(out_csv)
print(summary.to_string(index=False))
print()
print("wrote %s (%d rows, %d columns)" % (out_csv, len(summary), len(summary.columns)))

short = summary[summary.figures_on_disk < len(FIGS)]
print()
if not len(summary):
    print("VERDICT: UNJUDGED -- the chosen set is empty, so no site was scored")
elif len(short):
    print("VERDICT: FAIL -- %d of %d sites are missing one of the five figures (%s)"
          % (len(short), len(summary),
             "; ".join("%s has %d of %d" % (r.site, r.figures_on_disk, len(FIGS))
                       for r in short.itertuples())))
else:
    print("VERDICT: PASS -- all %d sites carry all %d figures on disk (%d files), and the summary row of each "
          "is written" % (len(summary), len(FIGS), len(summary) * len(FIGS)))
'''),

("md", r"""## What was written"""),

("code", '''rows = []
for p in sorted(set(WRITTEN)):
    p = Path(p)
    if p.exists():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for r in tuple(RATES):
    for s in CHOSEN:
        for suffix in (".npz", ".json"):
            p = WORK / ("cache_%dhz" % r) / (s + suffix)
            if p.exists():
                rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print(files.to_string(index=False))
print()
print("%d files, %.1f MB" % (len(files), files.kb.sum() / 1024))
for r in tuple(RATES):
    d = WORK / ("cache_%dhz" % r)
    mb = sum(p.stat().st_size for p in d.glob("*.npz")) / 1024 ** 2 if d.exists() else 0.0
    print("cache_%dhz %.1f MB over %d files" % (r, mb, len(list(d.glob("*.npz"))) if d.exists() else 0))
print()
tm = pd.DataFrame(TIMING).T
tm["total_s"] = tm.sum(axis=1)
print("seconds per site")
print(tm.to_string())
print()
print("%d sites, %.1f minutes in all, %.1f s a site on average"
      % (len(tm), tm.total_s.sum() / 60.0, tm.total_s.mean()))
'''),
]


NOTEBOOKS = {"01_survey.ipynb": WB01, "02_records.ipynb": WB02}


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python (%s)" % KERNEL, "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


def main(argv):
    names = sorted(NOTEBOOKS)
    if argv:
        names = [n for n in names if any(n.startswith(a) for a in argv)]
    for name in names:
        path = HERE / name
        nbf.write(nb(NOTEBOOKS[name]), path)
        print("wrote %s (%d cells)" % (path, len(NOTEBOOKS[name])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
