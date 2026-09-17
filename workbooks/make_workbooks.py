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
DEAD_FRACTION = 0.2           # a day of a line is dead below this fraction of the line's own median daily std
DEAD_ABS_MV_PER_KM = 0.0      # or below this absolute floor in mV/km; 0 = off (a line dead all record reads weak)
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
the state `dead` (high-passed std under `DEAD_FRACTION` = 0.2 of the line's own median daily std, or under
the absolute floor `DEAD_ABS_MV_PER_KM` where one is set; the threshold is relative because the level of a
line scales with its dipole length and differs between surveys), `sound` (coherence with H at least 0.4 and
Ex-Ey under 0.5), `common` (Ex-Ey at least 0.6 and coherence with H under 0.3) or `weak`
(`auslamp_proc.look.elines`). A line dead for its whole record sits at its own noise floor and reads `weak`
under the relative rule; the absolute floor is the switch for that case.
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
    el = look.elines(t0, arrays, fs=1.0, dead_fraction=DEAD_FRACTION,        # <- DEAD_FRACTION, DEAD_ABS_MV_PER_KM
                     dead_abs_mv_per_km=DEAD_ABS_MV_PER_KM)
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


# ===================================================================== 03 processing

WB03_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
RUN = "first"                 # the run name; every site of one run lands in <site>/<RUN>_<stamp>/
SITES = "all"                 # "all" | "largest" (the register's largest group) | a group name | ["Q49", "Q50"]
MAX_SITES = 0                 # 0 = every chosen site; a cap keeps an example short and names what it kept
WORK_ROOT = None              # None = survey.yaml work_root; every reference and product lands under it
KINDS = ["single", "remote", "stack", "obs", "stack_obs"]    # the code keys of the five reference kinds
RATES = [1]                   # [1] or [10]; a 10 Hz pass is one lane at 22-37 GB and 7-15 min a product
PARAMS = "kaiser20_75"        # the Aurora parameter set: kaiser20_50 | dpss4_75 | kaiser20_w512
LANES = 3                     # concurrent single-site subprocesses; 1 at 10 Hz, where a pass peaks at 37 GB
REDO = False                  # True remakes a product whose EDI is already on disk
'''

WB03_RULES = '''# ---- the rule thresholds: a change here changes which reference every product was built on ----
COH_MIN = 0.5                 # the remote-site gate at 20-200 s; branches 1 and 2 need it
COH_RELAX = 0.3               # branch 3 relaxes to this where no clean candidate reaches COH_MIN
MIN_OVERLAP_DAYS = 20         # the overlap floor, taken with 0.75 x the days the target can use
STACK_CUTOFF = 0.5            # a member below this fleet coherence at 100-1000 s is refused
STACK_MAX = 8                 # the best this many members enter the stack
STACK_MIN = 2                 # fewer than this and the stack is refused: one member is a remote site renamed
EVENT_FRAC_MAX = 0.02         # clean: at most this fraction of the 600 s chunks flagged as events
BASELINE_MAX = 10             # clean: at most this multiple of the survey median baseline
'''

WB03_SETUP = '''import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

import auslamp_proc
from auslamp_proc import survey as SV, geo, observatory
from auslamp_proc.raw import cache
from auslamp_proc.process import KIND_WORD, aurora_run, edi as EDI, frame as FR, mth5_build
from auslamp_proc.process import provenance as PROV, references as REF, transients as TR

# aurora and mth5 re-arm loguru when they are first imported, which would put tens of thousands of INFO
# lines into this workbook. Import them, then drop the sinks. The "Window is longer than the time series"
# ERROR is expected on a short run: a one-hour run holds no complete window at the deep decimation levels.
aurora_run.silence_loggers()
import aurora

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 400)

REPO = Path(auslamp_proc.__file__).resolve().parent.parent
sv = SV.load_survey(SURVEY)
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
WORK = Path(sv.cfg["work_root"])

def cached(site, rate=1):
    return (WORK / ("cache_%dhz" % rate) / ("%s.npz" % site)).exists()

# nothing here reads the raw tree: a site enters the run and the reference pool only where workbook 02 has
# built its cache, and the sites that have none are named rather than passed over
ALL_SITES = [s for s in sv.sites.site if cached(s)]
NO_CACHE = [s for s in sv.sites.site if not cached(s)]
CHOSEN, WHY = SV.select_sites(sv, SITES, 0)
CHOSEN = [s for s in CHOSEN if s in ALL_SITES]
if MAX_SITES and len(CHOSEN) > int(MAX_SITES):          # <- MAX_SITES caps the cached set, not the table
    WHY += "; capped at %d of %d with a cache" % (int(MAX_SITES), len(CHOSEN))
    CHOSEN = CHOSEN[:int(MAX_SITES)]
OUT = WORK / "survey"
OUT.mkdir(parents=True, exist_ok=True)
WRITTEN = []

# the pool thresholds are survey.yaml values; the parameter cell above governs this run of the workbook
sv.cfg.setdefault("pool", {})
sv.cfg["pool"]["event_frac_max"] = EVENT_FRAC_MAX
sv.cfg["pool"]["baseline_max"] = BASELINE_MAX

def store(rate, spec=None):
    """The reference store at one rate, carrying this workbook's thresholds."""
    return REF.Store(sv, ALL_SITES, rate=rate, coh_min=COH_MIN, coh_relax=COH_RELAX,
                     min_overlap_days=MIN_OVERLAP_DAYS, cutoff=STACK_CUTOFF, n_max=STACK_MAX,
                     n_min=STACK_MIN)

ST = store(RATES[0])
OBS = (sv.cfg.get("observatory") or {}).get("code", "")

# The stamp is the launch time in UTC of the whole run, and every site of one run shares it. A run is
# resumed rather than restarted: where <site>/<RUN>_<stamp>/ folders already exist the latest of their
# stamps is taken up again, so re-running this workbook finishes a run that was stopped part way and REDO
# False leaves the products already on disk alone. A RUN name with no folder yet mints a new stamp.
_stamps = sorted({p.name.split("_", 1)[1] for s in CHOSEN for p in (WORK / s).glob(RUN + "_*")
                  if p.is_dir()})
STAMP = _stamps[-1] if _stamps else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

print("survey       %s" % sv.cfg["name"])
print("work root    %s" % WORK)
print("sites        %d chosen from %s" % (len(CHOSEN), WHY))
print("             %s" % " ".join(CHOSEN))
print("pool from    %d of the %d sites in sites.csv, the ones with a 1 Hz cache (a reference may come from "
      "a site this run does not process)" % (len(ALL_SITES), len(sv.sites)))
if NO_CACHE:
    print("no cache     %s -- run workbook 02 over them to bring them into the pool" % " ".join(NO_CACHE))
print("kinds        %s" % ", ".join("%s (%s)" % (KIND_WORD[k], k) for k in KINDS))
print("rates        %s Hz, parameter set %s, %d lane(s)" % (", ".join(str(r) for r in RATES), PARAMS, LANES))
print("run          %s_%s  ->  <work_root>/<site>/  (%s)"
      % (RUN, STAMP, "resumed: the folders of this run name already exist" if _stamps else "a new run"))
print("engine       Aurora %s, mth5 %s" % (aurora.__version__, __import__("mth5").__version__))
print("observatory  %s at %s" % (OBS, (sv.cfg.get("observatory") or {}).get("archive", "")))
'''

WB03 = [
("md", r"""# 03 -- Processing: Aurora over the sites and the reference kinds

This workbook turns each site's cache into transfer functions. It applies the signs and the frame from
`decisions.csv`, scans each record for transients, builds the four references a site can be processed
against, writes one MTH5 per product with one run per kept stretch, runs Aurora, and writes an EDI and an
XML with the provenance of what was done.

Five kinds are built, and they are not alternatives to choose between here: every one of them is made, and
which of them becomes the product of record is workbook 06's question.

| word | what it is | code key | what it is for |
|---|---|---|---|
| single station | the site's own H and E | `single` | the control that says what a reference buys; biased down wherever the site's H noise is coherent with itself |
| remote site | one other site's H | `remote` | cancels the noise that is uncorrelated between the two sites; costs the variance of one more record |
| fleet stack | a coherence-weighted mean of several sites' H | `stack` | averages down each member's own noise, and needs the members aligned first |
| observatory | an INTERMAGNET one-second record | `obs` | far and quiet, and reaches the long periods |
| stack + observatory | the stack with the observatory as a member | `stack_obs` | the stack's short end with the observatory's long end |

The frame is fixed and stated in every file: the horizontal magnetic pair of each site is turned so its mean
Hy is zero, which takes out the fluxgate's hand-compass misalignment, and the IGRF declination is recorded
and not applied. Every member of a reference is turned into its own mean-field frame before it is stacked.

The record is never cut. A mask that removes a transient is applied by giving Aurora one run per kept
stretch of at least 3,600 s: cutting and concatenating puts a step at every join, and a step common to E, H
and the reference is coherent between them, so a robust regression fits it rather than down-weighting it.

Six checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported FAILED and is
not revised afterwards.

At 10 Hz Aurora reads about 8 per cent low at 4-32 s against its own 1 Hz product (AusLAMP Victoria,
2026-09-11). The 10 Hz products are made and each carries that sentence in its own file; they are not
spliced here, which is workbook 06's work."""),

("code", WB03_PARAMS),
("code", WB03_RULES),

("md", r"""## The survey, the sites and the run

Everything below runs on `surveys/<SURVEY>/` and writes under that survey's `work_root`. The run name and
the launch stamp name one folder per site, `<site>/<RUN>_<stamp>/`, so every product of one run sits
together with the provenance of the run that made it."""),

("code", WB03_SETUP),

("md", r"""## The sites and their decisions

The `sites.csv` facts each product's header carries, and the `decisions.csv` row it was built under. A sign
cell reading `decide` is used as +1 and recorded as undecided: the product is made, its EDI carries a
`signs_undecided` line naming the channels, and its provenance carries a caveat. Nothing is decided silently.

`remote_site` is the analyst's choice of remote where it names a site; where it reads `decide` the
five-branch rule below chooses one. `stack_members`, `rot_regimes` and `rot_drop` work the same way: a value
governs, `decide` hands the question to the rule."""),

("code", '''rows = []
for s in CHOSEN:
    r = sv.site(s)
    d = sv.decision(s)
    signs = {c: FR.read_sign(d.get(FR.SIGN_COLUMN[c])) for c in FR.CHANNELS}
    rows.append(dict(site=s, days=r.days, lat=r.lat, lon=r.lon, dipole_n_m=r.dipole_n_m,
                     dipole_e_m=r.dipole_e_m, declination_deg=r.declination_deg,
                     signs=" ".join("%s%+d" % (c, signs[c][0]) for c in FR.CHANNELS),
                     undecided=" ".join(c for c in FR.CHANNELS if not signs[c][1]) or "none",
                     remote_site=d.get("remote_site", ""), stack_members=d.get("stack_members", ""),
                     rot_regimes=d.get("rot_regimes", ""), rot_drop=d.get("rot_drop", "")))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
print()
n_und = int((tab.undecided != "none").sum())
print("%d of %d sites carry at least one undecided sign; every such product says so in its own EDI"
      % (n_und, len(tab)))
print("%d sites name a remote in decisions.csv, %d leave it to the rule"
      % (int((~tab.remote_site.str.lower().isin(["decide", "", "nan"])).sum()),
         int(tab.remote_site.str.lower().isin(["decide", "", "nan"]).sum())))
print("%d sites name their own stack members; the rest take the clean pool of their overlap group"
      % int((~tab.stack_members.str.lower().isin(["decide", "", "nan"])).sum()))
print()
print("the five kinds, as words and as the code keys that appear in file names")
print(pd.DataFrame([dict(word=KIND_WORD[k], key=k) for k in REF.KINDS_WITH_STORE +
                    ("single",)]).to_string(index=False))
'''),

("md", r"""## The frame

Two things happen to a record before anything is estimated from it, in this order. Each channel is
multiplied by its `decisions.csv` sign. Then the horizontal magnetic pair is turned by
theta = atan2(mean Hy, mean Hx), which puts the mean Hy at zero; the electric lines are left as laid.

The rotation is a rigid turn of the pair, so it cannot change the total horizontal field at any sample: that
is what makes it testable. The angle is a property of how the sensor was laid, not of the field, and it is
recorded in every product as `h_rotation_deg`. The IGRF declination is recorded beside it and not applied --
turning by IGRF as well would rotate the record twice.

**This check fails if any site's rotated record has |mean Hy| > 1e-6 nT or a changed |H| at any sample.**
|H| is compared before and after the turn, sample by sample, on the same float64 arithmetic the processing
uses; the bound on the comparison is 1e-6 nT, which is eleven orders of magnitude below the 50,000 nT the
field itself carries."""),

("code", '''frame_rows = []
for s in CHOSEN:
    z = np.load(WORK / "cache_1hz" / ("%s.npz" % s), allow_pickle=False)
    h = {c: np.asarray(z[c], float) for c in ("Hx", "Hy")}
    z.close()
    d = sv.decision(s)
    h, applied, undecided = FR.apply_signs(h, d)
    before = np.hypot(h["Hx"], h["Hy"])
    turned, ang = FR.rotate_to_mean_field(h, regimes=FR.parse_regimes(d.get("rot_regimes")),
                                          drop=FR.parse_regimes(d.get("rot_drop")), fs=1.0)
    after = np.hypot(turned["Hx"], turned["Hy"])
    g = np.isfinite(before) & np.isfinite(after)
    all_signs = {c: FR.read_sign(d.get(FR.SIGN_COLUMN[c])) for c in FR.CHANNELS}
    frame_rows.append(dict(site=s, angle_deg=(ang if isinstance(ang, float) else "%d regimes" % len(ang)),
                           declination_deg=sv.site(s).declination_deg,
                           mean_Hy_nT=float(np.nanmean(turned["Hy"])),
                           worst_dH_nT=float(np.nanmax(np.abs(after[g] - before[g]))) if g.any() else np.nan,
                           samples=int(g.sum()),
                           H_signs=" ".join("%s%+d" % (c, applied[c]) for c in ("Hx", "Hy")),
                           undecided=" ".join(c for c in FR.CHANNELS if not all_signs[c][1]) or "none"))
    del h, turned, before, after
fr = pd.DataFrame(frame_rows)
print(fr.to_string(index=False))
print()
print("the angle is the sensor's own misalignment, and the declination beside it is the field's; the two "
      "are different quantities and neither is the other")

bad_hy = fr[fr.mean_Hy_nT.abs() > 1e-6]
bad_h = fr[fr.worst_dH_nT > 1e-6]
print()
if not len(fr) or not fr.samples.sum():
    print("VERDICT: UNJUDGED -- no site produced a rotated record to score")
elif len(bad_hy) or len(bad_h):
    print("VERDICT: FAIL -- %d of %d sites leave a mean Hy above 1e-6 nT (%s); %d change |H| by more than "
          "1e-6 nT (%s)"
          % (len(bad_hy), len(fr), "; ".join("%s %.3g nT" % (r.site, r.mean_Hy_nT)
                                             for r in bad_hy.itertuples()) or "none",
             len(bad_h), "; ".join("%s %.3g nT" % (r.site, r.worst_dH_nT)
                                   for r in bad_h.itertuples()) or "none"))
else:
    print("VERDICT: PASS -- all %d sites rotate to a mean Hy of at most %.3g nT and change |H| by at most "
          "%.3g nT over %d samples; the angles run %.2f to %.2f deg and the declinations %.2f to %.2f deg"
          % (len(fr), fr.mean_Hy_nT.abs().max(), fr.worst_dH_nT.max(), int(fr.samples.sum()),
             pd.to_numeric(fr.angle_deg, errors="coerce").min(),
             pd.to_numeric(fr.angle_deg, errors="coerce").max(),
             pd.to_numeric(fr.declination_deg, errors="coerce").min(),
             pd.to_numeric(fr.declination_deg, errors="coerce").max()))
'''),

("md", r"""## The clean pool

A site may reference another only if its own magnetic record is fit to. The tail scan walks each rotated
record in 600 s chunks and takes the median band power of Hx, Hy, Ex and Ey over 20-200 s
(`auslamp_proc.process.transients.scan`, survey.yaml `pool`), and two tests are read off it.

The event test is fleet-normalised. A chunk is an event where the site's own power exceeds 100 times its own
median AND the rest of the fleet does not see the same thing: either fewer than 3 other sites were recording
then, or this site runs more than 10 times the median of the others. A substorm lifts every site in the
array within the same ten minutes and is signal; a power-cycle or a vehicle lifts one, and only the second is
a fault. The site's own median cannot tell them apart.

The baseline test catches the other failure: a site that never spikes because it is loud all the time. The
denominator is one median over every site with a scan, so the level is a property of the instrument and the
band rather than of whoever was deployed alongside.

The band is 20-200 s and not the 2-20 s the campaign scan reads at 10 Hz. A 1 Hz cache is decimated from
10 Hz, so its 2-20 s band sits on the anti-alias filter's roll-off and reads each unit's own noise floor:
over these 23 fluxgates the 2-20 s chunk medians span five decades, where the 20-200 s medians hold inside a
factor of 10 at 22 of the 23. A baseline that is not comparable between sites cannot be judged against a
survey median.

    clean = event fraction <= EVENT_FRAC_MAX and baseline <= BASELINE_MAX x the survey median

**This check fails if the scan is UNJUDGED at any site, that is if no chunk was scored there.**"""),

("code", '''scan_rows = []
for s in ALL_SITES:                              # <- every site in sites.csv: the pool is not the chosen set
    t = time.time()
    if TR.load_series(s, WORK) is None:
        t0, arrays, meta = cache.load(s, WORK, 1)
        d = sv.decision(s)
        arrays, _a, _u = FR.apply_signs(arrays, d)
        arrays, _ang = FR.rotate_to_mean_field(arrays, regimes=FR.parse_regimes(d.get("rot_regimes")),
                                               drop=FR.parse_regimes(d.get("rot_drop")), fs=1.0)
        row = TR.scan(s, t0, arrays, WORK, fs=1.0, cfg=sv.cfg)
        del arrays
        print("%-9s scanned in %5.1f s" % (s, time.time() - t), flush=True)
    else:
        row = TR.summarise(s, *TR.load_series(s, WORK))
    scan_rows.append(row)

pool, pool_table = ST.clean_pool(force=True)
base = TR.survey_baseline(ALL_SITES, WORK)
scan_t = pd.DataFrame(scan_rows).merge(pool_table, on="site")
print()
print("the survey median baseline is %.4g nT^2/Hz at 20-200 s over %d sites with a scan"
      % (base, len(scan_t)))
print(scan_t[["site", "chunks", "days", "Hx_median", "Hy_median", "event_frac", "baseline", "clean",
              "reason"]].round(4).to_string(index=False))
print()
print("the pool (%d of %d): %s" % (len(pool), len(scan_t), " ".join(pool)))
print("excluded, with the reason:")
for r in pool_table[~pool_table.clean].itertuples():
    print("   %-9s %s" % (r.site, r.reason))

unjudged = pool_table[~pool_table.judged]
print()
if not len(pool_table):
    print("VERDICT: UNJUDGED -- no site was scanned, so the pool scored nothing")
elif len(unjudged):
    print("VERDICT: FAIL -- the scan scored no chunk at %d of %d sites (%s)"
          % (len(unjudged), len(pool_table), ", ".join(unjudged.site)))
else:
    print("VERDICT: PASS -- every one of the %d sites was scored on at least one chunk, the thinnest being "
          "%d chunks at %s; %d sites are clean at events <= %s and baseline <= %sx, and the %d excluded are "
          "each named with the limb they failed"
          % (len(pool_table), int(scan_t.chunks.min()), scan_t.site[scan_t.chunks.idxmin()],
             len(pool), EVENT_FRAC_MAX, BASELINE_MAX, len(pool_table) - len(pool)))
'''),

("md", r"""## The remote site

Every candidate in the target's own overlap group is scored: the distance in km, the usable overlap in days,
its event fraction and baseline from the scan above, and the event-free 20-200 s coherence of the pair. The
coherence is a chunk median and never one pass over the whole record -- one transient inside one Welch
segment otherwise takes a whole record's score to the floor -- and nothing is concatenated: the uncut overlap
is walked in whole 1,024 s segments and a segment is used only where the mask is true right through it, so
the estimator never reads a join it made itself.

A candidate is scored only where its overlap reaches min(MIN_OVERLAP_DAYS, 0.75 x the days the target can
use). The days the target can use, not the length of its cache: a cache is as long as the logger ran, not as
long as the magnetics are sound.

The choice is then the five-branch rule, and the branch is recorded in the product:

1. clean, coh >= COH_MIN, overlap >= 90 per cent -- the nearest of them
2. clean, coh >= COH_MIN -- the longest overlap
3. clean, coh >= COH_RELAX -- the most coherent
4. coh >= COH_MIN but not clean -- the fewest events
5. none of the above -- the nearest by km

Where `decisions.csv` names a remote, that site is used and the rule's own choice is printed beside it. The
two agreeing is a reading and not a criterion: the campaign chose its remotes on a deployment sheet and a
different coherence estimator, and this rule is an independent computation.

**This check fails if any chosen remote has no overlap with its target or is not in the target's overlap
group.**"""),

("code", '''groups = ST.groups()
remote_rows, cand_tables = [], {}
for s in CHOSEN:
    scores = ST.score_candidates(s)
    cand_tables[s] = scores
    ch = ST.remote_site(s, scores=scores)
    ov = ST.overlap(s, ch["name"]) if ch.get("name") else None
    coh = ch.get("coh")
    if coh is None and ch.get("name") and ov is not None:
        # a remote decisions.csv names from outside the clean pool is not in the candidate table, so it
        # carries no score there; it is measured here on the same rule so the column is never empty
        coh = round(float(ST.pair_coherence(s, ch["name"])["coh"]), 3)
    remote_rows.append(dict(site=s, campaign_remote=str(sv.decision(s).get("remote_site", "")),
                            rule_remote=ch.get("rule_name"), rule_branch=ch.get("rule_branch"),
                            rule_coh=ch.get("rule_coh"), chosen=ch.get("name"),
                            source=ch.get("source"), coh=coh,
                            overlap_days=(round((ov[1] - ov[0]) / 86400.0, 2) if ov else 0.0),
                            group=groups.get(s, ""), remote_group=groups.get(ch.get("name"), ""),
                            n_candidates=len(ch.get("candidates", [])), reason=ch.get("reason", "")[:150]))
rem = pd.DataFrame(remote_rows)
print("the candidates of %s, the first of the chosen set" % CHOSEN[0])
print(cand_tables[CHOSEN[0]].round(4).to_string(index=False))
print()
print(rem[["site", "campaign_remote", "rule_remote", "rule_branch", "rule_coh", "chosen", "source",
           "coh", "overlap_days", "group", "remote_group"]].to_string(index=False))
print()
differ = rem[(rem.campaign_remote.str.lower() != "decide") & (rem.campaign_remote != rem.rule_remote)]
print("the campaign's remote and the rule's choice differ at %d of %d sites (a reading, not a criterion):"
      % (len(differ), len(rem)))
for r in differ.itertuples():
    print("   %-9s campaign %-8s rule %-8s (branch %s, coh %s)"
          % (r.site, r.campaign_remote, r.rule_remote, r.rule_branch, r.rule_coh))
print()
print("the branch the rule took, over the chosen set")
print(rem.rule_branch.value_counts().sort_index().to_string())
print()
for r in rem.itertuples():
    print("   %-9s %s" % (r.site, r.reason))

no_ov = rem[rem.overlap_days <= 0]
wrong_group = rem[(rem.group != "") & (rem.group != rem.remote_group)]
print()
if not len(rem):
    print("VERDICT: UNJUDGED -- no site was offered a remote")
elif len(no_ov) or len(wrong_group) or rem.chosen.isna().any():
    print("VERDICT: FAIL -- %d of %d chosen remotes do not overlap their target (%s); %d sit outside the "
          "target's overlap group (%s); %d sites were offered no remote at all (%s)"
          % (len(no_ov), len(rem), ", ".join(no_ov.site) or "none",
             len(wrong_group), "; ".join("%s -> %s (%s vs %s)" % (r.site, r.chosen, r.group, r.remote_group)
                                         for r in wrong_group.itertuples()) or "none",
             int(rem.chosen.isna().sum()), ", ".join(rem.site[rem.chosen.isna()]) or "none"))
else:
    print("VERDICT: PASS -- all %d chosen remotes overlap their target, the shortest by %.2f d at %s, and "
          "every one sits in the target's own overlap group; %d came from decisions.csv and %d from the rule"
          % (len(rem), rem.overlap_days.min(), rem.site[rem.overlap_days.idxmin()],
             int((rem.source == "decisions.csv").sum()), int((rem.source != "decisions.csv").sum())))
'''),

("md", r"""## The fleet stack

A member's weight is its median coherence with the FLEET at 100-1000 s, and never its coherence with the
target. Whether a reference is a good measurement of the regional field is a question about the reference
and the field; weighting a member by how well it agrees with the target puts the target's own noise into
its reference (Ben's rule, 2026-09-06). The fleet table below is one median per member over its pairs with
the rest of the pool, and it is the same table for every target.

Members below STACK_CUTOFF are refused, the best STACK_MAX are kept, and a stack with fewer than STACK_MIN
members is refused outright: a one-member stack is a remote site renamed.

Each member is aligned first. The lag is the median 5-20 s cross-correlation over up to eight event-free
two-day windows spread across the record, and it is accepted only where it is one constant, spread at most
1.0 s: no single shift fixes a free-running clock. A member whose lag cannot be measured or is not one
constant is kept at lag 0 with the reason recorded, because these are GPS-disciplined loggers.

Each member is then demeaned over its own finite samples, and a sample where a member is NaN does not add to
that member's weight there, so the mean is over whoever is sound and the weights renormalise per sample.
Where no member is sound the stack is zero and its mask is False: a zero reference contributes nothing to
either the cross- or the auto-spectrum, so those windows drop out of the estimate instead of biasing it.

**This check fails if any stack carries a member weighted by its coherence with the TARGET rather than the
fleet, or a member outside the pool.** The first limb is scored by recomputing one site's weights from the
fleet coherence table printed above and comparing them with the weights in that site's own sidecar, and by
comparing the same weights against the target coherences of the candidate table: a weight that matches the
target column and not the fleet column is the wrong rule."""),

("code", '''weights, pairs = ST.fleet_weights()
fleet = pd.DataFrame([dict(member=k, fleet_coh_100_1000s=v,
                           pairs=sum(1 for p in pairs if k in p.split(":")))
                      for k, v in sorted(weights.items())])
print("the fleet coherence of every pool member at 100-1000 s, the median over its pairs with the rest")
print(fleet.to_string(index=False))
print()
print("the pairs it is the median of")
pf = pd.DataFrame([dict(pair=k, coh=v["coh"], chunks=v["chunks"]) for k, v in sorted(pairs.items())])
print(pf.to_string(index=False))

stack_rows, member_rows, stacks = [], [], {}
for s in CHOSEN:
    kept, lags, refused, notes = ST.stack_members(s)
    stacks[s] = (kept, lags, refused, notes)
    stack_rows.append(dict(site=s, n_members=len(kept),
                           members=" ".join(kept), weights=" ".join("%.3f" % v for v in kept.values()),
                           lags_s=" ".join("%+.2f" % lags.get(d, 0.0) for d in kept),
                           refused=len(refused), notes=len(notes)))
    for d, w in kept.items():
        member_rows.append(dict(site=s, member=d, weight=w, lag_s=lags.get(d, 0.0),
                                note=notes.get(d, ""), in_pool=d in pool))
    for d, why in sorted(refused.items()):
        member_rows.append(dict(site=s, member=d, weight=None, lag_s=None, note=why, in_pool=d in pool))
sk = pd.DataFrame(stack_rows)
mem = pd.DataFrame(member_rows)
print()
print(sk.to_string(index=False))
print()
print("every member and every refusal, per site")
print(mem.to_string(index=False))
print()
refused_sites = sk[sk.n_members < STACK_MIN]
print("stacks refused for want of members: %s"
      % (", ".join("%s (%d < %d)" % (r.site, r.n_members, STACK_MIN)
                   for r in refused_sites.itertuples()) or "none"))

# limb A, recomputed: the weight of every member has to be the fleet value and not the target value
probe = CHOSEN[0]
kept = stacks[probe][0]
cand = cand_tables[probe].set_index("name")
recomputed = {d: weights.get(d) for d in kept}
target_coh = {d: (float(cand.loc[d, "coh"]) if d in cand.index and cand.loc[d, "coh"] is not None
                  else None) for d in kept}
print()
print("%s: the weight in the store against the fleet table and against this site's own target coherence"
      % probe)
print(pd.DataFrame([dict(member=d, in_store=kept[d], fleet_table=recomputed[d], target_coh=target_coh[d])
                    for d in kept]).to_string(index=False))
wrong_rule = [d for d in kept if recomputed[d] is None or abs(kept[d] - recomputed[d]) > 1e-6]
looks_like_target = [d for d in kept if target_coh[d] is not None and abs(kept[d] - target_coh[d]) < 1e-6
                     and (recomputed[d] is None or abs(target_coh[d] - recomputed[d]) > 1e-6)]
outside = sorted(set(mem.member[(mem.weight.notna()) & (~mem.in_pool)]))
scored = int(mem.weight.notna().sum())
print()
if not scored:
    print("VERDICT: UNJUDGED -- no stack carries a member, so no weight was scored")
elif wrong_rule or looks_like_target or outside:
    print("VERDICT: FAIL -- %d of %s's %d members do not carry the fleet weight (%s); %d carry the target "
          "coherence instead (%s); %d members over all %d stacks sit outside the clean pool (%s)"
          % (len(wrong_rule), probe, len(kept), ", ".join(wrong_rule) or "none",
             len(looks_like_target), ", ".join(looks_like_target) or "none",
             len(outside), len(sk), ", ".join(outside) or "none"))
else:
    print("VERDICT: PASS -- all %d of %s's members carry the fleet coherence recomputed from the table "
          "above to within 1e-6 and none carries its target coherence (which differs by %.3f to %.3f), and "
          "every one of the %d members over %d stacks is in the %d-site clean pool"
          % (len(kept), probe, min(abs(kept[d] - target_coh[d]) for d in kept if target_coh[d] is not None),
             max(abs(kept[d] - target_coh[d]) for d in kept if target_coh[d] is not None),
             scored, len(sk), len(pool)))
'''),

("md", r"""## The observatory

The INTERMAGNET one-second record of the survey's own observatory, read out of the archive over each site's
window and turned into the mean-field frame of that window like any other member. It is never shifted: a
delay on a single reference cancels exactly in Z, so shifting it would change nothing and leaving it alone
keeps that invariance available as a check.

Its weight in the stack + observatory is measured on the same rule as a site member's -- the median
coherence with the pool at 100-1000 s -- so it enters at a weight on the members' own scale rather than at
its agreement with any one target.

The coherence of each site with the observatory is a reading. It says how much of the site's long-period
field the observatory shares from several hundred kilometres away, which is what the observatory kind can
buy and what it cannot: at the short end a distant observatory shares almost nothing."""),

("code", '''w_obs, obs_pairs = ST.observatory_weight(OBS)
print("observatory %s, %s" % (OBS, geo.OBSERVATORIES.get(OBS, ("", 0, 0))[0]))
print("its fleet weight at 100-1000 s is %s, the median over %d pool members"
      % (w_obs, sum(1 for v in obs_pairs.values() if v is not None)))
print()
obs_rows = []
for s in CHOSEN:
    p = ST.path("obs", s).with_suffix(".json")
    km = geo.distance_km(ST.position(s), OBS)
    # a site outside the pool has no pair in the weight table; it is measured here so the reading covers
    # every site this run processes
    c = obs_pairs.get(s)
    chunks = None
    if c is None:
        r = ST.site_observatory_coh(s, OBS)
        c = None if not np.isfinite(r["coh"]) else round(float(r["coh"]), 4)
        chunks = r["chunks"]
    obs_rows.append(dict(site=s, km=round(km, 1), coh_100_1000s=c, in_pool=s in pool,
                         chunks=chunks, built=p.exists()))
ob = pd.DataFrame(obs_rows)
print(ob.to_string(index=False))
print()
print("stack + observatory: the stack's members with %s added at weight %s and lag 0" % (OBS, w_obs))
print(pd.DataFrame([dict(site=s, members=" ".join(list(stacks[s][0]) + ([OBS] if w_obs else [])),
                         n=len(stacks[s][0]) + (1 if w_obs else 0)) for s in CHOSEN]).to_string(index=False))
'''),

("md", r"""## The references written to the store

Every reference is written once per rate to `<work_root>/references/<rate>hz/<kind>_<site>.npz`, with a
sidecar naming its members, their weights and lags, the refusals with their reasons, the frame and the time
it was built. A pass reads the store; it never rebuilds a reference of its own, so two products of the same
kind are built on the same array.

A 10 Hz store re-reads the 1 Hz specification -- the same pool, the same remote, the same members, weights
and lags -- on the 10 Hz grid after a spike screen. The decisions are made where they can be measured."""),

("code", '''built = {}
for rate in RATES:
    st = ST if rate == RATES[0] else store(rate)
    spec = None
    if rate != 1:
        spec = store(1)                     # <- the 10 Hz store takes its decisions from the 1 Hz one
    t = time.time()
    kinds = tuple(k for k in KINDS if k in REF.KINDS_WITH_STORE)
    got = REF.build_store(sv, CHOSEN, rate=rate, kinds=kinds, verbose=False, spec_store=spec, store=st)
    built[rate] = got
    print("%d Hz: %d sites, %d references, %.1f s" % (rate, len(got),
                                                      sum(len(v) for v in got.values()), time.time() - t))
    rows = []
    for s, d in sorted(got.items()):
        for k, info in sorted(d.items()):
            rows.append(dict(site=s, kind=k, coverage=info.get("coverage"),
                             members=" ".join(m["name"] for m in (info.get("members") or [])
                                              if m.get("role") != "refused"),
                             error=str(info.get("error", ""))[:90]))
    print(pd.DataFrame(rows).to_string(index=False))
'''),

("md", r"""## The bands and the parameter set

The band file's lines are FFT harmonics of the **window**, not of the record, so a file read at another
window length names different periods and nothing says so. File, level count and window are therefore one
object in the package, `aurora_run.BANDS`, and the tables below are read through Aurora's own band machinery
rather than off the text file.

The cascade is `[1] + [4] * (levels - 1)`: the rate falls by four at each level after the first.

Under it, what the transient mask and the hour floor cost each site. Aurora gets one run per kept stretch of
at least 3,600 s; a shorter stretch holds no complete window at the deep levels and is dropped. Two events
forty minutes apart therefore cost the forty minutes between them as well as themselves, and that second
loss is measured here rather than hidden inside the mask."""),

("code", '''tabs = {}
for key, bs in aurora_run.BANDS.items():
    tabs[key] = aurora_run.band_table(bs)
    print("%-5s %-22s %2d bands over %d levels, window %d samples, centres %.3f to %.1f s"
          % (key, bs.file.name, len(tabs[key]), tabs[key].level.nunique(), bs.window,
             tabs[key].centre_s.min(), tabs[key].centre_s.max()))
print()
key = "%dhz" % RATES[0]
print("the %s table, as Aurora computes it" % key)
print(tabs[key].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, (k, tab) in zip(axes, sorted(tabs.items())):
    for r in tab.itertuples():
        ax.plot([r.lower_s, r.upper_s], [r.level, r.level], lw=6, solid_capstyle="butt",
                color="C%d" % (r.level % 10), alpha=0.8)
        ax.plot(r.centre_s, r.level, "k|", ms=8)
    ax.set(xscale="log", xlabel="period (s)", ylabel="decimation level",
           title="%s: %s, %d bands" % (k, aurora_run.BANDS[k].file.name, len(tab)),
           yticks=range(tab.level.nunique()))
    ax.grid(alpha=0.25, which="both")
fig.tight_layout()
bands_png = OUT / "03_bands.png"
fig.savefig(bands_png, dpi=110)
plt.close(fig)
WRITTEN.append(bands_png)
display(Image(filename=str(bands_png)))

print()
print("the Aurora parameter sets the package ships; %s is this run's" % PARAMS)
for name, p in sorted(aurora_run.AURORA_PARAMS.items()):
    print("   %-14s %s" % (name, ", ".join("%s=%s" % kv for kv in p.items())))
print()
print("what the mask and the %g s run floor cost each site at %g Hz" % (TR.MIN_SEGMENT_S, RATES[0]))
mask_rows = []
for s in CHOSEN:
    t0, arrays, meta = cache.load(s, WORK, RATES[0])
    d = sv.decision(s)
    arrays, _a, _u = FR.apply_signs(arrays, d)
    arrays, _ang = FR.rotate_to_mean_field(arrays, regimes=FR.parse_regimes(d.get("rot_regimes")),
                                           drop=FR.parse_regimes(d.get("rot_drop")), fs=float(RATES[0]))
    ev = TR.site_events(s, ALL_SITES, WORK, sv.cfg)
    ee = TR.e_events(s, WORK, sv.cfg)
    keep, stats = TR.build_keep(t0, {c: arrays[c] for c in mth5_build.LOCAL_CHANNELS}, float(RATES[0]),
                                ev, (), ee)
    runs = TR.segments(keep, int(TR.MIN_SEGMENT_S * RATES[0]))
    mask_rows.append(dict(site=s, days=round(stats["n"] / 86400.0 / RATES[0], 2),
                          finite_frac=stats["finite_frac"], H_intervals=len(ev), E_bursts=len(ee),
                          kept_frac=stats["kept_frac"], runs=len(runs),
                          floor_loss_frac=round(TR.floor_dropped_frac(keep, float(RATES[0])), 5),
                          longest_run_d=round(max((L for _o, L in runs), default=0) / 86400.0 / RATES[0], 2)))
    del arrays, keep
    print("   %-9s %s" % (s, mask_rows[-1]), flush=True)
mk = pd.DataFrame(mask_rows)
print()
print(mk.to_string(index=False))
'''),

("md", r"""## The MTH5

One site and one kind, written and read back. The MTH5 is what Aurora reads: the local station with its five
channels and, where there is a reference, a second station carrying hx and hy only on the same grid and with
the same run ids, so the kernel dataset's run-interval intersection pairs them one to one.

The channels carry nanoTesla and milliVolt per kilometer and no filter at all. The cache is already in
physical units, so a filter here would be applied a second time; the spelling of the units matters too,
because `millivolts per kilometer` resolves to `unknown per kilometer` in mt_metadata and writes the channel
with no unit.

**This check fails if the MTH5 differs from the cache sample for sample, or if its run count differs from the
kept stretches.** Every sample of every channel of every run is compared with the array it was written from.
The bound is 1e-3 nT and mV/km, which is the file's own float32 resolution at these levels."""),

("code", '''probe_site = CHOSEN[0]                  # <- the first of the chosen set; any site answers the same question
probe_kind = "remote" if "remote" in KINDS else KINDS[0]
rate = RATES[0]
t0, arrays, meta = cache.load(probe_site, WORK, rate)
d = sv.decision(probe_site)
arrays, applied, undecided = FR.apply_signs(arrays, d)
arrays, ang = FR.rotate_to_mean_field(arrays, regimes=FR.parse_regimes(d.get("rot_regimes")),
                                      drop=FR.parse_regimes(d.get("rot_drop")), fs=float(rate))
local = {c: np.asarray(arrays[c], float) for c in mth5_build.LOCAL_CHANNELS}
rt0, rh, rmask, info = REF.load_reference(probe_kind, probe_site, rate, WORK)
ev = TR.site_events(probe_site, ALL_SITES, WORK, sv.cfg)
ee = TR.e_events(probe_site, WORK, sv.cfg)
ev_rem = [(pd.Timestamp(a).timestamp(), pd.Timestamp(b).timestamp())
          for a, b in (info.get("remote_events") or [])] if probe_kind == "remote" else []
keep, stats = TR.build_keep(t0, local, float(rate), ev, ev_rem, ee,
                            remote_mask=(None if rmask is None else rmask))
rid = REF.reference_station_id(probe_kind, info, OBS)
scratch = WORK / probe_site / "_mth5_check"
h5 = scratch / ("%s_%s.h5" % (probe_site, probe_kind))
t = time.time()
_p, segs = mth5_build.write_h5(h5, probe_site, local, (None if rh is None else (rid, rh)), t0, float(rate),
                               sv.cfg["name"], sv.site(probe_site), keep=keep,
                               reference_row=(sv.site(info["remote"]) if probe_kind == "remote" else None))
print("%s %s: %s written in %.1f s, %.1f MB" % (probe_site, probe_kind, h5.name, time.time() - t,
                                                h5.stat().st_size / 2 ** 20))
print("   stations  %s and %s" % (probe_site, rid or "none (single station)"))
print("   mask      keeps %.2f %% of %d samples; %d runs of at least %g s; the hour floor loses %.2f %% more"
      % (100 * stats["kept_frac"], stats["n"], len(segs), TR.MIN_SEGMENT_S,
         100 * TR.floor_dropped_frac(keep, float(rate))))
print("   runs      %s" % ", ".join("%03d %.2f d" % (i + 1, L / 86400.0 / rate)
                                    for i, (_o, L) in enumerate(segs[:12])))
back = mth5_build.read_back(h5, probe_site, local, segs, sv.cfg["name"])
print("   read back %s" % back)
n_kept_runs = len(TR.segments(keep, int(TR.MIN_SEGMENT_S * rate)))
left = mth5_build.remove(h5)          # a handle HDF5 has not released is said, not raised
if left:
    print("   %s" % left)
import shutil; shutil.rmtree(scratch, ignore_errors=True)
del arrays, local, rh

print()
if not back["samples_compared"]:
    print("VERDICT: UNJUDGED -- the MTH5 carried no sample to compare")
elif back["problems"] or back["runs_in_file"] != n_kept_runs:
    print("VERDICT: FAIL -- %d channel-runs differ from the cache (%s); the file holds %d runs where the "
          "mask leaves %d kept stretches of at least %g s"
          % (len(back["problems"]), "; ".join(back["problems"][:6]), back["runs_in_file"], n_kept_runs,
             TR.MIN_SEGMENT_S))
else:
    print("VERDICT: PASS -- %s's %s MTH5 carries all %d samples of its 5 channels over %d runs to within "
          "%.3g nT and mV/km of the rotated cache, and its run count is the %d kept stretches of at least "
          "%g s the mask leaves"
          % (probe_site, probe_kind, back["samples_compared"], back["runs_in_file"],
             back["worst_difference"], n_kept_runs, TR.MIN_SEGMENT_S))
'''),

("md", r"""## The run

One subprocess per site, `LANES` of them at a time, each running
`python -m auslamp_proc.process.run` over that site's kinds and rates. A lane pins its BLAS threads to three:
a lane that takes every core makes three lanes slower than one, and the memory a pass peaks at is per lane.
Measured over the 115 products of AusLAMP Queensland Phase 1 on 2026-09-16, whose records run 12-62 days, a
1 Hz pass peaks at 2.5-4.8 GB and takes 140-912 s; the 15 products of the 10 Hz pass over the same records
peak at 22-37 GB and take 420-912 s. That is why `LANES` is 3 at 1 Hz and 1 at 10 Hz: three 10 Hz lanes
would want more than 100 GB.

The CLI is resumable by design. With `REDO` False a product whose EDI is already on disk is left alone and
reported as `exists`, so a run stopped part way is finished by re-running this cell. A product that fails is
caught, its error goes into its own ledger row, and the next product runs.

Each product writes `<site>_<kind>_<rate>hz_<params>.edi` and `.xml` into `<site>/<RUN>_<stamp>/`, beside
`log.txt` and `provenance.json`, and appends one row to `<work_root>/survey/runs.csv`.

**This check fails if any requested product is missing or FAILED in the ledger, or if any product lacks a
tipper, or if any provenance.json lacks the decisions.csv row it used.** The tipper is the third limb because
it comes out of the same pass as the impedance -- hz is in the local station -- so a product without one was
run on a record missing its vertical channel."""),

("code", '''import subprocess
import concurrent.futures as cf

ENV = dict(os.environ)
ENV.update(OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")   # <- 3 threads a lane

def one_site(site):
    cmd = [sys.executable, "-m", "auslamp_proc.process.run", "--survey", SURVEY, "--site", site,
           "--run", RUN, "--stamp", STAMP, "--params", PARAMS, "--quiet",
           "--kinds"] + list(KINDS) + ["--rates"] + [str(r) for r in RATES]
    if REDO:
        cmd.append("--redo")
    if WORK_ROOT:
        cmd += ["--work-root", str(WORK_ROOT)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=ENV)
    return site, r.returncode, round(time.time() - t, 1), (r.stdout or "").strip()[-300:], \\
        (r.stderr or "").strip()[-400:]

t_run = time.time()
print("%d site(s) in %d lane(s), %d kind(s) at %s Hz -> %s_%s"
      % (len(CHOSEN), LANES, len(KINDS), ", ".join(str(r) for r in RATES), RUN, STAMP), flush=True)
done = 0
with cf.ThreadPoolExecutor(max_workers=LANES) as ex:
    for site, rc, secs, out, err in ex.map(one_site, CHOSEN):
        done += 1
        print("%2d/%2d %-9s exit %d %7.1f s  %s%s" % (done, len(CHOSEN), site, rc, secs, out,
                                                      ("  || " + err) if rc else ""), flush=True)
wall = time.time() - t_run
print()
print("wall time %.1f min over %d sites in %d lanes; a resumed run finds its products on disk and this is "
      "the time to check them, not the time they cost" % (wall / 60.0, len(CHOSEN), LANES))

ledger = pd.read_csv(WORK / "survey" / "runs.csv")
led = ledger[(ledger["stamp"].astype(str) == STAMP) & (ledger.site.isin(CHOSEN))].copy()
# runs.csv is append-only, so re-running this workbook adds an `exists` row beside the `made` row of the
# same product. The row that says what a product cost is the one written when it was made, so that row
# wins and one row per product is kept.
led["_made"] = (led.status == "made").astype(int)
led = (led.sort_values(["site", "kind", "rate_hz", "_made"])
       .drop_duplicates(["site", "kind", "rate_hz"], keep="last").drop(columns="_made"))
print()
print(led[["site", "kind", "rate_hz", "params", "remote", "n_runs", "mask_dropped_frac",
           "floor_dropped_frac", "seconds", "peak_rss_mb", "status", "error"]].to_string(index=False))
print()
print("per kind")
print(led.groupby("kind").agg(products=("status", "size"), made=("status", lambda v: int((v == "made").sum())),
                              failed=("status", lambda v: int((v == "FAILED").sum())),
                              seconds=("seconds", "median"),
                              peak_rss_mb=("peak_rss_mb", "max")).to_string())

want = [(s, k, float(r)) for s in CHOSEN for k in KINDS for r in RATES]
have = {(r.site, r.kind, float(r.rate_hz)): r for r in led.itertuples()}
missing = [w for w in want if w not in have]
failed = led[led.status == "FAILED"]
no_tipper, no_prov = [], []
for w in want:
    r = have.get(w)
    if r is None or str(r.status) == "FAILED":
        continue
    p = Path(str(r.edi))
    if not p.exists():
        missing.append(w)
        continue
    if not EDI.has_tipper(p):
        no_tipper.append("%s %s %g Hz" % w)
for s in sorted({w[0] for w in want}):
    pj = WORK / s / ("%s_%s" % (RUN, STAMP)) / "provenance.json"
    d = PROV.read(pj)
    if not d or not d.get("decisions_row") or not d["decisions_row"].get("site"):
        no_prov.append(s)
print()
if not want:
    print("VERDICT: UNJUDGED -- no product was requested")
elif missing or len(failed) or no_tipper or no_prov:
    print("VERDICT: FAIL -- %d of %d requested products are missing (%s); %d are FAILED in the ledger (%s); "
          "%d carry no tipper (%s); %d of %d provenance files lack the decisions.csv row (%s)"
          % (len(missing), len(want), "; ".join("%s %s %g Hz" % m for m in missing[:8]) or "none",
             len(failed), "; ".join("%s %s: %s" % (r.site, r.kind, str(r.error)[:80])
                                    for r in failed.itertuples()) or "none",
             len(no_tipper), "; ".join(no_tipper[:8]) or "none",
             len(no_prov), len(CHOSEN), ", ".join(no_prov) or "none"))
else:
    print("VERDICT: PASS -- all %d requested products exist and none is FAILED (%d made, %d already on "
          "disk), every one carries a tipper, and all %d provenance files name the decisions.csv row they "
          "used; this execution took %.1f min, the products cost %.1f machine-minutes between them, and the "
          "largest peak was %.0f MB in one lane"
          % (len(want), int((led.status == "made").sum()), int((led.status == "exists").sum()),
             len(CHOSEN), wall / 60.0, led.seconds.sum() / 60.0, led.peak_rss_mb.max()))
'''),

("md", r"""## A first look

One line per site and kind at 100-1000 s: the two apparent resistivities, the two phases and the tipper
magnitude, read straight out of the EDI. Under it, the single station against the remote site as a ratio at
10-100 s and at 100-1000 s, which is what the reference bought at each end.

This is a reading and not a check. A single station biased down by H noise coherent with itself shows as a
ratio below one; a remote that shares the target's noise shows as a ratio near one where the single station
is known to be biased. The full figures, every product on one page, are workbook 04's."""),

("code", '''from mt_metadata.transfer_functions.core import TF

def read_product(path):
    tf = TF(fn=str(path))
    tf.read()
    p = np.asarray(tf.period, float)
    z = np.asarray(tf.impedance, complex)
    t = np.asarray(tf.tipper, complex) if tf.tipper is not None else None
    return p, z, t

def band_read(p, z, t, lo, hi):
    m = (p >= lo) & (p <= hi)
    if not m.any():
        return {}
    out = {}
    for name, (i, j) in (("xy", (0, 1)), ("yx", (1, 0))):
        zz = z[m, i, j]
        good = np.isfinite(zz) & (np.abs(zz) > 0)
        if not good.any():
            continue
        rho = 0.2 * p[m][good] * np.abs(zz[good]) ** 2
        ph = np.degrees(np.angle(zz[good]))
        out["rho_%s" % name] = float(np.median(rho))
        out["phase_%s" % name] = float(np.median(ph))
    if t is not None and np.isfinite(t[m]).any():
        out["tipper"] = float(np.nanmedian(np.abs(t[m]).max(axis=-1)))
    return out

look_rows = []
for r in led.itertuples():
    p_ = Path(str(r.edi))
    if not p_.exists():
        continue
    try:
        per, z, tp = read_product(p_)
        row = dict(site=r.site, kind=r.kind, rate_hz=r.rate_hz, periods=len(per),
                   shortest_s=round(float(np.min(per)), 3), longest_s=round(float(np.max(per)), 1))
        row.update({k: round(v, 3) for k, v in band_read(per, z, tp, 100.0, 1000.0).items()})
        for k, v in band_read(per, z, tp, 10.0, 100.0).items():
            row["%s_10_100" % k] = round(v, 3)
    except Exception as exc:
        row = dict(site=r.site, kind=r.kind, rate_hz=r.rate_hz,
                   note="%s: %s" % (type(exc).__name__, str(exc)[:60]))
    look_rows.append(row)
first = pd.DataFrame(look_rows)
print("the reading at 100-1000 s, per site and kind")
print(first.to_string(index=False))

if "single" in KINDS and "remote" in KINDS:
    piv = first.pivot_table(index="site", columns="kind",
                            values=["rho_xy", "rho_yx", "rho_xy_10_100", "rho_yx_10_100"])
    ratio_rows = []
    for s in sorted(set(first.site)):
        row = dict(site=s)
        for col, label in (("rho_xy_10_100", "xy 10-100 s"), ("rho_yx_10_100", "yx 10-100 s"),
                           ("rho_xy", "xy 100-1000 s"), ("rho_yx", "yx 100-1000 s")):
            try:
                a = piv.loc[s, (col, "single")]
                b = piv.loc[s, (col, "remote")]
                row[label] = round(float(a / b), 3) if np.isfinite(a) and np.isfinite(b) and b else np.nan
            except KeyError:
                row[label] = np.nan
        ratio_rows.append(row)
    print()
    print("single station over remote site, as a ratio of apparent resistivity")
    print(pd.DataFrame(ratio_rows).to_string(index=False))
'''),

("md", r"""## What was written"""),

("code", '''rows = []
run_dirs = [WORK / s / ("%s_%s" % (RUN, STAMP)) for s in CHOSEN]
for d in run_dirs:
    for p in sorted(d.glob("*")):
        if p.is_file():
            rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for rate in RATES:
    for p in sorted((WORK / "references" / ("%dhz" % rate)).glob("*")):
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for p in sorted(TR.tails_dir(WORK).glob("*.npz")):
    rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for p in list(WRITTEN) + [WORK / "survey" / "runs.csv"]:
    if Path(p).exists():
        rows.append(dict(file=str(p), kb=round(Path(p).stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d files, %.1f MB" % (len(files), files.kb.sum() / 1024))
print(files.head(60).to_string(index=False))
if len(files) > 60:
    print("   ... and %d more" % (len(files) - 60))
print()
d = run_dirs[0]
print("the run folder of %s" % CHOSEN[0])
for p in sorted(d.glob("*")):
    print("   %-52s %8.1f KB" % (p.name, p.stat().st_size / 1024))
print()
print("the first 30 processing_parameters lines of one EDI")
one = sorted(d.glob("*.edi"))
if one:
    txt = one[0].read_text(encoding="utf-8", errors="ignore")
    keep_lines = [ln.strip() for ln in txt.splitlines()
                  if "=" in ln and not ln.strip().startswith(">")]
    print(one[0].name)
    for ln in keep_lines[:30]:
        print("   %s" % ln[:160])
'''),
]


# ===================================================================== 04 the products

WB04_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
SITES = "all"                 # "all" | "largest" (the register's largest group) | a group name | ["Q49", "Q50"]
RUNS = "latest"               # "latest" = the newest stamp of every run name | "all" | ["first", "short10"]
KINDS = "all"                 # "all" | a list of the code keys: ["remote", "stack", "obs"]
RATES = "all"                 # "all" | [1] | [10]; both rates of a site sit on the same page
COMPARE = "all"               # "all" | a list of the survey.yaml source names | "none" draws no comparison
SHOW = None                   # None = the first chosen site; a site name shows that site's page inline
PER_PAGE = 6                  # sites a gallery page
WORK_ROOT = None              # None = survey.yaml work_root; every figure and table lands under it
'''

WB04_RULES = '''# ---- the agreement rule: a change here changes which pairs are called agreement ----
PERIOD_RANGE = (1, 50000)     # the periods drawn and scored, in s
AGREE_RHO = 0.20              # two curves agree where the rho ratio is within this fraction of one
AGREE_PHASE_DEG = 5.0         # ... and the phase difference is within this many degrees
AGREE_BAND = (5, 200)         # the band the agreement rule is read over, in s
BANDS = [(5, 10), (10, 100), (100, 1000), (1000, 10000)]   # the decades every table reports, in s
'''

WB04_SETUP = '''import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

# mt_metadata logs a warning per channel while an EDI is read, and this workbook reads several hundred
from loguru import logger as _loguru
_loguru.remove()

from auslamp_proc import agreement as AG, products as PR, readings as RD, survey as SV
from auslamp_proc.process import KIND_WORD
from auslamp_proc.figures import products as FIG

pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 600)

T0 = time.time()
sv = SV.load_survey(SURVEY)
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
WORK = Path(sv.cfg["work_root"])
OUT = WORK / "survey"
OUT.mkdir(parents=True, exist_ok=True)
WRITTEN = []

ASKED, WHY = SV.select_sites(sv, SITES, 0)
PROD, IGNORED = RD.deliverable(PR.find_products(sv, ASKED, runs=RUNS, kinds=KINDS, rates=RATES))
CHOSEN = [s for s in ASKED if s in set(PROD.site)]
NO_PRODUCT = [s for s in ASKED if s not in set(PROD.site)]
PAIRS = PR.choose_runs(PR.ledger(WORK), RUNS)
RUN_TAG = "-".join(sorted({name for name, _stamp in PAIRS})) or "none"
SHOW_SITE = SHOW or (CHOSEN[0] if CHOSEN else "")

def _dec(cell):
    try:
        return float(str(cell).strip())
    except ValueError:
        return None

DECLINATION = {r.site: _dec(r.declination_deg) for r in sv.sites.itertuples()}
DECLINATION = {k: v for k, v in DECLINATION.items() if v is not None and np.isfinite(v)}

_cache = {}
def read(path):
    """One TFData per file. A page, four tables and the gallery all ask for the same file."""
    if path not in _cache:
        _cache[path] = PR.read_tf(path)
    return _cache[path]

print("survey       %s" % sv.cfg["name"])
print("work root    %s" % WORK)
print("sites        %d of %d asked for carry a product (%s)" % (len(CHOSEN), len(ASKED), WHY))
print("             %s" % " ".join(CHOSEN))
if NO_PRODUCT:
    print("no product   %s -- run workbook 03 over them" % " ".join(NO_PRODUCT))
for name, stamp in PAIRS:
    n = int(((PROD.run == name) & (PROD.stamp == stamp)).sum())
    rates = sorted(set(PROD[PROD.run == name].rate_hz))
    print("run          %-9s %s  %3d product(s) at %s Hz over %d site(s)"
          % (name, stamp, n, ", ".join("%g" % r for r in rates),
             PROD[PROD.run == name].site.nunique()))
print("products     %d rows, %d on disk" % (len(PROD), int(PROD.on_disk.sum())))
print("kinds        %s" % ", ".join("%s (%s)" % (KIND_WORD[k], k) for k in RD.KINDS))
print("ignored      %d %s product(s) on disk: read by nothing in this workbook"
      % (len(IGNORED), RD.DROPPED_KIND))
print("periods      %g to %g s drawn and scored" % PERIOD_RANGE)
print("agreement    within %.0f %% in rho and %.1f deg in phase over %g-%g s"
      % (100 * AGREE_RHO, AGREE_PHASE_DEG, AGREE_BAND[0], AGREE_BAND[1]))
'''

WB04 = [
("md", r"""# 04 -- The products: every product on one page, run against run, the comparisons last

This workbook reads the transfer functions workbook 03 wrote and puts every product a site has on one page.
It reports what the products say about each other -- kind against kind, rate against rate, run against run --
and only then, in the last section, sets them beside a processing done outside this run.

Nothing here is estimated again and no product is altered. Every file is read through
`auslamp_proc.products.read_tf`, which applies two rules before a curve is used: the EDI empty-data value
1e32 is masked component by component, together with the no-information convention of a zero impedance
carrying an error of 1e9, so a fill never enters a median; and the periods are sorted and duplicates dropped
before any interpolation, because some writers emit them unsorted and interpolation on an unsorted grid is
silently wrong. The yx phase is folded into the first quadrant by +180 deg in
`auslamp_proc.products.rho_phase`, which is what every panel and table below reads.

The four kinds are the same words the earlier workbooks use, and each is one product of the same record:

| word | what it is | code key |
|---|---|---|
| remote site | one other site's H as the reference | `remote` |
| fleet stack | a coherence-weighted mean of several sites' H | `stack` |
| observatory | an INTERMAGNET one-second record as the reference | `obs` |
| stack + observatory | the stack with the observatory as a member | `stack_obs` |

The single-station estimate is not one of them. It is biased low by whatever noise sits in the site's own H
and its error bars do not carry that bias, so it is not a kind of this package: a single-station file left in
a run folder by an earlier pass is counted and named in the first cell, and no table or figure below reads it.

Four checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported FAILED and is
not revised afterwards.

The last section is a comparison and not a test of the truth. Two independent processings of the same field
are two measurements, neither an oracle; the section is a shape check, and a difference in level is a gain, a
dipole length or a frame before it is the earth. Every source outside this run declares the frame its tensors
are in -- geomagnetic, geographic or instrument -- in `surveys/<SURVEY>/survey.yaml`, and the workbook refuses
a source that declares none: a tensor drawn on our axes in an undeclared frame is a different object on the
same picture."""),

("code", WB04_PARAMS),
("code", WB04_RULES),

("md", r"""## The run and the sites

Everything below reads `<work_root>/survey/runs.csv` and the run folders `<site>/<run>_<stamp>/` that workbook
03 wrote. The ledger is appended to, and a resumed run writes a second row per product, so
`auslamp_proc.products.ledger` reduces it to one row per site, run, stamp, kind and rate before anything is
counted. `RUNS = "latest"` keeps the newest stamp of every run name, so a 1 Hz run and a 10 Hz run under
different names are both read and a repeated run name is not counted twice."""),

("code", WB04_SETUP),

("md", r"""## The products found

One row per product: where it came from, how many periods it carries, what it cost, and which reference it
was built on. The `remote` and `members` columns are the ledger's, so a stack names the sites it averaged and
the weight each carried.

**This check fails if any product named in runs.csv is missing from disk, does not read as a transfer function
with finite impedance at 5 or more periods, or carries unsorted or duplicated periods.** The three limbs are
independent of the ledger that names the products: the first opens the file, the second counts the finite
off-diagonal elements the reader is left with after the fill is masked, and the third compares the file's own
period order with the sorted one. The run folders are also walked for EDIs the ledger does not name, which is
the other half of the same question and is reported beside the verdict."""),

("code", '''rows, missing, too_short, out_of_order = [], [], [], []
for r in PROD.itertuples():
    if not r.on_disk:
        missing.append("%s %s %g Hz: %s" % (r.site, r.kind, r.rate_hz, Path(r.path).name))
        continue
    try:
        tf = read(r.path)
    except Exception as exc:
        missing.append("%s %s %g Hz: unreadable, %s" % (r.site, r.kind, r.rate_hz, type(exc).__name__))
        continue
    finite = max(tf.meta["n_finite"]["xy"], tf.meta["n_finite"]["yx"])
    if finite < 5:
        too_short.append("%s %s %g Hz: %d finite period(s)" % (r.site, r.kind, r.rate_hz, finite))
    if not tf.meta["source_sorted"] or tf.meta["n_duplicate_periods"]:
        out_of_order.append("%s %s %g Hz: sorted %s, %d duplicate(s)"
                            % (r.site, r.kind, r.rate_hz, tf.meta["source_sorted"],
                               tf.meta["n_duplicate_periods"]))
    rows.append(dict(site=r.site, run=r.run, kind=r.kind, rate_hz=r.rate_hz, params=r.params,
                     periods=tf.meta["n_periods"], shortest_s=round(float(tf.period.min()), 2),
                     longest_s=round(float(tf.period.max()), 0), finite_xy=tf.meta["n_finite"]["xy"],
                     finite_yx=tf.meta["n_finite"]["yx"], tipper=tf.meta["has_tipper"],
                     seconds=r.seconds, rss_mb=r.peak_rss_mb, remote=r.remote,
                     members=(str(r.members)[:46] if r.members else "")))
found = pd.DataFrame(rows)
print(found.to_string(index=False))
print()
print("the reference kinds, as words and as the code keys the file names carry")
print(PR.kind_words(RD.KINDS).to_string(index=False))
print()
if len(IGNORED):
    print("%d %s product(s) sit in the chosen run folders and are ignored by this workbook: %s"
          % (len(IGNORED), RD.DROPPED_KIND, " ".join(sorted(Path(p).name for p in IGNORED.path))[:400]))
    print()
spare = PR.unledgered(WORK, CHOSEN, PAIRS)
print("%d EDI(s) sit in the chosen run folders without a ledger row%s"
      % (len(spare), (": " + "; ".join(Path(p).name for p in spare[:6])) if spare else ""))

n = len(PROD)
if not n:
    print("VERDICT: UNJUDGED -- runs.csv names no product for the chosen sites and runs")
elif missing or too_short or out_of_order:
    print("VERDICT: FAIL -- %d of %d products are missing or unreadable (%s); %d read fewer than 5 finite "
          "periods (%s); %d carry unsorted or duplicated periods (%s)"
          % (len(missing), n, "; ".join(missing[:4]) or "none", len(too_short),
             "; ".join(too_short[:4]) or "none", len(out_of_order), "; ".join(out_of_order[:4]) or "none"))
else:
    print("VERDICT: PASS -- all %d products named in runs.csv are on disk, each reads as a transfer function "
          "with %d to %d finite off-diagonal periods over %.2f to %.0f s, and every one carries its periods "
          "sorted and unique"
          % (n, int(found[["finite_xy", "finite_yx"]].max(axis=1).min()),
             int(found[["finite_xy", "finite_yx"]].max(axis=1).max()),
             float(found.shortest_s.min()), float(found.longest_s.max())))
'''),

("md", r"""## One site, every product

Every product of one site on one page: apparent resistivity and phase for xy and yx, and the tipper as real
parts in filled circles and imaginary parts as open triangles on a dotted line. The kind is the colour, the
rate is the line style (1 Hz solid, 10 Hz dashed) and the run is the marker, so the three questions the page
answers are readable at once. The y limits of the rho panels are the 2nd to 98th percentile of every curve
drawn, padded half a decade each way; the phase panels are fixed at 0-90 deg. The header lines under the
title are the product's own, taken from the EDI's processing_parameters.

What to look for. The four kinds -- remote site, fleet stack, observatory, stack + observatory -- should lie
on each other at 100-1000 s, where the field is large and every reference sees the same source; a kind that
departs from the others there is the one to read, not the average of them. At the short end the kinds part
where the references stop seeing the same source, which is where the choice of reference is worth the most.
The error bars grow at the long end, where the record runs out of independent windows.
The tipper's real and imaginary parts are drawn together because a real part alone cannot be told from a leak.

A page is written for every chosen site to `<work_root>/<site>/products_<run>.png`."""),

("code", '''pages = []
for site in CHOSEN:
    grp = PROD[PROD.site == site]
    on = grp[grp.on_disk]
    head = PR.metadata_lines(read(on.iloc[0].path).meta) if len(on) else []
    path, _index = FIG.site_page(
        site, grp, read, WORK / site / ("products_%s.png" % RUN_TAG),
        title="%s: every product of run %s, %s" % (site, RUN_TAG, sv.cfg["name"]),
        header_lines=head, period_range=PERIOD_RANGE)
    pages.append(path)
WRITTEN += pages
print("%d site page(s) written, one per chosen site" % len(pages))
print("showing %s" % SHOW_SITE)
display(Image(filename=str(WORK / SHOW_SITE / ("products_%s.png" % RUN_TAG))))
'''),

("md", r"""## Kind against kind

For every site and rate, every pair of kinds is scored per decade: the median ratio of apparent resistivity
(the ratio of the squared impedance magnitudes) and the median phase difference in degrees, over the periods
both curves carry. `auslamp_proc.agreement.on_grid` puts the second curve on the first's periods by linear
interpolation in log period of log |Z| and of the unwrapped phase, with no extrapolation beyond the second
curve's own first and last valid node and no bridging of a hole wider than 0.30 decades.

The rule: two kinds agree where the rho ratio is within AGREE_RHO of one and the phase difference is within
AGREE_PHASE_DEG, both over AGREE_BAND. At the default values that is 20 per cent and 5 deg over 5-200 s.
Agreement between two references that share no magnetics is the evidence that neither is carrying its own
noise into the estimate; disagreement between them is not resolved here, and is workbook 06's question.

**This check fails if the table is UNJUDGED (no pair scored).** A site with one kind yields no pair, and a
survey where no pair could be put on a common period range scores nothing at all; either is a failure, because
a table of no rows cannot support the reading above it."""),

("code", '''kk = AG.kind_vs_kind(PROD, read=read, bands=[tuple(b) for b in BANDS], agree_rho=AGREE_RHO,
                     agree_phase=AGREE_PHASE_DEG, agree_band=tuple(AGREE_BAND))
band_tag = AG.band_label(*AGREE_BAND)
# a survey whose chosen run holds one kind per site yields no pair at all, and kind_vs_kind comes back with
# no rows and no columns: the check reads UNJUDGED rather than raising on a column that was never made
scored = kk[(kk.band == band_tag) & (kk.n > 0)] if len(kk) else kk
counts = []
for (site, rate), g in (scored.groupby(["site", "rate_hz"]) if len(scored) else []):
    pairs = g.groupby(["kind_a", "kind_b"]).agrees.all()
    counts.append(dict(site=site, rate_hz=rate, pairs=int(len(pairs)), agreeing=int(pairs.sum()),
                       worst_rho_ratio=round(float(g.rho_ratio.iloc[
                           int(np.nanargmax(np.abs(np.log(g.rho_ratio.values))))]), 3),
                       worst_phase_deg=round(float(g.phase_diff_deg.iloc[
                           int(np.nanargmax(np.abs(g.phase_diff_deg.values)))]), 2),
                       disagreeing=" ".join(sorted({"%s-%s" % k for k, v in pairs.items() if not v}))))
agree_counts = pd.DataFrame(counts)
print("both components inside %.0f %% and %.1f deg over %s; a pair counts as agreeing only where both do"
      % (100 * AGREE_RHO, AGREE_PHASE_DEG, band_tag))
print(agree_counts.to_string(index=False))
print()
print("the per-decade table, the first site as an example")
if len(CHOSEN) and len(kk):
    ex = kk[(kk.site == CHOSEN[0]) & (kk.band != band_tag)]
    print(ex[["kind_a", "kind_b", "rate_hz", "band", "component", "rho_ratio", "phase_diff_deg",
              "n"]].round(3).to_string(index=False))
path = OUT / ("agreement_kinds_%s.csv" % RUN_TAG)
kk.round(4).to_csv(path, index=False)
WRITTEN.append(path)
print()
print("-> %s (%d rows)" % (path, len(kk)))

if not len(scored):
    print("VERDICT: UNJUDGED -- no pair of kinds was scored over %s" % band_tag)
else:
    n_pairs = int(agree_counts.pairs.sum())
    n_agree = int(agree_counts.agreeing.sum())
    print("VERDICT: PASS -- %d pairs of kinds scored over %d site-rate(s); %d of them agree within %.0f %% "
          "and %.1f deg over %s and %d do not (%s)"
          % (n_pairs, len(agree_counts), n_agree, 100 * AGREE_RHO, AGREE_PHASE_DEG, band_tag,
             n_pairs - n_agree,
             " ".join(sorted({x for s in agree_counts.disagreeing for x in s.split() if x})) or "none"))
'''),

("md", r"""## Rate against rate

Where a site carries both a 1 Hz and a 10 Hz product of the same kind, the two are compared on the bands each
side of the 16 s join: 8-16 s below and 32-100 s above, with 18-36 s left as a guard band because the Earth
Data logger writes an instrument line at 20.6 s. The number reported is the difference in apparent resistivity
as a percentage of the 1 Hz level, and the phase difference in degrees.

Aurora at 10 Hz reads about 8 per cent low at 4-32 s against its own 1 Hz product (AusLAMP Victoria,
2026-09-11), and every 10 Hz product carries that sentence in its own file. Nothing is spliced here; the join
is workbook 06's work, and this section is the measurement it will be decided on. A reading, not a check."""),

("code", '''rows = []
for site in sorted(set(PROD.site)):
    g = PROD[(PROD.site == site) & PROD.on_disk]
    for kind in sorted(set(g.kind)):
        a = g[(g.kind == kind) & (g.rate_hz == 1.0)]
        b = g[(g.kind == kind) & (g.rate_hz == 10.0)]
        if not len(a) or not len(b):
            continue
        for _, r in AG.rate_step(read(a.iloc[0].path), read(b.iloc[0].path)).iterrows():
            rows.append(dict(site=site, kind=kind, component=r.component,
                             below_8_16_pct=round(r.below_pct, 1), below_phase_deg=round(r.below_phase_deg, 2),
                             above_32_100_pct=round(r.above_pct, 1),
                             above_phase_deg=round(r.above_phase_deg, 2),
                             n_below=int(r.n_below), n_above=int(r.n_above)))
steps = pd.DataFrame(rows)
if not len(steps):
    print("no site carries both a 1 Hz and a 10 Hz product of the same kind in this run; the rate step is "
          "not scored")
else:
    print("the 1 Hz level minus the 10 Hz level, as a per cent of the 1 Hz apparent resistivity")
    print(steps.to_string(index=False))
    print()
    print("median below the join %+.1f %%, above %+.1f %%; the 10 Hz product reads %s at 8-16 s"
          % (steps.below_8_16_pct.median(), steps.above_32_100_pct.median(),
             "low" if steps.below_8_16_pct.median() > 0 else "high"))
    ten = PROD[(PROD.rate_hz == 10.0) & PROD.on_disk]
    if len(ten):
        caveat = read(ten.iloc[0].path).meta["parameters"].get("caveat_10hz", "")
        print("the caveat every 10 Hz product carries: %s" % caveat)
'''),

("md", r"""## Run against run

Where a site has the same kind and rate under two runs, the two are put on the ten-per-decade grid
T = 10^(k/10) from 3.16 s to 50,119 s, which belongs to neither of them, and the largest change per decade is
reported. Two runs of the same site and kind differ only in what the run itself did -- a mask, a window, a
reference the rule chose differently on the day -- so a change here is the size of that decision.

A site with one run yields nothing to compare, which is stated as a reading and not as a verdict: there is no
criterion a single run can fail."""),

("code", '''rr = AG.run_vs_run(PROD, read=read, bands=[tuple(b) for b in BANDS])
if not len(rr):
    per_site = PROD.groupby(["site", "kind", "rate_hz"]).run.nunique()
    print("UNJUDGED: no site carries the same kind and rate under two runs -- %d site-kind-rate combination(s) "
          "each have one run, so there is nothing to compare across runs"
          % int((per_site == 1).sum()))
    print("the runs present: %s" % ", ".join("%s_%s" % p for p in PAIRS))
else:
    worst = (rr.assign(change=lambda d: np.abs(np.log(d.rho_ratio)))
               .sort_values("change", ascending=False)
               .groupby(["site", "kind", "rate_hz"]).head(1))
    print("the largest change per site, kind and rate, over the decades %s"
          % ", ".join(AG.band_label(*b) for b in BANDS))
    print(worst[["site", "kind", "rate_hz", "run_a", "run_b", "band", "component", "rho_ratio",
                 "phase_diff_deg", "n"]].round(3).to_string(index=False))
'''),

("md", r"""## Smoothness

A reading of each curve on its own, with no oracle and no other product. Each point of log rho and of the
phase is compared with a weighted quadratic fitted to its three neighbours each side, the point itself left
out, with the product's own error bars as the weights; the residual over the combined sigma is the score, and
a point above 4 is a jump. Curvature is absorbed by the quadratic, so a steep smooth curve scores nothing and
a single-period spike or a step at a decimation seam stands out. Reported beside it: the largest phase step
between adjacent periods in degrees, taken the short way round the circle so a curve crossing +-180 deg is not
read as a 360 deg step, the same step as a rate in degrees a decade with the spacing floored at 0.05 decades,
and the median residual against the local one-dimensional relation phi = 45 deg x (1 - d log rho / d log T),
which runs 10-15 deg at a two- or three-dimensional site and is read together with the jumps, never alone.

A jump count is a property of the bars as much as of the curve: a product with small bars and a real bend
scores jumps that a noisier product of the same earth does not."""),

("code", '''rows = []
for r in PROD[PROD.on_disk].itertuples():
    for _, s in AG.smoothness(read(r.path), period_range=PERIOD_RANGE).iterrows():
        rows.append(dict(site=r.site, kind=r.kind, rate_hz=r.rate_hz, component=s.component,
                         periods=int(s.n_periods), jumps=int(s.n_jumps),
                         jumps_per_decade=s.jumps_per_decade,
                         phase_step_deg=round(float(s.max_phase_step_deg), 1),
                         phase_step_deg_per_decade=round(float(s.max_phase_step_deg_per_decade), 1),
                         median_abs_dphi_deg=round(float(s.median_abs_dphi_deg), 1)))
smooth = pd.DataFrame(rows)
by_product = (smooth.groupby(["site", "kind", "rate_hz"])
              .agg(jumps_per_decade=("jumps_per_decade", "mean"),
                   phase_step_deg=("phase_step_deg", "max"),
                   phase_step_per_decade=("phase_step_deg_per_decade", "max"),
                   dphi=("median_abs_dphi_deg", "median")).reset_index().round(2))
print("per kind, the mean over the %d product(s) of that kind" % len(PROD[PROD.on_disk]))
print(smooth.groupby("kind")[["jumps_per_decade", "phase_step_deg", "phase_step_deg_per_decade",
                              "median_abs_dphi_deg"]].mean().round(2).to_string())
print()
ranked = by_product.sort_values(["jumps_per_decade", "dphi"])   # <- the rho-phase residual breaks the ties
print("the five cleanest products by jumps per decade over %g-%g s, the rho-phase residual breaking ties"
      % PERIOD_RANGE)
print(ranked.head(5).to_string(index=False))
print()
print("the five roughest")
print(ranked.tail(5).to_string(index=False))
'''),

("md", r"""## The comparisons, last

This section sets our products beside a processing done outside this run. It comes last, and it is labelled a
comparison and never the truth, for one reason: two independent processings of the same field are two
measurements. Neither is an oracle. A difference in level is a gain, a dipole length or a frame before it is
the earth, and a difference that the declination turn removes was never a difference in the earth at all.

Every source is declared in `surveys/<SURVEY>/survey.yaml` with the frame its tensors are in and a note
saying what it is, and `auslamp_proc.products.comparison_sources` refuses one that declares neither:

| frame | what is done to the source |
|---|---|
| geomagnetic | compared as it is: the source is already in the mean-field frame ours are served in |
| geographic | turned into our frame by +declination_deg from sites.csv (Z' = R Z R^T, T' = T R^T, R = [[cos, sin], [-sin, cos]]), which is the inverse of the to_geographic_north_deg angle every product of ours carries |
| instrument | compared as laid, with the note printed beside the table |

The comparison is what moves; our products are never turned. A source declaring `rho_factor` -- a level its
own record says it is out by -- is multiplied by it in apparent resistivity, and the factor is stated in every
figure title and every table row it enters.

The reading per site, kind and component follows the same three explanations a difference can have: a SCALE
is a constant ratio with the phase untouched, which is a dipole length or a gain and is the only one a number
can fix; a FRAME is a disagreement the declination turn removes; a FAULT is neither -- the ratio wanders with
period, or the phase disagrees and the turn does not fix it. It is a reading and not a check.

**This check fails if a declared source lacks a frame declaration, or if the frame turn applied to any tensor
changes a rotation invariant by more than 1e-9 relative.** The second limb is an independent observable of the
turn itself: Zxy - Zyx, Zxx + Zyy and det Z are unchanged by Z' = R Z R^T for any rotation R, so a turn that
moves one of them is a scale, a reflection or an index slip and not a rotation. Each is compared with the
largest element of the same tensor at the same period, because a one-dimensional site has Zxx + Zyy = 0
exactly and a relative bound on zero is one no arithmetic can meet."""),

("code", '''SOURCES = PR.comparison_sources(sv, COMPARE)
vs_tables, no_frame, turn_bad, turn_n = {}, [], [], 0
for src in SOURCES:
    print("source       %s" % src["name"])
    print("  folder     %s" % src["folder"])
    print("  frame      %s" % (src["frame"] or "NOT DECLARED"))
    print("  note       %s" % (src["note"] or "NOT DECLARED"))
    if src["rho_factor"] != 1.0:
        print("  rho_factor %.3f applied to the comparison, as survey.yaml declares" % src["rho_factor"])
    if src["error"]:
        no_frame.append("%s: %s" % (src["name"], src["error"]))
        print("  REFUSED    %s" % src["error"])
        print()
        continue
    tab = AG.versus_comparison(PROD, src, DECLINATION, read=read, bands=[tuple(b) for b in BANDS],
                               agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE_DEG,
                               agree_band=tuple(AGREE_BAND))
    vs_tables[src["name"]] = tab
    for site in sorted(set(PROD.site)):
        dec = DECLINATION.get(site)
        if dec is None:
            continue
        for _kind, tf in PR.load_comparison(src, site, dec).items():
            turned = PR.turn_to_our_frame(tf, dec)
            ok = np.all(np.isfinite(tf.z.reshape(len(tf.period), -1)), axis=1)
            if not ok.any():
                continue
            scale = np.nanmax(np.abs(tf.z[ok]).reshape(int(ok.sum()), -1), axis=1)
            for label, f in (("Zxy - Zyx", lambda a: a[:, 0, 1] - a[:, 1, 0]),
                             ("Zxx + Zyy", lambda a: a[:, 0, 0] + a[:, 1, 1]),
                             ("det Z", lambda a: a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0])):
                d = np.abs(f(turned.z[ok]) - f(tf.z[ok])) / np.maximum(scale ** (2 if label == "det Z" else 1),
                                                                       1e-30)
                turn_n += 1
                if np.nanmax(d) > 1e-9:
                    turn_bad.append("%s %s %s: %.2e" % (src["name"], site, label, float(np.nanmax(d))))
    hit = tab[(tab.band == AG.band_label(*AGREE_BAND)) & (tab.n > 0)] if len(tab) else tab
    print("  scored     %d site(s), %d kind(s), %d row(s); %d of %d readings agree"
          % (tab.site.nunique() if len(tab) else 0, tab.kind.nunique() if len(tab) else 0, len(tab),
             int((hit.reading == "agrees").sum()) if len(hit) else 0, len(hit)))
    print()
print("%d tensor invariant(s) tested across %d source(s)" % (turn_n, len(vs_tables)))
'''),

("md", r"""### What the comparison says, per site and kind

The table below is the 100-1000 s band: the median ratio of our apparent resistivity to the source's, the
median phase difference in degrees, and the reading. A ratio above one means our level is the higher of the
two. The source's frame and note are printed above; the numbers are a shape check."""),

("code", '''for name, tab in vs_tables.items():
    src = [s for s in SOURCES if s["name"] == name][0]
    mid = tab[(tab.band == "100-1000 s") & (tab.n > 0)] if len(tab) else tab
    if not len(mid):
        print("%s: nothing scored at 100-1000 s" % name)
        continue
    piv = mid.pivot_table(index=["site", "kind"], columns="component",
                          values=["rho_ratio", "phase_diff_deg"]).round(3)
    print("%s -- ours over the source at 100-1000 s, frame %s%s"
          % (name, src["frame"],
             "" if src["rho_factor"] == 1.0 else ", the source scaled by %.2f in rho" % src["rho_factor"]))
    print(piv.to_string())
    off = mid[(np.abs(mid.rho_ratio - 1.0) > AGREE_RHO) | (np.abs(mid.phase_diff_deg) > AGREE_PHASE_DEG)]
    print()
    print("%d of %d site-kind-component rows differ by more than %.0f %% or %.1f deg"
          % (len(off), len(mid), 100 * AGREE_RHO, AGREE_PHASE_DEG))
    if len(off):
        print(off[["site", "kind", "component", "rho_ratio", "phase_diff_deg", "n"]]
              .round(3).to_string(index=False))
    print()
    calls = tab[(tab.band == AG.band_label(*AGREE_BAND)) & (tab.n > 0)]
    print("the reading over %s" % AG.band_label(*AGREE_BAND))
    print(calls.reading.str.split(" ").str[0].value_counts().to_string())
    path = OUT / ("vs_%s_%s.csv" % (name, RUN_TAG))
    tab.round(4).to_csv(path, index=False)
    WRITTEN.append(path)
    print("-> %s (%d rows)" % (path, len(tab)))
    print()
'''),

("md", r"""### The page redrawn with the source behind

The same page as above with every declared source drawn in black with its own error bars, behind our products.
The title says what it is. Written to `<work_root>/<site>/products_<run>_vs_<source>.png`."""),

("code", '''vs_pages = []
for src in SOURCES:
    if src["error"]:
        continue
    for site in CHOSEN:
        dec = DECLINATION.get(site)
        try:
            loaded = PR.load_comparison(src, site, dec)
        except Exception:
            loaded = {}
        if not loaded:
            continue
        comps = [("%s %s" % (src["name"], k or "site"), tf) for k, tf in sorted(loaded.items())]
        grp = PROD[PROD.site == site]
        on = grp[grp.on_disk]
        head = PR.metadata_lines(read(on.iloc[0].path).meta) if len(on) else []
        head = head + ["comparison: %s, frame %s%s" % (src["name"], src["frame"],
                                                       "" if src["rho_factor"] == 1.0 else
                                                       ", rho x %.2f" % src["rho_factor"]),
                       "comparison note: %s" % src["note"][:100]]
        path, _index = FIG.site_page(
            site, grp, read, WORK / site / ("products_%s_vs_%s.png" % (RUN_TAG, src["name"])),
            comparisons=comps, header_lines=head, period_range=PERIOD_RANGE,
            title="%s: our products with %s behind in black -- a comparison, not truth" % (site, src["name"]))
        vs_pages.append(path)
WRITTEN += vs_pages
print("%d page(s) redrawn with a source behind" % len(vs_pages))
if vs_pages:
    shown = [p for p in vs_pages if Path(p).parent.name == SHOW_SITE] or vs_pages
    display(Image(filename=str(shown[0])))

if not SOURCES:
    print("VERDICT: UNJUDGED -- survey.yaml declares no comparison source, or COMPARE is \\"none\\"")
elif no_frame or turn_bad:
    print("VERDICT: FAIL -- %d declared source(s) carry no frame or no note (%s); %d of %d rotation "
          "invariants move by more than 1e-9 relative (%s)"
          % (len(no_frame), "; ".join(no_frame) or "none", len(turn_bad), turn_n,
             "; ".join(turn_bad[:4]) or "none"))
elif not turn_n:
    print("VERDICT: UNJUDGED -- no comparison tensor was found to test the frame turn on")
else:
    print("VERDICT: PASS -- all %d declared source(s) carry a frame and a note (%s), and the frame turn "
          "leaves every one of the %d rotation invariants tested unchanged to better than 1e-9 relative"
          % (len(SOURCES), ", ".join("%s %s" % (s["name"], s["frame"]) for s in SOURCES), turn_n))
'''),

("md", r"""## The gallery

PER_PAGE sites a page, every product of the run, rho and phase for xy and yx. The legend is on the first page
only, so the panels carry no repeated key. Each page is written to
`<work_root>/survey/gallery_<run>_p<k>.png` and every curve drawn is named in the index CSV beside them.

**This check fails if any chosen site has no page and no gallery panel.** The two are counted from the files
and the index rather than from the list of sites: a site page is scored by the file existing on disk, and a
gallery panel by at least one row in the index naming that site."""),

("code", '''pages, index, index_path = FIG.gallery(
    CHOSEN, PROD, read, OUT, stem="gallery_%s" % RUN_TAG, per_page=int(PER_PAGE),
    period_range=PERIOD_RANGE, title="%s, run %s" % (sv.cfg["name"], RUN_TAG))
WRITTEN += list(pages) + [index_path]
print("%d gallery page(s), %d curve(s) drawn" % (len(pages), len(index)))
print(index.groupby("page").site.nunique().to_string())
display(Image(filename=str(pages[0])))

no_page = [s for s in CHOSEN if not (WORK / s / ("products_%s.png" % RUN_TAG)).exists()]
no_panel = [s for s in CHOSEN if s not in set(index.site)] if len(index) else list(CHOSEN)
if not CHOSEN:
    print("VERDICT: UNJUDGED -- no site was chosen, so no page and no panel were drawn")
elif no_page or no_panel:
    print("VERDICT: FAIL -- %d of %d chosen sites have no page (%s) and %d have no gallery panel (%s)"
          % (len(no_page), len(CHOSEN), " ".join(no_page) or "none", len(no_panel),
             " ".join(no_panel) or "none"))
else:
    print("VERDICT: PASS -- all %d chosen sites carry a page and at least one gallery panel; %d curves on "
          "%d gallery page(s) at %d sites a page"
          % (len(CHOSEN), len(index), len(pages), int(PER_PAGE)))
'''),

("md", r"""## What was written"""),

("code", '''rows = []
for p in WRITTEN:
    p = Path(p)
    if p.exists():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d files, %.1f MB, in %.1f minutes" % (len(files), files.kb.sum() / 1024, (time.time() - T0) / 60))
print(files.head(60).to_string(index=False))
if len(files) > 60:
    print("   ... and %d more" % (len(files) - 60))
print()
print("ignored: %d %s product(s) on disk, read by nothing above" % (len(IGNORED), RD.DROPPED_KIND))
for p in sorted(IGNORED.path):
    print("   %s" % p)
'''),
]


# ===================================================================== 05 one site, in depth

WB05_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
SITE = "Q53N"                 # one site; Q53N carries a shared centre and sound magnetics (see below)
RUN = "site"                  # the forms' run name; the folder is <work_root>/<SITE>/<RUN>_<stamp>
STAMP = None                  # None = the newest <RUN>_* folder if there is one, else a new stamp
BASELINE_KIND = "remote"      # the reference every form is built on: remote | stack | obs | stack_obs
RATES = [1, 10]               # [1] is the 1 Hz lane alone; 10 adds the short end of section 7
COMPONENTS = ["xy", "yx"]     # the components the masks, the windows and the hours are selected for
REDO = False                  # True remakes a form whose EDI is already in the run folder
WRITE_DECISIONS = False       # decisions are Ben's: True writes the proposed cells into decisions.csv
WORK_ROOT = None              # None = survey.yaml work_root; every product and figure lands under it
'''

WB05_RULES = '''# ---- the method parameters: a change here changes what a form is built on ----
K_NEAREST = 2                 # the neighbours the daily magnetics test reads a day against
FLEET_NEAR_KM = 150           # the distance the fleet test reads its own pairs and its control pairs over
NEG_STRETCH = None            # None = the shifted pair alone | ("2025-10-05", "2025-10-06") a stretch of junk
DAY_THR = 0.5                 # a day is kept where its line's observatory multiple coherence reaches this
COH_BAND = (20, 200)          # the component mask's band, in s
COH_MIN = 0.5                 # ... and the running median below which an hour is masked (a live line > 0.85)
SMOOTH_H = 6                  # the running median's length, in hours
MIN_HOLE_H = 6                # a masked run shorter than this is given back
HOURS_BAND = (2, 50)          # the best-hours selecting band, in s
HOURS_FRACTION = 0.25         # the share of the candidate hours the selection keeps
CONTIG_HOURS = [2, 4, 6, 24]  # the contiguous controls, tiled from the first whole hour, in hours a window
SEED = 20260916               # the named seed every random control is drawn under
CENTRE_DAYS = 3               # the days of highest Ex-Ey coherence the residual test is read over
REPLACE_CHANNEL = None        # None = from the DC flags and the candidates table | "Hx" | "Hy"
LENDER = None                 # None = the nearest candidate that is not in the reference | a site name
BAR_MARGIN = 0.20             # a delivered selection must beat its control on the bar by this fraction
BAR_BAND = (10, 1000)         # the band every bar in the forms table is read over, in s
AGREE_BAND = (100, 1000)      # the band agreement with the baseline is read over, in s
DECADES = [(5, 10), (10, 100), (100, 1000), (1000, 10000)]   # the decades the tables report, in s
SHORT_BANDS = [(0.4, 0.65), (0.65, 1.4), (1.4, 4), (4, 32), (32, 1000)]  # the 10 Hz bands, in s
'''

WB05_SETUP = '''import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import json
import shutil
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

from loguru import logger as _loguru
_loguru.remove()

from auslamp_proc import agreement as AG, look as LK, products as PR, splice as SP, survey as SV
from auslamp_proc.process import KIND_WORD, references as REF
from auslamp_proc.process import aurora_run as AR
from auslamp_proc.process.edi import TEN_HZ_CAVEAT
from auslamp_proc.process.transients import MIN_SEGMENT_S as TR_MIN_SEGMENT_S
from auslamp_proc.site import centre as CE, deliver as DL, forms as FM, masks as MK
from auslamp_proc.site import recipe as RC, replace as RP
from auslamp_proc.figures import site_forms as FF
from auslamp_proc.raw import cache as CACHE

pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 400)

T0 = time.time()
sv = SV.load_survey(SURVEY)
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
WORK = Path(sv.cfg["work_root"])
SITES = list(sv.sites.site)
SITE_DIR = WORK / SITE
PARAMS = "kaiser20_75"
LINE = {"xy": "Ex", "yx": "Ey"}
COMP_H = {"xy": "Hy", "yx": "Hx"}
WRITTEN = []
FORM_ROWS = []
_tfs = {}


def run_dir():
    """The run folder: the newest <RUN>_* where STAMP is None and one exists, else <RUN>_<stamp>.

    A form already in the folder is reported as `exists` and not remade, so the workbook can be re-run at a
    second rate without repeating the first.
    """
    if STAMP:
        return SITE_DIR / ("%s_%s" % (RUN, STAMP))
    old = sorted(SITE_DIR.glob("%s_*" % RUN))
    return old[-1] if old else SITE_DIR / ("%s_%s" % (RUN, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")))


OUT = run_dir()
OUT.mkdir(parents=True, exist_ok=True)


def form(name, **kw):
    """One form, recorded in FORM_ROWS. Every keyword is auslamp_proc.site.forms.run_form's."""
    kw.setdefault("kind", BASELINE_KIND)
    kw.setdefault("rate", 1)
    kw.setdefault("params", PARAMS)
    kw.setdefault("redo", REDO)
    row = FM.run_form(sv, SITE, name, OUT, **kw)
    FORM_ROWS.append(row)
    return row


def read(path):
    """One transfer function, cached: a page, a table and a verdict ask for the same file."""
    path = str(path)
    if path not in _tfs:
        _tfs[path] = PR.read_tf(path)
    return _tfs[path]


def rows_by_form():
    return {r["form"]: r for r in FORM_ROWS}


def made_product(name):
    """The product of one form where it was made and is on disk, else None."""
    r = rows_by_form().get(name, {})
    if r.get("status") in ("made", "exists") and Path(str(r.get("product") or "")).exists():
        return r
    return None


def baseline_path():
    """Workbook 03's product of the chosen kind at 1 Hz, copied into this run folder as the `whole` form.

    The baseline is not re-estimated: it is the file workbook 04 read, under the name the forms table uses,
    so every form is read against the product the earlier workbooks left.
    """
    led = PR.ledger(WORK)
    rows = led[(led.site == SITE) & (led.kind == BASELINE_KIND) & (led.rate_hz == 1.0)]
    if not len(rows):
        return None, "runs.csv names no %s product of %s at 1 Hz" % (BASELINE_KIND, SITE)
    r = rows.sort_values("stamp").iloc[-1]
    src = PR.product_path(WORK, SITE, r.run, r.stamp, r.kind, r.rate_hz, r.params)
    if not src.exists():
        return None, "%s is not on disk" % src
    dst = OUT / FM.product_name(SITE, "whole", BASELINE_KIND, 1, PARAMS)
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst, "copied from %s" % src.parent.name


BASE, BASE_WHY = baseline_path()
FORM_ROWS.append(dict(site=SITE, form="whole", kind=BASELINE_KIND, rate_hz=1.0, params=PARAMS,
                      product=str(BASE) if BASE else "", controls="", criterion="the baseline itself",
                      seed=None, status=("exists" if BASE else "FAILED"),
                      error=("" if BASE else BASE_WHY), days=None, n_runs=None, seconds=None))

ELINES = CE.read_elines(sv, SITE)
DC = pd.read_csv(SITE_DIR / "dc.csv") if (SITE_DIR / "dc.csv").exists() else pd.DataFrame()

print("survey       %s" % sv.cfg["name"])
print("site         %s -- %s" % (SITE, sv.site(SITE).notes or "no note in sites.csv"))
print("work root    %s" % WORK)
print("run folder   %s" % OUT)
print("baseline     %s (%s)" % (Path(BASE).name if BASE else "NONE", BASE_WHY))
print("rates        %s Hz; components %s" % (", ".join(str(r) for r in RATES), ", ".join(COMPONENTS)))
print("seed         %d, used by every random control in this workbook" % SEED)
'''

WB05 = [
("md", r"""# 05 -- One site, in depth

One site is taken apart: which days its magnetics are usable, where in time each electric line is worth
using, whether its two lines share a centre electrode, what its 10 Hz record adds at the short end, and
whether a magnetic channel is worth borrowing from a neighbour. Each answer is built as a form -- one pass
over the same site with one thing changed -- and every form lands in the same run folder as a product
carrying the header a workbook 03 product carries. Section 10 then composes the answers: one frame and, per
impedance row, which hours that row is estimated on and at which rate, assembled into one product.

The rule this workbook is written around: a selection of hours or days is never delivered without its
controls. Each selection carries

1. a random selection of the same size, drawn without replacement from the same scored pool under a seed
   written into the product's header;
2. a contiguous selection of the same duration -- windows of 2, 4, 6 and 24 h tiled from the first whole hour,
   the top windows by mean score until the kept duration matches the selection's to within one window. One
   Aurora window at the deepest decimation level is 65,536 s, so scattered hours cannot reach the long periods
   and only a contiguous comparison at the same cost decides a long-period claim;
3. the selecting statistic scored kept against dropped; the random control is expected not to separate;
4. the product itself scored against its controls on the 10-1000 s bar, on smoothness and on agreement with
   the baseline at 100-1000 s.

The control is scored on the product that would be delivered, not only on the statistic that chose the hours.
A selection whose product does not beat its random control has narrowed nothing but its error bars, and is
not promoted.

Selecting on the target's own E-H coherence uses the target's own response and favours the hours where the
linear model already fits: for a dead electrode that is the truth, for a merely noisy one it can bias the
estimate towards the quiet hours. The thresholds therefore sit well below live -- 0.5, where a live line
reads above 0.85 -- and the component mask asks for a sustained run rather than a single low hour.

Nine checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported as FAIL and is
not revised afterwards. The recipe's window check is scored only where `survey.yaml` records a window for the
site; elsewhere it prints the window it found as a reading and writes no verdict.

The site the parameter cell opens on is AusLAMP Queensland Phase 1's Q53N: its two electric lines read an
Ex-Ey coherence of 0.95-0.99 on every day of the record while its magnetometer passes the DC test against
IGRF, so the shared-centre test of section 6 has something to find and the magnetics every other section
rests on are not themselves in doubt. It has neighbours 89 and 110 km away on two sides, which is what the
fleet test, the clock test and the borrowed channel of section 8 need."""),

("code", WB05_PARAMS),
("code", WB05_RULES),
("code", WB05_SETUP),

("md", r"""## 1. The site as workbooks 02 to 04 left it

The record figure, the magnetometer DC test and the per-day state of the two electric lines are workbook 02's;
the products and their agreement are workbook 03's and 04's. Nothing here is recomputed. What the three
tables say is what selects the methods below: a magnetometer that fails the DC test sends the work to section
8, a line that dies mid-record sends it to section 4, and two lines carrying one voltage send it to section
6."""),

("code", '''print("the magnetometer against IGRF (workbook 02)")
print(DC.to_string(index=False) if len(DC) else "no dc.csv: run workbook 02 over this site")
print()
keep_cols = ("site", "days_scored", "Ex_sound", "Ex_weak", "Ex_common", "Ex_dead",
             "Ey_sound", "Ey_weak", "Ey_common", "Ey_dead", "both_sound_days")
summary = pd.DataFrame([{k: v for k, v in LK.eline_summary(ELINES, SITE).items() if k in keep_cols}])
print("the two electric lines by day (workbook 02)")
print(summary.to_string(index=False))
print()
print("Ex-Ey coherence over the record: median %.3f, maximum %.3f over %d scored day(s)"
      % (float(np.nanmedian(ELINES.coh_Ex_Ey)), float(np.nanmax(ELINES.coh_Ex_Ey)),
         int(ELINES.coh_Ex_Ey.notna().sum())))
for name in ("01_record.png", "02_coherence_bands.png"):
    p = SITE_DIR / name
    if p.exists():
        display(Image(filename=str(p)))
'''),

("md", r"""### What this workbook will try, and why

The methods are keyed to the three tables above: a common-mode signature on the two lines calls for the
shared-centre test of section 6, a magnetic flag or a channel reading low against its neighbours calls for
the replacement of section 8, and a line that dies mid-record calls for the window of section 4. The
selections of sections 3 and 5 apply wherever a line is alive but noisy."""),

("code", '''common = float(np.nanmedian(ELINES.coh_Ex_Ey)) if len(ELINES) else np.nan
# DC["flags"], not DC.flags: a DataFrame carries a `flags` attribute of its own and attribute access
# reaches that object rather than the column
flags = str(DC["flags"].iloc[0]) if len(DC) and "flags" in DC.columns else ""
dead = {c: int((ELINES["%s_state" % LINE[c]] == "dead").sum()) if len(ELINES) else 0 for c in COMPONENTS}
plan = [
    dict(section="2 the magnetics by day", applies=True, why="every later test rests on Hx and Hy"),
    dict(section="3 the quality map and the day masks", applies=True,
         why="where in time each line follows the field"),
    dict(section="4 windows", applies=bool(len(ELINES)),
         why="a line dead on %s of %d day(s)" % ("/".join(str(dead[c]) for c in COMPONENTS), len(ELINES))),
    dict(section="5 best hours", applies=True, why="a line alive but noisy"),
    dict(section="6 the shared centre", applies=bool(np.isfinite(common) and common >= 0.5),
         why="median Ex-Ey coherence %.2f" % common),
    dict(section="7 the 10 Hz forms", applies=10 in RATES,
         why="the short end the 1 Hz cache cannot reach"),
    dict(section="8 replacement magnetics", applies=True, why="DC flags: %s" % (flags or "none")),
]
print(pd.DataFrame(plan).to_string(index=False))
'''),

("md", r"""## 2. The magnetics by day

Three tests, each with its control.

The daily test reads each UTC day's total field against IGRF and the fluctuation of Hx and Hz -- the standard
deviation after a 3,000 s high-pass -- against the median of the K_NEAREST sites recording that day. A day is
sound where the field is within 5 per cent of IGRF and both fluctuations sit within a factor 3 of the
neighbours'. Without a neighbour that day only the field is judged and the row says so.

The fleet test takes one stretch and scores the 100-1000 s Hx-Hx and Hy-Hy coherence of every pair of the
site and the sites covering it. Its positive limb reads the site's median against the sites within
FLEET_NEAR_KM against the control pairs at the same distances: a site coherent with the fleet the way its
neighbours are with each other is a site whose magnetics can be used. A site incoherent with every other
site -- a median Hx coherence under 0.3 -- is named and dropped from the control, because a dead sensor or a
clock hours out is not a control.

Its negative limb is a shifted pair: the same stretch and the same estimator on the site's Hx against each
neighbour's Hx taken 12 h later (and Hy likewise), the median over the pairs, which must read under 0.3. A
coherent pair and a shifted one differ by the whole of what the test measures, and the shifted pair exists at
every site; a stretch of junk magnetics, the control the method was first written with, does not exist at a
sound site (NEG_STRETCH accepts one where there is).

The clock test reads the lag of the peak cross-correlation of the despiked Hx against a reference site over a
+-12 h search, the peak refined to a fraction of a sample with a parabola through its two neighbours. The
series correlated is the 5-20 s band-passed record, not the 3,000 s high-passed one the coherence tests use:
a high-pass leaves the daily variation in, and a half-day sinusoid correlated against a neighbour peaks at the
edge of a +-12 h search as readily as at zero (at Q17 that gave a lag of hours where the clock is right to
0.4 s). The search stays at +-12 h so an hour-scale offset is still found.

A day is counted only where the field test holds and three guards pass: the peak correlation reaches 0.5, the
peak stands at least 1.5 times the median correlation over that day's other lags, and the peak is not within
5 per cent of the search edge. Fewer than 5 counted days leaves the clock UNJUDGED and no median is reported.

**This check fails if the shifted-pair control reads a 100-1000 s coherence of 0.3 or more on either
horizontal channel or could not be scored, if the site's own median falls below 0.8 of the control pairs' on
either channel, if fewer than 5 days could be timed, or if the clock lag's median exceeds 10 s.**"""),

("code", '''dm = MK.daily_magnetics_test(sv, SITE, K_NEAREST, SITES)
print("neighbours: %s" % ", ".join(dm.attrs.get("neighbours", [])) or "none overlapping")
print(dm.round(3).to_string(index=False))
judged = dm[dm.judged]
print()
print("%d of %d day(s) sound, %d judged against a neighbour"
      % (int(dm.sound.sum()), len(dm), int(dm.judged.sum())))
'''),

("md", r"""The daily test as three strips, one row per day: the field against IGRF with its 5 per cent
band, and the Hx and Hz fluctuation ratios against the neighbours' median with the factor-3 band. A day
outside any band is red; a day with no neighbour is grey and judged on the field alone. What to look for is
a run of red days at one end of the record -- the transit hours, or a sensor that moved -- rather than red
days scattered through it, which is a neighbour problem and not the site's."""),

("code", '''fig = FF.daily_magnetics(dm, SITE, OUT / "06_daily_magnetics.png",
                        neighbours=dm.attrs.get("neighbours", []))
WRITTEN.append(OUT / "06_daily_magnetics.png")
display(Image(filename=str(OUT / "06_daily_magnetics.png")))
'''),

("code", '''# the stretch the fleet test reads: the first two sound days. NEG_STRETCH stays available for a stretch
# of junk magnetics, and the shifted pair is the control the criterion is written on
t0_rec, n_rec = MK.span(sv, SITE, 1)
day_start = {r.day: int(pd.Timestamp(r.day, tz="UTC").timestamp()) for r in dm.itertuples()}
sound = [r.day for r in dm.itertuples() if r.sound and r.judged]
# the sound day nearest the middle of the record, not the first: the first sound day sits beside the transit
# hours, and the shifted control needs the neighbours' records to reach 12 h either side of the stretch
MIDDLE = sound[len(sound) // 2] if sound else None
FLEET_STRETCH = ((day_start[MIDDLE], day_start[MIDDLE] + 2 * 86400) if MIDDLE
                 else (t0_rec + 86400, t0_rec + 3 * 86400))
print("fleet stretch  %s .. %s" % (pd.Timestamp(FLEET_STRETCH[0], unit="s"),
                                   pd.Timestamp(FLEET_STRETCH[1], unit="s")))
print("shifted control: the same stretch of each neighbour taken %g h later" % (MK.SHIFT_CONTROL_S / 3600))
NEG = (tuple(int(pd.Timestamp(x, tz="UTC").timestamp()) for x in NEG_STRETCH) if NEG_STRETCH else None)
print("NEG_STRETCH    %s" % (NEG_STRETCH or "none: no stretch of junk magnetics was named"))
ft = MK.fleet_test(sv, SITE, FLEET_STRETCH[0], FLEET_STRETCH[1], negative=NEG,
                   near_km=FLEET_NEAR_KM, sites=SITES)
print(ft["table"].round(3).to_string(index=False) if len(ft["table"]) else "no pair scored")
print()
print("%s: Hx %.2f Hy %.2f over %d pair(s) | control pairs Hx %.2f Hy %.2f over %d pair(s)%s"
      % (SITE, ft["site_hx"], ft["site_hy"], ft["n_pairs_site"], ft["control_hx"], ft["control_hy"],
         ft["n_pairs_control"],
         ("; dropped from the control as incoherent with the fleet: " + " ".join(ft["junk"]))
         if ft["junk"] else ""))
if len(ft.get("shifted_table", [])):
    print()
    print("the shifted pair, %g h, one row per neighbour" % (MK.SHIFT_CONTROL_S / 3600))
    print(ft["shifted_table"].round(3).to_string(index=False))
print("shifted control: Hx %.2f Hy %.2f over %d pair(s), and it must read under %.1f"
      % (ft["shifted_hx"], ft["shifted_hy"], ft["n_shifted"], MK.NEGATIVE_MAX))
if NEG_STRETCH:
    print("NEG_STRETCH:     Hx %.2f Hy %.2f (a reading beside the shifted control)"
          % (ft["negative_hx"], ft["negative_hy"]))
'''),

("md", r"""One bar per pair of the fleet table, the site's own pairs first and the control pairs behind,
with the shifted pair of each neighbour drawn beside it. What to look for is the shifted bars sitting on the
floor while the unshifted ones stand: that gap is what the test measures, and a site whose own bars
sit down with the shifted ones is not measuring the field the fleet measures."""),

("code", '''fig = FF.fleet_bars(ft, SITE, OUT / "07_fleet.png", near_km=FLEET_NEAR_KM,
                    floor=MK.NEGATIVE_MAX)
WRITTEN.append(OUT / "07_fleet.png")
display(Image(filename=str(OUT / "07_fleet.png")))
'''),

("code", '''ck = MK.clock_test(sv, SITE, sites=SITES)
print("the clock against %s (%.0f km) on the %g-%g s band, searched to +-%g h"
      % (ck["ref"], ck.get("km", np.nan), MK.CLOCK_BAND_S[0], MK.CLOCK_BAND_S[1],
         MK.CLOCK_MAXLAG_S / 3600))
if len(ck["table"]):
    print(ck["table"].round(3).to_string(index=False))
print("%d day(s) counted of %d; %s"
      % (ck["n_days"], len(ck["table"]),
         ("median lag %+.2f s" % ck["median_lag_s"]) if ck["judged"]
         else "UNJUDGED, and no median is reported: %s" % ck["reason"]))
fig = FF.clock_lags(ck, SITE, OUT / "08_clock.png", pass_s=MK.CLOCK_PASS_S,
                    edge_fraction=MK.CLOCK_EDGE_FRACTION, peak_ratio=MK.CLOCK_PEAK_RATIO,
                    maxlag_s=MK.CLOCK_MAXLAG_S)
WRITTEN.append(OUT / "08_clock.png")
display(Image(filename=str(OUT / "08_clock.png")))
'''),

("code", '''fail = []
if not ft.get("shifted_judged"):
    fail.append("the shifted-pair control was not scored (no neighbour's record reaches %g h either side of "
                "the stretch), so the fleet test was not shown able to fail" % (MK.SHIFT_CONTROL_S / 3600))
elif max(ft["shifted_hx"], ft["shifted_hy"]) >= MK.NEGATIVE_MAX:
    fail.append("the shifted-pair control reads Hx %.2f Hy %.2f over %d pair(s), at or above %.1f"
                % (ft["shifted_hx"], ft["shifted_hy"], ft["n_shifted"], MK.NEGATIVE_MAX))
if not ft.get("positive_ok"):
    fail.append("the site reads Hx %.2f Hy %.2f against the control pairs' %.2f and %.2f, below %.1f of them"
                % (ft["site_hx"], ft["site_hy"], ft["control_hx"], ft["control_hy"],
                   MK.FLEET_CONTROL_FRACTION))
if not ck["judged"]:
    fail.append("UNJUDGED on the clock against %s: %s" % (ck["ref"], ck["reason"]))
elif abs(ck["median_lag_s"]) > MK.CLOCK_PASS_S:
    fail.append("the clock lag's median is %+.2f s over %d day(s), beyond %.0f s"
                % (ck["median_lag_s"], ck["n_days"], MK.CLOCK_PASS_S))
if fail:
    print("VERDICT: FAIL -- %s; for the record the shifted-pair control reads Hx %.2f Hy %.2f over %d "
          "pair(s) and the clock counted %d day(s) of %d on the %g-%g s band"
          % ("; ".join(fail), ft["shifted_hx"], ft["shifted_hy"], ft["n_shifted"], ck["n_days"],
             len(ck["table"]), MK.CLOCK_BAND_S[0], MK.CLOCK_BAND_S[1]))
else:
    print("VERDICT: PASS -- the shifted-pair control reads Hx %.2f Hy %.2f over %d pair(s), both under "
          "%.1f, where the same pairs unshifted read Hx %.2f Hy %.2f; over the sound stretch the site reads "
          "Hx %.2f Hy %.2f against the control pairs' %.2f and %.2f, above %.1f of them; the clock lag's "
          "median is %+.2f s over %d counted day(s) on the 5-20 s band, within %.0f s%s"
          % (ft["shifted_hx"], ft["shifted_hy"], ft["n_shifted"], MK.NEGATIVE_MAX, ft["site_hx"],
             ft["site_hy"], ft["site_hx"], ft["site_hy"], ft["control_hx"], ft["control_hy"],
             MK.FLEET_CONTROL_FRACTION, ck["median_lag_s"], ck["n_days"], MK.CLOCK_PASS_S,
             ("; NEG_STRETCH reads Hx %.2f Hy %.2f as a reading beside it"
              % (ft["negative_hx"], ft["negative_hy"])) if NEG_STRETCH else ""))
'''),

("md", r"""## 3. The quality map and the day masks

The quality map is where in time each electric line follows the magnetic field. Three scales, each with the
statistic the scale can carry: the hourly 4-50 s coherence of each line with the H it couples to, the daily
50-1000 s multiple coherence of each line with the local pair and with the observatory pair, and one
1000-10000 s number on the whole record decimated to 0.1 Hz. The multiple coherence is bias-corrected --
g2c = (g2 - p/nu) / (1 - p/nu) with p = 2 predictors and nu = 0.82 times the number of segments -- because an
estimate from few segments saturates at 1 or sits on the bias floor, which is what an uncorrected hourly
estimate over 50-1000 s reads at every site.

What the number is, and what it is not. It is the fraction of the electric line's power the magnetic field
explains, so it measures local electric noise. A low value does not mean the component cannot be measured: a
remote reference beats the noise the target and the reference do not share, which is why the sites with a
noisy shared centre sit at the bottom of this table and still deliver products. Nor does a high value mean
the row is right: a site whose second line repeats the first has the highest coherence in a survey and an
unusable yx row, which is a geometry fault that coherence with the magnetic field cannot see.

The day mask keeps the whole days whose observatory multiple coherence reaches DAY_THR; those columns carry
no local magnetic noise, which is why the mask is taken on them. Its control keeps the same number of days
drawn without replacement from the same pool of scored days under the seed printed below.

**This check fails if any mask lacks its control, if a control differs in size from its selection, if a mask
keeps too few days to be scored at all, or if a promoted mask does not beat its control on the 10-1000 s bar
by at least 20 per cent.** A mask that does not beat its control is not promoted and is printed as such;
that is a finding about the record, not a failure of the check. The check is on whether every selection has a
control of the right size, whether anything was scored at all, and whether the promotions agree with the bar
they are made on."""),

("code", '''t = time.time()
QM = MK.quality_map(sv, SITE)
print("the quality map in %.0f s: %d hour(s), %d day(s), observatory %s"
      % (time.time() - t, QM["hours"], QM["n_days"], QM["observatory"]))
print(pd.DataFrame([MK.quality_row(QM)]).round(3).to_string(index=False))
fig = FF.quality_images(QM, SITE, OUT / "09_quality_map.png", thr=DAY_THR)
WRITTEN.append(OUT / "09_quality_map.png")
display(Image(filename=str(OUT / "09_quality_map.png")))
'''),

("code", '''DAY_MASKS, rows = {}, []
for comp in COMPONENTS:
    keep, ctrl, info = MK.day_mask(QM, comp, DAY_THR, SEED)
    DAY_MASKS[comp] = (keep, ctrl, info)
    rows.append({k: v for k, v in info.items() if k not in ("kept_days", "control_days")})
print("the day mask per component, on the %s columns, threshold %.2f, seed %d"
      % (QM["observatory"], DAY_THR, SEED))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for comp in COMPONENTS:
    print("   %s kept    %s" % (comp, DAY_MASKS[comp][2]["kept_days"][:110]))
    print("   %s control %s" % (comp, DAY_MASKS[comp][2]["control_days"][:110]))
'''),

("md", r"""The record with the masks drawn on it: the kept days of each component as a solid span in its own
colour, the random control of the same size hatched. What to look for is whether the kept days sit together
at one end of the record -- in which case the mask is a window in disguise and the contiguous comparison of
section 5 is the one to read -- or scattered through it, and whether the control's days look any different to
the eye than the selection's."""),

("code", '''_t0m, _arrm, _metam = CACHE.load(SITE, WORK, 1)
spans = []
for comp in COMPONENTS:
    keep, ctrl, info = DAY_MASKS[comp]
    spans.append(("%s kept, %d day(s)" % (comp, info["days_kept"]), FF.SPAN_COLOUR[comp], None,
                  FF.mask_spans(keep, _t0m)))
    spans.append(("%s control, seed %d" % (comp, SEED), "0.4", "//", FF.mask_spans(ctrl, _t0m)))
fig = FF.record_spans(_t0m, _arrm, OUT / "10_day_masks.png", site=SITE, spans=spans,
                      title="%s: the day masks and their controls" % SITE,
                      caption="The record as a per-minute mean over its per-minute envelope, with the days "
                              "each component's mask keeps drawn as a solid span in that component's colour "
                              "and the random control of the same size hatched beside it. The control draws "
                              "the same number of days from the same pool under seed %d, so the two cost the "
                              "same and only the choosing differs: %s."
                              % (SEED, "; ".join("%s keeps %d day(s)" % (c, DAY_MASKS[c][2]["days_kept"])
                                                 for c in COMPONENTS)))
WRITTEN.append(OUT / "10_day_masks.png")
display(Image(filename=str(OUT / "10_day_masks.png")))
del _arrm
'''),

("code", '''THIN = {}
for comp in COMPONENTS:
    keep, ctrl, info = DAY_MASKS[comp]
    if not info["enough"]:
        # a mask keeping fewer than three days starves the long bands, so no pass is run on it: the pair is
        # UNJUDGED rather than made and then read
        THIN[comp] = info["days_kept"]
        print("   %s: the mask keeps %d day(s) of the %d scored, below the %d-day floor; no product is made"
              % (comp, info["days_kept"], info["days_scored"], MK.DAY_MIN_KEPT))
        continue
    form("daymask_%s" % comp, keep_extra=keep, seed=SEED,
         keep_name="%d whole day(s) at or above %.2f on the %s multiple coherence"
                   % (info["days_kept"], DAY_THR, QM["observatory"]),
         controls=["daymask_%s_control" % comp],
         criterion="beats its random control on the %g-%g s bar by %.0f %%"
                   % (BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN))
    form("daymask_%s_control" % comp, keep_extra=ctrl, seed=SEED,
         keep_name="%d day(s) drawn at random from the same pool, seed %d" % (info["days_control"], SEED))
if not THIN:
    print("   every component's mask clears the %d-day floor" % MK.DAY_MIN_KEPT)
'''),

("code", '''rows, bad, unjudged = [], [], []
for comp in COMPONENTS:
    info = DAY_MASKS[comp][2]
    if info["days_kept"] != info["days_control"]:
        bad.append("%s: the control keeps %d day(s) and the selection %d"
                   % (comp, info["days_control"], info["days_kept"]))
    if comp in THIN:
        unjudged.append("%s: the mask keeps %d day(s) of the %d scored, below the %d-day floor, so no "
                        "product was made and the pair is not scored"
                        % (comp, THIN[comp], info["days_scored"], MK.DAY_MIN_KEPT))
        continue
    for name in ("daymask_%s" % comp, "daymask_%s_control" % comp):
        r = made_product(name)
        if r is None:
            bad.append("%s was not made (%s)" % (name, str(rows_by_form().get(name, {}).get("error"))[:90]))
            continue
        rd = DL.reading(read(r["product"]), read(BASE) if BASE else None,
                        bands=[tuple(b) for b in DECADES], bar_band=tuple(BAR_BAND),
                        agree_band=tuple(AGREE_BAND))
        rows.append(dict(form=name, days=r.get("days"), runs=r.get("n_runs"), bar=rd.get("bar"),
                         bar_xy=rd.get("bar_xy"), bar_yx=rd.get("bar_yx"),
                         jumps_xy=rd.get("jumps_per_decade_xy"), jumps_yx=rd.get("jumps_per_decade_yx"),
                         rho_ratio_xy=rd.get("rho_ratio_xy"), rho_ratio_yx=rd.get("rho_ratio_yx")))
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no product to read")
beaten = []
for comp in COMPONENTS:
    a = next((r for r in rows if r["form"] == "daymask_%s" % comp), None)
    b = next((r for r in rows if r["form"] == "daymask_%s_control" % comp), None)
    if a and b:
        beaten.append((comp, DL.beats(a["bar"], b["bar"], BAR_MARGIN), a["bar"], b["bar"]))
print()
for comp, ok, x, y in beaten:
    print("   %s: bar %.4f against the control's %.4f -- %s"
          % (comp, x, y, "promoted" if ok else "NOT promoted: efficiency, not a different answer"))
promoted = [c for c, ok, _x, _y in beaten if ok]
inconsistent = [c for c, ok, x, y in beaten if ok and not DL.beats(x, y, BAR_MARGIN)]
if bad:
    print("VERDICT: FAIL -- %s" % "; ".join(bad))
elif inconsistent:
    print("VERDICT: FAIL -- %s promoted without beating its control on the %g-%g s bar by %.0f %%"
          % (", ".join(inconsistent), BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN))
elif unjudged:
    print("VERDICT: UNJUDGED -- %s%s" % ("; ".join(unjudged),
          ("; the pair(s) that were scored read " + "; ".join("%s %.4f against %.4f" % (c, x, y)
                                                              for c, ok, x, y in beaten)) if beaten else ""))
elif not beaten:
    print("VERDICT: UNJUDGED -- no mask and control pair was scored")
else:
    print("VERDICT: PASS -- every mask carries a control of the same day count drawn from the same pool "
          "under seed %d, every one of the %d pair(s) was scored on the %g-%g s bar (%s), and the %d "
          "promoted mask(s) (%s) each beat their control by at least %.0f %%"
          % (SEED, len(beaten), BAR_BAND[0], BAR_BAND[1],
             "; ".join("%s %.4f against %.4f" % (c, x, y) for c, ok, x, y in beaten),
             len(promoted), ", ".join(promoted) or "none", 100 * BAR_MARGIN))
'''),

("md", r"""The whole-record product, each mask's product and each control's product on one set of panels.
What to look for is whether the three curves lie on each other: where they do, the mask changed nothing but
the error bars, which is the reading the verdict above prints as efficiency rather than a different
answer."""),

("code", '''curves = [("whole", read(BASE), "k", "-")] if BASE else []
for k, comp in enumerate(COMPONENTS):
    for name, ls in (("daymask_%s" % comp, "-"), ("daymask_%s_control" % comp, "--")):
        r = made_product(name)
        if r:
            curves.append((name, read(r["product"]), "C%d" % (2 * k + (0 if ls == "-" else 1)), ls))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "11_daymask_products.png",
                         title="%s: the day-mask products" % SITE,
                         caption="Each day mask's product and its equal-size random control against the "
                                 "whole record in black, on the panels and the limits workbook 04 uses. The "
                                 "three passes differ only in which days went in, so a mask that bought a "
                                 "cleaner answer sits on the whole record's curve with a smaller error bar "
                                 "and its control does not; a mask and its control lying together say the "
                                 "selection bought the duration and not the days.",
                         period_range=(1, 50000))
    WRITTEN.append(OUT / "11_daymask_products.png")
    display(Image(filename=str(OUT / "11_daymask_products.png")))
else:
    print("no mask product was made, so there is nothing to draw against the whole record")
'''),

("md", r"""## 4. Windows

A line that dies mid-record is not a reason to throw the record away. The rule is the whole record for the
healthy row and for the tipper, the window for the other row, and both windows in the provenance. The window
proposed here is the longest run of sound days of that line in the elines table; where no run of sound days
reaches the floor in survey.yaml the weak days are admitted and the row says so, so the window is a proposal
and not a finding. A window already in decisions.csv is used instead.

The windowed pass slices everything to the window, H included: the point of a window is that this component's
estimate sees only the days its electrode was alive, and an estimator handed a longer H than E would be
given NaN over the rest. The merge then replaces exactly that component's two impedance rows in a copy of the
whole-record product; the station block, the position, the tipper and every other row carry across untouched.

The control is a random block of the same length placed elsewhere in the record under the seed printed below.
A window that buys nothing beyond its length is one a block of the same length placed anywhere would buy.

**This check fails if the merge changes any row other than the windowed component's two, if a window lacks
its equal-length random block, or if a promoted window does not beat that block on the 10-1000 s bar by at
least 20 per cent.** As in section 3, a window that does not beat its block is not promoted, which is a
finding about the site and not a failure of the check."""),

("code", '''WINDOWS, WIN_ROWS = {}, []
cell = str(sv.decision(SITE).get("windows", "")).strip()
from_decisions = {}
if cell and cell.lower() not in ("decide", "nan", "none", ""):
    try:
        from_decisions = json.loads(cell)
    except Exception as exc:
        print("decisions.csv windows does not parse as JSON (%s); the proposal below is used" % exc)
for comp in COMPONENTS:
    d = from_decisions.get(comp) or {}
    if d.get("t_start"):
        w = dict(component=comp, t_start=int(pd.Timestamp(d["t_start"], tz="UTC").timestamp()),
                 t_end=int(pd.Timestamp(d["t_end"], tz="UTC").timestamp()), days=d.get("days"),
                 n_days=d.get("days"), reason="decisions.csv: %s" % d.get("reason", ""), states="")
    else:
        w = MK.window_from_days(ELINES, comp, min_days=FM.min_window_days(sv))
    seed = SEED + (0 if comp == "xy" else 1)
    if w["t_start"]:
        a, b = MK.random_block(t0_rec, n_rec, int(w["t_end"] - w["t_start"]), seed,
                               exclude=(w["t_start"] - t0_rec, w["t_end"] - t0_rec))
        w["control"] = (t0_rec + a, t0_rec + b)
    else:
        w["control"] = None
    w["seed"] = seed
    WINDOWS[comp] = w
    WIN_ROWS.append(dict(component=comp, line=LINE[comp], t_start=w["t_start"], t_end=w["t_end"],
                         days=w["days"], n_days=w["n_days"], seed=seed,
                         control_start=(w["control"][0] if w["control"] else None),
                         reason=w["reason"][:120]))
print(pd.DataFrame(WIN_ROWS).to_string(index=False))
for comp in COMPONENTS:
    w = WINDOWS[comp]
    if w["t_start"]:
        print("   %s window %s .. %s (%.2f d); control block %s .. %s, the same length, seed %d"
              % (comp, pd.Timestamp(w["t_start"], unit="s"), pd.Timestamp(w["t_end"], unit="s"), w["days"],
                 pd.Timestamp(w["control"][0], unit="s"), pd.Timestamp(w["control"][1], unit="s"), w["seed"]))
'''),

("code", '''for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        print("   %s: no window proposed (%s)" % (comp, w["reason"]))
        continue
    form("window_%s" % comp, window=(w["t_start"], w["t_end"]), seed=w["seed"],
         keep_name="the %s window, %.2f d" % (comp, w["days"]),
         controls=["window_%s_control" % comp],
         criterion="beats an equal-length random block on the %g-%g s bar by %.0f %%"
                   % (BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN),
         extra_lines=["window_reason=%s" % w["reason"]])
    form("window_%s_control" % comp, window=w["control"], seed=w["seed"],
         keep_name="a random block of the same length elsewhere in the record, seed %d" % w["seed"])
'''),

("md", r"""The record with each component's window as a solid span and its equal-length random block
hatched. What to look for is the window sitting where that line's days are alive and the block landing
somewhere the line is neither obviously better nor worse: the two cost the same, so the difference between
their products is what the window bought."""),

("code", '''_t0w, _arrw, _metaw = CACHE.load(SITE, WORK, 1)
spans = []
for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        continue
    spans.append(("%s window, %.1f d" % (comp, w["days"]), FF.SPAN_COLOUR[comp], None,
                  [(w["t_start"], w["t_end"])]))
    spans.append(("%s block, seed %d" % (comp, w["seed"]), "0.4", "//", [w["control"]]))
fig = FF.record_spans(_t0w, _arrw, OUT / "12_windows_record.png", site=SITE, spans=spans,
                      title="%s: the windows and their equal-length blocks" % SITE,
                      caption="The record with each component's proposed window drawn as a solid span in "
                              "that component's colour and its control hatched beside it: a contiguous "
                              "block of the same length placed at random under the seed printed above. The "
                              "window is the longest run of days the elines table calls sound for that line, "
                              "so the control asks whether the answer came from the days chosen or from the "
                              "length alone.")
WRITTEN.append(OUT / "12_windows_record.png")
display(Image(filename=str(OUT / "12_windows_record.png")))
del _arrw
'''),

("code", '''MERGES = []
for comp in COMPONENTS:
    r = made_product("window_%s" % comp)
    if r is None or not BASE:
        continue
    out = OUT / FM.product_name(SITE, "merged_%s" % comp, BASELINE_KIND, 1, PARAMS)
    got = FM.merge_component(BASE, r["product"], comp, out_edi=out, verbose=False)
    MERGES.append(got)
    print("   %s: %s; rows changed %s (expected %s); every other row unchanged: %s"
          % (comp, got["how"], ", ".join(got["rows_changed"]) or "none", ", ".join(got["rows_expected"]),
             got["untouched_unchanged"]))
    FORM_ROWS.append(dict(site=SITE, form="merged_%s" % comp, kind=BASELINE_KIND, rate_hz=1.0,
                          params=PARAMS, product=str(out), controls="", seed=None, status="made",
                          criterion="the whole record for the healthy row and the tipper, the %s window for "
                                    "the other" % comp, error="", days=None, n_runs=None, seconds=None))
if not MERGES:
    print("   no windowed product was made, so nothing was merged")
'''),

("code", '''rows, bad = [], []
for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        bad.append("%s: no window was proposed (%s)" % (comp, w["reason"][:80]))
        continue
    for name in ("window_%s" % comp, "window_%s_control" % comp):
        r = made_product(name)
        if r is None:
            bad.append("%s was not made (%s)" % (name, str(rows_by_form().get(name, {}).get("error"))[:90]))
            continue
        rd = DL.reading(read(r["product"]), read(BASE) if BASE else None,
                        bands=[tuple(b) for b in DECADES], bar_band=tuple(BAR_BAND),
                        agree_band=tuple(AGREE_BAND))
        rows.append(dict(form=name, days=r.get("days"), bar=rd.get("bar"), bar_xy=rd.get("bar_xy"),
                         bar_yx=rd.get("bar_yx"), rho_ratio_xy=rd.get("rho_ratio_xy"),
                         rho_ratio_yx=rd.get("rho_ratio_yx"), phase_xy=rd.get("phase_diff_xy"),
                         per_decade=rd.get("per_decade", "")[:60]))
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no windowed product to read")
beaten = []
for comp in COMPONENTS:
    a = next((r for r in rows if r["form"] == "window_%s" % comp), None)
    b = next((r for r in rows if r["form"] == "window_%s_control" % comp), None)
    if a and b:
        beaten.append((comp, DL.beats(a["bar"], b["bar"], BAR_MARGIN), a["bar"], b["bar"]))
print()
for m in MERGES:
    print("   merge %s: rows changed %s; every other row unchanged %s"
          % (m["component"], ",".join(m["rows_changed"]) or "none", m["untouched_unchanged"]))
if bad:
    print("VERDICT: FAIL -- %s" % "; ".join(bad))
elif not MERGES:
    print("VERDICT: UNJUDGED -- no window was merged, so the merge was not scored")
elif not all(m["ok"] for m in MERGES):
    print("VERDICT: FAIL -- a merge changed a row other than its component's two: %s"
          % "; ".join("%s changed %s" % (m["component"], ",".join(m["rows_changed"]))
                      for m in MERGES if not m["ok"]))
elif not beaten:
    print("VERDICT: UNJUDGED -- no window and control pair was scored")
else:
    promoted = [c for c, ok, _x, _y in beaten if ok]
    inconsistent = [c for c, ok, x, y in beaten if ok and not DL.beats(x, y, BAR_MARGIN)]
    if inconsistent:
        print("VERDICT: FAIL -- %s promoted without beating its random block on the %g-%g s bar by %.0f %%"
              % (", ".join(inconsistent), BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN))
    else:
        print("VERDICT: PASS -- every merge changed exactly its component's two rows (%s) and left the rest "
              "untouched, every window carries an equal-length random block, and the %d promoted window(s) "
              "(%s) beat their block on the %g-%g s bar by at least %.0f %%: %s"
              % ("; ".join("%s -> %s" % (m["component"], ",".join(m["rows_changed"])) for m in MERGES),
                 len(promoted), ", ".join(promoted) or "none", BAR_BAND[0], BAR_BAND[1],
                 100 * BAR_MARGIN,
                 "; ".join("%s %.4f against %.4f%s" % (c, x, y, "" if ok else ", not promoted")
                           for c, ok, x, y in beaten)))
'''),

("md", r"""The windowed product, its random block and the merged file against the whole record. What to
look for is the merged curve following the whole record on the row the window did not touch and the windowed
curve on the row it did: that is the merge rule drawn, and a departure on the untouched row is the failure
the check above scores."""),

("code", '''curves = [("whole", read(BASE), "k", "-")] if BASE else []
styles = {"window": ("-", 0), "control": (":", 1), "merged": ("--", 2)}
for k, comp in enumerate(COMPONENTS):
    for name, key in (("window_%s" % comp, "window"), ("window_%s_control" % comp, "control"),
                      ("merged_%s" % comp, "merged")):
        r = made_product(name)
        if r:
            ls, off = styles[key]
            curves.append((name, read(r["product"]), "C%d" % (3 * k + off), ls))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "13_window_products.png",
                         title="%s: the window products and the merges" % SITE,
                         caption="Each component's windowed pass, the equal-length random block that "
                                 "controls it, and the merged file that carries the window's two impedance "
                                 "rows into a copy of the whole record, all against the whole record in "
                                 "black. The merged curve is the one delivered: only that component's two "
                                 "rows differ from the black, and the tipper and the other row are the "
                                 "whole record's.", period_range=(1, 50000))
    WRITTEN.append(OUT / "13_window_products.png")
    display(Image(filename=str(OUT / "13_window_products.png")))
else:
    print("no windowed product was made, so there is nothing to draw against the whole record")
'''),

("md", r"""## 5. Best hours

The hours of the record are scored by the coherence of the component's pair over HOURS_BAND on
non-overlapping one-hour windows, and the best HOURS_FRACTION of the candidate windows is kept. The candidate
pool is the hours whose day the elines table calls sound; where that leaves fewer than 24 candidates the weak
days are admitted and the selection says so.

Two controls are scored beside it. The random control keeps the same number of windows drawn without
replacement from the same pool under SEED and is expected not to separate kept from dropped; if it does, the
statistic is not measuring what the selection claims. The contiguous controls tile windows of CONTIG_HOURS
hours from the first whole hour, score each by the mean of its hours and take the top ones until the kept
duration matches the selection's to within one window; one Aurora window at the deepest decimation level is
65,536 s, so a scattered selection cannot reach the long periods and only a contiguous comparison at the same
cost decides a long-period claim. The products of all three are read against the baseline.

**This check fails if the ranked selection does not separate kept from dropped on the selecting statistic, if
the random control separates by more than a fifth of the ranked selection's separation, or if a contiguous
control's duration misses the selection's by more than one window.** A random split of a scored pool
separates by a small amount half the time, so the random control's limb is read as a share of the ranked
selection's own separation and not as a sign test, which would be a coin toss and not a criterion."""),

("code", '''HOURS = {}
for comp in COMPONENTS:
    t = time.time()
    HOURS[comp] = MK.best_hours(sv, SITE, comp, ELINES, band=tuple(HOURS_BAND), fraction=HOURS_FRACTION,
                                seed=SEED, contig_hours=tuple(CONTIG_HOURS))
    b = HOURS[comp]
    print("%s in %.0f s: %d candidate window(s) (%s), %d kept above %.3f, %.2f d; random %.2f d"
          % (comp, time.time() - t, b["n_candidates"], b["admitted"], b["n_selected"], b["threshold"],
             b["days_kept"], b["days_random"]))
rows = []
for comp in COMPONENTS:
    b = HOURS[comp]
    rows.append(dict(component=comp, set="ranked", n_windows=b["n_selected"], days=b["days_kept"],
                     median_kept=b["median_kept"], median_dropped=b["median_dropped"],
                     separates=b["separates"], matches=True))
    rows.append(dict(component=comp, set="random", n_windows=b["n_selected"], days=b["days_random"],
                     median_kept=b["random_median_kept"], median_dropped=b["random_median_dropped"],
                     separates=b["random_separates"], matches=True,
                     gap_share_of_ranked=b["random_gap_fraction"]))
    for h, c in sorted(b["contiguous"].items()):
        rows.append(dict(component=comp, set="contiguous %d h" % h, n_windows=c["n_windows"],
                         days=round(c["hours"] / 24.0, 3), median_kept=c["kept"],
                         median_dropped=c["dropped"], separates=c["discriminates"], matches=c["matches"]))
HOURS_TABLE = pd.DataFrame(rows)
print(HOURS_TABLE.round(4).to_string(index=False))
'''),


("code", '''CONTIG_PICK, THIN_HOURS = {}, {}
for comp in COMPONENTS:
    b = HOURS[comp]
    if not b["n_selected"]:
        THIN_HOURS[comp] = b["n_candidates"]
        print("   %s: %d candidate window(s) and %d selected; no product is made"
              % (comp, b["n_candidates"], b["n_selected"]))
        continue
    ok = [(h, c) for h, c in sorted(b["contiguous"].items()) if c["matches"] and c["n_windows"]]
    if not ok:
        print("   %s: no contiguous set matches the selection's duration; none is run" % comp)
        continue
    h, c = min(ok, key=lambda hc: abs(hc[1]["hours"] - b["n_selected"]))
    CONTIG_PICK[comp] = h
    form("hours_%s" % comp, keep_extra=b["keep"], seed=SEED,
         keep_name="the best %.0f %% of %d candidate hour window(s) by %g-%g s coherence"
                   % (100 * HOURS_FRACTION, b["n_candidates"], HOURS_BAND[0], HOURS_BAND[1]),
         controls=["hours_%s_random" % comp, "hours_%s_contig%dh" % (comp, h)],
         criterion="beats its random and contiguous controls on the %g-%g s bar by %.0f %%"
                   % (BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN),
         extra_lines=["selection_statistic=%g-%g s coherence of %s with %s, one value an hour; kept median "
                      "%.3f against the dropped %.3f"
                      % (HOURS_BAND[0], HOURS_BAND[1], LINE[comp], COMP_H[comp], b["median_kept"],
                         b["median_dropped"])])
    form("hours_%s_random" % comp, keep_extra=b["random_keep"], seed=SEED,
         keep_name="%d window(s) drawn at random from the same pool, seed %d" % (b["n_selected"], SEED))
    form("hours_%s_contig%dh" % (comp, h), keep_extra=b["contiguous"][h]["keep"], seed=SEED,
         keep_name="the top %d contiguous %d h window(s), %d h against the selection's %d h"
                   % (b["contiguous"][h]["n_windows"], h, b["contiguous"][h]["hours"], b["n_selected"]))
'''),

("md", r"""The hour score through the record, with the three selections drawn as rows of spans above it:
the kept hours, the random hours of the same size, and the contiguous windows of the same duration. What to
look for is where the kept hours sit relative to the score -- they are its top by construction -- and how
differently the three rows cover the record, because their coverage is the only thing separating them."""),

("code", '''fig = FF.hour_scores(HOURS, SITE, OUT / "14_hour_scores.png", comps=COMPONENTS,
                     contig_pick=CONTIG_PICK)
WRITTEN.append(OUT / "14_hour_scores.png")
display(Image(filename=str(OUT / "14_hour_scores.png")))
'''),

("code", '''rows, bad = [], []
for comp in COMPONENTS:
    b = HOURS[comp]
    if comp in THIN_HOURS:
        bad.append("%s: UNJUDGED -- %d candidate window(s), so nothing was selected and nothing was scored"
                   % (comp, THIN_HOURS[comp]))
        continue
    if not b["separates"]:
        bad.append("%s: the ranked selection's kept median %.3f is not above the dropped %.3f"
                   % (comp, b["median_kept"], b["median_dropped"]))
    if b["random_separates"]:
        bad.append("%s: the RANDOM control separates by %.4f, %.0f %% of the ranked selection's %.4f, above "
                   "the %.0f %% the criterion allows, so the statistic is not measuring the selection"
                   % (comp, b["random_gap"], 100 * b["random_gap_fraction"], b["gap"],
                      100 * b["random_gap_max"]))
    for h, c in sorted(b["contiguous"].items()):
        if not c["matches"]:
            bad.append("%s: the contiguous %d h set keeps %d h against the selection's %d h, more than one "
                       "window out" % (comp, h, c["hours"], b["n_selected"]))
    names = ["hours_%s" % comp, "hours_%s_random" % comp]
    if comp in CONTIG_PICK:
        names.append("hours_%s_contig%dh" % (comp, CONTIG_PICK[comp]))
    for name in names:
        r = made_product(name)
        if r is None:
            continue
        rd = DL.reading(read(r["product"]), read(BASE) if BASE else None,
                        bands=[tuple(x) for x in DECADES], bar_band=tuple(BAR_BAND),
                        agree_band=tuple(AGREE_BAND))
        rows.append(dict(form=name, days=r.get("days"), bar=rd.get("bar"), bar_xy=rd.get("bar_xy"),
                         bar_yx=rd.get("bar_yx"), rho_ratio_xy=rd.get("rho_ratio_xy"),
                         rho_ratio_yx=rd.get("rho_ratio_yx"), jumps_xy=rd.get("jumps_per_decade_xy")))
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no selection product to read")
print()
for comp in COMPONENTS:
    a = next((r for r in rows if r["form"] == "hours_%s" % comp), None)
    ctl = [r for r in rows if r["form"].startswith("hours_%s_" % comp)]
    if a and ctl:
        print("   %s: bar %.4f against %s -- %s"
              % (comp, a["bar"], ", ".join("%s %.4f" % (c["form"].split("_")[-1], c["bar"]) for c in ctl),
                 "promoted" if all(DL.beats(a["bar"], c["bar"], BAR_MARGIN) for c in ctl)
                 else "NOT promoted: efficiency, not a different answer"))
if bad:
    print("VERDICT: FAIL -- %s" % "; ".join(bad))
elif not len(HOURS_TABLE):
    print("VERDICT: UNJUDGED -- no component was scored")
else:
    print("VERDICT: PASS -- on every component the ranked selection separates kept from dropped (%s), the "
          "random control separates by at most a fifth of that (%s), and every contiguous set's duration "
          "matches the selection's to within one window"
          % ("; ".join("%s %.3f over %.3f" % (c, HOURS[c]["median_kept"], HOURS[c]["median_dropped"])
                       for c in COMPONENTS),
             "; ".join("%s %.4f against %.4f" % (c, HOURS[c]["random_gap"], HOURS[c]["gap"])
                       for c in COMPONENTS)))
'''),

("md", r"""The four products of each component on one set of panels: the whole record, the selection, its
random control and its contiguous control. What to look for is the long end, where a scattered selection
cannot reach and the contiguous one can: the deepest Aurora window is 65,536 s, so any claim above a few
thousand seconds belongs to the contiguous curve and not to the scattered one."""),

("code", '''curves = [("whole", read(BASE), "k", "-")] if BASE else []
for k, comp in enumerate(COMPONENTS):
    names = ["hours_%s" % comp, "hours_%s_random" % comp]
    if comp in CONTIG_PICK:
        names.append("hours_%s_contig%dh" % (comp, CONTIG_PICK[comp]))
    for j, name in enumerate(names):
        r = made_product(name)
        if r:
            curves.append((name, read(r["product"]), "C%d" % (3 * k + j), ("-", ":", "--")[j]))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "15_hours_products.png",
                         title="%s: the best-hours products" % SITE,
                         caption="The best hours of each component, the same number of hours drawn at "
                                 "random from the same pool, and the contiguous windows tiled to the same "
                                 "duration, all against the whole record in black. The three selections "
                                 "cost the same number of hours, so a curve that beats the other two beat "
                                 "them on which hours were chosen; the long periods are where an hourly "
                                 "selection is most likely to cost bandwidth rather than buy quality.",
                         period_range=(1, 50000))
    WRITTEN.append(OUT / "15_hours_products.png")
    display(Image(filename=str(OUT / "15_hours_products.png")))
else:
    print("no selection product was made, so there is nothing to draw against the whole record")
'''),

("md", r"""## 6. The shared centre and the arm diagonal

The EDL L layout is three electrodes: a shared centre C, a north arm of length L_N and an east arm of length
L_E, so Ex = (V_N - V_C)/L_N and Ey = (V_E - V_C)/L_E. A noisy centre puts one voltage on both lines, and
because the two lines divide that voltage by different lengths it does not arrive as the same field on both.
With the lines physically signed and a centre voltage n the model reads

    Ex = Ex_true + c,   Ey = Ey_true + s c (L_N / L_E),   c = -n / L_N,

with s = +1 for arms north and east (or south and west) and -1 where one arm is reversed. The E signs in
decisions.csv predict s; an undecided sign is never filled by convention and leaves the site UNJUDGED on that
prediction.

The residual test removes the part of each line that (Hx, Hy) explains, per frequency bin, from the
cross-spectra at 20-200 s on the CENTRE_DAYS days of highest Ex-Ey coherence, and reads what is left: the
complex gain g = S_ry,rx / S_rx,rx of a real shared centre is s L_N / L_E, so the model holds where the
residual coherence is at least 0.9 and |g| divided by the expected L_N / L_E lies between 0.85 and 1.18. The
sign of the real part of the gain is still the observed s. The clean diagonal is the one whose multiple
coherence with (Hx, Hy) is the higher.

The expected gain is L_N / L_E, not 1. With equal arms (Victoria's 50 m) the ratio is 1 and the two forms
coincide; with unequal arms they do not: Q58N, with arms of 9.0 m and 11.9 m, reads a gain of 0.76 at a
residual coherence of 1.00, and 9.0/11.9 = 0.756, so its two lines share one voltage. The arm lengths come
from sites.csv dipole_n_m and dipole_e_m; an `assume:` cell is used and named, and a site with no lengths on
file is UNJUDGED on this criterion.

The control is built, not found. The site's own Ex is paired with the nearest sound site's Ey, both read
against the site's own (Hx, Hy) over the same days, and the residual test is run on that pair: two electrodes
tens of kilometres apart have no common voltage, so the model must not hold there. A control site whose own
Ex-Ey coherence stays under 0.35 on every day need not exist -- a one-dimensional earth correlates the two
lines through the source field alone, and no site of AusLAMP Queensland Phase 1 clears that ceiling -- so the
found site is kept as a reading beside the built pair.

Where the model holds, the remedy is the arm diagonal: the voltage between the two arm electrodes carries no
centre at any pair of lengths, because the centre cancels in the difference of the two arm potentials.

    V_N - V_E = L_N Ex - L_E Ey,   d = sqrt(L_N^2 + L_E^2),   Ex' = (L_N Ex - L_E Ey) / d,

which is the field along the unit vector (L_N, -L_E)/d in (north, east), that is at theta = atan2(-L_E, L_N)
from north, over the separation d of the two arm electrodes. The orthogonal row Ey' = (L_E Ex + L_N Ey)/d
carries the centre and is kept for the record. Equal arms give theta = -45 deg and the pair reduces to
(Ex - Ey)/sqrt 2 and (Ex + Ey)/sqrt 2 exactly. The pair is E in the frame turned by theta, so a pass on the
variant cache gives R(theta) Z, and turning the H columns as well completes Z' = (R Z) R^T and T' = T R^T at
that theta. The x' row is the clean one; the y' row is kept for the record. The turn is checked on the elements to 1e-6 relative, on the determinant to 1e-5
of the squared Frobenius norm -- the relative form is meaningless at a near-singular period -- and on the
Frobenius norm to 1e-6. The trace is not an invariant of a column-only turn and its ratio is printed as the
counter-example.

**This check fails if the built control holds the model, if it could not be built, if the variant cache does
not reproduce (L_N Ex - L_E Ey)/d of the signed source at every finite sample, if the turn-back changes an
invariant beyond its tolerance, or if a sign prediction disagrees with the observed sign where both
exist.**"""),

("code", '''ARMS = CE.arm_lengths(sv, SITE)
G_EXPECTED = CE.expected_gain(ARMS["L_N"], ARMS["L_E"])
THETA = CE.diagonal_angle(ARMS["L_N"], ARMS["L_E"])
print("the arms at %s: %s" % (SITE, ARMS["note"]))
print("   the model predicts |g| = L_N / L_E = %.3f, and the arm diagonal lies at %+.2f deg from north over "
      "a separation of %.2f m (equal arms would give 1.000 and -45.00 deg)"
      % (G_EXPECTED, THETA, CE.diagonal_length(ARMS["L_N"], ARMS["L_E"])))
if ARMS["assumed"]:
    print("   %s carries an assumed length, which the product's provenance carries with it"
          % ", ".join(ARMS["assumed"]))
print()
BUILT = CE.built_control(sv, SITE, days=CENTRE_DAYS, elines=ELINES, members=SITES)
print("the built control: %s -- %s" % (BUILT.get("site"), BUILT.get("reason")))
CTRL = CE.control_site(sv, SITE, SITES)
print("the found control site (a reading): %s -- %s" % (CTRL.get("site"), CTRL.get("reason")))
if "table" in CTRL:
    print(CTRL["table"].head(6).round(3).to_string(index=False))
print("it qualifies under the %.2f ceiling on every scored day: %s"
      % (CE.CONTROL_EX_EY_MAX, CTRL.get("qualifies")))
print()
CENTRE = {SITE: CE.residual_test(sv, SITE, CENTRE_DAYS, ELINES)}
if BUILT.get("judged"):
    CENTRE[BUILT["site"]] = BUILT
if CTRL.get("site"):
    CENTRE[CTRL["site"]] = CE.residual_test(sv, CTRL["site"], CENTRE_DAYS,
                                            CE.read_elines(sv, CTRL["site"]))
cols = ["site", "days", "src_coh", "resid_coh", "L_N", "L_E", "gain_expected", "resid_gain", "gain_ratio",
        "s_pred", "s_obs", "sign_undecided", "mcoh_diff", "mcoh_sum", "model_holds", "control",
        "clean_pred", "clean_obs", "clean_by_H", "sign_agrees", "remedy_applicable"]
CENTRE_TABLE = pd.DataFrame([{k: v.get(k) for k in cols} for v in CENTRE.values() if v.get("judged")])
print(CENTRE_TABLE.round(3).to_string(index=False) if len(CENTRE_TABLE) else "no site produced a usable day")
'''),

("md", r"""The residual test drawn: the residual coherence and the complex gain over the band, for the site
and for the built control, with the 0.9 coherence line and the 0.85-1.18 band on the gain divided by the
L_N / L_E the arms predict, so 1 is what a shared centre would give at this site's lengths. What to look for
is the site's coherence riding along the top of the band with a scaled gain sitting inside the box while the
built control's coherence lies on the floor: the gap between the two lines is what says the test can tell a
shared centre from a pair that cannot have one."""),

("code", '''fig = FF.residual_panels(sv, SITE, CENTRE.get(SITE, {}), BUILT, OUT / "16_centre_residual.png",
                         days=CENTRE_DAYS, elines=ELINES, band_s=CE.BAND_S)
WRITTEN.append(OUT / "16_centre_residual.png")
display(Image(filename=str(OUT / "16_centre_residual.png")))
'''),

("code", '''NE, DIAG = dict(written=False), None
me = CENTRE.get(SITE, {})
if me.get("model_holds"):
    NE = CE.ne_variant(sv, SITE, 1, force=REDO)
    print("the NE variant cache: %s (%s; Ex' equals (L_N Ex - L_E Ey)/d of the signed source at every one of "
          "the %s finite samples: %s)"
          % (NE.get("path"), ("rewritten: the cache on disk was built at other arm lengths or another frame"
                              if NE.get("refreshed") else
                              "written" if NE.get("written") else "already on disk and checked again"),
             NE.get("n_finite"), NE.get("exact")))
    # a pass on a cache that has just been rebuilt at a different frame is not a pass on this cache
    DIAG = form("diagonal", variant="ne", apply_e_signs=False, turn_ne=True, turn_angle_deg=THETA,
                redo=bool(REDO or NE.get("refreshed")),
                criterion="the turn-back keeps the element, determinant and Frobenius invariants of a "
                          "column-only turn",
                extra_lines=["diagonal=x' along the arm diagonal, (L_N Ex - L_E Ey)/d with L_N %.4g m, "
                             "L_E %.4g m and d %.4g m, the clean row where the model holds; y' orthogonal "
                             "to it, (L_E Ex + L_N Ey)/d, kept for the record"
                             % (ARMS["L_N"], ARMS["L_E"], CE.diagonal_length(ARMS["L_N"], ARMS["L_E"])),
                             "frame_deg=%+.4f = atan2(-L_E, L_N); equal arms would give -45" % THETA])
    t = DIAG.get("turn") or {}
    if t:
        print("   the turn-back over %d period(s): elements %.2e (tolerance %.0e), determinant %.2e scaled "
              "(tolerance %.0e), Frobenius %.2e (tolerance %.0e); the trace ratio is %.3f, which is why the "
              "trace is not one of the criteria"
              % (t["n"], t["max_element_rel"], CE.ELEMENT_RTOL, t["max_det_scaled"], CE.DET_SCALED_TOL,
                 t["max_frobenius_rel"], CE.FROBENIUS_RTOL, t["trace_ratio"]))
    print("   the remedy %s at %s: the clean diagonal by the observed sign is the %s and by H coherence the "
          "%s" % ("applies" if me.get("remedy_applicable") else "does NOT apply", SITE,
                  me.get("clean_obs"), me.get("clean_by_H")))
else:
    print("the model does not hold at %s (residual coherence %.2f, gain %.2f against the %.3f the arms "
          "predict, a ratio of %.2f): no variant cache is written"
          % (SITE, me.get("resid_coh", np.nan), me.get("resid_gain", np.nan), G_EXPECTED,
             me.get("gain_ratio", np.nan)))
'''),

("md", r"""One day of the two signed lines above and the two arm diagonals below. What to look for is the two
lines moving together -- that common motion is the centre's voltage, arriving on each line divided by its own
arm length -- and the length-weighted difference beneath them, which is what the arm diagonal keeps after it
has cancelled."""),

("code", '''fig = FF.diagonal_day(sv, SITE, OUT / "17_diagonal_day.png", elines=ELINES)
WRITTEN.append(OUT / "17_diagonal_day.png")
display(Image(filename=str(OUT / "17_diagonal_day.png")))
'''),

("md", r"""The diagonal product against the baseline, where one was built. The x' row is the arm diagonal and
not the xy component, so the two curves are not the same quantity and are not expected to lie on each other:
what to look for is whether the x' row is smoother and carries a smaller bar than the row the shared centre
sits in."""),

("code", '''r = made_product("diagonal")
if r and BASE:
    fig = FF.form_panels([("whole", read(BASE), "k", "-"), ("diagonal", read(r["product"]), "C0", "-")],
                         SITE, OUT / "18_diagonal_product.png",
                         title="%s: the diagonal pass against the baseline" % SITE,
                         caption="The pass on the arm-diagonal cache, turned back by theta = %+.2f deg = "
                                 "atan2(-L_E, L_N) with L_N = %.4g m and L_E = %.4g m, against the "
                                 "whole-record baseline in black. The x' row is the field along the arm "
                                 "diagonal and not the xy component, so the two are different quantities "
                                 "and are not expected to lie on each other; what the figure is for is the "
                                 "size of the error bars on the row the shared centre sits in."
                                 % (THETA, ARMS["L_N"], ARMS["L_E"]), period_range=(1, 50000))
    WRITTEN.append(OUT / "18_diagonal_product.png")
    display(Image(filename=str(OUT / "18_diagonal_product.png")))
else:
    print("no diagonal product was built, so there is nothing to draw against the baseline")
'''),

("code", '''fail = []
me = CENTRE.get(SITE, {})
ctrl_row = CENTRE.get(CTRL.get("site"), {})
if not BUILT.get("judged"):
    fail.append("the control could not be built (%s), so the test was not shown able to fail"
                % BUILT.get("reason"))
elif BUILT.get("model_holds"):
    fail.append("the built control %s holds the model (residual coherence %.2f, gain %.2f, a ratio of %.2f "
                "to the %.3f the arms predict): the test finds a shared centre between two electrodes that "
                "cannot have one"
                % (BUILT["site"], BUILT["resid_coh"], BUILT["resid_gain"],
                   BUILT.get("gain_ratio", np.nan), G_EXPECTED))
if me.get("sign_undecided"):
    fail.append("UNJUDGED on the sign prediction at %s: %s undecided in decisions.csv, never filled by "
                "convention" % (SITE, me["sign_undecided"]))
elif me.get("model_holds") and me.get("sign_agrees") is False:
    fail.append("the sign prediction %+d disagrees with the observed %+d at %s"
                % (me["s_pred"], me["s_obs"], SITE))
if NE.get("exact") is False:
    fail.append("the variant cache does not reproduce (L_N Ex - L_E Ey)/d of the signed source at every one "
                "of its %s finite samples" % NE.get("n_finite"))
if DIAG and (DIAG.get("turn") or {}) and not (DIAG.get("turn") or {}).get("ok"):
    t = DIAG["turn"]
    fail.append("the turn-back moves an invariant: elements %.2e, determinant %.2e, Frobenius %.2e"
                % (t["max_element_rel"], t["max_det_scaled"], t["max_frobenius_rel"]))
t = (DIAG or {}).get("turn") or {}
sign = lambda v: ("%+d" % v) if isinstance(v, (int, float)) and np.isfinite(v) else "undecided"
turn_text = (("the turn-back on the diagonal holds the three invariants: elements %.1e, determinant %.1e "
              "scaled, Frobenius %.1e, with the trace ratio %.3f as the counter-example"
              % (t.get("max_element_rel", np.nan), t.get("max_det_scaled", np.nan),
                 t.get("max_frobenius_rel", np.nan), t.get("trace_ratio", np.nan))) if t
             else "no diagonal was built, so the turn-back was not scored")
read_text = ("%s has arms of %.4g m north and %.4g m east, so the model predicts |g| = %.3f; it reads a "
             "residual coherence of %.2f with a gain of %.2f, a ratio of %.2f, so the model %s there, and "
             "its observed sign %s %s the %s its E signs predict"
             % (SITE, ARMS["L_N"], ARMS["L_E"], G_EXPECTED, me.get("resid_coh", np.nan),
                me.get("resid_gain", np.nan), me.get("gain_ratio", np.nan),
                "HOLDS" if me.get("model_holds") else "does NOT hold", sign(me.get("s_obs")),
                "agrees with" if me.get("sign_agrees") else "is not scored against",
                sign(me.get("s_pred"))))
found_text = ("the found control site %s %s the %.2f ceiling (daily maximum %.3f) and reads a residual "
              "coherence of %.2f"
              % (CTRL.get("site"), "clears" if CTRL.get("qualifies") else "does not clear",
                 CE.CONTROL_EX_EY_MAX, CTRL.get("max_coh_Ex_Ey", np.nan),
                 ctrl_row.get("resid_coh", np.nan))) if CTRL.get("site") else "no control site was found"
if fail:
    print("VERDICT: FAIL -- %s; %s; %s; %s" % ("; ".join(fail), read_text, turn_text, found_text))
else:
    print("VERDICT: PASS -- the built control %s does not hold the model (residual coherence %.2f, gain "
          "%.2f); %s; %s; as a reading, %s"
          % (BUILT.get("site"), BUILT.get("resid_coh", np.nan), BUILT.get("resid_gain", np.nan),
             read_text, turn_text, found_text))
'''),

("md", r"""## 7. The 10 Hz forms

A site recorded at 10 Hz carries a short end the 1 Hz cache cannot reach. This section passes the whole 10 Hz
record against the same reference kind the 1 Hz baseline used and draws the two rates on one page.

The reference is BASELINE_KIND at 10 Hz, which is the remote's own 10 Hz record on this site's grid, and
where no store has been built at that rate the form is refused with that reason and no pass is run. The
single station is not a kind of this package, so a missing store is not fallen back from: a product estimated
on the site's own H is biased by its own noise and its error bars do not show the bias.

A form the method's own floor refuses is a reading; a form that crashes is a failure. Before the form reaches
Aurora the workbook measures what its cache leaves against the reference it will be passed with -- the mask,
the reference's own coverage, and the record cut into runs at the 3,600 s floor -- and prints the kept days
and the run count beside the whole record's. A form whose kept duration is zero is not passed: its row reads
`refused` with those numbers and the figure carries that sentence where its curve would have been, rather
than a traceback out of the estimator.

**This check fails if any form of this section ended in an exception rather than a stated refusal.**"""),

("code", '''REFUSED, COVER = [], {}
if 10 not in RATES:
    print("RATES does not include 10: this section reads the 10 Hz cache and is not run")
elif not (FM.cache_dir(WORK, 10) / ("%s.npz" % SITE)).exists():
    print("no 10 Hz cache at %s: this instrument records at 1 Hz"
          % (FM.cache_dir(WORK, 10) / ("%s.npz" % SITE)))
else:
    kind10 = BASELINE_KIND
    NO_STORE_10 = not (WORK / "references" / "10hz" / ("%s_%s.npz" % (kind10, SITE))).exists()
    if NO_STORE_10:
        # the single station is not a kind of this package, so a missing store refuses the form rather
        # than falling back to one: a product estimated on the site's own H is biased by its own noise
        why = ("refused: there is no 10 Hz %s reference store for %s, and the single station is not a kind "
               "of this package" % (kind10, SITE))
        FORM_ROWS.append(FM.refused_row(sv, SITE, "whole10", OUT, kind10, 10, PARAMS, why,
                                        criterion="not passed: the reference this form needs is not on "
                                                  "disk"))
        REFUSED.append(("whole10", why))
        print("   whole10 is NOT passed -- %s" % why)
    else:
        # what the cache leaves against the reference it will be passed with, measured BEFORE the pass: the
        # mask, the reference's own coverage and the run floor, cut into the runs Aurora would be handed. A
        # cache that leaves nothing is refused here with its numbers, rather than crashing the estimator
        COVER["whole10"] = FM.coverage(sv, SITE, kind=kind10, rate=10)
        c = COVER["whole10"]
        print("what the 10 Hz cache leaves against the %s reference, before the pass: the mask, the "
              "reference's own coverage, and the record cut into runs at the %g s floor"
              % (KIND_WORD.get(kind10, kind10), TR_MIN_SEGMENT_S))
        print("   %-9s %6.2f d over %3d run(s) of the record's %.2f d; %.1f %% of the samples kept, "
              "%.1f %% of the record lost to the floor"
              % ("whole10", c["days"], c["n_runs"], c["record_days"], 100 * c["kept_frac"],
                 100 * c["floor_dropped_frac"]))
        if c["empty"]:
            why = FM.refusal_sentence(c, c, what="the mask")
            FORM_ROWS.append(FM.refused_row(sv, SITE, "whole10", OUT, kind10, 10, PARAMS, why, cov=c,
                                            criterion="not passed: the floor leaves no run to hand "
                                                      "Aurora"))
            REFUSED.append(("whole10", why))
            print("   whole10 is NOT passed -- %s" % why)
        else:
            form("whole10", kind=kind10, rate=10,
                 criterion="the 10 Hz record against the same reference kind as the 1 Hz baseline")
    print()
    print("what the form cost the pass")
    r = rows_by_form().get("whole10", {})
    print("   %-9s %s d kept over %s run(s)%s"
          % ("whole10", r.get("days"), r.get("n_runs"),
             ("  [%s]" % r["reason"]) if r.get("reason") else ""))
'''),

("md", r"""The 10 Hz form against the 1 Hz whole-record baseline, which is the only place in the workbook
that says what the 10 Hz pass produced. What to look for is the dashed curve lying on the solid one where the
two rates overlap, the size of the error bars at the short end, and the step across the join line: the two
shaded bands are the ones a splice would score that step on, and the octave between them is the guard that
holds the logger's own instrument line and is scored by nothing."""),

("code", '''rows = rows_by_form()
curves = [("whole, 1 Hz", read(BASE), "k", "-")] if BASE else []
r = made_product("whole10")
if r:
    curves.append(("whole10, 10 Hz", read(r["product"]), "C0", "--"))
if len(curves) > 1:
    _bs = AR.bands_for(10)
    _k10 = rows.get("whole10", {}).get("kind", BASELINE_KIND)
    _cost = "%s s over %s run(s)" % (rows.get("whole10", {}).get("seconds"),
                                     rows.get("whole10", {}).get("n_runs"))
    # a form the floor refused carries its sentence where its curve would have been
    _gone = " ".join("%s is not drawn -- %s." % (n, w) for n, w in REFUSED)
    fig = FF.rate_panels(curves, SITE, OUT / "19_rates.png", join_s=SP.SPLICE_JOIN_S,
                         bands=(SP.STEP_BELOW, SP.STEP_ABOVE), period_range=(SP.SHORT_FLOOR_S, 2000),
                         title="%s: the 10 Hz form against the 1 Hz baseline" % SITE,
                         caption="The whole 10 Hz record dashed, on the same %s reference and the same %s "
                                 "parameters, against the 1 Hz whole-record baseline solid, from %g s to "
                                 "2,000 s. The 10 Hz pass reads the band file %s (%d levels, window %d "
                                 "samples) and the 1 Hz baseline reads %s (%d levels), so the two rates "
                                 "carry different band edges and land on different periods. The dash-dotted "
                                 "line at %g s is where a short end would join the 1 Hz row; the shaded "
                                 "columns %g-%g s and %g-%g s are the two bands the step at that join is "
                                 "scored on, and the octave between them is the guard band, scored by "
                                 "nothing. The pass took %s, and it ran on its own because one 10 Hz pass "
                                 "holds the whole record in memory. %s Caveat: %s."
                                 % (KIND_WORD.get(_k10, _k10), PARAMS, SP.SHORT_FLOOR_S,
                                    _bs.file.name, _bs.levels, _bs.window,
                                    AR.bands_for(1).file.name, AR.bands_for(1).levels,
                                    SP.SPLICE_JOIN_S, SP.STEP_BELOW[0], SP.STEP_BELOW[1],
                                    SP.STEP_ABOVE[0], SP.STEP_ABOVE[1], _cost, _gone, TEN_HZ_CAVEAT))
    WRITTEN.append(OUT / "19_rates.png")
    display(Image(filename=str(OUT / "19_rates.png")))
else:
    print("no 10 Hz form was made, so there is nothing to draw against the 1 Hz baseline")
'''),

("code", '''# a form that ended in an exception is a FAIL with the form named; a form the floor refused, with its
# runs and its days measured before the pass, is a reading and is not one
fail = ["the form %s ended in an exception rather than a stated refusal: %s"
        % (r["form"], str(r.get("error"))[:160])
        for r in FORM_ROWS if r.get("rate_hz") == 10.0 and r.get("status") == "FAILED"]
refused_text = "; ".join("%s %s" % (n, w) for n, w in REFUSED)
made10 = [r["form"] for r in FORM_ROWS if r.get("rate_hz") == 10.0 and r.get("status") in ("made", "exists")]
if fail:
    print("VERDICT: FAIL -- %s%s" % ("; ".join(fail), ("; " + refused_text) if refused_text else ""))
elif not made10 and not REFUSED:
    print("VERDICT: UNJUDGED -- no form of this section was reached, so nothing was scored")
else:
    print("VERDICT: PASS -- no form of this section ended in an exception: %s"
          % ("; ".join("%s was made" % n for n in made10) if made10 else "none was made",))
    if refused_text:
        print("   as a reading, %s" % refused_text)
'''),

("md", r"""## 8. Replacement magnetics and the lender

At long period the horizontal magnetic field is homogeneous over the site spacing, so a neighbour's H
measures the same field; at short period it is not. A borrowed long end therefore goes under the site's own
short periods and the two are spliced where borrowing stops costing.

The rule picks the worse channel only, and only where it is clearly worse than the other: a site
decorrelated from its neighbours for any reason -- distance, a quiet spell, its own noise -- has both
channels below any threshold, so a rule reading "any channel below a threshold" replaces both and wrecks the
site. The coherence is the whole-record mean over 100-1000 s, deliberately not the chunk median used
elsewhere, and both series are despiked first, because a whole-record mean has no median to hide behind and
one logger spike sets the number. The candidate must itself be sound against a third site at 0.80: a donor
vouched for by the site it stands in for has been vouched for by nobody. A donor that is a member of the
reference the pass reads is refused, because a reference sharing a channel with the local H is comparing a
channel with itself.

Four forms. A is the site's own H, the baseline of section 1. B replaces one channel with the nearest sound
site's same channel and keeps the site's own other channel. C replaces it with the fleet stack's channel and
is shown and never delivered, because a stack is a reference and never a local H; its pass reads the
observatory, since every member of the stack is inside its own local H. D borrows the whole pair: the tensor
of the site's E on the field a neighbour measured, which is an inter-site impedance, shown and never
delivered. D and the lender form are one construction and one pass.

The frames. The lender's pair is served in its own mean-field frame and is turned by minus the site's angle
into the site's sensor frame before it stands in for a sensor-frame channel; the tensor is then turned back
on the right by M R(-t), with M's row taken from the identity where the channel is borrowed and from R(t)
where it is the site's own. With both channels borrowed that is the rotation R(-t); with neither it is the
identity; with one of each it is not a rotation at all.

**This check fails if a delivered form borrows both horizontal channels, if a replacement is not turned into
the site's frame (the read-back identity of the basis correction), or if the lender is a member of the
reference the pass reads.**"""),

("code", '''t = time.time()
_t0r, _hr, ANGLE, _nr = RP.rotated_pair(sv, SITE, 1)
ANGLE = FM.mean_angle(ANGLE)
try:
    _a, _b, _c, REF_INFO = REF.load_reference(BASELINE_KIND, SITE, 1, WORK)
except Exception as exc:
    REF_INFO = {}
    print("no %s reference sidecar: %s" % (BASELINE_KIND, str(exc)[:110]))
IN_REF = RP.reference_members(REF_INFO)
CANDS = RP.candidates(sv, SITE, donors=SITES, max_donors=5, exclude=IN_REF)
print("the %s reference this pass reads is built from %s, and those sites cannot lend"
      % (KIND_WORD.get(BASELINE_KIND, BASELINE_KIND), ", ".join(IN_REF) or "nothing"))
print("the site's frame angle is %+.4f deg" % ANGLE)
print(CANDS.round(3).to_string(index=False) if len(CANDS)
      else "no candidate covers %g days of the record" % RP.MIN_OVERLAP_DAYS)
print("(the candidates table took %.0f s)" % (time.time() - t))
fig = FF.candidate_bars(CANDS, SITE, OUT / "23_candidates.png", threshold=RP.SUB_THRESHOLD,
                        donor_gate=RP.DONOR_GATE)
WRITTEN.append(OUT / "23_candidates.png")
display(Image(filename=str(OUT / "23_candidates.png")))
'''),

("code", '''CHAN, LEND = REPLACE_CHANNEL, LENDER
if len(CANDS):
    if CHAN is None:
        fired = CANDS[CANDS.rule_fires]
        CHAN = str((fired if len(fired) else CANDS).worst_channel.iloc[0])
    if LEND is None:
        ok = CANDS[CANDS.donor_sound] if CANDS.donor_sound.any() else CANDS
        LEND = str(ok.candidate.iloc[0])
NEEDED = bool(len(CANDS) and CANDS.rule_fires.any())
print("the substitution rule fires at %s: %s"
      % (SITE, "yes" if NEEDED else "no -- the site needs no replacement, and form B below is a "
         "demonstration on the channel the rule would pick"))
print("channel %s, lender %s%s" % (CHAN, LEND, "" if NEEDED else " (demonstration)"))
'''),

("code", '''REPLACED = []
if LEND and CHAN:
    t0l, hl, _angl, _nl = RP.rotated_pair(sv, LEND, 1)
    lend_pair = RP.on_grid(hl, t0l, t0_rec, n_rec)
    _t0b, local_b, _ang_b, _ap, _un = FM.load_local(sv, SITE, 1)
    b = RP.replace_channel(local_b, lend_pair, CHAN, ANGLE)
    print("form B: %s" % b["note"])
    REPLACED.append(dict(name="replace_%s_%s" % (CHAN, LEND), spec=b, shown_only=False, lender=LEND,
                         kind=BASELINE_KIND,
                         why="scored per decade against the own-H baseline, the worst decade governing"))
    d = RP.replace_channel(local_b, lend_pair, CHAN, ANGLE, whole_pair=True)
    print("form D: %s" % d["note"])
    REPLACED.append(dict(name="lender_%s" % LEND, spec=d, shown_only=True, lender=LEND,
                         kind=BASELINE_KIND,
                         why="shown and never delivered: an inter-site impedance"))
    try:
        _st0, sh, smask, _si = REF.load_reference("stack", SITE, 1, WORK)
        stack_pair = {c: np.where(np.asarray(smask, bool)[:n_rec], np.asarray(sh[c], float)[:n_rec], np.nan)
                      for c in ("Hx", "Hy")}
        c_ = RP.replace_channel(local_b, stack_pair, CHAN, ANGLE)
        print("form C: %s, the channel taken from the fleet stack; its pass reads the observatory, because "
              "every member of the stack is inside its own local H" % c_["note"])
        REPLACED.append(dict(name="replace_%s_stack" % CHAN, spec=c_, shown_only=True, lender=None,
                             kind="obs",
                             why="shown and never delivered: a stack is a reference, never a local H"))
    except Exception as exc:
        print("form C: no fleet stack store for %s (%s)" % (SITE, str(exc)[:90]))
    del local_b, lend_pair
else:
    print("no candidate lender: no replacement form is built")
'''),

("code", '''for item in REPLACED:
    r = form(item["name"], kind=item["kind"], local_h=item["spec"]["arrays"],
             correction=item["spec"]["correction"], lender=item["lender"], criterion=item["why"],
             extra_lines=["h_replacement=%s" % item["spec"]["note"],
                          "h_basis_correction_is_a_rotation=%s" % item["spec"]["is_rotation"]])
    r["inter_site"] = bool(item["shown_only"] or item["spec"]["inter_site"])
    r["borrowed"] = "Hx+Hy" if all(item["spec"]["borrowed"]) else (CHAN or "")
'''),

("code", '''rows = []
for item in REPLACED:
    r = made_product(item["name"])
    if r is None:
        rows.append(dict(form=item["name"],
                         error=str(rows_by_form().get(item["name"], {}).get("error"))[:110]))
        continue
    tf = read(r["product"])
    worst = ""
    if BASE:
        per = AG.per_decade(read(BASE), tf, bands=[tuple(b) for b in DECADES])
        p = per[np.isfinite(per.rho_ratio)]
        if len(p):
            k = int(np.nanargmax(np.abs(np.log10(p.rho_ratio.values))))
            worst = "%s %s %.2f, %+.1f deg" % (p.band.iloc[k], p.component.iloc[k], p.rho_ratio.iloc[k],
                                               p.phase_diff_deg.iloc[k])
    rows.append(dict(form=item["name"], kind=item["kind"], borrowed=r.get("borrowed"),
                     delivered=not r.get("inter_site"), bar_xy=DL.bar(tf, "xy", *BAR_BAND),
                     bar_yx=DL.bar(tf, "yx", *BAR_BAND), worst_decade_against_A=worst,
                     correction_reads_back=(r.get("turn") or {}).get("ok"),
                     correction_element=(r.get("turn") or {}).get("max_element_rel")))
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no replacement form was built")
'''),

("md", r"""Forms A to D on one set of panels: A the own-H baseline in black, B the single borrowed channel,
C the stack's channel and D the whole borrowed pair, the two shown-only forms drawn dashed. What to look for
is B following A while D sits off it: a single borrowed channel is a measurement of the same field, and a
whole borrowed pair is the inter-site impedance, which is a different quantity and is never delivered."""),

("code", '''curves = [("A: whole, own H", read(BASE), "k", "-")] if BASE else []
for j, item in enumerate(REPLACED):
    r = made_product(item["name"])
    if r:
        curves.append(("%s%s" % (item["name"], " (shown only)" if item["shown_only"] else ""),
                       read(r["product"]), "C%d" % j, "--" if item["shown_only"] else "-"))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "22_forms_abcd.png",
                         title="%s: the replacement forms against the own-H baseline; the dashed forms are "
                               "shown and never delivered" % SITE, period_range=(1, 50000))
    WRITTEN.append(OUT / "22_forms_abcd.png")
    display(Image(filename=str(OUT / "22_forms_abcd.png")))
else:
    print("no replacement form was built, so there is nothing to draw against the baseline")
'''),

("code", '''SPLICE = {}
b_name = "replace_%s_%s" % (CHAN, LEND) if (CHAN and LEND) else ""
r = made_product(b_name) if b_name else None
if BASE and r:
    a, b = read(BASE), read(r["product"])
    bg = AG.on_grid(a, b)
    for comp in ("xy", "yx"):
        ra, _ea, pa, _fa = PR.rho_phase(a.period, a.z, a.z_err, comp)
        rb, _eb, pb, _fb = PR.rho_phase(a.period, bg.z, bg.z_err, comp)
        SPLICE[comp] = RP.splice_period(a.period, rb, pb, ra, pa)
    print("the period from which the borrowed row stops costing, read against the own-H baseline:")
    for comp, s in SPLICE.items():
        print("   %s: t_c %s s over %d tested period(s), %.0f %% agreeing -- %s"
              % (comp, ("%.0f" % s["t_c"]) if np.isfinite(s["t_c"]) else "none", s["n_tested"],
                 100 * s["frac_agree"], s["rule"]))
    print("   below t_c the site's own short periods are kept and above it the borrowed row is delivered; a "
          "form with no t_c has no defensible splice and is not merged")
else:
    print("no single-channel replacement product to splice")
'''),

("code", '''fail = []
for item in REPLACED:
    r = rows_by_form().get(item["name"], {})
    if r.get("status") not in ("made", "exists"):
        if "lender" in str(r.get("error") or ""):
            fail.append("%s: %s" % (item["name"], r["error"]))
        continue
    if all(item["spec"]["borrowed"]) and not r.get("inter_site"):
        fail.append("%s borrows both horizontal channels and is not labelled an inter-site impedance"
                    % item["name"])
    t = r.get("turn") or {}
    if t and not t.get("ok"):
        fail.append("%s: the basis correction does not read back (elements %.2e, determinant %.2e)"
                    % (item["name"], t.get("max_element_rel", np.nan), t.get("max_det_scaled", np.nan)))
    if item["lender"] and item["lender"] in IN_REF:
        fail.append("the lender %s is a member of the %s reference %s reads"
                    % (item["lender"], item["kind"], item["name"]))
if not REPLACED:
    print("VERDICT: UNJUDGED -- no candidate lender was found, so no form was built and nothing was scored")
elif fail:
    print("VERDICT: FAIL -- %s" % "; ".join(fail))
else:
    print("VERDICT: PASS -- %d form(s) built; the lender %s is not a member of the %s reference (%s); every "
          "whole-pair form is labelled an inter-site impedance and is not delivered (%s); and every basis "
          "correction reads back to better than 1e-6 on the elements"
          % (len(REPLACED), LEND, BASELINE_KIND, ", ".join(IN_REF) or "no member",
             ", ".join(i["name"] for i in REPLACED if all(i["spec"]["borrowed"])) or "none built"))
'''),

("md", r"""## 9. The tipper-only delivery

The tipper is an H-only quantity and survives two dead electric lines, so a site with no deliverable
impedance can still deliver its tipper. The impedance rows are written as the empty-data fill and two INFO
lines say what the file is, so a reader cannot take the blank rows for a measurement.

The tipper is refused where the vertical channel is not measuring the vertical field, and neither fault is
visible in the tipper itself: Hz a copy of a horizontal channel, which reads a coherence of 1.00 with Hx and
makes the tipper a re-statement of the horizontal record, or a leak, where Hz carries the site's own
horizontal field at 1000-4000 s while carrying nothing of a neighbour's vertical field. A real vertical field
is coherent with a neighbour's at those periods, because the source is regional.

This section is a reading and not a check: it applies where both lines are dead, or on demand."""),

("code", '''TIP = DL.tipper_refusal(sv, SITE)
print("Hz with its own Hx %.3f (a copy at or above %.2f); with its own H %.3f (a leak at or above %.2f) "
      "while with %s's Hz %.3f (under %.2f)"
      % (TIP.get("coh_hz_hx", np.nan), DL.COPY_COH, TIP.get("coh_hz_own_h", np.nan), DL.LEAK_OWN_COH,
         TIP.get("neighbour"), TIP.get("coh_hz_neighbour_hz", np.nan), DL.LEAK_NEIGHBOUR_COH))
print("the refusal test: %s -- %s" % ("REFUSED" if TIP.get("refused") else "the tipper stands",
                                      TIP.get("reason")))
dead_lines = [c for c in COMPONENTS
              if len(ELINES) and (ELINES["%s_state" % LINE[c]] == "dead").mean() > 0.5]
if BASE and (len(dead_lines) == len(COMPONENTS) or TIP.get("refused")):
    out = OUT / FM.product_name(SITE, "tipper_only", BASELINE_KIND, 1, PARAMS)
    got = DL.tipper_only(BASE, out, KIND_WORD.get(BASELINE_KIND, BASELINE_KIND), TIP)
    print(got)
    if got.get("written"):
        FORM_ROWS.append(dict(site=SITE, form="tipper_only", kind=BASELINE_KIND, rate_hz=1.0, params=PARAMS,
                              product=str(out), controls="", seed=None, status="made",
                              criterion="an H-only delivery; the impedance rows carry no product of record",
                              error="", days=None, n_runs=None, seconds=None))
else:
    print("neither line is dead on most days (%s) and the tipper is not refused, so the tipper-only "
          "delivery does not apply here: the tipper ships inside the impedance products"
          % (", ".join(dead_lines) or "no line dead"))
'''),

("md", r"""## 10. The recipe

The sections above each change one thing and score it. This one composes them: a frame for the product and,
per impedance row, which hours that row is estimated on and at which sample rate. One row can come from the
whole record at 1 Hz and the other from a short window at 10 Hz, and the two are assembled into one tensor.

The cell below is the one to edit. `frame` is the frame both rows are passed in, so nothing is turned after
the assembly: `native` is the site's own sensor frame after the mean-field rotation, `diagonal` is section
6's arm diagonal. `hours` of a row is one of

| hours | what it selects |
|---|---|
| `whole` | the record |
| `f05`, `f10`, `f25` | workbook 03's hour selections (`process.selection`), scored on this frame's cache at this row's rate over 1-30 s, each carrying its `r25` control |
| `window:coherent` | the longest contiguous stretch in which both recorded lines read above `WINDOW_COH` with the magnetic field each couples to -- Ex with Hy, Ey with Hx -- over 20-200 s, per whole UTC hour |
| `window:<start UTC>/<hours>` | a stretch named in the cell, as an ISO UTC start and a count of hours |

The window is read on the two recorded lines -- Ex and Ey from the site's own 1 Hz cache as laid -- and never
on the frame's variant of them, whatever frame the rows are passed in: the window exists to find where both
electrodes were measuring, which is a fact about the electrodes and not about the frame their voltages are
later combined in. It is read at 1 Hz whatever a row's rate, because 200 s is measured on the long record.
Every window and every selection carries a control at the same cost: a stretch of the same length placed at
random elsewhere in the record under the seed, or the `r25` draw of the same number of hours.

A row at 10 Hz needs the 10 Hz cache of the site, the frame's variant of that cache, and the reference store
of BASELINE_KIND at 10 Hz, which is the remote's own 10 Hz record on this site's grid. Where one of the three
is not on disk the row is refused with the reason and no pass is run, by the rule of section 7: the single
station is not a kind of this package, so a missing store is not fallen back from.

The assembly. The x row supplies Zx'x' and Zx'y' and the y row Zy'x' and Zy'y'. The x row's product is the
base, so its period grid, its station block and its position carry through, and the y row is matched onto
that grid where the two grids are one and interpolated in log period onto it where they are not. A period the
y row's own grid does not reach carries the EDI empty-data value 1e32, which `products.read_tf` masks per
component, so the assembled file reads back with that row empty above the y row's longest period rather than
carrying a number nothing measured. A y row at 10 Hz reaches below the x row's shortest period, and those
periods are not carried either: one file holds one grid, and the grid is the x row's. The tipper is the named
row's. The header carries one `recipe=` line per row with the frame, the hours, the rate, the window and the
control's seed.

Where a row asks for what a section above has already built -- the whole record at 1 Hz in the native frame
is the baseline, and in the diagonal frame it is section 6's `diagonal` form -- that product is taken and no
second pass of the same specification is run.

What this composes at AusLAMP Queensland Phase 2's Q58N, whose two lines share a noisy centre electrode: the
x' row and the tipper from the arm diagonal over the whole record at 1 Hz, and the y' row from both recorded
lines over the 14 h the window rule finds, at 10 Hz against the same remote. That is the composition the
campaign reached by hand for this site, and check 1 below scores the rule written here against the window it
used, recorded in `survey.yaml`. At the shipped Q53N the recipe is frame native with both rows the whole
record at 1 Hz, which is the baseline itself and buys nothing: the section is there to be edited."""),

("code", '''# ---- the recipe: one composition of the sections above, one row at a time ----
RECIPE = dict(frame="native", x=dict(hours="whole", rate=1), y=dict(hours="whole", rate=1), tipper="x")
# frame     native | diagonal (section 6's arm diagonal); both rows are passed in it
# x, y      the x' and y' rows (x and y where the frame is native). hours is
#           whole | f05 | f10 | f25 | window:coherent | window:<start UTC>/<hours>; rate is 1 or 10
# tipper    which row's pass the tipper is taken from: x | y
WINDOW_COH = 0.5   # <- both recorded lines above this at 20-200 s, per whole UTC hour, is inside a window
'''),

("md", r"""The frame, the variant cache each rate needs, the window the rule finds, and what each row
resolves to. The five longest coherent stretches are printed beside the one chosen, so the margin between the
longest and the next is visible: a rule that picks a 14 h stretch over an 11 h one is a different claim from
one that picks 14 h over 3 h."""),

("code", '''FRAME = RC.frame_spec(sv, SITE, RECIPE["frame"])
RATES_USED = sorted({int(RECIPE[k].get("rate", 1)) for k in ("x", "y")})
print("frame        %s -- %s" % (FRAME["frame"], FRAME["note"]))
print("rows         x': %s at %g Hz; y': %s at %g Hz; the tipper from the %s row"
      % (RECIPE["x"]["hours"], RECIPE["x"]["rate"], RECIPE["y"]["hours"], RECIPE["y"]["rate"],
         RECIPE["tipper"]))
NE_CACHE = {}
for _r in RATES_USED:
    if FRAME["variant"] == "ne" and (FM.cache_dir(WORK, _r) / ("%s.npz" % SITE)).exists():
        _t = time.time()
        NE_CACHE[_r] = CE.ne_variant(sv, SITE, _r, force=REDO)
        print("   the %d Hz arm-diagonal cache in %.0f s: %s (%s; exact on %s finite sample(s): %s)"
              % (_r, time.time() - _t, NE_CACHE[_r].get("path"),
                 "rewritten at these arm lengths" if NE_CACHE[_r].get("refreshed") else
                 "written" if NE_CACHE[_r].get("written") else "already on disk and checked again",
                 NE_CACHE[_r].get("n_finite"), NE_CACHE[_r].get("exact")))
t = time.time()
COH_HOURS = RC.hour_coherence(sv, SITE, RC.WINDOW_RATE)
COHERENT = RC.coherent_window(COH_HOURS, WINDOW_COH)
print()
print("the window rule over %d whole UTC hour(s) of the %g Hz record in %.0f s"
      % (len(COH_HOURS), RC.WINDOW_RATE, time.time() - t))
print("   %s" % COHERENT["reason"])
for _h, _a, _b in COHERENT["runs"][:5]:
    print("   %4d h  %s .. %s UTC" % (_h, pd.Timestamp(_a, unit="s"), pd.Timestamp(_b, unit="s")))
'''),

("code", '''ROWS = {}
for key in ("x", "y"):
    spec = RECIPE[key]
    hours, rate = str(spec.get("hours", "whole")), int(spec.get("rate", 1))
    row = dict(row=key, hours=hours, rate=rate, seed=SEED + (0 if key == "x" else 1), window=None,
               control_window=None, keep=None, control_keep=None, control=None, selection=None,
               spans=[], control_spans=[], keep_name="the record", control_keep_name="", reuse="",
               product="", refused=RC.missing_inputs(sv, SITE, rate, FRAME["variant"], BASELINE_KIND))
    if not row["refused"] and hours.startswith("window"):
        w = COHERENT if hours == "window:coherent" else RC.named_window(hours)
        days = ((w["t_end"] - w["t_start"]) / 86400.0) if w.get("t_start") else 0.0
        if not w.get("t_start"):
            row["refused"] = "refused: %s" % w["reason"]
        elif days < FM.min_window_days(sv):
            row["refused"] = ("refused: the window is %.2f d, under the %g d floor in survey.yaml"
                              % (days, FM.min_window_days(sv)))
        else:
            c = RC.control_window(t0_rec, n_rec, w, row["seed"])
            row.update(window=w, control_window=c, control="recipe_%s_control" % key,
                       spans=[(w["t_start"], w["t_end"])], control_spans=[(c["t_start"], c["t_end"])],
                       keep_name="%s, %.2f h" % (hours, (w["t_end"] - w["t_start"]) / 3600.0),
                       control_keep_name="a stretch of the same length elsewhere in the record, seed %d"
                                         % row["seed"])
    elif not row["refused"] and hours in RC.SELECTION_KEYS:
        sel = RC.selection_hours(sv, SITE, hours, rate=rate, variant=FRAME["variant"], seed=SEED,
                                 elines=ELINES)
        row.update(selection=sel, keep=sel[hours]["keep"], control_keep=sel[RC.CONTROL_KEY]["keep"],
                   control="recipe_%s_control" % key,
                   spans=[tuple(x) for x in sel[hours]["hours"]],
                   control_spans=[tuple(x) for x in sel[RC.CONTROL_KEY]["hours"]],
                   keep_name="the best %.0f %% of %d scored hour(s) by %g-%g s coherence"
                             % (100 * sel[hours]["fraction"], sel["scored"], sel["band_s"][0],
                                sel["band_s"][1]),
                   control_keep_name="%d hour(s) drawn at random from the same pool, seed %d"
                                     % (sel[RC.CONTROL_KEY]["n_hours"], SEED))
    elif not row["refused"] and hours != "whole":
        row["refused"] = ("refused: hours=%s is none of whole, %s, window:coherent or window:<start>/<hours>"
                          % (hours, ", ".join(RC.SELECTION_KEYS)))
    if hours == "whole":
        row["spans"] = [(t0_rec, t0_rec + int(n_rec))]
        reuse = {"native": "whole", "diagonal": "diagonal"}[FRAME["frame"]]
        got = made_product(reuse) if rate == 1 else None
        if got:
            row.update(reuse=reuse, product=str(got["product"]))
    ROWS[key] = row
cols = ["row", "hours", "rate", "seed", "window_start", "window_end", "days", "control_start", "control",
        "reuse", "refused"]
print(pd.DataFrame([dict(
    row="%s (%s)" % (r["row"], RC.ROW_LABEL[r["row"]]), hours=r["hours"], rate=r["rate"], seed=r["seed"],
    window_start=(pd.Timestamp(r["window"]["t_start"], unit="s") if r["window"] else ""),
    window_end=(pd.Timestamp(r["window"]["t_end"], unit="s") if r["window"] else ""),
    days=round(sum(b - a for a, b in r["spans"]) / 86400.0, 3),
    control_start=(pd.Timestamp(r["control_window"]["t_start"], unit="s") if r["control_window"] else ""),
    control=(r["control"] or ""), reuse=(r["reuse"] or ""), refused=r["refused"][:70])
    for r in ROWS.values()])[cols].to_string(index=False))
for r in ROWS.values():
    print("   %s: %s" % (r["row"], r["keep_name"]))
'''),

("md", r"""The two hourly coherence series the window rule reads, with each row's stretch drawn above them
and its control beneath it. What to look for is the chosen stretch sitting where both series are above the
line at the same time -- one line alone above it is not a window, because the second row needs both
electrodes -- and the control landing somewhere the two series are neither obviously better nor worse: the
two cost the same, so the difference between their products is what the stretch bought."""),

("code", '''spans = []
for key in ("x", "y"):
    r = ROWS[key]
    spans.append(("%s row: %s" % (RC.ROW_LABEL[key], r["hours"]), FF.SPAN_COLOUR[RC.ROW_COMPONENT[key]],
                  None, r["spans"][:400]))
    if r["control_spans"]:
        spans.append(("%s control, seed %d" % (RC.ROW_LABEL[key], r["seed"]), "0.4", "//",
                      r["control_spans"][:400]))
fig = FF.recipe_spans(COH_HOURS, spans, SITE, OUT / "25_recipe_windows.png", coh_min=WINDOW_COH,
                      title="%s: the recipe's stretches over the coherence they were chosen on" % SITE,
                      caption="The squared coherence of each recorded electric line with the magnetic field "
                              "it couples to, over %g-%g s, one value per whole UTC hour on the %g Hz cache "
                              "as laid, with the %.2f line drawn. Above each panel: the stretch each row of "
                              "the recipe is estimated on, and beneath it that row's control at the same "
                              "cost. The window rule takes the longest contiguous run of hours with both "
                              "lines above the line; %d hour(s) of %d scored are above it here, in %d run(s)."
                              % (RC.WINDOW_BAND_S[0], RC.WINDOW_BAND_S[1], RC.WINDOW_RATE, WINDOW_COH,
                                 COHERENT["n_hours_above"], COHERENT["n_hours_scored"], COHERENT["n_runs"]))
WRITTEN.append(OUT / "25_recipe_windows.png")
display(Image(filename=str(OUT / "25_recipe_windows.png")))
'''),

("md", r"""One pass per row and one per control, each measured against the reference before it is run: the
mask, the reference's own coverage and the record cut into runs at the 3,600 s floor. A row whose cache
leaves nothing is refused there with those numbers rather than crashing the estimator, and a row whose inputs
are not on disk was refused above. The assembled product is then written from the two rows."""),

("code", '''for key in ("x", "y"):
    row = ROWS[key]
    name = "recipe_%s" % key
    crit = ("the %s row: %s at %g Hz in the %s frame"
            % (RC.ROW_LABEL[key], row["hours"], row["rate"], FRAME["frame"]))
    if row["refused"]:
        FORM_ROWS.append(FM.refused_row(sv, SITE, name, OUT, BASELINE_KIND, row["rate"], PARAMS,
                                        row["refused"],
                                        controls=([row["control"]] if row["control"] else []),
                                        criterion="not passed: " + crit))
        print("   %-20s is NOT passed -- %s" % (name, row["refused"]))
        continue
    if row["reuse"]:
        FORM_ROWS.append(dict(site=SITE, form=name, kind=BASELINE_KIND, rate_hz=float(row["rate"]),
                              params=PARAMS, product=row["product"], controls="", seed=None,
                              status="exists", days=None, n_runs=None, seconds=None, error="",
                              criterion=crit + "; the %s form's product, which is the same specification, "
                                               "and no second pass of it is run" % row["reuse"]))
        print("   %-20s takes the %s form's product: %s"
              % (name, row["reuse"], Path(row["product"]).name))
        continue
    jobs = [(name, row["window"], row["keep"], row["keep_name"], crit,
             [row["control"]] if row["control"] else [])]
    if row["control"]:
        jobs.append((row["control"], row["control_window"], row["control_keep"], row["control_keep_name"],
                     "the control of the %s row at the same cost" % RC.ROW_LABEL[key], []))
    for job_name, window, keep, keep_name, criterion, controls in jobs:
        extra = RC.window_mask(t0_rec, n_rec, window) if window else keep
        cov = FM.coverage(sv, SITE, kind=BASELINE_KIND, rate=row["rate"], variant=FRAME["variant"],
                          apply_e_signs=FRAME["apply_e_signs"], keep_extra=extra, keep_name=keep_name)
        print("   %-20s %6.2f d over %3d run(s) of the record's %.2f d, measured before the pass"
              % (job_name, cov["days"], cov["n_runs"], cov["record_days"]))
        if cov["empty"]:
            why = ("refused: %s leaves %d run(s) of %g s against the %s reference, of a record of %.2f d"
                   % (keep_name, cov["n_runs"], cov["min_segment_s"],
                      KIND_WORD.get(BASELINE_KIND, BASELINE_KIND), cov["record_days"]))
            FORM_ROWS.append(FM.refused_row(sv, SITE, job_name, OUT, BASELINE_KIND, row["rate"], PARAMS,
                                            why, cov=cov, controls=controls,
                                            criterion="not passed: the floor leaves no run to hand Aurora"))
            print("      NOT passed -- %s" % why)
            continue
        form(job_name, kind=BASELINE_KIND, rate=row["rate"], variant=FRAME["variant"],
             apply_e_signs=FRAME["apply_e_signs"], turn_ne=FRAME["turn_ne"],
             turn_angle_deg=FRAME["turn_angle_deg"],
             window=((window["t_start"], window["t_end"]) if window else None),
             keep_extra=(None if window else keep), keep_name=keep_name, seed=row["seed"],
             controls=controls, criterion=criterion,
             redo=bool(REDO or (NE_CACHE.get(row["rate"]) or {}).get("refreshed")),
             extra_lines=RC.recipe_lines(RECIPE, ROWS))
'''),

("code", '''ASSEMBLED, ROW_READINGS = None, []
for key in ("x", "y"):
    r = made_product("recipe_%s" % key)
    if r is None:
        continue
    comp = RC.ROW_COMPONENT[key]
    ctrl_name = ROWS[key]["control"]
    ctrl = made_product(ctrl_name) if ctrl_name else None
    bar_row = DL.bar(read(r["product"]), comp, *BAR_BAND)
    bar_ctrl = DL.bar(read(ctrl["product"]), comp, *BAR_BAND) if ctrl else np.nan
    # each row on its own element and never on the product's bar, which is the better of the two rows and
    # would answer for the row a window was not chosen for
    ROW_READINGS.append(dict(row=RC.ROW_LABEL[key], form="recipe_%s" % key, element="Z%s" % comp,
                             hours=ROWS[key]["hours"], rate_hz=ROWS[key]["rate"], days=r.get("days"),
                             bar=bar_row, control=(ctrl_name or "none"), control_bar=bar_ctrl,
                             promoted=(DL.beats(bar_row, bar_ctrl, BAR_MARGIN)
                                       if np.isfinite(bar_ctrl) else None)))
rx, ry = made_product("recipe_x"), made_product("recipe_y")
if rx and ry:
    for key, r in (("x", rx), ("y", ry)):
        ROWS[key]["product"] = str(r["product"])
    out = OUT / FM.product_name(SITE, "recipe", BASELINE_KIND, ROWS["x"]["rate"], PARAMS)
    ASSEMBLED = RC.assemble(rx["product"], ry["product"], out, tipper=RECIPE["tipper"],
                            lines=RC.recipe_lines(RECIPE, ROWS))
    scored = [d for d in ROW_READINGS if d["promoted"] is not None]
    me = CENTRE.get(SITE, {})
    model_ok = True if FRAME["frame"] == "native" else bool(me.get("model_holds"))
    why = []
    # a recipe whose rows are all the whole record carries no control at all, so it carries no evidence of
    # its own and is read like every other form with no control: the vacuous "all of nothing was beaten" is
    # not a promotion
    if not scored:
        why.append("no row carries a control, because every row is the whole record, so the assembled "
                   "product carries no evidence of its own")
    elif not all(d["promoted"] for d in scored):
        why.append("the %s row does not beat its control on the %g-%g s bar of %s by %.0f %%"
                   % (", ".join(d["row"] for d in scored if not d["promoted"]), BAR_BAND[0], BAR_BAND[1],
                      ", ".join(d["element"] for d in scored if not d["promoted"]), 100 * BAR_MARGIN))
    if not model_ok:
        why.append("the shared-centre model does not hold at %s (residual coherence %.2f, gain ratio "
                   "%.2f), and the diagonal frame rests on it"
                   % (SITE, me.get("resid_coh", np.nan), me.get("gain_ratio", np.nan)))
    bars = "; ".join("%s %.4f against %.4f" % (d["row"], d["bar"], d["control_bar"]) for d in scored)
    frame_held = ("the shared-centre model holds at %s, which the diagonal frame rests on" % SITE
                  if FRAME["frame"] == "diagonal" else "the frame is the site's own")
    verdict = ("NOT a candidate: %s" % "; ".join(why)) if why else \\
              ("a candidate: every row carrying a control beats it on the %g-%g s bar of its own element by "
               "at least %.0f %% (%s), and %s"
               % (BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN, bars, frame_held))
    FORM_ROWS.append(dict(site=SITE, form="recipe", kind=BASELINE_KIND,
                          rate_hz=float(ROWS["x"]["rate"]), params=PARAMS, product=str(out), controls="",
                          seed=None, status="made", days=None, n_runs=None, seconds=None, error="",
                          criterion="the %s frame: Zx'x' and Zx'y' from the x row, Zy'x' and Zy'y' from the "
                                    "y row, the tipper from the %s row"
                                    % (FRAME["frame"], RECIPE["tipper"]),
                          candidate_rule=dict(candidate=bool(not why), verdict=verdict)))
    WRITTEN.append(out)
    print("   the y row covers %d of %d period(s), %.4g to %.4g s; above that the row carries the %g fill"
          % (ASSEMBLED["n_y_periods"], ASSEMBLED["n_periods"], ASSEMBLED["y_min_s"], ASSEMBLED["y_max_s"],
             ASSEMBLED["fill"]))
    for ln in RC.recipe_lines(RECIPE, ROWS):
        print("   %s" % ln)
    print("   the assembled product is %s" % verdict)
else:
    print("no assembled product: the %s row was not made"
          % ", ".join(k for k in ("x", "y") if not made_product("recipe_%s" % k)))
'''),

("md", r"""The assembled product against the whole-record baseline and against each row's control. What to
look for is the x' panels lying on the curve the row they came from drew, the y' panels stopping at the
dash-dotted line -- the longest period that row reaches, above which the file carries the empty value -- and
each control sitting off the row it controls by more than its error bars, which is what says the stretch was
worth choosing."""),

("code", '''curves = [("whole, the baseline", read(BASE), "k", "-")] if BASE else []
if ASSEMBLED:
    curves.append(("recipe", read(ASSEMBLED["path"]), "C0", "-"))
for j, key in enumerate(("x", "y")):
    r = made_product("recipe_%s_control" % key)
    if r:
        curves.append(("%s control" % RC.ROW_LABEL[key], read(r["product"]), "C%d" % (j + 1), ":"))
if ASSEMBLED and len(curves) > 1:
    fig = FF.recipe_product(curves, SITE, OUT / "26_recipe_product.png",
                            join_s=ASSEMBLED["y_max_s"],
                            title="%s: the assembled recipe against the baseline and the controls" % SITE,
                            caption="The assembled product in the %s frame: the first row from the x row's "
                                    "pass (%s at %g Hz) and the second from the y row's (%s at %g Hz), with "
                                    "the tipper from the %s row, against the whole-record 1 Hz baseline in "
                                    "black and against each row's control at the same cost, dotted. %s. The "
                                    "dash-dotted line at %.4g s is the longest period the y row reaches; "
                                    "above it that row carries the EDI empty value and nothing is drawn. The "
                                    "two panels are two rows of one tensor and not two estimates of one "
                                    "quantity, so they are not expected to lie on each other, and where the "
                                    "recipe's frame is not the site's own the black curve is a different "
                                    "quantity again and is drawn for scale alone."
                                    % (FRAME["frame"], ROWS["x"]["hours"], ROWS["x"]["rate"],
                                       ROWS["y"]["hours"], ROWS["y"]["rate"], RECIPE["tipper"],
                                       ASSEMBLED["how"], ASSEMBLED["y_max_s"]))
    WRITTEN.append(OUT / "26_recipe_product.png")
    display(Image(filename=str(OUT / "26_recipe_product.png")))
else:
    print("no assembled product, so there is nothing to draw against the baseline")
'''),

("md", r"""**Where `survey.yaml` `checks.recipe_window` records a window for this site, this check fails if
the rule's window does not land within `tolerance_h` of it at each end.** The rule the package writes has to
reproduce the choice the campaign made by hand, and the only site where that comparison exists is Q58N of
AusLAMP Queensland Phase 2, whose recorded window is 2026-03-22 12:00 to 2026-03-23 02:00 UTC. Where no window
is on record the cell prints the one the rule found as a reading and writes no verdict."""),

("code", '''want_all = (sv.cfg.get("checks") or {}).get("recipe_window") or {}
tol_h = float(want_all.get("tolerance_h", 1))
want = want_all.get(SITE) or {}
found = (("the rule found %d h, %s to %s UTC"
          % (COHERENT["hours"], pd.Timestamp(COHERENT["t_start"], unit="s"),
             pd.Timestamp(COHERENT["t_end"], unit="s"))) if COHERENT["t_start"]
         else ("the rule found no stretch: %s" % COHERENT["reason"]))
if not want:
    # a reading and no verdict: survey.yaml records no window for this site, so the criterion has nothing
    # to be scored against and a verdict line here would be a judgement on nothing
    print("%s. survey.yaml checks.recipe_window records no window for %s, so this check is not scored here "
          "and the window above is a reading. The sites it is scored at are %s"
          % (found, SITE, ", ".join(sorted(k for k in want_all if k != "tolerance_h")) or "none"))
elif not COHERENT["t_start"]:
    print("VERDICT: FAIL -- %s, where the record carries %s to %s UTC (%s)"
          % (found, want["t_start"], want["t_end"], str(want.get("source"))[:110]))
else:
    a = int(pd.Timestamp(want["t_start"]).timestamp())
    b = int(pd.Timestamp(want["t_end"]).timestamp())
    da, db = abs(COHERENT["t_start"] - a) / 3600.0, abs(COHERENT["t_end"] - b) / 3600.0
    if max(da, db) > tol_h:
        print("VERDICT: FAIL -- %s, which misses the recorded %s to %s UTC by %.2f h at the start and "
              "%.2f h at the end, beyond the %g h the check allows (%s)"
              % (found, want["t_start"], want["t_end"], da, db, tol_h, str(want.get("source"))[:110]))
    else:
        print("VERDICT: PASS -- %s, within %.2f h and %.2f h of the recorded %s to %s UTC, both inside the "
              "%g h the check allows; the next longest stretch the rule found is %s (%s)"
              % (found, da, db, want["t_start"], want["t_end"], tol_h,
                 ("%d h" % COHERENT["runs"][1][0]) if len(COHERENT["runs"]) > 1 else "none",
                 str(want.get("source"))[:110]))
'''),

("md", r"""**This check fails if any row's pass ended in an exception rather than a stated refusal.** A row
refused before its pass, with its runs and its days measured first, is a reading and not a failure; a row
that ended in a traceback is a failure with the row named.

Each row against its own control is a reading and not a limb of the criterion, as in sections 3, 4 and 5: a
row that does not beat its control by BAR_MARGIN bought efficiency and not a different answer, and reads
`not promoted`. The comparison is per row and at the same cost -- the x' row against the x' row's control on
Zx'y', the y' row against the y' row's on Zy'x' -- because the assembled product's bar is the better of its
two rows and would answer for the row a window was not chosen for. Both bars are printed either way, and the
same readings decide whether the assembled product is a candidate in section 11's table."""),

("code", '''fail, note = [], []
for key in ("x", "y"):
    for name in ("recipe_%s" % key, ROWS[key]["control"]):
        if not name:
            continue
        r = rows_by_form().get(name, {})
        if r.get("status") == "FAILED":
            fail.append("%s ended in an exception rather than a stated refusal: %s"
                        % (name, str(r.get("error"))[:150]))
        elif r.get("status") == "refused":
            note.append("%s was refused before the pass: %s" % (name, str(r.get("reason"))[:110]))
        elif made_product(name) is None:
            note.append("%s carries no product on disk" % name)
print(pd.DataFrame(ROW_READINGS).round(4).to_string(index=False) if ROW_READINGS
      else "no recipe row was made")
for d in ROW_READINGS:
    if d["promoted"] is None:
        print("   %s reads %.4f on the %g-%g s bar of %s and carries no control, so it is read and not "
              "promoted" % (d["row"], d["bar"], BAR_BAND[0], BAR_BAND[1], d["element"]))
    else:
        print("   %s reads %.4f on the %g-%g s bar of %s against its control's %.4f -- %s"
              % (d["row"], d["bar"], BAR_BAND[0], BAR_BAND[1], d["element"], d["control_bar"],
                 "promoted" if d["promoted"] else
                 "not promoted: efficiency, not a different answer"))
if ASSEMBLED and BASE and all(ROWS[k]["reuse"] == "whole" for k in ("x", "y")):
    # both rows of this recipe ARE the baseline, so the assembly has to give the baseline back, element for
    # element. Where a row is anything else the two files are different quantities and the ratio says nothing
    same = RC.row_comparison(read(BASE), read(ASSEMBLED["path"]), 1.0, 100000.0)
    print()
    print("both rows of this recipe are the whole-record baseline, so the assembly must return it: the "
          "assembled product against the baseline, every element over the whole band")
    print(same.round(4).to_string(index=False))
print()
scored = [d for d in ROW_READINGS if d["promoted"] is not None]
readings = ("; as readings, %s"
            % "; ".join("%s %.4f against its control's %.4f, %s"
                        % (d["row"], d["bar"], d["control_bar"],
                           "promoted" if d["promoted"] else "not promoted")
                        for d in scored)) if scored else \\
           "; no row carries a control, because every row is the whole record"
if fail:
    print("VERDICT: FAIL -- %s%s%s" % ("; ".join(fail), ("; " + "; ".join(note)) if note else "", readings))
elif not ROW_READINGS:
    print("VERDICT: UNJUDGED -- no recipe row was passed, so no pass could be scored%s"
          % (": " + "; ".join(note) if note else ""))
else:
    print("VERDICT: PASS -- %d row(s) were passed and none ended in an exception (%s)%s%s"
          % (len(ROW_READINGS),
             "; ".join("%s %s at %g Hz" % (d["row"], d["hours"], d["rate_hz"]) for d in ROW_READINGS),
             ("; " + "; ".join(note)) if note else "", readings))
'''),

("md", r"""The comparison, last and never truth. The campaign's own product of this composition, where
`survey.yaml` `checks.recipe_comparison` names one for this site, read row by row in the same frame over the
band it names. It is a shape check on two paths to the same product and carries no verdict: the two were
estimated from the same raw record and agree or differ for reasons neither file can settle."""),

("code", '''cmp_row = ((sv.cfg.get("checks") or {}).get("recipe_comparison") or {}).get(SITE) or {}
cmp_path = Path(str(cmp_row.get("path", "")))
if not cmp_row:
    print("survey.yaml checks.recipe_comparison names no product for %s: there is nothing to compare "
          "against, which is the case at every site but the one the campaign salvaged by hand" % SITE)
elif not ASSEMBLED:
    print("no assembled product, so the comparison with %s is not made" % cmp_path.name)
elif not cmp_path.exists():
    print("%s is named in survey.yaml but is not on disk, so the comparison is not made" % cmp_path)
else:
    lo, hi = tuple(cmp_row.get("band_s", (5, 100)))
    tab = RC.row_comparison(read(ASSEMBLED["path"]), read(str(cmp_path)), lo, hi)
    print("%s against %s, %g-%g s, the campaign's product over ours" % (Path(ASSEMBLED["path"]).name,
                                                                        cmp_path.name, lo, hi))
    print(tab.round(4).to_string(index=False))
    print()
    print("the frame of the comparison is %s and ours is %s; %s"
          % (cmp_row.get("frame"), FRAME["frame"], str(cmp_row.get("note"))[:400]))
'''),

("md", r"""## 11. The forms table

Every form with the products it is read against, the criterion in words, the verdict and the reading. This is
the file workbook 06 reads. A form is a candidate only where it beats every control it carries on the
10-1000 s bar by BAR_MARGIN. A form with no control is read and never promoted on this table alone, and a form
that borrows both horizontal channels is never a candidate. Section 10's assembled `recipe` row is read the
same way through the rows it was assembled from, because its controls belong to them: it is a candidate where
at least one of those rows carries a control, where every row that carries one beat it by BAR_MARGIN on its
own element, and, in the diagonal frame, where section 6's shared-centre model holds at this site; its
`verdict` cell states which of the three decided it. A recipe whose rows are all the whole record carries no
control at all and is read and not promoted, like any other form with none.

The decisions.csv cells this workbook proposes are printed below and are not written unless WRITE_DECISIONS
is True. Decisions are the analyst's."""),

("code", '''TABLE = DL.forms_table(FORM_ROWS, OUT / "forms.csv", baseline_path=BASE, bar_band=tuple(BAR_BAND),
                       agree_band=tuple(AGREE_BAND), margin=BAR_MARGIN)
WRITTEN.append(OUT / "forms.csv")
cols = ["form", "kind", "rate_hz", "status", "controls", "bar_10_1000", "control_bar", "rho_ratio_xy",
        "phase_diff_xy", "rho_ratio_yx", "phase_diff_yx", "verdict", "candidate"]
print(TABLE[cols].round(4).to_string(index=False))
print()
print("%d form(s), %d made, %d candidate(s) for workbook 06: %s"
      % (len(TABLE), int((TABLE.status == "made").sum()), int(TABLE.candidate.sum()),
         " ".join(TABLE[TABLE.candidate].form) or "none"))
FM.write_run_provenance(sv, SITE, OUT, RUN, OUT.name.split("_", 1)[-1], 1, PARAMS, BASELINE_KIND, FORM_ROWS)
print("provenance.json rewritten with %d form entries" % len(FORM_ROWS))
'''),

("md", r"""The same table as a figure: one bar per form on the 10-1000 s bar, its control marked as a red
diamond beside it, a candidate in green. What to look for is a green bar well to the left of its diamond --
that gap is the 20 per cent margin -- and how many forms sit on top of the baseline, which is the reading
that the record was already being used for what it is worth."""),

("code", '''fig = FF.forms_bars(TABLE, SITE, OUT / "24_forms_bars.png", band=tuple(BAR_BAND), margin=BAR_MARGIN)
WRITTEN.append(OUT / "24_forms_bars.png")
display(Image(filename=str(OUT / "24_forms_bars.png")))
'''),

("code", '''evidence = []
for c in COMPONENTS:
    bars = ";".join("%s %.4f" % (r.form, r.bar_10_1000) for r in TABLE.itertuples()
                    if str(r.form).startswith("window_%s" % c) and np.isfinite(r.bar_10_1000))
    evidence.append(dict(component=c, fraction=HOURS_FRACTION, record_days=round(n_rec / 86400.0, 2),
                         in_use=False,
                         evidence="control: an equal-length random block, seed %d; %g-%g s bar %s"
                                  % (WINDOWS[c]["seed"], BAR_BAND[0], BAR_BAND[1], bars or "not scored")))
WIN_TABLE = MK.windows_table(SITE, OUT.name, WINDOWS, evidence)
WIN_TABLE.to_csv(SITE_DIR / "windows.csv", index=False)
WRITTEN.append(SITE_DIR / "windows.csv")
print(WIN_TABLE.to_string(index=False))
print()
proposed = MK.write_windows(SV.blank_decisions([SITE]), SITE, WINDOWS)
flags = []
me = CENTRE.get(SITE, {})
if me.get("model_holds"):
    flags.append("shared centre: residual coherence %.2f, gain %.2f, s %+d; the remedy %s"
                 % (me["resid_coh"], me["resid_gain"], me["s_obs"],
                    "applies" if me["remedy_applicable"] else
                    "does NOT apply (the clean diagonal by H is the %s and the observed one is the %s)"
                    % (me["clean_by_H"], me["clean_obs"])))
if NEEDED:
    flags.append("magnetic replacement: %s from %s" % (CHAN, LEND))
proposed.loc[proposed.site == SITE, "flags"] = "; ".join(flags) or "decide"
print("the decisions.csv cells this workbook proposes for %s" % SITE)
print(proposed[["site", "windows", "flags"]].to_string(index=False))
if WRITE_DECISIONS:
    path = SV.SURVEYS / SURVEY / "decisions.csv"
    cur = sv.decisions.copy()
    for col in ("windows", "flags"):
        cur.loc[cur.site == SITE, col] = proposed[col].iloc[0]
    print(SV.write_table(path, cur, SV.DECISIONS_COLUMNS))
else:
    print("NOT written: WRITE_DECISIONS is False, and decisions are the analyst's")
'''),

("md", r"""## 12. What was written"""),

("code", '''rows = []
for p in list(WRITTEN) + sorted(OUT.glob("*")):
    p = Path(p)
    if p.exists() and p.is_file():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d file(s), %.1f MB, in %.1f minutes" % (len(files), files.kb.sum() / 1024, (time.time() - T0) / 60))
print(files.to_string(index=False))
print()
cost = pd.DataFrame([dict(form=r.get("form"), rate_hz=r.get("rate_hz"), status=r.get("status"),
                          days=r.get("days"), runs=r.get("n_runs"), seconds=r.get("seconds"),
                          error=str(r.get("error") or "")[:60]) for r in FORM_ROWS])
print("the forms and what each cost")
print(cost.to_string(index=False))
'''),
]


# ===================================================================== 06 the final transfer function

WB06_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/: queensland_phase2 | queensland_phase3 | victoria
SITE = "Q53N"                 # one site; a survey is delivered site by site by changing this and re-running
RUNS = "all"                  # "all" = every run folder of the site | "latest" | ["first", "short10"]
RECORD_RATES = [1]            # the delivery rate the product of record is chosen among; [1, 10] admits a
                              # 10 Hz product as a whole row, which stops near 1,200 s at this survey
TRIM_TO_HELD = True           # the delivered file carries the periods inside the chosen product's held
                              # band; False delivers every period the sources carry and marks the file
HALVES = True                 # True runs the two half passes over each chosen product; False reads the
                              # ones on disk and leaves the reading UNJUDGED where there are none
LANES = 2                     # concurrent single-site subprocesses for the half passes
TIPPER_FROM = "xy"            # "xy" | "yx" | a reference kind: which product the tipper is taken from
RESAMPLE = False              # True writes final/<site>_resampled.edi on the ten-per-decade grid as well
WORK_ROOT = None              # None = survey.yaml work_root; every file this workbook writes lands under it
'''

WB06_RULES = '''# ---- the three response tests: a change here changes which products are sound ----
QUALITY_BAND = (10, 1000)     # the band the tests, the bar and the choice are read over, in s
QUADRANT_MIN = 0.70           # the phase test: raise it and a product with a few noisy periods fails
SLOPE_TOL = 0.25              # the slope test: the tolerance on |d log rho / d log T| <= 1, for noise
SLOPE_MIN = 0.80              # ... lower it and a rougher curve passes
BAR_MAX = 1.0                 # the error test: raise it and a product whose error bars are the size of
                              # its own impedance is delivered
MIN_PERIODS = 8               # ... the periods of the band that must carry an error under that ceiling
HELD_BAR_MAX = 0.20           # a period is held where its bar is under this and its phase is in quadrant;
                              # raise it and the delivered file reaches further with worse bars

# ---- the choice: a change here changes which product is the record ----
AGREE_RHO = 0.20              # two products agree within this fraction in apparent resistivity
AGREE_PHASE = 5.0             # ... and this many degrees in phase
AGREE_BAND = (5, 200)         # ... over this band, in s; a product that agrees with no other kind is
                              # never the product of record

# ---- the splice: a change here changes which 10 Hz rows are delivered ----
SPLICE_JOIN_S = 16            # the period the 10 Hz row joins the 1 Hz row at, in s
SPLICE_MAX_STEP_PCT = 2.0     # the step at the join, in per cent, on the worse of the two ruled bands
SPLICE_GUARD = (18, 36)       # measured and printed per row and scored by nothing, in s
SHORT_FLOOR_S = 0.6           # nothing below this is delivered, in s
CONTROL_BAR_BAND = (2, 16)    # the band a 10 Hz selection is read against its random control over, in s
CONTROL_MARGIN = 0.20         # ... and the fraction of the control bar a selection must beat it by

PERIOD_RANGE = (0.3, 50000)   # the periods drawn and scored, in s
BANDS = [(5, 10), (10, 100), (100, 1000), (1000, 10000)]   # the decades every table reports, in s
'''

WB06_SETUP = '''import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

# mt_metadata logs a warning per channel while an EDI is read, and this workbook reads several hundred
from loguru import logger as _loguru
_loguru.remove()

import auslamp_proc
from auslamp_proc import final as FN, halves as HV, products as PR
from auslamp_proc import readings as RD, splice as SP, survey as SV
from auslamp_proc.process import KIND_WORD
from auslamp_proc.site import deliver as DL
from auslamp_proc.figures import final as FIG

pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 700)

T0 = time.time()
REPO = Path(auslamp_proc.__file__).resolve().parent.parent
sv = SV.load_survey(SURVEY)
if WORK_ROOT:
    sv.cfg["work_root"] = WORK_ROOT
WORK = Path(sv.cfg["work_root"])
OUT = WORK / "survey"
OUT.mkdir(parents=True, exist_ok=True)
WRITTEN = []

EVERY, WHY = SV.select_sites(sv, "all", 0)
FOCUS = str(SITE).strip()
if FOCUS not in set(EVERY):
    raise ValueError("%s is not a site of %s (%s)" % (FOCUS, SURVEY, " ".join(EVERY)))
ASKED = [FOCUS]
PROD, IGNORED = RD.deliverable(RD.all_products(sv, ASKED, runs=RUNS))
CHOSEN = [s for s in ASKED if s in set(PROD.site)]
NO_PRODUCT = [s for s in ASKED if s not in set(PROD.site)]
CHOICES = FN.read_choices(FN.choices_path(sv))

_tfs = {}
def read(path):
    """One TFData per file: the readings, the agreement, the splice and four figures ask for the same file."""
    path = str(path)
    if path not in _tfs:
        _tfs[path] = PR.read_tf(path)
    return _tfs[path]

def final_dir(site):
    return WORK / str(site) / "final"

def picks_for(site, record):
    """The record rows of one site as the merge wants them, or an empty dict."""
    out = {}
    for r in record[(record.site == site) & (record["product"] != "none")].itertuples():
        out[r.component] = dict(path=r.path, kind=r.kind, kind_word=r.kind_word, form=r.form,
                                selection=r.selection, run=r.run, stamp=r.stamp, rate_hz=r.rate_hz,
                                bar=r.bar, held_lo_s=r.held_lo_s, held_hi_s=r.held_hi_s,
                                product=r.product)
    return out

print("survey       %s" % sv.cfg["name"])
print("work root    %s" % WORK)
print("site         %s -- %s" % (FOCUS, sv.site(FOCUS).notes or "no note in sites.csv"))
if NO_PRODUCT:
    print("no product   %s -- run workbook 03 over it" % " ".join(NO_PRODUCT))
print("products     %d rows, %d on disk, over %d run folder(s)"
      % (len(PROD), int(PROD.on_disk.sum()), PROD.groupby(["run", "stamp"]).ngroups))
for (run, stamp), g in PROD.groupby(["run", "stamp"]):
    print("run          %-9s %-14s %3d product(s) at %s Hz over %2d site(s); selections %s"
          % (run, stamp, len(g), ", ".join("%g" % r for r in sorted(set(g.rate_hz))), g.site.nunique(),
             ", ".join(sorted(set(g.selection)))))
print("forms        %d product(s) workbook 05 left" % int((PROD.form != "").sum()))
print("kinds        %s" % ", ".join("%s (%s)" % (KIND_WORD[k], k) for k in RD.KINDS))
print("ignored      %d %s product(s) on disk: read by nothing in this workbook"
      % (len(IGNORED), RD.DROPPED_KIND))
print("delivery     the product of record is chosen among the %s Hz products; the delivered file is %s"
      % (", ".join(str(r) for r in RECORD_RATES),
         "trimmed to the chosen product's held band" if TRIM_TO_HELD else "not trimmed"))
print("choices      %s: %d row(s), %d of them an analyst's"
      % (FN.choices_path(sv), len(CHOICES), int((CHOICES.chosen_by == "analyst").sum())))
print("engine       auslamp_proc %s" % auslamp_proc.__version__)
'''

WB06 = [
("md", r"""# 06 -- one site's final transfer function

A student goes through the sites one by one and sees what every processing produced, what worked and what
did not, before the final curve is merged.

This workbook reads every product one site has -- workbook 03's reference kinds at both rates and every
selection of hours a 10 Hz pass ran on, and workbook 05's forms -- puts the same three tests to all of them,
proposes one product per component, records the choice that was made, and writes one EDI.

| word | what it is | code key |
|---|---|---|
| remote site | one other site's H as the reference | `remote` |
| fleet stack | a coherence-weighted mean of several sites' H | `stack` |
| observatory | an INTERMAGNET one-second record as the reference | `obs` |
| stack + observatory | the stack with the observatory as a member | `stack_obs` |

The single station is not a kind of this package: it is biased low by whatever noise sits in H, and its error
bars do not show that bias. A single-station file an earlier pass left in a run folder is counted, named and
read by nothing here.

The three tests a product is put to are the ones students learn first -- the phase in its quadrant, the
apparent resistivity changing no faster than the period, and the error bar -- and a product passes them or
fails one by name.

A delivered file carries the measurement: it holds the periods inside the chosen product's held band, and the
periods dropped are named in the manifest with the reason.

One site is delivered per run. A survey is delivered site by site, by changing `SITE` and running the
workbook again; each run appends that site's rows to the survey's four tables and leaves every other site's
untouched.

Four checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported as FAIL and is
not revised afterwards."""),

("code", WB06_PARAMS),
("code", WB06_RULES),

("md", r"""## The products this workbook reads

Everything below reads `<work_root>/survey/runs.csv` and the run folders `<site>/<run>_<stamp>/`. `RUNS`
`"all"` keeps every run folder of the site, which is what this workbook wants: the 1 Hz kinds, the 10 Hz
passes and workbook 05's forms are all products of the same record and are all put to the same three tests.

A product carries three labels beside its site: the reference kind, the selection of hours it was estimated
on, and the form where workbook 05 made it. The 1 Hz pass runs on the whole record and its files carry no
selection tag; the 10 Hz pass runs on the most coherent hours by 1-30 s E-H coherence and never on the whole
record, so its files carry one: `f05`, `f10` and `f25` are the best 5, 10 and 25 per cent of hours and `r25`
is the random 25 per cent that controls them. A whole-record 10 Hz product, where a run folder holds one, is
a control and is read as `whole`. The name each product answers to in the choice cell is `<kind>_<rate>hz`,
with the selection appended where a 10 Hz pass ran on one, and a form's own name where workbook 05 made it.

`SITE` names the one site this run delivers."""),

("code", WB06_SETUP),

("md", r"""## 1. Everything this site produced

Every product of this site -- kind by rate by selection by form -- on the four panels, coloured by reference
kind, 10 Hz dashed and a workbook 05 form dotted. A product that fails a test is drawn grey and carries the
failed test's name at its curve. The readings table of the site is printed beside it.

The three tests, each read over QUALITY_BAND:

| test | what it measures | the bound |
|---|---|---|
| phase | the fraction of the band whose phase sits in (0, 90) deg, the yx phase folded by +180 deg | at least QUADRANT_MIN |
| slope | the fraction of adjacent periods whose log-log slope of apparent resistivity lies inside the bound | at least SLOPE_MIN |
| error | the median relative error of the impedance magnitude, and the periods carrying an error under that ceiling | at most BAR_MAX, and at least MIN_PERIODS |

A product that fails the phase test carries a sign fault on an E or an H line, and the table says so. The
slope bound is one decade of apparent resistivity a decade of period, which is the most a one-dimensional
earth can produce (Weidelt 1972; Parker and Booker 1996), widened by SLOPE_TOL for noise; a two- or
three-dimensional response can exceed it, so the test is a screen and not a law. The error test sits beside
the two shape tests because a shape alone passes on noise.

The held band is not a test. It is the periods whose bar is under HELD_BAR_MAX with the phase in quadrant,
and it is the band the delivered file is trimmed to under TRIM_TO_HELD.

What to look for: the kinds lying on each other at 100-1000 s, where the field is large and every reference
sees the same source; the grey curves, which are where a line, a sign or an error bar failed; and the held
band in the table, which is how far the delivered file reaches.

**This check fails if any product in this site's run folders is missing from the readings table, or if any
row's three test results cannot be recomputed from its own EDI.** The recomputation opens each file again and
works out the two fractions, the bar and the period count directly from the periods and the tensor, so each
verdict is scored against the file and not against the table that carries it."""),

("code", '''t_read = time.time()
READINGS = RD.readings_table(PROD, read=read, band=tuple(QUALITY_BAND), agree_band=tuple(AGREE_BAND),
                             agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE, quadrant_min=QUADRANT_MIN,
                             slope_tol=SLOPE_TOL, slope_min=SLOPE_MIN, bar_max=BAR_MAX,
                             min_periods=MIN_PERIODS, held_bar_max=HELD_BAR_MAX)
print("%d reading(s) over %d product(s) of %d site(s) in %.1f s"
      % (len(READINGS), len(PROD), len(CHOSEN), time.time() - t_read))
print()
SHOW = READINGS[READINGS.site == FOCUS].copy()
SHOW["held_band_s"] = ["%.4g-%.4g" % (a, b) if np.isfinite(a) and np.isfinite(b) else "none"
                       for a, b in zip(SHOW.held_lo_s, SHOW.held_hi_s)]
cols = ["component", "product", "rate_hz", "selection", "form", "phase_frac", "slope_frac", "bar",
        "n_periods", "held_band_s", "agree_kinds", "passes", "fails"]
print("every product of %s, and what the three tests say of each" % FOCUS)
print(SHOW[cols].round(3).to_string(index=False))
print()
print("how many products pass, per kind and selection, over %d site(s)" % len(CHOSEN))
print(READINGS[READINGS.status == "ok"].pivot_table(index=["kind", "selection"], columns="component",
                                                    values="passes", aggfunc="sum").to_string())
print()
print("which test a product failed, the leading one of each")
print(READINGS[(READINGS.status == "ok") & (~READINGS.passes.astype(bool))]
      .fails.str.split(" ").str[0].value_counts().to_string())

# the three tests worked out again from each EDI, not read back from the table above
in_table = {str(p).lower() for p in READINGS.path}
set_ignored = {str(p).lower() for p in IGNORED.path}
walked, missing, bad = 0, [], []
for folder in sorted({Path(p).parent for p in PROD.path}):
    for p in sorted(Path(folder).glob("*.edi")):
        walked += 1
        if str(p).lower() not in in_table and str(p).lower() not in set_ignored:
            missing.append(str(p))
for r in READINGS[READINGS.status == "ok"].itertuples():
    tf = read(r.path)
    i, j = PR.COMPONENTS[r.component]
    per = np.asarray(tf.period, float)
    zz, ee = tf.z[:, i, j], np.asarray(tf.z_err, float)[:, i, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.isfinite(zz) & (np.abs(zz) > 0) & np.isfinite(ee), ee / np.abs(zz), np.nan)
    ph = np.degrees(np.angle(zz)) + (180.0 if r.component == "yx" else 0.0)
    ph = np.where(ph > 180, ph - 360, ph)
    rho = 0.2 * per * np.abs(zz) ** 2
    band = (per >= QUALITY_BAND[0]) & (per <= QUALITY_BAND[1])
    mp = band & np.isfinite(ph)
    qf = float(np.mean((ph[mp] > 0) & (ph[mp] < 90))) if mp.any() else np.nan
    ms = band & np.isfinite(rho) & (rho > 0)
    sl = (np.diff(np.log10(rho[ms])) / np.diff(np.log10(per[ms]))) if ms.sum() > 1 else np.zeros(0)
    sf = float(np.mean(np.abs(sl) <= 1.0 + SLOPE_TOL)) if len(sl) else np.nan
    me = band & np.isfinite(rel)
    bb = float(np.median(rel[me])) if me.any() else np.nan
    nn = int((rel[me] < BAR_MAX).sum())
    got = (bool(qf >= QUADRANT_MIN), bool(sf >= SLOPE_MIN), bool(bb <= BAR_MAX and nn >= MIN_PERIODS))
    if got != (bool(r.pass_phase), bool(r.pass_slope), bool(r.pass_error)):
        bad.append("%s %s %s: the table says %s, the file says %s"
                   % (r.site, r.component, r.product,
                      (bool(r.pass_phase), bool(r.pass_slope), bool(r.pass_error)), got))

drawn = []
for p, g in READINGS[(READINGS.site == FOCUS) & (READINGS.status == "ok")].groupby("path", sort=False):
    r0 = g.iloc[0]
    names = sorted({w.split(" ")[0] for s in g.fails for w in str(s).split("; ") if w})
    drawn.append(dict(tf=read(p), label=r0["product"], kind=r0.kind, rate_hz=r0.rate_hz, form=r0.form,
                      passes=bool(g.passes.all()),
                      fails=("fails %s" % ", ".join(names)) if names else ""))
fig_prod = FIG.products_page(
    FOCUS, drawn, final_dir(FOCUS) / ("%s_products.png" % FOCUS), period_range=tuple(PERIOD_RANGE),
    title="%s: every product of this site" % FOCUS,
    caption="Every product %s carries, the reference kind as the colour, 10 Hz dashed and a workbook 05 "
            "form dotted, over %g-%g s. A product that fails one of the three response tests over %g-%g s "
            "is drawn grey, carries the failed test at its long end and does not set the y limits. The "
            "tests are the phase in (0, 90) deg at %.0f per cent of the band, the log-log slope of rho "
            "inside +-%.2f at %.0f per cent of the adjacent pairs, and the median error over the impedance "
            "magnitude at or under %.2f with at least %d periods under it."
            % (FOCUS, PERIOD_RANGE[0], PERIOD_RANGE[1], QUALITY_BAND[0], QUALITY_BAND[1],
               100 * QUADRANT_MIN, 1.0 + SLOPE_TOL, 100 * SLOPE_MIN, BAR_MAX, MIN_PERIODS))
WRITTEN.append(fig_prod)
display(Image(filename=str(fig_prod)))

n_scored = int((READINGS.status == "ok").sum())
if not n_scored:
    print("VERDICT: UNJUDGED -- no product of the chosen site(s) could be read, so no row was scored")
elif missing or bad:
    print("VERDICT: FAIL -- %d of the %d EDI(s) in the run folders are missing from the readings table "
          "(%s); %d row(s) carry a verdict the file itself does not reproduce (%s)"
          % (len(missing), walked, "; ".join(Path(m).name for m in missing[:4]) or "none", len(bad),
             "; ".join(bad[:3]) or "none"))
else:
    print("VERDICT: PASS -- all %d EDI(s) in the run folders are in the readings table or are among the %d "
          "%s product(s) named and ignored, and every one of the %d scored rows carries the three verdicts "
          "the file itself gives when the phase fraction, the slope fraction, the bar and the period count "
          "are worked out again" % (walked, len(IGNORED), RD.DROPPED_KIND, n_scored))
'''),

("md", r"""## 2. The rule's proposal

Per component, the product of record as the rule chooses it: among the products at the delivery rate that
pass the three tests and agree with at least one product of another reference kind over AGREE_BAND, the one
with the smallest bar over QUALITY_BAND, ties broken by the longest period held.

Corroboration comes from another kind because two references that share no magnetics cannot carry the same
noise into the estimate. Two products agree where the median departure in apparent resistivity is within
AGREE_RHO and the median departure in phase is within AGREE_PHASE; the departure is the median of the
absolute differences and not the difference of the medians, so a curve that wobbles about another by more
than the tolerance is not agreement.

The choice is made at the delivery rate, `RECORD_RATES`, which is 1 Hz. A 10 Hz product of this survey stops
near 1,200 s and the row a long-period survey delivers has to cover the delivery band; the 10 Hz short end
enters the file through the join below, at 16 s, and not as the whole row.

Which products may be delivered is workbook 05's call. A workbook 03 product may always be delivered; a form
may only where its forms.csv row marks it a candidate, which it does where the form beats every control it
carries on the 10-1000 s bar by 20 per cent and is not an inter-site impedance, and for the assembled recipe
where every row it was built from that carries a control beat it and its frame holds. A form that is not a
candidate is read, scored and reported in section 1, is never the product of record, and does not corroborate
another row.

A component with no proposal is a result. A component whose sound products do not corroborate each other is
one where the references disagree; a component with nothing that passes the three tests is one whose record
carries no transfer function of that orientation.

The figure draws the proposed curves with their error bars over the products they were chosen against, each
rejected product named in grey. What to look for: a rejected curve lying on the chosen one says the choice
was between equals and the bar decided it; one parting from it at the long end says the references disagree
where the field is small.

This is the proposal, and section 3 is where it is accepted or replaced."""),

("code", '''RECORD = RD.product_of_record(READINGS, agree_band=tuple(AGREE_BAND), band=tuple(QUALITY_BAND),
                              rates=[float(r) for r in RECORD_RATES])
show = ["site", "component", "product", "kind_word", "bar", "n_periods", "held_lo_s", "held_hi_s",
        "agree_n", "agree_kinds"]
print("the rule's proposal, per site and component")
print(RECORD[show].round(5).to_string(index=False))
print()
NONE = RECORD[RECORD["product"] == "none"]
print("%d of %d site-components have no proposal" % (len(NONE), len(RECORD)))
if len(NONE):
    print(NONE[["site", "component", "why"]].to_string(index=False))
print()
print("what the proposal at %s was chosen among" % FOCUS)
for r in RECORD[(RECORD.site == FOCUS) & (RECORD["product"] != "none")].itertuples():
    print("  %s  %-22s  <- %s" % (r.component, r.product, r.why))
    print("     the sound products and their bars: %s" % r.alternatives)
print()
AGREEMENT = pd.concat([RD.agreement_matrix(PROD[PROD.site == FOCUS], read=read, comp=c,
                                           agree_band=tuple(AGREE_BAND), agree_rho=AGREE_RHO,
                                           agree_phase=AGREE_PHASE)
                       for c in RD.COMPONENTS], ignore_index=True)
ex = AGREEMENT[(AGREEMENT.component == "xy") & (~AGREEMENT.same_kind)] if len(AGREEMENT) else AGREEMENT
if len(ex):
    print("%s, component xy: the median departure in rho between products of different kinds" % FOCUS)
    print(ex.pivot_table(index="a", columns="b", values="rho_dev").round(3).to_string())

USED = {str(r.path) for r in RECORD[(RECORD.site == FOCUS) & (RECORD["product"] != "none")].itertuples()}
REJECTED = {}
for r in READINGS[(READINGS.site == FOCUS) & READINGS.passes.astype(bool)].itertuples():
    if str(r.path) not in USED:
        REJECTED[r.product] = read(r.path)
PROPOSED = [("%s: %s" % (r.component, r.product), r.kind, read(r.path))
            for r in RECORD[(RECORD.site == FOCUS) & (RECORD["product"] != "none")].itertuples()]
if PROPOSED:
    fig_prop = FIG.over_rejected(
        FOCUS, PROPOSED, sorted(REJECTED.items()),
        final_dir(FOCUS) / ("%s_proposal.png" % FOCUS), period_range=tuple(PERIOD_RANGE),
        title="%s: the rule's proposal over the products it was chosen against" % FOCUS,
        caption="The proposed product of each component with its error bars, over the %d product(s) of %s "
                "that pass the three response tests and were not chosen, drawn grey and named. The rule "
                "takes the smallest median error over the impedance magnitude across %g-%g s among the "
                "products that pass all three tests and agree with another reference kind within %.0f per "
                "cent in rho and %.1f deg in phase over %g-%g s, ties broken by the longest period held."
                % (len(REJECTED), FOCUS, QUALITY_BAND[0], QUALITY_BAND[1], 100 * AGREE_RHO, AGREE_PHASE,
                   AGREE_BAND[0], AGREE_BAND[1]))
    WRITTEN.append(fig_prop)
    display(Image(filename=str(fig_prop)))
'''),

("md", r"""### Reproducibility on halves

Each proposed product is re-estimated twice, once on each half of its own record, and the two are put through
the agreement rule over QUALITY_BAND. A product whose two halves disagree was estimated on something that
changed inside the record.

The record is not cut. Each half is a keep mask handed to the estimator on top of the transient mask, so the
mask is applied inside the pass and one Aurora run is written per kept stretch, exactly as the whole-record
product was. The split is at the midpoint sample index, so the two halves are the same length whatever the
gaps hold; the days each half keeps after the transient mask are reported beside the verdict, because a
record whose second half is mostly masked reproduces on a shorter record than its first.

The cost is two passes per proposed product, and only the proposed products are run. `HALVES` False reads
the passes already on disk and leaves the reading UNJUDGED where there are none.

A product that does not reproduce is flagged in the delivery record and is not removed: the rule chose it on
its soundness and its corroboration, and the reason it did not reproduce is written beside it for a reader
to weigh.

**This check fails if the halves are UNJUDGED for every proposed product -- no half pass ran and none was
found on disk.** A reading over no items cannot support the column it fills.

The departures are two numbers per product and are printed beside the proposal, not drawn: the proposal
table carries the median absolute departure in apparent resistivity and in phase between the two halves,
and the verdict names every product that does not reproduce and the worst departure over all of them."""),

("code", '''import subprocess
import concurrent.futures as cf

CAND = RD.candidates(RECORD)
_stamps = sorted([s for s in {RD.split_run_folder(p.name)[1] for site in CHOSEN
                              for p in (WORK / site).glob("halves_*") if p.is_dir()} if s])
HALF_STAMP = _stamps[-1] if _stamps else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
print("%d proposed product(s), %d site lane job(s), stamp %s (%s)"
      % (len(CAND), CAND.groupby(["site", "rate_hz"]).ngroups if len(CAND) else 0, HALF_STAMP,
         "resumed" if _stamps else "new"))
print(CAND.to_string(index=False))

ENV = dict(os.environ)
ENV.update(OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")   # <- 3 threads a lane

def one_site(job):
    """One lane: every candidate kind of one site in turn. Two lanes on one site would share a scratch."""
    site, rate, kinds = job
    cmd = [sys.executable, "-m", "auslamp_proc.halves", "--survey", SURVEY, "--site", site,
           "--rate", str(int(rate)), "--stamp", HALF_STAMP, "--quiet", "--kinds"] + list(kinds)
    if WORK_ROOT:
        cmd += ["--work-root", str(WORK_ROOT)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=ENV)
    return site, ",".join(kinds), r.returncode, round(time.time() - t, 1), (r.stderr or "").strip()[-200:]

t_half = time.time()
if HALVES and len(CAND):
    jobs = (CAND.groupby(["site", "rate_hz"])["kind"].apply(lambda v: sorted(set(v)))
            .reset_index().values.tolist())
    print()
    print("%d lane job(s) over %d lane(s); a half pass already on disk is reported and not remade"
          % (len(jobs), LANES), flush=True)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=LANES) as ex:
        for site, kinds, rc, secs, err in ex.map(one_site, jobs):
            done += 1
            print("%2d/%2d %-9s %-22s exit %d %7.1f s %s" % (done, len(jobs), site, kinds, rc, secs,
                                                             err if rc else ""), flush=True)
HALF_RECORDS = []
for r in CAND.itertuples():
    rec = HV.read_existing(WORK, r.site, r.kind, int(r.rate_hz), stamp=HALF_STAMP,
                           band=tuple(QUALITY_BAND), agree_rho=AGREE_RHO, agree_phase=AGREE_PHASE)
    if rec is None:
        rec = dict(site=r.site, kind=r.kind, rate_hz=float(r.rate_hz), params="", stamp=HALF_STAMP,
                   half1="", half2="", half1_days=None, half2_days=None, seconds=None,
                   error="no half pass on disk", components={})
    HALF_RECORDS.append(rec)
HALF = HV.halves_table(HALF_RECORDS, out_path=OUT / "HALVES.csv")
WRITTEN.append(OUT / "HALVES.csv")
print()
print("%d proposed product(s), two half passes each; %.1f min of wall time in this cell"
      % (len(CAND), (time.time() - t_half) / 60.0))
print(HALF[["site", "kind", "rate_hz", "component", "half1_days", "half2_days", "rho_dev",
            "phase_dev_deg", "n", "reproducible"]].round(4).to_string(index=False))
print()
print(HALF.reproducible.value_counts().to_string())

# the reading goes into the record beside the choice, never in place of it
rep = {(r.site, r.kind, r.component): (r.reproducible, r.why, r.rho_dev, r.phase_dev_deg)
       for r in HALF.itertuples()}
vals, whys, devs, dphs = [], [], [], []
for r in RECORD.itertuples():
    v, w, rd, pdg = rep.get((r.site, r.kind, r.component),
                            ("UNJUDGED", "no half pass for this product", np.nan, np.nan))
    vals.append(v if r.product != "none" else "")
    whys.append(w)
    devs.append(rd)
    dphs.append(pdg)
RECORD["reproducible"] = vals
RECORD["halves_rho_dev"] = devs
RECORD["halves_phase_dev_deg"] = dphs
RECORD["note"] = [("does not reproduce on halves: %s" % w) if v == "no" else n
                  for v, w, n in zip(vals, whys, RECORD.note)]
RECORD["flagged"] = [("not reproducible" if v == "no" else
                      ("halves UNJUDGED" if v == "UNJUDGED" else "")) for v in vals]
print()
print("the proposal with the split-half departures beside it")
print(RECORD[RECORD["product"] != "none"][
    ["site", "component", "product", "bar", "held_lo_s", "held_hi_s", "agree_n", "halves_rho_dev",
     "halves_phase_dev_deg", "reproducible"]].round(4).to_string(index=False))

judged = HALF[HALF.reproducible != "UNJUDGED"]
if not len(CAND):
    print("VERDICT: UNJUDGED -- no product was proposed, so there was nothing to re-estimate on halves")
elif not len(judged):
    print("VERDICT: UNJUDGED -- the halves were scored for none of the %d proposed product(s): %s"
          % (len(CAND), "; ".join(sorted(set(HALF.why)))[:200]))
else:
    n_no = int((HALF.reproducible == "no").sum())
    print("VERDICT: PASS -- %d of %d product-component readings were scored on two half passes each; %d "
          "reproduce within %.0f %% in rho and %.1f deg in phase over %g-%g s and %d do not (%s); the "
          "worst departure over all of them is %.3f in rho and %.2f deg in phase, and a product that does "
          "not reproduce is flagged in the delivery record and not removed"
          % (len(judged), len(HALF), len(judged) - n_no, 100 * AGREE_RHO, AGREE_PHASE, QUALITY_BAND[0],
             QUALITY_BAND[1], n_no,
             "; ".join("%s %s %.3f in rho, %.2f deg" % (r.site, r.component, r.rho_dev, r.phase_dev_deg)
                       for r in HALF[HALF.reproducible == "no"].itertuples()) or "none",
             float(np.nanmax(judged.rho_dev)), float(np.nanmax(judged.phase_dev_deg))))
'''),

("md", r"""### The 10 Hz join, as the splice rule chooses it

The delivered file is the 1 Hz row above SPLICE_JOIN_S and a 10 Hz row below it, where the step at the join
holds. Three measurements decide it, in this order.

**The decomposition first.** Each 10 Hz product is read against the 1 Hz product of the same kind over
4-32 s. The median of that ratio over the kinds is the rate effect -- what the two processing paths say
about the same band -- and the spread across kinds at one rate is the kind effect. A survey whose rate
effect exceeds 4 per cent cannot be joined at all, because the join would deliver the difference between
two processing paths as a bend in the earth. Workbook 03 measures that departure on the survey's own
whole-record 10 Hz products and writes it into every 10 Hz file; this is the measurement that says whether
the survey has a short end to deliver.

The gate is read over the whole-record 10 Hz pass and over the random 25 per cent that controls the
selections, and over no other row. A selection of the most coherent hours is not comparable to a 1 Hz
product of the whole record: its departure over 4-32 s carries the selection as well as the rate, and a gate
read over it would refuse the join for a difference the join does not deliver. Every 10 Hz row is in the
table beside the gate, with `in_gate` saying which ones the median was taken over.

**The control gate.** The 10 Hz pass runs on the most coherent hours, so a selection has to earn its place
against a random selection of the same size: a row is eligible only where its 2-16 s impedance bar beats its
kind's `r25` control by CONTROL_MARGIN. A selection that does not beat a random one of the same size buys
efficiency, not a different answer. The whole-record 10 Hz pass is a control rather than a selection and is
admitted without the gate; `r25` itself is never promoted; a selection whose kind has no `r25` product is
UNJUDGED on the gate and is not eligible, and the table says so rather than passing it.

**The acceptance.** The step at the join is measured on two bands and the worse one governs: 8-16 s below,
where the delivered row would be the 10 Hz one, and 32-100 s above, where it is the 1 Hz one. A row whose
worse step exceeds SPLICE_MAX_STEP_PCT is reported not joined and keeps its 1 Hz row untouched. The band
between them is not scored: SPLICE_GUARD holds the Earth Data logger's 20.6 s instrument line, which is not
an earth response and is reproducible only to 5-14 per cent between honest processing paths, so a 2 per cent
criterion there would measure the line and not the join. The guard band is measured, printed and scored by
nothing.

Inside what the gate and the acceptance leave, the xy row is chosen on its 4-32 s level against the row it
joins and the yx row on its own short-end bar. Nothing below SHORT_FLOOR_S is delivered.

The step is read against the proposed product of that same component, which is the row a 10 Hz product
would join: the delivered row of a component is that product's own row.

The numbers are printed and not drawn. The table below carries, per 10 Hz row, the departure on each ruled
band, the guard band, the 4-32 s level, the short-end bar, the control gate's verdict and `in_gate`, which
says whether the row is one of the two the survey-wide rate gate was read over. What to look for: a row
whose two ruled bands sit on opposite sides of zero is a row whose step is a bend and not an offset, and
that is what the two-band rule exists to catch; the guard band should scatter more widely than either ruled
band, which is why nothing scores it. The step of every row that was joined is scored again in section 4,
from the delivered file against the 1 Hz row it joined."""),

("code", '''DEC = SP.decompose(PROD, read=read, band=tuple(SP.LEVEL_BAND), rate_max_pct=SP.RATE_PATH_MAX_PCT)
if len(DEC):
    print("the 10 Hz path against the 1 Hz path of the same kind over %g-%g s, in per cent"
          % SP.LEVEL_BAND)
    print(DEC.round(2).to_string(index=False))
    print()
    print("the rate against the kind")
    RATE = SP.rate_versus_kind(DEC, SP.RATE_PATH_MAX_PCT)
    print(RATE.round(2).to_string(index=False))
    RATE_OK = {r.component: bool(r.can_splice) for r in RATE.itertuples()}
    print("the rate path gate: %s"
          % ", ".join("%s %s (median %+.2f %%, the ceiling is %.0f %%)"
                      % (r.component, "passes" if r.can_splice else "REFUSES the join",
                         r.rate_pct_median, SP.RATE_PATH_MAX_PCT) for r in RATE.itertuples()))
else:
    RATE_OK = {}
    print("no site carries both a 1 Hz and a 10 Hz product of the same kind: there is no rate path to read")
print()

JOIN, SPLICE_SCORED = {}, []
for site in CHOSEN:
    proposed = picks_for(site, RECORD)
    shorts = {}
    for r in PROD[(PROD.site == site) & (PROD.rate_hz == 10.0) & PROD.on_disk].itertuples():
        shorts[(r.kind, r.selection, r.form)] = (read(r.path), r.path)
    for comp in RD.COMPONENTS:
        if comp not in proposed or not shorts:
            JOIN[(site, comp)] = dict(pick=None, scored=[], why=(
                "no 10 Hz product of this site" if comp in proposed
                else "no product of record on this component"))
            continue
        # the row a 10 Hz product joins is the delivered row, and on this component that IS the proposed
        # product's own row, so the step is read against it and not against the other component's source
        sel = SP.select_rows(site, read(proposed[comp]["path"]), shorts, READINGS, join=SPLICE_JOIN_S,
                             max_step_pct=SPLICE_MAX_STEP_PCT, guard=tuple(SPLICE_GUARD),
                             control_band=tuple(CONTROL_BAR_BAND), control_margin=CONTROL_MARGIN,
                             rate_ok=RATE_OK)
        JOIN[(site, comp)] = sel[comp]
        p = sel[comp]["pick"] or {}
        for c in sel[comp]["scored"]:
            SPLICE_SCORED.append(dict(site=site, component=comp, kind=c["kind"],
                                      selection=c["selection"], form=c.get("form", ""),
                                      file=Path(c["file"]).name,
                                      run=RD.split_run_folder(Path(c["file"]).parent.name)[0],
                                      in_gate=bool(c["selection"] in SP.GATE_SELECTIONS),
                                      passes=c["passes"], eligible=c["eligible"],
                                      control_verdict=c["control_verdict"],
                                      step_below_pct=c["step_below_pct"],
                                      step_above_pct=c["step_above_pct"], step_pct=c["step_pct"],
                                      guard_pct=c["guard_pct"], level_4_32_pct=c["level_4_32_pct"],
                                      short_bar=c["short_bar"],
                                      chosen=bool(p and c["file"] == p.get("file"))))
print("the join the splice rule proposes, per site and component")
for (site, comp), s in sorted(JOIN.items()):
    print("  %-8s %s  %s" % (site, comp, s.get("why", "")))
print()
SCORED = pd.DataFrame(SPLICE_SCORED)
if len(SCORED):
    SCORED.round(4).to_csv(OUT / "SPLICE_CANDIDATES.csv", index=False)
    WRITTEN.append(OUT / "SPLICE_CANDIDATES.csv")
    print("every 10 Hz row that was scored, %d of them; the departures are in per cent of the 1 Hz level "
          "over %g-%g s below the join, %g-%g s above it and %g-%g s in the guard band"
          % (len(SCORED), SP.STEP_BELOW[0], SP.STEP_BELOW[1], SP.STEP_ABOVE[0], SP.STEP_ABOVE[1],
             SPLICE_GUARD[0], SPLICE_GUARD[1]))
    print(SCORED[["site", "component", "kind", "selection", "form", "run", "in_gate", "passes",
                  "eligible", "step_below_pct", "step_above_pct", "guard_pct", "level_4_32_pct",
                  "short_bar", "chosen"]].round(3).to_string(index=False))
    print()
    print("the guard band over every row, measured and scored by nothing: median %+.1f %%, worst %+.1f %%"
          % (float(SCORED.guard_pct.median()),
             float(SCORED.guard_pct.iloc[int(np.nanargmax(np.abs(SCORED.guard_pct.values)))])))
'''),

("md", r"""## 3. The choice

The cell below is the analyst's. `product` is `"rule"` or a product name from the table of section 1;
`periods` is `"held"` or a pair `(lo_s, hi_s)`; `join` is `"rule"`, `None` for no 10 Hz row, or a period in
s. The tipper takes `"rule"`, which is TIPPER_FROM, or a component.

The delivery is rebuilt from the choice and not from the proposal; the two agree wherever the cell says
`"rule"`. Every departure is written into the delivered file's header as a `choice=` line and into
`surveys/<SURVEY>/final_choices.csv`, which this workbook reads before it proposes anything. A row there
marked `analyst` binds and is never overwritten by the rule; a row marked `rule` is refreshed. This run
writes the rows of this site and leaves every other site's as they stand.

The tipper is refused where the vertical channel is not measuring the vertical field: Hz a copy of a
horizontal channel, which reads a coherence of 1.00 with Hx, or Hz carrying the site's own horizontal field
at 1000-4000 s while carrying nothing of a neighbour's vertical field. The test runs at every site that
delivers, and a refused tipper is not written. A site whose magnetics are under 80 per cent finite is a test
that ran and could not judge; the tipper is written and the count is printed beside the verdict.

**This check fails if a chosen product does not pass the three tests, if chosen periods fall outside its
held band without TRIM_TO_HELD False being set, if the merged file does not read back equal to its sources,
or if the tipper refusal test was not run.**

The choice is a table and not a figure: where it is the rule's, section 2 has already drawn it against the
products it was made among, and where it is not, the departure is named here and written into the delivered
file's header. The chosen curve itself is drawn in section 4, which is the file it becomes."""),

("code", '''CHOICE = dict(xy=dict(product="rule", periods="held", join="rule"),
              yx=dict(product="rule", periods="held", join="rule"),
              tipper=dict(product="rule"))
'''),

("code", '''RULE_SPEC = dict(product="rule", periods="held", join="rule")

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def spec_for(site, comp):
    """(the choice in force, who made it, its note): an analyst row first, then the cell, then the rule."""
    a = CHOICES[(CHOICES.site == site) & (CHOICES.component == comp) & (CHOICES.chosen_by == "analyst")]
    if len(a):
        r = a.iloc[0]
        lo, hi, j = _f(r.periods_lo), _f(r.periods_hi), _f(r["join"])
        return (dict(product=str(r["product"]),
                     periods=((lo, hi) if np.isfinite(lo) and np.isfinite(hi) else "held"),
                     join=(j if np.isfinite(j) else None)), "analyst", str(r.note or ""))
    return dict(CHOICE.get(comp, RULE_SPEC)), "rule", ""

def row_named(site, comp, name):
    """The readings row a product name picks out at one site and component, or None."""
    g = READINGS[(READINGS.site == site) & (READINGS.component == comp) & (READINGS.status == "ok")
                 & (READINGS["product"] == str(name))]
    if not len(g):
        return None
    r = g.sort_values("bar").iloc[0]
    return dict(path=r.path, kind=r.kind, kind_word=r.kind_word, form=r.form, selection=r.selection,
                run=r.run, stamp=r.stamp, rate_hz=float(r.rate_hz), bar=float(r.bar),
                held_lo_s=_f(r.held_lo_s), held_hi_s=_f(r.held_hi_s), product=str(r["product"]),
                passes=bool(r.passes))

MERGES, TIPPER_ONLY, DELIVERED, SPLICE_ROWS, IDENTITY = [], [], {}, [], []
CHOICE_ROWS, DEPART, REFUSAL, OUTSIDE, NOT_SOUND = [], [], {}, [], []
t_merge = time.time()
for site in CHOSEN:
    rule_picks = picks_for(site, RECORD)
    picks, bands, joins, lines, made_by = {}, {}, {}, [], {}
    for comp in RD.COMPONENTS:
        spec, by, note = spec_for(site, comp)
        want = str(spec.get("product", "rule") or "rule")
        pick = rule_picks.get(comp) if want in ("", "rule") else row_named(site, comp, want)
        if pick is None:
            if want not in ("", "rule"):
                DEPART.append("%s %s: %s names no product of this site, so the row is empty"
                              % (site, comp, want))
            continue
        if want in ("", "rule"):
            g = READINGS[(READINGS.site == site) & (READINGS.component == comp)
                         & (READINGS.path == pick["path"])]
            pick["passes"] = bool(len(g) and bool(g.iloc[0].passes))
        held = (_f(pick.get("held_lo_s")), _f(pick.get("held_hi_s")))
        per = spec.get("periods", "held")
        if isinstance(per, str):
            band = (held if TRIM_TO_HELD and np.isfinite(held[0]) and np.isfinite(held[1]) else None)
        else:
            band = (float(per[0]), float(per[1]))
            if np.isfinite(held[0]) and np.isfinite(held[1]) and (band[0] < held[0] or band[1] > held[1]):
                OUTSIDE.append("%s %s: %g-%g s asked for, %g-%g s held"
                               % (site, comp, band[0], band[1], held[0], held[1]))
            DEPART.append("%s %s: periods %g-%g s instead of the held band"
                          % (site, comp, band[0], band[1]))
            lines.append("choice_%s_periods=%g-%g s, chosen by the analyst" % (comp, band[0], band[1]))
        j = spec.get("join", "rule")
        prop = JOIN.get((site, comp)) or {}
        if isinstance(j, str) and j.strip().lower() == "rule":
            joins[comp] = (prop.get("pick"), float(SPLICE_JOIN_S), prop.get("why", ""))
        elif j is None:
            joins[comp] = (None, float(SPLICE_JOIN_S),
                           "the choice cell asks for no 10 Hz row on this component")
            DEPART.append("%s %s: no 10 Hz row, against the rule's proposal" % (site, comp))
            lines.append("choice_%s_join=none, chosen by the analyst" % comp)
        else:
            joins[comp] = (prop.get("pick"), float(j),
                           "the choice cell joins at %g s; %s" % (float(j), prop.get("why", "")))
            DEPART.append("%s %s: join at %g s instead of %g s" % (site, comp, float(j), SPLICE_JOIN_S))
            lines.append("choice_%s_join=%g s, chosen by the analyst" % (comp, float(j)))
        if want not in ("", "rule"):
            DEPART.append("%s %s: product %s instead of the rule's %s"
                          % (site, comp, want, (rule_picks.get(comp) or {}).get("product", "none")))
            lines.append("choice_%s=product %s, chosen by the analyst%s"
                         % (comp, want, (": " + note) if note else ""))
        picks[comp], bands[comp], made_by[comp] = pick, band, (by, note)
    if not picks:
        continue
    try:
        REFUSAL[site] = DL.tipper_refusal(sv, site)
    except Exception as exc:
        REFUSAL[site] = dict(site=site, judged=False, refused=False,
                             reason="the refusal test could not run: %s" % str(exc)[:120])
    refused = bool(REFUSAL[site].get("refused"))
    tspec = str((CHOICE.get("tipper") or {}).get("product", "rule"))
    tip = None if refused else (TIPPER_FROM if tspec in ("", "rule") else tspec)
    lines.append("tipper_refusal=%s (Hz with its own Hx %.2f, with its own H %.2f, with %s's Hz %.2f)"
                 % (REFUSAL[site].get("reason", ""), _f(REFUSAL[site].get("coh_hz_hx")),
                    _f(REFUSAL[site].get("coh_hz_own_h")), REFUSAL[site].get("neighbour", "none"),
                    _f(REFUSAL[site].get("coh_hz_neighbour_hz"))))
    base_path = final_dir(site) / ("%s_1hz.edi" % site)
    m = FN.merge(sv, site, picks, base_path, tipper_from=tip, record=RECORD, keep_band=bands,
                 extra_lines=lines, verbose=False)
    m["picks"] = picks
    MERGES.append(m)
    if not m.get("written"):
        continue
    joined = {c: (v[0] if v and v[0] else None) for c, v in joins.items()}
    join_s = next((v[1] for c, v in joins.items() if v[0]), float(SPLICE_JOIN_S))
    out_path = final_dir(site) / ("%s.edi" % site)
    res = SP.splice(base_path, {c: v for c, v in joined.items() if v}, out_path, join=join_s,
                    floor=SHORT_FLOOR_S, guard=tuple(SPLICE_GUARD))
    ident = SP.unspliced_unchanged(base_path, out_path, spliced=res["spliced"])
    ident.update(site=site, spliced=" ".join(res["spliced"]) or "none",
                 periods_added=res["periods_added"])
    IDENTITY.append(ident)
    DELIVERED[site] = out_path
    m["delivered"] = str(out_path)
    m["n_delivered_periods"] = int(len(PR.read_tf(out_path).period)) if out_path.exists() else 0
    for comp in RD.COMPONENTS:
        p = joined.get(comp) or {}
        SPLICE_ROWS.append(dict(site=site, component=comp, spliced=bool(p), kind=p.get("kind", ""),
                                kind_word=p.get("kind_word", ""), selection=p.get("selection", ""),
                                form=p.get("form", ""), control_verdict=p.get("control_verdict", ""),
                                bar_2_16=p.get("bar_2_16", np.nan),
                                control_bar_2_16=p.get("control_bar_2_16", np.nan),
                                step_pct=p.get("step_pct", np.nan),
                                step_below_pct=p.get("step_below_pct", np.nan),
                                step_above_pct=p.get("step_above_pct", np.nan),
                                guard_pct=p.get("guard_pct", np.nan),
                                level_4_32_pct=p.get("level_4_32_pct", np.nan),
                                short_bar=p.get("short_bar", np.nan), passes=p.get("passes", False),
                                join_s=(join_s if p else np.nan),
                                shortest_period_s=p.get("shortest_period_s", np.nan),
                                n_short_periods=p.get("n_short_periods", 0),
                                why=(joins.get(comp) or (None, 0.0, ""))[2], all_kinds="",
                                file=p.get("file", "")))
    for comp, pick in sorted(picks.items()):
        by, note = made_by[comp]
        CHOICE_ROWS.append(FN.choice_row(site, comp, pick["product"], periods=bands.get(comp),
                                         join=(join_s if joined.get(comp) else None), chosen_by=by,
                                         note=note))
    CHOICE_ROWS.append(FN.choice_row(site, "tipper", (m.get("tipper_from") or "refused"), periods=None,
                                     join=None, chosen_by="rule",
                                     note=str(REFUSAL[site].get("reason", ""))))
    NOT_SOUND += ["%s %s: %s fails the three response tests" % (site, c, v["product"])
                  for c, v in picks.items() if not v.get("passes", True)]

SPLICE = SP.splice_table(SPLICE_ROWS)
print("%d site(s) merged and joined in %.1f min" % (len(MERGES), (time.time() - t_merge) / 60.0))
print()
print("the choice, per site and component")
print(pd.DataFrame(CHOICE_ROWS)[["site", "component", "product", "periods_lo", "periods_hi", "join",
                                 "chosen_by", "note"]].round(3).to_string(index=False)
      if CHOICE_ROWS else "nothing was chosen")
print()
print("%d departure(s) from the rule" % len(DEPART))
for d in DEPART:
    print("   %s" % d)
print()
print("the join, per site and component")
print(SPLICE[["site", "component", "spliced", "kind", "selection", "form", "join_s", "step_pct",
              "step_below_pct", "step_above_pct", "guard_pct", "level_4_32_pct", "short_bar",
              "shortest_period_s"]].round(3).to_string(index=False) if len(SPLICE) else "nothing joined")
print()
WROTE = FN.write_choices(FN.choices_path(sv), CHOICE_ROWS)
WRITTEN.append(WROTE["path"])
print("-> %s: %d row(s) written, %d analyst row(s) kept, %d row(s) in the file"
      % (WROTE["path"], WROTE["written"], WROTE["analyst_kept"], WROTE["rows"]))

bad_merge = [m["site"] for m in MERGES if not m.get("ok")]
# the criterion is that the test RAN at every delivered site. A test that ran and could not judge -- a
# site whose magnetics are under 80 per cent finite -- is a stated reading and is reported beside the
# verdict, because the tipper it governs is still written and a reader has to know the test said nothing
not_run = [s for s in sorted(DELIVERED) if s not in REFUSAL]
unjudged = [s for s in sorted(DELIVERED) if s in REFUSAL and not REFUSAL[s].get("judged")]
outside = OUTSIDE if TRIM_TO_HELD else []
print()
print("the tipper refusal test: %d delivered site(s) judged, %d refused, %d not judged (%s)"
      % (len(DELIVERED) - len(unjudged) - len(not_run),
         sum(1 for s in DELIVERED if REFUSAL.get(s, {}).get("refused")), len(unjudged),
         "; ".join("%s: %s" % (s, REFUSAL[s].get("reason", "")) for s in unjudged) or "none"))
if not MERGES:
    print("VERDICT: UNJUDGED -- no site has a product on either component, so nothing was chosen or merged")
elif NOT_SOUND or outside or bad_merge or not_run:
    print("VERDICT: FAIL -- %d chosen product(s) do not pass the three response tests (%s); %d chosen "
          "band(s) fall outside the held band with TRIM_TO_HELD True (%s); %d merged file(s) do not read "
          "back equal to their sources (%s); the tipper refusal test was not run at %d delivered site(s) "
          "(%s)"
          % (len(NOT_SOUND), "; ".join(NOT_SOUND[:3]) or "none", len(outside),
             "; ".join(outside[:3]) or "none", len(bad_merge), " ".join(bad_merge) or "none",
             len(not_run), " ".join(not_run) or "none"))
else:
    print("VERDICT: PASS -- all %d chosen product(s) over %d site(s) pass the three response tests, every "
          "chosen band lies inside its product's held band, all %d merged file(s) read back with every row "
          "equal to its source to better than %.0e relative, and the tipper refusal test ran at all %d "
          "delivered site(s), refusing %d and judging %d of them"
          % (sum(len(m.get("picks") or {}) for m in MERGES), len(MERGES), len(MERGES), FN.READBACK_RTOL,
             len(DELIVERED), sum(1 for s in DELIVERED if REFUSAL.get(s, {}).get("refused")),
             len(DELIVERED) - len(unjudged)))
'''),

("md", r"""## 4. The delivered file

One file per site in `<work_root>/<site>/final/`. `<site>_1hz.edi` is the merge of the two chosen 1 Hz
products, trimmed, and `<site>.edi` is the delivered file, which is that merge with any 10 Hz row joined
below the join period.

The merge takes the first impedance row (Zxx, Zxy) from the xy product and the second (Zyx, Zyy) from the yx
product, each with its errors, and the tipper from the product the choice names. A product on another period
grid is aligned by nearest period within 1 per cent and never interpolated: an interpolated row is a third
curve and not either product. A component with no chosen product leaves its row empty and the INFO block
says so, so a reader cannot take an empty row for a measurement.

Under TRIM_TO_HELD the file carries the periods inside each row's chosen band and no others: a value outside
that band is dropped from its row, and a period left carrying neither off-diagonal element is dropped from
the grid with the tipper on it. Every dropped period is named in the manifest with the band that dropped it.
With TRIM_TO_HELD False every period the sources carry is written and the file's INFO block says it was not
trimmed.

A site with no impedance on either component can still deliver its tipper, which is an H-only quantity and
survives two dead electric lines. Its impedance rows are written as the EDI empty-data fill and two INFO
lines name what the file is.

The INFO block carries which product each row came from, the flags and the notes per component, the frame
block, the trim line, the join line where a 10 Hz row is in the file, the cache the product was built from,
the 10 Hz caveat, any `choice=` line and the package version and date. The frame is stated in three lines:
the tensor is served in the frame it was processed in, the IGRF declination is recorded and not applied, and
the angle to turn the tensor by for true geographic north is given with the transformation.

**This check fails if any merged file does not read back with its rows equal to its sources to 1e-9
relative, if the two-source control fails where the two sources carry different yx rows, if any period or
value of a row that was not joined differs from its 1 Hz input by more than 1e-12 relative, if a joined row
reads further than SPLICE_MAX_STEP_PCT from its 1 Hz row below the join or moves it above the join, or if
any delivered file lacks the frame block or the declination.** The two-source control is what says the merge
took rows from two files: the merged Zyx must differ from the xy source's wherever the two sources differ
there. The step is worked out again here from the files -- the joined 10 Hz file against the 1 Hz row it
joined below the join, and the delivered row against that same 1 Hz row above it -- so both numbers are
scored against the files and not against the table that proposed them.

The figure is the delivered curve with its own error bars and the join marked, and nothing behind it."""),

("code", '''for site in CHOSEN:
    if picks_for(site, RECORD) or site in DELIVERED:
        continue
    g = READINGS[(READINGS.site == site) & (READINGS.status == "ok") & (READINGS.rate_hz == 1.0)
                 & (READINGS.form == "")]
    if not len(g) or not g.bar.notna().any():
        continue
    src = g.loc[g.bar.idxmin()]
    if read(src.path).t is None:
        continue
    try:
        refusal = DL.tipper_refusal(sv, site)
    except Exception as exc:
        refusal = dict(site=site, judged=False, refused=False,
                       reason="the refusal test could not run: %s" % str(exc)[:120])
    REFUSAL[site] = refusal
    out = FN.tipper_only(sv, site, src.path, final_dir(site) / ("%s.edi" % site), kind=src.kind_word,
                         refusal=refusal, record=RECORD)
    out["picks"] = {"tipper": dict(path=src.path, kind=src.kind, form=src.form, run=src.run,
                                   stamp=src.stamp, rate_hz=src.rate_hz)}
    out["delivered"] = out.get("path")
    TIPPER_ONLY.append(out)
    if out.get("written"):
        DELIVERED[site] = Path(out["path"])
    print("   %-8s tipper only from the %s product: %s" % (site, src.kind_word, out.get("reason", "")))

print()
print("%d merged file(s), %d tipper-only file(s), %d site(s) with nothing to deliver"
      % (len(MERGES), len([t for t in TIPPER_ONLY if t.get("written")]), len(CHOSEN) - len(DELIVERED)))
mtab = pd.DataFrame([dict(site=m["site"], file=Path(m.get("delivered") or m["path"]).name,
                          xy=m.get("xy", ""), yx=m.get("yx", ""), tipper=m.get("tipper_from", ""),
                          periods=m.get("n_delivered_periods", 0), dropped=m.get("n_dropped", 0),
                          readback=("%.1e" % m.get("worst_readback_relative", np.nan)),
                          check=("PASS" if m.get("ok") else "FAIL"), control=m.get("control", ""),
                          xml=("yes" if m.get("xml") else ("no: " + str(m.get("xml_error", ""))[:40])))
                     for m in MERGES])
print(mtab.to_string(index=False) if len(mtab) else "no site was merged")
print()
for m in MERGES:
    if m.get("n_dropped"):
        gone = np.asarray(m.get("dropped_periods_s", []), float)
        print("   %-8s %d period(s) dropped, %.4g to %.4g s: %s"
              % (m["site"], len(gone), float(np.min(gone)), float(np.max(gone)),
                 m.get("dropped_reason", "")))
print()
IDENT = pd.DataFrame(IDENTITY)
if len(IDENT):
    print("every period and value of a row that was not joined, against its 1 Hz input")
    print(IDENT[["site", "spliced", "periods_added", "n_base_periods", "n_out_periods",
                 "worst_period_relative", "worst_value_relative", "ok"]].to_string(index=False))
print()
JOINED_AT = {r.site: float(r.join_s) for r in SPLICE.itertuples()
             if r.spliced and np.isfinite(r.join_s)} if len(SPLICE) else {}
if FOCUS in DELIVERED:
    print("the INFO block of %s" % FOCUS)
    for line in PR.read_tf(DELIVERED[FOCUS]).meta["lines"]:
        print("   %s" % line)
    fig_del = FIG.delivered_page(
        FOCUS, read(DELIVERED[FOCUS]), final_dir(FOCUS) / ("%s_delivered.png" % FOCUS),
        period_range=tuple(PERIOD_RANGE), join_s=JOINED_AT.get(FOCUS),
        title="%s: the delivered transfer function" % FOCUS,
        caption="%s over %g-%g s with its own error bars: the xy row from %s, the yx row from %s, %s. The "
                "file carries %d period(s)%s."
                % (Path(DELIVERED[FOCUS]).name, PERIOD_RANGE[0], PERIOD_RANGE[1],
                   ", ".join("%s" % v["product"] for c, v in sorted(
                       (next((m for m in MERGES if m["site"] == FOCUS), {}).get("picks") or {}).items())
                       if c == "xy") or "none",
                   ", ".join("%s" % v["product"] for c, v in sorted(
                       (next((m for m in MERGES if m["site"] == FOCUS), {}).get("picks") or {}).items())
                       if c == "yx") or "none",
                   "trimmed to the held band" if TRIM_TO_HELD else "not trimmed",
                   int(len(read(DELIVERED[FOCUS]).period)),
                   (", %d dropped by the trim" % next((m.get("n_dropped", 0) for m in MERGES
                                                       if m["site"] == FOCUS), 0))))
    WRITTEN.append(fig_del)
    display(Image(filename=str(fig_del)))

no_frame = []
for site, p in sorted(DELIVERED.items()):
    kv = PR.read_tf(p).meta["parameters"]
    lack = [k for k in ("reference_frame", "declination_deg", "to_geographic_north_deg")
            if not str(kv.get(k, "")).strip()]
    if lack:
        no_frame.append("%s: %s" % (site, ", ".join(lack)))
bad_merge = [m["site"] for m in MERGES if not m.get("ok")]
no_control = [m["site"] for m in MERGES if "control FAIL" in str(m.get("control", ""))]
shifted = [r.site for r in IDENT.itertuples() if not r.ok] if len(IDENT) else []
# the two numbers the join has to answer for, worked out here from the files and not from the table above.
# BELOW the join: the 10 Hz file that was joined against the 1 Hz row it joined, the median ratio of
# apparent resistivity over the ruled band, which is the step. ABOVE it: the delivered row against the same
# 1 Hz row, which must be the same numbers, because a join must not move the row it joined onto.
STEP_BACK, over = [], []
for r in (SPLICE.itertuples() if len(SPLICE) else []):
    if not r.spliced or r.site not in DELIVERED or not str(r.file):
        continue
    pb, rb, _e, _ph, _pe = RD.curve(PR.read_tf(final_dir(r.site) / ("%s_1hz.edi" % r.site)), r.component)
    ps, rs, _e2, _ph2, _pe2 = RD.curve(PR.read_tf(r.file), r.component)
    pd_, rd_, _e3, _ph3, _pe3 = RD.curve(PR.read_tf(DELIVERED[r.site]), r.component)
    mb = (pb >= SP.STEP_BELOW[0]) & (pb <= SP.STEP_BELOW[1]) & np.isfinite(rb) & (rb > 0)
    ms = np.isfinite(rs) & (rs > 0)
    v = np.exp(np.interp(np.log(pb[mb]), np.log(ps[ms]), np.log(rs[ms]),
                         left=np.nan, right=np.nan)) / rb[mb] if mb.any() and ms.any() else np.zeros(0)
    below = 100.0 * (float(np.median(v[np.isfinite(v)])) - 1.0) if np.isfinite(v).any() else np.nan
    ma = (pb >= SP.STEP_ABOVE[0]) & (pb <= SP.STEP_ABOVE[1]) & np.isfinite(rb) & (rb > 0)
    idx = [int(np.argmin(np.abs(pd_ - t))) for t in pb[ma]]
    above = (float(np.nanmax(np.abs(np.asarray(rd_)[idx] / rb[ma] - 1.0))) if idx else np.nan)
    STEP_BACK.append(dict(site=r.site, component=r.component, join_s=float(r.join_s),
                          step_below_pct=below, worst_above_relative=above, n_below=int(np.isfinite(v).sum())))
    if np.isfinite(below) and abs(below) > SPLICE_MAX_STEP_PCT:
        over.append("%s %s %+.1f %% below the join" % (r.site, r.component, below))
    if np.isfinite(above) and above > 1e-6:
        over.append("%s %s moved the 1 Hz row above the join by %.2e relative"
                    % (r.site, r.component, above))
if STEP_BACK:
    print()
    print("the join scored again from the files: the joined 10 Hz row against the 1 Hz row over %g-%g s, "
          "and the delivered row against the same 1 Hz row over %g-%g s"
          % (SP.STEP_BELOW[0], SP.STEP_BELOW[1], SP.STEP_ABOVE[0], SP.STEP_ABOVE[1]))
    print(pd.DataFrame(STEP_BACK).round(4).to_string(index=False))
if not MERGES and not TIPPER_ONLY:
    print("VERDICT: UNJUDGED -- no site has a chosen product on either component, so no file was written")
elif bad_merge or no_frame or shifted or over:
    print("VERDICT: FAIL -- %d merged file(s) do not read back equal to their sources to %.0e relative or "
          "fail the two-source control (%s; control failures %s); %d file(s) moved a period or a value of "
          "a row that was not joined (%s); %d joined row(s) do not hold %.1f %% at the join or moved the "
          "1 Hz row above it (%s); %d delivered file(s) lack a frame line (%s)"    # <- the join, rescored
          % (len(bad_merge), FN.READBACK_RTOL, " ".join(bad_merge) or "none",
             " ".join(no_control) or "none", len(shifted), " ".join(shifted) or "none",
             len(over), SPLICE_MAX_STEP_PCT, "; ".join(over[:4]) or "none",
             len(no_frame), "; ".join(no_frame) or "none"))
else:
    n_control = sum(1 for m in MERGES if "control PASS" in str(m.get("control", "")))
    print("VERDICT: PASS -- all %d merged file(s) read back with every row equal to its source to better "
          "than %.0e relative; the two-source control holds at the %d site(s) whose two sources carry "
          "different yx rows and is n/a at the other %d; every period and value of a row that was not "
          "joined came back identical to its 1 Hz input to better than %.0e relative over %d file(s); each "
          "of the %d joined row(s) reads inside %.1f %% of its 1 Hz row below the join and leaves that row "
          "untouched above it; all %d delivered file(s) carry the frame, the declination and the angle to "
          "geographic north"
          % (len(MERGES), FN.READBACK_RTOL, n_control, len(MERGES) - n_control, SP.IDENTITY_TOL,
             len(IDENT), len(STEP_BACK), SPLICE_MAX_STEP_PCT, len(DELIVERED)))
'''),

("md", r"""## 5. What was written

The delivery of this site is appended to the survey's four tables under `<work_root>/survey/`, which is
where the record of a survey accumulates one row per delivered site as the workbook is run site by site:
PRODUCTS_OF_RECORD.csv the proposal per component with its reason, READINGS.csv every product and every
statistic it was read on, SPLICE.csv what was done to each row at the join, and FINAL_MANIFEST.csv the
sha256 of the delivered file and of every product it came from. A re-run of this site replaces that site's
rows and leaves every other site's untouched."""),

("code", '''REC_OUT = FN.write_record(OUT, RECORD, READINGS, MERGES + TIPPER_ONLY, splice=SPLICE, sites=CHOSEN)
WRITTEN += list(REC_OUT["written"].values())
MAN = REC_OUT["manifest"]
if RESAMPLE:
    for site, p in sorted(DELIVERED.items()):
        r = FN.resample(p, final_dir(site) / ("%s_resampled.edi" % site))
        WRITTEN.append(r["path"])
    print("%d file(s) also written on the ten-per-decade grid, beside the delivered file and never in "
          "place of it" % len(DELIVERED))
for name, p in sorted(REC_OUT["written"].items()):
    print("   %-24s %s" % (name, p))
print("   the manifest holds %d delivered file(s) of %s, %d of them written by this run"
      % (MAN[MAN.role == "final"].site.nunique(), sv.cfg["name"], len(DELIVERED)))
print()

rows = []
for p in list(WRITTEN) + [q for s in sorted(DELIVERED) for q in
                          (Path(DELIVERED[s]), Path(DELIVERED[s]).with_suffix(".xml"))]:
    p = Path(p)
    if p.exists():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d files, %.1f MB, in %.1f minutes" % (len(files), files.kb.sum() / 1024, (time.time() - T0) / 60))
print(files.head(80).to_string(index=False))
if len(files) > 80:
    print("   ... and %d more" % (len(files) - 80))
print()
print("ignored: %d %s product(s) on disk, read by nothing above" % (len(IGNORED), RD.DROPPED_KIND))
for p in sorted(IGNORED.path):
    print("   %s" % p)
'''),
]


NOTEBOOKS = {"01_survey.ipynb": WB01, "02_records.ipynb": WB02, "03_process.ipynb": WB03,
             "04_products.ipynb": WB04, "05_site.ipynb": WB05, "06_final.ipynb": WB06}


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
