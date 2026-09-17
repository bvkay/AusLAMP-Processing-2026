r"""Workbook 03, one site: its references, its transfer functions and the page of them.

The cells are a Python list of ("md", text) and ("code", source). workbooks/make_workbooks.py imports the
list from here and writes 03_site.ipynb.

@author: ben kay (ben@auscope.org.au)
"""

WB03_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/; copy surveys/_template/ for your own survey
SITE = "Q53N"                 # the site this workbook processes; one site is processed at a time
RUN = "first"                 # the run name; this site's 1 Hz transfer functions land in <SITE>/<RUN>_<stamp>/
WORK_ROOT = None              # None = survey.yaml work_root; every reference and transfer function lands there
KINDS = ["remote", "stack", "obs", "stack_obs"]              # the code keys of the four reference kinds
RATES = [1, 10]               # 1 is the whole record; 10 runs on the stretch below, never the whole record
PARAMS = "kaiser20_75"        # the Aurora parameter set: kaiser20_50 | dpss4_75 | kaiser20_w512
REDO = False                  # True remakes a transfer function whose EDI is already on disk
'''

WB03_RULES = '''# ---- the rule thresholds: a change here changes which reference the transfer functions rest on ----
COH_MIN = 0.5                 # the remote-site gate at 20-200 s; branches 1 and 2 need it
COH_RELAX = 0.3               # branch 3 relaxes to this where no clean candidate reaches COH_MIN
MIN_OVERLAP_DAYS = 20         # the overlap floor, taken with 0.75 x the days the target can use
STACK_CUTOFF = 0.5            # a member below this fleet coherence at 100-1000 s is refused
STACK_MAX = 8                 # the best this many members enter the stack
STACK_MIN = 2                 # fewer than this and the stack is refused: one member is a remote site renamed
EVENT_FRAC_MAX = 0.02         # clean: at most this fraction of the 600 s chunks flagged as events
BASELINE_MAX = 10             # clean: at most this multiple of the survey median baseline
'''

WB03_SELECT = '''# ---- the stretch the 10 Hz pass runs on: one rule, used here and by workbook 04 ----
KINDS10 = ["remote", "stack"]  # the kinds at 10 Hz; the observatory is not one of them
WINDOW_COH = 0.5              # an hour is inside a stretch where BOTH lines read above this at 20-200 s
STRETCH_MAX_H = 48            # a stretch longer than this is cut to its best 48 contiguous hours
SEED = 20260916               # the draw the control stretch of the same length is placed under
RUN10 = "sel10"               # the run name the 10 Hz transfer functions land in, <SITE>/<RUN10>_<stamp>/
'''

WB03_SETUP = '''import json
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from IPython.display import Image, display

import auslamp_proc
from auslamp_proc import agreement as AG, geo, readings as RD, survey as SV
from auslamp_proc import transfer_functions as TFN
from auslamp_proc.raw import cache
from auslamp_proc.figures import process as FIGP, transfer_functions as FIG
from auslamp_proc.process import KIND_WORD, aurora_run, batch as BATCH, edi as EDI, frame as FR
from auslamp_proc.process import mth5_build, provenance as PROV, rate as RATE, references as REF
from auslamp_proc.process import selection as SEL, transients as TR

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

# nothing here reads the raw tree: a site enters the reference pool only where workbook 02 has built its
# cache, and the sites that have none are named rather than passed over
ALL_SITES = [s for s in sv.sites.site if cached(s)]
NO_CACHE = [s for s in sv.sites.site if not cached(s)]
if SITE not in ALL_SITES:
    raise SystemExit("%s has no 1 Hz cache under %s; run workbook 02 over it first" % (SITE, WORK))
OUT = WORK / "survey"
OUT.mkdir(parents=True, exist_ok=True)
SITE_DIR = WORK / SITE
SITE_DIR.mkdir(parents=True, exist_ok=True)
WRITTEN = []

RATES1 = [r for r in RATES if int(r) == 1]      # the whole-record lane
RATES10 = [r for r in RATES if int(r) == 10]    # the stretch lane; a 10 Hz pass is never the whole record

def show(path):
    """Record a figure in WRITTEN and draw it inline. One call per figure."""
    WRITTEN.append(Path(path))
    display(Image(filename=str(path)))

def quiet_window(arrays, fs=1.0, hours=6.0, skip_days=1.0):
    """The first sample of the quietest whole window of `hours`: every series finite and the target's own
    per-minute Hx spread the smallest. A window is scored on the record as it is, so `quiet` is the field's
    own quiet and not a judgement about the site."""
    w = int(round(hours * 3600 * fs))
    n = min(len(v) for v in arrays)
    best, best_v = None, np.inf
    for i in range(int(skip_days * 86400 * fs), max(0, n - w + 1), w):
        if not all(np.isfinite(v[i:i + w]).all() for v in arrays):
            continue
        s = float(np.std(np.asarray(arrays[0][i:i + w], float)))
        if s < best_v:
            best, best_v = i, s
    return best

# the pool thresholds are survey.yaml values; the parameter cell above governs this run of the workbook
sv.cfg.setdefault("pool", {})
sv.cfg["pool"]["event_frac_max"] = EVENT_FRAC_MAX
sv.cfg["pool"]["baseline_max"] = BASELINE_MAX

def store(rate):
    """The reference store at one rate, carrying this workbook's thresholds."""
    return REF.Store(sv, ALL_SITES, rate=rate, coh_min=COH_MIN, coh_relax=COH_RELAX,
                     min_overlap_days=MIN_OVERLAP_DAYS, cutoff=STACK_CUTOFF, n_max=STACK_MAX,
                     n_min=STACK_MIN)

ST = store(RATES[0])
OBS = (sv.cfg.get("observatory") or {}).get("code", "")

# The stamp is the launch time in UTC of the run. A run is resumed rather than restarted: where
# <SITE>/<RUN>_<stamp>/ already exists the latest of those stamps is taken up again, so re-running this
# workbook finishes a run that was stopped part way and REDO False leaves what is on disk alone.
STAMP = BATCH.resume_stamp(WORK, [SITE], RUN)
STAMP10 = BATCH.resume_stamp(WORK, [SITE], RUN10)

print("survey       %s" % sv.cfg["name"])
print("work root    %s" % WORK)
print("site         %s" % SITE)
print("pool from    %d of the %d sites in sites.csv, the ones with a 1 Hz cache (a reference may come from "
      "a site this workbook does not process)" % (len(ALL_SITES), len(sv.sites)))
if NO_CACHE:
    print("no cache     %s -- run workbook 02 over them to bring them into the pool" % " ".join(NO_CACHE))
print("kinds        %s" % ", ".join("%s (%s)" % (KIND_WORD[k], k) for k in KINDS))
print("rates        %s Hz, parameter set %s" % (", ".join(str(r) for r in RATES), PARAMS))
print("10 Hz        the longest stretch with both lines above %.2f at %g-%g s, at most %d h, and a control "
      "of the same length under seed %d" % (WINDOW_COH, SEL.SCORE_BAND_S[0], SEL.SCORE_BAND_S[1],
                                            STRETCH_MAX_H, SEED))
print("run          %s_%s and %s_%s  ->  %s" % (RUN, STAMP, RUN10, STAMP10, SITE_DIR))
print("engine       Aurora %s, mth5 %s" % (aurora.__version__, __import__("mth5").__version__))
print("observatory  %s at %s" % (OBS, (sv.cfg.get("observatory") or {}).get("archive", "")))
'''

WB03 = [
("md", r"""# 03 -- One site: its references and its transfer functions

This workbook turns one site's cache into transfer functions. It applies the decisions.csv row and the frame,
scans the record for transients, builds the four references the site can be processed against, writes one
MTH5 per pass with one run per kept stretch, runs Aurora, writes an EDI and an XML with the provenance of
what was done, and draws every transfer function of the site on one page.

One site is processed at a time. A survey is processed by the batch entry, which runs this same code path
one lane per site:

    python -m auslamp_proc.process.batch --survey queensland_phase1 --sites all --lanes 3

Four kinds are built, and they are not alternatives to choose between here: every one of them is made, and
which of them becomes the transfer function of record is workbook 05's question.

| word | what it is | code key | what it is for |
|---|---|---|---|
| remote site | one other site's H | `remote` | cancels the noise that is uncorrelated between the two sites; costs the variance of one more record |
| fleet stack | a coherence-weighted mean of several sites' H | `stack` | averages down each member's own noise, and needs the members aligned first |
| observatory | an INTERMAGNET one-second record | `obs` | far and quiet, and reaches the long periods |
| stack + observatory | the stack with the observatory as a member | `stack_obs` | the stack's short end with the observatory's long end |

The single station is not a kind of this package: noise in H biases it low and its error bars carry no sign
of that bias.

The frame is fixed and stated in every file: the horizontal magnetic pair is turned so its mean Hy is zero,
which takes out the fluxgate's hand-compass misalignment, and the IGRF declination is recorded and not
applied. Every member of a reference is turned into its own mean-field frame before it is stacked.

The record is never cut. A mask that removes a transient is applied by giving Aurora one run per kept stretch
of at least 3,600 s: cutting and concatenating puts a step at every join, and a step common to E, H and the
reference is coherent between them, so a robust regression fits it rather than down-weighting it.

Ten checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported as FAIL and is
not revised afterwards.

Every step shows what it did as a figure beside the table that decided it, so a parameter can be changed for
a re-run on what the figure shows rather than on the numbers alone; each figure's caption names the
parameters that move it and what moving them does.

The 1 Hz pass is the whole record. The 10 Hz pass is not: it runs on the longest stretch of hours in which
both electric lines follow the field, with a stretch of the same length placed at random elsewhere as its
control. What the 10 Hz row reads against the 1 Hz row is measured on this survey's own whole-record 10 Hz
passes over 4-32 s, printed below and written into every 10 Hz file. The two rates are not joined here, which
is workbook 05's work."""),

("code", WB03_PARAMS),
("code", WB03_RULES),
("code", WB03_SELECT),

("md", r"""## The survey, the site and the run

Everything below runs on `surveys/<SURVEY>/` and writes under that survey's `work_root`. The run name and the
launch stamp name one folder, `<SITE>/<RUN>_<stamp>/`, so every transfer function of one run sits together
with the provenance of the run that made it. The reference pool is every site with a 1 Hz cache and not the
one site being processed: a reference comes from another site's record."""),

("code", WB03_SETUP),

("md", r"""## The site and its decisions

The `sites.csv` facts every file's header carries, and the `decisions.csv` row the pass is built under. Six
hand decisions and the five signs are applied in one place, `process.frame.apply_decisions`, in a fixed
order: the electric exchange, the magnetic exchange, the magnetic gain, the electric shift, the signs, and a
borrowed magnetic pair. `raw.cache.load_decided` is the read side and is what the pass itself calls, so the
list printed below is the list every file of this site carries as its own `decision=` lines.

A cell reading `decide` is used as the neutral value and recorded as undecided: the transfer function is
made, its EDI carries a `signs_undecided` line naming the channels, and its provenance carries a caveat.
Nothing is decided silently.

One decision of this survey cannot be written down. At Q86N the campaign put the fleet stack's H in for a
leaky own H; the `h_lender` column takes a site name and a stack is not a site, so the cell is left `decide`
rather than written `none`, which would be false. The cell below names every open decision of this site for
the same reason: an open cell is reported, not assumed."""),

("code", '''row = sv.site(SITE)
dec = sv.decision(SITE)
signs = {c: FR.read_sign(dec.get(FR.SIGN_COLUMN[c])) for c in FR.CHANNELS}
print(pd.DataFrame([dict(site=SITE, days=row.days, lat=row.lat, lon=row.lon, dipole_n_m=row.dipole_n_m,
                         dipole_e_m=row.dipole_e_m, declination_deg=row.declination_deg,
                         signs=" ".join("%s%+d" % (c, signs[c][0]) for c in FR.CHANNELS),
                         undecided=" ".join(c for c in FR.CHANNELS if not signs[c][1]) or "none",
                         remote_site=dec.get("remote_site", ""),
                         stack_members=dec.get("stack_members", ""),
                         rot_regimes=dec.get("rot_regimes", ""),
                         rot_drop=dec.get("rot_drop", ""))]).to_string(index=False))

t_dec = time.time()
T0_SITE, ARRAYS, DECISIONS_APPLIED, RECORD = cache.load_decided(SITE, sv, rate=RATES1[0])
print()
print("the decided record: %d samples at %g Hz, read in %.1f s"
      % (len(ARRAYS["Hx"]), RATES1[0], time.time() - t_dec))
print()
print("the decisions applied to %s, in the order they are applied:" % SITE)
for d in DECISIONS_APPLIED:
    print("   %s" % d)
if not DECISIONS_APPLIED:
    print("   none: every cell of this row is the neutral value")
print()
OPEN = list(RECORD.get("open") or [])
print("open decisions, still reading decide: %s" % ("; ".join(OPEN) or "none"))
print("undecided signs: %s" % (" ".join(RECORD.get("undecided") or []) or "none"))
print()
print("the four kinds, as words and as the code keys that appear in file names")
print(pd.DataFrame([dict(word=KIND_WORD[k], key=k)
                    for k in REF.KINDS_WITH_STORE]).to_string(index=False))
'''),

("md", r"""## The frame

Once the decisions are applied, one thing more happens to the record before anything is estimated from it:
the horizontal magnetic pair is turned by theta = atan2(mean Hy, mean Hx), which puts the mean Hy at zero.
The electric lines are left as laid.

The rotation is a rigid turn of the pair, so it cannot change the total horizontal field at any sample,
which is what the check below tests. The angle is a property of how the sensor was laid, not of the field, and it is
recorded in every file as `h_rotation_deg`. The IGRF declination is recorded beside it and not applied --
turning by IGRF as well would rotate the record twice.

**This check fails if the rotated record has |mean Hy| > 1e-6 nT or a changed |H| at any sample.** |H| is
compared before and after the turn, sample by sample, on the same float64 arithmetic the processing uses; the
bound on the comparison is 1e-6 nT, which is eleven orders of magnitude below the 50,000 nT the field itself
carries."""),

("code", '''h0 = {c: np.asarray(ARRAYS[c], float).copy() for c in ("Hx", "Hy")}
before = np.hypot(h0["Hx"], h0["Hy"])
h1, ANGLE = FR.rotate_to_mean_field({c: h0[c].copy() for c in h0},
                                    regimes=FR.parse_regimes(dec.get("rot_regimes")),
                                    drop=FR.parse_regimes(dec.get("rot_drop")), fs=float(RATES1[0]))
after = np.hypot(h1["Hx"], h1["Hy"])
g = np.isfinite(before) & np.isfinite(after)
fr = pd.DataFrame([dict(site=SITE,
                        angle_deg=(ANGLE if isinstance(ANGLE, float) else "%d regimes" % len(ANGLE)),
                        declination_deg=row.declination_deg,
                        mean_Hy_nT=float(np.nanmean(h1["Hy"])),
                        worst_dH_nT=(float(np.nanmax(np.abs(after[g] - before[g]))) if g.any() else np.nan),
                        samples=int(g.sum()),
                        H_signs=" ".join("%s%+d" % (c, RECORD["signs"][c]) for c in ("Hx", "Hy")))])
print(fr.to_string(index=False))
print()
print("the angle is the sensor's own misalignment and the declination beside it is the field's; the two are "
      "different quantities and neither is the other")

bad_hy = fr[fr.mean_Hy_nT.abs() > 1e-6]
bad_h = fr[fr.worst_dH_nT > 1e-6]
print()
if not int(fr.samples.sum()):
    print("VERDICT: UNJUDGED -- the rotated record carries no finite sample to score")
elif len(bad_hy) or len(bad_h):
    print("VERDICT: FAIL -- %s leaves a mean Hy of %.3g nT and changes |H| by %.3g nT over %d sample(s)"
          % (SITE, float(fr.mean_Hy_nT.iloc[0]), float(fr.worst_dH_nT.iloc[0]), int(fr.samples.iloc[0])))
else:
    print("VERDICT: PASS -- %s rotates to a mean Hy of %.3g nT and changes |H| by at most %.3g nT over %d "
          "sample(s); the angle is %s and the declination %.2f deg"
          % (SITE, float(fr.mean_Hy_nT.iloc[0]), float(fr.worst_dH_nT.iloc[0]), int(fr.samples.iloc[0]),
             fr.angle_deg.iloc[0], float(row.declination_deg)))
del before, after, g
'''),

("md", r"""What the turn did. The two clouds have the same shape and the same extent and differ by a rotation
about the origin, because the turn is rigid; the turned cloud's mean vector lies along the Hx axis, which is
what a mean Hy of zero means. A cloud that changed shape would be a scaling or a sign, not a rotation."""),

("code", '''show(FIGP.frame_hodogram(h0, h1, ANGLE, SITE, fr, SITE_DIR / "03_frame.png", fs=float(RATES1[0])))
del h0, h1
'''),

("md", r"""## The clean pool

A site may reference another only if its own magnetic record is fit to. The tail scan walks each rotated
record in 600 s chunks and takes the median band power of Hx, Hy, Ex and Ey over 20-200 s
(`auslamp_proc.process.transients.scan`, survey.yaml `pool`), and two tests are read off it.

The event test is fleet-normalised. A chunk is an event where the site's own power exceeds 100 times its own
median and the rest of the fleet does not see the same thing: either fewer than 3 other sites were recording
then, or this site runs more than 10 times the median of the others. A substorm lifts every site in the array
within the same ten minutes and is signal; a power-cycle or a vehicle lifts one, and only the second is a
fault. The site's own median cannot tell them apart.

The baseline test catches the other failure: a site that never spikes because it is loud all the time. The
denominator is one median over every site with a scan, so the level is a property of the instrument and the
band rather than of whoever was deployed alongside.

The band is 20-200 s. On a 1 Hz cache the 2-20 s band sits on the anti-alias filter's roll-off and reads each
unit's own noise floor.

    clean = event fraction <= EVENT_FRAC_MAX and baseline <= BASELINE_MAX x the survey median

The scan runs over every site with a cache, quietly, because the pool is a property of the survey and not of
this site. This site's own row is printed under the summary, with the two thresholds beside it.

The record the scan reads carries every decision of `decisions.csv` except `h_lender`, which is the record
`references.Store.rotated` reads for a member, a remote or a candidate. The pool asks whether a site's own
magnetic record is fit for another site to reference, so a site whose H is borrowed is judged on the record
it wrote itself and not on its lender's.

**This check fails if any site the scan reached was scored on no chunk.**"""),

("code", '''t_pool = time.time()
scan_rows, scanned_now = [], 0
for s in ALL_SITES:                              # <- every site with a cache: the pool is not this site
    if TR.load_series(s, WORK) is None:
        d_s = sv.decision(s)
        # with_lender False: the pool asks whether a site's OWN magnetic record is fit for another site to
        # reference, which is the record references.Store.rotated reads, so a site whose H is borrowed is
        # never judged on its lender's record and offered under its own name
        t0s, arrays_s, _applied_s, _rec_s = cache.load_decided(s, sv, rate=1, with_lender=False)
        arrays_s, _ang = FR.rotate_to_mean_field(arrays_s,
                                                 regimes=FR.parse_regimes(d_s.get("rot_regimes")),
                                                 drop=FR.parse_regimes(d_s.get("rot_drop")), fs=1.0)
        scan_rows.append(TR.scan(s, t0s, arrays_s, WORK, fs=1.0, cfg=sv.cfg))
        del arrays_s
        scanned_now += 1
    else:
        scan_rows.append(TR.summarise(s, *TR.load_series(s, WORK)))

pool, pool_table = ST.clean_pool(force=True)
base = TR.survey_baseline(ALL_SITES, WORK)
scan_t = pd.DataFrame(scan_rows).merge(pool_table, on="site")
print("%d site(s) scanned now, %d read from the scan already on disk, %.1f s"
      % (scanned_now, len(ALL_SITES) - scanned_now, time.time() - t_pool))
print("the survey median baseline is %.4g nT^2/Hz at 20-200 s over %d site(s) with a scan"
      % (base, len(scan_t)))
print("the pool (%d of %d): %s" % (len(pool), len(scan_t), " ".join(pool)))
print("excluded, with the reason:")
for r in pool_table[~pool_table.clean].itertuples():
    print("   %-9s %s" % (r.site, r.reason))
print()
print("%s against the two thresholds" % SITE)
mine = scan_t[scan_t.site == SITE]
print(mine[["site", "chunks", "days", "Hx_median", "Hy_median", "event_frac", "baseline", "clean",
            "reason"]].round(4).to_string(index=False))
print("   event_frac %.4f against EVENT_FRAC_MAX %.4f; baseline %.2fx against BASELINE_MAX %gx"
      % (float(mine.event_frac.iloc[0]), EVENT_FRAC_MAX, float(mine.baseline.iloc[0]), BASELINE_MAX))

unjudged = pool_table[~pool_table.judged]
print()
if not len(pool_table):
    print("VERDICT: UNJUDGED -- no site was scanned, so the pool scored nothing")
elif len(unjudged):
    print("VERDICT: FAIL -- the scan scored no chunk at %d of %d site(s) (%s)"
          % (len(unjudged), len(pool_table), ", ".join(unjudged.site)))
else:
    print("VERDICT: PASS -- every one of the %d site(s) was scored on at least one chunk, the thinnest "
          "being %d chunk(s) at %s; %d are clean at events <= %s and baseline <= %sx, and %s is %s"
          % (len(pool_table), int(scan_t.chunks.min()), scan_t.site[scan_t.chunks.idxmin()], len(pool),
             EVENT_FRAC_MAX, BASELINE_MAX, SITE,
             "in the pool" if SITE in pool else "excluded: %s" % mine.reason.iloc[0]))
'''),

("md", r"""## The remote site

Every candidate in this site's own overlap group is scored: the distance in km, the usable overlap in days,
its event fraction and baseline from the scan above, and the event-free 20-200 s coherence of the pair. The
coherence is a chunk median and never one pass over the whole record -- one transient inside one Welch
segment otherwise takes a whole record's score to the floor -- and nothing is concatenated: the uncut overlap
is walked in whole 1,024 s segments and a segment is used only where the mask is true right through it, so
the estimator never reads a join it made itself.

A candidate is scored only where its overlap reaches min(MIN_OVERLAP_DAYS, 0.75 x the days the target can
use). The days the target can use, not the length of its cache: a cache is as long as the logger ran, not as
long as the magnetics are sound.

The choice is then the five-branch rule, and the branch is recorded in the file:

1. clean, coh >= COH_MIN, overlap >= 90 per cent -- the nearest of them
2. clean, coh >= COH_MIN -- the longest overlap
3. clean, coh >= COH_RELAX -- the most coherent
4. coh >= COH_MIN but not clean -- the fewest events
5. none of the above -- the nearest by km

Where `decisions.csv` names a remote, that site is used and the rule's own choice is printed beside it as a
reading.

**This check fails if the chosen remote has no overlap with this site or is not in this site's overlap
group.**"""),

("code", '''groups = ST.groups()
scores = ST.score_candidates(SITE)
ch = ST.remote_site(SITE, scores=scores)
ov = ST.overlap(SITE, ch["name"]) if ch.get("name") else None
coh = ch.get("coh")
if coh is None and ch.get("name") and ov is not None:
    # a remote decisions.csv names from outside the clean pool is not in the candidate table, so it carries
    # no score there; it is measured here on the same rule so the column is never empty
    coh = round(float(ST.pair_coherence(SITE, ch["name"])["coh"]), 3)
rem = pd.DataFrame([dict(site=SITE, campaign_remote=str(dec.get("remote_site", "")),
                         rule_remote=ch.get("rule_name"), rule_branch=ch.get("rule_branch"),
                         rule_coh=ch.get("rule_coh"), chosen=ch.get("name"), source=ch.get("source"),
                         coh=coh, overlap_days=(round((ov[1] - ov[0]) / 86400.0, 2) if ov else 0.0),
                         group=groups.get(SITE, ""), remote_group=groups.get(ch.get("name"), ""),
                         n_candidates=len(ch.get("candidates", [])))])
print("the candidates of %s" % SITE)
print(scores.round(4).to_string(index=False))
print()
print(rem.to_string(index=False))
print()
print("the rule took branch %s: %s" % (ch.get("rule_branch"), ch.get("reason", "")))
named = str(dec.get("remote_site", "")).strip()
if named.lower() not in ("decide", "", "nan", "none") and named != ch.get("rule_name"):
    print("decisions.csv names %s and the rule would take %s: a reading, not a criterion"
          % (named, ch.get("rule_name")))

no_ov = float(rem.overlap_days.iloc[0]) <= 0
wrong_group = bool(rem.group.iloc[0]) and rem.group.iloc[0] != rem.remote_group.iloc[0]
print()
if rem.chosen.isna().all():
    print("VERDICT: FAIL -- %s was offered no remote at all: %s" % (SITE, ch.get("reason", "")))
elif no_ov or wrong_group:
    print("VERDICT: FAIL -- the chosen remote %s overlaps %s by %.2f d and sits in group %s against this "
          "site's %s" % (rem.chosen.iloc[0], SITE, float(rem.overlap_days.iloc[0]),
                         rem.remote_group.iloc[0] or "none", rem.group.iloc[0] or "none"))
else:
    print("VERDICT: PASS -- the chosen remote %s overlaps %s by %.2f d and sits in this site's own overlap "
          "group %s; it came from %s, at a 20-200 s coherence of %s"
          % (rem.chosen.iloc[0], SITE, float(rem.overlap_days.iloc[0]), rem.group.iloc[0] or "none",
             rem.source.iloc[0], rem.coh.iloc[0]))
'''),

("md", r"""The candidates as a scatter. Look at where they sit relative to the two coherence lines and the
overlap line: a site whose candidates all sit below COH_MIN took branch 3 or 5 and its remote is the best of
a weak field, not a good reference. A site with several candidates above COH_MIN and to the right of the
overlap line had a real choice, and the rule took the nearest of them."""),

("code", '''named_show = str(dec.get("remote_site", "")).strip()
named_show = named_show if named_show.lower() not in ("decide", "", "nan", "none") else None
show(FIGP.remote_scatter(scores, ch, SITE, SITE_DIR / "03_remote.png",
                         branches={ch.get("rule_branch"): 1}, coh_min=COH_MIN, coh_relax=COH_RELAX,
                         named=named_show))
'''),

("md", r"""## The fleet stack

A member's weight is its median coherence with the fleet at 100-1000 s, and never its coherence with the
target: whether a reference is a good measurement of the regional field is a question about the reference and
the field, and weighting a member by its agreement with the target puts the target's own noise into its
reference. The fleet table below is one median per member over its pairs with the rest of the pool, and it is
the same table for every target.

Members below STACK_CUTOFF are refused, the best STACK_MAX are kept, and a stack with fewer than STACK_MIN
members is refused outright: a one-member stack is a remote site renamed. A site that lends this site its H
is refused membership, because a lender is never a member of a reference that feeds the site it lends to.

Each member is aligned first. The lag is the median 5-20 s cross-correlation over up to eight event-free
two-day windows spread across the record, and it is accepted only where it is one constant, spread at most
1.0 s: no single shift fixes a free-running clock. A member whose lag cannot be measured or is not one
constant is kept at lag 0 with the reason recorded, because these are GPS-disciplined loggers.

A member passes four steps between its cache and the sum, in this order. Its own transient chunks are NaN.
Then `references.EDGE_S` = 120 s inside every record end and every gap end is NaN, at every rate: the samples
either side of a break carry the logger's settling, thousands of nT at a PR6-24. Then the lag is applied as a
Lanczos-windowed sinc of `align.LANCZOS_A` = 16 taps either side, which is local: one impulsive sample stays
where it is, a NaN reaches only 32 samples, and the error against an analytic sine shifted 0.37 s is 0.3 per
cent of the amplitude at 2.3 s, the shortest period the 1 Hz band file estimates.

Then `references.member_screen` compares the members with each other. Where three or more are finite, a
sample whose first difference departs from the median over them by more than max(`SCREEN_K` = 30 x MAD,
`SCREEN_FLOOR_NT` = 3.0 nT) is flagged; a spike the whole fleet sees is the field, is carried by the median,
and is not flagged. Where exactly two are finite the median of two first differences does not name the
culprit, so a sample is flagged where one member's own first difference passes that bar and the other's is
under 1.5 nT, and where both pass it neither is flagged. The flagged span, from two samples before to three
after and merged with its neighbours, is bridged by a straight line inside that member where the span is at
most 12 s.

Then `references.level_match` sets each member on the fleet's own level: its offset from the median of the
members finite at each sample is taken in two robust steps -- the median of each `LEVEL_BLOCK_S` = 60 s
block, then the running median of 60 of those blocks, `LEVEL_WIN_S` = 3600 s -- interpolated back to the
sample grid and subtracted, so a member entering or leaving the sum does not step the level.

A sample where a member is NaN does not add to that member's weight there, so the mean is over whoever is
sound and the weights renormalise per sample. A sample carried by fewer than `STACK_MIN` site members is zero
and its mask False, on the same rule the member list is judged on.

**This check fails if this site's stack carries a member weighted by its coherence with the target rather
than the fleet, or a member outside the pool.** The first limb is scored by recomputing the weights from the
fleet coherence table printed above and comparing them with the weights in this site's own sidecar, and by
comparing the same weights against the target coherences of the candidate table: a weight that matches the
target column and not the fleet column is the wrong rule."""),

("code", '''weights, pairs = ST.fleet_weights()
fleet = pd.DataFrame([dict(member=k, fleet_coh_100_1000s=v,
                           pairs=sum(1 for p in pairs if k in p.split(":")))
                      for k, v in sorted(weights.items())])
print("the fleet coherence of every pool member at 100-1000 s, the median over its pairs with the rest")
print(fleet.to_string(index=False))

kept, lags, refused, notes = ST.stack_members(SITE)
mem = pd.DataFrame([dict(site=SITE, member=d, weight=w, lag_s=lags.get(d, 0.0), note=notes.get(d, ""),
                         in_pool=d in pool) for d, w in kept.items()]
                   + [dict(site=SITE, member=d, weight=None, lag_s=None, note=why, in_pool=d in pool)
                      for d, why in sorted(refused.items())])
print()
print("%s: %d member(s), and every refusal with its reason" % (SITE, len(kept)))
print(mem.to_string(index=False))
if len(kept) < STACK_MIN:
    print("the stack is refused for want of members: %d < %d" % (len(kept), STACK_MIN))

# limb A, recomputed: the weight of every member has to be the fleet value and not the target value
cand = scores.set_index("name")
recomputed = {d: weights.get(d) for d in kept}
target_coh = {d: (float(cand.loc[d, "coh"]) if d in cand.index and cand.loc[d, "coh"] is not None else None)
              for d in kept}
print()
print("the weight in the store against the fleet table and against this site's own target coherence")
print(pd.DataFrame([dict(member=d, in_store=kept[d], fleet_table=recomputed[d], target_coh=target_coh[d])
                    for d in kept]).to_string(index=False))
wrong_rule = [d for d in kept if recomputed[d] is None or abs(kept[d] - recomputed[d]) > 1e-6]
looks_like_target = [d for d in kept if target_coh[d] is not None and abs(kept[d] - target_coh[d]) < 1e-6
                     and (recomputed[d] is None or abs(target_coh[d] - recomputed[d]) > 1e-6)]
outside = sorted(set(mem.member[(mem.weight.notna()) & (~mem.in_pool)]))
gaps = [abs(kept[d] - target_coh[d]) for d in kept if target_coh[d] is not None]
print()
if not len(kept):
    print("VERDICT: UNJUDGED -- %s's stack carries no member, so no weight was scored" % SITE)
elif wrong_rule or looks_like_target or outside:
    print("VERDICT: FAIL -- %d of %s's %d member(s) do not carry the fleet weight (%s); %d carry the target "
          "coherence instead (%s); %d sit outside the clean pool (%s)"
          % (len(wrong_rule), SITE, len(kept), ", ".join(wrong_rule) or "none",
             len(looks_like_target), ", ".join(looks_like_target) or "none",
             len(outside), ", ".join(outside) or "none"))
else:
    print("VERDICT: PASS -- all %d of %s's member(s) carry the fleet coherence recomputed from the table "
          "above to within 1e-6 and none carries its target coherence (which differs by %.3f to %.3f), and "
          "every one is in the %d-site clean pool"
          % (len(kept), SITE, min(gaps) if gaps else float("nan"), max(gaps) if gaps else float("nan"),
             len(pool)))
'''),

("md", r"""The weights and what they produce. On the left, look at how far the kept members stand above
STACK_CUTOFF: a stack whose members all sit just over the line is a stack of marginal references, and raising
the cutoff would refuse it outright. The lag written on each bar is what the alignment measured; on
GPS-disciplined loggers these are fractions of a second and a member whose lag could not be measured is kept
at zero with the reason in the table above.

On the right, the stack is drawn over the target and two of its members through six hours of one quiet day,
every trace demeaned, with a mark under the traces wherever the spike screen bridged a member sample. The
stack follows the same field as the target; whether it is rougher or smoother than its members is the
number the check below scores."""),

("code", '''t0s, n_s = ST.window(SITE)
_t, h_show, _ang, _n = ST.rotated(SITE)
members_drawn = list(kept)[:2]
try:
    _rt, rh_stack, _rm, stack_info = REF.load_reference("stack", SITE, RATES1[0], WORK)
except FileNotFoundError:
    rh_stack, stack_info = None, {}
grids = [np.asarray(h_show["Hx"], float)]
labels = ["%s (the target)" % SITE]
colours = ["C3"]
for m in members_drawn:
    grids.append(np.asarray(ST._on_grid(m, t0s, n_s, drop_events=True, lag_s=lags.get(m))["Hx"], float))
    labels.append("%s (member, weight %.2f)" % (m, kept[m]))
    colours.append("C0" if len(colours) == 1 else "C1")
if rh_stack is not None:
    grids.append(np.asarray(rh_stack["Hx"], float))
    labels.append("the stack")
    colours.append("C2")
i0 = quiet_window(grids, fs=float(RATES1[0]), hours=6.0)
w = int(6 * 3600 * RATES1[0])
series = ([] if i0 is None else [(lb, g[i0:i0 + w], c) for lb, g, c in zip(labels, grids, colours)])
# the marks under the traces are the spans the spike screen bridged, read from the store's own npz and cut
# to the drawn window; Hx is the channel drawn, so Hx is the channel marked
scr_show = stack_info.get("screen") or {}
bridged = ([] if i0 is None else
           [(max(int(a) - i0, 0), min(int(b) - i0, w))
            for a, b in REF.bridged_spans("stack", SITE, RATES1[0], WORK, "Hx")
            if int(b) > i0 and int(a) < i0 + w])
show(FIGP.stack_weights(kept, refused, lags, SITE, SITE_DIR / "03_stack.png", series=series,
                        t_start=(None if i0 is None else t0s + i0 / RATES1[0]),
                        cutoff=STACK_CUTOFF, n_max=STACK_MAX, n_min=STACK_MIN, fs=float(RATES1[0]),
                        bridged=bridged, steps=(scr_show.get("steps_per_million") or {}),
                        screen_k=REF.SCREEN_K, screen_floor_nt=REF.SCREEN_FLOOR_NT, edge_s=REF.EDGE_S,
                        step_nt=REF.STEP_NT, level_win_s=REF.LEVEL_WIN_S))
del grids, rh_stack
'''),

("md", r"""**This check fails if either of this site's stack channels carries a rate of steps > 2 nT per
million samples above the median rate of its own screened members.** A step of more than
`references.STEP_NT` = 2.0 nT from one sample to the next is not the field on a quiet day at these periods:
it is a logger spike or a settling sample at the edge of a record. The rate is counted over each channel's
finite samples in order, the same way on a member and on the stack built from it, so a screen that leaves
holes is charged for them.

The yardstick is the members as they enter the sum, screened and set on the fleet's level, and not as they
came off the logger. A stack is a combination of what goes into it and has to be at least as clean as its
median input; the raw member rate moves with whatever the neighbours' own noise happens to be and measures
the wrong thing. Both medians are in the sidecar, raw and screened, beside the stack's own rate, and this
cell reads them and recomputes nothing.

Both stack kinds are scored on the same yardstick. The stack + observatory is a combination of the same site
members with one more, and the observatory is set on the members' established level one-sidedly -- it never
enters the median it is matched against -- so its composite answers to the site members' median as the fleet
stack does."""),

("code", '''# the numbers come from each sidecar; nothing here recomputes a reference
step_rows = []
for kind in ("stack", "stack_obs"):
    p = ST.path(kind, SITE).with_suffix(".json")
    if not p.exists():
        continue
    scr = json.loads(p.read_text(encoding="utf-8")).get("screen") or {}
    sp = scr.get("steps_per_million") or {}
    members = scr.get("members") or {}
    obs = scr.get("observatory") or {}
    for c in ("Hx", "Hy"):
        a = (sp.get("stack") or {}).get(c)
        b = (sp.get("members_median_screened") or {}).get(c)
        if a is None or b is None:
            continue
        per = [v.get(c) or {} for v in members.values()]
        off = [float(v.get("offset_rms_nt") or 0.0) for v in per]
        off += [float((v.get(c) or {}).get("offset_rms_nt") or 0.0) for v in obs.values()]
        step_rows.append(dict(site=SITE, kind=kind, channel=c, stack_steps=float(a),
                              members_screened=float(b),
                              members_raw=(sp.get("members_median_raw") or {}).get(c),
                              margin=round(float(b) - float(a), 1),
                              flagged=sum(int(v.get("flagged") or 0) for v in per),
                              bridged=sum(int(v.get("bridged") or 0) for v in per),
                              nan_spans=sum(int(v.get("nan_spans") or 0) for v in per),
                              offset_rms_nt=round(max(off or [0.0]), 2),
                              on_floor=sum(1 for v in per if "floor" in str(v.get("threshold_from")))))
sp_tab = pd.DataFrame(step_rows)
if len(sp_tab):
    print("steps above %g nT per million samples: each stack kind against the median of its screened site "
          "members, with the raw median beside it and what the screen and the level match did" % REF.STEP_NT)
    print(sp_tab.to_string(index=False))
    print()
    print("on_floor counts the members whose threshold was set by SCREEN_FLOOR_NT = %g nT rather than by "
          "SCREEN_K = %g x MAD; offset_rms_nt is the largest offset the level match removed, the "
          "observatory's included where the kind carries one" % (REF.SCREEN_FLOOR_NT, REF.SCREEN_K))
rough = sp_tab[sp_tab.stack_steps > sp_tab.members_screened] if len(sp_tab) else sp_tab
print()
if not len(sp_tab):
    print("VERDICT: UNJUDGED -- no stack sidecar of %s carries the statistic, so nothing was scored; the "
          "store section below writes the sidecars and this cell reads them on a re-run" % SITE)
elif len(rough):
    worst = rough.loc[rough.margin.idxmin()]
    print("VERDICT: FAIL -- %d of %d stack-channel(s) of %s are rougher than the median of their own "
          "screened members. The worst is %s %s at %.0f steps above %g nT per million samples against a "
          "screened members' median of %.0f"
          % (len(rough), len(sp_tab), SITE, worst.kind, worst.channel, worst.stack_steps, REF.STEP_NT,
             worst.members_screened))
else:
    worst = sp_tab.loc[sp_tab.margin.idxmin()]
    print("VERDICT: PASS -- all %d stack-channel(s) of %s over %d kind(s) are at least as smooth as the "
          "median of their own screened members. The narrowest margin is %s %s at %.0f steps above %g nT "
          "per million samples against a screened members' median of %.0f, and the screen flagged %d "
          "sample(s), bridged %d span(s) and left %d NaN"
          % (len(sp_tab), SITE, sp_tab.kind.nunique(), worst.kind, worst.channel, worst.stack_steps,
             REF.STEP_NT, worst.members_screened, int(sp_tab.flagged.sum()), int(sp_tab.bridged.sum()),
             int(sp_tab.nan_spans.sum())))
'''),

("md", r"""## The observatory

The INTERMAGNET one-second record of the survey's own observatory, read out of the archive over this site's
window and turned into the mean-field frame of that window like any other member. It is never shifted: a
delay on a single reference cancels exactly in Z, so shifting it would change nothing and leaving it alone
keeps that invariance available as a check.

Its weight in the stack + observatory is measured on the same rule as a site member's -- the median coherence
with the pool at 100-1000 s -- so it enters at a weight on the members' own scale rather than at its
agreement with any one target.

The coherence of this site with the observatory is a reading. It says how much of the site's long-period
field the observatory shares from several hundred kilometres away, which is what the observatory kind can buy
and what it cannot: at the short end a distant observatory shares almost nothing. The bars beside it are
every pool site's, so this site's number can be read against the survey's.

On the right the two Hx records are drawn over each other through six hours of one quiet day: the large
excursions are shared and the small ones are not, which is what the observatory kind rests on."""),

("code", '''w_obs, obs_pairs = ST.observatory_weight(OBS)
print("observatory %s, %s" % (OBS, geo.OBSERVATORIES.get(OBS, ("", 0, 0))[0]))
print("its fleet weight at 100-1000 s is %s, the median over %d pool member(s)"
      % (w_obs, sum(1 for v in obs_pairs.values() if v is not None)))
obs_rows = []
for s in ALL_SITES:
    c = obs_pairs.get(s)
    chunks = None
    if c is None and s == SITE:
        # this site is reported whether or not it is in the pool, so the reading covers what is processed
        r_obs = ST.site_observatory_coh(s, OBS)
        c = None if not np.isfinite(r_obs["coh"]) else round(float(r_obs["coh"]), 4)
        chunks = r_obs["chunks"]
    obs_rows.append(dict(site=s, km=round(geo.distance_km(ST.position(s), OBS), 1), coh_100_1000s=c,
                         in_pool=s in pool, chunks=chunks,
                         built=ST.path("obs", s).with_suffix(".json").exists()))
ob = pd.DataFrame(obs_rows)
print()
print("%s: %.1f km from %s, coherence at 100-1000 s %s"
      % (SITE, float(ob.km[ob.site == SITE].iloc[0]), OBS, ob.coh_100_1000s[ob.site == SITE].iloc[0]))
print("stack + observatory at %s: %s" % (SITE, " ".join(list(kept) + ([OBS] if w_obs else []))))

try:
    _ot, oh, _om, _oi = REF.load_reference("obs", SITE, RATES1[0], WORK)
except FileNotFoundError:
    oh = None
obs_series, j0 = [], None
w = int(6 * 3600 * RATES1[0])
if oh is not None:
    pair = [np.asarray(h_show["Hx"], float), np.asarray(oh["Hx"], float)]
    j0 = quiet_window(pair, fs=float(RATES1[0]), hours=6.0)
    if j0 is not None:
        obs_series = [("%s Hx" % SITE, pair[0][j0:j0 + w], "C3"), ("%s Hx" % OBS, pair[1][j0:j0 + w], "C0")]
show(FIGP.observatory_bars(ob, OUT / "03_observatory.png", series=obs_series,
                           t_start=(None if j0 is None else t0s + j0 / RATES1[0]),
                           code=OBS, site=SITE, fs=float(RATES1[0])))
del oh
'''),

("md", r"""## The references written to the store

Every reference is written once per rate to `<work_root>/references/<rate>hz/<kind>_<site>.npz`, with a
sidecar naming its members, their weights and lags, the refusals with their reasons, the frame and the time
it was built. A pass reads the store; it never rebuilds a reference of its own, so two transfer functions of
the same kind are built on the same array.

A 10 Hz store re-reads the 1 Hz specification -- the same pool, the same remote, the same members, weights
and lags -- on the 10 Hz grid after a spike screen. The decisions are made where they can be measured."""),

("code", '''built = {}
for rate in RATES:
    t = time.time()
    # the observatory is not one of the 10 Hz kinds, so no observatory reference is built on that grid
    kinds = tuple(k for k in (KINDS if int(rate) == 1 else KINDS10) if k in REF.KINDS_WITH_STORE)
    st = ST if int(rate) == RATES[0] else store(rate)
    spec = None if int(rate) == 1 else store(1)     # <- the 10 Hz store takes its decisions from the 1 Hz one
    got = REF.build_store(sv, [SITE], rate=rate, kinds=kinds, verbose=False, spec_store=spec, store=st)
    built[rate] = got
    print("%d Hz: %d reference(s), %.1f s" % (rate, sum(len(v) for v in got.values()), time.time() - t))
    print(pd.DataFrame([dict(site=s, kind=k, coverage=info.get("coverage"),
                             members=" ".join(m["name"] for m in (info.get("members") or [])
                                              if m.get("role") != "refused"),
                             error=str(info.get("error", ""))[:90])
                        for s, d in sorted(got.items())
                        for k, info in sorted(d.items())]).to_string(index=False))
'''),

("md", r"""## The bands and the parameter set

The band file's lines are FFT harmonics of the **window**, not of the record, so a file read at another
window length names different periods and nothing says so. File, level count and window are therefore one
object in the package, `aurora_run.BANDS`, and the tables below are read through Aurora's own band machinery
rather than off the text file.

The cascade is `[1] + [4] * (levels - 1)`: the rate falls by four at each level after the first.

Under it, what the transient mask and the hour floor cost this site. Aurora gets one run per kept stretch of
at least 3,600 s; a shorter stretch holds no complete window at the deep levels and is dropped. Two events
forty minutes apart therefore cost the forty minutes between them as well as themselves, and that second loss
is measured here rather than hidden inside the mask."""),

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

show(FIGP.band_ladder(tabs, {k: bs.file.name for k, bs in aurora_run.BANDS.items()},
                      OUT / "03_bands.png", params=PARAMS))

print()
print("the Aurora parameter sets the package ships; %s is this run's" % PARAMS)
for name, p in sorted(aurora_run.AURORA_PARAMS.items()):
    print("   %-14s %s" % (name, ", ".join("%s=%s" % kv for kv in p.items())))
print()
print("what the mask and the %g s run floor cost %s at %g Hz" % (TR.MIN_SEGMENT_S, SITE, RATES1[0]))
arrm, ANGLE_M = FR.rotate_to_mean_field({c: np.asarray(ARRAYS[c], float).copy() for c in ARRAYS},
                                        regimes=FR.parse_regimes(dec.get("rot_regimes")),
                                        drop=FR.parse_regimes(dec.get("rot_drop")), fs=float(RATES1[0]))
ev_site = TR.site_events(SITE, ALL_SITES, WORK, sv.cfg)
ev_e = TR.e_events(SITE, WORK, sv.cfg)
keep_show, stats = TR.build_keep(T0_SITE, {c: arrm[c] for c in mth5_build.LOCAL_CHANNELS},
                                 float(RATES1[0]), ev_site, (), ev_e)
runs = TR.segments(keep_show, int(TR.MIN_SEGMENT_S * RATES1[0]))
print(pd.DataFrame([dict(site=SITE, days=round(stats["n"] / 86400.0 / RATES1[0], 2),
                         finite_frac=stats["finite_frac"], H_intervals=len(ev_site), E_bursts=len(ev_e),
                         kept_frac=stats["kept_frac"], runs=len(runs),
                         floor_loss_frac=round(TR.floor_dropped_frac(keep_show, float(RATES1[0])), 5),
                         longest_run_d=round(max((L for _o, L in runs), default=0) / 86400.0 / RATES1[0],
                                             2))]).to_string(index=False))
'''),

("md", r"""The mask over the record. Look at where the masked spans fall and at the row of runs underneath: a
run is green where the stretch is at least MIN_SEGMENT_S and became one Aurora run, and red where the floor
threw it away. Two events an hour apart cost the hour between them as well as themselves, and a record
carrying many short events loses far more than the sum of the events. That second loss is what the red bars
show and it is the reason to look at the figure rather than at the kept fraction alone."""),

("code", '''show(FIGP.mask_record(T0_SITE, arrm, keep_show, SITE_DIR / "03_mask.png", site=SITE,
                      spans=[("H transient", "C3", ev_site), ("E burst", "C1", ev_e)],
                      fs=float(RATES1[0]), min_segment_s=TR.MIN_SEGMENT_S))
'''),

("md", r"""## The MTH5

One kind, written and read back. The MTH5 is what Aurora reads: the local station with its five channels and,
where there is a reference, a second station carrying hx and hy only on the same grid and with the same run
ids, so the kernel dataset's run-interval intersection pairs them one to one.

The channels carry nanoTesla and milliVolt per kilometer and no filter at all. The cache is already in
physical units, so a filter here would be applied a second time; the spelling of the units matters too,
because `millivolts per kilometer` resolves to `unknown per kilometer` in mt_metadata and writes the channel
with no unit.

**This check fails if the MTH5 differs from the cache sample for sample, or if its run count differs from the
kept stretches.** Every sample of every channel of every run is compared with the array it was written from.
The bound is 1e-3 nT and mV/km, which is the file's own float32 resolution at these levels."""),

("code", '''probe_kind = "remote" if "remote" in KINDS else KINDS[0]
rate = RATES1[0]
local = {c: np.asarray(arrm[c], float) for c in mth5_build.LOCAL_CHANNELS}
rt0, rh, rmask, info = REF.load_reference(probe_kind, SITE, rate, WORK)
ev_rem = [(pd.Timestamp(a).timestamp(), pd.Timestamp(b).timestamp())
          for a, b in (info.get("remote_events") or [])] if probe_kind == "remote" else []
keep, stats = TR.build_keep(T0_SITE, local, float(rate), ev_site, ev_rem, ev_e,
                            remote_mask=(None if rmask is None else rmask))
rid = REF.reference_station_id(probe_kind, info, OBS)
scratch = SITE_DIR / "_mth5_check"
h5 = scratch / ("%s_%s.h5" % (SITE, probe_kind))
t = time.time()
_p, segs = mth5_build.write_h5(h5, SITE, local, (None if rh is None else (rid, rh)), T0_SITE, float(rate),
                               sv.cfg["name"], row, keep=keep,
                               reference_row=(sv.site(info["remote"]) if probe_kind == "remote" else None))
print("%s %s: %s written in %.1f s, %.1f MB" % (SITE, probe_kind, h5.name, time.time() - t,
                                                h5.stat().st_size / 2 ** 20))
print("   stations  %s and %s" % (SITE, rid or "none"))
print("   mask      keeps %.2f %% of %d samples; %d runs of at least %g s; the hour floor loses %.2f %% more"
      % (100 * stats["kept_frac"], stats["n"], len(segs), TR.MIN_SEGMENT_S,
         100 * TR.floor_dropped_frac(keep, float(rate))))
print("   runs      %s" % ", ".join("%03d %.2f d" % (i + 1, L / 86400.0 / rate)
                                    for i, (_o, L) in enumerate(segs[:12])))
back = mth5_build.read_back(h5, SITE, local, segs, sv.cfg["name"])
print("   read back %s" % back)
n_kept_runs = len(TR.segments(keep, int(TR.MIN_SEGMENT_S * rate)))
left = mth5_build.remove(h5)          # a handle HDF5 has not released is said, not raised
if left:
    print("   %s" % left)
import shutil; shutil.rmtree(scratch, ignore_errors=True)
del local, rh, arrm, ARRAYS

print()
if not back["samples_compared"]:
    print("VERDICT: UNJUDGED -- the MTH5 carried no sample to compare")
elif back["problems"] or back["runs_in_file"] != n_kept_runs:
    print("VERDICT: FAIL -- %d channel-run(s) differ from the cache (%s); the file holds %d run(s) where the "
          "mask leaves %d kept stretch(es) of at least %g s"
          % (len(back["problems"]), "; ".join(back["problems"][:6]), back["runs_in_file"], n_kept_runs,
             TR.MIN_SEGMENT_S))
else:
    print("VERDICT: PASS -- %s's %s MTH5 carries all %d samples of its 5 channels over %d run(s) to within "
          "%.3g nT and mV/km of the decided and rotated cache, and its run count is the %d kept stretch(es) "
          "of at least %g s the mask leaves"
          % (SITE, probe_kind, back["samples_compared"], back["runs_in_file"], back["worst_difference"],
             n_kept_runs, TR.MIN_SEGMENT_S))
'''),

("md", r"""## The run

One subprocess, running `python -m auslamp_proc.process.run` over this site's kinds at 1 Hz. The lane pins
its BLAS threads to three: a lane that takes every core makes concurrent lanes slower than one, and the
memory a pass peaks at is per lane. Over AusLAMP Queensland Phase 1, whose records run 12-62 days, a 1 Hz
pass peaks at 2.5-4.8 GB and takes 140-900 s.

The same lane is what the batch entry runs, one per site:

    python -m auslamp_proc.process.batch --survey queensland_phase1 --sites all --lanes 3

The CLI is resumable by design. With `REDO` False a transfer function whose EDI is already on disk is left
alone and reported as `exists`, so a run stopped part way is finished by re-running this cell. A pass that
fails is caught, its error goes into its own ledger row, and the next one runs.

Each pass writes `<SITE>_<kind>_<rate>hz_<params>.edi` and `.xml` into `<SITE>/<RUN>_<stamp>/`, beside
`log.txt` and `provenance.json`, and appends one row to `<work_root>/survey/runs.csv`.

**This check fails if any requested transfer function is missing or FAILED in the ledger, or if any of them
lacks a tipper, or if provenance.json lacks the decisions.csv row it used.** The tipper is the third limb
because it comes out of the same pass as the impedance -- hz is in the local station -- so a file without one
was run on a record missing its vertical channel."""),

("code", '''t_run = time.time()
print("%s: %d kind(s) at %s Hz -> %s_%s"
      % (SITE, len(KINDS), ", ".join(str(r) for r in RATES1), RUN, STAMP), flush=True)
res = BATCH.lane(SITE, SURVEY, RUN, STAMP, KINDS, RATES1, PARAMS, redo=REDO,
                 work_root=(str(WORK_ROOT) if WORK_ROOT else ""), repo=REPO)
wall = time.time() - t_run
print("exit %d in %7.1f s  %s%s" % (res["returncode"], res["seconds"], res["stdout"],
                                    ("  || " + res["stderr"]) if res["returncode"] else ""))
print("wall time %.1f min; a resumed run finds its files on disk and this is the time to check them, not "
      "the time they cost" % (wall / 60.0))

ledger = pd.read_csv(WORK / "survey" / "runs.csv")
led = ledger[(ledger["run"].astype(str) == RUN) & (ledger["stamp"].astype(str) == STAMP)
             & (ledger.site == SITE) & (ledger.kind.isin(KINDS))].copy()
# runs.csv is append-only in both directions: a kind this package no longer builds keeps its old rows, and
# re-running this workbook adds an `exists` row beside the `made` row of the same pass. The row that says
# what a pass cost is the one written when it was made, so that row wins and one row per pass is kept.
led["_made"] = (led.status == "made").astype(int)
led = (led.sort_values(["kind", "rate_hz", "_made"])
       .drop_duplicates(["kind", "rate_hz"], keep="last").drop(columns="_made"))
print()
print(led[["site", "kind", "rate_hz", "params", "remote", "n_runs", "mask_dropped_frac",
           "floor_dropped_frac", "seconds", "peak_rss_mb", "status", "error"]].to_string(index=False))

want = [(SITE, k, float(r)) for k in KINDS for r in RATES1]
have = {(r.site, r.kind, float(r.rate_hz)): r for r in led.itertuples()}
missing = [w for w in want if w not in have]
failed = led[led.status == "FAILED"]
no_tipper = []
for w in want:
    r = have.get(w)
    if r is None or str(r.status) == "FAILED":
        continue
    p = Path(str(r.edi))
    if not p.exists():
        missing.append(w)
    elif not EDI.has_tipper(p):
        no_tipper.append("%s %s %g Hz" % w)
prov = PROV.read(WORK / SITE / ("%s_%s" % (RUN, STAMP)) / "provenance.json")
no_prov = not prov or not prov.get("decisions_row") or not prov["decisions_row"].get("site")
print()
if not want:
    print("VERDICT: UNJUDGED -- no transfer function was requested")
elif missing or len(failed) or no_tipper or no_prov:
    print("VERDICT: FAIL -- %d of %d requested transfer function(s) are missing (%s); %d are FAILED (%s); "
          "%d carry no tipper (%s); provenance.json %s the decisions.csv row"
          % (len(missing), len(want), "; ".join("%s %s %g Hz" % m for m in missing[:8]) or "none",
             len(failed), "; ".join("%s: %s" % (r.kind, str(r.error)[:80])
                                    for r in failed.itertuples()) or "none",
             len(no_tipper), "; ".join(no_tipper[:8]) or "none",
             "lacks" if no_prov else "names"))
else:
    print("VERDICT: PASS -- all %d requested transfer function(s) exist and none is FAILED (%d made, %d "
          "already on disk), every one carries a tipper, and provenance.json names the decisions.csv row it "
          "used; this execution took %.1f min and the passes cost %.1f machine-minute(s) between them, the "
          "largest peak being %.0f MB"
          % (len(want), int((led.status == "made").sum()), int((led.status == "exists").sum()),
             wall / 60.0, led.seconds.sum() / 60.0, led.peak_rss_mb.max()))
'''),

("md", r"""## The 10 Hz stretch

A 10 Hz pass is run on a stretch of hours and never on the whole record. The 10 Hz row is the short end,
below the 16 s join, and no deep level is delivered from it, so a stretch costs it nothing: a single Aurora
window at the deepest level of the 1 Hz band file is 65,536 s, which no stretch of hours could hold, while
the 10 Hz band file's own levels sit inside an hour.

One rule chooses the hours, and this workbook and workbook 04 both use it
(`auslamp_proc.process.selection`):

- each whole UTC hour of the 1 Hz cache is scored by the median squared coherence over 20-200 s, Welch at
  1,024 s segments, of Ex with Hy and of Ey with Hx, on the decided record;
- a stretch is a contiguous run of hours in which both lines read above `WINDOW_COH` = 0.5;
- the selection is the longest stretch, and where it runs longer than `STRETCH_MAX_H` = 48 h the 48
  contiguous hours inside it with the highest mean score are taken;
- the control is a stretch of the same length placed at random elsewhere in the record under `SEED`, not
  overlapping the selection.

The control is what makes the selection testable: a stretch that does not beat a stretch of the same length
placed anywhere bought efficiency, not a different answer. The two tags are `stretch` and `control`, and the
score series is written to `<work_root>/<SITE>/hour_scores.csv` with the chosen bounds in
`hour_selection.json`.

A kept hour is 3,600 s, which at 10 Hz is 36,000 samples and exactly the run floor, so an isolated kept hour
survives as one Aurora run and adjacent kept hours merge into one longer run. A kept hour the transient mask
or a gap cuts into falls under the floor and is dropped, which each pass's `floor_dropped_frac` reports."""),

("code", '''t_sel = time.time()
SCORES = SEL.site_scores(sv, SITE, rate=SEL.SCORE_RATE)
SELECTION = SEL.build(sv, SITE, coh_min=WINDOW_COH, max_h=STRETCH_MAX_H, seed=SEED)
print("%s: %d whole UTC hour(s) scored in %.1f s" % (SITE, len(SCORES), time.time() - t_sel))
print(SEL.summary_rows(WORK, [SITE]).to_string(index=False))
print()
for tag, item in sorted(SELECTION.items()):
    print("%-8s %s" % (tag, item["reason"]))
    if item.get("refused"):
        print("         REFUSED before any pass: %s" % item["refused"])
    elif item.get("n_runs_kept") is not None:
        print("         the mask leaves %d run(s) of at least %g s inside it, %s s in all"
              % (item["n_runs_kept"], TR.MIN_SEGMENT_S, item.get("kept_seconds")))
print()
print("the score is the median %g-%g s squared coherence of Ex with Hy and of Ey with Hx over each whole UTC "
      "hour of the 1 Hz cache, Welch at %d samples; a kept hour is 3600 s = 36000 samples at 10 Hz, which is "
      "the run floor" % (SEL.SCORE_BAND_S[0], SEL.SCORE_BAND_S[1], SEL.nperseg_for(SEL.SCORE_RATE)))
HAS10 = cached(SITE, 10)
if not HAS10:
    print("no 10 Hz cache at %s -- run workbook 02 at 10 Hz over this site to bring it into this lane" % SITE)
REFUSED10 = {t_: i["refused"] for t_, i in SELECTION.items() if i.get("refused")}
if SELECTION.get(SEL.STRETCH, {}).get("t_start") is None:
    print("no hour reads above %.2f on both lines, so %s carries no 10 Hz transfer function"
          % (WINDOW_COH, SITE))
elif REFUSED10:
    print("the rule refuses %d of the %d stretch(es) at %s before any pass; a stretch is refused on what "
          "its mask leaves and never on its length" % (len(REFUSED10), len(SELECTION), SITE))
'''),

("md", r"""### The run at 10 Hz

One subprocess again, over the two kinds and the two tags: four transfer functions. The observatory is not a
kind at 10 Hz. A stretch of at most 48 h of 10 Hz peaks at a few GB where a whole-record pass peaks at
22-37 GB, which is why the 10 Hz lane runs on a stretch at all.

The names carry the tag between the rate and the parameter set, `<SITE>_<kind>_10hz_<tag>_<params>.edi`; the
ledger's `selection` column reads `stretch`, `control` or `whole`, the EDI carries a `selection=` line, and
`provenance.json` carries the stretch's start, end, hours, threshold and seed.

A workbook run whose `RATES` holds no 10 Hz asks for nothing here. The hours are still scored and the stretch
still chosen above, because the score is a property of the record, and the check below scores them."""),

("code", '''TAGS10 = [t for t in SEL.TAGS if t in SELECTION and SELECTION[t]["n_hours"] > 0]
l10 = pd.DataFrame(columns=["site", "kind", "rate_hz", "selection", "n_runs", "mask_dropped_frac",
                            "floor_dropped_frac", "seconds", "peak_rss_mb", "status", "error", "edi"])
wall10 = 0.0
if not RATES10 or not TAGS10 or not HAS10:
    print("RATES = %s, %d stretch(es) carry an hour and the 10 Hz cache is %s, so no 10 Hz transfer function "
          "is asked for here" % (RATES, len(TAGS10), "present" if HAS10 else "absent"))
else:
    t_10 = time.time()
    print("%s: %d kind(s) over %s at 10 Hz -> %s_%s"
          % (SITE, len(KINDS10), ", ".join(TAGS10), RUN10, STAMP10), flush=True)
    r10 = BATCH.lane(SITE, SURVEY, RUN10, STAMP10, KINDS10, RATES10, PARAMS, selections=TAGS10, redo=REDO,
                     work_root=(str(WORK_ROOT) if WORK_ROOT else ""), repo=REPO)
    wall10 = time.time() - t_10
    print("exit %d in %7.1f s  %s%s" % (r10["returncode"], r10["seconds"], r10["stdout"],
                                        ("  || " + r10["stderr"]) if r10["returncode"] else ""))
    print("wall time %.1f min" % (wall10 / 60.0))
    ledger10 = pd.read_csv(WORK / "survey" / "runs.csv")
    l10 = ledger10[(ledger10["run"].astype(str) == RUN10) & (ledger10["stamp"].astype(str) == STAMP10)
                   & (ledger10.site == SITE) & (ledger10.kind.isin(KINDS10))].copy()
    l10["selection"] = l10.selection.astype(str)
    l10["_made"] = (l10.status == "made").astype(int)
    l10 = (l10.sort_values(["kind", "selection", "_made"])
           .drop_duplicates(["kind", "rate_hz", "selection"], keep="last").drop(columns="_made"))
    print()
    print(l10[["site", "kind", "rate_hz", "selection", "n_runs", "mask_dropped_frac", "floor_dropped_frac",
               "seconds", "peak_rss_mb", "status", "error"]].to_string(index=False))
'''),

("md", r"""### What the 10 Hz row reads against the 1 Hz row

The departure a 10 Hz row carries against its own 1 Hz row is a property of this survey's instrument, cache
and band file, so it is measured here and not quoted from elsewhere. It is measured on the whole-record 10 Hz
passes, because those are the only ones that differ from the 1 Hz row in rate alone; a stretch of hours
differs in which hours it holds as well. The 1 Hz row is interpolated in log period and log rho onto the
10 Hz periods inside 4-32 s, and a 10 Hz period the 1 Hz row does not reach is dropped rather than compared
against its end point. The sentence this cell prints is the one every 10 Hz file of this survey carries.

Under it, the same ratio for the stretch and for its control, which is a reading and not a check. A site
whose stretch and control depart by the same amount is not departing because of the choosing; workbook 05's
rate gate is what refuses such a site, on the site's own numbers."""),

("code", '''RATE_KIND = "remote"            # <- the kind the rate is measured on; the delivered kind at most sites
whole_led = TFN.ledger(WORK)
pairs, one_hz = [], {}
for s in ALL_SITES:
    a = whole_led[(whole_led.site == s) & (whole_led.kind == RATE_KIND) & (whole_led.rate_hz == 1.0)]
    if not len(a) or not Path(str(a.iloc[0].edi)).exists():
        continue
    one_hz[s] = TFN.read_tf(Path(str(a.iloc[0].edi)))
    for d in sorted((WORK / s).glob("*_*")):
        p = d / ("%s_%s_10hz_%s.edi" % (s, RATE_KIND, PARAMS))      # the whole record carries no tag
        if p.exists():
            pairs.append((s, TFN.read_tf(p), one_hz[s]))
            break
rate_rec = RATE.measure(pairs, RATE_KIND)
RATE.write_record(WORK, rate_rec)
print("the whole-record 10 Hz %s pass(es) of %d site(s), over %g-%g s"
      % (RATE_KIND, rate_rec["n_sites"], rate_rec["band_s"][0], rate_rec["band_s"][1]))
print(pd.DataFrame([dict(site=k, xy_pct=(None if v["xy"] is None else round(100 * (v["xy"] - 1), 1)),
                         yx_pct=(None if v["yx"] is None else round(100 * (v["yx"] - 1), 1)))
                    for k, v in sorted(rate_rec["per_site"].items())]).to_string(index=False))
print()
CAVEAT = RATE.caveat(rate_rec)                  # <- the sentence every 10 Hz file of this survey carries
print("caveat_10hz=%s" % CAVEAT)

# A file written before this survey was measured carries the sentence of its own day, and remaking a
# whole-record 10 Hz pass to change a sentence costs 22-38 GB and an hour a site. The one header line is
# rewritten in place instead: on the file's bytes, verified after the write to differ from the original in
# that line alone, and recorded in the run folder's provenance.
rw = RATE.rewrite_all(WORK, CAVEAT)
for d in sorted({Path(r["path"]).parent for r in rw if r["changed"]}):
    RATE.record_rewrite(d, [r for r in rw if Path(r["path"]).parent == d])
done_rw = [r for r in rw if r["changed"]]
already = [r for r in rw if not r["changed"] and "already carries" in r["reason"]]
refused_rw = [r for r in rw if not r["changed"] and "already carries" not in r["reason"]]
print()
print("the caveat line, rewritten in place: %d file(s) changed, %d already carried it, %d refused"
      % (len(done_rw), len(already), len(refused_rw)))
for r in refused_rw[:8]:
    print("   refused %-58s %s" % (Path(r["path"]).name, r["reason"]))

sel_dep = []
for tag in TAGS10:
    p = WORK / SITE / ("%s_%s" % (RUN10, STAMP10)) / ("%s_%s_10hz_%s_%s.edi" % (SITE, RATE_KIND, tag,
                                                                                PARAMS))
    if not p.exists() or SITE not in one_hz:
        continue
    d = RATE.departure(TFN.read_tf(p), one_hz[SITE])
    if d is None:
        continue
    sel_dep.append(dict(site=SITE, kind=RATE_KIND, selection=tag,
                        xy_pct=round(100 * (d["xy"] - 1), 1) if np.isfinite(d["xy"]) else None,
                        yx_pct=round(100 * (d["yx"] - 1), 1) if np.isfinite(d["yx"]) else None))
if sel_dep:
    print()
    print("the %s stretch and its control against this site's own 1 Hz row, per cent over %g-%g s"
          % (RATE_KIND, rate_rec["band_s"][0], rate_rec["band_s"][1]))
    print(pd.DataFrame(sel_dep).to_string(index=False))
'''),

("md", r"""**This check fails if the chosen stretch does not begin on a whole UTC hour, is not a whole number
of 3,600 s hours, falls outside the record it was chosen from or runs longer than STRETCH_MAX_H; if the
control does not keep the same number of hours as the stretch or overlaps it; if any requested 10 Hz transfer
function is missing or FAILED, carries the wrong selection in its ledger row, or does not name that selection
in its own EDI; if any 10 Hz file of this site carries a caveat other than the survey's measured sentence; or
if any ledger row of this site claiming a file names one that is not on disk.**

The bounds are scored against the cache the stretch was chosen from -- its own `t0` and sample count, read
off the npz -- and not against the selection file's own statement of itself. The EDI limb reads the
`selection=` line out of each written file, which is an observable independent of the ledger row beside it:
the ledger is what the run said it did and the file is what it wrote. The caveat limb reads every 10 Hz file
of this site on disk, not only the ones this run asked for, because one written before the survey's departure
was measured carries the sentence of its own day until the rewrite reaches it."""),

("code", '''z = np.load(WORK / ("cache_%dhz" % SEL.SCORE_RATE) / ("%s.npz" % SITE), allow_pickle=False)
T0_CACHE, N_CACHE = int(z["t0"][0]), int(len(z["Hx"]))
z.close()
T_END = T0_CACHE + N_CACHE / float(SEL.SCORE_RATE)

bad_bounds, n_scored = [], 0
for tag in SEL.TAGS:
    item = SELECTION.get(tag)
    if not item or item.get("t_start") is None:
        continue
    n_scored += 1
    ta, tb = int(item["t_start"]), int(item["t_end"])
    if ta % 3600 or (tb - ta) % 3600 or ta < T0_CACHE or tb > T_END:
        bad_bounds.append("%s %d-%d" % (tag, ta, tb))
    if (tb - ta) / 3600.0 > STRETCH_MAX_H:
        bad_bounds.append("%s runs %.0f h, over the %d h ceiling"
                          % (tag, (tb - ta) / 3600.0, STRETCH_MAX_H))
s_item = SELECTION.get(SEL.STRETCH) or {}
c_item = SELECTION.get(SEL.CONTROL) or {}
bad_control = []
if s_item.get("t_start") is not None and c_item.get("t_start") is not None:
    if s_item["n_hours"] != c_item["n_hours"]:
        bad_control.append("the control keeps %d h against the stretch's %d h"
                           % (c_item["n_hours"], s_item["n_hours"]))
    if not (c_item["t_end"] <= s_item["t_start"] or c_item["t_start"] >= s_item["t_end"]):
        bad_control.append("the control overlaps the stretch")
elif s_item.get("t_start") is not None:
    bad_control.append("the stretch carries no control")

want10 = [(SITE, k, t) for k in KINDS10 for t in TAGS10] if (RATES10 and HAS10) else []
# a stretch the rule refused on its mask asks for no transfer function: it is a stated refusal, named in
# the verdict, and is not counted among those that are missing
stated = ["%s: %s" % (t_, w) for t_, w in sorted(REFUSED10.items())]
have10 = {(r.site, r.kind, str(r.selection)): r for r in l10.itertuples()}
missing10, no_line, wrong_row = [], [], []
for w in want10:
    r = have10.get(w)
    if r is None or str(r.status) == "FAILED":
        missing10.append(w)
        continue
    p_ = Path(str(r.edi))
    if not p_.exists():
        missing10.append(w)
        continue
    if str(r.selection) != w[2]:
        wrong_row.append("%s %s %s -> %s" % (w[0], w[1], w[2], r.selection))
    said = EDI.read_parameter(p_, "selection")
    if not said.startswith(w[2]):
        no_line.append("%s %s %s: %s" % (w[0], w[1], w[2], said[:60] or "no selection= line"))
ten = sorted((WORK / SITE).glob("*/*_10hz_*.edi"))
stale_caveat = [p.name for p in ten if not RATE.carries_caveat(p, CAVEAT)]
rows_all = pd.read_csv(WORK / "survey" / "runs.csv")
claims = rows_all[(rows_all.site == SITE) & rows_all.status.isin(["made", "exists"])]
orphan_rows = ["%s %s %s %g Hz %s" % (r.run, r.stamp, r.kind, r.rate_hz, r.selection)
               for r in claims.itertuples() if not Path(str(r.edi)).exists()]
# a FAILED row for a tag the rule now refuses belongs to an earlier attempt, before the refusal existed:
# the failure limb scores the tags this run asked for
failed10 = (l10[(l10.status == "FAILED") & (l10.selection.isin(TAGS10))] if len(l10) else l10)
print()
print("%d stretch(es) scored against the cache; %d transfer function(s) requested; %d 10 Hz file(s) of this "
      "site scored on their caveat; %d ledger row(s) of this site claiming a file scored against the disk; "
      "%d stretch(es) the rule refused on their mask before any pass (%s)"
      % (n_scored, len(want10), len(ten), len(claims), len(stated), "; ".join(stated) or "none"))
if not n_scored:
    print("VERDICT: UNJUDGED -- no stretch was chosen at %s, so nothing was scored" % SITE)
elif (bad_bounds or bad_control or missing10 or len(failed10) or no_line or wrong_row or stale_caveat
      or orphan_rows):
    print("VERDICT: FAIL -- %d stretch bound(s) are not whole UTC hours inside the record or run over the "
          "ceiling (%s); the control %s (%s); %d of %d requested transfer function(s) are missing (%s); %d "
          "are FAILED (%s); %d carry no matching selection= line (%s); %d carry the wrong selection in the "
          "ledger row (%s); %d of %d 10 Hz file(s) carry a caveat that is not the survey's measured "
          "sentence (%s); %d of %d ledger row(s) claiming a file name one that is not on disk (%s)"
          % (len(bad_bounds), "; ".join(bad_bounds) or "none",
             "does not size or place as the rule states" if bad_control else "holds",
             "; ".join(bad_control) or "none",
             len(missing10), len(want10), "; ".join("%s %s %s" % m for m in missing10[:8]) or "none",
             len(failed10), "; ".join("%s %s: %s" % (r.kind, r.selection, str(r.error)[:70])
                                      for r in failed10.itertuples()) or "none",
             len(no_line), "; ".join(no_line[:6]) or "none",
             len(wrong_row), "; ".join(wrong_row[:6]) or "none",
             len(stale_caveat), len(ten), "; ".join(stale_caveat[:6]) or "none",
             len(orphan_rows), len(claims), "; ".join(orphan_rows[:6]) or "none"))
else:
    print("VERDICT: PASS -- the stretch is %d whole UTC hour(s) inside the record, at most %d h, and its "
          "control keeps the same %d h elsewhere without overlapping it; every one of the %d 10 Hz file(s) "
          "of this site carries the survey's own measured caveat and all %d ledger row(s) claiming a file "
          "name one that is there; %s"
          % (s_item.get("n_hours", 0), STRETCH_MAX_H, c_item.get("n_hours", 0), len(ten), len(claims),
             ("no 10 Hz transfer function was requested (RATES = %s), so none was scored" % RATES)
             if not want10 else
             ("all %d requested over %d kind(s) and the tags %s exist and none is FAILED, and every one "
              "names its selection in its ledger row and in its own EDI%s"
              % (len(want10), len(KINDS10), ", ".join(TAGS10),
                 ("; %d stretch(es) the rule refused on their mask asked for none (%s)"
                  % (len(stated), "; ".join(stated))) if stated else ""))))
'''),

("md", r"""The stretch and what it produced. In the upper panel the two hourly coherences run over the record
with the threshold both must hold drawn across them, and the chosen stretch and its control drawn above as
rows of spans: look at whether the chosen hours sit in one active spell or whether several spells of nearly
the same length were available, and at how far the two lines stand above the threshold inside the stretch.

In the lower panels the stretch and its control are drawn dashed against the 1 Hz row of the same kind,
solid. Look first at the short end, below the marked join, which is the only part a splice would ever use,
and at whether the stretch sits closer to the 1 Hz curve than its control does. Over 4-32 s the 10 Hz row
departs from the 1 Hz row by the rate's own figure, printed in the section above, whatever the hours, so an
offset common to both is the rate and not the choosing."""),

("code", '''KIND10_SHOW = "remote" if "remote" in KINDS10 else KINDS10[0]   # <- the kind the stretch is drawn for
sel_curves = []
base = led.edi[(led.kind == KIND10_SHOW) & (led.rate_hz == 1.0)]
if len(base) and Path(str(base.iloc[0])).exists():
    sel_curves.append(("1 Hz %s" % KIND_WORD[KIND10_SHOW], TFN.read_tf(Path(str(base.iloc[0]))), "k", "-"))
for j, tag in enumerate(TAGS10):
    r_ = l10[(l10.kind == KIND10_SHOW) & (l10.selection == tag)] if len(l10) else l10
    if not len(r_) or not Path(str(r_.edi.iloc[0])).exists():
        continue
    sel_curves.append(("10 Hz %s" % tag, TFN.read_tf(Path(str(r_.edi.iloc[0]))),
                       "0.45" if tag == SEL.CONTROL else "C%d" % j, "--"))
why10 = ("the 10 Hz pass was not run for this site: RATES holds no 10 Hz" if not RATES10 else
         ("the 10 Hz pass was not run for this site: it has no 10 Hz cache" if not HAS10 else
          "the 10 Hz pass was run for this site but wrote no transfer function of this kind"))
show(FIGP.hour_selection(SCORES, SELECTION, sel_curves, SITE, SITE_DIR / "03_selection_10hz.png",
                         join_s=float((sv.cfg.get("bands") or {}).get("splice_join_s", 16)),
                         band_s=SEL.SCORE_BAND_S, coh_min=WINDOW_COH, seed=SEED, no_ten_hz=why10))
'''),

("md", r"""## Every transfer function of this site

Every transfer function this site has, from every run in its folder, on one page: apparent resistivity and
phase for xy and yx, and the tipper as real parts in filled circles and imaginary parts as open triangles on
a dotted line. The kind is the colour, the rate is the line style (1 Hz solid, 10 Hz dashed) and the run is
the marker, so the three questions the page answers are readable at once. The y limits of the rho panels are
the 2nd to 98th percentile of every curve drawn, padded half a decade each way; the phase panels are fixed at
0-90 deg. The header lines under the title are the file's own, taken from the EDI's processing_parameters.

What to look for. The four kinds should lie on each other at 100-1000 s, where the field is large and every
reference sees the same source; a kind that departs from the others there is the one to read, not the average
of them. At the short end the kinds part where the references stop seeing the same source, which is where the
choice of reference is worth the most. The error bars grow at the long end, where the record runs out of
independent windows. The tipper's real and imaginary parts are drawn together because a real part alone
cannot be told from a leak.

Nothing here is estimated again and no file is altered. Every file is read through
`auslamp_proc.transfer_functions.read_tf`, which applies two rules before a curve is used: the EDI
empty-data value 1e32 is masked component by component, together with the no-information convention of a
zero impedance carrying an error of 1e9, so a fill never enters a median; and the periods are sorted and
duplicates dropped before any interpolation, because some writers emit them unsorted and interpolation on an
unsorted grid is silently wrong. The yx phase is folded into the first quadrant by +180 deg in
`auslamp_proc.transfer_functions.rho_phase`, which is what every panel and table below reads.

**This check fails if any transfer function of this site named in runs.csv is missing from disk, does not
read as a transfer function with finite impedance at 5 or more periods, or carries unsorted or duplicated
periods.** The three limbs are independent of the ledger that names them: the first opens the file, the
second counts the finite off-diagonal elements the reader is left with after the fill is masked, and the
third compares the file's own period order with the sorted one. The run folders are also walked for EDIs the
ledger does not name, which is the other half of the same question and is reported beside the verdict."""),

("code", '''PERIOD_RANGE = (1, 50000)     # <- the periods drawn and scored, in s
ALL_TF, IGNORED = RD.deliverable(TFN.find_transfer_functions(sv, [SITE], runs="all", work_root=WORK))
PAIRS_RUN = TFN.choose_runs(TFN.ledger(WORK), "all")
_cache = {}
def read(path):
    """One TFData per file. The page and the three tables all ask for the same file."""
    if path not in _cache:
        _cache[path] = TFN.read_tf(path)
    return _cache[path]

rows, missing_tf, too_short, out_of_order = [], [], [], []
for r in ALL_TF.itertuples():
    if not r.on_disk:
        missing_tf.append("%s %g Hz %s: %s" % (r.kind, r.rate_hz, r.selection, Path(r.path).name))
        continue
    try:
        tf = read(r.path)
    except Exception as exc:
        missing_tf.append("%s %g Hz %s: unreadable, %s"
                          % (r.kind, r.rate_hz, r.selection, type(exc).__name__))
        continue
    finite = max(tf.meta["n_finite"]["xy"], tf.meta["n_finite"]["yx"])
    if finite < 5:
        too_short.append("%s %g Hz %s: %d finite period(s)" % (r.kind, r.rate_hz, r.selection, finite))
    if not tf.meta["source_sorted"] or tf.meta["n_duplicate_periods"]:
        out_of_order.append("%s %g Hz %s: sorted %s, %d duplicate(s)"
                            % (r.kind, r.rate_hz, r.selection, tf.meta["source_sorted"],
                               tf.meta["n_duplicate_periods"]))
    rows.append(dict(run=r.run, kind=r.kind, rate_hz=r.rate_hz, selection=r.selection,
                     periods=tf.meta["n_periods"], shortest_s=round(float(tf.period.min()), 2),
                     longest_s=round(float(tf.period.max()), 0), finite_xy=tf.meta["n_finite"]["xy"],
                     finite_yx=tf.meta["n_finite"]["yx"], tipper=tf.meta["has_tipper"],
                     seconds=r.seconds, rss_mb=r.peak_rss_mb, remote=r.remote,
                     members=(str(r.members)[:46] if r.members else "")))
found = pd.DataFrame(rows)
print(found.to_string(index=False))
print()
on = ALL_TF[ALL_TF.on_disk]
head = TFN.metadata_lines(read(on.iloc[0].path).meta) if len(on) else []
page, _index = FIG.site_page(
    SITE, ALL_TF, read, SITE_DIR / "03_transfer_functions.png",
    title="%s: every transfer function on one page" % SITE,
    caption="Every transfer function of %s on disk, over %g-%g s: the kind is the colour, the rate the "
            "line style (1 Hz solid, 10 Hz dashed) and the run the marker. The rho limits are the 2nd to "
            "98th percentile of every curve drawn, padded half a decade each way, and the phase panels "
            "are fixed at 0-90 deg. What to change: PERIOD_RANGE sets the periods drawn."
            % (SITE, PERIOD_RANGE[0], PERIOD_RANGE[1]),
    header_lines=head, period_range=PERIOD_RANGE)
show(page)
if len(IGNORED):
    print("%d %s file(s) sit in this site's run folders and are read by nothing here: %s"
          % (len(IGNORED), RD.DROPPED_KIND, " ".join(sorted(Path(p).name for p in IGNORED.path))[:300]))
spare = TFN.unledgered(WORK, [SITE], PAIRS_RUN)
print("%d EDI(s) sit in this site's run folders without a ledger row%s"
      % (len(spare), (": " + "; ".join(Path(p).name for p in spare[:6])) if spare else ""))

n = len(ALL_TF)
print()
if not n:
    print("VERDICT: UNJUDGED -- runs.csv names no transfer function for %s" % SITE)
elif missing_tf or too_short or out_of_order:
    print("VERDICT: FAIL -- %d of %d are missing or unreadable (%s); %d read fewer than 5 finite periods "
          "(%s); %d carry unsorted or duplicated periods (%s)"
          % (len(missing_tf), n, "; ".join(missing_tf[:4]) or "none", len(too_short),
             "; ".join(too_short[:4]) or "none", len(out_of_order),
             "; ".join(out_of_order[:4]) or "none"))
else:
    print("VERDICT: PASS -- all %d transfer function(s) of %s named in runs.csv are on disk, each reads with "
          "%d to %d finite off-diagonal periods over %.2f to %.0f s, and every one carries its periods "
          "sorted and unique"
          % (n, SITE, int(found[["finite_xy", "finite_yx"]].max(axis=1).min()),
             int(found[["finite_xy", "finite_yx"]].max(axis=1).max()),
             float(found.shortest_s.min()), float(found.longest_s.max())))
'''),

("md", r"""### Kind against kind

For each rate, every pair of kinds is scored per decade: the median ratio of apparent resistivity (the ratio
of the squared impedance magnitudes) and the median phase difference in degrees, over the periods both curves
carry. `auslamp_proc.agreement.on_grid` puts the second curve on the first's periods by linear interpolation
in log period of log |Z| and of the unwrapped phase, with no extrapolation beyond the second curve's own
first and last valid node and no bridging of a hole wider than 0.30 decades.

The rule: two kinds agree where the rho ratio is within AGREE_RHO of one and the phase difference is within
AGREE_PHASE_DEG, both over AGREE_BAND. At the values below that is 20 per cent and 5 deg over 5-200 s.
Agreement between two references that share no magnetics is the evidence that neither is carrying its own
noise into the estimate; disagreement between them is not resolved here, and is workbook 05's question.

**This check fails if the table is UNJUDGED, that is if no pair was scored.** A site with one kind yields no
pair, and a rate where no pair could be put on a common period range scores nothing at all; either is a
failure, because a table of no rows cannot support the reading above it."""),

("code", '''AGREE_RHO = 0.20              # <- two curves agree where the rho ratio is within this fraction of one
AGREE_PHASE_DEG = 5.0         # ... and the phase difference is within this many degrees
AGREE_BAND = (5, 200)         # the band the agreement rule is read over, in s
BANDS = [(5, 10), (10, 100), (100, 1000), (1000, 10000)]   # the decades the table reports, in s

kk = AG.kind_vs_kind(ALL_TF, read=read, bands=[tuple(b) for b in BANDS], agree_rho=AGREE_RHO,
                     agree_phase=AGREE_PHASE_DEG, agree_band=tuple(AGREE_BAND))
band_tag = AG.band_label(*AGREE_BAND)
scored_kk = kk[(kk.band == band_tag) & (kk.n > 0)] if len(kk) else kk
counts = []
for (site_k, rate_k), g in (scored_kk.groupby(["site", "rate_hz"]) if len(scored_kk) else []):
    pr = g.groupby(["kind_a", "kind_b"]).agrees.all()
    counts.append(dict(site=site_k, rate_hz=rate_k, pairs=int(len(pr)), agreeing=int(pr.sum()),
                       worst_rho_ratio=round(float(g.rho_ratio.iloc[
                           int(np.nanargmax(np.abs(np.log(g.rho_ratio.values))))]), 3),
                       worst_phase_deg=round(float(g.phase_diff_deg.iloc[
                           int(np.nanargmax(np.abs(g.phase_diff_deg.values)))]), 2),
                       disagreeing=" ".join(sorted({"%s-%s" % k for k, v in pr.items() if not v}))))
agree_counts = pd.DataFrame(counts)
print("both components inside %.0f %% and %.1f deg over %s; a pair counts as agreeing only where both do"
      % (100 * AGREE_RHO, AGREE_PHASE_DEG, band_tag))
print(agree_counts.to_string(index=False))
print()
print("the per-decade table")
if len(kk):
    print(kk[kk.band != band_tag][["kind_a", "kind_b", "rate_hz", "band", "component", "rho_ratio",
                                   "phase_diff_deg", "n"]].round(3).to_string(index=False))
path = SITE_DIR / "03_agreement_kinds.csv"
kk.round(4).to_csv(path, index=False)
WRITTEN.append(path)
print()
print("-> %s (%d rows)" % (path, len(kk)))
print()
if not len(scored_kk):
    print("VERDICT: UNJUDGED -- no pair of kinds was scored over %s" % band_tag)
else:
    n_pairs = int(agree_counts.pairs.sum())
    n_agree = int(agree_counts.agreeing.sum())
    print("VERDICT: PASS -- %d pair(s) of kinds scored over %d rate(s); %d agree within %.0f %% and %.1f "
          "deg over %s and %d do not (%s)"
          % (n_pairs, len(agree_counts), n_agree, 100 * AGREE_RHO, AGREE_PHASE_DEG, band_tag,
             n_pairs - n_agree,
             " ".join(sorted({x for s in agree_counts.disagreeing for x in s.split() if x})) or "none"))
'''),

("md", r"""### Rate against rate

Where this site carries both a 1 Hz and a 10 Hz row of the same kind, the two are compared on the bands each
side of the 16 s join: 8-16 s below and 32-100 s above, with 18-36 s left as a guard band because the Earth
Data logger writes an instrument line at 20.6 s. The number reported is the difference in apparent
resistivity as a percentage of the 1 Hz level, and the phase difference in degrees.

Every 10 Hz file carries the survey's own measured departure at 4-32 s in its header, written above. Nothing
is spliced here; the join is workbook 05's work, and this section is the measurement it is decided on. A
reading, not a check."""),

("code", '''rows = []
g_all = ALL_TF[ALL_TF.on_disk]
for kind in sorted(set(g_all.kind)):
    a = g_all[(g_all.kind == kind) & (g_all.rate_hz == 1.0)]
    b = g_all[(g_all.kind == kind) & (g_all.rate_hz == 10.0)]
    if not len(a) or not len(b):
        continue
    for r_ in b.itertuples():
        for _, s_ in AG.rate_step(read(a.iloc[0].path), read(r_.path)).iterrows():
            rows.append(dict(kind=kind, selection=r_.selection, component=s_.component,
                             below_8_16_pct=round(s_.below_pct, 1),
                             below_phase_deg=round(s_.below_phase_deg, 2),
                             above_32_100_pct=round(s_.above_pct, 1),
                             above_phase_deg=round(s_.above_phase_deg, 2),
                             n_below=int(s_.n_below), n_above=int(s_.n_above)))
steps = pd.DataFrame(rows)
if not len(steps):
    print("%s carries no pair of a 1 Hz and a 10 Hz row of the same kind, so the rate step is not scored"
          % SITE)
else:
    print("the 1 Hz level minus the 10 Hz level, as a per cent of the 1 Hz apparent resistivity")
    print(steps.to_string(index=False))
    print()
    print("median below the join %+.1f %%, above %+.1f %%"
          % (steps.below_8_16_pct.median(), steps.above_32_100_pct.median()))
'''),

("md", r"""### Run against run

Where this site has the same kind, rate and selection under two runs, the two are put on the ten-per-decade
grid T = 10^(k/10), which belongs to neither of them, and the largest change per decade is reported. Two runs
of the same kind differ only in what the run itself did -- a mask, a stretch, a reference the rule chose
differently on the day -- so a change here is the size of that decision.

A site with one run of a kind yields nothing to compare, which is stated as a reading and not as a verdict:
there is no criterion a single run can fail."""),

("code", '''rr = AG.run_vs_run(ALL_TF, read=read, bands=[tuple(b) for b in BANDS])
if not len(rr):
    per_kind = ALL_TF.groupby(["kind", "rate_hz", "selection"]).run.nunique()
    print("UNJUDGED: %s carries no kind, rate and selection under two runs -- %d combination(s) each have "
          "one run, so there is nothing to compare across runs" % (SITE, int((per_kind == 1).sum())))
    print("the runs present: %s" % ", ".join("%s_%s" % p for p in PAIRS_RUN))
else:
    worst = (rr.assign(change=lambda d: np.abs(np.log(d.rho_ratio)))
               .sort_values("change", ascending=False)
               .groupby(["kind", "rate_hz"]).head(1))
    print("the largest change per kind and rate, over the decades %s"
          % ", ".join(AG.band_label(*b) for b in BANDS))
    print(worst[["kind", "rate_hz", "run_a", "run_b", "band", "component", "rho_ratio", "phase_diff_deg",
                 "n"]].round(3).to_string(index=False))
'''),

("md", r"""## What was written"""),

("code", '''rows = []
for d in (WORK / SITE / ("%s_%s" % (RUN, STAMP)), WORK / SITE / ("%s_%s" % (RUN10, STAMP10))):
    for p in sorted(d.glob("*")):
        if p.is_file():
            rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for rate in RATES:
    for p in sorted((WORK / "references" / ("%dhz" % rate)).glob("*%s*" % SITE)):
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for name in (SEL.SCORES_NAME, SEL.SELECTION_NAME):
    p = WORK / SITE / name
    if p.exists():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
for p in list(WRITTEN) + [WORK / "survey" / "runs.csv"]:
    if Path(p).exists():
        rows.append(dict(file=str(p), kb=round(Path(p).stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d files, %.1f MB" % (len(files), files.kb.sum() / 1024))
print(files.to_string(index=False))
print()
print("the %d figure(s) this workbook drew" % len(WRITTEN))
for p in WRITTEN:
    print("   %-64s %8.1f KB" % (str(p), Path(p).stat().st_size / 1024))
print()
print("the first 30 processing_parameters lines of one EDI")
one = sorted((WORK / SITE / ("%s_%s" % (RUN, STAMP))).glob("*.edi"))
if one:
    txt = one[0].read_text(encoding="utf-8", errors="ignore")
    keep_lines = [ln.strip() for ln in txt.splitlines() if "=" in ln and not ln.strip().startswith(">")]
    print(one[0].name)
    for ln in keep_lines[:30]:
        print("   %s" % ln[:160])
print()
print("the selection line of one 10 Hz file")
sel_one = sorted((WORK / SITE / ("%s_%s" % (RUN10, STAMP10))).glob("*.edi"))
if sel_one:
    print(sel_one[0].name)
    for ln in sel_one[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.strip().startswith(("selection=", "runs=", "caveat_10hz=")):
            print("   %s" % ln.strip()[:200])
'''),
]
