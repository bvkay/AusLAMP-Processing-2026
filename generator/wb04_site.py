r"""Workbook 04, one site in depth: the magnetics, the windows, the forms and the recipe.

The cells are a Python list of ("md", text) and ("code", source). generator/make_workbooks.py imports the
list from here and writes 04_site.ipynb.

@author: ben kay (ben@auscope.org.au)
"""

WB04_PARAMS = '''# ---- parameters: change these and re-run the workbook ----
SURVEY = "queensland_phase1"  # any folder under surveys/
SITE = "Q53N"                 # one site; Q53N carries a shared centre and sound magnetics (see below)
RUN = "site"                  # the forms' run name; the folder is <work_root>/<SITE>/<RUN>_<stamp>
STAMP = None                  # None = the newest <RUN>_* folder if there is one, else a new stamp
BASELINE_KIND = "remote"      # the reference every form is built on: remote | stack | obs | stack_obs
RATES = [1, 10]               # [1] is the 1 Hz lane alone; 10 adds the short end of section 5
COMPONENTS = ["xy", "yx"]     # the components the windows and the forms are read for
REDO = False                  # True remakes a form whose EDI is already in the run folder
WRITE_DECISIONS = False       # False by default, unlike workbooks 02 and 03. Those two MEASURE a sign and
                              # write it into a cell that reads `decide`; this one PROPOSES the `windows`
                              # and `flags` cells, which are free text describing what a campaign did, so
                              # they are the analyst's to write. True writes the proposal as it stands
WORK_ROOT = None              # None = survey.yaml work_root; every transfer function and figure lands there
'''

WB04_RULES = '''# ---- the method parameters: a change here changes what a form is built on ----
K_NEAREST = 2                 # the neighbours the daily magnetics test reads a day against
SEED = 20260916               # the named seed every control is drawn under
CENTRE_DAYS = 3               # the days of highest Ex-Ey coherence the residual test is read over
REPLACE_CHANNEL = None        # None = from the DC flags and the candidates table | "Hx" | "Hy"
LENDER = None                 # None = the nearest candidate that is not in the reference | a site name
BAR_MARGIN = 0.20             # a delivered selection must beat its control on the bar by this fraction
BAR_BAND = (10, 1000)         # the band every bar in the forms table is read over, in s
AGREE_BAND = (100, 1000)      # the band agreement with the baseline is read over, in s
DECADES = [(5, 10), (10, 100), (100, 1000), (1000, 10000)]   # the decades the tables report, in s
'''

WB04_SETUP = '''import os
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
from IPython.display import Image, display

from loguru import logger as _loguru
_loguru.remove()

from auslamp_proc import agreement as AG, look as LK, transfer_functions as TFN, splice as SP, survey as SV
from auslamp_proc.process import KIND_WORD, references as REF
from auslamp_proc.process import aurora_run as AR
from auslamp_proc.process import selection as SEL
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

t0_rec, n_rec = MK.span(sv, SITE, 1)


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
        _tfs[path] = TFN.read_tf(path)
    return _tfs[path]


def rows_by_form():
    return {r["form"]: r for r in FORM_ROWS}


def made_tf(name):
    """The transfer function of one form where it was made and is on disk, else None."""
    r = rows_by_form().get(name, {})
    if r.get("status") in ("made", "exists") and Path(str(r.get("transfer_function") or "")).exists():
        return r
    return None


def baseline_path():
    """Workbook 03's transfer function of the chosen kind at 1 Hz, copied here as the `whole` form.

    The baseline is not re-estimated: it is the file workbook 03 wrote, under the name the forms table uses,
    so every form is read against the transfer function the earlier workbooks left.
    """
    led = TFN.ledger(WORK)
    rows = led[(led.site == SITE) & (led.kind == BASELINE_KIND) & (led.rate_hz == 1.0)]
    if not len(rows):
        return None, "runs.csv names no %s transfer function of %s at 1 Hz" % (BASELINE_KIND, SITE)
    r = rows.sort_values("stamp").iloc[-1]
    src = TFN.tf_path(WORK, SITE, r.run, r.stamp, r.kind, r.rate_hz, r.params)
    if not src.exists():
        return None, "%s is not on disk" % src
    dst = OUT / FM.tf_name(SITE, "whole", BASELINE_KIND, 1, PARAMS)
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst, "copied from %s" % src.parent.name


BASE, BASE_WHY = baseline_path()
FORM_ROWS.append(dict(site=SITE, form="whole", kind=BASELINE_KIND, rate_hz=1.0, params=PARAMS,
                      transfer_function=str(BASE) if BASE else "", controls="", criterion="the baseline itself",
                      seed=None, status=("exists" if BASE else "FAILED"),
                      error=("" if BASE else BASE_WHY), days=None, n_runs=None, seconds=None))

ELINES = CE.read_elines(sv, SITE)
DC = pd.read_csv(SITE_DIR / "dc.csv") if (SITE_DIR / "dc.csv").exists() else pd.DataFrame()

print("survey       %s" % sv.cfg["name"])
print("site         %s -- %s" % (SITE, sv.site(SITE).notes or "no note in sites.csv"))
print("work root    %s" % WORK)
print("run folder   %s" % OUT)
print("baseline     %s (%s)" % (Path(BASE).name if BASE else "NONE", BASE_WHY))
print("record       %.2f d of 1 Hz samples from %s UTC" % (n_rec / 86400.0,
                                                           pd.Timestamp(t0_rec, unit="s")))
print("rates        %s Hz; components %s" % (", ".join(str(r) for r in RATES), ", ".join(COMPONENTS)))
print("seed         %d, used by every control in this workbook" % SEED)
'''

WB04 = [
("md", r"""# 04 -- One site, in depth

One site is taken apart: which days its magnetics are usable, where in time each electric line is worth
using, whether its two lines share a centre electrode, what its 10 Hz record adds at the short end, and
whether a magnetic channel is worth borrowing from a neighbour. Each answer is built as a form -- one pass
over the same site with one thing changed -- and every form lands in the same run folder as a transfer
function carrying the header a workbook 03 transfer function carries. Section 8 then composes the answers:
one frame and, per impedance row, which hours that row is estimated on and at which rate, assembled into one
transfer function.

The rule this workbook is written around: a selection of time is never delivered without its control. Each
selection carries

1. a control of the same length and the same shape -- a contiguous block placed at random elsewhere in the
   record under a seed written into the file's header -- so the two cost the same and only the choosing
   differs;
2. the rule that chose it with its threshold, so the selection can be read back from the table it was chosen
   on: the elines table for a window of days, the hour score table for a stretch of hours;
3. the transfer function itself scored against its control on the 10-1000 s bar, on smoothness, and on
   agreement with the baseline at 100-1000 s.

The control is scored on the transfer function that would be delivered, not only on the statistic that chose
the time. A selection whose transfer function does not beat its control has narrowed its error bars and
not changed the answer, and is not promoted.

Two rules choose time here and they answer to different faults. A 1 Hz window per impedance row, section 3,
is the longest run of days that row's own line is sound in the elines table: a line that died mid-record
takes weeks of the record with it, and the window is the weeks it was alive. A 10 Hz pass, section 5, and the
recipe's `window:coherent` row, section 8, take the longest contiguous stretch of whole UTC hours in which
the line reads above 0.5 over 20-200 s, because the short end is all that is wanted of them. Selecting on the
target's own E-H coherence uses the target's own response and favours the hours where the linear model
already fits, so the threshold sits well below live, where a live line reads above 0.85; and a coherence dip
at every quiet night is not a line dying, which is why the 1 Hz window is not chosen that way.

Seven checks state their failure criterion in bold above the cell and print a verdict below it. A check that
scores zero items prints UNJUDGED and counts as a failure. A criterion that is met is reported as FAIL and is
not revised afterwards. The recipe's window check is scored only where `survey.yaml` records a window for the
site; elsewhere it prints the stretch the rule found as a reading and writes no verdict.

The site the parameter cell opens on is AusLAMP Queensland Phase 1's Q53N: its two electric lines read an
Ex-Ey coherence of 0.95-0.99 on every day of the record while its magnetometer passes the DC test against
IGRF, so the shared-centre test of section 4 has something to find and the magnetics every other section
rests on are not themselves in doubt. It has neighbours 89 and 110 km away on two sides, which is what the
clock test and the borrowed channel of section 6 need."""),

("code", WB04_PARAMS),
("code", WB04_RULES),
("code", WB04_SETUP),

("md", r"""## 1. The site as workbooks 02 and 03 left it

The record figure, the magnetometer DC test and the per-day state of the two electric lines are workbook 02's;
the transfer functions and their agreement are workbook 03's. Nothing here is recomputed. What the three
tables say is what selects the methods below: a magnetometer that fails the DC test sends the work to section
6, a line that dies mid-record sends it to section 3, and two lines carrying one voltage send it to section
4."""),

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

("md", r"""### What this workbook tries, and why

The methods are keyed to the three tables above: a common-mode signature on the two lines calls for the
shared-centre test of section 4, a magnetic flag or a channel reading low against its neighbours calls for
the replacement of section 6, and a line that dies mid-record calls for the window of section 3. A line
alive but noisy throughout is not what a window is for: the days it was alive are all of them, and the
answer to noise is the reference, the mask or the recipe's coherent hours."""),

("code", '''common = float(np.nanmedian(ELINES.coh_Ex_Ey)) if len(ELINES) else np.nan
# DC["flags"], not DC.flags: a DataFrame carries a `flags` attribute of its own and attribute access
# reaches that object rather than the column
flags = str(DC["flags"].iloc[0]) if len(DC) and "flags" in DC.columns else ""
dead = {c: int((ELINES["%s_state" % LINE[c]] == "dead").sum()) if len(ELINES) else 0 for c in COMPONENTS}
plan = [
    dict(section="2 the magnetics by day", applies=True, why="every later test rests on Hx and Hy"),
    dict(section="3 windows", applies=bool(len(ELINES)),
         why="a line dead on %s of %d day(s)" % ("/".join(str(dead[c]) for c in COMPONENTS), len(ELINES))),
    dict(section="4 the shared centre", applies=bool(np.isfinite(common) and common >= 0.5),
         why="median Ex-Ey coherence %.2f" % common),
    dict(section="5 the 10 Hz forms", applies=10 in RATES,
         why="the short end the 1 Hz cache cannot reach"),
    dict(section="6 replacement magnetics", applies=True, why="DC flags: %s" % (flags or "none")),
]
print(pd.DataFrame(plan).to_string(index=False))
'''),

("md", r"""## 2. The magnetics by day

Two tests, each reading the site against something outside it.

The daily test reads each UTC day's total field against IGRF and the fluctuation of Hx and Hz -- the standard
deviation after a 3,000 s high-pass -- against the median of the K_NEAREST sites recording that day. A day is
sound where the field is within 5 per cent of IGRF and both fluctuations sit within a factor 3 of the
neighbours'. Without a neighbour that day only the field is judged and the row says so.

The clock test reads the lag of the peak cross-correlation of the despiked Hx against a reference site over a
+-12 h search, the peak refined to a fraction of a sample with a parabola through its two neighbours. The
series correlated is the 5-20 s band-passed record, not the 3,000 s high-passed one the daily test uses: a
high-pass leaves the daily variation in, and a half-day sinusoid correlated against a neighbour peaks at the
edge of a +-12 h search as readily as at zero (at Q17 that gave a lag of hours where the clock is right to
0.4 s). The search stays at +-12 h so an hour-scale offset is still found.

A day is counted only where the field test holds and three guards pass: the peak correlation reaches 0.5, the
peak stands at least 1.5 times the median correlation over that day's other lags, and the peak is not within
5 per cent of the search edge. Fewer than 5 counted days leaves the clock UNJUDGED and no median is reported.

**This check fails if no day of the record is sound on the daily test, if fewer than 5 days could be timed
against the reference site, or if the clock lag's median exceeds 10 s.**"""),

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
fig = FF.clock_lags(ck, SITE, OUT / "07_clock.png", pass_s=MK.CLOCK_PASS_S,
                    edge_fraction=MK.CLOCK_EDGE_FRACTION, peak_ratio=MK.CLOCK_PEAK_RATIO,
                    maxlag_s=MK.CLOCK_MAXLAG_S)
WRITTEN.append(OUT / "07_clock.png")
display(Image(filename=str(OUT / "07_clock.png")))
'''),

("code", '''fail = []
if not int(dm.sound.sum()):
    fail.append("no day of the record is sound on the daily test: %d day(s) scored and %d judged against a "
                "neighbour" % (len(dm), int(dm.judged.sum())))
if not ck["judged"]:
    fail.append("UNJUDGED on the clock against %s: %s" % (ck["ref"], ck["reason"]))
elif abs(ck["median_lag_s"]) > MK.CLOCK_PASS_S:
    fail.append("the clock lag's median is %+.2f s over %d day(s), beyond %.0f s"
                % (ck["median_lag_s"], ck["n_days"], MK.CLOCK_PASS_S))
if fail:
    print("VERDICT: FAIL -- %s; for the record %d of %d day(s) are sound and the clock counted %d day(s) of "
          "%d on the %g-%g s band"
          % ("; ".join(fail), int(dm.sound.sum()), len(dm), ck["n_days"], len(ck["table"]),
             MK.CLOCK_BAND_S[0], MK.CLOCK_BAND_S[1]))
else:
    print("VERDICT: PASS -- %d of %d day(s) are sound on the field and fluctuation tests, %d of them judged "
          "against a neighbour within %g per cent of IGRF and a factor %g of the %d nearest sites; and the "
          "clock lag's median is %+.2f s over %d counted day(s) of %d on the %g-%g s band, within %.0f s"
          % (int(dm.sound.sum()), len(dm), int(dm.judged.sum()), 100 * MK.F_TOLERANCE,
             MK.FLUCTUATION_FACTOR, K_NEAREST, ck["median_lag_s"], ck["n_days"], len(ck["table"]),
             MK.CLOCK_BAND_S[0], MK.CLOCK_BAND_S[1], MK.CLOCK_PASS_S))
'''),

("md", r"""## 3. Windows

A line that dies mid-record is not a reason to throw the record away. The rule is the whole record for the
healthy row and for the tipper, a window for the other row, and both windows in the provenance.

The window of a row is the longest run of days on which that row's own line is `sound` in the elines table
workbook 02 wrote -- the xy row on Ex, the yx row on Ey, because the x impedance row is estimated from Ex and
the y row from Ey. Where no run of sound days reaches the `floors.min_window_days` the survey names, the weak
days are admitted, the row says so, and the window is a proposal and not a finding. A window already in
decisions.csv is used instead of the proposal.

This is not the rule sections 5 and 8 apply, and the two answer to different faults. A 10 Hz pass, and the
recipe's `window:coherent` row, want the most coherent hours the record holds, because the short end is all
that is wanted of them; a 1 Hz window per row exists for a line that died, and what it wants is the weeks
that line was alive. An hour rule put to the second question returns hours where the answer needs weeks, and
a coherence dip at every quiet night is not a line dying.

The windowed pass slices everything to the window, H included: the point of a window is that this
component's estimate sees only the days its electrode was alive, and an estimator handed a longer H than E
would be given NaN over the rest. The merge then replaces exactly that component's two impedance rows in a
copy of the whole-record file; the station block, the position, the tipper and every other row carry across
untouched.

The control is a block of the same length placed at random elsewhere in the record under the seed printed
below: a window that buys nothing beyond its length is one a block of the same length placed anywhere would
buy.

A windowed form on disk is reused only where it records the bounds this rule gives now. Each pass writes the
two unix seconds, the hours, the name of the rule and its threshold into its own file and into the run
folder's provenance, and the cell below reads them back and compares them before it asks for the pass. A
difference, or a file that records no bounds at all, remakes the form and its control together and says so:
a window and its control are one pair, because the control is a block of the window's own length. This is
the guard the arm-diagonal cache of section 4 already carries, on the one thing a window is: the span it saw.
REDO False therefore keeps a form whose bounds match and no other.

**This check fails if the merge changes any row other than the windowed component's two, if a window lacks
its equal-length random block, or if a promoted window does not beat that block on the 10-1000 s bar by at
least 20 per cent.** A window that does not beat its block is not promoted, which is a finding about the
site and not a failure of the check."""),

("code", '''t = time.time()
HOUR_TABLE = SEL.site_scores(sv, SITE, rate=SEL.SCORE_RATE)
print("the hour score over %d whole UTC hour(s) of the %g Hz cache in %.0f s: the median squared coherence "
      "of Ex with Hy and of Ey with Hx over %g-%g s, Welch at %d s segments, written to %s. It is read by "
      "section 5's 10 Hz selection and by section 8's window:coherent rows, and not by this section"
      % (len(HOUR_TABLE), SEL.SCORE_RATE, time.time() - t, SEL.SCORE_BAND_S[0], SEL.SCORE_BAND_S[1],
         SEL.SCORE_SEGMENT_S, SEL.scores_path(WORK, SITE).name))
print()
MIN_WINDOW_DAYS = FM.min_window_days(sv)
WINDOWS, WIN_ROWS = {}, []
DEC_WINDOWS = {}
_cell = str(sv.decision(SITE).get("windows", "")).strip()
if _cell and _cell.lower() not in ("decide", "nan", "none", ""):
    try:
        DEC_WINDOWS = json.loads(_cell)
    except Exception as exc:
        print("decisions.csv windows does not parse as JSON (%s); the proposal below is used" % exc)
for comp in COMPONENTS:
    seed = SEED + (0 if comp == "xy" else 1)
    d = DEC_WINDOWS.get(comp) or {}
    if d.get("t_start"):
        w = dict(component=comp, t_start=int(pd.Timestamp(d["t_start"], tz="UTC").timestamp()),
                 t_end=int(pd.Timestamp(d["t_end"], tz="UTC").timestamp()), days=d.get("days"),
                 n_days=d.get("days"), states="", rule="decisions.csv",
                 reason="decisions.csv: %s" % d.get("reason", ""))
    else:
        w = dict(MK.window_from_days(ELINES, comp, min_days=MIN_WINDOW_DAYS), rule="")
        w["rule"] = "days:%s" % (w["states"] or "none")
    w["line"] = LINE[comp]
    w["seed"] = seed
    if w["t_start"]:
        a, b = MK.random_block(t0_rec, n_rec, int(w["t_end"] - w["t_start"]), seed,
                               exclude=(w["t_start"] - t0_rec, w["t_end"] - t0_rec))
        w["control"] = (t0_rec + a, t0_rec + b)
        w["control_reason"] = ("a block of the same %.2f d placed at random elsewhere in the record, "
                               "seed %d: %s to %s UTC"
                               % (w["days"], seed, pd.Timestamp(w["control"][0], unit="s"),
                                  pd.Timestamp(w["control"][1], unit="s")))
        w["bounds"] = FM.window_bounds(w, rule=w["rule"], threshold=MIN_WINDOW_DAYS)
        w["control_bounds"] = FM.window_bounds(dict(t_start=w["control"][0], t_end=w["control"][1]),
                                               rule="random_block", threshold=MIN_WINDOW_DAYS)
    else:
        w["control"], w["control_reason"] = None, ""
        w["bounds"], w["control_bounds"] = {}, {}
    WINDOWS[comp] = w
    WIN_ROWS.append(dict(component=comp, line=LINE[comp], against=COMP_H[comp], t_start=w["t_start"],
                         t_end=w["t_end"], days=w["days"], n_days=w["n_days"], states=w["states"],
                         rule=w["rule"], seed=seed,
                         control_start=(w["control"][0] if w["control"] else None)))
print("the window of each row: the longest run of sound days of that row's own line, floor %.2f d"
      % MIN_WINDOW_DAYS)
print(pd.DataFrame(WIN_ROWS).to_string(index=False))
for comp in COMPONENTS:
    w = WINDOWS[comp]
    print("   %s: %s" % (comp, w["reason"]))
    if w["control"]:
        print("      control: %s" % w["control_reason"])
'''),

("code", '''for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        print("   %s: no window proposed (%s)" % (comp, w["reason"]))
        continue
    # the guard: a form on disk is kept only where the bounds it records are the ones the rule gives now.
    # A window and its control are one pair -- the control is a block of the window's own length -- so a
    # difference on either remakes both.
    notes = []
    for name, want in (("window_%s" % comp, w["bounds"]),
                       ("window_%s_control" % comp, w["control_bounds"])):
        why = FM.bounds_note(FM.recorded_bounds(OUT, SITE, name, BASELINE_KIND, 1, PARAMS), want)
        if why:
            notes.append("%s: %s" % (name, why))
    stale = bool(notes)
    for note in notes:
        print("   stale, remade: %s" % note)
    if not stale:
        print("   %s: the pair on disk records the bounds the rule gives now (%s), so neither is remade"
              % (comp, FM.bounds_words(w["bounds"])))
    form("window_%s" % comp, window=(w["t_start"], w["t_end"]), seed=w["seed"],
         keep_name="the %s window, %.2f d" % (comp, w["days"]),
         controls=["window_%s_control" % comp], bounds=w["bounds"], redo=bool(REDO or stale),
         criterion="beats an equal-length random block on the %g-%g s bar by %.0f %%"
                   % (BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN),
         extra_lines=["window_rule=%s" % w["reason"]])
    form("window_%s_control" % comp, window=w["control"], seed=w["seed"],
         bounds=w["control_bounds"], redo=bool(REDO or stale),
         keep_name="a random block of the same length elsewhere in the record, seed %d" % w["seed"])
'''),

("md", r"""The record with each component's window as a solid span and its equal-length random block
hatched. What to look for is the window sitting where that line's days are alive and the block landing
somewhere the line is neither obviously better nor worse: the two cost the same, so the difference between
their transfer functions is what the window bought."""),

("code", '''_t0w, _arrw, _metaw = CACHE.load(SITE, WORK, 1)
spans = []
for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        continue
    spans.append(("%s window, %.1f d" % (comp, w["days"]), FF.SPAN_COLOUR[comp], None,
                  [(w["t_start"], w["t_end"])]))
    spans.append(("%s block, seed %d" % (comp, w["seed"]), "0.4", "//", [w["control"]]))
fig = FF.record_spans(_t0w, _arrw, OUT / "08_windows_record.png", site=SITE, spans=spans,
                      title="%s: the windows and their equal-length blocks" % SITE,
                      caption="The record as a per-minute mean over its per-minute envelope, with each "
                              "component's window drawn as a solid span in that component's colour and its "
                              "control hatched beside it. A window is the longest run of days the elines "
                              "table calls sound for that component's own line, with the weak days admitted "
                              "and named where no sound run reaches the %.2f d floor. The control is a "
                              "block of the same length placed at random elsewhere in the record under the "
                              "seed printed above, so the two cost the same and only the choosing differs: "
                              "%s."
                              % (MIN_WINDOW_DAYS,
                                 "; ".join("%s keeps %s d on %s days" % (c, WINDOWS[c]["days"],
                                                                         WINDOWS[c]["states"] or "decided")
                                           for c in COMPONENTS)))
WRITTEN.append(OUT / "08_windows_record.png")
display(Image(filename=str(OUT / "08_windows_record.png")))
del _arrw
'''),

("code", '''MERGES = []
for comp in COMPONENTS:
    r = made_tf("window_%s" % comp)
    if r is None or not BASE:
        continue
    out = OUT / FM.tf_name(SITE, "merged_%s" % comp, BASELINE_KIND, 1, PARAMS)
    got = FM.merge_component(BASE, r["transfer_function"], comp, out_edi=out, verbose=False)
    MERGES.append(got)
    print("   %s: %s; rows changed %s (expected %s); every other row unchanged: %s"
          % (comp, got["how"], ", ".join(got["rows_changed"]) or "none", ", ".join(got["rows_expected"]),
             got["untouched_unchanged"]))
    FORM_ROWS.append(dict(site=SITE, form="merged_%s" % comp, kind=BASELINE_KIND, rate_hz=1.0,
                          params=PARAMS, transfer_function=str(out), controls="", seed=None, status="made",
                          criterion="the whole record for the healthy row and the tipper, the %s window for "
                                    "the other" % comp, error="", days=None, n_runs=None, seconds=None))
if not MERGES:
    print("   no windowed transfer function was made, so nothing was merged")
'''),

("code", '''rows, bad = [], []
for comp in COMPONENTS:
    w = WINDOWS[comp]
    if not w["t_start"]:
        bad.append("%s: no window was proposed (%s)" % (comp, w["reason"][:80]))
        continue
    for name in ("window_%s" % comp, "window_%s_control" % comp):
        r = made_tf(name)
        if r is None:
            bad.append("%s was not made (%s)" % (name, str(rows_by_form().get(name, {}).get("error"))[:90]))
            continue
        rd = DL.reading(read(r["transfer_function"]), read(BASE) if BASE else None,
                        bands=[tuple(b) for b in DECADES], bar_band=tuple(BAR_BAND),
                        agree_band=tuple(AGREE_BAND))
        rows.append(dict(form=name, days=r.get("days"), bar=rd.get("bar"), bar_xy=rd.get("bar_xy"),
                         bar_yx=rd.get("bar_yx"), rho_ratio_xy=rd.get("rho_ratio_xy"),
                         rho_ratio_yx=rd.get("rho_ratio_yx"), phase_xy=rd.get("phase_diff_xy"),
                         per_decade=rd.get("per_decade", "")[:60]))
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows
      else "no windowed transfer function to read")
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
    print("VERDICT: UNJUDGED -- no window and block pair was scored")
else:
    promoted = [c for c, ok, _x, _y in beaten if ok]
    inconsistent = [c for c, ok, x, y in beaten if ok and not DL.beats(x, y, BAR_MARGIN)]
    if inconsistent:
        print("VERDICT: FAIL -- %s promoted without beating its random block on the %g-%g s bar by %.0f %%"
              % (", ".join(inconsistent), BAR_BAND[0], BAR_BAND[1], 100 * BAR_MARGIN))
    else:
        print("VERDICT: PASS -- every merge changed exactly its component's two rows (%s) and left the rest "
              "untouched, every window carries an equal-length random block drawn under its seed, and the "
              "%d promoted window(s) (%s) beat their block on the %g-%g s bar by at least %.0f %%: %s"
              % ("; ".join("%s -> %s" % (m["component"], ",".join(m["rows_changed"])) for m in MERGES),
                 len(promoted), ", ".join(promoted) or "none", BAR_BAND[0], BAR_BAND[1],
                 100 * BAR_MARGIN,
                 "; ".join("%s %.4f against %.4f%s" % (c, x, y, "" if ok else ", not promoted")
                           for c, ok, x, y in beaten)))
'''),

("md", r"""The windowed transfer function, its control and the merged file against the whole record. What to
look for is the merged curve following the whole record on the row the window did not touch and the
windowed curve on the row it did: that is the merge rule drawn, and a departure on the untouched row is the
failure the check above scores."""),

("code", '''curves = [("whole", read(BASE), "k", "-")] if BASE else []
styles = {"window": ("-", 0), "control": (":", 1), "merged": ("--", 2)}
for k, comp in enumerate(COMPONENTS):
    for name, key in (("window_%s" % comp, "window"), ("window_%s_control" % comp, "control"),
                      ("merged_%s" % comp, "merged")):
        r = made_tf(name)
        if r:
            ls, off = styles[key]
            curves.append((name, read(r["transfer_function"]), "C%d" % (3 * k + off), ls))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "09_window_transfer_functions.png",
                         title="%s: the window transfer functions and the merges" % SITE,
                         caption="Each component's windowed pass, the equal-length random block that "
                                 "sizes it, and the merged file that carries the window's two impedance "
                                 "rows into a copy of the whole record, all against the whole record in "
                                 "black. The merged curve is the one delivered: only that component's two "
                                 "rows differ from the black, and the tipper and the other row are the "
                                 "whole record's.", period_range=(1, 50000))
    WRITTEN.append(OUT / "09_window_transfer_functions.png")
    display(Image(filename=str(OUT / "09_window_transfer_functions.png")))
else:
    print("no windowed transfer function was made, so there is nothing to draw against the whole record")
'''),

("md", r"""## 4. The shared centre and the arm diagonal

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

The expected gain is L_N / L_E, not 1. With equal arms the ratio is 1 and the two forms coincide; with
unequal arms they do not: a site with arms of 9.0 m and 11.9 m reads a gain of 0.76 at a residual coherence
of 1.00, and 9.0/11.9 = 0.756, so its two lines share one voltage. The arm lengths come from sites.csv
dipole_n_m and dipole_e_m; an `assume:` cell is used and named, and a site with no lengths on file is
UNJUDGED on this criterion.

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
that theta. The x' row is the clean one; the y' row is kept for the record. The turn is checked on the
elements to 1e-6 relative, on the determinant to 1e-5 of the squared Frobenius norm -- the relative form is
meaningless at a near-singular period -- and on the Frobenius norm to 1e-6. The trace is not an invariant of
a column-only turn and its ratio is printed as the counter-example.

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
    print("   %s carries an assumed length, which the file's provenance carries with it"
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

("code", '''fig = FF.residual_panels(sv, SITE, CENTRE.get(SITE, {}), BUILT, OUT / "10_centre_residual.png",
                         days=CENTRE_DAYS, elines=ELINES, band_s=CE.BAND_S)
WRITTEN.append(OUT / "10_centre_residual.png")
display(Image(filename=str(OUT / "10_centre_residual.png")))
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

("code", '''fig = FF.diagonal_day(sv, SITE, OUT / "11_diagonal_day.png", elines=ELINES)
WRITTEN.append(OUT / "11_diagonal_day.png")
display(Image(filename=str(OUT / "11_diagonal_day.png")))
'''),

("md", r"""The diagonal pass and the baseline in one frame, where a diagonal was built. The two are drawn in
the diagonal frame: the pass on the NE cache already stands in it, and the whole-record baseline is turned
into it by Z' = R Z R^T and T' = T R^T at the same theta, its error bars combined in quadrature over both
turns. They are not two answers to compare. The NE cache holds the same two electric channels written in a
turned frame, passed against the same magnetics and the same reference, and an estimate that is linear in E
returns R Z, so the pass gives back the site's own tensor written in the diagonal frame. The robust
weighting is the one departure from that linearity: it scores each electric channel on its own residuals,
and the turned channels are not the two the baseline was weighted on, so the agreement is a fraction of a
per cent and not exact. What the diagonal buys is the error bar: the shared centre's voltage cancels out of
the x' row, which is the difference of the two arm potentials, and lands doubled on the y' row, which is
their sum.

What to look for is the two curves lying on each other on both rows, and the blue bars shorter than the
black on the x' row and longer on the y' row. The reading under the figure puts numbers on both over
100-1000 s, and names the largest departure it finds against the 1e-3 in rho a rotation on its own would
hold to."""),

("code", '''r = made_tf("diagonal")
if r and BASE:
    DIAG_TF = read(r["transfer_function"])
    BASE_D = CE.turn_tf(read(BASE), THETA)
    fig = FF.form_panels([("the baseline turned into the diagonal frame", BASE_D, "k", "-"),
                          ("the diagonal pass", DIAG_TF, "C0", "-")],
                         SITE, OUT / "12_diagonal_transfer_function.png", comps=("x'y'", "y'x'"),
                         title="%s: the diagonal pass and the baseline in the diagonal frame" % SITE,
                         caption="One tensor in one frame: the whole-record baseline turned into the "
                                 "diagonal frame in black, by Z' = R Z R^T at theta = %+.2f deg = "
                                 "atan2(-L_E, L_N) with L_N = %.4g m, L_E = %.4g m and d = %.2f m, and the "
                                 "pass on the arm-diagonal cache as it stands in blue -- the pass is the "
                                 "same transfer function written in that frame, not a second estimate of a "
                                 "different quantity, and what the diagonal buys is the error bar on the x' "
                                 "row, which the shared centre's voltage has cancelled out of."
                                 % (THETA, ARMS["L_N"], ARMS["L_E"],
                                    CE.diagonal_length(ARMS["L_N"], ARMS["L_E"])),
                         period_range=(1, 50000))
    WRITTEN.append(OUT / "12_diagonal_transfer_function.png")
    display(Image(filename=str(OUT / "12_diagonal_transfer_function.png")))

    P_D = np.asarray(DIAG_TF.period, float)
    IN_BAND = (P_D >= AGREE_BAND[0]) & (P_D <= AGREE_BAND[1])
    ROWS_D = (("x'", (0, 1)), ("y'", (1, 0)))
    print("the two estimates over %g-%g s, the diagonal pass against the turned baseline: a rotation of the "
          "same channels gives the same tensor, so the ratio reads 1.000 and the difference 0.0 deg up to "
          "the robust weighting" % AGREE_BAND)
    for row, (i, j) in ROWS_D:
        a, b = DIAG_TF.z[:, i, j], BASE_D.z[:, i, j]
        m = IN_BAND & np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        if not m.any():
            print("   the %s row: no period of the band carries both" % row)
            continue
        ratio = (np.abs(a[m]) / np.abs(b[m])) ** 2
        diff = np.degrees(np.angle(a[m] / b[m]))
        worst = int(np.argmax(np.abs(ratio - 1.0)))
        print("   the %s row over %d period(s): median rho ratio %.4f, median phase difference %+.3f deg; "
              "the largest departure is %.2f per cent in rho and %+.3f deg in phase, at %.0f s, which is %s "
              "the 1e-3 in rho a rotation on its own would hold to"
              % (row, int(m.sum()), np.median(ratio), np.median(diff), 100 * abs(ratio[worst] - 1.0),
                 diff[int(np.argmax(np.abs(diff)))], P_D[m][worst],
                 "within" if abs(ratio[worst] - 1.0) <= 1e-3 else "beyond"))
    print("the error bar over the same periods, the diagonal pass over the turned baseline: the x' row "
          "under 1 where the centre's voltage left it, the y' row above 1 where it landed")
    for row, (i, j) in ROWS_D:
        ea, eb = DIAG_TF.z_err[:, i, j], BASE_D.z_err[:, i, j]
        m = IN_BAND & np.isfinite(ea) & np.isfinite(eb) & (eb > 0)
        if not m.any():
            print("   the %s row: no period of the band carries both bars" % row)
            continue
        bar = ea[m] / eb[m]
        print("   the %s row over %d period(s): median ratio %.3f, range %.3f to %.3f"
              % (row, int(m.sum()), np.median(bar), np.min(bar), np.max(bar)))
else:
    print("no diagonal pass was built, so there is nothing to draw against the baseline")
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

("md", r"""## 5. The 10 Hz forms

A site recorded at 10 Hz carries a short end the 1 Hz cache cannot reach. This section passes the whole 10 Hz
record against the same reference kind the 1 Hz baseline used and draws the two rates on one page.

The reference is BASELINE_KIND at 10 Hz, which is the remote's own 10 Hz record on this site's grid, and
where no store has been built at that rate the form is refused with that reason and no pass is run. The
single station is not a kind of this package, so a missing store is not fallen back from: a transfer function
estimated on the site's own H is biased by its own noise and its error bars do not show the bias.

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
        # than falling back to one: an estimate on the site's own H is biased by its own noise
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
that says what the 10 Hz pass estimated. What to look for is the dashed curve lying on the solid one where
the two rates overlap, the size of the error bars at the short end, and the step across the join line: the
two shaded bands are the ones a splice would score that step on, and the octave between them is the guard
that holds the logger's own instrument line and is scored by nothing."""),

("code", '''rows = rows_by_form()
curves = [("whole, 1 Hz", read(BASE), "k", "-")] if BASE else []
r = made_tf("whole10")
if r:
    curves.append(("whole10, 10 Hz", read(r["transfer_function"]), "C0", "--"))
if len(curves) > 1:
    _bs = AR.bands_for(10)
    _k10 = rows.get("whole10", {}).get("kind", BASELINE_KIND)
    _cost = "%s s over %s run(s)" % (rows.get("whole10", {}).get("seconds"),
                                     rows.get("whole10", {}).get("n_runs"))
    # a form the floor refused carries its sentence where its curve would have been
    _gone = " ".join("%s is not drawn -- %s." % (n, w) for n, w in REFUSED)
    fig = FF.rate_panels(curves, SITE, OUT / "13_rates.png", join_s=SP.SPLICE_JOIN_S,
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
    WRITTEN.append(OUT / "13_rates.png")
    display(Image(filename=str(OUT / "13_rates.png")))
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

("md", r"""## 6. Replacement magnetics and the lender

At long period the horizontal magnetic field is homogeneous over the site spacing, so a neighbour's H
measures the same field; at short period it is not. A borrowed long end therefore goes under the site's own
short periods and the two are spliced where borrowing stops costing.

The rule picks the worse channel only, and only where it is clearly worse than the other: a site
decorrelated from its neighbours for any reason -- distance, a quiet spell, its own noise -- has both
channels below any threshold, so a rule reading "any channel below a threshold" replaces both and wrecks the
site. The coherence is the whole-record mean over 100-1000 s, deliberately not the chunk median used
elsewhere, and both series are despiked first, because a whole-record mean has no median to hide behind and
one logger spike sets the number. The candidate must itself be sound against a third site at 0.80, so that the site it stands in for is not
its only witness. A donor that is a member of the reference the pass reads is refused, because a reference
sharing a channel with the local H is comparing a channel with itself.

Four forms. A is the site's own H, the baseline of section 1. B replaces one channel with the nearest sound
site's same channel and keeps the site's own other channel. C replaces it with the fleet stack's channel; it is
drawn for comparison and not delivered, because a stack is a reference and never a local H, and its pass
reads the observatory, since every member of the stack is inside its own local H. D borrows the whole pair:
the tensor of the site's E on the field a neighbour measured, which is an inter-site impedance, drawn for
comparison and not delivered. D and the lender form are one construction and one pass.

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
fig = FF.candidate_bars(CANDS, SITE, OUT / "14_candidates.png", threshold=RP.SUB_THRESHOLD,
                        donor_gate=RP.DONOR_GATE)
WRITTEN.append(OUT / "14_candidates.png")
display(Image(filename=str(OUT / "14_candidates.png")))
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
    REPLACED.append(dict(name="replace_%s_%s" % (CHAN, LEND), spec=b, comparison_only=False, lender=LEND,
                         kind=BASELINE_KIND,
                         why="scored per decade against the own-H baseline, the worst decade governing"))
    d = RP.replace_channel(local_b, lend_pair, CHAN, ANGLE, whole_pair=True)
    print("form D: %s" % d["note"])
    REPLACED.append(dict(name="lender_%s" % LEND, spec=d, comparison_only=True, lender=LEND,
                         kind=BASELINE_KIND,
                         why="drawn for comparison; not a transfer function: an inter-site impedance"))
    try:
        _st0, sh, smask, _si = REF.load_reference("stack", SITE, 1, WORK)
        stack_pair = {c: np.where(np.asarray(smask, bool)[:n_rec], np.asarray(sh[c], float)[:n_rec], np.nan)
                      for c in ("Hx", "Hy")}
        c_ = RP.replace_channel(local_b, stack_pair, CHAN, ANGLE)
        print("form C: %s, the channel taken from the fleet stack; its pass reads the observatory, because "
              "every member of the stack is inside its own local H" % c_["note"])
        REPLACED.append(dict(name="replace_%s_stack" % CHAN, spec=c_, comparison_only=True, lender=None,
                             kind="obs",
                             why="drawn for comparison; not a transfer function: a stack is a reference, "
                                 "never a local H"))
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
    r["inter_site"] = bool(item["comparison_only"] or item["spec"]["inter_site"])
    r["borrowed"] = "Hx+Hy" if all(item["spec"]["borrowed"]) else (CHAN or "")
'''),

("code", '''rows = []
for item in REPLACED:
    r = made_tf(item["name"])
    if r is None:
        rows.append(dict(form=item["name"],
                         error=str(rows_by_form().get(item["name"], {}).get("error"))[:110]))
        continue
    tf = read(r["transfer_function"])
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
C the stack's channel and D the whole borrowed pair, the two comparison forms drawn dashed. What to look for
is B following A while D sits off it: a single borrowed channel is a measurement of the same field, and a
whole borrowed pair is the inter-site impedance, which is a different quantity."""),

("code", '''curves = [("A: whole, own H", read(BASE), "k", "-")] if BASE else []
for j, item in enumerate(REPLACED):
    r = made_tf(item["name"])
    if r:
        curves.append(("%s%s" % (item["name"], " (comparison)" if item["comparison_only"] else ""),
                       read(r["transfer_function"]), "C%d" % j, "--" if item["comparison_only"] else "-"))
if len(curves) > 1:
    fig = FF.form_panels(curves, SITE, OUT / "15_forms_abcd.png",
                         title="%s: the replacement forms against the own-H baseline; the dashed forms are "
                               "drawn for comparison and are not transfer functions" % SITE,
                         period_range=(1, 50000))
    WRITTEN.append(OUT / "15_forms_abcd.png")
    display(Image(filename=str(OUT / "15_forms_abcd.png")))
else:
    print("no replacement form was built, so there is nothing to draw against the baseline")
'''),

("code", '''SPLICE = {}
b_name = "replace_%s_%s" % (CHAN, LEND) if (CHAN and LEND) else ""
r = made_tf(b_name) if b_name else None
if BASE and r:
    a, b = read(BASE), read(r["transfer_function"])
    bg = AG.on_grid(a, b)
    for comp in ("xy", "yx"):
        ra, _ea, pa, _fa = TFN.rho_phase(a.period, a.z, a.z_err, comp)
        rb, _eb, pb, _fb = TFN.rho_phase(a.period, bg.z, bg.z_err, comp)
        SPLICE[comp] = RP.splice_period(a.period, rb, pb, ra, pa)
    print("the period from which the borrowed row stops costing, read against the own-H baseline:")
    for comp, s in SPLICE.items():
        print("   %s: t_c %s s over %d tested period(s), %.0f %% agreeing -- %s"
              % (comp, ("%.0f" % s["t_c"]) if np.isfinite(s["t_c"]) else "none", s["n_tested"],
                 100 * s["frac_agree"], s["rule"]))
    print("   below t_c the site's own short periods are kept and above it the borrowed row is delivered; a "
          "form with no t_c has no defensible splice and is not merged")
else:
    print("no single-channel replacement to splice")
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

("md", r"""## 7. The tipper-only delivery

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
    out = OUT / FM.tf_name(SITE, "tipper_only", BASELINE_KIND, 1, PARAMS)
    got = DL.tipper_only(BASE, out, KIND_WORD.get(BASELINE_KIND, BASELINE_KIND), TIP)
    print(got)
    if got.get("written"):
        FORM_ROWS.append(dict(site=SITE, form="tipper_only", kind=BASELINE_KIND, rate_hz=1.0, params=PARAMS,
                              transfer_function=str(out), controls="", seed=None, status="made",
                              criterion="an H-only delivery; the impedance rows carry no transfer function "
                                        "of record",
                              error="", days=None, n_runs=None, seconds=None))
else:
    print("neither line is dead on most days (%s) and the tipper is not refused, so the tipper-only "
          "delivery does not apply here: the tipper ships inside the impedance files"
          % (", ".join(dead_lines) or "no line dead"))
'''),

("md", r"""## 8. The recipe

The sections above each change one thing and score it. This one composes them: a frame for the transfer
function and, per impedance row, which hours that row is estimated on and at which sample rate. One row can
come from the whole record at 1 Hz and the other from a short stretch at 10 Hz, and the two are assembled
into one tensor.

The cell below is the one to edit. `frame` is the frame both rows are passed in, so nothing is turned after
the assembly: `native` is the site's own sensor frame after the mean-field rotation, `diagonal` is section
4's arm diagonal. `hours` of a row is one of

| hours | what it selects |
|---|---|
| `whole` | the record |
| `window:coherent` | the stretch the rule of section 3 chooses for that row's own line: the longest contiguous run of whole UTC hours in which that line reads above `WINDOW_COH` = 0.5 against the magnetic channel it couples to over 20-200 s, cut to its best `STRETCH_MAX_H` = 48 h where it runs longer |
| `window:<start UTC>/<hours>` | a stretch named in the cell, as an ISO UTC start and a count of hours |

A row's stretch is read on that row's own recorded line -- the x row on Ex against Hy, the y row on Ey
against Hx, both from the site's own 1 Hz cache as laid -- and never on the frame's variant of them, whatever
frame the rows are passed in: a stretch exists to find where that electrode was measuring, which is a fact
about the electrode and not about the frame its voltage is later combined in. It is read at 1 Hz whatever a
row's rate, because 200 s is measured on the long record. Every stretch carries a control at the same cost: a
stretch of the same length placed at random elsewhere in the record under the seed, not overlapping the
selection.

A row at 10 Hz needs the 10 Hz cache of the site, the frame's variant of that cache, and the reference store
of BASELINE_KIND at 10 Hz, which is the remote's own 10 Hz record on this site's grid. Where one of the three
is not on disk the row is refused with the reason and no pass is run, by the rule of section 5: the single
station is not a kind of this package, so a missing store is not fallen back from.

The assembly. The x row supplies Zx'x' and Zx'y' and the y row Zy'x' and Zy'y'. The x row's file is the base,
so its period grid, its station block and its position carry through, and the y row is matched onto that grid
where the two grids are one and interpolated in log period onto it where they are not. A period the y row's
own grid does not reach carries the EDI empty-data value 1e32, which `transfer_functions.read_tf` masks per
component, so the assembled file reads back with that row empty above the y row's longest period rather than
carrying a number nothing measured. A y row at 10 Hz reaches below the x row's shortest period, and those
periods are not carried either: one file holds one grid, and the grid is the x row's. The tipper is the named
row's. The header carries one `recipe=` line per row with the frame, the hours, the rate, the stretch and the
control's seed.

Where a row asks for what a section above has already built -- the whole record at 1 Hz in the native frame
is the baseline, and in the diagonal frame it is section 4's `diagonal` form -- that file is taken and no
second pass of the same specification is run.

What this composes at a site whose two lines share a noisy centre electrode: the x' row and the tipper from
the arm diagonal over the whole record at 1 Hz, and the y' row from that row's own stretch at 10 Hz against
the same remote. At the shipped Q53N the recipe is frame native with both rows the whole record at 1 Hz,
which is the baseline itself: the section is there to be edited."""),

("code", '''# ---- the recipe: one composition of the sections above, one row at a time ----
RECIPE = dict(frame="native", x=dict(hours="whole", rate=1), y=dict(hours="whole", rate=1), tipper="x")
# frame     native | diagonal (section 4's arm diagonal); both rows are passed in it
# x, y      the x' and y' rows (x and y where the frame is native). hours is
#           whole | window:coherent | window:<start UTC>/<hours>; rate is 1 or 10
# tipper    which row's pass the tipper is taken from: x | y
'''),

("md", r"""The frame, the variant cache each rate needs, the stretch the rule finds on each row's own line,
and what each row resolves to. The whole-record selection -- the longest stretch with both lines above the
threshold at once -- is printed beside them, because that is the selection a window recorded for the site by
hand is scored against below. The five longest runs are printed with it, so the margin between the longest
and the next is visible: a rule that picks a 14 h stretch over an 11 h one is a different claim from one that
picks 14 h over 3 h."""),

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
WHOLE_STRETCH = SEL.longest_stretch(HOUR_TABLE, coh_min=SEL.WINDOW_COH, lines=SEL.LINES,
                                    max_h=SEL.STRETCH_MAX_H)
print()
print("the rule over the %d whole UTC hour(s) of the %g Hz record scored in section 3"
      % (len(HOUR_TABLE), SEL.SCORE_RATE))
for _key in ("x", "y"):
    _s = SEL.longest_stretch(HOUR_TABLE, coh_min=SEL.WINDOW_COH, lines=(SEL.ROW_LINE[_key],),
                             max_h=SEL.STRETCH_MAX_H)
    print("   %s row on %s: %s" % (RC.ROW_LABEL[_key], SEL.ROW_LINE[_key], _s["reason"]))
print("   both lines:  %s" % WHOLE_STRETCH["reason"])
for _h, _a, _b in WHOLE_STRETCH["runs"][:5]:
    print("      %4d h  %s .. %s UTC" % (_h, pd.Timestamp(_a, unit="s"), pd.Timestamp(_b, unit="s")))
'''),

("code", '''ROWS = {}
for key in ("x", "y"):
    spec = RECIPE[key]
    hours, rate = str(spec.get("hours", "whole")), int(spec.get("rate", 1))
    row = dict(row=key, hours=hours, rate=rate, seed=SEED + (0 if key == "x" else 1), window=None,
               control_window=None, control=None, line=None, spans=[], control_spans=[],
               keep_name="the record", control_keep_name="", reuse="", transfer_function="",
               refused=RC.missing_inputs(sv, SITE, rate, FRAME["variant"], BASELINE_KIND))
    got = None
    if not row["refused"]:
        try:
            got = RC.row_hours(sv, SITE, hours, row=key, seed=row["seed"], table=HOUR_TABLE,
                               coh_min=SEL.WINDOW_COH, max_h=SEL.STRETCH_MAX_H)
        except ValueError as exc:
            row["refused"] = "refused: %s" % exc
    if got and got["window"] is not None:
        w, c = got["window"], got["control"]
        days = ((w["t_end"] - w["t_start"]) / 86400.0) if w.get("t_start") else 0.0
        if not w.get("t_start"):
            row["refused"] = "refused: %s" % w["reason"]
        elif days < FM.min_window_days(sv):
            row["refused"] = ("refused: the stretch is %.2f d, under the %g d floor in survey.yaml"
                              % (days, FM.min_window_days(sv)))
        else:
            row.update(window=w, control_window=c, control="recipe_%s_control" % key, line=got["line"],
                       spans=[(w["t_start"], w["t_end"])], control_spans=[(c["t_start"], c["t_end"])],
                       keep_name="%s on %s, %.2f h" % (hours, got["line"],
                                                       (w["t_end"] - w["t_start"]) / 3600.0),
                       control_keep_name="a stretch of the same length elsewhere in the record, seed %d"
                                         % row["seed"])
    if hours == "whole":
        row["spans"] = [(t0_rec, t0_rec + int(n_rec))]
        reuse = {"native": "whole", "diagonal": "diagonal"}[FRAME["frame"]]
        made = made_tf(reuse) if rate == 1 else None
        if made:
            row.update(reuse=reuse, transfer_function=str(made["transfer_function"]))
    ROWS[key] = row
cols = ["row", "hours", "rate", "line", "seed", "window_start", "window_end", "days", "control_start",
        "control", "reuse", "refused"]
print(pd.DataFrame([dict(
    row="%s (%s)" % (r["row"], RC.ROW_LABEL[r["row"]]), hours=r["hours"], rate=r["rate"],
    line=(r["line"] or ""), seed=r["seed"],
    window_start=(pd.Timestamp(r["window"]["t_start"], unit="s") if r["window"] else ""),
    window_end=(pd.Timestamp(r["window"]["t_end"], unit="s") if r["window"] else ""),
    days=round(sum(b - a for a, b in r["spans"]) / 86400.0, 3),
    control_start=(pd.Timestamp(r["control_window"]["t_start"], unit="s") if r["control_window"] else ""),
    control=(r["control"] or ""), reuse=(r["reuse"] or ""), refused=r["refused"][:70])
    for r in ROWS.values()])[cols].to_string(index=False))
for r in ROWS.values():
    print("   %s: %s" % (r["row"], r["keep_name"]))
'''),

("md", r"""The two hourly coherence series the rule reads, with each row's stretch drawn above them and its
control beneath it. What to look for is each row's stretch sitting where that row's own series is above the
line -- the x row is chosen on Ex with Hy alone and the y row on Ey with Hx alone, so the two rows need not
agree -- and each control landing somewhere its series is neither obviously better nor worse: the two cost
the same number of hours, so the difference between their transfer functions is what the stretch bought."""),

("code", '''spans = []
for key in ("x", "y"):
    r = ROWS[key]
    spans.append(("%s row: %s" % (RC.ROW_LABEL[key], r["hours"]), FF.SPAN_COLOUR[RC.ROW_COMPONENT[key]],
                  None, r["spans"][:400]))
    if r["control_spans"]:
        spans.append(("%s control, seed %d" % (RC.ROW_LABEL[key], r["seed"]), "0.4", "//",
                      r["control_spans"][:400]))
fig = FF.recipe_spans(HOUR_TABLE, spans, SITE, OUT / "16_recipe_windows.png", coh_min=SEL.WINDOW_COH,
                      title="%s: the recipe's stretches over the coherence they were chosen on" % SITE,
                      caption="The squared coherence of each recorded electric line with the magnetic field "
                              "it couples to, over %g-%g s at %d s Welch segments, one value per whole UTC "
                              "hour on the %g Hz cache as laid, with the %.2f line drawn. Above each panel: "
                              "the stretch each row of the recipe is estimated on, and beneath it that row's "
                              "control at the same cost. A row's stretch is the longest contiguous run of "
                              "hours with that row's own line above the threshold, cut to its best %d h "
                              "where it runs longer; with both lines named at once, %d hour(s) of %d scored "
                              "are above it here, in %d run(s)."
                              % (SEL.SCORE_BAND_S[0], SEL.SCORE_BAND_S[1], SEL.SCORE_SEGMENT_S,
                                 SEL.SCORE_RATE, SEL.WINDOW_COH, SEL.STRETCH_MAX_H,
                                 WHOLE_STRETCH["n_hours_above"], WHOLE_STRETCH["n_hours_scored"],
                                 WHOLE_STRETCH["n_runs"]))
WRITTEN.append(OUT / "16_recipe_windows.png")
display(Image(filename=str(OUT / "16_recipe_windows.png")))
'''),

("md", r"""One pass per row and one per control, each measured against the reference before it is run: the
mask, the reference's own coverage and the record cut into runs at the 3,600 s floor. A row whose cache
leaves nothing is refused there with those numbers rather than crashing the estimator, and a row whose inputs
are not on disk was refused above. The assembled file is then written from the two rows."""),

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
                              params=PARAMS, transfer_function=row["transfer_function"], controls="", seed=None,
                              status="exists", days=None, n_runs=None, seconds=None, error="",
                              criterion=crit + "; the %s form's file, which is the same specification, "
                                               "and no second pass of it is run" % row["reuse"]))
        print("   %-20s takes the %s form's file: %s"
              % (name, row["reuse"], Path(row["transfer_function"]).name))
        continue
    jobs = [(name, row["window"], row["keep_name"], crit, [row["control"]] if row["control"] else [])]
    if row["control"]:
        jobs.append((row["control"], row["control_window"], row["control_keep_name"],
                     "the control of the %s row at the same cost" % RC.ROW_LABEL[key], []))
    for job_name, window, keep_name, criterion, controls in jobs:
        extra = SEL.stretch_mask(t0_rec, n_rec, window, float(SEL.SCORE_RATE)) if window else None
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
             keep_name=keep_name, seed=row["seed"],
             controls=controls, criterion=criterion,
             redo=bool(REDO or (NE_CACHE.get(row["rate"]) or {}).get("refreshed")),
             extra_lines=RC.recipe_lines(RECIPE, ROWS))
'''),

("code", '''ASSEMBLED, ROW_READINGS = None, []
for key in ("x", "y"):
    r = made_tf("recipe_%s" % key)
    if r is None:
        continue
    comp = RC.ROW_COMPONENT[key]
    ctrl_name = ROWS[key]["control"]
    ctrl = made_tf(ctrl_name) if ctrl_name else None
    bar_row = DL.bar(read(r["transfer_function"]), comp, *BAR_BAND)
    bar_ctrl = DL.bar(read(ctrl["transfer_function"]), comp, *BAR_BAND) if ctrl else np.nan
    # each row on its own element and never on the assembled file's bar, which is the better of the two rows
    # and would answer for the row a stretch was not chosen for
    ROW_READINGS.append(dict(row=RC.ROW_LABEL[key], form="recipe_%s" % key, element="Z%s" % comp,
                             hours=ROWS[key]["hours"], rate_hz=ROWS[key]["rate"], days=r.get("days"),
                             bar=bar_row, control=(ctrl_name or "none"), control_bar=bar_ctrl,
                             promoted=(DL.beats(bar_row, bar_ctrl, BAR_MARGIN)
                                       if np.isfinite(bar_ctrl) else None)))
rx, ry = made_tf("recipe_x"), made_tf("recipe_y")
if rx and ry:
    for key, r in (("x", rx), ("y", ry)):
        ROWS[key]["transfer_function"] = str(r["transfer_function"])
    out = OUT / FM.tf_name(SITE, "recipe", BASELINE_KIND, ROWS["x"]["rate"], PARAMS)
    ASSEMBLED = RC.assemble(rx["transfer_function"], ry["transfer_function"], out, tipper=RECIPE["tipper"],
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
                   "transfer function carries no evidence of its own")
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
                          rate_hz=float(ROWS["x"]["rate"]), params=PARAMS, transfer_function=str(out), controls="",
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
    print("   the assembled transfer function is %s" % verdict)
else:
    print("no assembled transfer function: the %s row was not made"
          % ", ".join(k for k in ("x", "y") if not made_tf("recipe_%s" % k)))
'''),

("md", r"""The assembled transfer function against the whole-record baseline and against each row's control.
What to look for is the x' panels lying on the curve the row they came from drew, the y' panels stopping at
the dash-dotted line -- the longest period that row reaches, above which the file carries the empty value --
and each control sitting off the row it controls by more than its error bars, which is what says the stretch
was worth choosing."""),

("code", '''curves = [("whole, the baseline", read(BASE), "k", "-")] if BASE else []
if ASSEMBLED:
    curves.append(("recipe", read(ASSEMBLED["path"]), "C0", "-"))
for j, key in enumerate(("x", "y")):
    r = made_tf("recipe_%s_control" % key)
    if r:
        curves.append(("%s control" % RC.ROW_LABEL[key], read(r["transfer_function"]), "C%d" % (j + 1), ":"))
if ASSEMBLED and len(curves) > 1:
    fig = FF.recipe_transfer_function(curves, SITE, OUT / "17_recipe_transfer_function.png",
                            join_s=ASSEMBLED["y_max_s"],
                            title="%s: the assembled recipe against the baseline and the controls" % SITE,
                            caption="The assembled transfer function in the %s frame: the first row from the "
                                    "x row's pass (%s at %g Hz) and the second from the y row's (%s at "
                                    "%g Hz), with the tipper from the %s row, against the whole-record 1 Hz "
                                    "baseline in black and against each row's control at the same cost, "
                                    "dotted. %s. The dash-dotted line at %.4g s is the longest period the y "
                                    "row reaches; above it that row carries the EDI empty value and nothing "
                                    "is drawn. The two panels are two rows of one tensor and not two "
                                    "estimates of one quantity, so they are not expected to lie on each "
                                    "other, and where the recipe's frame is not the site's own the black "
                                    "curve is a different quantity again and is drawn for scale alone."
                                    % (FRAME["frame"], ROWS["x"]["hours"], ROWS["x"]["rate"],
                                       ROWS["y"]["hours"], ROWS["y"]["rate"], RECIPE["tipper"],
                                       ASSEMBLED["how"], ASSEMBLED["y_max_s"]))
    WRITTEN.append(OUT / "17_recipe_transfer_function.png")
    display(Image(filename=str(OUT / "17_recipe_transfer_function.png")))
else:
    print("no assembled transfer function, so there is nothing to draw against the baseline")
'''),

("md", r"""**Where `survey.yaml` `checks.recipe_window` records a window for this site, this check fails if
the rule's selection over both lines does not land within `tolerance_h` of it at each end.** The rule the
package writes has to reproduce the window the campaign reached by hand, and a hand-chosen window is one
window for the site, so it is scored against the selection over both lines and not against either row's own
stretch. Where no window is on record the cell prints the stretch the rule found as a reading and writes no
verdict."""),

("code", '''want_all = (sv.cfg.get("checks") or {}).get("recipe_window") or {}
tol_h = float(want_all.get("tolerance_h", 1))
want = want_all.get(SITE) or {}
found = (("the rule found %d h, %s to %s UTC"
          % (WHOLE_STRETCH["hours"], pd.Timestamp(WHOLE_STRETCH["t_start"], unit="s"),
             pd.Timestamp(WHOLE_STRETCH["t_end"], unit="s"))) if WHOLE_STRETCH["t_start"]
         else ("the rule found no stretch: %s" % WHOLE_STRETCH["reason"]))
if not want:
    # a reading and no verdict: survey.yaml records no window for this site, so the criterion has nothing
    # to be scored against and a verdict line here would be a judgement on nothing
    print("%s. survey.yaml checks.recipe_window records no window for %s, so this check is not scored here "
          "and the stretch above is a reading. The sites it is scored at are %s"
          % (found, SITE, ", ".join(sorted(k for k in want_all if k != "tolerance_h")) or "none"))
elif not WHOLE_STRETCH["t_start"]:
    print("VERDICT: FAIL -- %s, where the record carries %s to %s UTC (%s)"
          % (found, want["t_start"], want["t_end"], str(want.get("source"))[:110]))
else:
    a = int(pd.Timestamp(want["t_start"]).timestamp())
    b = int(pd.Timestamp(want["t_end"]).timestamp())
    da, db = abs(WHOLE_STRETCH["t_start"] - a) / 3600.0, abs(WHOLE_STRETCH["t_end"] - b) / 3600.0
    if max(da, db) > tol_h:
        print("VERDICT: FAIL -- %s, which misses the recorded %s to %s UTC by %.2f h at the start and "
              "%.2f h at the end, beyond the %g h the check allows (%s)"
              % (found, want["t_start"], want["t_end"], da, db, tol_h, str(want.get("source"))[:110]))
    else:
        print("VERDICT: PASS -- %s, within %.2f h and %.2f h of the recorded %s to %s UTC, both inside the "
              "%g h the check allows; the next longest stretch the rule found is %s (%s)"
              % (found, da, db, want["t_start"], want["t_end"], tol_h,
                 ("%d h" % WHOLE_STRETCH["runs"][1][0]) if len(WHOLE_STRETCH["runs"]) > 1 else "none",
                 str(want.get("source"))[:110]))
'''),

("md", r"""**This check fails if any row's pass ended in an exception rather than a stated refusal.** A row
refused before its pass, with its runs and its days measured first, is a reading and not a failure; a row
that ended in a traceback is a failure with the row named.

Each row against its own control is a reading and not a limb of the criterion, as in section 3: a row that
does not beat its control by BAR_MARGIN bought efficiency and not a different answer, and reads
`not promoted`. The comparison is per row and at the same cost -- the x' row against the x' row's control on
Zx'y', the y' row against the y' row's on Zy'x' -- because the assembled file's bar is the better of its two
rows and would answer for the row a stretch was not chosen for. Both bars are printed either way, and the
same readings decide whether the assembled transfer function is a candidate in section 9's table."""),

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
        elif made_tf(name) is None:
            note.append("%s carries no file on disk" % name)
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
          "assembled transfer function against the baseline, every element over the whole band")
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

("md", r"""## 9. The forms table

Every form with the transfer functions it is read against, the criterion in words, the verdict and the
reading. This is the file workbook 05 reads. A form is a candidate only where it beats every control it
carries on the 10-1000 s bar by BAR_MARGIN. A form with no control is read and never promoted on this table
alone, and a form that borrows both horizontal channels is never a candidate. Section 8's assembled `recipe`
row is read the same way through the rows it was assembled from, because its controls belong to them: it is a
candidate where at least one of those rows carries a control, where every row that carries one beat it by
BAR_MARGIN on its own element, and, in the diagonal frame, where section 4's shared-centre model holds at
this site; its `verdict` cell states which of the three decided it. A recipe whose rows are all the whole
record carries no control at all and is read and not promoted, like any other form with none.

The decisions.csv cells this workbook proposes are printed below and are not written unless WRITE_DECISIONS
is True. Decisions are the analyst's."""),

("code", '''REFUSED = [r for r in FORM_ROWS if not FM.is_form_name(r.get("form"))]
if REFUSED:
    print("refused, not carried into the table: %d row(s) name a form no section of this workbook makes "
          "(%s); the set is auslamp_proc.site.forms.FORM_NAMES"
          % (len(REFUSED), ", ".join(sorted({str(r.get("form")) for r in REFUSED}))))
FORM_ROWS[:] = [r for r in FORM_ROWS if FM.is_form_name(r.get("form"))]
TABLE = DL.forms_table(FORM_ROWS, OUT / "forms.csv", baseline_path=BASE, bar_band=tuple(BAR_BAND),
                       agree_band=tuple(AGREE_BAND), margin=BAR_MARGIN)
WRITTEN.append(OUT / "forms.csv")
cols = ["form", "kind", "rate_hz", "status", "controls", "bar_10_1000", "control_bar", "rho_ratio_xy",
        "phase_diff_xy", "rho_ratio_yx", "phase_diff_yx", "verdict", "candidate"]
print(TABLE[cols].round(4).to_string(index=False))
print()
print("%d form(s), %d made, %d candidate(s) for workbook 05: %s"
      % (len(TABLE), int((TABLE.status == "made").sum()), int(TABLE.candidate.sum()),
         " ".join(TABLE[TABLE.candidate].form) or "none"))
FM.write_run_provenance(sv, SITE, OUT, RUN, OUT.name.split("_", 1)[-1], 1, PARAMS, BASELINE_KIND, FORM_ROWS)
print("provenance.json rewritten with %d form entries" % len(FORM_ROWS))
'''),

("md", r"""The same table as a figure: one bar per form on the 10-1000 s bar, its control marked as a red
diamond beside it, a candidate in green. What to look for is a green bar well to the left of its diamond --
that gap is the 20 per cent margin -- and how many forms sit on top of the baseline, which is the reading
that the record was already being used for what it is worth."""),

("code", '''fig = FF.forms_bars(TABLE, SITE, OUT / "18_forms_bars.png", band=tuple(BAR_BAND), margin=BAR_MARGIN)
WRITTEN.append(OUT / "18_forms_bars.png")
display(Image(filename=str(OUT / "18_forms_bars.png")))
'''),

("code", '''evidence = []
for c in COMPONENTS:
    bars = ";".join("%s %.4f" % (r.form, r.bar_10_1000) for r in TABLE.itertuples()
                    if str(r.form).startswith("window_%s" % c) and np.isfinite(r.bar_10_1000))
    evidence.append(dict(component=c, record_days=round(n_rec / 86400.0, 2), in_use=False,
                         evidence="control: a block of the same length placed at random elsewhere, "
                                  "seed %d; %g-%g s bar %s"
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

("md", r"""## 10. What was written

The run folder is also read back for strays: an EDI whose form name is outside the set this workbook can
produce, which is a form left by a workbook that has since been cut. It is named here and read by nothing --
workbook 05 skips it and names it in its own first cell -- because the section that made it took with it the
criterion the form was judged on, and a form with no criterion cannot be read against one."""),

("code", '''rows = []
for p in list(WRITTEN) + sorted(OUT.glob("*")):
    p = Path(p)
    if p.exists() and p.is_file():
        rows.append(dict(file=str(p), kb=round(p.stat().st_size / 1024, 1)))
files = pd.DataFrame(rows).drop_duplicates("file").sort_values("file")
print("%d file(s), %.1f MB, in %.1f minutes" % (len(files), files.kb.sum() / 1024, (time.time() - T0) / 60))
print(files.to_string(index=False))
print()
STRAYS = FM.strays(OUT, [SITE])
if STRAYS:
    print("%d stray transfer function(s) in the run folder: a form no section of this workbook makes, read "
          "by nothing here and skipped by workbook 05" % len(STRAYS))
    for p, name in STRAYS:
        print("   %-14s %s (%.1f kB)" % (name, p.name, p.stat().st_size / 1024))
else:
    print("no stray transfer function in the run folder: every EDI here carries a form this workbook makes")
print()
cost = pd.DataFrame([dict(form=r.get("form"), rate_hz=r.get("rate_hz"), status=r.get("status"),
                          days=r.get("days"), runs=r.get("n_runs"), seconds=r.get("seconds"),
                          error=str(r.get("error") or "")[:60]) for r in FORM_ROWS])
print("the forms and what each cost")
print(cost.to_string(index=False))
'''),
]
