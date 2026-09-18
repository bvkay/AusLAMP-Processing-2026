# AusLAMP-Processing-2026

Long-period magnetotelluric processing for the AusLAMP surveys recorded on Earth Data Logger PR6-24 (with a
Bartington Mag-03 fluxgate, 10 Hz) or LEMI-424 (1 Hz) instruments, as five Jupyter workbooks on one Python
package. The workbooks take a survey from a folder of raw time series to one transfer function per site;
every function lives in `auslamp_proc`, so the batch entry and a single-site run share one code path and any
cell can be re-run. Built on the IAGA-DVI-DataStandards packages mt-metadata, mt-io and mth5, with Aurora as
the estimator.

The worked example is AusLAMP Queensland Phase 1 (23 EDL sites, September-November 2025), the survey the
workbooks open on. The raw time series are not in this repository: point `raw_root` and `work_root` in
`surveys/<survey>/survey.yaml` at where they live on your machine. The survey folders of Phases 2 (18 sites,
March-May 2026) and 3 (15 sites, June-July 2026) are included as filled-in examples of survey.yaml, sites.csv
and decisions.csv; their raw is not shipped either. To run your own survey, copy `surveys/_template/` and
point it at your own raw folder; `surveys/_template/SITES_COLUMNS.md` says what every cell of every table
means and which workbook writes it.

Author: Ben Kay (bvkay). Started 2026-09-13; re-cut for the workbook layout 2026-09-16.

## Install

    conda env create -f environment.yml
    conda activate auslamp-processing-2026
    python -m ipykernel install --user --name auslamp-processing-2026 --display-name "Python (auslamp-processing-2026)"

The readers are the released packages (mt-io 0.0.5, mth5 0.6.9, mt-metadata 1.0.11); nothing depends on a fork.

Launch the workbooks as `python -m jupyterlab`, and run the runner as `python generator/run_workbooks.py 01`,
with the environment's own interpreter. The `jupyter` dispatcher resolves its subcommands from PATH, which on
a machine with a base Anaconda install is a different nbconvert from the environment's, so `python -m jupyter
...` silently escapes the environment where `python -m jupyterlab` and `python -m nbconvert` do not.

## The workbooks

Workbook 01 is survey-wide; workbooks 02 to 05 work on one site at a time, named in each one's `SITE`
parameter, because one site is processed at a time.

| workbook | what it does |
|---|---|
| `01_survey` | reads the raw folder into a site table (instrument, serial, dipoles, positions from the logger's own GPS, dates with the EDL week rollover, declination), draws the map and the deployment register, and checks or fetches the observatory record for the span |
| `02_records` | builds each chosen site's cache quietly (every file placed on one absolute axis, gaps NaN, no sign or frame) and draws one site's record, band coherence, coherence maps, spectra and spectrograms, with the magnetometer DC test and the per-day state of each electric line -- look before processing |
| `03_site` | one site: its frame, its clean-pool row, its remote site, its fleet stack, the observatory, the references written, the bands, the MTH5 check, Aurora at 1 Hz over the four reference kinds and at 10 Hz on the longest coherent stretch with a control of the same length, then every transfer function of the site on one page with kind against kind, rate against rate and run against run beneath it |
| `04_site` | one site in depth: the magnetics day by day, the daily and clock tests, a window per impedance row with its control, the arm diagonal for a shared centre, the 10 Hz short end, a magnetic channel borrowed from a neighbour, and the recipe that composes them -- one frame and, per impedance row, which hours at which rate -- into one transfer function |
| `05_final` | one site's final transfer function: every one the site has put to the three response tests (phase in quadrant, the slope bound, the error bar), the rule's proposal, the analyst's choice recorded, the 10 Hz join, one EDI per site |

A survey is processed by the batch entry, which runs workbook 03's own code path one lane per site:

    python -m auslamp_proc.process.batch --survey queensland_phase1 --sites all --lanes 3

`--redo --sites Q49 Q84` rebuilds those sites' references and remakes every transfer function of theirs, and
leaves every other site alone.

Each workbook is generated from its own module -- `generator/wb01_survey.py` to `generator/wb05_final.py`,
each holding that workbook as one Python list of markdown and code cells -- which
`generator/make_workbooks.py` imports and writes into `workbooks/`. A workbook is executed in place by
`generator/run_workbooks.py`, which reports whether the notebook EXECUTED -- it fails on a non-zero nbconvert
exit, on any cell carrying an error output and on any code cell that was not run -- and counts the verdicts
the workbook itself printed beside it, N PASS, M FAIL, K UNJUDGED. A green runner line and a FAIL verdict
are different statements and both are printed.

A student changes the parameter cell of a notebook (the site, the survey, a threshold) and runs it in place;
the notebook is the student's working copy and its parameter cell is the record of what was run. A
maintainer keeping the shipped notebooks as the repository executed them runs a copy instead, with
`--survey <name>` and `--set NAME=VALUE`, which leaves `workbooks/` untouched. A change to what a workbook
DOES is made in its module and the notebook regenerated, because the modules are the source the shipped
notebooks are checked against.

## Frame and units

A transfer function is estimated and served in geomagnetic north: each site's horizontal magnetics are
rotated so the mean Hy is zero, which removes the hand-compass misalignment. The IGRF declination is recorded
in every file and not applied. Magnetics in nT, electrics in mV/km, periods in s.

## Which hours a pass is run on

The 1 Hz pass is the whole record. The 10 Hz pass is not, and one rule chooses its hours
(`auslamp_proc.process.selection`): each whole UTC hour of the 1 Hz cache is scored by the median squared
coherence over 20-200 s (Welch, 1,024 s segments) of Ex with Hy and of Ey with Hx; a stretch is a contiguous
run of hours in which both lines read above 0.5; the selection is the longest stretch, cut to its best 48
contiguous hours where it runs longer; and the control is a stretch of the same length placed at random
elsewhere in the record, not overlapping the selection. A selection that does not beat its control bought
efficiency, not a different answer. The same rule chooses workbook 04's per-row windows, on that row's own
electric line.

## Layout

    auslamp_proc/     the package (survey tables, raw readers and placement, cache, look, geo, observatory, register, figures, processing, transfer functions and agreement)
    auslamp_proc/bands/   the two EMTF band files with the level count and window length each belongs to
    workbooks/        the five workbooks a student opens, one site at a time after the survey
    generator/        one module per workbook (the cells as Python lists), the generator that writes workbooks/ and the runner
    surveys/          one folder per survey: survey.yaml, sites.csv, decisions.csv, SITES_COLUMNS.md in _template/
    tools/            one-off builders: the coastline the map draws, and the decisions table a survey worked on before arrives with
    tests/            pytest over the raw readers, the register, the selection rule and the regression the delivery must reproduce

## Status

A verdict a run reports honestly over a condition the survey itself carries is named in that survey's
`survey.yaml` under `checks.retained_failures`, with the workbook, the site and the reason. The conventions
test reads them and excuses no other FAIL.

2026-09-18: a new survey is served by the template alone. The template's two tables ship as header-only
stubs and a header with no row is now created rather than preserved, so workbook 01 writes them on a copied
template. Workbook 02 measures `sign_hx` and `sign_hz` from the DC test against IGRF and workbook 03
measures `sign_ex` and `sign_ey` from the phase quadrant of the site's own remote-reference transfer
function; each writes only into cells that read `decide`, and workbook 03 remakes that site's transfer
functions under what it wrote. `survey.yaml` gains `dipoles`, the deployment sheet's arm lengths or one
default with its reason, which is the path an EDL survey has from the field to `sites.csv`;
`tools/build_queensland_sites.py` is gone, because its dipole duty is that block and its other columns --
serials, azimuths, notes, the campaign's signs and remotes -- were a one-off already applied to the three
shipped tables, each cell carrying its own `*_source`, and it could not be pointed at a new survey.
Workbook 01 refuses to use `decide` as an IAGA code and names the observatory to write. A run over fewer
than `pool.min_fleet` other sites turns the fleet-normalised event test off rather than flagging everything,
and workbook 03 reads a pool of two without a traceback.

2026-09-17: the five workbooks are cut from the six that preceded them -- the survey, the records, one site's
references and transfer functions, one site in depth, and one site's final transfer function -- and the
survey-wide pages of the old fourth workbook are gone with it. Workbook 03 processes one site at a time over
the four reference kinds at 1 Hz and over the remote site and the fleet stack at 10 Hz on the coherent
stretch and its control. The single station is not a kind of this package: a noisy H biases it low with no
sign of it in the error bars. The raw readers were checked bit for bit against the released mt-io on the GA
Victoria release. The repository's history before 2026-09-13 is an April 2026 exploration of the Victoria
MTH5 files, retired in the first commit of the package.
