# AusLAMP-Processing-2026

Long-period magnetotelluric processing for the AusLAMP surveys recorded on Earth Data Logger PR6-24 (with a
Bartington Mag-03 fluxgate, 10 Hz) or LEMI-424 (1 Hz) instruments, as six Jupyter workbooks on one Python package.
The workbooks take a survey from a folder of raw time series to one transfer function per site; every function lives
in `auslamp_proc`, so a batch run and a single-site run share one code path and any cell can be re-run. Built on the
IAGA-DVI-DataStandards packages mt-metadata, mt-io and mth5, with Aurora as the estimator.

The worked example is AusLAMP Queensland Phase 1 (23 EDL sites, September-November 2025), the survey the workbooks
open on; Phases 2 (18 sites, March-May 2026) and 3 (15 sites, June-July 2026) are kept as executed examples under
`workbooks/examples/`. The raw time series are not in this repository: point `raw_root` and `work_root` in
`surveys/<survey>/survey.yaml` at where they live on your machine. To run your own survey, copy `surveys/_template/`.

Author: Ben Kay (bvkay). Started 2026-09-13; re-cut for the workbook layout 2026-09-16.

## Install

    conda env create -f environment.yml
    conda activate auslamp-processing-2026
    python -m ipykernel install --user --name auslamp-processing-2026 --display-name "Python (auslamp-processing-2026)"

The readers are the released packages (mt-io 0.0.5, mth5 0.6.9, mt-metadata 1.0.11); nothing depends on a fork.

Launch the workbooks as `python -m jupyterlab`, and run the runner as `python workbooks/run_workbooks.py 01`,
with the environment's own interpreter. The `jupyter` dispatcher resolves its subcommands from PATH, which on a
machine with a base Anaconda install is a different nbconvert from the environment's, so `python -m jupyter ...`
silently escapes the environment where `python -m jupyterlab` and `python -m nbconvert` do not.

## The workbooks

| workbook | what it does |
|---|---|
| `01_survey` | reads the raw folder into a site table (instrument, serial, dipoles, positions from the logger's own GPS, dates with the EDL week rollover, declination), draws the map and the deployment register, and checks or fetches the observatory record for the span |
| `02_records` | builds each site's cache (every file placed on one absolute axis, gaps NaN, no sign or frame) and draws its record, band coherence, coherence maps, spectra and spectrograms, with the magnetometer DC test and the per-day state of each electric line -- look before processing |
| `03_process` | Aurora over the chosen sites and the four reference kinds (remote site, fleet stack, observatory, stack + observatory): at 1 Hz over the record, at 10 Hz on the most coherent hours with a random selection as the control; one folder per run with its provenance, and a figure for what each step did |
| `04_products` | every product of a site, group or survey on one page; kind against kind, rate against rate, run against run; each curve's smoothness; the survey gallery |
| `05_site` | one site in depth: the magnetics day by day, the fleet and clock tests, day masks, windows and best hours each with a random control, the arm diagonal for a shared centre, the 10 Hz short end, a magnetic channel borrowed from a neighbour, and the recipe that composes them -- one frame and, per impedance row, which hours at which rate -- into one product |
| `06_final` | one site's final transfer function: every product put to the three response tests (phase in quadrant, the slope bound, the error bar), the rule's proposal, the analyst's choice recorded, the 10 Hz join, one EDI per site |

Each workbook is generated from `workbooks/make_workbooks.py`, which holds it as one Python list of markdown and
code cells, and is executed in place by `workbooks/run_workbooks.py`, which fails on a non-zero nbconvert exit, on
any cell carrying an error output and on any code cell that was not run: edit the generator and re-run it, never
the notebook itself.

## Frame and units

Products are processed and served in geomagnetic north: each site's horizontal magnetics are rotated so the mean Hy
is zero, which removes the hand-compass misalignment. The IGRF declination is recorded in every file and not applied.
Magnetics in nT, electrics in mV/km, periods in s.

## Layout

    auslamp_proc/     the package (survey tables, raw readers and placement, cache, look, geo, observatory, register, figures, processing, products and agreement)
    auslamp_proc/bands/   the two EMTF band files with the level count and window length each belongs to
    workbooks/        the generator, the runner, the six workbooks, and examples/<survey>/ with the executed copies
    surveys/          one folder per survey: survey.yaml, sites.csv, decisions.csv, SITES_COLUMNS.md in _template/
    tools/            one-off builders: the coastline the map draws, the sheet cells merged into a survey's tables
    tests/            pytest over the raw readers and the register, and the regression the products must reproduce

## Status

2026-09-17: the six workbooks run end to end on Queensland Phase 1 (23 sites) and as examples on Phases 2 and 3,
with every check reporting; the times each step took are printed in the notebooks. Workbook 03 processes the
23 sites at 1 Hz over the four reference kinds (92 products) and at 10 Hz on the best 5, 10 and 25 per cent of
hours with a random 25 per cent as the control. The single station is not a kind of this package: a noisy H
biases it low with no sign of it in the error bars. The raw readers were checked bit for bit against the
released mt-io on the GA Victoria release. The repository's history before this date is an April 2026
exploration of the Victoria MTH5 files, retired in the first commit of the package.
