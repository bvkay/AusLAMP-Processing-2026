# AusLAMP-Processing-2026

Long-period magnetotelluric processing for the AusLAMP surveys recorded on Earth Data Logger PR6-24 (with a
Bartington Mag-03 fluxgate, 10 Hz) or LEMI-424 (1 Hz) instruments, as six Jupyter workbooks on one Python package.
The workbooks take a survey from a folder of raw time series to one transfer function per site; every function lives
in `auslamp_proc`, so a batch run and a single-site run share one code path and any cell can be re-run. Built on the
IAGA-DVI-DataStandards packages mt-metadata, mt-io and mth5, with Aurora as the estimator.

Worked examples: AusLAMP Victoria (GA eCat 150806, 81 EDL and 19 LEMI-424 sites, 2013-2018) and AusLAMP Queensland
Phases 1, 2 and 3 (EDL, 2025-26). To run them on another machine, edit `raw_root` and `work_root` in the survey's
`surveys/<survey>/survey.yaml`; to run your own survey, copy `surveys/_template/`.

Author: Ben Kay (bvkay). Started 2026-09-13; re-cut for the workbook layout 2026-09-16.

## Install

    conda env create -f environment.yml
    conda activate auslamp-processing-2026
    python -m ipykernel install --user --name auslamp-processing-2026 --display-name "Python (auslamp-processing-2026)"

The readers are the released packages (mt-io 0.0.5, mth5 0.6.9, mt-metadata v1.0.11); nothing depends on a fork.

Launch the workbooks as `python -m jupyterlab`, and run the runner as `python workbooks/run_workbooks.py 01`,
with the environment's own interpreter. The `jupyter` dispatcher resolves its subcommands from PATH, which on a
machine with a base Anaconda install is a different nbconvert from the environment's, so `python -m jupyter ...`
silently escapes the environment where `python -m jupyterlab` and `python -m nbconvert` do not.

## The workbooks

| workbook | what it does |
|---|---|
| `01_survey` | reads the raw folder into a site table (instrument, serial, dipoles, positions from the logger's own GPS, dates with the EDL week rollover, declination), draws the map and the deployment register, and checks or fetches the observatory record for the span |
| `02_records` | builds each site's cache and draws its record, band coherence, coherence maps, spectra and spectrograms -- look before processing |
| `03_process` | Aurora over the chosen sites and reference kinds (single station, remote site, fleet stack, observatory, stack + observatory), at 1 Hz and 10 Hz, one folder per run with its provenance |
| `04_products` | every product of a site, group or survey on one page; run against run; the release turned into our frame as a comparison |
| `05_site` | one site in depth: windows and masks with their random control, the north-minus-east diagonal, the notch, a magnetic channel replaced from a neighbour, the stack or the observatory |
| `06_final` | the readings rule over every product, the product of record per component, the splice, one EDI per site |

Each workbook is generated from `workbooks/make_workbooks.py`, which holds it as one Python list of markdown and
code cells, and is executed in place by `workbooks/run_workbooks.py`, which fails on a non-zero nbconvert exit, on
any cell carrying an error output and on any code cell that was not run: edit the generator and re-run it, never
the notebook itself.

## Frame and units

Products are processed and served in geomagnetic north: each site's horizontal magnetics are rotated so the mean Hy
is zero, which removes the hand-compass misalignment. The IGRF declination is recorded in every file and not applied.
The release EDIs (geographic north) are turned by the declination when compared. Magnetics in nT, electrics in mV/km,
periods in s.

## Layout

    auslamp_proc/     the package (survey tables, raw readers and placement, cache, geo, observatory, register, figures, processing)
    workbooks/        the generator, the runner and the six workbooks
    surveys/          one folder per survey: survey.yaml, sites.csv, decisions.csv, SITES_COLUMNS.md in _template/
    tools/            one-off builders: the coastline the map draws, the decisions table a previous survey seeds
    tests/            pytest over the raw readers and the register, and the regression the products must reproduce

## Status

2026-09-16: workbook 01 runs end to end on AusLAMP Victoria (100 sites, 243,157 data files, about 20 s) with nine
checks reporting. The repository's history before this date is an April 2026 exploration of the Victoria MTH5
files, retired in the first commit of the package.
