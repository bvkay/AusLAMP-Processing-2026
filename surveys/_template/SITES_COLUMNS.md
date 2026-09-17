# The survey's three tables: every column, what it means, who decides it

A survey is `survey.yaml` (the survey-level inputs), `sites.csv` (what was recorded, where and when) and
`decisions.csv` (what the analyst decides). A cell holds one of three things and there is no fourth:

| cell | meaning |
|---|---|
| a value | measured or read from a file; the neighbouring `*_source` column says where from |
| `assume:<value>` | used, and carried into the provenance of every transfer function as an assumption until it is replaced |
| `decide` | not yet decided; a later workbook measures it and writes it back with its source |

An empty cell means "not known, and nothing depends on it". Nothing is filled in silently. Re-running workbook
01 writes `sites_discovered.csv` into the work root every time and creates `sites.csv` once; after that it
prints the cells that differ and leaves `sites.csv` alone, and `auslamp_proc.survey.write_table` preserves
every `assume:` and `decide` cell a file already holds.

## sites.csv -- one row per site, written by workbook 01

| column | meaning | source / who decides |
|---|---|---|
| site | the station name the package uses | the raw folder name, never the file stem; repeats (R, a, b) stay distinct |
| raw_path | the site's raw folder | `survey.yaml` `raw_root` + `instruments_layout.<instrument>.subdir`, or `raw_overrides` |
| instrument | PR6-24+Mag-03, LEMI-424, LEMI-423 | which `instruments_layout` block the folder was found under |
| serial, serial_source | the logger serial and where it was read | LEMI `.INF` `%LEMI424 #NNNN`; the release MTH5 `data_logger.id`; an instrument register CSV |
| layout | edl_L (north and east arms with a shared centre) or lemi_cross | the instrument; the centre-electrode tests apply only to edl_L |
| sample_rate_hz | the rate read from the files | the miniSEED header (EDL) or the row spacing (LEMI); checked against `survey.yaml` |
| files | how many data files the site yielded | workbook 01 discovery |
| start_utc, end_utc | the first sample and the last sample, UTC | the first file's header and the last file's header plus its length |
| days | end minus start in days | derived |
| date_correction | `+1024 weeks` where the GPS week rollover was applied, else empty | `auslamp_proc.raw.edl.parse_stamp` against `survey.yaml` `years` |
| lat, lon, elev_m | the position of record | the instrument's own GPS: the EDL `.gps` median over daily files with >= 3 satellites, or the LEMI per-row fix |
| position_source | which of those, and what it was compared against | workbook 01; the release's own position is compared and never used |
| position_scatter_m | the 95th percentile spread of the site's own fixes about their median | workbook 01; over 100 m means something is wrong with the site |
| dipole_n_m, dipole_e_m | the arm lengths in metres | LEMI `.INF` `%L1 %L2`, the release MTH5 `dipole_length`, a field sheet, else `assume:<value>` |
| dipole_source | where each came from, or the reason for the assumption | as above; an assumption states its reason in full |
| azimuth_n_deg, azimuth_e_deg, azimuth_source | the arm directions | nominal 0 and 90 in the instrument frame unless a sheet says otherwise |
| declination_deg | IGRF at the site at the record midpoint | `auslamp_proc.geo.declination`; recorded, never applied |
| observatory | the nearest INTERMAGNET observatory with 1 s data over the span | `survey.yaml` default, overridable per site |
| notes | free text carried into the site's provenance | whoever writes it |

## decisions.csv -- one row per site, created by workbook 01 with `decide` everywhere

| column | meaning | who decides it |
|---|---|---|
| sign_hx, sign_hy, sign_hz | +1 / -1 / `decide`; Hx and Hz from the DC against IGRF (0.3 floor), Hy from the observatory east and two neighbours (undecided below abs(r) 0.3) | the look workbook; applied once and stored |
| sign_ex, sign_ey | +1 / -1 / `decide`; from the phase quadrant of a single-station Z with a sound H (>= 4 of 6 periods over 30-1000 s, monotone across days). An E line's sign is never read by comparing E fields between sites | the look workbook; the analyst approves the application |
| sign_source | where each sign came from, with its date | whoever decided it |
| e_exchange | `yes` / `no` / `decide`; the two recorded E lines were wired to each other's channel | the analyst, from the phase quadrant and the diagonal |
| e_exchange_source | the file:line or the measurement, with its date | whoever decided it |
| e_shift_s | seconds, or `decide`; E is advanced relative to H by this many seconds | the analyst, from the lag test |
| e_shift_source | the file:line or the measurement, with its date | whoever decided it |
| h_exchange | `yes` / `no` / `decide`; Hx and Hy were wired to each other's channel | the analyst, from the DC against IGRF |
| h_gain | a number, or `decide`; every magnetic channel is divided by it | the analyst, from the record's mean field against IGRF |
| h_gain_source | where `h_gain` and `h_exchange` came from, with the date and how each was measured | whoever decided them |
| h_lender | a site name, `none` or `decide`; that site's H stands in for this site's own | the analyst, from the magnetic health tests |
| h_lender_channels | which of Hx, Hy, Hz are borrowed; empty or `all` is all three. Electrics are never borrowed | the analyst |
| h_lender_source | where the lender came from, with its date | whoever decided it |
| rot_regimes | JSON list of `[start_day, end_day, angle_deg]` sensor-frame regimes; a magnetometer move makes two | the sensor-move scan |
| rot_drop | JSON list of `[start_day, end_day]` transit or fault days dropped from the H record | the look workbook |
| windows | JSON per component: `[start, end]` windows or `"whole"`; every window carries its reason. No workbook reads this cell: the hours a row is estimated on are chosen by `auslamp_proc.process.selection`, and a stretch the analyst names is given to workbook 04's recipe as `window:<ISO UTC start>/<hours>`. The cell is kept as the analyst's record of what a campaign did | the analyst |
| keep_mask | the path of a boolean .npy of hours to keep, or empty; `process.run` applies it on top of the transient mask and says loudly where the file it names is not there | the analyst |
| remote_site | the chosen remote site, or `decide` | the reference workbook |
| remote_source | where the remote site came from, with its date; a table is named in full | whoever chose it |
| stack_members | JSON list of fleet stack members, or `decide`; a lender is never a member of the reference it feeds | the reference workbook |
| flags | free text carried into the provenance of every transfer function | the analyst |

## The six hand decisions, and the one place they are applied

`e_exchange`, `e_shift_s`, `h_exchange`, `h_gain`, `h_lender` and `h_lender_channels` describe what the
field crew's wiring, the instrument's calibration and the site's own magnetics did to the record. They are
applied at processing time, never written into a cache, by one function --
`auslamp_proc.process.frame.apply_decisions` -- in one fixed order, and nothing applies a decision anywhere
else:

    e_exchange  ->  h_exchange  ->  h_gain  ->  e_shift_s  ->  the signs  ->  h_lender

The wiring exchanges come first because they say which line a channel carried; the gain is a calibration of
the channel as wired; the shift is a delay of the E line as wired; the signs are of the LINE and not of the
channel it was recorded on, so they come after the exchanges; the lender's channels come last because they
arrive carrying the LENDER's own decisions and must not take the borrowing site's.

**`e_exchange`.** The cache holds mV/km computed with the wrong arm length for a swapped channel:
`Ex_cache = V_east / (L_N g)`. So the swap puts each line back on its own channel and rescales it by the
ratio of the length the cache used to the length of the line the channel actually carried:

    Ex_out = Ey_cache x (L_E / L_N)        Ey_out = Ex_cache x (L_N / L_E)

with `L_N = dipole_n_m` and `L_E = dipole_e_m`. A site whose arm lengths are not both known is swapped and
not rescaled, and the transfer function says so. Signs apply after the swap, so `sign_ex` is the sign of the north
line wherever `e_exchange` is `yes`.

**`e_shift_s`.** Positive means E is ADVANCED: the E sample at `t` takes the recorded value at `t + s`,
which is the correction for an E line that LAGS H by `s`. It is applied to both electric channels at the
site's own rate by `process.align.shift`, the Lanczos-windowed sinc, which is local and does not spread an
impulsive sample. The same convention as `process.align.shift` and as the campaign's `qld_align.py:14-16`
("shift(x, lag_s) ADVANCES x: out[i] = x[i + lag_s*fs]"); the campaign's Q87 correction is named "E advanced
0.95 s" (`scripts/qc/qld_merge_edis.py:343-344`) and was applied to the tensor as `Z exp(i 2 pi dt / T)`,
which adds phase -- the sign that fills a phase deficit falling as 1/T.

**`h_gain`.** The site's magnetometer over-reads by this factor, so `Z = E/H` is low by it and the apparent
resistivity low by its square until every magnetic channel is divided. A tipper is unchanged by it, because
all three channels are divided alike.

**`h_lender`.** The named site's H, on this site's grid, with the LENDER's own decisions and its own
mean-field frame applied, its lag against this site measured by the 5-20 s rule of `process.align.shift_for`
and taken out with the Lanczos delay. A lender of a lender is not followed. Two consequences are enforced in
`process.references`: a site with `h_lender` is never a member of any reference, and its lender is never a
member of a reference that feeds it; both refusals are named in the reference's sidecar. Borrowing BOTH
horizontal channels makes the tensor an inter-site impedance -- this site's E on the field the lender
measured -- and every transfer function says so.

## The vocabulary, fixed

remote site, fleet stack, observatory, stack + observatory, member. The code keys are `remote`, `stack`,
`obs` and `stack_obs` and appear in code and in file names only. The single station is not a kind of this
package: noise in H biases it low and its error bars carry no sign of that bias.

What a pass estimates is a TRANSFER FUNCTION, and the one delivered per component is the transfer function of
record.
