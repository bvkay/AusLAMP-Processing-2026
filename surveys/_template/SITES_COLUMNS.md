# The survey's three tables: every column, what it means, who decides it

A survey is `survey.yaml` (the survey-level inputs), `sites.csv` (what was recorded, where and when) and
`decisions.csv` (what the analyst decides). A cell holds one of three things and there is no fourth:

| cell | meaning |
|---|---|
| a value | measured or read from a file; the neighbouring `*_source` column says where from |
| `assume:<value>` | used, and carried into every product's provenance as an assumption until it is replaced |
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
| rot_regimes | JSON list of `[start_day, end_day, angle_deg]` sensor-frame regimes; a magnetometer move makes two | the sensor-move scan |
| rot_drop | JSON list of `[start_day, end_day]` transit or fault days dropped from the H record | the look workbook |
| windows | JSON per component: `[start, end]` windows or `"whole"`; every window carries its reason | the processing workbook; the analyst decides |
| keep_mask | the path of a keep mask (best hours) or empty; a random selection of the same size is always built beside it | the processing workbook |
| remote_site | the chosen remote site, or `decide` | the reference workbook |
| remote_source | where the remote site came from, with its date; a table is named in full | whoever chose it |
| stack_members | JSON list of fleet stack members, or `decide`; a lender is never a member of the reference it feeds | the reference workbook |
| flags | free text carried into every product's provenance | the analyst |

## The vocabulary, fixed

single station, remote site, fleet stack, observatory, stack + observatory, member. The code keys are
`single`, `remote`, `stack`, `obs`, `stack_obs` and appear in code only.
