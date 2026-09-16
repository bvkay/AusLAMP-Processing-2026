# The band files

Two EMTF band-setup files, one per sample rate. Each line is `<level> <index_min> <index_max>`: a decimation
level and a range of FFT harmonics of the **window**, not of the record.

| file | rate | levels | bands | window | period range of the band centres |
|---|---|---|---|---|---|
| `bs_1hz_6lv_48.cfg` | 1 Hz | 6 | 48 | 256 samples | 2.303 s to 48,470 s |
| `bs_10hz_5lv_31.cfg` | 10 Hz | 5 | 31 | 256 samples | 1.073 s to 1,211.8 s |

The indices are harmonics of the window, so a file read at another window length names different periods and
nothing says so. File, level count and window length are therefore one object,
`auslamp_proc.process.aurora_run.BANDS`, keyed by rate:

    BANDS["1hz"]  -> BandSet(file=bs_1hz_6lv_48.cfg,  levels=6, window=256, rate_hz=1.0)
    BANDS["10hz"] -> BandSet(file=bs_10hz_5lv_31.cfg, levels=5, window=256, rate_hz=10.0)

`aurora_run.band_table(bandset)` reads the periods through Aurora's own machinery
(`EMTFBandSetupFile.compute_band_edges` then `Processing.assign_bands`), so the table printed in workbook 03
is what Aurora will use rather than a reading of the text.

The cascade is `[1] + [4] * (levels - 1)`: the rate falls by 4 at each level after the first, which at 1 Hz
gives 1, 1/4, 1/16, 1/64, 1/256 and 1/1024 Hz.

`bs_1hz_6lv_48.cfg` is copied from `work/bands/bs_wamt_1hz_6lv_48.cfg` and `bs_10hz_5lv_31.cfg` from
`work/bands/bs_qld_10hz.cfg` of the 2026 processing scripts.
