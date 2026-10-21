# Data layout, schemas and conventions

Individual-level physiological recordings and clinical data are not public (see the
Data availability statement of the paper and the [README](../README.md)). This file
documents what the code expects, so that the analysis can be read without the data and
re-run by anyone granted access under the data use agreement.

## Directory layout

```
raw/                     raw overnight ECG and skin-temperature files         (OVUL_RAW_DIR)
label/                   per-hospital ovulation-date workbooks                (OVUL_LABEL_A/B, OVUL_LABEL_A/B_SHEET)
measure_code_map.csv     recording-session to measurement-date map            (OVUL_MAP_CSV)
ovul_data/               derived tables                                       (OVUL_DATA_DIR_ROOT)
  pre/                   10-minute masters and QC lists (stage 1)
  finalfinal/<family>/<method>/y{2,3,4,5}_std.xlsx
```

Stage 1 of `scripts/reproduce.sh` builds `ovul_data/` from the raw files; `raw/`, `label/` and
`measure_code_map.csv` are needed for that stage only. Every later stage reads only
`ovul_data/finalfinal/`, which is what the data use agreement provides.

### Families: feature-extraction design x analysis window

`<family>` names one of the fifteen design-by-window cells of Supplementary Fig. 1 and
Supplementary Table 2 (windows: 1, trim 90/30 min; 2, trim 30/30 min; 3, 02:00-06:00;
4, recording midpoint +/- 2 h).

| family | design | window |
|---|---|---|
| `interp_ava`, `interp_ava30`, `interp_ava24`, `interp_ava24mid` | a, smoothed nightly level | 1, 2, 3, 4 |
| `interp_huawei`, `interp_huawei24`, `interp_huawei24mid` | b, day-over-day change | 2, 3, 4 |
| `interp_npj`, `interp_npj24`, `interp_npj24mid` | c, within-night block contrast | 2, 3, 4 |
| `interp_oura`, `interp_oura24`, `interp_oura24mid` | d, personal baseline deviation | 2, 3, 4 |
| `interp_24` | e, within-night sequence | 3 |
| `interp_24agg` | f, sequence summary | 3 |

The family carried forward to every reported comparison is `interp_huawei24mid`
(design b, window 4) with `locf` day-axis interpolation.

`<method>` is the interpolation: `locf` or `linear` on the day axis for designs a-d, and
`<day>_<slot>` (day in {locf, linear}, slot in {locf, linear, mean, forward}) for e and f.

`y{k}_std.xlsx` holds the `k`-class labels (k = 2, 3, 4, 5) under the phase definition used in
the paper. The four files share every column except the label column, which is named after the
file (`y2_std`, `y3_std`, `y4_std` or `y5_std`).

## Nightly table schema (`finalfinal/<family>/<method>/y{k}_std.xlsx`)

One row per recorded night. The columns of the carried-forward family are:

| Column | Meaning |
|---|---|
| `participant_id` | Participant identifier; the first letter gives the hospital (`S` = Hospital A, `G` = Hospital B) |
| `cycle` | Cycle number within the participant |
| `measure_date` | Date of the night |
| `file_name_x`, `ecg_start`, `used_from`, `used_to`, `used_min`, `record_min` | Recording bookkeeping: source file, ECG start, the four-hour window used, minutes used and recorded |
| `bbt`, `bbt_raw` | Nightly skin temperature (basal-body-temperature proxy), after and before day-axis imputation |
| `temp_day_imputed`, `temp_day_method` | Whether and how the night's temperature was imputed |
| `hr`, `hr_min`, `sdnn`, `rmssd`, `lfhf`, `ln_sdnn`, `ln_lfhf` | Nightly heart rate and heart-rate-variability statistics |
| `n_hrv_win`, `n_temp_pts` | Number of HRV windows and temperature samples in the night |
| `bbt_s3`, `hr_s3` | Three-night causal rolling means |
| `dbbt_1`, `dhr_1`, `dbbt_3`, `dhr_3` | Differences against the nights one and three earlier |
| `bbt_slope5`, `hr_slope5` | Five-night slopes along the day of cycle |
| `days_since_bbt_nadir`, `rising_streak` | Nights since the running temperature minimum; length of the current rising run |
| `label_date` | Date used for labelling the night |
| `day_of_cycle` | Day of cycle (1 = menstrual onset) |
| `menses_start`, `next_menses_start` | Menstrual onset of this cycle and of the next cycle |
| `ovulation_date` | Verified ovulation day (pelvic ultrasound, serum hormones and urinary LH; see Methods) |
| `y{k}_std` | Phase label of the night, `k` classes (one column per file) |
| `m_cycle`, `m_cycle_group` | Measured cycle length and its regularity stratum |

Other families carry their own feature columns (Supplementary Note 1). The within-night
sequence design (`interp_24`) is stored in long form, up to 24 rows per night, one per 10-minute
slot (`measure_time`), with the HRV quantities, `skin_temp`, `skin_temp_raw`, `temp_night_source`
and the observation masks `hrv_mask`, `temp_mask`, `hrv_dt`, `temp_dt`.
The 18 physiological columns supplied to every model are listed in the Methods (Experimental
Grid) and in `src/train/features.py` (`PHYSIO`).

## Conventions

- **Split.** Hospital A participants (`S-*`, 25 in the analytic set) train; Hospital B
  participants (`G-*`, 18) test. No participant appears in both. Result tables label this
  split `sg` (A to B); `gs` is the reverse direction, which is not reported.
- **Calendar block.** Cycle-history features are computed from prior cycles only; the
  current cycle's next onset is never used for prediction.
- **Duplicate nights.** One participant has 11 pairs of recordings on the same date. They
  are kept as separate rows, as in the analysis; scripts that align predictions across
  models therefore align by row position, not by (participant, date).
- **Ovulation-day rule.** Fixed in `scripts/day_error.py`: median offset of the nights
  predicted periovulatory, shifted back by the 1.5-day midpoint of the periovulatory band.
