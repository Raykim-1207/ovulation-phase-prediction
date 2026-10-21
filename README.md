# Verified ovulation timing and nocturnal wearable physiology — reproduction code

Companion code for

> Kim R., Lee J.H., Lee I., Cho J., Kim Y.K., Song C.-B., Kim J.S., Song H.S., Cho S.H.\*, Choi Y.S.\*
> **Ovulation occurs later and more variably than calendar rules assume: a hormonally verified
> evaluation of nocturnal wearable physiology.** Submitted to *npj Digital Medicine*, 2026.
> DOI to be added on publication.

The repository reproduces every data-driven table, figure and quoted statistic of the main text
and Supplementary Information from the derived nightly tables described in [Data](#data). It
contains the six feature-extraction designs (four re-implemented from published wearable studies,
two proposed here), the prespecified calendar rule, the proposed model (OvCast), the
re-implementations of the prior-study models and of five models proposed in 2024-2025, the
experimental grid, the participant-level bootstrap, the participant-level permutation tests,
and the table and figure generators.

## Layout

```
src/
  preprocess/           raw recordings -> 10-minute masters -> nightly tables
                        (00_master_<family>.py, 01_make_data.py ... 06_add_m_cycle.py)
  strict_cal_rule.py    calendar rules: cycle day 14 (primary), cycle day 15, preceding mean cycle length - 14
  model/                OvCast (ovcast.py), prior-study models (prior.py), 2024-2025 models (recent.py),
                        established learners (sota_model_ml.py)
  train/
    grid.py                  five-axis experimental grid (single split)
    bootstrap.py             main comparison, 1,000 participant-level bootstrap replicates
    recent_models.py         the five models proposed in 2024-2025 (GPU)
    strata.py                regularity strata
    ovulation_day_error.py   ovulation-day error
    ablation.py              component ablation of OvCast
    grid_bootstrap.py        the fifteen design-by-window cells under the bootstrap protocol
    tune_prior_models.py     hyperparameter selection for the prior-study models
    cohort.py, features.py, phase_labels.py, ovcast_config.py, grid_utils.py   shared modules
    picked_cfgs.json         prior-study hyperparameters used in the paper
scripts/
  reproduce.sh                the full pipeline in manuscript order
  vis_fig2_testcohort.py      participant-level tests between input compositions (Fig. 2 brackets)
  model_tests_testcohort.py   participant-level tests, OvCast vs every other model
  vis_day_error_test.py       ovulation-day error, participant-level test (prespecified 4-class)
  day_error_strong_cal.py     the same against a personalized calendar rule
  strata_testcohort.py        regularity strata, participant-level tests
  auroc_ovcast.py             AUROC / AUPRC of the original fit (Methods)
  axis_ranges.py, merge_grid.py, qc_all.py, merge_sota_patch.py   grid summaries and checks
  day_error.py                ovulation-day rule (fixed before the reported results were generated)
  make_*.py                   Supplementary Table bodies
  vis_results.py, vis_feature_designs.py   Fig. 2 and Supplementary Fig. 1
doc/
  DATA.md                     data layout, column schema and conventions
```

Stage outputs (`results/`, `scripts/pred_*`, `scripts/tables/`, `scripts/fig/`) and all
individual-level data are gitignored.

## System requirements

- Tested on Ubuntu 22.04.5 LTS, Python 3.10.12, Intel Xeon Gold 6526Y (64 cores).
- CPU only for everything except the five models proposed in 2024-2025, which were run on an
  NVIDIA RTX A5000 (24 GB); about 7 GB of GPU memory per worker.
- Package versions are pinned in `requirements.txt` (core) and `requirements-deep.txt`
  (2024-2025 models).

## Installation

```bash
conda env create -f environment.yml && conda activate ovul      # core
pip install -r requirements-deep.txt                            # optional: 2024-2025 models
```

or `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Data

Individual-level physiological recordings and clinical data cannot be deposited publicly: the
protocols approved by the institutional review boards of the two participating hospitals
(Hospital A and Hospital B) do not permit open release of overnight electrocardiography, which is
potentially re-identifiable. De-identified derived data sufficient to reproduce every reported
table and figure are available from the corresponding authors on request for non-commercial
research use, subject to a data use agreement and institutional review board approval.

The code expects the derived tables under `ovul_data/` (or the directory named by
`OVUL_DATA_DIR_ROOT`). Their layout, column schema and conventions are documented in
[`doc/DATA.md`](doc/DATA.md).

## Running

### Full pipeline

```bash
export OVUL_DATA_DIR_ROOT=$PWD/ovul_data
SKIP_PREPROCESS=1 bash scripts/reproduce.sh        # from the derived tables
# bash scripts/reproduce.sh                        # from raw recordings (needs OVUL_RAW_DIR)
```

Measured wall time on the system above (from the run logs of the manuscript):

| Stage | Script | Wall time |
|---|---|---|
| Five-axis grid, nightly designs | `grid.py`, 13 families as parallel processes | 52 min 18 s (longest process 49 min 6 s) |
| Five-axis grid, sequence designs | `grid.py`, 16 jobs as parallel processes | 9 min 41 s |
| Fifteen-cell bootstrap, 1,000 replicates | `grid_bootstrap.py`, 4 workers per cell, 15 cells in parallel | 9 min 33 s to 55 min 8 s per cell |
| Main bootstrap, 1,000 replicates | `bootstrap.py`, 28 workers | 78 min 56 s |
| 2024-2025 models, 1,000 replicates | `recent_models.py`, 2 GPUs x 2 workers | 1 h 54 min (TabM, Mamba); 4 h 25 min (KAN, RealMLP, ExcelFormer) |
| Regularity strata, 1,000 replicates | `strata.py`, 28 workers | 18 min 47 s |
| Ovulation-day error, 1,000 replicates | `ovulation_day_error.py`, 28 workers | 19 min 6 s |
| Ablation, 1,000 replicates | `ablation.py`, 28 workers | 2 h 19 min 50 s |

### Statistics, tables and figures only

Once the stage outputs exist, run the commands of stages 8 and 9 in `scripts/reproduce.sh`
(participant-level tests, AUROC, Supplementary Table bodies written to `scripts/tables/`, Fig. 2
and Supplementary Fig. 1). Fig. 2 needs the tests of stage 8 first.

### Where each manuscript item comes from

| Item | Produced by |
|---|---|
| Fig. 1 | schematic assembled by hand; counts from the analytic dataset |
| Fig. 2 (bars) | `src/train/bootstrap.py` -> `results/boot_full/boot.csv`; drawn by `scripts/vis_results.py` |
| Fig. 2 (brackets), composition tests in Results | `scripts/vis_fig2_testcohort.py` |
| Table 1 | `src/train/bootstrap.py` (4-class rows of `scripts/make_suppl_models.py`) |
| Table 2 | design and window definitions (`src/preprocess/`) |
| Five decision axes (Results) | `scripts/axis_ranges.py` |
| Model comparisons (Results) | `src/train/bootstrap.py`, `src/train/recent_models.py`; tests `scripts/model_tests_testcohort.py` |
| Ovulation-day error | `src/train/ovulation_day_error.py`; tests `scripts/vis_day_error_test.py`, `scripts/day_error_strong_cal.py` |
| Regularity strata | `src/train/strata.py`; tests `scripts/strata_testcohort.py` |
| Ablation | `src/train/ablation.py` |
| AUROC / AUPRC (Methods) | `scripts/auroc_ovcast.py` |
| Supplementary Fig. 1 | `scripts/vis_feature_designs.py` |
| Supplementary Table 2 | `src/train/grid_bootstrap.py`; `scripts/make_grid15.py` |
| Supplementary Table 3 | `src/train/bootstrap.py` (OvCast rows) |
| Supplementary Tables 4-5 | `src/train/bootstrap.py`, `src/train/recent_models.py`; `scripts/make_suppl_models.py` |
| Supplementary Table 6 | `src/train/ovulation_day_error.py`; `scripts/make_dayerr_table.py` |

Every random component is seeded (`SEED` in `src/train/features.py`), so a rerun on the same
hardware reproduces the reported values exactly, with one exception: the SVM baseline in
`grid.py` uses scikit-learn's unseeded Platt calibration (`SVC(probability=True)`), so its grid
metrics vary by up to 0.002 between runs. The 2024-2025 models were fitted on GPU; refitting
them on CPU changes their results slightly.

## Citation

```
Kim R., Lee J.H., Lee I., Cho J., Kim Y.K., Song C.-B., Kim J.S., Song H.S., Cho S.H., Choi Y.S.
"Ovulation occurs later and more variably than calendar rules assume: a hormonally verified
evaluation of nocturnal wearable physiology." Submitted to npj Digital Medicine (2026).
```

## License

MIT — see `LICENSE`. The repository contains no individual-level data.
