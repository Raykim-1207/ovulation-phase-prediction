#!/usr/bin/env bash
# Full pipeline, in the order used for the manuscript. Run from the repository root.
#
# Stages 1-2 need the raw recordings and label files, which are not public (see docs/DATA.md).
# From stage 3 onward only the derived nightly tables under $OVUL_DATA_DIR_ROOT are needed.
# Each stage writes under results/ (or the directory named by OVUL_RESULT_DIR).
set -euo pipefail

export OVUL_DATA_DIR_ROOT=${OVUL_DATA_DIR_ROOT:-$PWD/ovul_data}
export OVUL_FINAL_DIR=${OVUL_FINAL_DIR:-$OVUL_DATA_DIR_ROOT/finalfinal/interp_huawei24mid/locf}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
R=results
TRAIN=src/train
PRE=src/preprocess
mkdir -p $R

stage () { echo; echo "==== $1"; }

if [ "${SKIP_PREPROCESS:-0}" != 1 ]; then
  stage "1. raw recordings -> 10-minute masters -> nightly tables (needs OVUL_RAW_DIR)"
  (cd $PRE && for f in 00_master_*.py; do python3 "$f"; done)
  for s in 01_make_data 02_temp_day_interp 03_interpolation \
           04_fold_24_to_night 05_temp_hrv_nan; do (cd $PRE && python3 $s.py); done
  (cd $PRE && python3 06_add_m_cycle.py --root "$OVUL_DATA_DIR_ROOT/finalfinal")
  stage "2. physiological-range check of every family"
  python3 scripts/qc_all.py
fi

stage "3. five-axis grid (single split) and axis ranges; one process per family, as run for the paper"
for f in interp_ava interp_ava30 interp_ava24 interp_ava24mid interp_huawei interp_huawei24 \
         interp_huawei24mid interp_npj interp_npj24 interp_npj24mid interp_oura interp_oura24 interp_oura24mid; do
  python3 $TRAIN/grid.py --data-root "$OVUL_DATA_DIR_ROOT/finalfinal" \
    --out-root $R/grid_par/$f --families $f --yes &
done
wait
for m in locf_locf locf_linear locf_mean locf_forward linear_locf linear_linear linear_mean linear_forward; do
  for f in interp_24 interp_24agg; do
    python3 $TRAIN/grid.py --data-root "$OVUL_DATA_DIR_ROOT/finalfinal" \
      --out-root $R/grid_fill/${f#interp_}__$m --families $f --methods $m \
      --labels y2_std,y3_std,y4_std,y5_std --splits gs,sg --yes &
  done
done
wait
python3 scripts/merge_grid.py
python3 scripts/axis_ranges.py

stage "4. fifteen feature cells under the bootstrap protocol (Supplementary Table 2)"
for f in interp_ava interp_ava30 interp_ava24 interp_ava24mid interp_huawei interp_huawei24 \
         interp_huawei24mid interp_npj interp_npj24 interp_npj24mid interp_oura interp_oura24 interp_oura24mid; do
  python3 $TRAIN/grid_bootstrap.py --family $f --method locf --n-boot 1000 --workers $W
done
for f in interp_24 interp_24agg; do
  python3 $TRAIN/grid_bootstrap.py --family $f --method locf_locf --n-boot 1000 --workers $W
done

stage "5. main comparison: 1,000 participant-level bootstrap replicates (CPU models)"
# src/train/picked_cfgs.json holds the prior-study hyperparameters used in the manuscript.
OVUL_RESULT_DIR=$R/boot_full OVUL_DUMP_PRED_DIR=scripts/pred_models \
  python3 $TRAIN/bootstrap.py --n-boot 1000 --workers $W

stage "6. models proposed in 2024-2025 (GPU; restrict cards with OVUL_ALLOWED_GPUS)"
OVUL_RESULT_DIR=$R/boot_sota OVUL_DUMP_PRED_DIR=scripts/pred_models_gpu \
  python3 $TRAIN/recent_models.py --n-boot 1000 --models TabM,Mamba
OVUL_RESULT_DIR=$R/boot_sota2 OVUL_DUMP_PRED_DIR=scripts/pred_models_gpu \
  python3 $TRAIN/recent_models.py --n-boot 1000 --models KAN,RealMLP,ExcelFormer

stage "7. regularity strata, ovulation-day error, ablation"
OVUL_RESULT_DIR=$R/strata python3 $TRAIN/strata.py --n-boot 1000 --workers $W
OVUL_RESULT_DIR=$R/dayerr  python3 $TRAIN/ovulation_day_error.py --n-boot 1000 --workers $W   # also writes scripts/pred_nights/
OVUL_RESULT_DIR=results/ablation python3 $TRAIN/ablation.py --n-boot 1000 --workers $W

stage "8. participant-level tests (Fig. 2 brackets, Results, Supplementary Table 6)"
mkdir -p scripts/fig scripts/tables
python3 scripts/vis_fig2_testcohort.py          # input compositions
python3 scripts/model_tests_testcohort.py       # OvCast vs every other model
python3 scripts/vis_day_error_test.py           # ovulation-day error (prespecified 4-class test)
python3 scripts/day_error_strong_cal.py         # personalized calendar rule
python3 scripts/strata_testcohort.py            # regularity strata
python3 scripts/auroc_ovcast.py                 # AUROC / AUPRC (Methods)

stage "9. table bodies (LaTeX rows) and figures"
OVUL_BOOT_DIR=$R/grid_boot python3 scripts/make_grid15.py > scripts/tables/supp_table2.tex
python3 scripts/make_suppl_models.py macro       > scripts/tables/supp_table4.tex
python3 scripts/make_suppl_models.py ov          > scripts/tables/supp_table5.tex
python3 scripts/make_dayerr_table.py             > scripts/tables/supp_table6_upper.tex
python3 scripts/make_dayerr_table.py --tests     > scripts/tables/supp_table6_lower.tex
python3 scripts/vis_results.py --sig-tests scripts/fig/fig2_results_testcohort_tests.csv
for x in pdf png svg; do cp scripts/fig/fig2_results_bar_sig.$x scripts/fig/fig2_results.$x; done   # Fig. 2
python3 scripts/vis_feature_designs.py                                                            # Supplementary Fig. 1
