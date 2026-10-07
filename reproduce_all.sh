#!/usr/bin/env bash
# Reproduce the revised (leave-one-participant-out) results end to end, from
# data/har70/har70_full.csv. The run_*.py scripts are incremental and
# resume-safe (each skips participants already in its output CSV), so an
# interrupted run can simply be restarted.
set -euo pipefail
cd "$(dirname "$0")/code/imputation_methods"

# Single-threaded BLAS keeps the per-feature SVR from oversubscribing cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=../../results/har70_benchmark
mkdir -p "$OUT"

echo "[1/5] Baselines + two-stage KNN-SVR (18 folds)"
python3 run_benchmark_loso.py 0 18
mv -f loso_per_run.csv "$OUT"/ 2>/dev/null || true

echo "[2/5] Corrected three-stage: Full, KNN+MICE, MICE-within (18 folds)"
python3 run_corr_lean.py 0 18 "$OUT"/loso_corrected.csv

echo "[3/5] SVR+MICE ablation (18 folds)"
python3 run_svrmice.py 0 18 "$OUT"/loso_svrmice.csv

echo "[4/5] Temporal baselines LOCF / linear interpolation (18 folds)"
python3 run_temporal.py
mv -f loso_temporal.csv "$OUT"/ 2>/dev/null || true

echo "[5/5] Downstream activity classification for the proposed method (18 folds)"
python3 run_ds_lean.py 0 18 "$OUT"/loso_downstream_corr.csv

echo "Aggregating statistics ..."
python3 analyze_results.py

echo "Done. See $OUT/overall_rmse.csv, pairwise_full.csv, ablation_family.csv"
