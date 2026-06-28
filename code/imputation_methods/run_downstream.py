# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#
# run_downstream.py
# ---------------------------------------------------------------------------
# DOWNSTREAM VALIDATION  (paper Direction 6)
#
# Question answered:
#   "Does better imputation actually help a real task?"
#   We classify the ACTIVITY label (Mers / Repaus / Alergat / Scari) and measure
#   how classifier performance changes depending on how the missing sensor data
#   were imputed.
#
# Protocol (leakage-free, mirrors run_benchmark.py):
#   For each scenario x rate x seed:
#     1. Split rows into TRAIN / EVAL (disjoint).
#     2. Impute models are fit on TRAIN; missingness is applied to EVAL only.
#     3. Build six versions of the EVAL feature matrix:
#          - Complete  (upper bound: no missingness)
#          - Mean / KNN / SVR / MICE / MissForest / Hybrid imputed
#     4. Train a classifier ON TRAIN (complete features + activity labels),
#        then evaluate it on each imputed EVAL version.
#     5. Report Accuracy, macro-Precision, macro-Recall, macro-F1
#        (+ AUC if you later binarise the problem).
#
#   This isolates the effect of imputation quality on a fixed classifier.
#
# Output:
#   downstream_per_run.csv   - per (impute_method, scenario, rate, seed)
#   downstream_summary.csv   - mean +/- std per (impute_method)
#
# Requirements: numpy pandas scikit-learn ; same helper files as run_benchmark.py
# ---------------------------------------------------------------------------

import warnings; warnings.filterwarnings("ignore")
import os, tempfile
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from realistic_missing_data_generator import RealisticMissingDataGenerator
# reuse the exact same imputers as the benchmark for consistency
from run_benchmark import (imp_mean, imp_knn, imp_svr, imp_mice,
                           imp_missforest, imp_hybrid,
                           SCENARIO_METHODS, RATES, TRAIN_FRAC, N_SEEDS,
                           COMPLETE_CSV)

LABEL_COL = "Activity"

IMPUTERS = {"Complete": None,         # upper bound (no missingness applied)
            "Mean": imp_mean, "KNN": imp_knn, "SVR": imp_svr,
            "MICE": imp_mice, "MissForest": imp_missforest, "Hybrid": imp_hybrid}

def make_eval_with_missing(eval_df, gen_name, rate, seed, num_cols):
    np.random.seed(seed)
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "eval.csv"); eval_df.to_csv(p, index=False)
    gen = RealisticMissingDataGenerator(p, tmp)
    info = getattr(gen, gen_name)(rate)
    return pd.read_csv(info["path"])

def scores(y_true, y_pred):
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        Recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        F1=f1_score(y_true, y_pred, average="macro", zero_division=0))

def main():
    df = pd.read_csv(COMPLETE_CSV)
    num_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
                if c not in (LABEL_COL, "time", "participant")]
    df = df.dropna(subset=num_cols + [LABEL_COL]).reset_index(drop=True)
    n = len(df)
    print(f"Rows: {n} | features: {len(num_cols)} | classes: {df[LABEL_COL].nunique()}")

    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(TRAIN_FRAC * n)
        tr = df.iloc[perm[:cut]].reset_index(drop=True)
        ev = df.iloc[perm[cut:]].reset_index(drop=True)
        Xtr = tr[num_cols].to_numpy(float)
        ytr = tr[LABEL_COL].to_numpy()
        yev = ev[LABEL_COL].to_numpy()

        # one fixed classifier per seed, trained on COMPLETE train features
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)

        for label, gen_name, mech in SCENARIO_METHODS:
            for rate in RATES:
                masked = make_eval_with_missing(ev, gen_name, rate, seed, num_cols)
                Xev_nan = masked[num_cols].to_numpy(float)
                for mname, fn in IMPUTERS.items():
                    try:
                        if mname == "Complete":
                            Xeval = ev[num_cols].to_numpy(float)
                        else:
                            Xeval = fn(Xtr.copy(), Xev_nan.copy())
                        s = scores(yev, clf.predict(Xeval))
                    except Exception as e:
                        s = dict(Accuracy=np.nan, Precision=np.nan, Recall=np.nan, F1=np.nan)
                        print(f"  [warn] {mname} {label} {rate}: {e}")
                    rows.append(dict(impute=mname, scenario=label, mechanism=mech,
                                     rate=int(rate*100), seed=seed, **s))
            print(f"seed {seed} | {label} done")

    per = pd.DataFrame(rows)
    per.to_csv("downstream_per_run.csv", index=False)
    summ = (per.groupby("impute")
            .agg(Acc_mean=("Accuracy","mean"), Acc_std=("Accuracy","std"),
                 Prec_mean=("Precision","mean"),
                 Rec_mean=("Recall","mean"),
                 F1_mean=("F1","mean"), F1_std=("F1","std"))
            .reset_index().sort_values("F1_mean", ascending=False))
    summ.to_csv("downstream_summary.csv", index=False)
    print("\n=== DOWNSTREAM (activity classification) ===")
    print(summ.to_string(index=False))
    print("\nSaved: downstream_per_run.csv, downstream_summary.csv")

if __name__ == "__main__":
    main()
