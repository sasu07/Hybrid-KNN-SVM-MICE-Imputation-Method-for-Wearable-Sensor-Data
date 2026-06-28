# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#
# run_benchmark.py
# ---------------------------------------------------------------------------
# LEAKAGE-FREE EXTENDED BENCHMARK  (paper Directions 4 & 5)
#
# What it does
#   For each of the 6 scenarios x 3 missing rates (= 18 conditions), repeated
#   over N_SEEDS random seeds:
#     1. Split the COMPLETE dataset into disjoint TRAIN / EVAL row subsets.
#     2. Use your real RealisticMissingDataGenerator to create the missing
#        pattern, but apply it ONLY to the EVAL subset.
#     3. Fit every model that needs "complete" data ONLY on TRAIN.
#     4. Impute the masked EVAL positions; compare to held-out ground truth.
#   Methods compared: Mean, KNN, SVR, MICE, MissForest, Hybrid(no-leak).
#
# Output
#   results_per_run.csv     - one row per (method, scenario, rate, seed)
#   results_summary.csv     - mean +/- std per (method, scenario, rate)
#   results_overall.csv     - mean +/- std per method over all conditions
#   results_by_mechanism.csv- mean +/- std per (method, mechanism)
#
# HOW TO RUN
#   1. Put this file, hybrid_imputation_noleak.py, and your
#      realistic_missing_data_generator.py in the same folder.
#   2. Put original_data.csv (your 3,845-row complete dataset) in ./data/
#   3. pip install numpy pandas scikit-learn
#   4. python run_benchmark.py
# ---------------------------------------------------------------------------

import warnings; warnings.filterwarnings("ignore")
import os, tempfile
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from hybrid_imputation_noleak import HybridImputerNoLeak
from realistic_missing_data_generator import RealisticMissingDataGenerator

# ===========================  CONFIG  ======================================
COMPLETE_CSV = "data/har70_full.csv"
TRAIN_FRAC   = 0.6
N_SEEDS      = 10
RATES        = [0.10, 0.20, 0.30]
KNN_K        = 5
SVM_KERNEL, SVM_C, SVM_EPS = "rbf", 100, 0.1
MICE_ITER    = 10

SCENARIO_METHODS = [   # (label, generator-method-name, mechanism)
    ("MCAR_RandomSensorFailures", "random_sensor_failures",       "MCAR"),
    ("MCAR_ConnectionIssues",     "temporary_connection_issues",  "MCAR"),
    ("MAR_ActivityDependent",     "activity_dependent_failures",  "MAR"),
    ("MAR_BatteryDepletion",      "battery_depletion_patterns",   "MAR"),
    ("MNAR_ValueDependent",       "value_dependent_failures",     "MNAR"),
    ("MNAR_RangeLimitation",      "sensor_range_limitations",     "MNAR"),
]

# ===================  helper: generate one masked EVAL frame  ===============
def make_eval_with_missing(eval_df, gen_method_name, rate, seed):
    """
    Uses your RealisticMissingDataGenerator on the EVAL subset only.
    Returns (eval_with_nan_df, numerical_cols).
    """
    np.random.seed(seed)
    tmpdir = tempfile.mkdtemp()
    eval_path = os.path.join(tmpdir, "eval_complete.csv")
    eval_df.to_csv(eval_path, index=False)
    gen = RealisticMissingDataGenerator(eval_path, tmpdir)
    method = getattr(gen, gen_method_name)
    info = method(rate)                      # writes a csv and returns info dict
    masked = pd.read_csv(info["path"])
    num_cols = gen._get_numerical_columns()
    return masked, num_cols

# ===========================  imputers  ====================================
def imp_mean(Xtr, Xev):
    return SimpleImputer(strategy="mean").fit(Xtr).transform(Xev)

def imp_knn(Xtr, Xev):
    # per-column KNN regression, fit on TRAIN, applied to EVAL (no leakage)
    d = Xtr.shape[1]; out = Xev.copy()
    tr_mean = np.nanmean(Xtr, axis=0)
    Xtr_f = np.where(np.isnan(Xtr), tr_mean, Xtr)
    ev_f  = np.where(np.isnan(Xev), tr_mean, Xev)
    for j in range(d):
        mj = np.isnan(Xev[:, j])
        if mj.any():
            cols = [c for c in range(d) if c != j]
            knn = KNeighborsRegressor(n_neighbors=KNN_K, metric="manhattan", weights="uniform")
            knn.fit(Xtr_f[:, cols], Xtr[:, j])
            out[mj, j] = knn.predict(ev_f[mj][:, cols])
    return out

def imp_svr(Xtr, Xev):
    if len(Xtr) > 1500:
        Xtr = Xtr[np.random.RandomState(0).choice(len(Xtr), 1500, replace=False)]
    sc = StandardScaler().fit(Xtr); d = Xtr.shape[1]
    Xtr_s, Xev_s = sc.transform(Xtr), sc.transform(Xev)
    models = {}
    for j in range(d):
        cols = [c for c in range(d) if c != j]
        models[j] = SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS).fit(Xtr_s[:, cols], Xtr_s[:, j])
    out = Xev_s.copy(); cm = np.nanmean(Xev_s, axis=0)
    filled = np.where(np.isnan(out), cm, out)
    for j in range(d):
        mj = np.isnan(Xev_s[:, j])
        if mj.any():
            cols = [c for c in range(d) if c != j]
            out[mj, j] = models[j].predict(filled[mj][:, cols])
    return sc.inverse_transform(out)

def imp_mice(Xtr, Xev):
    return IterativeImputer(estimator=BayesianRidge(), max_iter=MICE_ITER,
                            random_state=0).fit(Xtr).transform(Xev)

def imp_missforest(Xtr, Xev):
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
        max_iter=MICE_ITER, random_state=0).fit(Xtr).transform(Xev)

def imp_hybrid(Xtr, Xev):
    h = HybridImputerNoLeak(knn_neighbors=KNN_K, svm_kernel=SVM_KERNEL, svm_C=SVM_C,
                            svm_epsilon=SVM_EPS, iterations=5, mice_max_iter=MICE_ITER)
    h.fit(Xtr)
    return h.transform(Xev)

IMPUTERS = {"Mean": imp_mean, "KNN": imp_knn, "SVR": imp_svr,
            "MICE": imp_mice, "MissForest": imp_missforest, "Hybrid": imp_hybrid}

# ===========================  metrics  =====================================
def metrics(true, pred, vr):
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae  = float(mean_absolute_error(true, pred))
    nrmse = rmse / vr if vr > 0 else np.nan
    r2 = float(r2_score(true, pred)) if len(true) > 1 else np.nan
    return rmse, mae, nrmse, r2

# ===========================  main  ========================================
def main():
    df = pd.read_csv(COMPLETE_CSV)
    num_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
                if c not in ("Activity", "time", "participant")]
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    n = len(df)
    print(f"Complete rows: {n} | numerical cols: {len(num_cols)}")

    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(TRAIN_FRAC * n)
        train_df = df.iloc[perm[:cut]].reset_index(drop=True)
        eval_df  = df.iloc[perm[cut:]].reset_index(drop=True)
        Xtr = train_df[num_cols].to_numpy(float)

        for label, gen_name, mech in SCENARIO_METHODS:
            for rate in RATES:
                masked_df, _ = make_eval_with_missing(eval_df, gen_name, rate, seed)
                Xev = masked_df[num_cols].to_numpy(float)
                Xev_true = eval_df[num_cols].to_numpy(float)
                mask = np.isnan(Xev)
                if mask.sum() == 0:
                    continue
                vr = float(np.nanmax(Xev_true) - np.nanmin(Xev_true))
                true_at = Xev_true[mask]
                for mname, fn in IMPUTERS.items():
                    try:
                        out = fn(Xtr.copy(), Xev.copy())
                        rmse, mae, nrmse, r2 = metrics(true_at, out[mask], vr)
                    except Exception as e:
                        rmse = mae = nrmse = r2 = np.nan
                        print(f"  [warn] {mname} {label} {rate}: {e}")
                    rows.append(dict(method=mname, scenario=label, mechanism=mech,
                                     rate=int(rate*100), seed=seed,
                                     RMSE=rmse, MAE=mae, NRMSE=nrmse, R2=r2))
            print(f"seed {seed} | {label} done")

    per = pd.DataFrame(rows)
    per.to_csv("results_per_run.csv", index=False)

    def agg(g): return g.agg(RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                             MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                             NRMSE_mean=("NRMSE","mean"),
                             R2_mean=("R2","mean"), R2_std=("R2","std")).reset_index()
    agg(per.groupby(["method","scenario","rate"])).to_csv("results_summary.csv", index=False)
    agg(per.groupby(["method","mechanism"])).to_csv("results_by_mechanism.csv", index=False)
    overall = agg(per.groupby("method")).sort_values("RMSE_mean")
    overall.to_csv("results_overall.csv", index=False)

    print("\n=== OVERALL (mean over 18 conditions x seeds) ===")
    print(overall.to_string(index=False))
    print("\nSaved: results_per_run.csv, results_summary.csv, "
          "results_by_mechanism.csv, results_overall.csv")

if __name__ == "__main__":
    main()
