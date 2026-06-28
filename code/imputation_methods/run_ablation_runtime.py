# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#
# run_ablation_runtime.py
# ---------------------------------------------------------------------------
# Produces the THREE remaining pieces the reviewer asked for, in ONE run:
#   (A) ABLATION    - KNN-only, SVR-only, MICE-only, KNN+SVR, KNN+MICE,
#                     SVR+MICE, and the full KNN+SVR+MICE hybrid.
#   (B) RUNTIME     - mean imputation time per method (seconds).
#   (C) PER-VARIABLE- RMSE per sensor group (accelerometer / gyroscope /
#                     orientation / heart-rate) for hybrid vs baselines.
#
# Same leakage-free protocol as run_benchmark.py (train/eval split, masking
# only on eval, models fit only on train). Reuses your real scenario generator.
#
# HOW TO RUN
#   Put this file next to: hybrid_imputation_noleak.py, run_benchmark.py,
#   realistic_missing_data_generator.py, and data/original_data.csv
#   pip install numpy pandas scikit-learn
#   python run_ablation_runtime.py
#
# OUTPUTS (send these back):
#   ablation_summary.csv          - mean +/- std RMSE/MAE/R2 per ablation variant
#   ablation_by_mechanism.csv     - per mechanism, per variant
#   runtime_summary.csv           - mean +/- std imputation time per method
#   per_variable_summary.csv      - RMSE per sensor group per method
# ---------------------------------------------------------------------------

import warnings; warnings.filterwarnings("ignore")
import os, time, tempfile
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from realistic_missing_data_generator import RealisticMissingDataGenerator
from run_benchmark import (imp_mean, imp_knn, imp_svr, imp_mice, imp_missforest,
                           imp_hybrid, SCENARIO_METHODS, RATES, TRAIN_FRAC,
                           N_SEEDS, COMPLETE_CSV, KNN_K, SVM_KERNEL, SVM_C,
                           SVM_EPS, MICE_ITER)

# ------------------ ablation imputers (leakage-free) ----------------------
def _scale_fit(Xtr):
    sc = StandardScaler().fit(Xtr); return sc

def abl_knn(Xtr, Xev):          # KNN only
    return imp_knn(Xtr, Xev)

def abl_svr(Xtr, Xev):          # SVR only
    return imp_svr(Xtr, Xev)

def abl_mice(Xtr, Xev):         # MICE only (BayesianRidge) -> use SVR-MICE to match hybrid's MICE
    return IterativeImputer(estimator=SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS),
                            max_iter=MICE_ITER, random_state=42).fit(Xtr).transform(Xev)

def _iter_knn_svr(Xtr, Xev, use_knn=True, use_svr=True, iters=5):
    sc = _scale_fit(Xtr); d = Xtr.shape[1]
    Xtr_s, Xev_s = sc.transform(Xtr), sc.transform(Xev)
    svr_models = {}
    if use_svr:
        for j in range(d):
            cols = [c for c in range(d) if c != j]
            svr_models[j] = SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS).fit(Xtr_s[:, cols], Xtr_s[:, j])
    cur = Xev_s.copy(); miss = np.isnan(Xev_s)
    # initial fill so SVR-only path never leaves NaNs
    cm0 = np.nanmean(cur, axis=0)
    ii0 = np.where(np.isnan(cur)); cur[ii0] = np.take(cm0, ii0[1])
    cur[~miss] = Xev_s[~miss]
    for _ in range(iters):
        filled = cur.copy(); cm = np.nanmean(cur, axis=0)
        ii = np.where(np.isnan(filled)); filled[ii] = np.take(cm, ii[1])
        knn_pred = cur.copy()
        if use_knn:
            for j in range(d):
                mj = miss[:, j]
                if mj.any():
                    cols = [c for c in range(d) if c != j]
                    knn = KNeighborsRegressor(n_neighbors=KNN_K, metric="manhattan", weights="uniform")
                    obs = ~np.isnan(Xev_s[:, j])
                    knn.fit(filled[obs][:, cols], Xev_s[obs, j])
                    knn_pred[mj, j] = knn.predict(filled[mj][:, cols])
        out = knn_pred.copy()
        if use_svr:
            for j in range(d):
                mj = miss[:, j]
                if mj.any():
                    cols = [c for c in range(d) if c != j]
                    p = svr_models[j].predict(out[mj][:, cols])
                    out[mj, j] = 0.6*p + 0.4*knn_pred[mj, j] if use_knn else p
        cur = out; cur[~miss] = Xev_s[~miss]
    return sc.inverse_transform(cur)

def abl_knn_svr(Xtr, Xev):  return _iter_knn_svr(Xtr, Xev, True,  True)
def abl_knn_mice(Xtr, Xev):
    pre = _iter_knn_svr(Xtr, Xev, True, False)          # KNN init
    return IterativeImputer(estimator=SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS),
                            max_iter=MICE_ITER, random_state=42).fit_transform(pre)
def abl_svr_mice(Xtr, Xev):
    pre = _iter_knn_svr(Xtr, Xev, False, True)          # SVR only
    return IterativeImputer(estimator=SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS),
                            max_iter=MICE_ITER, random_state=42).fit_transform(pre)
def abl_full(Xtr, Xev):     return imp_hybrid(Xtr, Xev)  # KNN+SVR+MICE

ABLATION = {
    "KNN only": abl_knn, "SVR only": abl_svr, "MICE only": abl_mice,
    "KNN+SVR": abl_knn_svr, "KNN+MICE": abl_knn_mice, "SVR+MICE": abl_svr_mice,
    "KNN+SVR+MICE (full)": abl_full,
}
# methods for runtime + per-variable (the published set)
FULLSET = {"Mean": imp_mean, "KNN": imp_knn, "SVR": imp_svr,
           "MICE": imp_mice, "MissForest": imp_missforest, "Hybrid": imp_hybrid}

def make_masked(eval_df, gen_name, rate, seed):
    np.random.seed(seed)
    tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "e.csv"); eval_df.to_csv(p, index=False)
    g = RealisticMissingDataGenerator(p, tmp)
    info = getattr(g, gen_name)(rate)
    return pd.read_csv(info["path"])

def sensor_group(col):
    c = col.lower()
    if "acc" in c: return "Accelerometer"
    if "gyro" in c: return "Gyroscope"
    if "orient" in c or c in ("roll","pitch","yaw"): return "Orientation"
    if "hrm" in c or "heart" in c: return "HeartRate"
    return "Other"

def metrics(t, p, vr):
    return (float(np.sqrt(mean_squared_error(t, p))), float(mean_absolute_error(t, p)),
            float(r2_score(t, p)) if len(t) > 1 else np.nan)

def main():
    df = pd.read_csv(COMPLETE_CSV)
    num = [c for c in df.select_dtypes(include=["float64","int64"]).columns if c not in ("Activity","time")]
    df = df.dropna(subset=num).reset_index(drop=True); n = len(df)
    groups = {c: sensor_group(c) for c in num}
    print(f"rows {n} | features {len(num)}")

    abl_rows, run_rows, var_rows = [], [], []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed); perm = rng.permutation(n); cut = int(TRAIN_FRAC*n)
        tr, ev = df.iloc[perm[:cut]].reset_index(drop=True), df.iloc[perm[cut:]].reset_index(drop=True)
        Xtr = tr[num].to_numpy(float); Xev_true = ev[num].to_numpy(float)
        for label, gen, mech in SCENARIO_METHODS:
            for rate in RATES:
                masked = make_masked(ev, gen, rate, seed)
                Xev = masked[num].to_numpy(float); mask = np.isnan(Xev)
                if mask.sum() == 0: continue
                vr = float(np.nanmax(Xev_true)-np.nanmin(Xev_true)); true_at = Xev_true[mask]

                # (A) ablation
                for vname, fn in ABLATION.items():
                    try:
                        out = fn(Xtr.copy(), Xev.copy()); r, m, r2 = metrics(true_at, out[mask], vr)
                    except Exception as e:
                        r=m=r2=np.nan; print("abl warn", vname, e)
                    abl_rows.append(dict(variant=vname, scenario=label, mechanism=mech,
                                         rate=int(rate*100), seed=seed, RMSE=r, MAE=m, R2=r2))
                # (B) runtime + (C) per-variable, on the published method set
                for mname, fn in FULLSET.items():
                    t0 = time.perf_counter()
                    out = fn(Xtr.copy(), Xev.copy())
                    dt = time.perf_counter()-t0
                    run_rows.append(dict(method=mname, scenario=label, rate=int(rate*100),
                                         seed=seed, seconds=dt))
                    # per-variable RMSE
                    for j, col in enumerate(num):
                        mj = mask[:, j]
                        if mj.any():
                            rr = float(np.sqrt(mean_squared_error(Xev_true[mj, j], out[mj, j])))
                            var_rows.append(dict(method=mname, group=groups[col],
                                                 column=col, rate=int(rate*100),
                                                 seed=seed, RMSE=rr))
            print(f"seed {seed} | {label} done")

    A = pd.DataFrame(abl_rows); R = pd.DataFrame(run_rows); V = pd.DataFrame(var_rows)

    (A.groupby("variant").agg(RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                              MAE_mean=("MAE","mean"), R2_mean=("R2","mean"))
       .reset_index().sort_values("RMSE_mean")).to_csv("ablation_summary.csv", index=False)
    (A.groupby(["variant","mechanism"]).agg(RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"))
       .reset_index()).to_csv("ablation_by_mechanism.csv", index=False)
    (R.groupby("method").agg(sec_mean=("seconds","mean"), sec_std=("seconds","std"))
       .reset_index().sort_values("sec_mean")).to_csv("runtime_summary.csv", index=False)
    (V.groupby(["method","group"]).agg(RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"))
       .reset_index()).to_csv("per_variable_summary.csv", index=False)

    print("\n=== ABLATION (overall RMSE) ===")
    print(pd.read_csv("ablation_summary.csv").to_string(index=False))
    print("\n=== RUNTIME (s) ===")
    print(pd.read_csv("runtime_summary.csv").to_string(index=False))
    print("\nSaved: ablation_summary.csv, ablation_by_mechanism.csv, "
          "runtime_summary.csv, per_variable_summary.csv")

if __name__ == "__main__":
    main()
