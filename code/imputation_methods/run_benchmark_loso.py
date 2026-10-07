# Copyright (c) 2025 Gabriel-Vasilica Sasu
# run_benchmark_loso.py
# ---------------------------------------------------------------------------
# PARTICIPANT-LEVEL LEAKAGE-FREE BENCHMARK  (addresses reviewer points 1 & 2)
#
# Key differences vs the earlier window-level benchmark:
#   * Splitting unit is the PARTICIPANT, not the window. For each of the 18
#     HAR70+ participants we hold that participant out as the EVAL set and use
#     the OTHER 17 as the clean TRAIN (donor) set  ->  Leave-One-Participant-Out
#     (LOPO). No window from an eval participant is ever seen in training, so
#     participant-level leakage is impossible.
#   * ALL model-based baselines (KNN, SVR, MICE, MissForest, Hybrid) are fit on
#     the SAME clean TRAIN subset (the 17 held-in participants) and applied to
#     the masked EVAL participant. The KNN donor pool is therefore stated and
#     identical across methods (Mean uses TRAIN column means). This removes the
#     "unequal information access" concern.
#   * The Activity label is NEVER part of the imputation feature matrix
#     (num_cols excludes Activity/time/participant); it is only kept aside for
#     the downstream task.
#   * Metrics are computed on standardized signals using the TRAIN scaler, so
#     RMSE is unit-free (z-score units) and comparable across the heterogeneous
#     channels. NRMSE (RMSE / value-range) is also reported.
#
# Output (per_participant granularity is what enables the cluster bootstrap):
#   loso_per_run.csv   - one row per (method, scenario, rate, eval_participant)
# ---------------------------------------------------------------------------

import warnings; warnings.filterwarnings("ignore")
import os, tempfile, time
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

from hybrid_imputation import HybridImputerNoLeak
from realistic_missing_data_generator import RealisticMissingDataGenerator

# ===========================  CONFIG  ======================================
def _find_data(name="har70_full.csv"):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("data", name),
        os.path.join("data", "har70", name),
        os.path.join(here, "data", name),
        os.path.join(here, "data", "har70", name),
        os.path.join(here, "..", "data", name),
        os.path.join(here, "..", "data", "har70", name),
        os.path.join(here, "..", "..", "data", name),
        os.path.join(here, "..", "..", "data", "har70", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("data", "har70", name)
COMPLETE_CSV = _find_data()
RATES        = [0.10, 0.20, 0.30]
KNN_K        = 5
SVM_KERNEL, SVM_C, SVM_EPS = "rbf", 100, 0.1
MICE_ITER    = 10
SVR_SUBSAMPLE = 800          # cap TRAIN rows for SVR/hybrid fit (speed only)

SCENARIO_METHODS = [
    ("MCAR_RandomSensorFailures", "random_sensor_failures",       "MCAR"),
    ("MCAR_ConnectionIssues",     "temporary_connection_issues",  "MCAR"),
    ("MAR_ActivityDependent",     "activity_dependent_failures",  "MAR"),
    ("MAR_BatteryDepletion",      "battery_depletion_patterns",   "MAR"),
    ("MNAR_ValueDependent",       "value_dependent_failures",     "MNAR"),
    ("MNAR_RangeLimitation",      "sensor_range_limitations",     "MNAR"),
]

# ===================  masked EVAL frame (uses the real generator)  ==========
def make_eval_with_missing(eval_df, gen_method_name, rate, seed):
    np.random.seed(seed)
    tmpdir = tempfile.mkdtemp()
    eval_path = os.path.join(tmpdir, "eval_complete.csv")
    eval_df.to_csv(eval_path, index=False)
    gen = RealisticMissingDataGenerator(eval_path, tmpdir)
    info = getattr(gen, gen_method_name)(rate)
    masked = pd.read_csv(info["path"])
    return masked

# ===========================  imputers  ====================================
# All take standardized TRAIN (Xtr_s) and standardized masked EVAL (Xev_s);
# return standardized imputed EVAL. Standardization uses the TRAIN scaler,
# so every method sees the same clean donor information.

def imp_mean(Xtr_s, Xev_s):
    return SimpleImputer(strategy="mean").fit(Xtr_s).transform(Xev_s)

def imp_knn(Xtr_s, Xev_s):
    d = Xtr_s.shape[1]; out = Xev_s.copy()
    tr_mean = np.nanmean(Xtr_s, axis=0)
    Xtr_f = np.where(np.isnan(Xtr_s), tr_mean, Xtr_s)
    ev_f  = np.where(np.isnan(Xev_s), tr_mean, Xev_s)
    for j in range(d):
        mj = np.isnan(Xev_s[:, j])
        if mj.any():
            cols = [c for c in range(d) if c != j]
            knn = KNeighborsRegressor(n_neighbors=KNN_K, metric="manhattan", weights="uniform")
            knn.fit(Xtr_f[:, cols], Xtr_s[:, j])   # donor pool = the 17 TRAIN participants
            out[mj, j] = knn.predict(ev_f[mj][:, cols])
    return out

def imp_svr(Xtr_s, Xev_s):
    Xtr = Xtr_s
    if len(Xtr) > SVR_SUBSAMPLE:
        Xtr = Xtr[np.random.RandomState(0).choice(len(Xtr), SVR_SUBSAMPLE, replace=False)]
    d = Xtr.shape[1]; models = {}
    for j in range(d):
        cols = [c for c in range(d) if c != j]
        models[j] = SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS).fit(Xtr[:, cols], Xtr[:, j])
    out = Xev_s.copy(); cm = np.nanmean(Xev_s, axis=0)
    filled = np.where(np.isnan(out), cm, out)
    for j in range(d):
        mj = np.isnan(Xev_s[:, j])
        if mj.any():
            cols = [c for c in range(d) if c != j]
            out[mj, j] = models[j].predict(filled[mj][:, cols])
    return out

def imp_mice(Xtr_s, Xev_s):
    return IterativeImputer(estimator=BayesianRidge(), max_iter=MICE_ITER,
                            random_state=0).fit(Xtr_s).transform(Xev_s)

def imp_missforest(Xtr_s, Xev_s):
    Xtr = Xtr_s
    if len(Xtr) > SVR_SUBSAMPLE:
        Xtr = Xtr[np.random.RandomState(0).choice(len(Xtr), SVR_SUBSAMPLE, replace=False)]
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=40, n_jobs=-1, random_state=0, max_depth=12),
        max_iter=5, random_state=0).fit(Xtr).transform(Xev_s)

def imp_hybrid(Xtr_s, Xev_s):
    # Hybrid operates in standardized space already; give it an identity-scaled
    # fit by pre-standardizing. We reuse its internal logic but feed standardized
    # data and undo its internal scaler by fitting on standardized TRAIN.
    h = HybridImputerNoLeak(knn_neighbors=KNN_K, svm_kernel=SVM_KERNEL, svm_C=SVM_C,
                            svm_epsilon=SVM_EPS, iterations=5, mice_max_iter=MICE_ITER)
    h.fit(Xtr_s)
    return h.transform(Xev_s)

def imp_knn_svr(Xtr_s, Xev_s):
    # two-stage ablation candidate: KNN init + SVR refine, NO MICE
    d = Xtr_s.shape[1]
    Xtr = Xtr_s
    if len(Xtr) > SVR_SUBSAMPLE:
        Xtr = Xtr[np.random.RandomState(0).choice(len(Xtr), SVR_SUBSAMPLE, replace=False)]
    svr = {}
    for j in range(d):
        cols = [c for c in range(d) if c != j]
        svr[j] = SVR(kernel=SVM_KERNEL, C=SVM_C, epsilon=SVM_EPS).fit(Xtr[:, cols], Xtr[:, j])
    miss = np.isnan(Xev_s); cur = Xev_s.copy()
    for _ in range(5):
        col_mean = np.nanmean(cur, axis=0)
        filled = np.where(np.isnan(cur), col_mean, cur)
        knn_pred = cur.copy()
        for j in range(d):
            mj = miss[:, j]
            if mj.any():
                cols = [c for c in range(d) if c != j]
                knn = KNeighborsRegressor(n_neighbors=KNN_K, metric="manhattan")
                obs = ~np.isnan(Xev_s[:, j])
                knn.fit(filled[obs][:, cols], Xev_s[obs, j])
                knn_pred[mj, j] = knn.predict(filled[mj][:, cols])
        for j in range(d):
            mj = miss[:, j]
            if mj.any():
                cols = [c for c in range(d) if c != j]
                pred = svr[j].predict(np.where(np.isnan(knn_pred), np.nanmean(knn_pred,axis=0), knn_pred)[mj][:, cols])
                cur[mj, j] = 0.6 * pred + 0.4 * knn_pred[mj, j]
        cur[~miss] = Xev_s[~miss]
    return cur

IMPUTERS = {"Mean": imp_mean, "KNN": imp_knn, "SVR": imp_svr,
            "MICE": imp_mice, "MissForest": imp_missforest,
            "KNN_SVR": imp_knn_svr, "Hybrid": imp_hybrid}

def metrics(true, pred, vr):
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae  = float(mean_absolute_error(true, pred))
    nrmse = rmse / vr if vr > 0 else np.nan
    r2 = float(r2_score(true, pred)) if len(true) > 1 else np.nan
    return rmse, mae, nrmse, r2

# ===========================  main  ========================================
def main(fold_start=0, fold_end=18, out_csv="loso_per_run.csv"):
    df = pd.read_csv(COMPLETE_CSV)
    num_cols = [c for c in df.select_dtypes(include=["float64","int64"]).columns
                if c not in ("Activity","time","participant")]
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    participants = sorted(df["participant"].unique())
    print(f"Rows: {len(df)} | features: {len(num_cols)} | participants: {len(participants)}")
    print(f"Protocol: LOPO folds [{fold_start}:{fold_end}] -> out={out_csv}")

    # skip participants already saved (robust incremental runs)
    done_pids = set()
    if os.path.exists(out_csv):
        try:
            done_pids = set(pd.read_csv(out_csv)["participant"].unique())
        except Exception:
            pass

    t0 = time.time()
    for fold_i, pid in enumerate(participants):
        if fold_i < fold_start or fold_i >= fold_end:
            continue
        if int(pid) in done_pids:
            print(f"  fold {fold_i} (p{pid}) already done, skipping")
            continue
        rows = []
        train_df = df[df.participant != pid].reset_index(drop=True)
        eval_df  = df[df.participant == pid].reset_index(drop=True)
        Xtr_raw  = train_df[num_cols].to_numpy(float)

        # TRAIN scaler shared by all methods -> unit-free, equal information
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr_s  = scaler.transform(Xtr_raw)

        for label, gen_name, mech in SCENARIO_METHODS:
            for rate in RATES:
                # seed tied to fold so masks differ per participant but are reproducible
                seed = 1000 + fold_i
                masked_df = make_eval_with_missing(eval_df[num_cols.__class__(num_cols)] if False else eval_df,
                                                   gen_name, rate, seed)
                Xev_true = eval_df[num_cols].to_numpy(float)
                Xev_raw  = masked_df[num_cols].to_numpy(float)
                mask = np.isnan(Xev_raw)
                if mask.sum() == 0:
                    continue
                Xev_s_true = scaler.transform(Xev_true)
                Xev_s      = Xev_s_true.copy(); Xev_s[mask] = np.nan
                true_at = Xev_s_true[mask]
                vr = float(np.nanmax(Xev_s_true) - np.nanmin(Xev_s_true))
                for mname, fn in IMPUTERS.items():
                    try:
                        out = fn(Xtr_s.copy(), Xev_s.copy())
                        rmse, mae, nrmse, r2 = metrics(true_at, out[mask], vr)
                    except Exception as e:
                        rmse = mae = nrmse = r2 = np.nan
                        print(f"  [warn] {mname} {label} {rate} p{pid}: {e}")
                    rows.append(dict(method=mname, scenario=label, mechanism=mech,
                                     rate=int(rate*100), participant=int(pid),
                                     RMSE=rmse, MAE=mae, NRMSE=nrmse, R2=r2))
        # incremental append after EACH fold -> partial progress is safe
        per = pd.DataFrame(rows)
        hdr = not os.path.exists(out_csv)
        per.to_csv(out_csv, mode="a", header=hdr, index=False)
        print(f"  fold {fold_i+1}/18 (participant {pid}) done, {len(rows)} rows  [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    import sys
    fs = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fe = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    main(fs, fe)
