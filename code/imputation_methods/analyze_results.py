#!/usr/bin/env python3
# Copyright (c) 2025 Gabriel-Vasilica Sasu
# analyze_results.py
# ---------------------------------------------------------------------------
# Participant-level statistical analysis for the KNN-SVR-MICE revision.
#
# This script reproduces the manuscript's headline numbers from the result
# CSVs in ../results/. The independent unit is the PARTICIPANT (18 clusters);
# every interval resamples participants (cluster bootstrap) and every p-value
# comes from a two-sided paired permutation test over the 18 participant-level
# mean differences, with Holm correction inside each pre-specified family.
#
# It intentionally does NOT re-use the old HybridImputerNoLeak path: the
# proposed method here is the corrected three-stage pipeline (KNN-SVR estimate
# -> within-participant MICE refinement INITIALIZED from that estimate), whose
# per-participant RMSE lives in ../results/loso_corrected.csv (method == 'Full').
#
# Inputs (all produced by the run_*.py scripts):
#   ../results/loso_corrected.csv              Full, KNN_MICE, MICE_within
#   ../results/baselines_and_hybrid_merged.csv Mean, KNN, SVR, MICE, MissForest, KNN_SVR
#   ../results/loso_temporal.csv               LOCF, LinInterp
#   ../results/loso_svrmice_A.csv / _B.csv     SVR_MICE ablation
#
# Outputs (written to ../results/):
#   overall_rmse.csv      per-method mean RMSE with participant-cluster 95% CI
#   pairwise_full.csv     Full vs each Table-3 baseline (diff, CI, d_z, Holm p)
#   ablation_family.csv   Full vs two-stage / KNN+MICE / SVR+MICE / mean-init
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# results live at <repo>/results/har70_benchmark (two levels up from this file)
def _find_results():
    for cand in (os.path.join(HERE, "..", "..", "results", "har70_benchmark"),
                 os.path.join(HERE, "..", "results", "har70_benchmark"),
                 os.path.join(HERE, "..", "results"),
                 os.path.join("results", "har70_benchmark")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(HERE, "..", "..", "results", "har70_benchmark")
RES = _find_results()
RNG = np.random.default_rng(42)
NBOOT = 5000
NPERM = 20000


def load_participant_means():
    """Return a wide table: index=participant, columns=method, values=mean RMSE
    averaged over that participant's 18 conditions (6 scenarios x 3 settings)."""
    frames = []

    corr = pd.read_csv(os.path.join(RES, "loso_corrected.csv"))
    frames.append(corr[["method", "participant", "rate", "scenario", "RMSE"]])

    base = pd.read_csv(os.path.join(RES, "baselines_and_hybrid_merged.csv"))
    base = base[base.method.isin(["Mean", "KNN", "SVR", "MICE", "MissForest", "KNN_SVR"])]
    frames.append(base[["method", "participant", "rate", "scenario", "RMSE"]])

    temp = pd.read_csv(os.path.join(RES, "loso_temporal.csv"))
    frames.append(temp[["method", "participant", "rate", "scenario", "RMSE"]])

    # SVR+MICE ablation: accept a single loso_svrmice.csv (fresh run) or the
    # fold-split _A/_B pair (shipped pre-computed results).
    sv_parts = [os.path.join(RES, n) for n in
                ("loso_svrmice.csv", "loso_svrmice_A.csv", "loso_svrmice_B.csv")]
    sv_frames = [pd.read_csv(p) for p in sv_parts if os.path.exists(p)]
    if sv_frames:
        sv = pd.concat(sv_frames, ignore_index=True).drop_duplicates(
            subset=["method", "participant", "rate", "scenario"])
        frames.append(sv[["method", "participant", "rate", "scenario", "RMSE"]])

    allrows = pd.concat(frames, ignore_index=True)
    pm = (allrows.groupby(["method", "participant"]).RMSE.mean()
          .reset_index())
    return pm


def cluster_ci(vals):
    v = np.asarray(vals, float)
    boot = np.array([np.nanmean(RNG.choice(v, size=len(v), replace=True))
                     for _ in range(NBOOT)])
    return float(np.nanmean(v)), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def paired_test(a, b):
    """Paired over participants. CI via cluster bootstrap of the mean diff;
    p via two-sided sign-flip permutation; effect size Cohen's d_z."""
    d = np.asarray(a, float) - np.asarray(b, float)
    diff = float(np.mean(d))
    boot = np.array([np.mean(RNG.choice(d, size=len(d), replace=True)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    sd = np.std(d, ddof=1)
    dz = diff / sd if sd > 0 else np.nan
    obs = abs(diff)
    cnt = sum(abs(np.mean(d * RNG.choice([1, -1], size=len(d)))) >= obs for _ in range(NPERM))
    p = (cnt + 1) / (NPERM + 1)
    return diff, float(lo), float(hi), float(dz), float(p)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (len(pvals) - rank) * pvals[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def main():
    pm = load_participant_means()
    wide = pm.pivot(index="participant", columns="method", values="RMSE")

    # ---- overall table ----
    order = ["Full", "KNN_SVR", "LinInterp", "LOCF", "MissForest", "KNN", "MICE", "SVR", "Mean"]
    rows = []
    for m in order:
        if m not in wide.columns:
            continue
        mean, lo, hi = cluster_ci(wide[m].dropna().values)
        rows.append(dict(method=m, RMSE=round(mean, 3),
                         CI_low=round(lo, 3), CI_high=round(hi, 3)))
    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(RES, "overall_rmse.csv"), index=False)
    print("=== Overall RMSE (participant-cluster 95% CI) ===")
    print(overall.to_string(index=False))

    # ---- Full vs Table-3 baselines (one family, Holm within) ----
    comps = [c for c in ["KNN", "SVR", "MICE", "MissForest", "LOCF", "LinInterp", "Mean", "KNN_SVR"]
             if c in wide.columns]
    recs = []
    for b in comps:
        diff, lo, hi, dz, p = paired_test(wide["Full"].values, wide[b].values)
        recs.append(dict(A="Full", B=b, mean_diff=round(diff, 4),
                         CI_low=round(lo, 4), CI_high=round(hi, 4),
                         cohens_dz=round(dz, 3), p_perm=p))
    pw = pd.DataFrame(recs)
    pw["p_holm"] = holm(pw.p_perm.values).round(5)
    pw["p_perm"] = pw.p_perm.round(5)
    pw.to_csv(os.path.join(RES, "pairwise_full.csv"), index=False)
    print("\n=== Full vs baselines (paired, Holm within family) ===")
    print(pw.to_string(index=False))

    # ---- ablation family: Full vs two-stage / KNN+MICE / SVR+MICE / mean-init ----
    abl_pairs = [("KNN_SVR", "two-stage KNN-SVR (no refinement)"),
                 ("KNN_MICE", "KNN estimate + refinement"),
                 ("SVR_MICE", "SVR estimate + refinement"),
                 ("MICE_within", "refinement from mean init")]
    recs = []
    for key, desc in abl_pairs:
        if key not in wide.columns:
            continue
        diff, lo, hi, dz, p = paired_test(wide["Full"].values, wide[key].values)
        recs.append(dict(contrast=f"Full - {key}", description=desc,
                         mean_diff=round(diff, 4), CI_low=round(lo, 4),
                         CI_high=round(hi, 4), cohens_dz=round(dz, 3), p_perm=p))
    abl = pd.DataFrame(recs)
    abl["p_holm"] = holm(abl.p_perm.values).round(5)
    abl["p_perm"] = abl.p_perm.round(5)
    abl.to_csv(os.path.join(RES, "ablation_family.csv"), index=False)
    print("\n=== Ablation family (Full vs each reduced variant) ===")
    print(abl.to_string(index=False))

    print("\nSaved: overall_rmse.csv, pairwise_full.csv, ablation_family.csv")


if __name__ == "__main__":
    main()
