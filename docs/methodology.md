# Detailed Methodology

This document describes the staged hybrid imputation method and the **leave-one-participant-out
(LOPO)** evaluation framework used in the revised study. It complements the accompanying *Electronics*
(MDPI) article.

> **Revision note.** An earlier version of this repository evaluated the method with a random
> window-level train/test split and repeated masking seeds, and reported a MICE stage that — as
> implemented — ran on an already-complete matrix and therefore did not change the result. Both issues
> were corrected in this version (see the top-level `README.md`, section "What changed in this
> update"). This document describes the corrected methodology only.

## 1. Overview

The hybrid imputation method combines three established techniques in a staged pipeline:

1. **K-Nearest Neighbors (KNN)** — a local, instance-based first estimate from similar windows.
2. **Support Vector Regression (SVR)** — a global, per-feature refinement of the estimate.
3. **Within-participant MICE refinement** — a final chained-equations pass, *initialized from the
   KNN–SVR estimate*, that re-estimates each missing cell from the held-out participant's own observed
   windows.

The method is evaluated against eight baselines — mean, KNN, SVR, standard MICE, MissForest, and two
temporal baselines (LOCF and linear interpolation along the window sequence) — plus its own two-stage
(KNN–SVR) ablation, on the public **HAR70+** benchmark (primary) and a real-world ambulatory **pilot**
(Dataset B).

## 2. Leave-one-participant-out (leakage-free) protocol

> This is the most important methodological point. Splitting must occur at the **participant** level,
> not the window level, or near-duplicate windows from the same person leak between training and test.

For each of the 18 HAR70+ participants (one **fold**):

1. That participant is **held out** in full as the evaluation set; the remaining **17 participants**
   form the complete, fully observed donor/training set.
2. Every cross-participant component — the `StandardScaler`, the KNN donor pool, the per-feature SVR
   models, and the model-based baselines (MICE, MissForest) — is fit **only on the 17 training
   participants**. No window from the evaluated participant is seen when they are fit.
3. Missing values are introduced **only into the held-out participant's windows**, along the
   time-ordered window sequence (not on shuffled rows), according to the scenario.
4. The within-participant MICE refinement adapts **only to the held-out participant's own observed
   entries**; the masked ground-truth values are excluded from all fitting and updates.
5. The reconstructed values are compared against the held-out ground truth, in standardized (z-score)
   units computed with the training scaler so that heterogeneous sensor channels are comparable.

**Inference is performed across participants.** Point estimates are means over the 18 held-out
participants; 95% confidence intervals come from a **cluster bootstrap that resamples participants**
(5,000 resamples); pairwise comparisons use **two-sided paired permutation tests** (20,000
permutations) on the participant-level mean difference, with **Holm correction** within each
pre-specified family and **Cohen's d_z** as an effect size. Masking seeds are *not* treated as
independent replicates. Seeds are fixed (`1000 + fold` for imputation masks, `2000 + fold` for the
downstream task).

## 3. The staged pipeline

### Stages 1–2 — cross-participant KNN–SVR estimate (`run_mice_functional.knn_svr_core`)
Repeated for five cycles:
1. **KNN step** — for each feature with missing entries, a per-feature KNN regression
   (k = 5, Manhattan distance, uniform weights) predicts the masked values from the other
   (mean-filled) features.
2. **SVR step** — per-feature SVR models (RBF kernel, C = 100, ε = 0.1), fit on the 17 training
   participants, refine the masked positions; the SVR and KNN estimates are blended 0.6 / 0.4
   (SVR / KNN). Observed entries are kept fixed.

### Stage 3 — within-participant MICE refinement (`run_corr_lean.refine`)
Initialized from the stage-1–2 estimate, then for ten cycles each missing feature cell is re-estimated
by a chained **BayesianRidge** regression fit on the **held-out participant's own observed rows**,
using the current estimates of the other features. This is the corrected stage: it operates on the
genuinely missing cells, not on an already-complete matrix.

## 4. Hyper-parameters

| Component | Parameters |
|---|---|
| KNN | `n_neighbors = 5`, `weights = uniform`, `metric = manhattan` |
| SVR | `kernel = rbf`, `C = 100`, `epsilon = 0.1`, `gamma = scale`; one SVR per target feature; features standardized; training rows capped at 800 for the SVR fit (speed only) |
| KNN–SVR estimate | 5 cycles; SVR/KNN blend 0.6 / 0.4 |
| Within-participant refinement | chained `BayesianRidge`, 10 cycles, fit on the held-out participant's observed rows |
| MICE (standalone baseline) | `IterativeImputer`, `estimator = BayesianRidge`, `max_iter = 10` |
| MissForest | `IterativeImputer`, `estimator = RandomForestRegressor`, `max_iter = 5` |
| Temporal baselines | LOCF (`ffill().bfill()`) and linear interpolation, along the time-ordered window sequence |
| RF classifier (downstream) | `n_estimators = 200`, trained on complete training features |
| Preprocessing | `StandardScaler` fit on the 17-participant training set only |

## 5. Realistic missing-data generation

The generator (`realistic_missing_data_generator.py`) is **dataset-agnostic**: it identifies sensor
groups from the column names so a whole group can fail together plausibly. It produces six scenarios
across Rubin's three mechanisms:

- **MCAR — Scenario 1 (Random Sensor Failures):** individual sensors fail for short random periods.
- **MCAR — Scenario 2 (Temporary Connection Issues):** all sensors drop simultaneously for a span.
- **MAR — Scenario 3 (Activity-Dependent Failures):** sensors fail more during specific activities.
- **MAR — Scenario 4 (Battery Depletion):** sensors fail in a power-based order over time.
- **MNAR — Scenario 5 (Value-Dependent Failures):** extreme values are more likely to be missing.
- **MNAR — Scenario 6 (Sensor Range Limitations):** out-of-range values are not recorded.

Scenarios 5 and 6 are MNAR-like: missingness depends on the value that would have been observed, but it
is induced synthetically so ground truth is available for evaluation.

**Nominal vs. realized missingness.** The 10/20/30% values are *nominal intensity parameters*, not
guaranteed global fractions. Because several rules are bounded, the realized fraction is
scenario-dependent (e.g. the activity-dependent rule `min(0.5, 3×rate)` saturates, so the 20% and 30%
settings realize the same ≈14%; value-dependent reaches only ≈2/4/5%). The measured per-participant
realized fractions for all 324 condition×participant cells are in
`results/har70_benchmark/realized_missing_rates_per_participant.csv`.

## 6. Evaluation

1. **Imputation accuracy** — RMSE, MAE, NRMSE, R² between reconstructed and held-out values, in
   standardized units (`run_benchmark_loso.py`, `run_corr_lean.py`).
2. **Statistical inference** — participant cluster bootstrap CIs and paired permutation tests with Holm
   correction and Cohen's d_z (`analyze_results.py`).
3. **Downstream validation** — a Random Forest activity classifier trained on complete training
   features and evaluated on the held-out participant's imputed windows (`run_ds_lean.py`).
4. **Ablation** — two-stage KNN–SVR vs. the full pipeline, and the per-learner contrasts
   (KNN+MICE, SVR+MICE, mean-initialized refinement), all under matched access to observed data
   (`run_corr_lean.py`, `run_svrmice.py`).
5. **Temporal baselines** — LOCF and linear interpolation along the window sequence
   (`run_temporal.py`).

All result CSVs are provided under `results/har70_benchmark/`.
