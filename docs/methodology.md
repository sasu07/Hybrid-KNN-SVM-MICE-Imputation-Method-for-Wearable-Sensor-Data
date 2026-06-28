# Detailed Methodology

This document describes the staged hybrid imputation method and the leakage-free evaluation
framework in detail. It complements the accompanying *Sensors* article.

## 1. Overview

The hybrid imputation method combines three established techniques in a staged, iterative pipeline
to harness their complementary strengths:

1. **K-Nearest Neighbors (KNN)** — a local, instance-based first estimate from similar observations.
2. **Support Vector Regression (SVR)** — a global, model-based refinement of the estimate.
3. **Multivariate Imputation by Chained Equations (MICE)** — a final pass enforcing multivariate
   consistency.

The method is evaluated against five baselines (mean, KNN, SVR, MICE, MissForest) under a realistic,
**leakage-free** protocol, on two datasets: the public **HAR70+** benchmark (primary) and a
real-world ambulatory **pilot** (Dataset B).

## 2. Leakage-free evaluation protocol

> This is the most important methodological point. Imputation models must never see the ground-truth
> values they are later evaluated on. We enforce a strict separation:

For each scenario × missing-rate × seed:

1. The complete dataset is split into a **training subset** and a disjoint **evaluation subset**.
2. All models that require complete data (the `StandardScaler`, the SVR models, MICE, MissForest)
   are fit **only on the training subset**.
3. Missingness is introduced **only into the evaluation subset**, according to the scenario.
4. The original values at the masked positions are held out as ground truth and are **never exposed**
   to any model during fitting or imputation.
5. The trained models impute the masked positions, and the reconstructed values are compared against
   the held-out ground truth.

Each condition is repeated over multiple random seeds; results are reported as mean ± standard
deviation. For efficiency, the SVR models cap their *training* size at a fixed number of rows; this
affects only model fitting, never the evaluation data, and does not introduce leakage.

## 3. The staged hybrid pipeline

The imputer (`HybridImputerNoLeak`) exposes a `fit` / `transform` interface.

### `fit(X_train)`
- Fit a `StandardScaler` on the (complete) training subset.
- For each column *j*, fit a per-column **SVR** model (RBF kernel, C = 100, ε = 0.1) that predicts
  column *j* from all other columns, on the standardized training data.

### `transform(X)` — impute an unseen matrix with NaNs
Repeated for five iterations:
1. **KNN step** — for each column with missing entries, a per-column KNN regression
   (k = 5, Manhattan distance, uniform weights) predicts the masked values from the other
   (mean-filled) columns.
2. **SVR step** — the pre-fit per-column SVR models refine the masked positions. The SVR and KNN
   estimates are blended with weights **0.6 / 0.4** (SVR / KNN).
3. Observed entries are kept fixed throughout.

Finally, a **MICE** pass (`IterativeImputer`) is applied to the current estimate to enforce
multivariate consistency, and the result is inverse-transformed to the original scale.

## 4. Hyper-parameters

| Component | Parameters |
|---|---|
| KNN | `n_neighbors = 5`, `weights = uniform`, `metric = manhattan` |
| SVR | `kernel = rbf`, `C = 100`, `epsilon = 0.1`, `gamma = scale`; one SVR per target column; features standardized |
| Hybrid | 5 KNN→SVR iterations; SVR/KNN blend 0.6 / 0.4; followed by MICE |
| MICE (in hybrid) | `IterativeImputer` (scikit-learn), `estimator = SVR(rbf, C=100, ε=0.1)`, `max_iter = 10` |
| MICE (baseline) | `IterativeImputer`, `estimator = BayesianRidge`, `max_iter = 10` |
| MissForest | `IterativeImputer`, `estimator = RandomForestRegressor`, `max_iter = 10` |
| RF classifier (downstream) | `n_estimators = 200`, trained on complete training features |
| Preprocessing | `StandardScaler` fit on the training subset only |

## 5. Realistic missing-data generation

The generator (`realistic_missing_data_generator.py`) is **dataset-agnostic**: it automatically
identifies sensor groups from the column names so that an entire group can fail together in a
physically plausible way. It produces six scenarios across the three mechanisms of Rubin's taxonomy:

- **MCAR — Scenario 1 (Random Sensor Failures):** individual sensors fail for short random periods.
- **MCAR — Scenario 2 (Temporary Connection Issues):** all sensors drop simultaneously for a span.
- **MAR — Scenario 3 (Activity-Dependent Failures):** sensors fail more during specific activities.
- **MAR — Scenario 4 (Battery Depletion):** sensors fail in a power-based order over time.
- **MNAR — Scenario 5 (Value-Dependent Failures):** extreme values are more likely to be missing.
- **MNAR — Scenario 6 (Sensor Range Limitations):** out-of-range values are not recorded.

Scenarios 5 and 6 are described as *MNAR-like*: the missingness depends on the value that would have
been observed (the defining feature of MNAR), but it is induced synthetically so that ground truth is
available for evaluation.

Each scenario is applied at three missing rates (10%, 20%, 30%), giving 18 test conditions.

## 6. Evaluation

1. **Imputation accuracy** — RMSE, MAE, NRMSE, R² between reconstructed and held-out ground-truth
   values (`run_benchmark.py`).
2. **Statistical significance** — paired t-tests between the hybrid method and each baseline, pairing
   observations by scenario, rate, and seed.
3. **Downstream validation** — a Random Forest activity classifier trained on complete training
   features and evaluated on imputed evaluation data, compared against a complete-data upper bound
   (`run_downstream.py`).
4. **Ablation** — the seven component combinations (KNN, SVR, MICE, and their pairwise and full
   combinations) (`run_ablation_runtime.py`).
5. **Runtime** — mean imputation time per method (`run_ablation_runtime.py`).

All result CSVs are provided under `results/`.

### SVM Regression

- Implementation: `SVR` from `scikit-learn`  
- Parameters:  
  - `kernel`: 'rbf'  
  - `C`: 100  
  - `gamma`: 'scale'  
  - `epsilon`: 0.1  

### MICE Imputation

- Implementation: `IterativeImputer` from `statsmodels`  
- Parameters:  
  - `max_iter`: 10  
  - `n_nearest_features`: 10  
  - `sample_posterior`: True  
  - `random_state`: 42  

## Evaluation Framework

The method is evaluated using a comprehensive framework:

1. **Data Preparation**:
   - Start with a complete dataset (without missing values)
   - Introduce missing values based on realistic scenarios
   - Apply imputation methods
   - Compare imputed values with original values

2. **Evaluation Metrics**:
   - Root Mean Square Error (RMSE)  
   - Mean Absolute Error (MAE)  
   - Coefficient of Determination (R²)  
   - Execution Time  

3. **Missing Data Scenarios**:
   - Three mechanisms: MCAR, MAR, MNAR  
   - Six scenario types based on realistic sensor failure modes  
   - Three missing data percentages: 10%, 20%, 30%
