# A Realistic Evaluation Framework and Staged Hybrid Pipeline (KNN–SVR–MICE) for Robust Missing-Data Imputation in Wearable Sensor Monitoring of Elderly Populations

This repository contains the full implementation, the realistic missing-data evaluation framework,
and all results for a **staged hybrid imputation method** (KNN → SVR → MICE) for missing data in
wearable-sensor monitoring of older adults.

It accompanies the article submitted to *Sensors* (MDPI). The method is evaluated under a strict
**leakage-free** protocol on two datasets:

- **Dataset A — HAR70+** (public benchmark, 18 older adults, 70–95 years) — the primary methodological
  validation.
- **Dataset B — real-world ambulatory pilot** (proprietary, 3,845 measurements) — a real-world case
  study from our own hardware implementation.

---

## Table of contents

- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Method summary](#method-summary)
- [Results at a glance](#results-at-a-glance)
- [Notes on data](#notes-on-data)
- [Citation](#citation)
- [License](#license)

---

## Key features

- **Leakage-free** staged hybrid imputer with an explicit `fit` / `transform` split, so imputation
  models never see the ground-truth values they are evaluated on.
- A **dataset-agnostic** realistic missing-data generator producing six device-driven failure
  scenarios across the MCAR, MAR, and MNAR mechanisms, at 10%, 20%, and 30% missing rates.
- A complete **evaluation pipeline**: six-method benchmark (mean, KNN, SVR, MICE, MissForest,
  Hybrid), paired-t-test significance, downstream activity-classification validation, ablation study,
  and runtime measurement.
- **All result CSVs** for both datasets are included under `results/`.

---

## Repository structure

```
.
├── code/
│   └── imputation_methods/
│       ├── hybrid_imputation.py               # leakage-free staged hybrid imputer (KNN→SVR→MICE)
│       ├── realistic_missing_data_generator.py# 6 scenarios × MCAR/MAR/MNAR (dataset-agnostic)
│       ├── run_benchmark.py                    # 6-method imputation benchmark
│       ├── run_downstream.py                   # downstream activity-classification validation
│       ├── run_ablation_runtime.py            # ablation + runtime + per-variable analysis
│       ├── run_all.py                          # runs the full HAR70+ pipeline end to end
│       ├── 01_extract_har70_features.py        # windowed feature extraction for HAR70+
│       ├── knn_imputation.py                   # standalone KNN reference implementation
│       └── svm_imputation.py                   # standalone SVR reference implementation
├── data/
│   └── original/
│       └── original_data.csv                   # real-world ambulatory pilot (Dataset B)
├── results/
│   ├── har70_benchmark/                        # all result CSVs on HAR70+ (primary)
│   │   ├── results_overall.csv
│   │   ├── results_by_mechanism.csv
│   │   ├── results_summary.csv
│   │   ├── results_per_run.csv
│   │   ├── downstream_summary.csv
│   │   ├── downstream_per_run.csv
│   │   ├── ablation_summary.csv
│   │   ├── ablation_by_mechanism.csv
│   │   ├── runtime_summary.csv
│   │   └── per_variable_summary.csv
│   └── pilot_dataset_B/                         # result CSVs on the pilot dataset
│       ├── results_overall.csv
│       └── downstream_summary.csv
├── docs/
│   └── methodology.md                          # detailed methodology (leakage-free protocol)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`.

---

## Reproducing the results

All scripts live in `code/imputation_methods/`. Run them from that directory.

### Dataset A — HAR70+ (primary benchmark)

HAR70+ is a public dataset and is **not redistributed here**. Download it from the
[UCI Machine Learning Repository (ID 780)](https://archive.ics.uci.edu/dataset/780/har70) or the
HARTH/HAR70+ project, and unzip it so that the 18 per-participant files `501.csv … 518.csv` sit in a
folder named `har70plus/` inside `code/imputation_methods/`.

```bash
cd code/imputation_methods
# place ./har70plus/501.csv ... 518.csv here, then:
python run_all.py
```

`run_all.py` runs, in order:

1. `01_extract_har70_features.py` — segments the raw 50 Hz back/thigh accelerometer signals into
   non-overlapping 3-second windows and extracts 19 numeric features → `data/har70_full.csv`.
2. `run_benchmark.py` — six-method imputation benchmark → `results_*.csv`.
3. `run_downstream.py` — activity-classification validation → `downstream_*.csv`.
4. `run_ablation_runtime.py` — ablation, runtime, and per-variable analysis →
   `ablation_*`, `runtime_*`, `per_variable_*`.

You can also run each step individually; each script has a short usage note at the top.

> **Tip:** the benchmark uses `N_SEEDS = 10` by default. For a quicker first pass, lower it near the
> top of `run_benchmark.py`.

### Dataset B — real-world pilot

The pilot dataset is included at `data/original/original_data.csv`. To run the benchmark on it,
set near the top of `run_benchmark.py` (and `run_downstream.py`):

```python
COMPLETE_CSV = "data/original/original_data.csv"
```

and run those scripts as above.

---

## Method summary

The hybrid pipeline (`hybrid_imputation.py`, class `HybridImputerNoLeak`) imputes a multivariate
sensor matrix in three stages, with an explicit `fit` / `transform` split:

1. **KNN initialization** — per-column k-nearest-neighbour regression (k = 5, Manhattan distance)
   provides a first estimate.
2. **SVR refinement** — per-column Support Vector Regression (RBF kernel, C = 100, ε = 0.1) refines
   the estimate over five iterations, blended with the KNN estimate (0.6 / 0.4).
3. **MICE refinement** — a final Multivariate Imputation by Chained Equations pass enforces
   multivariate consistency.

See [`docs/methodology.md`](docs/methodology.md) for the full description, including the leakage-free
evaluation protocol and the complete hyper-parameter table.

---

## Results at a glance

**HAR70+ benchmark** (leakage-free, 18 older adults, 13,571 windows, 10 seeds):

| Method | RMSE | R² |
|---|---|---|
| **Hybrid (proposed)** | **0.178** | **0.866** |
| KNN | 0.218 | 0.804 |
| MissForest | 0.223 | 0.805 |
| MICE | 0.235 | 0.792 |
| SVR | 0.314 | 0.600 |
| Mean | 0.322 | 0.645 |

The hybrid method significantly outperformed every baseline on both RMSE and R² (paired t-tests,
p < 0.001) and led under all three missingness mechanisms. In downstream activity classification it
came closest to the complete-data upper bound (95.7% accuracy vs. 99.5% on complete data). The same
ranking held on the real-world pilot (Dataset B). Full numbers are in `results/`.

---

## Notes on data

- **HAR70+** is openly licensed (CC BY 4.0) but is downloaded from the original source rather than
  redistributed here.
- The **pilot dataset** (`data/original/original_data.csv`) comes from our own hardware deployment.
  If you reuse it, please respect the privacy considerations described in the article.
- Generated intermediate files (e.g. `har70_full.csv`) and per-run outputs are not committed; they
  are regenerated by the scripts.

---

## Citation

If you use this code or framework, please cite the accompanying article (details to be added on
publication) and the HAR70+ dataset:

> Logacjov, A.; Bach, K.; Kongsvold, A.; Bårdstu, H.B.; Mork, P.J. HARTH: A Human Activity
> Recognition Dataset for Machine Learning. *Sensors* 2021, 21, 7853. doi:10.3390/s21237853

---
```
@thesis{
  title={Intelligent Systems for Enhanced Elderly Well-being: Advanced Data Imputation Strategies and Virtual Reality Applications for Cognitive Health},
  author={Vasilică-Gabriel SASU},
  year={2025},
  school={University Politehnica of Bucharest}
}
```
---

## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue on this repository or contact gabriel.sasuu@gmail.com
