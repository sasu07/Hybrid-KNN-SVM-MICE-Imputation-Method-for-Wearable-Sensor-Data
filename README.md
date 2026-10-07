# A Realistic Evaluation Framework and Staged Hybrid Pipeline (KNN–SVR–MICE) for Robust Missing-Data Imputation in Wearable Sensor Monitoring of Elderly Populations

This repository contains the implementation, the realistic missing-data evaluation framework, and all
results for a **staged hybrid imputation method** (KNN → SVR → within-participant MICE refinement) for
missing data in wearable-sensor monitoring of older adults.

It accompanies the article submitted to *Electronics* (MDPI). The method is evaluated under a strict
**leave-one-participant-out (LOPO), leakage-free** protocol on two datasets:

- **Dataset A — HAR70+** (public benchmark, 18 older adults, 70–95 years) — the primary methodological
  validation.
- **Dataset B — real-world ambulatory pilot** (proprietary, 3,845 measurements) — a small feasibility
  check on genuinely collected ambulatory data from our own hardware.

---

## What changed in this update

This version supersedes the earlier release. Two substantive corrections were made during peer review;
both are reflected in the code and results here, and the numbers below differ from the earlier version
accordingly.

1. **Participant-level evaluation (was window-level).** The earlier benchmark split the data into
   random train/test **rows** (`TRAIN_FRAC = 0.6`) and repeated it over random seeds. Because
   consecutive windows from the same person are near-duplicates, this leaked information between
   training and test and produced optimistic errors (the previously reported RMSE ≈ 0.18). The
   evaluation is now **leave-one-participant-out**: each of the 18 participants is held out in full
   while the other 17 form the training set, and all statistical inference resamples participants
   (cluster bootstrap, paired permutation tests, Holm correction, Cohen's d_z). The headline RMSE under
   this honest protocol is **0.637**.

2. **Corrected MICE stage.** As previously implemented, the final MICE pass ran on a matrix whose
   cells had already been filled by the KNN–SVR stage, so with no missing entries left to estimate it
   returned the two-stage output unchanged (a no-op). It is now a genuine **within-participant
   refinement initialized from the KNN–SVR estimate**, which produces a statistically significant
   improvement over the two-stage method (0.747 → 0.637, p < 0.001). The ablation (Table 7 in the
   article) now shows that all three stages contribute.

The old window-level scripts and their result CSVs have been removed to avoid confusion; the code and
results in this repository correspond to the revised manuscript.

---

## Table of contents

- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Method summary](#method-summary)
- [Results at a glance](#results-at-a-glance)
- [Mapping results to the article](#mapping-results-to-the-article)
- [Notes on data](#notes-on-data)
- [Citation](#citation)
- [License](#license)

---

## Key features

- **Leave-one-participant-out, leakage-free** evaluation: cross-participant components are fit only on
  the 17 held-in participants, so no window from an evaluated participant is seen during fitting.
- A **dataset-agnostic** realistic missing-data generator producing six device-driven failure
  scenarios across MCAR, MAR, and MNAR, at three nominal intensity settings (10/20/30%); realized
  fractions are measured and reported.
- A complete **evaluation pipeline**: nine-method benchmark (mean, KNN, SVR, MICE, MissForest, two
  temporal baselines, the two-stage KNN–SVR, and the full three-stage method), participant-level
  bootstrap + permutation inference, a matched-access ablation, downstream activity classification, and
  runtime.
- **All result CSVs** are included under `results/`, together with the three summary tables regenerated
  by `analyze_results.py`.

---

## Repository structure

```
.
├── code/
│   └── imputation_methods/
│       ├── 01_extract_har70_features.py     # windowed feature extraction for HAR70+
│       ├── realistic_missing_data_generator.py  # 6 scenarios × MCAR/MAR/MNAR (dataset-agnostic)
│       ├── run_benchmark_loso.py            # baselines + two-stage KNN–SVR, LOPO
│       ├── run_mice_functional.py           # knn_svr_core(): the stage 1–2 estimate
│       ├── run_corr_lean.py                 # corrected Full + KNN+MICE + MICE-within
│       ├── run_svrmice.py                    # SVR+MICE ablation
│       ├── run_temporal.py                   # LOCF + linear interpolation (time axis)
│       ├── run_ds_lean.py                    # downstream activity classification
│       ├── analyze_results.py                # participant-cluster CIs, permutation, Holm
│       └── hybrid_imputation.py              # legacy HybridImputerNoLeak (provenance only*)
├── data/
│   ├── har70/
│   │   └── har70_full.csv                    # 13,571 windows, 18 participants (501–518), 19 features
│   └── original/
│       └── original_data.csv                 # real-world ambulatory pilot (Dataset B)
├── results/
│   ├── har70_benchmark/                      # all LOPO result CSVs (see mapping below)
│   └── pilot_dataset_B/                       # result CSVs on the pilot dataset
├── docs/
│   └── methodology.md                        # detailed methodology (LOPO protocol)
├── reproduce_all.sh                          # end-to-end driver (resume-safe)
├── requirements.txt
├── LICENSE
└── README.md
```

\* `hybrid_imputation.py` is the original `HybridImputerNoLeak` whose MICE pass was the no-op described
above. It is kept only so the correction is auditable; it is **not** the proposed method and is not
called by `reproduce_all.sh`.

---

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Pinned versions used for the reported results: `numpy==1.24.3`, `pandas==2.0.0`,
`scikit-learn==1.2.2`.

---

## Reproducing the results

```bash
bash reproduce_all.sh
```

The driver runs, in order: the baselines + two-stage KNN–SVR, the corrected three-stage method, the
SVR+MICE ablation, the temporal baselines, the downstream task, and finally `analyze_results.py`, which
regenerates `overall_rmse.csv`, `pairwise_full.csv`, and `ablation_family.csv`. All scripts cap BLAS
threads to 1 and append to their output CSV after each fold, skipping participants already present, so
an interrupted run resumes by re-invoking the same command.

Individual stages also run directly, e.g.:

```bash
cd code/imputation_methods
python3 run_corr_lean.py 0 18 ../../results/har70_benchmark/loso_corrected.csv   # fold range [start,end)
python3 analyze_results.py
```

### Dataset A — HAR70+

The windowed feature table `data/har70/har70_full.csv` is included. To regenerate it from the raw
HAR70+ recordings, download the dataset from the
[UCI Machine Learning Repository (ID 780)](https://archive.ics.uci.edu/dataset/780/har70) so that the
18 per-participant files `501.csv … 518.csv` sit in a `har70plus/` folder, then run
`01_extract_har70_features.py`.

### Dataset B — real-world pilot

Included at `data/original/original_data.csv`; results are under `results/pilot_dataset_B/`.

---

## Method summary

The pipeline imputes a multivariate **feature** matrix (window-level HAR70+ features, not raw signals)
in three stages:

1. **KNN estimate** — per-feature k-nearest-neighbour regression (k = 5, Manhattan).
2. **SVR refinement** — per-feature SVR (RBF, C = 100, ε = 0.1), fit across the 17 training
   participants, blended 0.6 / 0.4 with the KNN estimate over five cycles → the **KNN–SVR estimate**.
3. **Within-participant MICE refinement** — initialized from the KNN–SVR estimate, then chained
   BayesianRidge regressions on the held-out participant's own observed windows (10 cycles).

See [`docs/methodology.md`](docs/methodology.md) for the full protocol and the complete hyper-parameter
table.

---

## Results at a glance

**HAR70+ benchmark** (leave-one-participant-out; participant-level means over all 18 participants and
18 conditions; standardized units, lower RMSE is better):

| Method | RMSE | 95% CI |
|---|---|---|
| **KNN–SVR–MICE (proposed)** | **0.637** | [0.604, 0.669] |
| Linear interpolation | 0.694 | [0.656, 0.734] |
| KNN–SVR (two-stage) | 0.747 | [0.692, 0.814] |
| LOCF | 0.781 | [0.732, 0.834] |
| MissForest | 0.952 | [0.882, 1.042] |
| KNN | 0.968 | [0.897, 1.068] |
| MICE | 0.987 | [0.922, 1.060] |
| Mean | 1.222 | [1.127, 1.333] |
| SVR | 1.330 | [1.266, 1.403] |

The proposed method had the lowest RMSE and highest R² (0.62) and significantly outperformed every
baseline (Holm-adjusted p < 0.001; Cohen's d_z from ≈1.0 to 5.0). Linear interpolation was the
strongest baseline and the most accurate method under purely random (MCAR) missingness, while the
proposed method led under the structured MAR and MNAR scenarios. In downstream activity classification
the leading imputers all performed close to the complete-data reference. The same ranking held on the
real-world pilot (Dataset B). Full numbers are in `results/`.

---

## Mapping results to the article

| File in `results/har70_benchmark/` | Article |
|---|---|
| `loso_corrected.csv` | Table 3 (Full = 0.637); Table 7 (KNN+MICE 0.682, MICE-within 0.655) |
| `baselines_and_hybrid_merged.csv` | Table 3 (Mean 1.222, KNN 0.968, SVR 1.330, MICE 0.987, MissForest 0.952, KNN–SVR 0.747) |
| `loso_temporal.csv` | Table 3 (LOCF 0.781, linear interp. 0.694) |
| `loso_svrmice_*.csv` | Table 7 (SVR+MICE 0.697) |
| `loso_downstream_corr.csv` | Section 4.5 (downstream activity classification) |
| `realized_missing_rates_per_participant.csv` | Section 3.2.2 / 4.3 (realized fractions ≈8/15/21%) |
| `overall_rmse.csv`, `pairwise_full.csv`, `ablation_family.csv` | Tables 3–4, 7; abstract/Conclusions statistics |

---

## Notes on data

- **HAR70+** is openly licensed (CC BY 4.0). The windowed feature table is included here for
  convenience; the raw recordings are downloaded from the original source.
- The **pilot dataset** (`data/original/original_data.csv`) comes from our own hardware deployment;
  if you reuse it, please respect the privacy considerations described in the article.

---

## Citation

If you use this code or framework, please cite the accompanying article (details to be added on
publication) and the HAR70+ dataset:

> Logacjov, A.; Bach, K.; Kongsvold, A.; Bårdstu, H.B.; Mork, P.J. HARTH: A Human Activity
> Recognition Dataset for Machine Learning. *Sensors* 2021, 21, 7853. doi:10.3390/s21237853

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue on this repository or contact gabriel.sasuu@gmail.com
