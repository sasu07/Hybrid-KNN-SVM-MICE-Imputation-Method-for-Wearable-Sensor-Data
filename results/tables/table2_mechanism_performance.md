## Table 2: Performance by Missing Data Mechanism

| Mechanism | KNN RMSE | SVM RMSE | Hybrid RMSE | Improvement vs KNN | Improvement vs SVM |
|-----------|----------|----------|-------------|---------------------|---------------------|
| MCAR | 6.761 ± 2.354 | 8.247 ± 2.012 | 6.624 ± 2.637 | 2.02% | 19.68% |
| MAR | 8.548 ± 7.983 | 7.882 ± 2.784 | 8.288 ± 7.248 | 3.04% | -5.15% |
| MNAR | 8.554 ± 1.213 | 9.008 ± 1.315 | 8.752 ± 1.603 | -2.31% | 2.84% |
| **Average** | **7.954** | **8.379** | **7.888** | **0.83%** | **5.86%** |

*Note: Values are presented as mean ± standard deviation. MCAR = Missing Completely At Random, MAR = Missing At Random, MNAR = Missing Not At Random. Improvement percentages are calculated for RMSE values (lower is better).*
