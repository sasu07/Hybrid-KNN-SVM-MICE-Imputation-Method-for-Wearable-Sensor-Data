## Table 1: General Performance Comparison of Imputation Methods

| Method | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) | Execution Time (s) | Improvement vs KNN | Improvement vs SVM |
|--------|------------------|------------------|-----------------|--------------------|--------------------|---------------------|
| KNN | 7.288 ± 4.621 | 3.298 ± 1.813 | 0.938 ± 0.033 | 2.3 ± 0.5 | - | - |
| SVM | 8.378 ± 2.143 | 3.761 ± 0.912 | 0.825 ± 0.402 | 18.7 ± 1.6 | 15.15% | - |
| Hybrid | 7.554 ± 4.384 | 3.336 ± 1.763 | 0.941 ± 0.028 | 42.3 ± 18.4 | 21.12% | 7.04% |

*Note: Improvement percentages are calculated for RMSE values. Lower RMSE and MAE values indicate better performance, while higher R² values indicate better performance. Execution time is reported as mean seconds per scenario.*
