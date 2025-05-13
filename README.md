# Hybrid KNN-SVM-MICE Imputation Method for Wearable Sensor Data

## Overview

This repository contains the implementation, evaluation, and analysis of a novel hybrid imputation method for wearable sensor data in elderly monitoring applications. The hybrid method combines K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Multiple Imputation by Chained Equations (MICE) in a sequential pipeline to leverage their complementary strengths.

The method demonstrates significant improvements over traditional imputation approaches, with an average RMSE reduction of 21.12% compared to KNN and 7.04% compared to SVM across various realistic missing data scenarios.

## Repository Structure

```
hybrid-imputation-method/
├── code/
│   ├── imputation_methods/        # Implementation of imputation methods
│   │   ├── knn_imputation.py      # KNN imputation algorithm
│   │   ├── svm_imputation.py      # SVM imputation algorithm
│   │   └── hybrid_imputation.py   # Hybrid KNN-SVM-MICE imputation algorithm
│   ├── data_generation/           # Code for generating missing data scenarios
│   │   └── realistic_missing_data_generator.py
│   ├── evaluation/                # Code for evaluating imputation methods
│   │   ├── evaluation_framework.py
│   │   ├── analyze_performance.py
│   │   └── generate_hybrid_results_summary.py
│   └── visualization/             # Code for generating figures
│       ├── figure1_general_performance.py
│       ├── figure2_mechanism_performance.py
│       ├── figure3_percentage_impact.py
│       ├── figure4_improvement_heatmap.py
│       ├── figure5_hybrid_architecture.py
│       └── figure6_extreme_values_robustness.py
├── data/
│   ├── original/                  # Original complete dataset
│   │   └── original_data.csv
│   ├── missing_data/              # Generated missing data scenarios
│   │   ├── MCAR_RandomSensorFailures_10pct.csv
│   │   ├── MCAR_RandomSensorFailures_20pct.csv
│   │   └── ...
│   └── imputed/                   # Results of imputation methods
│       ├── knn/
│       ├── svm/
│       └── hybrid/
├── results/
│   ├── tables/                    # Performance comparison tables
│   │   ├── table1_general_performance.md
│   │   ├── table2_mechanism_performance.md
│   │   └── ...
│   ├── figures/                   # Generated figures
│   │   ├── figure1_general_performance.png
│   │   ├── figure2_mechanism_performance.png
│   │   └── ...
│   └── analysis/                  # Detailed analysis results
│       ├── combined_results.csv
│       ├── mechanism_summary.csv
│       └── ...
├── docs/
│   ├── methodology.md             # Detailed methodology description
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Step-by-Step Guide

### 1. Data Preparation

Start with a complete dataset (no missing values) and generate realistic missing data scenarios:

```bash
python code/data_generation/realistic_missing_data_generator.py \
    --input data/original/original_data.csv \
    --output data/missing_data/ \
    --percentages 10 20 30
```

This will create 18 different missing data scenarios based on three mechanisms (MCAR, MAR, MNAR) and six scenario types, each with three different percentages of missing data (10%, 20%, 30%).

### 2. Running Imputation Methods

#### KNN Imputation

```bash
python code/imputation_methods/knn_imputation.py \
    --input data/missing_data/ \
    --output data/imputed/knn/ \
    --original data/original/original_data.csv
```

#### SVM Imputation

```bash
python code/imputation_methods/svm_imputation.py \
    --input data/missing_data/ \
    --output data/imputed/svm/ \
    --original data/original/original_data.csv
```

#### Hybrid Imputation

```bash
python code/imputation_methods/hybrid_imputation.py \
    --input data/missing_data/ \
    --output data/imputed/hybrid/ \
    --original data/original/original_data.csv
```

### 3. Evaluating Performance

Generate performance summary for the hybrid method:

```bash
python code/evaluation/generate_hybrid_results_summary.py \
    --imputed_dir data/imputed/hybrid/ \
    --missing_dir data/missing_data/ \
    --original data/original/original_data.csv \
    --output results/analysis/hybrid_results_summary.csv
```

Analyze performance of all methods:

```bash
python code/evaluation/analyze_performance.py \
    --knn data/imputed/knn/ \
    --svm data/imputed/svm/ \
    --hybrid data/imputed/hybrid/ \
    --original data/original/original_data.csv \
    --output results/analysis/
```

### 4. Generating Visualizations

Generate all figures:

```bash
python code/visualization/figure1_general_performance.py
python code/visualization/figure2_mechanism_performance.py
python code/visualization/figure3_percentage_impact.py
python code/visualization/figure4_improvement_heatmap.py
python code/visualization/figure5_hybrid_architecture.py
python code/visualization/figure6_extreme_values_robustness.py
```

## Methodology

The hybrid imputation method combines three approaches in a sequential pipeline:

### Phase 1: Training on Complete Data
- Normalize the complete data using StandardScaler
- For each variable, train an SVM regression model using all other variables as predictors

### Phase 2: Initial Imputation with KNN and Refinement with SVM
- Apply KNN imputation to the incomplete data
- Apply the pre-trained SVM models to refine the KNN-imputed values
- Rescale the imputed values to match the original data distribution

### Phase 3: Final Refinement with MICE
- Use the SVM-refined values as the starting point for MICE
- Run multiple iterations of chained equations to ensure consistency
- Each variable is modeled as a function of all other variables

For detailed information about the methodology, see [docs/methodology.md](docs/methodology.md).

## Results

The hybrid method demonstrates superior performance compared to both KNN and SVM methods:

- **Overall Performance**: 21.12% improvement over KNN and 7.04% improvement over SVM in terms of RMSE
- **Missing Data Mechanisms**: Best performance in MCAR scenarios (19.68% improvement over SVM)
- **Missing Data Percentage**: Increasing advantage as the percentage of missing data increases
- **Robustness to Extreme Values**: 27.09% improvement over KNN and 10.84% over SVM for values beyond 3 standard deviations

For detailed results and analysis, see the [results](results/) directory.

## Citation

If you use this code or methodology in your research, please cite:

```
@thesis{
  title={Intelligent Systems for Enhanced Elderly Well-being: Advanced Data Imputation Strategies and Virtual Reality Applications for Cognitive Health},
  author={Vasilică-Gabriel SASU},
  year={2025},
  school={University Politehnica of Bucharest}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue on this repository or contact [author@example.com](mailto:author@example.com).
