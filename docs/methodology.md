
# Detailed Methodology for the Hybrid Imputation Method

## Overview

The hybrid imputation method combines three well-established techniques in a sequential pipeline to harness their complementary strengths:

1. **K-Nearest Neighbors (KNN)**: Provides an initial imputation based on similarity between data points  
2. **Support Vector Machine (SVM)**: Refines the imputed values using regression models  
3. **Multiple Imputation by Chained Equations (MICE)**: Ensures consistency across variables through iterative refinement

## Detailed Methodology

### Phase 1: Training on Complete Data

In this phase, SVM regression models are trained using only complete data (observations without missing values):

1. **Data Preprocessing**:
   - Normalize the complete data using `StandardScaler`
   - Split features into predictor variables and target variables

2. **Model Training**:
   - For each variable, train an SVM regression model using all other variables as predictors
   - Store the trained models for use in the imputation phase

### Phase 2: Initial KNN Imputation and SVM Refinement

This phase handles the actual imputation of missing values:

1. **Initial KNN Imputation**:
   - Apply KNN imputation to the incomplete dataset
   - This yields a reasonable first estimate based on similar data points

2. **SVM Refinement**:
   - Apply the pre-trained SVM models to refine the KNN-imputed values
   - For each variable with missing values, make predictions using its corresponding SVM model
   - Replace the KNN-imputed values with the SVM predictions

3. **Rescaling**:
   - Rescale the imputed values to match the original data distribution
   - This ensures that the imputed values preserve the statistical properties of the original data

### Phase 3: Final Refinement with MICE

The final phase ensures consistency across variables:

1. **MICE Imputation**:
   - Use the SVM-refined values as the starting point for MICE
   - Run multiple iterations of chained equations to ensure variable consistency
   - Each variable is modeled as a function of all other variables

2. **Final Output**:
   - The final imputed dataset combines observed values and values refined through MICE

## Implementation Details

### KNN Imputation

- Implementation: `KNNImputer` from `scikit-learn`  
- Parameters:  
  - `n_neighbors`: 5  
  - `weights`: 'uniform'  
  - `metric`: 'nan_euclidean`  

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
