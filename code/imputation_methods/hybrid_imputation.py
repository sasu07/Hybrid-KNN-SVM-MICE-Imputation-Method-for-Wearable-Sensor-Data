import pandas as pd
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HybridImputer:
    """
    Hybrid imputation implementation combining KNN, SVM, and MICE for missing data scenarios.
    """
    
    def __init__(self, knn_neighbors=5, svm_kernel='rbf', svm_C=100, svm_epsilon=0.1, max_iter=10):
        """
        Initialize the hybrid imputer.
        
        Parameters:
        -----------
        knn_neighbors : int
            Number of neighbors to use for KNN imputation
        svm_kernel : str
            Kernel type to be used in the SVM algorithm
        svm_C : float
            Regularization parameter for SVM
        svm_epsilon : float
            Epsilon in the epsilon-SVR model
        max_iter : int
            Maximum number of iterations for MICE
        """
        self.knn_neighbors = knn_neighbors
        self.svm_kernel = svm_kernel
        self.svm_C = svm_C
        self.svm_epsilon = svm_epsilon
        self.max_iter = max_iter
        self.execution_time = 0
        
    def impute(self, data, complete_data, output_dir, scenario_name):
        """
        Impute missing values using the hybrid approach.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values to impute
        complete_data : pandas.DataFrame
            Original complete data for evaluation
        output_dir : str
            Directory to save imputed data
        scenario_name : str
            Name of the scenario being imputed
            
        Returns:
        --------
        pandas.DataFrame
            Data with imputed values
        dict
            Performance metrics
        """
        print(f"Imputing {scenario_name} using Hybrid KNN-SVM-MICE approach...")
        start_time = time.time()
        
        # Create a copy of the data
        imputed_data = data.copy()
        
        # Select numerical columns for imputation
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove metadata columns if they happen to be numerical
        metadata_columns = ['Activity', 'time']
        for col in metadata_columns:
            if col in numerical_columns:
                numerical_columns.remove(col)
        
        # Check if there are any missing values to impute
        total_missing = imputed_data[numerical_columns].isna().sum().sum()
        if total_missing == 0:
            print(f"No missing values to impute in {scenario_name}")
            # Return empty metrics if no imputation was performed
            metrics = {
                'rmse': {},
                'mae': {},
                'r2': {},
                'overall_rmse': 0,
                'overall_mae': 0,
                'overall_r2': 1,  # Perfect score since no imputation needed
                'execution_time': time.time() - start_time
            }
            return imputed_data, metrics
        
        # Store original missing values for evaluation
        missing_mask = {}
        original_values = {}
        
        for column in numerical_columns:
            missing_mask[column] = imputed_data[column].isna()
            if missing_mask[column].sum() > 0:
                original_values[column] = complete_data.loc[missing_mask[column], column].values
        
        # Step 1: Initial KNN imputation
        print(f"  Step 1: Initial KNN imputation...")
        knn_imputed = self._knn_imputation(imputed_data[numerical_columns])
        
        # Step 2: SVM refinement
        print(f"  Step 2: SVM refinement...")
        svm_imputed = self._svm_refinement(knn_imputed, complete_data[numerical_columns])
        
        # Step 3: Final MICE imputation
        print(f"  Step 3: Final MICE imputation...")
        final_imputed = self._mice_imputation(svm_imputed)
        
        # Replace original data with imputed data
        imputed_data[numerical_columns] = final_imputed
        
        # Calculate execution time
        self.execution_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(imputed_data, complete_data, missing_mask, original_values, numerical_columns)
        
        # Save the imputed data
        output_path = os.path.join(output_dir, f"{scenario_name}_Hybrid.csv")
        imputed_data.to_csv(output_path, index=False)
        
        return imputed_data, metrics
    
    def _knn_imputation(self, data):
        """
        Perform KNN imputation on the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values
            
        Returns:
        --------
        pandas.DataFrame
            Data with KNN imputed values
        """
        # Create a copy of the data
        imputed_data = data.copy()
        
        # Loop over each column to impute missing values
        for column in data.columns:
            missing_mask = data[column].isna()
            
            # Only fit the model if there are missing values in the column
            if missing_mask.sum() > 0:
                # Create a feature set excluding the column we're imputing
                X_impute = data.drop(columns=[column])
                
                # Fill temporary missing values in the other columns with mean
                X_impute_filled = X_impute.fillna(X_impute.mean())
                
                # Train KNeighborsRegressor on the non-missing data
                knn = KNeighborsRegressor(
                    n_neighbors=self.knn_neighbors, 
                    metric='manhattan',
                    weights='uniform'
                )
                knn.fit(X_impute_filled[~missing_mask], data.loc[~missing_mask, column])
                
                # Predict missing values
                imputed_data.loc[missing_mask, column] = knn.predict(X_impute_filled[missing_mask])
        
        return imputed_data
    
    def _svm_refinement(self, data, complete_data):
        """
        Refine imputed values using SVM.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with initial imputed values
        complete_data : pandas.DataFrame
            Original complete data for training
            
        Returns:
        --------
        pandas.DataFrame
            Data with SVM refined values
        """
        # Create a copy of the data
        refined_data = data.copy()
        
        # Standardize both datasets for consistency
        scaler = StandardScaler()
        complete_scaled = pd.DataFrame(
            scaler.fit_transform(complete_data), 
            columns=complete_data.columns
        )
        
        # Train SVR models on complete data
        svm_models = {}
        for column in data.columns:
            # Train a model for each column using the other columns as features
            X_train = complete_scaled.drop(columns=[column])
            y_train = complete_scaled[column]
            
            # Train the SVR model
            svr = SVR(kernel=self.svm_kernel, C=self.svm_C, epsilon=self.svm_epsilon)
            svr.fit(X_train, y_train)
            
            # Store the trained model
            svm_models[column] = svr
        
        # Standardize the data with initial imputation
        data_scaled = pd.DataFrame(
            scaler.transform(data), 
            columns=data.columns
        )
        
        # Identify initially imputed values (those that were missing in the original data)
        missing_mask = {}
        for column in data.columns:
            missing_mask[column] = data[column].isna().fillna(False)  # This should be all False now since we've done KNN imputation
        
        # Refine initially imputed values using the trained models
        for column in data.columns:
            # Create a feature set excluding the column to be refined
            X_refine = data_scaled.drop(columns=[column])
            
            # Predict refined values using the trained model
            preds_scaled = svm_models[column].predict(X_refine)
            
            # Transform predictions back to original scale
            col_idx = list(data.columns).index(column)
            preds = preds_scaled * scaler.scale_[col_idx] + scaler.mean_[col_idx]
            
            # Update only the initially imputed values
            # We'll use a weighted average of KNN and SVM predictions for initially missing values
            initially_missing = missing_mask[column]
            if initially_missing.sum() > 0:
                # 60% weight to SVM, 40% to KNN
                refined_data.loc[initially_missing, column] = 0.6 * preds[initially_missing] + 0.4 * refined_data.loc[initially_missing, column]
        
        return refined_data
    
    def _mice_imputation(self, data):
        """
        Perform final MICE imputation on the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with initial imputed values
            
        Returns:
        --------
        pandas.DataFrame
            Data with MICE imputed values
        """
        # Create a copy of the data
        imputed_data = data.copy()
        
        # Create and fit the MICE imputer
        mice_imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=42,
            estimator=SVR(kernel=self.svm_kernel, C=self.svm_C, epsilon=self.svm_epsilon)
        )
        
        # Fit and transform the data
        imputed_values = mice_imputer.fit_transform(imputed_data)
        
        # Convert back to DataFrame
        imputed_data = pd.DataFrame(imputed_values, columns=data.columns)
        
        return imputed_data
    
    def _calculate_metrics(self, imputed_data, complete_data, missing_mask, original_values, numerical_columns):
        """
        Calculate performance metrics for the imputation.
        
        Parameters:
        -----------
        imputed_data : pandas.DataFrame
            Data with imputed values
        complete_data : pandas.DataFrame
            Original complete data
        missing_mask : dict
            Dictionary of boolean masks indicating missing values
        original_values : dict
            Dictionary of original values for missing data
        numerical_columns : list
            List of numerical columns
            
        Returns:
        --------
        dict
            Performance metrics
        """
        metrics = {
            'rmse': {},
            'mae': {},
            'r2': {},
            'execution_time': self.execution_time
        }
        
        # Check if there are any values to evaluate
        if not original_values:
            # No missing values were imputed
            metrics['overall_rmse'] = 0
            metrics['overall_mae'] = 0
            metrics['overall_r2'] = 1  # Perfect score since no imputation needed
            return metrics
        
        # Calculate metrics for each column
        for column in numerical_columns:
            if column in missing_mask and missing_mask[column].sum() > 0:
                imputed_values = imputed_data.loc[missing_mask[column], column].values
                
                # Calculate RMSE
                metrics['rmse'][column] = np.sqrt(mean_squared_error(original_values[column], imputed_values))
                
                # Calculate MAE
                metrics['mae'][column] = mean_absolute_error(original_values[column], imputed_values)
                
                # Calculate R2
                metrics['r2'][column] = r2_score(original_values[column], imputed_values)
        
        # Calculate overall metrics
        all_original = np.concatenate([original_values[col] for col in original_values.keys()])
        all_imputed = np.concatenate([imputed_data.loc[missing_mask[col], col].values for col in original_values.keys()])
        
        metrics['overall_rmse'] = np.sqrt(mean_squared_error(all_original, all_imputed))
        metrics['overall_mae'] = mean_absolute_error(all_original, all_imputed)
        metrics['overall_r2'] = r2_score(all_original, all_imputed)
        
        return metrics

def run_hybrid_imputation():
    """
    Run hybrid imputation on all missing data scenarios.
    """
    # Set paths
    scenarios_dir = '/home/ubuntu/research_analysis/missing_data_scenarios'
    output_dir = '/home/ubuntu/research_analysis/imputed_data/hybrid'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scenario files (exclude summary files and original data)
    scenario_files = [f for f in os.listdir(scenarios_dir) 
                     if f.endswith('.csv') 
                     and f != 'original_data.csv'
                     and not f.startswith('scenarios_summary')
                     and not f.startswith('missing_percentages')]
    
    # Load complete data for training and evaluation
    complete_data_path = os.path.join(scenarios_dir, 'original_data.csv')
    complete_data = pd.read_csv(complete_data_path)
    
    # Create hybrid imputer
    imputer = HybridImputer(
        knn_neighbors=5,
        svm_kernel='rbf',
        svm_C=100,
        svm_epsilon=0.1,
        max_iter=10
    )
    
    # Initialize results dictionary
    results = {}
    
    # Process each scenario
    for scenario_file in scenario_files:
        # Extract scenario name (without .csv extension)
        scenario_name = os.path.splitext(scenario_file)[0]
        
        # Load scenario data
        scenario_path = os.path.join(scenarios_dir, scenario_file)
        data = pd.read_csv(scenario_path)
        
        # Impute missing values
        _, metrics = imputer.impute(data, complete_data, output_dir, scenario_name)
        
        # Store results
        results[scenario_name] = metrics
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Scenario': list(results.keys()),
        'RMSE': [results[s]['overall_rmse'] for s in results.keys()],
        'MAE': [results[s]['overall_mae'] for s in results.keys()],
        'R2': [results[s]['overall_r2'] for s in results.keys()],
        'Execution_Time': [results[s]['execution_time'] for s in results.keys()]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'hybrid_results_summary.csv'), index=False)
    
    # Create detailed results with per-column metrics
    detailed_results = {}
    
    # Get all columns
    all_columns = set()
    for scenario in results:
        all_columns.update(results[scenario]['rmse'].keys())
    
    # Initialize detailed results
    for metric in ['rmse', 'mae', 'r2']:
        detailed_results[metric] = pd.DataFrame(index=list(all_columns))
    
    # Fill detailed results
    for scenario in results:
        for metric in ['rmse', 'mae', 'r2']:
            for column in results[scenario][metric]:
                detailed_results[metric].loc[column, scenario] = results[scenario][metric][column]
    
    # Save detailed results
    for metric in detailed_results:
        detailed_results[metric].to_csv(os.path.join(output_dir, f'hybrid_detailed_{metric}.csv'))
    
    print(f"Hybrid imputation completed for {len(scenario_files)} scenarios.")
    print(f"Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    run_hybrid_imputation()
