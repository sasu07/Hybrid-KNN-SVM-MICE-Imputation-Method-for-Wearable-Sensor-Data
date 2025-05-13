import pandas as pd
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class KNNImputer:
    """
    KNN imputation implementation for missing data scenarios.
    """
    
    def __init__(self, n_neighbors=5, metric='manhattan', weights='uniform'):
        """
        Initialize the KNN imputer.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use for imputation
        metric : str
            Distance metric to use for nearest neighbors
        weights : str
            Weight function used in prediction
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.execution_time = 0
        
    def impute(self, data, output_dir, scenario_name):
        """
        Impute missing values using KNN.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values to impute
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
        print(f"Imputing {scenario_name} using KNN...")
        start_time = time.time()
        
        # Create a copy of the data
        imputed_data = data.copy()
        
        # Select numerical columns for imputation
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove metadata columns if they happen to be numerical
        metadata_columns = ['Activity', 'time']
        for col in metadata_columns:
            if col in numerical_columns and col in numerical_columns:
                numerical_columns.remove(col)
        
        # Store original missing values for evaluation
        missing_mask = {}
        original_values = {}
        
        # Create a complete dataset for evaluation
        # We'll use this to compare imputed values with actual values
        complete_data_path = '/home/ubuntu/research_analysis/missing_data_scenarios/original_data.csv'
        complete_data = pd.read_csv(complete_data_path)
        
        # Prepare data for KNeighborsRegressor
        X = imputed_data[numerical_columns].copy()
        
        # Check if there are any missing values to impute
        total_missing = X.isna().sum().sum()
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
        
        # Loop over each column to impute missing values
        for column in numerical_columns:
            missing_mask[column] = X[column].isna()
            
            # Store original values for evaluation
            if missing_mask[column].sum() > 0:
                original_values[column] = complete_data.loc[missing_mask[column], column].values
            
            # Only fit the model if there are missing values in the column
            if missing_mask[column].sum() > 0:
                # Create a feature set excluding the column we're imputing
                X_impute = X.drop(columns=[column])
                
                # Fill temporary missing values in the other columns with mean
                X_impute_filled = X_impute.fillna(X_impute.mean())
                
                # Train KNeighborsRegressor on the non-missing data
                knn = KNeighborsRegressor(
                    n_neighbors=self.n_neighbors, 
                    metric=self.metric,
                    weights=self.weights
                )
                knn.fit(X_impute_filled[~missing_mask[column]], X.loc[~missing_mask[column], column])
                
                # Predict missing values
                X.loc[missing_mask[column], column] = knn.predict(X_impute_filled[missing_mask[column]])
        
        # Replace original data with imputed data
        imputed_data[numerical_columns] = X
        
        # Calculate execution time
        self.execution_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(imputed_data, complete_data, missing_mask, original_values, numerical_columns)
        
        # Save the imputed data
        output_path = os.path.join(output_dir, f"{scenario_name}_KNN.csv")
        imputed_data.to_csv(output_path, index=False)
        
        return imputed_data, metrics
    
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

def run_knn_imputation():
    """
    Run KNN imputation on all missing data scenarios.
    """
    # Set paths
    scenarios_dir = '/home/ubuntu/research_analysis/missing_data_scenarios'
    output_dir = '/home/ubuntu/research_analysis/imputed_data/knn'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scenario files (exclude summary files and original data)
    scenario_files = [f for f in os.listdir(scenarios_dir) 
                     if f.endswith('.csv') 
                     and f != 'original_data.csv'
                     and not f.startswith('scenarios_summary')
                     and not f.startswith('missing_percentages')]
    
    # Create KNN imputer
    imputer = KNNImputer(n_neighbors=5, metric='manhattan', weights='uniform')
    
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
        _, metrics = imputer.impute(data, output_dir, scenario_name)
        
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
    
    results_df.to_csv(os.path.join(output_dir, 'knn_results_summary.csv'), index=False)
    
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
        detailed_results[metric].to_csv(os.path.join(output_dir, f'knn_detailed_{metric}.csv'))
    
    print(f"KNN imputation completed for {len(scenario_files)} scenarios.")
    print(f"Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    run_knn_imputation()
