import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def analyze_performance():
    """
    Analyze the performance of KNN, SVM, and Hybrid imputation methods across all scenarios.
    """
    # Set paths
    knn_results_path = 'research_analysis/imputed_data/knn/knn_results_summary.csv'
    svm_results_path = 'research_analysis/imputed_data/svm/svm_results_summary.csv'
    hybrid_results_path = 'research_analysis/imputed_data/hybrid/hybrid_results_summary.csv'
    output_dir = 'research_analysis/analysis_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all result files exist
    if not os.path.exists(knn_results_path):
        print(f"KNN results file not found: {knn_results_path}")
        return
    
    if not os.path.exists(svm_results_path):
        print(f"SVM results file not found: {svm_results_path}")
        return
    
    # For hybrid, we'll check if it exists but proceed even if it doesn't
    hybrid_exists = os.path.exists(hybrid_results_path)
    if not hybrid_exists:
        print(f"Hybrid results file not found: {hybrid_results_path}. Will proceed with KNN and SVM only.")
    
    # Load results
    knn_results = pd.read_csv(knn_results_path)
    svm_results = pd.read_csv(svm_results_path)
    
    if hybrid_exists:
        hybrid_results = pd.read_csv(hybrid_results_path)
    
    # Prepare combined results dataframe
    scenarios = knn_results['Scenario'].tolist()
    
    # Initialize combined results
    combined_results = pd.DataFrame({
        'Scenario': scenarios,
        'KNN_RMSE': knn_results['RMSE'].tolist(),
        'KNN_MAE': knn_results['MAE'].tolist(),
        'KNN_R2': knn_results['R2'].tolist(),
        'KNN_Time': knn_results['Execution_Time'].tolist(),
        'SVM_RMSE': svm_results['RMSE'].tolist(),
        'SVM_MAE': svm_results['MAE'].tolist(),
        'SVM_R2': svm_results['R2'].tolist(),
        'SVM_Time': svm_results['Execution_Time'].tolist()
    })
    
    if hybrid_exists:
        combined_results['Hybrid_RMSE'] = hybrid_results['RMSE'].tolist()
        combined_results['Hybrid_MAE'] = hybrid_results['MAE'].tolist()
        combined_results['Hybrid_R2'] = hybrid_results['R2'].tolist()
        combined_results['Hybrid_Time'] = hybrid_results['Execution_Time'].tolist()
    
    # Add scenario metadata
    combined_results['Missing_Mechanism'] = combined_results['Scenario'].apply(lambda x: x.split('_')[0])
    combined_results['Scenario_Type'] = combined_results['Scenario'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
    combined_results['Missing_Percentage'] = combined_results['Scenario'].apply(lambda x: int(x.split('_')[-1].replace('pct', '')))
    
    # Save combined results
    combined_results.to_csv(os.path.join(output_dir, 'combined_results.csv'), index=False)
    
    # Determine best method for each scenario
    combined_results['Best_Method_RMSE'] = combined_results[['KNN_RMSE', 'SVM_RMSE'] + (['Hybrid_RMSE'] if hybrid_exists else [])].idxmin(axis=1).apply(lambda x: x.split('_')[0])
    combined_results['Best_Method_MAE'] = combined_results[['KNN_MAE', 'SVM_MAE'] + (['Hybrid_MAE'] if hybrid_exists else [])].idxmin(axis=1).apply(lambda x: x.split('_')[0])
    combined_results['Best_Method_R2'] = combined_results[['KNN_R2', 'SVM_R2'] + (['Hybrid_R2'] if hybrid_exists else [])].idxmax(axis=1).apply(lambda x: x.split('_')[0])
    
    # Calculate improvement percentages
    if hybrid_exists:
        combined_results['Hybrid_vs_KNN_RMSE_Improvement'] = ((combined_results['KNN_RMSE'] - combined_results['Hybrid_RMSE']) / combined_results['KNN_RMSE']) * 100
        combined_results['Hybrid_vs_SVM_RMSE_Improvement'] = ((combined_results['SVM_RMSE'] - combined_results['Hybrid_RMSE']) / combined_results['SVM_RMSE']) * 100
        combined_results['Hybrid_vs_KNN_MAE_Improvement'] = ((combined_results['KNN_MAE'] - combined_results['Hybrid_MAE']) / combined_results['KNN_MAE']) * 100
        combined_results['Hybrid_vs_SVM_MAE_Improvement'] = ((combined_results['SVM_MAE'] - combined_results['Hybrid_MAE']) / combined_results['SVM_MAE']) * 100
        combined_results['Hybrid_vs_KNN_R2_Improvement'] = ((combined_results['Hybrid_R2'] - combined_results['KNN_R2']) / combined_results['KNN_R2']) * 100
        combined_results['Hybrid_vs_SVM_R2_Improvement'] = ((combined_results['Hybrid_R2'] - combined_results['SVM_R2']) / combined_results['SVM_R2']) * 100
    
    # Save enhanced results
    combined_results.to_csv(os.path.join(output_dir, 'enhanced_results.csv'), index=False)
    
    # Create summary by missing mechanism
    mechanism_summary = combined_results.groupby('Missing_Mechanism').agg({
        'KNN_RMSE': 'mean',
        'SVM_RMSE': 'mean',
        'KNN_MAE': 'mean',
        'SVM_MAE': 'mean',
        'KNN_R2': 'mean',
        'SVM_R2': 'mean',
        'KNN_Time': 'mean',
        'SVM_Time': 'mean'
    })
    
    if hybrid_exists:
        mechanism_summary['Hybrid_RMSE'] = combined_results.groupby('Missing_Mechanism')['Hybrid_RMSE'].mean()
        mechanism_summary['Hybrid_MAE'] = combined_results.groupby('Missing_Mechanism')['Hybrid_MAE'].mean()
        mechanism_summary['Hybrid_R2'] = combined_results.groupby('Missing_Mechanism')['Hybrid_R2'].mean()
        mechanism_summary['Hybrid_Time'] = combined_results.groupby('Missing_Mechanism')['Hybrid_Time'].mean()
    
    mechanism_summary.to_csv(os.path.join(output_dir, 'mechanism_summary.csv'))
    
    # Create summary by missing percentage
    percentage_summary = combined_results.groupby('Missing_Percentage').agg({
        'KNN_RMSE': 'mean',
        'SVM_RMSE': 'mean',
        'KNN_MAE': 'mean',
        'SVM_MAE': 'mean',
        'KNN_R2': 'mean',
        'SVM_R2': 'mean',
        'KNN_Time': 'mean',
        'SVM_Time': 'mean'
    })
    
    if hybrid_exists:
        percentage_summary['Hybrid_RMSE'] = combined_results.groupby('Missing_Percentage')['Hybrid_RMSE'].mean()
        percentage_summary['Hybrid_MAE'] = combined_results.groupby('Missing_Percentage')['Hybrid_MAE'].mean()
        percentage_summary['Hybrid_R2'] = combined_results.groupby('Missing_Percentage')['Hybrid_R2'].mean()
        percentage_summary['Hybrid_Time'] = combined_results.groupby('Missing_Percentage')['Hybrid_Time'].mean()
    
    percentage_summary.to_csv(os.path.join(output_dir, 'percentage_summary.csv'))
    
    # Create summary by scenario type
    scenario_summary = combined_results.groupby('Scenario_Type').agg({
        'KNN_RMSE': 'mean',
        'SVM_RMSE': 'mean',
        'KNN_MAE': 'mean',
        'SVM_MAE': 'mean',
        'KNN_R2': 'mean',
        'SVM_R2': 'mean',
        'KNN_Time': 'mean',
        'SVM_Time': 'mean'
    })
    
    if hybrid_exists:
        scenario_summary['Hybrid_RMSE'] = combined_results.groupby('Scenario_Type')['Hybrid_RMSE'].mean()
        scenario_summary['Hybrid_MAE'] = combined_results.groupby('Scenario_Type')['Hybrid_MAE'].mean()
        scenario_summary['Hybrid_R2'] = combined_results.groupby('Scenario_Type')['Hybrid_R2'].mean()
        scenario_summary['Hybrid_Time'] = combined_results.groupby('Scenario_Type')['Hybrid_Time'].mean()
    
    scenario_summary.to_csv(os.path.join(output_dir, 'scenario_summary.csv'))
    
    # Count best method occurrences
    best_method_counts = {
        'RMSE': combined_results['Best_Method_RMSE'].value_counts().to_dict(),
        'MAE': combined_results['Best_Method_MAE'].value_counts().to_dict(),
        'R2': combined_results['Best_Method_R2'].value_counts().to_dict()
    }
    
    # Save best method counts
    pd.DataFrame(best_method_counts).to_csv(os.path.join(output_dir, 'best_method_counts.csv'))
    
    print("Performance analysis completed. Results saved to:", output_dir)
    return combined_results, mechanism_summary, percentage_summary, scenario_summary, best_method_counts

def create_visualizations(combined_results, mechanism_summary, percentage_summary, scenario_summary, best_method_counts):
    """
    Create visualizations for the performance analysis results.
    """
    output_dir = '/home/ubuntu/research_analysis/analysis_results/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if hybrid results exist
    hybrid_exists = 'Hybrid_RMSE' in combined_results.columns
    
    # Set color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71'] if hybrid_exists else ['#3498db', '#e74c3c']
    methods = ['KNN', 'SVM', 'Hybrid'] if hybrid_exists else ['KNN', 'SVM']
    
    # 1. Overall performance comparison
    plt.figure(figsize=(12, 8))
    
    # RMSE comparison
    plt.subplot(2, 2, 1)
    rmse_data = [combined_results[f'{method}_RMSE'].mean() for method in methods]
    bars = plt.bar(methods, rmse_data, color=colors)
    plt.title('Average RMSE Across All Scenarios')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE comparison
    plt.subplot(2, 2, 2)
    mae_data = [combined_results[f'{method}_MAE'].mean() for method in methods]
    bars = plt.bar(methods, mae_data, color=colors)
    plt.title('Average MAE Across All Scenarios')
    plt.ylabel('MAE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # R2 comparison
    plt.subplot(2, 2, 3)
    r2_data = [combined_results[f'{method}_R2'].mean() for method in methods]
    bars = plt.bar(methods, r2_data, color=colors)
    plt.title('Average R² Across All Scenarios')
    plt.ylabel('R² (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Execution time comparison
    plt.subplot(2, 2, 4)
    time_data = [combined_results[f'{method}_Time'].mean() for method in methods]
    bars = plt.bar(methods, time_data, color=colors)
    plt.title('Average Execution Time Across All Scenarios')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_performance_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Performance by missing mechanism
    mechanisms = mechanism_summary.index.tolist()
    
    # RMSE by mechanism
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    x = np.arange(len(mechanisms))
    width = 0.25 if hybrid_exists else 0.35
    
    bars1 = plt.bar(x - width, mechanism_summary['KNN_RMSE'], width, label='KNN', color=colors[0])
    bars2 = plt.bar(x, mechanism_summary['SVM_RMSE'], width, label='SVM', color=colors[1])
    
    if hybrid_exists:
        bars3 = plt.bar(x + width, mechanism_summary['Hybrid_RMSE'], width, label='Hybrid', color=colors[2])
    
    plt.xlabel('Missing Data Mechanism')
    plt.ylabel('RMSE (lower is better)')
    plt.title('RMSE by Missing Data Mechanism')
    plt.xticks(x, mechanisms)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE by mechanism
    plt.subplot(2, 2, 2)
    
    bars1 = plt.bar(x - width, mechanism_summary['KNN_MAE'], width, label='KNN', color=colors[0])
    bars2 = plt.bar(x, mechanism_summary['SVM_MAE'], width, label='SVM', color=colors[1])
    
    if hybrid_exists:
        bars3 = plt.bar(x + width, mechanism_summary['Hybrid_MAE'], width, label='Hybrid', color=colors[2])
    
    plt.xlabel('Missing Data Mechanism')
    plt.ylabel('MAE (lower is better)')
    plt.title('MAE by Missing Data Mechanism')
    plt.xticks(x, mechanisms)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # R2 by mechanism
    plt.subplot(2, 2, 3)
    
    bars1 = plt.bar(x - width, mechanism_summary['KNN_R2'], width, label='KNN', color=colors[0])
    bars2 = plt.bar(x, mechanism_summary['SVM_R2'], width, label='SVM', color=colors[1])
    
    if hybrid_exists:
        bars3 = plt.bar(x + width, mechanism_summary['Hybrid_R2'], width, label='Hybrid', color=colors[2])
    
    plt.xlabel('Missing Data Mechanism')
    plt.ylabel('R² (higher is better)')
    plt.title('R² by Missing Data Mechanism')
    plt.xticks(x, mechanisms)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Time by mechanism
    plt.subplot(2, 2, 4)
    
    bars1 = plt.bar(x - width, mechanism_summary['KNN_Time'], width, label='KNN', color=colors[0])
    bars2 = plt.bar(x, mechanism_summary['SVM_Time'], width, label='SVM', color=colors[1])
    
    if hybrid_exists:
        bars3 = plt.bar(x + width, mechanism_summary['Hybrid_Time'], width, label='Hybrid', color=colors[2])
    
    plt.xlabel('Missing Data Mechanism')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time by Missing Data Mechanism')
    plt.xticks(x, mechanisms)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_mechanism.png'), dpi=300)
    plt.close()
    
    # 3. Performance by missing percentage
    percentages = percentage_summary.index.tolist()
    
    plt.figure(figsize=(14, 10))
    
    # RMSE by percentage
    plt.subplot(2, 2, 1)
    
    for i, method in enumerate(methods):
        plt.plot(percentages, percentage_summary[f'{method}_RMSE'], marker='o', linewidth=2, label=method, color=colors[i])
    
    plt.xlabel('Missing Data Percentage')
    plt.ylabel('RMSE (lower is better)')
    plt.title('RMSE by Missing Data Percentage')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    # MAE by percentage
    plt.subplot(2, 2, 2)
    
    for i, method in enumerate(methods):
        plt.plot(percentages, percentage_summary[f'{method}_MAE'], marker='o', linewidth=2, label=method, color=colors[i])
    
    plt.xlabel('Missing Data Percentage')
    plt.ylabel('MAE (lower is better)')
    plt.title('MAE by Missing Data Percentage')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    # R2 by percentage
    plt.subplot(2, 2, 3)
    
    for i, method in enumerate(methods):
        plt.plot(percentages, percentage_summary[f'{method}_R2'], marker='o', linewidth=2, label=method, color=colors[i])
    
    plt.xlabel('Missing Data Percentage')
    plt.ylabel('R² (higher is better)')
    plt.title('R² by Missing Data Percentage')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    # Time by percentage
    plt.subplot(2, 2, 4)
    
    for i, method in enumerate(methods):
        plt.plot(percentages, percentage_summary[f'{method}_Time'], marker='o', linewidth=2, label=method, color=colors[i])
    
    plt.xlabel('Missing Data Percentage')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time by Missing Data Percentage')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_percentage.png'), dpi=300)
    plt.close()
    
    # 4. Best method distribution
    plt.figure(figsize=(15, 5))
    
    metrics = ['RMSE', 'MAE', 'R2']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        # Get counts, ensuring all methods are represented
        counts = {}
        for method in methods:
            counts[method] = best_method_counts[metric].get(method, 0)
        
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', colors=colors[:len(counts)])
        plt.title(f'Best Method by {metric}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_method_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Heatmap of performance by scenario and method
    if hybrid_exists:
        # Prepare data for heatmap
        scenario_names = combined_results['Scenario'].tolist()
        
        # RMSE heatmap
        rmse_data = np.array([
            combined_results['KNN_RMSE'].tolist(),
            combined_results['SVM_RMSE'].tolist(),
            combined_results['Hybrid_RMSE'].tolist()
        ])
        
        plt.figure(figsize=(20, 10))
        
        # Create a custom colormap (blue to white to red)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#3498db', '#ffffff', '#e74c3c'])
        
        # Normalize data for better visualization
        vmax = np.max(rmse_data)
        vmin = np.min(rmse_data)
        
        sns.heatmap(rmse_data, annot=True, fmt='.4f', cmap=cmap, xticklabels=scenario_names, 
                    yticklabels=methods, cbar_kws={'label': 'RMSE (lower is better)'})
        
        plt.title('RMSE Comparison Across All Scenarios')
        plt.ylabel('Method')
        plt.xlabel('Scenario')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rmse_heatmap.png'), dpi=300)
        plt.close()
        
        # R2 heatmap
        r2_data = np.array([
            combined_results['KNN_R2'].tolist(),
            combined_results['SVM_R2'].tolist(),
            combined_results['Hybrid_R2'].tolist()
        ])
        
        plt.figure(figsize=(20, 10))
        
        # For R2, higher is better, so we invert the colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#e74c3c', '#ffffff', '#3498db'])
        
        # Normalize data for better visualization
        vmax = np.max(r2_data)
        vmin = np.min(r2_data)
        
        sns.heatmap(r2_data, annot=True, fmt='.4f', cmap=cmap, xticklabels=scenario_names, 
                    yticklabels=methods, cbar_kws={'label': 'R² (higher is better)'})
        
        plt.title('R² Comparison Across All Scenarios')
        plt.ylabel('Method')
        plt.xlabel('Scenario')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r2_heatmap.png'), dpi=300)
        plt.close()
        
        # 6. Improvement percentage of hybrid method
        plt.figure(figsize=(15, 10))
        
        # RMSE improvement
        plt.subplot(2, 2, 1)
        x = np.arange(len(scenario_names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, combined_results['Hybrid_vs_KNN_RMSE_Improvement'], width, label='vs KNN', color='#3498db')
        bars2 = plt.bar(x + width/2, combined_results['Hybrid_vs_SVM_RMSE_Improvement'], width, label='vs SVM', color='#e74c3c')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Scenario')
        plt.ylabel('Improvement %')
        plt.title('Hybrid Method RMSE Improvement')
        plt.xticks(x, scenario_names, rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # MAE improvement
        plt.subplot(2, 2, 2)
        
        bars1 = plt.bar(x - width/2, combined_results['Hybrid_vs_KNN_MAE_Improvement'], width, label='vs KNN', color='#3498db')
        bars2 = plt.bar(x + width/2, combined_results['Hybrid_vs_SVM_MAE_Improvement'], width, label='vs SVM', color='#e74c3c')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Scenario')
        plt.ylabel('Improvement %')
        plt.title('Hybrid Method MAE Improvement')
        plt.xticks(x, scenario_names, rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # R2 improvement
        plt.subplot(2, 2, 3)
        
        bars1 = plt.bar(x - width/2, combined_results['Hybrid_vs_KNN_R2_Improvement'], width, label='vs KNN', color='#3498db')
        bars2 = plt.bar(x + width/2, combined_results['Hybrid_vs_SVM_R2_Improvement'], width, label='vs SVM', color='#e74c3c')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Scenario')
        plt.ylabel('Improvement %')
        plt.title('Hybrid Method R² Improvement')
        plt.xticks(x, scenario_names, rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Average improvement by mechanism
        plt.subplot(2, 2, 4)
        
        mechanisms = combined_results['Missing_Mechanism'].unique()
        x = np.arange(len(mechanisms))
        
        rmse_imp_knn = [combined_results[combined_results['Missing_Mechanism'] == m]['Hybrid_vs_KNN_RMSE_Improvement'].mean() for m in mechanisms]
        rmse_imp_svm = [combined_results[combined_results['Missing_Mechanism'] == m]['Hybrid_vs_SVM_RMSE_Improvement'].mean() for m in mechanisms]
        
        bars1 = plt.bar(x - width/2, rmse_imp_knn, width, label='vs KNN', color='#3498db')
        bars2 = plt.bar(x + width/2, rmse_imp_svm, width, label='vs SVM', color='#e74c3c')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Missing Mechanism')
        plt.ylabel('Average RMSE Improvement %')
        plt.title('Hybrid Method Improvement by Mechanism')
        plt.xticks(x, mechanisms)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_improvement.png'), dpi=300)
        plt.close()
    
    print("Visualizations created and saved to:", output_dir)

if __name__ == "__main__":
    # Run performance analysis
    results = analyze_performance()
    
    # Check if analysis was successful
    if results:
        combined_results, mechanism_summary, percentage_summary, scenario_summary, best_method_counts = results
        
        # Create visualizations
        create_visualizations(combined_results, mechanism_summary, percentage_summary, scenario_summary, best_method_counts)
