"""
generate_hybrid_results_summary.py

Acest script generează fișierul hibrid_results_summary.csv pe baza datelor imputate, 
datelor cu valori lipsă și datelor originale.

Utilizare:
    python generate_hybrid_results_summary.py --imputed_dir /path/to/imputed/files 
                                             --missing_dir /path/to/missing/files 
                                             --original /path/to/original.csv 
                                             --output /path/to/output/hibrid_results_summary.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import glob
import re

def parse_arguments():
    """Parsează argumentele din linia de comandă."""
    parser = argparse.ArgumentParser(description='Generează hibrid_results_summary.csv')
    parser.add_argument('--imputed_dir', required=True, help='Directorul cu fișierele imputate')
    parser.add_argument('--missing_dir', required=True, help='Directorul cu fișierele cu date lipsă')
    parser.add_argument('--original', required=True, help='Calea către fișierul cu date originale')
    parser.add_argument('--output', required=True, help='Calea de ieșire pentru hibrid_results_summary.csv')
    parser.add_argument('--exclude_columns', default='time,Activity', help='Coloanele de exclus din evaluare, separate prin virgulă')
    return parser.parse_args()

def extract_scenario_info(filename):
    """Extrage informații despre scenariu din numele fișierului."""
    # Exemplu: MCAR_RandomSensorFailures_10pct_Hybrid.csv
    pattern = r'(MCAR|MAR|MNAR)_([A-Za-z]+)_(\d+)pct'
    match = re.search(pattern, filename)
    
    if match:
        mechanism = match.group(1)  # MCAR, MAR, MNAR
        scenario_type = match.group(2)  # RandomSensorFailures, etc.
        percentage = int(match.group(3))  # 10, 20, 30
        return mechanism, scenario_type, percentage
    
    return None, None, None

def calculate_metrics(imputed_data, original_data, missing_mask, exclude_columns):
    """Calculează metricile de performanță pentru datele imputate."""
    # Filtrăm coloanele excluse
    data_columns = [col for col in imputed_data.columns if col not in exclude_columns]
    
    # Inițializăm variabilele pentru metrici
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    valid_columns = 0
    
    # Calculăm metricile pentru fiecare coloană
    column_metrics = {}
    for col in data_columns:
        # Identificăm valorile imputate (cele care erau lipsă în setul original)
        imputed_mask = missing_mask[col].astype(bool)
        
        # Verificăm dacă există valori imputate pentru această coloană
        if imputed_mask.sum() > 0:
            # Extragem valorile originale și cele imputate
            y_true = original_data.loc[imputed_mask, col].values
            y_pred = imputed_data.loc[imputed_mask, col].values
            
            # Calculăm metricile
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # R2 poate da erori pentru unele coloane, așa că folosim try-except
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = np.nan
            
            # Adăugăm la totaluri
            total_rmse += rmse
            total_mae += mae
            if not np.isnan(r2):
                total_r2 += r2
            
            valid_columns += 1
            
            # Salvăm metricile pentru această coloană
            column_metrics[col] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
    
    # Calculăm mediile
    avg_rmse = total_rmse / valid_columns if valid_columns > 0 else np.nan
    avg_mae = total_mae / valid_columns if valid_columns > 0 else np.nan
    avg_r2 = total_r2 / valid_columns if valid_columns > 0 else np.nan
    
    return {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'r2': avg_r2,
        'column_metrics': column_metrics
    }

def process_files(args):
    """Procesează fișierele și generează rezultatele."""
    # Convertim exclude_columns într-o listă
    exclude_columns = [col.strip() for col in args.exclude_columns.split(',')]
    
    # Încărcăm datele originale
    original_data = pd.read_csv(args.original)
    
    # Inițializăm lista pentru rezultate
    results = []
    
    # Găsim toate fișierele imputate
    imputed_files = glob.glob(os.path.join(args.imputed_dir, '*_Hybrid.csv'))
    
    for imputed_file in imputed_files:
        # Extragem numele de bază al fișierului
        base_name = os.path.basename(imputed_file)
        
        # Extragem informațiile despre scenariu
        mechanism, scenario_type, percentage = extract_scenario_info(base_name)
        
        if mechanism is None:
            print(f"Nu s-au putut extrage informații din numele fișierului: {base_name}")
            continue
        
        # Construim numele fișierului cu date lipsă
        missing_file_pattern = f"{mechanism}_{scenario_type}_{percentage}pct.csv"
        missing_files = glob.glob(os.path.join(args.missing_dir, missing_file_pattern))
        
        if not missing_files:
            print(f"Nu s-a găsit fișierul cu date lipsă pentru: {missing_file_pattern}")
            continue
        
        missing_file = missing_files[0]
        
        print(f"Procesez: {base_name}")
        
        # Măsurăm timpul de execuție
        start_time = time.time()
        
        # Încărcăm datele
        imputed_data = pd.read_csv(imputed_file)
        missing_data = pd.read_csv(missing_file)
        
        # Creăm masca pentru valorile lipsă
        missing_mask = missing_data.isna()
        
        # Calculăm metricile
        metrics = calculate_metrics(imputed_data, original_data, missing_mask, exclude_columns)
        
        # Calculăm timpul de execuție
        execution_time = time.time() - start_time
        
        # Adăugăm rezultatele
        results.append({
            'Mechanism': mechanism,
            'ScenarioType': scenario_type,
            'Percentage': percentage,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R2': metrics['r2'],
            'ExecutionTime': execution_time
        })
    
    # Creăm DataFrame-ul cu rezultate
    results_df = pd.DataFrame(results)
    
    # Sortăm rezultatele
    results_df = results_df.sort_values(by=['Mechanism', 'ScenarioType', 'Percentage'])
    
    # Calculăm statistici pe mecanisme
    mechanism_stats = results_df.groupby('Mechanism').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean',
        'ExecutionTime': 'mean'
    }).reset_index()
    mechanism_stats['ScenarioType'] = 'Average'
    mechanism_stats['Percentage'] = 0
    
    # Calculăm statistici pe procente
    percentage_stats = results_df.groupby('Percentage').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean',
        'ExecutionTime': 'mean'
    }).reset_index()
    percentage_stats['Mechanism'] = 'Average'
    percentage_stats['ScenarioType'] = 'Average'
    
    # Calculăm statistici generale
    overall_stats = pd.DataFrame([{
        'Mechanism': 'Overall',
        'ScenarioType': 'Average',
        'Percentage': 0,
        'RMSE': results_df['RMSE'].mean(),
        'MAE': results_df['MAE'].mean(),
        'R2': results_df['R2'].mean(),
        'ExecutionTime': results_df['ExecutionTime'].mean()
    }])
    
    # Combinăm toate rezultatele
    final_results = pd.concat([results_df, mechanism_stats, percentage_stats, overall_stats])
    
    # Salvăm rezultatele
    final_results.to_csv(args.output, index=False)
    print(f"Rezultatele au fost salvate în: {args.output}")
    
    return final_results

def main():
    """Funcția principală."""
    args = parse_arguments()
    process_files(args)

if __name__ == "__main__":
    main()
