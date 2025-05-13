"""
Figure 3: Impact of Missing Data Percentage
This script generates a line chart showing how RMSE values change with increasing 
percentages of missing data (10%, 20%, 30%) for each imputation method.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Create data for the plot
percentages = [10, 20, 30]
knn_rmse = [6.955, 8.478, 8.431]
svm_rmse = [8.050, 8.380, 8.706]
hybrid_rmse = [7.546, 8.028, 8.091]

# Create figure
plt.figure(figsize=(10, 7))

# Create lines
plt.plot(percentages, knn_rmse, marker='o', linewidth=2, markersize=8, label='KNN', color='#0173B2')
plt.plot(percentages, svm_rmse, marker='s', linewidth=2, markersize=8, label='SVM', color='#029E73')
plt.plot(percentages, hybrid_rmse, marker='^', linewidth=2, markersize=8, label='Hybrid', color='#D55E00')

# Add data labels
for i, (knn, svm, hyb) in enumerate(zip(knn_rmse, svm_rmse, hybrid_rmse)):
    plt.text(percentages[i], knn+0.15, f'{knn:.3f}', ha='center', va='bottom', color='#0173B2')
    plt.text(percentages[i], svm+0.15, f'{svm:.3f}', ha='center', va='bottom', color='#029E73')
    plt.text(percentages[i], hyb+0.15, f'{hyb:.3f}', ha='center', va='bottom', color='#D55E00')

# Add labels and title
plt.xlabel('Percentage of Missing Data (%)', fontweight='bold')
plt.ylabel('Root Mean Square Error (RMSE)', fontweight='bold')
plt.title('Impact of Missing Data Percentage on Imputation Performance', fontweight='bold')
plt.xticks(percentages)

# Set y-axis limits with some padding
plt.ylim(6, 10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Create legend
plt.legend(loc='upper left')

# Add a note about interpretation
plt.figtext(0.5, 0.01, 'Note: Lower RMSE values indicate better performance. Values are averaged across all missing data mechanisms and scenarios.',
         ha='center', fontsize=10, style='italic')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figures/figure3_percentage_impact.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 has been generated and saved to figures/figure3_percentage_impact.png")
