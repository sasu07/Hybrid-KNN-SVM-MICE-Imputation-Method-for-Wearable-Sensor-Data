"""
Figure 1: General Performance Comparison of Imputation Methods
This script generates a bar chart comparing the performance metrics (RMSE, MAE, R²)
for all three imputation methods (KNN, SVM, Hybrid).
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
methods = ['KNN', 'SVM', 'Hybrid']
rmse_values = [7.288, 8.378, 7.554]
mae_values = [3.298, 3.761, 3.336]
r2_values = [0.938, 0.825, 0.941]

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Performance Comparison of Imputation Methods', fontweight='bold')

# Colors for the bars
colors = ['#0173B2', '#029E73', '#D55E00']

# Plot RMSE
bars1 = ax1.bar(methods, rmse_values, color=colors, width=0.6)
ax1.set_title('RMSE')
ax1.set_ylabel('Root Mean Square Error')
ax1.set_ylim(0, max(rmse_values) * 1.2)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.3f}', ha='center', va='bottom')

# Plot MAE
bars2 = ax2.bar(methods, mae_values, color=colors, width=0.6)
ax2.set_title('MAE')
ax2.set_ylabel('Mean Absolute Error')
ax2.set_ylim(0, max(mae_values) * 1.2)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.3f}', ha='center', va='bottom')

# Plot R²
bars3 = ax3.bar(methods, r2_values, color=colors, width=0.6)
ax3.set_title('R²')
ax3.set_ylabel('Coefficient of Determination')
ax3.set_ylim(0, 1.1)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Add value labels on top of bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom')

# Add a note about interpretation
fig.text(0.5, 0.01, 'Note: Lower RMSE and MAE values indicate better performance, while higher R² values indicate better performance.',
         ha='center', fontsize=10, style='italic')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figures/figure1_general_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 has been generated and saved to figures/figure1_general_performance.png")
