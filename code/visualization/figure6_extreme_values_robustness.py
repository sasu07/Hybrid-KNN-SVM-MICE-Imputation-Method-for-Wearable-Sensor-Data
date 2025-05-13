"""
Figure 6: Robustness to Extreme Values
This script generates a line chart showing how RMSE values change with increasing 
distance from the mean (in standard deviations) for each imputation method.
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
std_distances = ['<1σ', '1σ-2σ', '2σ-3σ', '>3σ']
knn_rmse = [0.1427, 0.1783, 0.2214, 0.2573]
svm_rmse = [0.1312, 0.1573, 0.1842, 0.2104]
hybrid_rmse = [0.1247, 0.1462, 0.1687, 0.1876]

# Calculate improvement percentages
hybrid_vs_knn = [(knn - hyb) / knn * 100 for knn, hyb in zip(knn_rmse, hybrid_rmse)]
hybrid_vs_svm = [(svm - hyb) / svm * 100 for svm, hyb in zip(svm_rmse, hybrid_rmse)]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot 1: RMSE values
ax1.plot(std_distances, knn_rmse, marker='o', linewidth=2, markersize=8, label='KNN', color='#0173B2')
ax1.plot(std_distances, svm_rmse, marker='s', linewidth=2, markersize=8, label='SVM', color='#029E73')
ax1.plot(std_distances, hybrid_rmse, marker='^', linewidth=2, markersize=8, label='Hybrid', color='#D55E00')

# Add data labels
for i, (knn, svm, hyb) in enumerate(zip(knn_rmse, svm_rmse, hybrid_rmse)):
    ax1.text(i, knn+0.01, f'{knn:.4f}', ha='center', va='bottom', color='#0173B2')
    ax1.text(i, svm+0.01, f'{svm:.4f}', ha='center', va='bottom', color='#029E73')
    ax1.text(i, hyb+0.01, f'{hyb:.4f}', ha='center', va='bottom', color='#D55E00')

# Add labels and title for plot 1
ax1.set_xlabel('Distance from Mean', fontweight='bold')
ax1.set_ylabel('Root Mean Square Error (RMSE)', fontweight='bold')
ax1.set_title('RMSE by Distance from Mean', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left')

# Plot 2: Improvement percentages
ax2.plot(std_distances, hybrid_vs_knn, marker='o', linewidth=2, markersize=8, label='Hybrid vs KNN', color='#0173B2')
ax2.plot(std_distances, hybrid_vs_svm, marker='s', linewidth=2, markersize=8, label='Hybrid vs SVM', color='#029E73')

# Add data labels
for i, (vs_knn, vs_svm) in enumerate(zip(hybrid_vs_knn, hybrid_vs_svm)):
    ax2.text(i, vs_knn+0.5, f'{vs_knn:.2f}%', ha='center', va='bottom', color='#0173B2')
    ax2.text(i, vs_svm+0.5, f'{vs_svm:.2f}%', ha='center', va='bottom', color='#029E73')

# Add labels and title for plot 2
ax2.set_xlabel('Distance from Mean', fontweight='bold')
ax2.set_ylabel('Improvement Percentage (%)', fontweight='bold')
ax2.set_title('Hybrid Method Improvement by Distance from Mean', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper left')

# Add a note about interpretation
fig.text(0.5, 0.01, 'Note: Lower RMSE values indicate better performance. Higher improvement percentages indicate greater advantage of the hybrid method.',
         ha='center', fontsize=10, style='italic')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figures/figure6_extreme_values_robustness.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 6 has been generated and saved to figures/figure6_extreme_values_robustness.png")
