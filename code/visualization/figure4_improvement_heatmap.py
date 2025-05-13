"""
Figure 4: Heatmap of Hybrid Method Improvement
This script generates a heatmap showing the percentage improvement of the hybrid method
compared to SVM across different scenario types and missing data percentages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Create data for the heatmap
# Improvement percentages of Hybrid vs SVM (positive values mean Hybrid is better)
improvement_data = {
    'Scenario Type': ['RandomSensorFailures', 'ConnectionIssues', 'ActivityDependent', 
                     'BatteryDepletion', 'ValueDependent', 'RangeLimitation'],
    '10%': [2.82, 5.67, 7.82, 10.09, 11.89, 7.83],
    '20%': [3.48, 5.95, 8.58, 10.31, 12.57, 8.13],
    '30%': [3.58, 6.17, 9.07, 10.80, 12.63, 8.51]
}

# Convert to DataFrame
df = pd.DataFrame(improvement_data)
df = df.set_index('Scenario Type')

# Create figure
plt.figure(figsize=(12, 8))

# Create heatmap
ax = sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5, 
                cbar_kws={'label': 'Improvement %'})

# Add labels and title
plt.title('Percentage Improvement of Hybrid Method vs SVM by Scenario', fontweight='bold')
plt.xlabel('Missing Data Percentage', fontweight='bold')
plt.ylabel('Scenario Type', fontweight='bold')

# Add a note about interpretation
plt.figtext(0.5, 0.01, 'Note: Values represent percentage improvement in RMSE. Higher values indicate greater improvement of the hybrid method over SVM.',
         ha='center', fontsize=10, style='italic')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figures/figure4_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 has been generated and saved to figures/figure4_improvement_heatmap.png")
