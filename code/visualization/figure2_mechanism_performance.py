"""
Figure 2: Performance by Missing Data Mechanism
This script generates a grouped bar chart comparing RMSE values for each imputation method
across different missing data mechanisms (MCAR, MAR, MNAR).
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
mechanisms = ['MCAR', 'MAR', 'MNAR', 'Average']
knn_rmse = [6.761, 8.548, 8.554, 7.954]
svm_rmse = [8.247, 7.882, 9.008, 8.379]
hybrid_rmse = [6.624, 8.288, 8.752, 7.888]

# Set width of bars
barWidth = 0.25
 
# Set positions of the bars on X axis
r1 = np.arange(len(mechanisms))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Create figure
plt.figure(figsize=(12, 8))
 
# Create bars
bars1 = plt.bar(r1, knn_rmse, width=barWidth, edgecolor='white', label='KNN', color='#0173B2')
bars2 = plt.bar(r2, svm_rmse, width=barWidth, edgecolor='white', label='SVM', color='#029E73')
bars3 = plt.bar(r3, hybrid_rmse, width=barWidth, edgecolor='white', label='Hybrid', color='#D55E00')
 
# Add values on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Add labels and title
plt.xlabel('Missing Data Mechanism', fontweight='bold')
plt.ylabel('Root Mean Square Error (RMSE)', fontweight='bold')
plt.title('Performance by Missing Data Mechanism', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(mechanisms))], mechanisms)
 
# Create legend & grid
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a note about interpretation
plt.figtext(0.5, 0.01, 'Note: Lower RMSE values indicate better performance. MCAR = Missing Completely At Random, MAR = Missing At Random, MNAR = Missing Not At Random.',
         ha='center', fontsize=10, style='italic')

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figures/figure2_mechanism_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 has been generated and saved to figures/figure2_mechanism_performance.png")
