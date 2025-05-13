"""
Figure 5: Hybrid Method Architecture
This script creates a visualization of the hybrid method architecture with annotations
for each component of the KNN-SVM-MICE pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Set the style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors
colors = {
    'complete_data': '#C7E9B4',
    'incomplete_data': '#FDDBC7',
    'pre_imputed': '#B3CDE3',
    'box': '#EEEEEE',
    'arrow': '#555555',
    'text': '#000000'
}

# Function to create a box
def create_box(x, y, width, height, label, color, fontsize=12):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return rect

# Function to create an arrow
def create_arrow(start, end, color='black', width=0.03):
    arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', color=color, linewidth=1.5,
                                   connectionstyle='arc3,rad=0.1', mutation_scale=20)
    ax.add_patch(arrow)
    return arrow

# Create the complete data flow (top)
create_box(1, 6, 2, 1, 'Complete Data', colors['complete_data'])
create_box(4, 6, 2, 1, 'Standard Scaler', colors['box'])
create_box(7, 6, 2, 1, 'SVR Models', colors['box'])

# Create the incomplete data flow (middle)
create_box(1, 3.5, 2, 1, 'Incomplete Data', colors['incomplete_data'])
create_box(4, 3.5, 2, 1, 'Standard Scaler', colors['box'])
create_box(7, 3.5, 2, 1, 'KNN Imputer', colors['box'])
create_box(10, 3.5, 2, 1, 'SVR Predict', colors['box'])
create_box(13, 3.5, 2, 1, 'Update & Rescale', colors['box'])

# Create the pre-imputed data flow (bottom)
create_box(7, 1, 2, 1, 'Pre-imputed Data', colors['pre_imputed'])
create_box(10, 1, 2, 1, 'Iterative Imputer\n(MICE)', colors['box'])
create_box(13, 1, 2, 1, 'Final Imputed\nData', colors['box'])

# Create arrows for complete data flow
create_arrow((3, 6.5), (4, 6.5), colors['arrow'])
create_arrow((6, 6.5), (7, 6.5), colors['arrow'])
create_arrow((9, 6.5), (10, 3.5), colors['arrow'])  # Connect to SVR Predict

# Create arrows for incomplete data flow
create_arrow((3, 4), (4, 4), colors['arrow'])
create_arrow((6, 4), (7, 4), colors['arrow'])
create_arrow((9, 4), (10, 4), colors['arrow'])
create_arrow((12, 4), (13, 4), colors['arrow'])
create_arrow((14, 4), (14, 1.5), colors['arrow'])  # Connect to Final Imputed Data

# Create arrows for pre-imputed data flow
create_arrow((9, 4), (8, 1.5), colors['arrow'])  # Connect KNN Imputer to Pre-imputed Data
create_arrow((9, 1.5), (10, 1.5), colors['arrow'])
create_arrow((12, 1.5), (13, 1.5), colors['arrow'])

# Add phase labels
ax.text(0.5, 7.2, 'Phase 1: Training on Complete Data', fontsize=14, fontweight='bold')
ax.text(0.5, 5.2, 'Phase 2: Initial Imputation with KNN and Refinement with SVR', fontsize=14, fontweight='bold')
ax.text(0.5, 2.7, 'Phase 3: Final Refinement with MICE', fontsize=14, fontweight='bold')

# Add title
plt.title('Hybrid KNN-SVM-MICE Imputation Method Architecture', fontweight='bold', pad=20)

# Remove axes
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Add a note about the architecture
plt.figtext(0.5, 0.01, 'Note: The hybrid method combines the strengths of KNN (initial imputation), SVM (refinement), and MICE (consistency).',
         ha='center', fontsize=10, style='italic')

# Save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figures/figure5_hybrid_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5 has been generated and saved to figures/figure5_hybrid_architecture.png")
