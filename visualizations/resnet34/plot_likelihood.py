import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ============================================================
# 1. Configuration & Data Loading
# ============================================================
# NOTE: Change the filename if your data file is named differently!
file_path = 'log_likelihood_values_4_classes.txt' 

try:
    df = pd.read_csv(file_path, sep=r'\s+')
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure it is in the same directory as the script.")
    exit()

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 16
})

layers = ['layer1', 'layer2', 'layer3', 'layer4']

layer_dims = {
    'layer1': 802816, 
    'layer2': 401408, 
    'layer3': 200704, 
    'layer4': 100352
}

# ============================================================
# 2. Plotting
# ============================================================
print("\n--- Generating Log-Likelihood Plots ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, l in enumerate(layers):
    ax = axes[i]
    epochs = df['Epoch']
    
    base_ll_val = df[f'{l}_Baseline'].mean()
    lcm_ll = df[f'{l}_LCM']
    
    ax.axhline(base_ll_val, linestyle='--', linewidth=2.5, color='#7f8c8d', label='Diagonal Covariance')
    
    ax.plot(epochs, lcm_ll, linestyle='-', linewidth=2.5, color='#e67e22', label='Full Covariance')
    
    total_dims = layer_dims.get(l, 0)
    ax.set_title(f'ResNet34 - {l.capitalize()} ($D = {total_dims:,}$)', fontsize=20, fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Total Log-Likelihood', fontsize=18)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    y_min_lcm = lcm_ll.iloc[5:].min()
    y_min = min(base_ll_val, y_min_lcm)
    y_max = max(base_ll_val, lcm_ll.max())
    
    y_range = y_max - y_min
    y_margin = y_range * 0.10
    
    if y_range == 0:
        y_margin = abs(y_min) * 0.1
        
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    if i == 0:
        ax.legend(fontsize=18, loc='lower right')

plt.tight_layout()

pdf_path = "resnet34_log_likelihood.pdf"
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Plots saved successfully as: {pdf_path}")

plt.show()