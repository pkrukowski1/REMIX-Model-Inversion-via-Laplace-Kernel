import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

layers = ['layer1', 'layer2', 'layer3', 'layer4']
layer_dims = {'layer1': 802816, 'layer2': 401408, 'layer3': 200704, 'layer4': 100352}

fig, axes = plt.subplots(2, 2, figsize=(11, 3.8), sharex=True)
axes = axes.flatten()

for i, l in enumerate(layers):
    ax = axes[i]
    epochs = df['Epoch']
    base_ll_val = df[f'{l}_Baseline'].mean()
    lcm_ll = df[f'{l}_LCM']
    
    ax.axhline(base_ll_val, linestyle='--', linewidth=1.6, color='#555555', zorder=3, label='Diag. Cov.')
    ax.plot(epochs, lcm_ll, linestyle='-', linewidth=2.2, color='#e67e22', zorder=4, label='Full Cov.')
    
    total_dims = layer_dims.get(l, 0)
    title_text = f'{l.capitalize()} ($D = {total_dims:,}$)'
    
    ax.text(0.5, 0.88, title_text, transform=ax.transAxes, ha='center', 
            fontweight='bold', fontsize=14,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2), zorder=5)
    
    if i >= 2: 
        ax.set_xlabel('Epoch', labelpad=2, fontsize=13)
    if i % 2 == 0: 
        ax.set_ylabel('Log-Likelihood', labelpad=2, fontsize=13)
    
    ax.grid(True, linestyle=':', alpha=0.4, zorder=1)
    
    converged_data = lcm_ll.iloc[40:] 
    y_min_focus = min(base_ll_val, converged_data.min())
    y_max_focus = max(base_ll_val, converged_data.max())
    y_range = y_max_focus - y_min_focus
    if y_range == 0: y_range = abs(y_max_focus) * 0.05

    ax.set_ylim(y_min_focus - y_range * 0.35, y_max_focus + y_range * 0.6)
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.tick_params(axis='both', which='major', pad=2, labelsize=11)
    
    if i == 0:
        leg = ax.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=0.9, 
                        handlelength=1.5, borderpad=0.4, fontsize=11)
        leg.set_zorder(5)

plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.16, wspace=0.18, hspace=0.04)

plt.savefig("resnet34_log_likelihood.pdf", dpi=300, bbox_inches='tight')
plt.show()