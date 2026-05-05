import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. Academic Styling (Enhanced Visibility)
# ==========================================
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelweight": "bold", 
    "axes.labelsize": 14,       
    "xtick.labelsize": 14,      # Increased tick value size
    "ytick.labelsize": 14,      # Increased tick value size
    "legend.fontsize": 12
})

# ==========================================
# 2. Input Data (Full Ablation Sweep)
# ==========================================
ablation_data = {
    "0.0001": [95.62, 91.10, 85.82, 81.39, 78.44, 75.58, 72.90, 69.86, 68.97, 67.38],
    "0.0005": [95.62, 91.54, 85.76, 80.96, 78.72, 76.08, 73.08, 70.83, 69.22, 67.36],
    "0.001":  [95.62, 91.80, 85.47, 81.39, 78.58, 74.95, 72.53, 70.62, 69.76, 66.85],
    "0.002":  [95.62, 91.71, 85.88, 81.13, 78.16, 75.82, 72.93, 71.44, 70.66, 67.16],
    "0.005":  [95.62, 91.71, 86.35, 81.26, 78.86, 75.27, 72.68, 70.49, 69.85, 66.98],
    "0.01":   [95.62, 91.27, 85.94, 82.04, 78.20, 76.02, 72.78, 70.49, 69.51, 67.69],
    "0.05":   [95.62, 91.97, 85.82, 82.04, 78.93, 75.27, 72.36, 70.14, 69.08, 67.38]
}

# ==========================================
# 3. Process Metrics
# ==========================================
labels = list(ablation_data.keys())
curves = list(ablation_data.values())

last_task_accuracies = [curve[-1] for curve in curves]
avg_incremental_accuracies = [np.mean(curve) for curve in curves]

x_indices = np.arange(len(labels))
width = 0.65
colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#34495e', '#f1c40f']

def autolabel(rects, ax):
    """Attach a text label above each bar for better data readability."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')

# --- PLOT 1: Final Task Accuracy ---
fig1, ax1 = plt.subplots(figsize=(8, 3.2), dpi=300)

rects1 = ax1.bar(x_indices, last_task_accuracies, width, color=colors, edgecolor='black', alpha=0.85, zorder=3)

ax1.set_title(r'Final Task Accuracy', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel(r'Regularization Weight ($\lambda_{\text{F}}$)', fontsize=14, labelpad=10)
ax1.set_ylabel('Accuracy (%)', fontsize=14, labelpad=10)

ax1.set_xticks(x_indices)
ax1.set_xticklabels(labels)

y_min1 = min(last_task_accuracies) - 0.2
y_max1 = max(last_task_accuracies) + 0.3
ax1.set_ylim(y_min1, y_max1)

ax1.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
autolabel(rects1, ax1)

sns.despine(fig1)
plt.tight_layout()
plt.savefig("last_task_accuracy_ablation.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- PLOT 2: Average Incremental Accuracy ---
fig2, ax2 = plt.subplots(figsize=(8, 3.2), dpi=300)

rects2 = ax2.bar(x_indices, avg_incremental_accuracies, width, color=colors, edgecolor='black', alpha=0.85, zorder=3)

ax2.set_title('Average Incremental Accuracy', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel(r'Regularization Weight ($\lambda_{\text{F}}$)', fontsize=14, labelpad=10)
ax2.set_ylabel('Accuracy (%)', fontsize=14, labelpad=10)

ax2.set_xticks(x_indices)
ax2.set_xticklabels(labels)

y_min2 = min(avg_incremental_accuracies) - 0.15
y_max2 = max(avg_incremental_accuracies) + 0.3
ax2.set_ylim(y_min2, y_max2)

ax2.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
autolabel(rects2, ax2)

sns.despine(fig2)
plt.tight_layout()
plt.savefig("average_incremental_accuracy_ablation.pdf", format='pdf', bbox_inches='tight')
plt.show()