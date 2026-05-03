import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Input Data (Full Ablation Sweep)
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
# 2. Process Metrics
# ==========================================
labels = list(ablation_data.keys())
curves = list(ablation_data.values())
tasks = np.arange(1, 11)

last_task_accuracies = [curve[-1] for curve in curves]
avg_incremental_accuracies = [np.mean(curve) for curve in curves]

# Plotting Settings
x_indices = np.arange(len(labels))
width = 0.6
colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#34495e', '#f1c40f']

# Helper for bar labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')

# # --- PLOT 1: Task-to-Task Accuracy Curve (\mathcal{A}_t) ---
# plt.figure(figsize=(10, 6))
# for (label, curve), color in zip(ablation_data.items(), colors):
#     plt.plot(tasks, curve, marker='o', markersize=4, linewidth=2, color=color, label=rf"$\alpha = {label}$")

# plt.title('Performance Evolution: Task-to-Task Average Accuracy ($\mathcal{A}_t$)', fontsize=13, fontweight='bold')
# plt.xlabel('Incremental Task Number ($t$)', fontsize=11)
# plt.ylabel('Accuracy (%)', fontsize=11)
# plt.xticks(tasks)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(title="Regularization Weight", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig("task_to_task_evolution.pdf", format='pdf')
# plt.show()

# --- PLOT 2: Last Task Accuracy ---
fig, ax1 = plt.subplots(figsize=(8, 5))
rects1 = ax1.bar(x_indices, last_task_accuracies, width, color=colors, edgecolor='black', alpha=0.8)
ax1.set_title(r'Final Task Accuracy', fontsize=15, fontweight='bold')
ax1.set_xticks(x_indices)
ax1.set_xticklabels(labels)
ax1.set_ylim(min(last_task_accuracies) - 1, max(last_task_accuracies) + 1.0)
ax1.grid(axis='y', linestyle=':', alpha=0.6)
autolabel(rects1, ax1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("last_task_accuracy_ablation.pdf", format='pdf')
plt.show()

# --- PLOT 3: Average Incremental Accuracy ---
fig, ax2 = plt.subplots(figsize=(8, 5))
rects2 = ax2.bar(x_indices, avg_incremental_accuracies, width, color=colors, edgecolor='black', alpha=0.8)
ax2.set_title('Average Incremental Accuracy', fontsize=15, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(labels)
ax2.set_ylim(min(avg_incremental_accuracies) - 0.5, max(avg_incremental_accuracies) + 0.5)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
autolabel(rects2, ax2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("average_incremental_accuracy_ablation.pdf", format='pdf')
plt.show()