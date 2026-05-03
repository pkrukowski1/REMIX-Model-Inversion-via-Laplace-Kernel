import numpy as np
import matplotlib.pyplot as plt
import warnings

# Provided Data Structure
remix_data = [
    [[92.7], [77.6, 76.1], [67.3, 66.5, 72.5], [60.6, 59.6, 65.3, 63.4], [58.9, 54.2, 58.3, 55.4, 64.6], [56.4, 50.7, 53.3, 53.8, 58.5, 48.5], [49.5, 50.1, 48.3, 48.9, 53.5, 46.1, 51.0], [47.8, 46.6, 46.5, 47.8, 51.6, 43.6, 49.3, 62.6], [47.6, 42.6, 45.8, 46.4, 50.2, 38.4, 45.7, 60.7, 48.8], [45.8, 39.4, 43.5, 44.9, 48.7, 37.3, 43.0, 59.2, 47.7, 49.1]],
    [[91.8], [77.9, 76.7], [69.1, 65.7, 70.9], [63.3, 59.3, 63.4, 64.3], [60.7, 53.5, 57.4, 54.3, 66.6], [58.6, 47.4, 51.7, 51.8, 58.5, 51.5], [55.3, 47.9, 49.5, 47.1, 53.2, 48.3, 50.4], [51.0, 42.7, 46.7, 44.2, 51.7, 46.0, 46.9, 66.0], [50.7, 39.0, 43.7, 42.5, 49.4, 43.0, 43.7, 62.4, 51.1], [48.2, 36.0, 41.3, 40.3, 47.3, 41.3, 39.6, 61.0, 50.0, 48.8]],
    [[91.5], [78.0, 73.9], [69.7, 66.6, 69.5], [63.9, 59.3, 63.2, 61.1], [60.6, 56.1, 56.6, 57.4, 66.3], [58.3, 51.7, 51.1, 52.6, 60.1, 49.1], [55.1, 51.1, 46.1, 48.4, 54.5, 47.6, 46.5], [49.0, 46.6, 44.0, 45.9, 52.3, 45.6, 42.8, 62.0], [48.3, 43.2, 42.4, 42.3, 50.4, 42.2, 40.8, 59.4, 47.7], [47.0, 42.5, 39.0, 41.6, 48.8, 41.1, 38.5, 56.4, 45.9, 47.1]]
]

pmi_data = [
    [[92.3], [78.0, 73.6], [68.8, 63.3, 72.0], [62.8, 59.0, 64.0, 62.5], [59.7, 52.9, 56.5, 57.1, 65.1], [55.2, 46.1, 51.4, 53.4, 58.4, 48.7], [53.0, 44.2, 46.3, 47.7, 52.8, 47.3, 49.6], [49.8, 41.3, 44.3, 45.2, 50.9, 45.0, 46.1, 62.5], [48.0, 38.3, 42.1, 43.1, 49.3, 40.8, 43.3, 59.3, 50.0], [45.7, 36.5, 39.2, 42.3, 45.0, 39.0, 42.2, 56.4, 49.0, 49.6]],
    [[92.7], [78.3, 73.8], [69.3, 62.7, 71.0], [62.3, 55.1, 63.3, 62.5], [58.0, 50.3, 56.8, 55.4, 64.0], [54.7, 44.6, 50.5, 52.0, 57.8, 49.5], [50.9, 42.6, 45.5, 48.4, 52.0, 49.6, 48.6], [48.0, 39.3, 45.3, 43.7, 50.8, 46.1, 46.4, 62.6], [46.0, 35.5, 41.5, 40.3, 47.5, 42.6, 42.8, 60.0, 44.5], [45.1, 35.0, 40.7, 39.1, 44.6, 41.2, 41.7, 57.8, 45.3, 46.7]],
    [[92.4], [78.7, 73.6], [67.9, 64.6, 72.2], [60.0, 56.6, 62.3, 61.7], [56.1, 49.6, 56.3, 55.9, 65.8], [52.4, 46.4, 49.5, 53.9, 59.8, 51.4], [48.8, 45.9, 45.2, 49.3, 53.6, 49.5, 53.8], [44.1, 42.0, 43.3, 46.7, 50.7, 47.3, 48.9, 65.2], [43.4, 39.5, 41.0, 44.6, 49.2, 46.1, 45.8, 63.3, 48.5], [43.2, 38.6, 37.7, 43.7, 47.4, 43.3, 42.2, 61.1, 46.3, 46.6]]
]

def extract_metrics(data):
    new_task_accs = []
    old_task_accs = []
    for seed_data in data:
        # The new task accuracy is always the last item in the list at each step
        seed_new = [step[-1] for step in seed_data]
        
        # Retention is the average accuracy of all tasks EXCEPT the newly learned one
        seed_old = []
        for i, step in enumerate(seed_data):
            if i == 0:
                seed_old.append(np.nan) # No "old" tasks exist at step 1
            else:
                seed_old.append(np.mean(step[:-1]))
                
        new_task_accs.append(seed_new)
        old_task_accs.append(seed_old)
        
    return np.array(new_task_accs), np.array(old_task_accs)

# Extract 
remix_new, remix_old = extract_metrics(remix_data)
pmi_new, pmi_old = extract_metrics(pmi_data)

# Calculate Means and Standard Deviations across the 3 seeds
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    remix_new_mean, remix_new_std = np.nanmean(remix_new, axis=0), np.nanstd(remix_new, axis=0)
    pmi_new_mean, pmi_new_std = np.nanmean(pmi_new, axis=0), np.nanstd(pmi_new, axis=0)
    
    remix_old_mean, remix_old_std = np.nanmean(remix_old, axis=0), np.nanstd(remix_old, axis=0)
    pmi_old_mean, pmi_old_std = np.nanmean(pmi_old, axis=0), np.nanstd(pmi_old, axis=0)

steps = np.arange(1, 11)

# Generate Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

# Plot 1: New Task Accuracy
axes[0].plot(steps, remix_new_mean, label='REMIX', color='#1f77b4', marker='o')
axes[0].fill_between(steps, remix_new_mean - remix_new_std, remix_new_mean + remix_new_std, color='#1f77b4', alpha=0.2)

axes[0].plot(steps, pmi_new_mean, label='PMI', color='#ff7f0e', marker='s')
axes[0].fill_between(steps, pmi_new_mean - pmi_new_std, pmi_new_mean + pmi_new_std, color='#ff7f0e', alpha=0.2)

axes[0].set_title('Learning Capacity (New Task Accuracy)', fontsize=16)
axes[0].set_xlabel('Task', fontsize=14)
axes[0].set_ylabel('Accuracy (%)', fontsize=14)
axes[0].set_xticks(steps)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=18)

axes[0].tick_params(axis='both', which='major', labelsize=14)

# Plot 2: Old Knowledge Retention
axes[1].plot(steps[1:], remix_old_mean[1:], label='REMIX', color='#1f77b4', marker='o')
axes[1].fill_between(steps[1:], remix_old_mean[1:] - remix_old_std[1:], remix_old_mean[1:] + remix_old_std[1:], color='#1f77b4', alpha=0.2)

axes[1].plot(steps[1:], pmi_old_mean[1:], label='PMI', color='#ff7f0e', marker='s')
axes[1].fill_between(steps[1:], pmi_old_mean[1:] - pmi_old_std[1:], pmi_old_mean[1:] + pmi_old_std[1:], color='#ff7f0e', alpha=0.2)

axes[1].set_title('Retention (Average Old Task Accuracy)', fontsize=16)
axes[1].set_xlabel('Task', fontsize=14)
axes[1].set_ylabel('Accuracy (%)', fontsize=14)
axes[1].set_xticks(steps)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=18)

axes[1].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
pdf_path = "resnet34_cifar100_acc.pdf"
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Plots saved successfully as: {pdf_path}")
plt.show()