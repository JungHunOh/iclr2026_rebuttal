import os
from matplotlib import pyplot as plt
import numpy as np

def smooth_curve(data, window_size=50):
    """Apply moving average smoothing to data."""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return smoothed

dataset = 'Code-Feedback'
smooth = 300
exp1 = 'commonsense/trained_models/gemma_commonsense_170k_lr0.0002_epoch1_bs32_r8_scale4_scaledadam_seed1/log.txt'
exp2 = 'commonsense/trained_models/gemma_commonsense_170k_lr0.0002_epoch1_bs32_r8_scale4_odloranoinit_seed1/log.txt'
exp3 = 'commonsense/trained_models/gemma_commonsense_170k_lr0.0002_epoch1_bs32_r8_scale4_base_seed1/log.txt'

label1 = exp1.split('/')[-2].split('_')[-2]
label2 = exp2.split('/')[-2].split('_')[-2]
label3 = exp3.split('/')[-2].split('_')[-2]

exp1_losses = []
exp2_losses = []
exp3_losses = []
with open(exp1) as f:
    lines = f.readlines()
    after_init = False if 'odlora' in exp1 else True
    for line in lines:
        if 'train_runtime' in line:
            after_init = True
        if 'grad_norm' in line and after_init:
            exp1_losses.append(float(line.split(' ')[1][:-1]))

with open(exp2) as f:
    lines = f.readlines()
    after_init = True if 'odlora' in exp2 else True
    for line in lines:
        if 'train_runtime' in line:
            after_init = True
        if 'grad_norm' in line and after_init:
            exp2_losses.append(float(line.split(' ')[1][:-1]))

with open(exp3) as f:
    lines = f.readlines()
    after_init = False if 'odlora' in exp3 else True
    for line in lines:
        if 'train_runtime' in line:
            after_init = True
        if 'grad_norm' in line and after_init:
            exp3_losses.append(float(line.split(' ')[1][:-1]))

# Apply smoothing to the losses
window_size = smooth  # Adjust this value to control smoothing amount
exp1_losses_smooth = smooth_curve(exp1_losses, window_size)
exp2_losses_smooth = smooth_curve(exp2_losses, window_size)
exp3_losses_smooth = smooth_curve(exp3_losses, window_size)

plt.figure(figsize=(6,4))

# Plot both original (faded) and smoothed curves
plt.plot(np.arange(len(exp1_losses_smooth)), exp1_losses_smooth, label=label1, linewidth=1)
plt.plot(np.arange(len(exp2_losses_smooth)), exp2_losses_smooth, label=label2, linewidth=1)
plt.plot(np.arange(len(exp3_losses_smooth)), exp3_losses_smooth, label=label3, linewidth=1)

plt.xlabel('Iteration', fontsize=17) 
plt.ylabel('Training Loss', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.title(dataset, fontsize=17)
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f'{dataset}.png', dpi=300, bbox_inches='tight')
