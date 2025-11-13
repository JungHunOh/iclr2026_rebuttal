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
smooth = 100
odlora = 'commonsense/trained_models/gemma_commonsense_170k_lr0.0002_epoch1_bs32_r8_scale4_odlora_seed1/log.txt'
odlorav2 = 'commonsense/trained_models/gemma_commonsense_170k_lr0.0002_epoch1_bs32_r8_scale4_scaledadam_seed1/log.txt'

odlora_losses = []
odlorav2_losses = []
with open(odlora) as f:
    lines = f.readlines()
    after_init = False if 'odlora' in odlora else True
    for line in lines:
        if 'train_runtime' in line:
            after_init = True
        if 'grad_norm' in line and after_init:
            odlora_losses.append(float(line.split(' ')[1][:-1]))

with open(odlorav2) as f:
    lines = f.readlines()
    after_init = False if 'odlora' in odlorav2 else True
    for line in lines:
        if 'train_runtime' in line:
            after_init = True
        if 'grad_norm' in line and after_init:
            odlorav2_losses.append(float(line.split(' ')[1][:-1]))

# Apply smoothing to the losses
window_size = smooth  # Adjust this value to control smoothing amount
odlora_losses_smooth = smooth_curve(odlora_losses, window_size)
odlorav2_losses_smooth = smooth_curve(odlorav2_losses, window_size)

plt.figure(figsize=(6,4))

# Plot both original (faded) and smoothed curves
plt.plot(np.arange(len(odlora_losses_smooth)), odlora_losses_smooth, label='OD-LoRA', linewidth=1)
plt.plot(np.arange(len(odlorav2_losses_smooth)), odlorav2_losses_smooth, label='OD-LoRAv2', linewidth=1)

plt.xlabel('Iteration', fontsize=17) 
plt.ylabel('Training Loss', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.title(dataset, fontsize=17)
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f'{dataset}.png', dpi=300, bbox_inches='tight')
