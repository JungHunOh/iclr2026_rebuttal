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

def plot_loss_curves(experiment_paths, dataset_name='Code-Feedback', smooth_window=300, 
                     figsize=(6,4), output_path=None):
    """
    Plot loss curves for multiple experiments.
    
    Args:
        experiment_paths (list): List of paths to log.txt files
        dataset_name (str): Name of the dataset for title
        smooth_window (int): Window size for smoothing
        figsize (tuple): Figure size
        output_path (str): Path to save the plot (optional)
    """
    # Extract labels and process losses for each experiment
    exp_data = []
    for exp_path in experiment_paths:
        # Extract label from path
        label = exp_path.split('/')[-2].split('_')[-2]
        
        # Read losses from log file
        losses = []
        with open(exp_path) as f:
            lines = f.readlines()
            after_init = False if 'odlora' in exp_path else True
            for line in lines:
                if 'train_runtime' in line:
                    after_init = True
                if 'grad_norm' in line and after_init:
                    losses.append(float(line.split(' ')[1][:-1]))
        
        # Apply smoothing
        losses_smooth = smooth_curve(losses, smooth_window)
        
        exp_data.append({
            'label': label,
            'losses': losses,
            'losses_smooth': losses_smooth
        })

    plt.figure(figsize=figsize)

    # Plot smoothed curves for all experiments
    for data in exp_data:
        plt.plot(np.arange(len(data['losses_smooth'])), data['losses_smooth'], 
                 label=data['label'], linewidth=1)

    plt.xlabel('Iteration', fontsize=17) 
    plt.ylabel('Training Loss', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.title(dataset_name, fontsize=17)
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{dataset_name}.png', dpi=300, bbox_inches='tight')
    
    return exp_data

# Example usage with the original 3 experiments
dataset = 'code'
smooth = 300

# Define experiments - you can add/remove as needed
experiments = [
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-4_base_seed1/log.txt',
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-4_lorauniform_seed1/log.txt',
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-4_odlora_seed1/log.txt',
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-4_odloralesscap_seed1/log.txt',
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-4_odlorascaling_seed1/log.txt',
    'code/trained_models/llama3_codefeedback_epoch1_bs32_r8_scale4_lr1e-5_fullft_seed1/log.txt',
]

# Use the function to plot
exp_data = plot_loss_curves(experiments, dataset, smooth)
