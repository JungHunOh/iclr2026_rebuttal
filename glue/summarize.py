#!/usr/bin/env python3
"""
Script to summarize and visualize GLUE experiment results.
Creates scatter plots for each dataset, differentiating results by 'r' (rank) using different colors.
X-axis: scale, Y-axis: accuracy
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


def parse_experiment_name(exp_name: str) -> Dict[str, any]:
    """
    Parse experiment name to extract parameters.
    Expected format: epoch{X}_lr{Y}_r{Z}_scale{W}_{method}_seed{S}
    """
    pattern = r'epoch(\d+)_lr([\d.]+)_r(\d+)_scale([\d.]+)_([^_]+)_seed(\d+)'
    match = re.match(pattern, exp_name)
    
    if match:
        return {
            'epoch': int(match.group(1)),
            'lr': float(match.group(2)),
            'r': int(match.group(3)),
            'scale': float(match.group(4)),
            'method': match.group(5),
            'seed': int(match.group(6))
        }
    else:
        # Fallback parsing for different formats
        parts = exp_name.split('_')
        params = {}
        for part in parts:
            if part.startswith('epoch'):
                params['epoch'] = int(part.replace('epoch', ''))
            elif part.startswith('lr'):
                params['lr'] = float(part.replace('lr', ''))
            elif part.startswith('r'):
                params['r'] = int(part.replace('r', ''))
            elif part.startswith('scale'):
                params['scale'] = float(part.replace('scale', ''))
            elif part.startswith('seed'):
                params['seed'] = int(part.replace('seed', ''))
            elif part not in ['epoch', 'lr', 'r', 'scale', 'seed']:
                params['method'] = part
        return params


def load_experiment_results(experiment_dir: str) -> Dict:
    """
    Load all experiment results from the given directory.
    Expected structure: {dataset}/{experiment_name}/eval_results.json
    Returns a dictionary with structured data instead of pandas DataFrame.
    """
    results = []
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Warning: Experiment directory {experiment_dir} does not exist.")
        return {'data': [], 'datasets': [], 'ranks': [], 'scales': [], 'methods': []}
    
    # Iterate through each dataset directory
    for dataset_dir in experiment_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"Processing dataset: {dataset_name}")
        
        # Iterate through each experiment in the dataset
        for exp_dir in dataset_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            exp_name = exp_dir.name
            eval_file = exp_dir / "model" / "eval_results.json"
            
            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    # Parse experiment parameters
                    params = parse_experiment_name(exp_name)
                    
                    # Combine data
                    result_entry = {
                        'dataset': dataset_name,
                        'experiment_name': exp_name,
                        **params,
                        **eval_data
                    }
                    
                    results.append(result_entry)
                    
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {eval_file}: {e}")
            else:
                print(f"No eval_results.json found in {exp_dir}/model/")
    
    # Extract unique values
    datasets = list(set([r['dataset'] for r in results]))
    ranks = sorted(list(set([r.get('r', 0) for r in results if 'r' in r])))
    scales = sorted(list(set([r.get('scale', 0) for r in results if 'scale' in r])))
    methods = list(set([r.get('method', '') for r in results if 'method' in r]))
    
    return {
        'data': results,
        'datasets': datasets,
        'ranks': ranks,
        'scales': scales,
        'methods': methods
    }


def determine_accuracy_column(data: List[Dict]) -> str:
    """
    Determine which column contains the accuracy metric.
    Prioritize eval_accuracy for GLUE tasks.
    """
    if not data:
        return None
        
    sample = data[0]
    
    # For GLUE tasks, prioritize eval_accuracy
    if 'eval_accuracy' in sample:
        return 'eval_accuracy'
    
    accuracy_columns = [
        'accuracy', 'eval_acc', 'acc',
        'eval_f1', 'f1', 'eval_matthews_correlation', 'matthews_correlation'
    ]
    
    for col in accuracy_columns:
        if col in sample:
            return col
    
    # If no standard accuracy column found, look for columns containing 'acc' or 'f1'
    for col in sample.keys():
        if 'acc' in col.lower() or 'f1' in col.lower() or 'correlation' in col.lower():
            return col
    
    print("Warning: No accuracy column found. Available columns:", list(sample.keys()))
    return None


def create_scatter_plots(results_dict: Dict, output_dir: str = "plots"):
    """
    Create separate scatter plots for each dataset and rank combination,
    with scale vs accuracy, differentiating methods by shape.
    """
    data = results_dict['data']
    if not data:
        print("No data to plot.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine accuracy column
    accuracy_col = determine_accuracy_column(data)
    if accuracy_col is None:
        print("Cannot create plots: no accuracy column found.")
        return
    
    # Get unique datasets and ranks
    datasets = results_dict['datasets']
    ranks = results_dict['ranks']
    methods = results_dict['methods']
    
    # Set up colors for different methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    color_map = dict(zip(methods, colors))
    
    # Create separate plots for each dataset and rank combination
    for dataset in datasets:
        for rank in ranks:
            # Filter data for this dataset and rank
            dataset_rank_data = [d for d in data if d['dataset'] == dataset and d.get('r', 0) == rank]
            
            if not dataset_rank_data:
                continue
            
            # Aggregate results by taking mean across seeds
            aggregated = defaultdict(lambda: defaultdict(list))
            
            for entry in dataset_rank_data:
                key = (entry.get('scale', 0), entry.get('method', ''))
                aggregated[key][accuracy_col].append(entry.get(accuracy_col, 0))
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot points for each scale and method
            for (scale, method), metrics in aggregated.items():
                acc_values = metrics[accuracy_col]
                if not acc_values:
                    continue
                    
                mean_acc = np.mean(acc_values)
                
                marker = 'o' if method == 'base' else 's'  # circle for base, square for others
                
                ax.scatter(
                    scale, 
                    mean_acc,
                    c=[color_map.get(method, 'gray')], 
                    label=f'{method}',
                    s=120,
                    marker=marker,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.8
                )
            
            # Customize the plot
            ax.set_xlabel('Scale', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset.upper()} - Scale vs Accuracy (r={rank})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')
            
            # Adjust layout and save
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'{dataset}_r{rank}_scale_vs_accuracy.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot for {dataset}, r={rank}: {plot_filename}")
    
    # Create a summary plot with all datasets and ranks
    #create_summary_plot(results_dict, accuracy_col, output_dir)


def create_summary_plot(results_dict: Dict, accuracy_col: str, output_dir: str):
    """
    Create a summary plot with all dataset and rank combinations in subplots.
    """
    data = results_dict['data']
    datasets = sorted(results_dict['datasets'])
    ranks = sorted(results_dict['ranks'])
    methods = results_dict['methods']
    
    # Calculate subplot layout
    n_plots = len(datasets) * len(ranks)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set up colors for different methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    color_map = dict(zip(methods, colors))
    
    plot_idx = 0
    for dataset in datasets:
        for rank in ranks:
            if plot_idx >= n_plots:
                break
                
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            if n_plots == 1:
                ax = axes[0]
            elif n_rows == 1:
                ax = axes[0, col]
            elif n_cols == 1:
                ax = axes[row, 0]
            else:
                ax = axes[row, col]
            
            # Filter data for this dataset and rank
            dataset_rank_data = [d for d in data if d['dataset'] == dataset and d.get('r', 0) == rank]
            
            if not dataset_rank_data:
                ax.set_title(f'{dataset.upper()}, r={rank}\n(No data)')
                ax.set_xlabel('Scale')
                ax.set_ylabel('Accuracy')
                plot_idx += 1
                continue
            
            # Aggregate results
            aggregated = defaultdict(lambda: defaultdict(list))
            
            for entry in dataset_rank_data:
                key = (entry.get('scale', 0), entry.get('method', ''))
                aggregated[key][accuracy_col].append(entry.get(accuracy_col, 0))
            
            # Plot
            for (scale, method), metrics in aggregated.items():
                acc_values = metrics[accuracy_col]
                if not acc_values:
                    continue
                    
                mean_acc = np.mean(acc_values)
                marker = 'o' if method == 'base' else 's'
                
                ax.scatter(
                    scale, 
                    mean_acc,
                    c=[color_map.get(method, 'gray')], 
                    label=f'{method}' if plot_idx == 0 else "",
                    s=50,
                    marker=marker,
                    alpha=0.7
                )
            
            ax.set_xlabel('Scale')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{dataset.upper()}, r={rank}')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Remove empty subplots
    for idx in range(plot_idx, n_rows * n_cols):
        if idx >= len(axes.flat):
            break
        axes.flat[idx].remove()
    
    # Add legend to the figure
    if methods:
        handles = []
        labels = []
        for method in methods:
            marker = 'o' if method == 'base' else 's'
            handles.append(plt.scatter([], [], c=color_map.get(method, 'gray'), marker=marker, s=50))
            labels.append(method)
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    summary_filename = os.path.join(output_dir, 'summary_all_datasets_ranks.png')
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {summary_filename}")


def create_sample_data(experiment_dir: str):
    """
    Create sample eval_results.json files for demonstration.
    """
    datasets = ['mrpc', 'rte']
    scales = [0.5, 1.0, 2.0]
    ranks = [8, 32]
    methods = ['base', 'oursnew']
    seeds = [1, 2, 3]
    
    # Different accuracy ranges for different datasets
    accuracy_ranges = {
        'mrpc': (0.78, 0.88),
        'rte': (0.62, 0.72)
    }
    
    for dataset in datasets:
        dataset_dir = Path(experiment_dir) / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for scale in scales:
            for rank in ranks:
                for method in methods:
                    for seed in seeds:
                        # Create experiment directory
                        epochs = 20 if dataset == 'mrpc' else 30
                        exp_name = f"epoch{epochs}_lr0.001_r{rank}_scale{scale}_{method}_seed{seed}"
                        exp_dir = dataset_dir / exp_name
                        exp_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate sample accuracy
                        base_acc = np.random.uniform(*accuracy_ranges[dataset])
                        
                        # Add some systematic effects
                        if method == 'oursnew':
                            base_acc += 0.02  # oursnew is slightly better
                        if rank == 32:
                            base_acc += 0.01  # higher rank is slightly better
                        if scale == 2.0:
                            base_acc -= 0.01  # higher scale might hurt performance
                        
                        # Add noise
                        accuracy = base_acc + np.random.normal(0, 0.01)
                        accuracy = np.clip(accuracy, 0, 1)
                        
                        # Create eval_results.json
                        results = {
                            'eval_accuracy': accuracy,
                            'eval_loss': np.random.uniform(0.3, 0.8),
                            'eval_f1': accuracy + np.random.normal(0, 0.005),
                            'eval_runtime': np.random.uniform(50, 150),
                            'eval_samples_per_second': np.random.uniform(20, 50),
                        }
                        
                        with open(exp_dir / 'eval_results.json', 'w') as f:
                            json.dump(results, f, indent=2)
    
    print(f"Created sample data in {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Summarize GLUE experiment results")
    parser.add_argument('--experiment_dir', default='experiment', 
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='plots', 
                       help='Directory to save plots')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample data for demonstration')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to summarize (e.g., mrpc, rte). If not specified, all datasets will be processed.')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_data(args.experiment_dir)
    
    # Load and process results
    print(f"Loading experiment results from {args.experiment_dir}")
    results_dict = load_experiment_results(args.experiment_dir)
    
    if not results_dict['data']:
        print("No data found. Use --create_sample to generate sample data.")
        return
    
    # Filter by dataset if specified
    if args.dataset:
        original_data = results_dict['data']
        filtered_data = [d for d in original_data if d['dataset'] == args.dataset]
        
        if not filtered_data:
            print(f"No data found for dataset '{args.dataset}'.")
            print(f"Available datasets: {results_dict['datasets']}")
            return
        
        # Update results_dict with filtered data
        results_dict['data'] = filtered_data
        results_dict['datasets'] = [args.dataset]
        # Recalculate other unique values for the filtered dataset
        ranks = sorted(list(set([r.get('r', 0) for r in filtered_data if 'r' in r])))
        scales = sorted(list(set([r.get('scale', 0) for r in filtered_data if 'scale' in r])))
        methods = list(set([r.get('method', '') for r in filtered_data if 'method' in r]))
        results_dict['ranks'] = ranks
        results_dict['scales'] = scales
        results_dict['methods'] = methods
        
        print(f"Filtered to dataset: {args.dataset}")
    
    data = results_dict['data']
    print(f"Loaded {len(data)} experiment results")
    print(f"Datasets: {results_dict['datasets']}")
    print(f"Ranks: {results_dict['ranks']}")
    print(f"Scales: {results_dict['scales']}")
    print(f"Methods: {results_dict['methods']}")
    
    # Create summary statistics
    print("\nSummary statistics:")
    accuracy_col = determine_accuracy_column(data)
    if accuracy_col:
        # Print aggregated statistics
        stats = defaultdict(lambda: defaultdict(list))
        for entry in data:
            key = (entry['dataset'], entry.get('r', 0), entry.get('scale', 0), entry.get('method', ''))
            stats[key][accuracy_col].append(entry.get(accuracy_col, 0))
        
        for (dataset, r, scale, method), metrics in stats.items():
            acc_values = metrics[accuracy_col]
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values) if len(acc_values) > 1 else 0
            print(f"{dataset}, r={r}, scale={scale}, {method}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(acc_values)})")
    
    # Create plots
    create_scatter_plots(results_dict, args.output_dir)
    
    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
