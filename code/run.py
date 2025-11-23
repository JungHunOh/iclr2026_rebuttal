#!/usr/bin/env python3
import os
import yaml
import random
import argparse
import subprocess
import sys
import socket

def run_training(config_path, gpu):
    """
    Run training with the specified config file and GPU setup.
    
    Args:
        config_path (str): Path to the YAML config file
        gpu (str): Comma-separated GPU IDs (e.g., "0,1,2")
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"PID: {os.getpid()}")
    print(f"Loading config from: {config_path}")
    print(f"Running training for {config['model']} with {config['method']} method")
    
    # Extract config values
    model = config['model']
    base_model = config['base_model']
    method = config['method']
    bs = config['bs']
    dataset = config['dataset']
    lr_values = config['lr'] if isinstance(config['lr'], list) else [config['lr']]
    mini_bs = config['mini_bs']
    epoch = config['epoch']
    seed = config['seed']
    target_modules = config['target_modules']
    r_values = config['r']
    scale = config['scale']
    max_steps = config['max_steps']
    lora_dropout = config['lora_dropout']

    # Calculate number of GPUs
    num_gpus = len(gpu.split(','))
    
    # Adjust mini_bs if necessary
    if num_gpus * mini_bs > bs:
        mini_bs = bs // num_gpus
        print(f"Adjusted mini_bs to {mini_bs} to fit batch size")
    
    # Run training for each lr and r combination
    for lr in lr_values:
        for r in r_values:
            # Create output directory
            output_dir = f"./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_{method}_seed{seed}/"
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare the command
            gradient_accumulation_steps = bs // mini_bs // num_gpus
            
            cmd = [
                "python3", "-m", "torch.distributed.launch",
                f"--master_port={find_free_port()}",
                f"--nproc_per_node={num_gpus}",
                "--use_env", "train.py",
                f"--model_name_or_path={base_model}",
                f"--dataset={config['dataset']}",
                f"--bf16={'True' if config['bf16'] else 'False'}",
                f"--fp16={'True' if config['fp16'] else 'False'}",
                f"--output_dir={output_dir}",
                f"--per_device_train_batch_size={mini_bs}",
                f"--per_device_eval_batch_size={config['per_device_eval_batch_size']}",
                f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                f"--eval_strategy={config['eval_strategy']}",
                f"--save_strategy={config['save_strategy']}",
                f"--learning_rate={lr}",
                f"--weight_decay={config['weight_decay']}",
                f"--warmup_ratio={config['warmup_ratio']}",
                f"--logging_steps={config['logging_steps']}",
                f"--max_steps={max_steps}",
                f"--num_train_epochs={epoch}",
                f"--lr_scheduler_type={config['lr_scheduler_type']}",
                f"--target_modules={target_modules}",
                f"--lora_r={r}",
                f"--lora_alpha={scale}",
                f"--seed={seed}",
                f"--lora_dropout={lora_dropout}"
                f"| tee {output_dir}log.txt"
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            
            # Run the command (use shell so the trailing "| tee ..." works)
            cmd_str = " ".join(cmd)
            print(f"Running command: {cmd_str}")
            result = subprocess.run(cmd_str, env=env, shell=True)
    

def main():
    parser = argparse.ArgumentParser(description="Run training with config file")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--gpu", default="0", help="Comma-separated GPU IDs (default: '0')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    run_training(args.config, args.gpu)

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    main()
