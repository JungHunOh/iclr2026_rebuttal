import os
import random

print('enter gpu')
gpu=input()

dataset='commonsense_170k'


for model in ['gemma']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'
    
    epoch=1
    bs=32
    mini_bs=16 if model == 'llama3' else 32
    scale=4

    if model == 'llama3':
        lr=1e-4
    elif model == 'gemma':
        lr=2e-4

    target_modules=["q_proj", "k_proj", "v_proj", "down_proj", "up_proj", "gate_proj", "o_proj"]

    for seed in [1]:
        #for method in ['base', 'pissa', 'dora', 'odlora', 'norslora', 'lora+']:
        for method in ['scaledadam']:
            for r in [8]:
                if 'odlora' in method or 'lorauniform' in method:
                    max_steps = 55
                else:
                    max_steps = -1
                
                if 'fullft' in method:
                    if model == 'llama3':
                        mini_bs = 1
                        lr = 5e-6
                    else:
                        mini_bs = 8
                        lr = 1e-5
                    
                num_gpus = len(gpu.split(','))

                if num_gpus * mini_bs > bs:
                        mini_bs = bs // num_gpus
                
                os.makedirs(f'./trained_models/{model}_{dataset}_lr{lr}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_{method}_seed{seed}/', exist_ok=True)
                cmd = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python3 -m torch.distributed.launch --master_port {random.randint(1000,2000)} --nproc_per_node={num_gpus} --use_env finetune.py '
                    f'--base_model {base_model} '
                    f'--data_path ./ft-training_set/{dataset}.json '
                    f'--output_dir ./trained_models/{model}_{dataset}_lr{lr}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_{method}_seed{seed}/ '
                    f'--batch_size {bs} '
                    f'--micro_batch_size {mini_bs} '
                    f'--num_epochs {epoch} '
                    f'--learning_rate {lr} '
                    f'--cutoff_len 256 '
                    f'--val_set_size 0 '
                    f'--adapter_name lora '
                    f'--lora_r {r} '
                    f'--lora_alpha {scale} '
                    f'--target_modules "{",".join(target_modules)}" '
                    f'--seed {seed} '
                    f'--max_steps {max_steps} '
                    f'| tee ./trained_models/{model}_{dataset}_lr{lr}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_{method}_seed{seed}/log.txt'
                )
                os.system(cmd)
                print(f'./trained_models/{model}_{dataset}_lr{lr}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_{method}_seed{seed}')
input()
