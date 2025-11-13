import os
import random

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

for model in ['llama3','gemma']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'
    elif model == 'llama2':
        base_model = 'meta-llama/Llama-2-7b-hf'

    bs=32

    dataset = 'metamath100k'
    dataset_name = 'MetaMathQA-395K'

    if model == 'llama3':
        lr = 1e-4
    elif model == 'llama2':
        lr = 2e-4
    elif model == 'gemma':
        lr = 2e-4

    epoch=1
    for seed in [1]:
        for target_modules in ['q_proj k_proj v_proj down_proj up_proj o_proj gate_proj']:
            target_modules_name = target_modules.replace(' ', '').replace('_proj','')
            for r, scale in [(32,4)]:
                for method in ['base', 'fullft', 'odlora', 'odlorascaling']:
                #for method in ['odlora']:
                    if 'dora' in method:
                        if model == 'llama3':
                            mini_bs=4
                        else:
                            mini_bs=8
                    else:
                        if model == 'llama3' or model == 'llama2':
                            mini_bs=8
                        else:
                            mini_bs=16
                    if 'odlora' in method or 'lorauniform' in method:
                        max_steps = 50
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
                    
                    os.makedirs(f'./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/', exist_ok=True)
                    os.system(f'CUDA_VISIBLE_DEVICES={gpu} python3 -m torch.distributed.launch --master_port {random.randint(1000,2000)}  --nproc_per_node={num_gpus} --use_env train_math.py \
                        --model_name_or_path {base_model}\
                        --data_path ft-training_set/{dataset_name}.json \
                        --data_length 100000 \
                        --bf16 True \
                        --fp16 False \
                        --output_dir ./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/\
                        --per_device_train_batch_size {mini_bs} \
                        --per_device_eval_batch_size 4 \
                        --gradient_accumulation_steps {bs//mini_bs//num_gpus} \
                        --evaluation_strategy "no" \
                        --save_strategy "no" \
                        --learning_rate {lr}\
                        --weight_decay 0 \
                        --warmup_ratio 0 \
                        --logging_steps 1 \
                        --max_steps {max_steps} \
                        --num_train_epochs {epoch} \
                        --lr_scheduler_type "cosine" \
                        --target_modules {target_modules} \
                        --lora_r {r} \
                        --lora_alpha {scale} \
                        --seed {seed} \
                        --lora_dropout 0.05 \
                        | tee ./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/log.txt\
                        ')
                    print(f'./trained_models/{model}_{dataset}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/')
input()