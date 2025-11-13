import os
import math

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

cuda_visible_devices = input()  # Set your desired GPU index here

ii = int(cuda_visible_devices)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

for model in ['vit-base', 'vit-large']:
    for dataset in ['cifar100', 'resisc45', 'food101', 'dtd', 'cub200', 'cars', 'sun397']:
    #for dataset in ['sun397']:
        bs = 256
        mini_bs = 64

        if dataset in ['resisc45', 'food101']:
            epoch = 10
        elif dataset in ['sun397']:
            epoch = 15
        elif dataset in ['cifar100']:
            epoch = 7
        elif dataset in ['dtd', 'cub200']:
            epoch = 25
        elif dataset in ['cars']:
            epoch = 20
        
        if dataset == 'cifar100' or dataset == 'sun397':
            lr = 1e-3
        elif dataset == 'cars':
            lr = 5e-3
        else:
            lr = 2e-3

        for target_modules in ['query value']:
            target_modules_name = target_modules.replace(' ', '')

            for seed in [1,2,3]:
                for mode in ['base', 'pissa', 'dora', 'odlora', 'norslora', 'lora+']:
                #for mode in ['base']:
                    for r in [8,32]:
                        for scale in [4]:
                            if 'odlora' in mode:
                                max_steps = 50
                            else:
                                max_steps = -1
                            
                            output_dir = f"./experiment/{dataset}/{model}_epoch{epoch}_bs{bs}_lr{lr}_alpha{scale}_r{r}_{mode}_{target_modules_name}/"
                            os.makedirs(output_dir, exist_ok=True)
                            cmd = (
                                f"python finetune.py "
                                f"--eval_strategy no "
                                f"--save_strategy no "
                                f"--gradient_accumulation_steps {bs//mini_bs} "
                                f"--dataloader_num_workers 8 "
                                f"--logging_steps 10 "
                                f"--label_names labels "
                                f"--remove_unused_columns False "
                                f"--per_device_train_batch_size {mini_bs} "
                                f"--per_device_eval_batch_size 256 "
                                f"--seed {seed} "
                                f"--num_train_epochs {epoch} "
                                f"--lora_rank {r} "
                                f"--output_dir {output_dir} "
                                f"--model_name {model} "
                                f"--finetuning_method lora "
                                f"--dataset_name {dataset} "
                                f"--clf_learning_rate {lr} "
                                f"--other_learning_rate {lr} "
                                f"--warmup_ratio 0 "
                                f"--weight_decay 0 "
                                f"--lora_alpha {scale} "
                                f"--target_modules {target_modules} "
                                f"--max_steps {max_steps} "
                                f"| tee {output_dir}/log.txt"
                            )
                            os.system(cmd)
