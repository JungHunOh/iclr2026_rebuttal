import os


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

print('enter gpu')
gpu = input()

#model = 'google-t5/t5-base'
model = 'microsoft/deberta-v3-base'

if model == 'google-t5/t5-base':
    model_name = 't5-base'
elif model == 'microsoft/deberta-v3-base':
    model_name = 'deberta-v3-base'

#for task in ["mrpc", "sst2", "cola", "mnli", "qnli", "rte", "qqp", "stsb"]:
for task in ["rte"]:
    for seed in [1]:
        #for target_modules in ['v', 'q v', 'q k v']: # t5
        for target_modules in ['value_proj', 'query_proj value_proj', 'query_proj key_proj value_proj']: # deberta-v3
            if model == 'google-t5/t5-base':
                target_modules_name = target_modules.replace(' ', '')
            else:
                target_modules_name = target_modules.replace(' ', '').replace('_proj','')
            for scale in [1]:
                for r in [32]:
                    if task == 'cola' or task == 'mrpc' or task == 'stsb' or task == 'rte':
                        epoch = 20
                    else:
                        epoch = 20

                    alpha = round(r**0.5 * scale,1)

                    bs = 128

                    for method in ['ourssvdtest']:  # ours, oursnew, adalora, lora
                    
                        lr = 1e-3

                        ratio = 0

                        output_dir = f"./experiment/{task}/{model_name}_epoch{epoch}_bs{bs}_lr{lr}_r{r}_scale{scale}_seed{seed}_{method}_{target_modules_name}"

                        cmd = (
                            f"export CUDA_VISIBLE_DEVICES={gpu} && "
                            f"python run_glue.py "
                            f"--model_name_or_path {model} "
                            f"--task_name {task} "
                            f"--do_train "
                            f"--do_eval "
                            f"--max_seq_length 256 "
                            f"--gradient_accumulation_steps {bs//32} "
                            f"--per_device_train_batch_size {32} "
                            f"--learning_rate {lr} "
                            f"--cls_lr 4e-3 "
                            f"--num_train_epochs {epoch} "
                            f"--output_dir {output_dir}/model "
                            f"--overwrite_output_dir "
                            f"--logging_steps 10 "
                            f"--logging_dir {output_dir}/log "
                            f"--eval_strategy no "
                            f"--save_strategy no "
                            f"--warmup_ratio 0 "
                            f"--lora_r {r} "
                            f"--lora_alpha {alpha} "
                            f"--seed {seed} "
                            f"--weight_decay 0 "
                            f"--prepare_ratio {ratio} "
                            f"--target_modules {target_modules} "
                        )

                        os.system(cmd)


# number of training samples
# 1. mrpc: 3668
# 2. sst2: 67349
# 3. cola: 8551
# 4. mnli: 392702
# 5. qnli: 104743
# 6. rte: 2490
# 7. wnli: 635
# 8. qqp: 363846
# 9. stsb: 5749

# number of epochs, LR (TSD)
# 1. mrpc: 30, 1e-3
# 2. sst2: 24, 8e-4
# 3. cola: 25, 8e-4
# 4. mnli: 12, 5e-4
# 5. qnli: 5, 5e-4
# 6. rte: 50, 1.2e-3
# 7. wnli: ?
# 8. qqp: 5, 1e-3
# 9. stsb: 25, 5e-4

# number of epochs, LR (adalora)
# 1. mrpc: 30, 1e-3
# 2. sst2: 24, 8e-4
# 3. cola: 25, 5e-4
# 4. mnli: 7, 5e-4
# 5. qnli: 5, 1.2e-3
# 6. rte: 50, 1.2e-3
# 7. wnli: ?
# 8. qqp: 5, 5e-4
# 9. stsb: 25, 2.2e-3

# lora github
# 1. mrpc: 30, 2e-4
# 2. sst2: 16, 6e-5
# 3. cola: 10, 1.3e-4
# 4. mnli: 5, 1e-4
# 5. qnli: 8, 1e-4
# 6. rte: 11, 2.6e-4
# 7. wnli: ?
# 8. qqp: 11, 1e-4
# 9. stsb: 25, 2.2e-3

# fixed hyper-parameters
# 1. mrpc: 20, 5e-4
# 2. sst2: 24, 8e-4
# 3. cola: 25, ?
# 4. mnli: ?, 5e-4
# 5. qnli: 5, ?
# 6. rte: 50, 1.2e-3
# 7. wnli: ?
# 8. qqp: 5, ?
# 9. stsb: 25, ?