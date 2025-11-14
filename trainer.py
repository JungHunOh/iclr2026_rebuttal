from transformers import Trainer
import torch
import math
import random
from torch.utils.data import SequentialSampler, DataLoader
from torch.optim._functional import adamw as functional_adamw

class CustomLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        mode = self.args.output_dir.split('_')[-2]

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        train_sampler = SequentialSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        mode = self.args.output_dir.split('_')[-2]

        if 'lorapro' in mode:
            super().create_optimizer_and_scheduler(num_training_steps)
        else:
            if self.optimizer is None:
                self.create_optimizer()
            
            param_groups = self.optimizer.param_groups
            assert len(param_groups) > 0

            new_param_groups = []
            for group in param_groups:
                if len(group['params']) > 10:
                    for module in self.model.modules():
                        if hasattr(module, 'lora_A'):
                            group_copy = {
                                'params': [module.lora_A['default'].weight, module.lora_B['default'].weight],
                                'lr': group.get('lr', 1e-3),
                                'betas': group.get('betas', (0.9, 0.999)),
                                'eps': group.get('eps', 1e-8),
                                'weight_decay': group.get('weight_decay', 0.0),
                                'amsgrad': getattr(self.optimizer, 'amsgrad', False)
                            }
                            new_param_groups.append(group_copy)
                else:
                    group_copy = {
                        'params': group['params'],
                        'lr': group.get('lr', 1e-3),
                        'betas': group.get('betas', (0.9, 0.999)),
                        'eps': group.get('eps', 1e-8),
                        'weight_decay': group.get('weight_decay', 0.0),
                        'amsgrad': getattr(self.optimizer, 'amsgrad', False)
                    }
                    new_param_groups.append(group_copy)

            assert type(self.optimizer) is torch.optim.AdamW, "only support AdamW optimizer"
            self.optimizer = CustomAdamW(
                new_param_groups,
                total_steps=num_training_steps,
                model=self.model,
                mode=mode,
                before_init=self.args.max_steps > 0
            )

            if self.args.max_steps > 0:
                self.lr_scheduler = NoScheduling(self.optimizer)
            else:
                self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

class CustomAdamW(torch.optim.AdamW):
    def __init__(self, params, total_steps, model=None, mode='base', before_init=False, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0
        self.mode = mode
        self.before_init = before_init
        self.interval = int(self.mode.split('interval')[-1]) if 'interval' in self.mode else 1
        self.total_steps = total_steps

        if hasattr(self.model, 'classifier'):
            self.classifier_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        elif hasattr(self.model, 'classification_head'):
            self.classifier_params = [p for p in self.model.classification_head.parameters() if p.requires_grad]

        layer = 0
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                if not hasattr(module, 'layer_idx'):
                    module.layer_idx = layer
                    layer += 1
                if not self.before_init and 'odlora' in self.mode:
                    if hasattr(module, 'proj_a'):
                        torch.nn.init.zeros_(module.lora_A['default'].weight)
                        torch.nn.init.zeros_(module.lora_B['default'].weight)

    def step(self, closure=None):
        self._step_count += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        with torch.no_grad():
            for group in self.param_groups[:-1]:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []
                max_exp_avg_sqs = []

                A, B = group['params']

                if A.grad is None or B.grad is None:
                    continue

                dA, dB = A.grad, B.grad

                params_with_grad.append(A)
                
                state_A = self.state[A]
                
                if len(state_A) == 0:
                    state_A['step'] = 0
                    state_A['exp_avg'] = torch.zeros_like(A.data)
                    state_A['exp_avg_sq'] = torch.zeros_like(A.data)
                
                exp_avg_A, exp_avg_sq_A = state_A['exp_avg'], state_A['exp_avg_sq']
                beta1, beta2 = group['betas']
                state_A['step'] += 1

                if 'scaledadam' in self.mode and 'noscale' not in self.mode:
                    try:
                        dA_scaled = torch.inverse(B.data.T @ B.data + 1e-6 * torch.eye(B.data.shape[1], device=B.data.device)) @ dA.data
                    except:
                        dA_scaled = torch.eye(B.data.shape[1], device=B.data.device) @ dA.data
                else:
                    dA_scaled = dA.data
                
                assert dA_scaled.shape == dA.data.shape
                
                exp_avg_A.mul_(beta1).add_(dA_scaled, alpha=1 - beta1)
                exp_avg_sq_A.mul_(beta2).addcmul_(dA_scaled, dA_scaled, value=1 - beta2)
                denom_A = exp_avg_sq_A.sqrt().add_(group['eps'])

                grads.append(dA_scaled)
                exp_avgs.append(exp_avg_A)
                exp_avg_sqs.append(exp_avg_sq_A)
                state_steps.append(torch.tensor(state_A['step'], device=A.data.device, dtype=torch.float32))
                #A.data.addcdiv_(exp_avg_A, denom_A, value=-step_size)

                params_with_grad.append(B)

                state_B = self.state[B]
                if len(state_B) == 0:
                    state_B['step'] = 0
                    state_B['exp_avg'] = torch.zeros_like(B.data)
                    state_B['exp_avg_sq'] = torch.zeros_like(B.data)
                
                exp_avg_B, exp_avg_sq_B = state_B['exp_avg'], state_B['exp_avg_sq']
                state_B['step'] += 1

                if 'scaledadam' in self.mode and 'noscale' not in self.mode:
                    try:
                        dB_scaled = dB.data @ torch.inverse(A.data @ A.data.T + 1e-6 * torch.eye(A.data.shape[0], device=A.data.device))
                    except:
                        dB_scaled = dB.data @ torch.eye(A.data.shape[0], device=A.data.device)
                else:
                    dB_scaled = dB.data

                assert dB_scaled.shape == dB.data.shape

                exp_avg_B.mul_(beta1).add_(dB_scaled, alpha=1 - beta1)
                exp_avg_sq_B.mul_(beta2).addcmul_(dB_scaled, dB_scaled, value=1 - beta2)
                denom_B = exp_avg_sq_B.sqrt().add_(group['eps'])

                grads.append(dB_scaled)
                exp_avgs.append(exp_avg_B)
                exp_avg_sqs.append(exp_avg_sq_B)
                state_steps.append(torch.tensor(state_B['step'], device=B.data.device, dtype=torch.float32))

                #B.data.addcdiv_(exp_avg_B, denom_B, value=-step_size)

                functional_adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group['amsgrad'],
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group.get('maximize', False),
                    foreach=group.get('foreach', None),
                    capturable=group.get('capturable', False),
                    differentiable=group.get('differentiable', False)
                )

        for p in self.param_groups[-1]['params']:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            max_exp_avg_sqs = []

            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.param_groups[-1]['betas']

            state['step'] += 1

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.param_groups[-1]['eps'])

            step_size = self.param_groups[-1]['lr']

            functional_adamw(
                [p],
                [grad],
                [exp_avg],
                [exp_avg_sq],
                max_exp_avg_sqs,
                [torch.tensor(state['step'], device=p.data.device, dtype=torch.float32)],
                amsgrad=self.param_groups[-1]['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=step_size,
                weight_decay=self.param_groups[-1]['weight_decay'],
                eps=self.param_groups[-1]['eps'],
                maximize=self.param_groups[-1].get('maximize', False),
                foreach=self.param_groups[-1].get('foreach', None),
                capturable=self.param_groups[-1].get('capturable', False),
                differentiable=self.param_groups[-1].get('differentiable', False)
            )
            
        if self._step_count % 10 == 0 and self.before_init:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    with torch.no_grad():
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        if hasattr(module, 'proj_a') and hasattr(module, 'proj_b') and 'lorauniform' not in self.mode:
                            u, s, v = torch.svd_lowrank(lora_B @ module.proj_a + module.proj_b @ lora_A, q=module.lora_A['default'].weight.shape[0]+10, niter=4)
                        else:
                            u, s, v = torch.svd_lowrank(lora_B @ lora_A, q=module.lora_A['default'].weight.shape[0]+10, niter=4)
                        u = u[:,:-10]
                        s = s[:-10]
                        v = v[:,:-10]
                        if 'lorauniform' in self.mode:
                            lora_B.data = u.clone().contiguous()
                            lora_A.data = v.T.clone().contiguous()
                            module.detached_a = v.T.clone().contiguous()
                            module.detached_b = u.clone().contiguous()
                        else:
                            torch.nn.init.zeros_(module.lora_A['default'].weight)
                            torch.nn.init.zeros_(module.lora_B['default'].weight)
                            if 'scaling' in self.mode:
                                module.proj_b = (u@torch.sqrt(torch.diag(s))).clone().contiguous()
                                module.proj_a = (torch.sqrt(torch.diag(s))@v.T).clone().contiguous()
                            else:
                                module.proj_b = u.clone().contiguous()
                                module.proj_a = v.T.clone().contiguous()
                            module.do_one = True
                        if hasattr(self.model, 'classifier'):
                            for p, p_init in zip(self.model.classifier.parameters(), self.classifier_params):
                                if p.requires_grad:
                                    p.data = p_init.data.clone().contiguous()
                        elif hasattr(self.model, 'classification_head'):
                            for p, p_init in zip(self.model.classification_head.parameters(), self.classifier_params):
                                if p.requires_grad:
                                    p.data = p_init.data.clone().contiguous()
                        self.state.clear()

        if not self.before_init and 'odlora' in self.mode and self._step_count % self.interval == 0:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    with torch.no_grad():
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        if self._step_count == 10:
                            if hasattr(module, 'proj_a') and hasattr(module, 'proj_b'):
                                u, s, v = torch.svd_lowrank(lora_B @ module.proj_a + module.proj_b @ lora_A, q=module.lora_A['default'].weight.shape[0]+10, niter=4)
                            else:
                                u, s, v = torch.svd_lowrank(lora_B @ lora_A, q=module.lora_A['default'].weight.shape[0]+10, niter=4)
                                module.do_one = True
                            u = u[:,:-10]
                            s = s[:-10]
                            v = v[:,:-10]
                            if 'scaling' in self.mode:
                                module.proj_b = (0.5**0.5)*(u@torch.sqrt(torch.diag(s))).clone().contiguous()
                                module.proj_a = (0.5**0.5)*(torch.sqrt(torch.diag(s))@v.T).clone().contiguous()
                                module.lora_A['default'].weight.data = (0.5**0.5) * (torch.sqrt(torch.diag(s)) @ v.T).clone().contiguous()
                                module.lora_B['default'].weight.data = (0.5**0.5) * (u @ torch.sqrt(torch.diag(s))).clone().contiguous()
                            else:
                                module.proj_b = u.clone().contiguous()
                                module.proj_a = v.T.clone().contiguous()
                                module.lora_A['default'].weight.data = 0.5 * (torch.diag(s) @ v.T).clone().contiguous()
                                module.lora_B['default'].weight.data = 0.5 * (u @ torch.diag(s)).clone().contiguous()
                            self.state.clear()
                        elif self._step_count > 10 and self._step_count % (self._step_count // (self.total_steps//5) + 1)==0:
                            if (torch.norm(module.lora_B['default'].weight, dim=0) > 1e-2).all() and (torch.norm(module.lora_A['default'].weight, dim=1) > 1e-2).all():
                                Q_A, R_A = torch.linalg.qr(lora_A.T, mode='reduced')
                                Q_B, R_B = torch.linalg.qr(lora_B, mode='reduced')
                                if 'qrab' in self.mode:
                                    module.lora_A['default'].weight.data = (Q_A * torch.diag(R_A)).T.clone().contiguous()
                                    module.lora_B['default'].weight.data = (Q_B * torch.diag(R_B)).clone().contiguous()
                                if 'noproj' not in self.mode:
                                    if 'scaling' in self.mode:
                                        module.proj_a = (Q_A * torch.diag(R_A)).T.clone().contiguous()
                                        module.proj_b = (Q_B * torch.diag(R_B)).clone().contiguous()
                                    else:
                                        module.proj_a = (Q_A * torch.sign(torch.diag(R_A))).T.clone().contiguous()
                                        module.proj_b = (Q_B * torch.sign(torch.diag(R_B))).clone().contiguous()
                            
        return loss

import torch
from torch.optim.lr_scheduler import _LRScheduler

class FixedThenLinearDecayWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, fixed_steps, total_steps, warmup_steps=0, final_lr=1e-6, last_epoch=-1):
        self.fixed_steps = fixed_steps
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.fixed_steps:
                # Fixed LR
                lr = base_lr
            elif step < self.fixed_steps + self.warmup_steps:
                # Linear warmup
                # Linear warmup from 0 to base_lr
                progress = (step - self.fixed_steps) / max(1, self.warmup_steps)
                lr = base_lr * progress
            else:
                # Linear decay after fixed_steps
                decay_steps = self.total_steps - self.fixed_steps
                t = min(max(0, step - self.fixed_steps), decay_steps) / max(1, decay_steps)
                lr = self.final_lr + (base_lr - self.final_lr) * (1 - t)
            lrs.append(lr)
        return lrs

class NoScheduling(_LRScheduler):
    def __init__(self, optimizer=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.base_lrs]
