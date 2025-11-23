#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import io
import torch
import transformers
from torch.utils.data import Dataset
#from transformers import Trainer
import argparse
import json
import random
import numpy as np

from peft import get_peft_model, TaskType, LoraConfig

from tqdm import tqdm
from functools import partial, reduce

import sys
sys.path.append("../")
from trainer import CustomLoRATrainer as Trainer

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
#### 28
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_length: int = field(default=100000, metadata={"help": "Length of the training data."})


@dataclass
class LoRAArguments:
    adapter_name: str = field(default='lora')
    lora_r: int = field(default=None, metadata={"help": "Rank of the low-rank decomposition."})
    lora_alpha: float = field(default=None)
    lora_init: any = field(default=True)
    lora_dropout: float = field(default=0.05)
    target_modules: str = field(default=None, metadata={"help": "Target modules to apply LoRA to, separated by commas."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            data_path = data_path_map[data_path]
        except:
            data_path = data_path
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
        list_data_dict = list_data_dict[:data_args.data_length]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        # support slicing like ds[:32] -> returns a Subset
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            indices = list(range(start, stop, step))
            return torch.utils.data.Subset(self, indices)
        # support list/tuple/ndarray of indices
        if isinstance(i, (list, tuple, np.ndarray)):
            indices = list(i)
            return torch.utils.data.Subset(self, indices)
        # single index
        if isinstance(i, int):
            if i < 0:
                i += len(self)
            return dict(input_ids=self.sources[i], labels=self.targets[i])
        raise TypeError(f"Invalid index type: {type(i)}")

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama-2" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if 'norslora' in training_args.output_dir:
        lora_args.lora_alpha = lora_args.lora_r * 2

    if 'loraga' in training_args.output_dir:
        from lora_ga import LoraGAConfig
        from lora_ga import get_peft_model as loraga_get_peft_model
        from accelerate import Accelerator
        from lora_ga.utils.lora_ga_utils import estimate_gradient, LoraGAContext
        from torch.utils.data import DataLoader

        config = LoraGAConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.target_modules.split(','),
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            init_lora_weights=True if 'pissa' not in training_args.output_dir else 'pissa_niter_4',
            use_dora=True if 'dora' in training_args.output_dir else False,
            use_rslora=True if 'norslora' not in training_args.output_dir else False,
            iters=32 // 2,
        )

        temp_set = data_module['train_dataset'][:32]

        named_grad = estimate_gradient(
            model=model,
            dataloader=DataLoader(temp_set, shuffle=False, batch_size=2, collate_fn=data_module['data_collator']),
            accelerator=Accelerator(),
            quant_flag=False,
        )
        with LoraGAContext(model=model, named_grad=named_grad):
            model = loraga_get_peft_model(model=model, peft_config=config)
        
    else:
        config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.target_modules.split(','),
                lora_dropout=lora_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                init_lora_weights=True if 'pissa' not in training_args.output_dir else 'pissa_niter_4',
                use_dora=True if 'dora' in training_args.output_dir else False,
                use_rslora=True if 'norslora' not in training_args.output_dir else False,
            )
    
        model = get_peft_model(model, config)
    
        
    if 'fullft' in training_args.output_dir:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(target in name for target in lora_args.target_modules.split(',')) and 'lora' not in name:
                param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}, Ratio: {100 * trainable_params / total_params:.2f}%")

    if 'odlora' in training_args.output_dir:
        assert training_args.max_steps > 0
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
    else:
        assert training_args.max_steps == -1

    training_args.max_steps = -1
    # if 'lorapro' in training_args.output_dir:
    #     ds_config = "../config/deepspeed_zero2.json"
    #     ds_config = os.path.abspath(os.path.expanduser(ds_config))
    #     assert os.path.isfile(ds_config), f"Deepspeed config not found: {ds_config}"
    #     training_args.deepspeed = ds_config
    #     training_args.optim = 'sgd'
    #     training_args = TrainingArguments(**training_args.to_dict())

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()

    trainer.save_state()
    # if os.environ.get('LOCAL_RANK') == '0':
    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    import torch.distributed as dist
    if dist.get_rank() != 0:
        exit(0)

    torch.cuda.empty_cache()

    return model, tokenizer, training_args.output_dir.split('/')[-2]

if __name__ == "__main__":
    model, tokenizer, name = train()
    import torch.distributed as dist
    if dist.get_rank() != 0:
        exit(0)
    else:
        from eval import gsm8k_test_noargs
        gsm8k_test_noargs(model, tokenizer, name, batch_size=100)
        if 'llama2' not in name:
            from eval_math import test_hendrycks_math
            test_hendrycks_math(model, tokenizer, name, batch_size=100)