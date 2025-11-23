import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import random

import numpy as np
import torch
import transformers
import utils
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model

import sys
sys.path.append("..")
from trainer import CustomLoRATrainer as Trainer

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    lora_r: int = field(default=16, metadata={"help": "Lora rank."})
    lora_alpha: float = field(default=16.0, metadata={"help": "Lora alpha."})

    target_modules: str = field(default="q_proj,k_proj,v_proj,down_proj,up_proj,o_proj,gate_proj")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset: str = field(default='alpaca', metadata={"help": "Dataset to use for training."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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
    """Dataset for supervised fine-tuning. Supports integer indexing, slicing and index lists."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # integer index
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])

        # slice
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]

        # list/tuple/ndarray/tensor of indices
        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(idx, (np.ndarray, torch.Tensor)):
                idx = list(idx.tolist())
            return [self[i] for i in idx]

        raise TypeError(f"Unsupported index type: {type(idx)}")


class SupervisedHFDataset(Dataset):
    """Dataset for supervised fine-tuning from a HuggingFace datasets.Dataset object.
    Supports integer indexing, slicing and index lists/arrays/tensors.
    """

    def __init__(self, hf_dataset, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        hf_dataset = hf_dataset.shuffle(seed=42).select(range(100000))

        hf_dataset = hf_dataset.rename_column("query", "instruction")
        hf_dataset = hf_dataset.rename_column("answer", "output")

        logging.warning("Formatting HuggingFace dataset inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in hf_dataset
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in hf_dataset]

        logging.warning("Tokenizing HuggingFace dataset inputs... This may take some time...")

        assert os.path.exists('/home/USER/.cache/huggingface/'), "Please make sure the cache directory exists."

        if 'llama' in tokenizer.name_or_path.lower():
            cache_path = '/home/USER/.cache/huggingface/code_feedback_llama_tokenized_dataset.pt'
        else:
            cache_path = '/home/USER/.cache/huggingface/code_feedback_tokenized_dataset.pt'

        if os.path.exists(cache_path):
            logging.warning(f"Loading tokenized data from cache: {cache_path}")
            data_dict = torch.load(cache_path)
        else:
            logging.warning(f"No cache found. Tokenizing and saving to: {cache_path}")
            data_dict = preprocess(sources, targets, tokenizer)
            try:
                torch.save(data_dict, cache_path)
            except Exception as e:
                logging.warning(f"Failed to write cache ({cache_path}): {e}")

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # integer index
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])

        # slice
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]

        # list/tuple/ndarray/tensor of indices
        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(idx, (np.ndarray, torch.Tensor)):
                idx = list(idx.tolist())
            return [self[i] for i in idx]

        raise TypeError(f"Unsupported index type: {type(idx)}")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_supervised_hf_data_module(tokenizer: transformers.PreTrainedTokenizer, hf_dataset) -> Dict:
    """Make dataset and collator for supervised fine-tuning from a HuggingFace dataset."""
    train_dataset = SupervisedHFDataset(hf_dataset, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

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
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if data_args.dataset == 'codefeedback':
        from datasets import load_dataset
        train_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train')
        data_module = make_supervised_hf_data_module(tokenizer=tokenizer, hf_dataset=train_dataset)

    if 'norslora' in training_args.output_dir:
        model_args.lora_alpha = model_args.lora_r * 2
    
    if 'loraga' in training_args.output_dir:
        from lora_ga import LoraGAConfig
        from lora_ga import get_peft_model as loraga_get_peft_model
        from accelerate import Accelerator
        from lora_ga.utils.lora_ga_utils import estimate_gradient, LoraGAContext
        from torch.utils.data import DataLoader

        config = LoraGAConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules.split(','),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
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
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules.split(','),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True if 'pissa' not in training_args.output_dir else 'pissa_niter_4',
            use_dora=True if 'dora' in training_args.output_dir else False,
            use_rslora=True if 'norslora' not in training_args.output_dir else False,
        )

        model = get_peft_model(model, lora_config)

    if 'fullft' in training_args.output_dir:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(target in name for target in model_args.target_modules.split(',')) and 'lora' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}, Ratio: {100 * trainable_params / total_params:.2f}%")
        
    if 'odlora' in training_args.output_dir:
        assert training_args.max_steps > 0
        # initialization phase
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
    else:
        assert training_args.max_steps == -1

    training_args.max_steps = -1

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    import torch.distributed as dist
    if dist.get_rank() != 0:
        exit(0)
    else:
        model.eval()

        from eval_humaneval import main as eval_humaneval
        import os
        eval_humaneval(model, tokenizer, training_args.output_dir.split('/')[-2])

        dir = os.path.join('./experiment/codefeedback',training_args.output_dir.split('/')[-2], "humaneval_samples.jsonl")

        os.system(f"python eval_human.py {dir}")

if __name__ == "__main__":
    train()


