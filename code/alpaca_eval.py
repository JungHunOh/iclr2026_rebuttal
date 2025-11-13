import json
import torch
import argparse
import math
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Constants ---
DATASET_NAME = "tatsu-lab/alpaca_eval"
DATASET_CONFIG = "alpaca_eval"
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

def eval(model, tokenizer, dir):
    save_dir = dir+'/alpaca_eval_outputs.json'
    model.eval()

    # --- 2. Load Dataset ---
    print("Loading AlpacaEval dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)["eval"]
    dataset = dataset.map(lambda example: {"instruction": PROMPT.format(instruction=example["instruction"])})

    # --- 3. Generate Outputs ---
    model_outputs = []
    print(f"\nStarting generation on {len(dataset)} examples...")

    batch_size = 50
    num_batches = math.ceil(len(dataset) / batch_size)
    print(f"\nStarting generation on {len(dataset)} examples in {num_batches} batches...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Batches"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_examples = [dataset[j] for j in batch_indices]
        instruction_batch = [ex['instruction'] for ex in batch_examples]

        inputs = tokenizer(instruction_batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Use greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode the batch of generated texts
        # We need to slice the prompt out of the generated tokens
        prompt_lengths = inputs['input_ids'].shape[1]
        decoded_texts = tokenizer.batch_decode(outputs[:, prompt_lengths:], skip_special_tokens=True)

        # Append results for the current batch
        for instruction, response_text in zip(instruction_batch, decoded_texts):
            # Remove the template parts from instruction to keep only the original instruction text
            original_instruction = instruction.split("### Instruction:")[-1].split("### Response:")[0].strip()
            model_outputs.append({
                "instruction": original_instruction,
                "output": response_text.strip(),
                "generator": dir
            })

    # --- 4. Save Outputs ---
    print(f"\nSaving outputs to {save_dir}...")
    with open(save_dir, "w") as f:
        json.dump(model_outputs, f, indent=4)

    print("--- Generation Complete! ---")
    print(f"Your outputs are ready at: ./{save_dir}")
    print("\nNext step: Run the alpaca_eval command on this file.")

    import os
    os.system(f"export OPENAI_API_KEY=tgp_v1_cPnYwhg9IUNxfvC-YhGvs8wzySBYED6B9SeH9xVgZIc && alpaca_eval --model_outputs {save_dir} --annotators_config alpaca_eval_llama3_70b_fn --output_path {dir}")
