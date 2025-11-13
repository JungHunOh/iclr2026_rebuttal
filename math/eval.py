import argparse
import jsonlines
import re
from fractions import Fraction
import transformers
import torch
import sys
import os
import time
from datetime import timedelta

MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        return round(float(frac.numerator / frac.denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def truncate_at_stop_tokens(text, stop_tokens):
    earliest_pos = len(text)
    for token in stop_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos]

def gsm8k_test(model_path, data_path, start=0, end=MAX_INT, batch_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt template:', problem_prompt)

    if os.path.isfile(f"experiment/{model_path.split('/')[-2]}_gsm8k.txt"):
        exit()

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length of inputs:', len(gsm8k_ins))

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    model.to("cuda")
    model.eval()

    res_completions = []
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

    total_batches = len(batch_gsm8k_ins)
    start_time = time.time()

    for batch_idx, batch_prompts in enumerate(batch_gsm8k_ins):
        # Tokenize and move inputs to device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, output in enumerate(outputs):
            gen_text = tokenizer.decode(output, skip_special_tokens=True)
            #gen_text = truncate_at_stop_tokens(gen_text, stop_tokens)
            res_completions.append(gen_text)

        # Added: Print progress and estimated finish time
        elapsed = time.time() - start_time
        batches_left = total_batches - (batch_idx + 1)
        est_remaining = (elapsed / (batch_idx + 1)) * batches_left
        est_finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remaining))
        print(f"[Batch {batch_idx+1}/{total_batches}] Elapsed: {timedelta(seconds=int(elapsed))}, Estimated finish: {est_finish}")

    invalid_outputs = []
    result = []
    for prompt, completion, prompt_answer in zip(gsm8k_ins, res_completions, gsm8k_answers):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    correct_count = sum(result)
    acc = correct_count / len(result)
    
    # Added: Final summary print
    print(f"\nFinal results:")
    print(f"Number of correct answers: {correct_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of invalid outputs: {len(invalid_outputs)}")

    # Added: Save results to file
    save_path = f"experiment/{model_path.split('/')[-2]}_gsm8k.txt"
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Number of correct answers: {correct_count}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Number of invalid outputs: {len(invalid_outputs)}\n")
    print(f"Results saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=100)  # batch size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(
        model_path=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size
    )



def gsm8k_test_noargs(model, tokenizer, name, data_path='./dataset/GSM8K_test.jsonl', start=0, end=sys.maxsize, batch_size=100):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt template:', problem_prompt)

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length of inputs:', len(gsm8k_ins))

    model.to("cuda")
    model.eval()

    res_completions = []
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

    total_batches = len(batch_gsm8k_ins)
    start_time = time.time()

    for batch_idx, batch_prompts in enumerate(batch_gsm8k_ins):
        # Tokenize and move inputs to device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, output in enumerate(outputs):
            gen_text = tokenizer.decode(output, skip_special_tokens=True)
            #gen_text = truncate_at_stop_tokens(gen_text, stop_tokens)
            res_completions.append(gen_text)

        # Added: Print progress and estimated finish time
        elapsed = time.time() - start_time
        batches_left = total_batches - (batch_idx + 1)
        est_remaining = (elapsed / (batch_idx + 1)) * batches_left
        est_finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remaining))
        print(f"[Batch {batch_idx+1}/{total_batches}] Elapsed: {timedelta(seconds=int(elapsed))}, Estimated finish: {est_finish}")

    invalid_outputs = []
    result = []
    for prompt, completion, prompt_answer in zip(gsm8k_ins, res_completions, gsm8k_answers):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    correct_count = sum(result)
    acc = correct_count / len(result)
    
    # Added: Final summary print
    print(f"\nFinal results:")
    print(f"Number of correct answers: {correct_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of invalid outputs: {len(invalid_outputs)}")

    # Added: Save results to file
    save_path = f"experiment/{name}_gsm8k.txt"
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Number of correct answers: {correct_count}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Number of invalid outputs: {len(invalid_outputs)}\n")
    print(f"Results saved to {save_path}")