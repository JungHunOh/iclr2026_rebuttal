import argparse
import json
import pdb
import jsonlines
import torch

import util
import sys
from tqdm import tqdm
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
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

def test_hendrycks_math(model, tokenizer, output_dir, start=0, end=MAX_INT, batch_size=200):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open("dataset/MATH_test.jsonl", "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    #sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    #llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)

    model.to("cuda")
    model.eval()

    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(tqdm(zip(batch_hendrycks_math_ins, hendrycks_math_answers), total=len(batch_hendrycks_math_ins), desc="Generating completions")):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            completions = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in completions:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)

    save_path = f"experiment/{output_dir}_math.txt"
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f'len invalid outputs ===={len(invalid_outputs)}')
        f_out.write(f'length===={len(results)}, acc===={acc}')
    print(f"Results saved to {save_path}")
