import re
import json
import os

def clean_generated_code(text: str) -> str:
    # 1. Remove markdown fences
    text = text.replace("```", "")

    # 2. Normalize tabs → spaces
    text = text.expandtabs(4)

    # 3. Remove triple-quoted docstrings/multiline strings (optional)
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)

    # 4. Process line by line: remove comments & empty lines
    lines = []
    for ll in text.splitlines():
        # strip inline comments
        code_only = re.sub(r'#.*', '', ll).rstrip()
        if code_only.strip():  # keep only non-empty lines
            lines.append(code_only)

    # 5. Detect indentation for normalization
    spaces_for_each_line = []
    for line in lines:
        m = re.match(r'^( *)', line)
        spaces_for_each_line.append(len(m.group(1)) if m else 0)

    try:
        def_line = next(
            i for i, line in enumerate(lines)
            if re.match(r'^\s*def\s+\w+', line) or re.match(r'^\s*class\s+\w+', line)
        )
        def_line_space = spaces_for_each_line[def_line]
    except StopIteration:
        def_line_space = 0  # fallback if no def/class found

    # 6. Normalize indentation to multiples of 4
    rank_unique_spaces = sorted(set(spaces_for_each_line))
    indentation_level = {}
    level = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            level += 1
            indentation_level[space] = level

    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_indent = "    " * indentation_level.get(space, 0)
        new_lines.append(new_indent + line.lstrip())

    return "\n".join(new_lines)

def post_process_and_eval(path):
    os.system(f"python process_jsonl.py {path}")
    path = path.replace(".jsonl", "_modified.jsonl")
    output_path = path.replace(".jsonl", "_cleaned.jsonl")

    with open(path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            if "completion" in obj:
                obj["completion"] = clean_generated_code(obj["completion"])
            fout.write(json.dumps(obj) + "\n")

    os.system(f"python eval_human.py {output_path}")
