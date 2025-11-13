import json

def process_jsonl_file(input_file):
    """
    Process JSONL file to remove the last part of completion texts after splitting by '\n'
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(input_file.replace('.jsonl','_modified.jsonl'), 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line.strip())
                
                # Split the completion by '\n' and remove the last part
                completion_parts = data['completion'].split('\n')
                if len(completion_parts) > 1:
                    # Remove the last part and join back with '\n'
                    data['completion'] = '\n'.join(completion_parts[:-1])
                
                # Write the modified data to output file
                outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    import fire
    fire.Fire(process_jsonl_file)
