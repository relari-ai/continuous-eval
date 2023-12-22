import json

def convert_json_to_jsonl(input_file, output_file, cols_rm=[]):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        data = json.load(f_in)
        for entry in data:
            f_out.write(json.dumps({k:v for k,v in entry.items() if k not in cols_rm}) + '\n')

# convert_json_to_jsonl('data/retrieval.json', 'data/retrieval.jsonl', cols_rm = ["answer"])
# convert_json_to_jsonl('data/correctness_dataset_vF.json', 'data/correctness.jsonl', cols_rm=["id_", "retrieved_contexts"])
# convert_json_to_jsonl('data/faithfulness.json', 'data/faithfulness.jsonl', cols_rm=["id_", "ground_truths"])

# convert_json_to_jsonl('data/correctness_dataset_vF.json', 'data/invalid.jsonl', cols_rm=["question"])