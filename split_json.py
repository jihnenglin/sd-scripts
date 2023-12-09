import os
import json

NUM_OF_DEVICES = 4

json_path = "~/sd-train/json/meta_clean.json"

def split_dict(dictionary, num_chunks):
    keys = list(dictionary.keys())
    chunk_size = len(keys) // num_chunks
    last_chunk_size = len(keys) % num_chunks
    key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys) - last_chunk_size, chunk_size)]
    key_chunks[0].extend(keys[-last_chunk_size:])
    result_dicts = [{key: dictionary[key] for key in chunk} for chunk in key_chunks]
    return result_dicts

with open(json_path, "r") as f:
    metadata = json.load(f)

split_metadata = split_dict(metadata, NUM_OF_DEVICES)

for i in range(NUM_OF_DEVICES):
    with open(f"{json_path[:-5]}_{i}.json", "w") as f:
        json.dump(split_metadata[i], f, indent=2)
