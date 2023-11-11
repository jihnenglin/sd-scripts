## Interrogating LoRA Weights
# Now you can check if your LoRA trained properly.
import os
import torch
import json
from safetensors.torch import load_file
from safetensors.torch import safe_open

root_dir = "~/sd-train"

# If you used `clip_skip = 2` during training, the values of `lora_te_text_model_encoder_layers_11_*` will be `0.0`, this is normal. These layers are not trained at this value of `Clip Skip`.
network_weight = os.path.join(root_dir, "LoRA/output/last.safetensors")
verbose = False

def is_safetensors(path):
    return os.path.splitext(path)[1].lower() == ".safetensors"

def load_weight_data(file_path):
    if is_safetensors(file_path):
        return load_file(file_path)
    else:
        return torch.load(file_path, map_location="cuda")

def extract_lora_weights(weight_data):
    lora_weights = [
        (key, weight_data[key])
        for key in weight_data.keys()
        if "lora_up" in key or "lora_down" in key
    ]
    return lora_weights

def print_lora_weight_stats(lora_weights):
    print(f"Number of LoRA modules: {len(lora_weights)}")

    for key, value in lora_weights:
        value = value.to(torch.float32)
        print(f"{key}, {torch.mean(torch.abs(value))}, {torch.min(torch.abs(value))}")

def print_metadata(file_path):
    if is_safetensors(file_path):
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
        if metadata is not None:
            print(f"\nLoad metadata for: {file_path}")
            print(json.dumps(metadata, indent=4))
    else:
        print("No metadata saved, your model is not in safetensors format")

def main(file_path, verbose: bool):
    weight_data = load_weight_data(file_path)

    if verbose:
        lora_weights = extract_lora_weights(weight_data)
        print_lora_weight_stats(lora_weights)

    print_metadata(file_path)

if __name__ == "__main__":
    main(network_weight, verbose)

input("Press the Enter key to continue: ")


# Check sample pics
import threading
from imjoy_elfinder.app import main

def start_file_explorer(root_dir=root_dir, port=8765):
    try:
        main(["--root-dir=" + root_dir, "--port=" + str(port)])
    except Exception as e:
        print("Error starting file explorer:", str(e))


def open_file_explorer(root_dir=root_dir, port=8765):
    thread = threading.Thread(target=start_file_explorer, args=[root_dir, port])
    thread.start()


# Example usage
sample_dir = os.path.join(root_dir, "LoRA/output/sample")
open_file_explorer(root_dir=sample_dir, port=8765)

input("Press the Enter key to continue: ")


## Visualize loss graph
import subprocess

training_logs_path = os.path.join(root_dir, "LoRA/logs")

repo_dir = os.path.join(root_dir, "sd-scripts")
os.chdir(repo_dir)
subprocess.Popen(f"tensorboard --logdir {training_logs_path}", shell=True)

input("Press the Enter key to continue: ")
