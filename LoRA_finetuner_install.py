## Install Dependencies
import os

# root_dir
root_dir = "~/sd-train"
deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "sd-scripts")
training_dir = os.path.join(root_dir, "LoRA")
pretrained_model = os.path.join(root_dir, "pretrained_model")
vae_dir = os.path.join(root_dir, "vae")
config_dir = os.path.join(training_dir, "config")

# repo_dir
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


def install_dependencies():
    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)


os.chdir(root_dir)

for dir in [
    deps_dir,
    training_dir,
    config_dir,
    pretrained_model,
    vae_dir
]:
    os.makedirs(dir, exist_ok=True)

os.chdir(repo_dir)

install_dependencies()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SAFETENSORS_FAST_GPU"] = "1"


## Locating Train Data Directory
# Define location of your training data. This cell will also create a folder based on your input.
# This folder will serve as the target folder for scraping, tagging, bucketing, and training in the next cell.
train_data_dir = os.path.join(root_dir, "LoRA/train_data")

os.makedirs(train_data_dir, exist_ok=True)
print(f"Your train data directory : {train_data_dir}")
