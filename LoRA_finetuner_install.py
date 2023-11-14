## Install Dependencies
import os

# root_dir
root_dir = "~/sd-train"
repo_dir = os.path.join(root_dir, "sd-scripts")
training_dir = os.path.join(root_dir, "LoRA")
pretrained_model = os.path.join(root_dir, "pretrained_model")
vae_dir = os.path.join(root_dir, "vae")
train_data_dir = os.path.join(root_dir, "train_data")
json_dir = os.path.join(root_dir, "json")
config_dir = os.path.join(training_dir, "config")

# repo_dir
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
finetune_dir = os.path.join(repo_dir, "finetune")


def install_dependencies():
    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)


os.chdir(root_dir)

for dir in [
    training_dir,
    config_dir,
    pretrained_model,
    vae_dir,
    train_data_dir,
    json_dir,
]:
    os.makedirs(dir, exist_ok=True)

os.chdir(repo_dir)

install_dependencies()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
