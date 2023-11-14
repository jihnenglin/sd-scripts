## Model Config
import os

root_dir = "~/sd-train"

v2 = False
v_parameterization = False
project_name = ""
if not project_name:
    project_name = "last"

pretrained_model_name_or_path = os.path.join(root_dir, "pretrained_model/AnyLoRA_noVae_fp16-pruned.safetensors")
vae = ""
output_dir = os.path.join(root_dir, "fine_tune/output")
# Resume training from saved state
resume_path = ""
#resume_path = os.path.join(output_dir, "last-state")

sample_dir = os.path.join(output_dir, "sample")
for dir in [output_dir, sample_dir]:
    os.makedirs(dir, exist_ok=True)

print("Project Name: ", project_name)
print("Model Version: Stable Diffusion V1.x") if not v2 else ""
print("Model Version: Stable Diffusion V2.x") if v2 and not v_parameterization else ""
print("Model Version: Stable Diffusion V2.x 768v") if v2 and v_parameterization else ""
print(
    "Pretrained Model Path: ", pretrained_model_name_or_path
) if pretrained_model_name_or_path else print("No Pretrained Model path specified.")
print("VAE Path: ", vae) if vae else print("No VAE path specified.")
print("Output Path: ", output_dir)
print("Resume Path: ", resume_path) if resume_path else print(
    "No resume path specified."
)

input("Press the Enter key to continue: ")


## Dataset Config
import toml
import glob

# This configuration is designed for `one concept` training. Refer to this [guide](https://rentry.org/kohyaminiguide#b-multi-concept-training) for multi-concept training.
dataset_repeats = 1
# If `recursive`, additionally make JSON files for every top-level folder (`dataset.subset`) in `train_data_dir`.
# If `recursive`, the additional JSON file names would be `{default_json_file_name[:-5]}_{folder_name}.json`
in_json = os.path.join(root_dir, "fine_tune/meta_lat.json")
resolution = 768  # [512, 640, 768, 896, 1024]
flip_aug = True
# keep heading N tokens when shuffling caption tokens (token means comma separated strings)
keep_tokens = 0
color_aug = True

def get_supported_images(folder):
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [file for ext in supported_extensions for file in glob.glob(f"{folder}/*{ext}")]

def get_subfolders_with_supported_images(folder):
    subfolders = [os.path.join(folder, subfolder) for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    return [subfolder for subfolder in subfolders if len(get_supported_images(subfolder)) > 0]

def get_folder_name_and_num_repeats(folder):
    folder_name = os.path.basename(folder)
    try:
        repeats, _ = folder_name.split('_', 1)
        num_repeats = int(repeats)
    except ValueError:
        num_repeats = 1

    return folder_name, num_repeats

train_data_dir = os.path.join(root_dir, "fine_tune/train_data")
train_supported_images = get_supported_images(train_data_dir)
train_subfolders = get_subfolders_with_supported_images(train_data_dir)

subsets = []

config = {
    "general": {
        "enable_bucket": True,
        "shuffle_caption": True,
        "keep_tokens": keep_tokens,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "resolution": resolution,
            "min_bucket_reso": 320 if resolution > 640 else 256,
            "max_bucket_reso": 1280 if resolution > 640 else 1024,
            "caption_dropout_rate": 0,
            "caption_tag_dropout_rate": 0,
            "caption_dropout_every_n_epochs": 0,
            "flip_aug": flip_aug,
            "color_aug": color_aug,
            "face_crop_aug_range": None,
            "subsets": subsets,
        }
    ],
}

if train_supported_images:
    subsets.append({
        "image_dir": train_data_dir,
        "metadata_file": in_json,
        "num_repeats": dataset_repeats,
    })

for subfolder in train_subfolders:
    folder_name, num_repeats = get_folder_name_and_num_repeats(subfolder)
    subsets.append({
        "image_dir": subfolder,
        "metadata_file": f"{in_json[:-5]}_{folder_name}.json",
        "num_repeats": num_repeats,
    })

config_dir = os.path.join(root_dir, "fine_tune/config")
dataset_config = os.path.join(config_dir, "dataset_config.toml")

for key in config:
    if isinstance(config[key], dict):
        for sub_key in config[key]:
            if config[key][sub_key] == "":
                config[key][sub_key] = None
    elif config[key] == "":
        config[key] = None

config_str = toml.dumps(config)

with open(dataset_config, "w") as f:
    f.write(config_str)

print(config_str)

input("Press the Enter key to continue: ")


## Optimizer Config
# `AdamW8bit` was the old `--use_8bit_adam`.
optimizer_type = "AdamW8bit"  # ["AdamW", "AdamW8bit", "PagedAdamW8bit", "PagedAdamW32bit", "Lion8bit", "PagedLion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor"]
# Set `learning_rate` to `1.0` if you use `DAdaptation` optimizer, as it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm.
# You probably need to specify `optimizer_args` for custom optimizer, like using `["decouple=true","weight_decay=0.6"]` for `DAdaptation`.
learning_rate = 2e-6
max_grad_norm = 1.0  # default = 1.0; 0 for no clipping
# Additional arguments for optimizer, e.g: `["decouple=true","weight_decay=0.6", "betas=0.9,0.999"]`
optimizer_args = ""
lr_scheduler = "constant"  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
lr_warmup_steps = 0
# You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
lr_scheduler_num_cycles = 0
lr_scheduler_power = 0
train_text_encoder = False
learning_rate_te = 1e-6

print(f"Using {optimizer_type} as Optimizer")
print("Learning rate: ", learning_rate)
print("max_grad_norm: ", max_grad_norm)
if optimizer_args:
    print(f"Optimizer Args :", optimizer_args)
print("Learning rate Scheduler:", lr_scheduler)
print("Learning rate warmup steps: ", lr_warmup_steps)
if lr_scheduler == "cosine_with_restarts":
    print("- lr_scheduler_num_cycles: ", lr_scheduler_num_cycles)
elif lr_scheduler == "polynomial":
    print("- lr_scheduler_power: ", lr_scheduler_power)
if train_text_encoder:
    print(f"Train Text Encoder")
    print("Text encoder learning rate: ", learning_rate_te)

input("Press the Enter key to continue: ")


## Training Config
save_precision = "fp16"  # [None, "float", "fp16", "bf16"] (None for not changing)
save_n_epochs_type = "save_every_n_epochs"  # ["save_every_n_epochs", "save_n_epoch_ratio"]
save_n_epochs_type_value = 5
save_state = False
train_batch_size = 8
max_token_length = 225
cross_attention = "sdpa" # [None, "mem_eff_attn", "xformers", "sdpa"]
max_train_n_type = "max_train_epochs" # ["max_train_steps", "max_train_epochs"]
max_train_n_type_value = 10
seed = 1450
gradient_checkpointing = False
gradient_accumulation_steps = 2
mixed_precision = "fp16"  # ["no","fp16","bf16"]
clip_skip = 2
logging_dir = os.path.join(root_dir, "fine_tune/logs")
noise_offset = 0.1
lowram = False
enable_sample_prompt = True
sampler = "ddim"  # ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
save_model_as = "safetensors"  # ["ckpt", "safetensors", "diffusers", "diffusers_safetensors"]
# Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends 5. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma = -1  # -1 to disable

repo_dir = os.path.join(root_dir, "sd-scripts")
os.chdir(repo_dir)

sample_str = f"""
  masterpiece, best quality, 1boy, male focus, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, stud earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt \
  --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry \
  --w 512 \
  --h 768 \
  --l 7 \
  --s 28
"""

config = {
    "model_arguments": {
        "v2": v2,
        "v_parameterization": v_parameterization if v2 and v_parameterization else False,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
    },
    "optimizer_arguments": {
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "optimizer_args": eval(optimizer_args) if optimizer_args else None,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "train_text_encoder": train_text_encoder,
        "learning_rate_te": learning_rate_te if train_text_encoder else None,
    },
    "dataset_arguments": {
        "debug_dataset": False,
    },
    "training_arguments": {
        "output_dir": output_dir,
        "output_name": project_name,
        "save_precision": save_precision,
        "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
        "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs": None,
        "save_last_n_epochs_state": None,
        "save_state": save_state,
        "resume": resume_path,
        "train_batch_size": train_batch_size,
        "max_token_length": 225,
        "mem_eff_attn": True if cross_attention == "mem_eff_attn" else False,
        "xformers": True if cross_attention == "xformers" else False,
        "sdpa": True if cross_attention == "sdpa" else False,
        "vae": vae,
        "max_train_steps": max_train_n_type_value if max_train_n_type == "max_train_steps" else None,
        "max_train_epochs": max_train_n_type_value if max_train_n_type == "max_train_epochs" else None,
        "max_data_loader_n_workers": 128,
        "persistent_data_loader_workers": True,
        "seed": seed if seed > 0 else None,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "clip_skip": clip_skip if not v2 else None,
        "logging_dir": logging_dir,
        "log_prefix": project_name,
        "noise_offset": noise_offset if noise_offset > 0 else None,
        "lowram": lowram,
        "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps": 100 if enable_sample_prompt and max_train_n_type == "max_train_steps" else 999999,
        "sample_every_n_epochs": 1 if enable_sample_prompt and max_train_n_type == "max_train_epochs" else 999999,
        "sample_sampler": sampler,
    },
    "saving_arguments": {
        "save_model_as": save_model_as
    },
}

config_path = os.path.join(config_dir, "config_file.toml")
prompt_path = os.path.join(config_dir, "sample_prompt.txt")

for key in config:
    if isinstance(config[key], dict):
        for sub_key in config[key]:
            if config[key][sub_key] == "":
                config[key][sub_key] = None
    elif config[key] == "":
        config[key] = None

config_str = toml.dumps(config)

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

write_file(config_path, config_str)
write_file(prompt_path, sample_str)

print(config_str)

input("Press the Enter key to continue: ")


## Start Training

# Check your config here if you want to edit something:
# - `sample_prompt` : ~/sd-train/fine_tune/config/sample_prompt.txt
# - `config_file` : ~/sd-train/fine_tune/config/config_file.toml
# - `dataset_config` : ~/sd-train/fine_tune/config/dataset_config.toml

# Generated sample can be seen here: ~/sd-train/fine_tune/output/sample

# You can import config from another session if you want.
import subprocess

sample_prompt = os.path.join(root_dir, "fine_tune/config/sample_prompt.txt")
config_file = os.path.join(root_dir, "fine_tune/config/config_file.toml")
dataset_config = os.path.join(root_dir, "fine_tune/config/dataset_config.toml")

accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")

accelerate_conf = {
    "config_file" : accelerate_config,
    "num_cpu_threads_per_process" : 1,
}

train_conf = {
    "sample_prompts" : sample_prompt,
    "dataset_config" : dataset_config,
    "config_file" : config_file
}

def make_args(config):
    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args

accelerate_args = make_args(accelerate_conf)
train_args = make_args(train_conf)
final_args = f"accelerate launch {accelerate_args} fine_tune.py {train_args}"

os.chdir(repo_dir)
subprocess.run(f"{final_args}", shell=True)
