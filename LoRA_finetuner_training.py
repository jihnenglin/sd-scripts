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
output_dir = os.path.join(root_dir, "LoRA/output")

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

input("Press the Enter key to continue: ")


## Dataset Config
dataset_repeats = 1
in_json = os.path.join(root_dir, "LoRA/meta_lat.json")
resolution = "768,768" # ["512,512", "768,768"]
# keep heading N tokens when shuffling caption tokens (token means comma separated strings)
keep_tokens = 0
color_aug = True
flip_aug = True

input("Press the Enter key to continue: ")


## LoRA and Optimizer Config

### LoRA Config:
network_category = "LoRA"  # ["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]

# Recommended values:

# | network_category | network_dim | network_alpha | conv_dim | conv_alpha |
# | :---: | :---: | :---: | :---: | :---: |
# | LoRA | 32 | 1 | - | - |
# | LoCon | 16 | 8 | 8 | 1 |
# | LoHa | 8 | 4 | 4 | 1 |

# - Note that `dropout` and `cp_decomposition` are not available in this notebook.

# `conv_dim` and `conv_alpha` are needed to train `LoCon` and `LoHa`; skip them if you are training normal `LoRA`. However, when in doubt, set `dim = alpha`.
conv_dim = 32
conv_alpha = 16
# It's recommended not to set `network_dim` and `network_alpha` higher than 64, especially for `LoHa`.
# If you want to use a higher value for `dim` or `alpha`, consider using a higher learning rate, as models with higher dimensions tend to learn faster.
network_dim = 32
network_alpha = 16
# You can specify this field for resume training.
network_weight = ""
network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"
network_args = "" if network_category == "LoRA" else [
    f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
    ]
# See here (https://github.com/kohya-ss/sd-scripts/pull/545?ref=blog.hinablue.me)
network_dropout = 0
scale_weight_norms = -1  # -1 to disable
### Optimizer Config:
# `NEW` Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends 5. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma = 5  # -1 to disable
# `AdamW8bit` was the old `--use_8bit_adam`.
optimizer_type = "AdamW8bit"  # ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
# Additional arguments for optimizer, e.g: `["decouple=True","weight_decay=0.6"]`
optimizer_args = ""
# Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer, because it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm.
# However, it is recommended to set `text_encoder_lr = 0.5 * unet_lr`.
# Also, you don't need to specify `learning_rate` value if both `unet_lr` and `text_encoder_lr` are defined.
train_unet = True
unet_lr = 1e-4
train_text_encoder = True
text_encoder_lr = 5e-5
lr_scheduler = "constant"  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
lr_warmup_steps = 0
# You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
lr_scheduler_num_cycles = 0
lr_scheduler_power = 0

if network_category == "LoHa":
  network_args.append("algo=loha")
elif network_category == "LoCon_Lycoris":
  network_args.append("algo=lora")

print("- LoRA Config:")
print(f"  - Min-SNR Weighting: {min_snr_gamma}") if not min_snr_gamma == -1 else ""
print(f"  - Loading network module: {network_module}")
if not network_category == "LoRA":
  print(f"  - network args: {network_args}")
print(f"  - {network_module} linear_dim set to: {network_dim}")
print(f"  - {network_module} linear_alpha set to: {network_alpha}")
if not network_category == "LoRA":
  print(f"  - {network_module} conv_dim set to: {conv_dim}")
  print(f"  - {network_module} conv_alpha set to: {conv_alpha}")

if not network_weight:
    print("  - No LoRA weight loaded.")
else:
    if os.path.exists(network_weight):
        print(f"  - Loading LoRA weight: {network_weight}")
    else:
        print(f"  - {network_weight} does not exist.")
        network_weight = ""

print(f"  - network_dropout: {network_dropout}")
print(f"  - scale_weight_norms: {scale_weight_norms}") if not scale_weight_norms == -1 else ""

print("- Optimizer Config:")
print(f"  - Additional network category: {network_category}")
print(f"  - Using {optimizer_type} as Optimizer")
if optimizer_args:
    print(f"  - Optimizer Args: {optimizer_args}")
if train_unet and train_text_encoder:
    print("  - Train UNet and Text Encoder")
    print(f"    - UNet learning rate: {unet_lr}")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
if train_unet and not train_text_encoder:
    print("  - Train UNet only")
    print(f"    - UNet learning rate: {unet_lr}")
if train_text_encoder and not train_unet:
    print("  - Train Text Encoder only")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
print(f"  - Learning rate Scheduler: {lr_scheduler}")
if lr_scheduler == "cosine_with_restarts":
    print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
elif lr_scheduler == "polynomial":
    print(f"  - lr_scheduler_power: {lr_scheduler_power}")

input("Press the Enter key to continue: ")


## Training Config
import toml

lowram = False
enable_sample_prompt = True
sampler = "ddim"  # ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
noise_offset = 0.1
num_epochs = 10
train_batch_size = 8
mixed_precision = "fp16"  # ["no","fp16","bf16"]
save_precision = "fp16"  # ["float", "fp16", "bf16"]
save_n_epochs_type = "save_every_n_epochs"  # ["save_every_n_epochs", "save_n_epoch_ratio"]
save_n_epochs_type_value = 5
save_model_as = "safetensors"  # ["ckpt", "pt", "safetensors"]
max_token_length = 225
clip_skip = 2
gradient_checkpointing = False
gradient_accumulation_steps = 8
seed = 1450  # @param {type:"number"}
logging_dir = os.path.join(root_dir, "LoRA/logs")
prior_loss_weight = 1.0

repo_dir = os.path.join(root_dir, "sd-scripts")
os.chdir(repo_dir)

train_data_dir = os.path.join(root_dir, "LoRA/train_data")

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
        "vae": vae,
    },
    "additional_network_arguments": {
        "no_metadata": False,
        "unet_lr": float(unet_lr) if train_unet else None,
        "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
        "network_weights": network_weight,
        "network_module": network_module,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_args": network_args,
        "network_train_unet_only": True if train_unet and not train_text_encoder else False,
        "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
        "training_comment": None,
        "network_dropout": network_dropout,
        "scale_weight_norms": scale_weight_norms if not scale_weight_norms == -1 else None,
    },
    "optimizer_arguments": {
        "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
        "optimizer_type": optimizer_type,
        "learning_rate": unet_lr,
        "max_grad_norm": 1.0,
        "optimizer_args": eval(optimizer_args) if optimizer_args else None,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
    },
    "dataset_arguments": {
        "debug_dataset": False,
        "in_json": in_json,
        "train_data_dir": train_data_dir,
        "dataset_repeats": dataset_repeats,
        "shuffle_caption": True,
        "keep_tokens": keep_tokens,
        "resolution": resolution,
        "caption_dropout_rate": 0,
        "caption_tag_dropout_rate": 0,
        "caption_dropout_every_n_epochs": 0,
        "color_aug": color_aug,
        "flip_aug": flip_aug,
        "face_crop_aug_range": None,
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    },
    "training_arguments": {
        "output_dir": output_dir,
        "output_name": project_name,
        "save_precision": save_precision,
        "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
        "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs": None,
        "save_state": None,
        "save_last_n_epochs_state": None,
        "resume": None,
        "train_batch_size": train_batch_size,
        "max_token_length": 225,
        "mem_eff_attn": False,
        "xformers": True,
        "max_train_epochs": num_epochs,
        "max_data_loader_n_workers": 8,
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
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps": None,
        "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
        "sample_sampler": sampler,
    },
    "saving_arguments": {
        "save_model_as": save_model_as
    },
}

config_dir = os.path.join(root_dir, "LoRA/config")
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
# - `sample_prompt` : /content/LoRA/config/sample_prompt.txt
# - `config_file` : /content/LoRA/config/config_file.toml

# Generated sample can be seen here: /content/LoRA/output/sample

# You can import config from another session if you want.
import subprocess

sample_prompt = os.path.join(root_dir, "LoRA/config/sample_prompt.txt")
config_file = os.path.join(root_dir, "LoRA/config/config_file.toml")
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")

accelerate_conf = {
    "config_file" : accelerate_config,
    "num_cpu_threads_per_process" : 1,
}

train_conf = {
    "sample_prompts" : sample_prompt,
    "config_file" : config_file
}

def train(config):
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

accelerate_args = train(accelerate_conf)
train_args = train(train_conf)
final_args = f"accelerate launch {accelerate_args} train_network.py {train_args}"

os.chdir(repo_dir)
subprocess.run(f"{final_args}", shell=True)
