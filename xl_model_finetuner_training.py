## Dataset Config
import os
import toml
import glob

root_dir = "~/sd-train"
repo_dir = os.path.join(root_dir, "sd-scripts")
train_data_dir = os.path.join(root_dir, "train_data")
json_dir = os.path.join(root_dir, "json")
config_dir = os.path.join(root_dir, "fine_tune/config")
output_dir = os.path.join(root_dir, "fine_tune/output")
model_path = os.path.join(root_dir, "pretrained_model/sd_xl_base_1.0.safetensors")
vae_path = None


# This configuration is designed for `one concept` training. Refer to this [guide](https://rentry.org/kohyaminiguide#b-multi-concept-training) for multi-concept training.
dataset_repeats = 1
# If `recursive`, additionally make JSON files for every top-level folder (`dataset.subset`) in `train_data_dir`.
# If `recursive`, the additional JSON file names would be `{default_json_file_name[:-5]}_{folder_name}.json`
in_json = os.path.join(json_dir, "meta_lat.json")
resolution = 1024  # [512, 640, 768, 896, 1024]
# keep heading N tokens when shuffling caption tokens (token means comma separated strings)
keep_tokens = 0
color_aug = False
flip_aug = False

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

train_supported_images = get_supported_images(train_data_dir)
train_subfolders = get_subfolders_with_supported_images(train_data_dir)

subsets = []

dataset_config = {
    "general": {
        "resolution": resolution,
        "enable_bucket": True,
        "min_bucket_reso": 512,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "keep_tokens": keep_tokens,
            "color_aug": color_aug,
            "flip_aug": flip_aug,
            "face_crop_aug_range": None,
            "caption_dropout_rate": 0,
            "caption_dropout_every_n_epochs": 0,
            "caption_tag_dropout_rate": 0,
            "subsets": subsets,
        }
    ],
}

if train_supported_images:
    subsets.append({
        "image_dir": train_data_dir,
        "num_repeats": dataset_repeats,
        "metadata_file": in_json,
    })

for subfolder in train_subfolders:
    folder_name, num_repeats = get_folder_name_and_num_repeats(subfolder)
    subsets.append({
        "image_dir": subfolder,
        "num_repeats": num_repeats,
        "metadata_file": f"{in_json[:-5]}_{folder_name}.json",
    })

dataset_config_file = os.path.join(config_dir, "dataset_config.toml")

for key in dataset_config:
    if isinstance(dataset_config[key], dict):
        for sub_key in dataset_config[key]:
            if dataset_config[key][sub_key] == "":
                dataset_config[key][sub_key] = None
    elif dataset_config[key] == "":
        dataset_config[key] = None

config_str = toml.dumps(dataset_config)

with open(dataset_config_file, "w") as f:
    f.write(config_str)

print(config_str)

input("Press the Enter key to continue: ")


## Optimizer Config
import ast

# Use `Adafactor` optimizer. `RMSprop 8bit` or `Adagrad 8bit` may work. `AdamW 8bit` doesn't seem to work.
optimizer_type = "AdaFactor"  # ["AdamW", "AdamW8bit", "Lion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation(DAdaptAdamPreprint)", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor"]
# Specify `optimizer_args` to add `additional` args for optimizer, e.g: `["weight_decay=0.6"]`
optimizer_args = "[ \"scale_parameter=False\", \"relative_step=False\", \"warmup_init=False\" ]"
### Learning Rate Config
# Different `optimizer_type` and `network_category` for some condition requires different learning rate. It's recommended to set `text_encoder_lr = 1/2 * unet_lr`
learning_rate = 2e-6
max_grad_norm = 0.0  # default = 1.0; 0.0 for no clipping. It is recommended to be set to 0.0 when using AdaFactor with fixed learning rate
train_text_encoder = False
# ViT-L
learning_rate_te1 = 2e-6
# BiG-G
learning_rate_te2 = 2e-6
### LR Scheduler Config
# `lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs.
lr_scheduler = "constant_with_warmup"  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
lr_warmup_steps = 100
# Specify `lr_scheduler_num` with `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial`
lr_scheduler_num = 0

if isinstance(optimizer_args, str):
    optimizer_args = optimizer_args.strip()
    if optimizer_args.startswith('[') and optimizer_args.endswith(']'):
        try:
            optimizer_args = ast.literal_eval(optimizer_args)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing optimizer_args: {e}\n")
            optimizer_args = []
    elif len(optimizer_args) > 0:
        print(f"WARNING! '{optimizer_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
        optimizer_args = []
    else:
        optimizer_args = []
else:
    optimizer_args = []

optimizer_config = {
    "optimizer_arguments": {
        "optimizer_type"          : optimizer_type,
        "learning_rate"           : learning_rate,
        "train_text_encoder"      : train_text_encoder,
        "learning_rate_te1"       : learning_rate_te1,
        "learning_rate_te2"       : learning_rate_te2,
        "max_grad_norm"           : max_grad_norm,
        "optimizer_args"          : optimizer_args,
        "lr_scheduler"            : lr_scheduler,
        "lr_warmup_steps"         : lr_warmup_steps,
        "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
        "lr_scheduler_type"       : None,
        "lr_scheduler_args"       : None,
    },
}

print(toml.dumps(optimizer_config))

input("Press the Enter key to continue: ")


## Advanced Training Config

### Resume With Optimizer State
optimizer_state_path      = ""
### Noise Control
noise_control_type        = "multires_noise" #@param ["none", "noise_offset", "multires_noise"]
#### a. Noise Offset
# Control and easily generating darker or light images by offset the noise when fine-tuning the model. Recommended value: `0.1`. Read [Diffusion With Offset Noise](https://www.crosslabs.org//blog/diffusion-with-offset-noise)
noise_offset_num          = 0.1
# [Experimental]
# Automatically adjusts the noise offset based on the absolute mean values of each channel in the latents when used with `--noise_offset`. Specify a value around 1/10 to the same magnitude as the `--noise_offset` for best results. Set `0` to disable.
adaptive_noise_scale      = 0.01
#### b. Multires Noise
# enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)
multires_noise_iterations = 10
multires_noise_discount   = 0.3
### Custom Train Function
# Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends `5`. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma             = 5

advanced_training_config = {
    "advanced_training_config": {
        "resume"                    : optimizer_state_path,
        "noise_offset"              : noise_offset_num if noise_control_type == "noise_offset" else None,
        "adaptive_noise_scale"      : adaptive_noise_scale if adaptive_noise_scale and noise_control_type == "noise_offset" else None,
        "multires_noise_iterations" : multires_noise_iterations if noise_control_type =="multires_noise" else None,
        "multires_noise_discount"   : multires_noise_discount if noise_control_type =="multires_noise" else None,
        "min_snr_gamma"             : min_snr_gamma if not min_snr_gamma == -1 else None,
    }
}

print(toml.dumps(advanced_training_config))

input("Press the Enter key to continue: ")


## Training Config
import random

json_dir = os.path.join(root_dir, "json")

### Project Config
project_name            = "sdxl_finetune"
# Get your `wandb_api_key` [here](https://wandb.ai/settings) to logs with wandb.
wandb_api_key           = ""
in_json                 = os.path.join(json_dir, "meta_lat.json")
### SDXL Config
gradient_checkpointing  = True
no_half_vae             = True
# Recommended parameter for SDXL training but if you enable it, `shuffle_caption` won't work
cache_text_encoder_outputs = True
# These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.
min_timestep = 0
max_timestep = 1000
resolution                  = 1024  # [512, 640, 768, 896, 1024]
### General Config
max_train_n_type            = "max_train_epochs"  # ["max_train_steps", "max_train_epochs"]
max_train_n_type_value      = 10
train_batch_size            = 4
max_data_loader_n_workers   = 32
gradient_accumulation_steps = 8
mixed_precision             = "bf16"  # ["no","fp16","bf16"]
seed                        = -1
### Save Output Config
save_precision              = "fp16"  # ["float", "fp16", "bf16"]
save_n_type                 = "save_every_n_epochs"  # ["save_every_n_epochs", "save_every_n_steps", "save_n_epoch_ratio"]
save_n_type_value           = 1
save_optimizer_state        = False
save_model_as               = "safetensors" # ["ckpt", "safetensors", "diffusers", "diffusers_safetensors"]
### Sample Prompt Config
enable_sample               = True
sampler                     = "euler_a"  # ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
positive_prompt             = "1boy, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, stud earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt, best quality, amazing quality, very aesthetic, absurdres"
negative_prompt             = "lowres, (bad), text, error, missing, extra, fewer, cropped, jpeg artifacts, worst quality, bad quality, watermark, bad aesthetic, unfinished, chromatic aberration, scan, scan artifacts, "
# Specify `prompt_from_caption` if you want to use caption as prompt instead. Will be chosen randomly.
prompt_from_caption         = "none"  # ["none", ".txt", ".caption"]
if prompt_from_caption != "none":
    custom_prompt           = ""
num_prompt                  = 2
sample_interval             = 100
logging_dir                 = os.path.join(root_dir, "fine_tune/logs")

os.chdir(repo_dir)

prompt_config = {
    "prompt": {
        "negative_prompt" : negative_prompt,
        "width"           : resolution,
        "height"          : resolution,
        "scale"           : 7,
        "sample_steps"    : 28,
        "subset"          : [],
    }
}

train_config = {
    "sdxl_arguments": {
        "cache_text_encoder_outputs" : cache_text_encoder_outputs,
        "enable_bucket"              : True,
        "no_half_vae"                : no_half_vae,
        "cache_latents"              : True if not color_aug else False,
        "cache_latents_to_disk"      : True if not color_aug else False,
        "min_timestep"               : min_timestep,
        "max_timestep"               : max_timestep,
        "shuffle_caption"            : True if not cache_text_encoder_outputs else False,
    },
    "model_arguments": {
        "pretrained_model_name_or_path" : model_path,
        "vae"                           : vae_path,
    },
    "dataset_arguments": {
        "debug_dataset"                 : False,
    },
    "training_arguments": {
        "output_dir"                    : output_dir,
        "output_name"                   : project_name if project_name else "last",
        "save_precision"                : save_precision,
        "save_every_n_epochs"           : save_n_type_value if save_n_type == "save_every_n_epochs" else None,
        "save_every_n_steps"            : save_n_type_value if save_n_type == "save_every_n_steps" else None,
        "save_n_epoch_ratio"            : save_n_type_value if save_n_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs"            : None,
        "save_state"                    : save_optimizer_state,
        "save_last_n_epochs_state"      : None,
        "train_batch_size"              : train_batch_size,
        "max_token_length"              : 225,
        "mem_eff_attn"                  : False,
        "xformers"                      : True,
        "max_train_steps"               : max_train_n_type_value if max_train_n_type == "max_train_steps" else None,
        "max_train_epochs"              : max_train_n_type_value if max_train_n_type == "max_train_epochs" else None,
        "max_data_loader_n_workers"     : max_data_loader_n_workers,
        "persistent_data_loader_workers": True,
        "seed"                          : seed if seed > 0 else None,
        "gradient_checkpointing"        : gradient_checkpointing,
        "gradient_accumulation_steps"   : gradient_accumulation_steps,
        "mixed_precision"               : mixed_precision,
    },
    "logging_arguments": {
        "log_with"          : "wandb" if wandb_api_key else "tensorboard",
        "log_tracker_name"  : project_name if wandb_api_key and not project_name == "last" else None,
        "logging_dir"       : logging_dir,
        "log_prefix"        : project_name if not wandb_api_key else None,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps"    : sample_interval,
        "sample_every_n_epochs"   : None,
        "sample_sampler"          : sampler,
    },
    "saving_arguments": {
        "save_model_as": "safetensors"
    },
}

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt):
    if enable_sample:
        search_pattern = os.path.join(train_data_dir, '**/*' + prompt_from_caption)
        caption_files = glob.glob(search_pattern, recursive=True)

        if not caption_files:
            if not custom_prompt:
                custom_prompt = "1boy, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, stud earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt, best quality, amazing quality, very aesthetic, absurdres"
            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = [
                {"prompt": positive_prompt + custom_prompt if positive_prompt else custom_prompt}
            ]
        else:
            selected_files = random.sample(caption_files, min(num_prompt, len(caption_files)))

            prompts = []
            for file in selected_files:
                with open(file, 'r') as f:
                    prompts.append(f.read().strip())

            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = []

            for prompt in prompts:
                new_prompt = {
                    "prompt": positive_prompt + prompt if positive_prompt else prompt,
                }
                new_prompt_config['prompt']['subset'].append(new_prompt)

        return new_prompt_config
    else:
        return prompt_config

def eliminate_none_variable(config):
    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    return config

try:
    train_config.update(optimizer_config)
except NameError:
    raise NameError("'optimizer_config' dictionary is missing. Please run  'Optimizer Config' cell.")

advanced_training_warning = False
try:
    train_config.update(advanced_training_config)
except NameError:
    advanced_training_warning = True
    pass

prompt_config       = prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt)

config_path         = os.path.join(config_dir, "config_file.toml")
prompt_path         = os.path.join(config_dir, "sample_prompt.toml")

config_str          = toml.dumps(eliminate_none_variable(train_config))
prompt_str          = toml.dumps(eliminate_none_variable(prompt_config))

write_file(config_path, config_str)
write_file(prompt_path, prompt_str)

print(config_str)

if advanced_training_warning:
    import textwrap
    error_message = "WARNING: This is not an error message, but the [advanced_training_config] dictionary is missing. Please run the 'Advanced Training Config' cell if you intend to use it, or continue to the next step."
    wrapped_message = textwrap.fill(error_message, width=80)
    print('\033[38;2;204;102;102m' + wrapped_message + '\033[0m\n')
    pass

print(prompt_str)

input("Press the Enter key to continue: ")


## Start Training
import subprocess

# Check your config here if you want to edit something:
# - `sample_prompt` : ~/sd-train/fine_tune/config/sample_prompt.toml
# - `dataset_config` : ~/sd-train/fine_tune/config/dataset_config.toml
# - `config_file` : ~/sd-train/fine_tune/config/config_file.toml

# You can import config from another session if you want.

sample_prompt       = os.path.join(root_dir, "fine_tune/config/sample_prompt.toml")
dataset_config_file = os.path.join(root_dir, "fine_tune/config/dataset_config.toml")
config_file         = os.path.join(root_dir, "fine_tune/config/config_file.toml")

accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
num_cpu_threads_per_process = 8

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

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

accelerate_conf = {
    "config_file" : accelerate_config,
    "num_cpu_threads_per_process" : num_cpu_threads_per_process,
}

train_conf = {
    "sample_prompts"  : sample_prompt if os.path.exists(sample_prompt) else None,
    "dataset_config"  : dataset_config_file,
    "config_file"     : config_file,
    "wandb_api_key"   : wandb_api_key if wandb_api_key else None,
}

accelerate_args = train(accelerate_conf)
train_args = train(train_conf)

final_args = f"accelerate launch {accelerate_args} sdxl_train.py {train_args}"

os.chdir(repo_dir)
subprocess.run(f"{final_args}", shell=True)
