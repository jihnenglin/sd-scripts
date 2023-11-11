import os
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

root_dir = "~/sd-train"
train_data_dir = os.path.join(root_dir, "LoRA/train_data")

os.chdir(root_dir)

test = os.listdir(train_data_dir)
# This section will delete unnecessary files and unsupported media such as `.mp4`, `.webm`, and `.gif`.
# Set the `convert` parameter to convert your transparent dataset with an alpha channel (RGBA) to RGB and give it a white background.
convert = True
# You can choose to give it a `random_color` background instead of white by checking the corresponding option.
random_color = False
# Use the `recursive` option to preprocess subfolders as well.
recursive = False


batch_size = 32
supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".npz",
    ".txt",
    ".json",
]

background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def clean_directory(directory):
    for item in os.listdir(directory):
        file_path = os.path.join(directory, item)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(item)[1]
            if file_ext not in supported_types:
                print(f"Deleting file {item} from {directory}")
                os.remove(file_path)
        elif os.path.isdir(file_path) and recursive:
            clean_directory(file_path)

def process_image(image_path):
    img = Image.open(image_path)
    img_dir, image_name = os.path.split(image_path)

    if img.mode in ("RGBA", "LA"):
        if random_color:
            background_color = random.choice(background_colors)
        else:
            background_color = (255, 255, 255)
        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[-1])

        if image_name.endswith(".webp"):
            bg = bg.convert("RGB")
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            bg.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            bg.save(image_path, "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp"):
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            img.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            img.save(image_path, "PNG")

def find_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".webp"):
                images.append(os.path.join(root, file))
    return images

clean_directory(train_data_dir)
images = find_images(train_data_dir)
num_batches = len(images) // batch_size + 1

if convert:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size
            batch = images[start:end]
            executor.map(process_image, batch)

    print("All images have been converted")

input("Press the Enter key to continue: ")


import subprocess

repo_dir = os.path.join(root_dir, "sd-scripts")
finetune_dir = os.path.join(repo_dir, "finetune")
os.chdir(finetune_dir)

# Merge tags and/or captions exist in `train_data_dir` into one metadata JSON file, which will be used as the input for the bucketing section.
metadata = os.path.join(root_dir, "LoRA/meta_clean.json")
# Use `recursive` option to process subfolders as well
recursive = False
# Use `clean_caption` option to clean such as duplicate tags, `women` to `girl`, etc
clean_caption = False

config = {
    "_train_data_dir": train_data_dir,
    "_out_json": metadata,
    "recursive": recursive,
    "full_path": recursive,
    "clean_caption": clean_caption
}

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

os.chdir(finetune_dir)
final_args = f"python merge_all_to_metadata.py {args}"
subprocess.run(f"{final_args}")

input("Press the Enter key to continue: ")


# This code will create buckets based on the `max_resolution` provided for multi-aspect ratio training, and then convert all images within the `train_data_dir` to latents.
v2 = False  # @param{type:"boolean"}
model_dir = os.path.join(root_dir, "pretrained_model/AnyLoRA_noVae_fp16-pruned.safetensors")
input_json = os.path.join(root_dir, "LoRA/meta_clean.json")
output_json = os.path.join(root_dir, "LoRA/meta_lat.json")
batch_size = 16
max_data_loader_n_workers = 4
max_resolution = "768,768"  # ["512,512", "640,640", "768,768"] {allow-input: false}
mixed_precision = "no"  # ["no", "fp16", "bf16"] {allow-input: false}
flip_aug = True
# Use the `recursive` option to process subfolders as well
recursive = False

config = {
    "_train_data_dir": train_data_dir,
    "_in_json": input_json,
    "_out_json": output_json,
    "_model_name_or_path": model_dir,
    "recursive": recursive,
    "full_path": recursive,
    "v2": v2,
    "flip_aug": flip_aug,
    "min_bucket_reso": 320 if max_resolution != "512,512" else 256,
    "max_bucket_reso": 1280 if max_resolution != "512,512" else 1024,
    "batch_size": batch_size,
    "max_data_loader_n_workers": max_data_loader_n_workers,
    "max_resolution": max_resolution,
    "mixed_precision": mixed_precision,
    "skip_existing": True
}

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

os.chdir(finetune_dir)
final_args = f"python prepare_buckets_latents.py {args}"
subprocess.run(f"{final_args}")
