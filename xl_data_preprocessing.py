## Data Cleaning
import os
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

root_dir = "~/sd-train"
repo_dir = os.path.join(root_dir, "sd-scripts")
finetune_dir = os.path.join(repo_dir, "finetune")
train_data_dir = os.path.join(root_dir, "train_data")
json_dir = os.path.join(root_dir, "json")
model_path = os.path.join(root_dir, "pretrained_model/sd_xl_base_1.0.safetensors")

os.chdir(root_dir)

# This section removes unsupported media types such as `.mp4`, `.webm`, and `.gif`, as well as any unnecessary files.
# To convert a transparent dataset with an alpha channel (RGBA) to RGB and give it a white background, set the `convert` parameter to `True`.
convert = True
# Alternatively, you can give the background a `random_color` instead of white by checking the corresponding option.
random_color = False
recursive = True

batch_size = 64
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


## Bucketing and Latents Caching
import time
import subprocess

# This code will create buckets based on the `bucket_resolution` provided for multi-aspect ratio training, and then convert all images within the `train_data_dir` to latents.
# If `recursive`, additionally make JSON files for every top-level folder (`dataset.subset`) in `train_data_dir`.
# If `recursive`, the additional JSON file names would be `{default_json_file_name[:-5]}_{folder_name}.json`
bucketing_json            = os.path.join(json_dir, "meta_lat.json")
metadata_json             = os.path.join(json_dir, "meta_clean.json")
batch_size                = 16
max_data_loader_n_workers = 32
bucket_resolution         = 1024  # [512, 640, 768, 896, 1024]
mixed_precision           = "no"  # ["no", "fp16", "bf16"]
flip_aug                  = False
# Use `clean_caption` option to clean such as duplicate tags, `women` to `girl`, etc
clean_caption             = False
# Use the `recursive` option to process subfolders as well
recursive                 = True
skip_existing             = True

metadata_config = {
    "_train_data_dir": train_data_dir,
    "_out_json": metadata_json,
    "recursive": recursive,
    "full_path": recursive,
    "clean_caption": clean_caption
}

bucketing_config = {
    "_train_data_dir": train_data_dir,
    "_in_json": metadata_json,
    "_out_json": bucketing_json,
    "_model_name_or_path": model_path,
    "recursive": recursive,
    "full_path": recursive,
    "flip_aug": flip_aug,
    "batch_size": batch_size,
    "max_data_loader_n_workers": max_data_loader_n_workers,
    "max_resolution": f"{bucket_resolution}, {bucket_resolution}",
    "min_bucket_reso": 512,
    "max_bucket_reso": 2048,
    "mixed_precision": mixed_precision,
    "skip_existing": skip_existing,
}

def generate_args(config):
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
    return args.strip()

def get_supported_images(folder):
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [file for ext in supported_extensions for file in glob.glob(f"{folder}/*{ext}")]

def get_subfolders_with_supported_images(folder):
    subfolders = [os.path.join(folder, subfolder) for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    return [subfolder for subfolder in subfolders if len(get_supported_images(subfolder)) > 0]

os.chdir(finetune_dir)

if not recursive:
    merge_metadata_args = generate_args(metadata_config)
    prepare_buckets_args = generate_args(bucketing_config)

    merge_metadata_command = f"python merge_all_to_metadata.py {merge_metadata_args}"
    prepare_buckets_command = f"python prepare_buckets_latents.py {prepare_buckets_args}"

    subprocess.run(f"{merge_metadata_command}", shell=True)
    time.sleep(1)
    subprocess.run(f"{prepare_buckets_command}", shell=True)

else:
    train_supported_images = get_supported_images(train_data_dir)
    train_subfolders = get_subfolders_with_supported_images(train_data_dir)

    if train_supported_images:
        folder_name = os.path.basename(train_data_dir)
        metadata_config["recursive"] = False
        bucketing_config["recursive"] = False

        merge_metadata_args = generate_args(metadata_config)
        prepare_buckets_args = generate_args(bucketing_config)

        merge_metadata_command = f"python merge_all_to_metadata.py {merge_metadata_args}"
        prepare_buckets_command = f"python prepare_buckets_latents.py {prepare_buckets_args}"

        subprocess.run(f"{merge_metadata_command}", shell=True)
        time.sleep(1)
        subprocess.run(f"{prepare_buckets_command}", shell=True)

        metadata_config["recursive"] = True
        bucketing_config["recursive"] = True

    for subfolder in train_subfolders:
        folder_name = os.path.basename(subfolder)
        metadata_config["_out_json"] = f"{metadata_json[:-5]}_{folder_name}.json"
        bucketing_config["_in_json"] = f"{metadata_json[:-5]}_{folder_name}.json"
        bucketing_config["_out_json"] = f"{bucketing_json[:-5]}_{folder_name}.json"

        merge_metadata_args = generate_args(metadata_config)
        prepare_buckets_args = generate_args(bucketing_config)

        merge_metadata_command = f"python merge_all_to_metadata.py {merge_metadata_args}"
        prepare_buckets_command = f"python prepare_buckets_latents.py {prepare_buckets_args}"

        subprocess.run(f"{merge_metadata_command}", shell=True)
        time.sleep(1)
        subprocess.run(f"{prepare_buckets_command}", shell=True)
