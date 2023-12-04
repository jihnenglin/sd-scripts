import os
import torch

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification

import library.model_util as model_util
import library.train_util as train_util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "~/sd-train"
train_data_dir_path = Path(os.path.join(root_dir, "scraped_data"))
recursive = False
batch_size = 32
max_data_loader_n_workers = 32

image_paths: list[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, recursive)]
print(f"found {len(image_paths)} images.")

dataset = train_util.ImageLoadingDataset(image_paths)
data = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_data_loader_n_workers,
    drop_last=False,
)

model = AutoModelForImageClassification.from_pretrained("cafeai/cafe_aesthetic").to(DEVICE)

for data_entry in data:
    print(data_entry)
"""
    if data_entry[0] is None:
        continue

    img_tensor, image_path = data_entry[0]
    if img_tensor is not None:
        image = transforms.functional.to_pil_image(img_tensor)
    else:
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
            continue

    outputs = model(**batch_inputs)
"""