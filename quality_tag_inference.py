import os
import torch

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

import library.model_util as model_util
import library.train_util as train_util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "~/sd-train"
train_data_dir_path = Path(os.path.join(root_dir, "scraped_data"))
recursive = False
batch_size = 32
max_data_loader_n_workers = 32
model_name = "cafeai/cafe_aesthetic"

class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = image_processor(image, return_tensors="pt")
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None
        return (torch.squeeze(inputs["pixel_values"], 0), img_path)

image_paths: list[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, recursive)]
print(f"found {len(image_paths)} images.")

image_processor = AutoImageProcessor.from_pretrained(model_name, device_map="cuda:0")
model = AutoModelForImageClassification.from_pretrained(model_name, device_map="cuda:0")

dataset = ImageLoadingDataset(image_paths)
data = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_data_loader_n_workers,
    drop_last=False,
)

with torch.no_grad():
    for data_entry in data:
        inputs, img_path = data_entry
        inputs = inputs.to(DEVICE)

        outputs = model(inputs)
        print(outputs)
        break
