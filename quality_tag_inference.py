import os

from pathlib import Path
from datasets import load_dataset
from transformers import pipeline

import library.model_util as model_util
import library.train_util as train_util

root_dir = "~/sd-train"
train_data_dir_path = Path(os.path.join(root_dir, "scraped_data"))
recursive = False

image_paths: list[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, recursive)]
print(f"found {len(image_paths)} images.")

#dataset = load_dataset("imagefolder", data_dir=scraped_data_dir)

#pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic", device=0)
