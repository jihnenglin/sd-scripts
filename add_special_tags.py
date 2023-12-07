from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
import os
import json

import library.train_util as train_util

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server

root_dir = "~/sd-train"
train_data_dir_path = Path(os.path.join(root_dir, "scraped_data"))
recursive = False
batch_size = 8
max_data_loader_n_workers = 32

quality_thresholds = [150, 100, 75, 0, -4]
quality_tag_names = ["best quality", "amazing quality", "great quality", "normal quality", "bad quality", "worst quality"]
aesthetic_thresholds = [6.675, 6.0, 5.0]
aesthetic_tag_names = ["best aesthetic", "great aesthetic", "normal aesthetic", "bad aesthetic"]
assert len(quality_tag_names) == len(quality_thresholds) + 1, \
       "The number of `quality_tag_names` should be equal to the number of `quality_thresholds` plus one"
assert len(aesthetic_tag_names) == len(aesthetic_thresholds) + 1, \
       "The number of `aesthetic_tag_names` should be equal to the number of `aesthetic_thresholds` plus one"

device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64

class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path)
            image = preprocess(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None
        return (image, img_path)

image_paths: list[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, recursive)]
print(f"found {len(image_paths)} images.")

dataset = ImageLoadingDataset(image_paths)
data = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_data_loader_n_workers,
    drop_last=False,
)

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def get_tag_name(score, thresholds, tag_names):
    not_last = False
    for j in range(len(thresholds)):
        if score >= thresholds[j]:
            not_last = True
            break
    if not_last:
        tag = tag_names[j]
    else:
        tag = tag_names[-1]
    return tag

model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()

with torch.no_grad():
    for data_entry in tqdm(data, smoothing=0.0):
        images, img_paths = data_entry
        images = images.to(device)
        image_features = model2.encode_image(images)

        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        scores = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

        for i in range(batch_size):
            aesthetic_tag = get_tag_name(scores[i], aesthetic_thresholds, aesthetic_tag_names)

            with open(f"{img_paths[i]}.json", "r") as f:
                metadata = json.load(f)
                quality_tag = get_tag_name(metadata["score"], quality_thresholds, quality_tag_names)
                print(scores[i], img_paths[i], f"{quality_tag}, {aesthetic_tag}, year {metadata['created_at'][:4]}, ")

        input("Press enter to continue")