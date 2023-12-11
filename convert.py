import os
from PIL import Image
import sys

image_path = os.path.join("/mnt/weights/cag/sd-train/male_focus_data", sys.argv[-1])

img = Image.open(image_path)
img_dir, image_name = os.path.split(image_path)

print(img.mode)

if img.mode in ("RGBA", "LA"):
    background_color = (255, 255, 255)
    bg = Image.new("RGB", img.size, background_color)
    bg.paste(img, mask=img.split()[-1])

    bg.save(image_path, "PNG")
    print(f" Converted image: {image_name}")
