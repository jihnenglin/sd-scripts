import os
from PIL import Image
import sys

# Convert single image

image_path = os.path.join("~/sd-train/train_data", sys.argv[-1])

img = Image.open(image_path)
img_dir, image_name = os.path.split(image_path)

print(img.mode)

if img.mode in ("RGBA", "LA"):
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
