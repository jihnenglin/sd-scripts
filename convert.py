import os
import html
from PIL import Image, ImageSequence
import sys

# Convert single image

image_path = sys.argv[-1]
tag_path = image_path + ".txt"

def pre_process_tag(tag_path):
    old_path = tag_path
    new_path = os.path.splitext(os.path.splitext(tag_path)[0])[0] + ".txt"

    os.rename(old_path, new_path)

    with open(new_path, "r") as f:
        contents = f.read()

    contents = html.unescape(contents)
    contents = contents.replace("_", " ")
    contents = ", ".join(contents.split("\n")[:-1])

    with open(new_path, "w") as f:
        f.write(contents)

def process_image(image_path):
    img = Image.open(image_path)
    img_dir, image_name = os.path.split(image_path)

    frames = list(ImageSequence.Iterator(img))
    if len(frames) > 1:
        print(f"Deleting file {image_name} from {img_dir}")
        os.remove(image_path)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert('RGBA')
        background_color = (255, 255, 255, 255)
        bg = Image.new("RGBA", img.size, background_color)
        bg.paste(img, (0, 0), img)
        bg = bg.convert("RGB")

        if image_name.endswith(".webp") or image_name.endswith(".gif"):
            new_image_path = os.path.splitext(image_path)[0] + ".png"
            bg.save(new_image_path, "PNG")
            try:
                Image.open(new_image_path)
                os.remove(image_path)
                print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
            except Exception as e:
                print(f"Error converting {new_image_path}: {e}")
        else:
            bg.save(image_path, "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp") or image_name.endswith(".gif"):
            new_image_path = os.path.splitext(image_path)[0] + ".png"
            img.save(new_image_path, "PNG")
            try:
                Image.open(new_image_path)
                os.remove(image_path)
                print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
            except Exception as e:
                print(f"Error converting {new_image_path}: {e}")

pre_process_tag(tag_path)
process_image(image_path)
