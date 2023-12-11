## Scrape Dataset
import os
import html
import subprocess

root_dir = "~/sd-train"
scraped_data_dir = os.path.join(root_dir, "scraped_data")

os.chdir(root_dir)
# Use `gallery-dl` to scrape images from an imageboard site. To specify `prompt(s)`, separate them with commas (e.g., `hito_komoru, touhou`).
booru = "Danbooru"  # ["Danbooru", "Gelbooru", "Safebooru"]
prompt = "male_focus,-animated"

# Alternatively, you can provide a `custom_url` instead of using a predefined site.
custom_url = ""

# Use the `sub_folder` option to organize the downloaded images into separate folders based on their concept or category.
sub_folder = ""

user_agent = "gdl/1.24.5"

# You can limit the number of images to download by using the `--range` option followed by the desired range (e.g., `1-200`).
range = "1-200"

write_metadata = True
write_tags = True
no_skip = False

additional_arguments = "--filename /O --no-part"

tags = prompt.split(',')
tags = '+'.join(tags)

replacement_dict = {" ": "", "(": "%28", ")": "%29", ":": "%3a"}
tags = ''.join(replacement_dict.get(c, c) for c in tags)

if sub_folder == "":
    image_dir = scraped_data_dir
elif sub_folder.startswith("/content"):
    image_dir = sub_folder
else:
    image_dir = os.path.join(scraped_data_dir, sub_folder)
    os.makedirs(image_dir, exist_ok=True)

if booru == "Danbooru":
    url = "https://danbooru.donmai.us/posts?tags={}".format(tags)
    # To download a single post
    #url = "https://danbooru.donmai.us/posts/"
elif booru == "Gelbooru":
    url = "https://gelbooru.com/index.php?page=post&s=list&tags={}".format(tags)
else:
    url = "https://safebooru.org/index.php?page=post&s=list&tags={}".format(tags)

valid_url = custom_url if custom_url else url

def scrape(config):
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

def pre_process_tags(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith(".txt"):
            old_path = item_path
            new_file_name = os.path.splitext(os.path.splitext(item)[0])[0] + ".txt"
            new_path = os.path.join(directory, new_file_name)

            os.rename(old_path, new_path)

            with open(new_path, "r") as f:
                contents = f.read()

            contents = html.unescape(contents)
            contents = contents.replace("_", " ")
            contents = ", ".join(contents.split("\n"))

            with open(new_path, "w") as f:
                f.write(contents)

        elif os.path.isdir(item_path):
            pre_process_tags(item_path)

get_url_config = {
    "_valid_url" : valid_url,
    "get-urls" : True,
    "no-skip" : no_skip,
    "range" : range if range else None,
    "user-agent" : user_agent
}

scrape_config = {
    "_valid_url" : valid_url,
    "directory" : image_dir,
    "no-skip" : no_skip,
    "write-metadata": write_metadata,
    "write-tags" : write_tags,
    "range" : range if range else None,
    "user-agent" : user_agent
}

get_url_args = scrape(get_url_config)
scrape_args = scrape(scrape_config)
scraper_text = os.path.join(root_dir, "scrape_this.txt")

if write_tags:
    subprocess.run(f"gallery-dl {scrape_args} {additional_arguments}", shell=True)
    pre_process_tags(scraped_data_dir)
else:
    cap = subprocess.run(f"gallery-dl {get_url_args} {additional_arguments}", shell=True, capture_output=True, text=True)
    with open(scraper_text, "w") as f:
        f.write(cap.stdout)

    os.chdir(image_dir)
    subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -i {scraper_text}", shell=True)
