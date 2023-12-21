## Scrape Dataset
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default=None, help="Scrape images that satisfy this condition")
parser.add_argument("--post", type=str, default=None, help="Scrape image post id")
parser.add_argument("--range", type=str, default=None, help="Index range specifying which files to download.")
parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
parser.add_argument("--overlap", type=int, default=0, help="Overlapping index range")
parser.add_argument("--write_metadata", action="store_false", help="Write metadata to separate JSON files")
parser.add_argument("--write-tags", action="store_false", help="Write image tags to separate text files")
parser.add_argument("--no_skip", action="store_true", help="Do not skip downloads; overwrite existing files")
args = parser.parse_args()

if args.range:
    _range = args.range.split("-")
    start, end = int(_range[0]), int(_range[1])
    chunk_size = (end - start + 1) // args.workers
    remainder = (end - start + 1) % args.workers
    id_chunks = [[i, i + chunk_size - 1] for i in range(start, end - remainder, chunk_size)]
    id_chunks[-1][-1] += remainder
    for i in id_chunks:
        if i[0] - args.overlap < 1:
            i[0] = 1
        else:
            i[0] -= args.overlap
    print(id_chunks)

root_dir = "~/sd-train"
scraped_data_dir = os.path.join(root_dir, "scraped_data")

os.chdir(root_dir)
# Use `gallery-dl` to scrape images from an imageboard site. To specify `prompt(s)`, separate them with commas (e.g., `hito_komoru, touhou`).
booru = "Danbooru"  # ["Danbooru", "Gelbooru", "Safebooru"]

# Alternatively, you can provide a `custom_url` instead of using a predefined site.
custom_url = ""

# Use the `sub_folder` option to organize the downloaded images into separate folders based on their concept or category.
sub_folder = ""

user_agent = "gdl/1.24.5"

additional_arguments = "--filename /O --no-part"

if args.prompt:
    tags = args.prompt.split(',')
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
    if args.prompt:
        url = "https://danbooru.donmai.us/posts?tags={}".format(tags)
    elif args.post:
        # To download a single post
        url = "https://danbooru.donmai.us/posts/{}".format(args.post)
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

get_url_config = {
    "_valid_url" : valid_url,
    "get-urls" : True,
    "no-skip" : args.no_skip,
    "range" : args.range if args.range else None,
    "user-agent" : user_agent
}

scrape_config = {
    "_valid_url" : valid_url,
    "directory" : image_dir,
    "no-skip" : args.no_skip,
    "write-metadata": args.write_metadata,
    "write-tags" : args.write_tags,
    "user-agent" : user_agent
}

get_url_args = scrape(get_url_config)
scrape_args = scrape(scrape_config)
scraper_text = os.path.join(root_dir, "scrape_this.txt")

if args.write_tags:
    if not args.range:
        subprocess.run(f"gallery-dl {scrape_args} {additional_arguments}", shell=True)
    else:
        for i in id_chunks:
            scrape_config["range"] = f"{i[0]}-{i[1]}"
            scrape_args = scrape(scrape_config)
            subprocess.Popen(f"nohup gallery-dl {scrape_args} {additional_arguments} &", shell=True)
else:
    cap = subprocess.run(f"gallery-dl {get_url_args} {additional_arguments}", shell=True, capture_output=True, text=True)
    with open(scraper_text, "w") as f:
        f.write(cap.stdout)

    os.chdir(image_dir)
    subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -i {scraper_text}", shell=True)
